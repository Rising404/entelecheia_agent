"""信任的 Attempt 输入契约和确定性的提示投影。

此模块拥有提供给 Attempt 决策的不可变数据，包括有界的提示/历史投影，以及其以失败关闭的输入防护。它不会调用模型、访问持久化存储、分发工具或选择系统提示。
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    model_validator,
)

from ...task_graph.contracts import InSessionTaskAcceptanceProposal
from ....tools.contracts import ToolSpec
from ...work_run import (
    AcceptanceProgressSnapshot,
    AcceptanceVerificationFeedback,
    AuxiliaryNodeSubject,
    DownstreamVerificationFeedback,
    ExecutionSubject,
    OutputWindow,
    TaskNodeSubject,
    ToolResult,
)
from ..paper_prompt_context import PaperAttemptContext
from ....persistent_turn_content.findings import (
    ExecutionFindingsActiveProjection,
    ExecutionFindingsOwnerKind,
    derive_execution_findings_ledger_id,
    sha256_json,
)
from ....persistent_turn_content.findings import EXECUTION_FINDINGS_TOOL_IDS
from ....tools.findings.projection import (
    project_execution_findings_tool_output_for_model,
)
from ..task_node.dependency_delivery_contracts import (
    TaskNodeDependencyDeliveries,
    build_task_node_dependency_model_payload,
    serialize_task_node_dependency_model_payload,
)
from ..task_node.input_limits import AttemptDecisionInputLimits
from ..task_node.source_context import TaskNodeSourceContext


class _AttemptInputContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)


class AttemptVerificationFeedback(_AttemptInputContract):
    """来自最近语义验证的完整类型结果。"""

    submitted_output_revision: int = Field(ge=1)
    acceptance_results: tuple[AcceptanceVerificationFeedback, ...] = Field(
        min_length=1
    )
    downstream_results: tuple[DownstreamVerificationFeedback, ...] = Field(
        default=(),
        max_length=8,
    )

    @model_validator(mode="after")
    def _require_unique_acceptance_ids(self) -> 'AttemptVerificationFeedback':
        ids = [item.acceptance_id for item in self.acceptance_results]
        if len(ids) != len(set(ids)):
            raise ValueError("verification feedback Acceptance IDs must be unique")
        gate_ids = [item.gate_id for item in self.downstream_results]
        if len(gate_ids) != len(set(gate_ids)):
            raise ValueError("downstream verification gate IDs must be unique")
        return self


class AttemptUserInput(_AttemptInputContract):
    """赋予此语义 Attempt 意义的不可变用户输入。

    ``prior_waiting_user_question`` 仅从创建此 Attempt 时消耗的精确闭合 ``request_user_input`` Attempt 中投影而来。它不声明 ``content`` 是一个充分的答案。
    """

    content: str = Field(min_length=1)
    prior_waiting_user_question: str | None = Field(default=None, min_length=1)


class PriorToolResultProjection(_AttemptInputContract):
    """一个先前的结果加上产生它的 Host 绑定的工具身份。"""

    tool_id: str = Field(min_length=1)
    tool_version: str = Field(min_length=1)
    result: ToolResult


class PriorToolResultsProjection(_AttemptInputContract):
    """一个 Host 绑定的 WorkRun 历史投影；此类别不选择限制。"""

    items: tuple[PriorToolResultProjection, ...] = ()
    truncated: bool = False

    @model_validator(mode="after")
    def _require_unique_results(self) -> 'PriorToolResultsProjection':
        result_ids = [item.result.tool_result_id for item in self.items]
        call_ids = [item.result.tool_call_id for item in self.items]
        if len(result_ids) != len(set(result_ids)):
            raise ValueError("prior ToolResult IDs must be unique")
        if len(call_ids) != len(set(call_ids)):
            raise ValueError("prior ToolCall IDs must be unique")
        return self


class AttemptDecisionInputTooLarge(RuntimeError):
    """精确完整的 Attempt 提示无法适应其 Host 模型。"""

    code = "attempt_model_input_too_large"

    def __init__(
        self,
        *,
        serialized_utf8_bytes: int,
        limits: AttemptDecisionInputLimits,
    ) -> None:
        self.serialized_utf8_bytes = serialized_utf8_bytes
        self.limits = limits
        super().__init__(
            "Attempt model input exceeds the configured profile: "
            f"profile_id={limits.profile_id!r}, serialized_utf8_bytes="
            f"{serialized_utf8_bytes}/{limits.max_serialized_utf8_bytes}"
        )


class PriorToolResultsInputTooLarge(AttemptDecisionInputTooLarge):
    """强制的前置 ToolResult 投影无法匹配其子限制。"""

    code = "attempt_prior_tool_results_input_too_large"

    def __init__(
        self,
        *,
        entry_count: int,
        serialized_utf8_bytes: int,
        limits: AttemptDecisionInputLimits,
    ) -> None:
        self.entry_count = entry_count
        self.serialized_utf8_bytes = serialized_utf8_bytes
        self.limits = limits
        RuntimeError.__init__(
            self,
            "prior ToolResults exceed the configured Attempt input profile: "
            f"profile_id={limits.profile_id!r}, entries="
            f"{entry_count}/{limits.max_prior_tool_result_items}, "
            "serialized_utf8_bytes="
            f"{serialized_utf8_bytes}/"
            f"{limits.max_prior_tool_results_serialized_utf8_bytes}",
        )


class AttemptDecisionInputUnsupported(RuntimeError):
    """受信任的 Attempt 投影无法编码为支持的 JSON。"""

    code = "attempt_model_input_unsupported"

    def __init__(self, *, profile_id: str) -> None:
        self.profile_id = profile_id
        super().__init__(
            "Attempt model input is not compact UTF-8 JSON supported by profile "
            f"{profile_id!r}"
        )


class RequiredPriorToolResultsUnavailable(RuntimeError):
    """AcceptanceProgress 引用了未经授权历史中的结果。"""

    code = "required_prior_tool_results_unavailable"

    def __init__(self, result_ids: Iterable[str]) -> None:
        self.result_ids = tuple(sorted(set(result_ids)))
        super().__init__(
            "AcceptanceProgress references unavailable prior ToolResults: "
            + ",".join(self.result_ids)
        )


class AttemptDecisionContext(_AttemptInputContract):
    """可信且不可变的输入，用于一个已创建的 Attempt 决策。

    身份和修订字段是 Host 的权威状态。它们故意 未包含在 ``AttemptDecision`` 中，因此提供者输出不能将 决策重新定向到另一个 WorkRun 或节点。
    """

    session_id: str = Field(min_length=1)
    turn_id: str = Field(min_length=1)
    work_run_id: str = Field(min_length=1)
    work_run_revision: int = Field(ge=1)
    attempt_id: str = Field(min_length=1)
    attempt_ordinal: int = Field(ge=1)
    user_input: AttemptUserInput
    subject: ExecutionSubject
    node_title: str = Field(min_length=1)
    node_objective: str = Field(min_length=1)
    acceptances: tuple[InSessionTaskAcceptanceProposal, ...] = Field(min_length=1)
    acceptance_progress: AcceptanceProgressSnapshot
    output_window: OutputWindow
    dependency_deliveries: TaskNodeDependencyDeliveries
    source_context: TaskNodeSourceContext | None = None
    prior_tool_results: PriorToolResultsProjection = PriorToolResultsProjection()
    execution_findings: ExecutionFindingsActiveProjection | None = None
    input_limits: AttemptDecisionInputLimits
    allowed_tools: tuple[ToolSpec, ...] = ()
    allow_user_input: bool = True
    verification_feedback: AttemptVerificationFeedback | None = None
    paper_resources: PaperAttemptContext | None = None

    @field_serializer("allowed_tools", when_used="json")
    def _serialize_allowed_tools(
        self,
        value: tuple[ToolSpec, ...],
    ) -> list[dict[str, Any]]:
        """在 Pydantic JSON 边界处解冻不可变的 ToolSpec 架构。"""

        return [tool.to_dict() for tool in value]

    @model_validator(mode="after")
    def _validate_trusted_bindings(self) -> 'AttemptDecisionContext':
        if (
            self.attempt_ordinal == 1
            and self.user_input.prior_waiting_user_question is not None
        ):
            raise ValueError("a first Attempt cannot consume a prior user question")
        if self.acceptance_progress.work_run_id != self.work_run_id:
            raise ValueError("AcceptanceProgress does not belong to this WorkRun")
        if self.output_window.work_run_id != self.work_run_id:
            raise ValueError("OutputWindow does not belong to this WorkRun")
        if self.acceptance_progress.subject != self.subject:
            raise ValueError("AcceptanceProgress subject does not match the current node")
        if (
            self.acceptance_progress.evaluated_output_revision
            != self.output_window.output_revision
        ):
            raise ValueError("AcceptanceProgress does not describe the current OutputWindow")
        if self.paper_resources is not None:
            if not isinstance(self.subject, TaskNodeSubject):
                raise ValueError(
                    "AuxiliaryNode Attempt cannot receive paper resources"
                )
            if self.paper_resources.session_id != self.session_id:
                raise ValueError("paper resources belong to another Session")
            if self.paper_resources.task_id != self.subject.task_id:
                raise ValueError("paper resources belong to another Task")
        if isinstance(self.subject, TaskNodeSubject):
            if self.source_context is None:
                raise ValueError("TaskNode Attempt requires frozen source context")
            self.source_context.require_exact_binding(
                session_id=self.session_id,
                subject=self.subject,
                acceptances=self.acceptances,
            )
        elif self.source_context is not None:
            raise ValueError(
                "AuxiliaryNode Attempt cannot receive generic source context"
            )

        acceptance_ids = [item.acceptance_id for item in self.acceptances]
        if len(acceptance_ids) != len(set(acceptance_ids)):
            raise ValueError("node Acceptance IDs must be unique")
        progress_ids = [item.acceptance_id for item in self.acceptance_progress.items]
        if progress_ids != acceptance_ids:
            raise ValueError("AcceptanceProgress must cover the full node Acceptance list")

        if any(
            item.result.attempt_id == self.attempt_id
            for item in self.prior_tool_results.items
        ):
            raise ValueError("current-Attempt ToolResults cannot be prior history")

        dependency_subjects = tuple(
            item.child_subject for item in self.dependency_deliveries.items
        )
        if not isinstance(self.subject, TaskNodeSubject):
            if dependency_subjects:
                raise ValueError(
                    "AuxiliaryNode Attempt cannot receive TaskNode dependencies"
                )
        elif any(
            subject.task_id != self.subject.task_id
            or subject.graph_revision != self.subject.graph_revision
            or subject.node_id == self.subject.node_id
            for subject in dependency_subjects
        ):
            raise ValueError(
                "Attempt dependency Deliveries must be current child subjects"
            )
        dependency_order = tuple(
            (item.child_ordinal, item.child_subject.node_id)
            for item in self.dependency_deliveries.items
        )
        if dependency_order != tuple(sorted(dependency_order)):
            raise ValueError("Attempt dependency Deliveries must use stable child order")

        tool_ids = [tool.tool_id for tool in self.allowed_tools]
        if len(tool_ids) != len(set(tool_ids)):
            raise ValueError("allowed tool IDs must be unique")
        if any(not isinstance(tool, ToolSpec) for tool in self.allowed_tools):
            raise ValueError("allowed_tools must contain ToolSpec values")
        exposed_findings_tools = EXECUTION_FINDINGS_TOOL_IDS.intersection(tool_ids)
        if exposed_findings_tools != EXECUTION_FINDINGS_TOOL_IDS:
            if exposed_findings_tools:
                raise ValueError(
                    "execution findings tools must be exposed as one complete pair"
                )
        elif self.execution_findings is None:
            raise ValueError(
                "execution findings tools require the current active projection"
            )
        if self.execution_findings is not None:
            expected_ledger_id = derive_execution_findings_ledger_id(
                owner_kind=ExecutionFindingsOwnerKind.WORK_RUN,
                execution_owner_id=self.work_run_id,
            )
            if self.execution_findings.ledger_id != expected_ledger_id:
                raise ValueError(
                    "execution findings projection belongs to another WorkRun"
                )

        if self.verification_feedback is not None:
            feedback_ids = [
                item.acceptance_id
                for item in self.verification_feedback.acceptance_results
            ]
            if feedback_ids != acceptance_ids:
                raise ValueError(
                    "verification feedback must cover the full node Acceptance list"
                )
            if (
                self.verification_feedback.submitted_output_revision
                != self.output_window.output_revision
            ):
                raise ValueError(
                    "verification feedback does not describe the current OutputWindow"
                )
        return self


def build_attempt_prompt_payload(context: AttemptDecisionContext) -> dict[str, Any]:
    """仅投影授权于此 Attempt 模型调用的事实。"""

    _require_exact_source_context(context)
    node: dict[str, Any] = {
        "subject": context.subject.model_dump(mode="json"),
        "title": context.node_title,
        "objective": context.node_objective,
        "acceptances": [
            item.model_dump(mode="json") for item in context.acceptances
        ],
    }
    # 保持标记主题转换期间的现有 TaskNode 提示字段。
    # AuxiliaryNode 的身份仅由 ``subject`` 承载，因此
    # 在模型边界处没有虚假的 TaskGraph 修订。
    if isinstance(context.subject, TaskNodeSubject):
        node.update(
            {
                "task_id": context.subject.task_id,
                "graph_revision": context.subject.graph_revision,
                "node_id": context.subject.node_id,
                "node_revision": context.subject.node_revision,
            }
        )
    payload = {
        "bindings": {
            "session_id": context.session_id,
            "turn_id": context.turn_id,
            "work_run_id": context.work_run_id,
            "work_run_revision": context.work_run_revision,
            "attempt_id": context.attempt_id,
            "attempt_ordinal": context.attempt_ordinal,
        },
        "user_input": context.user_input.model_dump(mode="json"),
        "node": node,
        "acceptance_progress": context.acceptance_progress.model_dump(mode="json"),
        "output_window": context.output_window.model_dump(mode="json"),
        "prior_tool_results": build_prior_tool_results_prompt_payload(
            context.prior_tool_results
        ),
        "execution_findings": (
            context.execution_findings.model_dump(mode="json")
            if context.execution_findings is not None
            else None
        ),
        "dependency_deliveries": build_task_node_dependency_model_payload(
            context.dependency_deliveries
        )["dependency_deliveries"],
        "source_context": (
            context.source_context.model_dump(mode="json")
            if context.source_context is not None
            else None
        ),
        "allowed_tools": [tool.to_dict() for tool in context.allowed_tools],
        "verification_feedback": (
            context.verification_feedback.model_dump(mode="json")
            if context.verification_feedback is not None
            else None
        ),
    }
    if context.paper_resources is not None:
        payload["paper_resources"] = context.paper_resources.to_dict()
    if isinstance(context.subject, AuxiliaryNodeSubject):
        payload["task_graph_proposal_contract"] = {
            "schema_version": "insession-task-graph-revision-v2",
            "action_kind": "submit_task_graph",
            "proposal_encoding": "direct_json_object",
            "host_materialization": "canonical_plain_text_output_window",
            "shape": {
                "root": {
                    "root_key": "ascii_local_key",
                    "nodes": [
                        {
                            "node_key": "ascii_local_key",
                            "node_kind": "root|subtask",
                            "parent_node_key": "null_for_root|parent_local_key",
                            "title": "nonempty_string",
                            "objective": "nonempty_deliverable_objective",
                            "source_anchor_ids": ["task_creation_source"],
                            "acceptance_criteria": [
                                {
                                    "acceptance_id": "ascii_local_key",
                                    "criterion": "observable_completion_condition",
                                    "source_anchor_ids": ["task_creation_source"],
                                }
                            ],
                            "constraints": [],
                        }
                    ],
                }
            },
            "allowed_source_anchor_ids": ["task_creation_source"],
            "required_source_anchor_ids": ["task_creation_source"],
            "limits": {
                "max_nodes_per_task": 64,
                "max_depth": 12,
                "max_acceptances_per_node": 64,
            },
            "complex_task_quality_target": {
                "adaptive_not_a_quota": True,
                "recommended_acceptances_per_node": [1, 2],
                "recommended_title_chars": 40,
                "recommended_objective_chars": 160,
                "recommended_acceptance_criterion_chars": 120,
            },
        }
    return payload


def _require_exact_source_context(context: AttemptDecisionContext) -> None:
    if isinstance(context.subject, TaskNodeSubject):
        if context.source_context is None:
            raise ValueError("TaskNode Attempt requires frozen source context")
        context.source_context.require_exact_binding(
            session_id=context.session_id,
            subject=context.subject,
            acceptances=context.acceptances,
        )
    elif context.source_context is not None:
        raise ValueError("AuxiliaryNode Attempt cannot receive generic source context")


def build_prior_tool_results_prompt_payload(
    projection: PriorToolResultsProjection,
) -> dict[str, Any]:
    """返回注入为 ``prior_tool_results`` 的精确 JSON 值。"""

    return {
        "items": [_prior_tool_result_prompt_item(item) for item in projection.items],
        "truncated": projection.truncated,
    }


def _prior_tool_result_prompt_item(
    item: PriorToolResultProjection,
) -> dict[str, Any]:
    result = item.result.model_dump(mode="json")
    if item.tool_id in EXECUTION_FINDINGS_TOOL_IDS:
        result["output"] = project_execution_findings_tool_output_for_model(
            result.get("output")
        )
    return {
        "tool_id": item.tool_id,
        "tool_version": item.tool_version,
        # Preserve the identity of the exact durable result, not the slimmer
        # model projection below.
        "result_sha256": sha256_json(item.result),
        **result,
    }


def serialize_attempt_prompt_payload(context: AttemptDecisionContext) -> str:
    """序列化一次，绑定精确，返回发送给提供者的字符串。"""

    serialized, serialized_utf8_bytes = _compact_json_utf8(
        build_attempt_prompt_payload(context),
        profile_id=context.input_limits.profile_id,
    )
    if serialized_utf8_bytes > context.input_limits.max_serialized_utf8_bytes:
        raise AttemptDecisionInputTooLarge(
            serialized_utf8_bytes=serialized_utf8_bytes,
            limits=context.input_limits,
        )
    return serialized


def attempt_prompt_serialized_utf8_bytes(
    context: AttemptDecisionContext,
) -> int:
    """测量模型调用使用的精确完整紧凑的 JSON 字节。"""

    _serialized, serialized_utf8_bytes = _compact_json_utf8(
        build_attempt_prompt_payload(context),
        profile_id=context.input_limits.profile_id,
    )
    return serialized_utf8_bytes


def prior_tool_results_serialized_utf8_bytes(
    projection: PriorToolResultsProjection,
    *,
    profile_id: str = "prior-tool-results-measurement",
) -> int:
    """测量模型请求的确切紧凑的 JSON 字节使用量。"""

    payload = build_prior_tool_results_prompt_payload(projection)
    _serialized, serialized_utf8_bytes = _compact_json_utf8(
        payload,
        profile_id=profile_id,
    )
    return serialized_utf8_bytes


def require_prior_tool_results_within_limits(
    projection: PriorToolResultsProjection,
    *,
    limits: AttemptDecisionInputLimits,
) -> None:
    """在模型调用前失败，如果确切的投影超过任一上限。"""

    entry_count = len(projection.items)
    byte_count = prior_tool_results_serialized_utf8_bytes(
        projection,
        profile_id=limits.profile_id,
    )
    if (
        entry_count > limits.max_prior_tool_result_items
        or byte_count > limits.max_prior_tool_results_serialized_utf8_bytes
    ):
        raise PriorToolResultsInputTooLarge(
            entry_count=entry_count,
            serialized_utf8_bytes=byte_count,
            limits=limits,
        )


def require_dependency_deliveries_within_limits(
    context: AttemptDecisionContext,
) -> None:
    """如果确切的子体超出其上限，则在提供者调用前失败。"""

    serialize_task_node_dependency_model_payload(
        context.dependency_deliveries,
        limits=context.input_limits.dependency_delivery_limits,
    )


def select_bounded_prior_tool_results(
    items: tuple[PriorToolResultProjection, ...],
    *,
    required_result_ids: Iterable[str],
    limits: AttemptDecisionInputLimits,
) -> PriorToolResultsProjection:
    """选择一个确定性的有限历史记录，不隐藏 Acceptance 支持。

    ``items`` 已按持久化时间顺序排列。每个当前由 AcceptanceProgress 引用的结果都是强制性的。剩余条目按最新优先考虑，而返回的投影则恢复到持久化顺序。不符合要求的可选条目将被跳过；缺失或超出限制的强制性集合将以失败关闭而不是被截断。
    """

    full = PriorToolResultsProjection(items=items, truncated=False)
    required = frozenset(required_result_ids)
    positions_by_id = {
        item.result.tool_result_id: position for position, item in enumerate(items)
    }
    missing = required.difference(positions_by_id)
    if missing:
        raise RequiredPriorToolResultsUnavailable(missing)

    try:
        require_prior_tool_results_within_limits(full, limits=limits)
    except AttemptDecisionInputTooLarge:
        pass
    else:
        return full

    selected_positions = {positions_by_id[result_id] for result_id in required}

    def projection_for(positions: set[int]) -> PriorToolResultsProjection:
        selected = tuple(items[position] for position in sorted(positions))
        return PriorToolResultsProjection(
            items=selected,
            truncated=len(selected) < len(items),
        )

    mandatory = projection_for(selected_positions)
    require_prior_tool_results_within_limits(mandatory, limits=limits)

    for position in range(len(items) - 1, -1, -1):
        if position in selected_positions:
            continue
        candidate_positions = {*selected_positions, position}
        candidate = projection_for(candidate_positions)
        try:
            require_prior_tool_results_within_limits(candidate, limits=limits)
        except AttemptDecisionInputTooLarge:
            continue
        selected_positions = candidate_positions

    bounded = projection_for(selected_positions)
    require_prior_tool_results_within_limits(bounded, limits=limits)
    return bounded


def mandatory_prior_tool_result_ids(
    items: tuple[PriorToolResultProjection, ...],
    *,
    supporting_result_ids: Iterable[str],
) -> frozenset[str]:
    """返回 Acceptance 支持的结果，这些结果后续的 Attempt 无法隐藏。"""

    return frozenset(supporting_result_ids)


def _compact_json_utf8(
    value: object,
    *,
    profile_id: str,
) -> tuple[str, int]:
    """返回支持的紧凑格式 JSON 及其精确的 UTF-8 大小。

    ``ensure_ascii=False`` 保留实际提供者负载。拒绝非有限或非 JSON 值、循环结构和孤立的代理码点，给调用者提供一个类型安全的以失败关闭路径，在任何模型事件或提供者调用之前。
    """

    try:
        serialized = json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )
        serialized_utf8_bytes = len(serialized.encode("utf-8"))
    except (TypeError, ValueError, UnicodeEncodeError, RecursionError) as exc:
        raise AttemptDecisionInputUnsupported(profile_id=profile_id) from exc
    return serialized, serialized_utf8_bytes


__all__ = [
    'AttemptDecisionContext',
    'AttemptDecisionInputLimits',
    "AttemptDecisionInputTooLarge",
    "AttemptDecisionInputUnsupported",
    'AttemptUserInput',
    'AttemptVerificationFeedback',
    'PriorToolResultProjection',
    "PriorToolResultsInputTooLarge",
    'PriorToolResultsProjection',
    "RequiredPriorToolResultsUnavailable",
    "attempt_prompt_serialized_utf8_bytes",
    "build_attempt_prompt_payload",
    "build_prior_tool_results_prompt_payload",
    "mandatory_prior_tool_result_ids",
    "prior_tool_results_serialized_utf8_bytes",
    "require_dependency_deliveries_within_limits",
    "require_prior_tool_results_within_limits",
    "select_bounded_prior_tool_results",
    "serialize_attempt_prompt_payload",
]
