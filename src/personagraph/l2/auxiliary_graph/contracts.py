"""任务所有 AuxiliaryGraph 规划的纯契约。

本模块仅是领域边界：不执行持久化、供应商调用或 Runtime 选择。
"""

from __future__ import annotations

import hashlib
import json
import math
from enum import StrEnum
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..task_graph.contracts import (
    InSessionTaskAcceptanceProposal,
    InSessionTaskGraphRevisionProposal,
    InSessionTaskNodeKind,
    InSessionTaskStatus,
)
from personagraph.l2.work_run import AuxiliaryNodeSubject


_DURABLE_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,199}$"
_LOCAL_KEY_PATTERN = r"^[a-z][a-z0-9_]{0,63}$"
_SHA256_PATTERN = r"^[0-9a-f]{64}$"
_OUTPUT_CONTRACT_PATTERN = r"^[a-z][a-z0-9._:-]{0,127}$"
_MAX_AUXILIARY_NODES = 64
_MAX_AUXILIARY_DEPTH = 12
_FORBIDDEN_AUXILIARY_OUTPUT_CONTRACTS = frozenset(
    {
        "call_tools",
        "submit_output_window",
        "submit_task_graph",
        "write_output_window",
    }
)
TASK_GRAPH_SEMANTIC_VERIFICATION_MAX_PROMPT_JSON_UTF8_BYTES = 1_500_000
TASK_GRAPH_SEMANTIC_VERIFICATION_MAX_RESULT_JSON_UTF8_BYTES = 131_072


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def planning_document_group_alias(*, session_id: str, document_id: str) -> str:
    """返回一个跨规划 reader 的文档 prompt 安全身份。"""

    session_id = _require_canonical_text(session_id, field_name="session_id")
    document_id = _require_canonical_text(document_id, field_name="document_id")
    digest = _canonical_sha256(
        {
            "schema_version": "planning-document-group-alias-v1",
            "session_id": session_id,
            "document_id": document_id,
        }
    )
    return f"document_{digest[:32]}"


def _require_canonical_text(value: str, *, field_name: str) -> str:
    if value != value.strip() or not value:
        raise ValueError(f"{field_name} must be non-empty canonical text")
    if "\x00" in value:
        raise ValueError(f"{field_name} cannot contain NUL")
    return value


def _require_unique(values: tuple[str, ...], *, field_name: str) -> tuple[str, ...]:
    if len(values) != len(set(values)):
        raise ValueError(f"{field_name} must be unique")
    return values


def _require_canonical_unique(
    values: tuple[str, ...], *, field_name: str
) -> tuple[str, ...]:
    if any(
        not value
        or value != value.strip()
        or len(value) > 500
        or "\x00" in value
        for value in values
    ):
        raise ValueError(f"{field_name} contain invalid canonical text")
    _require_unique(values, field_name=field_name)
    if values != tuple(sorted(values)):
        raise ValueError(f"{field_name} must use ascending canonical order")
    return values


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


# ---------------------------------------------------------------------------
# AuxiliaryGraph 目标与不可变 DAG 契约
# ---------------------------------------------------------------------------


class AuxiliaryPlanningGoalStatus(StrEnum):
    ACTIVE = "active"
    WAITING_USER = "waiting_user"
    WAITING_AUTHORIZATION = "waiting_authorization"
    WAITING_EXTERNAL = "waiting_external"
    INTERRUPTED = "interrupted"
    PROPOSAL_READY = "proposal_ready"
    GAPPED_READY = "gapped_ready"
    COMMITTED = "committed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    SUPERSEDED = "superseded"
    BUDGET_EXHAUSTED = "budget_exhausted"


class AuxiliaryPlanningGoal(_Contract):
    """基于冻结 TaskGraph base 的一个预算稳定规划 episode。"""

    schema_version: Literal["auxiliary-planning-goal-v1"] = (
        "auxiliary-planning-goal-v1"
    )
    session_id: str = Field(pattern=_DURABLE_ID_PATTERN)
    task_id: str = Field(pattern=_DURABLE_ID_PATTERN)
    auxiliary_graph_id: str = Field(pattern=_DURABLE_ID_PATTERN)
    goal_id: str = Field(pattern=_DURABLE_ID_PATTERN)
    base_task_graph_revision: int | None = Field(default=None, ge=1)
    target_task_graph_revision: int = Field(ge=1)
    creation_turn_id: str = Field(pattern=_DURABLE_ID_PATTERN)
    authorization_manifest_id: str = Field(pattern=_DURABLE_ID_PATTERN)
    budget_ledger_id: str = Field(pattern=_DURABLE_ID_PATTERN)
    status: AuxiliaryPlanningGoalStatus = AuxiliaryPlanningGoalStatus.ACTIVE
    state_version: int = Field(default=1, ge=1)

    @model_validator(mode="after")
    def _validate_target_revision(self) -> "AuxiliaryPlanningGoal":
        expected = (
            1
            if self.base_task_graph_revision is None
            else self.base_task_graph_revision + 1
        )
        if self.target_task_graph_revision != expected:
            raise ValueError(
                "target TaskGraph revision must be one above the frozen base"
            )
        return self


class AuxiliaryGraphAggregate(_Contract):
    """一个任务所有 AuxiliaryGraph 的当前指针投影。"""

    schema_version: Literal["auxiliary-graph-aggregate-v2"] = (
        "auxiliary-graph-aggregate-v2"
    )
    session_id: str = Field(pattern=_DURABLE_ID_PATTERN)
    task_id: str = Field(pattern=_DURABLE_ID_PATTERN)
    auxiliary_graph_id: str = Field(pattern=_DURABLE_ID_PATTERN)
    current_goal_id: str | None = Field(default=None, pattern=_DURABLE_ID_PATTERN)
    current_auxiliary_graph_revision: int | None = Field(default=None, ge=1)
    state_version: int = Field(ge=1)

    @model_validator(mode="after")
    def _validate_current_pointer(self) -> "AuxiliaryGraphAggregate":
        if (self.current_goal_id is None) != (
            self.current_auxiliary_graph_revision is None
        ):
            raise ValueError("current goal and revision pointers must be set together")
        return self


class AuxiliaryNodeKind(StrEnum):
    OBSERVE = "observe"
    ANALYZE = "analyze"
    CLARIFY = "clarify"
    VALIDATE = "validate"
    SYNTHESIZE = "synthesize"


class AuxiliaryNodeExecutorKind(StrEnum):
    HOST_PRIMITIVE = "host_primitive"
    MODEL_WORK_RUN = "model_work_run"
    USER_GATE = "user_gate"
    TERMINAL_PLANNER = "terminal_planner"


class AuxiliaryGraphRevisionReason(StrEnum):
    INITIAL = "initial"
    RESOURCE_CHANGED = "resource_changed"
    EVIDENCE_CHANGED = "evidence_changed"
    USER_RESPONSE = "user_response"
    NODE_FAILED = "node_failed"
    VERIFICATION_FAILED = "verification_failed"
    EXTERNAL_RESUMED = "external_resumed"
    AUTHORITY_CHANGED = "authority_changed"
    MANUAL_REPLAN = "manual_replan"


class AuxiliaryReplanTriggerReason(StrEnum):
    """从语义结算带入重新规划、由 Host 派生的原因。"""

    SEMANTIC_REVISION_REQUIRED = "semantic_revision_required"
    SEMANTIC_EVIDENCE_BLOCKED = "semantic_evidence_blocked"


class AuxiliaryReplanTriggerReceipt(_Contract):
    """证明为何可以替换一个图 revision 的不可变 authority。

    trigger 刻意存储哈希和类型化 disposition，而非模型编写的正文。Runtime 需要 reviewer
    finding 时可加载被引用的语义结算，同时此 receipt 保持为紧凑的 CAS 与重放边界。
    """

    schema_version: Literal["auxiliary-replan-trigger-receipt-v1"] = (
        "auxiliary-replan-trigger-receipt-v1"
    )
    trigger_id: str = Field(pattern=_DURABLE_ID_PATTERN)
    create_apply_id: str = Field(pattern=_DURABLE_ID_PATTERN)
    session_id: str = Field(pattern=_DURABLE_ID_PATTERN)
    task_id: str = Field(pattern=_DURABLE_ID_PATTERN)
    auxiliary_graph_id: str = Field(pattern=_DURABLE_ID_PATTERN)
    goal_id: str = Field(pattern=_DURABLE_ID_PATTERN)
    source_auxiliary_graph_revision: int = Field(ge=1)
    source_structure_sha256: str = Field(pattern=_SHA256_PATTERN)
    source_authority_snapshot_id: str = Field(pattern=_DURABLE_ID_PATTERN)
    source_authority_snapshot_sha256: str = Field(pattern=_SHA256_PATTERN)
    semantic_authority_projection_sha256: str = Field(pattern=_SHA256_PATTERN)
    semantic_prompt_payload_sha256: str = Field(pattern=_SHA256_PATTERN)
    task_graph_proposal_sha256: str = Field(pattern=_SHA256_PATTERN)
    semantic_settlement_id: str = Field(pattern=_DURABLE_ID_PATTERN)
    semantic_settlement_sha256: str = Field(pattern=_SHA256_PATTERN)
    semantic_result_sha256s: tuple[str, ...] = Field(min_length=1, max_length=2)
    budget_ledger_id: str = Field(pattern=_DURABLE_ID_PATTERN)
    budget_state_version: int = Field(ge=1)
    budget_snapshot_sha256: str = Field(pattern=_SHA256_PATTERN)
    evidence_epoch_sha256: str = Field(pattern=_SHA256_PATTERN)
    semantic_disposition: TaskGraphSemanticVerificationDisposition
    trigger_reason: AuxiliaryReplanTriggerReason
    revision_reason: Literal[AuxiliaryGraphRevisionReason.VERIFICATION_FAILED] = (
        AuxiliaryGraphRevisionReason.VERIFICATION_FAILED
    )
    created_turn_id: str = Field(pattern=_DURABLE_ID_PATTERN)
    receipt_sha256: str = Field(pattern=_SHA256_PATTERN)

    @field_validator("semantic_result_sha256s")
    @classmethod
    def _validate_result_hashes(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if len(values) != len(set(values)) or any(
            len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in values
        ):
            raise ValueError("semantic result hashes must be unique lowercase sha256s")
        return values

    @model_validator(mode="after")
    def _validate_trigger(self) -> "AuxiliaryReplanTriggerReceipt":
        if self.semantic_disposition is TaskGraphSemanticVerificationDisposition.PASS:
            raise ValueError("a passing semantic settlement cannot trigger replanning")
        expected_reason = (
            AuxiliaryReplanTriggerReason.SEMANTIC_EVIDENCE_BLOCKED
            if self.semantic_disposition
            is TaskGraphSemanticVerificationDisposition.BLOCKED
            else AuxiliaryReplanTriggerReason.SEMANTIC_REVISION_REQUIRED
        )
        if self.trigger_reason is not expected_reason:
            raise ValueError("replan trigger reason must be derived from disposition")
        expected_epoch = canonical_auxiliary_replan_evidence_epoch_sha256(
            session_id=self.session_id,
            task_id=self.task_id,
            auxiliary_graph_id=self.auxiliary_graph_id,
            goal_id=self.goal_id,
            source_auxiliary_graph_revision=self.source_auxiliary_graph_revision,
            source_structure_sha256=self.source_structure_sha256,
            source_authority_snapshot_id=self.source_authority_snapshot_id,
            source_authority_snapshot_sha256=(
                self.source_authority_snapshot_sha256
            ),
            semantic_authority_projection_sha256=(
                self.semantic_authority_projection_sha256
            ),
            semantic_prompt_payload_sha256=self.semantic_prompt_payload_sha256,
            task_graph_proposal_sha256=self.task_graph_proposal_sha256,
        )
        if self.evidence_epoch_sha256 != expected_epoch:
            raise ValueError("replan trigger evidence epoch does not match authority")
        expected_receipt = canonical_auxiliary_replan_trigger_receipt_sha256(
            **self.model_dump(mode="python", exclude={"receipt_sha256"})
        )
        if self.receipt_sha256 != expected_receipt:
            raise ValueError("replan trigger receipt hash does not match its payload")
        return self

    @classmethod
    def create(cls, **values: object) -> Self:
        values = dict(values)
        disposition = TaskGraphSemanticVerificationDisposition(
            values["semantic_disposition"]
        )
        values["semantic_disposition"] = disposition
        values["trigger_reason"] = (
            AuxiliaryReplanTriggerReason.SEMANTIC_EVIDENCE_BLOCKED
            if disposition is TaskGraphSemanticVerificationDisposition.BLOCKED
            else AuxiliaryReplanTriggerReason.SEMANTIC_REVISION_REQUIRED
        )
        values["revision_reason"] = AuxiliaryGraphRevisionReason.VERIFICATION_FAILED
        values["semantic_result_sha256s"] = tuple(
            values["semantic_result_sha256s"]  # type: ignore[arg-type]
        )
        values["evidence_epoch_sha256"] = (
            canonical_auxiliary_replan_evidence_epoch_sha256(
                session_id=str(values["session_id"]),
                task_id=str(values["task_id"]),
                auxiliary_graph_id=str(values["auxiliary_graph_id"]),
                goal_id=str(values["goal_id"]),
                source_auxiliary_graph_revision=int(
                    values["source_auxiliary_graph_revision"]
                ),
                source_structure_sha256=str(values["source_structure_sha256"]),
                source_authority_snapshot_id=str(
                    values["source_authority_snapshot_id"]
                ),
                source_authority_snapshot_sha256=str(
                    values["source_authority_snapshot_sha256"]
                ),
                semantic_authority_projection_sha256=str(
                    values["semantic_authority_projection_sha256"]
                ),
                semantic_prompt_payload_sha256=str(
                    values["semantic_prompt_payload_sha256"]
                ),
                task_graph_proposal_sha256=str(
                    values["task_graph_proposal_sha256"]
                ),
            )
        )
        values["receipt_sha256"] = canonical_auxiliary_replan_trigger_receipt_sha256(
            **values
        )
        return cls.model_validate(values)


def canonical_auxiliary_replan_evidence_epoch_sha256(
    *,
    session_id: str,
    task_id: str,
    auxiliary_graph_id: str,
    goal_id: str,
    source_auxiliary_graph_revision: int,
    source_structure_sha256: str,
    source_authority_snapshot_id: str,
    source_authority_snapshot_sha256: str,
    semantic_authority_projection_sha256: str,
    semantic_prompt_payload_sha256: str,
    task_graph_proposal_sha256: str,
) -> str:
    """独立于评审运行，为冻结证据/提案 epoch 计算哈希。"""

    return _canonical_sha256(
        {
            "schema_version": "auxiliary-replan-evidence-epoch-v1",
            "session_id": session_id,
            "task_id": task_id,
            "auxiliary_graph_id": auxiliary_graph_id,
            "goal_id": goal_id,
            "source_auxiliary_graph_revision": source_auxiliary_graph_revision,
            "source_structure_sha256": source_structure_sha256,
            "source_authority_snapshot_id": source_authority_snapshot_id,
            "source_authority_snapshot_sha256": (
                source_authority_snapshot_sha256
            ),
            "semantic_authority_projection_sha256": (
                semantic_authority_projection_sha256
            ),
            "semantic_prompt_payload_sha256": semantic_prompt_payload_sha256,
            "task_graph_proposal_sha256": task_graph_proposal_sha256,
        }
    )


def canonical_auxiliary_replan_trigger_receipt_sha256(**values: object) -> str:
    payload = dict(values)
    payload.pop("receipt_sha256", None)
    payload["schema_version"] = "auxiliary-replan-trigger-receipt-v1"
    for name in ("semantic_disposition", "trigger_reason", "revision_reason"):
        value = payload.get(name)
        if isinstance(value, StrEnum):
            payload[name] = value.value
    hashes = payload.get("semantic_result_sha256s")
    if isinstance(hashes, tuple):
        payload["semantic_result_sha256s"] = list(hashes)
    return _canonical_sha256(payload)


class AuxiliaryReplanTriggerApplicationReceipt(_Contract):
    """revision N+1 存在后写入的不可变完成记录。"""

    schema_version: Literal["auxiliary-replan-trigger-application-receipt-v1"] = (
        "auxiliary-replan-trigger-application-receipt-v1"
    )
    apply_id: str = Field(pattern=_DURABLE_ID_PATTERN)
    trigger_id: str = Field(pattern=_DURABLE_ID_PATTERN)
    trigger_receipt_sha256: str = Field(pattern=_SHA256_PATTERN)
    session_id: str = Field(pattern=_DURABLE_ID_PATTERN)
    task_id: str = Field(pattern=_DURABLE_ID_PATTERN)
    auxiliary_graph_id: str = Field(pattern=_DURABLE_ID_PATTERN)
    goal_id: str = Field(pattern=_DURABLE_ID_PATTERN)
    source_auxiliary_graph_revision: int = Field(ge=1)
    applied_auxiliary_graph_revision: int = Field(ge=2)
    applied_structure_sha256: str = Field(pattern=_SHA256_PATTERN)
    revision_reason: Literal[AuxiliaryGraphRevisionReason.VERIFICATION_FAILED] = (
        AuxiliaryGraphRevisionReason.VERIFICATION_FAILED
    )
    consumed_turn_id: str = Field(pattern=_DURABLE_ID_PATTERN)
    receipt_sha256: str = Field(pattern=_SHA256_PATTERN)

    @model_validator(mode="after")
    def _validate_application(self) -> "AuxiliaryReplanTriggerApplicationReceipt":
        if self.applied_auxiliary_graph_revision != (
            self.source_auxiliary_graph_revision + 1
        ):
            raise ValueError("a replan trigger must apply to the next graph revision")
        expected = canonical_auxiliary_replan_trigger_application_sha256(
            **self.model_dump(mode="python", exclude={"receipt_sha256"})
        )
        if self.receipt_sha256 != expected:
            raise ValueError("replan application receipt hash does not match payload")
        return self

    @classmethod
    def create(cls, **values: object) -> Self:
        values = dict(values)
        values["revision_reason"] = AuxiliaryGraphRevisionReason.VERIFICATION_FAILED
        values["receipt_sha256"] = (
            canonical_auxiliary_replan_trigger_application_sha256(**values)
        )
        return cls.model_validate(values)


def canonical_auxiliary_replan_trigger_application_sha256(
    **values: object,
) -> str:
    payload = dict(values)
    payload.pop("receipt_sha256", None)
    payload["schema_version"] = (
        "auxiliary-replan-trigger-application-receipt-v1"
    )
    reason = payload.get("revision_reason")
    if isinstance(reason, StrEnum):
        payload["revision_reason"] = reason.value
    return _canonical_sha256(payload)


class AuxiliaryNodeReference(_Contract):
    node_id: str = Field(pattern=_DURABLE_ID_PATTERN)
    node_revision: int = Field(ge=1)


class AuxiliaryNodeDefinition(_Contract):
    """跨 DAG revision 使用、由 Host 物化的不可变节点定义。"""

    schema_version: Literal["auxiliary-node-definition-v2"] = (
        "auxiliary-node-definition-v2"
    )
    node_id: str = Field(pattern=_DURABLE_ID_PATTERN)
    node_revision: int = Field(ge=1)
    ordinal: int = Field(ge=0, lt=_MAX_AUXILIARY_NODES)
    node_kind: AuxiliaryNodeKind
    executor_kind: AuxiliaryNodeExecutorKind
    title: str = Field(min_length=1, max_length=240)
    objective: str = Field(min_length=1, max_length=2_000)
    acceptance_criteria: tuple[InSessionTaskAcceptanceProposal, ...] = Field(
        min_length=1,
        max_length=64,
    )
    capability_profile_id: str | None = Field(
        default=None,
        pattern=_DURABLE_ID_PATTERN,
    )
    input_resource_aliases: tuple[str, ...] = Field(default=(), max_length=64)
    source_anchor_ids: tuple[str, ...] = Field(min_length=1, max_length=64)
    output_contract: str = Field(pattern=_OUTPUT_CONTRACT_PATTERN)
    required: bool = True
    semantic_fingerprint: str = Field(pattern=_SHA256_PATTERN)
    origin_node_ref: AuxiliaryNodeReference | None = None

    @field_validator("title", "objective")
    @classmethod
    def _validate_text(cls, value: str, info: object) -> str:
        return _require_canonical_text(value, field_name=str(getattr(info, "field_name")))

    @field_validator("input_resource_aliases", "source_anchor_ids")
    @classmethod
    def _validate_aliases(cls, values: tuple[str, ...], info: object) -> tuple[str, ...]:
        for value in values:
            if not value or len(value) > 200:
                raise ValueError(f"{getattr(info, 'field_name')} contain an invalid alias")
        return _require_canonical_unique(
            values,
            field_name=str(getattr(info, "field_name")),
        )

    @field_validator("acceptance_criteria")
    @classmethod
    def _validate_acceptance_ids(
        cls,
        values: tuple[InSessionTaskAcceptanceProposal, ...],
    ) -> tuple[InSessionTaskAcceptanceProposal, ...]:
        ids = tuple(item.acceptance_id for item in values)
        _require_unique(ids, field_name="Acceptance IDs")
        return values

    @model_validator(mode="after")
    def _validate_definition(self) -> "AuxiliaryNodeDefinition":
        _validate_node_executor_pair(
            node_kind=self.node_kind,
            executor_kind=self.executor_kind,
            capability_profile_id=self.capability_profile_id,
            output_contract=self.output_contract,
        )
        if any(
            not set(item.source_anchor_ids).issubset(self.source_anchor_ids)
            for item in self.acceptance_criteria
        ):
            raise ValueError("node Acceptance authority must be declared by its node")
        if self.node_revision == 1 and self.origin_node_ref is not None:
            raise ValueError("node revision one cannot declare an origin node")
        if self.node_revision > 1:
            if self.origin_node_ref is None:
                raise ValueError("a revised node definition requires its origin node")
            if (
                self.origin_node_ref.node_id != self.node_id
                or self.origin_node_ref.node_revision != self.node_revision - 1
            ):
                raise ValueError("origin node must be the immediately previous definition")
        expected = canonical_auxiliary_node_semantic_fingerprint(
            node_kind=self.node_kind,
            executor_kind=self.executor_kind,
            title=self.title,
            objective=self.objective,
            acceptance_criteria=self.acceptance_criteria,
            capability_profile_id=self.capability_profile_id,
            input_resource_aliases=self.input_resource_aliases,
            source_anchor_ids=self.source_anchor_ids,
            output_contract=self.output_contract,
            required=self.required,
        )
        if self.semantic_fingerprint != expected:
            raise ValueError("node semantic fingerprint does not match its definition")
        return self

    @classmethod
    def create(cls, **values: object) -> Self:
        """构造节点并封存其语义定义哈希。"""

        values = dict(values)
        acceptance_criteria = tuple(
            item
            if isinstance(item, InSessionTaskAcceptanceProposal)
            else InSessionTaskAcceptanceProposal.model_validate(item)
            for item in values["acceptance_criteria"]  # type: ignore[union-attr]
        )
        values["acceptance_criteria"] = acceptance_criteria
        values["semantic_fingerprint"] = canonical_auxiliary_node_semantic_fingerprint(
            node_kind=AuxiliaryNodeKind(values["node_kind"]),
            executor_kind=AuxiliaryNodeExecutorKind(values["executor_kind"]),
            title=str(values["title"]),
            objective=str(values["objective"]),
            acceptance_criteria=acceptance_criteria,
            capability_profile_id=values.get("capability_profile_id"),  # type: ignore[arg-type]
            input_resource_aliases=tuple(
                values.get("input_resource_aliases", ())  # type: ignore[arg-type]
            ),
            source_anchor_ids=tuple(values["source_anchor_ids"]),  # type: ignore[arg-type]
            output_contract=str(values["output_contract"]),
            required=bool(values.get("required", True)),
        )
        return cls.model_validate(values)


class AuxiliaryGraphEdge(_Contract):
    """从上游节点到其消费者的一项不可变依赖。"""

    source_node_id: str = Field(pattern=_DURABLE_ID_PATTERN)
    target_node_id: str = Field(pattern=_DURABLE_ID_PATTERN)
    required: Literal[True] = True

    @model_validator(mode="after")
    def _reject_self_edge(self) -> "AuxiliaryGraphEdge":
        if self.source_node_id == self.target_node_id:
            raise ValueError("an AuxiliaryGraph edge cannot target itself")
        return self


class AuxiliaryGraphRevision(_Contract):
    """不可变规划 DAG 的一个可自认证完整快照。"""

    schema_version: Literal["auxiliary-graph-revision-v2"] = (
        "auxiliary-graph-revision-v2"
    )
    session_id: str = Field(pattern=_DURABLE_ID_PATTERN)
    task_id: str = Field(pattern=_DURABLE_ID_PATTERN)
    auxiliary_graph_id: str = Field(pattern=_DURABLE_ID_PATTERN)
    goal_id: str = Field(pattern=_DURABLE_ID_PATTERN)
    auxiliary_graph_revision: int = Field(ge=1)
    parent_auxiliary_graph_revision: int | None = Field(default=None, ge=1)
    base_task_graph_revision: int | None = Field(default=None, ge=1)
    source_turn_id: str = Field(pattern=_DURABLE_ID_PATTERN)
    revision_reason: AuxiliaryGraphRevisionReason
    authority_snapshot_id: str = Field(pattern=_DURABLE_ID_PATTERN)
    authority_snapshot_sha256: str = Field(pattern=_SHA256_PATTERN)
    terminal_node_id: str = Field(pattern=_DURABLE_ID_PATTERN)
    nodes: tuple[AuxiliaryNodeDefinition, ...] = Field(
        min_length=1,
        max_length=_MAX_AUXILIARY_NODES,
    )
    edges: tuple[AuxiliaryGraphEdge, ...] = Field(default=(), max_length=512)
    structure_sha256: str = Field(pattern=_SHA256_PATTERN)

    @model_validator(mode="after")
    def _validate_revision(self) -> "AuxiliaryGraphRevision":
        if self.auxiliary_graph_revision == 1:
            if self.parent_auxiliary_graph_revision is not None:
                raise ValueError("AuxiliaryGraph revision one cannot have a parent")
            if self.revision_reason is not AuxiliaryGraphRevisionReason.INITIAL:
                raise ValueError("AuxiliaryGraph revision one must use the initial reason")
        else:
            if self.parent_auxiliary_graph_revision != self.auxiliary_graph_revision - 1:
                raise ValueError("AuxiliaryGraph revisions must form a contiguous chain")
            if self.revision_reason is AuxiliaryGraphRevisionReason.INITIAL:
                raise ValueError("only AuxiliaryGraph revision one may be initial")
        _validate_materialized_graph(
            nodes=self.nodes,
            edges=self.edges,
            terminal_node_id=self.terminal_node_id,
        )
        expected = canonical_auxiliary_graph_revision_sha256(
            session_id=self.session_id,
            task_id=self.task_id,
            auxiliary_graph_id=self.auxiliary_graph_id,
            goal_id=self.goal_id,
            auxiliary_graph_revision=self.auxiliary_graph_revision,
            parent_auxiliary_graph_revision=self.parent_auxiliary_graph_revision,
            base_task_graph_revision=self.base_task_graph_revision,
            source_turn_id=self.source_turn_id,
            revision_reason=self.revision_reason,
            authority_snapshot_id=self.authority_snapshot_id,
            authority_snapshot_sha256=self.authority_snapshot_sha256,
            terminal_node_id=self.terminal_node_id,
            nodes=self.nodes,
            edges=self.edges,
        )
        if self.structure_sha256 != expected:
            raise ValueError("AuxiliaryGraph structure hash does not match its payload")
        return self

    @classmethod
    def create(cls, **values: object) -> Self:
        """构造规范物化图 revision 并计算哈希。"""

        values = dict(values)
        nodes = tuple(
            item
            if isinstance(item, AuxiliaryNodeDefinition)
            else AuxiliaryNodeDefinition.model_validate(item)
            for item in values["nodes"]  # type: ignore[union-attr]
        )
        edges = tuple(
            item
            if isinstance(item, AuxiliaryGraphEdge)
            else AuxiliaryGraphEdge.model_validate(item)
            for item in values.get("edges", ())  # type: ignore[union-attr]
        )
        values["nodes"] = nodes
        values["edges"] = edges
        values["structure_sha256"] = canonical_auxiliary_graph_revision_sha256(
            session_id=str(values["session_id"]),
            task_id=str(values["task_id"]),
            auxiliary_graph_id=str(values["auxiliary_graph_id"]),
            goal_id=str(values["goal_id"]),
            auxiliary_graph_revision=int(values["auxiliary_graph_revision"]),
            parent_auxiliary_graph_revision=values.get(  # type: ignore[arg-type]
                "parent_auxiliary_graph_revision"
            ),
            base_task_graph_revision=values.get(  # type: ignore[arg-type]
                "base_task_graph_revision"
            ),
            source_turn_id=str(values["source_turn_id"]),
            revision_reason=AuxiliaryGraphRevisionReason(values["revision_reason"]),
            authority_snapshot_id=str(values["authority_snapshot_id"]),
            authority_snapshot_sha256=str(values["authority_snapshot_sha256"]),
            terminal_node_id=str(values["terminal_node_id"]),
            nodes=nodes,
            edges=edges,
        )
        return cls.model_validate(values)


def canonical_auxiliary_node_semantic_fingerprint(
    *,
    node_kind: AuxiliaryNodeKind,
    executor_kind: AuxiliaryNodeExecutorKind,
    title: str,
    objective: str,
    acceptance_criteria: tuple[InSessionTaskAcceptanceProposal, ...],
    capability_profile_id: str | None,
    input_resource_aliases: tuple[str, ...],
    source_anchor_ids: tuple[str, ...],
    output_contract: str,
    required: bool,
) -> str:
    payload = {
        "schema_version": "auxiliary-node-semantic-definition-v2",
        "node_kind": node_kind.value,
        "executor_kind": executor_kind.value,
        "title": title,
        "objective": objective,
        "acceptance_criteria": [
            item.model_dump(mode="json") for item in acceptance_criteria
        ],
        "capability_profile_id": capability_profile_id,
        "input_resource_aliases": list(input_resource_aliases),
        "source_anchor_ids": list(source_anchor_ids),
        "output_contract": output_contract,
        "required": required,
    }
    return _canonical_sha256(payload)


def canonical_auxiliary_graph_revision_sha256(
    *,
    session_id: str,
    task_id: str,
    auxiliary_graph_id: str,
    goal_id: str,
    auxiliary_graph_revision: int,
    parent_auxiliary_graph_revision: int | None,
    base_task_graph_revision: int | None,
    source_turn_id: str,
    revision_reason: AuxiliaryGraphRevisionReason,
    authority_snapshot_id: str,
    authority_snapshot_sha256: str,
    terminal_node_id: str,
    nodes: tuple[AuxiliaryNodeDefinition, ...],
    edges: tuple[AuxiliaryGraphEdge, ...],
) -> str:
    payload = {
        "schema_version": "auxiliary-graph-revision-v2",
        "session_id": session_id,
        "task_id": task_id,
        "auxiliary_graph_id": auxiliary_graph_id,
        "goal_id": goal_id,
        "auxiliary_graph_revision": auxiliary_graph_revision,
        "parent_auxiliary_graph_revision": parent_auxiliary_graph_revision,
        "base_task_graph_revision": base_task_graph_revision,
        "source_turn_id": source_turn_id,
        "revision_reason": revision_reason.value,
        "authority_snapshot_id": authority_snapshot_id,
        "authority_snapshot_sha256": authority_snapshot_sha256,
        "terminal_node_id": terminal_node_id,
        "nodes": [item.model_dump(mode="json") for item in nodes],
        "edges": [item.model_dump(mode="json") for item in edges],
    }
    return _canonical_sha256(payload)


def _validate_node_executor_pair(
    *,
    node_kind: AuxiliaryNodeKind,
    executor_kind: AuxiliaryNodeExecutorKind,
    capability_profile_id: str | None,
    output_contract: str,
) -> None:
    if output_contract in _FORBIDDEN_AUXILIARY_OUTPUT_CONTRACTS:
        raise ValueError("legacy Attempt actions cannot be node output contracts")
    if executor_kind is AuxiliaryNodeExecutorKind.TERMINAL_PLANNER:
        if node_kind is not AuxiliaryNodeKind.SYNTHESIZE:
            raise ValueError("terminal planner must be a synthesize node")
        if capability_profile_id is not None:
            raise ValueError("terminal planner cannot declare a tool capability profile")
        if output_contract != "task_graph_revision_proposal_v2":
            raise ValueError("terminal planner requires the TaskGraph proposal contract")
        return
    if output_contract == "task_graph_revision_proposal_v2":
        raise ValueError(
            "only the terminal planner may produce the TaskGraph proposal contract"
        )
    if (
        executor_kind is AuxiliaryNodeExecutorKind.HOST_PRIMITIVE
        and not output_contract.startswith("planning_context")
    ):
        raise ValueError(
            "a Host primitive must produce a planning_context output contract"
        )
    if node_kind is AuxiliaryNodeKind.SYNTHESIZE:
        raise ValueError("a synthesize node must use the terminal planner executor")
    if executor_kind is AuxiliaryNodeExecutorKind.USER_GATE:
        if node_kind is not AuxiliaryNodeKind.CLARIFY:
            raise ValueError("a user gate must be a clarify node")
        if capability_profile_id is not None:
            raise ValueError("a user gate cannot declare a tool capability profile")
        return
    if capability_profile_id is None:
        raise ValueError("executable investigation nodes require a capability profile")


def _validate_materialized_graph(
    *,
    nodes: tuple[AuxiliaryNodeDefinition, ...],
    edges: tuple[AuxiliaryGraphEdge, ...],
    terminal_node_id: str,
) -> None:
    node_ids = tuple(item.node_id for item in nodes)
    if len(node_ids) != len(set(node_ids)):
        raise ValueError("AuxiliaryGraph node IDs must be unique")
    if tuple(item.ordinal for item in nodes) != tuple(range(len(nodes))):
        raise ValueError("AuxiliaryGraph node ordinals must be contiguous from zero")
    terminals = tuple(
        item.node_id
        for item in nodes
        if item.node_kind is AuxiliaryNodeKind.SYNTHESIZE
        and item.executor_kind is AuxiliaryNodeExecutorKind.TERMINAL_PLANNER
    )
    if terminals != (terminal_node_id,):
        raise ValueError("AuxiliaryGraph requires exactly its declared terminal node")
    if not next(item for item in nodes if item.node_id == terminal_node_id).required:
        raise ValueError("AuxiliaryGraph terminal node must be required")
    node_id_set = set(node_ids)
    edge_pairs = tuple((item.source_node_id, item.target_node_id) for item in edges)
    if len(edge_pairs) != len(set(edge_pairs)):
        raise ValueError("AuxiliaryGraph edges must be unique")
    if any(source not in node_id_set or target not in node_id_set for source, target in edge_pairs):
        raise ValueError("AuxiliaryGraph edge references an unknown node")
    ordinal = {item.node_id: item.ordinal for item in nodes}
    expected_edge_order = tuple(
        sorted(
            edges,
            key=lambda item: (
                ordinal[item.source_node_id],
                ordinal[item.target_node_id],
            ),
        )
    )
    if edges != expected_edge_order:
        raise ValueError("AuxiliaryGraph edges must use canonical node-ordinal order")
    outgoing: dict[str, set[str]] = {node_id: set() for node_id in node_ids}
    indegree = {node_id: 0 for node_id in node_ids}
    for edge in edges:
        outgoing[edge.source_node_id].add(edge.target_node_id)
        indegree[edge.target_node_id] += 1
    if outgoing[terminal_node_id]:
        raise ValueError("AuxiliaryGraph terminal node must be a sink")
    frontier = [node_id for node_id in node_ids if indegree[node_id] == 0]
    visited: list[str] = []
    while frontier:
        current = frontier.pop(0)
        visited.append(current)
        for target in sorted(outgoing[current], key=ordinal.__getitem__):
            indegree[target] -= 1
            if indegree[target] == 0:
                frontier.append(target)
                frontier.sort(key=ordinal.__getitem__)
    if len(visited) != len(nodes):
        raise ValueError("AuxiliaryGraph must be acyclic")
    reverse: dict[str, set[str]] = {node_id: set() for node_id in node_ids}
    for source, target in edge_pairs:
        reverse[target].add(source)
    can_reach_terminal = {terminal_node_id}
    stack = [terminal_node_id]
    while stack:
        target = stack.pop()
        for source in reverse[target]:
            if source not in can_reach_terminal:
                can_reach_terminal.add(source)
                stack.append(source)
    required_ids = {item.node_id for item in nodes if item.required}
    if not required_ids.issubset(can_reach_terminal):
        raise ValueError("every required node must reach the terminal node")
    depth = {node_id: 1 for node_id in node_ids}
    for node_id in visited:
        for target in outgoing[node_id]:
            depth[target] = max(depth[target], depth[node_id] + 1)
    if max(depth.values()) > _MAX_AUXILIARY_DEPTH:
        raise ValueError("AuxiliaryGraph exceeds the maximum depth")


class AuxiliaryNodeProposal(_Contract):
    """由 Architect 编写、仅通过提案局部键寻址的节点。"""

    node_key: str = Field(pattern=_LOCAL_KEY_PATTERN)
    node_kind: AuxiliaryNodeKind
    executor_kind: AuxiliaryNodeExecutorKind
    title: str = Field(min_length=1, max_length=240)
    objective: str = Field(min_length=1, max_length=2_000)
    acceptance_criteria: tuple[InSessionTaskAcceptanceProposal, ...] = Field(
        min_length=1,
        max_length=64,
    )
    capability_profile_id: str | None = Field(
        default=None,
        pattern=_DURABLE_ID_PATTERN,
    )
    input_resource_aliases: tuple[str, ...] = Field(default=(), max_length=64)
    source_anchor_ids: tuple[str, ...] = Field(min_length=1, max_length=64)
    output_contract: str = Field(pattern=_OUTPUT_CONTRACT_PATTERN)
    required: bool = True
    origin_node_alias: str | None = Field(default=None, pattern=_LOCAL_KEY_PATTERN)

    @field_validator("title", "objective")
    @classmethod
    def _validate_text(cls, value: str, info: object) -> str:
        return _require_canonical_text(value, field_name=str(getattr(info, "field_name")))

    @field_validator("input_resource_aliases", "source_anchor_ids")
    @classmethod
    def _validate_aliases(cls, values: tuple[str, ...], info: object) -> tuple[str, ...]:
        for value in values:
            if not value or len(value) > 200:
                raise ValueError(f"{getattr(info, 'field_name')} contain an invalid alias")
        return _require_canonical_unique(
            values,
            field_name=str(getattr(info, "field_name")),
        )

    @field_validator("acceptance_criteria")
    @classmethod
    def _validate_acceptance_ids(
        cls,
        values: tuple[InSessionTaskAcceptanceProposal, ...],
    ) -> tuple[InSessionTaskAcceptanceProposal, ...]:
        _require_unique(
            tuple(item.acceptance_id for item in values),
            field_name="Acceptance IDs",
        )
        return values

    @model_validator(mode="after")
    def _validate_executor(self) -> "AuxiliaryNodeProposal":
        _validate_node_executor_pair(
            node_kind=self.node_kind,
            executor_kind=self.executor_kind,
            capability_profile_id=self.capability_profile_id,
            output_contract=self.output_contract,
        )
        if any(
            not set(item.source_anchor_ids).issubset(self.source_anchor_ids)
            for item in self.acceptance_criteria
        ):
            raise ValueError("proposal Acceptance authority must be declared by its node")
        return self


class AuxiliaryGraphEdgeProposal(_Contract):
    source_node_key: str = Field(pattern=_LOCAL_KEY_PATTERN)
    target_node_key: str = Field(pattern=_LOCAL_KEY_PATTERN)
    required: Literal[True] = True

    @model_validator(mode="after")
    def _reject_self_edge(self) -> "AuxiliaryGraphEdgeProposal":
        if self.source_node_key == self.target_node_key:
            raise ValueError("an AuxiliaryGraph proposal edge cannot target itself")
        return self


class AuxiliaryGraphStructureProposal(_Contract):
    """Architect 提议的完整不可信 DAG 形状。"""

    terminal_node_key: str = Field(pattern=_LOCAL_KEY_PATTERN)
    nodes: tuple[AuxiliaryNodeProposal, ...] = Field(
        min_length=1,
        max_length=_MAX_AUXILIARY_NODES,
    )
    edges: tuple[AuxiliaryGraphEdgeProposal, ...] = Field(default=(), max_length=512)

    @model_validator(mode="after")
    def _validate_topology(self) -> "AuxiliaryGraphStructureProposal":
        node_keys = tuple(item.node_key for item in self.nodes)
        if len(node_keys) != len(set(node_keys)):
            raise ValueError("AuxiliaryGraph proposal node keys must be unique")
        terminals = tuple(
            item.node_key
            for item in self.nodes
            if item.node_kind is AuxiliaryNodeKind.SYNTHESIZE
            and item.executor_kind is AuxiliaryNodeExecutorKind.TERMINAL_PLANNER
        )
        if terminals != (self.terminal_node_key,):
            raise ValueError("proposal requires exactly its declared terminal node")
        if not next(
            item for item in self.nodes if item.node_key == self.terminal_node_key
        ).required:
            raise ValueError("proposal terminal node must be required")
        key_set = set(node_keys)
        pairs = tuple((item.source_node_key, item.target_node_key) for item in self.edges)
        if len(pairs) != len(set(pairs)):
            raise ValueError("AuxiliaryGraph proposal edges must be unique")
        if any(source not in key_set or target not in key_set for source, target in pairs):
            raise ValueError("AuxiliaryGraph proposal edge references an unknown node")
        ordinal = {key: index for index, key in enumerate(node_keys)}
        expected_edges = tuple(
            sorted(
                self.edges,
                key=lambda item: (
                    ordinal[item.source_node_key],
                    ordinal[item.target_node_key],
                ),
            )
        )
        if self.edges != expected_edges:
            raise ValueError("proposal edges must use canonical node order")
        outgoing: dict[str, set[str]] = {key: set() for key in node_keys}
        indegree = {key: 0 for key in node_keys}
        for source, target in pairs:
            outgoing[source].add(target)
            indegree[target] += 1
        if outgoing[self.terminal_node_key]:
            raise ValueError("proposal terminal node must be a sink")
        frontier = [key for key in node_keys if indegree[key] == 0]
        visited: list[str] = []
        while frontier:
            current = frontier.pop(0)
            visited.append(current)
            for target in sorted(outgoing[current], key=ordinal.__getitem__):
                indegree[target] -= 1
                if indegree[target] == 0:
                    frontier.append(target)
                    frontier.sort(key=ordinal.__getitem__)
        if len(visited) != len(node_keys):
            raise ValueError("AuxiliaryGraph proposal must be acyclic")
        reverse: dict[str, set[str]] = {key: set() for key in node_keys}
        for source, target in pairs:
            reverse[target].add(source)
        reachable = {self.terminal_node_key}
        stack = [self.terminal_node_key]
        while stack:
            target = stack.pop()
            for source in reverse[target]:
                if source not in reachable:
                    reachable.add(source)
                    stack.append(source)
        required = {item.node_key for item in self.nodes if item.required}
        if not required.issubset(reachable):
            raise ValueError("every required proposal node must reach the terminal")
        depth = {key: 1 for key in node_keys}
        for key in visited:
            for target in outgoing[key]:
                depth[target] = max(depth[target], depth[key] + 1)
        if max(depth.values()) > _MAX_AUXILIARY_DEPTH:
            raise ValueError("AuxiliaryGraph proposal exceeds the maximum depth")
        return self


class AuxiliaryGraphRevisionProposalDisposition(StrEnum):
    CREATE_REVISION = "create_revision"
    CONTINUE_CURRENT = "continue_current"
    REVISE_REVISION = "revise_revision"
    SUPERSEDE_AND_REBASE = "supersede_and_rebase"
    TERMINAL_FAIL = "terminal_fail"


class AuxiliaryGraphRevisionProposal(_Contract):
    """不含 Host 所有持久身份的类型化 Architect 决定。"""

    schema_version: Literal["auxiliary-graph-revision-proposal-v2"] = (
        "auxiliary-graph-revision-proposal-v2"
    )
    disposition: AuxiliaryGraphRevisionProposalDisposition
    expected_current_auxiliary_graph_revision: int | None = Field(
        default=None,
        ge=1,
    )
    revision_reason: AuxiliaryGraphRevisionReason | None = None
    structure: AuxiliaryGraphStructureProposal | None = None
    explanation: str = Field(min_length=1, max_length=2_000)
    blocking_gap_ids: tuple[str, ...] = Field(default=(), max_length=128)
    requested_user_question: str | None = Field(
        default=None,
        min_length=1,
        max_length=2_000,
    )
    failure_reason: str | None = Field(default=None, min_length=1, max_length=2_000)

    @field_validator("explanation", "requested_user_question", "failure_reason")
    @classmethod
    def _validate_optional_text(cls, value: str | None, info: object) -> str | None:
        if value is None:
            return None
        return _require_canonical_text(value, field_name=str(getattr(info, "field_name")))

    @field_validator("blocking_gap_ids")
    @classmethod
    def _validate_gap_ids(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        return _require_canonical_unique(values, field_name="blocking gap IDs")

    @model_validator(mode="after")
    def _validate_disposition_shape(self) -> "AuxiliaryGraphRevisionProposal":
        disposition = self.disposition
        if disposition is AuxiliaryGraphRevisionProposalDisposition.CREATE_REVISION:
            if self.expected_current_auxiliary_graph_revision is not None:
                raise ValueError("create_revision requires a base-null AuxiliaryGraph")
            if self.revision_reason is not AuxiliaryGraphRevisionReason.INITIAL:
                raise ValueError("create_revision requires the initial revision reason")
            if self.structure is None:
                raise ValueError("create_revision requires a complete DAG structure")
        elif disposition is AuxiliaryGraphRevisionProposalDisposition.REVISE_REVISION:
            if self.expected_current_auxiliary_graph_revision is None:
                raise ValueError("revise_revision requires the expected current revision")
            if self.revision_reason in (None, AuxiliaryGraphRevisionReason.INITIAL):
                raise ValueError("revise_revision requires a non-initial reason")
            if self.structure is None:
                raise ValueError("revise_revision requires a complete DAG structure")
        else:
            if self.structure is not None or self.revision_reason is not None:
                raise ValueError("a non-revision disposition cannot carry a DAG structure")
        if disposition is AuxiliaryGraphRevisionProposalDisposition.CONTINUE_CURRENT:
            if self.expected_current_auxiliary_graph_revision is None:
                raise ValueError("continue_current requires an existing revision")
        if disposition is AuxiliaryGraphRevisionProposalDisposition.TERMINAL_FAIL:
            if self.failure_reason is None:
                raise ValueError("terminal_fail requires a failure reason")
        elif self.failure_reason is not None:
            raise ValueError("only terminal_fail may carry a failure reason")
        if self.requested_user_question is not None and not self.blocking_gap_ids:
            raise ValueError("a user question must identify at least one blocking gap")
        return self


# ---------------------------------------------------------------------------
# 冻结 authority 与 PlanningContextArtifact 契约
# ---------------------------------------------------------------------------


class PlanningAuthorityClass(StrEnum):
    AUTHORIZATION = "authorization"
    EVIDENCE = "evidence"
    GAP = "gap"


class PlanningAuthorityOriginKind(StrEnum):
    USER_INSTRUCTION_SPAN = "user_instruction_span"
    USER_ANSWER_SPAN = "user_answer_span"
    PRIOR_TASK_STATE = "prior_task_state"
    RETRIEVED_SOURCE_UNIT = "retrieved_source_unit"
    WORKSPACE_RESOURCE = "workspace_resource"
    TOOL_RESULT = "tool_result"
    VISUAL_UNIT = "visual_unit"
    MEMORY_RECORD = "memory_record"
    ARTIFACT = "artifact"
    PRIMITIVE_RESULT = "primitive_result"
    GAP_OBSERVATION = "gap_observation"


class PlanningAuthorityAnchor(_Contract):
    """私有源 authority 及向模型公开的不透明别名。"""

    schema_version: Literal["planning-authority-anchor-v1"] = (
        "planning-authority-anchor-v1"
    )
    anchor_id: str = Field(pattern=_DURABLE_ID_PATTERN)
    authority_snapshot_id: str = Field(pattern=_DURABLE_ID_PATTERN)
    projection_alias: str = Field(pattern=_LOCAL_KEY_PATTERN)
    authority_class: PlanningAuthorityClass
    origin_kind: PlanningAuthorityOriginKind
    origin_id: str = Field(min_length=1, max_length=500)
    source_revision: int | None = Field(default=None, ge=1)
    content_sha256: str = Field(pattern=_SHA256_PATTERN)
    item_ordinal: int = Field(ge=0, le=1_000_000)
    span_start: int | None = Field(default=None, ge=0)
    span_end: int | None = Field(default=None, ge=1)
    projection_sha256: str = Field(pattern=_SHA256_PATTERN)
    freshness_binding_sha256: str = Field(pattern=_SHA256_PATTERN)
    disclosure_receipt_id: str | None = Field(
        default=None,
        pattern=_DURABLE_ID_PATTERN,
    )

    @field_validator("origin_id")
    @classmethod
    def _validate_origin_id(cls, value: str) -> str:
        return _require_canonical_text(value, field_name="authority origin ID")

    @model_validator(mode="after")
    def _validate_authority(self) -> "PlanningAuthorityAnchor":
        authorization_origins = {
            PlanningAuthorityOriginKind.USER_INSTRUCTION_SPAN,
            PlanningAuthorityOriginKind.USER_ANSWER_SPAN,
            PlanningAuthorityOriginKind.PRIOR_TASK_STATE,
        }
        if self.authority_class is PlanningAuthorityClass.AUTHORIZATION:
            if self.origin_kind not in authorization_origins:
                raise ValueError("only user or verified prior Task state may authorize scope")
        elif self.origin_kind in authorization_origins:
            raise ValueError("user/prior-Task authority cannot masquerade as external evidence")
        if (self.authority_class is PlanningAuthorityClass.GAP) != (
            self.origin_kind is PlanningAuthorityOriginKind.GAP_OBSERVATION
        ):
            raise ValueError("gap authority requires a typed gap observation origin")
        if (self.span_start is None) != (self.span_end is None):
            raise ValueError("authority span bounds must be set together")
        if self.origin_kind in {
            PlanningAuthorityOriginKind.USER_INSTRUCTION_SPAN,
            PlanningAuthorityOriginKind.USER_ANSWER_SPAN,
        } and self.span_start is None:
            raise ValueError("user authority origins require an exact source span")
        if (
            self.origin_kind is PlanningAuthorityOriginKind.PRIOR_TASK_STATE
            and self.source_revision is None
        ):
            raise ValueError("prior Task state authority requires a source revision")
        if (
            self.span_start is not None
            and self.span_end <= self.span_start  # type: ignore[operator]
        ):
            raise ValueError("authority span end must exceed its start")
        return self


class PlanningAuthoritySnapshot(_Contract):
    """为一个规划目标冻结的可自认证 authority 命名空间。"""

    schema_version: Literal["planning-authority-snapshot-v1"] = (
        "planning-authority-snapshot-v1"
    )
    authority_snapshot_id: str = Field(pattern=_DURABLE_ID_PATTERN)
    session_id: str = Field(pattern=_DURABLE_ID_PATTERN)
    task_id: str = Field(pattern=_DURABLE_ID_PATTERN)
    auxiliary_graph_id: str = Field(pattern=_DURABLE_ID_PATTERN)
    goal_id: str = Field(pattern=_DURABLE_ID_PATTERN)
    source_turn_id: str = Field(pattern=_DURABLE_ID_PATTERN)
    anchors: tuple[PlanningAuthorityAnchor, ...] = Field(
        min_length=1,
    )
    snapshot_sha256: str = Field(pattern=_SHA256_PATTERN)

    @model_validator(mode="after")
    def _validate_snapshot(self) -> "PlanningAuthoritySnapshot":
        if any(
            item.authority_snapshot_id != self.authority_snapshot_id
            for item in self.anchors
        ):
            raise ValueError("every authority anchor must belong to its snapshot")
        anchor_ids = tuple(item.anchor_id for item in self.anchors)
        aliases = tuple(item.projection_alias for item in self.anchors)
        _require_unique(anchor_ids, field_name="authority anchor IDs")
        _require_unique(aliases, field_name="authority projection aliases")
        if aliases != tuple(sorted(aliases)):
            raise ValueError("authority anchors must use projection-alias order")
        if not any(
            item.authority_class is PlanningAuthorityClass.AUTHORIZATION
            for item in self.anchors
        ):
            raise ValueError("an authority snapshot requires an authorization anchor")
        expected = canonical_planning_authority_snapshot_sha256(
            authority_snapshot_id=self.authority_snapshot_id,
            session_id=self.session_id,
            task_id=self.task_id,
            auxiliary_graph_id=self.auxiliary_graph_id,
            goal_id=self.goal_id,
            source_turn_id=self.source_turn_id,
            anchors=self.anchors,
        )
        if self.snapshot_sha256 != expected:
            raise ValueError("authority snapshot hash does not match its payload")
        return self

    @classmethod
    def create(cls, **values: object) -> Self:
        values = dict(values)
        anchors = tuple(
            item
            if isinstance(item, PlanningAuthorityAnchor)
            else PlanningAuthorityAnchor.model_validate(item)
            for item in values["anchors"]  # type: ignore[union-attr]
        )
        values["anchors"] = anchors
        values["snapshot_sha256"] = canonical_planning_authority_snapshot_sha256(
            authority_snapshot_id=str(values["authority_snapshot_id"]),
            session_id=str(values["session_id"]),
            task_id=str(values["task_id"]),
            auxiliary_graph_id=str(values["auxiliary_graph_id"]),
            goal_id=str(values["goal_id"]),
            source_turn_id=str(values["source_turn_id"]),
            anchors=anchors,
        )
        return cls.model_validate(values)


def canonical_planning_authority_snapshot_sha256(
    *,
    authority_snapshot_id: str,
    session_id: str,
    task_id: str,
    auxiliary_graph_id: str,
    goal_id: str,
    source_turn_id: str,
    anchors: tuple[PlanningAuthorityAnchor, ...],
) -> str:
    return _canonical_sha256(
        {
            "schema_version": "planning-authority-snapshot-v1",
            "authority_snapshot_id": authority_snapshot_id,
            "session_id": session_id,
            "task_id": task_id,
            "auxiliary_graph_id": auxiliary_graph_id,
            "goal_id": goal_id,
            "source_turn_id": source_turn_id,
            "anchors": [item.model_dump(mode="json") for item in anchors],
        }
    )


class PlanningObservationStatus(StrEnum):
    SUCCESS = "success"
    NO_MATCH = "no_match"
    PARTIAL = "partial"
    BLOCKED = "blocked"
    FAILED = "failed"
    STALE = "stale"


class PlanningContextFact(_Contract):
    fact_id: str = Field(pattern=_LOCAL_KEY_PATTERN)
    statement: str = Field(min_length=1, max_length=4_000)
    evidence_anchor_ids: tuple[str, ...] = Field(min_length=1, max_length=32)

    @field_validator("statement")
    @classmethod
    def _validate_statement(cls, value: str) -> str:
        return _require_canonical_text(value, field_name="fact statement")

    @field_validator("evidence_anchor_ids")
    @classmethod
    def _validate_anchors(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        return _require_canonical_unique(values, field_name="fact evidence anchors")


class PlanningContextConstraint(_Contract):
    constraint_id: str = Field(pattern=_LOCAL_KEY_PATTERN)
    statement: str = Field(min_length=1, max_length=4_000)
    authorization_anchor_ids: tuple[str, ...] = Field(min_length=1, max_length=32)

    @field_validator("statement")
    @classmethod
    def _validate_statement(cls, value: str) -> str:
        return _require_canonical_text(value, field_name="constraint statement")

    @field_validator("authorization_anchor_ids")
    @classmethod
    def _validate_anchors(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        return _require_canonical_unique(
            values,
            field_name="constraint authorization anchors",
        )


class PlanningContextConflict(_Contract):
    conflict_id: str = Field(pattern=_LOCAL_KEY_PATTERN)
    statement: str = Field(min_length=1, max_length=4_000)
    evidence_anchor_ids: tuple[str, ...] = Field(min_length=2, max_length=32)
    blocking: bool

    @field_validator("statement")
    @classmethod
    def _validate_statement(cls, value: str) -> str:
        return _require_canonical_text(value, field_name="conflict statement")

    @field_validator("evidence_anchor_ids")
    @classmethod
    def _validate_anchors(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        return _require_canonical_unique(values, field_name="conflict evidence anchors")


class PlanningContextGap(_Contract):
    gap_id: str = Field(pattern=_LOCAL_KEY_PATTERN)
    observation_status: PlanningObservationStatus
    description: str = Field(min_length=1, max_length=4_000)
    blocking: bool
    affected_obligations: tuple[str, ...] = Field(min_length=1, max_length=64)
    evidence_anchor_ids: tuple[str, ...] = Field(default=(), max_length=32)
    resolution_hint: str | None = Field(default=None, min_length=1, max_length=2_000)

    @field_validator("description", "resolution_hint")
    @classmethod
    def _validate_text(cls, value: str | None, info: object) -> str | None:
        if value is None:
            return None
        return _require_canonical_text(value, field_name=str(getattr(info, "field_name")))

    @field_validator("affected_obligations", "evidence_anchor_ids")
    @classmethod
    def _validate_references(cls, values: tuple[str, ...], info: object) -> tuple[str, ...]:
        return _require_canonical_unique(
            values,
            field_name=str(getattr(info, "field_name")),
        )

    @model_validator(mode="after")
    def _validate_gap_status(self) -> "PlanningContextGap":
        if self.observation_status is PlanningObservationStatus.SUCCESS:
            raise ValueError("a successful observation cannot be represented as a gap")
        return self


class PlanningEvidenceRef(_Contract):
    evidence_anchor_id: str = Field(pattern=_DURABLE_ID_PATTERN)
    source_alias: str = Field(pattern=_LOCAL_KEY_PATTERN)
    locator: str = Field(min_length=1, max_length=1_000)
    content_sha256: str = Field(pattern=_SHA256_PATTERN)
    source_revision: int | None = Field(default=None, ge=1)
    freshness_binding_sha256: str = Field(pattern=_SHA256_PATTERN)

    @field_validator("locator")
    @classmethod
    def _validate_locator(cls, value: str) -> str:
        return _require_canonical_text(value, field_name="evidence locator")


class PlanningContextArtifact(_Contract):
    """供 planner 和 verifier 使用的不可变已验证规划观测。"""

    schema_version: Literal["planning-context-artifact-v1"] = (
        "planning-context-artifact-v1"
    )
    artifact_id: str = Field(pattern=_DURABLE_ID_PATTERN)
    session_id: str = Field(pattern=_DURABLE_ID_PATTERN)
    task_id: str = Field(pattern=_DURABLE_ID_PATTERN)
    auxiliary_graph_id: str = Field(pattern=_DURABLE_ID_PATTERN)
    goal_id: str = Field(pattern=_DURABLE_ID_PATTERN)
    producer_auxiliary_node: AuxiliaryNodeSubject
    producer_work_run_id: str | None = Field(default=None, pattern=_DURABLE_ID_PATTERN)
    producer_attempt_id: str | None = Field(default=None, pattern=_DURABLE_ID_PATTERN)
    producer_tool_result_ids: tuple[str, ...] = Field(default=(), max_length=64)
    producer_primitive_call_id: str | None = Field(
        default=None,
        pattern=_DURABLE_ID_PATTERN,
    )
    scope_snapshot_sha256: str = Field(pattern=_SHA256_PATTERN)
    facts: tuple[PlanningContextFact, ...] = Field(default=(), max_length=256)
    constraints: tuple[PlanningContextConstraint, ...] = Field(
        default=(),
        max_length=128,
    )
    conflicts: tuple[PlanningContextConflict, ...] = Field(
        default=(),
        max_length=128,
    )
    gaps: tuple[PlanningContextGap, ...] = Field(default=(), max_length=128)
    evidence_refs: tuple[PlanningEvidenceRef, ...] = Field(default=(), max_length=512)
    freshness_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    verification_receipt_id: str = Field(pattern=_DURABLE_ID_PATTERN)
    verification_receipt_sha256: str = Field(pattern=_SHA256_PATTERN)
    artifact_sha256: str = Field(pattern=_SHA256_PATTERN)

    @field_validator("producer_tool_result_ids")
    @classmethod
    def _validate_tool_results(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        return _require_canonical_unique(values, field_name="producer ToolResult IDs")

    @model_validator(mode="after")
    def _validate_artifact(self) -> "PlanningContextArtifact":
        subject = self.producer_auxiliary_node
        if subject.task_id != self.task_id or subject.auxiliary_graph_id != self.auxiliary_graph_id:
            raise ValueError("artifact producer must belong to its Task and AuxiliaryGraph")
        work_provenance = self.producer_work_run_id is not None
        primitive_provenance = self.producer_primitive_call_id is not None
        if work_provenance == primitive_provenance:
            raise ValueError("artifact requires exactly one WorkRun or primitive provenance")
        if work_provenance != (self.producer_attempt_id is not None):
            raise ValueError("WorkRun artifact provenance requires its Attempt")
        if primitive_provenance and self.producer_tool_result_ids:
            raise ValueError("primitive provenance cannot carry WorkRun ToolResults")
        if not any((self.facts, self.constraints, self.conflicts, self.gaps)):
            raise ValueError("a context artifact must contain a typed observation item")
        _require_unique(tuple(item.fact_id for item in self.facts), field_name="fact IDs")
        _require_unique(
            tuple(item.constraint_id for item in self.constraints),
            field_name="constraint IDs",
        )
        _require_unique(
            tuple(item.conflict_id for item in self.conflicts),
            field_name="conflict IDs",
        )
        _require_unique(tuple(item.gap_id for item in self.gaps), field_name="gap IDs")
        evidence_ids = tuple(item.evidence_anchor_id for item in self.evidence_refs)
        _require_unique(evidence_ids, field_name="evidence reference anchors")
        declared = set(evidence_ids)
        used = {
            anchor_id
            for item in (*self.facts, *self.conflicts, *self.gaps)
            for anchor_id in item.evidence_anchor_ids
        }
        if not used.issubset(declared):
            raise ValueError("context items reference undeclared evidence anchors")
        expected = canonical_planning_context_artifact_sha256(
            artifact_id=self.artifact_id,
            session_id=self.session_id,
            task_id=self.task_id,
            auxiliary_graph_id=self.auxiliary_graph_id,
            goal_id=self.goal_id,
            producer_auxiliary_node=self.producer_auxiliary_node,
            producer_work_run_id=self.producer_work_run_id,
            producer_attempt_id=self.producer_attempt_id,
            producer_tool_result_ids=self.producer_tool_result_ids,
            producer_primitive_call_id=self.producer_primitive_call_id,
            scope_snapshot_sha256=self.scope_snapshot_sha256,
            facts=self.facts,
            constraints=self.constraints,
            conflicts=self.conflicts,
            gaps=self.gaps,
            evidence_refs=self.evidence_refs,
            freshness_manifest_sha256=self.freshness_manifest_sha256,
            verification_receipt_id=self.verification_receipt_id,
            verification_receipt_sha256=self.verification_receipt_sha256,
        )
        if self.artifact_sha256 != expected:
            raise ValueError("context artifact hash does not match its payload")
        return self

    @classmethod
    def create(cls, **values: object) -> Self:
        values = dict(values)
        producer = values["producer_auxiliary_node"]
        if not isinstance(producer, AuxiliaryNodeSubject):
            producer = AuxiliaryNodeSubject.model_validate(producer)
        facts = tuple(
            item
            if isinstance(item, PlanningContextFact)
            else PlanningContextFact.model_validate(item)
            for item in values.get("facts", ())  # type: ignore[union-attr]
        )
        constraints = tuple(
            item
            if isinstance(item, PlanningContextConstraint)
            else PlanningContextConstraint.model_validate(item)
            for item in values.get("constraints", ())  # type: ignore[union-attr]
        )
        conflicts = tuple(
            item
            if isinstance(item, PlanningContextConflict)
            else PlanningContextConflict.model_validate(item)
            for item in values.get("conflicts", ())  # type: ignore[union-attr]
        )
        gaps = tuple(
            item
            if isinstance(item, PlanningContextGap)
            else PlanningContextGap.model_validate(item)
            for item in values.get("gaps", ())  # type: ignore[union-attr]
        )
        evidence_refs = tuple(
            item
            if isinstance(item, PlanningEvidenceRef)
            else PlanningEvidenceRef.model_validate(item)
            for item in values.get("evidence_refs", ())  # type: ignore[union-attr]
        )
        values.update(
            producer_auxiliary_node=producer,
            facts=facts,
            constraints=constraints,
            conflicts=conflicts,
            gaps=gaps,
            evidence_refs=evidence_refs,
        )
        values["artifact_sha256"] = canonical_planning_context_artifact_sha256(
            artifact_id=str(values["artifact_id"]),
            session_id=str(values["session_id"]),
            task_id=str(values["task_id"]),
            auxiliary_graph_id=str(values["auxiliary_graph_id"]),
            goal_id=str(values["goal_id"]),
            producer_auxiliary_node=producer,
            producer_work_run_id=values.get("producer_work_run_id"),  # type: ignore[arg-type]
            producer_attempt_id=values.get("producer_attempt_id"),  # type: ignore[arg-type]
            producer_tool_result_ids=tuple(
                values.get("producer_tool_result_ids", ())  # type: ignore[arg-type]
            ),
            producer_primitive_call_id=values.get(  # type: ignore[arg-type]
                "producer_primitive_call_id"
            ),
            scope_snapshot_sha256=str(values["scope_snapshot_sha256"]),
            facts=facts,
            constraints=constraints,
            conflicts=conflicts,
            gaps=gaps,
            evidence_refs=evidence_refs,
            freshness_manifest_sha256=str(values["freshness_manifest_sha256"]),
            verification_receipt_id=str(values["verification_receipt_id"]),
            verification_receipt_sha256=str(values["verification_receipt_sha256"]),
        )
        return cls.model_validate(values)


def canonical_planning_context_artifact_sha256(
    *,
    artifact_id: str,
    session_id: str,
    task_id: str,
    auxiliary_graph_id: str,
    goal_id: str,
    producer_auxiliary_node: AuxiliaryNodeSubject,
    producer_work_run_id: str | None,
    producer_attempt_id: str | None,
    producer_tool_result_ids: tuple[str, ...],
    producer_primitive_call_id: str | None,
    scope_snapshot_sha256: str,
    facts: tuple[PlanningContextFact, ...],
    constraints: tuple[PlanningContextConstraint, ...],
    conflicts: tuple[PlanningContextConflict, ...],
    gaps: tuple[PlanningContextGap, ...],
    evidence_refs: tuple[PlanningEvidenceRef, ...],
    freshness_manifest_sha256: str,
    verification_receipt_id: str,
    verification_receipt_sha256: str,
) -> str:
    return _canonical_sha256(
        {
            "schema_version": "planning-context-artifact-v1",
            "artifact_id": artifact_id,
            "session_id": session_id,
            "task_id": task_id,
            "auxiliary_graph_id": auxiliary_graph_id,
            "goal_id": goal_id,
            "producer_auxiliary_node": producer_auxiliary_node.model_dump(mode="json"),
            "producer_work_run_id": producer_work_run_id,
            "producer_attempt_id": producer_attempt_id,
            "producer_tool_result_ids": list(producer_tool_result_ids),
            "producer_primitive_call_id": producer_primitive_call_id,
            "scope_snapshot_sha256": scope_snapshot_sha256,
            "facts": [item.model_dump(mode="json") for item in facts],
            "constraints": [item.model_dump(mode="json") for item in constraints],
            "conflicts": [item.model_dump(mode="json") for item in conflicts],
            "gaps": [item.model_dump(mode="json") for item in gaps],
            "evidence_refs": [item.model_dump(mode="json") for item in evidence_refs],
            "freshness_manifest_sha256": freshness_manifest_sha256,
            "verification_receipt_id": verification_receipt_id,
            "verification_receipt_sha256": verification_receipt_sha256,
        }
    )


# ---------------------------------------------------------------------------
# 目标范围的规划 episode 预算
# ---------------------------------------------------------------------------


class PlanningEpisodeBudgetProfile(_Contract):
    """冻结的安全熔断限制；默认值为 083 中的生产 profile。"""

    profile_id: str = Field(
        default="planning_episode_production_v1",
        pattern=_DURABLE_ID_PATTERN,
    )
    rate_card_version: str = Field(
        default="planning_rate_card_v1",
        pattern=_DURABLE_ID_PATTERN,
    )
    hard_current_graph_nodes: int = Field(default=64, ge=1, le=512)
    hard_current_graph_depth: int = Field(default=12, ge=1, le=64)
    soft_auxiliary_graph_revisions: int = Field(default=8, ge=1, le=10_000)
    hard_auxiliary_graph_revisions: int = Field(default=12, ge=1, le=10_000)
    soft_distinct_auxiliary_nodes: int = Field(default=96, ge=1, le=100_000)
    hard_distinct_auxiliary_nodes: int = Field(default=128, ge=1, le=100_000)
    soft_work_runs: int = Field(default=48, ge=1, le=100_000)
    hard_work_runs: int = Field(default=64, ge=1, le=100_000)
    soft_attempts: int = Field(default=192, ge=1, le=1_000_000)
    hard_attempts: int = Field(default=256, ge=1, le=1_000_000)
    soft_logical_tool_calls: int = Field(default=192, ge=1, le=1_000_000)
    hard_logical_tool_calls: int = Field(default=256, ge=1, le=1_000_000)
    soft_logical_model_calls: int = Field(default=288, ge=1, le=1_000_000)
    hard_logical_model_calls: int = Field(default=384, ge=1, le=1_000_000)
    soft_physical_provider_tries: int = Field(default=576, ge=1, le=2_000_000)
    hard_physical_provider_tries: int = Field(default=768, ge=1, le=2_000_000)
    soft_autonomous_turns: int = Field(default=6, ge=1, le=10_000)
    hard_autonomous_turns: int = Field(default=8, ge=1, le=10_000)
    soft_model_input_tokens: int = Field(default=1_500_000, ge=1, le=2_000_000_000)
    hard_model_input_tokens: int = Field(default=2_000_000, ge=1, le=2_000_000_000)
    soft_model_output_tokens: int = Field(default=144_000, ge=1, le=2_000_000_000)
    hard_model_output_tokens: int = Field(default=192_000, ge=1, le=2_000_000_000)
    soft_injected_context_tokens: int = Field(default=288_000, ge=1, le=2_000_000_000)
    hard_injected_context_tokens: int = Field(default=384_000, ge=1, le=2_000_000_000)
    soft_evidence_units: int = Field(default=384, ge=1, le=10_000_000)
    hard_evidence_units: int = Field(default=512, ge=1, le=10_000_000)
    # 此处计算外部视觉*读取*次数，而非私有 authority 可绑定多少视觉别名。保持成本 guard
    # 独立于移除旧 64 别名投影截断的文档大小修复。
    soft_visual_units: int = Field(default=48, ge=1, le=1_000_000)
    hard_visual_units: int = Field(default=64, ge=1, le=1_000_000)
    soft_active_seconds: float = Field(default=2_700, ge=0, le=31_536_000)
    hard_active_seconds: float = Field(default=3_600, ge=0, le=31_536_000)
    soft_provider_cost_microusd: int = Field(default=6_000_000, ge=0, le=10**15)
    hard_provider_cost_microusd: int = Field(default=10_000_000, ge=0, le=10**15)

    @field_validator("soft_active_seconds", "hard_active_seconds")
    @classmethod
    def _validate_finite_seconds(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("planning active-time limits must be finite")
        return value

    @model_validator(mode="after")
    def _validate_soft_hard_pairs(self) -> "PlanningEpisodeBudgetProfile":
        for stem in _PLANNING_BUDGET_LIMIT_STEMS:
            if getattr(self, f"soft_{stem}") > getattr(self, f"hard_{stem}"):
                raise ValueError(f"soft {stem} limit cannot exceed its hard limit")
        return self


_PLANNING_BUDGET_LIMIT_STEMS = (
    "auxiliary_graph_revisions",
    "distinct_auxiliary_nodes",
    "work_runs",
    "attempts",
    "logical_tool_calls",
    "logical_model_calls",
    "physical_provider_tries",
    "autonomous_turns",
    "model_input_tokens",
    "model_output_tokens",
    "injected_context_tokens",
    "evidence_units",
    "visual_units",
    "active_seconds",
    "provider_cost_microusd",
)


class PlanningEpisodeBudgetUsage(_Contract):
    current_graph_nodes: int = Field(default=0, ge=0, le=10_000_000)
    current_graph_depth: int = Field(default=0, ge=0, le=10_000_000)
    auxiliary_graph_revisions: int = Field(default=0, ge=0, le=10_000_000)
    distinct_auxiliary_nodes: int = Field(default=0, ge=0, le=10_000_000)
    work_runs: int = Field(default=0, ge=0, le=10_000_000)
    attempts: int = Field(default=0, ge=0, le=10_000_000)
    logical_tool_calls: int = Field(default=0, ge=0, le=10_000_000)
    logical_model_calls: int = Field(default=0, ge=0, le=10_000_000)
    physical_provider_tries: int = Field(default=0, ge=0, le=20_000_000)
    autonomous_turns: int = Field(default=0, ge=0, le=10_000_000)
    model_input_tokens: int = Field(default=0, ge=0, le=10**12)
    model_output_tokens: int = Field(default=0, ge=0, le=10**12)
    injected_context_tokens: int = Field(default=0, ge=0, le=10**12)
    evidence_units: int = Field(default=0, ge=0, le=100_000_000)
    visual_units: int = Field(default=0, ge=0, le=10_000_000)
    active_seconds: float = Field(default=0, ge=0, le=31_536_000)
    provider_cost_microusd: int = Field(default=0, ge=0, le=10**15)

    @field_validator("active_seconds")
    @classmethod
    def _validate_finite_seconds(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("planning active-time usage must be finite")
        return value


class PlanningEpisodeBudgetDisposition(StrEnum):
    WITHIN_LIMIT = "within_limit"
    SOFT_LIMIT_REACHED = "soft_limit_reached"
    HARD_LIMIT_REACHED = "hard_limit_reached"


class PlanningEpisodeBudgetAssessment(_Contract):
    disposition: PlanningEpisodeBudgetDisposition
    soft_dimensions: tuple[str, ...] = Field(default=(), max_length=32)
    hard_dimensions: tuple[str, ...] = Field(default=(), max_length=32)

    @field_validator("soft_dimensions", "hard_dimensions")
    @classmethod
    def _validate_dimensions(cls, values: tuple[str, ...], info: object) -> tuple[str, ...]:
        return _require_canonical_unique(
            values,
            field_name=str(getattr(info, "field_name")),
        )

    @model_validator(mode="after")
    def _validate_disposition(self) -> "PlanningEpisodeBudgetAssessment":
        expected = (
            PlanningEpisodeBudgetDisposition.HARD_LIMIT_REACHED
            if self.hard_dimensions
            else PlanningEpisodeBudgetDisposition.SOFT_LIMIT_REACHED
            if self.soft_dimensions
            else PlanningEpisodeBudgetDisposition.WITHIN_LIMIT
        )
        if self.disposition is not expected:
            raise ValueError("budget disposition must be derived from exceeded dimensions")
        return self


class PlanningEpisodeBudgetExtensionReceipt(_Contract):
    """只追加的已授权增量；刻意不包含消耗量。"""

    schema_version: Literal["planning-episode-budget-extension-receipt-v1"] = (
        "planning-episode-budget-extension-receipt-v1"
    )
    extension_receipt_id: str = Field(pattern=_DURABLE_ID_PATTERN)
    goal_id: str = Field(pattern=_DURABLE_ID_PATTERN)
    approved_turn_id: str = Field(pattern=_DURABLE_ID_PATTERN)
    authorization_anchor_id: str = Field(pattern=_DURABLE_ID_PATTERN)
    reason: str = Field(min_length=1, max_length=2_000)
    profile_before: PlanningEpisodeBudgetProfile
    profile_after: PlanningEpisodeBudgetProfile
    previous_extension_receipt_sha256: str | None = Field(
        default=None,
        pattern=_SHA256_PATTERN,
    )
    receipt_sha256: str = Field(pattern=_SHA256_PATTERN)

    @field_validator("reason")
    @classmethod
    def _validate_reason(cls, value: str) -> str:
        return _require_canonical_text(value, field_name="budget extension reason")

    @model_validator(mode="after")
    def _validate_extension(self) -> "PlanningEpisodeBudgetExtensionReceipt":
        if self.profile_before.profile_id != self.profile_after.profile_id:
            raise ValueError("a budget extension cannot replace its profile identity")
        if self.profile_before.rate_card_version != self.profile_after.rate_card_version:
            raise ValueError("a budget extension cannot replace the frozen rate card")
        before = self.profile_before.model_dump(mode="python")
        after = self.profile_after.model_dump(mode="python")
        limit_names = (
            "hard_current_graph_nodes",
            "hard_current_graph_depth",
            *(
                f"{level}_{stem}"
                for stem in _PLANNING_BUDGET_LIMIT_STEMS
                for level in ("soft", "hard")
            ),
        )
        if any(after[name] < before[name] for name in limit_names):
            raise ValueError("a budget extension cannot reduce an existing allowance")
        if all(after[name] == before[name] for name in limit_names):
            raise ValueError("a budget extension must increase at least one allowance")
        expected = canonical_planning_episode_budget_extension_sha256(
            extension_receipt_id=self.extension_receipt_id,
            goal_id=self.goal_id,
            approved_turn_id=self.approved_turn_id,
            authorization_anchor_id=self.authorization_anchor_id,
            reason=self.reason,
            profile_before=self.profile_before,
            profile_after=self.profile_after,
            previous_extension_receipt_sha256=self.previous_extension_receipt_sha256,
        )
        if self.receipt_sha256 != expected:
            raise ValueError("budget extension receipt hash does not match its payload")
        return self

    @classmethod
    def create(cls, **values: object) -> Self:
        values = dict(values)
        profile_before = values["profile_before"]
        if not isinstance(profile_before, PlanningEpisodeBudgetProfile):
            profile_before = PlanningEpisodeBudgetProfile.model_validate(profile_before)
        profile_after = values["profile_after"]
        if not isinstance(profile_after, PlanningEpisodeBudgetProfile):
            profile_after = PlanningEpisodeBudgetProfile.model_validate(profile_after)
        values["profile_before"] = profile_before
        values["profile_after"] = profile_after
        values["receipt_sha256"] = canonical_planning_episode_budget_extension_sha256(
            extension_receipt_id=str(values["extension_receipt_id"]),
            goal_id=str(values["goal_id"]),
            approved_turn_id=str(values["approved_turn_id"]),
            authorization_anchor_id=str(values["authorization_anchor_id"]),
            reason=str(values["reason"]),
            profile_before=profile_before,
            profile_after=profile_after,
            previous_extension_receipt_sha256=values.get(  # type: ignore[arg-type]
                "previous_extension_receipt_sha256"
            ),
        )
        return cls.model_validate(values)


def canonical_planning_episode_budget_extension_sha256(
    *,
    extension_receipt_id: str,
    goal_id: str,
    approved_turn_id: str,
    authorization_anchor_id: str,
    reason: str,
    profile_before: PlanningEpisodeBudgetProfile,
    profile_after: PlanningEpisodeBudgetProfile,
    previous_extension_receipt_sha256: str | None,
) -> str:
    return _canonical_sha256(
        {
            "schema_version": "planning-episode-budget-extension-receipt-v1",
            "extension_receipt_id": extension_receipt_id,
            "goal_id": goal_id,
            "approved_turn_id": approved_turn_id,
            "authorization_anchor_id": authorization_anchor_id,
            "reason": reason,
            "profile_before": profile_before.model_dump(mode="json"),
            "profile_after": profile_after.model_dump(mode="json"),
            "previous_extension_receipt_sha256": previous_extension_receipt_sha256,
        }
    )


class PlanningEpisodeBudget(_Contract):
    """可自认证的累计目标预算快照。"""

    schema_version: Literal["planning-episode-budget-v1"] = (
        "planning-episode-budget-v1"
    )
    budget_ledger_id: str = Field(pattern=_DURABLE_ID_PATTERN)
    goal_id: str = Field(pattern=_DURABLE_ID_PATTERN)
    base_profile: PlanningEpisodeBudgetProfile = Field(
        default_factory=PlanningEpisodeBudgetProfile
    )
    usage: PlanningEpisodeBudgetUsage = Field(
        default_factory=PlanningEpisodeBudgetUsage
    )
    extensions: tuple[PlanningEpisodeBudgetExtensionReceipt, ...] = Field(
        default=(),
        max_length=128,
    )
    state_version: int = Field(default=1, ge=1)
    snapshot_sha256: str = Field(pattern=_SHA256_PATTERN)

    @property
    def effective_profile(self) -> PlanningEpisodeBudgetProfile:
        return self.extensions[-1].profile_after if self.extensions else self.base_profile

    @property
    def assessment(self) -> PlanningEpisodeBudgetAssessment:
        return assess_planning_episode_budget(
            profile=self.effective_profile,
            usage=self.usage,
        )

    @model_validator(mode="after")
    def _validate_budget(self) -> "PlanningEpisodeBudget":
        expected_profile = self.base_profile
        previous_hash: str | None = None
        seen_receipts: set[str] = set()
        for extension in self.extensions:
            if extension.extension_receipt_id in seen_receipts:
                raise ValueError("budget extension receipt IDs must be unique")
            seen_receipts.add(extension.extension_receipt_id)
            if extension.goal_id != self.goal_id:
                raise ValueError("budget extension must belong to its goal")
            if extension.profile_before != expected_profile:
                raise ValueError("budget extensions must form a contiguous profile chain")
            if extension.previous_extension_receipt_sha256 != previous_hash:
                raise ValueError("budget extension hash chain is discontinuous")
            expected_profile = extension.profile_after
            previous_hash = extension.receipt_sha256
        expected = canonical_planning_episode_budget_sha256(
            budget_ledger_id=self.budget_ledger_id,
            goal_id=self.goal_id,
            base_profile=self.base_profile,
            usage=self.usage,
            extensions=self.extensions,
            state_version=self.state_version,
        )
        if self.snapshot_sha256 != expected:
            raise ValueError("planning budget hash does not match its payload")
        return self

    @classmethod
    def create(cls, **values: object) -> Self:
        values = dict(values)
        base_profile = values.get("base_profile", PlanningEpisodeBudgetProfile())
        if not isinstance(base_profile, PlanningEpisodeBudgetProfile):
            base_profile = PlanningEpisodeBudgetProfile.model_validate(base_profile)
        usage = values.get("usage", PlanningEpisodeBudgetUsage())
        if not isinstance(usage, PlanningEpisodeBudgetUsage):
            usage = PlanningEpisodeBudgetUsage.model_validate(usage)
        extensions = tuple(
            item
            if isinstance(item, PlanningEpisodeBudgetExtensionReceipt)
            else PlanningEpisodeBudgetExtensionReceipt.model_validate(item)
            for item in values.get("extensions", ())  # type: ignore[union-attr]
        )
        state_version = int(values.get("state_version", 1))
        values["base_profile"] = base_profile
        values["usage"] = usage
        values["extensions"] = extensions
        values["state_version"] = state_version
        values["snapshot_sha256"] = canonical_planning_episode_budget_sha256(
            budget_ledger_id=str(values["budget_ledger_id"]),
            goal_id=str(values["goal_id"]),
            base_profile=base_profile,  # type: ignore[arg-type]
            usage=usage,  # type: ignore[arg-type]
            extensions=extensions,
            state_version=state_version,
        )
        return cls.model_validate(values)


def canonical_planning_episode_budget_sha256(
    *,
    budget_ledger_id: str,
    goal_id: str,
    base_profile: PlanningEpisodeBudgetProfile,
    usage: PlanningEpisodeBudgetUsage,
    extensions: tuple[PlanningEpisodeBudgetExtensionReceipt, ...],
    state_version: int,
) -> str:
    return _canonical_sha256(
        {
            "schema_version": "planning-episode-budget-v1",
            "budget_ledger_id": budget_ledger_id,
            "goal_id": goal_id,
            "base_profile": base_profile.model_dump(mode="json"),
            "usage": usage.model_dump(mode="json"),
            "extensions": [item.model_dump(mode="json") for item in extensions],
            "state_version": state_version,
        }
    )


def assess_planning_episode_budget(
    *,
    profile: PlanningEpisodeBudgetProfile,
    usage: PlanningEpisodeBudgetUsage,
) -> PlanningEpisodeBudgetAssessment:
    hard: set[str] = set()
    soft: set[str] = set()
    if usage.current_graph_nodes > profile.hard_current_graph_nodes:
        hard.add("current_graph_nodes")
    if usage.current_graph_depth > profile.hard_current_graph_depth:
        hard.add("current_graph_depth")
    for stem in _PLANNING_BUDGET_LIMIT_STEMS:
        consumed = getattr(usage, stem)
        if consumed >= getattr(profile, f"hard_{stem}"):
            hard.add(stem)
        elif consumed >= getattr(profile, f"soft_{stem}"):
            soft.add(stem)
    hard_dimensions = tuple(sorted(hard))
    soft_dimensions = tuple(sorted(soft))
    disposition = (
        PlanningEpisodeBudgetDisposition.HARD_LIMIT_REACHED
        if hard_dimensions
        else PlanningEpisodeBudgetDisposition.SOFT_LIMIT_REACHED
        if soft_dimensions
        else PlanningEpisodeBudgetDisposition.WITHIN_LIMIT
    )
    return PlanningEpisodeBudgetAssessment(
        disposition=disposition,
        soft_dimensions=soft_dimensions,
        hard_dimensions=hard_dimensions,
    )


# ---------------------------------------------------------------------------
# 独立 TaskGraph 语义验证 DTO
# ---------------------------------------------------------------------------


class TaskGraphSemanticVerificationDimension(StrEnum):
    GOAL_COVERAGE = "goal_coverage"
    NODE_NECESSITY_AND_AUTHORITY = "node_necessity_and_authority"
    ACCEPTANCE_VERIFIABILITY = "acceptance_verifiability"
    EDGE_DEPENDENCY_VALIDITY = "edge_dependency_validity"
    EVIDENCE_GROUNDING = "evidence_grounding"
    GAP_DISPOSITION = "gap_disposition"
    SCOPE_AND_CONSTRAINT_FIDELITY = "scope_and_constraint_fidelity"
    BASE_AUTHORITY_PRESERVATION = "base_authority_preservation"


class TaskGraphSemanticVerificationVerdict(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class TaskGraphSemanticVerificationDisposition(StrEnum):
    PASS = "pass"
    REVISE = "revise"
    BLOCKED = "blocked"


class TaskGraphSemanticFailureScope(StrEnum):
    """一项未通过语义 finding 的 reviewer 归因。

    reviewer 标识缺失修正所属位置，但绝不直接选择 Runtime 转换。Host 将这些 scope 聚合为
    :class:`TaskGraphSemanticTerminalRoute`。
    """

    TERMINAL_PROPOSAL = "terminal_proposal"
    AUXILIARY_INVESTIGATION = "auxiliary_investigation"
    MISSING_INFORMATION = "missing_information"
    MISSING_AUTHORITY = "missing_authority"


class TaskGraphSemanticTerminalRoute(StrEnum):
    """一个语义评审 quorum 的下一边界，由 Host 派生。"""

    PASS = "pass"
    RETRY_TERMINAL_ATTEMPT = "retry_terminal_attempt"
    REPLAN_AUXILIARY = "replan_auxiliary"
    BLOCKED = "blocked"


# replan receipt 与上方图 revision 契约一同声明，并刻意前向引用此处声明的语义 disposition。
AuxiliaryReplanTriggerReceipt.model_rebuild()


class TaskGraphSemanticVerificationItem(_Contract):
    dimension: TaskGraphSemanticVerificationDimension
    verdict: TaskGraphSemanticVerificationVerdict
    failure_scope: TaskGraphSemanticFailureScope | None
    finding: str = Field(min_length=1, max_length=4_000)
    affected_node_keys: tuple[str, ...] = Field(default=(), max_length=128)
    evidence_aliases: tuple[str, ...] = Field(default=(), max_length=1_024)
    gap_aliases: tuple[str, ...] = Field(default=(), max_length=256)

    @field_validator("finding")
    @classmethod
    def _validate_finding(cls, value: str) -> str:
        return _require_canonical_text(value, field_name="semantic verification finding")

    @field_validator("affected_node_keys", "evidence_aliases", "gap_aliases")
    @classmethod
    def _validate_references(cls, values: tuple[str, ...], info: object) -> tuple[str, ...]:
        return _require_canonical_unique(
            values,
            field_name=str(getattr(info, "field_name")),
        )

    @model_validator(mode="after")
    def _validate_failure_scope(self) -> "TaskGraphSemanticVerificationItem":
        if (
            self.verdict is TaskGraphSemanticVerificationVerdict.PASS
            and self.failure_scope is not None
        ):
            raise ValueError("a passing semantic finding cannot have a failure scope")
        if (
            self.verdict is not TaskGraphSemanticVerificationVerdict.PASS
            and self.failure_scope is None
        ):
            raise ValueError("a non-passing semantic finding requires a failure scope")
        if (
            self.verdict
            is TaskGraphSemanticVerificationVerdict.INSUFFICIENT_EVIDENCE
            and not self.gap_aliases
        ):
            raise ValueError("insufficient-evidence verdict requires a typed gap")
        if (
            self.verdict
            is TaskGraphSemanticVerificationVerdict.INSUFFICIENT_EVIDENCE
            and self.failure_scope
            not in {
                TaskGraphSemanticFailureScope.MISSING_INFORMATION,
                TaskGraphSemanticFailureScope.MISSING_AUTHORITY,
            }
        ):
            raise ValueError(
                "insufficient-evidence findings require missing-information or "
                "missing-authority scope"
            )
        if (
            self.failure_scope
            in {
                TaskGraphSemanticFailureScope.MISSING_INFORMATION,
                TaskGraphSemanticFailureScope.MISSING_AUTHORITY,
            }
            and not self.gap_aliases
        ):
            raise ValueError("missing-information/authority scope requires a typed gap")
        return self


class PlanningGoalConstraintPrompt(_Contract):
    constraint_id: str = Field(pattern=_LOCAL_KEY_PATTERN)
    statement: str = Field(min_length=1, max_length=4_000)
    authorization_aliases: tuple[str, ...] = Field(min_length=1, max_length=32)

    @field_validator("statement")
    @classmethod
    def _validate_statement(cls, value: str) -> str:
        return _require_canonical_text(value, field_name="goal constraint")

    @field_validator("authorization_aliases")
    @classmethod
    def _validate_aliases(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        return _require_canonical_unique(values, field_name="authorization aliases")


class PlanningGoalPromptContext(_Contract):
    """prompt 安全用户目标及其显式授权绑定。"""

    goal_id: str = Field(pattern=_DURABLE_ID_PATTERN)
    objective: str = Field(min_length=1, max_length=8_000)
    desired_output: str = Field(min_length=1, max_length=4_000)
    constraints: tuple[PlanningGoalConstraintPrompt, ...] = Field(
        default=(),
        max_length=64,
    )
    authorization_aliases: tuple[str, ...] = Field(min_length=1, max_length=128)

    @field_validator("objective", "desired_output")
    @classmethod
    def _validate_text(cls, value: str, info: object) -> str:
        return _require_canonical_text(value, field_name=str(getattr(info, "field_name")))

    @field_validator("authorization_aliases")
    @classmethod
    def _validate_aliases(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        return _require_canonical_unique(values, field_name="goal authorization aliases")

    @model_validator(mode="after")
    def _validate_constraints(self) -> "PlanningGoalPromptContext":
        _require_unique(
            tuple(item.constraint_id for item in self.constraints),
            field_name="goal constraint IDs",
        )
        allowed = set(self.authorization_aliases)
        if any(
            not set(item.authorization_aliases).issubset(allowed)
            for item in self.constraints
        ):
            raise ValueError("goal constraints must use declared authorization aliases")
        return self


class PlanningAuthoritySourceKind(StrEnum):
    USER_INSTRUCTION = "user_instruction"
    USER_ANSWER = "user_answer"
    PRIOR_TASK_STATE = "prior_task_state"
    DOCUMENT = "document"
    WORKSPACE = "workspace"
    TOOL_OBSERVATION = "tool_observation"
    VISUAL = "visual"
    MEMORY = "memory"
    ARTIFACT = "artifact"
    GAP = "gap"


class PlanningAuthoritySourceCard(_Contract):
    """不含私有 locator 或来源身份的 prompt 安全源卡片。"""

    alias: str = Field(pattern=_LOCAL_KEY_PATTERN)
    authority_class: PlanningAuthorityClass
    source_kind: PlanningAuthoritySourceKind
    document_group_alias: str | None = Field(
        default=None,
        pattern=_LOCAL_KEY_PATTERN,
        exclude_if=lambda value: value is None,
        description=(
            "Prompt-safe Host identity shared by source cards from one document."
        ),
    )
    source_label: str = Field(min_length=1, max_length=240)
    excerpt: str = Field(min_length=1, max_length=4_000)
    projection_sha256: str = Field(pattern=_SHA256_PATTERN)

    @field_validator("source_label", "excerpt")
    @classmethod
    def _validate_text(cls, value: str, info: object) -> str:
        return _require_canonical_text(value, field_name=str(getattr(info, "field_name")))

    @model_validator(mode="after")
    def _validate_source_class(self) -> "PlanningAuthoritySourceCard":
        authorization_kinds = {
            PlanningAuthoritySourceKind.USER_INSTRUCTION,
            PlanningAuthoritySourceKind.USER_ANSWER,
            PlanningAuthoritySourceKind.PRIOR_TASK_STATE,
        }
        if self.authority_class is PlanningAuthorityClass.AUTHORIZATION:
            if self.source_kind not in authorization_kinds:
                raise ValueError("prompt authorization cards require an authorized source kind")
        elif self.source_kind in authorization_kinds:
            raise ValueError("authorization source cards cannot be projected as evidence")
        if (self.authority_class is PlanningAuthorityClass.GAP) != (
            self.source_kind is PlanningAuthoritySourceKind.GAP
        ):
            raise ValueError("gap source cards require the gap authority class")
        if (
            self.document_group_alias is not None
            and self.source_kind is not PlanningAuthoritySourceKind.DOCUMENT
        ):
            raise ValueError("document group aliases require document source cards")
        return self


class PlanningAuthorityProjection(_Contract):
    schema_version: Literal["planning-authority-projection-v1"] = (
        "planning-authority-projection-v1"
    )
    authority_snapshot_id: str = Field(pattern=_DURABLE_ID_PATTERN)
    authority_snapshot_sha256: str = Field(pattern=_SHA256_PATTERN)
    cards: tuple[PlanningAuthoritySourceCard, ...] = Field(
        min_length=1,
    )
    projection_sha256: str = Field(pattern=_SHA256_PATTERN)

    @model_validator(mode="after")
    def _validate_projection(self) -> "PlanningAuthorityProjection":
        aliases = tuple(item.alias for item in self.cards)
        _require_unique(aliases, field_name="authority card aliases")
        if aliases != tuple(sorted(aliases)):
            raise ValueError("authority cards must use alias order")
        if not any(
            item.authority_class is PlanningAuthorityClass.AUTHORIZATION
            for item in self.cards
        ):
            raise ValueError("authority projection requires authorization context")
        expected = canonical_planning_authority_projection_sha256(
            authority_snapshot_id=self.authority_snapshot_id,
            authority_snapshot_sha256=self.authority_snapshot_sha256,
            cards=self.cards,
        )
        if self.projection_sha256 != expected:
            raise ValueError("authority projection hash does not match its cards")
        return self

    @classmethod
    def create(cls, **values: object) -> Self:
        values = dict(values)
        cards = tuple(
            item
            if isinstance(item, PlanningAuthoritySourceCard)
            else PlanningAuthoritySourceCard.model_validate(item)
            for item in values["cards"]  # type: ignore[union-attr]
        )
        values["cards"] = cards
        values["projection_sha256"] = canonical_planning_authority_projection_sha256(
            authority_snapshot_id=str(values["authority_snapshot_id"]),
            authority_snapshot_sha256=str(values["authority_snapshot_sha256"]),
            cards=cards,
        )
        return cls.model_validate(values)


def canonical_planning_authority_projection_sha256(
    *,
    authority_snapshot_id: str,
    authority_snapshot_sha256: str,
    cards: tuple[PlanningAuthoritySourceCard, ...],
) -> str:
    return _canonical_sha256(
        {
            "schema_version": "planning-authority-projection-v1",
            "authority_snapshot_id": authority_snapshot_id,
            "authority_snapshot_sha256": authority_snapshot_sha256,
            "cards": [item.model_dump(mode="json") for item in cards],
        }
    )


class PlanningContextFactProjection(_Contract):
    fact_alias: str = Field(pattern=_LOCAL_KEY_PATTERN)
    statement: str = Field(min_length=1, max_length=4_000)
    evidence_aliases: tuple[str, ...] = Field(min_length=1, max_length=32)

    @field_validator("statement")
    @classmethod
    def _validate_statement(cls, value: str) -> str:
        return _require_canonical_text(value, field_name="projected fact")

    @field_validator("evidence_aliases")
    @classmethod
    def _validate_aliases(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        return _require_canonical_unique(values, field_name="fact evidence aliases")


class PlanningContextConstraintProjection(_Contract):
    constraint_alias: str = Field(pattern=_LOCAL_KEY_PATTERN)
    statement: str = Field(min_length=1, max_length=4_000)
    authorization_aliases: tuple[str, ...] = Field(min_length=1, max_length=32)

    @field_validator("statement")
    @classmethod
    def _validate_statement(cls, value: str) -> str:
        return _require_canonical_text(value, field_name="projected constraint")

    @field_validator("authorization_aliases")
    @classmethod
    def _validate_aliases(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        return _require_canonical_unique(values, field_name="constraint authorization aliases")


class PlanningContextConflictProjection(_Contract):
    conflict_alias: str = Field(pattern=_LOCAL_KEY_PATTERN)
    statement: str = Field(min_length=1, max_length=4_000)
    evidence_aliases: tuple[str, ...] = Field(min_length=2, max_length=32)
    blocking: bool

    @field_validator("statement")
    @classmethod
    def _validate_statement(cls, value: str) -> str:
        return _require_canonical_text(value, field_name="projected conflict")

    @field_validator("evidence_aliases")
    @classmethod
    def _validate_aliases(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        return _require_canonical_unique(values, field_name="conflict evidence aliases")


class PlanningContextGapProjection(_Contract):
    gap_alias: str = Field(pattern=_LOCAL_KEY_PATTERN)
    observation_status: PlanningObservationStatus
    description: str = Field(min_length=1, max_length=4_000)
    blocking: bool
    affected_obligations: tuple[str, ...] = Field(min_length=1, max_length=64)
    evidence_aliases: tuple[str, ...] = Field(default=(), max_length=32)
    resolution_hint: str | None = Field(default=None, min_length=1, max_length=2_000)

    @field_validator("description", "resolution_hint")
    @classmethod
    def _validate_text(cls, value: str | None, info: object) -> str | None:
        if value is None:
            return None
        return _require_canonical_text(value, field_name=str(getattr(info, "field_name")))

    @field_validator("affected_obligations", "evidence_aliases")
    @classmethod
    def _validate_references(cls, values: tuple[str, ...], info: object) -> tuple[str, ...]:
        return _require_canonical_unique(
            values,
            field_name=str(getattr(info, "field_name")),
        )

    @model_validator(mode="after")
    def _validate_status(self) -> "PlanningContextGapProjection":
        if self.observation_status is PlanningObservationStatus.SUCCESS:
            raise ValueError("a projected gap cannot come from a successful observation")
        return self


class PlanningContextArtifactProjection(_Contract):
    """绑定到不可变私有 artifact 的 prompt 安全 ContextArtifact 视图。"""

    schema_version: Literal["planning-context-artifact-projection-v1"] = (
        "planning-context-artifact-projection-v1"
    )
    artifact_alias: str = Field(pattern=_LOCAL_KEY_PATTERN)
    artifact_id: str = Field(pattern=_DURABLE_ID_PATTERN)
    artifact_sha256: str = Field(pattern=_SHA256_PATTERN)
    producer_node_alias: str = Field(pattern=_LOCAL_KEY_PATTERN)
    facts: tuple[PlanningContextFactProjection, ...] = Field(default=(), max_length=256)
    constraints: tuple[PlanningContextConstraintProjection, ...] = Field(
        default=(), max_length=128
    )
    conflicts: tuple[PlanningContextConflictProjection, ...] = Field(
        default=(), max_length=128
    )
    gaps: tuple[PlanningContextGapProjection, ...] = Field(default=(), max_length=128)
    projection_sha256: str = Field(pattern=_SHA256_PATTERN)

    @model_validator(mode="after")
    def _validate_projection(self) -> "PlanningContextArtifactProjection":
        if not any((self.facts, self.constraints, self.conflicts, self.gaps)):
            raise ValueError("artifact projection requires a typed observation item")
        for values, label, attribute in (
            (self.facts, "projected fact aliases", "fact_alias"),
            (self.constraints, "projected constraint aliases", "constraint_alias"),
            (self.conflicts, "projected conflict aliases", "conflict_alias"),
            (self.gaps, "projected gap aliases", "gap_alias"),
        ):
            _require_unique(
                tuple(getattr(item, attribute) for item in values),
                field_name=label,
            )
        expected = canonical_planning_context_artifact_projection_sha256(
            artifact_alias=self.artifact_alias,
            artifact_id=self.artifact_id,
            artifact_sha256=self.artifact_sha256,
            producer_node_alias=self.producer_node_alias,
            facts=self.facts,
            constraints=self.constraints,
            conflicts=self.conflicts,
            gaps=self.gaps,
        )
        if self.projection_sha256 != expected:
            raise ValueError("artifact projection hash does not match its payload")
        return self

    @classmethod
    def create(cls, **values: object) -> Self:
        values = dict(values)
        facts = tuple(
            item
            if isinstance(item, PlanningContextFactProjection)
            else PlanningContextFactProjection.model_validate(item)
            for item in values.get("facts", ())  # type: ignore[union-attr]
        )
        constraints = tuple(
            item
            if isinstance(item, PlanningContextConstraintProjection)
            else PlanningContextConstraintProjection.model_validate(item)
            for item in values.get("constraints", ())  # type: ignore[union-attr]
        )
        conflicts = tuple(
            item
            if isinstance(item, PlanningContextConflictProjection)
            else PlanningContextConflictProjection.model_validate(item)
            for item in values.get("conflicts", ())  # type: ignore[union-attr]
        )
        gaps = tuple(
            item
            if isinstance(item, PlanningContextGapProjection)
            else PlanningContextGapProjection.model_validate(item)
            for item in values.get("gaps", ())  # type: ignore[union-attr]
        )
        values.update(
            facts=facts,
            constraints=constraints,
            conflicts=conflicts,
            gaps=gaps,
            projection_sha256=canonical_planning_context_artifact_projection_sha256(
                artifact_alias=str(values["artifact_alias"]),
                artifact_id=str(values["artifact_id"]),
                artifact_sha256=str(values["artifact_sha256"]),
                producer_node_alias=str(values["producer_node_alias"]),
                facts=facts,
                constraints=constraints,
                conflicts=conflicts,
                gaps=gaps,
            ),
        )
        return cls.model_validate(values)


def canonical_planning_context_artifact_projection_sha256(
    *,
    artifact_alias: str,
    artifact_id: str,
    artifact_sha256: str,
    producer_node_alias: str,
    facts: tuple[PlanningContextFactProjection, ...],
    constraints: tuple[PlanningContextConstraintProjection, ...],
    conflicts: tuple[PlanningContextConflictProjection, ...],
    gaps: tuple[PlanningContextGapProjection, ...],
) -> str:
    return _canonical_sha256(
        {
            "schema_version": "planning-context-artifact-projection-v1",
            "artifact_alias": artifact_alias,
            "artifact_id": artifact_id,
            "artifact_sha256": artifact_sha256,
            "producer_node_alias": producer_node_alias,
            "facts": [item.model_dump(mode="json") for item in facts],
            "constraints": [item.model_dump(mode="json") for item in constraints],
            "conflicts": [item.model_dump(mode="json") for item in conflicts],
            "gaps": [item.model_dump(mode="json") for item in gaps],
        }
    )


class PlanningContextPromptInputs(_Contract):
    """一个 ContextArtifact 及其 prompt-safe 来源卡片。"""

    source_cards: tuple[PlanningAuthoritySourceCard, ...]
    context_artifact: PlanningContextArtifactProjection

    @model_validator(mode="after")
    def _validate_prompt_inputs(self) -> "PlanningContextPromptInputs":
        aliases = tuple(item.alias for item in self.source_cards)
        if aliases != tuple(sorted(aliases)) or len(aliases) != len(set(aliases)):
            raise ValueError("source cards must use unique ascending aliases")
        cards = {item.alias: item for item in self.source_cards}
        evidence_aliases = {
            alias
            for item in (
                *self.context_artifact.facts,
                *self.context_artifact.conflicts,
                *self.context_artifact.gaps,
            )
            for alias in item.evidence_aliases
        }
        if not evidence_aliases.issubset(cards):
            raise ValueError("artifact projection references an absent evidence card")
        if any(
            cards[alias].authority_class is not PlanningAuthorityClass.EVIDENCE
            for alias in evidence_aliases
        ):
            raise ValueError("artifact evidence aliases require evidence authority")
        projected_gap_aliases = {
            item.gap_alias for item in self.context_artifact.gaps
        }
        gap_card_aliases = {
            alias
            for alias, card in cards.items()
            if card.authority_class is PlanningAuthorityClass.GAP
        }
        if projected_gap_aliases != gap_card_aliases:
            raise ValueError("gap cards must exactly cover projected typed gaps")
        return self


class TaskGraphSemanticDeliveryCoverage(StrEnum):
    FULL = "full"
    PARTIAL = "partial"


class TaskGraphSemanticBaseNode(_Contract):
    node_alias: str = Field(pattern=_LOCAL_KEY_PATTERN)
    node_revision: int = Field(ge=1)
    node_kind: InSessionTaskNodeKind
    parent_node_alias: str | None = Field(default=None, pattern=_LOCAL_KEY_PATTERN)
    title: str = Field(min_length=1, max_length=240)
    objective: str = Field(min_length=1, max_length=2_000)
    source_anchor_aliases: tuple[str, ...] = Field(min_length=1, max_length=64)
    acceptance_criteria: tuple[InSessionTaskAcceptanceProposal, ...] = Field(
        min_length=1, max_length=64
    )
    constraints: tuple[str, ...] = Field(default=(), max_length=32)
    status: InSessionTaskStatus
    completed_delivery_summary: str | None = Field(
        default=None, min_length=1, max_length=8_000
    )
    completed_delivery_coverage: TaskGraphSemanticDeliveryCoverage | None = None
    completed_delivery_gap_aliases: tuple[str, ...] = Field(default=(), max_length=64)
    delivery_authority_aliases: tuple[str, ...] = Field(default=(), max_length=64)

    @field_validator("title", "objective", "completed_delivery_summary")
    @classmethod
    def _validate_text(cls, value: str | None, info: object) -> str | None:
        if value is None:
            return None
        return _require_canonical_text(value, field_name=str(getattr(info, "field_name")))

    @field_validator(
        "source_anchor_aliases",
        "completed_delivery_gap_aliases",
        "delivery_authority_aliases",
    )
    @classmethod
    def _validate_aliases(cls, values: tuple[str, ...], info: object) -> tuple[str, ...]:
        return _require_canonical_unique(
            values,
            field_name=str(getattr(info, "field_name")),
        )

    @model_validator(mode="after")
    def _validate_node(self) -> "TaskGraphSemanticBaseNode":
        if self.node_kind is InSessionTaskNodeKind.ROOT and self.parent_node_alias is not None:
            raise ValueError("base root node cannot have a parent")
        if self.node_kind is InSessionTaskNodeKind.SUBTASK and self.parent_node_alias is None:
            raise ValueError("base subtask requires a parent")
        completed = self.status is InSessionTaskStatus.COMPLETED
        if completed != (self.completed_delivery_summary is not None):
            raise ValueError("completed base nodes require exactly one Delivery summary")
        if completed != (self.completed_delivery_coverage is not None):
            raise ValueError("completed base nodes require a Delivery coverage verdict")
        if not completed and (
            self.delivery_authority_aliases or self.completed_delivery_gap_aliases
        ):
            raise ValueError("only completed base nodes may carry Delivery authority")
        if self.completed_delivery_coverage is TaskGraphSemanticDeliveryCoverage.FULL:
            if self.completed_delivery_gap_aliases:
                raise ValueError("full Delivery coverage cannot carry gap aliases")
        elif self.completed_delivery_coverage is TaskGraphSemanticDeliveryCoverage.PARTIAL:
            if not self.completed_delivery_gap_aliases:
                raise ValueError("partial Delivery coverage requires a typed gap alias")
        return self


class TaskGraphSemanticBaseSnapshot(_Contract):
    schema_version: Literal["task-graph-semantic-base-snapshot-v1"] = (
        "task-graph-semantic-base-snapshot-v1"
    )
    base_task_graph_revision: int = Field(ge=1)
    root_node_alias: str = Field(pattern=_LOCAL_KEY_PATTERN)
    nodes: tuple[TaskGraphSemanticBaseNode, ...] = Field(min_length=1, max_length=512)
    source_snapshot_sha256: str = Field(pattern=_SHA256_PATTERN)
    projection_sha256: str = Field(pattern=_SHA256_PATTERN)

    @model_validator(mode="after")
    def _validate_snapshot(self) -> "TaskGraphSemanticBaseSnapshot":
        aliases = tuple(item.node_alias for item in self.nodes)
        _require_unique(aliases, field_name="base node aliases")
        roots = tuple(
            item.node_alias for item in self.nodes if item.node_kind is InSessionTaskNodeKind.ROOT
        )
        if roots != (self.root_node_alias,):
            raise ValueError("base snapshot requires exactly its declared root")
        known = set(aliases)
        if any(
            item.parent_node_alias is not None and item.parent_node_alias not in known
            for item in self.nodes
        ):
            raise ValueError("base snapshot node references an unknown parent")
        parent = {item.node_alias: item.parent_node_alias for item in self.nodes}
        for alias in aliases:
            seen: set[str] = set()
            current: str | None = alias
            while current is not None:
                if current in seen:
                    raise ValueError("base TaskGraph snapshot must be acyclic")
                seen.add(current)
                current = parent[current]
            if self.root_node_alias not in seen:
                raise ValueError("every base node must descend from the root")
        expected = canonical_task_graph_semantic_base_snapshot_sha256(
            base_task_graph_revision=self.base_task_graph_revision,
            root_node_alias=self.root_node_alias,
            nodes=self.nodes,
            source_snapshot_sha256=self.source_snapshot_sha256,
        )
        if self.projection_sha256 != expected:
            raise ValueError("base TaskGraph projection hash does not match its payload")
        return self

    @classmethod
    def create(cls, **values: object) -> Self:
        values = dict(values)
        nodes = tuple(
            item
            if isinstance(item, TaskGraphSemanticBaseNode)
            else TaskGraphSemanticBaseNode.model_validate(item)
            for item in values["nodes"]  # type: ignore[union-attr]
        )
        values["nodes"] = nodes
        values["projection_sha256"] = canonical_task_graph_semantic_base_snapshot_sha256(
            base_task_graph_revision=int(values["base_task_graph_revision"]),
            root_node_alias=str(values["root_node_alias"]),
            nodes=nodes,
            source_snapshot_sha256=str(values["source_snapshot_sha256"]),
        )
        return cls.model_validate(values)


def canonical_task_graph_semantic_base_snapshot_sha256(
    *,
    base_task_graph_revision: int,
    root_node_alias: str,
    nodes: tuple[TaskGraphSemanticBaseNode, ...],
    source_snapshot_sha256: str,
) -> str:
    return _canonical_sha256(
        {
            "schema_version": "task-graph-semantic-base-snapshot-v1",
            "base_task_graph_revision": base_task_graph_revision,
            "root_node_alias": root_node_alias,
            "nodes": [item.model_dump(mode="json") for item in nodes],
            "source_snapshot_sha256": source_snapshot_sha256,
        }
    )


class PlanningCapabilityEffect(StrEnum):
    READ_ONLY = "read_only"
    PROTECTED = "protected"


class PlanningCapabilityDescriptor(_Contract):
    capability_alias: str = Field(pattern=_LOCAL_KEY_PATTERN)
    label: str = Field(min_length=1, max_length=240)
    description: str = Field(min_length=1, max_length=2_000)
    available: bool
    effect: PlanningCapabilityEffect
    supported_operations: tuple[str, ...] = Field(min_length=1, max_length=64)
    supported_resource_kinds: tuple[str, ...] = Field(default=(), max_length=64)
    limitations: tuple[str, ...] = Field(default=(), max_length=32)

    @field_validator("label", "description")
    @classmethod
    def _validate_text(cls, value: str, info: object) -> str:
        return _require_canonical_text(value, field_name=str(getattr(info, "field_name")))

    @field_validator("supported_operations", "supported_resource_kinds", "limitations")
    @classmethod
    def _validate_values(cls, values: tuple[str, ...], info: object) -> tuple[str, ...]:
        if any(not value or value != value.strip() or len(value) > 500 for value in values):
            raise ValueError(f"{getattr(info, 'field_name')} contain invalid text")
        return _require_canonical_unique(
            values,
            field_name=str(getattr(info, "field_name")),
        )


class PlanningCapabilityCatalogProjection(_Contract):
    schema_version: Literal["planning-capability-catalog-projection-v1"] = (
        "planning-capability-catalog-projection-v1"
    )
    capability_catalog_snapshot_id: str = Field(pattern=_DURABLE_ID_PATTERN)
    capability_catalog_snapshot_sha256: str = Field(pattern=_SHA256_PATTERN)
    capabilities: tuple[PlanningCapabilityDescriptor, ...] = Field(
        min_length=1, max_length=256
    )
    projection_sha256: str = Field(pattern=_SHA256_PATTERN)

    @model_validator(mode="after")
    def _validate_catalog(self) -> "PlanningCapabilityCatalogProjection":
        aliases = tuple(item.capability_alias for item in self.capabilities)
        _require_unique(aliases, field_name="capability aliases")
        if aliases != tuple(sorted(aliases)):
            raise ValueError("capabilities must use alias order")
        expected = canonical_planning_capability_catalog_projection_sha256(
            capability_catalog_snapshot_id=self.capability_catalog_snapshot_id,
            capability_catalog_snapshot_sha256=self.capability_catalog_snapshot_sha256,
            capabilities=self.capabilities,
        )
        if self.projection_sha256 != expected:
            raise ValueError("capability catalog projection hash does not match its payload")
        return self

    @classmethod
    def create(cls, **values: object) -> Self:
        values = dict(values)
        capabilities = tuple(
            item
            if isinstance(item, PlanningCapabilityDescriptor)
            else PlanningCapabilityDescriptor.model_validate(item)
            for item in values["capabilities"]  # type: ignore[union-attr]
        )
        values["capabilities"] = capabilities
        values["projection_sha256"] = canonical_planning_capability_catalog_projection_sha256(
            capability_catalog_snapshot_id=str(values["capability_catalog_snapshot_id"]),
            capability_catalog_snapshot_sha256=str(values["capability_catalog_snapshot_sha256"]),
            capabilities=capabilities,
        )
        return cls.model_validate(values)


def canonical_planning_capability_catalog_projection_sha256(
    *,
    capability_catalog_snapshot_id: str,
    capability_catalog_snapshot_sha256: str,
    capabilities: tuple[PlanningCapabilityDescriptor, ...],
) -> str:
    return _canonical_sha256(
        {
            "schema_version": "planning-capability-catalog-projection-v1",
            "capability_catalog_snapshot_id": capability_catalog_snapshot_id,
            "capability_catalog_snapshot_sha256": capability_catalog_snapshot_sha256,
            "capabilities": [item.model_dump(mode="json") for item in capabilities],
        }
    )


class TaskGraphSemanticLineageDisposition(StrEnum):
    NEW = "new"
    REUSE = "reuse"
    REVISE = "revise"


class TaskGraphSemanticLineageProjection(_Contract):
    """prompt 安全 candidate-to-base lineage 提示；两端均为不透明别名。"""

    proposal_node_key: str = Field(pattern=_LOCAL_KEY_PATTERN)
    disposition: TaskGraphSemanticLineageDisposition
    base_node_alias: str | None = Field(default=None, pattern=_LOCAL_KEY_PATTERN)

    @model_validator(mode="after")
    def _validate_lineage(self) -> "TaskGraphSemanticLineageProjection":
        if self.disposition is TaskGraphSemanticLineageDisposition.NEW:
            if self.base_node_alias is not None:
                raise ValueError("new lineage cannot select a base node")
        elif self.base_node_alias is None:
            raise ValueError("reuse/revise lineage requires an opaque base node alias")
        return self


class TaskGraphRevisionCandidate(_Contract):
    """一份完整 positive-base TaskGraph 提案及其有序 lineage。

    模型只选择 prompt 安全 base-node 别名。持久节点身份和 carry 资格仍由 Host 所有。
    candidate lineage 被刻意设为完整且带位置，使下游语义评审不会意外根据部分或重排映射
    评审提案。
    """

    schema_version: Literal["task-graph-revision-candidate-v1"] = (
        "task-graph-revision-candidate-v1"
    )
    proposal: InSessionTaskGraphRevisionProposal
    lineage: tuple[TaskGraphSemanticLineageProjection, ...] = Field(
        min_length=1,
        max_length=512,
    )

    @model_validator(mode="after")
    def _validate_candidate(self) -> "TaskGraphRevisionCandidate":
        proposal_node_keys = tuple(
            item.node_key for item in self.proposal.root.nodes
        )
        lineage_node_keys = tuple(
            item.proposal_node_key for item in self.lineage
        )
        if lineage_node_keys != proposal_node_keys:
            raise ValueError(
                "revision candidate lineage must exactly cover proposal nodes "
                "in canonical order"
            )
        selected_base_aliases = tuple(
            item.base_node_alias
            for item in self.lineage
            if item.base_node_alias is not None
        )
        if len(selected_base_aliases) != len(set(selected_base_aliases)):
            raise ValueError(
                "a base node cannot map to multiple revision candidate nodes"
            )
        return self


class TaskGraphSemanticVerificationPromptPayload(_Contract):
    """唯一可序列化进 verifier prompt 的自包含值。"""

    schema_version: Literal["task-graph-semantic-verification-prompt-v1"] = (
        "task-graph-semantic-verification-prompt-v1"
    )
    goal: PlanningGoalPromptContext
    authority: PlanningAuthorityProjection
    base_task_graph: TaskGraphSemanticBaseSnapshot | None = None
    context_artifacts: tuple[PlanningContextArtifactProjection, ...] = Field(
        default=(), max_length=128
    )
    capabilities: PlanningCapabilityCatalogProjection
    budget: PlanningEpisodeBudget
    task_graph_proposal: InSessionTaskGraphRevisionProposal
    lineage: tuple[TaskGraphSemanticLineageProjection, ...] = Field(
        default=(),
        max_length=512,
    )
    payload_sha256: str = Field(pattern=_SHA256_PATTERN)

    @model_validator(mode="after")
    def _validate_payload(self) -> "TaskGraphSemanticVerificationPromptPayload":
        cards = {item.alias: item for item in self.authority.cards}
        if not set(self.goal.authorization_aliases).issubset(cards):
            raise ValueError("goal context references an unknown authority alias")
        if any(
            cards[alias].authority_class is not PlanningAuthorityClass.AUTHORIZATION
            for alias in self.goal.authorization_aliases
        ):
            raise ValueError("goal authorization aliases must have authorization authority")
        artifact_aliases = tuple(item.artifact_alias for item in self.context_artifacts)
        _require_unique(artifact_aliases, field_name="context artifact projection aliases")
        known_aliases = set(cards)
        projected_gap_aliases: set[str] = set()
        for artifact in self.context_artifacts:
            projected_gap_aliases.update(item.gap_alias for item in artifact.gaps)
            evidence_aliases = {
                alias
                for item in (*artifact.facts, *artifact.conflicts, *artifact.gaps)
                for alias in item.evidence_aliases
            }
            authorization_aliases = {
                alias
                for item in artifact.constraints
                for alias in item.authorization_aliases
            }
            if not (evidence_aliases | authorization_aliases).issubset(known_aliases):
                raise ValueError("artifact projection references an unknown authority alias")
            if any(
                cards[alias].authority_class is not PlanningAuthorityClass.AUTHORIZATION
                for alias in authorization_aliases
            ):
                raise ValueError("projected constraints require authorization authority")
            if any(
                cards[alias].authority_class is not PlanningAuthorityClass.EVIDENCE
                for alias in evidence_aliases
            ):
                raise ValueError("projected observations require evidence authority")
        gap_card_aliases = {
            alias
            for alias, card in cards.items()
            if card.authority_class is PlanningAuthorityClass.GAP
        }
        if gap_card_aliases != projected_gap_aliases:
            raise ValueError(
                "gap authority cards must exactly match projected typed gaps"
            )
        if self.base_task_graph is not None:
            for node in self.base_task_graph.nodes:
                if not set(node.source_anchor_aliases).issubset(known_aliases):
                    raise ValueError("base TaskGraph node references unknown authority")
                if not any(
                    cards[alias].authority_class
                    is PlanningAuthorityClass.AUTHORIZATION
                    for alias in node.source_anchor_aliases
                ):
                    raise ValueError("every base TaskGraph node requires authorization")
                if not set(node.delivery_authority_aliases).issubset(known_aliases):
                    raise ValueError("base Delivery references unknown authority")
                if not set(node.completed_delivery_gap_aliases).issubset(
                    projected_gap_aliases
                ):
                    raise ValueError(
                        "partial base Delivery references an unprojected typed gap"
                    )
        proposal_aliases = {
            alias
            for node in self.task_graph_proposal.root.nodes
            for alias in node.source_anchor_ids
        }
        if not proposal_aliases.issubset(known_aliases):
            raise ValueError("TaskGraph proposal references an unknown authority alias")
        for node in self.task_graph_proposal.root.nodes:
            if not any(
                cards[alias].authority_class is PlanningAuthorityClass.AUTHORIZATION
                for alias in node.source_anchor_ids
            ):
                raise ValueError("every TaskGraph node requires authorization authority")
            for acceptance in node.acceptance_criteria:
                if not set(acceptance.source_anchor_ids).issubset(known_aliases):
                    raise ValueError("Acceptance references an unknown authority alias")
                if not any(
                    cards[alias].authority_class is PlanningAuthorityClass.AUTHORIZATION
                    for alias in acceptance.source_anchor_ids
                ):
                    raise ValueError("every Acceptance requires authorization authority")
        proposal_node_keys = tuple(
            item.node_key for item in self.task_graph_proposal.root.nodes
        )
        if self.base_task_graph is None:
            if self.lineage:
                raise ValueError("base-null TaskGraph proposals cannot carry lineage")
        else:
            lineage_keys = tuple(item.proposal_node_key for item in self.lineage)
            if lineage_keys != proposal_node_keys:
                raise ValueError(
                    "revision lineage must exactly cover proposal nodes in canonical order"
                )
            base_aliases = {item.node_alias for item in self.base_task_graph.nodes}
            selected_base_aliases = tuple(
                item.base_node_alias
                for item in self.lineage
                if item.base_node_alias is not None
            )
            if len(selected_base_aliases) != len(set(selected_base_aliases)):
                raise ValueError("a base node cannot map to multiple proposal nodes")
            if not set(selected_base_aliases).issubset(base_aliases):
                raise ValueError("lineage references an unknown base node alias")
        expected = canonical_task_graph_semantic_verification_prompt_sha256(
            goal=self.goal,
            authority=self.authority,
            base_task_graph=self.base_task_graph,
            context_artifacts=self.context_artifacts,
            capabilities=self.capabilities,
            budget=self.budget,
            task_graph_proposal=self.task_graph_proposal,
            lineage=self.lineage,
        )
        if self.payload_sha256 != expected:
            raise ValueError("semantic verification prompt hash does not match its payload")
        if (
            len(_canonical_json(self.model_dump(mode="json")).encode("utf-8"))
            > TASK_GRAPH_SEMANTIC_VERIFICATION_MAX_PROMPT_JSON_UTF8_BYTES
        ):
            raise ValueError("semantic verification prompt exceeds its UTF-8 byte limit")
        return self

    @classmethod
    def create(cls, **values: object) -> Self:
        values = dict(values)
        goal = values["goal"]
        if not isinstance(goal, PlanningGoalPromptContext):
            goal = PlanningGoalPromptContext.model_validate(goal)
        authority = values["authority"]
        if not isinstance(authority, PlanningAuthorityProjection):
            authority = PlanningAuthorityProjection.model_validate(authority)
        base_task_graph = values.get("base_task_graph")
        if base_task_graph is not None and not isinstance(
            base_task_graph, TaskGraphSemanticBaseSnapshot
        ):
            base_task_graph = TaskGraphSemanticBaseSnapshot.model_validate(
                base_task_graph
            )
        artifacts = tuple(
            item
            if isinstance(item, PlanningContextArtifactProjection)
            else PlanningContextArtifactProjection.model_validate(item)
            for item in values.get("context_artifacts", ())  # type: ignore[union-attr]
        )
        capabilities = values["capabilities"]
        if not isinstance(capabilities, PlanningCapabilityCatalogProjection):
            capabilities = PlanningCapabilityCatalogProjection.model_validate(
                capabilities
            )
        budget = values["budget"]
        if not isinstance(budget, PlanningEpisodeBudget):
            budget = PlanningEpisodeBudget.model_validate(budget)
        task_graph_proposal = values["task_graph_proposal"]
        if not isinstance(task_graph_proposal, InSessionTaskGraphRevisionProposal):
            task_graph_proposal = InSessionTaskGraphRevisionProposal.model_validate(
                task_graph_proposal
            )
        lineage = tuple(
            item
            if isinstance(item, TaskGraphSemanticLineageProjection)
            else TaskGraphSemanticLineageProjection.model_validate(item)
            for item in values.get("lineage", ())  # type: ignore[union-attr]
        )
        values.update(
            goal=goal,
            authority=authority,
            base_task_graph=base_task_graph,
            context_artifacts=artifacts,
            capabilities=capabilities,
            budget=budget,
            task_graph_proposal=task_graph_proposal,
            lineage=lineage,
        )
        values["payload_sha256"] = canonical_task_graph_semantic_verification_prompt_sha256(
            goal=goal,
            authority=authority,
            base_task_graph=base_task_graph,
            context_artifacts=artifacts,
            capabilities=capabilities,
            budget=budget,
            task_graph_proposal=task_graph_proposal,
            lineage=lineage,
        )
        return cls.model_validate(values)


def canonical_task_graph_semantic_verification_prompt_sha256(
    *,
    goal: PlanningGoalPromptContext,
    authority: PlanningAuthorityProjection,
    base_task_graph: TaskGraphSemanticBaseSnapshot | None,
    context_artifacts: tuple[PlanningContextArtifactProjection, ...],
    capabilities: PlanningCapabilityCatalogProjection,
    budget: PlanningEpisodeBudget,
    task_graph_proposal: InSessionTaskGraphRevisionProposal,
    lineage: tuple[TaskGraphSemanticLineageProjection, ...],
) -> str:
    return _canonical_sha256(
        {
            "schema_version": "task-graph-semantic-verification-prompt-v1",
            "goal": goal.model_dump(mode="json"),
            "authority": authority.model_dump(mode="json"),
            "base_task_graph": (
                None if base_task_graph is None else base_task_graph.model_dump(mode="json")
            ),
            "context_artifacts": [
                item.model_dump(mode="json") for item in context_artifacts
            ],
            "capabilities": capabilities.model_dump(mode="json"),
            "budget": budget.model_dump(mode="json"),
            "task_graph_proposal": task_graph_proposal.model_dump(mode="json"),
            "lineage": [item.model_dump(mode="json") for item in lineage],
        }
    )


class TaskGraphSemanticReviewPolicy(_Contract):
    """用于派生精确 reviewer quorum、由 Host 封存的风险事实。

    这些事实刻意不公开私有资源身份。Store 必须根据 ``policy_source_sha256`` 指定的冻结
    resource、capability 和 base-execution authority 重新派生它们。
    """

    schema_version: Literal["task-graph-semantic-review-policy-v1"] = (
        "task-graph-semantic-review-policy-v1"
    )
    policy_source_sha256: str = Field(pattern=_SHA256_PATTERN)
    distinct_document_count: int = Field(ge=0, le=1_024)
    has_visual_input: bool
    requires_protected_effect: bool
    modifies_executed_task_graph: bool
    policy_sha256: str = Field(pattern=_SHA256_PATTERN)

    @model_validator(mode="after")
    def _validate_policy(self) -> "TaskGraphSemanticReviewPolicy":
        expected = canonical_task_graph_semantic_review_policy_sha256(
            policy_source_sha256=self.policy_source_sha256,
            distinct_document_count=self.distinct_document_count,
            has_visual_input=self.has_visual_input,
            requires_protected_effect=self.requires_protected_effect,
            modifies_executed_task_graph=self.modifies_executed_task_graph,
        )
        if self.policy_sha256 != expected:
            raise ValueError("semantic review policy hash does not match its facts")
        return self

    @classmethod
    def create(cls, **values: object) -> Self:
        values = dict(values)
        values["policy_sha256"] = canonical_task_graph_semantic_review_policy_sha256(
            policy_source_sha256=str(values["policy_source_sha256"]),
            distinct_document_count=int(values["distinct_document_count"]),
            has_visual_input=bool(values["has_visual_input"]),
            requires_protected_effect=bool(values["requires_protected_effect"]),
            modifies_executed_task_graph=bool(values["modifies_executed_task_graph"]),
        )
        return cls.model_validate(values)


def canonical_task_graph_semantic_review_policy_sha256(
    *,
    policy_source_sha256: str,
    distinct_document_count: int,
    has_visual_input: bool,
    requires_protected_effect: bool,
    modifies_executed_task_graph: bool,
) -> str:
    return _canonical_sha256(
        {
            "schema_version": "task-graph-semantic-review-policy-v1",
            "policy_source_sha256": policy_source_sha256,
            "distinct_document_count": distinct_document_count,
            "has_visual_input": has_visual_input,
            "requires_protected_effect": requires_protected_effect,
            "modifies_executed_task_graph": modifies_executed_task_graph,
        }
    )


def required_task_graph_semantic_reviewer_count(
    *,
    prompt_payload: TaskGraphSemanticVerificationPromptPayload,
    review_policy: TaskGraphSemanticReviewPolicy,
) -> int:
    """返回由 Host 派生的精确单 reviewer 或双 reviewer 策略。"""

    return (
        2
        if len(prompt_payload.task_graph_proposal.root.nodes) > 12
        or review_policy.distinct_document_count > 1
        or review_policy.has_visual_input
        or review_policy.requires_protected_effect
        or review_policy.modifies_executed_task_graph
        else 1
    )


class TaskGraphSemanticTerminalCandidateBinding(_Contract):
    """完成前首次评审的精确终止 WorkRun candidate。

    可选绑定允许现有语义请求表在节点 verifier 冻结 OutputWindow 前持久化评审。省略它
    可逐字节保留所有历史 completed-terminal 评审的请求/哈希形状。
    """

    work_run_id: str = Field(pattern=_DURABLE_ID_PATTERN)
    submitted_attempt_id: str = Field(pattern=_DURABLE_ID_PATTERN)
    node_verification_request_id: str = Field(pattern=_DURABLE_ID_PATTERN)
    output_revision: int = Field(ge=1)
    output_snapshot_sha256: str = Field(pattern=_SHA256_PATTERN)


class TaskGraphSemanticVerificationRequest(_Contract):
    """围绕一个自包含 prompt 安全 payload 的精确持久绑定。

    ``goal.authorization_manifest_id`` 指定 episode 级授权 authority。
    ``prompt_payload.authority.authority_snapshot_id`` 指定 revision 专属命名空间，其中还可
    包含证据与 gap anchor。二者被刻意设为不同身份。Store 必须在事务中重新加载二者，证明
    快照属于此精确目标，并证明每张授权卡都派生自目标 manifest（或单独接受的用户回答
    authority）。
    """

    schema_version: Literal["task-graph-semantic-verification-request-v1"] = (
        "task-graph-semantic-verification-request-v1"
    )
    verification_request_id: str = Field(pattern=_DURABLE_ID_PATTERN)
    logical_call_id: str = Field(pattern=_DURABLE_ID_PATTERN)
    verification_profile_id: str = Field(pattern=_DURABLE_ID_PATTERN)
    reviewer_ordinal: int = Field(ge=1, le=2)
    required_reviewer_count: int = Field(ge=1, le=2)
    goal: AuxiliaryPlanningGoal
    auxiliary_graph_revision: int = Field(ge=1)
    auxiliary_graph_structure_sha256: str = Field(pattern=_SHA256_PATTERN)
    prompt_payload: TaskGraphSemanticVerificationPromptPayload
    review_policy: TaskGraphSemanticReviewPolicy
    task_graph_proposal_sha256: str = Field(pattern=_SHA256_PATTERN)
    blocking_gap_aliases: tuple[str, ...] = Field(default=(), max_length=128)
    non_blocking_gap_aliases: tuple[str, ...] = Field(default=(), max_length=128)
    terminal_candidate_binding: (
        TaskGraphSemanticTerminalCandidateBinding | None
    ) = Field(default=None, exclude_if=lambda value: value is None)
    binding_sha256: str = Field(pattern=_SHA256_PATTERN)

    @property
    def task_graph_proposal(self) -> InSessionTaskGraphRevisionProposal:
        return self.prompt_payload.task_graph_proposal

    @property
    def budget(self) -> PlanningEpisodeBudget:
        return self.prompt_payload.budget

    @property
    def authority_projection(self) -> PlanningAuthorityProjection:
        return self.prompt_payload.authority

    @property
    def base_task_graph(self) -> TaskGraphSemanticBaseSnapshot | None:
        return self.prompt_payload.base_task_graph

    def to_prompt_payload(self) -> TaskGraphSemanticVerificationPromptPayload:
        """返回适配器唯一获准为模型序列化的值。"""

        return self.prompt_payload

    @field_validator("blocking_gap_aliases", "non_blocking_gap_aliases")
    @classmethod
    def _validate_gap_ids(cls, values: tuple[str, ...], info: object) -> tuple[str, ...]:
        return _require_canonical_unique(
            values,
            field_name=str(getattr(info, "field_name")),
        )

    @model_validator(mode="after")
    def _validate_request(self) -> "TaskGraphSemanticVerificationRequest":
        if self.reviewer_ordinal > self.required_reviewer_count:
            raise ValueError("reviewer ordinal cannot exceed the required reviewer count")
        goal = self.goal
        payload = self.prompt_payload
        expected_reviewer_count = required_task_graph_semantic_reviewer_count(
            prompt_payload=payload,
            review_policy=self.review_policy,
        )
        if self.required_reviewer_count != expected_reviewer_count:
            raise ValueError(
                "semantic verification reviewer count does not match Host-derived policy"
            )
        document_cards = sum(
            card.source_kind is PlanningAuthoritySourceKind.DOCUMENT
            for card in payload.authority.cards
        )
        if document_cards and self.review_policy.distinct_document_count < 1:
            raise ValueError("document context requires a nonzero document policy count")
        if any(
            card.source_kind is PlanningAuthoritySourceKind.VISUAL
            for card in payload.authority.cards
        ) and not self.review_policy.has_visual_input:
            raise ValueError("visual context must be declared by semantic review policy")
        if payload.goal.goal_id != goal.goal_id:
            raise ValueError("prompt-safe goal context must match the durable goal")
        if (
            payload.budget.goal_id != goal.goal_id
            or payload.budget.budget_ledger_id != goal.budget_ledger_id
        ):
            raise ValueError("semantic verifier budget must match its goal ledger")
        if (goal.base_task_graph_revision is None) != (
            payload.base_task_graph is None
        ):
            raise ValueError("base TaskGraph prompt snapshot must match base-null authority")
        if (
            payload.base_task_graph is not None
            and payload.base_task_graph.base_task_graph_revision
            != goal.base_task_graph_revision
        ):
            raise ValueError("base TaskGraph prompt snapshot has the wrong revision")
        if payload.base_task_graph is not None:
            base_by_alias = {
                node.node_alias: node for node in payload.base_task_graph.nodes
            }
            executed_base_aliases = {
                alias
                for alias, node in base_by_alias.items()
                if node.status is not InSessionTaskStatus.PROPOSED
            }
            if executed_base_aliases:
                reused_aliases = {
                    item.base_node_alias
                    for item in payload.lineage
                    if item.disposition is TaskGraphSemanticLineageDisposition.REUSE
                    and item.base_node_alias is not None
                }
                visible_modification = (
                    reused_aliases != set(base_by_alias)
                    or any(
                        item.disposition
                        is not TaskGraphSemanticLineageDisposition.REUSE
                        for item in payload.lineage
                    )
                )
                if (
                    visible_modification
                    and not self.review_policy.modifies_executed_task_graph
                ):
                    raise ValueError(
                        "executed base-graph modification must be declared by review policy"
                    )
        expected_proposal_hash = canonical_task_graph_revision_proposal_sha256(
            payload.task_graph_proposal
        )
        if self.task_graph_proposal_sha256 != expected_proposal_hash:
            raise ValueError("TaskGraph proposal hash does not match its payload")
        all_gaps = {
            item.gap_alias: item.blocking
            for artifact in payload.context_artifacts
            for item in artifact.gaps
        }
        if len(all_gaps) != sum(len(item.gaps) for item in payload.context_artifacts):
            raise ValueError("semantic verifier gap IDs must be globally unique")
        expected_blocking = tuple(sorted(key for key, blocking in all_gaps.items() if blocking))
        expected_non_blocking = tuple(
            sorted(key for key, blocking in all_gaps.items() if not blocking)
        )
        if self.blocking_gap_aliases != expected_blocking:
            raise ValueError("blocking gap manifest does not match context artifacts")
        if self.non_blocking_gap_aliases != expected_non_blocking:
            raise ValueError("non-blocking gap manifest does not match context artifacts")
        expected = canonical_task_graph_semantic_verification_request_sha256(
            verification_request_id=self.verification_request_id,
            logical_call_id=self.logical_call_id,
            verification_profile_id=self.verification_profile_id,
            reviewer_ordinal=self.reviewer_ordinal,
            required_reviewer_count=self.required_reviewer_count,
            goal=self.goal,
            auxiliary_graph_revision=self.auxiliary_graph_revision,
            auxiliary_graph_structure_sha256=self.auxiliary_graph_structure_sha256,
            prompt_payload=self.prompt_payload,
            review_policy=self.review_policy,
            task_graph_proposal_sha256=self.task_graph_proposal_sha256,
            blocking_gap_aliases=self.blocking_gap_aliases,
            non_blocking_gap_aliases=self.non_blocking_gap_aliases,
            terminal_candidate_binding=self.terminal_candidate_binding,
        )
        if self.binding_sha256 != expected:
            raise ValueError("semantic verification request hash does not match its input")
        return self

    @classmethod
    def create(cls, **values: object) -> Self:
        values = dict(values)
        prompt_payload = values["prompt_payload"]
        if not isinstance(
            prompt_payload, TaskGraphSemanticVerificationPromptPayload
        ):
            prompt_payload = TaskGraphSemanticVerificationPromptPayload.model_validate(
                prompt_payload
            )
        goal = values["goal"]
        if not isinstance(goal, AuxiliaryPlanningGoal):
            goal = AuxiliaryPlanningGoal.model_validate(goal)
        review_policy = values["review_policy"]
        if not isinstance(review_policy, TaskGraphSemanticReviewPolicy):
            review_policy = TaskGraphSemanticReviewPolicy.model_validate(review_policy)
        values["prompt_payload"] = prompt_payload
        values["goal"] = goal
        values["review_policy"] = review_policy
        candidate_binding = values.get("terminal_candidate_binding")
        if candidate_binding is not None and not isinstance(
            candidate_binding,
            TaskGraphSemanticTerminalCandidateBinding,
        ):
            candidate_binding = TaskGraphSemanticTerminalCandidateBinding.model_validate(
                candidate_binding
            )
        values["terminal_candidate_binding"] = candidate_binding
        proposal = prompt_payload.task_graph_proposal
        proposal_sha = canonical_task_graph_revision_proposal_sha256(proposal)
        values["task_graph_proposal_sha256"] = proposal_sha
        blocking_gap_aliases = tuple(
            sorted(
                gap.gap_alias
                for artifact in prompt_payload.context_artifacts  # type: ignore[union-attr]
                for gap in artifact.gaps
                if gap.blocking
            )
        )
        non_blocking_gap_aliases = tuple(
            sorted(
                gap.gap_alias
                for artifact in prompt_payload.context_artifacts  # type: ignore[union-attr]
                for gap in artifact.gaps
                if not gap.blocking
            )
        )
        values["blocking_gap_aliases"] = blocking_gap_aliases
        values["non_blocking_gap_aliases"] = non_blocking_gap_aliases
        values["binding_sha256"] = canonical_task_graph_semantic_verification_request_sha256(
            verification_request_id=str(values["verification_request_id"]),
            logical_call_id=str(values["logical_call_id"]),
            verification_profile_id=str(values["verification_profile_id"]),
            reviewer_ordinal=int(values["reviewer_ordinal"]),
            required_reviewer_count=int(values["required_reviewer_count"]),
            goal=goal,
            auxiliary_graph_revision=int(values["auxiliary_graph_revision"]),
            auxiliary_graph_structure_sha256=str(values["auxiliary_graph_structure_sha256"]),
            prompt_payload=prompt_payload,  # type: ignore[arg-type]
            review_policy=review_policy,
            task_graph_proposal_sha256=proposal_sha,
            blocking_gap_aliases=blocking_gap_aliases,
            non_blocking_gap_aliases=non_blocking_gap_aliases,
            terminal_candidate_binding=candidate_binding,
        )
        return cls.model_validate(values)


def canonical_task_graph_revision_proposal_sha256(
    proposal: InSessionTaskGraphRevisionProposal,
) -> str:
    return _canonical_sha256(proposal.model_dump(mode="json"))


def canonical_task_graph_semantic_verification_request_sha256(
    *,
    verification_request_id: str,
    logical_call_id: str,
    verification_profile_id: str,
    reviewer_ordinal: int,
    required_reviewer_count: int,
    goal: AuxiliaryPlanningGoal,
    auxiliary_graph_revision: int,
    auxiliary_graph_structure_sha256: str,
    prompt_payload: TaskGraphSemanticVerificationPromptPayload,
    review_policy: TaskGraphSemanticReviewPolicy,
    task_graph_proposal_sha256: str,
    blocking_gap_aliases: tuple[str, ...],
    non_blocking_gap_aliases: tuple[str, ...],
    terminal_candidate_binding: (
        TaskGraphSemanticTerminalCandidateBinding | None
    ) = None,
) -> str:
    payload = {
            "schema_version": "task-graph-semantic-verification-request-v1",
            "verification_request_id": verification_request_id,
            "logical_call_id": logical_call_id,
            "verification_profile_id": verification_profile_id,
            "reviewer_ordinal": reviewer_ordinal,
            "required_reviewer_count": required_reviewer_count,
            "goal": goal.model_dump(mode="json"),
            "auxiliary_graph_revision": auxiliary_graph_revision,
            "auxiliary_graph_structure_sha256": auxiliary_graph_structure_sha256,
            "prompt_payload": prompt_payload.model_dump(mode="json"),
            "review_policy": review_policy.model_dump(mode="json"),
            "task_graph_proposal_sha256": task_graph_proposal_sha256,
            "blocking_gap_aliases": list(blocking_gap_aliases),
            "non_blocking_gap_aliases": list(non_blocking_gap_aliases),
    }
    if terminal_candidate_binding is not None:
        payload["terminal_candidate_binding"] = terminal_candidate_binding.model_dump(
            mode="json"
        )
    return _canonical_sha256(payload)


class TaskGraphSemanticVerificationResult(_Contract):
    """总体 disposition 始终由 Host 派生的 reviewer 输出。"""

    schema_version: Literal["task-graph-semantic-verification-result-v1"] = (
        "task-graph-semantic-verification-result-v1"
    )
    verification_result_id: str = Field(pattern=_DURABLE_ID_PATTERN)
    verification_request_id: str = Field(pattern=_DURABLE_ID_PATTERN)
    request_binding_sha256: str = Field(pattern=_SHA256_PATTERN)
    logical_call_id: str = Field(pattern=_DURABLE_ID_PATTERN)
    verification_profile_id: str = Field(pattern=_DURABLE_ID_PATTERN)
    reviewer_ordinal: int = Field(ge=1, le=2)
    required_reviewer_count: int = Field(ge=1, le=2)
    items: tuple[TaskGraphSemanticVerificationItem, ...] = Field(
        min_length=len(TaskGraphSemanticVerificationDimension),
        max_length=len(TaskGraphSemanticVerificationDimension),
    )
    result_sha256: str = Field(pattern=_SHA256_PATTERN)

    @property
    def all_pass(self) -> bool:
        return all(
            item.verdict is TaskGraphSemanticVerificationVerdict.PASS
            for item in self.items
        )

    @property
    def host_disposition(self) -> TaskGraphSemanticVerificationDisposition:
        route = self.host_terminal_route
        if route is TaskGraphSemanticTerminalRoute.BLOCKED:
            return TaskGraphSemanticVerificationDisposition.BLOCKED
        if route is not TaskGraphSemanticTerminalRoute.PASS:
            return TaskGraphSemanticVerificationDisposition.REVISE
        return TaskGraphSemanticVerificationDisposition.PASS

    @property
    def host_terminal_route(self) -> TaskGraphSemanticTerminalRoute:
        return derive_task_graph_semantic_terminal_route((self,))

    @model_validator(mode="after")
    def _validate_result(self) -> "TaskGraphSemanticVerificationResult":
        if self.reviewer_ordinal > self.required_reviewer_count:
            raise ValueError("reviewer ordinal cannot exceed the required reviewer count")
        dimensions = tuple(item.dimension for item in self.items)
        if dimensions != tuple(TaskGraphSemanticVerificationDimension):
            raise ValueError("semantic verification items must cover all dimensions in order")
        expected = canonical_task_graph_semantic_verification_result_sha256(
            verification_result_id=self.verification_result_id,
            verification_request_id=self.verification_request_id,
            request_binding_sha256=self.request_binding_sha256,
            logical_call_id=self.logical_call_id,
            verification_profile_id=self.verification_profile_id,
            reviewer_ordinal=self.reviewer_ordinal,
            required_reviewer_count=self.required_reviewer_count,
            items=self.items,
        )
        if self.result_sha256 != expected:
            raise ValueError("semantic verification result hash does not match its output")
        if (
            len(_canonical_json(self.model_dump(mode="json")).encode("utf-8"))
            > TASK_GRAPH_SEMANTIC_VERIFICATION_MAX_RESULT_JSON_UTF8_BYTES
        ):
            raise ValueError("semantic verification result exceeds its UTF-8 byte limit")
        return self

    @classmethod
    def create(cls, **values: object) -> Self:
        values = dict(values)
        items = tuple(
            item
            if isinstance(item, TaskGraphSemanticVerificationItem)
            else TaskGraphSemanticVerificationItem.model_validate(item)
            for item in values["items"]  # type: ignore[union-attr]
        )
        values["items"] = items
        values["result_sha256"] = canonical_task_graph_semantic_verification_result_sha256(
            verification_result_id=str(values["verification_result_id"]),
            verification_request_id=str(values["verification_request_id"]),
            request_binding_sha256=str(values["request_binding_sha256"]),
            logical_call_id=str(values["logical_call_id"]),
            verification_profile_id=str(values["verification_profile_id"]),
            reviewer_ordinal=int(values["reviewer_ordinal"]),
            required_reviewer_count=int(values["required_reviewer_count"]),
            items=items,
        )
        return cls.model_validate(values)


def canonical_task_graph_semantic_verification_result_sha256(
    *,
    verification_result_id: str,
    verification_request_id: str,
    request_binding_sha256: str,
    logical_call_id: str,
    verification_profile_id: str,
    reviewer_ordinal: int,
    required_reviewer_count: int,
    items: tuple[TaskGraphSemanticVerificationItem, ...],
) -> str:
    return _canonical_sha256(
        {
            "schema_version": "task-graph-semantic-verification-result-v1",
            "verification_result_id": verification_result_id,
            "verification_request_id": verification_request_id,
            "request_binding_sha256": request_binding_sha256,
            "logical_call_id": logical_call_id,
            "verification_profile_id": verification_profile_id,
            "reviewer_ordinal": reviewer_ordinal,
            "required_reviewer_count": required_reviewer_count,
            "items": [item.model_dump(mode="json") for item in items],
        }
    )


def derive_task_graph_semantic_terminal_route(
    results: tuple[TaskGraphSemanticVerificationResult, ...],
) -> TaskGraphSemanticTerminalRoute:
    """为完整 reviewer 集合派生局部性最低的安全路由。

    显式缺少信息/authority 和证据不足始终会阻塞。上游调查 finding 优先于仅终止修复。当前
    每个未通过结果都携带显式失败 scope，因此 Host 绝不会根据不完整结果猜测路由。
    """

    if not results:
        raise ValueError("semantic terminal routing requires at least one result")
    if any(
        not isinstance(result, TaskGraphSemanticVerificationResult)
        for result in results
    ):
        raise TypeError("semantic terminal routing requires typed results")
    items = tuple(item for result in results for item in result.items)
    if all(
        item.verdict is TaskGraphSemanticVerificationVerdict.PASS
        for item in items
    ):
        return TaskGraphSemanticTerminalRoute.PASS
    if any(
        item.verdict
        is TaskGraphSemanticVerificationVerdict.INSUFFICIENT_EVIDENCE
        or item.failure_scope
        in {
            TaskGraphSemanticFailureScope.MISSING_INFORMATION,
            TaskGraphSemanticFailureScope.MISSING_AUTHORITY,
        }
        for item in items
    ):
        return TaskGraphSemanticTerminalRoute.BLOCKED
    if any(
        item.failure_scope
        is TaskGraphSemanticFailureScope.AUXILIARY_INVESTIGATION
        for item in items
    ):
        return TaskGraphSemanticTerminalRoute.REPLAN_AUXILIARY
    return TaskGraphSemanticTerminalRoute.RETRY_TERMINAL_ATTEMPT


def is_task_graph_semantic_user_information_block(
    *,
    requests: tuple[TaskGraphSemanticVerificationRequest, ...],
    results: tuple[TaskGraphSemanticVerificationResult, ...],
) -> bool:
    """区分普通澄清停止与正式 authority。

    此谓词被刻意设为机械判断：所有导致 BLOCKED 路由的 finding 都必须是显式类型为
    ``missing_information``、且绑定到冻结 ContextArtifact gap 的证据不足 finding。空 scope
    与 ``missing_authority`` 永远无法通过此边界。
    """

    if not requests or not results:
        return False
    if derive_task_graph_semantic_terminal_route(results) is not (
        TaskGraphSemanticTerminalRoute.BLOCKED
    ):
        return False
    gap_aliases = {
        alias
        for request in requests
        for alias in (
            *request.blocking_gap_aliases,
            *request.non_blocking_gap_aliases,
        )
    }
    route_findings = tuple(
        item
        for result in results
        for item in result.items
        if (
            item.verdict
            is TaskGraphSemanticVerificationVerdict.INSUFFICIENT_EVIDENCE
            or item.failure_scope
            in {
                TaskGraphSemanticFailureScope.MISSING_INFORMATION,
                TaskGraphSemanticFailureScope.MISSING_AUTHORITY,
            }
        )
    )
    return bool(
        gap_aliases
        and route_findings
        and all(
            item.verdict
            is TaskGraphSemanticVerificationVerdict.INSUFFICIENT_EVIDENCE
            and item.failure_scope
            is TaskGraphSemanticFailureScope.MISSING_INFORMATION
            and item.gap_aliases
            and set(item.gap_aliases).issubset(gap_aliases)
            for item in route_findings
        )
    )


class TaskGraphSemanticVerificationBindingCode(StrEnum):
    REQUEST_MISMATCH = "request_mismatch"
    UNKNOWN_NODE_ALIAS = "unknown_node_alias"
    UNKNOWN_EVIDENCE_ALIAS = "unknown_evidence_alias"
    UNKNOWN_GAP_ALIAS = "unknown_gap_alias"
    EVIDENCE_COVERAGE_MISMATCH = "evidence_coverage_mismatch"
    GAP_COVERAGE_MISMATCH = "gap_coverage_mismatch"
    BLOCKING_GAP_PASS = "blocking_gap_pass"
    QUORUM_MISMATCH = "quorum_mismatch"
    PRIVATE_BINDING_DISCLOSURE = "private_binding_disclosure"


class TaskGraphSemanticVerificationBindingError(ValueError):
    def __init__(
        self,
        code: TaskGraphSemanticVerificationBindingCode,
        message: str,
    ) -> None:
        self.code = code
        super().__init__(message)


def validate_task_graph_semantic_verification_result(
    *,
    request: TaskGraphSemanticVerificationRequest,
    result: TaskGraphSemanticVerificationResult,
) -> TaskGraphSemanticVerificationResult:
    """将模型别名绑定到一个请求，并拒绝私有绑定回显。"""

    if (
        result.verification_request_id != request.verification_request_id
        or result.request_binding_sha256 != request.binding_sha256
        or result.logical_call_id != request.logical_call_id
        or result.verification_profile_id != request.verification_profile_id
        or result.reviewer_ordinal != request.reviewer_ordinal
        or result.required_reviewer_count != request.required_reviewer_count
    ):
        raise TaskGraphSemanticVerificationBindingError(
            TaskGraphSemanticVerificationBindingCode.REQUEST_MISMATCH,
            "semantic verification result does not match its exact request binding",
        )
    node_keys = {
        item.node_key for item in request.task_graph_proposal.root.nodes
    }
    evidence_aliases = {
        item.alias
        for item in request.authority_projection.cards
        if item.authority_class is PlanningAuthorityClass.EVIDENCE
    }
    gap_aliases = {
        gap.gap_alias
        for artifact in request.prompt_payload.context_artifacts
        for gap in artifact.gaps
    }
    referenced_proposal_aliases = {
        alias
        for node in request.task_graph_proposal.root.nodes
        for alias in (
            *node.source_anchor_ids,
            *(
                source_alias
                for acceptance in node.acceptance_criteria
                for source_alias in acceptance.source_anchor_ids
            ),
        )
    }
    required_evidence_aliases = evidence_aliases & referenced_proposal_aliases
    private_tokens = _semantic_verification_private_binding_tokens(request)
    for item in result.items:
        if not set(item.affected_node_keys).issubset(node_keys):
            raise TaskGraphSemanticVerificationBindingError(
                TaskGraphSemanticVerificationBindingCode.UNKNOWN_NODE_ALIAS,
                "semantic verification result references an unknown proposal node",
            )
        if not set(item.evidence_aliases).issubset(evidence_aliases):
            raise TaskGraphSemanticVerificationBindingError(
                TaskGraphSemanticVerificationBindingCode.UNKNOWN_EVIDENCE_ALIAS,
                "semantic verification result references an unknown evidence alias",
            )
        if not set(item.gap_aliases).issubset(gap_aliases):
            raise TaskGraphSemanticVerificationBindingError(
                TaskGraphSemanticVerificationBindingCode.UNKNOWN_GAP_ALIAS,
                "semantic verification result references an unknown gap alias",
            )
        if (
            item.dimension
            is TaskGraphSemanticVerificationDimension.EVIDENCE_GROUNDING
            and set(item.evidence_aliases) != required_evidence_aliases
        ):
            raise TaskGraphSemanticVerificationBindingError(
                TaskGraphSemanticVerificationBindingCode.EVIDENCE_COVERAGE_MISMATCH,
                "evidence-grounding review must exactly cover proposal evidence",
            )
        if (
            item.dimension
            is TaskGraphSemanticVerificationDimension.GAP_DISPOSITION
            and set(item.gap_aliases) != gap_aliases
        ):
            raise TaskGraphSemanticVerificationBindingError(
                TaskGraphSemanticVerificationBindingCode.GAP_COVERAGE_MISMATCH,
                "gap-disposition review must exactly cover the frozen gap manifest",
            )
        if any(token in item.finding for token in private_tokens):
            raise TaskGraphSemanticVerificationBindingError(
                TaskGraphSemanticVerificationBindingCode.PRIVATE_BINDING_DISCLOSURE,
                "semantic verification finding disclosed a private binding token",
            )
    if request.blocking_gap_aliases:
        gap_item = next(
            item
            for item in result.items
            if item.dimension
            is TaskGraphSemanticVerificationDimension.GAP_DISPOSITION
        )
        if (
            gap_item.verdict
            is not TaskGraphSemanticVerificationVerdict.INSUFFICIENT_EVIDENCE
        ):
            raise TaskGraphSemanticVerificationBindingError(
                TaskGraphSemanticVerificationBindingCode.BLOCKING_GAP_PASS,
                "a frozen blocking gap requires an insufficient-evidence disposition",
            )
    return result


def validate_task_graph_semantic_verification_quorum(
    *,
    requests: tuple[TaskGraphSemanticVerificationRequest, ...],
    results: tuple[TaskGraphSemanticVerificationResult, ...],
) -> tuple[TaskGraphSemanticVerificationResult, ...]:
    """验证一个完整且独立调用的 reviewer quorum。

    结果哈希特定于 reviewer，因为序号和结果身份属于其 payload。因此，仅哈希不同并不能
    证明两次评审确实独立调用。此聚合 guard 还要求持久请求、逻辑调用和结果身份各不相同，
    同时将每个 reviewer 绑定到完全相同的冻结提案与 authority payload。
    """

    if not requests or len(requests) != len(results):
        raise TaskGraphSemanticVerificationBindingError(
            TaskGraphSemanticVerificationBindingCode.QUORUM_MISMATCH,
            "semantic verification quorum requires paired requests and results",
        )
    reviewer_count = len(requests)
    if reviewer_count not in (1, 2):
        raise TaskGraphSemanticVerificationBindingError(
            TaskGraphSemanticVerificationBindingCode.QUORUM_MISMATCH,
            "semantic verification quorum must contain one or two reviewers",
        )
    ordered_requests = tuple(sorted(requests, key=lambda item: item.reviewer_ordinal))
    ordered_results = tuple(sorted(results, key=lambda item: item.reviewer_ordinal))
    expected_ordinals = tuple(range(1, reviewer_count + 1))
    if (
        tuple(item.reviewer_ordinal for item in ordered_requests) != expected_ordinals
        or tuple(item.reviewer_ordinal for item in ordered_results) != expected_ordinals
        or any(item.required_reviewer_count != reviewer_count for item in ordered_requests)
        or any(item.required_reviewer_count != reviewer_count for item in ordered_results)
    ):
        raise TaskGraphSemanticVerificationBindingError(
            TaskGraphSemanticVerificationBindingCode.QUORUM_MISMATCH,
            "semantic verification quorum does not exactly cover reviewer ordinals",
        )
    for values, label in (
        (
            tuple(item.verification_request_id for item in ordered_requests),
            "request identities",
        ),
        (
            tuple(item.logical_call_id for item in ordered_requests),
            "logical call identities",
        ),
        (
            tuple(item.verification_result_id for item in ordered_results),
            "result identities",
        ),
        (
            tuple(item.result_sha256 for item in ordered_results),
            "result hashes",
        ),
    ):
        if len(values) != len(set(values)):
            raise TaskGraphSemanticVerificationBindingError(
                TaskGraphSemanticVerificationBindingCode.QUORUM_MISMATCH,
                f"semantic verification quorum reused {label}",
            )
    reference = ordered_requests[0]
    shared_request_payload = reference.model_dump(
        mode="json",
        exclude={
            "verification_request_id",
            "logical_call_id",
            "reviewer_ordinal",
            "binding_sha256",
        },
    )
    for request in ordered_requests[1:]:
        if request.model_dump(
            mode="json",
            exclude={
                "verification_request_id",
                "logical_call_id",
                "reviewer_ordinal",
                "binding_sha256",
            },
        ) != shared_request_payload:
            raise TaskGraphSemanticVerificationBindingError(
                TaskGraphSemanticVerificationBindingCode.QUORUM_MISMATCH,
                "semantic reviewers do not share one frozen proposal authority",
            )
    for request, result in zip(ordered_requests, ordered_results, strict=True):
        validate_task_graph_semantic_verification_result(
            request=request,
            result=result,
        )
    return ordered_results


def _semantic_verification_private_binding_tokens(
    request: TaskGraphSemanticVerificationRequest,
) -> set[str]:
    goal = request.goal
    prompt = request.prompt_payload
    tokens = {
        request.verification_request_id,
        request.logical_call_id,
        request.binding_sha256,
        request.auxiliary_graph_structure_sha256,
        goal.session_id,
        goal.task_id,
        goal.auxiliary_graph_id,
        goal.goal_id,
        goal.authorization_manifest_id,
        goal.budget_ledger_id,
        prompt.payload_sha256,
        prompt.authority.authority_snapshot_id,
        prompt.authority.authority_snapshot_sha256,
        prompt.authority.projection_sha256,
        prompt.capabilities.capability_catalog_snapshot_id,
        prompt.capabilities.capability_catalog_snapshot_sha256,
        prompt.capabilities.projection_sha256,
        prompt.budget.snapshot_sha256,
        request.review_policy.policy_source_sha256,
        request.review_policy.policy_sha256,
    }
    for artifact in prompt.context_artifacts:
        tokens.update(
            {
                artifact.artifact_id,
                artifact.artifact_sha256,
                artifact.projection_sha256,
            }
        )
    if prompt.base_task_graph is not None:
        tokens.update(
            {
                prompt.base_task_graph.source_snapshot_sha256,
                prompt.base_task_graph.projection_sha256,
            }
        )
    for extension in prompt.budget.extensions:
        tokens.add(extension.extension_receipt_id)
        tokens.add(extension.authorization_anchor_id)
        tokens.add(extension.receipt_sha256)
        if extension.previous_extension_receipt_sha256 is not None:
            tokens.add(extension.previous_extension_receipt_sha256)
    return {token for token in tokens if token}
