"""TaskNode 语义验证与交付绑定的纯契约。"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..task_graph.contracts import InSessionTaskAcceptanceProposal
from .budget import WorkRunBudgetDisposition, WorkRunBudgetTransition
from .contracts import (
    AcceptanceProgressSnapshot,
    AttemptStatus,
    Attempt,
    OutputWindow,
    ExecutionSubject,
    TaskNodeSubject,
    ToolResultStatus,
    ToolResult,
    WorkRunBudget,
    WorkRunStatus,
    WorkRun,
)
class _VerificationContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class VerificationVerdict(StrEnum):
    PASSED = "passed"
    NOT_SATISFIED = "not_satisfied"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class DownstreamVerificationDisposition(StrEnum):
    """由 Acceptance 下游语义门产生、归 Host 持有的路由。

    下游门绝不改写节点验证器给出的 Acceptance 判定。``RETRY_ATTEMPT`` 是唯一可使用普通
    WorkRun 未通过转换的处置方式；其他处置方式要求各自的控制器处理专用等待或重新规划转换。
    """

    PASS = "pass"
    RETRY_ATTEMPT = "retry_attempt"
    REPLAN = "replan"
    BLOCKED = "blocked"


class DownstreamVerificationFeedback(_VerificationContract):
    """来自审查已提交候选项之门的认证反馈。"""

    gate_id: str = Field(min_length=1, max_length=120)
    disposition: DownstreamVerificationDisposition
    finding: str = Field(min_length=1, max_length=4_000)
    repair_objective: str | None = Field(default=None, min_length=1, max_length=4_000)
    blocking_questions: tuple[str, ...] = Field(default=(), max_length=16)
    source_result_id: str = Field(min_length=1, max_length=200)
    source_result_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    affected_subject_ids: tuple[str, ...] = Field(default=(), max_length=128)

    @field_validator("blocking_questions", "affected_subject_ids")
    @classmethod
    def _require_unique_bounded_values(
        cls,
        values: tuple[str, ...],
    ) -> tuple[str, ...]:
        if len(values) != len(set(values)) or any(not value.strip() for value in values):
            raise ValueError("downstream feedback references must be unique non-empty text")
        return values

    @model_validator(mode="after")
    def _validate_route_shape(self) -> 'DownstreamVerificationFeedback':
        if self.disposition is DownstreamVerificationDisposition.PASS:
            if self.repair_objective is not None or self.blocking_questions:
                raise ValueError("PASS cannot carry repair or blocking work")
        elif self.disposition in {
            DownstreamVerificationDisposition.RETRY_ATTEMPT,
            DownstreamVerificationDisposition.REPLAN,
        }:
            if self.repair_objective is None or self.blocking_questions:
                raise ValueError(
                    f"{self.disposition.name} requires only a repair objective"
                )
        elif self.repair_objective is not None or not self.blocking_questions:
            raise ValueError("BLOCKED requires only blocking questions")
        return self


class AcceptanceVerificationFeedback(_VerificationContract):
    acceptance_id: str = Field(min_length=1)
    verdict: VerificationVerdict
    finding: str = Field(min_length=1)
    missing_requirements: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _validate_missing_requirements(self) -> 'AcceptanceVerificationFeedback':
        if any(not item.strip() for item in self.missing_requirements):
            raise ValueError("missing requirements must not be empty")
        if len(self.missing_requirements) != len(set(self.missing_requirements)):
            raise ValueError("missing requirements must be unique")
        if self.verdict is VerificationVerdict.PASSED and self.missing_requirements:
            raise ValueError("a passed Acceptance cannot have missing requirements")
        return self


class NodeVerificationResult(_VerificationContract):
    """一个绑定到 Host 的完整语义结果；``all_pass`` 由其他字段推导。"""

    verification_request_id: str = Field(min_length=1)
    verification_request_revision: int = Field(ge=1)
    work_run_id: str = Field(min_length=1)
    locked_work_run_revision: int = Field(ge=1)
    submitted_attempt_id: str = Field(min_length=1)
    acceptance_progress_revision: int = Field(ge=1)
    subject: ExecutionSubject
    output_revision: int = Field(ge=1)
    acceptance_results: tuple[AcceptanceVerificationFeedback, ...] = Field(
        min_length=1
    )
    downstream_results: tuple[DownstreamVerificationFeedback, ...] = Field(
        default=(),
        max_length=8,
    )
    all_pass: bool

    @model_validator(mode="after")
    def _validate_host_aggregate(self) -> 'NodeVerificationResult':
        ids = [item.acceptance_id for item in self.acceptance_results]
        if len(ids) != len(set(ids)):
            raise ValueError("verified Acceptance IDs must be unique")
        gate_ids = [item.gate_id for item in self.downstream_results]
        if len(gate_ids) != len(set(gate_ids)):
            raise ValueError("downstream verification gate IDs must be unique")
        derived = all(
            item.verdict is VerificationVerdict.PASSED
            for item in self.acceptance_results
        ) and all(
            item.disposition is DownstreamVerificationDisposition.PASS
            for item in self.downstream_results
        )
        if self.all_pass is not derived:
            raise ValueError("all_pass must equal the Host-derived aggregate")
        return self


class SupportingToolResult(_VerificationContract):
    """一个绑定到 WorkRun 的 ToolResult 及其不可变工具标识。"""

    work_run_id: str = Field(min_length=1)
    tool_id: str = Field(min_length=1)
    tool_version: str = Field(min_length=1)
    result: ToolResult

    @field_validator("result")
    @classmethod
    def _require_succeeded_result(cls, value: ToolResult) -> ToolResult:
        if value.status is not ToolResultStatus.SUCCEEDED:
            raise ValueError(
                "only a succeeded ToolResult can support an Acceptance"
            )
        return value


class TaskNodeVerificationRequestStatus(StrEnum):
    PENDING = "pending"
    INTERRUPTED = "interrupted"
    COMPLETED = "completed"


class TaskNodeVerificationRequest(_VerificationContract):
    """不复制正文或 ToolResult 的持久逻辑请求/检查点。"""

    schema_version: Literal["task-node-verification-request-v1"] = (
        "task-node-verification-request-v1"
    )
    verification_request_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    request_turn_id: str = Field(min_length=1)
    work_run_id: str = Field(min_length=1)
    subject: ExecutionSubject
    submitted_attempt_id: str = Field(min_length=1)
    output_revision: int = Field(ge=1)
    acceptance_progress_revision: int = Field(ge=1)
    acceptance_ids: tuple[str, ...] = Field(min_length=1)
    supporting_tool_result_ids: tuple[str, ...] = ()
    dependency_delivery_ids: tuple[str, ...] = ()
    locked_work_run_revision: int = Field(ge=1)
    prepared_budget: WorkRunBudget
    revision: int = Field(default=1, ge=1)
    status: TaskNodeVerificationRequestStatus = (
        TaskNodeVerificationRequestStatus.PENDING
    )
    technical_error_code: str | None = Field(default=None, min_length=1, max_length=160)

    @field_validator(
        "acceptance_ids",
        "supporting_tool_result_ids",
        "dependency_delivery_ids",
    )
    @classmethod
    def _require_unique_nonempty_ids(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if any(not value.strip() for value in values):
            raise ValueError("verification request IDs must not be empty")
        if len(values) != len(set(values)):
            raise ValueError("verification request IDs must be unique")
        return values

    @model_validator(mode="after")
    def _validate_status_shape(self) -> 'TaskNodeVerificationRequest':
        if self.status is TaskNodeVerificationRequestStatus.INTERRUPTED:
            if self.technical_error_code is None:
                raise ValueError("interrupted verification requires a technical error")
        elif self.technical_error_code is not None:
            raise ValueError("only interrupted verification may retain a technical error")
        return self


class TaskNodeVerificationRecord(_VerificationContract):
    request: TaskNodeVerificationRequest
    result: NodeVerificationResult | None = None

    @model_validator(mode="after")
    def _validate_result_shape(self) -> 'TaskNodeVerificationRecord':
        completed = (
            self.request.status is TaskNodeVerificationRequestStatus.COMPLETED
        )
        if completed != (self.result is not None):
            raise ValueError("only completed verification may carry a semantic result")
        if self.result is not None and (
            self.result.verification_request_id
            != self.request.verification_request_id
            or self.result.verification_request_revision != self.request.revision - 1
            or self.result.work_run_id != self.request.work_run_id
            or self.result.locked_work_run_revision
            != self.request.locked_work_run_revision
            or self.result.subject != self.request.subject
            or self.result.submitted_attempt_id
            != self.request.submitted_attempt_id
            or self.result.output_revision != self.request.output_revision
            or self.result.acceptance_progress_revision
            != self.request.acceptance_progress_revision
            or tuple(item.acceptance_id for item in self.result.acceptance_results)
            != self.request.acceptance_ids
        ):
            raise ValueError("semantic result does not match its verification request")
        return self


class PreparedTaskNodeVerification(_VerificationContract):
    """从一个持久请求的引用重建出的精确瞬态输入。"""

    record: TaskNodeVerificationRecord
    invocation_turn_id: str = Field(min_length=1)
    window_state_version: int = Field(ge=1)
    work_run: WorkRun
    submitted_attempt: Attempt
    acceptance_progress: AcceptanceProgressSnapshot
    node_title: str = Field(min_length=1, max_length=240)
    node_objective: str = Field(min_length=1, max_length=2_000)
    acceptances: tuple[InSessionTaskAcceptanceProposal, ...] = Field(min_length=1)
    locked_output_window: OutputWindow
    supporting_tool_results: tuple[SupportingToolResult, ...] = ()

    @model_validator(mode="after")
    def _validate_exact_projection(self) -> 'PreparedTaskNodeVerification':
        request = self.record.request
        if (
            request.status is not TaskNodeVerificationRequestStatus.PENDING
            or self.record.result is not None
        ):
            raise ValueError("only a pending request is callable verification input")
        if (
            self.work_run.work_run_id != request.work_run_id
            or self.work_run.subject != request.subject
            or self.work_run.revision <= request.locked_work_run_revision
            or self.work_run.status is not WorkRunStatus.ACTIVE
            or self.work_run.reason != "verification_pending"
        ):
            raise ValueError("prepared WorkRun does not match the verification request")
        if (
            self.submitted_attempt.attempt_id != request.submitted_attempt_id
            or self.submitted_attempt.work_run_id != request.work_run_id
            or self.submitted_attempt.status is not AttemptStatus.CLOSED
            or self.submitted_attempt.submitted_output_revision
            != request.output_revision
        ):
            raise ValueError("prepared submit Attempt does not match the request")
        if (
            self.acceptance_progress.work_run_id != request.work_run_id
            or self.acceptance_progress.subject != request.subject
            or self.acceptance_progress.revision
            != request.acceptance_progress_revision
            or self.acceptance_progress.evaluated_output_revision
            != request.output_revision
        ):
            raise ValueError("prepared AcceptanceProgress does not match the request")
        acceptance_ids = tuple(item.acceptance_id for item in self.acceptances)
        progress_ids = tuple(
            item.acceptance_id for item in self.acceptance_progress.items
        )
        if acceptance_ids != request.acceptance_ids or set(progress_ids) != set(
            request.acceptance_ids
        ):
            raise ValueError("prepared Acceptances do not match the request")
        if not all(
            item.model_claimed_satisfied for item in self.acceptance_progress.items
        ):
            raise ValueError("prepared AcceptanceProgress must remain fully satisfied")
        if (
            self.locked_output_window.work_run_id != request.work_run_id
            or self.locked_output_window.output_revision != request.output_revision
            or not self.locked_output_window.content.strip()
        ):
            raise ValueError("prepared OutputWindow does not match the request")
        supporting_ids = tuple(
            item.result.tool_result_id for item in self.supporting_tool_results
        )
        if set(supporting_ids) != set(request.supporting_tool_result_ids) or len(
            supporting_ids
        ) != len(request.supporting_tool_result_ids):
            raise ValueError("prepared supporting ToolResults do not match the request")
        if any(
            item.work_run_id != request.work_run_id
            for item in self.supporting_tool_results
        ):
            raise ValueError("prepared supporting ToolResult has the wrong WorkRun")
        return self


class TaskNodeDelivery(_VerificationContract):
    """对一个终态 WorkRun OutputWindow 的不可变语义所有权。"""

    schema_version: Literal["task-node-delivery-v1"] = "task-node-delivery-v1"
    delivery_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    work_run_id: str = Field(min_length=1)
    subject: TaskNodeSubject
    verification_request_id: str = Field(min_length=1)
    submitted_attempt_id: str = Field(min_length=1)
    output_revision: int = Field(ge=1)
    created_turn_id: str = Field(min_length=1)


class ResolvedTaskNodeDelivery(_VerificationContract):
    """基于仅含引用的持久交付生成的单快照读取投影。"""

    delivery: TaskNodeDelivery
    output_window: OutputWindow

    @model_validator(mode="after")
    def _validate_resolved_output(self) -> 'ResolvedTaskNodeDelivery':
        if (
            self.output_window.work_run_id != self.delivery.work_run_id
            or self.output_window.output_revision != self.delivery.output_revision
            or not self.output_window.content.strip()
        ):
            raise ValueError("resolved NodeDelivery OutputWindow binding is invalid")
        return self


class CurrentTaskNodeDeliveryResolutionKind(StrEnum):
    """不可变 Delivery 如何满足一个当前 TaskNode 主体。"""

    DIRECT = "direct"
    CARRIED = "carried"


class TaskNodeDeliveryCarryAuthority(_VerificationContract):
    """用于复用先前修订 Delivery 的已认证回执权威信息。"""

    schema_version: Literal["task-node-delivery-carry-authority-v1"] = (
        "task-node-delivery-carry-authority-v1"
    )
    carry_receipt_id: str = Field(min_length=1)
    carry_receipt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    apply_id: str = Field(min_length=1)
    source_delivery_id: str = Field(min_length=1)
    base_task_graph_revision: int = Field(ge=1)
    target_subject: TaskNodeSubject

    @model_validator(mode="after")
    def _validate_adjacent_revision(self) -> 'TaskNodeDeliveryCarryAuthority':
        if self.target_subject.graph_revision != self.base_task_graph_revision + 1:
            raise ValueError("carry authority must bind one adjacent target revision")
        return self


class ResolvedCurrentTaskNodeDelivery(_VerificationContract):
    """经认证可满足一个当前图主体的来源 Delivery。

    沿用投影绝不改写不可变来源 Delivery 的主体，而是保留该历史主体，并通过直接沿用回执
    绑定当前目标。如果来源早于直接基础版本，则 Store 已递归认证其间每个当前交付投影，
    直至同一个不可变来源 Delivery。
    """

    target_subject: TaskNodeSubject
    source_delivery: ResolvedTaskNodeDelivery
    resolution_kind: CurrentTaskNodeDeliveryResolutionKind
    carry_authority: TaskNodeDeliveryCarryAuthority | None = None

    @model_validator(mode="after")
    def _validate_resolution(self) -> 'ResolvedCurrentTaskNodeDelivery':
        source = self.source_delivery.delivery
        if self.resolution_kind is CurrentTaskNodeDeliveryResolutionKind.DIRECT:
            if self.carry_authority is not None or source.subject != self.target_subject:
                raise ValueError("direct current Delivery must own the target subject")
            return self

        authority = self.carry_authority
        if authority is None:
            raise ValueError("carried current Delivery requires carry authority")
        if (
            authority.target_subject != self.target_subject
            or authority.source_delivery_id != source.delivery_id
            or source.subject.task_id != self.target_subject.task_id
            or source.subject.graph_revision > authority.base_task_graph_revision
            or source.subject.node_id != self.target_subject.node_id
            or source.subject.node_revision != self.target_subject.node_revision
        ):
            raise ValueError("carried current Delivery binding is inconsistent")
        return self

    @classmethod
    def direct(
        cls,
        source_delivery: ResolvedTaskNodeDelivery,
    ) -> 'ResolvedCurrentTaskNodeDelivery':
        return cls(
            target_subject=source_delivery.delivery.subject,
            source_delivery=source_delivery,
            resolution_kind=CurrentTaskNodeDeliveryResolutionKind.DIRECT,
        )

    @property
    def delivery_id(self) -> str:
        return self.source_delivery.delivery.delivery_id


class TaskNodeVerificationMutationResult(_VerificationContract):
    """用于验证生命周期变更、可安全写入回执的小型结果。"""

    status: Literal["applied", "replayed"]
    verification_request_id: str = Field(min_length=1)
    verification_request_revision: int = Field(ge=1)
    verification_request_status: TaskNodeVerificationRequestStatus
    work_run_id: str = Field(min_length=1)
    work_run_revision: int = Field(ge=1)
    work_run_status: WorkRunStatus
    work_run_reason: str | None = None
    window_state_version: int = Field(ge=1)
    task_state_version: int | None = Field(default=None, ge=1)
    node_state_version: int | None = Field(default=None, ge=1)
    all_pass: bool | None = None
    delivery_id: str | None = None
    auxiliary_completion_id: str | None = None
    # 默认值 None 可保留在活跃时间结算接入验证生命周期变更之前写入的回执。
    budget_transition: WorkRunBudgetTransition | None = None

    @model_validator(mode="after")
    def _validate_lifecycle_projection(self) -> 'TaskNodeVerificationMutationResult':
        if self.verification_request_status is TaskNodeVerificationRequestStatus.PENDING:
            valid = (
                self.work_run_status is WorkRunStatus.ACTIVE
                and self.work_run_reason == "verification_pending"
                and self.all_pass is None
                and self.delivery_id is None
                and self.auxiliary_completion_id is None
            )
        elif self.verification_request_status is TaskNodeVerificationRequestStatus.INTERRUPTED:
            valid = (
                (
                    (
                        self.work_run_status is WorkRunStatus.INTERRUPTED
                        and self.work_run_reason == "verification_technical_failure"
                    )
                    or (
                        self.work_run_status is WorkRunStatus.FAILED
                        and self.work_run_reason == "work_run_limit_reached"
                        and self.budget_transition is not None
                        and self.budget_transition.disposition
                        is WorkRunBudgetDisposition.HARD_LIMIT_REACHED
                    )
                )
                and self.all_pass is None
                and self.delivery_id is None
                and self.auxiliary_completion_id is None
            )
        elif self.all_pass is True:
            valid = (
                self.work_run_status is WorkRunStatus.COMPLETED
                and self.work_run_reason == "verification_passed"
                and (
                    (self.delivery_id is not None
                        and self.auxiliary_completion_id is None)
                    or (self.delivery_id is None
                        and self.auxiliary_completion_id is not None)
                )
            )
        else:
            valid = (
                self.all_pass is False
                and (
                    (
                        self.work_run_status is WorkRunStatus.ACTIVE
                        and self.work_run_reason is None
                    )
                    or (
                        self.work_run_status is WorkRunStatus.TURN_LIMIT_REACHED
                        and self.work_run_reason == "turn_limit_reached"
                        and self.budget_transition is not None
                        and self.budget_transition.disposition
                        is WorkRunBudgetDisposition.SOFT_LIMIT_REACHED
                    )
                )
                and self.delivery_id is None
                and self.auxiliary_completion_id is None
            )
        if not valid:
            raise ValueError("verification mutation lifecycle projection is inconsistent")
        return self


__all__ = [
    'AcceptanceVerificationFeedback',
    'NodeVerificationResult',
    'PreparedTaskNodeVerification',
    'ResolvedTaskNodeDelivery',
    'SupportingToolResult',
    'TaskNodeDelivery',
    'TaskNodeVerificationMutationResult',
    'TaskNodeVerificationRecord',
    'TaskNodeVerificationRequestStatus',
    'TaskNodeVerificationRequest',
    'VerificationVerdict',
]
