"""WorkRun 执行域的纯契约。

此包特意不依赖 Session 持久化、Runtime 或提供商 SDK。这些模型描述相应层之后可以存储或
传输的值；它们不执行 I/O 或生命周期编排。
"""

from __future__ import annotations

import hashlib
import json
import math
from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ...output_protocol import (
    AcceptanceUpdate,
    CallToolsAction,
    SubmitOutputWindowAction,
    ToolCallProposal,  # noqa: F401
    WriteOutputWindowAction,
)
from ...persistent_turn_content import (
    AcceptanceProgressItem,
    OutputWindow,  # noqa: F401
    OutputWindowFormat,
    initialize_output_window,  # noqa: F401
)
from ...persistent_turn_content.json_values import freeze_json

class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


_freeze_json = freeze_json


class WorkRunStatus(StrEnum):
    ACTIVE = "active"
    PAUSED = "paused"
    WAITING_USER = "waiting_user"
    WAITING_AUTHORIZATION = "waiting_authorization"
    WAITING_EXTERNAL = "waiting_external"
    TURN_LIMIT_REACHED = "turn_limit_reached"
    INTERRUPTED = "interrupted"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AttemptStatus(StrEnum):
    ACTIVE = "active"
    CLOSED = "closed"


class ToolResultStatus(StrEnum):
    SUCCEEDED = "succeeded"
    REJECTED = "rejected"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    CANCELLED = "cancelled"
    COMPLETION_UNCONFIRMED = "completion_unconfirmed"


class TaskGraphExecutionReplanReason(StrEnum):
    """普通 TaskNode 无法在当前图下完成的原因。"""

    TASK_DECOMPOSITION_INCOMPLETE = "task_decomposition_incomplete"
    NODE_CONTRACT_INVALID = "node_contract_invalid"
    DEPENDENCY_STRUCTURE_INVALID = "dependency_structure_invalid"
    CAPABILITY_ASSIGNMENT_INVALID = "capability_assignment_invalid"


class WorkRunBudget(_Contract):
    """由 Host 持有的预算账本；它绝不是 AttemptDecision 的一部分。"""

    max_attempts: int = Field(default=32, ge=1)
    soft_active_seconds: float = Field(default=720, ge=0)
    hard_active_seconds: float = Field(default=900, ge=0)
    attempts_started: int = Field(default=0, ge=0)
    active_seconds_consumed: float = Field(default=0, ge=0)

    @field_validator(
        "soft_active_seconds",
        "hard_active_seconds",
        "active_seconds_consumed",
    )
    @classmethod
    def _require_finite_active_time(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("active-time budget values must be finite")
        return value

    @model_validator(mode="after")
    def _validate_budget_ledger(self) -> 'WorkRunBudget':
        if self.soft_active_seconds > self.hard_active_seconds:
            raise ValueError("soft active-time limit cannot exceed the hard limit")
        if self.attempts_started > self.max_attempts:
            raise ValueError("attempt count cannot exceed max_attempts")
        return self


class TaskNodeSubject(_Contract):
    kind: Literal["task_node"] = "task_node"
    task_id: str = Field(min_length=1)
    graph_revision: int = Field(ge=1)
    node_id: str = Field(min_length=1)
    node_revision: int = Field(ge=1)


class AuxiliaryNodeSubject(_Contract):
    kind: Literal["auxiliary_node"] = "auxiliary_node"
    task_id: str = Field(min_length=1)
    auxiliary_graph_id: str = Field(min_length=1)
    auxiliary_graph_revision: int = Field(ge=1)
    node_id: str = Field(min_length=1)
    node_revision: int = Field(ge=1)


ExecutionSubject = Annotated[
    TaskNodeSubject | AuxiliaryNodeSubject,
    Field(discriminator="kind"),
]


class WorkRun(_Contract):
    """一个执行实例；新创建的值已经处于活跃状态。"""

    schema_version: Literal["work-run-v2"] = "work-run-v2"
    work_run_id: str = Field(min_length=1)
    subject: ExecutionSubject
    revision: int = Field(default=1, ge=1)
    status: WorkRunStatus = WorkRunStatus.ACTIVE
    reason: str | None = Field(default=None, min_length=1, max_length=160)
    budget: WorkRunBudget = Field(default_factory=WorkRunBudget)

    @model_validator(mode="after")
    def _require_initial_revision_to_be_active(self) -> 'WorkRun':
        if self.revision == 1 and self.status is not WorkRunStatus.ACTIVE:
            raise ValueError("an initial-revision WorkRun must be active")
        return self


def create_work_run(*, work_run_id: str, subject: ExecutionSubject) -> WorkRun:
    """使用由 Host 持有的固定状态和预算默认值创建 WorkRun。"""

    return WorkRun(work_run_id=work_run_id, subject=subject)


class Attempt(_Contract):
    attempt_id: str = Field(min_length=1)
    work_run_id: str = Field(min_length=1)
    ordinal: int = Field(ge=1)
    status: AttemptStatus = AttemptStatus.ACTIVE
    submitted_output_revision: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def _require_submit_lock_to_be_closed(self) -> 'Attempt':
        if (
            self.submitted_output_revision is not None
            and self.status is not AttemptStatus.CLOSED
        ):
            raise ValueError("only a closed submit Attempt may lock an output revision")
        return self


class AcceptanceProgressSnapshot(_Contract):
    work_run_id: str = Field(min_length=1)
    subject: ExecutionSubject
    revision: int = Field(default=1, ge=1)
    evaluated_output_revision: int = Field(ge=1)
    items: tuple[AcceptanceProgressItem, ...] = Field(min_length=1)

    @field_validator("items")
    @classmethod
    def _require_unique_acceptance_ids(
        cls, values: tuple[AcceptanceProgressItem, ...]
    ) -> tuple[AcceptanceProgressItem, ...]:
        ids = [item.acceptance_id for item in values]
        if len(ids) != len(set(ids)):
            raise ValueError("Acceptance progress IDs must be unique")
        return values


class HostMaterializedToolCall(_Contract):
    """由 Host 物化的调用；ID、版本和效果均为 Host 权威信息。"""

    tool_call_id: str = Field(min_length=1)
    tool_id: str = Field(min_length=1)
    tool_version: str = Field(min_length=1)
    arguments: dict[str, Any]
    modifies_environment: bool

    @field_validator("arguments")
    @classmethod
    def _freeze_arguments(cls, value: dict[str, Any]) -> dict[str, Any]:
        return _freeze_json(value)


class HostMaterializedCallToolsAction(_Contract):
    kind: Literal["call_tools"] = "call_tools"
    calls: tuple[HostMaterializedToolCall, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_call_batch(self) -> 'HostMaterializedCallToolsAction':
        call_ids = [call.tool_call_id for call in self.calls]
        if len(call_ids) != len(set(call_ids)):
            raise ValueError("tool_call_id must be unique within an Attempt")
        if sum(call.modifies_environment for call in self.calls) > 1:
            raise ValueError("an Attempt may contain at most one environment-modifying call")
        return self


class HostMaterializedWriteOutputWindowAction(_Contract):
    """指向一个整体替换后 OutputWindow 修订的持久小型引用。"""

    kind: Literal["write_output_window"] = "write_output_window"
    work_run_id: str = Field(min_length=1)
    output_revision: int = Field(ge=1)
    format: OutputWindowFormat
    size_bytes: int = Field(ge=0)


class HostMaterializedSubmitOutputWindowAction(_Contract):
    """指向已提交验证之精确修订的持久小型引用。"""

    kind: Literal["submit_output_window"] = "submit_output_window"
    work_run_id: str = Field(min_length=1)
    output_revision: int = Field(ge=1)
    format: OutputWindowFormat
    size_bytes: int = Field(ge=1)


HostMaterializedOutputWindowAction = Annotated[
    HostMaterializedWriteOutputWindowAction
    | HostMaterializedSubmitOutputWindowAction,
    Field(discriminator="kind"),
]


class RequestUserInputAction(_Contract):
    kind: Literal["request_user_input"] = "request_user_input"
    question: str = Field(min_length=1, max_length=2000)


class RequestTaskGraphRevisionAction(_Contract):
    """由模型编写、用于替换当前 TaskGraph 快照的请求。

    其范围特意窄于普通的验证重试。它只能指出会导致当前 TaskNode 无法在冻结的节点、
    依赖及能力契约下完成的结构缺陷。
    """

    kind: Literal["request_task_graph_revision"] = "request_task_graph_revision"
    reason: TaskGraphExecutionReplanReason
    diagnosis: str = Field(min_length=1, max_length=4_000)
    revision_objective: str = Field(min_length=1, max_length=2_000)
    supporting_tool_result_ids: tuple[str, ...] = Field(default=(), max_length=64)

    @field_validator("supporting_tool_result_ids")
    @classmethod
    def _require_unique_supporting_results(
        cls,
        values: tuple[str, ...],
    ) -> tuple[str, ...]:
        if any(not value.strip() or len(value) > 200 for value in values):
            raise ValueError("supporting ToolResult IDs must be bounded identities")
        if len(values) != len(set(values)):
            raise ValueError("supporting ToolResult IDs must be unique")
        return values


AttemptAction = Annotated[
    CallToolsAction
    | WriteOutputWindowAction
    | SubmitOutputWindowAction
    | RequestUserInputAction
    | RequestTaskGraphRevisionAction,
    Field(discriminator="kind"),
]


class AttemptDecision(_Contract):
    """在 Attempt 开始时生成的唯一结构化决策。"""

    acceptance_updates: tuple[AcceptanceUpdate, ...] = ()
    action: AttemptAction

    @model_validator(mode="after")
    def _revision_request_cannot_mutate_progress(self) -> 'AttemptDecision':
        if (
            isinstance(self.action, RequestTaskGraphRevisionAction)
            and self.acceptance_updates
        ):
            raise ValueError(
                "TaskGraph revision requests cannot mutate Acceptance progress"
            )
        return self


HostAcceptedAttemptAction = Annotated[
    HostMaterializedCallToolsAction
    | HostMaterializedWriteOutputWindowAction
    | HostMaterializedSubmitOutputWindowAction
    | RequestUserInputAction
    | RequestTaskGraphRevisionAction,
    Field(discriminator="kind"),
]


class HostAcceptedAttemptDecision(_Contract):
    """经 Host 完成调用物化与效果推导后的模型决策。"""

    acceptance_updates: tuple[AcceptanceUpdate, ...] = ()
    action: HostAcceptedAttemptAction

    @model_validator(mode="after")
    def _revision_request_cannot_mutate_progress(
        self,
    ) -> 'HostAcceptedAttemptDecision':
        if (
            isinstance(self.action, RequestTaskGraphRevisionAction)
            and self.acceptance_updates
        ):
            raise ValueError(
                "TaskGraph revision requests cannot mutate Acceptance progress"
            )
        return self


class TaskGraphExecutionReplanRequest(_Contract):
    """针对一次精确 TaskGraph N -> N+1 转换的不可变执行期权威信息。"""

    schema_version: Literal["task-graph-execution-replan-request-v1"] = (
        "task-graph-execution-replan-request-v1"
    )
    request_id: str = Field(min_length=1, max_length=200)
    create_apply_id: str = Field(min_length=1, max_length=200)
    session_id: str = Field(min_length=1, max_length=200)
    task_id: str = Field(min_length=1, max_length=200)
    base_graph_revision: int = Field(ge=1)
    target_graph_revision: int = Field(ge=2)
    requesting_subject: TaskNodeSubject
    work_run_id: str = Field(min_length=1, max_length=200)
    attempt_id: str = Field(min_length=1, max_length=200)
    source_node_alias: str = Field(pattern=r"^base_node_[0-9]{3}$")
    reason: TaskGraphExecutionReplanReason
    diagnosis: str = Field(min_length=1, max_length=4_000)
    revision_objective: str = Field(min_length=1, max_length=2_000)
    supporting_tool_result_ids: tuple[str, ...] = Field(default=(), max_length=64)
    task_state_version: int = Field(ge=1)
    created_turn_id: str = Field(min_length=1, max_length=200)
    request_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def _validate_authority(self) -> 'TaskGraphExecutionReplanRequest':
        if self.target_graph_revision != self.base_graph_revision + 1:
            raise ValueError("execution replan target must be base revision + 1")
        if (
            self.requesting_subject.task_id != self.task_id
            or self.requesting_subject.graph_revision != self.base_graph_revision
        ):
            raise ValueError("execution replan subject crossed TaskGraph authority")
        if len(self.supporting_tool_result_ids) != len(
            set(self.supporting_tool_result_ids)
        ) or any(
            not value.strip() or len(value) > 200
            for value in self.supporting_tool_result_ids
        ):
            raise ValueError("execution replan support IDs are invalid")
        expected = hashlib.sha256(
            json.dumps(
                self.model_dump(mode="json", exclude={"request_sha256"}),
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        if self.request_sha256 != expected:
            raise ValueError("execution replan request hash is invalid")
        return self

    @classmethod
    def create(cls, **values: Any) -> 'TaskGraphExecutionReplanRequest':
        payload = dict(values)
        payload["request_sha256"] = "0" * 64
        provisional = cls.model_construct(**payload)
        payload["request_sha256"] = hashlib.sha256(
            json.dumps(
                provisional.model_dump(
                    mode="json",
                    exclude={"request_sha256"},
                ),
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        return cls.model_validate(payload)


class TaskGraphExecutionReplanApplication(_Contract):
    """通过相应 N+1 提交消费一个执行请求的不可变回执。"""

    schema_version: Literal["task-graph-execution-replan-application-v1"] = (
        "task-graph-execution-replan-application-v1"
    )
    apply_id: str = Field(min_length=1, max_length=200)
    request_id: str = Field(min_length=1, max_length=200)
    request_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    session_id: str = Field(min_length=1, max_length=200)
    task_id: str = Field(min_length=1, max_length=200)
    base_graph_revision: int = Field(ge=1)
    committed_graph_revision: int = Field(ge=2)
    task_graph_commit_apply_id: str = Field(min_length=1, max_length=200)
    consumed_turn_id: str = Field(min_length=1, max_length=200)
    receipt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def _validate_receipt(self) -> 'TaskGraphExecutionReplanApplication':
        if self.committed_graph_revision != self.base_graph_revision + 1:
            raise ValueError("execution replan application must commit base + 1")
        expected = hashlib.sha256(
            json.dumps(
                self.model_dump(mode="json", exclude={"receipt_sha256"}),
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        if self.receipt_sha256 != expected:
            raise ValueError("execution replan application hash is invalid")
        return self

    @classmethod
    def create(cls, **values: Any) -> 'TaskGraphExecutionReplanApplication':
        payload = dict(values)
        payload["receipt_sha256"] = "0" * 64
        provisional = cls.model_construct(**payload)
        payload["receipt_sha256"] = hashlib.sha256(
            json.dumps(
                provisional.model_dump(
                    mode="json",
                    exclude={"receipt_sha256"},
                ),
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        return cls.model_validate(payload)


class ToolResult(_Contract):
    """一个已物化 ToolCall 的不可变直接结果。

    状态会保留未来恢复所需的执行差异；特别是，``completion_unconfirmed`` 绝不会折叠为
    普通错误。该状态专用于尚无法证明已持久完成的有副作用 Operation；只读操作的超时或
    响应丢失结果仍是已知的 ``timed_out`` 或 ``failed``。``output`` 为必填项，
    但在没有直接输出时可以是 JSON null。
    """

    status: ToolResultStatus
    tool_result_id: str = Field(min_length=1)
    tool_call_id: str = Field(min_length=1)
    attempt_id: str = Field(min_length=1)
    ordinal: int = Field(ge=1)
    output: Any
    error_code: str | None = None
    error_message: str | None = None

    @field_validator("output")
    @classmethod
    def _freeze_output(cls, value: Any) -> Any:
        return _freeze_json(value)

    @model_validator(mode="after")
    def _validate_status_details(self) -> 'ToolResult':
        if self.status is ToolResultStatus.SUCCEEDED:
            if self.error_code is not None or self.error_message is not None:
                raise ValueError("a succeeded ToolResult cannot carry error details")
            return self
        if not self.error_code or not self.error_code.strip():
            raise ValueError("a non-succeeded ToolResult requires error_code")
        if not self.error_message or not self.error_message.strip():
            raise ValueError("a non-succeeded ToolResult requires error_message")
        return self
