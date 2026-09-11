"""不可变 AuxiliaryGraph 执行 frontier 契约。

持久化层从 SQLite 投影经认证 frontier，Runtime reducer 使用它精确选择一个下一动作。这些值
对象属于 AuxiliaryGraph 领域，因为它们描述图执行状态，却不查询存储或执行调度。
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from personagraph.l2.work_run.contracts import AuxiliaryNodeSubject, WorkRunStatus
from .contracts import (
    AuxiliaryNodeExecutorKind,
    AuxiliaryNodeReference,
    AuxiliaryPlanningGoalStatus,
    PlanningEpisodeBudgetDisposition,
)


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ReadyAuxiliaryNodeExecutionCandidate(_Contract):
    """一个可跨越 executor 接缝的新鲜当前 revision 节点。"""

    subject: AuxiliaryNodeSubject
    ordinal: int = Field(ge=0)
    local_node_key: str = Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")
    executor_kind: AuxiliaryNodeExecutorKind
    capability_profile_id: str | None
    node_state_version: int = Field(ge=1)
    dependency_completion_ids: tuple[str, ...] = ()


class RecoverableAuxiliaryNodeExecutionCandidate(_Contract):
    """一个已绑定到当前 节点的精确非终止 WorkRun。"""

    subject: AuxiliaryNodeSubject
    ordinal: int = Field(ge=0)
    local_node_key: str = Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")
    executor_kind: AuxiliaryNodeExecutorKind
    capability_profile_id: str | None
    node_status: Literal[
        "active",
        "waiting_user",
        "waiting_authorization",
        "waiting_external",
        "interrupted",
    ]
    node_state_version: int = Field(ge=1)
    work_run_id: str
    work_run_status: WorkRunStatus
    work_run_reason: str | None = None
    work_run_revision: int = Field(ge=1)
    current_attempt_id: str | None = None
    current_verification_request_id: str | None = None
    current_verification_request_revision: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def _validate_verification_cursor(
        self,
    ) -> "RecoverableAuxiliaryNodeExecutionCandidate":
        if (self.current_verification_request_id is None) != (
            self.current_verification_request_revision is None
        ):
            raise ValueError(
                "verification request identity and revision must be projected together"
            )
        return self


class RecoverableAuxiliaryPrimitiveInvocationCandidate(_Contract):
    """一个必须在新分派前结算的已预留 Host primitive。"""

    subject: AuxiliaryNodeSubject
    ordinal: int = Field(ge=0)
    local_node_key: str = Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")
    executor_kind: Literal[AuxiliaryNodeExecutorKind.HOST_PRIMITIVE] = (
        AuxiliaryNodeExecutorKind.HOST_PRIMITIVE
    )
    capability_profile_id: str
    node_state_version: int = Field(ge=1)
    primitive_call_id: str
    primitive_kind: Literal["resource_perception"]
    invocation_turn_id: str
    state_guard_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class AuxiliaryGraphExecutionFrontier(_Contract):
    """一个事务冻结的当前 revision 执行 frontier。

    投影按图序号顺序报告每个就绪 candidate，但不授予调度选择。非终止 WorkRun 会抑制所有
    新 candidate，因此外层 driver 必须恢复精确持久 cursor，才能分派其他内容。
    """

    session_id: str
    turn_id: str
    task_id: str
    auxiliary_graph_id: str
    goal_id: str
    auxiliary_graph_revision: int = Field(ge=1)
    terminal_node_id: str
    control_state_version: int = Field(ge=1)
    goal_state_version: int = Field(ge=1)
    revision_state_version: int = Field(ge=1)
    budget_state_version: int = Field(ge=1)
    base_task_graph_revision: int | None = Field(default=None, ge=1)
    target_task_graph_revision: int = Field(ge=1)
    task_state_version: int = Field(ge=1)
    authority_snapshot_id: str
    authority_snapshot_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    structure_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    budget_snapshot_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    budget_disposition: PlanningEpisodeBudgetDisposition
    goal_status: AuxiliaryPlanningGoalStatus
    revision_status: Literal[
        "active",
        "waiting_user",
        "waiting_authorization",
        "waiting_external",
        "interrupted",
        "proposal_ready",
        "gapped_ready",
        "committed",
        "failed",
        "cancelled",
        "superseded",
        "budget_exhausted",
    ]
    ready_fresh: tuple[ReadyAuxiliaryNodeExecutionCandidate, ...] = ()
    recoverable: tuple[RecoverableAuxiliaryNodeExecutionCandidate, ...] = ()
    recoverable_primitive: tuple[
        RecoverableAuxiliaryPrimitiveInvocationCandidate, ...
    ] = ()
    completed_node_refs: tuple[AuxiliaryNodeReference, ...] = ()
    blocking_node_refs: tuple[AuxiliaryNodeReference, ...] = ()

    @model_validator(mode="after")
    def _validate_single_cursor(self) -> "AuxiliaryGraphExecutionFrontier":
        if len(self.recoverable) > 1:
            raise ValueError("AuxiliaryGraph has more than one durable cursor")
        if len(self.recoverable_primitive) > 1:
            raise ValueError("AuxiliaryGraph has more than one primitive cursor")
        if self.recoverable and self.recoverable_primitive:
            raise ValueError("WorkRun and primitive cursors cannot coexist")
        if (self.recoverable or self.recoverable_primitive) and self.ready_fresh:
            raise ValueError("a recoverable cursor must suppress fresh dispatch")
        return self


__all__ = (
    "AuxiliaryGraphExecutionFrontier",
    "ReadyAuxiliaryNodeExecutionCandidate",
    "RecoverableAuxiliaryNodeExecutionCandidate",
    "RecoverableAuxiliaryPrimitiveInvocationCandidate",
)
