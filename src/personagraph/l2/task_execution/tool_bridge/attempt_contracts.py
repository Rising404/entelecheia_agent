"""Attempt 控制器与 Tool Bridge 共享的稳定请求契约。

这里的值描述模型 ``call_tools`` 提案前后的两次交接。它们刻意不感知由哪个控制器提出请求，也不感知由哪个具体 Bridge 验证、执行或持久化这些调用。
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ....tools.contracts import ToolSpec
from ...work_run import (
    AttemptDecision,
    AttemptStatus,
    CallToolsAction,
    HostAcceptedAttemptDecision,
    HostMaterializedCallToolsAction,
    WorkRunStatus,
)
from ...work_run.mutation_receipt_contracts import WorkExecutionMutationResult


class _AttemptToolBridgeContract(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        arbitrary_types_allowed=True,
    )


class AttemptToolBridgePreflightRequest(_AttemptToolBridgeContract):
    """纯 call_tools 预检在模型输出验证内部执行。"""

    session_id: str = Field(min_length=1, max_length=200)
    turn_id: str = Field(min_length=1, max_length=200)
    work_run_id: str = Field(min_length=1, max_length=200)
    attempt_id: str = Field(min_length=1, max_length=200)
    decision: AttemptDecision
    tool_call_ids: tuple[str, ...] = Field(min_length=1)
    allowed_tools: tuple[ToolSpec, ...] = ()
    catalog_snapshot: dict[str, object]

    @model_validator(mode="after")
    def _require_call_tools(self) -> 'AttemptToolBridgePreflightRequest':
        if not isinstance(self.decision.action, CallToolsAction):
            raise ValueError("Tool Bridge preflight accepts only call_tools decisions")
        if len(self.tool_call_ids) != len(self.decision.action.calls):
            raise ValueError("tool_call_ids must cover the proposal exactly")
        if len(self.tool_call_ids) != len(set(self.tool_call_ids)):
            raise ValueError("tool_call_ids must be unique")
        return self


class AttemptToolBridgeRequest(_AttemptToolBridgeContract):
    """交给具体 Tool Bridge 的有界执行请求。

    Bridge 负责 Host 物化、效果与参数检查、执行及 ToolResult 持久化。
    此请求只能携带同一逻辑模型请求内纯预检产生且已获 Host 接纳的结果。
    稳定的 ``apply_id`` 由控制器的调用者提供。
    """

    session_id: str = Field(min_length=1, max_length=200)
    turn_id: str = Field(min_length=1, max_length=200)
    work_run_id: str = Field(min_length=1, max_length=200)
    attempt_id: str = Field(min_length=1, max_length=200)
    expected_work_run_revision: int = Field(ge=1)
    expected_progress_revision: int = Field(ge=1)
    expected_output_revision: int = Field(ge=1)
    expected_window_revision: int = Field(ge=1)
    apply_id: str = Field(min_length=1, max_length=200)
    decision: HostAcceptedAttemptDecision
    allowed_tools: tuple[ToolSpec, ...] = ()
    catalog_snapshot: dict[str, object]
    recovery_mutation: WorkExecutionMutationResult | None = None

    @model_validator(mode="after")
    def _require_call_tools(self) -> 'AttemptToolBridgeRequest':
        if not isinstance(self.decision.action, HostMaterializedCallToolsAction):
            raise ValueError("Tool Bridge accepts only Host-materialized call_tools decisions")
        recovery = self.recovery_mutation
        if recovery is not None and (
            recovery.work_run_id != self.work_run_id
            or recovery.current_attempt_id != self.attempt_id
            or recovery.work_run_revision != self.expected_work_run_revision
            or recovery.acceptance_progress_revision
            != self.expected_progress_revision
            or recovery.output_window_revision != self.expected_output_revision
            or recovery.window_state_version != self.expected_window_revision
            or recovery.work_run_status is not WorkRunStatus.ACTIVE
            or recovery.work_run_reason is not None
            or recovery.attempt is None
            or recovery.attempt.attempt_id != self.attempt_id
            or recovery.attempt.status is not AttemptStatus.ACTIVE
        ):
            raise ValueError(
                "Tool Bridge recovery mutation must bind the exact active Attempt cursor"
            )
        return self


__all__ = [
    'AttemptToolBridgePreflightRequest',
    'AttemptToolBridgeRequest',
]
