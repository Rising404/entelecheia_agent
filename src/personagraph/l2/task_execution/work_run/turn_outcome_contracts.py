"""可冷导入的单个 TaskNode WorkRun 轮次停止契约。

这些值描述控制器的内部终态或中断投影。它们不会构造 Attempt 上下文、读取存储
权威、调用模型、执行工具或结算 WorkRun。
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

WorkRunTurnOutcome = Literal[
    "delivery_ready",
    "task_graph_revision_requested",
    "waiting_user",
    "waiting_external",
    "turn_limit_reached",
    "work_run_failed",
    "verification_interrupted",
    "internal_interrupted",
    "failed_closed",
]

WorkRunTurnFailureCode = Literal[
    "missing_authority_cas",
    "turn_window_stale",
    "task_details_unavailable",
    "task_authority_stale",
    "node_authority_stale",
    "node_not_ready",
    "task_not_linked",
    "existing_nonterminal_work_run",
    "catalog_projection_mismatch",
    "tool_bridge_unavailable",
    "work_run_create_failed",
    "active_attempt_resume_failed",
    "unprepared_verification_recovery_failed",
    "verification_recovery_failed",
    "waiting_user_continuation_failed",
]

class _OutcomeContract(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        arbitrary_types_allowed=True,
    )


class WorkRunTurnAuthorityUnavailable(RuntimeError):
    """原始轮次已不再持有可读的中断游标。"""


class WorkRunTurnApplicationResult(_OutcomeContract):
    """内部停止结果；绝不包含 OutputWindow 正文。

    ``internal_interrupted`` 不是 WorkRun 终态结果。失败后已重新读取其窗口版本，
    持有该轮次的编排器必须使用这一精确版本，在交接前将轮次标记为已中断。
    """

    outcome: WorkRunTurnOutcome
    work_run_id: str | None = Field(default=None, min_length=1, max_length=200)
    current_attempt_id: str | None = Field(default=None, min_length=1, max_length=200)
    delivery_id: str | None = Field(default=None, min_length=1, max_length=200)
    pending_user_question: str | None = Field(default=None, min_length=1)
    window_revision: int = Field(ge=1)
    failure_code: WorkRunTurnFailureCode | None = None
    interruption_reason: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def _require_outcome_payload(self) -> 'WorkRunTurnApplicationResult':
        if self.outcome == "delivery_ready":
            if self.work_run_id is None or self.delivery_id is None:
                raise ValueError("delivery_ready requires WorkRun and delivery IDs")
        elif self.delivery_id is not None:
            raise ValueError("only delivery_ready may carry a delivery ID")
        if self.outcome == "waiting_user":
            if self.pending_user_question is None or self.work_run_id is None:
                raise ValueError("waiting_user requires a WorkRun question")
        elif self.pending_user_question is not None:
            raise ValueError("only waiting_user may carry a question")
        if self.outcome in {
            "turn_limit_reached",
            "work_run_failed",
            "task_graph_revision_requested",
        }:
            if self.work_run_id is None:
                raise ValueError("budget stop requires a WorkRun ID")
        if (self.outcome == "failed_closed") is (self.failure_code is None):
            raise ValueError("only failed_closed requires a failure code")
        if self.outcome in {"verification_interrupted", "internal_interrupted"}:
            if self.interruption_reason is None or self.work_run_id is None:
                raise ValueError("interruption requires durable WorkRun bindings")
        elif self.interruption_reason is not None:
            raise ValueError("only an interruption carries its reason")
        return self


def _failed(
    code: WorkRunTurnFailureCode,
    *,
    window_revision: int,
    work_run_id: str | None = None,
    current_attempt_id: str | None = None,
) -> WorkRunTurnApplicationResult:
    return WorkRunTurnApplicationResult(
        outcome="failed_closed",
        work_run_id=work_run_id,
        current_attempt_id=current_attempt_id,
        window_revision=window_revision,
        failure_code=code,
    )


__all__ = [
    'WorkRunTurnApplicationResult',
    "WorkRunTurnAuthorityUnavailable",
    'WorkRunTurnFailureCode',
    'WorkRunTurnOutcome',
]
