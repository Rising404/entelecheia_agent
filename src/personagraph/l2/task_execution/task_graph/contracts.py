"""确定性 TaskGraph WorkRun 执行的冷请求/结果契约。

这些值绑定一次 TaskGraph 驱动器调用及其紧凑停止结果。它们不选择节点、不读取 Session 状态、
不创建 WorkRun、不调用模型，也不执行工具。
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from personagraph.l2.task_execution.work_run.turn_outcome_contracts import (
    WorkRunTurnApplicationResult,
)
from ..paper_prompt_context import PaperAttemptContext

from .profile import TaskGraphWorkRunProfile


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class TaskGraphWorkRunRequest(_Contract):
    session_id: str = Field(min_length=1, max_length=200)
    turn_id: str = Field(min_length=1, max_length=200)
    task_id: str = Field(min_length=1, max_length=200)
    expected_window_revision: int = Field(ge=1)
    profile: TaskGraphWorkRunProfile = Field(
        default_factory=TaskGraphWorkRunProfile
    )
    allow_user_input: bool = True
    initial_work_run_result: WorkRunTurnApplicationResult | None = None
    paper_resources: PaperAttemptContext | None = None

    @model_validator(mode="after")
    def _bind_paper_resources(self) -> 'TaskGraphWorkRunRequest':
        if self.paper_resources is None:
            return self
        if self.paper_resources.session_id != self.session_id:
            raise ValueError("paper resources belong to another Session")
        if self.paper_resources.task_id != self.task_id:
            raise ValueError("paper resources belong to another Task")
        return self


TaskGraphWorkRunStatus = Literal[
    "completed",
    "revision_required",
    "waiting_user",
    "waiting_external",
    "turn_limit_reached",
    "work_run_failed",
    "blocked",
    "deadline_reached",
    "verification_interrupted",
    "internal_interrupted",
    "failed_closed",
]


class TaskGraphWorkRunResult(_Contract):
    status: TaskGraphWorkRunStatus
    task_id: str = Field(min_length=1, max_length=200)
    final_delivery_id: str | None = Field(default=None, min_length=1, max_length=200)
    pending_question_attempt_ids: tuple[str, ...] = Field(default=(), max_length=64)
    work_run_ids: tuple[str, ...] = Field(default=(), max_length=64)
    window_state_version: int = Field(ge=1)
    last_work_run_outcome: str | None = Field(
        default=None,
        min_length=1,
        max_length=80,
    )
    failure_code: str | None = Field(default=None, min_length=1, max_length=160)
    interruption_reason: str | None = Field(default=None, min_length=1, max_length=200)

    @model_validator(mode="after")
    def _validate_shape(self) -> 'TaskGraphWorkRunResult':
        if len(self.pending_question_attempt_ids) != len(
            set(self.pending_question_attempt_ids)
        ):
            raise ValueError("pending question Attempt IDs must be unique")
        if len(self.work_run_ids) != len(set(self.work_run_ids)):
            raise ValueError("TaskGraph WorkRun IDs must be unique")
        if (self.status == "completed") is (self.final_delivery_id is None):
            raise ValueError("only completed TaskGraph results carry final Delivery")
        if self.status == "failed_closed" and self.failure_code is None:
            raise ValueError("failed_closed requires its typed failure code")
        if self.status != "failed_closed" and self.failure_code is not None:
            raise ValueError("only failed_closed carries a failure code")
        if self.status in {"verification_interrupted", "internal_interrupted"}:
            if self.interruption_reason is None:
                raise ValueError("interrupted TaskGraph result requires a reason")
        elif self.interruption_reason is not None:
            raise ValueError("only interrupted TaskGraph result carries a reason")
        return self


__all__ = (
    'TaskGraphWorkRunRequest',
    'TaskGraphWorkRunResult',
    'TaskGraphWorkRunStatus',
)
