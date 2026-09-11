"""执行层之间共享的冻结精确重放回执契约。

持久化层会在一次持久 WorkRun 变更后创建这些回执。Runtime 可以使用其中经过验证的游标和
重放状态，但这两项职责都不属于值对象本身。
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .budget import WorkRunBudgetTransition
from .contracts import Attempt, WorkRunStatus


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class WorkExecutionMutationResult(_Contract):
    """小型精确重放回执投影；绝不嵌入瞬态载荷。"""

    status: Literal["applied", "replayed"]
    work_run_id: str
    work_run_revision: int = Field(ge=1)
    work_run_status: WorkRunStatus
    work_run_reason: str | None = None
    current_attempt_id: str | None = None
    acceptance_progress_revision: int = Field(ge=1)
    output_window_revision: int = Field(ge=1)
    attempt: Attempt | None = None
    new_tool_call_ids: tuple[str, ...] = ()
    tool_result_id: str | None = None
    turn_work_run_link_revision: int | None = Field(default=None, ge=0)
    window_state_version: int = Field(ge=1)
    task_state_version: int | None = Field(default=None, ge=1)
    node_state_version: int | None = Field(default=None, ge=1)
    budget_transition: WorkRunBudgetTransition | None


__all__ = ['WorkExecutionMutationResult']
