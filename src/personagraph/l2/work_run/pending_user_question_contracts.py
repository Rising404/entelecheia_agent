"""执行层之间共享的冻结待处理用户问题投影。

此值携带继续一个等待中 TaskNode WorkRun 所需的精确过期防线。持久化层根据持久事实构造它；
Runtime 只把它作为已经验证的 Host 权威信息使用。
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from .contracts import ExecutionSubject


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PendingUserQuestion(_Contract):
    """仅供 Host 使用的过期防线，以及实际待处理交互内容。"""

    session_id: str = Field(min_length=1, max_length=200)
    work_run_id: str = Field(min_length=1, max_length=200)
    work_run_revision: int = Field(ge=1)
    subject: ExecutionSubject
    question_attempt_id: str = Field(min_length=1, max_length=200)
    question_attempt_ordinal: int = Field(ge=1)
    question_turn_id: str = Field(min_length=1, max_length=200)
    question: str = Field(min_length=1, max_length=2000)
    task_state_version: int = Field(ge=1)
    node_state_version: int = Field(ge=1)
    acceptance_progress_revision: int = Field(ge=1)
    output_window_revision: int = Field(ge=1)


__all__ = ['PendingUserQuestion']
