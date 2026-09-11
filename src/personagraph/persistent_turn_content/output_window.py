"""当前候选交付正文的可恢复状态。

``OutputWindow`` 是 Host 接受模型动作后保存的完整正文，不是模型动作本身，也不是
执行笔记。正文只允许整体替换；验证通过前它仍是候选内容，通过后由持久化层冻结为
正式交付。物理存储和冻结事务不属于本模块。
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class OutputWindowFormat(StrEnum):
    PLAIN_TEXT = "plain_text"
    MARKDOWN = "markdown"


class OutputWindow(_Contract):
    """一份带修订来源的完整候选正文；每次更新替换整份内容。"""

    schema_version: Literal["output-window-v1"] = "output-window-v1"
    work_run_id: str = Field(min_length=1)
    output_revision: int = Field(default=1, ge=1)
    format: OutputWindowFormat = OutputWindowFormat.PLAIN_TEXT
    content: str = ""
    updated_turn_id: str = Field(min_length=1)
    updated_attempt_id: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def _validate_update_provenance(self) -> 'OutputWindow':
        if self.output_revision == 1:
            if self.format != OutputWindowFormat.PLAIN_TEXT or self.content:
                raise ValueError("OutputWindow revision one must be canonical and empty")
            if self.updated_attempt_id is not None:
                raise ValueError("canonical OutputWindow revision one precedes any Attempt")
        elif self.updated_attempt_id is None:
            raise ValueError("a changed OutputWindow requires Attempt provenance")
        return self


def initialize_output_window(*, work_run_id: str, updated_turn_id: str) -> OutputWindow:
    """创建尚未经过任何 Attempt 修改的规范空窗口。"""

    return OutputWindow(work_run_id=work_run_id, updated_turn_id=updated_turn_id)


__all__ = [
    "OutputWindow",
    "OutputWindowFormat",
    "initialize_output_window",
]
