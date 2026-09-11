"""仅供 L2 WorkRun 使用的持久 OutputWindow 动作。"""

from __future__ import annotations

from typing import Literal

from pydantic import field_validator

from ..persistent_turn_content.output_window import OutputWindowFormat
from .actions import _Contract


class WriteOutputWindowAction(_Contract):
    """请求整体替换 L2 WorkRun 的候选正文但暂不提交验证。"""

    kind: Literal["write_output_window"] = "write_output_window"
    content: str
    format: OutputWindowFormat


class SubmitOutputWindowAction(_Contract):
    """请求以一份非空完整正文进入 L2 WorkRun 交付前验证。"""

    kind: Literal["submit_output_window"] = "submit_output_window"
    content: str
    format: OutputWindowFormat

    @field_validator("content")
    @classmethod
    def _require_nonempty_content(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("a submitted OutputWindow must not be empty")
        return value


__all__ = ["SubmitOutputWindowAction", "WriteOutputWindowAction"]
