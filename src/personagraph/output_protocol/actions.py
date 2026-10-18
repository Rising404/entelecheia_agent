"""跨 lane 共享的工具调用动作与验收更新。

这些对象只表达模型“希望 Host 做什么”，不执行工具，也不直接写数据库。
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..persistent_turn_content.json_values import freeze_json


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ToolCallProposal(_Contract):
    """模型提出的工具调用；真实 ID、权限和结果仍由 Host 决定。"""

    tool_id: str = Field(min_length=1)
    arguments: dict[str, Any]

    @field_validator("arguments")
    @classmethod
    def _freeze_arguments(cls, value: dict[str, Any]) -> dict[str, Any]:
        return freeze_json(value)


class CallToolsAction(_Contract):
    kind: Literal["call_tools"] = "call_tools"
    calls: tuple[ToolCallProposal, ...] = Field(min_length=1)


__all__ = [
    "CallToolsAction",
    "ToolCallProposal",
]
