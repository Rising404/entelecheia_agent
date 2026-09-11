"""跨 lane 共享的工具调用动作与验收更新。

这些对象只表达模型“希望 Host 做什么”，不执行工具，也不直接写数据库。
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator

from ..persistent_turn_content.evidence import EmptySupportJustification
from ..persistent_turn_content.acceptance import (
    validate_acceptance_empty_support,
    validate_acceptance_supporting_result_ids,
)
from ..persistent_turn_content.json_values import freeze_json


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class AcceptanceUpdate(_Contract):
    """请求完整替换一个明确标识的验收进度项。"""

    acceptance_id: str = Field(min_length=1)
    model_claimed_satisfied: bool
    supporting_tool_result_ids: tuple[str, ...] = ()
    empty_support_justification: EmptySupportJustification | None = None

    @field_validator("supporting_tool_result_ids")
    @classmethod
    def _validate_supporting_ids(
        cls, values: tuple[str, ...], info: ValidationInfo,
    ) -> tuple[str, ...]:
        return validate_acceptance_supporting_result_ids(
            values, claimed_satisfied=info.data.get("model_claimed_satisfied"),
        )

    @field_validator("empty_support_justification")
    @classmethod
    def _validate_empty_support_shape(
        cls, value: EmptySupportJustification | None, info: ValidationInfo,
    ) -> EmptySupportJustification | None:
        return validate_acceptance_empty_support(
            value, claimed_satisfied=info.data.get("model_claimed_satisfied"),
            supporting_ids=info.data.get("supporting_tool_result_ids", ()),
        )


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
    "AcceptanceUpdate",
    "CallToolsAction",
    "ToolCallProposal",
]
