"""L2 仍使用的验收进度项及共享字段校验；L1 不再维护逐项完成自评。"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator

from .evidence import EmptySupportJustification


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


def validate_acceptance_supporting_result_ids(
    values: tuple[str, ...],
    *,
    claimed_satisfied: bool | None,
) -> tuple[str, ...]:
    """提案与进度存储共用引用集合规则，不验证来源的语义或访问权限。"""

    if any(not value.strip() for value in values):
        raise ValueError("supporting ToolResult IDs must not be empty")
    if len(values) != len(set(values)):
        raise ValueError("supporting ToolResult IDs must be unique")
    if claimed_satisfied is False and values:
        raise ValueError("an unsatisfied Acceptance cannot retain supporting ToolResults")
    return values


def validate_acceptance_empty_support(
    value: EmptySupportJustification | None,
    *,
    claimed_satisfied: bool | None,
    supporting_ids: tuple[str, ...],
) -> EmptySupportJustification | None:
    """校验无普通证据说明与完成声明/证据集合的字段关系。"""

    if claimed_satisfied is False and value is not None:
        raise ValueError("an unsatisfied Acceptance cannot retain an empty-support justification")
    if supporting_ids and value is not None:
        raise ValueError("an Acceptance with supporting ToolResults cannot carry an empty-support justification")
    return value


class AcceptanceProgressItem(_Contract):
    acceptance_id: str = Field(min_length=1)
    model_claimed_satisfied: bool
    supporting_tool_result_ids: tuple[str, ...] = ()
    empty_support_justification: EmptySupportJustification | None = None

    @field_validator("supporting_tool_result_ids")
    @classmethod
    def _require_nonempty_unique_result_ids(
        cls,
        values: tuple[str, ...],
        info: ValidationInfo,
    ) -> tuple[str, ...]:
        return validate_acceptance_supporting_result_ids(
            values, claimed_satisfied=info.data.get("model_claimed_satisfied"),
        )

    @field_validator("empty_support_justification")
    @classmethod
    def _validate_empty_support(
        cls, value: EmptySupportJustification | None, info: ValidationInfo,
    ) -> EmptySupportJustification | None:
        return validate_acceptance_empty_support(
            value, claimed_satisfied=info.data.get("model_claimed_satisfied"),
            supporting_ids=info.data.get("supporting_tool_result_ids", ()),
        )


__all__ = [
    "AcceptanceProgressItem",
    "validate_acceptance_supporting_result_ids", "validate_acceptance_empty_support",
]
