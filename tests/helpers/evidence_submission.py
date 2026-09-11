from __future__ import annotations

from typing import Any


def empty_support_justification(
    reason_code: str = "candidate_is_primary_artifact",
) -> dict[str, str]:
    return {
        "schema_version": "empty-support-justification-v1",
        "reason_code": reason_code,
        "explanation": "The candidate delivery is the primary artifact under review.",
    }


def add_empty_support_justifications(value: Any) -> Any:
    """更新早于 V1 原因字段的嵌套 L1 模型测试夹具。"""

    if isinstance(value, dict):
        projected = {
            key: add_empty_support_justifications(item)
            for key, item in value.items()
        }
        if (
            projected.get("reported_status") == "completed"
            and projected.get("supporting_tool_call_ids") == []
            and projected.get("empty_support_justification") is None
        ):
            projected["empty_support_justification"] = (
                empty_support_justification()
            )
        if (
            projected.get("model_claimed_satisfied") is True
            and projected.get("supporting_tool_result_ids") == []
            and projected.get("empty_support_justification") is None
        ):
            projected["empty_support_justification"] = (
                empty_support_justification()
            )
        return projected
    if isinstance(value, list):
        return [add_empty_support_justifications(item) for item in value]
    if isinstance(value, tuple):
        return tuple(add_empty_support_justifications(item) for item in value)
    return value


__all__ = [
    "add_empty_support_justifications",
    "empty_support_justification",
]
