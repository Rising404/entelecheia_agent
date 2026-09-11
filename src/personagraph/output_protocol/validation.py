"""对模型输出提案执行不涉及语义判断的准入检查。"""

from __future__ import annotations

from collections.abc import Collection

from ..persistent_turn_content.evidence import EmptySupportJustification


def require_completed_support_submission(
    *,
    claimed_completed: bool,
    supporting_ids: Collection[str],
    empty_support_justification: EmptySupportJustification | None,
    subject_label: str,
) -> None:
    """要求无普通支持 ID 的完成声明明确说明原因，而不判断其语义真伪。"""

    if (
        claimed_completed
        and not supporting_ids
        and empty_support_justification is None
    ):
        raise ValueError(
            f"{subject_label} requires empty_support_justification when "
            "submitted as completed without supporting IDs"
        )


__all__ = ["require_completed_support_submission"]
