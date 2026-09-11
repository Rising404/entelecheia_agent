"""Entry 路由选择消费的 lane-neutral 结构契约。"""

from __future__ import annotations

from typing import Protocol


class EntryTaskMatchApplyResult(Protocol):
    """L2 mutation receipt shape consumed by the lane-neutral Entry."""

    created_insession_task_ids_by_local_key: dict[str, str]
    related_insession_task_ids: tuple[str, ...]
    window_state_version: int | None


__all__ = ["EntryTaskMatchApplyResult"]
