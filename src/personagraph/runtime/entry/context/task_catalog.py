"""Entry Task catalog 的目的中立打包与验证。"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol, TypeVar

from ....context_budget.token_counter import estimate_tokens


class EntryTaskCatalogItem(Protocol):
    """Entry catalog packing only needs this read-only projection shape."""

    insession_task_id: str
    goal_summary: str
    status: object
    current_graph_revision: int | None


_CatalogItemT = TypeVar("_CatalogItemT", bound=EntryTaskCatalogItem)


@dataclass(frozen=True, slots=True)
class EntryTaskCatalog:
    """Lane-neutral bounded catalog consumed by Entry classification."""

    items: tuple[EntryTaskCatalogItem, ...] = ()
    truncated: bool = False


def pack_insession_task_catalog(
    items: Iterable[_CatalogItemT],
    *,
    token_budget: int,
) -> EntryTaskCatalog:
    """按顺序保留紧凑根项，直至调用方提供的预算耗尽。

    store 已将非终态任务排在终态历史之前。此函数会显式标记省略，而不是跳过高成本
    条目并静默允许 classifier 虚构 task ID。预算 policy 位于 Entry Context 组合层；
    此处不存在 catalog 专用隐藏限制。
    """

    ordered = tuple(items)
    remaining = max(0, token_budget)
    selected: list[EntryTaskCatalogItem] = []
    for item in ordered:
        cost = estimate_tokens(_catalog_item_text(item))
        if cost > remaining:
            return EntryTaskCatalog(items=tuple(selected), truncated=True)
        selected.append(item)
        remaining -= cost
    return EntryTaskCatalog(items=tuple(selected), truncated=False)


def validate_catalog_task_ids(
    candidate_ids: Iterable[str],
    catalog: EntryTaskCatalog,
) -> tuple[str, ...]:
    """拒绝 classifier 提案中重复、未知或已省略的 task ID。"""

    ids = tuple(candidate_ids)
    if len(ids) != len(set(ids)):
        raise ValueError("entry classification contains duplicate insession task ids")
    catalog_ids = {item.insession_task_id for item in catalog.items}
    if not set(ids).issubset(catalog_ids):
        raise ValueError("entry classification references a task outside its catalog")
    return ids


def estimate_insession_task_catalog_tokens(catalog: EntryTaskCatalog) -> int:
    """加入 Prompt 前使用与打包相同的表示成本。"""

    return sum(estimate_tokens(_catalog_item_text(item)) for item in catalog.items)


def _catalog_item_text(item: EntryTaskCatalogItem) -> str:
    graph_revision = (
        str(item.current_graph_revision)
        if item.current_graph_revision is not None
        else "none"
    )
    pending_user_question = getattr(item, "pending_user_question", None)
    pending_question = (
        f"\npending_user_question={pending_user_question}"
        if pending_user_question is not None
        else ""
    )
    return (
        f"insession_task_id={item.insession_task_id}\n"
        f"status={getattr(item.status, 'value', item.status)}\n"
        f"graph_revision={graph_revision}\n"
        f"goal_summary={item.goal_summary}"
        f"{pending_question}"
    )


__all__ = [
    "EntryTaskCatalogItem",
    "EntryTaskCatalog",
    "estimate_insession_task_catalog_tokens",
    "pack_insession_task_catalog",
    "validate_catalog_task_ids",
]
