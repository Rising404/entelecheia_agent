"""供 Entry 上下文 task catalog 使用的纯模型可见 policy。

Entry 上下文组装负责两项持久化读取及其 token 预算。本模块只将已读取的 catalog 与
待处理问题事实组合成 classifier 可见的有序 catalog item。
"""

from __future__ import annotations

from typing import Iterable, Protocol, TypeVar


class _ProjectableCatalogItem(Protocol):
    insession_task_id: str
    pending_user_question: str | None

    def model_copy(self, *, update: dict[str, object]) -> object: ...


_CatalogItemT = TypeVar("_CatalogItemT", bound=_ProjectableCatalogItem)


def project_entry_task_catalog_items(
    *,
    catalog_items: tuple[_CatalogItemT, ...],
    pending_user_questions: Iterable[object],
) -> tuple[_CatalogItemT, ...]:
    """丰富恰好一个待处理问题，并以稳定方式提升其优先级。

    只有一个非空文本问题绑定到一个非空文本 task id 时，待处理交互才对模型可见。
    零个或多个问题会保留 catalog item 现有的有界 pending 事实。
    """

    pending_by_task: dict[str, list[str]] = {}
    for pending in pending_user_questions:
        task_id = getattr(pending, "insession_task_id", None)
        question = getattr(pending, "question", None)
        if (
            isinstance(task_id, str)
            and task_id
            and isinstance(question, str)
            and question
        ):
            pending_by_task.setdefault(task_id, []).append(question)
    enriched_catalog_items = tuple(
        item.model_copy(
            update={
                "pending_user_question": pending_by_task[item.insession_task_id][0]
            }
        )
        if len(pending_by_task.get(item.insession_task_id, ())) == 1
        else item
        for item in catalog_items
    )
    # 恰好一个待处理交互是使简短回答可寻址的唯一安全方式；稳定枚举保留原始任务顺序。
    return tuple(
        item
        for _index, item in sorted(
            enumerate(enriched_catalog_items),
            key=lambda pair: (
                getattr(pair[1], "pending_user_question", None) is None,
                pair[0],
            ),
        )
    )


__all__ = ["project_entry_task_catalog_items"]
