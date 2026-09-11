"""对已验证检索条目执行确定性 Source 池选择。

本模块在 Source 本地检索、融合和权威内容验证之后才开始工作。它绝不会跨 Source 比较
原始分数，也不会改变受信作用域。数值配额仍属于调用方所有的 Activation 配置；
Foundation 只提供冻结的池结构和长期记忆交替规则。
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum

from ..contracts import RetrievalCandidate, SourceType, SourceUnit


class SourceQuotaPool(StrEnum):
    """为四个 RAG Source 约定的三个上下文选择池。"""

    CURRENT_SESSION = "current_session"
    LONG_TERM_MEMORY = "long_term_memory"
    DOCUMENT = "document"


def pool_for_source(source_type: SourceType) -> SourceQuotaPool:
    if source_type is SourceType.CURRENT_SESSION:
        return SourceQuotaPool.CURRENT_SESSION
    if source_type in {SourceType.LONG_TERM_USER, SourceType.LONG_TERM_TASK}:
        return SourceQuotaPool.LONG_TERM_MEMORY
    if source_type is SourceType.DOCUMENT:
        return SourceQuotaPool.DOCUMENT
    raise ValueError(f"no source quota pool for {source_type!r}")


@dataclass(frozen=True, slots=True)
class SourceSelectionQuotas:
    """由受信配置提供、按 Source 池划分的可选条目上限。

    空映射会精确保留现有 Foundation 打包顺序。值刻意表示条目数量，而非 token 数量：
    特定模型的上下文 token 预算不属于本策略，并会在选中条目后由打包器应用。
    """

    pool_item_limits: Mapping[SourceQuotaPool, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        normalised: dict[SourceQuotaPool, int] = {}
        for pool, limit in self.pool_item_limits.items():
            parsed_pool = SourceQuotaPool(pool)
            parsed_limit = int(limit)
            if parsed_limit <= 0:
                raise ValueError("Source pool item limits must be greater than zero")
            normalised[parsed_pool] = parsed_limit
        object.__setattr__(self, "pool_item_limits", normalised)

    @property
    def enabled(self) -> bool:
        return bool(self.pool_item_limits)

    def item_limit(self, pool: SourceQuotaPool) -> int | None:
        return self.pool_item_limits.get(pool)


ResolvedCandidate = tuple[RetrievalCandidate, SourceUnit]


def ordered_quota_pools(source_order: Sequence[SourceType]) -> tuple[SourceQuotaPool, ...]:
    """在池间保留受信 Source 顺序，同时对共享池去重。"""

    result: list[SourceQuotaPool] = []
    seen: set[SourceQuotaPool] = set()
    for source_type in source_order:
        pool = pool_for_source(source_type)
        if pool not in seen:
            seen.add(pool)
            result.append(pool)
    return tuple(result)


def iter_pool_candidates(
    *,
    pool: SourceQuotaPool,
    resolved_by_source: Mapping[SourceType, Sequence[ResolvedCandidate]],
) -> Iterator[ResolvedCandidate]:
    """产出已排序候选，不跨 Source 比较分数。"""

    if pool is SourceQuotaPool.LONG_TERM_MEMORY:
        yield from _iter_long_term_memory_candidates(resolved_by_source)
        return
    source_type = {
        SourceQuotaPool.CURRENT_SESSION: SourceType.CURRENT_SESSION,
        SourceQuotaPool.DOCUMENT: SourceType.DOCUMENT,
    }[pool]
    yield from resolved_by_source.get(source_type, ())


def _iter_long_term_memory_candidates(
    resolved_by_source: Mapping[SourceType, Sequence[ResolvedCandidate]],
) -> Iterator[ResolvedCandidate]:
    """交替产出用户和任务结果，再让非空一侧使用剩余槽位。

    固定由 ``long_term_user`` 首先产出，只是为了确定性打破平局，并非相关性偏好：每个
    Source 保留自身排序，此处绝不会检查原始分数。
    """

    user_items = resolved_by_source.get(SourceType.LONG_TERM_USER, ())
    task_items = resolved_by_source.get(SourceType.LONG_TERM_TASK, ())
    user_index = task_index = 0
    while user_index < len(user_items) or task_index < len(task_items):
        if user_index < len(user_items):
            yield user_items[user_index]
            user_index += 1
        if task_index < len(task_items):
            yield task_items[task_index]
            task_index += 1
