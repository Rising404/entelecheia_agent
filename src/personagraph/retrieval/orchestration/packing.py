"""来源验证完成后的上下文打包。"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from ..contracts import (
    ContextPackOmission,
    ContextPackOmissionReason,
    RetrievalBudget,
    RetrievalCandidate,
    RetrievedItem,
    SourceType,
    SourceUnit,
)
from ..ports import TokenEstimator
from .selection import (
    ResolvedCandidate,
    SourceSelectionQuotas,
    iter_pool_candidates,
    ordered_quota_pools,
)


@dataclass(frozen=True, slots=True)
class ContextPackResult:
    """已选内容，以及打包限制造成的可计数遗漏。"""

    items: tuple[RetrievedItem, ...]
    packed_tokens: int
    diagnostics: tuple[str, ...]
    omissions: tuple[ContextPackOmission, ...]


class ContextPacker:
    """将一项 token 预算应用于旧版排序或已配置的来源配额。"""

    def __init__(
        self,
        *,
        token_estimator: TokenEstimator,
        source_selection_quotas: SourceSelectionQuotas,
    ) -> None:
        self._token_estimator = token_estimator
        self._source_selection_quotas = source_selection_quotas

    def pack(
        self,
        *,
        source_order: Sequence[SourceType],
        resolved_by_source: Mapping[SourceType, Sequence[ResolvedCandidate]],
        budget: RetrievalBudget,
    ) -> ContextPackResult:
        if not self._source_selection_quotas.enabled:
            return self._pack_in_configured_source_order(
                source_order=source_order,
                resolved_by_source=resolved_by_source,
                budget=budget,
            )
        return self._pack_with_source_quotas(
            source_order=source_order,
            resolved_by_source=resolved_by_source,
            budget=budget,
        )

    def _pack_with_source_quotas(
        self,
        *,
        source_order: Sequence[SourceType],
        resolved_by_source: Mapping[SourceType, Sequence[ResolvedCandidate]],
        budget: RetrievalBudget,
    ) -> ContextPackResult:
        items: list[RetrievedItem] = []
        omissions: list[tuple[SourceType, ContextPackOmissionReason]] = []
        diagnostics: list[str] = []
        used_tokens = 0
        pools = ordered_quota_pools(source_order)
        for pool_index, pool in enumerate(pools):
            candidates = tuple(
                iter_pool_candidates(
                    pool=pool,
                    resolved_by_source=resolved_by_source,
                )
            )
            accepted_from_pool = 0
            item_limit = self._source_selection_quotas.item_limit(pool)
            for candidate_index, (candidate, unit) in enumerate(candidates):
                if len(items) >= budget.max_items:
                    _record_omissions(
                        omissions,
                        candidates[candidate_index:],
                        ContextPackOmissionReason.MAX_ITEMS,
                    )
                    for remaining_pool in pools[pool_index + 1 :]:
                        _record_omissions(
                            omissions,
                            iter_pool_candidates(
                                pool=remaining_pool,
                                resolved_by_source=resolved_by_source,
                            ),
                            ContextPackOmissionReason.MAX_ITEMS,
                        )
                    return _pack_result(items, used_tokens, diagnostics, omissions)
                if item_limit is not None and accepted_from_pool >= item_limit:
                    diagnostics.append(f"source_pool_item_quota_reached:{pool.value}")
                    _record_omissions(
                        omissions,
                        candidates[candidate_index:],
                        ContextPackOmissionReason.SOURCE_POOL_ITEM_QUOTA,
                    )
                    break
                item = self._pack_candidate_if_within_budget(
                    candidate=candidate,
                    unit=unit,
                    used_tokens=used_tokens,
                    budget=budget,
                )
                if item is None:
                    omissions.append((candidate.ref.source_type, ContextPackOmissionReason.TOKEN_LIMIT))
                    continue
                items.append(item)
                used_tokens += item.estimated_tokens
                accepted_from_pool += 1
        return _pack_result(items, used_tokens, diagnostics, omissions)

    def _pack_in_configured_source_order(
        self,
        *,
        source_order: Sequence[SourceType],
        resolved_by_source: Mapping[SourceType, Sequence[ResolvedCandidate]],
        budget: RetrievalBudget,
    ) -> ContextPackResult:
        """在暴露遗漏项的同时保留配额前的选择顺序。"""

        items: list[RetrievedItem] = []
        omissions: list[tuple[SourceType, ContextPackOmissionReason]] = []
        used_tokens = 0
        for source_index, source_type in enumerate(source_order):
            candidates = tuple(resolved_by_source.get(source_type, ()))
            for candidate_index, (candidate, unit) in enumerate(candidates):
                if len(items) >= budget.max_items:
                    _record_omissions(
                        omissions,
                        candidates[candidate_index:],
                        ContextPackOmissionReason.MAX_ITEMS,
                    )
                    for remaining_source in source_order[source_index + 1 :]:
                        _record_omissions(
                            omissions,
                            resolved_by_source.get(remaining_source, ()),
                            ContextPackOmissionReason.MAX_ITEMS,
                        )
                    return _pack_result(items, used_tokens, (), omissions)
                item = self._pack_candidate_if_within_budget(
                    candidate=candidate,
                    unit=unit,
                    used_tokens=used_tokens,
                    budget=budget,
                )
                if item is None:
                    omissions.append((candidate.ref.source_type, ContextPackOmissionReason.TOKEN_LIMIT))
                    continue
                items.append(item)
                used_tokens += item.estimated_tokens
        return _pack_result(items, used_tokens, (), omissions)

    def _pack_candidate_if_within_budget(
        self,
        *,
        candidate: RetrievalCandidate,
        unit: SourceUnit,
        used_tokens: int,
        budget: RetrievalBudget,
    ) -> RetrievedItem | None:
        estimated_tokens = self._token_estimator(unit.content)
        if used_tokens + estimated_tokens > budget.context_token_limit:
            return None
        return RetrievedItem(
            ref=unit.ref,
            content=unit.content,
            citation=unit.citation,
            estimated_tokens=estimated_tokens,
            fused_rank=candidate.rank,
            query_index=candidate.query_index,
            fusion_score=candidate.fusion_score,
            reranked_rank=candidate.rerank_rank,
            reranker_score=candidate.rerank_score,
        )


def _record_omissions(
    omissions: list[tuple[SourceType, ContextPackOmissionReason]],
    candidates: Iterable[ResolvedCandidate],
    reason: ContextPackOmissionReason,
) -> None:
    omissions.extend((candidate.ref.source_type, reason) for candidate, _ in candidates)


def _pack_result(
    items: Sequence[RetrievedItem],
    packed_tokens: int,
    diagnostics: Sequence[str],
    omissions: Sequence[tuple[SourceType, ContextPackOmissionReason]],
) -> ContextPackResult:
    counts: dict[tuple[SourceType, ContextPackOmissionReason], int] = {}
    for source_type, reason in omissions:
        key = (source_type, reason)
        counts[key] = counts.get(key, 0) + 1
    return ContextPackResult(
        items=tuple(items),
        packed_tokens=packed_tokens,
        diagnostics=tuple(diagnostics),
        omissions=tuple(
            ContextPackOmission(source_type=source_type, reason=reason, count=count)
            for (source_type, reason), count in sorted(
                counts.items(),
                key=lambda item: (item[0][0].value, item[0][1].value),
            )
        ),
    )
