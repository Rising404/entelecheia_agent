"""在线检索与离线评估共享的确定性排序辅助函数。"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

from ..contracts import RetrievalCandidate, RetrievalMethod, SourceUnitRef


def fuse_candidates_by_rrf(
    candidates: Sequence[RetrievalCandidate],
    *,
    rrf_k: int,
) -> tuple[RetrievalCandidate, ...]:
    """使用确定性 RRF 融合一个查询的各方法本地排序。"""

    if rrf_k <= 0:
        raise ValueError("rrf_k must be greater than zero")
    if not candidates:
        return ()
    scores: dict[SourceUnitRef, float] = defaultdict(float)
    first_candidate: dict[SourceUnitRef, RetrievalCandidate] = {}
    contributors: dict[
        SourceUnitRef,
        list[tuple[RetrievalMethod, int, float]],
    ] = defaultdict(list)
    for candidate in candidates:
        scores[candidate.ref] += 1.0 / (rrf_k + candidate.rank)
        first_candidate.setdefault(candidate.ref, candidate)
        contributors[candidate.ref].append(
            (candidate.method, candidate.rank, candidate.raw_score)
        )
    ranked_refs = sorted(
        scores,
        key=lambda ref: (-scores[ref], ref.source_unit_id, ref.source_revision),
    )
    return tuple(
        RetrievalCandidate(
            ref=ref,
            method=first_candidate[ref].method,
            query_index=first_candidate[ref].query_index,
            rank=index + 1,
            raw_score=first_candidate[ref].raw_score,
            fusion_score=scores[ref],
            query_fused_rank=index + 1,
            fusion_contributors=tuple(
                sorted(
                    contributors[ref],
                    key=lambda item: (item[0].value, item[1]),
                )
            ),
        )
        for index, ref in enumerate(ranked_refs)
    )


def round_robin_query_candidates(
    candidates_by_query: Sequence[Sequence[RetrievalCandidate]],
    candidate_limit: int,
) -> list[RetrievalCandidate]:
    """在独立运行的查询之间公平填充 Source 候选预算。"""

    cursors = [0] * len(candidates_by_query)
    seen: set[SourceUnitRef] = set()
    selected: list[RetrievalCandidate] = []
    while len(selected) < candidate_limit:
        made_progress = False
        for query_index, candidates in enumerate(candidates_by_query):
            cursor = cursors[query_index]
            while cursor < len(candidates) and candidates[cursor].ref in seen:
                cursor += 1
            cursors[query_index] = cursor
            if cursor >= len(candidates):
                continue
            original = candidates[cursor]
            cursors[query_index] += 1
            seen.add(original.ref)
            selected.append(
                RetrievalCandidate(
                    ref=original.ref,
                    method=original.method,
                    query_index=original.query_index,
                    rank=len(selected) + 1,
                    raw_score=original.raw_score,
                    fusion_score=original.fusion_score,
                    query_fused_rank=original.query_fused_rank,
                    fusion_contributors=original.fusion_contributors,
                )
            )
            made_progress = True
            if len(selected) >= candidate_limit:
                break
        if not made_progress:
            break
    return selected
