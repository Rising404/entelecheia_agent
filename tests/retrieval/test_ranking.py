from __future__ import annotations

import pytest

from personagraph.retrieval.contracts import (
    RetrievalCandidate,
    RetrievalMethod,
    SourceType,
    SourceUnitRef,
)
from personagraph.retrieval.orchestration.ranking import (
    fuse_candidates_by_rrf,
    round_robin_query_candidates,
)


def _ref(unit_id: str) -> SourceUnitRef:
    return SourceUnitRef(SourceType.DOCUMENT, unit_id, "revision-1", f"hash-{unit_id}")


def test_rrf_preserves_method_contributors_and_query_rank_through_round_robin() -> None:
    shared = _ref("shared")
    dense_only = _ref("dense-only")
    fused = fuse_candidates_by_rrf(
        (
            RetrievalCandidate(
                ref=shared,
                method=RetrievalMethod.DENSE,
                query_index=0,
                rank=1,
                raw_score=0.91,
            ),
            RetrievalCandidate(
                ref=dense_only,
                method=RetrievalMethod.DENSE,
                query_index=0,
                rank=2,
                raw_score=0.72,
            ),
            RetrievalCandidate(
                ref=shared,
                method=RetrievalMethod.LEARNED_SPARSE,
                query_index=0,
                rank=2,
                raw_score=4.2,
            ),
        ),
        rrf_k=60,
    )

    assert fused[0].ref == shared
    assert fused[0].query_fused_rank == 1
    assert fused[0].fusion_score == pytest.approx((1 / 61) + (1 / 62))
    assert fused[0].fusion_contributors == (
        (RetrievalMethod.DENSE, 1, 0.91),
        (RetrievalMethod.LEARNED_SPARSE, 2, 4.2),
    )

    selected = round_robin_query_candidates((fused,), candidate_limit=2)

    assert selected[0].rank == 1
    assert selected[0].query_fused_rank == 1
    assert selected[0].fusion_contributors == fused[0].fusion_contributors
