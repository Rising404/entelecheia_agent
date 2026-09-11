from personagraph.retrieval.tooling.contracts import (
    RetrievalStatus,
    RetrievalToolResult,
)
from personagraph.retrieval.tooling.service.audit import _final_projection_evidence


def _project(item):
    return _final_projection_evidence(
        {"evidence": [item]},
        result=RetrievalToolResult(
            status=RetrievalStatus.COMPLETE,
            scope_snapshot_id="scope",
            retrieval_data_version="generation",
        ),
        item_by_ref={},
    )[0]


def test_final_evidence_audit_preserves_bounded_query_matches_without_text():
    matches = [
        {
            "query_index": 0,
            "rank": 3,
            "fused_rank": 5,
            "fusion_score": 0.03,
            "reranker_score": 0.4,
        },
        {
            "query_index": 1,
            "rank": 1,
            "fused_rank": 9,
            "fusion_score": 0.02,
            "reranker_score": 0.9,
        },
    ]
    actual = _project(
        {
            "rank": 1,
            "query_index": 1,
            "file_id": "file",
            "query_matches": [
                {**match, "snippet": "private text"} for match in matches
            ],
            "snippet": "private text",
        }
    )

    assert actual["query_index"] == 1
    assert actual["query_matches"] == matches
    assert "private text" not in repr(actual)


def test_final_evidence_audit_rejects_invalid_or_duplicate_match_positions():
    actual = _project(
        {
            "rank": 1,
            "query_index": True,
            "query_matches": [
                {
                    "query_index": 0,
                    "rank": 64,
                    "fused_rank": 64,
                    "fusion_score": float("nan"),
                },
                {"query_index": 0, "rank": 1, "fused_rank": 1},
                {"query_index": 4, "rank": 1, "fused_rank": 1},
                {"query_index": 1, "rank": 65, "fused_rank": 1},
                {"query_index": 2, "rank": 1, "fused_rank": 1},
            ],
        }
    )

    assert "query_index" not in actual
    assert actual["query_matches"] == [
        {
            "query_index": 0,
            "rank": 64,
            "fused_rank": 64,
            "fusion_score": None,
            "reranker_score": None,
        }
    ]
