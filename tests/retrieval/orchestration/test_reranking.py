from __future__ import annotations

from dataclasses import dataclass, field
import importlib.metadata
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from personagraph.retrieval.contracts import (
    RerankerRunStatus,
    RetrievalCandidate,
    RetrievalMethod,
    SourceType,
    SourceUnit,
    SourceUnitRef,
)
from personagraph.retrieval.orchestration.reranking import (
    BgeM3Reranker,
    RerankerUnavailable,
    rerank_verified_candidates,
)
from personagraph.retrieval.indexing.model_assets import LocalModelAssetRef


def _resolved(
    unit_id: str,
    content: str,
    *,
    source_type: SourceType = SourceType.CURRENT_SESSION,
    query_index: int = 0,
    rank: int = 1,
):
    ref = SourceUnitRef(source_type, unit_id, "r1", f"hash-{unit_id}")
    return (
        RetrievalCandidate(
            ref=ref,
            method=RetrievalMethod.DENSE,
            query_index=query_index,
            rank=rank,
            raw_score=1.0 / rank,
            fusion_score=1.0 / (60 + rank),
        ),
        SourceUnit(ref, content),
    )


@dataclass
class ScoringReranker:
    scores_by_passage: dict[str, float]
    calls: list[tuple[tuple[str, str], ...]] = field(default_factory=list)

    def fingerprint(self) -> str:
        return "test-cross-encoder"

    def score(self, pairs):
        normalized = tuple(pairs)
        self.calls.append(normalized)
        return tuple(self.scores_by_passage[passage] for _, passage in normalized)


def test_reranker_reorders_within_query_lanes_then_restores_round_robin_fairness():
    a = _resolved("a", "q0-low", query_index=0, rank=1)
    c = _resolved("c", "q1-low", query_index=1, rank=2)
    b = _resolved("b", "q0-high", query_index=0, rank=3)
    d = _resolved("d", "q1-high", query_index=1, rank=4)
    reranker = ScoringReranker(
        {
            "q0-low": 0.1,
            "q0-high": 0.9,
            "q1-low": 0.2,
            "q1-high": 0.8,
        }
    )

    result = rerank_verified_candidates(
        queries=("query zero", "query one"),
        resolved_by_source={SourceType.CURRENT_SESSION: (a, c, b, d)},
        reranker=reranker,
        candidate_limit_per_source=8,
    )

    ordered = result.resolved_by_source[SourceType.CURRENT_SESSION]
    assert [unit.content for _, unit in ordered] == ["q0-high", "q1-high", "q0-low", "q1-low"]
    assert [candidate.rank for candidate, _ in ordered] == [3, 4, 1, 2]
    assert [candidate.rerank_rank for candidate, _ in ordered] == [1, 2, 3, 4]
    assert reranker.calls == [
        (
            ("query zero", "q0-low"),
            ("query one", "q1-low"),
            ("query zero", "q0-high"),
            ("query one", "q1-high"),
        )
    ]
    assert result.outcomes[0].status is RerankerRunStatus.USED
    assert result.outcomes[0].scored_candidate_count == 4


def test_reranker_failure_opens_per_call_circuit_and_preserves_every_source_order():
    class FailingReranker:
        calls = 0

        def fingerprint(self):
            return "failing-reranker"

        def score(self, pairs):
            self.calls += 1
            raise RuntimeError("private model failure")

    reranker = FailingReranker()
    session = (
        _resolved("s1", "session one", rank=1),
        _resolved("s2", "session two", rank=2),
    )
    document = (
        _resolved("d1", "document one", source_type=SourceType.DOCUMENT, rank=1),
        _resolved("d2", "document two", source_type=SourceType.DOCUMENT, rank=2),
    )

    result = rerank_verified_candidates(
        queries=("query",),
        resolved_by_source={
            SourceType.CURRENT_SESSION: session,
            SourceType.DOCUMENT: document,
        },
        reranker=reranker,
        candidate_limit_per_source=8,
    )

    assert result.resolved_by_source[SourceType.CURRENT_SESSION] == session
    assert result.resolved_by_source[SourceType.DOCUMENT] == document
    assert reranker.calls == 1
    assert [outcome.status for outcome in result.outcomes] == [
        RerankerRunStatus.DEGRADED,
        RerankerRunStatus.DEGRADED,
    ]
    assert result.outcomes[0].reason_code == "reranker_failed:RuntimeError"
    assert result.outcomes[1].reason_code == "reranker_circuit_open"


def test_single_verified_candidate_is_scored_for_cross_context_global_ranking():
    reranker = ScoringReranker({"only": 1.0})
    only = _resolved("only", "only")

    result = rerank_verified_candidates(
        queries=("query",),
        resolved_by_source={SourceType.CURRENT_SESSION: (only,)},
        reranker=reranker,
        candidate_limit_per_source=8,
    )

    scored = result.resolved_by_source[SourceType.CURRENT_SESSION]
    assert scored[0][0].rerank_score == 1.0
    assert scored[0][0].rerank_rank == 1
    assert reranker.calls == [(('query', 'only'),)]
    assert result.outcomes[0].status is RerankerRunStatus.USED
    assert result.outcomes[0].scored_candidate_count == 1


def test_stable_reranker_unavailable_reason_survives_orchestration():
    class UnavailableReranker:
        def fingerprint(self):
            return "unavailable-reranker"

        def score(self, pairs):
            raise RerankerUnavailable("bge_reranker_model_unavailable:OSError")

    result = rerank_verified_candidates(
        queries=("query",),
        resolved_by_source={
            SourceType.CURRENT_SESSION: (
                _resolved("first", "first", rank=1),
                _resolved("second", "second", rank=2),
            )
        },
        reranker=UnavailableReranker(),
        candidate_limit_per_source=8,
    )

    assert result.outcomes[0].status is RerankerRunStatus.DEGRADED
    assert result.outcomes[0].reason_code == "bge_reranker_model_unavailable:OSError"


def test_bge_adapter_is_lazy_bounded_and_rejects_unaligned_scores(monkeypatch):
    constructed: list[dict[str, object]] = []

    class FakeFlagReranker:
        def __init__(self, model_name, **kwargs):
            constructed.append({"model_name": model_name, **kwargs})

        def compute_score(self, pairs, **kwargs):
            assert kwargs == {"batch_size": 4, "max_length": 768, "normalize": False}
            return [0.25]

    monkeypatch.setitem(
        sys.modules,
        "FlagEmbedding",
        SimpleNamespace(FlagReranker=FakeFlagReranker),
    )
    reranker = BgeM3Reranker(batch_size=4, query_max_length=128, max_length=768)
    assert constructed == []

    with pytest.raises(RerankerUnavailable, match="unaligned"):
        reranker.score((("q", "first"), ("q", "second")))

    assert len(constructed) == 1
    loaded = constructed[0]
    assert Path(str(loaded.pop("model_name"))).is_dir()
    assert loaded == {
        "use_fp16": False,
        "devices": "cpu",
        "batch_size": 4,
        "query_max_length": 128,
        "max_length": 768,
        "normalize": False,
        "trust_remote_code": False,
    }


def test_reranker_fingerprint_binds_runtime_implementation_versions(
    tmp_path,
    monkeypatch,
):
    model_path = tmp_path / "reranker"
    model_path.mkdir()
    for name in (
        "config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "model.safetensors",
    ):
        (model_path / name).write_bytes(b"fixture")
    versions = {
        "FlagEmbedding": "1.4.0",
        "transformers": "4.57.6",
    }
    monkeypatch.setattr(
        importlib.metadata,
        "version",
        lambda distribution: versions[distribution],
    )
    asset = LocalModelAssetRef.path(
        model_path,
        canonical_identity="test:bge-reranker-v2-m3@weights-v1",
    )

    first = BgeM3Reranker(asset=asset).fingerprint()
    versions["FlagEmbedding"] = "1.4.1"
    second = BgeM3Reranker(asset=asset).fingerprint()

    assert "flagembedding=1.4.0" in first
    assert "transformers=4.57.6" in first
    assert "batch_size=8" in first
    assert first != second
    assert str(tmp_path) not in first

    snapshot = BgeM3Reranker(asset=asset, batch_size=4).diagnostic_snapshot()
    assert snapshot["batch_size"] == 4
    assert snapshot["query_max_length"] == 256
    assert snapshot["max_length"] == 1024
    assert snapshot["use_fp16"] is False
