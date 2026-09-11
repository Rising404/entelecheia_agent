from __future__ import annotations

from math import log2

import pytest

from personagraph.retrieval.contracts import SourceType, SourceUnitRef
from personagraph.retrieval.relevance import (
    RelevanceCase,
    RelevanceQualityGate,
    evaluate_ranked_retrieval,
    evaluate_relevance_gate,
)


def _ref(name: str, *, source_type: SourceType = SourceType.DOCUMENT) -> SourceUnitRef:
    return SourceUnitRef(source_type, name, "revision-1", f"hash-{name}")


def test_relevance_report_calculates_recall_mrr_and_ndcg_with_stable_deduplication():
    relevant_a = _ref("relevant-a")
    relevant_b = _ref("relevant-b")
    irrelevant = _ref("irrelevant")
    cases = (
        RelevanceCase("mixed-language", "BGE-M3 document version", frozenset({relevant_a, relevant_b})),
        RelevanceCase("path", "src/personagraph/retrieval", frozenset({_ref("relevant-c")})),
    )

    report = evaluate_ranked_retrieval(
        route_id="hybrid_rrf",
        cases=cases,
        ranked_refs_by_case_id={
            "mixed-language": (irrelevant, relevant_a, relevant_a, relevant_b),
            "path": (_ref("relevant-c"),),
        },
        cutoffs=(1, 2, 3),
    )

    first = report.cases[0]
    assert first.ranked_count == 3
    assert first.first_relevant_rank == 2
    assert first.recall_at_k == {1: 0.0, 2: 0.5, 3: 1.0}
    assert first.hit_at_k == {1: 0.0, 2: 1.0, 3: 1.0}
    assert first.ndcg_at_k[3] == pytest.approx(
        (1 / log2(3) + 1 / log2(4)) / (1 + 1 / log2(3))
    )
    assert report.mean_reciprocal_rank == pytest.approx(0.75)
    assert report.mean_recall_at_k[3] == 1.0
    assert report.hit_rate_at_k[1] == 0.5
    assert "content" not in repr(report).lower()


def test_relevance_gate_uses_only_explicit_thresholds_and_detects_missing_cases():
    relevant = _ref("relevant")
    cases = (
        RelevanceCase("found", "中文检索", frozenset({relevant})),
        RelevanceCase("missing", "code: foo()", frozenset({_ref("other")})),
    )
    report = evaluate_ranked_retrieval(
        route_id="bm25",
        cases=cases,
        ranked_refs_by_case_id={"found": (relevant,)},
        cutoffs=(1, 3),
    )

    result = evaluate_relevance_gate(
        report,
        RelevanceQualityGate(minimum_mrr=0.4, minimum_hit_rate_at_k={1: 0.4}),
    )

    assert result.passed is False
    assert result.violations == ("missing_result_cases:missing",)


def test_relevance_contract_rejects_unknown_cases_or_unreported_gate_cutoffs():
    case = RelevanceCase("only", "数字 128k", frozenset({_ref("relevant")}))
    with pytest.raises(ValueError, match="unknown case ids"):
        evaluate_ranked_retrieval(
            route_id="dense",
            cases=(case,),
            ranked_refs_by_case_id={"stale": ()},
        )

    report = evaluate_ranked_retrieval(
        route_id="dense",
        cases=(case,),
        ranked_refs_by_case_id={"only": (_ref("relevant"),)},
        cutoffs=(1,),
    )
    result = evaluate_relevance_gate(
        report,
        RelevanceQualityGate(minimum_recall_at_k={3: 1.0}),
    )
    assert result.passed is False
    assert result.violations == ("recall_cutoff_not_reported:3",)
