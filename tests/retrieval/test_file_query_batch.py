"""文件检索预算以 query 为单位，不能被文件数或跨 query 去重改写。"""

from dataclasses import dataclass, field
import hashlib

import pytest

from personagraph.retrieval.contracts import (
    QueryProposal,
    RetrievalBudget,
    RetrievalCandidate,
    RetrievalMethod,
    RetrievalRequest,
    SourceAccess,
    SourceAvailability,
    SourceDependency,
    SourceFilter,
    SourceType,
    SourceUnit,
    SourceUnitRef,
    TrustedRetrievalBoundary,
)
from personagraph.retrieval.policy import DefaultRetrievalPolicy
from personagraph.retrieval.query_guard import QueryGuard
from personagraph.retrieval.service import RetrievalService


@dataclass
class Source:
    units: dict = field(default_factory=dict)
    source_type: SourceType = SourceType.DOCUMENT
    fetched: list = field(default_factory=list)

    def open_retrieval_access(self, source_filter):
        return SourceAccess(self.source_type, source_filter, SourceAvailability.READY)

    def fetch_units(self, access, refs):
        self.fetched.extend(refs)
        scope = access.source_filter.as_mapping()["doc_id"]
        return [
            self.units[ref]
            for ref in refs
            if self.units[ref].citation["doc_id"] == scope
        ]


@dataclass
class Method:
    source: Source
    method: RetrievalMethod = RetrievalMethod.DENSE

    def search(
        self,
        query,
        *,
        query_index,
        source_filter,
        limit,
        retrieval_data_version_id=None,
    ):
        scope = source_filter.as_mapping()["doc_id"]
        units = [
            unit
            for unit in self.source.units.values()
            if unit.citation["doc_id"] == scope
        ]
        return tuple(
            RetrievalCandidate(unit.ref, self.method, query_index, rank, 1000 - rank)
            for rank, unit in enumerate(units[:limit], 1)
        )


@dataclass
class Reranker:
    calls: list = field(default_factory=list)
    fail_query: str | None = None

    def fingerprint(self):
        return "test-reranker"

    def score(self, pairs):
        self.calls.append(tuple(pairs))
        if pairs[0][0] == self.fail_query:
            raise RuntimeError("offline reranker failure")
        return [
            int(text.split()[1]) * (1 if query == "q1" else -1) for query, text in pairs
        ]


def setup_case(*, files=1, chunks=40, queries=("q0", "q1")):
    source = Source()
    requests = []
    for file_index in range(files):
        doc_id = f"doc-{file_index}"
        for index in range(chunks):
            text = f"chunk {index} document {file_index}"
            ref = SourceUnitRef(
                SourceType.DOCUMENT,
                f"{doc_id}:{index}",
                "v1",
                hashlib.sha256(text.encode()).hexdigest(),
            )
            source.units[ref] = SourceUnit(ref, text, {"doc_id": doc_id})
        source_filter = SourceFilter.from_mapping(
            SourceType.DOCUMENT, {"doc_id": doc_id}
        )
        requests.append(
            RetrievalRequest(
                request_id=doc_id,
                model_call_purpose="MODEL_RETRIEVE_FILES_V1",
                query_proposal=QueryProposal(
                    queries, source_hints=frozenset({SourceType.DOCUMENT})
                ),
                boundary=TrustedRetrievalBoundary(
                    {SourceType.DOCUMENT: source_filter},
                    {SourceType.DOCUMENT: SourceDependency.OPTIONAL},
                ),
            )
        )
    reranker = Reranker()
    service = RetrievalService(
        policy=DefaultRetrievalPolicy(
            default_methods=(
                RetrievalMethod.DENSE,
                RetrievalMethod.LEARNED_SPARSE,
                RetrievalMethod.BM25,
            )
        ),
        query_guard=QueryGuard(),
        sources={SourceType.DOCUMENT: source},
        readonly_sources={SourceType.DOCUMENT: source},
        methods={
            method: Method(source, method)
            for method in (
                RetrievalMethod.DENSE,
                RetrievalMethod.LEARNED_SPARSE,
                RetrievalMethod.BM25,
            )
        },
        reranker=reranker,
    )
    return service, source, reranker, tuple(requests)


def retrieve(service, requests, *, result_limit=96, token_limit=96000):
    return service.retrieve_file_query_batch(
        requests,
        RetrievalBudget(
            128, token_limit, result_limit
        ),
        result_limit=result_limit,
        readonly=True,
    )


def test_same_chunk_is_scored_for_each_query_before_final_deduplication():
    service, source, reranker, requests = setup_case(chunks=3)
    contexts = retrieve(service, requests, result_limit=3)
    items = [item for context in contexts for item in context.items]
    assert len(items) == 3
    assert len(source.fetched) == 3
    assert all(
        len(candidate.fusion_contributors) == 3
        for context in contexts
        for candidate in context.fusion_candidates
    )
    assert {query for pairs in reranker.calls for query, _ in pairs} == {"q0", "q1"}
    middle = next(item for item in items if item.ref.source_unit_id.endswith(":1"))
    assert [
        (match.query_index, match.reranker_score) for match in middle.query_matches
    ] == [(0, -1), (1, 1)]


def test_query_rrf_64_precedes_global_selection_and_retains_all_query_matches():
    service, _, _, requests = setup_case(chunks=80)
    items = [item for context in retrieve(service, requests) for item in context.items]
    assert len(items) == 64
    assert [
        sum(
            any(match.query_index == q for match in item.query_matches)
            for item in items
        )
        for q in range(2)
    ] == [64, 64]


def test_multiple_files_share_one_64_rrf_shortlist_per_query():
    service, source, reranker, requests = setup_case(files=3, chunks=100)
    retrieve(service, requests)
    assert [
        sum(q == query for pairs in reranker.calls for q, _ in pairs)
        for query in ("q0", "q1")
    ] == [64, 64]
    assert len(source.fetched) <= 128


def test_batch_reranker_failure_returns_only_fusion_ranking_not_partial_scores():
    service, _, reranker, requests = setup_case(chunks=5)
    reranker.fail_query = "q0"
    items = [
        item
        for context in retrieve(service, requests, result_limit=2)
        for item in context.items
    ]
    assert all(
        match.reranker_score is None
        for item in items
        for match in item.query_matches
    )
    assert any(
        "reranker_degraded" in code
        for context in retrieve(service, requests, result_limit=2)
        for code in context.diagnostic_codes
    )


def test_text_budget_is_applied_after_dedup_and_reports_omissions():
    service, _, _, requests = setup_case(chunks=3)
    contexts = retrieve(service, requests, result_limit=3, token_limit=8)
    assert sum(context.packed_tokens for context in contexts) <= 8
    assert any(context.truncated for context in contexts)
    assert any(
        omission.reason.value == "token_limit"
        for context in contexts
        for omission in context.pack_omissions
    )


def test_batch_rejects_mixed_query_sets():
    service, _, _, requests = setup_case(files=2)
    from dataclasses import replace

    requests = (
        requests[0],
        replace(requests[1], query_proposal=QueryProposal(("different",))),
    )
    with pytest.raises(ValueError, match="same queries"):
        retrieve(service, requests)


def test_same_authorized_source_in_two_scopes_does_not_duplicate_candidates_or_scores():
    service, _, reranker, requests = setup_case(chunks=3)
    contexts = retrieve(service, (requests[0], requests[0]), result_limit=3)
    assert sum(len(pairs) for pairs in reranker.calls) == 6
    assert [len(context.items) for context in contexts] == [3, 0]


def test_four_overlapping_queries_keep_64_unique_candidates_with_four_matches():
    service, _, reranker, requests = setup_case(
        chunks=128, queries=("q0", "q1", "q2", "q3")
    )
    reranker.score = lambda pairs: [
        1000 - abs(int(text.split()[1]) - int(query[-1]) * 32) for query, text in pairs
    ]
    contexts = retrieve(service, requests)
    items = [item for context in contexts for item in context.items]
    assert len(items) == 64
    assert [
        sum(
            any(match.query_index == query for match in item.query_matches)
            for item in items
        )
        for query in range(4)
    ] == [64] * 4


def test_repacking_selected_results_preserves_query_rank_after_token_omissions():
    from dataclasses import replace
    from personagraph.retrieval.orchestration.file_query_batch import pack_query_items

    service, _, _, requests = setup_case(chunks=3)
    items = [
        item
        for context in retrieve(service, requests, result_limit=3)
        for item in context.items
    ]
    items[0] = replace(items[0], estimated_tokens=1000)
    first = pack_query_items(items, result_limit=3, token_limit=20)
    second = pack_query_items(first.items, result_limit=3, token_limit=20)
    assert first.items == second.items


def test_rrf_shortlist_keeps_64_pairs_per_query_in_one_scoring_call():
    from collections import Counter

    service, _, reranker, requests = setup_case(files=3, chunks=100)
    retrieve(service, requests)
    assert len(reranker.calls) == 1
    assert Counter(query for query, _ in reranker.calls[0]) == {"q0": 64, "q1": 64}


def test_global_pack_returns_96_unique_chunks_with_all_query_matches():
    from personagraph.retrieval.contracts import RetrievedItem
    from personagraph.retrieval.orchestration.file_query_batch import pack_query_items

    items = []
    for query_index in range(2):
        for index in range(60):
            key = index + query_index * 50
            ref = SourceUnitRef(SourceType.DOCUMENT, f"chunk-{key:03}", "v1", "a" * 64)
            items.append(RetrievedItem(
                ref=ref, content=f"text {key}", citation={}, estimated_tokens=1,
                fused_rank=index + 1, query_index=query_index,
                reranker_score=float(key + query_index),
            ))
    packed = pack_query_items(items, result_limit=96, token_limit=96000)
    assert len(packed.items) == 96
    assert len({item.ref for item in packed.items}) == 96
    shared = next(item for item in packed.items if item.ref.source_unit_id == "chunk-055")
    assert {match.query_index for match in shared.query_matches} == {0, 1}
    assert shared.reranker_score == 56
    assert packed.items[0].ref.source_unit_id == "chunk-109"


def test_partial_scores_are_removed_before_fusion_fallback_and_repacking():
    from personagraph.retrieval.contracts import RetrievedItem
    from personagraph.retrieval.orchestration.file_query_batch import pack_query_items

    items = [RetrievedItem(
        ref=SourceUnitRef(SourceType.DOCUMENT, f"chunk-{index}", "v1", "a" * 64),
        content=f"text {index}", citation={}, estimated_tokens=1,
        fused_rank=index + 1, query_index=0, reranker_score=score,
        reranked_rank=3 - index if score is not None else None,
    ) for index, score in enumerate((0.1, 0.9, None))]
    first = pack_query_items(items, result_limit=2, token_limit=100)
    assert [item.ref for item in first.items] == [item.ref for item in items[:2]]
    assert all(item.reranker_score is None and item.reranked_rank is None for item in first.items)
    assert all(match.reranker_score is None for item in first.items for match in item.query_matches)
    assert first.items == pack_query_items(first.items, result_limit=2, token_limit=100).items
