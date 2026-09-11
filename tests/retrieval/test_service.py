from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import pytest

from personagraph.retrieval.contracts import (
    ContextCoverageLimitation,
    ContextStatus,
    ContextPackOmission,
    ContextPackOmissionReason,
    ContextVerificationDropReason,
    LongTermMemoryWriteGuardStatus,
    QueryProposal,
    RerankerRunStatus,
    RetrievalBudget,
    RetrievalCandidate,
    RetrievalMethod,
    RetrievalPlan,
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
from personagraph.retrieval.prompt_context import RetrievedContextPromptSerializer
from personagraph.retrieval.query_guard import QueryGuard
from personagraph.retrieval.orchestration.selection import SourceQuotaPool, SourceSelectionQuotas
from personagraph.retrieval.service import RetrievalService


def _ref(
    unit_id: str,
    *,
    source_type: SourceType = SourceType.CURRENT_SESSION,
    revision: str = "r1",
    content_hash: str = "h1",
) -> SourceUnitRef:
    return SourceUnitRef(source_type, unit_id, revision, content_hash)


@dataclass
class FakeSource:
    source_type: SourceType
    units: dict[SourceUnitRef, SourceUnit]
    state: SourceAvailability = SourceAvailability.READY
    coverage_facts: dict[str, str] = field(default_factory=dict)
    seen_filters: list[SourceFilter] = field(default_factory=list)

    def open_retrieval_access(self, source_filter: SourceFilter) -> SourceAccess:
        self.seen_filters.append(source_filter)
        return SourceAccess(
            self.source_type,
            source_filter,
            self.state,
            coverage_facts=self.coverage_facts,
        )

    def fetch_units(
        self,
        access: SourceAccess,
        refs: Sequence[SourceUnitRef],
    ) -> Sequence[SourceUnit]:
        self.seen_filters.append(access.source_filter)
        return [self.units[ref] for ref in refs if ref in self.units]


@dataclass
class FakeMethod:
    method: RetrievalMethod
    hits: dict[tuple[SourceType, str], tuple[RetrievalCandidate, ...]] = field(default_factory=dict)
    seen_filters: list[SourceFilter] = field(default_factory=list)
    seen_data_version_ids: list[str | None] = field(default_factory=list)
    fail: bool = False

    def search(self, query, *, query_index, source_filter, limit, retrieval_data_version_id=None):
        self.seen_filters.append(source_filter)
        self.seen_data_version_ids.append(retrieval_data_version_id)
        if self.fail:
            raise RuntimeError("simulated_backend_failure")
        return self.hits.get((source_filter.source_type, query), ())[:limit]


@dataclass
class TimeoutThenHitsMethod(FakeMethod):
    failures_before_success: int = 0
    calls: int = 0

    def search(self, query, *, query_index, source_filter, limit, retrieval_data_version_id=None):
        self.calls += 1
        if self.calls <= self.failures_before_success:
            raise TimeoutError("transient backend timeout")
        return super().search(
            query,
            query_index=query_index,
            source_filter=source_filter,
            limit=limit,
            retrieval_data_version_id=retrieval_data_version_id,
        )


@dataclass
class TimeoutThenReadySource(FakeSource):
    failures_before_ready: int = 0
    availability_calls: int = 0

    def open_retrieval_access(self, source_filter: SourceFilter) -> SourceAccess:
        self.availability_calls += 1
        if self.availability_calls <= self.failures_before_ready:
            raise TimeoutError("transient source timeout")
        return super().open_retrieval_access(source_filter)


@dataclass
class StaticCoverageProbe:
    source_type: SourceType
    limitation: ContextCoverageLimitation | None = None
    seen_data_version_ids: list[str | None] = field(default_factory=list)

    def check_source_coverage(
        self,
        access: SourceAccess,
        *,
        retrieval_data_version_id: str | None,
    ) -> ContextCoverageLimitation | None:
        assert access.source_type is self.source_type
        self.seen_data_version_ids.append(retrieval_data_version_id)
        return self.limitation


@dataclass
class FailingCoverageProbe:
    source_type: SourceType

    def check_source_coverage(
        self,
        access: SourceAccess,
        *,
        retrieval_data_version_id: str | None,
    ) -> ContextCoverageLimitation | None:
        del access, retrieval_data_version_id
        raise RuntimeError("simulated_cross_store_failure")


@dataclass
class ChangingDataVersionProvider:
    """证明服务只读取一次活跃派生版本。"""

    values: tuple[str, ...] = ("v1", "v2")
    calls: int = 0

    def active_retrieval_data_version_id(self) -> str | None:
        value = self.values[min(self.calls, len(self.values) - 1)]
        self.calls += 1
        return value


@dataclass
class UnavailableDataVersionProvider:
    calls: int = 0

    def active_retrieval_data_version_id(self) -> str | None:
        self.calls += 1
        return None


def _candidate(ref: SourceUnitRef, method: RetrievalMethod, query_index: int = 0, rank: int = 1):
    return RetrievalCandidate(
        ref=ref,
        method=method,
        query_index=query_index,
        rank=rank,
        raw_score=0.9,
    )


def _request(
    *,
    source_filters: dict[SourceType, SourceFilter],
    dependencies: dict[SourceType, SourceDependency] | None = None,
    query: str = "北桥项目",
    excluded: frozenset[SourceUnitRef] = frozenset(),
) -> RetrievalRequest:
    return RetrievalRequest(
        request_id="request-1",
        model_call_purpose="TEST",
        query_proposal=QueryProposal((query,)),
        boundary=TrustedRetrievalBoundary(source_filters, dependencies or {}),
        excluded_direct_refs=excluded,
    )


def _service(
    *,
    sources,
    methods,
    source_coverage_probes=None,
    source_access_revalidators=None,
    source_selection_quotas=None,
    data_version_provider=None,
    encoder_fingerprint="",
    reranker=None,
):
    return RetrievalService(
        policy=DefaultRetrievalPolicy(),
        query_guard=QueryGuard(),
        sources=sources,
        methods=methods,
        source_coverage_probes=source_coverage_probes,
        source_access_revalidators=source_access_revalidators,
        data_version_provider=data_version_provider,
        token_estimator=lambda text: len(text.split()),
        source_selection_quotas=source_selection_quotas,
        encoder_fingerprint=encoder_fingerprint,
        reranker=reranker,
    )


def _budget():
    return RetrievalBudget(candidate_limit_per_source=5, context_token_limit=30, max_items=5)


def test_service_passes_trusted_structural_filter_before_method_top_k_and_fuses_hits():
    ref = _ref("turn-8")
    source_filter = SourceFilter.from_mapping(SourceType.CURRENT_SESSION, {"session_id": "s-a"})
    source = FakeSource(SourceType.CURRENT_SESSION, {ref: SourceUnit(ref, "北桥项目下周交付")})
    dense = FakeMethod(RetrievalMethod.DENSE, {(SourceType.CURRENT_SESSION, "北桥项目"): (_candidate(ref, RetrievalMethod.DENSE),)})
    sparse = FakeMethod(RetrievalMethod.LEARNED_SPARSE, {(SourceType.CURRENT_SESSION, "北桥项目"): (_candidate(ref, RetrievalMethod.LEARNED_SPARSE),)})

    context = _service(
        sources={SourceType.CURRENT_SESSION: source},
        methods={RetrievalMethod.DENSE: dense, RetrievalMethod.LEARNED_SPARSE: sparse},
    ).retrieve_context(_request(source_filters={SourceType.CURRENT_SESSION: source_filter}), _budget())

    assert context.status is ContextStatus.COMPLETE
    assert [item.content for item in context.items] == ["北桥项目下周交付"]
    assert dense.seen_filters == [source_filter]
    assert sparse.seen_filters == [source_filter]
    assert source.seen_filters == [source_filter, source_filter]
    assert context.items[0].fused_rank == 1


def test_service_reranks_only_after_authoritative_content_verification_and_preserves_rrf_rank():
    first = _ref("turn-1", content_hash="h-first")
    second = _ref("turn-2", content_hash="h-second")
    source_filter = SourceFilter.from_mapping(SourceType.CURRENT_SESSION, {"session_id": "s-a"})
    source = FakeSource(
        SourceType.CURRENT_SESSION,
        {
            first: SourceUnit(first, "weak passage"),
            second: SourceUnit(second, "strong passage"),
        },
    )
    dense = FakeMethod(
        RetrievalMethod.DENSE,
        {
            (SourceType.CURRENT_SESSION, "北桥项目"): (
                _candidate(first, RetrievalMethod.DENSE, rank=1),
                _candidate(second, RetrievalMethod.DENSE, rank=2),
            )
        },
    )

    class FakeReranker:
        seen_pairs = ()

        def fingerprint(self):
            return "test-reranker"

        def score(self, pairs):
            self.seen_pairs = tuple(pairs)
            return (0.1, 0.9)

    reranker = FakeReranker()
    context = _service(
        sources={SourceType.CURRENT_SESSION: source},
        methods={RetrievalMethod.DENSE: dense},
        reranker=reranker,
    ).retrieve_context(
        _request(source_filters={SourceType.CURRENT_SESSION: source_filter}),
        _budget(),
    )

    assert reranker.seen_pairs == (
        ("北桥项目", "weak passage"),
        ("北桥项目", "strong passage"),
    )
    assert [item.ref for item in context.items] == [second, first]
    assert [item.fused_rank for item in context.items] == [2, 1]
    assert [item.reranked_rank for item in context.items] == [1, 2]
    assert [item.reranker_score for item in context.items] == [0.9, 0.1]
    assert context.reranker_fingerprint == "test-reranker"
    assert context.reranker_outcomes[0].status is RerankerRunStatus.USED


def test_reranker_failure_is_reported_as_partial_instead_of_complete():
    first = _ref("turn-1", content_hash="h-first")
    second = _ref("turn-2", content_hash="h-second")
    source_filter = SourceFilter.from_mapping(
        SourceType.CURRENT_SESSION,
        {"session_id": "s-a"},
    )
    source = FakeSource(
        SourceType.CURRENT_SESSION,
        {
            first: SourceUnit(first, "first passage"),
            second: SourceUnit(second, "second passage"),
        },
    )
    dense = FakeMethod(
        RetrievalMethod.DENSE,
        {
            (SourceType.CURRENT_SESSION, "北桥项目"): (
                _candidate(first, RetrievalMethod.DENSE, rank=1),
                _candidate(second, RetrievalMethod.DENSE, rank=2),
            )
        },
    )

    class FailingReranker:
        def fingerprint(self):
            return "test-reranker"

        def score(self, pairs):
            del pairs
            raise RuntimeError("simulated inference failure")

    context = _service(
        sources={SourceType.CURRENT_SESSION: source},
        methods={RetrievalMethod.DENSE: dense},
        reranker=FailingReranker(),
    ).retrieve_context(
        _request(source_filters={SourceType.CURRENT_SESSION: source_filter}),
        _budget(),
    )

    assert context.status is ContextStatus.PARTIAL
    assert [item.ref for item in context.items] == [first, second]
    assert context.reranker_outcomes[0].status is RerankerRunStatus.DEGRADED
    assert context.reranker_outcomes[0].reason_code == "reranker_failed:RuntimeError"
    assert (
        "reranker_degraded:current_session:reranker_failed:RuntimeError"
        in context.diagnostic_codes
    )


def test_partial_document_coverage_survives_a_no_match_into_the_prompt():
    source_filter = SourceFilter.from_mapping(
        SourceType.DOCUMENT,
        {"session_id": "s-a", "doc_id": "doc-1"},
    )
    coverage_facts = {
        "processing_status": "partial",
        "processing_diagnostic_codes": '["page_needs_vision"]',
        "needs_vision": "true",
        "coverage_gap": "true",
    }
    context = _service(
        sources={
            SourceType.DOCUMENT: FakeSource(
                SourceType.DOCUMENT,
                {},
                coverage_facts=coverage_facts,
            )
        },
        methods={
            RetrievalMethod.DENSE: FakeMethod(RetrievalMethod.DENSE),
            RetrievalMethod.LEARNED_SPARSE: FakeMethod(RetrievalMethod.LEARNED_SPARSE),
        },
    ).retrieve_context(
        _request(source_filters={SourceType.DOCUMENT: source_filter}),
        _budget(),
    )

    outcome = context.source_outcomes[SourceType.DOCUMENT]
    assert context.items == ()
    assert context.status is ContextStatus.PARTIAL
    assert outcome.retrieval.value == "no_match"
    assert dict(outcome.coverage_facts) == coverage_facts
    assert [item.reason_code for item in context.coverage_limitations if item.source_type is SourceType.DOCUMENT] == [
        "document_processing_coverage_partial"
    ]
    serialized = RetrievedContextPromptSerializer().serialize(context).text
    assert 'source="document"' in serialized
    assert 'retrieval="no_match"' in serialized
    assert 'processing_status="partial"' in serialized
    assert 'coverage_gap="true"' in serialized
    assert 'needs_vision="true"' in serialized
    assert "page_needs_vision" in serialized


def test_service_pins_one_data_version_for_all_methods_and_coverage_checks():
    source_filter = SourceFilter.from_mapping(SourceType.CURRENT_SESSION, {"session_id": "s-a"})
    dense = FakeMethod(RetrievalMethod.DENSE)
    sparse = FakeMethod(RetrievalMethod.LEARNED_SPARSE)
    coverage_probe = StaticCoverageProbe(SourceType.CURRENT_SESSION)
    provider = ChangingDataVersionProvider()

    _service(
        sources={SourceType.CURRENT_SESSION: FakeSource(SourceType.CURRENT_SESSION, {})},
        methods={RetrievalMethod.DENSE: dense, RetrievalMethod.LEARNED_SPARSE: sparse},
        source_coverage_probes={SourceType.CURRENT_SESSION: coverage_probe},
        data_version_provider=provider,
    ).retrieve_context(
        _request(source_filters={SourceType.CURRENT_SESSION: source_filter}),
        _budget(),
    )

    assert provider.calls == 1
    assert dense.seen_data_version_ids == ["v1"]
    assert sparse.seen_data_version_ids == ["v1"]
    assert coverage_probe.seen_data_version_ids == ["v1", "v1"]


@pytest.mark.parametrize(
    ("dependency", "expected_status"),
    (
        (SourceDependency.REQUIRED, ContextStatus.BLOCKED),
        (SourceDependency.OPTIONAL, ContextStatus.PARTIAL),
    ),
)
def test_configured_provider_without_exact_generation_never_queries_any_active_fallback(
    dependency,
    expected_status,
):
    source_filter = SourceFilter.from_mapping(
        SourceType.DOCUMENT,
        {"doc_id": "doc-1"},
    )
    source = FakeSource(SourceType.DOCUMENT, {})
    dense = FakeMethod(RetrievalMethod.DENSE)
    sparse = FakeMethod(RetrievalMethod.LEARNED_SPARSE)
    provider = UnavailableDataVersionProvider()

    context = _service(
        sources={SourceType.DOCUMENT: source},
        methods={
            RetrievalMethod.DENSE: dense,
            RetrievalMethod.LEARNED_SPARSE: sparse,
        },
        data_version_provider=provider,
    ).retrieve_context(
        _request(
            source_filters={SourceType.DOCUMENT: source_filter},
            dependencies={SourceType.DOCUMENT: dependency},
        ),
        _budget(),
    )

    assert provider.calls == 1
    assert source.seen_filters == []
    assert dense.seen_data_version_ids == []
    assert sparse.seen_data_version_ids == []
    assert context.status is expected_status
    assert context.items == ()
    assert context.method_outcomes == ()
    assert context.retrieval_data_version == ""
    assert context.diagnostic_codes == ("retrieval_data_version_unavailable",)
    assert (
        context.source_outcomes[SourceType.DOCUMENT].reason_code
        == "retrieval_data_version_unavailable"
    )


def test_coverage_check_runs_after_all_source_method_searches_even_without_hits():
    events: list[str] = []

    class OrderedMethod(FakeMethod):
        def search(self, query, *, query_index, source_filter, limit, retrieval_data_version_id=None):
            events.append(self.method.value)
            return super().search(
                query,
                query_index=query_index,
                source_filter=source_filter,
                limit=limit,
                retrieval_data_version_id=retrieval_data_version_id,
            )

    class OrderedCoverageProbe:
        source_type = SourceType.CURRENT_SESSION

        def check_source_coverage(self, access, *, retrieval_data_version_id):
            assert access.source_type is SourceType.CURRENT_SESSION
            assert retrieval_data_version_id is None
            events.append("coverage")
            return None

    source_filter = SourceFilter.from_mapping(SourceType.CURRENT_SESSION, {"session_id": "s-a"})
    _service(
        sources={SourceType.CURRENT_SESSION: FakeSource(SourceType.CURRENT_SESSION, {})},
        methods={
            RetrievalMethod.DENSE: OrderedMethod(RetrievalMethod.DENSE),
            RetrievalMethod.LEARNED_SPARSE: OrderedMethod(RetrievalMethod.LEARNED_SPARSE),
        },
        source_coverage_probes={SourceType.CURRENT_SESSION: OrderedCoverageProbe()},
    ).retrieve_context(
        _request(source_filters={SourceType.CURRENT_SESSION: source_filter}),
        _budget(),
    )

    assert events == ["coverage", "dense", "learned_sparse", "coverage"]


def test_initial_coverage_gap_remains_partial_when_sync_completes_during_search():
    source_filter = SourceFilter.from_mapping(SourceType.CURRENT_SESSION, {"session_id": "s-a"})

    @dataclass
    class GapThenCoveredProbe:
        source_type: SourceType = SourceType.CURRENT_SESSION
        calls: int = 0

        def check_source_coverage(self, access, *, retrieval_data_version_id):
            assert access.source_type is SourceType.CURRENT_SESSION
            del retrieval_data_version_id
            self.calls += 1
            if self.calls == 1:
                return ContextCoverageLimitation(
                    SourceType.CURRENT_SESSION,
                    "current_session_index_coverage_incomplete",
                )
            return None

    coverage_probe = GapThenCoveredProbe()
    context = _service(
        sources={SourceType.CURRENT_SESSION: FakeSource(SourceType.CURRENT_SESSION, {})},
        methods={
            RetrievalMethod.DENSE: FakeMethod(RetrievalMethod.DENSE),
            RetrievalMethod.LEARNED_SPARSE: FakeMethod(RetrievalMethod.LEARNED_SPARSE),
        },
        source_coverage_probes={SourceType.CURRENT_SESSION: coverage_probe},
    ).retrieve_context(
        _request(source_filters={SourceType.CURRENT_SESSION: source_filter}),
        _budget(),
    )

    outcome = context.source_outcomes[SourceType.CURRENT_SESSION]
    assert coverage_probe.calls == 2
    assert outcome.retrieval.value == "partial"
    assert outcome.reason_code == "current_session_index_coverage_incomplete"
    assert context.status is ContextStatus.PARTIAL


def test_final_source_access_revalidation_marks_mid_read_document_revision_change_partial():
    source_filter = SourceFilter.from_mapping(
        SourceType.DOCUMENT,
        {"doc_id": "doc-1", "session_id": "s-a"},
    )

    class VersionedDocumentSource(FakeSource):
        def open_retrieval_access(self, source_filter):
            self.seen_filters.append(source_filter)
            return SourceAccess(
                SourceType.DOCUMENT,
                source_filter,
                SourceAvailability.READY,
                source_snapshot_id="initial-snapshot",
                source_revision_map={"doc-1": "v1"},
            )

    @dataclass
    class ChangedDocumentAccess:
        source_type: SourceType = SourceType.DOCUMENT
        calls: int = 0

        def revalidate_retrieval_access(self, access):
            self.calls += 1
            return SourceAccess(
                SourceType.DOCUMENT,
                access.source_filter,
                SourceAvailability.READY,
                source_snapshot_id=access.source_snapshot_id,
                source_revision_map={"doc-1": "v2"},
            )

    revalidator = ChangedDocumentAccess()
    context = _service(
        sources={SourceType.DOCUMENT: VersionedDocumentSource(SourceType.DOCUMENT, {})},
        methods={
            RetrievalMethod.DENSE: FakeMethod(RetrievalMethod.DENSE),
            RetrievalMethod.LEARNED_SPARSE: FakeMethod(RetrievalMethod.LEARNED_SPARSE),
        },
        source_access_revalidators={SourceType.DOCUMENT: revalidator},
    ).retrieve_context(
        _request(source_filters={SourceType.DOCUMENT: source_filter}),
        _budget(),
    )

    outcome = context.source_outcomes[SourceType.DOCUMENT]
    assert revalidator.calls == 1
    assert outcome.availability is SourceAvailability.READY
    assert outcome.retrieval.value == "partial"
    assert outcome.reason_code == "document_source_changed_during_retrieval"
    assert context.status is ContextStatus.PARTIAL


def test_project_document_search_uses_authorized_doc_filters_without_duplicate_units():
    session_filter = SourceFilter.from_mapping(
        SourceType.DOCUMENT,
        {"session_id": "s-a"},
    )
    doc_1 = _ref(
        "document-v3:doc-1:chunk-1",
        source_type=SourceType.DOCUMENT,
        revision="docv-1",
        content_hash="a" * 64,
    )
    doc_2 = _ref(
        "document-v3:doc-2:chunk-1",
        source_type=SourceType.DOCUMENT,
        revision="docv-2",
        content_hash="b" * 64,
    )

    class ProjectDocumentSource(FakeSource):
        def open_retrieval_access(self, source_filter):
            self.seen_filters.append(source_filter)
            return SourceAccess(
                SourceType.DOCUMENT,
                source_filter,
                SourceAvailability.READY,
                source_snapshot_id="snapshot-1",
                source_revision_map={"doc-1": "docv-1", "doc-2": "docv-2"},
            )

        def catalog_source_filters(self, access):
            return tuple(
                SourceFilter.from_mapping(SourceType.DOCUMENT, {"doc_id": doc_id})
                for doc_id in sorted(access.source_revision_map)
            )

    source = ProjectDocumentSource(
        SourceType.DOCUMENT,
        {
            doc_1: SourceUnit(doc_1, "document one"),
            doc_2: SourceUnit(doc_2, "document two"),
        },
    )
    hits = {
        (SourceType.DOCUMENT, "北桥项目"): (
            _candidate(doc_1, RetrievalMethod.DENSE, rank=1),
            _candidate(doc_2, RetrievalMethod.DENSE, rank=2),
        )
    }
    dense = FakeMethod(RetrievalMethod.DENSE, hits)

    context = _service(
        sources={SourceType.DOCUMENT: source},
        methods={RetrievalMethod.DENSE: dense},
    ).retrieve_context(
        _request(source_filters={SourceType.DOCUMENT: session_filter}),
        _budget(),
    )

    assert [item.as_mapping() for item in dense.seen_filters] == [
        {"doc_id": "doc-1"},
        {"doc_id": "doc-2"},
    ]
    assert {item.ref for item in context.items} == {doc_1, doc_2}
    assert len(context.items) == 2


def test_final_source_access_revalidation_discards_results_when_document_becomes_blocked():
    source_filter = SourceFilter.from_mapping(
        SourceType.DOCUMENT,
        {"doc_id": "doc-1", "session_id": "s-a"},
    )

    class ReadyDocumentSource(FakeSource):
        def open_retrieval_access(self, source_filter):
            return SourceAccess(
                SourceType.DOCUMENT,
                source_filter,
                SourceAvailability.READY,
                source_revision_map={"doc-1": "v1"},
            )

    class BlockedDocumentAccess:
        source_type = SourceType.DOCUMENT

        def revalidate_retrieval_access(self, access):
            return SourceAccess(
                SourceType.DOCUMENT,
                access.source_filter,
                SourceAvailability.BLOCKED,
                reason_code="document_source_changed",
            )

    context = _service(
        sources={SourceType.DOCUMENT: ReadyDocumentSource(SourceType.DOCUMENT, {})},
        methods={
            RetrievalMethod.DENSE: FakeMethod(RetrievalMethod.DENSE),
            RetrievalMethod.LEARNED_SPARSE: FakeMethod(RetrievalMethod.LEARNED_SPARSE),
        },
        source_access_revalidators={SourceType.DOCUMENT: BlockedDocumentAccess()},
    ).retrieve_context(
        _request(source_filters={SourceType.DOCUMENT: source_filter}),
        _budget(),
    )

    outcome = context.source_outcomes[SourceType.DOCUMENT]
    assert outcome.availability is SourceAvailability.BLOCKED
    assert outcome.retrieval.value == "not_run"
    assert outcome.reason_code == "document_source_changed"
    assert context.status is ContextStatus.PARTIAL


def test_source_coverage_limitation_keeps_verified_hits_but_marks_context_partial():
    ref = _ref("turn-covered-partially")
    source_filter = SourceFilter.from_mapping(SourceType.CURRENT_SESSION, {"session_id": "s-a"})
    source = FakeSource(SourceType.CURRENT_SESSION, {ref: SourceUnit(ref, "已验证的当前证据")})
    dense = FakeMethod(
        RetrievalMethod.DENSE,
        {(SourceType.CURRENT_SESSION, "北桥项目"): (_candidate(ref, RetrievalMethod.DENSE),)},
    )
    sparse = FakeMethod(
        RetrievalMethod.LEARNED_SPARSE,
        {(SourceType.CURRENT_SESSION, "北桥项目"): (_candidate(ref, RetrievalMethod.LEARNED_SPARSE),)},
    )
    context = _service(
        sources={SourceType.CURRENT_SESSION: source},
        methods={RetrievalMethod.DENSE: dense, RetrievalMethod.LEARNED_SPARSE: sparse},
        source_coverage_probes={
            SourceType.CURRENT_SESSION: StaticCoverageProbe(
                SourceType.CURRENT_SESSION,
                ContextCoverageLimitation(
                    SourceType.CURRENT_SESSION,
                    "current_session_index_coverage_incomplete",
                ),
            )
        },
    ).retrieve_context(
        _request(source_filters={SourceType.CURRENT_SESSION: source_filter}),
        _budget(),
    )

    assert [item.content for item in context.items] == ["已验证的当前证据"]
    assert context.status is ContextStatus.PARTIAL
    assert context.source_outcomes[SourceType.CURRENT_SESSION].retrieval.value == "partial"
    assert context.source_outcomes[SourceType.CURRENT_SESSION].reason_code == (
        "current_session_index_coverage_incomplete"
    )
    assert [
        limitation
        for limitation in context.coverage_limitations
        if limitation.source_type is SourceType.CURRENT_SESSION
    ] == [
        ContextCoverageLimitation(
            SourceType.CURRENT_SESSION,
            "current_session_index_coverage_incomplete",
        )
    ]


def test_failed_coverage_probe_returns_safe_partial_instead_of_raising():
    source_filter = SourceFilter.from_mapping(SourceType.CURRENT_SESSION, {"session_id": "s-a"})
    context = _service(
        sources={SourceType.CURRENT_SESSION: FakeSource(SourceType.CURRENT_SESSION, {})},
        methods={
            RetrievalMethod.DENSE: FakeMethod(RetrievalMethod.DENSE),
            RetrievalMethod.LEARNED_SPARSE: FakeMethod(RetrievalMethod.LEARNED_SPARSE),
        },
        source_coverage_probes={
            SourceType.CURRENT_SESSION: FailingCoverageProbe(SourceType.CURRENT_SESSION)
        },
    ).retrieve_context(
        _request(source_filters={SourceType.CURRENT_SESSION: source_filter}),
        _budget(),
    )

    outcome = context.source_outcomes[SourceType.CURRENT_SESSION]
    assert context.status is ContextStatus.PARTIAL
    assert outcome.retrieval.value == "partial"
    assert outcome.reason_code == "source_index_coverage_check_unavailable"
    assert [
        limitation
        for limitation in context.coverage_limitations
        if limitation.source_type is SourceType.CURRENT_SESSION
    ] == [
        ContextCoverageLimitation(
            SourceType.CURRENT_SESSION,
            "source_index_coverage_check_unavailable",
        )
    ]


def test_source_hint_cannot_expand_the_trusted_boundary():
    ref = _ref("turn-8")
    session_filter = SourceFilter.from_mapping(SourceType.CURRENT_SESSION, {"session_id": "s-a"})
    source = FakeSource(SourceType.CURRENT_SESSION, {ref: SourceUnit(ref, "北桥项目下周交付")})
    dense = FakeMethod(RetrievalMethod.DENSE)
    sparse = FakeMethod(RetrievalMethod.LEARNED_SPARSE)
    request = _request(source_filters={SourceType.CURRENT_SESSION: session_filter})
    request = RetrievalRequest(
        request_id=request.request_id,
        model_call_purpose=request.model_call_purpose,
        query_proposal=QueryProposal(("北桥项目",), frozenset({SourceType.DOCUMENT})),
        boundary=request.boundary,
    )

    context = _service(
        sources={SourceType.CURRENT_SESSION: source},
        methods={RetrievalMethod.DENSE: dense, RetrievalMethod.LEARNED_SPARSE: sparse},
    ).retrieve_context(request, _budget())

    assert tuple(context.source_outcomes) == tuple(SourceType)
    assert context.source_outcomes[SourceType.DOCUMENT].availability is SourceAvailability.DISABLED
    assert context.source_outcomes[SourceType.DOCUMENT].retrieval.value == "not_run"
    assert context.source_outcomes[SourceType.DOCUMENT].reason_code == "source_excluded_by_trusted_boundary"
    assert not dense.seen_filters or all(item.source_type is SourceType.CURRENT_SESSION for item in dense.seen_filters)


def test_policy_omission_of_an_allowed_source_is_an_explicit_failed_outcome():
    source_filter = SourceFilter.from_mapping(SourceType.CURRENT_SESSION, {"session_id": "s-a"})

    class OmittingPolicy:
        def compile(self, request, budget):
            del request, budget
            return RetrievalPlan(source_plans=())

    context = RetrievalService(
        policy=OmittingPolicy(),
        query_guard=QueryGuard(),
        sources={},
        methods={},
        token_estimator=lambda text: len(text.split()),
    ).retrieve_context(
        _request(source_filters={SourceType.CURRENT_SESSION: source_filter}),
        _budget(),
    )

    outcome = context.source_outcomes[SourceType.CURRENT_SESSION]
    assert context.status is ContextStatus.PARTIAL
    assert outcome.availability is SourceAvailability.UNAVAILABLE
    assert outcome.retrieval.value == "failed"
    assert outcome.reason_code == "retrieval_policy_omitted_allowed_source"
    assert tuple(context.source_outcomes) == tuple(SourceType)


def test_missing_adapter_is_explicit_and_blocks_when_the_source_is_required():
    document_filter = SourceFilter.from_mapping(SourceType.DOCUMENT, {"session_id": "s-a"})
    optional = _service(
        sources={},
        methods={RetrievalMethod.DENSE: FakeMethod(RetrievalMethod.DENSE)},
    ).retrieve_context(
        _request(source_filters={SourceType.DOCUMENT: document_filter}),
        _budget(),
    )
    required = _service(
        sources={},
        methods={RetrievalMethod.DENSE: FakeMethod(RetrievalMethod.DENSE)},
    ).retrieve_context(
        _request(
            source_filters={SourceType.DOCUMENT: document_filter},
            dependencies={SourceType.DOCUMENT: SourceDependency.REQUIRED},
        ),
        _budget(),
    )

    optional_outcome = optional.source_outcomes[SourceType.DOCUMENT]
    required_outcome = required.source_outcomes[SourceType.DOCUMENT]
    assert optional.status is ContextStatus.PARTIAL
    assert required.status is ContextStatus.BLOCKED
    assert optional_outcome.availability is SourceAvailability.NOT_IMPLEMENTED
    assert required_outcome.availability is SourceAvailability.NOT_IMPLEMENTED
    assert optional_outcome.retrieval.value == "not_run"


def test_document_freshness_block_is_not_misreported_as_no_match():
    class FreshnessBlockedDocumentSource(FakeSource):
        def open_retrieval_access(self, source_filter: SourceFilter) -> SourceAccess:
            self.seen_filters.append(source_filter)
            return SourceAccess(
                SourceType.DOCUMENT,
                source_filter,
                SourceAvailability.BLOCKED,
                reason_code="document_source_changed",
            )

    source_filter = SourceFilter.from_mapping(SourceType.DOCUMENT, {"doc_id": "doc-1", "session_id": "s-a"})
    source = FreshnessBlockedDocumentSource(SourceType.DOCUMENT, {})
    dense = FakeMethod(RetrievalMethod.DENSE)
    sparse = FakeMethod(RetrievalMethod.LEARNED_SPARSE)

    optional = _service(
        sources={SourceType.DOCUMENT: source},
        methods={RetrievalMethod.DENSE: dense, RetrievalMethod.LEARNED_SPARSE: sparse},
    ).retrieve_context(
        _request(source_filters={SourceType.DOCUMENT: source_filter}),
        _budget(),
    )
    required = _service(
        sources={SourceType.DOCUMENT: FreshnessBlockedDocumentSource(SourceType.DOCUMENT, {})},
        methods={RetrievalMethod.DENSE: FakeMethod(RetrievalMethod.DENSE)},
    ).retrieve_context(
        _request(
            source_filters={SourceType.DOCUMENT: source_filter},
            dependencies={SourceType.DOCUMENT: SourceDependency.REQUIRED},
        ),
        _budget(),
    )

    outcome = optional.source_outcomes[SourceType.DOCUMENT]
    assert optional.status is ContextStatus.PARTIAL
    assert required.status is ContextStatus.BLOCKED
    assert outcome.availability is SourceAvailability.BLOCKED
    assert outcome.retrieval.value == "not_run"
    assert outcome.reason_code == "document_source_changed"
    assert dense.seen_filters == []
    assert sparse.seen_filters == []


def test_direct_history_reference_is_not_returned_twice():
    ref = _ref("turn-8")
    source_filter = SourceFilter.from_mapping(SourceType.CURRENT_SESSION, {"session_id": "s-a"})
    source = FakeSource(SourceType.CURRENT_SESSION, {ref: SourceUnit(ref, "已经直接注入的历史")})
    dense = FakeMethod(RetrievalMethod.DENSE, {(SourceType.CURRENT_SESSION, "北桥项目"): (_candidate(ref, RetrievalMethod.DENSE),)})
    sparse = FakeMethod(RetrievalMethod.LEARNED_SPARSE)

    context = _service(
        sources={SourceType.CURRENT_SESSION: source},
        methods={RetrievalMethod.DENSE: dense, RetrievalMethod.LEARNED_SPARSE: sparse},
    ).retrieve_context(
        _request(source_filters={SourceType.CURRENT_SESSION: source_filter}, excluded=frozenset({ref})),
        _budget(),
    )

    assert context.items == ()
    assert source.seen_filters == [source_filter]


def test_context_reports_countable_pack_omissions_without_hiding_truncation():
    first, second = _ref("turn-a"), _ref("turn-b")
    source_filter = SourceFilter.from_mapping(SourceType.CURRENT_SESSION, {"session_id": "s-a"})
    source = FakeSource(
        SourceType.CURRENT_SESSION,
        {
            first: SourceUnit(first, "first"),
            second: SourceUnit(second, "second"),
        },
    )
    dense = FakeMethod(
        RetrievalMethod.DENSE,
        {
            (SourceType.CURRENT_SESSION, "北桥项目"): (
                _candidate(first, RetrievalMethod.DENSE, rank=1),
                _candidate(second, RetrievalMethod.DENSE, rank=2),
            )
        },
    )

    context = _service(
        sources={SourceType.CURRENT_SESSION: source},
        methods={
            RetrievalMethod.DENSE: dense,
            RetrievalMethod.LEARNED_SPARSE: FakeMethod(RetrievalMethod.LEARNED_SPARSE),
        },
    ).retrieve_context(
        _request(source_filters={SourceType.CURRENT_SESSION: source_filter}),
        RetrievalBudget(candidate_limit_per_source=5, context_token_limit=30, max_items=1),
    )

    assert context.status is ContextStatus.PARTIAL
    assert context.truncated is True
    assert [item.ref for item in context.items] == [first]
    assert len(context.pack_omissions) == 1
    omission = context.pack_omissions[0]
    assert omission.source_type is SourceType.CURRENT_SESSION
    assert omission.reason is ContextPackOmissionReason.MAX_ITEMS
    assert omission.count == 1
    assert any(
        limitation.source_type is SourceType.DOCUMENT
        and limitation.reason_code == "source_excluded_by_trusted_boundary"
        for limitation in context.coverage_limitations
    )


def test_current_session_unavailable_blocks_all_context_content():
    source_filter = SourceFilter.from_mapping(SourceType.CURRENT_SESSION, {"session_id": "s-a"})
    source = FakeSource(SourceType.CURRENT_SESSION, {}, SourceAvailability.UNAVAILABLE)
    context = _service(
        sources={SourceType.CURRENT_SESSION: source},
        methods={
            RetrievalMethod.DENSE: FakeMethod(RetrievalMethod.DENSE),
            RetrievalMethod.LEARNED_SPARSE: FakeMethod(RetrievalMethod.LEARNED_SPARSE),
        },
    ).retrieve_context(_request(source_filters={SourceType.CURRENT_SESSION: source_filter}), _budget())

    assert context.status is ContextStatus.BLOCKED
    assert context.items == ()
    assert any(
        limitation.source_type is SourceType.CURRENT_SESSION
        and limitation.reason_code == "source_unavailable"
        for limitation in context.coverage_limitations
    )


def test_early_global_block_never_marks_long_term_memory_recall_as_clear():
    session_filter = SourceFilter.from_mapping(SourceType.CURRENT_SESSION, {"session_id": "s-a"})
    user_filter = SourceFilter.from_mapping(SourceType.LONG_TERM_USER, {"user_id": "u-a"})
    task_filter = SourceFilter.from_mapping(SourceType.LONG_TERM_TASK, {"task_id": "t-a"})
    context = _service(
        sources={
            SourceType.CURRENT_SESSION: FakeSource(
                SourceType.CURRENT_SESSION,
                {},
                SourceAvailability.UNAVAILABLE,
            ),
            SourceType.LONG_TERM_USER: FakeSource(SourceType.LONG_TERM_USER, {}),
            SourceType.LONG_TERM_TASK: FakeSource(SourceType.LONG_TERM_TASK, {}),
        },
        methods={
            RetrievalMethod.DENSE: FakeMethod(RetrievalMethod.DENSE),
            RetrievalMethod.LEARNED_SPARSE: FakeMethod(RetrievalMethod.LEARNED_SPARSE),
        },
    ).retrieve_context(
        _request(
            source_filters={
                SourceType.CURRENT_SESSION: session_filter,
                SourceType.LONG_TERM_USER: user_filter,
                SourceType.LONG_TERM_TASK: task_filter,
            }
        ),
        _budget(),
    )

    assert context.status is ContextStatus.BLOCKED
    assert context.long_term_memory_write_guard.status is LongTermMemoryWriteGuardStatus.NOT_EVALUATED


def test_long_term_recall_fault_blocks_both_memory_write_types_without_blocking_answer():
    user_filter = SourceFilter.from_mapping(SourceType.LONG_TERM_USER, {"user_id": "u-a"})
    task_filter = SourceFilter.from_mapping(SourceType.LONG_TERM_TASK, {"task_id": "t-a"})
    task_ref = _ref("memory-task-1", source_type=SourceType.LONG_TERM_TASK)
    task_source = FakeSource(
        SourceType.LONG_TERM_TASK,
        {task_ref: SourceUnit(task_ref, "北桥项目的任务事实")},
    )
    context = _service(
        sources={
            SourceType.LONG_TERM_USER: FakeSource(
                SourceType.LONG_TERM_USER,
                {},
                SourceAvailability.UNAVAILABLE,
            ),
            SourceType.LONG_TERM_TASK: task_source,
        },
        methods={
            RetrievalMethod.DENSE: FakeMethod(
                RetrievalMethod.DENSE,
                {(SourceType.LONG_TERM_TASK, "北桥项目"): (_candidate(task_ref, RetrievalMethod.DENSE),)},
            ),
            RetrievalMethod.LEARNED_SPARSE: FakeMethod(RetrievalMethod.LEARNED_SPARSE),
        },
    ).retrieve_context(
        _request(source_filters={SourceType.LONG_TERM_USER: user_filter, SourceType.LONG_TERM_TASK: task_filter}),
        _budget(),
    )

    assert context.status is ContextStatus.PARTIAL
    assert [item.ref for item in context.items] == [task_ref]
    assert context.long_term_memory_write_guard.status is LongTermMemoryWriteGuardStatus.BLOCKED
    assert context.long_term_memory_write_guard.blocking_sources == (SourceType.LONG_TERM_USER,)
    assert context.long_term_memory_write_guard.reason_codes == (
        "long_term_user:availability_unavailable",
    )


def test_long_term_memory_guard_is_clear_only_after_both_sources_complete_normally():
    user_filter = SourceFilter.from_mapping(SourceType.LONG_TERM_USER, {"user_id": "u-a"})
    task_filter = SourceFilter.from_mapping(SourceType.LONG_TERM_TASK, {"task_id": "t-a"})
    context = _service(
        sources={
            SourceType.LONG_TERM_USER: FakeSource(SourceType.LONG_TERM_USER, {}),
            SourceType.LONG_TERM_TASK: FakeSource(SourceType.LONG_TERM_TASK, {}),
        },
        methods={
            RetrievalMethod.DENSE: FakeMethod(RetrievalMethod.DENSE),
            RetrievalMethod.LEARNED_SPARSE: FakeMethod(RetrievalMethod.LEARNED_SPARSE),
        },
    ).retrieve_context(
        _request(source_filters={SourceType.LONG_TERM_USER: user_filter, SourceType.LONG_TERM_TASK: task_filter}),
        _budget(),
    )

    assert context.status is ContextStatus.COMPLETE
    assert context.long_term_memory_write_guard.status is LongTermMemoryWriteGuardStatus.CLEAR
    assert context.long_term_memory_write_guard.evaluated_sources == (
        SourceType.LONG_TERM_USER,
        SourceType.LONG_TERM_TASK,
    )
    assert context.long_term_memory_write_guard.blocking_sources == ()


def test_long_term_memory_guard_is_not_evaluated_when_one_source_is_outside_boundary():
    user_filter = SourceFilter.from_mapping(SourceType.LONG_TERM_USER, {"user_id": "u-a"})
    context = _service(
        sources={SourceType.LONG_TERM_USER: FakeSource(SourceType.LONG_TERM_USER, {})},
        methods={
            RetrievalMethod.DENSE: FakeMethod(RetrievalMethod.DENSE),
            RetrievalMethod.LEARNED_SPARSE: FakeMethod(RetrievalMethod.LEARNED_SPARSE),
        },
    ).retrieve_context(_request(source_filters={SourceType.LONG_TERM_USER: user_filter}), _budget())

    assert context.long_term_memory_write_guard.status is LongTermMemoryWriteGuardStatus.NOT_EVALUATED
    assert context.long_term_memory_write_guard.evaluated_sources == (SourceType.LONG_TERM_USER,)


def test_long_term_partial_source_verification_blocks_memory_writes():
    source_filter = SourceFilter.from_mapping(SourceType.LONG_TERM_USER, {"user_id": "u-a"})
    kept_ref = _ref("memory-user-1", source_type=SourceType.LONG_TERM_USER)
    missing_ref = _ref("memory-user-2", source_type=SourceType.LONG_TERM_USER)
    source = FakeSource(
        SourceType.LONG_TERM_USER,
        {kept_ref: SourceUnit(kept_ref, "可验证的用户长期事实")},
    )
    context = _service(
        sources={SourceType.LONG_TERM_USER: source},
        methods={
            RetrievalMethod.DENSE: FakeMethod(
                RetrievalMethod.DENSE,
                {
                    (SourceType.LONG_TERM_USER, "北桥项目"): (
                        _candidate(kept_ref, RetrievalMethod.DENSE),
                        _candidate(missing_ref, RetrievalMethod.DENSE, rank=2),
                    )
                },
            ),
            RetrievalMethod.LEARNED_SPARSE: FakeMethod(RetrievalMethod.LEARNED_SPARSE),
        },
    ).retrieve_context(_request(source_filters={SourceType.LONG_TERM_USER: source_filter}), _budget())

    assert context.source_outcomes[SourceType.LONG_TERM_USER].retrieval.value == "partial"
    assert context.long_term_memory_write_guard.status is LongTermMemoryWriteGuardStatus.BLOCKED
    assert context.long_term_memory_write_guard.blocking_sources == (SourceType.LONG_TERM_USER,)


def test_required_document_unavailable_blocks_context_but_optional_document_is_partial():
    source_filter = SourceFilter.from_mapping(SourceType.DOCUMENT, {"doc_id": "d-a"})
    source = FakeSource(SourceType.DOCUMENT, {}, SourceAvailability.UNAVAILABLE)
    service = _service(
        sources={SourceType.DOCUMENT: source},
        methods={RetrievalMethod.DENSE: FakeMethod(RetrievalMethod.DENSE)},
    )
    required = service.retrieve_context(
        _request(
            source_filters={SourceType.DOCUMENT: source_filter},
            dependencies={SourceType.DOCUMENT: SourceDependency.REQUIRED},
        ),
        _budget(),
    )
    optional = service.retrieve_context(
        _request(source_filters={SourceType.DOCUMENT: source_filter}),
        _budget(),
    )

    assert required.status is ContextStatus.BLOCKED
    assert optional.status is ContextStatus.PARTIAL


def test_stale_or_missing_source_content_is_dropped_after_lightweight_retrieval():
    ref = _ref("turn-8")
    stale_ref = _ref("turn-8", revision="r2", content_hash="h2")
    source_filter = SourceFilter.from_mapping(SourceType.CURRENT_SESSION, {"session_id": "s-a"})
    source = FakeSource(SourceType.CURRENT_SESSION, {stale_ref: SourceUnit(stale_ref, "更新后的正文")})
    dense = FakeMethod(RetrievalMethod.DENSE, {(SourceType.CURRENT_SESSION, "北桥项目"): (_candidate(ref, RetrievalMethod.DENSE),)})
    sparse = FakeMethod(RetrievalMethod.LEARNED_SPARSE)

    context = _service(
        sources={SourceType.CURRENT_SESSION: source},
        methods={RetrievalMethod.DENSE: dense, RetrievalMethod.LEARNED_SPARSE: sparse},
    ).retrieve_context(_request(source_filters={SourceType.CURRENT_SESSION: source_filter}), _budget())

    assert context.status is ContextStatus.PARTIAL
    assert context.items == ()
    assert "source_unit_missing:current_session" in context.diagnostic_codes
    assert context.verification_drops[0].source_type is SourceType.CURRENT_SESSION
    assert context.verification_drops[0].reason is ContextVerificationDropReason.SOURCE_UNIT_MISSING
    assert context.verification_drops[0].count == 1


def test_invalid_candidate_lane_is_rejected_without_overwriting_another_lane_no_match():
    session_filter = SourceFilter.from_mapping(SourceType.CURRENT_SESSION, {"session_id": "s-a"})
    bad_ref = _ref("doc-1", source_type=SourceType.DOCUMENT)
    source = FakeSource(SourceType.CURRENT_SESSION, {})
    dense = FakeMethod(RetrievalMethod.DENSE, {(SourceType.CURRENT_SESSION, "北桥项目"): (_candidate(bad_ref, RetrievalMethod.DENSE),)})
    sparse = FakeMethod(RetrievalMethod.LEARNED_SPARSE)

    context = _service(
        sources={SourceType.CURRENT_SESSION: source},
        methods={RetrievalMethod.DENSE: dense, RetrievalMethod.LEARNED_SPARSE: sparse},
    ).retrieve_context(_request(source_filters={SourceType.CURRENT_SESSION: session_filter}), _budget())

    assert context.status is ContextStatus.PARTIAL
    assert context.source_outcomes[SourceType.CURRENT_SESSION].retrieval.value == "no_match"
    assert any(outcome.status.value == "failed" for outcome in context.method_outcomes)


def test_bm25_is_a_degraded_fallback_not_a_default_third_rrf_vote():
    ref = _ref("turn-8")
    source_filter = SourceFilter.from_mapping(SourceType.CURRENT_SESSION, {"session_id": "s-a"})
    source = FakeSource(SourceType.CURRENT_SESSION, {ref: SourceUnit(ref, "北桥项目下周交付")})
    dense = FakeMethod(RetrievalMethod.DENSE, fail=True)
    sparse = FakeMethod(RetrievalMethod.LEARNED_SPARSE)
    bm25 = FakeMethod(
        RetrievalMethod.BM25,
        {(SourceType.CURRENT_SESSION, "北桥项目"): (_candidate(ref, RetrievalMethod.BM25),)},
    )

    context = _service(
        sources={SourceType.CURRENT_SESSION: source},
        methods={
            RetrievalMethod.DENSE: dense,
            RetrievalMethod.LEARNED_SPARSE: sparse,
            RetrievalMethod.BM25: bm25,
        },
    ).retrieve_context(_request(source_filters={SourceType.CURRENT_SESSION: source_filter}), _budget())

    bm25_outcome = next(outcome for outcome in context.method_outcomes if outcome.method is RetrievalMethod.BM25)
    assert [item.ref for item in context.items] == [ref]
    assert bm25_outcome.status.value == "degraded"
    assert bm25_outcome.degraded_from == (RetrievalMethod.DENSE,)
    assert bm25_outcome.attempt == 2


def test_bm25_can_be_an_explicit_primary_lane_without_running_twice():
    ref = _ref("turn-8")
    source_filter = SourceFilter.from_mapping(
        SourceType.CURRENT_SESSION,
        {"session_id": "s-a"},
    )
    source = FakeSource(
        SourceType.CURRENT_SESSION,
        {ref: SourceUnit(ref, "北桥项目下周交付")},
    )
    bm25 = FakeMethod(
        RetrievalMethod.BM25,
        {
            (SourceType.CURRENT_SESSION, "北桥项目"): (
                _candidate(ref, RetrievalMethod.BM25),
            )
        },
    )
    service = RetrievalService(
        policy=DefaultRetrievalPolicy(default_methods=(RetrievalMethod.BM25,)),
        query_guard=QueryGuard(),
        sources={SourceType.CURRENT_SESSION: source},
        methods={RetrievalMethod.BM25: bm25},
        token_estimator=lambda text: len(text.split()),
    )

    context = service.retrieve_context(
        _request(source_filters={SourceType.CURRENT_SESSION: source_filter}),
        _budget(),
    )

    bm25_outcomes = [
        outcome
        for outcome in context.method_outcomes
        if outcome.method is RetrievalMethod.BM25
    ]
    assert [item.ref for item in context.items] == [ref]
    assert len(bm25_outcomes) == 1
    assert bm25_outcomes[0].status.value == "used"
    assert bm25_outcomes[0].attempt == 1


def test_literal_boolean_is_only_used_after_bm25_fails():
    ref = _ref("turn-8")
    source_filter = SourceFilter.from_mapping(SourceType.CURRENT_SESSION, {"session_id": "s-a"})
    source = FakeSource(SourceType.CURRENT_SESSION, {ref: SourceUnit(ref, "北桥项目下周交付")})
    dense = FakeMethod(RetrievalMethod.DENSE, fail=True)
    sparse = FakeMethod(RetrievalMethod.LEARNED_SPARSE, fail=True)
    bm25 = FakeMethod(RetrievalMethod.BM25, fail=True)
    literal = FakeMethod(
        RetrievalMethod.LITERAL_BOOLEAN,
        {(SourceType.CURRENT_SESSION, "北桥项目"): (_candidate(ref, RetrievalMethod.LITERAL_BOOLEAN),)},
    )

    context = _service(
        sources={SourceType.CURRENT_SESSION: source},
        methods={
            RetrievalMethod.DENSE: dense,
            RetrievalMethod.LEARNED_SPARSE: sparse,
            RetrievalMethod.BM25: bm25,
            RetrievalMethod.LITERAL_BOOLEAN: literal,
        },
    ).retrieve_context(_request(source_filters={SourceType.CURRENT_SESSION: source_filter}), _budget())

    literal_outcome = next(
        outcome for outcome in context.method_outcomes if outcome.method is RetrievalMethod.LITERAL_BOOLEAN
    )
    assert [item.ref for item in context.items] == [ref]
    assert literal_outcome.status.value == "degraded"
    assert literal_outcome.degraded_from == (RetrievalMethod.BM25,)
    assert literal_outcome.attempt == 3


def test_multiple_queries_round_robin_candidates_before_source_budget_is_exhausted():
    source_filter = SourceFilter.from_mapping(SourceType.CURRENT_SESSION, {"session_id": "s-a"})
    ref_a, ref_b, ref_c = _ref("turn-a"), _ref("turn-b"), _ref("turn-c")
    source = FakeSource(
        SourceType.CURRENT_SESSION,
        {
            ref_a: SourceUnit(ref_a, "a"),
            ref_b: SourceUnit(ref_b, "b"),
            ref_c: SourceUnit(ref_c, "c"),
        },
    )
    dense = FakeMethod(
        RetrievalMethod.DENSE,
        {
            (SourceType.CURRENT_SESSION, "broad"): (
                _candidate(ref_a, RetrievalMethod.DENSE, query_index=0, rank=1),
                _candidate(ref_b, RetrievalMethod.DENSE, query_index=0, rank=2),
            ),
            (SourceType.CURRENT_SESSION, "narrow"): (
                _candidate(ref_c, RetrievalMethod.DENSE, query_index=1, rank=1),
            ),
        },
    )
    sparse = FakeMethod(RetrievalMethod.LEARNED_SPARSE)
    request = _request(source_filters={SourceType.CURRENT_SESSION: source_filter})
    request = RetrievalRequest(
        request_id=request.request_id,
        model_call_purpose=request.model_call_purpose,
        query_proposal=QueryProposal(("broad", "narrow")),
        boundary=request.boundary,
    )
    budget = RetrievalBudget(candidate_limit_per_source=3, context_token_limit=30, max_items=3)

    context = _service(
        sources={SourceType.CURRENT_SESSION: source},
        methods={RetrievalMethod.DENSE: dense, RetrievalMethod.LEARNED_SPARSE: sparse},
    ).retrieve_context(request, budget)

    assert [item.ref for item in context.items] == [ref_a, ref_c, ref_b]


def test_configured_long_term_memory_pool_alternates_user_and_task_without_cross_source_scores():
    user_filter = SourceFilter.from_mapping(SourceType.LONG_TERM_USER, {"principal_id": "p1"})
    task_filter = SourceFilter.from_mapping(
        SourceType.LONG_TERM_TASK,
        {"principal_id": "p1", "task_id": "task-1"},
    )
    user_first = _ref("user-1", source_type=SourceType.LONG_TERM_USER)
    user_second = _ref("user-2", source_type=SourceType.LONG_TERM_USER)
    task_first = _ref("task-1", source_type=SourceType.LONG_TERM_TASK)
    task_second = _ref("task-2", source_type=SourceType.LONG_TERM_TASK)
    dense = FakeMethod(
        RetrievalMethod.DENSE,
        {
            (SourceType.LONG_TERM_USER, "memory"): (
                _candidate(user_first, RetrievalMethod.DENSE, rank=1),
                _candidate(user_second, RetrievalMethod.DENSE, rank=2),
            ),
            (SourceType.LONG_TERM_TASK, "memory"): (
                _candidate(task_first, RetrievalMethod.DENSE, rank=1),
                _candidate(task_second, RetrievalMethod.DENSE, rank=2),
            ),
        },
    )
    sources = {
        SourceType.LONG_TERM_USER: FakeSource(
            SourceType.LONG_TERM_USER,
            {
                user_first: SourceUnit(user_first, "user first"),
                user_second: SourceUnit(user_second, "user second"),
            },
        ),
        SourceType.LONG_TERM_TASK: FakeSource(
            SourceType.LONG_TERM_TASK,
            {
                task_first: SourceUnit(task_first, "task first"),
                task_second: SourceUnit(task_second, "task second"),
            },
        ),
    }
    request = _request(
        source_filters={
            SourceType.LONG_TERM_USER: user_filter,
            SourceType.LONG_TERM_TASK: task_filter,
        },
        query="memory",
    )

    context = _service(
        sources=sources,
        methods={RetrievalMethod.DENSE: dense, RetrievalMethod.LEARNED_SPARSE: FakeMethod(RetrievalMethod.LEARNED_SPARSE)},
        source_selection_quotas=SourceSelectionQuotas(
            {SourceQuotaPool.LONG_TERM_MEMORY: 3},
        ),
    ).retrieve_context(
        request,
        RetrievalBudget(candidate_limit_per_source=5, context_token_limit=30, max_items=5),
    )

    assert [item.ref for item in context.items] == [user_first, task_first, user_second]
    assert "source_pool_item_quota_reached:long_term_memory" in context.diagnostic_codes
    assert context.truncated is True
    assert context.pack_omissions == (
        ContextPackOmission(
            source_type=SourceType.LONG_TERM_TASK,
            reason=ContextPackOmissionReason.SOURCE_POOL_ITEM_QUOTA,
            count=1,
        ),
    )


def test_long_term_memory_pool_borrows_unused_slots_when_one_source_has_no_hits():
    user_filter = SourceFilter.from_mapping(SourceType.LONG_TERM_USER, {"principal_id": "p1"})
    task_filter = SourceFilter.from_mapping(
        SourceType.LONG_TERM_TASK,
        {"principal_id": "p1", "task_id": "task-1"},
    )
    user_first = _ref("user-1", source_type=SourceType.LONG_TERM_USER)
    user_second = _ref("user-2", source_type=SourceType.LONG_TERM_USER)
    dense = FakeMethod(
        RetrievalMethod.DENSE,
        {
            (SourceType.LONG_TERM_USER, "memory"): (
                _candidate(user_first, RetrievalMethod.DENSE, rank=1),
                _candidate(user_second, RetrievalMethod.DENSE, rank=2),
            ),
        },
    )
    request = _request(
        source_filters={
            SourceType.LONG_TERM_USER: user_filter,
            SourceType.LONG_TERM_TASK: task_filter,
        },
        query="memory",
    )
    context = _service(
        sources={
            SourceType.LONG_TERM_USER: FakeSource(
                SourceType.LONG_TERM_USER,
                {
                    user_first: SourceUnit(user_first, "user first"),
                    user_second: SourceUnit(user_second, "user second"),
                },
            ),
            SourceType.LONG_TERM_TASK: FakeSource(SourceType.LONG_TERM_TASK, {}),
        },
        methods={RetrievalMethod.DENSE: dense, RetrievalMethod.LEARNED_SPARSE: FakeMethod(RetrievalMethod.LEARNED_SPARSE)},
        source_selection_quotas=SourceSelectionQuotas(
            {SourceQuotaPool.LONG_TERM_MEMORY: 3},
        ),
    ).retrieve_context(
        request,
        RetrievalBudget(candidate_limit_per_source=5, context_token_limit=30, max_items=5),
    )

    assert [item.ref for item in context.items] == [user_first, user_second]


def test_empty_source_quota_configuration_preserves_legacy_source_order():
    user_filter = SourceFilter.from_mapping(SourceType.LONG_TERM_USER, {"principal_id": "p1"})
    task_filter = SourceFilter.from_mapping(
        SourceType.LONG_TERM_TASK,
        {"principal_id": "p1", "task_id": "task-1"},
    )
    user_first = _ref("user-1", source_type=SourceType.LONG_TERM_USER)
    user_second = _ref("user-2", source_type=SourceType.LONG_TERM_USER)
    task_first = _ref("task-1", source_type=SourceType.LONG_TERM_TASK)
    dense = FakeMethod(
        RetrievalMethod.DENSE,
        {
            (SourceType.LONG_TERM_USER, "memory"): (
                _candidate(user_first, RetrievalMethod.DENSE, rank=1),
                _candidate(user_second, RetrievalMethod.DENSE, rank=2),
            ),
            (SourceType.LONG_TERM_TASK, "memory"): (
                _candidate(task_first, RetrievalMethod.DENSE, rank=1),
            ),
        },
    )
    request = _request(
        source_filters={
            SourceType.LONG_TERM_USER: user_filter,
            SourceType.LONG_TERM_TASK: task_filter,
        },
        query="memory",
    )
    context = _service(
        sources={
            SourceType.LONG_TERM_USER: FakeSource(
                SourceType.LONG_TERM_USER,
                {
                    user_first: SourceUnit(user_first, "user first"),
                    user_second: SourceUnit(user_second, "user second"),
                },
            ),
            SourceType.LONG_TERM_TASK: FakeSource(
                SourceType.LONG_TERM_TASK,
                {task_first: SourceUnit(task_first, "task first")},
            ),
        },
        methods={RetrievalMethod.DENSE: dense, RetrievalMethod.LEARNED_SPARSE: FakeMethod(RetrievalMethod.LEARNED_SPARSE)},
    ).retrieve_context(
        request,
        RetrievalBudget(candidate_limit_per_source=5, context_token_limit=30, max_items=3),
    )

    assert [item.ref for item in context.items] == [user_first, user_second, task_first]


def test_transient_method_failure_recovers_within_bounded_attempt_budget():
    ref = _ref("turn-8")
    source_filter = SourceFilter.from_mapping(SourceType.CURRENT_SESSION, {"session_id": "s-a"})
    source = FakeSource(SourceType.CURRENT_SESSION, {ref: SourceUnit(ref, "北桥项目下周交付")})
    dense = TimeoutThenHitsMethod(
        method=RetrievalMethod.DENSE,
        hits={(SourceType.CURRENT_SESSION, "北桥项目"): (_candidate(ref, RetrievalMethod.DENSE),)},
        failures_before_success=2,
    )
    sparse = FakeMethod(RetrievalMethod.LEARNED_SPARSE)

    context = _service(
        sources={SourceType.CURRENT_SESSION: source},
        methods={RetrievalMethod.DENSE: dense, RetrievalMethod.LEARNED_SPARSE: sparse},
    ).retrieve_context(_request(source_filters={SourceType.CURRENT_SESSION: source_filter}), _budget())

    dense_outcome = next(outcome for outcome in context.method_outcomes if outcome.method is RetrievalMethod.DENSE)
    assert [item.ref for item in context.items] == [ref]
    assert dense.calls == 3
    assert dense_outcome.infrastructure_attempts == 3
    assert context.status is ContextStatus.COMPLETE


def test_exhausted_method_circuit_prevents_repeating_backend_calls_for_later_queries():
    source_filter = SourceFilter.from_mapping(SourceType.CURRENT_SESSION, {"session_id": "s-a"})
    source = FakeSource(SourceType.CURRENT_SESSION, {})
    dense = TimeoutThenHitsMethod(method=RetrievalMethod.DENSE, failures_before_success=99)
    service = RetrievalService(
        policy=DefaultRetrievalPolicy(default_methods=(RetrievalMethod.DENSE,)),
        query_guard=QueryGuard(),
        sources={SourceType.CURRENT_SESSION: source},
        methods={RetrievalMethod.DENSE: dense},
        token_estimator=lambda text: len(text.split()),
    )
    request = _request(source_filters={SourceType.CURRENT_SESSION: source_filter})
    request = RetrievalRequest(
        request_id=request.request_id,
        model_call_purpose=request.model_call_purpose,
        query_proposal=QueryProposal(("broad", "narrow")),
        boundary=request.boundary,
    )

    context = service.retrieve_context(request, _budget())

    dense_outcomes = [outcome for outcome in context.method_outcomes if outcome.method is RetrievalMethod.DENSE]
    assert dense.calls == 3
    assert dense_outcomes[0].infrastructure_attempts == 3
    assert dense_outcomes[1].reason_code == "method_circuit_open"


def test_transient_source_availability_failure_is_retried_before_marking_context_blocked():
    source_filter = SourceFilter.from_mapping(SourceType.CURRENT_SESSION, {"session_id": "s-a"})
    source = TimeoutThenReadySource(
        source_type=SourceType.CURRENT_SESSION,
        units={},
        failures_before_ready=2,
    )

    context = _service(
        sources={SourceType.CURRENT_SESSION: source},
        methods={
            RetrievalMethod.DENSE: FakeMethod(RetrievalMethod.DENSE),
            RetrievalMethod.LEARNED_SPARSE: FakeMethod(RetrievalMethod.LEARNED_SPARSE),
        },
    ).retrieve_context(_request(source_filters={SourceType.CURRENT_SESSION: source_filter}), _budget())

    assert source.availability_calls == 3
    assert context.status is ContextStatus.COMPLETE


def test_a_result_names_the_encoder_that_produced_it():
    """与编码器脱离的召回结果无法在模型或索引版本变化后诊断，而过期召回
    恰恰会在此时出现。"""
    ref = _ref("turn-8")
    source_filter = SourceFilter.from_mapping(SourceType.CURRENT_SESSION, {"session_id": "s-a"})
    source = FakeSource(SourceType.CURRENT_SESSION, {ref: SourceUnit(ref, "北桥项目下周交付")})
    dense = FakeMethod(
        RetrievalMethod.DENSE,
        {(SourceType.CURRENT_SESSION, "北桥项目"): (_candidate(ref, RetrievalMethod.DENSE),)},
    )

    service = _service(
        sources={SourceType.CURRENT_SESSION: source},
        methods={RetrievalMethod.DENSE: dense},
        encoder_fingerprint="bge_m3:model=BAAI/bge-m3;dense=1024",
    )
    context = service.retrieve_context(
        _request(source_filters={SourceType.CURRENT_SESSION: source_filter}), _budget()
    )

    assert context.encoder_fingerprint == "bge_m3:model=BAAI/bge-m3;dense=1024"
    # 空值表示“调用方未提供”，绝不表示“不存在编码器”。
    assert isinstance(context.retrieval_data_version, str)
