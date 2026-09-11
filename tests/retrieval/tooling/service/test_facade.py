from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import json

from personagraph.retrieval.contracts import (
    ContextStatus,
    CorpusKey,
    RerankerOutcome,
    RerankerRunStatus,
    RetrievedContext,
    RetrievedItem,
    RetrievalRequest,
    SourceAvailability,
    SourceDependency,
    SourceOutcome,
    SourceRetrievalStatus,
    SourceType,
    SourceUnitRef,
)
from personagraph.retrieval.tooling.contracts import (
    FileRetrievalReadinessResult,
    FileRetrievalReadinessStatus,
)
from personagraph.retrieval.sources.identity import current_session_source_unit_id
from personagraph.retrieval.query_guard import (
    QueryGuardError,
    QueryGuardErrorCode,
)
from personagraph.retrieval.tooling.contracts import (
    FileRetrievalOrigin,
    FileRetrievalScope,
    FrozenFileRetrievalBinding,
    FrozenFileVersionBinding,
    FrozenHistoryRetrievalScope,
    HistoryScope,
    RetrievalCorpus,
    RetrievalStatus,
    RetrievalToolRequest,
)
from personagraph.retrieval.tooling.service import RetrievalServiceToolPort
from personagraph.retrieval.tooling.service.projection import (
    _BoundRetrievedItem,
    _pack_file_candidates,
)
from personagraph.tools.retrieval.history_retrieval_adapter import (
    build_history_retrieval_tool_registration,
)


GENERATION = "retrieval-generation-1"
SESSION_ID = "private-session-1"


def _hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


@dataclass
class _VersionProvider:
    values: tuple[str | None, ...] = (GENERATION,)
    calls: int = 0

    def active_retrieval_data_version_id(self) -> str | None:
        value = self.values[min(self.calls, len(self.values) - 1)]
        self.calls += 1
        return value


@dataclass
class _ContextService:
    context: RetrievedContext
    requests: list[RetrievalRequest] = field(default_factory=list)
    readonly_requests: list[RetrievalRequest] = field(default_factory=list)

    def retrieve_file_query_batch(self, requests, budget, *, result_limit, readonly):
        method = self.retrieve_context_readonly if readonly else self.retrieve_context
        return tuple(method(request, budget) for request in requests)

    def retrieve_context(
        self,
        request: RetrievalRequest,
        _budget: object,
    ) -> RetrievedContext:
        self.requests.append(request)
        return self.context

    def retrieve_context_unrecorded(
        self,
        request: RetrievalRequest,
        _budget: object,
    ) -> RetrievedContext:
        self.requests.append(request)
        return self.context

    def retrieve_context_readonly(
        self,
        request: RetrievalRequest,
        _budget: object,
    ) -> RetrievedContext:
        self.readonly_requests.append(request)
        return self.context


@dataclass
class _RoutingContextService:
    by_source: dict[SourceType, RetrievedContext]
    readonly_requests: list[RetrievalRequest] = field(default_factory=list)

    def retrieve_file_query_batch(self, requests, budget, *, result_limit, readonly):
        assert readonly
        return tuple(self.retrieve_context_readonly(request, budget) for request in requests)

    def retrieve_context(
        self,
        request: RetrievalRequest,
        budget: object,
    ) -> RetrievedContext:
        return self.retrieve_context_readonly(request, budget)

    def retrieve_context_readonly(
        self,
        request: RetrievalRequest,
        _budget: object,
    ) -> RetrievedContext:
        self.readonly_requests.append(request)
        source_types = tuple(request.boundary.source_filters)
        if len(source_types) != 1:
            raise AssertionError("session retrieval must issue one exact source call")
        return self.by_source[source_types[0]]


@dataclass(frozen=True)
class _Foundation:
    corpus_key: CorpusKey
    service: _ContextService
    data_version_provider: _VersionProvider


@dataclass
class _Readiness:
    document_id: str
    document_version_id: str
    seen_source_ids: list[str] = field(default_factory=list)

    def ensure_ready(
        self,
        request: RetrievalToolRequest,
        binding: FrozenFileRetrievalBinding,
    ) -> FileRetrievalReadinessResult:
        self.seen_source_ids.append(binding.source_id)
        return FileRetrievalReadinessResult(
            status=FileRetrievalReadinessStatus.READY,
            file_id=binding.source_id,
            retrieval_data_version=request.retrieval_data_version,
            document_id=self.document_id,
            document_version_id=self.document_version_id,
        )


@dataclass
class _BlockedReadiness:
    def ensure_ready(
        self,
        request: RetrievalToolRequest,
        binding: FrozenFileRetrievalBinding,
    ) -> FileRetrievalReadinessResult:
        return FileRetrievalReadinessResult(
            status=FileRetrievalReadinessStatus.BLOCKED,
            file_id=binding.source_id,
            reason_code="document_reader_unsupported",
        )


def _item(
    *,
    source_type: SourceType,
    source_unit_id: str,
    content: str,
    citation: dict[str, str],
    revision: str = "source-revision-1",
    query_index: int = 0,
    fused_rank: int = 1,
    reranker_score: float | None = None,
) -> RetrievedItem:
    return RetrievedItem(
        ref=SourceUnitRef(
            source_type=source_type,
            source_unit_id=source_unit_id,
            source_revision=revision,
            indexed_content_hash=_hash(content),
        ),
        content=content,
        citation=citation,
        estimated_tokens=8,
        fused_rank=fused_rank,
        query_index=query_index,
        reranker_score=reranker_score,
    )


def test_file_packing_applies_query_top_k_before_text_budget_without_source_reservations() -> None:
    request = RetrievalToolRequest(
        request_id="file-pack-fairness",
        corpus=RetrievalCorpus.FILES,
        query="mixed evidence",
        limit=2,
        context_token_limit=10,
        session_id=SESSION_ID,
        scope_snapshot_id="file-scope-1",
        retrieval_data_version=GENERATION,
        file_scope=FileRetrievalScope.SESSION_CORPUS,
    )

    def candidate(source_type: SourceType, suffix: str, tokens: int, rank: int):
        item = _item(
            source_type=source_type,
            source_unit_id=f"{source_type.value}-{suffix}",
            content=f"{source_type.value} {suffix}",
            citation=(
                {"doc_id": "doc-a", "source_version_id": "source-revision-1"}
                if source_type is SourceType.DOCUMENT
                else {
                    "file_id": "file-a",
                    "file_version_id": "file-version-a",
                    "picture_id": "picture-a",
                    "picture_unit_id": f"unit-{suffix}",
                    "observation_id": f"observation-{suffix}",
                }
            ),
        )
        object.__setattr__(item, "estimated_tokens", tokens)
        object.__setattr__(item, "fused_rank", rank)
        return _BoundRetrievedItem(item=item, authority_id=None, binding_index=0)

    evidence, _gaps, _truncated = _pack_file_candidates(
        request,
        (
            candidate(SourceType.DOCUMENT, "rank-1", 6, 1),
            candidate(SourceType.DOCUMENT, "rank-2", 4, 2),
            candidate(SourceType.PICTURE, "rank-1", 6, 1),
            candidate(SourceType.PICTURE, "rank-2", 4, 2),
        ),
    )

    assert {item.source_type for item in evidence} == {SourceType.DOCUMENT}
    assert sum(item.estimated_tokens for item in evidence) == 6
    assert _truncated
    assert any(gap.code == "port_token_limit_exceeded" for gap in _gaps)


def test_file_packing_globally_ranks_same_query_scores_across_bindings() -> None:
    request = RetrievalToolRequest(
        request_id="file-pack-global-rerank",
        corpus=RetrievalCorpus.FILES,
        query="target evidence",
        limit=2,
        context_token_limit=100,
        session_id=SESSION_ID,
        scope_snapshot_id="file-scope-1",
        retrieval_data_version=GENERATION,
        file_scope=FileRetrievalScope.SESSION_CORPUS,
    )
    low = _item(
        source_type=SourceType.DOCUMENT,
        source_unit_id="document-low",
        content="low",
        citation={"doc_id": "doc-low", "source_version_id": "source-revision-1"},
        fused_rank=1,
        reranker_score=0.1,
    )
    high = _item(
        source_type=SourceType.DOCUMENT,
        source_unit_id="document-high",
        content="high",
        citation={"doc_id": "doc-high", "source_version_id": "source-revision-1"},
        fused_rank=1,
        reranker_score=0.9,
    )

    evidence, _gaps, _truncated = _pack_file_candidates(
        request,
        (
            _BoundRetrievedItem(low, None, 0),
            _BoundRetrievedItem(high, None, 1),
        ),
    )

    assert [item.content for item in evidence] == ["high", "low"]


def test_file_packing_orders_merged_candidates_by_best_query_score() -> None:
    request = RetrievalToolRequest(
        request_id="file-pack-global-query-lanes",
        corpus=RetrievalCorpus.FILES,
        query="primary",
        query_variants=("source language",),
        limit=4,
        context_token_limit=100,
        session_id=SESSION_ID,
        scope_snapshot_id="file-scope-1",
        retrieval_data_version=GENERATION,
        file_scope=FileRetrievalScope.SESSION_CORPUS,
    )

    def bound(name: str, query_index: int, score: float, binding_index: int):
        return _BoundRetrievedItem(
            _item(
                source_type=SourceType.DOCUMENT,
                source_unit_id=name,
                content=name,
                citation={
                    "doc_id": f"doc-{name}",
                    "source_version_id": "source-revision-1",
                },
                query_index=query_index,
                reranker_score=score,
            ),
            None,
            binding_index,
        )

    evidence, _gaps, _truncated = _pack_file_candidates(
        request,
        (
            bound("q0-low", 0, 0.1, 0),
            bound("q1-low", 1, 0.2, 0),
            bound("q0-high", 0, 0.9, 1),
            bound("q1-high", 1, 0.8, 1),
        ),
    )

    assert [item.content for item in evidence] == [
        "q0-high",
        "q1-high",
        "q1-low",
        "q0-low",
    ]
    assert [item.query_index for item in evidence] == [0, 1, 1, 0]


def test_file_packing_falls_back_as_a_whole_when_any_reranker_score_is_missing() -> None:
    request = RetrievalToolRequest(
        request_id="file-pack-reranker-fallback",
        corpus=RetrievalCorpus.FILES,
        query="target evidence",
        limit=2,
        context_token_limit=100,
        session_id=SESSION_ID,
        scope_snapshot_id="file-scope-1",
        retrieval_data_version=GENERATION,
        file_scope=FileRetrievalScope.SESSION_CORPUS,
    )
    rrf_first = _item(
        source_type=SourceType.DOCUMENT,
        source_unit_id="rrf-first",
        content="rrf-first",
        citation={"doc_id": "doc-first", "source_version_id": "source-revision-1"},
        fused_rank=1,
    )
    partially_scored = _item(
        source_type=SourceType.DOCUMENT,
        source_unit_id="partially-scored",
        content="partially-scored",
        citation={"doc_id": "doc-second", "source_version_id": "source-revision-1"},
        fused_rank=2,
        reranker_score=0.99,
    )

    evidence, _gaps, _truncated = _pack_file_candidates(
        request,
        (
            _BoundRetrievedItem(rrf_first, None, 0),
            _BoundRetrievedItem(partially_scored, None, 0),
        ),
    )

    assert [item.content for item in evidence] == ["rrf-first", "partially-scored"]


def _context(
    item: RetrievedItem,
    *,
    generation: str = GENERATION,
) -> RetrievedContext:
    outcomes = {
        source_type: SourceOutcome(
            source_type=source_type,
            availability=(
                SourceAvailability.READY
                if source_type is item.ref.source_type
                else SourceAvailability.DISABLED
            ),
            retrieval=(
                SourceRetrievalStatus.MATCHED
                if source_type is item.ref.source_type
                else SourceRetrievalStatus.NOT_RUN
            ),
            dependency=SourceDependency.OPTIONAL,
        )
        for source_type in SourceType
    }
    return RetrievedContext(
        status=ContextStatus.COMPLETE,
        items=(item,),
        source_outcomes=outcomes,
        method_outcomes=(),
        configured_token_limit=500,
        packed_tokens=item.estimated_tokens,
        retrieval_data_version=generation,
    )


def _current_session_item() -> RetrievedItem:
    return _item(
        source_type=SourceType.CURRENT_SESSION,
        source_unit_id=current_session_source_unit_id(
            session_id=SESSION_ID,
            run_id="private-run-1",
            role="pair",
            ordinal=0,
        ),
        content="User prefers concise answers.",
        citation={
            "session_id": SESSION_ID,
            "user_turn_idx": "2",
            "assistant_turn_idx": "3",
            "path": "/private/history.sqlite",
        },
    )


def _history_request() -> RetrievalToolRequest:
    return RetrievalToolRequest(
        request_id="history-request-1",
        corpus=RetrievalCorpus.HISTORY,
        query="answer style",
        limit=4,
        context_token_limit=500,
        session_id=SESSION_ID,
        scope_snapshot_id="history-scope-1",
        retrieval_data_version=GENERATION,
        history_scopes=(HistoryScope.CURRENT_SESSION,),
        current_session_cutoff="assistant_turn_idx:3",
    )


def test_history_generation_drift_during_read_withholds_evidence() -> None:
    provider = _VersionProvider((GENERATION, "retrieval-generation-2"))
    service = _ContextService(_context(_current_session_item()))
    port = RetrievalServiceToolPort(
        history_foundation=_Foundation(CorpusKey.HISTORY, service, provider)
    )

    result, _deferred_audit = port.retrieve_history_unrecorded(_history_request())

    assert provider.calls == 2
    assert result.status is RetrievalStatus.BLOCKED
    assert result.evidence == ()
    assert result.retrieval_data_version == "retrieval-generation-2"
    assert [gap.code for gap in result.gaps] == ["retrieval_generation_stale"]


def test_file_query_is_exactly_scoped_and_rejects_another_document() -> None:
    selected_document_id = "private-document-a"
    revision = "document-version-1"
    wrong_document_item = _item(
        source_type=SourceType.DOCUMENT,
        source_unit_id="document-v3:private-document-b:chunk-1",
        content="Content from another document.",
        citation={
            "doc_id": "private-document-b",
            "source_version_id": revision,
        },
        revision=revision,
    )
    provider = _VersionProvider()
    service = _ContextService(_context(wrong_document_item))
    readiness = _Readiness(selected_document_id, revision)
    port = RetrievalServiceToolPort(
        file_foundation=_Foundation(CorpusKey.FILE, service, provider),
        file_readiness=readiness,
    )
    request = RetrievalToolRequest(
        request_id="file-request-1",
        corpus=RetrievalCorpus.FILES,
        query="target paragraph",
        limit=4,
        context_token_limit=500,
        session_id=SESSION_ID,
        scope_snapshot_id="file-scope-1",
        retrieval_data_version=GENERATION,
        file_scope=FileRetrievalScope.SELECTED_FILES,
        file_bindings=(
            FrozenFileRetrievalBinding(
                source_id="file_01",
                authority_id="private-authority-a",
                origin=FileRetrievalOrigin.WORKSPACE,
                expected_source_revision=revision,
            ),
        ),
    )

    result = port.retrieve(request)

    assert readiness.seen_source_ids == ["file_01"]
    assert service.requests[0].boundary.source_filters[
        SourceType.DOCUMENT
    ].as_mapping() == {
        "doc_id": selected_document_id,
        "session_id": SESSION_ID,
    }
    assert result.status is RetrievalStatus.BLOCKED
    assert result.evidence == ()
    assert [gap.code for gap in result.gaps] == ["retrieval_authority_mismatch"]

    service.context = _context(
        _item(
            source_type=SourceType.DOCUMENT,
            source_unit_id=f"document-v3:{selected_document_id}:chunk-1",
            content="Content from the selected document.",
            citation={
                "doc_id": selected_document_id,
                "source_version_id": revision,
            },
            revision=revision,
        )
    )

    authorized_result = port.retrieve(request)

    assert authorized_result.status is RetrievalStatus.COMPLETE
    assert authorized_result.evidence[0].authority_id == "private-authority-a"


def test_selected_file_retrieval_jointly_returns_document_and_picture() -> None:
    document_id = "private-document-a"
    document_version_id = "document-version-1"
    document = _item(
        source_type=SourceType.DOCUMENT,
        source_unit_id="document-v3:private-document-a:chunk-1",
        content="Document evidence.",
        citation={
            "doc_id": document_id,
            "source_version_id": document_version_id,
            "origin": "user_upload",
        },
        revision=document_version_id,
    )
    picture = _item(
        source_type=SourceType.PICTURE,
        source_unit_id="picture-observation-a",
        content="Picture evidence.",
        citation={
            "file_id": "file-a",
            "file_version_id": "file-version-a",
            "picture_id": "picture-a",
            "picture_unit_id": "picture-unit-a",
            "observation_id": "observation-a",
            "structured_payload": "must-not-survive",
        },
        revision="observation-revision-a",
    )
    outcomes = {
        source_type: SourceOutcome(
            source_type=source_type,
            availability=(
                SourceAvailability.READY
                if source_type in {SourceType.DOCUMENT, SourceType.PICTURE}
                else SourceAvailability.DISABLED
            ),
            retrieval=(
                SourceRetrievalStatus.MATCHED
                if source_type in {SourceType.DOCUMENT, SourceType.PICTURE}
                else SourceRetrievalStatus.NOT_RUN
            ),
            dependency=SourceDependency.OPTIONAL,
        )
        for source_type in SourceType
    }
    context = RetrievedContext(
        status=ContextStatus.COMPLETE,
        items=(document, picture),
        source_outcomes=outcomes,
        method_outcomes=(),
        configured_token_limit=500,
        packed_tokens=16,
        retrieval_data_version=GENERATION,
    )
    service = _ContextService(context)
    port = RetrievalServiceToolPort(
        file_foundation=_Foundation(CorpusKey.FILE, service, _VersionProvider()),
        file_readiness=_Readiness(document_id, document_version_id),
    )
    request = RetrievalToolRequest(
        request_id="file-request-mixed",
        corpus=RetrievalCorpus.FILES,
        query="mixed evidence",
        limit=4,
        context_token_limit=500,
        session_id=SESSION_ID,
        scope_snapshot_id="file-scope-1",
        retrieval_data_version=GENERATION,
        file_scope=FileRetrievalScope.SELECTED_FILES,
        file_bindings=(
            FrozenFileRetrievalBinding(
                source_id="file_01",
                authority_id="private-authority-a",
                origin=FileRetrievalOrigin.USER_UPLOAD,
                project_id="project-a",
                file_id="file-a",
                file_version_id="file-version-a",
            ),
        ),
    )

    result = port.retrieve(request)

    assert result.status is RetrievalStatus.COMPLETE
    assert {item.source_type for item in result.evidence} == {
        SourceType.DOCUMENT,
        SourceType.PICTURE,
    }
    assert set(service.requests[0].boundary.source_filters) == {
        SourceType.DOCUMENT,
        SourceType.PICTURE,
    }
    picture_evidence = next(
        item for item in result.evidence if item.source_type is SourceType.PICTURE
    )
    assert "structured_payload" not in picture_evidence.citation


def test_picture_only_file_returns_picture_without_fake_document() -> None:
    picture = _item(
        source_type=SourceType.PICTURE,
        source_unit_id="picture-observation-a",
        content="Picture-only evidence.",
        citation={
            "file_id": "file-a",
            "file_version_id": "file-version-a",
            "picture_id": "picture-a",
            "picture_unit_id": "picture-unit-a",
            "observation_id": "observation-a",
        },
        revision="observation-revision-a",
    )
    service = _ContextService(_context(picture))
    port = RetrievalServiceToolPort(
        file_foundation=_Foundation(CorpusKey.FILE, service, _VersionProvider()),
        file_readiness=_BlockedReadiness(),
    )
    request = RetrievalToolRequest(
        request_id="file-request-picture-only",
        corpus=RetrievalCorpus.FILES,
        query="visual evidence",
        limit=4,
        context_token_limit=500,
        session_id=SESSION_ID,
        scope_snapshot_id="file-scope-1",
        retrieval_data_version=GENERATION,
        file_scope=FileRetrievalScope.SELECTED_FILES,
        file_bindings=(
            FrozenFileRetrievalBinding(
                source_id="file_01",
                authority_id="private-authority-a",
                origin=FileRetrievalOrigin.USER_UPLOAD,
                project_id="project-a",
                file_id="file-a",
                file_version_id="file-version-a",
            ),
        ),
    )

    result = port.retrieve(request)

    assert [item.source_type for item in result.evidence] == [SourceType.PICTURE]
    assert result.status is RetrievalStatus.PARTIAL
    assert any(gap.source_type is SourceType.DOCUMENT for gap in result.gaps)
    assert set(service.requests[0].boundary.source_filters) == {SourceType.PICTURE}


def test_session_file_retrieval_uses_one_exact_picture_scope_per_file() -> None:
    document = _item(
        source_type=SourceType.DOCUMENT,
        source_unit_id="document-v3:private-document-a:chunk-1",
        content="Document evidence.",
        citation={
            "doc_id": "private-document-a",
            "source_version_id": "document-version-1",
            "origin": "user_upload",
        },
        revision="document-version-1",
    )
    picture = _item(
        source_type=SourceType.PICTURE,
        source_unit_id="picture-observation-a",
        content="Picture evidence.",
        citation={
            "file_id": "file-a",
            "file_version_id": "file-version-a",
            "picture_id": "picture-a",
            "picture_unit_id": "picture-unit-a",
            "observation_id": "observation-a",
        },
        revision="observation-revision-a",
    )
    service = _RoutingContextService(
        {
            SourceType.DOCUMENT: _context(document),
            SourceType.PICTURE: _context(picture),
        }
    )
    port = RetrievalServiceToolPort(
        file_foundation=_Foundation(CorpusKey.FILE, service, _VersionProvider()),
    )
    request = RetrievalToolRequest(
        request_id="file-session-mixed",
        corpus=RetrievalCorpus.FILES,
        query="mixed evidence",
        limit=4,
        context_token_limit=500,
        session_id=SESSION_ID,
        scope_snapshot_id="file-scope-1",
        retrieval_data_version=GENERATION,
        file_scope=FileRetrievalScope.SESSION_CORPUS,
        session_file_bindings=(
            FrozenFileVersionBinding(
                project_id="project-a",
                file_id="file-a",
                file_version_id="file-version-a",
            ),
        ),
    )

    result = port.retrieve(request)

    assert result.status is RetrievalStatus.COMPLETE
    assert {item.source_type for item in result.evidence} == {
        SourceType.DOCUMENT,
        SourceType.PICTURE,
    }
    picture_request = service.readonly_requests[1]
    assert set(picture_request.boundary.source_filters) == {SourceType.PICTURE}
    assert picture_request.boundary.source_filters[SourceType.PICTURE].as_mapping() == {
        "file_id": "file-a",
        "file_version_id": "file-version-a",
        "session_id": SESSION_ID,
    }

    unavailable = _context(picture)
    unavailable_outcomes = dict(unavailable.source_outcomes)
    unavailable_outcomes[SourceType.PICTURE] = replace(
        unavailable_outcomes[SourceType.PICTURE], availability=SourceAvailability.UNAVAILABLE,
        retrieval=SourceRetrievalStatus.NOT_RUN, reason_code="retrieval_backend_unavailable",
    )
    service.by_source[SourceType.PICTURE] = replace(
        unavailable, items=(), packed_tokens=0, status=ContextStatus.PARTIAL,
        source_outcomes=unavailable_outcomes,
    )
    degraded = port.retrieve(request)

    assert degraded.status is RetrievalStatus.PARTIAL
    assert {item.source_type for item in degraded.evidence} == {SourceType.DOCUMENT}
    assert any(
        gap.code == "retrieval_backend_unavailable"
        and gap.source_type is SourceType.PICTURE
        for gap in degraded.gaps
    )

def test_session_document_corpus_query_is_readonly_and_needs_no_readiness() -> None:
    document_id = "private-document-a"
    revision = "document-version-1"
    item = _item(
        source_type=SourceType.DOCUMENT,
        source_unit_id=f"document-v3:{document_id}:chunk-1",
        content="Content from the authorized Session corpus.",
        citation={
            "doc_id": document_id,
            "source_version_id": revision,
            "location": "paragraph:1",
            "origin": "workspace",
        },
        revision=revision,
    )
    provider = _VersionProvider()
    service = _ContextService(_context(item))
    port = RetrievalServiceToolPort(
        file_foundation=_Foundation(CorpusKey.FILE, service, provider),
    )
    request = RetrievalToolRequest(
        request_id="file-session-corpus-request-1",
        corpus=RetrievalCorpus.FILES,
        query="target paragraph",
        limit=4,
        context_token_limit=500,
        session_id=SESSION_ID,
        scope_snapshot_id="file-session-corpus-scope-1",
        retrieval_data_version=GENERATION,
        file_scope=FileRetrievalScope.SESSION_CORPUS,
    )

    result = port.retrieve(request)

    assert service.requests == []
    assert len(service.readonly_requests) == 1
    assert service.readonly_requests[0].boundary.source_filters[
        SourceType.DOCUMENT
    ].as_mapping() == {
        "document_evidence_scope": "user_evidence",
        "session_id": SESSION_ID,
    }
    assert result.status is RetrievalStatus.COMPLETE
    assert len(result.evidence) == 1
    assert result.evidence[0].authority_id is None
    assert result.evidence[0].citation["doc_id"] == document_id
    assert result.evidence[0].citation["source_version_id"] == revision


def test_session_document_corpus_rejects_a_mismatched_document_revision() -> None:
    item = _item(
        source_type=SourceType.DOCUMENT,
        source_unit_id="document-v3:private-document-a:chunk-1",
        content="Content with an invalid citation revision.",
        citation={
            "doc_id": "private-document-a",
            "source_version_id": "another-document-version",
            "origin": "workspace",
        },
        revision="document-version-1",
    )
    service = _ContextService(_context(item))
    port = RetrievalServiceToolPort(
        file_foundation=_Foundation(
            CorpusKey.FILE,
            service,
            _VersionProvider(),
        ),
    )
    request = RetrievalToolRequest(
        request_id="file-session-corpus-request-invalid-revision",
        corpus=RetrievalCorpus.FILES,
        query="target paragraph",
        limit=4,
        context_token_limit=500,
        session_id=SESSION_ID,
        scope_snapshot_id="file-session-corpus-scope-1",
        retrieval_data_version=GENERATION,
        file_scope=FileRetrievalScope.SESSION_CORPUS,
    )

    result = port.retrieve(request)

    assert len(service.readonly_requests) == 1
    assert result.status is RetrievalStatus.BLOCKED
    assert result.evidence == ()
    assert [gap.code for gap in result.gaps] == ["retrieval_authority_mismatch"]


def test_session_document_corpus_rejects_agent_output_evidence() -> None:
    revision = "document-version-1"
    item = _item(
        source_type=SourceType.DOCUMENT,
        source_unit_id="document-v3:private-document-a:chunk-1",
        content="Content generated by the agent.",
        citation={
            "doc_id": "private-document-a",
            "source_version_id": revision,
            "origin": "agent_output",
        },
        revision=revision,
    )
    service = _ContextService(_context(item))
    port = RetrievalServiceToolPort(
        file_foundation=_Foundation(
            CorpusKey.FILE,
            service,
            _VersionProvider(),
        ),
    )
    request = RetrievalToolRequest(
        request_id="file-session-corpus-agent-output",
        corpus=RetrievalCorpus.FILES,
        query="target paragraph",
        limit=4,
        context_token_limit=500,
        session_id=SESSION_ID,
        scope_snapshot_id="file-session-corpus-scope-1",
        retrieval_data_version=GENERATION,
        file_scope=FileRetrievalScope.SESSION_CORPUS,
    )

    result = port.retrieve(request)

    assert result.status is RetrievalStatus.BLOCKED
    assert result.evidence == ()
    assert [gap.code for gap in result.gaps] == ["retrieval_authority_mismatch"]


def test_history_deferred_audit_records_the_final_public_projection(
    monkeypatch,
) -> None:
    item = _current_session_item()
    provider = _VersionProvider()
    service = _ContextService(_context(item))
    captured: list[dict[str, object]] = []
    monkeypatch.setattr(
        "personagraph.trajectory.record_retrieval",
        lambda **kwargs: captured.append(kwargs),
    )
    scope = FrozenHistoryRetrievalScope(
        session_id=SESSION_ID,
        scope_snapshot_id="history-scope-1",
        retrieval_data_version=GENERATION,
        allowed_scopes=(HistoryScope.CURRENT_SESSION,),
        current_session_cutoff="assistant_turn_idx:3",
        max_items=4,
        context_token_limit=500,
    )
    registration = build_history_retrieval_tool_registration(
        scope,
        port=RetrievalServiceToolPort(
            history_foundation=_Foundation(CorpusKey.HISTORY, service, provider),
            trajectory_turn_id="turn-history-1",
        ),
    )

    public_result = registration.handler({"query": "answer style", "limit": 4})

    assert len(captured) == 1
    assert captured[0]["outcome"] == public_result["status"] == "complete"
    assert captured[0]["turn_id"] == "turn-history-1"
    assert len(captured[0]["evidence"]) == 1
    recorded_evidence = captured[0]["evidence"][0]
    assert recorded_evidence["handle"] == public_result["evidence"][0]["handle"]
    assert recorded_evidence["content_sha256"] == public_result["evidence"][0][
        "content_sha256"
    ]
    assert recorded_evidence["projection_status"] == "injected"
    assert recorded_evidence["packed_rank"] == 1
    encoded_audit = json.dumps(captured, ensure_ascii=False)
    assert item.content not in encoded_audit
    assert "/private/history.sqlite" not in encoded_audit


def test_file_reranker_failure_is_fail_soft_and_recorded_in_trajectory(
    monkeypatch,
) -> None:
    revision = "document-version-1"
    item = _item(
        source_type=SourceType.DOCUMENT,
        source_unit_id="document-v3:private-document-a:chunk-1",
        content="Fallback evidence remains available.",
        citation={
            "doc_id": "private-document-a",
            "source_version_id": revision,
            "origin": "workspace",
        },
        revision=revision,
    )
    context = replace(
        _context(item),
        reranker_outcomes=(
            RerankerOutcome(
                source_type=SourceType.DOCUMENT,
                status=RerankerRunStatus.DEGRADED,
                candidate_count=1,
                reason_code="bge_reranker_inference_failed:RuntimeError",
            ),
        ),
        reranker_fingerprint="bge-reranker-v2-m3:test",
    )
    captured: list[dict[str, object]] = []
    monkeypatch.setattr(
        "personagraph.trajectory.record_retrieval",
        lambda **kwargs: captured.append(kwargs),
    )
    request = RetrievalToolRequest(
        request_id="file-reranker-fallback-audit",
        corpus=RetrievalCorpus.FILES,
        query="fallback evidence",
        limit=4,
        context_token_limit=500,
        session_id=SESSION_ID,
        scope_snapshot_id="file-scope-1",
        retrieval_data_version=GENERATION,
        file_scope=FileRetrievalScope.SELECTED_FILES,
        file_bindings=(
            FrozenFileRetrievalBinding(
                source_id="file_01",
                authority_id="private-authority-a",
                origin=FileRetrievalOrigin.WORKSPACE,
            ),
        ),
    )
    port = RetrievalServiceToolPort(
        file_foundation=_Foundation(
            CorpusKey.FILE,
            _ContextService(context),
            _VersionProvider(),
        ),
        file_readiness=_Readiness("private-document-a", revision),
        file_readonly=True,
        trajectory_turn_id="turn-file-reranker-fallback",
    )

    result, deferred_audit = port.retrieve_file_readonly_unrecorded(request)
    deferred_audit.record()

    assert result.status is RetrievalStatus.COMPLETE
    assert len(result.evidence) == 1
    assert len(captured) == 1
    assert captured[0]["reranker_outcomes"][0]["status"] == "degraded"
    assert captured[0]["reranker_outcomes"][0]["reason_code"] == (
        "bge_reranker_inference_failed:RuntimeError"
    )
    diagnostic_codes = {
        item["code"] for item in captured[0]["diagnostics"]
    }
    assert "bge_reranker_inference_failed:RuntimeError" in diagnostic_codes
    assert "query_rerank_unavailable_rrf_fallback" in diagnostic_codes


def test_file_query_guard_reason_survives_the_tool_service_boundary() -> None:
    class RejectingService:
        def retrieve_file_query_batch(self, *_args, **_kwargs):
            return self.retrieve_context_readonly()

        def retrieve_context(self, *_args, **_kwargs):
            raise AssertionError("readonly File retrieval must use the readonly path")

        def retrieve_context_readonly(self, *_args, **_kwargs):
            raise QueryGuardError(
                "query proposal exceeds retrieval-token limit",
                code=QueryGuardErrorCode.QUERY_TOO_MANY_TOKENS,
            )

    request = RetrievalToolRequest(
        request_id="file-query-guard-code",
        corpus=RetrievalCorpus.FILES,
        query="oversized query",
        limit=4,
        context_token_limit=500,
        session_id=SESSION_ID,
        scope_snapshot_id="file-scope-1",
        retrieval_data_version=GENERATION,
        file_scope=FileRetrievalScope.SESSION_CORPUS,
    )
    port = RetrievalServiceToolPort(
        file_foundation=_Foundation(
            CorpusKey.FILE,
            RejectingService(),
            _VersionProvider(),
        ),
        file_readonly=True,
    )

    result, _deferred_audit = port.retrieve_file_readonly_unrecorded(request)

    assert result.status is RetrievalStatus.BLOCKED
    assert [gap.code for gap in result.gaps] == ["query_too_many_tokens"]
