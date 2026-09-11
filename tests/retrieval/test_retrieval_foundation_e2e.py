from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from personagraph.retrieval.contracts import (
    QueryProposal,
    RetrievalBudget,
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
from personagraph.retrieval.foundation import build_file_retrieval_foundation
from personagraph.retrieval.lifecycle.generation import (
    DOCUMENT_HYBRID_INDEX_RECIPE_CONTRACT,
    file_corpus_generation_spec,
)
from personagraph.retrieval.indexing.encoder import DeterministicLexicalEncoder
from personagraph.retrieval.indexing.methods import (
    BM25Retrieval,
    BgeM3EncodedText,
    DenseRetrieval,
    LearnedSparseRetrieval,
    LiteralBooleanFallback,
    SqliteRetrievalIndexWriter,
    SqliteRetrievalMethodStore,
)
from personagraph.retrieval.lifecycle.outbox import RetrievalUpdateEvent, RetrievalUpdateKind
from personagraph.retrieval.policy import DefaultRetrievalPolicy
from personagraph.retrieval.query_guard import QueryGuard
from personagraph.retrieval.orchestration.selection import SourceQuotaPool, SourceSelectionQuotas
from personagraph.retrieval.service import RetrievalService
from personagraph.retrieval.sqlite_store import (
    RetrievalDataVersionRole,
    RetrievalDataVersionState,
    SqliteRetrievalCatalog,
)
from personagraph.retrieval.lifecycle.sync import IndexableSourceUnit, RetrievalSyncService


def _vector(value: float) -> tuple[float, ...]:
    return (value,) + (0.0,) * 1023


@dataclass
class DeterministicEncoder:
    values: dict[str, BgeM3EncodedText]

    def encode(self, texts):
        return tuple(self.values[text] for text in texts)

    def token_ids(self, text: str):
        return self.values[text].token_ids

    def fingerprint(self) -> str:
        return "test-bge-m3"


@dataclass
class Reader:
    source_type: SourceType
    unit: IndexableSourceUnit

    def read_for_index(self, event):
        return self.unit if event.ref == self.unit.ref else None


@dataclass
class VerifiedSource:
    source_type: SourceType
    unit: SourceUnit

    def open_retrieval_access(self, source_filter):
        return SourceAccess(self.source_type, source_filter, SourceAvailability.READY)

    def fetch_units(self, access: SourceAccess, refs: Sequence[SourceUnitRef]):
        if access.source_filter.as_mapping() != {"session_id": "s1"}:
            return ()
        return [self.unit] if self.unit.ref in refs else []


def test_real_derived_indexes_sync_then_retrieve_and_revalidate_without_copying_source_text(tmp_path):
    content = "用户：北桥进展？\n\n助手：本周交付初稿。"
    query = "北桥交付"
    source_filter = SourceFilter.from_mapping(SourceType.CURRENT_SESSION, {"session_id": "s1"})
    ref = SourceUnitRef(SourceType.CURRENT_SESSION, "s1:run-1", "r1", "hash-1")
    encoder = DeterministicEncoder(
        {
            content: BgeM3EncodedText(_vector(1.0), {10: 0.8, 11: 0.6}, (10, 11, 11)),
            query: BgeM3EncodedText(_vector(1.0), {10: 1.0, 11: 0.9}, (10, 11)),
        }
    )
    catalog = SqliteRetrievalCatalog(tmp_path / "retrieval.sqlite")
    catalog.initialize()
    catalog.create_data_version(
        version_id="v1",
        fingerprint=encoder.fingerprint(),
        role=RetrievalDataVersionRole.ACTIVE,
        state=RetrievalDataVersionState.READY,
    )
    methods_store = SqliteRetrievalMethodStore(catalog=catalog, encoder=encoder)
    indexable = IndexableSourceUnit(ref=ref, source_filter=source_filter, content=content)
    sync = RetrievalSyncService(
        catalog=catalog,
        source_readers={SourceType.CURRENT_SESSION: Reader(SourceType.CURRENT_SESSION, indexable)},
        index_writer=SqliteRetrievalIndexWriter(methods_store),
    )
    event = RetrievalUpdateEvent(
        event_id="event-1",
        kind=RetrievalUpdateKind.UPSERT,
        ref=ref,
        retrieval_data_version="v1",
        occurred_at="2026-07-22T00:00:00+00:00",
    )
    assert sync.apply(event).status.value == "applied"

    service = RetrievalService(
        policy=DefaultRetrievalPolicy(),
        query_guard=QueryGuard(),
        sources={SourceType.CURRENT_SESSION: VerifiedSource(SourceType.CURRENT_SESSION, SourceUnit(ref, content))},
        methods={
            DenseRetrieval.method: DenseRetrieval(methods_store),
            LearnedSparseRetrieval.method: LearnedSparseRetrieval(methods_store),
            BM25Retrieval.method: BM25Retrieval(methods_store),
            LiteralBooleanFallback.method: LiteralBooleanFallback(methods_store),
        },
        token_estimator=lambda text: len(text.split()),
    )
    request = RetrievalRequest(
        request_id="request-1",
        model_call_purpose="TEST",
        query_proposal=QueryProposal((query,)),
        boundary=TrustedRetrievalBoundary(
            {SourceType.CURRENT_SESSION: source_filter},
            {SourceType.CURRENT_SESSION: SourceDependency.OPTIONAL},
        ),
    )
    context = service.retrieve_context(
        request,
        RetrievalBudget(candidate_limit_per_source=5, context_token_limit=100, max_items=5),
    )

    assert [item.content for item in context.items] == [content]
    with catalog.connect() as conn:
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(retrieval_units)").fetchall()}
        assert "content" not in columns
        assert conn.execute("SELECT COUNT(*) FROM dense_vectors").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM learned_sparse_postings").fetchone()[0] == 2
        assert conn.execute("SELECT COUNT(*) FROM bm25_unit_terms").fetchone()[0] == 1

    assert sync.apply(
        RetrievalUpdateEvent(
            event_id="event-2",
            kind=RetrievalUpdateKind.TRASH,
            ref=ref,
            retrieval_data_version="v1",
            occurred_at="2026-07-22T00:01:00+00:00",
        )
    ).status.value == "applied"
    after_trash = service.retrieve_context(
        request,
        RetrievalBudget(candidate_limit_per_source=5, context_token_limit=100, max_items=5),
    )
    assert after_trash.items == ()


def test_foundation_composition_is_independent_of_runtime_and_reports_only_safe_diagnostics(tmp_path):
    foundation = build_file_retrieval_foundation(
        db_path=tmp_path / "retrieval.sqlite",
        encoder=DeterministicEncoder({}),
    )

    snapshot = foundation.diagnostic_snapshot()
    assert snapshot["active_data_version"] is None
    assert snapshot["catalog_unit_count"] == 0
    assert snapshot["runtime_activation"] == "consumer_composed"
    assert snapshot["capabilities"]["encoder"] == {
        "fingerprint": "test-bge-m3",
        "learned_sparse_projection_assets_available": None,
    }
    assert snapshot["capabilities"]["retrieval_token_estimator"] == {
        "kind": "retrieval_encoder_token_ids",
        "fallback_count": 0,
        "cache_entries": 0,
    }
    assert foundation.service._token_estimator is foundation.retrieval_token_estimator
    assert "content" not in repr(snapshot).lower()


def test_symbol_only_document_unit_applies_instead_of_failing_the_generation(
    tmp_path,
):
    content = " | (.*)"
    source_filter = SourceFilter.from_mapping(
        SourceType.DOCUMENT,
        {"session_id": "s1", "doc_id": "d1"},
    )
    ref = SourceUnitRef(SourceType.DOCUMENT, "s1:d1:symbols", "r1", "hash-symbols")
    encoder = DeterministicLexicalEncoder()
    catalog = SqliteRetrievalCatalog(tmp_path / "retrieval.sqlite")
    catalog.initialize()
    catalog.create_data_version(
        version_id="v1",
        fingerprint=encoder.fingerprint(),
        role=RetrievalDataVersionRole.ACTIVE,
        state=RetrievalDataVersionState.READY,
    )
    method_store = SqliteRetrievalMethodStore(catalog=catalog, encoder=encoder)
    indexable = IndexableSourceUnit(
        ref=ref,
        source_filter=source_filter,
        content=content,
    )
    sync = RetrievalSyncService(
        catalog=catalog,
        source_readers={
            SourceType.DOCUMENT: Reader(SourceType.DOCUMENT, indexable)
        },
        index_writer=SqliteRetrievalIndexWriter(method_store),
    )

    result = sync.apply(
        RetrievalUpdateEvent(
            event_id="event-symbols",
            kind=RetrievalUpdateKind.UPSERT,
            ref=ref,
            retrieval_data_version="v1",
            occurred_at="2026-08-29T00:00:00+00:00",
        )
    )

    assert result.status.value == "applied"
    assert catalog.get_unit(ref, "v1").index_state.value == "ready"


def test_foundation_accepts_trusted_source_quota_configuration_when_consumer_composed(
    tmp_path,
):
    foundation = build_file_retrieval_foundation(
        db_path=tmp_path / "retrieval.sqlite",
        encoder=DeterministicEncoder({}),
        source_selection_quotas=SourceSelectionQuotas({SourceQuotaPool.DOCUMENT: 2}),
    )

    assert foundation.service._source_selection_quotas.item_limit(SourceQuotaPool.DOCUMENT) == 2
    assert foundation.diagnostic_snapshot()["runtime_activation"] == "consumer_composed"


def test_foundation_hides_a_drifted_active_generation_from_queries_and_diagnostics(tmp_path):
    spec = file_corpus_generation_spec(
        encoder_fingerprint="test-bge-m3",
        document_chunker_fingerprint="structure-first-test",
        document_chunk_contract_version=1,
        index_recipe=DOCUMENT_HYBRID_INDEX_RECIPE_CONTRACT,
    )
    foundation = build_file_retrieval_foundation(
        db_path=tmp_path / "retrieval.sqlite",
        encoder=DeterministicEncoder({}),
        generation_spec=spec,
    )
    foundation.catalog.create_data_version(
        version_id="legacy",
        fingerprint="encoder-only",
        role=RetrievalDataVersionRole.ACTIVE,
        state=RetrievalDataVersionState.READY,
    )

    assert foundation.data_version_provider.active_data_version() is None
    assert foundation.diagnostic_snapshot()["active_data_version"] is None
