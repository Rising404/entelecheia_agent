from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from personagraph.retrieval.contracts import (
    CURRENT_SESSION_TURN_CUTOFF_SCOPE_KEY,
    CURRENT_SESSION_TURN_INDEX_SCOPE_KEY,
    RetrievalStatus,
    RetrievalUnit,
    SourceFilter,
    SourceType,
    SourceUnitRef,
)
from personagraph.retrieval.indexing.methods import (
    BM25Retrieval,
    BgeM3EncodedText,
    DenseRetrieval,
    LearnedSparseRetrieval,
    RetrievalMethodUnavailable,
    SqliteRetrievalMethodStore,
)
from personagraph.retrieval.indexing.encoder import DeterministicLexicalEncoder
from personagraph.retrieval.sqlite_store import (
    RetrievalDataVersionRole,
    RetrievalDataVersionState,
    SqliteRetrievalCatalog,
)


def _vector(first: float) -> tuple[float, ...]:
    return (first,) + (0.0,) * 1023


@dataclass
class FakeBgeM3Encoder:
    encoded: dict[str, BgeM3EncodedText]
    tokenized: dict[str, tuple[int, ...]] = field(default_factory=dict)
    fail_encode: bool = False
    encode_calls: int = 0

    def encode(self, texts):
        self.encode_calls += 1
        if self.fail_encode:
            raise RetrievalMethodUnavailable("simulated_bge_forward_failure")
        return tuple(self.encoded[text] for text in texts)

    def token_ids(self, text: str):
        if text in self.tokenized:
            return self.tokenized[text]
        return self.encoded[text].token_ids

    def fingerprint(self) -> str:
        return "fake-bge-m3"


def _encoded(first: float, *, sparse: dict[int, float], tokens: tuple[int, ...]) -> BgeM3EncodedText:
    return BgeM3EncodedText(_vector(first), sparse, tokens)


def _catalog(tmp_path) -> SqliteRetrievalCatalog:
    catalog = SqliteRetrievalCatalog(tmp_path / "retrieval.sqlite")
    catalog.initialize()
    catalog.create_data_version(
        version_id="v1",
        fingerprint="test-bge-m3",
        role=RetrievalDataVersionRole.ACTIVE,
        state=RetrievalDataVersionState.READY,
    )
    return catalog


def _ref(unit_id: str, *, source_type: SourceType = SourceType.CURRENT_SESSION) -> SourceUnitRef:
    return SourceUnitRef(source_type, unit_id, "r1", f"hash-{unit_id}")


def _add_ready_unit(catalog, store, ref, source_filter, content):
    stored = catalog.upsert_pending_unit(
        RetrievalUnit(
            ref=ref,
            retrieval_data_version="v1",
            retrieval_status=RetrievalStatus.ACTIVE,
            source_filter=source_filter,
        )
    )
    store.index(stored, content)
    catalog.mark_unit_index_ready(stored.unit_id)
    return stored


def test_all_three_real_sqlite_methods_search_only_the_pre_filtered_source_scope(tmp_path):
    query = "北桥交付"
    encoder = FakeBgeM3Encoder(
        {
            query: _encoded(1.0, sparse={11: 1.0, 12: 0.5}, tokens=(11, 12)),
            "session-a": _encoded(1.0, sparse={11: 0.9, 12: 0.8}, tokens=(11, 12, 12)),
            "session-b": _encoded(1.0, sparse={11: 9.0, 12: 9.0}, tokens=(11, 12, 12, 12)),
            "document": _encoded(1.0, sparse={11: 9.0}, tokens=(11, 12)),
        }
    )
    catalog = _catalog(tmp_path)
    store = SqliteRetrievalMethodStore(catalog=catalog, encoder=encoder)
    session_a = SourceFilter.from_mapping(SourceType.CURRENT_SESSION, {"session_id": "a"})
    session_b = SourceFilter.from_mapping(SourceType.CURRENT_SESSION, {"session_id": "b"})
    document = SourceFilter.from_mapping(SourceType.DOCUMENT, {"doc_id": "d1"})
    good_ref = _ref("turn-a")
    _add_ready_unit(catalog, store, good_ref, session_a, "session-a")
    _add_ready_unit(catalog, store, _ref("turn-b"), session_b, "session-b")
    _add_ready_unit(catalog, store, _ref("doc-a", source_type=SourceType.DOCUMENT), document, "document")

    dense = DenseRetrieval(store).search(query, query_index=0, source_filter=session_a, limit=3)
    sparse = LearnedSparseRetrieval(store).search(query, query_index=0, source_filter=session_a, limit=3)
    bm25 = BM25Retrieval(store).search(query, query_index=0, source_filter=session_a, limit=3)
    health = {entry.method: entry for entry in store.method_index_health(catalog.active_units("v1")[0].unit_id)}

    assert [candidate.ref for candidate in dense] == [good_ref]
    assert [candidate.ref for candidate in sparse] == [good_ref]
    assert [candidate.ref for candidate in bm25] == [good_ref]
    assert all(entry.expected_state == "ready" and entry.representation_present for entry in health.values())


def test_current_session_cutoff_filters_newer_turns_before_top_k(tmp_path):
    query = "北桥交付"
    older = "较早但有效的北桥交付记录"
    newer = "更新且词项更强的北桥交付北桥交付记录"
    encoder = DeterministicLexicalEncoder()
    catalog = _catalog(tmp_path)
    store = SqliteRetrievalMethodStore(catalog=catalog, encoder=encoder)
    older_scope = SourceFilter.from_mapping(
        SourceType.CURRENT_SESSION,
        {
            "session_id": "a",
            CURRENT_SESSION_TURN_INDEX_SCOPE_KEY: "3",
        },
    )
    newer_scope = SourceFilter.from_mapping(
        SourceType.CURRENT_SESSION,
        {
            "session_id": "a",
            CURRENT_SESSION_TURN_INDEX_SCOPE_KEY: "5",
        },
    )
    frozen_scope = SourceFilter.from_mapping(
        SourceType.CURRENT_SESSION,
        {
            "session_id": "a",
            CURRENT_SESSION_TURN_CUTOFF_SCOPE_KEY: "3",
        },
    )
    older_ref = _ref("turn-older")
    _add_ready_unit(catalog, store, older_ref, older_scope, older)
    _add_ready_unit(catalog, store, _ref("turn-newer"), newer_scope, newer)

    assert catalog.active_unit_ids(
        data_version_id="v1",
        source_filter=frozen_scope,
    ) == (catalog.get_unit(older_ref, "v1").unit_id,)
    assert [
        candidate.ref
        for candidate in BM25Retrieval(store).search(
            query,
            query_index=0,
            source_filter=frozen_scope,
            limit=1,
        )
    ] == [older_ref]


def test_bge_forward_failure_still_builds_and_searches_bm25_when_tokenizer_is_available(tmp_path):
    query = "路径 docs/design.md"
    content = "文档路径"
    encoder = FakeBgeM3Encoder(
        {content: _encoded(1.0, sparse={42: 1.0}, tokens=(42, 77))},
        tokenized={query: (42, 77), content: (42, 77)},
        fail_encode=True,
    )
    catalog = _catalog(tmp_path)
    store = SqliteRetrievalMethodStore(catalog=catalog, encoder=encoder)
    source_filter = SourceFilter.from_mapping(SourceType.DOCUMENT, {"doc_id": "d1"})
    ref = _ref("doc-a", source_type=SourceType.DOCUMENT)
    _add_ready_unit(catalog, store, ref, source_filter, content)

    with pytest.raises(RetrievalMethodUnavailable):
        DenseRetrieval(store).search(query, query_index=0, source_filter=source_filter, limit=3)
    bm25 = BM25Retrieval(store).search(query, query_index=0, source_filter=source_filter, limit=3)
    health = {entry.method: entry for entry in store.method_index_health(catalog.active_units("v1")[0].unit_id)}

    assert [candidate.ref for candidate in bm25] == [ref]
    assert health[BM25Retrieval.method].expected_state == "ready"
    assert health[DenseRetrieval.method].expected_state == "absent"
    assert (
        health[DenseRetrieval.method].reason_code
        == "simulated_bge_forward_failure"
    )
    assert health[LearnedSparseRetrieval.method].expected_state == "absent"
    assert (
        health[LearnedSparseRetrieval.method].reason_code
        == "simulated_bge_forward_failure"
    )


def test_dependency_free_lexical_encoder_builds_a_stable_bm25_generation(tmp_path):
    query = "论文 方法"
    content = "论文方法与实验结果"
    encoder = DeterministicLexicalEncoder()
    assert encoder.token_ids(query) == DeterministicLexicalEncoder().token_ids(query)
    assert encoder.token_ids(query)
    catalog = _catalog(tmp_path)
    store = SqliteRetrievalMethodStore(catalog=catalog, encoder=encoder)
    source_filter = SourceFilter.from_mapping(SourceType.DOCUMENT, {"doc_id": "d1"})
    ref = _ref("doc-lexical", source_type=SourceType.DOCUMENT)

    _add_ready_unit(catalog, store, ref, source_filter, content)

    assert BM25Retrieval(store).search(
        query,
        query_index=0,
        source_filter=source_filter,
        limit=3,
    )[0].ref == ref
    health = {
        entry.method: entry
        for entry in store.method_index_health(catalog.active_units("v1")[0].unit_id)
    }
    assert health[BM25Retrieval.method].expected_state == "ready"
    assert health[DenseRetrieval.method].expected_state == "absent"
    assert health[LearnedSparseRetrieval.method].expected_state == "absent"


def test_dependency_free_lexical_encoder_indexes_symbol_only_source_units(tmp_path):
    query = ".*"
    content = " | (.*)"
    encoder = DeterministicLexicalEncoder()
    catalog = _catalog(tmp_path)
    store = SqliteRetrievalMethodStore(catalog=catalog, encoder=encoder)
    source_filter = SourceFilter.from_mapping(SourceType.DOCUMENT, {"doc_id": "d1"})
    ref = _ref("doc-symbols", source_type=SourceType.DOCUMENT)

    _add_ready_unit(catalog, store, ref, source_filter, content)

    assert BM25Retrieval(store).search(
        query,
        query_index=0,
        source_filter=source_filter,
        limit=3,
    )[0].ref == ref


def test_missing_learned_sparse_projection_assets_fail_closed_and_leave_dense_bm25_usable(tmp_path):
    query = "检索版本"
    content = "检索版本 retrieval-v3"
    encoder = FakeBgeM3Encoder(
        {
            content: _encoded(1.0, sparse={}, tokens=(42, 77)),
            query: _encoded(1.0, sparse={}, tokens=(42, 77)),
        }
    )
    encoder.learned_sparse_available = False
    catalog = _catalog(tmp_path)
    store = SqliteRetrievalMethodStore(catalog=catalog, encoder=encoder)
    source_filter = SourceFilter.from_mapping(SourceType.DOCUMENT, {"doc_id": "d1"})
    ref = _ref("doc-a", source_type=SourceType.DOCUMENT)
    _add_ready_unit(catalog, store, ref, source_filter, content)

    dense = DenseRetrieval(store).search(query, query_index=0, source_filter=source_filter, limit=3)
    with pytest.raises(RetrievalMethodUnavailable, match="projection_assets_unavailable"):
        LearnedSparseRetrieval(store).search(query, query_index=0, source_filter=source_filter, limit=3)
    bm25 = BM25Retrieval(store).search(query, query_index=0, source_filter=source_filter, limit=3)
    health = {entry.method: entry for entry in store.method_index_health(catalog.active_units("v1")[0].unit_id)}

    assert [candidate.ref for candidate in dense] == [ref]
    assert [candidate.ref for candidate in bm25] == [ref]
    assert health[DenseRetrieval.method].expected_state == "ready"
    assert health[LearnedSparseRetrieval.method].expected_state == "absent"
    assert health[BM25Retrieval.method].expected_state == "ready"


def test_purge_removes_every_derived_method_record(tmp_path):
    query = "北桥"
    content = "北桥内容"
    encoder = FakeBgeM3Encoder(
        {
            query: _encoded(1.0, sparse={11: 1.0}, tokens=(11,)),
            content: _encoded(1.0, sparse={11: 1.0}, tokens=(11,)),
        }
    )
    catalog = _catalog(tmp_path)
    store = SqliteRetrievalMethodStore(catalog=catalog, encoder=encoder)
    source_filter = SourceFilter.from_mapping(SourceType.CURRENT_SESSION, {"session_id": "a"})
    stored = _add_ready_unit(catalog, store, _ref("turn-a"), source_filter, content)

    store.purge(stored)
    assert DenseRetrieval(store).search(query, query_index=0, source_filter=source_filter, limit=3) == ()
    assert LearnedSparseRetrieval(store).search(query, query_index=0, source_filter=source_filter, limit=3) == ()
    assert BM25Retrieval(store).search(query, query_index=0, source_filter=source_filter, limit=3) == ()
