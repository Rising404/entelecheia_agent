from __future__ import annotations

from types import SimpleNamespace

import pytest

from personagraph.retrieval.contracts import RetrievalMethod
from personagraph.retrieval.indexing.adapters import SqliteRetrievalIndexWriter
from personagraph.retrieval.indexing.encoder import RetrievalMethodUnavailable
from personagraph.retrieval.indexing.methods import IndexedMethods, MethodIndexFailure


class _MethodStore:
    def __init__(self, indexed: IndexedMethods) -> None:
        self.indexed = indexed
        self.contents: list[str] = []

    def index(self, stored_unit, content: str) -> IndexedMethods:
        del stored_unit
        self.contents.append(content)
        return self.indexed

    def purge(self, stored_unit) -> None:
        del stored_unit


def test_writer_rejects_a_unit_missing_a_generation_required_method():
    store = _MethodStore(
        IndexedMethods(frozenset({RetrievalMethod.DENSE, RetrievalMethod.BM25}))
    )
    writer = SqliteRetrievalIndexWriter(
        store,
        required_methods=(
            RetrievalMethod.DENSE,
            RetrievalMethod.LEARNED_SPARSE,
            RetrievalMethod.BM25,
        ),
    )

    with pytest.raises(
        RetrievalMethodUnavailable,
        match="required_index_methods_unavailable:learned_sparse",
    ):
        writer.index(object(), SimpleNamespace(content="authoritative text"))

    assert store.contents == ["authoritative text"]


def test_writer_preserves_the_safe_upstream_failure_for_missing_required_methods():
    store = _MethodStore(
        IndexedMethods(
            frozenset({RetrievalMethod.BM25}),
            degraded_from=(RetrievalMethod.DENSE, RetrievalMethod.LEARNED_SPARSE),
            failures=(
                MethodIndexFailure(
                    methods=frozenset(
                        {RetrievalMethod.DENSE, RetrievalMethod.LEARNED_SPARSE}
                    ),
                    stage="encode",
                    safe_error_code="bge_m3_encode_failed:RuntimeError",
                ),
            ),
        )
    )
    writer = SqliteRetrievalIndexWriter(
        store,
        required_methods=(
            RetrievalMethod.DENSE,
            RetrievalMethod.LEARNED_SPARSE,
            RetrievalMethod.BM25,
        ),
    )

    with pytest.raises(RetrievalMethodUnavailable) as caught:
        writer.index(object(), SimpleNamespace(content="authoritative text"))

    assert caught.value.stage == "encode"
    assert caught.value.safe_error_code == "bge_m3_encode_failed:RuntimeError"


def test_writer_accepts_exact_required_method_coverage():
    methods = frozenset(
        {
            RetrievalMethod.DENSE,
            RetrievalMethod.LEARNED_SPARSE,
            RetrievalMethod.BM25,
        }
    )
    writer = SqliteRetrievalIndexWriter(
        _MethodStore(IndexedMethods(methods)),
        required_methods=methods,
    )

    writer.index(object(), SimpleNamespace(content="authoritative text"))


def test_literal_boolean_cannot_be_claimed_as_a_persisted_index_method():
    store = _MethodStore(IndexedMethods(frozenset({RetrievalMethod.BM25})))

    with pytest.raises(ValueError, match="query fallback"):
        SqliteRetrievalIndexWriter(
            store,
            required_methods=(RetrievalMethod.LITERAL_BOOLEAN,),
        )
