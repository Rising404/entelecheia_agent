"""共享 SQLite 方法存储之上的轻量检索方法端口。"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

from ..contracts import RetrievalMethod, SourceFilter
from .encoder import RetrievalMethodUnavailable
from ..sqlite_store import StoredUnit

if TYPE_CHECKING:
    from .methods import SqliteRetrievalMethodStore


class DenseRetrieval:
    method = RetrievalMethod.DENSE

    def __init__(self, store: SqliteRetrievalMethodStore) -> None:
        self._store = store

    def search(
        self,
        query: str,
        *,
        query_index: int,
        source_filter: SourceFilter,
        limit: int,
        retrieval_data_version_id: str | None = None,
    ):
        return self._store.search_dense(
            query,
            query_index=query_index,
            source_filter=source_filter,
            limit=limit,
            retrieval_data_version_id=retrieval_data_version_id,
        )


class LearnedSparseRetrieval:
    method = RetrievalMethod.LEARNED_SPARSE

    def __init__(self, store: SqliteRetrievalMethodStore) -> None:
        self._store = store

    def search(
        self,
        query: str,
        *,
        query_index: int,
        source_filter: SourceFilter,
        limit: int,
        retrieval_data_version_id: str | None = None,
    ):
        return self._store.search_learned_sparse(
            query,
            query_index=query_index,
            source_filter=source_filter,
            limit=limit,
            retrieval_data_version_id=retrieval_data_version_id,
        )


class BM25Retrieval:
    method = RetrievalMethod.BM25

    def __init__(self, store: SqliteRetrievalMethodStore) -> None:
        self._store = store

    def search(
        self,
        query: str,
        *,
        query_index: int,
        source_filter: SourceFilter,
        limit: int,
        retrieval_data_version_id: str | None = None,
    ):
        return self._store.search_bm25(
            query,
            query_index=query_index,
            source_filter=source_filter,
            limit=limit,
            retrieval_data_version_id=retrieval_data_version_id,
        )


class LiteralBooleanFallback:
    method = RetrievalMethod.LITERAL_BOOLEAN

    def __init__(self, store: SqliteRetrievalMethodStore) -> None:
        self._store = store

    def search(
        self,
        query: str,
        *,
        query_index: int,
        source_filter: SourceFilter,
        limit: int,
        retrieval_data_version_id: str | None = None,
    ):
        return self._store.search_literal_boolean(
            query,
            query_index=query_index,
            source_filter=source_filter,
            limit=limit,
            retrieval_data_version_id=retrieval_data_version_id,
        )


class SqliteRetrievalIndexWriter:
    """同步桥：一次权威 Source 读取写入所有派生方法。"""

    def __init__(
        self,
        store: SqliteRetrievalMethodStore,
        *,
        required_methods: Iterable[RetrievalMethod] = (),
    ) -> None:
        self._store = store
        required = frozenset(RetrievalMethod(method) for method in required_methods)
        if RetrievalMethod.LITERAL_BOOLEAN in required:
            raise ValueError("literal boolean is a query fallback, not an index method")
        self._required_methods = required

    def index(self, stored_unit: StoredUnit, source_unit) -> None:
        indexed = self._store.index(stored_unit, source_unit.content)
        missing = self._required_methods - indexed.methods
        if missing:
            upstream_failure = next(
                (
                    failure
                    for failure in indexed.failures
                    if failure.methods & missing
                ),
                None,
            )
            if upstream_failure is not None:
                raise RetrievalMethodUnavailable(
                    upstream_failure.safe_error_code,
                    stage=upstream_failure.stage,
                )
            names = ",".join(sorted(method.value for method in missing))
            raise RetrievalMethodUnavailable(
                f"required_index_methods_unavailable:{names}",
                stage="index_coverage",
            )

    def purge(self, stored_unit: StoredUnit) -> None:
        self._store.purge(stored_unit)
