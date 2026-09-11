"""检索 Unit 派生表示的 SQLite 存储。

公开方法类仍可从本模块导入。其编码器和端口适配器实现位于 ``retrieval.indexing``，
使本文件可以专注于共享派生存储生命周期和查询语义。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import re
import sqlite3
from typing import Any

from ..contracts import (
    CURRENT_SESSION_TURN_CUTOFF_SCOPE_KEY,
    CURRENT_SESSION_TURN_INDEX_SCOPE_KEY,
    RetrievalCandidate,
    RetrievalMethod,
    RetrievalStatus,
    SourceFilter,
    SourceType,
)
from .adapters import (
    BM25Retrieval,
    DenseRetrieval,
    LearnedSparseRetrieval,
    LiteralBooleanFallback,
    SqliteRetrievalIndexWriter,
)
from .encoder import (
    BgeM3EncodedText,
    BgeM3Encoder,
    RetrievalMethodUnavailable,
    normalise_sparse_weights,
)
from ..ports import BgeM3EncoderPort
from ..sqlite_store import SqliteRetrievalCatalog, StoredUnit, UnitIndexState


@dataclass(frozen=True, slots=True)
class IndexedMethods:
    """一次来源读取后为一个 Unit 成功写入的方法。"""

    methods: frozenset[RetrievalMethod]
    degraded_from: tuple[RetrievalMethod, ...] = ()
    failures: tuple[MethodIndexFailure, ...] = ()


@dataclass(frozen=True, slots=True)
class MethodIndexFailure:
    """不含来源正文的索引失败分类，可安全进入持久审计。"""

    methods: frozenset[RetrievalMethod]
    stage: str
    safe_error_code: str

    def __post_init__(self) -> None:
        if not self.methods:
            raise ValueError("method index failure must name at least one method")
        if (
            _SAFE_INDEX_DIAGNOSTIC_RE.fullmatch(self.stage) is None
            or _SAFE_INDEX_DIAGNOSTIC_RE.fullmatch(self.safe_error_code) is None
        ):
            raise ValueError("method index failure stage and code must be safe tokens")


@dataclass(frozen=True, slots=True)
class MethodIndexManifestEntry:
    """一个 Unit 与方法的发布记录，不含 Source 正文。"""

    method: RetrievalMethod
    state: str
    reason_code: str | None


@dataclass(frozen=True, slots=True)
class UnitMethodIndexHealth:
    """预期方法发布与物理派生表示之间的对照。"""

    method: RetrievalMethod
    expected_state: str | None
    representation_present: bool
    reason_code: str | None = None


class SqliteRetrievalMethodStore:
    """共享同一 Unit 目录的三个独立派生存储。"""

    def __init__(self, *, catalog: SqliteRetrievalCatalog, encoder: BgeM3EncoderPort) -> None:
        self._catalog = catalog
        self._encoder = encoder
        self._catalog.refresh_method_capabilities()

    def encoder_capability_snapshot(self) -> dict[str, object]:
        """供诊断使用的安全方法能力状态，不含 Source 数据。"""

        sparse_available = getattr(self._encoder, "learned_sparse_available", None)
        return {
            "fingerprint": self._encoder.fingerprint(),
            "learned_sparse_projection_assets_available": (
                bool(sparse_available) if sparse_available is not None else None
            ),
        }

    def index(self, stored_unit: StoredUnit, content: str) -> IndexedMethods:
        """写入所有可获得的索引，但不发布只构建了一半的 Unit。

        在此方法返回前，同步服务会让目录 Unit 保持 ``pending``。因此，仅稀疏存储失败时
        可以发布已验证的 Dense+BM25 降级 Unit；若 BGE 完全失败但 BGE tokenizer 仍可用，
        仍可以发布 BM25。
        """

        try:
            encoded = tuple(self._encoder.encode((content,)))[0]
        except Exception as exc:
            failure = _method_index_failure(
                (RetrievalMethod.DENSE, RetrievalMethod.LEARNED_SPARSE),
                exc,
                fallback_stage="encode",
                fallback_code="encoder_exception",
            )
            return self._index_bm25_only(
                stored_unit,
                content,
                failures=(failure,),
            )

        written: set[RetrievalMethod] = set()
        failures: list[MethodIndexFailure] = []
        with self._catalog.connect() as conn:
            try:
                self._replace_dense(conn, stored_unit.unit_id, encoded.dense_vector)
            except Exception as exc:
                failures.append(
                    _method_index_failure(
                        (RetrievalMethod.DENSE,),
                        exc,
                        fallback_stage="dense_write",
                        fallback_code="dense_index_write_exception",
                    )
                )
            else:
                written.add(RetrievalMethod.DENSE)
            try:
                self._replace_sparse(conn, stored_unit.unit_id, encoded.learned_sparse_weights)
            except Exception as exc:
                failures.append(
                    _method_index_failure(
                        (RetrievalMethod.LEARNED_SPARSE,),
                        exc,
                        fallback_stage="learned_sparse_write",
                        fallback_code="learned_sparse_index_write_exception",
                    )
                )
            else:
                written.add(RetrievalMethod.LEARNED_SPARSE)
            try:
                self._replace_bm25(conn, stored_unit.unit_id, encoded.token_ids)
            except Exception as exc:
                failures.append(
                    _method_index_failure(
                        (RetrievalMethod.BM25,),
                        exc,
                        fallback_stage="bm25_write",
                        fallback_code="bm25_index_write_exception",
                    )
                )
            else:
                written.add(RetrievalMethod.BM25)
            self._replace_method_manifest(
                conn,
                unit_id=stored_unit.unit_id,
                ready_methods=written,
                failures=failures,
            )
        if not written:
            primary = failures[0]
            raise RetrievalMethodUnavailable(
                primary.safe_error_code,
                stage=primary.stage,
            )
        degraded = tuple(
            method
            for method in (
                RetrievalMethod.DENSE,
                RetrievalMethod.LEARNED_SPARSE,
                RetrievalMethod.BM25,
            )
            if any(method in failure.methods for failure in failures)
        )
        return IndexedMethods(
            frozenset(written),
            degraded_from=degraded,
            failures=tuple(failures),
        )

    def _index_bm25_only(
        self,
        stored_unit: StoredUnit,
        content: str,
        *,
        failures: tuple[MethodIndexFailure, ...],
    ) -> IndexedMethods:
        token_ids = tuple(int(token_id) for token_id in self._encoder.token_ids(content))
        with self._catalog.connect() as conn:
            self._replace_bm25(conn, stored_unit.unit_id, token_ids)
            self._replace_method_manifest(
                conn,
                unit_id=stored_unit.unit_id,
                ready_methods={RetrievalMethod.BM25},
                failures=failures,
            )
        return IndexedMethods(
            methods=frozenset({RetrievalMethod.BM25}),
            degraded_from=(RetrievalMethod.DENSE, RetrievalMethod.LEARNED_SPARSE),
            failures=failures,
        )

    def purge(self, stored_unit: StoredUnit) -> None:
        with self._catalog.connect() as conn:
            if _table_exists(conn, "dense_vectors"):
                conn.execute("DELETE FROM dense_vectors WHERE unit_id=?", (stored_unit.unit_id,))
            conn.execute("DELETE FROM learned_sparse_postings WHERE unit_id=?", (stored_unit.unit_id,))
            self._delete_bm25(conn, stored_unit.unit_id)
            conn.execute("DELETE FROM retrieval_unit_method_indexes WHERE unit_id=?", (stored_unit.unit_id,))

    def method_index_health(self, unit_id: int) -> tuple[UnitMethodIndexHealth, ...]:
        """检查派生存储以进行协调，不读取 Source 内容。"""

        with self._catalog.connect() as conn:
            manifest_rows = conn.execute(
                "SELECT method, state, reason_code FROM retrieval_unit_method_indexes WHERE unit_id=?",
                (unit_id,),
            ).fetchall()
            manifest = {
                RetrievalMethod(row["method"]): (str(row["state"]), row["reason_code"])
                for row in manifest_rows
            }
            present = {
                RetrievalMethod.DENSE: _table_exists(conn, "dense_vectors")
                and conn.execute("SELECT 1 FROM dense_vectors WHERE unit_id=?", (unit_id,)).fetchone() is not None,
                RetrievalMethod.LEARNED_SPARSE: conn.execute(
                    "SELECT 1 FROM learned_sparse_postings WHERE unit_id=? LIMIT 1", (unit_id,)
                ).fetchone()
                is not None,
                RetrievalMethod.BM25: conn.execute(
                    "SELECT 1 FROM bm25_unit_terms WHERE unit_id=?", (unit_id,)
                ).fetchone()
                is not None,
            }
        return tuple(
            UnitMethodIndexHealth(
                method=method,
                expected_state=manifest.get(method, (None, None))[0],
                reason_code=manifest.get(method, (None, None))[1],
                representation_present=present[method],
            )
            for method in (
                RetrievalMethod.DENSE,
                RetrievalMethod.LEARNED_SPARSE,
                RetrievalMethod.BM25,
            )
        )

    def generation_method_index_health(
        self,
        data_version_id: str,
    ) -> dict[int, tuple[UnitMethodIndexHealth, ...]]:
        """一次读取整个 generation 的逐方法发布证明。

        Rollout 必须验证每个 Unit，而逐 Unit 打开 SQLite 连接会让大型文档库的发布检查
        退化为 N 次查询。本接口仍只返回派生表示是否存在，不读取任何 Source 正文。
        """

        if not isinstance(data_version_id, str) or not data_version_id.strip():
            raise ValueError("data_version_id must not be empty")
        with self._catalog.connect() as conn:
            return self.generation_method_index_health_in_connection(
                conn,
                data_version_id,
            )

    def generation_method_index_health_in_connection(
        self,
        conn: sqlite3.Connection,
        data_version_id: str,
    ) -> dict[int, tuple[UnitMethodIndexHealth, ...]]:
        """在调用方持有的 SQLite 快照/写围栏内验证完整 generation。

        发布方必须在冻结 Document 权威状态的同一事务中复核方法 manifest 与物理表示；
        另开连接会在 readiness 检查和活动指针交换之间留下竞态窗口。
        """

        if not isinstance(conn, sqlite3.Connection):
            raise TypeError("conn must be sqlite3.Connection")
        if not isinstance(data_version_id, str) or not data_version_id.strip():
            raise ValueError("data_version_id must not be empty")
        unit_ids = tuple(
            int(row["unit_id"])
            for row in conn.execute(
                "SELECT unit_id FROM retrieval_units WHERE data_version_id=? "
                "ORDER BY unit_id",
                (data_version_id,),
            ).fetchall()
        )
        if not unit_ids:
            return {}
        manifest_rows = conn.execute(
            "SELECT m.unit_id, m.method, m.state, m.reason_code "
            "FROM retrieval_unit_method_indexes AS m "
            "JOIN retrieval_units AS u ON u.unit_id=m.unit_id "
            "WHERE u.data_version_id=?",
            (data_version_id,),
        ).fetchall()
        manifests = {
            (int(row["unit_id"]), RetrievalMethod(row["method"])): (
                str(row["state"]),
                row["reason_code"],
            )
            for row in manifest_rows
        }
        present: dict[RetrievalMethod, set[int]] = {
            RetrievalMethod.DENSE: set(),
            RetrievalMethod.LEARNED_SPARSE: set(),
            RetrievalMethod.BM25: set(),
        }
        if _table_exists(conn, "dense_vectors"):
            present[RetrievalMethod.DENSE] = {
                int(row["unit_id"])
                for row in conn.execute(
                    "SELECT d.unit_id FROM dense_vectors AS d "
                    "JOIN retrieval_units AS u ON u.unit_id=d.unit_id "
                    "WHERE u.data_version_id=?",
                    (data_version_id,),
                ).fetchall()
            }
        present[RetrievalMethod.LEARNED_SPARSE] = {
            int(row["unit_id"])
            for row in conn.execute(
                "SELECT DISTINCT p.unit_id FROM learned_sparse_postings AS p "
                "JOIN retrieval_units AS u ON u.unit_id=p.unit_id "
                "WHERE u.data_version_id=?",
                (data_version_id,),
            ).fetchall()
        }
        present[RetrievalMethod.BM25] = {
            int(row["unit_id"])
            for row in conn.execute(
                "SELECT b.unit_id FROM bm25_unit_terms AS b "
                "JOIN retrieval_units AS u ON u.unit_id=b.unit_id "
                "WHERE u.data_version_id=?",
                (data_version_id,),
            ).fetchall()
        }
        methods = (
            RetrievalMethod.DENSE,
            RetrievalMethod.LEARNED_SPARSE,
            RetrievalMethod.BM25,
        )
        return {
            unit_id: tuple(
                UnitMethodIndexHealth(
                    method=method,
                    expected_state=manifests.get((unit_id, method), (None, None))[0],
                    reason_code=manifests.get((unit_id, method), (None, None))[1],
                    representation_present=unit_id in present[method],
                )
                for method in methods
            )
            for unit_id in unit_ids
        }

    def orphaned_representation_unit_ids(self) -> dict[RetrievalMethod, tuple[int, ...]]:
        """查找其目录 Unit 已被移除的派生行。"""

        with self._catalog.connect() as conn:
            dense_ids = ()
            if _table_exists(conn, "dense_vectors"):
                dense_ids = tuple(
                    int(row["unit_id"])
                    for row in conn.execute(
                        "SELECT d.unit_id FROM dense_vectors d "
                        "LEFT JOIN retrieval_units u ON u.unit_id=d.unit_id "
                        "WHERE u.unit_id IS NULL ORDER BY d.unit_id"
                    ).fetchall()
                )
            sparse_ids = tuple(
                int(row["unit_id"])
                for row in conn.execute(
                    "SELECT DISTINCT p.unit_id FROM learned_sparse_postings p "
                    "LEFT JOIN retrieval_units u ON u.unit_id=p.unit_id "
                    "WHERE u.unit_id IS NULL ORDER BY p.unit_id"
                ).fetchall()
            )
            bm25_ids = tuple(
                int(row["unit_id"])
                for row in conn.execute(
                    "SELECT b.unit_id FROM bm25_unit_terms b "
                    "LEFT JOIN retrieval_units u ON u.unit_id=b.unit_id "
                    "WHERE u.unit_id IS NULL ORDER BY b.unit_id"
                ).fetchall()
            )
        return {
            RetrievalMethod.DENSE: dense_ids,
            RetrievalMethod.LEARNED_SPARSE: sparse_ids,
            RetrievalMethod.BM25: bm25_ids,
        }

    def purge_orphaned_representation(self, unit_id: int) -> None:
        """只删除已无目录权威依据的 Unit 派生行。"""

        with self._catalog.connect() as conn:
            if _table_exists(conn, "dense_vectors"):
                conn.execute("DELETE FROM dense_vectors WHERE unit_id=?", (unit_id,))
            conn.execute("DELETE FROM learned_sparse_postings WHERE unit_id=?", (unit_id,))
            self._delete_bm25(conn, unit_id)
            conn.execute("DELETE FROM retrieval_unit_method_indexes WHERE unit_id=?", (unit_id,))

    def search_dense(
        self,
        query: str,
        *,
        query_index: int,
        source_filter: SourceFilter,
        limit: int,
        retrieval_data_version_id: str | None = None,
    ) -> tuple[RetrievalCandidate, ...]:
        self._require_positive_limit(limit)
        if not _capability_available(self._catalog, "sqlite_vec"):
            raise RetrievalMethodUnavailable("sqlite_vec_unavailable")
        encoded = self._query_encoding(query)
        data_version_id = self._resolve_data_version_id(retrieval_data_version_id)
        if data_version_id is None:
            return ()
        unit_ids = self._catalog.active_unit_ids(
            data_version_id=data_version_id,
            source_filter=source_filter,
        )
        if not unit_ids:
            return ()
        try:
            import sqlite_vec

            query_vector = sqlite_vec.serialize_float32(encoded.dense_vector)
        except Exception as exc:
            raise RetrievalMethodUnavailable(f"dense_query_serialization_failed:{type(exc).__name__}") from exc
        distances: list[tuple[int, float]] = []
        with self._catalog.connect() as conn:
            for unit_id_batch in _batches(unit_ids, 900):
                placeholders = ",".join("?" for _ in unit_id_batch)
                rows = conn.execute(
                    "SELECT unit_id, distance FROM dense_vectors "
                    f"WHERE embedding MATCH ? AND k=? AND unit_id IN ({placeholders}) ORDER BY distance",
                    [query_vector, limit, *unit_id_batch],
                ).fetchall()
                distances.extend((int(row["unit_id"]), float(row["distance"])) for row in rows)
        return self._candidates_from_unit_scores(
            source_filter=source_filter,
            data_version_id=data_version_id,
            query_index=query_index,
            method=RetrievalMethod.DENSE,
            ordered_scores=sorted(distances, key=lambda item: (item[1], item[0]))[:limit],
            higher_is_better=False,
        )

    def search_learned_sparse(
        self,
        query: str,
        *,
        query_index: int,
        source_filter: SourceFilter,
        limit: int,
        retrieval_data_version_id: str | None = None,
    ) -> tuple[RetrievalCandidate, ...]:
        self._require_positive_limit(limit)
        if getattr(self._encoder, "learned_sparse_available", None) is False:
            raise RetrievalMethodUnavailable("learned_sparse_projection_assets_unavailable")
        encoded = self._query_encoding(query)
        query_weights = _bounded_sparse_weights(encoded.learned_sparse_weights)
        if not query_weights:
            return ()
        data_version_id = self._resolve_data_version_id(retrieval_data_version_id)
        if data_version_id is None:
            return ()
        values_sql = ", ".join("(?, ?)" for _ in query_weights)
        clauses, params = _unit_filter_sql(
            source_filter,
            data_version_id=data_version_id,
            alias="u",
        )
        query_params: list[Any] = []
        for token_id, weight in query_weights.items():
            query_params.extend((token_id, weight))
        sql = (
            f"WITH query_weights(token_id, weight) AS (VALUES {values_sql}) "
            "SELECT p.unit_id, SUM(p.weight * q.weight) AS score "
            "FROM learned_sparse_postings p "
            "JOIN query_weights q ON q.token_id=p.token_id "
            "JOIN retrieval_units u ON u.unit_id=p.unit_id "
            f"WHERE {' AND '.join(clauses)} "
            "GROUP BY p.unit_id ORDER BY score DESC, p.unit_id LIMIT ?"
        )
        with self._catalog.connect() as conn:
            rows = conn.execute(sql, [*query_params, *params, limit]).fetchall()
        return self._candidates_from_unit_scores(
            source_filter=source_filter,
            data_version_id=data_version_id,
            query_index=query_index,
            method=RetrievalMethod.LEARNED_SPARSE,
            ordered_scores=[(int(row["unit_id"]), float(row["score"])) for row in rows],
            higher_is_better=True,
        )

    def search_bm25(
        self,
        query: str,
        *,
        query_index: int,
        source_filter: SourceFilter,
        limit: int,
        retrieval_data_version_id: str | None = None,
    ) -> tuple[RetrievalCandidate, ...]:
        self._require_positive_limit(limit)
        if not _capability_available(self._catalog, "fts5"):
            raise RetrievalMethodUnavailable("fts5_unavailable")
        token_ids = _bounded_token_ids(self._encoder.token_ids(query))
        if not token_ids:
            return ()
        data_version_id = self._resolve_data_version_id(retrieval_data_version_id)
        if data_version_id is None:
            return ()
        clauses, params = _unit_filter_sql(
            source_filter,
            data_version_id=data_version_id,
            alias="u",
        )
        match_query = " OR ".join(_token_term(token_id) for token_id in dict.fromkeys(token_ids))
        sql = (
            "SELECT f.rowid AS unit_id, bm25(bm25_fts) AS score "
            "FROM bm25_fts f JOIN retrieval_units u ON u.unit_id=f.rowid "
            f"WHERE bm25_fts MATCH ? AND {' AND '.join(clauses)} "
            "ORDER BY score ASC, f.rowid LIMIT ?"
        )
        with self._catalog.connect() as conn:
            rows = conn.execute(sql, [match_query, *params, limit]).fetchall()
        return self._candidates_from_unit_scores(
            source_filter=source_filter,
            data_version_id=data_version_id,
            query_index=query_index,
            method=RetrievalMethod.BM25,
            ordered_scores=[(int(row["unit_id"]), float(row["score"])) for row in rows],
            higher_is_better=False,
        )

    def search_literal_boolean(
        self,
        query: str,
        *,
        query_index: int,
        source_filter: SourceFilter,
        limit: int,
        retrieval_data_version_id: str | None = None,
    ) -> tuple[RetrievalCandidate, ...]:
        """基于影子词项的保守最终回退，而非 BM25 替代品。"""

        self._require_positive_limit(limit)
        token_ids = _bounded_token_ids(self._encoder.token_ids(query))
        if not token_ids:
            return ()
        data_version_id = self._resolve_data_version_id(retrieval_data_version_id)
        if data_version_id is None:
            return ()
        clauses, params = _unit_filter_sql(
            source_filter,
            data_version_id=data_version_id,
            alias="u",
        )
        terms = tuple(dict.fromkeys(_token_term(token_id) for token_id in token_ids))
        term_cases = " + ".join("CASE WHEN b.shadow_terms LIKE ? THEN 1 ELSE 0 END" for _ in terms)
        matching_clause = " OR ".join("b.shadow_terms LIKE ?" for _ in terms)
        sql = (
            f"SELECT b.unit_id, ({term_cases}) AS score FROM bm25_unit_terms b "
            "JOIN retrieval_units u ON u.unit_id=b.unit_id "
            f"WHERE {' AND '.join(clauses)} AND ({matching_clause}) "
            "ORDER BY score DESC, b.unit_id LIMIT ?"
        )
        like_params = [f"%{term}%" for term in terms]
        with self._catalog.connect() as conn:
            rows = conn.execute(sql, [*like_params, *params, *like_params, limit]).fetchall()
        return self._candidates_from_unit_scores(
            source_filter=source_filter,
            data_version_id=data_version_id,
            query_index=query_index,
            method=RetrievalMethod.LITERAL_BOOLEAN,
            ordered_scores=[(int(row["unit_id"]), float(row["score"])) for row in rows],
            higher_is_better=True,
        )

    def _query_encoding(self, query: str) -> BgeM3EncodedText:
        encode_query = getattr(self._encoder, "encode_query", None)
        if callable(encode_query):
            return encode_query(query)
        return tuple(self._encoder.encode((query,)))[0]

    def _active_version_id(self) -> str | None:
        active = self._catalog.active_data_version()
        return active.id if active is not None else None

    def _resolve_data_version_id(self, retrieval_data_version_id: str | None) -> str | None:
        """调用方提供请求固定版本时使用该版本。"""

        if retrieval_data_version_id is None:
            return self._active_version_id()
        if not isinstance(retrieval_data_version_id, str) or not retrieval_data_version_id.strip():
            raise ValueError("retrieval_data_version_id must be a non-empty string when provided")
        return retrieval_data_version_id

    def _replace_dense(self, conn: sqlite3.Connection, unit_id: int, vector: Sequence[float]) -> None:
        if not _table_exists(conn, "dense_vectors"):
            raise RetrievalMethodUnavailable("dense_store_unavailable")
        try:
            import sqlite_vec

            serialised = sqlite_vec.serialize_float32(vector)
            conn.execute("DELETE FROM dense_vectors WHERE unit_id=?", (unit_id,))
            conn.execute(
                "INSERT INTO dense_vectors(unit_id, embedding, scope_key) VALUES (?, ?, ?)",
                (unit_id, serialised, "catalog_scope"),
            )
        except Exception as exc:
            raise RetrievalMethodUnavailable(f"dense_index_write_failed:{type(exc).__name__}") from exc

    @staticmethod
    def _replace_sparse(
        conn: sqlite3.Connection,
        unit_id: int,
        weights: Mapping[int, float],
    ) -> None:
        normalised = normalise_sparse_weights(weights)
        if not normalised:
            raise RetrievalMethodUnavailable("sparse_index_has_no_terms")
        conn.execute("DELETE FROM learned_sparse_postings WHERE unit_id=?", (unit_id,))
        conn.executemany(
            "INSERT INTO learned_sparse_postings(unit_id, token_id, weight) VALUES (?, ?, ?)",
            [(unit_id, token_id, weight) for token_id, weight in normalised.items()],
        )

    @staticmethod
    def _replace_method_manifest(
        conn: sqlite3.Connection,
        *,
        unit_id: int,
        ready_methods: set[RetrievalMethod],
        failures: Sequence[MethodIndexFailure],
    ) -> None:
        failure_by_method = {
            method: failure.safe_error_code
            for failure in failures
            for method in failure.methods
        }
        absent = set(failure_by_method)
        if ready_methods & absent:
            raise RetrievalMethodUnavailable("method_index_manifest_conflict")
        now = datetime.now(timezone.utc).isoformat()
        entries = []
        for method in (
            RetrievalMethod.DENSE,
            RetrievalMethod.LEARNED_SPARSE,
            RetrievalMethod.BM25,
        ):
            if method in ready_methods:
                state, reason = "ready", None
            else:
                state = "absent"
                reason = failure_by_method.get(method, "not_written")
            entries.append((unit_id, method.value, state, reason, now))
        conn.execute("DELETE FROM retrieval_unit_method_indexes WHERE unit_id=?", (unit_id,))
        conn.executemany(
            "INSERT INTO retrieval_unit_method_indexes(unit_id, method, state, reason_code, updated_at) "
            "VALUES (?, ?, ?, ?, ?)",
            entries,
        )

    @staticmethod
    def _replace_bm25(conn: sqlite3.Connection, unit_id: int, token_ids: Sequence[int]) -> None:
        if not _table_exists(conn, "bm25_fts"):
            raise RetrievalMethodUnavailable("bm25_store_unavailable")
        shadow_terms = _shadow_terms(token_ids)
        if not shadow_terms:
            raise RetrievalMethodUnavailable("bm25_index_has_no_terms")
        SqliteRetrievalMethodStore._delete_bm25(conn, unit_id)
        conn.execute(
            "INSERT INTO bm25_unit_terms(unit_id, shadow_terms) VALUES (?, ?)",
            (unit_id, shadow_terms),
        )
        conn.execute(
            "INSERT INTO bm25_fts(rowid, shadow_terms) VALUES (?, ?)",
            (unit_id, shadow_terms),
        )

    @staticmethod
    def _delete_bm25(conn: sqlite3.Connection, unit_id: int) -> None:
        row = conn.execute(
            "SELECT shadow_terms FROM bm25_unit_terms WHERE unit_id=?", (unit_id,)
        ).fetchone()
        if row is not None and _table_exists(conn, "bm25_fts"):
            conn.execute(
                "INSERT INTO bm25_fts(bm25_fts, rowid, shadow_terms) VALUES ('delete', ?, ?)",
                (unit_id, str(row["shadow_terms"])),
            )
        conn.execute("DELETE FROM bm25_unit_terms WHERE unit_id=?", (unit_id,))

    def _candidates_from_unit_scores(
        self,
        *,
        source_filter: SourceFilter,
        data_version_id: str,
        query_index: int,
        method: RetrievalMethod,
        ordered_scores: Sequence[tuple[int, float]],
        higher_is_better: bool,
    ) -> tuple[RetrievalCandidate, ...]:
        if not ordered_scores:
            return ()
        unit_ids = [unit_id for unit_id, _ in ordered_scores]
        placeholders = ",".join("?" for _ in unit_ids)
        with self._catalog.connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM retrieval_units WHERE unit_id IN ({placeholders})",
                unit_ids,
            ).fetchall()
        refs_by_unit_id = {
            int(row["unit_id"]): _ref_from_row(row)
            for row in rows
            if row["source_type"] == source_filter.source_type.value
            and row["data_version_id"] == data_version_id
            and row["retrieval_status"] == RetrievalStatus.ACTIVE.value
            and row["index_state"] == UnitIndexState.READY.value
        }
        candidates: list[RetrievalCandidate] = []
        for rank, (unit_id, score) in enumerate(ordered_scores, start=1):
            ref = refs_by_unit_id.get(unit_id)
            if ref is None:
                continue
            candidates.append(
                RetrievalCandidate(
                    ref=ref,
                    method=method,
                    query_index=query_index,
                    rank=rank,
                    raw_score=score if higher_is_better else -score,
                )
            )
        return tuple(candidates)

    @staticmethod
    def _require_positive_limit(limit: int) -> None:
        if limit <= 0:
            raise ValueError("limit must be greater than zero")


def _bounded_sparse_weights(weights: Mapping[int, float], maximum: int = 256) -> dict[int, float]:
    normalised = normalise_sparse_weights(weights)
    ranked = sorted(normalised.items(), key=lambda item: (-item[1], item[0]))[:maximum]
    return dict(ranked)


def _method_index_failure(
    methods: Sequence[RetrievalMethod],
    exc: Exception,
    *,
    fallback_stage: str,
    fallback_code: str,
) -> MethodIndexFailure:
    stage = str(getattr(exc, "stage", None) or fallback_stage)
    safe_code = str(
        getattr(exc, "safe_error_code", None)
        or f"{fallback_code}:{type(exc).__name__}"
    )
    return MethodIndexFailure(
        methods=frozenset(methods),
        stage=stage,
        safe_error_code=safe_code,
    )


_SAFE_INDEX_DIAGNOSTIC_RE = re.compile(r"[A-Za-z0-9_.:-]{1,160}")


def _bounded_token_ids(token_ids: Sequence[int], maximum: int = 256) -> tuple[int, ...]:
    return tuple(int(token_id) for token_id in token_ids[:maximum] if int(token_id) >= 0)


def _shadow_terms(token_ids: Sequence[int]) -> str:
    return " ".join(_token_term(token_id) for token_id in token_ids if int(token_id) >= 0)


def _token_term(token_id: int) -> str:
    return f"t{int(token_id)}"


def _unit_filter_sql(
    source_filter: SourceFilter,
    *,
    data_version_id: str,
    alias: str,
) -> tuple[list[str], list[str]]:
    clauses = [
        f"{alias}.data_version_id=?",
        f"{alias}.source_type=?",
        f"{alias}.retrieval_status=?",
        f"{alias}.index_state=?",
    ]
    params = [
        data_version_id,
        source_filter.source_type.value,
        RetrievalStatus.ACTIVE.value,
        UnitIndexState.READY.value,
    ]
    for key, value in source_filter.scope:
        if key == CURRENT_SESSION_TURN_CUTOFF_SCOPE_KEY:
            clauses.append(
                f"CAST(json_extract({alias}.scope_json, ?) AS INTEGER) <= CAST(? AS INTEGER)"
            )
            params.extend((f"$.{CURRENT_SESSION_TURN_INDEX_SCOPE_KEY}", value))
        else:
            clauses.append(f"json_extract({alias}.scope_json, ?) = ?")
            params.extend((f"$.{key}", value))
    return clauses, params


def _capability_available(catalog: SqliteRetrievalCatalog, name: str) -> bool:
    capability = catalog.capability(name)
    return capability is not None and capability[0]


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE name=? AND type IN ('table', 'virtual table')", (table,)
    ).fetchone()
    return row is not None


def _batches(values: Sequence[int], size: int) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(values[index : index + size]) for index in range(0, len(values), size))


def _ref_from_row(row: sqlite3.Row):
    from ..contracts import SourceUnitRef

    return SourceUnitRef(
        source_type=SourceType(row["source_type"]),
        source_unit_id=str(row["source_unit_id"]),
        source_revision=str(row["source_revision"]),
        indexed_content_hash=str(row["indexed_content_hash"]),
    )


__all__ = [
    "BM25Retrieval",
    "BgeM3EncodedText",
    "BgeM3Encoder",
    "DenseRetrieval",
    "IndexedMethods",
    "LearnedSparseRetrieval",
    "LiteralBooleanFallback",
    "MethodIndexManifestEntry",
    "RetrievalMethodUnavailable",
    "SqliteRetrievalIndexWriter",
    "SqliteRetrievalMethodStore",
    "UnitMethodIndexHealth",
]
