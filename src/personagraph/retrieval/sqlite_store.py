"""统一 Retrieval SQLite catalog 的当前事务与 schema owner。

Catalog 保存 generation、Unit 指针/范围/哈希、索引状态、能力探测和同步回执；Dense、learned
sparse 与 BM25 的派生表示表也在同一 schema 中声明，由 ``indexing.methods`` 负责具体读写。
这里不保存权威 Source 正文，私有整数 ``unit_id`` 只用于同一派生目录内连接这些表示。

本文件暂留包根是一个明确的结构债，而不是建议长期模仿的布局：Indexing、Lifecycle、Sources
和 Project-document 事务都依赖同一个连接与原子提交边界，直接把它塞进 ``indexing`` 会错误暗示
索引算法拥有 generation/receipt 生命周期。后续应按 schema/capability 与 catalog CRUD 拆入专门
storage 子包，但必须一次迁移 transaction caller，并保持表名、schema version、CAS、WAL、receipt
幂等和 Project 共库事务不变。在真实 API 与 Bench 基线建立前不做机械拆文件。
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import StrEnum
import json
from pathlib import Path
import sqlite3
from collections.abc import Sequence
from typing import Iterator

from ..workspace.storage.context import current as current_project_documents
from .contracts import (
    CURRENT_SESSION_TURN_CUTOFF_SCOPE_KEY,
    CURRENT_SESSION_TURN_INDEX_SCOPE_KEY,
    RetrievalStatus,
    RetrievalUnit,
    SourceFilter,
    SourceType,
    SourceUnitRef,
)


SCHEMA_VERSION = 4


class RetrievalDataVersionRole(StrEnum):
    ACTIVE = "active"
    STAGING = "staging"
    PREVIOUS = "previous"


class RetrievalDataVersionState(StrEnum):
    BUILDING = "building"
    READY = "ready"
    FAILED = "failed"


class UnitIndexState(StrEnum):
    PENDING = "pending"
    READY = "ready"
    FAILED = "failed"


class SyncReceiptStatus(StrEnum):
    PROCESSING = "processing"
    APPLIED = "applied"
    RETRYABLE_FAILED = "retryable_failed"
    TERMINAL_FAILED = "terminal_failed"


@dataclass(frozen=True, slots=True)
class RetrievalDataVersion:
    id: str
    fingerprint: str
    role: RetrievalDataVersionRole
    state: RetrievalDataVersionState
    created_at: str
    activated_at: str | None


@dataclass(frozen=True, slots=True)
class StoredUnit:
    unit_id: int
    unit: RetrievalUnit
    index_state: UnitIndexState
    created_at: str
    updated_at: str


@dataclass(frozen=True, slots=True)
class SyncReceipt:
    event_id: str
    status: SyncReceiptStatus
    attempts: int
    reason_code: str | None
    updated_at: str


class RetrievalCatalogError(RuntimeError):
    pass


class RetrievalProjectContextRequired(RetrievalCatalogError):
    """默认目录未绑定任何项目所有的检索数据库。"""


class SqliteRetrievalCatalog:
    """同步、generation 管理和方法存储共享的单一目录 API。

    默认绑定当前 Project documents 数据库以参与同一事务；测试、迁移和 current-session 派生库
    可以显式提供路径。每次连接统一启用 foreign keys、WAL、busy timeout 与可用的 sqlite-vec，
    调用方不应自行复制一套连接初始化或绕过本 Catalog 修改生命周期状态。
    """

    def __init__(self, db_path: Path | str | None = None) -> None:
        self._project_documents = None
        if db_path is not None:
            self._db_path = Path(db_path)
            return

        self._project_documents = current_project_documents()
        if self._project_documents is None:
            raise RetrievalProjectContextRequired(
                "default file retrieval requires a bound project documents "
                "database; explicitly pass db_path only for a migration/test"
            )
        self._db_path = self._project_documents.db_path

    @property
    def db_path(self) -> Path:
        return self._db_path

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA busy_timeout = 5000")
        _try_load_sqlite_vec(conn)
        try:
            yield conn
        except Exception:
            conn.rollback()
            raise
        else:
            conn.commit()
        finally:
            conn.close()

    def initialize(self) -> None:
        if self._project_documents is not None:
            self._project_documents.initialize()
        with self.connect() as conn:
            if _is_project_documents_schema(conn):
                _require_project_retrieval_tables(conn)
                return
            current_version = int(conn.execute("PRAGMA user_version").fetchone()[0])
            if current_version > SCHEMA_VERSION:
                raise RetrievalCatalogError(
                    f"unsupported future retrieval schema version: {current_version}"
                )
            if current_version == 0:
                conn.executescript(_SCHEMA_SQL)
                conn.execute("PRAGMA user_version = 1")
                current_version = 1
            while current_version < SCHEMA_VERSION:
                migration = _MIGRATIONS.get(current_version + 1)
                if migration is None:
                    raise RetrievalCatalogError(
                        f"missing retrieval schema migration from version {current_version}"
                    )
                migration(conn)
                current_version += 1
                conn.execute(f"PRAGMA user_version = {current_version}")

    def active_data_version(self) -> RetrievalDataVersion | None:
        """返回唯一已发布派生数据版本（若存在）。"""

        self.initialize()
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM retrieval_data_versions WHERE role=?",
                (RetrievalDataVersionRole.ACTIVE.value,),
            ).fetchone()
        return self._data_version_from_row(row) if row else None

    def active_data_version_in_transaction(
        self,
        conn: sqlite3.Connection,
    ) -> RetrievalDataVersion | None:
        """在调用方持有的写事务中读取活动 generation，不执行 DDL 或提交。"""

        _require_active_transaction(conn)
        row = conn.execute(
            "SELECT * FROM retrieval_data_versions WHERE role=?",
            (RetrievalDataVersionRole.ACTIVE.value,),
        ).fetchone()
        return self._data_version_from_row(row) if row else None

    def active_retrieval_data_version_id(self) -> str | None:
        """为一次逻辑读取捕获活动且就绪的派生数据身份。"""

        active = self.active_data_version()
        if active is None or active.state is not RetrievalDataVersionState.READY:
            return None
        return active.id

    def active_unit_ids(
        self,
        *,
        data_version_id: str,
        source_filter: SourceFilter,
    ) -> tuple[int, ...]:
        """在任何 Top-K 前返回封闭且受信的候选全集。

        过滤键来自由受信策略编译的 ``SourceFilter``，而非模型提供的 SQL 片段。Unit 可以
        携带比请求边界更多的作用域元数据；每个请求键都必须匹配，因此更窄的受信边界是
        安全的。
        """

        source_filter = _catalog_query_filter(source_filter)
        self.initialize()
        clauses = [
            "data_version_id=?",
            "source_type=?",
            "retrieval_status=?",
            "index_state=?",
        ]
        params: list[str] = [
            data_version_id,
            source_filter.source_type.value,
            RetrievalStatus.ACTIVE.value,
            UnitIndexState.READY.value,
        ]
        for key, value in source_filter.scope:
            clause, path = _trusted_scope_sql_clause(key, column="scope_json")
            clauses.append(clause)
            params.extend([path, value])
        sql = "SELECT unit_id FROM retrieval_units WHERE " + " AND ".join(clauses) + " ORDER BY unit_id"
        with self.connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return tuple(int(row["unit_id"]) for row in rows)

    def active_ready_source_binding_keys(
        self,
        *,
        data_version_id: str,
        source_filter: SourceFilter,
        keys: Sequence[tuple[str, str, str]],
    ) -> frozenset[tuple[str, str, str]]:
        """返回被可搜索目录行覆盖的精确当前来源绑定。

        内容哈希属于身份的一部分。若只匹配来源 Unit 和 revision，由错误内容构建的目录行
        也会看似已覆盖。此窄桥绝不能回退到全目录扫描：调用方已经知道自己需要证明的小型
        权威集合。
        """

        if not isinstance(data_version_id, str) or not data_version_id.strip():
            raise ValueError("data_version_id must be a non-empty string")
        if not isinstance(source_filter, SourceFilter):
            raise ValueError("source_filter must be a SourceFilter")
        source_filter = _catalog_query_filter(source_filter)
        normalized: list[tuple[str, str, str]] = []
        seen: set[tuple[str, str, str]] = set()
        for key in keys:
            if not isinstance(key, tuple) or len(key) != 3:
                raise ValueError(
                    "keys must contain (source_unit_id, source_revision, "
                    "indexed_content_hash) tuples"
                )
            source_unit_id, source_revision, indexed_content_hash = key
            if (
                not isinstance(source_unit_id, str)
                or not source_unit_id.strip()
                or not isinstance(source_revision, str)
                or not source_revision.strip()
                or not isinstance(indexed_content_hash, str)
                or not indexed_content_hash.strip()
            ):
                raise ValueError("source binding keys must contain non-empty strings")
            normalized_key = (source_unit_id, source_revision, indexed_content_hash)
            if normalized_key not in seen:
                seen.add(normalized_key)
                normalized.append(normalized_key)
        if not normalized:
            return frozenset()

        self.initialize()
        covered: set[tuple[str, str, str]] = set()
        # 将 VALUES CTE 控制在 SQLite 常见的 999 参数上限以内：每项绑定三个身份参数、
        # 四个目录过滤参数，以及每个受信结构作用域键两个参数。
        batch_size = max(1, (900 - 4 - (2 * len(source_filter.scope))) // 3)
        with self.connect() as conn:
            for start in range(0, len(normalized), batch_size):
                batch = normalized[start : start + batch_size]
                values_sql = ", ".join("(?, ?, ?)" for _ in batch)
                params: list[str] = []
                for source_unit_id, source_revision, indexed_content_hash in batch:
                    params.extend((source_unit_id, source_revision, indexed_content_hash))
                params.extend(
                    (
                        data_version_id,
                        source_filter.source_type.value,
                        RetrievalStatus.ACTIVE.value,
                        UnitIndexState.READY.value,
                    )
                )
                scope_clauses: list[str] = []
                for key, value in source_filter.scope:
                    clause, path = _trusted_scope_sql_clause(
                        key,
                        column="unit.scope_json",
                    )
                    scope_clauses.append(clause)
                    params.extend((path, value))
                rows = conn.execute(
                    "WITH requested(source_unit_id, source_revision, indexed_content_hash) "
                    "AS (VALUES "
                    + values_sql
                    + ") "
                    "SELECT unit.source_unit_id, unit.source_revision, "
                    "unit.indexed_content_hash "
                    "FROM retrieval_units AS unit "
                    "JOIN requested ON requested.source_unit_id=unit.source_unit_id "
                    "AND requested.source_revision=unit.source_revision "
                    "AND requested.indexed_content_hash=unit.indexed_content_hash "
                    "WHERE unit.data_version_id=? AND unit.source_type=? "
                    "AND unit.retrieval_status=? AND unit.index_state=?"
                    + (" AND " + " AND ".join(scope_clauses) if scope_clauses else ""),
                    params,
                ).fetchall()
                covered.update(
                    (
                        str(row["source_unit_id"]),
                        str(row["source_revision"]),
                        str(row["indexed_content_hash"]),
                    )
                    for row in rows
                )
        return frozenset(covered)

    def active_ready_distinct_scope_value_count(
        self,
        *,
        data_version_id: str,
        source_filter: SourceFilter,
        scope_key: str,
    ) -> int:
        """Count covered source records by one trusted indexed scope dimension."""

        if scope_key != CURRENT_SESSION_TURN_INDEX_SCOPE_KEY:
            raise ValueError("unsupported distinct retrieval scope key")
        selected = _catalog_query_filter(source_filter)
        clauses = [
            "data_version_id=?",
            "source_type=?",
            "retrieval_status=?",
            "index_state=?",
        ]
        params: list[str] = [
            data_version_id,
            selected.source_type.value,
            RetrievalStatus.ACTIVE.value,
            UnitIndexState.READY.value,
        ]
        for key, value in selected.scope:
            clause, path = _trusted_scope_sql_clause(key, column="scope_json")
            clauses.append(clause)
            params.extend((path, value))
        self.initialize()
        with self.connect() as conn:
            row = conn.execute(
                "SELECT COUNT(DISTINCT json_extract(scope_json, ?)) AS count "
                "FROM retrieval_units WHERE " + " AND ".join(clauses),
                (f"$.{scope_key}", *params),
            ).fetchone()
        return int(row["count"] if row is not None else 0)

    def capability(self, name: str) -> tuple[bool, str | None] | None:
        self.initialize()
        with self.connect() as conn:
            row = conn.execute(
                "SELECT available, detail FROM retrieval_capabilities WHERE name=?", (name,)
            ).fetchone()
        if row is None:
            return None
        return bool(row["available"]), row["detail"]

    def refresh_method_capabilities(self) -> None:
        """依赖变化后重新探测可选 FTS5 和 sqlite-vec。"""

        self.initialize()
        with self.connect() as conn:
            _ensure_method_capabilities(conn)

    def create_data_version(
        self,
        *,
        version_id: str,
        fingerprint: str,
        role: RetrievalDataVersionRole = RetrievalDataVersionRole.STAGING,
        state: RetrievalDataVersionState = RetrievalDataVersionState.BUILDING,
    ) -> RetrievalDataVersion:
        self.initialize()
        now = _now()
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO retrieval_data_versions "
                "(id, fingerprint, role, state, created_at, activated_at) VALUES (?, ?, ?, ?, ?, NULL)",
                (version_id, fingerprint, role.value, state.value, now),
            )
            return self._data_version_from_row(
                conn.execute("SELECT * FROM retrieval_data_versions WHERE id=?", (version_id,)).fetchone()
            )

    def get_data_version(self, version_id: str) -> RetrievalDataVersion | None:
        self.initialize()
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM retrieval_data_versions WHERE id=?", (version_id,)).fetchone()
        return self._data_version_from_row(row) if row else None

    def get_data_version_in_transaction(
        self,
        conn: sqlite3.Connection,
        version_id: str,
    ) -> RetrievalDataVersion | None:
        """在调用方持有的写事务中读取精确 generation，不打开第二个连接。"""

        _require_active_transaction(conn)
        row = conn.execute(
            "SELECT * FROM retrieval_data_versions WHERE id=?",
            (version_id,),
        ).fetchone()
        return self._data_version_from_row(row) if row else None

    def get_data_version_by_fingerprint(
        self,
        fingerprint: str,
    ) -> RetrievalDataVersion | None:
        if not isinstance(fingerprint, str) or not fingerprint.strip():
            raise ValueError("retrieval data version fingerprint must not be empty")
        self.initialize()
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM retrieval_data_versions WHERE fingerprint=?",
                (fingerprint,),
            ).fetchone()
        return self._data_version_from_row(row) if row else None

    def list_data_versions(self) -> tuple[RetrievalDataVersion, ...]:
        """按确定性发布顺序列出供运维使用的 generation。

        此接口刻意只返回元数据。它适合诊断和回滚选择，但不表示 PREVIOUS generation
        仍足够完整、可以发布。
        """

        self.initialize()
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM retrieval_data_versions "
                "ORDER BY CASE role WHEN 'active' THEN 0 WHEN 'staging' THEN 1 ELSE 2 END, "
                "COALESCE(activated_at, created_at) DESC, id"
            ).fetchall()
        return tuple(self._data_version_from_row(row) for row in rows)

    def data_version_index_state_counts(self, version_id: str) -> dict[UnitIndexState, int]:
        """返回一个 generation 的指针级就绪计数。"""

        if not isinstance(version_id, str) or not version_id.strip():
            raise ValueError("version_id must not be empty")
        self.initialize()
        with self.connect() as conn:
            version = conn.execute(
                "SELECT 1 FROM retrieval_data_versions WHERE id=?",
                (version_id,),
            ).fetchone()
            if version is None:
                raise RetrievalCatalogError(
                    f"unknown retrieval data version: {version_id}"
                )
            rows = conn.execute(
                "SELECT index_state, COUNT(*) AS count FROM retrieval_units "
                "WHERE data_version_id=? GROUP BY index_state",
                (version_id,),
            ).fetchall()
        counts = {state: 0 for state in UnitIndexState}
        for row in rows:
            counts[UnitIndexState(str(row["index_state"]))] = int(row["count"])
        return counts

    def restore_previous_data_version_in_transaction(
        self,
        conn: sqlite3.Connection,
        version_id: str,
        *,
        expected_active_version_id: str | None,
        require_active_match: bool = True,
    ) -> RetrievalDataVersion:
        """在调用方持有的 Document 写围栏内恢复精确 PREVIOUS。

        此低层转换只重复结构门禁；Source manifest 与逐方法表示必须由调用方在同一事务
        中先行证明。空 generation 在任何情况下都不能成为 ACTIVE。
        """

        if not isinstance(version_id, str) or not version_id.strip():
            raise ValueError("version_id must not be empty")
        if not isinstance(require_active_match, bool):
            raise TypeError("require_active_match must be a bool")
        if expected_active_version_id is not None and (
            not isinstance(expected_active_version_id, str)
            or not expected_active_version_id.strip()
        ):
            raise ValueError("expected_active_version_id must not be empty")
        _require_active_transaction(conn)
        active_row = conn.execute(
            "SELECT id FROM retrieval_data_versions WHERE role=?",
            (RetrievalDataVersionRole.ACTIVE.value,),
        ).fetchone()
        actual_active_version_id = (
            str(active_row["id"]) if active_row is not None else None
        )
        if require_active_match and actual_active_version_id != expected_active_version_id:
            raise RetrievalCatalogError(
                "active retrieval data version changed after rollback diagnosis"
            )
        row = conn.execute(
            "SELECT * FROM retrieval_data_versions WHERE id=?",
            (version_id,),
        ).fetchone()
        if row is None:
            raise RetrievalCatalogError(
                f"unknown retrieval data version: {version_id}"
            )
        target = self._data_version_from_row(row)
        if (
            target.role is not RetrievalDataVersionRole.PREVIOUS
            or target.state is not RetrievalDataVersionState.READY
        ):
            raise RetrievalCatalogError(
                "only a ready previous retrieval data version may be restored"
            )
        counts = conn.execute(
            "SELECT COUNT(*) AS total, "
            "SUM(CASE WHEN index_state<>? THEN 1 ELSE 0 END) AS incomplete "
            "FROM retrieval_units WHERE data_version_id=?",
            (UnitIndexState.READY.value, version_id),
        ).fetchone()
        if int(counts["total"] or 0) == 0:
            raise RetrievalCatalogError(
                "empty previous retrieval data version requires index rebuild"
            )
        if int(counts["incomplete"] or 0):
            raise RetrievalCatalogError(
                "previous retrieval data version requires index rebuild"
            )
        now = _now()
        conn.execute(
            "UPDATE retrieval_data_versions SET role=? WHERE role=?",
            (
                RetrievalDataVersionRole.PREVIOUS.value,
                RetrievalDataVersionRole.ACTIVE.value,
            ),
        )
        conn.execute(
            "UPDATE retrieval_data_versions SET role=?, activated_at=? WHERE id=?",
            (RetrievalDataVersionRole.ACTIVE.value, now, version_id),
        )
        restored = conn.execute(
            "SELECT * FROM retrieval_data_versions WHERE id=?",
            (version_id,),
        ).fetchone()
        return self._data_version_from_row(restored)

    def reopen_previous_data_version_for_rebuild(
        self,
        version_id: str,
        *,
        expected_active_version_id: str | None,
    ) -> RetrievalDataVersion:
        """把不可服务的 PREVIOUS 重新打开为可恢复 STAGING rebuild。

        旧 ACTIVE 始终保持服务。所有仍存在的 Unit 先变回 pending，使回填必须重新写入
        三种派生表示；``activated_at`` 保留为本次重建周期的稳定 repair namespace。
        """

        if not isinstance(version_id, str) or not version_id.strip():
            raise ValueError("version_id must not be empty")
        if expected_active_version_id is not None and (
            not isinstance(expected_active_version_id, str)
            or not expected_active_version_id.strip()
        ):
            raise ValueError("expected_active_version_id must not be empty")
        self.initialize()
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            active = self.active_data_version_in_transaction(conn)
            active_id = active.id if active is not None else None
            if active_id != expected_active_version_id:
                raise RetrievalCatalogError(
                    "active retrieval data version changed before rebuild reopen"
                )
            staging = conn.execute(
                "SELECT id FROM retrieval_data_versions WHERE role=?",
                (RetrievalDataVersionRole.STAGING.value,),
            ).fetchone()
            if staging is not None:
                raise RetrievalCatalogError("a staging version already exists")
            row = conn.execute(
                "SELECT * FROM retrieval_data_versions WHERE id=?",
                (version_id,),
            ).fetchone()
            if row is None:
                raise RetrievalCatalogError(
                    f"unknown retrieval data version: {version_id}"
                )
            target = self._data_version_from_row(row)
            if (
                target.role is not RetrievalDataVersionRole.PREVIOUS
                or target.state is not RetrievalDataVersionState.READY
                or target.activated_at is None
            ):
                raise RetrievalCatalogError(
                    "only an activated ready previous generation may be rebuilt"
                )
            conn.execute(
                "UPDATE retrieval_data_versions SET role=?, state=? WHERE id=?",
                (
                    RetrievalDataVersionRole.STAGING.value,
                    RetrievalDataVersionState.BUILDING.value,
                    version_id,
                ),
            )
            conn.execute(
                "UPDATE retrieval_units SET index_state=?, updated_at=? "
                "WHERE data_version_id=?",
                (UnitIndexState.PENDING.value, _now(), version_id),
            )
            updated = conn.execute(
                "SELECT * FROM retrieval_data_versions WHERE id=?",
                (version_id,),
            ).fetchone()
        return self._data_version_from_row(updated)

    def ensure_staging_data_version(
        self,
        *,
        version_id: str,
        fingerprint: str,
    ) -> tuple[RetrievalDataVersion, bool]:
        """创建一个构建中 generation，或重放其精确持久身份。"""

        for name, value in (("version_id", version_id), ("fingerprint", fingerprint)):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must not be empty")
        self.initialize()
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            id_row = conn.execute(
                "SELECT * FROM retrieval_data_versions WHERE id=?",
                (version_id,),
            ).fetchone()
            if id_row is not None:
                existing = self._data_version_from_row(id_row)
                if existing.fingerprint != fingerprint:
                    raise RetrievalCatalogError("retrieval data version id collision")
                if existing.role is RetrievalDataVersionRole.PREVIOUS:
                    raise RetrievalCatalogError(
                        "a previous retrieval data version is not writable"
                    )
                return existing, True

            fingerprint_row = conn.execute(
                "SELECT * FROM retrieval_data_versions WHERE fingerprint=?",
                (fingerprint,),
            ).fetchone()
            if fingerprint_row is not None:
                raise RetrievalCatalogError("retrieval data version fingerprint collision")
            staging = conn.execute(
                "SELECT id FROM retrieval_data_versions WHERE role=?",
                (RetrievalDataVersionRole.STAGING.value,),
            ).fetchone()
            if staging is not None:
                raise RetrievalCatalogError("a staging version already exists")
            conn.execute(
                "INSERT INTO retrieval_data_versions "
                "(id, fingerprint, role, state, created_at, activated_at) "
                "VALUES (?, ?, ?, ?, ?, NULL)",
                (
                    version_id,
                    fingerprint,
                    RetrievalDataVersionRole.STAGING.value,
                    RetrievalDataVersionState.BUILDING.value,
                    _now(),
                ),
            )
            row = conn.execute(
                "SELECT * FROM retrieval_data_versions WHERE id=?",
                (version_id,),
            ).fetchone()
        return self._data_version_from_row(row), False

    def mark_data_version_ready(self, version_id: str) -> RetrievalDataVersion:
        return self._transition_staging_data_version(
            version_id,
            from_state=RetrievalDataVersionState.BUILDING,
            to_state=RetrievalDataVersionState.READY,
            to_role=RetrievalDataVersionRole.STAGING,
            allow_exact_replay=True,
        )

    def mark_data_version_ready_in_transaction(
        self,
        conn: sqlite3.Connection,
        version_id: str,
    ) -> RetrievalDataVersion:
        """在调用方持有的发布围栏内标记一个完整 STAGING generation。"""

        _require_active_transaction(conn)
        return self._transition_staging_data_version_in_transaction(
            conn,
            version_id,
            from_state=RetrievalDataVersionState.BUILDING,
            to_state=RetrievalDataVersionState.READY,
            to_role=RetrievalDataVersionRole.STAGING,
            allow_exact_replay=True,
        )

    def invalidate_ready_staging_data_version(
        self,
        version_id: str,
    ) -> RetrievalDataVersion:
        """来源失效后重新打开未发布的暂存 generation。

        调用方必须持有同时保护激活操作的来源权威发布围栏。此转换刻意不向
        ACTIVE/PREVIOUS generation 开放：它只撤回未发布的 READY 证明，以便在重新证明
        精确语料前应用持久 PURGE 和后续来源 UPSERT。
        """

        return self._transition_staging_data_version(
            version_id,
            from_state=RetrievalDataVersionState.READY,
            to_state=RetrievalDataVersionState.BUILDING,
            to_role=RetrievalDataVersionRole.STAGING,
            allow_exact_replay=True,
        )

    def invalidate_ready_staging_data_version_in_transaction(
        self,
        conn: sqlite3.Connection,
        version_id: str,
    ) -> RetrievalDataVersion:
        """在来源权威写围栏内撤销未发布 READY 证明。"""

        _require_active_transaction(conn)
        return self._transition_staging_data_version_in_transaction(
            conn,
            version_id,
            from_state=RetrievalDataVersionState.READY,
            to_state=RetrievalDataVersionState.BUILDING,
            to_role=RetrievalDataVersionRole.STAGING,
            allow_exact_replay=True,
        )

    def mark_data_version_failed(self, version_id: str) -> RetrievalDataVersion:
        return self._transition_staging_data_version(
            version_id,
            from_state=RetrievalDataVersionState.BUILDING,
            to_state=RetrievalDataVersionState.FAILED,
            to_role=RetrievalDataVersionRole.PREVIOUS,
            allow_exact_replay=True,
        )

    def reopen_failed_data_version_for_rebuild(
        self,
        version_id: str,
    ) -> RetrievalDataVersion:
        """Reopen the exact failed generation without discarding recoverable Units.

        A failed staging build is derived state, not an authority tombstone.  Keeping
        already-ready Units lets deterministic sync receipts replay safely while the
        failed Unit is retried.  The transition remains fenced by the single-STAGING
        constraint, so a concurrent recipe cannot be overwritten.
        """

        if not isinstance(version_id, str) or not version_id.strip():
            raise ValueError("version_id must not be empty")
        self.initialize()
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT * FROM retrieval_data_versions WHERE id=?",
                (version_id,),
            ).fetchone()
            if row is None:
                raise RetrievalCatalogError(
                    f"unknown retrieval data version: {version_id}"
                )
            existing = self._data_version_from_row(row)
            if (
                existing.role is RetrievalDataVersionRole.STAGING
                and existing.state is RetrievalDataVersionState.BUILDING
            ):
                return existing
            if (
                existing.role is not RetrievalDataVersionRole.PREVIOUS
                or existing.state is not RetrievalDataVersionState.FAILED
            ):
                raise RetrievalCatalogError(
                    "only a failed previous generation may be reopened"
                )
            competing = conn.execute(
                "SELECT id FROM retrieval_data_versions WHERE role=? AND id<>?",
                (RetrievalDataVersionRole.STAGING.value, version_id),
            ).fetchone()
            if competing is not None:
                raise RetrievalCatalogError("a staging version already exists")
            conn.execute(
                "UPDATE retrieval_data_versions SET role=?, state=? WHERE id=?",
                (
                    RetrievalDataVersionRole.STAGING.value,
                    RetrievalDataVersionState.BUILDING.value,
                    version_id,
                ),
            )
            reopened = conn.execute(
                "SELECT * FROM retrieval_data_versions WHERE id=?",
                (version_id,),
            ).fetchone()
        return self._data_version_from_row(reopened)

    def activate_data_version(self, version_id: str) -> RetrievalDataVersion:
        self.initialize()
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            return self.activate_data_version_in_transaction(conn, version_id)

    def activate_data_version_in_transaction(
        self,
        conn: sqlite3.Connection,
        version_id: str,
    ) -> RetrievalDataVersion:
        """在调用方持有的写事务中原子交换 ACTIVE generation。"""

        _require_active_transaction(conn)
        row = conn.execute(
            "SELECT * FROM retrieval_data_versions WHERE id=?",
            (version_id,),
        ).fetchone()
        if row is None:
            raise RetrievalCatalogError(f"unknown retrieval data version: {version_id}")
        existing = self._data_version_from_row(row)
        if (
            existing.role is RetrievalDataVersionRole.ACTIVE
            and existing.state is RetrievalDataVersionState.READY
        ):
            return existing
        if (
            existing.role is not RetrievalDataVersionRole.STAGING
            or existing.state is not RetrievalDataVersionState.READY
        ):
            raise RetrievalCatalogError("only a ready staging version may become active")
        now = _now()
        conn.execute(
            "UPDATE retrieval_data_versions SET role=? WHERE role=?",
            (
                RetrievalDataVersionRole.PREVIOUS.value,
                RetrievalDataVersionRole.ACTIVE.value,
            ),
        )
        conn.execute(
            "UPDATE retrieval_data_versions SET role=?, activated_at=? WHERE id=?",
            (RetrievalDataVersionRole.ACTIVE.value, now, version_id),
        )
        active_row = conn.execute(
            "SELECT * FROM retrieval_data_versions WHERE id=?",
            (version_id,),
        ).fetchone()
        if active_row is None:  # pragma: no cover - 同事务内刚完成精确主键更新
            raise RetrievalCatalogError(
                f"activated retrieval data version disappeared: {version_id}"
            )
        return self._data_version_from_row(active_row)

    def upsert_pending_unit(self, unit: RetrievalUnit) -> StoredUnit:
        """登记新的不可变内容绑定，但不使其可搜索。"""

        if (
            self._project_documents is not None
            and unit.ref.source_type is SourceType.DOCUMENT
            and unit.source_filter.as_mapping()
            != {"doc_id": _project_document_id_from_unit(unit.ref.source_unit_id)}
        ):
            raise RetrievalCatalogError(
                "project document Units require one doc_id-only catalog scope"
            )
        self.initialize()
        now = _now()
        scope_json = _scope_json(unit.source_filter)
        ref = unit.ref
        with self.connect() as conn:
            # generation 的角色可能在激活期间变化。应在与 Unit upsert 相同的写锁下校验，
            # 使 ACTIVE 目标无法在检查与提交之间变为 PREVIOUS。
            conn.execute("BEGIN IMMEDIATE")
            self._require_data_version(unit.retrieval_data_version, conn=conn)
            conn.execute(
                "INSERT INTO retrieval_units "
                "(data_version_id, source_type, source_unit_id, source_revision, indexed_content_hash, "
                "retrieval_status, index_state, scope_json, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(data_version_id, source_type, source_unit_id, source_revision, indexed_content_hash) "
                "DO UPDATE SET scope_json=excluded.scope_json, retrieval_status=excluded.retrieval_status, "
                "updated_at=excluded.updated_at",
                (
                    unit.retrieval_data_version,
                    ref.source_type.value,
                    ref.source_unit_id,
                    ref.source_revision,
                    ref.indexed_content_hash,
                    unit.retrieval_status.value,
                    UnitIndexState.PENDING.value,
                    scope_json,
                    now,
                    now,
                ),
            )
            row = self._find_unit_row(conn, ref, unit.retrieval_data_version)
        if row is None:  # 防御性检查：成功执行 upsert 后理论上不可能发生
            raise RetrievalCatalogError("unit upsert did not return a catalog record")
        return self._stored_unit_from_row(row)

    def mark_unit_index_ready(self, unit_id: int) -> None:
        self._set_index_state(unit_id, UnitIndexState.READY)

    def mark_unit_index_failed(self, unit_id: int) -> None:
        self._set_index_state(unit_id, UnitIndexState.FAILED)

    def get_unit(self, ref: SourceUnitRef, data_version_id: str) -> StoredUnit | None:
        self.initialize()
        with self.connect() as conn:
            row = self._find_unit_row(conn, ref, data_version_id)
        return self._stored_unit_from_row(row) if row else None

    def get_unit_in_transaction(
        self,
        conn: sqlite3.Connection,
        ref: SourceUnitRef,
        data_version_id: str,
    ) -> StoredUnit | None:
        """在调用方快照中读取一个精确 Unit，不打开第二个连接。"""

        _require_active_transaction(conn)
        row = self._find_unit_row(conn, ref, data_version_id)
        return self._stored_unit_from_row(row) if row else None

    def set_unit_retrieval_status(
        self,
        ref: SourceUnitRef,
        data_version_id: str,
        status: RetrievalStatus,
    ) -> StoredUnit | None:
        """改变可见性，并保护发布不作用于冻结 generation。

        将 Unit 移到 ``ACTIVE`` 属于发布操作，因此要求 generation 可写。对于 previous
        或 failed generation，将其移到 ``TRASHED`` 仍然合法，因为 PURGE 会在删除方法
        数据和指针前使用该破坏性转换。
        """

        self.initialize()
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = self._find_unit_row(conn, ref, data_version_id)
            if row is None:
                return None
            if status is RetrievalStatus.ACTIVE:
                self._require_data_version(data_version_id, conn=conn)
            conn.execute(
                "UPDATE retrieval_units SET retrieval_status=?, updated_at=? WHERE unit_id=?",
                (status.value, _now(), int(row["unit_id"])),
            )
            row = conn.execute("SELECT * FROM retrieval_units WHERE unit_id=?", (int(row["unit_id"]),)).fetchone()
        return self._stored_unit_from_row(row)

    def delete_unit(self, unit_id: int) -> None:
        self.initialize()
        with self.connect() as conn:
            conn.execute("DELETE FROM retrieval_units WHERE unit_id=?", (unit_id,))

    def mark_unit_index_pending(self, unit_id: int) -> None:
        """受控修复重写索引期间，将一个 Unit 移出搜索。"""

        self._set_index_state(unit_id, UnitIndexState.PENDING)

    def list_stored_units(self, data_version_id: str | None = None) -> tuple[StoredUnit, ...]:
        """读取仅含指针的 Unit 记录，供诊断或受控修复使用。"""

        self.initialize()
        with self.connect() as conn:
            return self._list_stored_units_in_connection(conn, data_version_id)

    def list_stored_units_for_source_filter(
        self,
        source_filter: SourceFilter,
        *,
        data_version_id: str | None = None,
    ) -> tuple[StoredUnit, ...]:
        """List exact derived pointers selected by a trusted lifecycle scope."""

        if not isinstance(source_filter, SourceFilter):
            raise TypeError("source_filter must be a SourceFilter")
        selected = _catalog_query_filter(source_filter)
        clauses = ["source_type=?"]
        params: list[str] = [selected.source_type.value]
        if data_version_id is not None:
            if not isinstance(data_version_id, str) or not data_version_id.strip():
                raise ValueError("data_version_id must not be empty")
            clauses.append("data_version_id=?")
            params.append(data_version_id)
        for key, value in selected.scope:
            clause, path = _trusted_scope_sql_clause(key, column="scope_json")
            clauses.append(clause)
            params.extend((path, value))
        self.initialize()
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM retrieval_units WHERE "
                + " AND ".join(clauses)
                + " ORDER BY unit_id",
                params,
            ).fetchall()
        return tuple(self._stored_unit_from_row(row) for row in rows)

    def delete_sync_receipts(self, event_ids: Sequence[str]) -> int:
        """Delete derived replay receipts whose owning source scope was purged."""

        normalized = tuple(dict.fromkeys(event_ids))
        if any(not isinstance(event_id, str) or not event_id.strip() for event_id in normalized):
            raise ValueError("event_ids must contain non-empty strings")
        if not normalized:
            return 0
        self.initialize()
        deleted = 0
        with self.connect() as conn:
            for start in range(0, len(normalized), 900):
                batch = normalized[start : start + 900]
                placeholders = ",".join("?" for _ in batch)
                deleted += conn.execute(
                    f"DELETE FROM retrieval_sync_receipts WHERE event_id IN ({placeholders})",
                    batch,
                ).rowcount
        return deleted

    def list_stored_units_in_transaction(
        self,
        conn: sqlite3.Connection,
        data_version_id: str | None = None,
    ) -> tuple[StoredUnit, ...]:
        """在调用方快照中列出 Unit，不初始化、提交或获取第二个锁。"""

        _require_active_transaction(conn)
        return self._list_stored_units_in_connection(conn, data_version_id)

    def active_units(self, data_version_id: str) -> list[StoredUnit]:
        self.initialize()
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM retrieval_units WHERE data_version_id=? "
                "AND retrieval_status=? AND index_state=? ORDER BY unit_id",
                (data_version_id, RetrievalStatus.ACTIVE.value, UnitIndexState.READY.value),
            ).fetchall()
        return [self._stored_unit_from_row(row) for row in rows]

    def start_receipt(self, event_id: str) -> SyncReceipt:
        self.initialize()
        now = _now()
        with self.connect() as conn:
            existing = conn.execute("SELECT * FROM retrieval_sync_receipts WHERE event_id=?", (event_id,)).fetchone()
            if existing is None:
                conn.execute(
                    "INSERT INTO retrieval_sync_receipts "
                    "(event_id, status, attempts, reason_code, updated_at) VALUES (?, ?, 1, NULL, ?)",
                    (event_id, SyncReceiptStatus.PROCESSING.value, now),
                )
            elif existing["status"] != SyncReceiptStatus.APPLIED.value:
                conn.execute(
                    "UPDATE retrieval_sync_receipts SET status=?, attempts=attempts+1, reason_code=NULL, updated_at=? "
                    "WHERE event_id=?",
                    (SyncReceiptStatus.PROCESSING.value, now, event_id),
                )
            row = conn.execute("SELECT * FROM retrieval_sync_receipts WHERE event_id=?", (event_id,)).fetchone()
        return self._receipt_from_row(row)

    def get_receipt(self, event_id: str) -> SyncReceipt | None:
        self.initialize()
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM retrieval_sync_receipts WHERE event_id=?", (event_id,)).fetchone()
        return self._receipt_from_row(row) if row else None

    def finish_receipt(
        self,
        event_id: str,
        *,
        status: SyncReceiptStatus,
        reason_code: str | None = None,
    ) -> SyncReceipt:
        if status is SyncReceiptStatus.PROCESSING:
            raise ValueError("finish_receipt cannot retain processing status")
        self.initialize()
        with self.connect() as conn:
            conn.execute(
                "UPDATE retrieval_sync_receipts SET status=?, reason_code=?, updated_at=? WHERE event_id=?",
                (status.value, reason_code, _now(), event_id),
            )
            row = conn.execute("SELECT * FROM retrieval_sync_receipts WHERE event_id=?", (event_id,)).fetchone()
        if row is None:
            raise RetrievalCatalogError(f"receipt does not exist: {event_id}")
        return self._receipt_from_row(row)

    def _transition_staging_data_version(
        self,
        version_id: str,
        *,
        from_state: RetrievalDataVersionState,
        to_state: RetrievalDataVersionState,
        to_role: RetrievalDataVersionRole,
        allow_exact_replay: bool,
    ) -> RetrievalDataVersion:
        self.initialize()
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            return self._transition_staging_data_version_in_transaction(
                conn,
                version_id,
                from_state=from_state,
                to_state=to_state,
                to_role=to_role,
                allow_exact_replay=allow_exact_replay,
            )

    def _transition_staging_data_version_in_transaction(
        self,
        conn: sqlite3.Connection,
        version_id: str,
        *,
        from_state: RetrievalDataVersionState,
        to_state: RetrievalDataVersionState,
        to_role: RetrievalDataVersionRole,
        allow_exact_replay: bool,
    ) -> RetrievalDataVersion:
        _require_active_transaction(conn)
        row = conn.execute(
            "SELECT * FROM retrieval_data_versions WHERE id=?",
            (version_id,),
        ).fetchone()
        if row is None:
            raise RetrievalCatalogError(f"unknown retrieval data version: {version_id}")
        existing = self._data_version_from_row(row)
        if (
            allow_exact_replay
            and existing.role is to_role
            and existing.state is to_state
        ):
            return existing
        if (
            existing.role is not RetrievalDataVersionRole.STAGING
            or existing.state is not from_state
        ):
            raise RetrievalCatalogError(
                f"data version transition requires staging {from_state.value}"
            )
        conn.execute(
            "UPDATE retrieval_data_versions SET role=?, state=? WHERE id=?",
            (to_role.value, to_state.value, version_id),
        )
        updated_row = conn.execute(
            "SELECT * FROM retrieval_data_versions WHERE id=?",
            (version_id,),
        ).fetchone()
        return self._data_version_from_row(updated_row)

    def _require_data_version(
        self,
        version_id: str,
        *,
        conn: sqlite3.Connection | None = None,
    ) -> RetrievalDataVersion:
        """要求 generation 可以合法接收发布写入。

        变更调用方传入已经由 ``BEGIN IMMEDIATE`` 加锁的连接。为兼容诊断和测试，仍保留
        可选自有连接路径，但绝不能把它用作写入前检查边界。
        """

        if conn is None:
            self.initialize()
            with self.connect() as owned_conn:
                return self._require_data_version(version_id, conn=owned_conn)
        row = conn.execute(
            "SELECT * FROM retrieval_data_versions WHERE id=?",
            (version_id,),
        ).fetchone()
        if row is None:
            raise RetrievalCatalogError(f"unknown retrieval data version: {version_id}")
        version = self._data_version_from_row(row)
        writable = (
            version.role is RetrievalDataVersionRole.ACTIVE
            and version.state is RetrievalDataVersionState.READY
        ) or (
            version.role is RetrievalDataVersionRole.STAGING
            and version.state is RetrievalDataVersionState.BUILDING
        )
        if not writable:
            raise RetrievalCatalogError(
                f"retrieval data version is not writable: {version_id}"
            )
        return version

    def require_writable_data_version_in_transaction(
        self,
        conn: sqlite3.Connection,
        version_id: str,
    ) -> RetrievalDataVersion:
        """在调用方写事务内重新证明 generation 仍可接收来源提交。"""

        _require_active_transaction(conn)
        return self._require_data_version(version_id, conn=conn)

    def _list_stored_units_in_connection(
        self,
        conn: sqlite3.Connection,
        data_version_id: str | None,
    ) -> tuple[StoredUnit, ...]:
        if data_version_id is None:
            rows = conn.execute(
                "SELECT * FROM retrieval_units ORDER BY unit_id"
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM retrieval_units WHERE data_version_id=? ORDER BY unit_id",
                (data_version_id,),
            ).fetchall()
        return tuple(self._stored_unit_from_row(row) for row in rows)

    @staticmethod
    def _find_unit_row(conn: sqlite3.Connection, ref: SourceUnitRef, data_version_id: str):
        return conn.execute(
            "SELECT * FROM retrieval_units WHERE data_version_id=? AND source_type=? "
            "AND source_unit_id=? AND source_revision=? AND indexed_content_hash=?",
            (
                data_version_id,
                ref.source_type.value,
                ref.source_unit_id,
                ref.source_revision,
                ref.indexed_content_hash,
            ),
        ).fetchone()

    @staticmethod
    def _data_version_from_row(row: sqlite3.Row) -> RetrievalDataVersion:
        return RetrievalDataVersion(
            id=str(row["id"]),
            fingerprint=str(row["fingerprint"]),
            role=RetrievalDataVersionRole(row["role"]),
            state=RetrievalDataVersionState(row["state"]),
            created_at=str(row["created_at"]),
            activated_at=row["activated_at"],
        )

    @staticmethod
    def _stored_unit_from_row(row: sqlite3.Row) -> StoredUnit:
        source_type = SourceType(row["source_type"])
        source_filter = SourceFilter.from_mapping(source_type, json.loads(row["scope_json"]))
        unit = RetrievalUnit(
            ref=SourceUnitRef(
                source_type=source_type,
                source_unit_id=str(row["source_unit_id"]),
                source_revision=str(row["source_revision"]),
                indexed_content_hash=str(row["indexed_content_hash"]),
            ),
            retrieval_data_version=str(row["data_version_id"]),
            retrieval_status=RetrievalStatus(row["retrieval_status"]),
            source_filter=source_filter,
        )
        return StoredUnit(
            unit_id=int(row["unit_id"]),
            unit=unit,
            index_state=UnitIndexState(row["index_state"]),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
        )

    @staticmethod
    def _receipt_from_row(row: sqlite3.Row) -> SyncReceipt:
        return SyncReceipt(
            event_id=str(row["event_id"]),
            status=SyncReceiptStatus(row["status"]),
            attempts=int(row["attempts"]),
            reason_code=row["reason_code"],
            updated_at=str(row["updated_at"]),
        )

    def _set_index_state(self, unit_id: int, state: UnitIndexState) -> None:
        self.initialize()
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT data_version_id FROM retrieval_units WHERE unit_id=?",
                (unit_id,),
            ).fetchone()
            if row is None:
                raise RetrievalCatalogError(f"unknown retrieval unit id: {unit_id}")
            self._require_data_version(str(row["data_version_id"]), conn=conn)
            updated = conn.execute(
                "UPDATE retrieval_units SET index_state=?, updated_at=? WHERE unit_id=?",
                (state.value, _now(), unit_id),
            ).rowcount
        if not updated:
            raise RetrievalCatalogError(f"unknown retrieval unit id: {unit_id}")


def _scope_json(source_filter: SourceFilter | None) -> str:
    if source_filter is None:
        return "{}"
    return json.dumps(source_filter.as_mapping(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _project_document_id_from_unit(source_unit_id: str) -> str:
    marker = "document-v3:"
    if not source_unit_id.startswith(marker):
        raise RetrievalCatalogError(
            "project document Units require a document-v3 identity"
        )
    document_id, separator, producer_chunk_id = source_unit_id[len(marker):].partition(":")
    if not separator or not document_id or not producer_chunk_id:
        raise RetrievalCatalogError("project document Unit identity is malformed")
    return document_id


def _catalog_query_filter(source_filter: SourceFilter) -> SourceFilter:
    """从项目索引查询中移除 Session 授权元数据。"""

    if source_filter.source_type is not SourceType.DOCUMENT:
        return source_filter
    scope = source_filter.as_mapping()
    scope.pop("session_id", None)
    return SourceFilter.from_mapping(SourceType.DOCUMENT, scope)


def _trusted_scope_sql_clause(key: str, *, column: str) -> tuple[str, str]:
    """编译封闭的纯 Host 截止条件，不接受 SQL 片段。"""

    if key == CURRENT_SESSION_TURN_CUTOFF_SCOPE_KEY:
        return (
            f"CAST(json_extract({column}, ?) AS INTEGER) <= CAST(? AS INTEGER)",
            f"$.{CURRENT_SESSION_TURN_INDEX_SCOPE_KEY}",
        )
    return f"json_extract({column}, ?) = ?", f"$.{key}"


def _is_project_documents_schema(conn: sqlite3.Connection) -> bool:
    """在查询旧版 PRAGMA 版本前识别统一 schema。"""

    metadata_table = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='schema_meta'"
    ).fetchone()
    if metadata_table is None:
        return False
    try:
        rows = conn.execute("SELECT schema_name FROM schema_meta").fetchall()
    except sqlite3.Error as exc:
        raise RetrievalCatalogError("invalid schema_meta table in retrieval database") from exc
    if any(str(row["schema_name"]) == "project_documents" for row in rows):
        return True
    raise RetrievalCatalogError("retrieval database has an unsupported schema_meta identity")


def _require_project_retrieval_tables(conn: sqlite3.Connection) -> None:
    """若统一 schema 所有者尚未完成初始化，则明确失败。"""

    required = {
        "retrieval_data_versions",
        "retrieval_units",
        "retrieval_sync_receipts",
        "retrieval_capabilities",
        "retrieval_unit_method_indexes",
        "learned_sparse_postings",
        "bm25_unit_terms",
    }
    actual = {
        str(row["name"])
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        ).fetchall()
    }
    missing = sorted(required - actual)
    if missing:
        raise RetrievalCatalogError(
            "project documents database is missing retrieval tables: " + ", ".join(missing)
        )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _require_active_transaction(conn: sqlite3.Connection) -> None:
    """拒绝把 transaction-scoped API 降级成无围栏的普通连接操作。"""

    if not isinstance(conn, sqlite3.Connection):
        raise TypeError("conn must be a sqlite3.Connection")
    if not conn.in_transaction:
        raise RuntimeError("operation requires an active caller-owned transaction")


def _try_load_sqlite_vec(conn: sqlite3.Connection) -> tuple[bool, str | None]:
    """为此连接加载 sqlite-vec，同时不使目录因失败而不可用。

    SQLite 扩展按连接登记。目录在最小 SQLite 构建上仍有用途，因此扩展缺失会由 schema
    能力探针记录，而不会被视为致命启动错误。
    """

    try:
        import sqlite_vec

        conn.enable_load_extension(True)
        try:
            sqlite_vec.load(conn)
        finally:
            conn.enable_load_extension(False)
    except Exception as exc:  # 可选原生扩展边界
        return False, f"{type(exc).__name__}:{exc}"
    return True, None


def _set_capability(conn: sqlite3.Connection, name: str, available: bool, detail: str | None) -> None:
    conn.execute(
        "INSERT INTO retrieval_capabilities (name, available, detail, checked_at) VALUES (?, ?, ?, ?) "
        "ON CONFLICT(name) DO UPDATE SET available=excluded.available, detail=excluded.detail, "
        "checked_at=excluded.checked_at",
        (name, int(available), detail, _now()),
    )


def _migration_v2_method_indexes(conn: sqlite3.Connection) -> None:
    """创建派生方法表，并记录可选 SQLite 能力。"""

    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS retrieval_capabilities (
            name TEXT PRIMARY KEY,
            available INTEGER NOT NULL CHECK(available IN (0, 1)),
            detail TEXT,
            checked_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS learned_sparse_postings (
            unit_id INTEGER NOT NULL,
            token_id INTEGER NOT NULL,
            weight REAL NOT NULL CHECK(weight > 0),
            PRIMARY KEY(unit_id, token_id),
            FOREIGN KEY(unit_id) REFERENCES retrieval_units(unit_id) ON DELETE CASCADE
        ) WITHOUT ROWID;
        CREATE INDEX IF NOT EXISTS idx_learned_sparse_postings_token
            ON learned_sparse_postings(token_id, unit_id);

        CREATE TABLE IF NOT EXISTS bm25_unit_terms (
            unit_id INTEGER PRIMARY KEY,
            shadow_terms TEXT NOT NULL,
            FOREIGN KEY(unit_id) REFERENCES retrieval_units(unit_id) ON DELETE CASCADE
        );
        """
    )

    _ensure_method_capabilities(conn)


def _ensure_method_capabilities(conn: sqlite3.Connection) -> None:
    try:
        conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS bm25_fts USING fts5(shadow_terms, content='')")
    except sqlite3.OperationalError as exc:
        _set_capability(conn, "fts5", False, f"{type(exc).__name__}:{exc}")
    else:
        _set_capability(conn, "fts5", True, None)

    vec_available, vec_detail = _try_load_sqlite_vec(conn)
    if not vec_available:
        _set_capability(conn, "sqlite_vec", False, vec_detail)
        return
    try:
        conn.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS dense_vectors USING vec0("
            "unit_id INTEGER PRIMARY KEY, embedding float[1024], scope_key TEXT)"
        )
    except sqlite3.OperationalError as exc:
        _set_capability(conn, "sqlite_vec", False, f"{type(exc).__name__}:{exc}")
    else:
        _set_capability(conn, "sqlite_vec", True, None)


def _migration_v3_method_index_manifest(conn: sqlite3.Connection) -> None:
    """记录每个 Unit 实际写入了哪些派生检索方法。"""

    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS retrieval_unit_method_indexes (
            unit_id INTEGER NOT NULL,
            method TEXT NOT NULL CHECK(method IN ('dense', 'learned_sparse', 'bm25')),
            state TEXT NOT NULL CHECK(state IN ('ready', 'absent')),
            reason_code TEXT,
            updated_at TEXT NOT NULL,
            PRIMARY KEY(unit_id, method),
            FOREIGN KEY(unit_id) REFERENCES retrieval_units(unit_id) ON DELETE CASCADE
        ) WITHOUT ROWID;
        CREATE INDEX IF NOT EXISTS idx_retrieval_unit_method_indexes_state
            ON retrieval_unit_method_indexes(method, state, unit_id);
        """
    )


def _migration_v4_source_coverage_lookup(conn: sqlite3.Connection) -> None:
    """支持精确来源绑定覆盖检查，无需扫描目录。"""

    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_retrieval_units_source_coverage "
        "ON retrieval_units("
        "data_version_id, source_type, source_unit_id, source_revision, "
        "retrieval_status, index_state)"
    )


_MIGRATIONS = {
    2: _migration_v2_method_indexes,
    3: _migration_v3_method_index_manifest,
    4: _migration_v4_source_coverage_lookup,
}


_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS retrieval_data_versions (
    id TEXT PRIMARY KEY,
    fingerprint TEXT NOT NULL UNIQUE,
    role TEXT NOT NULL CHECK(role IN ('active', 'staging', 'previous')),
    state TEXT NOT NULL CHECK(state IN ('building', 'ready', 'failed')),
    created_at TEXT NOT NULL,
    activated_at TEXT
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_retrieval_one_active_version
    ON retrieval_data_versions(role) WHERE role='active';
CREATE UNIQUE INDEX IF NOT EXISTS idx_retrieval_one_staging_version
    ON retrieval_data_versions(role) WHERE role='staging';

CREATE TABLE IF NOT EXISTS retrieval_units (
    unit_id INTEGER PRIMARY KEY,
    data_version_id TEXT NOT NULL,
    source_type TEXT NOT NULL,
    source_unit_id TEXT NOT NULL,
    source_revision TEXT NOT NULL,
    indexed_content_hash TEXT NOT NULL,
    retrieval_status TEXT NOT NULL CHECK(retrieval_status IN ('active', 'trashed')),
    index_state TEXT NOT NULL CHECK(index_state IN ('pending', 'ready', 'failed')),
    scope_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY(data_version_id) REFERENCES retrieval_data_versions(id) ON DELETE RESTRICT,
    UNIQUE(data_version_id, source_type, source_unit_id, source_revision, indexed_content_hash)
);
CREATE INDEX IF NOT EXISTS idx_retrieval_units_search_scope
    ON retrieval_units(data_version_id, source_type, retrieval_status, index_state, unit_id);
CREATE INDEX IF NOT EXISTS idx_retrieval_units_source_ref
    ON retrieval_units(source_type, source_unit_id, source_revision);
CREATE INDEX IF NOT EXISTS idx_retrieval_units_source_coverage
    ON retrieval_units(
        data_version_id, source_type, source_unit_id, source_revision,
        retrieval_status, index_state
    );

CREATE TABLE IF NOT EXISTS retrieval_sync_receipts (
    event_id TEXT PRIMARY KEY,
    status TEXT NOT NULL CHECK(status IN ('processing', 'applied', 'retryable_failed', 'terminal_failed')),
    attempts INTEGER NOT NULL CHECK(attempts >= 1),
    reason_code TEXT,
    updated_at TEXT NOT NULL
);
"""
