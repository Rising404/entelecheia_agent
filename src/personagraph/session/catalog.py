"""共享项目目录中的 Session 元数据，以及单 Session 数据库定位器。

该目录刻意比 Session 数据库更小，只拥有枚举和路由 Session 所需的元数据；对话、模型
输入与输出、附件和运行时状态属于 ``sessions/<id>/session.sqlite``。

生产 ``session.store`` 门面使用此目录为每个 Session 定位一个物理数据库。显式覆盖旧版
存储路径的调用方仍走单数据库兼容接缝。
"""

from __future__ import annotations

import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from personagraph.configuration import paths
from .persistence.metadata import creation_requests
from .persistence.metadata.creation_requests import (
    CreationRequestConflict as CreationRequestConflict,
    CreationRequestInProgress as CreationRequestInProgress,
)


SCHEMA_VERSION = 1
SESSION_DATABASE_FILENAME = "session.sqlite"
SESSION_STATUSES = frozenset({"active", "archived", "trashed"})
FOLDER_STATUSES = SESSION_STATUSES

_SESSION_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}\Z")
_UNSET = object()
_EXPECTED_TABLE_COLUMNS = {
    "session_folders": {
        "id",
        "name",
        "parent_id",
        "status",
        "sort_order",
        "created_at",
        "updated_at",
        "previous_status",
    },
    "sessions": {
        "id",
        "project_id",
        "folder_id",
        "title",
        "persona_id",
        "status",
        "db_path",
        "created_at",
        "last_active_at",
        "updated_at",
        "archived_at",
        "deleted_at",
        "previous_status",
    },
    "session_purge_tombstones": {
        "session_id",
        "project_id",
        "db_path",
        "state",
        "started_at",
        "completed_at",
    },
}

_CREATE_FOLDERS_SQL = """
CREATE TABLE IF NOT EXISTS session_folders (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL CHECK (length(trim(name)) > 0),
    parent_id   TEXT,
    status      TEXT NOT NULL DEFAULT 'active'
                CHECK (status IN ('active', 'archived', 'trashed')),
    sort_order  INTEGER NOT NULL DEFAULT 0,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL,
    previous_status TEXT,
    CHECK (length(id) BETWEEN 1 AND 255),
    CHECK (instr(id, char(0)) = 0),
    CHECK (parent_id IS NULL OR parent_id <> id),
    FOREIGN KEY (parent_id) REFERENCES session_folders(id) ON DELETE RESTRICT
)
"""

_CREATE_SESSIONS_SQL = """
CREATE TABLE IF NOT EXISTS sessions (
    id              TEXT PRIMARY KEY,
    project_id      TEXT CHECK (
                        project_id IS NULL OR length(trim(project_id)) > 0
                    ),
    folder_id       TEXT,
    title           TEXT,
    persona_id      TEXT,
    status          TEXT NOT NULL DEFAULT 'active'
                    CHECK (status IN ('active', 'archived', 'trashed')),
    db_path         TEXT NOT NULL UNIQUE,
    created_at      TEXT NOT NULL,
    last_active_at  TEXT NOT NULL,
    updated_at      TEXT NOT NULL,
    archived_at     TEXT,
    deleted_at      TEXT,
    previous_status TEXT,
    CHECK (length(id) BETWEEN 1 AND 128),
    CHECK (id NOT GLOB '*[^A-Za-z0-9_-]*'),
    CHECK (substr(id, 1, 1) GLOB '[A-Za-z0-9]'),
    CHECK (db_path = 'sessions/' || id || '/session.sqlite'),
    FOREIGN KEY (folder_id) REFERENCES session_folders(id) ON DELETE RESTRICT
)
"""

_CREATE_PURGE_TOMBSTONES_SQL = """
CREATE TABLE IF NOT EXISTS session_purge_tombstones (
    session_id   TEXT PRIMARY KEY,
    project_id   TEXT,
    db_path      TEXT NOT NULL UNIQUE,
    state        TEXT NOT NULL CHECK (state IN ('pending', 'completed')),
    started_at   TEXT NOT NULL,
    completed_at TEXT,
    CHECK (db_path = 'sessions/' || session_id || '/session.sqlite')
)
"""

_CREATE_INDEXES_SQL = (
    "CREATE INDEX IF NOT EXISTS sessions_project_status_activity_idx "
    "ON sessions(project_id, status, last_active_at DESC)",
    "CREATE INDEX IF NOT EXISTS sessions_folder_status_idx ON sessions(folder_id, status)",
    "CREATE INDEX IF NOT EXISTS session_folders_parent_sort_idx "
    "ON session_folders(parent_id, sort_order, name)",
)


class SessionCatalogError(RuntimeError):
    """无法安全打开共享项目与 Session 目录。"""


class SessionCatalogSchemaError(SessionCatalogError):
    """磁盘目录不是受支持的目录 schema。"""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def validate_session_id(session_id: str) -> str:
    """返回安全 Session ID，或拒绝可能影响路由的值。"""

    if not isinstance(session_id, str) or _SESSION_ID_PATTERN.fullmatch(session_id) is None:
        raise ValueError(
            "session_id must be 1-128 ASCII letters, digits, '_' or '-', "
            "and must start with a letter or digit"
        )
    return session_id


def _state_root(state_dir: str | Path | None) -> Path:
    configured = paths.STATE_DIR if state_dir is None else Path(state_dir)
    return Path(configured).expanduser().resolve()


def catalog_db_path(*, state_dir: str | Path | None = None) -> Path:
    """返回唯一共享项目与 Session 目录，不创建它。"""

    if state_dir is not None:
        configured = _state_root(state_dir) / "project_catalog.sqlite"
    else:
        configured = Path(paths.PROJECT_CATALOG_DB_PATH).expanduser()
    return paths.resolved_project_catalog_path(configured)


def sessions_directory(*, state_dir: str | Path | None = None) -> Path:
    """返回规范 sessions 目录，并拒绝逸出范围的符号链接。"""

    root = _state_root(state_dir)
    if state_dir is not None:
        configured = root / "sessions"
    else:
        configured = Path(getattr(paths, "SESSIONS_DIR", root / "sessions")).expanduser()
    resolved = configured.resolve()
    if resolved.parent != root or resolved.name != "sessions":
        raise ValueError("sessions directory must resolve to <state_dir>/sessions")
    return resolved


def session_db_relative_path(session_id: str) -> str:
    """返回目录中存储的可移植路径。"""

    safe_id = validate_session_id(session_id)
    return f"sessions/{safe_id}/{SESSION_DATABASE_FILENAME}"


def session_db_path(
    session_id: str,
    *,
    state_dir: str | Path | None = None,
) -> Path:
    """解析一个 Session 数据库；遇到目录穿越时以关闭方式失败。"""

    safe_id = validate_session_id(session_id)
    root = sessions_directory(state_dir=state_dir)
    candidate = (root / safe_id / SESSION_DATABASE_FILENAME).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError("session database path escapes the sessions directory") from exc
    if candidate.parent.parent != root:
        raise ValueError("session database path must be exactly sessions/<id>/session.sqlite")
    return candidate


def _validate_nonempty_identifier(value: str, *, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip() or "\x00" in value:
        raise ValueError(f"{name} must be a non-empty, trimmed string without NUL bytes")
    return value


def _validate_status(status: str, *, allowed: frozenset[str], name: str) -> str:
    if status not in allowed:
        choices = ", ".join(sorted(allowed))
        raise ValueError(f"invalid {name}: {status!r}; expected one of {choices}")
    return status


class SessionCatalog:
    """Session 路由与生命周期元数据的小型 SQLite 权威源。"""

    def __init__(self, *, state_dir: str | Path | None = None) -> None:
        self.db_path = catalog_db_path(state_dir=state_dir)
        self.state_dir = self.db_path.parent

    def session_db_path(self, session_id: str) -> Path:
        return session_db_path(session_id, state_dir=self.state_dir)

    def _connect(self) -> sqlite3.Connection:
        conn: sqlite3.Connection | None = None
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(self.db_path, timeout=5.0)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute("PRAGMA busy_timeout = 5000")
            return conn
        except (OSError, sqlite3.Error) as exc:
            if conn is not None:
                conn.close()
            raise SessionCatalogError(f"failed to open shared catalog at {self.db_path}") from exc

    def initialize(self) -> None:
        """创建空目录，或校验现有目录。"""

        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            version = int(conn.execute("PRAGMA user_version").fetchone()[0])
            if version not in {0, SCHEMA_VERSION}:
                raise SessionCatalogSchemaError(
                    f"unsupported shared catalog schema version {version}; expected {SCHEMA_VERSION}"
                )
            conn.execute(_CREATE_FOLDERS_SQL)
            conn.execute(_CREATE_SESSIONS_SQL)
            self._ensure_additive_lifecycle_columns(conn)
            conn.execute(_CREATE_PURGE_TOMBSTONES_SQL)
            creation_requests.initialize_schema(conn)
            for statement in _CREATE_INDEXES_SQL:
                conn.execute(statement)
            if version == 0:
                conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            self._validate_schema(conn)

    @staticmethod
    def _ensure_additive_lifecycle_columns(conn: sqlite3.Connection) -> None:
        """原地升级由初始分区补丁创建的目录。

        共享目录的 ``user_version`` 也被项目注册表使用，因此纯生命周期的增量变化不能
        擅自把它用作迁移计数器。两个列都可为 null，且不含内容。
        """

        folder_columns = {
            str(row["name"])
            for row in conn.execute('PRAGMA table_info("session_folders")').fetchall()
        }
        if "previous_status" not in folder_columns:
            conn.execute("ALTER TABLE session_folders ADD COLUMN previous_status TEXT")
        session_columns = {
            str(row["name"])
            for row in conn.execute('PRAGMA table_info("sessions")').fetchall()
        }
        if "previous_status" not in session_columns:
            conn.execute("ALTER TABLE sessions ADD COLUMN previous_status TEXT")

    @classmethod
    def _validate_schema(cls, conn: sqlite3.Connection) -> None:
        for table, expected_columns in _EXPECTED_TABLE_COLUMNS.items():
            actual_columns = {
                str(row["name"])
                for row in conn.execute(f'PRAGMA table_info("{table}")').fetchall()
            }
            if actual_columns != expected_columns:
                raise SessionCatalogSchemaError(
                    f"session-owned catalog table {table!r} columns differ: "
                    f"expected {sorted(expected_columns)}, found {sorted(actual_columns)}"
                )

    def create_session(
        self,
        *,
        session_id: str,
        project_id: str | None = None,
        title: str | None = None,
        persona_id: str | None = None,
        folder_id: str | None = None,
        status: str = "active",
        occurred_at: str | None = None,
        creation_request_id: str | None = None,
    ) -> dict[str, Any]:
        safe_id = validate_session_id(session_id)
        safe_project_id = (
            _validate_nonempty_identifier(project_id, name="project_id")
            if project_id is not None
            else None
        )
        safe_status = _validate_status(status, allowed=SESSION_STATUSES, name="session status")
        if folder_id is not None:
            _validate_nonempty_identifier(folder_id, name="folder_id")
        self.session_db_path(safe_id)
        now = occurred_at or _utc_now()
        self.initialize()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO sessions "
                "(id, project_id, folder_id, title, persona_id, status, db_path, "
                "created_at, last_active_at, updated_at, archived_at, deleted_at, "
                "previous_status) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)",
                (
                    safe_id,
                    safe_project_id,
                    folder_id,
                    title,
                    persona_id,
                    safe_status,
                    session_db_relative_path(safe_id),
                    now,
                    now,
                    now,
                    now if safe_status == "archived" else None,
                    now if safe_status == "trashed" else None,
                ),
            )
            if creation_request_id is not None:
                creation_requests.publish(
                    conn, request_id=creation_request_id, session_id=safe_id,
                )
            row = conn.execute("SELECT * FROM sessions WHERE id=?", (safe_id,)).fetchone()
        return dict(row)

    def reserve_creation_request(self, request_id: str, request_hash: str) -> str | None:
        self.initialize()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            return creation_requests.reserve(
                conn, request_id=request_id, request_hash=request_hash, occurred_at=_utc_now(),
            )

    def release_unpublished_creation(self, request_id: str) -> bool:
        self.initialize()
        with self._connect() as conn:
            return creation_requests.release_unpublished(conn, request_id=request_id)

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        safe_id = validate_session_id(session_id)
        self.initialize()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT session.* FROM sessions AS session WHERE session.id=? "
                "AND NOT EXISTS (SELECT 1 FROM session_purge_tombstones AS purge "
                "WHERE purge.session_id=session.id AND purge.state='pending')",
                (safe_id,),
            ).fetchone()
        return dict(row) if row is not None else None

    def list_sessions(
        self,
        *,
        project_id: str | None = None,
        status: str | None = None,
        folder_id: str | None = None,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if project_id is not None:
            clauses.append("project_id=?")
            params.append(_validate_nonempty_identifier(project_id, name="project_id"))
        if status is not None:
            clauses.append("status=?")
            params.append(_validate_status(status, allowed=SESSION_STATUSES, name="session status"))
        if folder_id is not None:
            clauses.append("folder_id=?")
            params.append(_validate_nonempty_identifier(folder_id, name="folder_id"))
        clauses.append(
            "NOT EXISTS (SELECT 1 FROM session_purge_tombstones AS purge "
            "WHERE purge.session_id=sessions.id AND purge.state='pending')"
        )
        sql = "SELECT * FROM sessions"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY last_active_at DESC, id"
        self.initialize()
        with self._connect() as conn:
            return [dict(row) for row in conn.execute(sql, params).fetchall()]

    def update_session(
        self,
        session_id: str,
        *,
        title: str | None | object = _UNSET,
        persona_id: str | None | object = _UNSET,
        project_id: str | None | object = _UNSET,
        folder_id: str | None | object = _UNSET,
        status: str | object = _UNSET,
        last_active_at: str | object = _UNSET,
        occurred_at: str | None = None,
    ) -> dict[str, Any] | None:
        safe_id = validate_session_id(session_id)
        assignments: list[str] = []
        params: list[Any] = []
        if title is not _UNSET:
            assignments.append("title=?")
            params.append(title)
        if persona_id is not _UNSET:
            assignments.append("persona_id=?")
            params.append(persona_id)
        if project_id is not _UNSET:
            if project_id is not None:
                _validate_nonempty_identifier(project_id, name="project_id")
            assignments.append("project_id=?")
            params.append(project_id)
        if folder_id is not _UNSET:
            if folder_id is not None:
                _validate_nonempty_identifier(folder_id, name="folder_id")
            assignments.append("folder_id=?")
            params.append(folder_id)
        if last_active_at is not _UNSET:
            _validate_nonempty_identifier(last_active_at, name="last_active_at")
            assignments.append("last_active_at=?")
            params.append(last_active_at)
        now = occurred_at or _utc_now()
        if status is not _UNSET:
            safe_status = _validate_status(
                status,
                allowed=SESSION_STATUSES,
                name="session status",
            )
            assignments.append("status=?")
            params.append(safe_status)
            if safe_status == "active":
                assignments.extend(
                    ("archived_at=NULL", "deleted_at=NULL", "previous_status=NULL")
                )
            elif safe_status == "archived":
                assignments.extend(
                    (
                        "archived_at=COALESCE(archived_at, ?)",
                        "deleted_at=NULL",
                        "previous_status=NULL",
                    )
                )
                params.append(now)
            else:
                assignments.extend(
                    (
                        "previous_status=CASE WHEN status!='trashed' "
                        "THEN status ELSE previous_status END",
                        "deleted_at=COALESCE(deleted_at, ?)",
                    )
                )
                params.append(now)
        if not assignments:
            return self.get_session(safe_id)
        assignments.append("updated_at=?")
        params.extend((now, safe_id))
        self.initialize()
        with self._connect() as conn:
            updated = conn.execute(
                f"UPDATE sessions SET {', '.join(assignments)} WHERE id=?",
                params,
            ).rowcount
            if not updated:
                return None
            row = conn.execute("SELECT * FROM sessions WHERE id=?", (safe_id,)).fetchone()
        return dict(row)

    def begin_session_purge(
        self,
        session_id: str,
        *,
        occurred_at: str | None = None,
    ) -> dict[str, Any] | None:
        """修改 Session 数据库前持久隐藏清除意图。"""

        safe_id = validate_session_id(session_id)
        now = occurred_at or _utc_now()
        self.initialize()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT id, project_id, db_path FROM sessions WHERE id=?",
                (safe_id,),
            ).fetchone()
            if row is None:
                return None
            existing = conn.execute(
                "SELECT * FROM session_purge_tombstones WHERE session_id=?",
                (safe_id,),
            ).fetchone()
            if existing is not None:
                return dict(existing)
            conn.execute(
                "INSERT INTO session_purge_tombstones "
                "(session_id, project_id, db_path, state, started_at, completed_at) "
                "VALUES (?, ?, ?, 'pending', ?, NULL)",
                (safe_id, row["project_id"], row["db_path"], now),
            )
            tombstone = conn.execute(
                "SELECT * FROM session_purge_tombstones WHERE session_id=?",
                (safe_id,),
            ).fetchone()
        return dict(tombstone)

    def begin_unpublished_session_purge(
        self,
        session_id: str,
        *,
        project_id: str | None = None,
        occurred_at: str | None = None,
    ) -> dict[str, Any]:
        """持久声明一个从未发布 catalog locator 的残留分区需要清理。"""

        safe_id = validate_session_id(session_id)
        safe_project_id = (
            _validate_nonempty_identifier(project_id, name="project_id")
            if project_id is not None
            else None
        )
        now = occurred_at or _utc_now()
        expected_db_path = session_db_relative_path(safe_id)
        self.initialize()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            if conn.execute(
                "SELECT 1 FROM sessions WHERE id=?",
                (safe_id,),
            ).fetchone() is not None:
                raise SessionCatalogError(
                    "cannot register unpublished cleanup for a published session"
                )
            existing = conn.execute(
                "SELECT * FROM session_purge_tombstones WHERE session_id=?",
                (safe_id,),
            ).fetchone()
            if existing is not None:
                return dict(existing)
            conn.execute(
                "INSERT INTO session_purge_tombstones "
                "(session_id, project_id, db_path, state, started_at, completed_at) "
                "VALUES (?, ?, ?, 'pending', ?, NULL)",
                (safe_id, safe_project_id, expected_db_path, now),
            )
            tombstone = conn.execute(
                "SELECT * FROM session_purge_tombstones WHERE session_id=?",
                (safe_id,),
            ).fetchone()
        assert tombstone is not None
        return dict(tombstone)

    def list_pending_unpublished_session_purges(
        self,
        *,
        limit: int = 16,
    ) -> list[dict[str, Any]]:
        """列出没有活动 locator 的有界残留分区清理声明。"""

        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise ValueError("limit must be a positive integer")
        self.initialize()
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT purge.* FROM session_purge_tombstones AS purge "
                "WHERE purge.state='pending' AND NOT EXISTS ("
                "SELECT 1 FROM sessions AS session "
                "WHERE session.id=purge.session_id"
                ") ORDER BY purge.started_at, purge.session_id LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def complete_session_purge(
        self,
        session_id: str,
        *,
        occurred_at: str | None = None,
    ) -> bool:
        """仅在 Session 载荷已清除后移除活动定位器。"""

        safe_id = validate_session_id(session_id)
        now = occurred_at or _utc_now()
        self.initialize()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            tombstone = conn.execute(
                "SELECT state FROM session_purge_tombstones WHERE session_id=?",
                (safe_id,),
            ).fetchone()
            if tombstone is None:
                return False
            conn.execute("DELETE FROM sessions WHERE id=?", (safe_id,))
            conn.execute(
                "UPDATE session_purge_tombstones SET state='completed', completed_at=? "
                "WHERE session_id=?",
                (now, safe_id),
            )
        return True

    def cancel_session_purge(self, session_id: str) -> bool:
        """Session 载荷变更失败时清除待处理墓碑。"""

        safe_id = validate_session_id(session_id)
        self.initialize()
        with self._connect() as conn:
            return bool(
                conn.execute(
                    "DELETE FROM session_purge_tombstones "
                    "WHERE session_id=? AND state='pending'",
                    (safe_id,),
                ).rowcount
            )

    def get_session_purge_tombstone(self, session_id: str) -> dict[str, Any] | None:
        safe_id = validate_session_id(session_id)
        self.initialize()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM session_purge_tombstones WHERE session_id=?",
                (safe_id,),
            ).fetchone()
        return dict(row) if row is not None else None

    def create_folder(
        self,
        *,
        folder_id: str,
        name: str,
        parent_id: str | None = None,
        status: str = "active",
        sort_order: int = 0,
        occurred_at: str | None = None,
    ) -> dict[str, Any]:
        safe_id = _validate_nonempty_identifier(folder_id, name="folder_id")
        if parent_id is not None:
            _validate_nonempty_identifier(parent_id, name="parent_id")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("folder name must not be empty")
        safe_status = _validate_status(status, allowed=FOLDER_STATUSES, name="folder status")
        now = occurred_at or _utc_now()
        self.initialize()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO session_folders "
                "(id, name, parent_id, status, sort_order, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (safe_id, name, parent_id, safe_status, int(sort_order), now, now),
            )
            row = conn.execute("SELECT * FROM session_folders WHERE id=?", (safe_id,)).fetchone()
        return dict(row)

    def get_folder(self, folder_id: str) -> dict[str, Any] | None:
        safe_id = _validate_nonempty_identifier(folder_id, name="folder_id")
        self.initialize()
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM session_folders WHERE id=?", (safe_id,)).fetchone()
        return dict(row) if row is not None else None

    def list_folders(self, *, status: str | None = None) -> list[dict[str, Any]]:
        params: list[Any] = []
        sql = "SELECT * FROM session_folders"
        if status is not None:
            sql += " WHERE status=?"
            params.append(_validate_status(status, allowed=FOLDER_STATUSES, name="folder status"))
        sql += " ORDER BY sort_order, name, id"
        self.initialize()
        with self._connect() as conn:
            return [dict(row) for row in conn.execute(sql, params).fetchall()]

    def update_folder(
        self,
        folder_id: str,
        *,
        name: str | object = _UNSET,
        parent_id: str | None | object = _UNSET,
        status: str | object = _UNSET,
        sort_order: int | object = _UNSET,
        occurred_at: str | None = None,
    ) -> dict[str, Any] | None:
        safe_id = _validate_nonempty_identifier(folder_id, name="folder_id")
        assignments: list[str] = []
        params: list[Any] = []
        if name is not _UNSET:
            if not isinstance(name, str) or not name.strip():
                raise ValueError("folder name must not be empty")
            assignments.append("name=?")
            params.append(name)
        if parent_id is not _UNSET:
            if parent_id is not None:
                _validate_nonempty_identifier(parent_id, name="parent_id")
                if parent_id == safe_id:
                    raise ValueError("folder cannot be its own parent")
            assignments.append("parent_id=?")
            params.append(parent_id)
        if status is not _UNSET:
            safe_status = _validate_status(
                status,
                allowed=FOLDER_STATUSES,
                name="folder status",
            )
            assignments.append("status=?")
            params.append(safe_status)
            if safe_status == "trashed":
                assignments.append(
                    "previous_status=CASE WHEN status!='trashed' "
                    "THEN status ELSE previous_status END"
                )
            else:
                assignments.append("previous_status=NULL")
        if sort_order is not _UNSET:
            assignments.append("sort_order=?")
            params.append(int(sort_order))
        if not assignments:
            return self.get_folder(safe_id)
        now = occurred_at or _utc_now()
        assignments.append("updated_at=?")
        params.append(now)
        self.initialize()
        with self._connect() as conn:
            if parent_id is not _UNSET and parent_id is not None:
                descendant = conn.execute(
                    "WITH RECURSIVE descendants(id) AS ("
                    " SELECT id FROM session_folders WHERE parent_id=?"
                    " UNION"
                    " SELECT child.id FROM session_folders child"
                    " JOIN descendants parent ON child.parent_id=parent.id"
                    ") SELECT 1 FROM descendants WHERE id=?",
                    (safe_id, parent_id),
                ).fetchone()
                if descendant is not None:
                    raise ValueError("folder cannot be moved into one of its descendants")
            params.append(safe_id)
            updated = conn.execute(
                f"UPDATE session_folders SET {', '.join(assignments)} WHERE id=?",
                params,
            ).rowcount
            if not updated:
                return None
            row = conn.execute("SELECT * FROM session_folders WHERE id=?", (safe_id,)).fetchone()
        return dict(row)

    def delete_folder(self, folder_id: str) -> tuple[bool, str]:
        """从共享元数据权威源删除空叶文件夹。"""

        safe_id = _validate_nonempty_identifier(folder_id, name="folder_id")
        self.initialize()
        with self._connect() as conn:
            if conn.execute(
                "SELECT 1 FROM session_folders WHERE id=?",
                (safe_id,),
            ).fetchone() is None:
                return False, "folder_not_found"
            session_count = int(
                conn.execute(
                    "SELECT COUNT(*) FROM sessions "
                    "WHERE folder_id=? AND status!='trashed'",
                    (safe_id,),
                ).fetchone()[0]
            )
            if session_count:
                return False, f"folder_not_empty({session_count})"
            child_count = int(
                conn.execute(
                    "SELECT COUNT(*) FROM session_folders WHERE parent_id=?",
                    (safe_id,),
                ).fetchone()[0]
            )
            if child_count:
                return False, f"folder_not_empty({child_count})"
            conn.execute("DELETE FROM session_folders WHERE id=?", (safe_id,))
        return True, "deleted"
