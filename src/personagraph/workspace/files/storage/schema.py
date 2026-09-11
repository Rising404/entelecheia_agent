"""Workspace File authority 对共享项目数据库贡献的稳定 DDL。"""

from __future__ import annotations

import sqlite3


FILE_REQUIRED_TABLES = frozenset({"files", "file_versions", "file_events"})


_STATEMENTS = (
    """
    CREATE TABLE files (
        id TEXT PRIMARY KEY,
        project_id TEXT NOT NULL CHECK(length(project_id) > 0),
        relative_path TEXT NOT NULL CHECK(
            length(trim(relative_path)) > 0
            AND relative_path NOT IN ('.', '..')
            AND substr(relative_path, 1, 1) <> '/'
            AND relative_path NOT LIKE '../%'
            AND relative_path NOT LIKE '%/../%'
            AND instr(relative_path, '\\') = 0
        ),
        origin TEXT NOT NULL CHECK(origin IN (
            'user_upload', 'workspace_existing', 'agent_output'
        )),
        media_type TEXT,
        current_version_id TEXT,
        observed_mtime_ns INTEGER CHECK(
            observed_mtime_ns IS NULL OR observed_mtime_ns >= 0
        ),
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        UNIQUE(project_id, relative_path),
        FOREIGN KEY(current_version_id) REFERENCES file_versions(id) ON DELETE SET NULL
    )
    """,
    """
    CREATE TABLE file_versions (
        id TEXT PRIMARY KEY,
        file_id TEXT NOT NULL,
        version_number INTEGER NOT NULL CHECK(version_number > 0),
        producer TEXT NOT NULL CHECK(producer IN (
            'user_upload', 'workspace_existing', 'agent_output'
        )),
        content_sha256 TEXT NOT NULL CHECK(
            length(content_sha256) = 64
            AND content_sha256 NOT GLOB '*[^0-9a-f]*'
        ),
        size_bytes INTEGER NOT NULL CHECK(size_bytes >= 0),
        source_mtime_ns INTEGER CHECK(source_mtime_ns IS NULL OR source_mtime_ns >= 0),
        created_at TEXT NOT NULL,
        UNIQUE(file_id, version_number),
        UNIQUE(id, file_id),
        FOREIGN KEY(file_id) REFERENCES files(id) ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE file_events (
        event_id TEXT PRIMARY KEY,
        file_id TEXT NOT NULL,
        file_version_id TEXT,
        event_type TEXT NOT NULL CHECK(length(event_type) > 0),
        origin TEXT NOT NULL CHECK(origin IN (
            'user_upload', 'workspace_existing', 'agent_output'
        )),
        relative_path TEXT NOT NULL CHECK(length(trim(relative_path)) > 0),
        metadata_json TEXT NOT NULL DEFAULT '{}' CHECK(json_valid(metadata_json)),
        occurred_at TEXT NOT NULL,
        FOREIGN KEY(file_id) REFERENCES files(id) ON DELETE CASCADE,
        FOREIGN KEY(file_version_id) REFERENCES file_versions(id) ON DELETE RESTRICT
    )
    """,
    """
    CREATE TRIGGER trg_files_project_insert
    BEFORE INSERT ON files
    WHEN NEW.project_id != (
        SELECT project_id FROM schema_meta WHERE schema_name = 'project_documents'
    )
    BEGIN
        SELECT RAISE(ABORT, 'file project_id does not match database project');
    END
    """,
    """
    CREATE TRIGGER trg_files_project_update
    BEFORE UPDATE OF project_id ON files
    WHEN NEW.project_id != OLD.project_id
      OR NEW.project_id != (
          SELECT project_id FROM schema_meta WHERE schema_name = 'project_documents'
      )
    BEGIN
        SELECT RAISE(ABORT, 'file project_id is immutable');
    END
    """,
    "CREATE INDEX idx_file_events_file ON file_events(file_id, occurred_at, event_id)",
)


def initialize_file_schema(conn: sqlite3.Connection) -> None:
    """在调用方事务中安装 File contribution。"""

    for statement in _STATEMENTS:
        conn.execute(statement)


def validate_file_schema(
    conn: sqlite3.Connection, *, require_observation: bool = True,
) -> None:
    present = _application_objects(conn)
    missing = sorted(FILE_REQUIRED_TABLES - present)
    if missing:
        raise ValueError("workspace file schema is incomplete: " + ", ".join(missing))
    columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(files)")}
    if require_observation and "observed_mtime_ns" not in columns:
        raise ValueError("workspace file schema is missing current observation metadata")


def migrate_file_observation_schema(conn: sqlite3.Connection) -> None:
    """在 v5→v6 迁移事务中独立当前观察，保留历史内容版本的捕获元数据。"""

    columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(files)")}
    if "observed_mtime_ns" in columns:
        return
    conn.execute(
        "ALTER TABLE files ADD COLUMN observed_mtime_ns INTEGER "
        "CHECK(observed_mtime_ns IS NULL OR observed_mtime_ns >= 0)"
    )
    conn.execute(
        "UPDATE files SET observed_mtime_ns=("
        "SELECT source_mtime_ns FROM file_versions WHERE id=files.current_version_id)"
    )


def _application_objects(conn: sqlite3.Connection) -> frozenset[str]:
    return frozenset(
        str(row[0])
        for row in conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE name NOT LIKE 'sqlite_%' "
            "AND type IN ('table', 'view', 'trigger')"
        ).fetchall()
    )
