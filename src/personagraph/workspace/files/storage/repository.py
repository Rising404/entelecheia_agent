"""显式 SQLite 连接上的 Workspace 文件 SQL primitive。"""

from __future__ import annotations

import json
import sqlite3

from ..contracts import (
    FileSource,
    ProjectFileObservation,
    ProjectFileRecord,
    ProjectFileVersionRecord,
    RegisteredProjectFileVersion,
)


def register_observation(
    conn: sqlite3.Connection,
    *,
    project_id: str,
    relative_path: str,
    source: FileSource,
    file_id: str,
    created_file: bool,
    new_file_version_id: str,
    new_event_id: str,
    media_type: str | None,
    occurred_at: str,
    observation: ProjectFileObservation,
) -> RegisteredProjectFileVersion:
    """在调用方拥有的事务内登记一次已完成的稳定文件观察。"""

    if created_file:
        conn.execute(
            "INSERT INTO files "
            "(id, project_id, relative_path, origin, media_type, "
            "current_version_id, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, NULL, ?, ?)",
            (
                file_id,
                project_id,
                relative_path,
                source.value,
                media_type,
                occurred_at,
                occurred_at,
            ),
        )
        version_number = 1
    else:
        version_number = int(
            conn.execute(
                "SELECT COALESCE(MAX(version_number), 0) + 1 "
                "FROM file_versions WHERE file_id=?",
                (file_id,),
            ).fetchone()[0]
        )

    conn.execute(
        "INSERT INTO file_versions "
        "(id, file_id, version_number, producer, content_sha256, "
        "size_bytes, source_mtime_ns, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            new_file_version_id,
            file_id,
            version_number,
            source.value,
            observation.content_sha256,
            observation.size_bytes,
            observation.source_mtime_ns,
            occurred_at,
        ),
    )
    conn.execute(
        "UPDATE files SET current_version_id=?, observed_mtime_ns=?, "
        "media_type=COALESCE(?, media_type), "
        "updated_at=? WHERE id=? AND project_id=?",
        (
            new_file_version_id,
            observation.source_mtime_ns,
            media_type,
            occurred_at,
            file_id,
            project_id,
        ),
    )
    conn.execute(
        "INSERT INTO file_events "
        "(event_id, file_id, file_version_id, event_type, origin, "
        "relative_path, metadata_json, occurred_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            new_event_id,
            file_id,
            new_file_version_id,
            "file_registered" if created_file else "version_registered",
            source.value,
            relative_path,
            json.dumps({"source_mtime_ns": observation.source_mtime_ns}),
            occurred_at,
        ),
    )
    file_row = conn.execute(
        "SELECT * FROM files WHERE id=? AND project_id=?",
        (file_id, project_id),
    ).fetchone()
    version_row = conn.execute(
        "SELECT * FROM file_versions WHERE id=? AND file_id=?",
        (new_file_version_id, file_id),
    ).fetchone()
    assert file_row is not None and version_row is not None
    return RegisteredProjectFileVersion(
        file=_file_from_row(file_row),
        version=_version_from_row(version_row),
        created_file=created_file,
    )


def record_current_observation(
    conn: sqlite3.Connection,
    *,
    file: ProjectFileRecord,
    version: ProjectFileVersionRecord,
    source: FileSource,
    observation: ProjectFileObservation,
    event_id: str,
    occurred_at: str,
) -> RegisteredProjectFileVersion:
    """相同内容只更新当前观察并追加事件，不改写历史版本。"""

    conn.execute(
        "UPDATE files SET observed_mtime_ns=?, updated_at=? "
        "WHERE id=? AND project_id=? AND current_version_id=?",
        (
            observation.source_mtime_ns,
            occurred_at,
            file.file_id,
            file.project_id,
            version.file_version_id,
        ),
    )
    conn.execute(
        "INSERT INTO file_events "
        "(event_id, file_id, file_version_id, event_type, origin, "
        "relative_path, metadata_json, occurred_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            event_id,
            file.file_id,
            version.file_version_id,
            "file_observed",
            source.value,
            file.relative_path,
            json.dumps({"source_mtime_ns": observation.source_mtime_ns}),
            occurred_at,
        ),
    )
    updated = get_file(conn, project_id=file.project_id, file_id=file.file_id)
    assert updated is not None
    return RegisteredProjectFileVersion(
        file=updated,
        version=version,
        created_file=False,
    )


def get_file(
    conn: sqlite3.Connection,
    *,
    project_id: str,
    file_id: str,
) -> ProjectFileRecord | None:
    row = conn.execute(
        "SELECT * FROM files WHERE id=? AND project_id=?",
        (file_id, project_id),
    ).fetchone()
    return _file_from_row(row) if row is not None else None


def get_file_by_relative_path(
    conn: sqlite3.Connection,
    *,
    project_id: str,
    relative_path: str,
) -> ProjectFileRecord | None:
    row = conn.execute(
        "SELECT * FROM files WHERE project_id=? AND relative_path=?",
        (project_id, relative_path),
    ).fetchone()
    return _file_from_row(row) if row is not None else None


def get_file_with_version(
    conn: sqlite3.Connection,
    *,
    project_id: str,
    file_id: str,
    file_version_id: str,
) -> tuple[ProjectFileRecord, ProjectFileVersionRecord] | None:
    row = conn.execute(
        "SELECT "
        "file.id AS joined_file_id, "
        "file.project_id AS joined_project_id, "
        "file.relative_path AS joined_relative_path, "
        "file.origin AS joined_file_origin, "
        "file.media_type AS joined_media_type, "
        "file.current_version_id AS joined_current_version_id, "
        "file.observed_mtime_ns AS joined_observed_mtime_ns, "
        "file.created_at AS joined_file_created_at, "
        "file.updated_at AS joined_file_updated_at, "
        "version.id AS joined_version_id, "
        "version.file_id AS joined_version_file_id, "
        "version.version_number AS joined_version_number, "
        "version.producer AS joined_version_producer, "
        "version.content_sha256 AS joined_content_sha256, "
        "version.size_bytes AS joined_size_bytes, "
        "version.source_mtime_ns AS joined_source_mtime_ns, "
        "version.created_at AS joined_version_created_at "
        "FROM files AS file "
        "JOIN file_versions AS version ON version.file_id=file.id "
        "WHERE file.id=? AND version.id=? AND file.project_id=?",
        (file_id, file_version_id, project_id),
    ).fetchone()
    if row is None:
        return None
    return _file_with_version_from_joined_row(row)


def get_version(
    conn: sqlite3.Connection,
    *,
    project_id: str,
    file_version_id: str,
) -> ProjectFileVersionRecord | None:
    row = conn.execute(
        "SELECT version.* FROM file_versions AS version "
        "JOIN files AS file ON file.id=version.file_id "
        "WHERE version.id=? AND file.project_id=?",
        (file_version_id, project_id),
    ).fetchone()
    return _version_from_row(row) if row is not None else None


def list_current_files(conn: sqlite3.Connection, *, project_id: str) -> tuple[ProjectFileRecord, ...]:
    """只枚举已有登记；不访问目录，不为未知文件创建记录。"""
    rows = conn.execute(
        "SELECT * FROM files WHERE project_id=? AND current_version_id IS NOT NULL ORDER BY id",
        (project_id,),
    ).fetchall()
    return tuple(_file_from_row(row) for row in rows)


def list_versions(
    conn: sqlite3.Connection,
    *,
    project_id: str,
    file_id: str,
) -> tuple[ProjectFileVersionRecord, ...]:
    rows = conn.execute(
        "SELECT version.* FROM file_versions AS version "
        "JOIN files AS file ON file.id=version.file_id "
        "WHERE version.file_id=? AND file.project_id=? "
        "ORDER BY version.version_number",
        (file_id, project_id),
    ).fetchall()
    return tuple(_version_from_row(row) for row in rows)


def _file_from_row(row: sqlite3.Row) -> ProjectFileRecord:
    return ProjectFileRecord(
        file_id=str(row["id"]),
        project_id=str(row["project_id"]),
        relative_path=str(row["relative_path"]),
        source=FileSource(str(row["origin"])),
        media_type=str(row["media_type"]) if row["media_type"] is not None else None,
        current_version_id=(
            str(row["current_version_id"])
            if row["current_version_id"] is not None
            else None
        ),
        observed_mtime_ns=(
            int(row["observed_mtime_ns"])
            if row["observed_mtime_ns"] is not None
            else None
        ),
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
    )


def _version_from_row(row: sqlite3.Row) -> ProjectFileVersionRecord:
    return ProjectFileVersionRecord(
        file_version_id=str(row["id"]),
        file_id=str(row["file_id"]),
        version_number=int(row["version_number"]),
        source=FileSource(str(row["producer"])),
        content_sha256=str(row["content_sha256"]),
        size_bytes=int(row["size_bytes"]),
        source_mtime_ns=(
            int(row["source_mtime_ns"])
            if row["source_mtime_ns"] is not None
            else None
        ),
        created_at=str(row["created_at"]),
    )


def _file_with_version_from_joined_row(
    row: sqlite3.Row,
) -> tuple[ProjectFileRecord, ProjectFileVersionRecord]:
    return (
        ProjectFileRecord(
            file_id=str(row["joined_file_id"]),
            project_id=str(row["joined_project_id"]),
            relative_path=str(row["joined_relative_path"]),
            source=FileSource(str(row["joined_file_origin"])),
            media_type=(
                str(row["joined_media_type"])
                if row["joined_media_type"] is not None
                else None
            ),
            current_version_id=(
                str(row["joined_current_version_id"])
                if row["joined_current_version_id"] is not None
                else None
            ),
            observed_mtime_ns=(
                int(row["joined_observed_mtime_ns"])
                if row["joined_observed_mtime_ns"] is not None
                else None
            ),
            created_at=str(row["joined_file_created_at"]),
            updated_at=str(row["joined_file_updated_at"]),
        ),
        ProjectFileVersionRecord(
            file_version_id=str(row["joined_version_id"]),
            file_id=str(row["joined_version_file_id"]),
            version_number=int(row["joined_version_number"]),
            source=FileSource(str(row["joined_version_producer"])),
            content_sha256=str(row["joined_content_sha256"]),
            size_bytes=int(row["joined_size_bytes"]),
            source_mtime_ns=(
                int(row["joined_source_mtime_ns"])
                if row["joined_source_mtime_ns"] is not None
                else None
            ),
            created_at=str(row["joined_version_created_at"]),
        ),
    )
