"""显式 SQLite 连接上的 Document 与 DocumentVersion 元数据原语。"""

from __future__ import annotations

from collections.abc import Mapping
import sqlite3
from typing import Any


_EDITABLE_FIELDS = frozenset({"summary", "title", "tags"})


def list_documents(conn: sqlite3.Connection) -> tuple[sqlite3.Row, ...]:
    return tuple(
        conn.execute(
            "SELECT d.id, d.path, d.title, d.mime, d.summary, d.tags, d.task_id, "
            "d.n_chunks, d.added_at, v.processing_status, v.diagnostics_json "
            "FROM documents AS d LEFT JOIN document_versions AS v "
            "ON v.id=d.current_version_id ORDER BY d.added_at DESC"
        ).fetchall()
    )


def get_document(
    conn: sqlite3.Connection,
    document_id: str,
) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT d.*, v.processing_status, v.diagnostics_json "
        "FROM documents AS d LEFT JOIN document_versions AS v "
        "ON v.id=d.current_version_id WHERE d.id=?",
        (document_id,),
    ).fetchone()


def get_documents(
    conn: sqlite3.Connection,
    document_ids: tuple[str, ...],
) -> tuple[sqlite3.Row, ...]:
    if not document_ids:
        return ()
    placeholders = ",".join("?" for _ in document_ids)
    return tuple(
        conn.execute(
            "SELECT d.*, v.file_version_id, v.processing_status, v.diagnostics_json "
            "FROM documents AS d LEFT JOIN document_versions AS v "
            "ON v.id=d.current_version_id "
            f"WHERE d.id IN ({placeholders})",
            document_ids,
        ).fetchall()
    )


def get_document_freshness_rows(
    conn: sqlite3.Connection,
    document_ids: tuple[str, ...],
) -> tuple[sqlite3.Row, ...]:
    if not document_ids:
        return ()
    placeholders = ",".join("?" for _ in document_ids)
    return tuple(
        conn.execute(
            "SELECT d.id, d.path, d.source_sha256, d.current_version_id, "
            "v.processing_status, v.diagnostics_json FROM documents AS d "
            "LEFT JOIN document_versions AS v ON v.id=d.current_version_id "
            f"WHERE d.id IN ({placeholders})",
            document_ids,
        ).fetchall()
    )


def edit_document(
    conn: sqlite3.Connection,
    document_id: str,
    fields: Mapping[str, Any],
) -> bool:
    allowed = {
        key: value
        for key, value in fields.items()
        if key in _EDITABLE_FIELDS and value is not None
    }
    if not allowed:
        return False
    assignments = ", ".join(f"{key}=?" for key in allowed)
    return (
        conn.execute(
            f"UPDATE documents SET {assignments} WHERE id=?",
            (*allowed.values(), document_id),
        ).rowcount
        > 0
    )


def document_exists(conn: sqlite3.Connection, document_id: str) -> bool:
    return (
        conn.execute(
            "SELECT 1 FROM documents WHERE id=?",
            (document_id,),
        ).fetchone()
        is not None
    )


def delete_document(conn: sqlite3.Connection, document_id: str) -> bool:
    conn.execute("DELETE FROM doc_chunks WHERE doc_id=?", (document_id,))
    conn.execute("DELETE FROM document_versions WHERE doc_id=?", (document_id,))
    return (
        conn.execute("DELETE FROM documents WHERE id=?", (document_id,)).rowcount
        > 0
    )


def list_document_versions(
    conn: sqlite3.Connection,
    document_id: str,
) -> tuple[sqlite3.Row, ...]:
    return tuple(
        conn.execute(
            "SELECT * FROM document_versions WHERE doc_id=? "
            "ORDER BY version_number DESC",
            (document_id,),
        ).fetchall()
    )


def current_version_id(
    conn: sqlite3.Connection,
    document_id: str,
) -> str | None:
    row = conn.execute(
        "SELECT current_version_id FROM documents WHERE id=?",
        (document_id,),
    ).fetchone()
    if row is None or row["current_version_id"] is None:
        return None
    return str(row["current_version_id"])
