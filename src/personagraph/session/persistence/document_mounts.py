"""Session 所有的文档挂载与检索快照持久化。

项目数据库拥有文件字节、文档版本、块和派生检索 Unit。挂载则是 Session 能力：删除
一个 Session 时必须移除其访问权，但不能删除共享项目材料。
"""

from __future__ import annotations

import sqlite3
from typing import Any


def mount_document(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    document_id: str,
    mounted_at: str,
) -> bool:
    """授予此 Session 对一个项目所有 Document 的访问权。"""

    return bool(
        conn.execute(
            "INSERT OR IGNORE INTO doc_mounts "
            "(doc_id, session_id, mounted_at) VALUES (?, ?, ?)",
            (document_id, session_id, mounted_at),
        ).rowcount
    )


def unmount_document(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    document_id: str,
) -> bool:
    """撤销一个 Session 挂载，不触碰项目 Document。"""

    return bool(
        conn.execute(
            "DELETE FROM doc_mounts WHERE doc_id=? AND session_id=?",
            (document_id, session_id),
        ).rowcount
    )


def is_document_mounted(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    document_id: str,
) -> bool:
    return (
        conn.execute(
            "SELECT 1 FROM doc_mounts WHERE doc_id=? AND session_id=?",
            (document_id, session_id),
        ).fetchone()
        is not None
    )


def list_document_mounts(
    conn: sqlite3.Connection,
    *,
    session_id: str,
) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT doc_id, session_id, mounted_at FROM doc_mounts "
        "WHERE session_id=? ORDER BY mounted_at, doc_id",
        (session_id,),
    ).fetchall()
    return [dict(row) for row in rows]


def create_retrieval_snapshot(
    conn: sqlite3.Connection,
    *,
    snapshot_id: str,
    session_id: str,
    manifest_hash: str,
    manifest_json: str,
    reason: str,
    created_at: str,
) -> None:
    conn.execute(
        "INSERT INTO doc_retrieval_snapshots "
        "(id, session_id, manifest_hash, manifest_json, reason, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (
            snapshot_id,
            session_id,
            manifest_hash,
            manifest_json,
            reason,
            created_at,
        ),
    )


def get_retrieval_snapshot(
    conn: sqlite3.Connection,
    *,
    snapshot_id: str,
    session_id: str,
) -> dict[str, Any] | None:
    row = conn.execute(
        "SELECT * FROM doc_retrieval_snapshots WHERE id=? AND session_id=?",
        (snapshot_id, session_id),
    ).fetchone()
    return dict(row) if row is not None else None
