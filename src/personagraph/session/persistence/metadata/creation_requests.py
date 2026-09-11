"""共享 catalog 内的 Session 创建请求收据；不拥有连接、路径或 Session 内容。"""

from __future__ import annotations

import sqlite3


class CreationRequestConflict(ValueError):
    """同一请求 ID 被用于不同创建参数。"""


class CreationRequestInProgress(RuntimeError):
    """已有创建者，且尚未原子发布 Session locator。"""


def initialize_schema(conn: sqlite3.Connection) -> None:
    # project/session 共享 user_version=1；该可重复的增量表有自己的持久格式标记，
    # 不擅自推进 Project owner 使用的数据库版本。
    conn.execute("""
        CREATE TABLE IF NOT EXISTS session_creation_requests (
            request_id TEXT PRIMARY KEY,
            schema_version INTEGER NOT NULL DEFAULT 1 CHECK (schema_version = 1),
            request_hash TEXT NOT NULL CHECK (length(request_hash) = 64),
            session_id TEXT UNIQUE,
            created_at TEXT NOT NULL
        )
    """)
    columns = {row[1] for row in conn.execute("PRAGMA table_info(session_creation_requests)")}
    if columns != {"request_id", "schema_version", "request_hash", "session_id", "created_at"}:
        raise ValueError("unsupported Session creation request schema")
    if conn.execute(
        "SELECT 1 FROM session_creation_requests WHERE schema_version != 1 LIMIT 1"
    ).fetchone():
        raise ValueError("unsupported Session creation request version")


def reserve(
    conn: sqlite3.Connection, *, request_id: str, request_hash: str, occurred_at: str,
) -> str | None:
    """返回已发布 Session，或原子占用新请求；未完成请求不采用超时接管。"""
    row = conn.execute(
        "SELECT request_hash, session_id FROM session_creation_requests WHERE request_id=?",
        (request_id,),
    ).fetchone()
    if row is not None:
        if row["request_hash"] != request_hash:
            raise CreationRequestConflict("Session creation parameters differ")
        if row["session_id"] is None:
            raise CreationRequestInProgress("Session creation result is not published")
        return str(row["session_id"])
    conn.execute(
        "INSERT INTO session_creation_requests (request_id, request_hash, created_at) VALUES (?, ?, ?)",
        (request_id, request_hash, occurred_at),
    )
    return None


def publish(conn: sqlite3.Connection, *, request_id: str, session_id: str) -> None:
    """必须与 Session locator INSERT 同事务，不能先发布再补写收据。"""
    changed = conn.execute(
        "UPDATE session_creation_requests SET session_id=? WHERE request_id=? AND session_id IS NULL",
        (session_id, request_id),
    ).rowcount
    if changed != 1:
        raise CreationRequestConflict("Session creation request is not reserved")


def release_unpublished(conn: sqlite3.Connection, *, request_id: str) -> bool:
    # 成功收据保留到 Session 清除之后，避免旧请求把删除的会话重新创建出来。
    return conn.execute(
        "DELETE FROM session_creation_requests WHERE request_id=? AND session_id IS NULL",
        (request_id,),
    ).rowcount == 1
