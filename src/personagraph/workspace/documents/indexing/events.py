"""在当前事务内把 Document 状态变化投影给索引端口。"""

from __future__ import annotations

from collections.abc import Callable, Sequence
import sqlite3

from .contracts import (
    DocumentIndexChunk,
    DocumentIndexEnqueueRequest,
    DocumentIndexEventKind,
)
from .ports import DocumentIndexPort


def validate_data_version(data_version: str | None) -> None:
    if data_version is not None and not data_version.strip():
        raise ValueError("retrieval_data_version must not be empty when provided")


def validate_data_version_selection(
    data_version: str | None,
    provider: Callable[[sqlite3.Connection, str | None], str | None] | None,
) -> None:
    validate_data_version(data_version)
    if provider is not None and not callable(provider):
        raise TypeError("retrieval_data_version_provider must be callable")
    if data_version is not None and provider is not None:
        raise ValueError(
            "provide retrieval_data_version or retrieval_data_version_provider, not both"
        )


def resolve_data_version(
    conn: sqlite3.Connection,
    document_id: str,
    data_version: str | None,
    provider: Callable[[sqlite3.Connection, str | None], str | None] | None,
) -> str | None:
    candidate = current_ingest_data_version(conn, document_id)
    selected = provider(conn, candidate) if provider is not None else data_version
    validate_data_version(selected)
    return selected


def current_ingest_data_version(
    conn: sqlite3.Connection,
    document_id: str,
) -> str | None:
    current = conn.execute(
        "SELECT current_version_id FROM documents WHERE id=?",
        (document_id,),
    ).fetchone()
    if current is None or current["current_version_id"] is None:
        return None
    rows = conn.execute(
        "SELECT DISTINCT retrieval_data_version FROM document_ingest_jobs "
        "WHERE document_id=? AND document_version_id=? "
        "AND retrieval_data_version IS NOT NULL ORDER BY retrieval_data_version",
        (document_id, str(current["current_version_id"])),
    ).fetchall()
    candidates = tuple(str(row["retrieval_data_version"]) for row in rows)
    if len(candidates) > 1:
        raise RuntimeError("current document is bound to multiple retrieval generations")
    return candidates[0] if candidates else None


def enqueue_document_upserts(
    conn: sqlite3.Connection,
    *,
    doc_id: str,
    retrieval_data_version: str | None,
    index_port: DocumentIndexPort | None,
    session_ids: tuple[str, ...] | None = None,
) -> tuple[str, ...]:
    """重放当前 Document 版本的确定性索引绑定。

    UPSERT 的事件身份可跨崩溃重建，因此其时间也必须来自同一不可变来源版本，不能使用
    每次尝试时的墙上时钟。生命周期转换使用单独的入口并保留真实转换时间。
    """

    return _enqueue_document_index_events(
        conn,
        doc_id=doc_id,
        kind=DocumentIndexEventKind.UPSERT,
        retrieval_data_version=retrieval_data_version,
        occurred_at=_current_version_activated_at(conn, doc_id),
        index_port=index_port,
        session_ids=session_ids,
    )


def enqueue_document_lifecycle_events(
    conn: sqlite3.Connection,
    *,
    doc_id: str,
    kind: str | DocumentIndexEventKind,
    retrieval_data_version: str | None,
    occurred_at: str,
    index_port: DocumentIndexPort | None,
    session_ids: tuple[str, ...] | None = None,
) -> tuple[str, ...]:
    """投影一次非幂等 Document 生命周期转换。"""

    event_kind = DocumentIndexEventKind(kind)
    if event_kind is DocumentIndexEventKind.UPSERT:
        raise ValueError("use enqueue_document_upserts for deterministic bindings")
    return _enqueue_document_index_events(
        conn,
        doc_id=doc_id,
        kind=event_kind,
        retrieval_data_version=retrieval_data_version,
        occurred_at=occurred_at,
        index_port=index_port,
        session_ids=session_ids,
    )


def _enqueue_document_index_events(
    conn: sqlite3.Connection,
    *,
    doc_id: str,
    kind: DocumentIndexEventKind,
    retrieval_data_version: str | None,
    occurred_at: str,
    index_port: DocumentIndexPort | None,
    session_ids: tuple[str, ...] | None,
) -> tuple[str, ...]:
    """把受影响的当前块交给注入端口；Workspace 不构造索引身份。"""

    if retrieval_data_version is None:
        return ()
    if index_port is None:
        raise RuntimeError(
            "retrieval_data_version requires an injected DocumentIndexPort"
        )
    rows = conn.execute(
        "SELECT id, source_version_id, content, producer_chunk_id "
        "FROM doc_chunks WHERE doc_id=? ORDER BY seq, id",
        (doc_id,),
    ).fetchall()
    # 带类型 Project 块的索引身份不依赖挂载。后台共享准备可以没有 Session port。
    if all(row["producer_chunk_id"] is not None for row in rows):
        legacy_session_ids: tuple[str, ...] = ()
        typed_identity_session_id: str | None = None
    elif session_ids is None:
        from ..mounting import current_mount_session_id, is_mounted_in_connection

        current_session_id = current_mount_session_id()
        legacy_session_ids = (
            (current_session_id,)
            if is_mounted_in_connection(conn, doc_id, current_session_id)
            else ()
        )
        typed_identity_session_id = current_session_id
    else:
        legacy_session_ids = tuple(dict.fromkeys(session_ids))
        typed_identity_session_id = session_ids[0] if session_ids else None
    request = DocumentIndexEnqueueRequest(
        kind=kind,
        document_id=doc_id,
        data_version_id=retrieval_data_version,
        occurred_at=occurred_at,
        legacy_session_ids=legacy_session_ids,
        typed_identity_session_id=typed_identity_session_id,
        chunks=tuple(
            DocumentIndexChunk(
                storage_chunk_id=str(row["id"]),
                source_version_id=str(row["source_version_id"]),
                content=str(row["content"]),
                producer_chunk_id=(
                    str(row["producer_chunk_id"])
                    if row["producer_chunk_id"] is not None
                    else None
                ),
            )
            for row in rows
        ),
    )
    return index_port.enqueue_in_transaction(conn, request)


def _current_version_activated_at(
    conn: sqlite3.Connection,
    doc_id: str,
) -> str:
    row = conn.execute(
        "SELECT v.activated_at FROM documents AS d "
        "JOIN document_versions AS v ON v.id=d.current_version_id "
        "WHERE d.id=? AND v.status='active'",
        (doc_id,),
    ).fetchone()
    if row is None:
        raise RuntimeError("document has no active version to index")
    activated_at = str(row["activated_at"] or "").strip()
    if not activated_at:
        raise RuntimeError("active document version has no activation timestamp")
    return activated_at


def capture_event_ids(
    output: list[str] | None,
    event_ids: Sequence[str],
) -> None:
    if output is not None:
        output.extend(event_ids)
