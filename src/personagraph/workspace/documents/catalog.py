"""Document 目录、元数据编辑与删除应用能力。"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
import sqlite3
from typing import Any

from personagraph.workspace.storage.context import connect_current, initialize_current

from .contracts import project_document_processing
from .indexing.events import (
    enqueue_document_lifecycle_events,
    resolve_data_version,
    validate_data_version_selection,
)
from .indexing.ports import DocumentIndexPort
from .mounting import detach_from_current_session, mounted_docs
from .storage import metadata as repository


def list_documents(session_id: str | None = None) -> list[dict[str, Any]]:
    """列出全部文档，或仅列出挂载到指定 Session 的文档。"""

    initialize_current()
    if session_id is not None:
        return mounted_docs(session_id)
    with connect_current() as conn:
        rows = repository.list_documents(conn)
    return [project_document_processing(dict(row)) for row in rows]


def get_document(document_id: str) -> dict[str, Any] | None:
    initialize_current()
    with connect_current() as conn:
        row = repository.get_document(conn, document_id)
    return project_document_processing(dict(row)) if row else None


def edit(document_id: str, **fields: Any) -> bool:
    initialize_current()
    with connect_current() as conn:
        return repository.edit_document(conn, document_id, fields)


def remove(
    document_id: str,
    *,
    retrieval_data_version: str | None = None,
    retrieval_data_version_provider: (
        Callable[[sqlite3.Connection, str | None], str | None] | None
    ) = None,
    document_index_port: DocumentIndexPort | None = None,
) -> bool:
    """删除项目 Document 权威；交互调用方负责显式确认。"""

    validate_data_version_selection(
        retrieval_data_version,
        retrieval_data_version_provider,
    )
    initialize_current()
    with connect_current() as conn:
        conn.execute("BEGIN IMMEDIATE")
        if not repository.document_exists(conn, document_id):
            return False
        target = resolve_data_version(
            conn,
            document_id,
            retrieval_data_version,
            retrieval_data_version_provider,
        )
        enqueue_document_lifecycle_events(
            conn,
            doc_id=document_id,
            kind="purge",
            retrieval_data_version=target,
            occurred_at=_now_utc(),
            index_port=document_index_port,
        )
        removed = repository.delete_document(conn, document_id)
    if removed:
        detach_from_current_session(document_id)
    return removed


def list_document_versions(document_id: str) -> list[dict[str, Any]]:
    initialize_current()
    with connect_current() as conn:
        rows = repository.list_document_versions(conn, document_id)
    return [dict(row) for row in rows]


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()
