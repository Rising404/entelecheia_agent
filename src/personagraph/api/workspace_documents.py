"""文档元数据、挂载、清理和视图的应用操作。"""

from __future__ import annotations

import sqlite3
from typing import Any

__all__ = [
    "delete_document",
    "detach_document",
    "get_document",
    "list_documents",
    "patch_document",
]


def list_documents(session_id: str | None = None) -> list[dict[str, Any]]:
    from ..workspace.documents import application as docstore

    return [_doc_view(item) for item in docstore.list_documents(session_id=session_id)]


def get_document(doc_id: str) -> dict[str, Any] | None:
    from ..workspace.documents import application as docstore

    document = docstore.get_document(doc_id)
    return _doc_view(document) if document else None


def patch_document(doc_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    from ..workspace.documents import application as docstore

    if not docstore.get_document(doc_id):
        return {"ok": False, "reason": "document_not_found", "doc_id": doc_id}
    fields = {
        key: payload[key]
        for key in ("summary", "title", "tags")
        if key in payload
    }
    if not fields:
        return {"ok": False, "reason": "no_editable_fields"}
    docstore.edit(doc_id, **fields)
    document = docstore.get_document(doc_id)
    return {"ok": True, "document": _doc_view(document) if document else None}


def delete_document(doc_id: str) -> dict[str, Any]:
    from ..workspace.documents import application as docstore
    from ..retrieval.sources.events import SqliteDocumentIndexPort

    if not docstore.remove(
        doc_id,
        retrieval_data_version_provider=_document_retrieval_cleanup_data_version_id,
        document_index_port=SqliteDocumentIndexPort(),
    ):
        return {"ok": False, "reason": "document_not_found", "doc_id": doc_id}
    return {"ok": True, "doc_id": doc_id, "deleted": True}


def detach_document(doc_id: str, session_id: str) -> dict[str, Any]:
    from ..workspace.documents import application as docstore

    if not docstore.get_document(doc_id):
        return {"ok": False, "reason": "document_not_found", "doc_id": doc_id}
    if not docstore.detach(
        doc_id,
        session_id,
    ):
        return {
            "ok": False,
            "reason": "document_mount_not_found",
            "doc_id": doc_id,
            "session_id": session_id,
        }
    return {"ok": True, "doc_id": doc_id, "session_id": session_id, "detached": True}


def _document_retrieval_cleanup_data_version_id(
    conn: sqlite3.Connection,
    staging_candidate: str | None,
) -> str | None:
    """将源移除绑定到已发布或未发布的 D0 generation。

    没有活动 generation 表示不存在需要退役的已发布 retrieval 副本。但当前 Document 操作
    仍可能填充 BUILDING 或 READY 暂存 generation。源移除在与发布相同的项目文档写 fence
    下运行，因此 READY 证明可失效回 BUILDING，并在后续任何激活尝试前接收持久 PURGE。
    """

    from ..retrieval.sqlite_store import (
        RetrievalDataVersionRole,
        RetrievalDataVersionState,
        SqliteRetrievalCatalog,
    )

    catalog = SqliteRetrievalCatalog()
    active = catalog.active_data_version_in_transaction(conn)
    if active is not None:
        if (
            active.role is not RetrievalDataVersionRole.ACTIVE
            or active.state is not RetrievalDataVersionState.READY
        ):
            raise RuntimeError("active retrieval generation is not ready")
        if staging_candidate is not None and staging_candidate != active.id:
            raise RuntimeError(
                "current document is bound to a different retrieval generation"
            )
        return active.id
    if staging_candidate is None:
        return None

    target = catalog.get_data_version_in_transaction(conn, staging_candidate)
    if target is None or target.role is not RetrievalDataVersionRole.STAGING:
        raise RuntimeError("document staging retrieval generation is unavailable")
    if target.state is RetrievalDataVersionState.READY:
        target = catalog.invalidate_ready_staging_data_version_in_transaction(
            conn,
            staging_candidate,
        )
    if target.state is not RetrievalDataVersionState.BUILDING:
        raise RuntimeError("document staging retrieval generation is not writable")
    return target.id


def _doc_view(doc: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": doc.get("id"),
        "title": doc.get("title"),
        "mime": doc.get("mime"),
        "summary": doc.get("summary"),
        "tags": doc.get("tags"),
        "task_id": doc.get("task_id"),
        "n_chunks": doc.get("n_chunks"),
        "path": doc.get("path"),
        "added_at": doc.get("added_at"),
        "processing_status": doc.get("processing_status"),
        "diagnostics": doc.get("diagnostics"),
        "needs_vision": doc.get("needs_vision"),
        "source_sha256": doc.get("source_sha256"),
    }
