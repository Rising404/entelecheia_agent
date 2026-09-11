"""Workspace 文档元数据 HTTP 用例。

文档解析和持久摄取 job 被刻意放在 ``document_ingest.py``；此 controller 只负责元数据和
挂载操作。
"""

from __future__ import annotations

from typing import Any

from .. import workspace_documents as document_service
from .common import required_str, require_workspace_session
from .errors import ApiError

__all__ = [
    "delete_document",
    "detach_document",
    "get_document",
    "list_documents",
    "patch_document",
]


def list_documents(params: dict[str, Any] | None = None) -> dict[str, Any]:
    session_id = required_str(params or {}, "session_id")
    require_workspace_session(session_id)
    return {"documents": document_service.list_documents(session_id=session_id)}


def get_document(
    doc_id: str,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    session_id = required_str(params or {}, "session_id")
    require_workspace_session(session_id)
    document = document_service.get_document(doc_id)
    if not document:
        raise ApiError("DOCUMENT_NOT_FOUND", "文档不存在", status=404, details={"doc_id": doc_id})
    return {"document": document}


def patch_document(doc_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    """只编辑用户所有的文档元数据。"""

    session_id = required_str(payload, "session_id")
    require_workspace_session(session_id)
    result = document_service.patch_document(doc_id, payload)
    if result.get("ok"):
        return {"document": result.get("document")}
    if result.get("reason") == "document_not_found":
        raise ApiError("DOCUMENT_NOT_FOUND", "文档不存在", status=404, details={"doc_id": doc_id})
    if result.get("reason") == "no_editable_fields":
        raise ApiError("NO_EDITABLE_FIELDS", "无可编辑字段（summary/title/tags）")
    raise ApiError("DOCUMENT_PATCH_FAILED", "文档更新失败", details={"doc_id": doc_id})


def delete_document(
    doc_id: str,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """删除文档索引，但不删除原始源文件。"""

    session_id = required_str(params or {}, "session_id")
    require_workspace_session(session_id)
    result = document_service.delete_document(doc_id)
    if result.get("ok"):
        return result
    if result.get("reason") == "document_not_found":
        raise ApiError("DOCUMENT_NOT_FOUND", "文档不存在", status=404, details={"doc_id": doc_id})
    raise ApiError("DOCUMENT_DELETE_FAILED", "文档删除失败", details={"doc_id": doc_id})


def detach_document(doc_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    """从 Session 卸载一个文档，但不删除其索引。"""

    session_id = required_str(payload, "session_id")
    require_workspace_session(session_id)
    result = document_service.detach_document(doc_id, session_id)
    if result.get("ok"):
        return result
    if result.get("reason") == "document_not_found":
        raise ApiError("DOCUMENT_NOT_FOUND", "文档不存在", status=404, details={"doc_id": doc_id})
    if result.get("reason") == "document_mount_not_found":
        raise ApiError(
            "DOCUMENT_MOUNT_NOT_FOUND",
            "文档未挂载到该会话",
            status=404,
            details={"doc_id": doc_id, "session_id": session_id},
        )
    raise ApiError("DOCUMENT_DETACH_FAILED", "文档解除挂载失败", details={"doc_id": doc_id})
