"""聊天附件上传与列表 API 服务。"""

from __future__ import annotations

from typing import Any
from urllib.parse import unquote

from ...session.attachments.application import (
    SessionAttachmentQuotaExceeded,
    accept_upload,
)
from ...workspace.files.attachments import (
    MAX_ATTACHMENT_BYTES,
    MAX_ATTACHMENTS_PER_TURN,
    AttachmentTooLarge,
)
from ...session import store as session_store
from .common import require_session
from .errors import ApiError


# Header 必须使用 latin-1，因此客户端会对 UTF-8 文件名进行百分号编码。
FILENAME_HEADER = "X-Attachment-Filename"


def decode_upload_filename(raw_header: str | None) -> str:
    """解码客户端提供的名称，但不信任其中任何内容。"""

    if not raw_header:
        return "file"
    try:
        return unquote(raw_header, encoding="utf-8", errors="replace")
    except Exception:
        return "file"


def upload_attachment(
    session_id: str,
    *,
    payload: bytes,
    filename: str,
    declared_media_type: str | None,
) -> dict[str, Any]:
    """为 Session 存储一个上传文件，尚未绑定任何 Turn。"""

    require_session(session_id)
    try:
        accepted = accept_upload(
            session_id=session_id,
            raw_name=filename,
            declared_media_type=declared_media_type,
            payload=payload,
            store=session_store,
        )
    except AttachmentTooLarge as exc:
        raise ApiError(
            "ATTACHMENT_TOO_LARGE",
            "文件超过单个附件大小上限",
            status=413,
            details={"max_bytes": exc.limit_bytes, "size_bytes": exc.actual_bytes},
        ) from exc
    except SessionAttachmentQuotaExceeded as exc:
        raise ApiError(
            "SESSION_ATTACHMENT_QUOTA_EXCEEDED",
            "该会话的附件总量已达上限，请先清理",
            status=413,
            details={
                "max_bytes": exc.limit_bytes,
                "used_bytes": exc.used_bytes,
                "size_bytes": exc.incoming_bytes,
            },
        ) from exc
    return {"attachment": accepted.to_public_view()}


def delete_session_attachment(session_id: str, attachment_id: str) -> dict[str, Any]:
    """丢弃用户在发送前移除的暂存上传。"""

    require_session(session_id)
    removed = session_store.delete_unbound_attachment(
        session_id=session_id, attachment_id=attachment_id
    )
    if removed is None:
        raise ApiError(
            "ATTACHMENT_NOT_REMOVABLE",
            "附件不存在，或已随某轮消息发送而不可删除",
            status=404,
            details={"attachment_id": attachment_id},
        )
    # 文件身份属于 Project，删除未发送的 Session 绑定不得删除 Project 文件。
    return {"ok": True, "attachment_id": attachment_id, "deleted": True}


def list_session_attachments(session_id: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    """列出尚未发送的上传，使重新打开的窗口仍能显示它们。"""

    require_session(session_id)
    del params
    return {
        "attachments": [
            {
                "attachment_id": str(row["attachment_id"]),
                "name": str(row["original_name"]),
                "media_type": str(row["media_type"]),
                "size_bytes": int(row["size_bytes"]),
                "kind": str(row["kind"]),
                "created_at": str(row["created_at"]),
            }
            for row in session_store.list_unbound_attachments(session_id)
        ],
        "limits": {
            "max_attachment_bytes": MAX_ATTACHMENT_BYTES,
            "max_attachments_per_turn": MAX_ATTACHMENTS_PER_TURN,
        },
    }
