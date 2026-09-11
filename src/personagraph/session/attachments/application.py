"""接收单个聊天附件的应用服务。

此处顺序很重要且经过刻意安排：任何字节落盘前检查资源边界，先把 payload 注册为
Project 文件/版本，再创建 Session 附件绑定。两步之间发生崩溃只会留下仍属于 Project 的
用户上传文件，不会产生指向不存在内容的 Session 记录。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol
from uuid import uuid4

from .contracts import AttachmentKind, AttachmentOrigin
from personagraph.workspace.files.attachments import (
    MAX_ATTACHMENT_BYTES,
    MAX_SESSION_ATTACHMENT_BYTES,
    AttachmentTooLarge,
    store_attachment,
)


class AttachmentStorePort(Protocol):
    """服务所需的持久化能力，由 API 组合根提供。"""

    def create_attachment(self, **kwargs: Any) -> dict[str, Any]: ...
    def session_attachment_total_bytes(self, session_id: str) -> int: ...


class SessionAttachmentQuotaExceeded(ValueError):
    """Session 已持有允许的最大附件材料量。"""

    def __init__(self, limit_bytes: int, used_bytes: int, incoming_bytes: int) -> None:
        super().__init__("session attachment quota exceeded")
        self.limit_bytes = limit_bytes
        self.used_bytes = used_bytes
        self.incoming_bytes = incoming_bytes


@dataclass(frozen=True)
class AcceptedAttachment:
    attachment_id: str
    original_name: str
    media_type: str
    declared_media_type: str | None
    size_bytes: int
    content_hash: str
    kind: AttachmentKind

    def to_public_view(self) -> dict[str, Any]:
        """客户端发送前在附件卡片上显示的数据形状。"""

        return {
            "attachment_id": self.attachment_id,
            "name": self.original_name,
            "media_type": self.media_type,
            "size_bytes": self.size_bytes,
            "kind": self.kind.value,
            # 可读性是根据实时预算逐 Turn 做出的决定，因此卡片只承诺该类型原则上是否可读。
            "readable": self.kind in {AttachmentKind.TEXT, AttachmentKind.DOCUMENT, AttachmentKind.IMAGE},
        }


def accept_upload(
    *,
    session_id: str,
    raw_name: str,
    declared_media_type: str | None,
    payload: bytes,
    store: AttachmentStorePort,
    max_bytes: int | None = None,
    session_quota_bytes: int | None = None,
) -> AcceptedAttachment:
    """存储一次上传，并将其登记为尚未绑定任何 Turn。

    不拒绝任何类型：不可读文件仍值得保留，以便告知模型它的存在，也使后续转录或 OCR 能力
    可以读取它，而无须用户重新上传。
    """

    # 在调用时解析，而不是作为会在导入时绑定的参数默认值；后者会导致测试和运维人员无法
    # patch 这些边界。
    max_bytes = MAX_ATTACHMENT_BYTES if max_bytes is None else max_bytes
    session_quota_bytes = (
        MAX_SESSION_ATTACHMENT_BYTES if session_quota_bytes is None else session_quota_bytes
    )
    if len(payload) > max_bytes:
        raise AttachmentTooLarge(max_bytes, len(payload))
    used = store.session_attachment_total_bytes(session_id)
    if used + len(payload) > session_quota_bytes:
        raise SessionAttachmentQuotaExceeded(session_quota_bytes, used, len(payload))

    attachment_id = f"att_{uuid4().hex}"
    stored = store_attachment(
        session_id=session_id,
        attachment_id=attachment_id,
        raw_name=raw_name,
        payload=payload,
        max_bytes=max_bytes,
    )
    attachment_record = dict(
        attachment_id=attachment_id,
        session_id=session_id,
        origin=AttachmentOrigin.USER_UPLOAD.value,
        original_name=stored.original_name,
        stored_rel_path=stored.stored_rel_path,
        media_type=stored.detected.media_type,
        # 仅为保持声明与内容不匹配情况可审计而保留。
        declared_media_type=declared_media_type,
        size_bytes=stored.size_bytes,
        content_hash=stored.content_hash,
        kind=stored.detected.kind.value,
    )
    attachment_record.update(
        project_id=stored.project_id,
        file_id=stored.file_id,
        file_version_id=stored.file_version_id,
    )
    store.create_attachment(**attachment_record)
    return AcceptedAttachment(
        attachment_id=attachment_id,
        original_name=stored.original_name,
        media_type=stored.detected.media_type,
        declared_media_type=declared_media_type,
        size_bytes=stored.size_bytes,
        content_hash=stored.content_hash,
        kind=stored.detected.kind,
    )
