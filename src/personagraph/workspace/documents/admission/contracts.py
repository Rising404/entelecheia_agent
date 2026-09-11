"""Turn 输入文件准入 Document 权威时使用的稳定合同。"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol

from personagraph.input_processing.documents import (
    DocumentChunk,
    DocumentPageManifest,
    DocumentPrepareFailure,
    PreparedDocumentIngest,
)
from personagraph.input_processing.files import SourceFingerprint


class AttachmentDocumentGapReason(StrEnum):
    """目标输入未成为 Document 权威状态的封闭 Host 原因。"""

    MALFORMED_BINDING = "malformed_binding"
    UNTRUSTED_ORIGIN = "untrusted_origin"
    PATH_OUTSIDE_SESSION_INPUT = "path_outside_session_input"
    SENSITIVE_PATH_DENIED = "sensitive_path_denied"
    TYPE_BINDING_MISMATCH = "type_binding_mismatch"
    CONTENT_RECEIPT_MISMATCH = "content_receipt_mismatch"
    UNSUPPORTED_FORMAT = "unsupported_format"
    UNSUPPORTED_LEGACY_OFFICE = "unsupported_legacy_office"
    PROCESSING_INCOMPLETE = "processing_incomplete"
    SOURCE_CHANGED = "source_changed"
    TOO_LARGE = "too_large"
    COMMIT_FAILED = "commit_failed"
    MOUNT_AUTHORITY_MISSING = "mount_authority_missing"


@dataclass(frozen=True, slots=True)
class AttachmentDocumentGap:
    """可安全保留在 Runtime 边界的无路径诊断。"""

    attachment_id: str
    reason: AttachmentDocumentGapReason
    detail_code: str | None = None

    def __post_init__(self) -> None:
        if not self.attachment_id or len(self.attachment_id) > 160:
            raise ValueError("attachment gap requires a bounded attachment id")
        if self.detail_code is not None and (
            not self.detail_code or len(self.detail_code) > 160
        ):
            raise ValueError("attachment gap detail code must be bounded")


@dataclass(frozen=True, slots=True)
class AttachmentDocumentMount:
    """一个 Turn 输入支持的已挂载 Document generation receipt。"""

    attachment_id: str
    document_id: str
    document_version_id: str
    source_sha256: str
    processing_status: str
    processing_diagnostic_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in ("attachment_id", "document_id", "document_version_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value or len(value) > 200:
                raise ValueError(f"{name} must be a bounded identity")
        if len(self.source_sha256) != 64 or any(
            character not in "0123456789abcdef"
            for character in self.source_sha256
        ):
            raise ValueError("mount receipt requires a canonical source hash")
        if self.processing_status not in {"complete", "partial"}:
            raise ValueError("mount receipt requires typed processing coverage")


@dataclass(frozen=True, slots=True)
class AttachmentDocumentBridgeResult:
    """绑定到某 Turn 的所有目标输入之完整结果。"""

    session_id: str
    turn_id: str
    mounts: tuple[AttachmentDocumentMount, ...] = ()
    ignored_attachment_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class PreparedAttachmentVisualIngest:
    """普通文本摄取会拒绝的严格纯视觉 generation。"""

    canonical_path: str
    title: str
    mime: str
    elements: tuple[Mapping[str, object], ...]
    document_chunks: tuple[DocumentChunk, ...]
    source_fingerprint: SourceFingerprint
    processor_fingerprint: str
    chunker_fingerprint: str
    processing_status: str
    processing_diagnostics: tuple[Mapping[str, object], ...]
    needs_vision: bool
    summary_preview: str
    page_manifest: DocumentPageManifest

    def __post_init__(self) -> None:
        if self.elements or self.document_chunks or self.summary_preview:
            raise ValueError("visual-only ingest cannot invent text or chunks")
        if self.processing_status != "partial" or not self.needs_vision:
            raise ValueError("visual-only ingest must retain partial visual coverage")
        if not any(
            unit.requires_visual_read
            for page in self.page_manifest.pages
            for unit in page.nontext_units
        ):
            raise ValueError("visual-only ingest requires a renderable visual unit")
        if any(page.text_element_ids for page in self.page_manifest.pages):
            raise ValueError("visual-only ingest cannot omit page text authority")


class AttachmentDocumentAuthorityError(RuntimeError):
    """至少一个目标输入无法安全进入 Document 权威状态。"""

    code = "attachment_document_authority_gap"

    def __init__(self, gaps: Sequence[AttachmentDocumentGap]) -> None:
        admitted = tuple(gaps)
        if not admitted:
            raise ValueError("attachment document authority error requires gaps")
        super().__init__(self.code)
        self.gaps = admitted


class TurnAttachmentStorePort(Protocol):
    def list_turn_attachments(
        self,
        session_id: str,
        turn_id: str,
    ) -> list[dict[str, Any]]: ...


class MountedDocumentStorePort(Protocol):
    def ingest(self, *args: Any, **kwargs: Any) -> dict[str, Any]: ...

    def mounted_docs(self, session_id: str | None) -> list[dict[str, Any]]: ...

    def get_mounted_current_document_resource_snapshot(
        self,
        doc_id: str,
        *,
        session_id: str,
        maximum_chunks: int,
    ) -> object | None: ...


PreparedAttachmentAuthority = PreparedDocumentIngest | PreparedAttachmentVisualIngest
PrepareDocument = Callable[[str], PreparedAttachmentAuthority | DocumentPrepareFailure]


__all__ = [
    "AttachmentDocumentAuthorityError",
    "AttachmentDocumentBridgeResult",
    "AttachmentDocumentGap",
    "AttachmentDocumentGapReason",
    "AttachmentDocumentMount",
    "MountedDocumentStorePort",
    "PrepareDocument",
    "PreparedAttachmentAuthority",
    "PreparedAttachmentVisualIngest",
    "TurnAttachmentStorePort",
]
