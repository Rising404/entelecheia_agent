"""Workspace Document 与派生索引之间的中性请求合同。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class DocumentIndexEventKind(StrEnum):
    UPSERT = "upsert"
    TRASH = "trash"
    RESTORE = "restore"
    PURGE = "purge"


class DocumentIndexAuditError(RuntimeError):
    """索引事件或覆盖观察无法证明其声明的 Document 绑定。"""


@dataclass(frozen=True, slots=True)
class DocumentIndexChunk:
    """一次索引事件所绑定的不可变当前块。"""

    storage_chunk_id: str
    source_version_id: str
    content: str
    producer_chunk_id: str | None = None

    def __post_init__(self) -> None:
        if not self.storage_chunk_id.strip() or not self.source_version_id.strip():
            raise ValueError("document index chunk identity must not be empty")
        if self.producer_chunk_id is not None and not self.producer_chunk_id.strip():
            raise ValueError("producer_chunk_id must be non-empty when provided")


@dataclass(frozen=True, slots=True)
class DocumentIndexEnqueueRequest:
    """在调用方权威事务内写入的一批 Document 索引生命周期事件。"""

    kind: DocumentIndexEventKind
    document_id: str
    data_version_id: str
    occurred_at: str
    legacy_session_ids: tuple[str, ...]
    typed_identity_session_id: str | None
    chunks: tuple[DocumentIndexChunk, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.kind, DocumentIndexEventKind):
            raise TypeError("kind must be DocumentIndexEventKind")
        for name in ("document_id", "data_version_id", "occurred_at"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must not be empty")
        if any(
            not isinstance(value, str) or not value.strip()
            for value in self.legacy_session_ids
        ):
            raise ValueError(
                "legacy_session_ids must contain only non-empty strings"
            )
        if len(set(self.legacy_session_ids)) != len(self.legacy_session_ids):
            raise ValueError("legacy_session_ids must not contain duplicates")
        if (
            self.typed_identity_session_id is not None
            and not self.typed_identity_session_id.strip()
        ):
            raise ValueError(
                "typed_identity_session_id must be non-empty when provided"
            )


@dataclass(frozen=True, slots=True)
class DocumentIndexCoverageSnapshot:
    """索引端对一个 Document 版本及其已链接事件的精确观察。"""

    binding_digest: str
    binding_count: int
    mapped_event_count: int
    applied_event_count: int

    def __post_init__(self) -> None:
        if len(self.binding_digest) != 64 or any(
            character not in "0123456789abcdef"
            for character in self.binding_digest
        ):
            raise ValueError("binding_digest must be a lowercase SHA-256 digest")
        for name in (
            "binding_count",
            "mapped_event_count",
            "applied_event_count",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.applied_event_count > self.mapped_event_count:
            raise ValueError("applied_event_count cannot exceed mapped_event_count")


__all__ = [
    "DocumentIndexAuditError",
    "DocumentIndexChunk",
    "DocumentIndexCoverageSnapshot",
    "DocumentIndexEnqueueRequest",
    "DocumentIndexEventKind",
]
