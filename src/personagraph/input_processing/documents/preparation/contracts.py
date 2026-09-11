"""无持久副作用的文档准备结果合同。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from ...files import SourceFingerprint
from ..chunking import DocumentChunk
from ..contracts import DocumentElement, DocumentPageManifest


@dataclass(frozen=True, slots=True)
class DocumentPrepareFailure:
    """对未准入文档的安全且不含内容说明。"""

    reason: str
    facts: Mapping[str, object]

    def __post_init__(self) -> None:
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError("document prepare failure reason must not be empty")
        object.__setattr__(self, "facts", MappingProxyType(dict(self.facts)))

    def to_result(self) -> dict[str, object]:
        return {"ok": False, "reason": self.reason, **self.facts}


@dataclass(frozen=True, slots=True)
class PreparedDocumentIngest:
    """已准备好交给一次 Workspace authority 事务的精确临时 payload。"""

    canonical_path: str
    title: str
    mime: str
    elements: tuple[Mapping[str, object], ...]
    source_elements: tuple[DocumentElement, ...]
    document_chunks: tuple[DocumentChunk, ...]
    source_fingerprint: SourceFingerprint
    processor_fingerprint: str
    chunker_fingerprint: str
    processing_status: str
    processing_diagnostics: tuple[Mapping[str, object], ...]
    needs_vision: bool
    summary_preview: str
    page_manifest: DocumentPageManifest | None = None

    def __post_init__(self) -> None:
        for name in (
            "canonical_path",
            "title",
            "mime",
            "processor_fingerprint",
            "chunker_fingerprint",
            "processing_status",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must not be empty")
        source_elements = tuple(self.source_elements)
        if not self.elements or not source_elements or not self.document_chunks:
            raise ValueError("prepared document must contain admitted text and chunks")
        if any(
            not isinstance(element, DocumentElement)
            or not (element.text or "").strip()
            for element in source_elements
        ):
            raise ValueError(
                "prepared source_elements must contain exact text DocumentElement values"
            )
        source_ids = tuple(element.element_id for element in source_elements)
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("prepared source element identities must be unique")
        object.__setattr__(self, "source_elements", source_elements)
        object.__setattr__(
            self,
            "elements",
            tuple(MappingProxyType(dict(element)) for element in self.elements),
        )
        object.__setattr__(
            self,
            "processing_diagnostics",
            tuple(
                MappingProxyType(dict(diagnostic))
                for diagnostic in self.processing_diagnostics
            ),
        )


__all__ = ["DocumentPrepareFailure", "PreparedDocumentIngest"]
