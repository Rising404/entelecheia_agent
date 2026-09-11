"""Workspace 持久 Document 权威对外使用的稳定值对象。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
from typing import Any, Protocol, runtime_checkable

from personagraph.input_processing.documents import DocumentPageManifest


DOCUMENT_CHUNK_CONTRACT_VERSION = 1
PERSISTABLE_PROCESSING_STATUSES = frozenset(
    {"complete", "partial", "legacy_unknown"}
)
PARTIAL_PROCESSING_DIAGNOSTIC_CODES = frozenset(
    {
        "fallback_reader_used",
        "page_empty",
        "page_needs_vision",
        "parser_partial",
    }
)


@runtime_checkable
class DocumentMountPort(Protocol):
    """Document application 使用的最小 Session 挂载与快照能力。"""

    def current_session_id(self) -> str: ...

    def mount_document(self, document_id: str, session_id: str) -> bool: ...

    def unmount_document(self, document_id: str, session_id: str) -> bool: ...

    def is_document_mounted(self, document_id: str, session_id: str) -> bool: ...

    def list_document_mounts(self, session_id: str) -> list[dict[str, Any]]: ...

    def create_document_retrieval_snapshot(
        self,
        *,
        snapshot_id: str,
        session_id: str,
        manifest_hash: str,
        manifest_json: str,
        reason: str,
        created_at: str,
    ) -> None: ...

    def get_document_retrieval_snapshot(
        self,
        snapshot_id: str,
        session_id: str,
    ) -> dict[str, Any] | None: ...


@dataclass(frozen=True, slots=True)
class MountedDocumentIndexBinding:
    """不含正文的当前 Document 块身份。"""

    doc_id: str
    chunk_id: str
    source_version_id: str
    indexed_content_hash: str
    producer_chunk_id: str | None = None


@dataclass(frozen=True, slots=True)
class MountedDocumentIndexBindingSnapshot:
    """先前打开的 Document revision 清单是否仍为当前版本。"""

    source_snapshot_is_current: bool
    bindings: tuple[MountedDocumentIndexBinding, ...] = ()
    binding_enumeration_complete: bool = True


@dataclass(frozen=True, slots=True)
class DocumentIngestCommitReceipt:
    """由一个调用方所有的收录事务生成的权威引用。"""

    document_id: str
    document_version_id: str
    retrieval_event_ids: tuple[str, ...]
    n_chunks: int
    deduped: bool
    reindexed: bool


@dataclass(frozen=True, slots=True)
class DocumentChunkPageBinding:
    """一个当前生产者块的内容不透明、精确页面来源。"""

    storage_chunk_id: str
    producer_chunk_id: str
    sequence: int
    element_ids: tuple[str, ...]
    source_pages: tuple[int, ...]
    content_sha256: str
    content_utf8_bytes: int
    content_json_utf8_bytes: int
    chunk_contract_version: int

    def __post_init__(self) -> None:
        if len(self.content_sha256) != 64 or any(
            character not in "0123456789abcdef"
            for character in self.content_sha256
        ):
            raise ValueError("chunk page binding requires a canonical content hash")
        if self.content_utf8_bytes < 1:
            raise ValueError("chunk page binding requires a positive content byte size")
        if self.content_json_utf8_bytes < self.content_utf8_bytes + 2:
            raise ValueError(
                "chunk page binding requires an exact JSON string byte size"
            )
        if self.chunk_contract_version != DOCUMENT_CHUNK_CONTRACT_VERSION:
            raise ValueError("chunk page binding uses an unsupported chunk contract")


@dataclass(frozen=True, slots=True)
class CurrentDocumentPageAuthority:
    """限定于一个已挂载当前来源版本的纯 Host 页面权威信息。"""

    session_id: str
    document_id: str
    document_version_id: str
    source_sha256: str
    processing_status: str
    processing_diagnostic_codes: tuple[str, ...]
    page_manifest: DocumentPageManifest
    chunks: tuple[DocumentChunkPageBinding, ...]


@dataclass(frozen=True, slots=True)
class CurrentDocumentResourceChunk:
    """来自已挂载当前 Document generation 的精确带类型块。"""

    storage_chunk_id: str
    producer_chunk_id: str
    sequence: int
    locator: str
    content: str
    content_sha256: str
    source_pages: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class CurrentDocumentResourceSnapshot:
    """供规划资源感知使用的纯 Host 有界内容快照。"""

    session_id: str
    document_id: str
    document_version_id: str
    source_sha256: str
    file_extension: str
    processing_status: str
    processing_diagnostic_codes: tuple[str, ...]
    total_chunk_count: int
    physical_page_count: int | None
    page_inventory_status: str
    chunks: tuple[CurrentDocumentResourceChunk, ...]
    truncated: bool


def validate_processing_coverage(
    processing_status: object,
    diagnostics: object,
) -> tuple[str, tuple[dict[str, object], ...] | None]:
    """校验持久 DocumentVersion 的状态与诊断组合。"""

    if (
        not isinstance(processing_status, str)
        or processing_status not in PERSISTABLE_PROCESSING_STATUSES
    ):
        raise ValueError(
            "processing_status must be complete, partial, or legacy_unknown"
        )
    if diagnostics is None:
        normalized: tuple[dict[str, object], ...] | None = None
    else:
        if isinstance(diagnostics, (str, bytes)) or not isinstance(
            diagnostics, Sequence
        ):
            raise ValueError("processing diagnostics must be a sequence of objects")
        if not all(isinstance(item, Mapping) for item in diagnostics):
            raise ValueError("processing diagnostics must be a sequence of objects")
        normalized = tuple(dict(item) for item in diagnostics)

    if processing_status == "legacy_unknown":
        if normalized is not None:
            raise ValueError(
                "legacy_unknown processing_status cannot carry diagnostics"
            )
        return processing_status, None
    if normalized is None:
        raise ValueError("known processing_status requires a diagnostics payload")
    if processing_status == "complete":
        if normalized:
            raise ValueError("complete processing_status cannot carry diagnostics")
        return processing_status, normalized
    if not normalized:
        raise ValueError("partial processing_status requires diagnostics")
    if any(
        diagnostic.get("code") not in PARTIAL_PROCESSING_DIAGNOSTIC_CODES
        for diagnostic in normalized
    ):
        raise ValueError(
            "partial processing diagnostics must use the admissible coverage-gap codes"
        )
    return processing_status, normalized


def processing_coverage_projection(
    processing_status: object,
    diagnostics_json: object,
) -> dict[str, Any]:
    """把一个持久覆盖分类投影为调用方可用的结构。"""

    if processing_status == "legacy_unknown":
        status, _ = validate_processing_coverage(
            processing_status,
            None if diagnostics_json is None else diagnostics_json,
        )
        return {
            "processing_status": status,
            "diagnostics": None,
            "needs_vision": None,
        }
    if not isinstance(diagnostics_json, str):
        diagnostics_payload: object = None
    else:
        try:
            diagnostics_payload = json.loads(diagnostics_json)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(
                "current document processing diagnostics are corrupt"
            ) from exc
    try:
        status, normalized = validate_processing_coverage(
            processing_status,
            diagnostics_payload,
        )
    except ValueError as exc:
        raise ValueError("current document processing coverage is corrupt") from exc
    assert normalized is not None
    diagnostics = list(normalized)
    return {
        "processing_status": status,
        "diagnostics": diagnostics,
        "needs_vision": any(
            diagnostic.get("code") == "page_needs_vision"
            for diagnostic in diagnostics
        ),
    }


def project_document_processing(document: dict[str, Any]) -> dict[str, Any]:
    """仅向应用调用方投影当前版本的持久覆盖分类。"""

    raw_diagnostics = document.pop("diagnostics_json", None)
    document.update(
        processing_coverage_projection(
            document.get("processing_status"),
            raw_diagnostics,
        )
    )
    return document
