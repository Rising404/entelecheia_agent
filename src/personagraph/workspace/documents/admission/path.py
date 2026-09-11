"""把纯文档准备结果交给 Workspace Document 权威。"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Protocol

from personagraph.input_processing.documents.chunking import (
    ChunkingProfile,
    DocumentChunk,
)
from personagraph.input_processing.documents.contracts import (
    DocumentElement,
    DocumentPageManifest,
)
from personagraph.input_processing.documents.preparation import (
    DocumentPrepareFailure,
    prepare_document_path,
)
from personagraph.input_processing.files import SourceFingerprint
from personagraph.workspace.documents.indexing import DocumentIndexPort


class DocumentStorePort(Protocol):
    """路径准入用例所需的最小 Document application 表面。"""

    def ingest(
        self,
        path: str,
        title: str,
        mime: str,
        elements: list[dict[str, object]],
        session_id: str | None = None,
        summary: str = "",
        *,
        source_fingerprint: SourceFingerprint | None = None,
        processor_fingerprint: str | None = None,
        source_elements: tuple[DocumentElement, ...] | None = None,
        document_chunks: tuple[DocumentChunk, ...] | None = None,
        chunker_fingerprint: str | None = None,
        processing_status: str | None = None,
        processing_diagnostics: tuple[dict[str, object], ...] | None = None,
        page_manifest: DocumentPageManifest | None = None,
        retrieval_data_version: str | None = None,
        file_id: str | None = None,
        file_version_id: str | None = None,
        document_index_port: DocumentIndexPort | None = None,
    ) -> dict[str, Any]: ...

    def mounted_docs(self, session_id: str | None) -> list[dict[str, Any]]: ...

    def edit(self, doc_id: str, **fields: Any) -> bool: ...


SummaryBuilder = Callable[[str, str], str]


def ingest_document_path(
    path_str: str,
    *,
    session_id: str | None,
    with_summary: bool,
    document_store: DocumentStorePort,
    summarize: SummaryBuilder | None = None,
    chunking_profile: ChunkingProfile | None = None,
    retrieval_data_version: str | None = None,
    file_id: str | None = None,
    file_version_id: str | None = None,
    document_index_port: DocumentIndexPort | None = None,
) -> dict[str, Any]:
    """准备一个稳定来源，并把可准入结果交给注入的 Workspace store。"""

    prepared = prepare_document_path(path_str, chunking_profile=chunking_profile)
    if isinstance(prepared, DocumentPrepareFailure):
        return prepared.to_result()

    stored = document_store.ingest(
        prepared.canonical_path,
        prepared.title,
        prepared.mime,
        [dict(element) for element in prepared.elements],
        processor_fingerprint=prepared.processor_fingerprint,
        source_elements=prepared.source_elements,
        document_chunks=prepared.document_chunks,
        chunker_fingerprint=prepared.chunker_fingerprint,
        processing_status=prepared.processing_status,
        processing_diagnostics=tuple(
            dict(diagnostic) for diagnostic in prepared.processing_diagnostics
        ),
        page_manifest=prepared.page_manifest,
        session_id=session_id,
        source_fingerprint=prepared.source_fingerprint,
        retrieval_data_version=retrieval_data_version,
        file_id=file_id,
        file_version_id=file_version_id,
        document_index_port=document_index_port,
    )
    if (
        with_summary
        and not stored["deduped"]
        and not stored.get("reindexed", False)
        and summarize is not None
    ):
        summary = _safe_summary(
            summarize,
            prepared.title,
            prepared.summary_preview,
        )
        if summary:
            document_store.edit(stored["doc_id"], summary=summary)

    document = _mounted_document(
        document_store.mounted_docs(session_id),
        stored["doc_id"],
    )
    return {
        "ok": True,
        **stored,
        "title": (document or {}).get("title", prepared.title),
        "summary": (document or {}).get("summary", ""),
        "processing_status": prepared.processing_status,
        "needs_vision": prepared.needs_vision,
        "diagnostics": [
            dict(diagnostic) for diagnostic in prepared.processing_diagnostics
        ],
    }


def _safe_summary(summarize: SummaryBuilder, title: str, preview: str) -> str:
    """可选 enrichment 绝不能使已经持久化的摄取失效。"""

    try:
        return str(summarize(title, preview) or "")[:200]
    except Exception:
        return ""


def _mounted_document(
    documents: list[dict[str, Any]],
    doc_id: str,
) -> Mapping[str, Any] | None:
    return next((document for document in documents if document.get("id") == doc_id), None)


__all__ = ["DocumentStorePort", "ingest_document_path"]
