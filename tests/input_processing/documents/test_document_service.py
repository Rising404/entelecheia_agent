from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from personagraph.input_processing.documents import (
    DocumentChunk,
    DocumentPrepareFailure,
    PreparedDocumentIngest,
    prepare_document_path,
)
from personagraph.input_processing.files import MAX_DOCUMENT_FILE_BYTES
from personagraph.workspace.documents.admission import ingest_document_path


class _DocumentStore:
    """用小型替身证明输入服务需要的是端口，而不是 memory.sqlite。"""

    def __init__(self) -> None:
        self.ingest_calls: list[dict[str, Any]] = []
        self.documents = [{"id": "doc_1", "title": "brief", "summary": ""}]
        self.ingest_result = {
            "doc_id": "doc_1",
            "n_chunks": 1,
            "deduped": False,
            "reindexed": False,
        }

    def ingest(self, path: str, title: str, mime: str, elements: list[dict[str, object]], **kwargs: Any) -> dict[str, Any]:
        self.ingest_calls.append({
            "path": path,
            "title": title,
            "mime": mime,
            "elements": elements,
            **kwargs,
        })
        return dict(self.ingest_result)

    def mounted_docs(self, session_id: str | None) -> list[dict[str, Any]]:
        return self.documents if session_id == "session_1" else []

    def edit(self, doc_id: str, **fields: Any) -> bool:
        for document in self.documents:
            if document["id"] == doc_id:
                document.update(fields)
                return True
        return False


def test_prepare_document_path_is_a_write_free_exact_source_checkpoint(tmp_path):
    path = tmp_path / "brief.md"
    path.write_text("# 方案\n\n先完成文档摄取。", encoding="utf-8")

    prepared = prepare_document_path(str(path))

    assert isinstance(prepared, PreparedDocumentIngest)
    assert prepared.canonical_path == str(path.resolve())
    assert prepared.source_fingerprint.sha256
    assert prepared.processor_fingerprint == "plain_text@4"
    assert prepared.processing_status == "complete"
    assert prepared.processing_diagnostics == ()
    assert prepared.document_chunks
    assert prepared.summary_preview


def test_prepare_document_path_rejects_oversized_source_during_bounded_hash(tmp_path):
    path = tmp_path / "oversized.pdf"
    with path.open("wb") as stream:
        stream.truncate(MAX_DOCUMENT_FILE_BYTES + 1)

    prepared = prepare_document_path(str(path))

    assert isinstance(prepared, DocumentPrepareFailure)
    assert prepared.reason == "too_large"


def test_pdf_page_manifest_crosses_the_service_port_without_projection_loss(
    text_pdf, monkeypatch
):
    from personagraph.input_processing.documents import DocumentPageManifest
    from personagraph.input_processing.documents import readers

    monkeypatch.setenv(readers.ENGINE_ENV_VAR, readers.NATIVE_ENGINE)
    prepared = prepare_document_path(str(text_pdf))
    assert isinstance(prepared, PreparedDocumentIngest)
    assert isinstance(prepared.page_manifest, DocumentPageManifest)

    store = _DocumentStore()
    response = ingest_document_path(
        str(text_pdf),
        session_id="session_1",
        with_summary=False,
        document_store=store,
    )

    assert response["ok"] is True
    assert store.ingest_calls[0]["page_manifest"] == prepared.page_manifest


@pytest.mark.parametrize(
    ("name", "expected"),
    (("notes.md", "plain_text@4"), ("paper.pdf", "pdfplumber@8")),
)
def test_native_processor_recipe_is_known_without_parsing_a_document(name, expected):
    from personagraph.input_processing.documents.readers import (
        NATIVE_ENGINE,
        configured_processor_fingerprint,
    )

    assert str(
        configured_processor_fingerprint(Path(name), engine=NATIVE_ENGINE)
    ) == expected


def test_public_intake_service_hands_complete_typed_result_to_an_injected_store(tmp_path):
    path = tmp_path / "brief.md"
    path.write_text("# 方案\n\n先完成文档摄取。", encoding="utf-8")
    store = _DocumentStore()

    result = ingest_document_path(
        str(path),
        session_id="session_1",
        with_summary=True,
        document_store=store,
        summarize=lambda title, preview: f"{title}: {preview[:8]}",
    )

    assert result["ok"] is True
    assert result["doc_id"] == "doc_1"
    assert result["summary"].startswith("brief:")
    assert len(store.ingest_calls) == 1
    call = store.ingest_calls[0]
    assert call["processor_fingerprint"] == "plain_text@4"
    assert call["source_fingerprint"].sha256
    assert call["elements"]
    assert "chunks" not in call
    assert isinstance(call["document_chunks"], tuple)
    assert call["document_chunks"]
    assert all(isinstance(chunk, DocumentChunk) for chunk in call["document_chunks"])
    typed_chunk = call["document_chunks"][0]
    assert typed_chunk.chunk_id.startswith("ch_")
    assert typed_chunk.span.start.char_range is not None
    assert typed_chunk.section_path == ("方案",)
    assert typed_chunk.element_ids
    assert call["processing_status"] == "complete"
    assert call["processing_diagnostics"] == ()


def test_public_intake_service_admits_partial_text_with_explicit_coverage(tmp_path, monkeypatch):
    from personagraph.input_processing.documents import (
        DiagnosticCode,
        DocumentElement,
        DocumentLocator,
        ElementKind,
        ProcessingDiagnostic,
        ProcessingResult,
        ProcessorFingerprint,
    )
    from personagraph.input_processing.documents.preparation import (
        service as document_service,
    )

    path = tmp_path / "paper.pdf"
    path.write_bytes(b"%PDF-1.7\nplaceholder")
    store = _DocumentStore()
    processing = ProcessingResult(
        elements=(
            DocumentElement(
                element_id="text-page",
                kind=ElementKind.PARAGRAPH,
                text="方法部分可读",
                locator=DocumentLocator(page=1),
            ),
            DocumentElement(
                element_id="figure-page",
                kind=ElementKind.IMAGE,
                text=None,
                locator=DocumentLocator(page=2),
                needs_vision=True,
            ),
        ),
        processor=ProcessorFingerprint("test-reader", "1"),
        diagnostics=(
            ProcessingDiagnostic(
                DiagnosticCode.PAGE_NEEDS_VISION,
                locator=DocumentLocator(page=2),
                detail="figure not interpreted",
            ),
        ),
    )
    monkeypatch.setattr(document_service, "read_document", lambda _: processing)

    result = ingest_document_path(
        str(path),
        session_id="session_1",
        with_summary=False,
        document_store=store,
    )

    assert result["ok"] is True
    assert result["processing_status"] == "partial"
    assert result["needs_vision"] is True
    assert result["diagnostics"] == [{
        "code": "page_needs_vision",
        "at": "p2",
        "detail": "figure not interpreted",
    }]
    assert store.ingest_calls[0]["processing_status"] == "partial"
    assert store.ingest_calls[0]["processing_diagnostics"] == tuple(result["diagnostics"])


def test_source_byte_change_does_not_rename_an_unchanged_chunk(tmp_path):
    path = tmp_path / "brief.md"
    path.write_text("# A\n\none\n\n# B\n\nstable\n", encoding="utf-8")
    store = _DocumentStore()

    first = ingest_document_path(
        str(path),
        session_id="session_1",
        with_summary=False,
        document_store=store,
    )
    path.write_text("# A\n\nsomething much longer\n\n# B\n\nstable\n", encoding="utf-8")
    second = ingest_document_path(
        str(path),
        session_id="session_1",
        with_summary=False,
        document_store=store,
    )

    assert first["ok"] and second["ok"]
    first_stable = next(
        chunk
        for chunk in store.ingest_calls[0]["document_chunks"]
        if chunk.section_path == ("B",)
    )
    second_stable = next(
        chunk
        for chunk in store.ingest_calls[1]["document_chunks"]
        if chunk.section_path == ("B",)
    )
    assert first_stable.text == second_stable.text
    assert first_stable.span != second_stable.span
    assert first_stable.chunk_id == second_stable.chunk_id


def test_reindexed_document_does_not_regenerate_a_user_summary(tmp_path):
    path = tmp_path / "brief.md"
    path.write_text("# 方案\n\n原始内容。", encoding="utf-8")
    store = _DocumentStore()
    store.ingest_result["reindexed"] = True
    store.documents[0].update(title="用户标题", summary="用户摘要")

    result = ingest_document_path(
        str(path),
        session_id="session_1",
        with_summary=True,
        document_store=store,
        summarize=lambda _title, _preview: pytest.fail("reindex must not regenerate summary"),
    )

    assert result["ok"] is True
    assert result["title"] == "用户标题"
    assert result["summary"] == "用户摘要"
