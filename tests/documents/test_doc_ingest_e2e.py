"""不与运行时耦合的文档摄取与挂载工作区集成。"""
from contextlib import contextmanager
from dataclasses import dataclass
import json
from typing import Iterator

import pytest

from personagraph.workspace.documents import application as ds
from personagraph.retrieval.profile import (
    durable_chunking_profile,
)
from personagraph.session import store as session_store
from tests.documents._authority import ingest_registered_document_fixture


@dataclass(frozen=True)
class _ProjectSessions:
    ids: dict[str, str]

    def __getitem__(self, name: str) -> str:
        return self.ids[name]

    @contextmanager
    def scope(self, name: str) -> Iterator[None]:
        with session_store.session_database_scope(self.ids[name]):
            yield


@pytest.fixture(autouse=True)
def _project_documents(tmp_path, partitioned_project_state, monkeypatch):
    class _OfflineEncoder:
        def token_ids(self, text: str):
            del text
            raise RuntimeError("offline test tokenizer")

        def fingerprint(self) -> str:
            return "offline-test-tokenizer"

    from personagraph.retrieval.indexing import encoder as encoder_module

    monkeypatch.setattr(encoder_module, "BgeM3Encoder", _OfflineEncoder)
    durable_chunking_profile.cache_clear()
    sessions = _ProjectSessions({
        name: session_store.create_session(
            "Entelecheia",
            title=f"document ingest {name}",
            working_dir=str(tmp_path),
        )
        for name in ("sess1",)
    })
    with sessions.scope("sess1"):
        yield sessions
    durable_chunking_profile.cache_clear()


def test_registered_document_fixture_commits_text(tmp_path, _project_documents):
    f = tmp_path / "notes.md"
    f.write_text("# 竞赛要点\n" + "评审重视创意与完成度。\n" * 100, encoding="utf-8")
    session_id = _project_documents["sess1"]
    r = ingest_registered_document_fixture(f, session_id=session_id)
    assert r["ok"] and r["n_chunks"] >= 1
    assert ds.mounted_docs(session_id)[0]["title"] == "notes"


def test_partial_typed_processing_persists_readable_text_and_coverage_gap(
    tmp_path, monkeypatch, _project_documents
):
    """可读文本保持可用，同时未读页面仍作为持久事实保留。"""
    from personagraph.input_processing.documents import (
        DiagnosticCode,
        DocumentElement,
        DocumentLocator,
        ElementKind,
        ProcessingDiagnostic,
        ProcessingResult,
        ProcessorFingerprint,
    )
    from personagraph.input_processing.documents.preparation import service as document_service
    source = tmp_path / "mixed.pdf"
    source.write_bytes(b"%PDF-1.7\nplaceholder")
    result = ProcessingResult(
        elements=(
            DocumentElement(
                element_id="text-page",
                kind=ElementKind.PARAGRAPH,
                text="可读的第一页",
                locator=DocumentLocator(page=1),
            ),
            DocumentElement(
                element_id="scan-page",
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
                detail="scanned page",
            ),
        ),
    )
    monkeypatch.setattr(document_service, "read_document", lambda path: result)

    session_id = _project_documents["sess1"]
    response = ingest_registered_document_fixture(source, session_id=session_id)

    assert response["ok"] is True
    assert response["processing_status"] == "partial"
    assert response["needs_vision"] is True
    assert response["diagnostics"] == [{
        "code": "page_needs_vision",
        "at": "p2",
        "detail": "scanned page",
    }]
    mounted = ds.mounted_docs(session_id)
    assert [document["id"] for document in mounted] == [response["doc_id"]]
    versions = ds.list_document_versions(response["doc_id"])
    assert versions[0]["processing_status"] == "partial"
    assert json.loads(versions[0]["diagnostics_json"]) == response["diagnostics"]
    chunks = ds.doc_read(response["doc_id"], session_id=session_id)
    assert [chunk["content"] for chunk in chunks] == ["可读的第一页"]
    assert {chunk["processing_status"] for chunk in chunks} == {"partial"}
    assert [json.loads(chunk["diagnostics_json"]) for chunk in chunks] == [
        response["diagnostics"]
    ]
    freshness = ds.preflight_mounted_documents(session_id, response["doc_id"])
    assert freshness["ok"] is True
    assert freshness["status"] == "verified_current"
    assert freshness["documents"] == [{
        "doc_id": response["doc_id"],
        "version_id": versions[0]["id"],
        "processing_status": "partial",
        "diagnostics": response["diagnostics"],
        "needs_vision": True,
        "status": "verified_current",
    }]


def test_parser_partial_crosses_prepare_and_document_commit(
    tmp_path, monkeypatch, _project_documents
):
    from personagraph.input_processing.documents import (
        DiagnosticCode,
        DocumentElement,
        DocumentLocator,
        ElementKind,
        ProcessingDiagnostic,
        ProcessingResult,
        ProcessorFingerprint,
    )
    from personagraph.input_processing.documents.preparation import service as document_service
    source = tmp_path / "annotation-partial.pdf"
    source.write_bytes(b"%PDF-1.7\nplaceholder")
    result = ProcessingResult(
        elements=(
            DocumentElement(
                element_id="readable-body",
                kind=ElementKind.PARAGRAPH,
                text="Readable body survived optional annotation failure.",
                locator=DocumentLocator(page=1),
            ),
        ),
        processor=ProcessorFingerprint("test-reader", "1"),
        diagnostics=(
            ProcessingDiagnostic(
                DiagnosticCode.PARSER_PARTIAL,
                locator=DocumentLocator(page=1),
                detail="AnnotationInventory:MalformedPDFException",
            ),
        ),
    )
    monkeypatch.setattr(document_service, "read_document", lambda path: result)

    session_id = _project_documents["sess1"]
    response = ingest_registered_document_fixture(
        source,
        session_id=session_id,
    )

    assert response["ok"] is True
    assert response["processing_status"] == "partial"
    assert response["needs_vision"] is False
    assert response["diagnostics"] == [{
        "code": "parser_partial",
        "at": "p1",
        "detail": "AnnotationInventory:MalformedPDFException",
    }]
    version = ds.list_document_versions(response["doc_id"])[0]
    assert version["processing_status"] == "partial"
    assert json.loads(version["diagnostics_json"]) == response["diagnostics"]
    chunks = ds.doc_read(response["doc_id"], session_id=session_id)
    assert [chunk["content"] for chunk in chunks] == [
        "Readable body survived optional annotation failure."
    ]


def test_fatal_typed_processing_rejects_all_content_without_store_mutation(
    tmp_path, monkeypatch, _project_documents
):
    from personagraph.input_processing.documents import (
        DiagnosticCode,
        DocumentElement,
        DocumentLocator,
        ElementKind,
        ProcessingDiagnostic,
        ProcessingResult,
        ProcessorFingerprint,
    )
    from personagraph.input_processing.documents.preparation import service as document_service
    source = tmp_path / "truncated.pdf"
    source.write_bytes(b"%PDF-1.7\nplaceholder")
    result = ProcessingResult(
        elements=(
            DocumentElement(
                element_id="text-page",
                kind=ElementKind.PARAGRAPH,
                text="不能作为完整来源落库的文本",
                locator=DocumentLocator(page=1),
            ),
        ),
        processor=ProcessorFingerprint("test-reader", "1"),
        diagnostics=(
            ProcessingDiagnostic(
                DiagnosticCode.LIMIT_REACHED,
                locator=DocumentLocator(page=2),
                detail="reader hard limit",
            ),
        ),
    )
    monkeypatch.setattr(document_service, "read_document", lambda path: result)

    session_id = _project_documents["sess1"]
    response = ingest_registered_document_fixture(source, session_id=session_id)

    assert response == {
        "ok": False,
        "reason": "document_processing_incomplete",
        "path": str(source.resolve()),
        "needs_vision": False,
        "diagnostics": [{
            "code": "limit_reached",
            "at": "p2",
            "detail": "reader hard limit",
        }],
    }
    assert ds.mounted_docs(session_id) == []


def test_changed_physical_source_blocks_tool_until_same_path_is_reingested(
    tmp_path, _project_documents
):
    source = tmp_path / "rules.md"
    source.write_text("旧版规则：创意占四十分。\n" * 60, encoding="utf-8")
    session_id = _project_documents["sess1"]
    first = ingest_registered_document_fixture(source, session_id=session_id)
    assert first["ok"] and first["deduped"] is False

    initial = ds.preflight_mounted_documents(session_id)
    assert initial["ok"] is True
    assert "创意占四十分" in str(
        ds.doc_read(
            first["doc_id"],
            n=100,
            session_id=session_id,
            expected_version_id=initial["version_map"][first["doc_id"]],
        )
    )

    source.write_text("新版规则：创意占五十分。\n" * 60, encoding="utf-8")
    blocked = ds.preflight_mounted_documents(session_id)
    assert blocked["ok"] is False
    assert blocked["status"] == "freshness_blocked"
    assert blocked["documents"][0]["status"] == "source_changed"

    refreshed = ingest_registered_document_fixture(source, session_id=session_id)
    assert refreshed["ok"] and refreshed["reindexed"] is True
    assert refreshed["doc_id"] == first["doc_id"]
    versions = ds.list_document_versions(first["doc_id"])
    assert [version["version_number"] for version in versions] == [2, 1]
    assert [version["status"] for version in versions] == ["active", "superseded"]
    assert versions[0]["supersedes_version_id"] == versions[1]["id"]
    current = ds.preflight_mounted_documents(session_id)
    assert current["status"] == "verified_current"
    assert current["documents"][0]["version_id"] == versions[0]["id"]
    chunks = ds.doc_read(
        first["doc_id"],
        n=100,
        session_id=session_id,
        expected_version_id=current["version_map"][first["doc_id"]],
    )
    assert all(
        chunk["source_version_id"] == versions[0]["id"]
        for chunk in chunks
    )
    snapshot = ds.get_retrieval_snapshot(current["snapshot_id"], session_id)
    assert snapshot and json.loads(snapshot["manifest_json"]) == {first["doc_id"]: versions[0]["id"]}
    assert "创意占五十分" in str(chunks)
    assert "创意占四十分" not in str(chunks)


def test_deleted_physical_source_blocks_doc_read_with_explicit_status(
    tmp_path, _project_documents
):
    source = tmp_path / "temporary.md"
    source.write_text("可检索内容。\n" * 60, encoding="utf-8")
    session_id = _project_documents["sess1"]
    result = ingest_registered_document_fixture(source, session_id=session_id)
    source.unlink()

    blocked = ds.preflight_mounted_documents(session_id, result["doc_id"])
    assert blocked["ok"] is False
    report = blocked["documents"][0]
    assert report["doc_id"] == result["doc_id"]
    assert report["status"] == "source_missing"
    assert report["version_id"]


def test_same_content_at_two_physical_paths_keeps_distinct_source_identity(
    tmp_path, _project_documents
):
    source_a = tmp_path / "a.md"
    source_b = tmp_path / "b.md"
    source_a.write_text("同一份内容。\n" * 60, encoding="utf-8")
    source_b.write_text("同一份内容。\n" * 60, encoding="utf-8")

    session_id = _project_documents["sess1"]
    first = ingest_registered_document_fixture(source_a, session_id=session_id)
    second = ingest_registered_document_fixture(source_b, session_id=session_id)
    assert first["doc_id"] != second["doc_id"]
    assert second["deduped"] is False


def test_snapshot_version_map_cannot_silently_read_reingested_chunks(
    tmp_path, _project_documents
):
    source = tmp_path / "snapshot.md"
    source.write_text("v1 内容：创意占四十分。\n" * 60, encoding="utf-8")
    session_id = _project_documents["sess1"]
    first = ingest_registered_document_fixture(source, session_id=session_id)
    snapshot = ds.preflight_mounted_documents(session_id, first["doc_id"])
    assert snapshot["ok"] and snapshot["snapshot_id"]

    source.write_text("v2 内容：创意占五十分。\n" * 60, encoding="utf-8")
    ingest_registered_document_fixture(source, session_id=session_id)

    version_id = snapshot["version_map"][first["doc_id"]]
    assert ds.doc_read(
        first["doc_id"],
        session_id=session_id,
        expected_version_id=version_id,
    ) == []


def test_ingest_respects_scope_and_reader_support_without_name_denials(tmp_path, _project_documents):
    notes = tmp_path / "secret_notes.txt"
    notes.write_text("authorized project notes", encoding="utf-8")
    session_id = _project_documents["sess1"]
    result = ingest_registered_document_fixture(notes, session_id=session_id)
    assert result["ok"] and result["n_chunks"] >= 1
    assert (
        ingest_registered_document_fixture("/etc/hosts", session_id=session_id)[
            "reason"
        ]
        == "outside_project_root"
    )
    exe = tmp_path / "a.bin"
    exe.write_bytes(b"\x00")
    assert (
        ingest_registered_document_fixture(exe, session_id=session_id)["reason"]
        == "unsupported_format"
    )
