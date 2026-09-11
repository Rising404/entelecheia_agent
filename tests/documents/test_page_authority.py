from __future__ import annotations

import json
import sqlite3

import pytest

from personagraph.input_processing.documents import (
    ChunkSpan,
    DocumentChunk,
    DocumentLocator,
    DocumentPageInventoryStatus,
    DocumentPageManifest,
    DocumentPageRecord,
    DocumentPageState,
    ElementKind,
    evaluate_page_authority_eligibility,
)
from personagraph.workspace.documents import application as docstore
from tests.documents._authority import (
    ProjectDocumentAuthority,
    bound_project_document_authority,
)


@pytest.fixture
def authority(tmp_path, seed_session_ids):
    seed_session_ids("s1", "s2")
    with bound_project_document_authority(tmp_path) as value:
        yield value


def _manifest(detector: str = "detector@1") -> DocumentPageManifest:
    return DocumentPageManifest(
        physical_page_count=10,
        inventory_status=DocumentPageInventoryStatus.COMPLETE,
        detector_fingerprint=detector,
        detector_capabilities=(
            "figure_inventory",
            "formula_inventory",
            "nontext_unit_inventory",
            "physical_page_inventory",
            "table_inventory",
            "text_element_source_pages",
            "typed_page_diagnostics",
        ),
        pages=tuple(
            DocumentPageRecord(
                page_number=page,
                state=DocumentPageState.TEXT,
                text_element_ids=(f"e{page}",),
            )
            for page in range(1, 11)
        ),
    )


def _chunk() -> DocumentChunk:
    return DocumentChunk(
        chunk_id="ch_all_pages",
        text="complete paper evidence",
        span=ChunkSpan(DocumentLocator(page=1), DocumentLocator(page=10)),
        section_path=(),
        element_ids=tuple(f"e{page}" for page in range(1, 11)),
        token_count=4,
        kind=ElementKind.PARAGRAPH,
        source_pages=tuple(range(1, 11)),
    )


def _ingest(
    authority: ProjectDocumentAuthority,
    *,
    manifest: DocumentPageManifest,
    session_id: str = "s1",
):
    with authority.session(session_id):
        return authority.ingest(
            "paper.pdf",
            "paper",
            "pdf",
            [{"content": "complete paper evidence", "loc": "p1-p10"}],
            session_id=session_id,
            processor_fingerprint="reader@1",
            document_chunks=(_chunk(),),
            chunker_fingerprint="chunker@1",
            processing_status="complete",
            processing_diagnostics=(),
            page_manifest=manifest,
        )


def test_page_manifest_and_exact_chunk_source_pages_round_trip_through_one_version(authority):
    manifest = _manifest()
    result = _ingest(authority, manifest=manifest)
    version = docstore.list_document_versions(result["doc_id"])[0]
    chunks = docstore.list_current_document_chunks(result["doc_id"], session_id="s1")

    assert version["page_manifest_sha256"] == manifest.manifest_sha256
    assert version["physical_page_count"] == 10
    assert json.loads(version["page_manifest_json"]) == manifest.to_dict()
    assert json.loads(chunks[0]["source_pages_json"]) == list(range(1, 11))

    resource = docstore.get_mounted_current_document_resource_snapshot(
        result["doc_id"],
        session_id="s1",
        maximum_chunks=1,
    )
    assert resource is not None
    assert resource.physical_page_count == 10
    assert resource.page_inventory_status == "complete"
    assert resource.processing_status == "complete"

    authority = docstore.get_current_document_page_authority(
        result["doc_id"],
        expected_version_id=version["id"],
        session_id="s1",
    )
    assert authority is not None
    assert authority.page_manifest == manifest
    assert authority.chunks[0].source_pages == tuple(range(1, 11))
    assert authority.chunks[0].element_ids == tuple(f"e{page}" for page in range(1, 11))
    assert authority.chunks[0].content_utf8_bytes == len(
        "complete paper evidence".encode("utf-8")
    )
    assert len(authority.chunks[0].content_sha256) == 64
    assert authority.chunks[0].chunk_contract_version == 1
    assert evaluate_page_authority_eligibility(
        authority.page_manifest,
        authority.chunks,
        processing_diagnostic_codes=authority.processing_diagnostic_codes,
    ).eligible is True


def test_exact_page_authority_is_current_version_and_session_scoped(authority):
    result = _ingest(authority, manifest=_manifest())
    version_id = docstore.list_document_versions(result["doc_id"])[0]["id"]

    assert docstore.get_current_document_page_authority(
        result["doc_id"], expected_version_id=version_id, session_id="s1"
    ) is not None
    assert docstore.get_current_document_page_authority(
        result["doc_id"], expected_version_id="wrong-version", session_id="s1"
    ) is None
    assert docstore.get_current_document_page_authority(
        result["doc_id"], expected_version_id=version_id, session_id="s2"
    ) is None
    assert docstore.get_current_document_page_authority(
        "wrong-doc", expected_version_id=version_id, session_id="s1"
    ) is None


def test_same_sha_manifest_reindex_freezes_both_lineages_without_dropping_provenance(authority):
    first = _ingest(authority, manifest=_manifest("detector@1"))
    first_version = docstore.list_document_versions(first["doc_id"])[0]

    second = _ingest(authority, manifest=_manifest("detector@2"))
    versions = docstore.list_document_versions(first["doc_id"])

    assert second["doc_id"] == first["doc_id"]
    assert second["reindexed"] is True
    assert [version["page_manifest_sha256"] for version in versions] == [
        _manifest("detector@2").manifest_sha256,
        _manifest("detector@1").manifest_sha256,
    ]
    assert len({version["source_sha256"] for version in versions}) == 1
    assert docstore.get_current_document_page_authority(
        first["doc_id"], expected_version_id=first_version["id"], session_id="s1"
    ) is None


def test_page_manifest_lineage_columns_cannot_be_mutated_in_place(authority):
    result = _ingest(authority, manifest=_manifest())
    version_id = docstore.list_document_versions(result["doc_id"])[0]["id"]

    with pytest.raises(sqlite3.IntegrityError, match="page manifest lineage is immutable"):
        with authority.database.connect() as conn:
            conn.execute(
                "UPDATE document_versions SET page_manifest_sha256=? WHERE id=?",
                ("0" * 64, version_id),
            )


def test_manifest_bound_ingest_rejects_missing_or_fabricated_chunk_page_sets(authority):
    missing = _chunk()
    missing = DocumentChunk(
        chunk_id=missing.chunk_id,
        text=missing.text,
        span=missing.span,
        section_path=missing.section_path,
        element_ids=missing.element_ids,
        token_count=missing.token_count,
        kind=missing.kind,
        source_pages=(1, 10),
    )

    with pytest.raises(ValueError, match="source_pages"):
        authority.ingest(
            "bad.pdf",
            "bad",
            "pdf",
            [{"content": "bad", "loc": "p1-p10"}],
            session_id="s1",
            processor_fingerprint="reader@1",
            document_chunks=(missing,),
            chunker_fingerprint="chunker@1",
            processing_status="complete",
            processing_diagnostics=(),
            page_manifest=_manifest(),
        )
    assert docstore.list_documents() == []


def test_same_sha_reindex_cannot_silently_drop_an_existing_manifest(authority):
    result = _ingest(authority, manifest=_manifest())
    version_id = docstore.list_document_versions(result["doc_id"])[0]["id"]

    with pytest.raises(ValueError, match="cannot drop"):
        authority.ingest(
            "paper.pdf",
            "paper",
            "pdf",
            [{"content": "complete paper evidence", "loc": "p1-p10"}],
            session_id="s1",
            processor_fingerprint="reader@2",
            document_chunks=(_chunk(),),
            chunker_fingerprint="chunker@2",
            processing_status="complete",
            processing_diagnostics=(),
        )

    assert docstore.list_document_versions(result["doc_id"])[0]["id"] == version_id


def test_exact_getter_reauthenticates_durable_manifest_instead_of_trusting_columns(authority):
    result = _ingest(authority, manifest=_manifest())
    version_id = docstore.list_document_versions(result["doc_id"])[0]["id"]
    with authority.database.connect() as conn:
        conn.execute("DROP TRIGGER trg_document_versions_page_manifest_immutable")
        payload = _manifest().to_dict()
        payload["detector_fingerprint"] = "tampered-detector"
        conn.execute(
            "UPDATE document_versions SET page_manifest_json=? WHERE id=?",
            (
                json.dumps(
                    payload,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                version_id,
            ),
        )

    with pytest.raises(ValueError, match="manifest"):
        docstore.get_current_document_page_authority(
            result["doc_id"], expected_version_id=version_id, session_id="s1"
        )
