from __future__ import annotations

from dataclasses import replace
from contextlib import nullcontext
import hashlib
import json
import sqlite3

import pytest

from personagraph.input_processing.documents import (
    ChunkSpan,
    DocumentChunk,
    DocumentElement,
    DocumentLocator,
    ElementKind,
    PreparedDocumentIngest,
)
from personagraph.input_processing.files import fingerprint_file
from personagraph.workspace.documents import application as docstore
from personagraph.workspace.documents.inspection import inspect_current_file_document
from personagraph.retrieval.sqlite_store import (
    RetrievalDataVersionRole,
    RetrievalDataVersionState,
    SqliteRetrievalCatalog,
)
from tests.documents._authority import (
    ProjectDocumentAuthority,
    bound_project_document_authority,
)


@pytest.fixture
def authority(tmp_path, seed_session_ids):
    seed_session_ids("s1", "s2")
    with bound_project_document_authority(tmp_path) as value:
        yield value


def _typed_chunk(
    *,
    producer_chunk_id: str = "ch_shared",
    text: str = "Methods\nEvidence-grounded synthesis.",
) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=producer_chunk_id,
        text=text,
        span=ChunkSpan(
            start=DocumentLocator(
                page=2,
                ordinal=3,
                section_path=("Methods", "Retrieval"),
                bbox=(1.25, 2.5, 100.75, 20.0),
                char_range=(10, 40),
            ),
            end=DocumentLocator(
                page=3,
                ordinal=4,
                section_path=("Methods", "Retrieval"),
                bbox=(2.0, 4.0, 90.0, 18.0),
                char_range=(41, 75),
            ),
        ),
        section_path=("Methods", "Retrieval"),
        element_ids=("el_1", "el_2"),
        token_count=17,
        kind=ElementKind.PARAGRAPH,
        was_split=True,
    )


def _source_element(text: str, *, element_id: str = "el_1") -> DocumentElement:
    return DocumentElement(
        element_id=element_id,
        kind=ElementKind.PARAGRAPH,
        text=text,
        locator=DocumentLocator(page=2),
        source_pages=(2,),
    )


def _create_retrieval_generation(authority: ProjectDocumentAuthority) -> None:
    authority.database.initialize()
    catalog = SqliteRetrievalCatalog(authority.database.db_path)
    catalog.initialize()
    catalog.create_data_version(
        version_id="retrieval-v1",
        fingerprint="typed-authority-test@1",
        role=RetrievalDataVersionRole.STAGING,
        state=RetrievalDataVersionState.BUILDING,
    )


def test_native_file_chunk_reads_are_readonly_and_reject_unmounted_or_stale_versions(authority, monkeypatch):
    result = _ingest_typed(authority, "/workspace/native.pdf", _typed_chunk())
    document_id = result["doc_id"]
    with authority.database.connect() as connection:
        lineage = connection.execute(
            "SELECT d.file_id, v.file_version_id, v.id FROM documents AS d "
            "JOIN document_versions AS v ON v.id=d.current_version_id WHERE d.id=?",
            (document_id,),
        ).fetchone()
    file_id, file_version_id, document_version_id = lineage
    monkeypatch.setattr(authority.database, "initialize", lambda: pytest.fail("read initialized the database"))
    document = docstore.get_current_file_document(
        file_id=file_id, file_version_id=file_version_id, session_id="s1",
    )
    assert document is not None and document.document_id == document_id
    chunk = docstore.read_current_file_chunk(
        file_id=file_id, file_version_id=file_version_id,
        document_version_id=document_version_id, session_id="s1", sequence=0,
    )
    assert chunk is not None and chunk.content == _typed_chunk().text
    assert docstore.get_current_chunk_document(chunk_id=chunk.chunk_id, session_id="s1") == document
    assert docstore.get_current_chunk_document(chunk_id=chunk.chunk_id, session_id="s2") is None
    assert docstore.get_current_chunk_document(chunk_id="missing-native-chunk", session_id="s1") is None
    assert docstore.read_current_file_chunk(
        file_id=file_id, file_version_id=file_version_id,
        document_version_id=document_version_id, session_id="s1", chunk_id=chunk.chunk_id,
    ) == chunk
    assert docstore.get_current_file_document(
        file_id=file_id, file_version_id=file_version_id, session_id="s2",
    ) is None
    assert docstore.read_current_file_chunk(
        file_id=file_id, file_version_id=file_version_id,
        document_version_id="stale-version", session_id="s1", chunk_id=chunk.chunk_id,
    ) is None
    with sqlite3.connect(authority.database.db_path) as connection:
        connection.execute("UPDATE doc_chunks SET source_version_id='stale-version' WHERE id=?", (chunk.chunk_id,))
    assert docstore.get_current_chunk_document(chunk_id=chunk.chunk_id, session_id="s1") is None


def _ingest_typed(
    authority: ProjectDocumentAuthority,
    path: str,
    chunk: DocumentChunk,
    *,
    session_id: str = "s1",
    source_payload: str | bytes | None = None,
    processing_status: str = "complete",
    processing_diagnostics: tuple[dict[str, object], ...] = (),
    retrieval_data_version: str | None = None,
    processor_fingerprint: str = "test-reader@1",
    chunker_fingerprint: str = "structure-first@test",
):
    relative_path = path.removeprefix("/workspace/").lstrip("/")
    return authority.ingest(
        relative_path,
        "paper",
        "pdf",
        [{"content": "Evidence-grounded synthesis.", "loc": "p2-p3"}],
        session_id=session_id,
        source_payload=(
            "Evidence-grounded synthesis."
            if source_payload is None
            else source_payload
        ),
        retrieval_data_version=retrieval_data_version,
        processor_fingerprint=processor_fingerprint,
        document_chunks=(chunk,),
        chunker_fingerprint=chunker_fingerprint,
        processing_status=processing_status,
        processing_diagnostics=processing_diagnostics,
    )


def test_typed_chunk_round_trips_canonical_span_metadata_and_content_identity(authority):
    chunk = _typed_chunk()
    result = _ingest_typed(authority, "/workspace/paper.pdf", chunk)

    stored = docstore.list_current_document_chunks(result["doc_id"], session_id="s1")
    assert len(stored) == 1
    row = stored[0]
    assert row["id"] != chunk.chunk_id
    assert row["producer_chunk_id"] == chunk.chunk_id
    assert row["chunk_contract_version"] == 1
    assert row["content_sha256"] == hashlib.sha256(chunk.text.encode("utf-8")).hexdigest()
    assert json.loads(row["span_json"]) == {
        "end": {
            "bbox": [2.0, 4.0, 90.0, 18.0],
            "char_range": [41, 75],
            "ordinal": 4,
            "page": 3,
            "section_path": ["Methods", "Retrieval"],
        },
        "start": {
            "bbox": [1.25, 2.5, 100.75, 20.0],
            "char_range": [10, 40],
            "ordinal": 3,
            "page": 2,
            "section_path": ["Methods", "Retrieval"],
        },
    }
    assert json.loads(row["metadata_json"]) == {
        "element_ids": ["el_1", "el_2"],
        "kind": "paragraph",
        "section_path": ["Methods", "Retrieval"],
        "token_count": 17,
        "was_split": True,
    }
    version = docstore.list_document_versions(result["doc_id"])[0]
    assert version["processing_status"] == "complete"
    assert json.loads(version["diagnostics_json"]) == []
    assert version["processor_fingerprint"] == "test-reader@1"
    assert version["chunker_fingerprint"] == "structure-first@test"
    assert version["chunk_contract_version"] == 1

    snapshot = docstore.get_mounted_document_index_binding_snapshot(
        "s1",
        version_map={result["doc_id"]: row["source_version_id"]},
        maximum_bindings=10,
    )
    assert snapshot.source_snapshot_is_current is True
    assert len(snapshot.bindings) == 1
    assert snapshot.bindings[0].indexed_content_hash == hashlib.sha256(
        chunk.text.strip().encode("utf-8")
    ).hexdigest()


@pytest.mark.parametrize("session_id", ["s1", None])
def test_prepared_ingest_can_join_a_caller_owned_authority_transaction(
    authority, monkeypatch, session_id,
):
    _create_retrieval_generation(authority)
    source, registration = authority.write(
        "paper.pdf",
        "Evidence-grounded synthesis.",
        media_type="pdf",
    )
    prepared = PreparedDocumentIngest(
        canonical_path=str(source),
        title="paper",
        mime="pdf",
        elements=({"content": "Evidence-grounded synthesis.", "loc": "p2-p3"},),
        source_elements=(_source_element("Evidence-grounded synthesis."),),
        document_chunks=(_typed_chunk(),),
        source_fingerprint=fingerprint_file(source),
        processor_fingerprint="test-reader@1",
        chunker_fingerprint="structure-first@test",
        processing_status="complete",
        processing_diagnostics=(),
        needs_vision=False,
        summary_preview="Evidence-grounded synthesis.",
    )
    if session_id is None:
        from personagraph.workspace.documents import mounting

        def unexpected_mount_port():
            raise AssertionError("shared document commit must not use a Session mount port")

        monkeypatch.setattr(mounting, "_mount_port", unexpected_mount_port)
    scope = authority.session(session_id) if session_id else nullcontext()
    with scope:
        with authority.database.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            receipt = docstore.ingest_prepared_in_transaction(
                conn,
                prepared=prepared,
                session_id=session_id,
                retrieval_data_version="retrieval-v1",
                document_index_port=authority.document_index_port,
                file_id=registration.file.file_id,
                file_version_id=registration.version.file_version_id,
            )
            assert receipt.document_id
            assert receipt.document_version_id
            assert receipt.retrieval_event_ids
            assert conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 1
            source_snapshot = conn.execute(
                "SELECT source_elements_json, source_elements_sha256 "
                "FROM document_versions WHERE id=?",
                (receipt.document_version_id,),
            ).fetchone()
            assert json.loads(source_snapshot["source_elements_json"]) == [{
                "content": "Evidence-grounded synthesis.",
                "element_id": "el_1",
                "locator": "p2",
                "source_pages": [2],
            }]
            assert source_snapshot["source_elements_sha256"] == hashlib.sha256(
                source_snapshot["source_elements_json"].encode("utf-8")
            ).hexdigest()
            conn.rollback()

    assert docstore.list_documents() == []


def test_prepared_reindex_receipt_keeps_old_purge_and_new_upsert_event_ids(authority):
    _create_retrieval_generation(authority)
    source, first_registration = authority.write(
        "paper.pdf",
        "Old evidence.",
        media_type="pdf",
    )
    first = PreparedDocumentIngest(
        canonical_path=str(source),
        title="paper",
        mime="pdf",
        elements=({"content": "Old evidence.", "loc": "p2"},),
        source_elements=(_source_element("Old evidence."),),
        document_chunks=(_typed_chunk(text="Old evidence."),),
        source_fingerprint=fingerprint_file(source),
        processor_fingerprint="test-reader@1",
        chunker_fingerprint="structure-first@test",
        processing_status="complete",
        processing_diagnostics=(),
        needs_vision=False,
        summary_preview="Old evidence.",
    )
    with authority.session("s1"):
        with authority.database.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            original = docstore.ingest_prepared_in_transaction(
                conn,
                prepared=first,
                session_id="s1",
                retrieval_data_version="retrieval-v1",
                document_index_port=authority.document_index_port,
                file_id=first_registration.file.file_id,
                file_version_id=first_registration.version.file_version_id,
            )
            conn.commit()

    source, second_registration = authority.write(
        "paper.pdf",
        "New evidence.",
        media_type="pdf",
    )
    second = replace(
        first,
        elements=({"content": "New evidence.", "loc": "p2"},),
        source_elements=(_source_element("New evidence."),),
        document_chunks=(_typed_chunk(text="New evidence."),),
        source_fingerprint=fingerprint_file(source),
        summary_preview="New evidence.",
    )
    with authority.session("s1"):
        with authority.database.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            updated = docstore.ingest_prepared_in_transaction(
                conn,
                prepared=second,
                session_id="s1",
                retrieval_data_version="retrieval-v1",
                document_index_port=authority.document_index_port,
                file_id=second_registration.file.file_id,
                file_version_id=second_registration.version.file_version_id,
            )
            placeholders = ",".join("?" for _ in updated.retrieval_event_ids)
            events = conn.execute(
                "SELECT kind, source_revision FROM retrieval_update_outbox "
                f"WHERE event_id IN ({placeholders}) ORDER BY kind",
                updated.retrieval_event_ids,
            ).fetchall()
            conn.commit()

    assert updated.document_id == original.document_id
    assert updated.document_version_id != original.document_version_id
    assert {(row["kind"], row["source_revision"]) for row in events} == {
        ("purge", original.document_version_id),
        ("upsert", updated.document_version_id),
    }


def test_shared_content_artifact_copies_exact_source_elements_to_new_version(
    authority,
):
    _create_retrieval_generation(authority)
    payload = "Evidence-grounded synthesis."
    first_source, first_registration = authority.write(
        "first.pdf", payload, media_type="pdf",
    )
    second_source, second_registration = authority.write(
        "second.pdf", payload, media_type="pdf",
    )
    prepared = PreparedDocumentIngest(
        canonical_path=str(first_source),
        title="first",
        mime="pdf",
        elements=({"content": payload, "loc": "p2"},),
        source_elements=(_source_element(payload),),
        document_chunks=(_typed_chunk(text=payload),),
        source_fingerprint=fingerprint_file(first_source),
        processor_fingerprint="test-reader@1",
        chunker_fingerprint="structure-first@test",
        processing_status="complete",
        processing_diagnostics=(),
        needs_vision=False,
        summary_preview=payload,
    )
    with authority.session("s1"):
        with authority.database.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            first = docstore.ingest_prepared_in_transaction(
                conn,
                prepared=prepared,
                session_id="s1",
                retrieval_data_version="retrieval-v1",
                document_index_port=authority.document_index_port,
                file_id=first_registration.file.file_id,
                file_version_id=first_registration.version.file_version_id,
            )
            conn.commit()
        with authority.database.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            assert docstore.has_reusable_content_artifact(
                conn,
                canonical_path=str(second_source),
                source_sha256=second_registration.version.content_sha256,
                processor_fingerprint=prepared.processor_fingerprint,
                chunker_fingerprint=prepared.chunker_fingerprint,
                chunk_contract_version=1,
            )
            second = docstore.reuse_content_artifact_in_transaction(
                conn,
                canonical_path=str(second_source),
                session_id="s1",
                source_fingerprint=fingerprint_file(second_source),
                processor_fingerprint=prepared.processor_fingerprint,
                chunker_fingerprint=prepared.chunker_fingerprint,
                chunk_contract_version=1,
                retrieval_data_version="retrieval-v1",
                document_index_port=authority.document_index_port,
                file_id=second_registration.file.file_id,
                file_version_id=second_registration.version.file_version_id,
            )
            assert second is not None
            snapshots = conn.execute(
                "SELECT id, source_elements_json, source_elements_sha256 "
                "FROM document_versions WHERE id IN (?, ?) ORDER BY id",
                (first.document_version_id, second.document_version_id),
            ).fetchall()
            assert len(snapshots) == 2
            assert len({row["source_elements_json"] for row in snapshots}) == 1
            assert len({row["source_elements_sha256"] for row in snapshots}) == 1
            conn.commit()

        inspected = inspect_current_file_document(
            file_id=second_registration.file.file_id,
            file_version_id=second_registration.version.file_version_id,
            document_version_id=second.document_version_id,
            session_id="s1",
        )
    assert inspected is not None
    assert inspected.source_elements is not None
    assert tuple(element.content for element in inspected.source_elements) == (payload,)


def test_duplicate_producer_id_in_one_generation_fails_before_any_write(authority):
    first = _typed_chunk(producer_chunk_id="ch_duplicate")
    duplicate = replace(first, text="Different content under the same producer ID.")

    with pytest.raises(ValueError, match="duplicate producer_chunk_id"):
        authority.ingest(
            "duplicate.pdf",
            "duplicate",
            "pdf",
            [{"content": "duplicate", "loc": "p1"}],
            session_id="s1",
            processor_fingerprint="test-reader@1",
            document_chunks=(first, duplicate),
            chunker_fingerprint="structure-first@test",
            processing_status="complete",
            processing_diagnostics=(),
        )

    assert docstore.list_documents() == []


def test_typed_generation_without_source_fingerprint_fails_before_any_write(authority):
    with pytest.raises(ValueError, match="source_fingerprint"):
        docstore.ingest(
            "/workspace/unbound.pdf",
            "unbound",
            "pdf",
            [{"content": "typed content", "loc": "p1"}],
            session_id="s1",
            processor_fingerprint="test-reader@1",
            document_chunks=(_typed_chunk(),),
            chunker_fingerprint="structure-first@test",
            processing_status="complete",
            processing_diagnostics=(),
        )

    assert docstore.list_documents() == []


def test_same_producer_id_is_allowed_in_different_documents(authority):
    chunk = _typed_chunk(producer_chunk_id="ch_same")

    first = _ingest_typed(authority, "/workspace/a.pdf", chunk)
    second = _ingest_typed(authority, "/workspace/b.pdf", chunk)

    assert first["doc_id"] != second["doc_id"]
    with authority.database.connect() as conn:
        rows = conn.execute(
            "SELECT id, doc_id, producer_chunk_id FROM doc_chunks ORDER BY doc_id"
        ).fetchall()
    assert len(rows) == 2
    assert {row["producer_chunk_id"] for row in rows} == {"ch_same"}
    assert len({row["id"] for row in rows}) == 2


def test_source_sha_change_creates_a_new_version_even_when_extracted_text_is_equal(authority):
    chunk = _typed_chunk()
    first = _ingest_typed(
        authority,
        "/workspace/paper.pdf",
        chunk,
        source_payload="physical source v1",
    )
    second = _ingest_typed(
        authority,
        "/workspace/paper.pdf",
        chunk,
        source_payload="physical source v2",
    )

    assert second["doc_id"] == first["doc_id"]
    assert second["deduped"] is False
    assert second["reindexed"] is True
    versions = docstore.list_document_versions(first["doc_id"])
    assert [version["version_number"] for version in versions] == [2, 1]
    assert versions[0]["source_sha256"] != versions[1]["source_sha256"]


def test_version_ledger_retains_each_generations_processor_chunker_and_contract(authority):
    chunk = _typed_chunk()
    first = _ingest_typed(
        authority,
        "/workspace/paper.pdf",
        chunk,
        processor_fingerprint="test-reader@1",
        chunker_fingerprint="structure-first@1",
    )
    second = _ingest_typed(
        authority,
        "/workspace/paper.pdf",
        chunk,
        processor_fingerprint="test-reader@2",
        chunker_fingerprint="structure-first@2",
    )

    assert second["doc_id"] == first["doc_id"]
    assert second["reindexed"] is True
    versions = docstore.list_document_versions(first["doc_id"])
    assert [version["processor_fingerprint"] for version in versions] == [
        "test-reader@2",
        "test-reader@1",
    ]
    assert [version["chunker_fingerprint"] for version in versions] == [
        "structure-first@2",
        "structure-first@1",
    ]
    assert [version["chunk_contract_version"] for version in versions] == [1, 1]


def test_processing_status_and_diagnostics_are_canonical_and_rejected_is_not_persistable(authority):
    diagnostic = {"detail": "page image", "at": "p4", "code": "page_needs_vision"}
    result = _ingest_typed(
        authority,
        "/workspace/partial.pdf",
        _typed_chunk(),
        processing_status="partial",
        processing_diagnostics=(diagnostic,),
    )

    version = docstore.list_document_versions(result["doc_id"])[0]
    assert version["processing_status"] == "partial"
    assert version["diagnostics_json"] == json.dumps(
        [diagnostic], ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )

    with pytest.raises(ValueError, match="processing_status"):
        _ingest_typed(
            authority,
            "/workspace/rejected.pdf",
            _typed_chunk(),
            processing_status="rejected",
        )
    assert {document["id"] for document in docstore.list_documents()} == {result["doc_id"]}


def test_parser_partial_processing_diagnostic_is_persistable(authority):
    diagnostic = {
        "detail": "AnnotationInventory:MalformedPDFException",
        "at": "p4",
        "code": "parser_partial",
    }

    result = _ingest_typed(
        authority,
        "/workspace/annotation-partial.pdf",
        _typed_chunk(),
        processing_status="partial",
        processing_diagnostics=(diagnostic,),
    )

    version = docstore.list_document_versions(result["doc_id"])[0]
    assert version["processing_status"] == "partial"
    assert json.loads(version["diagnostics_json"]) == [diagnostic]


@pytest.mark.parametrize(
    "diagnostic",
    [
        {"code": "permission_denied", "at": "p1", "detail": None},
        {"code": "future_coverage_gap", "at": "p1", "detail": None},
        {"at": "p1", "detail": None},
    ],
)
def test_partial_processing_rejects_non_admissible_diagnostics_before_write(authority, diagnostic):
    with pytest.raises(ValueError, match="processing diagnostics"):
        _ingest_typed(
            authority,
            "/workspace/invalid-partial.pdf",
            _typed_chunk(),
            processing_status="partial",
            processing_diagnostics=(diagnostic,),
        )

    assert docstore.list_documents() == []


@pytest.mark.parametrize(
    ("processing_status", "diagnostics_json"),
    [
        ("complete", '[{"code":"permission_denied"}]'),
        ("partial", "[]"),
        ("legacy_unknown", "[]"),
        ("future_status", "[]"),
    ],
)
def test_current_processing_projection_rejects_corrupt_status_diagnostic_pairs(
    authority,
    processing_status,
    diagnostics_json,
):
    result = _ingest_typed(authority, "/workspace/corrupt.pdf", _typed_chunk())
    version_id = docstore.list_document_versions(result["doc_id"])[0]["id"]
    with authority.database.connect() as conn:
        conn.execute(
            "UPDATE document_versions SET processing_status=?, diagnostics_json=? WHERE id=?",
            (processing_status, diagnostics_json, version_id),
        )

    with pytest.raises(ValueError, match="processing"):
        docstore.get_document(result["doc_id"])
    with pytest.raises(ValueError, match="processing"):
        docstore.preflight_documents({result["doc_id"]})


def test_processing_provenance_change_replaces_same_source_generation_and_current_projection(authority):
    diagnostic = {"code": "page_needs_vision", "at": "p4", "detail": "page image"}
    first = _ingest_typed(
        authority,
        "/workspace/paper.pdf",
        _typed_chunk(),
        processing_status="partial",
        processing_diagnostics=(diagnostic,),
    )

    partial = docstore.get_document(first["doc_id"])
    assert partial and partial["processing_status"] == "partial"
    assert partial["diagnostics"] == [diagnostic]
    assert partial["needs_vision"] is True
    partial_version_id = docstore.list_document_versions(first["doc_id"])[0]["id"]
    partial_chunk = docstore.get_current_typed_document_chunk(
        first["doc_id"],
        "ch_shared",
        expected_version_id=partial_version_id,
        session_id="s1",
    )
    assert partial_chunk and partial_chunk["processing_status"] == "partial"
    assert json.loads(partial_chunk["diagnostics_json"]) == [diagnostic]

    second = _ingest_typed(authority, "/workspace/paper.pdf", _typed_chunk())

    assert second["doc_id"] == first["doc_id"]
    assert second["deduped"] is False
    assert second["reindexed"] is True
    versions = docstore.list_document_versions(first["doc_id"])
    assert [version["processing_status"] for version in versions] == ["complete", "partial"]
    current = docstore.get_document(first["doc_id"])
    assert current and current["processing_status"] == "complete"
    assert current["diagnostics"] == []
    assert current["needs_vision"] is False
    current_version_id = versions[0]["id"]
    current_chunk = docstore.get_current_typed_document_chunk(
        first["doc_id"],
        "ch_shared",
        expected_version_id=current_version_id,
        session_id="s1",
    )
    assert current_chunk and current_chunk["processing_status"] == "complete"
    assert json.loads(current_chunk["diagnostics_json"]) == []
    assert docstore.get_current_typed_document_chunk(
        first["doc_id"],
        "ch_shared",
        expected_version_id=partial_version_id,
        session_id="s1",
    ) is None
    assert docstore.mounted_docs("s1")[0]["processing_status"] == "complete"
    assert docstore.list_documents()[0]["processing_status"] == "complete"


def test_typed_lookup_is_bound_to_document_current_version_and_session_mount(authority):
    chunk = _typed_chunk(producer_chunk_id="ch_lookup")
    result = _ingest_typed(authority, "/workspace/paper.pdf", chunk)
    version_id = docstore.list_document_versions(result["doc_id"])[0]["id"]

    current = docstore.get_current_typed_document_chunk(
        result["doc_id"],
        "ch_lookup",
        expected_version_id=version_id,
        session_id="s1",
    )

    assert current and current["producer_chunk_id"] == "ch_lookup"
    assert docstore.get_current_typed_document_chunk(
        "wrong-doc", "ch_lookup", expected_version_id=version_id, session_id="s1"
    ) is None
    assert docstore.get_current_typed_document_chunk(
        result["doc_id"], "ch_lookup", expected_version_id="wrong-version", session_id="s1"
    ) is None
    assert docstore.get_current_typed_document_chunk(
        result["doc_id"], "ch_lookup", expected_version_id=version_id, session_id="s2"
    ) is None
