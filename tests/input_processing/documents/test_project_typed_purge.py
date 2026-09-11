"""Project-scoped typed retrieval Units must outlive Session mount choices exactly."""

from __future__ import annotations

from personagraph.input_processing.documents import (
    ChunkSpan,
    DocumentChunk,
    DocumentLocator,
    ElementKind,
)
from personagraph.workspace.documents import application as documents
from personagraph.retrieval.sqlite_store import (
    RetrievalDataVersionRole,
    RetrievalDataVersionState,
    SqliteRetrievalCatalog,
)
from tests.documents._authority import bound_project_document_authority


def test_unmounted_typed_document_delete_still_emits_one_project_purge(
    tmp_path,
    seed_session_ids,
) -> None:
    seed_session_ids("session-a")
    chunk = DocumentChunk(
        chunk_id="producer-chunk-1",
        text="Project-owned evidence.",
        span=ChunkSpan(
            start=DocumentLocator(page=1, ordinal=0),
            end=DocumentLocator(page=1, ordinal=0),
        ),
        section_path=("Evidence",),
        element_ids=("element-1",),
        token_count=3,
        kind=ElementKind.PARAGRAPH,
        source_pages=(1,),
    )

    with bound_project_document_authority(tmp_path) as authority:
        with authority.session("session-a"):
            stored = authority.ingest(
                "evidence.md",
                "evidence",
                "md",
                [{"content": chunk.text, "loc": chunk.loc}],
                session_id="session-a",
                processor_fingerprint="reader@test",
                document_chunks=(chunk,),
                chunker_fingerprint="chunker@test",
                processing_status="complete",
            )
            catalog = SqliteRetrievalCatalog()
            catalog.create_data_version(
                version_id="active-documents-v1",
                fingerprint="document-generation-v1",
                role=RetrievalDataVersionRole.ACTIVE,
                state=RetrievalDataVersionState.READY,
            )
            assert documents.detach(stored["doc_id"], "session-a") is True
            assert documents.remove(
                stored["doc_id"],
                retrieval_data_version="active-documents-v1",
                document_index_port=authority.document_index_port,
            ) is True

        with authority.database.connect() as conn:
            rows = conn.execute(
                "SELECT kind, source_unit_id FROM retrieval_update_outbox"
            ).fetchall()

    assert [(row["kind"], row["source_unit_id"]) for row in rows] == [
        ("purge", f"document-v3:{stored['doc_id']}:producer-chunk-1")
    ]
