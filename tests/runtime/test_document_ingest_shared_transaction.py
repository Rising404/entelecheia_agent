from __future__ import annotations

import hashlib

import pytest

from personagraph.input_processing.documents import (
    ChunkSpan,
    DocumentChunk,
    DocumentLocator,
    ElementKind,
)
from personagraph.input_processing.files import fingerprint_file
from personagraph.workspace.files import (
    FileSource,
    WorkspaceFileAuthority,
)
from personagraph.workspace.storage import DocumentDatabase
from personagraph.workspace.documents import application as docstore
from personagraph.workspace.storage.context import bind
from personagraph.workspace.pictures import (
    PictureSourceLocator,
    PictureUnitLocator,
    ensure_picture_in_transaction,
    ensure_picture_unit_in_transaction,
)
from personagraph.workspace.pictures.observations import (
    PictureObservationDraft,
    PictureObservationModality,
    PictureObservationRepository,
    PictureObservationService,
)
from personagraph.retrieval.contracts import (
    RetrievalUnit,
    SourceFilter,
    SourceType,
    SourceUnitRef,
)
from personagraph.retrieval.lifecycle.generation import (
    DOCUMENT_HYBRID_INDEX_RECIPE_CONTRACT,
    file_corpus_generation_spec,
)
from personagraph.retrieval.sources.picture import PictureObservationSourceAdapter
from personagraph.retrieval.sources.identity import (
    mounted_document_chunk_ref_and_content,
    picture_observation_ref_and_content,
)
from personagraph.retrieval.sqlite_store import (
    RetrievalDataVersionRole,
    RetrievalDataVersionState,
    SqliteRetrievalCatalog,
)
from personagraph.retrieval.operations.document_generation import (
    DocumentGenerationAuthority,
)


_NOW = "2026-09-04T12:00:00+00:00"


def _project_database(tmp_path) -> DocumentDatabase:
    project_root = tmp_path / "project"
    project_root.mkdir()
    return DocumentDatabase(
        "shared-transaction-test",
        project_root,
        tmp_path / "var" / "projects" / "shared-transaction-test" / "documents.sqlite",
    )


def _generation_spec():
    return file_corpus_generation_spec(
        encoder_fingerprint="shared-transaction-encoder@1",
        document_chunker_fingerprint="shared-transaction-chunker@1",
        document_chunk_contract_version=docstore.DOCUMENT_CHUNK_CONTRACT_VERSION,
        index_recipe=DOCUMENT_HYBRID_INDEX_RECIPE_CONTRACT,
    )


def _authority(
    database: DocumentDatabase,
    catalog: SqliteRetrievalCatalog,
    spec,
) -> DocumentGenerationAuthority:
    return DocumentGenerationAuthority(
        catalog=catalog,
        generation_spec=spec,
        method_store=None,
        connect_documents=database.open_connection,
        picture_source_reader=PictureObservationSourceAdapter(),
    )


def _seed_current_picture_observation(
    database: DocumentDatabase,
    *,
    ordinal: int = 1,
):
    image_path = database.project_root / f"picture-{ordinal}.png"
    image_path.write_bytes(f"fake-png-{ordinal}".encode("utf-8"))
    registration = WorkspaceFileAuthority(database).ensure_current_path(
        image_path.name,
        source=FileSource.WORKSPACE_EXISTING,
        media_type="image/png",
    )
    picture_id = f"picture-{ordinal}"
    picture_unit_id = f"picture-unit-{ordinal}"
    text = f"current picture semantics {ordinal}"
    with database.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        ensure_picture_in_transaction(
            conn,
            file_id=registration.file.file_id,
            file_version_id=registration.version.file_version_id,
            source_locator=PictureSourceLocator.whole_file(),
            source_content_sha256=registration.version.content_sha256,
            source_media_type="image/png",
            picture_id=picture_id,
            created_at=_NOW,
        )
        ensure_picture_unit_in_transaction(
            conn,
            picture_id=picture_id,
            locator=PictureUnitLocator.full(),
            producer_fingerprint="shared-transaction-picture-raster@1",
            parent_picture_unit_id=None,
            pixel_sha256=hashlib.sha256(
                f"pixels-{ordinal}".encode("utf-8")
            ).hexdigest(),
            media_type="image/png",
            width=10,
            height=10,
            picture_unit_id=picture_unit_id,
            created_at=_NOW,
        )
        commit = PictureObservationService(
            PictureObservationRepository()
        ).commit_in_transaction(
            conn,
            PictureObservationDraft(
                picture_id=picture_id,
                picture_unit_id=picture_unit_id,
                logical_invocation_id=f"picture-call-{ordinal}",
                request_ordinal=0,
                modality=PictureObservationModality.VLM,
                purpose="semantic-description",
                kind="caption",
                text=text,
                uncertainty=0.1,
                processor_fingerprint="shared-transaction-vlm@1",
                prompt_fingerprint="shared-transaction-prompt@1",
            ),
            created_at=_NOW,
        )
        conn.commit()
    ref, _ = picture_observation_ref_and_content(
        observation_id=commit.observation.observation_id,
        payload_sha256=commit.observation.payload_sha256,
        text=text,
    )
    return ref, SourceFilter.from_mapping(
        SourceType.PICTURE,
        {
            "file_id": registration.file.file_id,
            "file_version_id": registration.version.file_version_id,
            "picture_id": picture_id,
        },
    )


def _store_ready_unit(
    catalog: SqliteRetrievalCatalog,
    *,
    target_id: str,
    ref: SourceUnitRef,
    source_filter: SourceFilter,
) -> None:
    stored = catalog.upsert_pending_unit(
        RetrievalUnit(
            ref=ref,
            retrieval_data_version=target_id,
            source_filter=source_filter,
        )
    )
    catalog.mark_unit_index_ready(stored.unit_id)


def test_bootstrap_publication_uses_one_documents_connection(tmp_path):
    database = _project_database(tmp_path)
    spec = _generation_spec()
    content = "同一个项目数据库中的文档权威与检索目录必须共用发布事务。"
    source_path = database.project_root / "paper.md"
    source_path.write_text(content, encoding="utf-8")

    with bind(database):
        catalog = SqliteRetrievalCatalog()
        target, replayed = catalog.ensure_staging_data_version(
            version_id=spec.version_id,
            fingerprint=spec.fingerprint,
        )
        assert replayed is False

        registration = WorkspaceFileAuthority(database).ensure_current_path(
            "paper.md",
            source=FileSource.WORKSPACE_EXISTING,
            media_type="text/markdown",
        )
        locator = DocumentLocator(page=1, ordinal=0, section_path=("Evidence",))
        stored = docstore.ingest(
            str(source_path),
            "paper",
            "text/markdown",
            [{"content": content, "loc": "p.1"}],
            file_id=registration.file.file_id,
            file_version_id=registration.version.file_version_id,
            source_fingerprint=fingerprint_file(source_path),
            processor_fingerprint="shared-transaction-reader@1",
            document_chunks=(
                DocumentChunk(
                    chunk_id="producer-chunk-1",
                    text=content,
                    span=ChunkSpan(start=locator, end=locator),
                    section_path=("Evidence",),
                    element_ids=("element-1",),
                    token_count=16,
                    kind=ElementKind.PARAGRAPH,
                ),
            ),
            chunker_fingerprint=spec.chunker_fingerprint,
            processing_status="complete",
        )

        with database.connect() as conn:
            chunk = conn.execute(
                "SELECT id, source_version_id, producer_chunk_id, content "
                "FROM doc_chunks WHERE doc_id=?",
                (stored["doc_id"],),
            ).fetchone()
        assert chunk is not None
        ref, _ = mounted_document_chunk_ref_and_content(
            session_id="project-index",
            chunk_id=str(chunk["id"]),
            source_version_id=str(chunk["source_version_id"]),
            content=str(chunk["content"]),
            doc_id=str(stored["doc_id"]),
            producer_chunk_id=str(chunk["producer_chunk_id"]),
        )
        indexed = catalog.upsert_pending_unit(
            RetrievalUnit(
                ref=ref,
                retrieval_data_version=target.id,
                source_filter=SourceFilter.from_mapping(
                    SourceType.DOCUMENT,
                    {"doc_id": str(stored["doc_id"])},
                ),
            )
        )
        catalog.mark_unit_index_ready(indexed.unit_id)

        published = _authority(
            database,
            catalog,
            spec,
        ).publish_bootstrap_generation()

        assert published.id == spec.version_id
        assert published.role is RetrievalDataVersionRole.ACTIVE
        assert published.state is RetrievalDataVersionState.READY
        with database.connect() as conn:
            assert conn.execute("PRAGMA foreign_key_check").fetchall() == []


def test_bootstrap_publication_requires_current_picture_coverage(tmp_path):
    database = _project_database(tmp_path)
    spec = _generation_spec()

    with bind(database):
        catalog = SqliteRetrievalCatalog()
        target, _ = catalog.ensure_staging_data_version(
            version_id=spec.version_id,
            fingerprint=spec.fingerprint,
        )
        picture_ref, picture_filter = _seed_current_picture_observation(database)
        authority = _authority(database, catalog, spec)

        with pytest.raises(
            RuntimeError,
            match="bootstrap_retrieval_coverage_incomplete",
        ):
            authority.publish_bootstrap_generation()

        _store_ready_unit(
            catalog,
            target_id=target.id,
            ref=picture_ref,
            source_filter=picture_filter,
        )
        published = authority.publish_bootstrap_generation()

        assert published.id == target.id
        assert published.role is RetrievalDataVersionRole.ACTIVE


def test_bootstrap_publication_rejects_an_extra_stale_picture_unit(tmp_path):
    database = _project_database(tmp_path)
    spec = _generation_spec()

    with bind(database):
        catalog = SqliteRetrievalCatalog()
        target, _ = catalog.ensure_staging_data_version(
            version_id=spec.version_id,
            fingerprint=spec.fingerprint,
        )
        current_ref, current_filter = _seed_current_picture_observation(database)
        _store_ready_unit(
            catalog,
            target_id=target.id,
            ref=current_ref,
            source_filter=current_filter,
        )
        stale_ref = SourceUnitRef(
            source_type=SourceType.PICTURE,
            source_unit_id="picture-observation-v1:stale-picture:stale-observation",
            source_revision="f" * 64,
            indexed_content_hash="e" * 64,
        )
        _store_ready_unit(
            catalog,
            target_id=target.id,
            ref=stale_ref,
            source_filter=SourceFilter.from_mapping(
                SourceType.PICTURE,
                {
                    "file_id": "stale-file",
                    "file_version_id": "stale-version",
                    "picture_id": "stale-picture",
                },
            ),
        )

        with pytest.raises(
            RuntimeError,
            match="bootstrap_retrieval_manifest_not_exact",
        ):
            _authority(database, catalog, spec).publish_bootstrap_generation()


def test_source_commit_revalidates_writable_generation_in_caller_transaction(
    tmp_path,
):
    database = _project_database(tmp_path)
    spec = _generation_spec()

    with bind(database):
        catalog = SqliteRetrievalCatalog()
        selected, _ = catalog.ensure_staging_data_version(
            version_id=spec.version_id,
            fingerprint=spec.fingerprint,
        )
        authority = _authority(database, catalog, spec)

        with database.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            catalog.mark_data_version_ready_in_transaction(conn, selected.id)
            assert conn.in_transaction is True

            with pytest.raises(RuntimeError, match="retrieval_target_not_writable"):
                authority._require_writable_source_commit_target_in_connection(
                    conn,
                    selected,
                )

            assert conn.in_transaction is True
            conn.rollback()

        restored = catalog.get_data_version(selected.id)
        assert restored is not None
        assert restored.role is RetrievalDataVersionRole.STAGING
        assert restored.state is RetrievalDataVersionState.BUILDING


def test_transaction_catalog_api_never_opens_or_commits_an_owned_connection(
    tmp_path,
    monkeypatch,
):
    database = _project_database(tmp_path)
    spec = _generation_spec()

    with bind(database):
        catalog = SqliteRetrievalCatalog()
        selected, _ = catalog.ensure_staging_data_version(
            version_id=spec.version_id,
            fingerprint=spec.fingerprint,
        )
        missing_ref = SourceUnitRef(
            source_type=SourceType.DOCUMENT,
            source_unit_id="document:missing:chunk",
            source_revision="missing-version",
            indexed_content_hash="missing-content-hash",
        )

        def unexpected_owned_operation():
            raise AssertionError("transaction API attempted an owned operation")

        with database.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            monkeypatch.setattr(catalog, "initialize", unexpected_owned_operation)
            monkeypatch.setattr(catalog, "connect", unexpected_owned_operation)

            assert catalog.get_data_version_in_transaction(conn, selected.id) == selected
            assert (
                catalog.require_writable_data_version_in_transaction(
                    conn,
                    selected.id,
                )
                == selected
            )
            assert catalog.get_unit_in_transaction(
                conn,
                missing_ref,
                selected.id,
            ) is None
            assert catalog.list_stored_units_in_transaction(conn, selected.id) == ()

            ready = catalog.mark_data_version_ready_in_transaction(conn, selected.id)
            assert ready.state is RetrievalDataVersionState.READY
            active = catalog.activate_data_version_in_transaction(conn, selected.id)
            assert active.role is RetrievalDataVersionRole.ACTIVE
            assert conn.in_transaction is True
            conn.rollback()

        restored = SqliteRetrievalCatalog().get_data_version(selected.id)
        assert restored is not None
        assert restored.role is RetrievalDataVersionRole.STAGING
        assert restored.state is RetrievalDataVersionState.BUILDING
