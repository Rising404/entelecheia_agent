from __future__ import annotations

import pytest

from personagraph.input_processing.documents import (
    ChunkSpan,
    DocumentChunk,
    DocumentLocator,
    ElementKind,
)
from personagraph.retrieval.indexing.encoder import DeterministicLexicalEncoder
from personagraph.retrieval.contracts import (
    SourceAvailability,
    SourceFilter,
    SourceType,
)
from personagraph.retrieval.profile import DocumentRetrievalProfile
from personagraph.retrieval.orchestration.reranking import BgeM3Reranker
from personagraph.retrieval.sources.picture import PictureObservationSourceAdapter
from personagraph.retrieval.sqlite_store import (
    RetrievalDataVersionRole,
    RetrievalDataVersionState,
    SqliteRetrievalCatalog,
)
from personagraph.workspace.ingestion.lifecycle import DocumentMaintenanceWorkerLifecycle
from personagraph.retrieval.operations.document_maintenance import (
    build_document_retrieval_composition,
    rollout_document_retrieval_generation,
)
from personagraph.workspace.ingestion.composition import build_document_maintenance_lifecycle
from tests.documents._authority import bound_project_document_authority


@pytest.fixture(autouse=True)
def authority(tmp_path):
    with bound_project_document_authority(tmp_path) as value:
        yield value


def test_default_document_maintenance_composition_uses_dependency_free_generation(
    authority,
    tmp_path,
):
    retrieval_path = tmp_path / "retrieval.sqlite"

    lifecycle = build_document_maintenance_lifecycle(
        worker_id="test-document-maintenance",
        retrieval_db_path=retrieval_path,
    )

    assert isinstance(lifecycle, DocumentMaintenanceWorkerLifecycle)
    assert authority.database.db_path.is_file()
    assert retrieval_path.is_file()
    assert not lifecycle.is_running
    assert (
        lifecycle._worker._indexing._generation_spec.encoder_fingerprint
        == DeterministicLexicalEncoder().fingerprint()
    )


def test_document_read_and_maintenance_compositions_share_one_exact_generation(
    tmp_path,
):
    retrieval_path = tmp_path / "retrieval.sqlite"

    composition = build_document_retrieval_composition(
        retrieval_db_path=retrieval_path,
    )
    lifecycle = build_document_maintenance_lifecycle(
        worker_id="test-shared-document-generation",
        retrieval_db_path=retrieval_path,
    )

    assert composition.foundation.generation_spec == composition.generation_spec
    assert (
        lifecycle._worker.generation_identity.version_id
        == composition.generation_spec.version_id
    )
    assert (
        lifecycle._worker.generation_identity.fingerprint
        == composition.generation_spec.fingerprint
    )
    assert (
        lifecycle._worker._chunking_profile.fingerprint()
        == composition.chunking_profile.fingerprint()
    )


def test_default_composition_keeps_online_picture_access_fail_closed(tmp_path):
    composition = build_document_retrieval_composition(
        retrieval_db_path=tmp_path / "retrieval.sqlite",
        profile=DocumentRetrievalProfile.lexical(),
    )
    picture_source = composition.foundation.source_adapters[SourceType.PICTURE]

    access = picture_source.open_retrieval_access(
        SourceFilter.from_mapping(
            SourceType.PICTURE,
            {
                "session_id": "session-1",
                "file_id": "file-1",
                "file_version_id": "version-1",
            },
        )
    )

    assert access.availability is SourceAvailability.BLOCKED
    assert access.reason_code == "picture_file_access_authority_missing"


def test_composition_forwards_an_explicit_picture_source_adapter(tmp_path):
    picture_source = PictureObservationSourceAdapter()

    composition = build_document_retrieval_composition(
        retrieval_db_path=tmp_path / "retrieval.sqlite",
        profile=DocumentRetrievalProfile.lexical(),
        picture_source_adapter=picture_source,
    )

    assert composition.foundation.source_adapters[SourceType.PICTURE] is picture_source


def test_explicit_bge_reranker_mode_is_lazy_and_visible_in_foundation_diagnostics(
    tmp_path,
):
    reranker = BgeM3Reranker()
    composition = build_document_retrieval_composition(
        retrieval_db_path=tmp_path / "retrieval.sqlite",
        profile=DocumentRetrievalProfile.lexical(),
        reranker=reranker,
    )

    assert isinstance(composition.reranker, BgeM3Reranker)
    snapshot = composition.foundation.diagnostic_snapshot()["capabilities"]["reranker"]
    assert snapshot["enabled"] is True
    assert snapshot["loaded"] is False
    assert snapshot["fingerprint"].startswith("bge_reranker_v2_m3:")


def test_explicit_off_mode_keeps_document_reranker_disabled(tmp_path):
    composition = build_document_retrieval_composition(
        retrieval_db_path=tmp_path / "retrieval.sqlite",
        profile=DocumentRetrievalProfile.lexical(),
    )

    assert composition.reranker is None
    assert composition.foundation.diagnostic_snapshot()["capabilities"]["reranker"] == {
        "enabled": False,
        "fingerprint": None,
    }


class _LexicalGeneration(DeterministicLexicalEncoder):
    def fingerprint(self) -> str:
        return "test-lexical-generation@2"


def test_document_generation_rollout_uses_the_worker_authority_fence(authority):
    chunk = DocumentChunk(
        chunk_id="chunk-rollout-1",
        text="Northbridge rollout evidence is indexed atomically.",
        span=ChunkSpan(
            start=DocumentLocator(ordinal=1, char_range=(0, 54)),
            end=DocumentLocator(ordinal=1, char_range=(0, 54)),
        ),
        section_path=(),
        element_ids=("element-rollout-1",),
        token_count=8,
        kind=ElementKind.PARAGRAPH,
    )
    authority.ingest(
        "rollout.md",
        "rollout",
        "text/markdown",
        [{"content": chunk.text, "loc": "L1"}],
        processor_fingerprint="test-reader@1",
        document_chunks=(chunk,),
        chunker_fingerprint="test-source-chunker@1",
        processing_status="complete",
        processing_diagnostics=(),
    )
    catalog = SqliteRetrievalCatalog(authority.database.db_path)
    catalog.initialize()
    catalog.create_data_version(
        version_id="old-lexical-active",
        fingerprint="old-lexical-active-fingerprint",
        role=RetrievalDataVersionRole.ACTIVE,
        state=RetrievalDataVersionState.READY,
    )

    report = rollout_document_retrieval_generation(
        profile=DocumentRetrievalProfile.lexical(),
        encoder=_LexicalGeneration(),
    )

    active = catalog.active_data_version()
    assert active is not None
    assert report.generation_id == active.id
    assert report.previous_generation_id == "old-lexical-active"
    assert catalog.get_data_version("old-lexical-active").role is (
        RetrievalDataVersionRole.PREVIOUS
    )
    assert report.method_coverage[-1].manifest_ready_count == 1
    assert report.method_coverage[-1].representation_present_count == 1
