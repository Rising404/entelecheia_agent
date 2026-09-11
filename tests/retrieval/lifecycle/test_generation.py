from __future__ import annotations

from dataclasses import replace

import pytest

from personagraph.retrieval.contracts import (
    RetrievalStatus,
    RetrievalUnit,
    SourceFilter,
    SourceType,
    SourceUnitRef,
)
from personagraph.retrieval.lifecycle.generation import (
    DOCUMENT_HYBRID_INDEX_RECIPE_CONTRACT,
    DOCUMENT_LEXICAL_INDEX_RECIPE_CONTRACT,
    DOCUMENT_SOURCE_IDENTITY_CONTRACT,
    ExactGenerationDataVersionProvider,
    RetrievalGenerationMismatch,
    RetrievalGenerationRestoreStatus,
    RetrievalGenerationSpec,
    document_paper_generation_spec,
    file_corpus_generation_spec,
    plan_previous_generation_restore,
    require_exact_active_generation,
)
from personagraph.workspace.pictures.observations import (
    PictureObservationWindowPolicy,
)
from personagraph.retrieval.sqlite_store import (
    RetrievalCatalogError,
    RetrievalDataVersionRole,
    RetrievalDataVersionState,
    SqliteRetrievalCatalog,
)


def _spec() -> RetrievalGenerationSpec:
    return RetrievalGenerationSpec(
        encoder_fingerprint="encoder-v1",
        chunker_fingerprint="chunker-v1",
        document_chunk_contract_version=1,
        retrieval_catalog_schema_version=4,
        source_identity_contract=DOCUMENT_SOURCE_IDENTITY_CONTRACT,
        index_recipe="dense1024+learned_sparse+bm25@1",
        source_types=("document",),
    )


def _add_ready_document_unit(
    catalog: SqliteRetrievalCatalog,
    *,
    data_version_id: str,
) -> None:
    stored = catalog.upsert_pending_unit(
        RetrievalUnit(
            ref=SourceUnitRef(
                source_type=SourceType.DOCUMENT,
                source_unit_id="document-v3:doc-1:chunk-1",
                source_revision="version-1",
                indexed_content_hash="hash-1",
            ),
            retrieval_data_version=data_version_id,
            retrieval_status=RetrievalStatus.ACTIVE,
            source_filter=SourceFilter.from_mapping(
                SourceType.DOCUMENT,
                {"doc_id": "doc-1"},
            ),
        )
    )
    catalog.mark_unit_index_ready(stored.unit_id)


def _restore_previous_in_write_transaction(
    catalog: SqliteRetrievalCatalog,
    generation_id: str,
    *,
    expected_active_generation_id: str | None,
):
    with catalog.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        return catalog.restore_previous_data_version_in_transaction(
            conn,
            generation_id,
            expected_active_version_id=expected_active_generation_id,
            require_active_match=True,
        )


def test_generation_fingerprint_is_canonical_and_binds_every_derived_authority():
    spec = _spec()

    assert spec.fingerprint == _spec().fingerprint
    assert spec.version_id.startswith("rdv_")
    assert len(spec.version_id) == 36
    for changed in (
        replace(spec, encoder_fingerprint="encoder-v2"),
        replace(spec, chunker_fingerprint="chunker-v2"),
        replace(spec, document_chunk_contract_version=2),
        replace(spec, retrieval_catalog_schema_version=5),
        replace(spec, source_identity_contract="project-document-source-identity-v4"),
        replace(spec, index_recipe="dense1024+bm25@2"),
    ):
        assert changed.fingerprint != spec.fingerprint
        assert changed.version_id != spec.version_id

    paper = document_paper_generation_spec(
        encoder_fingerprint="encoder-v1",
        chunker_fingerprint="chunker-v1",
        document_chunk_contract_version=1,
        index_recipe=DOCUMENT_HYBRID_INDEX_RECIPE_CONTRACT,
    )
    assert DOCUMENT_SOURCE_IDENTITY_CONTRACT == "project-document-source-identity-v3"
    assert paper.source_identity_contract == DOCUMENT_SOURCE_IDENTITY_CONTRACT
    assert paper.source_types == ("document",)
    assert paper.retrieval_catalog_schema_version == 4

    lexical = document_paper_generation_spec(
        encoder_fingerprint="lexical-v1",
        chunker_fingerprint="chunker-v1",
        document_chunk_contract_version=1,
        index_recipe=DOCUMENT_LEXICAL_INDEX_RECIPE_CONTRACT,
    )
    assert lexical.index_recipe == "rag-r1:bm25@1"
    assert lexical.version_id != paper.version_id


def test_generation_spec_rejects_ambiguous_or_noncanonical_shapes():
    with pytest.raises(ValueError, match="canonical"):
        replace(_spec(), source_types=("document", "document"))
    with pytest.raises(ValueError, match="canonical"):
        replace(_spec(), source_types=("document", "current_session"))
    with pytest.raises(ValueError, match="positive"):
        replace(_spec(), document_chunk_contract_version=0)
    with pytest.raises(ValueError, match="non-empty"):
        replace(_spec(), encoder_fingerprint=" ")
    with pytest.raises(TypeError, match="immutable tuple"):
        replace(_spec(), source_bindings=[])


def test_file_generation_binds_picture_observation_contract_and_fifo_policy():
    baseline = file_corpus_generation_spec(
        encoder_fingerprint="encoder-v1",
        document_chunker_fingerprint="chunker-v1",
        document_chunk_contract_version=1,
        index_recipe=DOCUMENT_HYBRID_INDEX_RECIPE_CONTRACT,
    )
    changed_contract = file_corpus_generation_spec(
        encoder_fingerprint="encoder-v1",
        document_chunker_fingerprint="chunker-v1",
        document_chunk_contract_version=1,
        index_recipe=DOCUMENT_HYBRID_INDEX_RECIPE_CONTRACT,
        picture_observation_contract="picture-observation-payload-v2",
    )
    changed_fifo = file_corpus_generation_spec(
        encoder_fingerprint="encoder-v1",
        document_chunker_fingerprint="chunker-v1",
        document_chunk_contract_version=1,
        index_recipe=DOCUMENT_HYBRID_INDEX_RECIPE_CONTRACT,
        picture_window_policy=PictureObservationWindowPolicy(
            max_active_entries=9
        ),
    )

    assert baseline.source_types == ("document", "picture")
    assert baseline.source_binding(SourceType.DOCUMENT).projection_fingerprint == (
        "chunker-v1"
    )
    assert baseline.source_binding(SourceType.PICTURE).projection_contract == (
        "picture-observation-payload-v1"
    )
    assert '"sources"' in baseline.canonical_json
    assert '"chunker_fingerprint"' not in baseline.canonical_json
    assert changed_contract.fingerprint != baseline.fingerprint
    assert changed_fifo.fingerprint != baseline.fingerprint


def test_active_generation_must_be_ready_and_exactly_match_runtime_spec(tmp_path):
    catalog = SqliteRetrievalCatalog(tmp_path / "retrieval.sqlite")
    catalog.initialize()
    spec = _spec()

    with pytest.raises(RetrievalGenerationMismatch) as missing:
        require_exact_active_generation(catalog, spec)
    assert missing.value.code == "active_generation_missing"

    catalog.create_data_version(
        version_id=spec.version_id,
        fingerprint="wrong-generation",
        role=RetrievalDataVersionRole.ACTIVE,
        state=RetrievalDataVersionState.READY,
    )
    with pytest.raises(RetrievalGenerationMismatch) as mismatch:
        require_exact_active_generation(catalog, spec)
    assert mismatch.value.code == "active_generation_fingerprint_mismatch"

    exact_catalog = SqliteRetrievalCatalog(tmp_path / "exact.sqlite")
    exact_catalog.initialize()
    expected = exact_catalog.create_data_version(
        version_id=spec.version_id,
        fingerprint=spec.fingerprint,
        role=RetrievalDataVersionRole.ACTIVE,
        state=RetrievalDataVersionState.READY,
    )
    assert require_exact_active_generation(exact_catalog, spec) == expected

    building_catalog = SqliteRetrievalCatalog(tmp_path / "building.sqlite")
    building_catalog.initialize()
    building_catalog.create_data_version(
        version_id=spec.version_id,
        fingerprint=spec.fingerprint,
        role=RetrievalDataVersionRole.ACTIVE,
        state=RetrievalDataVersionState.BUILDING,
    )
    with pytest.raises(RetrievalGenerationMismatch) as not_ready:
        require_exact_active_generation(building_catalog, spec)
    assert not_ready.value.code == "active_generation_not_ready"


def test_query_provider_never_exposes_an_active_generation_with_a_drifted_spec(tmp_path):
    catalog = SqliteRetrievalCatalog(tmp_path / "retrieval.sqlite")
    catalog.initialize()
    spec = _spec()
    catalog.create_data_version(
        version_id="legacy-id",
        fingerprint="encoder-only-legacy-fingerprint",
        role=RetrievalDataVersionRole.ACTIVE,
        state=RetrievalDataVersionState.READY,
    )

    provider = ExactGenerationDataVersionProvider(catalog=catalog, spec=spec)

    assert provider.active_data_version() is None
    assert provider.active_retrieval_data_version_id() is None


def test_query_provider_exposes_the_exact_active_ready_generation_id(tmp_path):
    catalog = SqliteRetrievalCatalog(tmp_path / "retrieval.sqlite")
    catalog.initialize()
    spec = _spec()
    catalog.create_data_version(
        version_id=spec.version_id,
        fingerprint=spec.fingerprint,
        role=RetrievalDataVersionRole.ACTIVE,
        state=RetrievalDataVersionState.READY,
    )

    provider = ExactGenerationDataVersionProvider(catalog=catalog, spec=spec)

    assert provider.active_retrieval_data_version_id() == spec.version_id


def test_previous_generation_restore_plan_precedes_low_level_atomic_pointer_swap(tmp_path):
    catalog = SqliteRetrievalCatalog(tmp_path / "retrieval.sqlite")
    catalog.initialize()
    catalog.create_data_version(
        version_id="v1",
        fingerprint="fingerprint-v1",
        role=RetrievalDataVersionRole.ACTIVE,
        state=RetrievalDataVersionState.READY,
    )
    _add_ready_document_unit(catalog, data_version_id="v1")
    catalog.create_data_version(
        version_id="v2",
        fingerprint="fingerprint-v2",
        role=RetrievalDataVersionRole.STAGING,
        state=RetrievalDataVersionState.READY,
    )
    catalog.activate_data_version("v2")

    plan = plan_previous_generation_restore(
        catalog,
        expected_fingerprint="fingerprint-v1",
    )
    restored = _restore_previous_in_write_transaction(
        catalog,
        "v1",
        expected_active_generation_id=plan.active_generation_id,
    )

    assert plan.status is RetrievalGenerationRestoreStatus.RESTORABLE
    assert plan.target_generation_id == "v1"
    assert restored.id == "v1"
    assert catalog.active_data_version().id == "v1"
    assert catalog.get_data_version("v2").role is RetrievalDataVersionRole.PREVIOUS


def test_empty_previous_generation_requires_rebuild_and_cannot_be_activated(tmp_path):
    catalog = SqliteRetrievalCatalog(tmp_path / "retrieval.sqlite")
    catalog.initialize()
    catalog.create_data_version(
        version_id="empty-v1",
        fingerprint="fingerprint-v1",
        role=RetrievalDataVersionRole.ACTIVE,
        state=RetrievalDataVersionState.READY,
    )
    catalog.create_data_version(
        version_id="v2",
        fingerprint="fingerprint-v2",
        role=RetrievalDataVersionRole.STAGING,
        state=RetrievalDataVersionState.READY,
    )
    catalog.activate_data_version("v2")

    plan = plan_previous_generation_restore(
        catalog,
        target_generation_id="empty-v1",
        expected_fingerprint="fingerprint-v1",
    )

    assert plan.status is RetrievalGenerationRestoreStatus.REBUILD_REQUIRED
    assert plan.reason_code == "previous_generation_empty"
    with pytest.raises(RetrievalCatalogError, match="empty previous"):
        _restore_previous_in_write_transaction(
            catalog,
            "empty-v1",
            expected_active_generation_id="v2",
        )
    assert catalog.active_data_version().id == "v2"


def test_previous_generation_restore_plan_requires_rebuild_on_fingerprint_drift(tmp_path):
    catalog = SqliteRetrievalCatalog(tmp_path / "retrieval.sqlite")
    catalog.initialize()
    catalog.create_data_version(
        version_id="v1",
        fingerprint="fingerprint-v1",
        role=RetrievalDataVersionRole.PREVIOUS,
        state=RetrievalDataVersionState.READY,
    )

    result = plan_previous_generation_restore(
        catalog,
        target_generation_id="v1",
        expected_fingerprint="a-new-runtime-fingerprint",
    )

    assert result.status is RetrievalGenerationRestoreStatus.REBUILD_REQUIRED
    assert result.reason_code == "previous_generation_fingerprint_mismatch"
    assert catalog.active_data_version() is None


def test_low_level_restore_rejects_active_pointer_drift_after_diagnosis(tmp_path):
    catalog = SqliteRetrievalCatalog(tmp_path / "retrieval.sqlite")
    catalog.initialize()
    catalog.create_data_version(
        version_id="v1",
        fingerprint="fingerprint-v1",
        role=RetrievalDataVersionRole.ACTIVE,
        state=RetrievalDataVersionState.READY,
    )
    catalog.create_data_version(
        version_id="v2",
        fingerprint="fingerprint-v2",
        role=RetrievalDataVersionRole.STAGING,
        state=RetrievalDataVersionState.READY,
    )
    catalog.activate_data_version("v2")

    with pytest.raises(RetrievalCatalogError, match="changed after rollback diagnosis"):
        _restore_previous_in_write_transaction(
            catalog,
            "v1",
            expected_active_generation_id="a-different-active-generation",
        )

    assert catalog.active_data_version().id == "v2"
    assert catalog.get_data_version("v1").role is RetrievalDataVersionRole.PREVIOUS
