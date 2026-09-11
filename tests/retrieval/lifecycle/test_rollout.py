from __future__ import annotations

from dataclasses import dataclass

import pytest

from personagraph.retrieval.lifecycle.backfill import RetrievalBackfillService
from personagraph.retrieval.contracts import (
    RetrievalMethod,
    SourceFilter,
    SourceType,
    SourceUnitRef,
)
from personagraph.retrieval.lifecycle.generation import (
    DOCUMENT_HYBRID_INDEX_RECIPE_CONTRACT,
    DOCUMENT_LEXICAL_INDEX_RECIPE_CONTRACT,
    file_corpus_generation_spec,
)
from personagraph.retrieval.lifecycle.rollout import (
    DocumentGenerationRolloutService,
    GenerationRolloutError,
    GenerationRolloutStage,
    GenerationRolloutStatus,
    generation_readiness_snapshot,
)
from personagraph.retrieval.indexing.encoder import DeterministicLexicalEncoder
from personagraph.retrieval.indexing.methods import (
    BgeM3EncodedText,
    SqliteRetrievalIndexWriter,
    SqliteRetrievalMethodStore,
)
from personagraph.retrieval.sqlite_store import (
    RetrievalCatalogError,
    RetrievalDataVersionRole,
    RetrievalDataVersionState,
    SqliteRetrievalCatalog,
)
from personagraph.retrieval.lifecycle.sync import IndexableSourceUnit, RetrievalSyncService


def _vector(value: float) -> tuple[float, ...]:
    return (value,) + (0.0,) * 1023


@dataclass
class FakeHybridEncoder:
    values: dict[str, BgeM3EncodedText]
    learned_sparse_available: bool = True

    def encode(self, texts):
        return tuple(self.values[text] for text in texts)

    def token_ids(self, text: str):
        return self.values[text].token_ids

    def fingerprint(self) -> str:
        return "fake-bge-m3:revision-1"


@dataclass
class SourceReader:
    source_type: SourceType
    units: tuple[IndexableSourceUnit, ...]

    def read_for_index(self, event):
        return next((unit for unit in self.units if unit.ref == event.ref), None)

    def list_indexable_units_for_backfill(self, source_filter):
        return tuple(
            unit for unit in self.units if source_filter.selects(unit.source_filter)
        )


@dataclass
class CatalogPublisher:
    catalog: SqliteRetrievalCatalog
    generation_id: str
    fail_after_ready: bool = False
    observed_active_before_publish: str | None = None
    observed_active_before_restore: str | None = None

    def publish_rebuilt_generation(self, *, expected_active_version_id):
        active = self.catalog.active_data_version()
        self.observed_active_before_publish = active.id if active is not None else None
        if self.observed_active_before_publish != expected_active_version_id:
            raise RetrievalCatalogError("active generation changed before publication")
        target = self.catalog.get_data_version(self.generation_id)
        assert target is not None
        if target.state is RetrievalDataVersionState.BUILDING:
            target = self.catalog.mark_data_version_ready(self.generation_id)
        if self.fail_after_ready:
            raise RuntimeError("simulated crash after READY")
        return self.catalog.activate_data_version(target.id)

    def restore_previous_generation(
        self,
        *,
        previous_generation_id,
        expected_fingerprint,
        expected_active_version_id,
    ):
        active = self.catalog.active_data_version()
        self.observed_active_before_restore = active.id if active is not None else None
        target = self.catalog.get_data_version(previous_generation_id)
        if target is None or target.fingerprint != expected_fingerprint:
            raise RetrievalCatalogError("rollback target identity changed")
        with self.catalog.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            return self.catalog.restore_previous_data_version_in_transaction(
                conn,
                previous_generation_id,
                expected_active_version_id=expected_active_version_id,
                require_active_match=True,
            )


def _unit(number: int) -> IndexableSourceUnit:
    content = f"第 {number} 个项目文档块"
    return IndexableSourceUnit(
        ref=SourceUnitRef(
            SourceType.DOCUMENT,
            f"document-v3:doc-{number}:chunk-{number}",
            f"version-{number}",
            f"hash-{number}",
        ),
        source_filter=SourceFilter.from_mapping(
            SourceType.DOCUMENT,
            {"doc_id": f"doc-{number}"},
        ),
        content=content,
    )


def _picture_unit(number: int) -> IndexableSourceUnit:
    content = f"第 {number} 个图片观察"
    return IndexableSourceUnit(
        ref=SourceUnitRef(
            SourceType.PICTURE,
            f"picture-observation-v1:picture-{number}:observation-{number}",
            f"picture-payload-{number}",
            f"picture-hash-{number}",
        ),
        source_filter=SourceFilter.from_mapping(
            SourceType.PICTURE,
            {
                "file_id": f"file-{number}",
                "file_version_id": f"file-version-{number}",
                "picture_id": f"picture-{number}",
            },
        ),
        content=content,
    )


def _hybrid_encoder(
    units: tuple[IndexableSourceUnit, ...],
    *,
    missing_sparse_for: str | None = None,
) -> FakeHybridEncoder:
    return FakeHybridEncoder({
        unit.content: BgeM3EncodedText(
            _vector(index + 1.0),
            (
                {}
                if unit.content == missing_sparse_for
                else {index + 10: 1.0}
            ),
            (index + 10, index + 20),
        )
        for index, unit in enumerate(units)
    })


def _spec(*, encoder_fingerprint: str, recipe: str):
    return file_corpus_generation_spec(
        encoder_fingerprint=encoder_fingerprint,
        document_chunker_fingerprint="document-chunker-v1",
        document_chunk_contract_version=1,
        index_recipe=recipe,
    )


def _harness(
    tmp_path,
    *,
    units,
    encoder,
    spec,
    picture_units=(),
    publisher=None,
    fault=None,
):
    catalog = SqliteRetrievalCatalog(tmp_path / "documents.sqlite")
    catalog.initialize()
    catalog.create_data_version(
        version_id="lexical-active-v1",
        fingerprint="lexical-active-fingerprint-v1",
        role=RetrievalDataVersionRole.ACTIVE,
        state=RetrievalDataVersionState.READY,
    )
    reader = SourceReader(SourceType.DOCUMENT, units)
    picture_reader = SourceReader(SourceType.PICTURE, picture_units)
    source_readers = {
        SourceType.DOCUMENT: reader,
        SourceType.PICTURE: picture_reader,
    }
    method_store = SqliteRetrievalMethodStore(catalog=catalog, encoder=encoder)
    sync = RetrievalSyncService(
        catalog=catalog,
        source_readers=source_readers,
        index_writer=SqliteRetrievalIndexWriter(method_store),
    )
    backfill = RetrievalBackfillService(
        catalog=catalog,
        source_readers=source_readers,
        sync_service=sync,
    )
    resolved_publisher = publisher or CatalogPublisher(catalog, spec.version_id)
    service = DocumentGenerationRolloutService(
        catalog=catalog,
        method_store=method_store,
        backfill_service=backfill,
        document_source_reader=reader,
        picture_source_reader=picture_reader,
        publisher=resolved_publisher,
        fault_injector=fault,
    )
    return catalog, reader, picture_reader, method_store, resolved_publisher, service


def test_hybrid_rollout_keeps_old_active_until_all_three_methods_are_ready(
    tmp_path,
):
    # A File generation may legitimately have no Picture observations yet.
    units = (_unit(1), _unit(2))
    encoder = _hybrid_encoder(units)
    spec = _spec(
        encoder_fingerprint=encoder.fingerprint(),
        recipe=DOCUMENT_HYBRID_INDEX_RECIPE_CONTRACT,
    )
    catalog, _, _, _, publisher, service = _harness(
        tmp_path,
        units=units,
        encoder=encoder,
        spec=spec,
    )

    report = service.rollout(spec)

    assert report.status is GenerationRolloutStatus.ACTIVATED
    assert report.source_unit_count == 2
    assert publisher.observed_active_before_publish == "lexical-active-v1"
    assert catalog.active_data_version().id == spec.version_id
    assert catalog.get_data_version("lexical-active-v1").role is RetrievalDataVersionRole.PREVIOUS
    assert {
        item.method: (
            item.required,
            item.manifest_ready_count,
            item.representation_present_count,
            item.ready,
        )
        for item in report.method_coverage
    } == {
        RetrievalMethod.DENSE: (True, 2, 2, True),
        RetrievalMethod.LEARNED_SPARSE: (True, 2, 2, True),
        RetrievalMethod.BM25: (True, 2, 2, True),
    }


def test_file_rollout_indexes_the_exact_document_picture_union(tmp_path):
    document_units = (_unit(1),)
    picture_units = (_picture_unit(1),)
    all_units = document_units + picture_units
    encoder = _hybrid_encoder(all_units)
    spec = _spec(
        encoder_fingerprint=encoder.fingerprint(),
        recipe=DOCUMENT_HYBRID_INDEX_RECIPE_CONTRACT,
    )
    catalog, _, _, _, _, service = _harness(
        tmp_path,
        units=document_units,
        picture_units=picture_units,
        encoder=encoder,
        spec=spec,
    )

    report = service.rollout(spec)

    assert report.source_unit_count == 2
    assert {
        stored.unit.ref.source_type
        for stored in catalog.list_stored_units(spec.version_id)
    } == {SourceType.DOCUMENT, SourceType.PICTURE}


def test_file_rollout_rejects_only_when_the_complete_union_is_empty(tmp_path):
    encoder = DeterministicLexicalEncoder()
    spec = _spec(
        encoder_fingerprint=encoder.fingerprint(),
        recipe=DOCUMENT_LEXICAL_INDEX_RECIPE_CONTRACT,
    )
    _, _, _, _, _, service = _harness(
        tmp_path,
        units=(),
        picture_units=(),
        encoder=encoder,
        spec=spec,
    )

    with pytest.raises(GenerationRolloutError) as failure:
        service.rollout(spec)

    assert failure.value.code == "generation_file_corpus_empty"


def test_file_rollout_rejects_picture_drift_during_backfill(tmp_path):
    document_units = (_unit(1),)
    picture_units = (_picture_unit(1),)
    encoder = _hybrid_encoder(document_units + picture_units)
    spec = _spec(
        encoder_fingerprint=encoder.fingerprint(),
        recipe=DOCUMENT_HYBRID_INDEX_RECIPE_CONTRACT,
    )
    state = {}

    def mutate_picture_manifest(stage):
        if stage is GenerationRolloutStage.BACKFILL_COMPLETE:
            state["picture_reader"].units = picture_units + (_picture_unit(2),)

    catalog, _, picture_reader, _, _, service = _harness(
        tmp_path,
        units=document_units,
        picture_units=picture_units,
        encoder=encoder,
        spec=spec,
        fault=mutate_picture_manifest,
    )
    state["picture_reader"] = picture_reader

    with pytest.raises(GenerationRolloutError) as failure:
        service.rollout(spec)

    assert failure.value.code == "generation_source_changed_during_backfill"
    assert catalog.active_data_version().id == "lexical-active-v1"

def test_rollout_fault_before_publish_leaves_old_active_and_building_staging(tmp_path):
    units = (_unit(1),)
    encoder = _hybrid_encoder(units)
    spec = _spec(
        encoder_fingerprint=encoder.fingerprint(),
        recipe=DOCUMENT_HYBRID_INDEX_RECIPE_CONTRACT,
    )

    def fail(stage):
        if stage is GenerationRolloutStage.BEFORE_PUBLISH:
            raise RuntimeError("simulated process loss")

    catalog, _, _, _, _, service = _harness(
        tmp_path,
        units=units,
        encoder=encoder,
        spec=spec,
        fault=fail,
    )

    with pytest.raises(RuntimeError, match="simulated process loss"):
        service.rollout(spec)

    assert catalog.active_data_version().id == "lexical-active-v1"
    target = catalog.get_data_version(spec.version_id)
    assert target.role is RetrievalDataVersionRole.STAGING
    assert target.state is RetrievalDataVersionState.BUILDING


def test_rollout_recovers_from_failure_after_ready_without_reindexing_active(tmp_path):
    units = (_unit(1),)
    encoder = _hybrid_encoder(units)
    spec = _spec(
        encoder_fingerprint=encoder.fingerprint(),
        recipe=DOCUMENT_HYBRID_INDEX_RECIPE_CONTRACT,
    )
    catalog, _, _, _, publisher, service = _harness(
        tmp_path,
        units=units,
        encoder=encoder,
        spec=spec,
    )
    publisher.fail_after_ready = True

    with pytest.raises(GenerationRolloutError) as failure:
        service.rollout(spec)
    assert failure.value.code == "generation_publish_failed"
    assert catalog.active_data_version().id == "lexical-active-v1"
    assert catalog.get_data_version(spec.version_id).state is RetrievalDataVersionState.READY

    publisher.fail_after_ready = False
    report = service.rollout(spec)

    assert report.status is GenerationRolloutStatus.ACTIVATED
    assert catalog.active_data_version().id == spec.version_id


def test_hybrid_rollout_refuses_dense_bm25_partial_generation(tmp_path):
    units = (_unit(1),)
    encoder = _hybrid_encoder(units, missing_sparse_for=units[0].content)
    spec = _spec(
        encoder_fingerprint=encoder.fingerprint(),
        recipe=DOCUMENT_HYBRID_INDEX_RECIPE_CONTRACT,
    )
    catalog, _, _, _, _, service = _harness(
        tmp_path,
        units=units,
        encoder=encoder,
        spec=spec,
    )

    with pytest.raises(GenerationRolloutError) as failure:
        service.rollout(spec)

    assert failure.value.code == "generation_method_not_ready:learned_sparse"
    assert catalog.active_data_version().id == "lexical-active-v1"
    target = catalog.get_data_version(spec.version_id)
    assert target.role is RetrievalDataVersionRole.STAGING
    assert target.state is RetrievalDataVersionState.BUILDING


def test_lexical_recipe_requires_bm25_and_requires_dense_sparse_to_be_absent(tmp_path):
    units = (_unit(1),)
    encoder = DeterministicLexicalEncoder()
    spec = _spec(
        encoder_fingerprint=encoder.fingerprint(),
        recipe=DOCUMENT_LEXICAL_INDEX_RECIPE_CONTRACT,
    )
    _, _, _, _, _, service = _harness(
        tmp_path,
        units=units,
        encoder=encoder,
        spec=spec,
    )

    report = service.rollout(spec)

    coverage = {item.method: item for item in report.method_coverage}
    assert coverage[RetrievalMethod.BM25].required
    assert coverage[RetrievalMethod.BM25].ready
    assert not coverage[RetrievalMethod.DENSE].required
    assert coverage[RetrievalMethod.DENSE].manifest_absent_count == 1
    assert coverage[RetrievalMethod.DENSE].representation_present_count == 0
    assert coverage[RetrievalMethod.DENSE].ready
    assert coverage[RetrievalMethod.LEARNED_SPARSE].ready


def test_same_spec_ready_previous_generation_is_atomically_restored(tmp_path):
    units = (_unit(1), _unit(2))
    encoder = _hybrid_encoder(units)
    spec = _spec(
        encoder_fingerprint=encoder.fingerprint(),
        recipe=DOCUMENT_HYBRID_INDEX_RECIPE_CONTRACT,
    )
    catalog, _, _, _, publisher, service = _harness(
        tmp_path,
        units=units,
        encoder=encoder,
        spec=spec,
    )
    service.rollout(spec)
    catalog.create_data_version(
        version_id="replacement-active-v2",
        fingerprint="replacement-fingerprint-v2",
        role=RetrievalDataVersionRole.STAGING,
        state=RetrievalDataVersionState.READY,
    )
    catalog.activate_data_version("replacement-active-v2")

    report = service.rollout(spec)

    assert report.status is GenerationRolloutStatus.ACTIVATED
    assert report.previous_generation_id == "replacement-active-v2"
    assert publisher.observed_active_before_restore == "replacement-active-v2"
    assert catalog.active_data_version().id == spec.version_id


def test_same_spec_previous_with_missing_representation_is_rebuilt(tmp_path):
    units = (_unit(1), _unit(2))
    encoder = _hybrid_encoder(units)
    spec = _spec(
        encoder_fingerprint=encoder.fingerprint(),
        recipe=DOCUMENT_HYBRID_INDEX_RECIPE_CONTRACT,
    )
    catalog, _, _, method_store, _, service = _harness(
        tmp_path,
        units=units,
        encoder=encoder,
        spec=spec,
    )
    service.rollout(spec)
    catalog.create_data_version(
        version_id="replacement-active-v2",
        fingerprint="replacement-fingerprint-v2",
        role=RetrievalDataVersionRole.STAGING,
        state=RetrievalDataVersionState.READY,
    )
    catalog.activate_data_version("replacement-active-v2")
    damaged = catalog.list_stored_units(spec.version_id)[0]
    with catalog.connect() as conn:
        conn.execute(
            "DELETE FROM learned_sparse_postings WHERE unit_id=?",
            (damaged.unit_id,),
        )

    report = service.rollout(spec)

    assert report.status is GenerationRolloutStatus.ACTIVATED
    assert catalog.active_data_version().id == spec.version_id
    health = method_store.generation_method_index_health(spec.version_id)
    assert all(
        entry.representation_present
        for entries in health.values()
        for entry in entries
    )


def test_safe_rollback_refuses_current_document_source_drift(tmp_path):
    units = (_unit(1), _unit(2))
    encoder = _hybrid_encoder(units + (_unit(3),))
    spec = _spec(
        encoder_fingerprint=encoder.fingerprint(),
        recipe=DOCUMENT_HYBRID_INDEX_RECIPE_CONTRACT,
    )
    catalog, reader, _, _, _, service = _harness(
        tmp_path,
        units=units,
        encoder=encoder,
        spec=spec,
    )
    service.rollout(spec)
    catalog.create_data_version(
        version_id="replacement-active-v2",
        fingerprint="replacement-fingerprint-v2",
        role=RetrievalDataVersionRole.STAGING,
        state=RetrievalDataVersionState.READY,
    )
    catalog.activate_data_version("replacement-active-v2")
    reader.units = units + (_unit(3),)

    with pytest.raises(GenerationRolloutError) as failure:
        service.rollback(
            spec,
            previous_generation_id=spec.version_id,
            expected_fingerprint=spec.fingerprint,
        )

    assert failure.value.code == "generation_file_coverage_incomplete"
    assert catalog.active_data_version().id == "replacement-active-v2"


@pytest.mark.parametrize(
    "current_picture_units",
    [(), (_picture_unit(1), _picture_unit(2))],
    ids=("missing-current-picture", "extra-current-picture"),
)
def test_safe_rollback_requires_the_exact_current_picture_manifest(
    tmp_path,
    current_picture_units,
):
    document_units = (_unit(1),)
    initial_picture_units = (_picture_unit(1),)
    encoder = _hybrid_encoder(document_units + initial_picture_units)
    spec = _spec(
        encoder_fingerprint=encoder.fingerprint(),
        recipe=DOCUMENT_HYBRID_INDEX_RECIPE_CONTRACT,
    )
    catalog, _, picture_reader, _, _, service = _harness(
        tmp_path,
        units=document_units,
        picture_units=initial_picture_units,
        encoder=encoder,
        spec=spec,
    )
    service.rollout(spec)
    catalog.create_data_version(
        version_id="replacement-active-v2",
        fingerprint="replacement-fingerprint-v2",
        role=RetrievalDataVersionRole.STAGING,
        state=RetrievalDataVersionState.READY,
    )
    catalog.activate_data_version("replacement-active-v2")
    picture_reader.units = current_picture_units

    with pytest.raises(GenerationRolloutError) as failure:
        service.rollback(
            spec,
            previous_generation_id=spec.version_id,
            expected_fingerprint=spec.fingerprint,
        )

    assert failure.value.code == "generation_file_coverage_incomplete"
    assert catalog.active_data_version().id == "replacement-active-v2"


def test_empty_generation_readiness_snapshot_keeps_complete_false_shape(tmp_path):
    catalog = SqliteRetrievalCatalog(tmp_path / "documents.sqlite")
    catalog.initialize()
    catalog.create_data_version(
        version_id="empty-staging",
        fingerprint="empty-staging-fingerprint",
    )
    method_store = SqliteRetrievalMethodStore(
        catalog=catalog,
        encoder=DeterministicLexicalEncoder(),
    )

    snapshot = generation_readiness_snapshot(
        catalog=catalog,
        method_store=method_store,
        generation_id="empty-staging",
        required_methods=(RetrievalMethod.BM25,),
    )

    assert [item.method for item in snapshot] == [
        RetrievalMethod.DENSE,
        RetrievalMethod.LEARNED_SPARSE,
        RetrievalMethod.BM25,
    ]
    assert all(item.unit_count == 0 and not item.ready for item in snapshot)
