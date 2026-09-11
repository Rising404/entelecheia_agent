from __future__ import annotations

from dataclasses import dataclass

from personagraph.retrieval.lifecycle.backfill import RetrievalBackfillService
from personagraph.retrieval.contracts import SourceFilter, SourceType, SourceUnitRef
from personagraph.retrieval.indexing.methods import (
    BgeM3EncodedText,
    SqliteRetrievalIndexWriter,
    SqliteRetrievalMethodStore,
)
from personagraph.retrieval.sqlite_store import (
    RetrievalDataVersionRole,
    RetrievalDataVersionState,
    SqliteRetrievalCatalog,
)
from personagraph.retrieval.lifecycle.sync import IndexableSourceUnit, RetrievalSyncService


def _vector(value: float) -> tuple[float, ...]:
    return (value,) + (0.0,) * 1023


@dataclass
class FakeEncoder:
    values: dict[str, BgeM3EncodedText]

    def encode(self, texts):
        return tuple(self.values[text] for text in texts)

    def token_ids(self, text: str):
        return self.values[text].token_ids

    def fingerprint(self) -> str:
        return "fake-bge"


@dataclass
class BackfillReader:
    source_type: SourceType
    units: tuple[IndexableSourceUnit, ...]

    def read_for_index(self, event):
        return next((unit for unit in self.units if unit.ref == event.ref), None)

    def list_indexable_units_for_backfill(self, source_filter):
        return self.units


def _service(tmp_path, units: tuple[IndexableSourceUnit, ...]):
    catalog = SqliteRetrievalCatalog(tmp_path / "retrieval.sqlite")
    catalog.initialize()
    catalog.create_data_version(
        version_id="staging-v1",
        fingerprint="fake-bge",
        role=RetrievalDataVersionRole.STAGING,
        state=RetrievalDataVersionState.BUILDING,
    )
    encoder = FakeEncoder(
        {
            unit.content: BgeM3EncodedText(_vector(index + 1), {index + 1: 1.0}, (index + 1,))
            for index, unit in enumerate(units)
        }
    )
    methods = SqliteRetrievalMethodStore(catalog=catalog, encoder=encoder)
    reader = BackfillReader(SourceType.CURRENT_SESSION, units)
    sync = RetrievalSyncService(
        catalog=catalog,
        source_readers={SourceType.CURRENT_SESSION: reader},
        index_writer=SqliteRetrievalIndexWriter(methods),
    )
    return catalog, RetrievalBackfillService(
        catalog=catalog,
        source_readers={SourceType.CURRENT_SESSION: reader},
        sync_service=sync,
    )


def test_backfill_is_explicitly_scoped_deterministic_and_idempotent(tmp_path):
    scope = SourceFilter.from_mapping(SourceType.CURRENT_SESSION, {"session_id": "s1"})
    foreign_scope = SourceFilter.from_mapping(SourceType.CURRENT_SESSION, {"session_id": "s2"})
    valid = IndexableSourceUnit(
        SourceUnitRef(SourceType.CURRENT_SESSION, "s1:run-1", "r1", "h1"),
        scope,
        "北桥交付初稿",
    )
    foreign = IndexableSourceUnit(
        SourceUnitRef(SourceType.CURRENT_SESSION, "s2:run-1", "r1", "h2"),
        foreign_scope,
        "不应越过当前会话范围",
    )
    catalog, service = _service(tmp_path, (foreign, valid))

    first = service.backfill(
        data_version_id="staging-v1",
        source_filters=(scope,),
        occurred_at="2026-07-22T00:00:00+00:00",
    )
    second = service.backfill(
        data_version_id="staging-v1",
        source_filters=(scope,),
        occurred_at="2026-07-22T00:01:00+00:00",
    )

    assert first.applied == 1
    assert first.source_reports[0].discovered == 2
    assert first.source_reports[0].skipped == 1
    assert first.source_reports[0].reason_codes == ("source_backfill_scope_mismatch",)
    assert second.applied == 0
    assert second.source_reports[0].replayed == 1
    assert len(catalog.active_units("staging-v1")) == 1


def test_backfill_reports_uninstalled_source_without_global_fallback(tmp_path):
    scope = SourceFilter.from_mapping(SourceType.CURRENT_SESSION, {"session_id": "s1"})
    _, service = _service(
        tmp_path,
        (
            IndexableSourceUnit(
                SourceUnitRef(SourceType.CURRENT_SESSION, "s1:run-1", "r1", "h1"),
                scope,
                "北桥交付初稿",
            ),
        ),
    )
    uninstalled_document = SourceFilter.from_mapping(
        SourceType.DOCUMENT,
        {"doc_id": "doc-1"},
    )

    report = service.backfill(
        data_version_id="staging-v1",
        source_filters=(uninstalled_document,),
    )

    assert report.applied == 0
    assert report.failed == 0
    assert report.source_reports[0].reason_codes == ("source_backfill_not_supported",)
