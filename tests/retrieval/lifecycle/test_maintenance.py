from __future__ import annotations

from dataclasses import dataclass

from personagraph.retrieval.contracts import (
    RetrievalMethod,
    SourceFilter,
    SourceType,
    SourceUnitRef,
)
from personagraph.retrieval.lifecycle.maintenance import ReconciliationIssueCode, RetrievalReconciler
from personagraph.retrieval.indexing.methods import (
    BgeM3EncodedText,
    SqliteRetrievalIndexWriter,
    SqliteRetrievalMethodStore,
)
from personagraph.retrieval.lifecycle.outbox import RetrievalUpdateEvent, RetrievalUpdateKind
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
    mapping: dict[str, BgeM3EncodedText]

    def encode(self, texts):
        return tuple(self.mapping[text] for text in texts)

    def token_ids(self, text: str):
        return self.mapping[text].token_ids

    def fingerprint(self) -> str:
        return "fake-bge"


@dataclass
class Reader:
    source_type: SourceType
    unit: IndexableSourceUnit | None

    def read_for_index(self, event):
        if self.unit is None or event.ref != self.unit.ref:
            return None
        return self.unit

    def read_current_for_reconcile(self, ref):
        if self.unit is None or ref.source_type is not self.source_type:
            return None
        return self.unit


def _setup(tmp_path):
    content = "北桥本周交付初稿"
    source_filter = SourceFilter.from_mapping(SourceType.CURRENT_SESSION, {"session_id": "s1"})
    ref = SourceUnitRef(SourceType.CURRENT_SESSION, "s1:run-1", "r1", "h1")
    source = IndexableSourceUnit(ref=ref, source_filter=source_filter, content=content)
    catalog = SqliteRetrievalCatalog(tmp_path / "retrieval.sqlite")
    catalog.initialize()
    catalog.create_data_version(
        version_id="v1",
        fingerprint="fake-bge",
        role=RetrievalDataVersionRole.ACTIVE,
        state=RetrievalDataVersionState.READY,
    )
    encoder = FakeEncoder({content: BgeM3EncodedText(_vector(1.0), {7: 1.0}, (7, 8))})
    methods = SqliteRetrievalMethodStore(catalog=catalog, encoder=encoder)
    reader = Reader(SourceType.CURRENT_SESSION, source)
    sync = RetrievalSyncService(
        catalog=catalog,
        source_readers={SourceType.CURRENT_SESSION: reader},
        index_writer=SqliteRetrievalIndexWriter(methods),
    )
    event = RetrievalUpdateEvent(
        event_id="upsert-1",
        kind=RetrievalUpdateKind.UPSERT,
        ref=ref,
        retrieval_data_version="v1",
        occurred_at="2026-07-22T00:00:00+00:00",
    )
    assert sync.apply(event).status.value == "applied"
    reconciler = RetrievalReconciler(
        catalog=catalog,
        method_store=methods,
        source_readers={SourceType.CURRENT_SESSION: reader},
        sync_service=sync,
    )
    return catalog, methods, reader, reconciler


def test_reconcile_detects_missing_representation_and_repair_rebuilds_it(tmp_path):
    catalog, _, _, reconciler = _setup(tmp_path)
    stored = catalog.active_units("v1")[0]
    with catalog.connect() as conn:
        conn.execute("DELETE FROM dense_vectors WHERE unit_id=?", (stored.unit_id,))

    report = reconciler.scan()
    missing = [issue for issue in report.issues if issue.code is ReconciliationIssueCode.EXPECTED_REPRESENTATION_MISSING]
    assert [(issue.unit_id, issue.method.value) for issue in missing] == [(stored.unit_id, "dense")]

    results = reconciler.repair(missing)
    assert [result.status for result in results] == ["applied"]
    assert reconciler.scan().healthy


def test_reconcile_reports_source_drift_but_never_auto_corrects_authority(tmp_path):
    _, _, reader, reconciler = _setup(tmp_path)
    source = reader.unit
    assert source is not None
    drifted_ref = SourceUnitRef(SourceType.CURRENT_SESSION, "s1:run-1", "r2", "h2")
    reader.unit = IndexableSourceUnit(ref=drifted_ref, source_filter=source.source_filter, content=source.content)

    report = reconciler.scan()
    drift = [issue for issue in report.issues if issue.code is ReconciliationIssueCode.SOURCE_REF_MISMATCH]
    assert len(drift) == 1
    result = reconciler.repair(drift)[0]
    assert result.action == "requires_authority_event"
    assert result.status == "skipped"


def test_reconcile_purges_dense_orphan_that_has_no_catalog_unit(tmp_path):
    catalog, methods, _, reconciler = _setup(tmp_path)
    stored = catalog.active_units("v1")[0]
    catalog.delete_unit(stored.unit_id)

    report = reconciler.scan("v1")
    orphan = [issue for issue in report.issues if issue.code is ReconciliationIssueCode.ORPHAN_DERIVED_REPRESENTATION]
    assert [(issue.unit_id, issue.method.value) for issue in orphan] == [(stored.unit_id, "dense")]
    assert reconciler.repair(orphan)[0].action == "purge_orphaned_derived"
    assert methods.orphaned_representation_unit_ids()[RetrievalMethod.DENSE] == ()
