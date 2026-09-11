from __future__ import annotations

from dataclasses import dataclass, field
import sqlite3

import pytest

from personagraph.retrieval.contracts import (
    RetrievalStatus,
    SourceFilter,
    SourceType,
    SourceUnitRef,
)
from personagraph.retrieval.indexing.encoder import RetrievalMethodUnavailable
from personagraph.retrieval.lifecycle.outbox import (
    OutboxStatus,
    RetrievalUpdateEvent,
    RetrievalUpdateKind,
    SqliteRetrievalOutbox,
)
from personagraph.retrieval.sqlite_store import (
    RetrievalDataVersionRole,
    RetrievalDataVersionState,
    SqliteRetrievalCatalog,
    SyncReceiptStatus,
)
from personagraph.retrieval.lifecycle.sync import (
    IndexableSourceUnit,
    RetrievalOutboxConsumer,
    RetrievalSyncService,
    SyncApplyStatus,
)


NOW = "2026-07-22T00:00:00+00:00"


def _ref(revision: str = "r1", content_hash: str = "h1") -> SourceUnitRef:
    return SourceUnitRef(SourceType.CURRENT_SESSION, "turn-pair-1", revision, content_hash)


def _event(kind: RetrievalUpdateKind, event_id: str, ref: SourceUnitRef | None = None):
    return RetrievalUpdateEvent(
        event_id=event_id,
        kind=kind,
        ref=ref or _ref(),
        retrieval_data_version="v1",
        occurred_at=NOW,
    )


@dataclass
class FakeReader:
    source_type: SourceType = SourceType.CURRENT_SESSION
    units: dict[SourceUnitRef, IndexableSourceUnit] = field(default_factory=dict)

    def read_for_index(self, event):
        return self.units.get(event.ref)


@dataclass
class FakeWriter:
    indexed: list[tuple[int, str]] = field(default_factory=list)
    purged: list[int] = field(default_factory=list)
    fail: bool = False
    unavailable: bool = False

    def index(self, stored_unit, source_unit):
        if self.unavailable:
            raise RetrievalMethodUnavailable("simulated_encoder_dependency_missing")
        if self.fail:
            raise RuntimeError("simulated_index_write_failure")
        self.indexed.append((stored_unit.unit_id, source_unit.content))

    def purge(self, stored_unit):
        if self.fail:
            raise RuntimeError("simulated_index_purge_failure")
        self.purged.append(stored_unit.unit_id)


def _catalog(tmp_path):
    catalog = SqliteRetrievalCatalog(tmp_path / "retrieval.sqlite")
    catalog.initialize()
    catalog.create_data_version(
        version_id="v1",
        fingerprint="test-fingerprint",
        role=RetrievalDataVersionRole.ACTIVE,
        state=RetrievalDataVersionState.READY,
    )
    return catalog


def _source_unit(ref: SourceUnitRef | None = None):
    ref = ref or _ref()
    return IndexableSourceUnit(
        ref=ref,
        source_filter=SourceFilter.from_mapping(SourceType.CURRENT_SESSION, {"session_id": "s1"}),
        content="北桥项目本周需要交付",
    )


def _service(tmp_path, *, writer: FakeWriter | None = None, reader: FakeReader | None = None):
    catalog = _catalog(tmp_path)
    reader = reader or FakeReader(units={_ref(): _source_unit()})
    writer = writer or FakeWriter()
    service = RetrievalSyncService(
        catalog=catalog,
        source_readers={SourceType.CURRENT_SESSION: reader},
        index_writer=writer,
    )
    return catalog, reader, writer, service


def test_upsert_indexes_once_publishes_after_success_and_replays_idempotently(tmp_path):
    catalog, _, writer, service = _service(tmp_path)
    event = _event(RetrievalUpdateKind.UPSERT, "event-1")

    first = service.apply(event)
    replay = service.apply(event)

    assert first.status is SyncApplyStatus.APPLIED
    assert replay.status is SyncApplyStatus.APPLIED
    assert replay.replayed is True
    assert len(writer.indexed) == 1
    active = catalog.active_units("v1")
    assert len(active) == 1
    assert active[0].unit.ref == event.ref
    assert catalog.get_receipt(event.event_id).status is SyncReceiptStatus.APPLIED


def test_applied_receipt_rebuilds_when_derived_unit_was_removed(tmp_path):
    catalog, _, writer, service = _service(tmp_path)
    event = _event(RetrievalUpdateKind.UPSERT, "event-1")
    assert service.apply(event).status is SyncApplyStatus.APPLIED
    stored = catalog.get_unit(event.ref, event.retrieval_data_version)
    assert stored is not None
    catalog.delete_unit(stored.unit_id)

    recovered = service.apply(event)

    assert recovered.status is SyncApplyStatus.APPLIED
    assert recovered.replayed is False
    assert len(writer.indexed) == 2
    assert catalog.get_unit(event.ref, event.retrieval_data_version) is not None


def test_index_failure_never_publishes_active_unit_and_is_retryable(tmp_path):
    writer = FakeWriter(fail=True)
    catalog, _, _, service = _service(tmp_path, writer=writer)

    result = service.apply(_event(RetrievalUpdateKind.UPSERT, "event-1"))

    assert result.status is SyncApplyStatus.RETRYABLE_FAILED
    assert catalog.active_units("v1") == []
    assert catalog.get_receipt("event-1").status is SyncReceiptStatus.RETRYABLE_FAILED


def test_unavailable_retrieval_capability_is_terminal_not_an_infinite_retry(tmp_path):
    writer = FakeWriter(unavailable=True)
    catalog, _, _, service = _service(tmp_path, writer=writer)

    result = service.apply(_event(RetrievalUpdateKind.UPSERT, "event-1"))

    assert result.status is SyncApplyStatus.TERMINAL_FAILED
    assert result.reason_code == "retrieval_method_unavailable"
    assert catalog.active_units("v1") == []
    assert catalog.get_receipt("event-1").status is SyncReceiptStatus.TERMINAL_FAILED


def test_unavailable_index_marks_new_unit_failed_and_keeps_safe_failure_classification(
    tmp_path,
):
    writer = FakeWriter(unavailable=True)
    catalog, _, _, service = _service(tmp_path, writer=writer)
    event = _event(RetrievalUpdateKind.UPSERT, "event-1")

    result = service.apply(event)

    stored = catalog.get_unit(event.ref, event.retrieval_data_version)
    assert stored is not None
    assert stored.index_state.value == "failed"
    assert result.failure_stage == "index_write"
    assert result.safe_error_code == "simulated_encoder_dependency_missing"


def test_trash_restore_and_purge_follow_the_unit_lifecycle(tmp_path):
    catalog, _, writer, service = _service(tmp_path)
    ref = _ref()
    service.apply(_event(RetrievalUpdateKind.UPSERT, "upsert", ref))
    stored = catalog.get_unit(ref, "v1")

    assert service.apply(_event(RetrievalUpdateKind.TRASH, "trash", ref)).status is SyncApplyStatus.APPLIED
    assert catalog.active_units("v1") == []
    assert service.apply(_event(RetrievalUpdateKind.RESTORE, "restore", ref)).status is SyncApplyStatus.APPLIED
    assert len(catalog.active_units("v1")) == 1
    assert len(writer.indexed) == 1  # 哈希与修订版本校验通过后复用保留的索引

    assert service.apply(_event(RetrievalUpdateKind.PURGE, "purge", ref)).status is SyncApplyStatus.APPLIED
    assert catalog.get_unit(ref, "v1") is None
    assert writer.purged == [stored.unit_id]


def test_source_ref_mismatch_is_terminal_not_a_silent_partial_index(tmp_path):
    changed = _ref(revision="r2", content_hash="h2")
    reader = FakeReader(units={_ref(): _source_unit(changed)})
    catalog, _, _, service = _service(tmp_path, reader=reader)

    result = service.apply(_event(RetrievalUpdateKind.UPSERT, "event-1"))

    assert result.status is SyncApplyStatus.TERMINAL_FAILED
    assert result.reason_code == "source_ref_mismatch"
    assert catalog.active_units("v1") == []


def test_outbox_consumer_marks_authority_event_applied_after_sync(tmp_path):
    catalog, _, _, service = _service(tmp_path)
    del catalog
    outbox = SqliteRetrievalOutbox()
    authority = sqlite3.connect(tmp_path / "authority.sqlite")
    try:
        outbox.initialize(authority)
        authority.commit()
        event = _event(RetrievalUpdateKind.UPSERT, "event-1")
        outbox.enqueue(authority, event)
        authority.commit()

        results = RetrievalOutboxConsumer(outbox=outbox, sync_service=service).consume_due(
            authority,
            worker_id="worker-a",
            now=NOW,
        )

        assert [result.status for result in results] == [SyncApplyStatus.APPLIED]
        assert outbox.get_status(authority, "event-1") is OutboxStatus.APPLIED
    finally:
        authority.close()


def test_outbox_consumer_applies_upsert_before_trash_for_the_same_source_unit(tmp_path):
    catalog, _, _, service = _service(tmp_path)
    outbox = SqliteRetrievalOutbox()
    authority = sqlite3.connect(tmp_path / "authority.sqlite")
    try:
        outbox.initialize(authority)
        ref = _ref()
        outbox.enqueue(authority, _event(RetrievalUpdateKind.UPSERT, "upsert", ref))
        outbox.enqueue(authority, _event(RetrievalUpdateKind.TRASH, "trash", ref))
        authority.commit()
        consumer = RetrievalOutboxConsumer(outbox=outbox, sync_service=service)

        first = consumer.consume_due(authority, worker_id="worker-a", now=NOW)
        assert [result.event_id for result in first] == ["upsert"]
        assert len(catalog.active_units("v1")) == 1

        second = consumer.consume_due(authority, worker_id="worker-a", now=NOW)
        assert [result.event_id for result in second] == ["trash"]
        stored = catalog.get_unit(ref, "v1")
        assert stored is not None
        assert stored.unit.retrieval_status is RetrievalStatus.TRASHED
        assert catalog.active_units("v1") == []
    finally:
        authority.close()


def test_outbox_consumer_persists_pointer_only_attempt_audit(tmp_path):
    writer = FakeWriter(unavailable=True)
    _, _, _, service = _service(tmp_path, writer=writer)
    outbox = SqliteRetrievalOutbox()
    authority = sqlite3.connect(tmp_path / "authority.sqlite")
    try:
        outbox.initialize(authority)
        event = _event(RetrievalUpdateKind.UPSERT, "event-1")
        outbox.enqueue(authority, event)
        authority.commit()

        RetrievalOutboxConsumer(outbox=outbox, sync_service=service).consume_due(
            authority,
            worker_id="synchronous-ingest:123:private-instance",
            now=NOW,
            limit=20,
        )

        audits = outbox.list_attempt_audits(authority)
        assert len(audits) == 1
        audit = audits[0]
        assert audit.event_id == "event-1"
        assert audit.attempt == 1
        assert audit.worker_kind == "synchronous-ingest"
        assert audit.worker_instance_hash != "synchronous-ingest:123:private-instance"
        assert len(audit.worker_instance_hash) == 16
        assert len(audit.batch_id) == 16
        assert audit.batch_limit == 20
        assert audit.batch_size == 1
        assert audit.batch_ordinal == 1
        assert audit.outcome == "terminal_failed"
        assert audit.failure_stage == "index_write"
        assert audit.safe_error_code == "simulated_encoder_dependency_missing"
        assert not hasattr(audit, "content")
        assert not hasattr(audit, "source_path")
        assert not hasattr(audit, "traceback")
    finally:
        authority.close()


def test_consumer_claims_only_its_fixed_source_route(tmp_path):
    _, _, _, service = _service(tmp_path)
    outbox = SqliteRetrievalOutbox()
    authority = sqlite3.connect(tmp_path / "authority.sqlite")
    try:
        outbox.initialize(authority)
        outbox.enqueue(authority, _event(RetrievalUpdateKind.UPSERT, "session-event"))
        document_ref = SourceUnitRef(SourceType.DOCUMENT, "document-1", "r1", "h1")
        outbox.enqueue(
            authority,
            _event(RetrievalUpdateKind.UPSERT, "document-event", document_ref),
        )
        authority.commit()

        consumer = RetrievalOutboxConsumer(
            outbox=outbox,
            sync_service=service,
            allowed_source_types=(SourceType.CURRENT_SESSION,),
        )
        results = consumer.consume_due(authority, worker_id="session-worker", now=NOW)

        assert [result.event_id for result in results] == ["session-event"]
        assert outbox.get_status(authority, "session-event") is OutboxStatus.APPLIED
        assert outbox.get_status(authority, "document-event") is OutboxStatus.PENDING
    finally:
        authority.close()


def test_consumer_rejects_a_route_outside_sync_authority(tmp_path):
    _, _, _, service = _service(tmp_path)

    with pytest.raises(ValueError, match="exceed the sync service authority"):
        RetrievalOutboxConsumer(
            outbox=SqliteRetrievalOutbox(),
            sync_service=service,
            allowed_source_types=(SourceType.DOCUMENT,),
        )


def test_sync_rejects_nonowned_source_even_for_nonreading_lifecycle_events(tmp_path):
    catalog, _, _, service = _service(tmp_path)
    document_ref = SourceUnitRef(SourceType.DOCUMENT, "document-1", "r1", "h1")

    result = service.apply(_event(RetrievalUpdateKind.PURGE, "wrong-corpus", document_ref))

    assert result.status is SyncApplyStatus.TERMINAL_FAILED
    assert result.reason_code == "source_type_not_allowed"
    assert catalog.get_receipt("wrong-corpus").status is SyncReceiptStatus.TERMINAL_FAILED
