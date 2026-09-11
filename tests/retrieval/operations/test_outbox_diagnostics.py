from __future__ import annotations

import sqlite3

import pytest

from personagraph.retrieval.contracts import SourceType, SourceUnitRef
from personagraph.retrieval.operations.diagnostics import (
    authority_outbox_snapshot,
    requeue_authority_outbox_terminal_failure,
)
from personagraph.retrieval.lifecycle.outbox import (
    RetrievalUpdateEvent,
    RetrievalUpdateKind,
    SqliteRetrievalOutbox,
)


NOW = "2026-07-22T00:00:00+00:00"


def _terminal_event(db_path) -> None:
    event = RetrievalUpdateEvent(
        event_id="dead-letter-1",
        kind=RetrievalUpdateKind.UPSERT,
        ref=SourceUnitRef(SourceType.LONG_TERM_USER, "memory-1", "r1", "hash-1"),
        retrieval_data_version="retrieval-v1",
        occurred_at=NOW,
    )
    with sqlite3.connect(db_path) as conn:
        outbox = SqliteRetrievalOutbox()
        outbox.initialize(conn)
        assert outbox.enqueue(conn, event)
        outbox.claim_due(conn, worker_id="worker", now=NOW, lease_seconds=30, limit=1)
        outbox.mark_terminal_failure(
            conn,
            event_id=event.event_id,
            worker_id="worker",
            now="2026-07-22T00:00:01+00:00",
            reason_code="derived_index_unavailable",
        )


def test_authority_outbox_snapshot_is_pointer_only_and_read_only(tmp_path):
    db_path = tmp_path / "memory.sqlite"
    _terminal_event(db_path)

    snapshot = authority_outbox_snapshot(authority_db_path=db_path, limit=5)

    assert snapshot["available"] is True
    assert snapshot["manual_requeue_available"] is True
    assert snapshot["attempt_audit_available"] is True
    assert snapshot["attempt_audit_count"] == 0
    assert snapshot["attempt_audit_truncated"] is False
    assert snapshot["status_counts"]["terminal_failed"] == 1
    assert snapshot["attempt_audits"] == []
    assert snapshot["manual_actions"] == []
    dead_letter = snapshot["terminal_failures"][0]
    assert dead_letter["event_id"] == "dead-letter-1"
    assert dead_letter["source_unit_id"] == "memory-1"
    assert dead_letter["reason_code"] == "derived_index_unavailable"
    assert "content" not in dead_letter


def test_authority_snapshot_exposes_safe_attempt_audit_for_evaluation(tmp_path):
    db_path = tmp_path / "documents.sqlite"
    event = RetrievalUpdateEvent(
        event_id="index-failure-1",
        kind=RetrievalUpdateKind.UPSERT,
        ref=SourceUnitRef(SourceType.DOCUMENT, "chunk-1", "r1", "hash-1"),
        retrieval_data_version="retrieval-v1",
        occurred_at=NOW,
    )
    with sqlite3.connect(db_path) as conn:
        outbox = SqliteRetrievalOutbox()
        outbox.initialize(conn)
        outbox.enqueue(conn, event)
        outbox.claim_due(
            conn,
            worker_id="synchronous-ingest:private-instance",
            now=NOW,
            lease_seconds=30,
            limit=20,
        )
        outbox.mark_terminal_failure(
            conn,
            event_id=event.event_id,
            worker_id="synchronous-ingest:private-instance",
            now=NOW,
            reason_code="retrieval_method_unavailable",
        )
        batch_id = outbox.attempt_batch_id(
            worker_id="synchronous-ingest:private-instance",
            occurred_at=NOW,
            event_ids=(event.event_id,),
        )
        outbox.record_attempt_audit(
            conn,
            event_id=event.event_id,
            worker_id="synchronous-ingest:private-instance",
            batch_id=batch_id,
            batch_limit=20,
            batch_size=1,
            batch_ordinal=1,
            outcome="terminal_failed",
            failure_stage="encode",
            safe_error_code="bge_m3_encode_failed:RuntimeError",
            occurred_at=NOW,
        )

    snapshot = authority_outbox_snapshot(authority_db_path=db_path)

    assert snapshot["attempt_audit_available"] is True
    assert snapshot["attempt_audit_count"] == 1
    assert snapshot["attempt_audit_truncated"] is False
    assert snapshot["attempt_audits"] == [
        {
            "event_id": "index-failure-1",
            "kind": "upsert",
            "source_type": "document",
            "source_unit_id": "chunk-1",
            "data_version_id": "retrieval-v1",
            "attempt": 1,
            "worker_kind": "synchronous-ingest",
            "worker_instance_hash": snapshot["attempt_audits"][0][
                "worker_instance_hash"
            ],
            "batch_id": batch_id,
            "batch_limit": 20,
            "batch_size": 1,
            "batch_ordinal": 1,
            "outcome": "terminal_failed",
            "failure_stage": "encode",
            "safe_error_code": "bge_m3_encode_failed:RuntimeError",
            "occurred_at": NOW,
        }
    ]
    assert "private-instance" not in str(snapshot["attempt_audits"])

    truncated = authority_outbox_snapshot(authority_db_path=db_path, limit=1)
    assert truncated["attempt_audit_count"] == 1
    assert truncated["attempt_audit_truncated"] is False


def test_authority_snapshot_marks_attempt_audits_truncated_by_limit(tmp_path):
    db_path = tmp_path / "documents.sqlite"
    worker_id = "document-maintenance:private-instance"
    with sqlite3.connect(db_path) as conn:
        outbox = SqliteRetrievalOutbox()
        outbox.initialize(conn)
        events = tuple(
            RetrievalUpdateEvent(
                event_id=f"event-{index}",
                kind=RetrievalUpdateKind.UPSERT,
                ref=SourceUnitRef(
                    SourceType.DOCUMENT,
                    f"chunk-{index}",
                    "r1",
                    f"hash-{index}",
                ),
                retrieval_data_version="retrieval-v1",
                occurred_at=NOW,
            )
            for index in range(2)
        )
        for event in events:
            outbox.enqueue(conn, event)
        conn.commit()
        claimed = outbox.claim_due(
            conn,
            worker_id=worker_id,
            now=NOW,
            lease_seconds=30,
            limit=20,
        )
        batch_id = outbox.attempt_batch_id(
            worker_id=worker_id,
            occurred_at=NOW,
            event_ids=tuple(event.event_id for event in claimed),
        )
        for ordinal, event in enumerate(claimed, start=1):
            outbox.mark_applied(
                conn,
                event_id=event.event_id,
                worker_id=worker_id,
                now=NOW,
            )
            outbox.record_attempt_audit(
                conn,
                event_id=event.event_id,
                worker_id=worker_id,
                batch_id=batch_id,
                batch_limit=20,
                batch_size=2,
                batch_ordinal=ordinal,
                outcome="applied",
                failure_stage=None,
                safe_error_code=None,
                occurred_at=NOW,
            )
        conn.commit()

    snapshot = authority_outbox_snapshot(authority_db_path=db_path, limit=1)

    assert snapshot["attempt_audit_count"] == 2
    assert snapshot["attempt_audit_truncated"] is True
    assert len(snapshot["attempt_audits"]) == 1


def test_operator_requeue_records_reason_and_returns_event_to_normal_consumer(tmp_path):
    db_path = tmp_path / "sessions.sqlite"
    _terminal_event(db_path)

    result = requeue_authority_outbox_terminal_failure(
        authority_db_path=db_path,
        event_id="dead-letter-1",
        actor="operator:bob",
        reason="retrieval schema repaired",
        occurred_at="2026-07-22T00:10:00+00:00",
    )

    assert result["new_status"] == "pending"
    assert result["previous_reason_code"] == "derived_index_unavailable"
    snapshot = authority_outbox_snapshot(authority_db_path=db_path)
    assert snapshot["status_counts"]["pending"] == 1
    assert snapshot["terminal_failures"] == []
    assert snapshot["manual_actions"] == [
        {
            "action_id": result["action_id"],
            "event_id": "dead-letter-1",
            "action": "manual_requeue",
            "actor": "operator:bob",
            "reason": "retrieval schema repaired",
            "previous_status": "terminal_failed",
            "previous_reason_code": "derived_index_unavailable",
            "occurred_at": "2026-07-22T00:10:00+00:00",
        }
    ]


def test_authority_outbox_snapshot_reports_unavailable_db_without_creating_it(tmp_path):
    missing = tmp_path / "missing.sqlite"

    snapshot = authority_outbox_snapshot(authority_db_path=missing)

    assert snapshot["available"] is False
    assert snapshot["reason_code"] == "authority_database_not_found"
    assert snapshot["manual_requeue_available"] is False
    assert snapshot["attempt_audit_count"] == 0
    assert snapshot["attempt_audit_truncated"] is False
    assert not missing.exists()
    with pytest.raises(FileNotFoundError):
        requeue_authority_outbox_terminal_failure(
            authority_db_path=missing,
            event_id="event-1",
            actor="operator:alice",
            reason="test",
        )
