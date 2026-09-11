from __future__ import annotations

from dataclasses import replace
import sqlite3

import pytest

from personagraph.retrieval.lifecycle import outbox as outbox_module
from personagraph.retrieval.contracts import SourceType, SourceUnitRef
from personagraph.retrieval.lifecycle.outbox import (
    OutboxStatus,
    RetrievalOutboxIdempotencyConflict,
    RetrievalUpdateEvent,
    RetrievalUpdateKind,
    SqliteRetrievalOutbox,
)


NOW = "2026-07-22T00:00:00+00:00"


def _event(event_id: str = "event-1") -> RetrievalUpdateEvent:
    return RetrievalUpdateEvent(
        event_id=event_id,
        kind=RetrievalUpdateKind.UPSERT,
        ref=SourceUnitRef(SourceType.CURRENT_SESSION, "turn-1", "r1", "h1"),
        retrieval_data_version="v1",
        occurred_at=NOW,
    )


def _event_for_source(event_id: str, source_type: SourceType) -> RetrievalUpdateEvent:
    return RetrievalUpdateEvent(
        event_id=event_id,
        kind=RetrievalUpdateKind.UPSERT,
        ref=SourceUnitRef(source_type, f"unit-{event_id}", "r1", f"h-{event_id}"),
        retrieval_data_version="v1",
        occurred_at=NOW,
    )


def _connection(tmp_path):
    conn = sqlite3.connect(tmp_path / "authority.sqlite")
    SqliteRetrievalOutbox().initialize(conn)
    conn.commit()
    return conn


def _mutate_event(
    event: RetrievalUpdateEvent,
    field_name: str,
) -> RetrievalUpdateEvent:
    if field_name == "kind":
        return replace(event, kind=RetrievalUpdateKind.TRASH)
    if field_name == "source_type":
        return replace(event, ref=replace(event.ref, source_type=SourceType.DOCUMENT))
    if field_name == "source_unit_id":
        return replace(event, ref=replace(event.ref, source_unit_id="turn-2"))
    if field_name == "source_revision":
        return replace(event, ref=replace(event.ref, source_revision="r2"))
    if field_name == "indexed_content_hash":
        return replace(event, ref=replace(event.ref, indexed_content_hash="h2"))
    if field_name == "data_version_id":
        return replace(event, retrieval_data_version="v2")
    if field_name == "occurred_at":
        return replace(event, occurred_at="2026-07-22T00:00:01+00:00")
    raise AssertionError(f"unsupported mutation: {field_name}")


def test_outbox_cannot_recreate_the_retired_global_memory_authority(
    tmp_path,
    monkeypatch,
):
    state_dir = tmp_path / "state"
    retired_path = state_dir / "memory" / "memory.sqlite"
    retired_path.parent.mkdir(parents=True)
    monkeypatch.setattr(outbox_module, "STATE_DIR", state_dir)

    with sqlite3.connect(retired_path) as conn:
        with pytest.raises(RuntimeError, match="global memory Outbox authority is retired"):
            SqliteRetrievalOutbox().initialize(conn)
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()

    assert tables == []


def test_outbox_cannot_recreate_retired_tables_in_a_session_authority():
    with sqlite3.connect(":memory:") as conn:
        conn.execute("CREATE TABLE sessions (id TEXT PRIMARY KEY)")

        with pytest.raises(RuntimeError, match="Session History Outbox is retired"):
            SqliteRetrievalOutbox().initialize(conn)

        tables = {
            str(row[0])
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert "retrieval_update_outbox" not in tables
        assert "retrieval_outbox_manual_actions" not in tables


def test_enqueue_shares_authority_transaction_and_rolls_back_with_it(tmp_path):
    outbox = SqliteRetrievalOutbox()
    conn = _connection(tmp_path)
    try:
        conn.execute("BEGIN")
        assert outbox.enqueue(conn, _event())
        conn.rollback()
        assert outbox.get_status(conn, "event-1") is None
    finally:
        conn.close()


def test_exact_event_replay_is_idempotent_and_does_not_consume_sequence(tmp_path):
    outbox = SqliteRetrievalOutbox()
    conn = _connection(tmp_path)
    try:
        first = _event("event-1")
        second = _event("event-2")

        assert outbox.enqueue(conn, first) is True
        assert outbox.enqueue(conn, first) is False
        assert outbox.enqueue(conn, second) is True
        conn.commit()

        assert [
            tuple(row)
            for row in conn.execute(
                "SELECT authority_sequence, event_id FROM retrieval_update_outbox "
                "ORDER BY authority_sequence"
            ).fetchall()
        ] == [(1, "event-1"), (2, "event-2")]
    finally:
        conn.close()


@pytest.mark.parametrize(
    "field_name",
    (
        "kind",
        "source_type",
        "source_unit_id",
        "source_revision",
        "indexed_content_hash",
        "data_version_id",
        "occurred_at",
    ),
)
def test_conflicting_event_replay_fails_closed_and_caller_can_roll_back(
    tmp_path,
    field_name: str,
):
    outbox = SqliteRetrievalOutbox()
    conn = _connection(tmp_path)
    try:
        original = _event()
        assert outbox.enqueue(conn, original) is True
        conn.execute("CREATE TABLE caller_writes (value TEXT PRIMARY KEY)")
        conn.commit()

        conn.execute("BEGIN IMMEDIATE")
        conn.execute("INSERT INTO caller_writes (value) VALUES ('must-roll-back')")
        with pytest.raises(RetrievalOutboxIdempotencyConflict) as raised:
            outbox.enqueue(conn, _mutate_event(original, field_name))
        conn.rollback()

        assert raised.value.event_id == original.event_id
        assert raised.value.conflicting_fields == (field_name,)
        assert conn.execute("SELECT COUNT(*) FROM caller_writes").fetchone()[0] == 0
        assert conn.execute(
            "SELECT COUNT(*) FROM retrieval_update_outbox"
        ).fetchone()[0] == 1
    finally:
        conn.close()


def test_outbox_claims_once_and_requires_owner_lease_to_finish(tmp_path):
    outbox = SqliteRetrievalOutbox()
    conn = _connection(tmp_path)
    try:
        assert outbox.enqueue(conn, _event())
        conn.commit()
        claimed = outbox.claim_due(conn, worker_id="worker-a", now=NOW, lease_seconds=30, limit=10)
        assert [event.event_id for event in claimed] == ["event-1"]
        assert outbox.get_status(conn, "event-1") is OutboxStatus.PROCESSING

        with pytest.raises(RuntimeError, match="worker lease"):
            outbox.mark_applied(conn, event_id="event-1", worker_id="worker-b", now=NOW)

        outbox.mark_applied(conn, event_id="event-1", worker_id="worker-a", now=NOW)
        conn.commit()
        assert outbox.get_status(conn, "event-1") is OutboxStatus.APPLIED
        assert outbox.claim_due(conn, worker_id="worker-a", now=NOW, lease_seconds=30, limit=10) == []
    finally:
        conn.close()


def test_outbox_uses_authority_sequence_when_audit_timestamps_move_backwards(tmp_path):
    """Lifecycle causality must not depend on event IDs or caller audit clocks."""

    outbox = SqliteRetrievalOutbox()
    conn = _connection(tmp_path)
    try:
        ref = SourceUnitRef(SourceType.CURRENT_SESSION, "turn-1", "r1", "h1")
        events = (
            RetrievalUpdateEvent(
                "z-upsert",
                RetrievalUpdateKind.UPSERT,
                ref,
                "v1",
                "2026-07-22T00:00:02+00:00",
            ),
            RetrievalUpdateEvent(
                "a-trash",
                RetrievalUpdateKind.TRASH,
                ref,
                "v1",
                "2026-07-22T00:00:01+00:00",
            ),
            RetrievalUpdateEvent(
                "b-restore",
                RetrievalUpdateKind.RESTORE,
                ref,
                "v1",
                "2026-07-22T00:00:00+00:00",
            ),
        )
        for event in events:
            assert outbox.enqueue(conn, event)
        conn.commit()

        assert [
            tuple(row)
            for row in conn.execute(
                "SELECT authority_sequence, event_id FROM retrieval_update_outbox "
                "ORDER BY authority_sequence"
            ).fetchall()
        ] == [(1, "z-upsert"), (2, "a-trash"), (3, "b-restore")]
        claimed_ids: list[str] = []
        for expected in events:
            claimed = outbox.claim_due(
                conn,
                worker_id="worker-a",
                now="2026-07-22T00:01:00+00:00",
                lease_seconds=30,
                limit=10,
            )
            assert [event.event_id for event in claimed] == [expected.event_id]
            claimed_ids.append(claimed[0].event_id)
            outbox.mark_applied(
                conn,
                event_id=expected.event_id,
                worker_id="worker-a",
                now="2026-07-22T00:01:00+00:00",
            )
            conn.commit()

        assert claimed_ids == [event.event_id for event in events]
    finally:
        conn.close()


def test_claim_limit_skips_blocked_successor_and_fills_from_other_streams(tmp_path):
    outbox = SqliteRetrievalOutbox()
    conn = _connection(tmp_path)
    try:
        ref_a = SourceUnitRef(SourceType.CURRENT_SESSION, "stream-a", "r1", "h-a")
        ref_b = SourceUnitRef(SourceType.CURRENT_SESSION, "stream-b", "r1", "h-b")
        ref_c = SourceUnitRef(SourceType.CURRENT_SESSION, "stream-c", "r1", "h-c")
        events = (
            RetrievalUpdateEvent("a-1", RetrievalUpdateKind.UPSERT, ref_a, "v1", NOW),
            RetrievalUpdateEvent("a-2", RetrievalUpdateKind.TRASH, ref_a, "v1", NOW),
            RetrievalUpdateEvent("b-1", RetrievalUpdateKind.UPSERT, ref_b, "v1", NOW),
            RetrievalUpdateEvent("c-1", RetrievalUpdateKind.UPSERT, ref_c, "v1", NOW),
        )
        for event in events:
            assert outbox.enqueue(conn, event)
        conn.commit()

        first_batch = outbox.claim_due(
            conn,
            worker_id="worker-a",
            now=NOW,
            lease_seconds=30,
            limit=2,
        )
        assert [event.event_id for event in first_batch] == ["a-1", "b-1"]
        for event in first_batch:
            outbox.mark_applied(
                conn,
                event_id=event.event_id,
                worker_id="worker-a",
                now=NOW,
            )
        conn.commit()

        second_batch = outbox.claim_due(
            conn,
            worker_id="worker-a",
            now=NOW,
            lease_seconds=30,
            limit=2,
        )
        assert [event.event_id for event in second_batch] == ["a-2", "c-1"]
    finally:
        conn.close()


def test_two_workers_cannot_claim_successor_while_predecessor_is_processing(tmp_path):
    outbox = SqliteRetrievalOutbox()
    first_conn = _connection(tmp_path)
    second_conn = sqlite3.connect(tmp_path / "authority.sqlite")
    outbox.initialize(second_conn)
    try:
        ref_a = SourceUnitRef(SourceType.CURRENT_SESSION, "stream-a", "r1", "h-a")
        ref_b = SourceUnitRef(SourceType.CURRENT_SESSION, "stream-b", "r1", "h-b")
        for event in (
            RetrievalUpdateEvent("a-1", RetrievalUpdateKind.UPSERT, ref_a, "v1", NOW),
            RetrievalUpdateEvent("a-2", RetrievalUpdateKind.TRASH, ref_a, "v1", NOW),
            RetrievalUpdateEvent("b-1", RetrievalUpdateKind.UPSERT, ref_b, "v1", NOW),
        ):
            assert outbox.enqueue(first_conn, event)
        first_conn.commit()

        first_claim = outbox.claim_due(
            first_conn,
            worker_id="worker-a",
            now=NOW,
            lease_seconds=30,
            limit=1,
        )
        assert [event.event_id for event in first_claim] == ["a-1"]
        second_claim = outbox.claim_due(
            second_conn,
            worker_id="worker-b",
            now=NOW,
            lease_seconds=30,
            limit=10,
        )
        assert [event.event_id for event in second_claim] == ["b-1"]
        outbox.mark_applied(
            second_conn,
            event_id="b-1",
            worker_id="worker-b",
            now=NOW,
        )
        second_conn.commit()

        outbox.mark_applied(
            first_conn,
            event_id="a-1",
            worker_id="worker-a",
            now=NOW,
        )
        first_conn.commit()
        successor = outbox.claim_due(
            second_conn,
            worker_id="worker-b",
            now=NOW,
            lease_seconds=30,
            limit=10,
        )
        assert [event.event_id for event in successor] == ["a-2"]
    finally:
        first_conn.close()
        second_conn.close()


def test_same_source_unit_in_different_data_versions_does_not_cross_block(tmp_path):
    outbox = SqliteRetrievalOutbox()
    conn = _connection(tmp_path)
    try:
        ref = SourceUnitRef(SourceType.CURRENT_SESSION, "stream-a", "r1", "h-a")
        assert outbox.enqueue(
            conn,
            RetrievalUpdateEvent("v1-event", RetrievalUpdateKind.UPSERT, ref, "v1", NOW),
        )
        assert outbox.enqueue(
            conn,
            RetrievalUpdateEvent("v2-event", RetrievalUpdateKind.UPSERT, ref, "v2", NOW),
        )
        conn.commit()

        claimed = outbox.claim_due(
            conn,
            worker_id="worker-a",
            now=NOW,
            lease_seconds=30,
            limit=10,
        )
        assert [event.event_id for event in claimed] == ["v1-event", "v2-event"]
    finally:
        conn.close()


def test_claim_due_filters_source_types_in_sql_and_leaves_other_events_pending(tmp_path):
    outbox = SqliteRetrievalOutbox()
    conn = _connection(tmp_path)
    try:
        events = (
            _event_for_source("document", SourceType.DOCUMENT),
            _event_for_source("session", SourceType.CURRENT_SESSION),
            _event_for_source("user-memory", SourceType.LONG_TERM_USER),
            _event_for_source("task-memory", SourceType.LONG_TERM_TASK),
        )
        for event in events:
            assert outbox.enqueue(conn, event)
        conn.commit()

        claimed = outbox.claim_due(
            conn,
            worker_id="history-memory-worker",
            now=NOW,
            lease_seconds=30,
            limit=10,
            allowed_source_types=(SourceType.LONG_TERM_USER, SourceType.LONG_TERM_TASK),
        )

        assert [event.event_id for event in claimed] == ["user-memory", "task-memory"]
        assert outbox.get_status(conn, "user-memory") is OutboxStatus.PROCESSING
        assert outbox.get_status(conn, "task-memory") is OutboxStatus.PROCESSING
        assert outbox.get_status(conn, "document") is OutboxStatus.PENDING
        assert outbox.get_status(conn, "session") is OutboxStatus.PENDING
    finally:
        conn.close()


def test_claim_due_rejects_an_empty_source_route(tmp_path):
    outbox = SqliteRetrievalOutbox()
    conn = _connection(tmp_path)
    try:
        with pytest.raises(ValueError, match="must not be empty"):
            outbox.claim_due(
                conn,
                worker_id="worker-a",
                now=NOW,
                lease_seconds=30,
                limit=10,
                allowed_source_types=(),
            )
    finally:
        conn.close()


def test_retryable_failure_is_not_claimed_before_its_backoff_expires(tmp_path):
    outbox = SqliteRetrievalOutbox()
    conn = _connection(tmp_path)
    try:
        outbox.enqueue(conn, _event())
        outbox.enqueue(conn, _event("event-2"))
        conn.commit()
        assert [item.event_id for item in outbox.claim_due(
            conn,
            worker_id="worker-a",
            now=NOW,
            lease_seconds=30,
            limit=10,
        )] == ["event-1"]
        outbox.mark_retryable_failure(
            conn,
            event_id="event-1",
            worker_id="worker-a",
            now=NOW,
            retry_after_seconds=60,
            reason_code="temporary_database_busy",
        )
        conn.commit()
        assert outbox.get_status(conn, "event-1") is OutboxStatus.RETRYABLE_FAILED
        assert outbox.claim_due(conn, worker_id="worker-b", now="2026-07-22T00:00:30+00:00", lease_seconds=30, limit=10) == []
        assert [item.event_id for item in outbox.claim_due(
            conn,
            worker_id="worker-b",
            now="2026-07-22T00:01:01+00:00",
            lease_seconds=30,
            limit=10,
        )] == ["event-1"]
        outbox.mark_applied(
            conn,
            event_id="event-1",
            worker_id="worker-b",
            now="2026-07-22T00:01:01+00:00",
        )
        conn.commit()
        assert [item.event_id for item in outbox.claim_due(
            conn,
            worker_id="worker-b",
            now="2026-07-22T00:01:01+00:00",
            lease_seconds=30,
            limit=10,
        )] == ["event-2"]
    finally:
        conn.close()


def test_terminal_failure_can_be_audited_and_explicitly_requeued(tmp_path):
    outbox = SqliteRetrievalOutbox()
    conn = _connection(tmp_path)
    try:
        assert outbox.enqueue(conn, _event())
        conn.commit()
        assert [item.event_id for item in outbox.claim_due(
            conn,
            worker_id="worker-a",
            now=NOW,
            lease_seconds=30,
            limit=10,
        )] == ["event-1"]
        outbox.mark_terminal_failure(
            conn,
            event_id="event-1",
            worker_id="worker-a",
            now="2026-07-22T00:00:05+00:00",
            reason_code="unsupported_source_schema",
        )
        conn.commit()

        assert outbox.status_counts(conn)[OutboxStatus.TERMINAL_FAILED] == 1
        failures = outbox.list_terminal_failures(conn)
        assert len(failures) == 1
        assert failures[0].event.event_id == "event-1"
        assert failures[0].attempts == 1
        assert failures[0].reason_code == "unsupported_source_schema"

        action = outbox.requeue_terminal_failure(
            conn,
            event_id="event-1",
            actor="operator:alice",
            reason="schema migration has completed",
            now="2026-07-22T00:05:00+00:00",
        )

        assert outbox.get_status(conn, "event-1") is OutboxStatus.PENDING
        assert outbox.status_counts(conn)[OutboxStatus.PENDING] == 1
        assert outbox.list_terminal_failures(conn) == ()
        assert action.previous_status is OutboxStatus.TERMINAL_FAILED
        assert action.previous_reason_code == "unsupported_source_schema"
        assert outbox.list_manual_actions(conn) == (action,)
        attempts = conn.execute(
            "SELECT attempts, reason_code, lease_token, lease_until FROM retrieval_update_outbox WHERE event_id='event-1'"
        ).fetchone()
        assert tuple(attempts) == (1, None, None, None)
    finally:
        conn.close()


def test_terminal_predecessor_blocks_successor_until_requeued_and_applied(tmp_path):
    outbox = SqliteRetrievalOutbox()
    conn = _connection(tmp_path)
    try:
        assert outbox.enqueue(conn, _event("event-1"))
        assert outbox.enqueue(conn, _event("event-2"))
        conn.commit()
        assert [item.event_id for item in outbox.claim_due(
            conn,
            worker_id="worker-a",
            now=NOW,
            lease_seconds=30,
            limit=10,
        )] == ["event-1"]
        outbox.mark_terminal_failure(
            conn,
            event_id="event-1",
            worker_id="worker-a",
            now=NOW,
            reason_code="unsupported_source_schema",
        )
        conn.commit()

        assert outbox.claim_due(
            conn,
            worker_id="worker-b",
            now=NOW,
            lease_seconds=30,
            limit=10,
        ) == []
        outbox.requeue_terminal_failure(
            conn,
            event_id="event-1",
            actor="operator:alice",
            reason="schema migration has completed",
            now=NOW,
        )
        assert [item.event_id for item in outbox.claim_due(
            conn,
            worker_id="worker-b",
            now=NOW,
            lease_seconds=30,
            limit=10,
        )] == ["event-1"]
        outbox.mark_applied(
            conn,
            event_id="event-1",
            worker_id="worker-b",
            now=NOW,
        )
        conn.commit()
        assert [item.event_id for item in outbox.claim_due(
            conn,
            worker_id="worker-b",
            now=NOW,
            lease_seconds=30,
            limit=10,
        )] == ["event-2"]
    finally:
        conn.close()


def test_purge_supersedes_a_terminal_predecessor_for_the_same_source_unit(tmp_path):
    outbox = SqliteRetrievalOutbox()
    conn = _connection(tmp_path)
    try:
        ref = _event("event-1").ref
        assert outbox.enqueue(conn, _event("event-1"))
        assert outbox.enqueue(
            conn,
            RetrievalUpdateEvent(
                "event-purge",
                RetrievalUpdateKind.PURGE,
                ref,
                "v1",
                NOW,
            ),
        )
        conn.commit()

        first = outbox.claim_due(
            conn,
            worker_id="worker-a",
            now=NOW,
            lease_seconds=30,
            limit=10,
        )
        assert [item.event_id for item in first] == ["event-1"]
        outbox.mark_terminal_failure(
            conn,
            event_id="event-1",
            worker_id="worker-a",
            now=NOW,
            reason_code="source_unit_missing",
        )
        conn.commit()

        cleanup = outbox.claim_due(
            conn,
            worker_id="worker-b",
            now=NOW,
            lease_seconds=30,
            limit=10,
        )
        assert [item.event_id for item in cleanup] == ["event-purge"]
    finally:
        conn.close()


def test_manual_requeue_rejects_nonterminal_or_missing_events(tmp_path):
    outbox = SqliteRetrievalOutbox()
    conn = _connection(tmp_path)
    try:
        outbox.enqueue(conn, _event())
        conn.commit()
        with pytest.raises(ValueError, match="only terminal_failed"):
            outbox.requeue_terminal_failure(
                conn,
                event_id="event-1",
                actor="operator:alice",
                reason="should not apply",
                now=NOW,
            )
        with pytest.raises(KeyError, match="outbox event not found"):
            outbox.requeue_terminal_failure(
                conn,
                event_id="missing",
                actor="operator:alice",
                reason="should not apply",
                now=NOW,
            )
        assert outbox.list_manual_actions(conn) == ()
    finally:
        conn.close()
