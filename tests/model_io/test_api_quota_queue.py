from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
import sqlite3
from threading import Barrier

import pytest

from personagraph.model_io.api_quota_queue import (
    MINUTE_SECONDS,
    WEEK_SECONDS,
    ModelApiQuotaQueue,
    QuotaDispatchBlocked,
    QuotaLeasePhaseError,
    QuotaLeaseStale,
    QuotaLimits,
    QuotaQueueError,
    QuotaReservationImpossible,
    QuotaScopeConfigurationConflict,
    QuotaTicket,
    QuotaTicketConflict,
    QuotaTicketNotAcquirable,
    QuotaWaitReason,
    derive_quota_scope_hash,
)


@dataclass
class _Clock:
    value: float = 1_000.0
    sleeps: list[float] = field(default_factory=list)

    def __call__(self) -> float:
        return self.value

    def sleep(self, seconds: float) -> None:
        assert seconds >= 0
        self.sleeps.append(seconds)
        self.value += seconds

    def advance(self, seconds: float) -> None:
        self.value += seconds


def _scope(*, credential: str = "secret-key", group: str | None = None) -> str:
    return derive_quota_scope_hash(
        base_url="https://models.example.test/api/v1/",
        credential=credential,
        quota_group=group,
    )


def _queue(
    tmp_path: Path,
    clock: _Clock,
    *,
    rpm: int | None = 100,
    tpm: int | None = 100_000,
    weekly: int | None = 1_000_000,
    in_flight: int | None = 2,
    queued_ttl: float = 15,
) -> tuple[ModelApiQuotaQueue, str]:
    queue = ModelApiQuotaQueue(
        tmp_path / "quota.sqlite3",
        clock=clock,
        sleep=clock.sleep,
        queued_ticket_ttl_seconds=queued_ttl,
    )
    scope = _scope()
    queue.register_scope(scope, QuotaLimits(rpm, tpm, weekly, in_flight))
    return queue, scope


def _enqueue(
    queue: ModelApiQuotaQueue,
    scope: str,
    clock: _Clock,
    logical_call_id: str,
    *,
    physical_ordinal: int | None = 1,
    tokens: int = 10,
    eligible_in: float = 0,
    deadline_in: float = 1_000,
    ticket_id: str | None = None,
    owner: str = "worker",
) -> QuotaTicket:
    return queue.enqueue(
        scope_hash=scope,
        logical_call_id=logical_call_id,
        physical_ordinal=physical_ordinal,
        token_reservation=tokens,
        queue_owner=owner,
        eligible_at=clock() + eligible_in,
        deadline=clock() + deadline_in,
        ticket_id=ticket_id,
    )


def test_initialization_opens_one_sqlite_connection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = ModelApiQuotaQueue._connect
    opened: list[sqlite3.Connection] = []

    def counted(queue: ModelApiQuotaQueue) -> sqlite3.Connection:
        connection = original(queue)
        opened.append(connection)
        return connection

    monkeypatch.setattr(ModelApiQuotaQueue, "_connect", counted)
    ModelApiQuotaQueue(tmp_path / "one-initialize-connection.sqlite3")

    assert len(opened) == 1


def _acquire(
    queue: ModelApiQuotaQueue,
    ticket: QuotaTicket,
    *,
    owner: str | None = None,
    lease_seconds: float = 30,
):
    result = queue.try_acquire_ticket(
        ticket.ticket_id,
        lease_owner=owner or ticket.queue_owner,
        lease_seconds=lease_seconds,
    )
    assert result.lease is not None, result
    assert result.lease.ticket.ticket_id == ticket.ticket_id
    return result.lease


def _dispatch_and_settle(
    queue: ModelApiQuotaQueue,
    lease,
    *,
    outcome: str = "succeeded",
    actual_tokens: int | None = 1,
):
    queue.mark_dispatched(lease)
    return queue.settle(
        lease,
        outcome=outcome,
        actual_tokens=actual_tokens,
    )


def test_scope_hash_can_share_explicit_group_and_database_never_stores_key(
    tmp_path: Path,
) -> None:
    clock = _Clock()
    key = "sk-plain-text-must-never-be-persisted"
    grouped_a = _scope(credential=key, group="sjtu-account")
    grouped_b = _scope(credential="another-key", group="sjtu-account")
    ungrouped_a = _scope(credential=key)
    ungrouped_b = _scope(credential="another-key")

    assert grouped_a == grouped_b
    assert ungrouped_a != ungrouped_b
    queue = ModelApiQuotaQueue(tmp_path / "quota.sqlite3", clock=clock)
    queue.register_scope(grouped_a, QuotaLimits(10, 100_000, 1_000_000_000, 2))

    with sqlite3.connect(queue.database_path) as conn:
        columns = {
            row[1]
            for row in conn.execute("PRAGMA table_info(model_api_quota_scopes)")
        }
        values = conn.execute("SELECT * FROM model_api_quota_scopes").fetchone()
    assert "credential" not in columns
    assert "api_key" not in columns
    assert values is not None
    persisted_files = b"".join(
        path.read_bytes()
        for path in tmp_path.glob("quota.sqlite3*")
        if path.is_file()
    )
    assert key.encode("utf-8") not in persisted_files


def test_ticket_id_is_only_idempotency_identity(tmp_path: Path) -> None:
    clock = _Clock()
    queue, scope = _queue(tmp_path, clock)
    first = _enqueue(
        queue,
        scope,
        clock,
        "call-a",
        physical_ordinal=None,
        ticket_id="stable-admission-id",
    )
    replay = _enqueue(
        queue,
        scope,
        clock,
        "call-a",
        physical_ordinal=None,
        ticket_id="stable-admission-id",
    )
    assert replay == first
    with pytest.raises(QuotaTicketConflict):
        _enqueue(
            queue,
            scope,
            clock,
            "call-a",
            physical_ordinal=None,
            tokens=11,
            ticket_id="stable-admission-id",
        )

    same_runtime_ordinal = _enqueue(
        queue,
        scope,
        clock,
        "call-a",
        physical_ordinal=None,
    )
    assert same_runtime_ordinal.ticket_id != first.ticket_id


def test_register_verifies_but_configure_changes_policy_with_history(
    tmp_path: Path,
) -> None:
    clock = _Clock()
    queue, scope = _queue(tmp_path, clock, rpm=100, in_flight=None)
    with pytest.raises(QuotaScopeConfigurationConflict):
        queue.register_scope(scope, QuotaLimits(2, 100_000, 1_000_000, None))

    first = _enqueue(queue, scope, clock, "call-a")
    second = _enqueue(queue, scope, clock, "call-b")
    _dispatch_and_settle(queue, _acquire(queue, first))
    _dispatch_and_settle(queue, _acquire(queue, second))
    third = _enqueue(queue, scope, clock, "call-c")

    queue.configure_scope(scope, QuotaLimits(2, 100_000, 1_000_000, None))
    blocked = queue.try_acquire_ticket(
        third.ticket_id, lease_owner="worker", lease_seconds=30
    )
    assert blocked.blocked_by == (QuotaWaitReason.REQUESTS_PER_MINUTE,)
    queue.configure_scope(scope, QuotaLimits(3, 100_000, 1_000_000, None))
    third_lease = _acquire(queue, third)
    queue.abandon_before_dispatch(third_lease, disposition="cancel")

    oversized_after_edit = _enqueue(
        queue, scope, clock, "call-d", tokens=20
    )
    queue.configure_scope(scope, QuotaLimits(3, 10, 1_000_000, None))
    terminal = queue.try_acquire_ticket(
        oversized_after_edit.ticket_id,
        lease_owner="worker",
        lease_seconds=30,
    )
    assert terminal.blocked_by == (QuotaWaitReason.TICKET_TERMINAL,)


def test_caller_only_acquires_own_ticket_after_fifo_predecessor(
    tmp_path: Path,
) -> None:
    clock = _Clock()
    queue, scope = _queue(tmp_path, clock, in_flight=1)
    first = _enqueue(queue, scope, clock, "call-a", owner="worker-a")
    second = _enqueue(queue, scope, clock, "call-b", owner="worker-b")

    blocked = queue.try_acquire_ticket(
        second.ticket_id, lease_owner="worker-b", lease_seconds=30
    )
    assert blocked.lease is None
    assert blocked.blocked_by == (QuotaWaitReason.FIFO_PREDECESSOR,)
    with pytest.raises(QuotaTicketNotAcquirable):
        queue.try_acquire_ticket(
            first.ticket_id, lease_owner="worker-b", lease_seconds=30
        )
    first_lease = _acquire(queue, first)
    _dispatch_and_settle(queue, first_lease)
    assert _acquire(queue, second).ticket.ticket_id == second.ticket_id


def test_failed_retry_rejoins_after_waiting_first_attempt(tmp_path: Path) -> None:
    clock = _Clock()
    queue, scope = _queue(tmp_path, clock, in_flight=1)
    first = _enqueue(queue, scope, clock, "call-a")
    waiting = _enqueue(queue, scope, clock, "call-b")
    _dispatch_and_settle(
        queue,
        _acquire(queue, first),
        outcome="failed",
        actual_tokens=3,
    )
    retry = _enqueue(queue, scope, clock, "call-a", physical_ordinal=2)

    waiting_lease = _acquire(queue, waiting)
    blocked_retry = queue.try_acquire_ticket(
        retry.ticket_id, lease_owner="worker", lease_seconds=30
    )
    assert QuotaWaitReason.MAX_IN_FLIGHT in blocked_retry.blocked_by
    _dispatch_and_settle(queue, waiting_lease)
    assert _acquire(queue, retry).ticket.physical_ordinal == 2


def test_delayed_retry_does_not_block_ready_first_attempt(tmp_path: Path) -> None:
    clock = _Clock()
    queue, scope = _queue(tmp_path, clock, in_flight=1)
    retry = _enqueue(
        queue,
        scope,
        clock,
        "retrying-call",
        physical_ordinal=2,
        eligible_in=20,
    )
    fresh = _enqueue(queue, scope, clock, "fresh-call")

    _dispatch_and_settle(queue, _acquire(queue, fresh))
    blocked = queue.try_acquire_ticket(
        retry.ticket_id, lease_owner="worker", lease_seconds=30
    )
    assert blocked.blocked_by == (QuotaWaitReason.NOT_YET_ELIGIBLE,)
    assert blocked.retry_at == clock() + 20


def test_two_phase_admission_and_pre_dispatch_abandon_remove_charge(
    tmp_path: Path,
) -> None:
    clock = _Clock()
    queue, scope = _queue(tmp_path, clock, rpm=1, tpm=10, in_flight=1)
    first = _enqueue(queue, scope, clock, "call-a", tokens=10)
    lease = _acquire(queue, first)

    snapshot = queue.usage_snapshot(scope)
    assert snapshot.requests_last_minute == 0
    assert snapshot.tokens_last_minute == 0
    assert snapshot.admitted_not_dispatched == 1
    with pytest.raises(QuotaLeasePhaseError):
        queue.settle(lease, outcome="succeeded", actual_tokens=1)

    requeued = queue.abandon_before_dispatch(lease, disposition="requeue")
    assert requeued.status == "queued"
    snapshot = queue.usage_snapshot(scope)
    assert snapshot.requests_last_minute == 0
    assert snapshot.tokens_last_minute == 0
    assert snapshot.admitted_not_dispatched == 0

    reacquired = _acquire(queue, requeued)
    queue.mark_dispatched(reacquired)
    with pytest.raises(QuotaLeasePhaseError):
        queue.abandon_before_dispatch(reacquired, disposition="cancel")


def test_rpm_wait_heartbeats_with_injected_sleep(tmp_path: Path) -> None:
    clock = _Clock()
    queue, scope = _queue(tmp_path, clock, rpm=1, in_flight=1, queued_ttl=15)
    first = _enqueue(queue, scope, clock, "call-a")
    _dispatch_and_settle(queue, _acquire(queue, first), actual_tokens=4)
    second = _enqueue(queue, scope, clock, "call-b")

    blocked = queue.try_acquire_ticket(
        second.ticket_id, lease_owner="worker", lease_seconds=30
    )
    assert blocked.blocked_by == (QuotaWaitReason.REQUESTS_PER_MINUTE,)
    assert blocked.retry_at == 1_000 + MINUTE_SECONDS

    lease = queue.wait_for_ticket(
        second.ticket_id,
        lease_owner="worker",
        lease_seconds=30,
    )
    assert lease is not None
    assert lease.ticket.ticket_id == second.ticket_id
    assert sum(clock.sleeps) == MINUTE_SECONDS
    assert max(clock.sleeps) <= 5
    assert queue.usage_snapshot(scope).requests_last_minute == 0
    assert queue.usage_snapshot(scope).admitted_not_dispatched == 1


def test_tpm_actual_correction_and_weekly_rolling_window(tmp_path: Path) -> None:
    clock = _Clock()
    queue, scope = _queue(
        tmp_path,
        clock,
        tpm=10,
        weekly=10,
        in_flight=1,
        queued_ttl=WEEK_SECONDS + 200,
    )
    first = _enqueue(queue, scope, clock, "call-a", tokens=8)
    _dispatch_and_settle(queue, _acquire(queue, first), actual_tokens=3)
    second = _enqueue(queue, scope, clock, "call-b", tokens=7)
    _dispatch_and_settle(queue, _acquire(queue, second), actual_tokens=7)
    assert queue.usage_snapshot(scope).tokens_last_minute == 10

    clock.advance(MINUTE_SECONDS + 1)
    third = _enqueue(
        queue,
        scope,
        clock,
        "call-c",
        tokens=1,
        deadline_in=WEEK_SECONDS + 100,
    )
    blocked = queue.try_acquire_ticket(
        third.ticket_id, lease_owner="worker", lease_seconds=30
    )
    assert blocked.blocked_by == (QuotaWaitReason.TOKENS_PER_WEEK,)
    assert queue.wait_for_ticket(
        third.ticket_id,
        lease_owner="worker",
        lease_seconds=30,
        stop_at=clock() + 5,
    ) is None
    assert clock.sleeps[-1] == 5

    clock.advance(WEEK_SECONDS - MINUTE_SECONDS - 6)
    assert _acquire(queue, third).ticket.ticket_id == third.ticket_id


def test_one_ticket_larger_than_limit_fails_before_queueing(tmp_path: Path) -> None:
    clock = _Clock()
    queue, scope = _queue(tmp_path, clock, tpm=10, weekly=100)
    with pytest.raises(QuotaReservationImpossible):
        _enqueue(queue, scope, clock, "too-large", tokens=11)


def test_pre_dispatch_lease_expiry_requeues_at_tail_without_usage(
    tmp_path: Path,
) -> None:
    clock = _Clock()
    queue, scope = _queue(tmp_path, clock, rpm=100, in_flight=1)
    first = _enqueue(queue, scope, clock, "call-a", owner="crashed")
    second = _enqueue(queue, scope, clock, "call-b", owner="healthy")
    stale = _acquire(queue, first, lease_seconds=5)
    clock.advance(5)

    with pytest.raises(QuotaLeaseStale):
        queue.mark_dispatched(stale)
    second_lease = _acquire(queue, second)
    _dispatch_and_settle(queue, second_lease)
    recovered = _acquire(queue, first)
    assert recovered.ticket.ticket_id == first.ticket_id
    snapshot = queue.usage_snapshot(scope)
    assert snapshot.requests_last_minute == 1
    assert snapshot.admitted_not_dispatched == 1


def test_post_dispatch_expiry_requires_reconciliation_and_never_releases(
    tmp_path: Path,
) -> None:
    clock = _Clock()
    queue, scope = _queue(tmp_path, clock, rpm=100, in_flight=1)
    first = _enqueue(queue, scope, clock, "call-a", owner="crashed")
    second = _enqueue(queue, scope, clock, "call-b", owner="healthy")
    uncertain = _acquire(queue, first, lease_seconds=5)
    queue.mark_dispatched(uncertain)
    clock.advance(5)

    with pytest.raises(QuotaLeaseStale):
        queue.settle(uncertain, outcome="succeeded", actual_tokens=1)
    stored = queue.get_ticket(first.ticket_id)
    assert stored is not None
    assert stored.status == "reconciliation_required"
    assert _acquire(queue, second).ticket.ticket_id == second.ticket_id
    terminal = queue.try_acquire_ticket(
        first.ticket_id, lease_owner="crashed", lease_seconds=30
    )
    assert terminal.blocked_by == (QuotaWaitReason.TICKET_TERMINAL,)
    reconciled = queue.reconcile_uncertain_dispatch(
        first.ticket_id,
        queue_owner="crashed",
        outcome="failed",
        actual_tokens=2,
    )
    assert reconciled.status == "settled"
    assert queue.usage_snapshot(scope).tokens_last_minute == 2


def test_queued_heartbeat_evicts_crashed_predecessor_but_keeps_waiter_alive(
    tmp_path: Path,
) -> None:
    clock = _Clock()
    queue, scope = _queue(tmp_path, clock, in_flight=1, queued_ttl=6)
    crashed = _enqueue(queue, scope, clock, "call-a", owner="crashed")
    healthy = _enqueue(queue, scope, clock, "call-b", owner="healthy")

    blocked = queue.try_acquire_ticket(
        healthy.ticket_id, lease_owner="healthy", lease_seconds=30
    )
    assert blocked.blocked_by == (QuotaWaitReason.FIFO_PREDECESSOR,)
    lease = queue.wait_for_ticket(
        healthy.ticket_id,
        lease_owner="healthy",
        lease_seconds=30,
        stop_at=clock() + 10,
    )
    assert lease is not None
    assert lease.ticket.ticket_id == healthy.ticket_id
    assert queue.get_ticket(crashed.ticket_id).status == "expired"  # type: ignore[union-attr]
    assert max(clock.sleeps) <= 2


def test_mark_dispatched_atomically_extends_then_lease_can_renew(
    tmp_path: Path,
) -> None:
    clock = _Clock()
    queue, scope = _queue(tmp_path, clock, in_flight=1, queued_ttl=30)
    ticket = _enqueue(queue, scope, clock, "long-request")
    lease = _acquire(queue, ticket, lease_seconds=5)
    clock.advance(4)
    dispatched = queue.mark_dispatched(lease, lease_seconds=20)
    assert dispatched.lease_until == clock() + 20
    clock.advance(15)

    renewed = queue.renew_lease(dispatched, lease_seconds=10)
    assert renewed.lease_until == clock() + 10
    queue.settle(renewed, outcome="succeeded", actual_tokens=1)


def test_mark_dispatched_rechecks_cooldown_added_after_admission(
    tmp_path: Path,
) -> None:
    clock = _Clock()
    queue, scope = _queue(tmp_path, clock, in_flight=1)
    ticket = _enqueue(queue, scope, clock, "cooled-after-admission")
    lease = _acquire(queue, ticket)
    queue.apply_rate_limit_cooldown(scope, retry_after_seconds=20)

    with pytest.raises(QuotaDispatchBlocked) as captured:
        queue.mark_dispatched(lease)

    assert getattr(captured.value, "blocked_by") == (
        QuotaWaitReason.SCOPE_COOLDOWN,
    )
    assert getattr(captured.value, "retry_at") == clock() + 20
    assert queue.usage_snapshot(scope).requests_last_minute == 0
    assert queue.get_ticket(ticket.ticket_id).status == "leased"  # type: ignore[union-attr]
    requeued = queue.abandon_before_dispatch(lease, disposition="requeue")
    assert requeued.status == "queued"


def test_mark_dispatched_rechecks_tightened_policy_without_double_counting_self(
    tmp_path: Path,
) -> None:
    clock = _Clock()
    queue, scope = _queue(tmp_path, clock, rpm=100, tpm=100, weekly=100)
    target = _enqueue(queue, scope, clock, "target", tokens=10, owner="target")
    other = _enqueue(queue, scope, clock, "other", tokens=10, owner="other")
    target_lease = _acquire(queue, target, owner="target")
    other_lease = _acquire(queue, other, owner="other")
    _dispatch_and_settle(queue, other_lease, actual_tokens=10)

    queue.configure_scope(scope, QuotaLimits(1, 20, 100, 1))
    with pytest.raises(QuotaDispatchBlocked) as captured:
        queue.mark_dispatched(target_lease)

    assert getattr(captured.value, "blocked_by") == (
        QuotaWaitReason.REQUESTS_PER_MINUTE,
    )
    assert queue.usage_snapshot(scope).requests_last_minute == 1
    queue.abandon_before_dispatch(target_lease, disposition="cancel")

    clock.advance(MINUTE_SECONDS)
    exact_fit = _enqueue(queue, scope, clock, "exact-fit", tokens=20)
    exact_fit_lease = _acquire(queue, exact_fit)
    dispatched = queue.mark_dispatched(exact_fit_lease)
    queue.settle(dispatched, outcome="succeeded", actual_tokens=20)


@pytest.mark.parametrize(
    ("tightened", "expected_reason"),
    (
        (QuotaLimits(100, 19, 100, None), QuotaWaitReason.TOKENS_PER_MINUTE),
        (QuotaLimits(100, 100, 19, None), QuotaWaitReason.TOKENS_PER_WEEK),
    ),
)
def test_mark_dispatched_rechecks_latest_rolling_token_usage(
    tmp_path: Path,
    tightened: QuotaLimits,
    expected_reason: QuotaWaitReason,
) -> None:
    clock = _Clock()
    queue, scope = _queue(tmp_path, clock, rpm=100, tpm=100, weekly=100)
    target = _enqueue(queue, scope, clock, "target", tokens=10, owner="target")
    other = _enqueue(queue, scope, clock, "other", tokens=10, owner="other")
    target_lease = _acquire(queue, target, owner="target")
    other_lease = _acquire(queue, other, owner="other")
    _dispatch_and_settle(queue, other_lease, actual_tokens=10)
    queue.configure_scope(scope, tightened)

    with pytest.raises(QuotaDispatchBlocked) as captured:
        queue.mark_dispatched(target_lease)

    assert captured.value.blocked_by == (expected_reason,)
    assert queue.usage_snapshot(scope).requests_last_minute == 1
    queue.abandon_before_dispatch(target_lease, disposition="cancel")


def test_mark_dispatched_rechecks_other_in_flight_reservations(
    tmp_path: Path,
) -> None:
    clock = _Clock()
    queue, scope = _queue(tmp_path, clock, in_flight=2)
    target = _enqueue(queue, scope, clock, "target", owner="target")
    other = _enqueue(queue, scope, clock, "other", owner="other")
    target_lease = _acquire(queue, target, owner="target")
    _acquire(queue, other, owner="other")
    queue.configure_scope(scope, QuotaLimits(100, 100_000, 1_000_000, 1))

    with pytest.raises(QuotaDispatchBlocked) as captured:
        queue.mark_dispatched(target_lease)

    assert captured.value.blocked_by == (QuotaWaitReason.MAX_IN_FLIGHT,)
    assert queue.usage_snapshot(scope).requests_last_minute == 0
    queue.abandon_before_dispatch(target_lease, disposition="requeue")


def test_predispatch_lease_and_dispatch_never_cross_ticket_deadline(
    tmp_path: Path,
) -> None:
    clock = _Clock()
    queue, scope = _queue(tmp_path, clock, in_flight=1)
    ticket = _enqueue(queue, scope, clock, "deadline", deadline_in=10)
    lease = _acquire(queue, ticket, lease_seconds=30)

    assert lease.lease_until == ticket.deadline
    clock.advance(10)
    with pytest.raises(QuotaQueueError):
        queue.mark_dispatched(lease, lease_seconds=60)

    stored = queue.get_ticket(ticket.ticket_id)
    assert stored is not None
    assert stored.status == "expired"
    assert queue.usage_snapshot(scope).requests_last_minute == 0


def test_429_scope_cooldown_and_ticket_deadline(tmp_path: Path) -> None:
    clock = _Clock()
    queue, scope = _queue(tmp_path, clock, in_flight=1, queued_ttl=30)
    doomed = _enqueue(queue, scope, clock, "short", deadline_in=10)
    long = _enqueue(queue, scope, clock, "long", deadline_in=100)
    assert queue.apply_rate_limit_cooldown(
        scope, retry_after_seconds=20
    ) == clock() + 20

    terminal = queue.try_acquire_ticket(
        doomed.ticket_id, lease_owner="worker", lease_seconds=30
    )
    assert terminal.blocked_by == (QuotaWaitReason.TICKET_TERMINAL,)
    assert queue.get_ticket(doomed.ticket_id).status == "expired"  # type: ignore[union-attr]
    blocked = queue.try_acquire_ticket(
        long.ticket_id, lease_owner="worker", lease_seconds=30
    )
    assert blocked.blocked_by == (QuotaWaitReason.SCOPE_COOLDOWN,)
    clock.advance(20)
    assert _acquire(queue, long).ticket.ticket_id == long.ticket_id


def test_all_none_limits_are_unlimited_without_magic_integer(tmp_path: Path) -> None:
    clock = _Clock()
    queue, scope = _queue(
        tmp_path,
        clock,
        rpm=None,
        tpm=None,
        weekly=None,
        in_flight=None,
    )
    first = _enqueue(queue, scope, clock, "call-a", tokens=10**9, owner="a")
    second = _enqueue(queue, scope, clock, "call-b", tokens=10**9, owner="b")

    _acquire(queue, first)
    _acquire(queue, second)
    assert queue.usage_snapshot(scope).in_flight == 2


def test_two_instances_never_give_caller_another_ticket(tmp_path: Path) -> None:
    clock = _Clock()
    queue_a, scope = _queue(tmp_path, clock, in_flight=1)
    queue_b = ModelApiQuotaQueue(tmp_path / "quota.sqlite3", clock=clock)
    first = _enqueue(queue_a, scope, clock, "call-a", owner="worker-a")
    second = _enqueue(queue_a, scope, clock, "call-b", owner="worker-b")
    barrier = Barrier(2)

    def contend(queue: ModelApiQuotaQueue, ticket: QuotaTicket):
        barrier.wait()
        return queue.try_acquire_ticket(
            ticket.ticket_id,
            lease_owner=ticket.queue_owner,
            lease_seconds=30,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(
            executor.map(
                lambda args: contend(*args),
                ((queue_a, first), (queue_b, second)),
            )
        )

    leases = [result.lease for result in results if result.lease is not None]
    assert len(leases) == 1
    assert leases[0].ticket.ticket_id == first.ticket_id
    blocked = next(result for result in results if result.lease is None)
    assert set(blocked.blocked_by) <= {
        QuotaWaitReason.FIFO_PREDECESSOR,
        QuotaWaitReason.MAX_IN_FLIGHT,
    }
    _dispatch_and_settle(queue_a, leases[0])
    assert _acquire(queue_b, second).ticket.ticket_id == second.ticket_id
