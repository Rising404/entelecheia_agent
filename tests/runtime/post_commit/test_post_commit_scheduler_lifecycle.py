"""提交后调度必须跨重试、重启与显式等待保持真实完成状态。"""

from contextlib import contextmanager
from copy import deepcopy
from threading import Event, Lock

import pytest

from personagraph.runtime.post_commit import scheduler


class _Store:
    def __init__(self, status="pending"):
        self.lock = Lock()
        self.scope_entered = Event()
        self.jobs = [{"job_kind": "session_retrieval_index", "status": status,
                      "reason_code": None}]
        self.window = {"window_state": "post_commit_pending", "turn_id": "turn-1",
                       "state_version": 1}

    @contextmanager
    def session_database_scope(self, session_id):
        assert session_id == "session-1"
        self.scope_entered.set()
        yield

    def list_session_ids_for_post_commit_recovery(self):
        return ("session-1",)

    def get_session(self, session_id):
        assert session_id == "session-1"
        return {"id": session_id, "status": "active"}

    def inspect_turn_execution(self, session_id):
        with self.lock:
            return deepcopy({"window": self.window, "post_commit_jobs": self.jobs})

    def list_turn_post_commit_jobs(self, turn_id):
        assert turn_id == "turn-1"
        with self.lock:
            return deepcopy(self.jobs)

    def release_turn_execution_window(self, **kwargs):
        with self.lock:
            assert all(job["status"] in {"applied", "waived"} for job in self.jobs)
            self.window = {"window_state": "empty", "turn_id": None,
                           "state_version": 2}

    def set_status(self, status):
        with self.lock:
            self.jobs[0]["status"] = status


@pytest.fixture
def store():
    value = _Store()
    yield value
    scheduler.stop_turn_post_commit_workers(store=value, timeout_seconds=2)
    scheduler.resume_turn_post_commit_scheduling(store=value)


def _settle(store):
    store.set_status("applied")
    store.release_turn_execution_window()


def test_transient_failure_is_retried_without_another_chat_post(store, monkeypatch):
    calls = []
    monkeypatch.setattr(scheduler, "POST_COMMIT_POLL_SECONDS", 0.01, raising=False)

    def process(**_kwargs):
        calls.append(1)
        if len(calls) == 1:
            store.set_status("retryable_failed")
        else:
            _settle(store)

    monkeypatch.setattr(scheduler, "_process_due_turn_post_commit_jobs", process)
    scheduler.schedule_turn_post_commit_jobs(session_id="session-1", store=store)
    result = scheduler.wait_for_turn_post_commit_jobs(
        session_id="session-1", turn_id="turn-1", store=store, timeout_seconds=2,
    )
    assert result.status == "settled"
    assert result.window_released is True
    assert result.timed_out is False
    assert len(calls) == 2
    assert store.scope_entered.is_set()


def test_duplicate_scheduling_runs_one_handler_and_timeout_does_not_claim_success(
    store, monkeypatch,
):
    started, release = Event(), Event()
    calls = []

    def process(**_kwargs):
        calls.append(1)
        started.set()
        assert release.wait(2)
        _settle(store)

    monkeypatch.setattr(scheduler, "_process_due_turn_post_commit_jobs", process)
    scheduler.schedule_turn_post_commit_jobs(session_id="session-1", store=store)
    assert started.wait(1)
    try:
        for _ in range(8):
            scheduler.schedule_turn_post_commit_jobs(session_id="session-1", store=store)
        result = scheduler.wait_for_turn_post_commit_jobs(
            session_id="session-1", turn_id="turn-1", store=store, timeout_seconds=0.02,
        )
        assert result.status == "pending"
        assert result.timed_out is True
        assert result.window_released is False
        assert len(calls) == 1
    finally:
        release.set()
    result = scheduler.wait_for_turn_post_commit_jobs(
        session_id="session-1", turn_id="turn-1", store=store, timeout_seconds=2,
    )
    assert result.status == "settled"


def test_terminal_failure_does_not_auto_retry_or_unlock_input(store, monkeypatch):
    store.set_status("terminal_failed")
    calls = []
    monkeypatch.setattr(scheduler, "_process_due_turn_post_commit_jobs", lambda **kw: calls.append(kw))
    result = scheduler.wait_for_turn_post_commit_jobs(
        session_id="session-1", turn_id="turn-1", store=store, timeout_seconds=1,
    )
    assert result.status == "failed"
    assert result.window_released is False
    assert result.timed_out is False
    assert calls == []


def test_explicit_retry_can_restart_a_finished_worker(store, monkeypatch):
    store.set_status("terminal_failed")
    assert scheduler.wait_for_turn_post_commit_jobs(
        session_id="session-1", turn_id="turn-1", store=store, timeout_seconds=1,
    ).status == "failed"
    store.set_status("pending")  # 模拟经持久版本校验后的显式 retry。
    monkeypatch.setattr(scheduler, "_process_due_turn_post_commit_jobs", lambda **kw: _settle(store))
    assert scheduler.wait_for_turn_post_commit_jobs(
        session_id="session-1", turn_id="turn-1", store=store, timeout_seconds=1,
    ).status == "settled"


def test_host_start_rediscovers_processing_jobs_without_user_input(store, monkeypatch):
    from personagraph.runtime.post_commit.lifecycle import build_turn_post_commit_lifecycle

    store.set_status("processing")
    ran = Event()

    def process(**_kwargs):
        # 真实 runner 仅认领已到期租约；此替身只验证宿主确实重新调度。
        _settle(store)
        ran.set()

    monkeypatch.setattr(scheduler, "_process_due_turn_post_commit_jobs", process)
    lifecycle = build_turn_post_commit_lifecycle(store=store)
    try:
        assert lifecycle.start() is True
        assert lifecycle.start() is False
        assert ran.wait(2)
    finally:
        assert lifecycle.stop(timeout_seconds=2) is True


def test_shutdown_is_bounded_and_retains_unfinished_work(store, monkeypatch):
    started, release = Event(), Event()

    def process(**_kwargs):
        started.set()
        assert release.wait(2)
        _settle(store)

    monkeypatch.setattr(scheduler, "_process_due_turn_post_commit_jobs", process)
    scheduler.schedule_turn_post_commit_jobs(session_id="session-1", store=store)
    assert started.wait(1)
    try:
        assert scheduler.stop_turn_post_commit_workers(store=store, timeout_seconds=0.01) is False
        assert store.inspect_turn_execution("session-1")["window"]["window_state"] == "post_commit_pending"
    finally:
        release.set()
    assert scheduler.stop_turn_post_commit_workers(store=store, timeout_seconds=2) is True


def test_discovery_isolates_an_unreadable_session(store, monkeypatch):
    from personagraph.runtime.post_commit.lifecycle import build_turn_post_commit_lifecycle

    ran = Event()
    monkeypatch.setattr(
        store, "list_session_ids_for_post_commit_recovery", lambda: ("bad-session", "session-1"),
    )

    def process(**_kwargs):
        _settle(store)
        ran.set()

    monkeypatch.setattr(scheduler, "_process_due_turn_post_commit_jobs", process)
    lifecycle = build_turn_post_commit_lifecycle(store=store)
    try:
        lifecycle.start()
        assert ran.wait(2)
    finally:
        assert lifecycle.stop(timeout_seconds=2)


def test_shutdown_freezes_workers_before_waiting_for_discovery(store):
    from personagraph.runtime.post_commit.lifecycle import build_turn_post_commit_lifecycle

    lifecycle = build_turn_post_commit_lifecycle(store=store)

    class DiscoveryThread:
        def join(self, *, timeout):
            assert id(store) in scheduler._STOPPING_STORES

        def is_alive(self):
            return False

    lifecycle._thread = DiscoveryThread()
    assert lifecycle.stop(timeout_seconds=1)


@pytest.mark.parametrize("timeout", [-1, float("inf"), float("nan"), True])
def test_wait_rejects_invalid_deadlines(store, timeout):
    with pytest.raises(ValueError):
        scheduler.wait_for_turn_post_commit_jobs(
            session_id="session-1", turn_id="turn-1", store=store, timeout_seconds=timeout,
        )
