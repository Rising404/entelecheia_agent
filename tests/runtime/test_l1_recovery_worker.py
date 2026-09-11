from __future__ import annotations

from threading import Event, Lock
import time

from personagraph.runtime.l1 import recovery_worker
from personagraph.runtime.turn.contracts import EntryTurnResult


_FEATURES = {
    "l1_max_attempts": 12,
    "l1_max_tool_calls_per_attempt": 8,
}


class _RecoveryStore:
    def __init__(
        self,
        candidates: dict[str, tuple[str, str | None]],
    ) -> None:
        self._candidates = candidates

    def inspect_turn_execution(self, session_id: str) -> dict[str, object]:
        run_id, _stage = self._candidates[session_id]
        return {
            "window": {
                "window_state": "active",
                "turn_id": f"turn-{session_id}",
                "current_l1_turn_run_id": run_id,
                "current_work_run_id": None,
                "current_attempt_id": None,
            }
        }

    def get_l1_turn_execution(self, **kwargs: object) -> dict[str, object] | None:
        session_id = str(kwargs["session_id"])
        run_id, stage = self._candidates[session_id]
        return {
            "run": {"l1_turn_run_id": run_id, "status": "active"},
            "state": (
                None
                if stage is None
                else {
                    "stage": stage,
                    "execution_config_json": "{}",
                    "execution_config_hash": "0" * 64,
                }
            ),
        }

    def list_sessions(self, *args: object, **kwargs: object) -> list[dict[str, object]]:
        del args, kwargs
        return [{"id": session_id} for session_id in sorted(self._candidates)]


def _wait_until(predicate, *, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("condition did not become true")


def test_scheduler_is_nonblocking_and_deduplicates_one_session(monkeypatch) -> None:
    session_id = "l1-recovery-deduplicated"
    store = _RecoveryStore({session_id: ("l1-run-one", "model")})
    started = Event()
    release = Event()
    calls: list[str] = []

    def blocking_resume(**kwargs: object):
        calls.append(str(kwargs["session_id"]))
        started.set()
        assert release.wait(2)
        return "not_waiting"

    monkeypatch.setattr(
        recovery_worker,
        "resume_active_l1_entry_turn",
        blocking_resume,
    )

    before = time.monotonic()
    assert recovery_worker.schedule_l1_turn_recovery(
        session_id=session_id,
        features=_FEATURES,
        store=store,
    )
    assert recovery_worker.schedule_l1_turn_recovery(
        session_id=session_id,
        features=_FEATURES,
        store=store,
    )
    assert time.monotonic() - before < 0.25
    assert started.wait(1)
    assert calls == [session_id]
    release.set()
    _wait_until(lambda: session_id not in recovery_worker._ACTIVE_SESSIONS)


def test_startup_scan_schedules_only_initialized_active_l1_runs(monkeypatch) -> None:
    store = _RecoveryStore(
        {
            "initialized": ("l1-run-initialized", "bootstrap"),
            "created-only": ("l1-run-created", None),
        }
    )
    calls: list[str] = []
    calls_lock = Lock()

    def record_resume(**kwargs: object):
        with calls_lock:
            calls.append(str(kwargs["session_id"]))
        return "not_waiting"

    monkeypatch.setattr(
        recovery_worker,
        "resume_active_l1_entry_turn",
        record_resume,
    )

    assert recovery_worker.recover_active_l1_turns(
        features=_FEATURES,
        store=store,
    ) == 1
    _wait_until(lambda: calls == ["initialized"])
    _wait_until(
        lambda: "initialized" not in recovery_worker._ACTIVE_SESSIONS
    )
    assert "created-only" not in recovery_worker._ACTIVE_SESSIONS


def test_worker_polls_busy_then_schedules_post_commit(monkeypatch) -> None:
    session_id = "l1-recovery-busy"
    store = _RecoveryStore({session_id: ("l1-run-busy", "finalizing")})
    outcomes: list[object] = [
        "busy",
        EntryTurnResult(
            session_id=session_id,
            turn_id=f"turn-{session_id}",
            status="completed",
            processing_level="L1",
            window_state="post_commit_pending",
            window_revision=8,
            reply="Recovered answer",
        ),
    ]
    post_commit: list[str] = []

    monkeypatch.setattr(
        recovery_worker,
        "resume_active_l1_entry_turn",
        lambda **_kwargs: outcomes.pop(0),
    )
    monkeypatch.setattr(recovery_worker, "L1_RECOVERY_POLL_SECONDS", 0.0)
    monkeypatch.setattr(
        recovery_worker,
        "schedule_turn_post_commit_jobs",
        lambda **kwargs: post_commit.append(str(kwargs["session_id"])),
    )

    recovery_worker._ACTIVE_SESSIONS.add(session_id)
    recovery_worker._run_scheduled_recovery(
        session_id=session_id,
        features=_FEATURES,
        store=store,
    )

    assert outcomes == []
    assert post_commit == [session_id]
    assert session_id not in recovery_worker._ACTIVE_SESSIONS


def test_created_run_without_state_is_not_startup_recoverable() -> None:
    session_id = "l1-created-without-state"
    store = _RecoveryStore({session_id: ("l1-run-created", None)})

    assert not recovery_worker._has_recoverable_l1_turn(
        session_id=session_id,
        store=store,
    )
