"""启动持久化且已初始化的 L1 TurnRuns 的重新发现。"""

from __future__ import annotations

import logging
from contextlib import AbstractContextManager
from threading import Event, Lock, Thread
from typing import Any, Protocol

from personagraph.configuration.features import load_features
from ..turn.contracts import EntryTurnResult
from ..post_commit.scheduler import schedule_turn_post_commit_jobs


L1_RECOVERY_POLL_SECONDS = 2.0
L1_RECOVERY_MAX_BUSY_POLLS = 180

_ACTIVE_SESSIONS: set[str] = set()
_ACTIVE_LOCK = Lock()
_LOG = logging.getLogger(__name__)


class L1RecoveryWorkerStore(Protocol):
    def inspect_turn_execution(self, session_id: str) -> dict[str, object]: ...

    def get_l1_turn_execution(self, **kwargs: object) -> dict[str, object] | None: ...

    def list_sessions(
        self,
        *args: object,
        **kwargs: object,
    ) -> list[dict[str, object]]: ...


def schedule_l1_turn_recovery(
    *,
    session_id: str,
    features: dict[str, Any] | None = None,
    store: L1RecoveryWorkerStore | None = None,
) -> bool:
    """为已有且可恢复的 L1 run 安排一个守护线程，不接受新用户 Turn。

    进程内 _ACTIVE_SESSIONS 避免同会话重复 worker；True 也可能表示已有 worker，
    不代表恢复成功。真正认领原 run 的跨进程租约仍由 Entry / Store 执行。
    """

    if not session_id.strip():
        raise ValueError("session_id must not be blank")
    resolved_store = _default_store() if store is None else store
    if not _has_recoverable_l1_turn(
        session_id=session_id,
        store=resolved_store,
    ):
        return False
    with _ACTIVE_LOCK:
        if session_id in _ACTIVE_SESSIONS:
            return True
        _ACTIVE_SESSIONS.add(session_id)
    try:
        Thread(
            target=_run_scheduled_recovery,
            kwargs={
                "session_id": session_id,
                "features": dict(features or load_features(None)),
                "store": resolved_store,
            },
            name=f"l1-turn-recovery-{session_id[:24]}",
            daemon=True,
        ).start()
    except RuntimeError:
        with _ACTIVE_LOCK:
            _ACTIVE_SESSIONS.discard(session_id)
        _LOG.exception("could not start L1 Turn recovery worker")
        return False
    return True


def recover_active_l1_turns(
    *,
    features: dict[str, Any] | None = None,
    store: L1RecoveryWorkerStore | None = None,
) -> int:
    """Host 启动时枚举 Session，把可恢复的已初始化 L1 run 交给后台调度。

    返回调度计数，不等待模型执行或正式交付完成；不能把它当作 startup 恢复成功数。
    """

    scheduled = 0
    resolved_store = _default_store() if store is None else store
    resolved_features = dict(features or load_features(None))
    for session in resolved_store.list_sessions(status="all"):
        session_id = str(session.get("id") or "")
        if session_id and schedule_l1_turn_recovery(
            session_id=session_id,
            features=resolved_features,
            store=resolved_store,
        ):
            scheduled += 1
    return scheduled


def _run_scheduled_recovery(
    *,
    session_id: str,
    features: dict[str, Any],
    store: L1RecoveryWorkerStore,
) -> None:
    try:
        with _session_database_scope(session_id):
            _run_scheduled_recovery_scoped(
                session_id=session_id,
                features=features,
                store=store,
            )
    except Exception:
        _LOG.exception("L1 Turn recovery worker exited unexpectedly")
    finally:
        with _ACTIVE_LOCK:
            _ACTIVE_SESSIONS.discard(session_id)


def _run_scheduled_recovery_scoped(
    *,
    session_id: str,
    features: dict[str, Any],
    store: L1RecoveryWorkerStore,
) -> None:
    """在已重建 Session scope 内恢复原 L1 run，只对 busy 做有界等待。

    业务结果返回后停止本 worker；若正式交付进入 post_commit_pending，则接力调度
    派生任务。这里的 busy 轮询不是模型重试，也不重置原 Turn 的 deadline。
    """

    busy_polls = 0
    while _has_recoverable_l1_turn(session_id=session_id, store=store):
        outcome = resume_active_l1_entry_turn(
            session_id=session_id,
            features=features,
            store=store,  # type: ignore[arg-type]
        )
        if outcome == "busy":
            busy_polls += 1
            if busy_polls >= L1_RECOVERY_MAX_BUSY_POLLS:
                _LOG.warning(
                    "L1 Turn recovery remained busy for session %s",
                    session_id,
                )
                return
            Event().wait(L1_RECOVERY_POLL_SECONDS)
            continue
        if isinstance(outcome, EntryTurnResult) and (
            outcome.window_state == "post_commit_pending"
        ):
            schedule_turn_post_commit_jobs(
                session_id=session_id,
                store=store,  # type: ignore[arg-type]
            )
        return


def resume_active_l1_entry_turn(**kwargs: object) -> object:
    """Lazy Entry seam retained for tests and recovery-worker monkeypatching."""

    from ..entry import resume_active_l1_entry_turn as resume

    return resume(**kwargs)


def _default_store() -> L1RecoveryWorkerStore:
    from ...session import store

    return store


def _session_database_scope(session_id: str) -> AbstractContextManager[object]:
    from ...session import store

    return store.session_database_scope(session_id)


def _has_recoverable_l1_turn(
    *,
    session_id: str,
    store: L1RecoveryWorkerStore,
) -> bool:
    try:
        inspection = store.inspect_turn_execution(session_id)
        window = inspection.get("window")
        if (
            not isinstance(window, dict)
            or str(window.get("window_state") or "") != "active"
            or not window.get("turn_id")
            or not window.get("current_l1_turn_run_id")
            or window.get("current_work_run_id") is not None
            or window.get("current_attempt_id") is not None
        ):
            return False
        execution = store.get_l1_turn_execution(
            session_id=session_id,
            turn_id=str(window["turn_id"]),
        )
        if not isinstance(execution, dict):
            return False
        run = execution.get("run")
        state = execution.get("state")
        return bool(
            isinstance(run, dict)
            and isinstance(state, dict)
            and str(run.get("status") or "") == "active"
            and str(run.get("l1_turn_run_id") or "")
            == str(window.get("current_l1_turn_run_id") or "")
            and str(state.get("stage") or "")
            in {"bootstrap", "model", "tool", "observation", "finalizing"}
        )
    except Exception:
        return False


__all__ = [
    "L1_RECOVERY_MAX_BUSY_POLLS",
    "L1_RECOVERY_POLL_SECONDS",
    'L1RecoveryWorkerStore',
    "recover_active_l1_turns",
    "schedule_l1_turn_recovery",
]
