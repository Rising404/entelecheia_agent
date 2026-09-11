"""宿主级提交后任务发现与关闭；业务处理仍只经过 scheduler/runner。"""

from __future__ import annotations

import logging
from threading import Event, Lock, Thread
from time import monotonic
from typing import TYPE_CHECKING

from .scheduler import (
    resume_turn_post_commit_scheduling,
    schedule_turn_post_commit_jobs,
    stop_turn_post_commit_workers,
    _validate_timeout,
)

if TYPE_CHECKING:
    from .contracts import PostCommitDiscoveryStore

_LOG = logging.getLogger(__name__)
POST_COMMIT_DISCOVERY_SECONDS = 2.0


class TurnPostCommitLifecycle:
    """启动/定期发现包括重启前遗留的窗口；状态读取接口不承担后台写入。"""

    def __init__(self, *, store: PostCommitDiscoveryStore):
        self._store = store
        self._stop = Event()
        self._lock = Lock()
        self._thread: Thread | None = None

    def start(self) -> bool:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return False
            resume_turn_post_commit_scheduling(store=self._store)
            self._stop.clear()
            self._thread = Thread(target=self._run, name="turn-post-commit-discovery", daemon=True)
            self._thread.start()
            return True

    def stop(self, *, timeout_seconds: float) -> bool:
        _validate_timeout(timeout_seconds)
        deadline = monotonic() + timeout_seconds
        self._stop.set()
        # 先冻结接单并通知现有 worker；发现器若阻塞，不能让其它 worker 继续接单。
        stop_turn_post_commit_workers(store=self._store, timeout_seconds=0)
        if self._thread is not None:
            self._thread.join(timeout=max(0.0, deadline - monotonic()))
        workers_stopped = stop_turn_post_commit_workers(
            store=self._store, timeout_seconds=max(0.0, deadline - monotonic()),
        )
        return workers_stopped and (self._thread is None or not self._thread.is_alive())

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                session_ids = self._store.list_session_ids_for_post_commit_recovery()
                for session_id in session_ids:
                    if self._stop.is_set():
                        break
                    self._discover_session(session_id)
            except Exception:
                _LOG.exception("could not discover turn post-commit work")
            self._stop.wait(POST_COMMIT_DISCOVERY_SECONDS)

    def _discover_session(self, session_id: str) -> None:
        try:
            # 先走目录定位校验：缺失的历史数据库不能因 inspect 的连接而被创建成空库。
            session = self._store.get_session(session_id)
            if session is None or session.get("status") == "trashed":
                return
            with self._store.session_database_scope(session_id):
                inspection = self._store.inspect_turn_execution(session_id)
            window = inspection.get("window")
            if isinstance(window, dict) and window.get("window_state") == "post_commit_pending":
                schedule_turn_post_commit_jobs(session_id=session_id, store=self._store)
        except Exception:
            # 一个损坏/被并发删除的 Session 不能阻止其他会话恢复。
            _LOG.exception("could not discover post-commit work for session %s", session_id)


def build_turn_post_commit_lifecycle(*, store=None) -> TurnPostCommitLifecycle:
    if store is None:
        from ...session import store as session_store

        store = session_store
    return TurnPostCommitLifecycle(store=store)
