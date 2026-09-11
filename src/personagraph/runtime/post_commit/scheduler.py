"""按 Session 去重唤醒持久提交后工作，持续重试并提供显式收尾等待。

线程仍不是完成权威：SQLite jobs/租约决定是否可执行，宿主生命周期负责重新发现和停止，
短命调用方必须等待结算。不能以收到回答或线程退出代替索引完成。
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field, replace
from threading import Event, Lock, Thread
from time import monotonic
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .contracts import ScheduledTurnPostCommitJobStore, TurnPostCommitSettlement


_LOG = logging.getLogger(__name__)
POST_COMMIT_POLL_SECONDS = 1.0
_WORKERS_LOCK = Lock()
_WORKERS: dict[tuple[int, str], _Worker] = {}
_STOPPING_STORES: dict[int, object] = {}


@dataclass
class _Worker:
    stop: Event = field(default_factory=Event)
    done: Event = field(default_factory=Event)
    thread: Thread | None = None


def schedule_turn_post_commit_jobs(
    *,
    session_id: str,
    store: ScheduledTurnPostCommitJobStore,
) -> None:
    """非阻塞唤醒；同进程同 Store/Session 只保留一个持续 worker。"""

    if not session_id.strip():
        raise ValueError("session_id must not be blank")
    key = (id(store), session_id)
    with _WORKERS_LOCK:
        if id(store) in _STOPPING_STORES or key in _WORKERS:
            return
        worker = _Worker()
        worker.thread = Thread(
            target=_worker_loop,
            kwargs={"session_id": session_id, "store": store, "worker": worker},
            name=f"turn-post-commit-{session_id[:24]}",
            daemon=True,
        )
        _WORKERS[key] = worker
        try:
            worker.thread.start()
        except BaseException:
            _WORKERS.pop(key, None)
            raise


def _worker_loop(*, session_id: str, store: ScheduledTurnPostCommitJobStore, worker: _Worker) -> None:
    try:
        while not worker.stop.is_set():
            try:
                with store.session_database_scope(session_id):
                    inspection = store.inspect_turn_execution(session_id)
                    window = inspection.get("window")
                    jobs = inspection.get("post_commit_jobs")
                    if not isinstance(window, dict) or window.get("window_state") != "post_commit_pending":
                        return
                    if not isinstance(jobs, list) or not jobs:
                        return
                    states = {job.get("status") for job in jobs if isinstance(job, dict)}
                    if states <= {"applied", "waived"}:
                        from .runner import release_turn_window_if_post_commit_settled

                        release_turn_window_if_post_commit_settled(session_id=session_id, store=store)
                        return
                    if not states & {"pending", "processing", "retryable_failed"}:
                        return  # 终态失败只能由显式用户操作重试或跳过。
                _run_scheduled_post_commit_jobs(session_id=session_id, store=store)
            except Exception:
                _LOG.exception("turn post-commit scheduling pass failed")
            # 持久 next_retry_at / lease_until 仍由 claim 检查；等待不是重新执行的授权。
            worker.stop.wait(POST_COMMIT_POLL_SECONDS)
    finally:
        with _WORKERS_LOCK:
            _WORKERS.pop((id(store), session_id), None)
        worker.done.set()


def wait_for_turn_post_commit_jobs(
    *, session_id: str, turn_id: str, store: ScheduledTurnPostCommitJobStore,
    timeout_seconds: float,
) -> TurnPostCommitSettlement:
    """等待同一异步路径结算；到期返回真实未完成状态，不撤销已提交的回答。"""
    from .contracts import TurnPostCommitSettlement
    from .settlement import read_turn_post_commit_settlement

    _validate_timeout(timeout_seconds)
    if not session_id.strip() or not turn_id.strip():
        raise ValueError("session_id and turn_id must not be blank")
    deadline = monotonic() + timeout_seconds
    while True:
        try:
            state = read_turn_post_commit_settlement(session_id=session_id, turn_id=turn_id, store=store)
        except Exception:
            _LOG.exception("could not read turn post-commit settlement")
            return TurnPostCommitSettlement("unavailable", turn_id, False)
        if state.status != "pending":
            return state
        remaining = deadline - monotonic()
        if remaining <= 0:
            return replace(state, timed_out=True)
        schedule_turn_post_commit_jobs(session_id=session_id, store=store)
        with _WORKERS_LOCK:
            worker = _WORKERS.get((id(store), session_id))
        if worker is not None:
            worker.done.wait(min(remaining, POST_COMMIT_POLL_SECONDS))
        else:
            Event().wait(min(remaining, POST_COMMIT_POLL_SECONDS))


def stop_turn_post_commit_workers(*, store: ScheduledTurnPostCommitJobStore, timeout_seconds: float) -> bool:
    """停止本宿主继续接单并有界等待；正在 I/O 的线程不伪装为已取消或已完成。"""
    _validate_timeout(timeout_seconds)
    deadline = monotonic() + timeout_seconds
    with _WORKERS_LOCK:
        _STOPPING_STORES[id(store)] = store
        workers = [worker for (store_id, _), worker in _WORKERS.items() if store_id == id(store)]
        for worker in workers:
            worker.stop.set()
    for worker in workers:
        worker.done.wait(max(0.0, deadline - monotonic()))
    return all(worker.done.is_set() for worker in workers)


def resume_turn_post_commit_scheduling(*, store: ScheduledTurnPostCommitJobStore) -> None:
    """宿主显式启动时重新允许接单；调用不会执行或结算任何 job。"""
    with _WORKERS_LOCK:
        _STOPPING_STORES.pop(id(store), None)


def _validate_timeout(value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError("timeout_seconds must be finite and non-negative")


def _run_scheduled_post_commit_jobs(
    *,
    session_id: str,
    store: ScheduledTurnPostCommitJobStore,
) -> None:
    """后台线程先重建原 Session/Project scope，再处理一项持久 post-commit job。

    Thread 不自动继承请求 ContextVar，不能直接沿用调用方的数据库假设。
    异常只记进程日志；原 Window/job 留给后续调度恢复，不把正式回复改成失败。
    """

    try:
        # ContextVars 不会传播到新建的 ``Thread``。重建完整的会话/项目存储作用域，
        # 确保此尽力而为的工作线程不会回退到进程全局旧数据库。
        with store.session_database_scope(session_id):
            _process_due_turn_post_commit_jobs(session_id=session_id, store=store)
    except Exception:
        # 持久窗口仍处于持有状态，后续读取会暴露未完成作业。不要让未处理的守护
        # 线程异常造成正式回复本身失败的错误印象。
        _LOG.exception("turn post-commit worker exited unexpectedly")


def _process_due_turn_post_commit_jobs(
    *,
    session_id: str,
    store: ScheduledTurnPostCommitJobStore,
) -> object:
    """在已建立存储 scope 的 worker 内惰性加载 runner 并处理一项到期任务。

    函数返回不代表所有会话作业都已结算；实际数量与 Window 释放结果由 runner 返回。
    """

    from .runner import process_due_turn_post_commit_jobs

    # 每项之间返回调度循环检查 stop，关停期间不再认领下一项持久工作。
    return process_due_turn_post_commit_jobs(session_id=session_id, store=store, max_jobs=1)
