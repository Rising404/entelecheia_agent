"""同步 Runtime 边界的进程内 Session 运行租约。"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from collections.abc import Iterator

_LOCKS: dict[str, threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()


class SessionRunBusyError(RuntimeError):
    code = "SESSION_RUN_BUSY"
    retryable = True

    def __init__(self, session_id: str) -> None:
        super().__init__(
            "Another runtime operation is already active for this session."
        )
        self.message = str(self)
        self.details = {"session_id": session_id}


@contextmanager
def session_run_guard(session_id: str | None) -> Iterator[None]:
    """用非阻塞进程内锁限制同一 Session 只能有一个活动 run / resume。

    不排队等待：冲突立即抛 SessionRunBusyError，Entry 再判断是不是同 request-id
    的只读重放。这个锁不跨进程/重启，持久 Window revision 与 lease 仍由 Store 校验。
    无 session_id 的内部调用跳过 guard，不能据此推断生产会话受到并发保护。
    """
    if not session_id:
        yield
        return

    with _LOCKS_GUARD:
        lock = _LOCKS.setdefault(session_id, threading.Lock())
    if not lock.acquire(blocking=False):
        raise SessionRunBusyError(session_id)
    try:
        yield
    finally:
        lock.release()


def is_session_run_active(session_id: str | None) -> bool:
    """返回此 Runtime 进程当前是否持有 Session 运行租约。

    这有意只作为本地存活提示，而非持久化权威状态。TurnExecutionWindow 仍是跨重启
    事实；下一输入审计仅用此提示避免将仍在本地运行的 Turn 误判为进程丢失恢复候选项。
    """

    if not session_id:
        return False
    with _LOCKS_GUARD:
        lock = _LOCKS.get(session_id)
        return bool(lock and lock.locked())
