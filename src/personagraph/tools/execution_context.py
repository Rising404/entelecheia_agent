"""单次工具执行的 Host 身份与取消上下文，不属于模型可填写的工具参数。

上游事件只读；超时设置本调用自己的事件，不取消同 Turn 的其他工具。同步线程无法
强杀，checkpoint 是合作式停止边界，最终完成时间与取消请求时间分别记录。
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
import threading
import time


class ToolInvocationCancelled(Exception):
    """处理器在安全边界确认本次调用已经取消或超时。"""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


class ToolExecutionContext:
    """可跨本次工作线程共享、不能被模型覆盖的执行信号。"""

    def __init__(
        self,
        *,
        deadline_monotonic: float | None,
        parent_cancellation_event: threading.Event | None = None,
        clock: Callable[[], float] = time.monotonic,
        logical_tool_call_id: str | None = None,
        continuation_check: Callable[[], bool] | None = None,
    ) -> None:
        if logical_tool_call_id is not None and (
            not isinstance(logical_tool_call_id, str)
            or not logical_tool_call_id.strip()
        ):
            raise ValueError("logical_tool_call_id must be a non-empty Host identity")
        # 上层持久调用身份只向下透传；这里不按参数生成内容缓存键，也不签发恢复身份。
        self.logical_tool_call_id = logical_tool_call_id
        self.deadline_monotonic = deadline_monotonic
        self.cancellation_event = threading.Event()
        self._parent_event = parent_cancellation_event
        self._clock = clock
        self._continuation_check = continuation_check
        self._next_continuation_check = 0.0
        self._lock = threading.Lock()
        self._reason: str | None = None
        self._requested_at: float | None = None
        self._acknowledged_at: float | None = None
        self._settled_status: str | None = None
        self._handler_finished = False
        self._settlement_observers: list[Callable[[str], None]] = []

    def cancel(self, reason: str) -> None:
        with self._lock:
            if self._reason is None:
                self._reason = reason
                self._requested_at = self._clock()
                self.cancellation_event.set()

    def interruption_code(self) -> str | None:
        if self._parent_event is not None and self._parent_event.is_set():
            self.cancel("execution_cancelled")
        if (
            self.deadline_monotonic is not None
            and self._clock() >= self.deadline_monotonic
        ):
            self.cancel("execution_timeout")
        with self._lock:
            return self._reason

    def remaining_seconds(self) -> float | None:
        if self.deadline_monotonic is None:
            return None
        return max(0.0, self.deadline_monotonic - self._clock())

    def checkpoint(self) -> None:
        if (
            self._continuation_check is not None
            and self.interruption_code() is None
            and self._clock() >= self._next_continuation_check
        ):
            # 领域等待循环可以频繁让出控制；持久执行权检查不随毫秒级轮询放大。
            self._next_continuation_check = self._clock() + 1.0
            try:
                current = self._continuation_check() is True
            except Exception:
                current = False
            if not current:
                self.cancel("execution_authority_lost")
        reason = self.interruption_code()
        if reason is not None:
            self.acknowledge()
            raise ToolInvocationCancelled(reason)

    def acknowledge(self) -> None:
        with self._lock:
            if self._reason is not None and self._acknowledged_at is None:
                self._acknowledged_at = self._clock()

    def on_settled(self, observer: Callable[[str], None]) -> None:
        """只供审计使用：终态确定且处理器退出后，才记录交付与完整清理状态。

        超时返回不会等待工作线程；其审计回调在工作线程随后退出时运行一次。
        """

        with self._lock:
            status = self._settled_status
            if status is None or not self._handler_finished:
                self._settlement_observers.append(observer)
                return
        self._notify(observer, status)

    def settle(self, status: str) -> None:
        with self._lock:
            if self._settled_status is not None:
                return
            self._settled_status = status
            observers = self._take_observers_if_finished()
        for observer in observers:
            self._notify(observer, status)

    def finish_handler(self) -> None:
        self.acknowledge()
        with self._lock:
            self._handler_finished = True
            status = self._settled_status
            observers = self._take_observers_if_finished() if status is not None else ()
        for observer in observers:
            self._notify(observer, status)

    def _take_observers_if_finished(self) -> tuple[Callable[[str], None], ...]:
        if not self._handler_finished:
            return ()
        observers = tuple(self._settlement_observers)
        self._settlement_observers.clear()
        return observers

    @staticmethod
    def _notify(observer: Callable[[str], None], status: str) -> None:
        try:
            observer(status)
        except Exception:
            # 观测失败不能改变已确定的工具结果或触发再次执行。
            pass

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            elapsed = (
                max(0, round((self._acknowledged_at - self._requested_at) * 1000))
                if self._acknowledged_at is not None and self._requested_at is not None
                else None
            )
            return {
                "cancellation_requested": self._reason is not None,
                "cancellation_reason": self._reason,
                "cancellation_acknowledged": self._acknowledged_at is not None,
                "cancellation_ack_ms": elapsed,
                "handler_finished": self._handler_finished,
            }


_CURRENT_EXECUTION: ContextVar[ToolExecutionContext | None] = ContextVar(
    "tool_execution_context", default=None
)


def current_tool_execution() -> ToolExecutionContext | None:
    return _CURRENT_EXECUTION.get()


@contextmanager
def tool_execution_scope(control: ToolExecutionContext) -> Iterator[None]:
    token = _CURRENT_EXECUTION.set(control)
    try:
        yield
    finally:
        _CURRENT_EXECUTION.reset(token)
