"""一次检索调用的合作式控制与正文无关计时。

文件工具在域边界绑定窄取消端口，编码、召回、重排共享同一调用上下文；离线索引及
历史检索不绑定时行为不变。上下文不含证据正文，也不会投影到模型的工具结果。
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
import threading
import time

from .ports import RetrievalCancellationPort


class RerankingDeadlineReached(Exception):
    """尚未超时，但应停止可选评分并为完整证据投影保留时间。"""


class RetrievalExecution:
    """每次物理检索独立的审计收集器，不跨调用共享取消状态。"""

    def __init__(
        self,
        control: RetrievalCancellationPort,
        *,
        clock: Callable[[], float] = time.monotonic,
        projection_reserve_s: float = 2.0,
    ) -> None:
        if projection_reserve_s < 0:
            raise ValueError("projection reserve must be non-negative")
        self.control = control
        self.clock = clock
        self.projection_reserve_s = projection_reserve_s
        self._stages: dict[str, dict[str, int]] = {}
        self._metrics: dict[str, int | float | bool | str] = {}
        self._audit_lock = threading.Lock()

    def checkpoint(self) -> None:
        self.control.checkpoint()

    def check_rerank_budget(self, *, expected_compute_s: float = 0.0) -> None:
        self.checkpoint()
        remaining = self.control.remaining_seconds()
        if (
            remaining is not None
            and remaining <= self.projection_reserve_s + expected_compute_s
        ):
            self.set_metric("reranker_deadline_degraded", True)
            self.set_metric(
                "reranker_next_batch_estimate_ms",
                max(0, round(expected_compute_s * 1000)),
            )
            raise RerankingDeadlineReached("reranker_projection_reserve_reached")

    @contextmanager
    def measure(self, stage: str) -> Iterator[None]:
        started = self.clock()
        try:
            yield
        finally:
            self.record_duration(stage, self.clock() - started)

    def record_duration(self, stage: str, elapsed_s: float) -> None:
        with self._audit_lock:
            value = self._stages.setdefault(stage, {"calls": 0, "duration_ms": 0})
            value["calls"] += 1
            value["duration_ms"] += max(0, round(elapsed_s * 1000))

    def increment(self, name: str, amount: int = 1) -> None:
        with self._audit_lock:
            self._metrics[name] = int(self._metrics.get(name, 0)) + amount

    def set_metric(self, name: str, value: int | float | bool | str) -> None:
        with self._audit_lock:
            self._metrics[name] = value

    def snapshot(self) -> Mapping[str, object]:
        with self._audit_lock:
            audit = {
                "stages": {key: dict(value) for key, value in self._stages.items()},
                "metrics": dict(self._metrics),
            }
        return {**audit, **self.control.snapshot()}


_EXECUTION: ContextVar[RetrievalExecution | None] = ContextVar(
    "retrieval_execution", default=None
)
def current_execution() -> RetrievalExecution | None:
    return _EXECUTION.get()


@contextmanager
def execution_scope(execution: RetrievalExecution) -> Iterator[None]:
    token = _EXECUTION.set(execution)
    try:
        execution.checkpoint()
        yield
    finally:
        _EXECUTION.reset(token)


def checkpoint() -> None:
    execution = current_execution()
    if execution is not None:
        execution.checkpoint()


@contextmanager
def measure(stage: str) -> Iterator[None]:
    execution = current_execution()
    if execution is None:
        yield
    else:
        with execution.measure(stage):
            yield


@contextmanager
def wait_for_lock(
    lock: threading.Lock,
    stage: str,
    *,
    reserve_for_projection: bool = False,
) -> Iterator[None]:
    """等待可取消；取得资源后再查一次，已撤销的排队者绝不开始计算。"""

    execution = current_execution()

    def check() -> None:
        checkpoint()
        if reserve_for_projection and execution is not None:
            execution.check_rerank_budget()

    with measure(stage):
        check()
        while not lock.acquire(timeout=0.01):
            check()
    try:
        check()
        yield
    finally:
        lock.release()
