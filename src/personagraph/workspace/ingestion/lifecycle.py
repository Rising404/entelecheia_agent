"""持久 Document ingest worker 的非权威进程生命周期。"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
import logging
import threading
from typing import Protocol, runtime_checkable

from .storage import DocumentMaintenanceRunReport


_LOG = logging.getLogger(__name__)


@runtime_checkable
class _MaintenanceWorkerPort(Protocol):
    def drain_outbox_once(self, *, limit: int | None = None) -> int: ...

    def run_once(self, *, limit: int = 16) -> DocumentMaintenanceRunReport: ...


class DocumentMaintenanceWorkerLifecycle:
    """周期驱动一个可恢复 worker，wake 仅用于缩短持久工作的等待时间。"""

    def __init__(
        self,
        worker: _MaintenanceWorkerPort,
        *,
        poll_interval_seconds: float = 1.0,
        run_limit: int = 16,
        thread_name: str = "personagraph-document-maintenance",
        scope_factory: Callable[[], AbstractContextManager[object]] | None = None,
        on_started: Callable[["DocumentMaintenanceWorkerLifecycle"], None] | None = None,
        on_stopped: Callable[["DocumentMaintenanceWorkerLifecycle"], None] | None = None,
        after_pass: Callable[[], object] | None = None,
    ) -> None:
        if not isinstance(worker, _MaintenanceWorkerPort):
            raise TypeError("worker must implement the document maintenance port")
        if (
            isinstance(poll_interval_seconds, bool)
            or not isinstance(poll_interval_seconds, (int, float))
            or poll_interval_seconds <= 0
        ):
            raise ValueError("poll_interval_seconds must be positive")
        if isinstance(run_limit, bool) or not isinstance(run_limit, int) or run_limit <= 0:
            raise ValueError("run_limit must be a positive integer")
        if not isinstance(thread_name, str) or not thread_name.strip():
            raise ValueError("thread_name must be a non-empty string")
        self._worker = worker
        self._poll_interval_seconds = float(poll_interval_seconds)
        self._run_limit = run_limit
        self._thread_name = thread_name
        self._scope_factory = scope_factory
        self._on_started = on_started
        self._on_stopped = on_stopped
        self._after_pass = after_pass
        self._lock = threading.Lock()
        self._wake_event = threading.Event()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._passes = 0
        self._failures = 0
        self._last_error_type: str | None = None

    @property
    def is_running(self) -> bool:
        with self._lock:
            return self._thread is not None and self._thread.is_alive()

    @property
    def passes(self) -> int:
        with self._lock:
            return self._passes

    @property
    def failures(self) -> int:
        with self._lock:
            return self._failures

    @property
    def last_error_type(self) -> str | None:
        with self._lock:
            return self._last_error_type

    def start(self) -> bool:
        """启动 daemon 轮询器；精确活动重放返回 false。"""

        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return False
            self._stop_event = threading.Event()
            self._wake_event = threading.Event()
            thread = threading.Thread(
                target=self._run,
                name=self._thread_name,
                daemon=True,
            )
            self._thread = thread
        registered = False
        try:
            if self._on_started is not None:
                self._on_started(self)
                registered = True
            thread.start()
        except Exception:
            if registered and self._on_stopped is not None:
                self._on_stopped(self)
            with self._lock:
                if self._thread is thread:
                    self._thread = None
            raise
        return True

    def wake(self) -> None:
        """请求及时扫描，不改变任何持久 completion 状态。"""

        self._wake_event.set()

    def stop(self, *, timeout_seconds: float = 5.0) -> bool:
        """请求关闭，并至多等待调用方给定的有界超时。"""

        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or timeout_seconds < 0
        ):
            raise ValueError("timeout_seconds must be non-negative")
        with self._lock:
            thread = self._thread
        if thread is None:
            return True
        self._stop_event.set()
        self._wake_event.set()
        thread.join(float(timeout_seconds))
        stopped = not thread.is_alive()
        if stopped:
            with self._lock:
                if self._thread is thread:
                    self._thread = None
            if self._on_stopped is not None:
                self._on_stopped(self)
        return stopped

    def _run(self) -> None:
        scope = self._scope_factory() if self._scope_factory is not None else nullcontext()
        with scope:
            self._run_scoped()

    def _run_scoped(self) -> None:
        while not self._stop_event.is_set():
            pass_failed = False
            for operation in (
                self._worker.drain_outbox_once,
                lambda: self._worker.run_once(limit=self._run_limit),
                self._worker.drain_outbox_once,
                *((self._after_pass,) if self._after_pass is not None else ()),
            ):
                if self._stop_event.is_set():
                    break
                try:
                    report = operation()
                    if isinstance(report, DocumentMaintenanceRunReport) and (
                        report.retryable_failed or report.terminal_failed or report.lease_lost
                    ):
                        # Handled job failures do not escape as exceptions. Keep
                        # their safe counts without exposing paths or source text.
                        _LOG.warning(
                            "document_maintenance_run_incomplete claimed=%d applied=%d "
                            "retryable_failed=%d terminal_failed=%d lease_lost=%d",
                            report.claimed,
                            report.applied,
                            report.retryable_failed,
                            report.terminal_failed,
                            report.lease_lost,
                            extra={
                                "maintenance_claimed": report.claimed,
                                "maintenance_applied": report.applied,
                                "maintenance_retryable_failed": report.retryable_failed,
                                "maintenance_terminal_failed": report.terminal_failed,
                                "maintenance_lease_lost": report.lease_lost,
                            },
                        )
                except Exception as exc:
                    pass_failed = True
                    with self._lock:
                        self._failures += 1
                        self._last_error_type = type(exc).__name__
            if not pass_failed:
                with self._lock:
                    self._passes += 1
                    self._last_error_type = None
            if self._stop_event.is_set():
                break
            self._wake_event.wait(self._poll_interval_seconds)
            self._wake_event.clear()


__all__ = ["DocumentMaintenanceWorkerLifecycle"]
