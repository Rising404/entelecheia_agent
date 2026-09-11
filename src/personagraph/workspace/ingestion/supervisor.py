"""发现 Project 并托管逐 Project 文档维护线程。"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import threading
import time
from typing import Protocol

from personagraph.workspace.storage.context import bind as bind_project_documents
from personagraph.workspace.storage.database import DocumentDatabase


class ProjectMaintenanceLifecycle(Protocol):
    """Supervisor 所拥有的最小子生命周期契约。"""

    def start(self) -> bool: ...

    def stop(self, *, timeout_seconds: float) -> bool: ...


class ProjectDescriptor(Protocol):
    """Supervisor 所需的 Project catalog 只读投影。"""

    project_id: str
    canonical_root: str
    documents_db_path: str


ProjectLoader = Callable[[], Sequence[ProjectDescriptor]]
ProjectLifecycleBuilder = Callable[
    [ProjectDescriptor, DocumentDatabase],
    ProjectMaintenanceLifecycle,
]


@dataclass(frozen=True, slots=True)
class _ManagedProject:
    signature: tuple[str, str]
    database: DocumentDatabase
    lifecycle: ProjectMaintenanceLifecycle


class ProjectDocumentMaintenanceSupervisor:
    """轮询 Project catalog，并让每个 Project 拥有独立维护 lifecycle。

    Catalog 只负责定位 Project；每个子 lifecycle 在构造时绑定对应的
    :class:`DocumentDatabase`，随后由子 lifecycle 将同一绑定带入自己的 worker
    线程。Supervisor 的发现线程不是作业权威，丢失一次轮询只会延迟发现。
    """

    def __init__(
        self,
        *,
        project_loader: ProjectLoader,
        lifecycle_builder: ProjectLifecycleBuilder,
        poll_interval_seconds: float = 1.0,
        thread_name: str = "personagraph-project-document-maintenance",
        child_stop_timeout_seconds: float = 5.0,
    ) -> None:
        if not callable(project_loader):
            raise TypeError("project_loader must be callable")
        if not callable(lifecycle_builder):
            raise TypeError("lifecycle_builder must be callable")
        for name, value in (
            ("poll_interval_seconds", poll_interval_seconds),
            ("child_stop_timeout_seconds", child_stop_timeout_seconds),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or value <= 0
            ):
                raise ValueError(f"{name} must be positive")
        if not isinstance(thread_name, str) or not thread_name.strip():
            raise ValueError("thread_name must be a non-empty string")

        self._project_loader = project_loader
        self._lifecycle_builder = lifecycle_builder
        self._poll_interval_seconds = float(poll_interval_seconds)
        self._thread_name = thread_name
        self._child_stop_timeout_seconds = float(child_stop_timeout_seconds)
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._wake_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._managed: dict[str, _ManagedProject] = {}
        self._failures = 0
        self._last_error_type: str | None = None

    @property
    def is_running(self) -> bool:
        with self._lock:
            return self._thread is not None and self._thread.is_alive()

    @property
    def managed_project_ids(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(sorted(self._managed))

    @property
    def failures(self) -> int:
        with self._lock:
            return self._failures

    @property
    def last_error_type(self) -> str | None:
        with self._lock:
            return self._last_error_type

    def start(self) -> bool:
        """同步接管已有 Project，再启动发现新 Project 的 daemon 线程。"""

        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return False
            self._stop_event = threading.Event()
            self._wake_event = threading.Event()

        try:
            self._reconcile_projects(raise_errors=True)
        except Exception:
            self._stop_all_children(timeout_seconds=self._child_stop_timeout_seconds)
            raise

        thread = threading.Thread(
            target=self._run,
            name=self._thread_name,
            daemon=True,
        )
        with self._lock:
            self._thread = thread
        thread.start()
        return True

    def wake(self) -> None:
        """请求尽快重新扫描 catalog；周期扫描仍是恢复路径。"""

        self._wake_event.set()

    def stop(self, *, timeout_seconds: float = 5.0) -> bool:
        """停止发现线程并在同一有界期限内停止所有子 lifecycle。"""

        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or timeout_seconds < 0
        ):
            raise ValueError("timeout_seconds must be non-negative")
        deadline = time.monotonic() + float(timeout_seconds)
        self._stop_event.set()
        self._wake_event.set()
        with self._lock:
            thread = self._thread
        if thread is not None:
            thread.join(max(0.0, deadline - time.monotonic()))
        supervisor_stopped = thread is None or not thread.is_alive()
        if supervisor_stopped:
            with self._lock:
                if self._thread is thread:
                    self._thread = None
        children_stopped = self._stop_all_children(
            timeout_seconds=max(0.0, deadline - time.monotonic())
        )
        return supervisor_stopped and children_stopped

    def _run(self) -> None:
        while not self._stop_event.is_set():
            self._wake_event.wait(self._poll_interval_seconds)
            self._wake_event.clear()
            if self._stop_event.is_set():
                break
            self._reconcile_projects(raise_errors=False)

    def _reconcile_projects(self, *, raise_errors: bool) -> None:
        try:
            projects = tuple(self._project_loader())
            project_by_id = {project.project_id: project for project in projects}
            if len(project_by_id) != len(projects):
                raise ValueError("project catalog returned duplicate project ids")
        except Exception as exc:
            self._record_failure(exc)
            if raise_errors:
                raise
            return

        with self._lock:
            stale_ids = tuple(set(self._managed) - set(project_by_id))
        errors: list[Exception] = []
        for project_id in stale_ids:
            try:
                self._retire_project(project_id)
            except Exception as exc:
                self._record_failure(exc)
                errors.append(exc)

        for project in projects:
            try:
                self._ensure_project(project)
            except Exception as exc:
                self._record_failure(exc)
                errors.append(exc)

        if errors and raise_errors:
            raise errors[0]
        if not errors:
            with self._lock:
                self._last_error_type = None

    def _ensure_project(self, project: ProjectDescriptor) -> None:
        if self._stop_event.is_set():
            return
        signature = (project.canonical_root, project.documents_db_path)
        with self._lock:
            current = self._managed.get(project.project_id)
        if current is not None and current.signature == signature:
            return
        if current is not None:
            self._retire_project(project.project_id)

        database = DocumentDatabase(
            project_id=project.project_id,
            project_root=project.canonical_root,
            db_path=project.documents_db_path,
        )
        with bind_project_documents(database):
            lifecycle = self._lifecycle_builder(project, database)
        if self._stop_event.is_set():
            return
        lifecycle.start()
        if self._stop_event.is_set():
            lifecycle.stop(timeout_seconds=self._child_stop_timeout_seconds)
            return
        managed = _ManagedProject(
            signature=signature,
            database=database,
            lifecycle=lifecycle,
        )
        with self._lock:
            self._managed[project.project_id] = managed

    def _retire_project(self, project_id: str) -> None:
        with self._lock:
            managed = self._managed.get(project_id)
        if managed is None:
            return
        stopped = managed.lifecycle.stop(
            timeout_seconds=self._child_stop_timeout_seconds
        )
        if not stopped:
            raise RuntimeError("project document maintenance lifecycle did not stop")
        with self._lock:
            if self._managed.get(project_id) is managed:
                del self._managed[project_id]

    def _stop_all_children(self, *, timeout_seconds: float) -> bool:
        deadline = time.monotonic() + timeout_seconds
        with self._lock:
            managed_items = tuple(self._managed.items())
        all_stopped = True
        for project_id, managed in managed_items:
            stopped = managed.lifecycle.stop(
                timeout_seconds=max(0.0, deadline - time.monotonic())
            )
            all_stopped = stopped and all_stopped
            if stopped:
                with self._lock:
                    if self._managed.get(project_id) is managed:
                        del self._managed[project_id]
        return all_stopped

    def _record_failure(self, exc: Exception) -> None:
        with self._lock:
            self._failures += 1
            self._last_error_type = type(exc).__name__


__all__ = [
    "ProjectDocumentMaintenanceSupervisor",
    "ProjectLifecycleBuilder",
    "ProjectLoader",
    "ProjectMaintenanceLifecycle",
]
