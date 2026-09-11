"""Project 文件准备的进程内执行 owner；索引实现通过构造函数注入。"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import threading

from personagraph.workspace.storage.database import DocumentDatabase
from .indexing_ports import IngestionGenerationIdentity
from .lifecycle import DocumentMaintenanceWorkerLifecycle
from .storage import DocumentMaintenanceRunReport
from .worker import DocumentMaintenanceWorker


class DocumentIngestOwnerConflict(RuntimeError):
    """同一 Project 已有后台 ingest owner 时拒绝第二个同步 owner。"""


@dataclass(frozen=True, slots=True)
class DocumentIngestExecutionOwner:
    """File preparation 可使用的单一 Project ingest 执行权威。

    后台 owner 只接受 ``wake``，调用线程不得直接进入其 worker/encoder。没有后台
    lifecycle 的离线入口才获得同步 ``run_once`` 权限。这样 pending job 仍由 SQLite
    状态恢复，但同一 Project 不会同时构造两份 BGE 模型并按 Outbox batch 分流。
    """

    generation_identity: IngestionGenerationIdentity
    kind: str
    _worker: DocumentMaintenanceWorker | None = None
    _lifecycle: DocumentMaintenanceWorkerLifecycle | None = None

    def __post_init__(self) -> None:
        if self.kind not in {"background", "synchronous"}:
            raise ValueError("document ingest owner kind is invalid")
        if self.kind == "background":
            if self._lifecycle is None or self._worker is not None:
                raise ValueError("background owner requires only a lifecycle")
        elif self._worker is None or self._lifecycle is not None:
            raise ValueError("synchronous owner requires only a worker")

    @property
    def can_run_synchronously(self) -> bool:
        return self.kind == "synchronous"

    def wake(self) -> None:
        if self._lifecycle is not None:
            self._lifecycle.wake()

    def run_once(self, *, limit: int) -> DocumentMaintenanceRunReport:
        if self._worker is None:
            raise DocumentIngestOwnerConflict(
                "background document ingest owner cannot run on the request thread"
            )
        return self._worker.run_once(limit=limit)

    def advance_outbox(self, limit: int) -> int:
        """视觉等已落库来源共享原 consumer；后台存在时请求线程仅唤醒它。"""

        if self._lifecycle is not None:
            self._lifecycle.wake()
            return 0
        assert self._worker is not None
        return self._worker.drain_outbox_once(limit=limit)

    @classmethod
    def synchronous(
        cls,
        worker: DocumentMaintenanceWorker,
    ) -> 'DocumentIngestExecutionOwner':
        return cls(
            generation_identity=worker.generation_identity,
            kind="synchronous",
            _worker=worker,
        )


@dataclass(frozen=True, slots=True)
class _BackgroundIngestOwner:
    generation_identity: IngestionGenerationIdentity
    lifecycle: DocumentMaintenanceWorkerLifecycle


_SYNCHRONOUS_WORKER_LOCK = threading.RLock()
_SYNCHRONOUS_WORKER: tuple[str, str, DocumentMaintenanceWorker] | None = None
_BACKGROUND_INGEST_OWNERS: dict[str, _BackgroundIngestOwner] = {}


def _database_key(database: DocumentDatabase) -> str:
    return str(database.db_path.expanduser().resolve())


def register_background_ingest_owner(
    database: DocumentDatabase,
    *,
    worker: DocumentMaintenanceWorker,
    lifecycle: DocumentMaintenanceWorkerLifecycle,
) -> None:
    """原子注册一个 Project 的唯一后台 consumer/encoder owner。"""

    global _SYNCHRONOUS_WORKER

    key = _database_key(database)
    with _SYNCHRONOUS_WORKER_LOCK:
        existing = _BACKGROUND_INGEST_OWNERS.get(key)
        if existing is not None and existing.lifecycle is not lifecycle:
            raise DocumentIngestOwnerConflict(
                "project already has an active document ingest owner"
            )
        _BACKGROUND_INGEST_OWNERS[key] = _BackgroundIngestOwner(
            generation_identity=worker.generation_identity,
            lifecycle=lifecycle,
        )
        if _SYNCHRONOUS_WORKER is not None and _SYNCHRONOUS_WORKER[0] == key:
            _SYNCHRONOUS_WORKER = None


def unregister_background_ingest_owner(
    database: DocumentDatabase,
    *,
    lifecycle: DocumentMaintenanceWorkerLifecycle,
) -> None:
    key = _database_key(database)
    with _SYNCHRONOUS_WORKER_LOCK:
        existing = _BACKGROUND_INGEST_OWNERS.get(key)
        if existing is not None and existing.lifecycle is lifecycle:
            del _BACKGROUND_INGEST_OWNERS[key]


def resolve_document_ingest_owner(
    database: DocumentDatabase,
    *,
    profile_key: str,
    build_worker: Callable[[], DocumentMaintenanceWorker],
) -> DocumentIngestExecutionOwner:
    """复用同一 Project 的唯一 owner；外层仅注入同步 worker 的索引配方装配。"""

    key = _database_key(database)
    with _SYNCHRONOUS_WORKER_LOCK:
        background = _BACKGROUND_INGEST_OWNERS.get(key)
        if background is not None:
            return DocumentIngestExecutionOwner(
                generation_identity=background.generation_identity,
                kind="background",
                _lifecycle=background.lifecycle,
            )
        return DocumentIngestExecutionOwner.synchronous(
            synchronous_ingest_worker(
                database,
                profile_key=profile_key,
                build_worker=build_worker,
            )
        )


def synchronous_ingest_worker(
    database: DocumentDatabase,
    *,
    profile_key: str,
    build_worker: Callable[[], DocumentMaintenanceWorker],
) -> DocumentMaintenanceWorker:
    """按 Project 数据库和不可变配方复用实例，禁止抢占后台 owner。"""

    global _SYNCHRONOUS_WORKER

    database_key = _database_key(database)
    with _SYNCHRONOUS_WORKER_LOCK:
        if database_key in _BACKGROUND_INGEST_OWNERS:
            raise DocumentIngestOwnerConflict(
                "project background document ingest owner is active"
            )
        cached = _SYNCHRONOUS_WORKER
        if (
            cached is not None
            and cached[0] == database_key
            and cached[1] == profile_key
        ):
            return cached[2]
        worker = build_worker()
        _SYNCHRONOUS_WORKER = (database_key, profile_key, worker)
        return worker


def reset_synchronous_ingest_worker() -> None:
    """释放测试/离线 owner cache；生产 lifecycle 应先显式停止。"""

    global _SYNCHRONOUS_WORKER

    with _SYNCHRONOUS_WORKER_LOCK:
        _SYNCHRONOUS_WORKER = None
        _BACKGROUND_INGEST_OWNERS.clear()


__all__ = [
    "DocumentIngestExecutionOwner",
    "DocumentIngestOwnerConflict",
    "register_background_ingest_owner",
    "unregister_background_ingest_owner",
    "resolve_document_ingest_owner",
    "synchronous_ingest_worker",
    "reset_synchronous_ingest_worker",
]
