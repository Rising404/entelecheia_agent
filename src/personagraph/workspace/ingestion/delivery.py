"""把共享处理结果逐个交付给请求会话；不把跨库挂载伪装成源提交事务。"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
import sqlite3

from personagraph.input_processing.files import (
    SourceChangedDuringReadError,
    SourceFingerprint,
    SourceSizeLimitError,
    fingerprint_file,
)
from .storage import (
    DocumentIngestJobStatus,
    DocumentIngestJob,
    FilePreparationDeliveryStatus,
    SqliteDocumentIngestJobStore,
    SqliteFilePreparationRequestStore,
)


@dataclass(frozen=True, slots=True)
class FilePreparationDeliveryResult:
    delivered: bool
    reason_code: str | None = None


class FilePreparationDelivery:
    """先幂等挂载、后记回执；崩溃重试不重跑 parser 或改变共享 job。"""

    def __init__(
        self,
        *,
        connect_documents: Callable[[], sqlite3.Connection],
        validate_source_authority: Callable[[str, str], bool],
        mount_document: Callable[[str, str], bool],
        is_mounted: Callable[[str, str], bool],
        session_scope_factory: Callable[[str], AbstractContextManager[object]] | None = None,
        source_fingerprint: Callable[[Path], SourceFingerprint] = fingerprint_file,
    ) -> None:
        self._connect = connect_documents
        self._authorize = validate_source_authority
        self._mount = mount_document
        self._is_mounted = is_mounted
        self._scope = session_scope_factory
        self._fingerprint = source_fingerprint
        self._requests = SqliteFilePreparationRequestStore()
        self._jobs = SqliteDocumentIngestJobStore()

    def deliver_pending(self, *, job_id: str | None = None, limit: int = 100) -> int:
        with self._connect() as conn:
            pending = self._requests.list_pending(conn, job_id=job_id, limit=limit)
        return sum(self.deliver_request(request.request_id).delivered for request in pending)

    def deliver_request(self, request_id: str) -> FilePreparationDeliveryResult:
        with self._connect() as conn:
            request = self._requests.get(conn, request_id)
            if request is None:
                return FilePreparationDeliveryResult(False, "request_missing")
            if request.delivery_status is FilePreparationDeliveryStatus.BLOCKED:
                return FilePreparationDeliveryResult(False, request.reason_code)
            job = self._jobs.get(conn, request.job_id)
            if job is None or job.status is not DocumentIngestJobStatus.APPLIED:
                return FilePreparationDeliveryResult(False, "file_indexing_pending")
        try:
            scope = self._scope(request.session_id) if self._scope else nullcontext()
            with scope:
                reason = self._rejection(job, request.session_id)
                if reason:
                    if request.delivery_status is FilePreparationDeliveryStatus.PENDING:
                        self._block(request_id, reason)
                    return FilePreparationDeliveryResult(False, reason)
                # 不持有 Project 写锁跨入 Session 库。即便挂载成功后进程退出，下次仍可幂等重做。
                self._mount(str(job.document_id), request.session_id)
                if self._is_mounted(str(job.document_id), request.session_id) is not True:
                    return FilePreparationDeliveryResult(False, "file_mount_pending")
                reason = self._rejection(job, request.session_id)
                if reason:
                    if request.delivery_status is FilePreparationDeliveryStatus.PENDING:
                        self._block(request_id, reason)
                    return FilePreparationDeliveryResult(False, reason)
                with self._connect() as conn:
                    if not self._jobs.document_target_is_current(conn, job):
                        return FilePreparationDeliveryResult(False, "prepared_document_version_changed")
                    self._requests.mark_mounted(conn, request_id, now=_now())
                return FilePreparationDeliveryResult(True)
        except Exception:
            # 暂时的连接/Session scope/mount错误保留pending，独立于已完成共享job。
            return FilePreparationDeliveryResult(False, "file_mount_pending")

    def _rejection(self, job: DocumentIngestJob, session_id: str) -> str | None:
        if self._authorize(session_id, job.canonical_path) is not True:
            return "file_authority_denied"
        try:
            observed = self._fingerprint(Path(job.canonical_path))
        except SourceSizeLimitError:
            return "file_too_large"
        except SourceChangedDuringReadError:
            return "source_changed_during_read"
        except (FileNotFoundError, IsADirectoryError, NotADirectoryError, PermissionError):
            return "source_unavailable"
        if (observed.sha256, observed.size_bytes) != (job.source_sha256, job.source_size):
            return "frozen_source_mismatch"
        with self._connect() as conn:
            if not self._jobs.document_target_is_current(conn, job):
                return "prepared_document_version_changed"
        return None

    def _block(self, request_id: str, reason_code: str) -> None:
        with self._connect() as conn:
            self._requests.mark_blocked(conn, request_id, reason_code=reason_code, now=_now())


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
