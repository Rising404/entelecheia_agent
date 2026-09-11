"""按需准备已授权的文件：验证来源观察、持久化任务并有界等待执行。

执行者、处理配方及授权端口由组合层注入；本模块不理解模型资源句柄或检索后端。
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
import math
from pathlib import Path
from dataclasses import replace
import sqlite3
import time

from personagraph.input_processing.files import (
    SourceFingerprint,
    fingerprint_file,
)
from personagraph.input_processing.documents import ChunkingProfile
from personagraph.input_processing.documents.readers import (
    configured_processor_fingerprint,
)
from personagraph.workspace.storage.context import (
    initialize_current,
    open_current_connection,
    require_current,
)
from .storage import (
    DocumentIngestJob,
    DocumentIngestJobIdCollision,
    DocumentIngestJobStatus,
    SqliteDocumentIngestJobStore,
    SqliteFilePreparationRequestStore,
    FilePreparationDeliveryStatus,
)
from personagraph.workspace.files import (
    WorkspaceFileAuthority, FileSource, ProjectFileError, find_current_file_in_connection,
)
from personagraph.workspace.documents import application as docstore
from .execution import DocumentIngestExecutionOwner
from .delivery import FilePreparationDelivery
from .identity import derive_file_preparation_request_id
from .request import (
    FingerprintFile, ProcessorFingerprint, SourceAuthorityValidator,
    build_file_preparation_request,
    observe_file_preparation_source,
)
from .contracts import FilePreparationResult, FilePreparationStatus


DEFAULT_SYNCHRONOUS_FILE_BYTES = 8 * 1024 * 1024
DEFAULT_SYNCHRONOUS_PASSES = 2
DEFAULT_SYNCHRONOUS_RUN_LIMIT = 16
DEFAULT_PENDING_WAIT_SECONDS = 0.0
_PENDING_POLL_SECONDS = 0.25
_STALE_TERMINAL_REASONS = frozenset({
    "frozen_source_mismatch",
    "frozen_processor_mismatch",
    "frozen_chunk_recipe_mismatch",
    "prepared_source_path_mismatch",
})


def prepare_file(
    *,
    session_id: str,
    canonical_path: str,
    frozen_fingerprint: SourceFingerprint,
    ingest_owner: DocumentIngestExecutionOwner,
    chunking_profile: ChunkingProfile,
    validate_source_authority: SourceAuthorityValidator,
    connect_documents: Callable[[], sqlite3.Connection] = open_current_connection,
    job_store: SqliteDocumentIngestJobStore | None = None,
    request_id: str | None = None,
    with_summary: bool = False,
    source_fingerprint: FingerprintFile = fingerprint_file,
    processor_fingerprint: ProcessorFingerprint = configured_processor_fingerprint,
    synchronous_max_bytes: int = DEFAULT_SYNCHRONOUS_FILE_BYTES,
    synchronous_passes: int = DEFAULT_SYNCHRONOUS_PASSES,
    synchronous_run_limit: int = DEFAULT_SYNCHRONOUS_RUN_LIMIT,
    pending_wait_seconds: float = DEFAULT_PENDING_WAIT_SECONDS,
    checkpoint: Callable[[], None] | None = None,
) -> FilePreparationResult:
    """确保一个冻结且已授权文件可搜索，否则报告原因。

    返回任务、文档及版本身份，不添加模型侧资源包装。大文件持久化入队后保持
    ``PENDING``，交给后台
    维护生命周期；有界小文件则同步处理，在规范 worker 可立即完成时提供写后读。
    """

    _require_non_negative_integer("synchronous_max_bytes", synchronous_max_bytes)
    _require_positive_integer("synchronous_passes", synchronous_passes)
    _require_positive_integer("synchronous_run_limit", synchronous_run_limit)
    _require_non_negative_number("pending_wait_seconds", pending_wait_seconds)
    # 来源观测、登记和同步处理也消耗调用方的预算，不在 polling 前重新起算。
    deadline = time.monotonic() + float(pending_wait_seconds)
    if checkpoint is not None:
        checkpoint()

    if type(with_summary) is not bool:
        raise TypeError("with_summary must be a bool")
    source = observe_file_preparation_source(
        session_id=session_id, canonical_path=canonical_path,
        frozen_fingerprint=frozen_fingerprint, chunking_profile=chunking_profile,
        validate_source_authority=validate_source_authority,
        source_fingerprint=source_fingerprint, processor_fingerprint=processor_fingerprint,
    )
    if isinstance(source, FilePreparationResult):
        return source
    initialize_current()
    database = require_current()
    # 模型的持久 ToolCall 可以提供稳定 request_id。先复验原来源，避免重放时先注册/
    # 入队另一个版本，再在请求冲突处失败。来源变化由用户/模型发起新的调用来处理。
    if request_id is not None:
        refusal = check_preparation_replay_source(
            session_id=session_id, request_id=request_id, canonical_path=source.canonical_path,
            frozen_fingerprint=source.fingerprint, connect_documents=connect_documents,
            job_store=job_store,
        )
        if refusal is not None:
            return refusal
    authority = WorkspaceFileAuthority(database)
    try:
        relative_path = Path(source.canonical_path).relative_to(database.project_root).as_posix()
        previous = authority.get_file_by_relative_path(relative_path)
        registered = authority.ensure_current_path(
            relative_path, source=previous.source if previous else FileSource.WORKSPACE_EXISTING,
        )
    except (OSError, ValueError, ProjectFileError):
        return _blocked("file_registration_unavailable")
    if (
        registered.version.content_sha256 != source.fingerprint.sha256
        or registered.version.size_bytes != source.fingerprint.size_bytes
        or registered.file.observed_mtime_ns != source.fingerprint.mtime_ns
    ):
        return FilePreparationResult(
            status=FilePreparationStatus.STALE, reason_code="frozen_source_mismatch",
        )
    spec = build_file_preparation_request(
        source, file_id=registered.file.file_id,
        file_version_id=registered.version.file_version_id,
        generation_identity=ingest_owner.generation_identity,
    )
    operations = job_store or SqliteDocumentIngestJobStore()
    requests = SqliteFilePreparationRequestStore()
    now = datetime.now(timezone.utc).isoformat()
    if checkpoint is not None:
        checkpoint()
    try:
        with connect_documents() as conn:
            conn.execute("BEGIN IMMEDIATE")
            current_file = find_current_file_in_connection(conn, database, relative_path)
            if current_file is None or current_file[1].file_version_id != spec.file_version_id:
                conn.rollback()
                return FilePreparationResult(
                    status=FilePreparationStatus.STALE, reason_code="frozen_source_mismatch",
                )
            if validate_source_authority(session_id, source.canonical_path) is not True:
                conn.rollback()
                return _blocked("file_authority_denied")
            enqueued = operations.enqueue_in_transaction(conn, request=spec, now=now)
            operation_id = enqueued.job.job_id
            resolved_request_id = request_id or derive_file_preparation_request_id(
                session_id=session_id, job_id=operation_id,
                source_mtime_ns=source.fingerprint.mtime_ns, with_summary=with_summary,
            )
            requests.create_in_transaction(
                conn, request_id=resolved_request_id, job_id=operation_id,
                session_id=session_id, source_mtime_ns=source.fingerprint.mtime_ns,
                with_summary=with_summary, now=now,
            )
            conn.commit()
    except DocumentIngestJobIdCollision:
        return _blocked("file_readiness_operation_collision")

    current = enqueued.job
    # Wake 发生在 durable enqueue 之后；丢失这条进程内信号也只会退回周期扫描。
    # 后台 owner 不把 worker/encoder 暴露给请求线程，因此绝不会与 lifecycle 并发消费
    # 同一 Outbox。离线/测试入口没有 lifecycle 时仍保留原有同步写后读能力。
    ingest_owner.wake()
    if (
        current.status not in {
            DocumentIngestJobStatus.APPLIED,
            DocumentIngestJobStatus.TERMINAL_FAILED,
        }
        and frozen_fingerprint.size_bytes <= synchronous_max_bytes
        and ingest_owner.can_run_synchronously
    ):
        for _ in range(synchronous_passes):
            if checkpoint is not None:
                checkpoint()
            ingest_owner.run_once(limit=synchronous_run_limit)
            with connect_documents() as conn:
                refreshed = operations.get(conn, operation_id)
            if refreshed is None:
                return _blocked("file_readiness_operation_missing")
            current = refreshed
            if current.status in {
                DocumentIngestJobStatus.APPLIED,
                DocumentIngestJobStatus.TERMINAL_FAILED,
            }:
                break

    # 等待已有持久操作及本 Session 的交付；不重入 worker，不把 polling 交给模型。
    # SQLite 是唯一事实源，进程停止后原 ToolCall 可重放同 request，不依赖内存通知。
    while True:
        if checkpoint is not None:
            checkpoint()
        with connect_documents() as conn:
            current = operations.get(conn, operation_id)
        if current is None:
            return _blocked("file_readiness_operation_missing")
        result = _deliver_and_project(
            current=current, request_id=resolved_request_id, session_id=session_id,
            replayed=enqueued.replayed,
            expected_data_version=ingest_owner.generation_identity.version_id,
            connect_documents=connect_documents, operations=operations, requests=requests,
            validate_source_authority=validate_source_authority,
            source_fingerprint=source_fingerprint,
        )
        if checkpoint is not None:
            checkpoint()
        remaining = deadline - time.monotonic()
        if result.status is not FilePreparationStatus.PENDING or remaining <= 0:
            return result
        time.sleep(min(_PENDING_POLL_SECONDS, remaining))


def check_preparation_replay_source(
    *, session_id: str, request_id: str, canonical_path: str, frozen_fingerprint: SourceFingerprint,
    connect_documents: Callable[[], sqlite3.Connection] = open_current_connection,
    job_store: SqliteDocumentIngestJobStore | None = None,
) -> FilePreparationResult | None:
    """已有请求不可因恢复时文件变化而换来源；没有绑定时允许首次准备。"""

    with connect_documents() as conn:
        previous = SqliteFilePreparationRequestStore().get(conn, request_id)
        job = ((job_store or SqliteDocumentIngestJobStore()).get(conn, previous.job_id)
               if previous is not None else None)
    if previous is None:
        return None
    if previous.session_id != session_id or job is None:
        return _blocked("file_readiness_operation_collision")
    if (
        job.canonical_path != canonical_path
        or job.source_sha256 != frozen_fingerprint.sha256
        or job.source_size != frozen_fingerprint.size_bytes
        or previous.source_mtime_ns != frozen_fingerprint.mtime_ns
    ):
        return FilePreparationResult(
            status=FilePreparationStatus.STALE, reason_code="frozen_source_mismatch",
            operation_id=job.job_id, request_id=request_id,
            file_id=job.file_id, file_version_id=job.file_version_id,
        )
    return None


def _deliver_and_project(
    *, current: DocumentIngestJob, request_id: str, session_id: str, replayed: bool,
    expected_data_version: str, connect_documents: Callable[[], sqlite3.Connection],
    operations: SqliteDocumentIngestJobStore, requests: SqliteFilePreparationRequestStore,
    validate_source_authority: SourceAuthorityValidator, source_fingerprint: FingerprintFile,
) -> FilePreparationResult:
    delivery = None
    if current.status is DocumentIngestJobStatus.APPLIED:
        delivery = FilePreparationDelivery(
            connect_documents=connect_documents,
            validate_source_authority=validate_source_authority,
            mount_document=docstore.mount_document, is_mounted=docstore.is_mounted,
            source_fingerprint=source_fingerprint,
        ).deliver_request(request_id)
    with connect_documents() as conn:
        request = requests.get(conn, request_id)
    result = _project_job(
        current, replayed=replayed, expected_data_version=expected_data_version,
    )
    result = replace(
        result, request_id=request_id,
        file_id=current.file_id, file_version_id=current.file_version_id,
    )
    if request is None:
        return replace(result, status=FilePreparationStatus.BLOCKED, reason_code="request_missing")
    if request.delivery_status is FilePreparationDeliveryStatus.BLOCKED:
        return replace(
            result, status=FilePreparationStatus.BLOCKED, reason_code=request.reason_code,
            document_id=None, document_version_id=None,
        )
    if result.status is FilePreparationStatus.READY:
        if delivery is not None and not delivery.delivered:
            status = FilePreparationStatus.PENDING
            if delivery.reason_code in {"file_authority_denied", "source_unavailable", "file_too_large"}:
                status = FilePreparationStatus.BLOCKED
            elif delivery.reason_code in {"frozen_source_mismatch", "prepared_document_version_changed", "source_changed_during_read"}:
                status = FilePreparationStatus.STALE
            return replace(
                result, status=status, reason_code=delivery.reason_code,
                document_id=None, document_version_id=None,
            )
        if (
            request.delivery_status is not FilePreparationDeliveryStatus.MOUNTED
            or not docstore.is_mounted(str(current.document_id), session_id)
        ):
            return replace(result, status=FilePreparationStatus.PENDING, reason_code="file_mount_pending")
        with connect_documents() as conn:
            if not operations.document_target_is_current(conn, current):
                return replace(result, status=FilePreparationStatus.STALE, reason_code="prepared_document_version_changed")
    return result


def _project_job(
    job: DocumentIngestJob,
    *,
    replayed: bool,
    expected_data_version: str,
) -> FilePreparationResult:
    if job.status is DocumentIngestJobStatus.APPLIED:
        if job.retrieval_data_version != expected_data_version:
            return FilePreparationResult(
                status=FilePreparationStatus.STALE,
                operation_id=job.job_id,
                reason_code="retrieval_generation_changed",
                replayed=replayed,
            )
        return FilePreparationResult(
            status=FilePreparationStatus.READY,
            operation_id=job.job_id,
            replayed=replayed,
            retrieval_data_version=job.retrieval_data_version,
            document_id=job.document_id,
            document_version_id=job.document_version_id,
        )
    if job.status is DocumentIngestJobStatus.TERMINAL_FAILED:
        status = (
            FilePreparationStatus.STALE
            if job.reason_code in _STALE_TERMINAL_REASONS
            else FilePreparationStatus.BLOCKED
        )
        return FilePreparationResult(
            status=status,
            operation_id=job.job_id,
            reason_code=job.reason_code or "file_retrieval_terminal_failure",
            replayed=replayed,
        )
    return FilePreparationResult(
        status=FilePreparationStatus.PENDING,
        operation_id=job.job_id,
        # retry/checkpoint 细节保持 Host 私有。对模型而言，每个非终态持久化阶段只有一个
        # 稳定含义：索引仍待完成。
        reason_code="file_indexing_pending",
        replayed=replayed,
        retrieval_data_version=job.retrieval_data_version,
    )


def _blocked(reason_code: str) -> FilePreparationResult:
    return FilePreparationResult(
        status=FilePreparationStatus.BLOCKED,
        reason_code=reason_code,
    )


def _require_non_negative_integer(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def _require_positive_integer(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _require_non_negative_number(name: str, value: float) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError(f"{name} must be a non-negative number")


__all__ = [
    "DEFAULT_SYNCHRONOUS_FILE_BYTES",
    "FilePreparationResult",
    "FilePreparationStatus",
    "prepare_file",
]
