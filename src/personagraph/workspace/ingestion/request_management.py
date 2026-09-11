"""Session 文件准备请求的查询、幂等重放与显式重试。"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import sqlite3

from personagraph.configuration.paths import deny_reason
from personagraph.input_processing.documents import ChunkingProfile
from personagraph.input_processing.documents.readers import (
    configured_processor_fingerprint,
)
from personagraph.input_processing.files import (
    SourceChangedDuringReadError,
    SourceFingerprint,
    SourceSizeLimitError,
    fingerprint_file,
)
from personagraph.workspace.documents import application as docstore
from personagraph.workspace.storage.context import (
    initialize_current,
    open_current_connection,
)
from .storage import (
    DocumentIngestJob,
    DocumentIngestJobStatus,
    FilePreparationDeliveryStatus,
    FilePreparationRequest,
    SqliteDocumentIngestJobStore,
    SqliteFilePreparationRequestStore,
)
from .indexing_ports import IngestionGenerationIdentity
from .request import ProcessorFingerprint, SourceAuthorityValidator


class FilePreparationRequestCollision(RuntimeError):
    """同一客户端请求 ID 被复用于另一份命令。"""


class FilePreparationRequestIntegrityError(RuntimeError):
    """请求指向的共享处理作业缺失或发生了不可能的漂移。"""


class FilePreparationRequestNotRetryable(RuntimeError):
    """请求已经完成交付，不能再次进入准备状态。"""


@dataclass(frozen=True, slots=True)
class FilePreparationOperation:
    """一个 Session 请求及其指向的共享处理作业。"""

    request: FilePreparationRequest
    processing_job: DocumentIngestJob


@dataclass(frozen=True, slots=True)
class FilePreparationRetryResult:
    operation: FilePreparationOperation
    replayed: bool


RequeueLinkedOutboxEvents = Callable[
    [sqlite3.Connection, tuple[str, ...], str],
    None,
]


def file_preparation_retry_rejection(
    operation: FilePreparationOperation,
    *,
    generation_identity: IngestionGenerationIdentity,
    chunking_profile: ChunkingProfile,
    validate_source_authority: SourceAuthorityValidator,
    source_fingerprint: Callable[[Path], SourceFingerprint] = fingerprint_file,
    processor_fingerprint: ProcessorFingerprint = configured_processor_fingerprint,
) -> str | None:
    """证明原请求仍对应当前 authority、来源、配方与目标 generation。"""

    request = operation.request
    job = operation.processing_job
    source = Path(job.canonical_path).expanduser()
    try:
        canonical = source.resolve(strict=True)
    except OSError:
        return "source_unavailable"
    if str(canonical) != job.canonical_path or not canonical.is_file():
        return "prepared_source_path_mismatch"
    try:
        authorized = validate_source_authority(request.session_id, str(canonical))
    except Exception:
        authorized = False
    if authorized is not True:
        return "file_authority_denied"
    if denied := deny_reason(canonical):
        return denied
    try:
        current = source_fingerprint(canonical)
    except SourceSizeLimitError:
        return "too_large"
    except SourceChangedDuringReadError:
        return "source_changed_during_ingest"
    except OSError:
        return "source_unavailable"
    if (
        current.sha256 != job.source_sha256
        or current.size_bytes != job.source_size
        or current.mtime_ns != request.source_mtime_ns
    ):
        return "frozen_source_mismatch"
    try:
        processor = processor_fingerprint(canonical)
    except (OSError, RuntimeError, ValueError):
        return "frozen_processor_mismatch"
    if processor is None or str(processor) != job.processor_fingerprint:
        return "frozen_processor_mismatch"
    if (
        chunking_profile.fingerprint() != job.chunker_fingerprint
        or docstore.DOCUMENT_CHUNK_CONTRACT_VERSION != job.chunk_contract_version
    ):
        return "frozen_chunk_recipe_mismatch"
    if (
        generation_identity.version_id != job.target_generation_id
        or generation_identity.fingerprint != job.target_generation_fingerprint
    ):
        return "retrieval_generation_changed"
    return None


def find_file_preparation_replay(
    *,
    request_id: str,
    session_id: str,
    canonical_path: str,
    with_summary: bool,
    connect_documents: Callable[[], sqlite3.Connection] = open_current_connection,
) -> FilePreparationOperation | None:
    """返回精确客户端命令的既有请求；ID 复用一律报告碰撞。

    命令重放只比较客户端稳定字段。首次接受时冻结的文件字节、mtime 与处理
    generation 不会因响应丢失后的重试而被重新解释。
    """

    initialize_current()
    with connect_documents() as conn:
        operation = _load_operation(conn, request_id)
    if operation is None:
        return None
    if (
        operation.request.session_id != session_id
        or operation.request.with_summary is not with_summary
        or operation.processing_job.canonical_path != canonical_path
    ):
        raise FilePreparationRequestCollision(
            "file preparation request_id was reused for a different command"
        )
    return operation


def get_file_preparation_operation(
    *,
    request_id: str,
    session_id: str,
    connect_documents: Callable[[], sqlite3.Connection] = open_current_connection,
) -> FilePreparationOperation | None:
    """按 Session 所有权读取请求；共享作业 ID 本身不构成访问权限。"""

    initialize_current()
    with connect_documents() as conn:
        request = SqliteFilePreparationRequestStore().get(conn, request_id)
        if request is None or request.session_id != session_id:
            return None
        return _require_operation(conn, request)


def list_file_preparation_operations(
    *,
    session_id: str,
    limit: int,
    delivery_statuses: Sequence[FilePreparationDeliveryStatus] | None = None,
    pending_job_statuses: Sequence[DocumentIngestJobStatus] | None = None,
    connect_documents: Callable[[], sqlite3.Connection] = open_current_connection,
) -> tuple[FilePreparationOperation, ...]:
    """列出一个 Session 的请求，并在同一只读快照中解析共享作业。"""

    initialize_current()
    with connect_documents() as conn:
        requests = SqliteFilePreparationRequestStore().list_for_session(
            conn,
            session_id,
            delivery_statuses=delivery_statuses,
            pending_job_statuses=pending_job_statuses,
            limit=limit,
        )
        return tuple(_require_operation(conn, request) for request in requests)


def retry_file_preparation_operation(
    *,
    request_id: str,
    session_id: str,
    expected_processing_job_id: str,
    requeue_linked_outbox_events: RequeueLinkedOutboxEvents,
    connect_documents: Callable[[], sqlite3.Connection] = open_current_connection,
) -> FilePreparationRetryResult | None:
    """原子恢复已验证请求及其必要的共享终态工作。

    调用方必须先重新验证当前来源 authority、字节、处理配方与目标 generation，
    并把验证得到的精确共享作业 ID 传入。此函数在写事务内再次核对不可变绑定。
    """

    if not callable(requeue_linked_outbox_events):
        raise TypeError("requeue_linked_outbox_events must be callable")
    initialize_current()
    request_store = SqliteFilePreparationRequestStore()
    job_store = SqliteDocumentIngestJobStore()
    now = datetime.now(timezone.utc).isoformat()
    with connect_documents() as conn:
        conn.execute("BEGIN IMMEDIATE")
        try:
            request = request_store.get(conn, request_id)
            if request is None or request.session_id != session_id:
                conn.commit()
                return None
            if request.job_id != expected_processing_job_id:
                raise FilePreparationRequestIntegrityError(
                    "file preparation request changed after retry validation"
                )
            job = job_store.get(conn, request.job_id)
            if job is None:
                raise FilePreparationRequestIntegrityError(
                    "file preparation processing job is missing"
                )
            if request.delivery_status is FilePreparationDeliveryStatus.MOUNTED:
                raise FilePreparationRequestNotRetryable(
                    "mounted file preparation request cannot be retried"
                )

            replayed = True
            if job.status is DocumentIngestJobStatus.TERMINAL_FAILED:
                event_ids = job_store.list_outbox_event_ids(conn, job.job_id)
                requeue_linked_outbox_events(conn, event_ids, now)
                job = job_store.retry_terminal_failure_in_transaction(
                    conn,
                    job_id=job.job_id,
                    now=now,
                )
                replayed = False
            if request.delivery_status is FilePreparationDeliveryStatus.BLOCKED:
                request = request_store.retry_blocked_in_transaction(
                    conn,
                    request_id,
                    now=now,
                )
                replayed = False
        except BaseException:
            conn.rollback()
            raise
        else:
            conn.commit()
    return FilePreparationRetryResult(
        operation=FilePreparationOperation(request, job),
        replayed=replayed,
    )


def _load_operation(
    conn: sqlite3.Connection,
    request_id: str,
) -> FilePreparationOperation | None:
    request = SqliteFilePreparationRequestStore().get(conn, request_id)
    return None if request is None else _require_operation(conn, request)


def _require_operation(
    conn: sqlite3.Connection,
    request: FilePreparationRequest,
) -> FilePreparationOperation:
    job = SqliteDocumentIngestJobStore().get(conn, request.job_id)
    if job is None:
        raise FilePreparationRequestIntegrityError(
            "file preparation processing job is missing"
        )
    return FilePreparationOperation(request=request, processing_job=job)


__all__ = [
    "FilePreparationOperation",
    "FilePreparationRequestCollision",
    "FilePreparationRequestIntegrityError",
    "FilePreparationRequestNotRetryable",
    "FilePreparationRetryResult",
    "file_preparation_retry_rejection",
    "find_file_preparation_replay",
    "get_file_preparation_operation",
    "list_file_preparation_operations",
    "retry_file_preparation_operation",
]
