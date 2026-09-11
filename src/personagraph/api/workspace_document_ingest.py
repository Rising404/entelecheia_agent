"""持久文档摄取的应用组合与 Session 请求边界。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

__all__ = [
    "enqueue_document_ingest_job",
    "get_document_ingest_job",
    "list_document_ingest_jobs",
    "retry_document_ingest_job",
]


def _workspace_ingest_path_rejection(
    *,
    session_id: str | None,
    canonical_path: Path,
) -> str | None:
    """对 Host 私有 workspace 源返回稳定拒绝。"""

    if not isinstance(session_id, str) or not session_id:
        return None

    from ..session import store as session_store
    from ..workspace.binding import (
        ReservedWorkspacePathError,
        is_reserved_workspace_path,
    )

    session = session_store.get_session(session_id)
    working_dir = (session or {}).get("working_dir")
    if not isinstance(working_dir, str) or not working_dir.strip():
        return None
    try:
        if is_reserved_workspace_path(working_dir, canonical_path):
            return "agent_private_path"
    except ReservedWorkspacePathError as exc:
        # 托管附件可位于 workspace 之外并由自身 receipt 授权；其他无法安全分类的路径
        # 则必须失败关闭。
        if exc.code == "outside_bound_root":
            return None
        return "workspace_path_unsafe"
    return None


def enqueue_document_ingest_job(
    *,
    job_id: str,
    path: str,
    session_id: str,
    with_summary: bool,
) -> dict[str, Any]:
    """接受一个 Session 请求，并让共享处理作业在后台执行。

    HTTP 的 ``job_id`` 是客户端请求幂等键；底层内容寻址的共享作业拥有独立 ID。
    精确重放先读取已冻结请求，因此不会因响应丢失期间源文件发生变化而改写命令。
    """

    from ..workspace.ingestion.request_management import (
        FilePreparationRequestCollision,
        find_file_preparation_replay,
        get_file_preparation_operation,
    )

    canonical = Path(path).expanduser().resolve()
    if rejected := _workspace_ingest_path_rejection(
        session_id=session_id,
        canonical_path=canonical,
    ):
        return {"ok": False, "reason": rejected}
    try:
        replay = find_file_preparation_replay(
            request_id=job_id,
            session_id=session_id,
            canonical_path=str(canonical),
            with_summary=with_summary,
        )
    except FilePreparationRequestCollision:
        return {"ok": False, "reason": "job_id_collision"}
    if replay is not None:
        return {"ok": True, "operation": replay, "replayed": True}

    rejected = _new_ingest_source_rejection(canonical)
    if rejected is not None:
        return rejected

    from ..input_processing.files import (
        SourceChangedDuringReadError,
        SourceSizeLimitError,
        fingerprint_file,
    )

    try:
        frozen = fingerprint_file(canonical)
    except SourceSizeLimitError:
        return {"ok": False, "reason": "too_large"}
    except SourceChangedDuringReadError:
        return {"ok": False, "reason": "source_changed_during_ingest"}
    except OSError as exc:
        return {"ok": False, "reason": f"read_error:{type(exc).__name__}"}

    from ..retrieval.profile import durable_chunking_profile
    from ..session.workspace_authority import (
        validate_current_session_workspace_authority,
    )
    from ..workspace.ingestion.composition import resolve_document_ingest_owner
    from ..workspace.ingestion.preparation import prepare_file

    prepared = prepare_file(
        session_id=session_id,
        canonical_path=str(canonical),
        frozen_fingerprint=frozen,
        request_id=job_id,
        with_summary=with_summary,
        ingest_owner=resolve_document_ingest_owner(),
        chunking_profile=durable_chunking_profile(),
        validate_source_authority=validate_current_session_workspace_authority,
        synchronous_max_bytes=0,
        pending_wait_seconds=0,
    )
    if prepared.request_id is None:
        return {
            "ok": False,
            "reason": _public_enqueue_rejection(prepared.reason_code),
        }
    operation = get_file_preparation_operation(
        request_id=prepared.request_id,
        session_id=session_id,
    )
    if operation is None:
        return {"ok": False, "reason": "file_readiness_operation_missing"}
    return {
        "ok": True,
        "operation": operation,
        # 新请求复用共享处理结果不等于客户端请求重放。
        "replayed": False,
    }


def _new_ingest_source_rejection(canonical: Path) -> dict[str, Any] | None:
    from ..configuration.paths import deny_reason
    from ..input_processing.documents.readers import (
        configured_processor_fingerprint,
        supported_suffixes,
    )

    denied = deny_reason(canonical)
    if denied:
        return {"ok": False, "reason": denied}
    if not canonical.is_file():
        return {"ok": False, "reason": "not_a_file"}
    if configured_processor_fingerprint(canonical) is None:
        return {
            "ok": False,
            "reason": "unsupported_format",
            "supported": supported_suffixes(),
        }
    return None


def _public_enqueue_rejection(reason: str | None) -> str:
    return {
        "file_too_large": "too_large",
        "unsupported_file_format": "unsupported_format",
        "source_changed_during_read": "source_changed_during_ingest",
        "source_unavailable": "read_error:OSError",
    }.get(str(reason or ""), str(reason or "ingest_rejected"))


def list_document_ingest_jobs(
    *,
    session_id: str,
    limit: int,
    statuses: tuple[str, ...] | None = None,
) -> tuple[Any, ...]:
    from ..workspace.ingestion.request_management import (
        list_file_preparation_operations,
    )
    from ..workspace.ingestion.storage import (
        DocumentIngestJobStatus,
        FilePreparationDeliveryStatus,
    )

    delivery_statuses: list[FilePreparationDeliveryStatus] | None = None
    pending_job_statuses: list[DocumentIngestJobStatus] | None = None
    if statuses is not None:
        delivery_statuses = []
        pending_job_statuses = []
        for status in statuses:
            if status == "queued":
                pending_job_statuses.append(DocumentIngestJobStatus.PENDING)
            elif status == "running":
                pending_job_statuses.extend((
                    DocumentIngestJobStatus.PROCESSING,
                    DocumentIngestJobStatus.APPLIED,
                ))
            elif status == "succeeded":
                delivery_statuses.append(FilePreparationDeliveryStatus.MOUNTED)
            elif status == "failed":
                delivery_statuses.append(FilePreparationDeliveryStatus.BLOCKED)
                pending_job_statuses.extend((
                    DocumentIngestJobStatus.RETRYABLE_FAILED,
                    DocumentIngestJobStatus.TERMINAL_FAILED,
                ))
            else:
                raise ValueError(f"unsupported public document ingest status: {status}")

    return list_file_preparation_operations(
        session_id=session_id,
        limit=limit,
        delivery_statuses=delivery_statuses,
        pending_job_statuses=pending_job_statuses,
    )


def get_document_ingest_job(
    *,
    request_id: str,
    session_id: str,
) -> Any | None:
    from ..workspace.ingestion.request_management import (
        get_file_preparation_operation,
    )

    return get_file_preparation_operation(
        request_id=request_id,
        session_id=session_id,
    )


def retry_document_ingest_job(
    *,
    request_id: str,
    session_id: str,
) -> dict[str, Any]:
    """重新验证当前来源后，显式恢复请求及必要的共享终态工作。"""

    from ..workspace.ingestion.request_management import (
        FilePreparationRequestNotRetryable,
        file_preparation_retry_rejection,
        get_file_preparation_operation,
        retry_file_preparation_operation,
    )
    from ..workspace.ingestion.composition import resolve_document_ingest_owner

    operation = get_file_preparation_operation(
        request_id=request_id,
        session_id=session_id,
    )
    if operation is None:
        return {"ok": False, "reason": "job_not_found"}
    if getattr(operation.request.delivery_status, "value", "") == "mounted":
        return {
            "ok": False,
            "reason": "request_already_mounted",
            "operation": operation,
        }
    owner = resolve_document_ingest_owner()
    from ..retrieval.profile import durable_chunking_profile
    from ..session.workspace_authority import (
        validate_current_session_workspace_authority,
    )

    if rejected := file_preparation_retry_rejection(
        operation,
        generation_identity=owner.generation_identity,
        chunking_profile=durable_chunking_profile(),
        validate_source_authority=validate_current_session_workspace_authority,
    ):
        return {"ok": False, "reason": rejected, "operation": operation}
    try:
        retried = retry_file_preparation_operation(
            request_id=request_id,
            session_id=session_id,
            expected_processing_job_id=operation.processing_job.job_id,
            requeue_linked_outbox_events=_requeue_linked_outbox_events,
        )
    except FilePreparationRequestNotRetryable:
        return {
            "ok": False,
            "reason": "request_already_mounted",
            "operation": operation,
        }
    if retried is None:
        return {"ok": False, "reason": "job_not_found"}

    owner.wake()
    return {
        "ok": True,
        "operation": retried.operation,
        "replayed": retried.replayed,
    }


def _requeue_linked_outbox_events(
    conn: Any,
    event_ids: tuple[str, ...],
    now: str,
) -> None:
    """Retrieval adapter：只在调用方持有的同一事务中恢复死信事件。"""

    from ..retrieval.lifecycle.outbox import OutboxStatus, SqliteRetrievalOutbox

    outbox = SqliteRetrievalOutbox()
    for event_id in event_ids:
        outcome = outbox.get_outcome_in_transaction(conn, event_id)
        if outcome is None or outcome[0] is not OutboxStatus.TERMINAL_FAILED:
            continue
        outbox.requeue_terminal_failure_in_transaction(
            conn,
            event_id=event_id,
            actor="document_ingest_job_api",
            reason="manual_document_ingest_job_retry",
            now=now,
        )
