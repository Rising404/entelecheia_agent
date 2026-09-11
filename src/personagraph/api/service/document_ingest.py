"""持久文档摄取 job 的 HTTP 服务。

本模块把路径 authority 与持久摄取应用服务失败转换为稳定 :class:`ApiError` 值，
但自身不解析文档。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .. import workspace_document_ingest as document_ingest_service
from .. import workspace_documents as document_service
from .. import workspace_files
from .common import (
    is_safe_id,
    optional_int,
    required_str,
    require_workspace_session,
)
from .errors import ApiError

__all__ = [
    "enqueue_document_ingest_job",
    "get_document_ingest_job",
    "list_document_ingest_jobs",
    "retry_document_ingest_job",
]


def _ingest_failure_hint(reason: str | None) -> str:
    """将摄取原因转换为用户可操作提示。"""

    value = str(reason or "")
    if value == "not_a_file":
        return "找不到该文件，请检查路径是否正确、文件是否存在。"
    if value == "unsupported_format":
        return "暂不支持该文件格式（支持 txt/md/pdf/docx/pptx/csv 等文本类文档）。"
    if value.startswith("read_error"):
        return "读取文件失败，请确认文件未损坏且有读取权限。"
    if value.startswith("denied"):
        return "该路径被安全策略拒绝（敏感文件名/系统目录/密钥文件不可收录）。"
    if value == "source_changed_during_ingest":
        return "文件在收录过程中发生变化；请等待写入完成后重新收录。"
    return "收录失败，请检查文件路径与格式。"


_PUBLIC_DOCUMENT_INGEST_STATUS = {
    "pending": "queued",
    "processing": "running",
    "applied": "succeeded",
    "retryable_failed": "failed",
    "terminal_failed": "failed",
}
_DOCUMENT_INGEST_STATUS_FILTERS = {
    "queued": ("queued",),
    "running": ("running",),
    "succeeded": ("succeeded",),
    "failed": ("failed",),
    # 运维客户端可以使用内部拼写，但过滤和响应都立即归一化为公开状态机。
    "pending": ("queued",),
    "processing": ("running",),
    "applied": ("succeeded",),
    "retryable_failed": ("failed",),
    "terminal_failed": ("failed",),
}


def enqueue_document_ingest_job(payload: dict[str, Any]) -> dict[str, Any]:
    """接受一个持久操作，但不内联解析文档。"""

    job_id = required_str(payload, "job_id")
    if len(job_id) > 128 or not is_safe_id(job_id):
        raise ApiError(
            "INVALID_DOCUMENT_INGEST_JOB_ID",
            "job_id 只能包含字母、数字、下划线和连字符，且不能超过 128 个字符",
            status=400,
            details={"field": "job_id"},
        )
    requested_path = required_str(payload, "path")
    session_id = required_str(payload, "session_id")
    require_workspace_session(session_id)

    try:
        canonical_path = workspace_files.resolve_session_path(
            session_id,
            requested_path,
        )
    except workspace_files.WorkspaceFileError as exc:
        raise ApiError(
            exc.code,
            exc.message,
            status=exc.status,
            details={"path": requested_path},
        ) from exc
    with_summary_value = payload.get("with_summary")
    with_summary = True if with_summary_value is None else bool(with_summary_value)
    result = document_ingest_service.enqueue_document_ingest_job(
        job_id=job_id,
        path=str(canonical_path),
        session_id=session_id,
        with_summary=with_summary,
    )
    if not result.get("ok"):
        reason = str(result.get("reason") or "ingest_rejected")
        if reason == "job_id_collision":
            raise ApiError(
                "DOCUMENT_INGEST_JOB_ID_COLLISION",
                "job_id 已绑定到另一份收录请求",
                status=409,
                details={"job_id": job_id},
            )
        details: dict[str, Any] = {
            "reason": reason,
            "hint": _ingest_failure_hint(reason),
        }
        if result.get("supported") is not None:
            details["supported"] = result["supported"]
        raise ApiError(
            "DOCUMENT_INGEST_JOB_REJECTED",
            "文档收录任务未被接受",
            status=400,
            details=details,
        )
    operation = result["operation"]
    return {
        "accepted": True,
        "replayed": bool(result.get("replayed")),
        "job": _document_ingest_job_view(operation),
    }


def list_document_ingest_jobs(
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    params = params or {}
    session_id = required_str(params, "session_id")
    require_workspace_session(session_id)
    statuses = _document_ingest_status_filter(params.get("status"))
    limit = optional_int(params.get("limit"), 50)
    if limit is None or not 1 <= limit <= 200:
        raise ApiError(
            "INVALID_DOCUMENT_INGEST_JOB_LIMIT",
            "limit 必须在 1 到 200 之间",
            status=400,
            details={"limit": params.get("limit")},
        )
    operations = document_ingest_service.list_document_ingest_jobs(
        session_id=session_id,
        statuses=statuses,
        limit=limit,
    )
    return {
        "jobs": [_document_ingest_job_view(operation) for operation in operations]
    }


def get_document_ingest_job(
    job_id: str,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    session_id = required_str(params or {}, "session_id")
    require_workspace_session(session_id)
    operation = document_ingest_service.get_document_ingest_job(
        request_id=job_id,
        session_id=session_id,
    )
    if operation is None:
        raise ApiError(
            "DOCUMENT_INGEST_JOB_NOT_FOUND",
            "文档收录任务不存在",
            status=404,
            details={"job_id": job_id},
        )
    return {"job": _document_ingest_job_view(operation)}


def retry_document_ingest_job(
    job_id: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    session_id = required_str(payload, "session_id")
    require_workspace_session(session_id)
    current = document_ingest_service.get_document_ingest_job(
        request_id=job_id,
        session_id=session_id,
    )
    if current is None:
        raise ApiError(
            "DOCUMENT_INGEST_JOB_NOT_FOUND",
            "文档收录任务不存在",
            status=404,
            details={"job_id": job_id},
        )
    result = document_ingest_service.retry_document_ingest_job(
        request_id=job_id,
        session_id=session_id,
    )
    if not result.get("ok"):
        reason = str(result.get("reason") or "not_retryable")
        if reason == "job_not_found":
            raise ApiError(
                "DOCUMENT_INGEST_JOB_NOT_FOUND",
                "文档收录任务不存在",
                status=404,
                details={"job_id": job_id},
            )
        raise ApiError(
            "DOCUMENT_INGEST_JOB_NOT_RETRYABLE",
            "该文档收录任务当前不能重试",
            status=409,
            details={"job_id": job_id, "status": _public_job_status(current)},
        )
    operation = result["operation"]
    return {
        "accepted": True,
        "replayed": bool(result.get("replayed")),
        "job": _document_ingest_job_view(operation),
    }


def _document_ingest_status_filter(value: Any) -> tuple[str, ...] | None:
    raw = str(value or "").strip().lower()
    if not raw or raw == "all":
        return None
    requested = tuple(part.strip() for part in raw.split(",") if part.strip())
    if not requested or "all" in requested:
        raise ApiError(
            "INVALID_DOCUMENT_INGEST_JOB_STATUS",
            "status 取值非法",
            status=400,
            details={"status": value},
        )
    statuses: list[str] = []
    for item in requested:
        mapped = _DOCUMENT_INGEST_STATUS_FILTERS.get(item)
        if mapped is None:
            raise ApiError(
                "INVALID_DOCUMENT_INGEST_JOB_STATUS",
                "status 取值非法",
                status=400,
                details={"status": value},
            )
        for status in mapped:
            if status not in statuses:
                statuses.append(status)
    return tuple(statuses)


def _public_job_status(operation: Any) -> str:
    request_status = getattr(
        operation.request.delivery_status,
        "value",
        str(operation.request.delivery_status),
    )
    if request_status == "mounted":
        return "succeeded"
    if request_status == "blocked":
        return "failed"
    status = getattr(
        operation.processing_job.status,
        "value",
        str(operation.processing_job.status),
    )
    # 已应用但尚未完成当前 Session 的挂载仍属于运行中，不能提前发布成功。
    if status == "applied":
        return "running"
    return _PUBLIC_DOCUMENT_INGEST_STATUS.get(status, "failed")


def _document_ingest_job_view(operation: Any) -> dict[str, Any]:
    request = operation.request
    job = operation.processing_job
    public_status = _public_job_status(operation)
    durable_status = getattr(job.status, "value", str(job.status))
    delivered = getattr(
        request.delivery_status,
        "value",
        str(request.delivery_status),
    ) == "mounted"
    document = (
        document_service.get_document(job.document_id)
        if delivered and job.document_id is not None
        else None
    )
    return {
        # 公开 job_id 保持客户端请求幂等键语义；共享作业身份始终单列。
        "job_id": request.request_id,
        "processing_job_id": job.job_id,
        "session_id": request.session_id,
        "path": _document_ingest_job_path(operation),
        "with_summary": request.with_summary,
        "status": public_status,
        "stage": getattr(job.stage, "value", str(job.stage)),
        "can_retry": public_status == "failed" and (
            durable_status == "terminal_failed"
            or getattr(request.delivery_status, "value", "") == "blocked"
        ),
        "error": _document_ingest_job_error(operation, durable_status),
        "attempts": job.attempts,
        "next_retry_at": job.next_retry_at,
        "reason_code": _safe_document_ingest_reason(
            request.reason_code or job.reason_code
        )
        if public_status == "failed"
        else None,
        "document_id": job.document_id if delivered else None,
        "document_version_id": job.document_version_id if delivered else None,
        "retrieval_data_version": job.retrieval_data_version if delivered else None,
        "processing_status": (document or {}).get("processing_status"),
        "diagnostics": (document or {}).get("diagnostics"),
        "needs_vision": (document or {}).get("needs_vision"),
        "created_at": request.created_at,
        "updated_at": request.updated_at,
        "completed_at": request.updated_at if public_status in {"succeeded", "failed"} else None,
    }


def _document_ingest_job_error(
    operation: Any,
    durable_status: str,
) -> dict[str, Any] | None:
    request = operation.request
    job = operation.processing_job
    request_blocked = getattr(request.delivery_status, "value", "") == "blocked"
    if not request_blocked and durable_status not in {
        "retryable_failed",
        "terminal_failed",
    }:
        return None
    code = _safe_document_ingest_reason(
        request.reason_code or job.reason_code
    ) or "document_ingest_failed"
    message = _document_ingest_error_message(code)
    if code == "retrieval_method_unavailable":
        hint = (
            "请恢复该任务冻结的同一检索配置后手动重试。若索引配置已经升级，"
            "需要显式完整重建 DataVersion；当前界面尚不支持这项操作。"
        )
    elif request_blocked:
        hint = "请重新确认当前文件仍受该会话授权且内容未变化，然后手动重试此请求。"
    elif durable_status == "retryable_failed":
        hint = "这是暂时性失败，系统将按 next_retry_at 自动重试；无需重复提交任务。"
    else:
        hint = "请检查源文件和本地索引状态后手动重试；若文件已变更，请新建收录任务。"
    return {
        "code": code,
        "message": message[:160],
        "hint": hint[:240],
        "retryable": True,
    }


def _document_ingest_error_message(code: str) -> str:
    if code == "frozen_source_mismatch":
        return "源文件在任务创建后发生了变化"
    if code in {
        "frozen_processor_mismatch",
        "frozen_chunk_recipe_mismatch",
        "prepared_source_path_mismatch",
    }:
        return "任务冻结的文档处理配置已发生变化"
    if code.startswith("source_revalidation"):
        return "暂时无法重新验证源文件"
    if code == "retrieval_method_unavailable":
        return "本地论文索引所需的检索依赖或模型资产不可用"
    if "outbox" in code or "coverage" in code:
        return "文档索引覆盖尚未完成"
    if "generation" in code or "retrieval_target" in code:
        return "当前检索索引版本无法接收该文档"
    if code == "document_worker_internal_error":
        return "文档处理服务暂时不可用"
    return "文档收录未完成"


def _safe_document_ingest_reason(value: Any) -> str | None:
    """公开有界机器代码，绝不公开异常文本或文件系统数据。"""

    if value is None:
        return None
    reason = str(value)
    if (
        len(reason) <= 128
        and reason
        and all(char.isalnum() or char in {"_", "-", ".", ":"} for char in reason)
    ):
        return reason
    return "document_ingest_failed"


def _document_ingest_job_path(operation: Any) -> str:
    """返回相对 Session 的显示路径，绝不返回 authority 路径。"""

    from ...session import store as session_store

    canonical = Path(operation.processing_job.canonical_path)
    session = session_store.get_session(operation.request.session_id)
    working_dir = (session or {}).get("working_dir")
    if working_dir:
        try:
            return canonical.relative_to(
                Path(str(working_dir)).expanduser().resolve()
            ).as_posix()
        except ValueError:
            pass
    return canonical.name
