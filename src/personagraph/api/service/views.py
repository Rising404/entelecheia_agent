from __future__ import annotations

from typing import Any

from personagraph.model_io.gateway import ModelGatewayError
from ...runtime.turn.contracts import EntryTurnResult
from ..runtime_outcome import build_error_outcome
from .errors import ApiError


def session_summary(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": row.get("id"),
        "title": row.get("title"),
        "status": row.get("status"),
        "folder_id": row.get("folder_id"),
        "working_dir": row.get("working_dir"),
        "created_at": row.get("created_at"),
        "last_active_at": row.get("last_active_at"),
    }


def turn_view(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "turn_idx": row.get("turn_idx"),
        "role": row.get("role"),
        "content": row.get("content"),
        "created_at": row.get("created_at"),
    }


def model_api_error(exc: ModelGatewayError) -> ApiError:
    """将抛出的 ModelGatewayError 转成公开 API 错误、HTTP 状态和 outcome。

    Runtime 已正常返回的 incomplete EntryTurnResult 不经过这里；它仍由
    turn_result_view 投影。区分“API 调用抛错”与“已接受 Turn 中断”，避免混淆重试入口。
    """

    details = dict(exc.details)
    outcome = build_error_outcome(
        code=exc.code,
        domain="model",
        message=exc.message,
        retryable=exc.retryable,
        details=details,
    ).to_dict()
    status = model_error_http_status(exc)
    return ApiError(
        exc.code,
        model_error_message(exc.code),
        status=status,
        details=details,
        outcome=outcome,
    )

def model_error_http_status(exc: ModelGatewayError) -> int:
    if exc.code == "MODEL_CALL_TIMEOUT":
        return 504
    if exc.code == "MODEL_BAD_RESPONSE":
        return 502
    status_code = exc.details.get("status_code")
    if isinstance(status_code, int):
        if status_code in {401, 403}:
            return 502
        if status_code == 429 or status_code >= 500:
            return 503
        return 502
    if not exc.retryable and exc.details.get("physical_error_retryable") is not True:
        return 400
    return 503

def model_error_message(code: str) -> str:
    if code == "MODEL_CALL_TIMEOUT":
        return "模型调用超时，可在确认网络后重试本轮"
    if code == "MODEL_BAD_RESPONSE":
        return "模型返回结构异常"
    return "模型调用失败"

def turn_result_view(result: EntryTurnResult) -> dict[str, Any]:
    """将 EntryTurnResult 转成 JSON / SSE final 共用的公开结果，不重新判断执行是否成功。

    status 表达 Turn 结果，window_state 表达当前执行槽位；completed 与
    post_commit_pending 可以同时成立。reply 来自 Entry 的权威投影，不能在这里
    用候选或最后一条模型文本补出一份正式回答。
    """

    return {
        "session_id": result.session_id,
        "turn_id": result.turn_id,
        "status": result.status,
        "processing_level": result.processing_level,
        "reply": result.reply,
        "end_reason": result.end_reason,
        "error_code": result.error_code,
        "window_state": result.window_state,
        "window_revision": result.window_revision,
        "available_controls": list(result.available_controls),
    }
