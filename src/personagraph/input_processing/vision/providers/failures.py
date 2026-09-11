"""视觉 HTTP 边界的纯失败分类；只输出安全事实，不重试或读取外部状态。"""

from __future__ import annotations

import json
import math
import re
import urllib.error
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

from ..contracts import VisionFailureDiagnostics


def parse_retry_after(value: object, *, now: datetime) -> float | None:
    """只保留 Retry-After 的秒数；非法、负数或非有限值不进入记录。"""

    if not isinstance(value, str) or not value.strip() or len(value) > 1024:
        return None
    try:
        seconds = float(value)
    except ValueError:
        try:
            deadline = parsedate_to_datetime(value)
            if deadline.tzinfo is None:
                deadline = deadline.replace(tzinfo=timezone.utc)
            seconds = max(0.0, (deadline - now).total_seconds())
        except (TypeError, ValueError, OverflowError):
            return None
    return seconds if math.isfinite(seconds) and seconds >= 0 else None


def failure_diagnostics(
    *,
    phase: str,
    elapsed_ms: int,
    timeout_s: float,
    completion_uncertain: bool,
    error: BaseException | None = None,
    http_status: int | None = None,
    retry_after_s: float | None = None,
) -> VisionFailureDiagnostics:
    cause = _exception_cause(error) if error is not None else None
    return VisionFailureDiagnostics(
        phase=phase,
        elapsed_ms=elapsed_ms,
        timeout_s=timeout_s,
        completion_uncertain=completion_uncertain,
        exception_type=_safe_class_name(error),
        cause_type=_safe_class_name(cause),
        http_status=http_status,
        retry_after_s=retry_after_s,
    )


def classify_request_failure(
    error: Exception, *, elapsed_ms: int, timeout_s: float, now: datetime
) -> tuple[str, VisionFailureDiagnostics]:
    phase = "request"
    uncertain = False
    status = None
    retry_after = None
    # HTTPError 是 URLError 子类，必须先识别已收到的明确 HTTP 失败响应。
    if isinstance(error, urllib.error.HTTPError):
        status = error.code
        retry_after = parse_retry_after(
            error.headers.get("Retry-After") if error.headers is not None else None,
            now=now,
        )
        if status in {401, 403}:
            code = "vision_http_authentication_failed"
        elif status == 429:
            code = "vision_http_rate_limited"
        elif 500 <= status <= 599:
            code = "vision_http_server_error"
        else:
            code = "vision_http_request_rejected"
    elif isinstance(error, (json.JSONDecodeError, UnicodeError)):
        code = "vision_response_invalid"
        phase = "response_decode"
    elif isinstance(error, (urllib.error.URLError, TimeoutError, OSError)):
        uncertain = True
        code = (
            "vision_request_timeout"
            if isinstance(error, TimeoutError)
            or isinstance(_exception_cause(error), TimeoutError)
            else "vision_connection_failed"
        )
    else:
        # urlopen 对无效 URL 等本地请求问题也会抛 ValueError；不能伪称收到了响应。
        code = "vision_request_invalid"
    return code, failure_diagnostics(
        phase=phase,
        elapsed_ms=elapsed_ms,
        timeout_s=timeout_s,
        completion_uncertain=uncertain,
        error=error,
        http_status=status,
        retry_after_s=retry_after,
    )


def _exception_cause(error: BaseException) -> BaseException | None:
    if isinstance(error, urllib.error.URLError) and isinstance(error.reason, BaseException):
        return error.reason
    return error.__cause__


def _safe_class_name(error: BaseException | None) -> str | None:
    if error is None:
        return None
    name = type(error).__name__
    return name if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,127}", name) else None
