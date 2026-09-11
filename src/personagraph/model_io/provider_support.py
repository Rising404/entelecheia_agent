"""供应商适配器共享的方言校验、错误投影与计时辅助。"""

from __future__ import annotations

import json
import math
import time
from datetime import timezone
from email.utils import parsedate_to_datetime
from typing import Any

from .dialects import resolve_request_dialect
from .gateway_core import ModelGatewayError
from .tier_bindings import ModelTierBinding


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _message_text(content: object) -> str:
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False, default=str)


def _resolved_request_dialect(
    provider: str,
    base_url: str,
    binding: ModelTierBinding | None,
):
    from . import gateway as _gateway

    try:
        if binding is not None:
            return resolve_request_dialect(
                provider,
                base_url,
                binding.request_dialect,
            )
        raw_provider = _gateway.get_setting("provider", provider)
        return resolve_request_dialect(
            provider,
            base_url,
            _gateway.get_setting("request_dialect", "auto"),
            legacy_provider=raw_provider,
        )
    except ValueError as exc:
        raise ModelGatewayError(
            "MODEL_CALL_FAILED",
            "The configured request dialect does not match the model endpoint.",
            retryable=False,
            details={
                "provider": provider,
                "reason": "invalid_request_dialect",
            },
        ) from exc


def _require_endpoint_configuration(
    *, provider: str, base_url: str, model: str
) -> None:
    """在打开 HTTP 连接之前拒绝不完整端点。"""

    if not base_url:
        raise ModelGatewayError(
            "MODEL_CALL_FAILED",
            "A base URL is required for the selected model provider.",
            retryable=False,
            details={"provider": provider, "reason": "missing_base_url"},
        )
    if not model:
        raise ModelGatewayError(
            "MODEL_CALL_FAILED",
            "A model name is required for the selected model provider.",
            retryable=False,
            details={"provider": provider, "reason": "missing_model"},
        )


def _extract_finish_reason(data: dict[str, Any]) -> str | None:
    delta = data.get("delta")
    if isinstance(delta, dict) and delta.get("stop_reason"):
        return str(delta.get("stop_reason"))
    if data.get("stop_reason"):
        return str(data.get("stop_reason"))
    if data.get("finish_reason"):
        return str(data.get("finish_reason"))
    choices = data.get("choices")
    if isinstance(choices, list) and choices:
        reason = choices[0].get("finish_reason") if isinstance(choices[0], dict) else None
        return str(reason) if reason else None
    return None


def _elapsed_ms(start: float) -> int:
    return int((time.perf_counter() - start) * 1000)


def _redact_endpoint(endpoint: str) -> str:
    return endpoint.split("?", 1)[0]


def _safe_provider_error_details(response: object, api_key: object) -> dict[str, Any]:
    """保留可操作的供应商诊断信息，但不保留凭据。"""

    details: dict[str, Any] = {}
    headers = getattr(response, "headers", None)
    if headers is not None:
        for name in ("x-request-id", "request-id", "cf-ray"):
            try:
                value = headers.get(name)
            except Exception:
                value = None
            if isinstance(value, str) and value.strip():
                details["provider_request_id"] = value.strip()[:200]
                break
        retry_after_seconds = _safe_retry_after_seconds(headers)
        if retry_after_seconds is not None:
            details["retry_after_seconds"] = retry_after_seconds
    try:
        payload = response.json()  # type: ignore[union-attr]
    except Exception:
        return details
    error = payload.get("error") if isinstance(payload, dict) else None
    if not isinstance(error, dict):
        return details
    for source, target in (
        ("type", "provider_error_type"),
        ("code", "provider_error_code"),
        ("param", "provider_error_param"),
    ):
        value = error.get(source)
        if isinstance(value, (str, int, float, bool)):
            details[target] = str(value)[:200]
    message = error.get("message")
    if isinstance(message, str) and message.strip():
        safe = message.strip()
        secret = str(api_key or "")
        if secret:
            safe = safe.replace(secret, "[redacted]")
        details["provider_error_message"] = safe[:500]
    return details


def _safe_retry_after_seconds(headers: object) -> float | None:
    """解析标准 Retry-After 秒数/日期，但不保留响应头。"""

    try:
        raw = headers.get("retry-after")  # type: ignore[union-attr]
    except Exception:
        return None
    if not isinstance(raw, str) or not raw.strip():
        return None
    value = raw.strip()
    try:
        seconds = float(value)
    except ValueError:
        try:
            parsed = parsedate_to_datetime(value)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            seconds = parsed.timestamp() - time.time()
        except (TypeError, ValueError, OverflowError):
            return None
    if not math.isfinite(seconds) or seconds < 0:
        return None
    return min(seconds, 7 * 24 * 60 * 60)
