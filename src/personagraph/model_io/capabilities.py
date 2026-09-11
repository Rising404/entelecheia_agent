from __future__ import annotations

from datetime import datetime, timezone
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse, urlunparse

from ..configuration.paths import STATE_DIR
from .dialects import resolve_request_dialect


ControlTransport = Literal["native", "prompt_json"]
ConfiguredTransport = Literal["auto", "native", "prompt_json"]
CAPABILITY_RECORD_PATH = STATE_DIR / "model_capabilities.json"
_SAFE_EVIDENCE_KEYS = {
    "request_sent",
    "observed_kind",
    "provider_call_id",
    "argument_round_trip",
    "error_code",
    "status_code",
    "exception_type",
    "retryable",
}


def resolve_control_transport(
    configured: str | None,
    *,
    provider: str,
    base_url: str,
    model: str = "",
    request_dialect: str = "auto",
) -> tuple[ControlTransport, str]:
    mode = str(configured or "auto").strip().lower()
    if mode not in {"auto", "native", "prompt_json"}:
        raise ValueError("model_control_transport must be auto, native, or prompt_json")
    if mode == "prompt_json":
        return "prompt_json", "explicit_prompt_json"
    if mode == "native":
        return "native", "explicit_native"
    if native_probe_record(
        provider=provider,
        base_url=base_url,
        model=model,
        request_dialect=request_dialect,
    ):
        return "native", "verified_native_probe"
    return "prompt_json", "native_probe_not_verified"


def model_control_capability_summary(
    configured: str | None,
    *,
    provider: str,
    base_url: str,
    model: str,
    request_dialect: str = "auto",
) -> dict[str, Any]:
    try:
        record = native_probe_record(
            provider=provider,
            base_url=base_url,
            model=model,
            request_dialect=request_dialect,
        )
        selected, reason = resolve_control_transport(
            configured,
            provider=provider,
            base_url=base_url,
            model=model,
            request_dialect=request_dialect,
        )
        concrete_dialect = (
            record.get("request_dialect")
            if record
            else _resolved_dialect(provider, base_url, request_dialect)
        )
        host = _safe_endpoint_host(base_url)
    except (TypeError, ValueError):
    # 状态和 prompt 构建属于诊断/读取路径。环境变量或手工配置中的错误值不能使任一路径崩溃，
    # 且在缺少有效端点身份时，绝不能保留显式请求的原生传输。
        return {
            "configured": str(configured or "auto"),
            "effective": "prompt_json",
            "reason": "invalid_configuration",
            "provider": str(provider or ""),
            "model": str(model or ""),
            "request_dialect": str(request_dialect or "auto"),
            "endpoint_host": _safe_endpoint_host(base_url),
            "native_probe_verified": False,
            "native_probe_verified_at": None,
        }
    return {
        "configured": str(configured or "auto"),
        "effective": selected,
        "reason": reason,
        "provider": provider,
        "model": model,
        "request_dialect": concrete_dialect,
        "endpoint_host": host,
        "native_probe_verified": bool(record),
        "native_probe_verified_at": record.get("verified_at") if record else None,
    }


def _safe_endpoint_host(base_url: object) -> str:
    try:
        return urlparse(str(base_url or "")).hostname or ""
    except (TypeError, ValueError):
        return ""


def native_probe_record(
    *,
    provider: str,
    base_url: str,
    model: str,
    request_dialect: str = "auto",
) -> dict[str, Any] | None:
    identity = _identity(provider, base_url, model, request_dialect)
    for record in _load_records():
        if not isinstance(record, dict):
            continue
        if record.get("identity") == identity and record.get("capability") == "native_tool_use" and record.get("passed") is True:
            return dict(record)
    return None


def record_native_probe(
    *,
    provider: str,
    base_url: str,
    model: str,
    request_dialect: str = "auto",
    passed: bool,
    evidence: dict[str, Any],
) -> dict[str, Any]:
    """持久化安全且特定于模型/端点的探测结果；绝不存储凭据或原始响应文本。"""
    normalized_provider = _normalized_provider(provider)
    concrete_dialect = _resolved_dialect(
        normalized_provider, base_url, request_dialect
    )
    identity = _identity(
        normalized_provider, base_url, model, concrete_dialect
    )
    record = {
        "schema_version": 2,
        "identity": identity,
        "provider": normalized_provider,
        "request_dialect": concrete_dialect,
        "endpoint_host": urlparse(base_url).hostname or "",
        "model": model,
        "capability": "native_tool_use",
        "passed": bool(passed),
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "evidence": {
            key: value
            for key, value in evidence.items()
            if key in _SAFE_EVIDENCE_KEYS and isinstance(value, (str, int, float, bool, type(None)))
        },
    }
    records = [item for item in _load_records() if isinstance(item, dict) and item.get("identity") != identity]
    records.append(record)
    _atomic_write({"schema_version": 2, "records": records})
    return record


def _load_records() -> list[dict[str, Any]]:
    if not CAPABILITY_RECORD_PATH.exists():
        return []
    try:
        data = json.loads(CAPABILITY_RECORD_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    records = data.get("records") if isinstance(data, dict) else None
    return records if isinstance(records, list) else []


def _atomic_write(payload: dict[str, Any]) -> None:
    CAPABILITY_RECORD_PATH.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=str(CAPABILITY_RECORD_PATH.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        os.replace(temporary, CAPABILITY_RECORD_PATH)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _normalized_provider(provider: object) -> str:
    value = str(provider or "").strip().lower()
    return {
        "deepseek": "anthropic-compatible",
        "anthropic": "anthropic-compatible",
    }.get(value, value)


def _resolved_dialect(provider: str, base_url: str, request_dialect: str) -> str:
    normalized_provider = _normalized_provider(provider)
    return resolve_request_dialect(
        normalized_provider,
        base_url,
        request_dialect,
        legacy_provider=provider,
    ).value


def _identity(
    provider: str,
    base_url: str,
    model: str,
    request_dialect: str = "auto",
) -> str:
    normalized_provider = _normalized_provider(provider)
    concrete_dialect = _resolved_dialect(
        normalized_provider, base_url, request_dialect
    )
    parsed = urlparse(base_url.strip())
    normalized_url = urlunparse((parsed.scheme.lower(), parsed.netloc.lower(), parsed.path.rstrip("/"), "", "", ""))
    return f"{normalized_provider}|{concrete_dialect}|{normalized_url}|{model.strip()}"
