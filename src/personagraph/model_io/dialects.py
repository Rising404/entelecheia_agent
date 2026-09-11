"""叠加在两种线上协议之上的供应商专用请求控制。

``anthropic-compatible`` 与 ``openai-compatible`` 描述信封和响应解析器，但并不说明端点
接受哪些推理字段或 token 上限键。例如，DeepSeek 的 OpenAI 形状端点仍使用
``max_tokens``，而原生 OpenAI 推理模型使用 ``max_completion_tokens``。把两者视为同一
契约，会产生 JSON 有效、但对所选供应商无效的请求。

本模块被刻意保持为纯函数且只依赖 stdlib。HTTP 网关和持久端点指纹使用同一投影，因此
重试所描述的控制永远不会与实际线上发送的控制不同。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any
from urllib.parse import urlsplit


class RequestDialect(StrEnum):
    AUTO = "auto"
    DEEPSEEK_ANTHROPIC = "deepseek-anthropic"
    DEEPSEEK_OPENAI = "deepseek-openai"
    OPENAI_NATIVE = "openai-native"
    ANTHROPIC_NATIVE = "anthropic-native"
    GENERIC_OPENAI = "generic-openai"
    GENERIC_ANTHROPIC = "generic-anthropic"
    MOCK = "mock"


PROFILE_DIALECT_CHOICES: tuple[str, ...] = (
    "auto",
    "deepseek",
    "openai",
    "anthropic",
    "generic",
)

_OPENAI_DIALECTS = frozenset(
    {
        RequestDialect.DEEPSEEK_OPENAI,
        RequestDialect.OPENAI_NATIVE,
        RequestDialect.GENERIC_OPENAI,
    }
)
_ANTHROPIC_DIALECTS = frozenset(
    {
        RequestDialect.DEEPSEEK_ANTHROPIC,
        RequestDialect.ANTHROPIC_NATIVE,
        RequestDialect.GENERIC_ANTHROPIC,
    }
)


@dataclass(frozen=True, slots=True)
class RequestControls:
    """添加到一个供应商请求中的精确非消息字段。"""

    dialect: RequestDialect
    payload_fields: dict[str, Any]
    fingerprint: dict[str, Any]


def _hostname(base_url: object) -> str:
    try:
        return (urlsplit(str(base_url or "").strip()).hostname or "").lower()
    except ValueError:
        return ""


def _host_is(host: str, domain: str) -> bool:
    return host == domain or host.endswith("." + domain)


def _known_host_vendor(base_url: object) -> str | None:
    host = _hostname(base_url)
    for vendor, domain in (
        ("deepseek", "deepseek.com"),
        ("openai", "openai.com"),
        ("anthropic", "anthropic.com"),
    ):
        if _host_is(host, domain):
            return vendor
    return None


def _validate_known_host_matches_dialect(
    dialect: RequestDialect,
    base_url: object,
) -> None:
    observed = _known_host_vendor(base_url)
    if observed is None:
        return
    expected = {
        RequestDialect.DEEPSEEK_ANTHROPIC: "deepseek",
        RequestDialect.DEEPSEEK_OPENAI: "deepseek",
        RequestDialect.OPENAI_NATIVE: "openai",
        RequestDialect.ANTHROPIC_NATIVE: "anthropic",
    }.get(dialect)
    if expected != observed:
        raise ValueError(
            f"request dialect {dialect.value!r} conflicts with {observed} endpoint"
        )


def resolve_request_dialect(
    provider: object,
    base_url: object,
    configured: object = "auto",
    *,
    legacy_provider: object | None = None,
) -> RequestDialect:
    """将已存选项解析为一种具体且协议兼容的方言。

    ``auto`` 只识别第一方 hostname（以及旧 ``deepseek`` 供应商别名）。未知代理使用保守的
    通用契约，且不接收供应商专属推理字段。用户可以在其已存 profile 中显式为代理选择
    供应商方言。
    """

    protocol = str(provider or "").strip().lower()
    choice = str(configured or "auto").strip().lower() or "auto"
    old_name = str(legacy_provider or "").strip().lower()

    if protocol in {"", "mock"}:
        if choice not in {"auto", "mock"}:
            raise ValueError("request dialect is incompatible with mock provider")
        return RequestDialect.MOCK

    if choice in {"deepseek", "openai", "anthropic", "generic"}:
        dialect = {
            ("deepseek", "anthropic-compatible"): RequestDialect.DEEPSEEK_ANTHROPIC,
            ("deepseek", "openai-compatible"): RequestDialect.DEEPSEEK_OPENAI,
            ("openai", "openai-compatible"): RequestDialect.OPENAI_NATIVE,
            ("anthropic", "anthropic-compatible"): RequestDialect.ANTHROPIC_NATIVE,
            ("generic", "openai-compatible"): RequestDialect.GENERIC_OPENAI,
            ("generic", "anthropic-compatible"): RequestDialect.GENERIC_ANTHROPIC,
        }.get((choice, protocol))
        if dialect is None:
            raise ValueError(
                f"request dialect {choice!r} is incompatible with provider {protocol!r}"
            )
        _validate_known_host_matches_dialect(dialect, base_url)
        return dialect

    if choice != "auto":
        try:
            dialect = RequestDialect(choice)
        except ValueError as exc:
            raise ValueError(f"unknown request dialect: {choice}") from exc
        allowed = _OPENAI_DIALECTS if protocol == "openai-compatible" else _ANTHROPIC_DIALECTS
        if dialect not in allowed:
            raise ValueError(
                f"request dialect {choice!r} is incompatible with provider {protocol!r}"
            )
        _validate_known_host_matches_dialect(dialect, base_url)
        return dialect

    host = _hostname(base_url)
    if old_name == "deepseek" or _host_is(host, "deepseek.com"):
        return (
            RequestDialect.DEEPSEEK_OPENAI
            if protocol == "openai-compatible"
            else RequestDialect.DEEPSEEK_ANTHROPIC
        )
    if protocol == "openai-compatible":
        return (
            RequestDialect.OPENAI_NATIVE
            if _host_is(host, "openai.com")
            else RequestDialect.GENERIC_OPENAI
        )
    if protocol == "anthropic-compatible":
        return (
            RequestDialect.ANTHROPIC_NATIVE
            if _host_is(host, "anthropic.com")
            else RequestDialect.GENERIC_ANTHROPIC
        )
    raise ValueError(f"request dialect cannot resolve unsupported provider {protocol!r}")


def reasoning_capability(dialect: object) -> tuple[str, tuple[str, ...]]:
    """返回 UI 控制形状和供应商级推理强度选项。"""

    value = RequestDialect(str(getattr(dialect, "value", dialect)))
    if value in {
        RequestDialect.DEEPSEEK_ANTHROPIC,
        RequestDialect.DEEPSEEK_OPENAI,
    }:
        return "toggle_effort", ("low", "high", "max")
    if value is RequestDialect.ANTHROPIC_NATIVE:
    # Anthropic 推理强度控制整个响应，即使显式思考被禁用也仍有意义。它在 UI 中保持独立；
    # 具体模型不支持的组合由供应商拒绝。
        return "toggle_independent_effort", (
            "low", "medium", "high", "xhigh", "max"
        )
    if value is RequestDialect.OPENAI_NATIVE:
        return "effort", ("none", "minimal", "low", "medium", "high", "xhigh", "max")
    return "none", ()


def _effort_text(reasoning_effort: object | None) -> str | None:
    if reasoning_effort is None:
        return None
    text = str(getattr(reasoning_effort, "value", reasoning_effort)).strip().lower()
    return text or None


def _deepseek_effort(effort: str | None) -> str | None:
    if effort in {None, "none"}:
        return None
    if effort in {"minimal", "low"}:
        return "low"
    if effort in {"medium", "high", "xhigh"}:
        return "high"
    if effort == "max":
        return "max"
    return effort


def effective_reasoning_effort(
    dialect: object,
    reasoning_effort: object | None,
) -> str | None:
    """返回所选方言实际会收到的推理强度值。"""

    resolved = RequestDialect(str(getattr(dialect, "value", dialect)))
    effort = _effort_text(reasoning_effort)
    if resolved in {
        RequestDialect.DEEPSEEK_ANTHROPIC,
        RequestDialect.DEEPSEEK_OPENAI,
    }:
        return _deepseek_effort(effort)
    return effort


def reasoning_effort_is_compatible(
    dialect: object,
    reasoning_effort: object | None,
) -> bool:
    """持久化的显式推理强度在此处是否具有已定义语义。

    ``None``/``auto`` 始终表示供应商默认值。DeepSeek 通过文档化的 low/high/max 投影接受
    可移植分级值；``none`` 不是其显式思考开关的推理强度。原生 Anthropic 同样没有
    ``none`` 或 ``minimal`` 推理强度值。
    """

    resolved = RequestDialect(str(getattr(dialect, "value", dialect)))
    effort = _effort_text(reasoning_effort)
    if effort in {None, "auto"}:
        return True
    if resolved in {
        RequestDialect.DEEPSEEK_ANTHROPIC,
        RequestDialect.DEEPSEEK_OPENAI,
    }:
        return effort in {"minimal", "low", "medium", "high", "xhigh", "max"}
    if resolved is RequestDialect.ANTHROPIC_NATIVE:
        return effort in {"low", "medium", "high", "xhigh", "max"}
    if resolved is RequestDialect.OPENAI_NATIVE:
        return effort in {"none", "minimal", "low", "medium", "high", "xhigh", "max"}
    return False


def default_model_for_provider(provider: object) -> str:
    """返回网关针对某协议族的旧回退模型。"""

    normalized = str(provider or "").strip().lower()
    if normalized in {"deepseek", "anthropic-compatible"}:
        return "deepseek-v4-pro"
    if normalized == "anthropic":
    # 不存在永远且普遍有效的 Claude 模型 ID。原始旧 ``anthropic`` 供应商可以安全地隐含
    # Anthropic 端点，但绝不能静默继承 Entelecheia 的 DeepSeek 模型。
        return ""
    if normalized == "openai-compatible":
        return "gpt-4o-mini"
    return ""


def default_model_for_request_dialect(dialect: object) -> str:
    """仅在具体供应商无歧义时返回安全回退值。"""

    resolved = RequestDialect(str(getattr(dialect, "value", dialect)))
    if resolved in {
        RequestDialect.DEEPSEEK_ANTHROPIC,
        RequestDialect.DEEPSEEK_OPENAI,
    }:
        return "deepseek-v4-pro"
    if resolved is RequestDialect.OPENAI_NATIVE:
        return "gpt-4o-mini"
    # 原生 Anthropic 与任意兼容代理都要求显式模型。在此猜测可能向错误产品族发送语法有效的
    # 请求，这比可操作的配置错误更糟。
    return ""


def default_base_url_for_provider(provider: object) -> str:
    """返回网关针对某协议族的旧回退 URL。"""

    normalized = str(provider or "").strip().lower()
    if normalized == "anthropic":
        return "https://api.anthropic.com"
    if normalized in {"deepseek", "anthropic-compatible"}:
        return "https://api.deepseek.com/anthropic"
    if normalized == "openai-compatible":
        return "https://api.openai.com/v1"
    return ""


def build_request_controls(
    *,
    provider: str,
    dialect: object,
    thinking_enabled: bool | None,
    reasoning_effort: object | None,
    max_tokens: int,
    temperature: float | None,
    json_mode: bool,
) -> RequestControls:
    """构建精确线上字段及匹配的凭据安全指纹。"""

    resolved = resolve_request_dialect(provider, "", dialect)
    effort = _effort_text(reasoning_effort)
    fields: dict[str, Any] = {}
    reasoning_control = "not_supported"
    effective_effort: str | None = None

    if resolved is RequestDialect.OPENAI_NATIVE:
        fields["max_completion_tokens"] = max_tokens
        reasoning_control = "provider_default" if effort is None else "reasoning_effort"
        if effort is not None:
            fields["reasoning_effort"] = effort
            effective_effort = effort
        # 使用供应商默认推理强度时，所选模型仍可能推理；当前 OpenAI 推理模型会在该模式下
        # 拒绝 temperature。省略该字段对非推理模型同样有效。只有显式 ``none`` 才能让
        # 采样字段无歧义。
        if effort == "none" and temperature is not None:
            fields["temperature"] = temperature
    else:
        fields["max_tokens"] = max_tokens

    if resolved in {
        RequestDialect.DEEPSEEK_ANTHROPIC,
        RequestDialect.DEEPSEEK_OPENAI,
    }:
        enabled = None if thinking_enabled is None else bool(thinking_enabled)
        if enabled is not None:
            fields["thinking"] = {"type": "enabled" if enabled else "disabled"}
        reasoning_control = "provider_default" if enabled is None else "toggle_effort"
        if enabled is True:
            effective_effort = effective_reasoning_effort(resolved, effort)
            if effective_effort is not None:
                if resolved is RequestDialect.DEEPSEEK_ANTHROPIC:
                    fields["output_config"] = {"effort": effective_effort}
                else:
                    fields["reasoning_effort"] = effective_effort
        elif temperature is not None:
            fields["temperature"] = temperature
    elif resolved is RequestDialect.ANTHROPIC_NATIVE:
        enabled = None if thinking_enabled is None else bool(thinking_enabled)
        if enabled is not None:
            fields["thinking"] = {"type": "adaptive" if enabled else "disabled"}
        reasoning_control = (
            "provider_default"
            if enabled is None and effort is None
            else "toggle_independent_effort"
        )
        if reasoning_effort_is_compatible(resolved, effort) and effort is not None:
            fields["output_config"] = {"effort": effort}
            effective_effort = effort
        # 当前 Claude 各代模型会在多种有效思考配置下拒绝非默认采样参数（有些甚至在思考
        # 禁用时也会拒绝）。省略 temperature 具有可移植性；供应商默认值是原生模型族全部
        # 接受的唯一值。
    elif resolved in {
        RequestDialect.GENERIC_OPENAI,
        RequestDialect.GENERIC_ANTHROPIC,
    }:
        if temperature is not None:
            fields["temperature"] = temperature

    json_transport = "none"
    if json_mode:
        if resolved in {
            RequestDialect.OPENAI_NATIVE,
            RequestDialect.DEEPSEEK_OPENAI,
            RequestDialect.GENERIC_OPENAI,
            RequestDialect.DEEPSEEK_ANTHROPIC,
        }:
            fields["response_format"] = {"type": "json_object"}
            json_transport = "response_format_json_object"
        else:
        # prompt 已要求 JSON，调用方也会验证它。原生/通用 Anthropic Messages 没有不带
        # schema 的 json_object 开关，因此发送 OpenAI 字段只会产生 400。
            json_transport = "prompt_only"

    token_field = (
        "max_completion_tokens"
        if "max_completion_tokens" in fields
        else "max_tokens"
    )
    fingerprint = {
        "request_dialect": resolved.value,
        "token_limit_field": token_field,
        "temperature": fields.get("temperature", "omitted"),
        "json_transport": json_transport,
        "reasoning_control": reasoning_control,
        "thinking": (
            (fields.get("thinking") or {}).get("type")
            if isinstance(fields.get("thinking"), dict)
            else None
        ),
        "reasoning_effort": effective_effort,
    }
    return RequestControls(
        dialect=resolved,
        payload_fields=fields,
        fingerprint=fingerprint,
    )


__all__ = [
    "PROFILE_DIALECT_CHOICES",
    'RequestControls',
    'RequestDialect',
    "build_request_controls",
    "default_base_url_for_provider",
    "default_model_for_provider",
    "default_model_for_request_dialect",
    "effective_reasoning_effort",
    "reasoning_capability",
    "reasoning_effort_is_compatible",
    "resolve_request_dialect",
]
