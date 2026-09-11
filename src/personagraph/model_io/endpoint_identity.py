"""Entelecheia 已配置结构化模型端点的非密钥标识。

此所有者将配置解析为持久且凭据安全的端点事实。它特意不导入结构化补全网关、
模型提供商适配器、账本持久化或 Runtime 控制器。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from urllib.parse import SplitResult, urlsplit, urlunsplit

from .dialects import (
    build_request_controls,
    default_model_for_request_dialect,
    resolve_request_dialect,
)
from personagraph.configuration.app_settings import get_setting, resolve_global_model_endpoint


_ENDPOINT_IDENTITY_CONTRACT = "configured-structured-model-endpoint-v2"
_STRUCTURED_CONTROL_PROFILE = {
    "control_transport": "prompt_json",
    "json_mode": True,
    "stream": False,
    "temperature": 0,
}
# 绑定层级的调用还会固定是否运行隐藏推理，这会改变模型行为，进而改变重放结果的含义。
# 它使用独立契约，使绑定调用与未绑定调用绝不共享指纹，也使每个已存储未绑定行保留其预留时的精确标识。
_TIER_CONTROL_PROFILE_CONTRACT = "structured-completion-control-profile-tier-v2"


@dataclass(frozen=True, slots=True)
class ConfiguredStructuredModelEndpointIdentity:
    """有效结构化补全端点的非密钥标识。

    物理 URL 和凭据参与 ``endpoint_fingerprint``，但特意不出现在此投影中。将原像保留在本地，
    可防止持久逻辑请求、异常或调试 ``repr`` 持久化端点凭据或签名查询参数。

    """

    provider: str
    model: str
    protocol: str
    control_profile_id: str
    endpoint_fingerprint: str


def configured_structured_model_endpoint_identity(
    binding: object | None = None,
) -> ConfiguredStructuredModelEndpointIdentity:
    """解析 ``complete_structured`` 使用的精确有效标识。

    这是每个持久 Runtime 模型权威信息的唯一来源。它镜像网关的提供商别名、默认基础 URL、
    端点后缀和固定结构化输出控制，但不发起提供商请求。

    ``binding`` 必须与网关调用将收到的层级绑定相同。如果请求发往层级自身端点，却根据全局配置
    解析标识，就会记录调用从未使用的端点；本模块正是为防止这种制造出的权威信息而存在。
    不传入绑定会保留层级前行为，尤其会保留层级前指纹。
    """

    provider, model = configured_structured_model_facts(binding)
    if provider == "mock":
        protocol = "personagraph-mock-structured-v1"
        endpoint = "local://personagraph/mock-structured"
        credential_binding = "not-applicable"
    else:
        endpoint, protocol = _configured_physical_endpoint(
            provider,
            base_url=(
                str(getattr(binding, "base_url", ""))
                if binding is not None
                else None
            ),
        )
        api_key = (
            str(getattr(binding, "api_key", ""))
            if binding is not None
            else (get_setting("api_key") or "")
        )
        credential_binding = _sha256(
            {
                "contract": "configured-model-credential-binding-v1",
                "credential": api_key,
            }
        )
    control_profile_id = _sha256(
        {
            "contract": "structured-completion-control-profile-v1",
            **_STRUCTURED_CONTROL_PROFILE,
        }
        if binding is None
        else {
            "contract": _TIER_CONTROL_PROFILE_CONTRACT,
            **_STRUCTURED_CONTROL_PROFILE,
            "tier": str(getattr(getattr(binding, "tier", ""), "value", "")),
            **_wire_control_profile(provider, binding),
        }
    )
    endpoint_fingerprint = _sha256(
        {
            "contract": _ENDPOINT_IDENTITY_CONTRACT,
            "provider": provider,
            "protocol": protocol,
            "physical_endpoint": endpoint,
            "model": model,
            "credential_binding_sha256": credential_binding,
            "control_profile_id": control_profile_id,
        }
    )
    return ConfiguredStructuredModelEndpointIdentity(
        provider=provider,
        model=model,
        protocol=protocol,
        control_profile_id=control_profile_id,
        endpoint_fingerprint=endpoint_fingerprint,
    )


def _wire_control_profile(provider: str, binding: object) -> dict[str, object]:
    """使用网关的纯请求构建器作为标识来源。"""

    base_url = str(getattr(binding, "base_url", "") or "")
    dialect = resolve_request_dialect(
        provider,
        base_url,
        getattr(binding, "request_dialect", "auto"),
    )
    return build_request_controls(
        provider=provider,
        dialect=dialect,
        thinking_enabled=bool(getattr(binding, "thinking_enabled", False)),
        reasoning_effort=getattr(binding, "reasoning_effort", None),
    # 逻辑请求指纹持有数值输出预算；端点标识持有承载该预算的字段。
        max_tokens=0,
        temperature=0.0,
        json_mode=True,
    ).fingerprint


def _configured_physical_endpoint(
    provider: str, *, base_url: str | None = None
) -> tuple[str, str]:
    """解析一个提供商的物理端点。

    对指向自身配置的层级，``base_url`` 会覆盖全局设置。空覆盖值视为不存在，
    因此填写不完整的配置会回退，而不会生成不可用端点。
    """

    resolved_base_url = (
        resolve_global_model_endpoint().base_url
        if base_url is None
        else str(base_url).strip()
    ).rstrip("/")
    if provider == "anthropic-compatible":
        endpoint = (
            f"{resolved_base_url}/messages"
            if resolved_base_url.endswith("/v1")
            else f"{resolved_base_url}/v1/messages"
        )
        protocol = "anthropic-messages-2023-06-01"
    elif provider == "openai-compatible":
        endpoint = (
            resolved_base_url
            if resolved_base_url.endswith("/chat/completions")
            else f"{resolved_base_url}/chat/completions"
        )
        protocol = "openai-chat-completions-v1"
    else:
    # ``complete_structured`` 会在 I/O 前拒绝此提供商。它仍需确定性标识，
    # 使持久账本能够记录尝试的权威信息为何不同于受支持配置。
        endpoint = resolved_base_url
        protocol = "unsupported-structured-provider-v1"
    return _canonical_http_endpoint(endpoint), protocol


def _canonical_http_endpoint(endpoint: str) -> str:
    """只规范化线上含义等价的 URI 组件。"""

    try:
        parsed = urlsplit(endpoint)
        port = parsed.port
    except ValueError as exc:
        raise ValueError("configured structured model endpoint is invalid") from exc
    scheme = parsed.scheme.lower()
    if scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError(
            "configured structured model endpoint must be an absolute HTTP(S) URL"
        )
    host = parsed.hostname.lower()
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    if port is not None and not (
        (scheme == "http" and port == 80)
        or (scheme == "https" and port == 443)
    ):
        host = f"{host}:{port}"
    user_info = ""
    if "@" in parsed.netloc:
        user_info = f"{parsed.netloc.rsplit('@', 1)[0]}@"
    normalized = SplitResult(
        scheme=scheme,
        netloc=f"{user_info}{host}",
        path=parsed.path,
        query=parsed.query,
    # URL 片段不属于 HTTP 请求目标。
        fragment="",
    )
    return urlunsplit(normalized)


def _sha256(value: object) -> str:
    serialized = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def configured_structured_model_facts(
    binding: object | None = None,
) -> tuple[str, str]:
    """返回 ``complete_structured`` 发出的精确提供商/模型标签。

    M2 会在使用提供商前持久化这些事实，因此组合根必须解析网关将放入其 ``ModelResult`` 的
    同一规范化标签。此函数不执行提供商请求。

    传入 ``binding`` 时，标签来自该层级已解析端点，因此描述实际将发送的请求。
    """

    if binding is not None:
        configured_provider = str(
            getattr(binding, "provider", "mock") or "mock"
        ).strip().lower()
        dialect = resolve_request_dialect(
            configured_provider,
            getattr(binding, "base_url", ""),
            getattr(binding, "request_dialect", "auto"),
        )
        default_model = default_model_for_request_dialect(dialect)
        configured_model = str(
            getattr(binding, "model", "")
            or default_model
        ).strip()
    else:
        endpoint = resolve_global_model_endpoint()
        configured_provider = endpoint.raw_provider
        configured_model = endpoint.model
    if configured_provider in {"", "mock"}:
        return "mock", "mock-structured"
    if configured_provider in {
        "anthropic",
        "anthropic-compatible",
        "deepseek",
    }:
        return "anthropic-compatible", configured_model
    return configured_provider, configured_model


__all__ = [
    'ConfiguredStructuredModelEndpointIdentity',
    "configured_structured_model_endpoint_identity",
    "configured_structured_model_facts",
]
