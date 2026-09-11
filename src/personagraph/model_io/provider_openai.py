"""OpenAI-compatible 请求准备、HTTP 分派与 SSE 归一化。"""

from __future__ import annotations

import json
import os
import time
import uuid
from typing import Any

import httpx

from ..configuration.app_settings import (
    OPENAI_COMPATIBLE,
    resolve_global_model_endpoint,
)
from .context_budgeting import (
    PreparedAdmittedProviderRequest,
    prepare_and_admit_provider_request,
)
from .contracts import AssistantText
from .dialects import (
    build_request_controls,
    default_base_url_for_provider,
    default_model_for_request_dialect,
)
from .gateway_core import (
    ModelGatewayError,
    ModelResult,
    PreparedModelCall,
    _effective_model_timeout_s,
    _prepare_model_api_quota,
)
from .generation_stream import emit_stream_delta
from .openai import (
    normalize_openai_output,
    openai_tool_choice,
    openai_tool_definitions,
)
from .prompt_json import normalize_prompt_json_output
from .provider_support import (
    _elapsed_ms,
    _extract_finish_reason,
    _redact_endpoint,
    _require_endpoint_configuration,
    _resolved_request_dialect,
    _safe_provider_error_details,
)
from .tier_bindings import ModelTierBinding
from .trajectory_projection import (
    _admitted_payload_for_recording,
    _trajectory_reply_for_recording,
    _trajectory_response_for_recording,
)


def _gateway_get_setting(key: str, default: Any = None) -> Any:
    from . import gateway

    return gateway.get_setting(key, default)


def _record_model_call(**payload: Any) -> None:
    from . import gateway

    gateway.record_model_call(**payload)


def _gateway_openai_request(
    endpoint: str,
    admitted_request: PreparedAdmittedProviderRequest,
    api_key: str,
    timeout_s: float | None,
) -> dict[str, Any]:
    from . import gateway

    return gateway._openai_request(endpoint, admitted_request, api_key, timeout_s)


def _openai_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """OpenAI 把 system 当成一条普通消息，而不是顶层字段。"""

    return [
        {"role": item["role"], "content": _openai_message_content(item["content"])}
        for item in messages
        if item.get("role") in {"system", "user", "assistant"}
    ]


def _openai_message_content(content: Any) -> Any:
    """将 Entelecheia 的规范图像块转换为 Chat Completions 格式。"""

    if not isinstance(content, list):
        return content
    translated: list[Any] = []
    for block in content:
        if not isinstance(block, dict) or block.get("type") != "image":
            translated.append(block)
            continue
        source = block.get("source")
        if (
            not isinstance(source, dict)
            or source.get("type") != "base64"
            or not isinstance(source.get("media_type"), str)
            or not isinstance(source.get("data"), str)
        ):
            translated.append(block)
            continue
        translated.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": (
                        f"data:{source['media_type']};base64,{source['data']}"
                    )
                },
            }
        )
    return translated


def _extract_openai_text(data: dict[str, Any]) -> str:
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError(f"Could not extract text from model response keys={list(data.keys())}")
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    if not isinstance(message, dict):
        return ""
    return str(message.get("content") or "").strip()


def _extract_openai_reasoning(payload: Any) -> str:
    """推理内容在 OpenAI 形状里没有统一字段名，各家用的名字不同。

    只认已经见过的这几个；认不出来就当作没有，而不是猜一个键名去读。
    """

    if not isinstance(payload, dict):
        return ""
    for key in ("reasoning_content", "reasoning"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def openai_compatible_chat(
    messages: list[dict[str, Any]],
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
    json_mode: bool = False,
    timeout_s: float | None = None,
    stream: bool = False,
    tools: list[dict[str, Any]] | None = None,
    control_transport: str = "prompt_json",
    force_prompt_json: bool = False,
    tool_choice: dict[str, Any] | str | None = None,
    model_call_id: str | None = None,
    purpose: str = "task_generation",
    binding: ModelTierBinding | None = None,
) -> ModelResult:
    """准备、准入并分派一个 OpenAI 形状的请求。"""

    return prepare_openai_compatible_chat(
        messages,
        temperature=temperature,
        max_tokens=max_tokens,
        json_mode=json_mode,
        timeout_s=timeout_s,
        stream=stream,
        tools=tools,
        control_transport=control_transport,
        force_prompt_json=force_prompt_json,
        tool_choice=tool_choice,
        purpose=purpose,
        binding=binding,
    ).dispatch(model_call_id=model_call_id)


def prepare_openai_compatible_chat(
    messages: list[dict[str, Any]],
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
    json_mode: bool = False,
    timeout_s: float | None = None,
    stream: bool = False,
    tools: list[dict[str, Any]] | None = None,
    control_transport: str = "prompt_json",
    force_prompt_json: bool = False,
    tool_choice: dict[str, Any] | str | None = None,
    purpose: str = "task_generation",
    binding: ModelTierBinding | None = None,
    projection_epoch: str = "provider-wire",
    projection_generation: int = 0,
    configured_input_limit_tokens: int | None = None,
) -> PreparedModelCall:
    """把中性消息、冻结 binding 与请求控制转成 OpenAI wire payload，并做上下文准入。

    messages 保留 system 角色，内容块经 _openai_messages 适配；原生工具只在对应
    control transport 下加入。返回对象持有已准入字节和后续 dispatch 闭包，
    本函数不打开 HTTP；传入 binding 优先于全局 endpoint / thinking 配置。
    """

    if binding is not None:
        api_key = binding.api_key
        base_url = str(binding.base_url or "").strip().rstrip("/")
        dialect = _resolved_request_dialect(
            OPENAI_COMPATIBLE, base_url, binding
        )
        model = str(binding.model or "").strip() or default_model_for_request_dialect(
            dialect
        )
    else:
        endpoint_config = resolve_global_model_endpoint()
        api_key = _gateway_get_setting("api_key")
        base_url = (
            endpoint_config.base_url
            or default_base_url_for_provider(OPENAI_COMPATIBLE)
        ).rstrip("/")
        dialect = _resolved_request_dialect(
            OPENAI_COMPATIBLE, base_url, None
        )
        model = endpoint_config.model or default_model_for_request_dialect(dialect)
    max_tokens = max_tokens if max_tokens is not None else int(os.getenv("PERSONAGRAPH_MAX_TOKENS", "500"))
    temperature = temperature if temperature is not None else float(os.getenv("PERSONAGRAPH_TEMPERATURE", "0.2"))
    _require_endpoint_configuration(
        provider=OPENAI_COMPATIBLE,
        base_url=base_url,
        model=model,
    )
    if not api_key:
        raise ModelGatewayError(
            "MODEL_CALL_FAILED",
            "PERSONAGRAPH_API_KEY is required for openai-compatible provider.",
            retryable=False,
            details={"provider": "openai-compatible", "reason": "missing_api_key"},
        )

    endpoint = f"{base_url}/chat/completions" if not base_url.endswith("/chat/completions") else base_url
    request_json_mode = bool(json_mode or (
        force_prompt_json and control_transport == "prompt_json" and tools
    ))
    controls = build_request_controls(
        provider=OPENAI_COMPATIBLE,
        dialect=dialect,
        thinking_enabled=(binding.thinking_enabled if binding is not None else None),
        reasoning_effort=(binding.reasoning_effort if binding is not None else None),
        max_tokens=max_tokens,
        temperature=temperature,
        json_mode=request_json_mode,
    )
    payload: dict[str, Any] = {
        "model": model,
        "messages": _openai_messages(messages),
        **controls.payload_fields,
    }
    if control_transport == "native" and tools:
        payload["tools"] = openai_tool_definitions(tools)
        if tool_choice is not None:
            payload["tool_choice"] = openai_tool_choice(tool_choice)

    context_budget = prepare_and_admit_provider_request(
        payload,
        provider=OPENAI_COMPATIBLE,
        model=model,
        dialect=dialect,
        stream=stream,
        purpose=purpose,
        projection_epoch=projection_epoch,
        projection_generation=projection_generation,
        configured_input_limit_tokens=configured_input_limit_tokens,
    )

    def _dispatch(model_call_id: str | None) -> ModelResult:
        if stream:
            return _stream_openai_compatible_chat(
                endpoint=endpoint,
                admitted_request=context_budget,
                api_key=api_key,
                model=model,
                timeout_s=timeout_s,
                model_call_id=model_call_id,
                purpose=purpose,
                control_transport=control_transport,
                parse_prompt_control=bool(tools),
            )
        return _dispatch_openai_compatible_chat(
            endpoint=endpoint,
            admitted_request=context_budget,
            api_key=api_key,
            model=model,
            timeout_s=timeout_s,
            model_call_id=model_call_id,
            purpose=purpose,
            control_transport=control_transport,
            parse_prompt_control=bool(tools),
        )

    return PreparedModelCall(
        _dispatch=_dispatch,
        context_budget=context_budget,
        api_quota=_prepare_model_api_quota(
            context_budget=context_budget,
            provider=OPENAI_COMPATIBLE,
            base_url=base_url,
            api_key=api_key,
            timeout_s=timeout_s,
            binding=binding,
        ),
    )


def _dispatch_openai_compatible_chat(
    *,
    endpoint: str,
    admitted_request: PreparedAdmittedProviderRequest,
    api_key: str,
    model: str,
    timeout_s: float | None,
    model_call_id: str | None,
    purpose: str,
    control_transport: str,
    parse_prompt_control: bool,
) -> ModelResult:
    """发送已准入的精确请求，提取 OpenAI 响应，再记录 trajectory 并返回 ModelResult。

    recorder 位于收到/提取响应之后、Runtime 合同验证之前：这里返回 OK 不能证明
    Plan/Action 已获准。请求、reply 和 reasoning 先按 recording projection 处理；
    传输或提取失败可能在到达这条记录语句前抛出，由上层决定重试与终态诊断。
    """

    start = time.perf_counter()
    data = _gateway_openai_request(endpoint, admitted_request, api_key, timeout_s)
    try:
        reply = _extract_openai_text(data)
    except RuntimeError as exc:
        raise ModelGatewayError(
            "MODEL_BAD_RESPONSE",
            "Could not extract text from model provider response.",
            retryable=True,
            details={
                "provider": "openai-compatible",
                "response_keys": list(data.keys()) if isinstance(data, dict) else [],
                "exception_type": type(exc).__name__,
            },
        ) from exc

    usage = data.get("usage") or {}
    latency_ms = _elapsed_ms(start)
    resolved_call_id = model_call_id or str(uuid.uuid4())
    choice = (data.get("choices") or [{}])[0]
    message = choice.get("message") if isinstance(choice, dict) else None
    reasoning = _extract_openai_reasoning(message)
    if control_transport == "native":
        output = normalize_openai_output(message)
    elif parse_prompt_control:
        output = normalize_prompt_json_output(reply)
    else:
        output = AssistantText(text=reply)
    payload = _admitted_payload_for_recording(admitted_request)
    _record_model_call(
        model_call_id=resolved_call_id,
        purpose=purpose,
        provider="openai-compatible",
        model=model,
        payload=payload,
        # 归一成 Anthropic 那套块形状，轨迹里两家的记录才长得一样。
        response=_trajectory_response_for_recording(
            {
                "content": (
                    [{"type": "thinking", "thinking": reasoning}]
                    if reasoning
                    else []
                ),
                "usage": {
                    "input_tokens": usage.get("prompt_tokens"),
                    "output_tokens": usage.get("completion_tokens"),
                },
            }
        ),
        reply=_trajectory_reply_for_recording(reply),
        duration_ms=latency_ms,
    )
    return ModelResult(
        reply=reply,
        provider="openai-compatible",
        model=model,
        latency_ms=latency_ms,
        input_tokens=usage.get("prompt_tokens"),
        output_tokens=usage.get("completion_tokens"),
        finish_reason=choice.get("finish_reason"),
        output=output,
        model_call_id=resolved_call_id,
        purpose=purpose,
        control_transport=control_transport,
    )


def _openai_request(
    endpoint: str,
    admitted_request: PreparedAdmittedProviderRequest,
    api_key: str,
    timeout_s: float | None,
) -> dict[str, Any]:
    """发起一次请求，并采用与另一适配器相同的失败词汇。"""

    effective_timeout_s = _effective_model_timeout_s(timeout_s)
    try:
        with httpx.Client(timeout=effective_timeout_s) as client:
            response = client.post(
                endpoint,
                headers={
                    "authorization": f"Bearer {api_key}",
                    "content-type": "application/json",
                },
                content=admitted_request.admitted_request.body_for_dispatch(),
            )
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as exc:
        status_code = exc.response.status_code if exc.response is not None else None
        raise ModelGatewayError(
            "MODEL_CALL_FAILED",
            "Model provider returned an HTTP error.",
            retryable=status_code is None or status_code >= 500 or status_code == 429,
            details={
                "provider": "openai-compatible", "status_code": status_code,
                "endpoint": _redact_endpoint(endpoint), "exception_type": type(exc).__name__,
                **_safe_provider_error_details(exc.response, api_key),
            },
        ) from exc
    except httpx.TimeoutException as exc:
        raise ModelGatewayError(
            "MODEL_CALL_TIMEOUT", "Model provider request timed out.", retryable=True,
            details={
                "provider": "openai-compatible", "endpoint": _redact_endpoint(endpoint),
                "exception_type": type(exc).__name__,
                "timeout_s": effective_timeout_s,
            },
        ) from exc
    except httpx.RequestError as exc:
        raise ModelGatewayError(
            "MODEL_CALL_FAILED", "Model provider request failed.", retryable=True,
            details={
                "provider": "openai-compatible", "endpoint": _redact_endpoint(endpoint),
                "exception_type": type(exc).__name__,
            },
        ) from exc
    except ValueError as exc:
        raise ModelGatewayError(
            "MODEL_BAD_RESPONSE", "Model provider response was not valid JSON.", retryable=True,
            details={
                "provider": "openai-compatible", "endpoint": _redact_endpoint(endpoint),
                "exception_type": type(exc).__name__,
            },
        ) from exc


def _stream_openai_compatible_chat(
    *,
    endpoint: str,
    admitted_request: PreparedAdmittedProviderRequest,
    api_key: str,
    model: str,
    timeout_s: float | None,
    model_call_id: str | None,
    purpose: str,
    control_transport: str = "prompt_json",
    parse_prompt_control: bool = False,
) -> ModelResult:
    """读取 OpenAI 形状的 SSE，其中增量位于 choices[0].delta。"""

    effective_timeout_s = _effective_model_timeout_s(timeout_s)
    start = time.perf_counter()
    chunks: list[str] = []
    reasoning_chunks: list[str] = []
    usage: dict[str, Any] = {}
    finish_reason: str | None = None
    saw_payload = False
    native_tool_calls: dict[int, dict[str, Any]] = {}
    try:
        with httpx.Client(timeout=effective_timeout_s) as client:
            with client.stream(
                "POST",
                endpoint,
                headers={
                    "authorization": f"Bearer {api_key}",
                    "content-type": "application/json",
                },
                content=admitted_request.admitted_request.body_for_dispatch(),
            ) as response:
                response.raise_for_status()
                for raw_line in response.iter_lines():
                    line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else str(raw_line)
                    if not line.startswith("data:"):
                        continue
                    raw_data = line.removeprefix("data:").strip()
                    if not raw_data or raw_data == "[DONE]":
                        continue
                    data = json.loads(raw_data)
                    if not isinstance(data, dict):
                        raise ValueError("stream payload must be an object")
                    saw_payload = True
                    choice = (data.get("choices") or [{}])[0]
                    delta = choice.get("delta") if isinstance(choice, dict) else None
                    reasoning_chunks.append(_extract_openai_reasoning(delta))
                    text = str((delta or {}).get("content") or "")
                    if text:
                        chunks.append(text)
                        emit_stream_delta(text)
                    if isinstance(delta, dict) and isinstance(
                        delta.get("tool_calls"), list
                    ):
                        for raw_call in delta["tool_calls"]:
                            if not isinstance(raw_call, dict):
                                continue
                            index = raw_call.get("index")
                            if not isinstance(index, int) or index < 0:
                                continue
                            state = native_tool_calls.setdefault(
                                index,
                                {
                                    "id": "",
                                    "type": "function",
                                    "function": {"name": "", "arguments": ""},
                                },
                            )
                            if isinstance(raw_call.get("id"), str):
                                state["id"] += raw_call["id"]
                            function = raw_call.get("function")
                            if isinstance(function, dict):
                                if isinstance(function.get("name"), str):
                                    state["function"]["name"] += function["name"]
                                if isinstance(function.get("arguments"), str):
                                    state["function"]["arguments"] += function["arguments"]
                    if isinstance(choice, dict) and choice.get("finish_reason"):
                        finish_reason = str(choice["finish_reason"])
                    if isinstance(data.get("usage"), dict):
                        usage = data["usage"]
    except httpx.HTTPStatusError as exc:
        status_code = exc.response.status_code if exc.response is not None else None
        raise ModelGatewayError(
            "MODEL_CALL_FAILED", "Model provider returned an HTTP error.",
            retryable=status_code is None or status_code >= 500 or status_code == 429,
            details={
                "provider": "openai-compatible", "status_code": status_code,
                "endpoint": _redact_endpoint(endpoint), "exception_type": type(exc).__name__,
                **_safe_provider_error_details(exc.response, api_key),
            },
        ) from exc
    except httpx.TimeoutException as exc:
        raise ModelGatewayError(
            "MODEL_CALL_TIMEOUT", "Model provider request timed out.", retryable=True,
            details={
                "provider": "openai-compatible", "endpoint": _redact_endpoint(endpoint),
                "exception_type": type(exc).__name__,
            },
        ) from exc
    except (httpx.RequestError, ValueError) as exc:
        raise ModelGatewayError(
            "MODEL_CALL_FAILED", "Model provider stream failed.", retryable=True,
            details={
                "provider": "openai-compatible", "endpoint": _redact_endpoint(endpoint),
                "exception_type": type(exc).__name__,
            },
        ) from exc

    if not saw_payload:
        raise ModelGatewayError(
            "MODEL_BAD_RESPONSE", "Model provider stream did not contain a response payload.",
            retryable=True,
            details={"provider": "openai-compatible", "endpoint": _redact_endpoint(endpoint)},
        )

    reply = "".join(chunks).strip()
    reasoning = "".join(reasoning_chunks).strip()
    if control_transport == "native":
        output = normalize_openai_output(
            {
                "content": reply,
                "tool_calls": [
                    native_tool_calls[index]
                    for index in sorted(native_tool_calls)
                ],
            }
        )
    elif parse_prompt_control:
        output = normalize_prompt_json_output(reply)
    else:
        output = AssistantText(text=reply)
    latency_ms = _elapsed_ms(start)
    resolved_call_id = model_call_id or str(uuid.uuid4())
    payload = _admitted_payload_for_recording(admitted_request)
    _record_model_call(
        model_call_id=resolved_call_id,
        purpose=purpose,
        provider="openai-compatible",
        model=model,
        payload=payload,
        response=_trajectory_response_for_recording(
            {
                "content": (
                    [{"type": "thinking", "thinking": reasoning}]
                    if reasoning
                    else []
                ),
                "usage": {
                    "input_tokens": usage.get("prompt_tokens"),
                    "output_tokens": usage.get("completion_tokens"),
                },
            }
        ),
        reply=_trajectory_reply_for_recording(reply),
        duration_ms=latency_ms,
    )
    return ModelResult(
        reply=reply,
        provider="openai-compatible",
        model=model,
        latency_ms=latency_ms,
        input_tokens=usage.get("prompt_tokens"),
        output_tokens=usage.get("completion_tokens"),
        finish_reason=finish_reason,
        output=output,
        model_call_id=resolved_call_id,
        purpose=purpose,
        control_transport=control_transport,
    )

openai_compatible_chat.prepare = prepare_openai_compatible_chat  # type: ignore[attr-defined]
