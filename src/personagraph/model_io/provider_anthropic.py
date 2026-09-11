"""Anthropic-compatible 请求准备、HTTP 分派与 SSE 归一化。"""

from __future__ import annotations

import json
import os
import time
import uuid
from typing import Any

import httpx

from ..configuration.app_settings import (
    ANTHROPIC_COMPATIBLE,
    resolve_global_model_endpoint,
)
from .anthropic import anthropic_tool_definitions, normalize_anthropic_output
from .context_budgeting import (
    PreparedAdmittedProviderRequest,
    prepare_and_admit_provider_request,
)
from .contracts import AssistantText, ProtocolError
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


def anthropic_compatible_chat(
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
    model_call_id: str | None = None,
    purpose: str = "task_generation",
    tool_choice: dict[str, Any] | None = None,
    binding: ModelTierBinding | None = None,
) -> ModelResult:
    """准备、准入并分派一个 Anthropic 形状的请求。"""

    return prepare_anthropic_compatible_chat(
        messages,
        temperature=temperature,
        max_tokens=max_tokens,
        json_mode=json_mode,
        timeout_s=timeout_s,
        stream=stream,
        tools=tools,
        control_transport=control_transport,
        force_prompt_json=force_prompt_json,
        purpose=purpose,
        tool_choice=tool_choice,
        binding=binding,
    ).dispatch(model_call_id=model_call_id)


def prepare_anthropic_compatible_chat(
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
    purpose: str = "task_generation",
    tool_choice: dict[str, Any] | None = None,
    binding: ModelTierBinding | None = None,
    projection_epoch: str = "provider-wire",
    projection_generation: int = 0,
    configured_input_limit_tokens: int | None = None,
) -> PreparedModelCall:
    """准备 Anthropic 方言请求并执行上下文准入，不在此发送 HTTP。

    system 消息抽为顶层 system，user / assistant 留在 messages；binding 提供本
    调用冻结的 endpoint 与 thinking 控制，缺省时读取全局配置。工具和输出控制经
    方言适配后一起计量，返回的 dispatch 闭包随后复用同一份已准入字节。
    """

    if binding is not None:
        api_key = binding.api_key
        base_url = str(binding.base_url or "").strip().rstrip("/")
        dialect = _resolved_request_dialect(
            ANTHROPIC_COMPATIBLE, base_url, binding
        )
        model = str(binding.model or "").strip() or default_model_for_request_dialect(
            dialect
        )
    else:
        endpoint_config = resolve_global_model_endpoint()
        api_key = _gateway_get_setting("api_key")
        base_url = (
            endpoint_config.base_url
            or default_base_url_for_provider(ANTHROPIC_COMPATIBLE)
        ).rstrip("/")
        dialect = _resolved_request_dialect(
            ANTHROPIC_COMPATIBLE, base_url, None
        )
        model = endpoint_config.model or default_model_for_request_dialect(dialect)
    max_tokens = max_tokens if max_tokens is not None else int(os.getenv("PERSONAGRAPH_MAX_TOKENS", "500"))
    temperature = temperature if temperature is not None else float(os.getenv("PERSONAGRAPH_TEMPERATURE", "0.2"))
    _require_endpoint_configuration(
        provider=ANTHROPIC_COMPATIBLE,
        base_url=base_url,
        model=model,
    )
    if not api_key:
        raise ModelGatewayError(
            "MODEL_CALL_FAILED",
            "PERSONAGRAPH_API_KEY is required for anthropic-compatible provider.",
            retryable=False,
            details={"provider": "anthropic-compatible", "reason": "missing_api_key"},
        )

    system_parts = [m["content"] for m in messages if m["role"] == "system"]
    anthropic_messages = [
        {"role": m["role"], "content": m["content"]}
        for m in messages
        if m["role"] in {"user", "assistant"}
    ]
    endpoint = f"{base_url}/v1/messages" if not base_url.endswith("/v1") else f"{base_url}/messages"
    request_json_mode = bool(json_mode or (
        force_prompt_json and control_transport == "prompt_json" and tools
    ))
    controls = build_request_controls(
        provider=ANTHROPIC_COMPATIBLE,
        dialect=dialect,
        thinking_enabled=(binding.thinking_enabled if binding is not None else None),
        reasoning_effort=(binding.reasoning_effort if binding is not None else None),
        max_tokens=max_tokens,
        temperature=temperature,
        json_mode=request_json_mode,
    )
    payload: dict[str, Any] = {
        "model": model,
        "messages": anthropic_messages,
        **controls.payload_fields,
    }
    if system_parts:
        payload["system"] = "\n\n".join(system_parts)
    if control_transport == "native" and tools:
        payload["tools"] = anthropic_tool_definitions(tools)
        if tool_choice:
            payload["tool_choice"] = tool_choice

    context_budget = prepare_and_admit_provider_request(
        payload,
        provider=ANTHROPIC_COMPATIBLE,
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
            return _stream_anthropic_compatible_chat(
                endpoint=endpoint,
                admitted_request=context_budget,
                api_key=api_key,
                model=model,
                timeout_s=timeout_s,
                control_transport=control_transport,
                model_call_id=model_call_id,
                purpose=purpose,
                parse_prompt_control=bool(tools),
            )
        return _dispatch_anthropic_compatible_chat(
            endpoint=endpoint,
            admitted_request=context_budget,
            api_key=api_key,
            model=model,
            timeout_s=timeout_s,
            control_transport=control_transport,
            model_call_id=model_call_id,
            purpose=purpose,
            parse_prompt_control=bool(tools),
        )

    return PreparedModelCall(
        _dispatch=_dispatch,
        context_budget=context_budget,
        api_quota=_prepare_model_api_quota(
            context_budget=context_budget,
            provider=ANTHROPIC_COMPATIBLE,
            base_url=base_url,
            api_key=api_key,
            timeout_s=timeout_s,
            binding=binding,
        ),
    )


def _dispatch_anthropic_compatible_chat(
    *,
    endpoint: str,
    admitted_request: PreparedAdmittedProviderRequest,
    api_key: str,
    model: str,
    timeout_s: float | None,
    control_transport: str,
    model_call_id: str | None,
    purpose: str,
    parse_prompt_control: bool,
) -> ModelResult:
    """在剩余 HTTP timeout 上限内发送已准入 Anthropic 字节，并归一化响应/用量。

    httpx 传输、状态码或 JSON 错误转为 ModelGatewayError，是否重试由 Runtime 决定。
    获得可提取响应后才调用 trajectory recorder；结构化 reply/thinking 会先投影，
    原 ModelResult 仍交给上层验证，不能把网关返回当作最终答案通过。
    """

    effective_timeout_s = _effective_model_timeout_s(timeout_s)
    start = time.perf_counter()
    try:
        with httpx.Client(timeout=effective_timeout_s) as client:
            response = client.post(
                endpoint,
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                content=admitted_request.admitted_request.body_for_dispatch(),
            )
            response.raise_for_status()
            data = response.json()
    except httpx.HTTPStatusError as exc:
        status_code = exc.response.status_code if exc.response is not None else None
        raise ModelGatewayError(
            "MODEL_CALL_FAILED",
            "Model provider returned an HTTP error.",
            retryable=status_code is None or status_code >= 500 or status_code == 429,
            details={
                "provider": "anthropic-compatible",
                "status_code": status_code,
                "endpoint": _redact_endpoint(endpoint),
                "exception_type": type(exc).__name__,
                **_safe_provider_error_details(exc.response, api_key),
            },
        ) from exc
    except httpx.TimeoutException as exc:
        raise ModelGatewayError(
            "MODEL_CALL_TIMEOUT",
            "Model provider request timed out.",
            retryable=True,
            details={
                "provider": "anthropic-compatible",
                "endpoint": _redact_endpoint(endpoint),
                "exception_type": type(exc).__name__,
                "timeout_s": effective_timeout_s,
            },
        ) from exc
    except httpx.RequestError as exc:
        raise ModelGatewayError(
            "MODEL_CALL_FAILED",
            "Model provider request failed.",
            retryable=True,
            details={
                "provider": "anthropic-compatible",
                "endpoint": _redact_endpoint(endpoint),
                "exception_type": type(exc).__name__,
            },
        ) from exc
    except ValueError as exc:
        raise ModelGatewayError(
            "MODEL_BAD_RESPONSE",
            "Model provider response was not valid JSON.",
            retryable=True,
            details={
                "provider": "anthropic-compatible",
                "endpoint": _redact_endpoint(endpoint),
                "exception_type": type(exc).__name__,
            },
        ) from exc

    try:
        reply = _extract_anthropic_text(data)
    except RuntimeError as exc:
        raise ModelGatewayError(
            "MODEL_BAD_RESPONSE",
            "Could not extract text from model provider response.",
            retryable=True,
            details={
                "provider": "anthropic-compatible",
                "response_keys": list(data.keys()) if isinstance(data, dict) else [],
                "exception_type": type(exc).__name__,
            },
        ) from exc
    usage = data.get("usage", {})
    if control_transport == "native":
        output = normalize_anthropic_output(data.get("content"))
    elif parse_prompt_control:
        output = normalize_prompt_json_output(reply)
    else:
        output = AssistantText(text=reply)
    latency_ms = _elapsed_ms(start)
    resolved_call_id = model_call_id or str(uuid.uuid4())
    # 在此处而非调用方记录，因为只有这里仍同时持有两部分：实际发出的精确 payload，
    # 以及原始响应——包括上方 _extract_anthropic_text 丢弃的 thinking 块。记录操作不会
    # 导致调用失败；参见 recorder。
    payload = _admitted_payload_for_recording(admitted_request)
    _record_model_call(
        model_call_id=resolved_call_id,
        purpose=purpose,
        provider="anthropic-compatible",
        model=model,
        payload=payload,
        response=_trajectory_response_for_recording(data),
        reply=_trajectory_reply_for_recording(reply),
        duration_ms=latency_ms,
    )
    return ModelResult(
        reply=reply,
        provider="anthropic-compatible",
        model=model,
        latency_ms=latency_ms,
        input_tokens=usage.get("input_tokens"),
        output_tokens=usage.get("output_tokens"),
        cache_read_tokens=usage.get("cache_read_input_tokens"),
        cache_write_tokens=usage.get("cache_creation_input_tokens"),
        finish_reason=_extract_finish_reason(data),
        output=output,
        model_call_id=resolved_call_id,
        purpose=purpose,
        control_transport=control_transport,
    )


def _stream_anthropic_compatible_chat(
    *,
    endpoint: str,
    admitted_request: PreparedAdmittedProviderRequest,
    api_key: str,
    model: str,
    timeout_s: float | None,
    control_transport: str,
    model_call_id: str | None,
    purpose: str,
    parse_prompt_control: bool,
) -> ModelResult:
    """读取 Anthropic 兼容 SSE，同时保持现有 ModelResult 契约。"""
    effective_timeout_s = _effective_model_timeout_s(timeout_s)
    start = time.perf_counter()
    chunks: list[str] = []
    # 与 `chunks` 分开累积：推理内容不能进入调用方回复，而且流本就通过另一个 delta 字段
    # 传递它。
    thinking_chunks: list[str] = []
    usage: dict[str, Any] = {}
    finish_reason: str | None = None
    saw_payload = False
    native_blocks: list[dict[str, Any]] = []
    active_native_blocks: dict[int, dict[str, Any]] = {}
    try:
        with httpx.Client(timeout=effective_timeout_s) as client:
            with client.stream(
                "POST",
                endpoint,
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
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
                    thinking_chunks.append(_extract_stream_thinking(data))
                    text = _extract_stream_text(data)
                    if text:
                        chunks.append(text)
                        emit_stream_delta(text)
                    _consume_native_stream_event(data, active_native_blocks, native_blocks)
                    usage.update(_extract_stream_usage(data))
                    finish_reason = _extract_finish_reason(data) or finish_reason
    except httpx.HTTPStatusError as exc:
        status_code = exc.response.status_code if exc.response is not None else None
        raise ModelGatewayError(
            "MODEL_CALL_FAILED",
            "Model provider returned an HTTP error.",
            retryable=status_code is None or status_code >= 500 or status_code == 429,
            details={
                "provider": "anthropic-compatible",
                "status_code": status_code,
                "endpoint": _redact_endpoint(endpoint),
                "exception_type": type(exc).__name__,
                **_safe_provider_error_details(exc.response, api_key),
            },
        ) from exc
    except httpx.TimeoutException as exc:
        raise ModelGatewayError(
            "MODEL_CALL_TIMEOUT",
            "Model provider request timed out.",
            retryable=True,
            details={
                "provider": "anthropic-compatible",
                "endpoint": _redact_endpoint(endpoint),
                "exception_type": type(exc).__name__,
                "timeout_s": effective_timeout_s,
            },
        ) from exc
    except httpx.RequestError as exc:
        raise ModelGatewayError(
            "MODEL_CALL_FAILED",
            "Model provider request failed.",
            retryable=True,
            details={
                "provider": "anthropic-compatible",
                "endpoint": _redact_endpoint(endpoint),
                "exception_type": type(exc).__name__,
            },
        ) from exc
    except (ValueError, TypeError) as exc:
        raise ModelGatewayError(
            "MODEL_BAD_RESPONSE",
            "Model provider stream was not valid SSE JSON.",
            retryable=True,
            details={
                "provider": "anthropic-compatible",
                "endpoint": _redact_endpoint(endpoint),
                "exception_type": type(exc).__name__,
            },
        ) from exc

    if not saw_payload:
        raise ModelGatewayError(
            "MODEL_BAD_RESPONSE",
            "Model provider stream did not contain a response payload.",
            retryable=True,
            details={"provider": "anthropic-compatible", "endpoint": _redact_endpoint(endpoint)},
        )
    if control_transport == "native" and chunks:
        native_blocks.insert(0, {"type": "text", "text": "".join(chunks)})
    if control_transport == "native" and active_native_blocks:
        output = ProtocolError(
            code="incomplete_native_tool_call",
            message="The native stream ended before a tool_use block completed.",
            transport="native",
        )
    elif control_transport == "native":
        output = normalize_anthropic_output(native_blocks)
    elif parse_prompt_control:
        output = normalize_prompt_json_output("".join(chunks).strip())
    else:
        output = AssistantText(text="".join(chunks).strip())
    reply = "".join(chunks).strip()
    latency_ms = _elapsed_ms(start)
    resolved_call_id = model_call_id or str(uuid.uuid4())
    # 将流式推理重新组装成与非流式路径相同的块形状，使记录步骤不受答案到达方式影响，
    # 读取结果保持一致。
    thinking_text = "".join(thinking_chunks).strip()
    payload = _admitted_payload_for_recording(admitted_request)
    _record_model_call(
        model_call_id=resolved_call_id,
        purpose=purpose,
        provider="anthropic-compatible",
        model=model,
        payload=payload,
        response=_trajectory_response_for_recording(
            {
                "content": (
                    [{"type": "thinking", "thinking": thinking_text}]
                    if thinking_text
                    else []
                ),
                "usage": usage,
            }
        ),
        reply=_trajectory_reply_for_recording(reply),
        duration_ms=latency_ms,
    )
    return ModelResult(
        reply=reply,
        provider="anthropic-compatible",
        model=model,
        latency_ms=latency_ms,
        input_tokens=usage.get("input_tokens"),
        output_tokens=usage.get("output_tokens"),
        cache_read_tokens=usage.get("cache_read_input_tokens"),
        cache_write_tokens=usage.get("cache_creation_input_tokens"),
        finish_reason=finish_reason,
        output=output,
        model_call_id=resolved_call_id,
        purpose=purpose,
        control_transport=control_transport,
    )


def _extract_anthropic_text(data: dict[str, Any]) -> str:
    parts: list[str] = []
    for item in data.get("content", []):
        if isinstance(item, dict) and item.get("type") == "text":
            parts.append(item.get("text", ""))
    if parts:
        return "\n".join(part for part in parts if part).strip()
    if "completion" in data:
        return str(data["completion"]).strip()
    if "content" in data:
        # 偶发空 content（如截断/thinking-only 响应）：返回空串走调用方容错，不炸整轮
        return ""
    raise RuntimeError(f"Could not extract text from model response keys={list(data.keys())}")


def _extract_stream_text(data: dict[str, Any]) -> str:
    """接受 Anthropic SSE 增量和常见的 OpenAI 兼容回退形状。"""
    delta = data.get("delta")
    if isinstance(delta, dict) and isinstance(delta.get("text"), str):
        return delta["text"]
    choices = data.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        choice_delta = choices[0].get("delta")
        if isinstance(choice_delta, dict) and isinstance(choice_delta.get("content"), str):
            return choice_delta["content"]
    return ""


def _extract_stream_thinking(data: dict[str, Any]) -> str:
    """从流中提取一个推理增量。

    推理内容通过专属 delta 字段到达，因此上方文本提取器会对此类事件返回空值，使回复保持
    干净。它仅为 trajectory 保留——绝不会追加到调用方可见内容中。
    """

    delta = data.get("delta")
    if isinstance(delta, dict) and isinstance(delta.get("thinking"), str):
        return delta["thinking"]
    return ""


def _consume_native_stream_event(
    data: dict[str, Any],
    active: dict[int, dict[str, Any]],
    completed: list[dict[str, Any]],
) -> None:
    """累积 Anthropic tool_use 块，且不向 UI 暴露参数增量。"""
    event_type = data.get("type")
    index = data.get("index")
    if not isinstance(index, int):
        return
    if event_type == "content_block_start":
        block = data.get("content_block")
        if isinstance(block, dict) and block.get("type") == "tool_use":
            active[index] = {
                "type": "tool_use",
                "id": block.get("id"),
                "name": block.get("name"),
                "input": block.get("input") if isinstance(block.get("input"), dict) else {},
                "_partial_json": "",
            }
        return
    if event_type == "content_block_delta" and index in active:
        delta = data.get("delta")
        if isinstance(delta, dict) and delta.get("type") == "input_json_delta":
            partial = delta.get("partial_json")
            if isinstance(partial, str):
                active[index]["_partial_json"] += partial
        return
    if event_type != "content_block_stop" or index not in active:
        return
    block = active.pop(index)
    partial = block.pop("_partial_json", "")
    if partial:
        try:
            parsed = json.loads(partial)
        except ValueError:
            block["input"] = partial
        else:
            block["input"] = parsed
    completed.append(block)


def _extract_stream_usage(data: dict[str, Any]) -> dict[str, Any]:
    direct = data.get("usage")
    if isinstance(direct, dict):
        return dict(direct)
    message = data.get("message")
    nested = message.get("usage") if isinstance(message, dict) else None
    return dict(nested) if isinstance(nested, dict) else {}

anthropic_compatible_chat.prepare = prepare_anthropic_compatible_chat  # type: ignore[attr-defined]
