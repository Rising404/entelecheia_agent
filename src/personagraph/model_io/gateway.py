"""供应商中立的模型调用入口。

请求合同与队列控制、供应商线上适配、流事件、结构化调用和 trajectory 投影分别由
相邻模块拥有；本模块只组合这些能力，并保留测试所需的模块级注入接缝。
"""

from __future__ import annotations

import os
import time
import uuid
from typing import Any

import httpx

from ..configuration.app_settings import (
    ANTHROPIC_COMPATIBLE,
    OPENAI_COMPATIBLE,
    active_provider,
    get_setting,  # noqa: F401
)
from ..trajectory import record_model_call  # noqa: F401
from .contracts import AssistantText
from .gateway_core import (
    DEFAULT_MODEL_TIMEOUT_S,  # noqa: F401
    ModelGatewayError,
    ModelResult,
    PreparedModelCall,  # noqa: F401
    _effective_model_timeout_s,
    _is_provider_rate_limit,
    _prepare_model_api_quota,
    _provider_retry_after_seconds,
    _quota_for_endpoint,
    model_http_timeout_ceiling,  # noqa: F401
)
from .generation_stream import (
    complete_stream_generation,
    emit_stream_delta,
    stream_events,
    stream_generation,
    streaming_generation_active,
)
from .provider_anthropic import (
    _consume_native_stream_event,
    _dispatch_anthropic_compatible_chat,
    _extract_anthropic_text,
    _extract_stream_text,
    _extract_stream_thinking,
    _extract_stream_usage,
    _stream_anthropic_compatible_chat,
    anthropic_compatible_chat,
    prepare_anthropic_compatible_chat,
)
from .provider_openai import (
    _dispatch_openai_compatible_chat,
    _extract_openai_reasoning,
    _extract_openai_text,
    _openai_message_content,
    _openai_messages,
    _openai_request,  # noqa: F401
    _stream_openai_compatible_chat,
    openai_compatible_chat,
    prepare_openai_compatible_chat,
)
from .provider_support import (
    _elapsed_ms,
    _estimate_tokens,
    _extract_finish_reason,
    _message_text,
    _redact_endpoint,
    _require_endpoint_configuration,
    _resolved_request_dialect,
    _safe_provider_error_details,
    _safe_retry_after_seconds,
)
from .structured_calls import (
    build_structured_repair_messages,
    complete_structured,  # noqa: F401
    prepare_complete_structured,  # noqa: F401
)
from .trajectory_projection import (
    _admitted_payload_for_recording,
    _redact_owned_rejected_outputs_from_trajectory_payload,
    _trajectory_redacted_text_reference,
    _trajectory_reply_for_recording,
    _trajectory_response_for_recording,
)


PROMPT_JSON_REPAIR_MAX_TOKENS = 1024


def chat(
    user_input: str,
    messages: list[dict[str, Any]],
    *,
    model_tools: list[dict[str, Any]] | None = None,
    control_transport: str = "prompt_json",
    force_prompt_json: bool = False,
    model_call_id: str | None = None,
    purpose: str = "task_generation",
) -> ModelResult:
    provider = active_provider()
    if provider in {"mock", ""}:
        start = time.perf_counter()
        reply = mock_chat(user_input, messages)
        _emit_mock_deltas(reply)
        return ModelResult(
            reply=reply,
            provider="mock",
            model="mock-entelecheia",
            latency_ms=_elapsed_ms(start),
            input_tokens=_estimate_tokens("\n".join(_message_text(m.get("content")) for m in messages)),
            output_tokens=_estimate_tokens(reply),
            output=AssistantText(text=reply),
            model_call_id=model_call_id or str(uuid.uuid4()),
            purpose=purpose,
            control_transport=control_transport,
        )
    if provider == ANTHROPIC_COMPATIBLE:
        return anthropic_compatible_chat(
            messages,
            stream=streaming_generation_active(),
            tools=model_tools,
            control_transport=control_transport,
            force_prompt_json=force_prompt_json,
            max_tokens=(PROMPT_JSON_REPAIR_MAX_TOKENS if force_prompt_json else None),
            model_call_id=model_call_id,
            purpose=purpose,
        )

    if provider == OPENAI_COMPATIBLE:
        return openai_compatible_chat(
            messages,
            stream=streaming_generation_active(),
            tools=model_tools,
            control_transport=control_transport,
            force_prompt_json=force_prompt_json,
            max_tokens=(PROMPT_JSON_REPAIR_MAX_TOKENS if force_prompt_json else None),
            model_call_id=model_call_id,
            purpose=purpose,
        )
    raise ModelGatewayError(
        "MODEL_CALL_FAILED",
        "Unsupported PERSONAGRAPH_MODEL_PROVIDER.",
        retryable=False,
        details={"provider": provider, "reason": "unsupported_provider"},
    )


def complete_json(system_prompt: str, user_content: str) -> str:
    """通用单轮补全，用于辅助任务（如记忆抽取）。

    mock 返回确定性 no-op JSON，保证离线测试不会写入记忆、也不产生真实调用。
    真实 provider 走 anthropic-compatible，返回模型文本（期望是 JSON）。
    """
    provider = active_provider()
    if provider in {"mock", ""}:
        return '{"should_store": false, "items": []}'
    if provider == ANTHROPIC_COMPATIBLE:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        json_mode = os.getenv("PERSONAGRAPH_JSON_MODE", "0").strip().lower() in {"1", "true", "yes", "on"}
        # 抽取是结构化任务：固定 temperature=0 去随机，独立于对话温度
        return anthropic_compatible_chat(messages, temperature=0.0, json_mode=json_mode).reply

    if provider == OPENAI_COMPATIBLE:
        return openai_compatible_chat(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=0.0,
            json_mode=True,
        ).reply
    raise ModelGatewayError(
        "MODEL_CALL_FAILED",
        "Unsupported PERSONAGRAPH_MODEL_PROVIDER.",
        retryable=False,
        details={"provider": provider, "reason": "unsupported_provider"},
    )


def complete_text(system_prompt: str, user_content: str) -> str:
    """通用单轮文本补全（用于滚动摘要等）。mock 返回确定性占位摘要，离线不真实调用。"""
    provider = active_provider()
    if provider in {"mock", ""}:
        return "[mock-summary] " + _short_echo(user_content)
    if provider == ANTHROPIC_COMPATIBLE:
        return anthropic_compatible_chat(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=0.0,
        ).reply

    if provider == OPENAI_COMPATIBLE:
        return openai_compatible_chat(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=0.3,
        ).reply
    raise ModelGatewayError(
        "MODEL_CALL_FAILED",
        "Unsupported PERSONAGRAPH_MODEL_PROVIDER.",
        retryable=False,
        details={"provider": provider, "reason": "unsupported_provider"},
    )


def mock_chat(
    user_input: str,
    messages: list[dict[str, str]] | None = None,
) -> str:
    """离线确定性桩，仅用于测试/无网调试。

    只回显受限长度的用户输入，不解析角色卡或添加角色前缀。
    """
    del messages
    return f"[mock] {_short_echo(user_input)}"


def _emit_mock_deltas(reply: str) -> None:
    """让离线供应商也能用于演练相同的 UI 流式路径。"""
    for offset in range(0, len(reply), 12):
        emit_stream_delta(reply[offset:offset + 12])


def _short_echo(text: str) -> str:
    return text if len(text) <= 24 else text[:24] + "..."
