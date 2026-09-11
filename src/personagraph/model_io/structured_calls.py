"""有界结构化模型调用、修复对话与准备阶段 API。"""

from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from dataclasses import replace
from typing import Any, Callable

from ..configuration.app_settings import (
    ANTHROPIC_COMPATIBLE,
    OPENAI_COMPATIBLE,
)
from .gateway_core import (
    DEFAULT_MODEL_TIMEOUT_S,
    ModelGatewayError,
    ModelResult,
    PreparedModelCall,
)
from .provider_anthropic import prepare_anthropic_compatible_chat
from .provider_openai import prepare_openai_compatible_chat
from .provider_support import _elapsed_ms, _estimate_tokens
from .tier_bindings import ModelTierBinding


_STRUCTURED_REPAIR_ROLES = ("system", "user", "assistant", "user")


def _gateway_active_provider() -> str:
    from . import gateway

    return gateway.active_provider()


def build_structured_repair_messages(
    system_prompt: str,
    user_content: str,
    *,
    rejected_response_text: str,
    repair_user_content: str,
) -> list[dict[str, str]]:
    """构建唯一允许的整段响应修复对话。

    原始请求仍是冻结的前两条消息。最新被拒绝的 Provider 输出恰好以一条 assistant 消息
    表示，之后紧跟一条由 Host 编写的修复指令。

    四条消息重新生成完整响应，不累积所有失败输出，也不是在原 JSON 上打 patch。
    发送内容仍含精确被拒正文；trajectory 的对应 assistant Part 可按 hash 另行投影。
    """

    return _validated_structured_repair_messages(
        system_prompt,
        user_content,
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": rejected_response_text},
            {"role": "user", "content": repair_user_content},
        ],
    )


def _validated_structured_repair_messages(
    system_prompt: str,
    user_content: str,
    repair_messages: list[dict[str, str]],
) -> list[dict[str, str]]:
    if not isinstance(system_prompt, str) or not isinstance(user_content, str):
        raise TypeError("structured system and user content must be text")
    if not isinstance(repair_messages, list):
        raise TypeError("repair_messages must be a list")
    if len(repair_messages) != len(_STRUCTURED_REPAIR_ROLES):
        raise ValueError("repair_messages must contain exactly four messages")

    frozen: list[dict[str, str]] = []
    for index, (message, required_role) in enumerate(
        zip(repair_messages, _STRUCTURED_REPAIR_ROLES, strict=True)
    ):
        if not isinstance(message, dict):
            raise TypeError(f"repair_messages[{index}] must be a dict")
        if set(message) != {"role", "content"}:
            raise ValueError(
                f"repair_messages[{index}] must contain only role and content"
            )
        role = message["role"]
        content = message["content"]
        if role != required_role:
            raise ValueError(
                "repair_messages roles must be exactly "
                "[system, user, assistant, user]"
            )
        if not isinstance(content, str):
            raise TypeError(f"repair_messages[{index}].content must be text")
        frozen.append({"role": role, "content": content})

    if frozen[0]["content"] != system_prompt:
        raise ValueError("repair system message must equal the original system prompt")
    if frozen[1]["content"] != user_content:
        raise ValueError("repair user message must equal the original user content")
    if not frozen[3]["content"].strip():
        raise ValueError("repair instruction must not be blank")
    return frozen


def complete_structured(
    system_prompt: str,
    user_content: str,
    *,
    mock_payload: dict[str, Any] | Callable[[], dict[str, Any]],
    max_tokens: int = 1200,
    timeout_s: float = DEFAULT_MODEL_TIMEOUT_S,
    json_mode: bool = False,
    model_call_id: str | None = None,
    purpose: str = "structured_auxiliary",
    binding: ModelTierBinding | None = None,
    repair_messages: list[dict[str, str]] | None = None,
) -> ModelResult:
    """执行一次有界结构化辅助调用，并保留用量元数据。

    本函数**不**检查 ``finish_reason``。截断会在上一层
    :func:`personagraph.runtime.model_calls.request_model_with_retry` 中被捕获；该函数会在
    任何解析器看到回复之前拒绝因输出上限停止的响应。因此，新的结构化调用点必须经过该
    包装器；直接调用本函数会把已截断、甚至可能为空的正文交给 JSON 解析器。参见
    27/002 §6.4 A1。

    ``binding`` 选择某 tier 自己的端点和思考策略；若未提供，调用会与以往完全一样使用
    全局配置。

    Runtime 正常路径通过本函数挂载的 prepare 能力先准入，再由共享 retry wrapper
    控制 dispatch；本便捷函数本身只执行一次准备/分派，不推进 L1 Attempt。
    """
    return prepare_complete_structured(
        system_prompt,
        user_content,
        mock_payload=mock_payload,
        max_tokens=max_tokens,
        timeout_s=timeout_s,
        json_mode=json_mode,
        purpose=purpose,
        binding=binding,
        repair_messages=repair_messages,
    ).dispatch(model_call_id=model_call_id)


def prepare_complete_structured(
    system_prompt: str,
    user_content: str,
    *,
    mock_payload: dict[str, Any] | Callable[[], dict[str, Any]],
    max_tokens: int = 1200,
    timeout_s: float = DEFAULT_MODEL_TIMEOUT_S,
    json_mode: bool = False,
    purpose: str = "structured_auxiliary",
    binding: ModelTierBinding | None = None,
    projection_epoch: str = "provider-wire",
    projection_generation: int = 0,
    configured_input_limit_tokens: int | None = None,
    repair_messages: list[dict[str, str]] | None = None,
) -> PreparedModelCall:
    """选择 Provider 并准备结构化请求，返回尚未执行 I/O 的 PreparedModelCall。

    原始消息为 system + user；repair 必须是原 system/user + 最新被拒 assistant
    + Host 修复 user。Provider adapter 再转为各自 wire 方言并准入完整请求。
    同时标记结构化输出/被拒输入的 trajectory 投影规则；这不修改发给模型的正文。
    mock 分支只构造本地 ModelResult，不经过真实 Provider HTTP/记录边界。
    """

    provider = binding.provider if binding is not None else _gateway_active_provider()
    messages = (
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        if repair_messages is None
        else _validated_structured_repair_messages(
            system_prompt,
            user_content,
            repair_messages,
        )
    )
    trajectory_redacted_input_sha256s = (
        ()
        if repair_messages is None
        else (
            hashlib.sha256(messages[2]["content"].encode("utf-8")).hexdigest(),
        )
    )
    if provider in {"mock", ""}:
        resolved_mock_payload = (
            mock_payload() if callable(mock_payload) else mock_payload
        )
        if not isinstance(resolved_mock_payload, dict):
            raise TypeError("structured mock payload factory must return a dict")
        reply = json.dumps(
            resolved_mock_payload,
            ensure_ascii=False,
            sort_keys=True,
        )

        def _dispatch_mock(model_call_id: str | None) -> ModelResult:
            start = time.perf_counter()
            return ModelResult(
                reply=reply,
                provider="mock",
                model="mock-structured",
                latency_ms=_elapsed_ms(start),
                input_tokens=_estimate_tokens(
                    "\n".join(message["content"] for message in messages)
                ),
                output_tokens=_estimate_tokens(reply),
                finish_reason="end_turn",
                model_call_id=model_call_id or str(uuid.uuid4()),
                purpose=purpose,
            )

        return PreparedModelCall(
            _dispatch=_dispatch_mock,
            trajectory_redacted_input_sha256s=(
                trajectory_redacted_input_sha256s
            ),
            trajectory_redact_reply=True,
        )
    if provider == ANTHROPIC_COMPATIBLE:
        provider_json_mode = json_mode or (
            os.getenv("PERSONAGRAPH_JSON_MODE", "0").strip().lower()
            in {"1", "true", "yes", "on"}
        )
        prepared = prepare_anthropic_compatible_chat(
            messages,
            temperature=0.0,
            max_tokens=max_tokens,
            json_mode=provider_json_mode,
            timeout_s=timeout_s,
            purpose=purpose,
            binding=binding,
            projection_epoch=projection_epoch,
            projection_generation=projection_generation,
            configured_input_limit_tokens=configured_input_limit_tokens,
        )
        return replace(
            prepared,
            trajectory_redacted_input_sha256s=(
                trajectory_redacted_input_sha256s
            ),
            trajectory_redact_reply=True,
        )

    if provider == OPENAI_COMPATIBLE:
        prepared = prepare_openai_compatible_chat(
            messages,
            temperature=0.0,
            max_tokens=max_tokens,
            json_mode=True,
            timeout_s=timeout_s,
            purpose=purpose,
            binding=binding,
            projection_epoch=projection_epoch,
            projection_generation=projection_generation,
            configured_input_limit_tokens=configured_input_limit_tokens,
        )
        return replace(
            prepared,
            trajectory_redacted_input_sha256s=(
                trajectory_redacted_input_sha256s
            ),
            trajectory_redact_reply=True,
        )
    raise ModelGatewayError(
        "MODEL_CALL_FAILED",
        "Unsupported PERSONAGRAPH_MODEL_PROVIDER.",
        retryable=False,
        details={"provider": provider, "reason": "unsupported_provider"},
    )

complete_structured.prepare = prepare_complete_structured  # type: ignore[attr-defined]
