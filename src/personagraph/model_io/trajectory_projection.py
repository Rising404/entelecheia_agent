"""将实际模型请求/响应转换为 trajectory 记录视图（recording projection）。

这里改变的是记录副本，不是发送给 Provider 的 wire bytes。结构化输出与 repair
的被拒正文按身份/hash 投影；普通请求文本通常保留，因此“无正文投影”不意味着
所有 prompt 都已脱敏，更不保证任意 JSON 内嵌的检索正文都被移除。
"""

from __future__ import annotations

import hashlib
import json
from contextvars import ContextVar
from typing import Any

from .context_budgeting import PreparedAdmittedProviderRequest


_TRAJECTORY_REDACTED_INPUT_SHA256S: ContextVar[frozenset[str]] = ContextVar(
    "personagraph_trajectory_redacted_input_sha256s",
    default=frozenset(),
)
_TRAJECTORY_REDACT_PROVIDER_REPLY: ContextVar[bool] = ContextVar(
    "personagraph_trajectory_redact_provider_reply",
    default=False,
)


def _admitted_payload_for_recording(
    prepared: PreparedAdmittedProviderRequest,
) -> dict[str, Any]:
    """仅在已准入字节发送后，才物化 trajectory 输入。

    让已准备对象仅持有字节，可避免持久 attempt 等待分派时再保留一份完整 prompt 映射。
    此处解析不会创建第二份线上序列化，且 gate 会在返回正文前重新验证它。
    """

    payload = json.loads(
        prepared.admitted_request.body_for_dispatch().decode("utf-8")
    )
    if not isinstance(payload, dict):  # 最终门已经证明过这一点。
        raise RuntimeError("admitted provider payload was not a JSON object")
    _redact_owned_rejected_outputs_from_trajectory_payload(payload)
    return payload


def _trajectory_reply_for_recording(reply: str) -> str:
    """按当前分派 scope 决定 assistant reply 的记录形态。

    普通聊天原样返回；结构化调用只返回 response hash、字节数和 body owner，
    精确 Plan/Action 应去 Runtime 合同/账本读取，不能从这条 Part 还原正文。
    """

    if not _TRAJECTORY_REDACT_PROVIDER_REPLY.get():
        return reply
    return _trajectory_redacted_text_reference(
        reply,
        reference_kind="trajectory-redacted-structured-model-reply",
        body_owner="runtime_model_contract_layer",
    )


def _trajectory_response_for_recording(
    response: dict[str, Any],
) -> dict[str, Any]:
    """避免结构化隐藏推理进入重复的 trajectory。

    某些供应商会在 thinking 块内重复或重建 JSON 答案。精确结构化响应由 Runtime 契约层
    持有，因此结构化分派只记录按内容寻址的 thinking 标记，同时保留用量计数器。普通
    聊天调用保持不变。
    """

    if not _TRAJECTORY_REDACT_PROVIDER_REPLY.get():
        return response
    projected = dict(response)
    content = response.get("content")
    if not isinstance(content, list):
        projected["content"] = []
        return projected
    projected_blocks: list[dict[str, Any]] = []
    for block in content:
        if not isinstance(block, dict) or block.get("type") != "thinking":
            continue
        thinking = block.get("thinking")
        if not isinstance(thinking, str) or not thinking:
            continue
        projected_blocks.append(
            {
                "type": "thinking",
                "thinking": _trajectory_redacted_text_reference(
                    thinking,
                    reference_kind=(
                        "trajectory-redacted-structured-model-thinking"
                    ),
                    body_owner="provider_ephemeral_response",
                ),
            }
        )
    projected["content"] = projected_blocks
    return projected


def _trajectory_redacted_text_reference(
    text: str,
    *,
    reference_kind: str,
    body_owner: str,
) -> str:
    """生成被省略正文的内容身份标记；body_owner 说明归属，不是可直接读取的地址。

    hash 与 byte_count 基于完整 UTF-8 原文，标记不承诺原文一定有可访问的持久副本。
    """

    encoded = text.encode("utf-8")
    return json.dumps(
        {
            "reference_kind": reference_kind,
            "response_sha256": hashlib.sha256(encoded).hexdigest(),
            "byte_count": len(encoded),
            "body_owner": body_owner,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _redact_owned_rejected_outputs_from_trajectory_payload(
    payload: dict[str, Any],
) -> None:
    """避免已拒绝正文进入重复的 trajectory 副本。

    发送给 Provider 的已准入线上字节中仍包含精确 assistant 文本。分派期间，
    ``PreparedModelCall`` 只把其 SHA-256 限定到此记录投影中；匹配的输入 assistant 消息
    会被按内容寻址的元数据替换。Runtime 账本仍是已拒绝正文唯一的持久所有者。
    """

    redacted_sha256s = _TRAJECTORY_REDACTED_INPUT_SHA256S.get()
    if not redacted_sha256s:
        return
    messages = payload.get("messages")
    if not isinstance(messages, list):
        return
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        content = message.get("content")
        if not isinstance(content, str):
            continue
        response_sha256 = hashlib.sha256(content.encode("utf-8")).hexdigest()
        if response_sha256 not in redacted_sha256s:
            continue
        message["content"] = _trajectory_redacted_text_reference(
            content,
            reference_kind="trajectory-redacted-rejected-model-output",
            body_owner="runtime_model_rejected_output",
        )
