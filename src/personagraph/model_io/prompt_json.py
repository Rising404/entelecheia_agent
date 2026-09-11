from __future__ import annotations

import hashlib
import json
import re

from .contracts import AssistantText, ModelTurnOutput, ProtocolError, ToolCall, ToolCallBatch


_FENCE_RE = re.compile(r"^```[a-zA-Z]*\n?|\n?```$")
_TOOL_MARKER = '"tool_calls"'


def normalize_prompt_json_output(text: str) -> ModelTurnOutput:
    """解析旧 prompt-JSON 控制通道，不把无效控制当作正文。"""
    raw = text or ""
    candidate = _extract_candidate(raw)
    if candidate is None:
        return AssistantText(text=raw)
    try:
        payload = json.loads(candidate)
    except (TypeError, ValueError):
        return ProtocolError(
            code="malformed_tool_json",
            message="The model emitted a tool_calls candidate that was not valid JSON.",
            transport="prompt_json",
        )
    if not isinstance(payload, dict) or "tool_calls" not in payload:
        return AssistantText(text=raw)
    calls = payload.get("tool_calls")
    if not isinstance(calls, list) or not calls:
        return ProtocolError(
            code="invalid_tool_calls",
            message="tool_calls must be a non-empty list.",
            transport="prompt_json",
        )
    if len(calls) > 3:
        return ProtocolError(
            code="too_many_tool_calls",
            message="A model turn may request at most three tool calls.",
            transport="prompt_json",
        )
    normalized: list[ToolCall] = []
    digest = hashlib.sha256(candidate.encode("utf-8")).hexdigest()[:16]
    for index, call in enumerate(calls):
        if not isinstance(call, dict) or not isinstance(call.get("tool"), str) or not call["tool"].strip():
            return ProtocolError(
                code="invalid_tool_name",
                message=f"Tool call {index} has no valid tool name.",
                transport="prompt_json",
            )
        arguments = call.get("args", {})
        if not isinstance(arguments, dict):
            return ProtocolError(
                code="invalid_tool_arguments",
                message=f"Tool call {index} arguments must be an object.",
                transport="prompt_json",
            )
        call_id = call.get("id")
        if not isinstance(call_id, str) or not call_id.strip():
            call_id = f"prompt-{digest}-{index}"
        normalized.append(
            ToolCall(
                call_id=call_id,
                tool_name=call["tool"].strip(),
                arguments=arguments,
                transport="prompt_json",
            )
        )
    return ToolCallBatch(calls=normalized)


def _extract_candidate(text: str) -> str | None:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = _FENCE_RE.sub("", stripped).strip()
    if stripped.startswith("{"):
        if _TOOL_MARKER in stripped:
            return stripped
        return None
    if "\n" in stripped:
        last = stripped.splitlines()[-1].strip()
        if last.startswith("{") and _TOOL_MARKER in last:
            return last
    return None
