from __future__ import annotations

import json
from typing import Any

from .contracts import AssistantText, ModelTurnOutput, ProtocolError, ToolCall, ToolCallBatch


def openai_tool_definitions(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    definitions: list[dict[str, Any]] = []
    for tool in tools:
        tool_id = tool.get("id")
        if not isinstance(tool_id, str) or not tool_id:
            continue
        definitions.append(
            {
                "type": "function",
                "function": {
                    "name": tool_id,
                    "description": str(tool.get("description") or tool_id),
                    "parameters": (
                        tool.get("input_schema")
                        if isinstance(tool.get("input_schema"), dict)
                        else {"type": "object"}
                    ),
                },
            }
        )
    return definitions


def openai_tool_choice(choice: object) -> object:
    if not isinstance(choice, dict):
        return choice
    if choice.get("type") == "tool" and isinstance(choice.get("name"), str):
        return {
            "type": "function",
            "function": {"name": choice["name"]},
        }
    return choice


def normalize_openai_output(message: object) -> ModelTurnOutput:
    if not isinstance(message, dict):
        return ProtocolError(
            code="invalid_native_content",
            message="OpenAI response message must be an object.",
            transport="native",
        )
    content = message.get("content")
    text = content.strip() if isinstance(content, str) else ""
    raw_calls = message.get("tool_calls")
    if raw_calls is None:
        return AssistantText(text=text)
    if not isinstance(raw_calls, list):
        return ProtocolError(
            code="invalid_native_tool_call",
            message="OpenAI tool_calls must be a list.",
            transport="native",
        )
    if len(raw_calls) > 3:
        return ProtocolError(
            code="too_many_tool_calls",
            message="A model turn may request at most three tool calls.",
            transport="native",
        )
    calls: list[ToolCall] = []
    for raw in raw_calls:
        function = raw.get("function") if isinstance(raw, dict) else None
        call_id = raw.get("id") if isinstance(raw, dict) else None
        name = function.get("name") if isinstance(function, dict) else None
        encoded_arguments = (
            function.get("arguments") if isinstance(function, dict) else None
        )
        if not isinstance(call_id, str) or not call_id or not isinstance(name, str) or not name:
            return ProtocolError(
                code="invalid_native_tool_call",
                message="A native OpenAI tool call was missing id or function name.",
                transport="native",
            )
        try:
            arguments = json.loads(encoded_arguments)
        except (TypeError, ValueError):
            return ProtocolError(
                code="invalid_native_tool_arguments",
                message=f"Native tool call {call_id} arguments were not valid JSON.",
                transport="native",
            )
        if not isinstance(arguments, dict):
            return ProtocolError(
                code="invalid_native_tool_arguments",
                message=f"Native tool call {call_id} arguments must decode to an object.",
                transport="native",
            )
        calls.append(
            ToolCall(
                call_id=call_id,
                tool_name=name,
                arguments=arguments,
                transport="native",
            )
        )
    if calls:
        return ToolCallBatch(calls=calls, assistant_text=text or None)
    if text:
        return AssistantText(text=text)
    return ProtocolError(
        code="empty_native_content",
        message="OpenAI response contained neither text nor tool calls.",
        transport="native",
        retryable=False,
    )


__all__ = [
    "normalize_openai_output",
    "openai_tool_choice",
    "openai_tool_definitions",
]
