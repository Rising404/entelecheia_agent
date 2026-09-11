from __future__ import annotations

from typing import Any

from .contracts import AssistantText, ModelTurnOutput, ProtocolError, ToolCall, ToolCallBatch


def normalize_anthropic_output(content: object) -> ModelTurnOutput:
    if not isinstance(content, list):
        return ProtocolError(
            code="invalid_native_content",
            message="Anthropic content must be a list of typed blocks.",
            transport="native",
        )
    text_parts: list[str] = []
    calls: list[ToolCall] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if block_type == "text" and isinstance(block.get("text"), str):
            text_parts.append(block["text"])
            continue
        if block_type != "tool_use":
            continue
        call_id = block.get("id")
        name = block.get("name")
        arguments = block.get("input")
        if not isinstance(call_id, str) or not call_id or not isinstance(name, str) or not name:
            return ProtocolError(
                code="invalid_native_tool_call",
                message="A native tool_use block was missing id or name.",
                transport="native",
            )
        if not isinstance(arguments, dict):
            return ProtocolError(
                code="invalid_native_tool_arguments",
                message=f"Native tool call {call_id} input must be an object.",
                transport="native",
            )
        calls.append(ToolCall(call_id=call_id, tool_name=name, arguments=arguments, transport="native"))
    text = "\n".join(part for part in text_parts if part).strip()
    if calls:
        if len(calls) > 3:
            return ProtocolError(
                code="too_many_tool_calls",
                message="A model turn may request at most three tool calls.",
                transport="native",
            )
        return ToolCallBatch(calls=calls, assistant_text=text or None)
    if text_parts:
        return AssistantText(text=text)
    return ProtocolError(
        code="empty_native_content",
        message="Anthropic response contained neither text nor tool calls.",
        transport="native",
        retryable=False,
    )


def anthropic_tool_definitions(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    definitions: list[dict[str, Any]] = []
    for tool in tools:
        tool_id = tool.get("id")
        if not isinstance(tool_id, str) or not tool_id:
            continue
        definitions.append(
            {
                "name": tool_id,
                "description": str(tool.get("description") or tool_id),
                "input_schema": tool.get("input_schema") if isinstance(tool.get("input_schema"), dict) else {"type": "object"},
            }
        )
    return definitions


def assistant_message_content(output: ToolCallBatch) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    if output.assistant_text:
        blocks.append({"type": "text", "text": output.assistant_text})
    blocks.extend(
        {
            "type": "tool_use",
            "id": call.call_id,
            "name": call.tool_name,
            "input": call.arguments,
        }
        for call in output.calls
    )
    return blocks


def image_content_block(payload: bytes, media_type: str) -> dict[str, Any]:
    """构建一个 Anthropic 风格 base64 图像块。

    网关会原样转发消息 ``content``，因此线上已接受块列表；此处只负责块形状。
    """
    import base64

    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": media_type,
            "data": base64.b64encode(payload).decode("ascii"),
        },
    }


def text_content_block(text: str) -> dict[str, Any]:
    return {"type": "text", "text": text}
