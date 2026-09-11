from __future__ import annotations

from personagraph.model_io.contracts import AssistantText, ProtocolError, ToolCallBatch
from personagraph.model_io.openai import (
    normalize_openai_output,
    openai_tool_choice,
    openai_tool_definitions,
)


def test_openai_tool_definitions_use_function_schema():
    assert openai_tool_definitions(
        [{"id": "file_list", "description": "List files", "input_schema": {"type": "object"}}]
    ) == [
        {
            "type": "function",
            "function": {
                "name": "file_list",
                "description": "List files",
                "parameters": {"type": "object"},
            },
        }
    ]


def test_named_tool_choice_is_converted_to_openai_shape():
    assert openai_tool_choice({"type": "tool", "name": "file_list"}) == {
        "type": "function",
        "function": {"name": "file_list"},
    }


def test_openai_native_response_normalizes_tool_calls():
    output = normalize_openai_output(
        {
            "content": "I will inspect it.",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "file_list", "arguments": '{"path":"."}'},
                }
            ],
        }
    )

    assert isinstance(output, ToolCallBatch)
    assert output.assistant_text == "I will inspect it."
    assert output.calls[0].tool_name == "file_list"
    assert output.calls[0].arguments == {"path": "."}


def test_openai_native_invalid_arguments_are_a_protocol_error():
    output = normalize_openai_output(
        {
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "file_list", "arguments": "not json"},
                }
            ]
        }
    )
    assert isinstance(output, ProtocolError)
    assert output.code == "invalid_native_tool_arguments"


def test_openai_native_plain_text_remains_plain_text():
    output = normalize_openai_output({"content": "done"})
    assert isinstance(output, AssistantText)
    assert output.text == "done"
