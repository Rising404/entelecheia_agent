from personagraph.model_io.anthropic import normalize_anthropic_output
from personagraph.model_io.contracts import AssistantText, ProtocolError, ToolCallBatch
from personagraph.model_io.prompt_json import normalize_prompt_json_output


def test_prompt_json_normalizes_valid_calls_with_stable_ids():
    text = '{"tool_calls":[{"tool":"file_list","args":{"path":"."}}]}'
    first = normalize_prompt_json_output(text)
    second = normalize_prompt_json_output(text)
    assert isinstance(first, ToolCallBatch)
    assert first.calls[0].call_id == second.calls[0].call_id
    assert first.calls[0].transport == "prompt_json"


def test_prompt_json_does_not_silently_turn_invalid_arguments_into_empty_object():
    output = normalize_prompt_json_output('{"tool_calls":[{"tool":"file_list","args":"."}]}')
    assert isinstance(output, ProtocolError)
    assert output.code == "invalid_tool_arguments"


def test_malformed_tool_candidate_is_protocol_error_but_unrelated_json_is_text():
    assert isinstance(normalize_prompt_json_output('{"tool_calls":['), ProtocolError)
    assert isinstance(normalize_prompt_json_output('{"answer": 1}'), AssistantText)


def test_anthropic_mixed_text_and_tool_use_normalizes_to_one_batch():
    output = normalize_anthropic_output([
        {"type": "text", "text": "我先检查。"},
        {"type": "tool_use", "id": "toolu_1", "name": "file_list", "input": {"path": "."}},
    ])
    assert isinstance(output, ToolCallBatch)
    assert output.assistant_text == "我先检查。"
    assert output.calls[0].call_id == "toolu_1"
    assert output.calls[0].transport == "native"


def test_anthropic_invalid_tool_input_is_protocol_error():
    output = normalize_anthropic_output([
        {"type": "tool_use", "id": "toolu_1", "name": "file_list", "input": "bad"},
    ])
    assert isinstance(output, ProtocolError)
    assert output.code == "invalid_native_tool_arguments"
