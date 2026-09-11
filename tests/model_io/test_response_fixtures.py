import json
from pathlib import Path

from personagraph.model_io.anthropic import normalize_anthropic_output
from personagraph.model_io.contracts import ToolCallBatch
from personagraph.model_io.prompt_json import normalize_prompt_json_output


FIXTURES = Path(__file__).parents[1] / "fixtures" / "model_io"


def _load(name):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def test_frozen_prompt_json_fixture():
    fixture = _load("prompt_json_tool_call.json")
    output = normalize_prompt_json_output(fixture["raw_text"])
    assert output.kind == fixture["expected_kind"]
    assert len(output.calls) == fixture["expected_calls"]


def test_frozen_native_tool_and_mixed_fixtures():
    tool = normalize_anthropic_output(_load("anthropic_tool_use.json")["content"])
    mixed = normalize_anthropic_output(_load("anthropic_mixed_text_tool.json")["content"])
    assert isinstance(tool, ToolCallBatch)
    assert isinstance(mixed, ToolCallBatch)
    assert mixed.assistant_text == "我先检查目录。"


def test_frozen_parallel_native_fixture_preserves_provider_ids():
    output = normalize_anthropic_output(_load("anthropic_parallel_tools.json")["content"])
    assert isinstance(output, ToolCallBatch)
    assert [call.call_id for call in output.calls] == ["toolu_a", "toolu_b"]
