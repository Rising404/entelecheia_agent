import json

from personagraph.model_io import gateway as models
from personagraph.model_io.contracts import ToolCallBatch


class _Response:
    def __enter__(self): return self
    def __exit__(self, *_args): return False
    def raise_for_status(self): return None
    def iter_lines(self):
        payloads = [
            {"type": "content_block_start", "index": 0, "content_block": {"type": "tool_use", "id": "toolu_1", "name": "file_list", "input": {}}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "input_json_delta", "partial_json": '{"path":"."}'}},
            {"type": "content_block_stop", "index": 0},
            {"type": "message_delta", "delta": {"stop_reason": "tool_use"}},
        ]
        for payload in payloads:
            yield "data: " + json.dumps(payload)


class _Client:
    def __init__(self, calls): self.calls = calls
    def __enter__(self): return self
    def __exit__(self, *_args): return False
    def stream(self, method, url, **kwargs):
        self.calls.append(kwargs)
        return _Response()


def test_native_stream_accumulates_tool_input_and_sends_typed_definitions(monkeypatch):
    monkeypatch.setenv("PERSONAGRAPH_API_KEY", "test")
    monkeypatch.setenv("PERSONAGRAPH_BASE_URL", "https://api.anthropic.com")
    monkeypatch.setenv("PERSONAGRAPH_MODEL", "test-model")
    calls = []
    monkeypatch.setattr(models.httpx, "Client", lambda **_kwargs: _Client(calls))
    result = models.anthropic_compatible_chat(
        [{"role": "user", "content": "list"}],
        stream=True,
        control_transport="native",
        tools=[{"id": "file_list", "description": "List files", "input_schema": {"type": "object"}}],
    )
    assert "json" not in calls[0]
    body = json.loads(calls[0]["content"])
    assert body["stream"] is True
    assert body["tools"][0]["name"] == "file_list"
    assert isinstance(result.output, ToolCallBatch)
    assert result.output.calls[0].arguments == {"path": "."}
    assert result.reply == ""
