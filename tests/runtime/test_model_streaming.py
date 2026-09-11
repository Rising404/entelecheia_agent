from __future__ import annotations

import json

from personagraph.model_io import gateway as models


class _StreamResponse:
    def __init__(self, payloads: list[dict]):
        self.payloads = payloads

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def raise_for_status(self):
        return None

    def iter_lines(self):
        for payload in self.payloads:
            yield "event: content_block_delta"
            yield "data: " + json.dumps(payload, ensure_ascii=False)
            yield ""


class _StreamClient:
    def __init__(self, response: _StreamResponse, calls: list[dict]):
        self.response = response
        self.calls = calls

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def stream(self, method, url, **kwargs):
        self.calls.append({"method": method, "url": url, **kwargs})
        return self.response


def _configure(monkeypatch, payloads: list[dict]) -> list[dict]:
    monkeypatch.setenv("PERSONAGRAPH_API_KEY", "test-key")
    monkeypatch.setenv("PERSONAGRAPH_BASE_URL", "https://example.test/anthropic")
    monkeypatch.setenv("PERSONAGRAPH_MODEL", "test-model")
    calls: list[dict] = []
    response = _StreamResponse(payloads)
    monkeypatch.setattr(models.httpx, "Client", lambda *args, **kwargs: _StreamClient(response, calls))
    return calls


def test_provider_stream_emits_deltas_and_retains_usage(monkeypatch):
    calls = _configure(monkeypatch, [
        {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "逐字"}},
        {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "回复"}},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 2}},
    ])
    events: list[dict] = []

    with models.stream_events(events.append):
        with models.stream_generation("g1"):
            result = models.anthropic_compatible_chat([{"role": "user", "content": "hi"}], stream=True)
            models.complete_stream_generation()

    assert "json" not in calls[0]
    assert json.loads(calls[0]["content"])["stream"] is True
    assert result.reply == "逐字回复"
    assert result.output_tokens == 2
    assert result.finish_reason == "end_turn"
    assert [item["event"] for item in events] == ["answer_start", "delta", "delta", "answer_complete"]
    assert "".join(item.get("text", "") for item in events if item["event"] == "delta") == "逐字回复"


def test_tool_call_stream_is_never_released_as_provisional_chat_text(monkeypatch):
    _configure(monkeypatch, [
        {"type": "content_block_delta", "delta": {"type": "text_delta", "text": '{"tool_calls"'}},
        {"type": "content_block_delta", "delta": {"type": "text_delta", "text": ':[]}' }},
    ])
    events: list[dict] = []

    with models.stream_events(events.append):
        with models.stream_generation("tool-generation"):
            result = models.anthropic_compatible_chat([{"role": "user", "content": "hi"}], stream=True)
            models.complete_stream_generation(discard=True)

    assert result.reply == '{"tool_calls":[]}'
    assert [item["event"] for item in events] == ["answer_start", "answer_discard"]

def test_complete_tool_call_delta_is_never_released_as_provisional_chat_text(monkeypatch):
    _configure(monkeypatch, [
        {"type": "content_block_delta", "delta": {"type": "text_delta", "text": '{\n  "tool_calls": []\n}'}},
    ])
    events: list[dict] = []

    with models.stream_events(events.append):
        with models.stream_generation("complete-tool-generation"):
            result = models.anthropic_compatible_chat([{"role": "user", "content": "hi"}], stream=True)
            models.complete_stream_generation(discard=True)

    assert result.reply == '{\n  "tool_calls": []\n}'
    assert [item["event"] for item in events] == ["answer_start", "answer_discard"]
