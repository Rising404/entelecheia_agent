import json
from pathlib import Path

import pytest

from personagraph.model_io import gateway as models
from personagraph.model_io.contracts import AssistantText, ProtocolError, ToolCallBatch
from personagraph.model_io.tier_bindings import (
    EndpointOrigin,
    ModelTierBinding,
    ModelTier,
)


FIXTURES = Path(__file__).parents[1] / "fixtures" / "model_io"


@pytest.fixture(autouse=True)
def _explicit_test_model(monkeypatch):
    monkeypatch.setenv("PERSONAGRAPH_MODEL", "test-model")


def _l1_binding(
    *,
    provider: str,
    base_url: str,
    model: str,
    thinking_enabled: bool = False,
    request_dialect: str = "auto",
) -> ModelTierBinding:
    return ModelTierBinding(
        tier=ModelTier.L1,
        provider=provider,
        base_url=base_url,
        model=model,
        api_key="tier-secret",
        thinking_enabled=thinking_enabled,
        origin=EndpointOrigin.PROFILE,
        profile_id="profile-l1",
        profile_name="L1",
        request_dialect=request_dialect,
    )


class _Response:
    def __init__(self, data):
        self.data = data

    def raise_for_status(self): return None
    def json(self): return self.data


class _Client:
    def __init__(self, response, calls): self.response, self.calls = response, calls
    def __enter__(self): return self
    def __exit__(self, *_args): return False
    def post(self, url, **kwargs):
        self.calls.append({"url": url, **kwargs})
        return self.response


def _wire_body(call):
    assert "json" not in call
    assert isinstance(call["content"], bytes)
    return json.loads(call["content"])


def test_raw_anthropic_without_model_fails_before_http(monkeypatch):
    monkeypatch.setenv("PERSONAGRAPH_MODEL_PROVIDER", "anthropic")
    monkeypatch.setenv("PERSONAGRAPH_API_KEY", "test")
    monkeypatch.delenv("PERSONAGRAPH_BASE_URL", raising=False)
    monkeypatch.delenv("PERSONAGRAPH_MODEL", raising=False)

    def should_not_connect(**_kwargs):
        raise AssertionError("HTTP client must not be opened")

    monkeypatch.setattr(models.httpx, "Client", should_not_connect)

    with pytest.raises(models.ModelGatewayError) as raised:
        models.anthropic_compatible_chat([{"role": "user", "content": "hi"}])

    assert raised.value.retryable is False
    assert raised.value.details["reason"] == "missing_model"


def test_raw_anthropic_with_explicit_model_uses_native_default_endpoint(
    monkeypatch,
):
    data = {
        "content": [{"type": "text", "text": "ok"}],
        "stop_reason": "end_turn",
        "usage": {},
    }
    calls = []
    monkeypatch.setenv("PERSONAGRAPH_MODEL_PROVIDER", "anthropic")
    monkeypatch.setenv("PERSONAGRAPH_API_KEY", "test")
    monkeypatch.setenv("PERSONAGRAPH_MODEL", "claude-explicit")
    monkeypatch.delenv("PERSONAGRAPH_BASE_URL", raising=False)
    monkeypatch.setattr(
        models.httpx,
        "Client",
        lambda **_kwargs: _Client(_Response(data), calls),
    )

    models.anthropic_compatible_chat([{"role": "user", "content": "hi"}])

    assert calls[0]["url"] == "https://api.anthropic.com/v1/messages"
    assert _wire_body(calls[0])["model"] == "claude-explicit"
    assert "temperature" not in _wire_body(calls[0])


def test_invalid_environment_dialect_is_a_typed_preflight_failure(monkeypatch):
    monkeypatch.setenv("PERSONAGRAPH_MODEL_PROVIDER", "anthropic")
    monkeypatch.setenv("PERSONAGRAPH_API_KEY", "test")
    monkeypatch.setenv("PERSONAGRAPH_MODEL", "claude-explicit")
    monkeypatch.setenv("PERSONAGRAPH_REQUEST_DIALECT", "openai")

    def should_not_connect(**_kwargs):
        raise AssertionError("HTTP client must not be opened")

    monkeypatch.setattr(models.httpx, "Client", should_not_connect)

    with pytest.raises(models.ModelGatewayError) as raised:
        models.anthropic_compatible_chat([{"role": "user", "content": "hi"}])

    assert raised.value.retryable is False
    assert raised.value.details["reason"] == "invalid_request_dialect"


def test_nonstream_native_response_is_typed_and_request_contains_tools(monkeypatch):
    data = json.loads((FIXTURES / "anthropic_mixed_text_tool.json").read_text(encoding="utf-8"))
    calls = []
    monkeypatch.setenv("PERSONAGRAPH_API_KEY", "test")
    monkeypatch.setenv("PERSONAGRAPH_BASE_URL", "https://api.anthropic.com")
    monkeypatch.setattr(models.httpx, "Client", lambda **_kwargs: _Client(_Response(data), calls))
    result = models.anthropic_compatible_chat(
        [{"role": "user", "content": "list"}],
        control_transport="native",
        tools=[{"id": "file_list", "description": "List", "input_schema": {"type": "object"}}],
    )
    assert isinstance(result.output, ToolCallBatch)
    assert result.reply == "我先检查目录。"
    assert _wire_body(calls[0])["tools"][0]["name"] == "file_list"


def test_nonstream_prompt_json_is_typed_only_when_tools_are_in_scope(monkeypatch):
    data = {
        "content": [{"type": "text", "text": '{"tool_calls":[{"tool":"file_list","args":{}}]}'}],
        "stop_reason": "end_turn",
        "usage": {},
    }
    calls = []
    monkeypatch.setenv("PERSONAGRAPH_API_KEY", "test")
    monkeypatch.setenv("PERSONAGRAPH_BASE_URL", "https://api.deepseek.com/anthropic")
    monkeypatch.setattr(models.httpx, "Client", lambda **_kwargs: _Client(_Response(data), calls))
    tool_result = models.anthropic_compatible_chat(
        [{"role": "user", "content": "list"}],
        control_transport="prompt_json",
        tools=[{"id": "file_list", "description": "List", "input_schema": {"type": "object"}}],
    )
    text_result = models.anthropic_compatible_chat(
        [{"role": "user", "content": "structured auxiliary call"}],
        control_transport="prompt_json",
        tools=None,
    )
    assert isinstance(tool_result.output, ToolCallBatch)
    assert isinstance(text_result.output, AssistantText)
    assert "tools" not in _wire_body(calls[0])


def test_prompt_json_repair_can_request_json_object_only_when_explicitly_forced(monkeypatch):
    data = {
        "content": [{"type": "text", "text": '{"tool_calls":[{"tool":"file_list","args":{}}]}'}],
        "stop_reason": "end_turn",
        "usage": {},
    }
    calls = []
    monkeypatch.setenv("PERSONAGRAPH_API_KEY", "test")
    # ``response_format`` 是 DeepSeek 扩展，不属于通用或原生 Anthropic Messages 契约。
    monkeypatch.setenv("PERSONAGRAPH_BASE_URL", "https://api.deepseek.com/anthropic")
    monkeypatch.setattr(models.httpx, "Client", lambda **_kwargs: _Client(_Response(data), calls))
    tools = [{"id": "file_list", "description": "List", "input_schema": {"type": "object"}}]

    ordinary = models.anthropic_compatible_chat(
        [{"role": "user", "content": "list"}],
        control_transport="prompt_json",
        tools=tools,
    )
    repaired = models.anthropic_compatible_chat(
        [{"role": "user", "content": "emit the required tool call"}],
        control_transport="prompt_json",
        tools=tools,
        force_prompt_json=True,
    )

    assert isinstance(ordinary.output, ToolCallBatch)
    assert isinstance(repaired.output, ToolCallBatch)
    assert "response_format" not in _wire_body(calls[0])
    assert _wire_body(calls[1])["response_format"] == {"type": "json_object"}


def test_chat_reserves_the_profiled_output_budget_only_for_prompt_json_repair(monkeypatch):
    captured = []

    def fake_completion(*_args, **kwargs):
        captured.append(kwargs)
        return models.ModelResult(reply="ok", provider="fake", model="fake", latency_ms=1)

    # provider 走 env，而不是给 models.get_setting 打桩：chat() 现在问的是
    # active_provider()，它在 app_settings 里读设置并把 deepseek 这类旧名字
    # 归一成 anthropic-compatible。设 env 才能让那条归一化路径真的跑一遍。
    monkeypatch.setenv("PERSONAGRAPH_MODEL_PROVIDER", "deepseek")
    monkeypatch.setenv("PERSONAGRAPH_API_KEY", "must-not-be-used-or-recorded")
    monkeypatch.setattr(models, "anthropic_compatible_chat", fake_completion)
    common = {
        "model_tools": [{"id": "file_list"}],
        "control_transport": "prompt_json",
    }

    models.chat("list", [{"role": "user", "content": "list"}], **common)
    models.chat(
        "repair", [{"role": "user", "content": "repair"}],
        **common,
        force_prompt_json=True,
    )

    assert captured[0]["max_tokens"] is None
    assert captured[1]["max_tokens"] == models.PROMPT_JSON_REPAIR_MAX_TOKENS
    assert captured[0]["force_prompt_json"] is False
    assert captured[1]["force_prompt_json"] is True


def test_openai_chat_reserves_output_budget_only_for_prompt_json_repair(monkeypatch):
    captured = []

    def fake_completion(*_args, **kwargs):
        captured.append(kwargs)
        return models.ModelResult(reply="ok", provider="fake", model="fake", latency_ms=1)

    monkeypatch.setenv("PERSONAGRAPH_MODEL_PROVIDER", "openai-compatible")
    monkeypatch.setenv("PERSONAGRAPH_API_KEY", "must-not-be-used-or-recorded")
    monkeypatch.setattr(models, "openai_compatible_chat", fake_completion)
    common = {
        "model_tools": [{"id": "file_list"}],
        "control_transport": "prompt_json",
    }

    models.chat("list", [{"role": "user", "content": "list"}], **common)
    models.chat(
        "repair", [{"role": "user", "content": "repair"}],
        **common,
        force_prompt_json=True,
    )

    assert captured[0]["max_tokens"] is None
    assert captured[1]["max_tokens"] == models.PROMPT_JSON_REPAIR_MAX_TOKENS
    assert captured[0]["force_prompt_json"] is False
    assert captured[1]["force_prompt_json"] is True


def test_structured_openai_call_uses_the_exact_tier_binding(monkeypatch):
    captured = {}

    def request(endpoint, admitted_request, api_key, timeout_s):
        captured.update(
            endpoint=endpoint,
            payload=json.loads(
                admitted_request.admitted_request.body_for_dispatch()
            ),
            api_key=api_key,
            timeout_s=timeout_s,
        )
        return {
            "choices": [
                {"message": {"content": '{"ok":true}'}, "finish_reason": "stop"}
            ],
            "usage": {},
        }

    monkeypatch.setattr(models, "_openai_request", request)
    binding = _l1_binding(
        provider="openai-compatible",
        base_url="https://tier.example.test/v1",
        model="tier-openai-model",
    )

    result = models.complete_structured(
        "system",
        "user",
        mock_payload={"ok": True},
        binding=binding,
        json_mode=True,
    )

    assert result.provider == "openai-compatible"
    assert result.model == "tier-openai-model"
    assert captured["endpoint"] == "https://tier.example.test/v1/chat/completions"
    assert captured["payload"]["model"] == "tier-openai-model"
    assert captured["api_key"] == "tier-secret"


def test_structured_anthropic_call_uses_tier_endpoint_and_thinking(monkeypatch):
    data = {
        "content": [{"type": "text", "text": '{"ok":true}'}],
        "stop_reason": "end_turn",
        "usage": {},
    }
    calls = []
    monkeypatch.setattr(
        models.httpx,
        "Client",
        lambda **_kwargs: _Client(_Response(data), calls),
    )
    binding = _l1_binding(
        provider="anthropic-compatible",
        base_url="https://tier.example.test/anthropic",
        model="tier-anthropic-model",
        thinking_enabled=True,
        request_dialect="deepseek-anthropic",
    )

    result = models.complete_structured(
        "system",
        "user",
        mock_payload={"ok": True},
        binding=binding,
        json_mode=True,
    )

    assert result.provider == "anthropic-compatible"
    assert result.model == "tier-anthropic-model"
    assert calls[0]["url"] == "https://tier.example.test/anthropic/v1/messages"
    assert _wire_body(calls[0])["model"] == "tier-anthropic-model"
    assert _wire_body(calls[0])["thinking"] == {"type": "enabled"}
    assert calls[0]["headers"]["x-api-key"] == "tier-secret"


def test_native_anthropic_binding_uses_adaptive_controls_and_no_openai_json_field(
    monkeypatch,
):
    data = {
        "content": [{"type": "text", "text": '{"ok":true}'}],
        "stop_reason": "end_turn",
        "usage": {},
    }
    calls = []
    monkeypatch.setattr(
        models.httpx,
        "Client",
        lambda **_kwargs: _Client(_Response(data), calls),
    )
    binding = _l1_binding(
        provider="anthropic-compatible",
        base_url="https://api.anthropic.com",
        model="claude-sonnet",
        thinking_enabled=True,
        request_dialect="anthropic-native",
    )

    models.complete_structured(
        "system",
        "user",
        mock_payload={"ok": True},
        binding=binding,
        json_mode=True,
    )

    body = _wire_body(calls[0])
    assert body["thinking"] == {"type": "adaptive"}
    assert "temperature" not in body
    assert "response_format" not in body


class _StreamResponse:
    def __init__(self, payloads): self.payloads = payloads
    def __enter__(self): return self
    def __exit__(self, *_args): return False
    def raise_for_status(self): return None
    def iter_lines(self):
        for payload in self.payloads:
            yield "data: " + json.dumps(payload)


class _StreamClient:
    def __init__(self, payloads): self.payloads = payloads
    def __enter__(self): return self
    def __exit__(self, *_args): return False
    def stream(self, *_args, **_kwargs): return _StreamResponse(self.payloads)


def test_truncated_native_stream_is_protocol_error_without_fallback_replay(monkeypatch):
    fixture = json.loads((FIXTURES / "anthropic_truncated_stream.json").read_text(encoding="utf-8"))
    monkeypatch.setenv("PERSONAGRAPH_API_KEY", "test")
    monkeypatch.setenv("PERSONAGRAPH_BASE_URL", "https://api.anthropic.com")
    monkeypatch.setattr(models.httpx, "Client", lambda **_kwargs: _StreamClient(fixture["events"]))
    result = models.anthropic_compatible_chat(
        [{"role": "user", "content": "list"}],
        stream=True,
        control_transport="native",
        tools=[{"id": "file_list", "description": "List", "input_schema": {"type": "object"}}],
    )
    assert isinstance(result.output, ProtocolError)
    assert result.output.code == "incomplete_native_tool_call"
    assert result.finish_reason == "max_tokens"
