"""第二种协议家族，遵循与第一种相同的契约。

两种端点形态覆盖几乎所有值得配置的提供方；同时支持二者的目的，是让网关以上
的组件无法分辨由谁作答：相同的 ModelResult、相同的失败词汇、相同的记录轨迹。
"""

from __future__ import annotations

import json

import httpx
import pytest

from personagraph.model_io import gateway as models
from personagraph.model_io.tier_bindings import (
    EndpointOrigin,
    ModelTierBinding,
    ModelTier,
    ReasoningEffort,
)
from personagraph.model_io.gateway import ModelGatewayError
from personagraph.model_io.contracts import ToolCallBatch


class _Response:
    def __init__(self, data=None, *, status_exc=None, lines=None):
        self._data = data
        self._status_exc = status_exc
        self._lines = lines or []

    def raise_for_status(self):
        if self._status_exc:
            raise self._status_exc

    def json(self):
        return self._data

    def iter_lines(self):
        return iter(self._lines)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _Client:
    def __init__(self, response):
        self._response = response
        self.sent = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def post(self, endpoint, **kwargs):
        self.sent = {"endpoint": endpoint, **kwargs}
        return self._response

    def stream(self, _method, endpoint, **kwargs):
        self.sent = {"endpoint": endpoint, **kwargs}
        return self._response


@pytest.fixture
def configured(monkeypatch):
    monkeypatch.setenv("PERSONAGRAPH_MODEL_PROVIDER", "openai-compatible")
    monkeypatch.setenv("PERSONAGRAPH_API_KEY", "sk-test")
    monkeypatch.setenv("PERSONAGRAPH_BASE_URL", "https://example.test/v1")
    monkeypatch.setenv("PERSONAGRAPH_MODEL", "gpt-test")


def _answers(monkeypatch, response):
    client = _Client(response)
    monkeypatch.setattr(models.httpx, "Client", lambda *a, **k: client)
    return client


def _wire_body(client):
    assert "json" not in client.sent
    assert isinstance(client.sent["content"], bytes)
    return json.loads(client.sent["content"])


COMPLETION = {
    "choices": [{"message": {"content": "3只。"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 11, "completion_tokens": 4},
}


def _binding(*, dialect: str, effort: ReasoningEffort | None, thinking: bool = True):
    return ModelTierBinding(
        tier=ModelTier.ATTEMPT,
        provider="openai-compatible",
        base_url=(
            "https://api.openai.com/v1"
            if dialect == "openai-native"
            else "https://api.deepseek.com/v1"
        ),
        model="reasoning-model",
        api_key="sk-tier",
        thinking_enabled=thinking,
        reasoning_effort=effort,
        request_dialect=dialect,
        origin=EndpointOrigin.PROFILE,
    )


# --- 请求形状 --------------------------------------------------------------------


def test_system_travels_as_a_message_not_a_top_level_field(configured, monkeypatch):
    """正是这项差异让它成为独立适配器，而不是一个开关。"""

    client = _answers(monkeypatch, _Response(COMPLETION))
    models.openai_compatible_chat([
        {"role": "system", "content": "你是分类器"},
        {"role": "user", "content": "几只猫？"},
    ])

    body = _wire_body(client)
    assert "system" not in body
    assert body["messages"][0] == {"role": "system", "content": "你是分类器"}
    assert client.sent["endpoint"] == "https://example.test/v1/chat/completions"
    assert client.sent["headers"]["authorization"] == "Bearer sk-test"


def test_a_base_url_that_already_names_the_route_is_left_alone(configured, monkeypatch):
    monkeypatch.setenv("PERSONAGRAPH_BASE_URL", "https://example.test/v1/chat/completions")
    client = _answers(monkeypatch, _Response(COMPLETION))
    models.openai_compatible_chat([{"role": "user", "content": "hi"}])
    assert client.sent["endpoint"] == "https://example.test/v1/chat/completions"


def test_openai_native_binding_uses_native_reasoning_request_shape(monkeypatch):
    client = _answers(monkeypatch, _Response(COMPLETION))

    models.openai_compatible_chat(
        [{"role": "user", "content": "hi"}],
        max_tokens=2048,
        temperature=0.0,
        json_mode=True,
        binding=_binding(dialect="openai-native", effort=ReasoningEffort.HIGH),
    )

    body = _wire_body(client)
    assert body["max_completion_tokens"] == 2048
    assert body["reasoning_effort"] == "high"
    assert "max_tokens" not in body
    assert "temperature" not in body


def test_blank_bound_model_matches_the_openai_gateway_default(monkeypatch):
    client = _answers(monkeypatch, _Response(COMPLETION))
    binding = ModelTierBinding(
        tier=ModelTier.ATTEMPT,
        provider="openai-compatible",
        base_url="https://api.openai.com/v1",
        model="",
        api_key="sk-tier",
        thinking_enabled=False,
        origin=EndpointOrigin.PROFILE,
        request_dialect="openai-native",
    )

    models.openai_compatible_chat(
        [{"role": "user", "content": "hi"}],
        binding=binding,
    )

    assert _wire_body(client)["model"] == "gpt-4o-mini"


def test_deepseek_openai_binding_keeps_its_distinct_reasoning_shape(monkeypatch):
    client = _answers(monkeypatch, _Response(COMPLETION))

    models.openai_compatible_chat(
        [{"role": "user", "content": "hi"}],
        max_tokens=2048,
        temperature=0.0,
        binding=_binding(
            dialect="deepseek-openai",
            effort=ReasoningEffort.XHIGH,
        ),
    )

    body = _wire_body(client)
    assert body["max_tokens"] == 2048
    assert body["thinking"] == {"type": "enabled"}
    assert body["reasoning_effort"] == "high"
    assert "max_completion_tokens" not in body


def test_nonstream_native_openai_tools_are_sent_and_typed(configured, monkeypatch):
    client = _answers(monkeypatch, _Response({
        "choices": [{
            "message": {
                "content": "先列目录。",
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "file_list", "arguments": '{"path":"."}'},
                }],
            },
            "finish_reason": "tool_calls",
        }],
        "usage": {},
    }))

    result = models.openai_compatible_chat(
        [{"role": "user", "content": "list"}],
        tools=[{"id": "file_list", "input_schema": {"type": "object"}}],
        control_transport="native",
        tool_choice={"type": "tool", "name": "file_list"},
    )

    assert _wire_body(client)["tools"][0]["function"]["name"] == "file_list"
    assert _wire_body(client)["tool_choice"] == {
        "type": "function",
        "function": {"name": "file_list"},
    }
    assert isinstance(result.output, ToolCallBatch)
    assert result.output.calls[0].arguments == {"path": "."}


def test_generic_chat_dispatches_to_openai_without_unbound_options(
    configured, monkeypatch
):
    captured = {}

    def completion(messages, **kwargs):
        captured.update(messages=messages, **kwargs)
        return models.ModelResult(
            reply="ok",
            provider="openai-compatible",
            model="gpt-test",
            latency_ms=1,
        )

    monkeypatch.setattr(models, "openai_compatible_chat", completion)

    result = models.chat(
        "hi",
        [{"role": "user", "content": "hi"}],
        model_call_id="chat-openai",
        purpose="ordinary_chat",
    )

    assert result.reply == "ok"
    assert captured["model_call_id"] == "chat-openai"
    assert captured["purpose"] == "ordinary_chat"


# --- 响应形状 --------------------------------------------------------------------


def test_the_result_looks_the_same_as_the_other_family(configured, monkeypatch):
    _answers(monkeypatch, _Response(COMPLETION))
    result = models.openai_compatible_chat([{"role": "user", "content": "几只猫？"}])

    assert result.reply == "3只。"
    assert result.provider == "openai-compatible"
    assert result.input_tokens == 11 and result.output_tokens == 4
    assert result.finish_reason == "stop"
    assert result.model_call_id


def test_a_response_without_choices_is_a_typed_bad_response(configured, monkeypatch):
    _answers(monkeypatch, _Response({"error": "nope"}))
    with pytest.raises(ModelGatewayError) as raised:
        models.openai_compatible_chat([{"role": "user", "content": "hi"}])
    assert raised.value.code == "MODEL_BAD_RESPONSE"


@pytest.mark.parametrize(
    "exc, code, retryable",
    [
        (httpx.TimeoutException("slow"), "MODEL_CALL_TIMEOUT", True),
        (httpx.ConnectError("down"), "MODEL_CALL_FAILED", True),
    ],
)
def test_transport_failures_use_the_same_vocabulary(configured, monkeypatch, exc, code, retryable):
    def _raise(*a, **k):
        raise exc

    monkeypatch.setattr(models.httpx, "Client", _raise)
    with pytest.raises(ModelGatewayError) as raised:
        models.openai_compatible_chat([{"role": "user", "content": "hi"}])
    assert raised.value.code == code
    assert raised.value.retryable is retryable


def test_a_server_error_is_retryable_and_a_client_error_is_not(configured, monkeypatch):
    for status, retryable in ((503, True), (400, False)):
        request = httpx.Request("POST", "https://example.test/v1/chat/completions")
        response = httpx.Response(status, request=request)
        _answers(monkeypatch, _Response(
            status_exc=httpx.HTTPStatusError("boom", request=request, response=response)
        ))
        with pytest.raises(ModelGatewayError) as raised:
            models.openai_compatible_chat([{"role": "user", "content": "hi"}])
        assert raised.value.retryable is retryable


@pytest.mark.parametrize(
    "header,expected",
    [
        ("12", 12.0),
        ("Thu, 01 Jan 1970 00:18:20 GMT", 100.0),
    ],
)
def test_rate_limit_retains_only_a_parsed_retry_after_delay(
    configured, monkeypatch, header, expected
):
    monkeypatch.setattr(models.time, "time", lambda: 1_000.0)
    request = httpx.Request("POST", "https://example.test/v1/chat/completions")
    response = httpx.Response(
        429,
        request=request,
        headers={"Retry-After": header},
    )
    _answers(
        monkeypatch,
        _Response(
            status_exc=httpx.HTTPStatusError(
                "rate limited", request=request, response=response
            )
        ),
    )

    with pytest.raises(ModelGatewayError) as raised:
        models.openai_compatible_chat([{"role": "user", "content": "hi"}])

    assert raised.value.retryable is True
    assert raised.value.details["status_code"] == 429
    assert raised.value.details["retry_after_seconds"] == expected
    assert "Retry-After" not in raised.value.details


def test_provider_parameter_error_is_actionable_and_key_redacted(configured, monkeypatch):
    request = httpx.Request("POST", "https://example.test/v1/chat/completions")
    response = httpx.Response(
        400,
        request=request,
        headers={"x-request-id": "req_dialect"},
        json={
            "error": {
                "message": "max_tokens is unsupported; credential sk-test",
                "type": "invalid_request_error",
                "param": "max_tokens",
                "code": "unsupported_parameter",
            }
        },
    )
    _answers(
        monkeypatch,
        _Response(
            status_exc=httpx.HTTPStatusError(
                "boom", request=request, response=response
            )
        ),
    )

    with pytest.raises(ModelGatewayError) as raised:
        models.openai_compatible_chat([{"role": "user", "content": "hi"}])

    details = raised.value.details
    assert details["provider_error_param"] == "max_tokens"
    assert details["provider_error_code"] == "unsupported_parameter"
    assert details["provider_request_id"] == "req_dialect"
    assert "sk-test" not in json.dumps(details)


def test_a_missing_key_is_refused_without_a_request(configured, monkeypatch):
    monkeypatch.delenv("PERSONAGRAPH_API_KEY", raising=False)
    monkeypatch.setattr(models, "get_setting", lambda key, default=None: None if key == "api_key" else default)
    with pytest.raises(ModelGatewayError) as raised:
        models.openai_compatible_chat([{"role": "user", "content": "hi"}])
    assert raised.value.details["reason"] == "missing_api_key"


# --- 推理内容 --------------------------------------------------------------------


def test_reasoning_is_kept_out_of_the_reply(configured, monkeypatch):
    """与另一协议家族规则相同：记录该内容，但绝不返回。"""

    _answers(monkeypatch, _Response({
        "choices": [{"message": {"content": "3只。", "reasoning_content": "先想一下"}}],
        "usage": {},
    }))
    result = models.openai_compatible_chat([{"role": "user", "content": "hi"}])
    assert result.reply == "3只。"


def test_an_unfamiliar_reasoning_field_is_ignored_rather_than_guessed(configured, monkeypatch):
    _answers(monkeypatch, _Response({
        "choices": [{"message": {"content": "答案", "thoughts": "别处的字段名"}}], "usage": {},
    }))
    assert models.openai_compatible_chat([{"role": "user", "content": "hi"}]).reply == "答案"


# --- 流式 -----------------------------------------------------------------------


def _sse(*events):
    return [f"data: {json.dumps(event)}" for event in events] + ["data: [DONE]"]


def test_streaming_reads_deltas_from_where_this_family_puts_them(configured, monkeypatch):
    _answers(monkeypatch, _Response(lines=_sse(
        {"choices": [{"delta": {"reasoning_content": "先想，"}}]},
        {"choices": [{"delta": {"content": "3"}}]},
        {"choices": [{"delta": {"content": "只。"}, "finish_reason": "stop"}]},
        {"choices": [{"delta": {}}], "usage": {"prompt_tokens": 5, "completion_tokens": 2}},
    )))
    result = models.openai_compatible_chat(
        [{"role": "user", "content": "hi"}],
        stream=True,
        timeout_s=5,
        model_call_id="mc-openai",
        purpose="probe",
    )

    assert result.reply == "3只。"      # 推理绝不混进回复
    assert result.finish_reason == "stop"
    assert result.output_tokens == 2


def test_native_stream_reassembles_openai_function_calls(configured, monkeypatch):
    _answers(monkeypatch, _Response(lines=_sse(
        {"choices": [{"delta": {"content": "先检查。"}}]},
        {"choices": [{"delta": {"tool_calls": [{
            "index": 0,
            "id": "call_1",
            "type": "function",
            "function": {"name": "file_list", "arguments": '{"path"'},
        }]}}]},
        {"choices": [{"delta": {"tool_calls": [{
            "index": 0,
            "function": {"arguments": ':"."}'},
        }]}, "finish_reason": "tool_calls"}]},
    )))

    result = models.openai_compatible_chat(
        [{"role": "user", "content": "list"}],
        stream=True,
        timeout_s=5,
        model_call_id="mc-native-openai",
        purpose="probe",
        control_transport="native",
        tools=[{"id": "file_list", "input_schema": {"type": "object"}}],
    )

    assert isinstance(result.output, ToolCallBatch)
    assert result.output.assistant_text == "先检查。"
    assert result.output.calls[0].arguments == {"path": "."}
    assert result.control_transport == "native"


def test_a_stream_that_said_nothing_is_a_typed_failure(configured, monkeypatch):
    _answers(monkeypatch, _Response(lines=["data: [DONE]"]))
    with pytest.raises(ModelGatewayError) as raised:
        models.openai_compatible_chat(
            [],
            stream=True,
            timeout_s=5,
            model_call_id=None,
            purpose="probe",
        )
    assert raised.value.code == "MODEL_BAD_RESPONSE"
