from __future__ import annotations

import base64
from dataclasses import replace
from hashlib import sha256
from io import BytesIO
import json
import random

from PIL import Image
import pytest

from personagraph.model_io import gateway as models
from personagraph.context_budget import ContextBudgetExceeded
from personagraph.model_io.tier_bindings import (
    EndpointOrigin,
    ModelTierBinding,
    ModelTier,
)
from personagraph.model_io.endpoint_profiles import ModelProfileQuota
from personagraph.model_io.gateway_core import _prepare_model_api_quota
from personagraph.runtime.l1.model_output_budgets import (
    l1_attempt_max_output_tokens,
)


def _png_bytes() -> bytes:
    pixels = random.Random(0).randbytes(512 * 512 * 3)
    image = Image.frombytes("RGB", (512, 512), pixels)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


class _Response:
    def __init__(self, provider: str, *, stream: bool) -> None:
        self.provider = provider
        self.stream = stream

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def raise_for_status(self):
        return None

    def json(self):
        if self.provider == "openai-compatible":
            return {
                "choices": [
                    {"message": {"content": "ok"}, "finish_reason": "stop"}
                ],
                "usage": {},
            }
        return {
            "content": [{"type": "text", "text": "ok"}],
            "stop_reason": "end_turn",
            "usage": {},
        }

    def iter_lines(self):
        if self.provider == "openai-compatible":
            yield 'data: {"choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}]}'
            yield "data: [DONE]"
            return
        yield 'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"ok"}}'
        yield 'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"}}'


class _Client:
    def __init__(self, response: _Response, calls: list[dict]) -> None:
        self.response = response
        self.calls = calls

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def post(self, endpoint: str, **kwargs):
        self.calls.append({"endpoint": endpoint, **kwargs})
        return self.response

    def stream(self, method: str, endpoint: str, **kwargs):
        self.calls.append({"method": method, "endpoint": endpoint, **kwargs})
        return self.response


def _binding(
    *,
    provider: str,
    dialect: str,
    base_url: str,
    model: str,
) -> ModelTierBinding:
    return ModelTierBinding(
        tier=ModelTier.ATTEMPT,
        provider=provider,
        base_url=base_url,
        model=model,
        api_key="tier-secret",
        thinking_enabled=True,
        request_dialect=dialect,
        origin=EndpointOrigin.PROFILE,
    )


@pytest.mark.parametrize("stream", [False, True], ids=["nonstream", "stream"])
@pytest.mark.parametrize(
    "provider,dialect,base_url,model,output_field",
    [
        (
            "anthropic-compatible",
            "anthropic-native",
            "https://api.anthropic.com",
            "claude-sonnet-4",
            "max_tokens",
        ),
        (
            "openai-compatible",
            "openai-native",
            "https://api.openai.com/v1",
            "gpt-4o",
            "max_completion_tokens",
        ),
        (
            "openai-compatible",
            "deepseek-openai",
            "https://api.deepseek.com/v1",
            "deepseek-chat",
            "max_tokens",
        ),
        (
            "anthropic-compatible",
            "deepseek-anthropic",
            "https://api.deepseek.com/anthropic",
            "deepseek-v4-pro",
            "max_tokens",
        ),
    ],
)
def test_provider_matrix_dispatches_the_identical_admitted_bytes(
    monkeypatch,
    *,
    stream: bool,
    provider: str,
    dialect: str,
    base_url: str,
    model: str,
    output_field: str,
) -> None:
    calls: list[dict] = []
    response = _Response(provider, stream=stream)
    monkeypatch.setattr(
        models.httpx,
        "Client",
        lambda **_kwargs: _Client(response, calls),
    )
    binding = _binding(
        provider=provider,
        dialect=dialect,
        base_url=base_url,
        model=model,
    )
    messages = [{"role": "user", "content": "before preparation"}]

    if provider == "openai-compatible":
        prepared = models.prepare_openai_compatible_chat(
            messages,
            binding=binding,
            max_tokens=64,
            stream=stream,
            purpose="wire_test",
        )
    else:
        prepared = models.prepare_anthropic_compatible_chat(
            messages,
            binding=binding,
            max_tokens=64,
            stream=stream,
            purpose="wire_test",
        )

    # 准备阶段只是纯预检：即使端点解析和最终计量，也会在打开 HTTP 客户端前完成。
    assert calls == []
    messages[0]["content"] = "mutated after admission"
    assert prepared.context_budget is not None
    exact_body = prepared.context_budget.admitted_request.body_for_dispatch()
    metadata = prepared.context_budget.metadata
    assert sha256(exact_body).hexdigest() == metadata.request_sha256

    result = prepared.dispatch(model_call_id="wire-call")

    assert result.reply == "ok"
    assert len(calls) == 1
    assert "json" not in calls[0]
    assert calls[0]["content"] is exact_body
    body = json.loads(exact_body)
    assert body["messages"][-1]["content"] == "before preparation"
    assert body["stream"] is stream
    assert body[output_field] == 64
    other_output_field = (
        "max_tokens"
        if output_field == "max_completion_tokens"
        else "max_completion_tokens"
    )
    assert other_output_field not in body
    assert (metadata.provider, metadata.model, metadata.dialect) == (
        provider,
        model,
        dialect,
    )
    assert metadata.stream is stream
    assert metadata.output_token_field == output_field


def test_hard_context_limit_fails_before_http_client_creation(monkeypatch) -> None:
    def should_not_open(**_kwargs):
        raise AssertionError("HTTP client must not be opened before admission")

    monkeypatch.setattr(models.httpx, "Client", should_not_open)
    binding = _binding(
        provider="openai-compatible",
        dialect="openai-native",
        base_url="https://api.openai.com/v1",
        model="gpt-4o",
    )

    with pytest.raises(ContextBudgetExceeded) as raised:
        models.prepare_openai_compatible_chat(
            [{"role": "user", "content": "x" * 4_000}],
            binding=binding,
            max_tokens=64,
            configured_input_limit_tokens=1,
            purpose="wire_test",
        )

    assert raised.value.receipt.admitted is False
    assert raised.value.receipt.provider == "openai-compatible"
    assert raised.value.receipt.dialect == "openai-native"


@pytest.mark.parametrize("stream", [False, True], ids=["nonstream", "stream"])
@pytest.mark.parametrize(
    "provider,dialect,base_url,model",
    [
        (
            "anthropic-compatible",
            "anthropic-native",
            "https://api.anthropic.com",
            "claude-sonnet-4",
        ),
        (
            "openai-compatible",
            "openai-native",
            "https://api.openai.com/v1",
            "gpt-4o",
        ),
    ],
)
def test_multimodal_wire_uses_vision_tokens_without_counting_base64_as_text(
    monkeypatch,
    *,
    stream: bool,
    provider: str,
    dialect: str,
    base_url: str,
    model: str,
) -> None:
    calls: list[dict] = []
    monkeypatch.setattr(
        models.httpx,
        "Client",
        lambda **_kwargs: _Client(_Response(provider, stream=stream), calls),
    )
    binding = _binding(
        provider=provider,
        dialect=dialect,
        base_url=base_url,
        model=model,
    )
    encoded_image = base64.b64encode(_png_bytes()).decode("ascii")
    image_block = {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/png",
            "data": encoded_image,
        },
    }
    messages = [
        {
            "role": "user",
            "content": [image_block, {"type": "text", "text": "inspect"}],
        }
    ]
    prepare = (
        models.prepare_openai_compatible_chat
        if provider == "openai-compatible"
        else models.prepare_anthropic_compatible_chat
    )

    prepared = prepare(
        messages,
        binding=binding,
        max_tokens=64,
        stream=stream,
        purpose="vision_wire_test",
        configured_input_limit_tokens=10_000,
    )

    assert calls == []
    assert prepared.context_budget is not None
    admitted = prepared.context_budget.admitted_request
    metadata = prepared.context_budget.metadata
    exact_body = admitted.body_for_dispatch()
    wire = json.loads(exact_body)
    wire_block = wire["messages"][0]["content"][0]
    if provider == "openai-compatible":
        assert wire_block["type"] == "image_url"
        assert wire_block["image_url"]["url"].endswith(encoded_image)
    else:
        assert wire_block["type"] == "image"
        assert wire_block["source"]["data"] == encoded_image
    assert metadata.measured_input_tokens < 10_000
    assert metadata.measured_input_tokens < len(encoded_image) // 4
    assert any(
        item.component_id == "vision_image_0001"
        for item in admitted.receipt.components
    )

    image_block["source"]["data"] = "mutated-after-prepare"
    result = prepared.dispatch(model_call_id="vision-wire-call")

    assert result.reply == "ok"
    assert calls[0]["content"] is exact_body


def test_prepare_complete_structured_defers_mock_dispatch() -> None:
    assert models.complete_structured.prepare is models.prepare_complete_structured
    prepared = models.prepare_complete_structured(
        "system",
        "user",
        mock_payload={"ok": True},
        purpose="wire_test",
    )

    assert prepared.context_budget is None
    result = prepared.dispatch(model_call_id="mock-wire")
    assert result.model_call_id == "mock-wire"
    assert json.loads(result.reply) == {"ok": True}


def test_quota_reserves_nominal_tokens_without_weakening_context_admission() -> None:
    binding = ModelTierBinding(
        tier=ModelTier.L1,
        provider="openai-compatible",
        base_url="https://models.example.test/v1",
        model="deepseek-chat",
        api_key="tier-secret",
        thinking_enabled=False,
        request_dialect="deepseek-openai",
        origin=EndpointOrigin.PROFILE,
        quota=ModelProfileQuota(tokens_per_minute=100_000),
    )
    prepared = models.prepare_openai_compatible_chat(
        [{"role": "user", "content": "bounded document context " * 4_000}],
        binding=binding,
        max_tokens=8_192,
        purpose="quota_estimate_test",
    )

    assert prepared.context_budget is not None
    assert prepared.api_quota is not None
    assert (
        prepared.context_budget.metadata.quota_input_token_estimate
        < prepared.context_budget.metadata.measured_input_tokens
    )
    assert prepared.api_quota.token_reservation == (
        prepared.context_budget.admitted_request.receipt.quota_input_token_estimate
        + 8_192
    )
    assert prepared.api_quota.token_reservation < 100_000


def test_quota_reservation_uses_the_sealed_receipt_not_metadata_projection() -> None:
    binding = ModelTierBinding(
        tier=ModelTier.L1,
        provider="openai-compatible",
        base_url="https://models.example.test/v1",
        model="deepseek-chat",
        api_key="tier-secret",
        thinking_enabled=False,
        request_dialect="deepseek-openai",
        origin=EndpointOrigin.PROFILE,
        quota=ModelProfileQuota(tokens_per_minute=100_000),
    )
    prepared = models.prepare_openai_compatible_chat(
        [{"role": "user", "content": "bounded document context " * 1_000}],
        binding=binding,
        max_tokens=8_192,
        purpose="sealed_quota_authority_test",
    )

    assert prepared.context_budget is not None
    sealed_estimate = (
        prepared.context_budget.admitted_request.receipt.quota_input_token_estimate
    )
    projected = prepared.context_budget.metadata.model_copy(
        update={"quota_input_token_estimate": 1}
    )
    copied_context = replace(prepared.context_budget, metadata=projected)
    quota = _prepare_model_api_quota(
        context_budget=copied_context,
        provider=binding.provider,
        base_url=binding.base_url,
        api_key=binding.api_key,
        timeout_s=60.0,
        binding=binding,
    )

    assert quota is not None
    assert quota.token_reservation == sealed_estimate + 8_192


def test_l1_attempt_budget_fits_long_tool_observation_under_300k_tpm() -> None:
    binding = ModelTierBinding(
        tier=ModelTier.L1,
        provider="openai-compatible",
        base_url="https://models.example.test/v1",
        model="deepseek-chat",
        api_key="tier-secret",
        thinking_enabled=False,
        request_dialect="deepseek-openai",
        origin=EndpointOrigin.PROFILE,
        quota=ModelProfileQuota(tokens_per_minute=300_000),
    )
    prepared = models.prepare_openai_compatible_chat(
        [{"role": "user", "content": "x" * 140_000}],
        binding=binding,
        max_tokens=l1_attempt_max_output_tokens(
            thinking_enabled=binding.thinking_enabled,
        ),
        purpose="l1_long_tool_observation_budget_test",
    )

    assert prepared.context_budget is not None
    assert prepared.api_quota is not None
    assert prepared.context_budget.metadata.quota_input_token_estimate > 43_000
    assert prepared.api_quota.token_reservation < 300_000
