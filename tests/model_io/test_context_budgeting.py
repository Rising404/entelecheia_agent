from __future__ import annotations

import base64
from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
import json
import math
import random

from PIL import Image
import pytest

from personagraph.context_budget import (
    ContextBudgetControlMismatch,
    ContextBudgetEncodingError,
    ContextBudgetExceeded,
    ContextBudgetInputError,
    ContextBudgetRouteMismatch,
    TokenEstimateKind,
    canonical_json_bytes,
)
from personagraph.model_io import context_budgeting
from personagraph.model_io.context_budgeting import (
    DEFAULT_CONTEXT_SAFETY_MARGIN_TOKENS,
    DEFAULT_CONTEXT_SOFT_PRESSURE_RATIO,
    DEFAULT_MAX_SERIALIZED_REQUEST_UTF8_BYTES,
    DEEPSEEK_QUOTA_HEURISTIC_HEADROOM_RATIO,
    RouteBoundJsonEnvelopeMeter,
    build_context_window_profile,
    prepare_and_admit_provider_request,
)


@dataclass
class _Counter:
    kind: str = "heuristic"
    requested: str = "heuristic"
    name: str = "test-counter"
    model: str | None = None
    fallback_reason: str | None = None
    calls: int = 0
    last_count: int = 0

    def count_text(self, text: str) -> int:
        self.calls += 1
        self.last_count = max(1, len(text) // 4)
        return self.last_count


def _png_bytes(
    *,
    width: int = 512,
    height: int = 512,
    noisy: bool = True,
) -> bytes:
    if noisy:
        pixels = random.Random(0).randbytes(width * height * 3)
        image = Image.frombytes("RGB", (width, height), pixels)
    else:
        image = Image.new("RGB", (width, height), color=(31, 63, 127))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _openai_payload(**updates: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "model": "gpt-5-test",
        "messages": [{"role": "user", "content": "private prompt"}],
        "max_completion_tokens": 256,
        "reasoning_effort": "high",
        "stream": False,
    }
    payload.update(updates)
    return payload


def _deepseek_payload(**updates: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "model": "deepseek-v4-pro",
        "messages": [{"role": "user", "content": "private prompt"}],
        "max_tokens": 512,
        "thinking": {"type": "enabled"},
    }
    payload.update(updates)
    return payload


def test_route_bound_meter_parses_exact_openai_wire_controls() -> None:
    counter = _Counter()
    payload = _openai_payload()
    encoded = canonical_json_bytes(payload)
    meter = RouteBoundJsonEnvelopeMeter(
        provider="openai-compatible",
        model="gpt-5-test",
        dialect="openai-native",
        counter=counter,
    )

    result = meter.measure(encoded)

    assert result.output_token_limit == 256
    assert result.shared_reasoning_reserve_tokens == 0
    assert result.input_tokens == len(encoded)
    assert (
        result.components[0].component_id
        == "wire_json_without_inline_image_data"
    )
    assert result.components[0].estimate_kind is TokenEstimateKind.HEURISTIC
    controls = {
        "model": "gpt-5-test",
        "max_completion_tokens": 256,
        "reasoning_effort": "high",
        "stream": False,
    }
    assert result.request_controls_sha256 == sha256(
        canonical_json_bytes(controls)
    ).hexdigest()
    assert counter.calls == 1


def test_heuristic_final_meter_uses_utf8_byte_upper_bound() -> None:
    payload = _openai_payload(
        messages=[{"role": "user", "content": "🙂かな한글" * 100}]
    )
    meter = RouteBoundJsonEnvelopeMeter(
        provider="openai-compatible",
        model="gpt-5-test",
        dialect="openai-native",
        counter=_Counter(),
    )

    encoded = canonical_json_bytes(payload)
    result = meter.measure(encoded)

    assert result.input_tokens == len(encoded)
    assert result.components[0].estimate_kind is TokenEstimateKind.HEURISTIC


def test_meter_reports_tokenizer_provenance_without_claiming_provider_exactness() -> None:
    counter = _Counter(kind="tokenizer", requested="tiktoken", name="tiktoken")
    meter = RouteBoundJsonEnvelopeMeter(
        provider="anthropic-compatible",
        model="deepseek-v4-pro",
        dialect="deepseek-anthropic",
        counter=counter,
    )
    payload = {
        **_deepseek_payload(),
        "stream": True,
        "system": "private system",
    }

    result = meter.measure(canonical_json_bytes(payload))

    assert meter.kind == "tokenizer"
    assert result.components[0].estimate_kind is TokenEstimateKind.TOKENIZER
    assert all(
        item.estimate_kind is not TokenEstimateKind.PROVIDER_EXACT
        for item in result.components
    )


@pytest.mark.parametrize("counter_kind", ["heuristic", "tokenizer"])
def test_inline_image_base64_is_not_counted_as_prompt_text(
    counter_kind: str,
) -> None:
    raw_image = _png_bytes()
    encoded_image = base64.b64encode(raw_image).decode("ascii")
    payload = {
        "model": "claude-sonnet-4",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": encoded_image,
                        },
                    },
                    {"type": "text", "text": "inspect this image"},
                ],
            }
        ],
        "max_tokens": 64,
    }

    prepared = prepare_and_admit_provider_request(
        payload,
        provider="anthropic-compatible",
        model="claude-sonnet-4",
        dialect="anthropic-native",
        stream=False,
        purpose="vision_test",
        projection_epoch="epoch-vision",
        projection_generation=1,
        configured_input_limit_tokens=10_000,
        counter=_Counter(
            kind=counter_kind,
            requested=counter_kind,
            name=f"test-{counter_kind}",
        ),
    )

    receipt = prepared.admitted_request.receipt
    assert receipt.input_tokens < 10_000
    assert receipt.input_tokens < len(encoded_image) // 4
    assert [item.component_id for item in receipt.components] == [
        "vision_image_0001",
        "wire_json_without_inline_image_data",
    ]
    assert json.loads(prepared.admitted_request.body_for_dispatch())["messages"][0][
        "content"
    ][0]["source"]["data"] == encoded_image


def test_compressed_large_image_is_billed_by_dimensions_not_only_bytes() -> None:
    width = height = 2_048
    raw_image = _png_bytes(width=width, height=height, noisy=False)
    encoded_image = base64.b64encode(raw_image).decode("ascii")
    payload = {
        "model": "claude-sonnet-4",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": encoded_image,
                        },
                    }
                ],
            }
        ],
        "max_tokens": 64,
    }

    prepared = prepare_and_admit_provider_request(
        payload,
        provider="anthropic-compatible",
        model="claude-sonnet-4",
        dialect="anthropic-native",
        stream=False,
        purpose="vision_dimension_test",
        projection_epoch="epoch-vision",
        projection_generation=1,
        configured_input_limit_tokens=10_000,
        counter=_Counter(),
    )

    vision = next(
        component
        for component in prepared.admitted_request.receipt.components
        if component.component_id == "vision_image_0001"
    )
    assert vision.tokens >= math.ceil(width / 28) * math.ceil(height / 28)
    assert vision.tokens > len(raw_image) // 1_024 * 6


def test_inline_image_shape_and_nonvision_route_fail_closed() -> None:
    invalid_payload = {
        "model": "claude-sonnet-4",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": "not-base64!",
                        },
                    }
                ],
            }
        ],
        "max_tokens": 64,
    }
    with pytest.raises(ContextBudgetEncodingError, match="invalid base64"):
        prepare_and_admit_provider_request(
            invalid_payload,
            provider="anthropic-compatible",
            model="claude-sonnet-4",
            dialect="anthropic-native",
            stream=False,
            purpose="vision_test",
            projection_epoch="epoch-vision",
            projection_generation=1,
            counter=_Counter(),
        )

    invalid_bytes_payload = {
        **invalid_payload,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": base64.b64encode(b"not-a-png").decode("ascii"),
                        },
                    }
                ],
            }
        ],
    }
    with pytest.raises(ContextBudgetEncodingError, match="safely inspected"):
        prepare_and_admit_provider_request(
            invalid_bytes_payload,
            provider="anthropic-compatible",
            model="claude-sonnet-4",
            dialect="anthropic-native",
            stream=False,
            purpose="vision_test",
            projection_epoch="epoch-vision",
            projection_generation=1,
            counter=_Counter(),
        )

    nonvision_payload = dict(invalid_payload)
    nonvision_payload["model"] = "deepseek-chat"
    nonvision_payload["messages"] = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": base64.b64encode(_png_bytes()).decode("ascii"),
                    },
                }
            ],
        }
    ]
    with pytest.raises(ContextBudgetInputError, match="does not admit"):
        prepare_and_admit_provider_request(
            nonvision_payload,
            provider="anthropic-compatible",
            model="deepseek-chat",
            dialect="deepseek-anthropic",
            stream=False,
            purpose="vision_test",
            projection_epoch="epoch-vision",
            projection_generation=1,
            counter=_Counter(),
        )


@pytest.mark.parametrize(
    "updates",
    [
        {"max_completion_tokens": None},
        {"max_completion_tokens": True},
        {"max_completion_tokens": 0},
        {"max_tokens": 256},
    ],
)
def test_missing_invalid_or_conflicting_output_control_fails_closed(
    updates: dict[str, object],
) -> None:
    payload = _openai_payload(**updates)
    if (
        "max_completion_tokens" in updates
        and updates["max_completion_tokens"] is None
    ):
        payload.pop("max_completion_tokens")

    with pytest.raises(ContextBudgetControlMismatch):
        prepare_and_admit_provider_request(
            payload,
            provider="openai-compatible",
            model="gpt-5-test",
            dialect="openai-native",
            stream=False,
            purpose="test",
            projection_epoch="epoch-1",
            projection_generation=1,
            counter=_Counter(),
        )


def test_wrong_token_field_for_dialect_fails_closed() -> None:
    payload = _openai_payload()
    payload["max_tokens"] = payload.pop("max_completion_tokens")

    with pytest.raises(ContextBudgetControlMismatch, match="request dialect"):
        prepare_and_admit_provider_request(
            payload,
            provider="openai-compatible",
            model="gpt-5-test",
            dialect="openai-native",
            stream=False,
            purpose="test",
            projection_epoch="epoch-1",
            projection_generation=1,
            counter=_Counter(),
        )


def test_prepare_inserts_stream_before_the_only_payload_serialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _deepseek_payload()
    private_text = str(payload["messages"])
    serialized_payload_calls = 0
    real_serializer = context_budgeting.canonical_json_bytes

    def spy(value: object) -> bytes:
        nonlocal serialized_payload_calls
        if isinstance(value, dict) and "messages" in value:
            serialized_payload_calls += 1
        return real_serializer(value)

    monkeypatch.setattr(context_budgeting, "canonical_json_bytes", spy)
    counter = _Counter()

    prepared = prepare_and_admit_provider_request(
        payload,
        provider="anthropic-compatible",
        model="deepseek-v4-pro",
        dialect="deepseek-anthropic",
        stream=True,
        purpose="runtime_l1_step",
        projection_epoch="turn-1:projection",
        projection_generation=4,
        counter=counter,
    )

    body = prepared.admitted_request.body_for_dispatch()
    decoded = json.loads(body)
    assert decoded["stream"] is True
    assert "stream" not in payload
    assert serialized_payload_calls == 1
    assert prepared.metadata.stream is True
    assert prepared.metadata.output_token_field == "max_tokens"
    assert prepared.metadata.output_token_limit == 512
    assert prepared.metadata.measured_input_tokens > 0
    assert prepared.metadata.quota_input_token_estimate > 0
    assert (
        prepared.metadata.quota_input_token_estimate
        < prepared.metadata.measured_input_tokens
    )
    assert prepared.metadata.quota_input_token_estimate == min(
        prepared.metadata.measured_input_tokens,
        math.ceil(
            counter.last_count * DEEPSEEK_QUOTA_HEURISTIC_HEADROOM_RATIO
        ),
    )
    assert counter.calls == 1
    assert (
        prepared.metadata.measured_input_tokens
        < prepared.metadata.input_budget_tokens
    )
    assert prepared.metadata.soft_input_limit_tokens <= prepared.metadata.input_budget_tokens
    assert prepared.metadata.pressure == "normal"
    assert prepared.metadata.request_sha256 == sha256(body).hexdigest()
    assert prepared.metadata.serialized_request_utf8_bytes == len(body)
    assert prepared.admitted_request.receipt.request_sha256 == (
        prepared.metadata.request_sha256
    )
    assert private_text not in repr(prepared)
    assert "private prompt" not in prepared.metadata.model_dump_json()


def test_prepare_rejects_a_conflicting_existing_stream_control() -> None:
    with pytest.raises(ContextBudgetControlMismatch, match="stream control conflicts"):
        prepare_and_admit_provider_request(
            _openai_payload(stream=True),
            provider="openai-compatible",
            model="gpt-5-test",
            dialect="openai-native",
            stream=False,
            purpose="test",
            projection_epoch="epoch-1",
            projection_generation=1,
            counter=_Counter(),
        )


def test_profile_uses_model_family_window_and_conservative_fixed_reserves() -> None:
    profile = build_context_window_profile(
        provider="anthropic-compatible",
        model="deepseek-v4-pro",
        dialect="deepseek-anthropic",
        output_token_limit=8_192,
    )

    assert profile.context_window_tokens == 1_000_000
    assert profile.reserved_output_tokens == 8_192
    assert profile.shared_reasoning_reserve_tokens == 0
    assert profile.safety_margin_tokens == DEFAULT_CONTEXT_SAFETY_MARGIN_TOKENS
    assert profile.soft_pressure_ratio == DEFAULT_CONTEXT_SOFT_PRESSURE_RATIO
    assert (
        profile.max_serialized_request_utf8_bytes
        == DEFAULT_MAX_SERIALIZED_REQUEST_UTF8_BYTES
    )
    assert profile.input_budget_tokens == 1_000_000 - 8_192 - 1_024


def test_deepseek_chat_uses_the_current_one_million_context_window() -> None:
    profile = build_context_window_profile(
        provider="openai-compatible",
        model="deepseek-chat",
        dialect="deepseek-openai",
        output_token_limit=65_536,
    )

    assert profile.context_window_tokens == 1_000_000
    assert profile.input_budget_tokens == 1_000_000 - 65_536 - 1_024


def test_unknown_model_with_no_remaining_input_window_fails_closed() -> None:
    with pytest.raises(ContextBudgetInputError, match="no safe input"):
        build_context_window_profile(
            provider="openai-compatible",
            model="custom-model-with-unknown-window",
            dialect="generic-openai",
            output_token_limit=32_000,
        )


def test_payload_route_model_and_protocol_dialect_are_bound() -> None:
    with pytest.raises(ContextBudgetRouteMismatch, match="payload model"):
        prepare_and_admit_provider_request(
            _openai_payload(),
            provider="openai-compatible",
            model="another-model",
            dialect="openai-native",
            stream=False,
            purpose="test",
            projection_epoch="epoch-1",
            projection_generation=1,
            counter=_Counter(),
        )
    with pytest.raises(ContextBudgetRouteMismatch, match="same route"):
        RouteBoundJsonEnvelopeMeter(
            provider="anthropic-compatible",
            model="gpt-5-test",
            dialect="openai-native",
            counter=_Counter(),
        )


def test_meter_rejects_non_object_or_non_finite_exact_bytes() -> None:
    meter = RouteBoundJsonEnvelopeMeter(
        provider="openai-compatible",
        model="gpt-5-test",
        dialect="openai-native",
        counter=_Counter(),
    )
    for body in (b"[]", b'{"x":NaN}', b"not-json"):
        with pytest.raises(ContextBudgetEncodingError):
            meter.measure(body)


def test_helper_uses_legacy_counter_factory_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    counter = _Counter()
    calls = 0

    def factory() -> _Counter:
        nonlocal calls
        calls += 1
        return counter

    monkeypatch.setattr(context_budgeting, "get_token_counter", factory)

    prepared = prepare_and_admit_provider_request(
        _deepseek_payload(),
        provider="anthropic-compatible",
        model="deepseek-v4-pro",
        dialect="deepseek-anthropic",
        stream=False,
        purpose="test",
        projection_epoch="epoch-1",
        projection_generation=1,
    )

    assert calls == 1
    assert counter.calls == 1
    assert prepared.metadata.meter_kind == "heuristic"


def test_tokenizer_meter_uses_the_same_context_and_quota_token_count() -> None:
    prepared = prepare_and_admit_provider_request(
        _deepseek_payload(),
        provider="anthropic-compatible",
        model="deepseek-v4-pro",
        dialect="deepseek-anthropic",
        stream=False,
        purpose="test",
        projection_epoch="epoch-1",
        projection_generation=1,
        counter=_Counter(kind="tokenizer", name="test-tokenizer"),
    )

    assert prepared.metadata.meter_kind == "tokenizer"
    assert (
        prepared.metadata.quota_input_token_estimate
        == prepared.metadata.measured_input_tokens
    )


def test_uncalibrated_heuristic_route_keeps_the_context_hard_bound() -> None:
    prepared = prepare_and_admit_provider_request(
        _openai_payload(),
        provider="openai-compatible",
        model="gpt-5-test",
        dialect="openai-native",
        stream=False,
        purpose="test",
        projection_epoch="epoch-1",
        projection_generation=1,
        counter=_Counter(),
    )

    assert prepared.metadata.meter_kind == "heuristic"
    assert (
        prepared.metadata.quota_input_token_estimate
        == prepared.metadata.measured_input_tokens
    )


def test_configured_input_cap_is_enforced_by_final_admission() -> None:
    with pytest.raises(ContextBudgetExceeded) as caught:
        prepare_and_admit_provider_request(
            _deepseek_payload(),
            provider="anthropic-compatible",
            model="deepseek-v4-pro",
            dialect="deepseek-anthropic",
            stream=False,
            purpose="test",
            projection_epoch="epoch-1",
            projection_generation=1,
            configured_input_limit_tokens=10,
            counter=_Counter(),
        )

    assert caught.value.stage == "provider_envelope"
    assert caught.value.receipt is not None
    assert caught.value.receipt.admitted is False
