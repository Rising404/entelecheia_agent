from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
import json

import pytest
from pydantic import ValidationError

from personagraph.context_budget import (
    AdmittedContextRequest,
    ContextBudgetControlMismatch,
    ContextBudgetEncodingError,
    ContextBudgetExceeded,
    ContextBudgetEnvelopeMismatch,
    ContextBudgetMeasurementError,
    ContextBudgetPressure,
    ContextBudgetRouteMismatch,
    ContextBudgetViolation,
    ContextWindowProfile,
    EnvelopeTokenComponent,
    ProviderEnvelopeMeterResult,
    TokenEstimateKind,
    admit_context_request,
    canonical_json_bytes,
    check_context_budget,
    measure_canonical_json_envelope,
    measure_envelope_with_meter,
    measure_precounted_envelope,
    measure_precounted_envelope_bytes,
    require_context_budget,
)


@dataclass
class CharacterCounter:
    name: str = "character-counter"
    kind: str = "heuristic"
    requested: str = "test"
    model: str | None = "model-a"
    fallback_reason: str | None = None

    def count_text(self, text: str) -> int:
        return len(text)


@dataclass
class ExactProviderMeter:
    name: str = "provider-meter-v1"
    kind: str = "provider_exact"
    requested: str = "provider-wire"
    provider: str = "provider-a"
    model: str = "model-a"
    dialect: str = "dialect-a"
    fallback_reason: str | None = None
    observed_body: bytes | None = None

    def measure(self, serialized_envelope: bytes) -> ProviderEnvelopeMeterResult:
        self.observed_body = serialized_envelope
        controls = canonical_json_bytes(
            {"max_tokens": 20, "reasoning_reserve": 10}
        )
        components = (
            EnvelopeTokenComponent(
                component_id="provider_request",
                tokens=7,
                estimate_kind=TokenEstimateKind.PROVIDER_EXACT,
            ),
        )
        return ProviderEnvelopeMeterResult(
            output_token_limit=20,
            shared_reasoning_reserve_tokens=10,
            request_controls_sha256=sha256(controls).hexdigest(),
            input_tokens=7,
            quota_input_token_estimate=7,
            components=components,
        )


@dataclass
class WireControlProviderMeter(ExactProviderMeter):
    """测试用计量器：从精确的请求字节中推导容量控制参数。"""

    def measure(self, serialized_envelope: bytes) -> ProviderEnvelopeMeterResult:
        self.observed_body = serialized_envelope
        envelope = json.loads(serialized_envelope)
        output_limit = envelope["max_tokens"]
        reasoning_reserve = envelope["reasoning_reserve"]
        controls = canonical_json_bytes(
            {
                "max_tokens": output_limit,
                "reasoning_reserve": reasoning_reserve,
            }
        )
        components = (
            EnvelopeTokenComponent(
                component_id="provider_request",
                tokens=7,
                estimate_kind=TokenEstimateKind.PROVIDER_EXACT,
            ),
        )
        return ProviderEnvelopeMeterResult(
            output_token_limit=output_limit,
            shared_reasoning_reserve_tokens=reasoning_reserve,
            request_controls_sha256=sha256(controls).hexdigest(),
            input_tokens=7,
            quota_input_token_estimate=7,
            components=components,
        )


def _profile(
    *,
    context_window: int = 100,
    configured_input_limit: int | None = None,
    body_limit: int | None = 10_000,
) -> ContextWindowProfile:
    return ContextWindowProfile(
        profile_id="route-a-v1",
        provider="provider-a",
        model="model-a",
        dialect="dialect-a",
        context_window_tokens=context_window,
        configured_input_limit_tokens=configured_input_limit,
        reserved_output_tokens=20,
        shared_reasoning_reserve_tokens=10,
        safety_margin_tokens=5,
        soft_pressure_ratio=0.8,
        max_serialized_request_utf8_bytes=body_limit,
    )


def _measurement(*, tokens: int, envelope: object | None = None):
    return measure_precounted_envelope(
        envelope if envelope is not None else {"messages": []},
        provider="provider-a",
        model="model-a",
        dialect="dialect-a",
        counter=CharacterCounter(),
        components=(
            EnvelopeTokenComponent(
                component_id="full_request",
                tokens=tokens,
                estimate_kind=TokenEstimateKind.HEURISTIC,
            ),
        ),
        output_token_limit=20,
        shared_reasoning_reserve_tokens=10,
        request_controls={"max_tokens": 20, "reasoning_reserve": 10},
    )


def test_reserves_define_the_only_hard_input_budget() -> None:
    profile = _profile()
    assert profile.input_budget_tokens == 65

    at_boundary = check_context_budget(
        _measurement(tokens=65),
        profile=profile,
        purpose="runtime_l1_step",
        projection_epoch="epoch-a",
        projection_generation=0,
    )
    over_boundary = check_context_budget(
        _measurement(tokens=66),
        profile=profile,
        purpose="runtime_l1_step",
        projection_epoch="epoch-a",
        projection_generation=1,
    )

    assert at_boundary.admitted is True
    assert at_boundary.pressure is ContextBudgetPressure.SOFT_LIMIT_EXCEEDED
    assert over_boundary.admitted is False
    assert over_boundary.pressure is ContextBudgetPressure.HARD_LIMIT_EXCEEDED
    assert over_boundary.violations == (ContextBudgetViolation.INPUT_TOKENS,)


def test_configured_input_limit_can_only_tighten_provider_capacity() -> None:
    tighter = _profile(context_window=100, configured_input_limit=45)
    looser = _profile(context_window=100, configured_input_limit=200)

    assert tighter.provider_input_capacity_tokens == 65
    assert tighter.input_budget_tokens == 45
    assert looser.provider_input_capacity_tokens == 65
    assert looser.input_budget_tokens == 65


def test_soft_pressure_is_observable_but_does_not_claim_a_hard_failure() -> None:
    receipt = require_context_budget(
        _measurement(tokens=52),
        profile=_profile(),
        purpose="runtime_l1_step",
        projection_epoch="epoch-a",
        projection_generation=3,
    )

    assert receipt.soft_input_limit_tokens == 52
    assert receipt.pressure is ContextBudgetPressure.SOFT_LIMIT_EXCEEDED
    assert receipt.admitted is True
    assert receipt.violations == ()


def test_hard_failure_raises_with_a_content_free_receipt() -> None:
    secret = "never persist this prompt"
    measurement = _measurement(tokens=99, envelope={"messages": [secret]})

    with pytest.raises(ContextBudgetExceeded) as caught:
        require_context_budget(
            measurement,
            profile=_profile(),
            purpose="runtime_l1_step",
            projection_epoch="epoch-a",
            projection_generation=4,
        )

    receipt = caught.value.receipt
    assert receipt.request_sha256 == measurement.request_sha256
    assert caught.value.limit == 65
    assert caught.value.estimated_tokens == 99
    assert caught.value.degraded == ()
    assert secret not in receipt.model_dump_json()
    assert secret not in str(caught.value)


def test_request_body_limit_is_independent_from_token_window() -> None:
    measurement = _measurement(tokens=1, envelope={"payload": "x" * 200})
    receipt = check_context_budget(
        measurement,
        profile=_profile(body_limit=20),
        purpose="runtime_l1_step",
        projection_epoch="epoch-a",
        projection_generation=0,
    )

    assert receipt.admitted is False
    assert receipt.violations == (ContextBudgetViolation.REQUEST_BODY_BYTES,)


def test_measurement_must_match_the_frozen_physical_route() -> None:
    measurement = measure_precounted_envelope(
        {"messages": []},
        provider="another-provider",
        model="model-a",
        dialect="dialect-a",
        counter=CharacterCounter(),
        components=(
            EnvelopeTokenComponent(
                component_id="full_request",
                tokens=1,
                estimate_kind=TokenEstimateKind.HEURISTIC,
            ),
        ),
        output_token_limit=20,
        shared_reasoning_reserve_tokens=10,
        request_controls={"max_tokens": 20, "reasoning_reserve": 10},
    )

    with pytest.raises(ContextBudgetRouteMismatch):
        check_context_budget(
            measurement,
            profile=_profile(),
            purpose="runtime_l1_step",
            projection_epoch="epoch-a",
            projection_generation=0,
        )


def test_canonical_json_meter_is_order_independent_and_counts_extras() -> None:
    first = measure_canonical_json_envelope(
        {"tools": [{"name": "read"}], "messages": [{"content": "hello"}]},
        provider="provider-a",
        model="model-a",
        dialect="dialect-a",
        counter=CharacterCounter(),
        estimate_kind=TokenEstimateKind.HEURISTIC,
        output_token_limit=20,
        shared_reasoning_reserve_tokens=10,
        request_controls={"max_tokens": 20, "reasoning_reserve": 10},
        extra_token_components={"chat_framing": 7, "images": 11},
    )
    second = measure_canonical_json_envelope(
        {"messages": [{"content": "hello"}], "tools": [{"name": "read"}]},
        provider="provider-a",
        model="model-a",
        dialect="dialect-a",
        counter=CharacterCounter(),
        estimate_kind=TokenEstimateKind.HEURISTIC,
        output_token_limit=20,
        shared_reasoning_reserve_tokens=10,
        request_controls={"reasoning_reserve": 10, "max_tokens": 20},
        extra_token_components={"chat_framing": 7, "images": 11},
    )

    assert first == second
    assert first.request_sha256 == second.request_sha256
    assert first.input_tokens == sum(item.tokens for item in first.components)
    assert {item.component_id for item in first.components} == {
        "canonical_json",
        "chat_framing",
        "images",
    }


def test_exact_wire_bytes_have_their_own_hash_and_body_size() -> None:
    components = (
        EnvelopeTokenComponent(
            component_id="full_request",
            tokens=5,
            estimate_kind=TokenEstimateKind.HEURISTIC,
        ),
    )
    compact = measure_precounted_envelope_bytes(
        b'{"a":1}',
        provider="provider-a",
        model="model-a",
        dialect="dialect-a",
        counter=CharacterCounter(),
        components=components,
        output_token_limit=20,
        shared_reasoning_reserve_tokens=10,
        request_controls={"max_tokens": 20, "reasoning_reserve": 10},
    )
    spaced = measure_precounted_envelope_bytes(
        b'{"a": 1}',
        provider="provider-a",
        model="model-a",
        dialect="dialect-a",
        counter=CharacterCounter(),
        components=components,
        output_token_limit=20,
        shared_reasoning_reserve_tokens=10,
        request_controls={"max_tokens": 20, "reasoning_reserve": 10},
    )

    assert compact.request_sha256 != spaced.request_sha256
    assert compact.request_sha256 == sha256(b'{"a":1}').hexdigest()
    assert compact.serialized_request_utf8_bytes == 7
    assert spaced.serialized_request_utf8_bytes == 8


def test_trusted_provider_meter_receives_the_exact_wire_bytes() -> None:
    body = b'{"messages":[{"content":"hello"}],"max_tokens":20}'
    meter = ExactProviderMeter()

    measurement = measure_envelope_with_meter(
        body,
        meter=meter,
    )

    assert meter.observed_body is body
    assert measurement.request_sha256 == sha256(body).hexdigest()
    assert measurement.input_tokens == 7
    assert measurement.token_counter.kind == "provider_exact"
    receipt = require_context_budget(
        measurement,
        profile=_profile(),
        purpose="runtime_l1_step",
        projection_epoch="epoch-a",
        projection_generation=0,
    )
    assert receipt.admitted is True


def test_counter_failure_is_typed_and_fails_closed() -> None:
    counter = CharacterCounter()

    def fail(_text: str) -> int:
        raise RuntimeError("tokenizer unavailable")

    counter.count_text = fail  # type: ignore[method-assign]
    with pytest.raises(ContextBudgetMeasurementError) as caught:
        measure_canonical_json_envelope(
            {"messages": []},
            provider="provider-a",
            model="model-a",
            dialect="dialect-a",
            counter=counter,
            estimate_kind=TokenEstimateKind.TOKENIZER,
            output_token_limit=20,
            shared_reasoning_reserve_tokens=10,
            request_controls={"max_tokens": 20, "reasoning_reserve": 10},
        )

    assert "tokenizer unavailable" not in str(caught.value)


def test_final_gate_catches_caller_input_and_tools_added_after_block_budget() -> None:
    measurement = measure_precounted_envelope(
        {
            "messages": [{"role": "user", "content": "current input"}],
            "tools": [{"type": "function", "function": {"name": "read"}}],
        },
        provider="provider-a",
        model="model-a",
        dialect="dialect-a",
        counter=CharacterCounter(),
        components=(
            EnvelopeTokenComponent(
                component_id="selected_blocks",
                tokens=40,
                estimate_kind=TokenEstimateKind.HEURISTIC,
            ),
            EnvelopeTokenComponent(
                component_id="caller_input",
                tokens=10,
                estimate_kind=TokenEstimateKind.HEURISTIC,
            ),
            EnvelopeTokenComponent(
                component_id="tool_schema",
                tokens=16,
                estimate_kind=TokenEstimateKind.HEURISTIC,
            ),
        ),
        output_token_limit=20,
        shared_reasoning_reserve_tokens=10,
        request_controls={"max_tokens": 20, "reasoning_reserve": 10},
    )

    receipt = check_context_budget(
        measurement,
        profile=_profile(),
        purpose="runtime_l1_step",
        projection_epoch="epoch-a",
        projection_generation=5,
    )

    assert 40 < receipt.input_budget_tokens
    assert receipt.input_tokens == 66
    assert receipt.admitted is False
    assert receipt.violations == (ContextBudgetViolation.INPUT_TOKENS,)


def test_wire_output_controls_must_match_reserved_capacity() -> None:
    measurement = measure_precounted_envelope(
        {"messages": [], "max_tokens": 90},
        provider="provider-a",
        model="model-a",
        dialect="dialect-a",
        counter=CharacterCounter(),
        components=(
            EnvelopeTokenComponent(
                component_id="full_request",
                tokens=1,
                estimate_kind=TokenEstimateKind.HEURISTIC,
            ),
        ),
        output_token_limit=90,
        shared_reasoning_reserve_tokens=10,
        request_controls={"max_tokens": 90, "reasoning_reserve": 10},
    )

    with pytest.raises(ContextBudgetControlMismatch):
        check_context_budget(
            measurement,
            profile=_profile(),
            purpose="runtime_l1_step",
            projection_epoch="epoch-a",
            projection_generation=0,
        )


def test_admitted_request_rejects_swapped_dispatch_bytes() -> None:
    body = canonical_json_bytes(
        {
            "messages": [{"content": "small"}],
            "max_tokens": 20,
            "reasoning_reserve": 10,
        }
    )
    meter = ExactProviderMeter()
    admitted = admit_context_request(
        body,
        meter=meter,
        profile=_profile(),
        purpose="runtime_l1_step",
        projection_epoch="epoch-a",
        projection_generation=0,
    )

    assert admitted.body_for_dispatch() == body
    assert meter.observed_body is body
    assert "small" not in repr(admitted)
    with pytest.raises(ContextBudgetEnvelopeMismatch):
        admitted.require_dispatch_body(
            canonical_json_bytes({"messages": [{"content": "substituted"}]})
        )


def test_admitted_request_cannot_be_reissued_with_dataclass_replace() -> None:
    body = b'{"messages":[],"max_tokens":20,"reasoning_reserve":10}'
    admitted = admit_context_request(
        body,
        meter=ExactProviderMeter(),
        profile=_profile(body_limit=100),
        purpose="runtime_l1_step",
        projection_epoch="epoch-a",
        projection_generation=0,
    )
    oversized = b'{"payload":"' + (b"x" * 1_000) + b'"}'
    forged_receipt = admitted.receipt.model_copy(
        update={
            "request_sha256": sha256(oversized).hexdigest(),
            "serialized_request_utf8_bytes": len(oversized),
        }
    )

    with pytest.raises(TypeError):
        replace(
            admitted,
            receipt=forged_receipt,
            _serialized_envelope=oversized,
        )
    assert not hasattr(admitted, "_admission_seal")
    with pytest.raises(TypeError):
        AdmittedContextRequest(
            receipt=forged_receipt,
            serialized_envelope=oversized,
        )


def test_dispatch_accessor_revalidates_its_seal_and_body_hash() -> None:
    body = b'{"messages":[],"max_tokens":20,"reasoning_reserve":10}'
    admitted = admit_context_request(
        body,
        meter=ExactProviderMeter(),
        profile=_profile(),
        purpose="runtime_l1_step",
        projection_epoch="epoch-a",
        projection_generation=0,
    )

    object.__setattr__(admitted, "_serialized_envelope", b'{"substituted":true}')
    with pytest.raises(ContextBudgetEnvelopeMismatch):
        admitted.body_for_dispatch()

    object.__setattr__(admitted, "_serialized_envelope", body)
    forged_receipt = admitted.receipt.model_copy(
        update={"purpose": "forged-purpose"}
    )
    object.__setattr__(admitted, "_receipt", forged_receipt)
    with pytest.raises(ContextBudgetEnvelopeMismatch):
        admitted.body_for_dispatch()


def test_precounted_measurement_cannot_issue_dispatch_authority() -> None:
    body = b'{"messages":[],"max_tokens":20,"reasoning_reserve":10}'
    measurement = measure_precounted_envelope_bytes(
        body,
        provider="provider-a",
        model="model-a",
        dialect="dialect-a",
        counter=CharacterCounter(),
        components=(
            EnvelopeTokenComponent(
                component_id="full_request",
                tokens=5,
                estimate_kind=TokenEstimateKind.HEURISTIC,
            ),
        ),
        output_token_limit=20,
        shared_reasoning_reserve_tokens=10,
        request_controls={"max_tokens": 20, "reasoning_reserve": 10},
    )

    with pytest.raises(TypeError):
        admit_context_request(
            body,
            measurement,
            profile=_profile(),
            purpose="runtime_l1_step",
            projection_epoch="epoch-a",
            projection_generation=0,
        )


def test_dispatch_gate_uses_controls_extracted_from_the_wire_bytes() -> None:
    body = b'{"messages":[],"max_tokens":999999,"reasoning_reserve":10}'

    with pytest.raises(ContextBudgetControlMismatch):
        admit_context_request(
            body,
            meter=WireControlProviderMeter(),
            profile=_profile(),
            purpose="runtime_l1_step",
            projection_epoch="epoch-a",
            projection_generation=0,
        )


def test_provider_meter_route_cannot_be_relabelled_by_the_caller() -> None:
    body = b'{"messages":[],"max_tokens":20,"reasoning_reserve":10}'

    with pytest.raises(ContextBudgetRouteMismatch):
        admit_context_request(
            body,
            meter=ExactProviderMeter(provider="provider-b"),
            profile=_profile(),
            purpose="runtime_l1_step",
            projection_epoch="epoch-a",
            projection_generation=0,
        )


def test_canonical_fallback_cannot_claim_provider_exact_measurement() -> None:
    with pytest.raises(ValueError, match="cannot claim provider-derived"):
        measure_canonical_json_envelope(
            {"messages": []},
            provider="provider-a",
            model="model-a",
            dialect="dialect-a",
            counter=CharacterCounter(),
            estimate_kind=TokenEstimateKind.PROVIDER_EXACT,
            output_token_limit=20,
            shared_reasoning_reserve_tokens=10,
            request_controls={"max_tokens": 20, "reasoning_reserve": 10},
        )


def test_empty_precounted_bill_fails_closed() -> None:
    with pytest.raises(ValidationError):
        measure_precounted_envelope_bytes(
            b'{"messages":[]}',
            provider="provider-a",
            model="model-a",
            dialect="dialect-a",
            counter=CharacterCounter(),
            components=(),
            output_token_limit=20,
            shared_reasoning_reserve_tokens=10,
            request_controls={"max_tokens": 20, "reasoning_reserve": 10},
        )


def test_provider_meter_rejects_quota_estimate_above_the_hard_bound() -> None:
    with pytest.raises(ValidationError, match="quota input estimate"):
        ProviderEnvelopeMeterResult(
            output_token_limit=20,
            shared_reasoning_reserve_tokens=0,
            request_controls_sha256="a" * 64,
            input_tokens=7,
            quota_input_token_estimate=8,
            components=(
                EnvelopeTokenComponent(
                    component_id="provider_request",
                    tokens=7,
                    estimate_kind=TokenEstimateKind.HEURISTIC,
                ),
            ),
        )


@pytest.mark.parametrize("body", [b"not-json", b"[]", b'{"x":NaN}'])
def test_exact_wire_measurement_requires_a_finite_json_object(body: bytes) -> None:
    with pytest.raises(ContextBudgetEncodingError):
        measure_precounted_envelope_bytes(
            body,
            provider="provider-a",
            model="model-a",
            dialect="dialect-a",
            counter=CharacterCounter(),
            components=(
                EnvelopeTokenComponent(
                    component_id="full_request",
                    tokens=1,
                    estimate_kind=TokenEstimateKind.HEURISTIC,
                ),
            ),
            output_token_limit=20,
            shared_reasoning_reserve_tokens=10,
            request_controls={"max_tokens": 20, "reasoning_reserve": 10},
        )


def test_canonical_fallback_requires_envelope_and_controls_objects() -> None:
    common = {
        "provider": "provider-a",
        "model": "model-a",
        "dialect": "dialect-a",
        "counter": CharacterCounter(),
        "estimate_kind": TokenEstimateKind.HEURISTIC,
        "output_token_limit": 20,
        "shared_reasoning_reserve_tokens": 10,
    }
    with pytest.raises(ContextBudgetEncodingError):
        measure_canonical_json_envelope(
            [],
            request_controls={"max_tokens": 20},
            **common,
        )
    with pytest.raises(ContextBudgetEncodingError):
        measure_canonical_json_envelope(
            {"messages": []},
            request_controls=[],
            **common,
        )


def test_profile_rejects_reserves_that_consume_the_whole_window() -> None:
    with pytest.raises(ValidationError, match="leave no input-token budget"):
        ContextWindowProfile(
            profile_id="bad",
            provider="p",
            model="m",
            dialect="d",
            context_window_tokens=100,
            reserved_output_tokens=80,
            shared_reasoning_reserve_tokens=10,
            safety_margin_tokens=10,
            soft_pressure_ratio=0.8,
        )


def test_numeric_limits_reject_bool_values() -> None:
    with pytest.raises(ValidationError):
        ContextWindowProfile(
            profile_id="bad-bool",
            provider="p",
            model="m",
            dialect="d",
            context_window_tokens=True,
            reserved_output_tokens=0,
            shared_reasoning_reserve_tokens=0,
            safety_margin_tokens=0,
            soft_pressure_ratio=0.8,
        )
