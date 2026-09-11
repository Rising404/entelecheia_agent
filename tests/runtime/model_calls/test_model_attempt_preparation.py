"""Focused coverage for the pre-authority model request preparation gate."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import pytest

from personagraph.model_io.gateway import ModelGatewayError, ModelResult
from personagraph.model_io.output_repair_contracts import (
    RuntimeModelOutputRepairFeedback,
)
from personagraph.model_io.tier_bindings import (
    EndpointOrigin,
    ModelTier,
    ModelTierBinding,
    current_model_tier_binding,
)
from personagraph.runtime.model_calls.attempt_preparation import (
    prepare_model_attempt_request,
)


@dataclass(frozen=True)
class _Prepared:
    label: str

    def dispatch(self, *, model_call_id: str) -> ModelResult:
        raise AssertionError("preparation tests must not dispatch Provider I/O")


class _DurableCall:
    def __init__(self, model_binding: object) -> None:
        self.model_binding = model_binding

    def terminal_state_error(self, message: str) -> RuntimeError:
        return RuntimeError(message)


def _feedback() -> RuntimeModelOutputRepairFeedback:
    # Validation of this contract belongs to model_io.  This unit only needs a
    # typed identity to prove that preparation forwards the exact checkpoint.
    return cast(RuntimeModelOutputRepairFeedback, object())


def test_preparation_selects_the_exact_request_variant_under_frozen_binding() -> None:
    binding = ModelTierBinding(
        tier=ModelTier.ATTEMPT,
        provider="test",
        base_url="https://example.invalid",
        model="test-model",
        thinking_enabled=False,
        origin=EndpointOrigin.GLOBAL,
    )
    durable_call = _DurableCall(binding)
    feedback = _feedback()
    observed: list[object] = []

    def prepare_initial() -> _Prepared:
        observed.append(("initial", current_model_tier_binding()))
        return _Prepared("initial")

    def prepare_repair(
        candidate_feedback: RuntimeModelOutputRepairFeedback,
        rejected_response_text: str,
    ) -> _Prepared:
        observed.append(
            (
                "repair",
                current_model_tier_binding(),
                candidate_feedback,
                rejected_response_text,
            )
        )
        return _Prepared("repair")

    initial = prepare_model_attempt_request(
        purpose="test",
        prepare_request=prepare_initial,
        prepare_repair_request=prepare_repair,
        repair_feedback=None,
        rejected_response_text=None,
        remaining_s=3.0,
        durable_call=durable_call,  # type: ignore[arg-type]
    )
    repaired = prepare_model_attempt_request(
        purpose="test",
        prepare_request=prepare_initial,
        prepare_repair_request=prepare_repair,
        repair_feedback=feedback,
        rejected_response_text="rejected",
        remaining_s=2.0,
        durable_call=durable_call,  # type: ignore[arg-type]
    )

    assert initial.label == "initial"
    assert repaired.label == "repair"
    assert observed == [
        ("initial", binding),
        ("repair", binding, feedback, "rejected"),
    ]
    assert current_model_tier_binding() is None


@pytest.mark.parametrize(
    ("durable_repair_recovery", "expected_message"),
    (
        (False, "Repair preparation requires its rejected response."),
        (
            True,
            "Repair recovery requires its prepared callback and rejected response.",
        ),
    ),
)
def test_preparation_preserves_missing_repair_material_errors(
    durable_repair_recovery: bool,
    expected_message: str,
) -> None:
    with pytest.raises(ModelGatewayError) as captured:
        prepare_model_attempt_request(
            purpose="test-purpose",
            prepare_request=lambda: _Prepared("initial"),
            prepare_repair_request=None,
            repair_feedback=_feedback(),
            rejected_response_text="rejected",
            remaining_s=None,
            durable_call=None,
            durable_repair_recovery=durable_repair_recovery,
        )

    assert captured.value.code == "MODEL_CONFIGURATION_FAILURE"
    assert captured.value.message == expected_message
    assert captured.value.retryable is False
    assert captured.value.details == {
        "purpose": "test-purpose",
        "reason": "missing_repair_material",
    }


@pytest.mark.parametrize(
    ("durable_repair_recovery", "expected_callback_name"),
    ((False, "prepare_request"), (True, "prepare_repair_request")),
)
def test_preparation_preserves_invalid_callback_result_errors(
    durable_repair_recovery: bool,
    expected_callback_name: str,
) -> None:
    feedback = _feedback() if durable_repair_recovery else None

    with pytest.raises(TypeError) as captured:
        prepare_model_attempt_request(
            purpose="test",
            prepare_request=lambda: object(),  # type: ignore[return-value]
            prepare_repair_request=(
                (lambda _feedback, _response: object())  # type: ignore[return-value]
                if durable_repair_recovery
                else None
            ),
            repair_feedback=feedback,
            rejected_response_text=(
                "rejected" if durable_repair_recovery else None
            ),
            remaining_s=None,
            durable_call=None,
            durable_repair_recovery=durable_repair_recovery,
        )

    assert str(captured.value) == (
        f"{expected_callback_name} must return an object with "
        "dispatch(*, model_call_id)"
    )


def test_preparation_rejects_an_invalid_durable_binding_before_callback() -> None:
    prepared = False

    def prepare_initial() -> _Prepared:
        nonlocal prepared
        prepared = True
        return _Prepared("initial")

    with pytest.raises(
        RuntimeError,
        match="durable model binding has the wrong contract",
    ):
        prepare_model_attempt_request(
            purpose="test",
            prepare_request=prepare_initial,
            prepare_repair_request=None,
            repair_feedback=None,
            rejected_response_text=None,
            remaining_s=None,
            durable_call=_DurableCall(object()),  # type: ignore[arg-type]
        )

    assert prepared is False
