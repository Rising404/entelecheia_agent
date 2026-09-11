from __future__ import annotations

from types import SimpleNamespace

import pytest

from personagraph.model_io.gateway import ModelResult
from personagraph.runtime.model_calls import (
    DurableModelCallReplay,
    DurableModelCallTerminalState,
    request_model_with_retry,
)
from personagraph.runtime.turn_events import RuntimeStage
from tests.helpers.prepared_model_provider import prepare_test_model_request


class _DurableCall:
    semantic_call_id = "logical_replay_validation"

    def __init__(self, replay: DurableModelCallReplay | None) -> None:
        self._replay = replay

    def require_current_state(self) -> None:
        return None

    def reserve(self, *, turn_id: str) -> object:
        return {"turn_id": turn_id}

    def replay_succeeded_result(self) -> DurableModelCallReplay | None:
        return self._replay

    def begin_physical_attempt(self, **_values: object) -> object:
        return SimpleNamespace(
            physical_ordinal=1,
            model_call_id="logical_replay_validation:physical:1",
        )

    def settle_physical_attempt(self, **_values: object) -> None:
        return None

    def typed_result_payload(self, **_values: object) -> object:
        return {"typed": True}

    def success_fingerprint(self, _result: object) -> str:
        return "success-fingerprint"

    def failure_fingerprint(self, **_values: object) -> str:
        return "failure-fingerprint"

    def terminal_state_error(self, message: str) -> DurableModelCallTerminalState:
        return DurableModelCallTerminalState(message)


def _replay(reply: str = "canonical-result") -> DurableModelCallReplay:
    return DurableModelCallReplay(
        model_result=ModelResult(
            reply=reply,
            provider="test",
            model="test",
            latency_ms=0,
            model_call_id="logical_replay_validation:physical:1",
        ),
        physical_ordinal=1,
    )


def _request(
    *,
    durable_call: _DurableCall,
    provider,
    validate,
    validate_replay=None,
):
    return request_model_with_retry(
        turn_id="turn-replay-validation",
        session_id="session-replay-validation",
        purpose="replay-validation-test",
        stage=RuntimeStage.VERIFICATION,
        prepare_request=prepare_test_model_request(provider),
        validate=validate,
        validate_replay=validate_replay,
        emit=lambda _event: None,
        durable_call=durable_call,  # type: ignore[arg-type]
        logical_model_call_id=durable_call.semantic_call_id,
    )


def test_fresh_provider_result_always_uses_strict_validator() -> None:
    strict_calls = 0
    replay_calls = 0

    def provider(model_call_id: str) -> ModelResult:
        return ModelResult(
            reply="provider-wire-result",
            provider="test",
            model="test",
            latency_ms=1,
            model_call_id=model_call_id,
        )

    def validate(result: ModelResult) -> str:
        nonlocal strict_calls
        strict_calls += 1
        assert result.reply == "provider-wire-result"
        return "strict-provider-value"

    def validate_replay(_result: ModelResult) -> str:
        nonlocal replay_calls
        replay_calls += 1
        raise AssertionError("fresh Provider output reached the replay validator")

    outcome = _request(
        durable_call=_DurableCall(None),
        provider=provider,
        validate=validate,
        validate_replay=validate_replay,
    )

    assert outcome.value == "strict-provider-value"
    assert outcome.replayed is False
    assert strict_calls == 1
    assert replay_calls == 0


def test_durable_success_uses_canonical_replay_validator() -> None:
    provider_calls = 0
    strict_calls = 0
    replay_calls = 0

    def provider(_model_call_id: str) -> ModelResult:
        nonlocal provider_calls
        provider_calls += 1
        raise AssertionError("durable success replay reached the Provider")

    def validate(_result: ModelResult) -> str:
        nonlocal strict_calls
        strict_calls += 1
        raise AssertionError("canonical replay reached the Provider-wire validator")

    def validate_replay(result: ModelResult) -> str:
        nonlocal replay_calls
        replay_calls += 1
        assert result.reply == "canonical-result"
        return "canonical-value"

    outcome = _request(
        durable_call=_DurableCall(_replay()),
        provider=provider,
        validate=validate,
        validate_replay=validate_replay,
    )

    assert outcome.value == "canonical-value"
    assert outcome.replayed is True
    assert provider_calls == 0
    assert strict_calls == 0
    assert replay_calls == 1


def test_durable_success_defaults_to_existing_validator() -> None:
    validate_calls = 0

    def validate(result: ModelResult) -> str:
        nonlocal validate_calls
        validate_calls += 1
        return f"default:{result.reply}"

    outcome = _request(
        durable_call=_DurableCall(_replay()),
        provider=lambda _model_call_id: pytest.fail("replay reached Provider"),
        validate=validate,
    )

    assert outcome.value == "default:canonical-result"
    assert outcome.replayed is True
    assert validate_calls == 1


def test_replay_validation_failure_fails_closed_without_provider_io() -> None:
    provider_calls = 0
    strict_calls = 0

    def provider(_model_call_id: str) -> ModelResult:
        nonlocal provider_calls
        provider_calls += 1
        raise AssertionError("invalid durable replay reached the Provider")

    def validate(_result: ModelResult) -> str:
        nonlocal strict_calls
        strict_calls += 1
        raise AssertionError("invalid durable replay reached the strict validator")

    with pytest.raises(DurableModelCallTerminalState, match="replay validation"):
        _request(
            durable_call=_DurableCall(_replay("invalid-canonical-result")),
            provider=provider,
            validate=validate,
            validate_replay=lambda _result: (_ for _ in ()).throw(
                ValueError("canonical result is corrupt")
            ),
        )

    assert provider_calls == 0
    assert strict_calls == 0
