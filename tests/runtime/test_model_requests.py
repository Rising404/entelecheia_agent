from __future__ import annotations

import hashlib
import inspect
from types import SimpleNamespace

import pytest

from personagraph.model_io import gateway as models
from personagraph.context_budget import ContextBudgetExceeded
from personagraph.model_io.gateway import ModelGatewayError, ModelResult
from personagraph.model_io.tier_bindings import (
    EndpointOrigin,
    ModelTierBinding,
    ModelTier,
    current_model_tier_binding,
)
from personagraph.runtime.model_calls import requests as model_requests
from personagraph.runtime.model_calls import (
    DurableModelCallReplay,
    DurableModelCallTerminalState,
)
from personagraph.runtime.model_calls import (
    BACKOFF_MAX_S,
    backoff_delay_s,
)
from personagraph.model_io.output_repair_contracts import (
    RUNTIME_MODEL_REJECTED_OUTPUT_MAX_UTF8_BYTES,
    RuntimeModelOutputRepairFeedback,
    RuntimeModelOutputRepairIssue,
    RuntimeModelOutputRepairIssueCoverage,
    RuntimeModelOutputRepairProtocol,
    runtime_model_output_repair_issue_sort_key,
)
from personagraph.runtime.model_calls import request_model_with_retry
from personagraph.model_io.output_validation import ModelOutputValidationError
from personagraph.runtime.turn_events import RuntimeStage
from tests.helpers.prepared_model_provider import prepare_test_model_request


def _result(model_call_id: str) -> ModelResult:
    return ModelResult(reply="ok", provider="test", model="test", latency_ms=1, model_call_id=model_call_id)


@pytest.mark.parametrize("inner_failure", ["format", "transport", "prepare"])
def test_exhausted_nested_review_never_regenerates_the_outer_response(
    monkeypatch, inner_failure: str,
) -> None:
    """A review failure owns its budget; the accepted draft is not another retry."""
    dispatches = {"draft": 0, "review": 0}
    outer_repairs: list[object] = []
    events: list[object] = []
    monkeypatch.setattr(model_requests, "_sleep", lambda _seconds: None)
    monkeypatch.setenv("PERSONAGRAPH_TRAJECTORY", "off")

    def draft(call_id: str) -> ModelResult:
        dispatches["draft"] += 1
        return _result(call_id)

    def review(call_id: str) -> ModelResult:
        dispatches["review"] += 1
        if inner_failure == "transport":
            raise ModelGatewayError("MODEL_CALL_FAILED", "offline failure", retryable=True)
        return _result(call_id)

    def validate_review(_result: ModelResult) -> str:
        raise ModelOutputValidationError("invalid offline review")

    def prepare_review():
        if inner_failure == "prepare":
            raise ModelGatewayError("MODEL_CALL_FAILED", "offline preparation failure", retryable=True)
        return prepare_test_model_request(review)()

    def validate_draft(_result: ModelResult) -> str:
        return request_model_with_retry(
            turn_id="nested-turn", session_id="nested-session", purpose="review",
            stage=RuntimeStage.VERIFICATION,
            prepare_request=prepare_review,
            prepare_repair_request=lambda _feedback, _body: prepare_test_model_request(review)(),
            repair_target_contract="offline-review", validate=validate_review,
            emit=events.append,
        ).value

    def repair_draft(feedback, _body):
        outer_repairs.append(feedback)
        return prepare_test_model_request(draft)()

    with pytest.raises(ModelGatewayError) as captured:
        request_model_with_retry(
            turn_id="nested-turn", session_id="nested-session", purpose="draft",
            stage=RuntimeStage.L1_BOOTSTRAP,
            prepare_request=prepare_test_model_request(draft),
            prepare_repair_request=repair_draft, repair_target_contract="offline-draft",
            validate=validate_draft, emit=events.append,
        )

    assert dispatches == {"draft": 1, "review": 0 if inner_failure == "prepare" else 6}
    assert outer_repairs == []
    assert captured.value.retryable is False
    assert captured.value.code == (
        "MODEL_BAD_RESPONSE" if inner_failure == "format" else "MODEL_CALL_FAILED"
    )
    assert events[-1].stage is RuntimeStage.L1_BOOTSTRAP
    assert events[-1].retryable is False
    assert events[-1].error_code.value == (
        "MODEL_OUTPUT_INVALID" if inner_failure == "format" else "MODEL_TRANSPORT_FAILURE"
    )


def test_retry_wrapper_exposes_only_the_prepared_request_boundary() -> None:
    parameters = inspect.signature(request_model_with_retry).parameters

    assert "request" not in parameters
    assert parameters["prepare_request"].default is inspect.Parameter.empty
    assert not hasattr(model_requests, "_dispatch_unprepared_request")


def test_quota_wait_precedes_durable_physical_attempt_and_started_event() -> None:
    operations: list[object] = []
    permit = object()

    class Prepared:
        def dispatch(self, *, model_call_id: str) -> ModelResult:
            raise AssertionError("quota-managed request must use its leased dispatch")

        def acquire_api_quota(
            self, *, model_call_id: str, wait_timeout_seconds: float
        ) -> object:
            operations.append(("quota_acquire", model_call_id, wait_timeout_seconds))
            return permit

        def abandon_api_quota(
            self, _permit: object | None, *, disposition: str = "cancel"
        ) -> None:
            operations.append(("quota_abandon", disposition))

        def dispatch_with_api_quota(
            self, *, model_call_id: str, permit: object | None
        ) -> ModelResult:
            assert permit is not None
            operations.append(("provider_dispatch", model_call_id))
            return _result(model_call_id)

    class DurableCall:
        semantic_call_id = "quota-logical-1"

        def require_current_state(self) -> None:
            operations.append("guard")

        def reserve(self, *, turn_id: str) -> object:
            operations.append(("logical_reserve", turn_id))
            return object()

        def replay_succeeded_result(self) -> None:
            operations.append("replay")
            return None

        def begin_physical_attempt(
            self, *, turn_id: str, max_physical_attempts: int
        ) -> object:
            operations.append(("physical_begin", turn_id, max_physical_attempts))
            return SimpleNamespace(
                physical_attempt_id="quota-physical-1",
                physical_ordinal=1,
                provider="test",
                model="test",
                model_call_id="quota-logical-1:physical:1",
            )

        def settle_physical_attempt(self, **values: object) -> None:
            operations.append(("physical_settle", values["outcome"]))

        def typed_result_payload(self, **_values: object) -> object:
            return {"validated": "ok"}

        def success_fingerprint(self, _result: object) -> str:
            return "quota-success"

        def failure_fingerprint(self, **_values: object) -> str:
            return "quota-failure"

        def terminal_state_error(self, message: str) -> DurableModelCallTerminalState:
            return DurableModelCallTerminalState(message)

    outcome = request_model_with_retry(
        turn_id="turn-quota-order",
        session_id="session-quota-order",
        purpose="quota-order",
        stage=RuntimeStage.VERIFICATION,
        prepare_request=lambda: Prepared(),
        validate=lambda result: result.reply,
        emit=lambda event: operations.append(("event", event.status.value)),
        durable_call=DurableCall(),  # type: ignore[arg-type]
        logical_model_call_id="quota-logical-1",
    )

    assert outcome.value == "ok"
    acquire_index = next(
        index for index, item in enumerate(operations)
        if isinstance(item, tuple) and item[0] == "quota_acquire"
    )
    physical_index = next(
        index for index, item in enumerate(operations)
        if isinstance(item, tuple) and item[0] == "physical_begin"
    )
    started_index = operations.index(("event", "started"))
    dispatch_index = next(
        index for index, item in enumerate(operations)
        if isinstance(item, tuple) and item[0] == "provider_dispatch"
    )
    assert acquire_index < physical_index < started_index < dispatch_index
    assert operations[acquire_index + 1] == "guard"
    assert physical_index == acquire_index + 2
    assert started_index == physical_index + 1
    assert dispatch_index == started_index + 1


@pytest.mark.parametrize("turn_expired_after_wait", [False, True])
def test_quota_wait_deadline_consumes_no_physical_attempt(turn_expired_after_wait: bool) -> None:
    operations: list[str] = []

    class Deadline:
        def expired(self) -> bool:
            return turn_expired_after_wait and "quota_wait" in operations

        def remaining_s(self) -> float:
            return 2.0

    class Prepared:
        def dispatch(self, *, model_call_id: str) -> ModelResult:
            raise AssertionError("provider must not run")

        def acquire_api_quota(self, **_values: object) -> object:
            operations.append("quota_wait")
            raise ModelGatewayError(
                "MODEL_QUOTA_WAIT_TIMEOUT",
                "quota deadline",
                retryable=False,
            )

        def abandon_api_quota(self, *_args: object, **_kwargs: object) -> None:
            operations.append("abandon")

        def dispatch_with_api_quota(self, **_values: object) -> ModelResult:
            raise AssertionError("provider must not run")

    class DurableCall:
        semantic_call_id = "quota-timeout-logical"

        def require_current_state(self) -> None:
            operations.append("guard")

        def reserve(self, *, turn_id: str) -> object:
            operations.append("logical_reserve")
            return object()

        def replay_succeeded_result(self) -> None:
            return None

        def begin_physical_attempt(self, **_values: object) -> object:
            operations.append("physical_begin")
            raise AssertionError("queue waiting must not consume a physical attempt")

        def terminal_state_error(self, message: str) -> DurableModelCallTerminalState:
            return DurableModelCallTerminalState(message)

    with pytest.raises(ModelGatewayError) as captured:
        request_model_with_retry(
            turn_id="turn-quota-timeout",
            session_id="session-quota-timeout",
            purpose="quota-timeout",
            stage=RuntimeStage.VERIFICATION,
            prepare_request=lambda: Prepared(),
            validate=lambda result: result.reply,
            emit=lambda _event: None,
            deadline=Deadline(),  # type: ignore[arg-type]
            durable_call=DurableCall(),  # type: ignore[arg-type]
            logical_model_call_id="quota-timeout-logical",
        )

    assert captured.value.code == (
        "TURN_DEADLINE_EXCEEDED" if turn_expired_after_wait else "MODEL_QUOTA_WAIT_TIMEOUT"
    )
    assert "physical_begin" not in operations


def test_turn_deadline_clamps_the_real_provider_http_timeout(monkeypatch) -> None:
    captured_timeouts: list[float] = []

    class Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {
                "choices": [
                    {"message": {"content": "ok"}, "finish_reason": "stop"}
                ],
                "usage": {},
            }

    class Client:
        def __init__(self, *, timeout: float) -> None:
            captured_timeouts.append(timeout)

        def __enter__(self):
            return self

        def __exit__(self, *_exc: object) -> bool:
            return False

        def post(self, *_args: object, **_kwargs: object) -> Response:
            return Response()

    class Deadline:
        def expired(self) -> bool:
            return False

        def remaining_s(self) -> float:
            return 7.25

    monkeypatch.setattr(models.httpx, "Client", Client)
    binding = ModelTierBinding(
        tier=ModelTier.ATTEMPT,
        provider="openai-compatible",
        base_url="https://example.invalid/v1",
        model="test-model",
        api_key="test-key",
        thinking_enabled=False,
        origin=EndpointOrigin.PROFILE,
    )

    outcome = request_model_with_retry(
        turn_id="turn-timeout-clamp",
        session_id="session-timeout-clamp",
        purpose="timeout-clamp",
        stage=RuntimeStage.L2_UNDERSTAND,
        prepare_request=prepare_test_model_request(
            lambda model_call_id: models.openai_compatible_chat(
                [{"role": "user", "content": "hello"}],
                timeout_s=600.0,
                model_call_id=model_call_id,
                purpose="timeout-clamp",
                binding=binding,
            )
        ),
        validate=lambda result: result.reply,
        emit=lambda _event: None,
        deadline=Deadline(),  # type: ignore[arg-type]
    )

    assert outcome.value == "ok"
    assert captured_timeouts == [7.25]


def test_deadline_expiring_immediately_before_provider_prevents_http_io() -> None:
    provider_calls = 0
    events = []

    class Deadline:
        def expired(self) -> bool:
            return False

        def remaining_s(self) -> float:
            return 0.0

    def provider(_model_call_id: str) -> ModelResult:
        nonlocal provider_calls
        provider_calls += 1
        raise AssertionError("expired Turn must not reach provider I/O")

    with pytest.raises(ModelGatewayError) as captured:
        request_model_with_retry(
            turn_id="turn-expired-before-provider",
            session_id="session-expired-before-provider",
            purpose="timeout-clamp",
            stage=RuntimeStage.L2_UNDERSTAND,
            prepare_request=prepare_test_model_request(provider),
            validate=lambda result: result.reply,
            emit=events.append,
            deadline=Deadline(),  # type: ignore[arg-type]
        )

    assert captured.value.code == "TURN_DEADLINE_EXCEEDED"
    assert provider_calls == 0
    assert events[-1].error_code.value == "TURN_DEADLINE_EXCEEDED"


def test_retry_wrapper_retries_invalid_typed_output_up_to_six_attempts():
    calls = 0
    events = []

    def request(model_call_id: str) -> ModelResult:
        nonlocal calls
        calls += 1
        return _result(model_call_id)

    def invalid(_result: ModelResult) -> str:
        raise ModelOutputValidationError("invalid")

    with pytest.raises(ModelGatewayError) as exc:
        request_model_with_retry(
            turn_id="turn-1",
            session_id="session-1",
            purpose="test",
            stage=RuntimeStage.CLASSIFY,
            prepare_request=prepare_test_model_request(request),
            validate=invalid,
            emit=events.append,
        )

    assert calls == 6
    assert exc.value.code == "MODEL_BAD_RESPONSE"
    failures = [event for event in events if event.status.value == "failed"]
    assert len(failures) == 6
    assert failures[-1].retryable is False


def test_host_logical_call_id_is_used_for_provider_events_and_result() -> None:
    provider_ids: list[str] = []
    events = []

    def request(model_call_id: str) -> ModelResult:
        provider_ids.append(model_call_id)
        return _result(model_call_id)

    outcome = request_model_with_retry(
        turn_id="turn-1",
        session_id="session-1",
        purpose="test",
        stage=RuntimeStage.VERIFICATION,
        prepare_request=prepare_test_model_request(request),
        validate=lambda result: result.reply,
        emit=events.append,
        logical_model_call_id="logical_call_01",
    )

    assert provider_ids == ["logical_call_01"]
    assert outcome.model_call_id == "logical_call_01"
    assert {event.model_call_id for event in events} == {"logical_call_01"}


def test_shared_wrapper_accepts_provider_neutral_durable_authority() -> None:
    operations: list[object] = []

    class GenericDurableCall:
        semantic_call_id = "generic_logical_01"

        def require_current_state(self) -> None:
            operations.append("guard")

        def reserve(self, *, turn_id: str) -> object:
            operations.append(("reserve", turn_id))
            return object()

        def begin_physical_attempt(
            self, *, turn_id: str, max_physical_attempts: int
        ) -> object:
            operations.append(("begin", turn_id, max_physical_attempts))
            return SimpleNamespace(
                physical_attempt_id="generic_physical_01",
                physical_ordinal=1,
                provider="test",
                model="test",
                model_call_id="generic_logical_01:physical:1",
            )

        def replay_succeeded_result(self) -> None:
            operations.append("replay_check")
            return None

        def settle_physical_attempt(self, **values: object) -> None:
            operations.append(
                (
                    "settle",
                    values["outcome"],
                    values["result_fingerprint"],
                    values.get("typed_result"),
                )
            )

        def typed_result_payload(
            self, *, model_result: object, value: object
        ) -> object:
            assert isinstance(model_result, ModelResult)
            return {"validated": value}

        def success_fingerprint(self, result: object) -> str:
            assert isinstance(result, ModelResult)
            return "success-fingerprint"

        def failure_fingerprint(self, **_values: object) -> str:
            return "failure-fingerprint"

        def terminal_state_error(self, message: str) -> DurableModelCallTerminalState:
            return DurableModelCallTerminalState(message)

    call = GenericDurableCall()
    outcome = request_model_with_retry(
        turn_id="turn-1",
        session_id="session-1",
        purpose="generic-ledger-test",
        stage=RuntimeStage.VERIFICATION,
        prepare_request=prepare_test_model_request(
            lambda model_call_id: _result(model_call_id)
        ),
        validate=lambda result: result.reply,
        emit=lambda _event: None,
        durable_call=call,  # type: ignore[arg-type]
        logical_model_call_id=call.semantic_call_id,
    )

    assert outcome.model_call_id == "generic_logical_01:physical:1"
    assert operations == [
        "guard",
        "guard",
        ("reserve", "turn-1"),
        "replay_check",
        "guard",
        ("begin", "turn-1", 6),
        "guard",
        ("settle", "succeeded", "success-fingerprint", {"validated": "ok"}),
    ]


def test_prepared_request_gates_before_reservation_and_dispatches_physical_id(
) -> None:
    operations: list[object] = []
    binding = ModelTierBinding(
        tier=ModelTier.ATTEMPT,
        provider="openai-compatible",
        base_url="https://example.invalid/v1",
        model="prepared-model",
        api_key="test-key",
        thinking_enabled=False,
        origin=EndpointOrigin.PROFILE,
    )

    class Deadline:
        def expired(self) -> bool:
            return False

        def remaining_s(self) -> float:
            return 7.25

    class Prepared:
        def dispatch(self, *, model_call_id: str) -> ModelResult:
            assert current_model_tier_binding() is binding
            assert models._effective_model_timeout_s(600.0) == 7.25
            operations.append(("dispatch", model_call_id))
            return _result(model_call_id)

    class GenericDurableCall:
        semantic_call_id = "prepared_logical_01"
        model_binding = binding

        def require_current_state(self) -> None:
            operations.append("guard")

        def reserve(self, *, turn_id: str) -> object:
            operations.append(("reserve", turn_id))
            return object()

        def replay_succeeded_result(self) -> None:
            operations.append("replay_check")
            return None

        def begin_physical_attempt(
            self, *, turn_id: str, max_physical_attempts: int
        ) -> object:
            operations.append(("begin", turn_id, max_physical_attempts))
            return SimpleNamespace(
                physical_attempt_id="prepared_physical_01",
                physical_ordinal=1,
                provider="test",
                model="prepared-model",
                model_call_id="prepared_logical_01:physical:1",
            )

        def settle_physical_attempt(self, **values: object) -> None:
            operations.append(("settle", values["outcome"]))

        def typed_result_payload(self, **_values: object) -> object:
            return {"validated": "ok"}

        def success_fingerprint(self, _result: object) -> str:
            return "prepared-success"

        def failure_fingerprint(self, **_values: object) -> str:
            return "prepared-failure"

        def terminal_state_error(self, message: str) -> DurableModelCallTerminalState:
            return DurableModelCallTerminalState(message)

    def prepare() -> Prepared:
        assert current_model_tier_binding() is binding
        assert models._effective_model_timeout_s(600.0) == 7.25
        operations.append("prepare")
        return Prepared()

    outcome = request_model_with_retry(
        turn_id="turn-prepared",
        session_id="session-prepared",
        purpose="prepared-order",
        stage=RuntimeStage.VERIFICATION,
        prepare_request=prepare,
        validate=lambda result: result.reply,
        emit=lambda event: operations.append(("event", event.status.value)),
        deadline=Deadline(),  # type: ignore[arg-type]
        durable_call=GenericDurableCall(),  # type: ignore[arg-type]
        logical_model_call_id="prepared_logical_01",
    )

    assert outcome.model_call_id == "prepared_logical_01:physical:1"
    assert current_model_tier_binding() is None
    assert operations == [
        "guard",
        "prepare",
        "guard",
        ("reserve", "turn-prepared"),
        "replay_check",
        ("begin", "turn-prepared", 6),
        ("event", "started"),
        ("dispatch", "prepared_logical_01:physical:1"),
        "guard",
        ("settle", "succeeded"),
        ("event", "completed"),
    ]


def test_prepared_gate_rejection_consumes_no_durable_or_event_authority() -> None:
    operations: list[str] = []

    class DurableCall:
        semantic_call_id = "prepared_rejected_01"

        def require_current_state(self) -> None:
            operations.append("guard")

        def reserve(self, *, turn_id: str) -> object:
            operations.append("reserve")
            raise AssertionError("context gate rejection must precede reserve")

        def terminal_state_error(self, message: str) -> DurableModelCallTerminalState:
            return DurableModelCallTerminalState(message)

    def reject() -> object:
        operations.append("prepare")
        raise ContextBudgetExceeded(limit=100, estimated_tokens=101)

    events: list[object] = []
    with pytest.raises(ContextBudgetExceeded):
        request_model_with_retry(
            turn_id="turn-prepared-rejected",
            session_id="session-prepared",
            purpose="prepared-rejected",
            stage=RuntimeStage.VERIFICATION,
            prepare_request=reject,  # type: ignore[arg-type]
            validate=lambda result: result.reply,
            emit=events.append,
            durable_call=DurableCall(),  # type: ignore[arg-type]
            logical_model_call_id="prepared_rejected_01",
        )

    assert operations == ["guard", "prepare"]
    assert events == []


def test_prepared_repair_gate_rejection_does_not_open_physical_attempt_two(
) -> None:
    operations: list[object] = []
    physical_ordinals: list[int] = []
    repair_feedback = []

    class Prepared:
        def dispatch(self, *, model_call_id: str) -> ModelResult:
            operations.append(("dispatch", model_call_id))
            return ModelResult(
                reply="invalid",
                provider="test",
                model="test",
                latency_ms=1,
                model_call_id=model_call_id,
            )

    class DurableCall:
        semantic_call_id = "prepared_repair_01"
        logical_request = SimpleNamespace(
            output_repair_protocol=(
                RuntimeModelOutputRepairProtocol.FOUR_MESSAGE_WHOLE_RESPONSE_REGENERATION
            ),
            typed_result_contract="prepared-repair-result-v1",
        )

        def require_current_state(self) -> None:
            operations.append("guard")

        def reserve(self, *, turn_id: str) -> object:
            operations.append("reserve")
            return object()

        def replay_succeeded_result(self) -> None:
            operations.append("replay_check")
            return None

        def recover_output_repair_feedback(self) -> None:
            operations.append("recover_repair")
            return None

        def begin_physical_attempt(self, **values: object) -> object:
            ordinal = len(physical_ordinals) + 1
            physical_ordinals.append(ordinal)
            operations.append(("begin", ordinal, values["output_repair_enabled"]))
            return SimpleNamespace(
                physical_attempt_id=f"prepared_repair_physical_{ordinal}",
                physical_ordinal=ordinal,
                provider="test",
                model="test",
                model_call_id=f"prepared_repair_01:physical:{ordinal}",
            )

        def settle_physical_attempt(self, **values: object) -> None:
            operations.append(("settle", values["outcome"]))

        def typed_result_payload(self, **_values: object) -> object:
            raise AssertionError("invalid response cannot produce typed result")

        def success_fingerprint(self, _result: object) -> str:
            raise AssertionError("invalid response cannot succeed")

        def failure_fingerprint(self, **_values: object) -> str:
            return "prepared-repair-failure"

        def terminal_state_error(self, message: str) -> DurableModelCallTerminalState:
            return DurableModelCallTerminalState(message)

    def prepare() -> Prepared:
        operations.append("prepare_initial")
        return Prepared()

    def prepare_repair(feedback, rejected_response_text: str) -> object:
        repair_feedback.append(feedback)
        assert rejected_response_text == "invalid"
        operations.append("prepare_repair")
        raise ContextBudgetExceeded(limit=100, estimated_tokens=101)

    events: list[object] = []
    with pytest.raises(ContextBudgetExceeded):
        request_model_with_retry(
            turn_id="turn-prepared-repair",
            session_id="session-prepared",
            purpose="prepared-repair",
            stage=RuntimeStage.VERIFICATION,
            prepare_request=prepare,
            prepare_repair_request=prepare_repair,
            repair_target_contract="prepared-repair-result-v1",
            validate=lambda _result: (_ for _ in ()).throw(
                ModelOutputValidationError(
                    "invalid",
                    repair_code="test_invalid",
                    safe_repair_reason="The typed result is invalid.",
                )
            ),
            emit=events.append,
            durable_call=DurableCall(),  # type: ignore[arg-type]
            logical_model_call_id="prepared_repair_01",
        )

    assert physical_ordinals == [1]
    assert len(repair_feedback) == 1
    assert repair_feedback[0].rejected_physical_ordinal == 1
    assert [event.status.value for event in events] == ["started", "failed"]
    assert operations.count("reserve") == 1
    assert ("dispatch", "prepared_repair_01:physical:1") in operations
    assert "prepare_repair" in operations


def test_started_event_failure_settles_uninvoked_durable_attempt_as_retryable() -> None:
    operations: list[object] = []
    provider_calls = 0

    class _EventSinkFailure(RuntimeError):
        pass

    class GenericDurableCall:
        semantic_call_id = "generic_logical_emit_failure"

        def require_current_state(self) -> None:
            operations.append("guard")

        def reserve(self, *, turn_id: str) -> object:
            operations.append(("reserve", turn_id))
            return object()

        def replay_succeeded_result(self) -> None:
            operations.append("replay_check")
            return None

        def begin_physical_attempt(
            self, *, turn_id: str, max_physical_attempts: int
        ) -> object:
            operations.append(("begin", turn_id, max_physical_attempts))
            return SimpleNamespace(
                physical_attempt_id="generic_physical_emit_failure",
                physical_ordinal=1,
                provider="test",
                model="test",
                model_call_id="generic_logical_emit_failure:physical:1",
            )

        def settle_physical_attempt(self, **values: object) -> None:
            operations.append(
                (
                    "settle",
                    values["outcome"],
                    values["result_fingerprint"],
                    values["error_code"],
                )
            )

        def typed_result_payload(self, **_values: object) -> object:
            raise AssertionError("event failure cannot produce a typed result")

        def success_fingerprint(self, _result: object) -> str:
            raise AssertionError("event failure cannot produce a success")

        def failure_fingerprint(
            self, *, error: BaseException, provider_result: object | None
        ) -> str:
            assert isinstance(error, _EventSinkFailure)
            assert provider_result is None
            return "event-failure-fingerprint"

        def terminal_state_error(self, message: str) -> DurableModelCallTerminalState:
            return DurableModelCallTerminalState(message)

    def provider(_model_call_id: str) -> ModelResult:
        nonlocal provider_calls
        provider_calls += 1
        raise AssertionError("STARTED event failure must precede Provider I/O")

    def emit(_event: object) -> None:
        raise _EventSinkFailure("runtime event sink rejected STARTED")

    with pytest.raises(_EventSinkFailure, match="event sink rejected"):
        request_model_with_retry(
            turn_id="turn-emit-failure",
            session_id="session-emit-failure",
            purpose="generic-ledger-event-failure",
            stage=RuntimeStage.VERIFICATION,
            prepare_request=prepare_test_model_request(provider),
            validate=lambda result: result.reply,
            emit=emit,
            durable_call=GenericDurableCall(),  # type: ignore[arg-type]
            logical_model_call_id="generic_logical_emit_failure",
        )

    assert provider_calls == 0
    assert operations == [
        "guard",
        "guard",
        ("reserve", "turn-emit-failure"),
        "replay_check",
        "guard",
        ("begin", "turn-emit-failure", 6),
        (
            "settle",
            "retryable_failure",
            "event-failure-fingerprint",
            "RUNTIME_EVENT_EMIT_FAILED",
        ),
    ]


def test_durable_typed_success_replays_without_provider_io() -> None:
    operations: list[object] = []
    provider_calls = 0

    class ReplayDurableCall:
        semantic_call_id = "generic_logical_replay"

        def require_current_state(self) -> None:
            operations.append("guard")

        def reserve(self, *, turn_id: str) -> object:
            operations.append(("reserve", turn_id))
            return object()

        def replay_succeeded_result(self) -> DurableModelCallReplay:
            operations.append("replay_check")
            return DurableModelCallReplay(
                model_result=ModelResult(
                    reply='{"answer":"persisted"}',
                    provider="test",
                    model="test",
                    latency_ms=0,
                    finish_reason="replayed_typed_result",
                    model_call_id="generic_logical_replay:physical:2",
                ),
                physical_ordinal=2,
            )

        def begin_physical_attempt(self, **_values: object) -> object:
            raise AssertionError("replay must not open a new physical attempt")

        def settle_physical_attempt(self, **_values: object) -> None:
            raise AssertionError("replay must not settle another attempt")

        def typed_result_payload(self, **_values: object) -> object:
            raise AssertionError("replay has no fresh typed result")

        def success_fingerprint(self, _result: object) -> str:
            raise AssertionError("replay has no fresh provider result")

        def failure_fingerprint(self, **_values: object) -> str:
            raise AssertionError("replay has no fresh provider failure")

        def terminal_state_error(self, message: str) -> DurableModelCallTerminalState:
            return DurableModelCallTerminalState(message)

    def provider(_model_call_id: str) -> ModelResult:
        nonlocal provider_calls
        provider_calls += 1
        raise AssertionError("durable success must suppress provider I/O")

    events: list[object] = []
    outcome = request_model_with_retry(
        turn_id="turn-replay",
        session_id="session-1",
        purpose="generic-ledger-replay",
        stage=RuntimeStage.VERIFICATION,
        prepare_request=prepare_test_model_request(provider),
        validate=lambda result: result.reply,
        emit=events.append,
        durable_call=ReplayDurableCall(),  # type: ignore[arg-type]
        logical_model_call_id="generic_logical_replay",
    )

    assert provider_calls == 0
    assert outcome.value == '{"answer":"persisted"}'
    assert outcome.attempts == 2
    assert outcome.replayed is True
    assert operations == [
        "guard",
        "guard",
        ("reserve", "turn-replay"),
        "replay_check",
        "guard",
    ]
    assert [event.status.value for event in events] == ["completed"]


def test_durable_repair_requires_both_prepared_callbacks_before_side_effects() -> None:
    operations: list[str] = []

    class RepairDurableCall:
        semantic_call_id = "generic_logical_repair"
        logical_request = SimpleNamespace(
            output_repair_protocol=(
                RuntimeModelOutputRepairProtocol.FOUR_MESSAGE_WHOLE_RESPONSE_REGENERATION
            ),
            typed_result_contract="generic-result-v1",
        )

        def require_current_state(self) -> None:
            operations.append("guard")

        def reserve(self, *, turn_id: str) -> object:
            operations.append(f"reserve:{turn_id}")
            raise AssertionError("invalid composition must not reserve")

        def terminal_state_error(self, message: str) -> DurableModelCallTerminalState:
            return DurableModelCallTerminalState(message)

    def prepare() -> object:
        operations.append("prepare")
        raise AssertionError("invalid composition must not prepare")

    with pytest.raises(ValueError, match="requires prepare_repair_request"):
        request_model_with_retry(
            turn_id="turn-repair",
            session_id="session-repair",
            purpose="generic-ledger-repair",
            stage=RuntimeStage.VERIFICATION,
            prepare_request=prepare,  # type: ignore[arg-type]
            validate=lambda result: result.reply,
            emit=lambda _event: None,
            durable_call=RepairDurableCall(),  # type: ignore[arg-type]
            logical_model_call_id="generic_logical_repair",
        )

    assert operations == []


def test_durable_repair_target_mismatch_fails_before_side_effects() -> None:
    operations: list[str] = []

    class RepairDurableCall:
        semantic_call_id = "generic_logical_repair_target"
        logical_request = SimpleNamespace(
            output_repair_protocol=(
                RuntimeModelOutputRepairProtocol.FOUR_MESSAGE_WHOLE_RESPONSE_REGENERATION
            ),
            typed_result_contract="expected-result",
        )

        def terminal_state_error(self, message: str) -> DurableModelCallTerminalState:
            return DurableModelCallTerminalState(message)

    def prepare() -> object:
        operations.append("prepare")
        raise AssertionError("mismatched composition must not prepare")

    with pytest.raises(
        DurableModelCallTerminalState,
        match="differs from the durable logical request",
    ):
        request_model_with_retry(
            turn_id="turn-repair-target",
            session_id="session-repair-target",
            purpose="generic-ledger-repair-target",
            stage=RuntimeStage.VERIFICATION,
            prepare_request=prepare,  # type: ignore[arg-type]
            prepare_repair_request=lambda _feedback, _body: pytest.fail(
                "repair preparation ran"
            ),
            repair_target_contract="different-result",
            validate=lambda result: result.reply,
            emit=lambda _event: None,
            durable_call=RepairDurableCall(),  # type: ignore[arg-type]
            logical_model_call_id="generic_logical_repair_target",
        )

    assert operations == []


def test_invalid_durable_typed_replay_fails_closed_without_provider_io() -> None:
    class InvalidReplayDurableCall:
        semantic_call_id = "generic_logical_invalid_replay"

        def require_current_state(self) -> None:
            return None

        def reserve(self, *, turn_id: str) -> object:
            return object()

        def replay_succeeded_result(self) -> DurableModelCallReplay:
            return DurableModelCallReplay(
                model_result=ModelResult(
                    reply="not-valid-for-current-contract",
                    provider="test",
                    model="test",
                    latency_ms=0,
                    model_call_id="generic_logical_invalid_replay:physical:1",
                ),
                physical_ordinal=1,
            )

        def begin_physical_attempt(self, **_values: object) -> object:
            raise AssertionError

        def settle_physical_attempt(self, **_values: object) -> None:
            raise AssertionError

        def typed_result_payload(self, **_values: object) -> object:
            raise AssertionError

        def success_fingerprint(self, _result: object) -> str:
            raise AssertionError

        def failure_fingerprint(self, **_values: object) -> str:
            raise AssertionError

        def terminal_state_error(self, message: str) -> DurableModelCallTerminalState:
            return DurableModelCallTerminalState(message)

    with pytest.raises(DurableModelCallTerminalState, match="replay validation"):
        request_model_with_retry(
            turn_id="turn-replay",
            session_id="session-1",
            purpose="generic-ledger-replay",
            stage=RuntimeStage.VERIFICATION,
            prepare_request=prepare_test_model_request(
                lambda _model_call_id: (_ for _ in ()).throw(AssertionError)
            ),
            validate=lambda _result: (_ for _ in ()).throw(
                ModelOutputValidationError("current parser rejects stored body")
            ),
            emit=lambda _event: None,
            durable_call=InvalidReplayDurableCall(),  # type: ignore[arg-type]
            logical_model_call_id="generic_logical_invalid_replay",
        )


def test_host_logical_call_id_must_match_durable_call_before_provider_io() -> None:
    calls = 0

    def request(model_call_id: str) -> ModelResult:
        nonlocal calls
        calls += 1
        return _result(model_call_id)

    with pytest.raises(ValueError, match="durable semantic call authority"):
        request_model_with_retry(
            turn_id="turn-1",
            session_id="session-1",
            purpose="test",
            stage=RuntimeStage.VERIFICATION,
            prepare_request=prepare_test_model_request(request),
            validate=lambda result: result.reply,
            emit=lambda _event: None,
            logical_model_call_id="logical_call_01",
            durable_call=SimpleNamespace(semantic_call_id="logical_call_other"),  # type: ignore[arg-type]
        )

    assert calls == 0


def test_response_identity_splice_is_rejected_before_typed_validation() -> None:
    validations = 0

    def validate(result: ModelResult) -> str:
        nonlocal validations
        validations += 1
        return result.reply

    with pytest.raises(ModelGatewayError) as captured:
        request_model_with_retry(
            turn_id="turn-1",
            session_id="session-1",
            purpose="test",
            stage=RuntimeStage.VERIFICATION,
            prepare_request=prepare_test_model_request(
                lambda _model_call_id: _result("another_logical_call")
            ),
            validate=validate,
            emit=lambda _event: None,
            logical_model_call_id="logical_call_01",
        )

    assert captured.value.code == "MODEL_BAD_RESPONSE"
    assert captured.value.retryable is False
    assert validations == 0


def test_authority_bound_response_requires_a_model_call_id() -> None:
    with pytest.raises(ModelGatewayError) as captured:
        request_model_with_retry(
            turn_id="turn-1",
            session_id="session-1",
            purpose="test",
            stage=RuntimeStage.VERIFICATION,
            prepare_request=prepare_test_model_request(
                lambda _model_call_id: ModelResult(
                    reply="ok",
                    provider="legacy-test",
                    model="test",
                    latency_ms=1,
                )
            ),
            validate=lambda result: result.reply,
            emit=lambda _event: None,
            logical_model_call_id="logical_call_01",
        )

    assert captured.value.code == "MODEL_BAD_RESPONSE"
    assert captured.value.retryable is False


def test_non_durable_response_requires_a_model_call_id() -> None:
    with pytest.raises(ModelGatewayError) as captured:
        request_model_with_retry(
            turn_id="turn-1",
            session_id="session-1",
            purpose="test",
            stage=RuntimeStage.CLASSIFY,
            prepare_request=prepare_test_model_request(
                lambda _model_call_id: ModelResult(
                    reply="ok",
                    provider="test",
                    model="test",
                    latency_ms=1,
                )
            ),
            validate=lambda result: result.reply,
            emit=lambda _event: None,
        )

    assert captured.value.code == "MODEL_BAD_RESPONSE"
    assert captured.value.retryable is False


def test_retry_wrapper_stops_on_non_retryable_provider_failure():
    calls = 0
    events = []

    def request(_model_call_id: str) -> ModelResult:
        nonlocal calls
        calls += 1
        raise ModelGatewayError("MODEL_CALL_FAILED", "bad api key", retryable=False)

    with pytest.raises(ModelGatewayError):
        request_model_with_retry(
            turn_id="turn-1",
            session_id="session-1",
            purpose="test",
            stage=RuntimeStage.L0_GENERATE,
            prepare_request=prepare_test_model_request(request),
            validate=lambda result: result.reply,
            emit=events.append,
        )

    assert calls == 1
    assert [event.status.value for event in events] == ["started", "failed"]


def test_retry_wrapper_stops_on_non_retryable_output_validation_failure():
    calls = 0

    def request(model_call_id: str) -> ModelResult:
        nonlocal calls
        calls += 1
        return _result(model_call_id)

    with pytest.raises(ModelGatewayError) as exc:
        request_model_with_retry(
            turn_id="turn-1",
            session_id="session-1",
            purpose="test",
            stage=RuntimeStage.L2_UNDERSTAND,
            prepare_request=prepare_test_model_request(request),
            validate=lambda _result: (_ for _ in ()).throw(
                ModelOutputValidationError("budget exhausted", retryable=False)
            ),
            emit=lambda _event: None,
        )

    assert calls == 1
    assert exc.value.code == "MODEL_BAD_RESPONSE"
    assert exc.value.retryable is False


@pytest.mark.parametrize(
    "finish_reason",
    ("max_tokens", "length", "model_length", "token_limit"),
)
def test_provider_declared_output_limit_is_never_blindly_retried(
    finish_reason: str,
) -> None:
    calls = 0

    def request(model_call_id: str) -> ModelResult:
        nonlocal calls
        calls += 1
        return ModelResult(
            reply='{"apparently":"complete"}',
            provider="test",
            model="test",
            latency_ms=1,
            finish_reason=finish_reason,
            model_call_id=model_call_id,
        )

    with pytest.raises(ModelGatewayError) as exc:
        request_model_with_retry(
            turn_id="turn-1",
            session_id="session-1",
            purpose="test",
            stage=RuntimeStage.L2_PLAN,
            prepare_request=prepare_test_model_request(request),
            validate=lambda result: result.reply,
            emit=lambda _event: None,
        )

    assert calls == 1
    assert exc.value.code == "MODEL_BAD_RESPONSE"
    assert exc.value.retryable is False


def test_transport_retries_back_off_but_typed_output_retries_do_not(monkeypatch):
    """429/5xx 表示提供方需要余量；我们自己的解析判定并非如此。"""
    slept: list[float] = []
    monkeypatch.setattr(model_requests, "_sleep", slept.append)

    def rate_limited(_model_call_id: str) -> ModelResult:
        raise ModelGatewayError(
            "MODEL_CALL_FAILED", "rate limited", retryable=True, details={"status_code": 429}
        )

    with pytest.raises(ModelGatewayError):
        request_model_with_retry(
            turn_id="turn-1",
            session_id="session-1",
            purpose="test",
            stage=RuntimeStage.CLASSIFY,
            prepare_request=prepare_test_model_request(rate_limited),
            validate=lambda result: result.reply,
            emit=lambda _event: None,
        )

    # 六次物理尝试之间等待五次，最后一次之后不再等待。
    assert len(slept) == 5
    assert all(0.0 <= delay <= BACKOFF_MAX_S for delay in slept)

    slept.clear()
    with pytest.raises(ModelGatewayError):
        request_model_with_retry(
            turn_id="turn-2",
            session_id="session-1",
            purpose="test",
            stage=RuntimeStage.CLASSIFY,
            prepare_request=prepare_test_model_request(_result),
            validate=lambda _result: (_ for _ in ()).throw(ModelOutputValidationError("bad")),
            emit=lambda _event: None,
        )
    assert slept == []


def test_exhausted_transport_retries_record_one_terminal_failure(monkeypatch):
    calls = 0
    recorded: list[dict[str, object]] = []
    monkeypatch.setattr(model_requests, "_sleep", lambda _seconds: None)
    monkeypatch.setattr(
        model_requests,
        "_record_terminal_model_failure",
        lambda **values: recorded.append(values),
    )

    def unavailable(_model_call_id: str) -> ModelResult:
        nonlocal calls
        calls += 1
        raise ModelGatewayError(
            "MODEL_CALL_FAILED",
            "provider unavailable",
            retryable=True,
        )

    with pytest.raises(ModelGatewayError):
        request_model_with_retry(
            turn_id="turn-terminal-failure",
            session_id="session-terminal-failure",
            purpose="l1_attempt",
            stage=RuntimeStage.CLASSIFY,
            prepare_request=prepare_test_model_request(unavailable),
            validate=lambda result: result.reply,
            emit=lambda _event: None,
            max_attempts=3,
        )

    assert calls == 3
    assert len(recorded) == 1
    assert recorded[0]["purpose"] == "l1_attempt"
    assert recorded[0]["turn_id"] == "turn-terminal-failure"
    assert recorded[0]["session_id"] == "session-terminal-failure"
    assert recorded[0]["reason_code"] == "MODEL_CALL_FAILED"
    assert recorded[0]["attempts"] == 3
    assert isinstance(recorded[0]["duration_ms"], int)


def test_prepared_repair_receives_structured_feedback_and_rejected_response() -> None:
    rejected_body = '{"secret_source_excerpt":"must-not-be-echoed"}'
    repairs: list[tuple[RuntimeModelOutputRepairFeedback, str]] = []
    calls = 0

    class Prepared:
        def __init__(self, reply: str) -> None:
            self.reply = reply

        def dispatch(self, *, model_call_id: str) -> ModelResult:
            nonlocal calls
            calls += 1
            return ModelResult(
                reply=self.reply,
                provider="test",
                model="test",
                latency_ms=1,
                model_call_id=model_call_id,
            )

    def prepare_repair(
        feedback: RuntimeModelOutputRepairFeedback,
        rejected_response_text: str,
    ) -> Prepared:
        repairs.append((feedback, rejected_response_text))
        return Prepared("ok")

    def validate(result: ModelResult) -> str:
        if result.reply != "ok":
            raise ModelOutputValidationError(
                "unsafe detailed parser error",
                repair_code="test_contract_field_missing",
                safe_repair_reason="The required field is missing.",
            )
        return result.reply

    outcome = request_model_with_retry(
        turn_id="turn-repair",
        session_id="session-repair",
        purpose="test-repair",
        stage=RuntimeStage.L2_PLAN,
        prepare_request=lambda: Prepared(rejected_body),
        prepare_repair_request=prepare_repair,
        repair_target_contract="test-result-v1",
        validate=validate,
        emit=lambda _event: None,
    )

    assert calls == 2
    assert outcome.attempts == 2
    assert len(repairs) == 1
    feedback, rejected_response_text = repairs[0]
    assert rejected_response_text == rejected_body
    assert feedback.current_issues[0].code == "test_contract_field_missing"
    assert (
        feedback.current_issues[0].safe_explanation
        == "The required field is missing."
    )
    assert feedback.rejected_response_sha256 == hashlib.sha256(
        rejected_body.encode("utf-8")
    ).hexdigest()
    assert "must-not-be-echoed" not in feedback.model_dump_json()


def test_prepared_repair_keeps_transport_context_and_replaces_only_on_rejection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """传输重试复用 repair 上下文；新的错误正文会取代旧上下文。"""

    monkeypatch.setattr(model_requests, "_sleep", lambda _seconds: None)
    first_rejected = '{"proposal":"first-invalid"}'
    second_rejected = '{"proposal":"second-invalid"}'
    provider_steps: list[str | ModelGatewayError] = [
        first_rejected,
        ModelGatewayError(
            "MODEL_CALL_FAILED",
            "temporary transport failure",
            retryable=True,
        ),
        second_rejected,
        '{"proposal":"accepted"}',
    ]
    prepared_repairs: list[
        tuple[RuntimeModelOutputRepairFeedback, str]
    ] = []
    initial_preparations = 0

    class Prepared:
        def dispatch(self, *, model_call_id: str) -> ModelResult:
            step = provider_steps.pop(0)
            if isinstance(step, ModelGatewayError):
                raise step
            return ModelResult(
                reply=step,
                provider="test",
                model="test",
                latency_ms=1,
                model_call_id=model_call_id,
            )

    def prepare_initial() -> Prepared:
        nonlocal initial_preparations
        initial_preparations += 1
        return Prepared()

    def prepare_four_message_repair(
        feedback: RuntimeModelOutputRepairFeedback,
        rejected_response_text: str,
    ) -> Prepared:
        prepared_repairs.append((feedback, rejected_response_text))
        return Prepared()

    def validate(result: ModelResult) -> dict[str, str]:
        if result.reply == first_rejected:
            raise ModelOutputValidationError(
                "first output rejected",
                repair_code="first_contract_error",
                safe_repair_reason="第一份输出没有满足目标合同。",
            )
        if result.reply == second_rejected:
            raise ModelOutputValidationError(
                "second output rejected",
                repair_code="second_contract_error",
                safe_repair_reason="第二份输出没有满足目标合同。",
            )
        return {"proposal": "accepted"}

    outcome = request_model_with_retry(
        turn_id="turn-repair-context",
        session_id="session-repair-context",
        purpose="test-repair-context",
        stage=RuntimeStage.L2_PLAN,
        prepare_request=prepare_initial,
        prepare_repair_request=prepare_four_message_repair,
        repair_target_contract="test-output",
        validate=validate,
        emit=lambda _event: None,
    )

    assert outcome.attempts == 4
    assert outcome.value == {"proposal": "accepted"}
    assert initial_preparations == 1
    assert len(prepared_repairs) == 3
    first_feedback, first_body = prepared_repairs[0]
    transport_feedback, transport_body = prepared_repairs[1]
    replacement_feedback, replacement_body = prepared_repairs[2]
    assert first_body == transport_body == first_rejected
    assert first_feedback is transport_feedback
    assert first_feedback.rejected_physical_ordinal == 1
    assert first_feedback.rejected_response_sha256 == hashlib.sha256(
        first_rejected.encode("utf-8")
    ).hexdigest()
    assert first_feedback.current_issues[0].code == "first_contract_error"
    assert replacement_body == second_rejected
    assert replacement_feedback != first_feedback
    assert replacement_feedback.rejected_physical_ordinal == 3
    assert replacement_feedback.rejected_response_sha256 == hashlib.sha256(
        second_rejected.encode("utf-8")
    ).hexdigest()
    assert replacement_feedback.current_issues[0].code == "second_contract_error"
    assert provider_steps == []


def test_prepared_repair_uses_the_canonical_callback() -> None:
    repair_inputs: list[tuple[RuntimeModelOutputRepairFeedback, str]] = []
    replies = iter(("invalid", "accepted"))

    class Prepared:
        def dispatch(self, *, model_call_id: str) -> ModelResult:
            return ModelResult(
                reply=next(replies),
                provider="test",
                model="test",
                latency_ms=1,
                model_call_id=model_call_id,
            )

    def prepare_repair(
        feedback: RuntimeModelOutputRepairFeedback,
        rejected_response_text: str,
    ) -> Prepared:
        repair_inputs.append((feedback, rejected_response_text))
        return Prepared()

    outcome = request_model_with_retry(
        turn_id="turn-canonical-repair",
        session_id="session-canonical-repair",
        purpose="test-canonical-repair",
        stage=RuntimeStage.L2_PLAN,
        prepare_request=Prepared,
        prepare_repair_request=prepare_repair,
        repair_target_contract="test-output",
        validate=lambda result: (
            result.reply
            if result.reply == "accepted"
            else (_ for _ in ()).throw(
                ModelOutputValidationError(
                    "new output rejected",
                    repair_code="new_rejection",
                    safe_repair_reason="新输出没有满足目标合同。",
                )
            )
        ),
        emit=lambda _event: None,
    )

    assert outcome.value == "accepted"
    assert len(repair_inputs) == 1
    assert repair_inputs[0][1] == "invalid"
    assert repair_inputs[0][0].current_issues[0].code == "new_rejection"


def test_durable_repair_without_an_explicit_protocol_fails_closed() -> None:
    provider_calls = 0

    class DurableCall:
        semantic_call_id = "missing-repair-protocol"

        def terminal_state_error(self, message: str) -> DurableModelCallTerminalState:
            return DurableModelCallTerminalState(message)

    def prepare() -> object:
        nonlocal provider_calls
        provider_calls += 1
        return object()

    with pytest.raises(
        DurableModelCallTerminalState,
        match="requires an explicitly frozen protocol",
    ):
        request_model_with_retry(
            turn_id="turn-missing-repair-protocol",
            session_id="session-missing-repair-protocol",
            purpose="missing-repair-protocol",
            stage=RuntimeStage.L2_PLAN,
            prepare_request=prepare,  # type: ignore[arg-type]
            prepare_repair_request=lambda _feedback, _body: pytest.fail(
                "repair ran"
            ),
            validate=lambda result: result.reply,
            emit=lambda _event: None,
            durable_call=DurableCall(),  # type: ignore[arg-type]
            logical_model_call_id="missing-repair-protocol",
        )

    assert provider_calls == 0


def test_rejected_body_with_isolated_surrogate_terminalizes_attempt() -> None:
    settlements: list[dict[str, object]] = []
    repair_calls = 0
    provider_calls = 0

    class DurableCall:
        semantic_call_id = "repair-surrogate-logical"
        logical_request = SimpleNamespace(
            output_repair_protocol=(
                RuntimeModelOutputRepairProtocol.FOUR_MESSAGE_WHOLE_RESPONSE_REGENERATION
            ),
            typed_result_contract="test-output",
        )

        def require_current_state(self) -> None:
            return None

        def reserve(self, *, turn_id: str) -> object:
            assert turn_id == "turn-repair-surrogate"
            return object()

        def replay_succeeded_result(self) -> None:
            return None

        def recover_output_repair_feedback(self) -> None:
            return None

        def begin_physical_attempt(self, **values: object) -> object:
            assert values["output_repair_enabled"] is True
            assert values["output_repair_feedback"] is None
            return SimpleNamespace(
                physical_attempt_id="repair-surrogate-physical-1",
                physical_ordinal=1,
                provider="test",
                model="test",
                model_call_id="repair-surrogate-logical:physical:1",
            )

        def settle_physical_attempt(self, **values: object) -> None:
            settlements.append(dict(values))

        def failure_fingerprint(self, **_values: object) -> str:
            return "repair-surrogate-failure"

        def terminal_state_error(self, message: str) -> DurableModelCallTerminalState:
            return DurableModelCallTerminalState(message)

    def provider(model_call_id: str) -> ModelResult:
        nonlocal provider_calls
        provider_calls += 1
        return ModelResult(
            reply="\ud800",
            provider="test",
            model="test",
            latency_ms=1,
            model_call_id=model_call_id,
        )

    class Prepared:
        def dispatch(self, *, model_call_id: str) -> ModelResult:
            return provider(model_call_id)

    def repair(
        _feedback: RuntimeModelOutputRepairFeedback,
        _rejected_response_text: str,
    ) -> Prepared:
        nonlocal repair_calls
        repair_calls += 1
        raise AssertionError("non-UTF-8 rejected output entered repair")

    with pytest.raises(ModelGatewayError) as captured:
        request_model_with_retry(
            turn_id="turn-repair-surrogate",
            session_id="session-repair-surrogate",
            purpose="test-repair-surrogate",
            stage=RuntimeStage.L2_PLAN,
            prepare_request=Prepared,
            prepare_repair_request=repair,
            repair_target_contract="test-output",
            validate=lambda _result: (_ for _ in ()).throw(
                ModelOutputValidationError(
                    "invalid structured output",
                    repair_code="invalid_surrogate_output",
                    safe_repair_reason="输出不是可持久化的 UTF-8 文本。",
                )
            ),
            emit=lambda _event: None,
            durable_call=DurableCall(),  # type: ignore[arg-type]
            logical_model_call_id="repair-surrogate-logical",
        )

    assert captured.value.code == "MODEL_BAD_RESPONSE"
    assert captured.value.retryable is False
    assert provider_calls == 1
    assert repair_calls == 0
    assert len(settlements) == 1
    assert settlements[0]["outcome"] == "terminal_failure"
    assert settlements[0]["error_code"] == "MODEL_BAD_RESPONSE"
    assert "next_output_repair_feedback" not in settlements[0]
    assert "rejected_response_text" not in settlements[0]


def test_rejected_body_over_durable_limit_does_not_enter_repair() -> None:
    provider_calls = 0
    repair_calls = 0
    rejected = "x" * (RUNTIME_MODEL_REJECTED_OUTPUT_MAX_UTF8_BYTES + 1)

    def provider(model_call_id: str) -> ModelResult:
        nonlocal provider_calls
        provider_calls += 1
        return ModelResult(
            reply=rejected,
            provider="test",
            model="test",
            latency_ms=1,
            model_call_id=model_call_id,
        )

    class Prepared:
        def dispatch(self, *, model_call_id: str) -> ModelResult:
            return provider(model_call_id)

    def repair(
        _feedback: RuntimeModelOutputRepairFeedback,
        _rejected_response_text: str,
    ) -> Prepared:
        nonlocal repair_calls
        repair_calls += 1
        raise AssertionError("oversized rejected output entered repair")

    with pytest.raises(ModelGatewayError) as captured:
        request_model_with_retry(
            turn_id="turn-repair-oversized",
            session_id="session-repair-oversized",
            purpose="test-repair-oversized",
            stage=RuntimeStage.L2_PLAN,
            prepare_request=Prepared,
            prepare_repair_request=repair,
            repair_target_contract="test-output",
            validate=lambda _result: (_ for _ in ()).throw(
                ModelOutputValidationError(
                    "invalid structured output",
                    repair_code="oversized_rejected_output",
                    safe_repair_reason="输出超过可恢复修复正文的大小上限。",
                )
            ),
            emit=lambda _event: None,
        )

    assert captured.value.code == "MODEL_BAD_RESPONSE"
    assert captured.value.retryable is False
    assert provider_calls == 1
    assert repair_calls == 0


@pytest.mark.parametrize("wide_paths", [False, True])
def test_shared_builder_truncates_issue_count_or_envelope_bytes(
    wide_paths: bool,
) -> None:
    captured: list[RuntimeModelOutputRepairFeedback] = []
    replies = iter(("invalid", "accepted"))
    issue_count = 40 if wide_paths else 70
    issues = tuple(
        sorted(
            (
                RuntimeModelOutputRepairIssue(
                    category="schema",
                    code=f"schema.problem_{index:03d}",
                    paths=(
                        "/"
                        + ("p" * 1_500 if wide_paths else "items/")
                        + f"{index:03d}",
                    ),
                    safe_explanation="该位置不符合目标结构化合同。",
                )
                for index in range(issue_count)
            ),
            key=runtime_model_output_repair_issue_sort_key,
        )
    )

    def request(model_call_id: str) -> ModelResult:
        return ModelResult(
            reply=next(replies),
            provider="test",
            model="test",
            latency_ms=1,
            model_call_id=model_call_id,
        )

    class Prepared:
        def dispatch(self, *, model_call_id: str) -> ModelResult:
            return request(model_call_id)

    def repair(
        feedback: RuntimeModelOutputRepairFeedback,
        _rejected_response_text: str,
    ) -> Prepared:
        captured.append(feedback)
        return Prepared()

    outcome = request_model_with_retry(
        turn_id=f"turn-repair-issue-budget-{wide_paths}",
        session_id="session-repair-issue-budget",
        purpose="test-repair-issue-budget",
        stage=RuntimeStage.L2_PLAN,
        prepare_request=Prepared,
        prepare_repair_request=repair,
        repair_target_contract="test-output",
        validate=lambda result: (
            result.reply
            if result.reply == "accepted"
            else (_ for _ in ()).throw(
                ModelOutputValidationError(
                    "many current issues",
                    repair_code="many_current_issues",
                    safe_repair_reason="输出同时违反多项结构化合同要求。",
                    repair_issues=issues,
                    repair_issue_coverage=(
                        RuntimeModelOutputRepairIssueCoverage.COMPLETE
                    ),
                )
            )
        ),
        emit=lambda _event: None,
    )

    assert outcome.value == "accepted"
    assert len(captured) == 1
    feedback = captured[0]
    assert feedback.issue_coverage.value == "truncated"
    assert feedback.omitted_issue_count == issue_count - len(
        feedback.current_issues
    )
    assert 1 <= len(feedback.current_issues) <= 64
    if wide_paths:
        assert len(feedback.current_issues) < issue_count
    else:
        assert len(feedback.current_issues) == 64


def test_transport_retry_does_not_invent_output_repair_feedback(monkeypatch) -> None:
    monkeypatch.setattr(model_requests, "_sleep", lambda _seconds: None)
    calls = 0
    repairs = 0

    def request(model_call_id: str) -> ModelResult:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ModelGatewayError(
                "MODEL_CALL_FAILED",
                "rate limited",
                retryable=True,
                details={"status_code": 429},
            )
        return _result(model_call_id)

    class Prepared:
        def dispatch(self, *, model_call_id: str) -> ModelResult:
            return request(model_call_id)

    def prepare_repair(
        _feedback: RuntimeModelOutputRepairFeedback,
        _rejected_response_text: str,
    ) -> Prepared:
        nonlocal repairs
        repairs += 1
        raise AssertionError("transport retry must retain the original request")

    outcome = request_model_with_retry(
        turn_id="turn-transport-repair",
        session_id="session-repair",
        purpose="test-repair",
        stage=RuntimeStage.L2_PLAN,
        prepare_request=Prepared,
        prepare_repair_request=prepare_repair,
        repair_target_contract="test-result-v1",
        validate=lambda result: result.reply,
        emit=lambda _event: None,
    )

    assert outcome.attempts == 2
    assert calls == 2
    assert repairs == 0


def test_durable_repair_fails_before_provider_when_authority_lacks_checkpoint_seam() -> None:
    provider_calls = 0
    reserve_calls = 0

    class DurableCallWithoutCheckpoint:
        semantic_call_id = "no_repair_checkpoint"
        logical_request = SimpleNamespace(
            output_repair_protocol=(
                RuntimeModelOutputRepairProtocol.FOUR_MESSAGE_WHOLE_RESPONSE_REGENERATION
            ),
            typed_result_contract="test-result-v1",
        )

        def require_current_state(self) -> None:
            return None

        def reserve(self, *, turn_id: str) -> object:
            nonlocal reserve_calls
            reserve_calls += 1
            return object()

        def replay_succeeded_result(self) -> None:
            return None

        def terminal_state_error(self, message: str) -> DurableModelCallTerminalState:
            return DurableModelCallTerminalState(message)

    def provider(_model_call_id: str) -> ModelResult:
        nonlocal provider_calls
        provider_calls += 1
        raise AssertionError("unsupported durable repair reached Provider")

    class Prepared:
        def dispatch(self, *, model_call_id: str) -> ModelResult:
            return provider(model_call_id)

    with pytest.raises(DurableModelCallTerminalState, match="persisted output repair"):
        request_model_with_retry(
            turn_id="turn-no-checkpoint",
            session_id="session-repair",
            purpose="test-repair",
            stage=RuntimeStage.L2_PLAN,
            prepare_request=Prepared,
            prepare_repair_request=lambda _feedback, _body: Prepared(),
            repair_target_contract="test-result-v1",
            validate=lambda result: result.reply,
            emit=lambda _event: None,
            durable_call=DurableCallWithoutCheckpoint(),  # type: ignore[arg-type]
            logical_model_call_id="no_repair_checkpoint",
        )

    assert provider_calls == 0
    assert reserve_calls == 0


def test_output_repair_keeps_the_existing_six_attempt_limit() -> None:
    initial_calls = 0
    repair_calls = 0

    class Prepared:
        def __init__(self, *, repair: bool) -> None:
            self.repair = repair

        def dispatch(self, *, model_call_id: str) -> ModelResult:
            nonlocal initial_calls, repair_calls
            if self.repair:
                repair_calls += 1
            else:
                initial_calls += 1
            return _result(model_call_id)

    with pytest.raises(ModelGatewayError) as captured:
        request_model_with_retry(
            turn_id="turn-repair-limit",
            session_id="session-repair",
            purpose="test-repair",
            stage=RuntimeStage.L2_PLAN,
            prepare_request=lambda: Prepared(repair=False),
            prepare_repair_request=lambda _feedback, _body: Prepared(repair=True),
            repair_target_contract="test-result-v1",
            validate=lambda _result: (_ for _ in ()).throw(
                ModelOutputValidationError(
                    "still invalid",
                    repair_code="test_still_invalid",
                    safe_repair_reason="The required contract is still invalid.",
                )
            ),
            emit=lambda _event: None,
        )

    assert captured.value.code == "MODEL_BAD_RESPONSE"
    assert initial_calls == 1
    assert repair_calls == 5


def test_backoff_grows_exponentially_and_is_capped():
    ceilings = [max(backoff_delay_s(attempt) for _ in range(300)) for attempt in range(1, 7)]
    assert ceilings[0] < ceilings[1] < ceilings[2] < ceilings[3]
    assert all(delay <= BACKOFF_MAX_S for delay in ceilings)


def test_non_retryable_failure_preserves_the_original_cause():
    def request(_model_call_id: str) -> ModelResult:
        raise ModelGatewayError("MODEL_CALL_FAILED", "bad api key", retryable=False)

    with pytest.raises(ModelGatewayError) as exc:
        request_model_with_retry(
            turn_id="turn-1",
            session_id="session-1",
            purpose="test",
            stage=RuntimeStage.CLASSIFY,
            prepare_request=prepare_test_model_request(request),
            validate=lambda result: result.reply,
            emit=lambda _event: None,
        )
    assert exc.value.__cause__ is not None
