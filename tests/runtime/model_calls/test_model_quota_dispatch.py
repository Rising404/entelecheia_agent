from __future__ import annotations

from types import SimpleNamespace

import pytest

from personagraph.model_io.api_quota_controller import (
    DEFAULT_API_QUOTA_WAIT_TIMEOUT_SECONDS,
)
from personagraph.model_io.gateway import ModelGatewayError, ModelResult
from personagraph.runtime.model_calls.contracts import (
    DurableModelCallStateGuardRejected,
    DurableModelCallTerminalState,
)
from personagraph.runtime.model_calls.quota import (
    acquire_quota_dispatch_handle,
)
from personagraph.runtime.model_calls.requests import request_model_with_retry
from personagraph.runtime.turn_events import RuntimeStage
from personagraph.runtime.model_calls import requests as model_requests


_PERMIT = object()


def _result(model_call_id: str) -> ModelResult:
    return ModelResult(
        reply="ok",
        provider="test",
        model="test",
        latency_ms=1,
        model_call_id=model_call_id,
    )


class _PlainPrepared:
    def dispatch(self, *, model_call_id: str) -> ModelResult:
        return _result(model_call_id)


class _QuotaPrepared:
    def __init__(
        self,
        operations: list[object],
        *,
        permit: object | None = _PERMIT,
        dispatch_error: BaseException | None = None,
        acquired_state: dict[str, bool] | None = None,
    ) -> None:
        self.operations = operations
        self.permit = permit
        self.dispatch_error = dispatch_error
        self.acquired_state = acquired_state

    def dispatch(self, *, model_call_id: str) -> ModelResult:
        raise AssertionError("quota protocol requests must use their handoff")

    def acquire_api_quota(
        self,
        *,
        model_call_id: str,
        wait_timeout_seconds: float,
    ) -> object | None:
        self.operations.append(
            ("acquire", model_call_id, wait_timeout_seconds)
        )
        if self.acquired_state is not None:
            self.acquired_state["acquired"] = True
        return self.permit

    def abandon_api_quota(
        self,
        permit: object | None,
        *,
        disposition: str = "cancel",
    ) -> None:
        self.operations.append(("abandon", permit, disposition))

    def dispatch_with_api_quota(
        self,
        *,
        model_call_id: str,
        permit: object | None,
    ) -> ModelResult:
        self.operations.append(("dispatch", model_call_id, permit))
        if self.dispatch_error is not None:
            raise self.dispatch_error
        return _result(model_call_id)


class _StableDeadline:
    def expired(self) -> bool:
        return False

    def remaining_s(self) -> float:
        return 5.0


class _PreDispatchDurable:
    semantic_call_id = "quota-logical"

    def __init__(
        self,
        operations: list[object],
        *,
        acquired_state: dict[str, bool] | None = None,
        reject_guard_after_acquire: bool = False,
        begin_error: BaseException | None = None,
    ) -> None:
        self.operations = operations
        self.acquired_state = acquired_state
        self.reject_guard_after_acquire = reject_guard_after_acquire
        self.begin_error = begin_error

    def require_current_state(self) -> None:
        self.operations.append("guard")
        if (
            self.reject_guard_after_acquire
            and self.acquired_state is not None
            and self.acquired_state["acquired"]
        ):
            raise DurableModelCallStateGuardRejected("authority changed")

    def reserve(self, *, turn_id: str) -> object:
        self.operations.append(("reserve", turn_id))
        return object()

    def replay_succeeded_result(self) -> None:
        self.operations.append("replay")
        return None

    def begin_physical_attempt(
        self,
        *,
        turn_id: str,
        max_physical_attempts: int,
    ) -> object:
        self.operations.append(("begin", turn_id, max_physical_attempts))
        if self.begin_error is not None:
            raise self.begin_error
        return SimpleNamespace(
            physical_ordinal=1,
            model_call_id="quota-logical:physical:1",
        )

    def failure_fingerprint(
        self,
        *,
        error: BaseException,
        provider_result: object | None,
    ) -> str:
        assert provider_result is None
        self.operations.append(("fingerprint", type(error).__name__))
        return "quota-pre-dispatch-failure"

    def settle_physical_attempt(self, **values: object) -> None:
        self.operations.append(
            (
                "settle",
                values["outcome"],
                values.get("error_code"),
            )
        )

    def terminal_state_error(self, message: str) -> DurableModelCallTerminalState:
        return DurableModelCallTerminalState(message)


def _run_request(
    prepared: _QuotaPrepared,
    *,
    durable_call: object | None = None,
    deadline: object | None = None,
    emit=lambda _event: None,
) -> object:
    return request_model_with_retry(
        turn_id="turn-quota",
        session_id="session-quota",
        purpose="quota-boundary",
        stage=RuntimeStage.VERIFICATION,
        prepare_request=lambda: prepared,
        validate=lambda result: result.reply,
        emit=emit,
        deadline=deadline,  # type: ignore[arg-type]
        durable_call=durable_call,  # type: ignore[arg-type]
        logical_model_call_id=(
            durable_call.semantic_call_id  # type: ignore[union-attr]
            if durable_call is not None
            else "quota-logical"
        ),
    )


def test_factory_ignores_plain_prepared_requests() -> None:
    handle = acquire_quota_dispatch_handle(
        prepared_request=_PlainPrepared(),
        model_call_id="plain-call",
        wait_timeout_seconds=3.0,
    )

    assert handle is None


def test_handle_preserves_none_permit_and_default_wait_timeout() -> None:
    operations: list[object] = []
    prepared = _QuotaPrepared(operations, permit=None)

    handle = acquire_quota_dispatch_handle(
        prepared_request=prepared,
        model_call_id="unlimited-call",
        wait_timeout_seconds=None,
    )

    assert handle is not None
    result = handle.dispatch(model_call_id="unlimited-call")
    assert result.model_call_id == "unlimited-call"
    assert operations == [
        (
            "acquire",
            "unlimited-call",
            DEFAULT_API_QUOTA_WAIT_TIMEOUT_SECONDS,
        ),
        ("dispatch", "unlimited-call", None),
    ]


def test_abandon_retries_when_the_first_close_fails() -> None:
    operations: list[object] = []

    class FailingAbandon(_QuotaPrepared):
        def abandon_api_quota(
            self,
            permit: object | None,
            *,
            disposition: str = "cancel",
        ) -> None:
            super().abandon_api_quota(
                permit,
                disposition=disposition,
            )
            if len([item for item in operations if item[0] == "abandon"]) == 1:
                raise RuntimeError("close failed")

    handle = acquire_quota_dispatch_handle(
        prepared_request=FailingAbandon(operations),
        model_call_id="close-retry",
        wait_timeout_seconds=2.0,
    )
    assert handle is not None

    with pytest.raises(RuntimeError, match="close failed"):
        handle.abandon_if_not_handed_off()
    handle.abandon_if_not_handed_off()
    handle.abandon_if_not_handed_off()

    abandons = [item for item in operations if item[0] == "abandon"]
    assert abandons == [
        ("abandon", _PERMIT, "cancel"),
        ("abandon", _PERMIT, "cancel"),
    ]


def test_dispatch_handoff_suppresses_runtime_abandon_even_when_adapter_raises() -> None:
    operations: list[object] = []
    prepared = _QuotaPrepared(
        operations,
        dispatch_error=ModelGatewayError(
            "MODEL_CALL_FAILED",
            "provider failed after handoff",
            retryable=False,
        ),
    )
    handle = acquire_quota_dispatch_handle(
        prepared_request=prepared,
        model_call_id="handoff-error",
        wait_timeout_seconds=2.0,
    )
    assert handle is not None

    with pytest.raises(ModelGatewayError):
        handle.dispatch(model_call_id="handoff-error")
    handle.abandon_if_not_handed_off()

    assert operations == [
        ("acquire", "handoff-error", 2.0),
        ("dispatch", "handoff-error", _PERMIT),
    ]


def test_deadline_expiry_after_quota_wait_abandons_before_physical_attempt() -> None:
    operations: list[object] = []
    state = {"acquired": False}
    prepared = _QuotaPrepared(operations, acquired_state=state)
    durable = _PreDispatchDurable(operations)

    class Deadline(_StableDeadline):
        def remaining_s(self) -> float:
            return 0.0 if state["acquired"] else 5.0

    with pytest.raises(ModelGatewayError) as captured:
        _run_request(prepared, durable_call=durable, deadline=Deadline())

    assert captured.value.code == "TURN_DEADLINE_EXCEEDED"
    assert ("abandon", _PERMIT, "cancel") in operations
    assert not any(item[0] == "begin" for item in operations if isinstance(item, tuple))
    assert not any(item[0] == "dispatch" for item in operations if isinstance(item, tuple))


def test_guard_rejection_after_quota_wait_abandons_before_physical_attempt() -> None:
    operations: list[object] = []
    state = {"acquired": False}
    prepared = _QuotaPrepared(operations, acquired_state=state)
    durable = _PreDispatchDurable(
        operations,
        acquired_state=state,
        reject_guard_after_acquire=True,
    )

    with pytest.raises(DurableModelCallStateGuardRejected):
        _run_request(prepared, durable_call=durable)

    assert ("abandon", _PERMIT, "cancel") in operations
    assert not any(item[0] == "begin" for item in operations if isinstance(item, tuple))
    assert not any(item[0] == "dispatch" for item in operations if isinstance(item, tuple))


def test_physical_begin_failure_abandons_before_started_or_dispatch() -> None:
    operations: list[object] = []
    prepared = _QuotaPrepared(operations)
    durable = _PreDispatchDurable(
        operations,
        begin_error=RuntimeError("begin failed"),
    )
    events: list[object] = []

    with pytest.raises(RuntimeError, match="begin failed"):
        _run_request(prepared, durable_call=durable, emit=events.append)

    assert ("abandon", _PERMIT, "cancel") in operations
    assert not any(item[0] == "dispatch" for item in operations if isinstance(item, tuple))
    assert events == []


def test_started_event_failure_abandons_before_provider_dispatch() -> None:
    operations: list[object] = []
    prepared = _QuotaPrepared(operations)
    durable = _PreDispatchDurable(operations)

    def fail_started(_event: object) -> None:
        raise RuntimeError("event sink failed")

    with pytest.raises(RuntimeError, match="event sink failed"):
        _run_request(
            prepared,
            durable_call=durable,
            emit=fail_started,
        )

    settle_index = operations.index(
        ("settle", "retryable_failure", "RUNTIME_EVENT_EMIT_FAILED")
    )
    abandon_index = operations.index(("abandon", _PERMIT, "cancel"))
    assert settle_index < abandon_index
    assert not any(item[0] == "dispatch" for item in operations if isinstance(item, tuple))


@pytest.mark.parametrize(
    "setup_error",
    [
        ModelGatewayError(
            "MODEL_CONFIGURATION_FAILURE",
            "timeout scope failed",
            retryable=False,
        ),
        RuntimeError("timeout scope failed"),
    ],
)
def test_provider_scope_failure_abandons_before_dispatch(
    monkeypatch,
    setup_error: BaseException,
) -> None:
    operations: list[object] = []
    prepared = _QuotaPrepared(operations)

    def fail_scope(_remaining_s: float) -> object:
        raise setup_error

    monkeypatch.setattr(model_requests, "model_http_timeout_ceiling", fail_scope)

    with pytest.raises(type(setup_error), match="timeout scope failed"):
        _run_request(prepared, deadline=_StableDeadline())

    assert ("abandon", _PERMIT, "cancel") in operations
    assert not any(item[0] == "dispatch" for item in operations if isinstance(item, tuple))


def test_request_loop_does_not_abandon_after_dispatch_handoff_error() -> None:
    operations: list[object] = []
    prepared = _QuotaPrepared(
        operations,
        dispatch_error=ModelGatewayError(
            "MODEL_CALL_FAILED",
            "provider failed",
            retryable=False,
        ),
    )

    with pytest.raises(ModelGatewayError, match="provider failed"):
        _run_request(prepared)

    assert ("dispatch", "quota-logical", _PERMIT) in operations
    assert not any(item[0] == "abandon" for item in operations if isinstance(item, tuple))
