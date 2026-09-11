"""持久模型调用恢复投影的边界覆盖。"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import personagraph.runtime.model_calls as model_calls
from personagraph.runtime.model_calls import recovery
from personagraph.runtime.model_calls.contracts import RuntimeModelPhysicalOutcome


def _snapshot(**overrides: object) -> recovery.RuntimeModelCallRecoverySnapshot:
    values: dict[str, object] = {
        "logical_call_id": "logical-1",
        "provider": "mock",
        "model": "mock-structured",
        "endpoint_fingerprint": "a" * 64,
        "request_sha256": "b" * 64,
        "disposition": "ready",
        "physical_attempt_id": None,
        "physical_attempt_key": None,
        "physical_request_binding_sha256": None,
        "physical_ordinal": None,
        "started_turn_id": None,
        "provider_idempotency_key": None,
        "provider_request_id": None,
    }
    values.update(overrides)
    return recovery.RuntimeModelCallRecoverySnapshot(**values)  # type: ignore[arg-type]


def test_recovery_projection_validates_dispatch_identity_as_before() -> None:
    snapshot = _snapshot()
    assert snapshot.disposition == "ready"

    with pytest.raises(ValueError, match="physical identity is incomplete"):
        _snapshot(physical_attempt_id="physical-1")
    with pytest.raises(ValueError, match="external identity has no dispatch"):
        _snapshot(provider_request_id="provider-1")


def test_recovery_contracts_are_owned_by_the_recovery_module() -> None:
    assert (
        model_calls.RuntimeModelCallRecoveryDisposition
        is recovery.RuntimeModelCallRecoveryDisposition
    )
    assert (
        model_calls.RuntimeModelCallRecoverySnapshot
        is recovery.RuntimeModelCallRecoverySnapshot
    )


@pytest.mark.parametrize(
    ("settlement_outcome", "attempt_count", "max_attempts", "expected"),
    (
        (None, 0, 2, "ready"),
        (RuntimeModelPhysicalOutcome.SUCCEEDED, 2, 2, "succeeded"),
        (
            RuntimeModelPhysicalOutcome.TERMINAL_FAILURE,
            2,
            2,
            "terminal_failure",
        ),
        (
            RuntimeModelPhysicalOutcome.RETRYABLE_FAILURE,
            2,
            2,
            "physical_limit_exhausted",
        ),
    ),
)
def test_recovery_projector_preserves_terminal_precedence_and_attempt_limit(
    settlement_outcome: RuntimeModelPhysicalOutcome | None,
    attempt_count: int,
    max_attempts: int,
    expected: recovery.RuntimeModelCallRecoveryDisposition,
) -> None:
    logical_request = SimpleNamespace(
        logical_call_id="logical-1",
        provider="mock",
        model="mock-structured",
        endpoint_fingerprint="a" * 64,
        request_sha256="b" * 64,
        max_physical_attempts=max_attempts,
    )
    physical_attempts = tuple(
        SimpleNamespace(
            request=SimpleNamespace(
                physical_attempt_id=f"physical-{ordinal}",
                physical_attempt_key=f"physical-key-{ordinal}",
                binding_sha256="c" * 64,
                physical_ordinal=ordinal,
                started_turn_id=f"turn-{ordinal}",
                provider_idempotency_key=None,
            ),
            settlement=(
                None
                if settlement_outcome is None
                else SimpleNamespace(
                    outcome=settlement_outcome,
                    provider_request_id=f"provider-request-{ordinal}",
                )
            ),
        )
        for ordinal in range(1, attempt_count + 1)
    )

    snapshot = recovery.project_runtime_model_call_recovery(
        logical_request=logical_request,  # type: ignore[arg-type]
        physical_attempts=physical_attempts,  # type: ignore[arg-type]
    )

    assert snapshot.disposition == expected
    assert snapshot.logical_call_id == "logical-1"
    if attempt_count == 0:
        assert snapshot.physical_attempt_id is None
        assert snapshot.provider_request_id is None
    else:
        assert snapshot.physical_attempt_id == f"physical-{attempt_count}"
        assert snapshot.provider_request_id == f"provider-request-{attempt_count}"
