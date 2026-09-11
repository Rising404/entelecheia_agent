from __future__ import annotations

import ast
from dataclasses import dataclass, replace
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest

from personagraph.l2.task_execution.tool_bridge import (
    protected_dispatch as protected_dispatch_module,
)
from personagraph.l2.task_execution.tool_bridge.protected_dispatch import (
    RuntimeProtectedToolDispatcher,
    build_runtime_protected_tool_dispatcher,
)
from personagraph.runtime.tool_calls import (
    RuntimeLogicalToolCallAuthority,
    RuntimeToolCallAuthorityError,
    RuntimeToolCallStateGuardRejected,
    RuntimeToolCallTerminalState,
    RuntimeToolCallWaitingExternal,
    RuntimeToolDispatchObservation,
    RuntimeToolEffectClass,
    RuntimeToolLogicalRequest,
    RuntimeToolPhysicalAttemptRequest,
    RuntimeToolPhysicalAttemptSettlement,
    RuntimeToolPhysicalOutcome,
    RuntimeToolRetryAuthority,
)
from personagraph.runtime.tool_calls import authority as authority_module


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64


def test_tool_authority_and_dispatcher_require_an_injected_ledger() -> None:
    for module in (authority_module, protected_dispatch_module):
        tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
        imported_modules = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        } | {
            node.module or ""
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
        }
        assert not any(
            name == "session"
            or name.startswith("session.")
            or name == "personagraph.session"
            or name.startswith("personagraph.session.")
            for name in imported_modules
        )

    assert (
        inspect.signature(RuntimeLogicalToolCallAuthority)
        .parameters["store"]
        .default
        is inspect.Parameter.empty
    )
    assert (
        inspect.signature(RuntimeProtectedToolDispatcher)
        .parameters["ledger_store"]
        .default
        is inspect.Parameter.empty
    )
    assert (
        inspect.signature(build_runtime_protected_tool_dispatcher)
        .parameters["ledger_store"]
        .default
        is inspect.Parameter.empty
    )


@dataclass(frozen=True)
class _StoredPhysical:
    request: RuntimeToolPhysicalAttemptRequest
    settlement: RuntimeToolPhysicalAttemptSettlement | None = None


@dataclass(frozen=True)
class _StoredLogical:
    request: RuntimeToolLogicalRequest
    physical_attempts: tuple[_StoredPhysical, ...] = ()


class _MemoryLedger:
    def __init__(self) -> None:
        self.logical: _StoredLogical | None = None
        self.reserve_count = 0
        self.append_count = 0
        self.settle_count = 0

    def reserve_runtime_tool_logical_call(
        self, *, request: RuntimeToolLogicalRequest
    ) -> object:
        self.reserve_count += 1
        if self.logical is None:
            self.logical = _StoredLogical(request=request)
        elif self.logical.request != request:
            raise AssertionError("logical collision")
        return SimpleNamespace(logical_tool_call_id=request.logical_tool_call_id)

    def append_runtime_tool_physical_attempt(
        self, *, request: RuntimeToolPhysicalAttemptRequest
    ) -> object:
        assert self.logical is not None
        self.append_count += 1
        existing = tuple(
            item
            for item in self.logical.physical_attempts
            if item.request.physical_attempt_id == request.physical_attempt_id
        )
        if existing:
            if existing[0].request != request:
                raise AssertionError("physical collision")
        else:
            self.logical = replace(
                self.logical,
                physical_attempts=(
                    *self.logical.physical_attempts,
                    _StoredPhysical(request=request),
                ),
            )
        return SimpleNamespace(physical_attempt_id=request.physical_attempt_id)

    def settle_runtime_tool_physical_attempt(
        self, *, settlement: RuntimeToolPhysicalAttemptSettlement
    ) -> object:
        assert self.logical is not None
        self.settle_count += 1
        matched = False
        attempts: list[_StoredPhysical] = []
        for item in self.logical.physical_attempts:
            if item.request.physical_attempt_id != settlement.physical_attempt_id:
                attempts.append(item)
                continue
            matched = True
            if item.settlement is not None and item.settlement != settlement:
                raise AssertionError("settlement collision")
            attempts.append(replace(item, settlement=settlement))
        assert matched
        self.logical = replace(self.logical, physical_attempts=tuple(attempts))
        return SimpleNamespace(settlement_id=settlement.settlement_id)

    def get_runtime_tool_logical_call(
        self, *, session_id: str, logical_tool_call_id: str
    ) -> _StoredLogical | None:
        if (
            self.logical is None
            or self.logical.request.session_id != session_id
            or self.logical.request.logical_tool_call_id != logical_tool_call_id
        ):
            return None
        return self.logical


def _logical(
    *,
    effect: RuntimeToolEffectClass = RuntimeToolEffectClass.READ_ONLY,
    retry: RuntimeToolRetryAuthority = RuntimeToolRetryAuthority.READ_ONLY_REPLAY,
    logical_tool_call_id: str = "tool_call_01",
    work_run_id: str = "work_run_01",
    attempt_id: str = "attempt_01",
    provider_identity_sha256: str = SHA_C,
) -> RuntimeToolLogicalRequest:
    return RuntimeToolLogicalRequest.create(
        logical_tool_call_id=logical_tool_call_id,
        session_id="session_01",
        work_run_id=work_run_id,
        attempt_id=attempt_id,
        call_ordinal=1,
        invocation_turn_id="turn_01",
        catalog_snapshot_sha256=SHA_A,
        tool_id="document_read",
        contract_version="document-read-v2",
        implementation_version="document-read-impl-v3",
        provider_identity_sha256=provider_identity_sha256,
        effect_profile_sha256=SHA_B,
        effect_class=effect,
        retry_authority=retry,
        arguments={"document_alias": "doc_01", "page": 2},
        result_contract="document-read-result-v2",
        max_physical_attempts=4,
        state_guard_sha256=SHA_B,
    )


def _authority(
    ledger: _MemoryLedger,
    *,
    logical: RuntimeToolLogicalRequest | None = None,
    guard: str = SHA_B,
) -> RuntimeLogicalToolCallAuthority:
    return RuntimeLogicalToolCallAuthority(
        logical_request=logical or _logical(),
        state_guard_sha256=lambda: guard,
        store=ledger,
    )


def _observation(
    logical: RuntimeToolLogicalRequest,
    physical: RuntimeToolPhysicalAttemptRequest,
    *,
    outcome: RuntimeToolPhysicalOutcome = RuntimeToolPhysicalOutcome.SUCCEEDED,
    result: object = None,
    error_code: str | None = None,
    **overrides: object,
) -> RuntimeToolDispatchObservation:
    values: dict[str, object] = {
        "session_id": logical.session_id,
        "work_run_id": logical.work_run_id,
        "attempt_id": logical.attempt_id,
        "logical_tool_call_id": logical.logical_tool_call_id,
        "logical_request_binding_sha256": logical.binding_sha256,
        "physical_attempt_id": physical.physical_attempt_id,
        "physical_request_binding_sha256": physical.binding_sha256,
        "physical_ordinal": physical.physical_ordinal,
        "provider_identity_sha256": logical.provider_identity_sha256,
        "tool_id": logical.tool_id,
        "contract_version": logical.contract_version,
        "implementation_version": logical.implementation_version,
        "outcome": outcome,
        "provider_request_id": "provider-request-01",
        "duration_ms": 9,
        "error_code": error_code,
    }
    values.update(overrides)
    if outcome is RuntimeToolPhysicalOutcome.SUCCEEDED:
        return RuntimeToolDispatchObservation.create(
            result=(result if result is not None else {"page": 2, "text": "evidence"}),
            **values,
        )
    return RuntimeToolDispatchObservation.create(**values)


def _start(
    authority: RuntimeLogicalToolCallAuthority,
    *,
    turn_id: str = "turn_01",
) -> RuntimeToolPhysicalAttemptRequest:
    authority.reserve(turn_id=turn_id)
    return authority.begin_physical_attempt(
        turn_id=turn_id,
        max_physical_attempts=4,
    )


def test_reservation_success_and_settlement_are_exactly_replayable() -> None:
    ledger = _MemoryLedger()
    authority = _authority(ledger)
    authority.reserve(turn_id="turn_01")
    authority.reserve(turn_id="turn_02")
    physical = authority.begin_physical_attempt(
        turn_id="turn_01", max_physical_attempts=4
    )
    observation = _observation(authority.logical_request, physical)
    fresh = authority.settle_physical_attempt(
        turn_id="turn_01",
        physical=physical,
        observation=observation,
    )
    exact_replay = authority.settle_physical_attempt(
        turn_id="turn_01",
        physical=physical,
        observation=observation,
    )
    replay = authority.replay_succeeded_result()

    assert ledger.reserve_count == 2
    assert ledger.append_count == 1
    assert ledger.settle_count == 1
    assert exact_replay == fresh
    assert replay is not None
    assert replay.physical_attempt_id == physical.physical_attempt_id
    assert replay.value == {"page": 2, "text": "evidence"}
    assert replay.typed_result.result_contract == "document-read-result-v2"
    with pytest.raises(RuntimeToolCallTerminalState, match="terminal"):
        authority.begin_physical_attempt(
            turn_id="turn_03", max_physical_attempts=4
        )


def test_one_pending_physical_dispatch_blocks_duplicate_io() -> None:
    ledger = _MemoryLedger()
    authority = _authority(ledger)
    first = _start(authority)

    with pytest.raises(RuntimeToolCallWaitingExternal, match="reconciliation"):
        authority.replay_succeeded_result()
    with pytest.raises(RuntimeToolCallWaitingExternal, match="reconciliation"):
        authority.begin_physical_attempt(
            turn_id="turn_02", max_physical_attempts=4
        )

    assert first.physical_ordinal == 1
    assert ledger.append_count == 1


@pytest.mark.parametrize(
    ("effect", "retry", "expects_key"),
    [
        (
            RuntimeToolEffectClass.READ_ONLY,
            RuntimeToolRetryAuthority.READ_ONLY_REPLAY,
            False,
        ),
        (
            RuntimeToolEffectClass.PROTECTED_EFFECT,
            RuntimeToolRetryAuthority.PROVIDER_IDEMPOTENCY,
            True,
        ),
    ],
)
def test_retryable_failure_allows_only_authorized_next_physical_attempt(
    effect: RuntimeToolEffectClass,
    retry: RuntimeToolRetryAuthority,
    expects_key: bool,
) -> None:
    ledger = _MemoryLedger()
    logical = _logical(effect=effect, retry=retry)
    authority = _authority(ledger, logical=logical)
    first = _start(authority)
    authority.settle_physical_attempt(
        turn_id="turn_01",
        physical=first,
        observation=_observation(
            logical,
            first,
            outcome=RuntimeToolPhysicalOutcome.RETRYABLE_FAILURE,
            error_code="temporary_transport_failure",
        ),
    )
    assert authority.replay_succeeded_result() is None
    second = authority.begin_physical_attempt(
        turn_id="turn_02", max_physical_attempts=4
    )

    assert second.physical_ordinal == 2
    assert (first.provider_idempotency_key is not None) is expects_key
    assert second.provider_idempotency_key == first.provider_idempotency_key


def test_reconciliation_required_never_turns_retryable_fact_into_retry_grant() -> None:
    ledger = _MemoryLedger()
    logical = _logical(
        effect=RuntimeToolEffectClass.PROTECTED_EFFECT,
        retry=RuntimeToolRetryAuthority.RECONCILIATION_REQUIRED,
    )
    authority = _authority(ledger, logical=logical)
    physical = _start(authority)
    assert physical.provider_idempotency_key is None
    authority.settle_physical_attempt(
        turn_id="turn_01",
        physical=physical,
        observation=_observation(
            logical,
            physical,
            outcome=RuntimeToolPhysicalOutcome.RETRYABLE_FAILURE,
            error_code="known_transport_failure",
        ),
    )

    with pytest.raises(RuntimeToolCallTerminalState, match="no retry authority"):
        authority.replay_succeeded_result()
    with pytest.raises(RuntimeToolCallTerminalState, match="not authorized"):
        authority.begin_physical_attempt(
            turn_id="turn_02", max_physical_attempts=4
        )
    assert ledger.append_count == 1


def test_uncertain_outcome_always_waits_external_and_is_never_retried() -> None:
    ledger = _MemoryLedger()
    logical = _logical(
        effect=RuntimeToolEffectClass.PROTECTED_EFFECT,
        retry=RuntimeToolRetryAuthority.PROVIDER_IDEMPOTENCY,
    )
    authority = _authority(ledger, logical=logical)
    physical = _start(authority)
    authority.settle_physical_attempt(
        turn_id="turn_01",
        physical=physical,
        observation=_observation(
            logical,
            physical,
            outcome=RuntimeToolPhysicalOutcome.UNCERTAIN,
            error_code="response_unknown_after_dispatch",
        ),
    )

    with pytest.raises(RuntimeToolCallWaitingExternal, match="reconciliation"):
        authority.replay_succeeded_result()
    with pytest.raises(RuntimeToolCallWaitingExternal, match="reconciliation"):
        authority.begin_physical_attempt(
            turn_id="turn_02", max_physical_attempts=4
        )
    assert ledger.append_count == 1


def test_terminal_failure_closes_the_logical_call() -> None:
    ledger = _MemoryLedger()
    authority = _authority(ledger)
    physical = _start(authority)
    authority.settle_physical_attempt(
        turn_id="turn_01",
        physical=physical,
        observation=_observation(
            authority.logical_request,
            physical,
            outcome=RuntimeToolPhysicalOutcome.TERMINAL_FAILURE,
            error_code="invalid_document",
        ),
    )

    with pytest.raises(RuntimeToolCallTerminalState, match="terminal failure"):
        authority.replay_succeeded_result()
    with pytest.raises(RuntimeToolCallTerminalState, match="terminal"):
        authority.begin_physical_attempt(
            turn_id="turn_02", max_physical_attempts=4
        )


def test_state_guard_drift_rejects_reserve_and_next_dispatch() -> None:
    ledger = _MemoryLedger()
    stale = _authority(ledger, guard=SHA_A)
    with pytest.raises(RuntimeToolCallStateGuardRejected, match="no longer matches"):
        stale.reserve(turn_id="turn_01")
    assert ledger.logical is None

    current = _authority(ledger)
    current.reserve(turn_id="turn_01")
    stale_after_reserve = _authority(ledger, guard=SHA_A)
    with pytest.raises(RuntimeToolCallStateGuardRejected, match="no longer matches"):
        stale_after_reserve.begin_physical_attempt(
            turn_id="turn_01", max_physical_attempts=4
        )
    assert ledger.append_count == 0


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("provider_identity_sha256", SHA_A),
        ("work_run_id", "other_work_run"),
        ("attempt_id", "other_attempt"),
        ("logical_tool_call_id", "other_call"),
        ("physical_attempt_id", "other_physical"),
        ("tool_id", "other_tool"),
    ],
)
def test_observation_cannot_cross_provider_subject_or_call_identity(
    field: str,
    replacement: object,
) -> None:
    ledger = _MemoryLedger()
    authority = _authority(ledger)
    physical = _start(authority)
    observation = _observation(
        authority.logical_request,
        physical,
        **{field: replacement},
    )

    with pytest.raises(
        RuntimeToolCallAuthorityError,
        match="provider, subject, or call",
    ):
        authority.settle_physical_attempt(
            turn_id="turn_01",
            physical=physical,
            observation=observation,
        )
    assert ledger.settle_count == 0


def test_cross_logical_physical_request_cannot_be_substituted() -> None:
    ledger = _MemoryLedger()
    authority = _authority(ledger)
    _start(authority)

    other_ledger = _MemoryLedger()
    other_logical = _logical(
        logical_tool_call_id="other_call",
        work_run_id="other_work_run",
        attempt_id="other_attempt",
    )
    other = _authority(other_ledger, logical=other_logical)
    other_physical = _start(other)
    observation = _observation(other_logical, other_physical)

    with pytest.raises(RuntimeToolCallAuthorityError, match="provider, subject, or call"):
        authority.settle_physical_attempt(
            turn_id="turn_01",
            physical=other_physical,
            observation=observation,
        )


def test_host_revalidates_forged_logical_physical_and_settlement_self_hashes() -> None:
    valid = _logical()
    forged_logical = RuntimeToolLogicalRequest.model_construct(
        **(valid.model_dump() | {"tool_id": "forged_tool"})
    )
    with pytest.raises(RuntimeToolCallAuthorityError, match="self-hash"):
        _authority(_MemoryLedger(), logical=forged_logical)

    ledger = _MemoryLedger()
    authority = _authority(ledger)
    physical = _start(authority)
    forged_physical = RuntimeToolPhysicalAttemptRequest.model_construct(
        **(
            physical.model_dump()
            | {"started_turn_id": "forged_turn"}
        )
    )
    with pytest.raises(RuntimeToolCallAuthorityError, match="self-hash"):
        authority.settle_physical_attempt(
            turn_id="turn_01",
            physical=forged_physical,
            observation=_observation(authority.logical_request, physical),
        )

    valid_observation = _observation(authority.logical_request, physical)
    forged_observation = RuntimeToolDispatchObservation.model_construct(
        **(
            valid_observation.model_dump()
            | {"provider_identity_sha256": SHA_A}
        )
    )
    with pytest.raises(RuntimeToolCallAuthorityError, match="self-hash"):
        authority.settle_physical_attempt(
            turn_id="turn_01",
            physical=physical,
            observation=forged_observation,
        )

    settlement = authority.settle_physical_attempt(
        turn_id="turn_01",
        physical=physical,
        observation=_observation(authority.logical_request, physical),
    )
    assert ledger.logical is not None
    forged_settlement_values = settlement.model_dump()
    forged_settlement_values.update(
        provider_request_id="forged-provider-request",
        typed_result=settlement.typed_result,
    )
    forged_settlement = RuntimeToolPhysicalAttemptSettlement.model_construct(
        **forged_settlement_values
    )
    ledger.logical = replace(
        ledger.logical,
        physical_attempts=(
            replace(
                ledger.logical.physical_attempts[0],
                settlement=forged_settlement,
            ),
        ),
    )
    with pytest.raises(RuntimeToolCallAuthorityError, match="self-hash"):
        authority.replay_succeeded_result()


def test_stored_logical_subject_cannot_cross_the_requested_lookup() -> None:
    ledger = _MemoryLedger()
    authority = _authority(ledger)
    authority.reserve(turn_id="turn_01")
    assert ledger.logical is not None
    crossed = _logical(work_run_id="other_work_run")
    ledger.logical = replace(ledger.logical, request=crossed)

    with pytest.raises(RuntimeToolCallAuthorityError, match="immutable authority"):
        authority.replay_succeeded_result()
