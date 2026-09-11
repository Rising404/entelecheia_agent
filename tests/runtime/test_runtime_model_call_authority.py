from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest

from personagraph.model_io.gateway import ModelGatewayError, ModelResult
from personagraph.runtime.model_calls import (
    RuntimeModelLogicalRequest,
    RuntimeModelPhysicalAttemptRequest,
    RuntimeModelPhysicalAttemptSettlement,
)
from personagraph.model_io.output_repair_contracts import (
    RuntimeModelOutputRepairFeedback,
    RuntimeModelOutputRepairIssueCategory,
    RuntimeModelOutputRepairIssueCoverage,
    RuntimeModelOutputRepairIssue,
    RuntimeModelOutputRepairProtocol,
    RuntimeModelStructuredPrompt,
)
from personagraph.runtime.model_calls import (
    request_model_with_retry,
)
from personagraph.runtime.turn_deadline import TurnDeadlineExceeded
from personagraph.model_io.output_validation import ModelOutputValidationError
from personagraph.runtime.model_calls import (
    RuntimeLogicalModelCallAuthority,
    RuntimeModelCallAuthorityError,
    RuntimeModelCallWaitingExternal,
)
from personagraph.runtime.turn_events import RuntimeStage
from personagraph.session import store as session_store
from personagraph.session.runtime_call_ledger_store_facade import (
    build_runtime_call_ledger_store_facade,
)
from personagraph.session.persistence.calls.runtime_model_calls import (
    RuntimeModelCallPersistenceError,
    StoredRuntimeModelLogicalCall,
    StoredRuntimeModelPhysicalAttempt,
)
from tests.helpers.prepared_model_provider import prepare_test_model_request


SHA_A = "a" * 64
SHA_B = "b" * 64
_FOUR_MESSAGE = (
    RuntimeModelOutputRepairProtocol.FOUR_MESSAGE_WHOLE_RESPONSE_REGENERATION
)


class _MemoryLedger:
    def __init__(self) -> None:
        self.logical: StoredRuntimeModelLogicalCall | None = None
        self.rejected_outputs: dict[
            tuple[str, int, str], SimpleNamespace
        ] = {}

    def reserve_runtime_model_logical_call(
        self, *, request: RuntimeModelLogicalRequest
    ) -> object:
        if self.logical is None:
            self.logical = StoredRuntimeModelLogicalCall(request=request)
        elif self.logical.request != request:
            raise AssertionError("logical collision")
        return SimpleNamespace(logical_call=self.logical)

    def append_runtime_model_physical_attempt(
        self, *, request: RuntimeModelPhysicalAttemptRequest
    ) -> object:
        assert self.logical is not None
        self.logical = StoredRuntimeModelLogicalCall(
            request=self.logical.request,
            physical_attempts=(
                *self.logical.physical_attempts,
                StoredRuntimeModelPhysicalAttempt(request=request),
            ),
        )
        return SimpleNamespace(physical_attempt_id=request.physical_attempt_id)

    def settle_runtime_model_physical_attempt(
        self,
        *,
        settlement: RuntimeModelPhysicalAttemptSettlement,
        rejected_response_text: str | None = None,
    ) -> object:
        assert self.logical is not None
        feedback = settlement.next_output_repair_feedback
        if isinstance(feedback, RuntimeModelOutputRepairFeedback):
            assert rejected_response_text is not None
            assert hashlib.sha256(
                rejected_response_text.encode("utf-8")
            ).hexdigest() == feedback.rejected_response_sha256
            self.rejected_outputs[
                (
                    settlement.logical_call_id,
                    feedback.rejected_physical_ordinal,
                    feedback.rejected_response_sha256,
                )
            ] = SimpleNamespace(response_text=rejected_response_text)
        else:
            assert rejected_response_text is None
        attempts = tuple(
            StoredRuntimeModelPhysicalAttempt(
                request=item.request,
                settlement=(
                    settlement
                    if item.request.physical_attempt_id
                    == settlement.physical_attempt_id
                    else item.settlement
                ),
            )
            for item in self.logical.physical_attempts
        )
        self.logical = StoredRuntimeModelLogicalCall(
            request=self.logical.request,
            physical_attempts=attempts,
        )
        return SimpleNamespace(settlement_id=settlement.settlement_id)

    def get_runtime_model_rejected_output(
        self,
        *,
        session_id: str,
        logical_call_id: str,
        rejected_physical_ordinal: int,
        rejected_response_sha256: str,
    ) -> object | None:
        if (
            self.logical is None
            or self.logical.request.session_id != session_id
            or self.logical.request.logical_call_id != logical_call_id
        ):
            return None
        return self.rejected_outputs.get(
            (
                logical_call_id,
                rejected_physical_ordinal,
                rejected_response_sha256,
            )
        )

    def get_runtime_model_logical_call(
        self, *, session_id: str, logical_call_id: str
    ) -> StoredRuntimeModelLogicalCall | None:
        if (
            self.logical is None
            or self.logical.request.session_id != session_id
            or self.logical.request.logical_call_id != logical_call_id
        ):
            return None
        return self.logical


def _request(
    *,
    session_id: str = "session_01",
    turn_id: str = "turn_01",
    logical_call_id: str = "logical_runtime_01",
    repair_protocol: RuntimeModelOutputRepairProtocol | None = None,
) -> RuntimeModelLogicalRequest:
    structured_prompt = (
        RuntimeModelStructuredPrompt.create(
            system_prompt="system",
            user_content="user",
        )
        if repair_protocol is _FOUR_MESSAGE
        else None
    )
    return RuntimeModelLogicalRequest.create(
        logical_call_id=logical_call_id,
        session_id=session_id,
        invocation_turn_id=turn_id,
        call_kind="architect",
        purpose="runtime_architect_test",
        provider="test",
        model="test-model",
        endpoint_fingerprint=SHA_A,
        request_contract="architect-request-v1",
        request_payload={"goal": "bounded"},
        output_repair_protocol=repair_protocol,
        structured_prompt=structured_prompt,
        typed_result_contract="architect-provider-envelope-v1",
        max_physical_attempts=6,
        state_guard_sha256=SHA_B,
    )


def _authority(
    ledger: _MemoryLedger,
    *,
    guard: str = SHA_B,
    provider_idempotency_key: str | None = None,
    repair_protocol: RuntimeModelOutputRepairProtocol | None = None,
) -> RuntimeLogicalModelCallAuthority:
    return RuntimeLogicalModelCallAuthority(
        logical_request=_request(repair_protocol=repair_protocol),
        state_guard_sha256=lambda: guard,
        typed_replay_payload_builder=lambda result, _value: json.loads(
            result.reply
        ),
        provider_idempotency_key=provider_idempotency_key,
        store=ledger,
    )


def _session_store_authority(
    request: RuntimeModelLogicalRequest,
) -> RuntimeLogicalModelCallAuthority:
    return RuntimeLogicalModelCallAuthority(
        logical_request=request,
        state_guard_sha256=lambda: SHA_B,
        typed_replay_payload_builder=lambda result, _value: json.loads(
            result.reply
        ),
        store=build_runtime_call_ledger_store_facade(
            deps_factory=session_store.current_store_deps,
        ),
    )


def _seed_session_store_rejection(
    *, suffix: str,
) -> tuple[
    RuntimeModelLogicalRequest,
    RuntimeModelOutputRepairFeedback,
    str,
]:
    session_id = session_store.create_session("Entelecheia")
    turn_id = f"turn-repair-recovery-{suffix}"
    session_store.create_runtime_turn(
        turn_id=turn_id,
        session_id=session_id,
        source="runtime-model-repair-recovery-test",
        user_text="测试失败输出恢复",
    )
    request = _request(
        session_id=session_id,
        turn_id=turn_id,
        logical_call_id=f"logical-runtime-repair-recovery-{suffix}",
        repair_protocol=_FOUR_MESSAGE,
    )
    authority = _session_store_authority(request)
    authority.reserve(turn_id=turn_id)
    physical = authority.begin_physical_attempt(
        turn_id=turn_id,
        max_physical_attempts=6,
        output_repair_enabled=True,
    )
    rejected_response = '{"proposal":"durably-rejected"}'
    rejected = _rejected_result(physical, rejected_response)
    feedback = _repair_feedback(
        physical_ordinal=physical.physical_ordinal,
        reply=rejected_response,
    )
    failure = ModelGatewayError(
        "MODEL_BAD_RESPONSE",
        "typed output rejected",
        retryable=True,
    )
    authority.settle_physical_attempt(
        turn_id=turn_id,
        physical=physical,
        outcome="retryable_failure",
        result_fingerprint=authority.failure_fingerprint(
            error=failure,
            provider_result=rejected,
        ),
        provider_result=rejected,
        error_code="MODEL_BAD_RESPONSE",
        next_output_repair_feedback=feedback,
        rejected_response_text=rejected_response,
    )
    return request, feedback, rejected_response


def test_generic_authority_persists_and_replays_typed_success() -> None:
    ledger = _MemoryLedger()
    provider_calls = 0

    def provider(model_call_id: str) -> ModelResult:
        nonlocal provider_calls
        provider_calls += 1
        return ModelResult(
            reply='{"proposal":"bounded"}',
            provider="test",
            model="test-model",
            latency_ms=7,
            input_tokens=11,
            output_tokens=5,
            finish_reason="end_turn",
            model_call_id=model_call_id,
        )

    fresh = request_model_with_retry(
        turn_id="turn_01",
        session_id="session_01",
        purpose="runtime_architect_test",
        stage=RuntimeStage.L2_PLAN,
        prepare_request=prepare_test_model_request(provider),
        validate=lambda result: json.loads(result.reply),
        emit=lambda _event: None,
        durable_call=_authority(ledger),
        logical_model_call_id="logical_runtime_01",
    )
    replayed = request_model_with_retry(
        turn_id="turn_02",
        session_id="session_01",
        purpose="runtime_architect_test",
        stage=RuntimeStage.L2_PLAN,
        prepare_request=prepare_test_model_request(
            lambda _model_call_id: pytest.fail("replay reached Provider")
        ),
        validate=lambda result: json.loads(result.reply),
        emit=lambda _event: None,
        durable_call=_authority(ledger),
        logical_model_call_id="logical_runtime_01",
    )

    assert provider_calls == 1
    assert fresh.value == replayed.value == {"proposal": "bounded"}
    assert replayed.replayed is True
    assert ledger.logical is not None
    settlement = ledger.logical.physical_attempts[0].settlement
    assert settlement is not None and settlement.typed_result is not None
    assert settlement.typed_result.parsed() == {"proposal": "bounded"}
    assert settlement.usage.input_tokens == 11


def test_pre_provider_deadline_is_recoverable_without_blind_provider_retry() -> None:
    ledger = _MemoryLedger()
    provider_calls = 0

    class _ExpiresAfterLogicalReservation:
        def __init__(self) -> None:
            self.remaining_checks = 0

        def expired(self) -> bool:
            return False

        def remaining_s(self) -> float:
            self.remaining_checks += 1
            return 1.0 if self.remaining_checks <= 2 else 0.0

    def provider(model_call_id: str) -> ModelResult:
        nonlocal provider_calls
        provider_calls += 1
        return ModelResult(
            reply='{"proposal":"recovered"}',
            provider="test",
            model="test-model",
            latency_ms=7,
            model_call_id=model_call_id,
        )

    with pytest.raises(TurnDeadlineExceeded):
        request_model_with_retry(
            turn_id="turn_01",
            session_id="session_01",
            purpose="runtime_architect_test",
            stage=RuntimeStage.L2_PLAN,
            prepare_request=prepare_test_model_request(provider),
            validate=lambda result: json.loads(result.reply),
            emit=lambda _event: None,
            deadline=_ExpiresAfterLogicalReservation(),  # type: ignore[arg-type]
            durable_call=_authority(ledger),
            logical_model_call_id="logical_runtime_01",
        )

    assert provider_calls == 0
    assert ledger.logical is not None
    # 第二次截止时间检查必须发生在消耗持久物理尝试权威之前。否则，即使提供方 I/O
    # 一次也未发生，六次轮次边界竞争也可能耗尽逻辑调用。
    assert ledger.logical.physical_attempts == ()

    recovered = request_model_with_retry(
        turn_id="turn_02",
        session_id="session_01",
        purpose="runtime_architect_test",
        stage=RuntimeStage.L2_PLAN,
        prepare_request=prepare_test_model_request(provider),
        validate=lambda result: json.loads(result.reply),
        emit=lambda _event: None,
        durable_call=_authority(ledger),
        logical_model_call_id="logical_runtime_01",
    )

    assert recovered.value == {"proposal": "recovered"}
    assert recovered.attempts == 1
    assert provider_calls == 1
    assert ledger.logical is not None
    assert tuple(
        item.settlement.outcome.value
        for item in ledger.logical.physical_attempts
        if item.settlement is not None
    ) == ("succeeded",)


def test_pending_physical_attempt_blocks_duplicate_provider_dispatch() -> None:
    ledger = _MemoryLedger()
    authority = _authority(ledger)
    authority.reserve(turn_id="turn_01")
    authority.begin_physical_attempt(turn_id="turn_01", max_physical_attempts=6)

    with pytest.raises(RuntimeModelCallWaitingExternal, match="reconciliation"):
        request_model_with_retry(
            turn_id="turn_02",
            session_id="session_01",
            purpose="runtime_architect_test",
            stage=RuntimeStage.L2_PLAN,
            prepare_request=prepare_test_model_request(
                lambda _model_call_id: pytest.fail("pending call was duplicated")
            ),
            validate=lambda result: json.loads(result.reply),
            emit=lambda _event: None,
            durable_call=_authority(ledger),
            logical_model_call_id="logical_runtime_01",
        )

    assert ledger.logical is not None
    assert len(ledger.logical.physical_attempts) == 1


def test_pending_dispatch_has_explicit_read_and_settlement_seams() -> None:
    ledger = _MemoryLedger()
    authority = _authority(ledger)
    authority.reserve(turn_id="turn_01")
    authority.begin_physical_attempt(
        turn_id="turn_01",
        max_physical_attempts=6,
    )

    pending = authority.inspect_recovery()
    assert pending.disposition == "waiting_pending"
    assert pending.physical_attempt_id is not None
    settled = authority.settle_pending_external(
        turn_id="turn_02",
        outcome="retryable_failure",
        result_fingerprint=SHA_A,
        provider_request_id="provider-request-01",
        error_code="provider_timeout_reconciled",
    )

    assert settled.disposition == "retry_authorized"
    next_physical = authority.begin_physical_attempt(
        turn_id="turn_02",
        max_physical_attempts=6,
    )
    assert next_physical.physical_ordinal == 2


def test_uncertain_dispatch_remains_waiting_and_cannot_be_blindly_resettled() -> None:
    ledger = _MemoryLedger()
    authority = _authority(ledger)
    authority.reserve(turn_id="turn_01")
    physical = authority.begin_physical_attempt(
        turn_id="turn_01",
        max_physical_attempts=6,
    )
    authority.settle_physical_attempt(
        turn_id="turn_01",
        physical=physical,
        outcome="uncertain",
        result_fingerprint=SHA_A,
        provider_request_id="provider-request-uncertain",
        error_code="provider_result_uncertain",
    )

    uncertain = authority.inspect_recovery()
    assert uncertain.disposition == "waiting_uncertain"
    assert uncertain.provider_request_id == "provider-request-uncertain"
    with pytest.raises(RuntimeModelCallWaitingExternal, match="pending"):
        authority.settle_pending_external(
            turn_id="turn_02",
            outcome="retryable_failure",
            result_fingerprint=SHA_B,
            error_code="unsafe_retry_rejected",
        )
    with pytest.raises(RuntimeModelCallWaitingExternal, match="reconciliation"):
        request_model_with_retry(
            turn_id="turn_02",
            session_id="session_01",
            purpose="runtime_architect_test",
            stage=RuntimeStage.L2_PLAN,
            prepare_request=prepare_test_model_request(
                lambda _model_call_id: pytest.fail("uncertain call was resent")
            ),
            validate=lambda result: json.loads(result.reply),
            emit=lambda _event: None,
            durable_call=_authority(ledger),
            logical_model_call_id="logical_runtime_01",
        )


def test_state_guard_drift_prevents_reservation_and_provider_io() -> None:
    ledger = _MemoryLedger()
    with pytest.raises(Exception, match="no longer matches"):
        request_model_with_retry(
            turn_id="turn_01",
            session_id="session_01",
            purpose="runtime_architect_test",
            stage=RuntimeStage.L2_PLAN,
            prepare_request=prepare_test_model_request(
                lambda _model_call_id: pytest.fail("drift reached Provider")
            ),
            validate=lambda result: json.loads(result.reply),
            emit=lambda _event: None,
            durable_call=_authority(ledger, guard=SHA_A),
            logical_model_call_id="logical_runtime_01",
        )
    assert ledger.logical is None


def _rejected_result(physical, reply: str) -> ModelResult:
    return ModelResult(
        reply=reply,
        provider=physical.provider,
        model=physical.model,
        latency_ms=3,
        model_call_id=physical.model_call_id,
    )


def _repair_feedback(
    *,
    physical_ordinal: int,
    reply: str,
    code: str = "architect_guard_rejected",
    target_contract: str = "architect-provider-envelope-v1",
) -> RuntimeModelOutputRepairFeedback:
    return RuntimeModelOutputRepairFeedback(
        target_contract=target_contract,
        rejected_physical_ordinal=physical_ordinal,
        rejected_response_sha256=hashlib.sha256(
            reply.encode("utf-8")
        ).hexdigest(),
        issue_coverage=RuntimeModelOutputRepairIssueCoverage.COMPLETE,
        omitted_issue_count=0,
        current_issues=(
            RuntimeModelOutputRepairIssue(
                category=RuntimeModelOutputRepairIssueCategory.HOST_GUARD,
                code=code,
                paths=("/proposal",),
                safe_explanation="提案没有通过确定性的 Host 合同检查。",
            ),
        ),
    )


def test_repair_feedback_is_persisted_recovered_and_bound_to_dispatch_variant() -> None:
    ledger = _MemoryLedger()
    first_authority = _authority(ledger, repair_protocol=_FOUR_MESSAGE)
    first_authority.reserve(turn_id="turn_01")
    first = first_authority.begin_physical_attempt(
        turn_id="turn_01",
        max_physical_attempts=6,
        output_repair_enabled=True,
    )
    rejected = _rejected_result(first, '{"proposal":"invalid"}')
    feedback = _repair_feedback(
        physical_ordinal=first.physical_ordinal,
        reply=rejected.reply,
    )
    failure = ModelGatewayError(
        "MODEL_BAD_RESPONSE",
        "typed output rejected",
        retryable=True,
    )
    first_authority.settle_physical_attempt(
        turn_id="turn_01",
        physical=first,
        outcome="retryable_failure",
        result_fingerprint=first_authority.failure_fingerprint(
            error=failure,
            provider_result=rejected,
        ),
        provider_result=rejected,
        error_code="MODEL_BAD_RESPONSE",
        next_output_repair_feedback=feedback,
        rejected_response_text=rejected.reply,
    )

    restarted = _authority(ledger, repair_protocol=_FOUR_MESSAGE)
    assert restarted.recover_output_repair_feedback() == feedback
    second = restarted.begin_physical_attempt(
        turn_id="turn_02",
        max_physical_attempts=6,
        output_repair_enabled=True,
        output_repair_feedback=feedback,
    )
    assert second.request_sha256 == first.request_sha256
    assert second.dispatch_request_sha256 != first.dispatch_request_sha256
    assert second.output_repair_feedback == feedback

    restarted.settle_physical_attempt(
        turn_id="turn_02",
        physical=second,
        outcome="retryable_failure",
        result_fingerprint=SHA_A,
        error_code="MODEL_CALL_FAILED",
    )
    after_transport_restart = _authority(
        ledger,
        repair_protocol=_FOUR_MESSAGE,
    )
    assert after_transport_restart.recover_output_repair_feedback() == feedback
    third = after_transport_restart.begin_physical_attempt(
        turn_id="turn_03",
        max_physical_attempts=6,
        output_repair_enabled=True,
        output_repair_feedback=feedback,
    )
    assert third.dispatch_request_sha256 == second.dispatch_request_sha256
    assert third.binding_sha256 != second.binding_sha256


def test_nonrepair_logical_request_cannot_open_a_repair_physical_attempt() -> None:
    ledger = _MemoryLedger()
    authority = _authority(ledger)
    authority.reserve(turn_id="turn_01")

    with pytest.raises(
        RuntimeModelCallAuthorityError,
        match="differs from the logical request",
    ):
        authority.begin_physical_attempt(
            turn_id="turn_01",
            max_physical_attempts=6,
            output_repair_enabled=True,
        )

    assert ledger.logical is not None
    assert ledger.logical.physical_attempts == ()


def test_authority_rejects_feedback_for_another_result_contract() -> None:
    ledger = _MemoryLedger()
    authority = _authority(ledger, repair_protocol=_FOUR_MESSAGE)
    authority.reserve(turn_id="turn_01")
    physical = authority.begin_physical_attempt(
        turn_id="turn_01",
        max_physical_attempts=6,
        output_repair_enabled=True,
    )
    rejected = _rejected_result(physical, '{"proposal":"invalid"}')
    feedback = _repair_feedback(
        physical_ordinal=physical.physical_ordinal,
        reply=rejected.reply,
        target_contract="another-result-contract",
    )

    with pytest.raises(
        RuntimeModelCallAuthorityError,
        match="target differs from the logical result contract",
    ):
        authority.settle_physical_attempt(
            turn_id="turn_01",
            physical=physical,
            outcome="retryable_failure",
            result_fingerprint=authority.failure_fingerprint(
                error=ModelGatewayError(
                    "MODEL_BAD_RESPONSE",
                    "typed output rejected",
                    retryable=True,
                ),
                provider_result=rejected,
            ),
            provider_result=rejected,
            error_code="MODEL_BAD_RESPONSE",
            next_output_repair_feedback=feedback,
            rejected_response_text=rejected.reply,
        )

    assert ledger.logical is not None
    assert ledger.logical.physical_attempts[0].settlement is None


def test_rejected_body_is_exactly_bound_recovered_and_kept_across_transport() -> None:
    ledger = _MemoryLedger()
    authority = _authority(ledger, repair_protocol=_FOUR_MESSAGE)
    authority.reserve(turn_id="turn_01")
    first = authority.begin_physical_attempt(
        turn_id="turn_01",
        max_physical_attempts=6,
        output_repair_enabled=True,
    )
    rejected = _rejected_result(first, '{"proposal":"invalid"}')
    feedback = _repair_feedback(
        physical_ordinal=first.physical_ordinal,
        reply=rejected.reply,
    )
    failure = ModelGatewayError(
        "MODEL_BAD_RESPONSE",
        "typed output rejected",
        retryable=True,
    )

    with pytest.raises(
        RuntimeModelCallAuthorityError,
        match="differs from the rejected Provider response",
    ):
        authority.settle_physical_attempt(
            turn_id="turn_01",
            physical=first,
            outcome="retryable_failure",
            result_fingerprint=authority.failure_fingerprint(
                error=failure,
                provider_result=rejected,
            ),
            provider_result=rejected,
            error_code="MODEL_BAD_RESPONSE",
            next_output_repair_feedback=feedback,
            rejected_response_text='{"proposal":"different"}',
        )

    authority.settle_physical_attempt(
        turn_id="turn_01",
        physical=first,
        outcome="retryable_failure",
        result_fingerprint=authority.failure_fingerprint(
            error=failure,
            provider_result=rejected,
        ),
        provider_result=rejected,
        error_code="MODEL_BAD_RESPONSE",
        next_output_repair_feedback=feedback,
        rejected_response_text=rejected.reply,
    )

    restarted = _authority(ledger, repair_protocol=_FOUR_MESSAGE)
    assert restarted.recover_output_repair_feedback() == feedback
    assert restarted.recover_output_repair_response(feedback) == rejected.reply
    second = restarted.begin_physical_attempt(
        turn_id="turn_02",
        max_physical_attempts=6,
        output_repair_enabled=True,
        output_repair_feedback=feedback,
    )
    restarted.settle_physical_attempt(
        turn_id="turn_02",
        physical=second,
        outcome="retryable_failure",
        result_fingerprint=SHA_A,
        error_code="MODEL_CALL_FAILED",
    )

    after_transport_restart = _authority(ledger, repair_protocol=_FOUR_MESSAGE)
    assert after_transport_restart.recover_output_repair_feedback() == feedback
    assert (
        after_transport_restart.recover_output_repair_response(feedback)
        == rejected.reply
    )


def test_rejected_body_recovers_from_session_store_after_restart() -> None:
    request, feedback, rejected_response = _seed_session_store_rejection(
        suffix="success"
    )

    restarted = _session_store_authority(request)

    assert restarted.recover_output_repair_feedback() == feedback
    assert (
        restarted.recover_output_repair_response(feedback)
        == rejected_response
    )
    repair_inputs: list[
        tuple[RuntimeModelOutputRepairFeedback, str]
    ] = []

    class Prepared:
        def dispatch(self, *, model_call_id: str) -> ModelResult:
            return ModelResult(
                reply='{"proposal":"accepted-after-restart"}',
                provider="test",
                model="test-model",
                latency_ms=2,
                model_call_id=model_call_id,
            )

    def prepare_repair(
        recovered_feedback: RuntimeModelOutputRepairFeedback,
        recovered_response: str,
    ) -> Prepared:
        repair_inputs.append((recovered_feedback, recovered_response))
        return Prepared()

    outcome = request_model_with_retry(
        turn_id=request.invocation_turn_id,
        session_id=request.session_id,
        purpose="runtime_architect_test",
        stage=RuntimeStage.L2_PLAN,
        prepare_request=Prepared,
        prepare_repair_request=prepare_repair,
        repair_target_contract="architect-provider-envelope-v1",
        validate=lambda result: json.loads(result.reply),
        emit=lambda _event: None,
        durable_call=restarted,
        logical_model_call_id=request.logical_call_id,
    )

    assert outcome.attempts == 2
    assert outcome.value == {"proposal": "accepted-after-restart"}
    assert repair_inputs == [(feedback, rejected_response)]


@pytest.mark.parametrize("mutation", ("missing", "tampered"))
def test_rejected_body_recovery_fails_closed_when_store_record_is_invalid(
    mutation: str,
) -> None:
    request, feedback, _ = _seed_session_store_rejection(
        suffix=mutation
    )
    with session_store._connect() as conn:
        if mutation == "missing":
            conn.execute(
                "DELETE FROM insession_runtime_model_rejected_outputs "
                "WHERE logical_call_id=?",
                (request.logical_call_id,),
            )
        else:
            conn.execute(
                "UPDATE insession_runtime_model_rejected_outputs "
                "SET response_text=? WHERE logical_call_id=?",
                ('{"proposal":"forged"}', request.logical_call_id),
            )

    restarted = _session_store_authority(request)
    with pytest.raises(RuntimeModelCallPersistenceError):
        restarted.recover_output_repair_response(feedback)


def test_physical_retry_rejects_feedback_not_authorized_by_prior_settlement() -> None:
    ledger = _MemoryLedger()
    authority = _authority(ledger, repair_protocol=_FOUR_MESSAGE)
    authority.reserve(turn_id="turn_01")
    first = authority.begin_physical_attempt(
        turn_id="turn_01",
        max_physical_attempts=6,
        output_repair_enabled=True,
    )
    rejected = _rejected_result(first, "rejected-one")
    authorized = _repair_feedback(
        physical_ordinal=first.physical_ordinal,
        reply=rejected.reply,
    )
    authority.settle_physical_attempt(
        turn_id="turn_01",
        physical=first,
        outcome="retryable_failure",
        result_fingerprint=SHA_A,
        provider_result=rejected,
        error_code="MODEL_BAD_RESPONSE",
        next_output_repair_feedback=authorized,
        rejected_response_text=rejected.reply,
    )
    unauthorized = _repair_feedback(
        physical_ordinal=first.physical_ordinal,
        reply=rejected.reply,
        code="different_guard_reason",
    )

    with pytest.raises(RuntimeModelCallAuthorityError, match="authorized"):
        authority.begin_physical_attempt(
            turn_id="turn_02",
            max_physical_attempts=6,
            output_repair_enabled=True,
            output_repair_feedback=unauthorized,
        )


def test_durable_repair_mode_cannot_settle_bad_response_without_feedback() -> None:
    ledger = _MemoryLedger()
    authority = _authority(ledger, repair_protocol=_FOUR_MESSAGE)
    authority.reserve(turn_id="turn_01")
    physical = authority.begin_physical_attempt(
        turn_id="turn_01",
        max_physical_attempts=6,
        output_repair_enabled=True,
    )
    rejected = _rejected_result(physical, "invalid")

    with pytest.raises(RuntimeModelCallAuthorityError, match="requires feedback"):
        authority.settle_physical_attempt(
            turn_id="turn_01",
            physical=physical,
            outcome="retryable_failure",
            result_fingerprint=SHA_A,
            provider_result=rejected,
            error_code="MODEL_BAD_RESPONSE",
        )


def test_shared_wrapper_persists_each_distinct_repair_dispatch() -> None:
    ledger = _MemoryLedger()
    repair_feedback = []

    class Prepared:
        def __init__(self, reply: str) -> None:
            self.reply = reply

        def dispatch(self, *, model_call_id: str) -> ModelResult:
            return ModelResult(
                reply=self.reply,
                provider="test",
                model="test-model",
                latency_ms=2,
                model_call_id=model_call_id,
            )

    def prepare_repair(
        feedback: RuntimeModelOutputRepairFeedback,
        rejected_response_text: str,
    ) -> Prepared:
        repair_feedback.append(feedback)
        assert rejected_response_text == '{"proposal":"rejected"}'
        return Prepared('{"proposal":"accepted"}')

    def validate(result: ModelResult) -> dict[str, str]:
        parsed = json.loads(result.reply)
        if parsed["proposal"] != "accepted":
            raise ModelOutputValidationError(
                "proposal rejected",
                repair_code="architect_guard_rejected",
                safe_repair_reason=(
                    "The proposal violates a deterministic frozen Guard."
                ),
            )
        return parsed

    outcome = request_model_with_retry(
        turn_id="turn_01",
        session_id="session_01",
        purpose="runtime_architect_test",
        stage=RuntimeStage.L2_PLAN,
        prepare_request=lambda: Prepared('{"proposal":"rejected"}'),
        prepare_repair_request=prepare_repair,
        repair_target_contract="architect-provider-envelope-v1",
        validate=validate,
        emit=lambda _event: None,
        durable_call=_authority(ledger, repair_protocol=_FOUR_MESSAGE),
        logical_model_call_id="logical_runtime_01",
    )

    assert outcome.attempts == 2
    assert len(repair_feedback) == 1
    assert ledger.logical is not None
    first, second = ledger.logical.physical_attempts
    assert first.request.output_repair_enabled is True
    assert first.request.output_repair_feedback is None
    assert first.settlement is not None
    assert first.settlement.next_output_repair_feedback == repair_feedback[0]
    assert second.request.output_repair_feedback == repair_feedback[0]
    assert first.request.request_sha256 == second.request.request_sha256
    assert first.request.dispatch_request_sha256 != (
        second.request.dispatch_request_sha256
    )


def test_explicit_nonrepair_bad_response_cannot_be_upgraded_on_recovery() -> None:
    ledger = _MemoryLedger()
    authority = _authority(ledger)
    authority.reserve(turn_id="turn_01")
    first = authority.begin_physical_attempt(
        turn_id="turn_01",
        max_physical_attempts=6,
    )
    rejected = _rejected_result(first, "invalid-output")
    authority.settle_physical_attempt(
        turn_id="turn_01",
        physical=first,
        outcome="retryable_failure",
        result_fingerprint=SHA_A,
        provider_result=rejected,
        error_code="MODEL_BAD_RESPONSE",
    )

    with pytest.raises(RuntimeModelCallAuthorityError, match="no durable repair"):
        _authority(ledger).recover_output_repair_feedback()


def test_repair_mode_rejects_a_provider_key_bound_to_one_request_body() -> None:
    ledger = _MemoryLedger()
    authority = _authority(
        ledger,
        provider_idempotency_key="one-body-only-key",
        repair_protocol=_FOUR_MESSAGE,
    )
    authority.reserve(turn_id="turn_01")

    with pytest.raises(RuntimeModelCallAuthorityError, match="distinct idempotency"):
        authority.begin_physical_attempt(
            turn_id="turn_01",
            max_physical_attempts=6,
            output_repair_enabled=True,
        )

    assert ledger.logical is not None
    assert ledger.logical.physical_attempts == ()
