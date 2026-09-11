from __future__ import annotations

import hashlib

import pytest

from personagraph.l2.task_graph.task_matching import InSessionTaskMatchesProposal
from personagraph.runtime.model_calls import (
    RuntimeModelLogicalRequest,
    RuntimeModelPhysicalAttemptRequest,
    RuntimeModelPhysicalAttemptSettlement,
    RuntimeModelPhysicalOutcome,
    RuntimeModelTypedResult,
)
from personagraph.model_io.output_repair_contracts import (
    RuntimeModelOutputRepairFeedback,
    RuntimeModelOutputRepairIssueCategory,
    RuntimeModelOutputRepairIssueCoverage,
    RuntimeModelOutputRepairIssue,
    RuntimeModelOutputRepairProtocol,
    RuntimeModelStructuredPrompt,
)
from personagraph.session import store
from personagraph.session.l2_store import task_graph as task_graph_store
from personagraph.session.persistence import schema


SHA_A = "a" * 64
SHA_B = "b" * 64
_USER_TEXT = "请读取材料并形成一个受来源约束的计划"


def _repair_feedback(
    response_text: str,
    *,
    code: str = "architect_guard_rejected",
    target_contract: str = "auxiliary-graph-architect-result-v1",
) -> RuntimeModelOutputRepairFeedback:
    return RuntimeModelOutputRepairFeedback(
        target_contract=target_contract,
        rejected_physical_ordinal=1,
        rejected_response_sha256=hashlib.sha256(
            response_text.encode("utf-8")
        ).hexdigest(),
        issue_coverage=RuntimeModelOutputRepairIssueCoverage.FIRST_ONLY,
        omitted_issue_count=0,
        current_issues=(
            RuntimeModelOutputRepairIssue(
                category=RuntimeModelOutputRepairIssueCategory.HOST_GUARD,
                code=code,
                paths=("",),
                safe_explanation=(
                    "The proposal violates a deterministic frozen Guard."
                ),
            ),
        ),
    )


def _seed_session_turn(*, suffix: str = "one") -> tuple[str, str]:
    session_id = store.create_session("Entelecheia")
    turn_id = f"runtime-model-turn-{suffix}"
    store.create_runtime_turn(
        turn_id=turn_id,
        session_id=session_id,
        source="runtime_model_call_persistence_test",
        user_text=_USER_TEXT,
    )
    return session_id, turn_id


def _window_revision(session_id: str) -> int:
    window = store.get_turn_execution_window(session_id)
    assert window is not None
    return int(window["state_version"])


def _seed_task(*, suffix: str = "task") -> tuple[str, str, str]:
    session_id = store.create_session("Entelecheia")
    accepted = store.accept_turn_execution(
        session_id=session_id,
        client_request_id=f"runtime-model-{suffix}",
        source="runtime_model_call_persistence_test",
        user_text=_USER_TEXT,
        lease_owner="runtime-model-call-test",
    )
    turn_id = str(accepted["turn"]["turn_id"])
    applied = task_graph_store.apply_insession_task_matches(
        session_id=session_id,
        source_turn_id=turn_id,
        apply_id=f"runtime-model-match-{suffix}",
        proposal=InSessionTaskMatchesProposal.model_validate(
            {
                "task_matches": [
                    {
                        "match_type": "new_root",
                        "local_key": "root",
                        "title": "材料规划",
                        "objective": "读取材料并形成计划",
                        "source_excerpt": _USER_TEXT,
                    }
                ]
            }
        ),
        exposed_catalog_ids=(),
        expected_window_revision=_window_revision(session_id),
    )
    return (
        session_id,
        turn_id,
        applied.created_insession_task_ids_by_local_key["root"],
    )


def _logical(
    *,
    session_id: str,
    turn_id: str,
    logical_call_id: str = "logical-runtime-01",
    max_physical_attempts: int = 3,
    **overrides: object,
) -> RuntimeModelLogicalRequest:
    values: dict[str, object] = {
        "logical_call_id": logical_call_id,
        "session_id": session_id,
        "invocation_turn_id": turn_id,
        "call_kind": "auxiliary_graph_architect",
        "purpose": "runtime_auxiliary_graph_architect",
        "provider": "mock",
        "model": "mock-model-v1",
        "endpoint_fingerprint": SHA_A,
        "request_contract": "auxiliary-graph-architect-request-v1",
        "typed_result_contract": "auxiliary-graph-architect-result-v1",
        "max_physical_attempts": max_physical_attempts,
        "state_guard_sha256": SHA_B,
    }
    values.update(overrides)
    return RuntimeModelLogicalRequest.create(
        request_payload={"goal": "bounded", "sources": ["task_creation_source"]},
        **values,
    )


def _repair_logical(**values: object) -> RuntimeModelLogicalRequest:
    return _logical(
        output_repair_protocol=(
            RuntimeModelOutputRepairProtocol.FOUR_MESSAGE_WHOLE_RESPONSE_REGENERATION
        ),
        structured_prompt=RuntimeModelStructuredPrompt.create(
            system_prompt="frozen repair test system",
            user_content='{"frozen":true}',
        ),
        **values,
    )


def _physical(
    logical: RuntimeModelLogicalRequest,
    *,
    ordinal: int = 1,
    suffix: str = "01",
    **overrides: object,
) -> RuntimeModelPhysicalAttemptRequest:
    values: dict[str, object] = {
        "physical_attempt_id": f"physical-runtime-{suffix}",
        "physical_attempt_key": f"physical-runtime-key-{suffix}",
        "logical_call_id": logical.logical_call_id,
        "logical_request_binding_sha256": logical.binding_sha256,
        "physical_ordinal": ordinal,
        "started_turn_id": logical.invocation_turn_id,
        "provider": logical.provider,
        "model": logical.model,
        "endpoint_fingerprint": logical.endpoint_fingerprint,
        "request_sha256": logical.request_sha256,
        "provider_idempotency_key": f"provider-idempotency-{suffix}",
        "dispatch_authority_sha256": SHA_A,
    }
    values.update(overrides)
    return RuntimeModelPhysicalAttemptRequest.create(**values)


def _settlement(
    physical: RuntimeModelPhysicalAttemptRequest,
    *,
    outcome: RuntimeModelPhysicalOutcome,
    suffix: str = "01",
    **overrides: object,
) -> RuntimeModelPhysicalAttemptSettlement:
    succeeded = outcome is RuntimeModelPhysicalOutcome.SUCCEEDED
    values: dict[str, object] = {
        "settlement_id": f"settlement-runtime-{suffix}",
        "settle_apply_id": f"settle-runtime-{suffix}",
        "physical_attempt_id": physical.physical_attempt_id,
        "physical_request_binding_sha256": physical.binding_sha256,
        "logical_call_id": physical.logical_call_id,
        "physical_ordinal": physical.physical_ordinal,
        "settled_turn_id": physical.started_turn_id,
        "outcome": outcome,
        "provider_request_id": f"provider-request-{suffix}",
        "finish_reason": "end_turn" if succeeded else "error",
        "error_code": None if succeeded else "provider_failure",
        "typed_result": (
            RuntimeModelTypedResult.create(
                result_contract="auxiliary-graph-architect-result-v1",
                result_payload={"proposal": "bounded"},
            )
            if succeeded
            else None
        ),
        "outcome_fingerprint": SHA_B,
    }
    values.update(overrides)
    return RuntimeModelPhysicalAttemptSettlement.create(**values)


def _reserve_and_append(
    *,
    suffix: str,
    max_physical_attempts: int = 3,
) -> tuple[
    str,
    RuntimeModelLogicalRequest,
    RuntimeModelPhysicalAttemptRequest,
]:
    session_id, turn_id = _seed_session_turn(suffix=suffix)
    logical = _logical(
        session_id=session_id,
        turn_id=turn_id,
        logical_call_id=f"logical-runtime-{suffix}",
        max_physical_attempts=max_physical_attempts,
    )
    store.reserve_runtime_model_logical_call(request=logical)
    physical = _physical(logical, suffix=suffix)
    store.append_runtime_model_physical_attempt(request=physical)
    return session_id, logical, physical


def test_current_schema_exposes_complete_runtime_model_ledger() -> None:
    store.init_db()
    with store._connect() as conn:
        assert (
            int(conn.execute("PRAGMA user_version").fetchone()[0])
            == schema.SCHEMA_VERSION
        )
        tables = {
            str(row["name"])
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert {
            "insession_runtime_model_logical_calls",
            "insession_runtime_model_physical_attempts",
            "insession_runtime_model_call_settlement_receipts",
            "insession_runtime_model_rejected_outputs",
        } <= tables
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []


def test_logical_reservation_exact_replay_and_collision_fail_closed() -> None:
    session_id, turn_id = _seed_session_turn()
    logical = _logical(session_id=session_id, turn_id=turn_id)

    applied = store.reserve_runtime_model_logical_call(request=logical)
    replayed = store.reserve_runtime_model_logical_call(request=logical)

    assert applied.status == "applied"
    assert replayed.status == "replayed"
    assert replayed.logical_call.request == logical
    assert (
        store.get_runtime_model_logical_call(
            session_id="different-session",
            logical_call_id=logical.logical_call_id,
        )
        is None
    )

    collision = _logical(
        session_id=session_id,
        turn_id=turn_id,
        purpose="runtime_auxiliary_graph_architect_changed",
    )
    with pytest.raises(store.RuntimeModelCallIdentityCollision):
        store.reserve_runtime_model_logical_call(request=collision)

    with store._connect() as conn:
        assert int(
            conn.execute(
                "SELECT COUNT(*) FROM insession_runtime_model_logical_calls"
            ).fetchone()[0]
        ) == 1


def test_store_rejects_physical_repair_mode_not_frozen_by_logical_request() -> None:
    session_id, turn_id = _seed_session_turn(suffix="repair-mode-mismatch")
    logical = _logical(
        session_id=session_id,
        turn_id=turn_id,
        logical_call_id="logical-runtime-repair-mode-mismatch",
    )
    store.reserve_runtime_model_logical_call(request=logical)

    with pytest.raises(
        store.RuntimeModelCallPersistenceError,
        match="output-repair mode differs",
    ):
        store.append_runtime_model_physical_attempt(
            request=_physical(
                logical,
                suffix="repair-mode-mismatch",
                provider_idempotency_key=None,
                output_repair_enabled=True,
            )
        )

    stored = store.get_runtime_model_logical_call(
        session_id=session_id,
        logical_call_id=logical.logical_call_id,
    )
    assert stored is not None and stored.physical_attempts == ()


def test_pending_attempt_replay_retry_and_settlement_collision() -> None:
    _, logical, first = _reserve_and_append(suffix="retry", max_physical_attempts=2)

    replayed = store.append_runtime_model_physical_attempt(request=first)
    assert replayed.status == "replayed"
    with pytest.raises(store.RuntimeModelCallTerminalState, match="reconciliation"):
        store.append_runtime_model_physical_attempt(
            request=_physical(logical, ordinal=2, suffix="retry-second")
        )

    same_id_new_key = _physical(
        logical,
        suffix="retry-collision",
        physical_attempt_id=first.physical_attempt_id,
    )
    with pytest.raises(store.RuntimeModelCallIdentityCollision):
        store.append_runtime_model_physical_attempt(request=same_id_new_key)

    settlement = _settlement(
        first,
        outcome=RuntimeModelPhysicalOutcome.RETRYABLE_FAILURE,
        suffix="retry",
    )
    applied = store.settle_runtime_model_physical_attempt(settlement=settlement)
    settled_replay = store.settle_runtime_model_physical_attempt(
        settlement=settlement
    )
    assert applied.status == "applied"
    assert settled_replay.status == "replayed"

    collision = _settlement(
        first,
        outcome=RuntimeModelPhysicalOutcome.RETRYABLE_FAILURE,
        suffix="retry-other",
        settle_apply_id=settlement.settle_apply_id,
    )
    with pytest.raises(store.RuntimeModelCallIdentityCollision):
        store.settle_runtime_model_physical_attempt(settlement=collision)

    settlement_id_collision = _settlement(
        first,
        outcome=RuntimeModelPhysicalOutcome.RETRYABLE_FAILURE,
        suffix="retry-settlement-id-collision",
        settlement_id=settlement.settlement_id,
    )
    with pytest.raises(store.RuntimeModelCallIdentityCollision):
        store.settle_runtime_model_physical_attempt(
            settlement=settlement_id_collision
        )

    second = _physical(logical, ordinal=2, suffix="retry-second")
    appended = store.append_runtime_model_physical_attempt(request=second)
    assert appended.status == "applied"
    stored = appended.logical_call
    assert [
        item.settlement.outcome if item.settlement is not None else None
        for item in stored.physical_attempts
    ] == [RuntimeModelPhysicalOutcome.RETRYABLE_FAILURE, None]


def test_store_rejects_a_repair_prompt_not_authorized_by_prior_receipt() -> None:
    session_id, turn_id = _seed_session_turn(suffix="repair-chain")
    logical = _repair_logical(
        session_id=session_id,
        turn_id=turn_id,
        logical_call_id="logical-runtime-repair-chain",
    )
    store.reserve_runtime_model_logical_call(request=logical)
    premature = _repair_feedback("rejected")
    with pytest.raises(
        store.RuntimeModelCallPersistenceError,
        match="first physical model attempt",
    ):
        store.append_runtime_model_physical_attempt(
            request=_physical(
                logical,
                suffix="repair-chain-premature",
                provider_idempotency_key=None,
                output_repair_enabled=True,
                output_repair_feedback=premature,
            )
        )
    first = _physical(
        logical,
        suffix="repair-chain-first",
        provider_idempotency_key=None,
        output_repair_enabled=True,
    )
    store.append_runtime_model_physical_attempt(request=first)
    authorized = _repair_feedback("rejected")
    store.settle_runtime_model_physical_attempt(
        settlement=_settlement(
            first,
            outcome=RuntimeModelPhysicalOutcome.RETRYABLE_FAILURE,
            suffix="repair-chain-first",
            error_code="MODEL_BAD_RESPONSE",
            next_output_repair_feedback=authorized,
        ),
        rejected_response_text="rejected",
    )
    forged = _repair_feedback("rejected", code="different_guard_reason")
    second = _physical(
        logical,
        ordinal=2,
        suffix="repair-chain-second",
        provider_idempotency_key=None,
        output_repair_enabled=True,
        output_repair_feedback=forged,
    )

    with pytest.raises(
        store.RuntimeModelCallPersistenceError,
        match="authorized output-repair feedback",
    ):
        store.append_runtime_model_physical_attempt(request=second)

    stored = store.get_runtime_model_logical_call(
        session_id=session_id,
        logical_call_id=logical.logical_call_id,
    )
    assert stored is not None
    assert len(stored.physical_attempts) == 1


def test_rejected_response_is_settled_atomically_and_loaded_by_feedback() -> None:
    session_id, turn_id = _seed_session_turn(suffix="rejected-body")
    logical = _repair_logical(
        session_id=session_id,
        turn_id=turn_id,
        logical_call_id="logical-runtime-rejected-body",
    )
    store.reserve_runtime_model_logical_call(request=logical)
    physical = _physical(
        logical,
        suffix="rejected-body",
        provider_idempotency_key=None,
        output_repair_enabled=True,
    )
    store.append_runtime_model_physical_attempt(request=physical)
    rejected_response = '{"action":"submit","unexpected":true}'
    rejected_sha256 = hashlib.sha256(
        rejected_response.encode("utf-8")
    ).hexdigest()
    feedback = RuntimeModelOutputRepairFeedback(
        target_contract="auxiliary-graph-architect-result-v1",
        rejected_physical_ordinal=1,
        rejected_response_sha256=rejected_sha256,
        issue_coverage=RuntimeModelOutputRepairIssueCoverage.COMPLETE,
        omitted_issue_count=0,
        current_issues=(
            RuntimeModelOutputRepairIssue(
                category=RuntimeModelOutputRepairIssueCategory.SCHEMA,
                code="extra_field",
                paths=("",),
                safe_explanation="输出包含合同不允许的字段。",
            ),
        ),
    )
    settlement = _settlement(
        physical,
        outcome=RuntimeModelPhysicalOutcome.RETRYABLE_FAILURE,
        suffix="rejected-body",
        error_code="MODEL_BAD_RESPONSE",
        next_output_repair_feedback=feedback,
    )

    with pytest.raises(
        store.RuntimeModelCallPersistenceError,
        match="requires the exact rejected response",
    ):
        store.settle_runtime_model_physical_attempt(settlement=settlement)

    applied = store.settle_runtime_model_physical_attempt(
        settlement=settlement,
        rejected_response_text=rejected_response,
    )
    replayed = store.settle_runtime_model_physical_attempt(
        settlement=settlement,
        rejected_response_text=rejected_response,
    )
    with pytest.raises(
        store.RuntimeModelCallPersistenceError,
        match="requires the exact rejected response",
    ):
        store.settle_runtime_model_physical_attempt(settlement=settlement)

    assert applied.status == "applied"
    assert replayed.status == "replayed"
    record = store.get_runtime_model_rejected_output(
        session_id=session_id,
        logical_call_id=logical.logical_call_id,
        rejected_physical_ordinal=feedback.rejected_physical_ordinal,
        rejected_response_sha256=feedback.rejected_response_sha256,
    )
    assert record is not None
    assert record.response_text == rejected_response
    assert record.byte_count == len(rejected_response.encode("utf-8"))
    assert record.response_sha256 == rejected_sha256
    assert len(record.record_sha256) == 64
    assert (
        store.get_runtime_model_rejected_output(
            session_id="another-session",
            logical_call_id=logical.logical_call_id,
            rejected_physical_ordinal=1,
            rejected_response_sha256=rejected_sha256,
        )
        is None
    )
    assert store.purge_session(session_id) is True
    with store._connect() as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM insession_runtime_model_rejected_outputs"
        ).fetchone()[0] == 0
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []


def test_store_rejects_feedback_for_another_logical_result_contract() -> None:
    session_id, turn_id = _seed_session_turn(suffix="repair-target-mismatch")
    logical = _repair_logical(
        session_id=session_id,
        turn_id=turn_id,
        logical_call_id="logical-runtime-repair-target-mismatch",
    )
    store.reserve_runtime_model_logical_call(request=logical)
    physical = _physical(
        logical,
        suffix="repair-target-mismatch",
        provider_idempotency_key=None,
        output_repair_enabled=True,
    )
    store.append_runtime_model_physical_attempt(request=physical)
    rejected_response = '{"proposal":"invalid"}'
    feedback = _repair_feedback(
        rejected_response,
        target_contract="another-result-contract",
    )
    settlement = _settlement(
        physical,
        outcome=RuntimeModelPhysicalOutcome.RETRYABLE_FAILURE,
        suffix="repair-target-mismatch",
        error_code="MODEL_BAD_RESPONSE",
        next_output_repair_feedback=feedback,
    )

    with pytest.raises(
        store.RuntimeModelCallPersistenceError,
        match="repair target differs from its logical result contract",
    ):
        store.settle_runtime_model_physical_attempt(
            settlement=settlement,
            rejected_response_text=rejected_response,
        )

    stored = store.get_runtime_model_logical_call(
        session_id=session_id,
        logical_call_id=logical.logical_call_id,
    )
    assert stored is not None
    assert stored.physical_attempts[0].settlement is None


def test_rejected_response_hash_mismatch_rolls_back_whole_settlement() -> None:
    session_id, turn_id = _seed_session_turn(suffix="rejected-mismatch")
    logical = _repair_logical(
        session_id=session_id,
        turn_id=turn_id,
        logical_call_id="logical-runtime-rejected-mismatch",
    )
    store.reserve_runtime_model_logical_call(request=logical)
    physical = _physical(
        logical,
        suffix="rejected-mismatch",
        provider_idempotency_key=None,
        output_repair_enabled=True,
    )
    store.append_runtime_model_physical_attempt(request=physical)
    expected_response = '{"action":"expected"}'
    feedback = RuntimeModelOutputRepairFeedback(
        target_contract=logical.typed_result_contract,
        rejected_physical_ordinal=1,
        rejected_response_sha256=hashlib.sha256(
            expected_response.encode("utf-8")
        ).hexdigest(),
        issue_coverage=RuntimeModelOutputRepairIssueCoverage.COMPLETE,
        omitted_issue_count=0,
        current_issues=(
            RuntimeModelOutputRepairIssue(
                category=RuntimeModelOutputRepairIssueCategory.SCHEMA,
                code="schema_rejected",
                paths=("",),
                safe_explanation="输出没有满足目标合同。",
            ),
        ),
    )
    settlement = _settlement(
        physical,
        outcome=RuntimeModelPhysicalOutcome.RETRYABLE_FAILURE,
        suffix="rejected-mismatch",
        error_code="MODEL_BAD_RESPONSE",
        next_output_repair_feedback=feedback,
    )

    with pytest.raises(
        store.RuntimeModelCallPersistenceError,
        match="hash differs",
    ):
        store.settle_runtime_model_physical_attempt(
            settlement=settlement,
            rejected_response_text='{"action":"different"}',
        )

    stored = store.get_runtime_model_logical_call(
        session_id=session_id,
        logical_call_id=logical.logical_call_id,
    )
    assert stored is not None
    assert stored.physical_attempts[0].settlement is None
    with store._connect() as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM "
            "insession_runtime_model_call_settlement_receipts"
        ).fetchone()[0] == 0
        assert conn.execute(
            "SELECT COUNT(*) FROM insession_runtime_model_rejected_outputs"
        ).fetchone()[0] == 0


def test_non_rejection_settlement_cannot_store_a_rejected_response() -> None:
    session_id, logical, physical = _reserve_and_append(
        suffix="unexpected-rejected-body"
    )
    settlement = _settlement(
        physical,
        outcome=RuntimeModelPhysicalOutcome.SUCCEEDED,
        suffix="unexpected-rejected-body",
    )

    with pytest.raises(
        store.RuntimeModelCallPersistenceError,
        match="requires output-repair feedback",
    ):
        store.settle_runtime_model_physical_attempt(
            settlement=settlement,
            rejected_response_text='{"valid":true}',
        )

    stored = store.get_runtime_model_logical_call(
        session_id=session_id,
        logical_call_id=logical.logical_call_id,
    )
    assert stored is not None
    assert stored.physical_attempts[0].settlement is None


def test_rejected_response_tamper_fails_closed_on_recovery() -> None:
    session_id, turn_id = _seed_session_turn(suffix="rejected-tamper")
    logical = _repair_logical(
        session_id=session_id,
        turn_id=turn_id,
        logical_call_id="logical-runtime-rejected-tamper",
    )
    store.reserve_runtime_model_logical_call(request=logical)
    physical = _physical(
        logical,
        suffix="rejected-tamper",
        provider_idempotency_key=None,
        output_repair_enabled=True,
    )
    store.append_runtime_model_physical_attempt(request=physical)
    response = '{"bad":true}'
    response_sha256 = hashlib.sha256(response.encode("utf-8")).hexdigest()
    feedback = RuntimeModelOutputRepairFeedback(
        target_contract=logical.typed_result_contract,
        rejected_physical_ordinal=1,
        rejected_response_sha256=response_sha256,
        issue_coverage=RuntimeModelOutputRepairIssueCoverage.COMPLETE,
        omitted_issue_count=0,
        current_issues=(
            RuntimeModelOutputRepairIssue(
                category=RuntimeModelOutputRepairIssueCategory.SCHEMA,
                code="schema_rejected",
                paths=("",),
                safe_explanation="输出没有满足目标合同。",
            ),
        ),
    )
    store.settle_runtime_model_physical_attempt(
        settlement=_settlement(
            physical,
            outcome=RuntimeModelPhysicalOutcome.RETRYABLE_FAILURE,
            suffix="rejected-tamper",
            error_code="MODEL_BAD_RESPONSE",
            next_output_repair_feedback=feedback,
        ),
        rejected_response_text=response,
    )
    with store._connect() as conn:
        conn.execute(
            "UPDATE insession_runtime_model_rejected_outputs "
            "SET response_text=? WHERE physical_attempt_id=?",
            ('{"forged":true}', physical.physical_attempt_id),
        )

    with pytest.raises(
        store.RuntimeModelCallPersistenceError,
        match="rejected model output is corrupt",
    ):
        store.get_runtime_model_rejected_output(
            session_id=session_id,
            logical_call_id=logical.logical_call_id,
            rejected_physical_ordinal=1,
            rejected_response_sha256=response_sha256,
        )


def test_store_rejects_explicit_nonrepair_mode_upgrade_after_bad_response() -> None:
    session_id, turn_id = _seed_session_turn(suffix="explicit-repair-upgrade")
    logical = _logical(
        session_id=session_id,
        turn_id=turn_id,
        logical_call_id="logical-runtime-explicit-repair-upgrade",
    )
    store.reserve_runtime_model_logical_call(request=logical)
    first = _physical(
        logical,
        suffix="explicit-repair-upgrade-first",
        provider_idempotency_key=None,
    )
    assert first.output_repair_enabled is False
    store.append_runtime_model_physical_attempt(request=first)
    store.settle_runtime_model_physical_attempt(
        settlement=_settlement(
            first,
            outcome=RuntimeModelPhysicalOutcome.RETRYABLE_FAILURE,
            suffix="explicit-repair-upgrade-first",
            error_code="MODEL_BAD_RESPONSE",
        )
    )
    upgraded = _physical(
        logical,
        ordinal=2,
        suffix="explicit-repair-upgrade-second",
        provider_idempotency_key=None,
        output_repair_enabled=True,
    )

    with pytest.raises(
        store.RuntimeModelCallPersistenceError,
        match="differs from its logical request authority",
    ):
        store.append_runtime_model_physical_attempt(request=upgraded)


@pytest.mark.parametrize(
    "outcome",
    [
        RuntimeModelPhysicalOutcome.SUCCEEDED,
        RuntimeModelPhysicalOutcome.TERMINAL_FAILURE,
        RuntimeModelPhysicalOutcome.UNCERTAIN,
    ],
)
def test_success_terminal_and_uncertain_outcomes_never_grant_retry(
    outcome: RuntimeModelPhysicalOutcome,
) -> None:
    suffix = outcome.value
    _, logical, physical = _reserve_and_append(suffix=suffix)
    settlement = _settlement(physical, outcome=outcome, suffix=suffix)
    store.settle_runtime_model_physical_attempt(settlement=settlement)

    with pytest.raises(store.RuntimeModelCallTerminalState, match="terminal"):
        store.append_runtime_model_physical_attempt(
            request=_physical(logical, ordinal=2, suffix=f"{suffix}-second")
        )


def test_retryable_outcome_cannot_exceed_logical_attempt_limit() -> None:
    _, logical, first = _reserve_and_append(suffix="limit", max_physical_attempts=2)
    store.settle_runtime_model_physical_attempt(
        settlement=_settlement(
            first,
            outcome=RuntimeModelPhysicalOutcome.RETRYABLE_FAILURE,
            suffix="limit-first",
        )
    )
    second = _physical(logical, ordinal=2, suffix="limit-second")
    store.append_runtime_model_physical_attempt(request=second)
    store.settle_runtime_model_physical_attempt(
        settlement=_settlement(
            second,
            outcome=RuntimeModelPhysicalOutcome.RETRYABLE_FAILURE,
            suffix="limit-second",
        )
    )

    with pytest.raises(store.RuntimeModelCallTerminalState, match="limit"):
        store.append_runtime_model_physical_attempt(
            request=_physical(logical, ordinal=3, suffix="limit-third")
        )


@pytest.mark.parametrize(
    "tamper_sql,tamper_parameters",
    [
        (
            "UPDATE insession_runtime_model_logical_calls SET purpose=? "
            "WHERE logical_call_id=?",
            ("forged-purpose", "logical-runtime-tamper"),
        ),
        (
            "UPDATE insession_runtime_model_physical_attempts "
            "SET dispatch_authority_sha256=? WHERE physical_attempt_id=?",
            ("0" * 64, "physical-runtime-tamper"),
        ),
        (
            "UPDATE insession_runtime_model_call_settlement_receipts "
            "SET receipt_sha256=? WHERE physical_attempt_id=?",
            ("0" * 64, "physical-runtime-tamper"),
        ),
    ],
)
def test_indexed_projection_tamper_fails_closed(
    tamper_sql: str,
    tamper_parameters: tuple[str, str],
) -> None:
    session_id, _, physical = _reserve_and_append(suffix="tamper")
    store.settle_runtime_model_physical_attempt(
        settlement=_settlement(
            physical,
            outcome=RuntimeModelPhysicalOutcome.SUCCEEDED,
            suffix="tamper",
        )
    )
    with store._connect() as conn:
        conn.execute(tamper_sql, tamper_parameters)

    with pytest.raises(store.RuntimeModelCallPersistenceError, match="corrupt"):
        store.get_runtime_model_logical_call(
            session_id=session_id,
            logical_call_id="logical-runtime-tamper",
        )


def test_reference_constraints_reject_cross_owner_without_partial_write() -> None:
    session_id, turn_id, task_id = _seed_task(suffix="references-one")
    other_session_id, other_turn_id, other_task_id = _seed_task(
        suffix="references-two"
    )

    valid_task_request = _logical(
        session_id=session_id,
        turn_id=turn_id,
        logical_call_id="logical-valid-task",
        task_id=task_id,
    )
    assert (
        store.reserve_runtime_model_logical_call(request=valid_task_request).status
        == "applied"
    )

    invalid_requests = (
        _logical(
            session_id=session_id,
            turn_id=other_turn_id,
            logical_call_id="logical-cross-turn",
        ),
        _logical(
            session_id=session_id,
            turn_id=turn_id,
            logical_call_id="logical-cross-task",
            task_id=other_task_id,
        ),
        _logical(
            session_id=session_id,
            turn_id=turn_id,
            logical_call_id="logical-missing-goal",
            task_id=task_id,
            auxiliary_graph_id="missing-graph",
            goal_id="missing-goal",
        ),
        _logical(
            session_id=session_id,
            turn_id=turn_id,
            logical_call_id="logical-missing-subject",
            task_id=task_id,
            execution_subject_id="missing-subject",
        ),
    )
    for request in invalid_requests:
        with pytest.raises(store.RuntimeModelCallPersistenceError):
            store.reserve_runtime_model_logical_call(request=request)

    with store._connect() as conn:
        rows = conn.execute(
            "SELECT logical_call_id FROM insession_runtime_model_logical_calls "
            "ORDER BY logical_call_id"
        ).fetchall()
        assert [str(row["logical_call_id"]) for row in rows] == [
            "logical-valid-task"
        ]
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
    assert other_session_id != session_id


def test_physical_and_settlement_reference_failures_are_atomic() -> None:
    session_id, turn_id = _seed_session_turn(suffix="atomic-owner")
    _, foreign_turn_id = _seed_session_turn(suffix="atomic-foreign")
    logical = _logical(
        session_id=session_id,
        turn_id=turn_id,
        logical_call_id="logical-runtime-atomic",
    )
    store.reserve_runtime_model_logical_call(request=logical)

    invalid_physical = _physical(
        logical,
        suffix="atomic-invalid",
        started_turn_id=foreign_turn_id,
    )
    with pytest.raises(store.RuntimeModelCallPersistenceError):
        store.append_runtime_model_physical_attempt(request=invalid_physical)
    stored = store.get_runtime_model_logical_call(
        session_id=session_id,
        logical_call_id=logical.logical_call_id,
    )
    assert stored is not None and stored.physical_attempts == ()

    physical = _physical(logical, suffix="atomic-valid")
    store.append_runtime_model_physical_attempt(request=physical)
    invalid_settlement = _settlement(
        physical,
        outcome=RuntimeModelPhysicalOutcome.SUCCEEDED,
        suffix="atomic-invalid",
        settled_turn_id=foreign_turn_id,
    )
    with pytest.raises(store.RuntimeModelCallPersistenceError):
        store.settle_runtime_model_physical_attempt(settlement=invalid_settlement)

    stored = store.get_runtime_model_logical_call(
        session_id=session_id,
        logical_call_id=logical.logical_call_id,
    )
    assert stored is not None
    assert len(stored.physical_attempts) == 1
    assert stored.physical_attempts[0].settlement is None
    with store._connect() as conn:
        assert int(
            conn.execute(
                "SELECT COUNT(*) FROM "
                "insession_runtime_model_call_settlement_receipts"
            ).fetchone()[0]
        ) == 0


def test_session_purge_cascades_complete_runtime_model_ledger() -> None:
    session_id, _, physical = _reserve_and_append(suffix="purge")
    store.settle_runtime_model_physical_attempt(
        settlement=_settlement(
            physical,
            outcome=RuntimeModelPhysicalOutcome.SUCCEEDED,
            suffix="purge",
        )
    )

    assert store.purge_session(session_id) is True
    with store._connect() as conn:
        for table in (
            "insession_runtime_model_rejected_outputs",
            "insession_runtime_model_call_settlement_receipts",
            "insession_runtime_model_physical_attempts",
            "insession_runtime_model_logical_calls",
        ):
            assert int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]) == 0
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
