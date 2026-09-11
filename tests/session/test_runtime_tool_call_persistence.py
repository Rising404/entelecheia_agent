from __future__ import annotations

from dataclasses import dataclass
import hashlib
import inspect
import json
import sqlite3
from typing import get_type_hints

import pytest

from personagraph.l2.task_graph.lane_manifest import (
    InSessionTaskExecutionLaneManifest,
    InSessionTaskExecutionLaneMatch,
    InSessionTaskExecutionLane,
    canonical_insession_task_lane_manifest_sha256,
)
from personagraph.l2.task_graph.task_matching import (
    InSessionTaskMatchedSourceSpan,
)
from personagraph.runtime.tool_calls import (
    RuntimeToolEffectClass,
    RuntimeToolLedgerStore,
    RuntimeToolLogicalRequest,
    RuntimeToolPhysicalAttemptRequest,
    RuntimeToolPhysicalAttemptSettlement,
    RuntimeToolPhysicalOutcome,
    RuntimeToolRetryAuthority,
    RuntimeToolTypedResult,
    RuntimeLogicalToolCallAuthority,
    RuntimeToolDispatchObservation,
)
from personagraph.session import store
from personagraph.session.l2_store import work_run as work_run_store
from personagraph.session.persistence import schema
from personagraph.session.runtime_call_ledger_store_facade import (
    RuntimeCallLedgerStoreFacade,
)
from personagraph.l2.work_run import (
    HostAcceptedAttemptDecision,
    HostMaterializedCallToolsAction,
    HostMaterializedToolCall,
    TaskNodeSubject,
)


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
_USER_TEXT = "执行这个已经物化的工具调用"
_ARGUMENTS = {"value": "x"}


def test_session_ledger_facade_matches_runtime_tool_store_port() -> None:
    expected = {
        "reserve_runtime_tool_logical_call": (
            {"request": RuntimeToolLogicalRequest},
            object,
        ),
        "append_runtime_tool_physical_attempt": (
            {"request": RuntimeToolPhysicalAttemptRequest},
            object,
        ),
        "settle_runtime_tool_physical_attempt": (
            {"settlement": RuntimeToolPhysicalAttemptSettlement},
            object,
        ),
        "get_runtime_tool_logical_call": (
            {"session_id": str, "logical_tool_call_id": str},
            object | None,
        ),
    }
    for method_name, (argument_types, return_type) in expected.items():
        port_method = getattr(RuntimeToolLedgerStore, method_name)
        facade_method = getattr(RuntimeCallLedgerStoreFacade, method_name)
        port_signature = inspect.signature(port_method)
        facade_signature = inspect.signature(facade_method)
        assert tuple(facade_signature.parameters) == tuple(port_signature.parameters)

        port_hints = get_type_hints(port_method)
        facade_hints = get_type_hints(facade_method)
        for argument_name, expected_type in argument_types.items():
            port_parameter = port_signature.parameters[argument_name]
            facade_parameter = facade_signature.parameters[argument_name]
            assert port_parameter.kind is inspect.Parameter.KEYWORD_ONLY
            assert facade_parameter.kind is port_parameter.kind
            assert facade_parameter.default == port_parameter.default
            assert port_hints[argument_name] == expected_type
            assert facade_hints[argument_name] == expected_type
        assert port_hints["return"] == return_type


@dataclass(frozen=True)
class _MaterializedCall:
    session_id: str
    turn_id: str
    task_id: str
    work_run_id: str
    attempt_id: str
    tool_call_id: str
    catalog_snapshot_sha256: str
    modifies_environment: bool


def _window_revision(session_id: str) -> int:
    window = store.get_turn_execution_window(session_id)
    assert window is not None
    return int(window["state_version"])


def _seed_materialized_call(
    *,
    suffix: str,
    modifies_environment: bool = False,
) -> _MaterializedCall:
    session_id = store.create_session("Entelecheia")
    accepted = store.accept_turn_execution(
        session_id=session_id,
        client_request_id=f"runtime-tool-{suffix}",
        source="runtime_tool_call_persistence_test",
        user_text=_USER_TEXT,
        lease_owner="runtime-tool-call-test",
    )
    turn_id = str(accepted["turn"]["turn_id"])
    task_id = f"runtime-tool-task-{suffix}"
    node_id = f"runtime-tool-node-{suffix}"
    work_run_id = f"runtime-tool-run-{suffix}"
    attempt_id = f"runtime-tool-attempt-{suffix}"
    tool_call_id = f"runtime-tool-call-{suffix}"
    now = "2026-08-22T00:00:00+00:00"
    acceptances = json.dumps(
        [
            {
                "acceptance_id": "deliverable",
                "criterion": "完成工具读取",
                "source_anchor_ids": ["request"],
            }
        ],
        ensure_ascii=False,
    )
    lane = InSessionTaskExecutionLane(
        ordinal=0,
        insession_task_id=task_id,
        matches=(
            InSessionTaskExecutionLaneMatch(
                match_type="existing_root",
                source_span=InSessionTaskMatchedSourceSpan(
                    start=0,
                    end=len(_USER_TEXT),
                    text_sha256=hashlib.sha256(
                        _USER_TEXT.encode("utf-8")
                    ).hexdigest(),
                ),
                execution_requested=True,
            ),
        ),
        execution_requested=True,
    )
    manifest = InSessionTaskExecutionLaneManifest(
        lanes=(lane,),
        manifest_sha256=canonical_insession_task_lane_manifest_sha256((lane,)),
    )
    with store._connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            "INSERT INTO insession_tasks "
            "(insession_task_id, session_id, current_graph_revision, "
            "current_status, state_version, root_title, root_objective, "
            "created_turn_id, created_at, updated_at) "
            "VALUES (?, ?, 1, 'proposed', 1, '工具测试', '完成工具测试', ?, ?, ?)",
            (task_id, session_id, turn_id, now, now),
        )
        conn.execute(
            "INSERT INTO insession_task_graph_revisions "
            "(insession_task_id, graph_revision, source_turn_id, proposal_hash, "
            "source_anchors_json, authorization_anchor_ids_json, "
            "required_anchor_ids_json, created_at) "
            "VALUES (?, 1, ?, ?, '[]', '[]', '[]', ?)",
            (task_id, turn_id, f"proposal-{suffix}", now),
        )
        conn.execute(
            "INSERT INTO insession_task_graph_nodes "
            "(insession_task_id, graph_revision, insession_task_node_id, "
            "node_revision, node_kind, ordinal, title, objective, "
            "source_anchor_ids_json, acceptance_criteria_json, "
            "constraints_json, created_at) "
            "VALUES (?, 1, ?, 1, 'root', 0, '工具节点', '执行工具', "
            "'[\"request\"]', ?, '[]', ?)",
            (task_id, node_id, acceptances, now),
        )
        conn.execute(
            "INSERT INTO insession_task_node_states "
            "(insession_task_id, insession_task_node_id, node_revision, status, "
            "state_version, updated_at) VALUES (?, ?, 1, 'proposed', 1, ?)",
            (task_id, node_id, now),
        )
        conn.execute(
            "INSERT INTO insession_task_turn_links "
            "(session_id, turn_id, insession_task_id, insession_task_node_id, "
            "relation, created_at) VALUES (?, ?, ?, NULL, 'referenced', ?)",
            (session_id, turn_id, task_id, now),
        )
        conn.execute(
            "INSERT INTO insession_task_match_apply_receipts "
            "(apply_id, session_id, source_turn_id, proposal_hash, "
            "created_task_mapping_json, related_insession_task_ids_json, "
            "branch_intent_ids_json, turn_task_link_revision, "
            "window_state_version, execution_lane_manifest_json, "
            "execution_lane_manifest_hash, created_at) "
            "VALUES (?, ?, ?, ?, '{}', ?, '[]', 0, ?, ?, ?, ?)",
            (
                f"runtime-tool-lane-{suffix}",
                session_id,
                turn_id,
                f"lane-{suffix}",
                json.dumps([task_id]),
                _window_revision(session_id),
                manifest.model_dump_json(),
                manifest.manifest_sha256,
                now,
            ),
        )
    subject = TaskNodeSubject(
        task_id=task_id,
        graph_revision=1,
        node_id=node_id,
        node_revision=1,
    )
    created = work_run_store.create_task_node_work_run(
        session_id=session_id,
        turn_id=turn_id,
        subject=subject,
        expected_task_state_version=1,
        expected_node_state_version=1,
        expected_window_revision=_window_revision(session_id),
        apply_id=f"runtime-tool-create-{suffix}",
        work_run_id=work_run_id,
    )
    started = work_run_store.start_work_run_attempt(
        session_id=session_id,
        turn_id=turn_id,
        work_run_id=work_run_id,
        expected_work_run_revision=created.work_run_revision,
        expected_progress_revision=created.acceptance_progress_revision,
        expected_window_revision=created.window_state_version,
        apply_id=f"runtime-tool-start-{suffix}",
        catalog_snapshot={"revision": 1, "tools": ["read_test"]},
        attempt_id=attempt_id,
    )
    work_run_store.commit_work_run_attempt_decision(
        session_id=session_id,
        turn_id=turn_id,
        work_run_id=work_run_id,
        attempt_id=attempt_id,
        decision=HostAcceptedAttemptDecision(
            action=HostMaterializedCallToolsAction(
                calls=(
                    HostMaterializedToolCall(
                        tool_call_id=tool_call_id,
                        tool_id="read_test",
                        tool_version="1.0.0",
                        arguments=_ARGUMENTS,
                        modifies_environment=modifies_environment,
                    ),
                )
            )
        ),
        expected_work_run_revision=started.work_run_revision,
        expected_progress_revision=started.acceptance_progress_revision,
        expected_window_revision=started.window_state_version,
        apply_id=f"runtime-tool-decide-{suffix}",
    )
    with store._connect() as conn:
        catalog_hash = str(
            conn.execute(
                "SELECT catalog_snapshot_hash FROM insession_work_run_attempts "
                "WHERE work_run_id=? AND attempt_id=?",
                (work_run_id, attempt_id),
            ).fetchone()["catalog_snapshot_hash"]
        )
    return _MaterializedCall(
        session_id=session_id,
        turn_id=turn_id,
        task_id=task_id,
        work_run_id=work_run_id,
        attempt_id=attempt_id,
        tool_call_id=tool_call_id,
        catalog_snapshot_sha256=catalog_hash,
        modifies_environment=modifies_environment,
    )


def _logical(
    seed: _MaterializedCall,
    *,
    retry_authority: RuntimeToolRetryAuthority | None = None,
    max_physical_attempts: int = 3,
    **overrides: object,
) -> RuntimeToolLogicalRequest:
    effect_class = (
        RuntimeToolEffectClass.PROTECTED_EFFECT
        if seed.modifies_environment
        else RuntimeToolEffectClass.READ_ONLY
    )
    retry = retry_authority or (
        RuntimeToolRetryAuthority.PROVIDER_IDEMPOTENCY
        if seed.modifies_environment
        else RuntimeToolRetryAuthority.READ_ONLY_REPLAY
    )
    values: dict[str, object] = {
        "logical_tool_call_id": seed.tool_call_id,
        "session_id": seed.session_id,
        "work_run_id": seed.work_run_id,
        "attempt_id": seed.attempt_id,
        "call_ordinal": 1,
        "invocation_turn_id": seed.turn_id,
        "catalog_snapshot_sha256": seed.catalog_snapshot_sha256,
        "tool_id": "read_test",
        "contract_version": "read-test-contract-v1",
        "implementation_version": "1.0.0",
        "provider_identity_sha256": SHA_C,
        "effect_profile_sha256": SHA_B,
        "effect_class": effect_class,
        "retry_authority": retry,
        "result_contract": "read-test-result-v1",
        "max_physical_attempts": max_physical_attempts,
        "state_guard_sha256": SHA_A,
    }
    values.update(overrides)
    return RuntimeToolLogicalRequest.create(
        arguments=_ARGUMENTS,
        **values,
    )


def _physical(
    logical: RuntimeToolLogicalRequest,
    *,
    ordinal: int = 1,
    suffix: str = "one",
    **overrides: object,
) -> RuntimeToolPhysicalAttemptRequest:
    provider_key = (
        f"provider-idempotency-{logical.logical_tool_call_id}"
        if logical.retry_authority
        is RuntimeToolRetryAuthority.PROVIDER_IDEMPOTENCY
        else None
    )
    values: dict[str, object] = {
        "physical_attempt_id": f"runtime-tool-physical-{suffix}",
        "physical_attempt_key": f"runtime-tool-physical-key-{suffix}",
        "logical_tool_call_id": logical.logical_tool_call_id,
        "logical_request_binding_sha256": logical.binding_sha256,
        "physical_ordinal": ordinal,
        "started_turn_id": logical.invocation_turn_id,
        "retry_authority": logical.retry_authority,
        "provider_idempotency_key": provider_key,
        "dispatch_authority_sha256": SHA_B,
    }
    values.update(overrides)
    return RuntimeToolPhysicalAttemptRequest.create(**values)


def _settlement(
    physical: RuntimeToolPhysicalAttemptRequest,
    *,
    outcome: RuntimeToolPhysicalOutcome,
    suffix: str = "one",
    result_contract: str = "read-test-result-v1",
    **overrides: object,
) -> RuntimeToolPhysicalAttemptSettlement:
    succeeded = outcome is RuntimeToolPhysicalOutcome.SUCCEEDED
    values: dict[str, object] = {
        "settlement_id": f"runtime-tool-settlement-{suffix}",
        "settle_apply_id": f"runtime-tool-settle-{suffix}",
        "physical_attempt_id": physical.physical_attempt_id,
        "physical_request_binding_sha256": physical.binding_sha256,
        "logical_tool_call_id": physical.logical_tool_call_id,
        "physical_ordinal": physical.physical_ordinal,
        "settled_turn_id": physical.started_turn_id,
        "outcome": outcome,
        "provider_request_id": f"provider-request-{suffix}",
        "error_code": None if succeeded else "controlled_tool_failure",
        "duration_ms": 7,
        "typed_result": (
            RuntimeToolTypedResult.create(
                result_contract=result_contract,
                result={"value": "result"},
            )
            if succeeded
            else None
        ),
        "outcome_fingerprint": SHA_C,
    }
    values.update(overrides)
    return RuntimeToolPhysicalAttemptSettlement.create(**values)


def _reserve_and_append(
    *,
    suffix: str,
    modifies_environment: bool = False,
    retry_authority: RuntimeToolRetryAuthority | None = None,
    max_physical_attempts: int = 3,
) -> tuple[
    _MaterializedCall,
    RuntimeToolLogicalRequest,
    RuntimeToolPhysicalAttemptRequest,
]:
    seed = _seed_materialized_call(
        suffix=suffix,
        modifies_environment=modifies_environment,
    )
    logical = _logical(
        seed,
        retry_authority=retry_authority,
        max_physical_attempts=max_physical_attempts,
    )
    store.reserve_runtime_tool_logical_call(request=logical)
    physical = _physical(logical, suffix=suffix)
    store.append_runtime_tool_physical_attempt(request=physical)
    return seed, logical, physical


def test_current_schema_exposes_complete_runtime_tool_ledger() -> None:
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
            "insession_runtime_tool_logical_calls",
            "insession_runtime_tool_physical_attempts",
            "insession_runtime_tool_call_settlement_receipts",
        } <= tables
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []


def test_logical_reservation_exact_replay_collision_and_owner_binding() -> None:
    seed = _seed_materialized_call(suffix="logical")
    logical = _logical(seed)

    applied = store.reserve_runtime_tool_logical_call(request=logical)
    replayed = store.reserve_runtime_tool_logical_call(request=logical)

    assert applied.status == "applied"
    assert replayed.status == "replayed"
    assert replayed.logical_call.request == logical
    assert (
        store.get_runtime_tool_logical_call(
            session_id="different-session",
            logical_tool_call_id=logical.logical_tool_call_id,
        )
        is None
    )

    collision = _logical(seed, provider_identity_sha256=SHA_A)
    with pytest.raises(store.RuntimeToolCallIdentityCollision):
        store.reserve_runtime_tool_logical_call(request=collision)

    missing_call = logical.model_copy(
        update={"logical_tool_call_id": "missing-materialized-call"}
    )
    with pytest.raises(store.RuntimeToolCallPersistenceError):
        store.reserve_runtime_tool_logical_call(request=missing_call)
    with store._connect() as conn:
        assert int(
            conn.execute(
                "SELECT COUNT(*) FROM insession_runtime_tool_logical_calls"
            ).fetchone()[0]
        ) == 1


def test_runtime_authority_uses_injected_store_for_exact_success_replay() -> None:
    seed = _seed_materialized_call(suffix="authority-integration")
    logical = _logical(seed)
    authority = RuntimeLogicalToolCallAuthority(
        logical_request=logical,
        state_guard_sha256=lambda: SHA_A,
        store=store,
    )

    authority.reserve(turn_id=seed.turn_id)
    physical = authority.begin_physical_attempt(
        turn_id=seed.turn_id,
        max_physical_attempts=logical.max_physical_attempts,
    )
    observation = RuntimeToolDispatchObservation.create(
        session_id=seed.session_id,
        work_run_id=seed.work_run_id,
        attempt_id=seed.attempt_id,
        logical_tool_call_id=seed.tool_call_id,
        logical_request_binding_sha256=logical.binding_sha256,
        physical_attempt_id=physical.physical_attempt_id,
        physical_request_binding_sha256=physical.binding_sha256,
        physical_ordinal=physical.physical_ordinal,
        provider_identity_sha256=logical.provider_identity_sha256,
        tool_id=logical.tool_id,
        contract_version=logical.contract_version,
        implementation_version=logical.implementation_version,
        outcome=RuntimeToolPhysicalOutcome.SUCCEEDED,
        provider_request_id="authority-provider-request",
        duration_ms=3,
        result={"value": "durable"},
    )
    authority.settle_physical_attempt(
        turn_id=seed.turn_id,
        physical=physical,
        observation=observation,
    )

    replay = authority.replay_succeeded_result()
    assert replay is not None
    assert replay.physical_attempt_id == physical.physical_attempt_id
    assert replay.value == {"value": "durable"}


def test_pending_retryable_and_read_only_replay_semantics() -> None:
    _, logical, first = _reserve_and_append(suffix="readonly-retry")
    assert store.append_runtime_tool_physical_attempt(request=first).status == "replayed"
    crossed_identity = _physical(
        logical,
        suffix="readonly-retry-crossed",
        physical_attempt_id=first.physical_attempt_id,
    )
    with pytest.raises(store.RuntimeToolCallIdentityCollision):
        store.append_runtime_tool_physical_attempt(request=crossed_identity)
    with pytest.raises(store.RuntimeToolCallWaitingExternalState, match="pending"):
        store.append_runtime_tool_physical_attempt(
            request=_physical(logical, ordinal=2, suffix="readonly-retry-second")
        )

    store.settle_runtime_tool_physical_attempt(
        settlement=_settlement(
            first,
            outcome=RuntimeToolPhysicalOutcome.RETRYABLE_FAILURE,
            suffix="readonly-retry-first",
        )
    )
    second = _physical(logical, ordinal=2, suffix="readonly-retry-second")
    assert store.append_runtime_tool_physical_attempt(request=second).status == "applied"


def test_provider_idempotency_key_is_stable_and_reconciliation_has_no_retry() -> None:
    _, logical, first = _reserve_and_append(
        suffix="provider-idempotency",
        modifies_environment=True,
    )
    store.settle_runtime_tool_physical_attempt(
        settlement=_settlement(
            first,
            outcome=RuntimeToolPhysicalOutcome.RETRYABLE_FAILURE,
            suffix="provider-idempotency-first",
        )
    )
    changed_key = _physical(
        logical,
        ordinal=2,
        suffix="provider-idempotency-second",
        provider_idempotency_key="different-provider-key",
    )
    with pytest.raises(store.RuntimeToolCallPersistenceError, match="exact frozen key"):
        store.append_runtime_tool_physical_attempt(request=changed_key)
    stable = _physical(logical, ordinal=2, suffix="provider-idempotency-second")
    store.append_runtime_tool_physical_attempt(request=stable)

    seed = _seed_materialized_call(
        suffix="reconciliation",
        modifies_environment=True,
    )
    no_retry = _logical(
        seed,
        retry_authority=RuntimeToolRetryAuthority.RECONCILIATION_REQUIRED,
    )
    store.reserve_runtime_tool_logical_call(request=no_retry)
    physical = _physical(no_retry, suffix="reconciliation")
    store.append_runtime_tool_physical_attempt(request=physical)
    store.settle_runtime_tool_physical_attempt(
        settlement=_settlement(
            physical,
            outcome=RuntimeToolPhysicalOutcome.RETRYABLE_FAILURE,
            suffix="reconciliation",
        )
    )
    with pytest.raises(store.RuntimeToolCallTerminalState, match="no frozen retry"):
        store.append_runtime_tool_physical_attempt(
            request=_physical(no_retry, ordinal=2, suffix="reconciliation-second")
        )


@pytest.mark.parametrize(
    "outcome",
    [
        RuntimeToolPhysicalOutcome.SUCCEEDED,
        RuntimeToolPhysicalOutcome.TERMINAL_FAILURE,
    ],
)
def test_success_and_terminal_failure_never_grant_retry(
    outcome: RuntimeToolPhysicalOutcome,
) -> None:
    suffix = f"terminal-{outcome.value}"
    _, logical, physical = _reserve_and_append(suffix=suffix)
    store.settle_runtime_tool_physical_attempt(
        settlement=_settlement(physical, outcome=outcome, suffix=suffix)
    )
    with pytest.raises(store.RuntimeToolCallTerminalState, match="terminal"):
        store.append_runtime_tool_physical_attempt(
            request=_physical(logical, ordinal=2, suffix=f"{suffix}-second")
        )


def test_uncertain_outcome_waits_external_and_never_grants_retry() -> None:
    _, logical, physical = _reserve_and_append(
        suffix="uncertain",
        modifies_environment=True,
    )
    store.settle_runtime_tool_physical_attempt(
        settlement=_settlement(
            physical,
            outcome=RuntimeToolPhysicalOutcome.UNCERTAIN,
            suffix="uncertain",
        )
    )
    with pytest.raises(store.RuntimeToolCallWaitingExternalState, match="uncertain"):
        store.append_runtime_tool_physical_attempt(
            request=_physical(logical, ordinal=2, suffix="uncertain-second")
        )


def test_settlement_replay_collision_and_result_contract_are_atomic() -> None:
    seed, _, physical = _reserve_and_append(suffix="settlement")
    wrong_contract = _settlement(
        physical,
        outcome=RuntimeToolPhysicalOutcome.SUCCEEDED,
        suffix="settlement-wrong-contract",
        result_contract="different-result-v1",
    )
    with pytest.raises(store.RuntimeToolCallPersistenceError, match="result contract"):
        store.settle_runtime_tool_physical_attempt(settlement=wrong_contract)
    stored = store.get_runtime_tool_logical_call(
        session_id=seed.session_id,
        logical_tool_call_id=seed.tool_call_id,
    )
    assert stored is not None
    assert stored.physical_attempts[0].settlement is None

    settlement = _settlement(
        physical,
        outcome=RuntimeToolPhysicalOutcome.SUCCEEDED,
        suffix="settlement",
    )
    assert store.settle_runtime_tool_physical_attempt(settlement=settlement).status == "applied"
    assert store.settle_runtime_tool_physical_attempt(settlement=settlement).status == "replayed"
    collision = _settlement(
        physical,
        outcome=RuntimeToolPhysicalOutcome.SUCCEEDED,
        suffix="settlement-collision",
        settle_apply_id=settlement.settle_apply_id,
    )
    with pytest.raises(store.RuntimeToolCallIdentityCollision):
        store.settle_runtime_tool_physical_attempt(settlement=collision)
    settlement_id_collision = _settlement(
        physical,
        outcome=RuntimeToolPhysicalOutcome.SUCCEEDED,
        suffix="settlement-id-collision",
        settlement_id=settlement.settlement_id,
    )
    with pytest.raises(store.RuntimeToolCallIdentityCollision):
        store.settle_runtime_tool_physical_attempt(
            settlement=settlement_id_collision
        )


def test_turn_foreign_keys_reject_physical_and_settlement_without_partial_write() -> None:
    seed = _seed_materialized_call(suffix="turn-fk")
    logical = _logical(seed)
    store.reserve_runtime_tool_logical_call(request=logical)
    foreign_turn = "runtime-tool-unlinked-turn"
    store.create_runtime_turn(
        turn_id=foreign_turn,
        session_id=seed.session_id,
        source="runtime_tool_call_persistence_test",
        user_text="未关联的 Turn",
    )
    invalid = _physical(
        logical,
        suffix="turn-fk-invalid",
        started_turn_id=foreign_turn,
    )
    with pytest.raises(store.RuntimeToolCallPersistenceError):
        store.append_runtime_tool_physical_attempt(request=invalid)
    stored = store.get_runtime_tool_logical_call(
        session_id=seed.session_id,
        logical_tool_call_id=seed.tool_call_id,
    )
    assert stored is not None and stored.physical_attempts == ()

    physical = _physical(logical, suffix="turn-fk-valid")
    store.append_runtime_tool_physical_attempt(request=physical)
    invalid_settlement = _settlement(
        physical,
        outcome=RuntimeToolPhysicalOutcome.SUCCEEDED,
        suffix="turn-fk-invalid",
        settled_turn_id=foreign_turn,
    )
    with pytest.raises(store.RuntimeToolCallPersistenceError):
        store.settle_runtime_tool_physical_attempt(settlement=invalid_settlement)
    stored = store.get_runtime_tool_logical_call(
        session_id=seed.session_id,
        logical_tool_call_id=seed.tool_call_id,
    )
    assert stored is not None and stored.physical_attempts[0].settlement is None
    with store._connect() as conn:
        assert int(
            conn.execute(
                "SELECT COUNT(*) FROM "
                "insession_runtime_tool_call_settlement_receipts"
            ).fetchone()[0]
        ) == 0


@pytest.mark.parametrize(
    "tamper_sql,tamper_parameters",
    [
        (
            "UPDATE insession_runtime_tool_logical_calls "
            "SET effect_profile_sha256=? WHERE logical_tool_call_id=?",
            ("0" * 64, "runtime-tool-call-tamper-effect"),
        ),
        (
            "UPDATE insession_runtime_tool_physical_attempts "
            "SET dispatch_authority_sha256=? WHERE physical_attempt_id=?",
            ("0" * 64, "runtime-tool-physical-tamper-physical"),
        ),
        (
            "UPDATE insession_runtime_tool_call_settlement_receipts "
            "SET receipt_sha256=? WHERE physical_attempt_id=?",
            ("0" * 64, "runtime-tool-physical-tamper-receipt"),
        ),
        (
            "UPDATE insession_work_run_tool_calls SET arguments_json=? "
            "WHERE tool_call_id=?",
            ("{}", "runtime-tool-call-tamper-source"),
        ),
    ],
)
def test_raw_projection_and_materialized_source_splices_fail_closed(
    tamper_sql: str,
    tamper_parameters: tuple[str, str],
) -> None:
    suffix = tamper_parameters[1].removeprefix("runtime-tool-physical-")
    suffix = suffix.removeprefix("runtime-tool-call-")
    seed, _, physical = _reserve_and_append(suffix=suffix)
    store.settle_runtime_tool_physical_attempt(
        settlement=_settlement(
            physical,
            outcome=RuntimeToolPhysicalOutcome.SUCCEEDED,
            suffix=suffix,
        )
    )
    with store._connect() as conn:
        conn.execute(tamper_sql, tamper_parameters)

    with pytest.raises(store.RuntimeToolCallPersistenceError, match="corrupt"):
        store.get_runtime_tool_logical_call(
            session_id=seed.session_id,
            logical_tool_call_id=seed.tool_call_id,
        )


def test_session_purge_cascades_complete_runtime_tool_ledger() -> None:
    seed, _, physical = _reserve_and_append(suffix="purge")
    store.settle_runtime_tool_physical_attempt(
        settlement=_settlement(
            physical,
            outcome=RuntimeToolPhysicalOutcome.SUCCEEDED,
            suffix="purge",
        )
    )
    assert store.purge_session(seed.session_id) is True
    with store._connect() as conn:
        for table in (
            "insession_runtime_tool_call_settlement_receipts",
            "insession_runtime_tool_physical_attempts",
            "insession_runtime_tool_logical_calls",
        ):
            assert int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]) == 0
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []


def test_composite_foreign_key_rejects_cross_attempt_raw_splice() -> None:
    seed = _seed_materialized_call(suffix="raw-fk")
    logical = _logical(seed)
    store.reserve_runtime_tool_logical_call(request=logical)
    with store._connect() as conn:
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "UPDATE insession_runtime_tool_logical_calls SET attempt_id=? "
                "WHERE logical_tool_call_id=?",
                ("missing-attempt", seed.tool_call_id),
            )
    stored = store.get_runtime_tool_logical_call(
        session_id=seed.session_id,
        logical_tool_call_id=seed.tool_call_id,
    )
    assert stored is not None and stored.request == logical
