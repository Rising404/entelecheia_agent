from __future__ import annotations

import ast
import asyncio
import copy
import json
import time
from dataclasses import dataclass
from pathlib import Path

import pytest

from personagraph.l2.task_graph.contracts import InSessionTaskAcceptanceProposal
from personagraph.l2.task_execution.attempts.controller import (
    AttemptActiveTimeMeter,
    run_started_attempt,
)
from personagraph.l2.task_execution.tool_bridge.attempt_contracts import (
    AttemptToolBridgePreflightRequest,
)
from personagraph.l2.task_execution.attempts.decision import (
    AttemptDecisionContext,
    AttemptDecisionInputLimits,
    AttemptUserInput,
    PriorToolResultsProjection,
)
from personagraph.l2.task_execution.task_node.dependencies import (
    TaskNodeDependencyDeliveries,
    TaskNodeDependencyInputLimits,
)
from personagraph.l2.task_execution.tool_bridge.work_run_bridge import (
    SqliteWorkRunToolBridge,
    ToolBridgeAttemptClosed,
    ToolBridgeCallPersistence,
    ToolBridgeMaterialized,
    ToolBridgePersistencePlan,
    ToolBridgeRejected,
    ToolBridgeRejectionCode,
    execute_materialized_call_tools_attempt,
    preflight_and_materialize_call_tools_decision,
    tool_result_from_execution_outcome,
)
from personagraph.l2.task_execution.tool_bridge import (
    work_run_bridge as tool_bridge_module,
)
from personagraph.persistent_turn_content.findings import (
    ExecutionFindingsLedgerStatus,
    ExecutionFindingsOwnerKind,
)
from personagraph.model_io.output_validation import ModelOutputValidationError
from personagraph.tools.policy import (
    ProtectedToolExecutionAuthority,
)
from personagraph.l2.task_execution.tool_bridge.protected_dispatch import (
    RuntimeProtectedToolDispatcher,
)
# 先导入运行时，以维持模型网关当前的模块初始化顺序。
from personagraph.model_io.gateway import ModelGatewayError, ModelResult
from personagraph.session import store
from personagraph.session.l2_store import work_run as work_run_store
from personagraph.tools.catalog import CatalogSnapshot, ToolCatalog
from personagraph.tools.contracts import (
    ExecutionOutcome,
    ExecutionStatus,
    ToolError,
    ToolSourceDescriptor,
    ToolSourceKind,
    ToolSpec,
)
from personagraph.tools.effects import (
    DataEgress,
    EffectAction,
    EffectDescriptor,
    EffectResource,
    EffectScopeKind,
    ToolEffectProfile,
)
from personagraph.tools.policy import AuthorityFacts, ScopeGrant, ToolPolicyCore
from personagraph.tools.registration import (
    ExecutionMode,
    ToolExecutionProfile,
    ToolRegistration,
)
from personagraph.tools.findings.execution_findings_tools import (
    build_execution_findings_tool_registrations,
)
from personagraph.l2.work_run import (
    AttemptDecision,
    CallToolsAction,
    TaskNodeSubject,
    ToolCallProposal,
    ToolResultStatus,
    WorkRunStatus,
)
from tests.helpers.task_node_source_context import task_node_source_context
from tests.helpers.prepared_model_provider import as_prepared_test_provider


def _registration(
    tool_id: str,
    *,
    handler=None,
    contract_version: str = "contract-1",
    implementation_version: str = "implementation-1",
    action: EffectAction = EffectAction.READ,
    resource: EffectResource = EffectResource.MEMORY,
    scope_kind: EffectScopeKind = EffectScopeKind.LOCAL,
    data_egress: DataEgress = DataEgress.NONE,
    execution_profile: ToolExecutionProfile | None = None,
) -> ToolRegistration:
    return ToolRegistration(
        spec=ToolSpec(
            tool_id=tool_id,
            contract_version=contract_version,
            name=f"Test {tool_id}",
            description="Controlled in-process test registration.",
            input_schema={
                "type": "object",
                "required": ["value"],
                "properties": {"value": {"type": "string"}},
                "additionalProperties": False,
            },
            output_schema={
                "type": "object",
                "required": ["value"],
                "properties": {"value": {"type": "string"}},
                "additionalProperties": False,
            },
            catalog_tags=("read",),
        ),
        implementation_version=implementation_version,
        source=ToolSourceDescriptor(ToolSourceKind.LOCAL, "bridge-test"),
        handler=handler or (lambda payload: {"value": payload["value"]}),
        effect_profile=ToolEffectProfile(
            (
                EffectDescriptor(
                    resource,
                    action,
                    scope_kind,
                    data_egress=data_egress,
                ),
            )
        ),
        execution_profile=execution_profile or ToolExecutionProfile(),
    )


def _snapshot(*registrations: ToolRegistration) -> CatalogSnapshot:
    catalog = ToolCatalog()
    for registration in registrations:
        catalog.register(registration)
    return catalog.snapshot()


def _allowed_tools(snapshot: CatalogSnapshot) -> tuple[ToolSpec, ...]:
    return tuple(entry.registration.spec for entry in snapshot.exposed())


def _proposal(*calls: tuple[str, str]) -> AttemptDecision:
    return AttemptDecision(
        action=CallToolsAction(
            calls=tuple(
                ToolCallProposal(tool_id=tool_id, arguments={"value": value})
                for tool_id, value in calls
            )
        )
    )


def _materialize(
    proposal: AttemptDecision,
    snapshot: CatalogSnapshot,
    call_ids: tuple[str, ...],
    *,
    policy: ToolPolicyCore | None = None,
    authority: AuthorityFacts | None = None,
    allow_protected_effects: bool = False,
):
    return preflight_and_materialize_call_tools_decision(
        proposal,
        catalog_snapshot=snapshot,
        stored_catalog_snapshot=snapshot.to_descriptor(),
        allowed_tools=_allowed_tools(snapshot),
        tool_call_ids=call_ids,
        policy=policy,
        authority=authority,
        allow_protected_effects=allow_protected_effects,
    )


@dataclass(frozen=True)
class _StartedAttempt:
    session_id: str
    turn_id: str
    work_run_id: str
    attempt_id: str
    work_run_revision: int
    progress_revision: int
    window_revision: int


def _window_revision(session_id: str) -> int:
    window = store.get_turn_execution_window(session_id)
    assert window is not None
    return int(window["state_version"])


def _seed_started_attempt(snapshot: CatalogSnapshot) -> _StartedAttempt:
    session_id = store.create_session("Entelecheia")
    user_text = "执行测试节点"
    accepted = store.accept_turn_execution(
        session_id=session_id,
        client_request_id="tool-bridge-turn",
        source="runtime_test",
        user_text=user_text,
        lease_owner="tool-bridge-test",
    )
    turn_id = str(accepted["turn"]["turn_id"])
    now = "2026-08-14T00:00:00+00:00"
    acceptances = [
        {
            "acceptance_id": "deliverable",
            "criterion": "产出测试结果",
            "source_anchor_ids": ["request"],
        }
    ]
    source_anchors = [
        {
            "anchor_id": "request",
            "source_turn_id": turn_id,
            "source_kind": "current_user_instruction",
            "start": 0,
            "end": len(user_text),
            "excerpt": user_text,
        }
    ]
    with store._connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            "INSERT INTO insession_tasks "
            "(insession_task_id, session_id, current_graph_revision, current_status, "
            "state_version, root_title, root_objective, created_turn_id, created_at, updated_at) "
            "VALUES ('task-bridge', ?, 1, 'proposed', 1, '测试任务', '完成测试', ?, ?, ?)",
            (session_id, turn_id, now, now),
        )
        conn.execute(
            "INSERT INTO insession_task_graph_revisions "
            "(insession_task_id, graph_revision, source_turn_id, proposal_hash, "
            "source_anchors_json, authorization_anchor_ids_json, required_anchor_ids_json, created_at) "
            "VALUES ('task-bridge', 1, ?, 'proposal-hash', ?, "
            "'[\"request\"]', '[\"request\"]', ?)",
            (turn_id, json.dumps(source_anchors, ensure_ascii=False), now),
        )
        conn.execute(
            "INSERT INTO insession_task_graph_nodes "
            "(insession_task_id, graph_revision, insession_task_node_id, node_revision, "
            "node_kind, ordinal, title, objective, source_anchor_ids_json, "
            "acceptance_criteria_json, constraints_json, created_at) "
            "VALUES ('task-bridge', 1, 'node-bridge', 1, 'root', 0, "
            "'测试节点', '完成节点', '[\"request\"]', ?, '[]', ?)",
            (json.dumps(acceptances, ensure_ascii=False), now),
        )
        conn.execute(
            "INSERT INTO insession_task_node_states "
            "(insession_task_id, insession_task_node_id, node_revision, status, state_version, updated_at) "
            "VALUES ('task-bridge', 'node-bridge', 1, 'proposed', 1, ?)",
            (now,),
        )
        conn.execute(
            "INSERT INTO insession_task_turn_links "
            "(session_id, turn_id, insession_task_id, insession_task_node_id, relation, created_at) "
            "VALUES (?, ?, 'task-bridge', NULL, 'referenced', ?)",
            (session_id, turn_id, now),
        )
    work_run_store.create_task_node_work_run(
        session_id=session_id,
        turn_id=turn_id,
        subject=TaskNodeSubject(
            task_id="task-bridge",
            graph_revision=1,
            node_id="node-bridge",
            node_revision=1,
        ),
        expected_task_state_version=1,
        expected_node_state_version=1,
        expected_window_revision=_window_revision(session_id),
        apply_id="create-bridge-run",
        work_run_id="workrun-bridge",
    )
    work_run_store.start_work_run_attempt(
        session_id=session_id,
        turn_id=turn_id,
        work_run_id="workrun-bridge",
        expected_work_run_revision=1,
        expected_progress_revision=1,
        expected_window_revision=_window_revision(session_id),
        apply_id="start-bridge-attempt",
        catalog_snapshot=snapshot.to_descriptor(),
        attempt_id="attempt-bridge",
    )
    record = work_run_store.get_work_run(
        session_id=session_id,
        work_run_id="workrun-bridge",
    )
    return _StartedAttempt(
        session_id=session_id,
        turn_id=turn_id,
        work_run_id="workrun-bridge",
        attempt_id="attempt-bridge",
        work_run_revision=record.work_run.revision,
        progress_revision=record.acceptance_progress.revision,
        window_revision=_window_revision(session_id),
    )


def _plan(*call_ids: str) -> ToolBridgePersistencePlan:
    return ToolBridgePersistencePlan(
        decision_apply_id="bridge-decision",
        close_apply_id="bridge-close",
        calls=tuple(
            ToolBridgeCallPersistence(
                tool_call_id=call_id,
                tool_result_id=f"result-{ordinal}",
                result_apply_id=f"bridge-result-{ordinal}",
            )
            for ordinal, call_id in enumerate(call_ids, start=1)
        ),
    )


def _active_time_meter(delta: float = 1.0) -> AttemptActiveTimeMeter:
    readings = iter((0.0, delta))
    return AttemptActiveTimeMeter.starting_now(lambda: next(readings))


def _execute(
    started: _StartedAttempt,
    materialized: ToolBridgeMaterialized,
    snapshot: CatalogSnapshot,
    persistence: ToolBridgePersistencePlan,
    *,
    policy: ToolPolicyCore | None = None,
    authority: AuthorityFacts | None = None,
    executor=None,
    protected_dispatcher=None,
    protected_authority_by_key=None,
    durable_result_observer=None,
    active_time_meter: AttemptActiveTimeMeter | None = None,
    execution_store=work_run_store,
):
    return execute_materialized_call_tools_attempt(
        session_id=started.session_id,
        turn_id=started.turn_id,
        work_run_id=started.work_run_id,
        attempt_id=started.attempt_id,
        decision=materialized.decision,
        catalog_snapshot=snapshot,
        allowed_tools=_allowed_tools(snapshot),
        persistence=persistence,
        expected_work_run_revision=started.work_run_revision,
        expected_progress_revision=started.progress_revision,
        expected_window_revision=started.window_revision,
        active_time_meter=active_time_meter or _active_time_meter(),
        policy=policy,
        authority=authority,
        executor=executor,
        protected_dispatcher=protected_dispatcher,
        protected_authority_by_key=protected_authority_by_key or {},
        durable_result_observer=durable_result_observer,
        store=execution_store,
    )


def _execution_counts(started: _StartedAttempt) -> tuple[int, int, int]:
    with store._connect() as conn:
        calls = int(
            conn.execute(
                "SELECT COUNT(*) FROM insession_work_run_tool_calls WHERE work_run_id=?",
                (started.work_run_id,),
            ).fetchone()[0]
        )
        results = int(
            conn.execute(
                "SELECT COUNT(*) FROM insession_work_run_tool_results WHERE work_run_id=?",
                (started.work_run_id,),
            ).fetchone()[0]
        )
        receipts = int(
            conn.execute(
                "SELECT COUNT(*) FROM insession_work_run_apply_receipts WHERE work_run_id=?",
                (started.work_run_id,),
            ).fetchone()[0]
        )
        return calls, results, receipts


def test_work_run_findings_tool_is_internal_idempotent_state_and_closes_with_owner() -> None:
    record_tool, revise_tool = build_execution_findings_tool_registrations()
    source_tool = _registration("read.findings_source")
    snapshot = _snapshot(source_tool, record_tool, revise_tool)
    source_attempt = _seed_started_attempt(snapshot)
    source_materialized = _materialize(
        _proposal((source_tool.tool_id, "source-evidence")),
        snapshot,
        ("call-findings-source",),
    )
    assert isinstance(source_materialized, ToolBridgeMaterialized)
    source_closed = _execute(
        source_attempt,
        source_materialized,
        snapshot,
        ToolBridgePersistencePlan(
            decision_apply_id="findings-source-decision",
            close_apply_id="findings-source-close",
            calls=(
                ToolBridgeCallPersistence(
                    tool_call_id="call-findings-source",
                    tool_result_id="result-findings-source",
                    result_apply_id="findings-source-result-apply",
                ),
            ),
        ),
    )
    assert isinstance(source_closed, ToolBridgeAttemptClosed)
    source_result = source_closed.tool_results[0]

    after_source = work_run_store.get_work_run(
        session_id=source_attempt.session_id,
        work_run_id=source_attempt.work_run_id,
    )
    work_run_store.start_work_run_attempt(
        session_id=source_attempt.session_id,
        turn_id=source_attempt.turn_id,
        work_run_id=source_attempt.work_run_id,
        expected_work_run_revision=after_source.work_run.revision,
        expected_progress_revision=after_source.acceptance_progress.revision,
        expected_window_revision=_window_revision(source_attempt.session_id),
        apply_id="start-findings-writer-attempt",
        catalog_snapshot=snapshot.to_descriptor(),
        attempt_id="attempt-findings-writer",
    )
    after_start = work_run_store.get_work_run(
        session_id=source_attempt.session_id,
        work_run_id=source_attempt.work_run_id,
    )
    started = _StartedAttempt(
        session_id=source_attempt.session_id,
        turn_id=source_attempt.turn_id,
        work_run_id=source_attempt.work_run_id,
        attempt_id="attempt-findings-writer",
        work_run_revision=after_start.work_run.revision,
        progress_revision=after_start.acceptance_progress.revision,
        window_revision=_window_revision(source_attempt.session_id),
    )
    proposal = AttemptDecision(
        action=CallToolsAction(
            calls=(
                ToolCallProposal(
                    tool_id=record_tool.tool_id,
                    arguments={
                        "expected_ledger_revision": 0,
                        "items": [
                            {
                                "kind": "finding",
                                "claim": "The source tool returned the required evidence.",
                                "source_refs": [
                                    {
                                        "tool_result_id": source_result.tool_result_id,
                                    }
                                ],
                                "scope_keys": ["deliverable"],
                            }
                        ],
                    },
                ),
            )
        )
    )
    materialized = _materialize(
        proposal,
        snapshot,
        ("call-record-findings",),
    )
    assert isinstance(materialized, ToolBridgeMaterialized)
    assert materialized.decision.action.calls[0].modifies_environment is False

    closed = _execute(
        started,
        materialized,
        snapshot,
        ToolBridgePersistencePlan(
            decision_apply_id="findings-writer-decision",
            close_apply_id="findings-writer-close",
            calls=(
                ToolBridgeCallPersistence(
                    tool_call_id="call-record-findings",
                    tool_result_id="result-record-findings",
                    result_apply_id="findings-writer-result-apply",
                ),
            ),
        ),
    )
    assert isinstance(closed, ToolBridgeAttemptClosed)
    assert closed.tool_results[0].status is ToolResultStatus.SUCCEEDED
    assert closed.tool_results[0].output["ledger_revision"] == 1

    findings = store.get_execution_findings_ledger_for_owner(
        owner_kind=ExecutionFindingsOwnerKind.WORK_RUN,
        execution_owner_id=started.work_run_id,
    )
    assert findings is not None
    assert findings.ledger.status is ExecutionFindingsLedgerStatus.OPEN
    assert [item.claim for item in findings.active_projection.active_entries] == [
        "The source tool returned the required evidence."
    ]

    with store._connect() as conn:
        conn.execute(
            "UPDATE insession_work_runs SET status='cancelled', "
            "reason='test_terminal_settlement', updated_at=? "
            "WHERE work_run_id=?",
            ("2026-08-26T00:00:00+00:00", started.work_run_id),
        )
    terminal = store.get_execution_findings_ledger_for_owner(
        owner_kind=ExecutionFindingsOwnerKind.WORK_RUN,
        execution_owner_id=started.work_run_id,
    )
    assert terminal is not None
    assert terminal.ledger.status is ExecutionFindingsLedgerStatus.CLOSED


def test_work_run_findings_replays_after_ledger_commit_before_tool_result() -> None:
    record_tool, revise_tool = build_execution_findings_tool_registrations()
    snapshot = _snapshot(record_tool, revise_tool)
    started = _seed_started_attempt(snapshot)
    proposal = AttemptDecision(
        action=CallToolsAction(
            calls=(
                ToolCallProposal(
                    tool_id=record_tool.tool_id,
                    arguments={
                        "expected_ledger_revision": 0,
                        "items": [
                            {
                                "kind": "gap",
                                "claim": "The crash-safe gap survives replay.",
                                "scope_keys": ["deliverable"],
                            }
                        ],
                    },
                ),
            )
        )
    )
    materialized = _materialize(
        proposal,
        snapshot,
        ("call-findings-crash-replay",),
    )
    assert isinstance(materialized, ToolBridgeMaterialized)
    persistence = ToolBridgePersistencePlan(
        decision_apply_id="findings-crash-decision",
        close_apply_id="findings-crash-close",
        calls=(
            ToolBridgeCallPersistence(
                tool_call_id="call-findings-crash-replay",
                tool_result_id="result-findings-crash-replay",
                result_apply_id="findings-crash-result-apply",
            ),
        ),
    )

    class CrashOnceAfterLedgerCommit:
        crashed = False

        def __getattr__(self, name: str):
            return getattr(work_run_store, name)

        def append_work_run_tool_result(self, **kwargs):
            if not self.crashed:
                self.crashed = True
                raise RuntimeError("simulated process loss after ledger commit")
            return work_run_store.append_work_run_tool_result(**kwargs)

    crashing_store = CrashOnceAfterLedgerCommit()
    with pytest.raises(RuntimeError, match="simulated process loss"):
        _execute(
            started,
            materialized,
            snapshot,
            persistence,
            execution_store=crashing_store,
        )

    after_crash = store.get_execution_findings_ledger_for_owner(
        owner_kind=ExecutionFindingsOwnerKind.WORK_RUN,
        execution_owner_id=started.work_run_id,
    )
    assert after_crash is not None
    assert after_crash.ledger.revision == 1
    assert len(after_crash.ledger.entry_revisions) == 1

    replay = _execute(
        started,
        materialized,
        snapshot,
        persistence,
        execution_store=crashing_store,
    )
    assert isinstance(replay, ToolBridgeAttemptClosed)
    assert replay.tool_results[0].output["replayed"] is True
    final = store.get_execution_findings_ledger_for_owner(
        owner_kind=ExecutionFindingsOwnerKind.WORK_RUN,
        execution_owner_id=started.work_run_id,
    )
    assert final is not None
    assert final.ledger.revision == 1
    assert len(final.ledger.entry_revisions) == 1


def test_pure_preflight_binds_exact_exposure_schema_effect_and_implementation_version():
    registration = _registration(
        "read.controlled",
        contract_version="contract-7",
        implementation_version="implementation-19",
    )
    snapshot = _snapshot(registration)

    result = _materialize(
        _proposal(("read.controlled", "hello")),
        snapshot,
        ("call-one",),
    )

    assert isinstance(result, ToolBridgeMaterialized)
    call = result.decision.action.calls[0]
    assert call.tool_id == "read.controlled"
    assert call.tool_version == "implementation-19"
    assert call.modifies_environment is False
    assert call.arguments == {"value": "hello"}
    assert registration.contract_version == "contract-7"


def test_preflight_rejects_same_count_effect_authority_drift():
    registration = _registration("read.controlled")
    snapshot = _snapshot(registration)
    frozen = copy.deepcopy(snapshot.to_descriptor())
    frozen["entries"][0]["registration"]["effects"][0]["action"] = "update"

    result = preflight_and_materialize_call_tools_decision(
        _proposal(("read.controlled", "hello")),
        catalog_snapshot=snapshot,
        stored_catalog_snapshot=frozen,
        allowed_tools=_allowed_tools(snapshot),
        tool_call_ids=("call-one",),
    )

    assert isinstance(result, ToolBridgeRejected)
    assert result.code is ToolBridgeRejectionCode.CATALOG_SNAPSHOT_MISMATCH


def test_bridge_executes_one_real_readonly_registration_and_closes_attempt():
    invoked: list[dict[str, str]] = []
    registration = _registration(
        "read.controlled",
        implementation_version="implementation-42",
        handler=lambda payload: invoked.append(payload) or {"value": payload["value"].upper()},
    )
    snapshot = _snapshot(registration)
    started = _seed_started_attempt(snapshot)
    accepted = _materialize(
        _proposal(("read.controlled", "hello")),
        snapshot,
        ("call-one",),
    )
    assert isinstance(accepted, ToolBridgeMaterialized)

    result = _execute(started, accepted, snapshot, _plan("call-one"))

    assert isinstance(result, ToolBridgeAttemptClosed)
    assert invoked == [{"value": "hello"}]
    assert result.executed_tool_call_ids == ("call-one",)
    assert result.tool_results[0].status is ToolResultStatus.SUCCEEDED
    assert result.tool_results[0].output == {"value": "HELLO"}
    record = work_run_store.get_work_run(
        session_id=started.session_id,
        work_run_id=started.work_run_id,
    )
    assert record.current_attempt_id is None
    assert record.attempts[-1].attempt.status.value == "closed"
    assert record.tool_calls[0].call.tool_version == "implementation-42"
    assert record.tool_results == result.tool_results


def test_durable_result_observer_runs_after_result_commit_before_attempt_close():
    snapshot = _snapshot(_registration("read.observed"))
    started = _seed_started_attempt(snapshot)
    accepted = _materialize(
        _proposal(("read.observed", "value")),
        snapshot,
        ("call-observed",),
    )
    assert isinstance(accepted, ToolBridgeMaterialized)
    observations: list[str] = []

    def observe(*, session_id, turn_id, work_run_id, attempt_id, result):
        del turn_id
        record = work_run_store.get_work_run(
            session_id=session_id,
            work_run_id=work_run_id,
        )
        assert record.current_attempt_id == attempt_id
        assert result in record.tool_results
        observations.append(result.tool_result_id)

    result = _execute(
        started,
        accepted,
        snapshot,
        _plan("call-observed"),
        durable_result_observer=observe,
    )

    assert isinstance(result, ToolBridgeAttemptClosed)
    assert observations == ["result-1"]


def test_bridge_executes_multiple_readonly_calls_sequentially_as_independent():
    order: list[str] = []
    first = _registration(
        "read.first",
        handler=lambda payload: order.append("first") or {"value": payload["value"]},
    )
    second = _registration(
        "read.second",
        handler=lambda payload: order.append("second") or {"value": payload["value"]},
    )
    snapshot = _snapshot(first, second)
    started = _seed_started_attempt(snapshot)
    accepted = _materialize(
        _proposal(("read.first", "one"), ("read.second", "two")),
        snapshot,
        ("call-first", "call-second"),
    )
    assert isinstance(accepted, ToolBridgeMaterialized)

    result = _execute(
        started,
        accepted,
        snapshot,
        _plan("call-first", "call-second"),
    )

    assert isinstance(result, ToolBridgeAttemptClosed)
    assert order == ["first", "second"]
    assert [item.ordinal for item in result.tool_results] == [1, 2]
    assert _execution_counts(started)[:2] == (2, 2)


def test_unknown_or_invalid_call_rejects_whole_batch_without_store_write():
    snapshot = _snapshot(_registration("read.valid"))
    started = _seed_started_attempt(snapshot)
    baseline = _execution_counts(started)
    unknown = _materialize(
        _proposal(("read.valid", "ok"), ("read.unknown", "x")),
        snapshot,
        ("call-valid", "call-unknown"),
    )
    invalid = preflight_and_materialize_call_tools_decision(
        AttemptDecision(
            action=CallToolsAction(
                calls=(
                    ToolCallProposal(
                        tool_id="read.valid",
                        arguments={"value": 42},
                    ),
                )
            )
        ),
        catalog_snapshot=snapshot,
        stored_catalog_snapshot=snapshot.to_descriptor(),
        allowed_tools=_allowed_tools(snapshot),
        tool_call_ids=("call-invalid",),
    )

    assert isinstance(unknown, ToolBridgeRejected)
    assert unknown.code is ToolBridgeRejectionCode.TOOL_NOT_EXPOSED
    assert isinstance(invalid, ToolBridgeRejected)
    assert invalid.code is ToolBridgeRejectionCode.INVALID_TOOL_INPUT
    assert _execution_counts(started) == baseline
    record = work_run_store.get_work_run(
        session_id=started.session_id,
        work_run_id=started.work_run_id,
    )
    assert record.current_attempt_id == started.attempt_id
    assert record.attempts[-1].decision is None
    assert record.attempts[-1].action is None


def test_forged_allowed_tools_and_nonallow_policy_reject_before_store():
    protected_read = _registration(
        "read.protected",
        resource=EffectResource.FILESYSTEM,
        scope_kind=EffectScopeKind.WORKSPACE,
    )
    snapshot = _snapshot(protected_read)
    started = _seed_started_attempt(snapshot)
    baseline = _execution_counts(started)
    forged_spec = ToolSpec(
        tool_id="read.protected",
        contract_version="contract-1",
        name="Forged",
        description="Forged prompt contract.",
        input_schema={"type": "object"},
        output_schema={"type": "object"},
    )
    forged = preflight_and_materialize_call_tools_decision(
        _proposal(("read.protected", "x")),
        catalog_snapshot=snapshot,
        stored_catalog_snapshot=snapshot.to_descriptor(),
        allowed_tools=(forged_spec,),
        tool_call_ids=("call-write",),
    )
    policy_rejected = _materialize(
        _proposal(("read.protected", "x")),
        snapshot,
        ("call-write",),
    )

    assert isinstance(forged, ToolBridgeRejected)
    assert forged.code is ToolBridgeRejectionCode.EXPOSED_TOOL_SET_MISMATCH
    assert isinstance(policy_rejected, ToolBridgeRejected)
    assert policy_rejected.code is ToolBridgeRejectionCode.POLICY_NOT_ALLOWED
    assert policy_rejected.details["disposition"] == "authorization_required"
    assert _execution_counts(started) == baseline
    assert work_run_store.get_work_run(
        session_id=started.session_id,
        work_run_id=started.work_run_id,
    ).attempts[-1].decision is None


def test_executor_failure_and_proven_async_timeout_map_to_distinct_results():
    def fail(_payload):
        raise RuntimeError("controlled failure")

    async def slow(payload):
        await asyncio.sleep(1)
        return {"value": payload["value"]}

    failed = _registration("read.fail", handler=fail)
    timed = _registration(
        "read.timeout",
        handler=slow,
        execution_profile=ToolExecutionProfile(
            default_timeout_s=0.01,
            execution_mode=ExecutionMode.ASYNC,
        ),
    )
    snapshot = _snapshot(failed, timed)
    started = _seed_started_attempt(snapshot)
    accepted = _materialize(
        _proposal(("read.fail", "one"), ("read.timeout", "two")),
        snapshot,
        ("call-fail", "call-timeout"),
    )
    assert isinstance(accepted, ToolBridgeMaterialized)

    result = _execute(
        started,
        accepted,
        snapshot,
        _plan("call-fail", "call-timeout"),
    )

    assert isinstance(result, ToolBridgeAttemptClosed)
    assert [item.status for item in result.tool_results] == [
        ToolResultStatus.FAILED,
        ToolResultStatus.TIMED_OUT,
    ]
    assert result.tool_results[0].error_code == "tool_exception"
    assert result.tool_results[1].error_code == "execution_timeout"
    assert result.mutation.work_run_status is WorkRunStatus.ACTIVE


def test_read_only_sync_timeout_is_known_and_store_does_not_wait_external():
    def slow(payload):
        time.sleep(0.04)
        return {"value": payload["value"]}

    registration = _registration(
        "read.uncancellable",
        handler=slow,
        execution_profile=ToolExecutionProfile(default_timeout_s=0.005),
    )
    snapshot = _snapshot(registration)
    started = _seed_started_attempt(snapshot)
    accepted = _materialize(
        _proposal(("read.uncancellable", "late")),
        snapshot,
        ("call-unknown",),
    )
    assert isinstance(accepted, ToolBridgeMaterialized)

    result = _execute(
        started,
        accepted,
        snapshot,
        _plan("call-unknown"),
        active_time_meter=_active_time_meter(1.0),
    )

    assert isinstance(result, ToolBridgeAttemptClosed)
    assert result.tool_results[0].status is ToolResultStatus.TIMED_OUT
    assert result.mutation.work_run_status is WorkRunStatus.ACTIVE
    assert result.mutation.work_run_reason is None
    assert result.mutation.budget_transition is not None
    assert result.mutation.budget_transition.active_seconds_delta == 1.0
    record = work_run_store.get_work_run(
        session_id=started.session_id,
        work_run_id=started.work_run_id,
    )
    assert record.work_run.status is WorkRunStatus.ACTIVE
    assert record.work_run.budget.active_seconds_consumed == 1.0


@pytest.mark.parametrize(
    "retryable_outcome",
    [
        ExecutionOutcome(
            ExecutionStatus.FAILED,
            error=ToolError("tool_exception", "transient handler exception"),
        ),
        ExecutionOutcome(
            ExecutionStatus.FAILED,
            error=ToolError("invalid_tool_output", "transient invalid output"),
        ),
        ExecutionOutcome(
            ExecutionStatus.FAILED,
            error=ToolError("tool_output_too_large", "transient oversized output"),
        ),
        ExecutionOutcome(
            ExecutionStatus.TIMED_OUT,
            error=ToolError("execution_timeout", "known timeout"),
        ),
    ],
    ids=("exception", "invalid-output", "oversized-output", "known-timeout"),
)
def test_known_technical_failure_retries_same_invocation_up_to_four_physical_tries(
    retryable_outcome: ExecutionOutcome,
):
    registration = _registration(
        "read.retry",
        implementation_version="implementation-retry-7",
    )
    snapshot = _snapshot(registration)
    started = _seed_started_attempt(snapshot)
    accepted = _materialize(
        _proposal(("read.retry", "normalized")),
        snapshot,
        ("call-retry",),
    )
    assert isinstance(accepted, ToolBridgeMaterialized)
    persistence = ToolBridgePersistencePlan(
        decision_apply_id="bridge-decision",
        close_apply_id="bridge-close",
        calls=(
            ToolBridgeCallPersistence(
                tool_call_id="call-retry",
                tool_result_id="result-retry",
                result_apply_id="bridge-result-retry",
                deadline_monotonic=9876.5,
            ),
        ),
    )
    invocations = []
    active_now = 0.0

    class ScriptedExecutor:
        def execute(self, invocation):
            nonlocal active_now
            invocations.append(invocation)
            active_now += 2.0
            if len(invocations) <= 3:
                return retryable_outcome
            return ExecutionOutcome.succeeded({"value": "ok"})

    meter = AttemptActiveTimeMeter.starting_now(lambda: active_now)
    result = _execute(
        started,
        accepted,
        snapshot,
        persistence,
        executor=ScriptedExecutor(),
        active_time_meter=meter,
    )

    assert isinstance(result, ToolBridgeAttemptClosed)
    assert len(invocations) == 4
    assert len({id(item) for item in invocations}) == 1
    assert all(item.registration is registration for item in invocations)
    assert all(
        item.registration.implementation_version == "implementation-retry-7"
        for item in invocations
    )
    assert all(item.arguments == {"value": "normalized"} for item in invocations)
    assert all(item.deadline_monotonic == 9876.5 for item in invocations)
    assert result.executed_tool_call_ids == ("call-retry",)
    assert result.tool_results[0].status is ToolResultStatus.SUCCEEDED
    assert result.tool_results[0].output == {"value": "ok"}
    assert result.mutation.budget_transition is not None
    assert result.mutation.budget_transition.active_seconds_delta == 8.0
    loaded = work_run_store.get_work_run(
        session_id=started.session_id,
        work_run_id=started.work_run_id,
    )
    assert len(loaded.tool_results) == 1
    assert loaded.tool_results[0] == result.tool_results[0]


def test_execution_profile_can_reduce_but_not_expand_transparent_retry_limit():
    with pytest.raises(ValueError, match="integer from 0 to 3"):
        ToolExecutionProfile(max_transparent_retries=4)
    with pytest.raises(ValueError, match="integer from 0 to 3"):
        ToolExecutionProfile(max_transparent_retries=True)

    registration = _registration(
        "read.no-retry",
        execution_profile=ToolExecutionProfile(max_transparent_retries=0),
    )
    snapshot = _snapshot(registration)
    started = _seed_started_attempt(snapshot)
    accepted = _materialize(
        _proposal(("read.no-retry", "one")),
        snapshot,
        ("call-no-retry",),
    )
    assert isinstance(accepted, ToolBridgeMaterialized)
    invocations = 0

    class FailingExecutor:
        def execute(self, _invocation):
            nonlocal invocations
            invocations += 1
            return ExecutionOutcome(
                ExecutionStatus.FAILED,
                error=ToolError("tool_exception", "do not retry this profile"),
            )

    result = _execute(
        started,
        accepted,
        snapshot,
        _plan("call-no-retry"),
        executor=FailingExecutor(),
    )

    assert isinstance(result, ToolBridgeAttemptClosed)
    assert invocations == 1
    assert result.tool_results[0].status is ToolResultStatus.FAILED
    assert (
        registration.descriptor()["execution"]["max_transparent_retries"]
        == 0
    )


def test_exhausted_known_failure_persists_once_and_continues_independent_tail_call():
    first = _registration("read.always-fails")
    second = _registration("read.tail")
    snapshot = _snapshot(first, second)
    started = _seed_started_attempt(snapshot)
    accepted = _materialize(
        _proposal(("read.always-fails", "one"), ("read.tail", "two")),
        snapshot,
        ("call-fail", "call-tail"),
    )
    assert isinstance(accepted, ToolBridgeMaterialized)
    invocations: list[str] = []

    class ScriptedExecutor:
        def execute(self, invocation):
            tool_id = invocation.registration.tool_id
            invocations.append(tool_id)
            if tool_id == "read.always-fails":
                return ExecutionOutcome(
                    ExecutionStatus.FAILED,
                    error=ToolError("tool_exception", "still unavailable"),
                )
            return ExecutionOutcome.succeeded({"value": "tail-ok"})

    result = _execute(
        started,
        accepted,
        snapshot,
        _plan("call-fail", "call-tail"),
        executor=ScriptedExecutor(),
    )

    assert isinstance(result, ToolBridgeAttemptClosed)
    assert invocations == [
        "read.always-fails",
        "read.always-fails",
        "read.always-fails",
        "read.always-fails",
        "read.tail",
    ]
    assert [item.status for item in result.tool_results] == [
        ToolResultStatus.FAILED,
        ToolResultStatus.SUCCEEDED,
    ]
    assert result.tool_results[0].error_code == "tool_exception"
    assert result.tool_results[1].output == {"value": "tail-ok"}
    assert _execution_counts(started)[:2] == (2, 2)


def test_rejected_cancelled_and_business_failures_are_never_retried():
    registrations = tuple(
        _registration(tool_id)
        for tool_id in ("read.rejected", "read.cancelled", "read.business")
    )
    snapshot = _snapshot(*registrations)
    started = _seed_started_attempt(snapshot)
    accepted = _materialize(
        _proposal(
            ("read.rejected", "one"),
            ("read.cancelled", "two"),
            ("read.business", "three"),
        ),
        snapshot,
        ("call-rejected", "call-cancelled", "call-business"),
    )
    assert isinstance(accepted, ToolBridgeMaterialized)
    invocations: list[str] = []

    class ScriptedExecutor:
        def execute(self, invocation):
            tool_id = invocation.registration.tool_id
            invocations.append(tool_id)
            if tool_id == "read.rejected":
                return ExecutionOutcome(
                    ExecutionStatus.REJECTED,
                    error=ToolError("invalid_tool_input", "rejected at execution"),
                )
            if tool_id == "read.cancelled":
                return ExecutionOutcome(
                    ExecutionStatus.CANCELLED,
                    error=ToolError("execution_cancelled", "cancelled"),
                )
            return ExecutionOutcome(
                ExecutionStatus.FAILED,
                error=ToolError("business_rule_failed", "known business failure"),
            )

    result = _execute(
        started,
        accepted,
        snapshot,
        _plan("call-rejected", "call-cancelled", "call-business"),
        executor=ScriptedExecutor(),
    )

    assert isinstance(result, ToolBridgeAttemptClosed)
    assert invocations == ["read.rejected", "read.cancelled", "read.business"]
    assert [item.status for item in result.tool_results] == [
        ToolResultStatus.REJECTED,
        ToolResultStatus.CANCELLED,
        ToolResultStatus.FAILED,
    ]


def test_readonly_unconfirmed_response_is_retried_as_known_failure_then_batch_continues():
    registrations = tuple(
        _registration(tool_id)
        for tool_id in ("read.uncertain", "read.tail-one", "read.tail-two")
    )
    snapshot = _snapshot(*registrations)
    started = _seed_started_attempt(snapshot)
    accepted = _materialize(
        _proposal(
            ("read.uncertain", "one"),
            ("read.tail-one", "two"),
            ("read.tail-two", "three"),
        ),
        snapshot,
        ("call-uncertain", "call-tail-one", "call-tail-two"),
    )
    assert isinstance(accepted, ToolBridgeMaterialized)
    invocations: list[str] = []

    class UncertainExecutor:
        def execute(self, invocation):
            invocations.append(invocation.registration.tool_id)
            if invocation.registration.tool_id != "read.uncertain":
                return ExecutionOutcome.succeeded({"value": "tail-ok"})
            return ExecutionOutcome(
                ExecutionStatus.COMPLETION_UNCONFIRMED,
                error=ToolError(
                    "transport_completion_unconfirmed",
                    "response was lost after dispatch",
                ),
            )

    persistence = _plan("call-uncertain", "call-tail-one", "call-tail-two")
    executor = UncertainExecutor()
    result = _execute(
        started,
        accepted,
        snapshot,
        persistence,
        executor=executor,
    )

    assert isinstance(result, ToolBridgeAttemptClosed)
    assert invocations == [
        "read.uncertain",
        "read.uncertain",
        "read.uncertain",
        "read.uncertain",
        "read.tail-one",
        "read.tail-two",
    ]
    assert [item.status for item in result.tool_results] == [
        ToolResultStatus.FAILED,
        ToolResultStatus.SUCCEEDED,
        ToolResultStatus.SUCCEEDED,
    ]
    assert (
        result.tool_results[0].error_code
        == "read_only_response_unavailable"
    )
    assert result.executed_tool_call_ids == (
        "call-uncertain",
        "call-tail-one",
        "call-tail-two",
    )
    assert result.mutation.work_run_status is WorkRunStatus.ACTIVE
    assert result.mutation.work_run_reason is None
    replay = _execute(
        started,
        accepted,
        snapshot,
        persistence,
        executor=executor,
    )
    assert isinstance(replay, ToolBridgeAttemptClosed)
    assert len(invocations) == 6
    assert replay.executed_tool_call_ids == ()
    assert replay.replayed_tool_call_ids == (
        "call-uncertain",
        "call-tail-one",
        "call-tail-two",
    )
    assert replay.tool_results == result.tool_results


def test_exact_full_replay_reuses_store_receipts_and_never_reexecutes_handler():
    invocations = 0

    def read(payload):
        nonlocal invocations
        invocations += 1
        return {"value": payload["value"]}

    snapshot = _snapshot(_registration("read.once", handler=read))
    started = _seed_started_attempt(snapshot)
    accepted = _materialize(
        _proposal(("read.once", "one")),
        snapshot,
        ("call-once",),
    )
    assert isinstance(accepted, ToolBridgeMaterialized)
    persistence = _plan("call-once")

    first = _execute(started, accepted, snapshot, persistence)
    replay = _execute(started, accepted, snapshot, persistence)

    assert isinstance(first, ToolBridgeAttemptClosed)
    assert isinstance(replay, ToolBridgeAttemptClosed)
    assert invocations == 1
    assert replay.executed_tool_call_ids == ()
    assert replay.replayed_tool_call_ids == ("call-once",)
    assert replay.tool_results == first.tool_results
    assert _execution_counts(started)[:2] == (1, 1)


def test_modifying_call_is_unsupported_before_persistence_even_if_registered():
    snapshot = _snapshot(_registration("write.once", action=EffectAction.UPDATE))
    started = _seed_started_attempt(snapshot)
    baseline = _execution_counts(started)
    rejected = _materialize(
        _proposal(("write.once", "one")),
        snapshot,
        ("call-write",),
    )

    assert isinstance(rejected, ToolBridgeRejected)
    assert rejected.code is ToolBridgeRejectionCode.MODIFYING_CALL_UNSUPPORTED
    assert _execution_counts(started) == baseline


def test_approved_protected_call_uses_runtime_ledger_and_closes_once():
    invocations = 0

    def transmit(payload):
        nonlocal invocations
        invocations += 1
        return {"value": payload["value"].upper()}

    registration = _registration(
        "vision.analyze",
        handler=transmit,
        implementation_version="1+http-vision@1",
        action=EffectAction.TRANSMIT,
        resource=EffectResource.NETWORK,
        scope_kind=EffectScopeKind.SESSION,
        data_egress=DataEgress.CONTENT,
    )
    snapshot = _snapshot(registration)
    started = _seed_started_attempt(snapshot)
    authority = AuthorityFacts(
        approval_grants=(
            ScopeGrant(
                EffectResource.NETWORK,
                EffectAction.TRANSMIT,
                EffectScopeKind.SESSION,
                "*",
            ),
        )
    )
    accepted = _materialize(
        _proposal(("vision.analyze", "chart")),
        snapshot,
        ("call-protected",),
        authority=authority,
        allow_protected_effects=True,
    )
    assert isinstance(accepted, ToolBridgeMaterialized)
    assert accepted.decision.action.calls[0].modifies_environment is True

    result = _execute(
        started,
        accepted,
        snapshot,
        _plan("call-protected"),
        authority=authority,
        protected_dispatcher=RuntimeProtectedToolDispatcher(
            provider_identity_sha256="c" * 64,
            ledger_store=store,
        ),
        protected_authority_by_key={
            (registration.tool_id, registration.contract_version): (
                ProtectedToolExecutionAuthority(
                    approval_receipt_ids=("approved-protected-call",),
                    execution_backend_identity_sha256="c" * 64,
                    revalidate=lambda: True,
                )
            )
        },
    )

    assert isinstance(result, ToolBridgeAttemptClosed)
    assert invocations == 1
    assert result.tool_results[0].status is ToolResultStatus.SUCCEEDED
    assert result.tool_results[0].output == {"value": "CHART"}
    logical = store.get_runtime_tool_logical_call(
        session_id=started.session_id,
        logical_tool_call_id="call-protected",
    )
    assert logical is not None
    assert logical.request.invocation_turn_id == started.turn_id
    assert len(logical.physical_attempts) == 1
    assert logical.physical_attempts[0].settlement is not None


def test_exact_protected_authority_overrides_provider_and_revalidates_before_io():
    invocations = 0
    revalidations = 0

    def update(payload):
        nonlocal invocations
        invocations += 1
        return {"value": payload["value"].upper()}

    def revalidate() -> bool:
        nonlocal revalidations
        revalidations += 1
        return True

    registration = _registration(
        "index.prepare",
        handler=update,
        action=EffectAction.UPDATE,
        resource=EffectResource.RUNTIME_STATE,
        scope_kind=EffectScopeKind.EXECUTION,
    )
    snapshot = _snapshot(registration)
    started = _seed_started_attempt(snapshot)
    authority = AuthorityFacts(
        grants=(
            ScopeGrant(
                EffectResource.RUNTIME_STATE,
                EffectAction.UPDATE,
                EffectScopeKind.EXECUTION,
                "*",
            ),
        )
    )
    accepted = _materialize(
        _proposal((registration.tool_id, "candidate_01")),
        snapshot,
        ("call-exact-protected",),
        authority=authority,
        allow_protected_effects=True,
    )
    assert isinstance(accepted, ToolBridgeMaterialized)
    execution_authority = ProtectedToolExecutionAuthority(
        approval_receipt_ids=("candidate-scope-receipt",),
        execution_backend_identity_sha256="d" * 64,
        revalidate=revalidate,
    )

    result = _execute(
        started,
        accepted,
        snapshot,
        _plan("call-exact-protected"),
        authority=authority,
        protected_dispatcher=RuntimeProtectedToolDispatcher(
            provider_identity_sha256="c" * 64,
            ledger_store=store,
        ),
        protected_authority_by_key={
            (registration.tool_id, registration.contract_version): (
                execution_authority
            )
        },
    )

    assert isinstance(result, ToolBridgeAttemptClosed)
    assert invocations == 1
    # RuntimeLogicalToolCallAuthority 会在逻辑预留前检查一次，并在持久物理尝试
    # 预留之前立即再次检查。
    assert revalidations == 2
    logical = store.get_runtime_tool_logical_call(
        session_id=started.session_id,
        logical_tool_call_id="call-exact-protected",
    )
    assert logical is not None
    assert logical.request.provider_identity_sha256 == "d" * 64
    assert logical.request.state_guard_sha256 != "0" * 64


def test_stale_exact_protected_authority_rejects_without_physical_io():
    invocations = 0

    def update(payload):
        nonlocal invocations
        invocations += 1
        return {"value": payload["value"]}

    registration = _registration(
        "index.stale",
        handler=update,
        action=EffectAction.UPDATE,
        resource=EffectResource.RUNTIME_STATE,
        scope_kind=EffectScopeKind.EXECUTION,
    )
    snapshot = _snapshot(registration)
    started = _seed_started_attempt(snapshot)
    authority = AuthorityFacts(
        grants=(
            ScopeGrant(
                EffectResource.RUNTIME_STATE,
                EffectAction.UPDATE,
                EffectScopeKind.EXECUTION,
                "*",
            ),
        )
    )
    accepted = _materialize(
        _proposal((registration.tool_id, "candidate_02")),
        snapshot,
        ("call-stale-protected",),
        authority=authority,
        allow_protected_effects=True,
    )
    assert isinstance(accepted, ToolBridgeMaterialized)

    result = _execute(
        started,
        accepted,
        snapshot,
        _plan("call-stale-protected"),
        authority=authority,
        protected_dispatcher=RuntimeProtectedToolDispatcher(
            provider_identity_sha256="c" * 64,
            ledger_store=store,
        ),
        protected_authority_by_key={
            (registration.tool_id, registration.contract_version): (
                ProtectedToolExecutionAuthority(
                    approval_receipt_ids=("expired-candidate-scope",),
                    execution_backend_identity_sha256="d" * 64,
                    revalidate=lambda: False,
                )
            )
        },
    )

    assert isinstance(result, ToolBridgeAttemptClosed)
    assert invocations == 0
    assert result.tool_results[0].status is ToolResultStatus.REJECTED
    assert result.tool_results[0].error_code == "protected_tool_state_changed"
    assert (
        store.get_runtime_tool_logical_call(
            session_id=started.session_id,
            logical_tool_call_id="call-stale-protected",
        )
        is None
    )


def test_exact_authority_map_rejects_unbound_protected_call_before_persistence():
    registration = _registration(
        "index.unbound",
        action=EffectAction.UPDATE,
        resource=EffectResource.RUNTIME_STATE,
        scope_kind=EffectScopeKind.EXECUTION,
    )
    snapshot = _snapshot(registration)
    started = _seed_started_attempt(snapshot)
    authority = AuthorityFacts(
        grants=(
            ScopeGrant(
                EffectResource.RUNTIME_STATE,
                EffectAction.UPDATE,
                EffectScopeKind.EXECUTION,
                "*",
            ),
        )
    )
    accepted = _materialize(
        _proposal((registration.tool_id, "candidate_03")),
        snapshot,
        ("call-unbound-protected",),
        authority=authority,
        allow_protected_effects=True,
    )
    assert isinstance(accepted, ToolBridgeMaterialized)
    baseline = _execution_counts(started)

    rejected = _execute(
        started,
        accepted,
        snapshot,
        _plan("call-unbound-protected"),
        authority=authority,
        protected_dispatcher=RuntimeProtectedToolDispatcher(
            provider_identity_sha256="c" * 64,
            ledger_store=store,
        ),
        protected_authority_by_key={},
    )

    assert isinstance(rejected, ToolBridgeRejected)
    assert rejected.code is ToolBridgeRejectionCode.MODIFYING_CALL_UNSUPPORTED
    assert _execution_counts(started) == baseline
    stored = work_run_store.get_work_run(
        session_id=started.session_id,
        work_run_id=started.work_run_id,
    )
    assert stored.attempts[-1].decision is None


def test_sqlite_bridge_preflight_requires_exact_protected_authority():
    registration = _registration(
        "index.preflight",
        action=EffectAction.UPDATE,
        resource=EffectResource.RUNTIME_STATE,
        scope_kind=EffectScopeKind.EXECUTION,
    )
    snapshot = _snapshot(registration)
    authority = AuthorityFacts(
        grants=(
            ScopeGrant(
                EffectResource.RUNTIME_STATE,
                EffectAction.UPDATE,
                EffectScopeKind.EXECUTION,
                "*",
            ),
        )
    )
    bridge = SqliteWorkRunToolBridge(
        catalog_snapshot=snapshot,
        persistence_plan_factory=lambda _request: _plan("unused-call"),
        authority=authority,
        protected_dispatcher=RuntimeProtectedToolDispatcher(
            provider_identity_sha256="c" * 64,
            ledger_store=store,
        ),
        protected_authority_by_key={},
    )

    with pytest.raises(
        ModelOutputValidationError,
        match="modifying_call_unsupported",
    ):
        bridge.preflight(
            AttemptToolBridgePreflightRequest(
                session_id="session-preflight",
                turn_id="turn-preflight",
                work_run_id="work-run-preflight",
                attempt_id="attempt-preflight",
                decision=_proposal((registration.tool_id, "candidate_04")),
                tool_call_ids=("call-preflight",),
                allowed_tools=_allowed_tools(snapshot),
                catalog_snapshot=snapshot.to_descriptor(),
            )
        )


def test_protected_authority_binding_is_receipt_order_independent():
    first = ProtectedToolExecutionAuthority(
        approval_receipt_ids=("receipt-b", "receipt-a"),
        execution_backend_identity_sha256="e" * 64,
        revalidate=lambda: True,
    )
    second = ProtectedToolExecutionAuthority(
        approval_receipt_ids=("receipt-a", "receipt-b"),
        execution_backend_identity_sha256="e" * 64,
        revalidate=lambda: True,
    )
    different = ProtectedToolExecutionAuthority(
        approval_receipt_ids=("receipt-a",),
        execution_backend_identity_sha256="e" * 64,
        revalidate=lambda: True,
    )

    assert first.binding_sha256 == second.binding_sha256
    assert first.binding_sha256 != different.binding_sha256


@pytest.mark.parametrize("status", list(ExecutionStatus))
def test_execution_outcome_mapping_preserves_all_six_statuses(status: ExecutionStatus):
    outcome = (
        ExecutionOutcome.succeeded({"value": "ok"})
        if status is ExecutionStatus.SUCCEEDED
        else ExecutionOutcome(
            status,
            error=ToolError("controlled", "controlled outcome"),
        )
    )

    result = tool_result_from_execution_outcome(
        outcome,
        tool_result_id="result",
        tool_call_id="call",
        attempt_id="attempt",
        ordinal=1,
    )

    assert result.status.value == status.value
    assert (result.output == {"value": "ok"}) is (
        status is ExecutionStatus.SUCCEEDED
    )


def test_tool_bridge_has_no_legacy_loop_import():
    path = Path(tool_bridge_module.__file__)
    module = ast.parse(path.read_text(encoding="utf-8"))
    imports = {
        node.module or ""
        for node in ast.walk(module)
        if isinstance(node, ast.ImportFrom)
    }
    imports |= {
        alias.name
        for node in ast.walk(module)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert not any(name.endswith("tools.loop") for name in imports)
    assert not any(name.endswith("tools.registry") for name in imports)


def test_controller_real_bridge_charges_retries_backoff_and_tool_batch_once(
    monkeypatch,
):
    invocations = 0
    active_now = 0.0

    def advance(seconds: float) -> None:
        nonlocal active_now
        active_now += seconds

    def clock() -> float:
        return active_now

    def read(payload):
        nonlocal invocations
        invocations += 1
        advance(8.0)
        return {"value": payload["value"].upper()}

    monkeypatch.setattr(
        "personagraph.runtime.model_calls.requests.backoff_delay_s",
        lambda _attempt: 4.0,
    )
    monkeypatch.setattr(
        "personagraph.runtime.model_calls.requests._sleep",
        advance,
    )

    snapshot = _snapshot(_registration("read.integrated", handler=read))
    started = _seed_started_attempt(snapshot)
    stored = work_run_store.get_work_run(
        session_id=started.session_id,
        work_run_id=started.work_run_id,
    )
    current_attempt = stored.attempts[-1]
    acceptances = (
        InSessionTaskAcceptanceProposal(
            acceptance_id="deliverable",
            criterion="产出测试结果",
            source_anchor_ids=("request",),
        ),
    )
    context = AttemptDecisionContext(
        session_id=started.session_id,
        turn_id=started.turn_id,
        work_run_id=started.work_run_id,
        work_run_revision=stored.work_run.revision,
        attempt_id=started.attempt_id,
        attempt_ordinal=current_attempt.attempt.ordinal,
        user_input=AttemptUserInput(content="执行测试节点"),
        subject=stored.work_run.subject,
        node_title="测试节点",
        node_objective="完成节点",
        acceptances=acceptances,
        acceptance_progress=stored.acceptance_progress,
        output_window=stored.output_window,
        dependency_deliveries=TaskNodeDependencyDeliveries(),
        source_context=task_node_source_context(
            session_id=started.session_id,
            subject=stored.work_run.subject,
            acceptances=acceptances,
        ),
        prior_tool_results=PriorToolResultsProjection(),
        input_limits=AttemptDecisionInputLimits(
            profile_id="tool-bridge-attempt-test-v1",
            max_prior_tool_result_items=32,
            max_prior_tool_results_serialized_utf8_bytes=128_000,
            dependency_delivery_limits=TaskNodeDependencyInputLimits(
                profile_id="tool-bridge-attempt-dependencies-v1",
                max_items=32,
                max_serialized_utf8_bytes=256_000,
            ),
            max_serialized_utf8_bytes=1_000_000,
        ),
        allowed_tools=_allowed_tools(snapshot),
    )
    provider_call_ids: list[str] = []

    def provider(
        _system_prompt: str,
        _user_content: str,
        *,
        model_call_id: str,
        purpose: str,
    ) -> ModelResult:
        assert purpose == "runtime_work_run_attempt_decision"
        provider_call_ids.append(model_call_id)
        advance(1.0)
        if len(provider_call_ids) == 1:
            raise ModelGatewayError(
                "MODEL_TRANSPORT_TEST",
                "injected retryable transport failure",
                retryable=True,
            )
        value: object = 42 if len(provider_call_ids) == 2 else "fixed"
        return ModelResult(
            reply=json.dumps(
                {
                    "acceptance_updates": [],
                    "action": {
                        "kind": "call_tools",
                        "calls": [
                            {
                                "tool_id": "read.integrated",
                                "arguments": {"value": value},
                            }
                        ],
                    },
                }
            ),
            provider="test",
            model="test",
            latency_ms=1,
            model_call_id=model_call_id,
        )

    plan_factory_calls = 0

    def persistence_plan(request):
        nonlocal plan_factory_calls
        plan_factory_calls += 1
        call = request.decision.action.calls[0]
        return ToolBridgePersistencePlan(
            decision_apply_id=request.apply_id,
            close_apply_id=f"{request.apply_id}-close",
            calls=(
                ToolBridgeCallPersistence(
                    tool_call_id=call.tool_call_id,
                    tool_result_id="integrated-result",
                    result_apply_id=f"{request.apply_id}-result-1",
                ),
            ),
        )

    bridge = SqliteWorkRunToolBridge(
        catalog_snapshot=snapshot,
        persistence_plan_factory=persistence_plan,
    )
    active_time_meter = AttemptActiveTimeMeter.starting_now(clock)
    result = run_started_attempt(
        context,
        expected_window_revision=started.window_revision,
        apply_id="integrated-decision",
        provider=as_prepared_test_provider(provider),
        emit=lambda _event: None,
        active_time_meter=active_time_meter,
        tool_bridge=bridge,
        tool_call_id_factory=lambda attempt_id, ordinal: (
            f"{attempt_id}-call-{ordinal}"
        ),
    )

    assert result.action == "call_tools"
    assert result.model_call.physical_attempts == 3
    assert len(provider_call_ids) == 3
    assert len(set(provider_call_ids)) == 1
    assert plan_factory_calls == 1
    assert invocations == 1
    assert result.mutation.budget_transition is not None
    assert result.mutation.budget_transition.active_seconds_delta == 15.0
    assert active_time_meter.freeze() == 15.0
    assert clock() == 15.0
    loaded = work_run_store.get_work_run(
        session_id=started.session_id,
        work_run_id=started.work_run_id,
    )
    assert loaded.current_attempt_id is None
    assert loaded.attempts[-1].attempt.status.value == "closed"
    assert len(loaded.tool_results) == 1
    assert loaded.tool_results[0].tool_result_id == "integrated-result"
    assert loaded.tool_results[0].output == {"value": "FIXED"}
    assert loaded.work_run.budget.active_seconds_consumed == 15.0
    assert len(loaded.budget_charges) == 1
