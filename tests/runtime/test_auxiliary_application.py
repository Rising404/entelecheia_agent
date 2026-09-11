from __future__ import annotations

import json
from dataclasses import replace

import pytest

from personagraph.l2.auxiliary_graph import TaskGraphSemanticTerminalRoute
from personagraph.l2.task_graph.task_matching import InSessionTaskMatchesProposal
from personagraph.model_io.gateway import ModelResult
from personagraph.workspace.documents import application as docstore
from personagraph.l2.task_graph.production_gate import (
    TaskGraphProductionFailureCode,
)
from personagraph.l2.auxiliary_execution import application
from personagraph.l2.auxiliary_execution.terminal import composition as terminal
from personagraph.l2.auxiliary_execution.application import (
    AuxiliaryApplicationPorts,
    AuxiliaryApplicationRequest,
    AuxiliaryApplicationStatus,
    run_auxiliary_application_to_boundary,
)
from personagraph.l2.auxiliary_execution.planning.model_provider import (
    build_auxiliary_architect_structured_provider,
)
from personagraph.l2.auxiliary_execution.planning.replanning_controller import (
    AuxiliaryReplanningRequest,
    AuxiliaryReplanningStatus,
    run_auxiliary_replanning,
)
from personagraph.runtime.model_calls import DurableModelCallTerminalState
from personagraph.runtime.turn_deadline import TurnDeadlineExceeded
from personagraph.runtime.model_calls import (
    RuntimeModelCallWaitingExternal,
)
from personagraph.l2.task_execution.work_run.model_providers import (
    build_attempt_structured_provider,
)
from personagraph.session import store
from personagraph.session.l2_store import task_graph as task_graph_store
from personagraph.session.l2_store import auxiliary_graph as auxiliary_graph_store
from personagraph.session.l2_store import planning as planning_store
from personagraph.session.l2_store import semantic_verification as semantic_store
from personagraph.session.persistence.l2.auxiliary_graph import auxiliary_graphs
from tests.runtime.test_auxiliary_planning_controller import (
    _no_mounted_documents,
    _seed_task,
)
from tests.runtime.test_auxiliary_semantic_verification_controller import (
    _SemanticProvider,
    _authority_factory,
)
from tests.helpers.prepared_model_provider import as_prepared_test_provider


@pytest.fixture(autouse=True)
def _virtual_planning_sources_are_physically_current(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """该模块的挂载夹具使用虚构的非披露路径。"""

    def check(session_id: str, document_id: str | None = None):
        mounted = tuple(docstore.mounted_docs(session_id))
        selected = tuple(
            document
            for document in mounted
            if document_id is None or document.get("id") == document_id
        )
        if document_id is not None and not selected:
            return {
                "ok": False,
                "status": "freshness_blocked",
                "documents": [
                    {"doc_id": document_id, "status": "not_mounted"}
                ],
            }
        return {
            "ok": True,
            "status": "verified_current",
            "documents": [
                {
                    "doc_id": str(document["id"]),
                    "version_id": str(document["current_version_id"]),
                    "status": "verified_current",
                }
                for document in selected
            ],
        }

    monkeypatch.setattr(docstore, "check_mounted_document_freshness", check)


def _ledger_counts(session_id: str) -> tuple[int, int]:
    with store._connect() as conn:
        logical = conn.execute(
            "SELECT COUNT(*) FROM insession_runtime_model_logical_calls "
            "WHERE session_id=?",
            (session_id,),
        ).fetchone()[0]
        physical = conn.execute(
            "SELECT COUNT(*) FROM insession_runtime_model_physical_attempts "
            "WHERE session_id=?",
            (session_id,),
        ).fetchone()[0]
    return int(logical), int(physical)


def test_application_default_profile_covers_terminal_reasoning_envelope() -> None:
    profile = AuxiliaryApplicationPorts(
        model_ledger_store=store,
        emit=lambda _event: None
    ).work_run_model_profile

    assert profile.attempt_max_output_tokens == 131_072
    assert profile.verification_max_output_tokens == 131_072
    assert profile.timeout_s == 600.0


def test_application_ports_require_explicit_model_ledger() -> None:
    with pytest.raises(TypeError, match="model_ledger_store"):
        AuxiliaryApplicationPorts(emit=lambda _event: None)  # type: ignore[call-arg]


def test_application_ports_reject_an_incomplete_model_ledger() -> None:
    with pytest.raises(
        TypeError,
        match="model_ledger_store.reserve_runtime_model_logical_call",
    ):
        AuxiliaryApplicationPorts(
            model_ledger_store=object(),
            emit=lambda _event: None,
        )


def test_application_projects_visual_authority_failure_during_planning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_id, turn_id, task_id = _seed_task()
    _no_mounted_documents(monkeypatch, session_id)
    monkeypatch.setattr(
        application,
        "run_initial_auxiliary_planning",
        lambda **_kwargs: (_ for _ in ()).throw(
            application.MountedVisualPlanningAuthorityError(
                "visual authority is stale"
            )
        ),
    )

    result = run_auxiliary_application_to_boundary(
        AuxiliaryApplicationRequest(
            session_id=session_id,
            turn_id=turn_id,
            task_id=task_id,
            max_effect_steps=2,
        ),
        ports=AuxiliaryApplicationPorts(model_ledger_store=store, emit=lambda _event: None),
    )

    assert result.status is AuxiliaryApplicationStatus.FAILED
    assert result.reason_code == "mounted_visual_planning_authority_unavailable"


def test_application_projects_visual_authority_failure_during_task_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_id, turn_id, task_id = _seed_task()
    _no_mounted_documents(monkeypatch, session_id)
    monkeypatch.setattr(
        application,
        "freeze_mounted_document_planning_authority",
        lambda **_kwargs: (_ for _ in ()).throw(
            application.MountedVisualPlanningAuthorityError(
                "visual authority is corrupt"
            )
        ),
    )

    result = run_auxiliary_application_to_boundary(
        AuxiliaryApplicationRequest(
            session_id=session_id,
            turn_id=turn_id,
            task_id=task_id,
            max_effect_steps=8,
        ),
        ports=AuxiliaryApplicationPorts(model_ledger_store=store, emit=lambda _event: None),
    )

    assert result.status is AuxiliaryApplicationStatus.FAILED
    assert result.reason_code == "mounted_document_cognition_runtime_unavailable"


def test_application_root_commits_full_chain_and_reentry_has_no_effects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_id, turn_id, task_id = _seed_task()
    _no_mounted_documents(monkeypatch, session_id)
    real_authority_factory = (
        application.create_auxiliary_work_run_model_call_authority
    )
    admitted_call_kinds: list[str] = []

    def tracing_authority_factory(
        binding,
        *,
        rederive_state_guard_sha256,
        ledger_store,
    ):
        admitted_call_kinds.append(binding.call_kind)
        return real_authority_factory(
            binding,
            rederive_state_guard_sha256=rederive_state_guard_sha256,
            ledger_store=ledger_store,
        )

    monkeypatch.setattr(
        application,
        "create_auxiliary_work_run_model_call_authority",
        tracing_authority_factory,
    )
    request = AuxiliaryApplicationRequest(
        session_id=session_id,
        turn_id=turn_id,
        task_id=task_id,
        max_effect_steps=8,
    )
    ports = AuxiliaryApplicationPorts(model_ledger_store=store, emit=lambda _event: None)

    committed = run_auxiliary_application_to_boundary(request, ports=ports)

    assert committed.status is AuxiliaryApplicationStatus.COMMITTED
    assert committed.reason_code == "task_graph_revision_one_committed"
    assert committed.effect_steps == 4
    assert admitted_call_kinds == [
        "attempt_decision",
        "node_verification",
        "attempt_decision",
        "node_verification",
    ]
    task = task_graph_store.get_insession_task_details(session_id, task_id)
    assert task is not None
    assert task.current_graph_revision == 1
    before_reentry = _ledger_counts(session_id)

    replayed = run_auxiliary_application_to_boundary(request, ports=ports)

    assert replayed.status is AuxiliaryApplicationStatus.COMMITTED
    assert replayed.reason_code == "task_graph_already_committed"
    assert replayed.effect_steps == 0
    assert _ledger_counts(session_id) == before_reentry
    assert admitted_call_kinds == [
        "attempt_decision",
        "node_verification",
        "attempt_decision",
        "node_verification",
    ]


def test_application_executes_durable_user_gate_across_turns_then_commits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_id, question_turn_id, task_id = _seed_task()
    _no_mounted_documents(monkeypatch, session_id)
    physical_planning = build_auxiliary_architect_structured_provider()
    physical_attempt = build_attempt_structured_provider(
        AuxiliaryApplicationPorts(
            model_ledger_store=store,
            emit=lambda _event: None
        ).work_run_model_profile
    )
    question = "请明确希望计划覆盖最近几年。"
    answer = "覆盖最近三年。"
    planning_calls = 0
    gate_calls: list[str] = []
    terminal_calls = 0

    def planning_provider(*args, **kwargs):
        nonlocal planning_calls
        planning_calls += 1
        result = physical_planning(*args, **kwargs)
        proposal = json.loads(result.reply)
        node = proposal["structure"]["nodes"][0]
        node.update(
            {
                "node_kind": "clarify",
                "executor_kind": "user_gate",
                "title": "Clarify the blocking planning input",
                "objective": question,
                "acceptance_criteria": [
                    {
                        "acceptance_id": "user_answer_bound",
                        "criterion": "The exact user answer is durably bound.",
                        "source_anchor_ids": node["source_anchor_ids"],
                    }
                ],
                "capability_profile_id": None,
                "input_resource_aliases": [],
                "output_contract": "user_response_v1",
            }
        )
        return replace(
            result,
            reply=json.dumps(proposal, ensure_ascii=False),
        )

    planning_provider = as_prepared_test_provider(planning_provider)

    def attempt_provider(system_prompt, user_content, **kwargs):
        nonlocal terminal_calls
        payload = json.loads(user_content)
        contract = payload.get("user_gate_contract")
        if contract is None:
            terminal_calls += 1
            return physical_attempt(system_prompt, user_content, **kwargs)
        phase = contract["phase"]
        gate_calls.append(phase)
        if phase == "ask":
            reply = {
                "acceptance_updates": [],
                "action": {
                    "kind": "request_user_input",
                    "question": question,
                },
            }
        else:
            reply = {
                "acceptance_updates": [
                    {
                        "acceptance_id": "user_answer_bound",
                        "model_claimed_satisfied": True,
                    }
                ],
                "action": {
                    "kind": "submit_output_window",
                    "content": "accept_answer",
                    "format": "plain_text",
                },
            }
        return ModelResult(
            reply=json.dumps(reply, ensure_ascii=False),
            provider="mock",
            model="mock-structured",
            latency_ms=1,
            model_call_id=kwargs["model_call_id"],
            finish_reason="stop",
        )

    attempt_provider = as_prepared_test_provider(attempt_provider)

    ports = AuxiliaryApplicationPorts(
        model_ledger_store=store,
        emit=lambda _event: None,
        planning_provider=planning_provider,
        attempt_provider=attempt_provider,
    )
    first_request = AuxiliaryApplicationRequest(
        session_id=session_id,
        turn_id=question_turn_id,
        task_id=task_id,
        max_effect_steps=8,
    )

    waiting = run_auxiliary_application_to_boundary(first_request, ports=ports)

    assert waiting.status is AuxiliaryApplicationStatus.WAITING_USER
    assert waiting.requested_user_question == question
    assert waiting.effect_steps == 2
    assert gate_calls == ["ask"]
    assert terminal_calls == 0
    assert task_graph_store.get_insession_task_details(session_id, task_id).current_graph_revision is None
    finalized = store.finalize_turn_execution(
        session_id=session_id,
        turn_id=question_turn_id,
        expected_window_revision=int(
            store.get_turn_execution_window(session_id)["state_version"]
        ),
        processing_level="L2",
        assistant_content=question,
        post_commit_job_kinds=(),
    )
    store.release_turn_execution_window(
        session_id=session_id,
        turn_id=question_turn_id,
        expected_window_revision=int(finalized["window"]["state_version"]),
    )
    accepted = store.accept_turn_execution(
        session_id=session_id,
        client_request_id="application-user-gate-answer",
        source="auxiliary_v2_application_test",
        user_text=answer,
        lease_owner="auxiliary-v2-application-test",
    )
    answer_turn_id = str(accepted["turn"]["turn_id"])
    answer_window = store.get_turn_execution_window(session_id)
    assert answer_window is not None
    task_graph_store.apply_insession_task_matches(
        session_id=session_id,
        source_turn_id=answer_turn_id,
        apply_id="application-user-gate-answer-match",
        proposal=InSessionTaskMatchesProposal.model_validate(
            {
                "task_matches": [
                    {
                        "match_type": "existing_root",
                        "insession_task_id": task_id,
                        "source_excerpt": answer,
                        "execute_current": True,
                    }
                ]
            }
        ),
        exposed_catalog_ids=(task_id,),
        expected_window_revision=int(answer_window["state_version"]),
    )
    answer_request = AuxiliaryApplicationRequest(
        session_id=session_id,
        turn_id=answer_turn_id,
        task_id=task_id,
        max_effect_steps=8,
    )
    answer_frontier = auxiliary_graph_store.project_auxiliary_graph_execution_frontier(
        session_id=session_id,
        turn_id=answer_turn_id,
        insession_task_id=task_id,
    )
    assert len(answer_frontier.recoverable) == 1

    committed = run_auxiliary_application_to_boundary(answer_request, ports=ports)
    provider_counts = (planning_calls, len(gate_calls), terminal_calls)
    replayed_after_response_loss = run_auxiliary_application_to_boundary(
        answer_request,
        ports=AuxiliaryApplicationPorts(
            model_ledger_store=store,
            emit=lambda _event: None,
            planning_provider=lambda *_args, **_kwargs: pytest.fail(
                "committed UserGate replay reached planning Provider"
            ),
            attempt_provider=lambda *_args, **_kwargs: pytest.fail(
                "committed UserGate replay reached Attempt Provider"
            ),
            verification_provider=lambda *_args, **_kwargs: pytest.fail(
                "committed UserGate replay reached Verification Provider"
            ),
            semantic_provider=lambda *_args, **_kwargs: pytest.fail(
                "committed UserGate replay reached semantic Provider"
            ),
        ),
    )

    assert committed.status is AuxiliaryApplicationStatus.COMMITTED, (
        committed.reason_code,
        committed.last_driver_action,
        gate_calls,
    )
    assert committed.reason_code == "task_graph_revision_one_committed"
    assert replayed_after_response_loss.status is AuxiliaryApplicationStatus.COMMITTED
    assert replayed_after_response_loss.effect_steps == 0
    assert gate_calls == ["ask", "consume_answer"]
    assert terminal_calls == 1
    assert (planning_calls, len(gate_calls), terminal_calls) == provider_counts
    task = task_graph_store.get_insession_task_details(session_id, task_id)
    assert task is not None and task.current_graph_revision == 1


@pytest.mark.parametrize(
    ("failure", "expected_status", "expected_reason"),
    (
        (
            RuntimeModelCallWaitingExternal("uncertain dispatch"),
            AuxiliaryApplicationStatus.WAITING_EXTERNAL,
            "work_run_model_call_waiting_external",
        ),
        (
            TurnDeadlineExceeded(),
            AuxiliaryApplicationStatus.TURN_LIMIT_REACHED,
            "work_run_turn_deadline_reached",
        ),
        (
            DurableModelCallTerminalState("terminal model ledger state"),
            AuxiliaryApplicationStatus.FAILED,
            "DurableModelCallTerminalState",
        ),
    ),
)
def test_application_maps_work_run_model_authority_boundaries(
    monkeypatch: pytest.MonkeyPatch,
    failure: Exception,
    expected_status: AuxiliaryApplicationStatus,
    expected_reason: str,
) -> None:
    session_id, turn_id, task_id = _seed_task()
    _no_mounted_documents(monkeypatch, session_id)

    def fail_at_work_run(*_args, **_kwargs):
        raise failure

    monkeypatch.setattr(
        application,
        "run_auxiliary_model_node",
        fail_at_work_run,
    )
    result = run_auxiliary_application_to_boundary(
        AuxiliaryApplicationRequest(
            session_id=session_id,
            turn_id=turn_id,
            task_id=task_id,
            max_effect_steps=2,
        ),
        ports=AuxiliaryApplicationPorts(model_ledger_store=store, emit=lambda _event: None),
    )

    assert result.status is expected_status
    assert result.reason_code == expected_reason
    assert result.effect_steps == 2


def test_application_preserves_terminal_candidate_non_workrun_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    route = TaskGraphSemanticTerminalRoute.BLOCKED
    expected_reason = "terminal_candidate_semantic_blocked"
    session_id, turn_id, task_id = _seed_task()
    _no_mounted_documents(monkeypatch, session_id)
    review = terminal.AuxiliaryTerminalCandidateSemanticReview.model_construct(
        route=route,
        requests=(),
        results=(),
    )

    def require_route(**_kwargs):
        raise terminal.AuxiliaryTerminalCandidateSemanticRouteRequired(
            route=route,
            review=review,
        )

    monkeypatch.setattr(
        application,
        "run_auxiliary_terminal_candidate_semantic_gate",
        require_route,
    )

    result = run_auxiliary_application_to_boundary(
        AuxiliaryApplicationRequest(
            session_id=session_id,
            turn_id=turn_id,
            task_id=task_id,
            max_effect_steps=8,
        ),
        ports=AuxiliaryApplicationPorts(model_ledger_store=store, emit=lambda _event: None),
    )

    assert result.status is AuxiliaryApplicationStatus.FAILED
    assert result.reason_code == expected_reason
    assert result.effect_steps == 3
    task = task_graph_store.get_insession_task_details(session_id, task_id)
    assert task is not None and task.current_graph_revision is None


def test_application_replans_open_terminal_candidate_without_freezing_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_id, turn_id, task_id = _seed_task()
    _no_mounted_documents(monkeypatch, session_id)
    physical_planner = build_auxiliary_architect_structured_provider()
    planning_payloads: list[dict[str, object]] = []

    def planning_provider(system_prompt, user_content, **kwargs):
        planning_payloads.append(json.loads(user_content))
        return physical_planner(system_prompt, user_content, **kwargs)

    planning_provider = as_prepared_test_provider(planning_provider)

    semantic_provider = _SemanticProvider(
        failed_reviewer_ordinal=1,
        failure_scope="auxiliary_investigation",
    )
    semantic_authority_factory, _bindings = _authority_factory()
    request = AuxiliaryApplicationRequest(
        session_id=session_id,
        turn_id=turn_id,
        task_id=task_id,
        max_effect_steps=16,
        max_autonomous_replans=3,
    )
    committed = run_auxiliary_application_to_boundary(
        request,
        ports=AuxiliaryApplicationPorts(
            model_ledger_store=store,
            emit=lambda _event: None,
            planning_provider=planning_provider,
            semantic_provider=semantic_provider,
            semantic_model_call_authority_factory=semantic_authority_factory,
        ),
    )

    assert committed.status is AuxiliaryApplicationStatus.COMMITTED
    assert committed.reason_code == "task_graph_revision_one_committed"
    assert committed.replan_result is not None
    assert (
        committed.replan_result.status
        is AuxiliaryReplanningStatus.REPLANNED
    )
    assert len(planning_payloads) == 2
    assert planning_payloads[0]["replan_trigger"] is None
    assert planning_payloads[1]["replan_trigger"] is not None
    assert len(semantic_provider.calls) == 2

    details = auxiliary_graph_store.get_auxiliary_graph_for_task(
        session_id=session_id,
        insession_task_id=task_id,
    )
    assert details is not None
    assert details.auxiliary_graph_revision == 3
    assert details.parent_auxiliary_graph_revision == 2
    assert details.reason == "verification_failed"
    trigger = committed.replan_result.trigger
    assert trigger is not None
    assert planning_store.count_auxiliary_replan_triggers(
        session_id=session_id,
        task_id=task_id,
        goal_id=details.goal_id,
    ) == 1
    assert planning_store.get_active_auxiliary_replan_trigger(
        session_id=session_id,
        task_id=task_id,
    ) is None

    with store._connect() as conn:
        rejected = conn.execute(
            "SELECT run.status AS run_status, run.reason AS run_reason, "
            "output.frozen_at, "
            "(SELECT COUNT(*) FROM insession_auxiliary_node_completions_v2 "
            "WHERE work_run_id=run.work_run_id) AS completion_count "
            "FROM insession_work_runs AS run "
            "JOIN insession_auxiliary_v2_node_execution_subject_bindings AS binding "
            "ON binding.binding_id=(SELECT auxiliary_v2_binding_id "
            "FROM insession_execution_subjects "
            "WHERE execution_subject_id=run.execution_subject_id) "
            "JOIN insession_work_run_output_windows AS output "
            "ON output.work_run_id=run.work_run_id "
            "WHERE run.session_id=? AND run.auxiliary_graph_revision=2 "
            "AND binding.executor_kind='terminal_planner'",
            (session_id,),
        ).fetchone()
    assert rejected is not None
    assert rejected["run_status"] == "cancelled"
    assert rejected["run_reason"] == "terminal_candidate_semantic_replan"
    assert rejected["frozen_at"] is None
    assert int(rejected["completion_count"]) == 0

    def must_not_run(*_args, **_kwargs):
        raise AssertionError("committed candidate-replan replay reached a Provider")

    replayed = run_auxiliary_application_to_boundary(
        request,
        ports=AuxiliaryApplicationPorts(
            model_ledger_store=store,
            emit=lambda _event: None,
            planning_provider=must_not_run,
            attempt_provider=must_not_run,
            verification_provider=must_not_run,
            semantic_provider=must_not_run,
        ),
    )
    assert replayed.status is AuxiliaryApplicationStatus.COMMITTED
    assert replayed.reason_code == "task_graph_already_committed"
    assert replayed.effect_steps == 0
    assert len(planning_payloads) == 2
    assert len(semantic_provider.calls) == 2


def test_candidate_replan_revision_failure_rolls_back_cursor_and_recovers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_id, turn_id, task_id = _seed_task()
    _no_mounted_documents(monkeypatch, session_id)
    physical_planner = build_auxiliary_architect_structured_provider()
    planning_calls: list[dict[str, object]] = []

    def planning_provider(system_prompt, user_content, **kwargs):
        planning_calls.append(json.loads(user_content))
        return physical_planner(system_prompt, user_content, **kwargs)

    planning_provider = as_prepared_test_provider(planning_provider)

    semantic_provider = _SemanticProvider(
        failed_reviewer_ordinal=1,
        failure_scope="auxiliary_investigation",
    )
    semantic_authority_factory, _bindings = _authority_factory()
    stopped = run_auxiliary_application_to_boundary(
        AuxiliaryApplicationRequest(
            session_id=session_id,
            turn_id=turn_id,
            task_id=task_id,
            max_effect_steps=3,
        ),
        ports=AuxiliaryApplicationPorts(
            model_ledger_store=store,
            emit=lambda _event: None,
            planning_provider=planning_provider,
            semantic_provider=semantic_provider,
            semantic_model_call_authority_factory=semantic_authority_factory,
        ),
    )
    assert stopped.status is AuxiliaryApplicationStatus.STEP_LIMIT_REACHED

    with store._connect() as conn:
        settlement_row = conn.execute(
            "SELECT session_id, insession_task_id, auxiliary_graph_id, "
            "goal_id, auxiliary_graph_revision, frozen_prompt_payload_sha256 "
            "FROM insession_auxiliary_semantic_quorum_settlements "
            "WHERE session_id=? ORDER BY created_at DESC LIMIT 1",
            (session_id,),
        ).fetchone()
    assert settlement_row is not None
    settlement = semantic_store.get_auxiliary_semantic_quorum_settlement(
        session_id=str(settlement_row["session_id"]),
        task_id=str(settlement_row["insession_task_id"]),
        auxiliary_graph_id=str(settlement_row["auxiliary_graph_id"]),
        goal_id=str(settlement_row["goal_id"]),
        auxiliary_graph_revision=int(
            settlement_row["auxiliary_graph_revision"]
        ),
        frozen_prompt_payload_sha256=str(
            settlement_row["frozen_prompt_payload_sha256"]
        ),
    )
    assert settlement is not None
    candidate = settlement.requests[0].terminal_candidate_binding
    assert candidate is not None
    replan_request = AuxiliaryReplanningRequest(
        session_id=session_id,
        turn_id=turn_id,
        task_id=task_id,
        settlement=settlement,
    )
    real_validate = auxiliary_graphs._validate_revision_proposal

    def fail_after_candidate_archive(*_args, **_kwargs):
        raise RuntimeError("simulated revision failure after candidate archive")

    monkeypatch.setattr(
        auxiliary_graphs,
        "_validate_revision_proposal",
        fail_after_candidate_archive,
    )
    with pytest.raises(RuntimeError, match="after candidate archive"):
        run_auxiliary_replanning(
            replan_request,
            provider=planning_provider,
            ledger_store=store,
            emit=lambda _event: None,
        )

    with store._connect() as conn:
        rolled_back = conn.execute(
            "SELECT run.status AS run_status, run.reason AS run_reason, "
            "request.status AS request_status, node.status AS node_status, "
            "output.frozen_at "
            "FROM insession_work_runs AS run "
            "JOIN insession_work_run_verification_requests AS request "
            "ON request.verification_request_id=run.current_verification_request_id "
            "JOIN insession_auxiliary_node_states_v2 AS node "
            "ON node.auxiliary_graph_id=request.auxiliary_graph_id "
            "AND node.auxiliary_graph_revision=request.auxiliary_graph_revision "
            "AND node.auxiliary_node_id=request.auxiliary_node_id "
            "AND node.node_revision=request.node_revision "
            "JOIN insession_work_run_output_windows AS output "
            "ON output.work_run_id=run.work_run_id "
            "AND output.output_revision=request.output_revision "
            "WHERE run.work_run_id=?",
            (candidate.work_run_id,),
        ).fetchone()
    assert rolled_back is not None
    assert rolled_back["run_status"] == "active"
    assert rolled_back["run_reason"] == "verification_pending"
    assert rolled_back["request_status"] == "pending"
    assert rolled_back["node_status"] == "active"
    assert rolled_back["frozen_at"] is None

    monkeypatch.setattr(
        auxiliary_graphs,
        "_validate_revision_proposal",
        real_validate,
    )
    recovered = run_auxiliary_replanning(
        replan_request,
        provider=planning_provider,
        ledger_store=store,
        emit=lambda _event: None,
    )
    assert recovered.status is AuxiliaryReplanningStatus.REPLANNED
    assert recovered.model_replayed is True
    assert len(planning_calls) == 2

    with store._connect() as conn:
        archived = conn.execute(
            "SELECT status, reason, current_verification_request_id "
            "FROM insession_work_runs WHERE work_run_id=?",
            (candidate.work_run_id,),
        ).fetchone()
    assert archived is not None
    assert archived["status"] == "cancelled"
    assert archived["reason"] == "terminal_candidate_semantic_replan"
    assert archived["current_verification_request_id"] is None

    details = auxiliary_graph_store.get_auxiliary_graph_for_task(
        session_id=session_id,
        insession_task_id=task_id,
    )
    assert details is not None
    assert details.auxiliary_graph_revision == 3
    assert details.parent_auxiliary_graph_revision == 2
    assert details.reason == "verification_failed"


def test_application_retries_terminal_semantic_failure_inside_same_work_run_and_commits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_id, turn_id, task_id = _seed_task()
    _no_mounted_documents(monkeypatch, session_id)
    physical_planner = build_auxiliary_architect_structured_provider()
    planning_payloads: list[dict[str, object]] = []

    def planning_provider(system_prompt, user_content, **kwargs):
        planning_payloads.append(json.loads(user_content))
        return physical_planner(system_prompt, user_content, **kwargs)

    planning_provider = as_prepared_test_provider(planning_provider)

    physical_attempt = build_attempt_structured_provider(
        AuxiliaryApplicationPorts(
            model_ledger_store=store,
            emit=lambda _event: None
        ).work_run_model_profile
    )
    attempt_payloads: list[dict[str, object]] = []

    def attempt_provider(system_prompt, user_content, **kwargs):
        payload = json.loads(user_content)
        attempt_payloads.append(payload)
        result = physical_attempt(system_prompt, user_content, **kwargs)
        if payload.get("verification_feedback") is None:
            return result
        repaired = json.loads(result.reply)
        repaired["action"]["proposal"]["root"]["nodes"][0]["objective"] += (
            "（已依据整体验证反馈订正）"
        )
        return replace(
            result,
            reply=json.dumps(
                repaired,
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        )

    attempt_provider = as_prepared_test_provider(attempt_provider)

    semantic_provider = _SemanticProvider(failed_reviewer_ordinal=1)
    semantic_authority_factory, _bindings = _authority_factory()
    request = AuxiliaryApplicationRequest(
        session_id=session_id,
        turn_id=turn_id,
        task_id=task_id,
        max_effect_steps=12,
        max_autonomous_replans=3,
    )
    ports = AuxiliaryApplicationPorts(
        model_ledger_store=store,
        emit=lambda _event: None,
        planning_provider=planning_provider,
        attempt_provider=attempt_provider,
        semantic_provider=semantic_provider,
        semantic_model_call_authority_factory=semantic_authority_factory,
    )

    committed = run_auxiliary_application_to_boundary(request, ports=ports)

    assert committed.status is AuxiliaryApplicationStatus.COMMITTED
    assert committed.reason_code == "task_graph_revision_one_committed"
    assert committed.effect_steps == 4
    assert committed.replan_result is None
    assert len(planning_payloads) == 1
    assert planning_payloads[0]["replan_trigger"] is None
    assert len(semantic_provider.calls) == 2
    repaired_attempts = [
        payload
        for payload in attempt_payloads
        if payload.get("verification_feedback") is not None
    ]
    assert len(repaired_attempts) == 1
    downstream = repaired_attempts[0]["verification_feedback"][
        "downstream_results"
    ]
    assert downstream[0]["gate_id"] == "task_graph_semantic"
    assert downstream[0]["disposition"] == "retry_attempt"
    assert "requires revision" in downstream[0]["finding"]

    details = auxiliary_graph_store.get_auxiliary_graph_for_task(
        session_id=session_id,
        insession_task_id=task_id,
    )
    task = task_graph_store.get_insession_task_details(session_id, task_id)
    assert details is not None
    # 修订版 1 是主机引导；唯一的架构师提案创建修订版 2。语义重试不得创建修订版 3。
    assert details.auxiliary_graph_revision == 2
    assert details.parent_auxiliary_graph_revision == 1
    assert task is not None and task.current_graph_revision == 1
    assert planning_store.count_auxiliary_replan_triggers(
        session_id=session_id,
        task_id=task_id,
        goal_id=details.goal_id,
    ) == 0
    assert planning_store.get_active_auxiliary_replan_trigger(
        session_id=session_id,
        task_id=task_id,
    ) is None
    with store._connect() as conn:
        work_runs = int(
            conn.execute(
                "SELECT COUNT(*) FROM insession_work_runs WHERE session_id=?",
                (session_id,),
            ).fetchone()[0]
        )
        attempt_counts = tuple(
            int(row["attempt_count"])
            for row in conn.execute(
                "SELECT COUNT(attempt.attempt_id) AS attempt_count "
                "FROM insession_work_runs AS work_run "
                "JOIN insession_work_run_attempts AS attempt "
                "ON attempt.work_run_id=work_run.work_run_id "
                "WHERE work_run.session_id=? "
                "GROUP BY work_run.work_run_id ORDER BY attempt_count",
                (session_id,),
            ).fetchall()
        )
    assert work_runs == 2
    assert attempt_counts == (1, 2)

    def must_not_run(*_args, **_kwargs):
        raise AssertionError("committed application reentry reached a Provider")

    replayed = run_auxiliary_application_to_boundary(
        request,
        ports=AuxiliaryApplicationPorts(
            model_ledger_store=store,
            emit=lambda _event: None,
            planning_provider=must_not_run,
            attempt_provider=must_not_run,
            verification_provider=must_not_run,
            semantic_provider=must_not_run,
        ),
    )

    assert replayed.status is AuxiliaryApplicationStatus.COMMITTED
    assert replayed.reason_code == "task_graph_already_committed"
    assert replayed.effect_steps == 0
    assert len(planning_payloads) == 1
    assert len(semantic_provider.calls) == 2


def test_application_retries_terminal_production_failure_before_semantic_review(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_id, turn_id, task_id = _seed_task()
    _no_mounted_documents(monkeypatch, session_id)
    physical_attempt = build_attempt_structured_provider(
        AuxiliaryApplicationPorts(
            model_ledger_store=store,
            emit=lambda _event: None
        ).work_run_model_profile
    )
    attempt_payloads: list[dict[str, object]] = []

    def attempt_provider(system_prompt, user_content, **kwargs):
        payload = json.loads(user_content)
        attempt_payloads.append(payload)
        result = physical_attempt(system_prompt, user_content, **kwargs)
        if payload.get("verification_feedback") is None:
            return result
        repaired = json.loads(result.reply)
        repaired["action"]["proposal"]["root"]["nodes"][0]["objective"] += (
            " (production-gate coverage repaired)"
        )
        return replace(
            result,
            reply=json.dumps(repaired, separators=(",", ":")),
        )

    attempt_provider = as_prepared_test_provider(attempt_provider)

    real_evaluate = terminal.evaluate_task_graph_production
    production_calls = 0

    def fail_required_source_once(proposal, *, context):
        nonlocal production_calls
        production_calls += 1
        evaluated = real_evaluate(proposal, context=context)
        if production_calls != 1:
            return evaluated
        return evaluated.model_copy(
            update={
                "passed": False,
                "failure_codes": (
                    TaskGraphProductionFailureCode.REQUIRED_SOURCE_UNCOVERED,
                ),
                "findings": (),
                "evaluation_sha256": "a" * 64,
            }
        )

    monkeypatch.setattr(
        terminal,
        "evaluate_task_graph_production",
        fail_required_source_once,
    )
    semantic_provider = _SemanticProvider()
    semantic_authority_factory, _bindings = _authority_factory()
    committed = run_auxiliary_application_to_boundary(
        AuxiliaryApplicationRequest(
            session_id=session_id,
            turn_id=turn_id,
            task_id=task_id,
            max_effect_steps=12,
        ),
        ports=AuxiliaryApplicationPorts(
            model_ledger_store=store,
            emit=lambda _event: None,
            attempt_provider=attempt_provider,
            semantic_provider=semantic_provider,
            semantic_model_call_authority_factory=semantic_authority_factory,
        ),
    )

    assert committed.status is AuxiliaryApplicationStatus.COMMITTED
    assert committed.reason_code == "task_graph_revision_one_committed"
    assert production_calls == 2
    assert len(semantic_provider.calls) == 1
    repaired = [
        payload
        for payload in attempt_payloads
        if payload.get("verification_feedback") is not None
    ]
    assert len(repaired) == 1
    downstream = repaired[0]["verification_feedback"]["downstream_results"]
    assert downstream[0]["gate_id"] == "task_graph_production"
    assert downstream[0]["disposition"] == "retry_attempt"
    assert "required_source_uncovered" in downstream[0]["finding"]
