from __future__ import annotations

import hashlib
import json

import pytest

from personagraph.model_io.gateway import ModelResult
from personagraph.l2.auxiliary_execution import (
    application as auxiliary_application,
)
from personagraph.l2.auxiliary_execution.planning import (
    goal_successor_controller as successor_controller,
)
from personagraph.l2.task_graph import (
    InSessionTaskAcceptanceProposal,
    InSessionTaskMatchesProposal,
)
from personagraph.l2.auxiliary_execution.application import (
    AuxiliaryApplicationPorts,
    AuxiliaryApplicationRequest,
    AuxiliaryApplicationStatus,
    run_auxiliary_application_to_boundary,
)
from personagraph.l2.auxiliary_execution.planning.architect_adapter import (
    build_terminal_only_auxiliary_graph_bootstrap_proposal,
)
from personagraph.l2.auxiliary_execution.planning.goal_successor_controller import (
    AuxiliaryGoalSuccessorPlanningError,
    AuxiliaryGoalSuccessorPlanningRequest,
    AuxiliaryGoalSuccessorPlanningResult,
    AuxiliaryGoalSuccessorPlanningStatus,
    derive_auxiliary_goal_successor_ids,
    run_auxiliary_goal_successor_planning,
)
from personagraph.l2.auxiliary_execution.planning.goal_supersede_controller import (
    AuxiliaryGoalSupersedeRequest,
    run_auxiliary_goal_supersede,
)
from personagraph.l2.auxiliary_execution.planning.model_provider import (
    build_auxiliary_architect_structured_provider,
)
from personagraph.runtime.model_calls import (
    RuntimeModelCallWaitingExternal,
)
from personagraph.session import store
from personagraph.session.l2_store import task_graph as task_graph_store
from personagraph.session.l2_store import auxiliary_graph as auxiliary_graph_store
from personagraph.session.l2_store import planning as planning_store
from tests.runtime.test_auxiliary_goal_supersede_controller import (
    _accept_target_change_turn,
)
from tests.runtime.test_auxiliary_positive_planning_controller import (
    _accept_continuation_turn,
)
from tests.runtime.test_auxiliary_planning_controller import (
    _no_mounted_documents,
)
from tests.session.test_auxiliary_goal_supersede_persistence import (
    _commit_task_graph_one,
    _seed_goal,
)
from tests.session.test_auxiliary_graph_persistence import _seed_task_shell
from tests.session.test_auxiliary_graph_persistence import _window_revision


def _base_drift_receipt(prefix: str):
    session_id, turn_id, task_id, details = _seed_goal(prefix)
    task = _commit_task_graph_one(
        session_id=session_id,
        turn_id=turn_id,
        task_id=task_id,
    )
    superseded = run_auxiliary_goal_supersede(
        AuxiliaryGoalSupersedeRequest(
            session_id=session_id,
            turn_id=turn_id,
            task_id=task_id,
            reason=planning_store.PlanningGoalSupersedeReason.BASE_DRIFT,
            auxiliary_graph_id=details.auxiliary_graph_id,
            goal_id=details.goal_id,
            expected_task_state_version=task.task_state_version,
            expected_control_state_version=details.control_state_version,
            expected_goal_state_version=details.goal_state_version,
            expected_revision_state_version=details.revision_state_version,
            expected_budget_state_version=details.budget_state_version,
            expected_current_auxiliary_graph_revision=(
                details.auxiliary_graph_revision
            ),
            expected_base_task_graph_revision=details.base_task_graph_revision,
            observed_task_graph_revision=task.committed_graph_revision,
        )
    )
    return session_id, turn_id, task_id, details, superseded.store_result.receipt


def _target_change_receipt(prefix: str):
    session_id, creation_turn_id, task_id, details = _seed_goal(prefix)
    turn_id, user_text = _accept_target_change_turn(
        session_id=session_id,
        prior_turn_id=creation_turn_id,
        task_id=task_id,
    )
    task = task_graph_store.get_insession_task_details(session_id, task_id)
    assert task is not None
    replacement = "比较论文的实验设计与消融结果"
    superseded = run_auxiliary_goal_supersede(
        AuxiliaryGoalSupersedeRequest(
            session_id=session_id,
            turn_id=turn_id,
            task_id=task_id,
            reason=planning_store.PlanningGoalSupersedeReason.USER_TARGET_CHANGED,
            auxiliary_graph_id=details.auxiliary_graph_id,
            goal_id=details.goal_id,
            expected_task_state_version=task.task_state_version,
            expected_control_state_version=details.control_state_version,
            expected_goal_state_version=details.goal_state_version,
            expected_revision_state_version=details.revision_state_version,
            expected_budget_state_version=details.budget_state_version,
            expected_current_auxiliary_graph_revision=(
                details.auxiliary_graph_revision
            ),
            expected_base_task_graph_revision=details.base_task_graph_revision,
            observed_task_graph_revision=task.current_graph_revision,
            replacement_objective=replacement,
            source_start=0,
            source_end=len(user_text),
            source_sha256=hashlib.sha256(user_text.encode()).hexdigest(),
        )
    )
    return (
        session_id,
        turn_id,
        task_id,
        details,
        superseded.store_result.receipt,
        replacement,
    )


def _accept_explicit_target_change_turn(
    *,
    session_id: str,
    prior_turn_id: str,
    task_id: str,
) -> tuple[str, str, str]:
    finalized = store.finalize_turn_execution(
        session_id=session_id,
        turn_id=prior_turn_id,
        expected_window_revision=_window_revision(session_id),
        processing_level="L2",
        assistant_content="已记录初始规划。",
        post_commit_job_kinds=(),
    )
    store.release_turn_execution_window(
        session_id=session_id,
        turn_id=prior_turn_id,
        expected_window_revision=int(finalized["window"]["state_version"]),
    )
    user_text = "改一下目标：只比较论文的实验设计与消融结果"
    excerpt = "目标：只比较论文的实验设计与消融结果"
    replacement = "比较论文的实验设计与消融结果"
    accepted = store.accept_turn_execution(
        session_id=session_id,
        client_request_id="goal-successor-explicit-target-change",
        source="runtime_test",
        user_text=user_text,
        lease_owner="goal-successor-controller-test",
    )
    turn_id = str(accepted["turn"]["turn_id"])
    task_graph_store.apply_insession_task_matches(
        session_id=session_id,
        source_turn_id=turn_id,
        apply_id="goal-successor-explicit-target-link",
        proposal=InSessionTaskMatchesProposal.model_validate(
            {
                "task_matches": [
                    {
                        "match_type": "existing_root_target_change",
                        "insession_task_id": task_id,
                        "replacement_objective": replacement,
                        "source_excerpt": excerpt,
                        "execute_current": True,
                    }
                ]
            }
        ),
        exposed_catalog_ids=(task_id,),
        expected_window_revision=_window_revision(session_id),
    )
    return turn_id, excerpt, replacement


@pytest.mark.parametrize("kind", ["base_drift", "user_target_changed"])
def test_authenticated_receipt_bootstraps_fresh_goal_then_model_plans_and_replays(
    kind: str,
) -> None:
    if kind == "base_drift":
        session_id, turn_id, task_id, before, receipt = _base_drift_receipt(
            "goal-successor-drift"
        )
    else:
        (
            session_id,
            turn_id,
            task_id,
            before,
            receipt,
            _replacement,
        ) = _target_change_receipt("goal-successor-target")

    ids = derive_auxiliary_goal_successor_ids(receipt)
    physical = build_auxiliary_architect_structured_provider()
    calls: list[dict[str, object]] = []

    def provider(*args: object, **kwargs: object) -> ModelResult:
        payload = json.loads(str(args[1]))
        calls.append(payload)
        assert payload["goal"]["goal_id"] == ids.goal_id
        assert payload["goal"]["objective"] == receipt.next_goal_objective
        assert payload["current_revision"]["auxiliary_graph_revision"] == (
            receipt.superseded_auxiliary_graph_revision + 1
        )
        assert payload["current_revision"]["base_task_graph_revision"] == (
            receipt.next_base_task_graph_revision
        )
        return physical(*args, **kwargs)  # type: ignore[arg-type]

    request = AuxiliaryGoalSuccessorPlanningRequest(
        session_id=session_id,
        turn_id=turn_id,
        task_id=task_id,
        supersede_receipt=receipt,
    )
    planned = run_auxiliary_goal_successor_planning(
        request,
        ledger_store=store,
        emit=lambda _event: None,
        provider=provider,
    )

    assert planned.status is AuxiliaryGoalSuccessorPlanningStatus.PLANNED
    assert planned.bootstrapped is True
    assert planned.bootstrap_commit is not None
    assert planned.bootstrap_commit.committed_auxiliary_graph_revision == (
        before.auxiliary_graph_revision + 1
    )
    assert planned.bootstrap_commit.goal_id == ids.goal_id
    assert planned.revision_commit is not None
    assert planned.revision_commit.committed_auxiliary_graph_revision == (
        before.auxiliary_graph_revision + 2
    )
    assert planned.architect_decision is not None
    assert planned.architect_decision.logical_call_id == ids.logical_call_id
    assert ids.logical_call_id != f"{receipt.superseded_goal_id}:model"
    assert planned.details.goal_id == ids.goal_id
    assert planned.details.base_task_graph_revision == (
        receipt.next_base_task_graph_revision
    )
    assert planned.details.target_task_graph_revision == (
        receipt.next_target_task_graph_revision
    )
    assert planned.details.goal_objective == receipt.next_goal_objective
    assert all(
        node.local_node_key != "goal_successor_bootstrap"
        for node in planned.details.nodes
    )

    replayed = run_auxiliary_goal_successor_planning(
        request,
        ledger_store=store,
        emit=lambda _event: None,
        provider=lambda *_args, **_kwargs: pytest.fail(
            "completed successor planning reached Provider"
        ),
    )
    assert (
        replayed.status
        is AuxiliaryGoalSuccessorPlanningStatus.ALREADY_PLANNED
    )
    assert replayed.details == planned.details
    assert replayed.model_replayed is True
    assert len(calls) == 1


def test_successor_replays_succeeded_architect_across_turn_after_precommit_loss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_id, source_turn_id, task_id, _before, receipt = _base_drift_receipt(
        "goal-successor-cross-turn-precommit"
    )
    ids = derive_auxiliary_goal_successor_ids(receipt)
    provider = build_auxiliary_architect_structured_provider()
    provider_calls = 0



    def count_provider(*args: object, **kwargs: object) -> ModelResult:
        nonlocal provider_calls
        provider_calls += 1
        return provider(*args, **kwargs)  # type: ignore[arg-type]

    real_commit = auxiliary_graph_store.commit_auxiliary_graph_revision

    def lose_before_architect_commit(**kwargs: object):
        if kwargs["apply_id"] == ids.architect_apply_id:
            raise RuntimeError("simulated successor precommit response loss")
        return real_commit(**kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(
        auxiliary_graph_store,
        "commit_auxiliary_graph_revision",
        lose_before_architect_commit,
    )
    with pytest.raises(RuntimeError, match="successor precommit response loss"):
        run_auxiliary_goal_successor_planning(
            AuxiliaryGoalSuccessorPlanningRequest(
                session_id=session_id,
                turn_id=source_turn_id,
                task_id=task_id,
                supersede_receipt=receipt,
            ),
            ledger_store=store,
            emit=lambda _event: None,
            provider=count_provider,
        )
    assert provider_calls == 1

    continuation_turn_id = _accept_continuation_turn(
        session_id=session_id,
        prior_turn_id=source_turn_id,
        task_id=task_id,
        execute_current=True,
    )
    monkeypatch.setattr(
        auxiliary_graph_store,
        "commit_auxiliary_graph_revision",
        real_commit,
    )
    recovered = run_auxiliary_goal_successor_planning(
        AuxiliaryGoalSuccessorPlanningRequest(
            session_id=session_id,
            turn_id=continuation_turn_id,
            task_id=task_id,
            supersede_receipt=receipt,
        ),
        ledger_store=store,
        emit=lambda _event: None,
        provider=lambda *_args, **_kwargs: pytest.fail(
            "settled successor Architect call reached Provider after handoff"
        ),
    )

    assert recovered.status is AuxiliaryGoalSuccessorPlanningStatus.PLANNED
    assert recovered.model_replayed is True
    assert recovered.bootstrapped is False
    assert recovered.supersede_receipt == receipt
    logical = store.get_runtime_model_logical_call(
        session_id=session_id,
        logical_call_id=ids.logical_call_id,
    )
    assert logical is not None
    assert logical.request.invocation_turn_id == source_turn_id
    assert len(logical.physical_attempts) == 1
    assert provider_calls == 1


def test_successor_retryable_architect_attempt_uses_same_logical_call_on_later_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_id, source_turn_id, task_id, _before, receipt = _base_drift_receipt(
        "goal-successor-cross-turn-retryable"
    )
    ids = derive_auxiliary_goal_successor_ids(receipt)
    real_request = successor_controller.request_auxiliary_graph_architect

    def leave_retryable(
        _request: object,
        *,
        invocation_turn_id: str,
        provider: object,
        emit: object,
        deadline: object,
        durable_call: object,
    ) -> object:
        del provider, emit, deadline
        assert invocation_turn_id == source_turn_id
        assert durable_call is not None
        durable_call.require_current_state()
        durable_call.reserve(turn_id=invocation_turn_id)
        physical = durable_call.begin_physical_attempt(
            turn_id=invocation_turn_id,
            max_physical_attempts=(
                durable_call.logical_request.max_physical_attempts
            ),
            output_repair_enabled=True,
        )
        durable_call.settle_physical_attempt(
            turn_id=invocation_turn_id,
            physical=physical,
            outcome="retryable_failure",
            result_fingerprint="a" * 64,
            error_code="MODEL_RATE_LIMIT",
        )
        raise RuntimeError("simulated host boundary after retryable attempt")

    monkeypatch.setattr(
        successor_controller,
        "request_auxiliary_graph_architect",
        leave_retryable,
    )
    with pytest.raises(RuntimeError, match="retryable attempt"):
        run_auxiliary_goal_successor_planning(
            AuxiliaryGoalSuccessorPlanningRequest(
                session_id=session_id,
                turn_id=source_turn_id,
                task_id=task_id,
                supersede_receipt=receipt,
            ),
            ledger_store=store,
            emit=lambda _event: None,
            provider=lambda *_args, **_kwargs: pytest.fail(
                "synthetic retry setup reached Provider"
            ),
        )

    continuation_turn_id = _accept_continuation_turn(
        session_id=session_id,
        prior_turn_id=source_turn_id,
        task_id=task_id,
        execute_current=True,
    )
    monkeypatch.setattr(
        successor_controller,
        "request_auxiliary_graph_architect",
        real_request,
    )
    provider_calls = 0
    provider = build_auxiliary_architect_structured_provider()

    def count_provider(*args: object, **kwargs: object) -> ModelResult:
        nonlocal provider_calls
        provider_calls += 1
        return provider(*args, **kwargs)  # type: ignore[arg-type]

    recovered = run_auxiliary_goal_successor_planning(
        AuxiliaryGoalSuccessorPlanningRequest(
            session_id=session_id,
            turn_id=continuation_turn_id,
            task_id=task_id,
            supersede_receipt=receipt,
        ),
        ledger_store=store,
        emit=lambda _event: None,
        provider=count_provider,
    )

    assert recovered.status is AuxiliaryGoalSuccessorPlanningStatus.PLANNED
    assert recovered.model_attempts == 2
    assert recovered.model_replayed is False
    logical = store.get_runtime_model_logical_call(
        session_id=session_id,
        logical_call_id=ids.logical_call_id,
    )
    assert logical is not None
    assert logical.request.invocation_turn_id == source_turn_id
    assert len(logical.physical_attempts) == 2
    assert logical.physical_attempts[1].request.started_turn_id == continuation_turn_id
    assert provider_calls == 1


@pytest.mark.parametrize("outcome", ["pending", "uncertain"])
def test_successor_pending_or_uncertain_architect_waits_external_after_turn_handoff(
    monkeypatch: pytest.MonkeyPatch,
    outcome: str,
) -> None:
    session_id, source_turn_id, task_id, _before, receipt = _base_drift_receipt(
        f"goal-successor-cross-turn-{outcome}"
    )
    real_request = successor_controller.request_auxiliary_graph_architect

    def leave_unreconciled(
        _request: object,
        *,
        invocation_turn_id: str,
        provider: object,
        emit: object,
        deadline: object,
        durable_call: object,
    ) -> object:
        del provider, emit, deadline
        assert durable_call is not None
        durable_call.require_current_state()
        durable_call.reserve(turn_id=invocation_turn_id)
        physical = durable_call.begin_physical_attempt(
            turn_id=invocation_turn_id,
            max_physical_attempts=(
                durable_call.logical_request.max_physical_attempts
            ),
            output_repair_enabled=True,
        )
        if outcome == "uncertain":
            durable_call.settle_physical_attempt(
                turn_id=invocation_turn_id,
                physical=physical,
                outcome="uncertain",
                result_fingerprint="b" * 64,
                error_code="MODEL_RESPONSE_UNCERTAIN",
            )
        raise RuntimeError(f"simulated {outcome} provider boundary")

    monkeypatch.setattr(
        successor_controller,
        "request_auxiliary_graph_architect",
        leave_unreconciled,
    )
    with pytest.raises(RuntimeError, match=outcome):
        run_auxiliary_goal_successor_planning(
            AuxiliaryGoalSuccessorPlanningRequest(
                session_id=session_id,
                turn_id=source_turn_id,
                task_id=task_id,
                supersede_receipt=receipt,
            ),
            ledger_store=store,
            emit=lambda _event: None,
            provider=lambda *_args, **_kwargs: pytest.fail(
                "synthetic unreconciled setup reached Provider"
            ),
        )
    continuation_turn_id = _accept_continuation_turn(
        session_id=session_id,
        prior_turn_id=source_turn_id,
        task_id=task_id,
        execute_current=True,
    )
    monkeypatch.setattr(
        successor_controller,
        "request_auxiliary_graph_architect",
        real_request,
    )

    with pytest.raises(RuntimeModelCallWaitingExternal):
        run_auxiliary_goal_successor_planning(
            AuxiliaryGoalSuccessorPlanningRequest(
                session_id=session_id,
                turn_id=continuation_turn_id,
                task_id=task_id,
                supersede_receipt=receipt,
            ),
            ledger_store=store,
            emit=lambda _event: None,
            provider=lambda *_args, **_kwargs: pytest.fail(
                f"{outcome} successor call was blindly resent"
            ),
        )


def test_successor_rejects_self_consistent_but_unstored_supersede_receipt() -> None:
    session_id, turn_id, task_id, _before, receipt = _base_drift_receipt(
        "goal-successor-forged"
    )
    forged_values = receipt.model_dump(
        mode="python",
        exclude={"receipt_sha256"},
    )
    forged_values["next_goal_objective"] = "forged replacement objective"
    forged = planning_store.PlanningGoalSupersedeReceipt.create(**forged_values)

    with pytest.raises(
        AuxiliaryGoalSuccessorPlanningError,
        match="authenticated",
    ):
        run_auxiliary_goal_successor_planning(
            AuxiliaryGoalSuccessorPlanningRequest(
                session_id=session_id,
                turn_id=turn_id,
                task_id=task_id,
                supersede_receipt=forged,
            ),
            ledger_store=store,
            emit=lambda _event: None,
            provider=lambda *_args, **_kwargs: pytest.fail(
                "forged receipt reached Provider"
            ),
        )


def test_application_detects_pending_target_change_receipt_and_runs_successor() -> None:
    (
        session_id,
        turn_id,
        task_id,
        _before,
        receipt,
        replacement,
    ) = _target_change_receipt("goal-successor-app-pending")
    physical = build_auxiliary_architect_structured_provider()

    result = run_auxiliary_application_to_boundary(
        AuxiliaryApplicationRequest(
            session_id=session_id,
            turn_id=turn_id,
            task_id=task_id,
            max_effect_steps=1,
        ),
        ports=AuxiliaryApplicationPorts(
            model_ledger_store=store,
            emit=lambda _event: None,
            planning_provider=physical,
        ),
    )

    assert result.status is AuxiliaryApplicationStatus.STEP_LIMIT_REACHED
    assert isinstance(
        result.planning_result,
        AuxiliaryGoalSuccessorPlanningResult,
    )
    assert result.planning_result.status is (
        AuxiliaryGoalSuccessorPlanningStatus.PLANNED
    )
    assert result.planning_result.details.goal_objective == replacement
    assert result.planning_result.supersede_receipt == receipt


def test_pending_target_change_receipt_starts_successor_on_later_turn_without_rebinding_source(
) -> None:
    (
        session_id,
        source_turn_id,
        task_id,
        _before,
        receipt,
        replacement,
    ) = _target_change_receipt("goal-successor-cross-turn-pending")
    assert receipt.source_binding is not None
    source_binding = receipt.source_binding
    continuation_turn_id = _accept_continuation_turn(
        session_id=session_id,
        prior_turn_id=source_turn_id,
        task_id=task_id,
        execute_current=True,
    )
    provider = build_auxiliary_architect_structured_provider()
    calls = 0

    def count_provider(*args: object, **kwargs: object) -> ModelResult:
        nonlocal calls
        calls += 1
        return provider(*args, **kwargs)  # type: ignore[arg-type]

    result = run_auxiliary_application_to_boundary(
        AuxiliaryApplicationRequest(
            session_id=session_id,
            turn_id=continuation_turn_id,
            task_id=task_id,
            max_effect_steps=1,
        ),
        ports=AuxiliaryApplicationPorts(
            model_ledger_store=store,
            emit=lambda _event: None,
            planning_provider=count_provider,
        ),
    )

    assert result.status is AuxiliaryApplicationStatus.STEP_LIMIT_REACHED
    assert isinstance(result.planning_result, AuxiliaryGoalSuccessorPlanningResult)
    assert result.planning_result.status is AuxiliaryGoalSuccessorPlanningStatus.PLANNED
    assert result.planning_result.details.goal_objective == replacement
    assert result.planning_result.supersede_receipt == receipt
    assert result.planning_result.supersede_receipt.invocation_turn_id == source_turn_id
    assert result.planning_result.supersede_receipt.source_binding == source_binding
    assert source_binding.source_turn_id == source_turn_id
    logical = store.get_runtime_model_logical_call(
        session_id=session_id,
        logical_call_id=result.planning_result.ids.logical_call_id,
    )
    assert logical is not None
    assert logical.request.invocation_turn_id == continuation_turn_id
    assert calls == 1


def test_application_consumes_guarded_lane_target_change_without_manual_receipt() -> None:
    session_id, creation_turn_id, task_id, before = _seed_goal(
        "goal-successor-app-explicit-target"
    )
    turn_id, excerpt, replacement = _accept_explicit_target_change_turn(
        session_id=session_id,
        prior_turn_id=creation_turn_id,
        task_id=task_id,
    )

    result = run_auxiliary_application_to_boundary(
        AuxiliaryApplicationRequest(
            session_id=session_id,
            turn_id=turn_id,
            task_id=task_id,
            max_effect_steps=2,
        ),
        ports=AuxiliaryApplicationPorts(
            model_ledger_store=store,
            emit=lambda _event: None,
            planning_provider=build_auxiliary_architect_structured_provider(),
        ),
    )

    assert result.status is AuxiliaryApplicationStatus.STEP_LIMIT_REACHED
    assert isinstance(result.planning_result, AuxiliaryGoalSuccessorPlanningResult)
    receipt = result.planning_result.supersede_receipt
    assert receipt.reason is planning_store.PlanningGoalSupersedeReason.USER_TARGET_CHANGED
    assert receipt.superseded_goal_id == before.goal_id
    assert receipt.next_goal_objective == replacement
    assert receipt.source_binding is not None
    assert receipt.source_binding.source_turn_id == turn_id
    assert receipt.source_binding.source_sha256 == hashlib.sha256(excerpt.encode()).hexdigest()
    assert result.planning_result.details.goal_objective == replacement

    replayed = run_auxiliary_application_to_boundary(
        AuxiliaryApplicationRequest(
            session_id=session_id,
            turn_id=turn_id,
            task_id=task_id,
            max_effect_steps=1,
        ),
        ports=AuxiliaryApplicationPorts(
            model_ledger_store=store,
            emit=lambda _event: None,
            planning_provider=lambda *_args, **_kwargs: pytest.fail(
                "replayed target change reached the Architect Provider"
            ),
        ),
    )
    assert replayed.status is AuxiliaryApplicationStatus.STEP_LIMIT_REACHED
    with store._connect() as conn:
        receipt_count = int(
            conn.execute(
                "SELECT COUNT(*) FROM "
                "insession_auxiliary_graph_revision_apply_receipts_v2 "
                "WHERE operation='supersede_goal' AND session_id=? "
                "AND insession_task_id=? AND invocation_turn_id=?",
                (session_id, task_id, turn_id),
            ).fetchone()[0]
        )
    assert receipt_count == 1


def test_application_never_infers_target_change_from_ordinary_existing_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_id, creation_turn_id, task_id, before = _seed_goal(
        "goal-successor-app-no-inference"
    )
    turn_id, _user_text = _accept_target_change_turn(
        session_id=session_id,
        prior_turn_id=creation_turn_id,
        task_id=task_id,
    )
    monkeypatch.setattr(
        auxiliary_application,
        "run_auxiliary_goal_supersede",
        lambda *_args, **_kwargs: pytest.fail(
            "ordinary existing_root was inferred as a target change"
        ),
    )

    result = run_auxiliary_application_to_boundary(
        AuxiliaryApplicationRequest(
            session_id=session_id,
            turn_id=turn_id,
            task_id=task_id,
            max_effect_steps=1,
        ),
        ports=AuxiliaryApplicationPorts(
            model_ledger_store=store,
            emit=lambda _event: None,
            planning_provider=build_auxiliary_architect_structured_provider(),
        ),
    )

    assert result.status in {
        AuxiliaryApplicationStatus.STEP_LIMIT_REACHED,
        AuxiliaryApplicationStatus.FAILED,
    }
    current = auxiliary_graph_store.get_auxiliary_graph_for_task(
        session_id=session_id,
        insession_task_id=task_id,
    )
    assert current is not None
    assert current.goal_id == before.goal_id
    with store._connect() as conn:
        receipt_count = int(
            conn.execute(
                "SELECT COUNT(*) FROM "
                "insession_auxiliary_graph_revision_apply_receipts_v2 "
                "WHERE operation='supersede_goal' AND session_id=? "
                "AND insession_task_id=? AND invocation_turn_id=?",
                (session_id, task_id, turn_id),
            ).fetchone()[0]
        )
    assert receipt_count == 0


def test_application_follows_explicit_base_drift_supersede_route_into_successor() -> None:
    session_id, turn_id, task_id = _seed_task_shell()
    task = task_graph_store.get_insession_task_details(session_id, task_id)
    assert task is not None
    bootstrap = build_terminal_only_auxiliary_graph_bootstrap_proposal(
        terminal_node_key="bootstrap_terminal",
        title="Establish durable planning authority",
        objective="Require an Architect replacement before execution.",
        source_anchor_ids=("task_creation_source",),
        acceptance_criteria=(
            InSessionTaskAcceptanceProposal(
                acceptance_id="architect_revision_committed",
                criterion="The Architect replaces this non-executable shell.",
                source_anchor_ids=("task_creation_source",),
            ),
        ),
    )
    initial = auxiliary_graph_store.commit_auxiliary_graph_revision(
        session_id=session_id,
        turn_id=turn_id,
        insession_task_id=task_id,
        expected_task_state_version=task.task_state_version,
        expected_base_task_graph_revision=None,
        expected_control_state_version=None,
        expected_current_auxiliary_graph_revision=None,
        apply_id="goal-successor-app-initial-shell",
        goal_objective=task.objective,
        proposal=bootstrap,
        authority_context={"anchors": []},
        budget_profile={"profile_id": "planning-test-v1"},
        auxiliary_graph_id="goal-successor-app-graph",
        goal_id="goal-successor-app-old-goal",
    )
    task_graph = _commit_task_graph_one(
        session_id=session_id,
        turn_id=turn_id,
        task_id=task_id,
    )
    assert task_graph.committed_graph_revision == 1
    physical = build_auxiliary_architect_structured_provider()
    calls = 0

    def provider(
        system_prompt: str,
        user_content: str,
        *,
        model_call_id: str,
        purpose: str,
    ) -> ModelResult:
        nonlocal calls
        calls += 1
        payload = json.loads(user_content)
        if payload["goal"]["goal_id"] == initial.goal_id:
            return ModelResult(
                reply=json.dumps(
                    {
                        "disposition": "supersede_and_rebase",
                        "expected_current_auxiliary_graph_revision": 1,
                        "revision_reason": None,
                        "structure": None,
                        "explanation": (
                            "The TaskGraph base advanced after this goal was frozen."
                        ),
                        "blocking_gap_ids": [],
                        "requested_user_question": None,
                        "failure_reason": None,
                    }
                ),
                provider="mock",
                model="mock-structured",
                latency_ms=1,
                model_call_id=model_call_id,
                purpose=purpose,
            )
        return physical(
            system_prompt,
            user_content,
            model_call_id=model_call_id,
            purpose=purpose,
        )

    result = run_auxiliary_application_to_boundary(
        AuxiliaryApplicationRequest(
            session_id=session_id,
            turn_id=turn_id,
            task_id=task_id,
            max_effect_steps=2,
        ),
        ports=AuxiliaryApplicationPorts(
            model_ledger_store=store,
            emit=lambda _event: None,
            planning_provider=provider,
        ),
    )

    assert result.status is AuxiliaryApplicationStatus.STEP_LIMIT_REACHED
    assert isinstance(
        result.planning_result,
        AuxiliaryGoalSuccessorPlanningResult,
    )
    assert result.planning_result.status is (
        AuxiliaryGoalSuccessorPlanningStatus.PLANNED
    )
    assert result.planning_result.details.base_task_graph_revision == 1
    assert result.planning_result.details.target_task_graph_revision == 2
    assert result.planning_result.bootstrap_commit is not None
    assert result.planning_result.bootstrap_commit.committed_auxiliary_graph_revision == 2
    assert result.planning_result.revision_commit is not None
    assert result.planning_result.revision_commit.committed_auxiliary_graph_revision == 3
    assert calls == 2


def test_application_supersedes_persisted_plan_on_base_drift_before_frontier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_id, turn_id, task_id = _seed_task_shell()
    _no_mounted_documents(monkeypatch, session_id)
    provider = build_auxiliary_architect_structured_provider()
    initial = run_auxiliary_application_to_boundary(
        AuxiliaryApplicationRequest(
            session_id=session_id,
            turn_id=turn_id,
            task_id=task_id,
            max_effect_steps=1,
        ),
        ports=AuxiliaryApplicationPorts(
            model_ledger_store=store,
            emit=lambda _event: None,
            planning_provider=provider,
        ),
    )
    assert initial.status is AuxiliaryApplicationStatus.STEP_LIMIT_REACHED
    assert initial.planning_result is not None
    stale = initial.planning_result.details
    assert stale.auxiliary_graph_revision == 2
    assert stale.base_task_graph_revision is None
    assert stale.goal_status == "active"

    advanced = _commit_task_graph_one(
        session_id=session_id,
        turn_id=turn_id,
        task_id=task_id,
    )
    assert advanced.committed_graph_revision == 1
    successor_calls = 0

    def successor_provider(*args: object, **kwargs: object) -> ModelResult:
        nonlocal successor_calls
        successor_calls += 1
        payload = json.loads(str(args[1]))
        assert payload["goal"]["goal_id"] != stale.goal_id
        return provider(*args, **kwargs)  # type: ignore[arg-type]

    recovered = run_auxiliary_application_to_boundary(
        AuxiliaryApplicationRequest(
            session_id=session_id,
            turn_id=turn_id,
            task_id=task_id,
            max_effect_steps=2,
        ),
        ports=AuxiliaryApplicationPorts(
            model_ledger_store=store,
            emit=lambda _event: None,
            planning_provider=successor_provider,
        ),
    )

    assert recovered.status is AuxiliaryApplicationStatus.STEP_LIMIT_REACHED
    assert recovered.reason_code == "application_effect_step_limit_reached"
    assert isinstance(
        recovered.planning_result,
        AuxiliaryGoalSuccessorPlanningResult,
    )
    assert recovered.planning_result.status is (
        AuxiliaryGoalSuccessorPlanningStatus.PLANNED
    )
    receipt = recovered.planning_result.supersede_receipt
    assert receipt.reason is planning_store.PlanningGoalSupersedeReason.BASE_DRIFT
    assert receipt.superseded_goal_id == stale.goal_id
    assert receipt.previous_base_task_graph_revision is None
    assert receipt.observed_task_graph_revision == 1
    assert recovered.planning_result.details.base_task_graph_revision == 1
    assert recovered.planning_result.details.auxiliary_graph_revision == 4
    assert successor_calls == 1
