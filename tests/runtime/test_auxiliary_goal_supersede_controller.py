from __future__ import annotations

import hashlib

from personagraph.l2.task_graph.task_matching import (
    InSessionTaskMatchesProposal,
)
from personagraph.l2.auxiliary_execution.planning.goal_supersede_controller import (
    AuxiliaryGoalSupersedeControllerStatus,
    AuxiliaryGoalSupersedeRequest,
    run_auxiliary_goal_supersede,
)
from personagraph.session import store
from personagraph.session.l2_store import task_graph as task_graph_store
from personagraph.session.l2_store import planning as planning_store
from tests.session.test_auxiliary_goal_supersede_persistence import (
    _commit_task_graph_one,
    _seed_goal,
)
from tests.session.test_auxiliary_graph_persistence import _window_revision


def _accept_target_change_turn(
    *,
    session_id: str,
    prior_turn_id: str,
    task_id: str,
) -> tuple[str, str]:
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
    accepted = store.accept_turn_execution(
        session_id=session_id,
        client_request_id="goal-supersede-target-change",
        source="runtime_test",
        user_text=user_text,
        lease_owner="goal-supersede-controller-test",
    )
    turn_id = str(accepted["turn"]["turn_id"])
    task_graph_store.apply_insession_task_matches(
        session_id=session_id,
        source_turn_id=turn_id,
        apply_id="goal-supersede-target-link",
        proposal=InSessionTaskMatchesProposal.model_validate(
            {
                "task_matches": [
                    {
                        "match_type": "existing_root",
                        "insession_task_id": task_id,
                        "source_excerpt": user_text,
                        "execute_current": True,
                    }
                ]
            }
        ),
        exposed_catalog_ids=(task_id,),
        expected_window_revision=_window_revision(session_id),
    )
    return turn_id, user_text


def test_user_target_change_controller_returns_authenticated_replayable_receipt() -> None:
    session_id, creation_turn_id, task_id, details = _seed_goal(
        "supersede-controller"
    )
    turn_id, user_text = _accept_target_change_turn(
        session_id=session_id,
        prior_turn_id=creation_turn_id,
        task_id=task_id,
    )
    task = task_graph_store.get_insession_task_details(session_id, task_id)
    assert task is not None
    request = AuxiliaryGoalSupersedeRequest(
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
        replacement_objective="比较论文的实验设计与消融结果",
        source_start=0,
        source_end=len(user_text),
        source_sha256=hashlib.sha256(user_text.encode()).hexdigest(),
    )

    applied = run_auxiliary_goal_supersede(request)

    assert (
        applied.status
        is AuxiliaryGoalSupersedeControllerStatus.SUPERSEDED
    )
    assert applied.store_result.status == "applied"
    receipt = applied.store_result.receipt
    assert receipt.next_goal_objective == "比较论文的实验设计与消融结果"
    assert receipt.source_binding is not None
    assert receipt.source_binding.source_turn_id == turn_id
    assert receipt.source_binding.source_sha256 == request.source_sha256
    assert receipt.next_base_task_graph_revision is None
    assert receipt.next_target_task_graph_revision == 1
    assert receipt.receipt_sha256 == applied.receipt_sha256

    replayed = run_auxiliary_goal_supersede(request)
    assert (
        replayed.status
        is AuxiliaryGoalSupersedeControllerStatus.ALREADY_SUPERSEDED
    )
    assert replayed.store_result.status == "replayed"
    assert replayed.receipt_sha256 == applied.receipt_sha256

    with store._connect() as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM insession_auxiliary_graph_goals "
            "WHERE auxiliary_graph_id=?",
            (details.auxiliary_graph_id,),
        ).fetchone()[0] == 1


def test_base_drift_controller_preserves_objective_for_later_rebase() -> None:
    session_id, turn_id, task_id, details = _seed_goal(
        "supersede-controller-drift"
    )
    task_graph = _commit_task_graph_one(
        session_id=session_id,
        turn_id=turn_id,
        task_id=task_id,
    )
    request = AuxiliaryGoalSupersedeRequest(
        session_id=session_id,
        turn_id=turn_id,
        task_id=task_id,
        reason=planning_store.PlanningGoalSupersedeReason.BASE_DRIFT,
        auxiliary_graph_id=details.auxiliary_graph_id,
        goal_id=details.goal_id,
        expected_task_state_version=task_graph.task_state_version,
        expected_control_state_version=details.control_state_version,
        expected_goal_state_version=details.goal_state_version,
        expected_revision_state_version=details.revision_state_version,
        expected_budget_state_version=details.budget_state_version,
        expected_current_auxiliary_graph_revision=(
            details.auxiliary_graph_revision
        ),
        expected_base_task_graph_revision=details.base_task_graph_revision,
        observed_task_graph_revision=1,
    )

    result = run_auxiliary_goal_supersede(request)

    assert result.status is AuxiliaryGoalSupersedeControllerStatus.SUPERSEDED
    assert result.store_result.receipt.source_binding is None
    assert result.store_result.receipt.next_goal_objective == details.goal_objective
    assert result.store_result.receipt.next_base_task_graph_revision == 1
    assert result.store_result.receipt.next_target_task_graph_revision == 2
