from __future__ import annotations

import hashlib

import pytest

from personagraph.l2.task_graph.contracts import (
    InSessionTaskGraphRevisionProposal,
    InSessionTaskGraphRevisionValidationContext,
    InSessionTaskSourceAnchor,
)
from personagraph.session import store
from personagraph.session.l2_store import task_graph as task_graph_store
from personagraph.session.l2_store import work_run as work_run_store
from personagraph.session.l2_store import auxiliary_graph as auxiliary_graph_store
from personagraph.session.l2_store import planning as planning_store
from personagraph.session.persistence.l2.auxiliary_graph import auxiliary_graphs
from personagraph.session.persistence.l2.task_graph import insession_tasks as insession_task_records
from personagraph.l2.work_run import (
    AuxiliaryNodeSubject,
    HostAcceptedAttemptDecision,
    HostMaterializedCallToolsAction,
    HostMaterializedToolCall,
    RequestTaskGraphRevisionAction,
    TaskGraphExecutionReplanReason,
    TaskNodeSubject,
    ToolResultStatus,
    ToolResult,
)
from tests.session.test_auxiliary_graph_persistence import (
    _commit_initial,
    _seed_task_shell,
    _revision_proposal,
    _window_revision,
)


def _task_graph_proposal() -> InSessionTaskGraphRevisionProposal:
    return InSessionTaskGraphRevisionProposal.model_validate(
        {
            "root": {
                "root_key": "paper",
                "nodes": [
                    {
                        "node_key": "paper",
                        "node_kind": "root",
                        "title": "论文分析",
                        "objective": "分析论文并形成结论",
                        "source_anchor_ids": ["original_request"],
                        "acceptance_criteria": [
                            {
                                "acceptance_id": "analysis_complete",
                                "criterion": "形成可核对结论",
                                "source_anchor_ids": ["original_request"],
                            }
                        ],
                    },
                    {
                        "node_key": "extract_method",
                        "node_kind": "subtask",
                        "parent_node_key": "paper",
                        "title": "提取方法",
                        "objective": "提取论文方法",
                        "source_anchor_ids": ["original_request"],
                        "acceptance_criteria": [
                            {
                                "acceptance_id": "method_complete",
                                "criterion": "方法信息可核对",
                                "source_anchor_ids": ["original_request"],
                            }
                        ],
                    },
                ],
            }
        }
    )


def _seed_goal(
    prefix: str,
    *,
    observe_executor: str = "host_primitive",
):
    session_id, turn_id, task_id = _seed_task_shell()
    if observe_executor == "host_primitive":
        _commit_initial(
            session_id=session_id,
            turn_id=turn_id,
            task_id=task_id,
            apply_id=f"{prefix}-graph-apply",
            graph_id=f"{prefix}-graph",
            goal_id=f"{prefix}-goal",
        )
    else:
        auxiliary_graphs.commit_auxiliary_graph_revision(
            store._deps(),
            session_id=session_id,
            turn_id=turn_id,
            insession_task_id=task_id,
            expected_task_state_version=1,
            expected_base_task_graph_revision=None,
            expected_control_state_version=None,
            expected_current_auxiliary_graph_revision=None,
            apply_id=f"{prefix}-graph-apply",
            goal_objective="形成可执行且受来源约束的任务图",
            proposal=_revision_proposal(observe_executor=observe_executor),
            authority_context={"anchors": []},
            budget_profile={"profile_id": "planning-test-v1"},
            auxiliary_graph_id=f"{prefix}-graph",
            goal_id=f"{prefix}-goal",
        )
    details = auxiliary_graph_store.get_auxiliary_graph_for_task(
        session_id=session_id,
        insession_task_id=task_id,
    )
    assert details is not None
    return session_id, turn_id, task_id, details


def _commit_task_graph_one(
    *,
    session_id: str,
    turn_id: str,
    task_id: str,
):
    task = task_graph_store.get_insession_task_details(session_id, task_id)
    assert task is not None
    source_text = "请分析论文并给出一份可执行计划"
    return insession_task_records.commit_insession_task_graph_revision(
        store._deps(),
        session_id=session_id,
        source_turn_id=turn_id,
        target_insession_task_id=task_id,
        expected_current_graph_revision=None,
        expected_task_state_version=task.task_state_version,
        expected_window_revision=_window_revision(session_id),
        apply_id="goal-supersede-task-graph-one",
        proposal=_task_graph_proposal(),
        trusted_context=InSessionTaskGraphRevisionValidationContext(
            session_id=session_id,
            source_turn_id=turn_id,
            target_insession_task_id=task_id,
            expected_current_graph_revision=None,
            source_anchors=(
                InSessionTaskSourceAnchor(
                    anchor_id="original_request",
                    source_turn_id=turn_id,
                    source_kind="current_user_instruction",
                    start=0,
                    end=len(source_text),
                    excerpt=source_text,
                ),
            ),
            authorization_anchor_ids=("original_request",),
            required_anchor_ids=("original_request",),
        ),
    )


def _command(
    *,
    prefix: str,
    session_id: str,
    turn_id: str,
    task_id: str,
    details,
    reason,
    expected_task_state_version: int,
    observed_task_graph_revision: int | None,
    replacement_objective: str | None = None,
    source_start: int | None = None,
    source_end: int | None = None,
    source_sha256: str | None = None,
):
    return planning_store.SupersedeAuxiliaryPlanningGoalCommand(
        apply_id=f"{prefix}-supersede-apply",
        session_id=session_id,
        invocation_turn_id=turn_id,
        task_id=task_id,
        auxiliary_graph_id=details.auxiliary_graph_id,
        goal_id=details.goal_id,
        reason=reason,
        expected_task_state_version=expected_task_state_version,
        expected_control_state_version=details.control_state_version,
        expected_goal_state_version=details.goal_state_version,
        expected_revision_state_version=details.revision_state_version,
        expected_budget_state_version=details.budget_state_version,
        expected_current_auxiliary_graph_revision=(
            details.auxiliary_graph_revision
        ),
        expected_base_task_graph_revision=details.base_task_graph_revision,
        observed_task_graph_revision=observed_task_graph_revision,
        replacement_objective=replacement_objective,
        source_turn_id=(turn_id if source_start is not None else None),
        source_start=source_start,
        source_end=source_end,
        source_sha256=source_sha256,
    )


def _create_active_execution_replan_request(
    *,
    prefix: str,
    session_id: str,
    turn_id: str,
    task_id: str,
):
    task = task_graph_store.get_insession_task_details(session_id, task_id)
    assert task is not None and task.current_graph_revision is not None
    node = task.nodes[-1]
    subject = TaskNodeSubject(
        task_id=task_id,
        graph_revision=task.current_graph_revision,
        node_id=str(node["insession_task_node_id"]),
        node_revision=int(node["node_revision"]),
    )
    created = work_run_store.create_task_node_work_run(
        session_id=session_id,
        turn_id=turn_id,
        subject=subject,
        expected_task_state_version=task.task_state_version,
        expected_node_state_version=int(node["state_version"]),
        expected_window_revision=_window_revision(session_id),
        apply_id=f"{prefix}-create-run",
        work_run_id=f"{prefix}-run",
    )
    started = work_run_store.start_work_run_attempt(
        session_id=session_id,
        turn_id=turn_id,
        work_run_id=created.work_run_id,
        expected_work_run_revision=created.work_run_revision,
        expected_progress_revision=created.acceptance_progress_revision,
        expected_window_revision=created.window_state_version,
        apply_id=f"{prefix}-start-attempt",
        catalog_snapshot={"revision": 1, "tools": []},
        attempt_id=f"{prefix}-attempt",
    )
    work_run_store.commit_work_run_attempt_decision(
        session_id=session_id,
        turn_id=turn_id,
        work_run_id=created.work_run_id,
        attempt_id=started.current_attempt_id,
        decision=HostAcceptedAttemptDecision(
            acceptance_updates=(),
            action=RequestTaskGraphRevisionAction(
                reason=TaskGraphExecutionReplanReason.TASK_DECOMPOSITION_INCOMPLETE,
                diagnosis="当前节点职责混杂，无法可靠继续执行。",
                revision_objective="拆分调查与综合节点并修订当前源节点。",
                supporting_tool_result_ids=(),
            ),
        ),
        expected_work_run_revision=started.work_run_revision,
        expected_progress_revision=started.acceptance_progress_revision,
        expected_window_revision=started.window_state_version,
        apply_id=f"{prefix}-request-revision",
        active_seconds_delta=1.0,
    )
    request = work_run_store.get_active_task_graph_execution_replan_request(
        session_id=session_id,
        task_id=task_id,
    )
    assert request is not None
    return request


def test_base_drift_supersedes_goal_and_revision_atomically_and_replays() -> None:
    session_id, turn_id, task_id, details = _seed_goal("supersede-drift")
    task_graph = _commit_task_graph_one(
        session_id=session_id,
        turn_id=turn_id,
        task_id=task_id,
    )
    command = _command(
        prefix="supersede-drift",
        session_id=session_id,
        turn_id=turn_id,
        task_id=task_id,
        details=details,
        reason=planning_store.PlanningGoalSupersedeReason.BASE_DRIFT,
        expected_task_state_version=task_graph.task_state_version,
        observed_task_graph_revision=1,
    )

    applied = planning_store.supersede_auxiliary_planning_goal(command=command)

    assert applied.status == "applied"
    receipt = applied.receipt
    assert receipt.reason is planning_store.PlanningGoalSupersedeReason.BASE_DRIFT
    assert receipt.superseded_goal_id == details.goal_id
    assert receipt.superseded_auxiliary_graph_revision == 1
    assert receipt.previous_base_task_graph_revision is None
    assert receipt.next_base_task_graph_revision == 1
    assert receipt.next_target_task_graph_revision == 2
    assert receipt.next_goal_objective == details.goal_objective
    assert receipt.control_state_version_after == details.control_state_version + 1
    assert receipt.goal_state_version_after == details.goal_state_version + 1
    assert (
        receipt.revision_state_version_after
        == details.revision_state_version + 1
    )
    assert receipt.budget_state_version_after == details.budget_state_version

    current = auxiliary_graph_store.get_auxiliary_graph_for_task(
        session_id=session_id,
        insession_task_id=task_id,
    )
    assert current is not None
    assert current.goal_status == "superseded"
    assert current.revision_status == "superseded"
    assert current.goal_id == details.goal_id
    assert current.auxiliary_graph_revision == details.auxiliary_graph_revision

    replayed = planning_store.supersede_auxiliary_planning_goal(command=command)
    assert replayed == applied.model_copy(update={"status": "replayed"})
    with store._connect() as conn:
        row = conn.execute(
            "SELECT operation, COUNT(*) AS count FROM "
            "insession_auxiliary_graph_revision_apply_receipts_v2 "
            "WHERE apply_id=?",
            (command.apply_id,),
        ).fetchone()
    assert row is not None
    assert tuple(row) == ("supersede_goal", 1)

    with pytest.raises(planning_store.AuxiliaryGoalSupersedeIdentityCollision):
        planning_store.supersede_auxiliary_planning_goal(
            command=command.model_copy(
                update={"observed_task_graph_revision": None}
            )
        )


def test_stale_revision_cas_rolls_back_every_supersede_write() -> None:
    session_id, turn_id, task_id, details = _seed_goal("supersede-stale")
    task = task_graph_store.get_insession_task_details(session_id, task_id)
    assert task is not None
    command = _command(
        prefix="supersede-stale",
        session_id=session_id,
        turn_id=turn_id,
        task_id=task_id,
        details=details,
        reason=planning_store.PlanningGoalSupersedeReason.USER_TARGET_CHANGED,
        expected_task_state_version=task.task_state_version,
        observed_task_graph_revision=None,
        replacement_objective="只比较论文的实验设计",
        source_start=0,
        source_end=len("请分析论文并给出一份可执行计划"),
        source_sha256=hashlib.sha256(
            "请分析论文并给出一份可执行计划".encode()
        ).hexdigest(),
    ).model_copy(
        update={
            "expected_revision_state_version": (
                details.revision_state_version + 1
            )
        }
    )

    with pytest.raises(planning_store.AuxiliaryGoalSupersedeStaleAuthority):
        planning_store.supersede_auxiliary_planning_goal(command=command)

    after = auxiliary_graph_store.get_auxiliary_graph_for_task(
        session_id=session_id,
        insession_task_id=task_id,
    )
    assert after is not None
    assert after.goal_status == "active"
    assert after.revision_status == "active"
    assert after.control_state_version == details.control_state_version
    with store._connect() as conn:
        count = conn.execute(
            "SELECT COUNT(*) FROM "
            "insession_auxiliary_graph_revision_apply_receipts_v2 "
            "WHERE apply_id=?",
            (command.apply_id,),
        ).fetchone()[0]
    assert count == 0


def test_active_execution_replan_rejects_supersede_before_any_write() -> None:
    session_id, turn_id, task_id, _details = _seed_goal(
        "supersede-execution-replan"
    )
    _commit_task_graph_one(
        session_id=session_id,
        turn_id=turn_id,
        task_id=task_id,
    )
    request = _create_active_execution_replan_request(
        prefix="supersede-execution-replan",
        session_id=session_id,
        turn_id=turn_id,
        task_id=task_id,
    )
    task = task_graph_store.get_insession_task_details(session_id, task_id)
    before = auxiliary_graph_store.get_auxiliary_graph_for_task(
        session_id=session_id,
        insession_task_id=task_id,
    )
    assert task is not None and before is not None
    command = _command(
        prefix="supersede-execution-replan",
        session_id=session_id,
        turn_id=turn_id,
        task_id=task_id,
        details=before,
        reason=planning_store.PlanningGoalSupersedeReason.BASE_DRIFT,
        expected_task_state_version=task.task_state_version,
        observed_task_graph_revision=task.current_graph_revision,
    )

    with pytest.raises(
        planning_store.AuxiliaryGoalSupersedeStaleAuthority,
        match="TaskGraph revision authority",
    ):
        planning_store.supersede_auxiliary_planning_goal(command=command)

    assert work_run_store.get_active_task_graph_execution_replan_request(
        session_id=session_id,
        task_id=task_id,
    ) == request
    after = auxiliary_graph_store.get_auxiliary_graph_for_task(
        session_id=session_id,
        insession_task_id=task_id,
    )
    assert after is not None
    assert after.goal_status == before.goal_status == "active"
    assert after.revision_status == before.revision_status == "active"
    assert after.goal_state_version == before.goal_state_version
    assert after.revision_state_version == before.revision_state_version
    assert after.control_state_version == before.control_state_version
    with store._connect() as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM "
            "insession_auxiliary_graph_revision_apply_receipts_v2 "
            "WHERE apply_id=?",
            (command.apply_id,),
        ).fetchone()[0] == 0


def test_receipt_is_sufficient_for_a_later_explicit_fresh_goal_bootstrap() -> None:
    session_id, turn_id, task_id, details = _seed_goal("supersede-bootstrap")
    task_graph = _commit_task_graph_one(
        session_id=session_id,
        turn_id=turn_id,
        task_id=task_id,
    )
    superseded = planning_store.supersede_auxiliary_planning_goal(
        command=_command(
            prefix="supersede-bootstrap",
            session_id=session_id,
            turn_id=turn_id,
            task_id=task_id,
            details=details,
            reason=planning_store.PlanningGoalSupersedeReason.BASE_DRIFT,
            expected_task_state_version=task_graph.task_state_version,
            observed_task_graph_revision=1,
        )
    )
    receipt = superseded.receipt
    with store._connect() as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM insession_auxiliary_graph_goals "
            "WHERE auxiliary_graph_id=?",
            (details.auxiliary_graph_id,),
        ).fetchone()[0] == 1

    fresh = auxiliary_graph_store.commit_auxiliary_graph_revision(
        session_id=session_id,
        turn_id=turn_id,
        insession_task_id=task_id,
        expected_task_state_version=receipt.task_state_version_after,
        expected_base_task_graph_revision=receipt.next_base_task_graph_revision,
        expected_control_state_version=receipt.control_state_version_after,
        expected_current_auxiliary_graph_revision=(
            receipt.superseded_auxiliary_graph_revision
        ),
        apply_id="supersede-bootstrap-fresh-goal",
        goal_objective=receipt.next_goal_objective,
        proposal=_revision_proposal(reason="authority_changed"),
        authority_context={"anchors": []},
        budget_profile={"profile_id": "planning-test-v1"},
        auxiliary_graph_id=receipt.auxiliary_graph_id,
        goal_id="supersede-bootstrap-next-goal",
    )

    assert fresh.status == "applied"
    assert fresh.goal_id == "supersede-bootstrap-next-goal"
    current = auxiliary_graph_store.get_auxiliary_graph_for_task(
        session_id=session_id,
        insession_task_id=task_id,
    )
    assert current is not None
    assert current.goal_id == fresh.goal_id
    assert current.base_task_graph_revision == receipt.next_base_task_graph_revision
    assert current.target_task_graph_revision == receipt.next_target_task_graph_revision
    assert current.goal_objective == receipt.next_goal_objective


def test_pending_auxiliary_work_run_fails_closed_without_partial_write() -> None:
    session_id, turn_id, task_id, details = _seed_goal(
        "supersede-pending",
        observe_executor="model_work_run",
    )
    task = task_graph_store.get_insession_task_details(session_id, task_id)
    assert task is not None
    node = details.nodes[0]
    work_run_store.create_auxiliary_node_work_run(
        session_id=session_id,
        turn_id=turn_id,
        subject=AuxiliaryNodeSubject(
            task_id=task_id,
            auxiliary_graph_id=details.auxiliary_graph_id,
            auxiliary_graph_revision=details.auxiliary_graph_revision,
            node_id=node.auxiliary_node_id,
            node_revision=node.node_revision,
        ),
        expected_task_state_version=task.task_state_version,
        expected_node_state_version=node.state_version,
        expected_window_revision=_window_revision(session_id),
        apply_id="supersede-pending-create-run",
        work_run_id="supersede-pending-run",
    )
    task = task_graph_store.get_insession_task_details(session_id, task_id)
    assert task is not None
    command = _command(
        prefix="supersede-pending",
        session_id=session_id,
        turn_id=turn_id,
        task_id=task_id,
        details=details,
        reason=planning_store.PlanningGoalSupersedeReason.USER_TARGET_CHANGED,
        expected_task_state_version=task.task_state_version,
        observed_task_graph_revision=None,
        replacement_objective="改为只分析消融实验",
        source_start=0,
        source_end=len("请分析论文并给出一份可执行计划"),
        source_sha256=hashlib.sha256(
            "请分析论文并给出一份可执行计划".encode()
        ).hexdigest(),
    )

    with pytest.raises(planning_store.AuxiliaryGoalSupersedeUnsafeWorkRun):
        planning_store.supersede_auxiliary_planning_goal(command=command)

    after = auxiliary_graph_store.get_auxiliary_graph_for_task(
        session_id=session_id,
        insession_task_id=task_id,
    )
    assert after is not None
    assert after.goal_status == "active"
    assert after.revision_status == "active"
    with store._connect() as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM "
            "insession_auxiliary_graph_revision_apply_receipts_v2 "
            "WHERE apply_id=?",
            (command.apply_id,),
        ).fetchone()[0] == 0


def test_completion_unconfirmed_work_run_requires_external_reconciliation() -> None:
    session_id, turn_id, task_id, details = _seed_goal(
        "supersede-uncertain",
        observe_executor="model_work_run",
    )
    task = task_graph_store.get_insession_task_details(session_id, task_id)
    assert task is not None
    node = details.nodes[0]
    created = work_run_store.create_auxiliary_node_work_run(
        session_id=session_id,
        turn_id=turn_id,
        subject=AuxiliaryNodeSubject(
            task_id=task_id,
            auxiliary_graph_id=details.auxiliary_graph_id,
            auxiliary_graph_revision=details.auxiliary_graph_revision,
            node_id=node.auxiliary_node_id,
            node_revision=node.node_revision,
        ),
        expected_task_state_version=task.task_state_version,
        expected_node_state_version=node.state_version,
        expected_window_revision=_window_revision(session_id),
        apply_id="supersede-uncertain-create-run",
        work_run_id="supersede-uncertain-run",
    )
    started = work_run_store.start_work_run_attempt(
        session_id=session_id,
        turn_id=turn_id,
        work_run_id="supersede-uncertain-run",
        expected_work_run_revision=created.work_run_revision,
        expected_progress_revision=created.acceptance_progress_revision,
        expected_window_revision=created.window_state_version,
        apply_id="supersede-uncertain-start",
        catalog_snapshot={"revision": 1, "tools": ["effectful_test"]},
        attempt_id="supersede-uncertain-attempt",
    )
    decided = work_run_store.commit_work_run_attempt_decision(
        session_id=session_id,
        turn_id=turn_id,
        work_run_id="supersede-uncertain-run",
        attempt_id="supersede-uncertain-attempt",
        decision=HostAcceptedAttemptDecision(
            acceptance_updates=(),
            action=HostMaterializedCallToolsAction(
                calls=(
                    HostMaterializedToolCall(
                        tool_call_id="supersede-uncertain-call",
                        tool_id="effectful_test",
                        tool_version="1.0.0",
                        arguments={"value": "x"},
                        modifies_environment=True,
                    ),
                )
            ),
        ),
        expected_work_run_revision=started.work_run_revision,
        expected_progress_revision=started.acceptance_progress_revision,
        expected_window_revision=started.window_state_version,
        apply_id="supersede-uncertain-decision",
    )
    appended = work_run_store.append_work_run_tool_result(
        session_id=session_id,
        turn_id=turn_id,
        work_run_id="supersede-uncertain-run",
        result=ToolResult(
            status=ToolResultStatus.COMPLETION_UNCONFIRMED,
            tool_result_id="supersede-uncertain-result",
            tool_call_id="supersede-uncertain-call",
            attempt_id="supersede-uncertain-attempt",
            ordinal=1,
            output=None,
            error_code="transport_completion_unconfirmed",
            error_message="dispatch result is not known",
        ),
        expected_work_run_revision=decided.work_run_revision,
        expected_progress_revision=decided.acceptance_progress_revision,
        expected_window_revision=decided.window_state_version,
        apply_id="supersede-uncertain-result-apply",
    )
    assert appended.work_run_status.value == "active"
    closed = work_run_store.close_work_run_attempt(
        session_id=session_id,
        turn_id=turn_id,
        work_run_id="supersede-uncertain-run",
        attempt_id="supersede-uncertain-attempt",
        expected_work_run_revision=appended.work_run_revision,
        expected_progress_revision=appended.acceptance_progress_revision,
        expected_window_revision=appended.window_state_version,
        apply_id="supersede-uncertain-close",
        active_seconds_delta=1,
    )
    assert closed.work_run_status.value == "waiting_external"
    assert closed.work_run_reason == "operation_completion_unconfirmed"
    with store._connect() as conn:
        uncertain = conn.execute(
            "SELECT status FROM insession_work_run_tool_results "
            "WHERE tool_result_id='supersede-uncertain-result'"
        ).fetchone()
        aggregate = conn.execute(
            "SELECT task.current_status, node.status, goal.status, rev.status "
            "FROM insession_tasks AS task "
            "JOIN insession_auxiliary_graph_v2_containers AS control "
            "ON control.session_id=task.session_id "
            "AND control.insession_task_id=task.insession_task_id "
            "JOIN insession_auxiliary_graph_goals AS goal "
            "ON goal.goal_id=control.current_goal_id "
            "JOIN insession_auxiliary_graph_revision_states_v2 AS rev "
            "ON rev.auxiliary_graph_id=control.auxiliary_graph_id "
            "AND rev.auxiliary_graph_revision="
            "control.current_auxiliary_graph_revision "
            "JOIN insession_auxiliary_node_states_v2 AS node "
            "ON node.auxiliary_graph_id=control.auxiliary_graph_id "
            "AND node.auxiliary_graph_revision="
            "control.current_auxiliary_graph_revision "
            "WHERE task.insession_task_id=? AND node.auxiliary_node_id=?",
            (task_id, node.auxiliary_node_id),
        ).fetchone()
    assert uncertain is not None
    assert uncertain["status"] == "completion_unconfirmed"
    assert tuple(aggregate) == (
        "waiting_external",
        "waiting_external",
        "waiting_external",
        "waiting_external",
    )

    current = auxiliary_graph_store.get_auxiliary_graph_for_task(
        session_id=session_id,
        insession_task_id=task_id,
    )
    task = task_graph_store.get_insession_task_details(session_id, task_id)
    assert current is not None and task is not None
    command = _command(
        prefix="supersede-uncertain",
        session_id=session_id,
        turn_id=turn_id,
        task_id=task_id,
        details=current,
        reason=planning_store.PlanningGoalSupersedeReason.USER_TARGET_CHANGED,
        expected_task_state_version=task.task_state_version,
        observed_task_graph_revision=None,
        replacement_objective="改为只分析消融实验",
        source_start=0,
        source_end=len("请分析论文并给出一份可执行计划"),
        source_sha256=hashlib.sha256(
            "请分析论文并给出一份可执行计划".encode()
        ).hexdigest(),
    )

    with pytest.raises(
        planning_store.AuxiliaryGoalSupersedeUnsafeWorkRun,
        match="completion-unconfirmed",
    ):
        planning_store.supersede_auxiliary_planning_goal(command=command)

    after = auxiliary_graph_store.get_auxiliary_graph_for_task(
        session_id=session_id,
        insession_task_id=task_id,
    )
    assert after is not None
    assert after.goal_status != "superseded"
    assert after.revision_status != "superseded"
