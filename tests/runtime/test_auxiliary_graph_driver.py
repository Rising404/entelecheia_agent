from __future__ import annotations

import pytest

from personagraph.l2.auxiliary_graph import (
    AuxiliaryGraphExecutionFrontier,
    AuxiliaryNodeExecutorKind,
    AuxiliaryNodeReference,
    AuxiliaryPlanningGoalStatus,
    PlanningEpisodeBudgetDisposition,
    ReadyAuxiliaryNodeExecutionCandidate,
    RecoverableAuxiliaryNodeExecutionCandidate,
    RecoverableAuxiliaryPrimitiveInvocationCandidate,
)
from personagraph.l2.auxiliary_execution.driver import (
    AuxiliaryGraphDriverAction,
    canonical_auxiliary_graph_driver_state_guard,
    decide_auxiliary_graph_driver_step,
)
from personagraph.l2.work_run import AuxiliaryNodeSubject, WorkRunStatus


def _subject(node_id: str) -> AuxiliaryNodeSubject:
    return AuxiliaryNodeSubject(
        task_id="task_01",
        auxiliary_graph_id="aux_graph_01",
        auxiliary_graph_revision=1,
        node_id=node_id,
        node_revision=1,
    )


def _ready(
    node_id: str,
    ordinal: int,
    executor: AuxiliaryNodeExecutorKind,
) -> ReadyAuxiliaryNodeExecutionCandidate:
    return ReadyAuxiliaryNodeExecutionCandidate(
        subject=_subject(node_id),
        ordinal=ordinal,
        local_node_key=node_id,
        executor_kind=executor,
        capability_profile_id=(
            None
            if executor is AuxiliaryNodeExecutorKind.TERMINAL_PLANNER
            else "readonly_context"
        ),
        node_state_version=1,
    )


def _frontier(**changes: object) -> AuxiliaryGraphExecutionFrontier:
    values: dict[str, object] = {
        "session_id": "session_01",
        "turn_id": "turn_01",
        "task_id": "task_01",
        "auxiliary_graph_id": "aux_graph_01",
        "goal_id": "goal_01",
        "auxiliary_graph_revision": 1,
        "terminal_node_id": "terminal",
        "control_state_version": 1,
        "goal_state_version": 1,
        "revision_state_version": 1,
        "budget_state_version": 1,
        "base_task_graph_revision": None,
        "target_task_graph_revision": 1,
        "task_state_version": 1,
        "authority_snapshot_id": "authority_01",
        "authority_snapshot_sha256": "a" * 64,
        "structure_sha256": "b" * 64,
        "budget_snapshot_sha256": "c" * 64,
        "budget_disposition": PlanningEpisodeBudgetDisposition.WITHIN_LIMIT,
        "goal_status": AuxiliaryPlanningGoalStatus.ACTIVE,
        "revision_status": "active",
    }
    values.update(changes)
    return AuxiliaryGraphExecutionFrontier(**values)


def test_fresh_dispatch_is_source_ordered_and_executor_typed() -> None:
    frontier = _frontier(
        ready_fresh=(
            _ready("observe", 0, AuxiliaryNodeExecutorKind.HOST_PRIMITIVE),
            _ready("analyze", 1, AuxiliaryNodeExecutorKind.MODEL_WORK_RUN),
        )
    )

    decision = decide_auxiliary_graph_driver_step(frontier)

    assert decision.action is AuxiliaryGraphDriverAction.RUN_HOST_PRIMITIVE
    assert decision.subject == _subject("observe")
    assert [item.node_id for item in decision.ready_node_refs] == [
        "observe",
        "analyze",
    ]


@pytest.mark.parametrize(
    ("status", "action"),
    (
        (WorkRunStatus.ACTIVE, AuxiliaryGraphDriverAction.RESUME_WORK_RUN),
        (WorkRunStatus.INTERRUPTED, AuxiliaryGraphDriverAction.RESUME_WORK_RUN),
        (WorkRunStatus.WAITING_USER, AuxiliaryGraphDriverAction.WAIT_USER),
        (
            WorkRunStatus.WAITING_AUTHORIZATION,
            AuxiliaryGraphDriverAction.WAIT_AUTHORIZATION,
        ),
        (
            WorkRunStatus.WAITING_EXTERNAL,
            AuxiliaryGraphDriverAction.WAIT_EXTERNAL,
        ),
        (WorkRunStatus.PAUSED, AuxiliaryGraphDriverAction.STOP_TURN),
        (WorkRunStatus.TURN_LIMIT_REACHED, AuxiliaryGraphDriverAction.STOP_TURN),
    ),
)
def test_recoverable_cursor_always_wins_before_any_fresh_dispatch(
    status: WorkRunStatus,
    action: AuxiliaryGraphDriverAction,
) -> None:
    candidate = RecoverableAuxiliaryNodeExecutionCandidate(
        subject=_subject("observe"),
        ordinal=0,
        local_node_key="observe",
        executor_kind=AuxiliaryNodeExecutorKind.MODEL_WORK_RUN,
        capability_profile_id="readonly_context",
        node_status="active",
        node_state_version=2,
        work_run_id="work_run_01",
        work_run_status=status,
        work_run_revision=3,
    )
    decision = decide_auxiliary_graph_driver_step(
        _frontier(recoverable=(candidate,))
    )

    assert decision.action is action
    assert decision.work_run_id == "work_run_01"
    assert decision.subject == _subject("observe")


def test_hard_budget_stops_before_a_fresh_external_call() -> None:
    decision = decide_auxiliary_graph_driver_step(
        _frontier(
            budget_disposition=(
                PlanningEpisodeBudgetDisposition.HARD_LIMIT_REACHED
            ),
            ready_fresh=(
                _ready(
                    "observe",
                    0,
                    AuxiliaryNodeExecutorKind.HOST_PRIMITIVE,
                ),
            ),
        )
    )

    assert decision.action is AuxiliaryGraphDriverAction.BUDGET_EXHAUSTED


def test_hard_budget_also_blocks_an_active_work_run_redispatch() -> None:
    candidate = RecoverableAuxiliaryNodeExecutionCandidate(
        subject=_subject("observe"),
        ordinal=0,
        local_node_key="observe",
        executor_kind=AuxiliaryNodeExecutorKind.MODEL_WORK_RUN,
        capability_profile_id="readonly_context",
        node_status="active",
        node_state_version=2,
        work_run_id="work_run_01",
        work_run_status=WorkRunStatus.ACTIVE,
        work_run_revision=3,
    )

    decision = decide_auxiliary_graph_driver_step(
        _frontier(
            budget_disposition=(
                PlanningEpisodeBudgetDisposition.HARD_LIMIT_REACHED
            ),
            recoverable=(candidate,),
        )
    )

    assert decision.action is AuxiliaryGraphDriverAction.BUDGET_EXHAUSTED


def test_reserved_host_primitive_is_recovered_before_any_fresh_dispatch() -> None:
    candidate = RecoverableAuxiliaryPrimitiveInvocationCandidate(
        subject=_subject("observe"),
        ordinal=0,
        local_node_key="observe",
        capability_profile_id="mounted_document_read",
        node_state_version=1,
        primitive_call_id="primitive_call_01",
        primitive_kind="resource_perception",
        invocation_turn_id="turn_01",
        state_guard_sha256="d" * 64,
    )

    decision = decide_auxiliary_graph_driver_step(
        _frontier(recoverable_primitive=(candidate,))
    )

    assert decision.action is AuxiliaryGraphDriverAction.RESUME_HOST_PRIMITIVE
    assert decision.subject == _subject("observe")
    assert decision.primitive_call_id == "primitive_call_01"


def test_soft_budget_allows_only_convergence_or_user_input() -> None:
    investigation = decide_auxiliary_graph_driver_step(
        _frontier(
            budget_disposition=(
                PlanningEpisodeBudgetDisposition.SOFT_LIMIT_REACHED
            ),
            ready_fresh=(
                _ready(
                    "observe",
                    0,
                    AuxiliaryNodeExecutorKind.HOST_PRIMITIVE,
                ),
            ),
        )
    )
    convergence = decide_auxiliary_graph_driver_step(
        _frontier(
            budget_disposition=(
                PlanningEpisodeBudgetDisposition.SOFT_LIMIT_REACHED
            ),
            ready_fresh=(
                _ready(
                    "terminal",
                    1,
                    AuxiliaryNodeExecutorKind.TERMINAL_PLANNER,
                ),
            ),
        )
    )

    assert investigation.action is AuxiliaryGraphDriverAction.REQUEST_REVISION
    assert convergence.action is AuxiliaryGraphDriverAction.RUN_TERMINAL_PLANNER


def test_initial_planning_bootstrap_can_only_request_architect_revision() -> None:
    bootstrap = _ready(
        "bootstrap_terminal",
        0,
        AuxiliaryNodeExecutorKind.TERMINAL_PLANNER,
    )
    decision = decide_auxiliary_graph_driver_step(
        _frontier(
            terminal_node_id="bootstrap_terminal",
            ready_fresh=(bootstrap,),
        )
    )

    assert decision.action is AuxiliaryGraphDriverAction.REQUEST_REVISION
    assert decision.reason_code == "initial_planning_bootstrap_requires_architect"
    assert decision.subject is None


def test_blocker_requests_revision_and_completed_terminal_seals() -> None:
    blocked = decide_auxiliary_graph_driver_step(
        _frontier(
            blocking_node_refs=(
                AuxiliaryNodeReference(node_id="observe", node_revision=1),
            )
        )
    )
    sealed = decide_auxiliary_graph_driver_step(
        _frontier(
            completed_node_refs=(
                AuxiliaryNodeReference(node_id="terminal", node_revision=1),
            )
        )
    )

    assert blocked.action is AuxiliaryGraphDriverAction.REQUEST_REVISION
    assert sealed.action is AuxiliaryGraphDriverAction.SEAL_REVISION


def test_ready_goal_commits_and_unclassified_stall_fails_closed() -> None:
    commit = decide_auxiliary_graph_driver_step(
        _frontier(goal_status=AuxiliaryPlanningGoalStatus.PROPOSAL_READY)
    )
    stalled = decide_auxiliary_graph_driver_step(_frontier())

    assert commit.action is AuxiliaryGraphDriverAction.COMMIT_READY_PROPOSAL
    assert stalled.action is AuxiliaryGraphDriverAction.FAILED_CLOSED


def test_state_guard_is_stable_and_changes_with_any_authority_version() -> None:
    frontier = _frontier(
        ready_fresh=(
            _ready("observe", 0, AuxiliaryNodeExecutorKind.HOST_PRIMITIVE),
        )
    )

    first = canonical_auxiliary_graph_driver_state_guard(frontier)
    replay = canonical_auxiliary_graph_driver_state_guard(frontier)
    changed = canonical_auxiliary_graph_driver_state_guard(
        frontier.model_copy(update={"budget_state_version": 2})
    )

    assert first == replay
    assert len(first) == 64
    assert changed != first
