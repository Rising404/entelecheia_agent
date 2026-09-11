from __future__ import annotations

import pytest

from personagraph.l2.task_graph.contracts import InSessionTaskDetails, InSessionTaskStatus
from personagraph.l2.task_execution.task_node.frontier import (
    TaskAggregateReason,
    TaskNodeFrontierError,
    TaskNodeFrontierErrorCode,
    TaskNodeRuntimeFact,
    build_task_node_tree,
    project_ready_fresh_nodes,
    reduce_task_aggregate,
)


TASK_ID = "task_root"


def _node(
    node_id: str,
    *,
    ordinal: int,
    parent: str | None,
    status: InSessionTaskStatus = InSessionTaskStatus.PROPOSED,
) -> dict[str, object]:
    return {
        "insession_task_node_id": node_id,
        "node_revision": 1,
        "node_kind": "root" if parent is None else "subtask",
        "ordinal": ordinal,
        "parent_insession_task_node_id": parent,
        "status": status.value,
        "state_version": 1,
    }


def _details(
    nodes: tuple[dict[str, object], ...],
    *,
    status: InSessionTaskStatus = InSessionTaskStatus.ACTIVE,
) -> InSessionTaskDetails:
    return InSessionTaskDetails(
        insession_task_id=TASK_ID,
        session_id="session_one",
        task_state_version=3,
        title="root",
        objective="finish root",
        current_graph_revision=1,
        status=status,
        nodes=nodes,
    )


def _fact(
    node_id: str,
    *,
    run: str | None = None,
    delivery: str | None = None,
) -> TaskNodeRuntimeFact:
    return TaskNodeRuntimeFact(
        node_id=node_id,
        recoverable_work_run_id=run,
        current_delivery_id=delivery,
    )


def test_frontier_runs_leaves_in_source_order_then_parent_with_exact_dependencies():
    tree = build_task_node_tree(
        _details(
            (
                _node(TASK_ID, ordinal=0, parent=None),
                _node("child_b", ordinal=2, parent=TASK_ID),
                _node("child_a", ordinal=1, parent=TASK_ID),
            )
        )
    )
    initial = project_ready_fresh_nodes(
        tree,
        (_fact(TASK_ID), _fact("child_a"), _fact("child_b")),
    )
    assert [item.subject.node_id for item in initial] == ["child_a", "child_b"]

    completed_tree = build_task_node_tree(
        _details(
            (
                _node(TASK_ID, ordinal=0, parent=None),
                _node("child_a", ordinal=1, parent=TASK_ID, status=InSessionTaskStatus.COMPLETED),
                _node("child_b", ordinal=2, parent=TASK_ID, status=InSessionTaskStatus.COMPLETED),
            )
        )
    )
    parent = project_ready_fresh_nodes(
        completed_tree,
        (_fact(TASK_ID), _fact("child_a", delivery="delivery_a"), _fact("child_b", delivery="delivery_b")),
    )
    assert len(parent) == 1
    assert parent[0].subject.node_id == TASK_ID
    assert parent[0].dependency_node_ids == ("child_a", "child_b")
    assert parent[0].dependency_delivery_ids == ("delivery_a", "delivery_b")


def test_recoverable_node_is_not_a_fresh_candidate():
    tree = build_task_node_tree(
        _details(
            (_node(TASK_ID, ordinal=0, parent=None, status=InSessionTaskStatus.ACTIVE),)
        )
    )
    assert project_ready_fresh_nodes(tree, (_fact(TASK_ID, run="run_one"),)) == ()


@pytest.mark.parametrize(
    ("active_status", "waiting_status", "expected"),
    (
        (InSessionTaskStatus.PROPOSED, InSessionTaskStatus.AWAITING_USER, InSessionTaskStatus.ACTIVE),
        (InSessionTaskStatus.BLOCKED, InSessionTaskStatus.AWAITING_USER, InSessionTaskStatus.AWAITING_USER),
        (InSessionTaskStatus.BLOCKED, InSessionTaskStatus.WAITING_EXTERNAL, InSessionTaskStatus.WAITING_EXTERNAL),
    ),
)
def test_task_aggregate_precedence(active_status, waiting_status, expected):
    tree = build_task_node_tree(
        _details(
            (
                _node(TASK_ID, ordinal=0, parent=None),
                _node("branch_active", ordinal=1, parent=TASK_ID, status=active_status),
                _node("branch_wait", ordinal=2, parent=TASK_ID, status=waiting_status),
            )
        )
    )
    waiting_run = "run_wait" if waiting_status in {InSessionTaskStatus.AWAITING_USER, InSessionTaskStatus.WAITING_EXTERNAL} else None
    aggregate = reduce_task_aggregate(
        tree,
        (_fact(TASK_ID), _fact("branch_active"), _fact("branch_wait", run=waiting_run)),
    )
    assert aggregate.status is expected


def test_retryable_interrupted_node_keeps_task_active():
    tree = build_task_node_tree(
        _details((_node(TASK_ID, ordinal=0, parent=None, status=InSessionTaskStatus.INTERRUPTED),))
    )
    aggregate = reduce_task_aggregate(tree, (_fact(TASK_ID),))
    assert aggregate.status is InSessionTaskStatus.ACTIVE
    assert aggregate.reason is TaskAggregateReason.RUNNABLE


def test_pristine_task_stays_proposed_and_dead_end_is_blocked():
    pristine = build_task_node_tree(
        _details(
            (_node(TASK_ID, ordinal=0, parent=None),),
            status=InSessionTaskStatus.PROPOSED,
        )
    )
    assert reduce_task_aggregate(pristine, (_fact(TASK_ID),)).status is InSessionTaskStatus.PROPOSED

    blocked = build_task_node_tree(
        _details(
            (_node(TASK_ID, ordinal=0, parent=None, status=InSessionTaskStatus.BLOCKED),)
        )
    )
    aggregate = reduce_task_aggregate(blocked, (_fact(TASK_ID),))
    assert aggregate.status is InSessionTaskStatus.BLOCKED
    assert aggregate.reason is TaskAggregateReason.BLOCKED


def test_awaiting_user_precedes_waiting_external_without_a_runnable_branch():
    tree = build_task_node_tree(
        _details(
            (
                _node(TASK_ID, ordinal=0, parent=None),
                _node("awaiting", ordinal=1, parent=TASK_ID, status=InSessionTaskStatus.AWAITING_USER),
                _node("external", ordinal=2, parent=TASK_ID, status=InSessionTaskStatus.WAITING_EXTERNAL),
            )
        )
    )
    aggregate = reduce_task_aggregate(
        tree,
        (_fact(TASK_ID), _fact("awaiting", run="run_awaiting"), _fact("external", run="run_external")),
    )
    assert aggregate.status is InSessionTaskStatus.AWAITING_USER


def test_all_completed_only_requests_finish_gate_and_never_completes_task():
    tree = build_task_node_tree(
        _details((_node(TASK_ID, ordinal=0, parent=None, status=InSessionTaskStatus.COMPLETED),))
    )
    aggregate = reduce_task_aggregate(tree, (_fact(TASK_ID, delivery="delivery_root"),))
    assert aggregate.status is InSessionTaskStatus.ACTIVE
    assert aggregate.reason is TaskAggregateReason.FINISH_GATE_REQUIRED
    assert aggregate.finish_gate_candidate is True


def test_completed_and_cancelled_task_statuses_are_preserved():
    for status in (InSessionTaskStatus.COMPLETED, InSessionTaskStatus.CANCELLED):
        tree = build_task_node_tree(
            _details(
                (_node(TASK_ID, ordinal=0, parent=None, status=InSessionTaskStatus.COMPLETED),),
                status=status,
            )
        )
        aggregate = reduce_task_aggregate(tree, (_fact(TASK_ID, delivery="delivery_root"),))
        assert aggregate.status is status
        assert aggregate.reason is TaskAggregateReason.TERMINAL_PRESERVED


@pytest.mark.parametrize(
    ("nodes", "code"),
    (
        (
            (_node(TASK_ID, ordinal=0, parent=None), _node(TASK_ID, ordinal=1, parent=TASK_ID)),
            TaskNodeFrontierErrorCode.DUPLICATE_NODE_ID,
        ),
        (
            (_node(TASK_ID, ordinal=0, parent=None), _node("child", ordinal=2, parent=TASK_ID)),
            TaskNodeFrontierErrorCode.INVALID_ORDINALS,
        ),
        (
            (_node(TASK_ID, ordinal=0, parent=None), _node("child", ordinal=1, parent="missing")),
            TaskNodeFrontierErrorCode.INVALID_PARENT,
        ),
    ),
)
def test_corrupt_tree_fails_closed(nodes, code):
    with pytest.raises(TaskNodeFrontierError) as exc:
        build_task_node_tree(_details(nodes))
    assert exc.value.code is code


def test_runtime_facts_must_exactly_cover_nodes_and_completed_nodes_require_delivery():
    tree = build_task_node_tree(
        _details(
            (
                _node(TASK_ID, ordinal=0, parent=None),
                _node("child", ordinal=1, parent=TASK_ID, status=InSessionTaskStatus.COMPLETED),
            )
        )
    )
    with pytest.raises(TaskNodeFrontierError) as missing:
        project_ready_fresh_nodes(tree, (_fact(TASK_ID),))
    assert missing.value.code is TaskNodeFrontierErrorCode.FACT_COVERAGE_MISMATCH

    with pytest.raises(TaskNodeFrontierError) as invalid:
        project_ready_fresh_nodes(tree, (_fact(TASK_ID), _fact("child")))
    assert invalid.value.code is TaskNodeFrontierErrorCode.INVALID_RUNTIME_FACT
