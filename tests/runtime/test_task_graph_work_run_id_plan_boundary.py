"""冷态 TaskGraph WorkRun 稳定 ID 推导的边界覆盖。"""

from __future__ import annotations

import hashlib

import pytest

from personagraph.l2.task_execution.task_graph import id_plan
from personagraph.l2.work_run import TaskNodeSubject


def _subject(*, node_revision: int = 3) -> TaskNodeSubject:
    return TaskNodeSubject(
        task_id="task-1",
        graph_revision=2,
        node_id="node-1",
        node_revision=node_revision,
    )


def test_stable_work_run_ids_are_deterministic_and_subject_bound() -> None:
    subject = _subject()
    plan = id_plan.derive_task_graph_work_run_stable_ids(
        session_id="session-1",
        turn_id="turn-1",
        subject=subject,
    )
    identity = "\0".join(("session-1", "turn-1", "task-1", "2", "node-1", "3"))
    expected_namespace = "task-graph-v1-" + hashlib.sha256(
        identity.encode("utf-8")
    ).hexdigest()[:40]

    assert plan.namespace == expected_namespace
    assert plan.work_run_id == f"{expected_namespace}:workrun"
    assert plan == id_plan.derive_task_graph_work_run_stable_ids(
        session_id="session-1",
        turn_id="turn-1",
        subject=subject,
    )
    assert plan != id_plan.derive_task_graph_work_run_stable_ids(
        session_id="session-1",
        turn_id="turn-1",
        subject=_subject(node_revision=4),
    )


def test_recovery_and_safe_lane_detach_ids_preserve_failure_and_hash_guards() -> None:
    recovered = id_plan.recover_task_graph_work_run_stable_ids(
        "task-graph-v1-namespace:workrun"
    )
    assert recovered.namespace == "task-graph-v1-namespace"
    assert recovered.work_run_id == "task-graph-v1-namespace:workrun"
    with pytest.raises(ValueError, match="no stable controller namespace"):
        id_plan.recover_task_graph_work_run_stable_ids(
            "task-graph-v1-namespace:attempt:1"
        )

    work_run_id = "task-graph-v1-namespace:workrun"
    expected = "task-graph-detach-v1:" + hashlib.sha256(
        f"turn-1\0{work_run_id}".encode("utf-8")
    ).hexdigest()
    detach_id = id_plan.derive_task_graph_safe_lane_detach_apply_id(
        turn_id="turn-1",
        work_run_id=work_run_id,
    )
    assert detach_id == expected
    assert detach_id != id_plan.derive_task_graph_safe_lane_detach_apply_id(
        turn_id="turn-2",
        work_run_id=work_run_id,
    )
