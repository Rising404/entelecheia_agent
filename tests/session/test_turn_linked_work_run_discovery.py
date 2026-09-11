from __future__ import annotations

import json

import pytest

from personagraph.session import store
from personagraph.session.l2_store import work_run as work_run_store
from personagraph.session.persistence.l2.work_run.work_execution import (
    _ensure_task_node_execution_subject,
)
from personagraph.l2.work_run import TaskNodeSubject, WorkRunStatus


def _new_turn(*, suffix: str) -> tuple[str, str]:
    session_id = store.create_session("Entelecheia", title=f"discovery-{suffix}")
    accepted = store.accept_turn_execution(
        session_id=session_id,
        client_request_id=f"request-{suffix}",
        source="runtime_test",
        user_text="继续关联任务",
        lease_owner="work-run-discovery-test",
    )
    return session_id, str(accepted["turn"]["turn_id"])


def _insert_task_and_run(
    *,
    session_id: str,
    turn_id: str,
    task_id: str,
    node_id: str,
    work_run_id: str,
    status: str,
    linked: bool,
    created_at: str,
) -> None:
    acceptance_json = json.dumps(
        [
            {
                "acceptance_id": "deliverable",
                "criterion": "提供交付文本",
                "source_anchor_ids": ["request"],
            }
        ],
        ensure_ascii=False,
    )
    with store._connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            "INSERT INTO insession_tasks "
            "(insession_task_id, session_id, current_graph_revision, current_status, "
            "state_version, root_title, root_objective, created_turn_id, created_at, updated_at) "
            "VALUES (?, ?, 1, 'active', 1, ?, '完成测试任务', ?, ?, ?)",
            (task_id, session_id, task_id, turn_id, created_at, created_at),
        )
        conn.execute(
            "INSERT INTO insession_task_graph_revisions "
            "(insession_task_id, graph_revision, source_turn_id, proposal_hash, "
            "source_anchors_json, authorization_anchor_ids_json, required_anchor_ids_json, created_at) "
            "VALUES (?, 1, ?, ?, '[]', '[]', '[]', ?)",
            (task_id, turn_id, f"proposal-{task_id}", created_at),
        )
        conn.execute(
            "INSERT INTO insession_task_graph_nodes "
            "(insession_task_id, graph_revision, insession_task_node_id, node_revision, "
            "node_kind, ordinal, title, objective, source_anchor_ids_json, "
            "acceptance_criteria_json, constraints_json, created_at) "
            "VALUES (?, 1, ?, 1, 'root', 0, ?, '完成节点', '[]', ?, '[]', ?)",
            (task_id, node_id, task_id, acceptance_json, created_at),
        )
        conn.execute(
            "INSERT INTO insession_task_node_states "
            "(insession_task_id, insession_task_node_id, node_revision, status, "
            "state_version, updated_at) VALUES (?, ?, 1, 'active', 1, ?)",
            (task_id, node_id, created_at),
        )
        if linked:
            conn.execute(
                "INSERT INTO insession_task_turn_links "
                "(session_id, turn_id, insession_task_id, insession_task_node_id, "
                "relation, created_at) VALUES (?, ?, ?, NULL, 'referenced', ?)",
                (session_id, turn_id, task_id, created_at),
            )
        execution_subject_id = _ensure_task_node_execution_subject(
            conn,
            session_id=session_id,
            subject=TaskNodeSubject(
                task_id=task_id,
                graph_revision=1,
                node_id=node_id,
                node_revision=1,
            ),
            created_at=created_at,
        )
        conn.execute(
            "INSERT INTO insession_work_runs "
            "(work_run_id, execution_subject_id, session_id, subject_kind, "
            "insession_task_id, graph_revision, "
            "insession_task_node_id, node_revision, status, reason, revision, "
            "max_attempts, soft_active_seconds, hard_active_seconds, attempts_started, "
            "active_seconds_consumed, current_attempt_id, created_turn_id, updated_turn_id, "
            "created_at, updated_at, current_verification_request_id) "
            "VALUES (?, ?, ?, 'task_node', ?, 1, ?, 1, ?, NULL, 1, "
            "32, 720, 900, 0, 0, NULL, ?, ?, ?, ?, NULL)",
            (
                work_run_id,
                execution_subject_id,
                session_id,
                task_id,
                node_id,
                status,
                turn_id,
                turn_id,
                created_at,
                created_at,
            ),
        )


def _link_work_run(
    *,
    session_id: str,
    turn_id: str,
    work_run_id: str,
    link_revision: int,
) -> None:
    with store._connect() as conn:
        conn.execute(
            "INSERT INTO insession_work_run_turn_links "
            "(session_id, turn_id, work_run_id, link_revision, relation, created_at) "
            "VALUES (?, ?, ?, ?, 'started', ?)",
            (
                session_id,
                turn_id,
                work_run_id,
                link_revision,
                f"2026-08-14T00:00:{link_revision:02d}+00:00",
            ),
        )


def test_discovery_returns_zero_when_turn_has_no_linked_nonterminal_run() -> None:
    session_id, turn_id = _new_turn(suffix="zero")

    assert work_run_store.list_turn_linked_nonterminal_work_runs(
        session_id=session_id,
        turn_id=turn_id,
    ) == ()


def test_discovery_returns_one_typed_candidate_and_excludes_unlinked_task() -> None:
    session_id, turn_id = _new_turn(suffix="one")
    _insert_task_and_run(
        session_id=session_id,
        turn_id=turn_id,
        task_id="task-linked",
        node_id="node-linked",
        work_run_id="run-linked",
        status="waiting_user",
        linked=True,
        created_at="2026-08-14T00:00:01+00:00",
    )
    _insert_task_and_run(
        session_id=session_id,
        turn_id=turn_id,
        task_id="task-unlinked",
        node_id="node-unlinked",
        work_run_id="run-unlinked",
        status="interrupted",
        linked=False,
        created_at="2026-08-14T00:00:02+00:00",
    )
    _insert_task_and_run(
        session_id=session_id,
        turn_id=turn_id,
        task_id="task-terminal",
        node_id="node-terminal",
        work_run_id="run-terminal",
        status="completed",
        linked=True,
        created_at="2026-08-14T00:00:03+00:00",
    )

    candidates = work_run_store.list_turn_linked_nonterminal_work_runs(
        session_id=session_id,
        turn_id=turn_id,
    )

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.work_run_id == "run-linked"
    assert candidate.subject.task_id == "task-linked"
    assert candidate.status is WorkRunStatus.WAITING_USER


def test_discovery_returns_all_candidates_in_stable_task_link_order() -> None:
    session_id, turn_id = _new_turn(suffix="many")
    _insert_task_and_run(
        session_id=session_id,
        turn_id=turn_id,
        task_id="task-z-first-linked",
        node_id="node-z",
        work_run_id="run-z",
        status="waiting_user",
        linked=True,
        created_at="2026-08-14T00:00:02+00:00",
    )
    _insert_task_and_run(
        session_id=session_id,
        turn_id=turn_id,
        task_id="task-a-second-linked",
        node_id="node-a",
        work_run_id="run-a",
        status="interrupted",
        linked=True,
        created_at="2026-08-14T00:00:01+00:00",
    )

    first = work_run_store.list_turn_linked_nonterminal_work_runs(
        session_id=session_id,
        turn_id=turn_id,
    )
    second = work_run_store.list_turn_linked_nonterminal_work_runs(
        session_id=session_id,
        turn_id=turn_id,
    )

    assert tuple(item.work_run_id for item in first) == ("run-z", "run-a")
    assert second == first


def test_discovery_rejects_a_turn_from_another_session() -> None:
    first_session_id, _ = _new_turn(suffix="cross-a")
    _, second_turn_id = _new_turn(suffix="cross-b")

    with pytest.raises(work_run_store.WorkExecutionPersistenceError, match="outside"):
        work_run_store.list_turn_linked_nonterminal_work_runs(
            session_id=first_session_id,
            turn_id=second_turn_id,
        )


def test_exact_turn_link_projection_includes_terminal_runs_without_scope_leakage() -> None:
    first_session_id, first_turn_id = _new_turn(suffix="linked-ids-first")
    second_turn_id = "turn-linked-ids-second"
    store.create_runtime_turn(
        turn_id=second_turn_id,
        session_id=first_session_id,
        source="runtime_test",
        user_text="另一个 Turn",
    )
    second_session_id, other_session_turn_id = _new_turn(
        suffix="linked-ids-other-session"
    )

    for task_id, node_id, work_run_id, turn_id, created_at in (
        (
            "task-first-late",
            "node-first-late",
            "run-first-late",
            first_turn_id,
            "2026-08-14T00:00:01+00:00",
        ),
        (
            "task-first-early",
            "node-first-early",
            "run-first-early",
            first_turn_id,
            "2026-08-14T00:00:02+00:00",
        ),
        (
            "task-second-turn",
            "node-second-turn",
            "run-second-turn",
            second_turn_id,
            "2026-08-14T00:00:03+00:00",
        ),
    ):
        _insert_task_and_run(
            session_id=first_session_id,
            turn_id=turn_id,
            task_id=task_id,
            node_id=node_id,
            work_run_id=work_run_id,
            status="completed",
            linked=True,
            created_at=created_at,
        )
    _insert_task_and_run(
        session_id=second_session_id,
        turn_id=other_session_turn_id,
        task_id="task-other-session",
        node_id="node-other-session",
        work_run_id="run-other-session",
        status="completed",
        linked=True,
        created_at="2026-08-14T00:00:04+00:00",
    )

    # 以乱序方式插入，证明稳定响应顺序由 link_revision 决定，而非行插入顺序
    # 或 WorkRun 创建时间。
    _link_work_run(
        session_id=first_session_id,
        turn_id=first_turn_id,
        work_run_id="run-first-late",
        link_revision=2,
    )
    _link_work_run(
        session_id=first_session_id,
        turn_id=first_turn_id,
        work_run_id="run-first-early",
        link_revision=1,
    )
    _link_work_run(
        session_id=first_session_id,
        turn_id=second_turn_id,
        work_run_id="run-second-turn",
        link_revision=1,
    )
    _link_work_run(
        session_id=second_session_id,
        turn_id=other_session_turn_id,
        work_run_id="run-other-session",
        link_revision=1,
    )

    expected = ("run-first-early", "run-first-late")
    assert work_run_store.list_turn_linked_work_run_ids(
        session_id=first_session_id,
        turn_id=first_turn_id,
    ) == expected
    assert work_run_store.list_turn_linked_work_run_ids(
        session_id=first_session_id,
        turn_id=first_turn_id,
    ) == expected
    assert work_run_store.list_turn_linked_work_run_ids(
        session_id=first_session_id,
        turn_id=second_turn_id,
    ) == ("run-second-turn",)
    assert work_run_store.list_turn_linked_work_run_ids(
        session_id=second_session_id,
        turn_id=other_session_turn_id,
    ) == ("run-other-session",)
    with pytest.raises(work_run_store.WorkExecutionPersistenceError, match="outside"):
        work_run_store.list_turn_linked_work_run_ids(
            session_id=first_session_id,
            turn_id=other_session_turn_id,
        )
