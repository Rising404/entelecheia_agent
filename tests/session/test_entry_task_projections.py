"""Entry Task 只读投影的独立持久化边界。"""

from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys

import pytest

from personagraph.session.insession_task_contracts import (
    InSessionTaskPersistenceError,
)
from personagraph.session.entry_task_contracts import (
    parse_entry_pending_question_decision_json,
)
from personagraph.session.persistence.deps import StoreDeps
from personagraph.session.persistence.turns.entry_tasks import (
    list_entry_pending_task_questions,
    list_entry_task_catalog,
    list_turn_insession_task_ids,
    list_turn_linked_work_run_ids,
)


_SCHEMA = """
CREATE TABLE sessions (id TEXT PRIMARY KEY);
CREATE TABLE runtime_turns (
    turn_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL
);
CREATE TABLE runtime_turn_inputs (
    session_id TEXT NOT NULL,
    turn_id TEXT NOT NULL,
    turn_idx INTEGER NOT NULL
);
CREATE TABLE session_turns (
    session_id TEXT NOT NULL,
    turn_idx INTEGER NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL
);
CREATE TABLE insession_tasks (
    insession_task_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    root_title TEXT NOT NULL,
    root_objective TEXT NOT NULL,
    current_status TEXT NOT NULL,
    current_graph_revision INTEGER,
    created_turn_id TEXT NOT NULL,
    creation_source_start INTEGER,
    creation_source_end INTEGER,
    creation_source_sha256 TEXT,
    state_version INTEGER NOT NULL DEFAULT 1,
    updated_at TEXT NOT NULL
);
CREATE TABLE insession_task_graph_revisions (
    insession_task_id TEXT NOT NULL,
    graph_revision INTEGER NOT NULL,
    source_turn_id TEXT NOT NULL,
    source_anchors_json TEXT NOT NULL,
    authorization_anchor_ids_json TEXT NOT NULL
);
CREATE TABLE insession_task_turn_links (
    link_id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    turn_id TEXT NOT NULL,
    insession_task_id TEXT NOT NULL
);
CREATE TABLE insession_work_runs (
    work_run_id TEXT PRIMARY KEY,
    execution_subject_id TEXT,
    session_id TEXT NOT NULL,
    subject_kind TEXT,
    insession_task_id TEXT,
    graph_revision INTEGER,
    insession_task_node_id TEXT,
    auxiliary_graph_id TEXT,
    auxiliary_graph_revision INTEGER,
    auxiliary_node_id TEXT,
    node_revision INTEGER,
    status TEXT NOT NULL,
    reason TEXT,
    revision INTEGER NOT NULL DEFAULT 1,
    max_attempts INTEGER NOT NULL DEFAULT 32,
    soft_active_seconds REAL NOT NULL DEFAULT 720,
    hard_active_seconds REAL NOT NULL DEFAULT 900,
    attempts_started INTEGER NOT NULL DEFAULT 0,
    active_seconds_consumed REAL NOT NULL DEFAULT 0,
    current_attempt_id TEXT,
    current_verification_request_id TEXT,
    created_at TEXT NOT NULL DEFAULT '',
    updated_at TEXT NOT NULL DEFAULT ''
);
CREATE TABLE insession_work_run_turn_links (
    link_id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    turn_id TEXT NOT NULL,
    work_run_id TEXT NOT NULL,
    link_revision INTEGER NOT NULL
);
CREATE TABLE insession_task_graph_nodes (
    insession_task_id TEXT NOT NULL,
    graph_revision INTEGER NOT NULL,
    insession_task_node_id TEXT NOT NULL,
    node_revision INTEGER NOT NULL,
    ordinal INTEGER NOT NULL
);
CREATE TABLE insession_task_graph_edges (
    insession_task_id TEXT NOT NULL,
    graph_revision INTEGER NOT NULL,
    parent_insession_task_node_id TEXT NOT NULL,
    child_insession_task_node_id TEXT NOT NULL
);
CREATE TABLE insession_task_node_states (
    insession_task_id TEXT NOT NULL,
    insession_task_node_id TEXT NOT NULL,
    node_revision INTEGER NOT NULL,
    status TEXT NOT NULL,
    state_version INTEGER NOT NULL
);
CREATE TABLE insession_task_node_execution_subject_bindings (
    binding_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    insession_task_id TEXT NOT NULL,
    graph_revision INTEGER NOT NULL,
    insession_task_node_id TEXT NOT NULL,
    node_revision INTEGER NOT NULL
);
CREATE TABLE insession_execution_subjects (
    execution_subject_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    insession_task_id TEXT NOT NULL,
    subject_kind TEXT NOT NULL,
    subject_contract_version TEXT NOT NULL,
    task_node_binding_id TEXT,
    auxiliary_v2_binding_id TEXT
);
CREATE TABLE insession_work_run_attempts (
    attempt_id TEXT PRIMARY KEY,
    work_run_id TEXT NOT NULL,
    turn_id TEXT NOT NULL,
    input_turn_id TEXT NOT NULL,
    predecessor_question_attempt_id TEXT,
    ordinal INTEGER NOT NULL,
    status TEXT NOT NULL,
    action TEXT,
    decision_json TEXT,
    progress_revision_before INTEGER,
    progress_revision_after INTEGER,
    committed_output_revision INTEGER,
    submitted_output_revision INTEGER,
    budget_before_json TEXT,
    budget_after_json TEXT,
    close_reason TEXT,
    closed_at TEXT,
    budget_charge_id TEXT
);
CREATE TABLE insession_work_run_acceptance_progress (
    work_run_id TEXT PRIMARY KEY,
    progress_revision INTEGER NOT NULL
);
CREATE TABLE insession_work_run_output_windows (
    work_run_id TEXT PRIMARY KEY,
    output_revision INTEGER NOT NULL
);
CREATE TABLE insession_work_run_budget_charges (
    budget_charge_id TEXT PRIMARY KEY,
    operation TEXT NOT NULL,
    session_id TEXT NOT NULL,
    work_run_id TEXT NOT NULL,
    turn_id TEXT NOT NULL,
    checkpoint_id TEXT NOT NULL,
    work_run_revision_before INTEGER NOT NULL,
    work_run_revision_after INTEGER NOT NULL,
    window_state_version_before INTEGER NOT NULL,
    window_state_version_after INTEGER NOT NULL,
    active_seconds_delta REAL NOT NULL,
    active_seconds_before REAL NOT NULL,
    active_seconds_after REAL NOT NULL,
    disposition TEXT NOT NULL,
    work_run_status_after TEXT NOT NULL,
    work_run_reason_after TEXT,
    created_at TEXT NOT NULL
);
CREATE TABLE insession_work_run_apply_receipts (
    apply_id TEXT PRIMARY KEY,
    operation TEXT NOT NULL,
    session_id TEXT NOT NULL,
    work_run_id TEXT NOT NULL,
    payload_hash TEXT NOT NULL,
    result_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE insession_work_run_tool_calls (
    tool_call_id TEXT PRIMARY KEY,
    work_run_id TEXT NOT NULL,
    attempt_id TEXT NOT NULL
);
CREATE TABLE insession_work_run_tool_results (
    tool_result_id TEXT PRIMARY KEY,
    work_run_id TEXT NOT NULL,
    attempt_id TEXT NOT NULL
);
CREATE TABLE insession_auxiliary_v2_node_execution_subject_bindings (
    binding_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    insession_task_id TEXT NOT NULL,
    auxiliary_graph_id TEXT NOT NULL,
    goal_id TEXT NOT NULL,
    auxiliary_graph_revision INTEGER NOT NULL,
    auxiliary_node_id TEXT NOT NULL,
    node_revision INTEGER NOT NULL,
    executor_kind TEXT NOT NULL,
    definition_sha256 TEXT NOT NULL
);
CREATE TABLE insession_auxiliary_graph_v2_containers (
    auxiliary_graph_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    insession_task_id TEXT NOT NULL,
    current_goal_id TEXT,
    current_auxiliary_graph_revision INTEGER,
    state_version INTEGER NOT NULL
);
CREATE TABLE insession_auxiliary_graph_goals (
    goal_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    insession_task_id TEXT NOT NULL,
    auxiliary_graph_id TEXT NOT NULL,
    base_task_graph_revision INTEGER,
    status TEXT NOT NULL,
    state_version INTEGER NOT NULL
);
CREATE TABLE insession_auxiliary_graph_revision_snapshots (
    auxiliary_graph_id TEXT NOT NULL,
    auxiliary_graph_revision INTEGER NOT NULL,
    insession_task_id TEXT NOT NULL,
    goal_id TEXT NOT NULL,
    structure_contract_version TEXT NOT NULL
);
CREATE TABLE insession_auxiliary_graph_revision_states_v2 (
    auxiliary_graph_id TEXT NOT NULL,
    auxiliary_graph_revision INTEGER NOT NULL,
    status TEXT NOT NULL,
    state_version INTEGER NOT NULL
);
CREATE TABLE insession_auxiliary_graph_revision_nodes_v2 (
    auxiliary_graph_id TEXT NOT NULL,
    auxiliary_graph_revision INTEGER NOT NULL,
    auxiliary_node_id TEXT NOT NULL,
    node_revision INTEGER NOT NULL,
    ordinal INTEGER NOT NULL
);
CREATE TABLE insession_auxiliary_node_definitions_v2 (
    auxiliary_graph_id TEXT NOT NULL,
    auxiliary_node_id TEXT NOT NULL,
    node_revision INTEGER NOT NULL,
    executor_kind TEXT NOT NULL,
    definition_sha256 TEXT NOT NULL
);
CREATE TABLE insession_auxiliary_node_states_v2 (
    auxiliary_graph_id TEXT NOT NULL,
    auxiliary_graph_revision INTEGER NOT NULL,
    auxiliary_node_id TEXT NOT NULL,
    node_revision INTEGER NOT NULL,
    status TEXT NOT NULL,
    state_version INTEGER NOT NULL
);
CREATE TABLE insession_auxiliary_v2_execution_apply_receipts (
    apply_id TEXT PRIMARY KEY,
    operation TEXT NOT NULL,
    session_id TEXT NOT NULL,
    insession_task_id TEXT NOT NULL,
    work_run_id TEXT NOT NULL,
    execution_subject_id TEXT NOT NULL,
    auxiliary_graph_id TEXT NOT NULL,
    goal_id TEXT NOT NULL,
    auxiliary_graph_revision INTEGER NOT NULL,
    auxiliary_node_id TEXT NOT NULL,
    node_revision INTEGER NOT NULL,
    invocation_turn_id TEXT NOT NULL,
    payload_sha256 TEXT NOT NULL,
    result_json TEXT NOT NULL,
    result_sha256 TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""


@pytest.fixture
def projection_store(tmp_path: Path) -> tuple[StoreDeps, Path]:
    database = tmp_path / "entry-task-projections.sqlite"
    with sqlite3.connect(database) as conn:
        conn.executescript(_SCHEMA)

    def connect() -> sqlite3.Connection:
        conn = sqlite3.connect(database)
        conn.row_factory = sqlite3.Row
        return conn

    return (
        StoreDeps(
            init_db=lambda: None,
            connect=connect,
            now=lambda: "2026-09-03T00:00:00+00:00",
            new_id=lambda: "unused",
        ),
        database,
    )


def _seed_turn(
    database: Path,
    *,
    session_id: str,
    turn_id: str,
    turn_idx: int,
    text: str,
) -> None:
    with sqlite3.connect(database) as conn:
        conn.execute("INSERT OR IGNORE INTO sessions (id) VALUES (?)", (session_id,))
        conn.execute(
            "INSERT INTO runtime_turns (turn_id, session_id) VALUES (?, ?)",
            (turn_id, session_id),
        )
        conn.execute(
            "INSERT INTO runtime_turn_inputs (session_id, turn_id, turn_idx) "
            "VALUES (?, ?, ?)",
            (session_id, turn_id, turn_idx),
        )
        conn.execute(
            "INSERT INTO session_turns (session_id, turn_idx, role, content) "
            "VALUES (?, ?, 'user', ?)",
            (session_id, turn_idx, text),
        )


def _insert_shell_task(
    database: Path,
    *,
    task_id: str,
    session_id: str,
    turn_id: str,
    text: str,
    status: str,
    updated_at: str,
    source_hash: str | None = None,
) -> None:
    with sqlite3.connect(database) as conn:
        conn.execute(
            "INSERT INTO insession_tasks "
            "(insession_task_id, session_id, root_title, root_objective, "
            "current_status, current_graph_revision, created_turn_id, "
            "creation_source_start, creation_source_end, "
            "creation_source_sha256, updated_at) "
            "VALUES (?, ?, ?, ?, ?, NULL, ?, 0, ?, ?, ?)",
            (
                task_id,
                session_id,
                task_id,
                f"objective-{task_id}",
                status,
                turn_id,
                len(text),
                source_hash
                or hashlib.sha256(text.encode("utf-8")).hexdigest(),
                updated_at,
            ),
        )


def _question_decision(question: str) -> str:
    return json.dumps(
        {
            "acceptance_updates": [],
            "action": {"kind": "request_user_input", "question": question},
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _seed_question_settlement(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    work_run_id: str,
    turn_id: str,
    attempt_id: str,
    question: str,
) -> str:
    apply_id = f"apply-{work_run_id}"
    decision_json = _question_decision(question)
    decision = json.loads(decision_json)
    conn.execute(
        "INSERT INTO insession_work_run_attempts "
        "(attempt_id, work_run_id, turn_id, input_turn_id, ordinal, "
        "status, action, decision_json, progress_revision_before, "
        "progress_revision_after, budget_before_json, budget_after_json, "
        "close_reason, closed_at, budget_charge_id) "
        "VALUES (?, ?, ?, ?, 1, 'closed', 'request_user_input', ?, 1, 1, "
        "?, ?, 'request_user_input', ?, ?)",
        (
            attempt_id,
            work_run_id,
            turn_id,
            turn_id,
            decision_json,
            "{}",
            "{}",
            "2026-09-03T00:00:01+00:00",
            apply_id,
        ),
    )
    created_at = "2026-09-03T00:00:01+00:00"
    conn.execute(
        "INSERT INTO insession_work_run_budget_charges VALUES "
        "(?, 'commit_attempt_decision', ?, ?, ?, ?, 2, 3, 1, 2, "
        "1.0, 0.0, 1.0, 'within_limit', 'waiting_user', 'needs_input', ?)",
        (
            apply_id,
            session_id,
            work_run_id,
            turn_id,
            f"commit_attempt_decision:{apply_id}",
            created_at,
        ),
    )
    payload = {
        "session_id": session_id,
        "turn_id": turn_id,
        "work_run_id": work_run_id,
        "attempt_id": attempt_id,
        "decision": decision,
        "expected_work_run_revision": 2,
        "expected_progress_revision": 1,
        "expected_window_revision": 1,
        "active_seconds_delta": 1.0,
    }

    def canonical(value: object) -> str:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    conn.execute(
        "INSERT INTO insession_work_run_apply_receipts VALUES "
        "(?, 'commit_attempt_decision', ?, ?, ?, ?, ?)",
        (
            apply_id,
            session_id,
            work_run_id,
            hashlib.sha256(canonical(payload).encode("utf-8")).hexdigest(),
            "{}",
            created_at,
        ),
    )
    return apply_id


def _seed_generic_pending_question(
    database: Path,
    *,
    session_id: str,
    task_id: str,
    work_run_id: str,
    question: str,
    updated_at: str,
) -> None:
    turn_id = f"turn-{work_run_id}"
    node_id = f"node-{work_run_id}"
    attempt_id = f"attempt-{work_run_id}"
    binding_id = f"binding-{work_run_id}"
    subject_id = f"subject-{work_run_id}"
    _seed_turn(
        database,
        session_id=session_id,
        turn_id=turn_id,
        turn_idx=int(work_run_id.rsplit("-", 1)[-1]),
        text=f"input for {work_run_id}",
    )
    with sqlite3.connect(database) as conn:
        conn.execute(
            "INSERT INTO insession_tasks "
            "(insession_task_id, session_id, root_title, root_objective, "
            "current_status, current_graph_revision, created_turn_id, "
            "state_version, updated_at) VALUES (?, ?, ?, ?, "
            "'awaiting_user', 1, ?, 2, ?)",
            (task_id, session_id, task_id, task_id, turn_id, updated_at),
        )
        conn.execute(
            "INSERT INTO insession_task_graph_revisions "
            "(insession_task_id, graph_revision, source_turn_id, "
            "source_anchors_json, authorization_anchor_ids_json) "
            "VALUES (?, 1, ?, '[]', '[]')",
            (task_id, turn_id),
        )
        conn.execute(
            "INSERT INTO insession_task_graph_nodes VALUES (?, 1, ?, 1, 0)",
            (task_id, node_id),
        )
        conn.execute(
            "INSERT INTO insession_task_node_states VALUES (?, ?, 1, "
            "'awaiting_user', 2)",
            (task_id, node_id),
        )
        conn.execute(
            "INSERT INTO insession_task_node_execution_subject_bindings "
            "VALUES (?, ?, ?, 1, ?, 1)",
            (binding_id, session_id, task_id, node_id),
        )
        conn.execute(
            "INSERT INTO insession_execution_subjects "
            "(execution_subject_id, session_id, insession_task_id, "
            "subject_kind, subject_contract_version, task_node_binding_id, "
            "auxiliary_v2_binding_id) VALUES "
            "(?, ?, ?, 'task_node', 'task_node_v1', ?, NULL)",
            (subject_id, session_id, task_id, binding_id),
        )
        conn.execute(
            "INSERT INTO insession_work_runs "
            "(work_run_id, execution_subject_id, session_id, subject_kind, "
            "insession_task_id, graph_revision, insession_task_node_id, "
            "node_revision, status, reason, revision, attempts_started, "
            "active_seconds_consumed, created_at, updated_at) VALUES "
            "(?, ?, ?, 'task_node', ?, 1, ?, 1, 'waiting_user', "
            "'needs_input', 3, 1, 1.0, ?, ?)",
            (
                work_run_id,
                subject_id,
                session_id,
                task_id,
                node_id,
                updated_at,
                updated_at,
            ),
        )
        conn.execute(
            "INSERT INTO insession_work_run_turn_links "
            "(session_id, turn_id, work_run_id, link_revision) "
            "VALUES (?, ?, ?, 1)",
            (session_id, turn_id, work_run_id),
        )
        _seed_question_settlement(
            conn,
            session_id=session_id,
            work_run_id=work_run_id,
            turn_id=turn_id,
            attempt_id=attempt_id,
            question=question,
        )


def _seed_auxiliary_pending_question(
    database: Path,
    *,
    session_id: str,
    task_id: str,
    work_run_id: str,
    question: str,
) -> None:
    turn_id = f"turn-{work_run_id}"
    graph_id = f"graph-{work_run_id}"
    goal_id = f"goal-{work_run_id}"
    node_id = f"node-{work_run_id}"
    attempt_id = f"attempt-{work_run_id}"
    binding_id = f"binding-{work_run_id}"
    subject_id = f"subject-{work_run_id}"
    definition_sha256 = "d" * 64
    _seed_turn(
        database,
        session_id=session_id,
        turn_id=turn_id,
        turn_idx=int(work_run_id.rsplit("-", 1)[-1]),
        text=f"input for {work_run_id}",
    )
    with sqlite3.connect(database) as conn:
        conn.execute(
            "INSERT INTO insession_tasks "
            "(insession_task_id, session_id, root_title, root_objective, "
            "current_status, current_graph_revision, created_turn_id, "
            "state_version, updated_at) VALUES (?, ?, ?, ?, "
            "'awaiting_user', 1, ?, 2, '2026-09-03T00:00:00+00:00')",
            (task_id, session_id, task_id, task_id, turn_id),
        )
        conn.execute(
            "INSERT INTO insession_auxiliary_graph_v2_containers "
            "VALUES (?, ?, ?, ?, 1, 2)",
            (graph_id, session_id, task_id, goal_id),
        )
        conn.execute(
            "INSERT INTO insession_auxiliary_graph_goals "
            "VALUES (?, ?, ?, ?, 1, 'waiting_user', 2)",
            (goal_id, session_id, task_id, graph_id),
        )
        conn.execute(
            "INSERT INTO insession_auxiliary_graph_revision_snapshots "
            "VALUES (?, 1, ?, ?, 'auxiliary-graph-revision-v2')",
            (graph_id, task_id, goal_id),
        )
        conn.execute(
            "INSERT INTO insession_auxiliary_graph_revision_states_v2 "
            "VALUES (?, 1, 'waiting_user', 2)",
            (graph_id,),
        )
        conn.execute(
            "INSERT INTO insession_auxiliary_graph_revision_nodes_v2 "
            "VALUES (?, 1, ?, 1, 0)",
            (graph_id, node_id),
        )
        conn.execute(
            "INSERT INTO insession_auxiliary_node_definitions_v2 "
            "VALUES (?, ?, 1, 'model_work_run', ?)",
            (graph_id, node_id, definition_sha256),
        )
        conn.execute(
            "INSERT INTO insession_auxiliary_node_states_v2 "
            "VALUES (?, 1, ?, 1, 'waiting_user', 2)",
            (graph_id, node_id),
        )
        conn.execute(
            "INSERT INTO insession_auxiliary_v2_node_execution_subject_bindings "
            "VALUES (?, ?, ?, ?, ?, 1, ?, 1, 'model_work_run', ?)",
            (
                binding_id,
                session_id,
                task_id,
                graph_id,
                goal_id,
                node_id,
                definition_sha256,
            ),
        )
        conn.execute(
            "INSERT INTO insession_execution_subjects "
            "(execution_subject_id, session_id, insession_task_id, "
            "subject_kind, subject_contract_version, task_node_binding_id, "
            "auxiliary_v2_binding_id) VALUES "
            "(?, ?, ?, 'auxiliary_node', 'auxiliary_node_v2', NULL, ?)",
            (subject_id, session_id, task_id, binding_id),
        )
        conn.execute(
            "INSERT INTO insession_work_runs "
            "(work_run_id, execution_subject_id, session_id, subject_kind, "
            "insession_task_id, auxiliary_graph_id, auxiliary_graph_revision, "
            "auxiliary_node_id, node_revision, status, reason, revision, "
            "attempts_started, active_seconds_consumed, created_at, updated_at) "
            "VALUES "
            "(?, ?, ?, 'auxiliary_node', ?, ?, 1, ?, 1, 'waiting_user', "
            "'needs_input', 3, 1, 1.0, '2026-09-03T00:00:00+00:00', "
            "'2026-09-03T00:00:00+00:00')",
            (work_run_id, subject_id, session_id, task_id, graph_id, node_id),
        )
        conn.execute(
            "INSERT INTO insession_work_run_turn_links "
            "(session_id, turn_id, work_run_id, link_revision) "
            "VALUES (?, ?, ?, 1)",
            (session_id, turn_id, work_run_id),
        )
        apply_id = _seed_question_settlement(
            conn,
            session_id=session_id,
            work_run_id=work_run_id,
            turn_id=turn_id,
            attempt_id=attempt_id,
            question=question,
        )
        result = {
            "operation": "commit_waiting_user_attempt",
            "work_run_id": work_run_id,
            "execution_subject_id": subject_id,
            "execution_subject_contract_version": "auxiliary_node_v2",
            "subject": {
                "kind": "auxiliary_node",
                "task_id": task_id,
                "auxiliary_graph_id": graph_id,
                "auxiliary_graph_revision": 1,
                "node_id": node_id,
                "node_revision": 1,
            },
            "goal_id": goal_id,
            "question_sha256": hashlib.sha256(question.encode("utf-8")).hexdigest(),
            "attempt": {
                "attempt_id": attempt_id,
                "work_run_id": work_run_id,
                "ordinal": 1,
                "status": "closed",
            },
        }
        result_json = json.dumps(
            result,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        payload_sha256 = str(
            conn.execute(
                "SELECT payload_hash FROM insession_work_run_apply_receipts "
                "WHERE apply_id=?",
                (apply_id,),
            ).fetchone()[0]
        )
        conn.execute(
            "INSERT INTO insession_auxiliary_v2_execution_apply_receipts "
            "VALUES (?, 'commit_waiting_user_attempt', ?, ?, ?, ?, ?, ?, "
            "1, ?, 1, ?, ?, ?, ?, ?)",
            (
                apply_id,
                session_id,
                task_id,
                work_run_id,
                subject_id,
                graph_id,
                goal_id,
                node_id,
                turn_id,
                payload_sha256,
                result_json,
                hashlib.sha256(result_json.encode("utf-8")).hexdigest(),
                "2026-09-03T00:00:01+00:00",
            ),
        )


def test_catalog_empty_unknown_and_invalid_limit_are_explicit(
    projection_store: tuple[StoreDeps, Path],
) -> None:
    deps, database = projection_store
    with sqlite3.connect(database) as conn:
        conn.execute("INSERT INTO sessions (id) VALUES ('session-empty')")

    assert list_entry_task_catalog(deps, "session-empty") == ()
    assert list_entry_task_catalog(deps, "not-created", limit=0) == ()
    with pytest.raises(InSessionTaskPersistenceError, match="unknown Session"):
        list_entry_task_catalog(deps, "not-created")


def test_catalog_preserves_order_and_omits_unverifiable_rows(
    projection_store: tuple[StoreDeps, Path],
) -> None:
    deps, database = projection_store
    text = "plan the current task"
    _seed_turn(
        database,
        session_id="session-catalog",
        turn_id="turn-catalog",
        turn_idx=0,
        text=text,
    )
    _insert_shell_task(
        database,
        task_id="task-b",
        session_id="session-catalog",
        turn_id="turn-catalog",
        text=text,
        status="active",
        updated_at="2026-09-03T00:00:02+00:00",
    )
    _insert_shell_task(
        database,
        task_id="task-a",
        session_id="session-catalog",
        turn_id="turn-catalog",
        text=text,
        status="active",
        updated_at="2026-09-03T00:00:02+00:00",
    )
    _insert_shell_task(
        database,
        task_id="task-terminal",
        session_id="session-catalog",
        turn_id="turn-catalog",
        text=text,
        status="completed",
        updated_at="2026-09-03T23:59:59+00:00",
    )
    _insert_shell_task(
        database,
        task_id="task-corrupt",
        session_id="session-catalog",
        turn_id="turn-catalog",
        text=text,
        status="active",
        updated_at="2026-09-03T23:59:59+00:00",
        source_hash="0" * 64,
    )
    anchor = {
        "anchor_id": "current_request",
        "source_turn_id": "turn-catalog",
        "source_kind": "current_user_instruction",
        "start": 0,
        "end": len(text),
        "excerpt": text,
    }
    with sqlite3.connect(database) as conn:
        conn.execute(
            "INSERT INTO insession_tasks "
            "(insession_task_id, session_id, root_title, root_objective, "
            "current_status, current_graph_revision, created_turn_id, "
            "updated_at) VALUES "
            "('task-graph', 'session-catalog', 'Graph', 'Graph objective', "
            "'awaiting_user', 1, 'turn-catalog', "
            "'2026-09-03T00:00:03+00:00')"
        )
        conn.execute(
            "INSERT INTO insession_task_graph_revisions "
            "(insession_task_id, graph_revision, source_turn_id, "
            "source_anchors_json, authorization_anchor_ids_json) "
            "VALUES ('task-graph', 1, 'turn-catalog', ?, ?)",
            (json.dumps([anchor]), json.dumps(["current_request"])),
        )

    catalog = list_entry_task_catalog(deps, "session-catalog")

    assert [item.insession_task_id for item in catalog] == [
        "task-graph",
        "task-a",
        "task-b",
        "task-terminal",
    ]
    assert catalog[0].goal_summary == "Graph：Graph objective"
    assert catalog[0].current_graph_revision == 1
    assert catalog[-1].status == "completed"
    assert [item.insession_task_id for item in list_entry_task_catalog(
        deps,
        "session-catalog",
        limit=2,
    )] == ["task-graph", "task-a"]


def test_pending_questions_are_one_snapshot_generic_first_and_stably_ordered(
    projection_store: tuple[StoreDeps, Path],
) -> None:
    deps, database = projection_store
    with sqlite3.connect(database) as conn:
        conn.execute("INSERT INTO sessions (id) VALUES ('session-empty-pending')")
    assert list_entry_pending_task_questions(
        deps,
        session_id="session-empty-pending",
    ) == ()
    with pytest.raises(InSessionTaskPersistenceError, match="unknown Session"):
        list_entry_pending_task_questions(deps, session_id="missing-session")

    _seed_generic_pending_question(
        database,
        session_id="session-pending",
        task_id="task-generic-later",
        work_run_id="run-2",
        question="generic later?",
        updated_at="2026-09-03T00:00:02+00:00",
    )
    _seed_generic_pending_question(
        database,
        session_id="session-pending",
        task_id="task-generic-first",
        work_run_id="run-1",
        question="generic first?",
        updated_at="2026-09-03T00:00:01+00:00",
    )
    _seed_auxiliary_pending_question(
        database,
        session_id="session-pending",
        task_id="task-aux-z",
        work_run_id="run-4",
        question="aux z?",
    )
    _seed_auxiliary_pending_question(
        database,
        session_id="session-pending",
        task_id="task-aux-a",
        work_run_id="run-3",
        question="aux a?",
    )

    pending = list_entry_pending_task_questions(
        deps,
        session_id="session-pending",
    )

    assert [item.insession_task_id for item in pending] == [
        "task-generic-first",
        "task-generic-later",
        "task-aux-a",
        "task-aux-z",
    ]
    assert [item.question for item in pending] == [
        "generic first?",
        "generic later?",
        "aux a?",
        "aux z?",
    ]


def test_pending_question_decision_is_strict_and_receipt_bound(
    projection_store: tuple[StoreDeps, Path],
) -> None:
    deps, database = projection_store
    _seed_generic_pending_question(
        database,
        session_id="session-corrupt-decision",
        task_id="task-corrupt-decision",
        work_run_id="run-1",
        question="original?",
        updated_at="2026-09-03T00:00:01+00:00",
    )
    with sqlite3.connect(database) as conn:
        conn.execute(
            "UPDATE insession_work_run_attempts SET decision_json=?",
            (_question_decision("changed after settlement?"),),
        )

    with pytest.raises(
        InSessionTaskPersistenceError,
        match="detached from its commit receipt",
    ):
        list_entry_pending_task_questions(
            deps,
            session_id="session-corrupt-decision",
        )

    with sqlite3.connect(database) as conn:
        conn.execute(
            "UPDATE insession_work_run_attempts SET decision_json=?",
            (_question_decision("original?"),),
        )
        conn.execute(
            "UPDATE insession_work_run_apply_receipts "
            "SET operation='start_attempt'"
        )
    with pytest.raises(
        InSessionTaskPersistenceError,
        match="exact generic settlement receipt",
    ):
        list_entry_pending_task_questions(
            deps,
            session_id="session-corrupt-decision",
        )

    duplicate_key = (
        '{"acceptance_updates":[],"action":{"kind":"request_user_input",'
        '"question":"one","question":"two"}}'
    )
    with pytest.raises(ValueError, match="corrupt"):
        parse_entry_pending_question_decision_json(duplicate_key)
    coercible_update = (
        '{"acceptance_updates":[{"acceptance_id":"a",'
        '"model_claimed_satisfied":1,"supporting_tool_result_ids":[],'
        '"empty_support_justification":null}],"action":'
        '{"kind":"request_user_input","question":"question?"}}'
    )
    with pytest.raises(ValueError, match="canonical"):
        parse_entry_pending_question_decision_json(coercible_update)


def test_pending_questions_fail_closed_on_authority_and_attempt_corruption(
    projection_store: tuple[StoreDeps, Path],
) -> None:
    deps, database = projection_store
    _seed_generic_pending_question(
        database,
        session_id="session-corrupt-authority",
        task_id="task-corrupt-authority",
        work_run_id="run-1",
        question="still current?",
        updated_at="2026-09-03T00:00:01+00:00",
    )
    with sqlite3.connect(database) as conn:
        conn.execute(
            "UPDATE insession_execution_subjects "
            "SET subject_contract_version='unsupported_contract'"
        )
    with pytest.raises(InSessionTaskPersistenceError, match="supported"):
        list_entry_pending_task_questions(
            deps,
            session_id="session-corrupt-authority",
        )

    with sqlite3.connect(database) as conn:
        conn.execute(
            "UPDATE insession_execution_subjects "
            "SET subject_contract_version='task_node_v1'"
        )
        conn.execute(
            "UPDATE insession_task_node_states SET status='active'"
        )
    with pytest.raises(InSessionTaskPersistenceError, match="detached"):
        list_entry_pending_task_questions(
            deps,
            session_id="session-corrupt-authority",
        )

    deps_two, database_two = projection_store
    # The fixture is function-scoped; reset the mutated row and exercise an Attempt
    # aggregate break independently in the same durable snapshot.
    with sqlite3.connect(database_two) as conn:
        conn.execute(
            "UPDATE insession_task_node_states SET status='awaiting_user'"
        )
        conn.execute(
            "UPDATE insession_work_runs SET attempts_started=2"
        )
    with pytest.raises(InSessionTaskPersistenceError, match="Attempt count"):
        list_entry_pending_task_questions(
            deps_two,
            session_id="session-corrupt-authority",
        )
    with sqlite3.connect(database_two) as conn:
        conn.execute("UPDATE insession_work_runs SET attempts_started=1")
        conn.execute(
            "INSERT INTO insession_work_run_attempts "
            "(attempt_id, work_run_id, turn_id, input_turn_id, "
            "predecessor_question_attempt_id, ordinal, status) "
            "VALUES ('consuming-attempt', 'run-1', 'turn-run-1', "
            "'turn-run-1', 'attempt-run-1', 2, 'active')"
        )
    with pytest.raises(InSessionTaskPersistenceError, match="already consumed"):
        list_entry_pending_task_questions(
            deps_two,
            session_id="session-corrupt-authority",
        )


def test_pending_task_question_requires_completed_children(
    projection_store: tuple[StoreDeps, Path],
) -> None:
    deps, database = projection_store
    _seed_generic_pending_question(
        database,
        session_id="session-child-not-ready",
        task_id="task-child-not-ready",
        work_run_id="run-1",
        question="may I continue?",
        updated_at="2026-09-03T00:00:01+00:00",
    )
    with sqlite3.connect(database) as conn:
        conn.execute(
            "INSERT INTO insession_task_graph_nodes VALUES "
            "('task-child-not-ready', 1, 'child-node', 1, 1)"
        )
        conn.execute(
            "INSERT INTO insession_task_node_states VALUES "
            "('task-child-not-ready', 'child-node', 1, 'active', 1)"
        )
        conn.execute(
            "INSERT INTO insession_task_graph_edges VALUES "
            "('task-child-not-ready', 1, 'node-run-1', 'child-node')"
        )

    with pytest.raises(InSessionTaskPersistenceError, match="execution-ready"):
        list_entry_pending_task_questions(
            deps,
            session_id="session-child-not-ready",
        )


def test_auxiliary_question_receipt_and_single_cursor_fail_closed(
    projection_store: tuple[StoreDeps, Path],
) -> None:
    deps, database = projection_store
    _seed_auxiliary_pending_question(
        database,
        session_id="session-aux-corrupt",
        task_id="task-aux-corrupt",
        work_run_id="run-1",
        question="verified question?",
    )
    with sqlite3.connect(database) as conn:
        conn.execute(
            "UPDATE insession_auxiliary_v2_execution_apply_receipts "
            "SET result_sha256=?",
            ("0" * 64,),
        )
    with pytest.raises(InSessionTaskPersistenceError, match="receipt hash"):
        list_entry_pending_task_questions(
            deps,
            session_id="session-aux-corrupt",
        )

    with sqlite3.connect(database) as conn:
        result_json = str(
            conn.execute(
                "SELECT result_json FROM "
                "insession_auxiliary_v2_execution_apply_receipts"
            ).fetchone()[0]
        )
        conn.execute(
            "UPDATE insession_auxiliary_v2_execution_apply_receipts "
            "SET result_sha256=?",
            (hashlib.sha256(result_json.encode("utf-8")).hexdigest(),),
        )
    corrupted_result = json.loads(result_json)
    corrupted_result["question_sha256"] = "0" * 64
    corrupted_result_json = json.dumps(
        corrupted_result,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    with sqlite3.connect(database) as conn:
        conn.execute(
            "UPDATE insession_auxiliary_v2_execution_apply_receipts "
            "SET result_json=?, result_sha256=?",
            (
                corrupted_result_json,
                hashlib.sha256(corrupted_result_json.encode("utf-8")).hexdigest(),
            ),
        )
    with pytest.raises(InSessionTaskPersistenceError, match="receipt owner"):
        list_entry_pending_task_questions(
            deps,
            session_id="session-aux-corrupt",
        )

    with sqlite3.connect(database) as conn:
        conn.execute(
            "UPDATE insession_auxiliary_v2_execution_apply_receipts "
            "SET result_json=?, result_sha256=?",
            (
                result_json,
                hashlib.sha256(result_json.encode("utf-8")).hexdigest(),
            ),
        )
        conn.execute(
            "INSERT INTO insession_execution_subjects "
            "(execution_subject_id, session_id, insession_task_id, "
            "subject_kind, subject_contract_version, task_node_binding_id, "
            "auxiliary_v2_binding_id) VALUES "
            "('subject-second', 'session-aux-corrupt', 'task-aux-corrupt', "
            "'auxiliary_node', 'auxiliary_node_v2', NULL, 'binding-run-1')"
        )
        conn.execute(
            "INSERT INTO insession_work_runs "
            "(work_run_id, execution_subject_id, session_id, subject_kind, "
            "insession_task_id, auxiliary_graph_id, auxiliary_graph_revision, "
            "auxiliary_node_id, node_revision, status, reason, revision, "
            "attempts_started, active_seconds_consumed, created_at, updated_at) "
            "SELECT 'run-second', 'subject-second', session_id, subject_kind, "
            "insession_task_id, auxiliary_graph_id, auxiliary_graph_revision, "
            "auxiliary_node_id, node_revision, status, reason, revision, "
            "attempts_started, active_seconds_consumed, created_at, updated_at "
            "FROM insession_work_runs WHERE work_run_id='run-1'"
        )
    with pytest.raises(InSessionTaskPersistenceError, match="single cursor"):
        list_entry_pending_task_questions(
            deps,
            session_id="session-aux-corrupt",
        )


def test_turn_link_projections_are_empty_ordered_and_scope_checked(
    projection_store: tuple[StoreDeps, Path],
) -> None:
    deps, database = projection_store
    _seed_turn(
        database,
        session_id="session-one",
        turn_id="turn-one",
        turn_idx=0,
        text="one",
    )
    _seed_turn(
        database,
        session_id="session-two",
        turn_id="turn-two",
        turn_idx=0,
        text="two",
    )

    assert list_turn_insession_task_ids(deps, "session-one", "turn-one") == ()
    assert list_turn_linked_work_run_ids(
        deps,
        session_id="session-one",
        turn_id="turn-one",
    ) == ()

    with sqlite3.connect(database) as conn:
        for task_id in ("task-first", "task-second"):
            conn.execute(
                "INSERT INTO insession_task_turn_links "
                "(session_id, turn_id, insession_task_id) "
                "VALUES ('session-one', 'turn-one', ?)",
                (task_id,),
            )
        conn.execute(
            "INSERT INTO insession_task_turn_links "
            "(session_id, turn_id, insession_task_id) "
            "VALUES ('session-one', 'turn-one', 'task-first')"
        )
        conn.executemany(
            "INSERT INTO insession_work_runs (work_run_id, session_id, status) "
            "VALUES (?, 'session-one', ?)",
            (("run-late", "completed"), ("run-early", "active")),
        )
        # 插入顺序故意与 link_revision 相反。
        conn.execute(
            "INSERT INTO insession_work_run_turn_links "
            "(session_id, turn_id, work_run_id, link_revision) "
            "VALUES ('session-one', 'turn-one', 'run-late', 2)"
        )
        conn.execute(
            "INSERT INTO insession_work_run_turn_links "
            "(session_id, turn_id, work_run_id, link_revision) "
            "VALUES ('session-one', 'turn-one', 'run-early', 1)"
        )

    assert list_turn_insession_task_ids(deps, "session-one", "turn-one") == (
        "task-first",
        "task-second",
    )
    assert list_turn_linked_work_run_ids(
        deps,
        session_id="session-one",
        turn_id="turn-one",
    ) == ("run-early", "run-late")
    with pytest.raises(InSessionTaskPersistenceError, match="outside"):
        list_turn_insession_task_ids(deps, "session-one", "turn-two")
    with pytest.raises(InSessionTaskPersistenceError, match="outside"):
        list_turn_linked_work_run_ids(
            deps,
            session_id="session-one",
            turn_id="turn-two",
        )


def test_entry_task_projection_modules_have_a_cold_l2_boundary() -> None:
    repository = Path(__file__).resolve().parents[2]
    source_paths = (
        repository / "src/personagraph/session/entry_task_contracts.py",
        repository / 'src/personagraph/session/persistence/turns/entry_tasks.py',
        repository / 'src/personagraph/session/persistence/turns/pending_questions.py',
    )
    for source_path in source_paths:
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        imported_modules = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module is not None
        }
        imported_modules.update(
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        )
        assert not any(
            name == "personagraph.l2" or name.startswith("personagraph.l2.")
            for name in imported_modules
        )

    environment = os.environ.copy()
    existing_path = environment.get("PYTHONPATH")
    source_root = str(repository / "src")
    environment["PYTHONPATH"] = (
        source_root if not existing_path else f"{source_root}{os.pathsep}{existing_path}"
    )
    probe_code = '\nimport importlib.abc\nimport sqlite3\nimport sys\nimport tempfile\nfrom pathlib import Path\n\nclass RejectL2(importlib.abc.MetaPathFinder):\n    def find_spec(self, fullname, path=None, target=None):\n        if fullname == "personagraph.l2" or fullname.startswith("personagraph.l2."):\n            raise AssertionError(f"forbidden L2 import: {fullname}")\n        return None\n\nsys.meta_path.insert(0, RejectL2())\nfrom personagraph.session.persistence.deps import StoreDeps\nfrom personagraph.session.persistence.turns.entry_tasks import (\n    list_entry_pending_task_questions,\n)\n\nwith tempfile.TemporaryDirectory() as directory:\n    database = Path(directory) / "cold-pending.sqlite"\n    with sqlite3.connect(database) as connection:\n        connection.executescript("""\n        CREATE TABLE sessions (id TEXT PRIMARY KEY);\n        CREATE TABLE insession_work_runs (\n            work_run_id TEXT PRIMARY KEY,\n            execution_subject_id TEXT,\n            session_id TEXT NOT NULL,\n            subject_kind TEXT,\n            insession_task_id TEXT,\n            graph_revision INTEGER,\n            insession_task_node_id TEXT,\n            auxiliary_graph_id TEXT,\n            auxiliary_graph_revision INTEGER,\n            auxiliary_node_id TEXT,\n            node_revision INTEGER,\n            status TEXT NOT NULL,\n            reason TEXT,\n            revision INTEGER,\n            attempts_started INTEGER,\n            current_attempt_id TEXT,\n            current_verification_request_id TEXT,\n            updated_at TEXT NOT NULL\n        );\n        CREATE TABLE insession_execution_subjects (\n            execution_subject_id TEXT PRIMARY KEY,\n            session_id TEXT NOT NULL,\n            insession_task_id TEXT NOT NULL,\n            subject_kind TEXT NOT NULL,\n            subject_contract_version TEXT NOT NULL,\n            task_node_binding_id TEXT,\n            auxiliary_v2_binding_id TEXT\n        );\n        INSERT INTO sessions (id) VALUES (\'session-cold\');\n        """)\n    def connect():\n        connection = sqlite3.connect(database)\n        connection.row_factory = sqlite3.Row\n        return connection\n    deps = StoreDeps(\n        init_db=lambda: None,\n        connect=connect,\n        now=lambda: "unused",\n        new_id=lambda: "unused",\n    )\n    assert list_entry_pending_task_questions(\n        deps,\n        session_id="session-cold",\n    ) == ()\n\nassert not any(\n    name == "personagraph.l2" or name.startswith("personagraph.l2.")\n    for name in sys.modules\n), sys.modules\n'
    probe = subprocess.run(
        [sys.executable, "-c", probe_code],
        cwd=repository,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert probe.returncode == 0, probe.stderr
