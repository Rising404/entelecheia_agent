"""持久执行投影按真实 Attempt 顺序提供工具结果，而非不透明 ID 的字典序。"""

import hashlib
import json
import sqlite3

import pytest

from personagraph.runtime.l1.semantic_evidence import project_review_evidence
from personagraph.session.persistence.deps import StoreDeps
from personagraph.session.persistence.l1.turn_runs import get_l1_turn_execution


@pytest.fixture
def execution(tmp_path):
    path = tmp_path / "execution.sqlite"

    def connect():
        conn = sqlite3.connect(path)
        conn.row_factory = sqlite3.Row
        return conn

    with connect() as conn:
        conn.executescript("""
            CREATE TABLE l1_turn_runs (
                l1_turn_run_id TEXT PRIMARY KEY, session_id TEXT, turn_id TEXT,
                run_revision INTEGER);
            CREATE TABLE l1_turn_run_states (l1_turn_run_id TEXT);
            CREATE TABLE l1_turn_steps (
                step_id TEXT PRIMARY KEY, l1_turn_run_id TEXT, ordinal INTEGER,
                observation_json TEXT, observation_hash TEXT);
            CREATE TABLE l1_turn_tool_calls (
                tool_call_id TEXT PRIMARY KEY, l1_turn_run_id TEXT, step_id TEXT,
                call_ordinal INTEGER, tool_id TEXT, status TEXT,
                outcome_json TEXT, outcome_hash TEXT);
            CREATE TABLE l1_turn_plan_revisions (l1_turn_run_id TEXT, revision INTEGER);
            INSERT INTO l1_turn_runs VALUES ('run', 'session', 'turn', 1);
        """)
        for ordinal in range(1, 11):
            # Deliberately reverse ID order, including two calls in the latest batch.
            step = f"step-{99 - ordinal}"
            conn.execute(
                "INSERT INTO l1_turn_steps VALUES (?, 'run', ?, NULL, NULL)",
                (step, ordinal),
            )
            for call_ordinal in ((2, 1) if ordinal == 10 else (1,)):
                outcome = json.dumps({"result": {"position": [ordinal, call_ordinal]}})
                conn.execute(
                    "INSERT INTO l1_turn_tool_calls VALUES "
                    "(?, 'run', ?, ?, 'read_pdf_text', 'succeeded', ?, ?)",
                    (f"call-{ordinal}-{call_ordinal}", step, call_ordinal, outcome,
                     hashlib.sha256(outcome.encode()).hexdigest()),
                )
        # A different run must not enter the projection even if it has newer results.
        conn.execute("INSERT INTO l1_turn_steps VALUES ('foreign', 'other', 99, NULL, NULL)")
        conn.execute(
            "INSERT INTO l1_turn_tool_calls VALUES "
            "('foreign', 'other', 'foreign', 1, 'read_pdf_text', 'succeeded', '{}', ?)",
            (hashlib.sha256(b"{}").hexdigest(),),
        )
    deps = StoreDeps(init_db=lambda: None, connect=connect, now=lambda: "", new_id=lambda: "")
    return get_l1_turn_execution(deps, session_id="session", turn_id="turn")


def test_execution_projection_orders_calls_by_attempt_then_batch_ordinal(execution):
    assert [attempt["ordinal"] for attempt in execution["attempts"]] == list(range(1, 11))
    assert [call["tool_call_id"] for call in execution["tool_calls"]] == [
        *(f"call-{ordinal}-1" for ordinal in range(1, 11)), "call-10-2",
    ]


def test_recent_review_evidence_uses_latest_persisted_results_without_references(execution):
    evidence = project_review_evidence(references=(), execution=execution)
    assert [item["result"]["position"] for item in evidence["results"]] == [
        [10, 2], [10, 1], [9, 1], [8, 1], [7, 1], [6, 1], [5, 1], [4, 1],
    ]
    assert evidence["omitted_recent_result_count"] == 3
