"""Entry replay lane receipt gate 的独立持久化覆盖。"""

from __future__ import annotations

import ast
from pathlib import Path
import sqlite3

import pytest

from personagraph.session.insession_task_contracts import (
    InSessionTaskPersistenceError,
)
from personagraph.session.persistence.turns import entry_replay
from personagraph.session.persistence.deps import StoreDeps
from personagraph.session.persistence.turns.entry_replay import has_turn_task_execution_lane_receipt


_SCHEMA = """
CREATE TABLE sessions (id TEXT PRIMARY KEY);
CREATE TABLE runtime_turns (
    turn_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL
);
CREATE TABLE insession_task_match_apply_receipts (
    apply_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    source_turn_id TEXT NOT NULL,
    execution_lane_manifest_json TEXT,
    execution_lane_manifest_hash TEXT
);
"""


@pytest.fixture
def replay_store(tmp_path: Path) -> tuple[StoreDeps, Path]:
    database = tmp_path / "entry-replay.sqlite"
    with sqlite3.connect(database) as conn:
        conn.executescript(_SCHEMA)
        conn.executemany(
            "INSERT INTO sessions (id) VALUES (?)",
            (("session-one",), ("session-two",)),
        )
        conn.executemany(
            "INSERT INTO runtime_turns (turn_id, session_id) VALUES (?, ?)",
            (("turn-one", "session-one"), ("turn-two", "session-two")),
        )

    def connect() -> sqlite3.Connection:
        conn = sqlite3.connect(database)
        conn.row_factory = sqlite3.Row
        return conn

    return (
        StoreDeps(
            init_db=lambda: None,
            connect=connect,
            now=lambda: "unused",
            new_id=lambda: "unused",
        ),
        database,
    )


def _insert_receipt(
    database: Path,
    *,
    apply_id: str,
    manifest_json: str | None,
    manifest_hash: str | None,
) -> None:
    with sqlite3.connect(database) as conn:
        conn.execute(
            "INSERT INTO insession_task_match_apply_receipts "
            "(apply_id, session_id, source_turn_id, "
            "execution_lane_manifest_json, execution_lane_manifest_hash) "
            "VALUES (?, 'session-one', 'turn-one', ?, ?)",
            (apply_id, manifest_json, manifest_hash),
        )


def test_gate_requires_an_owned_turn_and_returns_false_without_a_receipt(
    replay_store: tuple[StoreDeps, Path],
) -> None:
    deps, _database = replay_store

    assert not has_turn_task_execution_lane_receipt(
        deps,
        "session-one",
        "turn-one",
    )
    with pytest.raises(InSessionTaskPersistenceError, match="outside"):
        has_turn_task_execution_lane_receipt(
            deps,
            "session-two",
            "turn-one",
        )
    with pytest.raises(InSessionTaskPersistenceError, match="unknown"):
        has_turn_task_execution_lane_receipt(
            deps,
            "session-one",
            "turn-missing",
        )


def test_gate_accepts_only_one_receipt_with_nonempty_manifest_fields(
    replay_store: tuple[StoreDeps, Path],
) -> None:
    deps, database = replay_store
    _insert_receipt(
        database,
        apply_id="apply-one",
        manifest_json="not-yet-validated-json",
        manifest_hash="not-yet-validated-hash",
    )

    assert has_turn_task_execution_lane_receipt(
        deps,
        "session-one",
        "turn-one",
    )

    with sqlite3.connect(database) as conn:
        conn.execute(
            "UPDATE insession_task_match_apply_receipts "
            "SET execution_lane_manifest_hash='   ' WHERE apply_id='apply-one'"
        )
    assert not has_turn_task_execution_lane_receipt(
        deps,
        "session-one",
        "turn-one",
    )

    with sqlite3.connect(database) as conn:
        conn.execute(
            "UPDATE insession_task_match_apply_receipts "
            "SET execution_lane_manifest_hash='hash-one' WHERE apply_id='apply-one'"
        )
    _insert_receipt(
        database,
        apply_id="apply-two",
        manifest_json="{}",
        manifest_hash="hash-two",
    )
    assert not has_turn_task_execution_lane_receipt(
        deps,
        "session-one",
        "turn-one",
    )


def test_gate_wraps_database_read_failures(
    replay_store: tuple[StoreDeps, Path],
) -> None:
    deps, database = replay_store
    with sqlite3.connect(database) as conn:
        conn.execute("DROP TABLE insession_task_match_apply_receipts")

    with pytest.raises(InSessionTaskPersistenceError, match="could not be read"):
        has_turn_task_execution_lane_receipt(
            deps,
            "session-one",
            "turn-one",
        )


def test_entry_replay_persistence_owner_has_no_l2_dependency() -> None:
    tree = ast.parse(Path(entry_replay.__file__).read_text(encoding="utf-8"))
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
