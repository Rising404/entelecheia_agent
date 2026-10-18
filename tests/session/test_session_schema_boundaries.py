from __future__ import annotations

import ast
import sqlite3
from pathlib import Path

import pytest

from personagraph.session import store
from personagraph.session.persistence import schema
from personagraph.session.persistence import current_schema as current_schema_baseline
from personagraph.session.persistence.current_schema import CURRENT_SCHEMA_SQL


_SESSION_FILE_AUTHORITY_INDEXES = {
    "idx_session_workspace_active_grant",
    "idx_session_workspace_grant_history",
    "idx_session_workspace_active_write_grant",
    "idx_session_workspace_write_grant_history",
}

def _connection() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _schema_objects(conn: sqlite3.Connection) -> set[tuple[str, str]]:
    return {
        (str(row["type"]), str(row["name"]))
        for row in conn.execute(
            "SELECT type, name FROM sqlite_master WHERE name NOT LIKE 'sqlite_%'"
        ).fetchall()
    }


def test_fresh_database_creates_the_complete_current_baseline() -> None:
    conn = _connection()
    schema.initialize_schema(conn)

    assert schema.schema_version(conn) == schema.SCHEMA_VERSION == store.SCHEMA_VERSION
    assert schema._schema_fingerprint(conn) == schema.CURRENT_SCHEMA_FINGERPRINT
    assert conn.execute("PRAGMA foreign_key_check").fetchall() == []

    objects = _schema_objects(conn)
    required_tables = {
        "sessions",
        "session_turns",
        "session_working_memory",
        "runtime_turns",
        "runtime_turn_inputs",
        "runtime_turn_attachment_bindings",
        "session_workspace_read_grants",
        "session_workspace_root_observations",
        "session_workspace_write_grants",
        "turn_execution_windows",
        "runtime_model_logical_calls",
        "l1_turn_runs",
        "l1_turn_tool_calls",
        "execution_findings_ledgers",
        "doc_mounts",
        "doc_retrieval_snapshots",
        "trajectory_blobs",
        "trajectory_steps",
        "trajectory_parts",
    }
    assert required_tables <= {name for kind, name in objects if kind == "table"}
    assert _SESSION_FILE_AUTHORITY_INDEXES <= {
        name for kind, name in objects if kind == "index"
    }
    session_columns = {
        str(row["name"])
        for row in conn.execute("PRAGMA table_info(sessions)").fetchall()
    }
    assert "cascade_operation_id" not in session_columns
    turn_commit_columns = {
        str(row["name"])
        for row in conn.execute(
            "PRAGMA table_info(session_turn_commits)"
        ).fetchall()
    }
    assert {
        "history_retrieval_data_version",
        "node_delivery_id",
    }.isdisjoint(turn_commit_columns)
    assert (
        "index",
        "uq_session_turn_commits_node_delivery_id",
    ) not in objects
    assert not {
        name
        for kind, name in objects
        if kind == "table" and "paper_dossier" in name
    }


def test_current_database_reopens_without_schema_mutation() -> None:
    conn = _connection()
    schema.initialize_schema(conn)
    before = _schema_objects(conn)

    schema.initialize_schema(conn)

    assert _schema_objects(conn) == before


def test_current_database_missing_a_required_invariant_fails_closed() -> None:
    conn = _connection()
    schema.initialize_schema(conn)
    conn.execute(
        'DROP TRIGGER "trg_runtime_turn_execution_snapshot_immutable"'
    )
    conn.commit()

    with pytest.raises(schema.SchemaDriftError, match="schema differs"):
        schema.initialize_schema(conn)


def test_baseline_creation_is_atomic_and_retryable(monkeypatch) -> None:
    conn = _connection()
    statements = schema._sql_statements(CURRENT_SCHEMA_SQL)
    original = schema._sql_statements
    monkeypatch.setattr(
        schema,
        "_sql_statements",
        lambda _script: (statements[0], "THIS IS NOT VALID SQL;"),
    )

    with pytest.raises(sqlite3.OperationalError):
        schema.initialize_schema(conn)

    assert schema.schema_version(conn) == 0
    assert _schema_objects(conn) == set()

    monkeypatch.setattr(schema, "_sql_statements", original)
    schema.initialize_schema(conn)
    assert schema.schema_version(conn) == schema.SCHEMA_VERSION


@pytest.mark.parametrize(
    "version", (1, 14, 45, 92, 93, 94, 95, 97, 98, 99, 100, 101, 102)
)
def test_noncurrent_version_fails_closed(version: int) -> None:
    conn = _connection()
    conn.execute(f"PRAGMA user_version = {version}")

    with pytest.raises(
        schema.UnsupportedSchemaVersion,
        match=rf"schema version {version}.*only an empty database or version {schema.SCHEMA_VERSION}",
    ):
        schema.initialize_schema(conn)

    assert schema.schema_version(conn) == version
    assert _schema_objects(conn) == set()


def test_future_version_fails_closed() -> None:
    conn = _connection()
    future_version = schema.SCHEMA_VERSION + 1
    conn.execute(f"PRAGMA user_version = {future_version}")
    expected = (
        rf"schema version {future_version}.*only an empty database or version "
        rf"{schema.SCHEMA_VERSION}"
    )

    with pytest.raises(
        schema.UnsupportedSchemaVersion,
        match=expected,
    ):
        schema.initialize_schema(conn)

    assert schema.schema_version(conn) == future_version
    assert _schema_objects(conn) == set()


def test_unversioned_nonempty_database_fails_closed() -> None:
    conn = _connection()
    conn.execute("CREATE TABLE legacy_state (id TEXT PRIMARY KEY)")

    with pytest.raises(
        schema.UnsupportedSchemaVersion,
        match=rf"schema unversioned.*only an empty database or version {schema.SCHEMA_VERSION}",
    ):
        schema.initialize_schema(conn)

    assert _schema_objects(conn) == {("table", "legacy_state")}


def test_unversioned_database_with_only_a_view_fails_closed() -> None:
    conn = _connection()
    conn.execute("CREATE VIEW legacy_view AS SELECT 1 AS value")

    with pytest.raises(
        schema.UnsupportedSchemaVersion,
        match=rf"schema unversioned.*only an empty database or version {schema.SCHEMA_VERSION}",
    ):
        schema.initialize_schema(conn)

    assert _schema_objects(conn) == {("view", "legacy_view")}


def test_current_database_with_broken_foreign_keys_fails_closed() -> None:
    conn = _connection()
    schema.initialize_schema(conn)
    conn.commit()
    conn.execute("PRAGMA foreign_keys = OFF")
    conn.execute(
        "INSERT INTO session_turns "
        "(session_id, turn_idx, role, content, created_at) "
        "VALUES ('missing', 0, 'user', 'orphan', NULL)"
    )
    conn.commit()

    with pytest.raises(schema.SchemaDriftError, match="foreign-key integrity"):
        schema.initialize_schema(conn)


def test_session_turn_foreign_key_rejects_orphan_and_cascades() -> None:
    conn = _connection()
    schema.initialize_schema(conn)

    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO session_turns VALUES ('missing', 0, 'user', 'hello', NULL)"
        )

    conn.execute("INSERT INTO sessions (id, persona_id) VALUES ('s1', 'Entelecheia')")
    conn.execute(
        "INSERT INTO session_turns VALUES ('s1', 0, 'user', 'hello', NULL)"
    )
    conn.execute("DELETE FROM sessions WHERE id='s1'")
    assert conn.execute("SELECT COUNT(*) FROM session_turns").fetchone()[0] == 0


def test_schema_runtime_has_no_path_store_or_business_dependencies() -> None:
    for module_path in (
        Path(schema.__file__),
        Path(current_schema_baseline.__file__),
    ):
        tree = ast.parse(module_path.read_text(encoding="utf-8"))
        imported: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.append(node.module or "")
        assert not any(
            forbidden in module
            for module in imported
            for forbidden in ("paths", "store", "graph", "memory", "context_store")
        )
