from __future__ import annotations

import ast
import sqlite3
from pathlib import Path

import pytest

from personagraph.session import store
from personagraph.session.persistence import schema
from personagraph.session.persistence import current_schema as current_schema_baseline
from personagraph.session.persistence.current_schema import CURRENT_SCHEMA_SQL


_LANE_INVARIANT_TRIGGERS = {
    "trg_turn_window_execution_lane_exclusive_insert",
    "trg_turn_window_execution_lane_exclusive_update",
}

_SESSION_FILE_AUTHORITY_INDEXES = {
    "idx_session_workspace_active_grant",
    "idx_session_workspace_grant_history",
    "idx_session_workspace_active_write_grant",
    "idx_session_workspace_write_grant_history",
}

_RETIRED_AUXILIARY_TABLES = {
    "insession_auxiliary_graph_apply_receipts",
    "insession_auxiliary_graph_commit_bindings",
    "insession_auxiliary_graph_nodes",
    "insession_auxiliary_graph_revisions",
    "insession_auxiliary_graphs",
    "insession_auxiliary_node_completions",
    "insession_auxiliary_node_states",
    "insession_auxiliary_terminal_proposal_receipts",
    "insession_auxiliary_v52_node_execution_subject_bindings",
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
        "insession_tasks",
        "insession_task_graph_revisions",
        "insession_task_graph_nodes",
        "insession_work_runs",
        "insession_work_run_attempts",
        "insession_work_run_tool_calls",
        "insession_work_run_tool_results",
        "insession_work_run_output_windows",
        "insession_auxiliary_graph_v2_containers",
        "insession_auxiliary_graph_revision_snapshots",
        "insession_runtime_model_logical_calls",
        "insession_runtime_tool_logical_calls",
        "doc_mounts",
        "doc_retrieval_snapshots",
        "trajectory_blobs",
        "trajectory_steps",
        "trajectory_parts",
    }
    assert required_tables <= {name for kind, name in objects if kind == "table"}
    table_names = {name for kind, name in objects if kind == "table"}
    assert _RETIRED_AUXILIARY_TABLES.isdisjoint(table_names)
    baseline_ddl = "\n".join(
        str(row[0] or "")
        for row in conn.execute(
            "SELECT sql FROM sqlite_master WHERE sql IS NOT NULL"
        ).fetchall()
    )
    assert all(
        marker not in baseline_ddl
        for marker in (
            "auxiliary_v52",
            "legacy-v52",
            "legacy_v52_projection",
            "auxiliary-graph-v1",
            "source_auxiliary_terminal_receipt_id",
        )
    )
    assert {
        "insession_task_graph_legacy_v41_revisions",
        "insession_task_graph_nodes_legacy_v41",
        "insession_task_graph_edges_legacy_v41",
        "insession_auxiliary_legacy_goal_bindings",
        "work_episodes",
        "work_pause_records",
        "work_plan_apply_receipts",
        "work_run_action_evidence",
        "work_run_action_evidence_apply_receipts",
        "work_run_action_reconciliations",
        "work_run_apply_receipts",
        "work_run_artifact_verification_apply_receipts",
        "work_run_artifact_verifications",
        "work_run_finish_records",
        "work_run_obligation_waivers",
        "work_run_partial_delivery_records",
        "work_run_revisions",
        "work_run_verification_apply_receipts",
        "work_run_verification_ledgers",
        "work_run_waiver_apply_receipts",
        "work_runs",
        "work_state_apply_receipts",
        "work_state_revisions",
        "session_external_purge_requests",
        "session_lifecycle_cascades",
        "session_lifecycle_cascade_targets",
        "session_lifecycle_cascade_outbox_events",
        "session_promotion_candidates",
        "retrieval_update_outbox",
        "retrieval_outbox_manual_actions",
    }.isdisjoint(table_names)
    assert {
        "idx_session_lifecycle_cascades_folder_time",
        "idx_session_lifecycle_cascade_targets_session",
        "idx_session_lifecycle_cascade_outbox_events_target",
        "idx_retrieval_outbox_due",
        "idx_retrieval_outbox_manual_actions_event",
    }.isdisjoint({name for kind, name in objects if kind == "index"})
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
    assert _LANE_INVARIANT_TRIGGERS <= {
        name for kind, name in objects if kind == "trigger"
    }
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
        'DROP TRIGGER "trg_turn_window_execution_lane_exclusive_insert"'
    )
    conn.commit()

    with pytest.raises(schema.SchemaDriftError, match="schema differs"):
        schema.initialize_schema(conn)


def test_current_lane_invariants_reject_mixed_execution_owners() -> None:
    conn = _connection()
    schema.initialize_schema(conn)
    conn.execute("INSERT INTO sessions (id, persona_id) VALUES ('s', 'Entelecheia')")
    conn.execute(
        "INSERT INTO runtime_turns "
        "(turn_id, session_id, source, user_text, status, received_at) "
        "VALUES ('turn', 's', 'test', 'hello', 'running', 'now')"
    )

    with pytest.raises(sqlite3.IntegrityError, match="mutually exclusive"):
        conn.execute(
            "INSERT INTO turn_execution_windows "
            "(turn_id, session_id, window_state, updated_at, "
            "current_work_run_id, current_l1_turn_run_id) "
            "VALUES ('turn', 's', 'active', 'now', 'work', 'l1')"
        )


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


_CANDIDATE_LEDGER_SCHEMA_103 = """
CREATE TABLE runtime_turn_file_candidate_bindings (
            session_id TEXT NOT NULL,
            turn_id TEXT NOT NULL,
            candidate_ordinal INTEGER NOT NULL
                CHECK(candidate_ordinal BETWEEN 1 AND 512),
            candidate_id TEXT NOT NULL CHECK(length(candidate_id) = 42),
            relative_path TEXT NOT NULL
                CHECK(length(relative_path) BETWEEN 1 AND 2048),
            format TEXT NOT NULL CHECK(length(format) BETWEEN 1 AND 64),
            size_bytes INTEGER NOT NULL CHECK(size_bytes >= 0),
            mtime_ns INTEGER NOT NULL CHECK(mtime_ns >= 0),
            device INTEGER NOT NULL CHECK(device >= 0),
            inode INTEGER NOT NULL CHECK(inode >= 0),
            candidate_namespace_id TEXT NOT NULL
                CHECK(length(candidate_namespace_id) BETWEEN 1 AND 256),
            authority_id TEXT NOT NULL
                CHECK(length(authority_id) BETWEEN 1 AND 256),
            frozen_identity TEXT NOT NULL
                CHECK(length(frozen_identity) BETWEEN 1 AND 256),
            workspace_boundary_sha256 TEXT NOT NULL CHECK(
                length(workspace_boundary_sha256) = 64
                AND workspace_boundary_sha256 NOT GLOB '*[^0-9a-f]*'
            ),
            workspace_read_grant_id TEXT NOT NULL
                CHECK(length(workspace_read_grant_id) BETWEEN 1 AND 200),
            created_at TEXT NOT NULL,
            PRIMARY KEY(turn_id, candidate_id),
            UNIQUE(turn_id, relative_path),
            UNIQUE(turn_id, candidate_ordinal),
            FOREIGN KEY(session_id, turn_id)
                REFERENCES runtime_turns(session_id, turn_id) ON DELETE CASCADE,
            FOREIGN KEY(workspace_read_grant_id)
                REFERENCES session_workspace_read_grants(grant_id) ON DELETE RESTRICT
        );

CREATE TRIGGER trg_runtime_turn_file_candidate_bindings_immutable
BEFORE UPDATE ON runtime_turn_file_candidate_bindings
BEGIN
    SELECT RAISE(ABORT, 'runtime Turn file candidate bindings are immutable');
END;

"""


def _schema_103_with_receipts() -> sqlite3.Connection:
    conn = _connection()
    schema.initialize_schema(conn)
    conn.executescript(_CANDIDATE_LEDGER_SCHEMA_103)
    conn.execute("PRAGMA user_version = 103")
    conn.execute("INSERT INTO sessions (id, persona_id) VALUES ('s1', 'Entelecheia')")
    conn.execute("INSERT INTO session_turns VALUES ('s1', 0, 'user', 'preserved', NULL)")
    conn.execute("INSERT INTO runtime_turns(turn_id,session_id,source,user_text,status,received_at) "
                 "VALUES ('t1','s1','user','preserved','completed','now')")
    conn.execute("INSERT INTO session_attachments(attachment_id,session_id,origin,original_name,"
                 "stored_rel_path,media_type,size_bytes,content_hash,kind,created_at) "
                 "VALUES ('a1','s1','user_upload','sample.txt','private/sample.txt',"
                 "'text/plain',1,'hash','text','now')")
    conn.execute("INSERT INTO runtime_turn_attachment_bindings VALUES ('t1','a1',0,'now')")
    conn.execute("INSERT INTO session_workspace_read_grants VALUES "
                 "('grant','s1','/workspace','1','2',?,'now',NULL)", ('a' * 64,))
    conn.execute("INSERT INTO runtime_turn_file_candidate_bindings VALUES "
                 "('s1','t1',1,?,'sample.txt','txt',1,1,1,2,'namespace','authority',"
                 "'identity',?,'grant','now')", ('candidate_' + 'a' * 32, 'a' * 64))
    conn.commit()
    return conn


def test_schema_103_retires_only_candidate_ledger_and_preserves_receipts() -> None:
    conn = _schema_103_with_receipts()
    preserved_tables = ('sessions', 'session_turns', 'runtime_turns',
                        'session_attachments', 'runtime_turn_attachment_bindings',
                        'session_workspace_read_grants')
    before = {table: list(conn.execute(f'SELECT * FROM {table}')) for table in preserved_tables}
    schema.initialize_schema(conn)
    assert schema.schema_version(conn) == 104
    assert ('table', 'runtime_turn_file_candidate_bindings') not in _schema_objects(conn)
    assert ('trigger', 'trg_runtime_turn_file_candidate_bindings_immutable') not in _schema_objects(conn)
    assert before == {table: list(conn.execute(f'SELECT * FROM {table}')) for table in preserved_tables}
    schema.initialize_schema(conn)
    assert conn.execute('PRAGMA foreign_key_check').fetchall() == []


def test_schema_103_drift_is_rejected_before_retirement() -> None:
    conn = _schema_103_with_receipts()
    conn.execute('CREATE TABLE unexpected(value TEXT)')
    with pytest.raises(schema.SchemaDriftError, match='version 103 schema differs'):
        schema.initialize_schema(conn)
    assert schema.schema_version(conn) == 103
    assert conn.execute('SELECT count(*) FROM runtime_turn_file_candidate_bindings').fetchone()[0] == 1


def test_schema_103_retirement_rolls_back_if_final_validation_fails(monkeypatch) -> None:
    conn = _schema_103_with_receipts()
    def reject(_conn):
        raise schema.SchemaDriftError('test final validation failure')
    monkeypatch.setattr(schema, '_assert_baseline_schema', reject)
    with pytest.raises(schema.SchemaDriftError, match='test final validation'):
        schema.initialize_schema(conn)
    assert schema.schema_version(conn) == 103
    assert conn.execute('SELECT count(*) FROM runtime_turn_file_candidate_bindings').fetchone()[0] == 1
