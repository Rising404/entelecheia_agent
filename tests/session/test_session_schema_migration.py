"""原子迁移保留会话、上下文修订、路由身份和调用账本。"""

from __future__ import annotations

import hashlib
import json

import pytest

from personagraph.session.persistence import migration, schema
from personagraph.session import store
from personagraph.runtime import entry
from personagraph.runtime.entry.routing.policy import (
    TurnRoutingPolicy,
    freeze_turn_routing_policy,
    parse_turn_routing_policy_snapshot,
)


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha(value):
    return hashlib.sha256(value.encode()).hexdigest()


def _rows(conn, table):
    return [
        tuple(r) for r in conn.execute(f'SELECT rowid,* FROM "{table}" ORDER BY rowid')
    ]


@pytest.fixture(params=("L0", "L1"))
def previous_layout(bound_partitioned_session, tmp_path, monkeypatch, request):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    session_id = bound_partitioned_session(working_dir=workspace)
    from personagraph.runtime.entry.ingress import model as ingress_model
    from personagraph.model_io.contracts import ModelResult
    from tests.helpers.prepared_model_provider import as_prepared_test_provider

    monkeypatch.setattr(
        ingress_model,
        "complete_structured",
        as_prepared_test_provider(
            lambda *_args, **kw: ModelResult(
                reply=_json({"processing_level": request.param}),
                provider="mock",
                model="mock",
                latency_ms=1,
                model_call_id=kw["model_call_id"],
            )
        ),
    )
    result = entry.run_entry_turn(
        user_input="请调用工具查询今天的日期",
        session_id=session_id,
        client_request_id="preserved-turn",
        features={"l1_semantic_verification_mode": "off"},
        routing_policy=freeze_turn_routing_policy(
            TurnRoutingPolicy(l1_enabled=True), source="request_override"
        ),
        store=store,
    )
    assert result.status == "completed"
    conn = store._connect()
    # Exercise the historical column layout without retaining a historical business schema.
    for name in (
        "logical_calls",
        "physical_attempts",
        "call_settlement_receipts",
        "rejected_outputs",
    ):
        conn.execute(
            f"ALTER TABLE runtime_model_{name} RENAME TO insession_runtime_model_{name}"
        )
    for name in (
        "insession_task_id",
        "auxiliary_graph_id",
        "goal_id",
        "execution_subject_id",
    ):
        conn.execute(
            f"ALTER TABLE insession_runtime_model_logical_calls ADD COLUMN {name} TEXT"
        )
    for name in ("current_work_run_id", "current_attempt_id"):
        conn.execute(f"ALTER TABLE turn_execution_windows ADD COLUMN {name} TEXT")
    for row in conn.execute(
        "SELECT turn_id,snapshot_json FROM runtime_turn_routing_policy_snapshots"
    ).fetchall():
        payload = json.loads(row["snapshot_json"])
        payload["schema_version"] = payload["policy"]["schema_version"] = 1
        payload["policy"]["l2_enabled"] = False
        text = _json(payload)
        conn.execute(
            "UPDATE runtime_turn_routing_policy_snapshots SET snapshot_json=?,snapshot_hash=? WHERE turn_id=?",
            (text, _sha(text), row["turn_id"]),
        )
        conn.execute(
            "UPDATE l1_turn_runs SET routing_policy_snapshot_hash=? WHERE turn_id=?",
            (_sha(text), row["turn_id"]),
        )
    conn.execute("PRAGMA user_version=104")
    conn.commit()
    conn.execute("PRAGMA foreign_keys=ON")
    yield conn, session_id, result.turn_id
    conn.close()


def _migrate(conn):
    migration.migrate_session_schema(
        conn,
        create_schema=schema._create_current_schema,
        validate=schema._assert_baseline_schema,
    )


def test_migration_preserves_l1_records_context_revisions_and_model_replay(
    previous_layout,
):
    conn, session_id, turn_id = previous_layout
    unchanged = [
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
        if row[0]
        not in {
            "runtime_turn_routing_policy_snapshots",
            "l1_turn_runs",
            "turn_execution_windows",
            "insession_runtime_model_logical_calls",
        }
    ]
    before = {name: _rows(conn, name) for name in unchanged}
    logical_ids = [
        r[0]
        for r in conn.execute(
            "SELECT logical_call_id FROM insession_runtime_model_logical_calls"
        )
    ]
    run_row = conn.execute("SELECT * FROM l1_turn_runs").fetchone()
    run_before = dict(run_row) if run_row is not None else None
    if run_before is not None:
        assert logical_ids
    snapshot_before = conn.execute(
        "SELECT snapshot_hash FROM runtime_turn_routing_policy_snapshots"
    ).fetchone()[0]
    _migrate(conn)
    assert schema.schema_version(conn) == 105
    assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
    for old_name, rows in before.items():
        name = (
            old_name.removeprefix("insession_")
            if old_name.startswith("insession_runtime_model_")
            else old_name
        )
        assert _rows(conn, name) == rows, name
    snapshot = conn.execute(
        "SELECT snapshot_json,snapshot_hash FROM runtime_turn_routing_policy_snapshots"
    ).fetchone()
    assert snapshot[1] != snapshot_before
    assert parse_turn_routing_policy_snapshot(
        snapshot[0], expected_sha256=snapshot[1]
    ).allows("L1")
    run_row = conn.execute("SELECT * FROM l1_turn_runs").fetchone()
    if run_before is not None:
        run = dict(run_row)
        assert run.pop("routing_policy_snapshot_hash") == snapshot[1]
        run_before.pop("routing_policy_snapshot_hash")
        assert run == run_before
    else:
        assert run_row is None
    for logical_id in logical_ids:
        record = store.get_runtime_model_logical_call(
            session_id=session_id, logical_call_id=logical_id
        )
        assert record is not None
        assert record.physical_attempts[-1].settlement.outcome.value == "succeeded"
    assert store.get_committed_turn_pair(session_id, f"commit_{turn_id}")[
        "assistant_content"
    ]


@pytest.mark.parametrize("failure", ["policy_hash", "final_validation"])
def test_migration_failure_restores_original_database_and_foreign_keys(
    previous_layout, failure
):
    conn, _session_id, _turn_id = previous_layout
    if failure == "policy_hash":
        conn.execute(
            "UPDATE runtime_turn_routing_policy_snapshots SET snapshot_hash=?",
            ("0" * 64,),
        )
        conn.commit()
    before = tuple(conn.iterdump())

    def validate(_conn):
        raise ValueError("validation failed")

    with pytest.raises(ValueError):
        migration.migrate_session_schema(
            conn,
            create_schema=schema._create_current_schema,
            validate=validate
            if failure == "final_validation"
            else schema._assert_baseline_schema,
        )
    assert tuple(conn.iterdump()) == before
    assert schema.schema_version(conn) == 104
    assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1


def test_unrecognized_historical_layout_is_rejected_before_any_write(previous_layout):
    conn, _session_id, _turn_id = previous_layout
    before = tuple(conn.iterdump())
    with pytest.raises(schema.SchemaDriftError, match="version 104 schema differs"):
        schema.initialize_schema(conn)
    assert tuple(conn.iterdump()) == before
