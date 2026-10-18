"""将已验证的历史 schema 迁移到当前 Session 存储，原子保留 L0/L1 状态。"""

from __future__ import annotations

from collections.abc import Callable
import hashlib
import json
import sqlite3

from .current_schema import CURRENT_SCHEMA_SQL


def _quote(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _policy_payload(raw: str, expected_hash: str, *, snapshot: bool) -> tuple[str, str]:
    if _digest(raw) != expected_hash:
        raise ValueError("stored routing policy hash changed before migration")
    payload = json.loads(raw)
    policy = payload["policy"] if snapshot else payload
    if not isinstance(policy, dict) or set(policy) != {
        "schema_version",
        "l1_enabled",
        "l2_enabled",
    }:
        raise ValueError("unsupported persisted routing policy shape")
    if policy["schema_version"] != 1 or any(
        type(policy[k]) is not bool for k in ("l1_enabled", "l2_enabled")
    ):
        raise ValueError("invalid persisted routing policy")
    if policy["l1_enabled"] and policy["l2_enabled"]:
        raise ValueError("invalid persisted routing policy modes")
    current = {"schema_version": 2, "l1_enabled": policy["l1_enabled"]}
    if snapshot:
        if (
            set(payload)
            != {"schema_version", "source", "policy", "allowed_processing_levels"}
            or payload["schema_version"] != 1
        ):
            raise ValueError("unsupported persisted routing snapshot")
        previous_levels = (
            ["L0"]
            + (["L1"] if policy["l1_enabled"] else [])
            + (["L2"] if policy["l2_enabled"] else [])
        )
        if payload["allowed_processing_levels"] != previous_levels or payload[
            "source"
        ] not in {"default", "session_default", "request_override"}:
            raise ValueError("invalid persisted routing snapshot authority")
        levels = ["L0"] + (["L1"] if policy["l1_enabled"] else [])
        payload = {
            **payload,
            "schema_version": 2,
            "policy": current,
            "allowed_processing_levels": levels,
        }
    else:
        payload = current
    text = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return text, _digest(text)


def _migrate_policy_rows(conn: sqlite3.Connection) -> None:
    for row in conn.execute(
        "SELECT session_id, policy_json, policy_hash FROM session_turn_routing_policies"
    ).fetchall():
        payload, digest = _policy_payload(str(row[1]), str(row[2]), snapshot=False)
        conn.execute(
            "UPDATE session_turn_routing_policies SET policy_json=?, policy_hash=? WHERE session_id=?",
            (payload, digest, row[0]),
        )
    for row in conn.execute(
        "SELECT turn_id, snapshot_json, snapshot_hash FROM runtime_turn_routing_policy_snapshots"
    ).fetchall():
        payload, digest = _policy_payload(str(row[1]), str(row[2]), snapshot=True)
        conn.execute(
            "UPDATE runtime_turn_routing_policy_snapshots SET snapshot_json=?, snapshot_hash=? WHERE turn_id=?",
            (payload, digest, row[0]),
        )
        updated = conn.execute(
            "UPDATE l1_turn_runs SET routing_policy_snapshot_hash=? WHERE turn_id=? AND routing_policy_snapshot_hash=?",
            (digest, row[0], row[2]),
        ).rowcount
        count = conn.execute(
            "SELECT count(*) FROM l1_turn_runs WHERE turn_id=?", (row[0],)
        ).fetchone()[0]
        if updated != count:
            raise ValueError(
                "L1 run routing snapshot identity changed before migration"
            )


def migrate_session_schema(
    conn: sqlite3.Connection,
    *,
    create_schema: Callable[[sqlite3.Connection], None],
    validate: Callable[[sqlite3.Connection], None],
) -> None:
    """Only the caller's authenticated schema may reach this transactional migration."""
    if conn.in_transaction:
        raise RuntimeError(
            "schema migration requires a connection without an active transaction"
        )
    with sqlite3.connect(":memory:") as target:
        target.executescript(CURRENT_SCHEMA_SQL)
        tables = [
            r[0]
            for r in target.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
            )
        ]
        columns = {
            name: [r[1] for r in target.execute(f"PRAGMA table_info({_quote(name)})")]
            for name in tables
        }
        triggers = target.execute(
            "SELECT name,sql FROM sqlite_master WHERE type='trigger'"
        ).fetchall()
    previous_tables = [
        r[0]
        for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
    ]
    foreign_keys = int(conn.execute("PRAGMA foreign_keys").fetchone()[0])
    conn.execute("PRAGMA foreign_keys=OFF")
    conn.execute("SAVEPOINT session_schema_migration")
    try:
        conn.execute(
            "CREATE TEMP TABLE retained_turn_ids AS SELECT turn_id FROM runtime_turns WHERE processing_level IS NULL OR processing_level IN ('L0','L1')"
        )
        conn.execute(
            "CREATE TEMP TABLE removed_message_ids AS SELECT i.session_id, i.turn_idx FROM runtime_turn_inputs i WHERE i.turn_id NOT IN (SELECT turn_id FROM retained_turn_ids) UNION SELECT c.session_id, c.assistant_turn_idx FROM session_turn_commits c WHERE c.turn_id IS NOT NULL AND c.turn_id NOT IN (SELECT turn_id FROM retained_turn_ids)"
        )
        conn.execute(
            "CREATE TEMP TABLE retained_model_calls AS SELECT logical_call_id FROM insession_runtime_model_logical_calls WHERE invocation_turn_id IN (SELECT turn_id FROM retained_turn_ids) AND insession_task_id IS NULL AND auxiliary_graph_id IS NULL AND goal_id IS NULL AND execution_subject_id IS NULL"
        )
        conn.execute(
            "CREATE TEMP TABLE retained_findings AS SELECT ledger_id FROM execution_findings_ledgers WHERE owner_kind='l1_turn_run'"
        )
        conn.execute(
            "CREATE TEMP TABLE retained_trace_steps AS SELECT step_id FROM trajectory_steps "
            "WHERE turn_id IS NULL OR turn_id NOT IN (SELECT turn_id FROM runtime_turns) "
            "OR turn_id IN (SELECT turn_id FROM retained_turn_ids)"
        )
        for name in tables:
            old_name = (
                "insession_" + name if name.startswith("runtime_model_") else name
            )
            names = columns[name]
            predicates = []
            if "turn_id" in names:
                predicates.append(
                    "(turn_id IS NULL OR turn_id NOT IN (SELECT turn_id FROM runtime_turns) OR turn_id IN (SELECT turn_id FROM retained_turn_ids))"
                )
            if name.startswith("runtime_model_"):
                predicates.append(
                    "logical_call_id IN (SELECT logical_call_id FROM retained_model_calls)"
                )
            if name.startswith("execution_finding"):
                predicates.append(
                    "ledger_id IN (SELECT ledger_id FROM retained_findings)"
                )
            if name == "session_turns":
                predicates.append(
                    "(session_id,turn_idx) NOT IN (SELECT session_id,turn_idx FROM removed_message_ids)"
                )
            if name == "runtime_turn_input_file_refs":
                predicates.append(
                    "message_id IN (SELECT message_id FROM runtime_turn_inputs "
                    "WHERE turn_id IN (SELECT turn_id FROM retained_turn_ids))"
                )
            if name == "trajectory_parts":
                predicates.append(
                    "step_id IN (SELECT step_id FROM retained_trace_steps)"
                )
            if name == "trajectory_blobs":
                predicates.append(
                    "(sha256 NOT IN (SELECT blob_sha256 FROM trajectory_parts "
                    "WHERE step_id NOT IN (SELECT step_id FROM retained_trace_steps)) "
                    "OR sha256 IN (SELECT blob_sha256 FROM trajectory_parts "
                    "WHERE step_id IN (SELECT step_id FROM retained_trace_steps)))"
                )
            if name == "turn_execution_windows":
                predicates.append(
                    "current_work_run_id IS NULL AND current_attempt_id IS NULL"
                )
            where = " WHERE " + " AND ".join(predicates) if predicates else ""
            selected = ",".join(_quote(n) for n in names)
            conn.execute(
                f"CREATE TEMP TABLE {_quote('saved_' + name)} AS SELECT rowid AS saved_rowid,{selected} FROM {_quote(old_name)}{where}"
            )
        sequences = (
            conn.execute("SELECT name,seq FROM sqlite_sequence").fetchall()
            if "sqlite_sequence"
            in {r[0] for r in conn.execute("SELECT name FROM sqlite_master")}
            else []
        )
        for name in previous_tables:
            conn.execute(f"DROP TABLE {_quote(name)}")
        create_schema(conn)
        # Restoring persisted facts must not replay context-revision side effects.
        for name, _sql in triggers:
            conn.execute(f"DROP TRIGGER {_quote(name)}")
        for name in tables:
            selected = ",".join(_quote(n) for n in columns[name])
            conn.execute(
                f"INSERT INTO {_quote(name)}(rowid,{selected}) SELECT saved_rowid,{selected} FROM temp.{_quote('saved_' + name)}"
            )
        for name, seq in sequences:
            if name in tables:
                conn.execute(
                    "UPDATE sqlite_sequence SET seq=MAX(seq,?) WHERE name=?",
                    (seq, name),
                )
        _migrate_policy_rows(conn)
        # A summary that included removed messages must be rebuilt from the retained
        # transcript; unrelated Session summaries retain their exact content/version.
        conn.execute(
            "UPDATE session_working_memory SET running_summary='', summarized_upto=0, "
            "summarized_through_turn_id=NULL, status='stale', state_version=state_version+1, "
            "last_error_code=NULL WHERE EXISTS (SELECT 1 FROM removed_message_ids m "
            "WHERE m.session_id=session_working_memory.session_id "
            "AND m.turn_idx < session_working_memory.summarized_upto)"
        )
        for _name, sql in triggers:
            conn.execute(sql)
        validate(conn)
        for name in [
            *("saved_" + n for n in tables),
            "retained_turn_ids",
            "removed_message_ids",
            "retained_model_calls",
            "retained_findings",
            "retained_trace_steps",
        ]:
            conn.execute(f"DROP TABLE temp.{_quote(name)}")
        conn.execute("RELEASE SAVEPOINT session_schema_migration")
    except BaseException:
        conn.execute("ROLLBACK TO SAVEPOINT session_schema_migration")
        conn.execute("RELEASE SAVEPOINT session_schema_migration")
        raise
    finally:
        conn.execute(f"PRAGMA foreign_keys={foreign_keys}")
