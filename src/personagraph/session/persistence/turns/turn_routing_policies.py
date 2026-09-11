"""持久 Session 默认值与不可变 AcceptedTurn 路由快照。"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from typing import Literal

from ..deps import StoreDeps


RoutingPolicySource = Literal[
    "default",
    "session_default",
    "request_override",
]


def get_session_turn_routing_policy(
    deps: StoreDeps,
    *,
    session_id: str,
) -> dict[str, object] | None:
    deps.init_db()
    with deps.connect() as conn:
        if not _table_exists(conn, "session_turn_routing_policies"):
            return None
        row = conn.execute(
            "SELECT session_id, policy_json, policy_hash, updated_at "
            "FROM session_turn_routing_policies WHERE session_id=?",
            (session_id,),
        ).fetchone()
    if row is None:
        return None
    result = dict(row)
    validate_session_policy_payload(
        policy_json=str(result["policy_json"]),
        policy_hash=str(result["policy_hash"]),
    )
    return result


def validate_snapshot_payload(
    *,
    source: str | None,
    snapshot_json: str | None,
    snapshot_hash: str | None,
) -> tuple[RoutingPolicySource, str, str] | None:
    values = (source, snapshot_json, snapshot_hash)
    if all(value is None for value in values):
        return None
    if any(value is None for value in values):
        raise ValueError("routing-policy snapshot fields must be provided together")
    assert source is not None and snapshot_json is not None and snapshot_hash is not None
    if source not in {
        "default",
        "session_default",
        "request_override",
    }:
        raise ValueError("invalid routing-policy snapshot source")
    _validate_canonical_json_hash(snapshot_json, snapshot_hash)
    parsed = json.loads(snapshot_json)
    if not isinstance(parsed, dict) or parsed.get("source") != source:
        raise ValueError("routing-policy snapshot source does not match its payload")
    return source, snapshot_json, snapshot_hash  # type: ignore[return-value]


def validate_session_policy_payload(
    *,
    policy_json: str | None,
    policy_hash: str | None,
) -> tuple[str, str] | None:
    if policy_json is None and policy_hash is None:
        return None
    if policy_json is None or policy_hash is None:
        raise ValueError("Session routing-policy fields must be provided together")
    _validate_canonical_json_hash(policy_json, policy_hash)
    if not isinstance(json.loads(policy_json), dict):
        raise ValueError("Session routing policy must be a JSON object")
    return policy_json, policy_hash


def insert_turn_snapshot_in_transaction(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    turn_id: str,
    source: RoutingPolicySource,
    snapshot_json: str,
    snapshot_hash: str,
    created_at: str,
) -> None:
    conn.execute(
        "INSERT INTO runtime_turn_routing_policy_snapshots "
        "(turn_id, session_id, source, snapshot_json, snapshot_hash, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (
            turn_id,
            session_id,
            source,
            snapshot_json,
            snapshot_hash,
            created_at,
        ),
    )


def upsert_session_policy_in_transaction(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    policy_json: str,
    policy_hash: str,
    updated_at: str,
) -> None:
    conn.execute(
        "INSERT INTO session_turn_routing_policies "
        "(session_id, policy_json, policy_hash, updated_at) VALUES (?, ?, ?, ?) "
        "ON CONFLICT(session_id) DO UPDATE SET "
        "policy_json=excluded.policy_json, policy_hash=excluded.policy_hash, "
        "updated_at=excluded.updated_at",
        (session_id, policy_json, policy_hash, updated_at),
    )


def load_turn_snapshot_in_transaction(
    conn: sqlite3.Connection,
    *,
    turn_id: str,
) -> dict[str, object] | None:
    if not _table_exists(conn, "runtime_turn_routing_policy_snapshots"):
        return None
    row = conn.execute(
        "SELECT turn_id, session_id, source, snapshot_json, snapshot_hash, created_at "
        "FROM runtime_turn_routing_policy_snapshots WHERE turn_id=?",
        (turn_id,),
    ).fetchone()
    return dict(row) if row is not None else None


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    return (
        conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
            (table,),
        ).fetchone()
        is not None
    )


def _validate_canonical_json_hash(payload: str, expected_hash: str) -> None:
    if not isinstance(payload, str) or not payload:
        raise ValueError("routing-policy JSON must be non-empty")
    if (
        not isinstance(expected_hash, str)
        or len(expected_hash) != 64
        or any(char not in "0123456789abcdef" for char in expected_hash)
    ):
        raise ValueError("routing-policy hash must be lowercase sha256")
    parsed = json.loads(payload)
    canonical = json.dumps(
        parsed,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    if canonical != payload:
        raise ValueError("routing-policy JSON must use canonical serialization")
    actual_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    if actual_hash != expected_hash:
        raise ValueError("routing-policy hash does not match its payload")


__all__ = [
    "get_session_turn_routing_policy",
    "insert_turn_snapshot_in_transaction",
    "load_turn_snapshot_in_transaction",
    "upsert_session_policy_in_transaction",
    "validate_session_policy_payload",
    "validate_snapshot_payload",
]
