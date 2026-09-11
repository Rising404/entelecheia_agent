"""持久 SessionContext 清除操作与重置边界查询。

清除操作只移除派生 SessionContext 表。对话、证据和工作记忆仍是权威数据。
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from typing import Any, Sequence

from .. import store as session_store
from .models import EvidenceRecord


class ContextResetError(RuntimeError):
    """无法在保证安全的情况下应用清除请求时抛出。"""

    def __init__(self, code: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(code)
        self.code = code
        self.details = details or {}


def _connect() -> sqlite3.Connection:
    return session_store.connect_session_context_authority()


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _reset_id(session_id: str, request_id: str) -> str:
    raw = f"{session_id}\x1f{request_id}"
    return "reset:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]


def _row_to_reset(row: sqlite3.Row) -> dict[str, Any]:
    report = json.loads(str(row["report_json"]))
    if not isinstance(report, dict):
        raise ContextResetError("reset_report_invalid", details={"reset_id": row["reset_id"]})
    return report


def clear_session_context(
    session_id: str,
    *,
    request_id: str,
    actor: str,
    reason: str,
) -> dict[str, Any]:
    """原子清除派生状态，并持久化幂等重置边界。"""
    request_id = request_id.strip()
    actor = actor.strip()
    reason = reason.strip()
    if not request_id or len(request_id) > 200:
        raise ContextResetError("request_id_invalid")
    if not actor or len(actor) > 200:
        raise ContextResetError("actor_invalid")
    if not reason or len(reason) > 1000:
        raise ContextResetError("reason_invalid")

    with _connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        if conn.execute("SELECT 1 FROM sessions WHERE id=?", (session_id,)).fetchone() is None:
            raise ContextResetError("unknown_session", details={"session_id": session_id})

        existing = conn.execute(
            "SELECT * FROM session_context_resets WHERE session_id=? AND request_id=?",
            (session_id, request_id),
        ).fetchone()
        if existing is not None:
            if str(existing["actor"]) != actor or str(existing["reason"]) != reason:
                raise ContextResetError(
                    "request_id_collision",
                    details={"session_id": session_id, "request_id": request_id},
                )
            return _row_to_reset(existing)

        created_at = datetime.now(timezone.utc).isoformat()
        reset_id = _reset_id(session_id, request_id)
        cutoff_row = conn.execute(
            "SELECT COALESCE(MAX(turn_idx), -1) AS cutoff FROM session_turns WHERE session_id=?",
            (session_id,),
        ).fetchone()
        cutoff_turn_idx = int(cutoff_row["cutoff"])
        event_cutoff_row = conn.execute(
            "SELECT COALESCE(MAX(rowid), -1) AS cutoff FROM session_evidence_events"
            " WHERE session_id=?",
            (session_id,),
        ).fetchone()
        cutoff_event_rowid = int(event_cutoff_row["cutoff"])
        table_names = (
            "session_state_items",
            "session_state_transitions",
            "session_observation_candidates",
        )
        cleared_counts = {
            table: int(conn.execute(
                f"SELECT COUNT(*) FROM {table} WHERE session_id=?", (session_id,)
            ).fetchone()[0])
            for table in table_names
        }

        for table in table_names:
            conn.execute(f"DELETE FROM {table} WHERE session_id=?", (session_id,))

        report = {
            "reset_id": reset_id,
            "session_id": session_id,
            "request_id": request_id,
            "cutoff_turn_idx": cutoff_turn_idx,
            "cutoff_event_rowid": cutoff_event_rowid,
            "created_at": created_at,
            "actor": actor,
            "reason": reason,
            "cleared_counts": cleared_counts,
            "preserved": {
                "transcript": True,
                "evidence_events": True,
                "working_memory": True,
            },
        }
        conn.execute(
            "INSERT INTO session_context_resets"
            " (reset_id, session_id, request_id, cutoff_turn_idx, cutoff_event_rowid,"
            " created_at, actor, reason, report_json)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                reset_id,
                session_id,
                request_id,
                cutoff_turn_idx,
                cutoff_event_rowid,
                created_at,
                actor,
                reason,
                _json(report),
            ),
        )
    return report


def list_resets(session_id: str) -> list[dict[str, Any]]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM session_context_resets"
            " WHERE session_id=? ORDER BY created_at, reset_id",
            (session_id,),
        ).fetchall()
    return [_row_to_reset(row) for row in rows]


def latest_reset(session_id: str) -> dict[str, Any] | None:
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM session_context_resets"
            " WHERE session_id=? ORDER BY created_at DESC, reset_id DESC LIMIT 1",
            (session_id,),
        ).fetchone()
    return _row_to_reset(row) if row is not None else None


def evidence_after_latest_reset(
    session_id: str,
    records: Sequence[EvidenceRecord],
) -> tuple[list[EvidenceRecord], dict[str, Any] | None]:
    """按最新持久重置边界过滤权威证据。"""
    boundary = latest_reset(session_id)
    if boundary is None:
        return list(records), None
    cutoff = int(boundary["cutoff_turn_idx"])
    event_cutoff = int(boundary["cutoff_event_rowid"])
    with _connect() as conn:
        post_reset_event_ids = {
            str(row["id"])
            for row in conn.execute(
                "SELECT id FROM session_evidence_events"
                " WHERE session_id=? AND rowid>?",
                (session_id, event_cutoff),
            ).fetchall()
        }
    output = []
    for record in records:
        if record.session_id != session_id:
            continue
        turn_idx = record.metadata.get("turn_idx")
        if turn_idx is not None:
            if int(turn_idx) > cutoff:
                output.append(record)
        elif record.id in post_reset_event_ids:
            output.append(record)
    return output, boundary
