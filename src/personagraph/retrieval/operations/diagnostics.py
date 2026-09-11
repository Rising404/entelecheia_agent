"""Content-free operational views for the current retrieval foundation."""

from __future__ import annotations

from pathlib import Path
import sqlite3
from typing import Any

from ..foundation import build_file_retrieval_foundation
from ..lifecycle.outbox import OutboxStatus, SqliteRetrievalOutbox, now_utc


def retrieval_health_snapshot(*, db_path: Path | str | None = None) -> dict[str, object]:
    """报告目录和协调状态，不返回 Source 正文。"""

    foundation = build_file_retrieval_foundation(db_path=db_path)
    reconcile = foundation.reconciler.scan()
    return {
        **foundation.diagnostic_snapshot(),
        "reconciliation": {
            "data_version_id": reconcile.data_version_id,
            "checked_units": reconcile.checked_units,
            "healthy": reconcile.healthy,
            "issue_counts": _issue_counts(reconcile),
        },
    }


def authority_outbox_snapshot(
    *,
    authority_db_path: Path | str,
    limit: int = 20,
) -> dict[str, object]:
    """检查一个权威 Outbox，不读取或修改来源正文。

    Outbox 同时存在于 Session 和记忆权威数据库中。本函数刻意要求显式路径，而不猜测
    来源：操作人员必须明确知道自己要检查哪个权威源。
    """

    if limit <= 0:
        raise ValueError("limit must be greater than zero")
    path = Path(authority_db_path).expanduser()
    if not path.is_file():
        return _unavailable_outbox_snapshot(path, "authority_database_not_found")
    try:
        connection = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
        connection.row_factory = sqlite3.Row
    except sqlite3.Error as exc:
        return _unavailable_outbox_snapshot(path, f"authority_database_unreadable:{type(exc).__name__}")
    try:
        if not _has_outbox_table(connection):
            return _unavailable_outbox_snapshot(path, "authority_outbox_not_initialized")
        counts = {status.value: 0 for status in OutboxStatus}
        for row in connection.execute(
            "SELECT status, COUNT(*) AS count FROM retrieval_update_outbox GROUP BY status"
        ):
            counts[str(row["status"])] = int(row["count"])
        terminal_rows = connection.execute(
            "SELECT event_id, kind, source_type, source_unit_id, source_revision, "
            "indexed_content_hash, data_version_id, occurred_at, attempts, reason_code, updated_at "
            "FROM retrieval_update_outbox WHERE status=? "
            "ORDER BY updated_at DESC, event_id DESC LIMIT ?",
            (OutboxStatus.TERMINAL_FAILED.value, limit),
        ).fetchall()
        manual_requeue_available = _has_manual_action_table(connection)
        manual_rows = connection.execute(
            "SELECT action_id, event_id, action, actor, reason, previous_status, "
            "previous_reason_code, occurred_at FROM retrieval_outbox_manual_actions "
            "ORDER BY occurred_at DESC, action_id DESC LIMIT ?",
            (limit,),
        ).fetchall() if manual_requeue_available else []
        attempt_audit_available = _has_attempt_audit_table(connection)
        attempt_audit_count = (
            int(
                connection.execute(
                    "SELECT COUNT(*) FROM retrieval_outbox_attempt_audits"
                ).fetchone()[0]
            )
            if attempt_audit_available
            else 0
        )
        attempt_rows = connection.execute(
            "SELECT a.event_id, a.attempt, a.worker_kind, a.worker_instance_hash, "
            "a.batch_id, a.batch_limit, a.batch_size, a.batch_ordinal, a.outcome, "
            "a.failure_stage, a.safe_error_code, a.occurred_at, o.kind, "
            "o.source_type, o.source_unit_id, o.data_version_id "
            "FROM retrieval_outbox_attempt_audits AS a "
            "JOIN retrieval_update_outbox AS o ON o.event_id=a.event_id "
            "ORDER BY a.occurred_at DESC, a.event_id, a.attempt DESC LIMIT ?",
            (limit,),
        ).fetchall() if attempt_audit_available else []
        return {
            "authority_db_path": str(path),
            "available": True,
            "manual_requeue_available": manual_requeue_available,
            "manual_requeue_reason_code": None if manual_requeue_available else "authority_outbox_manual_actions_not_initialized",
            "attempt_audit_available": attempt_audit_available,
            "attempt_audit_count": attempt_audit_count,
            "attempt_audit_truncated": attempt_audit_count > len(attempt_rows),
            "status_counts": counts,
            "terminal_failures": [_terminal_failure_row(row) for row in terminal_rows],
            "attempt_audits": [_attempt_audit_row(row) for row in attempt_rows],
            "manual_actions": [_manual_action_row(row) for row in manual_rows],
        }
    finally:
        connection.close()


def requeue_authority_outbox_terminal_failure(
    *,
    authority_db_path: Path | str,
    event_id: str,
    actor: str,
    reason: str,
    occurred_at: str | None = None,
) -> dict[str, object]:
    """审计并重新入队权威 Outbox 中恰好一个终态事件。

    这是刻意收窄的运维动作。它既不绕过常规 Outbox 消费者，也不直接写入派生检索数据。
    """

    path = Path(authority_db_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"authority database not found: {path}")
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        if not _has_outbox_table(conn):
            raise RuntimeError("authority outbox is not initialized")
        if not _has_manual_action_table(conn):
            raise RuntimeError("authority outbox manual-requeue ledger is not initialized; migrate the authority database first")
        action = SqliteRetrievalOutbox().requeue_terminal_failure(
            conn,
            event_id=event_id,
            actor=actor,
            reason=reason,
            now=occurred_at or now_utc(),
        )
        return {
            "authority_db_path": str(path),
            "event_id": action.event_id,
            "action_id": action.action_id,
            "action": action.action,
            "actor": action.actor,
            "reason": action.reason,
            "previous_status": action.previous_status.value,
            "previous_reason_code": action.previous_reason_code,
            "occurred_at": action.occurred_at,
            "new_status": OutboxStatus.PENDING.value,
        }
    finally:
        conn.close()


def _has_outbox_table(conn: sqlite3.Connection) -> bool:
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='retrieval_update_outbox'"
    ).fetchone() is not None


def _has_manual_action_table(conn: sqlite3.Connection) -> bool:
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='retrieval_outbox_manual_actions'"
    ).fetchone() is not None


def _has_attempt_audit_table(conn: sqlite3.Connection) -> bool:
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' "
        "AND name='retrieval_outbox_attempt_audits'"
    ).fetchone() is not None


def _unavailable_outbox_snapshot(path: Path, reason_code: str) -> dict[str, object]:
    return {
        "authority_db_path": str(path),
        "available": False,
        "reason_code": reason_code,
        "manual_requeue_available": False,
        "manual_requeue_reason_code": reason_code,
        "attempt_audit_available": False,
        "attempt_audit_count": 0,
        "attempt_audit_truncated": False,
        "status_counts": {status.value: 0 for status in OutboxStatus},
        "terminal_failures": [],
        "attempt_audits": [],
        "manual_actions": [],
    }


def _terminal_failure_row(row: sqlite3.Row) -> dict[str, Any]:
    """只呈现指针级死信数据，绝不包含 Source 内容。"""

    return {
        "event_id": str(row["event_id"]),
        "kind": str(row["kind"]),
        "source_type": str(row["source_type"]),
        "source_unit_id": str(row["source_unit_id"]),
        "source_revision": str(row["source_revision"]),
        "indexed_content_hash": str(row["indexed_content_hash"]),
        "data_version_id": str(row["data_version_id"]),
        "occurred_at": str(row["occurred_at"]),
        "attempts": int(row["attempts"]),
        "reason_code": row["reason_code"],
        "updated_at": str(row["updated_at"]),
    }


def _manual_action_row(row: sqlite3.Row) -> dict[str, object]:
    return {
        "action_id": str(row["action_id"]),
        "event_id": str(row["event_id"]),
        "action": str(row["action"]),
        "actor": str(row["actor"]),
        "reason": str(row["reason"]),
        "previous_status": str(row["previous_status"]),
        "previous_reason_code": row["previous_reason_code"],
        "occurred_at": str(row["occurred_at"]),
    }


def _attempt_audit_row(row: sqlite3.Row) -> dict[str, object]:
    """正式评测可消费的索引尝试审计，不含来源数据或原始 worker ID。"""

    return {
        "event_id": str(row["event_id"]),
        "kind": str(row["kind"]),
        "source_type": str(row["source_type"]),
        "source_unit_id": str(row["source_unit_id"]),
        "data_version_id": str(row["data_version_id"]),
        "attempt": int(row["attempt"]),
        "worker_kind": str(row["worker_kind"]),
        "worker_instance_hash": str(row["worker_instance_hash"]),
        "batch_id": str(row["batch_id"]),
        "batch_limit": int(row["batch_limit"]),
        "batch_size": int(row["batch_size"]),
        "batch_ordinal": int(row["batch_ordinal"]),
        "outcome": str(row["outcome"]),
        "failure_stage": row["failure_stage"],
        "safe_error_code": row["safe_error_code"],
        "occurred_at": str(row["occurred_at"]),
    }


def _issue_counts(reconcile) -> dict[str, int]:
    counts: dict[str, int] = {}
    for issue in reconcile.issues:
        counts[issue.code.value] = counts.get(issue.code.value, 0) + 1
    return dict(sorted(counts.items()))
