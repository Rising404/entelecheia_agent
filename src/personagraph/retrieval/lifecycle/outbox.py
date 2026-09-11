"""用于权威侧检索事件的可复用 SQLite Outbox 原语。

调用方使用其现有权威事务调用 ``enqueue``。Outbox 不依赖检索数据库，因此绝不会让派生
索引成为权威源写入依赖。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import StrEnum
import hashlib
from pathlib import Path
import re
import sqlite3
import uuid

from ...configuration.paths import STATE_DIR
from ..contracts import SourceType, SourceUnitRef


class RetrievalUpdateKind(StrEnum):
    UPSERT = "upsert"
    TRASH = "trash"
    RESTORE = "restore"
    PURGE = "purge"


class OutboxStatus(StrEnum):
    PENDING = "pending"
    PROCESSING = "processing"
    APPLIED = "applied"
    RETRYABLE_FAILED = "retryable_failed"
    TERMINAL_FAILED = "terminal_failed"


class RetrievalOutboxIdempotencyConflict(RuntimeError):
    """An event ID was replayed with a different immutable envelope."""

    def __init__(self, event_id: str, conflicting_fields: Sequence[str]) -> None:
        self.event_id = event_id
        self.conflicting_fields = tuple(conflicting_fields)
        super().__init__(
            f"outbox event {event_id!r} conflicts on immutable fields: "
            + ", ".join(self.conflicting_fields)
        )


@dataclass(frozen=True, slots=True)
class RetrievalUpdateEvent:
    event_id: str
    kind: RetrievalUpdateKind
    ref: SourceUnitRef
    retrieval_data_version: str
    occurred_at: str

    def __post_init__(self) -> None:
        if not self.event_id.strip() or not self.retrieval_data_version.strip() or not self.occurred_at.strip():
            raise ValueError("event_id, retrieval_data_version and occurred_at must not be empty")


@dataclass(frozen=True, slots=True)
class OutboxTerminalFailure:
    """一个权威侧终态事件的不含内容死信视图。"""

    event: RetrievalUpdateEvent
    attempts: int
    reason_code: str | None
    updated_at: str


@dataclass(frozen=True, slots=True)
class OutboxManualAction:
    """针对死信事件的可审计人工操作。"""

    action_id: str
    event_id: str
    action: str
    actor: str
    reason: str
    previous_status: OutboxStatus
    previous_reason_code: str | None
    occurred_at: str


@dataclass(frozen=True, slots=True)
class OutboxAttemptAudit:
    """一次消费尝试的安全分类，不含正文、路径、原始 worker ID 或 traceback。"""

    event_id: str
    attempt: int
    worker_kind: str
    worker_instance_hash: str
    batch_id: str
    batch_limit: int
    batch_size: int
    batch_ordinal: int
    outcome: str
    failure_stage: str | None
    safe_error_code: str | None
    occurred_at: str


class SqliteRetrievalOutbox:
    """具有幂等认领语义的权威源本地 Outbox 持久化。"""

    def initialize(self, conn: sqlite3.Connection) -> None:
        _reject_retired_session_authority(conn)
        _reject_retired_global_memory_authority(conn)
        conn.row_factory = sqlite3.Row
        conn.executescript(_OUTBOX_SCHEMA_SQL)

    def enqueue(self, conn: sqlite3.Connection, event: RetrievalUpdateEvent) -> bool:
        """在现有权威事务内追加，不执行提交。

        ``initialize`` 必须在权威存储启动期间、此事务开始前运行。在此处运行 DDL 可能让
        SQLite 隐式提交原本应保持原子的权威写入。
        """
        _reject_retired_session_authority(conn)
        existing = self._immutable_event_row(conn, event.event_id)
        if existing is not None:
            self._validate_replay(existing, event)
            return False
        inserted = conn.execute(
            "INSERT INTO retrieval_update_outbox "
            "(event_id, kind, source_type, source_unit_id, source_revision, indexed_content_hash, "
            "data_version_id, occurred_at, status, attempts, next_retry_at, lease_token, lease_until, reason_code, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, NULL, NULL, NULL, ?) "
            "ON CONFLICT(event_id) DO NOTHING",
            (
                event.event_id,
                event.kind.value,
                event.ref.source_type.value,
                event.ref.source_unit_id,
                event.ref.source_revision,
                event.ref.indexed_content_hash,
                event.retrieval_data_version,
                event.occurred_at,
                OutboxStatus.PENDING.value,
                event.occurred_at,
                event.occurred_at,
            ),
        ).rowcount
        if inserted:
            return True
        existing = self._immutable_event_row(conn, event.event_id)
        if existing is None:
            raise RuntimeError("outbox event conflict disappeared before validation")
        self._validate_replay(existing, event)
        return False

    @staticmethod
    def _immutable_event_row(
        conn: sqlite3.Connection,
        event_id: str,
    ) -> sqlite3.Row | None:
        return conn.execute(
            "SELECT kind, source_type, source_unit_id, source_revision, "
            "indexed_content_hash, data_version_id, occurred_at "
            "FROM retrieval_update_outbox WHERE event_id=?",
            (event_id,),
        ).fetchone()

    @staticmethod
    def _validate_replay(
        row: sqlite3.Row,
        event: RetrievalUpdateEvent,
    ) -> None:
        expected = {
            "kind": event.kind.value,
            "source_type": event.ref.source_type.value,
            "source_unit_id": event.ref.source_unit_id,
            "source_revision": event.ref.source_revision,
            "indexed_content_hash": event.ref.indexed_content_hash,
            "data_version_id": event.retrieval_data_version,
            "occurred_at": event.occurred_at,
        }
        conflicting_fields = tuple(
            field_name
            for field_name, expected_value in expected.items()
            if str(row[field_name]) != expected_value
        )
        if conflicting_fields:
            raise RetrievalOutboxIdempotencyConflict(
                event.event_id,
                conflicting_fields,
            )

    def claim_due(
        self,
        conn: sqlite3.Connection,
        *,
        worker_id: str,
        now: str,
        lease_seconds: int,
        limit: int,
        allowed_source_types: Sequence[SourceType] | None = None,
        data_version_id: str | None = None,
    ) -> list[RetrievalUpdateEvent]:
        if not worker_id.strip() or lease_seconds <= 0 or limit <= 0:
            raise ValueError("worker_id, lease_seconds and limit must be valid")
        allowed = _normalize_allowed_source_types(allowed_source_types)
        if data_version_id is not None and (
            not isinstance(data_version_id, str) or not data_version_id.strip()
        ):
            raise ValueError("data_version_id must be non-empty or None")
        self.initialize(conn)
        lease_until = _add_seconds(now, lease_seconds)
        conn.execute("BEGIN IMMEDIATE")
        try:
            where_sql = (
                "((candidate.status IN (?, ?) AND candidate.next_retry_at <= ?) "
                "OR (candidate.status=? AND candidate.lease_until IS NOT NULL "
                "AND candidate.lease_until <= ?)) "
                "AND NOT EXISTS ("
                "SELECT 1 FROM retrieval_update_outbox AS predecessor "
                "WHERE predecessor.data_version_id=candidate.data_version_id "
                "AND predecessor.source_type=candidate.source_type "
                "AND predecessor.source_unit_id=candidate.source_unit_id "
                "AND predecessor.authority_sequence<candidate.authority_sequence "
                "AND predecessor.status<>? "
                "AND NOT (candidate.kind=? AND predecessor.status=?)"
                ")"
            )
            query_parameters: list[object] = [
                OutboxStatus.PENDING.value,
                OutboxStatus.RETRYABLE_FAILED.value,
                now,
                OutboxStatus.PROCESSING.value,
                now,
                OutboxStatus.APPLIED.value,
                RetrievalUpdateKind.PURGE.value,
                OutboxStatus.TERMINAL_FAILED.value,
            ]
            if allowed is not None:
                placeholders = ",".join("?" for _ in allowed)
                where_sql += f" AND candidate.source_type IN ({placeholders})"
                query_parameters.extend(source_type.value for source_type in allowed)
            if data_version_id is not None:
                where_sql += " AND candidate.data_version_id=?"
                query_parameters.append(data_version_id)
            rows = conn.execute(
                "SELECT candidate.* FROM retrieval_update_outbox AS candidate WHERE "
                + where_sql
                + " ORDER BY candidate.authority_sequence LIMIT ?",
                (*query_parameters, limit),
            ).fetchall()
            event_ids = [str(row["event_id"]) for row in rows]
            if event_ids:
                placeholders = ",".join("?" for _ in event_ids)
                conn.execute(
                    f"UPDATE retrieval_update_outbox SET status=?, attempts=attempts+1, lease_token=?, lease_until=?, reason_code=NULL, updated_at=? "
                    f"WHERE event_id IN ({placeholders})",
                    [OutboxStatus.PROCESSING.value, worker_id, lease_until, now, *event_ids],
                )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        return [self._event_from_row(row) for row in rows]

    def mark_applied(self, conn: sqlite3.Connection, *, event_id: str, worker_id: str, now: str) -> None:
        self._finish(conn, event_id=event_id, worker_id=worker_id, now=now, status=OutboxStatus.APPLIED)

    def mark_retryable_failure(
        self,
        conn: sqlite3.Connection,
        *,
        event_id: str,
        worker_id: str,
        now: str,
        retry_after_seconds: int,
        reason_code: str,
    ) -> None:
        self._finish(
            conn,
            event_id=event_id,
            worker_id=worker_id,
            now=now,
            status=OutboxStatus.RETRYABLE_FAILED,
            next_retry_at=_add_seconds(now, retry_after_seconds),
            reason_code=reason_code,
        )

    def mark_terminal_failure(
        self,
        conn: sqlite3.Connection,
        *,
        event_id: str,
        worker_id: str,
        now: str,
        reason_code: str,
    ) -> None:
        self._finish(
            conn,
            event_id=event_id,
            worker_id=worker_id,
            now=now,
            status=OutboxStatus.TERMINAL_FAILED,
            reason_code=reason_code,
        )

    def get_status(self, conn: sqlite3.Connection, event_id: str) -> OutboxStatus | None:
        outcome = self.get_outcome(conn, event_id)
        return outcome[0] if outcome is not None else None

    def get_outcome(
        self,
        conn: sqlite3.Connection,
        event_id: str,
    ) -> tuple[OutboxStatus, str | None] | None:
        """返回一项持久事件结果，不暴露 Source 内容。"""

        self.initialize(conn)
        return self.get_outcome_in_transaction(conn, event_id)

    def get_outcome_in_transaction(
        self,
        conn: sqlite3.Connection,
        event_id: str,
    ) -> tuple[OutboxStatus, str | None] | None:
        """在调用方所有的事务内读取一项结果，不执行 DDL。"""

        row = conn.execute(
            "SELECT status, reason_code FROM retrieval_update_outbox WHERE event_id=?",
            (event_id,),
        ).fetchone()
        if row is None:
            return None
        return OutboxStatus(row["status"]), row["reason_code"]

    def status_counts(self, conn: sqlite3.Connection) -> dict[OutboxStatus, int]:
        """返回所有生命周期计数，不实例化 Source 内容。"""

        self.initialize(conn)
        rows = conn.execute(
            "SELECT status, COUNT(*) AS count FROM retrieval_update_outbox GROUP BY status"
        ).fetchall()
        counts = {status: 0 for status in OutboxStatus}
        counts.update({OutboxStatus(row["status"]): int(row["count"]) for row in rows})
        return counts

    def list_terminal_failures(
        self,
        conn: sqlite3.Connection,
        *,
        limit: int = 50,
    ) -> tuple[OutboxTerminalFailure, ...]:
        """列出死信事件；调用方只会收到引用和原因码。"""

        if limit <= 0:
            raise ValueError("limit must be greater than zero")
        self.initialize(conn)
        rows = conn.execute(
            "SELECT * FROM retrieval_update_outbox WHERE status=? "
            "ORDER BY updated_at DESC, event_id LIMIT ?",
            (OutboxStatus.TERMINAL_FAILED.value, limit),
        ).fetchall()
        return tuple(
            OutboxTerminalFailure(
                event=self._event_from_row(row),
                attempts=int(row["attempts"]),
                reason_code=row["reason_code"],
                updated_at=str(row["updated_at"]),
            )
            for row in rows
        )

    @staticmethod
    def attempt_batch_id(
        *,
        worker_id: str,
        occurred_at: str,
        event_ids: Sequence[str],
    ) -> str:
        if not worker_id.strip() or not occurred_at.strip() or not event_ids:
            raise ValueError("attempt batch identity inputs must not be empty")
        payload = "\0".join((worker_id, occurred_at, *event_ids)).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:16]

    def record_attempt_audit(
        self,
        conn: sqlite3.Connection,
        *,
        event_id: str,
        worker_id: str,
        batch_id: str,
        batch_limit: int,
        batch_size: int,
        batch_ordinal: int,
        outcome: str,
        failure_stage: str | None,
        safe_error_code: str | None,
        occurred_at: str,
    ) -> OutboxAttemptAudit:
        """在消费结果同一事务内保存可供正式评测读取的安全审计。"""

        if (
            not event_id.strip()
            or not worker_id.strip()
            or not batch_id.strip()
            or not occurred_at.strip()
        ):
            raise ValueError("attempt audit identity fields must not be empty")
        if batch_limit <= 0 or batch_size <= 0 or not 1 <= batch_ordinal <= batch_size:
            raise ValueError("attempt audit batch values are invalid")
        normalized_outcome = str(outcome).strip()
        if normalized_outcome not in {
            OutboxStatus.APPLIED.value,
            OutboxStatus.RETRYABLE_FAILED.value,
            OutboxStatus.TERMINAL_FAILED.value,
        }:
            raise ValueError("attempt audit outcome is invalid")
        normalized_stage = _optional_safe_diagnostic_token(failure_stage)
        normalized_code = _optional_safe_diagnostic_token(safe_error_code)
        if normalized_outcome != OutboxStatus.APPLIED.value and (
            normalized_stage is None or normalized_code is None
        ):
            raise ValueError("failed attempt audits require a safe stage and error code")
        row = conn.execute(
            "SELECT attempts, status FROM retrieval_update_outbox WHERE event_id=?",
            (event_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"outbox event not found: {event_id}")
        if str(row["status"]) != normalized_outcome:
            raise RuntimeError("attempt audit outcome does not match the outbox event")
        audit = OutboxAttemptAudit(
            event_id=event_id,
            attempt=int(row["attempts"]),
            worker_kind=_worker_kind(worker_id),
            worker_instance_hash=hashlib.sha256(worker_id.encode("utf-8")).hexdigest()[:16],
            batch_id=batch_id,
            batch_limit=batch_limit,
            batch_size=batch_size,
            batch_ordinal=batch_ordinal,
            outcome=normalized_outcome,
            failure_stage=normalized_stage,
            safe_error_code=normalized_code,
            occurred_at=occurred_at,
        )
        conn.execute(
            "INSERT INTO retrieval_outbox_attempt_audits "
            "(event_id, attempt, worker_kind, worker_instance_hash, batch_id, "
            "batch_limit, batch_size, batch_ordinal, outcome, failure_stage, "
            "safe_error_code, occurred_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                audit.event_id,
                audit.attempt,
                audit.worker_kind,
                audit.worker_instance_hash,
                audit.batch_id,
                audit.batch_limit,
                audit.batch_size,
                audit.batch_ordinal,
                audit.outcome,
                audit.failure_stage,
                audit.safe_error_code,
                audit.occurred_at,
            ),
        )
        return audit

    def list_attempt_audits(
        self,
        conn: sqlite3.Connection,
        *,
        event_id: str | None = None,
        limit: int = 100,
    ) -> tuple[OutboxAttemptAudit, ...]:
        if limit <= 0:
            raise ValueError("limit must be greater than zero")
        self.initialize(conn)
        if event_id is None:
            rows = conn.execute(
                "SELECT * FROM retrieval_outbox_attempt_audits "
                "ORDER BY occurred_at DESC, event_id, attempt DESC LIMIT ?",
                (limit,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM retrieval_outbox_attempt_audits WHERE event_id=? "
                "ORDER BY attempt DESC LIMIT ?",
                (event_id, limit),
            ).fetchall()
        return tuple(_attempt_audit_from_row(row) for row in rows)

    def requeue_terminal_failure(
        self,
        conn: sqlite3.Connection,
        *,
        event_id: str,
        actor: str,
        reason: str,
        now: str,
    ) -> OutboxManualAction:
        """显式重新入队一个死信事件，同时保留其失败审计。

        此操作本身不执行同步。它只把终态权威事件转换回 ``pending``；之后仍必须由普通的
        租约消费者认领并处理。
        """

        self.initialize(conn)
        conn.execute("BEGIN IMMEDIATE")
        try:
            action = self.requeue_terminal_failure_in_transaction(
                conn,
                event_id=event_id,
                actor=actor,
                reason=reason,
                now=now,
            )
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        return action

    def requeue_terminal_failure_in_transaction(
        self,
        conn: sqlite3.Connection,
        *,
        event_id: str,
        actor: str,
        reason: str,
        now: str,
    ) -> OutboxManualAction:
        """在调用方所有的权威事务内重新入队一个终态事件。"""

        if not event_id.strip() or not actor.strip() or not reason.strip() or not now.strip():
            raise ValueError("event_id, actor, reason and now must not be empty")
        row = conn.execute(
            "SELECT status, reason_code FROM retrieval_update_outbox WHERE event_id=?",
            (event_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"outbox event not found: {event_id}")
        previous_status = OutboxStatus(row["status"])
        if previous_status is not OutboxStatus.TERMINAL_FAILED:
            raise ValueError("only terminal_failed events may be manually requeued")
        action = OutboxManualAction(
            action_id=uuid.uuid4().hex,
            event_id=event_id,
            action="manual_requeue",
            actor=actor,
            reason=reason,
            previous_status=previous_status,
            previous_reason_code=row["reason_code"],
            occurred_at=now,
        )
        conn.execute(
            "UPDATE retrieval_update_outbox SET status=?, next_retry_at=?, lease_token=NULL, "
            "lease_until=NULL, reason_code=NULL, updated_at=? WHERE event_id=?",
            (OutboxStatus.PENDING.value, now, now, event_id),
        )
        conn.execute(
            "INSERT INTO retrieval_outbox_manual_actions "
            "(action_id, event_id, action, actor, reason, previous_status, previous_reason_code, occurred_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                action.action_id,
                action.event_id,
                action.action,
                action.actor,
                action.reason,
                action.previous_status.value,
                action.previous_reason_code,
                action.occurred_at,
            ),
        )
        return action

    def list_manual_actions(
        self,
        conn: sqlite3.Connection,
        *,
        event_id: str | None = None,
        limit: int = 50,
    ) -> tuple[OutboxManualAction, ...]:
        if limit <= 0:
            raise ValueError("limit must be greater than zero")
        self.initialize(conn)
        if event_id is None:
            rows = conn.execute(
                "SELECT * FROM retrieval_outbox_manual_actions ORDER BY occurred_at DESC, action_id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM retrieval_outbox_manual_actions WHERE event_id=? "
                "ORDER BY occurred_at DESC, action_id DESC LIMIT ?",
                (event_id, limit),
            ).fetchall()
        return tuple(
            OutboxManualAction(
                action_id=str(row["action_id"]),
                event_id=str(row["event_id"]),
                action=str(row["action"]),
                actor=str(row["actor"]),
                reason=str(row["reason"]),
                previous_status=OutboxStatus(row["previous_status"]),
                previous_reason_code=row["previous_reason_code"],
                occurred_at=str(row["occurred_at"]),
            )
            for row in rows
        )

    def _finish(
        self,
        conn: sqlite3.Connection,
        *,
        event_id: str,
        worker_id: str,
        now: str,
        status: OutboxStatus,
        next_retry_at: str | None = None,
        reason_code: str | None = None,
    ) -> None:
        self.initialize(conn)
        updated = conn.execute(
            "UPDATE retrieval_update_outbox SET status=?, next_retry_at=COALESCE(?, next_retry_at), "
            "lease_token=NULL, lease_until=NULL, reason_code=?, updated_at=? "
            "WHERE event_id=? AND status=? AND lease_token=?",
            (
                status.value,
                next_retry_at,
                reason_code,
                now,
                event_id,
                OutboxStatus.PROCESSING.value,
                worker_id,
            ),
        ).rowcount
        if not updated:
            raise RuntimeError("outbox event is not owned by this worker lease")

    @staticmethod
    def _event_from_row(row: sqlite3.Row) -> RetrievalUpdateEvent:
        return RetrievalUpdateEvent(
            event_id=str(row["event_id"]),
            kind=RetrievalUpdateKind(row["kind"]),
            ref=SourceUnitRef(
                source_type=SourceType(row["source_type"]),
                source_unit_id=str(row["source_unit_id"]),
                source_revision=str(row["source_revision"]),
                indexed_content_hash=str(row["indexed_content_hash"]),
            ),
            retrieval_data_version=str(row["data_version_id"]),
            occurred_at=str(row["occurred_at"]),
        )


def _normalize_allowed_source_types(
    allowed_source_types: Sequence[SourceType] | None,
) -> tuple[SourceType, ...] | None:
    if allowed_source_types is None:
        return None
    if isinstance(allowed_source_types, (str, bytes)):
        raise ValueError("allowed_source_types must be a SourceType sequence")
    try:
        normalized = tuple(
            sorted(
                {SourceType(source_type) for source_type in allowed_source_types},
                key=lambda source_type: source_type.value,
            )
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("allowed_source_types contains an invalid SourceType") from exc
    if not normalized:
        raise ValueError("allowed_source_types must not be empty")
    return normalized


_SAFE_DIAGNOSTIC_TOKEN_RE = re.compile(r"[A-Za-z0-9_.:-]{1,160}")


def _optional_safe_diagnostic_token(value: object | None) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    if _SAFE_DIAGNOSTIC_TOKEN_RE.fullmatch(normalized) is None:
        return "redacted_diagnostic_code"
    return normalized


def _worker_kind(worker_id: str) -> str:
    candidate = worker_id.split(":", 1)[0].strip()
    normalized = _optional_safe_diagnostic_token(candidate)
    return normalized or "worker"


def _attempt_audit_from_row(row: sqlite3.Row) -> OutboxAttemptAudit:
    return OutboxAttemptAudit(
        event_id=str(row["event_id"]),
        attempt=int(row["attempt"]),
        worker_kind=str(row["worker_kind"]),
        worker_instance_hash=str(row["worker_instance_hash"]),
        batch_id=str(row["batch_id"]),
        batch_limit=int(row["batch_limit"]),
        batch_size=int(row["batch_size"]),
        batch_ordinal=int(row["batch_ordinal"]),
        outcome=str(row["outcome"]),
        failure_stage=row["failure_stage"],
        safe_error_code=row["safe_error_code"],
        occurred_at=str(row["occurred_at"]),
    )


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _add_seconds(value: str, seconds: int) -> str:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return (parsed + timedelta(seconds=seconds)).isoformat()


def _reject_retired_session_authority(conn: sqlite3.Connection) -> None:
    """绝不在 ``session.sqlite`` 内重新创建或写入 History Outbox。"""

    if conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='sessions'"
    ).fetchone() is not None:
        raise RuntimeError(
            "the Session History Outbox is retired; session.sqlite cannot be "
            "used as a retrieval Outbox authority"
        )


def _reject_retired_global_memory_authority(conn: sqlite3.Connection) -> None:
    """禁止调用方绕过已退役的记忆存储连接门禁。

    Outbox 实例不拥有其 SQLite 连接。因此，直接调用 ``sqlite3.connect`` 可能绕过
    ``memory.store``，并静默重建原生产 ``var/memory/memory.sqlite``，其中只有 Outbox
    表。现在只有项目 Documents 托管生产 Outbox；旧全局权威源永远不是有效生产目标。

    仍支持内存数据库，以及显式重定向的迁移或测试数据库。
    """

    row = conn.execute("PRAGMA database_list").fetchone()
    if row is None or not str(row[2] or "").strip():
        return
    connected_path = Path(str(row[2])).expanduser().resolve()
    retired_path = (STATE_DIR / "memory" / "memory.sqlite").expanduser().resolve()
    if connected_path == retired_path:
        raise RuntimeError(
            "the global memory Outbox authority is retired; use a project "
            "documents.sqlite authority"
        )


_OUTBOX_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS retrieval_update_outbox (
    authority_sequence INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id TEXT NOT NULL UNIQUE,
    kind TEXT NOT NULL CHECK(kind IN ('upsert', 'trash', 'restore', 'purge')),
    source_type TEXT NOT NULL,
    source_unit_id TEXT NOT NULL,
    source_revision TEXT NOT NULL,
    indexed_content_hash TEXT NOT NULL,
    data_version_id TEXT NOT NULL,
    occurred_at TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('pending', 'processing', 'applied', 'retryable_failed', 'terminal_failed')),
    attempts INTEGER NOT NULL DEFAULT 0 CHECK(attempts >= 0),
    next_retry_at TEXT NOT NULL,
    lease_token TEXT,
    lease_until TEXT,
    reason_code TEXT,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_retrieval_outbox_due
    ON retrieval_update_outbox(status, next_retry_at, lease_until, authority_sequence);
CREATE INDEX IF NOT EXISTS idx_retrieval_outbox_causal_predecessor
    ON retrieval_update_outbox(
        data_version_id, source_type, source_unit_id, authority_sequence, status
    );

CREATE TABLE IF NOT EXISTS retrieval_outbox_attempt_audits (
    event_id TEXT NOT NULL,
    attempt INTEGER NOT NULL CHECK(attempt >= 1),
    worker_kind TEXT NOT NULL,
    worker_instance_hash TEXT NOT NULL,
    batch_id TEXT NOT NULL,
    batch_limit INTEGER NOT NULL CHECK(batch_limit >= 1),
    batch_size INTEGER NOT NULL CHECK(batch_size >= 1),
    batch_ordinal INTEGER NOT NULL CHECK(batch_ordinal >= 1 AND batch_ordinal <= batch_size),
    outcome TEXT NOT NULL CHECK(outcome IN ('applied', 'retryable_failed', 'terminal_failed')),
    failure_stage TEXT,
    safe_error_code TEXT,
    occurred_at TEXT NOT NULL,
    PRIMARY KEY(event_id, attempt),
    FOREIGN KEY(event_id) REFERENCES retrieval_update_outbox(event_id) ON DELETE RESTRICT
);
CREATE INDEX IF NOT EXISTS idx_retrieval_outbox_attempt_audits_batch
    ON retrieval_outbox_attempt_audits(batch_id, batch_ordinal);

CREATE TABLE IF NOT EXISTS retrieval_outbox_manual_actions (
    action_id TEXT PRIMARY KEY,
    event_id TEXT NOT NULL,
    action TEXT NOT NULL CHECK(action IN ('manual_requeue')),
    actor TEXT NOT NULL,
    reason TEXT NOT NULL,
    previous_status TEXT NOT NULL CHECK(previous_status IN ('terminal_failed')),
    previous_reason_code TEXT,
    occurred_at TEXT NOT NULL,
    FOREIGN KEY(event_id) REFERENCES retrieval_update_outbox(event_id) ON DELETE RESTRICT
);
CREATE INDEX IF NOT EXISTS idx_retrieval_outbox_manual_actions_event
    ON retrieval_outbox_manual_actions(event_id, occurred_at DESC, action_id DESC);
"""
