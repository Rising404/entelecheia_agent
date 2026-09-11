"""Session 对共享文件处理作业的独立请求与交付账本。"""

from __future__ import annotations

from collections.abc import Sequence
import sqlite3

from .contracts import (
    DocumentIngestJobIdCollision,
    DocumentIngestJobStatus,
    DocumentIngestJobTransitionError,
    FilePreparationDeliveryStatus,
    FilePreparationRequest,
    _require_non_negative_integer,
    _require_positive_integer,
    _require_text,
)
from .repository import _immediate_transaction, _normalize_timestamp, _rows_by_name


_COLUMN_NAMES = (
    "request_id",
    "job_id",
    "session_id",
    "with_summary",
    "source_mtime_ns",
    "delivery_status",
    "reason_code",
    "created_at",
    "updated_at",
)
_COLUMNS = ", ".join(_COLUMN_NAMES)


class SqliteFilePreparationRequestStore:
    """只记录请求与交付状态，不将某个 Session 的授权变成共享作业权限。"""

    def create_in_transaction(
        self,
        conn: sqlite3.Connection,
        *,
        request_id: str,
        job_id: str,
        session_id: str,
        with_summary: bool,
        source_mtime_ns: int,
        now: str,
    ) -> FilePreparationRequest:
        if not conn.in_transaction:
            raise RuntimeError("create_in_transaction requires an active transaction")
        for name, value in (("request_id", request_id), ("job_id", job_id), ("session_id", session_id)):
            _require_text(name, value)
        if type(with_summary) is not bool:
            raise TypeError("with_summary must be a bool")
        _require_non_negative_integer("source_mtime_ns", source_mtime_ns)
        timestamp = _normalize_timestamp("now", now)
        existing = self.get(conn, request_id)
        if existing is not None:
            if (
                existing.job_id != job_id or existing.session_id != session_id
                or existing.with_summary is not with_summary
                or existing.source_mtime_ns != source_mtime_ns
            ):
                raise DocumentIngestJobIdCollision(
                    "file preparation request_id was reused for a different payload"
                )
            return existing
        conn.execute(
            "INSERT INTO document_ingest_requests "
            f"({_COLUMNS}) VALUES (?, ?, ?, ?, ?, 'pending', NULL, ?, ?)",
            (request_id, job_id, session_id, int(with_summary), source_mtime_ns, timestamp, timestamp),
        )
        return self._require(conn, request_id)

    def get(self, conn: sqlite3.Connection, request_id: str) -> FilePreparationRequest | None:
        _require_text("request_id", request_id)
        _rows_by_name(conn)
        row = conn.execute(
            f"SELECT {_COLUMNS} FROM document_ingest_requests WHERE request_id=?",
            (request_id,),
        ).fetchone()
        return _request_from_row(row) if row is not None else None

    def list_for_job(
        self, conn: sqlite3.Connection, job_id: str, *,
        delivery_status: FilePreparationDeliveryStatus | None = None, limit: int = 256,
    ) -> tuple[FilePreparationRequest, ...]:
        _require_text("job_id", job_id)
        if delivery_status is None:
            return self._list(conn, "job_id=?", (job_id,), limit=limit)
        return self._list(
            conn, "job_id=? AND delivery_status=?",
            (job_id, FilePreparationDeliveryStatus(delivery_status).value), limit=limit,
        )

    def list_for_session(
        self,
        conn: sqlite3.Connection,
        session_id: str,
        *,
        delivery_statuses: Sequence[FilePreparationDeliveryStatus] | None = None,
        pending_job_statuses: Sequence[DocumentIngestJobStatus] | None = None,
        limit: int = 100,
    ) -> tuple[FilePreparationRequest, ...]:
        _require_text("session_id", session_id)
        if delivery_statuses is None and pending_job_statuses is None:
            return self._list(
                conn,
                "session_id=?",
                (session_id,),
                limit=limit,
                newest_first=True,
            )

        _require_positive_integer("limit", limit)
        delivery_values = tuple(dict.fromkeys(
            FilePreparationDeliveryStatus(status).value
            for status in (delivery_statuses or ())
        ))
        job_values = tuple(dict.fromkeys(
            DocumentIngestJobStatus(status).value
            for status in (pending_job_statuses or ())
        ))
        matches: list[str] = []
        values: list[object] = [session_id]
        if delivery_values:
            matches.append(
                "requests.delivery_status IN "
                f"({', '.join('?' for _ in delivery_values)})"
            )
            values.extend(delivery_values)
        if job_values:
            matches.append(
                "(requests.delivery_status=? AND jobs.status IN "
                f"({', '.join('?' for _ in job_values)}))"
            )
            values.append(FilePreparationDeliveryStatus.PENDING.value)
            values.extend(job_values)
        if not matches:
            return ()

        _rows_by_name(conn)
        selected_columns = ", ".join(
            f"requests.{column} AS {column}" for column in _COLUMN_NAMES
        )
        rows = conn.execute(
            f"SELECT {selected_columns} FROM document_ingest_requests AS requests "
            "JOIN document_ingest_jobs AS jobs ON jobs.job_id=requests.job_id "
            f"WHERE requests.session_id=? AND ({' OR '.join(matches)}) "
            "ORDER BY requests.created_at DESC, requests.request_id DESC LIMIT ?",
            (*values, limit),
        ).fetchall()
        return tuple(_request_from_row(row) for row in rows)

    def list_pending(
        self, conn: sqlite3.Connection, *, job_id: str | None = None, limit: int = 100,
    ) -> tuple[FilePreparationRequest, ...]:
        # 先筛选可交付结果再限流；尚在解析的大量请求不能饿死已完成作业的订阅者。
        predicate = (
            "delivery_status='pending' AND job_id IN ("
            "SELECT job_id FROM document_ingest_jobs WHERE status='applied')"
        )
        values: tuple[object, ...] = ()
        if job_id is not None:
            _require_text("job_id", job_id)
            predicate += " AND job_id=?"
            values = (job_id,)
        return self._list(conn, predicate, values, limit=limit)

    def mark_mounted(
        self, conn: sqlite3.Connection, request_id: str, *, now: str,
    ) -> FilePreparationRequest:
        """外层完成真实、可重放的 Session 挂载之后记录交付。"""

        return self._set_delivery(
            conn, request_id, status=FilePreparationDeliveryStatus.MOUNTED,
            reason_code=None, now=now,
        )

    def mark_blocked(
        self, conn: sqlite3.Connection, request_id: str, *, reason_code: str, now: str,
    ) -> FilePreparationRequest:
        _require_text("reason_code", reason_code)
        return self._set_delivery(
            conn, request_id, status=FilePreparationDeliveryStatus.BLOCKED,
            reason_code=reason_code, now=now,
        )

    def retry_blocked(
        self, conn: sqlite3.Connection, request_id: str, *, now: str,
    ) -> FilePreparationRequest:
        """仅供外层重新核实授权与来源后的显式重试，不自动恢复被拒绝的请求。"""

        return self._set_delivery(
            conn, request_id, status=FilePreparationDeliveryStatus.PENDING,
            reason_code=None, now=now,
        )

    def retry_blocked_in_transaction(
        self, conn: sqlite3.Connection, request_id: str, *, now: str,
    ) -> FilePreparationRequest:
        """与共享 job、Outbox 的显式重试复用调用方事务。"""

        return self._set_delivery_in_transaction(
            conn, request_id, status=FilePreparationDeliveryStatus.PENDING,
            reason_code=None, now=now,
        )

    def _set_delivery(
        self,
        conn: sqlite3.Connection,
        request_id: str,
        *,
        status: FilePreparationDeliveryStatus,
        reason_code: str | None,
        now: str,
    ) -> FilePreparationRequest:
        with _immediate_transaction(conn):
            return self._set_delivery_in_transaction(
                conn, request_id, status=status, reason_code=reason_code, now=now,
            )

    def _set_delivery_in_transaction(
        self, conn: sqlite3.Connection, request_id: str, *,
        status: FilePreparationDeliveryStatus, reason_code: str | None, now: str,
    ) -> FilePreparationRequest:
        if not conn.in_transaction:
            raise RuntimeError("delivery transition requires an active transaction")
        timestamp = _normalize_timestamp("now", now)
        current = self._require(conn, request_id)
        if current.delivery_status is status and current.reason_code == reason_code:
            return current
        allowed = (
            current.delivery_status is FilePreparationDeliveryStatus.PENDING
            or (current.delivery_status is FilePreparationDeliveryStatus.BLOCKED
                and status is FilePreparationDeliveryStatus.PENDING)
        )
        if not allowed:
            raise DocumentIngestJobTransitionError("file preparation delivery is already final")
        if status is FilePreparationDeliveryStatus.MOUNTED:
            job = conn.execute(
                "SELECT status FROM document_ingest_jobs WHERE job_id=?", (current.job_id,),
            ).fetchone()
            if job is None or str(job["status"]) != "applied":
                raise DocumentIngestJobTransitionError(
                    "mounted delivery requires an applied shared job"
                )
        conn.execute(
            "UPDATE document_ingest_requests SET delivery_status=?, reason_code=?, updated_at=? "
            "WHERE request_id=?",
            (status.value, reason_code, timestamp, request_id),
        )
        return self._require(conn, request_id)

    def _require(self, conn: sqlite3.Connection, request_id: str) -> FilePreparationRequest:
        request = self.get(conn, request_id)
        if request is None:
            raise KeyError(f"file preparation request not found: {request_id}")
        return request

    @staticmethod
    def _list(
        conn: sqlite3.Connection, predicate: str, values: tuple[object, ...], *,
        limit: int, newest_first: bool = False,
    ) -> tuple[FilePreparationRequest, ...]:
        _require_positive_integer("limit", limit)
        _rows_by_name(conn)
        order = "created_at DESC, request_id DESC" if newest_first else "created_at, request_id"
        rows = conn.execute(
            f"SELECT {_COLUMNS} FROM document_ingest_requests WHERE {predicate} "
            f"ORDER BY {order} LIMIT ?", (*values, limit),
        ).fetchall()
        return tuple(_request_from_row(row) for row in rows)


def _request_from_row(row: sqlite3.Row) -> FilePreparationRequest:
    return FilePreparationRequest(
        request_id=str(row["request_id"]), job_id=str(row["job_id"]),
        session_id=str(row["session_id"]), with_summary=bool(row["with_summary"]),
        source_mtime_ns=int(row["source_mtime_ns"]),
        delivery_status=FilePreparationDeliveryStatus(str(row["delivery_status"])),
        reason_code=str(row["reason_code"]) if row["reason_code"] is not None else None,
        created_at=str(row["created_at"]), updated_at=str(row["updated_at"]),
    )


__all__ = ["SqliteFilePreparationRequestStore"]
