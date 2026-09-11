"""用于文档收录的持久化权威源本地操作。

源文档及其抽取出的块仍归 ``docstore`` 所有。本模块仅持久化操作信封、恢复检查点、
稳定的权威引用，以及仅含指针的检索 Outbox 事件 ID。因此 API 可以安全轮询，工作器也
可以在进程崩溃后安全恢复，而不会把内存中的 future 当作完成证明。

每个独立变更都拥有一个短 ``BEGIN IMMEDIATE`` 事务。已经在组合权威文档写入和
Outbox 追加的工作器可以改用明确的 ``*_in_transaction`` 方法；这些方法绝不会提交或
回滚调用方的事务。
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import sqlite3
from typing import Iterator, Sequence
import uuid

from ...documents.indexing import DocumentIndexAuditError, DocumentIndexAuditPort
from .contracts import (
    DocumentIngestCoverageProof,
    DocumentIngestEnqueueResult,
    DocumentIngestJob,
    DocumentIngestJobIdCollision,
    DocumentIngestJobLeaseError,
    DocumentIngestJobRequest,
    DocumentIngestJobStage,
    DocumentIngestJobStatus,
    DocumentIngestJobTransitionError,
    _require_text,
)


_JOB_COLUMNS = (
    "job_id, payload_fingerprint, file_id, file_version_id, canonical_path, "
    "source_sha256, source_size, processor_fingerprint, "
    "chunker_fingerprint, chunk_contract_version, "
    "target_generation_id, target_generation_fingerprint, "
    "stage, status, attempts, next_retry_at, lease_owner, lease_token, lease_until, "
    "completion_lease_token, "
    "document_id, document_version_id, retrieval_data_version, "
    "coverage_data_version_fingerprint, coverage_binding_digest, "
    "coverage_expected_bindings, coverage_covered_bindings, coverage_mapped_events, "
    "coverage_applied_events, coverage_checked_at, reason_code, "
    "created_at, updated_at, completed_at"
)
_STAGE_ORDER = {
    DocumentIngestJobStage.PARSING: 0,
    DocumentIngestJobStage.CHUNKED: 1,
    DocumentIngestJobStage.INDEXING: 2,
    DocumentIngestJobStage.COVERAGE_READY: 3,
    DocumentIngestJobStage.ACTIVE: 4,
}


class SqliteDocumentIngestJobStore:
    """为崩溃后可恢复的文档收录工作提供 SQLite 持久化。"""

    def __init__(
        self,
        *,
        index_audit_port: DocumentIndexAuditPort | None = None,
    ) -> None:
        if index_audit_port is not None and not isinstance(
            index_audit_port, DocumentIndexAuditPort
        ):
            raise TypeError("index_audit_port must satisfy DocumentIndexAuditPort")
        self._index_audit_port = index_audit_port

    def enqueue(
        self,
        conn: sqlite3.Connection,
        *,
        request: DocumentIngestJobRequest,
        now: str,
    ) -> DocumentIngestEnqueueResult:
        """创建一个操作，或返回其精确幂等重放结果。"""

        with _immediate_transaction(conn):
            return self.enqueue_in_transaction(conn, request=request, now=now)

    def enqueue_in_transaction(
        self,
        conn: sqlite3.Connection,
        *,
        request: DocumentIngestJobRequest,
        now: str,
    ) -> DocumentIngestEnqueueResult:
        """将共享作业与 Session request 放在调用方的同一短事务中接受。"""

        if not conn.in_transaction:
            raise RuntimeError("enqueue_in_transaction requires an active transaction")
        timestamp = _normalize_timestamp("now", now)
        _rows_by_name(conn)
        existing = self._get_row(conn, request.job_id)
        if existing is None:
            existing = conn.execute(
                f"SELECT {_JOB_COLUMNS} FROM document_ingest_jobs "
                "WHERE file_version_id=? AND processor_fingerprint=? "
                "AND chunker_fingerprint=? AND chunk_contract_version=? "
                "AND target_generation_id=? AND target_generation_fingerprint=?",
                (
                    request.file_version_id, request.processor_fingerprint,
                    request.chunker_fingerprint, request.chunk_contract_version,
                    request.target_generation_id, request.target_generation_fingerprint,
                ),
            ).fetchone()
        if existing is not None:
            if not _row_matches_request(existing, request):
                raise DocumentIngestJobIdCollision(
                    "document ingest identity was reused for a different payload"
                )
            return DocumentIngestEnqueueResult(job=_job_from_row(existing), replayed=True)
        source = conn.execute(
            "SELECT 1 FROM file_versions WHERE id=? AND file_id=? "
            "AND content_sha256=? AND size_bytes=?",
            (request.file_version_id, request.file_id, request.source_sha256, request.source_size),
        ).fetchone()
        if source is None:
            raise DocumentIngestJobTransitionError(
                "job source must match its exact registered FileVersion"
            )
        conn.execute(
            "INSERT INTO document_ingest_jobs ("
            "job_id, payload_fingerprint, file_id, file_version_id, canonical_path, "
            "source_sha256, source_size, processor_fingerprint, "
            "chunker_fingerprint, chunk_contract_version, "
            "target_generation_id, target_generation_fingerprint, "
            "stage, status, attempts, next_retry_at, created_at, updated_at"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?, ?)",
            (
                request.job_id, request.payload_fingerprint,
                request.file_id, request.file_version_id, request.canonical_path,
                request.source_sha256, request.source_size, request.processor_fingerprint,
                request.chunker_fingerprint, request.chunk_contract_version,
                request.target_generation_id, request.target_generation_fingerprint,
                DocumentIngestJobStage.PARSING.value, DocumentIngestJobStatus.PENDING.value,
                timestamp, timestamp, timestamp,
            ),
        )
        return DocumentIngestEnqueueResult(
            job=_job_from_row(self._require_row(conn, request.job_id)), replayed=False,
        )

    def get(
        self,
        conn: sqlite3.Connection,
        job_id: str,
    ) -> DocumentIngestJob | None:
        _require_text("job_id", job_id)
        _rows_by_name(conn)
        row = self._get_row(conn, job_id)
        return _job_from_row(row) if row is not None else None

    def list(
        self,
        conn: sqlite3.Connection,
        *,
        statuses: Sequence[DocumentIngestJobStatus] | None = None,
        limit: int = 50,
    ) -> tuple[DocumentIngestJob, ...]:
        if limit <= 0:
            raise ValueError("limit must be greater than zero")
        normalized_statuses = (
            tuple(DocumentIngestJobStatus(status) for status in statuses)
            if statuses is not None
            else ()
        )
        if statuses is not None and not normalized_statuses:
            return ()

        predicates: list[str] = []
        parameters: list[object] = []
        if normalized_statuses:
            placeholders = ",".join("?" for _ in normalized_statuses)
            predicates.append(f"status IN ({placeholders})")
            parameters.extend(status.value for status in normalized_statuses)
        where = f" WHERE {' AND '.join(predicates)}" if predicates else ""
        _rows_by_name(conn)
        rows = conn.execute(
            f"SELECT {_JOB_COLUMNS} FROM document_ingest_jobs{where} "
            "ORDER BY created_at DESC, job_id DESC LIMIT ?",
            (*parameters, limit),
        ).fetchall()
        return tuple(_job_from_row(row) for row in rows)

    def claim_due(
        self,
        conn: sqlite3.Connection,
        *,
        worker_id: str,
        now: str,
        lease_seconds: int,
        limit: int = 16,
        target_generation_id: str | None = None,
        target_generation_fingerprint: str | None = None,
    ) -> tuple[DocumentIngestJob, ...]:
        """原子租用到期工作，包括崩溃工作器遗弃的工作。"""

        _require_text("worker_id", worker_id)
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be greater than zero")
        if limit <= 0:
            raise ValueError("limit must be greater than zero")
        timestamp = _normalize_timestamp("now", now)
        lease_until = _add_seconds(timestamp, lease_seconds)
        _rows_by_name(conn)
        generation_predicates: list[str] = []
        generation_values: list[object] = []
        for name, value in (
            ("target_generation_id", target_generation_id),
            ("target_generation_fingerprint", target_generation_fingerprint),
        ):
            if value is not None:
                _require_text(name, value)
                generation_predicates.append(f"{name}=?")
                generation_values.append(value)
        generation_filter = (
            " AND " + " AND ".join(generation_predicates) if generation_predicates else ""
        )
        with _immediate_transaction(conn):
            rows = conn.execute(
                f"SELECT {_JOB_COLUMNS} FROM document_ingest_jobs WHERE "
                "((status IN (?, ?) AND next_retry_at <= ?) OR "
                "(status=? AND lease_until <= ?)) "
                + generation_filter + " ORDER BY created_at, job_id LIMIT ?",
                (
                    DocumentIngestJobStatus.PENDING.value,
                    DocumentIngestJobStatus.RETRYABLE_FAILED.value,
                    timestamp,
                    DocumentIngestJobStatus.PROCESSING.value,
                    timestamp,
                    *generation_values,
                    limit,
                ),
            ).fetchall()
            claimed: list[DocumentIngestJob] = []
            for row in rows:
                lease_token = uuid.uuid4().hex
                updated = conn.execute(
                    "UPDATE document_ingest_jobs SET status=?, attempts=attempts+1, "
                    "next_retry_at=NULL, lease_owner=?, lease_token=?, lease_until=?, "
                    "reason_code=NULL, updated_at=? WHERE job_id=? AND "
                    "((status IN (?, ?) AND next_retry_at <= ?) OR "
                    "(status=? AND lease_until <= ?))",
                    (
                        DocumentIngestJobStatus.PROCESSING.value,
                        worker_id,
                        lease_token,
                        lease_until,
                        timestamp,
                        row["job_id"],
                        DocumentIngestJobStatus.PENDING.value,
                        DocumentIngestJobStatus.RETRYABLE_FAILED.value,
                        timestamp,
                        DocumentIngestJobStatus.PROCESSING.value,
                        timestamp,
                    ),
                ).rowcount
                if updated:
                    claimed.append(_job_from_row(self._require_row(conn, str(row["job_id"]))))
            return tuple(claimed)

    def renew_lease(
        self,
        conn: sqlite3.Connection,
        *,
        job_id: str,
        worker_id: str,
        lease_token: str,
        now: str,
        lease_seconds: int,
    ) -> DocumentIngestJob:
        _require_text("job_id", job_id)
        _require_text("worker_id", worker_id)
        _require_text("lease_token", lease_token)
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be greater than zero")
        timestamp = _normalize_timestamp("now", now)
        lease_until = _add_seconds(timestamp, lease_seconds)
        _rows_by_name(conn)
        with _immediate_transaction(conn):
            self._require_live_lease(
                conn,
                job_id=job_id,
                worker_id=worker_id,
                lease_token=lease_token,
                now=timestamp,
            )
            conn.execute(
                "UPDATE document_ingest_jobs SET lease_until=?, updated_at=? "
                "WHERE job_id=?",
                (lease_until, timestamp, job_id),
            )
            return _job_from_row(self._require_row(conn, job_id))

    def checkpoint_stage(
        self,
        conn: sqlite3.Connection,
        *,
        job_id: str,
        worker_id: str,
        lease_token: str,
        stage: DocumentIngestJobStage,
        now: str,
        document_id: str | None = None,
        document_version_id: str | None = None,
        retrieval_data_version: str | None = None,
        outbox_event_ids: Sequence[str] = (),
        coverage_proof: DocumentIngestCoverageProof | None = None,
    ) -> DocumentIngestJob:
        """在自身的短事务中推进一个检查点。"""

        _rows_by_name(conn)
        with _immediate_transaction(conn):
            return self.checkpoint_stage_in_transaction(
                conn,
                job_id=job_id,
                worker_id=worker_id,
                lease_token=lease_token,
                stage=stage,
                now=now,
                document_id=document_id,
                document_version_id=document_version_id,
                retrieval_data_version=retrieval_data_version,
                outbox_event_ids=outbox_event_ids,
                coverage_proof=coverage_proof,
            )

    def checkpoint_stage_in_transaction(
        self,
        conn: sqlite3.Connection,
        *,
        job_id: str,
        worker_id: str,
        lease_token: str,
        stage: DocumentIngestJobStage,
        now: str,
        document_id: str | None = None,
        document_version_id: str | None = None,
        retrieval_data_version: str | None = None,
        outbox_event_ids: Sequence[str] = (),
        coverage_proof: DocumentIngestCoverageProof | None = None,
    ) -> DocumentIngestJob:
        """在调用方的权威事务内推进一个检查点。

        这是 ``document/version/chunks + Outbox + operation checkpoint`` 的组合点。
        该方法刻意不打开或结束事务，从而保证崩溃时这些写入要么全部可见，要么全部
        不可见。
        """

        if not conn.in_transaction:
            raise RuntimeError("checkpoint_stage_in_transaction requires an active transaction")
        _require_text("job_id", job_id)
        _require_text("worker_id", worker_id)
        _require_text("lease_token", lease_token)
        target_stage = DocumentIngestJobStage(stage)
        if target_stage is DocumentIngestJobStage.ACTIVE:
            raise DocumentIngestJobTransitionError(
                "active is a terminal checkpoint; use mark_applied"
            )
        timestamp = _normalize_timestamp("now", now)
        for name, value in (
            ("document_id", document_id),
            ("document_version_id", document_version_id),
            ("retrieval_data_version", retrieval_data_version),
        ):
            if value is not None:
                _require_text(name, value)
        event_ids = tuple(outbox_event_ids)
        if len(set(event_ids)) != len(event_ids):
            raise ValueError("outbox_event_ids must be unique")
        for event_id in event_ids:
            _require_text("event_id", event_id)

        _rows_by_name(conn)
        current = self._require_live_lease(
            conn,
            job_id=job_id,
            worker_id=worker_id,
            lease_token=lease_token,
            now=timestamp,
        )
        current_stage = DocumentIngestJobStage(str(current["stage"]))
        stage_delta = _STAGE_ORDER[target_stage] - _STAGE_ORDER[current_stage]
        if stage_delta < 0:
            raise DocumentIngestJobTransitionError(
                f"document ingest stage cannot regress from {current_stage.value} "
                f"to {target_stage.value}"
            )
        if stage_delta > 1:
            raise DocumentIngestJobTransitionError(
                f"document ingest stage cannot skip from {current_stage.value} "
                f"to {target_stage.value}"
            )
        if coverage_proof is not None and not isinstance(
            coverage_proof, DocumentIngestCoverageProof
        ):
            raise TypeError("coverage_proof must be a DocumentIngestCoverageProof")
        if target_stage is DocumentIngestJobStage.COVERAGE_READY:
            if coverage_proof is None:
                raise DocumentIngestJobTransitionError(
                    "coverage_ready requires an exact durable coverage proof"
                )
            if current_stage is DocumentIngestJobStage.COVERAGE_READY and event_ids:
                raise DocumentIngestJobTransitionError(
                    "coverage_ready evidence is frozen and cannot accept more events"
                )
        elif coverage_proof is not None:
            raise DocumentIngestJobTransitionError(
                "coverage_proof is only valid for coverage_ready"
            )

        merged_document_id = _merge_reference(
            "document_id", current["document_id"], document_id
        )
        merged_version_id = _merge_reference(
            "document_version_id", current["document_version_id"], document_version_id
        )
        merged_data_version = _merge_reference(
            "retrieval_data_version",
            current["retrieval_data_version"],
            retrieval_data_version,
        )
        if target_stage is DocumentIngestJobStage.PARSING and any(
            value is not None
            for value in (document_id, document_version_id, retrieval_data_version)
        ):
            raise DocumentIngestJobTransitionError(
                "authority references may only be frozen at the durable chunked stage"
            )
        if _STAGE_ORDER[target_stage] >= _STAGE_ORDER[DocumentIngestJobStage.CHUNKED]:
            if not (merged_document_id and merged_version_id and merged_data_version):
                raise DocumentIngestJobTransitionError(
                    "chunked and later stages require all document and retrieval references"
                )
            if merged_data_version != str(current["target_generation_id"]):
                raise DocumentIngestJobTransitionError(
                    "checkpoint retrieval_data_version does not match the job target generation"
                )
            self._require_current_document_target(
                conn,
                job=current,
                document_id=merged_document_id,
                document_version_id=merged_version_id,
            )

        if event_ids:
            try:
                self._require_index_audit_port().validate_job_events_in_transaction(
                    conn,
                    event_ids=event_ids,
                    document_id=str(merged_document_id),
                    document_version_id=str(merged_version_id),
                    data_version_id=str(merged_data_version),
                )
            except DocumentIndexAuditError as exc:
                raise DocumentIngestJobTransitionError(str(exc)) from exc
            for event_id in event_ids:
                self._record_outbox_event_in_transaction(
                    conn,
                    job_id=job_id,
                    event_id=event_id,
                    recorded_at=timestamp,
                )
        coverage_values: tuple[object, ...]
        if target_stage is DocumentIngestJobStage.COVERAGE_READY:
            assert coverage_proof is not None
            if coverage_proof.retrieval_data_version_fingerprint != str(
                current["target_generation_fingerprint"]
            ):
                raise DocumentIngestJobTransitionError(
                    "coverage proof fingerprint does not match the job target generation"
                )
            if coverage_proof.retrieval_data_version_id != merged_data_version:
                raise DocumentIngestJobTransitionError(
                    "coverage proof data version does not match retrieval_data_version"
                )
            expected_count, expected_digest, mapped_events, applied_events = (
                self._require_exact_coverage_proof(
                    conn,
                    document_id=str(merged_document_id),
                    document_version_id=str(merged_version_id),
                    job_id=job_id,
                    proof=coverage_proof,
                )
            )
            existing_fingerprint = _optional_text(
                current["coverage_data_version_fingerprint"]
            )
            if (
                existing_fingerprint is not None
                and existing_fingerprint
                != coverage_proof.retrieval_data_version_fingerprint
            ):
                raise DocumentIngestJobTransitionError(
                    "coverage proof is immutable after coverage_ready"
                )
            if current_stage is DocumentIngestJobStage.COVERAGE_READY:
                if (
                    _optional_text(current["coverage_binding_digest"])
                    != expected_digest
                    or current["coverage_expected_bindings"] is None
                    or int(current["coverage_expected_bindings"]) != expected_count
                    or current["coverage_covered_bindings"] is None
                    or int(current["coverage_covered_bindings"])
                    != coverage_proof.covered_binding_count
                    or current["coverage_mapped_events"] is None
                    or int(current["coverage_mapped_events"]) != mapped_events
                    or current["coverage_applied_events"] is None
                    or int(current["coverage_applied_events"]) != applied_events
                ):
                    raise DocumentIngestJobTransitionError(
                        "coverage_ready replay does not match its durable proof"
                    )
                return _job_from_row(current)
            coverage_values = (
                coverage_proof.retrieval_data_version_fingerprint,
                expected_digest,
                expected_count,
                coverage_proof.covered_binding_count,
                mapped_events,
                applied_events,
                timestamp,
            )
        else:
            coverage_values = (
                current["coverage_data_version_fingerprint"],
                current["coverage_binding_digest"],
                current["coverage_expected_bindings"],
                current["coverage_covered_bindings"],
                current["coverage_mapped_events"],
                current["coverage_applied_events"],
                current["coverage_checked_at"],
            )

        conn.execute(
            "UPDATE document_ingest_jobs SET stage=?, document_id=?, "
            "document_version_id=?, retrieval_data_version=?, "
            "coverage_data_version_fingerprint=?, coverage_binding_digest=?, "
            "coverage_expected_bindings=?, coverage_covered_bindings=?, "
            "coverage_mapped_events=?, coverage_applied_events=?, coverage_checked_at=?, "
            "updated_at=? WHERE job_id=?",
            (
                target_stage.value,
                merged_document_id,
                merged_version_id,
                merged_data_version,
                *coverage_values,
                timestamp,
                job_id,
            ),
        )
        return _job_from_row(self._require_row(conn, job_id))

    def mark_applied(
        self,
        conn: sqlite3.Connection,
        *,
        job_id: str,
        worker_id: str,
        lease_token: str,
        now: str,
    ) -> DocumentIngestJob:
        _rows_by_name(conn)
        with _immediate_transaction(conn):
            return self.mark_applied_in_transaction(
                conn,
                job_id=job_id,
                worker_id=worker_id,
                lease_token=lease_token,
                now=now,
            )

    def mark_applied_in_transaction(
        self,
        conn: sqlite3.Connection,
        *,
        job_id: str,
        worker_id: str,
        lease_token: str,
        now: str,
    ) -> DocumentIngestJob:
        """在调用方事务内激活一个覆盖已就绪的操作。"""

        if not conn.in_transaction:
            raise RuntimeError("mark_applied_in_transaction requires an active transaction")
        _require_text("job_id", job_id)
        _require_text("worker_id", worker_id)
        _require_text("lease_token", lease_token)
        timestamp = _normalize_timestamp("now", now)
        _rows_by_name(conn)
        row = self._require_row(conn, job_id)
        if DocumentIngestJobStatus(str(row["status"])) is DocumentIngestJobStatus.APPLIED:
            if str(row["completion_lease_token"] or "") != lease_token:
                raise DocumentIngestJobLeaseError(
                    "applied document ingest replay requires its completion lease token"
                )
            return _job_from_row(row)
        row = self._require_live_lease(
            conn,
            job_id=job_id,
            worker_id=worker_id,
            lease_token=lease_token,
            now=timestamp,
        )
        if DocumentIngestJobStage(str(row["stage"])) is not DocumentIngestJobStage.COVERAGE_READY:
            raise DocumentIngestJobTransitionError(
                "only a coverage_ready document ingest job may become active"
            )
        if not (
            row["document_id"]
            and row["document_version_id"]
            and row["retrieval_data_version"]
        ):
            raise DocumentIngestJobTransitionError(
                "active document ingest requires all authority references"
            )
        self._require_current_document_target(
            conn,
            job=row,
            document_id=str(row["document_id"]),
            document_version_id=str(row["document_version_id"]),
        )
        self._require_persisted_coverage_still_current(conn, job=row)
        conn.execute(
            "UPDATE document_ingest_jobs SET stage=?, status=?, next_retry_at=NULL, "
            "lease_owner=NULL, lease_token=NULL, lease_until=NULL, reason_code=NULL, "
            "completion_lease_token=?, updated_at=?, completed_at=? WHERE job_id=?",
            (
                DocumentIngestJobStage.ACTIVE.value,
                DocumentIngestJobStatus.APPLIED.value,
                lease_token,
                timestamp,
                timestamp,
                job_id,
            ),
        )
        return _job_from_row(self._require_row(conn, job_id))

    def mark_retryable_failure(
        self,
        conn: sqlite3.Connection,
        *,
        job_id: str,
        worker_id: str,
        lease_token: str,
        now: str,
        retry_after_seconds: int,
        reason_code: str,
    ) -> DocumentIngestJob:
        if retry_after_seconds <= 0:
            raise ValueError("retry_after_seconds must be greater than zero")
        return self._mark_failure(
            conn,
            job_id=job_id,
            worker_id=worker_id,
            lease_token=lease_token,
            now=now,
            reason_code=reason_code,
            status=DocumentIngestJobStatus.RETRYABLE_FAILED,
            retry_after_seconds=retry_after_seconds,
        )

    def mark_terminal_failure(
        self,
        conn: sqlite3.Connection,
        *,
        job_id: str,
        worker_id: str,
        lease_token: str,
        now: str,
        reason_code: str,
    ) -> DocumentIngestJob:
        return self._mark_failure(
            conn,
            job_id=job_id,
            worker_id=worker_id,
            lease_token=lease_token,
            now=now,
            reason_code=reason_code,
            status=DocumentIngestJobStatus.TERMINAL_FAILED,
            retry_after_seconds=None,
        )

    def retry_terminal_failure(
        self,
        conn: sqlite3.Connection,
        *,
        job_id: str,
        now: str,
    ) -> DocumentIngestJob:
        """显式重新入队死信工作，同时保留其检查点。"""

        _require_text("job_id", job_id)
        timestamp = _normalize_timestamp("now", now)
        _rows_by_name(conn)
        with _immediate_transaction(conn):
            return self.retry_terminal_failure_in_transaction(
                conn,
                job_id=job_id,
                now=timestamp,
            )

    def retry_terminal_failure_in_transaction(
        self,
        conn: sqlite3.Connection,
        *,
        job_id: str,
        now: str,
    ) -> DocumentIngestJob:
        """在调用方拥有的权威事务内重新入队终态工作。"""

        _require_text("job_id", job_id)
        timestamp = _normalize_timestamp("now", now)
        _rows_by_name(conn)
        row = self._require_row(conn, job_id)
        if (
            DocumentIngestJobStatus(str(row["status"]))
            is not DocumentIngestJobStatus.TERMINAL_FAILED
        ):
            raise DocumentIngestJobTransitionError(
                "only a terminal_failed document ingest job may be retried"
            )
        conn.execute(
            "UPDATE document_ingest_jobs SET status=?, next_retry_at=?, "
            "lease_owner=NULL, lease_token=NULL, lease_until=NULL, updated_at=?, "
            "completed_at=NULL WHERE job_id=?",
            (
                DocumentIngestJobStatus.PENDING.value,
                timestamp,
                timestamp,
                job_id,
            ),
        )
        return _job_from_row(self._require_row(conn, job_id))

    def list_outbox_event_ids(
        self,
        conn: sqlite3.Connection,
        job_id: str,
    ) -> tuple[str, ...]:
        _require_text("job_id", job_id)
        rows = conn.execute(
            "SELECT event_id FROM document_ingest_job_events WHERE job_id=? "
            "ORDER BY recorded_at, event_id",
            (job_id,),
        ).fetchall()
        return tuple(str(row[0]) for row in rows)

    def _mark_failure(
        self,
        conn: sqlite3.Connection,
        *,
        job_id: str,
        worker_id: str,
        lease_token: str,
        now: str,
        reason_code: str,
        status: DocumentIngestJobStatus,
        retry_after_seconds: int | None,
    ) -> DocumentIngestJob:
        _require_text("job_id", job_id)
        _require_text("worker_id", worker_id)
        _require_text("lease_token", lease_token)
        _require_text("reason_code", reason_code)
        timestamp = _normalize_timestamp("now", now)
        next_retry_at = (
            _add_seconds(timestamp, retry_after_seconds)
            if retry_after_seconds is not None
            else None
        )
        completed_at = (
            timestamp if status is DocumentIngestJobStatus.TERMINAL_FAILED else None
        )
        _rows_by_name(conn)
        with _immediate_transaction(conn):
            self._require_live_lease(
                conn,
                job_id=job_id,
                worker_id=worker_id,
                lease_token=lease_token,
                now=timestamp,
            )
            conn.execute(
                "UPDATE document_ingest_jobs SET status=?, next_retry_at=?, "
                "lease_owner=NULL, lease_token=NULL, lease_until=NULL, reason_code=?, "
                "updated_at=?, completed_at=? WHERE job_id=?",
                (
                    status.value,
                    next_retry_at,
                    reason_code,
                    timestamp,
                    completed_at,
                    job_id,
                ),
            )
            return _job_from_row(self._require_row(conn, job_id))

    def _require_exact_coverage_proof(
        self,
        conn: sqlite3.Connection,
        *,
        document_id: str,
        document_version_id: str,
        job_id: str,
        proof: DocumentIngestCoverageProof,
    ) -> tuple[int, str, int, int]:
        event_ids = self.list_outbox_event_ids(conn, job_id)
        try:
            snapshot = self._require_index_audit_port().inspect_job_coverage_in_transaction(
                conn,
                event_ids=event_ids,
                document_id=document_id,
                document_version_id=document_version_id,
                data_version_id=proof.retrieval_data_version_id,
            )
        except DocumentIndexAuditError as exc:
            raise DocumentIngestJobTransitionError(str(exc)) from exc
        if (
            proof.covered_binding_count != snapshot.binding_count
            or proof.covered_binding_digest != snapshot.binding_digest
        ):
            raise DocumentIngestJobTransitionError(
                "coverage proof does not exactly match current document authority bindings"
            )
        if snapshot.applied_event_count != snapshot.mapped_event_count:
            raise DocumentIngestJobTransitionError(
                "linked Outbox events must all be applied before coverage_ready"
            )
        return (
            snapshot.binding_count,
            snapshot.binding_digest,
            snapshot.mapped_event_count,
            snapshot.applied_event_count,
        )

    def _require_persisted_coverage_still_current(
        self,
        conn: sqlite3.Connection,
        *,
        job: sqlite3.Row,
    ) -> None:
        document_id = _optional_text(job["document_id"])
        document_version_id = _optional_text(job["document_version_id"])
        digest = _optional_text(job["coverage_binding_digest"])
        data_version_fingerprint = _optional_text(
            job["coverage_data_version_fingerprint"]
        )
        expected_count = job["coverage_expected_bindings"]
        covered_count = job["coverage_covered_bindings"]
        data_version_id = _optional_text(job["retrieval_data_version"])
        if (
            document_id is None
            or document_version_id is None
            or digest is None
            or data_version_fingerprint is None
            or data_version_id is None
            or expected_count is None
            or covered_count is None
            or job["coverage_checked_at"] is None
        ):
            raise DocumentIngestJobTransitionError(
                "active document ingest requires a durable coverage proof"
            )
        event_ids = self.list_outbox_event_ids(conn, str(job["job_id"]))
        try:
            snapshot = self._require_index_audit_port().inspect_job_coverage_in_transaction(
                conn,
                event_ids=event_ids,
                document_id=document_id,
                document_version_id=document_version_id,
                data_version_id=data_version_id,
            )
        except DocumentIndexAuditError as exc:
            raise DocumentIngestJobTransitionError(
                "durable coverage proof is no longer current"
            ) from exc
        if (
            data_version_id != str(job["target_generation_id"])
            or data_version_fingerprint != str(job["target_generation_fingerprint"])
            or int(expected_count) != snapshot.binding_count
            or int(covered_count) != snapshot.binding_count
            or digest != snapshot.binding_digest
            or job["coverage_mapped_events"] is None
            or int(job["coverage_mapped_events"])
            != snapshot.mapped_event_count
            or job["coverage_applied_events"] is None
            or int(job["coverage_applied_events"])
            != snapshot.applied_event_count
            or snapshot.mapped_event_count != snapshot.applied_event_count
        ):
            raise DocumentIngestJobTransitionError(
                "durable coverage proof is no longer current"
            )

    def _record_outbox_event_in_transaction(
        self,
        conn: sqlite3.Connection,
        *,
        job_id: str,
        event_id: str,
        recorded_at: str,
    ) -> None:
        conn.execute(
            "INSERT OR IGNORE INTO document_ingest_job_events "
            "(job_id, event_id, recorded_at) VALUES (?, ?, ?)",
            (job_id, event_id, recorded_at),
        )

    def document_target_is_current(
        self, conn: sqlite3.Connection, job: DocumentIngestJob,
    ) -> bool:
        """只读核对持久任务快照及其精确当前来源；不将历史完成状态冒充当前可用。"""

        if not isinstance(job, DocumentIngestJob):
            raise TypeError("job must be DocumentIngestJob")
        _rows_by_name(conn)
        row = self._get_row(conn, job.job_id)
        if row is None or _job_from_row(row) != job:
            return False
        if not job.document_id or not job.document_version_id:
            return False
        try:
            self._require_current_document_target(
                conn, job=row, document_id=job.document_id,
                document_version_id=job.document_version_id,
            )
        except DocumentIngestJobTransitionError:
            return False
        return True

    def _require_current_document_target(
        self,
        conn: sqlite3.Connection,
        *,
        job: sqlite3.Row,
        document_id: str,
        document_version_id: str,
    ) -> None:
        target = conn.execute(
            "SELECT 1 FROM documents d "
            "JOIN document_versions v "
            "ON v.id=d.current_version_id AND v.doc_id=d.id "
            "JOIN files f ON f.id=d.file_id AND f.current_version_id=v.file_version_id "
            "JOIN file_versions fv ON fv.id=v.file_version_id AND fv.file_id=f.id "
            "AND fv.content_sha256=v.source_sha256 AND fv.size_bytes=v.source_size "
            "WHERE d.id=? AND v.id=? AND d.file_id=? AND v.file_version_id=? "
            "AND d.path=? AND v.source_sha256=? "
            "AND v.source_size=? "
            "AND v.processor_fingerprint=? AND v.chunker_fingerprint=? "
            "AND v.chunk_contract_version=?",
            (
                document_id,
                document_version_id,
                str(job["file_id"]),
                str(job["file_version_id"]),
                str(job["canonical_path"]),
                str(job["source_sha256"]),
                int(job["source_size"]),
                str(job["processor_fingerprint"]),
                str(job["chunker_fingerprint"]),
                int(job["chunk_contract_version"]),
            ),
        ).fetchone()
        if target is None:
            raise DocumentIngestJobTransitionError(
                "document_version_id does not belong to the exact current FileVersion "
                "or match the job's frozen source/recipe"
            )

    def _require_index_audit_port(self) -> DocumentIndexAuditPort:
        if self._index_audit_port is None:
            raise RuntimeError(
                "document ingest index transitions require an index audit port"
            )
        return self._index_audit_port

    @staticmethod
    def _get_row(conn: sqlite3.Connection, job_id: str) -> sqlite3.Row | None:
        return conn.execute(
            f"SELECT {_JOB_COLUMNS} FROM document_ingest_jobs WHERE job_id=?",
            (job_id,),
        ).fetchone()

    def _require_row(self, conn: sqlite3.Connection, job_id: str) -> sqlite3.Row:
        row = self._get_row(conn, job_id)
        if row is None:
            raise KeyError(f"document ingest job not found: {job_id}")
        return row

    def _require_live_lease(
        self,
        conn: sqlite3.Connection,
        *,
        job_id: str,
        worker_id: str,
        lease_token: str,
        now: str,
    ) -> sqlite3.Row:
        row = self._require_row(conn, job_id)
        if (
            str(row["status"]) != DocumentIngestJobStatus.PROCESSING.value
            or str(row["lease_owner"] or "") != worker_id
            or str(row["lease_token"] or "") != lease_token
            or row["lease_until"] is None
            or str(row["lease_until"]) <= now
        ):
            raise DocumentIngestJobLeaseError(
                "document ingest job is not owned by this exact live lease"
            )
        return row


def _job_from_row(row: sqlite3.Row) -> DocumentIngestJob:
    return DocumentIngestJob(
        job_id=str(row["job_id"]),
        payload_fingerprint=str(row["payload_fingerprint"]),
        file_id=str(row["file_id"]),
        file_version_id=str(row["file_version_id"]),
        canonical_path=str(row["canonical_path"]),
        source_sha256=str(row["source_sha256"]),
        source_size=int(row["source_size"]),
        processor_fingerprint=str(row["processor_fingerprint"]),
        chunker_fingerprint=str(row["chunker_fingerprint"]),
        chunk_contract_version=int(row["chunk_contract_version"]),
        target_generation_id=str(row["target_generation_id"]),
        target_generation_fingerprint=str(row["target_generation_fingerprint"]),
        stage=DocumentIngestJobStage(str(row["stage"])),
        status=DocumentIngestJobStatus(str(row["status"])),
        attempts=int(row["attempts"]),
        next_retry_at=_optional_text(row["next_retry_at"]),
        lease_owner=_optional_text(row["lease_owner"]),
        lease_token=_optional_text(row["lease_token"]),
        lease_until=_optional_text(row["lease_until"]),
        completion_lease_token=_optional_text(row["completion_lease_token"]),
        document_id=_optional_text(row["document_id"]),
        document_version_id=_optional_text(row["document_version_id"]),
        retrieval_data_version=_optional_text(row["retrieval_data_version"]),
        coverage_data_version_fingerprint=_optional_text(
            row["coverage_data_version_fingerprint"]
        ),
        coverage_binding_digest=_optional_text(row["coverage_binding_digest"]),
        coverage_expected_bindings=(
            int(row["coverage_expected_bindings"])
            if row["coverage_expected_bindings"] is not None
            else None
        ),
        coverage_covered_bindings=(
            int(row["coverage_covered_bindings"])
            if row["coverage_covered_bindings"] is not None
            else None
        ),
        coverage_mapped_events=(
            int(row["coverage_mapped_events"])
            if row["coverage_mapped_events"] is not None
            else None
        ),
        coverage_applied_events=(
            int(row["coverage_applied_events"])
            if row["coverage_applied_events"] is not None
            else None
        ),
        coverage_checked_at=_optional_text(row["coverage_checked_at"]),
        reason_code=_optional_text(row["reason_code"]),
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
        completed_at=_optional_text(row["completed_at"]),
    )


def _row_matches_request(row: sqlite3.Row, request: DocumentIngestJobRequest) -> bool:
    return (
        str(row["payload_fingerprint"]) == request.payload_fingerprint
        and str(row["file_id"]) == request.file_id
        and str(row["file_version_id"]) == request.file_version_id
        and str(row["canonical_path"]) == request.canonical_path
        and str(row["source_sha256"]) == request.source_sha256
        and int(row["source_size"]) == request.source_size
        and str(row["processor_fingerprint"]) == request.processor_fingerprint
        and str(row["chunker_fingerprint"]) == request.chunker_fingerprint
        and int(row["chunk_contract_version"]) == request.chunk_contract_version
        and str(row["target_generation_id"]) == request.target_generation_id
        and str(row["target_generation_fingerprint"]) == request.target_generation_fingerprint
    )


def _merge_reference(name: str, current: object, proposed: str | None) -> str | None:
    existing = _optional_text(current)
    if proposed is None:
        return existing
    if existing is not None and existing != proposed:
        raise DocumentIngestJobTransitionError(
            f"{name} is immutable after its first durable checkpoint"
        )
    return proposed


def _optional_text(value: object) -> str | None:
    return None if value is None else str(value)


def _normalize_timestamp(name: str, value: str) -> str:
    _require_text(name, value)
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{name} must include a timezone")
    return parsed.astimezone(timezone.utc).isoformat(timespec="microseconds")


def _add_seconds(value: str, seconds: int) -> str:
    parsed = datetime.fromisoformat(value)
    return (parsed + timedelta(seconds=seconds)).isoformat(timespec="microseconds")


def _rows_by_name(conn: sqlite3.Connection) -> None:
    if conn.row_factory is not sqlite3.Row:
        conn.row_factory = sqlite3.Row


@contextmanager
def _immediate_transaction(conn: sqlite3.Connection) -> Iterator[None]:
    if conn.in_transaction:
        raise RuntimeError("standalone document ingest mutation requires no active transaction")
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield
    except BaseException:
        conn.rollback()
        raise
    else:
        conn.commit()
