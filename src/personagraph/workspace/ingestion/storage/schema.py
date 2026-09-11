"""Workspace ingestion 对共享项目数据库贡献的持久作业 DDL。"""

from __future__ import annotations

import sqlite3


INGESTION_REQUIRED_TABLES = frozenset(
    {"document_ingest_jobs", "document_ingest_job_events", "document_ingest_requests"}
)


_STATEMENTS = (
    """
    CREATE TABLE document_ingest_jobs (
        job_id TEXT PRIMARY KEY,
        payload_fingerprint TEXT NOT NULL,
        file_id TEXT NOT NULL,
        file_version_id TEXT NOT NULL,
        canonical_path TEXT NOT NULL,
        source_sha256 TEXT NOT NULL CHECK(
            length(source_sha256) = 64
            AND source_sha256 NOT GLOB '*[^0-9a-f]*'
        ),
        source_size INTEGER NOT NULL CHECK(source_size >= 0),
        processor_fingerprint TEXT NOT NULL CHECK(length(processor_fingerprint) > 0),
        chunker_fingerprint TEXT NOT NULL CHECK(length(chunker_fingerprint) > 0),
        chunk_contract_version INTEGER NOT NULL CHECK(chunk_contract_version > 0),
        target_generation_id TEXT NOT NULL CHECK(length(target_generation_id) > 0),
        target_generation_fingerprint TEXT NOT NULL CHECK(
            length(target_generation_fingerprint) > 0
        ),
        stage TEXT NOT NULL CHECK(stage IN (
            'parsing', 'chunked', 'indexing', 'coverage_ready', 'active'
        )),
        status TEXT NOT NULL CHECK(status IN (
            'pending', 'processing', 'applied', 'retryable_failed', 'terminal_failed'
        )),
        attempts INTEGER NOT NULL DEFAULT 0 CHECK(attempts >= 0),
        next_retry_at TEXT,
        lease_owner TEXT,
        lease_token TEXT,
        lease_until TEXT,
        completion_lease_token TEXT,
        document_id TEXT,
        document_version_id TEXT,
        retrieval_data_version TEXT,
        coverage_data_version_fingerprint TEXT,
        coverage_binding_digest TEXT,
        coverage_expected_bindings INTEGER,
        coverage_covered_bindings INTEGER,
        coverage_mapped_events INTEGER,
        coverage_applied_events INTEGER,
        coverage_checked_at TEXT,
        reason_code TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        completed_at TEXT,
        FOREIGN KEY(file_version_id, file_id) REFERENCES file_versions(id, file_id)
            ON DELETE RESTRICT,
        UNIQUE(file_version_id, processor_fingerprint, chunker_fingerprint,
            chunk_contract_version, target_generation_id, target_generation_fingerprint),
        CHECK(retrieval_data_version IS NULL OR retrieval_data_version=target_generation_id),
        CHECK(coverage_data_version_fingerprint IS NULL OR
            coverage_data_version_fingerprint=target_generation_fingerprint),
        CHECK(
            (status = 'processing'
                AND lease_owner IS NOT NULL
                AND lease_token IS NOT NULL
                AND lease_until IS NOT NULL)
            OR
            (status != 'processing'
                AND lease_owner IS NULL
                AND lease_token IS NULL
                AND lease_until IS NULL)
        ),
        CHECK(
            (status IN ('pending', 'retryable_failed') AND next_retry_at IS NOT NULL)
            OR
            (status NOT IN ('pending', 'retryable_failed') AND next_retry_at IS NULL)
        ),
        CHECK(
            (status IN ('applied', 'terminal_failed') AND completed_at IS NOT NULL)
            OR
            (status NOT IN ('applied', 'terminal_failed') AND completed_at IS NULL)
        ),
        CHECK(
            (status = 'applied' AND completion_lease_token IS NOT NULL)
            OR (status != 'applied' AND completion_lease_token IS NULL)
        ),
        CHECK((stage = 'active' AND status = 'applied') OR stage != 'active'),
        CHECK((status = 'applied' AND stage = 'active') OR status != 'applied'),
        CHECK(document_version_id IS NULL OR document_id IS NOT NULL),
        CHECK(retrieval_data_version IS NULL OR document_version_id IS NOT NULL),
        CHECK(
            stage != 'parsing'
            OR (
                document_id IS NULL
                AND document_version_id IS NULL
                AND retrieval_data_version IS NULL
            )
        ),
        CHECK(
            stage NOT IN ('chunked', 'indexing', 'coverage_ready', 'active')
            OR (
                document_id IS NOT NULL
                AND document_version_id IS NOT NULL
                AND retrieval_data_version IS NOT NULL
            )
        ),
        CHECK(
            (stage IN ('coverage_ready', 'active')
                AND coverage_data_version_fingerprint IS NOT NULL
                AND length(coverage_data_version_fingerprint) > 0
                AND coverage_binding_digest IS NOT NULL
                AND length(coverage_binding_digest) = 64
                AND coverage_binding_digest NOT GLOB '*[^0-9a-f]*'
                AND coverage_expected_bindings IS NOT NULL
                AND coverage_expected_bindings > 0
                AND coverage_covered_bindings IS NOT NULL
                AND coverage_covered_bindings = coverage_expected_bindings
                AND coverage_mapped_events IS NOT NULL
                AND coverage_mapped_events >= 0
                AND coverage_applied_events IS NOT NULL
                AND coverage_applied_events = coverage_mapped_events
                AND coverage_checked_at IS NOT NULL)
            OR
            (stage NOT IN ('coverage_ready', 'active')
                AND coverage_data_version_fingerprint IS NULL
                AND coverage_binding_digest IS NULL
                AND coverage_expected_bindings IS NULL
                AND coverage_covered_bindings IS NULL
                AND coverage_mapped_events IS NULL
                AND coverage_applied_events IS NULL
                AND coverage_checked_at IS NULL)
        ),
        CHECK(status != 'processing' OR attempts > 0),
        CHECK(
            status NOT IN ('retryable_failed', 'terminal_failed')
            OR reason_code IS NOT NULL
        )
    )
    """,
    """
    CREATE TABLE document_ingest_job_events (
        job_id TEXT NOT NULL,
        event_id TEXT NOT NULL,
        recorded_at TEXT NOT NULL,
        PRIMARY KEY(job_id, event_id),
        FOREIGN KEY(job_id) REFERENCES document_ingest_jobs(job_id) ON DELETE CASCADE,
        FOREIGN KEY(event_id) REFERENCES retrieval_update_outbox(event_id) ON DELETE RESTRICT
    )
    """,
    "CREATE INDEX idx_document_ingest_jobs_due "
    "ON document_ingest_jobs(status, next_retry_at, lease_until, created_at, job_id)",
    "CREATE INDEX idx_document_ingest_jobs_generation "
    "ON document_ingest_jobs(target_generation_id, target_generation_fingerprint, status)",
    "CREATE INDEX idx_document_ingest_job_events_job "
    "ON document_ingest_job_events(job_id, recorded_at, event_id)",
    "CREATE INDEX idx_document_ingest_job_events_event "
    "ON document_ingest_job_events(event_id, job_id)",
    """
    CREATE TRIGGER trg_document_ingest_jobs_immutable_request
    BEFORE UPDATE OF job_id, payload_fingerprint, file_id, file_version_id, canonical_path,
        source_sha256, source_size, processor_fingerprint,
        chunker_fingerprint, chunk_contract_version, target_generation_id,
        target_generation_fingerprint
    ON document_ingest_jobs
    WHEN NEW.job_id != OLD.job_id
      OR NEW.payload_fingerprint != OLD.payload_fingerprint
      OR NEW.file_id != OLD.file_id
      OR NEW.file_version_id != OLD.file_version_id
      OR NEW.canonical_path != OLD.canonical_path
      OR NEW.source_sha256 != OLD.source_sha256
      OR NEW.source_size != OLD.source_size
      OR NEW.processor_fingerprint != OLD.processor_fingerprint
      OR NEW.chunker_fingerprint != OLD.chunker_fingerprint
      OR NEW.chunk_contract_version != OLD.chunk_contract_version
      OR NEW.target_generation_id != OLD.target_generation_id
      OR NEW.target_generation_fingerprint != OLD.target_generation_fingerprint
    BEGIN
        SELECT RAISE(ABORT, 'document ingest request payload is immutable');
    END
    """,
    """
    CREATE TRIGGER trg_document_ingest_jobs_monotonic_stage
    BEFORE UPDATE OF stage ON document_ingest_jobs
    WHEN (
        CASE NEW.stage
            WHEN 'parsing' THEN 0 WHEN 'chunked' THEN 1 WHEN 'indexing' THEN 2
            WHEN 'coverage_ready' THEN 3 WHEN 'active' THEN 4
        END
    ) NOT IN (
        CASE OLD.stage
            WHEN 'parsing' THEN 0 WHEN 'chunked' THEN 1 WHEN 'indexing' THEN 2
            WHEN 'coverage_ready' THEN 3 WHEN 'active' THEN 4
        END,
        CASE OLD.stage
            WHEN 'parsing' THEN 1 WHEN 'chunked' THEN 2 WHEN 'indexing' THEN 3
            WHEN 'coverage_ready' THEN 4 WHEN 'active' THEN 4
        END
    )
    BEGIN
        SELECT RAISE(ABORT, 'document ingest stage must advance exactly once');
    END
    """,
    """
    CREATE TRIGGER trg_document_ingest_jobs_immutable_refs
    BEFORE UPDATE OF document_id, document_version_id, retrieval_data_version,
        completion_lease_token,
        coverage_data_version_fingerprint, coverage_binding_digest,
        coverage_expected_bindings, coverage_covered_bindings,
        coverage_mapped_events, coverage_applied_events, coverage_checked_at
    ON document_ingest_jobs
    WHEN (OLD.document_id IS NOT NULL AND NEW.document_id IS NOT OLD.document_id)
      OR (OLD.document_version_id IS NOT NULL
          AND NEW.document_version_id IS NOT OLD.document_version_id)
      OR (OLD.retrieval_data_version IS NOT NULL
          AND NEW.retrieval_data_version IS NOT OLD.retrieval_data_version)
      OR (OLD.completion_lease_token IS NOT NULL
          AND NEW.completion_lease_token IS NOT OLD.completion_lease_token)
      OR (OLD.coverage_data_version_fingerprint IS NOT NULL
          AND NEW.coverage_data_version_fingerprint
              IS NOT OLD.coverage_data_version_fingerprint)
      OR (OLD.coverage_binding_digest IS NOT NULL
          AND NEW.coverage_binding_digest IS NOT OLD.coverage_binding_digest)
      OR (OLD.coverage_expected_bindings IS NOT NULL
          AND NEW.coverage_expected_bindings IS NOT OLD.coverage_expected_bindings)
      OR (OLD.coverage_covered_bindings IS NOT NULL
          AND NEW.coverage_covered_bindings IS NOT OLD.coverage_covered_bindings)
      OR (OLD.coverage_mapped_events IS NOT NULL
          AND NEW.coverage_mapped_events IS NOT OLD.coverage_mapped_events)
      OR (OLD.coverage_applied_events IS NOT NULL
          AND NEW.coverage_applied_events IS NOT OLD.coverage_applied_events)
      OR (OLD.coverage_checked_at IS NOT NULL
          AND NEW.coverage_checked_at IS NOT OLD.coverage_checked_at)
    BEGIN
        SELECT RAISE(ABORT, 'document ingest authority references are immutable');
    END
    """,
    """
    CREATE TABLE document_ingest_requests (
        request_id TEXT PRIMARY KEY,
        job_id TEXT NOT NULL,
        session_id TEXT NOT NULL CHECK(length(session_id) > 0),
        with_summary INTEGER NOT NULL CHECK(with_summary IN (0, 1)),
        source_mtime_ns INTEGER NOT NULL CHECK(source_mtime_ns >= 0),
        delivery_status TEXT NOT NULL CHECK(delivery_status IN ('pending', 'mounted', 'blocked')),
        reason_code TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        FOREIGN KEY(job_id) REFERENCES document_ingest_jobs(job_id) ON DELETE CASCADE,
        CHECK((delivery_status='blocked' AND reason_code IS NOT NULL)
            OR (delivery_status!='blocked' AND reason_code IS NULL))
    )
    """,
    "CREATE INDEX idx_document_ingest_requests_session "
    "ON document_ingest_requests(session_id, created_at DESC, request_id DESC)",
    "CREATE INDEX idx_document_ingest_requests_job "
    "ON document_ingest_requests(job_id, delivery_status, created_at, request_id)",
    "CREATE INDEX idx_document_ingest_requests_pending "
    "ON document_ingest_requests(delivery_status, created_at, request_id)",
    """
    CREATE TRIGGER trg_document_ingest_requests_immutable_payload
    BEFORE UPDATE OF request_id, job_id, session_id, with_summary, source_mtime_ns, created_at
    ON document_ingest_requests
    WHEN NEW.request_id != OLD.request_id OR NEW.job_id != OLD.job_id
      OR NEW.session_id != OLD.session_id OR NEW.with_summary != OLD.with_summary
      OR NEW.source_mtime_ns != OLD.source_mtime_ns OR NEW.created_at != OLD.created_at
    BEGIN
        SELECT RAISE(ABORT, 'file preparation request payload is immutable');
    END
    """,
    """
    CREATE TRIGGER trg_document_ingest_requests_delivery
    BEFORE UPDATE OF delivery_status, reason_code ON document_ingest_requests
    WHEN (OLD.delivery_status != 'pending'
        AND NOT (OLD.delivery_status='blocked' AND NEW.delivery_status='pending')
        AND (NEW.delivery_status != OLD.delivery_status OR NEW.reason_code IS NOT OLD.reason_code))
      OR (NEW.delivery_status='mounted' AND NOT EXISTS (
          SELECT 1 FROM document_ingest_jobs WHERE job_id=NEW.job_id AND status='applied'))
    BEGIN
        SELECT RAISE(ABORT, 'file preparation delivery transition is invalid');
    END
    """,
)


class IngestionSchemaContributionError(RuntimeError):
    """Ingestion contribution 缺失或损坏。"""


def initialize_ingestion_schema(conn: sqlite3.Connection) -> None:
    for statement in _STATEMENTS:
        conn.execute(statement)


def validate_ingestion_schema(conn: sqlite3.Connection) -> None:
    present = frozenset(
        str(row[0])
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    )
    missing = sorted(INGESTION_REQUIRED_TABLES - present)
    if missing:
        raise IngestionSchemaContributionError(
            "workspace ingestion schema is incomplete: " + ", ".join(missing)
        )
    columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(document_ingest_jobs)")}
    if not {"file_id", "file_version_id", "target_generation_id", "target_generation_fingerprint"} <= columns:
        raise IngestionSchemaContributionError("workspace ingestion shared job contract is missing")
    if columns & {"session_id", "with_summary", "source_mtime_ns"}:
        raise IngestionSchemaContributionError("workspace ingestion job still contains Session request state")


def migrate_shared_ingestion_schema(conn: sqlite3.Connection) -> None:
    """v6→v7 按已批准的退役策略清除旧任务，不删除任何来源或索引内容。

    旧任务未冻结共享 FileVersion/目标 generation，不能安全恢复；未完成任务需重新发起。
    仅项目 schema 迁移调用此贡献，当前版本重开不会再次执行清理。
    """

    if not conn.in_transaction:
        raise RuntimeError("shared ingestion migration requires an active transaction")
    conn.execute("DROP TABLE IF EXISTS document_ingest_requests")
    conn.execute("DROP TABLE document_ingest_job_events")
    conn.execute("DROP TABLE document_ingest_jobs")
    initialize_ingestion_schema(conn)
