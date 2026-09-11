from __future__ import annotations

import hashlib
from contextlib import contextmanager
import sqlite3

import pytest

from personagraph.workspace.storage import DocumentDatabase
from personagraph.workspace.storage.context import bind
from personagraph.workspace.ingestion.storage import (
    DocumentIngestJobIdCollision,
    DocumentIngestJobLeaseError,
    DocumentIngestJobRequest,
    DocumentIngestJobStage,
    DocumentIngestJobStatus,
    DocumentIngestJobTransitionError,
    DocumentIngestCoverageProof,
    FilePreparationDeliveryStatus,
    SqliteFilePreparationRequestStore,
    SqliteDocumentIngestJobStore,
    document_ingest_binding_digest,
)
from personagraph.retrieval.contracts import SourceType, SourceUnitRef
from personagraph.retrieval.lifecycle.outbox import (
    RetrievalUpdateEvent,
    RetrievalUpdateKind,
    SqliteRetrievalOutbox,
)
from personagraph.retrieval.sources.events import SqliteDocumentIndexPort


NOW = "2026-08-19T12:00:00+00:00"


@contextmanager
def _connection(tmp_path, name: str = "documents.sqlite"):
    project_root = tmp_path / "project"
    project_root.mkdir(exist_ok=True)
    database = DocumentDatabase(
        project_id="ingest-jobs-test",
        project_root=project_root,
        db_path=tmp_path / name,
    )
    from personagraph.session import store as session_store

    with bind(database), session_store.session_database_scope("session-1"):
        session_store.init_db()
        with session_store._connect() as session_conn:
            session_conn.execute(
                "INSERT OR IGNORE INTO sessions "
                "(id, persona_id, title, status, created_at, last_active_at) "
                "VALUES ('session-1', 'Entelecheia', 'ingest jobs', 'active', ?, ?)",
                (NOW, NOW),
            )
        conn = database.open_connection()
        try:
            conn.execute(
                "INSERT OR IGNORE INTO retrieval_data_versions "
                "(id, fingerprint, role, state, created_at, activated_at) "
                "VALUES ('retrieval-v1', 'generation-fingerprint-v1', "
                "'active', 'ready', ?, ?)",
                (NOW, NOW),
            )
            _register_source_version(conn)
            conn.commit()
            yield conn
        finally:
            conn.close()


def _register_source_version(conn: sqlite3.Connection) -> None:
    conn.execute(
        "INSERT INTO files (id,project_id,relative_path,origin,current_version_id,created_at,updated_at) "
        "VALUES ('file-doc-1','ingest-jobs-test','papers/paper.pdf','workspace_existing',NULL,?,?)",
        (NOW, NOW),
    )
    conn.execute(
        "INSERT INTO file_versions (id,file_id,version_number,producer,content_sha256,"
        "size_bytes,source_mtime_ns,created_at) VALUES "
        "('file-docv-1','file-doc-1',1,'workspace_existing',?,1024,123456789,?)",
        ("a" * 64, NOW),
    )
    conn.execute("UPDATE files SET current_version_id='file-docv-1' WHERE id='file-doc-1'")


def _request(
    job_id: str = "ingest-1",
    *,
    file_id: str = "file-doc-1",
    file_version_id: str = "file-docv-1",
    canonical_path: str = "/workspace/papers/paper.pdf",
    source_sha256: str = "a" * 64,
    source_size: int = 1024,
    processor_fingerprint: str = "native-pdf@1",
    chunker_fingerprint: str = "structure_first@1+heuristic",
    chunk_contract_version: int = 1,
    target_generation_id: str = "retrieval-v1",
    target_generation_fingerprint: str = "generation-fingerprint-v1",
) -> DocumentIngestJobRequest:
    return DocumentIngestJobRequest(
        job_id=job_id,
        file_id=file_id,
        file_version_id=file_version_id,
        canonical_path=canonical_path,
        source_sha256=source_sha256,
        source_size=source_size,
        processor_fingerprint=processor_fingerprint,
        chunker_fingerprint=chunker_fingerprint,
        chunk_contract_version=chunk_contract_version,
        target_generation_id=target_generation_id,
        target_generation_fingerprint=target_generation_fingerprint,
    )


def _enqueue_and_claim(
    conn: sqlite3.Connection,
    *,
    worker_id: str = "worker-a",
):
    jobs = SqliteDocumentIngestJobStore(
        index_audit_port=SqliteDocumentIndexPort(),
    )
    jobs.enqueue(conn, request=_request(), now=NOW)
    claimed = jobs.claim_due(
        conn,
        worker_id=worker_id,
        now=NOW,
        lease_seconds=30,
        limit=10,
    )
    assert len(claimed) == 1
    return jobs, claimed[0]


def _insert_document_authority(
    conn: sqlite3.Connection,
    *,
    doc_id: str = "doc-1",
    version_id: str = "docv-1",
    version_number: int = 1,
    session_id: str = "session-1",
    source_sha256: str = "a" * 64,
    source_size: int = 1024,
    source_mtime_ns: int = 123456789,
    processor_fingerprint: str = "native-pdf@1",
    chunker_fingerprint: str = "structure_first@1+heuristic",
    chunk_contract_version: int = 1,
    canonical_path: str = "/workspace/papers/paper.pdf",
) -> None:
    file_id = f"file-{doc_id}"
    file_version_id = f"file-{version_id}"
    if conn.execute("SELECT 1 FROM files WHERE id=?", (file_id,)).fetchone() is None:
        conn.execute(
            "INSERT INTO files (id, project_id, relative_path, origin, media_type, "
            "created_at, updated_at) VALUES (?, 'ingest-jobs-test', ?, "
            "'workspace_existing', 'application/pdf', ?, ?)",
            (file_id, f"papers/{doc_id}.pdf", NOW, NOW),
        )
    conn.execute(
        "INSERT OR IGNORE INTO file_versions (id, file_id, version_number, producer, "
        "content_sha256, size_bytes, source_mtime_ns, created_at) "
        "VALUES (?, ?, ?, 'workspace_existing', ?, ?, ?, ?)",
        (
            file_version_id,
            file_id,
            int(conn.execute("SELECT COALESCE(MAX(version_number), 0)+1 FROM file_versions WHERE file_id=?", (file_id,)).fetchone()[0]),
            source_sha256,
            source_size,
            source_mtime_ns,
            NOW,
        ),
    )
    if conn.execute("SELECT 1 FROM documents WHERE id=?", (doc_id,)).fetchone() is None:
        conn.execute(
            "INSERT INTO documents (id, path, title, mime, hash, content_hash, "
            "current_version_id, file_id) VALUES (?, ?, ?, 'pdf', ?, ?, NULL, ?)",
            (
                doc_id,
                canonical_path,
                doc_id,
                f"hash-{doc_id}",
                f"hash-{doc_id}",
                file_id,
            ),
        )
    conn.execute(
        "INSERT INTO document_versions (id, doc_id, version_number, source_sha256, "
        "source_size, source_mtime_ns, status, captured_at, processing_status, "
        "diagnostics_json, processor_fingerprint, chunker_fingerprint, "
        "chunk_contract_version, file_version_id) VALUES (?, ?, ?, ?, ?, ?, "
        "'active', ?, 'complete', '[]', ?, ?, ?, ?)",
        (
            version_id,
            doc_id,
            version_number,
            source_sha256,
            source_size,
            source_mtime_ns,
            NOW,
            processor_fingerprint,
            chunker_fingerprint,
            chunk_contract_version,
            file_version_id,
        ),
    )
    conn.execute(
        "UPDATE documents SET current_version_id=? WHERE id=?",
        (version_id, doc_id),
    )
    conn.execute(
        "UPDATE files SET current_version_id=?, updated_at=? WHERE id=?",
        (file_version_id, NOW, file_id),
    )
    content = f"Evidence for {doc_id} at {version_id}."
    conn.execute(
        "INSERT INTO doc_chunks (id, doc_id, seq, loc, content, source_version_id, "
        "processor_fingerprint, chunker_fingerprint, producer_chunk_id, content_sha256, "
        "chunk_contract_version) VALUES (?, ?, 0, 'p1', ?, ?, ?, ?, 'chunk-1', ?, ?)",
        (
            f"storage-{doc_id}-{version_id}",
            doc_id,
            content,
            version_id,
            processor_fingerprint,
            chunker_fingerprint,
            hashlib.sha256(content.encode("utf-8")).hexdigest(),
            chunk_contract_version,
        ),
    )
    conn.commit()
    # 共享作业的完成不依赖任何 Session mount；挂载由请求交付单独负责。


def _current_upsert_event(
    *,
    doc_id: str = "doc-1",
    version_id: str = "docv-1",
    session_id: str = "session-1",
    data_version: str = "retrieval-v1",
    event_id: str = "current-upsert",
) -> RetrievalUpdateEvent:
    content = f"Evidence for {doc_id} at {version_id}."
    return RetrievalUpdateEvent(
        event_id=event_id,
        kind=RetrievalUpdateKind.UPSERT,
        ref=SourceUnitRef(
            SourceType.DOCUMENT,
            f"document-v3:{doc_id}:chunk-1",
            version_id,
            hashlib.sha256(content.encode("utf-8")).hexdigest(),
        ),
        retrieval_data_version=data_version,
        occurred_at=NOW,
    )


def _coverage_proof(*events: RetrievalUpdateEvent) -> DocumentIngestCoverageProof:
    return DocumentIngestCoverageProof(
        retrieval_data_version_id="retrieval-v1",
        retrieval_data_version_fingerprint="generation-fingerprint-v1",
        covered_binding_digest=document_ingest_binding_digest(
            tuple(
                (
                    event.ref.source_unit_id,
                    event.ref.source_revision,
                    event.ref.indexed_content_hash,
                )
                for event in events
                if event.kind in {RetrievalUpdateKind.UPSERT, RetrievalUpdateKind.RESTORE}
            )
        ),
        covered_binding_count=sum(
            event.kind in {RetrievalUpdateKind.UPSERT, RetrievalUpdateKind.RESTORE}
            for event in events
        ),
    )


@pytest.mark.parametrize(
    "changed",
    [
        {"file_id": "different-file"},
        {"file_version_id": "different-version"},
        {"canonical_path": "/workspace/papers/other.pdf"},
        {"source_sha256": "b" * 64},
        {"source_size": 2048},
        {"processor_fingerprint": "docling@2"},
        {"chunker_fingerprint": "structure_first@2+tokens"},
        {"chunk_contract_version": 2},
        {"target_generation_id": "other-generation"},
        {"target_generation_fingerprint": "other-fingerprint"},
    ],
)
def test_enqueue_is_exactly_idempotent_and_rejects_payload_collision(tmp_path, changed):
    jobs = SqliteDocumentIngestJobStore(
        index_audit_port=SqliteDocumentIndexPort(),
    )
    with _connection(tmp_path) as conn:
        created = jobs.enqueue(conn, request=_request(), now=NOW)
        replay = jobs.enqueue(
            conn,
            request=_request(),
            now="2026-08-19T12:00:10+00:00",
        )

        assert created.replayed is False
        assert replay.replayed is True
        assert replay.job == created.job
        assert replay.job.stage is DocumentIngestJobStage.PARSING
        assert replay.job.status is DocumentIngestJobStatus.PENDING

        with pytest.raises(DocumentIngestJobIdCollision, match="different payload"):
            jobs.enqueue(conn, request=_request(**changed), now=NOW)

        stored = jobs.get(conn, "ingest-1")
        assert stored == created.job


def test_shared_job_keeps_session_requests_and_delivery_independent(tmp_path):
    jobs = SqliteDocumentIngestJobStore()
    requests = SqliteFilePreparationRequestStore()
    with _connection(tmp_path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        first = jobs.enqueue_in_transaction(conn, request=_request(), now=NOW)
        second = jobs.enqueue_in_transaction(conn, request=_request("other-id"), now=NOW)
        assert second.replayed and second.job == first.job
        first_request = requests.create_in_transaction(
            conn, request_id="request-1", job_id=first.job.job_id, session_id="session-1",
            with_summary=False, source_mtime_ns=10, now=NOW,
        )
        second_request = requests.create_in_transaction(
            conn, request_id="request-2", job_id=second.job.job_id, session_id="session-2",
            with_summary=True, source_mtime_ns=20, now=NOW,
        )
        assert requests.create_in_transaction(
            conn, request_id="request-1", job_id=first.job.job_id, session_id="session-1",
            with_summary=False, source_mtime_ns=10, now=NOW,
        ) == first_request
        conn.commit()
        assert len(jobs.list(conn)) == 1
        assert requests.list_for_job(conn, first.job.job_id) == (first_request, second_request)
        assert requests.list_for_session(conn, "session-2") == (second_request,)
        assert requests.list_pending(conn) == ()  # 共享作业尚未应用，不能交付。
        conn.execute("BEGIN IMMEDIATE")
        with pytest.raises(DocumentIngestJobIdCollision):
            requests.create_in_transaction(
                conn, request_id="request-1", job_id=first.job.job_id, session_id="session-2",
                with_summary=False, source_mtime_ns=10, now=NOW,
            )
        conn.rollback()
        with pytest.raises(DocumentIngestJobTransitionError, match="applied shared job"):
            requests.mark_mounted(conn, "request-1", now=NOW)
        blocked = requests.mark_blocked(conn, "request-1", reason_code="authority_revoked", now=NOW)
        assert blocked.delivery_status is FilePreparationDeliveryStatus.BLOCKED
        assert requests.list_for_job(
            conn, first.job.job_id, delivery_status=FilePreparationDeliveryStatus.PENDING,
            limit=1,
        ) == (second_request,)
        conn.execute("BEGIN IMMEDIATE")
        requests.retry_blocked_in_transaction(conn, "request-1", now=NOW)
        conn.rollback()
        assert requests.get(conn, "request-1") == blocked
        assert requests.retry_blocked(conn, "request-1", now=NOW) == first_request
        assert requests.get(conn, "request-2") == second_request


def test_session_request_status_filter_is_applied_before_limit(tmp_path):
    jobs = SqliteDocumentIngestJobStore()
    requests = SqliteFilePreparationRequestStore()
    with _connection(tmp_path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        shared = jobs.enqueue_in_transaction(conn, request=_request(), now=NOW).job
        older = requests.create_in_transaction(
            conn,
            request_id="older-blocked",
            job_id=shared.job_id,
            session_id="session-1",
            with_summary=False,
            source_mtime_ns=10,
            now=NOW,
        )
        newer = requests.create_in_transaction(
            conn,
            request_id="newer-pending",
            job_id=shared.job_id,
            session_id="session-1",
            with_summary=False,
            source_mtime_ns=10,
            now="2026-08-19T12:00:01+00:00",
        )
        requests.create_in_transaction(
            conn,
            request_id="other-session-blocked",
            job_id=shared.job_id,
            session_id="session-2",
            with_summary=False,
            source_mtime_ns=10,
            now="2026-08-19T12:00:02+00:00",
        )
        conn.commit()
        older = requests.mark_blocked(
            conn,
            older.request_id,
            reason_code="authority_revoked",
            now="2026-08-19T12:00:03+00:00",
        )
        requests.mark_blocked(
            conn,
            "other-session-blocked",
            reason_code="authority_revoked",
            now="2026-08-19T12:00:03+00:00",
        )

        assert requests.list_for_session(
            conn,
            "session-1",
            delivery_statuses=(FilePreparationDeliveryStatus.BLOCKED,),
            limit=1,
        ) == (older,)
        assert requests.list_for_session(
            conn,
            "session-1",
            pending_job_statuses=(DocumentIngestJobStatus.PENDING,),
            limit=1,
        ) == (newer,)
        assert requests.list_for_session(
            conn,
            "session-1",
            delivery_statuses=(),
            pending_job_statuses=(),
            limit=1,
        ) == ()


def test_claim_filters_the_exact_target_generation(tmp_path):
    jobs = SqliteDocumentIngestJobStore()
    with _connection(tmp_path) as conn:
        jobs.enqueue(conn, request=_request("job-a"), now=NOW)
        jobs.enqueue(conn, request=_request(
            "job-b", target_generation_id="retrieval-v2",
            target_generation_fingerprint="generation-fingerprint-v2",
        ), now=NOW)
        assert jobs.claim_due(
            conn, worker_id="worker-a", now=NOW, lease_seconds=30,
            target_generation_id="retrieval-v1", target_generation_fingerprint="wrong",
        ) == ()
        claimed = jobs.claim_due(
            conn, worker_id="worker-b", now=NOW, lease_seconds=30,
            target_generation_id="retrieval-v2",
            target_generation_fingerprint="generation-fingerprint-v2",
        )
        assert [job.job_id for job in claimed] == ["job-b"]
        assert jobs.get(conn, "job-a").status is DocumentIngestJobStatus.PENDING


def test_enqueue_payload_is_immutable_even_to_direct_sql(tmp_path):
    jobs = SqliteDocumentIngestJobStore(
        index_audit_port=SqliteDocumentIndexPort(),
    )
    with _connection(tmp_path) as conn:
        jobs.enqueue(conn, request=_request(), now=NOW)
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            conn.execute(
                "UPDATE document_ingest_jobs SET canonical_path=? WHERE job_id=?",
                ("/workspace/rebound.pdf", "ingest-1"),
            )

        row = conn.execute(
            "SELECT canonical_path FROM document_ingest_jobs WHERE job_id='ingest-1'"
        ).fetchone()
        assert row[0] == "/workspace/papers/paper.pdf"


def test_claim_renew_and_finish_require_the_exact_live_lease(tmp_path):
    with _connection(tmp_path) as conn:
        jobs, claimed = _enqueue_and_claim(conn)
        _insert_document_authority(conn)
        outbox = SqliteRetrievalOutbox()
        event = _current_upsert_event()
        conn.execute("BEGIN IMMEDIATE")
        assert outbox.enqueue(conn, event)
        conn.commit()

        assert claimed.status is DocumentIngestJobStatus.PROCESSING
        assert claimed.attempts == 1
        assert claimed.lease_owner == "worker-a"
        assert claimed.lease_token
        assert jobs.claim_due(
            conn,
            worker_id="worker-b",
            now="2026-08-19T12:00:10+00:00",
            lease_seconds=30,
        ) == ()

        with pytest.raises(DocumentIngestJobLeaseError, match="live lease"):
            jobs.renew_lease(
                conn,
                job_id=claimed.job_id,
                worker_id="worker-a",
                lease_token="wrong-token",
                now="2026-08-19T12:00:10+00:00",
                lease_seconds=30,
            )

        renewed = jobs.renew_lease(
            conn,
            job_id=claimed.job_id,
            worker_id="worker-a",
            lease_token=claimed.lease_token or "",
            now="2026-08-19T12:00:10+00:00",
            lease_seconds=40,
        )
        assert renewed.lease_token == claimed.lease_token
        assert renewed.lease_until == "2026-08-19T12:00:50.000000+00:00"

        jobs.checkpoint_stage(
            conn,
            job_id=claimed.job_id,
            worker_id="worker-a",
            lease_token=claimed.lease_token or "",
            stage=DocumentIngestJobStage.CHUNKED,
            now="2026-08-19T12:00:20+00:00",
            document_id="doc-1",
            document_version_id="docv-1",
            retrieval_data_version="retrieval-v1",
            outbox_event_ids=(event.event_id,),
        )
        jobs.checkpoint_stage(
            conn,
            job_id=claimed.job_id,
            worker_id="worker-a",
            lease_token=claimed.lease_token or "",
            stage=DocumentIngestJobStage.INDEXING,
            now="2026-08-19T12:00:21+00:00",
        )
        assert outbox.claim_due(
            conn,
            worker_id="indexer",
            now="2026-08-19T12:00:22+00:00",
            lease_seconds=30,
            limit=10,
        ) == [event]
        outbox.mark_applied(
            conn,
            event_id=event.event_id,
            worker_id="indexer",
            now="2026-08-19T12:00:23+00:00",
        )
        conn.commit()
        jobs.checkpoint_stage(
            conn,
            job_id=claimed.job_id,
            worker_id="worker-a",
            lease_token=claimed.lease_token or "",
            stage=DocumentIngestJobStage.COVERAGE_READY,
            now="2026-08-19T12:00:24+00:00",
            coverage_proof=_coverage_proof(event),
        )
        applied = jobs.mark_applied(
            conn,
            job_id=claimed.job_id,
            worker_id="worker-a",
            lease_token=claimed.lease_token or "",
            now="2026-08-19T12:00:25+00:00",
        )
        assert applied.status is DocumentIngestJobStatus.APPLIED
        assert applied.stage is DocumentIngestJobStage.ACTIVE
        assert applied.completed_at == "2026-08-19T12:00:25.000000+00:00"
        assert applied.lease_owner is None
        assert applied.lease_token is None

        requests = SqliteFilePreparationRequestStore()
        incomplete = jobs.enqueue(
            conn, request=_request("incomplete-job", target_generation_id="future-generation"),
            now=NOW,
        ).job
        conn.execute("BEGIN IMMEDIATE")
        for request_id, job_id, requested_at in (
            ("older-incomplete", incomplete.job_id, NOW),
            ("ready-request", applied.job_id, "2026-08-19T12:00:26+00:00"),
        ):
            requests.create_in_transaction(
                conn, request_id=request_id, job_id=job_id, session_id="session-1",
                with_summary=False, source_mtime_ns=123456789, now=requested_at,
            )
        conn.commit()
        assert [item.request_id for item in requests.list_pending(conn, limit=1)] == ["ready-request"]
        assert requests.list_for_session(conn, "session-1", limit=1)[0].request_id == "ready-request"
        mounted = requests.mark_mounted(conn, "ready-request", now="2026-08-19T12:00:27+00:00")
        assert mounted.delivery_status is FilePreparationDeliveryStatus.MOUNTED
        assert requests.mark_mounted(conn, "ready-request", now=NOW) == mounted
        with pytest.raises(DocumentIngestJobTransitionError, match="already final"):
            requests.retry_blocked(conn, "ready-request", now=NOW)
        assert requests.list_pending(conn) == ()

        # 工作器可能在提交后丢失响应；再次完成属于安全重放。
        assert jobs.mark_applied(
            conn,
            job_id=claimed.job_id,
            worker_id="worker-a",
            lease_token=claimed.lease_token or "",
            now="2026-08-19T12:00:26+00:00",
        ) == applied
        with pytest.raises(DocumentIngestJobLeaseError, match="completion lease token"):
            jobs.mark_applied(
                conn,
                job_id=claimed.job_id,
                worker_id="worker-a",
                lease_token="wrong-token",
                now="2026-08-19T12:00:27+00:00",
            )


def test_expired_claim_is_recovered_and_old_lease_is_fenced(tmp_path):
    with _connection(tmp_path) as conn:
        jobs, first = _enqueue_and_claim(conn, worker_id="same-worker-name")
        reclaimed = jobs.claim_due(
            conn,
            worker_id="same-worker-name",
            now="2026-08-19T12:00:31+00:00",
            lease_seconds=30,
        )

        assert len(reclaimed) == 1
        second = reclaimed[0]
        assert second.attempts == 2
        assert second.lease_token != first.lease_token

        with pytest.raises(DocumentIngestJobLeaseError, match="live lease"):
            jobs.mark_terminal_failure(
                conn,
                job_id=first.job_id,
                worker_id="same-worker-name",
                lease_token=first.lease_token or "",
                now="2026-08-19T12:00:32+00:00",
                reason_code="stale_worker_result",
            )

        failed = jobs.mark_terminal_failure(
            conn,
            job_id=second.job_id,
            worker_id="same-worker-name",
            lease_token=second.lease_token or "",
            now="2026-08-19T12:00:32+00:00",
            reason_code="unsupported_document",
        )
        assert failed.status is DocumentIngestJobStatus.TERMINAL_FAILED
        assert failed.reason_code == "unsupported_document"


def test_retryable_backoff_and_explicit_terminal_retry_preserve_checkpoint(tmp_path):
    jobs = SqliteDocumentIngestJobStore(
        index_audit_port=SqliteDocumentIndexPort(),
    )
    with _connection(tmp_path) as conn:
        _insert_document_authority(conn)
        jobs.enqueue(conn, request=_request(), now=NOW)
        first = jobs.claim_due(
            conn, worker_id="worker-a", now=NOW, lease_seconds=30
        )[0]
        chunked = jobs.checkpoint_stage(
            conn,
            job_id=first.job_id,
            worker_id="worker-a",
            lease_token=first.lease_token or "",
            stage=DocumentIngestJobStage.CHUNKED,
            now="2026-08-19T12:00:05+00:00",
            document_id="doc-1",
            document_version_id="docv-1",
            retrieval_data_version="retrieval-v1",
        )
        assert chunked.stage is DocumentIngestJobStage.CHUNKED
        failed = jobs.mark_retryable_failure(
            conn,
            job_id=first.job_id,
            worker_id="worker-a",
            lease_token=first.lease_token or "",
            now="2026-08-19T12:00:06+00:00",
            retry_after_seconds=60,
            reason_code="database_busy",
        )
        assert failed.next_retry_at == "2026-08-19T12:01:06.000000+00:00"
        assert jobs.claim_due(
            conn,
            worker_id="worker-b",
            now="2026-08-19T12:01:05+00:00",
            lease_seconds=30,
        ) == ()

        second = jobs.claim_due(
            conn,
            worker_id="worker-b",
            now="2026-08-19T12:01:07+00:00",
            lease_seconds=30,
        )[0]
        assert second.stage is DocumentIngestJobStage.CHUNKED
        terminal = jobs.mark_terminal_failure(
            conn,
            job_id=second.job_id,
            worker_id="worker-b",
            lease_token=second.lease_token or "",
            now="2026-08-19T12:01:08+00:00",
            reason_code="parser_contract_unknown",
        )

        with pytest.raises(DocumentIngestJobIdCollision, match="different payload"):
            jobs.enqueue(
                conn,
                request=_request(source_sha256="b" * 64),
                now="2026-08-19T12:01:30+00:00",
            )

        retried = jobs.retry_terminal_failure(
            conn,
            job_id=terminal.job_id,
            now="2026-08-19T12:02:00+00:00",
        )
        assert retried.status is DocumentIngestJobStatus.PENDING
        assert retried.stage is DocumentIngestJobStage.CHUNKED
        assert retried.reason_code == "parser_contract_unknown"
        assert retried.completed_at is None
        assert jobs.claim_due(
            conn,
            worker_id="worker-c",
            now="2026-08-19T12:02:00+00:00",
            lease_seconds=30,
        )[0].attempts == 3


def test_checkpoint_and_outbox_link_can_share_one_rollback_boundary(tmp_path):
    jobs = SqliteDocumentIngestJobStore(
        index_audit_port=SqliteDocumentIndexPort(),
    )
    outbox = SqliteRetrievalOutbox()
    with _connection(tmp_path) as conn:
        _insert_document_authority(conn)
        jobs.enqueue(conn, request=_request(), now=NOW)
        claimed = jobs.claim_due(
            conn, worker_id="worker-a", now=NOW, lease_seconds=60
        )[0]
        event = _current_upsert_event(event_id="document-event-1")

        conn.execute("BEGIN IMMEDIATE")
        assert outbox.enqueue(conn, event)
        jobs.checkpoint_stage_in_transaction(
            conn,
            job_id=claimed.job_id,
            worker_id="worker-a",
            lease_token=claimed.lease_token or "",
            stage=DocumentIngestJobStage.CHUNKED,
            now="2026-08-19T12:00:05+00:00",
            document_id="doc-1",
            document_version_id="docv-1",
            retrieval_data_version="retrieval-v1",
            outbox_event_ids=(event.event_id,),
        )
        conn.rollback()

        rolled_back = jobs.get(conn, claimed.job_id)
        assert rolled_back is not None
        assert rolled_back.stage is DocumentIngestJobStage.PARSING
        assert outbox.get_status(conn, event.event_id) is None
        assert jobs.list_outbox_event_ids(conn, claimed.job_id) == ()

        conn.execute("BEGIN IMMEDIATE")
        assert outbox.enqueue(conn, event)
        jobs.checkpoint_stage_in_transaction(
            conn,
            job_id=claimed.job_id,
            worker_id="worker-a",
            lease_token=claimed.lease_token or "",
            stage=DocumentIngestJobStage.CHUNKED,
            now="2026-08-19T12:00:06+00:00",
            document_id="doc-1",
            document_version_id="docv-1",
            retrieval_data_version="retrieval-v1",
            outbox_event_ids=(event.event_id,),
        )
        conn.commit()

        assert jobs.list_outbox_event_ids(conn, claimed.job_id) == (event.event_id,)
        chunked = jobs.get(conn, claimed.job_id)
        assert chunked is not None
        assert chunked.stage is DocumentIngestJobStage.CHUNKED
        assert chunked.document_id == "doc-1"
        assert chunked.document_version_id == "docv-1"
        assert chunked.retrieval_data_version == "retrieval-v1"
        jobs.checkpoint_stage(
            conn,
            job_id=claimed.job_id,
            worker_id="worker-a",
            lease_token=claimed.lease_token or "",
            stage=DocumentIngestJobStage.INDEXING,
            now="2026-08-19T12:00:07+00:00",
        )

        # 同一处理身份复用原作业，不为第二个请求复制已冻结的事件/租约。
        replayed = jobs.enqueue(
            conn, request=_request("another-caller-id"),
            now="2026-08-19T12:00:08+00:00",
        )
        assert replayed.replayed is True
        assert replayed.job.job_id == claimed.job_id
        assert jobs.get(conn, "another-caller-id") is None
        assert jobs.list_outbox_event_ids(conn, replayed.job.job_id) == (event.event_id,)


def test_coverage_ready_requires_order_applied_events_and_exact_durable_proof(tmp_path):
    jobs = SqliteDocumentIngestJobStore(
        index_audit_port=SqliteDocumentIndexPort(),
    )
    outbox = SqliteRetrievalOutbox()
    with _connection(tmp_path) as conn:
        _insert_document_authority(conn)
        jobs.enqueue(conn, request=_request(), now=NOW)
        claimed = jobs.claim_due(
            conn, worker_id="worker-a", now=NOW, lease_seconds=120
        )[0]
        event = _current_upsert_event()
        conn.execute("BEGIN IMMEDIATE")
        assert outbox.enqueue(conn, event)
        conn.commit()

        with pytest.raises(DocumentIngestJobTransitionError, match="cannot skip"):
            jobs.checkpoint_stage(
                conn,
                job_id=claimed.job_id,
                worker_id="worker-a",
                lease_token=claimed.lease_token or "",
                stage=DocumentIngestJobStage.INDEXING,
                now="2026-08-19T12:00:01+00:00",
                document_id="doc-1",
                document_version_id="docv-1",
                retrieval_data_version="retrieval-v1",
                outbox_event_ids=(event.event_id,),
            )

        jobs.checkpoint_stage(
            conn,
            job_id=claimed.job_id,
            worker_id="worker-a",
            lease_token=claimed.lease_token or "",
            stage=DocumentIngestJobStage.CHUNKED,
            now="2026-08-19T12:00:02+00:00",
            document_id="doc-1",
            document_version_id="docv-1",
            retrieval_data_version="retrieval-v1",
            outbox_event_ids=(event.event_id,),
        )
        jobs.checkpoint_stage(
            conn,
            job_id=claimed.job_id,
            worker_id="worker-a",
            lease_token=claimed.lease_token or "",
            stage=DocumentIngestJobStage.INDEXING,
            now="2026-08-19T12:00:03+00:00",
        )

        with pytest.raises(DocumentIngestJobTransitionError, match="Outbox events.*applied"):
            jobs.checkpoint_stage(
                conn,
                job_id=claimed.job_id,
                worker_id="worker-a",
                lease_token=claimed.lease_token or "",
                stage=DocumentIngestJobStage.COVERAGE_READY,
                now="2026-08-19T12:00:04+00:00",
                coverage_proof=_coverage_proof(event),
            )

        assert outbox.claim_due(
            conn,
            worker_id="indexer",
            now="2026-08-19T12:00:04+00:00",
            lease_seconds=30,
            limit=10,
        ) == [event]
        outbox.mark_applied(
            conn,
            event_id=event.event_id,
            worker_id="indexer",
            now="2026-08-19T12:00:05+00:00",
        )
        conn.commit()

        wrong_proof = DocumentIngestCoverageProof(
            retrieval_data_version_id="retrieval-v1",
            retrieval_data_version_fingerprint="generation-fingerprint-v1",
            covered_binding_digest="f" * 64,
            covered_binding_count=1,
        )
        with pytest.raises(DocumentIngestJobTransitionError, match="coverage proof"):
            jobs.checkpoint_stage(
                conn,
                job_id=claimed.job_id,
                worker_id="worker-a",
                lease_token=claimed.lease_token or "",
                stage=DocumentIngestJobStage.COVERAGE_READY,
                now="2026-08-19T12:00:07+00:00",
                coverage_proof=wrong_proof,
            )

        ready = jobs.checkpoint_stage(
            conn,
            job_id=claimed.job_id,
            worker_id="worker-a",
            lease_token=claimed.lease_token or "",
            stage=DocumentIngestJobStage.COVERAGE_READY,
            now="2026-08-19T12:00:07+00:00",
            coverage_proof=_coverage_proof(event),
        )
        assert ready.coverage_expected_bindings == 1
        assert ready.coverage_covered_bindings == 1
        assert ready.coverage_mapped_events == 1
        assert ready.coverage_applied_events == 1
        assert ready.coverage_checked_at == "2026-08-19T12:00:07.000000+00:00"
        assert jobs.checkpoint_stage(
            conn,
            job_id=claimed.job_id,
            worker_id="worker-a",
            lease_token=claimed.lease_token or "",
            stage=DocumentIngestJobStage.COVERAGE_READY,
            now="2026-08-19T12:00:08+00:00",
            coverage_proof=_coverage_proof(event),
        ) == ready

        # 带类型的 document-v3 Unit 属于 Project；Session 挂载不参与该
        # Project 检索绑定的完整性证明。
        applied = jobs.mark_applied(
            conn,
            job_id=claimed.job_id,
            worker_id="worker-a",
            lease_token=claimed.lease_token or "",
            now="2026-08-19T12:00:09+00:00",
        )
        assert applied.stage is DocumentIngestJobStage.ACTIVE
        assert applied.status is DocumentIngestJobStatus.APPLIED


def test_coverage_proof_is_bound_to_the_jobs_exact_retrieval_data_version(tmp_path):
    jobs = SqliteDocumentIngestJobStore(
        index_audit_port=SqliteDocumentIndexPort(),
    )
    outbox = SqliteRetrievalOutbox()
    with _connection(tmp_path) as conn:
        _insert_document_authority(conn)
        jobs.enqueue(conn, request=_request(), now=NOW)
        claimed = jobs.claim_due(
            conn, worker_id="worker-a", now=NOW, lease_seconds=120
        )[0]
        event = _current_upsert_event()
        conn.execute("BEGIN IMMEDIATE")
        assert outbox.enqueue(conn, event)
        conn.commit()
        jobs.checkpoint_stage(
            conn,
            job_id=claimed.job_id,
            worker_id="worker-a",
            lease_token=claimed.lease_token or "",
            stage=DocumentIngestJobStage.CHUNKED,
            now="2026-08-19T12:00:01+00:00",
            document_id="doc-1",
            document_version_id="docv-1",
            retrieval_data_version="retrieval-v1",
            outbox_event_ids=(event.event_id,),
        )
        jobs.checkpoint_stage(
            conn,
            job_id=claimed.job_id,
            worker_id="worker-a",
            lease_token=claimed.lease_token or "",
            stage=DocumentIngestJobStage.INDEXING,
            now="2026-08-19T12:00:02+00:00",
        )
        assert outbox.claim_due(
            conn,
            worker_id="indexer",
            now="2026-08-19T12:00:03+00:00",
            lease_seconds=30,
            limit=10,
        ) == [event]
        outbox.mark_applied(
            conn,
            event_id=event.event_id,
            worker_id="indexer",
            now="2026-08-19T12:00:04+00:00",
        )
        conn.commit()

        proof = _coverage_proof(event)
        wrong_generation = DocumentIngestCoverageProof(
            retrieval_data_version_id="retrieval-v2",
            retrieval_data_version_fingerprint=proof.retrieval_data_version_fingerprint,
            covered_binding_digest=proof.covered_binding_digest,
            covered_binding_count=proof.covered_binding_count,
        )
        with pytest.raises(DocumentIngestJobTransitionError, match="data version"):
            jobs.checkpoint_stage(
                conn,
                job_id=claimed.job_id,
                worker_id="worker-a",
                lease_token=claimed.lease_token or "",
                stage=DocumentIngestJobStage.COVERAGE_READY,
                now="2026-08-19T12:00:05+00:00",
                coverage_proof=wrong_generation,
            )
        wrong_fingerprint = DocumentIngestCoverageProof(
            retrieval_data_version_id=proof.retrieval_data_version_id,
            retrieval_data_version_fingerprint="different-generation-recipe",
            covered_binding_digest=proof.covered_binding_digest,
            covered_binding_count=proof.covered_binding_count,
        )
        with pytest.raises(DocumentIngestJobTransitionError, match="fingerprint"):
            jobs.checkpoint_stage(
                conn, job_id=claimed.job_id, worker_id="worker-a",
                lease_token=claimed.lease_token or "",
                stage=DocumentIngestJobStage.COVERAGE_READY,
                now="2026-08-19T12:00:05+00:00", coverage_proof=wrong_fingerprint,
            )


def test_mark_applied_rechecks_the_current_indexed_content_hash(tmp_path):
    jobs = SqliteDocumentIngestJobStore(
        index_audit_port=SqliteDocumentIndexPort(),
    )
    outbox = SqliteRetrievalOutbox()
    with _connection(tmp_path) as conn:
        _insert_document_authority(conn)
        jobs.enqueue(conn, request=_request(), now=NOW)
        claimed = jobs.claim_due(
            conn, worker_id="worker-a", now=NOW, lease_seconds=120
        )[0]
        event = _current_upsert_event()
        conn.execute("BEGIN IMMEDIATE")
        assert outbox.enqueue(conn, event)
        conn.commit()
        jobs.checkpoint_stage(
            conn,
            job_id=claimed.job_id,
            worker_id="worker-a",
            lease_token=claimed.lease_token or "",
            stage=DocumentIngestJobStage.CHUNKED,
            now="2026-08-19T12:00:01+00:00",
            document_id="doc-1",
            document_version_id="docv-1",
            retrieval_data_version="retrieval-v1",
            outbox_event_ids=(event.event_id,),
        )
        jobs.checkpoint_stage(
            conn,
            job_id=claimed.job_id,
            worker_id="worker-a",
            lease_token=claimed.lease_token or "",
            stage=DocumentIngestJobStage.INDEXING,
            now="2026-08-19T12:00:02+00:00",
        )
        assert outbox.claim_due(
            conn,
            worker_id="indexer",
            now="2026-08-19T12:00:03+00:00",
            lease_seconds=30,
            limit=10,
        ) == [event]
        outbox.mark_applied(
            conn,
            event_id=event.event_id,
            worker_id="indexer",
            now="2026-08-19T12:00:04+00:00",
        )
        conn.commit()
        jobs.checkpoint_stage(
            conn,
            job_id=claimed.job_id,
            worker_id="worker-a",
            lease_token=claimed.lease_token or "",
            stage=DocumentIngestJobStage.COVERAGE_READY,
            now="2026-08-19T12:00:05+00:00",
            coverage_proof=_coverage_proof(event),
        )

        conn.execute(
            "UPDATE doc_chunks SET content='changed without changing the identity' "
            "WHERE doc_id='doc-1' AND source_version_id='docv-1'"
        )
        conn.commit()
        with pytest.raises(DocumentIngestJobTransitionError, match="no longer current"):
            jobs.mark_applied(
                conn,
                job_id=claimed.job_id,
                worker_id="worker-a",
                lease_token=claimed.lease_token or "",
                now="2026-08-19T12:00:06+00:00",
            )


def test_coverage_proof_rejects_an_empty_document_binding_set():
    with pytest.raises(ValueError, match="covered_binding_count"):
        DocumentIngestCoverageProof(
            retrieval_data_version_id="retrieval-v1",
            retrieval_data_version_fingerprint="generation-fingerprint-v1",
            covered_binding_digest=document_ingest_binding_digest(()),
            covered_binding_count=0,
        )


def test_current_upsert_mapping_requires_a_real_chunk_identity_and_hash(tmp_path):
    jobs = SqliteDocumentIngestJobStore(
        index_audit_port=SqliteDocumentIndexPort(),
    )
    outbox = SqliteRetrievalOutbox()
    with _connection(tmp_path) as conn:
        _insert_document_authority(conn)
        jobs.enqueue(conn, request=_request(), now=NOW)
        claimed = jobs.claim_due(
            conn, worker_id="worker-a", now=NOW, lease_seconds=60
        )[0]
        forged = RetrievalUpdateEvent(
            event_id="forged-current-upsert",
            kind=RetrievalUpdateKind.UPSERT,
            ref=SourceUnitRef(
                SourceType.DOCUMENT,
                "document-v3:doc-1:not-a-real-chunk",
                "docv-1",
                "f" * 64,
            ),
            retrieval_data_version="retrieval-v1",
            occurred_at=NOW,
        )
        conn.execute("BEGIN IMMEDIATE")
        assert outbox.enqueue(conn, forged)
        conn.commit()

        with pytest.raises(DocumentIngestJobTransitionError, match="real current chunk"):
            jobs.checkpoint_stage(
                conn,
                job_id=claimed.job_id,
                worker_id="worker-a",
                lease_token=claimed.lease_token or "",
                stage=DocumentIngestJobStage.CHUNKED,
                now="2026-08-19T12:00:05+00:00",
                document_id="doc-1",
                document_version_id="docv-1",
                retrieval_data_version="retrieval-v1",
                outbox_event_ids=(forged.event_id,),
            )
        assert jobs.list_outbox_event_ids(conn, claimed.job_id) == ()
        assert jobs.get(conn, claimed.job_id).stage is DocumentIngestJobStage.PARSING


def test_database_constraints_reject_stage_skips_missing_refs_and_ref_rebinding(tmp_path):
    jobs = SqliteDocumentIngestJobStore(
        index_audit_port=SqliteDocumentIndexPort(),
    )
    with _connection(tmp_path) as conn:
        _insert_document_authority(conn)
        jobs.enqueue(conn, request=_request(), now=NOW)
        claimed = jobs.claim_due(
            conn, worker_id="worker-a", now=NOW, lease_seconds=60
        )[0]

        with pytest.raises(sqlite3.IntegrityError, match="advance exactly once"):
            conn.execute(
                "UPDATE document_ingest_jobs SET stage='indexing', document_id='doc-1', "
                "document_version_id='docv-1', retrieval_data_version='retrieval-v1' "
                "WHERE job_id='ingest-1'"
            )
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "UPDATE document_ingest_jobs SET stage='chunked' WHERE job_id='ingest-1'"
            )
        conn.rollback()

        jobs.checkpoint_stage(
            conn,
            job_id=claimed.job_id,
            worker_id="worker-a",
            lease_token=claimed.lease_token or "",
            stage=DocumentIngestJobStage.CHUNKED,
            now="2026-08-19T12:00:05+00:00",
            document_id="doc-1",
            document_version_id="docv-1",
            retrieval_data_version="retrieval-v1",
        )
        with pytest.raises(sqlite3.IntegrityError, match="references are immutable"):
            conn.execute(
                "UPDATE document_ingest_jobs SET document_id='doc-2' "
                "WHERE job_id='ingest-1'"
            )


def test_stage_and_reference_regressions_fail_closed(tmp_path):
    with _connection(tmp_path) as conn:
        jobs, claimed = _enqueue_and_claim(conn)
        _insert_document_authority(conn)
        jobs.checkpoint_stage(
            conn,
            job_id=claimed.job_id,
            worker_id="worker-a",
            lease_token=claimed.lease_token or "",
            stage=DocumentIngestJobStage.CHUNKED,
            now="2026-08-19T12:00:05+00:00",
            document_id="doc-1",
            document_version_id="docv-1",
            retrieval_data_version="retrieval-v1",
        )
        jobs.checkpoint_stage(
            conn,
            job_id=claimed.job_id,
            worker_id="worker-a",
            lease_token=claimed.lease_token or "",
            stage=DocumentIngestJobStage.INDEXING,
            now="2026-08-19T12:00:06+00:00",
        )

        with pytest.raises(DocumentIngestJobTransitionError, match="regress"):
            jobs.checkpoint_stage(
                conn,
                job_id=claimed.job_id,
                worker_id="worker-a",
                lease_token=claimed.lease_token or "",
                stage=DocumentIngestJobStage.CHUNKED,
                now="2026-08-19T12:00:07+00:00",
            )
        with pytest.raises(DocumentIngestJobTransitionError, match="document_id"):
            jobs.checkpoint_stage(
                conn,
                job_id=claimed.job_id,
                worker_id="worker-a",
                lease_token=claimed.lease_token or "",
                stage=DocumentIngestJobStage.INDEXING,
                now="2026-08-19T12:00:07+00:00",
                document_id="doc-2",
            )


def test_checkpoint_rejects_a_version_owned_by_another_document(tmp_path):
    with _connection(tmp_path) as conn:
        _insert_document_authority(conn, doc_id="doc-1", version_id="docv-1")
        _insert_document_authority(conn, doc_id="doc-2", version_id="docv-2")
        jobs, claimed = _enqueue_and_claim(conn)

        with pytest.raises(DocumentIngestJobTransitionError, match="does not belong"):
            jobs.checkpoint_stage(
                conn,
                job_id=claimed.job_id,
                worker_id="worker-a",
                lease_token=claimed.lease_token or "",
                stage=DocumentIngestJobStage.CHUNKED,
                now="2026-08-19T12:00:05+00:00",
                document_id="doc-1",
                document_version_id="docv-2",
                retrieval_data_version="retrieval-v1",
            )

        unchanged = jobs.get(conn, claimed.job_id)
        assert unchanged is not None
        assert unchanged.stage is DocumentIngestJobStage.PARSING
        assert unchanged.document_id is None


def test_checkpoint_reuses_document_content_with_an_earlier_mtime_observation(tmp_path):
    with _connection(tmp_path) as conn:
        _insert_document_authority(conn, source_mtime_ns=123456788)
        jobs, claimed = _enqueue_and_claim(conn)

        result = jobs.checkpoint_stage(
            conn,
            job_id=claimed.job_id,
            worker_id="worker-a",
            lease_token=claimed.lease_token or "",
            stage=DocumentIngestJobStage.CHUNKED,
            now="2026-08-19T12:00:05+00:00",
            document_id="doc-1",
            document_version_id="docv-1",
            retrieval_data_version="retrieval-v1",
        )

        assert result.stage is DocumentIngestJobStage.CHUNKED
        assert result.file_version_id == "file-docv-1"
        assert jobs.document_target_is_current(conn, result)
        assert conn.execute(
            "SELECT source_mtime_ns FROM document_versions WHERE id='docv-1'"
        ).fetchone()[0] == 123456788

        for number, source_hash in ((2, "b" * 64), (3, "a" * 64)):
            conn.execute(
                "INSERT INTO file_versions (id, file_id, version_number, producer, "
                "content_sha256, size_bytes, source_mtime_ns, created_at) "
                "VALUES (?, 'file-doc-1', ?, 'workspace_existing', ?, 1024, ?, ?)",
                (f"file-revision-{number}", number, source_hash, 123456789 + number, NOW),
            )
        conn.execute(
            "UPDATE files SET current_version_id='file-revision-3' WHERE id='file-doc-1'"
        )
        conn.commit()
        assert not jobs.document_target_is_current(conn, result)
        with pytest.raises(DocumentIngestJobTransitionError, match="frozen source/recipe"):
            jobs.checkpoint_stage(
                conn,
                job_id=claimed.job_id,
                worker_id="worker-a",
                lease_token=claimed.lease_token or "",
                stage=DocumentIngestJobStage.INDEXING,
                now="2026-08-19T12:00:06+00:00",
            )


def test_checkpoint_rejects_authority_built_from_a_different_source_or_recipe(tmp_path):
    with _connection(tmp_path) as conn:
        _insert_document_authority(
            conn,
            processor_fingerprint="docling@changed",
        )
        jobs, claimed = _enqueue_and_claim(conn)

        with pytest.raises(DocumentIngestJobTransitionError, match="frozen source/recipe"):
            jobs.checkpoint_stage(
                conn,
                job_id=claimed.job_id,
                worker_id="worker-a",
                lease_token=claimed.lease_token or "",
                stage=DocumentIngestJobStage.CHUNKED,
                now="2026-08-19T12:00:05+00:00",
                document_id="doc-1",
                document_version_id="docv-1",
                retrieval_data_version="retrieval-v1",
            )

        unchanged = jobs.get(conn, claimed.job_id)
        assert unchanged is not None
        assert unchanged.stage is DocumentIngestJobStage.PARSING


def test_outbox_mapping_accepts_old_revision_cleanup_but_rejects_another_document(tmp_path):
    jobs = SqliteDocumentIngestJobStore(
        index_audit_port=SqliteDocumentIndexPort(),
    )
    outbox = SqliteRetrievalOutbox()
    with _connection(tmp_path) as conn:
        _insert_document_authority(conn, doc_id="doc-1", version_id="docv-old")
        _insert_document_authority(
            conn,
            doc_id="doc-1",
            version_id="docv-new",
            version_number=2,
        )
        _insert_document_authority(conn, doc_id="doc-2", version_id="docv-other")
        jobs.enqueue(conn, request=_request(file_version_id="file-docv-new"), now=NOW)
        claimed = jobs.claim_due(
            conn, worker_id="worker-a", now=NOW, lease_seconds=60
        )[0]

        old_cleanup = RetrievalUpdateEvent(
            event_id="old-cleanup",
            kind=RetrievalUpdateKind.PURGE,
            ref=SourceUnitRef(
                SourceType.DOCUMENT,
                "document-v3:doc-1:old-chunk",
                "docv-old",
                "1" * 64,
            ),
            retrieval_data_version="retrieval-v1",
            occurred_at=NOW,
        )
        unrelated = RetrievalUpdateEvent(
            event_id="unrelated-document",
            kind=RetrievalUpdateKind.PURGE,
            ref=SourceUnitRef(
                SourceType.DOCUMENT,
                "document-v3:doc-2:chunk",
                "docv-other",
                "2" * 64,
            ),
            retrieval_data_version="retrieval-v1",
            occurred_at=NOW,
        )
        conn.execute("BEGIN IMMEDIATE")
        assert outbox.enqueue(conn, old_cleanup)
        assert outbox.enqueue(conn, unrelated)
        conn.commit()

        jobs.checkpoint_stage(
            conn,
            job_id=claimed.job_id,
            worker_id="worker-a",
            lease_token=claimed.lease_token or "",
            stage=DocumentIngestJobStage.CHUNKED,
            now="2026-08-19T12:00:05+00:00",
            document_id="doc-1",
            document_version_id="docv-new",
            retrieval_data_version="retrieval-v1",
            outbox_event_ids=(old_cleanup.event_id,),
        )
        assert jobs.list_outbox_event_ids(conn, claimed.job_id) == (old_cleanup.event_id,)

        jobs.checkpoint_stage(
            conn,
            job_id=claimed.job_id,
            worker_id="worker-a",
            lease_token=claimed.lease_token or "",
            stage=DocumentIngestJobStage.INDEXING,
            now="2026-08-19T12:00:06+00:00",
        )

        with pytest.raises(DocumentIngestJobTransitionError, match="same document"):
            jobs.checkpoint_stage(
                conn,
                job_id=claimed.job_id,
                worker_id="worker-a",
                lease_token=claimed.lease_token or "",
                stage=DocumentIngestJobStage.INDEXING,
                now="2026-08-19T12:00:07+00:00",
                outbox_event_ids=(unrelated.event_id,),
            )
        assert jobs.list_outbox_event_ids(conn, claimed.job_id) == (old_cleanup.event_id,)


def test_get_list_and_missing_or_invalid_retries_are_bounded(tmp_path):
    jobs = SqliteDocumentIngestJobStore(
        index_audit_port=SqliteDocumentIndexPort(),
    )
    with _connection(tmp_path) as conn:
        jobs.enqueue(conn, request=_request("ingest-1"), now=NOW)
        jobs.enqueue(
            conn,
            request=_request("ingest-2", target_generation_id="retrieval-v2"),
            now="2026-08-19T12:00:01+00:00",
        )

        assert jobs.get(conn, "missing") is None
        assert [job.job_id for job in jobs.list(conn)] == ["ingest-2", "ingest-1"]
        assert [job.job_id for job in jobs.list(
            conn, statuses=(DocumentIngestJobStatus.PENDING,), limit=1
        )] == ["ingest-2"]

        with pytest.raises(KeyError, match="not found"):
            jobs.retry_terminal_failure(conn, job_id="missing", now=NOW)
        with pytest.raises(DocumentIngestJobTransitionError, match="terminal_failed"):
            jobs.retry_terminal_failure(conn, job_id="ingest-1", now=NOW)
