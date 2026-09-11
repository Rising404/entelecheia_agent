from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from personagraph.api import (
    router,
    service,
    workspace_document_ingest,
)
from personagraph.workspace.ingestion.storage import (
    FilePreparationDeliveryStatus,
    SqliteDocumentIngestJobStore,
    SqliteFilePreparationRequestStore,
)
from personagraph.workspace.storage.context import connect_current
from personagraph.retrieval.contracts import SourceType, SourceUnitRef
from personagraph.retrieval.lifecycle.outbox import (
    OutboxStatus,
    RetrievalUpdateEvent,
    RetrievalUpdateKind,
    SqliteRetrievalOutbox,
)
from personagraph.session import store as session_store


pytestmark = pytest.mark.usefixtures("partitioned_project_state")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _route_payload(method: str, target: str, body: dict) -> dict:
    return router.dispatch_response(method, target, body).payload


def _session() -> tuple[str, Path]:
    session = _route_payload(
        "POST",
        "/api/sessions",
        {},
    )["session"]
    return str(session["id"]), Path(session["working_dir"])


def _register_retrieval_version(conn, *, version_id: str, now: str) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO retrieval_data_versions "
        "(id, fingerprint, role, state, created_at, activated_at) "
        "VALUES (?, ?, 'staging', 'building', ?, NULL)",
        (version_id, f"fixture:{version_id}", now),
    )


def test_enqueue_is_a_202_durable_job_and_does_not_parse_inline():
    session_id, workspace = _session()
    paper = workspace / "paper.md"
    paper.write_text("# Method\n\nEvidence.", encoding="utf-8")

    response = router.dispatch_response(
        "POST",
        "/api/document-ingest-jobs",
        {
            "job_id": "ingest-job-1",
            "session_id": session_id,
            "path": "paper.md",
            "with_summary": False,
        },
    )

    assert response.status == 202
    assert response.payload["accepted"] is True
    assert response.payload["replayed"] is False
    assert response.payload["job"]["job_id"] == "ingest-job-1"
    assert response.payload["job"]["processing_job_id"].startswith("file-processing:")
    assert response.payload["job"]["processing_job_id"] != "ingest-job-1"
    assert response.payload["job"]["status"] == "queued"
    assert response.payload["job"]["stage"] == "parsing"
    assert response.payload["job"]["path"] == "paper.md"
    assert "lease_token" not in response.payload["job"]
    assert "canonical_path" not in response.payload["job"]
    assert _route_payload(
        "GET",
        f"/api/documents?session_id={session_id}",
        {},
    )["documents"] == []

    listed = _route_payload(
        "GET",
        f"/api/document-ingest-jobs?session_id={session_id}",
        {},
    )
    assert [job["job_id"] for job in listed["jobs"]] == ["ingest-job-1"]
    detail = _route_payload(
        "GET",
        f"/api/document-ingest-jobs/ingest-job-1?session_id={session_id}",
        {},
    )
    assert detail["job"] == response.payload["job"]


def test_agent_private_workspace_paths_are_rejected_before_enqueue():
    session_id, workspace = _session()
    private_file = (
        workspace / ".personagraph" / "output" / session_id / "draft.md"
    )
    private_file.parent.mkdir(parents=True, exist_ok=True)
    private_file.write_text("host-private draft", encoding="utf-8")

    # 即使调用方已经解析文件系统路径，较低层的应用边界也不得解析或持久化通用任务。
    with session_store.session_database_scope(session_id):
        durable = workspace_document_ingest.enqueue_document_ingest_job(
            job_id="private-direct-job",
            path=str(private_file),
            session_id=session_id,
            with_summary=False,
        )
        assert durable == {"ok": False, "reason": "agent_private_path"}
        assert (
            workspace_document_ingest.get_document_ingest_job(
                request_id="private-direct-job",
                session_id=session_id,
            )
            is None
        )

    # 公开 HTTP 路由会保留稳定且带类型的路径策略响应。
    with pytest.raises(service.ApiError) as durable_http:
        _route_payload(
            "POST",
            "/api/document-ingest-jobs",
            {
                "job_id": "private-http-job",
                "session_id": session_id,
                "path": ".personagraph/output/%s/draft.md" % session_id,
            },
        )
    assert durable_http.value.status == 403
    assert durable_http.value.code == "AGENT_PRIVATE_PATH"
    assert _route_payload(
        "GET",
        f"/api/document-ingest-jobs?session_id={session_id}",
        {},
    ) == {"jobs": []}


def test_synchronous_document_ingest_route_is_retired():
    session_id, workspace = _session()
    source = workspace / "paper.md"
    source.write_text("retired synchronous intake", encoding="utf-8")

    with pytest.raises(service.ApiError) as error:
        _route_payload(
            "POST",
            "/api/documents",
            {"path": "paper.md", "session_id": session_id},
        )

    assert error.value.status == 404
    assert error.value.code == "NOT_FOUND"


def test_enqueue_exact_replay_is_idempotent_and_payload_collision_is_rejected():
    session_id, workspace = _session()
    first = workspace / "first.md"
    second = workspace / "second.md"
    first.write_text("first", encoding="utf-8")
    second.write_text("second", encoding="utf-8")
    body = {
        "job_id": "stable-client-operation",
        "session_id": session_id,
        "path": "first.md",
        "with_summary": False,
    }

    first_response = router.dispatch_response(
        "POST", "/api/document-ingest-jobs", body
    )
    replay = router.dispatch_response(
        "POST", "/api/document-ingest-jobs", body
    )

    assert first_response.status == replay.status == 202
    assert replay.payload["replayed"] is True
    assert replay.payload["job"] == first_response.payload["job"]

    # 外部来源状态由首次接受的操作冻结。同一客户端命令因响应丢失而重试时，
    # 不得依据更新后的字节重新解释并转化为 ID 冲突。
    first.write_text("changed after the accepted response", encoding="utf-8")
    replay_after_source_change = router.dispatch_response(
        "POST", "/api/document-ingest-jobs", body
    )
    assert replay_after_source_change.status == 202
    assert replay_after_source_change.payload["replayed"] is True
    assert replay_after_source_change.payload["job"] == first_response.payload["job"]

    with pytest.raises(service.ApiError) as collision:
        router.dispatch_response(
            "POST",
            "/api/document-ingest-jobs",
            {**body, "path": "second.md"},
        )
    assert collision.value.status == 409
    assert collision.value.code == "DOCUMENT_INGEST_JOB_ID_COLLISION"


def test_job_reads_and_retry_are_exactly_session_scoped():
    first_session, first_workspace = _session()
    second_session, _second_workspace = _session()
    paper = first_workspace / "paper.md"
    paper.write_text("paper", encoding="utf-8")
    router.dispatch_response(
        "POST",
        "/api/document-ingest-jobs",
        {
            "job_id": "owned-job",
            "session_id": first_session,
            "path": "paper.md",
        },
    )

    assert _route_payload(
        "GET",
        f"/api/document-ingest-jobs?session_id={second_session}",
        {},
    ) == {"jobs": []}
    with pytest.raises(service.ApiError) as hidden:
        _route_payload(
            "GET",
            f"/api/document-ingest-jobs/owned-job?session_id={second_session}",
            {},
        )
    assert hidden.value.status == 404

    retry = router.dispatch_response(
        "POST",
        "/api/document-ingest-jobs/owned-job/retry",
        {"session_id": first_session},
    )
    assert retry.status == 202
    assert retry.payload["job"]["job_id"] == "owned-job"


def test_sessions_share_processing_job_but_not_request_authority():
    first_session, workspace = _session()
    second_session = session_store.create_session(
        "Entelecheia",
        title="second request owner",
        working_dir=str(workspace),
        require_workspace_read_authority=True,
    )
    paper = workspace / "shared.md"
    paper.write_text("same project file", encoding="utf-8")

    first_response = router.dispatch_response(
        "POST",
        "/api/document-ingest-jobs",
        {
            "job_id": "shared-request-a",
            "session_id": first_session,
            "path": "shared.md",
            "with_summary": False,
        },
    )
    second_response = router.dispatch_response(
        "POST",
        "/api/document-ingest-jobs",
        {
            "job_id": "shared-request-b",
            "session_id": second_session,
            "path": "shared.md",
            "with_summary": True,
        },
    )
    first = first_response.payload["job"]
    second = second_response.payload["job"]

    assert first_response.payload["replayed"] is False
    assert second_response.payload["replayed"] is False
    assert first["job_id"] == "shared-request-a"
    assert second["job_id"] == "shared-request-b"
    assert first["processing_job_id"] == second["processing_job_id"]
    assert first["session_id"] == first_session
    assert second["session_id"] == second_session
    assert first["with_summary"] is False
    assert second["with_summary"] is True
    assert [job["job_id"] for job in _route_payload(
        "GET",
        f"/api/document-ingest-jobs?session_id={first_session}",
        {},
    )["jobs"]] == ["shared-request-a"]
    assert [job["job_id"] for job in _route_payload(
        "GET",
        f"/api/document-ingest-jobs?session_id={second_session}",
        {},
    )["jobs"]] == ["shared-request-b"]

    for hidden_id in (first["job_id"], first["processing_job_id"]):
        with pytest.raises(service.ApiError) as hidden:
            _route_payload(
                "GET",
                f"/api/document-ingest-jobs/{hidden_id}?session_id={second_session}",
                {},
            )
        assert hidden.value.status == 404

    with pytest.raises(service.ApiError) as shared_id_collision:
        router.dispatch_response(
            "POST",
            "/api/document-ingest-jobs",
            {
                "job_id": "shared-request-a",
                "session_id": second_session,
                "path": "shared.md",
            },
        )
    assert shared_id_collision.value.status == 409
    with pytest.raises(service.ApiError) as processing_id_retry:
        router.dispatch_response(
            "POST",
            f"/api/document-ingest-jobs/{first['processing_job_id']}/retry",
            {"session_id": second_session},
        )
    assert processing_id_retry.value.status == 404


def test_terminal_failure_projects_bounded_error_and_http_retry_requeues():
    session_id, workspace = _session()
    paper = workspace / "changed.md"
    paper.write_text("original", encoding="utf-8")
    accepted = router.dispatch_response(
        "POST",
        "/api/document-ingest-jobs",
        {"job_id": "terminal-job", "session_id": session_id, "path": "changed.md"},
    )
    processing_job_id = accepted.payload["job"]["processing_job_id"]

    operation_store = SqliteDocumentIngestJobStore()
    outbox = SqliteRetrievalOutbox()
    with session_store.session_database_scope(session_id), connect_current() as conn:
        now = _now()
        _register_retrieval_version(
            conn,
            version_id="retrieval-version-1",
            now=now,
        )
        event = RetrievalUpdateEvent(
            event_id="terminal-job-event",
            kind=RetrievalUpdateKind.UPSERT,
            ref=SourceUnitRef(
                SourceType.DOCUMENT,
                "session:document-v2:doc:chunk",
                "version-1",
                "a" * 64,
            ),
            retrieval_data_version="retrieval-version-1",
            occurred_at=now,
        )
        outbox.enqueue(conn, event)
        claimed_events = outbox.claim_due(
            conn,
            worker_id="retrieval-worker",
            now=now,
            lease_seconds=30,
            limit=1,
        )
        assert [claimed.event_id for claimed in claimed_events] == [event.event_id]
        outbox.mark_terminal_failure(
            conn,
            event_id=event.event_id,
            worker_id="retrieval-worker",
            now=now,
            reason_code="retrieval_method_unavailable",
        )
        conn.execute(
            "INSERT INTO document_ingest_job_events(job_id, event_id, recorded_at) "
            "VALUES (?, ?, ?)",
            (processing_job_id, event.event_id, now),
        )
        conn.commit()
        claimed = operation_store.claim_due(
            conn,
            worker_id="test-worker",
            now=now,
            lease_seconds=30,
            limit=1,
        )
        assert len(claimed) == 1
        operation_store.mark_terminal_failure(
            conn,
            job_id=processing_job_id,
            worker_id="test-worker",
            lease_token=claimed[0].lease_token or "",
            now=_now(),
            reason_code="frozen_source_mismatch",
        )
        SqliteFilePreparationRequestStore().mark_blocked(
            conn,
            "terminal-job",
            reason_code="frozen_source_mismatch",
            now=_now(),
        )

    failed = _route_payload(
        "GET",
        f"/api/document-ingest-jobs/terminal-job?session_id={session_id}",
        {},
    )["job"]
    assert failed["status"] == "failed"
    assert failed["can_retry"] is True
    assert failed["error"] == {
        "code": "frozen_source_mismatch",
        "message": "源文件在任务创建后发生了变化",
        "hint": "请重新确认当前文件仍受该会话授权且内容未变化，然后手动重试此请求。",
        "retryable": True,
    }
    assert "canonical_path" not in failed
    assert "lease_token" not in failed

    retried = router.dispatch_response(
        "POST",
        "/api/document-ingest-jobs/terminal-job/retry",
        {"session_id": session_id},
    )
    assert retried.status == 202
    assert retried.payload["replayed"] is False
    assert retried.payload["job"]["status"] == "queued"
    assert retried.payload["job"]["can_retry"] is False
    assert retried.payload["job"]["error"] is None
    assert retried.payload["job"]["reason_code"] is None
    with session_store.session_database_scope(session_id), connect_current() as conn:
        assert outbox.get_status(conn, "terminal-job-event") is OutboxStatus.PENDING
        actions = outbox.list_manual_actions(conn, event_id="terminal-job-event")
        assert (
            SqliteFilePreparationRequestStore()
            .get(conn, "terminal-job")
            .delivery_status
            is FilePreparationDeliveryStatus.PENDING
        )
    assert len(actions) == 1
    assert actions[0].actor == "document_ingest_job_api"


def test_retryable_failure_explains_automatic_retry_without_manual_action():
    session_id, workspace = _session()
    paper = workspace / "temporary.md"
    paper.write_text("paper", encoding="utf-8")
    accepted = router.dispatch_response(
        "POST",
        "/api/document-ingest-jobs",
        {"job_id": "automatic-job", "session_id": session_id, "path": "temporary.md"},
    )
    processing_job_id = accepted.payload["job"]["processing_job_id"]

    operation_store = SqliteDocumentIngestJobStore()
    with session_store.session_database_scope(session_id), connect_current() as conn:
        claimed = operation_store.claim_due(
            conn,
            worker_id="test-worker",
            now=_now(),
            lease_seconds=30,
            limit=1,
        )
        assert len(claimed) == 1
        operation_store.mark_retryable_failure(
            conn,
            job_id=processing_job_id,
            worker_id="test-worker",
            lease_token=claimed[0].lease_token or "",
            now=_now(),
            retry_after_seconds=60,
            reason_code="document_worker_internal_error",
        )

    failed = _route_payload(
        "GET",
        f"/api/document-ingest-jobs/automatic-job?session_id={session_id}",
        {},
    )["job"]
    assert failed["status"] == "failed"
    assert failed["can_retry"] is False
    assert failed["next_retry_at"] is not None
    assert failed["error"]["code"] == "document_worker_internal_error"
    assert failed["error"]["retryable"] is True
    assert "自动重试" in failed["error"]["hint"]
    assert _route_payload(
        "GET",
        f"/api/document-ingest-jobs?session_id={session_id}&status=failed",
        {},
    )["jobs"] == [failed]


def test_manual_retry_revalidates_frozen_source_before_mutation():
    session_id, workspace = _session()
    paper = workspace / "retry-source.md"
    paper.write_text("original", encoding="utf-8")
    accepted = router.dispatch_response(
        "POST",
        "/api/document-ingest-jobs",
        {
            "job_id": "retry-source-request",
            "session_id": session_id,
            "path": "retry-source.md",
        },
    ).payload["job"]
    processing_job_id = accepted["processing_job_id"]
    jobs = SqliteDocumentIngestJobStore()
    with session_store.session_database_scope(session_id), connect_current() as conn:
        claimed = jobs.claim_due(
            conn,
            worker_id="source-retry-worker",
            now=_now(),
            lease_seconds=30,
            limit=1,
        )[0]
        jobs.mark_terminal_failure(
            conn,
            job_id=processing_job_id,
            worker_id="source-retry-worker",
            lease_token=claimed.lease_token or "",
            now=_now(),
            reason_code="document_worker_internal_error",
        )
    paper.write_text("changed", encoding="utf-8")

    with pytest.raises(service.ApiError) as rejected:
        router.dispatch_response(
            "POST",
            "/api/document-ingest-jobs/retry-source-request/retry",
            {"session_id": session_id},
        )

    assert rejected.value.status == 409
    with session_store.session_database_scope(session_id), connect_current() as conn:
        assert jobs.get(conn, processing_job_id).status.value == "terminal_failed"


def test_manual_retry_requeues_request_events_and_job_in_one_transaction(
    monkeypatch,
):
    session_id, workspace = _session()
    paper = workspace / "terminal.md"
    paper.write_text("terminal", encoding="utf-8")
    accepted = router.dispatch_response(
        "POST",
        "/api/document-ingest-jobs",
        {"job_id": "atomic-retry-job", "session_id": session_id, "path": "terminal.md"},
    )
    processing_job_id = accepted.payload["job"]["processing_job_id"]
    jobs = SqliteDocumentIngestJobStore()
    outbox = SqliteRetrievalOutbox()
    event_ids = ("atomic-event-1", "atomic-event-2")
    with session_store.session_database_scope(session_id), connect_current() as conn:
        now = _now()
        _register_retrieval_version(
            conn,
            version_id="retrieval-version-1",
            now=now,
        )
        for index, event_id in enumerate(event_ids):
            outbox.enqueue(
                conn,
                RetrievalUpdateEvent(
                    event_id=event_id,
                    kind=RetrievalUpdateKind.UPSERT,
                    ref=SourceUnitRef(
                        SourceType.DOCUMENT,
                        f"session:document-v2:doc:chunk-{index}",
                        "version-1",
                        str(index + 1) * 64,
                    ),
                    retrieval_data_version="retrieval-version-1",
                    occurred_at=now,
                ),
            )
        claimed_events = outbox.claim_due(
            conn,
            worker_id="retrieval-worker",
            now=now,
            lease_seconds=30,
            limit=2,
        )
        for event in claimed_events:
            outbox.mark_terminal_failure(
                conn,
                event_id=event.event_id,
                worker_id="retrieval-worker",
                now=now,
                reason_code="retrieval_method_unavailable",
            )
            conn.execute(
                "INSERT INTO document_ingest_job_events(job_id, event_id, recorded_at) "
                "VALUES (?, ?, ?)",
                (processing_job_id, event.event_id, now),
            )
        conn.commit()
        claimed_job = jobs.claim_due(
            conn,
            worker_id="document-worker",
            now=now,
            lease_seconds=30,
            limit=1,
        )[0]
        jobs.mark_terminal_failure(
            conn,
            job_id=claimed_job.job_id,
            worker_id="document-worker",
            lease_token=claimed_job.lease_token or "",
            now=now,
            reason_code="retrieval_method_unavailable",
        )
        SqliteFilePreparationRequestStore().mark_blocked(
            conn,
            "atomic-retry-job",
            reason_code="retrieval_method_unavailable",
            now=now,
        )

    projected = _route_payload(
        "GET",
        f"/api/document-ingest-jobs/atomic-retry-job?session_id={session_id}",
        {},
    )["job"]
    assert projected["error"]["code"] == "retrieval_method_unavailable"
    assert "同一检索配置" in projected["error"]["hint"]

    original = SqliteFilePreparationRequestStore.retry_blocked_in_transaction

    def fail_after_request_write(self, conn, request_id, **kwargs):
        original(self, conn, request_id, **kwargs)
        raise RuntimeError("fault_after_request_retry")

    monkeypatch.setattr(
        SqliteFilePreparationRequestStore,
        "retry_blocked_in_transaction",
        fail_after_request_write,
    )
    with pytest.raises(RuntimeError, match="fault_after_request_retry"):
        with session_store.session_database_scope(session_id):
            workspace_document_ingest.retry_document_ingest_job(
                request_id="atomic-retry-job",
                session_id=session_id,
            )

    with session_store.session_database_scope(session_id), connect_current() as conn:
        assert jobs.get(conn, processing_job_id).status.value == "terminal_failed"
        assert (
            SqliteFilePreparationRequestStore()
            .get(conn, "atomic-retry-job")
            .delivery_status
            is FilePreparationDeliveryStatus.BLOCKED
        )
        assert [outbox.get_status(conn, event_id) for event_id in event_ids] == [
            OutboxStatus.TERMINAL_FAILED,
            OutboxStatus.TERMINAL_FAILED,
        ]
        assert outbox.list_manual_actions(conn) == ()
