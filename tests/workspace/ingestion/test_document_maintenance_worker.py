from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import os
from pathlib import Path
import sqlite3
import threading
import time

import pytest

from personagraph.api import workspace_documents
from personagraph.input_processing.files import fingerprint_file
from personagraph.input_processing.documents.readers import (
    configured_processor_fingerprint,
)
from personagraph.input_processing.documents.preparation import (
    DocumentPrepareFailure,
    PreparedDocumentIngest,
    prepare_document_path,
)
from personagraph.input_processing.documents import (
    ChunkSpan,
    DocumentChunk,
    DocumentLocator,
    ElementKind,
)
from personagraph.workspace.files import (
    FileSource,
    WorkspaceFileAuthority,
)
from personagraph.workspace.storage import DocumentDatabase
from personagraph.workspace.storage.context import bind as bind_documents
from personagraph.workspace.documents import application as docstore
from personagraph.workspace.ingestion.storage import (
    DocumentIngestJobRequest,
    DocumentIngestJobStage,
    DocumentIngestJobStatus,
    SqliteDocumentIngestJobStore,
    SqliteFilePreparationRequestStore,
    FilePreparationDeliveryStatus,
)
from personagraph.retrieval.profile import durable_chunking_profile
from personagraph.retrieval.contracts import (
    RetrievalStatus,
    SourceAvailability,
    RetrievalUnit,
    SourceFilter,
    SourceType,
    SourceUnitRef,
)
from personagraph.retrieval.lifecycle.generation import (
    DOCUMENT_HYBRID_INDEX_RECIPE_CONTRACT,
    file_corpus_generation_spec,
)
from personagraph.retrieval.lifecycle.outbox import (
    OutboxStatus,
    RetrievalUpdateEvent,
    RetrievalUpdateKind,
    SqliteRetrievalOutbox,
)
from personagraph.retrieval.sources.events import SqliteDocumentIndexPort
from personagraph.retrieval.sources.document import MountedDocumentChunkSourceAdapter
from personagraph.retrieval.sources.identity import (
    parse_mounted_document_chunk_source_unit_id,
)
from personagraph.retrieval.sqlite_store import (
    RetrievalDataVersionRole,
    RetrievalDataVersionState,
    SqliteRetrievalCatalog,
)
from personagraph.retrieval.lifecycle.sync import RetrievalOutboxConsumer, RetrievalSyncService
from personagraph.workspace.ingestion.lifecycle import DocumentMaintenanceWorkerLifecycle
from personagraph.workspace.ingestion.worker import DocumentMaintenanceWorker
from personagraph.workspace.ingestion.delivery import FilePreparationDelivery
from personagraph.retrieval.operations.ingestion_index import build_document_maintenance_worker
from personagraph.retrieval.operations.document_generation import DocumentGenerationAuthority
from personagraph.retrieval.sources.picture import PictureObservationSourceAdapter
from personagraph.session.workspace_authority import (
    validate_current_session_workspace_authority,
)
from personagraph.session import store as session_store
from tests.documents._authority import bound_project_document_authority
from personagraph.workspace.binding import is_reserved_workspace_path


class _Clock:
    def __init__(self) -> None:
        self._offset = timedelta(seconds=1)

    def __call__(self) -> str:
        return (datetime.now(timezone.utc) + self._offset).isoformat()

    def advance(self, seconds: int) -> None:
        self._offset += timedelta(seconds=seconds)


class _IndexingConsumer:
    """小型真实目录索引器；权威内容绝不会经过该替身。"""

    def __init__(
        self,
        *,
        outbox: SqliteRetrievalOutbox,
        catalog: SqliteRetrievalCatalog,
        skip_kinds: frozenset[RetrievalUpdateKind] = frozenset(),
        wrong_scope: bool = False,
        terminal_reason: str | None = None,
    ) -> None:
        self.outbox = outbox
        self.catalog = catalog
        self.skip_kinds = skip_kinds
        self.wrong_scope = wrong_scope
        self.terminal_reason = terminal_reason
        self.calls = 0

    def consume_due(
        self,
        conn: sqlite3.Connection,
        *,
        worker_id: str,
        now: str,
        lease_seconds: int = 30,
        limit: int = 20,
    ) -> tuple[object, ...]:
        self.calls += 1
        events = self.outbox.claim_due(
            conn,
            worker_id=worker_id,
            now=now,
            lease_seconds=lease_seconds,
            limit=limit,
        )
        for event in events:
            if self.terminal_reason is not None:
                self.outbox.mark_terminal_failure(
                    conn,
                    event_id=event.event_id,
                    worker_id=worker_id,
                    now=now,
                    reason_code=self.terminal_reason,
                )
                conn.commit()
                continue
            if event.kind in self.skip_kinds:
                continue
            if event.kind in {
                RetrievalUpdateKind.UPSERT,
                RetrievalUpdateKind.RESTORE,
            }:
                identity = parse_mounted_document_chunk_source_unit_id(
                    event.ref.source_unit_id
                )
                assert identity is not None
                if self.wrong_scope:
                    scope = {}
                elif identity.is_project_scoped:
                    scope = {"doc_id": identity.doc_id}
                else:
                    scope = {"session_id": identity.session_id}
                    if identity.doc_id is not None:
                        scope["doc_id"] = identity.doc_id
                stored = self.catalog.upsert_pending_unit(
                    RetrievalUnit(
                        ref=event.ref,
                        retrieval_data_version=event.retrieval_data_version,
                        retrieval_status=RetrievalStatus.ACTIVE,
                        source_filter=SourceFilter.from_mapping(
                            SourceType.DOCUMENT,
                            scope,
                        ),
                    )
                )
                self.catalog.mark_unit_index_ready(stored.unit_id)
            elif event.kind is RetrievalUpdateKind.PURGE:
                stored = self.catalog.get_unit(
                    event.ref,
                    event.retrieval_data_version,
                )
                if stored is not None:
                    self.catalog.delete_unit(stored.unit_id)
            self.outbox.mark_applied(
                conn,
                event_id=event.event_id,
                worker_id=worker_id,
                now=now,
            )
            conn.commit()
        return ()


class _NoopIndexWriter:
    def index(self, stored_unit, source_unit) -> None:
        del stored_unit, source_unit

    def purge(self, stored_unit) -> None:
        del stored_unit


class _SimulatedCrash(BaseException):
    pass


class _WorkspaceAuthority:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.session_exists = True
        self.session_ids = {"session-1"}

    def __call__(self, session_id: str, canonical_path: str) -> bool:
        if not self.session_exists or session_id not in self.session_ids:
            return False
        try:
            candidate = Path(canonical_path).resolve()
            candidate.relative_to(self.root.resolve())
        except ValueError:
            return False
        try:
            if is_reserved_workspace_path(self.root, candidate):
                return False
        except ValueError:
            return False
        return self.root.is_dir()


class _ScopedWorker:
    """Run a maintenance worker under the project/session route it serves."""

    def __init__(self, harness: "_Harness", worker: DocumentMaintenanceWorker) -> None:
        self._harness = harness
        self._worker = worker

    def __getattr__(self, name: str):
        return getattr(self._worker, name)

    def run_once(self, *args, **kwargs):
        with bind_documents(self._harness.database):
            return self._worker.run_once(*args, **kwargs)



class _HeartbeatObservingJobStore(SqliteDocumentIngestJobStore):
    def __init__(self, *, clock: _Clock, advance_seconds: int) -> None:
        super().__init__(
            index_audit_port=SqliteDocumentIndexPort(),
        )
        self._clock = clock
        self._advance_seconds = advance_seconds
        self._condition = threading.Condition()
        self.heartbeat_renewals = 0

    def renew_lease(self, conn, **kwargs):
        renewed = super().renew_lease(conn, **kwargs)
        if threading.current_thread().name.startswith(
            "personagraph-document-job-heartbeat"
        ):
            self._clock.advance(self._advance_seconds)
            with self._condition:
                self.heartbeat_renewals += 1
                self._condition.notify_all()
        return renewed

    def wait_for_heartbeats(self, count: int, *, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        with self._condition:
            while self.heartbeat_renewals < count:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._condition.wait(remaining)
            return True


class _Harness:
    def __init__(self, tmp_path: Path) -> None:
        self.database = DocumentDatabase(
            "maintenance-test",
            tmp_path,
            tmp_path / "documents.sqlite",
        )
        self.database.initialize()
        self.clock = _Clock()
        self.jobs = SqliteDocumentIngestJobStore(
            index_audit_port=SqliteDocumentIndexPort(),
        )
        self.requests = SqliteFilePreparationRequestStore()
        self.outbox = SqliteRetrievalOutbox()
        self.catalog = SqliteRetrievalCatalog(self.database.db_path)
        self.catalog.initialize()
        self.profile = durable_chunking_profile()
        self.authority = _WorkspaceAuthority(tmp_path)
        self.spec = file_corpus_generation_spec(
            encoder_fingerprint="test-encoder@1",
            document_chunker_fingerprint=self.profile.fingerprint(),
            document_chunk_contract_version=docstore.DOCUMENT_CHUNK_CONTRACT_VERSION,
            index_recipe=DOCUMENT_HYBRID_INDEX_RECIPE_CONTRACT,
        )
        session_store.init_db()
        with session_store._connect() as conn:
            for session_id in ("session-1", "session-2"):
                conn.execute(
                    "INSERT OR IGNORE INTO sessions "
                    "(id, persona_id, title, status, created_at, last_active_at) "
                    "VALUES (?, 'Entelecheia', 'maintenance test', 'active', ?, ?)",
                    (session_id, self.clock(), self.clock()),
                )

    def connect(self) -> sqlite3.Connection:
        return self.database.open_connection()

    def link_project_file(
        self,
        canonical_path: str,
        media_type: str | None,
    ) -> tuple[str, str]:
        path = Path(canonical_path).expanduser().resolve()
        relative_path = path.relative_to(self.database.project_root.resolve()).as_posix()
        registration = WorkspaceFileAuthority(self.database).ensure_current_path(
            relative_path,
            source=FileSource.WORKSPACE_EXISTING,
            media_type=media_type,
        )
        return (
            registration.file.file_id,
            registration.version.file_version_id,
        )

    @contextmanager
    def scope(self, session_id: str = "session-1"):
        with bind_documents(self.database):
            with session_store.session_database_scope(session_id):
                yield

    @contextmanager
    def request_scope(self, session_id: str):
        with session_store.session_database_scope(session_id):
            yield

    def enqueue(
        self,
        path: Path,
        *,
        job_id: str,
        with_summary: bool = False,
        session_id: str = "session-1",
    ) -> None:
        source = fingerprint_file(path.resolve())
        processor = configured_processor_fingerprint(path.resolve())
        assert processor is not None
        file_id, file_version_id = self.link_project_file(str(path.resolve()), None)
        request = DocumentIngestJobRequest(
            job_id=job_id,
            file_id=file_id,
            file_version_id=file_version_id,
            canonical_path=str(path.resolve()),
            source_sha256=source.sha256,
            source_size=source.size_bytes,
            processor_fingerprint=str(processor),
            chunker_fingerprint=self.profile.fingerprint(),
            chunk_contract_version=docstore.DOCUMENT_CHUNK_CONTRACT_VERSION,
            target_generation_id=self.spec.version_id,
            target_generation_fingerprint=self.spec.fingerprint,
        )
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            enqueued = self.jobs.enqueue_in_transaction(conn, request=request, now=self.clock())
            self.requests.create_in_transaction(
                conn,
                request_id=f"{job_id}:{session_id}",
                job_id=enqueued.job.job_id,
                session_id=session_id,
                with_summary=with_summary,
                source_mtime_ns=source.mtime_ns,
                now=self.clock(),
            )
            conn.commit()

    def worker(
        self,
        *,
        worker_id: str = "document-worker",
        consumer: _IndexingConsumer | None = None,
        prepare=prepare_document_path,
        fault_hook=None,
        lease_seconds: int = 30,
        lease_heartbeat_interval_seconds: float | None = None,
    ) -> _ScopedWorker:
        kwargs = dict(
            worker_id=worker_id,
            connect_documents=self.connect,
            job_store=self.jobs,
            request_store=self.requests,
            request_delivery=FilePreparationDelivery(
                connect_documents=self.connect,
                validate_source_authority=self.authority,
                mount_document=docstore.mount_document,
                is_mounted=docstore.is_mounted,
                session_scope_factory=self.request_scope,
            ).deliver_pending,
            job_scope_factory=self.request_scope,
            outbox=self.outbox,
            outbox_consumer=consumer
            or _IndexingConsumer(outbox=self.outbox, catalog=self.catalog),
            catalog=self.catalog,
            generation_spec=self.spec,
            chunking_profile=self.profile,
            now=self.clock,
            prepare_document=prepare,
            validate_source_authority=self.authority,
            link_project_file=self.link_project_file,
            fault_hook=fault_hook,
            lease_seconds=lease_seconds,
            retry_after_seconds=5,
            max_outbox_batches=16,
        )
        if lease_heartbeat_interval_seconds is not None:
            kwargs["lease_heartbeat_interval_seconds"] = (
                lease_heartbeat_interval_seconds
            )
        return _ScopedWorker(self, build_document_maintenance_worker(**kwargs))

    def job(self, job_id: str):
        with self.connect() as conn:
            job = self.jobs.get(conn, job_id)
        assert job is not None
        return job


def _write_document(path: Path, text: str = "# Durable ingest\n\nEvidence survives a crash.") -> None:
    path.write_text(text, encoding="utf-8")


def _seed_current_mounted_document(
    harness: _Harness,
    *,
    document_id: str,
    version_id: str,
    session_id: str,
    path: Path,
) -> tuple[str, str]:
    """填充一个已摄取挂载，以验证引导重放策略。"""

    content = f"Historical evidence for {document_id}."
    typed_chunk = DocumentChunk(
        chunk_id=f"historical-{document_id}",
        text=content,
        span=ChunkSpan(
            start=DocumentLocator(page=1, ordinal=0),
            end=DocumentLocator(page=1, ordinal=0),
        ),
        section_path=(),
        element_ids=(f"historical-element-{document_id}",),
        token_count=max(1, len(content.split())),
        kind=ElementKind.PARAGRAPH,
    )
    del document_id, version_id
    with harness.scope(session_id):
        relative_path = path.resolve().relative_to(harness.database.project_root).as_posix()
        registration = WorkspaceFileAuthority(harness.database).ensure_current_path(
            relative_path,
            source=FileSource.WORKSPACE_EXISTING,
            media_type="text/markdown",
        )
        stored = docstore.ingest(
            str(path.resolve()),
            path.stem,
            "text/markdown",
            [{"content": content, "loc": "p1"}],
            session_id=session_id,
            file_id=registration.file.file_id,
            file_version_id=registration.version.file_version_id,
            source_fingerprint=fingerprint_file(path),
            processor_fingerprint="historical-test-reader@1",
            document_chunks=(typed_chunk,),
            chunker_fingerprint=harness.profile.fingerprint(),
            processing_status="complete",
        )
        current = docstore.list_document_versions(str(stored["doc_id"]))[0]
    return str(stored["doc_id"]), str(current["id"])


def test_default_workspace_authority_resolves_symlinks_against_current_session_root(
    tmp_path,
    monkeypatch,
):
    from personagraph.session import store as session_store

    root = tmp_path / "workspace"
    root.mkdir()
    inside = root / "inside.md"
    _write_document(inside)
    private = root / ".personagraph" / "output" / "session-1" / "draft.md"
    private.parent.mkdir(parents=True)
    _write_document(private)
    outside = tmp_path / "outside.md"
    _write_document(outside)
    escape = root / "escape.md"
    escape.symlink_to(outside)
    session = {"status": "active", "working_dir": str(root)}
    monkeypatch.setattr(session_store, "get_session", lambda _session_id: session)

    assert validate_current_session_workspace_authority(
        "session-1",
        str(inside.resolve()),
    ) is True
    assert validate_current_session_workspace_authority(
        "session-1",
        str(private.resolve()),
    ) is False
    assert validate_current_session_workspace_authority(
        "session-1",
        str(escape),
    ) is False
    session["working_dir"] = str(tmp_path / "missing-root")
    assert validate_current_session_workspace_authority(
        "session-1",
        str(inside.resolve()),
    ) is False


def test_an_unbound_session_still_cannot_read_anywhere_else(tmp_path, monkeypatch):
    from personagraph.session import store as session_store

    monkeypatch.setattr(
        session_store, "get_session", lambda _id: {"status": "active", "working_dir": None}
    )
    elsewhere = tmp_path / "elsewhere.md"
    _write_document(elsewhere)

    assert validate_current_session_workspace_authority(
        "session-unbound", str(elsewhere)
    ) is False


def test_first_document_bootstraps_exact_generation_and_applies_job(tmp_path):
    harness = _Harness(tmp_path)
    path = tmp_path / "paper.md"
    _write_document(path)
    harness.enqueue(path, job_id="job-bootstrap", with_summary=True)

    report = harness.worker().run_once()

    job = harness.job("job-bootstrap")
    active = harness.catalog.active_data_version()
    assert report.claimed == 1
    assert report.applied == 1
    with harness.connect() as conn:
        request = harness.requests.get(conn, "job-bootstrap:session-1")
    assert request is not None
    assert request.with_summary is True
    assert request.delivery_status is FilePreparationDeliveryStatus.MOUNTED
    assert job.status is DocumentIngestJobStatus.APPLIED
    assert job.stage is DocumentIngestJobStage.ACTIVE
    assert active is not None
    assert active.id == harness.spec.version_id
    assert active.fingerprint == harness.spec.fingerprint
    assert active.role is RetrievalDataVersionRole.ACTIVE
    assert active.state is RetrievalDataVersionState.READY


def test_unavailable_retrieval_capability_terminalizes_job_after_one_attempt(tmp_path):
    harness = _Harness(tmp_path)
    path = tmp_path / "paper.md"
    _write_document(path)
    harness.enqueue(path, job_id="job-missing-retrieval-capability")
    consumer = _IndexingConsumer(
        outbox=harness.outbox,
        catalog=harness.catalog,
        terminal_reason="retrieval_method_unavailable",
    )

    first = harness.worker(consumer=consumer).run_once()
    second = harness.worker(consumer=consumer).run_once()

    job = harness.job("job-missing-retrieval-capability")
    assert first.claimed == 1
    assert first.terminal_failed == 1
    assert second.claimed == 0
    assert job.status is DocumentIngestJobStatus.TERMINAL_FAILED
    assert job.reason_code == "retrieval_method_unavailable"
    assert job.attempts == 1


def test_unusable_mounted_authority_blocks_bootstrap_publication(tmp_path):
    harness = _Harness(tmp_path)
    empty_source = tmp_path / "empty.md"
    empty_source.write_text("", encoding="utf-8")
    with harness.scope():
        registration = WorkspaceFileAuthority(harness.database).ensure_current_path(
            "empty.md",
            source=FileSource.WORKSPACE_EXISTING,
            media_type="text/markdown",
        )
        docstore.ingest(
            str(empty_source.resolve()),
            "empty",
            "text/markdown",
            [],
            session_id="session-1",
            file_id=registration.file.file_id,
            file_version_id=registration.version.file_version_id,
            source_fingerprint=fingerprint_file(empty_source),
        )
    path = tmp_path / "paper.md"
    _write_document(path)
    harness.enqueue(path, job_id="job-existing-corpus")

    report = harness.worker().run_once()

    job = harness.job("job-existing-corpus")
    assert report.terminal_failed == 1
    assert job.reason_code == "document_authority_unusable"
    target = harness.catalog.get_data_version(harness.spec.version_id)
    assert target is not None
    assert target.role is RetrievalDataVersionRole.STAGING
    assert target.state is RetrievalDataVersionState.BUILDING
    assert harness.catalog.active_data_version() is None


def test_bootstrap_refuses_a_historical_agent_private_mount_without_indexing_it(
    tmp_path,
):
    harness = _Harness(tmp_path)
    private = tmp_path / ".personagraph" / "output" / "session-1" / "draft.md"
    private.parent.mkdir(parents=True)
    _write_document(private, "agent-owned draft")
    private_document_id, _ = _seed_current_mounted_document(
        harness,
        document_id="private-mounted-document",
        version_id="private-mounted-version",
        session_id="session-1",
        path=private,
    )
    requested = tmp_path / "user-material.md"
    _write_document(requested, "user-owned material")
    harness.enqueue(requested, job_id="job-private-bootstrap")

    report = harness.worker().run_once()

    job = harness.job("job-private-bootstrap")
    target = harness.catalog.get_data_version(harness.spec.version_id)
    assert report.terminal_failed == 1
    assert job.reason_code == "workspace_authority_revoked"
    assert target is not None
    assert target.role is RetrievalDataVersionRole.STAGING
    assert target.state is RetrievalDataVersionState.BUILDING
    assert harness.catalog.active_data_version() is None
    with harness.connect() as conn:
        private_events = conn.execute(
            "SELECT COUNT(*) FROM retrieval_update_outbox "
            "WHERE source_unit_id LIKE ?",
            (f"%{private_document_id}%",),
        ).fetchone()[0]
    assert private_events == 0


def test_fatal_prepare_rejection_does_not_strand_empty_staging_generation(tmp_path):
    harness = _Harness(tmp_path)
    path = tmp_path / "paper.md"
    _write_document(path)
    harness.enqueue(path, job_id="job-prepare-rejected")

    def reject_prepare(_path_str: str, *, chunking_profile=None):
        return DocumentPrepareFailure("unsupported_format", {})

    report = harness.worker(prepare=reject_prepare).run_once()

    assert report.terminal_failed == 1
    assert (
        harness.job("job-prepare-rejected").reason_code
        == "document_parse_rejected_terminal"
    )
    assert harness.catalog.get_data_version(harness.spec.version_id) is None


def test_chunked_crash_reclaims_without_reparse_or_duplicate_version(tmp_path):
    harness = _Harness(tmp_path)
    path = tmp_path / "paper.md"
    _write_document(path)
    harness.enqueue(path, job_id="job-crash")
    prepare_calls = 0

    def counted_prepare(path_str: str, *, chunking_profile=None):
        nonlocal prepare_calls
        prepare_calls += 1
        return prepare_document_path(path_str, chunking_profile=chunking_profile)

    def crash_after_commit(point: str, _job) -> None:
        if point == "after_source_commit":
            raise _SimulatedCrash

    with pytest.raises(_SimulatedCrash):
        harness.worker(
            prepare=counted_prepare,
            fault_hook=crash_after_commit,
        ).run_once()

    crashed = harness.job("job-crash")
    assert crashed.status is DocumentIngestJobStatus.PROCESSING
    assert crashed.stage is DocumentIngestJobStage.CHUNKED
    harness.clock.advance(31)

    resumed = harness.worker(
        worker_id="document-worker-restarted",
        prepare=counted_prepare,
    ).run_once()

    assert resumed.applied == 1
    assert prepare_calls == 1
    with harness.connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM document_versions").fetchone()[0] == 1


def test_equal_bytes_at_distinct_paths_reuse_durable_parse_but_keep_authority(
    tmp_path,
):
    harness = _Harness(tmp_path)
    first_path = tmp_path / "first-source.md"
    second_path = tmp_path / "second-source.md"
    body = "# Shared bytes\n\nThe provenance of each physical source stays distinct."
    _write_document(first_path, body)
    _write_document(second_path, body)
    harness.enqueue(first_path, job_id="job-content-origin")
    assert harness.worker().run_once().applied == 1
    first_job = harness.job("job-content-origin")

    harness.authority.session_ids.add("session-2")
    harness.enqueue(
        second_path,
        job_id="job-content-reuse",
        session_id="session-2",
    )
    parse_calls = 0

    def must_not_parse(_path_str: str, *, chunking_profile=None):
        del chunking_profile
        nonlocal parse_calls
        parse_calls += 1
        raise AssertionError("exact durable parse artifact should have been reused")

    def crash_after_reused_commit(point: str, _job) -> None:
        if point == "after_source_commit":
            raise _SimulatedCrash

    with pytest.raises(_SimulatedCrash):
        harness.worker(
            prepare=must_not_parse,
            fault_hook=crash_after_reused_commit,
        ).run_once()

    crashed = harness.job("job-content-reuse")
    assert parse_calls == 0
    assert crashed.stage is DocumentIngestJobStage.CHUNKED
    assert crashed.status is DocumentIngestJobStatus.PROCESSING
    assert crashed.document_id != first_job.document_id
    assert crashed.document_version_id != first_job.document_version_id

    # 恢复从持久 CHUNKED 检查点继续，既不重新解析，也不创建另一权威代次。
    harness.clock.advance(31)
    resumed = harness.worker(
        worker_id="document-worker-restarted",
        prepare=must_not_parse,
    ).run_once()
    assert resumed.applied == 1
    assert parse_calls == 0
    indexed = harness.catalog.list_stored_units(harness.spec.version_id)
    with harness.connect() as conn:
        documents = conn.execute(
            "SELECT id, path, source_sha256, current_version_id "
            "FROM documents ORDER BY path"
        ).fetchall()
        chunks = conn.execute(
            "SELECT id, doc_id, source_version_id, producer_chunk_id, content "
            "FROM doc_chunks ORDER BY doc_id, seq"
        ).fetchall()
    mounts = []
    for session_id in ("session-1", "session-2"):
        with harness.scope(session_id):
            mounts.extend(session_store.list_document_mounts(session_id))
    assert len(documents) == 2
    assert {row["path"] for row in documents} == {
        str(first_path.resolve()),
        str(second_path.resolve()),
    }
    assert len({row["id"] for row in documents}) == 2
    assert len({row["current_version_id"] for row in documents}) == 2
    assert len({row["source_sha256"] for row in documents}) == 1
    assert len({row["id"] for row in chunks}) == len(chunks)
    assert {row["doc_id"] for row in chunks} == {row["id"] for row in documents}
    assert {row["source_version_id"] for row in chunks} == {
        row["current_version_id"] for row in documents
    }
    assert len({row["content"] for row in chunks}) == 1
    assert {(row["doc_id"], row["session_id"]) for row in mounts} == {
        (first_job.document_id, "session-1"),
        (crashed.document_id, "session-2"),
    }
    assert len({item.unit.ref.source_unit_id for item in indexed}) == len(chunks)
    assert {
        tuple(item.unit.source_filter.scope)
        for item in indexed
        if item.unit.source_filter is not None
    } == {
        (("doc_id", first_job.document_id),),
        (("doc_id", crashed.document_id),),
    }


def test_slow_prepare_heartbeats_exact_lease_and_commits_source_once(tmp_path):
    harness = _Harness(tmp_path)
    path = tmp_path / "slow-paper.md"
    _write_document(path)
    harness.enqueue(path, job_id="job-slow-parse")
    observing_jobs = _HeartbeatObservingJobStore(
        clock=harness.clock,
        advance_seconds=2,
    )
    harness.jobs = observing_jobs

    def slow_prepare(path_str: str, *, chunking_profile=None):
    # 三次成功续租使权威测试时钟推进六秒，即三秒认领租约的两倍。若没有周期性
    # 精确令牌心跳，每次解析尝试都会在此丢失租约。
        assert observing_jobs.wait_for_heartbeats(3, timeout=2)
        return prepare_document_path(path_str, chunking_profile=chunking_profile)

    report = harness.worker(
        prepare=slow_prepare,
        lease_seconds=3,
        lease_heartbeat_interval_seconds=0.01,
    ).run_once()

    job = harness.job("job-slow-parse")
    assert observing_jobs.heartbeat_renewals >= 3
    assert report.applied == 1
    assert report.lease_lost == 0
    assert job.status is DocumentIngestJobStatus.APPLIED
    assert job.attempts == 1
    with harness.connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM document_versions").fetchone()[0] == 1


@pytest.mark.parametrize("phase", ["index_and_prove", "publish_proven_target"])
def test_long_indexing_and_publication_keep_job_lease_live(tmp_path, monkeypatch, phase):
    harness = _Harness(tmp_path)
    path = tmp_path / "slow-index.md"
    _write_document(path)
    harness.enqueue(path, job_id="job-slow-index")
    observing_jobs = _HeartbeatObservingJobStore(clock=harness.clock, advance_seconds=2)
    harness.jobs = observing_jobs
    worker = harness.worker(lease_seconds=3, lease_heartbeat_interval_seconds=0.01)
    indexing = worker._worker._indexing
    original = getattr(indexing, phase)

    def slow_phase(**kwargs):
        target = observing_jobs.heartbeat_renewals + 3
        if not observing_jobs.wait_for_heartbeats(target, timeout=1):
            # Make expiry deterministic on the broken path, without waiting for
            # a real multi-minute embedding operation.
            harness.clock.advance(4)
        return original(**kwargs)

    monkeypatch.setattr(indexing, phase, slow_phase)

    report = worker.run_once(limit=1)

    assert report.applied == 1
    assert report.lease_lost == 0
    assert observing_jobs.heartbeat_renewals >= 3
    job = harness.job("job-slow-index")
    assert job.status is DocumentIngestJobStatus.APPLIED
    assert job.attempts == 1
    with harness.connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM document_versions").fetchone()[0] == 1
        attempts = conn.execute("SELECT attempts FROM retrieval_update_outbox").fetchall()
    assert attempts and all(row[0] == 1 for row in attempts)
    with harness.scope():
        assert len(session_store.list_document_mounts("session-1")) == 1


@pytest.mark.parametrize("phase", ["index_and_prove", "publish_proven_target"])
@pytest.mark.parametrize("failure", ["lease_stolen", "heartbeat_storage_failed"])
def test_long_phase_heartbeat_failure_does_not_apply_or_mount_and_can_recover(
    tmp_path, monkeypatch, phase, failure,
):
    harness = _Harness(tmp_path)
    path = tmp_path / "lease-fenced.md"
    _write_document(path)
    harness.enqueue(path, job_id="job-lease-fenced")
    worker = harness.worker(lease_seconds=3, lease_heartbeat_interval_seconds=0.01)
    indexing = worker._worker._indexing
    original_phase = getattr(indexing, phase)
    original_renew = harness.jobs.renew_lease
    phase_started = threading.Event()
    heartbeat_failed = threading.Event()

    def fail_heartbeat(conn, **kwargs):
        if phase_started.is_set() and threading.current_thread().name.startswith(
            "personagraph-document-job-heartbeat"
        ):
            try:
                if failure == "lease_stolen":
                    conn.execute(
                        "UPDATE document_ingest_jobs SET lease_owner='other-worker', "
                        "lease_token='other-token' WHERE job_id=?",
                        (kwargs["job_id"],),
                    )
                    conn.commit()
                    return original_renew(conn, **kwargs)
                raise OSError("private storage diagnostic must not be published")
            finally:
                heartbeat_failed.set()
        return original_renew(conn, **kwargs)

    def slow_phase(**kwargs):
        phase_started.set()
        assert heartbeat_failed.wait(1)
        return original_phase(**kwargs)

    monkeypatch.setattr(harness.jobs, "renew_lease", fail_heartbeat)
    monkeypatch.setattr(indexing, phase, slow_phase)
    report = worker.run_once(limit=1)

    assert heartbeat_failed.is_set()
    assert report.applied == 0
    job = harness.job("job-lease-fenced")
    assert job.stage is (
        DocumentIngestJobStage.INDEXING
        if phase == "index_and_prove"
        else DocumentIngestJobStage.COVERAGE_READY
    )
    if failure == "lease_stolen":
        assert report.lease_lost == 1
        assert job.status is DocumentIngestJobStatus.PROCESSING
        assert job.lease_owner == "other-worker"
        assert job.lease_token == "other-token"
    else:
        assert report.retryable_failed == 1
        assert report.lease_lost == 0
        assert job.status is DocumentIngestJobStatus.RETRYABLE_FAILED
        assert job.reason_code == "document_job_heartbeat_failed"
    with harness.scope():
        assert session_store.list_document_mounts("session-1") == []

    # Once the rightful lease can be reclaimed, durable source/index work is
    # reused; the prior worker never upgrades its stale token or declares ready.
    monkeypatch.setattr(harness.jobs, "renew_lease", original_renew)
    monkeypatch.setattr(indexing, phase, original_phase)
    harness.clock.advance(6)
    recovered = worker.run_once(limit=1)
    assert recovered.applied == 1
    assert harness.job("job-lease-fenced").attempts == 2
    with harness.connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM document_versions").fetchone()[0] == 1
        attempts = conn.execute("SELECT attempts FROM retrieval_update_outbox").fetchall()
    assert attempts and all(row[0] == 1 for row in attempts)


def test_frozen_target_fingerprint_mismatch_after_source_commit_fails_closed(tmp_path):
    harness = _Harness(tmp_path)
    path = tmp_path / "paper.md"
    _write_document(path)
    harness.enqueue(path, job_id="job-target-mismatch")

    def crash_after_commit(point: str, _job) -> None:
        if point == "after_source_commit":
            raise _SimulatedCrash

    with pytest.raises(_SimulatedCrash):
        harness.worker(fault_hook=crash_after_commit).run_once()
    with harness.catalog.connect() as conn:
        conn.execute(
            "UPDATE retrieval_data_versions SET fingerprint='tampered-generation' "
            "WHERE id=?",
            (harness.spec.version_id,),
        )
    harness.clock.advance(31)

    report = harness.worker(worker_id="document-worker-restarted").run_once()

    job = harness.job("job-target-mismatch")
    assert report.terminal_failed == 1
    assert job.status is DocumentIngestJobStatus.TERMINAL_FAILED
    assert job.reason_code == "retrieval_target_fingerprint_mismatch"
    assert harness.catalog.active_data_version() is None


def test_bootstrap_rejects_catalog_unit_with_widened_scope(tmp_path):
    harness = _Harness(tmp_path)
    path = tmp_path / "paper.md"
    _write_document(path)
    harness.enqueue(path, job_id="job-scope")
    consumer = _IndexingConsumer(
        outbox=harness.outbox,
        catalog=harness.catalog,
        wrong_scope=True,
    )

    report = harness.worker(consumer=consumer).run_once()

    job = harness.job("job-scope")
    target = harness.catalog.get_data_version(harness.spec.version_id)
    assert report.retryable_failed == 1
    assert job.status is DocumentIngestJobStatus.RETRYABLE_FAILED
    assert job.reason_code == "retrieval_coverage_incomplete"
    assert target is not None
    assert target.role is RetrievalDataVersionRole.STAGING
    assert target.state is RetrievalDataVersionState.BUILDING
    assert harness.catalog.active_data_version() is None


def test_bootstrap_rejects_extra_ready_active_document_unit(tmp_path):
    harness = _Harness(tmp_path)
    target, _ = harness.catalog.ensure_staging_data_version(
        version_id=harness.spec.version_id,
        fingerprint=harness.spec.fingerprint,
    )
    stale_ref = SourceUnitRef(
        source_type=SourceType.DOCUMENT,
        source_unit_id="old-session:document-v2:deleted-doc:stale-chunk",
        source_revision="deleted-version",
        indexed_content_hash="b" * 64,
    )
    stale = harness.catalog.upsert_pending_unit(
        RetrievalUnit(
            ref=stale_ref,
            retrieval_data_version=target.id,
            retrieval_status=RetrievalStatus.ACTIVE,
            source_filter=SourceFilter.from_mapping(
                SourceType.DOCUMENT,
                {"doc_id": "deleted-doc", "session_id": "old-session"},
            ),
        )
    )
    harness.catalog.mark_unit_index_ready(stale.unit_id)
    path = tmp_path / "paper.md"
    _write_document(path)
    harness.enqueue(path, job_id="job-extra-catalog-unit")

    report = harness.worker().run_once()

    job = harness.job("job-extra-catalog-unit")
    target = harness.catalog.get_data_version(harness.spec.version_id)
    assert report.retryable_failed == 1
    assert job.reason_code == "bootstrap_retrieval_manifest_not_exact"
    assert target is not None
    assert target.role is RetrievalDataVersionRole.STAGING
    assert target.state is RetrievalDataVersionState.BUILDING
    assert harness.catalog.active_data_version() is None


def test_active_generation_fingerprint_mismatch_fails_before_source_commit(tmp_path):
    harness = _Harness(tmp_path)
    harness.catalog.create_data_version(
        version_id="legacy-active",
        fingerprint="legacy-encoder-only",
        role=RetrievalDataVersionRole.ACTIVE,
        state=RetrievalDataVersionState.READY,
    )
    path = tmp_path / "paper.md"
    _write_document(path)
    harness.enqueue(path, job_id="job-mismatch")

    report = harness.worker().run_once()

    job = harness.job("job-mismatch")
    assert report.terminal_failed == 1
    assert job.status is DocumentIngestJobStatus.TERMINAL_FAILED
    assert job.reason_code == "active_generation_fingerprint_mismatch"
    with harness.connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 0


def test_frozen_source_change_fails_closed_before_target_creation(tmp_path):
    harness = _Harness(tmp_path)
    path = tmp_path / "paper.md"
    _write_document(path)
    harness.enqueue(path, job_id="job-source-changed")
    _write_document(path, "# Replaced\n\nDifferent bytes and authority.")

    report = harness.worker().run_once()

    job = harness.job("job-source-changed")
    assert report.terminal_failed == 1
    assert job.reason_code == "frozen_source_mismatch"
    assert harness.catalog.get_data_version(harness.spec.version_id) is None


def test_mtime_change_during_attempt_retries_without_changing_content_identity(tmp_path):
    harness = _Harness(tmp_path)
    path = tmp_path / "paper.md"
    _write_document(path)
    harness.enqueue(path, job_id="job-attempt-touch")
    initial = fingerprint_file(path)

    def touch_then_prepare(path_str: str, *, chunking_profile=None):
        os.utime(path, ns=(initial.mtime_ns, initial.mtime_ns + 1_000_000))
        return prepare_document_path(path_str, chunking_profile=chunking_profile)

    report = harness.worker(prepare=touch_then_prepare).run_once()

    job = harness.job("job-attempt-touch")
    assert report.retryable_failed == 1
    assert job.reason_code == "source_changed_during_attempt"
    assert fingerprint_file(path).sha256 == job.source_sha256
    assert harness.catalog.get_data_version(harness.spec.version_id) is None
    with harness.connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 0

    harness.clock.advance(5)
    assert harness.worker().run_once().applied == 1
    assert harness.job(job.job_id).file_version_id == job.file_version_id


@pytest.mark.parametrize("revocation", ["session_deleted", "working_dir_changed"])
def test_current_workspace_authority_is_reopened_before_any_source_write(
    tmp_path,
    revocation,
):
    harness = _Harness(tmp_path)
    path = tmp_path / "paper.md"
    _write_document(path)
    harness.enqueue(path, job_id=f"job-authority-{revocation}")
    if revocation == "session_deleted":
        harness.authority.session_exists = False
    else:
        replacement = tmp_path / "replacement-workspace"
        replacement.mkdir()
        harness.authority.root = replacement

    report = harness.worker().run_once()

    job = harness.job(f"job-authority-{revocation}")
    assert report.terminal_failed == 0
    assert report.retryable_failed == 1
    assert job.status is DocumentIngestJobStatus.RETRYABLE_FAILED
    assert job.reason_code == "file_preparation_authority_unavailable"
    assert harness.catalog.get_data_version(harness.spec.version_id) is None
    with harness.connect() as conn:
        request = harness.requests.get(conn, f"{job.job_id}:session-1")
        assert request is not None
        assert request.delivery_status is FilePreparationDeliveryStatus.BLOCKED
        assert request.reason_code == "workspace_authority_revoked"
        assert conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 0


def test_partial_prepared_document_is_durably_indexed_and_applied(tmp_path):
    harness = _Harness(tmp_path)
    path = tmp_path / "paper.md"
    _write_document(path)
    harness.enqueue(path, job_id="job-partial")

    def prepare_partial(path_str: str, *, chunking_profile=None):
        prepared = prepare_document_path(path_str, chunking_profile=chunking_profile)
        assert isinstance(prepared, PreparedDocumentIngest)
        return replace(
            prepared,
            processing_status="partial",
            processing_diagnostics=(
                {"code": "page_needs_vision", "at": "p2", "detail": "figure"},
            ),
            needs_vision=True,
        )

    report = harness.worker(prepare=prepare_partial).run_once()

    assert report.applied == 1
    with harness.connect() as conn:
        row = conn.execute(
            "SELECT processing_status, diagnostics_json FROM document_versions"
        ).fetchone()
    assert row["processing_status"] == "partial"
    assert "page_needs_vision" in row["diagnostics_json"]


def test_linked_old_purge_must_apply_before_incremental_job_can_finish(tmp_path):
    harness = _Harness(tmp_path)
    path = tmp_path / "paper.md"
    _write_document(path, "# V1\n\nOld chunk material.")
    harness.enqueue(path, job_id="job-v1")
    assert harness.worker().run_once().applied == 1

    _write_document(path, "# V2\n\nEntirely new chunk material for replacement.")
    harness.enqueue(path, job_id="job-v2")
    consumer = _IndexingConsumer(
        outbox=harness.outbox,
        catalog=harness.catalog,
        skip_kinds=frozenset({RetrievalUpdateKind.PURGE}),
    )

    report = harness.worker(consumer=consumer).run_once()

    job = harness.job("job-v2")
    assert report.retryable_failed == 1
    assert job.stage is DocumentIngestJobStage.INDEXING
    assert job.status is DocumentIngestJobStatus.RETRYABLE_FAILED
    assert job.reason_code == "retrieval_outbox_incomplete"
    with harness.connect() as conn:
        linked = harness.jobs.list_outbox_event_ids(conn, "job-v2")
        statuses = {harness.outbox.get_status(conn, event_id) for event_id in linked}
    assert OutboxStatus.PROCESSING in statuses


def test_crash_after_activation_reenters_coverage_ready_and_settles_job(tmp_path):
    harness = _Harness(tmp_path)
    path = tmp_path / "paper.md"
    _write_document(path)
    harness.enqueue(path, job_id="job-activate-crash")
    prepare_calls = 0

    def counted_prepare(path_str: str, *, chunking_profile=None):
        nonlocal prepare_calls
        prepare_calls += 1
        return prepare_document_path(path_str, chunking_profile=chunking_profile)

    def crash_after_activation(point: str, _job) -> None:
        if point == "after_data_version_activate":
            raise _SimulatedCrash

    with pytest.raises(_SimulatedCrash):
        harness.worker(
            prepare=counted_prepare,
            fault_hook=crash_after_activation,
        ).run_once()

    assert harness.job("job-activate-crash").stage is DocumentIngestJobStage.COVERAGE_READY
    active = harness.catalog.active_data_version()
    assert active is not None and active.id == harness.spec.version_id
    harness.clock.advance(31)

    report = harness.worker(
        worker_id="document-worker-restarted",
        prepare=counted_prepare,
    ).run_once()

    assert report.applied == 1
    assert prepare_calls == 1
    assert harness.job("job-activate-crash").status is DocumentIngestJobStatus.APPLIED


def test_crash_after_ready_reuses_frozen_staging_and_then_activates(tmp_path):
    harness = _Harness(tmp_path)
    path = tmp_path / "paper.md"
    _write_document(path)
    harness.enqueue(path, job_id="job-ready-crash")

    def crash_after_ready(point: str, _job) -> None:
        if point == "after_data_version_ready":
            raise _SimulatedCrash

    with pytest.raises(_SimulatedCrash):
        harness.worker(fault_hook=crash_after_ready).run_once()

    target = harness.catalog.get_data_version(harness.spec.version_id)
    assert target is not None
    assert target.role is RetrievalDataVersionRole.STAGING
    assert target.state is RetrievalDataVersionState.READY
    assert harness.job("job-ready-crash").stage is DocumentIngestJobStage.COVERAGE_READY
    harness.clock.advance(31)

    report = harness.worker(worker_id="document-worker-restarted").run_once()

    assert report.applied == 1
    active = harness.catalog.active_data_version()
    assert active is not None
    assert active.id == harness.spec.version_id
    assert harness.job("job-ready-crash").status is DocumentIngestJobStatus.APPLIED


def test_delete_before_first_publication_reopens_staging_and_next_ingest_activates(
    tmp_path,
):
    harness = _Harness(tmp_path)
    first_path = tmp_path / "first-paper.md"
    _write_document(first_path, "# First\n\nThis source will be deleted before publish.")
    harness.enqueue(first_path, job_id="job-first-ready")

    def crash_after_ready(point: str, _job) -> None:
        if point == "after_data_version_ready":
            raise _SimulatedCrash

    with pytest.raises(_SimulatedCrash):
        harness.worker(fault_hook=crash_after_ready).run_once()
    first = harness.job("job-first-ready")
    assert first.document_id is not None

    def cleanup_generation(
        conn: sqlite3.Connection,
        candidate: str | None,
    ) -> str | None:
        assert conn.in_transaction
        assert candidate == harness.spec.version_id
        assert harness.catalog.active_data_version_in_transaction(conn) is None
        target = harness.catalog.get_data_version_in_transaction(conn, candidate)
        assert target is not None
        if target.state is RetrievalDataVersionState.READY:
            target = harness.catalog.invalidate_ready_staging_data_version_in_transaction(
                conn,
                candidate,
            )
        assert target.role is RetrievalDataVersionRole.STAGING
        assert target.state is RetrievalDataVersionState.BUILDING
        return target.id

    with harness.scope():
        assert docstore.remove(
            first.document_id,
            retrieval_data_version_provider=cleanup_generation,
            document_index_port=SqliteDocumentIndexPort(),
        )
    target = harness.catalog.get_data_version(harness.spec.version_id)
    assert target is not None
    assert target.state is RetrievalDataVersionState.BUILDING

    harness.clock.advance(31)
    old_recovery = harness.worker(worker_id="document-worker-old-recovery").run_once()
    assert old_recovery.terminal_failed == 1

    second_path = tmp_path / "second-paper.md"
    _write_document(second_path, "# Second\n\nA fresh source after bootstrap invalidation.")
    harness.enqueue(second_path, job_id="job-second-after-delete")
    report = harness.worker(worker_id="document-worker-second").run_once()

    assert report.applied == 1
    active = harness.catalog.active_data_version()
    assert active is not None
    assert active.id == harness.spec.version_id
    assert harness.job("job-second-after-delete").status is DocumentIngestJobStatus.APPLIED


def test_applied_cleanup_resolves_removed_pending_upsert_for_bootstrap_gate(
    tmp_path,
    monkeypatch,
):
    harness = _Harness(tmp_path)
    monkeypatch.setattr(
        "personagraph.retrieval.sqlite_store.SqliteRetrievalCatalog",
        lambda *args, **kwargs: harness.catalog,
    )
    monkeypatch.setattr(
        "personagraph.retrieval.sources.document._session_retrieval_availability",
        lambda _session_id: SourceAvailability.READY,
    )
    sync = RetrievalSyncService(
        catalog=harness.catalog,
        source_readers={SourceType.DOCUMENT: MountedDocumentChunkSourceAdapter()},
        index_writer=_NoopIndexWriter(),
    )
    consumer = RetrievalOutboxConsumer(outbox=harness.outbox, sync_service=sync)
    old_path = tmp_path / "old-paper.md"
    _write_document(old_path, "# Old\n\nDeleted before its pending UPSERT ran.")
    harness.enqueue(old_path, job_id="job-old-pending-upsert")

    def crash_after_source_commit(point: str, _job) -> None:
        if point == "after_source_commit":
            raise _SimulatedCrash()

    with pytest.raises(_SimulatedCrash):
        harness.worker(
            consumer=consumer,
            fault_hook=crash_after_source_commit,
        ).run_once(limit=1)
    old = harness.job("job-old-pending-upsert")
    assert old.stage is DocumentIngestJobStage.CHUNKED
    assert old.document_id is not None
    with harness.scope():
        assert workspace_documents.delete_document(old.document_id)["deleted"] is True
        with harness.connect() as conn:
            conn.execute(
                "UPDATE retrieval_update_outbox "
                "SET occurred_at='2000-01-01T00:00:00+00:00' "
                "WHERE kind='purge'"
            )

    with harness.scope():
        with harness.connect() as conn:
            consumer.consume_due(
                conn,
                worker_id="drain-deleted-source",
                now=harness.clock(),
                lease_seconds=30,
                limit=20,
            )
            # Events for one source unit are causally serialized. The terminal
            # UPSERT must settle before its cleanup PURGE becomes claimable.
            consumer.consume_due(
                conn,
                worker_id="drain-deleted-source",
                now=harness.clock(),
                lease_seconds=30,
                limit=20,
            )
            outcomes = conn.execute(
                "SELECT kind, status, reason_code FROM retrieval_update_outbox "
                "ORDER BY authority_sequence"
            ).fetchall()
    assert [tuple(row) for row in outcomes] == [
        ("upsert", "terminal_failed", "source_unit_missing"),
        ("purge", "applied", None),
    ]

    fresh_path = tmp_path / "fresh-paper.md"
    _write_document(fresh_path, "# Fresh\n\nThe replacement corpus is authoritative.")
    harness.enqueue(fresh_path, job_id="job-fresh-after-cleanup")
    report = harness.worker(
        worker_id="document-worker-fresh",
        consumer=consumer,
    ).run_once(limit=1)

    assert report.applied == 1
    assert harness.job("job-fresh-after-cleanup").status is DocumentIngestJobStatus.APPLIED
    active = harness.catalog.active_data_version()
    assert active is not None
    assert active.id == harness.spec.version_id


def test_thread_lifecycle_scans_on_start_and_wake_only_reduces_latency(tmp_path):
    harness = _Harness(tmp_path)
    first = tmp_path / "first.md"
    _write_document(first, "# First\n\nRecovered from durable startup scan.")
    harness.enqueue(first, job_id="job-startup-scan")
    worker = harness.worker()
    lifecycle = DocumentMaintenanceWorkerLifecycle(
        worker._worker,
        poll_interval_seconds=60,
        run_limit=4,
        scope_factory=lambda: bind_documents(harness.database),
    )

    assert lifecycle.start() is True
    deadline = time.monotonic() + 10
    while (
        harness.job("job-startup-scan").status is not DocumentIngestJobStatus.APPLIED
        and time.monotonic() < deadline
    ):
        time.sleep(0.01)
    assert harness.job("job-startup-scan").status is DocumentIngestJobStatus.APPLIED

    second = tmp_path / "second.md"
    _write_document(second, "# Second\n\nWake is an optimization, not authority.")
    harness.enqueue(second, job_id="job-wake")
    lifecycle.wake()
    deadline = time.monotonic() + 10
    while (
        (
            harness.job("job-wake").status is not DocumentIngestJobStatus.APPLIED
            or lifecycle.passes < 2
        )
        and time.monotonic() < deadline
    ):
        time.sleep(0.01)

    assert harness.job("job-wake").status is DocumentIngestJobStatus.APPLIED
    assert lifecycle.passes >= 2
    assert lifecycle.failures == 0
    assert lifecycle.stop(timeout_seconds=1) is True


def test_thread_lifecycle_drains_standalone_document_outbox_without_job(tmp_path):
    harness = _Harness(tmp_path)
    harness.catalog.create_data_version(
        version_id=harness.spec.version_id,
        fingerprint=harness.spec.fingerprint,
        role=RetrievalDataVersionRole.ACTIVE,
        state=RetrievalDataVersionState.READY,
    )
    ref = SourceUnitRef(
        source_type=SourceType.DOCUMENT,
        source_unit_id="session-1:document-v2:detached-doc:old-chunk",
        source_revision="detached-version",
        indexed_content_hash="a" * 64,
    )
    stored = harness.catalog.upsert_pending_unit(
        RetrievalUnit(
            ref=ref,
            retrieval_data_version=harness.spec.version_id,
            retrieval_status=RetrievalStatus.ACTIVE,
            source_filter=SourceFilter.from_mapping(
                SourceType.DOCUMENT,
                {"doc_id": "detached-doc", "session_id": "session-1"},
            ),
        )
    )
    harness.catalog.mark_unit_index_ready(stored.unit_id)
    event = RetrievalUpdateEvent(
        event_id="standalone-detach-purge",
        kind=RetrievalUpdateKind.PURGE,
        ref=ref,
        retrieval_data_version=harness.spec.version_id,
        occurred_at=harness.clock(),
    )
    with harness.connect() as conn:
        assert harness.outbox.enqueue(conn, event) is True
        conn.commit()
    worker = harness.worker()
    lifecycle = DocumentMaintenanceWorkerLifecycle(
        worker._worker,
        poll_interval_seconds=60,
        scope_factory=lambda: bind_documents(harness.database),
    )

    assert lifecycle.start() is True
    deadline = time.monotonic() + 3
    status = None
    while time.monotonic() < deadline:
        with harness.connect() as conn:
            status = harness.outbox.get_status(conn, event.event_id)
        if status is OutboxStatus.APPLIED:
            break
        time.sleep(0.01)

    assert status is OutboxStatus.APPLIED
    assert harness.catalog.get_unit(ref, harness.spec.version_id) is None
    assert lifecycle.stop(timeout_seconds=1) is True


def test_single_owner_indexes_more_than_one_twenty_event_batch(tmp_path):
    """20 是 claim 上限，不是一个文档的索引上限。"""

    harness = _Harness(tmp_path)
    path = tmp_path / "multi-batch.md"
    body = "\n\n".join(
        f"## Section {index}\n" + (f"evidence-{index} " * 150)
        for index in range(150)
    )
    _write_document(path, body)
    prepared = prepare_document_path(
        str(path),
        chunking_profile=harness.profile,
    )
    assert isinstance(prepared, PreparedDocumentIngest)
    assert len(prepared.document_chunks) > 20
    harness.enqueue(path, job_id="job-multiple-outbox-batches")
    consumer = _IndexingConsumer(outbox=harness.outbox, catalog=harness.catalog)

    report = harness.worker(
        consumer=consumer,
        prepare=lambda _path, *, chunking_profile=None: prepared,
    ).run_once(limit=1)

    assert report.applied == 1
    assert consumer.calls == (len(prepared.document_chunks) + 19) // 20
    assert len(harness.catalog.list_stored_units(harness.spec.version_id)) == len(
        prepared.document_chunks
    )


# --- 同步收录复用的进程内 worker ------------------------------------------------


def test_the_synchronous_worker_is_built_once_per_project(tmp_path):
    """模式初始化与编码器指纹计算是逐进程成本，而非逐文件成本。"""

    from personagraph.workspace.ingestion.composition import (
        synchronous_ingest_worker,
    )

    with bound_project_document_authority(tmp_path):
        assert synchronous_ingest_worker() is synchronous_ingest_worker()


def test_the_synchronous_worker_follows_the_bound_project_database(tmp_path):
    """被持有的工作器绝不能跨越项目文档权威。"""

    from personagraph.workspace.ingestion.composition import (
        synchronous_ingest_worker,
    )

    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    with bound_project_document_authority(first_root):
        first = synchronous_ingest_worker()
    with bound_project_document_authority(second_root):
        assert synchronous_ingest_worker() is not first


def test_publishing_an_active_generation_returns_it_unchanged(tmp_path):
    """重新发布是空操作，因此重复摄取绝不会反复扰动索引。"""

    from personagraph.retrieval.sqlite_store import RetrievalDataVersionRole

    harness = _Harness(tmp_path)
    path = tmp_path / "paper.md"
    _write_document(path)
    harness.enqueue(path, job_id="job-publish", with_summary=False)
    harness.worker().run_once()

    generation = DocumentGenerationAuthority(
        catalog=harness.catalog,
        generation_spec=harness.spec,
        method_store=None,
        connect_documents=harness.connect,
        picture_source_reader=PictureObservationSourceAdapter(),
    )
    with harness.scope():
        first = generation.publish_bootstrap_generation()
        assert first.role is RetrievalDataVersionRole.ACTIVE
        assert generation.publish_bootstrap_generation() == first
