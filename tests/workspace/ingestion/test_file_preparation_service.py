"""文件准备应用入口的真实数据库与索引集成测试。"""

from __future__ import annotations

from pathlib import Path
import os
import pytest

from personagraph.input_processing.files import fingerprint_file
from personagraph.input_processing.documents.preparation import (
    PreparedDocumentIngest,
    prepare_document_path,
)
from personagraph.workspace.files import (
    FileSource,
    WorkspaceFileAuthority,
)
from personagraph.workspace.documents import application as docstore
from personagraph.workspace.storage.context import current as current_documents
from personagraph.retrieval.profile import durable_chunking_profile
from personagraph.retrieval.contracts import RetrievalStatus
from personagraph.retrieval.sources.identity import (
    mounted_document_chunk_ref_and_content,
)
from personagraph.retrieval.sqlite_store import UnitIndexState
from personagraph.retrieval.operations.document_maintenance import (
    build_document_retrieval_composition,
)
from personagraph.workspace.ingestion.composition import (
    build_document_maintenance_lifecycle,
    resolve_document_ingest_owner,
    synchronous_ingest_worker,
)
from personagraph.retrieval.profile import DocumentRetrievalProfile
from personagraph.workspace.ingestion.contracts import FilePreparationStatus
from personagraph.workspace.ingestion.preparation import prepare_file as _prepare_file
from personagraph.workspace.ingestion.delivery import FilePreparationDelivery
from personagraph.workspace.ingestion.storage import (
    DocumentIngestJobStatus,
    FilePreparationDeliveryStatus,
    SqliteDocumentIngestJobStore,
    SqliteFilePreparationRequestStore,
)
from personagraph.session.workspace_authority import validate_current_session_workspace_authority
from personagraph.session import store as session_store


def _write_markdown(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def prepare_file(*, session_id: str, **kwargs):
    """Exercise the production entry under its required Session→Project route."""

    with session_store.session_database_scope(session_id):
        return _prepare_file(
            session_id=session_id,
            ingest_owner=resolve_document_ingest_owner(),
            chunking_profile=durable_chunking_profile(),
            validate_source_authority=validate_current_session_workspace_authority,
            **kwargs,
        )


def _mount_without_retrieval(path: Path, *, session_id: str) -> str:
    prepared = prepare_document_path(
        str(path.resolve()),
        chunking_profile=durable_chunking_profile(),
    )
    assert isinstance(prepared, PreparedDocumentIngest)
    with session_store.session_database_scope(session_id):
        database = current_documents()
        assert database is not None
        relative_path = path.resolve().relative_to(database.project_root).as_posix()
        registration = WorkspaceFileAuthority(database).ensure_current_path(
            relative_path,
            source=FileSource.WORKSPACE_EXISTING,
            media_type=prepared.mime,
        )
        stored = docstore.ingest(
            prepared.canonical_path,
            prepared.title,
            prepared.mime,
            [dict(element) for element in prepared.elements],
            session_id=session_id,
            file_id=registration.file.file_id,
            file_version_id=registration.version.file_version_id,
            source_fingerprint=prepared.source_fingerprint,
            processor_fingerprint=prepared.processor_fingerprint,
            source_elements=prepared.source_elements,
            document_chunks=prepared.document_chunks,
            chunker_fingerprint=prepared.chunker_fingerprint,
            processing_status=prepared.processing_status,
            processing_diagnostics=tuple(
                dict(item) for item in prepared.processing_diagnostics
            ),
            page_manifest=prepared.page_manifest,
            retrieval_data_version=None,
        )
    return str(stored["doc_id"])


def _current_authority_manifest(session_id: str) -> set[
    tuple[str, str, str, tuple[tuple[str, str], ...]]
]:
    with session_store.session_database_scope(session_id):
        database = current_documents()
        assert database is not None
        mounted_ids = {
            str(row["doc_id"])
            for row in session_store.list_document_mounts(session_id)
        }
        if not mounted_ids:
            return set()
        placeholders = ",".join("?" for _ in mounted_ids)
        with database.connect() as conn:
            rows = conn.execute(
                "SELECT d.id AS document_id, c.id AS chunk_id, "
                "c.source_version_id, c.content, c.producer_chunk_id "
                "FROM documents AS d "
                "JOIN doc_chunks AS c ON c.doc_id=d.id "
                "AND c.source_version_id=d.current_version_id "
                f"WHERE d.id IN ({placeholders}) "
                "ORDER BY d.id, c.seq, c.id",
                tuple(sorted(mounted_ids)),
            ).fetchall()
    expected = set()
    for row in rows:
        producer_chunk_id = (
            str(row["producer_chunk_id"])
            if row["producer_chunk_id"] is not None
            else None
        )
        ref, _ = mounted_document_chunk_ref_and_content(
            session_id=session_id,
            chunk_id=str(row["chunk_id"]),
            source_version_id=str(row["source_version_id"]),
            content=str(row["content"]),
            doc_id=(
                str(row["document_id"])
                if producer_chunk_id is not None
                else None
            ),
            producer_chunk_id=producer_chunk_id,
        )
        expected.add((
            ref.source_unit_id,
            ref.source_revision,
            ref.indexed_content_hash,
            (
                (("doc_id", str(row["document_id"])),)
                if producer_chunk_id is not None
                else (
                    ("doc_id", str(row["document_id"])),
                    ("session_id", session_id),
                )
            ),
        ))
    return expected


def _active_catalog_manifest(session_id: str) -> set[
    tuple[str, str, str, tuple[tuple[str, str], ...]]
]:
    with session_store.session_database_scope(session_id):
        composition = build_document_retrieval_composition()
    active = composition.foundation.catalog.active_data_version()
    assert active is not None
    stored = composition.foundation.catalog.list_stored_units(active.id)
    assert all(unit.index_state is UnitIndexState.READY for unit in stored)
    assert all(
        unit.unit.retrieval_status is RetrievalStatus.ACTIVE for unit in stored
    )
    return {
        (
            item.unit.ref.source_unit_id,
            item.unit.ref.source_revision,
            item.unit.ref.indexed_content_hash,
            item.unit.source_filter.scope if item.unit.source_filter else (),
        )
        for item in stored
    }


def test_workspace_file_is_synchronously_indexed_and_exact_replay_is_ready(
    tmp_path: Path,
    partitioned_project_state,
) -> None:
    del partitioned_project_state
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    source = workspace / "evidence.md"
    _write_markdown(source, "# Evidence\n\nThe blue whale result is 73 percent.")
    session_id = session_store.create_session(
        "Entelecheia",
        working_dir=str(workspace.resolve()),
    )
    frozen = fingerprint_file(source.resolve())

    first = prepare_file(
        session_id=session_id,
        canonical_path=str(source.resolve()),
        frozen_fingerprint=frozen,
    )
    second = prepare_file(
        session_id=session_id,
        canonical_path=str(source.resolve()),
        frozen_fingerprint=frozen,
    )

    assert first.status is FilePreparationStatus.READY
    assert second.status is FilePreparationStatus.READY
    assert second.replayed is True
    assert second.operation_id == first.operation_id
    assert first.retrieval_data_version == second.retrieval_data_version
    assert not hasattr(first, "canonical_path")
    assert str(source.resolve()) not in repr(first)
    assert _active_catalog_manifest(session_id) == _current_authority_manifest(session_id)


def test_sessions_share_processing_but_receive_independent_delivery(
    tmp_path: Path, partitioned_project_state, monkeypatch,
) -> None:
    del partitioned_project_state
    workspace = tmp_path / "shared"
    source = workspace / "shared.md"
    _write_markdown(source, "# Shared\n\nOne parse, separate session receipts.")
    sessions = [
        session_store.create_session("Entelecheia", working_dir=str(workspace))
        for _ in range(3)
    ]
    with session_store.session_database_scope(sessions[0]):
        worker = synchronous_ingest_worker()
    parsed: list[str] = []
    original_prepare = worker._prepare_document

    def count_parse(path, **kwargs):
        parsed.append(path)
        return original_prepare(path, **kwargs)

    monkeypatch.setattr(worker, "_prepare_document", count_parse)
    first, second = [
        prepare_file(
            session_id=session_id, canonical_path=str(source),
            frozen_fingerprint=fingerprint_file(source), synchronous_max_bytes=0,
        )
        for session_id in sessions[:2]
    ]
    assert first.status is second.status is FilePreparationStatus.PENDING
    assert first.operation_id == second.operation_id
    assert first.file_version_id == second.file_version_id
    assert first.request_id != second.request_id
    with session_store.session_database_scope(sessions[0]):
        assert worker.run_once().applied == 1
    ready = [
        prepare_file(
            session_id=session_id, canonical_path=str(source),
            frozen_fingerprint=fingerprint_file(source),
        )
        for session_id in sessions
    ]
    assert parsed == [str(source)]
    assert all(result.status is FilePreparationStatus.READY for result in ready)
    assert {result.operation_id for result in ready} == {first.operation_id}
    assert len({result.request_id for result in ready}) == 3
    for session_id, result in zip(sessions, ready, strict=True):
        with session_store.session_database_scope(session_id):
            assert docstore.is_mounted(result.document_id, session_id)

    previous = fingerprint_file(source)
    os.utime(source, ns=(previous.mtime_ns, previous.mtime_ns + 1_000_000))
    touched = prepare_file(
        session_id=sessions[0], canonical_path=str(source),
        frozen_fingerprint=fingerprint_file(source),
    )
    assert touched.status is FilePreparationStatus.READY
    assert touched.operation_id == first.operation_id
    assert touched.file_version_id == first.file_version_id
    assert touched.request_id != first.request_id
    assert parsed == [str(source)]

    _write_markdown(source, "# Changed\n\nA genuinely new file content version.")
    changed = prepare_file(
        session_id=sessions[0], canonical_path=str(source),
        frozen_fingerprint=fingerprint_file(source),
    )
    assert changed.status is FilePreparationStatus.READY
    assert changed.file_id == first.file_id
    assert changed.file_version_id != first.file_version_id
    assert changed.operation_id != first.operation_id
    assert parsed == [str(source), str(source)]


@pytest.mark.parametrize("source_deleted", (False, True))
def test_delivery_retries_after_mount_without_reprocessing_shared_job(
    tmp_path: Path, partitioned_project_state, source_deleted: bool,
) -> None:
    del partitioned_project_state
    workspace = tmp_path / "delivery"
    source = workspace / "evidence.md"
    _write_markdown(source, "# Evidence\n\nA mount can precede its durable receipt.")
    session_id = session_store.create_session("Entelecheia", working_dir=str(workspace))
    queued = prepare_file(
        session_id=session_id, canonical_path=str(source),
        frozen_fingerprint=fingerprint_file(source), synchronous_max_bytes=0,
    )
    with session_store.session_database_scope(session_id):
        worker = synchronous_ingest_worker()
        original_delivery = worker._request_delivery
        worker._request_delivery = lambda **kwargs: 0
        try:
            assert worker.run_once().applied == 1
        finally:
            worker._request_delivery = original_delivery
        database = current_documents()
        assert database is not None
        requests = SqliteFilePreparationRequestStore()
        jobs = SqliteDocumentIngestJobStore()

        def mount_then_interrupt(document_id, target_session):
            docstore.mount_document(document_id, target_session)
            raise RuntimeError("simulated interruption after Session commit")

        delivery = FilePreparationDelivery(
            connect_documents=database.open_connection,
            validate_source_authority=validate_current_session_workspace_authority,
            mount_document=mount_then_interrupt, is_mounted=docstore.is_mounted,
        )
        assert delivery.deliver_request(queued.request_id).delivered is False
        with database.connect() as conn:
            request = requests.get(conn, queued.request_id)
            job = jobs.get(conn, queued.operation_id)
        assert request.delivery_status is FilePreparationDeliveryStatus.PENDING
        assert job.status is DocumentIngestJobStatus.APPLIED
        assert docstore.is_mounted(job.document_id, session_id)
        if source_deleted:
            source.unlink()
        assert original_delivery(job_id=job.job_id) == (0 if source_deleted else 1)
        with database.connect() as conn:
            request = requests.get(conn, queued.request_id)
        expected = FilePreparationDeliveryStatus.BLOCKED if source_deleted else FilePreparationDeliveryStatus.MOUNTED
        assert request.delivery_status is expected
        if source_deleted:
            assert request.reason_code == "source_unavailable"
        assert worker.run_once().claimed == 0


def test_revoked_request_does_not_poison_other_session_processing(
    tmp_path: Path, partitioned_project_state,
) -> None:
    del partitioned_project_state
    workspace = tmp_path / "independent-authority"
    source = workspace / "evidence.md"
    _write_markdown(source, "# Evidence\n\nThe remaining session can still process this file.")
    sessions = [session_store.create_session("Entelecheia", working_dir=str(workspace)) for _ in range(2)]
    queued = [
        prepare_file(
            session_id=session_id, canonical_path=str(source),
            frozen_fingerprint=fingerprint_file(source), synchronous_max_bytes=0,
        )
        for session_id in sessions
    ]
    assert queued[0].operation_id == queued[1].operation_id
    assert session_store.trash_session(sessions[0]) is True
    with session_store.session_database_scope(sessions[1]):
        assert synchronous_ingest_worker().run_once().applied == 1
        database = current_documents()
        assert database is not None
        with database.connect() as conn:
            requests = [SqliteFilePreparationRequestStore().get(conn, item.request_id) for item in queued]
        assert requests[0].delivery_status is FilePreparationDeliveryStatus.BLOCKED
        assert requests[1].delivery_status is FilePreparationDeliveryStatus.MOUNTED


def test_completed_receipt_does_not_bypass_current_authority(
    tmp_path: Path, partitioned_project_state,
) -> None:
    del partitioned_project_state
    workspace = tmp_path / "revoked"
    source = workspace / "evidence.md"
    _write_markdown(source, "# Evidence\n\nAn old delivery is not current authority.")
    session_id = session_store.create_session("Entelecheia", working_dir=str(workspace))
    frozen = fingerprint_file(source)
    first = prepare_file(
        session_id=session_id, canonical_path=str(source), frozen_fingerprint=frozen,
    )
    assert first.status is FilePreparationStatus.READY
    authorization = iter((True, True, False))
    with session_store.session_database_scope(session_id):
        result = _prepare_file(
            session_id=session_id, canonical_path=str(source), frozen_fingerprint=frozen,
            ingest_owner=resolve_document_ingest_owner(),
            chunking_profile=durable_chunking_profile(),
            validate_source_authority=lambda *_: next(authorization),
        )
    assert result.status is FilePreparationStatus.BLOCKED
    assert result.reason_code == "file_authority_denied"
    assert result.document_id is None


def test_active_background_owner_is_woken_and_settles_file_readiness(
    tmp_path: Path,
    partitioned_project_state,
) -> None:
    del partitioned_project_state
    workspace = tmp_path / "workspace-background"
    workspace.mkdir()
    source = workspace / "evidence.md"
    _write_markdown(source, "# Evidence\n\nThe background owner indexes this file.")
    session_id = session_store.create_session(
        "Entelecheia",
        working_dir=str(workspace.resolve()),
    )
    with session_store.session_database_scope(session_id):
        lifecycle = build_document_maintenance_lifecycle(
            profile=DocumentRetrievalProfile.lexical(),
        )
        assert lifecycle.start() is True
        try:
            result = prepare_file(
                session_id=session_id,
                canonical_path=str(source.resolve()),
                frozen_fingerprint=fingerprint_file(source.resolve()),
                pending_wait_seconds=10.0,
            )
        finally:
            assert lifecycle.stop(timeout_seconds=2.0) is True

    assert result.status is FilePreparationStatus.READY
    assert _active_catalog_manifest(session_id) == _current_authority_manifest(session_id)


def test_previously_mounted_managed_attachment_bootstraps_the_whole_file_corpus(
    tmp_path: Path,
    partitioned_project_state,
) -> None:
    del partitioned_project_state
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    workspace_source = workspace / "already-mounted.md"
    _write_markdown(workspace_source, "# Existing\n\nWorkspace prior evidence.")
    session_id = session_store.create_session(
        "Entelecheia",
        working_dir=str(workspace.resolve()),
    )
    _mount_without_retrieval(workspace_source, session_id=session_id)

    managed = workspace / "附件" / "att_managed_01" / "attached.md"
    _write_markdown(managed, "# Attached\n\nManaged attachment evidence.")
    _mount_without_retrieval(managed, session_id=session_id)
    frozen = fingerprint_file(managed.resolve())

    result = prepare_file(
        session_id=session_id,
        canonical_path=str(managed.resolve()),
        frozen_fingerprint=frozen,
    )
    replay = prepare_file(
        session_id=session_id,
        canonical_path=str(managed.resolve()),
        frozen_fingerprint=frozen,
    )

    assert result.status is FilePreparationStatus.READY
    assert replay.status is FilePreparationStatus.READY
    assert replay.replayed is True
    assert replay.operation_id == result.operation_id
    # 引导过程覆盖整个语料库：较早的工作区挂载与已去重附件都不得从 ACTIVE 代次消失。
    assert _active_catalog_manifest(session_id) == _current_authority_manifest(session_id)
    assert len(_active_catalog_manifest(session_id)) >= 2

    _write_markdown(managed, "# Changed\n\nManaged source drifted.")
    stale = prepare_file(
        session_id=session_id,
        canonical_path=str(managed.resolve()),
        frozen_fingerprint=frozen,
    )
    assert stale.status is FilePreparationStatus.STALE
    assert stale.operation_id is None
    assert str(managed.resolve()) not in repr(stale)


def test_source_drift_returns_stale_without_exposing_or_replacing_the_path(
    tmp_path: Path,
    partitioned_project_state,
) -> None:
    del partitioned_project_state
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    source = workspace / "changing.md"
    _write_markdown(source, "# Version one\n\nFrozen evidence.")
    session_id = session_store.create_session(
        "Entelecheia",
        working_dir=str(workspace.resolve()),
    )
    frozen = fingerprint_file(source.resolve())

    pending = prepare_file(
        session_id=session_id,
        canonical_path=str(source.resolve()),
        frozen_fingerprint=frozen,
        synchronous_max_bytes=0,
    )
    _write_markdown(source, "# Version two\n\nThe source changed after freezing.")
    stale = prepare_file(
        session_id=session_id,
        canonical_path=str(source.resolve()),
        frozen_fingerprint=frozen,
        synchronous_max_bytes=0,
    )

    assert pending.status is FilePreparationStatus.PENDING
    assert stale.status is FilePreparationStatus.STALE
    assert stale.reason_code == "frozen_source_mismatch"
    assert stale.operation_id is None
    assert str(source.resolve()) not in repr(stale)
    with session_store.session_database_scope(session_id):
        database = current_documents()
        assert database is not None
        with database.connect() as conn:
            assert conn.execute(
                "SELECT COUNT(*) FROM document_ingest_jobs"
            ).fetchone()[0] == 1


def test_preparation_result_has_no_candidate_or_alias_transport_fields(
    tmp_path: Path,
    partitioned_project_state,
) -> None:
    del partitioned_project_state
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    source = workspace / "nested.md"
    _write_markdown(source, "# Nested\n\nAuthorized evidence.")
    session_id = session_store.create_session(
        "Entelecheia",
        working_dir=str(workspace.resolve()),
    )

    result = prepare_file(
        session_id=session_id,
        canonical_path=str(source.resolve()),
        frozen_fingerprint=fingerprint_file(source.resolve()),
    )

    assert result.status is FilePreparationStatus.READY
    assert not hasattr(result, "alias")
    assert not hasattr(result, "candidate_id")
    assert result.document_id
    assert result.document_version_id


def test_pending_wait_budget_includes_source_observation_time(
    tmp_path: Path, partitioned_project_state, monkeypatch,
) -> None:
    from types import SimpleNamespace
    from personagraph.workspace.ingestion import preparation
    from personagraph.workspace.ingestion.indexing_ports import IngestionGenerationIdentity

    del partitioned_project_state
    source = tmp_path / "wait-workspace" / "evidence.md"
    _write_markdown(source, "# Evidence\n\nOnly enqueue; no parser or model execution.")
    session_id = session_store.create_session(
        "Entelecheia", working_dir=str(source.parent),
    )
    frozen = fingerprint_file(source)
    clock = [100.0]
    sleeps = []

    def observe_source(path):
        assert Path(path) == source
        clock[0] += 2.0
        return frozen

    def sleep(seconds):
        sleeps.append(seconds)
        clock[0] += seconds

    monkeypatch.setattr(preparation, "time", SimpleNamespace(
        monotonic=lambda: clock[0], sleep=sleep,
    ))
    owner = SimpleNamespace(
        generation_identity=IngestionGenerationIdentity("test-generation", "f" * 64),
        can_run_synchronously=False,
        wake=lambda: None,
    )
    with session_store.session_database_scope(session_id):
        result = _prepare_file(
            session_id=session_id, canonical_path=str(source),
            frozen_fingerprint=frozen, source_fingerprint=observe_source,
            ingest_owner=owner, chunking_profile=durable_chunking_profile(),
            validate_source_authority=validate_current_session_workspace_authority,
            processor_fingerprint=lambda _: "offline-text-reader",
            synchronous_max_bytes=0, pending_wait_seconds=1.0,
        )

    assert result.status is FilePreparationStatus.PENDING
    assert result.operation_id is not None
    assert sleeps == []
    assert clock[0] == 102.0


def test_ingestion_application_forwards_wait_budget(monkeypatch):
    from types import SimpleNamespace
    from personagraph.workspace.ingestion import application
    from personagraph.workspace.ingestion.contracts import FilePreparationResult

    generation = object()
    owner = SimpleNamespace(generation_identity=generation)
    source = SimpleNamespace(media_type="text/plain", canonical_path="/work/a.md", fingerprint=object())
    expected = FilePreparationResult(status=FilePreparationStatus.READY)
    forwarded = {}

    def prepare_file(**kwargs):
        forwarded.update(kwargs)
        return expected

    monkeypatch.setattr(application, "prepare_file", prepare_file)
    service = application.FileIngestionService(
        session_id="session-a", access=object(), generation_identity=generation,
        chunking_profile=object(), validate_source=lambda *_: True, resolve_owner=lambda: owner,
    )
    assert service.prepare(source, pending_wait_seconds=17.0) is expected
    assert forwarded["pending_wait_seconds"] == 17.0
    assert forwarded["frozen_fingerprint"] is source.fingerprint


def test_preparation_replay_keeps_exact_source_and_does_not_enqueue_changed_content(
    tmp_path, partitioned_project_state,
):
    source = tmp_path / "wait-source" / "evidence.md"
    _write_markdown(source, "# Original\n\nFrozen before suspension.")
    session_id = session_store.create_session("Entelecheia", working_dir=str(source.parent))
    kwargs = dict(session_id=session_id, canonical_path=str(source),
                  request_id="same-tool-call-input-0", synchronous_max_bytes=0)
    first = prepare_file(**kwargs, frozen_fingerprint=fingerprint_file(source))
    replayed = prepare_file(**kwargs, frozen_fingerprint=fingerprint_file(source))
    assert first.status is replayed.status is FilePreparationStatus.PENDING
    assert first.request_id == replayed.request_id
    assert first.operation_id == replayed.operation_id

    _write_markdown(source, "# Changed\n\nDo not silently continue with this version.")
    changed = prepare_file(**kwargs, frozen_fingerprint=fingerprint_file(source))
    assert changed.status is FilePreparationStatus.STALE
    assert changed.reason_code == "frozen_source_mismatch"
    with session_store.session_database_scope(session_id):
        with current_documents().connect() as conn:
            assert conn.execute("SELECT count(*) FROM document_ingest_jobs").fetchone()[0] == 1
            assert conn.execute("SELECT count(*) FROM document_ingest_requests").fetchone()[0] == 1
            assert conn.execute("SELECT count(*) FROM file_versions").fetchone()[0] == 1


def test_same_call_request_cannot_deliver_into_another_session(tmp_path, partitioned_project_state):
    source = tmp_path / "shared-request" / "evidence.md"
    _write_markdown(source, "# Original\n\nTwo independent sessions.")
    sessions = [session_store.create_session("Entelecheia", working_dir=str(source.parent)) for _ in range(2)]
    kwargs = dict(canonical_path=str(source), frozen_fingerprint=fingerprint_file(source),
                  request_id="same-tool-call-input-0", synchronous_max_bytes=0)
    first = prepare_file(session_id=sessions[0], **kwargs)
    other = prepare_file(session_id=sessions[1], **kwargs)
    assert first.status is FilePreparationStatus.PENDING
    assert other.status is FilePreparationStatus.BLOCKED
    assert other.reason_code == "file_readiness_operation_collision"


def test_document_request_cannot_replay_via_image_preparation(tmp_path, partitioned_project_state):
    from PIL import Image
    from personagraph.workspace.files.access import FileAccess
    from personagraph.workspace.ingestion.application import FileIngestionService

    source = tmp_path / "media-change" / "evidence.md"
    _write_markdown(source, "# Original\n\nThis call selected a document, not a new image.")
    session_id = session_store.create_session("Entelecheia", working_dir=str(source.parent))
    first = prepare_file(
        session_id=session_id, canonical_path=str(source), frozen_fingerprint=fingerprint_file(source),
        request_id="same-tool-call-input-0", synchronous_max_bytes=0,
    )
    assert first.status is FilePreparationStatus.PENDING
    Image.new("RGB", (24, 16)).save(source.parent / "replacement.png")

    with session_store.session_database_scope(session_id):
        database = current_documents()
        access = FileAccess(database=database, validate_path=lambda _: True)
        service = FileIngestionService(
            session_id=session_id, access=access,
            generation_identity=resolve_document_ingest_owner().generation_identity,
            chunking_profile=durable_chunking_profile(),
            validate_source=validate_current_session_workspace_authority,
            resolve_owner=resolve_document_ingest_owner,
        )
        result = service.prepare(access.resolve_path("replacement.png"), request_id=first.request_id)
        assert result.status is FilePreparationStatus.STALE
        assert result.reason_code == "frozen_source_mismatch"
        assert result.operation_id == first.operation_id
        with database.connect() as conn:
            assert conn.execute("SELECT count(*) FROM files").fetchone()[0] == 1
            assert conn.execute("SELECT count(*) FROM document_ingest_jobs").fetchone()[0] == 1


def test_preparation_wait_continues_until_session_mount_is_delivered(
    tmp_path, partitioned_project_state, monkeypatch,
):
    from personagraph.workspace.ingestion.delivery import FilePreparationDeliveryResult

    source = tmp_path / "mount-wait" / "evidence.md"
    _write_markdown(source, "# Evidence\n\nWait for this session's mount, not just index readiness.")
    session_id = session_store.create_session("Entelecheia", working_dir=str(source.parent))
    original = FilePreparationDelivery.deliver_request
    blocked = [True]
    waits = []

    def deliver(self, request_id):
        if blocked[0]:
            return FilePreparationDeliveryResult(False, "file_mount_pending")
        return original(self, request_id)

    monkeypatch.setattr(FilePreparationDelivery, "deliver_request", deliver)
    queued = prepare_file(session_id=session_id, canonical_path=str(source),
                          frozen_fingerprint=fingerprint_file(source), request_id="mount-call")
    assert queued.status is FilePreparationStatus.PENDING
    assert queued.reason_code == "file_mount_pending"
    with session_store.session_database_scope(session_id):
        with current_documents().connect() as conn:
            job = SqliteDocumentIngestJobStore().get(conn, queued.operation_id)
            assert job.status is DocumentIngestJobStatus.APPLIED
        assert not docstore.is_mounted(job.document_id, session_id)

    def checkpoint():
        waits.append(1)
        if len(waits) > 4:
            blocked[0] = False

    ready = prepare_file(session_id=session_id, canonical_path=str(source),
                         frozen_fingerprint=fingerprint_file(source), request_id="mount-call",
                         pending_wait_seconds=2, checkpoint=checkpoint)
    assert ready.status is FilePreparationStatus.READY
    assert ready.request_id == queued.request_id
    with session_store.session_database_scope(session_id):
        assert docstore.is_mounted(ready.document_id, session_id)
