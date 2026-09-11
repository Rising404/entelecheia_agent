from __future__ import annotations

from pathlib import Path

from personagraph.input_processing.files import fingerprint_file
from personagraph.workspace.documents import application as docstore
from personagraph.workspace.storage.context import current as current_documents
from personagraph.retrieval.profile import durable_chunking_profile
from personagraph.retrieval.lifecycle.generation import (
    DOCUMENT_LEXICAL_INDEX_RECIPE_CONTRACT,
    file_corpus_generation_spec,
)
from personagraph.workspace.ingestion.composition import (
    synchronous_ingest_worker,
)
from personagraph.workspace.ingestion.lifecycle import DocumentMaintenanceWorkerLifecycle
from personagraph.workspace.ingestion.execution import DocumentIngestExecutionOwner
from personagraph.workspace.ingestion.indexing_ports import IngestionGenerationIdentity
from personagraph.workspace.ingestion.contracts import FilePreparationStatus
from personagraph.workspace.ingestion.preparation import prepare_file
from personagraph.workspace.ingestion.state import resolve_file_preparation_state
from personagraph.session import store as session_store
from personagraph.session.workspace_authority import (
    validate_current_session_workspace_authority,
)


def _source_context(tmp_path: Path, *, name: str = "evidence.md") -> tuple[str, Path]:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    source = workspace / name
    source.write_text("# Evidence\n\nThe answer is in this file.\n", encoding="utf-8")
    session_id = session_store.create_session(
        "Entelecheia",
        working_dir=str(workspace.resolve()),
    )
    return session_id, source.resolve()


def _document_generation():
    spec = file_corpus_generation_spec(
        encoder_fingerprint="test-lexical-encoder-v1",
        document_chunker_fingerprint=durable_chunking_profile().fingerprint(),
        document_chunk_contract_version=docstore.DOCUMENT_CHUNK_CONTRACT_VERSION,
        index_recipe=DOCUMENT_LEXICAL_INDEX_RECIPE_CONTRACT,
    )
    return IngestionGenerationIdentity(spec.version_id, spec.fingerprint)


def _job_count(session_id: str) -> int:
    with session_store.session_database_scope(session_id):
        database = current_documents()
        assert database is not None
        with database.connect() as conn:
            return int(
                conn.execute("SELECT COUNT(*) FROM document_ingest_jobs").fetchone()[0]
            )


def _resolve(*, session_id: str, **kwargs):
    with session_store.session_database_scope(session_id):
        return resolve_file_preparation_state(
            session_id=session_id,
            chunking_profile=durable_chunking_profile(),
            validate_source_authority=validate_current_session_workspace_authority,
            **kwargs,
        )


def _ensure(*, session_id: str, worker, **kwargs):
    with session_store.session_database_scope(session_id):
        return prepare_file(
            session_id=session_id,
            ingest_owner=DocumentIngestExecutionOwner.synchronous(worker),
            chunking_profile=durable_chunking_profile(),
            validate_source_authority=validate_current_session_workspace_authority,
            **kwargs,
        )


def _worker(session_id: str):
    with session_store.session_database_scope(session_id):
        return synchronous_ingest_worker()


def _documents_db_path(session_id: str) -> Path:
    with session_store.session_database_scope(session_id):
        database = current_documents()
        assert database is not None
        return database.db_path


def test_missing_file_preparation_is_pure_pending_without_enqueue_or_worker(
    tmp_path: Path,
    partitioned_project_state,
) -> None:
    del partitioned_project_state
    session_id, source = _source_context(tmp_path)
    frozen = fingerprint_file(source)
    before = _job_count(session_id)
    lookup_calls: list[str] = []

    def lookup_existing_job(operation_id: str):
        lookup_calls.append(operation_id)
        return None

    result = _resolve(
        session_id=session_id,
        canonical_path=str(source),
        frozen_fingerprint=frozen,
        generation_identity=_document_generation(),
        lookup_existing_job=lookup_existing_job,
    )

    assert result.status is FilePreparationStatus.PENDING
    assert result.reason_code == "file_not_prepared"
    # 未登记 FileVersion 的纯查询不能为了推导 job 身份而隐式注册文件。
    assert result.operation_id is None
    assert result.file_id is None
    assert result.file_version_id is None
    assert lookup_calls == []
    assert _job_count(session_id) == before == 0
    with session_store.session_database_scope(session_id):
        database = current_documents()
        assert database is not None
        with database.connect() as conn:
            for table in ("files", "file_versions", "document_ingest_requests"):
                assert conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0
    assert str(source) not in repr(result)


def test_missing_file_lookup_does_not_initialize_a_project_document_database(
    tmp_path: Path,
    partitioned_project_state,
) -> None:
    del partitioned_project_state
    session_id, source = _source_context(tmp_path)
    frozen = fingerprint_file(source)
    documents_db_path = _documents_db_path(session_id)
    assert not documents_db_path.exists()

    result = _resolve(
        session_id=session_id,
        canonical_path=str(source),
        frozen_fingerprint=frozen,
        generation_identity=_document_generation(),
    )

    assert result.status is FilePreparationStatus.PENDING
    assert result.reason_code == "file_not_prepared"
    assert not documents_db_path.exists()


def test_existing_pending_operation_is_observed_without_running_it(
    tmp_path: Path,
    partitioned_project_state,
) -> None:
    del partitioned_project_state
    session_id, source = _source_context(tmp_path)
    frozen = fingerprint_file(source)
    worker = _worker(session_id)
    enqueued = _ensure(
        session_id=session_id,
        canonical_path=str(source),
        frozen_fingerprint=frozen,
        worker=worker,
        synchronous_max_bytes=0,
    )
    assert enqueued.status is FilePreparationStatus.PENDING
    assert enqueued.operation_id is not None

    with session_store.session_database_scope(session_id):
        database = current_documents()
        assert database is not None
        with database.connect() as conn:
            before = conn.execute(
                "SELECT status, attempts FROM document_ingest_jobs WHERE job_id=?",
                (enqueued.operation_id,),
            ).fetchone()
    assert before is not None

    observed = _resolve(
        session_id=session_id,
        canonical_path=str(source),
        frozen_fingerprint=frozen,
        generation_identity=worker.generation_identity,
    )

    with session_store.session_database_scope(session_id):
        database = current_documents()
        assert database is not None
        with database.connect() as conn:
            after = conn.execute(
                "SELECT status, attempts FROM document_ingest_jobs WHERE job_id=?",
                (enqueued.operation_id,),
            ).fetchone()
    assert observed.status is FilePreparationStatus.PENDING
    assert observed.reason_code == "file_indexing_pending"
    assert observed.operation_id == enqueued.operation_id
    assert after == before


def test_explicit_pending_wait_drives_the_durable_job_to_readiness(
    tmp_path: Path,
    partitioned_project_state,
) -> None:
    del partitioned_project_state
    session_id, source = _source_context(tmp_path)
    frozen = fingerprint_file(source)
    worker = _worker(session_id)
    lifecycle = DocumentMaintenanceWorkerLifecycle(
        worker,
        poll_interval_seconds=0.01,
        run_limit=1,
        scope_factory=lambda: session_store.session_database_scope(session_id),
    )
    lifecycle.start()
    try:
        prepared = _ensure(
            session_id=session_id,
            canonical_path=str(source),
            frozen_fingerprint=frozen,
            worker=worker,
            synchronous_max_bytes=0,
            pending_wait_seconds=5.0,
        )
    finally:
        lifecycle.stop(timeout_seconds=5.0)

    assert prepared.status is FilePreparationStatus.READY
    assert prepared.document_id is not None
    assert prepared.document_version_id is not None


def test_existing_applied_operation_is_ready_with_private_document_handles(
    tmp_path: Path,
    partitioned_project_state,
) -> None:
    del partitioned_project_state
    session_id, source = _source_context(tmp_path)
    frozen = fingerprint_file(source)
    worker = _worker(session_id)
    prepared = _ensure(
        session_id=session_id,
        canonical_path=str(source),
        frozen_fingerprint=frozen,
        worker=worker,
    )
    assert prepared.status is FilePreparationStatus.READY

    observed = _resolve(
        session_id=session_id,
        canonical_path=str(source),
        frozen_fingerprint=frozen,
        generation_identity=worker.generation_identity,
    )

    assert observed.status is FilePreparationStatus.READY
    assert observed.operation_id == prepared.operation_id
    assert observed.document_id == prepared.document_id
    assert observed.document_version_id == prepared.document_version_id
    assert observed.retrieval_data_version == prepared.retrieval_data_version
    assert str(source) not in repr(observed)


def test_source_drift_is_stale_and_does_not_create_a_preparation_job(
    tmp_path: Path,
    partitioned_project_state,
) -> None:
    del partitioned_project_state
    session_id, source = _source_context(tmp_path)
    frozen = fingerprint_file(source)
    source.write_text("# Changed\n\nThis source drifted.\n", encoding="utf-8")

    result = _resolve(
        session_id=session_id,
        canonical_path=str(source),
        frozen_fingerprint=frozen,
        generation_identity=_document_generation(),
    )

    assert result.status is FilePreparationStatus.STALE
    assert result.reason_code == "frozen_source_mismatch"
    assert result.operation_id is None
    assert _job_count(session_id) == 0
