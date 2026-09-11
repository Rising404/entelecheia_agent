"""只读查询项目共享准备状态和当前会话访问，不入队、不登记、不挂载。"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
import sqlite3

from personagraph.input_processing.documents import ChunkingProfile
from personagraph.input_processing.files import SourceFingerprint, fingerprint_file
from personagraph.input_processing.documents.readers import configured_processor_fingerprint
from personagraph.workspace.storage.context import current, require_current
from personagraph.workspace.files import find_current_file_in_connection
from personagraph.workspace.documents import application as docstore
from .storage import (
    DocumentIngestJob, DocumentIngestJobStatus, SqliteDocumentIngestJobStore,
    SqliteFilePreparationRequestStore,
)
from .contracts import FilePreparationResult, FilePreparationStatus
from .indexing_ports import IngestionGenerationIdentity
from .request import (
    FingerprintFile, ProcessorFingerprint, SourceAuthorityValidator,
    build_file_preparation_request, observe_file_preparation_source,
)

_STALE_TERMINAL_REASONS = frozenset({
    "frozen_source_mismatch", "frozen_processor_mismatch",
    "frozen_chunk_recipe_mismatch", "prepared_source_path_mismatch",
})
ExistingJobLookup = Callable[[str], DocumentIngestJob | None]


def has_file_preparation_request(*, session_id: str, request_id: str) -> bool:
    """只证明原 Session 已持久绑定文档 job；不创建库、索引或图片准备状态。

    是否仍是同一来源由后续 prepare 的冻结来源校验负责；此查询只授予重入原请求
    的资格，不代表文档已完成或可挂载。
    """
    database = current()
    if database is None:
        return False
    try:
        with database.connect_readonly() as conn:
            if conn is None:
                return False
            request = SqliteFilePreparationRequestStore().get(conn, request_id)
            return bool(
                request is not None and request.session_id == session_id
                and SqliteDocumentIngestJobStore().get(conn, request.job_id) is not None
            )
    except (OSError, sqlite3.Error, ValueError):
        return False


def resolve_file_preparation_state(
    *,
    session_id: str,
    canonical_path: str,
    frozen_fingerprint: SourceFingerprint,
    generation_identity: IngestionGenerationIdentity,
    chunking_profile: ChunkingProfile,
    validate_source_authority: SourceAuthorityValidator,
    source_fingerprint: FingerprintFile = fingerprint_file,
    processor_fingerprint: ProcessorFingerprint = configured_processor_fingerprint,
    lookup_existing_job: ExistingJobLookup | None = None,
) -> FilePreparationResult:
    if not isinstance(generation_identity, IngestionGenerationIdentity):
        raise TypeError("generation_identity must be IngestionGenerationIdentity")
    if lookup_existing_job is not None and not callable(lookup_existing_job):
        raise TypeError("lookup_existing_job must be callable")
    source = observe_file_preparation_source(
        session_id=session_id, canonical_path=canonical_path,
        frozen_fingerprint=frozen_fingerprint, chunking_profile=chunking_profile,
        validate_source_authority=validate_source_authority,
        source_fingerprint=source_fingerprint, processor_fingerprint=processor_fingerprint,
    )
    if isinstance(source, FilePreparationResult):
        return source
    database = require_current()
    try:
        relative_path = Path(canonical_path).relative_to(database.project_root).as_posix()
        with database.connect_readonly() as conn:
            current_file = (
                find_current_file_in_connection(conn, database, relative_path) if conn else None
            )
    except (OSError, sqlite3.Error, ValueError):
        return _blocked("file_readiness_lookup_unavailable")
    if current_file is None:
        return FilePreparationResult(
            status=FilePreparationStatus.PENDING, reason_code="file_not_prepared",
        )
    file, version = current_file
    identity = dict(file_id=file.file_id, file_version_id=version.file_version_id)
    if (version.content_sha256, version.size_bytes) != (
        source.fingerprint.sha256, source.fingerprint.size_bytes,
    ):
        return FilePreparationResult(
            status=FilePreparationStatus.STALE, reason_code="file_content_changed", **identity,
        )
    expected = build_file_preparation_request(
        source, generation_identity=generation_identity, **identity,
    )
    try:
        existing = (lookup_existing_job or lookup_file_preparation_job_readonly)(expected.job_id)
    except Exception:
        return _blocked("file_readiness_lookup_unavailable", operation_id=expected.job_id)
    if existing is None:
        return FilePreparationResult(
            status=FilePreparationStatus.PENDING, operation_id=expected.job_id,
            reason_code="file_not_prepared", **identity,
        )
    if not isinstance(existing, DocumentIngestJob):
        return _blocked("file_readiness_lookup_contract_violation", operation_id=expected.job_id)
    if existing.job_id != expected.job_id or existing.payload_fingerprint != expected.payload_fingerprint:
        return _blocked("file_readiness_operation_collision", operation_id=expected.job_id)
    result = replace(
        _project_existing_job(existing, expected_generation=generation_identity), **identity,
    )
    if result.status is not FilePreparationStatus.READY:
        return result
    try:
        with database.connect_readonly() as conn:
            if conn is None or not SqliteDocumentIngestJobStore().document_target_is_current(conn, existing):
                return replace(result, status=FilePreparationStatus.STALE, reason_code="prepared_document_version_changed")
            if not docstore.is_mounted_in_connection(conn, str(existing.document_id), session_id):
                return replace(result, status=FilePreparationStatus.PENDING, reason_code="file_mount_required")
    except Exception:
        return replace(result, status=FilePreparationStatus.BLOCKED, reason_code="file_readiness_lookup_unavailable")
    return result


def lookup_file_preparation_job_readonly(operation_id: str) -> DocumentIngestJob | None:
    _require_identifier("operation_id", operation_id)
    with require_current().connect_readonly() as conn:
        if conn is None:
            return None
        return SqliteDocumentIngestJobStore().get(conn, operation_id)


def _project_existing_job(
    job: DocumentIngestJob,
    *,
    expected_generation: IngestionGenerationIdentity,
) -> FilePreparationResult:
    if job.status is DocumentIngestJobStatus.APPLIED:
        if job.retrieval_data_version != expected_generation.version_id:
            return _stale(
                "retrieval_generation_changed",
                operation_id=job.job_id,
            )
        if not job.document_id or not job.document_version_id:
            return _blocked(
                "prepared_document_reference_missing",
                operation_id=job.job_id,
            )
        return FilePreparationResult(
            status=FilePreparationStatus.READY,
            operation_id=job.job_id,
            retrieval_data_version=job.retrieval_data_version,
            document_id=job.document_id,
            document_version_id=job.document_version_id,
        )
    if job.status is DocumentIngestJobStatus.TERMINAL_FAILED:
        if job.reason_code in _STALE_TERMINAL_REASONS:
            return _stale(
                job.reason_code or "file_retrieval_terminal_failure",
                operation_id=job.job_id,
            )
        return _blocked(
            job.reason_code or "file_retrieval_terminal_failure",
            operation_id=job.job_id,
        )
    return FilePreparationResult(
        status=FilePreparationStatus.PENDING,
        operation_id=job.job_id,
        reason_code=job.reason_code or "file_indexing_pending",
        retrieval_data_version=job.retrieval_data_version,
    )


def _blocked(
    reason_code: str,
    *,
    operation_id: str | None = None,
) -> FilePreparationResult:
    return FilePreparationResult(
        status=FilePreparationStatus.BLOCKED,
        operation_id=operation_id,
        reason_code=reason_code,
    )


def _stale(
    reason_code: str,
    *,
    operation_id: str | None = None,
) -> FilePreparationResult:
    return FilePreparationResult(
        status=FilePreparationStatus.STALE,
        operation_id=operation_id,
        reason_code=reason_code,
    )


def _require_identifier(name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


__all__ = [
    "resolve_file_preparation_state", "lookup_file_preparation_job_readonly",
    "has_file_preparation_request",
]
