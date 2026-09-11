"""读取当前文件观察并构造精确准备请求；不访问任务库、不启动解析。"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
import re

from personagraph.configuration.paths import deny_reason
from personagraph.input_processing.documents import ChunkingProfile
from personagraph.input_processing.files import (
    MAX_DOCUMENT_FILE_BYTES,
    SourceChangedDuringReadError,
    SourceFingerprint,
    SourceSizeLimitError,
)
from personagraph.workspace.documents import application as docstore
from .contracts import FilePreparationResult, FilePreparationStatus
from .identity import derive_file_preparation_operation_id
from .storage import DocumentIngestJobRequest
from .indexing_ports import IngestionGenerationIdentity

FingerprintFile = Callable[[Path], SourceFingerprint]
ProcessorFingerprint = Callable[[Path], object | None]
SourceAuthorityValidator = Callable[[str, str], bool]


@dataclass(frozen=True, slots=True)
class FilePreparationSource:
    canonical_path: str
    fingerprint: SourceFingerprint
    processor_fingerprint: str
    chunker_fingerprint: str
    chunk_contract_version: int


def observe_file_preparation_source(
    *,
    session_id: str,
    canonical_path: str,
    frozen_fingerprint: SourceFingerprint,
    chunking_profile: ChunkingProfile,
    validate_source_authority: SourceAuthorityValidator,
    source_fingerprint: FingerprintFile,
    processor_fingerprint: ProcessorFingerprint,
) -> FilePreparationSource | FilePreparationResult:
    """prepare 与只读 state 使用同一来源校验、处理配方与任务身份。"""

    _require_identifier("session_id", session_id)
    _require_frozen_fingerprint(frozen_fingerprint)
    if not isinstance(chunking_profile, ChunkingProfile):
        raise TypeError("chunking_profile must be ChunkingProfile")
    if not callable(validate_source_authority):
        raise TypeError("validate_source_authority must be callable")

    path = _canonical_file_path(canonical_path)
    if path is None:
        return _blocked("canonical_file_required")
    try:
        authorized = validate_source_authority(session_id, str(path))
    except Exception:
        authorized = False
    if authorized is not True:
        return _blocked("file_authority_denied")
    denied = deny_reason(path)
    if denied:
        return _blocked(denied)

    try:
        observed = source_fingerprint(path)
    except SourceSizeLimitError:
        return _blocked("file_too_large")
    except SourceChangedDuringReadError:
        return _stale("source_changed_during_read")
    except OSError:
        return _blocked("source_unavailable")
    if observed != frozen_fingerprint:
        return _stale("frozen_source_mismatch")

    try:
        processor = processor_fingerprint(path)
    except (OSError, RuntimeError, ValueError):
        return _blocked("processor_fingerprint_unavailable")
    if processor is None:
        return _blocked("unsupported_file_format")

    return FilePreparationSource(
        canonical_path=str(path),
        fingerprint=observed,
        processor_fingerprint=str(processor),
        chunker_fingerprint=chunking_profile.fingerprint(),
        chunk_contract_version=docstore.DOCUMENT_CHUNK_CONTRACT_VERSION,
    )


def build_file_preparation_request(
    source: FilePreparationSource,
    *,
    file_id: str,
    file_version_id: str,
    generation_identity: IngestionGenerationIdentity,
) -> DocumentIngestJobRequest:
    """同一项目文件版本与处理配方使用一个共享任务，与会话及mtime无关。"""

    material = dict(
        file_id=file_id,
        file_version_id=file_version_id,
        canonical_path=source.canonical_path,
        source_sha256=source.fingerprint.sha256,
        source_size=source.fingerprint.size_bytes,
        processor_fingerprint=source.processor_fingerprint,
        chunker_fingerprint=source.chunker_fingerprint,
        chunk_contract_version=source.chunk_contract_version,
        target_generation_id=generation_identity.version_id,
        target_generation_fingerprint=generation_identity.fingerprint,
    )
    return DocumentIngestJobRequest(
        job_id=derive_file_preparation_operation_id(**material),
        **material,
    )



def _canonical_file_path(value: str) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    source_path = Path(value).expanduser()
    if not source_path.is_absolute():
        return None
    try:
        resolved = source_path.resolve(strict=True)
    except OSError:
        return None
    if str(source_path) != str(resolved) or not resolved.is_file():
        return None
    return resolved


def _blocked(reason_code: str) -> FilePreparationResult:
    return FilePreparationResult(
        status=FilePreparationStatus.BLOCKED,
        reason_code=reason_code,
    )


def _stale(reason_code: str) -> FilePreparationResult:
    return FilePreparationResult(
        status=FilePreparationStatus.STALE,
        reason_code=reason_code,
    )


def _require_identifier(name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _require_frozen_fingerprint(value: SourceFingerprint) -> None:
    if not isinstance(value, SourceFingerprint):
        raise TypeError("frozen_fingerprint must be SourceFingerprint")
    if re.fullmatch(r"[0-9a-f]{64}", value.sha256) is None:
        raise ValueError("frozen_fingerprint.sha256 must be lowercase SHA-256")
    _require_non_negative_integer("frozen_fingerprint.size_bytes", value.size_bytes)
    _require_non_negative_integer("frozen_fingerprint.mtime_ns", value.mtime_ns)
    if value.size_bytes > MAX_DOCUMENT_FILE_BYTES:
        raise ValueError("frozen_fingerprint exceeds the document size policy")


def _require_non_negative_integer(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
