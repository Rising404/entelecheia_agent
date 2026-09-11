"""Workspace ingestion 持有的文档处理状态与恢复合同。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import hashlib
import json
from typing import Sequence


class DocumentIngestJobStage(StrEnum):
    PARSING = "parsing"
    CHUNKED = "chunked"
    INDEXING = "indexing"
    COVERAGE_READY = "coverage_ready"
    ACTIVE = "active"


class DocumentIngestJobStatus(StrEnum):
    PENDING = "pending"
    PROCESSING = "processing"
    APPLIED = "applied"
    RETRYABLE_FAILED = "retryable_failed"
    TERMINAL_FAILED = "terminal_failed"


class FilePreparationDeliveryStatus(StrEnum):
    PENDING = "pending"
    MOUNTED = "mounted"
    BLOCKED = "blocked"


@dataclass(frozen=True, slots=True)
class DocumentMaintenanceRunReport:
    """一次有界维护过程的无内容结果。"""

    claimed: int = 0
    applied: int = 0
    retryable_failed: int = 0
    terminal_failed: int = 0
    lease_lost: int = 0

    @property
    def processed(self) -> int:
        return self.applied + self.retryable_failed + self.terminal_failed


class DocumentIngestJobError(RuntimeError):
    """被拒绝的持久收录操作所用的错误基类。"""


class DocumentIngestJobIdCollision(DocumentIngestJobError):
    """同一公开幂等键被复用于不同的请求载荷。"""


class DocumentIngestJobLeaseError(DocumentIngestJobError):
    """变更未携带该操作当前有效的精确租约。"""


class DocumentIngestJobTransitionError(DocumentIngestJobError):
    """请求的状态转换会削弱持久操作的权威性。"""


@dataclass(frozen=True, slots=True)
class DocumentIngestJobRequest:
    """与调用方提供的 ``job_id`` 精确绑定的不可变载荷。"""

    job_id: str
    file_id: str
    file_version_id: str
    canonical_path: str
    source_sha256: str
    source_size: int
    processor_fingerprint: str
    chunker_fingerprint: str
    chunk_contract_version: int
    target_generation_id: str
    target_generation_fingerprint: str

    def __post_init__(self) -> None:
        _require_text("job_id", self.job_id)
        _require_text("file_id", self.file_id)
        _require_text("file_version_id", self.file_version_id)
        _require_text("canonical_path", self.canonical_path)
        _require_sha256("source_sha256", self.source_sha256)
        _require_non_negative_integer("source_size", self.source_size)
        _require_text("processor_fingerprint", self.processor_fingerprint)
        _require_text("chunker_fingerprint", self.chunker_fingerprint)
        _require_positive_integer("chunk_contract_version", self.chunk_contract_version)
        _require_text("target_generation_id", self.target_generation_id)
        _require_text("target_generation_fingerprint", self.target_generation_fingerprint)

    @property
    def payload_fingerprint(self) -> str:
        material = json.dumps(
            {
                "canonical_path": self.canonical_path,
                "chunk_contract_version": self.chunk_contract_version,
                "chunker_fingerprint": self.chunker_fingerprint,
                "processor_fingerprint": self.processor_fingerprint,
                "file_id": self.file_id,
                "file_version_id": self.file_version_id,
                "source_sha256": self.source_sha256,
                "source_size": self.source_size,
                "target_generation_id": self.target_generation_id,
                "target_generation_fingerprint": self.target_generation_fingerprint,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(material.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class DocumentIngestCoverageProof:
    """精确检索 generation 对当前 Document 绑定的覆盖结果。"""

    retrieval_data_version_id: str
    retrieval_data_version_fingerprint: str
    covered_binding_digest: str
    covered_binding_count: int

    def __post_init__(self) -> None:
        _require_text("retrieval_data_version_id", self.retrieval_data_version_id)
        _require_text(
            "retrieval_data_version_fingerprint",
            self.retrieval_data_version_fingerprint,
        )
        _require_sha256("covered_binding_digest", self.covered_binding_digest)
        _require_positive_integer("covered_binding_count", self.covered_binding_count)


@dataclass(frozen=True, slots=True)
class DocumentIngestJob:
    job_id: str
    payload_fingerprint: str
    file_id: str
    file_version_id: str
    canonical_path: str
    source_sha256: str
    source_size: int
    processor_fingerprint: str
    chunker_fingerprint: str
    chunk_contract_version: int
    target_generation_id: str
    target_generation_fingerprint: str
    stage: DocumentIngestJobStage
    status: DocumentIngestJobStatus
    attempts: int
    next_retry_at: str | None
    lease_owner: str | None
    lease_token: str | None
    lease_until: str | None
    completion_lease_token: str | None
    document_id: str | None
    document_version_id: str | None
    retrieval_data_version: str | None
    coverage_data_version_fingerprint: str | None
    coverage_binding_digest: str | None
    coverage_expected_bindings: int | None
    coverage_covered_bindings: int | None
    coverage_mapped_events: int | None
    coverage_applied_events: int | None
    coverage_checked_at: str | None
    reason_code: str | None
    created_at: str
    updated_at: str
    completed_at: str | None


@dataclass(frozen=True, slots=True)
class DocumentIngestEnqueueResult:
    job: DocumentIngestJob
    replayed: bool


@dataclass(frozen=True, slots=True)
class FilePreparationRequest:
    """一个 Session 对共享处理结果的独立交付请求与来源观察。"""

    request_id: str
    job_id: str
    session_id: str
    with_summary: bool
    source_mtime_ns: int
    delivery_status: FilePreparationDeliveryStatus
    reason_code: str | None
    created_at: str
    updated_at: str


def document_ingest_binding_digest(
    bindings: Sequence[tuple[str, str, str]],
) -> str:
    """精确检索覆盖探针与任务存储共享的规范摘要。"""

    normalized: list[tuple[str, str, str]] = []
    for binding in bindings:
        if not isinstance(binding, (tuple, list)) or len(binding) != 3:
            raise ValueError(
                "coverage bindings must be "
                "(source_unit_id, source_revision, indexed_content_hash) triples"
            )
        source_unit_id, source_revision, indexed_content_hash = binding
        _require_text("source_unit_id", source_unit_id)
        _require_text("source_revision", source_revision)
        _require_sha256("indexed_content_hash", indexed_content_hash)
        normalized.append((source_unit_id, source_revision, indexed_content_hash))
    if len(set(normalized)) != len(normalized):
        raise ValueError("coverage bindings must be unique")
    material = json.dumps(
        {
            "bindings": sorted(normalized),
            "contract": "document-ingest-coverage-v2",
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _require_text(name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must not be empty")
    if any(ord(character) < 32 for character in value):
        raise ValueError(f"{name} must not contain control characters")


def _require_sha256(name: str, value: str) -> None:
    _require_text(name, value)
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _require_non_negative_integer(name: str, value: int) -> None:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def _require_positive_integer(name: str, value: int) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


__all__ = [
    "DocumentIngestCoverageProof",
    "DocumentIngestEnqueueResult",
    "DocumentIngestJob",
    "DocumentIngestJobError",
    "DocumentIngestJobIdCollision",
    "DocumentIngestJobLeaseError",
    "DocumentIngestJobRequest",
    "DocumentIngestJobStage",
    "DocumentIngestJobStatus",
    "DocumentIngestJobTransitionError",
    "FilePreparationDeliveryStatus",
    "FilePreparationRequest",
    "document_ingest_binding_digest",
]
