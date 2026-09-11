"""Validated reads from one exact mounted-document generation.

This module is the single data-plane reader shared by model-visible document
tools and Host planning adapters.  It owns physical-source freshness checks,
DocStore snapshot identity validation, and contiguous chunk/hash validation;
callers only project the resulting typed window into their own contracts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
import hashlib
import json
from typing import Callable, Iterator, Mapping

from personagraph.workspace.documents import application as docstore


class FrozenMountedDocumentReadFailure(StrEnum):
    """Stable reasons why an exact frozen generation cannot be read."""

    UNAVAILABLE = "unavailable"
    STALE = "stale"
    READ_FAILED = "read_failed"


class FrozenMountedDocumentReadError(RuntimeError):
    """Typed failure that keeps private storage details out of projections."""

    def __init__(
        self,
        failure: FrozenMountedDocumentReadFailure,
        *,
        observed_document_version_id: str | None = None,
        observed_source_sha256: str | None = None,
        observed_processing_status: str | None = None,
        private_detail: str = "",
    ) -> None:
        if not isinstance(failure, FrozenMountedDocumentReadFailure):
            raise TypeError("failure must be FrozenMountedDocumentReadFailure")
        observed_identity = (
            observed_document_version_id,
            observed_source_sha256,
            observed_processing_status,
        )
        if failure is FrozenMountedDocumentReadFailure.STALE and any(
            value is None for value in observed_identity
        ):
            raise ValueError("stale reads require a complete observed identity")
        super().__init__(private_detail)
        self.failure = failure
        self.observed_document_version_id = observed_document_version_id
        self.observed_source_sha256 = observed_source_sha256
        self.observed_processing_status = observed_processing_status


@dataclass(frozen=True, slots=True)
class FrozenMountedDocument:
    """Exact private identity authorized for a mounted-document read.

    ``total_chunk_count`` and ``processing_diagnostic_codes`` extend the core
    identity when a control-plane binding freezes the complete snapshot.  A
    bounded Host read may carry only the core resource identity while still
    using the same reader and validation path.
    """

    session_id: str
    resource_alias: str
    document_id: str = field(repr=False)
    document_version_id: str = field(repr=False)
    source_sha256: str = field(repr=False)
    processing_status: str
    resource_format: str
    media_type: str
    file_extension: str
    total_chunk_count: int | None = None
    processing_diagnostic_codes: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("session_id", self.session_id),
            ("resource_alias", self.resource_alias),
            ("document_id", self.document_id),
            ("document_version_id", self.document_version_id),
            ("processing_status", self.processing_status),
            ("resource_format", self.resource_format),
            ("media_type", self.media_type),
            ("file_extension", self.file_extension),
        ):
            if not isinstance(value, str) or not value or len(value) > 500:
                raise ValueError(f"{name} must be a bounded non-empty string")
        if (
            not isinstance(self.source_sha256, str)
            or len(self.source_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.source_sha256
            )
        ):
            raise ValueError("source_sha256 must be a canonical SHA-256 digest")
        if self.processing_status not in {"complete", "partial"}:
            raise ValueError("processing_status must be complete or partial")
        if not self.file_extension.startswith("."):
            raise ValueError("file_extension must include its leading dot")
        exact_values = (
            self.total_chunk_count,
            self.processing_diagnostic_codes,
        )
        if (exact_values[0] is None) != (exact_values[1] is None):
            raise ValueError(
                "chunk count and diagnostic codes must be frozen together"
            )
        if self.total_chunk_count is not None and (
            isinstance(self.total_chunk_count, bool)
            or not isinstance(self.total_chunk_count, int)
            or self.total_chunk_count < 0
        ):
            raise ValueError("total_chunk_count cannot be negative")
        if self.processing_diagnostic_codes is not None and (
            not isinstance(self.processing_diagnostic_codes, tuple)
            or any(
                not isinstance(code, str) or not code
                for code in self.processing_diagnostic_codes
            )
        ):
            raise ValueError("processing_diagnostic_codes must be bounded strings")

    @property
    def has_exact_snapshot_binding(self) -> bool:
        return self.total_chunk_count is not None

    @property
    def freshness_binding_sha256(self) -> str:
        """Return the canonical freshness binding for a fully frozen scope."""

        if not self.has_exact_snapshot_binding:
            raise ValueError("document does not carry an exact snapshot binding")
        return _sha256_value(
            {
                "schema_version": "mounted-document-freshness-binding-v2",
                "session_id": self.session_id,
                "resource_alias": self.resource_alias,
                "resource_id": self.document_id,
                "resource_version": self.document_version_id,
                "content_sha256": self.source_sha256,
                "coverage": self.processing_status,
                "resource_format": self.resource_format,
                "media_type": self.media_type,
                "file_extension": self.file_extension,
                "total_chunk_count": self.total_chunk_count,
                "processing_diagnostic_codes": list(
                    self.processing_diagnostic_codes or ()
                ),
            }
        )


@dataclass(frozen=True, slots=True)
class FrozenMountedDocumentChunk:
    """Private-path-free chunk from one validated frozen window."""

    producer_chunk_id: str = field(repr=False)
    sequence: int
    content: str
    content_sha256: str
    source_pages: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class FrozenMountedDocumentWindow:
    """Validated, contiguous and bounded window from a frozen generation."""

    document_version_id: str
    source_sha256: str
    file_extension: str
    processing_status: str
    processing_diagnostic_codes: tuple[str, ...]
    total_chunk_count: int
    chunks: tuple[FrozenMountedDocumentChunk, ...]
    truncated: bool


SnapshotLoader = Callable[..., docstore.CurrentDocumentResourceSnapshot | None]
FreshnessChecker = Callable[[str, str | None], Mapping[str, object]]


class FrozenMountedDocumentReader:
    """Read exact frozen windows without exposing paths or DocStore identities."""

    def __init__(
        self,
        *,
        snapshot_loader: SnapshotLoader | None = None,
        freshness_checker: FreshnessChecker | None = None,
    ) -> None:
        self._snapshot_loader = (
            snapshot_loader
            or docstore.get_mounted_current_document_resource_snapshot
        )
        self._freshness_checker = (
            freshness_checker or docstore.check_mounted_document_freshness
        )

    def read_window(
        self,
        document: FrozenMountedDocument,
        *,
        start_sequence: int,
        maximum_chunks: int,
    ) -> FrozenMountedDocumentWindow:
        """Read one bounded window after validating the physical source once."""

        _validate_read_bounds(start_sequence, maximum_chunks)
        snapshot = self._load_validated_window(
            document,
            start_sequence=start_sequence,
            maximum_chunks=maximum_chunks,
        )
        self._require_current_physical_source(document, snapshot=snapshot)
        return snapshot

    def iter_windows(
        self,
        document: FrozenMountedDocument,
        *,
        maximum_chunks: int,
    ) -> Iterator[FrozenMountedDocumentWindow]:
        """Traverse the complete frozen generation with one freshness check."""

        _validate_read_bounds(0, maximum_chunks)
        if not isinstance(document, FrozenMountedDocument):
            raise TypeError("document must be FrozenMountedDocument")
        if not document.has_exact_snapshot_binding:
            raise FrozenMountedDocumentReadError(
                FrozenMountedDocumentReadFailure.READ_FAILED,
                private_detail="complete_traversal_requires_exact_snapshot",
            )
        start = 0
        first = True
        expected_total = document.total_chunk_count or 0
        while first or start < expected_total:
            window = self._load_validated_window(
                document,
                start_sequence=start,
                maximum_chunks=maximum_chunks,
            )
            if first:
                self._require_current_physical_source(document, snapshot=window)
                first = False
            yield window
            if not window.chunks:
                if start < window.total_chunk_count:
                    raise _read_error(
                        FrozenMountedDocumentReadFailure.READ_FAILED,
                        snapshot=window,
                        private_detail="incomplete_chunk_window",
                    )
                return
            next_start = window.chunks[-1].sequence + 1
            if next_start <= start:
                raise _read_error(
                    FrozenMountedDocumentReadFailure.READ_FAILED,
                    snapshot=window,
                    private_detail="cursor_did_not_advance",
                )
            start = next_start
            if start >= window.total_chunk_count:
                return

    def _load_validated_window(
        self,
        document: FrozenMountedDocument,
        *,
        start_sequence: int,
        maximum_chunks: int,
    ) -> FrozenMountedDocumentWindow:
        if not isinstance(document, FrozenMountedDocument):
            raise TypeError("document must be FrozenMountedDocument")
        try:
            raw = self._snapshot_loader(
                document.document_id,
                session_id=document.session_id,
                start_sequence=start_sequence,
                maximum_chunks=maximum_chunks,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            raise FrozenMountedDocumentReadError(
                FrozenMountedDocumentReadFailure.READ_FAILED,
                private_detail=type(exc).__name__,
            ) from exc
        if raw is None:
            raise FrozenMountedDocumentReadError(
                FrozenMountedDocumentReadFailure.UNAVAILABLE
            )
        try:
            window = _validated_window(
                raw,
                expected_start=start_sequence,
                maximum_chunks=maximum_chunks,
            )
        except (TypeError, ValueError) as exc:
            raise FrozenMountedDocumentReadError(
                FrozenMountedDocumentReadFailure.READ_FAILED,
                private_detail=type(exc).__name__,
            ) from exc
        if not _snapshot_matches_document(window, raw=raw, document=document):
            raise _read_error(
                FrozenMountedDocumentReadFailure.STALE,
                snapshot=window,
            )
        return window

    def _require_current_physical_source(
        self,
        document: FrozenMountedDocument,
        *,
        snapshot: FrozenMountedDocumentWindow,
    ) -> None:
        try:
            report = self._freshness_checker(
                document.session_id,
                document.document_id,
            )
        except Exception as exc:
            raise _read_error(
                FrozenMountedDocumentReadFailure.READ_FAILED,
                snapshot=snapshot,
                private_detail=type(exc).__name__,
            ) from exc
        freshness = _classify_physical_freshness(
            report,
            document_id=document.document_id,
            expected_version_id=document.document_version_id,
        )
        if freshness == "unavailable":
            raise _read_error(
                FrozenMountedDocumentReadFailure.UNAVAILABLE,
                snapshot=snapshot,
            )
        if freshness != "current":
            raise _read_error(
                FrozenMountedDocumentReadFailure.STALE,
                snapshot=snapshot,
            )


def _validated_window(
    raw: docstore.CurrentDocumentResourceSnapshot,
    *,
    expected_start: int,
    maximum_chunks: int,
) -> FrozenMountedDocumentWindow:
    if not isinstance(raw, docstore.CurrentDocumentResourceSnapshot):
        raise TypeError("snapshot loader returned an unsupported value")
    if (
        isinstance(raw.total_chunk_count, bool)
        or not isinstance(raw.total_chunk_count, int)
        or raw.total_chunk_count < 0
    ):
        raise ValueError("snapshot chunk count is invalid")
    if (
        not all(
            isinstance(value, str) and value
            for value in (
                raw.session_id,
                raw.document_id,
                raw.document_version_id,
                raw.source_sha256,
                raw.file_extension,
                raw.processing_status,
            )
        )
        or len(raw.source_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in raw.source_sha256
        )
        or raw.processing_status not in {"complete", "partial"}
        or not isinstance(raw.processing_diagnostic_codes, tuple)
        or any(
            not isinstance(code, str) or not code
            for code in raw.processing_diagnostic_codes
        )
        or not isinstance(raw.chunks, tuple)
        or len(raw.chunks) > maximum_chunks
        or not isinstance(raw.truncated, bool)
    ):
        raise ValueError("snapshot identity is invalid")
    chunks: list[FrozenMountedDocumentChunk] = []
    producer_ids: set[str] = set()
    for expected_sequence, chunk in enumerate(raw.chunks, start=expected_start):
        if (
            not isinstance(chunk, docstore.CurrentDocumentResourceChunk)
            or isinstance(chunk.sequence, bool)
            or not isinstance(chunk.sequence, int)
            or chunk.sequence != expected_sequence
            or chunk.sequence >= raw.total_chunk_count
            or not isinstance(chunk.producer_chunk_id, str)
            or not chunk.producer_chunk_id
            or chunk.producer_chunk_id in producer_ids
            or not isinstance(chunk.content, str)
            or not chunk.content.strip()
            or hashlib.sha256(chunk.content.encode("utf-8")).hexdigest()
            != chunk.content_sha256
            or tuple(sorted(set(chunk.source_pages))) != chunk.source_pages
            or any(
                isinstance(page, bool) or not isinstance(page, int) or page < 1
                for page in chunk.source_pages
            )
        ):
            raise ValueError("snapshot chunk window is invalid")
        producer_ids.add(chunk.producer_chunk_id)
        chunks.append(
            FrozenMountedDocumentChunk(
                producer_chunk_id=chunk.producer_chunk_id,
                sequence=chunk.sequence,
                content=chunk.content,
                content_sha256=chunk.content_sha256,
                source_pages=chunk.source_pages,
            )
        )
    return FrozenMountedDocumentWindow(
        document_version_id=raw.document_version_id,
        source_sha256=raw.source_sha256,
        file_extension=raw.file_extension,
        processing_status=raw.processing_status,
        processing_diagnostic_codes=tuple(raw.processing_diagnostic_codes),
        total_chunk_count=raw.total_chunk_count,
        chunks=tuple(chunks),
        truncated=raw.truncated,
    )


def _snapshot_matches_document(
    window: FrozenMountedDocumentWindow,
    *,
    raw: docstore.CurrentDocumentResourceSnapshot,
    document: FrozenMountedDocument,
) -> bool:
    if not (
        raw.session_id == document.session_id
        and raw.document_id == document.document_id
        and window.document_version_id == document.document_version_id
        and window.source_sha256 == document.source_sha256
        and window.file_extension == document.file_extension
        and window.processing_status == document.processing_status
    ):
        return False
    if not document.has_exact_snapshot_binding:
        return True
    return bool(
        window.total_chunk_count == document.total_chunk_count
        and window.processing_diagnostic_codes
        == document.processing_diagnostic_codes
    )


def _classify_physical_freshness(
    report: object,
    *,
    document_id: str,
    expected_version_id: str,
) -> str:
    if not isinstance(report, Mapping):
        return "stale"
    documents = report.get("documents")
    if not isinstance(documents, list):
        return "stale"
    matching = tuple(
        item
        for item in documents
        if isinstance(item, Mapping)
        and str(item.get("doc_id") or "") == document_id
    )
    if len(matching) != 1:
        return "stale"
    document = matching[0]
    if document.get("status") in {"not_mounted", "document_not_found"}:
        return "unavailable"
    if (
        report.get("ok") is True
        and report.get("status") == "verified_current"
        and document.get("status") == "verified_current"
        and str(document.get("version_id") or "") == expected_version_id
    ):
        return "current"
    return "stale"


def _validate_read_bounds(start_sequence: int, maximum_chunks: int) -> None:
    if (
        isinstance(start_sequence, bool)
        or not isinstance(start_sequence, int)
        or start_sequence < 0
    ):
        raise ValueError("start_sequence must be a non-negative integer")
    if (
        isinstance(maximum_chunks, bool)
        or not isinstance(maximum_chunks, int)
        or not 1 <= maximum_chunks <= 256
    ):
        raise ValueError("maximum_chunks must be within 1..256")


def _read_error(
    failure: FrozenMountedDocumentReadFailure,
    *,
    snapshot: FrozenMountedDocumentWindow,
    private_detail: str = "",
) -> FrozenMountedDocumentReadError:
    return FrozenMountedDocumentReadError(
        failure,
        observed_document_version_id=snapshot.document_version_id,
        observed_source_sha256=snapshot.source_sha256,
        observed_processing_status=snapshot.processing_status,
        private_detail=private_detail,
    )


def _sha256_value(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


__all__ = [
    "FrozenMountedDocument",
    "FrozenMountedDocumentChunk",
    "FrozenMountedDocumentReadError",
    "FrozenMountedDocumentReadFailure",
    "FrozenMountedDocumentReader",
    "FrozenMountedDocumentWindow",
]
