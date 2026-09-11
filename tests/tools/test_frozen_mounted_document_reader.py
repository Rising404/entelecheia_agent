from __future__ import annotations

from dataclasses import replace
import hashlib

import pytest

from personagraph.workspace.documents import application as docstore
from personagraph.tools.documents.frozen_mounted_document_reader import (
    FrozenMountedDocument,
    FrozenMountedDocumentReadError,
    FrozenMountedDocumentReadFailure,
    FrozenMountedDocumentReader,
)


SESSION_ID = "session-reader"
DOCUMENT_ID = "private-document-id"
VERSION_ID = "document-version-1"
SOURCE_SHA256 = hashlib.sha256(b"source").hexdigest()


def _chunk(sequence: int) -> docstore.CurrentDocumentResourceChunk:
    content = f"evidence {sequence}"
    return docstore.CurrentDocumentResourceChunk(
        storage_chunk_id=f"private-storage-{sequence}",
        producer_chunk_id=f"producer-{sequence}",
        sequence=sequence,
        locator=f"/private/source.txt#chunk={sequence}",
        content=content,
        content_sha256=hashlib.sha256(content.encode("utf-8")).hexdigest(),
        source_pages=(sequence + 1,),
    )


def _document(*, total_chunk_count: int | None = 3) -> FrozenMountedDocument:
    diagnostics = None if total_chunk_count is None else ()
    return FrozenMountedDocument(
        session_id=SESSION_ID,
        resource_alias="mounted_document_01",
        document_id=DOCUMENT_ID,
        document_version_id=VERSION_ID,
        source_sha256=SOURCE_SHA256,
        processing_status="complete",
        resource_format="txt",
        media_type="text/plain",
        file_extension=".txt",
        total_chunk_count=total_chunk_count,
        processing_diagnostic_codes=diagnostics,
    )


def _snapshot(
    *,
    start_sequence: int,
    maximum_chunks: int,
    chunks: tuple[docstore.CurrentDocumentResourceChunk, ...] | None = None,
    total_chunk_count: int = 3,
) -> docstore.CurrentDocumentResourceSnapshot:
    available = tuple(_chunk(index) for index in range(total_chunk_count))
    selected = (
        available[start_sequence : start_sequence + maximum_chunks]
        if chunks is None
        else chunks
    )
    return docstore.CurrentDocumentResourceSnapshot(
        session_id=SESSION_ID,
        document_id=DOCUMENT_ID,
        document_version_id=VERSION_ID,
        source_sha256=SOURCE_SHA256,
        file_extension=".txt",
        processing_status="complete",
        processing_diagnostic_codes=(),
        total_chunk_count=total_chunk_count,
        physical_page_count=None,
        page_inventory_status="unavailable",
        chunks=selected,
        truncated=(
            start_sequence > 0
            or start_sequence + maximum_chunks < total_chunk_count
        ),
    )


def _current_report(_session_id: str, document_id: str | None):
    return {
        "ok": True,
        "status": "verified_current",
        "documents": [
            {
                "doc_id": document_id,
                "version_id": VERSION_ID,
                "status": "verified_current",
            }
        ],
    }


def test_complete_traversal_is_contiguous_and_checks_physical_source_once():
    starts: list[int] = []
    freshness_calls: list[str | None] = []

    def load(_document_id: str, **kwargs):
        starts.append(kwargs["start_sequence"])
        return _snapshot(
            start_sequence=kwargs["start_sequence"],
            maximum_chunks=kwargs["maximum_chunks"],
        )

    def freshness(session_id: str, document_id: str | None):
        assert session_id == SESSION_ID
        freshness_calls.append(document_id)
        return _current_report(session_id, document_id)

    windows = tuple(
        FrozenMountedDocumentReader(
            snapshot_loader=load,
            freshness_checker=freshness,
        ).iter_windows(_document(), maximum_chunks=2)
    )

    assert starts == [0, 2]
    assert freshness_calls == [DOCUMENT_ID]
    assert [
        chunk.sequence for window in windows for chunk in window.chunks
    ] == [0, 1, 2]
    serialized = repr(windows)
    assert "/private/source.txt" not in serialized
    assert "private-storage" not in serialized


@pytest.mark.parametrize(
    "corrupt_chunk",
    (
        replace(_chunk(0), sequence=1),
        replace(_chunk(0), content_sha256="f" * 64),
    ),
)
def test_corrupt_or_non_contiguous_chunks_fail_before_projection(corrupt_chunk):
    reader = FrozenMountedDocumentReader(
        snapshot_loader=lambda _document_id, **kwargs: _snapshot(
            start_sequence=kwargs["start_sequence"],
            maximum_chunks=kwargs["maximum_chunks"],
            chunks=(corrupt_chunk,),
        ),
        freshness_checker=_current_report,
    )

    with pytest.raises(FrozenMountedDocumentReadError) as captured:
        reader.read_window(_document(), start_sequence=0, maximum_chunks=1)

    assert captured.value.failure is FrozenMountedDocumentReadFailure.READ_FAILED


def test_physical_source_drift_is_a_typed_stale_failure():
    reader = FrozenMountedDocumentReader(
        snapshot_loader=lambda _document_id, **kwargs: _snapshot(
            start_sequence=kwargs["start_sequence"],
            maximum_chunks=kwargs["maximum_chunks"],
        ),
        freshness_checker=lambda _session_id, document_id: {
            "ok": False,
            "status": "freshness_blocked",
            "documents": [
                {"doc_id": document_id, "status": "source_changed"}
            ],
        },
    )

    with pytest.raises(FrozenMountedDocumentReadError) as captured:
        reader.read_window(_document(), start_sequence=0, maximum_chunks=1)

    assert captured.value.failure is FrozenMountedDocumentReadFailure.STALE
    assert captured.value.observed_document_version_id == VERSION_ID
    assert captured.value.observed_source_sha256 == SOURCE_SHA256


def test_complete_traversal_rejects_a_core_only_binding():
    reader = FrozenMountedDocumentReader(
        snapshot_loader=lambda _document_id, **kwargs: _snapshot(
            start_sequence=kwargs["start_sequence"],
            maximum_chunks=kwargs["maximum_chunks"],
        ),
        freshness_checker=_current_report,
    )

    with pytest.raises(FrozenMountedDocumentReadError) as captured:
        tuple(reader.iter_windows(_document(total_chunk_count=None), maximum_chunks=2))

    assert captured.value.failure is FrozenMountedDocumentReadFailure.READ_FAILED
