"""Document 当前版本、块、页面与有界资源快照读取能力。"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any

from personagraph.input_processing.documents import DocumentPageManifest
from personagraph.workspace.storage.context import (
    connect_current,
    current,
    initialize_current,
)

from .contracts import (
    DOCUMENT_CHUNK_CONTRACT_VERSION,
    CurrentDocumentPageAuthority,
    CurrentDocumentResourceChunk,
    CurrentDocumentResourceSnapshot,
    DocumentChunkPageBinding,
    processing_coverage_projection,
)
from .mounting import is_mounted
from .mounting import is_mounted_in_connection
from .storage import chunks as repository


@dataclass(frozen=True, slots=True)
class CurrentFileDocument:
    """Current parsed File version, without content or Session-derived identity."""

    file_id: str
    file_version_id: str
    document_id: str
    document_version_id: str
    source_sha256: str
    total_chunk_count: int
    processing_status: str
    physical_page_count: int | None
    added_at: str | None


@dataclass(frozen=True, slots=True)
class CurrentFileChunk:
    document: CurrentFileDocument
    chunk_id: str
    producer_chunk_id: str
    sequence: int
    content: str = field(repr=False)
    content_sha256: str
    locator: str
    source_pages: tuple[int, ...]


@contextmanager
def _readonly_document_connection():
    database = current()
    if database is None:
        yield None
        return
    with database.connect_readonly() as connection:
        yield connection


def get_current_file_document(
    *,
    file_id: str,
    file_version_id: str,
    session_id: str | None = None,
) -> CurrentFileDocument | None:
    """Read exact current parsing lineage; a Session additionally requires a mount.

    The no-Session form is for an already authorized Host File operation. It does
    not itself grant access to a File and never registers, mounts, or parses it.
    """

    with _readonly_document_connection() as connection:
        if connection is None:
            return None
        return _current_file_document(connection, file_id, file_version_id, session_id)


def get_current_document_file(*, document_id: str, session_id: str) -> CurrentFileDocument | None:
    """Resolve a mounted Document to its exact current Project File lineage."""

    with _readonly_document_connection() as connection:
        if connection is None:
            return None
        row = repository.get_document_file_identity_row(connection, document_id)
        if row is None:
            return None
        return _current_file_document(connection, row["file_id"], row["file_version_id"], session_id)


def get_current_chunk_document(*, chunk_id: str, session_id: str) -> CurrentFileDocument | None:
    """Resolve a native chunk to its mounted current lineage without initialization.

    The caller still authorizes the physical File before reading its content.
    No alias is allocated and a stale chunk never resolves to a newer chunk.
    """
    if not chunk_id or not session_id:
        return None
    with _readonly_document_connection() as connection:
        if connection is None:
            return None
        row = repository.get_current_chunk_file_identity(connection, chunk_id)
        if row is None:
            return None
        return _current_file_document(
            connection, row["file_id"], row["file_version_id"], session_id,
        )


def _current_file_document(connection, file_id, file_version_id, session_id):
    row = repository.get_current_file_document_row(
        connection, file_id=file_id, file_version_id=file_version_id,
    )
    if row is None or (
        session_id is not None
        and not is_mounted_in_connection(connection, row["document_id"], session_id)
    ):
        return None
    if row["processing_status"] not in {"complete", "partial"}:
        return None
    return CurrentFileDocument(
        file_id=file_id,
        file_version_id=file_version_id,
        document_id=row["document_id"],
        document_version_id=row["document_version_id"],
        source_sha256=row["source_sha256"],
        total_chunk_count=int(row["n_chunks"] or 0),
        processing_status=row["processing_status"],
        physical_page_count=row["physical_page_count"],
        added_at=row["added_at"],
    )


def read_current_file_chunk(
    *,
    file_id: str,
    file_version_id: str,
    document_version_id: str,
    session_id: str,
    chunk_id: str | None = None,
    sequence: int | None = None,
) -> CurrentFileChunk | None:
    """Read one mounted current chunk by native ID or sequence, without writes."""

    if not session_id or (chunk_id is None) == (sequence is None):
        raise ValueError("exact chunk read requires a Session and one selector")
    with _readonly_document_connection() as connection:
        if connection is None:
            return None
        document = _current_file_document(connection, file_id, file_version_id, session_id)
        if document is None or document.document_version_id != document_version_id:
            return None
        row = repository.get_current_file_chunk_row(
            connection, document_id=document.document_id,
            document_version_id=document_version_id, chunk_id=chunk_id, sequence=sequence,
        )
        if row is None:
            return None
        content = row["content"]
        digest = row["content_sha256"]
        if (
            not isinstance(content, str) or not content
            or hashlib.sha256(content.encode("utf-8")).hexdigest() != digest
            or not row["producer_chunk_id"]
        ):
            return None
        pages = json.loads(row["source_pages_json"] or "[]")
        if (
            not isinstance(pages, list)
            or any(isinstance(page, bool) or not isinstance(page, int) or page < 1 for page in pages)
            or pages != sorted(set(pages))
        ):
            return None
        return CurrentFileChunk(
            document=document, chunk_id=row["id"], producer_chunk_id=row["producer_chunk_id"],
            sequence=int(row["seq"]), content=content, content_sha256=digest,
            locator=str(row["loc"] or ""), source_pages=tuple(pages),
        )


def doc_read(
    document_id: str,
    seq_from: int = 0,
    n: int = 2,
    *,
    session_id: str | None = None,
    expected_version_id: str | None = None,
) -> list[dict[str, Any]]:
    """读取当前文档块，并对 Session 调用方强制检查挂载。"""

    if not is_mounted(document_id, session_id):
        return []
    initialize_current()
    with connect_current() as conn:
        rows = repository.read_current_chunks(
            conn,
            document_id,
            sequence_from=seq_from,
            maximum_chunks=n,
            expected_version_id=expected_version_id,
        )
    return [dict(row) for row in rows]


def get_current_document_chunk(
    chunk_id: str,
    *,
    session_id: str | None = None,
) -> dict[str, Any] | None:
    """读取一个当前块，并证明其挂载和版本资格。"""

    initialize_current()
    with connect_current() as conn:
        row = repository.get_current_chunk(
            conn,
            chunk_id,
            include_file_origin=current() is not None,
        )
    if row is None or row["source_version_id"] != row["current_version_id"]:
        return None
    document_id = str(row["doc_id"])
    if session_id is not None and not is_mounted(document_id, session_id):
        return None
    return dict(row)


def get_current_typed_document_chunk(
    document_id: str,
    producer_chunk_id: str,
    *,
    expected_version_id: str,
    session_id: str | None,
) -> dict[str, Any] | None:
    """解析一个带可选 Session 能力的有类型当前块。"""

    if not all(
        isinstance(value, str) and value.strip()
        for value in (document_id, producer_chunk_id, expected_version_id)
    ):
        return None
    if session_id is not None and (
        not isinstance(session_id, str)
        or not session_id.strip()
        or not is_mounted(document_id, session_id)
    ):
        return None
    initialize_current()
    with connect_current() as conn:
        row = repository.get_current_typed_chunk(
            conn,
            document_id,
            producer_chunk_id,
            expected_version_id,
            include_file_origin=current() is not None,
        )
    return dict(row) if row else None


def get_current_document_page_authority(
    document_id: str,
    *,
    expected_version_id: str,
    session_id: str,
) -> CurrentDocumentPageAuthority | None:
    """读取精确的当前页面权威信息，不暴露其他挂载或版本。"""

    if not all(
        isinstance(value, str) and value.strip()
        for value in (document_id, expected_version_id, session_id)
    ):
        return None
    if not is_mounted(document_id, session_id):
        return None
    initialize_current()
    with connect_current() as conn:
        version, rows = repository.get_current_page_authority_rows(
            conn,
            document_id,
            expected_version_id,
        )
    if version is None:
        return None
    manifest_fields = (
        version["page_manifest_json"],
        version["page_manifest_sha256"],
        version["physical_page_count"],
    )
    if manifest_fields == (None, None, None):
        return None
    if any(value is None for value in manifest_fields):
        raise ValueError("current document page manifest lineage is corrupt")
    try:
        manifest = DocumentPageManifest.from_json(str(version["page_manifest_json"]))
    except ValueError as exc:
        raise ValueError("current document page manifest is corrupt") from exc
    if (
        manifest.canonical_json() != version["page_manifest_json"]
        or manifest.manifest_sha256 != version["page_manifest_sha256"]
        or manifest.physical_page_count != version["physical_page_count"]
    ):
        raise ValueError("current document page manifest identity is corrupt")
    source_sha256 = version["version_source_sha256"]
    if (
        not isinstance(source_sha256, str)
        or len(source_sha256) != 64
        or source_sha256 != version["document_source_sha256"]
    ):
        raise ValueError("current document source lineage is corrupt")
    coverage = processing_coverage_projection(
        version["processing_status"],
        version["diagnostics_json"],
    )
    raw_diagnostics = coverage["diagnostics"]
    if raw_diagnostics is None:
        raise ValueError("page manifest cannot be bound to legacy processing coverage")
    diagnostic_codes = tuple(
        str(diagnostic.get("code") or "") for diagnostic in raw_diagnostics
    )
    if any(not code for code in diagnostic_codes):
        raise ValueError("current document processing diagnostic identity is corrupt")

    bindings: list[DocumentChunkPageBinding] = []
    element_pages: dict[str, set[int]] = {}
    for page in manifest.pages:
        for element_id in page.text_element_ids:
            element_pages.setdefault(element_id, set()).add(page.page_number)
    covered_ids: set[str] = set()
    for row in rows:
        producer_chunk_id = row["producer_chunk_id"]
        if not isinstance(producer_chunk_id, str) or not producer_chunk_id.strip():
            raise ValueError("current document chunk producer identity is corrupt")
        try:
            source_pages_payload = json.loads(row["source_pages_json"])
            metadata_payload = json.loads(row["metadata_json"])
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(
                "current document chunk page authority is corrupt"
            ) from exc
        if (
            not isinstance(source_pages_payload, list)
            or not source_pages_payload
            or any(
                isinstance(page, bool) or not isinstance(page, int) or page < 1
                for page in source_pages_payload
            )
            or source_pages_payload != sorted(set(source_pages_payload))
            or _canonical_json(source_pages_payload) != row["source_pages_json"]
            or not isinstance(metadata_payload, Mapping)
            or _canonical_json(metadata_payload) != row["metadata_json"]
        ):
            raise ValueError("current document chunk page authority is corrupt")
        raw_element_ids = metadata_payload.get("element_ids")
        if (
            not isinstance(raw_element_ids, list)
            or not raw_element_ids
            or any(
                not isinstance(item, str) or not item.strip()
                for item in raw_element_ids
            )
            or len(raw_element_ids) != len(set(raw_element_ids))
        ):
            raise ValueError("current document chunk element authority is corrupt")
        expected_pages: set[int] = set()
        for element_id in raw_element_ids:
            pages = element_pages.get(element_id)
            if pages is None:
                raise ValueError(
                    "current document chunk references an unknown element"
                )
            covered_ids.add(element_id)
            expected_pages.update(pages)
        source_pages = tuple(source_pages_payload)
        if tuple(sorted(expected_pages)) != source_pages:
            raise ValueError("current document chunk source_pages are not exact")
        content = row["content"]
        content_sha256 = row["content_sha256"]
        if not isinstance(content, str) or not content:
            raise ValueError("current document chunk content authority is corrupt")
        canonical_content_sha256 = hashlib.sha256(content.encode("utf-8")).hexdigest()
        if (
            content_sha256 != canonical_content_sha256
            or row["chunk_contract_version"] != DOCUMENT_CHUNK_CONTRACT_VERSION
        ):
            raise ValueError("current document chunk content identity is corrupt")
        bindings.append(
            DocumentChunkPageBinding(
                storage_chunk_id=str(row["id"]),
                producer_chunk_id=producer_chunk_id,
                sequence=int(row["seq"]),
                element_ids=tuple(raw_element_ids),
                source_pages=source_pages,
                content_sha256=canonical_content_sha256,
                content_utf8_bytes=len(content.encode("utf-8")),
                content_json_utf8_bytes=len(
                    json.dumps(
                        content,
                        ensure_ascii=False,
                        allow_nan=False,
                        separators=(",", ":"),
                    ).encode("utf-8")
                ),
                chunk_contract_version=DOCUMENT_CHUNK_CONTRACT_VERSION,
            )
        )
    if covered_ids != set(element_pages):
        raise ValueError("current document page authority has unchunked text elements")
    return CurrentDocumentPageAuthority(
        session_id=session_id,
        document_id=document_id,
        document_version_id=expected_version_id,
        source_sha256=source_sha256,
        processing_status=str(coverage["processing_status"]),
        processing_diagnostic_codes=diagnostic_codes,
        page_manifest=manifest,
        chunks=tuple(bindings),
    )


def get_mounted_current_document_resource_snapshot(
    document_id: str,
    *,
    session_id: str,
    start_sequence: int = 0,
    maximum_chunks: int,
) -> CurrentDocumentResourceSnapshot | None:
    """原子加载有界、经哈希校验的当前 Document 窗口。"""

    if not all(
        isinstance(value, str) and value.strip()
        for value in (document_id, session_id)
    ):
        return None
    if not is_mounted(document_id, session_id):
        return None
    if (
        isinstance(maximum_chunks, bool)
        or not isinstance(maximum_chunks, int)
        or not 1 <= maximum_chunks <= 256
    ):
        raise ValueError("maximum_chunks must be within 1..256")
    if (
        isinstance(start_sequence, bool)
        or not isinstance(start_sequence, int)
        or start_sequence < 0
    ):
        raise ValueError("start_sequence must be a non-negative integer")

    initialize_current()
    with connect_current() as conn:
        conn.execute("BEGIN")
        document, rows, extent = repository.get_current_resource_rows(
            conn,
            document_id,
            start_sequence=start_sequence,
            maximum_chunks_with_sentinel=maximum_chunks + 1,
        )
    if document is None:
        return None
    assert extent is not None
    source_sha256 = document["version_source_sha256"]
    total_chunk_count = document["n_chunks"]
    if (
        not isinstance(source_sha256, str)
        or len(source_sha256) != 64
        or any(character not in "0123456789abcdef" for character in source_sha256)
        or source_sha256 != document["document_source_sha256"]
        or document["current_version_id"] != document["document_version_id"]
        or document["chunk_contract_version"] != DOCUMENT_CHUNK_CONTRACT_VERSION
        or isinstance(total_chunk_count, bool)
        or not isinstance(total_chunk_count, int)
        or total_chunk_count < 0
    ):
        raise ValueError("current document resource identity is corrupt")
    physical_page_count, page_inventory_status = _page_inventory(document)
    _validate_chunk_extent(extent, total_chunk_count)

    coverage = processing_coverage_projection(
        document["processing_status"],
        document["diagnostics_json"],
    )
    raw_diagnostics = coverage["diagnostics"]
    if (
        coverage["processing_status"] not in {"complete", "partial"}
        or not isinstance(raw_diagnostics, list)
    ):
        raise ValueError("current document resource lacks typed coverage authority")
    diagnostic_codes = tuple(
        str(item.get("code") or "")
        for item in raw_diagnostics
        if isinstance(item, Mapping)
    )
    if len(diagnostic_codes) != len(raw_diagnostics) or any(
        not value for value in diagnostic_codes
    ):
        raise ValueError("current document resource diagnostics are corrupt")

    remaining_chunk_count = max(0, total_chunk_count - start_sequence)
    if len(rows) != min(remaining_chunk_count, maximum_chunks + 1):
        raise ValueError("current document resource chunk count is corrupt")
    chunks = _validated_resource_chunks(
        rows,
        start_sequence=start_sequence,
        expected_version_id=str(document["document_version_id"]),
    )
    private_path = document["path"]
    if not isinstance(private_path, str) or not private_path.strip():
        raise ValueError("current document resource path identity is corrupt")
    return CurrentDocumentResourceSnapshot(
        session_id=session_id,
        document_id=document_id,
        document_version_id=str(document["document_version_id"]),
        source_sha256=source_sha256,
        file_extension=Path(private_path).suffix.lower(),
        processing_status=str(coverage["processing_status"]),
        processing_diagnostic_codes=diagnostic_codes,
        total_chunk_count=total_chunk_count,
        physical_page_count=physical_page_count,
        page_inventory_status=page_inventory_status,
        chunks=tuple(chunks[:maximum_chunks]),
        truncated=start_sequence > 0 or remaining_chunk_count > maximum_chunks,
    )


def list_current_document_chunks(
    document_id: str,
    *,
    session_id: str | None = None,
) -> list[dict[str, Any]]:
    initialize_current()
    if session_id is not None and not is_mounted(document_id, session_id):
        return []
    with connect_current() as conn:
        rows = repository.list_current_chunks(conn, document_id)
    return [dict(row) for row in rows]


def _page_inventory(document: Mapping[str, Any]) -> tuple[int | None, str]:
    fields = (
        document["page_manifest_json"],
        document["page_manifest_sha256"],
        document["physical_page_count"],
    )
    if fields == (None, None, None):
        return None, "unavailable"
    if any(value is None for value in fields):
        raise ValueError("current document page manifest lineage is corrupt")
    try:
        manifest = DocumentPageManifest.from_json(str(fields[0]))
    except ValueError as exc:
        raise ValueError("current document page manifest is corrupt") from exc
    if (
        manifest.canonical_json() != fields[0]
        or manifest.manifest_sha256 != fields[1]
        or manifest.physical_page_count != fields[2]
    ):
        raise ValueError("current document page manifest identity is corrupt")
    status = manifest.inventory_status.value
    return (
        None if status == "unavailable" else manifest.physical_page_count,
        status,
    )


def _validate_chunk_extent(extent: Mapping[str, Any], expected_count: int) -> None:
    observed = int(extent["chunk_count"])
    distinct = int(extent["distinct_sequence_count"])
    if expected_count == 0:
        valid = (
            observed == 0
            and extent["minimum_sequence"] is None
            and extent["maximum_sequence"] is None
        )
    else:
        valid = (
            observed == expected_count
            and distinct == observed
            and int(extent["minimum_sequence"]) == 0
            and int(extent["maximum_sequence"]) == expected_count - 1
        )
    if not valid:
        raise ValueError("current document resource chunk extent is corrupt")


def _validated_resource_chunks(
    rows: tuple[Mapping[str, Any], ...],
    *,
    start_sequence: int,
    expected_version_id: str,
) -> list[CurrentDocumentResourceChunk]:
    chunks: list[CurrentDocumentResourceChunk] = []
    seen_producer_ids: set[str] = set()
    for expected_sequence, row in enumerate(rows, start=start_sequence):
        content = row["content"]
        producer_chunk_id = row["producer_chunk_id"]
        content_sha256 = row["content_sha256"]
        source_pages_json = row["source_pages_json"]
        if (
            int(row["seq"]) != expected_sequence
            or row["source_version_id"] != expected_version_id
            or not isinstance(producer_chunk_id, str)
            or not producer_chunk_id.strip()
            or producer_chunk_id in seen_producer_ids
            or not isinstance(content, str)
            or not content.strip()
            or not isinstance(content_sha256, str)
            or hashlib.sha256(content.encode("utf-8")).hexdigest()
            != content_sha256
            or row["chunk_contract_version"] != DOCUMENT_CHUNK_CONTRACT_VERSION
            or not isinstance(source_pages_json, str)
        ):
            raise ValueError("current document resource chunk authority is corrupt")
        seen_producer_ids.add(producer_chunk_id)
        try:
            raw_source_pages = json.loads(source_pages_json)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(
                "current document resource chunk page authority is corrupt"
            ) from exc
        if (
            not isinstance(raw_source_pages, list)
            or any(
                isinstance(page, bool) or not isinstance(page, int) or page < 1
                for page in raw_source_pages
            )
            or raw_source_pages != sorted(set(raw_source_pages))
            or _canonical_json(raw_source_pages) != source_pages_json
        ):
            raise ValueError(
                "current document resource chunk page authority is corrupt"
            )
        chunks.append(
            CurrentDocumentResourceChunk(
                storage_chunk_id=str(row["id"]),
                producer_chunk_id=producer_chunk_id,
                sequence=expected_sequence,
                locator=str(row["loc"] or ""),
                content=content,
                content_sha256=content_sha256,
                source_pages=tuple(raw_source_pages),
            )
        )
    return chunks


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
