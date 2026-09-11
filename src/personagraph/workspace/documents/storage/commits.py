"""显式 SQLite 连接上的 Document generation 提交原语。

本模块不读取文件、不访问 Session/Retrieval、不生成业务 ID 或时间，也不开始、提交或
回滚事务。调用方必须先冻结并验证全部输入，再提供稳定身份和时间戳。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import sqlite3


@dataclass(frozen=True, slots=True)
class ChunkWrite:
    seq: int
    loc: str
    content: str
    producer_chunk_id: str | None = None
    span_json: str | None = None
    metadata_json: str | None = None
    content_sha256: str | None = None
    chunk_contract_version: int | None = None
    source_pages_json: str | None = None


def validate_source_elements_artifact(
    *,
    source_elements_json: str | None,
    source_elements_sha256: str | None,
    content_hash: str,
    required: bool = False,
) -> None:
    """验证切块前文本快照的完整形状、内容身份和版本内容绑定。"""

    if source_elements_json is None and source_elements_sha256 is None:
        if required:
            raise ValueError("source elements artifact is unavailable")
        return
    if source_elements_json is None or source_elements_sha256 is None:
        raise ValueError("source elements artifact requires JSON and hash")
    if (
        len(source_elements_sha256) != 64
        or any(character not in "0123456789abcdef" for character in source_elements_sha256)
        or hashlib.sha256(source_elements_json.encode("utf-8")).hexdigest()
        != source_elements_sha256
    ):
        raise ValueError("source elements artifact hash mismatch")
    try:
        payload = json.loads(source_elements_json)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("source elements artifact is not valid JSON") from exc
    if not isinstance(payload, list) or not payload:
        raise ValueError("source elements artifact must be a non-empty list")
    identities: set[str] = set()
    contents: list[str] = []
    for item in payload:
        if not isinstance(item, Mapping) or set(item) != {
            "element_id",
            "content",
            "locator",
            "source_pages",
        }:
            raise ValueError("source elements artifact contains an invalid element")
        element_id = item["element_id"]
        content = item["content"]
        locator = item["locator"]
        pages = item["source_pages"]
        if (
            not isinstance(element_id, str)
            or not element_id.strip()
            or element_id in identities
            or not isinstance(content, str)
            or not content.strip()
            or not isinstance(locator, str)
            or not isinstance(pages, list)
            or any(
                isinstance(page, bool) or not isinstance(page, int) or page < 1
                for page in pages
            )
            or pages != sorted(set(pages))
        ):
            raise ValueError("source elements artifact contains invalid fields")
        identities.add(element_id)
        contents.append(content)
    expected_content_hash = hashlib.sha256(
        "\n".join(contents).encode("utf-8")
    ).hexdigest()[:16]
    if content_hash != expected_content_hash:
        raise ValueError("source elements artifact does not match document content")


def find_document_by_path(
    conn: sqlite3.Connection,
    path: str,
    *,
    include_file_link: bool,
) -> sqlite3.Row | None:
    file_columns = (
        "d.file_id, v.file_version_id, "
        if include_file_link
        else "NULL AS file_id, NULL AS file_version_id, "
    )
    return conn.execute(
        "SELECT d.id, d.content_hash, d.hash, d.n_chunks, d.source_sha256, "
        f"{file_columns}"
        "v.page_manifest_sha256 AS current_page_manifest_sha256 "
        "FROM documents AS d LEFT JOIN document_versions AS v "
        "ON v.id=d.current_version_id WHERE d.path=? "
        "ORDER BY added_at DESC LIMIT 1",
        (path,),
    ).fetchone()


def find_document_by_content_hash(
    conn: sqlite3.Connection,
    content_hash: str,
) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT id, n_chunks FROM documents "
        "WHERE COALESCE(content_hash, hash)=? ORDER BY added_at LIMIT 1",
        (content_hash,),
    ).fetchone()


def document_path_exists(conn: sqlite3.Connection, path: str) -> bool:
    return (
        conn.execute(
            "SELECT 1 FROM documents WHERE path=? LIMIT 1",
            (path,),
        ).fetchone()
        is not None
    )


def insert_document(
    conn: sqlite3.Connection,
    *,
    document_id: str,
    path: str,
    title: str,
    mime: str,
    dedupe_key: str,
    content_hash: str,
    summary: str,
    source_session: str | None,
    chunk_count: int,
    added_at: str,
    source_sha256: str | None,
    source_size: int | None,
    source_mtime_ns: int | None,
    source_verified_at: str | None,
    freshness_status: str,
    file_id: str | None,
) -> None:
    values = (
        document_id,
        path,
        title,
        mime,
        dedupe_key,
        content_hash,
        summary,
        source_session,
        chunk_count,
        added_at,
        source_sha256,
        source_size,
        source_mtime_ns,
        source_verified_at,
        freshness_status,
        None,
    )
    if file_id is None:
        conn.execute(
            "INSERT INTO documents (id, path, title, mime, hash, content_hash, "
            "summary, source_session, n_chunks, added_at, source_sha256, "
            "source_size, source_mtime_ns, source_verified_at, "
            "freshness_status, current_version_id) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            values,
        )
    else:
        conn.execute(
            "INSERT INTO documents (id, path, title, mime, hash, content_hash, "
            "summary, source_session, n_chunks, added_at, source_sha256, "
            "source_size, source_mtime_ns, source_verified_at, "
            "freshness_status, current_version_id, file_id) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (*values, file_id),
        )


def find_reusable_content_artifact(
    conn: sqlite3.Connection,
    *,
    canonical_path: str,
    source_sha256: str,
    processor_fingerprint: str,
    chunker_fingerprint: str,
    chunk_contract_version: int,
    file_extension: str,
) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT d.id AS document_id, d.current_version_id AS document_version_id, "
        "COALESCE(d.content_hash, d.hash) AS content_hash, "
        "v.processing_status, v.diagnostics_json, v.page_manifest_json, "
        "v.page_manifest_sha256, v.physical_page_count, "
        "v.content_hash AS version_content_hash, v.source_elements_json, "
        "v.source_elements_sha256 "
        "FROM documents AS d JOIN document_versions AS v "
        "ON v.id=d.current_version_id "
        "WHERE d.path<>? AND d.source_sha256=? AND v.source_sha256=? "
        "AND v.status='active' AND v.processor_fingerprint=? "
        "AND v.chunker_fingerprint=? AND v.chunk_contract_version=? "
        "AND lower(COALESCE(d.mime, ''))=lower(?) "
        "AND v.source_elements_json IS NOT NULL "
        "AND v.source_elements_sha256 IS NOT NULL "
        "AND EXISTS (SELECT 1 FROM doc_chunks AS c "
        "WHERE c.doc_id=d.id AND c.source_version_id=v.id) "
        "ORDER BY v.activated_at DESC, d.id LIMIT 1",
        (
            canonical_path,
            source_sha256,
            source_sha256,
            processor_fingerprint,
            chunker_fingerprint,
            chunk_contract_version,
            file_extension,
        ),
    ).fetchone()


def load_reusable_content_chunks(
    conn: sqlite3.Connection,
    *,
    artifact: sqlite3.Row,
    processor_fingerprint: str,
    chunker_fingerprint: str,
    chunk_contract_version: int,
) -> tuple[ChunkWrite, ...]:
    rows = conn.execute(
        "SELECT seq, loc, content, producer_chunk_id, span_json, metadata_json, "
        "content_sha256, chunk_contract_version, source_pages_json, "
        "processor_fingerprint, chunker_fingerprint "
        "FROM doc_chunks WHERE doc_id=? AND source_version_id=? "
        "ORDER BY seq, id",
        (str(artifact["document_id"]), str(artifact["document_version_id"])),
    ).fetchall()
    writes: list[ChunkWrite] = []
    for expected_seq, row in enumerate(rows):
        content = str(row["content"] or "")
        content_sha256 = str(row["content_sha256"] or "")
        if (
            int(row["seq"]) != expected_seq
            or row["processor_fingerprint"] != processor_fingerprint
            or row["chunker_fingerprint"] != chunker_fingerprint
            or row["chunk_contract_version"] != chunk_contract_version
            or not row["producer_chunk_id"]
            or row["span_json"] is None
            or row["metadata_json"] is None
            or row["source_pages_json"] is None
            or hashlib.sha256(content.encode("utf-8")).hexdigest()
            != content_sha256
        ):
            return ()
        try:
            span = json.loads(str(row["span_json"]))
            metadata = json.loads(str(row["metadata_json"]))
            source_pages = json.loads(str(row["source_pages_json"]))
        except (TypeError, ValueError, json.JSONDecodeError):
            return ()
        if (
            not isinstance(span, Mapping)
            or not isinstance(metadata, Mapping)
            or not isinstance(source_pages, list)
        ):
            return ()
        writes.append(
            ChunkWrite(
                seq=expected_seq,
                loc=str(row["loc"] or ""),
                content=content,
                producer_chunk_id=str(row["producer_chunk_id"]),
                span_json=str(row["span_json"]),
                metadata_json=str(row["metadata_json"]),
                content_sha256=content_sha256,
                chunk_contract_version=chunk_contract_version,
                source_pages_json=str(row["source_pages_json"]),
            )
        )
    return tuple(writes)


def unique_dedupe_key(
    conn: sqlite3.Connection,
    content_hash: str,
    document_id: str,
) -> str:
    row = conn.execute(
        "SELECT id FROM documents WHERE hash=?",
        (content_hash,),
    ).fetchone()
    return (
        content_hash
        if row is None or row["id"] == document_id
        else f"{content_hash}:{document_id}"
    )


def stored_processing_differs(
    conn: sqlite3.Connection,
    *,
    document_id: str,
    processor_fingerprint: str | None,
    chunker_fingerprint: str,
    typed: bool,
    processing_status: str,
    diagnostics_json: str | None,
    page_manifest_json: str | None,
    page_manifest_sha256: str | None,
    physical_page_count: int | None,
    source_elements_json: str | None,
    source_elements_sha256: str | None,
    chunk_writes: Sequence[ChunkWrite],
) -> bool:
    if processor_fingerprint is None and not typed:
        return False
    current = conn.execute(
        "SELECT v.processing_status, v.diagnostics_json, v.page_manifest_json, "
        "v.page_manifest_sha256, v.physical_page_count, "
        "v.source_elements_json, v.source_elements_sha256 "
        "FROM documents AS d LEFT JOIN document_versions AS v "
        "ON v.id=d.current_version_id WHERE d.id=?",
        (document_id,),
    ).fetchone()
    if (
        current is None
        or current["processing_status"] != processing_status
        or current["diagnostics_json"] != diagnostics_json
        or current["page_manifest_json"] != page_manifest_json
        or current["page_manifest_sha256"] != page_manifest_sha256
        or current["physical_page_count"] != physical_page_count
        or current["source_elements_json"] != source_elements_json
        or current["source_elements_sha256"] != source_elements_sha256
    ):
        return True
    rows = conn.execute(
        "SELECT c.seq, c.loc, c.content, c.processor_fingerprint, "
        "c.chunker_fingerprint, c.producer_chunk_id, c.span_json, "
        "c.metadata_json, c.content_sha256, c.chunk_contract_version, "
        "c.source_pages_json FROM doc_chunks AS c "
        "JOIN documents AS d ON d.id=c.doc_id "
        "WHERE c.doc_id=? AND c.source_version_id=d.current_version_id "
        "ORDER BY c.seq, c.id",
        (document_id,),
    ).fetchall()
    if len(rows) != len(chunk_writes):
        return True
    for row, expected in zip(rows, chunk_writes, strict=True):
        if (
            row["seq"] != expected.seq
            or row["loc"] != expected.loc
            or row["content"] != expected.content
            or row["processor_fingerprint"] != processor_fingerprint
            or row["chunker_fingerprint"] != chunker_fingerprint
            or row["producer_chunk_id"] != expected.producer_chunk_id
            or row["span_json"] != expected.span_json
            or row["metadata_json"] != expected.metadata_json
            or row["content_sha256"] != expected.content_sha256
            or row["chunk_contract_version"] != expected.chunk_contract_version
            or row["source_pages_json"] != expected.source_pages_json
        ):
            return True
    return False


def update_source_provenance(
    conn: sqlite3.Connection,
    *,
    document_id: str,
    source_sha256: str,
    source_size: int,
    source_mtime_ns: int,
    verified_at: str,
    file_id: str | None,
    file_version_id: str | None,
) -> None:
    """刷新文档的最新来源观察，不改写已发布处理版本的捕获记录。"""

    if file_id is not None:
        current = conn.execute(
            "SELECT d.file_id, v.file_version_id FROM documents AS d "
            "JOIN document_versions AS v ON v.id=d.current_version_id WHERE d.id=?",
            (document_id,),
        ).fetchone()
        if current is None or tuple(current) != (file_id, file_version_id):
            raise ValueError("source observation cannot rebind a document version")
    conn.execute(
        "UPDATE documents SET source_sha256=?, source_size=?, source_mtime_ns=?, "
        "source_verified_at=?, freshness_status='verified_current' "
        "WHERE id=?",
        (
            source_sha256,
            source_size,
            source_mtime_ns,
            verified_at,
            document_id,
        ),
    )


def activate_document_version(
    conn: sqlite3.Connection,
    *,
    version_id: str,
    document_id: str,
    content_hash: str,
    source_sha256: str | None,
    source_size: int | None,
    source_mtime_ns: int | None,
    captured_at: str,
    processing_status: str,
    diagnostics_json: str | None,
    processor_fingerprint: str | None,
    chunker_fingerprint: str,
    chunk_contract_version: int | None,
    page_manifest_json: str | None,
    page_manifest_sha256: str | None,
    physical_page_count: int | None,
    file_version_id: str | None,
    source_elements_json: str | None,
    source_elements_sha256: str | None,
) -> None:
    validate_source_elements_artifact(
        source_elements_json=source_elements_json,
        source_elements_sha256=source_elements_sha256,
        content_hash=content_hash,
    )
    current = conn.execute(
        "SELECT current_version_id FROM documents WHERE id=?",
        (document_id,),
    ).fetchone()
    previous_id = current["current_version_id"] if current else None
    last = conn.execute(
        "SELECT COALESCE(MAX(version_number), 0) AS version_number "
        "FROM document_versions WHERE doc_id=?",
        (document_id,),
    ).fetchone()
    if previous_id:
        conn.execute(
            "UPDATE document_versions SET status='superseded' "
            "WHERE id=? AND status='active'",
            (previous_id,),
        )
    values = (
        version_id,
        document_id,
        int(last["version_number"] or 0) + 1,
        source_sha256,
        content_hash,
        source_size,
        source_mtime_ns,
        "active",
        previous_id,
        captured_at,
        captured_at,
        processing_status,
        diagnostics_json,
        processor_fingerprint,
        chunker_fingerprint,
        chunk_contract_version,
        page_manifest_json,
        page_manifest_sha256,
        physical_page_count,
        file_version_id,
        source_elements_json,
        source_elements_sha256,
    )
    conn.execute(
        "INSERT INTO document_versions (id, doc_id, version_number, "
        "source_sha256, content_hash, source_size, source_mtime_ns, status, "
        "supersedes_version_id, captured_at, activated_at, processing_status, "
        "diagnostics_json, processor_fingerprint, chunker_fingerprint, "
        "chunk_contract_version, page_manifest_json, page_manifest_sha256, "
        "physical_page_count, file_version_id, source_elements_json, "
        "source_elements_sha256) VALUES ("
        + ",".join("?" for _ in values)
        + ")",
        values,
    )
    conn.execute(
        "UPDATE documents SET current_version_id=? WHERE id=?",
        (version_id, document_id),
    )


def replace_document_contents(
    conn: sqlite3.Connection,
    *,
    document_id: str,
    path: str,
    mime: str,
    content_hash: str,
    chunk_writes: Sequence[ChunkWrite],
    source_session: str | None,
    file_id: str | None,
    source_sha256: str,
    source_size: int,
    source_mtime_ns: int,
    verified_at: str,
    version_id: str,
    processor_fingerprint: str | None,
    chunker_fingerprint: str,
    processing_status: str,
    diagnostics_json: str | None,
    chunk_contract_version: int | None,
    page_manifest_json: str | None,
    page_manifest_sha256: str | None,
    physical_page_count: int | None,
    file_version_id: str | None,
    source_elements_json: str | None,
    source_elements_sha256: str | None,
    chunk_ids: Sequence[str],
) -> None:
    conn.execute("DELETE FROM doc_chunks WHERE doc_id=?", (document_id,))
    values = (
        path,
        mime,
        unique_dedupe_key(conn, content_hash, document_id),
        content_hash,
        source_session,
        len(chunk_writes),
        verified_at,
        source_sha256,
        source_size,
        source_mtime_ns,
        verified_at,
    )
    if file_id is None:
        conn.execute(
            "UPDATE documents SET path=?, mime=?, hash=?, content_hash=?, "
            "source_session=?, n_chunks=?, added_at=?, source_sha256=?, "
            "source_size=?, source_mtime_ns=?, source_verified_at=?, "
            "freshness_status='verified_current' WHERE id=?",
            (*values, document_id),
        )
    else:
        conn.execute(
            "UPDATE documents SET path=?, mime=?, hash=?, content_hash=?, "
            "source_session=?, n_chunks=?, added_at=?, source_sha256=?, "
            "source_size=?, source_mtime_ns=?, source_verified_at=?, "
            "freshness_status='verified_current', file_id=? WHERE id=?",
            (*values, file_id, document_id),
        )
    activate_document_version(
        conn,
        version_id=version_id,
        document_id=document_id,
        content_hash=content_hash,
        source_sha256=source_sha256,
        source_size=source_size,
        source_mtime_ns=source_mtime_ns,
        captured_at=verified_at,
        processing_status=processing_status,
        diagnostics_json=diagnostics_json,
        processor_fingerprint=processor_fingerprint,
        chunker_fingerprint=chunker_fingerprint,
        chunk_contract_version=chunk_contract_version,
        page_manifest_json=page_manifest_json,
        page_manifest_sha256=page_manifest_sha256,
        physical_page_count=physical_page_count,
        file_version_id=file_version_id,
        source_elements_json=source_elements_json,
        source_elements_sha256=source_elements_sha256,
    )
    insert_chunk_generation(
        conn,
        document_id=document_id,
        version_id=version_id,
        chunk_writes=chunk_writes,
        processor_fingerprint=processor_fingerprint,
        chunker_fingerprint=chunker_fingerprint,
        chunk_ids=chunk_ids,
    )


def insert_chunk_generation(
    conn: sqlite3.Connection,
    *,
    document_id: str,
    version_id: str,
    chunk_writes: Sequence[ChunkWrite],
    processor_fingerprint: str | None,
    chunker_fingerprint: str,
    chunk_ids: Sequence[str],
) -> None:
    if len(chunk_ids) != len(chunk_writes):
        raise ValueError("chunk_ids must match chunk_writes")
    for chunk_id, chunk in zip(chunk_ids, chunk_writes, strict=True):
        conn.execute(
            "INSERT INTO doc_chunks (id, doc_id, seq, loc, content, "
            "source_version_id, processor_fingerprint, chunker_fingerprint, "
            "producer_chunk_id, span_json, metadata_json, content_sha256, "
            "chunk_contract_version, source_pages_json) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                chunk_id,
                document_id,
                chunk.seq,
                chunk.loc,
                chunk.content,
                version_id,
                processor_fingerprint,
                chunker_fingerprint,
                chunk.producer_chunk_id,
                chunk.span_json,
                chunk.metadata_json,
                chunk.content_sha256,
                chunk.chunk_contract_version,
                chunk.source_pages_json,
            ),
        )
