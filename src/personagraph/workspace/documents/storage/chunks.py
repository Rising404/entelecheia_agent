"""显式 SQLite 连接上的当前 Document 块与页面查询原语。"""

from __future__ import annotations

import sqlite3


def get_file_inspection_rows(conn, *, document_id: str, document_version_id: str, include_source_text: bool):
    """读取分块目录和切块前文本快照；调用方负责当前文件/会话版本资格。"""
    rows = tuple(conn.execute(
        "SELECT id, seq, loc, metadata_json, source_pages_json FROM doc_chunks "
        "WHERE doc_id=? AND source_version_id=? ORDER BY seq, id",
        (document_id, document_version_id),
    ).fetchall())
    source = conn.execute(
        "SELECT source_elements_json, source_elements_sha256 FROM document_versions "
        "WHERE doc_id=? AND id=?", (document_id, document_version_id),
    ).fetchone() if include_source_text else None
    return rows, source


def get_document_file_identity_row(conn: sqlite3.Connection, document_id: str):
    return conn.execute(
        "SELECT d.file_id, v.file_version_id FROM documents AS d "
        "JOIN document_versions AS v ON v.id=d.current_version_id AND v.doc_id=d.id "
        "WHERE d.id=?", (document_id,),
    ).fetchone()


def get_current_file_document_row(
    conn: sqlite3.Connection, *, file_id: str, file_version_id: str,
) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT d.id AS document_id, v.id AS document_version_id, "
        "v.source_sha256, d.n_chunks, v.processing_status, v.physical_page_count, d.added_at "
        "FROM files AS f JOIN file_versions AS fv ON fv.id=f.current_version_id "
        "AND fv.file_id=f.id JOIN documents AS d ON d.file_id=f.id "
        "JOIN document_versions AS v ON v.id=d.current_version_id AND v.doc_id=d.id "
        "AND v.file_version_id=fv.id "
        "WHERE f.id=? AND fv.id=? AND v.source_sha256=fv.content_sha256 "
        "AND d.source_sha256=v.source_sha256",
        (file_id, file_version_id),
    ).fetchone()


def get_current_file_chunk_row(
    conn: sqlite3.Connection, *, document_id: str, document_version_id: str,
    chunk_id: str | None, sequence: int | None,
) -> sqlite3.Row | None:
    selector = "c.id=?" if chunk_id is not None else "c.seq=?"
    return conn.execute(
        "SELECT c.* FROM doc_chunks AS c JOIN documents AS d ON d.id=c.doc_id "
        "AND d.current_version_id=c.source_version_id "
        "WHERE c.doc_id=? AND c.source_version_id=? AND " + selector,
        (document_id, document_version_id, chunk_id if chunk_id is not None else sequence),
    ).fetchone()


def read_current_chunks(
    conn: sqlite3.Connection,
    document_id: str,
    *,
    sequence_from: int,
    maximum_chunks: int,
    expected_version_id: str | None,
) -> tuple[sqlite3.Row, ...]:
    query = (
        "SELECT c.*, v.processing_status, v.diagnostics_json "
        "FROM doc_chunks AS c JOIN documents AS d ON d.id=c.doc_id "
        "JOIN document_versions AS v ON v.id=d.current_version_id "
        "AND v.id=c.source_version_id "
    )
    if expected_version_id is None:
        parameters: tuple[object, ...] = (
            document_id,
            sequence_from,
            maximum_chunks,
        )
        suffix = "WHERE c.doc_id=? AND c.seq>=? ORDER BY c.seq LIMIT ?"
    else:
        parameters = (
            document_id,
            expected_version_id,
            sequence_from,
            maximum_chunks,
        )
        suffix = (
            "WHERE c.doc_id=? AND c.source_version_id=? AND c.seq>=? "
            "ORDER BY c.seq LIMIT ?"
        )
    return tuple(conn.execute(query + suffix, parameters).fetchall())


def get_current_chunk(
    conn: sqlite3.Connection,
    chunk_id: str,
    *,
    include_file_origin: bool,
) -> sqlite3.Row | None:
    file_origin = (
        "(SELECT origin FROM files WHERE id=d.file_id)"
        if include_file_origin
        else "NULL"
    )
    return conn.execute(
        "SELECT c.*, d.title, d.path, d.current_version_id, "
        "d.source_mtime_ns, d.added_at, d.file_id, v.file_version_id, "
        f"{file_origin} AS file_origin, "
        "v.processing_status, v.diagnostics_json "
        "FROM doc_chunks AS c JOIN documents AS d ON d.id=c.doc_id "
        "JOIN document_versions AS v ON v.id=d.current_version_id "
        "AND v.id=c.source_version_id WHERE c.id=?",
        (chunk_id,),
    ).fetchone()


def get_current_chunk_file_identity(conn: sqlite3.Connection, chunk_id: str) -> sqlite3.Row | None:
    """Current native ID lookup, deliberately excluding the chunk body."""
    return conn.execute(
        "SELECT d.file_id,v.file_version_id FROM doc_chunks AS c "
        "JOIN documents AS d ON d.id=c.doc_id "
        "JOIN document_versions AS v ON v.id=d.current_version_id "
        "AND v.id=c.source_version_id WHERE c.id=?",
        (chunk_id,),
    ).fetchone()


def get_current_typed_chunk(
    conn: sqlite3.Connection,
    document_id: str,
    producer_chunk_id: str,
    expected_version_id: str,
    *,
    include_file_origin: bool,
) -> sqlite3.Row | None:
    file_origin = (
        "(SELECT origin FROM files WHERE id=d.file_id)"
        if include_file_origin
        else "NULL"
    )
    return conn.execute(
        "SELECT c.*, d.title, d.path, d.current_version_id, "
        "d.source_mtime_ns, d.added_at, d.file_id, v.file_version_id, "
        f"{file_origin} AS file_origin, "
        "v.processing_status, v.diagnostics_json "
        "FROM doc_chunks AS c "
        "JOIN documents AS d ON d.id=c.doc_id "
        "JOIN document_versions AS v ON v.id=d.current_version_id "
        "AND v.id=c.source_version_id "
        "WHERE c.doc_id=? AND c.producer_chunk_id=? "
        "AND c.source_version_id=? AND d.current_version_id=?",
        (
            document_id,
            producer_chunk_id,
            expected_version_id,
            expected_version_id,
        ),
    ).fetchone()


def get_current_page_authority_rows(
    conn: sqlite3.Connection,
    document_id: str,
    expected_version_id: str,
) -> tuple[sqlite3.Row | None, tuple[sqlite3.Row, ...]]:
    version = conn.execute(
        "SELECT d.id AS document_id, d.current_version_id, "
        "d.source_sha256 AS document_source_sha256, "
        "v.id AS document_version_id, v.source_sha256 AS version_source_sha256, "
        "v.processing_status, v.diagnostics_json, v.page_manifest_json, "
        "v.page_manifest_sha256, v.physical_page_count "
        "FROM documents AS d "
        "JOIN document_versions AS v ON v.id=d.current_version_id "
        "WHERE d.id=? AND d.current_version_id=? AND v.id=?",
        (document_id, expected_version_id, expected_version_id),
    ).fetchone()
    if version is None:
        return None, ()
    rows = tuple(
        conn.execute(
            "SELECT c.id, c.producer_chunk_id, c.seq, c.metadata_json, "
            "c.source_pages_json, c.content, c.content_sha256, "
            "c.chunk_contract_version FROM doc_chunks AS c "
            "WHERE c.doc_id=? AND c.source_version_id=? ORDER BY c.seq, c.id",
            (document_id, expected_version_id),
        ).fetchall()
    )
    return version, rows


def get_current_resource_rows(
    conn: sqlite3.Connection,
    document_id: str,
    *,
    start_sequence: int,
    maximum_chunks_with_sentinel: int,
) -> tuple[sqlite3.Row | None, tuple[sqlite3.Row, ...], sqlite3.Row | None]:
    document = conn.execute(
        "SELECT d.id AS document_id, d.path, d.mime, d.n_chunks, "
        "d.current_version_id, d.source_sha256 AS document_source_sha256, "
        "v.id AS document_version_id, "
        "v.source_sha256 AS version_source_sha256, "
        "v.processing_status, v.diagnostics_json, "
        "v.chunk_contract_version, v.page_manifest_json, "
        "v.page_manifest_sha256, v.physical_page_count "
        "FROM documents AS d "
        "JOIN document_versions AS v ON v.id=d.current_version_id "
        "WHERE d.id=?",
        (document_id,),
    ).fetchone()
    if document is None:
        return None, (), None
    version_id = str(document["document_version_id"])
    rows = tuple(
        conn.execute(
            "SELECT id, producer_chunk_id, seq, loc, content, "
            "source_version_id, content_sha256, chunk_contract_version, "
            "source_pages_json FROM doc_chunks "
            "WHERE doc_id=? AND source_version_id=? AND seq>=? "
            "ORDER BY seq, id LIMIT ?",
            (
                document_id,
                version_id,
                start_sequence,
                maximum_chunks_with_sentinel,
            ),
        ).fetchall()
    )
    extent = conn.execute(
        "SELECT COUNT(*) AS chunk_count, "
        "COUNT(DISTINCT seq) AS distinct_sequence_count, "
        "MIN(seq) AS minimum_sequence, MAX(seq) AS maximum_sequence "
        "FROM doc_chunks WHERE doc_id=? AND source_version_id=?",
        (document_id, version_id),
    ).fetchone()
    return document, rows, extent


def list_current_chunks(
    conn: sqlite3.Connection,
    document_id: str,
) -> tuple[sqlite3.Row, ...]:
    row = conn.execute(
        "SELECT current_version_id FROM documents WHERE id=?",
        (document_id,),
    ).fetchone()
    if row is None or not row["current_version_id"]:
        return ()
    return tuple(
        conn.execute(
            "SELECT c.*, v.processing_status, v.diagnostics_json "
            "FROM doc_chunks AS c JOIN document_versions AS v ON v.id=? "
            "AND v.id=c.source_version_id WHERE c.doc_id=? ORDER BY c.seq, c.id",
            (row["current_version_id"], document_id),
        ).fetchall()
    )


def get_index_binding_rows(
    conn: sqlite3.Connection,
    document_ids: tuple[str, ...],
    *,
    maximum_rows: int,
) -> tuple[sqlite3.Row, ...]:
    if not document_ids or maximum_rows <= 0:
        return ()
    placeholders = ",".join("?" for _ in document_ids)
    return tuple(
        conn.execute(
            "SELECT c.doc_id, c.id, c.source_version_id, c.producer_chunk_id, "
            "c.content, c.content_sha256 FROM doc_chunks AS c "
            "JOIN documents AS d ON d.id=c.doc_id "
            f"WHERE c.doc_id IN ({placeholders}) "
            "AND c.source_version_id=d.current_version_id "
            "ORDER BY c.doc_id, c.seq, c.id LIMIT ?",
            (*document_ids, maximum_rows),
        ).fetchall()
    )


def get_current_version_rows(
    conn: sqlite3.Connection,
    document_ids: tuple[str, ...],
) -> tuple[sqlite3.Row, ...]:
    if not document_ids:
        return ()
    placeholders = ",".join("?" for _ in document_ids)
    return tuple(
        conn.execute(
            "SELECT id, current_version_id FROM documents "
            f"WHERE id IN ({placeholders})",
            document_ids,
        ).fetchall()
    )
