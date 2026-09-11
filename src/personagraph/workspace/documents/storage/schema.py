"""Workspace Document authority 对共享项目数据库贡献的稳定 DDL。"""

from __future__ import annotations

import sqlite3


DOCUMENT_REQUIRED_TABLES = frozenset(
    {
        "documents",
        "document_versions",
        "doc_chunks",
    }
)


_SOURCE_ELEMENTS_TRIGGERS = (
    """
    CREATE TRIGGER trg_document_versions_source_elements_insert_shape
    BEFORE INSERT ON document_versions
    WHEN (
        (NEW.source_elements_json IS NULL) != (NEW.source_elements_sha256 IS NULL)
        OR (
            NEW.source_elements_json IS NOT NULL
            AND (
                CASE
                    WHEN json_valid(NEW.source_elements_json) != 1 THEN 1
                    WHEN json_type(NEW.source_elements_json) != 'array' THEN 1
                    ELSE 0
                END
                OR length(NEW.source_elements_sha256) != 64
                OR NEW.source_elements_sha256 GLOB '*[^0-9a-f]*'
            )
        )
    )
    BEGIN
        SELECT RAISE(ABORT, 'invalid document source elements lineage');
    END
    """,
    """
    CREATE TRIGGER trg_document_versions_source_elements_immutable
    BEFORE UPDATE OF source_elements_json, source_elements_sha256
    ON document_versions
    WHEN NEW.source_elements_json IS NOT OLD.source_elements_json
      OR NEW.source_elements_sha256 IS NOT OLD.source_elements_sha256
    BEGIN
        SELECT RAISE(ABORT, 'document source elements lineage is immutable');
    END
    """,
)


_STATEMENTS = (
    """
    CREATE TABLE documents (
        id TEXT PRIMARY KEY,
        path TEXT,
        title TEXT,
        mime TEXT,
        hash TEXT UNIQUE,
        summary TEXT,
        tags TEXT,
        task_id TEXT,
        source_session TEXT,
        n_chunks INTEGER,
        added_at TEXT,
        content_hash TEXT,
        source_sha256 TEXT,
        source_size INTEGER,
        source_mtime_ns INTEGER,
        source_verified_at TEXT,
        freshness_status TEXT DEFAULT 'legacy_unverified',
        current_version_id TEXT,
        file_id TEXT
    )
    """,
    """
    CREATE TABLE document_versions (
        id TEXT PRIMARY KEY,
        doc_id TEXT NOT NULL,
        version_number INTEGER NOT NULL,
        source_sha256 TEXT,
        content_hash TEXT,
        source_size INTEGER,
        source_mtime_ns INTEGER,
        status TEXT NOT NULL,
        supersedes_version_id TEXT,
        captured_at TEXT NOT NULL,
        activated_at TEXT,
        processing_status TEXT NOT NULL DEFAULT 'legacy_unknown',
        diagnostics_json TEXT,
        processor_fingerprint TEXT,
        chunker_fingerprint TEXT,
        chunk_contract_version INTEGER,
        page_manifest_json TEXT,
        page_manifest_sha256 TEXT,
        physical_page_count INTEGER,
        file_version_id TEXT,
        source_elements_json TEXT,
        source_elements_sha256 TEXT,
        UNIQUE(doc_id, version_number)
    )
    """,
    """
    CREATE TABLE doc_chunks (
        id TEXT PRIMARY KEY,
        doc_id TEXT,
        seq INTEGER,
        loc TEXT,
        content TEXT,
        source_version_id TEXT,
        processor_fingerprint TEXT,
        chunker_fingerprint TEXT,
        producer_chunk_id TEXT,
        span_json TEXT,
        metadata_json TEXT,
        content_sha256 TEXT,
        chunk_contract_version INTEGER,
        source_pages_json TEXT
    )
    """,
    "CREATE INDEX idx_doc_chunks ON doc_chunks(doc_id, seq)",
    "CREATE INDEX idx_document_versions_current "
    "ON document_versions(doc_id, status, version_number DESC)",
    "CREATE INDEX idx_doc_chunks_processor "
    "ON doc_chunks(doc_id, processor_fingerprint, chunker_fingerprint)",
    "CREATE UNIQUE INDEX idx_doc_chunks_producer_generation "
    "ON doc_chunks(doc_id, source_version_id, producer_chunk_id) "
    "WHERE producer_chunk_id IS NOT NULL",
    "CREATE INDEX idx_document_versions_page_manifest "
    "ON document_versions(doc_id, page_manifest_sha256) "
    "WHERE page_manifest_sha256 IS NOT NULL",
    "CREATE UNIQUE INDEX idx_doc_chunks_sequence_generation "
    "ON doc_chunks(doc_id, source_version_id, seq) "
    "WHERE source_version_id IS NOT NULL",
    "CREATE UNIQUE INDEX idx_documents_file_id "
    "ON documents(file_id) WHERE file_id IS NOT NULL",
    "CREATE INDEX idx_document_versions_file_version "
    "ON document_versions(file_version_id) WHERE file_version_id IS NOT NULL",
    """
    CREATE TRIGGER trg_document_versions_page_manifest_insert_shape
    BEFORE INSERT ON document_versions
    WHEN (
        (NEW.page_manifest_json IS NULL) != (NEW.page_manifest_sha256 IS NULL)
        OR (NEW.page_manifest_json IS NULL) != (NEW.physical_page_count IS NULL)
        OR (
            NEW.page_manifest_sha256 IS NOT NULL
            AND (
                length(NEW.page_manifest_sha256) != 64
                OR NEW.page_manifest_sha256 GLOB '*[^0-9a-f]*'
                OR NEW.physical_page_count < 0
            )
        )
    )
    BEGIN
        SELECT RAISE(ABORT, 'invalid document page manifest lineage');
    END
    """,
    """
    CREATE TRIGGER trg_document_versions_page_manifest_immutable
    BEFORE UPDATE OF page_manifest_json, page_manifest_sha256, physical_page_count
    ON document_versions
    WHEN NEW.page_manifest_json IS NOT OLD.page_manifest_json
      OR NEW.page_manifest_sha256 IS NOT OLD.page_manifest_sha256
      OR NEW.physical_page_count IS NOT OLD.physical_page_count
    BEGIN
        SELECT RAISE(ABORT, 'document page manifest lineage is immutable');
    END
    """,
    *_SOURCE_ELEMENTS_TRIGGERS,
)


_FILE_LINKAGE_TRIGGERS = (
    """
    CREATE TRIGGER IF NOT EXISTS trg_documents_require_file
    BEFORE INSERT ON documents
    WHEN NEW.file_id IS NULL
      OR NOT EXISTS (SELECT 1 FROM files WHERE id=NEW.file_id)
    BEGIN
        SELECT RAISE(ABORT, 'project document requires a registered file_id');
    END
    """,
    """
    CREATE TRIGGER IF NOT EXISTS trg_documents_file_immutable
    BEFORE UPDATE OF file_id ON documents
    WHEN NEW.file_id IS NULL
      OR NEW.file_id IS NOT OLD.file_id
      OR NOT EXISTS (SELECT 1 FROM files WHERE id=NEW.file_id)
    BEGIN
        SELECT RAISE(ABORT, 'project document file_id is immutable');
    END
    """,
    """
    CREATE TRIGGER IF NOT EXISTS trg_document_versions_require_file_version
    BEFORE INSERT ON document_versions
    WHEN NEW.file_version_id IS NULL
      OR NOT EXISTS (
          SELECT 1 FROM documents AS document
          JOIN file_versions AS version
            ON version.id=NEW.file_version_id
           AND version.file_id=document.file_id
          WHERE document.id=NEW.doc_id
            AND version.content_sha256=NEW.source_sha256
            AND version.size_bytes=NEW.source_size
      )
    BEGIN
        SELECT RAISE(ABORT, 'document version requires its exact registered file version');
    END
    """,
    """
    CREATE TRIGGER IF NOT EXISTS trg_document_versions_file_version_update
    BEFORE UPDATE OF doc_id, file_version_id, source_sha256, source_size, source_mtime_ns
    ON document_versions
    WHEN NEW.doc_id IS NOT OLD.doc_id
      OR NEW.file_version_id IS NOT OLD.file_version_id
      OR NEW.source_sha256 IS NOT OLD.source_sha256
      OR NEW.source_size IS NOT OLD.source_size
      OR NEW.source_mtime_ns IS NOT OLD.source_mtime_ns
      OR NEW.file_version_id IS NULL
      OR NOT EXISTS (
          SELECT 1 FROM documents AS document
          JOIN file_versions AS version
            ON version.id=NEW.file_version_id
           AND version.file_id=document.file_id
          WHERE document.id=NEW.doc_id
            AND version.content_sha256=NEW.source_sha256
            AND version.size_bytes=NEW.source_size
      )
    BEGIN
        SELECT RAISE(ABORT, 'document version file linkage is immutable and exact');
    END
    """,
    """
    CREATE TRIGGER IF NOT EXISTS trg_files_restrict_linked_delete
    BEFORE DELETE ON files
    WHEN EXISTS (SELECT 1 FROM documents WHERE file_id=OLD.id)
    BEGIN
        SELECT RAISE(ABORT, 'cannot delete a file linked to a project document');
    END
    """,
    """
    CREATE TRIGGER IF NOT EXISTS trg_file_versions_restrict_linked_delete
    BEFORE DELETE ON file_versions
    WHEN EXISTS (SELECT 1 FROM document_versions WHERE file_version_id=OLD.id)
    BEGIN
        SELECT RAISE(ABORT, 'cannot delete a file version linked to a document version');
    END
    """,
)


class DocumentSchemaContributionError(RuntimeError):
    """Document contribution 缺失、损坏或仍含退役关联。"""


def initialize_document_schema(conn: sqlite3.Connection) -> None:
    for statement in _STATEMENTS:
        conn.execute(statement)


def validate_document_schema(conn: sqlite3.Connection) -> None:
    present = _application_objects(conn)
    missing = sorted(DOCUMENT_REQUIRED_TABLES - present)
    if missing:
        raise DocumentSchemaContributionError(
            "workspace document schema is incomplete: " + ", ".join(missing)
        )


def validate_document_source_elements_schema(conn: sqlite3.Connection) -> None:
    """验证当前 schema 已安装精确源元素列及其不可变围栏。"""

    columns = {
        str(row[1]) for row in conn.execute("PRAGMA table_info(document_versions)")
    }
    missing_columns = sorted(
        {"source_elements_json", "source_elements_sha256"} - columns
    )
    trigger_names = {
        str(row[0])
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='trigger'"
        ).fetchall()
    }
    missing_triggers = sorted(
        {
            "trg_document_versions_source_elements_insert_shape",
            "trg_document_versions_source_elements_immutable",
        }
        - trigger_names
    )
    if missing_columns or missing_triggers:
        details = [
            *(f"column:{name}" for name in missing_columns),
            *(f"trigger:{name}" for name in missing_triggers),
        ]
        raise DocumentSchemaContributionError(
            "workspace document source elements schema is incomplete: "
            + ", ".join(details)
        )


def ensure_document_file_linkage(conn: sqlite3.Connection) -> None:
    unlinked_documents = int(
        conn.execute("SELECT COUNT(*) FROM documents WHERE file_id IS NULL").fetchone()[0]
    )
    unlinked_versions = int(
        conn.execute(
            "SELECT COUNT(*) FROM document_versions WHERE file_version_id IS NULL"
        ).fetchone()[0]
    )
    invalid_documents = int(
        conn.execute(
            "SELECT COUNT(*) FROM documents AS document "
            "LEFT JOIN files AS file ON file.id=document.file_id "
            "WHERE file.id IS NULL"
        ).fetchone()[0]
    )
    invalid_versions = int(
        conn.execute(
            "SELECT COUNT(*) FROM document_versions AS document_version "
            "JOIN documents AS document ON document.id=document_version.doc_id "
            "LEFT JOIN file_versions AS file_version "
            "ON file_version.id=document_version.file_version_id "
            "AND file_version.file_id=document.file_id "
            "AND file_version.content_sha256=document_version.source_sha256 "
            "AND file_version.size_bytes=document_version.source_size "
            "WHERE file_version.id IS NULL"
        ).fetchone()[0]
    )
    if unlinked_documents or unlinked_versions or invalid_documents or invalid_versions:
        raise DocumentSchemaContributionError(
            "project documents contain retired or invalid file/version linkage; "
            "re-ingest them into the project file ledger"
        )
    for statement in _FILE_LINKAGE_TRIGGERS:
        conn.execute(statement)


def migrate_document_content_version_schema(conn: sqlite3.Connection) -> None:
    """在项目 schema 迁移事务中固定处理版本来源，移除历史 mtime 的身份约束。"""

    for name in (
        "trg_document_versions_require_file_version",
        "trg_document_versions_file_version_update",
    ):
        conn.execute(f"DROP TRIGGER IF EXISTS {name}")
    ensure_document_file_linkage(conn)


def migrate_document_source_elements_schema(conn: sqlite3.Connection) -> None:
    """追加切块前文本快照列；历史版本保持不可推断的 ``NULL/NULL``。"""

    columns = {
        str(row[1]) for row in conn.execute("PRAGMA table_info(document_versions)")
    }
    expected = {"source_elements_json", "source_elements_sha256"}
    present = expected & columns
    if present and present != expected:
        raise DocumentSchemaContributionError(
            "document source elements migration found a partial column pair"
        )
    if not present:
        conn.execute(
            "ALTER TABLE document_versions ADD COLUMN source_elements_json TEXT"
        )
        conn.execute(
            "ALTER TABLE document_versions ADD COLUMN source_elements_sha256 TEXT"
        )
    for statement in _SOURCE_ELEMENTS_TRIGGERS:
        trigger_statement = statement.replace(
            "CREATE TRIGGER ", "CREATE TRIGGER IF NOT EXISTS ", 1
        )
        conn.execute(trigger_statement)


def _application_objects(conn: sqlite3.Connection) -> frozenset[str]:
    return frozenset(
        str(row[0])
        for row in conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE name NOT LIKE 'sqlite_%' "
            "AND type IN ('table', 'view', 'trigger')"
        ).fetchall()
    )
