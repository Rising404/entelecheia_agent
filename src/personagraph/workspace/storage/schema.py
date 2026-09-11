"""共享项目数据库的 contribution composition 与绑定身份。"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import sqlite3
import uuid

from ..documents.storage.schema import (
    DocumentSchemaContributionError,
    ensure_document_file_linkage,
    initialize_document_schema,
    migrate_document_content_version_schema,
    migrate_document_source_elements_schema,
    validate_document_schema,
    validate_document_source_elements_schema,
)
from ..files.storage.schema import (
    initialize_file_schema,
    migrate_file_observation_schema,
    validate_file_schema,
)
from ..ingestion.storage.schema import (
    IngestionSchemaContributionError,
    initialize_ingestion_schema,
    migrate_shared_ingestion_schema,
    validate_ingestion_schema,
)
from ..pictures.storage.schema import (
    PictureSchemaError,
    initialize_picture_schema,
    migrate_picture_observation_question_schema,
    validate_picture_schema,
)


SCHEMA_NAME = "project_documents"
SCHEMA_VERSION = 9
FILE_ORIGINS = ("user_upload", "workspace_existing", "agent_output")


class DocumentSchemaError(RuntimeError):
    """数据库不符合共享 Workspace 模式契约。"""


_SCHEMA_META_SQL = """
CREATE TABLE schema_meta (
    schema_name TEXT PRIMARY KEY CHECK(schema_name = 'project_documents'),
    schema_version INTEGER NOT NULL CHECK(schema_version >= 1),
    project_id TEXT NOT NULL CHECK(length(project_id) > 0),
    project_root TEXT NOT NULL CHECK(length(project_root) > 0),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
)
"""


_RETRIEVAL_SOURCE_TYPE_CHECK_SQL = (
    "CHECK(source_type IN ('document', 'picture'))"
)


def _retrieval_units_table_sql(table_name: str) -> str:
    if table_name not in {"retrieval_units", "retrieval_units_v5"}:
        raise ValueError("unsupported retrieval units table name")
    return f"""
    CREATE TABLE {table_name} (
        unit_id INTEGER PRIMARY KEY,
        data_version_id TEXT NOT NULL,
        source_type TEXT NOT NULL {_RETRIEVAL_SOURCE_TYPE_CHECK_SQL},
        source_unit_id TEXT NOT NULL,
        source_revision TEXT NOT NULL,
        indexed_content_hash TEXT NOT NULL CHECK(
            length(indexed_content_hash) = 64
            AND indexed_content_hash NOT GLOB '*[^0-9a-f]*'
        ),
        retrieval_status TEXT NOT NULL CHECK(retrieval_status IN ('active', 'trashed')),
        index_state TEXT NOT NULL CHECK(index_state IN ('pending', 'ready', 'failed')),
        scope_json TEXT NOT NULL DEFAULT '{{}}' CHECK(json_valid(scope_json)),
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        FOREIGN KEY(data_version_id) REFERENCES retrieval_data_versions(id) ON DELETE RESTRICT,
        UNIQUE(
            data_version_id, source_type, source_unit_id,
            source_revision, indexed_content_hash
        )
    )
    """


def _retrieval_update_outbox_table_sql(table_name: str) -> str:
    if table_name not in {
        "retrieval_update_outbox",
        "retrieval_update_outbox_v5",
    }:
        raise ValueError("unsupported retrieval outbox table name")
    return f"""
    CREATE TABLE {table_name} (
        authority_sequence INTEGER PRIMARY KEY AUTOINCREMENT,
        event_id TEXT NOT NULL UNIQUE,
        kind TEXT NOT NULL CHECK(kind IN ('upsert', 'trash', 'restore', 'purge')),
        source_type TEXT NOT NULL {_RETRIEVAL_SOURCE_TYPE_CHECK_SQL},
        source_unit_id TEXT NOT NULL,
        source_revision TEXT NOT NULL,
        indexed_content_hash TEXT NOT NULL CHECK(
            length(indexed_content_hash) = 64
            AND indexed_content_hash NOT GLOB '*[^0-9a-f]*'
        ),
        data_version_id TEXT NOT NULL,
        occurred_at TEXT NOT NULL,
        status TEXT NOT NULL CHECK(status IN (
            'pending', 'processing', 'applied', 'retryable_failed', 'terminal_failed'
        )),
        attempts INTEGER NOT NULL DEFAULT 0 CHECK(attempts >= 0),
        next_retry_at TEXT NOT NULL,
        lease_token TEXT,
        lease_until TEXT,
        reason_code TEXT,
        updated_at TEXT NOT NULL,
        FOREIGN KEY(data_version_id) REFERENCES retrieval_data_versions(id) ON DELETE RESTRICT
    )
    """


_RETRIEVAL_UNIT_INDEX_STATEMENTS = (
    "CREATE INDEX idx_retrieval_units_search_scope "
    "ON retrieval_units(data_version_id, source_type, retrieval_status, index_state, unit_id)",
    "CREATE INDEX idx_retrieval_units_source_ref "
    "ON retrieval_units(source_type, source_unit_id, source_revision)",
    "CREATE INDEX idx_retrieval_units_source_coverage "
    "ON retrieval_units(data_version_id, source_type, source_unit_id, source_revision, "
    "retrieval_status, index_state)",
)


_RETRIEVAL_OUTBOX_INDEX_STATEMENTS = (
    "CREATE INDEX idx_retrieval_outbox_due "
    "ON retrieval_update_outbox(status, next_retry_at, lease_until, authority_sequence)",
    "CREATE INDEX idx_retrieval_outbox_causal_predecessor "
    "ON retrieval_update_outbox("
    "data_version_id, source_type, source_unit_id, authority_sequence, status)",
)


_RETRIEVAL_MIGRATION_INBOUND_CHILDREN = {
    "retrieval_units": frozenset(
        {
            "learned_sparse_postings",
            "bm25_unit_terms",
            "retrieval_unit_method_indexes",
        }
    ),
    "retrieval_update_outbox": frozenset(
        {
            "retrieval_outbox_manual_actions",
            "document_ingest_job_events",
            "retrieval_outbox_attempt_audits",
        }
    ),
}


# Retrieval 仍与 Workspace authority 共用物理数据库，但其 DDL 被隔离在本组合层；
# S5 将把这一组声明迁至 Retrieval 自己的 schema contribution。
_RETRIEVAL_STATEMENTS = (
    """
    CREATE TABLE retrieval_data_versions (
        id TEXT PRIMARY KEY,
        fingerprint TEXT NOT NULL UNIQUE,
        role TEXT NOT NULL CHECK(role IN ('active', 'staging', 'previous')),
        state TEXT NOT NULL CHECK(state IN ('building', 'ready', 'failed')),
        created_at TEXT NOT NULL,
        activated_at TEXT
    )
    """,
    _retrieval_units_table_sql("retrieval_units"),
    """
    CREATE TABLE retrieval_sync_receipts (
        event_id TEXT PRIMARY KEY,
        status TEXT NOT NULL CHECK(status IN (
            'processing', 'applied', 'retryable_failed', 'terminal_failed'
        )),
        attempts INTEGER NOT NULL CHECK(attempts >= 1),
        reason_code TEXT,
        updated_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE retrieval_capabilities (
        name TEXT PRIMARY KEY,
        available INTEGER NOT NULL CHECK(available IN (0, 1)),
        detail TEXT,
        checked_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE learned_sparse_postings (
        unit_id INTEGER NOT NULL,
        token_id INTEGER NOT NULL,
        weight REAL NOT NULL CHECK(weight > 0),
        PRIMARY KEY(unit_id, token_id),
        FOREIGN KEY(unit_id) REFERENCES retrieval_units(unit_id) ON DELETE CASCADE
    ) WITHOUT ROWID
    """,
    """
    CREATE TABLE bm25_unit_terms (
        unit_id INTEGER PRIMARY KEY,
        shadow_terms TEXT NOT NULL,
        FOREIGN KEY(unit_id) REFERENCES retrieval_units(unit_id) ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE retrieval_unit_method_indexes (
        unit_id INTEGER NOT NULL,
        method TEXT NOT NULL CHECK(method IN ('dense', 'learned_sparse', 'bm25')),
        state TEXT NOT NULL CHECK(state IN ('ready', 'absent')),
        reason_code TEXT,
        updated_at TEXT NOT NULL,
        PRIMARY KEY(unit_id, method),
        FOREIGN KEY(unit_id) REFERENCES retrieval_units(unit_id) ON DELETE CASCADE
    ) WITHOUT ROWID
    """,
    _retrieval_update_outbox_table_sql("retrieval_update_outbox"),
    """
    CREATE TABLE retrieval_outbox_manual_actions (
        action_id TEXT PRIMARY KEY,
        event_id TEXT NOT NULL,
        action TEXT NOT NULL CHECK(action = 'manual_requeue'),
        actor TEXT NOT NULL,
        reason TEXT NOT NULL,
        previous_status TEXT NOT NULL CHECK(previous_status = 'terminal_failed'),
        previous_reason_code TEXT,
        occurred_at TEXT NOT NULL,
        FOREIGN KEY(event_id) REFERENCES retrieval_update_outbox(event_id) ON DELETE RESTRICT
    )
    """,
    "CREATE UNIQUE INDEX idx_retrieval_one_active_version "
    "ON retrieval_data_versions(role) WHERE role='active'",
    "CREATE UNIQUE INDEX idx_retrieval_one_staging_version "
    "ON retrieval_data_versions(role) WHERE role='staging'",
    *_RETRIEVAL_UNIT_INDEX_STATEMENTS,
    "CREATE INDEX idx_learned_sparse_postings_token "
    "ON learned_sparse_postings(token_id, unit_id)",
    "CREATE INDEX idx_retrieval_unit_method_indexes_state "
    "ON retrieval_unit_method_indexes(method, state, unit_id)",
    *_RETRIEVAL_OUTBOX_INDEX_STATEMENTS,
    "CREATE INDEX idx_retrieval_outbox_manual_actions_event "
    "ON retrieval_outbox_manual_actions(event_id, occurred_at DESC, action_id DESC)",
)


_RETRIEVAL_REQUIRED_TABLES = frozenset(
    {
        "retrieval_data_versions",
        "retrieval_units",
        "retrieval_sync_receipts",
        "retrieval_capabilities",
        "retrieval_unit_method_indexes",
        "learned_sparse_postings",
        "bm25_unit_terms",
        "retrieval_update_outbox",
        "retrieval_outbox_manual_actions",
    }
)


def initialize_schema(
    conn: sqlite3.Connection,
    *,
    project_id: str,
    project_root: Path,
) -> None:
    """初始化或验证项目绑定数据库，不复用 PRAGMA user_version。"""

    conn.execute("BEGIN IMMEDIATE")
    try:
        existing_objects = _application_objects(conn)
        if "schema_meta" not in existing_objects and existing_objects:
            raise DocumentSchemaError(
                "refusing to adopt a non-empty database without project schema metadata"
            )
        if "schema_meta" not in existing_objects:
            conn.execute(_SCHEMA_META_SQL)
            row = None
        else:
            row = conn.execute(
                "SELECT schema_version, project_id, project_root "
                "FROM schema_meta WHERE schema_name=?",
                (SCHEMA_NAME,),
            ).fetchone()
            if row is None:
                raise DocumentSchemaError("project schema metadata row is missing")

        if row is None:
            initialize_file_schema(conn)
            for statement in _RETRIEVAL_STATEMENTS:
                conn.execute(statement)
            initialize_document_schema(conn)
            initialize_ingestion_schema(conn)
            ensure_document_file_linkage(conn)
            _initialize_picture_contribution(conn)
            now = _now()
            conn.execute(
                "INSERT INTO schema_meta "
                "(schema_name, schema_version, project_id, project_root, "
                "created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    SCHEMA_NAME,
                    SCHEMA_VERSION,
                    project_id,
                    str(project_root),
                    now,
                    now,
                ),
            )
        else:
            _validate_project_binding(
                row,
                project_id=project_id,
                project_root=project_root,
            )
            if int(row["schema_version"]) == 2:
                _validate_workspace_contributions(
                    conn, include_picture=False, require_file_observation=False,
                    require_shared_ingestion=False,
                    require_document_source_elements=False,
                )
                ensure_document_file_linkage(conn)
                conn.execute(
                    "UPDATE schema_meta SET schema_version=?, updated_at=? "
                    "WHERE schema_name=?",
                    (3, _now(), SCHEMA_NAME),
                )
                row = _schema_row(conn)
            if int(row["schema_version"]) == 3:
                _validate_workspace_contributions(
                    conn, include_picture=False, require_file_observation=False,
                    require_shared_ingestion=False,
                    require_document_source_elements=False,
                )
                _initialize_picture_contribution(conn)
                conn.execute(
                    "UPDATE schema_meta SET schema_version=?, updated_at=? "
                    "WHERE schema_name=?",
                    (4, _now(), SCHEMA_NAME),
                )
                row = _schema_row(conn)
            if int(row["schema_version"]) == 4:
                _validate_workspace_contributions(
                    conn, include_picture=True, require_file_observation=False,
                    require_shared_ingestion=False,
                    require_document_source_elements=False,
                )
                _migrate_retrieval_source_type_expansion(conn)
                conn.execute(
                    "UPDATE schema_meta SET schema_version=?, updated_at=? "
                    "WHERE schema_name=?",
                    (5, _now(), SCHEMA_NAME),
                )
                row = _schema_row(conn)

            if int(row["schema_version"]) == 5:
                _validate_workspace_contributions(
                    conn, include_picture=True, require_file_observation=False,
                    require_shared_ingestion=False,
                    require_document_source_elements=False,
                )
                migrate_file_observation_schema(conn)
                migrate_document_content_version_schema(conn)
                conn.execute(
                    "UPDATE schema_meta SET schema_version=?, updated_at=? "
                    "WHERE schema_name=?",
                    (6, _now(), SCHEMA_NAME),
                )
                row = _schema_row(conn)

            if int(row["schema_version"]) == 6:
                _validate_workspace_contributions(
                    conn, include_picture=True, require_shared_ingestion=False,
                    require_document_source_elements=False,
                )
                migrate_shared_ingestion_schema(conn)
                conn.execute(
                    "UPDATE schema_meta SET schema_version=?, updated_at=? "
                    "WHERE schema_name=?",
                    (7, _now(), SCHEMA_NAME),
                )
                row = _schema_row(conn)

            if int(row["schema_version"]) == 7:
                _validate_workspace_contributions(
                    conn, include_picture=True,
                    require_document_source_elements=False,
                )
                migrate_document_source_elements_schema(conn)
                conn.execute(
                    "UPDATE schema_meta SET schema_version=?, updated_at=? "
                    "WHERE schema_name=?",
                    (8, _now(), SCHEMA_NAME),
                )
                row = _schema_row(conn)

            if int(row["schema_version"]) == 8:
                _validate_workspace_contributions(
                    conn,
                    include_picture=False,
                )
                migrate_picture_observation_question_schema(conn)
                conn.execute(
                    "UPDATE schema_meta SET schema_version=?, updated_at=? "
                    "WHERE schema_name=?",
                    (9, _now(), SCHEMA_NAME),
                )
                row = _schema_row(conn)

        assert row is not None or _schema_row(conn) is not None
        current_row = row if row is not None else _schema_row(conn)
        assert current_row is not None
        _validate_schema_identity(
            current_row,
            project_id=project_id,
            project_root=project_root,
        )
        _validate_workspace_contributions(conn, include_picture=True)
        _validate_retrieval_source_type_contract(conn)
        ensure_document_file_linkage(conn)
        _ensure_method_capabilities(conn)
        conn.commit()
    except BaseException:
        conn.rollback()
        raise


def load_sqlite_vec(conn: sqlite3.Connection) -> tuple[bool, str | None]:
    """在单个连接上注册 sqlite-vec，并返回可诊断能力结果。"""

    try:
        import sqlite_vec

        conn.enable_load_extension(True)
        try:
            sqlite_vec.load(conn)
        finally:
            conn.enable_load_extension(False)
    except Exception as exc:
        return False, f"{type(exc).__name__}:{exc}"
    return True, None


def _validate_workspace_contributions(
    conn: sqlite3.Connection,
    *,
    include_picture: bool,
    require_file_observation: bool = True,
    require_shared_ingestion: bool = True,
    require_document_source_elements: bool = True,
) -> None:
    try:
        validate_file_schema(conn, require_observation=require_file_observation)
        validate_document_schema(conn)
        if require_document_source_elements:
            validate_document_source_elements_schema(conn)
        if require_shared_ingestion:
            validate_ingestion_schema(conn)
        _validate_retrieval_schema(conn)
        if include_picture:
            validate_picture_schema(conn)
    except (
        ValueError, DocumentSchemaContributionError,
        IngestionSchemaContributionError, PictureSchemaError,
    ) as exc:
        raise DocumentSchemaError(
            f"project workspace schema is invalid: {exc}"
        ) from exc


def _ensure_method_capabilities(conn: sqlite3.Connection) -> None:
    try:
        conn.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS bm25_fts "
            "USING fts5(shadow_terms, content='')"
        )
    except sqlite3.OperationalError as exc:
        _set_capability(conn, "fts5", False, f"{type(exc).__name__}:{exc}")
    else:
        _set_capability(conn, "fts5", True, None)

    vec_available, vec_detail = load_sqlite_vec(conn)
    if not vec_available:
        _set_capability(conn, "sqlite_vec", False, vec_detail)
        return
    try:
        conn.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS dense_vectors USING vec0("
            "unit_id INTEGER PRIMARY KEY, embedding float[1024], scope_key TEXT)"
        )
    except sqlite3.OperationalError as exc:
        _set_capability(conn, "sqlite_vec", False, f"{type(exc).__name__}:{exc}")
    else:
        _set_capability(conn, "sqlite_vec", True, None)


def _set_capability(
    conn: sqlite3.Connection,
    name: str,
    available: bool,
    detail: str | None,
) -> None:
    conn.execute(
        "INSERT INTO retrieval_capabilities "
        "(name, available, detail, checked_at) VALUES (?, ?, ?, ?) "
        "ON CONFLICT(name) DO UPDATE SET "
        "available=excluded.available, detail=excluded.detail, "
        "checked_at=excluded.checked_at",
        (name, int(available), detail, _now()),
    )


def _validate_retrieval_schema(conn: sqlite3.Connection) -> None:
    missing = sorted(_RETRIEVAL_REQUIRED_TABLES - _application_objects(conn))
    if missing:
        raise DocumentSchemaError(
            "project retrieval schema is incomplete: " + ", ".join(missing)
        )


def _validate_retrieval_source_type_contract(conn: sqlite3.Connection) -> None:
    """Validate the owned CHECK wire contract and both canonical values."""

    for table_name in ("retrieval_units", "retrieval_update_outbox"):
        row = conn.execute(
            "SELECT sql FROM sqlite_schema WHERE type='table' AND name=?",
            (table_name,),
        ).fetchone()
        if row is None or row[0] is None:
            raise DocumentSchemaError(
                f"retrieval source_type contract is missing for {table_name}"
            )
        normalized_sql = " ".join(str(row[0]).casefold().split())
        normalized_check = " ".join(
            _RETRIEVAL_SOURCE_TYPE_CHECK_SQL.casefold().split()
        )
        if normalized_sql.count(normalized_check) != 1:
            raise DocumentSchemaError(
                f"{table_name}.source_type must use the canonical document/picture CHECK"
            )

    probe_id = uuid.uuid4().hex
    data_version_id = f"schema-probe-version-{probe_id}"
    conn.execute("SAVEPOINT validate_retrieval_source_types")
    try:
        conn.execute(
            "INSERT INTO retrieval_data_versions "
            "(id, fingerprint, role, state, created_at) "
            "VALUES (?, ?, 'previous', 'ready', ?)",
            (data_version_id, f"schema-probe:{probe_id}", _now()),
        )
        for ordinal, source_type in enumerate(("document", "picture"), start=1):
            conn.execute(
                "INSERT INTO retrieval_units "
                "(data_version_id, source_type, source_unit_id, source_revision, "
                "indexed_content_hash, retrieval_status, index_state, scope_json, "
                "created_at, updated_at) "
                "VALUES (?, ?, ?, 'schema-probe-revision', ?, 'active', 'ready', "
                "'{}', ?, ?)",
                (
                    data_version_id,
                    source_type,
                    f"schema-probe-unit-{ordinal}-{probe_id}",
                    f"{ordinal:x}".rjust(64, "0"),
                    _now(),
                    _now(),
                ),
            )
            conn.execute(
                "INSERT INTO retrieval_update_outbox "
                "(event_id, kind, source_type, source_unit_id, source_revision, "
                "indexed_content_hash, data_version_id, occurred_at, status, attempts, "
                "next_retry_at, updated_at) "
                "VALUES (?, 'upsert', ?, ?, 'schema-probe-revision', ?, ?, ?, "
                "'pending', 0, ?, ?)",
                (
                    f"schema-probe-event-{ordinal}-{probe_id}",
                    source_type,
                    f"schema-probe-unit-{ordinal}-{probe_id}",
                    f"{ordinal:x}".rjust(64, "0"),
                    data_version_id,
                    _now(),
                    _now(),
                    _now(),
                ),
            )
    except sqlite3.IntegrityError as exc:
        raise DocumentSchemaError(
            "retrieval source_type contract must accept document and picture"
        ) from exc
    finally:
        conn.execute("ROLLBACK TO SAVEPOINT validate_retrieval_source_types")
        conn.execute("RELEASE SAVEPOINT validate_retrieval_source_types")


def _validate_retrieval_migration_inbound_foreign_keys(
    conn: sqlite3.Connection,
) -> None:
    inbound_children = {
        target: set() for target in _RETRIEVAL_MIGRATION_INBOUND_CHILDREN
    }
    table_names = [
        str(row[0])
        for row in conn.execute(
            "SELECT name FROM sqlite_schema WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%'"
        ).fetchall()
    ]
    for table_name in table_names:
        quoted_table_name = table_name.replace('"', '""')
        foreign_keys = conn.execute(
            f'PRAGMA foreign_key_list("{quoted_table_name}")'
        ).fetchall()
        for foreign_key in foreign_keys:
            target = str(foreign_key[2]).casefold()
            if target in inbound_children:
                inbound_children[target].add(table_name)

    for target, allowed_children in _RETRIEVAL_MIGRATION_INBOUND_CHILDREN.items():
        unknown_children = sorted(inbound_children[target] - allowed_children)
        if unknown_children:
            raise DocumentSchemaError(
                f"schema v4 to v5 migration found unknown inbound foreign keys "
                f"to {target}: {', '.join(unknown_children)}"
            )


def _migrate_retrieval_source_type_expansion(conn: sqlite3.Connection) -> None:
    """Widen the two Retrieval source discriminators without losing child rows."""

    _validate_retrieval_migration_inbound_foreign_keys(conn)
    conn.execute(_retrieval_units_table_sql("retrieval_units_v5"))
    conn.execute(
        "INSERT INTO retrieval_units_v5 "
        "(unit_id, data_version_id, source_type, source_unit_id, source_revision, "
        "indexed_content_hash, retrieval_status, index_state, scope_json, "
        "created_at, updated_at) "
        "SELECT unit_id, data_version_id, source_type, source_unit_id, source_revision, "
        "indexed_content_hash, retrieval_status, index_state, scope_json, "
        "created_at, updated_at FROM retrieval_units"
    )
    conn.execute(
        "CREATE TEMP TABLE migration_v5_sparse_postings AS "
        "SELECT unit_id, token_id, weight FROM learned_sparse_postings"
    )
    conn.execute(
        "CREATE TEMP TABLE migration_v5_bm25_unit_terms AS "
        "SELECT unit_id, shadow_terms FROM bm25_unit_terms"
    )
    conn.execute(
        "CREATE TEMP TABLE migration_v5_method_indexes AS "
        "SELECT unit_id, method, state, reason_code, updated_at "
        "FROM retrieval_unit_method_indexes"
    )
    conn.execute("DELETE FROM learned_sparse_postings")
    conn.execute("DELETE FROM bm25_unit_terms")
    conn.execute("DELETE FROM retrieval_unit_method_indexes")
    conn.execute("DROP TABLE retrieval_units")
    conn.execute("ALTER TABLE retrieval_units_v5 RENAME TO retrieval_units")
    for statement in _RETRIEVAL_UNIT_INDEX_STATEMENTS:
        conn.execute(statement)
    conn.execute(
        "INSERT INTO learned_sparse_postings (unit_id, token_id, weight) "
        "SELECT unit_id, token_id, weight FROM migration_v5_sparse_postings"
    )
    conn.execute(
        "INSERT INTO bm25_unit_terms (unit_id, shadow_terms) "
        "SELECT unit_id, shadow_terms FROM migration_v5_bm25_unit_terms"
    )
    conn.execute(
        "INSERT INTO retrieval_unit_method_indexes "
        "(unit_id, method, state, reason_code, updated_at) "
        "SELECT unit_id, method, state, reason_code, updated_at "
        "FROM migration_v5_method_indexes"
    )
    conn.execute("DROP TABLE migration_v5_sparse_postings")
    conn.execute("DROP TABLE migration_v5_bm25_unit_terms")
    conn.execute("DROP TABLE migration_v5_method_indexes")

    conn.execute(
        _retrieval_update_outbox_table_sql("retrieval_update_outbox_v5")
    )
    conn.execute(
        "INSERT INTO retrieval_update_outbox_v5 "
        "(authority_sequence, event_id, kind, source_type, source_unit_id, source_revision, "
        "indexed_content_hash, data_version_id, occurred_at, status, attempts, "
        "next_retry_at, lease_token, lease_until, reason_code, updated_at) "
        "SELECT rowid, event_id, kind, source_type, source_unit_id, source_revision, "
        "indexed_content_hash, data_version_id, occurred_at, status, attempts, "
        "next_retry_at, lease_token, lease_until, reason_code, updated_at "
        "FROM retrieval_update_outbox"
    )
    conn.execute(
        "CREATE TEMP TABLE migration_v5_manual_actions AS "
        "SELECT action_id, event_id, action, actor, reason, previous_status, "
        "previous_reason_code, occurred_at FROM retrieval_outbox_manual_actions"
    )
    conn.execute(
        "CREATE TEMP TABLE migration_v5_ingest_job_events AS "
        "SELECT job_id, event_id, recorded_at FROM document_ingest_job_events"
    )
    has_attempt_audits = (
        "retrieval_outbox_attempt_audits" in _application_objects(conn)
    )
    if has_attempt_audits:
        conn.execute(
            "CREATE TEMP TABLE migration_v5_attempt_audits AS "
            "SELECT event_id, attempt, worker_kind, worker_instance_hash, batch_id, "
            "batch_limit, batch_size, batch_ordinal, outcome, failure_stage, "
            "safe_error_code, occurred_at FROM retrieval_outbox_attempt_audits"
        )
        conn.execute("DELETE FROM retrieval_outbox_attempt_audits")
    conn.execute("DELETE FROM retrieval_outbox_manual_actions")
    conn.execute("DELETE FROM document_ingest_job_events")
    conn.execute("DROP TABLE retrieval_update_outbox")
    conn.execute(
        "ALTER TABLE retrieval_update_outbox_v5 RENAME TO retrieval_update_outbox"
    )
    for statement in _RETRIEVAL_OUTBOX_INDEX_STATEMENTS:
        conn.execute(statement)
    conn.execute(
        "INSERT INTO retrieval_outbox_manual_actions "
        "(action_id, event_id, action, actor, reason, previous_status, "
        "previous_reason_code, occurred_at) "
        "SELECT action_id, event_id, action, actor, reason, previous_status, "
        "previous_reason_code, occurred_at FROM migration_v5_manual_actions"
    )
    conn.execute(
        "INSERT INTO document_ingest_job_events (job_id, event_id, recorded_at) "
        "SELECT job_id, event_id, recorded_at FROM migration_v5_ingest_job_events"
    )
    if has_attempt_audits:
        conn.execute(
            "INSERT INTO retrieval_outbox_attempt_audits "
            "(event_id, attempt, worker_kind, worker_instance_hash, batch_id, "
            "batch_limit, batch_size, batch_ordinal, outcome, failure_stage, "
            "safe_error_code, occurred_at) "
            "SELECT event_id, attempt, worker_kind, worker_instance_hash, batch_id, "
            "batch_limit, batch_size, batch_ordinal, outcome, failure_stage, "
            "safe_error_code, occurred_at FROM migration_v5_attempt_audits"
        )
        conn.execute("DROP TABLE migration_v5_attempt_audits")
    conn.execute("DROP TABLE migration_v5_manual_actions")
    conn.execute("DROP TABLE migration_v5_ingest_job_events")

    foreign_key_failures = conn.execute("PRAGMA foreign_key_check").fetchall()
    if foreign_key_failures:
        details = ", ".join(
            f"{row[0]}:{row[1]}->{row[2]}" for row in foreign_key_failures
        )
        raise DocumentSchemaError(
            "project schema v4 to v5 migration broke foreign keys: " + details
        )


def _schema_row(conn: sqlite3.Connection) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT schema_version, project_id, project_root "
        "FROM schema_meta WHERE schema_name=?",
        (SCHEMA_NAME,),
    ).fetchone()


def _application_objects(conn: sqlite3.Connection) -> frozenset[str]:
    return frozenset(
        str(row[0])
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE name NOT LIKE 'sqlite_%' "
            "AND type IN ('table', 'view', 'trigger')"
        ).fetchall()
    )


def _validate_schema_identity(
    row: sqlite3.Row,
    *,
    project_id: str,
    project_root: Path,
) -> None:
    version = int(row["schema_version"])
    if version > SCHEMA_VERSION:
        raise DocumentSchemaError(
            f"unsupported future project document schema version: {version}"
        )
    if version < SCHEMA_VERSION:
        raise DocumentSchemaError(
            f"missing project document schema migration from version {version}"
        )
    _validate_project_binding(
        row,
        project_id=project_id,
        project_root=project_root,
    )


def _validate_project_binding(
    row: sqlite3.Row,
    *,
    project_id: str,
    project_root: Path,
) -> None:
    if str(row["project_id"]) != project_id:
        raise DocumentSchemaError("documents database belongs to another project_id")
    if str(row["project_root"]) != str(project_root):
        raise DocumentSchemaError("documents database belongs to another project_root")


def _initialize_picture_contribution(conn: sqlite3.Connection) -> None:
    try:
        initialize_picture_schema(conn)
    except PictureSchemaError as exc:
        raise DocumentSchemaError("project picture schema is invalid") from exc


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
