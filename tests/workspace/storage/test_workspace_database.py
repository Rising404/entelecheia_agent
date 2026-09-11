from __future__ import annotations

import hashlib
import sqlite3

import pytest
import sqlite_vec

from personagraph.workspace.storage import (
    DocumentDatabase,
    DocumentDatabaseError,
    InvalidProjectIdError,
)
from personagraph.workspace.storage.schema import (
    SCHEMA_NAME,
    SCHEMA_VERSION,
)
from personagraph.workspace.storage.context import bind
from personagraph.workspace.documents import application as docstore
from personagraph.workspace.documents import mounting as document_mounting
from personagraph.workspace.pictures.storage.schema import PICTURE_REQUIRED_TABLES
from personagraph.workspace.files import FileSource, WorkspaceFileAuthority


PUBLIC_TABLES = {
    "schema_meta",
    "files",
    "file_versions",
    "file_events",
    "documents",
    "document_versions",
    "doc_chunks",
    "retrieval_data_versions",
    "retrieval_units",
    "retrieval_sync_receipts",
    "retrieval_capabilities",
    "retrieval_unit_method_indexes",
    "learned_sparse_postings",
    "bm25_unit_terms",
    "bm25_fts",
    "dense_vectors",
    "retrieval_update_outbox",
    "retrieval_outbox_manual_actions",
} | PICTURE_REQUIRED_TABLES


def _database(tmp_path, project_id: str) -> DocumentDatabase:
    project_root = tmp_path / "roots" / project_id
    project_root.mkdir(parents=True)
    return DocumentDatabase(
        project_id,
        project_root,
        tmp_path / "var" / "projects" / project_id / "documents.sqlite",
    )


def _insert_file(database: DocumentDatabase, file_id: str, relative_path: str) -> None:
    with database.connect() as conn:
        conn.execute(
            "INSERT INTO files "
            "(id, project_id, relative_path, origin, media_type, created_at, updated_at) "
            "VALUES (?, ?, ?, 'workspace_existing', 'text/plain', ?, ?)",
            (
                file_id,
                database.project_id,
                relative_path,
                "2026-08-30T00:00:00+00:00",
                "2026-08-30T00:00:00+00:00",
            ),
        )


def _drop_picture_contribution(conn: sqlite3.Connection) -> None:
    for table_name in sorted(PICTURE_REQUIRED_TABLES, reverse=True):
        conn.execute(f'DROP TABLE "{table_name}"')


def _insert_retrieval_data_version(
    conn: sqlite3.Connection,
    data_version_id: str = "retrieval-v1",
) -> None:
    conn.execute(
        "INSERT INTO retrieval_data_versions "
        "(id, fingerprint, role, state, created_at) "
        "VALUES (?, ?, 'active', 'ready', '2026-09-04T00:00:00+00:00')",
        (data_version_id, f"fingerprint:{data_version_id}"),
    )


def _insert_retrieval_unit(
    conn: sqlite3.Connection,
    *,
    unit_id: int,
    source_type: str,
    source_unit_id: str,
    data_version_id: str = "retrieval-v1",
) -> None:
    conn.execute(
        "INSERT INTO retrieval_units "
        "(unit_id, data_version_id, source_type, source_unit_id, source_revision, "
        "indexed_content_hash, retrieval_status, index_state, scope_json, "
        "created_at, updated_at) "
        "VALUES (?, ?, ?, ?, 'revision-1', ?, 'active', 'ready', '{}', ?, ?)",
        (
            unit_id,
            data_version_id,
            source_type,
            source_unit_id,
            f"{unit_id:x}".rjust(64, "0"),
            "2026-09-04T00:00:00+00:00",
            "2026-09-04T00:00:00+00:00",
        ),
    )


def _insert_retrieval_outbox_event(
    conn: sqlite3.Connection,
    *,
    event_id: str,
    source_type: str,
    source_unit_id: str,
    status: str = "pending",
    attempts: int = 0,
    reason_code: str | None = None,
    data_version_id: str = "retrieval-v1",
) -> None:
    conn.execute(
        "INSERT INTO retrieval_update_outbox "
        "(event_id, kind, source_type, source_unit_id, source_revision, "
        "indexed_content_hash, data_version_id, occurred_at, status, attempts, "
        "next_retry_at, reason_code, updated_at) "
        "VALUES (?, 'upsert', ?, ?, 'revision-1', ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            event_id,
            source_type,
            source_unit_id,
            "a" * 64,
            data_version_id,
            "2026-09-04T00:00:00+00:00",
            status,
            attempts,
            "2026-09-04T00:00:00+00:00",
            reason_code,
            "2026-09-04T00:00:00+00:00",
        ),
    )


def _rewrite_source_checks(
    conn: sqlite3.Connection,
    *,
    replacement: str,
    table_names: tuple[str, ...] = (
        "retrieval_units",
        "retrieval_update_outbox",
    ),
) -> None:
    current_check = "CHECK(source_type IN ('document', 'picture'))"
    conn.execute("PRAGMA writable_schema = ON")
    try:
        for table_name in table_names:
            row = conn.execute(
                "SELECT sql FROM sqlite_schema WHERE type='table' AND name=?",
                (table_name,),
            ).fetchone()
            assert row is not None
            table_sql = str(row[0])
            assert current_check in table_sql
            conn.execute(
                "UPDATE sqlite_schema SET sql=? WHERE type='table' AND name=?",
                (table_sql.replace(current_check, replacement), table_name),
            )
    finally:
        conn.execute("PRAGMA writable_schema = OFF")


def _convert_fresh_schema_to_v4(conn: sqlite3.Connection) -> None:
    """Make an empty fresh outbox carry the exact historical v4 wire schema."""

    _rewrite_source_checks(
        conn,
        replacement="CHECK(source_type = 'document')",
        table_names=("retrieval_units",),
    )
    conn.execute("DROP TABLE retrieval_update_outbox")
    conn.execute(
        "CREATE TABLE retrieval_update_outbox ("
        "event_id TEXT PRIMARY KEY, "
        "kind TEXT NOT NULL CHECK(kind IN ('upsert', 'trash', 'restore', 'purge')), "
        "source_type TEXT NOT NULL CHECK(source_type = 'document'), "
        "source_unit_id TEXT NOT NULL, source_revision TEXT NOT NULL, "
        "indexed_content_hash TEXT NOT NULL CHECK("
        "length(indexed_content_hash) = 64 "
        "AND indexed_content_hash NOT GLOB '*[^0-9a-f]*'), "
        "data_version_id TEXT NOT NULL, occurred_at TEXT NOT NULL, "
        "status TEXT NOT NULL CHECK(status IN ("
        "'pending', 'processing', 'applied', 'retryable_failed', 'terminal_failed')), "
        "attempts INTEGER NOT NULL DEFAULT 0 CHECK(attempts >= 0), "
        "next_retry_at TEXT NOT NULL, lease_token TEXT, lease_until TEXT, "
        "reason_code TEXT, updated_at TEXT NOT NULL, "
        "FOREIGN KEY(data_version_id) REFERENCES retrieval_data_versions(id) "
        "ON DELETE RESTRICT)"
    )
    conn.execute(
        "CREATE INDEX idx_retrieval_outbox_due ON retrieval_update_outbox("
        "status, next_retry_at, lease_until, occurred_at)"
    )
    conn.execute(
        "UPDATE schema_meta SET schema_version=4 WHERE schema_name=?",
        (SCHEMA_NAME,),
    )


def test_one_project_database_owns_document_and_retrieval_tables(tmp_path):
    database = _database(tmp_path, "project-a")

    database.initialize()

    with database.connect() as conn:
        tables = {
            str(row["name"])
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        metadata = conn.execute(
            "SELECT * FROM schema_meta WHERE schema_name=?",
            (SCHEMA_NAME,),
        ).fetchone()
        capabilities = {
            str(row["name"]): bool(row["available"])
            for row in conn.execute(
                "SELECT name, available FROM retrieval_capabilities"
            ).fetchall()
        }
        user_version = int(conn.execute("PRAGMA user_version").fetchone()[0])
        foreign_key_failures = conn.execute("PRAGMA foreign_key_check").fetchall()
        document_version_columns = {
            str(row["name"])
            for row in conn.execute("PRAGMA table_info(document_versions)")
        }
        picture_observation_columns = {
            str(row["name"])
            for row in conn.execute("PRAGMA table_info(picture_observations)")
        }
        trigger_names = {
            str(row["name"])
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='trigger'"
            ).fetchall()
        }

    assert PUBLIC_TABLES <= tables
    assert {"doc_mounts", "doc_retrieval_snapshots"}.isdisjoint(tables)
    assert "session_owned_resource_purge_receipts" not in tables
    assert metadata is not None
    assert metadata["schema_version"] == SCHEMA_VERSION
    assert metadata["project_id"] == "project-a"
    assert metadata["project_root"] == str(database.project_root)
    assert capabilities == {"fts5": True, "sqlite_vec": True}
    assert user_version == 0
    assert foreign_key_failures == []
    assert {"source_elements_json", "source_elements_sha256"} <= (
        document_version_columns
    )
    assert "question" in picture_observation_columns
    assert {
        "trg_document_versions_source_elements_insert_shape",
        "trg_document_versions_source_elements_immutable",
    } <= trigger_names


def test_v8_workspace_migrates_picture_questions_without_rebuilding_project(
    tmp_path,
) -> None:
    database = _database(tmp_path, "project-a")
    database.initialize()
    with database.connect() as conn:
        conn.execute("ALTER TABLE picture_observations DROP COLUMN question")
        conn.execute(
            "UPDATE schema_meta SET schema_version=8 WHERE schema_name=?",
            (SCHEMA_NAME,),
        )

    reopened = DocumentDatabase(
        database.project_id,
        database.project_root,
        database.db_path,
    )
    reopened.initialize()

    with reopened.connect() as conn:
        assert conn.execute(
            "SELECT schema_version FROM schema_meta WHERE schema_name=?",
            (SCHEMA_NAME,),
        ).fetchone()[0] == SCHEMA_VERSION
        assert "question" in {
            str(row["name"])
            for row in conn.execute("PRAGMA table_info(picture_observations)")
        }


def test_fresh_v5_retrieval_source_checks_accept_only_document_and_picture(tmp_path):
    database = _database(tmp_path, "project-a")

    with database.connect() as conn:
        _insert_retrieval_data_version(conn)
        _insert_retrieval_unit(
            conn,
            unit_id=1,
            source_type="document",
            source_unit_id="doc-chunk-1",
        )
        _insert_retrieval_unit(
            conn,
            unit_id=2,
            source_type="picture",
            source_unit_id="picture-unit-1",
        )
        _insert_retrieval_outbox_event(
            conn,
            event_id="event-document",
            source_type="document",
            source_unit_id="doc-chunk-1",
        )
        _insert_retrieval_outbox_event(
            conn,
            event_id="event-picture",
            source_type="picture",
            source_unit_id="picture-unit-1",
        )

        with pytest.raises(sqlite3.IntegrityError):
            _insert_retrieval_unit(
                conn,
                unit_id=3,
                source_type="audio",
                source_unit_id="audio-unit-1",
            )
        with pytest.raises(sqlite3.IntegrityError):
            _insert_retrieval_outbox_event(
                conn,
                event_id="event-audio",
                source_type="audio",
                source_unit_id="audio-unit-1",
            )


def test_reopened_v5_keeps_picture_rows_and_closed_source_checks(tmp_path):
    first = _database(tmp_path, "project-a")
    with first.connect() as conn:
        _insert_retrieval_data_version(conn)
        _insert_retrieval_unit(
            conn,
            unit_id=7,
            source_type="picture",
            source_unit_id="picture-unit-7",
        )
        _insert_retrieval_outbox_event(
            conn,
            event_id="event-picture-7",
            source_type="picture",
            source_unit_id="picture-unit-7",
        )

    reopened = DocumentDatabase(first.project_id, first.project_root, first.db_path)
    with reopened.connect() as conn:
        assert conn.execute(
            "SELECT source_type FROM retrieval_units WHERE unit_id=7"
        ).fetchone()[0] == "picture"
        assert conn.execute(
            "SELECT source_type FROM retrieval_update_outbox "
            "WHERE event_id='event-picture-7'"
        ).fetchone()[0] == "picture"
        with pytest.raises(sqlite3.IntegrityError):
            _insert_retrieval_unit(
                conn,
                unit_id=8,
                source_type="video",
                source_unit_id="video-unit-8",
            )


@pytest.mark.parametrize(
    "replacement",
    (
        "CHECK(source_type = 'document')",
        "",
        "CHECK(source_type IN ('document', 'picture', 'audio'))",
    ),
    ids=("document-only", "open-text", "widened"),
)
def test_reopened_v5_fails_closed_for_tampered_source_type_contract(
    tmp_path,
    replacement: str,
):
    database = _database(tmp_path, "project-a")
    with database.connect() as conn:
        _rewrite_source_checks(conn, replacement=replacement)

    reopened = DocumentDatabase(
        database.project_id,
        database.project_root,
        database.db_path,
    )
    with pytest.raises(DocumentDatabaseError) as raised:
        reopened.initialize()

    assert "source_type" in str(raised.value.__cause__)


def test_project_session_detach_does_not_trash_shared_retrieval_units(
    tmp_path,
) -> None:
    database = _database(tmp_path, "project-a")

    class _SessionMounts:
        def __init__(self) -> None:
            self.unmounted: list[tuple[str, str]] = []

        @staticmethod
        def current_session_id() -> str:
            return "session-a"

        @staticmethod
        def mount_document(document_id: str, session_id: str) -> bool:
            del document_id, session_id
            return True

        @staticmethod
        def is_document_mounted(document_id: str, session_id: str) -> bool:
            return (document_id, session_id) == ("doc-a", "session-a")

        def unmount_document(self, document_id: str, session_id: str) -> bool:
            self.unmounted.append((document_id, session_id))
            return True

        @staticmethod
        def list_document_mounts(session_id: str) -> list[dict[str, object]]:
            del session_id
            return []

        @staticmethod
        def create_document_retrieval_snapshot(**kwargs: object) -> None:
            del kwargs

        @staticmethod
        def get_document_retrieval_snapshot(
            snapshot_id: str,
            session_id: str,
        ) -> dict[str, object] | None:
            del snapshot_id, session_id
            return None

    mounts = _SessionMounts()

    with document_mounting.bind_document_mount_port(mounts), bind(database):
        assert docstore.detach("doc-a", "session-a")

    assert mounts.unmounted == [("doc-a", "session-a")]
    with database.connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM retrieval_update_outbox").fetchone()[0] == 0


def test_fresh_handle_reopens_initialized_database(tmp_path):
    first = _database(tmp_path, "project-a")
    first.initialize()
    reopened = DocumentDatabase(first.project_id, first.project_root, first.db_path)

    with reopened.connect() as conn:
        metadata = conn.execute(
            "SELECT project_id, project_root FROM schema_meta WHERE schema_name=?",
            (SCHEMA_NAME,),
        ).fetchone()

    assert metadata is not None
    assert metadata["project_id"] == first.project_id
    assert metadata["project_root"] == str(first.project_root)


def test_content_version_migration_preserves_rows_and_separates_observations(tmp_path):
    database = _database(tmp_path, "project-a")
    source = database.project_root / "notes.txt"
    source.write_text("original bytes", encoding="utf-8")
    registered = WorkspaceFileAuthority(database).ensure_current_path(
        "notes.txt", source=FileSource.WORKSPACE_EXISTING,
    )
    version = registered.version
    with database.connect() as conn:
        conn.execute(
            "INSERT INTO documents (id, path, file_id, current_version_id) "
            "VALUES ('doc-a', ?, ?, 'docver-a')",
            (str(source), registered.file.file_id),
        )
        conn.execute(
            "INSERT INTO document_versions (id, doc_id, version_number, status, "
            "captured_at, file_version_id, source_sha256, source_size, source_mtime_ns) "
            "VALUES ('docver-a', 'doc-a', 1, 'active', ?, ?, ?, ?, ?)",
            (version.created_at, version.file_version_id, version.content_sha256,
             version.size_bytes, version.source_mtime_ns),
        )
        original_version = tuple(conn.execute("SELECT * FROM file_versions").fetchone())
        original_document = tuple(conn.execute("SELECT * FROM document_versions").fetchone())
        original_events = [tuple(row) for row in conn.execute("SELECT * FROM file_events")]
        # Recreate the pre-observation schema, including the old timestamp linkage.
        for name in (
            "trg_document_versions_require_file_version",
            "trg_document_versions_file_version_update",
        ):
            old_sql = conn.execute(
                "SELECT sql FROM sqlite_schema WHERE name=?", (name,),
            ).fetchone()[0]
            if name.endswith("_update"):
                start = old_sql.index("WHEN ")
                end = old_sql.index("OR NOT EXISTS", start)
                old_sql = old_sql[:start] + "WHEN NEW.file_version_id IS NULL " + old_sql[end:]
            old_sql = old_sql.replace(
                "AND version.size_bytes=NEW.source_size",
                "AND version.size_bytes=NEW.source_size "
                "AND version.source_mtime_ns IS NEW.source_mtime_ns",
            )
            conn.execute(f"DROP TRIGGER {name}")
            conn.execute(old_sql)
        conn.execute("ALTER TABLE files DROP COLUMN observed_mtime_ns")
        conn.execute("UPDATE schema_meta SET schema_version=5")

    reopened = DocumentDatabase(database.project_id, database.project_root, database.db_path)
    with reopened.connect() as conn:
        assert conn.execute("SELECT schema_version FROM schema_meta").fetchone()[0] == SCHEMA_VERSION
        assert conn.execute("SELECT observed_mtime_ns FROM files").fetchone()[0] == version.source_mtime_ns
        assert tuple(conn.execute("SELECT * FROM file_versions").fetchone()) == original_version
        assert tuple(conn.execute("SELECT * FROM document_versions").fetchone()) == original_document
        assert [tuple(row) for row in conn.execute("SELECT * FROM file_events")] == original_events
        # Same content version, later parsing observation: valid, without rewriting history.
        conn.execute(
            "INSERT INTO document_versions (id, doc_id, version_number, status, "
            "captured_at, file_version_id, source_sha256, source_size, source_mtime_ns) "
            "VALUES ('docver-b', 'doc-a', 2, 'active', ?, ?, ?, ?, ?)",
            (version.created_at, version.file_version_id, version.content_sha256,
             version.size_bytes, version.source_mtime_ns + 1),
        )
        for field, replacement in (
            ("source_mtime_ns", version.source_mtime_ns + 1),
            ("file_version_id", "another-version"),
        ):
            with pytest.raises(sqlite3.IntegrityError, match="immutable"):
                conn.execute(
                    f"UPDATE document_versions SET {field}=? WHERE id='docver-a'",
                    (replacement,),
                )
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []

    again = DocumentDatabase(database.project_id, database.project_root, database.db_path)
    with again.connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM document_versions").fetchone()[0] == 2


def test_shared_ingestion_migration_retires_old_jobs_and_preserves_corpus(tmp_path):
    database = _database(tmp_path, "project-a")
    source = database.project_root / "retained.txt"
    source.write_text("retained document bytes", encoding="utf-8")
    registered = WorkspaceFileAuthority(database).ensure_current_path(
        "retained.txt", source=FileSource.WORKSPACE_EXISTING,
    )
    version = registered.version
    retained_tables = (
        "files", "file_versions", "file_events", "documents", "document_versions", "doc_chunks",
        "pictures", "picture_units", "picture_observations",
        "retrieval_data_versions", "retrieval_units", "retrieval_update_outbox",
    )
    with database.connect() as conn:
        conn.execute(
            "INSERT INTO documents (id,path,file_id,current_version_id) "
            "VALUES ('retained-doc',?,?,'retained-docv')",
            (str(source), registered.file.file_id),
        )
        conn.execute(
            "INSERT INTO document_versions (id,doc_id,version_number,status,captured_at,"
            "file_version_id,source_sha256,source_size,source_mtime_ns) "
            "VALUES ('retained-docv','retained-doc',1,'active',?,?,?,?,?)",
            (version.created_at, version.file_version_id, version.content_sha256,
             version.size_bytes, version.source_mtime_ns),
        )
        _insert_retrieval_data_version(conn)
        _insert_retrieval_unit(conn, unit_id=71, source_type="document", source_unit_id="retained-unit")
        _insert_retrieval_outbox_event(conn, event_id="retained-event", source_type="document", source_unit_id="retained-unit")
        before = {table: [tuple(row) for row in conn.execute(f"SELECT * FROM {table}")] for table in retained_tables}
        # 退役任务无需解释 recipe 或猜 generation；模拟没有任何新身份字段的历史形状。
        conn.execute("DROP TABLE document_ingest_requests")
        conn.execute("DROP TABLE document_ingest_job_events")
        conn.execute("DROP TABLE document_ingest_jobs")
        conn.execute(
            "CREATE TABLE document_ingest_jobs (job_id TEXT PRIMARY KEY, "
            "session_id TEXT NOT NULL, source_mtime_ns INTEGER NOT NULL)"
        )
        conn.execute(
            "CREATE TABLE document_ingest_job_events (job_id TEXT NOT NULL,event_id TEXT NOT NULL,"
            "recorded_at TEXT NOT NULL,PRIMARY KEY(job_id,event_id),"
            "FOREIGN KEY(job_id) REFERENCES document_ingest_jobs(job_id) ON DELETE CASCADE,"
            "FOREIGN KEY(event_id) REFERENCES retrieval_update_outbox(event_id) ON DELETE RESTRICT)"
        )
        conn.execute("INSERT INTO document_ingest_jobs VALUES ('old-job','old-session',1)")
        conn.execute("INSERT INTO document_ingest_job_events VALUES ('old-job','retained-event','2026-09-05T00:00:00+00:00')")
        conn.execute("UPDATE schema_meta SET schema_version=6")

    reopened = DocumentDatabase(database.project_id, database.project_root, database.db_path)
    with reopened.connect() as conn:
        assert conn.execute("SELECT schema_version FROM schema_meta").fetchone()[0] == SCHEMA_VERSION
        for table, rows in before.items():
            assert [tuple(row) for row in conn.execute(f"SELECT * FROM {table}")] == rows
        for table in ("document_ingest_jobs", "document_ingest_job_events", "document_ingest_requests"):
            assert conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
    # 正常重开只验证 schema，不能重复执行退役步骤。
    with DocumentDatabase(database.project_id, database.project_root, database.db_path).connect() as conn:
        for table, rows in before.items():
            assert [tuple(row) for row in conn.execute(f"SELECT * FROM {table}")] == rows


def test_source_elements_migration_preserves_legacy_versions_as_unavailable(tmp_path):
    database = _database(tmp_path, "project-a")
    source = database.project_root / "legacy.txt"
    source.write_text("legacy source text", encoding="utf-8")
    registered = WorkspaceFileAuthority(database).ensure_current_path(
        "legacy.txt", source=FileSource.WORKSPACE_EXISTING,
    )
    version = registered.version
    with database.connect() as conn:
        conn.execute(
            "INSERT INTO documents (id, path, file_id, current_version_id) "
            "VALUES ('legacy-doc', ?, ?, 'legacy-doc-version')",
            (str(source), registered.file.file_id),
        )
        conn.execute(
            "INSERT INTO document_versions (id, doc_id, version_number, status, "
            "captured_at, file_version_id, source_sha256, source_size) "
            "VALUES ('legacy-doc-version', 'legacy-doc', 1, 'active', ?, ?, ?, ?)",
            (
                version.created_at,
                version.file_version_id,
                version.content_sha256,
                version.size_bytes,
            ),
        )
        for trigger in (
            "trg_document_versions_source_elements_insert_shape",
            "trg_document_versions_source_elements_immutable",
        ):
            conn.execute(f"DROP TRIGGER {trigger}")
        conn.execute("ALTER TABLE document_versions DROP COLUMN source_elements_json")
        conn.execute("ALTER TABLE document_versions DROP COLUMN source_elements_sha256")
        conn.execute(
            "UPDATE schema_meta SET schema_version=7 WHERE schema_name=?",
            (SCHEMA_NAME,),
        )

    reopened = DocumentDatabase(
        database.project_id, database.project_root, database.db_path,
    )
    with reopened.connect() as conn:
        migrated = conn.execute(
            "SELECT source_elements_json, source_elements_sha256 "
            "FROM document_versions WHERE id='legacy-doc-version'"
        ).fetchone()
        assert tuple(migrated) == (None, None)
        assert conn.execute(
            "SELECT schema_version FROM schema_meta WHERE schema_name=?",
            (SCHEMA_NAME,),
        ).fetchone()[0] == SCHEMA_VERSION
        source_elements_json = (
            '[{"element_id":"el_1","content":"legacy source text",'
            '"locator":"c0-18","source_pages":[]}]'
        )
        source_elements_sha256 = hashlib.sha256(
            source_elements_json.encode("utf-8")
        ).hexdigest()
        with pytest.raises(sqlite3.IntegrityError, match="source elements lineage"):
            conn.execute(
                "INSERT INTO document_versions (id, doc_id, version_number, status, "
                "captured_at, file_version_id, source_sha256, source_size, "
                "source_elements_json) VALUES "
                "('invalid-pair', 'legacy-doc', 2, 'active', ?, ?, ?, ?, ?)",
                (
                    version.created_at,
                    version.file_version_id,
                    version.content_sha256,
                    version.size_bytes,
                    source_elements_json,
                ),
            )
        conn.execute(
            "INSERT INTO document_versions (id, doc_id, version_number, status, "
            "captured_at, file_version_id, source_sha256, source_size, "
            "source_elements_json, source_elements_sha256) VALUES "
            "('valid-snapshot', 'legacy-doc', 2, 'active', ?, ?, ?, ?, ?, ?)",
            (
                version.created_at,
                version.file_version_id,
                version.content_sha256,
                version.size_bytes,
                source_elements_json,
                source_elements_sha256,
            ),
        )
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            conn.execute(
                "UPDATE document_versions SET source_elements_json='[]' "
                "WHERE id='valid-snapshot'"
            )

    with DocumentDatabase(
        database.project_id, database.project_root, database.db_path,
    ).connect() as conn:
        assert conn.execute(
            "SELECT source_elements_json FROM document_versions "
            "WHERE id='legacy-doc-version'"
        ).fetchone()[0] is None


def test_v2_database_reopen_runs_v3_and_v4_migrations(tmp_path):
    database = _database(tmp_path, "project-a")
    database.initialize()
    with sqlite3.connect(database.db_path) as conn:
        trigger_names = (
            "trg_documents_require_file",
            "trg_documents_file_immutable",
            "trg_document_versions_require_file_version",
            "trg_document_versions_file_version_update",
            "trg_files_restrict_linked_delete",
            "trg_file_versions_restrict_linked_delete",
        )
        for trigger_name in trigger_names:
            conn.execute(f"DROP TRIGGER {trigger_name}")
        _drop_picture_contribution(conn)
        conn.execute(
            "UPDATE schema_meta SET schema_version=2 WHERE schema_name=?",
            (SCHEMA_NAME,),
        )

    reopened = DocumentDatabase(
        database.project_id,
        database.project_root,
        database.db_path,
    )
    reopened.initialize()

    with reopened.connect() as conn:
        version = conn.execute(
            "SELECT schema_version FROM schema_meta WHERE schema_name=?",
            (SCHEMA_NAME,),
        ).fetchone()[0]
        installed = {
            str(row[0])
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='trigger'"
            ).fetchall()
        }
        tables = {
            str(row[0])
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
    assert version == SCHEMA_VERSION
    assert set(trigger_names) <= installed
    assert PICTURE_REQUIRED_TABLES <= tables


def test_v3_database_reopen_installs_picture_contribution(tmp_path):
    database = _database(tmp_path, "project-a")
    database.initialize()
    with sqlite3.connect(database.db_path) as conn:
        _drop_picture_contribution(conn)
        conn.execute(
            "UPDATE schema_meta SET schema_version=3 WHERE schema_name=?",
            (SCHEMA_NAME,),
        )

    reopened = DocumentDatabase(
        database.project_id,
        database.project_root,
        database.db_path,
    )
    reopened.initialize()

    with reopened.connect() as conn:
        version = conn.execute(
            "SELECT schema_version FROM schema_meta WHERE schema_name=?",
            (SCHEMA_NAME,),
        ).fetchone()[0]
        tables = {
            str(row[0])
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
    assert version == SCHEMA_VERSION
    assert PICTURE_REQUIRED_TABLES <= tables


def test_v4_to_v5_migration_preserves_nonempty_retrieval_graph(tmp_path):
    database = _database(tmp_path, "project-a")
    with database.connect() as conn:
        _insert_retrieval_data_version(conn)
        _insert_retrieval_unit(
            conn,
            unit_id=41,
            source_type="document",
            source_unit_id="doc-chunk-41",
        )
        conn.execute(
            "INSERT INTO learned_sparse_postings (unit_id, token_id, weight) "
            "VALUES (41, 101, 0.75)"
        )
        conn.execute(
            "INSERT INTO bm25_unit_terms (unit_id, shadow_terms) "
            "VALUES (41, 't101 t202')"
        )
        conn.executemany(
            "INSERT INTO retrieval_unit_method_indexes "
            "(unit_id, method, state, reason_code, updated_at) "
            "VALUES (41, ?, 'ready', NULL, '2026-09-04T00:00:00+00:00')",
            (("dense",), ("learned_sparse",), ("bm25",)),
        )
        conn.execute(
            "INSERT INTO dense_vectors (unit_id, embedding, scope_key) VALUES (?, ?, ?)",
            (41, sqlite_vec.serialize_float32([0.125] * 1024), "scope-41"),
        )
        conn.execute(
            "INSERT INTO bm25_fts (rowid, shadow_terms) VALUES (41, 't101 t202')"
        )
        _convert_fresh_schema_to_v4(conn)
        conn.execute(
            "CREATE TABLE retrieval_outbox_attempt_audits ("
            "event_id TEXT NOT NULL, attempt INTEGER NOT NULL, "
            "worker_kind TEXT NOT NULL, worker_instance_hash TEXT NOT NULL, "
            "batch_id TEXT NOT NULL, batch_limit INTEGER NOT NULL, "
            "batch_size INTEGER NOT NULL, batch_ordinal INTEGER NOT NULL, "
            "outcome TEXT NOT NULL, failure_stage TEXT, safe_error_code TEXT, "
            "occurred_at TEXT NOT NULL, PRIMARY KEY(event_id, attempt), "
            "FOREIGN KEY(event_id) REFERENCES retrieval_update_outbox(event_id) "
            "ON DELETE RESTRICT)"
        )
        conn.execute(
            "CREATE INDEX idx_retrieval_outbox_attempt_audits_batch "
            "ON retrieval_outbox_attempt_audits(batch_id, batch_ordinal)"
        )
        _insert_retrieval_outbox_event(
            conn,
            event_id="event-41",
            source_type="document",
            source_unit_id="doc-chunk-41",
            status="terminal_failed",
            attempts=3,
            reason_code="index_failed",
        )
        _insert_retrieval_outbox_event(
            conn,
            event_id="event-42-old",
            source_type="document",
            source_unit_id="doc-chunk-41",
        )
        conn.execute(
            "UPDATE retrieval_update_outbox SET occurred_at=? WHERE event_id=?",
            ("2025-01-01T00:00:00+00:00", "event-42-old"),
        )
        conn.execute(
            "INSERT INTO retrieval_outbox_attempt_audits "
            "(event_id, attempt, worker_kind, worker_instance_hash, batch_id, "
            "batch_limit, batch_size, batch_ordinal, outcome, failure_stage, "
            "safe_error_code, occurred_at) "
            "VALUES ('event-41', 3, 'document-indexer', 'worker-hash', 'batch-41', "
            "8, 1, 1, 'terminal_failed', 'write', 'safe-index-error', "
            "'2026-09-04T00:00:30+00:00')"
        )
        conn.execute(
            "INSERT INTO retrieval_outbox_manual_actions "
            "(action_id, event_id, action, actor, reason, previous_status, "
            "previous_reason_code, occurred_at) "
            "VALUES ('action-41', 'event-41', 'manual_requeue', 'operator', "
            "'retry after repair', 'terminal_failed', 'index_failed', "
            "'2026-09-04T00:01:00+00:00')"
        )


    reopened = DocumentDatabase(
        database.project_id,
        database.project_root,
        database.db_path,
    )
    reopened.initialize()

    with reopened.connect() as conn:
        assert conn.execute(
            "SELECT schema_version FROM schema_meta WHERE schema_name=?",
            (SCHEMA_NAME,),
        ).fetchone()[0] == SCHEMA_VERSION
        assert tuple(
            conn.execute(
                "SELECT source_type, source_unit_id, retrieval_status, index_state "
                "FROM retrieval_units WHERE unit_id=41"
            ).fetchone()
        ) == ("document", "doc-chunk-41", "active", "ready")
        assert tuple(
            conn.execute(
                "SELECT unit_id, token_id, weight FROM learned_sparse_postings"
            ).fetchone()
        ) == (41, 101, 0.75)
        assert tuple(
            conn.execute(
                "SELECT unit_id, shadow_terms FROM bm25_unit_terms"
            ).fetchone()
        ) == (41, "t101 t202")
        assert [
            tuple(row)
            for row in conn.execute(
                "SELECT unit_id, method, state FROM retrieval_unit_method_indexes "
                "ORDER BY method"
            ).fetchall()
        ] == [
            (41, "bm25", "ready"),
            (41, "dense", "ready"),
            (41, "learned_sparse", "ready"),
        ]
        assert tuple(
            conn.execute(
                "SELECT unit_id, scope_key FROM dense_vectors WHERE unit_id=41"
            ).fetchone()
        ) == (41, "scope-41")
        assert [
            int(row[0])
            for row in conn.execute(
                "SELECT rowid FROM bm25_fts WHERE bm25_fts MATCH 't101'"
            ).fetchall()
        ] == [41]
        assert tuple(
            conn.execute(
                "SELECT authority_sequence, event_id, source_type, source_unit_id, "
                "status, attempts, reason_code "
                "FROM retrieval_update_outbox WHERE event_id='event-41'"
            ).fetchone()
        ) == (
            1,
            "event-41",
            "document",
            "doc-chunk-41",
            "terminal_failed",
            3,
            "index_failed",
        )
        assert [
            tuple(row)
            for row in conn.execute(
                "SELECT authority_sequence, event_id "
                "FROM retrieval_update_outbox ORDER BY authority_sequence"
            ).fetchall()
        ] == [(1, "event-41"), (2, "event-42-old")]
        assert tuple(
            conn.execute(
                "SELECT action_id, event_id, previous_status "
                "FROM retrieval_outbox_manual_actions"
            ).fetchone()
        ) == ("action-41", "event-41", "terminal_failed")
        assert tuple(
            conn.execute(
                "SELECT event_id, attempt, batch_id, outcome, safe_error_code "
                "FROM retrieval_outbox_attempt_audits"
            ).fetchone()
        ) == (
            "event-41",
            3,
            "batch-41",
            "terminal_failed",
            "safe-index-error",
        )
        assert conn.execute("SELECT * FROM document_ingest_job_events").fetchall() == []
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
        index_names = {
            str(row[0])
            for row in conn.execute(
                "SELECT name FROM sqlite_schema WHERE type='index'"
            ).fetchall()
        }
        assert {
            "idx_retrieval_units_search_scope",
            "idx_retrieval_units_source_ref",
            "idx_retrieval_units_source_coverage",
            "idx_retrieval_outbox_due",
            "idx_retrieval_outbox_causal_predecessor",
            "idx_retrieval_outbox_attempt_audits_batch",
            "idx_retrieval_outbox_manual_actions_event",
            "idx_document_ingest_job_events_job",
            "idx_document_ingest_job_events_event",
        } <= index_names

        _insert_retrieval_unit(
            conn,
            unit_id=42,
            source_type="picture",
            source_unit_id="picture-unit-42",
        )
        _insert_retrieval_outbox_event(
            conn,
            event_id="event-picture-42",
            source_type="picture",
            source_unit_id="picture-unit-42",
        )
        assert conn.execute(
            "SELECT authority_sequence FROM retrieval_update_outbox "
            "WHERE event_id='event-picture-42'"
        ).fetchone()[0] == 3
        with pytest.raises(sqlite3.IntegrityError):
            _insert_retrieval_unit(
                conn,
                unit_id=43,
                source_type="audio",
                source_unit_id="audio-unit-43",
            )
        with pytest.raises(sqlite3.IntegrityError):
            _insert_retrieval_outbox_event(
                conn,
                event_id="event-audio-43",
                source_type="audio",
                source_unit_id="audio-unit-43",
            )


@pytest.mark.parametrize("unknown_child", ("unit", "outbox"))
def test_v4_to_v5_rejects_unknown_inbound_fk_without_losing_rows(
    tmp_path,
    unknown_child: str,
):
    database = _database(tmp_path, "project-a")
    with database.connect() as conn:
        _insert_retrieval_data_version(conn)
        _insert_retrieval_unit(
            conn,
            unit_id=71,
            source_type="document",
            source_unit_id="doc-chunk-71",
        )
        _convert_fresh_schema_to_v4(conn)
        _insert_retrieval_outbox_event(
            conn,
            event_id="event-71",
            source_type="document",
            source_unit_id="doc-chunk-71",
        )
        if unknown_child == "unit":
            child_table = "extra_unit_metadata"
            conn.execute(
                "CREATE TABLE extra_unit_metadata ("
                "metadata_id TEXT PRIMARY KEY, unit_id INTEGER NOT NULL, "
                "payload TEXT NOT NULL, FOREIGN KEY(unit_id) "
                "REFERENCES retrieval_units(unit_id) ON DELETE CASCADE)"
            )
            conn.execute(
                "INSERT INTO extra_unit_metadata (metadata_id, unit_id, payload) "
                "VALUES ('metadata-71', 71, 'preserve-me')"
            )
        else:
            child_table = "extra_outbox_metadata"
            conn.execute(
                "CREATE TABLE extra_outbox_metadata ("
                "metadata_id TEXT PRIMARY KEY, event_id TEXT NOT NULL, "
                "payload TEXT NOT NULL, FOREIGN KEY(event_id) "
                "REFERENCES retrieval_update_outbox(event_id) ON DELETE RESTRICT)"
            )
            conn.execute(
                "INSERT INTO extra_outbox_metadata (metadata_id, event_id, payload) "
                "VALUES ('metadata-71', 'event-71', 'preserve-me')"
            )

    reopened = DocumentDatabase(
        database.project_id,
        database.project_root,
        database.db_path,
    )
    with pytest.raises(DocumentDatabaseError) as raised:
        reopened.initialize()

    assert "unknown inbound foreign keys" in str(raised.value.__cause__)
    assert child_table in str(raised.value.__cause__)
    with sqlite3.connect(database.db_path) as conn:
        assert conn.execute(
            "SELECT schema_version FROM schema_meta WHERE schema_name=?",
            (SCHEMA_NAME,),
        ).fetchone()[0] == 4
        assert conn.execute("SELECT COUNT(*) FROM retrieval_units").fetchone()[0] == 1
        assert conn.execute(
            "SELECT COUNT(*) FROM retrieval_update_outbox"
        ).fetchone()[0] == 1
        assert conn.execute(
            f'SELECT metadata_id, payload FROM "{child_table}"'
        ).fetchone() == ("metadata-71", "preserve-me")
        assert conn.execute(
            "SELECT COUNT(*) FROM sqlite_schema "
            "WHERE name IN ('retrieval_units_v5', 'retrieval_update_outbox_v5')"
        ).fetchone()[0] == 0
        source_sql = conn.execute(
            "SELECT sql FROM sqlite_schema WHERE name='retrieval_units'"
        ).fetchone()[0]
        assert "CHECK(source_type = 'document')" in source_sql


def test_current_database_rejects_missing_picture_contribution(tmp_path):
    database = _database(tmp_path, "project-a")
    database.initialize()
    with sqlite3.connect(database.db_path) as conn:
        conn.execute("DROP TABLE picture_observations")

    reopened = DocumentDatabase(
        database.project_id,
        database.project_root,
        database.db_path,
    )
    with pytest.raises(DocumentDatabaseError) as raised:
        reopened.initialize()

    assert "picture_observations" in str(raised.value.__cause__)


def test_future_project_schema_version_fails_closed(tmp_path):
    database = _database(tmp_path, "project-a")
    database.initialize()
    with sqlite3.connect(database.db_path) as conn:
        conn.execute(
            "UPDATE schema_meta SET schema_version=? WHERE schema_name=?",
            (SCHEMA_VERSION + 1, SCHEMA_NAME),
        )

    reopened = DocumentDatabase(
        database.project_id,
        database.project_root,
        database.db_path,
    )
    with pytest.raises(DocumentDatabaseError) as raised:
        reopened.initialize()

    assert "unsupported future project document schema" in str(
        raised.value.__cause__
    )


def test_project_schema_rejects_a_document_without_registered_file(tmp_path):
    database = _database(tmp_path, "project-a")

    with database.connect() as conn:
        with pytest.raises(sqlite3.IntegrityError, match="registered file_id"):
            conn.execute(
                "INSERT INTO documents (id, hash, file_id) "
                "VALUES ('unlinked-document', 'content-hash', NULL)"
            )


def test_two_projects_have_isolated_rows_and_schema_identity(tmp_path):
    project_a = _database(tmp_path, "project-a")
    project_b = _database(tmp_path, "project-b")
    _insert_file(project_a, "file-a", "notes/a.txt")
    _insert_file(project_b, "file-b", "notes/b.txt")

    with project_a.connect() as conn:
        rows_a = conn.execute("SELECT id, project_id FROM files").fetchall()
    with project_b.connect() as conn:
        rows_b = conn.execute("SELECT id, project_id FROM files").fetchall()

    assert project_a.db_path != project_b.db_path
    assert [(row["id"], row["project_id"]) for row in rows_a] == [
        ("file-a", "project-a")
    ]
    assert [(row["id"], row["project_id"]) for row in rows_b] == [
        ("file-b", "project-b")
    ]

    with project_a.connect() as conn:
        with pytest.raises(sqlite3.IntegrityError, match="does not match"):
            conn.execute(
                "INSERT INTO files "
                "(id, project_id, relative_path, origin, created_at, updated_at) "
                "VALUES ('wrong', 'project-b', 'wrong.txt', 'user_upload', 'now', 'now')"
            )


@pytest.mark.parametrize(
    "project_id",
    (
        "",
        ".",
        "..",
        "../escape",
        "project/child",
        "project\\child",
        "project child",
        "-leading-hyphen",
        "x" * 129,
    ),
)
def test_invalid_project_id_is_rejected_before_database_creation(tmp_path, project_id):
    db_path = tmp_path / "var" / "projects" / "unsafe" / "documents.sqlite"

    with pytest.raises(InvalidProjectIdError):
        DocumentDatabase(project_id, tmp_path / "root", db_path)

    assert not db_path.exists()


def test_file_origin_constraints_are_closed(tmp_path):
    database = _database(tmp_path, "project-a")
    _insert_file(database, "file-a", "file.txt")

    with database.connect() as conn:
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO files "
                "(id, project_id, relative_path, origin, created_at, updated_at) "
                "VALUES ('bad-origin', 'project-a', 'file.txt', 'session_output', 'now', 'now')"
            )
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO file_versions "
                "(id, file_id, version_number, producer, content_sha256, size_bytes, created_at) "
                "VALUES ('v1', 'file-a', 1, 'agent_modified', ?, 1, 'now')",
                ("0" * 64,),
            )


def test_database_identity_cannot_be_rebound_to_another_project(tmp_path):
    database = _database(tmp_path, "project-a")
    database.initialize()
    other_root = tmp_path / "roots" / "project-b"
    other_root.mkdir(parents=True)
    rebound = DocumentDatabase("project-b", other_root, database.db_path)

    with pytest.raises(DocumentDatabaseError):
        rebound.initialize()
