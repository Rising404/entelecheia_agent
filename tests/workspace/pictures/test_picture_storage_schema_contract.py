from __future__ import annotations

import sqlite3

import pytest

from personagraph.workspace.pictures.observations import (
    PictureObservationDraft,
    PictureObservationModality,
    PictureObservationRepository,
    PictureObservationService,
)
from personagraph.workspace.pictures.storage import schema


def picture_storage_test_connection(*, foreign_keys: bool = True) -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.execute(f"PRAGMA foreign_keys = {1 if foreign_keys else 0}")
    conn.executescript(
        """
        CREATE TABLE files (
            id TEXT PRIMARY KEY,
            current_version_id TEXT
        );
        CREATE TABLE file_versions (
            id TEXT PRIMARY KEY,
            file_id TEXT NOT NULL,
            content_sha256 TEXT NOT NULL,
            UNIQUE(id, file_id),
            FOREIGN KEY(file_id) REFERENCES files(id) ON DELETE CASCADE
        );
        """
    )
    return conn


def install_picture_storage(conn: sqlite3.Connection) -> None:
    conn.execute("BEGIN IMMEDIATE")
    schema.initialize_picture_schema(conn)
    assert conn.in_transaction is True
    conn.commit()


def test_picture_schema_installs_only_its_contribution_and_reopens() -> None:
    conn = picture_storage_test_connection()
    conn.execute("BEGIN IMMEDIATE")

    schema.initialize_picture_schema(conn)

    assert conn.in_transaction is True
    tables = {
        str(row[0])
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
    }
    assert schema.PICTURE_REQUIRED_TABLES <= tables
    assert "picture_observation_units" not in tables
    assert "workspace_picture_schema_meta" not in tables
    assert "schema_meta" not in tables
    assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
    observation_columns = {
        str(row[1])
        for row in conn.execute(
            "PRAGMA table_info(picture_observations)"
        ).fetchall()
    }
    assert {
        "picture_unit_id",
        "question",
        "request_sha256",
        "structured_payload_json",
    } <= observation_columns
    assert "input_sha256" not in observation_columns
    conn.commit()

    conn.execute("BEGIN IMMEDIATE")
    schema.initialize_picture_schema(conn)
    schema.validate_picture_schema(conn)
    conn.rollback()


def test_question_migration_preserves_legacy_observation_as_unasked() -> None:
    conn = picture_storage_test_connection()
    install_picture_storage(conn)
    conn.execute(
        "INSERT INTO files (id, current_version_id) VALUES ('file-1', 'version-1')"
    )
    conn.execute(
        "INSERT INTO file_versions (id, file_id, content_sha256) "
        "VALUES ('version-1', 'file-1', ?)",
        ("a" * 64,),
    )
    conn.execute(
        "INSERT INTO pictures "
        "(picture_id, file_id, file_version_id, source_kind, source_locator_json, "
        "source_locator_sha256, source_content_sha256, source_media_type, created_at) "
        "VALUES ('picture-1', 'file-1', 'version-1', 'whole_file', "
        "'{\"kind\":\"whole_file\",\"payload\":{}}', ?, ?, 'image/png', 'now')",
        ("b" * 64, "a" * 64),
    )
    conn.execute(
        "INSERT INTO picture_units "
        "(picture_unit_id, picture_id, unit_kind, unit_locator_json, "
        "unit_locator_sha256, producer_fingerprint, parent_picture_unit_id, "
        "pixel_sha256, media_type, width, height, created_at) "
        "VALUES ('unit-1', 'picture-1', 'full', "
        "'{\"kind\":\"full\",\"payload\":{}}', ?, 'renderer-v1', NULL, ?, "
        "'image/png', 1, 1, 'now')",
        ("c" * 64, "d" * 64),
    )
    legacy_draft = PictureObservationDraft(
        picture_id="picture-1",
        picture_unit_id="unit-1",
        logical_invocation_id="call-1",
        request_ordinal=0,
        modality=PictureObservationModality.VLM,
        purpose="caption",
        kind="caption",
        text="legacy answer",
        uncertainty=None,
        processor_fingerprint="processor-v1",
        prompt_fingerprint="prompt-v1",
    )
    committed = PictureObservationService(PictureObservationRepository()).commit_in_transaction(
        conn,
        legacy_draft,
        created_at="2026-09-04T12:00:00+00:00",
    )
    conn.commit()
    conn.execute("ALTER TABLE picture_observations DROP COLUMN question")
    conn.commit()
    conn.execute("BEGIN IMMEDIATE")

    schema.migrate_picture_observation_question_schema(conn)

    conn.commit()
    assert conn.execute(
        "SELECT question, text FROM picture_observations WHERE observation_id=?",
        (committed.observation.observation_id,),
    ).fetchone() == (None, "legacy answer")
    loaded = PictureObservationRepository().get_by_id(
        conn,
        observation_id=committed.observation.observation_id,
    )
    assert loaded == committed.observation
    schema.validate_picture_schema(conn)


def test_picture_schema_requires_foreign_keys_and_caller_transaction() -> None:
    conn = picture_storage_test_connection()

    with pytest.raises(schema.PictureSchemaTransactionRequired):
        schema.initialize_picture_schema(conn)

    foreign_keys_off = picture_storage_test_connection(foreign_keys=False)
    foreign_keys_off.execute("BEGIN IMMEDIATE")
    with pytest.raises(schema.PictureSchemaError, match="foreign_keys"):
        schema.initialize_picture_schema(foreign_keys_off)


def test_picture_schema_rejects_partial_owned_contribution() -> None:
    conn = picture_storage_test_connection()
    conn.execute("CREATE TABLE pictures (picture_id TEXT PRIMARY KEY)")
    conn.commit()
    conn.execute("BEGIN IMMEDIATE")

    with pytest.raises(schema.PictureSchemaError, match="partial"):
        schema.initialize_picture_schema(conn)


def test_picture_schema_rejects_retired_observation_join_table() -> None:
    conn = picture_storage_test_connection()
    install_picture_storage(conn)
    conn.execute(
        "CREATE TABLE picture_observation_units ("
        "observation_id TEXT NOT NULL, picture_unit_id TEXT NOT NULL)"
    )

    with pytest.raises(
        schema.PictureSchemaError,
        match="retired.*picture_observation_units",
    ):
        schema.validate_picture_schema(conn)


def test_picture_schema_rejects_same_named_noop_immutable_trigger() -> None:
    conn = picture_storage_test_connection()
    install_picture_storage(conn)
    conn.execute("DROP TRIGGER trg_picture_observations_immutable_update")
    conn.execute(
        "CREATE TRIGGER trg_picture_observations_immutable_update "
        "BEFORE UPDATE ON picture_observations BEGIN SELECT 1; END"
    )

    with pytest.raises(
        schema.PictureSchemaError,
        match="trg_picture_observations_immutable_update.*unexpected SQL",
    ):
        schema.validate_picture_schema(conn)


def test_picture_schema_rejects_same_named_index_with_wrong_definition() -> None:
    conn = picture_storage_test_connection()
    install_picture_storage(conn)
    conn.execute("DROP INDEX idx_picture_observations_fifo")
    conn.execute(
        "CREATE INDEX idx_picture_observations_fifo "
        "ON picture_observations(observation_id)"
    )

    with pytest.raises(
        schema.PictureSchemaError,
        match="idx_picture_observations_fifo.*unexpected SQL",
    ):
        schema.validate_picture_schema(conn)


def test_picture_schema_rejects_tampered_observation_composite_fk() -> None:
    conn = picture_storage_test_connection()
    install_picture_storage(conn)
    row = conn.execute(
        "SELECT sql FROM sqlite_schema WHERE name='picture_observations'"
    ).fetchone()
    assert row is not None
    table_sql = str(row[0])
    tampered_sql = table_sql.replace(
        "FOREIGN KEY(picture_unit_id, picture_id)",
        "FOREIGN KEY(picture_unit_id)",
    ).replace(
        "REFERENCES picture_units(picture_unit_id, picture_id)",
        "REFERENCES picture_units(picture_unit_id)",
    )
    assert tampered_sql != table_sql
    conn.execute("PRAGMA writable_schema = ON")
    try:
        conn.execute(
            "UPDATE sqlite_schema SET sql=? WHERE name='picture_observations'",
            (tampered_sql,),
        )
    finally:
        conn.execute("PRAGMA writable_schema = OFF")

    with pytest.raises(
        schema.PictureSchemaError,
        match="picture_observations.*unexpected SQL",
    ):
        schema.validate_picture_schema(conn)


def test_picture_schema_requires_exact_file_version_composite_key() -> None:
    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(
        """
        CREATE TABLE files (id TEXT PRIMARY KEY, current_version_id TEXT);
        CREATE TABLE file_versions (
            id TEXT PRIMARY KEY,
            file_id TEXT NOT NULL,
            content_sha256 TEXT NOT NULL
        );
        """
    )
    conn.execute("BEGIN IMMEDIATE")

    with pytest.raises(schema.PictureSchemaError, match=r"UNIQUE file_versions"):
        schema.initialize_picture_schema(conn)


def test_observation_fifo_is_picture_partitioned_and_append_only() -> None:
    conn = picture_storage_test_connection()
    install_picture_storage(conn)
    digest_a = "a" * 64
    digest_b = "b" * 64
    source = '{"kind":"whole_file","payload":{}}'
    source_hash = "2" * 64
    full = '{"kind":"full","payload":{}}'
    full_hash = "3" * 64
    conn.execute("BEGIN IMMEDIATE")
    for file_id, version_id, digest in (
        ("file-a", "version-a", digest_a),
        ("file-b", "version-b", digest_b),
    ):
        conn.execute(
            "INSERT INTO files (id, current_version_id) VALUES (?, ?)",
            (file_id, version_id),
        )
        conn.execute(
            "INSERT INTO file_versions (id, file_id, content_sha256) VALUES (?, ?, ?)",
            (version_id, file_id, digest),
        )
        conn.execute(
            "INSERT INTO pictures "
            "(picture_id, file_id, file_version_id, source_kind, "
            "source_locator_json, source_locator_sha256, source_content_sha256, "
            "source_media_type, created_at) VALUES (?, ?, ?, 'whole_file', ?, ?, ?, "
            "'image/png', '2026-09-04T00:00:00+00:00')",
            (f"picture-{file_id}", file_id, version_id, source, source_hash, digest),
        )
        conn.execute(
            "INSERT INTO picture_units "
            "(picture_unit_id, picture_id, unit_kind, unit_locator_json, "
            "unit_locator_sha256, producer_fingerprint, parent_picture_unit_id, "
            "pixel_sha256, "
            "media_type, width, height, created_at) "
            "VALUES (?, ?, 'full', ?, ?, 'source-image@1', NULL, ?, "
            "'image/png', 1, 1, "
            "'2026-09-04T00:00:00+00:00')",
            (f"unit-{file_id}", f"picture-{file_id}", full, full_hash, digest),
        )

    def append(
        observation_id: str,
        picture_id: str,
        invocation: str,
        *,
        picture_unit_id: str | None = None,
        structured_payload_json: str | None = None,
    ) -> int:
        bound_unit_id = picture_unit_id or picture_id.replace("picture-", "unit-", 1)
        cursor = conn.execute(
            "INSERT INTO picture_observations "
            "(observation_id, picture_id, picture_unit_id, logical_invocation_id, "
            "request_ordinal, modality, purpose, kind, text, "
            "structured_payload_json, uncertainty, processor_fingerprint, "
            "prompt_fingerprint, request_sha256, output_sha256, payload_sha256, "
            "created_at) "
            "VALUES (?, ?, ?, ?, 0, 'ocr', 'read', 'text', ?, ?, NULL, "
            "'ocr@1', NULL, ?, ?, ?, '2026-09-04T00:00:00+00:00')",
            (
                observation_id,
                picture_id,
                bound_unit_id,
                invocation,
                observation_id,
                structured_payload_json,
                "4" * 64,
                "5" * 64,
                "6" * 64,
            ),
        )
        return int(cursor.lastrowid)

    first = append("observation-a1", "picture-file-a", "call-a1")
    second = append("observation-b1", "picture-file-b", "call-b1")
    third = append("observation-a2", "picture-file-a", "call-a2")
    assert first < second < third
    structured = append(
        "observation-b2",
        "picture-file-b",
        "call-b2",
        structured_payload_json=(
            '{"contract":"picture-structured-v1","payload":{"labels":[]}}'
        ),
    )
    assert structured > third
    with pytest.raises(sqlite3.IntegrityError, match="CHECK constraint"):
        append(
            "observation-b3",
            "picture-file-b",
            "call-b3",
            structured_payload_json='{"payload":{}}',
        )

    newest = conn.execute(
        "SELECT observation_id FROM picture_observations "
        "WHERE picture_id = ? ORDER BY sequence DESC LIMIT 2",
        ("picture-file-a",),
    ).fetchall()
    assert [str(row[0]) for row in reversed(newest)] == [
        "observation-a1",
        "observation-a2",
    ]

    with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
        append(
            "observation-cross-picture",
            "picture-file-a",
            "call-cross-picture",
            picture_unit_id="unit-file-b",
        )
    with pytest.raises(sqlite3.IntegrityError, match="immutable"):
        conn.execute(
            "UPDATE picture_observations SET text = 'changed' "
            "WHERE observation_id = 'observation-a1'"
        )
    with pytest.raises(sqlite3.IntegrityError, match="immutable"):
        conn.execute(
            "DELETE FROM picture_observations "
            "WHERE observation_id = 'observation-a1'"
        )
