from __future__ import annotations

import sqlite3

import pytest

from personagraph.workspace.pictures import (
    PictureRecord,
    PictureSourceLocator,
    PictureUnitLocator,
    PictureUnitRecord,
)
from personagraph.workspace.pictures.storage import repository, schema


def _connection() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(
        """
        CREATE TABLE files (id TEXT PRIMARY KEY, current_version_id TEXT);
        CREATE TABLE file_versions (
            id TEXT PRIMARY KEY,
            file_id TEXT NOT NULL,
            content_sha256 TEXT NOT NULL,
            UNIQUE(id, file_id),
            FOREIGN KEY(file_id) REFERENCES files(id) ON DELETE CASCADE
        );
        """
    )
    conn.execute("BEGIN IMMEDIATE")
    schema.initialize_picture_schema(conn)
    conn.commit()
    return conn


def _picture(*, picture_id: str, locator: PictureSourceLocator) -> PictureRecord:
    return PictureRecord(
        picture_id=picture_id,
        file_id="file",
        file_version_id="version",
        source_locator=locator,
        source_content_sha256="a" * 64,
        source_media_type="image/png",
        created_at="2026-09-04T00:00:00+00:00",
    )


def test_repository_insert_primitive_neither_generates_nor_adjudicates_replay() -> None:
    conn = _connection()
    locator = PictureSourceLocator.whole_file()
    candidate = _picture(picture_id="caller-owned-picture-id", locator=locator)

    with pytest.raises(repository.PictureTransactionRequired):
        repository.insert_picture_if_absent_in_transaction(conn, candidate)

    conn.execute("BEGIN IMMEDIATE")
    conn.execute(
        "INSERT INTO files (id, current_version_id) VALUES ('file', 'version')"
    )
    conn.execute(
        "INSERT INTO file_versions (id, file_id, content_sha256) "
        "VALUES ('version', 'file', ?)",
        ("a" * 64,),
    )
    assert repository.insert_picture_if_absent_in_transaction(conn, candidate) is True

    replay_candidate = _picture(
        picture_id="another-caller-owned-id",
        locator=locator,
    )
    assert (
        repository.insert_picture_if_absent_in_transaction(conn, replay_candidate)
        is False
    )
    assert repository.get_picture_for_source(conn, "version", locator) == candidate


def test_repository_unit_primitive_accepts_only_complete_records() -> None:
    conn = _connection()
    conn.execute("BEGIN IMMEDIATE")
    conn.execute(
        "INSERT INTO files (id, current_version_id) VALUES ('file', 'version')"
    )
    conn.execute(
        "INSERT INTO file_versions (id, file_id, content_sha256) "
        "VALUES ('version', 'file', ?)",
        ("a" * 64,),
    )
    picture = _picture(
        picture_id="caller-owned-picture-id",
        locator=PictureSourceLocator.whole_file(),
    )
    assert repository.insert_picture_if_absent_in_transaction(conn, picture)
    unit = PictureUnitRecord(
        picture_unit_id="caller-owned-unit-id",
        picture_id=picture.picture_id,
        locator=PictureUnitLocator.full(),
        producer_fingerprint="source-image@1",
        parent_picture_unit_id=None,
        pixel_sha256="b" * 64,
        media_type="image/png",
        width=10,
        height=20,
        created_at="2026-09-04T00:00:01+00:00",
    )

    assert repository.insert_picture_unit_if_absent_in_transaction(conn, unit) is True
    assert repository.insert_picture_unit_if_absent_in_transaction(conn, unit) is False
    assert repository.get_picture_unit(conn, unit.picture_unit_id) == unit


def test_repository_surfaces_persistence_collision_without_admission_policy() -> None:
    conn = _connection()
    conn.execute("BEGIN IMMEDIATE")
    conn.execute(
        "INSERT INTO files (id, current_version_id) VALUES ('file', 'version')"
    )
    conn.execute(
        "INSERT INTO file_versions (id, file_id, content_sha256) "
        "VALUES ('version', 'file', ?)",
        ("a" * 64,),
    )
    first = _picture(
        picture_id="same-id",
        locator=PictureSourceLocator.whole_file(),
    )
    assert repository.insert_picture_if_absent_in_transaction(conn, first)
    different_binding = PictureRecord(
        picture_id="same-id",
        file_id="file",
        file_version_id="version",
        source_locator=PictureSourceLocator.from_payload(
            "embedded_asset",
            {"relationship": "rId1"},
        ),
        source_content_sha256="b" * 64,
        source_media_type="image/png",
        created_at="2026-09-04T00:00:02+00:00",
    )

    with pytest.raises(repository.PicturePersistenceConflict):
        repository.insert_picture_if_absent_in_transaction(conn, different_binding)
