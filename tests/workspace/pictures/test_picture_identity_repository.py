from __future__ import annotations

import sqlite3

import pytest

from personagraph.workspace.pictures import (
    PictureAdmissionError,
    PictureAdmissionForeignKeysRequired,
    PictureAdmissionTransactionRequired,
    PictureDocumentSurfaceKind,
    PictureRegistrationConflict,
    PictureSourceKind,
    PictureSourceLocator,
    PictureUnitKind,
    PictureUnitLocator,
    ensure_picture_in_transaction,
    ensure_picture_unit_in_transaction,
)
from personagraph.workspace.pictures.storage import repository, schema


def _connection() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA foreign_keys = ON")
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
    conn.execute("BEGIN IMMEDIATE")
    schema.initialize_picture_schema(conn)
    conn.commit()
    return conn


def _seed_file(
    conn: sqlite3.Connection,
    *,
    file_id: str,
    versions: tuple[tuple[str, str], ...],
    current_version_id: str,
) -> None:
    conn.execute(
        "INSERT INTO files (id, current_version_id) VALUES (?, ?)",
        (file_id, current_version_id),
    )
    conn.executemany(
        "INSERT INTO file_versions (id, file_id, content_sha256) VALUES (?, ?, ?)",
        ((version_id, file_id, digest) for version_id, digest in versions),
    )


def test_picture_and_units_are_idempotent_exact_version_identities() -> None:
    conn = _connection()
    digest_v1 = "a" * 64
    digest_v2 = "b" * 64
    conn.execute("BEGIN IMMEDIATE")
    _seed_file(
        conn,
        file_id="file-1",
        versions=(("version-1", digest_v1), ("version-2", digest_v2)),
        current_version_id="version-2",
    )

    whole = PictureSourceLocator.whole_file()
    first = ensure_picture_in_transaction(
        conn,
        file_id="file-1",
        file_version_id="version-1",
        source_locator=whole,
        source_content_sha256=digest_v1,
        source_media_type="image/png",
        picture_id="picture-v1",
    )
    duplicate = ensure_picture_in_transaction(
        conn,
        file_id="file-1",
        file_version_id="version-1",
        source_locator=whole,
        source_content_sha256=digest_v1,
        source_media_type="image/png",
        picture_id="ignored-idempotent-candidate",
    )
    current = ensure_picture_in_transaction(
        conn,
        file_id="file-1",
        file_version_id="version-2",
        source_locator=whole,
        source_content_sha256=digest_v2,
        source_media_type="image/png",
        picture_id="picture-v2",
    )

    assert first.created is True
    assert duplicate.created is False
    assert duplicate.picture.picture_id == first.picture.picture_id
    assert current.picture.picture_id != first.picture.picture_id
    assert repository.picture_is_current(conn, first.picture.picture_id) is False
    assert repository.picture_is_current(conn, current.picture.picture_id) is True
    assert repository.get_current_picture_for_file(conn, "file-1", whole) == current.picture

    full = ensure_picture_unit_in_transaction(
        conn,
        picture_id=current.picture.picture_id,
        locator=PictureUnitLocator.full(),
        producer_fingerprint="source-image@1",
        parent_picture_unit_id=None,
        pixel_sha256=digest_v2,
        media_type="image/png",
        width=128,
        height=96,
        picture_unit_id="unit-full",
    )
    crop_locator = PictureUnitLocator.from_payload(
        PictureUnitKind.CROP,
        {"x": 0, "y": 0, "width": 64, "height": 48},
    )
    crop = ensure_picture_unit_in_transaction(
        conn,
        picture_id=current.picture.picture_id,
        locator=crop_locator,
        producer_fingerprint="cropper@1",
        parent_picture_unit_id=full.unit.picture_unit_id,
        pixel_sha256="c" * 64,
        media_type="image/png",
        width=64,
        height=48,
        picture_unit_id="unit-crop",
    )
    duplicate_crop = ensure_picture_unit_in_transaction(
        conn,
        picture_id=current.picture.picture_id,
        locator=crop_locator,
        producer_fingerprint="cropper@1",
        parent_picture_unit_id=full.unit.picture_unit_id,
        pixel_sha256="c" * 64,
        media_type="image/png",
        width=64,
        height=48,
        picture_unit_id="ignored-unit-candidate",
    )

    assert crop.created is True
    assert duplicate_crop.created is False
    assert duplicate_crop.unit == crop.unit
    render = ensure_picture_unit_in_transaction(
        conn,
        picture_id=current.picture.picture_id,
        locator=PictureUnitLocator.from_payload(
            PictureUnitKind.RENDER,
            {"dpi": 144, "background": "white"},
        ),
        producer_fingerprint="renderer@1",
        parent_picture_unit_id=None,
        pixel_sha256="d" * 64,
        media_type="image/png",
        width=128,
        height=96,
        picture_unit_id="unit-render",
    )
    same_crop_other_parent = ensure_picture_unit_in_transaction(
        conn,
        picture_id=current.picture.picture_id,
        locator=crop_locator,
        producer_fingerprint="cropper@1",
        parent_picture_unit_id=render.unit.picture_unit_id,
        pixel_sha256="e" * 64,
        media_type="image/png",
        width=64,
        height=48,
        picture_unit_id="unit-crop-render",
    )
    assert same_crop_other_parent.created is True
    assert same_crop_other_parent.unit.picture_unit_id != crop.unit.picture_unit_id

    producer_upgrade = ensure_picture_unit_in_transaction(
        conn,
        picture_id=current.picture.picture_id,
        locator=crop_locator,
        producer_fingerprint="cropper@2",
        parent_picture_unit_id=full.unit.picture_unit_id,
        pixel_sha256="f" * 64,
        media_type="image/png",
        width=64,
        height=48,
        picture_unit_id="unit-crop-v2",
    )
    assert producer_upgrade.created is True
    with pytest.raises(PictureRegistrationConflict, match="different"):
        ensure_picture_unit_in_transaction(
            conn,
            picture_id=current.picture.picture_id,
            locator=crop_locator,
            producer_fingerprint="cropper@1",
            parent_picture_unit_id=full.unit.picture_unit_id,
            pixel_sha256="0" * 64,
            media_type="image/png",
            width=64,
            height=48,
        )

    assert {
        unit.picture_unit_id
        for unit in repository.list_picture_units(conn, current.picture.picture_id)
    } == {
        "unit-full",
        "unit-crop",
        "unit-render",
        "unit-crop-render",
        "unit-crop-v2",
    }
    assert repository.picture_unit_binding_is_current(
        conn,
        picture_unit_id=crop.unit.picture_unit_id,
        picture_id=current.picture.picture_id,
        locator=crop_locator,
        parent_picture_unit_id=full.unit.picture_unit_id,
        producer_fingerprint="cropper@1",
        pixel_sha256="c" * 64,
    )
    conn.commit()


def test_same_locator_with_changed_source_content_hash_fails_closed() -> None:
    conn = _connection()
    conn.execute("BEGIN IMMEDIATE")
    _seed_file(
        conn,
        file_id="document-file",
        versions=(("document-version", "a" * 64),),
        current_version_id="document-version",
    )
    locator = PictureSourceLocator.from_payload(
        PictureSourceKind.EMBEDDED_ASSET,
        {"page": 3, "relationship": "rId7"},
    )
    ensure_picture_in_transaction(
        conn,
        file_id="document-file",
        file_version_id="document-version",
        source_locator=locator,
        source_content_sha256="b" * 64,
        source_media_type="image/png",
    )

    with pytest.raises(PictureRegistrationConflict, match="different"):
        ensure_picture_in_transaction(
            conn,
            file_id="document-file",
            file_version_id="document-version",
            source_locator=locator,
            source_content_sha256="c" * 64,
            source_media_type="image/png",
        )


def test_document_surface_keeps_source_identity_separate_from_render_recipe() -> None:
    conn = _connection()
    source_sha256 = "a" * 64
    conn.execute("BEGIN IMMEDIATE")
    _seed_file(
        conn,
        file_id="slides",
        versions=(("slides-v1", source_sha256),),
        current_version_id="slides-v1",
    )
    surface = PictureSourceLocator.document_surface(
        PictureDocumentSurfaceKind.PPTX_SLIDE,
        4,
    )

    admitted = ensure_picture_in_transaction(
        conn,
        file_id="slides",
        file_version_id="slides-v1",
        source_locator=surface,
        source_content_sha256=source_sha256,
        source_media_type=(
            "application/vnd.openxmlformats-officedocument.presentationml.presentation"
        ),
    )

    assert admitted.picture.source_locator == surface
    assert admitted.picture.source_media_type.startswith("application/")
    with pytest.raises(PictureRegistrationConflict, match="file version"):
        ensure_picture_in_transaction(
            conn,
            file_id="slides",
            file_version_id="slides-v1",
            source_locator=surface,
            source_content_sha256="b" * 64,
            source_media_type=(
                "application/vnd.openxmlformats-officedocument.presentationml.presentation"
            ),
        )


def test_write_api_requires_transaction_and_exact_parent_picture() -> None:
    conn = _connection()
    with pytest.raises(PictureAdmissionError) as transaction_failure:
        ensure_picture_in_transaction(
            conn,
            file_id="missing",
            file_version_id="missing",
            source_locator=PictureSourceLocator.whole_file(),
            source_content_sha256="a" * 64,
            source_media_type="image/png",
        )
    assert isinstance(
        transaction_failure.value,
        PictureAdmissionTransactionRequired,
    )

    conn.execute("BEGIN IMMEDIATE")
    _seed_file(
        conn,
        file_id="file-a",
        versions=(("version-a", "a" * 64),),
        current_version_id="version-a",
    )
    _seed_file(
        conn,
        file_id="file-b",
        versions=(("version-b", "b" * 64),),
        current_version_id="version-b",
    )
    picture_a = ensure_picture_in_transaction(
        conn,
        file_id="file-a",
        file_version_id="version-a",
        source_locator=PictureSourceLocator.whole_file(),
        source_content_sha256="a" * 64,
        source_media_type="image/png",
    ).picture
    picture_b = ensure_picture_in_transaction(
        conn,
        file_id="file-b",
        file_version_id="version-b",
        source_locator=PictureSourceLocator.whole_file(),
        source_content_sha256="b" * 64,
        source_media_type="image/png",
    ).picture
    parent = ensure_picture_unit_in_transaction(
        conn,
        picture_id=picture_a.picture_id,
        locator=PictureUnitLocator.full(),
        producer_fingerprint="source-image@1",
        parent_picture_unit_id=None,
        pixel_sha256="a" * 64,
        media_type="image/png",
        width=10,
        height=10,
    ).unit

    with pytest.raises(PictureRegistrationConflict, match="same picture"):
        ensure_picture_unit_in_transaction(
            conn,
            picture_id=picture_b.picture_id,
            locator=PictureUnitLocator.from_payload(
                PictureUnitKind.TILE,
                {"row": 0, "column": 0},
            ),
            producer_fingerprint="tiler@1",
            parent_picture_unit_id=parent.picture_unit_id,
            pixel_sha256="c" * 64,
            media_type="image/png",
            width=5,
            height=5,
        )


def test_write_api_requires_foreign_keys_and_rejects_explicit_empty_values() -> None:
    foreign_keys_off = _connection()
    foreign_keys_off.execute("PRAGMA foreign_keys = OFF")
    foreign_keys_off.execute("BEGIN IMMEDIATE")
    with pytest.raises(PictureAdmissionError, match="foreign_keys") as foreign_key_failure:
        ensure_picture_in_transaction(
            foreign_keys_off,
            file_id="file",
            file_version_id="version",
            source_locator=PictureSourceLocator.whole_file(),
            source_content_sha256="a" * 64,
            source_media_type="image/png",
        )
    assert isinstance(
        foreign_key_failure.value,
        PictureAdmissionForeignKeysRequired,
    )

    conn = _connection()
    conn.execute("BEGIN IMMEDIATE")
    _seed_file(
        conn,
        file_id="file",
        versions=(("version", "a" * 64),),
        current_version_id="version",
    )
    with pytest.raises(ValueError, match="picture_id"):
        ensure_picture_in_transaction(
            conn,
            file_id="file",
            file_version_id="version",
            source_locator=PictureSourceLocator.whole_file(),
            source_content_sha256="a" * 64,
            source_media_type="image/png",
            picture_id="",
        )
    with pytest.raises(ValueError, match="created_at"):
        ensure_picture_in_transaction(
            conn,
            file_id="file",
            file_version_id="version",
            source_locator=PictureSourceLocator.whole_file(),
            source_content_sha256="a" * 64,
            source_media_type="image/png",
            created_at="",
        )


def test_locator_contract_is_canonical_and_fail_closed() -> None:
    locator = PictureSourceLocator.document_surface(
        PictureDocumentSurfaceKind.PPTX_SLIDE,
        2,
    )
    assert locator.canonical_json == (
        '{"kind":"document_surface","payload":'
        '{"ordinal":2,"surface_kind":"pptx_slide"}}'
    )
    assert locator == PictureSourceLocator(
        kind=PictureSourceKind.DOCUMENT_SURFACE,
        canonical_json=locator.canonical_json,
    )

    with pytest.raises(ValueError, match="canonical"):
        PictureSourceLocator(
            kind=PictureSourceKind.DOCUMENT_SURFACE,
            canonical_json=(
                '{"payload":{"ordinal":2,"surface_kind":"pptx_slide"}, '
                '"kind":"document_surface"}'
            ),
        )
    with pytest.raises(ValueError, match="payload must be empty"):
        PictureSourceLocator.from_payload(
            PictureSourceKind.WHOLE_FILE,
            {"path": "must-not-be-authority"},
        )
    with pytest.raises(ValueError, match="requires only"):
        PictureSourceLocator.from_payload(
            PictureSourceKind.DOCUMENT_SURFACE,
            {"surface_kind": "pptx_slide", "ordinal": 2, "scale": 2},
        )
