from __future__ import annotations

from datetime import date, datetime, timezone

import pytest

from personagraph.input_processing.files import detection
from personagraph.workspace.files import attachments as storage
from personagraph.workspace.files import uploads
from personagraph.workspace.storage.context import current
from personagraph.workspace.files import (
    FileSource,
    WorkspaceFileAuthority,
)
from personagraph.session import store as session_store


PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64


@pytest.fixture
def project_session(tmp_path, partitioned_project_state):
    project_root = tmp_path / "project"
    project_root.mkdir()
    session_id = session_store.create_session(
        "Entelecheia",
        title="attachment storage",
        working_dir=str(project_root),
    )
    return session_id, project_root


def _store(session_id: str, **kwargs):
    with session_store.session_database_scope(session_id):
        return storage.store_attachment(session_id=session_id, **kwargs)


@pytest.mark.parametrize(
    "raw",
    [
        "../../etc/passwd",
        "..\\..\\Windows\\system32\\cmd.exe",
        "x/../../y.md",
        "..",
        "/abs/path.txt",
    ],
)
def test_untrusted_names_cannot_address_a_directory(raw):
    safe = detection.sanitize_original_name(raw)
    assert "/" not in safe and "\\" not in safe
    assert safe not in {"", ".", ".."}


def test_unbound_upload_fails_closed_without_creating_a_storage_tree(
    tmp_path,
    partitioned_project_state,
):
    with pytest.raises(storage.AttachmentProjectContextRequired):
        storage.store_attachment(
            session_id="s1",
            attachment_id="att_1",
            raw_name="evidence.png",
            payload=PNG,
        )
    assert not (tmp_path / "project").exists()
    assert not (partitioned_project_state / "session_files").exists()


def test_upload_lands_once_in_project_and_registers_immutable_identity(project_session):
    session_id, project_root = project_session
    stored = _store(
        session_id,
        attachment_id="att_1",
        raw_name="../../evil.png",
        payload=PNG,
    )

    assert stored.absolute_path.is_relative_to(project_root / "附件")
    bucket, uploaded_date, filename = stored.absolute_path.relative_to(project_root).parts
    assert bucket == "附件"
    assert date.fromisoformat(uploaded_date).isoformat() == uploaded_date
    assert filename == "evil.png"
    assert "att_1" not in stored.stored_rel_path
    assert stored.absolute_path.read_bytes() == PNG
    assert stored.detected.media_type == "image/png"
    assert stored.project_id and stored.file_id == "att_1" and stored.file_version_id
    assert not (project_root / "input").exists()
    assert not (project_root / "output").exists()

    with session_store.session_database_scope(session_id):
        database = current()
        assert database is not None
        record = WorkspaceFileAuthority(database).get_file(stored.file_id)
    assert record is not None
    assert record.relative_path == stored.stored_rel_path
    assert record.source is FileSource.USER_UPLOAD
    assert record.current_version_id == stored.file_version_id


def test_same_name_uploads_cannot_collide(project_session, monkeypatch):
    session_id, _project_root = project_session
    monkeypatch.setattr(
        uploads,
        "_normalize_timestamp",
        lambda _created_at: datetime(2026, 8, 30, tzinfo=timezone.utc),
    )
    first = _store(
        session_id,
        attachment_id="att_1",
        raw_name="shot.png",
        payload=PNG,
    )
    second = _store(
        session_id,
        attachment_id="att_2",
        raw_name="shot.png",
        payload=PNG + b"x",
    )
    assert first.absolute_path != second.absolute_path
    assert first.absolute_path.read_bytes() != second.absolute_path.read_bytes()
    assert first.stored_rel_path == "附件/2026-08-30/shot.png"
    assert second.stored_rel_path == "附件/2026-08-30/shot_2.png"
    assert first.file_id != second.file_id
    assert first.file_version_id != second.file_version_id


def test_previously_registered_uuid_attachment_path_remains_readable(project_session):
    session_id, project_root = project_session
    relative_path = "附件/20260830T123456123456Z_att_legacy/shot.png"
    existing = project_root / relative_path
    existing.parent.mkdir(parents=True)
    existing.write_bytes(PNG)

    with session_store.session_database_scope(session_id):
        database = current()
        assert database is not None
        registered = WorkspaceFileAuthority(database).register_path(
            relative_path,
            source=FileSource.USER_UPLOAD,
            file_id="att_legacy",
            media_type="image/png",
        )

        assert storage.read_verified_attachment(
            session_id,
            relative_path,
            expected_size_bytes=registered.version.size_bytes,
            expected_sha256=registered.version.content_sha256,
        ) == PNG
        assert storage.classify_session_path(session_id, existing) is storage.SessionStorageArea.INPUT
        assert registered.file.relative_path == relative_path


def test_extension_follows_bytes_not_claimed_name(project_session):
    session_id, _project_root = project_session
    stored = _store(
        session_id,
        attachment_id="att_1",
        raw_name="totally_a_doc.txt",
        payload=PNG,
    )
    assert stored.detected.media_type == "image/png"
    assert stored.original_name.endswith(".png")


def test_proven_plain_text_cannot_keep_a_richer_spoofed_suffix(project_session):
    session_id, _project_root = project_session
    stored = _store(
        session_id,
        attachment_id="att_text",
        raw_name="notes.pdf",
        payload=b"plain UTF-8 evidence 73500\n",
    )
    assert stored.detected.kind.value == "text"
    assert stored.detected.media_type == "text/plain"
    assert stored.absolute_path.suffix == ".txt"


def test_oversized_payload_is_refused_before_touching_disk(project_session):
    session_id, project_root = project_session
    with session_store.session_database_scope(session_id):
        with pytest.raises(storage.AttachmentTooLarge):
            storage.store_attachment(
                session_id=session_id,
                attachment_id="att_1",
                raw_name="big.bin",
                payload=b"x" * 2048,
                max_bytes=1024,
            )
    assert not (project_root / "附件").exists()


def test_read_requires_a_registered_project_path(project_session):
    session_id, project_root = project_session
    stored = _store(
        session_id,
        attachment_id="att_1",
        raw_name="a.png",
        payload=PNG,
    )
    unregistered = project_root / "unregistered.txt"
    unregistered.write_text("not authority", encoding="utf-8")

    with session_store.session_database_scope(session_id):
        assert storage.read_attachment(session_id, stored.stored_rel_path) == PNG
        with pytest.raises(storage.AttachmentContentMismatch):
            storage.read_attachment(session_id, "unregistered.txt")
        with pytest.raises(storage.AttachmentContentMismatch):
            storage.read_attachment(session_id, "../../../../etc/passwd")


def test_verified_read_rejects_same_size_payload_replacement(project_session):
    session_id, _project_root = project_session
    stored = _store(
        session_id,
        attachment_id="att_1",
        raw_name="a.png",
        payload=PNG,
    )
    replacement = b"\x89PNG\r\n\x1a\n" + b"\xff" * 64
    assert len(replacement) == stored.size_bytes
    stored.absolute_path.write_bytes(replacement)

    with session_store.session_database_scope(session_id):
        with pytest.raises(storage.AttachmentContentMismatch):
            storage.read_verified_attachment(
                session_id,
                stored.stored_rel_path,
                expected_size_bytes=stored.size_bytes,
                expected_sha256=stored.content_hash,
            )


@pytest.mark.parametrize("session_id", ["", "..", "a/b", "a\\b"])
def test_invalid_session_ids_never_resolve_to_a_project(session_id, project_session):
    valid_session_id, _project_root = project_session
    with session_store.session_database_scope(valid_session_id):
        with pytest.raises(ValueError):
            storage.session_root(session_id)


def test_registered_upload_is_recognised_as_user_material(project_session):
    session_id, _project_root = project_session
    stored = _store(
        session_id,
        attachment_id="att_1",
        raw_name="notes.png",
        payload=PNG,
    )
    with session_store.session_database_scope(session_id):
        assert storage.classify_session_path(session_id, stored.absolute_path) is (
            storage.SessionStorageArea.INPUT
        )


def test_registered_agent_file_is_recognised_as_agent_output(project_session):
    session_id, project_root = project_session
    produced = project_root / "draft" / "report.md"
    produced.parent.mkdir()
    produced.write_text("agent draft", encoding="utf-8")

    with session_store.session_database_scope(session_id):
        database = current()
        assert database is not None
        WorkspaceFileAuthority(database).register_path(
            "draft/report.md",
            source=FileSource.AGENT_OUTPUT,
        )
        assert storage.classify_session_path(session_id, produced) is (
            storage.SessionStorageArea.OUTPUT
        )


def test_unregistered_or_symlinked_project_paths_receive_no_authority(project_session):
    session_id, project_root = project_session
    unregistered = project_root / "stray.md"
    unregistered.write_text("not registered", encoding="utf-8")
    outside = project_root.parent / "outside.md"
    outside.write_text("outside", encoding="utf-8")
    planted = project_root / "looks-local.md"
    planted.symlink_to(outside)

    with session_store.session_database_scope(session_id):
        assert storage.classify_session_path(session_id, project_root) is None
        assert storage.classify_session_path(session_id, unregistered) is None
        assert storage.classify_session_path(session_id, planted) is None
