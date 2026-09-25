from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from threading import Barrier

import pytest

from personagraph.workspace.files import admission
from personagraph.workspace.files import (
    FileSource,
    ProjectFilePathError,
    WorkspaceFileAuthority,
    ProjectUploadConflict,
    ProjectUploadPathError,
    ProjectUploadService,
)
from personagraph.workspace.storage import DocumentDatabase
from personagraph.workspace.storage.context import bind as bind_project_documents
from personagraph.input_processing.files import fingerprint_file
from personagraph.workspace.documents import application as docstore


FIXED_TIME = datetime(2026, 8, 30, 12, 34, 56, 123456, tzinfo=timezone.utc)


def _database(tmp_path, project_id: str = "project-a") -> DocumentDatabase:
    root = tmp_path / "roots" / project_id
    root.mkdir(parents=True)
    return DocumentDatabase(
        project_id,
        root,
        tmp_path / "var" / "projects" / project_id / "documents.sqlite",
    )


def test_upload_lands_once_in_project_root_and_registers_user_source(tmp_path):
    database = _database(tmp_path)
    service = ProjectUploadService(database)
    payload = "第一版".encode()

    stored = service.store_upload(
        original_name="../../研究?笔记.txt",
        payload=payload,
        file_id="file-a",
        created_at=FIXED_TIME,
    )

    assert stored.relative_path == "附件/2026-08-30/研究_笔记.txt"
    assert stored.absolute_path == database.project_root / stored.relative_path
    assert stored.absolute_path.read_bytes() == payload
    assert stored.file_id == "file-a"
    assert stored.registration.file.source is FileSource.USER_UPLOAD
    assert stored.registration.version.source is FileSource.USER_UPLOAD
    assert stored.registration.version.version_number == 1
    assert stored.content_sha256 == hashlib.sha256(payload).hexdigest()
    assert not tuple(database.project_root.rglob("*.part"))

    with database.connect() as conn:
        file_row = conn.execute("SELECT * FROM files").fetchone()
        version_row = conn.execute("SELECT * FROM file_versions").fetchone()
        event_row = conn.execute("SELECT * FROM file_events").fetchone()
    assert file_row["origin"] == "user_upload"
    assert version_row["producer"] == "user_upload"
    assert event_row["origin"] == "user_upload"
    assert file_row["current_version_id"] == version_row["id"]


def test_same_day_same_name_uploads_keep_separate_bytes_and_identities(tmp_path):
    database = _database(tmp_path)
    service = ProjectUploadService(database)
    uploads = [
        service.store_upload(
            original_name="notes.txt",
            payload=f"independent upload {number}".encode(),
            file_id=f"file-{number}",
            created_at=FIXED_TIME,
        )
        for number in range(3)
    ]

    assert [upload.relative_path for upload in uploads] == [
        "附件/2026-08-30/notes.txt",
        "附件/2026-08-30/notes_2.txt",
        "附件/2026-08-30/notes_3.txt",
    ]
    assert len({upload.file_id for upload in uploads}) == 3
    assert len({upload.file_version_id for upload in uploads}) == 3
    for number, upload in enumerate(uploads):
        assert upload.absolute_path.read_bytes() == f"independent upload {number}".encode()
        assert upload.registration.version.version_number == 1
        assert upload.content_sha256 == hashlib.sha256(upload.absolute_path.read_bytes()).hexdigest()


def test_different_days_keep_original_name_in_separate_date_directories(tmp_path):
    database = _database(tmp_path)
    service = ProjectUploadService(database)
    first = service.store_upload(
        original_name="notes.txt",
        payload=b"day one",
        file_id="file-a",
        created_at=FIXED_TIME,
    )
    second = service.store_upload(
        original_name="notes.txt",
        payload=b"day two",
        file_id="file-b",
        created_at=FIXED_TIME.replace(day=31),
    )

    assert first.relative_path == "附件/2026-08-30/notes.txt"
    assert second.relative_path == "附件/2026-08-31/notes.txt"
    assert first.absolute_path.read_bytes() == b"day one"
    assert second.absolute_path.read_bytes() == b"day two"


def test_same_name_suffix_respects_utf8_filename_limit(tmp_path):
    database = _database(tmp_path)
    service = ProjectUploadService(database)
    original_name = "研" * 83 + "aa.txt"
    assert len(original_name.encode("utf-8")) == 255
    first = service.store_upload(
        original_name=original_name,
        payload=b"first upload",
        file_id="file-a",
        created_at=FIXED_TIME,
    )
    second = service.store_upload(
        original_name=original_name,
        payload=b"second upload",
        file_id="file-b",
        created_at=FIXED_TIME,
    )

    assert first.absolute_path.name == original_name
    assert second.absolute_path.name.endswith("_2.txt")
    assert len(second.absolute_path.name.encode("utf-8")) <= 255
    assert first.absolute_path.read_bytes() == b"first upload"
    assert second.absolute_path.read_bytes() == b"second upload"


def test_concurrent_same_name_uploads_publish_independent_registered_files(tmp_path):
    database = _database(tmp_path)
    with database.connect():
        pass
    count = 4
    start = Barrier(count)

    def upload(number):
        start.wait(timeout=10)
        return ProjectUploadService(database).store_upload(
            original_name="notes.txt",
            payload=f"concurrent upload {number}".encode(),
            file_id=f"file-{number}",
            created_at=FIXED_TIME,
        )

    with ThreadPoolExecutor(max_workers=count) as executor:
        stored = list(executor.map(upload, range(count)))

    assert {item.relative_path for item in stored} == {
        "附件/2026-08-30/notes.txt",
        "附件/2026-08-30/notes_2.txt",
        "附件/2026-08-30/notes_3.txt",
        "附件/2026-08-30/notes_4.txt",
    }
    authority = WorkspaceFileAuthority(database)
    for number, item in enumerate(stored):
        assert item.absolute_path.read_bytes() == f"concurrent upload {number}".encode()
        assert authority.get_file_by_relative_path(item.relative_path) == item.registration.file
        assert authority.list_versions(item.file_id) == (item.registration.version,)
    assert not tuple(database.project_root.rglob("*.part"))


def test_upload_does_not_reuse_a_deleted_but_registered_path(tmp_path):
    database = _database(tmp_path)
    service = ProjectUploadService(database)
    first = service.store_upload(
        original_name="notes.txt",
        payload=b"original",
        file_id="file-a",
        created_at=FIXED_TIME,
    )
    first.absolute_path.unlink()

    second = service.store_upload(
        original_name="notes.txt",
        payload=b"replacement upload",
        file_id="file-b",
        created_at=FIXED_TIME,
    )

    assert second.relative_path == "附件/2026-08-30/notes_2.txt"
    assert not first.absolute_path.exists()
    assert second.absolute_path.read_bytes() == b"replacement upload"
    authority = WorkspaceFileAuthority(database)
    assert authority.get_file(first.file_id) == first.registration.file
    assert authority.list_versions(first.file_id) == (first.registration.version,)
    assert second.file_id != first.file_id


def test_same_relative_path_appends_version_and_old_version_remains_queryable(
    tmp_path,
):
    database = _database(tmp_path)
    authority = WorkspaceFileAuthority(database)
    stored = ProjectUploadService(database, authority).store_upload(
        original_name="notes.txt",
        payload=b"version one",
        file_id="file-a",
        created_at=FIXED_TIME,
    )
    first_version_id = stored.file_version_id
    stored.absolute_path.write_bytes(b"version two")

    second = authority.register_path(
        stored.relative_path,
        source=FileSource.USER_UPLOAD,
        file_id=stored.file_id,
        occurred_at="2026-08-30T12:35:00Z",
    )

    versions = authority.list_versions(stored.file_id)
    old_version = authority.get_version(first_version_id)
    current_file = authority.get_file(stored.file_id)
    assert [version.version_number for version in versions] == [1, 2]
    assert [version.content_sha256 for version in versions] == [
        hashlib.sha256(b"version one").hexdigest(),
        hashlib.sha256(b"version two").hexdigest(),
    ]
    assert old_version == versions[0]
    assert second.version == versions[1]
    assert second.created_file is False
    assert current_file is not None
    assert current_file.current_version_id == second.version.file_version_id


def test_ensure_current_path_observes_each_request_once(tmp_path, monkeypatch):
    database = _database(tmp_path)
    path = database.project_root / "notes.txt"
    path.write_bytes(b"stable")
    authority = WorkspaceFileAuthority(database)
    observed: list[str] = []
    original = admission.observe_project_file

    def observe_once(root, relative_path):
        observed.append(relative_path)
        return original(root, relative_path)

    monkeypatch.setattr(admission, "observe_project_file", observe_once)

    first = authority.ensure_current_path(
        "notes.txt",
        source=FileSource.WORKSPACE_EXISTING,
    )
    replay = authority.ensure_current_path(
        "notes.txt",
        source=FileSource.WORKSPACE_EXISTING,
    )

    assert observed == ["notes.txt", "notes.txt"]
    assert replay.version == first.version
    assert authority.list_versions(first.file.file_id) == (first.version,)


def test_mtime_change_records_observation_without_advancing_content_version(tmp_path):
    database = _database(tmp_path)
    path = database.project_root / "notes.txt"
    path.write_bytes(b"first content")
    authority = WorkspaceFileAuthority(database)
    first = authority.ensure_current_path(
        "notes.txt",
        source=FileSource.WORKSPACE_EXISTING,
    )
    modified_ns = path.stat().st_mtime_ns + 1_000_000_000
    os.utime(path, ns=(modified_ns, modified_ns))

    observed = authority.ensure_current_path(
        "notes.txt",
        source=FileSource.WORKSPACE_EXISTING,
    )
    replay = authority.ensure_current_path(
        "notes.txt",
        source=FileSource.WORKSPACE_EXISTING,
    )

    assert observed.version == replay.version == first.version
    assert observed.file.file_id == first.file.file_id
    assert observed.file.observed_mtime_ns == modified_ns
    assert authority.get_version(first.version.file_version_id) == first.version
    assert authority.list_versions(first.file.file_id) == (first.version,)
    with database.connect() as conn:
        events = conn.execute(
            "SELECT * FROM file_events WHERE event_type='file_observed'"
        ).fetchall()
    assert len(events) == 1
    assert events[0]["file_version_id"] == first.version.file_version_id
    assert json.loads(events[0]["metadata_json"]) == {"source_mtime_ns": modified_ns}

    # 同大小、同时间的新字节仍须推进内容版本，不能用 stat 代替哈希。
    path.write_bytes(b"other content")
    os.utime(path, ns=(modified_ns, modified_ns))
    changed = authority.ensure_current_path(
        "notes.txt",
        source=FileSource.WORKSPACE_EXISTING,
    )
    assert changed.file.file_id == first.file.file_id
    assert changed.version.file_version_id != first.version.file_version_id
    assert changed.version.content_sha256 == hashlib.sha256(b"other content").hexdigest()
    assert authority.get_version(first.version.file_version_id) == first.version


def test_get_file_with_version_returns_one_exact_project_owned_pair(tmp_path):
    database = _database(tmp_path)
    authority = WorkspaceFileAuthority(database)
    first = ProjectUploadService(database, authority).store_upload(
        original_name="notes.txt",
        payload=b"version one",
        file_id="file-a",
        created_at=FIXED_TIME,
    )
    first_version_id = first.file_version_id
    first.absolute_path.write_bytes(b"version two")
    second = authority.register_path(
        first.relative_path,
        source=FileSource.USER_UPLOAD,
        file_id=first.file_id,
        occurred_at="2026-08-30T12:35:00Z",
    )

    exact = authority.get_file_with_version(first.file_id, first_version_id)

    assert exact is not None
    file_record, version_record = exact
    assert file_record.file_id == first.file_id
    assert file_record.project_id == database.project_id
    assert file_record.current_version_id == second.version.file_version_id
    assert version_record.file_version_id == first_version_id
    assert version_record.file_id == file_record.file_id
    assert version_record.content_sha256 == hashlib.sha256(b"version one").hexdigest()


def test_get_file_with_version_rejects_cross_file_pair_and_invalid_ids(tmp_path):
    database = _database(tmp_path)
    service = ProjectUploadService(database)
    first = service.store_upload(
        original_name="first.txt",
        payload=b"first",
        file_id="file-a",
        created_at=FIXED_TIME,
    )
    second = service.store_upload(
        original_name="second.txt",
        payload=b"second",
        file_id="file-b",
        created_at="2026-08-30T12:35:00Z",
    )
    authority = WorkspaceFileAuthority(database)

    assert authority.get_file_with_version(
        first.file_id,
        second.file_version_id,
    ) is None
    with pytest.raises(ValueError, match="file_id"):
        authority.get_file_with_version("../file-a", first.file_version_id)
    with pytest.raises(ValueError, match="file_version_id"):
        authority.get_file_with_version(first.file_id, "../version-a")


@pytest.mark.parametrize("source", tuple(FileSource))
def test_project_document_generation_links_to_exact_file_version(tmp_path, source):
    database = _database(tmp_path, f"project-{source.value.replace('_', '-')}")
    relative_path = "evidence.txt"
    absolute_path = database.project_root / relative_path
    absolute_path.write_text("project evidence", encoding="utf-8")
    registration = WorkspaceFileAuthority(database).register_path(
        relative_path,
        source=source,
        media_type="text/plain",
    )

    with bind_project_documents(database):
        stored = docstore.ingest(
            str(absolute_path.resolve()),
            "evidence",
            "text/plain",
            [{"content": "project evidence", "loc": "L1"}],
            file_id=registration.file.file_id,
            file_version_id=registration.version.file_version_id,
            source_fingerprint=fingerprint_file(absolute_path),
        )

    with database.connect() as conn:
        row = conn.execute(
            "SELECT document.file_id, version.file_version_id, file.origin, "
            "file_version.producer FROM documents AS document "
            "JOIN document_versions AS version "
            "ON version.id=document.current_version_id "
            "JOIN files AS file ON file.id=document.file_id "
            "JOIN file_versions AS file_version "
            "ON file_version.id=version.file_version_id "
            "WHERE document.id=?",
            (stored["doc_id"],),
        ).fetchone()

    assert row is not None
    assert row["file_id"] == registration.file.file_id
    assert row["file_version_id"] == registration.version.file_version_id
    assert row["origin"] == source.value
    assert row["producer"] == source.value


def test_project_document_ingest_fails_closed_without_file_linkage(tmp_path):
    database = _database(tmp_path)
    source = database.project_root / "unregistered.txt"
    source.write_text("unregistered", encoding="utf-8")

    with bind_project_documents(database):
        with pytest.raises(ValueError, match="requires file_id and file_version_id"):
            docstore.ingest(
                str(source.resolve()),
                "unregistered",
                "text/plain",
                [{"content": "unregistered", "loc": "L1"}],
                source_fingerprint=fingerprint_file(source),
            )

    with database.connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 0


@pytest.mark.parametrize(
    "relative_path",
    ("", "/absolute.txt", "../outside.txt", "dir/../outside.txt", "dir\\file.txt"),
)
def test_registration_rejects_path_traversal_before_reading(tmp_path, relative_path):
    database = _database(tmp_path)

    with pytest.raises(ProjectFilePathError):
        WorkspaceFileAuthority(database).register_path(
            relative_path,
            source=FileSource.WORKSPACE_EXISTING,
        )


def test_registration_rejects_symlink_file_and_directory_escape(tmp_path):
    database = _database(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("secret", encoding="utf-8")
    (database.project_root / "linked-dir").symlink_to(
        outside,
        target_is_directory=True,
    )
    (database.project_root / "linked-file.txt").symlink_to(outside / "secret.txt")
    authority = WorkspaceFileAuthority(database)

    with pytest.raises(ProjectFilePathError):
        authority.register_path(
            "linked-dir/secret.txt",
            source=FileSource.WORKSPACE_EXISTING,
        )
    with pytest.raises(ProjectFilePathError):
        authority.register_path(
            "linked-file.txt",
            source=FileSource.WORKSPACE_EXISTING,
        )

    with database.connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM files").fetchone()[0] == 0


def test_upload_rejects_symlinked_attachment_bucket_without_writing_outside(
    tmp_path,
):
    database = _database(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (database.project_root / "附件").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ProjectUploadPathError):
        ProjectUploadService(database).store_upload(
            original_name="notes.txt",
            payload=b"do not escape",
            file_id="file-a",
            created_at=FIXED_TIME,
        )

    assert list(outside.iterdir()) == []


def test_upload_rejects_symlinked_date_directory(tmp_path):
    database = _database(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    attachments = database.project_root / "附件"
    attachments.mkdir()
    directory = "2026-08-30"
    (attachments / directory).symlink_to(outside, target_is_directory=True)

    with pytest.raises(ProjectUploadPathError):
        ProjectUploadService(database).store_upload(
            original_name="notes.txt",
            payload=b"do not escape",
            file_id="file-a",
            created_at=FIXED_TIME,
        )

    assert list(outside.iterdir()) == []


def test_upload_never_overwrites_a_preplanted_final_symlink(tmp_path):
    database = _database(tmp_path)
    outside = tmp_path / "outside.txt"
    outside.write_bytes(b"outside")
    destination = database.project_root / "附件" / "2026-08-30"
    destination.mkdir(parents=True)
    (destination / "notes.txt").symlink_to(outside)

    with pytest.raises(ProjectUploadConflict):
        ProjectUploadService(database).store_upload(
            original_name="notes.txt",
            payload=b"replacement",
            file_id="file-a",
            created_at=FIXED_TIME,
        )

    assert outside.read_bytes() == b"outside"
    assert (destination / "notes.txt").is_symlink()
    assert not tuple(destination.glob("*.part"))


def test_unsafe_file_id_is_rejected_before_project_write(tmp_path):
    database = _database(tmp_path)

    with pytest.raises(ValueError):
        ProjectUploadService(database).store_upload(
            original_name="notes.txt",
            payload=b"payload",
            file_id="../escape",
            created_at=FIXED_TIME,
        )

    assert list(database.project_root.iterdir()) == []


def test_upload_removes_published_file_when_database_registration_fails(tmp_path):
    database = _database(tmp_path)
    first = ProjectUploadService(database).store_upload(
        original_name="notes.txt",
        payload=b"already registered",
        file_id="file-a",
        created_at=FIXED_TIME,
    )

    class FailingAuthority(WorkspaceFileAuthority):
        def register_path(self, *args, **kwargs):
            raise RuntimeError("database unavailable")

    with pytest.raises(RuntimeError, match="database unavailable"):
        ProjectUploadService(database, FailingAuthority(database)).store_upload(
            original_name="notes.txt",
            payload=b"payload",
            file_id="file-b",
            created_at=FIXED_TIME,
        )

    assert list(first.absolute_path.parent.iterdir()) == [first.absolute_path]
    assert first.absolute_path.read_bytes() == b"already registered"
    authority = WorkspaceFileAuthority(database)
    assert authority.get_file(first.file_id) == first.registration.file
    assert authority.get_file("file-b") is None
