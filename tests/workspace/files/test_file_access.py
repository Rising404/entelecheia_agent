"""Only the selected path is observed; identities never replace live authority."""

import os
import pytest

from personagraph.workspace.files import FileSource, WorkspaceFileAuthority
from personagraph.workspace.files.access import FileAccess, FileAccessError
from personagraph.workspace.storage.database import DocumentDatabase


def test_readonly_access_does_not_create_store_and_tracks_real_versions(tmp_path):
    root = tmp_path / "files"
    root.mkdir()
    path = root / "notes.txt"
    path.write_text("original")
    database = DocumentDatabase("project", root, tmp_path / "file-access-state" / "documents.sqlite")
    access = FileAccess(database=database, validate_path=lambda _: True)
    unregistered = access.resolve_path("notes.txt")
    assert unregistered.file_id is None and not database.db_path.parent.exists()
    authority = WorkspaceFileAuthority(database)
    first = authority.ensure_current_path("notes.txt", source=FileSource.WORKSPACE_EXISTING)
    source = access.resolve_file(file_id=first.file.file_id)
    assert access.revalidate(source)
    stamp = path.stat()
    os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns + 1000000))
    touched = access.resolve_file(file_id=first.file.file_id)
    assert touched.file_version_id == source.file_version_id
    assert not access.revalidate(source)
    path.write_text("changed")
    with pytest.raises(FileAccessError, match="file_content_changed"):
        access.resolve_file(file_id=source.file_id)
    changed = access.resolve_file(file_id=source.file_id, allow_changed=True)
    assert changed.file_id == source.file_id and changed.file_version_id is None
    assert access.revalidate(changed)


@pytest.mark.parametrize("target", ["../escape.txt", "directory", "link.txt", "pipe"])
def test_unsafe_targets_produce_per_file_access_errors(tmp_path, target):
    root = tmp_path / "files"
    root.mkdir()
    (root / "directory").mkdir()
    (tmp_path / "escape.txt").write_text("private")
    (root / "link.txt").symlink_to(tmp_path / "escape.txt")
    os.mkfifo(root / "pipe")
    database = DocumentDatabase("project", root, tmp_path / "documents.sqlite")
    access = FileAccess(database=database, validate_path=lambda _: True)
    with pytest.raises(FileAccessError):
        access.resolve_path(target)
    assert not database.db_path.exists()


def test_authorized_versions_include_outputs_but_still_apply_current_access(tmp_path):
    root = tmp_path / "files"
    root.mkdir()
    database = DocumentDatabase("project", root, tmp_path / "documents.sqlite")
    authority = WorkspaceFileAuthority(database)
    (root / "draft.txt").write_text("agent output")
    (root / "blocked.txt").write_text("not authorized")
    output = authority.ensure_current_path("draft.txt", source=FileSource.AGENT_OUTPUT)
    authority.ensure_current_path("blocked.txt", source=FileSource.WORKSPACE_EXISTING)
    access = FileAccess(
        database=database, validate_path=lambda path: path == str(root / "draft.txt"),
    )

    assert access.authorized_versions() == (
        (output.file.file_id, output.file.current_version_id),
    )
