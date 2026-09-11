"""产物新建权限不会覆盖既存文件，也不通过路径或链接扩大作用域。"""

from concurrent.futures import ThreadPoolExecutor

import pytest

from personagraph.workspace.files import (
    FileSource, ProjectFileError, ProjectFilePathError, ProjectOutputConflict,
    ProjectOutputService, WorkspaceFileAuthority,
)
from personagraph.workspace.files.access import FileAccess
from personagraph.workspace.storage.database import DocumentDatabase


@pytest.fixture
def output_project(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    database = DocumentDatabase("project", root, tmp_path / "state" / "documents.sqlite")
    identity = root.stat()
    return database, ProjectOutputService(
        database, root_device=identity.st_dev, root_inode=identity.st_ino,
    )


def test_create_output_registers_real_file_identity_and_never_overwrites(output_project):
    database, service = output_project
    result = service.create_text(path="reports/summary.md", content="# Result\n\n准确率 84.25%。")
    target = database.project_root / "output/reports/summary.md"
    assert result.file.relative_path == "output/reports/summary.md"
    assert result.file.source is FileSource.AGENT_OUTPUT
    assert result.version.source is FileSource.AGENT_OUTPUT
    assert result.version.size_bytes == target.stat().st_size
    authority = WorkspaceFileAuthority(database)
    assert authority.get_file_with_version(
        result.file.file_id, result.version.file_version_id,
    ) == (result.file, result.version)
    access = FileAccess(database=database, validate_path=lambda _: True)
    assert access.resolve_file(file_id=result.file.file_id).canonical_path == str(target)

    with pytest.raises(ProjectOutputConflict):
        service.create_text(path="reports/summary.md", content="replace user file")
    assert target.read_text() == "# Result\n\n准确率 84.25%。"
    assert not list(target.parent.glob(".output-*.part"))
    named = service.create_text(path="secret-format.md", content="public format documentation")
    assert named.file.relative_path == "output/secret-format.md"


@pytest.mark.parametrize("path", [
    "../outside.txt", "../.personagraph/state", "/tmp/absolute.txt", "C:/absolute.txt",
    "nested/../../outside.txt", "nested\\file.txt", ".git/config", "",
])
def test_output_rejects_outside_private_and_invalid_paths_before_creating(output_project, path):
    database, service = output_project
    with pytest.raises(ProjectFilePathError):
        service.create_text(path=path, content="not written")
    assert not (database.project_root / "output").exists()
    assert not database.db_path.exists()


@pytest.mark.parametrize("linked", ["output", "nested", "target"])
def test_output_rejects_symlinks_without_touching_their_targets(output_project, tmp_path, linked):
    database, service = output_project
    outside = tmp_path / "outside"
    outside.mkdir()
    original = outside / "result.txt"
    original.write_text("untouched")
    output = database.project_root / "output"
    if linked == "output":
        output.symlink_to(outside, target_is_directory=True)
        path = "result.txt"
    else:
        output.mkdir()
        if linked == "nested":
            (output / "nested").symlink_to(outside, target_is_directory=True)
            path = "nested/result.txt"
        else:
            (output / "result.txt").symlink_to(original)
            path = "result.txt"
    with pytest.raises(ProjectFilePathError):
        service.create_text(path=path, content="replace")
    assert original.read_text() == "untouched"
    assert not database.db_path.exists()


def test_concurrent_output_publication_has_one_winner(output_project):
    database, service = output_project
    # 初始化数据库不参与待测的文件 create-if-absent 竞争。
    with database.connect():
        pass

    def create(content):
        try:
            return service.create_text(path="race.txt", content=content)
        except ProjectOutputConflict:
            return None

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(create, ["first", "second"]))
    assert sum(result is not None for result in results) == 1
    assert (database.project_root / "output/race.txt").read_text() in {"first", "second"}
    assert not list((database.project_root / "output").glob(".output-*.part"))


def test_failed_registration_removes_only_the_unregistered_output(output_project, monkeypatch):
    database, service = output_project

    def fail(*_args, **_kwargs):
        raise ProjectFileError("registration failed")

    monkeypatch.setattr(WorkspaceFileAuthority, "register_path", fail)
    with pytest.raises(ProjectFileError, match="registration failed"):
        service.create_text(path="failed.txt", content="unregistered")
    assert list((database.project_root / "output").iterdir()) == []


def test_root_replacement_and_oversized_text_are_rejected(output_project):
    database, service = output_project
    with pytest.raises(ValueError, match="128 KiB"):
        service.create_text(path="large.txt", content="字" * 50_000)
    root = database.project_root
    root.rename(root.with_name("original"))
    root.mkdir()
    with pytest.raises(ProjectFilePathError, match="root changed"):
        service.create_text(path="changed.txt", content="not written")
    assert not (root / "output").exists()
