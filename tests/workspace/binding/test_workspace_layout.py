"""私有绑定工作区布局的领域行为。

这些测试有意止于文件系统边界。将返回的布局绑定到会话记录、暴露输出工具，
以及从发现结果中排除该目录树，均是由各自所有者负责的独立切片。
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from personagraph.workspace.binding import (
    LAYOUT_DIRECTORY_NAME,
    MANIFEST_FILE_NAME,
    MANIFEST_SCHEMA_VERSION,
    WorkspaceLayoutError,
    ensure_or_open_layout,
)


def test_new_root_gets_one_private_layout_manifest_and_session_directories(
    tmp_path: Path,
) -> None:
    root = tmp_path / "project"
    root.mkdir()
    original = root / "brief.md"
    original.write_text("user material", encoding="utf-8")

    layout = ensure_or_open_layout(root, session_id="session-001")

    assert layout.root == root.resolve()
    assert layout.layout_root == root / LAYOUT_DIRECTORY_NAME
    assert layout.manifest_created is True
    assert layout.output_dir == root / ".personagraph" / "output" / "session-001"
    assert layout.staging_dir == root / ".personagraph" / "staging" / "session-001"
    assert layout.output_dir.is_dir()
    assert layout.staging_dir.is_dir()
    assert original.read_text(encoding="utf-8") == "user material"

    manifest = json.loads(layout.manifest_path.read_text(encoding="utf-8"))
    assert manifest == {
        "created_by": "personagraph.workspace.layout",
        "root_identity": {
            "device": layout.root_identity.device,
            "inode": layout.root_identity.inode,
        },
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "workspace_id": layout.workspace_id,
    }


def test_existing_valid_layout_is_idempotent_and_separates_session_trees(
    tmp_path: Path,
) -> None:
    root = tmp_path / "project"
    root.mkdir()

    first = ensure_or_open_layout(root, session_id="session-a")
    bytes_before = first.manifest_path.read_bytes()
    second = ensure_or_open_layout(root, session_id="session-a")
    other = ensure_or_open_layout(root, session_id="session-b")

    assert second.manifest_created is False
    assert second.workspace_id == first.workspace_id
    assert second.root_identity == first.root_identity
    assert second.manifest_path.read_bytes() == bytes_before
    assert second.output_dir == first.output_dir
    assert other.output_dir != first.output_dir
    assert other.staging_dir != first.staging_dir
    assert other.output_dir.is_dir()
    assert other.staging_dir.is_dir()


def test_relative_root_is_canonicalised_before_it_is_returned(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "project"
    root.mkdir()
    monkeypatch.chdir(tmp_path)

    layout = ensure_or_open_layout(Path("project"), session_id="session-1")

    assert layout.root == root.resolve()


def test_an_empty_private_directory_can_be_safely_provisioned(tmp_path: Path) -> None:
    root = tmp_path / "project"
    (root / LAYOUT_DIRECTORY_NAME).mkdir(parents=True)

    layout = ensure_or_open_layout(root, session_id="session-1")

    assert layout.manifest_created is True
    assert layout.manifest_path.is_file()
    assert layout.output_dir.is_dir()
    assert layout.staging_dir.is_dir()


def test_a_leftover_private_manifest_temp_is_recovered_after_publish(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    first = ensure_or_open_layout(root, session_id="session-1")
    temporary = first.layout_root / f".manifest-{'a' * 32}.tmp"
    temporary.write_text("interrupted cleanup", encoding="utf-8")

    reopened = ensure_or_open_layout(root, session_id="session-2")

    assert reopened.workspace_id == first.workspace_id
    assert not temporary.exists()
    assert reopened.output_dir.is_dir()


def test_requested_workspace_identity_is_persisted_and_rechecked(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()

    first = ensure_or_open_layout(
        root,
        session_id="session-1",
        workspace_id="workspace-owned-by-binding",
    )

    assert first.workspace_id == "workspace-owned-by-binding"
    with pytest.raises(WorkspaceLayoutError) as raised:
        ensure_or_open_layout(
            root,
            session_id="session-2",
            workspace_id="another-workspace",
        )
    assert raised.value.code == "foreign_manifest"
    assert not (root / ".personagraph" / "output" / "session-2").exists()


def test_root_path_may_not_be_or_traverse_a_symlink(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    link = tmp_path / "project-link"
    link.symlink_to(root, target_is_directory=True)

    with pytest.raises(WorkspaceLayoutError) as raised:
        ensure_or_open_layout(link, session_id="session-1")

    assert raised.value.code == "unsafe_root"
    assert not (root / ".personagraph").exists()


def test_layout_directory_symlink_is_rejected_without_touching_its_target(
    tmp_path: Path,
) -> None:
    root = tmp_path / "project"
    root.mkdir()
    target = tmp_path / "other-layout"
    target.mkdir()
    (root / LAYOUT_DIRECTORY_NAME).symlink_to(target, target_is_directory=True)

    with pytest.raises(WorkspaceLayoutError) as raised:
        ensure_or_open_layout(root, session_id="session-1")

    assert raised.value.code == "unsafe_layout"
    assert list(target.iterdir()) == []


def test_unknown_nonempty_layout_without_manifest_fails_closed(tmp_path: Path) -> None:
    root = tmp_path / "project"
    layout_root = root / LAYOUT_DIRECTORY_NAME
    layout_root.mkdir(parents=True)
    (layout_root / "someone-elses-file.txt").write_text("do not adopt", encoding="utf-8")

    with pytest.raises(WorkspaceLayoutError) as raised:
        ensure_or_open_layout(root, session_id="session-1")

    assert raised.value.code == "unmanaged_layout"
    assert not (layout_root / "output").exists()
    assert not (layout_root / MANIFEST_FILE_NAME).exists()


@pytest.mark.parametrize(
    "payload",
    [
        b"not json\n",
        b"[]\n",
        b'{"schema_version": 1}\n',
    ],
)
def test_malformed_manifest_fails_closed(tmp_path: Path, payload: bytes) -> None:
    root = tmp_path / "project"
    layout_root = root / LAYOUT_DIRECTORY_NAME
    layout_root.mkdir(parents=True)
    (layout_root / MANIFEST_FILE_NAME).write_bytes(payload)

    with pytest.raises(WorkspaceLayoutError) as raised:
        ensure_or_open_layout(root, session_id="session-1")

    assert raised.value.code in {"malformed_manifest", "unsupported_manifest"}
    assert not (layout_root / "output").exists()


def test_foreign_root_identity_manifest_fails_closed(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    first = ensure_or_open_layout(root, session_id="session-1")
    manifest = json.loads(first.manifest_path.read_text(encoding="utf-8"))
    manifest["root_identity"]["inode"] += 1
    first.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(WorkspaceLayoutError) as raised:
        ensure_or_open_layout(root, session_id="session-2")

    assert raised.value.code == "foreign_manifest"
    assert not (root / ".personagraph" / "output" / "session-2").exists()


def test_managed_children_may_not_be_symlinks(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    first = ensure_or_open_layout(root, session_id="session-1")
    other = tmp_path / "other-output"
    other.mkdir()
    os.rmdir(first.output_dir)
    os.rmdir(first.output_root)
    first.output_root.symlink_to(other, target_is_directory=True)

    with pytest.raises(WorkspaceLayoutError) as raised:
        ensure_or_open_layout(root, session_id="session-2")

    assert raised.value.code == "unsafe_output_root"
    assert not (other / "session-2").exists()


@pytest.mark.parametrize("session_id", ("", ".", "..", "a/b", "a\\b", " space"))
def test_session_directory_identifier_cannot_escape_private_tree(
    tmp_path: Path,
    session_id: str,
) -> None:
    root = tmp_path / "project"
    root.mkdir()

    with pytest.raises(WorkspaceLayoutError) as raised:
        ensure_or_open_layout(root, session_id=session_id)

    assert raised.value.code == "invalid_identifier"
    assert not (root / LAYOUT_DIRECTORY_NAME).exists()


def test_empty_string_root_is_not_silently_treated_as_the_current_directory() -> None:
    with pytest.raises(WorkspaceLayoutError) as raised:
        ensure_or_open_layout("", session_id="session-1")

    assert raised.value.code == "invalid_root"
