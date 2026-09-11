"""在没有路径歧义的情况下识别智能体私有工作区树。"""

from __future__ import annotations

import ast
from pathlib import Path, PurePosixPath

import pytest

from personagraph.workspace.binding import (
    RESERVED_DIRECTORY_NAME,
    ReservedWorkspacePathError,
    is_reserved_workspace_path,
    normalize_workspace_relative_path,
    workspace_relative_path,
)


def test_root_is_not_reserved_but_reserved_root_and_descendants_are(tmp_path: Path) -> None:
    root = tmp_path / "bound-workspace"
    root.mkdir()

    assert is_reserved_workspace_path(root, root) is False
    assert is_reserved_workspace_path(root, ".") is False
    assert is_reserved_workspace_path(root, RESERVED_DIRECTORY_NAME) is True
    assert is_reserved_workspace_path(root, root / RESERVED_DIRECTORY_NAME) is True
    assert is_reserved_workspace_path(root, ".personagraph/output/session-1/report.md") is True
    assert is_reserved_workspace_path(root, ".PERSONAGRAPH/output/session-1/report.md") is True


@pytest.mark.parametrize(
    "candidate",
    (
        ".personagraph-notes.md",
        "notes/.personagraph-notes.md",
        "notes/personagraph/.personagraph-state.txt",
        "notes/.personagraph/archive.txt",
    ),
)
def test_only_the_reserved_root_child_is_private(tmp_path: Path, candidate: str) -> None:
    root = tmp_path / "bound-workspace"
    root.mkdir()

    assert is_reserved_workspace_path(root, candidate) is False


def test_relative_paths_normalize_to_stable_posix_form(tmp_path: Path) -> None:
    root = tmp_path / "bound-workspace"
    root.mkdir()

    assert normalize_workspace_relative_path("notes//2026/./brief.md") == PurePosixPath(
        "notes/2026/brief.md"
    )
    assert workspace_relative_path(root, root / "notes" / "brief.md") == PurePosixPath(
        "notes/brief.md"
    )
    assert workspace_relative_path(root, ".") == PurePosixPath(".")


@pytest.mark.parametrize(
    ("candidate", "code"),
    (
        ("", "invalid_path"),
        ("../outside.txt", "path_traversal"),
        ("notes/../outside.txt", "path_traversal"),
        ("/etc/passwd", "absolute_path"),
        ("C:/outside.txt", "absolute_path"),
        (r"notes\\private.txt", "invalid_path"),
    ),
)
def test_unsafe_relative_forms_are_stably_rejected(
    candidate: str, code: str
) -> None:
    with pytest.raises(ReservedWorkspacePathError) as raised:
        normalize_workspace_relative_path(candidate)

    assert raised.value.code == code


def test_absolute_candidate_outside_the_exact_bound_root_is_rejected(
    tmp_path: Path,
) -> None:
    root = tmp_path / "bound-workspace"
    root.mkdir()
    outside = tmp_path / "another-workspace"
    outside.mkdir()

    with pytest.raises(ReservedWorkspacePathError) as raised:
        is_reserved_workspace_path(root, outside / RESERVED_DIRECTORY_NAME)

    assert raised.value.code == "outside_bound_root"


def test_existing_symlink_components_are_rejected_even_when_they_point_inside_root(
    tmp_path: Path,
) -> None:
    root = tmp_path / "bound-workspace"
    root.mkdir()
    (root / "inside").mkdir()
    (root / "inside" / "placeholder.txt").write_text("x", encoding="utf-8")
    (root / "link").symlink_to(root / "inside", target_is_directory=True)
    (root / RESERVED_DIRECTORY_NAME).symlink_to(root / "inside", target_is_directory=True)

    for candidate in ("link/.personagraph", RESERVED_DIRECTORY_NAME):
        with pytest.raises(ReservedWorkspacePathError) as raised:
            is_reserved_workspace_path(root, candidate)
        assert raised.value.code == "symlink_path"


def test_missing_reserved_descendants_can_be_classified_before_provisioning(
    tmp_path: Path,
) -> None:
    root = tmp_path / "bound-workspace"
    root.mkdir()

    assert is_reserved_workspace_path(root, ".personagraph/staging/session-1") is True


def test_symlink_bound_root_is_rejected(tmp_path: Path) -> None:
    real_root = tmp_path / "real-workspace"
    real_root.mkdir()
    linked_root = tmp_path / "linked-workspace"
    linked_root.symlink_to(real_root, target_is_directory=True)

    with pytest.raises(ReservedWorkspacePathError) as raised:
        is_reserved_workspace_path(linked_root, ".personagraph")

    assert raised.value.code == "symlink_path"


def test_reserved_policy_depends_only_on_the_standard_library() -> None:
    module_path = (
        Path(__file__).resolve().parents[3]
        / "src"
        / "personagraph"
        / "workspace"
        / "binding"
        / "reserved_paths.py"
    )
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    imported = {
        node.module.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    } | {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }

    assert imported <= {"__future__", "os", "pathlib", "stat"}
