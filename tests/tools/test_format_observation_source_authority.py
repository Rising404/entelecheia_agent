"""格式观察冻结来源权威的边界覆盖。"""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path

import pytest

from personagraph.tools.documents import (
    format_observation_source_authority,
)
from personagraph.tools.execution import ToolBusinessFailure
from personagraph.tools.workspace.workspace_tools import FrozenWorkspaceToolBoundary


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"))


def _error_code(callable_) -> str:
    with pytest.raises(ToolBusinessFailure) as raised:
        callable_()
    return raised.value.error.code


def test_source_authority_imports_only_its_tool_boundary_dependencies() -> None:
    """来源权威不能取得读取器、视觉、运行时或目录的所有权。"""

    tree = _tree(Path(format_observation_source_authority.__file__))
    imported_modules = {
        node.module
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    assert imported_modules == {
        "__future__",
        "collections.abc",
        "pathlib",
        "typing",
        "configuration.paths",
        "workspace.binding",
        "execution",
        "workspace.workspace_tools",
    }
    assert any(
        isinstance(node, ast.Import)
        and any(alias.name == "hashlib" for alias in node.names)
        for node in tree.body
    )
    forbidden = ("format_observation_tools", "documents", "vision", "runtime")
    assert not any(
        forbidden_name in format_observation_source_authority.__dict__
        for forbidden_name in forbidden
    )


def test_source_authority_resolves_only_permitted_frozen_workspace_files(
    tmp_path: Path,
) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    target = root / "notes.txt"
    target.write_text("trusted source", encoding="utf-8")
    boundary = FrozenWorkspaceToolBoundary(session_id="session-1", root=root)

    resolved, relative = format_observation_source_authority._resolve_file(
        boundary,
        {"path": " notes.txt "},
        frozenset({".txt"}),
    )
    assert resolved == target.resolve()
    assert relative == "notes.txt"

    assert _error_code(
        lambda: format_observation_source_authority._resolve_file(
            boundary,
            {"path": "../outside.txt"},
            frozenset({".txt"}),
        )
    ) == "workspace_path_blocked"
    assert _error_code(
        lambda: format_observation_source_authority._resolve_file(
            boundary,
            {"path": str(target.resolve())},
            frozenset({".txt"}),
        )
    ) == "workspace_path_blocked"
    assert _error_code(
        lambda: format_observation_source_authority._resolve_file(
            boundary,
            {"path": "notes.txt"},
            frozenset({".pdf"}),
        )
    ) == "format_mismatch"


def test_source_authority_rejects_agent_private_workspace_files(
    tmp_path: Path,
) -> None:
    root = tmp_path / "workspace"
    private = root / ".personagraph" / "output" / "session-1" / "draft.txt"
    private.parent.mkdir(parents=True)
    private.write_text("not user evidence", encoding="utf-8")
    boundary = FrozenWorkspaceToolBoundary(session_id="session-1", root=root)

    assert _error_code(
        lambda: format_observation_source_authority._resolve_file(
            boundary,
            {"path": ".personagraph/output/session-1/draft.txt"},
            frozenset({".txt"}),
        )
    ) == "workspace_path_blocked"


def test_source_authority_fingerprints_bounded_input_and_rejects_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "notes.txt"
    target.write_bytes(b"first")

    identity = format_observation_source_authority._source_identity(target)
    assert identity == {
        "suffix": ".txt",
        "byte_count": 5,
        "sha256": hashlib.sha256(b"first").hexdigest(),
    }
    format_observation_source_authority._assert_unchanged(target, identity)

    target.write_bytes(b"second")
    assert _error_code(
        lambda: format_observation_source_authority._assert_unchanged(target, identity)
    ) == "source_changed_during_read"

    monkeypatch.setattr(format_observation_source_authority, "MAX_SOURCE_BYTES", 3)
    assert _error_code(
        lambda: format_observation_source_authority._source_identity(target)
    ) == "source_too_large"


def test_source_authority_guard_checks_the_frozen_root_before_and_after_handler() -> None:
    calls: list[str] = []

    class RecordingBoundary:
        def require_current_root(self) -> Path:
            calls.append("root")
            return Path(".")

    def handler(payload: dict[str, object]) -> dict[str, object]:
        calls.append("handler")
        return {"payload": payload}

    guarded = format_observation_source_authority._guard_workspace_handler(
        RecordingBoundary(),  # type: ignore[arg-type]
        handler,
    )

    assert guarded({"path": "notes.txt"}) == {"payload": {"path": "notes.txt"}}
    assert calls == ["root", "handler", "root"]
