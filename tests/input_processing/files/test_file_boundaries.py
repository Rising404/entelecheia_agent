"""输入文件识别子域的依赖与公开路径围栏。"""

from __future__ import annotations

import ast
from importlib.util import find_spec
from pathlib import Path

import pytest


FILES_ROOT = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "personagraph"
    / "input_processing"
    / "files"
)

FORBIDDEN_PREFIXES = (
    "personagraph.api",
    "personagraph.model_io",
    "personagraph.retrieval",
    "personagraph.runtime",
    "personagraph.session",
    "personagraph.tools",
    "personagraph.workspace",
)


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module)
    return found


@pytest.mark.parametrize("module", sorted(FILES_ROOT.glob("*.py")), ids=lambda p: p.name)
def test_file_processing_does_not_import_calling_layers(module: Path) -> None:
    for name in _imports(module):
        assert not name.startswith(FORBIDDEN_PREFIXES), f"{module.name} imports {name}"


def test_files_package_exposes_only_the_bounded_processing_surface() -> None:
    from personagraph.input_processing import files

    assert {
        "DetectedType",
        "FileKind",
        "detect_type",
        "read_validated_ooxml",
        "sanitize_original_name",
        "validate_ooxml",
    } <= set(files.__all__)


@pytest.mark.parametrize(
    "module_name",
    (
        "personagraph.input_processing.file_detection",
        "personagraph.input_processing.ooxml",
    ),
)
def test_retired_input_file_modules_have_no_compatibility_shell(module_name: str) -> None:
    assert find_spec(module_name) is None
