"""Guard the pure Vision-processing boundary and retired state owners."""

from __future__ import annotations

import ast
from importlib.util import find_spec
from pathlib import Path

import pytest


VISION_ROOT = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "personagraph"
    / "input_processing"
    / "vision"
)
FORBIDDEN_PREFIXES = (
    "personagraph.api",
    "personagraph.retrieval",
    "personagraph.runtime",
    "personagraph.session",
    "personagraph.tools",
    "personagraph.workspace",
)


def _modules() -> tuple[Path, ...]:
    return tuple(sorted(VISION_ROOT.rglob("*.py")))


def _imports(path: Path) -> tuple[str, ...]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            imports.append(node.module)
    return tuple(imports)


def _module_exists(module_name: str) -> bool:
    try:
        return find_spec(module_name) is not None
    except ModuleNotFoundError:
        return False


@pytest.mark.parametrize(
    "path",
    _modules(),
    ids=lambda path: str(path.relative_to(VISION_ROOT)),
)
def test_vision_processing_does_not_depend_on_product_state(path: Path) -> None:
    violations = tuple(
        imported
        for imported in _imports(path)
        if imported.startswith(FORBIDDEN_PREFIXES)
    )

    assert violations == ()


def test_vision_processing_has_explicit_subdomains() -> None:
    assert {
        "imaging",
        "ocr",
        "providers",
    } <= {path.name for path in VISION_ROOT.iterdir() if path.is_dir()}
    assert not (VISION_ROOT / "disclosure.py").exists()
    assert not (VISION_ROOT / "durable_call_ledger.py").exists()


@pytest.mark.parametrize(
    "module_name",
    (
        "personagraph.input_processing.vision.adapters",
        "personagraph.input_processing.vision.disclosure",
        "personagraph.input_processing.vision.durable_call_ledger",
        "personagraph.input_processing.vision.http_adapter",
        "personagraph.input_processing.vision.payload",
    ),
)
def test_retired_vision_modules_have_no_compatibility_shell(
    module_name: str,
) -> None:
    assert not _module_exists(module_name)


def test_stateful_vision_owners_are_outside_input_processing() -> None:
    assert not _module_exists("personagraph.session.visual_disclosure")
    assert _module_exists("personagraph.runtime.model_calls.vision")


def test_vision_facade_does_not_resell_stateful_ledgers() -> None:
    from personagraph.input_processing import vision

    assert "DisclosureLedger" not in vision.__all__
    assert not hasattr(vision, "SqliteMountedVisualCallLedger")
