"""Entry 上下文子包的路径、冷导入与依赖方向守卫。"""

from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
import subprocess
import sys


PERSONAGRAPH_ROOT = Path(__file__).resolve().parents[4] / "src" / "personagraph"
CONTEXT_ROOT = PERSONAGRAPH_ROOT / "runtime" / "entry" / "context"


def test_context_modules_have_one_owner_and_retired_paths_are_absent() -> None:
    assert {path.name for path in CONTEXT_ROOT.glob("*.py")} == {
        "__init__.py",
        "application.py",
        "attachments.py",
        "contracts.py",
        "ports.py",
        "task_catalog.py",
        "task_catalog_projection.py",
    }

    for module_name in (
        "attachments",
        "context_assembly",
        "task_catalog",
        "task_catalog_projection",
    ):
        assert not (
            PERSONAGRAPH_ROOT / "runtime" / "entry" / f"{module_name}.py"
        ).exists()
        assert importlib.util.find_spec(
            f"personagraph.runtime.entry.{module_name}"
        ) is None


def test_context_package_import_is_cold_and_exports_no_compatibility_surface() -> None:
    code = """
import json
import sys
import personagraph.runtime.entry.context as context

loaded = {
    name
    for name in sys.modules
    if name == 'personagraph' or name.startswith('personagraph.')
}
allowed = {
    'personagraph',
    'personagraph.runtime',
    'personagraph.runtime.entry',
    'personagraph.runtime.entry.context',
}
print(json.dumps({
    'exports': context.__all__,
    'unexpected': sorted(loaded - allowed),
}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(completed.stdout) == {"exports": [], "unexpected": []}


def test_context_application_does_not_import_l1_or_l2() -> None:
    tree = ast.parse((CONTEXT_ROOT / "application.py").read_text(encoding="utf-8"))
    imports = {
        node.module or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }
    imports.update(
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    )

    assert not any(
        name.startswith(("personagraph.runtime.l1", "personagraph.l2"))
        for name in imports
    )
