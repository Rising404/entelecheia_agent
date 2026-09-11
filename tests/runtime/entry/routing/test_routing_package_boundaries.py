"""Entry 路由子包的单一所有权、冷导入与惰性 L2 边界。"""

from __future__ import annotations

import ast
import importlib.abc
import importlib.util
import json
from pathlib import Path
import subprocess
import sys


PERSONAGRAPH_ROOT = Path(__file__).resolve().parents[4] / "src" / "personagraph"
ROUTING_ROOT = PERSONAGRAPH_ROOT / "runtime" / "entry" / "routing"


def test_routing_modules_have_one_owner_and_retired_paths_are_absent() -> None:
    assert {path.name for path in ROUTING_ROOT.glob("*.py")} == {
        "__init__.py",
        "contracts.py",
        "policy.py",
        "ports.py",
        "selection.py",
        "task_admission.py",
    }

    for module_name in ("routing_policy", "task_admission", "task_routing"):
        assert not (
            PERSONAGRAPH_ROOT / "runtime" / "entry" / f"{module_name}.py"
        ).exists()
        assert (
            importlib.util.find_spec(f"personagraph.runtime.entry.{module_name}")
            is None
        )


def test_routing_package_import_is_cold_and_has_no_compatibility_exports() -> None:
    code = """
import json
import sys
import personagraph.runtime.entry.routing as routing

loaded = {
    name
    for name in sys.modules
    if name == 'personagraph' or name.startswith('personagraph.')
}
allowed = {
    'personagraph',
    'personagraph.runtime',
    'personagraph.runtime.entry',
    'personagraph.runtime.entry.routing',
}
print(json.dumps({
    'exports': routing.__all__,
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


def test_l2_adapters_are_only_imported_inside_l2_branches() -> None:
    for module_name in ("selection.py", "task_admission.py"):
        tree = ast.parse((ROUTING_ROOT / module_name).read_text(encoding="utf-8"))
        top_level_imports = {
            node.module or "" for node in tree.body if isinstance(node, ast.ImportFrom)
        }
        top_level_imports.update(
            alias.name
            for node in tree.body
            if isinstance(node, ast.Import)
            for alias in node.names
        )

        assert not any(name.startswith("personagraph.l2") for name in top_level_imports)


def test_l0_l1_routing_and_admission_do_not_initialize_l2() -> None:
    code = """
import importlib.abc
from types import SimpleNamespace
import sys

class RejectL2(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'personagraph.l2' or fullname.startswith('personagraph.l2.'):
            raise AssertionError(f'cold Entry routing imported {fullname}')
        return None

sys.meta_path.insert(0, RejectL2())
from personagraph.runtime.entry.routing.selection import select_entry_processing_route
from personagraph.runtime.entry.routing.task_admission import admit_entry_task_matches

classification = SimpleNamespace(processing_level='L1', task_matches=())
accepted = SimpleNamespace(session_id='session-1', turn_id='turn-1')
context = SimpleNamespace(task_catalog=SimpleNamespace(truncated=False))
route = select_entry_processing_route(
    accepted=accepted,
    classification=classification,
    applied=None,
    processing_level='L1',
)
admission = admit_entry_task_matches(
    accepted=accepted,
    classification=classification,
    context=context,
    ingress=SimpleNamespace(),
    expected_window_revision=1,
)
print(route.kind, admission.processing_level)
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )

    assert completed.stdout.strip() == "l1 L1"
