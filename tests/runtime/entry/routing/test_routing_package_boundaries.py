"""Entry 路由子包的单一所有权、冷导入。"""

from __future__ import annotations

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
        "policy.py",
        "selection.py",
    }

    for module_name in ("routing_policy",):
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
