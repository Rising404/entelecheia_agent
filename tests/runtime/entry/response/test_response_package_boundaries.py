"""Entry direct-response 子包的所有权与冷导入围栏。"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys


PERSONAGRAPH_ROOT = Path(__file__).resolve().parents[4] / "src" / "personagraph"
ENTRY_ROOT = PERSONAGRAPH_ROOT / "runtime" / "entry"
RESPONSE_ROOT = ENTRY_ROOT / "response"


def test_response_model_has_one_explicit_owner_and_old_combined_model_is_absent() -> (
    None
):
    assert {path.name for path in RESPONSE_ROOT.glob("*.py")} == {
        "__init__.py",
        "model.py",
    }
    assert not (ENTRY_ROOT / "model.py").exists()
    assert importlib.util.find_spec("personagraph.runtime.entry.model") is None


def test_importing_response_package_is_cold_and_exports_no_compatibility_surface() -> (
    None
):
    code = """
import json
import sys
import personagraph.runtime.entry.response as response

loaded = {
    name
    for name in sys.modules
    if name == 'personagraph' or name.startswith('personagraph.')
}
allowed = {
    'personagraph',
    'personagraph.runtime',
    'personagraph.runtime.entry',
    'personagraph.runtime.entry.response',
}
print(json.dumps({
    'exports': response.__all__,
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
