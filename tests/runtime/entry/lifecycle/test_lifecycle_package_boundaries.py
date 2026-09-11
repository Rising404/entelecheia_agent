"""Entry 生命周期子包的物理归属与冷启动边界。"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from personagraph.runtime.entry import lifecycle


_LIFECYCLE_MODULES = (
    "active_window",
    "contracts",
    "persisted_turn",
    "ports",
    "replay",
    "settlement",
    "window_audit",
)

_RETIRED_ENTRY_MODULES = (
    "active_window_authority.py",
    "persisted_turn_projection.py",
    "replay_reconciliation.py",
    "settlement_recovery.py",
    "turn_window_audit.py",
)


def test_lifecycle_implementations_have_one_physical_owner() -> None:
    lifecycle_root = Path(lifecycle.__file__).resolve().parent
    entry_root = lifecycle_root.parent

    assert lifecycle.__all__ == []
    assert {
        path.stem
        for path in lifecycle_root.glob("*.py")
        if path.name != "__init__.py"
    } == set(_LIFECYCLE_MODULES)
    for retired_name in _RETIRED_ENTRY_MODULES:
        assert not (entry_root / retired_name).exists()


def test_importing_lifecycle_modules_does_not_load_an_execution_lane() -> None:
    code = """
import importlib
import json
import sys

for name in (
    'active_window',
    'persisted_turn',
    'replay',
    'settlement',
    'window_audit',
):
    importlib.import_module(f'personagraph.runtime.entry.lifecycle.{name}')

blocked_prefixes = (
    'personagraph.l2',
    'personagraph.runtime.l1',
)
print(json.dumps(sorted(
    name for name in sys.modules if name.startswith(blocked_prefixes)
)))
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(completed.stdout) == []
