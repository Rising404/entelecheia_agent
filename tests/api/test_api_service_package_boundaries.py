"""API service facade 的冷导入检查。"""

from __future__ import annotations

import json
import subprocess
import sys


def test_importing_session_service_is_importable() -> None:
    source = r'''
import importlib
import importlib.abc
import json
import sys


importlib.import_module("personagraph.api.service.sessions")
print(json.dumps(sorted(
    name for name in sys.modules
    if name == "torch"
)))
'''
    completed = subprocess.run(
        [sys.executable, "-c", source],
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(completed.stdout) == []
