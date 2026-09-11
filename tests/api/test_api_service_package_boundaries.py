"""API service facade 不应在普通 L1 路径上提前加载 L2。"""

from __future__ import annotations

import json
import subprocess
import sys


def test_importing_session_service_does_not_initialize_l2() -> None:
    source = r'''
import importlib
import importlib.abc
import json
import sys


class _RejectL2(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname == "personagraph.l2" or fullname.startswith("personagraph.l2."):
            raise AssertionError(f"ordinary API import attempted to load {fullname}")
        return None


sys.meta_path.insert(0, _RejectL2())
importlib.import_module("personagraph.api.service.sessions")
print(json.dumps(sorted(
    name for name in sys.modules
    if name == "personagraph.l2" or name.startswith("personagraph.l2.")
)))
'''
    completed = subprocess.run(
        [sys.executable, "-c", source],
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(completed.stdout) == []
