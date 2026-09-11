from __future__ import annotations

import json
import subprocess
import sys


def test_configuration_paths_cold_import_stays_stdlib_only() -> None:
    code = """
import json
import sys
import personagraph.configuration.paths

blocked = (
    'personagraph.configuration.app_settings',
    'personagraph.configuration.features',
    'personagraph.model_io',
    'personagraph.runtime',
)
loaded = sorted(
    name
    for name in sys.modules
    if any(name == prefix or name.startswith(prefix + '.') for prefix in blocked)
)
print(json.dumps(loaded))
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(completed.stdout) == []
