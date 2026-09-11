from __future__ import annotations

import subprocess
import sys


def _import_in_fresh_process(source: str) -> None:
    completed = subprocess.run(
        [sys.executable, "-c", source],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_model_gateway_can_be_imported_before_runtime_entry() -> None:
    _import_in_fresh_process(
        "import personagraph.model_io.gateway; import personagraph.runtime.entry"
    )


def test_runtime_public_entry_remains_available_lazily() -> None:
    _import_in_fresh_process(
        "from personagraph.runtime import EntryTurnResult, run_entry_turn; "
        "assert callable(run_entry_turn); assert EntryTurnResult.__name__ == 'EntryTurnResult'"
    )
