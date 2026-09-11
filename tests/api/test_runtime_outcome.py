from __future__ import annotations

import json
import subprocess
import sys

from personagraph.api.runtime_outcome import build_error_outcome


def test_error_outcome_failed_retryable():
    outcome = build_error_outcome(
        code="MODEL_CALL_FAILED",
        domain="model",
        message="timeout",
        retryable=True,
        details={"exception_type": "TimeoutException"},
    ).to_dict()

    assert outcome["ok"] is False
    assert outcome["status"] == "failed"
    assert outcome["error"]["code"] == "MODEL_CALL_FAILED"
    assert outcome["error"]["retryable"] is True
    assert outcome["retry"]["action"] == "retry_turn"


def test_outcome_module_does_not_load_legacy_turn_or_graph_on_import() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import json, sys; import personagraph.api.runtime_outcome; "
            "blocked = {'personagraph.runtime.turn', 'personagraph.graph', 'langgraph'}; "
            "print(json.dumps(sorted(blocked & set(sys.modules))))",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == []
