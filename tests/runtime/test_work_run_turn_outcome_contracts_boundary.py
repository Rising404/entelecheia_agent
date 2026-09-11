"""冷态 WorkRun 轮次结果契约的边界覆盖。"""

from __future__ import annotations

import ast
import json
from pathlib import Path
import subprocess
import sys

import pytest

from personagraph.l2.task_execution.task_graph import controller as task_graph
from personagraph.l2.task_execution.work_run import (
    turn_controller as controller,
)
from personagraph.l2.task_execution.work_run import (
    turn_outcome_contracts as outcomes,
)


_MOVED_PUBLIC_NAMES = (
    'WorkRunTurnApplicationResult',
    "WorkRunTurnAuthorityUnavailable",
    'WorkRunTurnFailureCode',
    'WorkRunTurnOutcome',
)


def _from_import_names(path: Path, module_name: str) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module == module_name
        for alias in node.names
    }


def test_outcome_contract_owner_has_no_runtime_execution_imports() -> None:
    tree = ast.parse(Path(outcomes.__file__).read_text(encoding="utf-8"))
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imports |= {
        node.module or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }

    forbidden = {
        "attempt_controller",
        "attempt_decision",
        "model_requests",
        "node_verification",
        "session",
        "session.store",
        "work_run_tool_bridge",
        "work_run_turn_controller",
    }
    assert not (imports & forbidden)


def test_outcome_contract_cold_import_does_not_load_runtime_execution() -> None:
    code = """
import json
import sys
import personagraph.l2.task_execution.work_run.turn_outcome_contracts
blocked = {
    'personagraph.model_io.gateway',
    'personagraph.session',
    'personagraph.session.store',
    'personagraph.l2.task_execution.attempts.controller',
    'personagraph.l2.task_execution.attempts.decision',
    'personagraph.l2.task_execution.verification.decision',
    'personagraph.l2.task_execution.tool_bridge.work_run_bridge',
    'personagraph.l2.task_execution.work_run.turn_controller',
    'personagraph.runtime.attempt_controller',
    'personagraph.runtime.attempt_decision',
    'personagraph.runtime.model_calls.requests',
    'personagraph.runtime.node_verification',
    'personagraph.runtime.work_run_tool_bridge',
    'personagraph.runtime.work_run_turn_controller',
}
print(json.dumps(sorted(blocked & set(sys.modules))))
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(completed.stdout) == []


def test_controller_reexports_exact_outcome_contract_objects() -> None:
    imported = _from_import_names(
        Path(controller.__file__),
        "turn_outcome_contracts",
    )
    assert set(_MOVED_PUBLIC_NAMES) <= imported
    for name in _MOVED_PUBLIC_NAMES:
        assert getattr(controller, name) is getattr(outcomes, name)
    assert controller._failed is outcomes._failed


def test_outcome_contracts_preserve_payload_and_failure_guards() -> None:
    delivered = outcomes.WorkRunTurnApplicationResult(
        outcome="delivery_ready",
        work_run_id="work-run-1",
        delivery_id="delivery-1",
        window_revision=3,
    )
    assert delivered.delivery_id == "delivery-1"
    failed = outcomes._failed(
        "node_not_ready",
        work_run_id="work-run-1",
        current_attempt_id="attempt-1",
        window_revision=4,
    )
    assert failed.outcome == "failed_closed"
    assert failed.failure_code == "node_not_ready"
    with pytest.raises(ValueError, match="delivery_ready requires"):
        outcomes.WorkRunTurnApplicationResult(
            outcome="delivery_ready",
            work_run_id="work-run-1",
            window_revision=3,
        )
    with pytest.raises(ValueError, match="only failed_closed requires"):
        outcomes.WorkRunTurnApplicationResult(
            outcome="waiting_external",
            work_run_id="work-run-1",
            window_revision=3,
            failure_code="node_not_ready",
        )
def test_task_graph_uses_its_narrow_contract_owner() -> None:
    # 请求 DTO 有自己的语义所有者。该冷态结果所有者有意只携带结果/停止契约。
    assert _from_import_names(
        Path(task_graph.__file__),
        "personagraph.l2.task_execution.task_graph.contracts",
    ) == {
        'TaskGraphWorkRunRequest',
        'TaskGraphWorkRunResult',
        'TaskGraphWorkRunStatus',
    }
    assert not _from_import_names(
        Path(task_graph.__file__),
        "personagraph.l2.task_execution.work_run.turn_outcome_contracts",
    )
