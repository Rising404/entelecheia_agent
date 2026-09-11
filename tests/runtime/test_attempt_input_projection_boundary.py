"""尝试输入契约与提示投影的边界覆盖。"""

from __future__ import annotations

import ast
import json
from pathlib import Path
import subprocess
import sys

from personagraph.l2.task_execution.attempts import (
    context_authority,
    controller,
    decision,
    input_projection as projection,
)
from personagraph.l2.auxiliary_execution.work_run import (
    controller as auxiliary_work_run,
)
from personagraph.l2.task_execution.work_run import turn_controller as work_run
from personagraph.l2.task_execution.task_node.dependencies import TaskNodeDependencyInputLimits
from personagraph.l2.work_run import ToolResultStatus, ToolResult


_MOVED_PUBLIC_NAMES = (
    'AttemptDecisionContext',
    'AttemptDecisionInputLimits',
    "AttemptDecisionInputTooLarge",
    "AttemptDecisionInputUnsupported",
    'AttemptUserInput',
    'AttemptVerificationFeedback',
    'PriorToolResultProjection',
    "PriorToolResultsInputTooLarge",
    'PriorToolResultsProjection',
    "RequiredPriorToolResultsUnavailable",
    "attempt_prompt_serialized_utf8_bytes",
    "build_attempt_prompt_payload",
    "build_prior_tool_results_prompt_payload",
    "mandatory_prior_tool_result_ids",
    "prior_tool_results_serialized_utf8_bytes",
    "require_dependency_deliveries_within_limits",
    "require_prior_tool_results_within_limits",
    "select_bounded_prior_tool_results",
    "serialize_attempt_prompt_payload",
)


def _from_import_names(path: Path, *module_names: str) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module in module_names
        for alias in node.names
    }


def _result(ordinal: int) -> projection.PriorToolResultProjection:
    return projection.PriorToolResultProjection(
        tool_id="read_source",
        tool_version="1.0.0",
        result=ToolResult(
            status=ToolResultStatus.SUCCEEDED,
            tool_result_id=f"result-{ordinal}",
            tool_call_id=f"call-{ordinal}",
            attempt_id=f"attempt-{ordinal}",
            ordinal=ordinal,
            output={"ordinal": ordinal},
        ),
    )


def _limits(*, max_items: int, max_bytes: int) -> projection.AttemptDecisionInputLimits:
    return projection.AttemptDecisionInputLimits(
        profile_id="attempt-input-projection-boundary",
        max_prior_tool_result_items=max_items,
        max_prior_tool_results_serialized_utf8_bytes=max_bytes,
        dependency_delivery_limits=TaskNodeDependencyInputLimits(
            profile_id="attempt-input-projection-dependencies",
            max_items=0,
            max_serialized_utf8_bytes=1,
        ),
        max_serialized_utf8_bytes=10_000,
    )


def test_input_projection_owner_excludes_model_execution_and_store_imports() -> None:
    tree = ast.parse(Path(projection.__file__).read_text(encoding="utf-8"))
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
        "session.store",
        "task_node_model_authority",
    }
    assert not (imports & forbidden)


def test_input_projection_cold_import_does_not_load_model_or_runtime_execution() -> None:
    code = """
import json
import sys
import personagraph.l2.task_execution.attempts.input_projection
blocked = {
    'personagraph.model_io.gateway',
    'personagraph.session.store',
    'personagraph.l2.task_execution.attempts.controller',
    'personagraph.l2.task_execution.attempts.decision',
    'personagraph.l2.task_execution.tool_bridge.work_run_bridge',
    'personagraph.l2.task_execution.work_run.turn_controller',
    'personagraph.runtime.attempt_controller',
    'personagraph.runtime.attempt_decision',
    'personagraph.runtime.model_calls.requests',
    'personagraph.l2.task_execution.task_node.model_authority',
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


def test_attempt_decision_imports_exact_input_projection_objects_it_uses() -> None:
    imported = _from_import_names(Path(decision.__file__), "input_projection")
    assert set(_MOVED_PUBLIC_NAMES) <= imported
    for name in _MOVED_PUBLIC_NAMES:
        assert getattr(decision, name) is getattr(projection, name)


def test_runtime_consumers_depend_on_input_projection_not_model_port() -> None:
    expected_projection_imports = {
        context_authority: {
            'AttemptDecisionContext',
            'AttemptUserInput',
            'AttemptVerificationFeedback',
            'PriorToolResultProjection',
            "mandatory_prior_tool_result_ids",
            "select_bounded_prior_tool_results",
        },
        controller: {'AttemptDecisionContext'},
        auxiliary_work_run: {
            'AttemptDecisionContext',
            "AttemptDecisionInputTooLarge",
            'AttemptVerificationFeedback',
            'PriorToolResultProjection',
            "mandatory_prior_tool_result_ids",
            "select_bounded_prior_tool_results",
        },
        work_run: {
            'AttemptDecisionContext',
            "AttemptDecisionInputTooLarge",
            "AttemptDecisionInputUnsupported",
            'AttemptVerificationFeedback',
            'PriorToolResultProjection',
            'PriorToolResultsProjection',
            "RequiredPriorToolResultsUnavailable",
            "mandatory_prior_tool_result_ids",
            "select_bounded_prior_tool_results",
        },
    }
    for consumer, names in expected_projection_imports.items():
        assert names <= _from_import_names(
            Path(consumer.__file__),
            "attempt_input_projection",
            "input_projection",
            "attempts.input_projection",
            "personagraph.l2.task_execution.attempts.input_projection",
        )

    for consumer in (auxiliary_work_run, work_run):
        assert {'AttemptDecisionInputLimits'} <= _from_import_names(
            Path(consumer.__file__),
            "task_node_input_limits",
            "runtime.task_node_input_limits",
            "task_node.input_limits",
            "personagraph.l2.task_execution.task_node.input_limits",
        )


def test_history_projection_preserves_exact_json_measurement_and_stable_selection() -> None:
    oldest, middle, newest = (_result(1), _result(2), _result(3))
    expected = projection.PriorToolResultsProjection(
        items=(oldest, newest),
        truncated=True,
    )
    serialized = json.dumps(
        projection.build_prior_tool_results_prompt_payload(expected),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    limits = _limits(
        max_items=2,
        max_bytes=len(serialized.encode("utf-8")),
    )

    bounded = projection.select_bounded_prior_tool_results(
        (oldest, middle, newest),
        required_result_ids=(oldest.result.tool_result_id,),
        limits=limits,
    )

    assert bounded == expected
    assert projection.prior_tool_results_serialized_utf8_bytes(expected) == len(
        serialized.encode("utf-8")
    )
