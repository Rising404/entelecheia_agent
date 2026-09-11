"""TaskNode 工具运行时 DTO 的所有权与局部一致性覆盖。"""

from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from personagraph.l2.task_execution.task_node import (
    tool_runtime_contracts as task_node_tool_runtime_contracts,
)
from personagraph.l2.task_execution.task_graph import (
    controller as task_graph_work_run_controller,
)
from personagraph.tools.catalog import CatalogSnapshot
from personagraph.l2.work_run import TaskNodeSubject


_PUBLIC_NAMES = (
    'TaskNodeToolRuntimeBinding',
    "TaskNodeToolRuntimeFactory",
    'TaskNodeToolRuntimePlan',
    'TaskNodeToolRuntime',
)


def _tree(module: object) -> ast.Module:
    return ast.parse(Path(module.__file__).read_text(encoding="utf-8"))  # type: ignore[attr-defined]


def _imported_names(module: object, module_name: str) -> set[str]:
    return {
        alias.name
        for node in ast.walk(_tree(module))
        if isinstance(node, ast.ImportFrom) and node.module == module_name
        for alias in node.names
    }


def _subject(*, task_id: str = "task-1", revision: int = 1) -> TaskNodeSubject:
    return TaskNodeSubject(
        task_id=task_id,
        graph_revision=revision,
        node_id="node-1",
        node_revision=1,
    )


def test_task_node_runtime_contracts_have_one_owner_and_controller_aliases() -> None:
    assert tuple(task_node_tool_runtime_contracts.__all__) == _PUBLIC_NAMES
    for name in _PUBLIC_NAMES:
        assert getattr(task_graph_work_run_controller, name) is getattr(
            task_node_tool_runtime_contracts,
            name,
        )

    controller_definitions = {
        node.name
        for node in _tree(task_graph_work_run_controller).body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef))
    }
    assert not {
        'TaskNodeToolRuntime',
        'TaskNodeToolRuntimeBinding',
        'TaskNodeToolRuntimePlan',
    } & controller_definitions
    assert _imported_names(
        task_graph_work_run_controller,
        "personagraph.l2.task_execution.task_node.tool_runtime_contracts",
    ) == set(_PUBLIC_NAMES)


def test_task_node_runtime_contract_owner_stays_out_of_controller_and_scheduler() -> None:
    tree = _tree(task_node_tool_runtime_contracts)
    relative_imports = {
        (node.level, node.module)
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.level > 0
    }
    assert relative_imports == {(2, "paper_prompt_context")}
    absolute_imports = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.level == 0
    }
    assert {
        "personagraph.l2.task_execution.tool_bridge.contracts",
        "personagraph.l2.work_run",
        "personagraph.tools.catalog",
    } <= absolute_imports
    assert not {
        "session_store",
        "run_task_graph_work_runs",
        "preflight_task_node_tool_runtimes",
        'TurnEvent',
    } & set(task_node_tool_runtime_contracts.__dict__)

    repository_root = Path(__file__).resolve().parents[2]
    environment = dict(os.environ)
    existing_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = os.pathsep.join(
        value
        for value in (str(repository_root / "src"), existing_pythonpath)
        if value
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import json
import sys
import personagraph.l2.task_execution.task_node.tool_runtime_contracts
blocked = {
    'personagraph.input_processing.documents.storage.repository',
    'personagraph.l2.task_execution.task_graph.controller',
    'personagraph.runtime.entry',
    'personagraph.runtime.orchestrated_work_run',
    'personagraph.runtime.work_run_turn_controller',
}
print(json.dumps(sorted(blocked & set(sys.modules))))
""",
        ],
        cwd=repository_root,
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(completed.stdout) == []


def test_task_node_runtime_contracts_keep_local_guards_and_narrow_consumers() -> None:
    empty_catalog = CatalogSnapshot(revision=1, entries=())
    runtime = task_node_tool_runtime_contracts.TaskNodeToolRuntime(
        catalog_snapshot=empty_catalog,
    )
    subject = _subject()
    plan = task_node_tool_runtime_contracts.TaskNodeToolRuntimePlan(
        session_id="session-1",
        task_id="task-1",
        graph_revision=1,
        default_runtime=runtime,
    )
    assert plan.runtime_for(subject) is runtime
    with pytest.raises(ValueError, match="crossed TaskGraph authority"):
        plan.runtime_for(_subject(task_id="task-2"))

    bound_plan = task_node_tool_runtime_contracts.TaskNodeToolRuntimePlan(
        session_id="session-1",
        task_id="task-1",
        graph_revision=1,
        default_runtime=runtime,
        bindings=(
            task_node_tool_runtime_contracts.TaskNodeToolRuntimeBinding(
                subject=subject,
                runtime=runtime,
            ),
        ),
    )
    with pytest.raises(ValueError, match="absent from the frozen runtime plan"):
        bound_plan.runtime_for(
            TaskNodeSubject(
                task_id="task-1",
                graph_revision=1,
                node_id="node-2",
                node_revision=1,
            )
        )
    with pytest.raises(FrozenInstanceError):
        runtime.catalog_snapshot = empty_catalog  # type: ignore[misc]
