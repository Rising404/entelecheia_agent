"""TaskGraph WorkRun 调用契约的所有权与依赖覆盖。"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

from personagraph.l2.task_execution.task_graph import (
    contracts as task_graph_work_run_contracts,
)
from personagraph.l2.task_execution.task_graph import (
    controller as task_graph_work_run_controller,
)
from personagraph.l2.auxiliary_execution.delivery import (
    composition as auxiliary_task_delivery_composition,
)


_PUBLIC_NAMES = (
    'TaskGraphWorkRunRequest',
    'TaskGraphWorkRunResult',
    'TaskGraphWorkRunStatus',
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


def test_task_graph_work_run_contracts_have_one_owner_and_controller_aliases() -> None:
    assert tuple(task_graph_work_run_contracts.__all__) == _PUBLIC_NAMES
    for name in _PUBLIC_NAMES:
        assert getattr(task_graph_work_run_controller, name) is getattr(
            task_graph_work_run_contracts,
            name,
        )

    controller_definitions = {
        node.name
        for node in _tree(task_graph_work_run_controller).body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef))
    }
    assert not {
        'TaskGraphWorkRunRequest',
        'TaskGraphWorkRunResult',
    } & controller_definitions


def test_task_graph_work_run_contract_owner_is_cold_without_controller() -> None:
    tree = _tree(task_graph_work_run_contracts)
    relative_imports = {
        (node.level, node.module)
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.level > 0
    }
    assert relative_imports == {
        (1, "profile"),
        (2, "paper_prompt_context"),
    }
    absolute_imports = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.level == 0
    }
    assert {
        "personagraph.l2.task_execution.work_run.turn_outcome_contracts",
    } <= absolute_imports
    assert not {
        "session",
        "sqlite3",
        "work_run_turn_controller",
    } & set(task_graph_work_run_contracts.__dict__)

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
import personagraph.l2.task_execution.task_graph.contracts
blocked = {
    'personagraph.l2.task_execution.task_graph.controller',
    'personagraph.runtime.work_run_turn_controller',
    'personagraph.session',
    'personagraph.session.store',
    'personagraph.session.persistence',
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


def test_execution_consumers_depend_on_narrow_task_graph_contract_owner() -> None:
    assert set(_PUBLIC_NAMES) <= _imported_names(
        task_graph_work_run_controller,
        "personagraph.l2.task_execution.task_graph.contracts",
    )
    expected_consumer_imports = {
        auxiliary_task_delivery_composition: {
            'TaskGraphWorkRunRequest',
            'TaskGraphWorkRunResult',
        },
    }
    for module, names in expected_consumer_imports.items():
        assert names <= _imported_names(
            module,
            "personagraph.l2.task_execution.task_graph.contracts",
        )
        assert not names & _imported_names(
            module,
            "personagraph.l2.task_execution.task_graph.controller",
        )
