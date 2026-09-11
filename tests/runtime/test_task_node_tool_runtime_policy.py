"""纯 TaskNode 工具运行时选择与守卫的边界覆盖。"""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from personagraph.l2.task_execution.task_node import (
    tool_runtime_contracts as task_node_tool_runtime_contracts,
)
from personagraph.l2.task_execution.task_node import (
    tool_runtime_policy as task_node_tool_runtime_policy,
)
from personagraph.l2.task_execution.task_graph import (
    controller as task_graph_work_run_controller,
)
from personagraph.l2.task_execution.task_graph.contracts import (
    TaskGraphWorkRunRequest,
)
from personagraph.tools.catalog import CatalogSnapshot


_PUBLIC_NAMES = (
    "ordinary_task_node_model_authority_factory",
    "validate_task_node_tool_runtime",
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


def _request() -> TaskGraphWorkRunRequest:
    return TaskGraphWorkRunRequest(
        session_id="session-1",
        turn_id="turn-1",
        task_id="task-1",
        expected_window_revision=1,
    )


def _runtime(**values: object) -> task_node_tool_runtime_contracts.TaskNodeToolRuntime:
    return task_node_tool_runtime_contracts.TaskNodeToolRuntime(
        catalog_snapshot=CatalogSnapshot(revision=1, entries=()),
        **values,  # type: ignore[arg-type]
    )


def test_policy_has_one_owner_and_controller_aliases() -> None:
    assert tuple(task_node_tool_runtime_policy.__all__) == _PUBLIC_NAMES
    assert task_graph_work_run_controller._ordinary_task_node_model_authority_factory is (
        task_node_tool_runtime_policy.ordinary_task_node_model_authority_factory
    )
    assert task_graph_work_run_controller._validate_node_tool_runtime is (
        task_node_tool_runtime_policy.validate_task_node_tool_runtime
    )
    assert _imported_names(
        task_graph_work_run_controller,
        "personagraph.l2.task_execution.task_node.tool_runtime_policy",
    ) == set(_PUBLIC_NAMES)


def test_generic_model_authority_factory_is_not_paper_special_cased() -> None:
    def factory(_binding, **_values):
        return object()

    assert task_node_tool_runtime_policy.ordinary_task_node_model_authority_factory(
        _runtime(),
        factory=factory,
    ) is factory
    assert task_node_tool_runtime_policy.ordinary_task_node_model_authority_factory(
        _runtime(paper_resources=object()),
        factory=factory,
    ) is factory


def test_runtime_policy_rejects_unbound_catalog_and_crossed_paper_scope() -> None:
    request = _request()
    exposed_runtime = task_node_tool_runtime_contracts.TaskNodeToolRuntime(
        catalog_snapshot=SimpleNamespace(exposed=lambda: True),  # type: ignore[arg-type]
    )
    with pytest.raises(ValueError, match="exposed node Tool catalog"):
        task_node_tool_runtime_policy.validate_task_node_tool_runtime(
            exposed_runtime,
            request=request,
        )
    with pytest.raises(ValueError, match="node paper resources crossed"):
        task_node_tool_runtime_policy.validate_task_node_tool_runtime(
            _runtime(paper_resources=object()),
            request=request,
        )


def test_policy_owner_has_only_contract_dependencies() -> None:
    relative_imports = {
        (node.level, node.module)
        for node in ast.walk(_tree(task_node_tool_runtime_policy))
        if isinstance(node, ast.ImportFrom) and node.level > 0
    }
    assert relative_imports == {
        (1, "model_authority_contracts"),
        (1, "tool_runtime_contracts"),
        (2, "task_graph.contracts"),
        (2, "paper_prompt_context"),
    }
    absolute_imports = {
        node.module
        for node in ast.walk(_tree(task_node_tool_runtime_policy))
        if isinstance(node, ast.ImportFrom) and node.level == 0
    }
    assert {
        "personagraph.l2.work_run",
    } <= absolute_imports
