"""工具桥消费者共享的故障关闭能力检查。"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from personagraph.l2.task_execution.tool_bridge import (
    contracts as tool_bridge_contracts,
)
from personagraph.l2.task_execution.work_run import (
    turn_controller as work_run_turn_controller,
)


class _Bridge:
    def __init__(self, declaration: object) -> None:
        self._declaration = declaration

    @property
    def supports_protected_recovery(self) -> object:
        return self._declaration


class _ExplodingBridge:
    @property
    def supports_protected_recovery(self) -> bool:
        raise RuntimeError("bridge capability probe failed")


def _imported_names(module: object, module_name: str) -> set[str]:
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))  # type: ignore[attr-defined]
    return {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == module_name
        for alias in node.names
    }


@pytest.mark.parametrize(
    ("bridge", "expected"),
    (
        (None, False),
        (_Bridge(False), False),
        (_Bridge(0), False),
        (_Bridge(1), False),
        (_Bridge("true"), False),
        (_Bridge(True), True),
        (_ExplodingBridge(), False),
    ),
)
def test_protected_recovery_requires_an_explicit_true_declaration(
    bridge: object | None,
    expected: bool,
) -> None:
    assert (
        tool_bridge_contracts.bridge_supports_protected_recovery(  # type: ignore[arg-type]
            bridge
        )
        is expected
    )


def test_work_run_controller_uses_the_shared_capability_contract() -> None:
    assert tuple(tool_bridge_contracts.__all__) == (
        "AttemptToolBridge",
        "CatalogRebindableAttemptToolBridge",
        "DurableToolResultObserver",
        "bridge_supports_catalog_rebind",
        "bridge_supports_protected_recovery",
    )
    assert _imported_names(
        work_run_turn_controller,
        "tool_bridge.contracts",
    ) == {"AttemptToolBridge", "bridge_supports_protected_recovery"}
    tree = ast.parse(
        Path(work_run_turn_controller.__file__).read_text(encoding="utf-8")  # type: ignore[attr-defined]
    )
    functions = {
        node.name
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
    }
    assert "_bridge_supports_protected_recovery" not in functions
