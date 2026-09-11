from __future__ import annotations

import ast
from pathlib import Path

from personagraph.session.persistence.l2.auxiliary_graph import auxiliary_node_execution_bindings
from personagraph.session.persistence.l2.auxiliary_graph import auxiliary_continuation
from personagraph.session.persistence.l2.work_run import work_execution


def _relative_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.level > 0
        and node.module is not None
    }


def _imported_names(path: Path, module: str) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.level == 1
        and node.module == module
        for alias in node.names
    }


def test_verification_consumers_import_the_narrow_binding_owner() -> None:
    binding_names = {
        "_auxiliary_record_from_row",
        "_load_auxiliary_request_row",
        "_revalidate_auxiliary_request_binding",
    }
    assert binding_names <= _imported_names(
        Path(work_execution.__file__),
        "auxiliary_node_execution_bindings",
    )
    assert {
        "_auxiliary_record_from_row",
        "_load_auxiliary_request_row",
        "_revalidate_auxiliary_request_binding",
    } <= _imported_names(
        Path(auxiliary_continuation.__file__),
        "auxiliary_node_execution_bindings",
    )
    assert "auxiliary_graphs" not in _relative_modules(
        Path(auxiliary_continuation.__file__)
    )
    assert not hasattr(
        auxiliary_node_execution_bindings,
        "_load_auxiliary_v2_request_row",
    )
    assert not hasattr(
        auxiliary_node_execution_bindings,
        "_revalidate_auxiliary_v2_request_binding",
    )
