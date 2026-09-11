"""Import and ownership locks for Auxiliary execution."""

from __future__ import annotations

import ast
import os
from pathlib import Path
import subprocess
import sys

_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_CANONICAL_ROOT = (
    _REPOSITORY_ROOT / "src" / "personagraph" / "l2" / "auxiliary_execution"
)
_RETIRED_MODULES = frozenset(
    {
        "personagraph.runtime.auxiliary_graph_architect",
        "personagraph.runtime.auxiliary_graph_driver",
        "personagraph.runtime.auxiliary_node_retrieval_tool_runtime",
        "personagraph.runtime.auxiliary_node_retrieval_composition",
        "personagraph.runtime.auxiliary_v2_architect_adapter",
        "personagraph.runtime.auxiliary_v2_application",
        "personagraph.runtime.auxiliary_v2_document_authority",
        "personagraph.runtime.auxiliary_v2_goal_successor_controller",
        "personagraph.runtime.auxiliary_v2_goal_supersede_controller",
        "personagraph.runtime.auxiliary_v2_host_primitive_controller",
        "personagraph.runtime.auxiliary_v2_model_authority",
        "personagraph.runtime.auxiliary_v2_model_binding_contracts",
        "personagraph.runtime.auxiliary_v2_planning_controller",
        "personagraph.runtime.auxiliary_v2_planning_model_provider",
        "personagraph.runtime.auxiliary_planning_profiles",
        "personagraph.runtime.auxiliary_v2_positive_planning_controller",
        "personagraph.runtime.auxiliary_v2_production_chain",
        "personagraph.runtime.auxiliary_v2_replanning_controller",
        "personagraph.runtime.auxiliary_v2_semantic_model_binding_contracts",
        "personagraph.runtime.auxiliary_v2_semantic_model_provider",
        "personagraph.runtime.auxiliary_v2_semantic_verification_controller",
        "personagraph.runtime.auxiliary_task_delivery_composition",
        "personagraph.runtime.auxiliary_v2_task_document_scope",
        "personagraph.runtime.auxiliary_v2_terminal_composition",
        "personagraph.runtime.auxiliary_v2_terminal_id_contracts",
        "personagraph.runtime.auxiliary_v2_visual_resource",
        "personagraph.runtime.auxiliary_work_run_controller",
        "personagraph.runtime.auxiliary_v2_work_run_contracts",
        "personagraph.runtime.auxiliary_v2_work_run_profile",
        "personagraph.runtime.mounted_visual_resources",
        "personagraph.runtime.mounted_document_resource_read_port",
        "personagraph.runtime.mounted_document_cognition_tools",
        "personagraph.runtime.mounted_visual_cognition_tools",
        "personagraph.runtime.task_graph_semantic_verification",
    }
)


def test_auxiliary_execution_package_import_is_cold() -> None:
    environment = dict(os.environ)
    existing_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = os.pathsep.join(
        value
        for value in (str(_REPOSITORY_ROOT / "src"), existing_pythonpath)
        if value
    )
    source = """
import sys
import personagraph.l2.auxiliary_execution
assert 'personagraph.l2.auxiliary_execution.planning.architect' not in sys.modules
assert 'personagraph.l2.auxiliary_execution.verification.controller' not in sys.modules
assert 'personagraph.session.store' not in sys.modules
assert 'personagraph.model_io.gateway' not in sys.modules
"""

    completed = subprocess.run(
        [sys.executable, "-c", source],
        cwd=_REPOSITORY_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_auxiliary_leaf_packages_import_are_cold() -> None:
    environment = dict(os.environ)
    existing_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = os.pathsep.join(
        value
        for value in (str(_REPOSITORY_ROOT / "src"), existing_pythonpath)
        if value
    )
    source = """
import sys
import personagraph.l2.auxiliary_execution.delivery
import personagraph.l2.auxiliary_execution.planning
import personagraph.l2.auxiliary_execution.terminal
import personagraph.l2.auxiliary_execution.work_run
assert 'personagraph.l2.auxiliary_execution.delivery.composition' not in sys.modules
assert 'personagraph.l2.auxiliary_execution.planning.mounted_document_authority' not in sys.modules
assert 'personagraph.l2.auxiliary_execution.planning.mounted_document_resource_read_port' not in sys.modules
assert 'personagraph.l2.task_execution.tool_bridge.mounted_visual_adapter' not in sys.modules
assert 'personagraph.l2.auxiliary_execution.planning.mounted_visual_resource' not in sys.modules
assert 'personagraph.tools.documents.mounted_document_cognition_tools' not in sys.modules
assert 'personagraph.l2.auxiliary_execution.terminal.composition' not in sys.modules
assert 'personagraph.l2.auxiliary_execution.work_run.controller' not in sys.modules
assert 'personagraph.session.store' not in sys.modules
assert 'personagraph.model_io.gateway' not in sys.modules
"""

    completed = subprocess.run(
        [sys.executable, "-c", source],
        cwd=_REPOSITORY_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_canonical_group_does_not_import_retired_runtime_leaves() -> None:
    violations: dict[str, list[str]] = {}
    for path in sorted(_CANONICAL_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imported = {
            node.module or ""
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
        }
        imported.update(
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        )
        retired = sorted(
            name
            for name in imported
            if any(
                name == retired_module
                or name.startswith(f"{retired_module}.")
                for retired_module in _RETIRED_MODULES
            )
        )
        if retired:
            violations[str(path.relative_to(_REPOSITORY_ROOT))] = retired

    assert violations == {}


def test_migrated_auxiliary_controller_owners_are_physically_retired() -> None:
    runtime_root = _REPOSITORY_ROOT / "src/personagraph/runtime"
    planning_root = _CANONICAL_ROOT / "planning"

    assert not (
        runtime_root / "auxiliary_v2_goal_successor_controller.py"
    ).exists()
    assert not (
        runtime_root / "auxiliary_v2_goal_supersede_controller.py"
    ).exists()
    assert not (
        runtime_root / "auxiliary_v2_host_primitive_controller.py"
    ).exists()
    assert not (
        runtime_root / "auxiliary_node_retrieval_tool_runtime.py"
    ).exists()
    assert not (
        runtime_root / "auxiliary_node_retrieval_composition.py"
    ).exists()
    assert not (runtime_root / "auxiliary_v2_production_chain.py").exists()
    assert not (
        runtime_root / "auxiliary_task_delivery_composition.py"
    ).exists()
    assert not (runtime_root / "auxiliary_work_run_controller.py").exists()
    assert not (runtime_root / "auxiliary_v2_application.py").exists()
    assert not (runtime_root / "auxiliary_v2_terminal_composition.py").exists()
    assert not (runtime_root / "auxiliary_v2_visual_resource.py").exists()
    assert not (runtime_root / "auxiliary_v2_document_authority.py").exists()
    assert not (runtime_root / "mounted_document_cognition_tools.py").exists()
    assert not (runtime_root / "mounted_visual_cognition_tools.py").exists()
    assert not (runtime_root / "mounted_document_resource_read_port.py").exists()
    assert not (runtime_root / "mounted_visual_resources.py").exists()
    assert (planning_root / "goal_supersede_controller.py").is_file()
    assert (planning_root / "goal_successor_controller.py").is_file()
    assert (planning_root / "host_primitive_controller.py").is_file()
    assert (planning_root / "mounted_document_resource_read_port.py").is_file()
    assert (planning_root / "mounted_document_authority.py").is_file()
    assert not (planning_root / "mounted_document_cognition_tools.py").exists()
    assert (
        _REPOSITORY_ROOT
        / "src/personagraph/tools/documents/mounted_document_cognition_tools.py"
    ).is_file()
    assert not (planning_root / "mounted_visual_cognition_tools.py").exists()
    assert (
        _REPOSITORY_ROOT
        / "src/personagraph/tools/visual/mounted_visual_tools.py"
    ).is_file()
    assert (planning_root / "mounted_visual_resource.py").is_file()
    assert not (runtime_root / "task_graph_semantic_verification.py").exists()
    assert (
        _CANONICAL_ROOT / "verification/task_graph_semantic.py"
    ).is_file()
    assert (
        _CANONICAL_ROOT / "work_run/retrieval_runtime.py"
    ).is_file()
    assert (
        _CANONICAL_ROOT / "work_run/retrieval_composition.py"
    ).is_file()
    assert (_CANONICAL_ROOT / "work_run/controller.py").is_file()
    assert (_CANONICAL_ROOT / "application.py").is_file()
    assert (_CANONICAL_ROOT / "terminal/composition.py").is_file()
    assert (_CANONICAL_ROOT / "production_chain.py").is_file()
    assert (_CANONICAL_ROOT / "delivery/composition.py").is_file()
