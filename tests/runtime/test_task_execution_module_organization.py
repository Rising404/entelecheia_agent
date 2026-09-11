from __future__ import annotations

import ast
import json
from pathlib import Path
import subprocess
import sys


_RETIRED_MODULES = frozenset(
    {
        "personagraph.runtime.attempt_active_time",
        "personagraph.runtime.attempt_context_authority",
        "personagraph.runtime.attempt_controller",
        "personagraph.runtime.attempt_decision",
        "personagraph.runtime.attempt_input_projection",
        "personagraph.runtime.attempt_tool_bridge_contracts",
        "personagraph.runtime.execution_findings",
        "personagraph.runtime.execution_findings_tool_catalog",
        "personagraph.runtime.execution_findings_tool_runtime",
        "personagraph.runtime.l2_file_retrieval_candidate_composition",
        "personagraph.runtime.mounted_document_work_run_adapter",
        "personagraph.runtime.node_verification",
        "personagraph.runtime.node_verification_active_time",
        "personagraph.runtime.node_verification_controller",
        "personagraph.runtime.paper_prompt_context",
        "personagraph.runtime.tool_bridge_contracts",
        "personagraph.runtime.tool_bridge_persistence_contracts",
        "personagraph.runtime.tool_bridge_preflight",
        "personagraph.runtime.tool_bridge_preflight_contracts",
        "personagraph.runtime.task_delivery_validation_controller_contracts",
        "personagraph.runtime.task_delivery_candidate_gate",
        "personagraph.runtime.task_delivery_validation",
        "personagraph.runtime.task_delivery_validation_controller",
        "personagraph.runtime.task_delivery_validation_id_plan",
        "personagraph.runtime.task_delivery_validation_model_contracts",
        "personagraph.runtime.task_delivery_validation_model_provider",
        "personagraph.runtime.task_graph_work_run_contracts",
        "personagraph.runtime.task_graph_work_run_controller",
        "personagraph.runtime.task_graph_work_run_id_plan",
        "personagraph.runtime.task_graph_work_run_profile",
        "personagraph.runtime.task_node_dependencies",
        "personagraph.runtime.task_node_dependency_delivery_contracts",
        "personagraph.runtime.task_node_document_tool_runtime",
        "personagraph.runtime.task_node_frontier",
        "personagraph.runtime.task_node_input_limits",
        "personagraph.runtime.task_node_model_authority_contracts",
        "personagraph.runtime.task_node_model_authority",
        "personagraph.runtime.task_node_model_binding_contracts",
        "personagraph.runtime.task_node_retrieval_composition",
        "personagraph.runtime.task_node_retrieval_tool_runtime",
        "personagraph.runtime.task_node_source_context",
        "personagraph.runtime.task_node_tool_runtime_contracts",
        "personagraph.runtime.task_node_tool_runtime_policy",
        "personagraph.runtime.work_run_execution_findings",
        "personagraph.runtime.work_run_model_profile",
        "personagraph.runtime.work_run_model_providers",
        "personagraph.runtime.work_run_stable_ids",
        "personagraph.runtime.work_run_tool_bridge",
        "personagraph.runtime.work_run_turn_controller",
        "personagraph.runtime.work_run_turn_outcome_contracts",
        "personagraph.runtime.work_run_turn_request_contracts",
    }
)


def _resolved_import(module_name: str, node: ast.ImportFrom) -> str:
    if node.level == 0:
        return node.module or ""
    package_parts = module_name.rsplit(".", 1)[0].split(".")
    keep = len(package_parts) - (node.level - 1)
    imported_parts = (node.module or "").split(".") if node.module else []
    return ".".join([*package_parts[:keep], *imported_parts])


def _imports_for(path: Path, module_name: str) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {
        _resolved_import(module_name, node)
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }


def test_task_execution_packages_and_pure_contracts_remain_cold() -> None:
    code = """
import importlib
import json
import sys

package_names = (
    'personagraph.l2.task_execution',
    'personagraph.l2.task_execution.attempts',
    'personagraph.l2.task_execution.delivery',
    'personagraph.l2.task_execution.task_graph',
    'personagraph.l2.task_execution.task_node',
    'personagraph.l2.task_execution.verification',
    'personagraph.l2.task_execution.tool_bridge',
    'personagraph.l2.task_execution.work_run',
)
for name in package_names:
    importlib.import_module(name)
package_leaves = sorted(
    name
    for name in sys.modules
    if any(name.startswith(package + '.') for package in package_names[1:])
)

for name in (
    'personagraph.l2.task_execution.attempts.active_time',
    'personagraph.l2.task_execution.verification.active_time',
    'personagraph.l2.task_execution.tool_bridge.persistence_contracts',
    'personagraph.l2.task_execution.work_run.model_profile',
    'personagraph.l2.task_execution.work_run.turn_outcome_contracts',
):
    importlib.import_module(name)

blocked_prefixes = {
    'personagraph.model_io.gateway',
    'personagraph.session',
    'personagraph.tools',
    'personagraph.l2.work_run',
    'personagraph.l2.task_execution.attempts.context_authority',
    'personagraph.l2.task_execution.attempts.controller',
    'personagraph.l2.task_execution.attempts.decision',
    'personagraph.l2.task_execution.attempts.input_projection',
    'personagraph.l2.task_execution.verification.controller',
    'personagraph.l2.task_execution.verification.decision',
    'personagraph.l2.task_execution.tool_bridge.preflight',
    'personagraph.l2.task_execution.tool_bridge.work_run_bridge',
    'personagraph.l2.task_execution.work_run.execution_findings',
    'personagraph.l2.task_execution.work_run.model_providers',
    'personagraph.l2.task_execution.work_run.turn_controller',
}
blocked = sorted(
    name
    for name in sys.modules
    if any(name == prefix or name.startswith(prefix + '.') for prefix in blocked_prefixes)
)
print(json.dumps({'package_leaves': package_leaves, 'blocked': blocked}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )

    observed = json.loads(completed.stdout)
    assert observed == {"package_leaves": [], "blocked": []}


def test_canonical_modules_do_not_import_legacy_task_execution_paths() -> None:
    package_dir = (
        Path(__file__).resolve().parents[2]
        / "src/personagraph/l2/task_execution"
    )
    forbidden = _RETIRED_MODULES

    for path in package_dir.rglob("*.py"):
        relative = path.relative_to(package_dir).with_suffix("")
        module_name = ".".join(
            ("personagraph", "l2", "task_execution", *relative.parts)
        )
        assert not (_imports_for(path, module_name) & forbidden)


def test_migrated_task_execution_owners_are_physically_retired() -> None:
    package_dir = Path(__file__).resolve().parents[2] / "src/personagraph"
    runtime_dir = package_dir / "runtime"
    task_execution_dir = package_dir / "l2/task_execution"

    assert not (runtime_dir / "l2_file_retrieval_candidate_composition.py").exists()
    assert not (runtime_dir / "paper_prompt_context.py").exists()
    assert not (runtime_dir / "execution_findings_tool_catalog.py").exists()
    assert not (runtime_dir / "task_node_model_authority.py").exists()
    assert not (runtime_dir / "task_node_retrieval_tool_runtime.py").exists()
    assert not (runtime_dir / "task_node_retrieval_composition.py").exists()
    assert not (runtime_dir / "task_node_document_tool_runtime.py").exists()
    assert not (runtime_dir / "mounted_document_work_run_adapter.py").exists()
    assert not (runtime_dir / "task_graph_work_run_controller.py").exists()
    for retired_name in (
        "task_delivery_candidate_gate.py",
        "task_delivery_validation.py",
        "task_delivery_validation_controller.py",
        "task_delivery_validation_model_provider.py",
    ):
        assert not (runtime_dir / retired_name).exists()
    assert not (task_execution_dir / "tool_bridge/file_retrieval_candidates.py").exists()
    assert (
        task_execution_dir / "tool_bridge/execution_findings_catalog.py"
    ).is_file()
    assert (task_execution_dir / "paper_prompt_context.py").is_file()
    assert (task_execution_dir / "task_node/model_authority.py").is_file()
    assert not (task_execution_dir / "task_node/retrieval_runtime.py").exists()
    assert not (task_execution_dir / "task_node/retrieval_composition.py").exists()
    assert (task_execution_dir / "task_node/document_tool_runtime.py").is_file()
    assert (task_execution_dir / "task_graph/controller.py").is_file()
    assert (
        task_execution_dir / "tool_bridge/mounted_document_adapter.py"
    ).is_file()
    assert (
        task_execution_dir / "tool_bridge/workspace_readonly_adapter.py"
    ).is_file()
    for canonical_name in (
        "candidate_gate.py",
        "model_contracts.py",
        "model_provider.py",
        "validation.py",
    ):
        assert (task_execution_dir / "delivery" / canonical_name).is_file()
    for retired_name in (
        "controller.py",
        "controller_contracts.py",
        "id_plan.py",
    ):
        assert not (task_execution_dir / "delivery" / retired_name).exists()


def test_shared_execution_findings_contract_has_persistent_content_ownership() -> None:
    source_dir = Path(__file__).resolve().parents[2] / "src/personagraph"
    package_dir = source_dir / "l2/task_execution"
    assert not (package_dir / "execution_findings.py").exists()
    assert not (source_dir / "runtime/execution_findings.py").exists()
    assert not (source_dir / "runtime/execution_findings_tool_runtime.py").exists()
    assert (source_dir / "persistent_turn_content/findings.py").is_file()
    assert (source_dir / "tools/findings/dispatcher.py").is_file()

    protocol_contract_users = {
        package_dir / "attempts/input_projection.py": (
            "personagraph.l2.task_execution.attempts.input_projection"
        ),
        package_dir / "work_run/execution_findings.py": (
            "personagraph.l2.task_execution.work_run.execution_findings"
        ),
    }
    for path, module_name in protocol_contract_users.items():
        assert "personagraph.persistent_turn_content.findings" in _imports_for(
            path, module_name
        )

    tool_id_users = {
        package_dir / "attempts/decision.py": (
            "personagraph.l2.task_execution.attempts.decision"
        ),
        package_dir / "attempts/input_projection.py": (
            "personagraph.l2.task_execution.attempts.input_projection"
        ),
        package_dir / "verification/decision.py": (
            "personagraph.l2.task_execution.verification.decision"
        ),
        package_dir / "tool_bridge/preflight.py": (
            "personagraph.l2.task_execution.tool_bridge.preflight"
        ),
        package_dir / "work_run/execution_findings.py": (
            "personagraph.l2.task_execution.work_run.execution_findings"
        ),
    }
    for path, module_name in tool_id_users.items():
        assert "personagraph.tools.findings.contracts" in _imports_for(
            path, module_name
        )
