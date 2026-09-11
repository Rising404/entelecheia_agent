"""让活跃测试套件独立于已退役的运行时编排。"""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path


TEST_ROOT = Path(__file__).parents[1]
REPO_ROOT = TEST_ROOT.parent
FORBIDDEN_IMPORTS = (
    "personagraph.graph",
    "personagraph.persona",
    "personagraph.runtime.checkpoints",
    "personagraph.runtime.event_journal",
    "personagraph.runtime.event_projection",
    "personagraph.runtime.events",
    "personagraph.runtime.ingress_adapter",
    "personagraph.runtime.supervisor",
    "personagraph.runtime.supervisor_capabilities",
    "personagraph.runtime.supervisor_detail_view",
    "personagraph.runtime.supervisor_provider_profile",
    "personagraph.runtime.supervisor_shadow",
    "personagraph.runtime.supervisor_taxonomy",
    "personagraph.quality",
)

RETIRED_MODULES = (
    "personagraph.api.service.memories",
    "personagraph.api.service.session_promotions",
    "personagraph.api.service.workspace",
    "personagraph.api.service.workspace_reminders",
    "personagraph.api.service.workspace_tasks",
    "personagraph.api.workspace_reminders",
    "personagraph.api.workspace_service",
    "personagraph.cli",
    "personagraph.cli_commands",
    "personagraph.cli_output",
    "personagraph.cli_parser",
    "personagraph.context",
    "personagraph.context.assembly",
    "personagraph.context.budget",
    "personagraph.context.blocks",
    "personagraph.context.degradation",
    "personagraph.context.projection_policy",
    "personagraph.context.projection_shadow",
    "personagraph.context.purpose_call",
    "personagraph.execution",
    "personagraph.export",
    "personagraph.graph",
    "personagraph.infrastructure",
    "personagraph.mcp_servers",
    "personagraph.memory.calibration",
    "personagraph.memory.conflict",
    "personagraph.memory.consolidation",
    "personagraph.memory.evaluation",
    "personagraph.memory.extractor",
    "personagraph.memory.promotion",
    "personagraph.memory.promotion_bridge",
    "personagraph.memory.reminders",
    "personagraph.memory.temporal",
    "personagraph.memory.validator",
    "personagraph.persona",
    "personagraph.persona.legacy_facts",
    "personagraph.observability",
    "personagraph.quality",
    "personagraph.quality.task_graph_production_eval",
    "personagraph.runtime.auxiliary_graph_controller",
    "personagraph.runtime.auxiliary_graph_frontier",
    "personagraph.runtime.auxiliary_graph_runtime_profile",
    "personagraph.runtime.auxiliary_v2_dependencies",
    "personagraph.runtime.auxiliary_v2_execution_composition",
    "personagraph.runtime.background",
    "personagraph.runtime.checkpoints",
    "personagraph.runtime.detail_view",
    "personagraph.runtime.event_journal",
    "personagraph.runtime.event_projection",
    "personagraph.runtime.events",
    "personagraph.runtime.file_retrieval_candidates",
    "personagraph.runtime.file_retrieval_v2_preparation",
    "personagraph.runtime.file_retrieval_v2_refs",
    "personagraph.runtime.history_retrieval_composition",
    "personagraph.runtime.history_memory_write",
    "personagraph.runtime.ingress_adapter",
    "personagraph.runtime.l1.file_retrieval_candidate_v2_composition",
    "personagraph.runtime.memory_consolidation_worker",
    "personagraph.runtime.operations",
    "personagraph.runtime.paper_evidence_identity_contracts",
    "personagraph.runtime.paper_tool_boundary",
    "personagraph.runtime.paper_tool_chunk_contracts",
    "personagraph.runtime.paper_tool_identity_contracts",
    "personagraph.runtime.paper_tool_output_contracts",
    "personagraph.runtime.paper_tool_wire_contracts",
    "personagraph.runtime.paper_tools",
    "personagraph.runtime.pending_review",
    "personagraph.runtime.runtime_model_call_reconciliation",
    "personagraph.runtime.runtime_model_call_reconciliation_contracts",
    "personagraph.runtime.shadow_observations",
    "personagraph.runtime.supervisor",
    "personagraph.runtime.supervisor_capabilities",
    "personagraph.runtime.supervisor_detail_view",
    "personagraph.runtime.supervisor_provider_profile",
    "personagraph.runtime.supervisor_shadow",
    "personagraph.runtime.supervisor_taxonomy",
    "personagraph.runtime.task_graph_evidence_refs",
    "personagraph.runtime.turn_attachment_file_candidates",
    "personagraph.runtime_eval_worker",
    "personagraph.session.promotion",
    "personagraph.security.development",
    "personagraph.security.workspace_policy",
    "personagraph.skills",
    "personagraph.tools.approvals",
    "personagraph.tools.defaults",
    "personagraph.tools.discovery",
    "personagraph.tools.doc_tools",
    "personagraph.tools.draft_tools",
    "personagraph.tools.execution_tools",
    "personagraph.tools.export_tools",
    "personagraph.tools.file_tools",
    "personagraph.tools.loop",
    "personagraph.tools.memory_tools",
    "personagraph.tools.quality_tools",
    "personagraph.tools.registry",
    "personagraph.tools.reminder_tools",
    "personagraph.tools.scholar_tools",
    "personagraph.tools.web_research",
    "personagraph.tools.web_tools",
    "personagraph.workflows",
)

RETIRED_ASSETS = (
    "configs/r3_runtime_details_shadow.yaml",
    "configs/r4_supervisor_debug_shadow.yaml",
    "configs/runtime_events_r2_dev.yaml",
    "data/artifacts",
    "data/evals",
    "data/policy",
    "data/skills",
    "data/web_cache",
    "data/world",
    "docs/SERIAL_GRAPH_DEMO_EVAL.md",
    "frontend/src/features/documents/DocumentListPanel.spec.js",
    "frontend/src/features/documents/DocumentListPanel.vue",
    "frontend/src/features/runtime/RuntimeDetailPanel.vue",
    "frontend/src/features/runtime/SupervisorShadowPanel.spec.js",
    "frontend/src/features/runtime/SupervisorShadowPanel.vue",
    "frontend/src/features/runtime/runtimeDetail.js",
    "frontend/src/features/runtime/runtimeDetail.spec.js",
    "frontend/src/features/runtime/supervisorShadowDetail.js",
    "frontend/src/features/runtime/supervisorShadowDetail.spec.js",
    "frontend/src/features/tasks/TaskListPanel.spec.js",
    "frontend/src/features/tasks/TaskListPanel.vue",
    "var/web_cache",
)


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            imported.add(module)
            imported.update(f"{module}.{alias.name}" for alias in node.names)
    return imported


def _module_exists(module: str) -> bool:
    try:
        spec = importlib.util.find_spec(module)
    except ModuleNotFoundError:
        # 父包已移除时，查找已退役的子模块同样会失败；这正是期望的缺失状态。
        return False
    if spec is None:
        return False
    # 本地运行留下的 ignored ``__pycache__`` 会令无源码目录被识别成 namespace
    # package；它不是可分发模块，也不应让退休守卫误报。
    if spec.loader is None and spec.submodule_search_locations:
        return any(
            path.suffix == ".py"
            for root in spec.submodule_search_locations
            for path in Path(root).rglob("*.py")
        )
    return True


def test_active_tests_do_not_import_retired_runtime_orchestration():
    violations: list[str] = []
    for path in sorted(TEST_ROOT.rglob("test_*.py")):
        for imported in _imports(path):
            if any(
                imported == forbidden or imported.startswith(f"{forbidden}.")
                for forbidden in FORBIDDEN_IMPORTS
            ):
                violations.append(f"{path.relative_to(TEST_ROOT)}: {imported}")
    assert violations == []


def test_current_turn_contracts_are_not_the_retired_flat_module() -> None:
    assert _module_exists("personagraph.runtime.turn.contracts")
    assert not (REPO_ROOT / "src/personagraph/runtime/turn.py").exists()


def test_retired_modules_are_absent() -> None:
    assert {
        module
        for module in RETIRED_MODULES
        if _module_exists(module)
    } == set()


def test_retired_runtime_assets_are_absent() -> None:
    assert {
        path
        for path in RETIRED_ASSETS
        if (REPO_ROOT / path).exists()
    } == set()
