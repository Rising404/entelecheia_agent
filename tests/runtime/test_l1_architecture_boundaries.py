"""确保 L1 独立于重量级 L2 领域的架构锁。"""

from __future__ import annotations

import ast
import json
from pathlib import Path
import subprocess
import sys

from personagraph.runtime.l1.ports import L1StorePort


_ROOT = Path(__file__).resolve().parents[2]
_L1_RUNTIME = _ROOT / "src" / "personagraph" / "runtime" / "l1"
_L2_ROOT = _ROOT / "src" / "personagraph" / "l2"
_L1_PERSISTENCE = (
    _ROOT
    / "src"
    / "personagraph"
    / "session"
    / "persistence"
    / "l1"
    / "turn_runs.py"
)
_MOUNTED_DOCUMENT_SOURCE_AUTHORITY = (
    _ROOT
    / "src"
    / "personagraph"
    / "tools"
    / "documents"
    / "mounted_document_source_authority.py"
)
_L1_TOOL_RUNTIME = _L1_RUNTIME / "tool_runtime.py"
_L1_OUTPUT_PROTOCOL = (
    _ROOT / "src" / "personagraph" / "output_protocol" / "l1.py"
)
_WORKSPACE_READ_SOURCE = (
    _ROOT
    / "src"
    / "personagraph"
    / "tools"
    / "workspace"
    / "session_read_source.py"
)
_FORBIDDEN_IMPORT_FRAGMENTS = (
    "auxiliary_graph",
    "task_graph",
    "work_run",
    "work_execution",
)


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
    return imported


def _eager_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imported: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
    return imported


def _imported_symbols(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }


def test_l1_runtime_and_persistence_do_not_import_l2_execution_modules() -> None:
    paths = (*sorted(_L1_RUNTIME.glob("*.py")), _L1_PERSISTENCE)
    violations = {
        str(path.relative_to(_ROOT)): sorted(
            module
            for module in _imports(path)
            if any(fragment in module for fragment in _FORBIDDEN_IMPORT_FRAGMENTS)
        )
        for path in paths
    }
    assert {path: imports for path, imports in violations.items() if imports} == {}


def test_l1_does_not_import_output_window_contracts() -> None:
    paths = (
        *sorted(_L1_RUNTIME.glob("*.py")),
        _L1_PERSISTENCE,
        _L1_OUTPUT_PROTOCOL,
    )
    forbidden = {"OutputWindow", "SubmitOutputWindowAction"}
    violations = {
        str(path.relative_to(_ROOT)): sorted(
            _imported_symbols(path) & forbidden
        )
        for path in paths
    }

    assert {path: names for path, names in violations.items() if names} == {}


def test_l1_controller_cold_import_does_not_load_output_window_modules() -> None:
    code = r'''
import importlib
import json
import sys

importlib.import_module("personagraph.runtime.l1.controller")
forbidden = {
    "personagraph.output_protocol.output_window",
    "personagraph.persistent_turn_content.output_window",
}
print(json.dumps(sorted(forbidden & set(sys.modules))))
'''
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(completed.stdout) == []


def test_l2_sources_do_not_import_l1_private_implementations() -> None:
    violations = {
        str(path.relative_to(_ROOT)): sorted(
            module
            for module in _imports(path)
            if module == "runtime.l1"
            or module.startswith("runtime.l1.")
            or module == "personagraph.runtime.l1"
            or module.startswith("personagraph.runtime.l1.")
        )
        for path in sorted(_L2_ROOT.rglob("*.py"))
    }

    assert {path: imports for path, imports in violations.items() if imports} == {}


def test_turn_attachment_visual_runtime_host_path_is_cold() -> None:
    code = r'''
import importlib
import importlib.abc
import json
import sys


class _RejectL2(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname == "personagraph.l2" or fullname.startswith("personagraph.l2."):
            raise AssertionError(f"L1 visual runtime attempted to import {fullname}")
        return None


sys.meta_path.insert(0, _RejectL2())
for module in (
    "personagraph.tools.visual.mounted_visual_source_authority",
    "personagraph.workspace.documents.admission.turn_inputs",
    "personagraph.tools.visual.file_visual_source_authority",
    "personagraph.tools.visual.file_visual_adapter",
):
    importlib.import_module(module)
print(json.dumps(sorted(
    name
    for name in sys.modules
    if name == "personagraph.l2" or name.startswith("personagraph.l2.")
)))
'''
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(completed.stdout) == []


def test_l1_controller_store_port_cannot_mutate_l2_domain_objects() -> None:
    assert {
        name
        for name, value in vars(L1StorePort).items()
        if callable(value) and not name.startswith("_")
    } == {
        "apply_execution_findings_mutation",
        "begin_l1_protected_tool_dispatch",
        "close_l1_attempt",
        "close_execution_findings_ledger",
        "commit_l1_attempt_decision",
        "create_execution_findings_ledger",
        "fail_l1_turn_run",
        "get_l1_attempt_state_guard",
        "get_l1_turn_execution",
        "get_execution_findings_ledger_for_owner",
        "initialize_l1_turn_run",
        "reject_l1_final_reply_candidate",
        "reserve_l1_tool_call",
        "settle_l1_protected_tool_dispatch",
        "settle_l1_tool_call",
        "start_l1_attempt",
    }


def test_mounted_document_source_authority_is_cold() -> None:
    imports = _eager_imports(_MOUNTED_DOCUMENT_SOURCE_AUTHORITY)
    assert not {
        module
        for module in imports
        if module == "personagraph.l2" or module.startswith("personagraph.l2.")
    }

    code = r'''
import importlib
import importlib.abc
import json
import sys


class _RejectL2(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname == "personagraph.l2" or fullname.startswith("personagraph.l2."):
            raise AssertionError(f"mounted Document source authority imported {fullname}")
        return None


sys.meta_path.insert(0, _RejectL2())
importlib.import_module(
    "personagraph.tools.documents.mounted_document_source_authority"
)
print(json.dumps(sorted(
    name
    for name in sys.modules
    if name == "personagraph.l2" or name.startswith("personagraph.l2.")
)))
'''
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(completed.stdout) == []


def test_current_l1_session_store_and_chat_api_cold_import_without_l2() -> None:
    code = r'''
import importlib
import importlib.abc
import json
import sys


class _RejectL2(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname == "personagraph.l2" or fullname.startswith("personagraph.l2."):
            raise AssertionError(f"L1 attempted to import {fullname}")
        return None


sys.meta_path.insert(0, _RejectL2())
modules = (
    "personagraph.output_protocol",
    "personagraph.persistent_turn_content",
    "personagraph.runtime.l1",
    "personagraph.runtime.l1.attachment_contracts",
    "personagraph.runtime.l1.controller",
    "personagraph.runtime.l1.corpus_contracts",
    "personagraph.runtime.l1.corpus_manifest",
    "personagraph.runtime.l1.execution_config",
    "personagraph.runtime.l1.model",
    "personagraph.runtime.l1.model_authority",
    "personagraph.runtime.l1.plan_revision",
    "personagraph.runtime.l1.ports",
    "personagraph.runtime.l1.protected_tool_dispatch",
    "personagraph.runtime.l1.recovery",
    "personagraph.runtime.l1.recovery_worker",
    "personagraph.runtime.l1.history_retrieval_composition",
    "personagraph.runtime.l1.identity",
    "personagraph.runtime.l1.semantic_contracts",
    "personagraph.runtime.l1.semantic_verification",
    "personagraph.runtime.l1.tool_catalog_snapshot",
    "personagraph.runtime.l1.tool_runtime",
    "personagraph.runtime.l1.verification",
    "personagraph.tools.workspace.session_read_source",
    "personagraph.session.store",
    "personagraph.api.service.sessions",
)
for module in modules:
    importlib.import_module(module)
print(json.dumps({
    "loaded_l2": sorted(
        name
        for name in sys.modules
        if name == "personagraph.l2" or name.startswith("personagraph.l2.")
    )
}))
'''
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(completed.stdout) == {"loaded_l2": []}


def test_retired_l1_attachment_adapter_is_absent() -> None:
    assert not (
        _ROOT
        / "src/personagraph/runtime/l1/attachment_tool_runtime.py"
    ).exists()


def test_workspace_host_composition_has_no_l2_work_run_bridge() -> None:
    workspace_imports = _imports(_WORKSPACE_READ_SOURCE)
    assert not {
        module
        for module in workspace_imports
        if module == "personagraph.l2" or module.startswith("personagraph.l2.")
    }

    workspace_tree = ast.parse(
        _WORKSPACE_READ_SOURCE.read_text(encoding="utf-8"),
        filename=str(_WORKSPACE_READ_SOURCE),
    )
    runtime_class = next(
        node
        for node in workspace_tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == 'SessionWorkspaceReadonlyRuntime'
    )
    runtime_fields = {
        node.target.id
        for node in runtime_class.body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
    }
    assert "tool_bridge" not in runtime_fields
    builder = next(
        node
        for node in workspace_tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "build_session_workspace_readonly_runtime"
    )
    assert "include_work_run_bridge" not in {
        argument.arg
        for argument in (*builder.args.args, *builder.args.kwonlyargs)
    }

    l1_tree = ast.parse(
        _L1_TOOL_RUNTIME.read_text(encoding="utf-8"),
        filename=str(_L1_TOOL_RUNTIME),
    )
    calls = [
        node
        for node in ast.walk(l1_tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "build_session_workspace_readonly_runtime"
    ]
    assert len(calls) == 1
    assert "include_work_run_bridge" not in {
        keyword.arg for keyword in calls[0].keywords
    }
