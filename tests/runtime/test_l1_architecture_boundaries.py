"""L1 组合根与模块的冷导入检查。"""

from __future__ import annotations

import ast
import json
from pathlib import Path
import subprocess
import sys


_ROOT = Path(__file__).resolve().parents[2]
_L1_RUNTIME = _ROOT / "src" / "personagraph" / "runtime" / "l1"
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


def _eager_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imported: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
    return imported


def test_turn_attachment_visual_runtime_host_path_is_cold() -> None:
    code = r'''
import importlib
import importlib.abc
import json
import sys


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
    if name.startswith("personagraph.runtime.entry.application")
)))
'''
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(completed.stdout) == []


def test_mounted_document_source_authority_is_cold() -> None:
    imports = _eager_imports(_MOUNTED_DOCUMENT_SOURCE_AUTHORITY)
    assert not {
        module
        for module in imports
        if module.startswith("personagraph.runtime")
    }

    code = r'''
import importlib
import importlib.abc
import json
import sys


importlib.import_module(
    "personagraph.tools.documents.mounted_document_source_authority"
)
print(json.dumps(sorted(
    name
    for name in sys.modules
    if name.startswith("personagraph.runtime.entry.application")
)))
'''
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(completed.stdout) == []


def test_current_l1_session_store_and_chat_api_cold_import() -> None:
    code = r'''
import importlib
import importlib.abc
import json
import sys


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
    "imported": len(modules)
}))
'''
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(completed.stdout) == {"imported": 25}


def test_retired_l1_attachment_adapter_is_absent() -> None:
    assert not (
        _ROOT
        / "src/personagraph/runtime/l1/attachment_tool_runtime.py"
    ).exists()
