"""附件职责硬切后的所有权与冷导入守卫。"""

from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
import subprocess
import sys


PERSONAGRAPH_ROOT = (
    Path(__file__).resolve().parents[3] / "src" / "personagraph"
)


def _absolute_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            found.add(node.module)
    return found


def test_retired_input_processing_attachment_package_is_absent() -> None:
    retired = PERSONAGRAPH_ROOT / "input_processing" / "attachments"

    assert not retired.exists()
    assert importlib.util.find_spec("personagraph.input_processing.attachments") is None


def test_attachment_owners_are_explicit() -> None:
    assert (PERSONAGRAPH_ROOT / "session" / "attachments" / "contracts.py").is_file()
    assert (PERSONAGRAPH_ROOT / "session" / "attachments" / "application.py").is_file()
    assert (PERSONAGRAPH_ROOT / "workspace" / "files" / "attachments.py").is_file()
    assert (PERSONAGRAPH_ROOT / "workspace" / "files" / "turn_inputs.py").is_file()
    assert (
        PERSONAGRAPH_ROOT / "runtime" / "entry" / "context" / "attachments.py"
    ).is_file()
    retired_entry_projection = (
        PERSONAGRAPH_ROOT / "runtime" / "entry" / "attachments.py"
    )
    assert not retired_entry_projection.exists()
    assert importlib.util.find_spec("personagraph.runtime.entry.attachments") is None
    assert (
        PERSONAGRAPH_ROOT / "runtime" / "l1" / "turn_file_sources.py"
    ).is_file()
    retired_entry_source = (
        PERSONAGRAPH_ROOT / "runtime" / "entry" / "turn_file_sources.py"
    )
    assert not retired_entry_source.exists()
    assert importlib.util.find_spec(
        "personagraph.runtime.entry.turn_file_sources"
    ) is None


def test_session_attachment_contracts_do_not_depend_on_runtime_or_retrieval() -> None:
    imports = _absolute_imports(
        PERSONAGRAPH_ROOT / "session" / "attachments" / "contracts.py"
    )

    assert not any(
        name.startswith(("personagraph.runtime", "personagraph.retrieval"))
        for name in imports
    )


def test_session_attachment_import_does_not_eager_load_runtime_adapters() -> None:
    code = """
import importlib
import json
import sys

importlib.import_module('personagraph.session.attachments')
loaded = sorted(
    name
    for name in sys.modules
    if name in {
        'personagraph.runtime.entry.context.attachments',
        'personagraph.runtime.l1.turn_file_sources',
    }
)
print(json.dumps(loaded))
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(completed.stdout) == []
