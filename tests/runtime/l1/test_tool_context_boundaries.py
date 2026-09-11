"""历史工具、数据合同、存储与 L1 编排维持各自目录职责。"""

import ast
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3] / "src" / "personagraph"


@pytest.mark.parametrize(("relative", "forbidden"), [
    ("persistent_turn_content/tool_results.py", {"sqlite3", "runtime", "session", "tools"}),
    ("tools/tool_history", {"sqlite3", "runtime", "session"}),
    ("session/persistence/l1/tool_history.py", {"runtime", "tools"}),
    ("runtime/l1/tool_context/projection.py", {"sqlite3", "session", "model_io"}),
])
def test_history_responsibility_import_boundaries(relative, forbidden):
    target = ROOT / relative
    paths = tuple(target.glob("*.py")) if target.is_dir() else (target,)
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        modules = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                modules.append(node.module or "")
        assert not {part for module in modules for part in module.split(".")} & forbidden, path
