from __future__ import annotations

import ast
import json
from pathlib import Path
import subprocess
import sys


PACKAGE_ROOT = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "personagraph"
    / "context_budget"
)


def test_context_budget_core_has_no_domain_or_provider_dependencies() -> None:
    allowed_absolute_roots = set(sys.stdlib_module_names) | {"pydantic"}
    for path in PACKAGE_ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name.split(".", 1)[0] in allowed_absolute_roots, path
            elif isinstance(node, ast.ImportFrom):
                if node.level == 1:
                    continue
                assert node.level == 0, (
                    f"relative import escapes context_budget: {path}:{node.lineno}"
                )
                root = (node.module or "").split(".", 1)[0]
                assert root in allowed_absolute_roots, path


def test_cold_import_does_not_load_runtime_provider_or_embedding_stacks() -> None:
    code = """
import json
import sys
import personagraph.context_budget
from pathlib import Path
blocked_prefixes = (
    'personagraph.api',
    'personagraph.graph',
    'personagraph.memory',
    'personagraph.model_io',
    'personagraph.persona',
    'personagraph.runtime',
    'personagraph.session',
    'personagraph.tools',
    'httpx',
    'tiktoken',
    'transformers',
    'sentence_transformers',
)
loaded = sorted(
    name for name in sys.modules
    if any(name == prefix or name.startswith(prefix + '.') for prefix in blocked_prefixes)
)
print(json.dumps({
    'loaded': loaded,
    'package_file': str(Path(personagraph.context_budget.__file__).resolve()),
}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )

    result = json.loads(completed.stdout)
    assert result["loaded"] == []
    assert Path(result["package_file"]).is_relative_to(PACKAGE_ROOT)
