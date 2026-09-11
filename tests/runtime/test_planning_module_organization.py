from __future__ import annotations

import ast
import json
from pathlib import Path
import subprocess
import sys


_RETIRED_SEMANTIC_COMPILATION_MODULES = (
    "personagraph.runtime.semantic_compilation_candidate_contracts",
    "personagraph.runtime.semantic_compilation",
)

_RETIRED_MODULES = frozenset(
    {
        *_RETIRED_SEMANTIC_COMPILATION_MODULES,
        "personagraph.runtime.planning_context_invocation_contracts",
        "personagraph.runtime.planning_context_primitives",
        "personagraph.runtime.task_graph_context_builder",
        "personagraph.runtime.task_graph_context",
        "personagraph.runtime.task_graph_queries",
        "personagraph.runtime.task_graph_retrieval_projection",
        "personagraph.runtime.task_graph_retrieval_request",
        "personagraph.runtime.planning_resource_perception",
        "personagraph.runtime.mounted_file_retrieval_authority",
        "personagraph.runtime.file_retrieval_authority",
        "personagraph.l2.planning.mounted_file_retrieval_authority",
    }
)


def test_planning_package_and_contract_modules_remain_cold() -> None:
    code = """
import importlib
import json
import sys

importlib.import_module('personagraph.l2.planning')
package_loaded = sorted(
    name
    for name in sys.modules
    if name.startswith('personagraph.l2.planning.')
)
importlib.import_module('personagraph.l2.planning.semantic_compilation')
semantic_compilation_package_loaded = sorted(
    name
    for name in sys.modules
    if name.startswith('personagraph.l2.planning.semantic_compilation.')
)
importlib.import_module('personagraph.l2.planning.invocation_contracts')
importlib.import_module('personagraph.l2.planning.semantic_compilation.contracts')
importlib.import_module('personagraph.l2.planning.semantic_compilation.validation')
blocked_prefixes = {
    'personagraph.model_io.gateway',
    'personagraph.retrieval',
    'personagraph.session',
    'personagraph.workspace',
    'personagraph.l2.planning.resource_perception',
}
blocked = sorted(
    name
    for name in sys.modules
    if any(name == prefix or name.startswith(prefix + '.') for prefix in blocked_prefixes)
)
print(json.dumps({
    'package_loaded': package_loaded,
    'semantic_compilation_package_loaded': semantic_compilation_package_loaded,
    'blocked': blocked,
}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )

    observed = json.loads(completed.stdout)
    assert observed == {
        "package_loaded": [],
        "semantic_compilation_package_loaded": [],
        "blocked": [],
    }


def test_semantic_compilation_runtime_owners_are_retired() -> None:
    code = f"""
import importlib
import json

retired = {list(_RETIRED_SEMANTIC_COMPILATION_MODULES)!r}
unexpectedly_importable = []
for module_name in retired:
    try:
        importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name != module_name:
            raise
        continue
    unexpectedly_importable.append(module_name)
print(json.dumps(unexpectedly_importable))
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(completed.stdout) == []


def test_canonical_planning_modules_do_not_import_legacy_planning_paths() -> None:
    package_dir = (
        Path(__file__).resolve().parents[2]
        / "src/personagraph/l2/planning"
    )
    retired_suffixes = frozenset(
        name.removeprefix("personagraph.") for name in _RETIRED_MODULES
    )

    for path in package_dir.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported_modules = {
            node.module or ""
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
        }
        assert not {
            imported
            for imported in imported_modules
            if any(
                imported == suffix or imported.endswith("." + suffix)
                for suffix in retired_suffixes
            )
        }


def test_mounted_file_retrieval_authority_owner_is_physically_retired() -> None:
    package_dir = Path(__file__).resolve().parents[2] / "src/personagraph"

    retired_paths = (
        package_dir / "runtime/mounted_file_retrieval_authority.py",
        package_dir / "runtime/file_retrieval_authority.py",
        package_dir / "l2/planning/mounted_file_retrieval_authority.py",
    )

    assert all(not path.exists() for path in retired_paths)
