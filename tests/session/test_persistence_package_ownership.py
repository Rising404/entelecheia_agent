"""Session SQLite 的目录归属和 L1/shared 冷导入边界。"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys


_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_SESSION_ROOT = _REPOSITORY_ROOT / "src/personagraph/session"
_PERSISTENCE_ROOT = _SESSION_ROOT / "persistence"
_SHARED_PACKAGES = ("l1", "metadata", "history", "turns", "calls")


def test_persistence_root_only_contains_shared_infrastructure() -> None:
    assert {path.stem for path in _PERSISTENCE_ROOT.glob("*.py")} == {
        "__init__",
        "deps",
        "schema",
        "current_schema",
        "document_mounts",
        "execution_findings",
    }


def test_l1_and_shared_persistence_cold_import_without_l2(tmp_path: Path) -> None:
    modules = ["personagraph.session.store"]
    for package in _SHARED_PACKAGES:
        package_path = _PERSISTENCE_ROOT / package
        assert (package_path / "__init__.py").is_file(), package
        module_prefix = f"personagraph.session.persistence.{package}"
        modules.append(module_prefix)
        modules.extend(
            f"{module_prefix}.{path.stem}"
            for path in sorted(package_path.glob("*.py"))
            if path.stem != "__init__"
        )

    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            """
import importlib
import json
import sys

sys.path.insert(0, sys.argv[1])
for module in json.loads(sys.argv[2]):
    importlib.import_module(module)
forbidden = (
    "personagraph.l2",
    "personagraph.session.l2_store",
    "personagraph.session.persistence.l2",
)
print(json.dumps(sorted(
    name for name in sys.modules
    if any(name == prefix or name.startswith(prefix + ".") for prefix in forbidden)
)))
""",
            str(_REPOSITORY_ROOT / "src"),
            json.dumps(modules),
        ],
        cwd=_REPOSITORY_ROOT,
        env={
            **os.environ,
            "PERSONAGRAPH_STATE_DIR": str(tmp_path / "state"),
            "PERSONAGRAPH_LOCAL_CONFIG_DIR": str(tmp_path / "config"),
            "PERSONAGRAPH_MODEL_PROVIDER": "mock",
        },
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == []


def test_retired_retrieval_and_turn_execution_ports_stay_absent() -> None:
    for relative in (
        "persistence/auxiliary_node_retrieval_capabilities",
        "persistence/task_node_retrieval_capabilities",
        "l2_store/retrieval",
        "persistence/turn_execution_port",
    ):
        retired = _SESSION_ROOT / relative
        assert not retired.with_suffix(".py").exists(), relative
        assert not retired.exists(), relative
