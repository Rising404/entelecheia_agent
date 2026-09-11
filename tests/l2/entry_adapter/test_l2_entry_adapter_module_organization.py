"""L2 Entry 适配层的唯一归属与依赖方向围栏。"""

from __future__ import annotations

import ast
from importlib.util import resolve_name
from pathlib import Path


_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_SOURCE_ROOT = _REPOSITORY_ROOT / "src/personagraph"
_ADAPTER_ROOT = _SOURCE_ROOT / "l2/entry_adapter"
_RUNTIME_ENTRY_ROOT = _SOURCE_ROOT / "runtime/entry"


def _resolved_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    package = "personagraph.l2.entry_adapter"
    imported: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        module = node.module or ""
        imported.add(
            resolve_name(f"{'.' * node.level}{module}", package)
            if node.level
            else module
        )
    imported.update(
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    )
    return imported


def test_l2_entry_leaves_have_one_physical_owner() -> None:
    assert {
        path.name for path in _ADAPTER_ROOT.glob("*.py") if path.is_file()
    } == {
        "__init__.py",
        "application.py",
        "auxiliary_outcome.py",
        "executor.py",
        "no_public_stop.py",
        "replay.py",
        "settlement.py",
        "task_admission.py",
        "task_routing.py",
    }

    for retired_name in (
        "auxiliary_outcome_policy.py",
        "task_executor_composition.py",
        "authoritative_no_public_stop_projection.py",
    ):
        assert not (_RUNTIME_ENTRY_ROOT / retired_name).exists()


def test_runtime_entry_no_longer_advertises_l2_adapter_modules() -> None:
    package_source = (_RUNTIME_ENTRY_ROOT / "__init__.py").read_text(encoding="utf-8")

    for retired_module in (
        "auxiliary_outcome_policy",
        "task_executor_composition",
        "authoritative_no_public_stop_projection",
    ):
        assert retired_module not in package_source


def test_l2_entry_adapter_does_not_depend_on_runtime_entry_internals() -> None:
    violations: dict[str, list[str]] = {}
    for path in sorted(_ADAPTER_ROOT.glob("*.py")):
        forbidden = sorted(
            module
            for module in _resolved_imports(path)
            if module == "personagraph.runtime.entry"
            or module.startswith("personagraph.runtime.entry.")
        )
        if forbidden:
            violations[path.name] = forbidden

    assert violations == {}


def test_public_task_routing_keeps_only_lane_neutral_ownership() -> None:
    source = (_RUNTIME_ENTRY_ROOT / "routing/selection.py").read_text(
        encoding="utf-8"
    )

    for retired_symbol in (
        "EntryTaskRoutingStorePort",
        "_L2TaskRoutingStore",
        "_L2_TASK_ROUTING_STORE",
        "_KNOWN_TASK_STATUSES",
        "EntryTaskRoutingAuthorityError",
        "_select_auxiliary_production_target",
    ):
        assert retired_symbol not in source

    tree = ast.parse(source)
    top_level_modules = {
        node.module or "" for node in tree.body if isinstance(node, ast.ImportFrom)
    }
    top_level_modules.update(
        alias.name
        for node in tree.body
        if isinstance(node, ast.Import)
        for alias in node.names
    )
    assert all(
        module != "personagraph.l2" and not module.startswith("personagraph.l2.")
        for module in top_level_modules
    )


def test_task_match_persistence_has_one_owner() -> None:
    entry_source = (_RUNTIME_ENTRY_ROOT / "routing/task_admission.py").read_text(
        encoding="utf-8"
    )
    adapter_source = (_ADAPTER_ROOT / "task_admission.py").read_text(encoding="utf-8")

    assert "def _apply_entry_task_matches(" not in entry_source
    assert adapter_source.count("def apply_persisted_task_matches(") == 1


def test_l2_replay_reads_have_one_owner() -> None:
    entry_source = (
        _RUNTIME_ENTRY_ROOT / "lifecycle/replay.py"
    ).read_text(encoding="utf-8")
    adapter_source = (_ADAPTER_ROOT / "replay.py").read_text(encoding="utf-8")

    for retired_symbol in (
        "EntryReplayTaskExecutionLaneStorePort",
        "class _SessionTaskExecutionLaneStore",
        "def _require_task_execution_lane_manifest(",
    ):
        assert retired_symbol not in entry_source
    assert adapter_source.count("def require_task_execution_lane_manifest(") == 1
