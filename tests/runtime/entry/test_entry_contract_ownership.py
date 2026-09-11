"""Entry 共享契约拆分后的单一所有权围栏。"""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
import re
import subprocess
import sys


PERSONAGRAPH_ROOT = Path(__file__).resolve().parents[3] / "src" / "personagraph"
ENTRY_ROOT = PERSONAGRAPH_ROOT / "runtime" / "entry"
RUNTIME_ROOT = PERSONAGRAPH_ROOT / "runtime"


def test_retired_entry_contract_aggregate_is_absent() -> None:
    assert not (ENTRY_ROOT / "contracts.py").exists()
    assert importlib.util.find_spec("personagraph.runtime.entry.contracts") is None


def test_entry_contracts_are_owned_by_their_responsibility_packages() -> None:
    assert (ENTRY_ROOT / "ingress" / "model_contracts.py").is_file()
    assert (ENTRY_ROOT / "context" / "contracts.py").is_file()
    assert (ENTRY_ROOT / "routing" / "contracts.py").is_file()
    assert (ENTRY_ROOT / "lifecycle" / "contracts.py").is_file()
    assert (ENTRY_ROOT / "ports.py").is_file()


def test_entry_event_emitter_has_one_runtime_owner() -> None:
    definition = re.compile(r"^EntryEventEmitter\s*=", re.MULTILINE)
    owners = [
        path.relative_to(RUNTIME_ROOT).as_posix()
        for path in RUNTIME_ROOT.rglob("*.py")
        if definition.search(path.read_text(encoding="utf-8"))
    ]

    assert owners == ["turn_events.py"]


def test_entry_application_store_is_only_an_explicit_owner_composition() -> None:
    tree = ast.parse((ENTRY_ROOT / "ports.py").read_text(encoding="utf-8"))
    port = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "EntryApplicationStorePort"
    )

    assert not any(isinstance(node, ast.FunctionDef) for node in port.body)
    assert {
        base.id
        for base in port.bases
        if isinstance(base, ast.Name)
    } == {
        "EntryContextAssemblyStorePort",
        "EntryLifecycleStorePort",
        "EntryRoutingStorePort",
        "L1EntryBootstrapStorePort",
        "L1StorePort",
        "Protocol",
    }
    assert not any(
        isinstance(node, ast.ClassDef) and node.name == "EntryStorePort"
        for node in tree.body
    )


def test_importing_entry_store_composition_does_not_initialize_l2() -> None:
    source = """
import json
import sys
import personagraph.runtime.entry.ports

print(json.dumps(sorted(
    name for name in sys.modules
    if name == 'personagraph.l2' or name.startswith('personagraph.l2.')
)))
"""
    completed = subprocess.run(
        [sys.executable, "-c", source],
        check=True,
        capture_output=True,
        text=True,
    )

    assert completed.stdout.strip() == "[]"
