"""Regression coverage for the Runtime-owned ModelCall package boundary."""

from __future__ import annotations

import ast
from dataclasses import MISSING, fields
import json
from pathlib import Path
import subprocess
import sys
from typing import get_type_hints

import personagraph.runtime.model_calls as model_calls


_RETIRED_RUNTIME_MODULES = (
    "model_call_authority.py",
    "model_call_ledger_contracts.py",
    "model_call_policy.py",
    "model_requests.py",
    "runtime_model_call_authority.py",
)

_PUBLIC_MODEL_CALL_SYMBOLS = (
    "DurableLogicalModelCallAuthority",
    "MAX_MODEL_ATTEMPTS",
    "ModelRequestResult",
    "RuntimeLogicalModelCallAuthority",
    "RuntimeModelLedgerStore",
    "backoff_delay_s",
    "request_model_with_retry",
)

_SESSION_INDEPENDENT_MODEL_CALL_MODULES = (
    "attempt_preparation.py",
    "authority.py",
    "contracts.py",
    "observability.py",
    "output_repair.py",
    "policy.py",
    "quota.py",
    "recovery.py",
    "request_policy.py",
    "requests.py",
)


def _model_calls_directory() -> Path:
    return Path(model_calls.__file__).resolve().parent


def _imported_modules(source_path: Path) -> tuple[str, ...]:
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            if node.level == 0:
                imported.append(node.module)
                continue
            package_parts = model_calls.__package__.split(".")
            prefix = package_parts[: len(package_parts) - node.level + 1]
            imported.append(".".join((*prefix, node.module)))
    return tuple(imported)


def test_retired_model_call_modules_are_absent_from_runtime_root() -> None:
    runtime_directory = _model_calls_directory().parent

    remaining = [
        module_name
        for module_name in _RETIRED_RUNTIME_MODULES
        if (runtime_directory / module_name).exists()
    ]

    assert remaining == []


def test_runtime_model_call_authority_requires_an_explicit_ledger_store() -> None:
    store_field = next(
        field
        for field in fields(model_calls.RuntimeLogicalModelCallAuthority)
        if field.name == "store"
    )

    assert store_field.default is MISSING
    assert store_field.default_factory is MISSING


def test_model_call_core_does_not_depend_on_session_implementation() -> None:
    violations = {
        source_path.name: tuple(
            module_name
            for module_name in _imported_modules(source_path)
            if module_name == "personagraph.session"
            or module_name.startswith("personagraph.session.")
        )
        for source_path in (
            _model_calls_directory() / name
            for name in _SESSION_INDEPENDENT_MODEL_CALL_MODULES
        )
    }

    assert {name: imports for name, imports in violations.items() if imports} == {}


def test_attempt_preparation_does_not_own_executor_side_effect_order() -> None:
    imports = set(
        _imported_modules(_model_calls_directory() / "attempt_preparation.py")
    )

    assert imports.isdisjoint(
        {
            "personagraph.model_io.api_quota_controller",
            "personagraph.runtime.model_calls.requests",
            "personagraph.runtime.turn_deadline",
            "personagraph.runtime.turn_events",
        }
    )


def test_session_summary_adapter_does_not_reach_session_persistence() -> None:
    imports = _imported_modules(_model_calls_directory() / "session_summary.py")

    assert "personagraph.session.store" not in imports
    assert not any(
        module_name.startswith("personagraph.session.persistence")
        for module_name in imports
    )


def test_model_calls_facade_exports_the_public_runtime_contract() -> None:
    missing = [
        symbol_name
        for symbol_name in _PUBLIC_MODEL_CALL_SYMBOLS
        if not hasattr(model_calls, symbol_name)
    ]

    assert missing == []
    assert set(_PUBLIC_MODEL_CALL_SYMBOLS) <= set(model_calls.__all__)


def test_model_call_contracts_cold_import_does_not_load_execution_or_session() -> None:
    code = """
import json
import sys
import personagraph.runtime.model_calls.contracts
blocked = {
    'personagraph.model_io.gateway',
    'personagraph.runtime.model_calls.requests',
    'personagraph.session',
    'personagraph.session.store',
}
print(json.dumps(sorted(blocked & set(sys.modules))))
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(completed.stdout) == []


def test_model_call_authority_cold_import_does_not_load_provider_facade() -> None:
    code = """
import json
import sys
import personagraph.runtime.model_calls.authority
blocked = {
    'personagraph.model_io.gateway',
    'personagraph.model_io.provider_anthropic',
    'personagraph.model_io.provider_openai',
    'personagraph.model_io.structured_calls',
    'personagraph.session',
    'personagraph.session.store',
}
print(json.dumps(sorted(blocked & set(sys.modules))))
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(completed.stdout) == []


def test_model_attempt_preparation_cold_import_does_not_load_provider_facade() -> None:
    code = """
import json
import sys
import personagraph.runtime.model_calls.attempt_preparation
blocked = {
    'personagraph.model_io.gateway',
    'personagraph.model_io.provider_anthropic',
    'personagraph.model_io.provider_openai',
    'personagraph.model_io.structured_calls',
    'personagraph.session',
    'personagraph.session.store',
}
print(json.dumps(sorted(blocked & set(sys.modules))))
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(completed.stdout) == []


def test_durable_replay_model_result_hint_remains_runtime_resolvable() -> None:
    from personagraph.model_io.contracts import ModelResult
    from personagraph.runtime.model_calls.contracts import DurableModelCallReplay

    assert get_type_hints(DurableModelCallReplay)["model_result"] is ModelResult
