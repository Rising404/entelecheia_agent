"""Lock structured-output repair to one canonical in-process implementation."""

from __future__ import annotations

import ast
from pathlib import Path
import re

from personagraph.model_io import output_repair_contracts as repair_contracts
from personagraph.model_io import prepared_structured_provider as repair_preparation
from personagraph.runtime.model_calls import contracts as runtime_ledger_contracts


_ROOT = Path(__file__).resolve().parents[2]
_SOURCE_ROOT = _ROOT / "src" / "personagraph"
_CONTRACTS = _SOURCE_ROOT / "model_io" / "output_repair_contracts.py"

# These names represented the retired parallel protocol, callback, and renderer
# paths. Serialized schema literals are intentionally checked separately below:
# a persisted boundary version is not permission to restore a second runtime.
_RETIRED_IDENTIFIERS = frozenset(
    {
        "RuntimeModelOutputRepairFeedbackV1",
        "RuntimeModelOutputRepairFeedbackV2",
        "RuntimeModelOutputRepairIssueCategoryV2",
        "RuntimeModelOutputRepairIssueCoverageV2",
        "RuntimeModelOutputRepairIssueV2",
        "RuntimeModelOutputRepairProtocolV1",
        "_V2_OUTPUT_REPAIR_PROTOCOL",
        "_durable_uses_v2_output_repair",
        "_reject_unprepared_v2_repair",
        "l1_model_call_uses_v2_output_repair",
        "maybe_prepare_structured_v2_repair_request",
        "prepared_repair_v2",
        "prepare_repair_request_v2",
        "repair_request",
        "repair_request_v2",
        "use_v2_repair",
        "v2_repair_enabled",
    }
)
_RETIRED_LITERALS = frozenset(
    {
        "legacy-feedback-only-v1",
        "runtime-model-output-repair-feedback-v1",
    }
)
_CURRENT_SERIALIZED_LITERALS = frozenset(
    {
        "runtime-model-output-repair-issue-v2",
        "runtime-model-output-repair-feedback-v2",
        "four-message-whole-response-regeneration-v1",
        "regenerate_complete_response",
    }
)
_CANONICAL_CONTRACT_EXPORTS = frozenset(
    {
        "RuntimeModelOutputRepairFeedback",
        "RuntimeModelOutputRepairIssue",
        "RuntimeModelOutputRepairIssueCategory",
        "RuntimeModelOutputRepairIssueCoverage",
        "RuntimeModelOutputRepairProtocol",
        'RuntimeModelStructuredPrompt',
    }
)
_CANONICAL_PREPARATION_EXPORTS = frozenset(
    {
        "durable_structured_provider_prompt",
        "prepare_structured_request",
        "prepare_structured_repair_request",
    }
)
_MOVED_LEDGER_NAMES = _CANONICAL_CONTRACT_EXPORTS | {
    "RUNTIME_MODEL_REJECTED_OUTPUT_MAX_UTF8_BYTES",
    "RUNTIME_MODEL_STRUCTURED_PROMPT_COMPONENT_MAX_UTF8_BYTES",
    "runtime_model_output_repair_issue_sort_key",
}
_RETIRED_MODULE_SUFFIXES = (
    "runtime.prepared_model_providers",
    "runtime.structured_output_repair",
)


def _python_identifiers(tree: ast.AST) -> set[str]:
    identifiers: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            identifiers.add(node.name)
        elif isinstance(node, ast.Name):
            identifiers.add(node.id)
        elif isinstance(node, ast.Attribute):
            identifiers.add(node.attr)
        elif isinstance(node, ast.arg):
            identifiers.add(node.arg)
        elif isinstance(node, ast.keyword) and node.arg is not None:
            identifiers.add(node.arg)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            identifiers.update(alias.name for alias in node.names)
            identifiers.update(alias.asname for alias in node.names if alias.asname)
    return identifiers


def _string_literals(tree: ast.AST) -> set[str]:
    return {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }


def test_output_repair_has_no_retired_runtime_identifiers() -> None:
    violations: dict[str, list[str]] = {}
    for path in sorted(_SOURCE_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        identifiers = _python_identifiers(tree)
        retired = sorted(identifiers & _RETIRED_IDENTIFIERS)
        versioned_repair_types = sorted(
            identifier
            for identifier in identifiers
            if re.fullmatch(r"RuntimeModelOutputRepair[A-Za-z]*V\d+", identifier)
        )
        found = sorted(set(retired) | set(versioned_repair_types))
        if found:
            violations[str(path.relative_to(_ROOT))] = found

    assert violations == {}


def test_output_repair_exports_only_the_canonical_model_io_surface() -> None:
    contract_exports = set(repair_contracts.__all__)
    preparation_exports = set(repair_preparation.__all__)

    assert _CANONICAL_CONTRACT_EXPORTS <= contract_exports
    assert _CANONICAL_PREPARATION_EXPORTS <= preparation_exports
    assert contract_exports.isdisjoint(_RETIRED_IDENTIFIERS)
    assert preparation_exports.isdisjoint(_RETIRED_IDENTIFIERS)


def test_output_repair_types_have_one_owner_and_no_runtime_compatibility_export() -> None:
    for name in _CANONICAL_CONTRACT_EXPORTS:
        value = getattr(repair_contracts, name)
        assert value.__module__ == "personagraph.model_io.output_repair_contracts"
        assert not hasattr(runtime_ledger_contracts, name)


def test_output_repair_consumers_use_only_canonical_module_paths() -> None:
    violations: dict[str, list[str]] = {}
    for path in sorted(_SOURCE_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        found: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                imported = {alias.name for alias in node.names}
                if module.endswith("model_call_ledger_contracts"):
                    found.update(imported & _MOVED_LEDGER_NAMES)
                if module.endswith(_RETIRED_MODULE_SUFFIXES):
                    found.add(module)
            elif isinstance(node, ast.Import):
                found.update(
                    alias.name
                    for alias in node.names
                    if alias.name.endswith(_RETIRED_MODULE_SUFFIXES)
                )
        if found:
            violations[str(path.relative_to(_ROOT))] = sorted(found)

    assert violations == {}


def test_output_repair_keeps_only_the_current_persistence_literals() -> None:
    source_literals: set[str] = set()
    for path in sorted(_SOURCE_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        source_literals.update(_string_literals(tree))

    assert source_literals.isdisjoint(_RETIRED_LITERALS)
    contract_literals = _string_literals(
        ast.parse(
            _CONTRACTS.read_text(encoding="utf-8"),
            filename=str(_CONTRACTS),
        )
    )
    assert _CURRENT_SERIALIZED_LITERALS <= contract_literals
