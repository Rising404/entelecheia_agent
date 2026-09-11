"""冷态规划上下文调用封印的边界覆盖。"""

from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError, replace
import json
from pathlib import Path
import subprocess
import sys

import pytest

from personagraph.l2.planning import invocation_contracts as contracts
from personagraph.l2.work_run.contracts import AuxiliaryNodeSubject


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64

_SCHEMA_BY_KIND = {
    contracts.PlanningContextPrimitiveKind.RESOURCE_PERCEPTION: (
        "planning-resource-perception-request-v1"
    ),
}


def _binding() -> contracts.FrozenPlanningContextArtifactBinding:
    return contracts.FrozenPlanningContextArtifactBinding(
        session_id="session_01",
        task_id="task_01",
        auxiliary_graph_id="aux_graph_01",
        goal_id="goal_01",
        producer_auxiliary_node=AuxiliaryNodeSubject(
            task_id="task_01",
            auxiliary_graph_id="aux_graph_01",
            auxiliary_graph_revision=1,
            node_id="observe_01",
            node_revision=1,
        ),
        primitive_call_id="primitive_call_01",
        artifact_id="context_artifact_01",
        verification_receipt_id="verification_receipt_01",
        authority_snapshot_id="authority_snapshot_01",
        scope_snapshot_sha256=SHA_A,
        alias_prefix="ctx",
        artifact_alias="artifact_01",
        producer_node_alias="observe_01",
    )


def _logical_request(
    kind: contracts.PlanningContextPrimitiveKind,
    binding: contracts.FrozenPlanningContextArtifactBinding,
) -> str:
    payload: dict[str, object] = {
        "schema_version": _SCHEMA_BY_KIND[kind],
        "binding": contracts._binding_payload(binding),
    }
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _freeze(
    kind: contracts.PlanningContextPrimitiveKind,
    *,
    binding: contracts.FrozenPlanningContextArtifactBinding | None = None,
    logical_request_json: str | None = None,
    **overrides: object,
) -> contracts.FrozenPlanningContextPrimitiveInvocation:
    resolved_binding = binding or _binding()
    values: dict[str, object] = {
        "primitive_kind": kind,
        "binding": resolved_binding,
        "logical_request_json": logical_request_json
        or _logical_request(kind, resolved_binding),
        "invocation_turn_id": "turn_01",
        "expected_task_state_version": 2,
        "expected_node_state_version": 3,
        "expected_control_state_version": 4,
        "expected_goal_state_version": 5,
        "expected_revision_state_version": 6,
        "expected_budget_state_version": 7,
        "authority_snapshot_sha256": SHA_A,
        "structure_sha256": SHA_B,
        "budget_snapshot_sha256": SHA_C,
    }
    values.update(overrides)
    return contracts.freeze_serialized_planning_context_primitive_invocation(
        **values,  # type: ignore[arg-type]
    )


def test_resource_perception_seals_the_exact_canonical_request() -> None:
    assert tuple(contracts.PlanningContextPrimitiveKind) == (
        contracts.PlanningContextPrimitiveKind.RESOURCE_PERCEPTION,
    )
    kind = contracts.PlanningContextPrimitiveKind.RESOURCE_PERCEPTION
    binding = _binding()
    invocation = _freeze(kind, binding=binding)

    assert invocation.primitive_kind is kind
    assert invocation.binding is binding
    assert json.loads(invocation.logical_request_json)["schema_version"] == (
        _SCHEMA_BY_KIND[kind]
    )
    assert len(invocation.logical_request_sha256) == 64
    assert len(invocation.state_guard_sha256) == 64
    with pytest.raises(FrozenInstanceError):
        invocation.invocation_turn_id = "turn_02"  # type: ignore[misc]


def test_seal_is_deterministic_and_every_authorizing_version_changes_its_guard() -> None:
    kind = contracts.PlanningContextPrimitiveKind.RESOURCE_PERCEPTION
    first = _freeze(kind)
    assert first == _freeze(kind)

    changed = _freeze(
        kind,
        expected_budget_state_version=8,
    )

    assert changed.logical_request_sha256 == first.logical_request_sha256
    assert changed.state_guard_sha256 != first.state_guard_sha256


def test_seal_fails_closed_for_noncanonical_hash_and_state_guard_drift() -> None:
    kind = contracts.PlanningContextPrimitiveKind.RESOURCE_PERCEPTION
    invocation = _freeze(kind)
    spaced_json = json.dumps(json.loads(invocation.logical_request_json), sort_keys=True)

    with pytest.raises(ValueError, match="^logical request JSON must use canonical encoding$"):
        _freeze(kind, logical_request_json=spaced_json)

    different_hash = (
        ("0" if invocation.logical_request_sha256[0] != "0" else "1")
        + invocation.logical_request_sha256[1:]
    )
    with pytest.raises(ValueError, match="^logical request hash does not match its JSON$"):
        replace(invocation, logical_request_sha256=different_hash)

    different_guard = (
        ("0" if invocation.state_guard_sha256[0] != "0" else "1")
        + invocation.state_guard_sha256[1:]
    )
    with pytest.raises(ValueError, match="^planning primitive state guard is invalid$"):
        replace(invocation, state_guard_sha256=different_guard)
    with pytest.raises(
        ValueError,
        match="^expected_task_state_version must be a positive integer$",
    ):
        _freeze(kind, expected_task_state_version=True)


def test_seal_rejects_binding_crossing() -> None:
    binding = _binding()
    resource_payload = json.loads(
        _logical_request(
            contracts.PlanningContextPrimitiveKind.RESOURCE_PERCEPTION,
            binding,
        )
    )
    resource_payload["binding"]["artifact_id"] = "other_artifact"  # type: ignore[index]
    crossed_binding = json.dumps(
        resource_payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    with pytest.raises(
        ValueError,
        match="^sealed logical request differs from its artifact binding$",
    ):
        _freeze(
            contracts.PlanningContextPrimitiveKind.RESOURCE_PERCEPTION,
            binding=binding,
            logical_request_json=crossed_binding,
        )


def test_persistence_consumers_depend_on_the_cold_contract_owner() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    expected_imports = {
        repository_root
        / "src/personagraph/session/persistence/l2/planning/primitive_invocations.py": {
            'FrozenPlanningContextArtifactBinding',
            'FrozenPlanningContextPrimitiveInvocation',
            'PlanningContextPrimitiveKind',
        },
    }

    for path, expected_names in expected_imports.items():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported_names = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            and (
                (node.module or "").endswith(
                    "planning_context_invocation_contracts"
                )
                or (node.module or "").endswith(
                    "l2.planning.invocation_contracts"
                )
            )
            for alias in node.names
        }
        assert expected_names <= imported_names


def test_contract_owner_has_only_contract_imports_and_cold_imports_without_io() -> None:
    tree = ast.parse(Path(contracts.__file__).read_text(encoding="utf-8"))
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imports |= {
        node.module or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }
    assert imports == {
        "__future__",
        "dataclasses",
        "enum",
        "hashlib",
        "json",
        "re",
        "work_run.contracts",
    }

    code = """
import json
import sys
import personagraph.l2.planning.invocation_contracts
blocked_prefixes = {
    'personagraph.model_io.gateway',
    'personagraph.retrieval',
    'personagraph.l2.planning.resource_perception',
    'personagraph.session',
    'personagraph.workspace',
}
blocked = sorted(
    name
    for name in sys.modules
    if any(name == prefix or name.startswith(prefix + '.') for prefix in blocked_prefixes)
)
print(json.dumps(blocked))
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(completed.stdout) == []
