"""冷态 AuxiliaryGraph 终态 ID 契约的边界覆盖。"""

from __future__ import annotations

import hashlib
import json
from typing import get_type_hints

import pytest
from pydantic import ValidationError

from personagraph.l2.auxiliary_graph.contracts import (
    TaskGraphSemanticTerminalCandidateBinding,
)
from personagraph.l2.auxiliary_execution.terminal import id_contracts as contracts


def _candidate_binding(*, output_revision: int = 2) -> TaskGraphSemanticTerminalCandidateBinding:
    return TaskGraphSemanticTerminalCandidateBinding(
        work_run_id="work-run-1",
        submitted_attempt_id="attempt-1",
        node_verification_request_id="verification-1",
        output_revision=output_revision,
        output_snapshot_sha256="a" * 64,
    )


def _stable_ids() -> contracts.AuxiliaryTerminalIdPlan:
    return contracts.derive_auxiliary_terminal_ids(
        session_id="session-1",
        task_id="task-1",
        auxiliary_graph_id="auxiliary-graph-1",
        goal_id="goal-1",
        auxiliary_graph_revision=2,
        structure_sha256="b" * 64,
    )


def test_terminal_id_contracts_are_deterministic_authority_bound_and_frozen() -> None:
    plan = _stable_ids()
    payload = {
        "schema_version": "auxiliary-v2-terminal-stable-ids-v1",
        "session_id": "session-1",
        "task_id": "task-1",
        "auxiliary_graph_id": "auxiliary-graph-1",
        "goal_id": "goal-1",
        "auxiliary_graph_revision": 2,
        "structure_sha256": "b" * 64,
    }
    expected_prefix = "auxv2terminal-" + hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()[:32]

    assert plan.semantic_settlement_id == f"{expected_prefix}:semantic-settlement"
    assert plan.reviewer_request_ids == (
        f"{expected_prefix}:semantic-request-1",
        f"{expected_prefix}:semantic-request-2",
    )
    assert plan == _stable_ids()
    assert plan != contracts.derive_auxiliary_terminal_ids(
        session_id="session-1",
        task_id="task-1",
        auxiliary_graph_id="auxiliary-graph-1",
        goal_id="goal-1",
        auxiliary_graph_revision=3,
        structure_sha256="b" * 64,
    )
    with pytest.raises(ValidationError, match="frozen"):
        plan.semantic_settlement_id = "other"  # type: ignore[misc]
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        contracts.AuxiliaryTerminalIdPlan(
            **plan.model_dump(),
            unexpected=True,
        )


def test_candidate_semantic_ids_bind_the_exact_candidate_and_canonical_hash() -> None:
    binding = _candidate_binding()
    plan = contracts.derive_auxiliary_terminal_candidate_semantic_ids(
        session_id="session-1",
        task_id="task-1",
        auxiliary_graph_id="auxiliary-graph-1",
        goal_id="goal-1",
        auxiliary_graph_revision=2,
        structure_sha256="b" * 64,
        candidate_binding=binding,
    )

    assert get_type_hints(
        contracts.derive_auxiliary_terminal_candidate_semantic_ids
    )["candidate_binding"] is TaskGraphSemanticTerminalCandidateBinding
    assert plan == contracts.derive_auxiliary_terminal_candidate_semantic_ids(
        session_id="session-1",
        task_id="task-1",
        auxiliary_graph_id="auxiliary-graph-1",
        goal_id="goal-1",
        auxiliary_graph_revision=2,
        structure_sha256="b" * 64,
        candidate_binding=binding,
    )
    assert plan != contracts.derive_auxiliary_terminal_candidate_semantic_ids(
        session_id="session-1",
        task_id="task-1",
        auxiliary_graph_id="auxiliary-graph-1",
        goal_id="goal-1",
        auxiliary_graph_revision=2,
        structure_sha256="b" * 64,
        candidate_binding=_candidate_binding(output_revision=3),
    )
    assert contracts._sha256_value({"a": {"x": 1, "y": 2}, "b": [2, 1]}) == (
        contracts._sha256_value({"b": [2, 1], "a": {"y": 2, "x": 1}})
    )
