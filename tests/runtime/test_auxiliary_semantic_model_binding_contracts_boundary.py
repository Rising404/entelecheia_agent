"""冷态 AuxiliaryGraph 语义模型绑定的边界覆盖。"""

from __future__ import annotations

import hashlib
import json

import pytest

from personagraph.l2.auxiliary_execution.verification import (
    model_binding_contracts as contracts,
)


def _request_payload() -> dict[str, object]:
    return {
        "schema_version": "auxiliary-v2-semantic-reviewer-model-call-v1",
        "verification_result_id": "verification-result-1",
        "verification_request": {
            "schema_version": "task-graph-semantic-verification-request-v1",
            "review": "one immutable semantic request",
        },
    }


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _binding() -> contracts.AuxiliarySemanticReviewerModelCallBinding:
    request_json = _canonical_json(_request_payload())
    return contracts.AuxiliarySemanticReviewerModelCallBinding(
        logical_call_id="logical-call-1",
        verification_request_id="verification-request-1",
        verification_result_id="verification-result-1",
        reviewer_ordinal=1,
        required_reviewer_count=2,
        session_id="session-1",
        task_id="task-1",
        auxiliary_graph_id="auxiliary-graph-1",
        goal_id="goal-1",
        auxiliary_graph_revision=2,
        invocation_turn_id="turn-1",
        request_json=request_json,
        request_sha256=hashlib.sha256(request_json.encode("utf-8")).hexdigest(),
        state_guard_sha256="a" * 64,
    )


def test_binding_request_is_canonical_hash_guarded_and_replayable() -> None:
    binding = _binding()

    assert binding.max_physical_attempts == 6
    assert binding.request_contract == "auxiliary-v2-semantic-reviewer-model-call-v1"
    assert binding.typed_result_contract == (
        "task-graph-semantic-verification-model-envelope-v1"
    )
    assert binding.request_sha256 == hashlib.sha256(
        binding.request_json.encode("utf-8")
    ).hexdigest()
    assert json.loads(binding.request_json) == _request_payload()
    assert contracts.AuxiliarySemanticReviewerModelCallBinding.model_validate_json(
        binding.model_dump_json()
    ) == binding
    assert contracts._sha256_value({"a": {"x": 1, "y": 2}, "b": [2, 1]}) == (
        contracts._sha256_value({"b": [2, 1], "a": {"y": 2, "x": 1}})
    )

    tampered_hash = binding.model_dump(mode="json")
    tampered_hash["request_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="not canonical"):
        contracts.AuxiliarySemanticReviewerModelCallBinding.model_validate(
            tampered_hash
        )

    noncanonical_json = json.dumps(
        _request_payload(),
        ensure_ascii=False,
        separators=(",", ": "),
    )
    noncanonical = binding.model_dump(mode="json")
    noncanonical["request_json"] = noncanonical_json
    noncanonical["request_sha256"] = hashlib.sha256(
        noncanonical_json.encode("utf-8")
    ).hexdigest()
    with pytest.raises(ValueError, match="not canonical"):
        contracts.AuxiliarySemanticReviewerModelCallBinding.model_validate(
            noncanonical
        )
