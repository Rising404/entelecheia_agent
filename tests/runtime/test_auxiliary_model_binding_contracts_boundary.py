"""冷态 AuxiliaryGraph 模型调用绑定的边界覆盖。"""

from __future__ import annotations

import hashlib
import json

import pytest

from personagraph.l2.auxiliary_graph import AuxiliaryNodeExecutorKind
from personagraph.l2.auxiliary_execution.adapters import (
    model_binding_contracts as contracts,
)
from personagraph.l2.work_run import AuxiliaryNodeSubject


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _subject() -> AuxiliaryNodeSubject:
    return AuxiliaryNodeSubject(
        task_id="task-1",
        auxiliary_graph_id="auxiliary-graph-1",
        auxiliary_graph_revision=2,
        node_id="node-1",
        node_revision=3,
    )


def _attempt_user_payload() -> dict[str, object]:
    subject = _subject()
    return {
        "bindings": {
            "session_id": "session-1",
            "turn_id": "turn-2",
            "work_run_id": "work-run-1",
            "work_run_revision": 4,
            "attempt_id": "attempt-1",
            "attempt_ordinal": 1,
            "subject": subject.model_dump(mode="json"),
        },
        "node": {"acceptances": []},
    }


def _binding(
    *,
    user_content: str | None = None,
) -> contracts.AuxiliaryBoundModelCall:
    return contracts.AuxiliaryBoundModelCall.create(
        call_kind="attempt_decision",
        logical_call_id="logical-call-1",
        session_id="session-1",
        goal_id="goal-1",
        subject=_subject(),
        executor_kind=AuxiliaryNodeExecutorKind.MODEL_WORK_RUN,
        request_turn_id="turn-1",
        invocation_turn_id="turn-2",
        work_run_id="work-run-1",
        work_run_revision=4,
        attempt_id="attempt-1",
        attempt_ordinal=1,
        verification_request_id=None,
        verification_request_revision=None,
        system_prompt="Execute only the frozen WorkRun request.",
        user_content=user_content or _canonical_json(_attempt_user_payload()),
        state_guard_sha256="a" * 64,
    )


def test_binding_payload_and_hash_guards_are_canonical_and_tamper_evident() -> None:
    binding = _binding()

    assert binding.max_physical_attempts == 6
    assert binding.request_sha256 == hashlib.sha256(
        binding.request_json.encode("utf-8")
    ).hexdigest()
    assert binding.binding_sha256 == hashlib.sha256(
        _canonical_json(
            binding.model_dump(mode="json", exclude={"binding_sha256"})
        ).encode("utf-8")
    ).hexdigest()
    assert json.loads(binding.request_json)["user_content"] == json.loads(
        binding.user_content
    )
    assert contracts.AuxiliaryBoundModelCall.model_validate_json(
        binding.model_dump_json()
    ) == binding

    tampered = binding.model_dump(mode="json")
    tampered["binding_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="binding hash does not match"):
        contracts.AuxiliaryBoundModelCall.model_validate(tampered)

    noncanonical = json.dumps(
        _attempt_user_payload(),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    with pytest.raises(ValueError, match="must be one canonical JSON object"):
        _binding(user_content=noncanonical)
