"""纯 TaskNode 模型调用绑定的边界覆盖。"""

from __future__ import annotations

import hashlib
import json

import pytest

from personagraph.l2.task_execution.task_node import model_binding_contracts as contracts
from personagraph.l2.work_run import TaskNodeSubject


def _subject() -> TaskNodeSubject:
    return TaskNodeSubject(
        task_id="task-1",
        graph_revision=2,
        node_id="node-1",
        node_revision=3,
    )


def _attempt_user_content() -> str:
    return json.dumps(
        {
            "bindings": {
                "session_id": "session-1",
                "turn_id": "turn-2",
                "work_run_id": "work-run-1",
                "work_run_revision": 4,
                "attempt_id": "attempt-1",
                "attempt_ordinal": 1,
            },
            "node": {
                "subject": _subject().model_dump(mode="json"),
                "objective": "produce one bounded result",
            },
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _binding() -> contracts.TaskNodeBoundModelCall:
    return contracts.TaskNodeBoundModelCall.create(
        call_kind="attempt_decision",
        logical_call_id="logical-call-1",
        session_id="session-1",
        subject=_subject(),
        request_turn_id="turn-1",
        invocation_turn_id="turn-2",
        work_run_id="work-run-1",
        dispatch_work_run_revision=4,
        attempt_id="attempt-1",
        attempt_ordinal=1,
        verification_request_id=None,
        verification_request_revision=None,
        locked_work_run_revision=None,
        system_prompt="TaskNode system prompt",
        user_content=_attempt_user_content(),
        state_guard_sha256="a" * 64,
    )


def test_binding_payload_and_hash_guards_are_stable() -> None:
    binding = _binding()
    assert binding.max_physical_attempts == 6
    assert binding.request_sha256 == hashlib.sha256(
        binding.request_json.encode("utf-8")
    ).hexdigest()
    assert binding.binding_sha256 == contracts.task_node_model_state_guard_sha256(
        binding.model_dump(mode="json", exclude={"binding_sha256"})
    )
    assert contracts.TaskNodeBoundModelCall.model_validate_json(
        binding.model_dump_json()
    ) == binding
    assert contracts.task_node_model_state_guard_sha256(
        {"b": [2, 1], "a": {"y": 1, "x": 2}}
    ) == contracts.task_node_model_state_guard_sha256(
        {"a": {"x": 2, "y": 1}, "b": [2, 1]}
    )

    tampered = binding.model_dump(mode="json")
    tampered["binding_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="binding hash does not match"):
        contracts.TaskNodeBoundModelCall.model_validate(tampered)
