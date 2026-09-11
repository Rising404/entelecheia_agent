"""注入式 TaskNode 模型权威契约的边界覆盖。"""

from __future__ import annotations

import pytest

from personagraph.l2.task_execution.task_node import model_authority_contracts as contracts


def _factory(*_args: object, **_kwargs: object) -> None:
    return None


def test_plan_preserves_identity_and_callable_guards() -> None:
    plan = contracts.TaskNodeModelCallPlan(
        logical_call_id="logical-call-1",
        request_turn_id="turn-1",
        authority_factory=_factory,
        rederive_state_guard_sha256=lambda: "a" * 64,
    )
    assert plan.logical_call_id == "logical-call-1"
    assert plan.authority_factory is _factory

    with pytest.raises(ValueError, match="logical_call_id must be non-empty"):
        contracts.TaskNodeModelCallPlan(
            logical_call_id="",
            request_turn_id="turn-1",
            authority_factory=_factory,
            rederive_state_guard_sha256=lambda: "a" * 64,
        )
    with pytest.raises(TypeError, match="authority_factory must be callable"):
        contracts.TaskNodeModelCallPlan(
            logical_call_id="logical-call-1",
            request_turn_id="turn-1",
            authority_factory=None,  # type: ignore[arg-type]
            rederive_state_guard_sha256=lambda: "a" * 64,
        )
