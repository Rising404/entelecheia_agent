"""工具桥持久化计划契约的边界覆盖。"""

from __future__ import annotations

import pytest

from personagraph.l2.task_execution.tool_bridge import persistence_contracts as contracts


def _plan(**overrides: object) -> contracts.ToolBridgePersistencePlan:
    values: dict[str, object] = {
        "decision_apply_id": "decision-1",
        "close_apply_id": "close-1",
        "calls": (
            contracts.ToolBridgeCallPersistence(
                tool_call_id="call-1",
                tool_result_id="result-1",
                result_apply_id="result-apply-1",
            ),
        ),
    }
    values.update(overrides)
    return contracts.ToolBridgePersistencePlan(**values)  # type: ignore[arg-type]


def test_persistence_plan_enforces_unique_host_identities_and_finite_deadline() -> None:
    plan = _plan()
    assert plan.calls[0].tool_call_id == "call-1"

    duplicate_call = contracts.ToolBridgeCallPersistence(
        tool_call_id="call-1",
        tool_result_id="result-2",
        result_apply_id="result-apply-2",
    )
    with pytest.raises(ValueError, match="tool_call_id values must be unique"):
        _plan(calls=(plan.calls[0], duplicate_call))

    with pytest.raises(ValueError):
        contracts.ToolBridgeCallPersistence(
            tool_call_id="call-2",
            tool_result_id="result-2",
            result_apply_id="result-apply-2",
            deadline_monotonic=float("nan"),
        )
