"""共享尝试/工具桥请求契约的边界覆盖。"""

from __future__ import annotations

import pytest

from personagraph.l2.task_execution.tool_bridge import attempt_contracts as contracts
from personagraph.l2.work_run import (
    AttemptDecision,
    CallToolsAction,
    HostAcceptedAttemptDecision,
    HostMaterializedCallToolsAction,
    HostMaterializedToolCall,
    OutputWindowFormat,
    ToolCallProposal,
    WorkExecutionMutationResult,
    WriteOutputWindowAction,
)


def _proposal(*, call_count: int = 1) -> AttemptDecision:
    return AttemptDecision(
        action=CallToolsAction(
            calls=tuple(
                ToolCallProposal(
                    tool_id=f"read-tool-{ordinal}",
                    arguments={"ordinal": ordinal},
                )
                for ordinal in range(1, call_count + 1)
            )
        )
    )


def _accepted_decision() -> HostAcceptedAttemptDecision:
    return HostAcceptedAttemptDecision(
        action=HostMaterializedCallToolsAction(
            calls=(
                HostMaterializedToolCall(
                    tool_call_id="call-1",
                    tool_id="read-tool-1",
                    tool_version="read-tool-impl-v1",
                    arguments={"ordinal": 1},
                    modifies_environment=False,
                ),
            )
        )
    )


def _preflight_values(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "session_id": "session-1",
        "turn_id": "turn-1",
        "work_run_id": "work-run-1",
        "attempt_id": "attempt-1",
        "decision": _proposal(),
        "tool_call_ids": ("call-1",),
        "catalog_snapshot": {"revision": 1, "entries": []},
    }
    values.update(overrides)
    return values


def _dispatch_values(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "session_id": "session-1",
        "turn_id": "turn-1",
        "work_run_id": "work-run-1",
        "attempt_id": "attempt-1",
        "expected_work_run_revision": 3,
        "expected_progress_revision": 4,
        "expected_output_revision": 5,
        "expected_window_revision": 6,
        "apply_id": "apply-1",
        "decision": _accepted_decision(),
        "catalog_snapshot": {"revision": 1, "entries": []},
    }
    values.update(overrides)
    return values




def test_request_contracts_preserve_call_and_recovery_cursor_guards() -> None:
    assert contracts.AttemptToolBridgePreflightRequest(
        **_preflight_values()
    ).tool_call_ids == ("call-1",)
    assert contracts.AttemptToolBridgeRequest(
        **_dispatch_values()
    ).apply_id == "apply-1"

    with pytest.raises(ValueError, match="preflight accepts only call_tools"):
        contracts.AttemptToolBridgePreflightRequest(
            **_preflight_values(
                decision=AttemptDecision(
                    action=WriteOutputWindowAction(
                        content="not a tool proposal",
                        format=OutputWindowFormat.PLAIN_TEXT,
                    )
                )
            )
        )
    with pytest.raises(ValueError, match="tool_call_ids must be unique"):
        contracts.AttemptToolBridgePreflightRequest(
            **_preflight_values(
                decision=_proposal(call_count=2),
                tool_call_ids=("call-1", "call-1"),
            )
        )
    with pytest.raises(ValueError, match="exact active Attempt cursor"):
        contracts.AttemptToolBridgeRequest(
            **_dispatch_values(
                recovery_mutation=WorkExecutionMutationResult.model_construct(
                    work_run_id="wrong-work-run"
                )
            )
        )
