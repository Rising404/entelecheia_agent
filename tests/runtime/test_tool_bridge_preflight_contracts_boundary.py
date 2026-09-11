"""工具桥预检契约的边界覆盖。"""

from __future__ import annotations

import pytest

from personagraph.l2.task_execution.tool_bridge import preflight_contracts as contracts
from personagraph.l2.work_run import (
    HostAcceptedAttemptDecision,
    HostMaterializedCallToolsAction,
    HostMaterializedToolCall,
)


def _decision() -> HostAcceptedAttemptDecision:
    return HostAcceptedAttemptDecision(
        action=HostMaterializedCallToolsAction(
            calls=(
                HostMaterializedToolCall(
                    tool_call_id="call-1",
                    tool_id="read-document",
                    tool_version="read-document-impl-v1",
                    arguments={"document_id": "document-1"},
                    modifies_environment=False,
                ),
            )
        )
    )


def test_preflight_contracts_preserve_rejection_and_materialization_shape() -> None:
    rejected = contracts.ToolBridgeRejected(
        code=contracts.ToolBridgeRejectionCode.INVALID_TOOL_INPUT,
        message="The proposal arguments do not satisfy the exposed input schema.",
        call_ordinal=1,
        tool_id="read-document",
        details={"violations": []},
    )
    assert rejected.status == "rejected"

    materialized = contracts.ToolBridgeMaterialized(decision=_decision())
    assert materialized.status == "materialized"
    assert materialized.decision == _decision()

    with pytest.raises(ValueError):
        contracts.ToolBridgeRejected(
            code=contracts.ToolBridgeRejectionCode.INVALID_TOOL_INPUT,
            message="",
        )
