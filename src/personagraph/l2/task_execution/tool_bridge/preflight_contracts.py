"""Tool Bridge 预检的纯接受与拒绝契约。"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from ...work_run import HostAcceptedAttemptDecision


class _PreflightContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class ToolBridgeRejectionCode(StrEnum):
    NOT_CALL_TOOLS = "not_call_tools"
    ID_PLAN_MISMATCH = "id_plan_mismatch"
    ATTEMPT_NOT_CURRENT = "attempt_not_current"
    ATTEMPT_ALREADY_DECIDED = "attempt_already_decided"
    CATALOG_SNAPSHOT_MISMATCH = "catalog_snapshot_mismatch"
    EXPOSED_TOOL_SET_MISMATCH = "exposed_tool_set_mismatch"
    TOOL_NOT_EXPOSED = "tool_not_exposed"
    AMBIGUOUS_EXPOSED_TOOL = "ambiguous_exposed_tool"
    INVALID_TOOL_INPUT = "invalid_tool_input"
    TOO_MANY_MODIFYING_CALLS = "too_many_modifying_calls"
    MODIFYING_CALL_UNSUPPORTED = "modifying_call_unsupported"
    POLICY_NOT_ALLOWED = "policy_not_allowed"
    INVALID_ACCEPTANCE_UPDATES = "invalid_acceptance_updates"
    MATERIALIZED_DECISION_MISMATCH = "materialized_decision_mismatch"
    STORED_RESULT_MISMATCH = "stored_result_mismatch"


class ToolBridgeRejected(_PreflightContract):
    status: Literal["rejected"] = "rejected"
    code: ToolBridgeRejectionCode
    message: str = Field(min_length=1)
    call_ordinal: int | None = Field(default=None, ge=1)
    tool_id: str | None = Field(default=None, min_length=1)
    details: dict[str, Any] = Field(default_factory=dict)


class ToolBridgeMaterialized(_PreflightContract):
    """模型输出验证返回的纯 Host 接受结果。"""

    status: Literal["materialized"] = "materialized"
    decision: HostAcceptedAttemptDecision


ToolBridgePreflightResult = ToolBridgeMaterialized | ToolBridgeRejected


__all__ = [
    'ToolBridgeMaterialized',
    'ToolBridgePreflightResult',
    'ToolBridgeRejected',
    "ToolBridgeRejectionCode",
]
