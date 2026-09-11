"""将 lane-neutral 工作区 Tool 来源适配到 L2 WorkRun bridge。"""

from __future__ import annotations

import hashlib
import json

from ....tools.workspace.session_read_source import (
    SessionWorkspaceReadonlyRuntime,
)
from ....runtime.tool_calls import RuntimeToolLedgerStore
from .attempt_contracts import AttemptToolBridgeRequest
from .persistence_contracts import (
    ToolBridgeCallPersistence,
    ToolBridgePersistencePlan,
)
from .protected_dispatch import build_runtime_protected_tool_dispatcher
from .work_run_bridge import SqliteWorkRunToolBridge


def build_workspace_readonly_work_run_bridge(
    source: SessionWorkspaceReadonlyRuntime,
    *,
    ledger_store: RuntimeToolLedgerStore,
) -> SqliteWorkRunToolBridge:
    """将一份冻结 Host 工作区来源绑定到 L2 WorkRun 持久化。"""

    if not isinstance(source, SessionWorkspaceReadonlyRuntime):
        raise TypeError("source must be SessionWorkspaceReadonlyRuntime")
    return SqliteWorkRunToolBridge(
        catalog_snapshot=source.catalog_snapshot,
        persistence_plan_factory=_workspace_tool_persistence_plan,
        authority=source.authority,
        protected_dispatcher=build_runtime_protected_tool_dispatcher(
            source.protected_authority_by_key,
            ledger_store=ledger_store,
        ),
        protected_authority_by_key=source.protected_authority_by_key,
    )


def _workspace_tool_persistence_plan(
    request: AttemptToolBridgeRequest,
) -> ToolBridgePersistencePlan:
    """为只读调用派生与图族无关的持久 ID。"""

    common = {
        "schema_version": "workspace-tool-persistence-plan-v1",
        "session_id": request.session_id,
        "work_run_id": request.work_run_id,
        "attempt_id": request.attempt_id,
        "decision_apply_id": request.apply_id,
    }
    return ToolBridgePersistencePlan(
        decision_apply_id=request.apply_id,
        close_apply_id=_stable_id("workspace-tool-close", common),
        calls=tuple(
            ToolBridgeCallPersistence(
                tool_call_id=call.tool_call_id,
                tool_result_id=_stable_id(
                    "workspace-tool-result",
                    {**common, "tool_call_id": call.tool_call_id},
                ),
                result_apply_id=_stable_id(
                    "workspace-tool-result-apply",
                    {**common, "tool_call_id": call.tool_call_id},
                ),
            )
            for call in request.decision.action.calls
        ),
    )


def _stable_id(prefix: str, value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"{prefix}-{hashlib.sha256(encoded).hexdigest()}"


__all__ = ["build_workspace_readonly_work_run_bridge"]
