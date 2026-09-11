"""将 lane-neutral 挂载文档工具来源适配到 L2 WorkRun bridge。"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from ....tools.catalog import CatalogSnapshot
from ....tools.catalog.binding import ToolBinding
from ....tools.documents.mounted_document_catalog import (
    build_mounted_document_binding_facts,
    build_mounted_document_tool_bindings,
)
from ....tools.documents.mounted_document_cognition_tools import (
    FrozenMountedDocumentToolScope,
    build_mounted_document_cognition_tool_source,
)
from ....tools.policy import AuthorityFacts
from .attempt_contracts import AttemptToolBridgeRequest
from ...auxiliary_execution.planning.mounted_document_authority import (
    FrozenMountedDocumentPlanningAuthority,
)
from .persistence_contracts import (
    ToolBridgeCallPersistence,
    ToolBridgePersistencePlan,
)
from .work_run_bridge import SqliteWorkRunToolBridge


@dataclass(frozen=True, slots=True)
class SessionMountedDocumentCognitionRuntime:
    """一个 Task 授权挂载作用域的不可变 catalog 与 WorkRun bridge。"""

    session_id: str
    catalog_snapshot: CatalogSnapshot
    tool_bridge: SqliteWorkRunToolBridge
    scope_snapshot_sha256: str
    tool_ids: tuple[str, ...]
    authority: AuthorityFacts
    contextual_bindings: tuple[ToolBinding, ...]


def build_session_mounted_document_cognition_runtime(
    authority: FrozenMountedDocumentPlanningAuthority,
) -> SessionMountedDocumentCognitionRuntime | None:
    """把共享工具来源组合成带持久化语义的 WorkRun runtime。"""

    if not isinstance(authority, FrozenMountedDocumentPlanningAuthority):
        raise TypeError(
            "authority must be FrozenMountedDocumentPlanningAuthority"
        )
    scope = FrozenMountedDocumentToolScope(
        session_id=authority.session_id,
        scope_snapshot_sha256=authority.scope_snapshot_sha256,
        documents=tuple(
            binding.frozen_document for binding in authority.bindings
        ),
    )
    source = build_mounted_document_cognition_tool_source(scope)
    if source is None:
        return None
    contextual_bindings = build_mounted_document_tool_bindings(
        tuple(
            source.catalog_snapshot.resolve(tool_id)
            for tool_id in source.tool_ids
        ),
        facts=build_mounted_document_binding_facts(scope),
    )
    bridge = SqliteWorkRunToolBridge(
        catalog_snapshot=source.catalog_snapshot,
        persistence_plan_factory=_persistence_plan,
        authority=source.authority,
    )
    return SessionMountedDocumentCognitionRuntime(
        session_id=source.session_id,
        catalog_snapshot=source.catalog_snapshot,
        tool_bridge=bridge,
        scope_snapshot_sha256=source.scope_snapshot_sha256,
        tool_ids=source.tool_ids,
        authority=source.authority,
        contextual_bindings=contextual_bindings,
    )


def _persistence_plan(
    request: AttemptToolBridgeRequest,
) -> ToolBridgePersistencePlan:
    common = {
        "schema_version": "mounted-document-cognition-persistence-v1",
        "session_id": request.session_id,
        "work_run_id": request.work_run_id,
        "attempt_id": request.attempt_id,
        "decision_apply_id": request.apply_id,
    }
    return ToolBridgePersistencePlan(
        decision_apply_id=request.apply_id,
        close_apply_id=_stable_id("mounted-document-tool-close", common),
        calls=tuple(
            ToolBridgeCallPersistence(
                tool_call_id=call.tool_call_id,
                tool_result_id=_stable_id(
                    "mounted-document-tool-result",
                    {**common, "tool_call_id": call.tool_call_id},
                ),
                result_apply_id=_stable_id(
                    "mounted-document-tool-result-apply",
                    {**common, "tool_call_id": call.tool_call_id},
                ),
            )
            for call in request.decision.action.calls
        ),
    )


def _stable_id(prefix: str, value: object) -> str:
    return f"{prefix}-{_sha256_value(value)}"


def _sha256_value(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


__all__ = [
    'SessionMountedDocumentCognitionRuntime',
    "build_session_mounted_document_cognition_runtime",
]
