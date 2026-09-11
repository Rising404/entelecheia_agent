"""面向冻结工作区、绑定会话的写入工具来源。

本模块将工作区写入注册绑定到会话及其显式写入批准回执。它刻意不创建 WorkRun
桥接器或 L1 分派器：每条执行通道持有自己的持久受保护操作账本，本来源只提供
注册项、contextual Binding 与当前权威。
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass, replace

from ...session import store as session_store
from ..catalog.binding import ToolBinding
from ..policy import AuthorityFacts, ScopeGrant
from .workspace_tools import FrozenWorkspaceToolBoundary
from .workspace_write_catalog import (
    WorkspaceWriteBindingFacts,
    build_workspace_write_tool_bindings,
    derive_workspace_write_source_fingerprint,
)
from .workspace_write_tools import (
    build_workspace_write_tool_registrations,
)
from ..effects import EffectAction, EffectResource, EffectScopeKind
from ..registration import ToolRegistration
from ...session.local_file_authority import (
    LocalFileAction,
    LocalFileAuthorizationDecision,
    SqliteSessionFileAuthority,
)
from ..policy import ProtectedToolExecutionAuthority


@dataclass(frozen=True, slots=True)
class SessionWorkspaceActionToolSource:
    """绑定会话、但可能仍在等待批准的操作能力。"""

    session_id: str
    boundary: FrozenWorkspaceToolBoundary
    registrations: tuple[ToolRegistration, ...]
    workspace_write_bindings: tuple[ToolBinding]
    authority: AuthorityFacts
    protected_authority_by_key: dict[
        tuple[str, str], ProtectedToolExecutionAuthority
    ]
    write_approval_receipt_id: str | None
    write_authorization_reason: str
    scope_snapshot_sha256: str


def build_session_workspace_action_tool_source(
    *,
    session_id: str,
    boundary: FrozenWorkspaceToolBoundary,
    boundary_sha256: str,
    protected_dispatch_sha256: str,
    operation_ledger_sha256: str,
) -> SessionWorkspaceActionToolSource:
    """绑定工作区写入器，并报告是否存在显式批准。

    即使没有授权，写入器仍对模型可见，使策略能够返回精确的
    ``authorization_required`` 结果。只有当前有效的 UPDATE 授权才会同时提供
    批准事实和可执行的受保护操作回执。调用通道仅注入无秘密的分派和账本
    协议摘要；某次 ToolCall 的 operation identity 仍由该通道在预留时生成。
    """

    if not isinstance(session_id, str) or not session_id.strip():
        raise ValueError("session_id must be non-empty")
    if boundary.session_id != session_id:
        raise ValueError("workspace action source crossed Session authority")
    if session_store.get_session(session_id) is None:
        raise ValueError("session does not exist")

    file_authority = SqliteSessionFileAuthority()
    frozen_file_authority = SqliteSessionFileAuthority(file_authority.path)
    decision = file_authority.authorize(
        session_id=session_id,
        candidate=boundary.root,
        action=LocalFileAction.UPDATE,
        working_root=boundary.root,
    )
    root_scope = str(boundary.root)
    grants = [
        ScopeGrant(
            EffectResource.FILESYSTEM,
            EffectAction.READ,
            EffectScopeKind.WORKSPACE,
            root_scope,
        )
    ]
    approvals: tuple[ScopeGrant, ...] = ()
    protected_authority_by_key: dict[
        tuple[str, str], ProtectedToolExecutionAuthority
    ] = {}
    receipt_id = decision.grant_id if decision.allowed else None
    binding_facts = WorkspaceWriteBindingFacts(
        boundary_sha256=boundary_sha256,
        update_authority_sha256=_sha256_value(
            {
                "schema_version": "session-workspace-update-authority-binding-v1",
                "session_id": session_id,
                "boundary_sha256": boundary_sha256,
                "action": LocalFileAction.UPDATE.value,
                "allowed": decision.allowed,
                "reason": decision.reason.value,
                "grant_id": receipt_id,
                "authority_store_sha256": _sha256_value(
                    {
                        "schema_version": (
                            "sqlite-session-file-authority-store-v1"
                        ),
                        "database_path": str(file_authority.path),
                    }
                ),
            }
        ),
        protected_dispatch_sha256=protected_dispatch_sha256,
        operation_ledger_sha256=operation_ledger_sha256,
    )
    source_fingerprint = derive_workspace_write_source_fingerprint(
        binding_facts
    )
    registrations = tuple(
        replace(
            registration,
            source=replace(
                registration.source,
                fingerprint=source_fingerprint,
            ),
        )
        for registration in build_workspace_write_tool_registrations(boundary)
    )
    workspace_write_bindings = build_workspace_write_tool_bindings(
        registrations,
        facts=binding_facts,
    )
    if decision.allowed and receipt_id:
        update = ScopeGrant(
            EffectResource.FILESYSTEM,
            EffectAction.UPDATE,
            EffectScopeKind.WORKSPACE,
            root_scope,
        )
        grants.append(update)
        approvals = (update,)
        for registration in registrations:
            protected_authority_by_key[
                (registration.tool_id, registration.contract_version)
            ] = ProtectedToolExecutionAuthority(
                approval_receipt_ids=(receipt_id,),
                execution_backend_identity_sha256=_sha256_value(
                    {
                        "schema_version": "workspace-local-write-backend-v2",
                        "root_device": boundary.root_device,
                        "root_inode": boundary.root_inode,
                        "root": root_scope,
                        "session_id": session_id,
                        "tool": registration.descriptor(),
                    }
                ),
                revalidate=_write_grant_is_current(
                    authority=frozen_file_authority,
                    session_id=session_id,
                    boundary=boundary,
                    receipt_id=receipt_id,
                ),
            )
    authority = AuthorityFacts(
        grants=tuple(grants),
        approval_grants=approvals,
    )
    scope_snapshot_sha256 = _sha256_value(
        {
            "schema_version": "session-workspace-action-source-v2",
            "session_id": session_id,
            "root": root_scope,
            "root_device": boundary.root_device,
            "root_inode": boundary.root_inode,
            "write_authorization_reason": decision.reason.value,
            "write_approval_receipt_id": receipt_id,
            "registrations": [item.descriptor() for item in registrations],
            "workspace_write_bindings": [
                item.descriptor() for item in workspace_write_bindings
            ],
        }
    )
    return SessionWorkspaceActionToolSource(
        session_id=session_id,
        boundary=boundary,
        registrations=registrations,
        workspace_write_bindings=workspace_write_bindings,
        authority=authority,
        protected_authority_by_key=protected_authority_by_key,
        write_approval_receipt_id=receipt_id,
        write_authorization_reason=decision.reason.value,
        scope_snapshot_sha256=scope_snapshot_sha256,
    )


def _write_grant_is_current(
    *,
    authority: SqliteSessionFileAuthority,
    session_id: str,
    boundary: FrozenWorkspaceToolBoundary,
    receipt_id: str,
) -> Callable[[], bool]:
    def revalidate() -> bool:
        try:
            boundary.require_current_root()
            decision: LocalFileAuthorizationDecision = authority.authorize(
                session_id=session_id,
                candidate=boundary.root,
                action=LocalFileAction.UPDATE,
                working_root=boundary.root,
            )
        except (OSError, RuntimeError, ValueError):
            return False
        return bool(decision.allowed and decision.grant_id == receipt_id)

    return revalidate


def _sha256_value(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    'SessionWorkspaceActionToolSource',
    "build_session_workspace_action_tool_source",
]
