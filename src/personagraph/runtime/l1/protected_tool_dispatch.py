"""ToolCall：延迟安全地分发 L1 负责的受保护工具效果。

L1 使用自身的 ``ready -> dispatching -> terminal`` 状态机。跨 I/O 边界的未知
效果不得自动再次执行；唯一可恢复项是已绑定持久文档请求的文件准备，只继续等待
并补结算原请求，不创建第二个物理效果身份。
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from ...tools.contracts import ExecutionOutcome, ExecutionStatus, ToolError
from ...tools.execution import ResolvedInvocation, ToolExecutor
from ...tools.registration import ToolRegistration
from .identity import canonical_json, sha256_json
from .ports import L1StorePort
from .recovery import can_resume_file_preparation_call


def l1_protected_dispatch_identity_sha256() -> str:
    """Identify the trusted L1 route without binding one operation or receipt."""

    return sha256_json(
        {
            "schema_version": "l1-protected-tool-dispatch-route-v1",
            "dispatcher": 'L1ProtectedToolDispatcher',
            "authority_revalidation": (
                "before_and_after_durable_dispatch_begin"
            ),
            "physical_boundary": "ready_to_dispatching_before_handler",
            "uncertain_completion": "resume_bound_file_preparation_else_settle_unconfirmed",
        }
    )


def l1_protected_operation_ledger_identity_sha256() -> str:
    """Identify the L1 protected-operation ledger protocol, not a ToolCall."""

    return sha256_json(
        {
            "schema_version": "l1-protected-operation-ledger-v1",
            "store_port": "L1StorePort",
            "reserve": "reserve_l1_tool_call",
            "begin": "begin_l1_protected_tool_dispatch",
            "settle": "settle_l1_protected_tool_dispatch",
            "execution_class": "protected_effect",
            "physical_attempt_identity": "l1-protected-physical-attempt-v1",
            "maximum_physical_attempts": 1,
        }
    )


@dataclass(frozen=True, slots=True)
class L1ProtectedToolDispatchRequest:
    """确切的已预留 L1 效果允许跨 I/O 一次。"""

    session_id: str
    turn_id: str
    l1_turn_run_id: str
    attempt_id: str
    tool_call_id: str
    expected_window_revision: int
    expected_lease_owner: str | None
    protected_operation_binding_sha256: str
    registration: ToolRegistration
    arguments: Mapping[str, Any]
    deadline_monotonic: float
    revalidate_authority: Callable[[], bool]
    continuation_check: Callable[[], bool] | None = None


@dataclass(frozen=True, slots=True)
class L1ProtectedToolDispatchResult:
    """一个结果，其持久化结算位控制控制器恢复。"""

    outcome: ExecutionOutcome
    tool_call: Mapping[str, object] | None
    durably_settled: bool


class L1ProtectedToolDispatcher:
    """在调用其处理器之前，使一个 L1 保护操作持久化。"""

    def __init__(
        self,
        *,
        store: L1StorePort,
        executor: ToolExecutor | None = None,
    ) -> None:
        self._store = store
        self._executor = executor

    def dispatch(
        self,
        request: L1ProtectedToolDispatchRequest,
    ) -> L1ProtectedToolDispatchResult:
        physical_attempt_id = _physical_attempt_id(
            request.protected_operation_binding_sha256
        )
        authority_valid_before_begin = _authority_is_current(
            request.revalidate_authority
        )
        try:
            begun = self._store.begin_l1_protected_tool_dispatch(
                session_id=request.session_id,
                turn_id=request.turn_id,
                l1_turn_run_id=request.l1_turn_run_id,
                attempt_id=request.attempt_id,
                tool_call_id=request.tool_call_id,
                expected_window_revision=request.expected_window_revision,
                expected_lease_owner=request.expected_lease_owner,
                protected_operation_binding_sha256=(
                    request.protected_operation_binding_sha256
                ),
                physical_attempt_id=physical_attempt_id,
            )
        except Exception:
            # 我们无法证明存储故障发生在之前还是
            # 之后发生。将这种情况视为未知状态，而不是
            # 让控制器回退到直接执行器调用。
            return L1ProtectedToolDispatchResult(
                outcome=_completion_unconfirmed(),
                tool_call=None,
                durably_settled=False,
            )

        stored = begun.get("tool_call")
        if not isinstance(stored, Mapping):
            return L1ProtectedToolDispatchResult(
                outcome=_completion_unconfirmed(),
                tool_call=None,
                durably_settled=False,
            )
        if begun.get("started") is not True and not can_resume_file_preparation_call(
            session_id=request.session_id, tool_call=stored,
        ):
            # 一般副作用仍不可重放。文档准备必须证明每项原输入已持久绑定，才可由
            # 同一 handler 重新校验原来源并继续观察原请求、补交原 physical receipt。
            return self._settle(
                request=request,
                physical_attempt_id=physical_attempt_id,
                outcome=_completion_unconfirmed(),
            )

        if not authority_valid_before_begin or not _authority_is_current(
            request.revalidate_authority
        ):
            return self._settle(
                request=request,
                physical_attempt_id=physical_attempt_id,
                outcome=ExecutionOutcome.rejected(
                    ToolError(
                        "protected_tool_authorization_revoked",
                        "The protected ToolCall authority is no longer current.",
                    )
                ),
            )

        outcome = (self._executor or ToolExecutor()).execute(
            ResolvedInvocation(
                registration=request.registration,
                arguments=dict(request.arguments),
                deadline_monotonic=request.deadline_monotonic,
                logical_tool_call_id=request.tool_call_id,
                continuation_check=request.continuation_check,
            )
        )
        return self._settle(
            request=request,
            physical_attempt_id=physical_attempt_id,
            outcome=outcome,
        )

    def _settle(
        self,
        *,
        request: L1ProtectedToolDispatchRequest,
        physical_attempt_id: str,
        outcome: ExecutionOutcome,
    ) -> L1ProtectedToolDispatchResult:
        payload = outcome.to_dict()
        try:
            settled = self._store.settle_l1_protected_tool_dispatch(
                tool_call_id=request.tool_call_id,
                protected_operation_binding_sha256=(
                    request.protected_operation_binding_sha256
                ),
                physical_attempt_id=physical_attempt_id,
                outcome_status=outcome.status.value,
                outcome_json=canonical_json(payload),
                outcome_hash=sha256_json(payload),
            )
        except Exception:
            # 处理程序已经运行过（或者之前的调用可能已经运行过）。切勿将失败的结算转换为重试权限。
            # 运行).  从未将失败的结算转换为重试权限.
            return L1ProtectedToolDispatchResult(
                outcome=_completion_unconfirmed(),
                tool_call=None,
                durably_settled=False,
            )
        stored = settled.get("tool_call")
        if not isinstance(stored, Mapping):
            return L1ProtectedToolDispatchResult(
                outcome=_completion_unconfirmed(),
                tool_call=None,
                durably_settled=False,
            )
        return L1ProtectedToolDispatchResult(
            outcome=outcome,
            tool_call=stored,
            durably_settled=True,
        )


def _authority_is_current(check: Callable[[], bool]) -> bool:
    try:
        return check() is True
    except Exception:
        return False


def _physical_attempt_id(operation_binding_sha256: str) -> str:
    if len(operation_binding_sha256) != 64:
        raise ValueError("protected operation binding must be sha256")
    return "l1phys_" + hashlib.sha256(
        canonical_json(
            {
                "schema_version": "l1-protected-physical-attempt-v1",
                "operation_binding_sha256": operation_binding_sha256,
            }
        ).encode("utf-8")
    ).hexdigest()


def _completion_unconfirmed() -> ExecutionOutcome:
    return ExecutionOutcome(
        status=ExecutionStatus.COMPLETION_UNCONFIRMED,
        error=ToolError(
            "protected_tool_completion_unconfirmed",
            "The protected ToolCall may have crossed its execution boundary.",
        ),
    )


__all__ = [
    'L1ProtectedToolDispatchRequest',
    'L1ProtectedToolDispatchResult',
    'L1ProtectedToolDispatcher',
    "l1_protected_dispatch_identity_sha256",
    "l1_protected_operation_ledger_identity_sha256",
]
