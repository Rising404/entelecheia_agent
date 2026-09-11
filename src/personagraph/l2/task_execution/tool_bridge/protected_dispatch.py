"""一次已批准受保护 ToolCall 的崩溃安全派发。

普通 Tool 执行器知道如何验证和调用一项注册，但特意不持有 Operation 标识。此适配器通过现有
Runtime 逻辑/物理工具调用账本补足该边界：物理尝试在 I/O 前即持久化；成功可精确重放；
中断或不确定调用绝不会自动再次发送。

批准并非在此决定。调用方必须先凭精确批准回执通过 Tool Policy（视觉分析使用披露授予）。
将该决策放在本模块之外，使持久回执成为已发生事项的证据，而非权限的替代品。
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from ....tools.catalog import CatalogSnapshot
from ....tools.contracts import ExecutionOutcome, ExecutionStatus, ToolError
from ....tools.execution import ResolvedInvocation, ToolExecutor
from ....tools.registration import ToolRegistration
from ....tools.policy import ProtectedToolExecutionAuthority
from ....runtime.tool_calls import (
    RuntimeLogicalToolCallAuthority,
    RuntimeToolCallAuthorityError,
    RuntimeToolCallStateGuardRejected,
    RuntimeToolCallTerminalState,
    RuntimeToolCallWaitingExternal,
    RuntimeToolDispatchObservation,
    RuntimeToolEffectClass,
    RuntimeToolLedgerStore,
    RuntimeToolLogicalRequest,
    RuntimeToolPhysicalOutcome,
    RuntimeToolRetryAuthority,
)


_ERROR_CODE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/+@-]{0,199}$")
_RESULT_CONTRACT = "runtime-tool-output-v1"


@dataclass(frozen=True, slots=True)
class ProtectedToolDispatchRequest:
    session_id: str
    turn_id: str
    invocation_turn_id: str
    work_run_id: str
    attempt_id: str
    call_ordinal: int
    tool_call_id: str
    catalog_snapshot: CatalogSnapshot
    registration: ToolRegistration
    arguments: Mapping[str, object]
    deadline_monotonic: float | None
    state_guard_sha256: str
    rederive_state_guard_sha256: Callable[[], str]
    provider_identity_sha256: str | None = None


class RuntimeProtectedToolDispatcher:
    """通过绑定提供商的持久账本运行受保护调用。

    对由一个提供商支持的目录，构造器标识仍是兼容默认值。拥有逐注册精确权威信息的桥可以在请求上
    覆盖它，使一个派发器能够安全服务多个受保护后端，而不削弱持久逻辑调用标识。
    """

    def __init__(
        self,
        *,
        provider_identity_sha256: str,
        ledger_store: RuntimeToolLedgerStore,
        executor: ToolExecutor | None = None,
    ) -> None:
        if not re.fullmatch(r"[0-9a-f]{64}", provider_identity_sha256):
            raise ValueError("provider_identity_sha256 must be a canonical hash")
        self._provider_identity_sha256 = provider_identity_sha256
        self._ledger_store = ledger_store
        self._executor = executor

    def dispatch(
        self,
        request: ProtectedToolDispatchRequest,
        *,
        executor: ToolExecutor | None = None,
    ) -> ExecutionOutcome:
        registration = request.registration
        provider_identity_sha256 = (
            self._provider_identity_sha256
            if request.provider_identity_sha256 is None
            else request.provider_identity_sha256
        )
        if not re.fullmatch(r"[0-9a-f]{64}", provider_identity_sha256):
            raise ValueError("provider_identity_sha256 must be a canonical hash")
        logical = RuntimeToolLogicalRequest.create(
            logical_tool_call_id=request.tool_call_id,
            session_id=request.session_id,
            work_run_id=request.work_run_id,
            attempt_id=request.attempt_id,
            call_ordinal=request.call_ordinal,
        # 语义 ToolCall 会跨越 Turn 重新绑定而保留。一旦预留，其原始 Turn 仍属于不可变账本绑定；
        # 恢复流程仅将当前 Turn 用于新的物理事实。
            invocation_turn_id=request.invocation_turn_id,
            catalog_snapshot_sha256=_sha256_value(
                request.catalog_snapshot.to_descriptor()
            ),
            tool_id=registration.tool_id,
            contract_version=registration.contract_version,
            implementation_version=registration.implementation_version,
            provider_identity_sha256=provider_identity_sha256,
            effect_profile_sha256=_sha256_value(
                [item.to_dict() for item in registration.effect_profile.effects]
            ),
            effect_class=RuntimeToolEffectClass.PROTECTED_EFFECT,
        # 当前聊天补全视觉传输既不发送幂等键，也不支持提供商查询。因此响应缺失需要协调，
        # 且绝不授予重试。
            retry_authority=RuntimeToolRetryAuthority.RECONCILIATION_REQUIRED,
            arguments=dict(request.arguments),
            result_contract=_RESULT_CONTRACT,
            max_physical_attempts=1,
            state_guard_sha256=request.state_guard_sha256,
        )
        authority = RuntimeLogicalToolCallAuthority(
            logical_request=logical,
            state_guard_sha256=request.rederive_state_guard_sha256,
            store=self._ledger_store,
        )
        try:
            authority.reserve(turn_id=request.turn_id)
            replay = authority.replay_succeeded_result()
            if replay is not None:
                value = replay.value
                if not isinstance(value, Mapping):
                    return _failed(
                        "protected_tool_replay_invalid",
                        "Protected ToolCall replay is not a JSON object.",
                    )
                return ExecutionOutcome.succeeded(
                    dict(value),
                    metadata={
                        "durable_replay": True,
                        "physical_attempt_id": replay.physical_attempt_id,
                    },
                )
            physical = authority.begin_physical_attempt(
                turn_id=request.turn_id,
                max_physical_attempts=1,
            )
        except RuntimeToolCallWaitingExternal:
            return _unconfirmed()
        except RuntimeToolCallStateGuardRejected:
            return ExecutionOutcome.rejected(
                ToolError(
                    "protected_tool_state_changed",
                    "Protected ToolCall authority changed before dispatch.",
                )
            )
        except RuntimeToolCallTerminalState:
            return _failed(
                "protected_tool_terminal_state",
                "Protected ToolCall already has a terminal durable outcome.",
            )
        except RuntimeToolCallAuthorityError:
            return _failed(
                "protected_tool_authority_failed",
                "Protected ToolCall authority could not be established.",
            )

        invocation = ResolvedInvocation(
            registration=registration,
            arguments=request.arguments,
            deadline_monotonic=request.deadline_monotonic,
        )
        outcome = (executor or self._executor or ToolExecutor()).execute(invocation)
        physical_outcome = _physical_outcome(outcome)
        error_code = None
        if physical_outcome is not RuntimeToolPhysicalOutcome.SUCCEEDED:
            error_code = _bounded_error_code(outcome)
        observation_values: dict[str, object] = {
            "session_id": logical.session_id,
            "work_run_id": logical.work_run_id,
            "attempt_id": logical.attempt_id,
            "logical_tool_call_id": logical.logical_tool_call_id,
            "logical_request_binding_sha256": logical.binding_sha256,
            "physical_attempt_id": physical.physical_attempt_id,
            "physical_request_binding_sha256": physical.binding_sha256,
            "physical_ordinal": physical.physical_ordinal,
            "provider_identity_sha256": logical.provider_identity_sha256,
            "tool_id": logical.tool_id,
            "contract_version": logical.contract_version,
            "implementation_version": logical.implementation_version,
            "outcome": physical_outcome,
            "error_code": error_code,
        }
        if physical_outcome is RuntimeToolPhysicalOutcome.SUCCEEDED:
            observation = RuntimeToolDispatchObservation.create(
                result=dict(outcome.result or {}),
                **observation_values,
            )
        else:
            observation = RuntimeToolDispatchObservation.create(
                **observation_values
            )
        try:
            authority.settle_physical_attempt(
                turn_id=request.turn_id,
                physical=physical,
                observation=observation,
            )
        except RuntimeToolCallAuthorityError:
            # I/O 已跨越边界。结算丢失绝不能变成自动二次发送的权限。
            return _unconfirmed()
        if physical_outcome is RuntimeToolPhysicalOutcome.UNCERTAIN:
            return _unconfirmed()
        return outcome


def build_runtime_protected_tool_dispatcher(
    authorities: Mapping[
        tuple[str, str], ProtectedToolExecutionAuthority
    ]
    | None,
    *,
    ledger_store: RuntimeToolLedgerStore,
) -> RuntimeProtectedToolDispatcher | None:
    """为一组精确注册权威创建 L2 WorkRun 持久派发器。

    每次请求都会携带其精确后端身份；构造器身份只是在旧调用遗漏逐注册
    身份时使用的确定性默认值。
    """

    if not authorities:
        return None
    identities: set[str] = set()
    for key, authority in authorities.items():
        if (
            not isinstance(key, tuple)
            or len(key) != 2
            or not all(isinstance(item, str) and item for item in key)
        ):
            raise TypeError("protected authority key must identify one registration")
        if not isinstance(authority, ProtectedToolExecutionAuthority):
            raise TypeError("protected authority map contains an invalid value")
        identities.add(authority.execution_backend_identity_sha256)
    return RuntimeProtectedToolDispatcher(
        provider_identity_sha256=min(identities),
        ledger_store=ledger_store,
    )


def _physical_outcome(outcome: ExecutionOutcome) -> RuntimeToolPhysicalOutcome:
    if outcome.status is ExecutionStatus.SUCCEEDED:
        return RuntimeToolPhysicalOutcome.SUCCEEDED
    if outcome.status in {
        ExecutionStatus.COMPLETION_UNCONFIRMED,
        ExecutionStatus.TIMED_OUT,
        ExecutionStatus.CANCELLED,
    }:
        return RuntimeToolPhysicalOutcome.UNCERTAIN
    if outcome.error is not None and outcome.error.code in {
        "tool_exception",
        "execution_timeout",
        "execution_cancelled",
        "response_unknown_after_dispatch",
    }:
        return RuntimeToolPhysicalOutcome.UNCERTAIN
    return RuntimeToolPhysicalOutcome.TERMINAL_FAILURE


def _bounded_error_code(outcome: ExecutionOutcome) -> str:
    candidate = outcome.error.code if outcome.error is not None else "tool_dispatch_failed"
    return candidate if _ERROR_CODE.fullmatch(candidate) else "tool_dispatch_failed"


def _unconfirmed() -> ExecutionOutcome:
    return ExecutionOutcome(
        ExecutionStatus.COMPLETION_UNCONFIRMED,
        error=ToolError(
            "protected_tool_completion_unconfirmed",
            "The protected ToolCall may have reached its provider and requires reconciliation.",
        ),
    )


def _failed(code: str, message: str) -> ExecutionOutcome:
    return ExecutionOutcome(ExecutionStatus.FAILED, error=ToolError(code, message))


def _sha256_value(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "build_runtime_protected_tool_dispatcher",
    'ProtectedToolDispatchRequest',
    'RuntimeProtectedToolDispatcher',
]
