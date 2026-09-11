"""Runtime 模型与工具调用账本的 Store 组合门面。

``session.store`` 保留既定公开导入路径。本模块只负责动态依赖组合：为每项持久模型
或工具账本操作解析新的 ``StoreDeps``，并把原始逻辑调用、物理尝试或结算输入直接转发给
对应账本记录。它刻意不拥有 Store 全局状态、Runtime 控制策略、事务或 schema。
"""

from __future__ import annotations

from collections.abc import Callable

from ..runtime.model_calls.contracts import (
    RuntimeModelLogicalRequest,
    RuntimeModelPhysicalAttemptRequest,
    RuntimeModelPhysicalAttemptSettlement,
)
from ..runtime.tool_calls import (
    RuntimeToolLogicalRequest,
    RuntimeToolPhysicalAttemptRequest,
    RuntimeToolPhysicalAttemptSettlement,
)
from .persistence.calls import runtime_model_calls, runtime_tool_calls
from .persistence.deps import StoreDeps


# 保留既定 Store 契约身份；校验和事务仍归底层记录所有。
RuntimeModelCallPersistenceError = runtime_model_calls.RuntimeModelCallPersistenceError
RuntimeModelCallIdentityCollision = runtime_model_calls.RuntimeModelCallIdentityCollision
RuntimeModelCallTerminalState = runtime_model_calls.RuntimeModelCallTerminalState
RuntimeModelLedgerMutationResult = runtime_model_calls.RuntimeModelLedgerMutationResult
StoredRuntimeModelLogicalCall = runtime_model_calls.StoredRuntimeModelLogicalCall
StoredRuntimeModelPhysicalAttempt = runtime_model_calls.StoredRuntimeModelPhysicalAttempt
StoredRuntimeModelRejectedOutput = runtime_model_calls.StoredRuntimeModelRejectedOutput

RuntimeToolCallPersistenceError = runtime_tool_calls.RuntimeToolCallPersistenceError
RuntimeToolCallIdentityCollision = runtime_tool_calls.RuntimeToolCallIdentityCollision
RuntimeToolCallTerminalState = runtime_tool_calls.RuntimeToolCallTerminalState
RuntimeToolCallWaitingExternalState = runtime_tool_calls.RuntimeToolCallWaitingExternalState
RuntimeToolLedgerMutationResult = runtime_tool_calls.RuntimeToolLedgerMutationResult
StoredRuntimeToolLogicalCall = runtime_tool_calls.StoredRuntimeToolLogicalCall
StoredRuntimeToolPhysicalAttempt = runtime_tool_calls.StoredRuntimeToolPhysicalAttempt


class RuntimeCallLedgerStoreFacade:
    """保留 Store 的 Runtime 账本接口，但不拥有持久化。"""

    def __init__(self, *, deps_factory: Callable[[], StoreDeps]) -> None:
        self._deps_factory = deps_factory

    def reserve_runtime_model_logical_call(
        self,
        *,
        request: RuntimeModelLogicalRequest,
    ) -> RuntimeModelLedgerMutationResult:
        """预留或精确重放一个与 provider 无关的逻辑模型请求。"""

        return runtime_model_calls.reserve_runtime_model_logical_call(
            self._deps_factory(),
            request=request,
        )

    def append_runtime_model_physical_attempt(
        self,
        *,
        request: RuntimeModelPhysicalAttemptRequest,
    ) -> RuntimeModelLedgerMutationResult:
        """追加或精确重放一次 provider 调用前的物理分派。"""

        return runtime_model_calls.append_runtime_model_physical_attempt(
            self._deps_factory(),
            request=request,
        )

    def settle_runtime_model_physical_attempt(
        self,
        *,
        settlement: RuntimeModelPhysicalAttemptSettlement,
        rejected_response_text: str | None = None,
    ) -> RuntimeModelLedgerMutationResult:
        """结算或精确重放一次物理模型调用尝试。"""

        return runtime_model_calls.settle_runtime_model_physical_attempt(
            self._deps_factory(),
            settlement=settlement,
            rejected_response_text=rejected_response_text,
        )

    def get_runtime_model_logical_call(
        self,
        *,
        session_id: str,
        logical_call_id: str,
    ) -> StoredRuntimeModelLogicalCall | None:
        """加载一个经过认证的逻辑请求及其有序尝试。"""

        return runtime_model_calls.get_runtime_model_logical_call(
            self._deps_factory(),
            session_id=session_id,
            logical_call_id=logical_call_id,
        )

    def get_runtime_model_rejected_output(
        self,
        *,
        session_id: str,
        logical_call_id: str,
        rejected_physical_ordinal: int,
        rejected_response_sha256: str,
    ) -> StoredRuntimeModelRejectedOutput | None:
        """加载修复反馈指定的精确已认证正文。"""

        return runtime_model_calls.get_runtime_model_rejected_output(
            self._deps_factory(),
            session_id=session_id,
            logical_call_id=logical_call_id,
            rejected_physical_ordinal=rejected_physical_ordinal,
            rejected_response_sha256=rejected_response_sha256,
        )

    def reserve_runtime_tool_logical_call(
        self,
        *,
        request: RuntimeToolLogicalRequest,
    ) -> RuntimeToolLedgerMutationResult:
        """预留或精确重放一个 WorkRun 所有的逻辑工具请求。"""

        return runtime_tool_calls.reserve_runtime_tool_logical_call(
            self._deps_factory(),
            request=request,
        )

    def append_runtime_tool_physical_attempt(
        self,
        *,
        request: RuntimeToolPhysicalAttemptRequest,
    ) -> RuntimeToolLedgerMutationResult:
        """追加或精确重放一次分派前物理工具尝试。"""

        return runtime_tool_calls.append_runtime_tool_physical_attempt(
            self._deps_factory(),
            request=request,
        )

    def settle_runtime_tool_physical_attempt(
        self,
        *,
        settlement: RuntimeToolPhysicalAttemptSettlement,
    ) -> RuntimeToolLedgerMutationResult:
        """结算或精确重放一次物理工具调用尝试。"""

        return runtime_tool_calls.settle_runtime_tool_physical_attempt(
            self._deps_factory(),
            settlement=settlement,
        )

    def get_runtime_tool_logical_call(
        self,
        *,
        session_id: str,
        logical_tool_call_id: str,
    ) -> StoredRuntimeToolLogicalCall | None:
        """加载一个经过认证的工具请求及其有序分派。"""

        return runtime_tool_calls.get_runtime_tool_logical_call(
            self._deps_factory(),
            session_id=session_id,
            logical_tool_call_id=logical_tool_call_id,
        )


def build_runtime_call_ledger_store_facade(
    *,
    deps_factory: Callable[[], StoreDeps],
) -> RuntimeCallLedgerStoreFacade:
    """构建 Store 账本门面，暂不解析依赖。"""

    return RuntimeCallLedgerStoreFacade(deps_factory=deps_factory)
