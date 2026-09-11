"""Entry Turn 生命周期变更与结算所需的持久化边界。"""

from __future__ import annotations

from typing import Protocol

from ...turn_events import TurnEvent
from .replay import EntryReplayReadPort
from .settlement import EntrySettlementRecoveryStorePort
from .window_audit import TurnWindowAuditStore


class EntryTurnLookupStorePort(Protocol):
    """按客户端 request id 查找已接受 Turn。"""

    def get_turn_execution_for_client_request(
        self,
        *,
        session_id: str,
        client_request_id: str,
    ) -> dict[str, object] | None: ...


class EntryTurnAcceptanceStorePort(TurnWindowAuditStore, Protocol):
    """接受新 Turn 以及解决旧 Window 冲突所需的持久化表面。"""

    def accept_turn_execution(self, **kwargs: object) -> dict[str, object]: ...


class EntryTurnMutationStorePort(Protocol):
    """推进、记录或中断当前 Entry Turn 所需的基础变更。"""

    def append_runtime_turn_event(
        self,
        event: TurnEvent,
        *,
        active_window_lease_owner: str | None = None,
    ) -> int: ...

    def advance_turn_execution_window(self, **kwargs: object) -> dict[str, object]: ...

    def mark_turn_execution_interrupted(
        self,
        **kwargs: object,
    ) -> dict[str, object]: ...


class EntryTurnFinalizationStorePort(
    EntryTurnMutationStorePort,
    EntrySettlementRecoveryStorePort,
    Protocol,
):
    """提交正式、已验证或无公开输出的权威 Turn 结果。"""

    def finalize_turn_execution(self, **kwargs: object) -> dict[str, object]: ...

    def finalize_verified_turn_execution(
        self,
        **kwargs: object,
    ) -> dict[str, object]: ...

    def finalize_authoritative_referenced_turn_execution(
        self,
        **kwargs: object,
    ) -> dict[str, object]: ...

    def mark_authoritative_no_public_turn_stop(
        self,
        **kwargs: object,
    ) -> dict[str, object]: ...


class EntryReplayStorePort(
    EntryReplayReadPort,
    EntryTurnFinalizationStorePort,
    Protocol,
):
    """重放投影及其可证明结算恢复所需的完整 Entry 表面。"""


class EntryLifecycleStorePort(
    EntryTurnLookupStorePort,
    EntryTurnAcceptanceStorePort,
    EntryTurnFinalizationStorePort,
    EntryReplayReadPort,
    Protocol,
):
    """Entry composition root 提供的完整生命周期持久化表面。"""


__all__ = [
    "EntryLifecycleStorePort",
    "EntryReplayStorePort",
    "EntryTurnAcceptanceStorePort",
    "EntryTurnFinalizationStorePort",
    "EntryTurnLookupStorePort",
    "EntryTurnMutationStorePort",
]
