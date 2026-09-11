"""由有界 L1 执行通道负责的精简 Host 端口。"""

from __future__ import annotations

from typing import Protocol

from ..model_calls.contracts import RuntimeModelLedgerStore
from ..turn_events import TurnEvent


class L1ReplayRecoveryStore(Protocol):
    """重新认领一个未完成 L1 TurnRun 所需的恢复表面。"""

    def claim_l1_turn_run_resume(self, **kwargs: object) -> dict[str, object]: ...

    def renew_l1_turn_run_resume_lease(self, **kwargs: object) -> bool: ...

    def append_runtime_turn_event(
        self,
        event: TurnEvent,
        *,
        active_window_lease_owner: str | None = None,
    ) -> int: ...


class L1EntryBootstrapStorePort(Protocol):
    """创建并冻结一次新 L1 TurnRun 所需的唯一写入。"""

    def create_l1_turn_run(self, **kwargs: object) -> dict[str, object]: ...


class L1StorePort(RuntimeModelLedgerStore, L1ReplayRecoveryStore, Protocol):
    """仅允许 L1 控制器使用的持久化能力。"""

    def initialize_l1_turn_run(self, **kwargs: object) -> dict[str, object]: ...

    def create_execution_findings_ledger(self, **kwargs: object) -> object: ...

    def get_execution_findings_ledger_for_owner(
        self, **kwargs: object
    ) -> object: ...

    def apply_execution_findings_mutation(self, **kwargs: object) -> object: ...

    def close_execution_findings_ledger(self, **kwargs: object) -> object: ...

    def start_l1_attempt(self, **kwargs: object) -> dict[str, object]: ...

    def get_l1_attempt_state_guard(self, **kwargs: object) -> str: ...

    def commit_l1_attempt_decision(self, **kwargs: object) -> dict[str, object]: ...

    def reject_l1_final_reply_candidate(
        self, **kwargs: object
    ) -> dict[str, object]: ...

    def reserve_l1_tool_call(self, **kwargs: object) -> dict[str, object]: ...

    def begin_l1_protected_tool_dispatch(
        self, **kwargs: object
    ) -> dict[str, object]: ...

    def settle_l1_tool_call(self, **kwargs: object) -> dict[str, object]: ...

    def settle_l1_protected_tool_dispatch(
        self, **kwargs: object
    ) -> dict[str, object]: ...

    def close_l1_attempt(self, **kwargs: object) -> dict[str, object]: ...

    def get_l1_turn_execution(self, **kwargs: object) -> dict[str, object] | None: ...

    def fail_l1_turn_run(self, **kwargs: object) -> None: ...


__all__ = [
    "L1EntryBootstrapStorePort",
    "L1ReplayRecoveryStore",
    "L1StorePort",
]
