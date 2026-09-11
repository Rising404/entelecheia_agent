"""L0、L1 和 L2 共用的冷 Turn 生命周期契约。

这些值描述持久化 Turn 边界，而无需导入 Entry 编排、L1/L2、
model provider 或 Session 实现。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from personagraph.entry_execution_snapshot import (
    EntryExecutionSnapshot,
    EntryFeatureScalar,
)


ProcessingLevel = Literal["L0", "L1", "L2"]
RoutingPolicySource = Literal[
    "default",
    "session_default",
    "request_override",
]


@dataclass(frozen=True, slots=True)
class TurnRoutingPolicy:
    """一个 Turn 的用户控制 processing-level 开关。"""

    schema_version: int = 1
    l1_enabled: bool = True
    l2_enabled: bool = False

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("routing-policy schema_version must be 1")
        if type(self.l1_enabled) is not bool or type(self.l2_enabled) is not bool:
            raise ValueError("routing-policy switches must be booleans")
        if self.l1_enabled and self.l2_enabled:
            raise ValueError("l1_enabled and l2_enabled are mutually exclusive")

    @property
    def allowed_processing_levels(self) -> tuple[ProcessingLevel, ...]:
        levels: list[ProcessingLevel] = ["L0"]
        if self.l1_enabled:
            levels.append("L1")
        if self.l2_enabled:
            levels.append("L2")
        return tuple(levels)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "l1_enabled": self.l1_enabled,
            "l2_enabled": self.l2_enabled,
        }


@dataclass(frozen=True, slots=True)
class TurnRoutingPolicySnapshot:
    """随一个 AcceptedTurn 冻结的不可变路由权威状态。"""

    source: RoutingPolicySource
    policy: TurnRoutingPolicy
    allowed_processing_levels: tuple[ProcessingLevel, ...]
    schema_version: int = 1

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("routing-policy snapshot schema_version must be 1")
        if self.source not in {
            "default",
            "session_default",
            "request_override",
        }:
            raise ValueError("invalid routing-policy snapshot source")
        if not isinstance(self.policy, TurnRoutingPolicy):
            raise ValueError("routing-policy snapshot requires the current policy")
        if self.allowed_processing_levels != self.policy.allowed_processing_levels:
            raise ValueError("allowed_processing_levels does not match policy switches")

    def allows(self, level: ProcessingLevel) -> bool:
        return level in self.allowed_processing_levels

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "source": self.source,
            "policy": self.policy.to_dict(),
            "allowed_processing_levels": list(self.allowed_processing_levels),
        }


DEFAULT_TURN_ROUTING_POLICY_SNAPSHOT = TurnRoutingPolicySnapshot(
    source="default",
    policy=TurnRoutingPolicy(l1_enabled=True, l2_enabled=False),
    allowed_processing_levels=("L0", "L1"),
)


EntryTurnStatus = Literal["running", "completed", "incomplete"]
EntryWindowState = Literal["empty", "active", "post_commit_pending", "interrupted"]


class ProcessingRouteAuthorityError(RuntimeError):
    """A processing lane could not authenticate its durable routing facts."""


@dataclass(frozen=True)
class EntryTurnResult:
    """权威 Runtime entry 路径的公开结果契约。

    Turn 确认一次交互，而非 Task 或 WorkRun 完成。字段有意只包含公开交付与
    execution-window 事实；私有 trace、模型诊断和旧版批准载荷不会跨越此边界。
    """

    session_id: str
    turn_id: str
    status: EntryTurnStatus
    processing_level: ProcessingLevel | None
    window_state: EntryWindowState
    window_revision: int
    reply: str | None = None
    end_reason: str | None = None
    error_code: str | None = None
    related_insession_task_ids: tuple[str, ...] = ()
    work_run_ids: tuple[str, ...] = ()
    available_controls: tuple[str, ...] = ()
    delivery: dict[str, Any] | None = None
    pending_decision: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.session_id or not self.turn_id:
            raise ValueError("EntryTurnResult requires session_id and turn_id")
        if self.window_revision < 0:
            raise ValueError("window_revision must be non-negative")
        if self.status == "completed" and self.reply is None:
            raise ValueError("completed EntryTurnResult requires a formal reply")
        if self.status != "completed" and self.reply is not None:
            raise ValueError("only completed EntryTurnResult may carry a reply")


@dataclass(frozen=True)
class EntryRecoveryProjection:
    """关于此 Session 最近中断 Turn 的安全紧凑事实。"""

    turn_id: str
    end_reason: str
    error_code: str | None
    input_message_id: str


@dataclass(frozen=True)
class AcceptedEntryTurn:
    """一个已准备好至多一次 Runtime 执行的持久化已接受 Turn。"""

    session_id: str
    turn_id: str
    client_request_id: str
    user_input: str
    attachment_ids: tuple[str, ...]
    window_revision: int
    replayed: bool
    execution_snapshot: EntryExecutionSnapshot
    turn_status: EntryTurnStatus = "running"
    processing_level: ProcessingLevel | None = None
    end_reason: str | None = None
    error_code: str | None = None
    window_state: EntryWindowState = "active"
    recovery_projection: EntryRecoveryProjection | None = None
    input_message_id: str | None = None
    routing_policy: TurnRoutingPolicySnapshot = (
        DEFAULT_TURN_ROUTING_POLICY_SNAPSHOT
    )

    def __post_init__(self) -> None:
        if not self.session_id or not self.turn_id or not self.client_request_id:
            raise ValueError("AcceptedEntryTurn requires stable Session, Turn, and request ids")
        if self.window_revision < 0:
            raise ValueError("window_revision must be non-negative")
        if self.input_message_id is not None and not self.input_message_id.strip():
            raise ValueError("input_message_id must be non-empty when provided")
        if not isinstance(
            self.execution_snapshot,
            EntryExecutionSnapshot,
        ):
            raise ValueError("execution_snapshot must use the current contract")


__all__ = [
    "AcceptedEntryTurn",
    "EntryRecoveryProjection",
    "EntryExecutionSnapshot",
    "EntryFeatureScalar",
    "EntryTurnResult",
    "EntryTurnStatus",
    "EntryWindowState",
    "ProcessingRouteAuthorityError",
    "DEFAULT_TURN_ROUTING_POLICY_SNAPSHOT",
    "ProcessingLevel",
    "RoutingPolicySource",
    "TurnRoutingPolicySnapshot",
    "TurnRoutingPolicy",
]
