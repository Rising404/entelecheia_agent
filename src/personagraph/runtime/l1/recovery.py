"""用于未完成 L1 TurnRun 的精确恢复租约。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import json
from threading import Event, Thread
from typing import Any, Callable

from ...configuration.features import DEFAULT_TURN_WALL_CLOCK_BUDGET_S
from ...persistent_turn_content.delivery import (
    L1TerminalNotification,
    build_l1_terminal_notification,
)
from ...tools.contracts import ExecutionStatus
from ...tools.files.file_tools import (
    PREPARE_FILES_TOOL_ID, derive_file_preparation_tool_request_id, file_input_schema,
)
from ...tools.schema_validation import ToolSchemaCompiler
from ...workspace.ingestion.state import has_file_preparation_request
from ..turn.contracts import AcceptedEntryTurn
from ..turn.timing import ACTIVE_TURN_WINDOW_HEARTBEAT_TTL_S
from ..turn_deadline import TurnDeadline
from ..turn_events import TurnEvent, project_turn_event
from .identity import sha256_json
from .ports import L1ReplayRecoveryStore, L1StorePort


class L1ReplayRecoveryError(RuntimeError):
    """Store 已应用 L1 声明，却返回了无效的恢复回执。"""


def can_resume_pending_file_preparations(
    *, accepted: AcceptedEntryTurn, l1_turn_run_id: str, store: L1StorePort,
) -> bool:
    """只放行全部已绑定持久文档请求的 pending prepare，不重放未知副作用。

    原 deadline/租约仍由恢复 owner 掌管，原来源由工具内的 request 绑定复验。
    图片、入队前中断、混合批次或无法证明的状态都保留原来的保守停止行为。
    """
    try:
        execution = store.get_l1_turn_execution(
            session_id=accepted.session_id, turn_id=accepted.turn_id,
        )
        if not isinstance(execution, dict):
            return False
        run = execution.get("run")
        if not isinstance(run, dict) or (
            run.get("session_id"), run.get("turn_id"), run.get("l1_turn_run_id")
        ) != (accepted.session_id, accepted.turn_id, l1_turn_run_id):
            return False
        calls = execution.get("tool_calls")
        if not isinstance(calls, list) or not calls:
            return False
        terminal_statuses = {item.value for item in ExecutionStatus} - {
            ExecutionStatus.COMPLETION_UNCONFIRMED.value,
        }
        pending = []
        for call in calls:
            if not isinstance(call, dict):
                return False
            if call.get("status") == "pending":
                pending.append(call)
            elif call.get("status") not in terminal_statuses:
                return False
        if not pending:
            return False
        for call in pending:
            if not can_resume_file_preparation_call(
                session_id=accepted.session_id, tool_call=call,
            ):
                return False
        return True
    except Exception:
        # 恢复资格证明失败时不猜测，不初始化/修复任何外部状态。
        return False


def can_resume_file_preparation_call(
    *, session_id: str, tool_call: Mapping[str, object],
) -> bool:
    """判断一次已派发文档准备是否可继续观察原请求；不授权任何新的物理效果。"""
    try:
        if (
            tool_call.get("session_id") != session_id
            or tool_call.get("tool_id") != PREPARE_FILES_TOOL_ID
            or tool_call.get("status") != "pending"
            or tool_call.get("execution_class") != "protected_effect"
            or tool_call.get("protected_phase") != "dispatching"
            or tool_call.get("outcome_json") is not None
        ):
            return False
        arguments = json.loads(tool_call["arguments_json"])
        ToolSchemaCompiler().compile(file_input_schema(), role="input").validate(arguments)
        if sha256_json(arguments) != tool_call.get("arguments_hash"):
            return False
        return all(
            has_file_preparation_request(
                session_id=session_id,
                request_id=derive_file_preparation_tool_request_id(tool_call["tool_call_id"], index),
            )
            for index in range(len(arguments["files"]))
        )
    except Exception:
        return False


@dataclass(slots=True)
class ClaimedL1Replay:
    """一个围栏 L1 重放租约及其原始剩余 Turn 预算。"""

    accepted: AcceptedEntryTurn
    lease_owner: str
    l1_turn_run_id: str
    window_revision: int
    deadline: TurnDeadline
    has_unconfirmed_tool_call: bool
    execution_config_json: object
    execution_config_hash: object
    store: L1ReplayRecoveryStore
    on_stream_event: Callable[[dict[str, Any]], None] | None
    terminal_notification: L1TerminalNotification | None = None
    heartbeat_interval_s: float = 30.0
    heartbeat_started: bool = False
    _heartbeat_stop: Event = field(default_factory=Event, init=False, repr=False)
    _lease_lost: Event = field(default_factory=Event, init=False, repr=False)
    _heartbeat_thread: Thread | None = field(default=None, init=False, repr=False)

    def start(self) -> bool:
        thread = Thread(
            target=self._heartbeat,
            name=f"l1-replay-heartbeat-{self.accepted.session_id[:24]}",
            daemon=True,
        )
        try:
            thread.start()
        except RuntimeError:
            return False
        self._heartbeat_thread = thread
        self.heartbeat_started = True
        return True

    def close(self) -> None:
        self._heartbeat_stop.set()
        if self._heartbeat_thread is not None:
            self._heartbeat_thread.join(timeout=1.0)

    def lease_is_current(self) -> bool:
        if self._lease_lost.is_set():
            return False
        try:
            current = self.store.renew_l1_turn_run_resume_lease(
                session_id=self.accepted.session_id,
                turn_id=self.accepted.turn_id,
                l1_turn_run_id=self.l1_turn_run_id,
                lease_owner=self.lease_owner,
            )
        except Exception:
            current = False
        if not current:
            self._lease_lost.set()
        return current

    def emit(self, event: TurnEvent) -> int | None:
        try:
            sequence = self.store.append_runtime_turn_event(
                event,
                active_window_lease_owner=self.lease_owner,
            )
        except Exception:
            return None
        if self.on_stream_event is not None:
            try:
                self.on_stream_event({
                    "event": "runtime_event",
                    "runtime_event": project_turn_event(
                        event,
                        sequence=sequence,
                    ).model_dump(mode="json"),
                })
            except OSError:
                pass
        return sequence

    def _heartbeat(self) -> None:
        from ...session import store as session_store

        with session_store.session_database_scope(self.accepted.session_id):
            while not self._heartbeat_stop.wait(self.heartbeat_interval_s):
                if not self.lease_is_current():
                    return


def claim_replayed_l1_turn(
    *,
    accepted: AcceptedEntryTurn,
    features: dict[str, Any],
    lease_owner: str,
    on_stream_event: Callable[[dict[str, Any]], None] | None,
    store: L1ReplayRecoveryStore,
) -> ClaimedL1Replay | None:
    """为原 accepted Turn 认领精确 L1 恢复租约；不合格、忙碌或认领失败返回 None。

    Store 是能否恢复的唯一裁定者。成功后携带原 run/config/window revision 并启动
    heartbeat；调用方用完必须 close。此操作不创建新用户 Turn，也不重置执行预算。
    """

    if accepted.turn_status != "running" or not accepted.routing_policy.allows("L1"):
        return None
    try:
        claimed = store.claim_l1_turn_run_resume(
            session_id=accepted.session_id,
            turn_id=accepted.turn_id,
            lease_owner=lease_owner,
            lease_seconds=max(1, int(ACTIVE_TURN_WINDOW_HEARTBEAT_TTL_S)),
        )
    except Exception:
        return None
    if str(claimed.get("status") or "") != "applied":
        return None
    window = _required_mapping(claimed, "window")
    run = _required_mapping(claimed, "run")
    state = _required_mapping(claimed, "state")
    terminal_failure_code = claimed.get("terminal_failure_code")
    terminal_notification = (
        build_l1_terminal_notification(terminal_failure_code)
        if isinstance(terminal_failure_code, str) else None
    )
    if terminal_failure_code is not None and terminal_notification is None:
        raise L1ReplayRecoveryError("L1 delivery recovery has an unsupported terminal reason")
    recovery = ClaimedL1Replay(
        accepted=accepted,
        lease_owner=lease_owner,
        l1_turn_run_id=str(run["l1_turn_run_id"]),
        window_revision=int(window["state_version"]),
        deadline=_deadline_for_l1_resume(claimed=claimed, features=features),
        has_unconfirmed_tool_call=claimed.get("has_unconfirmed_tool_call") is True,
        execution_config_json=state.get("execution_config_json"),
        execution_config_hash=state.get("execution_config_hash"),
        store=store,
        on_stream_event=on_stream_event,
        terminal_notification=terminal_notification,
    )
    recovery.start()
    return recovery


def _deadline_for_l1_resume(
    *,
    claimed: dict[str, object],
    features: dict[str, Any],
) -> TurnDeadline:
    """把持久绝对 deadline 转成当前进程的剩余单调时钟预算。

    缺少 deadline 时从原 received_at 推算，绝不从恢复时刻赠送完整时长；
    无时区、坏时间或无法解析的状态都折算为零剩余预算（fail closed）。
    """

    state = claimed.get("state")
    deadline_at = state.get("deadline_at") if isinstance(state, dict) else None
    try:
        if isinstance(deadline_at, str):
            expires_at = datetime.fromisoformat(deadline_at)
        else:
            turn = _required_mapping(claimed, "turn")
            received_at = datetime.fromisoformat(str(turn["received_at"]))
            budget_s = float(
                features.get(
                    "turn_wall_clock_budget_s",
                    DEFAULT_TURN_WALL_CLOCK_BUDGET_S,
                )
            )
            expires_at = received_at + timedelta(seconds=max(0.0, budget_s))
        if expires_at.tzinfo is None or expires_at.utcoffset() is None:
            raise ValueError("L1 replay deadline must be timezone-aware")
        remaining_s = (
            expires_at.astimezone(timezone.utc) - datetime.now(timezone.utc)
        ).total_seconds()
    except (KeyError, TypeError, ValueError, OverflowError):
        remaining_s = 0.0
    return TurnDeadline.starting_now(remaining_s)


def _required_mapping(
    value: dict[str, object],
    key: str,
) -> dict[str, object]:
    item = value.get(key)
    if not isinstance(item, dict):
        raise L1ReplayRecoveryError(f"L1 recovery receipt omitted {key}")
    return item


__all__ = [
    'ClaimedL1Replay',
    "L1ReplayRecoveryError",
    'L1ReplayRecoveryStore',
    "claim_replayed_l1_turn",
]
