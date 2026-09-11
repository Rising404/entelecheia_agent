"""对已接受 Entry Turn 执行对账，而不重新运行 Runtime 工作。

此 controller 负责客户端请求重放的读取侧决策树：投影已完成 Turn，识别两种范围
严格受限的结算恢复，并从持久化事实重建不完整 L2 引用。Entry 有意保留全部变更
权威。L2 receipt/manifest 读取由 L2 adapter 按需提供；结算回调仍绑定 Entry 的
finalizer、CAS 规则、租约处理和事件 policy，因此本模块绝不直接调用 Store 写入。
"""

from __future__ import annotations

from typing import Any, Protocol

from ...turn.contracts import AcceptedEntryTurn, EntryTurnResult
from ...turn_events import RuntimeStage


class EntryReplayReadPort(Protocol):
    """投影或对账一次 request-id 重放所需的持久化事实。"""

    def get_committed_turn_pair(
        self,
        session_id: str,
        run_id: str,
    ) -> dict[str, Any] | None: ...

    def inspect_turn_execution(self, session_id: str) -> dict[str, object]: ...

    def list_turn_insession_task_ids(
        self,
        session_id: str,
        turn_id: str,
    ) -> tuple[str, ...]: ...

    def list_turn_linked_work_run_ids(
        self,
        *,
        session_id: str,
        turn_id: str,
    ) -> tuple[str, ...]: ...


class EntryReplayAuthoritativeSettlementReconciler(Protocol):
    """供完全结算的权威 lane 集合使用、由 Entry 持有的写入回调。"""

    def __call__(
        self,
        *,
        revision: int,
        related_insession_task_ids: tuple[str, ...],
    ) -> EntryTurnResult | None: ...


def reconcile_replayed_entry_turn(
    *,
    accepted: AcceptedEntryTurn,
    store: EntryReplayReadPort,
    reconcile_authoritative_settlement: EntryReplayAuthoritativeSettlementReconciler,
    task_execution_lane_store: object | None = None,
) -> EntryTurnResult:
    """返回已接受 request-id 重放的权威结果。

    controller 绝不启动 provider 或工具路径。只有相应持久化 receipt 证明工作已在
    响应丢失前完成后，它才可请求 Entry 重新进入一个现有 Store finalizer。
    """

    # 已确定的处理层级优先；尚未分类时才用该 Turn 冻结的策略识别 L1 模式。
    # 真实 L2 历史执行仍须由其持久化 manifest 对账，不能被策略开关覆盖。
    l1_replay = accepted.processing_level == "L1" or (
        accepted.processing_level is None and accepted.routing_policy.allows("L1")
    )
    if accepted.turn_status == "completed":
        committed = store.get_committed_turn_pair(
            accepted.session_id,
            f"commit_{accepted.turn_id}",
        )
        if committed is None:
            raise RuntimeError("completed Turn is missing its formal delivery")
        return EntryTurnResult(
            session_id=accepted.session_id,
            turn_id=accepted.turn_id,
            status="completed",
            processing_level=accepted.processing_level,
            reply=str(committed["assistant_content"]),
            error_code=accepted.error_code,
            end_reason=accepted.end_reason,
            related_insession_task_ids=(
                () if l1_replay else store.list_turn_insession_task_ids(
                    accepted.session_id,
                    accepted.turn_id,
                )
            ),
            work_run_ids=(
                () if l1_replay else store.list_turn_linked_work_run_ids(
                    session_id=accepted.session_id,
                    turn_id=accepted.turn_id,
                )
            ),
            window_state=accepted.window_state,
            window_revision=accepted.window_revision,
        )

    if accepted.turn_status == "running" and accepted.window_state == "interrupted":
        try:
            inspected = store.inspect_turn_execution(accepted.session_id)
            interrupted_window = inspected.get("window")
        except Exception:
            interrupted_window = None
        if (
            isinstance(interrupted_window, dict)
            and str(interrupted_window.get("turn_id") or "") == accepted.turn_id
            and str(interrupted_window.get("window_state") or "") == "interrupted"
            and str(interrupted_window.get("interruption_reason") or "")
            == "CONTEXT_BUDGET_EXCEEDED"
        ):
            return EntryTurnResult(
                session_id=accepted.session_id,
                turn_id=accepted.turn_id,
                status="incomplete",
                processing_level=accepted.processing_level,
                end_reason="context_budget_exceeded",
                error_code="CONTEXT_BUDGET_EXCEEDED",
                window_state="interrupted",
                window_revision=int(interrupted_window.get("state_version") or 0),
            )

    if accepted.turn_status == "running" and l1_replay:
        try:
            inspected = store.inspect_turn_execution(accepted.session_id)
            l1_window = inspected.get("window")
        except Exception:
            l1_window = None
        if (
            isinstance(l1_window, dict)
            and str(l1_window.get("turn_id") or "") == accepted.turn_id
            and str(l1_window.get("window_state") or "") == "interrupted"
            and l1_window.get("current_l1_turn_run_id") is not None
        ):
            reason = str(l1_window.get("interruption_reason") or "")
            end_reason = None
            if reason == "TOOL_COMPLETION_UNCONFIRMED":
                # 未确认副作用只能投影持久中断；request-id 重放不能重新派发工具。
                end_reason = "tool_completion_unconfirmed"
            elif (
                reason == "L1_RUNTIME_NOT_READY"
                and str(l1_window.get("stage") or "") == RuntimeStage.L1_BOOTSTRAP.value
            ):
                end_reason = "l1_runtime_not_ready"
            if end_reason is not None:
                return EntryTurnResult(
                    session_id=accepted.session_id,
                    turn_id=accepted.turn_id,
                    status="incomplete",
                    processing_level="L1",
                    end_reason=end_reason,
                    error_code=reason,
                    window_state="interrupted",
                    window_revision=int(l1_window.get("state_version") or 0),
                )

    if accepted.turn_status == "running" and not l1_replay:
        # 恢复由持久化 lane manifest 与所属 Window 选择，而非由当前 rollout 标志
        # 选择。重试可能在重启或配置回滚后到达，但仍必须投影原始已接受执行请求产生的
        # 精确已结算事实。
        from ....l2.entry_adapter.replay import (
            inspect_task_execution_lane_manifest,
        )

        manifest_found, manifest = inspect_task_execution_lane_manifest(
            replay_store=store,
            task_execution_lane_store=task_execution_lane_store,
            session_id=accepted.session_id,
            turn_id=accepted.turn_id,
        )
        if manifest_found:
            try:
                inspected = store.inspect_turn_execution(accepted.session_id)
                window = inspected.get("window")
                window_state = (
                    str(window.get("window_state") or "")
                    if isinstance(window, dict)
                    else ""
                )
                stage = (
                    str(window.get("stage") or "") if isinstance(window, dict) else ""
                )
                may_settle = (
                    bool(getattr(manifest, "lanes", ()))
                    and isinstance(window, dict)
                    and str(window.get("turn_id") or "") == accepted.turn_id
                    and (
                        (
                            window_state == "active"
                            and stage
                            in {
                                RuntimeStage.L2_PLAN.value,
                                RuntimeStage.PERSIST.value,
                            }
                        )
                        or window_state == "interrupted"
                    )
                    and window.get("current_work_run_id") is None
                    and window.get("current_attempt_id") is None
                )
                revision = int(window.get("state_version") or 0) if may_settle else 0
            except Exception:
                may_settle = False
                revision = 0
        else:
            may_settle = False
            revision = 0
        if may_settle and revision >= 1:
            recovered = reconcile_authoritative_settlement(
                revision=revision,
                related_insession_task_ids=store.list_turn_insession_task_ids(
                    accepted.session_id,
                    accepted.turn_id,
                ),
            )
            if recovered is not None:
                return recovered

    replay_processing_level = accepted.processing_level
    replay_related_task_ids: tuple[str, ...] = ()
    replay_work_run_ids: tuple[str, ...] = ()
    if accepted.turn_status == "incomplete" and not l1_replay:
        from ....l2.entry_adapter.replay import (
            inspect_task_execution_lane_manifest,
        )

        manifest_found, _manifest = inspect_task_execution_lane_manifest(
            replay_store=store,
            task_execution_lane_store=task_execution_lane_store,
            session_id=accepted.session_id,
            turn_id=accepted.turn_id,
        )
        if manifest_found:
            # 即使 Session Window 已推进，已验证 lane manifest 仍是该 Turn 属于
            # L2 的持久化权威。缺少此 receipt 的 incomplete Turn 会保留其持久化
            # 元数据，而不会被猜测成 L2。
            replay_processing_level = "L2"
            replay_related_task_ids = store.list_turn_insession_task_ids(
                accepted.session_id,
                accepted.turn_id,
            )
            replay_work_run_ids = store.list_turn_linked_work_run_ids(
                session_id=accepted.session_id,
                turn_id=accepted.turn_id,
            )

    return EntryTurnResult(
        session_id=accepted.session_id,
        turn_id=accepted.turn_id,
        status=accepted.turn_status,
        processing_level=replay_processing_level,
        end_reason=accepted.end_reason,
        error_code=accepted.error_code,
        related_insession_task_ids=replay_related_task_ids,
        work_run_ids=replay_work_run_ids,
        # Store 在 request-id 重放时返回当前 Session Window。此 schema 中不存在历史
        # Window 快照，因此保留该契约，而不伪造旧的中断投影。
        window_state=accepted.window_state,
        window_revision=accepted.window_revision,
    )


__all__ = [
    "EntryReplayAuthoritativeSettlementReconciler",
    "EntryReplayReadPort",
    "reconcile_replayed_entry_turn",
]
