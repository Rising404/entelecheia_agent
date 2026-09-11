"""为一个已接受 Runtime Turn 执行有界只读上下文组装。

本模块将可信的 accepted-Turn 事实转成模型输入上下文。Entry 保留生命周期权威：
Window 变更、事件、租约归属、路由、模型调用、最终化与中断均位于此边界之外。
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from ....context_budget.token_counter import estimate_tokens
from .attachments import AttachmentProjection
from .attachments import (
    attachment_projection_token_cost,
    build_on_demand_attachment_projection,
)
from ....session.session_summary import SessionSummaryStatus
from .contracts import EntryContext
from .ports import EntryContextAssemblyStorePort
from ..ingress.policy import (
    ENTRY_MAX_HISTORY_PAIRS,
    build_entry_runtime_snapshot,
    entry_capability_ceiling,
    entry_history_budget,
    entry_task_catalog_budget,
    select_history_within_budget,
    select_session_summary,
)
from .task_catalog_projection import project_entry_task_catalog_items
from ..ingress.contracts import AttachmentRef, TrustedTurnEnvelope
from .task_catalog import (
    EntryTaskCatalog,
    estimate_insession_task_catalog_tokens,
    pack_insession_task_catalog,
)
from ...turn.contracts import AcceptedEntryTurn
from ...turn_deadline import TurnDeadline, TurnDeadlineExceeded


class AttachmentProjectionBuildError(RuntimeError):
    """无法以权威方式投影已接受附件绑定。"""


def build_entry_context(
    *,
    accepted: AcceptedEntryTurn,
    features: dict[str, Any],
    store: EntryContextAssemblyStorePort,
    deadline: TurnDeadline | None = None,
) -> EntryContext:
    """构建可信且受预算约束的模型上下文，而不改变持久化状态。

    ``deadline`` 由 Entry 生命周期持有；本层只在同步组装步骤之间观察同一个 Turn
    的墙钟预算，既不创建也不重置预算。它不能中断正在执行的 Store 查询，但能阻止
    超时后的后续上下文工作和 provider effect。

    阅读上下文时区分来源：用户原文来自 accepted Turn；附件仅投影可信元数据，
    正文留给按需工具；历史只选已提交的完整对话对；summary 仅在 OK 状态下使用。
    history 与 task catalog 都受剩余预算裁剪，这个 EntryContext 也不是最终 Provider
    wire payload：classifier 和 L1 会继续各自选字段，网关还要做精确请求准入。
    """

    _require_turn_time(deadline)
    user_text_tokens = estimate_tokens(accepted.user_input)
    context_hard_limit = int(features.get("context_guard_limit", 24000))
    attachments = build_turn_attachments(
        session_id=accepted.session_id,
        turn_id=accepted.turn_id,
        store=store,
        deadline=deadline,
    )
    _require_turn_time(deadline)
    envelope = TrustedTurnEnvelope(
        turn_id=accepted.turn_id,
        session_id=accepted.session_id,
        received_at=datetime.now(timezone.utc),
        input_kind="user_text",
        user_text=accepted.user_input,
        attachments=tuple(
            AttachmentRef(
                attachment_id=item.attachment_id,
                media_type=item.media_type,
            )
            for item in attachments.items
        ),
    )
    attachment_tokens = attachment_projection_token_cost(attachments)
    try:
        summary_state = store.get_session_summary_state(accepted.session_id)
    except Exception:
        # 损坏或缺失的派生 summary 不得阻塞有效 Turn。
        summary_state = None
        summary_status: Literal["empty", "ok", "stale", "unavailable"] = "unavailable"
    else:
        summary_status = "empty" if summary_state is None else summary_state.status.value
    summary = select_session_summary(
        (
            summary_state.running_summary.strip()
            if summary_state is not None and summary_state.status is SessionSummaryStatus.OK
            else None
        ),
        context_hard_limit=context_hard_limit,
        user_text_tokens=user_text_tokens,
    )
    history_pairs = select_history_within_budget(
        store.list_committed_turn_pairs(accepted.session_id, limit=ENTRY_MAX_HISTORY_PAIRS),
        budget_tokens=entry_history_budget(
            context_hard_limit=context_hard_limit,
            user_text_tokens=user_text_tokens,
            summary_tokens=estimate_tokens(summary) if summary else 0,
            attachment_tokens=attachment_tokens,
        ),
    )
    history_tokens = sum(
        estimate_tokens(pair["user"]) + estimate_tokens(pair["assistant"])
        for pair in history_pairs
    )
    # L1 模式只处理当前 Turn，不读取长期任务目录或任务等待问题。
    task_catalog = EntryTaskCatalog()
    if not accepted.routing_policy.allows("L1"):
        catalog_items = store.list_insession_task_catalog(accepted.session_id)
        pending_user_questions = store.list_pending_user_questions(
            session_id=accepted.session_id
        )
        ordered_catalog_items = project_entry_task_catalog_items(
            catalog_items=catalog_items,
            pending_user_questions=pending_user_questions,
        )
        task_catalog = pack_insession_task_catalog(
            ordered_catalog_items,
            token_budget=entry_task_catalog_budget(
                context_hard_limit=context_hard_limit,
                user_text_tokens=user_text_tokens,
                summary_tokens=estimate_tokens(summary) if summary else 0,
                attachment_tokens=attachment_tokens,
                history_tokens=history_tokens,
            ),
        )
    context = EntryContext(
        envelope=envelope,
        snapshot=build_entry_runtime_snapshot(accepted.session_id),
        ceiling=entry_capability_ceiling(features),
        estimated_input_tokens=(
            user_text_tokens
            + (estimate_tokens(summary) if summary else 0)
            + attachment_tokens
            + history_tokens
            + estimate_insession_task_catalog_tokens(task_catalog)
        ),
        history_pairs=history_pairs,
        session_summary=summary,
        session_summary_status=summary_status,
        attachments=attachments,
        recovery_projection=accepted.recovery_projection,
        task_catalog=task_catalog,
        routing_policy=accepted.routing_policy,
    )
    _require_turn_time(deadline)
    return context


def build_turn_attachments(
    *,
    session_id: str,
    turn_id: str,
    store: EntryContextAssemblyStorePort,
    deadline: TurnDeadline | None = None,
) -> AttachmentProjection:
    """构建不读取附件内容、仅含可信元数据的附件投影。

    deadline 检查是同步 Store/投影步骤前后的协作式停止点，不是附件领域参数，
    也不会取消已经开始的数据库调用。
    """

    _require_turn_time(deadline)
    try:
        records = store.list_turn_attachments(session_id, turn_id)
    except Exception as exc:
        # 空回退会抹去已接受文件，同时允许模型像未提供文件一样作答。
        raise AttachmentProjectionBuildError(
            "failed to load the accepted attachment bindings"
        ) from exc
    _require_turn_time(deadline)
    if not records:
        return AttachmentProjection()
    projection = build_on_demand_attachment_projection(records)
    _require_turn_time(deadline)
    return projection


def _require_turn_time(deadline: TurnDeadline | None) -> None:
    """在开始另一个同步步骤前抛出公开的有类型停止。"""

    if deadline is not None and deadline.expired():
        raise TurnDeadlineExceeded()


__all__ = [
    "AttachmentProjectionBuildError",
    "build_entry_context",
    "build_turn_attachments",
]
