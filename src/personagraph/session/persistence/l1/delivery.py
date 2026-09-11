"""在调用方已有事务中核验 L1 运行通知，不修改失败状态或另行发布消息。"""

from __future__ import annotations

import sqlite3

from ....persistent_turn_content.delivery import (
    L1TerminalNotification,
    build_l1_terminal_notification,
)
from ...turn_execution_contracts import TurnExecutionPersistenceError


def require_terminal_notification(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    turn_id: str,
    failure_code: str,
    assistant_content: str,
    bound_run_id: str | None = None,
) -> L1TerminalNotification:
    """只允许既有失败 run 的固定运行说明，不能借通知通道发布任意候选正文。

    首次提交还须匹配当前 Window 绑定；幂等重放可发生在 Window 已释放之后，届时
    用不可变 Session/Turn 归属查询唯一 run。失败码保持原始值，公开分类由内容合同提供。
    """

    notification = build_l1_terminal_notification(failure_code)
    if notification is None or notification.reply != assistant_content:
        raise TurnExecutionPersistenceError("L1 terminal notification content is not authorized")
    rows = conn.execute(
        "SELECT r.l1_turn_run_id, r.status, s.stage, s.failure_code "
        "FROM l1_turn_runs AS r JOIN l1_turn_run_states AS s "
        "ON s.l1_turn_run_id=r.l1_turn_run_id "
        "WHERE r.session_id=? AND r.turn_id=?",
        (session_id, turn_id),
    ).fetchall()
    if len(rows) != 1:
        raise TurnExecutionPersistenceError("L1 terminal notification has no unique persisted run")
    run = rows[0]
    if (
        run["status"] != "failed"
        or run["stage"] != "failed"
        or run["failure_code"] != notification.failure_code
        or (bound_run_id is not None and run["l1_turn_run_id"] != bound_run_id)
    ):
        raise TurnExecutionPersistenceError("L1 terminal notification crossed failed-run authority")
    return notification
