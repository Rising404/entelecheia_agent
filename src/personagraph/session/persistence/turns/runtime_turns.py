"""新 Runtime 受信 Turn 信封与生命周期事实的持久化。"""

from __future__ import annotations

import sqlite3
from typing import Literal

from ...turn_execution_contracts import TurnExecutionLeaseConflict, TurnExecutionPersistenceError
from ..deps import StoreDeps
from ....runtime.turn_events import TurnEvent


RuntimeTurnStatus = Literal["running", "completed", "failed", "rejected"]
ProcessingLevelValue = Literal["L0", "L1", "L2"]
MAX_RUNTIME_TURN_EVENT_PAGE_SIZE = 500

_RUNTIME_TURN_EVENT_COLUMNS = (
    "event_id",
    "session_id",
    "turn_id",
    "parent_event_id",
    "insession_task_id",
    "insession_task_node_id",
    "work_run_id",
    "attempt_id",
    "stage",
    "status",
    "occurred_at",
    "duration_ms",
    "model_call_id",
    "model_attempt",
    "operation_id",
    "error_code",
    "retryable",
    "diagnostic_ref",
)
_RUNTIME_TURN_EVENT_SELECT = "rowid AS sequence, " + ", ".join(_RUNTIME_TURN_EVENT_COLUMNS)


class UnknownRuntimeTurnEventCursor(ValueError):
    """请求的事件游标不在当前 Session 中。"""


def create_runtime_turn(
    deps: StoreDeps,
    *,
    turn_id: str,
    session_id: str,
    source: str,
    user_text: str,
) -> dict[str, object]:
    """在任何模型请求开始前持久化一个已接受受信输入。"""

    if not turn_id.strip() or not session_id.strip():
        raise ValueError("turn_id and session_id must not be empty")
    deps.init_db()
    now = deps.now()
    with deps.connect() as conn:
        session = conn.execute("SELECT 1 FROM sessions WHERE id=?", (session_id,)).fetchone()
        if session is None:
            raise ValueError(f"unknown session: {session_id}")
        existing = conn.execute(
            "SELECT turn_id, session_id, source, user_text, status, processing_level, "
            "error_code, received_at, completed_at FROM runtime_turns WHERE turn_id=?",
            (turn_id,),
        ).fetchone()
        if existing is not None:
            if str(existing["session_id"]) != session_id:
                raise ValueError("runtime turn id collision across sessions")
            return dict(existing)
        conn.execute(
            "INSERT INTO runtime_turns "
            "(turn_id, session_id, source, user_text, status, received_at) "
            "VALUES (?, ?, ?, ?, 'running', ?)",
            (turn_id, session_id, source, user_text, now),
        )
        conn.execute("UPDATE sessions SET last_active_at=? WHERE id=?", (now, session_id))
    return {
        "turn_id": turn_id,
        "session_id": session_id,
        "source": source,
        "user_text": user_text,
        "status": "running",
        "processing_level": None,
        "error_code": None,
        "received_at": now,
        "completed_at": None,
    }


def complete_runtime_turn(
    deps: StoreDeps,
    *,
    turn_id: str,
    status: RuntimeTurnStatus,
    processing_level: ProcessingLevelValue | None,
    error_code: str | None = None,
) -> dict[str, object]:
    """完成先前持久化的 Turn，不写入对话消息。"""

    if status == "completed" and processing_level is None:
        raise ValueError("completed runtime turn requires processing_level")
    if status != "completed" and error_code is None:
        raise ValueError("failed or rejected runtime turn requires error_code")
    deps.init_db()
    now = deps.now()
    with deps.connect() as conn:
        existing = conn.execute(
            "SELECT session_id, status, processing_level, error_code, received_at, completed_at "
            "FROM runtime_turns WHERE turn_id=?",
            (turn_id,),
        ).fetchone()
        if existing is None:
            raise ValueError(f"unknown runtime turn: {turn_id}")
        if str(existing["status"]) != "running":
            return {"turn_id": turn_id, **dict(existing)}
        conn.execute(
            "UPDATE runtime_turns SET status=?, processing_level=?, error_code=?, completed_at=? WHERE turn_id=?",
            (status, processing_level, error_code, now, turn_id),
        )
        row = conn.execute(
            "SELECT turn_id, session_id, source, user_text, status, processing_level, "
            "error_code, received_at, completed_at FROM runtime_turns WHERE turn_id=?",
            (turn_id,),
        ).fetchone()
    return dict(row)


def list_pending_carry_over_turns(
    deps: StoreDeps,
    session_id: str,
    *,
    error_codes: tuple[str, ...],
    limit: int = 16,
) -> list[dict[str, object]]:
    """读取仍待向前合并、已接受但未回答的输入。

    只有输入已接受且从未回答的失败才符合条件，因此调用方传入封闭错误码集合，而不是让
    此查询决定策略。被拒输入刻意不结转：重新合并超限 Turn 只会产生更大的超限 Turn。
    """

    if not error_codes or limit <= 0:
        return []
    deps.init_db()
    placeholders = ",".join("?" for _ in error_codes)
    with deps.connect() as conn:
        rows = conn.execute(
    # 向前结转的是失败 Turn 实际运行时携带的全部内容；对于已经合并的 Turn，就是其有效文本。
            "SELECT turn_id, COALESCE(effective_user_text, user_text) AS user_text, received_at "
            "FROM runtime_turns "
            "WHERE session_id=? AND status='failed' AND input_carried_into_turn_id IS NULL "
            f"AND error_code IN ({placeholders}) "
            "ORDER BY received_at ASC, turn_id ASC LIMIT ?",
            (session_id, *error_codes, limit),
        ).fetchall()
    return [dict(row) for row in rows]


def list_turn_input_segments(deps: StoreDeps, turn_id: str) -> list[dict[str, object]]:
    """重建组成合并 Turn 的各条独立用户话语。

    用户实际发送的每条消息各返回一项，最旧项在前，以当前 Turn 自身消息结束。未合并任何
    内容的 Turn 只返回自身，因此调用方无需对常见路径做特殊处理。
    """

    deps.init_db()
    with deps.connect() as conn:
        row = conn.execute(
            "SELECT turn_id, session_id, user_text, received_at FROM runtime_turns WHERE turn_id=?",
            (turn_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"unknown runtime turn: {turn_id}")
        segments = [dict(row)]
        frontier = [turn_id]
    # 每个 Turn 恰好由一个后续 Turn 消费，因此反向遍历该关系必然终止，且不会再次访问 Turn。
        while frontier:
            sources = conn.execute(
                "SELECT turn_id, session_id, user_text, received_at FROM runtime_turns "
                f"WHERE input_carried_into_turn_id IN ({','.join('?' for _ in frontier)})",
                tuple(frontier),
            ).fetchall()
            if not sources:
                break
            segments.extend(dict(item) for item in sources)
            frontier = [str(item["turn_id"]) for item in sources]
    segments.sort(key=lambda item: (str(item["received_at"]), str(item["turn_id"])))
    return segments


def attach_runtime_turn_carry_over(
    deps: StoreDeps,
    *,
    turn_id: str,
    merged_user_text: str,
    carried_turn_ids: tuple[str, ...],
) -> int:
    """在一个事务中采用合并输入并消费其来源。

    消费与采用不可分离：若来源标记为已消费但合并文本未进入当前 Turn，会静默丢失用户
    输入；若有合并文本却未消费来源，则会永久重放。

    ``user_text`` 永不覆盖，始终保留用户在当前 Turn 发送的单条消息，使合并背后的各条
    话语仍可重建；合并文本位于 ``effective_user_text``。
    """

    if not carried_turn_ids:
        return 0
    deps.init_db()
    with deps.connect() as conn:
        owner = conn.execute(
            "SELECT session_id, status FROM runtime_turns WHERE turn_id=?", (turn_id,)
        ).fetchone()
        if owner is None:
            raise ValueError(f"unknown runtime turn: {turn_id}")
        if str(owner["status"]) != "running":
            raise ValueError("carry-over can only be adopted by a running turn")
        conn.execute(
            "UPDATE runtime_turns SET effective_user_text=? WHERE turn_id=?",
            (merged_user_text, turn_id),
        )
        placeholders = ",".join("?" for _ in carried_turn_ids)
        cursor = conn.execute(
            f"UPDATE runtime_turns SET input_carried_into_turn_id=? "
            f"WHERE turn_id IN ({placeholders}) AND session_id=? "
            "AND input_carried_into_turn_id IS NULL",
            (turn_id, *carried_turn_ids, str(owner["session_id"])),
        )
        return int(cursor.rowcount or 0)


def append_runtime_turn_event(
    deps: StoreDeps,
    event: TurnEvent,
    *,
    active_window_lease_owner: str | None = None,
) -> int:
    """追加一个幂等事件，并返回其持久序列。

    ``runtime_turn_events_v1`` 没有可变业务状态，因此 SQLite rowid 是理想追加序列。它
    全局递增，所以在每个 Session 内也严格递增，同时避免第二序列权威或迁移所有计数器。
    拥有活动 Turn 的 Runtime 会传入租约所有者，使事件追加与心跳续租成为一个事务；过期
    恢复进程因而绝不会只观察到事件而看不到其存活事实。
    """

    if event.session_id is None:
        raise ValueError("persisted runtime event requires session_id")
    if active_window_lease_owner is not None and not active_window_lease_owner.strip():
        raise ValueError("active_window_lease_owner must not be empty")
    deps.init_db()
    payload = event.model_dump(mode="json")
    with deps.connect() as conn:
        if active_window_lease_owner is not None:
            conn.execute("BEGIN IMMEDIATE")
            _require_owned_active_window_before_event_append(
                conn,
                event=event,
                lease_owner=active_window_lease_owner,
            )
        sequence = _append_runtime_turn_event_in_transaction(
            conn,
            event=event,
            payload=payload,
        )
        if active_window_lease_owner is not None:
            _refresh_owned_active_window_heartbeat_in_transaction(
                conn,
                event=event,
                lease_owner=active_window_lease_owner,
                now=deps.now(),
            )
    if sequence < 1:  # pragma: no cover - sqlite 在此始终提供 rowid。
        raise RuntimeError("runtime event insert did not return a durable sequence")
    return sequence


def _append_runtime_turn_event_in_transaction(
    conn: sqlite3.Connection,
    *,
    event: TurnEvent,
    payload: dict[str, object],
) -> int:
    existing = conn.execute(
        "SELECT rowid AS sequence, session_id, turn_id "
        "FROM runtime_turn_events_v1 WHERE event_id=?",
        (event.event_id,),
    ).fetchone()
    if existing is not None:
        if (
            str(existing["session_id"]) != event.session_id
            or str(existing["turn_id"]) != event.turn_id
        ):
            raise ValueError("runtime event id collision across turns")
        return int(existing["sequence"])
    cursor = conn.execute(
        "INSERT INTO runtime_turn_events_v1 "
        "(event_id, session_id, turn_id, parent_event_id, insession_task_id, insession_task_node_id, "
        "work_run_id, attempt_id, stage, status, occurred_at, duration_ms, model_call_id, model_attempt, "
        "operation_id, error_code, retryable, diagnostic_ref) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            payload["event_id"], payload["session_id"], payload["turn_id"],
            payload["parent_event_id"], payload["insession_task_id"],
            payload["insession_task_node_id"], payload["work_run_id"], payload["attempt_id"],
            payload["stage"], payload["status"], payload["occurred_at"], payload["duration_ms"],
            payload["model_call_id"], payload["model_attempt"], payload["operation_id"],
            payload["error_code"], int(bool(payload["retryable"])), payload["diagnostic_ref"],
        ),
    )
    return int(cursor.lastrowid)


def _require_owned_active_window_before_event_append(
    conn: sqlite3.Connection,
    *,
    event: TurnEvent,
    lease_owner: str,
) -> None:
    window = conn.execute(
        "SELECT turn_id, window_state, lease_owner, heartbeat_at "
        "FROM turn_execution_windows WHERE session_id=?",
        (event.session_id,),
    ).fetchone()
    if window is None or str(window["turn_id"] or "") != event.turn_id:
        raise TurnExecutionPersistenceError("Runtime event Turn does not own the current Window")
    if str(window["window_state"]) != "active":
        return
    if str(window["lease_owner"] or "") != lease_owner:
        raise TurnExecutionLeaseConflict(
            expected_heartbeat_at=None,
            actual_heartbeat_at=(
                str(window["heartbeat_at"])
                if window["heartbeat_at"] is not None
                else None
            ),
        )


def _refresh_owned_active_window_heartbeat_in_transaction(
    conn: sqlite3.Connection,
    *,
    event: TurnEvent,
    lease_owner: str,
    now: str,
) -> None:
    updated = conn.execute(
        "UPDATE turn_execution_windows SET heartbeat_at=?, updated_at=? "
        "WHERE session_id=? AND turn_id=? AND window_state='active' AND lease_owner=?",
        (now, now, event.session_id, event.turn_id, lease_owner),
    ).rowcount
    if updated:
        return
    current = conn.execute(
        "SELECT turn_id, window_state, lease_owner, heartbeat_at "
        "FROM turn_execution_windows WHERE session_id=?",
        (event.session_id,),
    ).fetchone()
    if current is not None and str(current["turn_id"] or "") == event.turn_id:
        if str(current["window_state"]) != "active":
            return
        raise TurnExecutionLeaseConflict(
            expected_heartbeat_at=None,
            actual_heartbeat_at=(
                str(current["heartbeat_at"])
                if current["heartbeat_at"] is not None
                else None
            ),
        )
    raise TurnExecutionPersistenceError("Runtime event Turn lost its execution Window")


def list_runtime_turn_events(
    deps: StoreDeps,
    session_id: str,
    *,
    after: str | None = None,
    limit: int = 100,
) -> dict[str, object]:
    """读取一页按 Session 隔离且按时间排序的新 Runtime 事件。

    持久化层保留私有字段供诊断使用。调用方必须先投影记录，再返回给客户端。
    """
    if not session_id.strip():
        raise ValueError("session_id must not be empty")
    if isinstance(limit, bool) or not 1 <= limit <= MAX_RUNTIME_TURN_EVENT_PAGE_SIZE:
        raise ValueError(f"limit must be between 1 and {MAX_RUNTIME_TURN_EVENT_PAGE_SIZE}")
    deps.init_db()
    with deps.connect() as conn:
        if after:
            cursor = conn.execute(
                "SELECT rowid AS sequence FROM runtime_turn_events_v1 "
                "WHERE session_id=? AND event_id=?",
                (session_id, after),
            ).fetchone()
            if cursor is None:
                raise UnknownRuntimeTurnEventCursor(
                    f"unknown runtime turn event cursor for session: {after}"
                )
            rows = conn.execute(
                f"SELECT {_RUNTIME_TURN_EVENT_SELECT} "
                "FROM runtime_turn_events_v1 WHERE session_id=? AND "
                "rowid > ? ORDER BY rowid ASC LIMIT ?",
                (session_id, cursor["sequence"], limit + 1),
            ).fetchall()
            has_more = len(rows) > limit
            page_rows = rows[:limit]
        else:
            rows = conn.execute(
                f"SELECT {_RUNTIME_TURN_EVENT_SELECT} "
                "FROM runtime_turn_events_v1 WHERE session_id=? "
                "ORDER BY rowid DESC LIMIT ?",
                (session_id, limit),
            ).fetchall()
            page_rows = list(reversed(rows))
            has_more = False
    events = [dict(row) for row in page_rows]
    return {
        "events": events,
        "next_after": str(events[-1]["event_id"]) if events else after,
        "has_more": has_more,
        "limit": limit,
    }
