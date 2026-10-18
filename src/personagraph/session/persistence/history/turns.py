"""权威对话记录、幂等 turn-pair 与工作记忆持久化。"""

from __future__ import annotations

from dataclasses import dataclass
import sqlite3
from typing import Any

from ...turn_execution_contracts import TurnExecutionPersistenceError
from ..deps import StoreDeps


class TranscriptIntegrityError(TurnExecutionPersistenceError):
    """正式对话记录的提交绑定不完整或不唯一。"""


@dataclass(frozen=True, slots=True)
class CommittedTurnPairIndexState:
    """某 Session 中所有已提交 pair 的无内容 revision 标记。"""

    pair_count: int
    latest_user_turn_idx: int | None
    latest_created_at: str | None

    def __post_init__(self) -> None:
        if isinstance(self.pair_count, bool) or not isinstance(self.pair_count, int):
            raise ValueError("pair_count must be an integer")
        if self.pair_count < 0:
            raise ValueError("pair_count must not be negative")
        if self.pair_count == 0:
            if self.latest_user_turn_idx is not None or self.latest_created_at is not None:
                raise ValueError("empty pair state cannot declare a latest pair")
            return
        if (
            isinstance(self.latest_user_turn_idx, bool)
            or not isinstance(self.latest_user_turn_idx, int)
            or self.latest_user_turn_idx < 0
        ):
            raise ValueError("non-empty pair state requires a non-negative latest turn index")
        if not isinstance(self.latest_created_at, str) or not self.latest_created_at.strip():
            raise ValueError("non-empty pair state requires a latest timestamp")


@dataclass(frozen=True, slots=True)
class CommittedTurnPairIndexBinding:
    """一个指向可供检索的已提交 pair 的无内容指针。"""

    run_id: str
    created_at: str

    def __post_init__(self) -> None:
        for name, value in (("run_id", self.run_id), ("created_at", self.created_at)):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")


@dataclass(frozen=True, slots=True)
class CommittedTurnPairIndexBindingSnapshot:
    """当前已提交 pair 指针的有界无内容列表。"""

    bindings: tuple[CommittedTurnPairIndexBinding, ...]
    binding_enumeration_complete: bool

    def __post_init__(self) -> None:
        if not isinstance(self.bindings, tuple) or any(
            not isinstance(binding, CommittedTurnPairIndexBinding) for binding in self.bindings
        ):
            raise ValueError("bindings must be a tuple of CommittedTurnPairIndexBinding values")
        if not isinstance(self.binding_enumeration_complete, bool):
            raise ValueError("binding_enumeration_complete must be a bool")
        run_ids = {binding.run_id for binding in self.bindings}
        if len(run_ids) != len(self.bindings):
            raise ValueError("bindings must not contain duplicate run ids")


def append_assistant_for_accepted_input_in_transaction(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    turn_id: str,
    assistant_content: str,
    occurred_at: str,
) -> dict[str, int | bool]:
    """提交先前已接受 Runtime 输入的 assistant 半部分。

    ``runtime_turn_inputs`` 指向 accepted Turn 事务中插入的用户记录。因此最终化
    只写入正式 assistant 消息与 pair commit；再次插入用户记录会使同一请求在
    会话历史中出现两次。

    调用方提供事务连接，本 helper 不自行 commit。稳定 commit_<turn_id> 将已接受的
    user_turn_idx 与新增 assistant_turn_idx 关联成完整对话对，供历史与摘要读取。
    重放先核对同 Session / 同正文，返回 created=False；不同正文不能覆盖已交付内容。
    """

    existing = conn.execute(
        "SELECT c.run_id, c.session_id, c.user_turn_idx, c.assistant_turn_idx, "
        "c.created_at, u.content AS user_content, "
        "a.content AS assistant_content "
        "FROM session_turn_commits c "
        "JOIN session_turns u ON u.session_id=c.session_id AND u.turn_idx=c.user_turn_idx "
        "AND u.role='user' "
        "JOIN session_turns a ON a.session_id=c.session_id AND a.turn_idx=c.assistant_turn_idx "
        "AND a.role='assistant' WHERE c.turn_id=?",
        (turn_id,),
    ).fetchone()
    if existing is not None:
        if str(existing["session_id"]) != session_id:
            raise ValueError(f"turn commit turn_id collision: {turn_id}")
        if str(existing["assistant_content"]) != assistant_content:
            raise ValueError("accepted turn finalized with different assistant content")
        return {
            "user_turn_idx": int(existing["user_turn_idx"]),
            "assistant_turn_idx": int(existing["assistant_turn_idx"]),
            "created": False,
        }

    input_row = conn.execute(
        "SELECT i.turn_idx, t.content AS user_content FROM runtime_turn_inputs i "
        "JOIN session_turns t ON t.session_id=i.session_id AND t.turn_idx=i.turn_idx "
        "WHERE i.turn_id=? AND i.session_id=? AND t.role='user'",
        (turn_id, session_id),
    ).fetchone()
    if input_row is None:
        raise ValueError(f"accepted runtime turn input not found: {turn_id}")
    session = conn.execute("SELECT title FROM sessions WHERE id=?", (session_id,)).fetchone()
    if session is None:
        raise ValueError(f"unknown session: {session_id}")

    next_row = conn.execute(
        "SELECT COALESCE(MAX(turn_idx), -1) + 1 AS next_idx FROM session_turns WHERE session_id=?",
        (session_id,),
    ).fetchone()
    assistant_idx = int(next_row["next_idx"])
    user_idx = int(input_row["turn_idx"])
    conn.execute(
        "INSERT INTO session_turns (session_id, turn_idx, role, content, created_at) VALUES (?, ?, 'assistant', ?, ?)",
        (session_id, assistant_idx, assistant_content, occurred_at),
    )
    conn.execute(
        "INSERT INTO session_turn_commits "
        "(run_id, turn_id, session_id, user_turn_idx, assistant_turn_idx, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (
            f"commit_{turn_id}",
            turn_id,
            session_id,
            user_idx,
            assistant_idx,
            occurred_at,
        ),
    )
    if not session["title"]:
        conn.execute(
            "UPDATE sessions SET title=? WHERE id=?",
            (str(input_row["user_content"])[:20], session_id),
        )
    conn.execute("UPDATE sessions SET last_active_at=? WHERE id=?", (occurred_at, session_id))
    return {"user_turn_idx": user_idx, "assistant_turn_idx": assistant_idx, "created": True}


def _project_turn_rows_in_transaction(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    rows: list[sqlite3.Row],
) -> list[dict[str, Any]]:
    projected: list[dict[str, Any]] = []
    seen_turn_indices: set[int] = set()
    for row in rows:
        turn_idx = int(row["turn_idx"])
        if turn_idx in seen_turn_indices:
            raise TranscriptIntegrityError(
                "transcript row is bound to multiple formal commits"
            )
        seen_turn_indices.add(turn_idx)
        role = str(row["role"])
        content = str(row["content"])
        projected.append(
            {
                "turn_idx": turn_idx,
                "role": role,
                "content": content,
                "created_at": row["created_at"],
            }
        )
    return projected


def _committed_pair_from_row(
    conn: sqlite3.Connection,
    row: sqlite3.Row,
) -> dict[str, Any]:
    return dict(row)


def get_turns(deps: StoreDeps, session_id: str) -> list[dict[str, Any]]:
    deps.init_db()
    with deps.connect() as conn:
        return _get_turns_in_transaction(conn, session_id=session_id)


def _get_turns_in_transaction(
    conn: sqlite3.Connection,
    *,
    session_id: str,
) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT t.turn_idx, t.role, t.content, t.created_at, "
        "c.run_id AS commit_run_id, c.turn_id "
        "FROM session_turns AS t LEFT JOIN session_turn_commits AS c "
        "ON c.session_id=t.session_id "
        "AND c.assistant_turn_idx=t.turn_idx AND t.role='assistant' "
        "WHERE t.session_id=? ORDER BY t.turn_idx ASC",
        (session_id,),
    ).fetchall()
    return _project_turn_rows_in_transaction(
        conn,
        session_id=session_id,
        rows=rows,
    )


def get_turn(deps: StoreDeps, session_id: str, turn_idx: int) -> dict[str, Any] | None:
    deps.init_db()
    with deps.connect() as conn:
        rows = conn.execute(
            "SELECT t.turn_idx, t.role, t.content, t.created_at, "
            "c.run_id AS commit_run_id, c.turn_id "
            "FROM session_turns AS t LEFT JOIN session_turn_commits AS c "
            "ON c.session_id=t.session_id "
            "AND c.assistant_turn_idx=t.turn_idx AND t.role='assistant' "
            "WHERE t.session_id=? AND t.turn_idx=?",
            (session_id, int(turn_idx)),
        ).fetchall()
        if not rows:
            return None
        projected = _project_turn_rows_in_transaction(
            conn,
            session_id=session_id,
            rows=rows,
        )
        return projected[0]


def get_committed_turn_pair(deps: StoreDeps, session_id: str, run_id: str) -> dict[str, Any] | None:
    """根据稳定 run id 返回一个完整、权威的 user/assistant pair。

    Retrieval 索引完整 pair，而非流式片段或原始 runtime 日志。该 pair 仍由 session
    领域持有；调用方不会获得可写数据库句柄，也无法查看另一 session 的 pair。
    """

    deps.init_db()
    with deps.connect() as conn:
        row = conn.execute(
            "SELECT c.run_id, c.turn_id, c.session_id, c.user_turn_idx, "
            "c.assistant_turn_idx, c.created_at, u.content AS user_content, "
            "a.content AS assistant_content "
            "FROM session_turn_commits c "
            "JOIN session_turns u ON u.session_id=c.session_id AND u.turn_idx=c.user_turn_idx AND u.role='user' "
            "JOIN session_turns a ON a.session_id=c.session_id AND a.turn_idx=c.assistant_turn_idx AND a.role='assistant' "
            "WHERE c.session_id=? AND c.run_id=?",
            (session_id, run_id),
        ).fetchone()
        return _committed_pair_from_row(conn, row) if row is not None else None


def get_committed_turn_pair_by_turn_id(
    deps: StoreDeps,
    session_id: str,
    turn_id: str,
) -> dict[str, Any] | None:
    """返回一个精确的正式 Runtime pair。"""

    deps.init_db()
    with deps.connect() as conn:
        row = conn.execute(
            "SELECT c.run_id, c.turn_id, c.session_id, c.user_turn_idx, "
            "c.assistant_turn_idx, c.created_at, "
            "u.content AS user_content, a.content AS assistant_content "
            "FROM session_turn_commits c "
            "JOIN session_turns u ON u.session_id=c.session_id "
            "AND u.turn_idx=c.user_turn_idx AND u.role='user' "
            "JOIN session_turns a ON a.session_id=c.session_id "
            "AND a.turn_idx=c.assistant_turn_idx AND a.role='assistant' "
            "WHERE c.session_id=? AND c.turn_id=?",
            (session_id, turn_id),
        ).fetchone()
        if row is None:
            return None
        return _committed_pair_from_row(conn, row)


def list_committed_turn_pairs(
    deps: StoreDeps,
    session_id: str,
    *,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """按时间顺序列出完整 turn-pair 指针。

    ``limit`` 返回*最近* N 个 pair，但仍按时间排序。Runtime 调用方只会投影较短的
    最近窗口；在热点路径上加载整个对话记录再切取尾部，其开销会随 session 长度
    线性增长。Backfill 调用方省略该参数并保留完整扫描。
    """

    deps.init_db()
    select = (
        "SELECT c.run_id, c.turn_id, c.session_id, c.user_turn_idx, "
        "c.assistant_turn_idx, c.created_at, u.content AS user_content, "
        "a.content AS assistant_content "
        "FROM session_turn_commits c "
        "JOIN session_turns u ON u.session_id=c.session_id AND u.turn_idx=c.user_turn_idx AND u.role='user' "
        "JOIN session_turns a ON a.session_id=c.session_id AND a.turn_idx=c.assistant_turn_idx AND a.role='assistant' "
        "WHERE c.session_id=?"
    )
    with deps.connect() as conn:
        if limit is None:
            rows = conn.execute(
                f"{select} ORDER BY c.user_turn_idx, c.run_id", (session_id,)
            ).fetchall()
            return [_committed_pair_from_row(conn, row) for row in rows]
        if limit <= 0:
            return []
        rows = conn.execute(
            f"{select} ORDER BY c.user_turn_idx DESC, c.run_id DESC LIMIT ?",
            (session_id, limit),
        ).fetchall()
        return [_committed_pair_from_row(conn, row) for row in reversed(rows)]


def get_committed_turn_pair_index_state(
    deps: StoreDeps,
    session_id: str,
) -> CommittedTurnPairIndexState:
    """读取紧凑 revision 标记，而不加载对话记录内容。"""

    deps.init_db()
    with deps.connect() as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS pair_count, MAX(user_turn_idx) AS latest_user_turn_idx, "
            "MAX(created_at) AS latest_created_at "
            "FROM session_turn_commits WHERE session_id=?",
            (session_id,),
        ).fetchone()
    assert row is not None
    pair_count = int(row["pair_count"])
    return CommittedTurnPairIndexState(
        pair_count=pair_count,
        latest_user_turn_idx=(
            int(row["latest_user_turn_idx"])
            if row["latest_user_turn_idx"] is not None
            else None
        ),
        latest_created_at=(
            str(row["latest_created_at"])
            if row["latest_created_at"] is not None
            else None
        ),
    )


def get_committed_turn_pair_index_binding_snapshot(
    deps: StoreDeps,
    session_id: str,
    *,
    maximum_bindings: int,
) -> CommittedTurnPairIndexBindingSnapshot:
    """返回有界 pair 指针，而不读取用户或 assistant 文本。"""

    if (
        isinstance(maximum_bindings, bool)
        or not isinstance(maximum_bindings, int)
        or maximum_bindings <= 0
    ):
        raise ValueError("maximum_bindings must be a positive integer")
    deps.init_db()
    with deps.connect() as conn:
        rows = conn.execute(
            "SELECT run_id, created_at FROM session_turn_commits WHERE session_id=? "
            "ORDER BY user_turn_idx, run_id LIMIT ?",
            (session_id, maximum_bindings + 1),
        ).fetchall()
    complete = len(rows) <= maximum_bindings
    visible_rows = rows if complete else rows[:maximum_bindings]
    return CommittedTurnPairIndexBindingSnapshot(
        bindings=tuple(
            CommittedTurnPairIndexBinding(
                run_id=str(row["run_id"]),
                created_at=str(row["created_at"]),
            )
            for row in visible_rows
        ),
        binding_enumeration_complete=complete,
    )


def get_user_turn_markers(deps: StoreDeps, session_id: str) -> list[tuple[int, str]]:
    """返回有序用户 turn 索引/时间戳，而不加载对话记录文本。"""
    deps.init_db()
    with deps.connect() as conn:
        rows = conn.execute(
            "SELECT turn_idx, created_at FROM session_turns"
            " WHERE session_id=? AND role='user' ORDER BY turn_idx",
            (session_id,),
        ).fetchall()
    return [(int(row["turn_idx"]), str(row["created_at"])) for row in rows]


def get_history_messages(deps: StoreDeps, session_id: str) -> list[dict[str, str]]:
    return [{"role": turn["role"], "content": turn["content"]} for turn in get_turns(deps, session_id)]
