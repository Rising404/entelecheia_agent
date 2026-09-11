"""权威对话记录、幂等 turn-pair 与工作记忆持久化。"""

from __future__ import annotations

from dataclasses import dataclass
import sqlite3
from typing import Any, Literal

from ...entry_task_contracts import parse_entry_pending_question_decision_json
from ...turn_execution_contracts import TurnExecutionPersistenceError
from ..deps import StoreDeps


class TranscriptDeliveryReferenceError(TurnExecutionPersistenceError):
    """某条正式 assistant 记录无法解析其已验证 Delivery 正文。"""


VERIFIED_DELIVERY_JOIN_SEPARATOR = "\n\n"


@dataclass(frozen=True, slots=True)
class FormalTurnReference:
    """正式 Turn 中一个按来源排序且仅含引用的公开项。"""

    reference_kind: Literal["node_delivery", "pending_question_attempt"]
    reference_id: str

    def __post_init__(self) -> None:
        if self.reference_kind not in {
            "node_delivery",
            "pending_question_attempt",
        }:
            raise ValueError("unsupported formal Turn reference kind")
        if not isinstance(self.reference_id, str) or not self.reference_id.strip():
            raise ValueError("formal Turn reference id must be a non-empty string")


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
        if _load_commit_references_in_transaction(
            conn,
            run_id=str(existing["run_id"]),
        ):
            raise ValueError(
                "verified reference-only Turn cannot replay through the inline path"
            )
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


def append_verified_assistant_deliveries_for_accepted_input_in_transaction(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    turn_id: str,
    delivery_ids: tuple[str, ...],
    occurred_at: str,
) -> dict[str, int | bool]:
    """提交一个绑定到有序冻结 Delivery 的空 assistant 槽位。

    调用方持有外围权威事务，且已检查 Task/Turn 生命周期状态。此辅助函数会强制
    对话记录标识和仅引用正文不变量，包括精确元组重放。
    """

    _validate_delivery_ids(delivery_ids)
    return append_referenced_assistant_for_accepted_input_in_transaction(
        conn,
        session_id=session_id,
        turn_id=turn_id,
        reference_items=tuple(
            FormalTurnReference(
                reference_kind="node_delivery",
                reference_id=delivery_id,
            )
            for delivery_id in delivery_ids
        ),
        occurred_at=occurred_at,
    )


def append_referenced_assistant_for_accepted_input_in_transaction(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    turn_id: str,
    reference_items: tuple[FormalTurnReference, ...],
    occurred_at: str,
) -> dict[str, int | bool]:
    """提交一个绑定到按来源排序引用的空 assistant 槽位。"""

    _validate_formal_reference_items(reference_items)

    existing = conn.execute(
        "SELECT c.run_id, c.session_id, c.turn_id, c.user_turn_idx, c.assistant_turn_idx, "
        "c.created_at, u.content AS user_content, "
        "a.content AS assistant_content "
        "FROM session_turn_commits AS c "
        "JOIN session_turns AS u "
        "ON u.session_id=c.session_id AND u.turn_idx=c.user_turn_idx "
        "AND u.role='user' "
        "JOIN session_turns AS a "
        "ON a.session_id=c.session_id AND a.turn_idx=c.assistant_turn_idx "
        "AND a.role='assistant' WHERE c.turn_id=?",
        (turn_id,),
    ).fetchone()
    if existing is not None:
        if str(existing["session_id"]) != session_id:
            raise ValueError(f"turn commit turn_id collision: {turn_id}")
        persisted_items = _load_commit_references_in_transaction(
            conn,
            run_id=str(existing["run_id"]),
        )
        if persisted_items != reference_items:
            raise ValueError(
                "accepted Turn finalized with another formal reference tuple"
            )
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
    session = conn.execute(
        "SELECT title FROM sessions WHERE id=?", (session_id,)
    ).fetchone()
    if session is None:
        raise ValueError(f"unknown session: {session_id}")

    # 创建正式对话记录前，验证每个精确引用。
    for item in reference_items:
        _resolve_formal_reference_content_in_transaction(
            conn,
            session_id=session_id,
            turn_id=turn_id,
            reference=item,
        )
    next_row = conn.execute(
        "SELECT COALESCE(MAX(turn_idx), -1) + 1 AS next_idx "
        "FROM session_turns WHERE session_id=?",
        (session_id,),
    ).fetchone()
    assistant_idx = int(next_row["next_idx"])
    user_idx = int(input_row["turn_idx"])
    conn.execute(
        "INSERT INTO session_turns "
        "(session_id, turn_idx, role, content, created_at) "
        "VALUES (?, ?, 'assistant', '', ?)",
        (session_id, assistant_idx, occurred_at),
    )
    try:
        commit_run_id = f"commit_{turn_id}"
        conn.execute(
            "INSERT INTO session_turn_commits "
            "(run_id, turn_id, session_id, user_turn_idx, assistant_turn_idx, "
            "created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (
                commit_run_id,
                turn_id,
                session_id,
                user_idx,
                assistant_idx,
                occurred_at,
            ),
        )
        conn.executemany(
            "INSERT INTO session_turn_commit_references "
            "(run_id, ordinal, reference_kind, node_delivery_id, "
            "question_attempt_id) VALUES (?, ?, ?, ?, ?)",
            tuple(
                (
                    commit_run_id,
                    ordinal,
                    item.reference_kind,
                    (
                        item.reference_id
                        if item.reference_kind == "node_delivery"
                        else None
                    ),
                    (
                        item.reference_id
                        if item.reference_kind == "pending_question_attempt"
                        else None
                    ),
                )
                for ordinal, item in enumerate(reference_items, start=1)
            ),
        )
    except sqlite3.IntegrityError as exc:
        raise ValueError(
            "verified Delivery is already bound to another transcript commit"
        ) from exc
    if not session["title"]:
        conn.execute(
            "UPDATE sessions SET title=? WHERE id=?",
            (str(input_row["user_content"])[:20], session_id),
        )
    conn.execute(
        "UPDATE sessions SET last_active_at=? WHERE id=?",
        (occurred_at, session_id),
    )
    return {
        "user_turn_idx": user_idx,
        "assistant_turn_idx": assistant_idx,
        "created": True,
    }


def _validate_delivery_ids(delivery_ids: tuple[str, ...]) -> None:
    if not isinstance(delivery_ids, tuple) or not delivery_ids:
        raise ValueError("delivery_ids must be a non-empty tuple")
    if any(
        not isinstance(delivery_id, str) or not delivery_id.strip()
        for delivery_id in delivery_ids
    ):
        raise ValueError("delivery_ids must contain non-empty strings")
    if len(set(delivery_ids)) != len(delivery_ids):
        raise ValueError("delivery_ids must not contain duplicates")


def _validate_formal_reference_items(
    reference_items: tuple[FormalTurnReference, ...],
) -> None:
    if not isinstance(reference_items, tuple) or not reference_items:
        raise ValueError("reference_items must be a non-empty tuple")
    if any(
        not isinstance(item, FormalTurnReference) for item in reference_items
    ):
        raise TypeError("reference_items must contain FormalTurnReference values")
    reference_ids = tuple(item.reference_id for item in reference_items)
    if len(set(reference_ids)) != len(reference_ids):
        raise ValueError("formal Turn reference IDs must not contain duplicates")


def _resolve_assistant_content_in_transaction(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    turn_id: str | None,
    node_delivery_id: str | None,
    inline_content: str,
) -> str:
    """从 inline 所有者或已验证所有者解析一个 assistant 正文。"""

    if node_delivery_id is None:
        return inline_content
    if inline_content != "":
        raise TranscriptDeliveryReferenceError(
            "verified assistant transcript row contains a duplicate inline body"
        )
    if turn_id is None:
        raise TranscriptDeliveryReferenceError(
            "verified assistant transcript commit has no Runtime Turn identity"
        )

    # 将完整 Delivery 验证集中在一处。简单 SQL join 会在其 request/run/output
    # 权威状态发生漂移时仍暴露正文。
    from ..l2.work_run.work_verification import _load_task_node_delivery

    try:
        resolved = _load_task_node_delivery(
            conn,
            session_id=session_id,
            delivery_id=node_delivery_id,
        )
    except (RuntimeError, ValueError) as exc:
        raise TranscriptDeliveryReferenceError(
            "verified assistant transcript Delivery reference is invalid"
        ) from exc
    if resolved.delivery.created_turn_id != turn_id:
        raise TranscriptDeliveryReferenceError(
            "verified assistant transcript Delivery belongs to another Turn"
        )
    return resolved.output_window.content


def _load_commit_references_in_transaction(
    conn: sqlite3.Connection,
    *,
    run_id: str,
) -> tuple[FormalTurnReference, ...]:
    """从唯一引用表加载某次 commit 的精确有序引用。"""

    rows = conn.execute(
        "SELECT ordinal, reference_kind, node_delivery_id, question_attempt_id "
        "FROM session_turn_commit_references "
        "WHERE run_id=? ORDER BY ordinal",
        (run_id,),
    ).fetchall()
    if not rows:
        return ()
    ordinals = tuple(int(row["ordinal"]) for row in rows)
    if ordinals != tuple(range(1, len(rows) + 1)):
        raise TranscriptDeliveryReferenceError(
            "verified assistant transcript Delivery ordinals are not contiguous"
        )
    references: list[FormalTurnReference] = []
    for row in rows:
        kind = str(row["reference_kind"])
        if kind == "node_delivery":
            if row["node_delivery_id"] is None or row["question_attempt_id"] is not None:
                raise TranscriptDeliveryReferenceError(
                    "verified assistant transcript Delivery reference is malformed"
                )
            reference_id = str(row["node_delivery_id"])
        elif kind == "pending_question_attempt":
            if row["question_attempt_id"] is None or row["node_delivery_id"] is not None:
                raise TranscriptDeliveryReferenceError(
                    "assistant transcript pending-question reference is malformed"
                )
            reference_id = str(row["question_attempt_id"])
        else:
            raise TranscriptDeliveryReferenceError(
                "assistant transcript reference kind is invalid"
            )
        try:
            references.append(
                FormalTurnReference(
                    reference_kind=kind,  # type: ignore[arg-type]
                    reference_id=reference_id,
                )
            )
        except ValueError as exc:
            raise TranscriptDeliveryReferenceError(
                "assistant transcript reference is invalid"
            ) from exc
    reference_items = tuple(references)
    if len({item.reference_id for item in reference_items}) != len(reference_items):
        raise TranscriptDeliveryReferenceError(
            "assistant transcript reference tuple is duplicated"
        )
    return reference_items


def _resolve_commit_assistant_content_in_transaction(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    turn_id: str | None,
    commit_run_id: str,
    inline_content: str,
) -> str:
    """从有序 commit 引用解析一份正式 assistant 正文。"""

    reference_items = _load_commit_references_in_transaction(
        conn,
        run_id=commit_run_id,
    )
    if not reference_items:
        return inline_content
    if inline_content != "":
        raise TranscriptDeliveryReferenceError(
            "verified assistant transcript row contains a duplicate inline body"
        )
    return VERIFIED_DELIVERY_JOIN_SEPARATOR.join(
        _resolve_formal_reference_content_in_transaction(
            conn,
            session_id=session_id,
            turn_id=turn_id,
            reference=reference,
        )
        for reference in reference_items
    )


def _resolve_formal_reference_content_in_transaction(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    turn_id: str | None,
    reference: FormalTurnReference,
) -> str:
    if reference.reference_kind == "node_delivery":
        return _resolve_assistant_content_in_transaction(
            conn,
            session_id=session_id,
            turn_id=turn_id,
            node_delivery_id=reference.reference_id,
            inline_content="",
        )
    if turn_id is None:
        raise TranscriptDeliveryReferenceError(
            "pending-question transcript commit has no Runtime Turn identity"
        )
    return _resolve_pending_question_attempt_in_transaction(
        conn,
        session_id=session_id,
        turn_id=turn_id,
        question_attempt_id=reference.reference_id,
    )


def _resolve_pending_question_attempt_in_transaction(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    turn_id: str,
    question_attempt_id: str,
) -> str:
    """解析不可变问题正文，不受后续消费影响。"""

    row = conn.execute(
        "SELECT attempt.work_run_id, attempt.turn_id, attempt.status, "
        "attempt.action, attempt.decision_json, attempt.close_reason "
        "FROM insession_work_run_attempts AS attempt "
        "JOIN insession_work_runs AS run ON run.work_run_id=attempt.work_run_id "
        "WHERE attempt.attempt_id=? AND run.session_id=?",
        (question_attempt_id, session_id),
    ).fetchone()
    if (
        row is None
        or str(row["turn_id"]) != turn_id
        or str(row["status"]) != "closed"
        or str(row["action"] or "") != "request_user_input"
        or str(row["close_reason"] or "") != "request_user_input"
        or row["decision_json"] is None
    ):
        raise TranscriptDeliveryReferenceError(
            "assistant transcript pending-question Attempt is invalid"
        )
    try:
        return parse_entry_pending_question_decision_json(
            str(row["decision_json"]),
        )
    except ValueError as exc:
        raise TranscriptDeliveryReferenceError(
            "assistant transcript pending-question payload is invalid"
        ) from exc


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
            raise TranscriptDeliveryReferenceError(
                "transcript row is bound to multiple formal commits"
            )
        seen_turn_indices.add(turn_idx)
        role = str(row["role"])
        content = str(row["content"])
        if role == "assistant" and row["commit_run_id"] is not None:
            content = _resolve_commit_assistant_content_in_transaction(
                conn,
                session_id=session_id,
                turn_id=(
                    str(row["turn_id"]) if row["turn_id"] is not None else None
                ),
                commit_run_id=str(row["commit_run_id"]),
                inline_content=content,
            )
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
    result = dict(row)
    result["assistant_content"] = _resolve_commit_assistant_content_in_transaction(
        conn,
        session_id=str(row["session_id"]),
        turn_id=str(row["turn_id"]) if row["turn_id"] is not None else None,
        commit_run_id=str(row["run_id"]),
        inline_content=str(row["assistant_content"]),
    )
    return result


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
        result = _committed_pair_from_row(conn, row)
        task_rows = conn.execute(
            "SELECT insession_task_id, MIN(link_id) AS first_link_id "
            "FROM insession_task_turn_links WHERE session_id=? AND turn_id=? "
            "GROUP BY insession_task_id ORDER BY first_link_id, insession_task_id",
            (session_id, turn_id),
        ).fetchall()
        # 只有正式 Turn 恰好关联一个根 Task 时，memory candidate 才能继承 Task
        # 权威状态。多 Task 或未链接的 Turn 仍限定于 user/session 范围，不猜测归属。
        result["task_id"] = (
            str(task_rows[0]["insession_task_id"])
            if len(task_rows) == 1
            else None
        )
        return result


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
