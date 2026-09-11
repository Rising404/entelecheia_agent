"""用于构建精确权威对话记录夹具的测试辅助模块。"""

from __future__ import annotations

from typing import Literal

from personagraph.session import store as session_store


__all__ = [
    "append_test_turn",
    "complete_test_turn_execution",
    "completed_test_turn_id",
]


def append_test_turn(
    session_id: str,
    role: Literal["user", "assistant"],
    content: str,
) -> int:
    """直接构造单条测试对话；生产代码没有对应的非原子写入口。"""

    deps = session_store._deps()
    deps.init_db()
    now = deps.now()
    with deps.connect() as conn:
        session = conn.execute(
            "SELECT title FROM sessions WHERE id=?",
            (session_id,),
        ).fetchone()
        if session is None:
            raise ValueError(f"unknown session: {session_id}")
        row = conn.execute(
            "SELECT COALESCE(MAX(turn_idx), -1) + 1 AS next_idx "
            "FROM session_turns WHERE session_id=?",
            (session_id,),
        ).fetchone()
        turn_idx = int(row["next_idx"])
        conn.execute(
            "INSERT INTO session_turns "
            "(session_id, turn_idx, role, content, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (session_id, turn_idx, role, content, now),
        )
        if role == "user" and not session["title"]:
            conn.execute(
                "UPDATE sessions SET title=? WHERE id=?",
                (content[:20], session_id),
            )
        conn.execute(
            "UPDATE sessions SET last_active_at=? WHERE id=?",
            (now, session_id),
        )
    return turn_idx


def complete_test_turn_execution(
    session_id: str,
    ordinal: int,
    *,
    user_content: str | None = None,
    assistant_content: str | None = None,
    post_commit_job_kinds: tuple[str, ...] = (),
) -> dict[str, object]:
    """构造一个走现行 Turn execution 状态机的正式测试交付。"""

    accepted = session_store.accept_turn_execution(
        session_id=session_id,
        client_request_id=f"post-commit-runtime-{ordinal}",
        source="runtime_test",
        user_text=(
            user_content if user_content is not None else f"用户消息 {ordinal}"
        ),
        lease_owner="runtime-test",
    )
    turn = accepted["turn"]  # type: ignore[assignment]
    window = accepted["window"]  # type: ignore[assignment]
    finalized = session_store.finalize_turn_execution(
        session_id=session_id,
        turn_id=str(turn["turn_id"]),  # type: ignore[index]
        expected_window_revision=int(window["state_version"]),  # type: ignore[index]
        processing_level="L0",
        assistant_content=(
            assistant_content
            if assistant_content is not None
            else f"助手回复 {ordinal}"
        ),
        post_commit_job_kinds=post_commit_job_kinds,
    )
    if not post_commit_job_kinds:
        final_window = finalized["window"]  # type: ignore[assignment]
        session_store.release_turn_execution_window(
            session_id=session_id,
            turn_id=str(turn["turn_id"]),  # type: ignore[index]
            expected_window_revision=int(final_window["state_version"]),  # type: ignore[index]
        )
    turn_id = str(turn["turn_id"])  # type: ignore[index]
    pair = session_store.get_committed_turn_pair(
        session_id,
        f"commit_{turn_id}",
    )
    assert pair is not None
    return {"accepted": accepted, "finalized": finalized, "pair": pair}


def completed_test_turn_id(result: dict[str, object]) -> str:
    """读取 ``complete_test_turn_execution`` 返回值中的权威 Turn ID。"""

    return str(result["accepted"]["turn"]["turn_id"])  # type: ignore[index]
