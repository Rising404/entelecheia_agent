"""Session 标识、列表/搜索、生命周期与 SQLite 清除持久化。"""

from __future__ import annotations

from typing import Any

from ..history import turns as transcript_turns
from ..deps import StoreDeps
from . import queries


def create_session(
    deps: StoreDeps,
    persona_id: str,
    title: str | None = None,
    folder_id: str | None = None,
    working_dir: str | None = None,
    session_id: str | None = None,
) -> str:
    deps.init_db()
    session_id = session_id or deps.new_id()
    now = deps.now()
    with deps.connect() as conn:
        conn.execute(
            "INSERT INTO sessions (id, persona_id, title, folder_id, working_dir, status, created_at, last_active_at)"
            " VALUES (?, ?, ?, ?, ?, 'active', ?, ?)",
            (session_id, persona_id, title, folder_id, working_dir, now, now),
        )
    return session_id


def get_session(deps: StoreDeps, session_id: str) -> dict[str, Any] | None:
    deps.init_db()
    with deps.connect() as conn:
        row = conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
    return dict(row) if row else None


def list_sessions(
    deps: StoreDeps,
    include_archived: bool = False,
    include_trashed: bool = False,
    *,
    status: str = "active",
    folder_id: str | None = None,
    persona_id: str | None = None,
    query: str | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    deps.init_db()
    with deps.connect() as conn:
        rows = conn.execute("SELECT * FROM sessions ORDER BY last_active_at DESC")
        return queries.list_sessions(
            (dict(row) for row in rows),
            lambda session_id: transcript_turns._get_turns_in_transaction(
                conn, session_id=session_id
            ),
            include_archived=include_archived,
            include_trashed=include_trashed,
            status=status,
            folder_id=folder_id,
            persona_id=persona_id,
            query=query,
            limit=limit,
        )


def search_sessions(
    deps: StoreDeps,
    query: str,
    status: str = "active",
    folder_id: str | None = None,
    persona_id: str | None = None,
    limit: int = 20,
) -> list[dict[str, Any]]:
    deps.init_db()
    with deps.connect() as conn:
        rows = conn.execute("SELECT * FROM sessions ORDER BY last_active_at DESC")
        return queries.search_sessions(
            (dict(row) for row in rows),
            lambda session_id: transcript_turns._get_turns_in_transaction(
                conn, session_id=session_id
            ),
            query=query,
            status=status,
            folder_id=folder_id,
            persona_id=persona_id,
            limit=limit,
        )


def rename_session(deps: StoreDeps, session_id: str, title: str) -> bool:
    deps.init_db()
    with deps.connect() as conn:
        return conn.execute("UPDATE sessions SET title = ? WHERE id = ?", (title, session_id)).rowcount > 0


def replace_session_title(deps: StoreDeps, session_id: str, *, expected: str | None, title: str) -> bool:
    """仅替换生成开始时的占位值，避免覆盖其后的人工作业。"""
    deps.init_db()
    with deps.connect() as conn:
        return conn.execute(
            "UPDATE sessions SET title=? WHERE id=? AND title IS ?",
            (title, session_id, expected),
        ).rowcount > 0


def archive_session(deps: StoreDeps, session_id: str) -> bool:
    deps.init_db()
    with deps.connect() as conn:
        return conn.execute(
            "UPDATE sessions SET status='archived', archived_at=? WHERE id=? AND status='active'",
            (deps.now(), session_id),
        ).rowcount > 0


def unarchive_session(deps: StoreDeps, session_id: str) -> bool:
    deps.init_db()
    with deps.connect() as conn:
        return conn.execute(
            "UPDATE sessions SET status='active', archived_at=NULL WHERE id=? AND status='archived'", (session_id,)
        ).rowcount > 0


def trash_session(deps: StoreDeps, session_id: str) -> bool:
    deps.init_db()
    with deps.connect() as conn:
        now = deps.now()
        updated = conn.execute(
            "UPDATE sessions SET previous_status=status, status='trashed', deleted_at=?"
            " WHERE id=? AND status IN ('active','archived')",
            (now, session_id),
        ).rowcount > 0
        return updated


def restore_session(deps: StoreDeps, session_id: str) -> bool:
    deps.init_db()
    with deps.connect() as conn:
        updated = conn.execute(
            "UPDATE sessions SET status=COALESCE(previous_status, 'active'), "
            "previous_status=NULL, deleted_at=NULL"
            " WHERE id=? AND status='trashed'",
            (session_id,),
        ).rowcount > 0
        return updated


def purge_session(deps: StoreDeps, session_id: str) -> bool:
    deps.init_db()
    with deps.connect() as conn:
        row = conn.execute(
            "SELECT 1 FROM sessions WHERE id=?",
            (session_id,),
        ).fetchone()
        found = row is not None
        # 显式清除整个 Session 时，先释放提交和执行窗口的 RESTRICT 引用。
        conn.execute("DELETE FROM session_turn_commits WHERE session_id=?", (session_id,))
        conn.execute(
            "DELETE FROM turn_execution_windows WHERE session_id=?",
            (session_id,),
        )
        conn.execute(
            "DELETE FROM runtime_turns WHERE session_id=?",
            (session_id,),
        )
        for table in (
            "session_context_repair_applies", "session_context_resets", "session_turn_commits", "session_observation_candidates",
            "session_state_transitions", "session_state_items", "session_evidence_events",
            "session_turns", "session_working_memory", "session_context_revisions",
        ):
            conn.execute(f"DELETE FROM {table} WHERE session_id=?", (session_id,))
        # 诊断轨迹是同一权威状态中由 Session 持有的记录。在 Session 标识前移除它们，
        # 且仅回收不再被任何剩余 step 引用的 blob；全部操作都在清除事务内完成。
        conn.execute(
            "DELETE FROM trajectory_steps WHERE session_id=?",
            (session_id,),
        )
        conn.execute(
            "DELETE FROM trajectory_blobs WHERE sha256 NOT IN "
            "(SELECT blob_sha256 FROM trajectory_parts)"
        )
        conn.execute("DELETE FROM sessions WHERE id=?", (session_id,))
    return found


def list_trashed(deps: StoreDeps) -> list[dict[str, Any]]:
    deps.init_db()
    with deps.connect() as conn:
        rows = conn.execute("SELECT * FROM sessions WHERE status='trashed' ORDER BY deleted_at DESC").fetchall()
    return [dict(row) for row in rows]
