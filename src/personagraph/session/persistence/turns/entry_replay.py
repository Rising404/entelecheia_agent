"""Entry replay 的 L2-free lane receipt 存在性投影。"""

from __future__ import annotations

import sqlite3

from ...insession_task_contracts import InSessionTaskPersistenceError
from ..deps import StoreDeps


def has_turn_task_execution_lane_receipt(
    deps: StoreDeps,
    session_id: str,
    turn_id: str,
) -> bool:
    """判断一个归属明确的 Turn 是否有唯一、非空的 lane receipt。

    这里只决定是否值得加载 L2 manifest validator。JSON、哈希与来源绑定的完整
    authority 校验仍由 TaskGraph persistence owner 负责。
    """

    _require_identifier("session_id", session_id)
    _require_identifier("turn_id", turn_id)
    deps.init_db()
    conn = deps.connect()
    try:
        conn.execute("BEGIN")
        _require_session_turn(conn, session_id=session_id, turn_id=turn_id)
        rows = conn.execute(
            "SELECT execution_lane_manifest_json, execution_lane_manifest_hash "
            "FROM insession_task_match_apply_receipts "
            "WHERE session_id=? AND source_turn_id=?",
            (session_id, turn_id),
        ).fetchall()
        eligible = (
            len(rows) == 1
            and _has_text(rows[0]["execution_lane_manifest_json"])
            and _has_text(rows[0]["execution_lane_manifest_hash"])
        )
        conn.commit()
        return eligible
    except sqlite3.Error as exc:
        if conn.in_transaction:
            conn.rollback()
        raise InSessionTaskPersistenceError(
            "Entry replay lane receipt gate could not be read"
        ) from exc
    except BaseException:
        if conn.in_transaction:
            conn.rollback()
        raise
    finally:
        conn.close()


def _require_session_turn(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    turn_id: str,
) -> None:
    row = conn.execute(
        "SELECT turn.session_id FROM runtime_turns AS turn "
        "JOIN sessions AS session ON session.id=turn.session_id "
        "WHERE turn.turn_id=?",
        (turn_id,),
    ).fetchone()
    if row is None or str(row["session_id"]) != session_id:
        raise InSessionTaskPersistenceError(
            "Entry replay Turn is unknown or outside this Session"
        )


def _require_identifier(name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip() or len(value) > 200:
        raise InSessionTaskPersistenceError(f"invalid {name}")


def _has_text(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


__all__ = ["has_turn_task_execution_lane_receipt"]
