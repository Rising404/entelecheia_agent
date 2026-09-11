"""Entry 的 L2-free Task 只读持久化投影。

本模块只读取 Session 持有的权威事实。它不执行 TaskGraph/WorkRun，也不把无法
验证来源的 Task 暴露给模型。每个公开读取都在一个显式 SQLite 快照内完成。
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from dataclasses import dataclass
from typing import Any

from ...entry_task_contracts import EntryPendingTaskQuestion, EntryTaskCatalogItem
from ...insession_task_contracts import InSessionTaskPersistenceError
from ..deps import StoreDeps
from .pending_questions import list_pending_question_proofs


_ALLOWED_SOURCE_KINDS = {
    "current_user_instruction",
    "current_user_context",
    "previously_authorized_task_state",
    "quoted_external",
    "attachment",
    "retrieved_document",
    "tool_observation",
    "memory",
    "gap",
}
_CURRENT_USER_SOURCE_KINDS = {
    "current_user_instruction",
    "current_user_context",
}
_LOCAL_KEY = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")


@dataclass(frozen=True)
class _StoredSourceAnchor:
    anchor_id: str
    source_turn_id: str
    source_kind: str
    gap_blocking: bool | None
    start: int
    end: int
    excerpt: str


def list_entry_task_catalog(
    deps: StoreDeps,
    session_id: str,
    *,
    limit: int | None = None,
) -> tuple[EntryTaskCatalogItem, ...]:
    """返回来源可验证的根 Task，并保持现行 catalog 的稳定排序。"""

    _require_identifier("session_id", session_id)
    if limit is not None and limit < 1:
        return ()
    deps.init_db()
    conn = deps.connect()
    try:
        conn.execute("BEGIN")
        _require_session(conn, session_id)
        rows = conn.execute(
            "SELECT task.insession_task_id, task.session_id, task.root_title, "
            "task.root_objective, task.current_status, "
            "task.current_graph_revision, task.created_turn_id, "
            "task.creation_source_start, task.creation_source_end, "
            "task.creation_source_sha256, revision.source_turn_id, "
            "revision.source_anchors_json, "
            "revision.authorization_anchor_ids_json "
            "FROM insession_tasks AS task "
            "LEFT JOIN insession_task_graph_revisions AS revision "
            "ON revision.insession_task_id=task.insession_task_id "
            "AND revision.graph_revision=task.current_graph_revision "
            "WHERE task.session_id=? "
            "ORDER BY CASE WHEN task.current_status IN "
            "('completed', 'cancelled') THEN 1 ELSE 0 END, "
            "task.updated_at DESC, task.insession_task_id",
            (session_id,),
        ).fetchall()
        items: list[EntryTaskCatalogItem] = []
        for row in rows:
            revision = row["current_graph_revision"]
            readable = (
                _has_readable_shell_provenance(conn, row)
                if revision is None
                else _has_readable_authorization_manifest(conn, row)
            )
            if not readable:
                continue
            try:
                item = EntryTaskCatalogItem(
                    insession_task_id=str(row["insession_task_id"]),
                    goal_summary=_goal_summary(
                        row["root_title"],
                        row["root_objective"],
                    ),
                    status=str(row["current_status"]),
                    current_graph_revision=(
                        int(revision) if revision is not None else None
                    ),
                )
            except (TypeError, ValueError):
                # Catalog 是安全暴露边界；一条损坏记录不能成为模型事实。
                continue
            items.append(item)
        conn.commit()
        selected = items[:limit] if limit is not None else items
        return tuple(selected)
    except BaseException:
        if conn.in_transaction:
            conn.rollback()
        raise
    finally:
        conn.close()


def list_entry_pending_task_questions(
    deps: StoreDeps,
    *,
    session_id: str,
) -> tuple[EntryPendingTaskQuestion, ...]:
    """返回同一 SQLite 快照内认证过的当前问题。"""

    _require_identifier("session_id", session_id)
    deps.init_db()
    conn = deps.connect()
    try:
        conn.execute("BEGIN")
        _require_session(conn, session_id)
        proofs = list_pending_question_proofs(conn, session_id=session_id)
        questions = tuple(
            EntryPendingTaskQuestion(
                insession_task_id=proof.insession_task_id,
                question=proof.question,
            )
            for proof in proofs
        )
        conn.commit()
        return questions
    except InSessionTaskPersistenceError:
        if conn.in_transaction:
            conn.rollback()
        raise
    except (IndexError, KeyError, TypeError, ValueError, sqlite3.Error) as exc:
        if conn.in_transaction:
            conn.rollback()
        raise InSessionTaskPersistenceError(
            "pending Task question projection is corrupt"
        ) from exc
    except BaseException:
        if conn.in_transaction:
            conn.rollback()
        raise
    finally:
        conn.close()


def list_turn_insession_task_ids(
    deps: StoreDeps,
    session_id: str,
    turn_id: str,
) -> tuple[str, ...]:
    """按首次持久链接顺序返回某 Turn 的根 Task。"""

    _require_identifier("session_id", session_id)
    _require_identifier("turn_id", turn_id)
    deps.init_db()
    conn = deps.connect()
    try:
        conn.execute("BEGIN")
        _require_session_turn(conn, session_id=session_id, turn_id=turn_id)
        rows = conn.execute(
            "SELECT insession_task_id, MIN(link_id) AS first_link_id "
            "FROM insession_task_turn_links "
            "WHERE session_id=? AND turn_id=? "
            "GROUP BY insession_task_id "
            "ORDER BY first_link_id, insession_task_id",
            (session_id, turn_id),
        ).fetchall()
        result = tuple(str(row["insession_task_id"]) for row in rows)
        conn.commit()
        return result
    except BaseException:
        if conn.in_transaction:
            conn.rollback()
        raise
    finally:
        conn.close()


def list_turn_linked_work_run_ids(
    deps: StoreDeps,
    *,
    session_id: str,
    turn_id: str,
) -> tuple[str, ...]:
    """按链接 revision 返回某 Turn 的全部 WorkRun，包括终态记录。"""

    _require_identifier("session_id", session_id)
    _require_identifier("turn_id", turn_id)
    deps.init_db()
    conn = deps.connect()
    try:
        conn.execute("BEGIN")
        _require_session_turn(conn, session_id=session_id, turn_id=turn_id)
        rows = conn.execute(
            "SELECT link.work_run_id "
            "FROM insession_work_run_turn_links AS link "
            "JOIN insession_work_runs AS run "
            "ON run.session_id=link.session_id "
            "AND run.work_run_id=link.work_run_id "
            "WHERE link.session_id=? AND link.turn_id=? "
            "ORDER BY link.link_revision, link.link_id",
            (session_id, turn_id),
        ).fetchall()
        result = tuple(str(row["work_run_id"]) for row in rows)
        conn.commit()
        return result
    except BaseException:
        if conn.in_transaction:
            conn.rollback()
        raise
    finally:
        conn.close()


def _has_readable_authorization_manifest(
    conn: sqlite3.Connection,
    task: sqlite3.Row,
) -> bool:
    try:
        source_turn_id = _required_text(task["source_turn_id"], max_length=200)
        anchors = _source_anchors_from_json(task["source_anchors_json"])
        authorization_anchor_ids = _ids_from_json(
            task["authorization_anchor_ids_json"]
        )
    except (IndexError, KeyError, TypeError, ValueError):
        return False
    anchors_by_id = {anchor.anchor_id: anchor for anchor in anchors}
    if not anchors or not authorization_anchor_ids:
        return False
    for anchor_id in authorization_anchor_ids:
        anchor = anchors_by_id.get(anchor_id)
        if anchor is None:
            return False
        if anchor.source_kind in _CURRENT_USER_SOURCE_KINDS:
            if anchor.source_turn_id != source_turn_id:
                return False
            try:
                user_text = _load_authoritative_user_input(
                    conn,
                    session_id=str(task["session_id"]),
                    turn_id=anchor.source_turn_id,
                )
            except InSessionTaskPersistenceError:
                return False
            if (
                anchor.end > len(user_text)
                or user_text[anchor.start : anchor.end] != anchor.excerpt
            ):
                return False
            continue
        if anchor.source_kind == "previously_authorized_task_state":
            if not _matches_shell_creation_anchor(conn, task, anchor):
                return False
            continue
        return False
    return True


def _has_readable_shell_provenance(
    conn: sqlite3.Connection,
    task: sqlite3.Row,
) -> bool:
    try:
        session_id = _required_text(task["session_id"], max_length=200)
        turn_id = _required_text(task["created_turn_id"], max_length=200)
        start = _nonnegative_int(task["creation_source_start"])
        end = _positive_int(task["creation_source_end"])
        expected_hash = _required_text(
            task["creation_source_sha256"],
            exact_length=64,
        )
    except (IndexError, KeyError, TypeError, ValueError):
        return False
    if end <= start:
        return False
    try:
        user_text = _load_authoritative_user_input(
            conn,
            session_id=session_id,
            turn_id=turn_id,
        )
    except InSessionTaskPersistenceError:
        return False
    if end > len(user_text):
        return False
    actual_hash = hashlib.sha256(user_text[start:end].encode("utf-8")).hexdigest()
    return actual_hash == expected_hash


def _matches_shell_creation_anchor(
    conn: sqlite3.Connection,
    task: sqlite3.Row,
    anchor: _StoredSourceAnchor,
) -> bool:
    try:
        session_id = _required_text(task["session_id"], max_length=200)
        created_turn_id = _required_text(task["created_turn_id"], max_length=200)
        start = _nonnegative_int(task["creation_source_start"])
        end = _positive_int(task["creation_source_end"])
        expected_hash = _required_text(
            task["creation_source_sha256"],
            exact_length=64,
        )
    except (IndexError, KeyError, TypeError, ValueError):
        return False
    if (
        anchor.source_turn_id != created_turn_id
        or anchor.start != start
        or anchor.end != end
        or hashlib.sha256(anchor.excerpt.encode("utf-8")).hexdigest()
        != expected_hash
    ):
        return False
    try:
        user_text = _load_authoritative_user_input(
            conn,
            session_id=session_id,
            turn_id=created_turn_id,
        )
    except InSessionTaskPersistenceError:
        return False
    return end <= len(user_text) and user_text[start:end] == anchor.excerpt


def _source_anchors_from_json(value: object) -> tuple[_StoredSourceAnchor, ...]:
    decoded = _decode_json(value)
    if not isinstance(decoded, list):
        raise ValueError("stored Task source anchors are not a list")
    return tuple(_source_anchor_from_value(item) for item in decoded)


def _source_anchor_from_value(value: object) -> _StoredSourceAnchor:
    if not isinstance(value, dict):
        raise ValueError("stored Task source anchor is not an object")
    required_keys = {
        "anchor_id",
        "source_turn_id",
        "source_kind",
        "start",
        "end",
        "excerpt",
    }
    if not required_keys <= value.keys() or not value.keys() <= (
        required_keys | {"gap_blocking"}
    ):
        raise ValueError("stored Task source anchor has an invalid shape")
    anchor_id = _required_text(value["anchor_id"], max_length=64)
    if _LOCAL_KEY.fullmatch(anchor_id) is None:
        raise ValueError("stored Task source anchor id is invalid")
    source_turn_id = _required_text(value["source_turn_id"], max_length=128)
    source_kind = _required_text(value["source_kind"])
    if source_kind not in _ALLOWED_SOURCE_KINDS:
        raise ValueError("stored Task source kind is invalid")
    gap_blocking = value.get("gap_blocking")
    if gap_blocking is not None and not isinstance(gap_blocking, bool):
        raise ValueError("stored Task gap disposition is invalid")
    if (source_kind == "gap") != (gap_blocking is not None):
        raise ValueError("stored Task gap disposition is inconsistent")
    start = _nonnegative_int(value["start"])
    end = _positive_int(value["end"])
    if end <= start:
        raise ValueError("stored Task source range is invalid")
    excerpt = _required_text(value["excerpt"], max_length=4_000)
    return _StoredSourceAnchor(
        anchor_id=anchor_id,
        source_turn_id=source_turn_id,
        source_kind=source_kind,
        gap_blocking=gap_blocking,
        start=start,
        end=end,
        excerpt=excerpt,
    )


def _ids_from_json(value: object) -> tuple[str, ...]:
    decoded = _decode_json(value)
    if not isinstance(decoded, list) or not all(
        isinstance(item, str) for item in decoded
    ):
        raise ValueError("stored Task authorization ids are invalid")
    return tuple(decoded)


def _decode_json(value: object) -> Any:
    try:
        return json.loads(str(value))
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("stored Task JSON is corrupt") from exc


def _load_authoritative_user_input(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    turn_id: str,
) -> str:
    row = conn.execute(
        "SELECT turn_input.content "
        "FROM runtime_turn_inputs AS input "
        "JOIN session_turns AS turn_input "
        "ON turn_input.session_id=input.session_id "
        "AND turn_input.turn_idx=input.turn_idx "
        "WHERE input.session_id=? AND input.turn_id=? "
        "AND turn_input.role='user'",
        (session_id, turn_id),
    ).fetchone()
    if row is None:
        raise InSessionTaskPersistenceError(
            "Task source Turn has no authoritative accepted user input"
        )
    return str(row["content"])


def _goal_summary(title: object, objective: object) -> str:
    return f"{str(title).strip()}：{str(objective).strip()}"


def _require_session(conn: sqlite3.Connection, session_id: str) -> None:
    if conn.execute(
        "SELECT 1 FROM sessions WHERE id=?",
        (session_id,),
    ).fetchone() is None:
        raise InSessionTaskPersistenceError("unknown Session")


def _require_session_turn(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    turn_id: str,
) -> None:
    row = conn.execute(
        "SELECT session_id FROM runtime_turns WHERE turn_id=?",
        (turn_id,),
    ).fetchone()
    if row is None or str(row["session_id"]) != session_id:
        raise InSessionTaskPersistenceError(
            "Turn is unknown or outside this Session"
        )


def _require_identifier(name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip() or len(value) > 200:
        raise InSessionTaskPersistenceError(f"invalid {name}")


def _required_text(
    value: object,
    *,
    max_length: int | None = None,
    exact_length: int | None = None,
) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("stored text is missing")
    if max_length is not None and len(value) > max_length:
        raise ValueError("stored text exceeds its bound")
    if exact_length is not None and len(value) != exact_length:
        raise ValueError("stored text has the wrong length")
    return value


def _nonnegative_int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("stored integer must be nonnegative")
    return value


def _positive_int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("stored integer must be positive")
    return value


__all__ = [
    "list_entry_pending_task_questions",
    "list_entry_task_catalog",
    "list_turn_insession_task_ids",
    "list_turn_linked_work_run_ids",
]
