"""文件夹 CRUD、层级校验与树投影。"""

from __future__ import annotations

import sqlite3
from typing import Any

from ..deps import StoreDeps


def create_folder(deps: StoreDeps, name: str, parent_id: str | None = None) -> str:
    deps.init_db()
    folder_id = deps.new_id()
    now = deps.now()
    with deps.connect() as conn:
        conn.execute(
            "INSERT INTO session_folders (id, name, parent_id, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
            (folder_id, name, parent_id, now, now),
        )
    return folder_id


def get_folder(deps: StoreDeps, folder_id: str) -> dict[str, Any] | None:
    deps.init_db()
    with deps.connect() as conn:
        row = conn.execute("SELECT * FROM session_folders WHERE id=?", (folder_id,)).fetchone()
    return dict(row) if row else None


def list_folders(deps: StoreDeps) -> list[dict[str, Any]]:
    deps.init_db()
    with deps.connect() as conn:
        rows = conn.execute("SELECT * FROM session_folders ORDER BY sort_order, name").fetchall()
        result = []
        for row in rows:
            count = conn.execute(
                "SELECT COUNT(*) AS c FROM sessions WHERE folder_id=? AND status!='trashed'", (row["id"],)
            ).fetchone()["c"]
            item = dict(row)
            item["session_count"] = count
            result.append(item)
    return result


def rename_folder(deps: StoreDeps, folder_id: str, name: str) -> bool:
    deps.init_db()
    with deps.connect() as conn:
        return conn.execute(
            "UPDATE session_folders SET name=?, updated_at=? WHERE id=?", (name, deps.now(), folder_id)
        ).rowcount > 0


def delete_folder(deps: StoreDeps, folder_id: str) -> tuple[bool, str]:
    deps.init_db()
    with deps.connect() as conn:
        if conn.execute("SELECT 1 FROM session_folders WHERE id=?", (folder_id,)).fetchone() is None:
            return False, "folder_not_found"
        count = conn.execute(
            "SELECT COUNT(*) AS c FROM sessions WHERE folder_id=? AND status!='trashed'", (folder_id,)
        ).fetchone()["c"]
        if count > 0:
            return False, f"folder_not_empty({count})"
        conn.execute("DELETE FROM session_folders WHERE id=?", (folder_id,))
    return True, "deleted"


def _descendant_folder_ids(conn: sqlite3.Connection, folder_id: str) -> list[str]:
    seen = {folder_id}
    queue = [folder_id]
    while queue:
        current = queue.pop()
        for row in conn.execute("SELECT id FROM session_folders WHERE parent_id=?", (current,)).fetchall():
            child = row["id"]
            if child not in seen:
                seen.add(child)
                queue.append(child)
    return list(seen)


def move_folder(deps: StoreDeps, folder_id: str, new_parent_id: str | None) -> tuple[bool, str]:
    deps.init_db()
    with deps.connect() as conn:
        if conn.execute("SELECT 1 FROM session_folders WHERE id=?", (folder_id,)).fetchone() is None:
            return False, "folder_not_found"
        if new_parent_id is not None:
            if new_parent_id == folder_id:
                return False, "cannot_move_into_self"
            if conn.execute("SELECT 1 FROM session_folders WHERE id=?", (new_parent_id,)).fetchone() is None:
                return False, "parent_not_found"
            if new_parent_id in _descendant_folder_ids(conn, folder_id):
                return False, "cannot_move_into_descendant"
        conn.execute(
            "UPDATE session_folders SET parent_id=?, updated_at=? WHERE id=?",
            (new_parent_id, deps.now(), folder_id),
        )
    return True, "moved"


def set_folder_status(
    deps: StoreDeps,
    folder_id: str,
    status: str,
) -> tuple[bool, str]:
    if status not in {"active", "archived", "trashed"}:
        return False, "invalid_status"
    deps.init_db()
    with deps.connect() as conn:
        if conn.execute("SELECT 1 FROM session_folders WHERE id=?", (folder_id,)).fetchone() is None:
            return False, "folder_not_found"
        ids = _descendant_folder_ids(conn, folder_id)
        placeholders = ",".join("?" * len(ids))
        now = deps.now()
        lifecycle_targets: tuple[str, ...] = ()
        if status == "trashed":
            lifecycle_targets = tuple(
                str(row["id"])
                for row in conn.execute(
                    f"SELECT id FROM sessions WHERE folder_id IN ({placeholders}) "
                    "AND status='active' ORDER BY id",
                    ids,
                ).fetchall()
            )
        elif status == "active":
            lifecycle_targets = tuple(
                str(row["id"])
                for row in conn.execute(
                    f"SELECT id FROM sessions WHERE folder_id IN ({placeholders}) "
                    "AND status='trashed' ORDER BY id",
                    ids,
                ).fetchall()
            )
        for fid in ids:
            row = conn.execute("SELECT status FROM session_folders WHERE id=?", (fid,)).fetchone()
            previous = row["status"] if row else "active"
            if status == "active":
                conn.execute(
                    "UPDATE session_folders SET status='active', previous_status=NULL, updated_at=? WHERE id=?",
                    (now, fid),
                )
            else:
                conn.execute(
                    "UPDATE session_folders SET status=?, previous_status=?, updated_at=? WHERE id=?",
                    (status, previous, now, fid),
                )
        if status == "active":
            conn.execute(
                "UPDATE sessions SET status=COALESCE(previous_status, 'active'), "
                "previous_status=NULL, deleted_at=NULL "
                f"WHERE folder_id IN ({placeholders}) AND status='trashed'",
                ids,
            )
            conn.execute(
                f"UPDATE sessions SET status='active', previous_status=NULL "
                f"WHERE folder_id IN ({placeholders}) "
                "AND status='archived' AND previous_status IS NOT NULL",
                ids,
            )
        elif status == "trashed":
            if lifecycle_targets:
                target_ids = list(lifecycle_targets)
                target_placeholders = ",".join("?" for _ in target_ids)
                conn.execute(
                    "UPDATE sessions SET previous_status=status, status='trashed', "
                    "deleted_at=? "
                    f"WHERE id IN ({target_placeholders}) AND status='active'",
                    [now, *target_ids],
                )
        else:
            conn.execute(
                f"UPDATE sessions SET previous_status=status, status=? WHERE folder_id IN ({placeholders}) AND status='active'",
                [status, *ids],
            )
    return True, f"folder_{status}"


def folder_tree(deps: StoreDeps, status: str = "active") -> list[dict[str, Any]]:
    deps.init_db()
    with deps.connect() as conn:
        if status == "all":
            rows = conn.execute("SELECT * FROM session_folders ORDER BY sort_order, name").fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM session_folders WHERE status=? ORDER BY sort_order, name", (status,)
            ).fetchall()
        nodes: dict[str, dict[str, Any]] = {}
        for row in rows:
            count = conn.execute(
                "SELECT COUNT(*) AS c FROM sessions WHERE folder_id=? AND status!='trashed'", (row["id"],)
            ).fetchone()["c"]
            item = dict(row)
            item["session_count"] = count
            item["children"] = []
            nodes[row["id"]] = item
    roots: list[dict[str, Any]] = []
    for node in nodes.values():
        parent = nodes.get(node.get("parent_id"))
        if parent is None:
            roots.append(node)
        else:
            parent["children"].append(node)
    return roots
