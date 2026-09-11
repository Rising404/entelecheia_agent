"""SQLite 持久化记录之上的 Session 与文件夹生命周期门面。

``session.store`` 仍是 API 和旧版调用方使用的兼容接口。本模块拥有组合多项持久化
记录的生命周期类操作，包括文件夹成员关系和状态变化。它只依赖 ``StoreDeps`` 和相邻
持久化模块，绝不依赖公开门面。
"""

from __future__ import annotations

from typing import Any

from . import folders, sessions
from ..deps import StoreDeps


def rename_session(deps: StoreDeps, session_id: str, title: str) -> bool:
    return sessions.rename_session(deps, session_id, title)


def archive_session(deps: StoreDeps, session_id: str) -> bool:
    return sessions.archive_session(deps, session_id)


def unarchive_session(deps: StoreDeps, session_id: str) -> bool:
    return sessions.unarchive_session(deps, session_id)


def trash_session(deps: StoreDeps, session_id: str) -> bool:
    return sessions.trash_session(deps, session_id)


def restore_session(deps: StoreDeps, session_id: str) -> bool:
    return sessions.restore_session(deps, session_id)


def purge_session(deps: StoreDeps, session_id: str) -> bool:
    return sessions.purge_session(deps, session_id)


def move_session(
    deps: StoreDeps,
    session_id: str,
    folder_id: str | None,
) -> bool:
    """仅当请求的目标存在时移动一个 Session。"""

    deps.init_db()
    if folder_id and folders.get_folder(deps, folder_id) is None:
        return False
    with deps.connect() as conn:
        cursor = conn.execute(
            "UPDATE sessions SET folder_id=? WHERE id=?",
            (folder_id, session_id),
        )
        return cursor.rowcount > 0


def list_trashed(deps: StoreDeps) -> list[dict[str, Any]]:
    return sessions.list_trashed(deps)


def create_folder(
    deps: StoreDeps,
    name: str,
    parent_id: str | None = None,
) -> str:
    return folders.create_folder(deps, name, parent_id)


def get_folder(deps: StoreDeps, folder_id: str) -> dict[str, Any] | None:
    return folders.get_folder(deps, folder_id)


def list_folders(deps: StoreDeps) -> list[dict[str, Any]]:
    return folders.list_folders(deps)


def rename_folder(deps: StoreDeps, folder_id: str, name: str) -> bool:
    return folders.rename_folder(deps, folder_id, name)


def delete_folder(deps: StoreDeps, folder_id: str) -> tuple[bool, str]:
    return folders.delete_folder(deps, folder_id)


def move_folder(
    deps: StoreDeps,
    folder_id: str,
    new_parent_id: str | None,
) -> tuple[bool, str]:
    return folders.move_folder(deps, folder_id, new_parent_id)


def set_folder_status(
    deps: StoreDeps,
    folder_id: str,
    status: str,
) -> tuple[bool, str]:
    return folders.set_folder_status(deps, folder_id, status)


def folder_tree(deps: StoreDeps, status: str = "active") -> list[dict[str, Any]]:
    return folders.folder_tree(deps, status)
