"""Workspace 文件 HTTP 服务：验证与稳定 API 错误转换。"""

from __future__ import annotations

from typing import Any

from .. import workspace_files
from .common import empty_to_none, required_str
from .errors import ApiError

__all__ = [
    "list_workspace_files",
    "read_workspace_file",
    "create_workspace_entry",
    "write_workspace_file",
    "delete_workspace_entry",
]


def _call(operation, *args: Any) -> dict[str, Any]:
    """将适配器的 workspace 失败转换为公开 API 错误。"""

    try:
        return operation(*args)
    except workspace_files.WorkspaceFileError as exc:
        raise ApiError(exc.code, exc.message, status=exc.status) from exc


def list_workspace_files(
    session_id: str,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    relative_path = empty_to_none((params or {}).get("path")) or ""
    return _call(workspace_files.list_dir, session_id, relative_path)


def read_workspace_file(
    session_id: str,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    relative_path = required_str(params or {}, "path")
    return _call(workspace_files.read_file, session_id, relative_path)


def create_workspace_entry(
    session_id: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    relative_path = required_str(payload, "path")
    kind = str(payload.get("kind") or "file")
    return _call(workspace_files.create_entry, session_id, relative_path, kind)


def write_workspace_file(
    session_id: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    relative_path = required_str(payload, "path")
    content = str(payload.get("content", ""))
    return _call(workspace_files.write_file, session_id, relative_path, content)


def delete_workspace_entry(
    session_id: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    relative_path = required_str(payload, "path")
    return _call(workspace_files.delete_to_trash, session_id, relative_path)
