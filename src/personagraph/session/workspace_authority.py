"""当前 Session 绑定目录的文件访问检查，不缓存授权事实。"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from personagraph.workspace.binding import (
    ReservedWorkspacePathError,
    is_reserved_workspace_path,
)


def validate_current_session_workspace_authority(
    session_id: str,
    canonical_path: str,
) -> bool:
    """Reopen Session authority for one canonical Project path; never cache it."""

    if not session_id.strip() or not canonical_path.strip():
        return False
    try:
        from . import store as session_store

        session = session_store.get_session(session_id)
        if session is None or str(session.get("status") or "") == "trashed":
            return False
        source_path_input = Path(canonical_path).expanduser()
        if not source_path_input.is_absolute():
            return False
        source_path = source_path_input.resolve()
        if not _within_session_working_dir(session, source_path):
            return False
        root = Path(str(session["working_dir"])).expanduser().resolve()
        return not is_reserved_workspace_path(root, source_path)
    except (OSError, ReservedWorkspacePathError, RuntimeError, ValueError):
        return False


def _within_session_working_dir(
    session: Mapping[str, Any],
    source_path: Path,
) -> bool:
    working_dir = session.get("working_dir")
    if not working_dir:
        return False
    root = Path(str(working_dir)).expanduser().resolve()
    if not root.is_dir():
        return False
    try:
        source_path.relative_to(root)
    except ValueError:
        return False
    return True


__all__ = ["validate_current_session_workspace_authority"]
