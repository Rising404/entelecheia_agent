"""按本机目录组织 Session 的 Project API。"""

from __future__ import annotations

from typing import Any

from ...session import project_catalog
from ...session import store as session_store
from .errors import ApiError

def _required_path(payload: dict[str, Any]) -> str:
    path = str((payload or {}).get("path") or "").strip()
    if not path:
        raise ApiError("PROJECT_PATH_REQUIRED", "缺少项目路径", status=400)
    return path


def list_projects(query: dict[str, Any] | None = None) -> dict[str, Any]:
    params = query or {}
    status = str(params.get("status") or "active")
    if status not in {"active", "archived", "trashed", "all"}:
        raise ApiError("INVALID_SESSION_STATUS", "status 取值非法", details={"status": status})

    term = str(params.get("query") or "").strip()
    sessions = session_store.list_sessions(status=status, query=term or None)
    grouped = project_catalog.group_sessions([dict(item) for item in sessions])
    if not term:
        return grouped

    # 搜索词也该能命中 project 本身：项目多起来之后，"我记得有个叫 xx 的目录"
    # 比"我记得那次对话说过什么"更常见。命中项目名或路径时，把它下面这个状态里
    # 的会话整组带出来，而不是只留下标题恰好也匹配的那几条。
    lowered = term.lower()
    matched_paths = [
        item["path"] for item in grouped["projects"]
        if lowered in item["name"].lower() or lowered in item["path"].lower()
    ]
    if matched_paths:
        whole = project_catalog.group_sessions(
            [dict(item) for item in session_store.list_sessions(status=status)]
        )
        by_path = {item["path"]: item for item in whole["projects"]}
        grouped["projects"] = [
            by_path.get(item["path"], item) if item["path"] in matched_paths else item
            for item in grouped["projects"]
        ]
    return grouped


def create_project(payload: dict[str, Any]) -> dict[str, Any]:
    """将目录登记为 Project；对已知目录重复调用也安全。"""

    project = project_catalog.remember(_required_path(payload), name=payload.get("name"))
    return {"project": {"path": project.path, "name": project.name, "pinned": project.pinned}}


def rename_project(payload: dict[str, Any]) -> dict[str, Any]:
    name = str((payload or {}).get("name") or "").strip()
    if not name:
        raise ApiError("PROJECT_NAME_REQUIRED", "缺少项目名称", status=400)
    project = project_catalog.rename(_required_path(payload), name)
    return {"project": {"path": project.path, "name": project.name, "pinned": project.pinned}}


def pin_project(payload: dict[str, Any]) -> dict[str, Any]:
    """置顶或取消置顶一个 Project。"""

    path = _required_path(payload)
    pinned = payload.get("pinned")
    if not isinstance(pinned, bool):
        raise ApiError("PROJECT_PINNED_REQUIRED", "缺少 pinned 布尔值", status=400)
    project = project_catalog.set_pinned(path, pinned)
    return {"project": {"path": project.path, "name": project.name, "pinned": project.pinned}}


def reorder_projects(payload: dict[str, Any]) -> dict[str, Any]:
    """持久化用户拖动形成的 Project 顺序。"""

    paths = (payload or {}).get("paths")
    if not isinstance(paths, list) or not paths:
        raise ApiError("PROJECT_PATHS_REQUIRED", "缺少项目路径列表", status=400)
    if not all(isinstance(item, str) for item in paths):
        raise ApiError("PROJECT_PATHS_REQUIRED", "项目路径必须是字符串", status=400)
    try:
        count = project_catalog.set_order(paths)
    except ValueError as err:
        raise ApiError("PROJECT_PATHS_INVALID", str(err), status=400) from err
    return {"ordered": count}


def forget_project(payload: dict[str, Any]) -> dict[str, Any]:
    """停止列出某目录；其 Session 不受影响，且仍保持分组。"""

    path = _required_path(payload)
    return {"forgotten": project_catalog.forget(path), "path": path}
