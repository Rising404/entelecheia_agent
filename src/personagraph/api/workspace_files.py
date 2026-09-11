"""工作区文件管理（FR-8）：用户在 UI 内对会话 working_dir 里的真实文件/文件夹增删改。

安全边界：
- 所有操作**禁闭在该会话的 working_dir 内**（解析后必须仍在根内，杜绝 `../` 逃逸）。
- **删除走系统回收站**（send2trash：macOS Trash / Windows 回收站），可恢复，非永久删。

返回显式结果 dict，由上层映射为稳定 API 错误码。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..session import store as session_store
from ..workspace.binding import (
    ReservedWorkspacePathError,
    is_reserved_workspace_path,
)

_TEXT_MAX_BYTES = 2_000_000       # 编辑器读文本上限 2MB
_LIST_MAX_ENTRIES = 500


class WorkspaceFileError(Exception):
    def __init__(self, code: str, message: str, status: int = 400):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status


def _session_root(session_id: str) -> Path:
    session = session_store.get_session(session_id)
    if session is None:
        raise WorkspaceFileError("SESSION_NOT_FOUND", "会话不存在", status=404)
    working_dir = session.get("working_dir")
    if not working_dir:
        raise WorkspaceFileError("NO_WORKING_DIR", "该会话未绑定工作目录，无法管理文件", status=400)
    root = Path(str(working_dir)).expanduser().resolve()
    if not root.is_dir():
        raise WorkspaceFileError("WORKING_DIR_MISSING", "工作目录不存在或不可访问", status=400)
    return root


def _resolve_within(root: Path, rel_path: str) -> Path:
    """把相对路径解析到 root 内的绝对路径；越界 → 抛错。

    FR-9 决策①：用户是自己工作目录的主人，**不再按"敏感文件名"拦截**——用户可以看/改/删
    自己项目里的 .env 等文件。安全靠：① 禁闭在工作目录内（防逃逸）② 删除进系统回收站（可恢复）
    ③ 不能删工作目录本身。「敏感文件」这条轴只用于约束 agent，不约束用户。
    """
    rel = str(rel_path or "").strip().lstrip("/")
    target = (root / rel).expanduser().resolve()
    # 逃逸检测：解析后必须仍在 root 内（含 symlink 解析）
    if target != root and root not in target.parents:
        raise WorkspaceFileError("PATH_ESCAPES_WORKSPACE", "路径超出当前工作目录", status=400)
    _reject_agent_private_path(root, target)
    return target


def resolve_session_path(session_id: str, raw_path: str) -> Path:
    """解析 API 提供的路径，并证明它保持在 Session 根目录内。

    文件管理器端点接受相对路径，而文档摄取历史上接受绝对路径。此共享检查支持两种形式，
    同时不允许把绝对路径静默重新解释为相对路径。
    """

    root = _session_root(session_id)
    candidate = Path(str(raw_path)).expanduser()
    target = (root / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()
    if target != root and root not in target.parents:
        raise WorkspaceFileError("PATH_ESCAPES_WORKSPACE", "路径超出当前工作目录", status=400)
    _reject_agent_private_path(root, target)
    return target


def _reject_agent_private_path(root: Path, target: Path) -> None:
    """避免 Agent 托管状态进入通用用户文件管理器。

    UI 仍可自由管理 `.env` 等用户所有隐藏文件；只有 Host provision 的根级控制树是私有的。
    """

    try:
        reserved = is_reserved_workspace_path(root, target)
    except ReservedWorkspacePathError as exc:
        raise WorkspaceFileError(
            "PATH_ESCAPES_WORKSPACE", "路径超出当前工作目录", status=400
        ) from exc
    if reserved:
        raise WorkspaceFileError(
            "AGENT_PRIVATE_PATH",
            "该路径由 agent 管理，不能通过通用文件管理接口访问",
            status=403,
        )


def _entry_view(root: Path, path: Path) -> dict[str, Any]:
    try:
        stat = path.stat()
    except OSError:
        stat = None
    return {
        "name": path.name,
        "rel_path": str(path.relative_to(root)) if path != root else "",
        "kind": "dir" if path.is_dir() else "file",
        "size": stat.st_size if stat and path.is_file() else None,
        "modified_at": stat.st_mtime if stat else None,
    }


def list_dir(session_id: str, rel_path: str = "") -> dict[str, Any]:
    root = _session_root(session_id)
    target = _resolve_within(root, rel_path)
    if not target.is_dir():
        raise WorkspaceFileError("NOT_A_DIRECTORY", "不是文件夹", status=400)
    entries = []
    for child in sorted(target.iterdir(), key=lambda p: (p.is_file(), p.name.lower())):
        try:
            if is_reserved_workspace_path(root, child):
                continue
        except ReservedWorkspacePathError as exc:
            raise WorkspaceFileError(
                "PATH_ESCAPES_WORKSPACE", "路径超出当前工作目录", status=400
            ) from exc
        if len(entries) >= _LIST_MAX_ENTRIES:
            break
        entries.append(_entry_view(root, child))   # 用户可见自己目录里的全部文件（含 .env 等）
    return {"root": str(root), "rel_path": str(target.relative_to(root)) if target != root else "",
            "entries": entries}


def create_entry(session_id: str, rel_path: str, kind: str) -> dict[str, Any]:
    root = _session_root(session_id)
    target = _resolve_within(root, rel_path)
    if kind not in {"file", "dir"}:
        raise WorkspaceFileError("INVALID_KIND", "kind 只能是 file 或 dir")
    if target.exists():
        raise WorkspaceFileError("ALREADY_EXISTS", "同名文件或文件夹已存在", status=409)
    if kind == "dir":
        target.mkdir(parents=True, exist_ok=False)
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.touch()
    return {"ok": True, "entry": _entry_view(root, target)}


def read_file(session_id: str, rel_path: str) -> dict[str, Any]:
    root = _session_root(session_id)
    target = _resolve_within(root, rel_path)
    if not target.is_file():
        raise WorkspaceFileError("NOT_A_FILE", "不是文件", status=400)
    if target.stat().st_size > _TEXT_MAX_BYTES:
        raise WorkspaceFileError("FILE_TOO_LARGE", "文件过大，编辑器仅支持 2MB 内文本", status=413)
    try:
        content = target.read_text(encoding="utf-8")
    except (UnicodeDecodeError, ValueError):
        raise WorkspaceFileError("NOT_TEXT", "该文件不是可编辑的文本文件", status=415)
    return {"rel_path": str(target.relative_to(root)), "content": content}


def write_file(session_id: str, rel_path: str, content: str) -> dict[str, Any]:
    root = _session_root(session_id)
    target = _resolve_within(root, rel_path)
    if target.is_dir():
        raise WorkspaceFileError("IS_A_DIRECTORY", "目标是文件夹，无法写入", status=400)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(str(content), encoding="utf-8")
    return {"ok": True, "entry": _entry_view(root, target)}


def delete_to_trash(session_id: str, rel_path: str) -> dict[str, Any]:
    from send2trash import send2trash

    root = _session_root(session_id)
    target = _resolve_within(root, rel_path)
    if target == root:
        raise WorkspaceFileError("CANNOT_DELETE_ROOT", "不能删除工作目录本身", status=400)
    if not target.exists():
        raise WorkspaceFileError("NOT_FOUND", "文件或文件夹不存在", status=404)
    send2trash(str(target))    # → 系统回收站，可恢复
    return {"ok": True, "rel_path": rel_path, "trashed": True}
