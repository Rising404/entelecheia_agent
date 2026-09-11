"""保留旧版面向路径 API 的稳定项目目录。

在此迁移阶段，Session 仍通过 ``working_dir`` 指定项目。目录会把该路径转换为持久项目标识，
并保留面向用户的名称、置顶状态与顺序。详细的 Session 和文档数据不属于此处。
"""

from __future__ import annotations

import json
import os
import sqlite3
import stat
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from ..configuration.paths import PROJECT_CATALOG_DB_PATH as _PROJECT_CATALOG_DB_PATH
from ..configuration.paths import project_documents_db_path
from ..configuration.paths import resolved_project_catalog_path


# 保持可变，以支持测试隔离及已经修补 DB_PATH 的调用方。
DB_PATH = _PROJECT_CATALOG_DB_PATH

MAX_NAME = 120

_CATALOG_SCHEMA_VERSION = 1
_SQLITE_BUSY_TIMEOUT_MS = 5_000
_UNORDERED = 1 << 30
_MAX_MANIFEST_BYTES = 64 * 1024
_FALLBACK_ID_NAMESPACE = uuid.UUID("6cc5ec1b-fbad-46bc-a9c2-dc952815350c")


@dataclass(frozen=True, slots=True)
class Project:
    # 前四个字段保留旧版位置参数构造契约。
    path: str
    name: str
    pinned: bool = False
    order: int | None = None
    project_id: str = ""
    canonical_root: str = ""
    documents_db_path: str = ""

    def to_public(self, *, sessions: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "project_id": self.project_id,
            "path": self.path,
            "canonical_root": self.canonical_root,
            "documents_db_path": self.documents_db_path,
            "name": self.name,
            "pinned": self.pinned,
            "sessions": sessions,
        }


def _path() -> Path:
    return resolved_project_catalog_path(DB_PATH)


def _connect() -> sqlite3.Connection:
    target = _path()
    target.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(
        target,
        timeout=_SQLITE_BUSY_TIMEOUT_MS / 1_000,
        isolation_level=None,
    )
    connection.row_factory = sqlite3.Row
    connection.execute(f"PRAGMA busy_timeout = {_SQLITE_BUSY_TIMEOUT_MS}")
    connection.execute("PRAGMA journal_mode = WAL")
    connection.execute("PRAGMA synchronous = NORMAL")

    version = int(connection.execute("PRAGMA user_version").fetchone()[0])
    if version not in {0, _CATALOG_SCHEMA_VERSION}:
        connection.close()
        raise RuntimeError(f"unsupported project catalog schema version: {version}")
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS projects (
            project_id TEXT PRIMARY KEY,
            path TEXT NOT NULL,
            canonical_root TEXT NOT NULL UNIQUE,
            name TEXT NOT NULL,
            pinned INTEGER NOT NULL DEFAULT 0 CHECK (pinned IN (0, 1)),
            sort_order INTEGER CHECK (sort_order IS NULL OR sort_order >= 0)
        )
        """
    )
    if version == 0:
        connection.execute(f"PRAGMA user_version = {_CATALOG_SCHEMA_VERSION}")
    return connection


@contextmanager
def _write_catalog() -> Iterator[sqlite3.Connection]:
    connection = _connect()
    try:
        connection.execute("BEGIN IMMEDIATE")
        yield connection
        connection.commit()
    except BaseException:
        connection.rollback()
        raise
    finally:
        connection.close()


def _clean_path(path: str) -> str:
    cleaned = str(path or "").strip()
    if not cleaned:
        raise ValueError("project path must not be empty")
    return cleaned


def _canonicalize_root(path: str) -> str:
    cleaned = _clean_path(path)
    try:
        supplied = Path(cleaned).expanduser()
        if not supplied.is_absolute():
            supplied = Path.cwd() / supplied
        return os.path.normcase(str(supplied.resolve(strict=False)))
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise ValueError("project path is invalid") from exc


def default_name(path: str) -> str:
    """文件夹自身的名称，也就是用户对该项目的称呼。"""

    return Path(path).name or path


def _provided_name(name: str | None) -> str | None:
    cleaned = str(name or "").strip()
    return cleaned[:MAX_NAME] if cleaned else None


def _workspace_project_id(canonical_root: str) -> str | None:
    """复用有效的工作区清单 ID，而不配置新文件。"""

    root = Path(canonical_root)
    manifest_path = root / ".personagraph" / "manifest.json"
    try:
        root_facts = os.stat(root)
        manifest_facts = os.lstat(manifest_path)
        if not stat.S_ISDIR(root_facts.st_mode) or not stat.S_ISREG(manifest_facts.st_mode):
            return None
        if manifest_facts.st_size > _MAX_MANIFEST_BYTES:
            return None
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(manifest_path, flags)
        try:
            opened_facts = os.fstat(descriptor)
            if not stat.S_ISREG(opened_facts.st_mode) or opened_facts.st_size > _MAX_MANIFEST_BYTES:
                return None
            with os.fdopen(descriptor, "r", encoding="utf-8") as stream:
                descriptor = -1
                manifest = json.load(stream)
        finally:
            if descriptor >= 0:
                os.close(descriptor)
    except (OSError, UnicodeError, ValueError, TypeError):
        return None

    identity = manifest.get("root_identity") if isinstance(manifest, dict) else None
    workspace_id = manifest.get("workspace_id") if isinstance(manifest, dict) else None
    if not isinstance(identity, dict) or not isinstance(workspace_id, str):
        return None
    if identity.get("device") != root_facts.st_dev or identity.get("inode") != root_facts.st_ino:
        return None
    try:
        project_documents_db_path(workspace_id)
    except ValueError:
        return None
    return workspace_id


def _new_project_id(canonical_root: str, *, deterministic: bool = False) -> str:
    workspace_id = _workspace_project_id(canonical_root)
    if workspace_id:
        return workspace_id
    identity = uuid.uuid5(_FALLBACK_ID_NAMESPACE, canonical_root) if deterministic else uuid.uuid4()
    return f"project-{identity.hex}"


def _row_to_project(row: sqlite3.Row) -> Project:
    project_id = str(row["project_id"])
    return Project(
        path=str(row["path"]),
        name=str(row["name"]),
        pinned=bool(row["pinned"]),
        order=_as_order(row["sort_order"]),
        project_id=project_id,
        canonical_root=str(row["canonical_root"]),
        documents_db_path=str(project_documents_db_path(project_id)),
    )


def _select_by_root(connection: sqlite3.Connection, canonical_root: str) -> sqlite3.Row | None:
    return connection.execute(
        "SELECT * FROM projects WHERE canonical_root = ?",
        (canonical_root,),
    ).fetchone()


def _ensure_project(
    connection: sqlite3.Connection,
    *,
    path: str,
    canonical_root: str,
    name: str | None = None,
) -> Project:
    row = _select_by_root(connection, canonical_root)
    chosen_name = _provided_name(name)
    if row is not None:
        if chosen_name is not None and row["name"] != chosen_name:
            connection.execute(
                "UPDATE projects SET name = ? WHERE project_id = ?",
                (chosen_name, row["project_id"]),
            )
            row = _select_by_root(connection, canonical_root)
        assert row is not None
        return _row_to_project(row)

    project_id = _new_project_id(canonical_root)
    row_with_same_id = connection.execute(
        "SELECT * FROM projects WHERE project_id = ?",
        (project_id,),
    ).fetchone()
    if row_with_same_id is not None:
        # 同一目录 inode 被移动时，经验证的清单 ID 会跟随工作区。
        # 复制的清单则会被标识检查拒绝。
        connection.execute(
            "UPDATE projects SET path = ?, canonical_root = ?, name = COALESCE(?, name) "
            "WHERE project_id = ?",
            (path, canonical_root, chosen_name, project_id),
        )
    else:
        connection.execute(
            """
            INSERT INTO projects(project_id, path, canonical_root, name, pinned, sort_order)
            VALUES (?, ?, ?, ?, 0, NULL)
            """,
            (project_id, path, canonical_root, chosen_name or default_name(path)),
        )
    row = _select_by_root(connection, canonical_root)
    assert row is not None
    return _row_to_project(row)


def remember(path: str, *, name: str | None = None) -> Project:
    """幂等注册一个目录，并返回其稳定标识。"""

    cleaned = _clean_path(path)
    canonical_root = _canonicalize_root(cleaned)
    with _write_catalog() as connection:
        return _ensure_project(
            connection,
            path=cleaned,
            canonical_root=canonical_root,
            name=name,
        )


def get_by_id(project_id: str) -> Project | None:
    """解析一个稳定项目标识，而不从路径推导。"""

    cleaned = str(project_id or "").strip()
    if not cleaned:
        raise ValueError("project_id must not be empty")
    with _connect() as connection:
        row = connection.execute(
            "SELECT * FROM projects WHERE project_id = ?",
            (cleaned,),
        ).fetchone()
    return _row_to_project(row) if row is not None else None


def get_by_path(path: str) -> Project | None:
    """根据规范文件系统根目录解析一个已注册项目。"""

    canonical_root = _canonicalize_root(path)
    with _connect() as connection:
        row = _select_by_root(connection, canonical_root)
    return _row_to_project(row) if row is not None else None


def list_registered_projects() -> tuple[Project, ...]:
    """列出所有持久化 Project locator，不读取 Session 或文档内容。"""

    with _connect() as connection:
        rows = connection.execute(
            "SELECT * FROM projects ORDER BY project_id"
        ).fetchall()
    return tuple(_row_to_project(row) for row in rows)


def rename(path: str, name: str) -> Project:
    cleaned = str(name or "").strip()
    if not cleaned:
        raise ValueError("project name must not be empty")
    return remember(path, name=cleaned)


def forget(path: str) -> bool:
    """清除呈现元数据，但不销毁稳定标识。"""

    canonical_root = _canonicalize_root(path)
    with _write_catalog() as connection:
        row = _select_by_root(connection, canonical_root)
        if row is None:
            return False
        connection.execute(
            "UPDATE projects SET name = ?, pinned = 0, sort_order = NULL WHERE project_id = ?",
            (default_name(str(row["path"])), row["project_id"]),
        )
        return True


def _fallback_project(path: str, canonical_root: str) -> Project:
    project_id = _new_project_id(canonical_root, deterministic=True)
    return Project(
        path=path,
        name=default_name(path),
        project_id=project_id,
        canonical_root=canonical_root,
        documents_db_path=str(project_documents_db_path(project_id)),
    )


def group_sessions(sessions: list[dict[str, Any]]) -> dict[str, Any]:
    """按规范项目根目录汇集 Session，同时保留旧有结构。"""

    grouped: dict[str, dict[str, Any]] = {}
    unbound: list[dict[str, Any]] = []
    for session in sessions:
        directory = str(session.get("working_dir") or "").strip()
        if not directory:
            unbound.append(session)
            continue
        try:
            canonical_root = _canonicalize_root(directory)
        except ValueError:
            # 格式错误的历史行不得导致其 Session 消失。
            canonical_root = directory
        bucket = grouped.setdefault(
            canonical_root,
            {"path": directory, "sessions": []},
        )
        bucket["sessions"].append(session)

    projects_by_root: dict[str, Project] = {}
    try:
        with _write_catalog() as connection:
            for canonical_root, bucket in grouped.items():
                projects_by_root[canonical_root] = _ensure_project(
                    connection,
                    path=str(bucket["path"]),
                    canonical_root=canonical_root,
                )
    except (OSError, RuntimeError, sqlite3.Error):
        # 不允许目录呈现元数据隐藏 Session 历史。
        projects_by_root = {
            canonical_root: _fallback_project(str(bucket["path"]), canonical_root)
            for canonical_root, bucket in grouped.items()
        }

    projects: list[dict[str, Any]] = []
    for canonical_root, bucket in grouped.items():
        project = projects_by_root[canonical_root]
        public = project.to_public(sessions=bucket["sessions"])
        public["order"] = project.order
        projects.append(public)
    projects.sort(
        key=lambda item: (
            not item["pinned"],
            _UNORDERED if item["order"] is None else item["order"],
            item["name"],
        )
    )
    return {"projects": projects, "unbound": unbound}


def _as_order(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def set_order(paths: list[str]) -> int:
    """持久化调用方提供的完整可见路径顺序。"""

    cleaned = [_clean_path(path) for path in paths if str(path or "").strip()]
    roots = [_canonicalize_root(path) for path in cleaned]
    if len(set(roots)) != len(roots):
        raise ValueError("project paths must be unique")
    with _write_catalog() as connection:
        for index, (path, canonical_root) in enumerate(zip(cleaned, roots, strict=True)):
            project = _ensure_project(
                connection,
                path=path,
                canonical_root=canonical_root,
            )
            connection.execute(
                "UPDATE projects SET sort_order = ? WHERE project_id = ?",
                (index, project.project_id),
            )
    return len(cleaned)


def set_pinned(path: str, pinned: bool) -> Project:
    """将项目置顶到列表顶部，或取消置顶。"""

    cleaned = _clean_path(path)
    canonical_root = _canonicalize_root(cleaned)
    with _write_catalog() as connection:
        project = _ensure_project(
            connection,
            path=cleaned,
            canonical_root=canonical_root,
        )
        connection.execute(
            "UPDATE projects SET pinned = ? WHERE project_id = ?",
            (int(bool(pinned)), project.project_id),
        )
        row = _select_by_root(connection, canonical_root)
        assert row is not None
        return _row_to_project(row)
