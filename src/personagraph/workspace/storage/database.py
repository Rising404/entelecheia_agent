"""绑定路径的单项目 Workspace 共库句柄。"""

from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
import re
import sqlite3
import threading
from typing import Iterator

from . import schema


_PROJECT_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")
_INITIALIZE_LOCK = threading.Lock()


class DocumentDatabaseError(RuntimeError):
    """无法安全打开项目 Workspace 数据库。"""


class InvalidProjectIdError(ValueError):
    """项目 ID 无法安全用作状态目录身份。"""


class DocumentDatabase:
    """``documents.sqlite`` 的显式 Workspace 项目绑定。

    调用方通过项目目录解析 ``db_path``。将该查找放在类外，可防止目录导入循环；
    更重要的是，可避免并发服务两个项目时修改进程全局路径。
    """

    def __init__(
        self,
        project_id: str,
        project_root: str | os.PathLike[str],
        db_path: str | os.PathLike[str],
    ) -> None:
        self.project_id = _validate_project_id(project_id)
        self.project_root = _normalize_path(project_root, name="project_root")
        self.db_path = _normalize_path(db_path, name="db_path")
        self._initialized = False

    def initialize(self) -> None:
        """创建或验证该项目的完整 Workspace 模式。"""

        if self._initialized:
            return
        with _INITIALIZE_LOCK:
            if self._initialized:
                return
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = self._open_connection()
            try:
                schema.initialize_schema(
                    conn,
                    project_id=self.project_id,
                    project_root=self.project_root,
                )
            except (OSError, sqlite3.Error, schema.DocumentSchemaError) as exc:
                raise DocumentDatabaseError(
                    f"failed to initialize project documents database: {self.db_path}"
                ) from exc
            finally:
                conn.close()
            self._initialized = True

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        """打开一个已初始化且拥有事务的 SQLite 连接。"""

        self.initialize()
        conn = self._open_connection()
        try:
            yield conn
        except BaseException:
            conn.rollback()
            raise
        else:
            conn.commit()
        finally:
            conn.close()

    def open_connection(self) -> sqlite3.Connection:
        """返回由调用方拥有的已初始化原始连接。

        大多数应用代码应使用 :meth:`connect`。在文档运行时整合期间，现有端口
        显式拥有提交/回滚/关闭职责的工作器适配器使用该窄桥接。
        """

        self.initialize()
        return self._open_connection()

    @contextmanager
    def connect_readonly(self) -> Iterator[sqlite3.Connection | None]:
        """读取已有库；库不存在时返回 None，不创建目录、迁移或登记数据。"""

        try:
            path = self.db_path.resolve(strict=True)
        except FileNotFoundError:
            yield None
            return
        conn = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True, timeout=5.0)
        try:
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA query_only=ON")
            yield conn
        finally:
            conn.close()

    def _open_connection(self) -> sqlite3.Connection:
        try:
            conn = sqlite3.connect(self.db_path, timeout=5.0)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute("PRAGMA busy_timeout = 5000")
            schema.load_sqlite_vec(conn)
            return conn
        except (OSError, sqlite3.Error) as exc:
            raise DocumentDatabaseError(
                f"failed to open project documents database: {self.db_path}"
            ) from exc


def _validate_project_id(value: str) -> str:
    if not isinstance(value, str) or _PROJECT_ID_PATTERN.fullmatch(value) is None:
        raise InvalidProjectIdError(
            "project_id must be 1-128 ASCII letters, digits, underscores, or hyphens "
            "and must start with a letter or digit"
        )
    return value


def _normalize_path(
    value: str | os.PathLike[str],
    *,
    name: str,
) -> Path:
    if isinstance(value, str) and not value.strip():
        raise ValueError(f"{name} must not be empty")
    try:
        path = Path(value).expanduser().resolve(strict=False)
    except (OSError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a valid filesystem path") from exc
    return path
