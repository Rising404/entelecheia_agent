"""单个执行上下文的显式 Workspace 数据库绑定。"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
import sqlite3
from typing import Iterator

from .database import DocumentDatabase


class ProjectDocumentContextError(RuntimeError):
    """Workspace 项目绑定将产生歧义。"""


_CURRENT: ContextVar[DocumentDatabase | None] = ContextVar(
    "personagraph_project_document_database",
    default=None,
)


def current() -> DocumentDatabase | None:
    """返回绑定到当前执行上下文的数据库（若有）。"""

    return _CURRENT.get()


def require_current() -> DocumentDatabase:
    """返回当前 Workspace 数据库；未绑定时以类型化错误关闭。"""

    database = current()
    if database is None:
        raise ProjectDocumentContextError(
            "workspace storage requires a bound project database"
        )
    return database


def initialize_current() -> None:
    """初始化当前 Workspace 的共享数据库。"""

    require_current().initialize()


def connect_current():
    """返回当前 Workspace 数据库的事务 context manager。"""

    return require_current().connect()


def open_current_connection() -> sqlite3.Connection:
    """打开一个由调用方负责提交、回滚和关闭的当前 Workspace 连接。"""

    return require_current().open_connection()


def current_database_path() -> Path:
    """返回当前 Workspace 权威数据库路径。"""

    return require_current().db_path


@contextmanager
def bind(database: DocumentDatabase) -> Iterator[DocumentDatabase]:
    """只绑定一个项目数据库，并在结束后恢复此前的空上下文。

    即使两个句柄指向同一项目，也拒绝嵌套项目绑定。静默遮蔽会让默认构造的
    检索适配器依赖调用栈位置，而不是显式权威。
    """

    if not isinstance(database, DocumentDatabase):
        raise TypeError("database must be a DocumentDatabase")
    if _CURRENT.get() is not None:
        raise ProjectDocumentContextError("project document context is already bound")
    token = _CURRENT.set(database)
    try:
        yield database
    finally:
        _CURRENT.reset(token)
