"""Session 与文件夹生命周期操作的 Store 组合门面。

``session.store`` 保留其稳定公开导入路径。本模块只拥有生命周期类操作的兼容组合：每次
调用解析新的 ``StoreDeps``，并转发给窄持久化所有者。它刻意排除 Session 创建、工作区
权威、对话与历史、工作记忆、导出格式化和 Runtime 策略。
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .persistence.metadata import lifecycle
from .persistence.deps import StoreDeps


class SessionLifecycleStoreFacade:
    """保留 Store 生命周期 API，但不拥有持久化行为。"""

    def __init__(self, *, deps_factory: Callable[[], StoreDeps]) -> None:
        self._deps_factory = deps_factory

    def rename_session(self, session_id: str, title: str) -> bool:
        return lifecycle.rename_session(self._deps_factory(), session_id, title)

    def archive_session(self, session_id: str) -> bool:
        return lifecycle.archive_session(self._deps_factory(), session_id)

    def unarchive_session(self, session_id: str) -> bool:
        return lifecycle.unarchive_session(self._deps_factory(), session_id)

    def trash_session(self, session_id: str) -> bool:
        """把 Session 移入回收站，同时保留其先前状态以供恢复。"""

        return lifecycle.trash_session(self._deps_factory(), session_id)

    def restore_session(self, session_id: str) -> bool:
        """把回收站中的 Session 恢复到先前状态；无法确定时回退为 active。"""

        return lifecycle.restore_session(self._deps_factory(), session_id)

    def purge_session(self, session_id: str) -> bool:
        """从权威数据库删除一个 Session 所有的聚合。"""

        return lifecycle.purge_session(self._deps_factory(), session_id)

    def move_session(self, session_id: str, folder_id: str | None) -> bool:
        return lifecycle.move_session(self._deps_factory(), session_id, folder_id)

    def list_trashed(self) -> list[dict[str, Any]]:
        return lifecycle.list_trashed(self._deps_factory())

    def create_folder(self, name: str, parent_id: str | None = None) -> str:
        return lifecycle.create_folder(self._deps_factory(), name, parent_id)

    def get_folder(self, folder_id: str) -> dict[str, Any] | None:
        return lifecycle.get_folder(self._deps_factory(), folder_id)

    def list_folders(self) -> list[dict[str, Any]]:
        return lifecycle.list_folders(self._deps_factory())

    def rename_folder(self, folder_id: str, name: str) -> bool:
        return lifecycle.rename_folder(self._deps_factory(), folder_id, name)

    def delete_folder(self, folder_id: str) -> tuple[bool, str]:
        """只删除不含非回收站 Session 的文件夹。"""

        return lifecycle.delete_folder(self._deps_factory(), folder_id)

    def move_folder(
        self,
        folder_id: str,
        new_parent_id: str | None,
    ) -> tuple[bool, str]:
        """移动文件夹，由持久化所有者防止形成环。"""

        return lifecycle.move_folder(self._deps_factory(), folder_id, new_parent_id)

    def set_folder_status(
        self,
        folder_id: str,
        status: str,
    ) -> tuple[bool, str]:
        """把文件夹生命周期状态级联到后代及其 Session。"""

        return lifecycle.set_folder_status(
            self._deps_factory(),
            folder_id,
            status,
        )

    def folder_tree(self, status: str = "active") -> list[dict[str, Any]]:
        """返回嵌套文件夹树及安全 Session 计数。"""

        return lifecycle.folder_tree(self._deps_factory(), status)


def build_session_lifecycle_store_facade(
    *,
    deps_factory: Callable[[], StoreDeps],
) -> SessionLifecycleStoreFacade:
    """构建 Store 生命周期门面，暂不解析依赖。"""

    return SessionLifecycleStoreFacade(deps_factory=deps_factory)
