"""分区 Session 的产品级生命周期操作。"""

from __future__ import annotations

import logging

from . import store as session_store


_LOG = logging.getLogger(__name__)


def _purge_session_retrieval_derived_state(session_id: str) -> int:
    """Remove rebuildable Session retrieval data while project routing still exists."""

    from ..retrieval.sources.session.lifecycle import purge_session_retrieval_index

    return purge_session_retrieval_index(session_id)


def _rebuild_session_retrieval_derived_state(session_id: str) -> str | None:
    from ..retrieval.sources.session.lifecycle import rebuild_session_retrieval_index

    return rebuild_session_retrieval_index(session_id, store=session_store)


def purge_session_fully(session_id: str) -> bool:
    """永久删除 Session 及其所有资源。

    已退役的 LangGraph 检查点与操作状态不再属于产品运行时契约，面向用户的删除操作会
    刻意不加载或维护这些状态。
    """
    # 先把来源置为不可读，再清项目级派生索引，最后销毁逐 Session 权威库。若派生
    # 清理失败，会话仍安全留在回收站，可由下一次 purge/restore 重试。
    with session_store.session_database_scope(session_id):
        session = session_store.get_session(session_id)
        if session is None:
            return False
        if session.get("status") != "trashed" and not session_store.trash_session(
            session_id
        ):
            return False
        _purge_session_retrieval_derived_state(session_id)
        # 附件行随 Session 删除；Project 文件由 Project 生命周期持有，不能因一个
        # Session 被清理而删除。
        return session_store.purge_session(session_id)


def empty_trash_fully() -> int:
    """永久删除回收站中的所有 Session。"""
    count = 0
    for session in session_store.list_trashed():
        if purge_session_fully(session["id"]):
            count += 1
    return count


def trash_session_fully(session_id: str) -> bool:
    """把一个 Session 移入可恢复的回收站状态。"""

    with session_store.session_database_scope(session_id):
        if not session_store.trash_session(session_id):
            return False
        _purge_session_retrieval_derived_state(session_id)
        return True


def restore_session_fully(session_id: str) -> bool:
    """从可恢复的回收站状态恢复一个 Session。"""

    with session_store.session_database_scope(session_id):
        session = session_store.get_session(session_id)
        if session is None or session.get("status") != "trashed":
            return False
        # 重试任何在 trash 后中断的派生清理；恢复后的下一 Turn 会自动重建。
        _purge_session_retrieval_derived_state(session_id)
        if not session_store.restore_session(session_id):
            return False
        # 派生索引不能改变已经成功的权威 restore 结果。失败只留在本地诊断中；下一
        # Turn 的低成本 coverage count 会识别缺口并再次触发 full backfill。
        try:
            _rebuild_session_retrieval_derived_state(session_id)
        except Exception:
            _LOG.exception("session retrieval rebuild after restore failed")
        return True
