"""API 注入现有 Project 维护轮询的视觉发布确认，不创建后台执行器。"""

from __future__ import annotations

from collections.abc import Callable
import logging

from ...runtime.model_calls.vision import SqliteMountedVisualCallLedger
from ...session.catalog import SessionCatalog
from ...session.store import session_database_route_scope
from ...workspace.storage.context import current as current_project_database
from ...workspace.storage.database import DocumentDatabase
from .project_observation_publication import VisualObservationPublisher


_LOG = logging.getLogger(__name__)


def build_visual_publication_recovery(database: DocumentDatabase) -> Callable[[], int]:
    """只扫描当前 Project 的既有 Session 回执；无文件解析、视觉请求或索引等待。"""

    def recover() -> int:
        if current_project_database() is not database:
            raise RuntimeError("visual publication recovery requires its project scope")
        recovered = 0
        for session in SessionCatalog().list_sessions(project_id=database.project_id):
            session_id = str(session["id"])
            try:
                with session_database_route_scope(session_id):
                    ledger = SqliteMountedVisualCallLedger()
                    if not ledger.path_for(session_id).is_file():
                        continue
                    recovered += VisualObservationPublisher(
                        session_id=session_id, call_ledger=ledger,
                    ).recover_ready()
            except Exception as exc:
                # 单会话路由/存储异常不阻断同 Project 其他回执；不得记录路径或正文。
                _LOG.warning(
                    "visual_publication_recovery_failed error_type=%s", type(exc).__name__,
                )
        return recovered

    return recover
