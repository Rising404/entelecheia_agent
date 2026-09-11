"""文件准备过程依赖的派生索引端口，不暴露检索存储和 generation 状态机。"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import sqlite3
from typing import Protocol

from .storage import DocumentIngestCoverageProof


@dataclass(frozen=True, slots=True)
class IngestionGenerationIdentity:
    """准备执行者绑定的索引配方身份；不是索引可查询状态。"""

    version_id: str
    fingerprint: str


class IngestionIndexPort(Protocol):
    """来源提交事务与可重建索引之间唯一的执行接缝。

    事务方法必须使用传入连接，不得另开连接或提交；其余方法只改变派生索引。
    不可恢复和暂时失败分别使用 worker_errors 中的终止/重试错误。
    """

    @property
    def generation_identity(self) -> IngestionGenerationIdentity: ...

    def select_source_target(self) -> str: ...

    def refresh_source_target(self, target_id: str) -> str: ...

    def require_write_target_in_transaction(
        self, conn: sqlite3.Connection, target_id: str,
    ) -> str: ...

    def ensure_source_events_in_transaction(
        self,
        conn: sqlite3.Connection,
        *,
        target_id: str,
        document_id: str,
        authorization_session_id: str,
        receipt_event_ids: tuple[str, ...],
    ) -> tuple[str, ...]: ...

    def index_and_prove(
        self,
        *,
        target_id: str | None,
        document_id: str | None,
        document_version_id: str | None,
        event_ids: tuple[str, ...],
        worker_id: str,
        now: Callable[[], str],
        lease_seconds: int,
        batch_limit: int,
        max_batches: int,
    ) -> DocumentIngestCoverageProof: ...

    def publish_proven_target(
        self,
        *,
        target_id: str | None,
        document_id: str | None,
        document_version_id: str | None,
        expected_binding_digest: str | None,
        expected_binding_count: int | None,
        expected_generation_fingerprint: str | None,
        on_transition: Callable[[str], None],
    ) -> None: ...

    def drain_once(
        self,
        *,
        worker_id: str,
        now: str,
        lease_seconds: int,
        limit: int,
    ) -> int: ...


__all__ = ["IngestionGenerationIdentity", "IngestionIndexPort"]
