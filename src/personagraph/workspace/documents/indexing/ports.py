"""派生 Document 索引实现必须满足的窄写入端口。"""

from __future__ import annotations

import sqlite3
from typing import Protocol, runtime_checkable

from .contracts import (
    DocumentIndexCoverageSnapshot,
    DocumentIndexEnqueueRequest,
)


@runtime_checkable
class DocumentIndexPort(Protocol):
    """在现有 Workspace authority 事务内追加仅含指针的索引事件。"""

    def enqueue_in_transaction(
        self,
        conn: sqlite3.Connection,
        request: DocumentIndexEnqueueRequest,
    ) -> tuple[str, ...]: ...


@runtime_checkable
class DocumentIndexAuditPort(Protocol):
    """校验作业链接的索引事件，并观察其精确覆盖状态。"""

    def validate_job_events_in_transaction(
        self,
        conn: sqlite3.Connection,
        *,
        event_ids: tuple[str, ...],
        document_id: str,
        document_version_id: str,
        data_version_id: str,
    ) -> None: ...

    def inspect_job_coverage_in_transaction(
        self,
        conn: sqlite3.Connection,
        *,
        event_ids: tuple[str, ...],
        document_id: str,
        document_version_id: str,
        data_version_id: str,
    ) -> DocumentIndexCoverageSnapshot: ...


__all__ = ["DocumentIndexAuditPort", "DocumentIndexPort"]
