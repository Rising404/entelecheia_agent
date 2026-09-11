"""从权威侧 Outbox 事件到派生检索数据的幂等桥。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
import sqlite3
from typing import Protocol

from ..contracts import RetrievalStatus, RetrievalUnit, SourceFilter, SourceType, SourceUnitRef
from ..indexing.encoder import RetrievalMethodUnavailable
from .outbox import (
    RetrievalUpdateEvent,
    RetrievalUpdateKind,
    SqliteRetrievalOutbox,
)
from ..sqlite_store import (
    SqliteRetrievalCatalog,
    StoredUnit,
    SyncReceiptStatus,
    UnitIndexState,
)


@dataclass(frozen=True, slots=True)
class IndexableSourceUnit:
    """用于生成派生方法索引的临时权威文本。"""

    ref: SourceUnitRef
    source_filter: SourceFilter
    content: str

    def __post_init__(self) -> None:
        if self.ref.source_type is not self.source_filter.source_type:
            raise ValueError("source_filter must match IndexableSourceUnit ref")
        if not self.content.strip():
            raise ValueError("indexable source content must not be empty")


class SyncSourceReader(Protocol):
    source_type: SourceType

    def read_for_index(self, event: RetrievalUpdateEvent) -> IndexableSourceUnit | None: ...


class ReconciliationSourceReader(SyncSourceReader, Protocol):
    """使用稳定来源身份读取当前权威 Unit。

    同步必须要求精确 ``SourceUnitRef``，使延迟事件绝不会意外索引较新的 revision。协调
    承担不同职责：把已索引引用与当前来源比较；因此它需要显式只读的当前状态探针。
    """

    def read_current_for_reconcile(self, ref: SourceUnitRef) -> IndexableSourceUnit | None: ...


class RetrievalIndexWriter(Protocol):
    """写入或移除一个 Unit 的所有方法专用派生表示。"""

    def index(self, stored_unit: StoredUnit, source_unit: IndexableSourceUnit) -> None: ...

    def purge(self, stored_unit: StoredUnit) -> None: ...


class SyncApplyStatus(StrEnum):
    APPLIED = "applied"
    RETRYABLE_FAILED = "retryable_failed"
    TERMINAL_FAILED = "terminal_failed"


@dataclass(frozen=True, slots=True)
class SyncResult:
    event_id: str
    status: SyncApplyStatus
    reason_code: str | None = None
    replayed: bool = False
    failure_stage: str | None = None
    safe_error_code: str | None = None


class SyncTerminalError(RuntimeError):
    pass


class RetrievalSyncService:
    """应用检索事件，但不让派生数据成为来源权威。"""

    def __init__(
        self,
        *,
        catalog: SqliteRetrievalCatalog,
        source_readers: Mapping[SourceType, SyncSourceReader],
        index_writer: RetrievalIndexWriter,
    ) -> None:
        self._catalog = catalog
        self._source_readers = dict(source_readers)
        self._index_writer = index_writer

    @property
    def source_types(self) -> tuple[SourceType, ...]:
        """此同步器可以实例化的固定来源权威。"""

        return tuple(sorted(self._source_readers, key=lambda item: item.value))

    def apply(self, event: RetrievalUpdateEvent) -> SyncResult:
        existing = self._catalog.get_receipt(event.event_id)
        if (
            existing is not None
            and existing.status is SyncReceiptStatus.APPLIED
            and self._applied_effect_is_present(event)
        ):
            return SyncResult(event.event_id, SyncApplyStatus.APPLIED, replayed=True)

        self._catalog.start_receipt(event.event_id)
        stored_unit: StoredUnit | None = None
        try:
            if event.ref.source_type not in self._source_readers:
                raise SyncTerminalError("source_type_not_allowed")
            if event.kind is RetrievalUpdateKind.UPSERT:
                stored_unit = self._upsert(event)
            elif event.kind is RetrievalUpdateKind.TRASH:
                stored_unit = self._catalog.set_unit_retrieval_status(
                    event.ref,
                    event.retrieval_data_version,
                    RetrievalStatus.TRASHED,
                )
            elif event.kind is RetrievalUpdateKind.RESTORE:
                stored_unit = self._restore(event)
            elif event.kind is RetrievalUpdateKind.PURGE:
                stored_unit = self._purge(event)
            else:  # 防御未来的枚举扩展
                raise SyncTerminalError("unsupported_event_kind")
        except SyncTerminalError as exc:
            reason_code = str(exc)
            self._catalog.finish_receipt(
                event.event_id,
                status=SyncReceiptStatus.TERMINAL_FAILED,
                reason_code=reason_code,
            )
            return SyncResult(
                event.event_id,
                SyncApplyStatus.TERMINAL_FAILED,
                reason_code,
                failure_stage="source_validation",
                safe_error_code=reason_code,
            )
        except RetrievalMethodUnavailable as exc:
            if stored_unit is not None:
                self._catalog.mark_unit_index_failed(stored_unit.unit_id)
            reason_code = "retrieval_method_unavailable"
            self._catalog.finish_receipt(
                event.event_id,
                status=SyncReceiptStatus.TERMINAL_FAILED,
                reason_code=reason_code,
            )
            return SyncResult(
                event.event_id,
                SyncApplyStatus.TERMINAL_FAILED,
                reason_code,
                failure_stage=exc.stage or "index_write",
                safe_error_code=exc.safe_error_code,
            )
        except Exception as exc:
            if stored_unit is not None:
                self._catalog.mark_unit_index_failed(stored_unit.unit_id)
            reason_code = f"sync_exception:{type(exc).__name__}"
            self._catalog.finish_receipt(
                event.event_id,
                status=SyncReceiptStatus.RETRYABLE_FAILED,
                reason_code=reason_code,
            )
            return SyncResult(
                event.event_id,
                SyncApplyStatus.RETRYABLE_FAILED,
                reason_code,
                failure_stage="sync_apply",
                safe_error_code=reason_code,
            )

        del stored_unit
        self._catalog.finish_receipt(event.event_id, status=SyncReceiptStatus.APPLIED)
        return SyncResult(event.event_id, SyncApplyStatus.APPLIED)

    def _applied_effect_is_present(self, event: RetrievalUpdateEvent) -> bool:
        """Do not let a stale APPLIED receipt conceal deleted/partial derived state."""

        stored = self._catalog.get_unit(event.ref, event.retrieval_data_version)
        if event.kind is RetrievalUpdateKind.PURGE:
            return stored is None
        if stored is None:
            return False
        if event.kind is RetrievalUpdateKind.TRASH:
            return stored.unit.retrieval_status is RetrievalStatus.TRASHED
        if event.kind in {RetrievalUpdateKind.UPSERT, RetrievalUpdateKind.RESTORE}:
            return (
                stored.unit.retrieval_status is RetrievalStatus.ACTIVE
                and stored.index_state is UnitIndexState.READY
            )
        return False

    def repair(self, event: RetrievalUpdateEvent) -> SyncResult:
        """安全重建一个现有 Unit 的派生表示。

        修复刻意显式执行，而不是重放已经应用的权威事件。Unit 会在重建索引前离开可搜索
        集合，因此部分重写的表示绝不会返回给调用方。
        """

        existing = self._catalog.get_receipt(event.event_id)
        if existing is not None and existing.status is SyncReceiptStatus.APPLIED:
            return SyncResult(event.event_id, SyncApplyStatus.APPLIED, replayed=True)

        self._catalog.start_receipt(event.event_id)
        stored_unit: StoredUnit | None = None
        try:
            if event.ref.source_type not in self._source_readers:
                raise SyncTerminalError("source_type_not_allowed")
            stored_unit = self._catalog.get_unit(event.ref, event.retrieval_data_version)
            if stored_unit is None:
                raise SyncTerminalError("repair_unit_missing")
            source_unit = self._read_source_unit(event)
            if source_unit.source_filter != stored_unit.unit.source_filter:
                raise SyncTerminalError("repair_source_scope_mismatch")
            self._catalog.mark_unit_index_pending(stored_unit.unit_id)
            self._index_pending_unit(stored_unit, source_unit)
        except SyncTerminalError as exc:
            reason_code = str(exc)
            self._catalog.finish_receipt(
                event.event_id,
                status=SyncReceiptStatus.TERMINAL_FAILED,
                reason_code=reason_code,
            )
            return SyncResult(
                event.event_id,
                SyncApplyStatus.TERMINAL_FAILED,
                reason_code,
                failure_stage="source_validation",
                safe_error_code=reason_code,
            )
        except RetrievalMethodUnavailable as exc:
            if stored_unit is not None:
                self._catalog.mark_unit_index_failed(stored_unit.unit_id)
            reason_code = "retrieval_method_unavailable"
            self._catalog.finish_receipt(
                event.event_id,
                status=SyncReceiptStatus.TERMINAL_FAILED,
                reason_code=reason_code,
            )
            return SyncResult(
                event.event_id,
                SyncApplyStatus.TERMINAL_FAILED,
                reason_code,
                failure_stage=exc.stage or "index_write",
                safe_error_code=exc.safe_error_code,
            )
        except Exception as exc:
            if stored_unit is not None:
                self._catalog.mark_unit_index_failed(stored_unit.unit_id)
            reason_code = f"repair_exception:{type(exc).__name__}"
            self._catalog.finish_receipt(
                event.event_id,
                status=SyncReceiptStatus.RETRYABLE_FAILED,
                reason_code=reason_code,
            )
            return SyncResult(
                event.event_id,
                SyncApplyStatus.RETRYABLE_FAILED,
                reason_code,
                failure_stage="repair",
                safe_error_code=reason_code,
            )

        self._catalog.finish_receipt(event.event_id, status=SyncReceiptStatus.APPLIED)
        return SyncResult(event.event_id, SyncApplyStatus.APPLIED)

    def _upsert(self, event: RetrievalUpdateEvent) -> StoredUnit:
        source_unit = self._read_source_unit(event)
        stored_unit = self._catalog.upsert_pending_unit(
            RetrievalUnit(
                ref=source_unit.ref,
                retrieval_data_version=event.retrieval_data_version,
                retrieval_status=RetrievalStatus.ACTIVE,
                source_filter=source_unit.source_filter,
            )
        )
        if stored_unit.index_state is not UnitIndexState.READY:
            self._index_pending_unit(stored_unit, source_unit)
            stored_unit = self._require_stored_unit(event)
        return stored_unit

    def _restore(self, event: RetrievalUpdateEvent) -> StoredUnit:
        source_unit = self._read_source_unit(event)
        stored_unit = self._catalog.get_unit(event.ref, event.retrieval_data_version)
        if stored_unit is None:
            stored_unit = self._catalog.upsert_pending_unit(
                RetrievalUnit(
                    ref=source_unit.ref,
                    retrieval_data_version=event.retrieval_data_version,
                    retrieval_status=RetrievalStatus.ACTIVE,
                    source_filter=source_unit.source_filter,
                )
            )
        if stored_unit.index_state is not UnitIndexState.READY:
            self._index_pending_unit(stored_unit, source_unit)
        restored = self._catalog.set_unit_retrieval_status(
            event.ref,
            event.retrieval_data_version,
            RetrievalStatus.ACTIVE,
        )
        if restored is None:
            raise SyncTerminalError("unit_missing_after_restore")
        return restored

    def _purge(self, event: RetrievalUpdateEvent) -> StoredUnit | None:
        stored_unit = self._catalog.get_unit(event.ref, event.retrieval_data_version)
        if stored_unit is None:
            return None  # 已清除视为幂等成功
        self._catalog.set_unit_retrieval_status(
            event.ref,
            event.retrieval_data_version,
            RetrievalStatus.TRASHED,
        )
        self._index_writer.purge(stored_unit)
        self._catalog.delete_unit(stored_unit.unit_id)
        return stored_unit

    def _read_source_unit(self, event: RetrievalUpdateEvent) -> IndexableSourceUnit:
        reader = self._source_readers.get(event.ref.source_type)
        if reader is None:
            raise SyncTerminalError("source_reader_not_installed")
        source_unit = reader.read_for_index(event)
        if source_unit is None:
            raise SyncTerminalError("source_unit_missing")
        if source_unit.ref != event.ref:
            raise SyncTerminalError("source_ref_mismatch")
        return source_unit

    def _require_stored_unit(self, event: RetrievalUpdateEvent) -> StoredUnit:
        stored_unit = self._catalog.get_unit(event.ref, event.retrieval_data_version)
        if stored_unit is None:
            raise SyncTerminalError("unit_missing_after_index")
        return stored_unit

    def _index_pending_unit(
        self,
        stored_unit: StoredUnit,
        source_unit: IndexableSourceUnit,
    ) -> None:
        """在异常离开 ``_upsert`` 前把已创建的 Unit 置为 failed。"""

        try:
            self._index_writer.index(stored_unit, source_unit)
        except Exception:
            self._catalog.mark_unit_index_failed(stored_unit.unit_id)
            raise
        self._catalog.mark_unit_index_ready(stored_unit.unit_id)


class RetrievalOutboxConsumer:
    """连接权威 Outbox 与同步服务、感知租约的适配器。"""

    def __init__(
        self,
        *,
        outbox: SqliteRetrievalOutbox,
        sync_service: RetrievalSyncService,
        retry_after_seconds: int = 5,
        allowed_source_types: Sequence[SourceType] | None = None,
        data_version_id: str | None = None,
    ) -> None:
        if retry_after_seconds <= 0:
            raise ValueError("retry_after_seconds must be greater than zero")
        supported = frozenset(sync_service.source_types)
        requested = (
            supported
            if allowed_source_types is None
            else frozenset(SourceType(source_type) for source_type in allowed_source_types)
        )
        if not requested:
            raise ValueError("allowed_source_types must not be empty")
        if not requested.issubset(supported):
            raise ValueError("allowed_source_types exceed the sync service authority")
        if data_version_id is not None and (
            not isinstance(data_version_id, str) or not data_version_id.strip()
        ):
            raise ValueError("data_version_id must be non-empty or None")
        self._outbox = outbox
        self._sync_service = sync_service
        self._retry_after_seconds = retry_after_seconds
        self._allowed_source_types = tuple(sorted(requested, key=lambda item: item.value))
        self._data_version_id = data_version_id

    def consume_due(
        self,
        conn: sqlite3.Connection,
        *,
        worker_id: str,
        now: str,
        lease_seconds: int = 30,
        limit: int = 20,
    ) -> tuple[SyncResult, ...]:
        events = self._outbox.claim_due(
            conn,
            worker_id=worker_id,
            now=now,
            lease_seconds=lease_seconds,
            limit=limit,
            allowed_source_types=self._allowed_source_types,
            data_version_id=self._data_version_id,
        )
        results: list[SyncResult] = []
        batch_size = len(events)
        batch_id = (
            self._outbox.attempt_batch_id(
                worker_id=worker_id,
                occurred_at=now,
                event_ids=tuple(event.event_id for event in events),
            )
            if events
            else None
        )
        for batch_ordinal, event in enumerate(events, start=1):
            result = self._sync_service.apply(event)
            if result.status is SyncApplyStatus.APPLIED:
                self._outbox.mark_applied(conn, event_id=event.event_id, worker_id=worker_id, now=now)
            elif result.status is SyncApplyStatus.RETRYABLE_FAILED:
                self._outbox.mark_retryable_failure(
                    conn,
                    event_id=event.event_id,
                    worker_id=worker_id,
                    now=now,
                    retry_after_seconds=self._retry_after_seconds,
                    reason_code=result.reason_code or "retryable_sync_failure",
                )
            else:
                self._outbox.mark_terminal_failure(
                    conn,
                    event_id=event.event_id,
                    worker_id=worker_id,
                    now=now,
                    reason_code=result.reason_code or "terminal_sync_failure",
                )
            self._outbox.record_attempt_audit(
                conn,
                event_id=event.event_id,
                worker_id=worker_id,
                batch_id=batch_id or "unreachable_empty_batch",
                batch_limit=limit,
                batch_size=batch_size,
                batch_ordinal=batch_ordinal,
                outcome=result.status.value,
                failure_stage=result.failure_stage,
                safe_error_code=result.safe_error_code,
                occurred_at=now,
            )
            conn.commit()
            results.append(result)
        return tuple(results)
