"""为可重建检索索引提供显式且受作用域约束的回填。

回填是面向操作人员的生命周期动作，绝不是查询侧隐式回退。调用方必须提供受信的
``SourceFilter`` 值；服务本身不会枚举每个用户、任务、Session 或文档。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from typing import Protocol

from ..contracts import SourceFilter, SourceType, SourceUnitRef
from .outbox import RetrievalUpdateEvent, RetrievalUpdateKind
from ..sqlite_store import RetrievalDataVersionState, SqliteRetrievalCatalog
from .sync import IndexableSourceUnit, RetrievalSyncService, SyncApplyStatus


class BackfillSourceReader(Protocol):
    source_type: SourceType

    def list_indexable_units_for_backfill(
        self,
        source_filter: SourceFilter,
    ) -> Sequence[IndexableSourceUnit]: ...


@dataclass(frozen=True, slots=True)
class BackfillSourceReport:
    source_filter: SourceFilter
    discovered: int
    applied: int
    replayed: int
    skipped: int
    failed: int
    reason_codes: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class BackfillReport:
    data_version_id: str
    source_reports: tuple[BackfillSourceReport, ...]

    @property
    def applied(self) -> int:
        return sum(report.applied for report in self.source_reports)

    @property
    def failed(self) -> int:
        return sum(report.failed for report in self.source_reports)


class RetrievalBackfillService:
    """将调用方选择的权威作用域索引到一个数据版本中。"""

    def __init__(
        self,
        *,
        catalog: SqliteRetrievalCatalog,
        source_readers: Mapping[SourceType, BackfillSourceReader],
        sync_service: RetrievalSyncService,
    ) -> None:
        self._catalog = catalog
        self._source_readers = dict(source_readers)
        self._sync_service = sync_service

    def backfill(
        self,
        *,
        data_version_id: str,
        source_filters: Sequence[SourceFilter],
        occurred_at: str | None = None,
        repair_existing: bool = False,
        repair_namespace: str | None = None,
    ) -> BackfillReport:
        """为显式 Source 作用域同步应用确定性的 upsert。

        ``source_filters`` 已是受信的策略或维护输入。若 Source 读取器返回所给过滤器以外
        的 Unit，该 Unit 会被跳过，而不会被静默扩展为全局重建。
        """

        if not isinstance(repair_existing, bool):
            raise TypeError("repair_existing must be a bool")
        if repair_existing:
            if not isinstance(repair_namespace, str) or not repair_namespace.strip():
                raise ValueError(
                    "repair_namespace is required when existing units are rebuilt"
                )
        elif repair_namespace is not None:
            raise ValueError(
                "repair_namespace is valid only when existing units are rebuilt"
            )
        data_version = self._catalog.get_data_version(data_version_id)
        if data_version is None:
            raise ValueError(f"unknown retrieval data version: {data_version_id}")
        if data_version.state is RetrievalDataVersionState.FAILED:
            raise ValueError("cannot backfill a failed retrieval data version")
        now = occurred_at or datetime.now(timezone.utc).isoformat()
        reports: list[BackfillSourceReport] = []
        seen_filters: set[SourceFilter] = set()
        for source_filter in source_filters:
            if source_filter in seen_filters:
                continue
            seen_filters.add(source_filter)
            reports.append(self._backfill_source(
                data_version_id,
                source_filter,
                now,
                repair_existing=repair_existing,
                repair_namespace=repair_namespace,
            ))
        return BackfillReport(data_version_id=data_version_id, source_reports=tuple(reports))

    def _backfill_source(
        self,
        data_version_id: str,
        source_filter: SourceFilter,
        occurred_at: str,
        *,
        repair_existing: bool,
        repair_namespace: str | None,
    ) -> BackfillSourceReport:
        reader = self._source_readers.get(source_filter.source_type)
        if reader is None:
            return BackfillSourceReport(
                source_filter=source_filter,
                discovered=0,
                applied=0,
                replayed=0,
                skipped=0,
                failed=0,
                reason_codes=("source_backfill_not_supported",),
            )
        try:
            source_units = tuple(reader.list_indexable_units_for_backfill(source_filter))
        except Exception as exc:
            return BackfillSourceReport(
                source_filter=source_filter,
                discovered=0,
                applied=0,
                replayed=0,
                skipped=0,
                failed=1,
                reason_codes=(f"source_backfill_list_failed:{type(exc).__name__}",),
            )

        applied = replayed = skipped = failed = 0
        reasons: set[str] = set()
        valid_units: list[IndexableSourceUnit] = []
        for source_unit in source_units:
            if not source_filter.selects(source_unit.source_filter):
                skipped += 1
                reasons.add("source_backfill_scope_mismatch")
                continue
            valid_units.append(source_unit)
        for source_unit in sorted(valid_units, key=_source_unit_sort_key):
            existing = self._catalog.get_unit(source_unit.ref, data_version_id)
            rebuild_existing = repair_existing and existing is not None
            event = RetrievalUpdateEvent(
                event_id=(
                    _backfill_repair_event_id(
                        data_version_id,
                        source_unit,
                        namespace=str(repair_namespace),
                    )
                    if rebuild_existing
                    else retrieval_backfill_event_id(data_version_id, source_unit)
                ),
                kind=RetrievalUpdateKind.UPSERT,
                ref=source_unit.ref,
                retrieval_data_version=data_version_id,
                occurred_at=occurred_at,
            )
            result = (
                self._sync_service.repair(event)
                if rebuild_existing
                else self._sync_service.apply(event)
            )
            if result.status is SyncApplyStatus.APPLIED:
                if result.replayed:
                    replayed += 1
                else:
                    applied += 1
            else:
                failed += 1
                reasons.add(result.reason_code or result.status.value)
        return BackfillSourceReport(
            source_filter=source_filter,
            discovered=len(source_units),
            applied=applied,
            replayed=replayed,
            skipped=skipped,
            failed=failed,
            reason_codes=tuple(sorted(reasons)),
        )


def retrieval_backfill_event_id(
    data_version_id: str,
    source_unit: IndexableSourceUnit,
) -> str:
    """Return the stable UPSERT receipt identity shared by backfill and cleanup."""

    return retrieval_backfill_event_id_for_ref(data_version_id, source_unit.ref)


def retrieval_backfill_event_id_for_ref(
    data_version_id: str,
    ref: SourceUnitRef,
) -> str:
    """Return a stable UPSERT identity without requiring authoritative content."""

    material = "\x1f".join(
        (
            "retrieval-backfill-v1",
            data_version_id,
            ref.source_type.value,
            ref.source_unit_id,
            ref.source_revision,
            ref.indexed_content_hash,
        )
    )
    return f"backfill:{hashlib.sha256(material.encode('utf-8')).hexdigest()[:32]}"


def _backfill_repair_event_id(
    data_version_id: str,
    source_unit: IndexableSourceUnit,
    *,
    namespace: str,
) -> str:
    """为一次已发布 generation 的重建周期生成可重放 repair 身份。"""

    ref = source_unit.ref
    material = "\x1f".join(
        (
            "retrieval-backfill-repair-v1",
            data_version_id,
            namespace,
            ref.source_type.value,
            ref.source_unit_id,
            ref.source_revision,
            ref.indexed_content_hash,
        )
    )
    return (
        "backfill-repair:"
        f"{hashlib.sha256(material.encode('utf-8')).hexdigest()[:32]}"
    )


def _source_unit_sort_key(source_unit: IndexableSourceUnit) -> tuple[str, str, str, str]:
    ref = source_unit.ref
    return (ref.source_type.value, ref.source_unit_id, ref.source_revision, ref.indexed_content_hash)
