"""协调并修复派生检索数据，但不成为 Source 权威源。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import StrEnum
import hashlib
from collections.abc import Mapping, Sequence

from ..contracts import RetrievalMethod, SourceType, SourceUnitRef
from ..indexing.methods import SqliteRetrievalMethodStore
from .outbox import RetrievalUpdateEvent, RetrievalUpdateKind
from ..sqlite_store import SqliteRetrievalCatalog, StoredUnit, UnitIndexState
from .sync import RetrievalSyncService, SyncSourceReader


class ReconciliationIssueCode(StrEnum):
    NO_ACTIVE_DATA_VERSION = "no_active_data_version"
    SOURCE_READER_NOT_INSTALLED = "source_reader_not_installed"
    SOURCE_RECONCILIATION_NOT_SUPPORTED = "source_reconciliation_not_supported"
    SOURCE_READ_FAILED = "source_read_failed"
    SOURCE_UNIT_MISSING = "source_unit_missing"
    SOURCE_REF_MISMATCH = "source_ref_mismatch"
    SOURCE_SCOPE_MISMATCH = "source_scope_mismatch"
    UNIT_NOT_READY = "unit_not_ready"
    METHOD_MANIFEST_MISSING = "method_manifest_missing"
    EXPECTED_REPRESENTATION_MISSING = "expected_representation_missing"
    ABSENT_REPRESENTATION_PRESENT = "absent_representation_present"
    ORPHAN_DERIVED_REPRESENTATION = "orphan_derived_representation"


@dataclass(frozen=True, slots=True)
class ReconciliationIssue:
    code: ReconciliationIssueCode
    data_version_id: str | None
    unit_id: int | None = None
    source_type: SourceType | None = None
    method: RetrievalMethod | None = None
    ref: SourceUnitRef | None = None


@dataclass(frozen=True, slots=True)
class ReconciliationReport:
    data_version_id: str | None
    checked_units: int
    issues: tuple[ReconciliationIssue, ...]

    @property
    def healthy(self) -> bool:
        return not self.issues


@dataclass(frozen=True, slots=True)
class RepairActionResult:
    issue: ReconciliationIssue
    action: str
    status: str
    reason_code: str | None = None


class RetrievalReconciler:
    """仅诊断并显式修复可重建的检索制品。"""

    def __init__(
        self,
        *,
        catalog: SqliteRetrievalCatalog,
        method_store: SqliteRetrievalMethodStore,
        source_readers: Mapping[SourceType, SyncSourceReader],
        sync_service: RetrievalSyncService,
    ) -> None:
        self._catalog = catalog
        self._method_store = method_store
        self._source_readers = dict(source_readers)
        self._sync_service = sync_service

    def scan(self, data_version_id: str | None = None) -> ReconciliationReport:
        if data_version_id is None:
            active = self._catalog.active_data_version()
            if active is None:
                return ReconciliationReport(
                    data_version_id=None,
                    checked_units=0,
                    issues=(
                        ReconciliationIssue(
                            code=ReconciliationIssueCode.NO_ACTIVE_DATA_VERSION,
                            data_version_id=None,
                        ),
                    ),
                )
            data_version_id = active.id

        issues: list[ReconciliationIssue] = []
        units = self._catalog.list_stored_units(data_version_id)
        for stored_unit in units:
            issues.extend(self._scan_unit(stored_unit))
        for method, unit_ids in self._method_store.orphaned_representation_unit_ids().items():
            for unit_id in unit_ids:
                issues.append(
                    ReconciliationIssue(
                        code=ReconciliationIssueCode.ORPHAN_DERIVED_REPRESENTATION,
                        data_version_id=data_version_id,
                        unit_id=unit_id,
                        method=method,
                    )
                )
        return ReconciliationReport(
            data_version_id=data_version_id,
            checked_units=len(units),
            issues=tuple(issues),
        )

    def repair(self, issues: Sequence[ReconciliationIssue]) -> tuple[RepairActionResult, ...]:
        """只应用调用方选择的确定性派生数据修复。

        此处绝不会自动纠正 Source 漂移：这需要新的权威生命周期事件。缺失或不匹配的表示
        可以根据同一已验证指针重建；孤立表示因已无目录权威依据，可以清除。
        """

        results: list[RepairActionResult] = []
        repaired_unit_ids: set[int] = set()
        purged_orphan_ids: set[int] = set()
        for issue in issues:
            if issue.code is ReconciliationIssueCode.ORPHAN_DERIVED_REPRESENTATION:
                if issue.unit_id is not None and issue.unit_id not in purged_orphan_ids:
                    self._method_store.purge_orphaned_representation(issue.unit_id)
                    purged_orphan_ids.add(issue.unit_id)
                results.append(RepairActionResult(issue, "purge_orphaned_derived", "applied"))
                continue

            if issue.unit_id is None or issue.unit_id in repaired_unit_ids:
                results.append(RepairActionResult(issue, "no_safe_auto_repair", "skipped"))
                continue
            if issue.code not in {
                ReconciliationIssueCode.METHOD_MANIFEST_MISSING,
                ReconciliationIssueCode.EXPECTED_REPRESENTATION_MISSING,
                ReconciliationIssueCode.ABSENT_REPRESENTATION_PRESENT,
                ReconciliationIssueCode.UNIT_NOT_READY,
            }:
                results.append(RepairActionResult(issue, "requires_authority_event", "skipped"))
                continue
            stored = self._stored_unit(issue.unit_id)
            if stored is None:
                results.append(RepairActionResult(issue, "unit_disappeared", "skipped"))
                continue
            result = self._sync_service.repair(self._repair_event(stored))
            repaired_unit_ids.add(issue.unit_id)
            results.append(
                RepairActionResult(
                    issue,
                    "rebuild_derived_indexes",
                    result.status.value,
                    result.reason_code,
                )
            )
        return tuple(results)

    def _scan_unit(self, stored_unit: StoredUnit) -> list[ReconciliationIssue]:
        unit = stored_unit.unit
        ref = unit.ref
        issues: list[ReconciliationIssue] = []
        if stored_unit.index_state is not UnitIndexState.READY:
            issues.append(
                ReconciliationIssue(
                    code=ReconciliationIssueCode.UNIT_NOT_READY,
                    data_version_id=unit.retrieval_data_version,
                    unit_id=stored_unit.unit_id,
                    source_type=ref.source_type,
                    ref=ref,
                )
            )
        reader = self._source_readers.get(ref.source_type)
        if reader is None:
            issues.append(
                ReconciliationIssue(
                    code=ReconciliationIssueCode.SOURCE_READER_NOT_INSTALLED,
                    data_version_id=unit.retrieval_data_version,
                    unit_id=stored_unit.unit_id,
                    source_type=ref.source_type,
                    ref=ref,
                )
            )
        else:
            read_current = getattr(reader, "read_current_for_reconcile", None)
            if not callable(read_current):
                issues.append(
                    ReconciliationIssue(
                        code=ReconciliationIssueCode.SOURCE_RECONCILIATION_NOT_SUPPORTED,
                        data_version_id=unit.retrieval_data_version,
                        unit_id=stored_unit.unit_id,
                        source_type=ref.source_type,
                        ref=ref,
                    )
                )
                source_unit = None
                source_read_failed = True
            else:
                source_read_failed = False
            try:
                if not source_read_failed:
                    source_unit = read_current(ref)
            except Exception:
                source_read_failed = True
                issues.append(
                    ReconciliationIssue(
                        code=ReconciliationIssueCode.SOURCE_READ_FAILED,
                        data_version_id=unit.retrieval_data_version,
                        unit_id=stored_unit.unit_id,
                        source_type=ref.source_type,
                        ref=ref,
                    )
                )
                source_unit = None
            if source_unit is None:
                if not source_read_failed:
                    issues.append(
                        ReconciliationIssue(
                            code=ReconciliationIssueCode.SOURCE_UNIT_MISSING,
                            data_version_id=unit.retrieval_data_version,
                            unit_id=stored_unit.unit_id,
                            source_type=ref.source_type,
                            ref=ref,
                        )
                    )
            elif source_unit.ref != ref:
                issues.append(
                    ReconciliationIssue(
                        code=ReconciliationIssueCode.SOURCE_REF_MISMATCH,
                        data_version_id=unit.retrieval_data_version,
                        unit_id=stored_unit.unit_id,
                        source_type=ref.source_type,
                        ref=ref,
                    )
                )
            elif source_unit.source_filter != unit.source_filter:
                issues.append(
                    ReconciliationIssue(
                        code=ReconciliationIssueCode.SOURCE_SCOPE_MISMATCH,
                        data_version_id=unit.retrieval_data_version,
                        unit_id=stored_unit.unit_id,
                        source_type=ref.source_type,
                        ref=ref,
                    )
                )

        health = self._method_store.method_index_health(stored_unit.unit_id)
        if all(entry.expected_state is None for entry in health):
            issues.append(
                ReconciliationIssue(
                    code=ReconciliationIssueCode.METHOD_MANIFEST_MISSING,
                    data_version_id=unit.retrieval_data_version,
                    unit_id=stored_unit.unit_id,
                    source_type=ref.source_type,
                    ref=ref,
                )
            )
        for entry in health:
            if entry.expected_state == "ready" and not entry.representation_present:
                issues.append(
                    ReconciliationIssue(
                        code=ReconciliationIssueCode.EXPECTED_REPRESENTATION_MISSING,
                        data_version_id=unit.retrieval_data_version,
                        unit_id=stored_unit.unit_id,
                        source_type=ref.source_type,
                        method=entry.method,
                        ref=ref,
                    )
                )
            elif entry.expected_state == "absent" and entry.representation_present:
                issues.append(
                    ReconciliationIssue(
                        code=ReconciliationIssueCode.ABSENT_REPRESENTATION_PRESENT,
                        data_version_id=unit.retrieval_data_version,
                        unit_id=stored_unit.unit_id,
                        source_type=ref.source_type,
                        method=entry.method,
                        ref=ref,
                    )
                )
        return issues

    def _stored_unit(self, unit_id: int) -> StoredUnit | None:
        for stored_unit in self._catalog.list_stored_units():
            if stored_unit.unit_id == unit_id:
                return stored_unit
        return None

    @staticmethod
    def _repair_event(stored_unit: StoredUnit) -> RetrievalUpdateEvent:
        material = ":".join(
            (
                str(stored_unit.unit_id),
                stored_unit.unit.retrieval_data_version,
                stored_unit.unit.ref.source_unit_id,
                stored_unit.unit.ref.source_revision,
                stored_unit.unit.ref.indexed_content_hash,
                stored_unit.updated_at,
            )
        )
        repair_id = hashlib.sha256(material.encode("utf-8")).hexdigest()[:24]
        return RetrievalUpdateEvent(
            event_id=f"reconcile-repair:{repair_id}",
            kind=RetrievalUpdateKind.UPSERT,
            ref=stored_unit.unit.ref,
            retrieval_data_version=stored_unit.unit.retrieval_data_version,
            occurred_at=datetime.now(timezone.utc).isoformat(),
        )
