"""Current Session 派生索引的发布、增量更新、清理与重建。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock

from ....workspace.storage.context import (
    current as current_project_documents,
)
from ...contracts import (
    CURRENT_SESSION_TURN_CUTOFF_SCOPE_KEY,
    CURRENT_SESSION_TURN_INDEX_SCOPE_KEY,
    RetrievalStatus,
    SourceFilter,
    SourceType,
)
from ...indexing.encoder import DeterministicLexicalEncoder
from ...lifecycle.backfill import (
    retrieval_backfill_event_id,
    retrieval_backfill_event_id_for_ref,
)
from ...lifecycle.generation import require_exact_active_generation
from ...lifecycle.outbox import RetrievalUpdateEvent, RetrievalUpdateKind
from ...lifecycle.sync import IndexableSourceUnit, SyncApplyStatus
from ...indexing.methods import SqliteRetrievalMethodStore
from ...sqlite_store import (
    RetrievalDataVersionRole,
    RetrievalDataVersionState,
    SqliteRetrievalCatalog,
    UnitIndexState,
)
from .composition import (
    SESSION_RETRIEVAL_DB_NAME,
    build_session_retrieval_composition,
)
from .contracts import (
    SessionRetrievalComposition,
    SessionRetrievalNotReady,
    SessionStoreReadPort,
)
from .projection import _assistant_turn_index, _pair_identity


_SESSION_INDEX_LOCK = RLock()


def ensure_session_retrieval_ready(
    composition: SessionRetrievalComposition,
    *,
    session_id: str,
    assistant_turn_cutoff: int,
) -> str:
    """增量回填冻结范围，并保证其精确 generation 已原子发布。"""

    if (
        isinstance(assistant_turn_cutoff, bool)
        or not isinstance(assistant_turn_cutoff, int)
        or assistant_turn_cutoff < 0
    ):
        raise ValueError("assistant_turn_cutoff must be non-negative")
    source_filter = SourceFilter.from_mapping(
        SourceType.CURRENT_SESSION,
        {
            "session_id": session_id,
            CURRENT_SESSION_TURN_CUTOFF_SCOPE_KEY: str(assistant_turn_cutoff),
        },
    )
    with _SESSION_INDEX_LOCK:
        _backfill_session_scope(composition, source_filter)
        return _publish_session_generation(composition)


def ensure_committed_session_pair_retrieval_ready(
    composition: SessionRetrievalComposition,
    *,
    pair: Mapping[str, object],
) -> str:
    """Index one exact committed pair and prove it is searchable before returning."""

    session_id, _run_id, _created_at = _pair_identity(pair)
    assistant_turn_idx = _assistant_turn_index(pair)
    composition.source_adapter.require_index_source_ready(session_id)
    units = composition.source_adapter.project_committed_pair(pair)
    if not units:
        raise SessionRetrievalNotReady("session_pair_projection_empty")
    exact_filter = SourceFilter.from_mapping(
        SourceType.CURRENT_SESSION,
        {
            "session_id": session_id,
            CURRENT_SESSION_TURN_INDEX_SCOPE_KEY: str(assistant_turn_idx),
        },
    )
    if any(not exact_filter.selects(unit.source_filter) for unit in units):
        raise SessionRetrievalNotReady("session_pair_projection_scope_mismatch")

    spec = composition.generation_spec
    catalog = composition.foundation.catalog
    with _SESSION_INDEX_LOCK:
        active = catalog.active_data_version()
        if active is None or (
            active.id != spec.version_id or active.fingerprint != spec.fingerprint
        ):
            _backfill_session_scope(
                composition,
                SourceFilter.from_mapping(
                    SourceType.CURRENT_SESSION,
                    {
                        "session_id": session_id,
                        CURRENT_SESSION_TURN_CUTOFF_SCOPE_KEY: str(assistant_turn_idx),
                    },
                ),
            )
        else:
            _writable_session_generation(composition)
        # 首次回填和增量共享同一证明：不能只凭 generation ready 就认定本次 pair 已写入。
        _ensure_session_units_ready(composition, units)
        return _publish_session_generation(composition)


def _backfill_session_scope(
    composition: SessionRetrievalComposition,
    source_filter: SourceFilter,
) -> None:
    """回填冻结前缀，再检查每个单元；健康单元由既有回执跳过编码。"""

    units = composition.source_adapter.list_indexable_units_for_backfill(source_filter)
    target = _writable_session_generation(composition)
    report = composition.foundation.backfill_service.backfill(
        data_version_id=composition.generation_spec.version_id,
        source_filters=(source_filter,),
    )
    if report.failed:
        if target.role is RetrievalDataVersionRole.STAGING:
            composition.foundation.catalog.mark_data_version_failed(target.id)
        reasons = sorted({
            reason for source_report in report.source_reports
            for reason in source_report.reason_codes
        })
        raise SessionRetrievalNotReady(
            "session_backfill_failed:" + ",".join(reasons or ("unknown",))
        )
    _ensure_session_units_ready(composition, units)


def _ensure_session_units_ready(
    composition: SessionRetrievalComposition,
    units: Sequence[IndexableSourceUnit],
) -> None:
    """只补缺失/损坏的精确单元，并验证当前配方所需的物理方法表示。"""

    foundation = composition.foundation
    version_id = composition.generation_spec.version_id
    for unit in units:
        if _session_unit_is_ready(composition, unit):
            continue
        stored = foundation.catalog.get_unit(unit.ref, version_id)
        if stored is not None:
            # APPLIED 仅证明曾成功；撤销损坏单元的 ready，避免旧回执阻止重新编码。
            foundation.catalog.mark_unit_index_pending(stored.unit_id)
        result = foundation.sync_service.apply(RetrievalUpdateEvent(
            event_id=retrieval_backfill_event_id(version_id, unit),
            kind=RetrievalUpdateKind.UPSERT,
            ref=unit.ref,
            retrieval_data_version=version_id,
            occurred_at=datetime.now(timezone.utc).isoformat(),
        ))
        if result.status is not SyncApplyStatus.APPLIED:
            raise SessionRetrievalNotReady(
                "session_pair_index_failed:" + (result.reason_code or result.status.value)
            )
    if any(not _session_unit_is_ready(composition, unit) for unit in units):
        raise SessionRetrievalNotReady("session_pair_index_coverage_incomplete")


def _session_unit_is_ready(
    composition: SessionRetrievalComposition,
    unit: IndexableSourceUnit,
) -> bool:
    foundation = composition.foundation
    stored = foundation.catalog.get_unit(unit.ref, composition.generation_spec.version_id)
    if (
        stored is None
        or stored.unit.source_filter != unit.source_filter
        or stored.unit.retrieval_status is not RetrievalStatus.ACTIVE
        or stored.index_state is not UnitIndexState.READY
    ):
        return False
    health = {
        item.method: item
        for item in foundation.method_store.method_index_health(stored.unit_id)
    }
    return all(
        method in health
        and health[method].expected_state == "ready"
        and health[method].representation_present
        for method in foundation.retrieval_methods
    )


def purge_session_retrieval_index(
    session_id: str,
    *,
    retrieval_db_path: Path | str | None = None,
) -> int:
    """Physically remove one Session's derived Units and method representations."""

    if not isinstance(session_id, str) or not session_id.strip():
        raise ValueError("session_id must not be blank")
    if retrieval_db_path is None:
        database = current_project_documents()
        if database is None:
            return 0
        db_path = database.db_path.with_name(SESSION_RETRIEVAL_DB_NAME)
    else:
        db_path = Path(retrieval_db_path)
    if not db_path.exists():
        return 0

    with _SESSION_INDEX_LOCK:
        catalog = SqliteRetrievalCatalog(db_path)
        catalog.initialize()
        method_store = SqliteRetrievalMethodStore(
            catalog=catalog,
            encoder=DeterministicLexicalEncoder(),
        )
        source_filter = SourceFilter.from_mapping(
            SourceType.CURRENT_SESSION,
            {"session_id": session_id},
        )
        units = catalog.list_stored_units_for_source_filter(source_filter)
        receipt_ids: list[str] = []
        for stored in units:
            method_store.purge(stored)
            catalog.delete_unit(stored.unit_id)
            receipt_ids.append(
                retrieval_backfill_event_id_for_ref(
                    stored.unit.retrieval_data_version,
                    stored.unit.ref,
                )
            )
        catalog.delete_sync_receipts(receipt_ids)
        return len(units)


def rebuild_session_retrieval_index(
    session_id: str,
    *,
    store: SessionStoreReadPort,
) -> str | None:
    """Rebuild the complete committed prefix after a lifecycle purge."""

    if current_project_documents() is None:
        return None
    composition = build_session_retrieval_composition(store=store)
    pairs = tuple(store.list_committed_turn_pairs(session_id, limit=1))
    if not pairs:
        return None
    return ensure_session_retrieval_ready(
        composition,
        session_id=session_id,
        assistant_turn_cutoff=_assistant_turn_index(pairs[-1]),
    )


def _writable_session_generation(
    composition: SessionRetrievalComposition,
):
    spec = composition.generation_spec
    catalog = composition.foundation.catalog
    active = catalog.active_data_version()
    target = catalog.get_data_version(spec.version_id)
    if active is not None and (
        active.id == spec.version_id and active.fingerprint == spec.fingerprint
    ):
        target = active
    elif target is None:
        target, _ = catalog.ensure_staging_data_version(
            version_id=spec.version_id,
            fingerprint=spec.fingerprint,
        )
    elif target.fingerprint != spec.fingerprint:
        raise SessionRetrievalNotReady("session_generation_identity_collision")

    if target.state is RetrievalDataVersionState.FAILED:
        try:
            target = catalog.reopen_failed_data_version_for_rebuild(spec.version_id)
        except Exception as exc:
            raise SessionRetrievalNotReady(
                "session_generation_failed_recovery_unavailable"
            ) from exc
    if target.role is RetrievalDataVersionRole.PREVIOUS:
        raise SessionRetrievalNotReady(
            "session_generation_rebuild_requires_derived_index_reset"
        )
    if (
        target.role is RetrievalDataVersionRole.STAGING
        and target.state is RetrievalDataVersionState.READY
    ):
        # ready 到激活之间可能崩溃；重新开放写入，先补齐本次冻结范围再发布。
        target = catalog.invalidate_ready_staging_data_version(spec.version_id)
    return target


def _publish_session_generation(
    composition: SessionRetrievalComposition,
) -> str:
    spec = composition.generation_spec
    catalog = composition.foundation.catalog
    target = catalog.get_data_version(spec.version_id)
    if target is None:
        raise SessionRetrievalNotReady("session_generation_disappeared")
    if (
        target.role is RetrievalDataVersionRole.STAGING
        and target.state is RetrievalDataVersionState.BUILDING
    ):
        target = catalog.mark_data_version_ready(spec.version_id)
    if target.role is RetrievalDataVersionRole.STAGING:
        catalog.activate_data_version(spec.version_id)
    return require_exact_active_generation(catalog, spec).id


__all__ = [
    "ensure_committed_session_pair_retrieval_ready",
    "ensure_session_retrieval_ready",
    "purge_session_retrieval_index",
    "rebuild_session_retrieval_index",
]
