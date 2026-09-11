"""Document 检索 generation 的显式重建与发布应用服务。

Rollout 只写一个新的 STAGING generation。旧 ACTIVE 在完整语料回填、精确指针覆盖和
逐方法物理表示全部证明之前保持不变；最终指针交换由 Document 权威侧 publisher 在其
写围栏内完成。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from .backfill import BackfillSourceReader, RetrievalBackfillService
from ..contracts import RetrievalMethod, RetrievalStatus, SourceFilter, SourceType
from .generation import (
    RetrievalGenerationRestorePlan,
    RetrievalGenerationRestoreStatus,
    RetrievalGenerationSpec,
    document_index_methods,
)
from .corpus import FILE_CORPUS
from ..indexing.methods import SqliteRetrievalMethodStore
from ..sqlite_store import (
    RetrievalDataVersion,
    RetrievalDataVersionRole,
    RetrievalDataVersionState,
    SqliteRetrievalCatalog,
    StoredUnit,
    UnitIndexState,
)
from .sync import IndexableSourceUnit


class GenerationRolloutStatus(StrEnum):
    ACTIVATED = "activated"
    ALREADY_ACTIVE = "already_active"


class GenerationRolloutStage(StrEnum):
    STAGING_ENSURED = "staging_ensured"
    BACKFILL_COMPLETE = "backfill_complete"
    READINESS_VERIFIED = "readiness_verified"
    BEFORE_PUBLISH = "before_publish"


class GenerationRolloutError(RuntimeError):
    """安全且不含 Source 内容的 rollout 失败。"""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class DocumentGenerationPublisher(Protocol):
    """在 Document 权威写围栏内证明并发布一个 READY generation。"""

    def publish_rebuilt_generation(
        self,
        *,
        expected_active_version_id: str | None,
    ) -> RetrievalDataVersion: ...

    def restore_previous_generation(
        self,
        *,
        previous_generation_id: str,
        expected_fingerprint: str,
        expected_active_version_id: str | None,
    ) -> RetrievalDataVersion: ...


@dataclass(frozen=True, slots=True)
class GenerationMethodCoverage:
    method: RetrievalMethod
    required: bool
    unit_count: int
    manifest_ready_count: int
    representation_present_count: int
    manifest_absent_count: int
    invalid_unit_count: int

    @property
    def ready(self) -> bool:
        return self.unit_count > 0 and self.invalid_unit_count == 0


@dataclass(frozen=True, slots=True)
class GenerationRolloutReport:
    status: GenerationRolloutStatus
    generation_id: str
    generation_fingerprint: str
    previous_generation_id: str | None
    source_unit_count: int
    method_coverage: tuple[GenerationMethodCoverage, ...]


@dataclass(frozen=True, slots=True)
class _SourceManifest:
    units: tuple[IndexableSourceUnit, ...]
    keys: frozenset[tuple[object, ...]]


class DocumentGenerationRolloutService:
    """把当前项目的完整 File corpus 发布为一个新 generation。

    类名作为现有维护 API 保留；其规范 manifest 已是 Document + Picture union。
    """

    def __init__(
        self,
        *,
        catalog: SqliteRetrievalCatalog,
        method_store: SqliteRetrievalMethodStore,
        backfill_service: RetrievalBackfillService,
        document_source_reader: BackfillSourceReader,
        picture_source_reader: BackfillSourceReader,
        publisher: DocumentGenerationPublisher,
        fault_injector: Callable[[GenerationRolloutStage], None] | None = None,
    ) -> None:
        if document_source_reader.source_type is not SourceType.DOCUMENT:
            raise ValueError("generation rollout requires a Document source reader")
        if picture_source_reader.source_type is not SourceType.PICTURE:
            raise ValueError("generation rollout requires a Picture source reader")
        self._catalog = catalog
        self._method_store = method_store
        self._backfill_service = backfill_service
        self._source_readers: Mapping[SourceType, BackfillSourceReader] = {
            SourceType.DOCUMENT: document_source_reader,
            SourceType.PICTURE: picture_source_reader,
        }
        self._publisher = publisher
        self._fault_injector = fault_injector

    def rollout(
        self,
        spec: RetrievalGenerationSpec,
    ) -> GenerationRolloutReport:
        """全量重建、验证并原子发布 ``spec``，绝不改写旧 ACTIVE。"""

        required_methods = self._require_supported_spec(spec)
        self._require_runtime_capabilities(spec, required_methods)
        initial_active = self._catalog.active_data_version()
        expected = self._read_full_source_manifest()

        if initial_active is not None and initial_active.id == spec.version_id:
            self._require_exact_generation_identity(initial_active, spec)
            coverage = self._require_generation_ready(
                spec.version_id,
                expected,
                required_methods,
            )
            return GenerationRolloutReport(
                status=GenerationRolloutStatus.ALREADY_ACTIVE,
                generation_id=spec.version_id,
                generation_fingerprint=spec.fingerprint,
                previous_generation_id=None,
                source_unit_count=len(expected.units),
                method_coverage=coverage,
            )

        existing_target = self._catalog.get_data_version(spec.version_id)
        rebuild_existing = False
        if (
            existing_target is not None
            and existing_target.role is RetrievalDataVersionRole.PREVIOUS
        ):
            self._require_exact_generation_identity(existing_target, spec)
            try:
                coverage = self._require_generation_ready(
                    existing_target.id,
                    expected,
                    required_methods,
                )
            except GenerationRolloutError:
                try:
                    target = self._catalog.reopen_previous_data_version_for_rebuild(
                        existing_target.id,
                        expected_active_version_id=(
                            initial_active.id if initial_active is not None else None
                        ),
                    )
                except Exception as exc:
                    raise GenerationRolloutError(
                        "generation_previous_rebuild_reopen_failed"
                    ) from exc
                rebuild_existing = True
            else:
                return self._restore_ready_previous(
                    spec=spec,
                    initial_active=initial_active,
                    expected=expected,
                    coverage=coverage,
                )
        else:
            try:
                target, _ = self._catalog.ensure_staging_data_version(
                    version_id=spec.version_id,
                    fingerprint=spec.fingerprint,
                )
            except Exception as exc:
                raise GenerationRolloutError("generation_staging_conflict") from exc

        self._require_exact_generation_identity(target, spec)
        if target.role is not RetrievalDataVersionRole.STAGING:
            raise GenerationRolloutError("generation_target_not_staging")
        if target.state is RetrievalDataVersionState.FAILED:
            raise GenerationRolloutError("generation_target_failed")
        self._fault(GenerationRolloutStage.STAGING_ENSURED)

        if target.state is RetrievalDataVersionState.READY:
            try:
                coverage = self._require_generation_ready(
                    target.id,
                    expected,
                    required_methods,
                )
            except GenerationRolloutError:
                target = self._catalog.invalidate_ready_staging_data_version(target.id)
            else:
                return self._publish(
                    spec=spec,
                    initial_active=initial_active,
                    expected=expected,
                    coverage=coverage,
                )
        if target.state is not RetrievalDataVersionState.BUILDING:
            raise GenerationRolloutError("generation_target_not_building")
        rebuild_existing = rebuild_existing or target.activated_at is not None

        self._purge_unexpected_staging_units(target.id, expected)
        report = self._backfill_service.backfill(
            data_version_id=target.id,
            source_filters=tuple(
                SourceFilter.from_mapping(source_type)
                for source_type in FILE_CORPUS.source_types
            ),
            repair_existing=rebuild_existing,
            repair_namespace=(
                f"{target.id}:{target.activated_at}"
                if rebuild_existing and target.activated_at is not None
                else None
            ),
        )
        if report.failed:
            raise GenerationRolloutError("generation_backfill_failed")
        if any(source_report.skipped for source_report in report.source_reports):
            raise GenerationRolloutError("generation_backfill_scope_mismatch")
        if any(
            "source_backfill_not_supported" in source_report.reason_codes
            for source_report in report.source_reports
        ):
            raise GenerationRolloutError("generation_file_source_unavailable")
        self._fault(GenerationRolloutStage.BACKFILL_COMPLETE)

        # Source 在长时间 BGE 编码期间可能变化。只有前后不含内容的完整绑定清单一致，
        # 本次回填才可以继续进入 READY。
        current = self._read_full_source_manifest()
        if current.keys != expected.keys:
            raise GenerationRolloutError("generation_source_changed_during_backfill")
        coverage = self._require_generation_ready(
            target.id,
            current,
            required_methods,
        )
        self._fault(GenerationRolloutStage.READINESS_VERIFIED)
        return self._publish(
            spec=spec,
            initial_active=initial_active,
            expected=current,
            coverage=coverage,
        )

    def rollback(
        self,
        spec: RetrievalGenerationSpec,
        *,
        previous_generation_id: str,
        expected_fingerprint: str,
    ) -> RetrievalGenerationRestorePlan:
        """在 Document 权威围栏内恢复与当前 runtime 精确匹配的 PREVIOUS。"""

        required_methods = self._require_supported_spec(spec)
        self._require_runtime_capabilities(spec, required_methods)
        if (
            previous_generation_id != spec.version_id
            or expected_fingerprint != spec.fingerprint
        ):
            raise GenerationRolloutError("generation_rollback_identity_mismatch")
        target = self._catalog.get_data_version(previous_generation_id)
        if target is None:
            raise GenerationRolloutError("generation_rollback_target_missing")
        self._require_exact_generation_identity(target, spec)
        if (
            target.role is not RetrievalDataVersionRole.PREVIOUS
            or target.state is not RetrievalDataVersionState.READY
        ):
            raise GenerationRolloutError("generation_rollback_target_not_ready_previous")
        expected = self._read_full_source_manifest()
        self._require_generation_ready(
            target.id,
            expected,
            required_methods,
        )
        initial_active = self._catalog.active_data_version()
        try:
            restored = self._publisher.restore_previous_generation(
                previous_generation_id=target.id,
                expected_fingerprint=target.fingerprint,
                expected_active_version_id=(
                    initial_active.id if initial_active is not None else None
                ),
            )
        except Exception as exc:
            reason = getattr(exc, "reason_code", None)
            code = (
                f"generation_rollback_refused:{reason}"
                if isinstance(reason, str) and reason
                else "generation_rollback_failed"
            )
            raise GenerationRolloutError(code) from exc
        self._require_exact_generation_identity(restored, spec)
        if (
            restored.role is not RetrievalDataVersionRole.ACTIVE
            or restored.state is not RetrievalDataVersionState.READY
        ):
            raise GenerationRolloutError("generation_rollback_did_not_activate")
        return RetrievalGenerationRestorePlan(
            status=RetrievalGenerationRestoreStatus.RESTORED,
            active_generation_id=restored.id,
            target_generation_id=restored.id,
            target_fingerprint=restored.fingerprint,
            reason_code=None,
            ready_unit_count=len(expected.units),
            pending_unit_count=0,
            failed_unit_count=0,
        )

    def _publish(
        self,
        *,
        spec: RetrievalGenerationSpec,
        initial_active: RetrievalDataVersion | None,
        expected: _SourceManifest,
        coverage: tuple[GenerationMethodCoverage, ...],
    ) -> GenerationRolloutReport:
        self._fault(GenerationRolloutStage.BEFORE_PUBLISH)
        try:
            active = self._publisher.publish_rebuilt_generation(
                expected_active_version_id=(
                    initial_active.id if initial_active is not None else None
                ),
            )
        except Exception as exc:
            reason = getattr(exc, "reason_code", None)
            code = (
                f"generation_publish_refused:{reason}"
                if isinstance(reason, str) and reason
                else "generation_publish_failed"
            )
            raise GenerationRolloutError(code) from exc
        self._require_exact_generation_identity(active, spec)
        if (
            active.role is not RetrievalDataVersionRole.ACTIVE
            or active.state is not RetrievalDataVersionState.READY
        ):
            raise GenerationRolloutError("generation_publish_did_not_activate")
        return GenerationRolloutReport(
            status=GenerationRolloutStatus.ACTIVATED,
            generation_id=active.id,
            generation_fingerprint=active.fingerprint,
            previous_generation_id=(
                initial_active.id if initial_active is not None else None
            ),
            source_unit_count=len(expected.units),
            method_coverage=coverage,
        )

    def _restore_ready_previous(
        self,
        *,
        spec: RetrievalGenerationSpec,
        initial_active: RetrievalDataVersion | None,
        expected: _SourceManifest,
        coverage: tuple[GenerationMethodCoverage, ...],
    ) -> GenerationRolloutReport:
        try:
            active = self._publisher.restore_previous_generation(
                previous_generation_id=spec.version_id,
                expected_fingerprint=spec.fingerprint,
                expected_active_version_id=(
                    initial_active.id if initial_active is not None else None
                ),
            )
        except Exception as exc:
            reason = getattr(exc, "reason_code", None)
            code = (
                f"generation_restore_refused:{reason}"
                if isinstance(reason, str) and reason
                else "generation_restore_failed"
            )
            raise GenerationRolloutError(code) from exc
        self._require_exact_generation_identity(active, spec)
        if (
            active.role is not RetrievalDataVersionRole.ACTIVE
            or active.state is not RetrievalDataVersionState.READY
        ):
            raise GenerationRolloutError("generation_restore_did_not_activate")
        return GenerationRolloutReport(
            status=GenerationRolloutStatus.ACTIVATED,
            generation_id=active.id,
            generation_fingerprint=active.fingerprint,
            previous_generation_id=(
                initial_active.id if initial_active is not None else None
            ),
            source_unit_count=len(expected.units),
            method_coverage=coverage,
        )

    @staticmethod
    def _require_supported_spec(
        spec: RetrievalGenerationSpec,
    ) -> tuple[RetrievalMethod, ...]:
        if not isinstance(spec, RetrievalGenerationSpec):
            raise TypeError("spec must be RetrievalGenerationSpec")
        if spec.source_types != FILE_CORPUS.generation_source_types:
            raise GenerationRolloutError("generation_source_types_not_file_corpus")
        try:
            return document_index_methods(spec.index_recipe)
        except ValueError as exc:
            raise GenerationRolloutError("generation_index_recipe_unsupported") from exc

    def _require_runtime_capabilities(
        self,
        spec: RetrievalGenerationSpec,
        required_methods: Sequence[RetrievalMethod],
    ) -> None:
        snapshot = self._method_store.encoder_capability_snapshot()
        if snapshot.get("fingerprint") != spec.encoder_fingerprint:
            raise GenerationRolloutError("generation_encoder_fingerprint_mismatch")
        required = frozenset(required_methods)
        if RetrievalMethod.BM25 in required:
            fts5 = self._catalog.capability("fts5")
            if fts5 is None or not fts5[0]:
                raise GenerationRolloutError("generation_bm25_capability_unavailable")
        if RetrievalMethod.DENSE in required:
            sqlite_vec = self._catalog.capability("sqlite_vec")
            if sqlite_vec is None or not sqlite_vec[0]:
                raise GenerationRolloutError("generation_dense_capability_unavailable")
        if (
            RetrievalMethod.LEARNED_SPARSE in required
            and snapshot.get("learned_sparse_projection_assets_available") is not True
        ):
            raise GenerationRolloutError("generation_sparse_capability_unavailable")

    def _read_full_source_manifest(self) -> _SourceManifest:
        all_units: list[IndexableSourceUnit] = []
        for source_type in FILE_CORPUS.source_types:
            reader = self._source_readers[source_type]
            try:
                units = tuple(
                    reader.list_indexable_units_for_backfill(
                        SourceFilter.from_mapping(source_type)
                    )
                )
            except Exception as exc:
                raise GenerationRolloutError(
                    f"generation_{source_type.value}_source_read_failed"
                ) from exc
            for unit in units:
                if unit.ref.source_type is not source_type:
                    raise GenerationRolloutError("generation_source_type_mismatch")
            all_units.extend(units)
        units = tuple(all_units)
        if not units:
            raise GenerationRolloutError("generation_file_corpus_empty")
        keys: set[tuple[object, ...]] = set()
        unique_units: list[IndexableSourceUnit] = []
        for unit in units:
            key = _unit_manifest_key(unit)
            if key in keys:
                continue
            keys.add(key)
            unique_units.append(unit)
        unique_units.sort(key=_source_unit_sort_key)
        return _SourceManifest(tuple(unique_units), frozenset(keys))

    def _purge_unexpected_staging_units(
        self,
        generation_id: str,
        expected: _SourceManifest,
    ) -> None:
        for stored in self._catalog.list_stored_units(generation_id):
            if _stored_unit_manifest_key(stored) in expected.keys:
                continue
            self._catalog.set_unit_retrieval_status(
                stored.unit.ref,
                generation_id,
                RetrievalStatus.TRASHED,
            )
            self._method_store.purge(stored)
            self._catalog.delete_unit(stored.unit_id)

    def _require_generation_ready(
        self,
        generation_id: str,
        expected: _SourceManifest,
        required_methods: Sequence[RetrievalMethod],
    ) -> tuple[GenerationMethodCoverage, ...]:
        stored_units = self._catalog.list_stored_units(generation_id)
        stored_keys = frozenset(
            _stored_unit_manifest_key(stored) for stored in stored_units
        )
        if stored_keys != expected.keys or len(stored_units) != len(expected.units):
            raise GenerationRolloutError("generation_file_coverage_incomplete")
        if any(
            stored.index_state is not UnitIndexState.READY
            or stored.unit.retrieval_status is not RetrievalStatus.ACTIVE
            for stored in stored_units
        ):
            raise GenerationRolloutError("generation_unit_not_ready")

        coverage = generation_readiness_snapshot(
            catalog=self._catalog,
            method_store=self._method_store,
            generation_id=generation_id,
            required_methods=required_methods,
        )
        for item in coverage:
            if not item.ready:
                raise GenerationRolloutError(
                    f"generation_method_not_ready:{item.method.value}"
                )
        return coverage

    @staticmethod
    def _require_exact_generation_identity(
        version: RetrievalDataVersion,
        spec: RetrievalGenerationSpec,
    ) -> None:
        if version.id != spec.version_id or version.fingerprint != spec.fingerprint:
            raise GenerationRolloutError("generation_target_identity_mismatch")

    def _fault(self, stage: GenerationRolloutStage) -> None:
        if self._fault_injector is not None:
            self._fault_injector(stage)


def _unit_manifest_key(unit: IndexableSourceUnit) -> tuple[object, ...]:
    ref = unit.ref
    return (
        ref.source_type.value,
        ref.source_unit_id,
        ref.source_revision,
        ref.indexed_content_hash,
        unit.source_filter.scope,
    )


def _stored_unit_manifest_key(stored: StoredUnit) -> tuple[object, ...]:
    unit = stored.unit
    ref = unit.ref
    return (
        ref.source_type.value,
        ref.source_unit_id,
        ref.source_revision,
        ref.indexed_content_hash,
        unit.source_filter.scope if unit.source_filter is not None else None,
    )


def _source_unit_sort_key(unit: IndexableSourceUnit) -> tuple[object, ...]:
    return _unit_manifest_key(unit)


def generation_readiness_snapshot(
    *,
    catalog: SqliteRetrievalCatalog,
    method_store: SqliteRetrievalMethodStore,
    generation_id: str,
    required_methods: Sequence[RetrievalMethod],
) -> tuple[GenerationMethodCoverage, ...]:
    """返回固定三方法形态的纯派生 readiness 投影。

    该投影适合运维和评测 manifest：它不读取 Source 内容或路径。空 generation 仍返回
    三项，但每项 ``ready`` 均为 false，避免把“没有任何索引”报告为可发布。
    """

    required = frozenset(required_methods)
    supported = frozenset({
        RetrievalMethod.DENSE,
        RetrievalMethod.LEARNED_SPARSE,
        RetrievalMethod.BM25,
    })
    if not required or not required.issubset(supported):
        raise ValueError("required_methods must be a non-empty supported method set")
    stored_units = catalog.list_stored_units(generation_id)
    health = method_store.generation_method_index_health(generation_id)
    coverage: list[GenerationMethodCoverage] = []
    for method in (
        RetrievalMethod.DENSE,
        RetrievalMethod.LEARNED_SPARSE,
        RetrievalMethod.BM25,
    ):
        entries = []
        missing = 0
        for stored in stored_units:
            entry = next(
                (
                    item
                    for item in health.get(stored.unit_id, ())
                    if item.method is method
                ),
                None,
            )
            if entry is None:
                missing += 1
            else:
                entries.append(entry)
        is_required = method in required
        invalid = missing + sum(
            1
            for entry in entries
            if (
                entry.expected_state != ("ready" if is_required else "absent")
                or entry.representation_present is not is_required
            )
        )
        coverage.append(GenerationMethodCoverage(
            method=method,
            required=is_required,
            unit_count=len(stored_units),
            manifest_ready_count=sum(
                entry.expected_state == "ready" for entry in entries
            ),
            representation_present_count=sum(
                entry.representation_present for entry in entries
            ),
            manifest_absent_count=sum(
                entry.expected_state == "absent" for entry in entries
            ),
            invalid_unit_count=invalid,
        ))
    return tuple(coverage)


__all__ = [
    "DocumentGenerationPublisher",
    "DocumentGenerationRolloutService",
    "GenerationMethodCoverage",
    "GenerationRolloutError",
    "GenerationRolloutReport",
    "GenerationRolloutStage",
    "GenerationRolloutStatus",
    "generation_readiness_snapshot",
]
