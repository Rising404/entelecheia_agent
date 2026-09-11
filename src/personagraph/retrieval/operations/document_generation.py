"""文档检索 generation 的选择、覆盖证明、发布与回滚权威。"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
import sqlite3
from typing import Iterator

from ..contracts import (
    RetrievalMethod,
    RetrievalStatus,
    SourceFilter,
    SourceType,
    SourceUnitRef,
)
from ..lifecycle.generation import (
    RetrievalGenerationSpec,
    document_index_methods,
)
from ..lifecycle.backfill import BackfillSourceReader
from ..indexing.methods import SqliteRetrievalMethodStore
from ..lifecycle.outbox import OutboxStatus, RetrievalUpdateKind
from ..sources.identity import mounted_document_chunk_ref_and_content
from ..sqlite_store import (
    RetrievalCatalogError,
    RetrievalDataVersion,
    RetrievalDataVersionRole,
    RetrievalDataVersionState,
    SqliteRetrievalCatalog,
    UnitIndexState,
)


@dataclass(frozen=True, slots=True)
class _CoverageBinding:
    ref: SourceUnitRef
    source_filter: SourceFilter

    @property
    def digest_tuple(self) -> tuple[str, str, str]:
        return (
            self.ref.source_unit_id,
            self.ref.source_revision,
            self.ref.indexed_content_hash,
        )


class DocumentGenerationTerminalFailure(RuntimeError):
    """当前来源或 generation 不允许继续索引/发布。"""

    def __init__(self, reason_code: str) -> None:
        self.reason_code = reason_code
        super().__init__(reason_code)


class DocumentGenerationRetryableFailure(RuntimeError):
    """派生索引尚未完成，可在后续维护批次继续。"""

    def __init__(self, reason_code: str) -> None:
        self.reason_code = reason_code
        super().__init__(reason_code)


class RetrievalPublicationRefused(RuntimeError):
    """bootstrap generation 可能尚未发布及其原因。"""

    def __init__(self, reason_code: str) -> None:
        self.reason_code = reason_code
        super().__init__(reason_code)


class DocumentGenerationAuthority:
    """冻结、证明并发布一个精确 File-corpus generation。"""

    def __init__(
        self,
        *,
        catalog: SqliteRetrievalCatalog,
        generation_spec: RetrievalGenerationSpec,
        method_store: SqliteRetrievalMethodStore | None,
        connect_documents: Callable[[], sqlite3.Connection],
        picture_source_reader: BackfillSourceReader,
    ) -> None:
        self._catalog = catalog
        self._generation_spec = generation_spec
        self._method_store = method_store
        self._connect_documents = connect_documents
        self._picture_source_reader = picture_source_reader

    def publish_bootstrap_generation(self) -> RetrievalDataVersion:
        """其中所有内容可搜索后激活 bootstrap generation。

        发布使已索引单元可达：查询只读取活动 generation。否则，刚完成提交与清空的
        同步调用方会在恰好从未运行持久化 job 的机器上留下一个无人查询的完整索引。

        此处不重新实现判断。两个转换都会在 memory 权威事务内运行现有 corpus 全局
        前置条件——每个事件已应用、每个挂载文档已覆盖、可搜索集合与预期集合完全
        相同。该检查保证发布安全，且不关心请求方身份。

        只接触 STAGING 目标。已活动 generation 原样返回，其他 role 均不发布：切换
        活动索引是全系统操作，只适合作为空 corpus 首次 ingest 的尾部步骤。
        """

        target = self._refresh_source_commit_target(
            self._select_source_commit_target()
        )
        try:
            if target.role is RetrievalDataVersionRole.ACTIVE:
                return target
            if target.role is not RetrievalDataVersionRole.STAGING:
                raise RetrievalPublicationRefused("retrieval_target_not_publishable")
            if target.state is RetrievalDataVersionState.BUILDING:
                target = self._transition_complete_bootstrap(
                    target,
                    self._catalog.mark_data_version_ready_in_transaction,
                )
            if target.state is not RetrievalDataVersionState.READY:
                raise RetrievalPublicationRefused("retrieval_target_not_publishable")
            return self._transition_complete_bootstrap(
                target,
                self._catalog.activate_data_version_in_transaction,
            )
        except (DocumentGenerationTerminalFailure, DocumentGenerationRetryableFailure) as exc:
            raise RetrievalPublicationRefused(exc.reason_code) from exc

    def publish_rebuilt_generation(
        self,
        *,
        expected_active_version_id: str | None,
    ) -> RetrievalDataVersion:
        """在 Document 权威围栏内发布一个完整重建的 generation。

        与首次 bootstrap 不同，重建允许旧 ACTIVE 继续服务直至最后一刻。调用方冻结其
        观察到的旧活动指针；READY 与 ACTIVE 两次转换都在同一 Document SQLite 的
        ``BEGIN IMMEDIATE`` 中重新验证该指针、完整 Source manifest 以及逐方法物理
        表示。任何并发切换或来源漂移都会失败关闭，而不会覆盖较新的活动 generation。
        """

        if expected_active_version_id is not None and (
            not isinstance(expected_active_version_id, str)
            or not expected_active_version_id.strip()
        ):
            raise ValueError("expected_active_version_id must be non-empty or None")
        if self._method_store is None:
            raise RetrievalPublicationRefused(
                "retrieval_method_publication_proof_unavailable"
            )
        target = self._catalog.get_data_version(self._generation_spec.version_id)
        try:
            if target is None:
                raise DocumentGenerationTerminalFailure("retrieval_target_missing")
            self._require_exact_identity(target)
            if target.role is RetrievalDataVersionRole.ACTIVE:
                with self._memory_connection() as conn:
                    conn.execute("BEGIN IMMEDIATE")
                    try:
                        self._require_expected_active_in_connection(
                            conn,
                            expected_active_version_id,
                        )
                        current = self._catalog.get_data_version_in_transaction(
                            conn,
                            target.id,
                        )
                        if current is None:
                            raise DocumentGenerationTerminalFailure("retrieval_target_missing")
                        self._require_exact_identity(current)
                    except Exception:
                        conn.rollback()
                        raise
                    else:
                        conn.commit()
                return current
            if target.role is not RetrievalDataVersionRole.STAGING:
                raise DocumentGenerationTerminalFailure("retrieval_target_not_publishable")
            if target.state is RetrievalDataVersionState.BUILDING:
                target = self._transition_complete_bootstrap(
                    target,
                    lambda conn, version_id: self._guarded_rebuild_transition(
                        conn,
                        version_id,
                        expected_active_version_id=expected_active_version_id,
                        transition=self._catalog.mark_data_version_ready_in_transaction,
                    ),
                )
            if target.state is not RetrievalDataVersionState.READY:
                raise DocumentGenerationTerminalFailure("retrieval_target_not_publishable")
            return self._transition_complete_bootstrap(
                target,
                lambda conn, version_id: self._guarded_rebuild_transition(
                    conn,
                    version_id,
                    expected_active_version_id=expected_active_version_id,
                    transition=self._catalog.activate_data_version_in_transaction,
                ),
            )
        except (DocumentGenerationTerminalFailure, DocumentGenerationRetryableFailure) as exc:
            raise RetrievalPublicationRefused(exc.reason_code) from exc
        except RetrievalCatalogError as exc:
            raise RetrievalPublicationRefused(
                "retrieval_catalog_transition_failed"
            ) from exc

    def restore_previous_generation(
        self,
        *,
        previous_generation_id: str,
        expected_fingerprint: str,
        expected_active_version_id: str | None,
    ) -> RetrievalDataVersion:
        """在同一 Document 写围栏内重验并恢复精确 PREVIOUS。"""

        if previous_generation_id != self._generation_spec.version_id or (
            expected_fingerprint != self._generation_spec.fingerprint
        ):
            raise RetrievalPublicationRefused(
                "retrieval_rollback_identity_mismatch"
            )
        if self._method_store is None:
            raise RetrievalPublicationRefused(
                "retrieval_method_publication_proof_unavailable"
            )
        try:
            with self._memory_connection() as conn:
                conn.execute("BEGIN IMMEDIATE")
                try:
                    self._require_expected_active_in_connection(
                        conn,
                        expected_active_version_id,
                    )
                    target = self._catalog.get_data_version_in_transaction(
                        conn,
                        previous_generation_id,
                    )
                    if target is None:
                        raise DocumentGenerationTerminalFailure("retrieval_target_missing")
                    self._require_exact_identity(target)
                    if (
                        target.role is not RetrievalDataVersionRole.PREVIOUS
                        or target.state is not RetrievalDataVersionState.READY
                    ):
                        raise DocumentGenerationTerminalFailure(
                            "retrieval_rollback_target_not_ready_previous"
                        )
                    self._require_complete_bootstrap_in_connection(conn, target)
                    restored = (
                        self._catalog.restore_previous_data_version_in_transaction(
                            conn,
                            target.id,
                            expected_active_version_id=expected_active_version_id,
                            require_active_match=True,
                        )
                    )
                except Exception:
                    conn.rollback()
                    raise
                else:
                    conn.commit()
            self._require_exact_identity(restored)
            return restored
        except (DocumentGenerationTerminalFailure, DocumentGenerationRetryableFailure) as exc:
            raise RetrievalPublicationRefused(exc.reason_code) from exc
        except RetrievalCatalogError as exc:
            raise RetrievalPublicationRefused(
                "retrieval_catalog_rollback_failed"
            ) from exc

    def _select_source_commit_target(self) -> RetrievalDataVersion:
        active = self._catalog.active_data_version()
        if active is not None:
            return self._require_exact_active(active)

        exact = self._catalog.get_data_version(self._generation_spec.version_id)
        if exact is not None:
            self._require_exact_identity(exact)
            if (
                exact.role is RetrievalDataVersionRole.STAGING
                and exact.state is RetrievalDataVersionState.BUILDING
            ):
                return exact
            if (
                exact.role is RetrievalDataVersionRole.STAGING
                and exact.state is RetrievalDataVersionState.READY
            ):
                return self._transition_complete_bootstrap(
                    exact,
                    self._catalog.activate_data_version_in_transaction,
                )
            raise DocumentGenerationTerminalFailure("retrieval_target_not_writable")

        try:
            target, _ = self._catalog.ensure_staging_data_version(
                version_id=self._generation_spec.version_id,
                fingerprint=self._generation_spec.fingerprint,
            )
        except Exception as exc:
            raise DocumentGenerationTerminalFailure("retrieval_staging_conflict") from exc
        self._require_exact_identity(target)
        if (
            target.role is not RetrievalDataVersionRole.STAGING
            or target.state is not RetrievalDataVersionState.BUILDING
        ):
            raise DocumentGenerationTerminalFailure("retrieval_target_not_writable")
        return target

    def _refresh_source_commit_target(
        self,
        selected: RetrievalDataVersion,
    ) -> RetrievalDataVersion:
        """若 catalog 生命周期在来源权威状态前发生变化，则失败关闭。"""

        current = self._catalog.get_data_version(selected.id)
        if current is None:
            raise DocumentGenerationTerminalFailure("retrieval_target_missing")
        self._require_exact_identity(current)
        active = self._catalog.active_data_version()
        if active is not None:
            if active.id != current.id:
                raise DocumentGenerationTerminalFailure("active_generation_fingerprint_mismatch")
            return self._require_exact_active(active)
        if (
            current.role is RetrievalDataVersionRole.STAGING
            and current.state is RetrievalDataVersionState.BUILDING
        ):
            return current
        if (
            current.role is RetrievalDataVersionRole.STAGING
            and current.state is RetrievalDataVersionState.READY
        ):
            return self._transition_complete_bootstrap(
                current,
                self._catalog.activate_data_version_in_transaction,
            )
        raise DocumentGenerationTerminalFailure("retrieval_target_not_writable")

    def _require_writable_source_commit_target_in_connection(
        self,
        conn: sqlite3.Connection,
        selected: RetrievalDataVersion,
    ) -> RetrievalDataVersion:
        """在来源提交锁内重验此前选择的精确 generation。"""

        current = self._catalog.get_data_version_in_transaction(conn, selected.id)
        if current is None:
            raise DocumentGenerationTerminalFailure("retrieval_target_missing")
        self._require_exact_identity(current)
        active = self._catalog.active_data_version_in_transaction(conn)
        if active is not None and active.id != current.id:
            raise DocumentGenerationTerminalFailure("active_generation_fingerprint_mismatch")
        try:
            writable = self._catalog.require_writable_data_version_in_transaction(
                conn,
                current.id,
            )
        except RetrievalCatalogError as exc:
            raise DocumentGenerationTerminalFailure("retrieval_target_not_writable") from exc
        self._require_exact_identity(writable)
        if active is not None:
            return self._require_exact_active(writable)
        if (
            writable.role is not RetrievalDataVersionRole.STAGING
            or writable.state is not RetrievalDataVersionState.BUILDING
        ):
            raise DocumentGenerationTerminalFailure("retrieval_target_not_writable")
        return writable

    def _transition_complete_bootstrap(
        self,
        target: RetrievalDataVersion,
        transition: Callable[
            [sqlite3.Connection, str],
            RetrievalDataVersion,
        ],
    ) -> RetrievalDataVersion:
        """发布一次 catalog 转换时冻结 memory 权威状态。"""

        self._require_exact_identity(target)
        with self._memory_connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                current = self._catalog.get_data_version_in_transaction(
                    conn,
                    target.id,
                )
                if current is None:
                    raise DocumentGenerationTerminalFailure("retrieval_target_missing")
                self._require_exact_identity(current)
                self._require_complete_bootstrap_in_connection(conn, current)
                transitioned = transition(conn, current.id)
            except Exception:
                conn.rollback()
                raise
            else:
                # 此事务不写入来源记录；commit 只在 catalog 转换后释放权威锁。
                conn.commit()
        self._require_exact_identity(transitioned)
        return transitioned

    def _guarded_rebuild_transition(
        self,
        conn: sqlite3.Connection,
        version_id: str,
        *,
        expected_active_version_id: str | None,
        transition: Callable[
            [sqlite3.Connection, str],
            RetrievalDataVersion,
        ],
    ) -> RetrievalDataVersion:
        self._require_expected_active_in_connection(
            conn,
            expected_active_version_id,
        )
        return transition(conn, version_id)

    def _require_expected_active_in_connection(
        self,
        conn: sqlite3.Connection,
        expected_active_version_id: str | None,
    ) -> None:
        active = self._catalog.active_data_version_in_transaction(conn)
        active_id = active.id if active is not None else None
        if active_id != expected_active_version_id:
            raise DocumentGenerationTerminalFailure("active_generation_changed_during_rollout")

    def _require_complete_bootstrap_in_connection(
        self,
        conn: sqlite3.Connection,
        target: RetrievalDataVersion,
    ) -> None:
        status_rows = conn.execute(
            "SELECT e.status, COUNT(*) AS count FROM retrieval_update_outbox e "
            "WHERE e.source_type IN (?, ?) AND e.data_version_id=? AND NOT ("
            "e.status=? AND EXISTS ("
            "SELECT 1 FROM retrieval_update_outbox cleanup "
            "WHERE cleanup.source_type=e.source_type "
            "AND cleanup.data_version_id=e.data_version_id "
            "AND cleanup.source_unit_id=e.source_unit_id "
            "AND cleanup.source_revision=e.source_revision "
            "AND cleanup.indexed_content_hash=e.indexed_content_hash "
            "AND cleanup.kind IN (?, ?) AND cleanup.status=? "
            "AND cleanup.authority_sequence > e.authority_sequence"
            ")) GROUP BY e.status",
            (
                SourceType.DOCUMENT.value,
                SourceType.PICTURE.value,
                target.id,
                OutboxStatus.TERMINAL_FAILED.value,
                RetrievalUpdateKind.TRASH.value,
                RetrievalUpdateKind.PURGE.value,
                OutboxStatus.APPLIED.value,
            ),
        ).fetchall()
        bindings = self._all_current_file_bindings(conn)
        status_counts = {
            OutboxStatus(str(row["status"])): int(row["count"])
            for row in status_rows
        }
        if status_counts.get(OutboxStatus.TERMINAL_FAILED, 0):
            raise DocumentGenerationTerminalFailure("bootstrap_retrieval_outbox_terminal_failure")
        if any(
            count
            for status, count in status_counts.items()
            if status is not OutboxStatus.APPLIED
        ):
            raise DocumentGenerationRetryableFailure("bootstrap_retrieval_outbox_incomplete")
        if not bindings:
            raise DocumentGenerationTerminalFailure("bootstrap_file_corpus_empty")
        if not self._bindings_are_covered(target, bindings, conn=conn):
            raise DocumentGenerationRetryableFailure("bootstrap_retrieval_coverage_incomplete")
        expected_manifest = frozenset(
            self._retrieval_manifest_entry(
                binding.ref,
                binding.source_filter,
            )
            for binding in bindings
        )
        searchable_manifest = frozenset(
            self._retrieval_manifest_entry(
                stored.unit.ref,
                stored.unit.source_filter,
            )
            for stored in self._catalog.list_stored_units_in_transaction(
                conn,
                target.id,
            )
            if (
                stored.index_state is UnitIndexState.READY
                and stored.unit.retrieval_status is RetrievalStatus.ACTIVE
            )
        )
        if searchable_manifest != expected_manifest:
            raise DocumentGenerationRetryableFailure("bootstrap_retrieval_manifest_not_exact")
        if self._method_store is not None:
            self._require_exact_method_coverage_in_connection(conn, target)

    def _require_exact_method_coverage_in_connection(
        self,
        conn: sqlite3.Connection,
        target: RetrievalDataVersion,
    ) -> None:
        required_methods = frozenset(
            document_index_methods(self._generation_spec.index_recipe)
        )
        health = self._method_store.generation_method_index_health_in_connection(
            conn,
            target.id,
        )
        stored_unit_ids = {
            stored.unit_id
            for stored in self._catalog.list_stored_units_in_transaction(
                conn,
                target.id,
            )
        }
        if set(health) != stored_unit_ids:
            raise DocumentGenerationRetryableFailure(
                "bootstrap_retrieval_method_coverage_incomplete"
            )
        for entries in health.values():
            by_method = {entry.method: entry for entry in entries}
            for method in (
                RetrievalMethod.DENSE,
                RetrievalMethod.LEARNED_SPARSE,
                RetrievalMethod.BM25,
            ):
                entry = by_method.get(method)
                required = method in required_methods
                if (
                    entry is None
                    or entry.expected_state != ("ready" if required else "absent")
                    or entry.representation_present is not required
                ):
                    raise DocumentGenerationRetryableFailure(
                        "bootstrap_retrieval_method_coverage_incomplete"
                    )

    @staticmethod
    def _retrieval_manifest_entry(
        ref: SourceUnitRef,
        source_filter: SourceFilter | None,
    ) -> tuple[str, str, str, str, tuple[tuple[str, str], ...] | None]:
        return (
            ref.source_type.value,
            ref.source_unit_id,
            ref.source_revision,
            ref.indexed_content_hash,
            source_filter.scope if source_filter is not None else None,
        )

    def _bindings_are_covered(
        self,
        target: RetrievalDataVersion,
        bindings: tuple[_CoverageBinding, ...],
        *,
        conn: sqlite3.Connection | None = None,
    ) -> bool:
        if not bindings:
            return False
        for binding in bindings:
            stored = (
                self._catalog.get_unit(binding.ref, target.id)
                if conn is None
                else self._catalog.get_unit_in_transaction(
                    conn,
                    binding.ref,
                    target.id,
                )
            )
            if (
                stored is None
                or stored.index_state is not UnitIndexState.READY
                or stored.unit.ref != binding.ref
                or stored.unit.retrieval_data_version != target.id
                or stored.unit.retrieval_status is not RetrievalStatus.ACTIVE
                or stored.unit.source_filter != binding.source_filter
            ):
                return False
        return True

    def _current_document_bindings(
        self,
        conn: sqlite3.Connection,
        *,
        document_id: str,
        document_version_id: str,
    ) -> tuple[_CoverageBinding, ...]:
        current = conn.execute(
            "SELECT current_version_id FROM documents WHERE id=?",
            (document_id,),
        ).fetchone()
        if current is None or str(current["current_version_id"] or "") != (
            document_version_id
        ):
            raise DocumentGenerationTerminalFailure("document_authority_changed")
        chunks = conn.execute(
            "SELECT id, producer_chunk_id, content, source_version_id FROM doc_chunks "
            "WHERE doc_id=? AND source_version_id=? ORDER BY seq, id",
            (document_id, document_version_id),
        ).fetchall()
        if not chunks:
            raise DocumentGenerationTerminalFailure("document_authority_unusable")
        bindings: list[_CoverageBinding] = []
        source_filter = SourceFilter.from_mapping(
            SourceType.DOCUMENT,
            {"doc_id": document_id},
        )
        for chunk in chunks:
            if chunk["producer_chunk_id"] is None:
                raise DocumentGenerationTerminalFailure("project_document_chunk_identity_missing")
            ref, _ = mounted_document_chunk_ref_and_content(
                session_id="project-index",
                chunk_id=str(chunk["id"]),
                source_version_id=str(chunk["source_version_id"]),
                content=str(chunk["content"]),
                doc_id=document_id,
                producer_chunk_id=str(chunk["producer_chunk_id"]),
            )
            bindings.append(_CoverageBinding(ref=ref, source_filter=source_filter))
        bindings.sort(key=lambda binding: binding.digest_tuple)
        return tuple(bindings)


    def _all_current_document_bindings(
        self,
        conn: sqlite3.Connection,
    ) -> tuple[_CoverageBinding, ...]:
        rows = conn.execute(
            "SELECT d.id, d.current_version_id FROM documents AS d "
            "WHERE d.current_version_id IS NOT NULL ORDER BY d.id"
        ).fetchall()

        bindings: list[_CoverageBinding] = []
        for row in rows:
            bindings.extend(
                self._current_document_bindings(
                    conn,
                    document_id=str(row["id"]),
                    document_version_id=str(row["current_version_id"]),
                )
            )
        bindings.sort(key=lambda binding: binding.digest_tuple)
        return tuple(bindings)

    def _all_current_file_bindings(
        self,
        conn: sqlite3.Connection,
    ) -> tuple[_CoverageBinding, ...]:
        """Freeze the complete Document + Picture authority manifest."""

        bindings = list(self._all_current_document_bindings(conn))
        read_picture_manifest = getattr(
            self._picture_source_reader,
            "list_indexable_units_for_backfill_in_transaction",
            None,
        )
        if not callable(read_picture_manifest):
            raise DocumentGenerationTerminalFailure(
                "bootstrap_picture_transaction_reader_unavailable"
            )
        try:
            picture_units = tuple(
                read_picture_manifest(
                    conn,
                    SourceFilter.from_mapping(SourceType.PICTURE)
                )
            )
        except Exception as exc:
            raise DocumentGenerationRetryableFailure("bootstrap_picture_source_read_failed") from exc
        for unit in picture_units:
            if (
                unit.ref.source_type is not SourceType.PICTURE
                or unit.source_filter.source_type is not SourceType.PICTURE
            ):
                raise DocumentGenerationTerminalFailure("bootstrap_picture_source_contract_invalid")
            bindings.append(
                _CoverageBinding(
                    ref=unit.ref,
                    source_filter=unit.source_filter,
                )
            )
        bindings.sort(
            key=lambda binding: self._retrieval_manifest_entry(
                binding.ref,
                binding.source_filter,
            )
        )
        return tuple(bindings)

    def _require_exact_target_version(
        self,
        target_id: str | None,
    ) -> RetrievalDataVersion:
        if not target_id:
            raise DocumentGenerationTerminalFailure("retrieval_target_missing")
        target = self._catalog.get_data_version(target_id)
        if target is None:
            raise DocumentGenerationTerminalFailure("retrieval_target_missing")
        self._require_exact_identity(target)
        active = self._catalog.active_data_version()
        if active is not None and active.id != target.id:
            raise DocumentGenerationTerminalFailure("active_generation_fingerprint_mismatch")
        if (
            target.role is RetrievalDataVersionRole.ACTIVE
            and target.state is RetrievalDataVersionState.READY
        ):
            return target
        if target.role is RetrievalDataVersionRole.STAGING and target.state in {
            RetrievalDataVersionState.BUILDING,
            RetrievalDataVersionState.READY,
        }:
            return target
        raise DocumentGenerationTerminalFailure("retrieval_target_not_writable")

    def _require_exact_active(
        self,
        active: RetrievalDataVersion,
    ) -> RetrievalDataVersion:
        if (
            active.role is not RetrievalDataVersionRole.ACTIVE
            or active.state is not RetrievalDataVersionState.READY
        ):
            raise DocumentGenerationTerminalFailure("active_generation_not_ready")
        if (
            active.id != self._generation_spec.version_id
            or active.fingerprint != self._generation_spec.fingerprint
        ):
            raise DocumentGenerationTerminalFailure("active_generation_fingerprint_mismatch")
        return active

    def _require_exact_identity(self, target: RetrievalDataVersion) -> None:
        if (
            target.id != self._generation_spec.version_id
            or target.fingerprint != self._generation_spec.fingerprint
        ):
            raise DocumentGenerationTerminalFailure("retrieval_target_fingerprint_mismatch")

    @contextmanager
    def _memory_connection(self) -> Iterator[sqlite3.Connection]:
        conn = self._connect_documents()
        if not isinstance(conn, sqlite3.Connection):
            raise TypeError("connect_documents must return sqlite3.Connection")
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
        finally:
            conn.close()


__all__ = [
    "DocumentGenerationAuthority",
    "RetrievalPublicationRefused",
]
