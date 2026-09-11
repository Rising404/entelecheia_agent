"""Workspace ingestion 的检索索引适配与显式装配。

解析、任务租约与来源提交归 Workspace；本模块只拥有 generation 选择、索引事件
排空、覆盖证明和安全发布，并将检索内部状态归一为 ingestion 端口的值与失败语义。
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
import sqlite3
from typing import Iterator, Protocol

from personagraph.input_processing.documents.chunking import ChunkingProfile
from personagraph.input_processing.documents.preparation import prepare_document_path
from personagraph.input_processing.documents.readers import configured_processor_fingerprint
from personagraph.input_processing.files import SourceFingerprint, fingerprint_file
from personagraph.workspace.documents import application as docstore
from personagraph.workspace.documents.indexing import DocumentIndexPort, enqueue_document_upserts
from personagraph.workspace.ingestion.indexing_ports import IngestionGenerationIdentity
from personagraph.workspace.ingestion.storage import (
    DocumentIngestCoverageProof,
    SqliteDocumentIngestJobStore,
    SqliteFilePreparationRequestStore,
    document_ingest_binding_digest,
)
from personagraph.workspace.ingestion.worker import (
    Clock,
    DocumentMaintenanceWorker,
    FaultHook,
    IngestPrepared,
    JobScopeFactory,
    PrepareDocument,
    ProjectFileLinker,
    RequestDelivery,
    SourceAuthorityValidator,
)
from personagraph.workspace.ingestion.worker_errors import (
    DocumentIngestRetryableFailure,
    DocumentIngestTerminalFailure,
)
from ..contracts import SourceType
from ..indexing.methods import SqliteRetrievalMethodStore
from ..lifecycle.backfill import BackfillSourceReader
from ..lifecycle.corpus import FILE_CORPUS
from ..lifecycle.generation import (
    RetrievalGenerationSpec,
    picture_observation_projection_fingerprint,
)
from ..lifecycle.outbox import OutboxStatus, SqliteRetrievalOutbox
from ..sources.events import SqliteDocumentIndexPort
from ..sources.picture import PictureObservationSourceAdapter
from ..sqlite_store import (
    RetrievalDataVersion,
    RetrievalDataVersionRole,
    RetrievalDataVersionState,
    SqliteRetrievalCatalog,
)
from .document_generation import (
    DocumentGenerationAuthority,
    DocumentGenerationRetryableFailure,
    DocumentGenerationTerminalFailure,
)


class _OutboxConsumer(Protocol):
    def consume_due(
        self,
        conn: sqlite3.Connection,
        *,
        worker_id: str,
        now: str,
        lease_seconds: int = 30,
        limit: int = 20,
    ) -> tuple[object, ...]: ...


@contextmanager
def _ingestion_failures() -> Iterator[None]:
    try:
        yield
    except DocumentGenerationRetryableFailure as exc:
        raise DocumentIngestRetryableFailure(exc.reason_code) from exc
    except DocumentGenerationTerminalFailure as exc:
        raise DocumentIngestTerminalFailure(exc.reason_code) from exc


class RetrievalIngestionIndex:
    """一个精确 File generation 对 Workspace 提供的索引能力。"""

    def __init__(
        self,
        *,
        catalog: SqliteRetrievalCatalog,
        generation_spec: RetrievalGenerationSpec,
        outbox_consumer: _OutboxConsumer,
        method_store: SqliteRetrievalMethodStore | None,
        connect_documents: Callable[[], sqlite3.Connection],
        outbox: SqliteRetrievalOutbox,
        chunking_profile: ChunkingProfile,
        document_index_port: DocumentIndexPort,
        picture_source_reader: BackfillSourceReader | None,
        validate_source_authority: SourceAuthorityValidator,
    ) -> None:
        if not isinstance(generation_spec, RetrievalGenerationSpec):
            raise TypeError("generation_spec must be RetrievalGenerationSpec")
        if generation_spec.source_types != FILE_CORPUS.generation_source_types:
            raise ValueError("document maintenance requires the complete File generation")
        if chunking_profile.fingerprint() != generation_spec.chunker_fingerprint:
            raise ValueError("generation spec and durable chunking profile must match")
        if (
            generation_spec.document_chunk_contract_version
            != docstore.DOCUMENT_CHUNK_CONTRACT_VERSION
        ):
            raise ValueError("generation spec and document chunk contract must match")
        resolved_picture_reader = (
            picture_source_reader
            if picture_source_reader is not None
            else PictureObservationSourceAdapter()
        )
        if resolved_picture_reader.source_type is not SourceType.PICTURE:
            raise ValueError("picture_source_reader must declare Picture source_type")
        picture_binding = generation_spec.source_binding(SourceType.PICTURE)
        try:
            observed_picture_projection = (
                picture_observation_projection_fingerprint(
                    observation_contract=picture_binding.projection_contract,
                    window_policy=getattr(
                        resolved_picture_reader,
                        "window_policy",
                        None,
                    ),
                )
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "picture_source_reader must expose the generation FIFO policy"
            ) from exc
        if observed_picture_projection != picture_binding.projection_fingerprint:
            raise ValueError("generation spec and Picture FIFO policy must match")

        if (
            method_store is not None
            and method_store.encoder_capability_snapshot().get("fingerprint")
            != generation_spec.encoder_fingerprint
        ):
            raise ValueError("generation spec and method-store encoder must match")
        self._catalog = catalog
        self._generation_spec = generation_spec
        self._outbox_consumer = outbox_consumer
        self._outbox = outbox
        self._connect_documents = connect_documents
        self._document_index_port = document_index_port
        self._validate_source_authority = validate_source_authority
        self._generation = DocumentGenerationAuthority(
            catalog=catalog,
            generation_spec=generation_spec,
            method_store=method_store,
            connect_documents=connect_documents,
            picture_source_reader=resolved_picture_reader,
        )

    @property
    def generation_identity(self) -> IngestionGenerationIdentity:
        return IngestionGenerationIdentity(
            version_id=self._generation_spec.version_id,
            fingerprint=self._generation_spec.fingerprint,
        )

    def select_source_target(self) -> str:
        with _ingestion_failures():
            return self._generation._select_source_commit_target().id

    def refresh_source_target(self, target_id: str) -> str:
        with _ingestion_failures():
            target = self._target(target_id)
            return self._generation._refresh_source_commit_target(target).id

    def require_write_target_in_transaction(
        self,
        conn: sqlite3.Connection,
        target_id: str,
    ) -> str:
        with _ingestion_failures():
            target = self._target(target_id, conn=conn)
            return (
                self._generation
                ._require_writable_source_commit_target_in_connection(conn, target)
                .id
            )

    def ensure_source_events_in_transaction(
        self,
        conn: sqlite3.Connection,
        *,
        target_id: str,
        document_id: str,
        authorization_session_id: str,
        receipt_event_ids: tuple[str, ...],
    ) -> tuple[str, ...]:
        with _ingestion_failures():
            target = self._target(target_id, conn=conn)
            # Bootstrap 覆盖整个已入库 File corpus；普通提交仅补足该文档的当前绑定。
            bootstrap = (
                target.role is RetrievalDataVersionRole.STAGING
                and target.state is RetrievalDataVersionState.BUILDING
            )
            ensured = self._enqueue_current_document_upserts(
                conn,
                retrieval_data_version=target.id,
                document_id=None if bootstrap else document_id,
                authorization_session_id=authorization_session_id,
            )
            return tuple(dict.fromkeys((
                *receipt_event_ids,
                *ensured.get(document_id, ()),
            )))

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
    ) -> DocumentIngestCoverageProof:
        with _ingestion_failures():
            self._generation._require_exact_target_version(target_id)
            outcomes = self._event_outcomes(event_ids)
            for _ in range(max_batches):
                self._raise_for_terminal_or_missing_event(outcomes)
                if all(
                    outcome is not None and outcome[0] is OutboxStatus.APPLIED
                    for outcome in outcomes
                ):
                    break
                self.drain_once(
                    worker_id=worker_id,
                    now=now(),
                    lease_seconds=lease_seconds,
                    limit=batch_limit,
                )
                outcomes = self._event_outcomes(event_ids)
            self._raise_for_terminal_or_missing_event(outcomes)
            if not all(
                outcome is not None and outcome[0] is OutboxStatus.APPLIED
                for outcome in outcomes
            ):
                raise DocumentIngestRetryableFailure("retrieval_outbox_incomplete")
            return self._require_coverage_proof(
                target_id=target_id,
                document_id=document_id,
                document_version_id=document_version_id,
            )

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
    ) -> None:
        with _ingestion_failures():
            target = self._generation._require_exact_target_version(target_id)
            proof = self._require_coverage_proof(
                target_id=target_id,
                document_id=document_id,
                document_version_id=document_version_id,
            )
            if (
                expected_binding_digest != proof.covered_binding_digest
                or expected_binding_count != proof.covered_binding_count
                or expected_generation_fingerprint
                != proof.retrieval_data_version_fingerprint
            ):
                raise DocumentIngestTerminalFailure("coverage_authority_changed")

            if target.role is RetrievalDataVersionRole.STAGING:
                if target.state is RetrievalDataVersionState.BUILDING:
                    target = self._generation._transition_complete_bootstrap(
                        target,
                        self._catalog.mark_data_version_ready_in_transaction,
                    )
                    on_transition("after_data_version_ready")
                if (
                    target.role is not RetrievalDataVersionRole.STAGING
                    or target.state is not RetrievalDataVersionState.READY
                ):
                    raise DocumentIngestTerminalFailure("retrieval_target_not_publishable")
                target = self._generation._transition_complete_bootstrap(
                    target,
                    self._catalog.activate_data_version_in_transaction,
                )
                on_transition("after_data_version_activate")

            if (
                target.role is not RetrievalDataVersionRole.ACTIVE
                or target.state is not RetrievalDataVersionState.READY
                or target.id != self._generation_spec.version_id
                or target.fingerprint != self._generation_spec.fingerprint
            ):
                raise DocumentIngestTerminalFailure("active_generation_fingerprint_mismatch")

    def drain_once(
        self,
        *,
        worker_id: str,
        now: str,
        lease_seconds: int,
        limit: int,
    ) -> int:
        with self._memory_connection() as conn:
            results = self._outbox_consumer.consume_due(
                conn,
                worker_id=worker_id,
                now=now,
                lease_seconds=lease_seconds,
                limit=limit,
            )
        return len(results)

    def _require_coverage_proof(
        self,
        *,
        target_id: str | None,
        document_id: str | None,
        document_version_id: str | None,
    ) -> DocumentIngestCoverageProof:
        if not document_id or not document_version_id:
            raise DocumentIngestTerminalFailure("document_authority_reference_missing")
        target = self._generation._require_exact_target_version(target_id)
        with self._memory_connection() as conn:
            bindings = self._generation._current_document_bindings(
                conn,
                document_id=document_id,
                document_version_id=document_version_id,
            )
        if not self._generation._bindings_are_covered(target, bindings):
            raise DocumentIngestRetryableFailure("retrieval_coverage_incomplete")
        digest_bindings = tuple(binding.digest_tuple for binding in bindings)
        return DocumentIngestCoverageProof(
            retrieval_data_version_id=target.id,
            retrieval_data_version_fingerprint=target.fingerprint,
            covered_binding_digest=document_ingest_binding_digest(digest_bindings),
            covered_binding_count=len(digest_bindings),
        )

    def _enqueue_current_document_upserts(
        self,
        conn: sqlite3.Connection,
        *,
        retrieval_data_version: str,
        document_id: str | None,
        authorization_session_id: str,
    ) -> dict[str, tuple[str, ...]]:
        """使当前项目绑定在一个精确 generation 中可达。

        ``document_id=None`` 是 bootstrap 形态，并有意扫描整个当前 corpus。UPSERT
        标识是确定性的，因此崩溃/重放或 DocStore 已发出的事件在 Outbox 边界均为
        no-op。此处不会隐藏无效挂载记录：corpus 全局发布证明仍会推导自身双向
        manifest，并在任何当前权威状态不可用或未进入索引时拒绝激活。
        """

        if not conn.in_transaction:
            raise RuntimeError("document bootstrap upserts require an authority transaction")
        parameters: tuple[object, ...] = ()
        document_predicate = ""
        if document_id is not None:
            if not document_id.strip():
                raise ValueError("document_id must not be empty")
            document_predicate = " AND d.id=?"
            parameters = (document_id,)
        rows = conn.execute(
            "SELECT d.id AS document_id, d.path AS canonical_path, "
            "SUM(CASE WHEN c.producer_chunk_id IS NULL THEN 1 ELSE 0 END) "
            "AS untyped_chunks FROM documents AS d "
            "JOIN doc_chunks AS c ON c.doc_id=d.id "
            "AND c.source_version_id=d.current_version_id "
            "WHERE d.current_version_id IS NOT NULL"
            + document_predicate
            + " GROUP BY d.id, d.path ORDER BY d.id",
            parameters,
        ).fetchall()
        sessions_by_document = {
            str(row["document_id"]): (
                str(row["canonical_path"]),
                (authorization_session_id,),
            )
            for row in rows
        }
        if any(int(row["untyped_chunks"]) for row in rows):
            raise DocumentIngestTerminalFailure("project_document_chunk_identity_missing")

        by_document: dict[str, tuple[str, ...]] = {}
        for current_document_id, (canonical_path, session_ids) in (
            sessions_by_document.items()
        ):
            for session_id in session_ids:
                self._require_source_authority_for(
                    session_id=session_id,
                    canonical_path=canonical_path,
                )
            by_document[current_document_id] = enqueue_document_upserts(
                conn,
                doc_id=current_document_id,
                retrieval_data_version=retrieval_data_version,
                index_port=self._document_index_port,
                session_ids=(),
            )
        return by_document

    def _require_source_authority_for(
        self,
        *,
        session_id: str,
        canonical_path: str,
    ) -> None:
        try:
            authorized = self._validate_source_authority(
                session_id,
                canonical_path,
            )
        except Exception as exc:
            raise DocumentIngestTerminalFailure(
                "workspace_authority_revoked"
            ) from exc
        if authorized is not True:
            raise DocumentIngestTerminalFailure("workspace_authority_revoked")

    def _target(
        self,
        target_id: str,
        *,
        conn: sqlite3.Connection | None = None,
    ) -> RetrievalDataVersion:
        target = (
            self._catalog.get_data_version_in_transaction(conn, target_id)
            if conn is not None
            else self._catalog.get_data_version(target_id)
        )
        if target is None:
            raise DocumentIngestTerminalFailure("retrieval_target_missing")
        return target

    def _event_outcomes(
        self,
        event_ids: tuple[str, ...],
    ) -> tuple[tuple[OutboxStatus, str | None] | None, ...]:
        with self._memory_connection() as conn:
            return tuple(self._outbox.get_outcome(conn, event_id) for event_id in event_ids)

    @staticmethod
    def _raise_for_terminal_or_missing_event(
        outcomes: tuple[tuple[OutboxStatus, str | None] | None, ...],
    ) -> None:
        if any(outcome is None for outcome in outcomes):
            raise DocumentIngestTerminalFailure("retrieval_outbox_event_missing")
        terminal_reasons = {
            reason
            for outcome in outcomes
            if outcome is not None
            for status, reason in (outcome,)
            if status is OutboxStatus.TERMINAL_FAILED and reason is not None
        }
        if "retrieval_method_unavailable" in terminal_reasons:
            raise DocumentIngestTerminalFailure("retrieval_method_unavailable")
        if any(
            outcome is not None and outcome[0] is OutboxStatus.TERMINAL_FAILED
            for outcome in outcomes
        ):
            raise DocumentIngestTerminalFailure("retrieval_outbox_terminal_failure")

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



def build_document_maintenance_worker(
    *,
    worker_id: str,
    catalog: SqliteRetrievalCatalog,
    generation_spec: RetrievalGenerationSpec,
    outbox_consumer: _OutboxConsumer,
    method_store: SqliteRetrievalMethodStore | None = None,
    connect_documents: Callable[[], sqlite3.Connection],
    job_store: SqliteDocumentIngestJobStore | None = None,
    request_store: SqliteFilePreparationRequestStore | None = None,
    request_delivery: RequestDelivery,
    outbox: SqliteRetrievalOutbox | None = None,
    chunking_profile: ChunkingProfile,
    now: Clock | None = None,
    prepare_document: PrepareDocument = prepare_document_path,
    ingest_prepared: IngestPrepared = docstore.ingest_prepared_in_transaction,
    document_index_port: DocumentIndexPort | None = None,
    picture_source_reader: BackfillSourceReader | None = None,
    source_fingerprint: Callable[[Path], SourceFingerprint] = fingerprint_file,
    processor_fingerprint: Callable[[Path], object | None] = (
        configured_processor_fingerprint
    ),
    validate_source_authority: SourceAuthorityValidator,
    link_project_file: ProjectFileLinker,
    job_scope_factory: JobScopeFactory | None = None,
    fault_hook: FaultHook | None = None,
    lease_seconds: int = 30,
    lease_heartbeat_interval_seconds: float | None = None,
    retry_after_seconds: int = 5,
    outbox_batch_limit: int = 20,
    max_outbox_batches: int = 16,
) -> DocumentMaintenanceWorker:
    """显式组合检索实现；核心 worker 不承担默认检索 backend 的构造。"""
    selected_port = document_index_port or SqliteDocumentIndexPort()
    indexing = RetrievalIngestionIndex(
        catalog=catalog,
        generation_spec=generation_spec,
        outbox_consumer=outbox_consumer,
        method_store=method_store,
        connect_documents=connect_documents,
        outbox=outbox or SqliteRetrievalOutbox(),
        chunking_profile=chunking_profile,
        document_index_port=selected_port,
        picture_source_reader=picture_source_reader,
        validate_source_authority=validate_source_authority,
    )
    return DocumentMaintenanceWorker(
        worker_id=worker_id,
        indexing=indexing,
        connect_documents=connect_documents,
        job_store=job_store,
        request_store=request_store,
        request_delivery=request_delivery,
        chunking_profile=chunking_profile,
        now=now,
        prepare_document=prepare_document,
        ingest_prepared=ingest_prepared,
        document_index_port=selected_port,
        source_fingerprint=source_fingerprint,
        processor_fingerprint=processor_fingerprint,
        validate_source_authority=validate_source_authority,
        link_project_file=link_project_file,
        job_scope_factory=job_scope_factory,
        fault_hook=fault_hook,
        lease_seconds=lease_seconds,
        lease_heartbeat_interval_seconds=lease_heartbeat_interval_seconds,
        retry_after_seconds=retry_after_seconds,
        outbox_batch_limit=outbox_batch_limit,
        max_outbox_batches=max_outbox_batches,
    )


__all__ = ["RetrievalIngestionIndex", "build_document_maintenance_worker"]
