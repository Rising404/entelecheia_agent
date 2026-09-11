"""支持崩溃恢复的持久化本地文档 ingest 编排。

worker 在一个项目的 ``documents.sqlite`` 中协调来源与派生索引状态，而 Session
挂载能力仍保留在该 Session 的 ``session.sqlite`` 中。来源提交会先冻结目标
generation，并在 ``CHUNKED`` 检查点原子记录每个 Outbox 事件；后续 attempt 可在
不重新解析或提交来源的情况下建立索引并证明精确覆盖。
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager, contextmanager, nullcontext
from contextvars import copy_context
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import sqlite3
import threading
from typing import Iterator

from personagraph.input_processing.documents.chunking import ChunkingProfile
from personagraph.input_processing.documents.preparation import (
    DocumentPrepareFailure,
    PreparedDocumentIngest,
    prepare_document_path,
)
from personagraph.input_processing.documents.readers import (
    configured_processor_fingerprint,
)
from personagraph.input_processing.files import (
    SourceChangedDuringReadError,
    SourceFingerprint,
    SourceSizeLimitError,
    fingerprint_file,
)
from personagraph.workspace.documents import application as docstore
from personagraph.workspace.documents.indexing import (
    DocumentIndexAuditPort,
    DocumentIndexPort,
)
from .storage import (
    DocumentIngestJob,
    DocumentIngestJobLeaseError,
    DocumentIngestJobStage,
    DocumentIngestJobStatus,
    DocumentIngestJobTransitionError,
    DocumentMaintenanceRunReport,
    FilePreparationRequest,
    FilePreparationDeliveryStatus,
    SqliteFilePreparationRequestStore,
    SqliteDocumentIngestJobStore,
)
from .indexing_ports import IngestionGenerationIdentity, IngestionIndexPort
from .worker_errors import (
    DocumentIngestRetryableFailure,
    DocumentIngestTerminalFailure,
)


PrepareDocument = Callable[..., PreparedDocumentIngest | DocumentPrepareFailure]
IngestPrepared = Callable[..., docstore.DocumentIngestCommitReceipt]
Clock = Callable[[], str]
FaultHook = Callable[[str, DocumentIngestJob], None]
SourceAuthorityValidator = Callable[[str, str], bool]
ProjectFileLinker = Callable[[str, str | None], tuple[str, str]]
JobScopeFactory = Callable[[str], AbstractContextManager[object]]
RequestDelivery = Callable[..., int]


@dataclass(frozen=True, slots=True)
class _RequestAttempt:
    request: FilePreparationRequest
    source_fingerprint: SourceFingerprint


class _RequestAuthorityRevoked(DocumentIngestRetryableFailure):
    def __init__(self, request_id: str) -> None:
        self.request_id = request_id
        super().__init__("workspace_authority_revoked")


class DocumentMaintenanceWorker:
    """租用一个持久化 ingest job，并将其驱动到安全停止点。

    解析、来源提交和持久化 job 的租约/检查点由 Workspace 持有；派生索引与
    generation 发布只通过窄端口调用，不读取检索内部的 catalog 或 Outbox 状态。

    Job 只证明项目级来源与索引；Session 授权来自独立 request，并在每次尝试重新选择。
    挂载交付在共享结果完成后独立执行，交付失败不会撤销一个已完成 job。
    """

    def __init__(
        self,
        *,
        worker_id: str,
        indexing: IngestionIndexPort,
        connect_documents: Callable[[], sqlite3.Connection],
        job_store: SqliteDocumentIngestJobStore | None = None,
        request_store: SqliteFilePreparationRequestStore | None = None,
        request_delivery: RequestDelivery,
        chunking_profile: ChunkingProfile,
        now: Clock | None = None,
        prepare_document: PrepareDocument = prepare_document_path,
        ingest_prepared: IngestPrepared = docstore.ingest_prepared_in_transaction,
        document_index_port: DocumentIndexPort,
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
    ) -> None:
        if not isinstance(worker_id, str) or not worker_id.strip():
            raise ValueError("worker_id must be a non-empty string")
        if not callable(connect_documents):
            raise TypeError("connect_documents must be callable")
        if not callable(link_project_file):
            raise TypeError("link_project_file must be callable")
        for name, value in (
            ("lease_seconds", lease_seconds),
            ("retry_after_seconds", retry_after_seconds),
            ("outbox_batch_limit", outbox_batch_limit),
            ("max_outbox_batches", max_outbox_batches),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        heartbeat_interval = (
            lease_seconds / 3
            if lease_heartbeat_interval_seconds is None
            else lease_heartbeat_interval_seconds
        )
        if (
            isinstance(heartbeat_interval, bool)
            or not isinstance(heartbeat_interval, (int, float))
            or heartbeat_interval <= 0
            or heartbeat_interval >= lease_seconds
        ):
            raise ValueError(
                "lease_heartbeat_interval_seconds must be positive and shorter "
                "than lease_seconds"
            )
        self._worker_id = worker_id
        self._indexing = indexing
        self._connect_documents = connect_documents
        selected_index_port = document_index_port
        if not isinstance(selected_index_port, DocumentIndexAuditPort):
            raise TypeError(
                "document maintenance requires an index port with audit support"
            )
        self._document_index_port = selected_index_port
        self._jobs = job_store or SqliteDocumentIngestJobStore(
            index_audit_port=selected_index_port,
        )
        self._requests = request_store or SqliteFilePreparationRequestStore()
        if not callable(request_delivery):
            raise TypeError("request_delivery must be callable")
        self._request_delivery = request_delivery
        self._chunking_profile = chunking_profile
        self._now = now or _utc_now
        self._prepare_document = prepare_document
        self._ingest_prepared = ingest_prepared
        self._source_fingerprint = source_fingerprint
        self._processor_fingerprint = processor_fingerprint
        self._validate_source_authority = validate_source_authority
        self._link_project_file = link_project_file
        self._job_scope_factory = job_scope_factory
        self._fault_hook = fault_hook
        self._lease_seconds = lease_seconds
        self._lease_heartbeat_interval_seconds = float(heartbeat_interval)
        self._lease_heartbeat_join_timeout_seconds = max(
            1.0,
            min(5.0, lease_seconds / 2),
        )
        self._retry_after_seconds = retry_after_seconds
        self._outbox_batch_limit = outbox_batch_limit
        self._max_outbox_batches = max_outbox_batches

    @property
    def generation_identity(self) -> IngestionGenerationIdentity:
        """此 worker 的派生索引端口所绑定的不可变 generation 身份。"""

        return self._indexing.generation_identity

    def run_once(self, *, limit: int = 16) -> DocumentMaintenanceRunReport:
        """认领并串行处理至多 ``limit`` 个到期 job。

        每次认领一个 job，使缓慢解析或索引过程不会导致批次中其余 job 的租约在启动前
        过期。有意不捕获 ``BaseException``：进程终止与显式崩溃注入必须让持久化
        lease/checkpoint 成为唯一恢复权威。
        """

        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise ValueError("limit must be a positive integer")
        claimed_count = 0
        applied = 0
        retryable_failed = 0
        terminal_failed = 0
        lease_lost = 0

        for _ in range(limit):
            with self._memory_connection() as conn:
                claimed = self._jobs.claim_due(
                    conn,
                    worker_id=self._worker_id,
                    now=self._now(),
                    lease_seconds=self._lease_seconds,
                    limit=1,
                    target_generation_id=self.generation_identity.version_id,
                    target_generation_fingerprint=self.generation_identity.fingerprint,
                )
            if not claimed:
                break
            job = claimed[0]
            claimed_count += 1
            request: FilePreparationRequest | None = None
            try:
                request = self._select_authorized_request(job)
                with self._request_scope(request.session_id):
                    attempt = _RequestAttempt(
                        request=request,
                        source_fingerprint=self._observe_source(job, request),
                    )
                    self._process_claimed(job, attempt)
            except _RequestAuthorityRevoked as exc:
                self._block_request(exc.request_id)
                if self._mark_retryable(job, exc.reason_code):
                    retryable_failed += 1
                else:
                    lease_lost += 1
            except DocumentIngestTerminalFailure as exc:
                # 索引端也会复验临时 Session。只对该 request 的撤权重试共享 job；
                # corpus 中其他不允许的来源仍维持明确的终止语义。
                request_revoked = (
                    exc.reason_code == "workspace_authority_revoked"
                    and request is not None
                    and not self._source_is_authorized(request.session_id, job.canonical_path)
                )
                if request_revoked:
                    self._block_request(request.request_id)
                    if self._mark_retryable(job, exc.reason_code):
                        retryable_failed += 1
                    else:
                        lease_lost += 1
                elif self._mark_terminal(job, exc.reason_code):
                    terminal_failed += 1
                else:
                    lease_lost += 1
            except DocumentIngestRetryableFailure as exc:
                if self._mark_retryable(job, exc.reason_code):
                    retryable_failed += 1
                else:
                    lease_lost += 1
            except DocumentIngestJobLeaseError:
                lease_lost += 1
            except Exception:
                if self._mark_retryable(job, "document_worker_internal_error"):
                    retryable_failed += 1
                else:
                    lease_lost += 1
            else:
                applied += 1

            self._request_delivery(job_id=job.job_id)

        # APPLIED 后才出现的新请求没有可认领的 job，仍须在本次维护中获得交付。
        self._request_delivery()
        return DocumentMaintenanceRunReport(
            claimed=claimed_count,
            applied=applied,
            retryable_failed=retryable_failed,
            terminal_failed=terminal_failed,
            lease_lost=lease_lost,
        )

    def drain_outbox_once(self, *, limit: int | None = None) -> int:
        """即使没有到期 ingest job，也应用一个有界 Outbox 批次。

        Session detach/delete 与其他文档生命周期操作会发布检索事件而不创建 ingest
        job。此独立清空过程让这些事件保持活跃；无需 wake 信号，因为生命周期会在
        每次持久化维护过程轮询本方法。
        """

        batch_limit = self._outbox_batch_limit if limit is None else limit
        if (
            isinstance(batch_limit, bool)
            or not isinstance(batch_limit, int)
            or batch_limit <= 0
        ):
            raise ValueError("limit must be a positive integer")
        return self._indexing.drain_once(
            worker_id=f"{self._worker_id}:retrieval",
            now=self._now(),
            lease_seconds=self._lease_seconds,
            limit=batch_limit,
        )

    def _process_claimed(
        self, job: DocumentIngestJob, attempt: _RequestAttempt,
    ) -> DocumentIngestJob:
        self._require_live_token(job)
        self._revalidate_attempt(job, attempt)
        current = job
        while True:
            if current.stage is DocumentIngestJobStage.PARSING:
                current = self._parse_and_commit(current, attempt)
                continue
            if current.stage is DocumentIngestJobStage.CHUNKED:
                current = self._checkpoint_indexing(current)
                continue
            if current.stage is DocumentIngestJobStage.INDEXING:
                current = self._index_and_prove(current)
                continue
            if current.stage is DocumentIngestJobStage.COVERAGE_READY:
                return self._publish_and_apply(current, attempt)
            if (
                current.stage is DocumentIngestJobStage.ACTIVE
                and current.status is DocumentIngestJobStatus.APPLIED
            ):
                return current
            raise DocumentIngestTerminalFailure("invalid_document_ingest_checkpoint")

    def _parse_and_commit(
        self, job: DocumentIngestJob, attempt: _RequestAttempt,
    ) -> DocumentIngestJob:
        renewed = self._renew(job)
        reused = self._reuse_durable_content_artifact(renewed, attempt)
        if reused is not None:
            return reused
        with self._heartbeat_job_lease(renewed):
            prepared = self._prepare_document(
                renewed.canonical_path,
                chunking_profile=self._chunking_profile,
            )
        if isinstance(prepared, DocumentPrepareFailure):
            if prepared.reason.startswith("read_error:") or prepared.reason == (
                "source_changed_during_ingest"
            ):
                raise DocumentIngestRetryableFailure("document_prepare_temporarily_failed")
            raise DocumentIngestTerminalFailure("document_parse_rejected_terminal")
        if not isinstance(prepared, PreparedDocumentIngest):
            raise DocumentIngestTerminalFailure("document_prepare_contract_violation")
        self._require_prepared_matches_job(prepared, renewed, attempt)
        # 解析不执行写入。只在准入成功后选择/创建检索 generation，同时仍在来源权威
        # 事务前将其冻结。致命 parser 拒绝不得遗留一个会阻塞未来 recipe 的空 BUILDING
        # staging generation 阶段。
        renewed = self._renew(renewed)
        self._require_source_authority(renewed, attempt.request)
        target = self._select_source_target(renewed)
        self._require_source_authority(renewed, attempt.request)
        target = self._indexing.refresh_source_target(target)
        file_link = self._link_source_file(
            prepared.canonical_path,
            prepared.mime,
        )
        self._require_file_link(renewed, file_link)

        with self._memory_connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                # 在任何 memory 权威写入前的最后安全点重新打开 Session 授权。此检查
                # 读取 sessions.sqlite，而非事务持有的 memory 连接。
                self._require_source_authority(renewed, attempt.request)
                target = (
                    self._indexing
                    .require_write_target_in_transaction(
                        conn,
                        target,
                    )
                )
                ingest_kwargs: dict[str, object] = {
                    "prepared": prepared,
                    "session_id": None,
                    "retrieval_data_version": target,
                    "document_index_port": self._document_index_port,
                }
                ingest_kwargs.update(
                    file_id=file_link[0],
                    file_version_id=file_link[1],
                )
                receipt = self._ingest_prepared(conn, **ingest_kwargs)
                checkpoint = self._checkpoint_source_commit_in_transaction(
                    conn,
                    job=renewed,
                    attempt=attempt,
                    target=target,
                    receipt=receipt,
                )
            except Exception:
                conn.rollback()
                raise
            else:
                conn.commit()
        self._fault("after_source_commit", checkpoint)
        return checkpoint

    def _reuse_durable_content_artifact(
        self,
        job: DocumentIngestJob,
        attempt: _RequestAttempt,
    ) -> DocumentIngestJob | None:
        """精确持久化有类型 artifact 已存在时跳过解析。

        初步查找避免为普通 cache-miss 路径创建空检索 generation。权威查找与 clone 会
        在来源事务内再次运行；并发 miss 只会回退到规范 parser。
        """

        with self._memory_connection() as conn:
            reusable = docstore.has_reusable_content_artifact(
                conn,
                canonical_path=job.canonical_path,
                source_sha256=job.source_sha256,
                processor_fingerprint=job.processor_fingerprint,
                chunker_fingerprint=job.chunker_fingerprint,
                chunk_contract_version=job.chunk_contract_version,
            )
        if not reusable:
            return None

        renewed = self._renew(job)
        self._revalidate_attempt(renewed, attempt)
        target = self._select_source_target(renewed)
        self._require_source_authority(renewed, attempt.request)
        target = self._indexing.refresh_source_target(target)
        file_link = self._link_source_file(renewed.canonical_path, None)
        self._require_file_link(renewed, file_link)
        with self._memory_connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                self._require_source_authority(renewed, attempt.request)
                target = (
                    self._indexing
                    .require_write_target_in_transaction(
                        conn,
                        target,
                    )
                )
                receipt = docstore.reuse_content_artifact_in_transaction(
                    conn,
                    canonical_path=renewed.canonical_path,
                    session_id=None,
                    source_fingerprint=attempt.source_fingerprint,
                    processor_fingerprint=renewed.processor_fingerprint,
                    chunker_fingerprint=renewed.chunker_fingerprint,
                    chunk_contract_version=renewed.chunk_contract_version,
                    retrieval_data_version=target,
                    document_index_port=self._document_index_port,
                    file_id=file_link[0],
                    file_version_id=file_link[1],
                )
                if receipt is None:
                    conn.rollback()
                    return None
                checkpoint = self._checkpoint_source_commit_in_transaction(
                    conn,
                    job=renewed,
                    attempt=attempt,
                    target=target,
                    receipt=receipt,
                )
            except Exception:
                conn.rollback()
                raise
            else:
                conn.commit()
        self._fault("after_source_commit", checkpoint)
        return checkpoint

    def _checkpoint_source_commit_in_transaction(
        self,
        conn: sqlite3.Connection,
        *,
        job: DocumentIngestJob,
        attempt: _RequestAttempt,
        target: str,
        receipt: docstore.DocumentIngestCommitReceipt,
    ) -> DocumentIngestJob:
        """将一次原子来源提交链接到其精确派生索引工作。"""

        linked_event_ids = self._indexing.ensure_source_events_in_transaction(
            conn,
            target_id=target,
            document_id=receipt.document_id,
            authorization_session_id=attempt.request.session_id,
            receipt_event_ids=receipt.retrieval_event_ids,
        )
        return self._jobs.checkpoint_stage_in_transaction(
            conn,
            job_id=job.job_id,
            worker_id=self._worker_id,
            lease_token=self._require_live_token(job),
            stage=DocumentIngestJobStage.CHUNKED,
            now=self._now(),
            document_id=receipt.document_id,
            document_version_id=receipt.document_version_id,
            retrieval_data_version=target,
            outbox_event_ids=linked_event_ids,
        )

    def _checkpoint_indexing(self, job: DocumentIngestJob) -> DocumentIngestJob:
        renewed = self._renew(job)
        with self._memory_connection() as conn:
            return self._jobs.checkpoint_stage(
                conn,
                job_id=renewed.job_id,
                worker_id=self._worker_id,
                lease_token=self._require_live_token(renewed),
                stage=DocumentIngestJobStage.INDEXING,
                now=self._now(),
            )

    def _index_and_prove(self, job: DocumentIngestJob) -> DocumentIngestJob:
        renewed = self._renew(job)
        # Outbox leases protect individual events, not this parent job. Model
        # loading, many encoding batches, and coverage checks can exceed its TTL.
        with self._heartbeat_job_lease(renewed):
            proof = self._indexing.index_and_prove(
                target_id=renewed.retrieval_data_version,
                document_id=renewed.document_id,
                document_version_id=renewed.document_version_id,
                event_ids=self._linked_event_ids(renewed.job_id),
                worker_id=f"{self._worker_id}:retrieval",
                now=self._now,
                lease_seconds=self._lease_seconds,
                batch_limit=self._outbox_batch_limit,
                max_batches=self._max_outbox_batches,
            )
        renewed = self._renew(renewed)
        with self._memory_connection() as conn:
            try:
                return self._jobs.checkpoint_stage(
                    conn,
                    job_id=renewed.job_id,
                    worker_id=self._worker_id,
                    lease_token=self._require_live_token(renewed),
                    stage=DocumentIngestJobStage.COVERAGE_READY,
                    now=self._now(),
                    coverage_proof=proof,
                )
            except DocumentIngestJobTransitionError as exc:
                raise DocumentIngestRetryableFailure(
                    "document_authority_changed_during_coverage"
                ) from exc

    def _publish_and_apply(
        self, job: DocumentIngestJob, attempt: _RequestAttempt,
    ) -> DocumentIngestJob:
        renewed = self._renew(job)
        self._revalidate_attempt(renewed, attempt)
        with self._heartbeat_job_lease(renewed):
            self._indexing.publish_proven_target(
                target_id=renewed.retrieval_data_version,
                document_id=renewed.document_id,
                document_version_id=renewed.document_version_id,
                expected_binding_digest=renewed.coverage_binding_digest,
                expected_binding_count=renewed.coverage_expected_bindings,
                expected_generation_fingerprint=renewed.coverage_data_version_fingerprint,
                on_transition=lambda point: self._fault(point, renewed),
            )

        renewed = self._renew(renewed)
        with self._memory_connection() as conn:
            return self._jobs.mark_applied(
                conn,
                job_id=renewed.job_id,
                worker_id=self._worker_id,
                lease_token=self._require_live_token(renewed),
                now=self._now(),
            )

    def _select_authorized_request(self, job: DocumentIngestJob) -> FilePreparationRequest:
        with self._memory_connection() as conn:
            requests = self._requests.list_for_job(
                conn, job.job_id,
                delivery_status=FilePreparationDeliveryStatus.PENDING,
                limit=256,
            )
        for request in requests:
            try:
                with self._request_scope(request.session_id):
                    authorized = self._source_is_authorized(
                        request.session_id, job.canonical_path,
                    )
            except Exception:
                authorized = False
            if authorized:
                return request
            self._block_request(request.request_id)
        raise DocumentIngestRetryableFailure("file_preparation_authority_unavailable")

    def _request_scope(self, session_id: str) -> AbstractContextManager[object]:
        return (
            self._job_scope_factory(session_id)
            if self._job_scope_factory is not None
            else nullcontext()
        )

    def _observe_source(
        self, job: DocumentIngestJob, request: FilePreparationRequest,
    ) -> SourceFingerprint:
        self._require_source_authority(job, request)
        path = Path(job.canonical_path)
        try:
            observed = self._source_fingerprint(path)
            processor = self._processor_fingerprint(path)
        except SourceSizeLimitError as exc:
            raise DocumentIngestTerminalFailure("frozen_source_size_exceeded") from exc
        except (SourceChangedDuringReadError, OSError) as exc:
            raise DocumentIngestRetryableFailure("source_revalidation_unavailable") from exc
        if observed.sha256 != job.source_sha256 or observed.size_bytes != job.source_size:
            raise DocumentIngestTerminalFailure("frozen_source_mismatch")
        if processor is None or str(processor) != job.processor_fingerprint:
            raise DocumentIngestTerminalFailure("frozen_processor_mismatch")
        if (
            job.chunker_fingerprint != self._chunking_profile.fingerprint()
            or job.chunk_contract_version != docstore.DOCUMENT_CHUNK_CONTRACT_VERSION
        ):
            raise DocumentIngestTerminalFailure("frozen_chunk_recipe_mismatch")
        self._require_file_link(job, self._link_source_file(job.canonical_path, None))
        return observed

    def _revalidate_attempt(
        self, job: DocumentIngestJob, attempt: _RequestAttempt,
    ) -> None:
        if self._observe_source(job, attempt.request) != attempt.source_fingerprint:
            raise DocumentIngestRetryableFailure("source_changed_during_attempt")

    def _require_source_authority(
        self, job: DocumentIngestJob, request: FilePreparationRequest,
    ) -> None:
        if not self._source_is_authorized(request.session_id, job.canonical_path):
            raise _RequestAuthorityRevoked(request.request_id)

    def _source_is_authorized(self, session_id: str, canonical_path: str) -> bool:
        try:
            return self._validate_source_authority(session_id, canonical_path) is True
        except Exception:
            return False

    def _block_request(self, request_id: str) -> None:
        with self._memory_connection() as conn:
            self._requests.mark_blocked(
                conn, request_id,
                reason_code="workspace_authority_revoked",
                now=self._now(),
            )

    def _select_source_target(self, job: DocumentIngestJob) -> str:
        if (
            job.target_generation_id != self.generation_identity.version_id
            or job.target_generation_fingerprint != self.generation_identity.fingerprint
        ):
            raise DocumentIngestTerminalFailure("frozen_generation_mismatch")
        target = self._indexing.select_source_target()
        if target != job.target_generation_id:
            raise DocumentIngestTerminalFailure("frozen_generation_mismatch")
        return target

    @staticmethod
    def _require_file_link(job: DocumentIngestJob, linked: tuple[str, str]) -> None:
        if linked != (job.file_id, job.file_version_id):
            raise DocumentIngestTerminalFailure("frozen_file_version_mismatch")

    def _link_source_file(
        self,
        canonical_path: str,
        media_type: str | None,
    ) -> tuple[str, str]:
        try:
            linked = self._link_project_file(canonical_path, media_type)
        except DocumentIngestTerminalFailure:
            raise
        except Exception as exc:
            raise DocumentIngestTerminalFailure(
                "document_project_file_link_failed"
            ) from exc
        if linked is None:
            raise DocumentIngestTerminalFailure(
                "document_project_file_link_missing"
            )
        return linked

    def _require_prepared_matches_job(
        self,
        prepared: PreparedDocumentIngest,
        job: DocumentIngestJob,
        attempt: _RequestAttempt,
    ) -> None:
        prepared_path = str(Path(prepared.canonical_path).resolve())
        if prepared_path != str(Path(job.canonical_path).resolve()):
            raise DocumentIngestTerminalFailure("prepared_source_path_mismatch")
        observed = prepared.source_fingerprint
        if (
            observed.sha256 != job.source_sha256
            or observed.size_bytes != job.source_size
        ):
            raise DocumentIngestTerminalFailure("frozen_source_mismatch")
        if observed != attempt.source_fingerprint:
            raise DocumentIngestRetryableFailure("source_changed_during_attempt")
        if prepared.processor_fingerprint != job.processor_fingerprint:
            raise DocumentIngestTerminalFailure("frozen_processor_mismatch")
        if prepared.chunker_fingerprint != job.chunker_fingerprint:
            raise DocumentIngestTerminalFailure("frozen_chunk_recipe_mismatch")

    def _linked_event_ids(self, job_id: str) -> tuple[str, ...]:
        with self._memory_connection() as conn:
            return self._jobs.list_outbox_event_ids(conn, job_id)

    def _renew(self, job: DocumentIngestJob) -> DocumentIngestJob:
        with self._memory_connection() as conn:
            return self._jobs.renew_lease(
                conn,
                job_id=job.job_id,
                worker_id=self._worker_id,
                lease_token=self._require_live_token(job),
                now=self._now(),
                lease_seconds=self._lease_seconds,
            )

    @contextmanager
    def _heartbeat_job_lease(self, job: DocumentIngestJob) -> Iterator[None]:
        """为长操作保持父 job 的精确租约；离开后才能推进其检查点。

        心跳只覆盖可能耗时的操作，不包住最终 mark_applied：终态已释放租约，
        此后继续续租会把一次真实成功误报成 lease_lost。
        """

        stop = threading.Event()
        error_lock = threading.Lock()
        heartbeat_error: list[BaseException] = []

        def renew_periodically() -> None:
            while not stop.wait(self._lease_heartbeat_interval_seconds):
                try:
                    self._renew(job)
                except BaseException as exc:
                    with error_lock:
                        heartbeat_error.append(exc)
                    stop.set()
                    return

        inherited_context = copy_context()
        thread = threading.Thread(
            target=lambda: inherited_context.run(renew_periodically),
            name=f"personagraph-document-job-heartbeat:{self._worker_id}",
            daemon=True,
        )
        thread.start()
        body_completed = False
        try:
            yield
            body_completed = True
        finally:
            stop.set()
            thread.join(self._lease_heartbeat_join_timeout_seconds)
            # 保留操作自身异常。在常规路径上，孤立续租或失败心跳使租约归属不确定时，
            # 来源权威状态不得继续推进。
            if body_completed:
                if thread.is_alive():
                    raise DocumentIngestRetryableFailure("document_job_heartbeat_stop_timeout")
                with error_lock:
                    error = heartbeat_error[0] if heartbeat_error else None
                if isinstance(error, DocumentIngestJobLeaseError):
                    raise error
                if error is not None:
                    raise DocumentIngestRetryableFailure(
                        "document_job_heartbeat_failed"
                    ) from error

    def _mark_retryable(self, job: DocumentIngestJob, reason_code: str) -> bool:
        try:
            with self._memory_connection() as conn:
                self._jobs.mark_retryable_failure(
                    conn,
                    job_id=job.job_id,
                    worker_id=self._worker_id,
                    lease_token=self._require_live_token(job),
                    now=self._now(),
                    retry_after_seconds=self._retry_after_seconds,
                    reason_code=reason_code,
                )
        except DocumentIngestJobLeaseError:
            return False
        return True

    def _mark_terminal(self, job: DocumentIngestJob, reason_code: str) -> bool:
        try:
            with self._memory_connection() as conn:
                self._jobs.mark_terminal_failure(
                    conn,
                    job_id=job.job_id,
                    worker_id=self._worker_id,
                    lease_token=self._require_live_token(job),
                    now=self._now(),
                    reason_code=reason_code,
                )
        except DocumentIngestJobLeaseError:
            return False
        return True

    @staticmethod
    def _require_live_token(job: DocumentIngestJob) -> str:
        if not job.lease_token:
            raise DocumentIngestJobLeaseError("document ingest job has no live lease token")
        return job.lease_token

    def _fault(self, point: str, job: DocumentIngestJob) -> None:
        if self._fault_hook is not None:
            self._fault_hook(point, job)

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



def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


__all__ = [
    "DocumentMaintenanceWorker",
]
