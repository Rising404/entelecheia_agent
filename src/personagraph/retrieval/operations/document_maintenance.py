"""文件检索配方装配与纯 generation 重建/发布入口。"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
import sqlite3

from personagraph.input_processing.documents.chunking import ChunkingProfile
from personagraph.workspace.documents import application as docstore
from personagraph.workspace.ingestion.worker import (
    DocumentMaintenanceWorker,
    JobScopeFactory,
    ProjectFileLinker,
    RequestDelivery,
    SourceAuthorityValidator,
)
from personagraph.workspace.storage.context import (
    ProjectDocumentContextError,
    current as current_project_documents,
    initialize_current,
)
from ..contracts import SourceType
from ..foundation import RetrievalFoundation, build_file_retrieval_foundation
from ..lifecycle.generation import (
    RetrievalGenerationRestorePlan,
    RetrievalGenerationSpec,
    file_corpus_generation_spec,
)
from ..lifecycle.outbox import SqliteRetrievalOutbox
from ..lifecycle.rollout import DocumentGenerationRolloutService, GenerationRolloutReport
from ..lifecycle.sync import RetrievalOutboxConsumer
from ..ports import BgeM3EncoderPort, RerankerPort, SourceRetrievalAdapter
from ..profile import (
    DocumentRetrievalCapability,
    DocumentRetrievalProfile,
    build_document_retrieval_runtime,
)
from .document_generation import DocumentGenerationAuthority
from .ingestion_index import build_document_maintenance_worker


@dataclass(frozen=True, slots=True)
class DocumentRetrievalComposition:
    """Document maintenance and File retrieval share one exact generation."""

    foundation: RetrievalFoundation
    generation_spec: RetrievalGenerationSpec
    chunking_profile: ChunkingProfile
    encoder: BgeM3EncoderPort
    reranker: RerankerPort | None
    requested_profile: DocumentRetrievalProfile
    effective_profile: DocumentRetrievalProfile
    capability: DocumentRetrievalCapability
    degraded_reason: str | None = None

def build_document_retrieval_composition(
    *,
    retrieval_db_path: Path | str | None = None,
    profile: DocumentRetrievalProfile | None = None,
    encoder: BgeM3EncoderPort | None = None,
    reranker: RerankerPort | None = None,
    picture_source_adapter: SourceRetrievalAdapter | None = None,
) -> DocumentRetrievalComposition:
    """构建规范的 Document + Picture File generation 组合。

    这里有意只负责构造：初始化 schema 并计算不可变指纹，但既不清空 Outbox 工作，
    也不发布 DataVersion。维护 worker 与 Runtime 论文工具必须消费同一组合，确保查询
    embedding 与已索引分块绝不会声称不同的 generation 标识。外层 Host 可注入一个
    已绑定 Session 文件权限的 Picture adapter；省略时在线 Picture 读取保持默认拒绝。
    """

    initialize_current()
    runtime = build_document_retrieval_runtime(profile)
    chunking_profile = runtime.chunking_profile
    retrieval_encoder = encoder or runtime.encoder
    retrieval_reranker = reranker or runtime.reranker
    generation_spec = file_corpus_generation_spec(
        encoder_fingerprint=retrieval_encoder.fingerprint(),
        document_chunker_fingerprint=chunking_profile.fingerprint(),
        document_chunk_contract_version=docstore.DOCUMENT_CHUNK_CONTRACT_VERSION,
        index_recipe=runtime.effective_profile.index_recipe,
    )
    foundation = build_file_retrieval_foundation(
        db_path=retrieval_db_path,
        encoder=retrieval_encoder,
        generation_spec=generation_spec,
        reranker=retrieval_reranker,
        retrieval_methods=runtime.effective_profile.retrieval_methods,
        picture_source_adapter=picture_source_adapter,
    )
    return DocumentRetrievalComposition(
        foundation=foundation,
        generation_spec=generation_spec,
        chunking_profile=chunking_profile,
        encoder=retrieval_encoder,
        reranker=retrieval_reranker,
        requested_profile=runtime.requested_profile,
        effective_profile=runtime.effective_profile,
        capability=runtime.capability,
        degraded_reason=runtime.degraded_reason,
    )


def rollout_document_retrieval_generation(
    *,
    retrieval_db_path: Path | str | None = None,
    profile: DocumentRetrievalProfile | None = None,
    encoder: BgeM3EncoderPort | None = None,
    reranker: RerankerPort | None = None,
) -> GenerationRolloutReport:
    """为当前绑定 Project 全量重建并原子发布精确 Document generation。

    这是旧 lexical generation 切换到 BGE-M3 的显式维护入口。重建期间旧 ACTIVE 继续
    服务；只有完整 Source manifest、逐 Unit 指针以及 recipe 要求的每种物理表示均在
    Document SQLite 权威围栏内再次成立时，活动指针才会交换。
    """

    project_database = current_project_documents()
    if project_database is None:
        raise ProjectDocumentContextError(
            "document generation rollout requires a bound project database"
        )
    if retrieval_db_path is not None and (
        Path(retrieval_db_path).expanduser().resolve()
        != project_database.db_path.expanduser().resolve()
    ):
        raise ValueError(
            "document generation rollout requires the co-located project documents database"
        )
    composition = build_document_retrieval_composition(
        retrieval_db_path=project_database.db_path,
        profile=profile,
        encoder=encoder,
        reranker=reranker,
    )
    service = _build_document_generation_service(
        composition,
    )
    return service.rollout(composition.generation_spec)


def restore_previous_document_retrieval_generation(
    *,
    previous_generation_id: str,
    expected_fingerprint: str,
    retrieval_db_path: Path | str | None = None,
    profile: DocumentRetrievalProfile | None = None,
    encoder: BgeM3EncoderPort | None = None,
    reranker: RerankerPort | None = None,
) -> RetrievalGenerationRestorePlan:
    """按当前显式 profile 重验 Source 与三方法表示后原子恢复 PREVIOUS。"""

    project_database = current_project_documents()
    if project_database is None:
        raise ProjectDocumentContextError(
            "document generation restore requires a bound project database"
        )
    if retrieval_db_path is not None and (
        Path(retrieval_db_path).expanduser().resolve()
        != project_database.db_path.expanduser().resolve()
    ):
        raise ValueError(
            "document generation restore requires the co-located project documents database"
        )
    composition = build_document_retrieval_composition(
        retrieval_db_path=project_database.db_path,
        profile=profile,
        encoder=encoder,
        reranker=reranker,
    )
    service = _build_document_generation_service(
        composition,
    )
    return service.rollback(
        composition.generation_spec,
        previous_generation_id=previous_generation_id,
        expected_fingerprint=expected_fingerprint,
    )


def _build_document_generation_service(
    composition: DocumentRetrievalComposition,
) -> DocumentGenerationRolloutService:
    project_database = current_project_documents()
    if project_database is None:
        raise ProjectDocumentContextError(
            "document generation requires a bound project database"
        )
    foundation = composition.foundation
    publisher = DocumentGenerationAuthority(
        catalog=foundation.catalog,
        generation_spec=composition.generation_spec,
        method_store=foundation.method_store,
        connect_documents=project_database.open_connection,
        picture_source_reader=foundation.source_adapters[SourceType.PICTURE],
    )
    return DocumentGenerationRolloutService(
        catalog=foundation.catalog,
        method_store=foundation.method_store,
        backfill_service=foundation.backfill_service,
        document_source_reader=foundation.source_adapters[SourceType.DOCUMENT],
        picture_source_reader=foundation.source_adapters[SourceType.PICTURE],
        publisher=publisher,
    )


def build_document_ingestion_worker(
    composition: DocumentRetrievalComposition,
    *,
    worker_id: str,
    connect_documents: Callable[[], sqlite3.Connection],
    validate_source_authority: SourceAuthorityValidator,
    link_project_file: ProjectFileLinker,
    request_delivery: RequestDelivery,
    job_scope_factory: JobScopeFactory | None = None,
) -> DocumentMaintenanceWorker:
    """将一个已构造的 File 检索配方注入 Workspace worker，不决定 Session 权限。"""

    foundation = composition.foundation
    outbox = SqliteRetrievalOutbox()
    return build_document_maintenance_worker(
        worker_id=worker_id,
        catalog=foundation.catalog,
        generation_spec=composition.generation_spec,
        outbox_consumer=RetrievalOutboxConsumer(
            outbox=outbox,
            sync_service=foundation.sync_service,
        ),
        method_store=foundation.method_store,
        outbox=outbox,
        chunking_profile=composition.chunking_profile,
        connect_documents=connect_documents,
        picture_source_reader=foundation.source_adapters[SourceType.PICTURE],
        validate_source_authority=validate_source_authority,
        link_project_file=link_project_file,
        request_delivery=request_delivery,
        job_scope_factory=job_scope_factory,
    )


__all__ = [
    "DocumentRetrievalComposition",
    "build_document_retrieval_composition",
    "build_document_ingestion_worker",
    "restore_previous_document_retrieval_generation",
    "rollout_document_retrieval_generation",
]
