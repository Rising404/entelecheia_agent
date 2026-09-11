"""File 与 current-session Retrieval 的生产组合根。

本模块把稳定合同与 ``indexing``、``sources``、``lifecycle``、``orchestration`` 中的具体实现
组装为一个 ``RetrievalFoundation``。它是多个子域依赖汇合的位置，因此保留在包根，而不归入
任一具体子包，避免 Indexing 或 Sources 反向取得整个系统的组合权。

Foundation 只负责构造可用组件和安全诊断快照；它不选择 L0/L1/L2、不决定某个 Turn 是否启用
检索、不扩大 Session/Document 授权，也不把模型输出写回权威来源。上述决策由外层 Runtime、
Source authority 与持久化事务各自负责。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from .lifecycle.backfill import RetrievalBackfillService
from .contracts import CorpusKey, RetrievalMethod, SourceType
from .lifecycle.corpus import FILE_CORPUS, SESSION_CORPUS, RetrievalCorpusBinding
from .sources.coverage.current_session import CurrentSessionIndexCoverageProbe
from .sources.coverage.document import DocumentIndexCoverageProbe
from .sources.coverage.picture import PictureIndexCoverageProbe
from .lifecycle.generation import (
    ExactGenerationDataVersionProvider,
    RetrievalGenerationSpec,
)
from .indexing.adapters import (
    BM25Retrieval,
    DenseRetrieval,
    LearnedSparseRetrieval,
    LiteralBooleanFallback,
    SqliteRetrievalIndexWriter,
)
from .indexing.encoder import DeterministicLexicalEncoder
from .indexing.methods import (
    SqliteRetrievalMethodStore,
)
from .lifecycle.maintenance import RetrievalReconciler
from .policy import DefaultRetrievalPolicy
from .ports import (
    BgeM3EncoderPort,
    RerankerPort,
    RetrievalDataVersionProvider,
    SourceRetrievalAdapter,
)
from .query_guard import QueryGuard
from .orchestration.selection import SourceSelectionQuotas
from .service import RetrievalService
from .sources.document import MountedDocumentChunkSourceAdapter
from .sources.picture import PictureObservationSourceAdapter
from .sqlite_store import RetrievalProjectContextRequired, SqliteRetrievalCatalog
from .lifecycle.sync import RetrievalSyncService
from .indexing.token_estimation import BgeM3RetrievalTokenEstimator


@dataclass(frozen=True, slots=True)
class RetrievalFoundation:
    """一次明确语料绑定下的完整 Retrieval 组件集合。

    File 与 current-session 使用同一内核但拥有独立 corpus/generation。调用方应冻结并复用这个
    bundle，而不是分别创建 Catalog、Service 和 DataVersionProvider 后自行混配。
    """

    corpus_key: CorpusKey
    catalog: SqliteRetrievalCatalog
    method_store: SqliteRetrievalMethodStore
    service: RetrievalService
    sync_service: RetrievalSyncService
    reconciler: RetrievalReconciler
    backfill_service: RetrievalBackfillService
    retrieval_token_estimator: BgeM3RetrievalTokenEstimator
    data_version_provider: RetrievalDataVersionProvider
    retrieval_methods: tuple[RetrievalMethod, ...]
    source_adapters: Mapping[SourceType, SourceRetrievalAdapter]
    generation_spec: RetrievalGenerationSpec | None = None
    reranker: RerankerPort | None = None

    def diagnostic_snapshot(self) -> dict[str, object]:
        """安全的运维状态：只含能力和数量，绝不包含 Source 内容。"""

        active = self.data_version_provider.active_data_version()
        token_estimator = self.retrieval_token_estimator.diagnostic_snapshot()
        with self.catalog.connect() as conn:
            units = int(conn.execute("SELECT COUNT(*) FROM retrieval_units").fetchone()[0])
            ready = int(
                conn.execute(
                    "SELECT COUNT(*) FROM retrieval_units WHERE retrieval_status='active' AND index_state='ready'"
                ).fetchone()[0]
            )
        return {
            "corpus": self.corpus_key.value,
            "active_data_version": active.id if active else None,
            "active_fingerprint": active.fingerprint if active else None,
            "retrieval_methods": tuple(method.value for method in self.retrieval_methods),
            "catalog_unit_count": units,
            "ready_unit_count": ready,
            "capabilities": {
                "fts5": self.catalog.capability("fts5"),
                "sqlite_vec": self.catalog.capability("sqlite_vec"),
                "encoder": self.method_store.encoder_capability_snapshot(),
                "reranker": _reranker_diagnostic_snapshot(self.reranker),
                "retrieval_token_estimator": {
                    "kind": token_estimator.kind,
                    "fallback_count": token_estimator.fallback_count,
                    "cache_entries": token_estimator.cache_entries,
                },
            },
            "runtime_activation": "consumer_composed",
        }


def build_file_retrieval_foundation(
    *,
    db_path: Path | str | None = None,
    encoder: BgeM3EncoderPort | None = None,
    source_selection_quotas: SourceSelectionQuotas | None = None,
    generation_spec: RetrievalGenerationSpec | None = None,
    reranker: RerankerPort | None = None,
    reranker_candidate_limit_per_source: int = 32,
    retrieval_methods: tuple[RetrievalMethod, ...] | None = None,
    picture_source_adapter: SourceRetrievalAdapter | None = None,
) -> RetrievalFoundation:
    """构建 Document + Picture File 语料；Picture 在线权限默认关闭。"""

    if (
        picture_source_adapter is not None
        and getattr(picture_source_adapter, "source_type", None)
        is not SourceType.PICTURE
    ):
        raise ValueError("picture_source_adapter must declare Picture source_type")

    return _build_retrieval_foundation(
        binding=FILE_CORPUS,
        db_path=db_path,
        encoder=encoder,
        source_selection_quotas=source_selection_quotas,
        generation_spec=generation_spec,
        reranker=reranker,
        reranker_candidate_limit_per_source=reranker_candidate_limit_per_source,
        retrieval_methods=retrieval_methods,
        source_adapters={
            SourceType.DOCUMENT: MountedDocumentChunkSourceAdapter(),
            SourceType.PICTURE: (
                picture_source_adapter
                if picture_source_adapter is not None
                else PictureObservationSourceAdapter()
            ),
        },
    )


def build_session_retrieval_foundation(
    *,
    db_path: Path | str,
    source_adapter: SourceRetrievalAdapter,
    encoder: BgeM3EncoderPort | None = None,
    source_selection_quotas: SourceSelectionQuotas | None = None,
    generation_spec: RetrievalGenerationSpec | None = None,
    reranker: RerankerPort | None = None,
    reranker_candidate_limit_per_source: int = 32,
    retrieval_methods: tuple[RetrievalMethod, ...] | None = None,
) -> RetrievalFoundation:
    """构建仅含当前 Session、由 retrieval 域拥有的独立派生索引。"""

    if getattr(source_adapter, "source_type", None) is not SourceType.CURRENT_SESSION:
        raise ValueError("session retrieval requires a current-session adapter")
    return _build_retrieval_foundation(
        binding=SESSION_CORPUS,
        db_path=db_path,
        encoder=encoder,
        source_selection_quotas=source_selection_quotas,
        generation_spec=generation_spec,
        reranker=reranker,
        reranker_candidate_limit_per_source=reranker_candidate_limit_per_source,
        retrieval_methods=retrieval_methods,
        source_adapters={SourceType.CURRENT_SESSION: source_adapter},
    )


def _build_retrieval_foundation(
    *,
    binding: RetrievalCorpusBinding,
    db_path: Path | str | None,
    encoder: BgeM3EncoderPort | None,
    source_selection_quotas: SourceSelectionQuotas | None,
    generation_spec: RetrievalGenerationSpec | None,
    reranker: RerankerPort | None,
    reranker_candidate_limit_per_source: int,
    retrieval_methods: tuple[RetrievalMethod, ...] | None,
    source_adapters: Mapping[SourceType, SourceRetrievalAdapter] | None = None,
) -> RetrievalFoundation:
    """在共享全部检索内核实现的同时构建一个语料库。

    此函数刻意不创建或激活 RetrievalDataVersion。发布步骤必须显式进行，因为模型或
    tokenizer 指纹变化需要安全的暂存重建，而非隐式切换生产版本。
    """

    if (
        generation_spec is not None
        and generation_spec.source_types != binding.generation_source_types
    ):
        raise ValueError(
            f"generation source_types do not match the {binding.key.value} corpus"
        )
    if db_path is not None:
        catalog = SqliteRetrievalCatalog(db_path)
    else:
        catalog = SqliteRetrievalCatalog()
    catalog.initialize()
    retrieval_encoder = encoder or DeterministicLexicalEncoder()
    data_version_provider: RetrievalDataVersionProvider = (
        ExactGenerationDataVersionProvider(catalog=catalog, spec=generation_spec)
        if generation_spec is not None
        else catalog
    )
    method_store = SqliteRetrievalMethodStore(catalog=catalog, encoder=retrieval_encoder)
    configured_methods = (
        tuple(RetrievalMethod(method) for method in retrieval_methods)
        if retrieval_methods is not None
        else None
    )
    if configured_methods is not None and not configured_methods:
        raise ValueError("retrieval_methods must not be empty")
    retrieval_token_estimator = BgeM3RetrievalTokenEstimator(retrieval_encoder)
    sources = _build_sources(binding, source_adapters=source_adapters)
    methods = {
        RetrievalMethod.DENSE: DenseRetrieval(method_store),
        RetrievalMethod.LEARNED_SPARSE: LearnedSparseRetrieval(method_store),
        RetrievalMethod.BM25: BM25Retrieval(method_store),
        RetrievalMethod.LITERAL_BOOLEAN: LiteralBooleanFallback(method_store),
    }
    sync_service = RetrievalSyncService(
        catalog=catalog,
        source_readers=sources,
        index_writer=SqliteRetrievalIndexWriter(
            method_store,
            required_methods=configured_methods or (),
        ),
    )
    backfill_sources = dict(sources)
    source_coverage_probes: dict[SourceType, object] = {}
    source_access_revalidators: dict[SourceType, object] = {}
    readonly_sources = None
    readonly_source_coverage_probes = None
    readonly_source_access_revalidators = None
    if binding.key is CorpusKey.FILE:
        document_source = sources[SourceType.DOCUMENT]
        picture_source = sources[SourceType.PICTURE]
        source_coverage_probes[SourceType.DOCUMENT] = DocumentIndexCoverageProbe(
            catalog=catalog,
            binding_reader=document_source,
        )
        source_coverage_probes[SourceType.PICTURE] = PictureIndexCoverageProbe(
            catalog=catalog,
            binding_reader=picture_source,
        )
        source_access_revalidators[SourceType.DOCUMENT] = document_source
        source_access_revalidators[SourceType.PICTURE] = picture_source
        # 候选优先的 File 检索承诺真正无写入的读取。因此为它提供专用 SourceExecution，
        # 其首次新鲜度观察为只读；普通来源仍作为重放调用方和维护诊断所用的持久审计路径。
        readonly_document_source = MountedDocumentChunkSourceAdapter(
            preflight_mode="read_only"
        )
        readonly_sources = {
            SourceType.DOCUMENT: readonly_document_source,
            SourceType.PICTURE: picture_source,
        }
        readonly_source_coverage_probes = {
            SourceType.DOCUMENT: DocumentIndexCoverageProbe(
                catalog=catalog,
                binding_reader=readonly_document_source,
            ),
            SourceType.PICTURE: PictureIndexCoverageProbe(
                catalog=catalog,
                binding_reader=picture_source,
            ),
        }
        readonly_source_access_revalidators = {
            SourceType.DOCUMENT: readonly_document_source,
            SourceType.PICTURE: picture_source,
        }
    else:
        session_source = sources[SourceType.CURRENT_SESSION]
        source_coverage_probes[SourceType.CURRENT_SESSION] = (
            CurrentSessionIndexCoverageProbe(
                catalog=catalog,
                binding_reader=session_source,
            )
        )
        source_access_revalidators[SourceType.CURRENT_SESSION] = session_source
    return RetrievalFoundation(
        corpus_key=binding.key,
        catalog=catalog,
        method_store=method_store,
        service=RetrievalService(
            policy=(
                DefaultRetrievalPolicy(default_methods=configured_methods)
                if configured_methods is not None
                else DefaultRetrievalPolicy()
            ),
            query_guard=QueryGuard(
                count_tokens=lambda text: len(
                    tuple(retrieval_encoder.token_ids(text))
                )
            ),
            sources=sources,
            methods=methods,
            source_coverage_probes=source_coverage_probes,
            source_access_revalidators=source_access_revalidators,
            readonly_sources=readonly_sources,
            readonly_source_coverage_probes=(
                readonly_source_coverage_probes
            ),
            readonly_source_access_revalidators=(
                readonly_source_access_revalidators
            ),
            data_version_provider=data_version_provider,
            token_estimator=retrieval_token_estimator,
            source_selection_quotas=source_selection_quotas,
            encoder_fingerprint=retrieval_encoder.fingerprint(),
            reranker=reranker,
            reranker_candidate_limit_per_source=reranker_candidate_limit_per_source,
        ),
        sync_service=sync_service,
        reconciler=RetrievalReconciler(
            catalog=catalog,
            method_store=method_store,
            source_readers=backfill_sources,
            sync_service=sync_service,
        ),
        backfill_service=RetrievalBackfillService(
            catalog=catalog,
            source_readers=backfill_sources,
            sync_service=sync_service,
        ),
        retrieval_token_estimator=retrieval_token_estimator,
        data_version_provider=data_version_provider,
        retrieval_methods=(
            configured_methods
            if configured_methods is not None
            else (
                RetrievalMethod.DENSE,
                RetrievalMethod.LEARNED_SPARSE,
            )
        ),
        source_adapters=MappingProxyType(dict(sources)),
        generation_spec=generation_spec,
        reranker=reranker,
    )


def _build_sources(
    binding: RetrievalCorpusBinding,
    *,
    source_adapters: Mapping[SourceType, SourceRetrievalAdapter] | None,
) -> dict[SourceType, SourceRetrievalAdapter]:
    if source_adapters is None:
        if binding.key is not CorpusKey.FILE:
            raise RetrievalProjectContextRequired(
                "session retrieval requires an explicit Source adapter"
            )
        return {
            SourceType.DOCUMENT: MountedDocumentChunkSourceAdapter(),
            SourceType.PICTURE: PictureObservationSourceAdapter(),
        }
    sources = dict(source_adapters)
    if set(sources) != set(binding.source_types):
        raise ValueError("Source adapters do not match retrieval corpus membership")
    for source_type, adapter in sources.items():
        if getattr(adapter, "source_type", None) is not source_type:
            raise ValueError("Source adapter declares a mismatched SourceType")
    return sources


def _reranker_diagnostic_snapshot(reranker: RerankerPort | None) -> dict[str, object]:
    if reranker is None:
        return {"enabled": False, "fingerprint": None}
    snapshot = getattr(reranker, "diagnostic_snapshot", None)
    if callable(snapshot):
        try:
            details = dict(snapshot())
        except Exception as exc:
            return {
                "enabled": True,
                "fingerprint": None,
                "diagnostic_error": type(exc).__name__,
            }
        return {"enabled": True, **details}
    try:
        fingerprint = reranker.fingerprint()
    except Exception as exc:
        return {
            "enabled": True,
            "fingerprint": None,
            "diagnostic_error": type(exc).__name__,
        }
    return {"enabled": True, "fingerprint": fingerprint}
