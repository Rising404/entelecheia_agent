"""需显式启用的真实 BGE-M3 与 SQLite 检索冒烟测试框架。

BGE-M3 权重在本地可用后，使用
``python -m personagraph.retrieval.operations.real_smoke`` 运行。
它只创建临时派生数据库，绝不读取应用 Source 记录，也不激活 Runtime 检索。
"""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from ..contracts import RetrievalStatus, RetrievalUnit, SourceFilter, SourceType, SourceUnitRef
from ..indexing.methods import SqliteRetrievalMethodStore
from ..profile import (
    DocumentRetrievalProfile,
    build_document_retrieval_runtime,
)
from ..sqlite_store import (
    RetrievalDataVersionRole,
    RetrievalDataVersionState,
    SqliteRetrievalCatalog,
)


def run_real_bge_smoke(
    *,
    profile: DocumentRetrievalProfile | None = None,
) -> dict[str, object]:
    """运行固定本地资产的三方法索引、查询与真实 reranker 前向。

    该入口使用与 GUI 相同的 production profile/preflight，不接受请求期下载，也不把
    lexical fallback 包装成 Hybrid 成功。两个短块足以覆盖 Dense、learned sparse、
    BM25 与二阶段重排，同时避免把应用 Source 内容带入临时数据库。
    """

    contents = (
        "北桥项目本周提交检索系统初稿，需核验 BGE-M3 稀疏召回。",
        "南岸食堂今天更新了早餐菜单和咖啡供应时间。",
    )
    query = "北桥检索初稿"
    runtime = build_document_retrieval_runtime(
        profile or DocumentRetrievalProfile.production()
    )
    encoder = runtime.encoder
    reranker = runtime.reranker
    if reranker is None:
        raise RuntimeError("real_bge_smoke_requires_reranker")
    source_filter = SourceFilter.from_mapping(
        SourceType.DOCUMENT,
        {"doc_id": "real-bge-smoke-doc"},
    )
    with TemporaryDirectory(prefix="personagraph-real-bge-") as root:
        catalog = SqliteRetrievalCatalog(Path(root) / "documents.sqlite")
        catalog.initialize()
        catalog.create_data_version(
            version_id="real-bge-v1",
            fingerprint=encoder.fingerprint(),
            role=RetrievalDataVersionRole.ACTIVE,
            state=RetrievalDataVersionState.READY,
        )
        method_store = SqliteRetrievalMethodStore(catalog=catalog, encoder=encoder)
        stored_units = []
        indexed_methods = []
        for index, content in enumerate(contents, start=1):
            ref = SourceUnitRef(
                SourceType.DOCUMENT,
                f"document-v3:real-bge-smoke-doc:chunk-{index}",
                "r1",
                f"real-bge-smoke-hash-{index}",
            )
            stored_unit = catalog.upsert_pending_unit(
                RetrievalUnit(
                    ref=ref,
                    retrieval_data_version="real-bge-v1",
                    retrieval_status=RetrievalStatus.ACTIVE,
                    source_filter=source_filter,
                )
            )
            indexed = method_store.index(stored_unit, content)
            catalog.mark_unit_index_ready(stored_unit.unit_id)
            stored_units.append(stored_unit)
            indexed_methods.append(indexed.methods)
        dense = method_store.search_dense(
            query,
            query_index=0,
            source_filter=source_filter,
            limit=3,
        )
        bm25 = method_store.search_bm25(
            query,
            query_index=0,
            source_filter=source_filter,
            limit=3,
        )
        sparse = method_store.search_learned_sparse(
            query,
            query_index=0,
            source_filter=source_filter,
            limit=3,
        )
        if not dense or not sparse or not bm25:
            raise RuntimeError("real_bge_three_method_smoke_missed_expected_unit")
        reranker_scores = reranker.score(tuple((query, content) for content in contents))
        if len(reranker_scores) != len(contents):
            raise RuntimeError("real_bge_reranker_returned_unaligned_scores")
        if reranker_scores[0] <= reranker_scores[1]:
            raise RuntimeError("real_bge_reranker_missed_relevant_smoke_passage")
        required_method_names = {
            method.value for method in runtime.effective_profile.retrieval_methods
        }
        if any(
            {method.value for method in methods} != required_method_names
            for methods in indexed_methods
        ):
            raise RuntimeError("real_bge_indexed_method_set_incomplete")
        return {
            "requested_profile": runtime.requested_profile.mode.value,
            "effective_profile": runtime.effective_profile.mode.value,
            "profile_fingerprint": runtime.effective_profile.fingerprint(),
            "encoder_fingerprint": encoder.fingerprint(),
            "reranker_fingerprint": reranker.fingerprint(),
            "status": "complete",
            "learned_sparse_available": True,
            "degraded_route": None,
            "indexed_methods": sorted(required_method_names),
            "dense_hits": len(dense),
            "learned_sparse_hits": len(sparse),
            "bm25_hits": len(bm25),
            "reranker_pair_count": len(reranker_scores),
            "reranker_relevant_first": reranker_scores[0] > reranker_scores[1],
            "method_index_health": [
                [
                    {
                        "method": health.method.value,
                        "expected_state": health.expected_state,
                        "representation_present": health.representation_present,
                    }
                    for health in method_store.method_index_health(stored_unit.unit_id)
                ]
                for stored_unit in stored_units
            ],
            "capability": runtime.capability.diagnostic_snapshot(),
            "encoder": encoder.diagnostic_snapshot(),
            "reranker": reranker.diagnostic_snapshot(),
        }


if __name__ == "__main__":  # pragma: no cover - 显式运维测试框架
    import json

    print(json.dumps(run_real_bge_smoke(), ensure_ascii=False, indent=2))
