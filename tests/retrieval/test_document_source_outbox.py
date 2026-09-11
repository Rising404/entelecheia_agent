from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

import pytest

from personagraph.input_processing.documents import (
    ChunkSpan,
    DocumentChunk,
    DocumentLocator,
    ElementKind,
)
from personagraph.workspace.documents import application as docstore
from personagraph.retrieval.contracts import (
    QueryProposal,
    RetrievalBudget,
    RetrievalMethod,
    RetrievalRequest,
    SourceDependency,
    SourceFilter,
    SourceType,
    TrustedRetrievalBoundary,
)
from personagraph.retrieval.sources.coverage.document import DocumentIndexCoverageProbe
from personagraph.retrieval.indexing.methods import (
    BM25Retrieval,
    BgeM3EncodedText,
    DenseRetrieval,
    LearnedSparseRetrieval,
    LiteralBooleanFallback,
    SqliteRetrievalIndexWriter,
    SqliteRetrievalMethodStore,
)
from personagraph.retrieval.lifecycle.outbox import SqliteRetrievalOutbox
from personagraph.retrieval.policy import DefaultRetrievalPolicy
from personagraph.retrieval.query_guard import QueryGuard
from personagraph.retrieval.service import RetrievalService
from personagraph.retrieval.sources import document as source_adapters
from personagraph.retrieval.sources.document import MountedDocumentChunkSourceAdapter
from personagraph.retrieval.sqlite_store import (
    RetrievalDataVersionRole,
    RetrievalDataVersionState,
    SqliteRetrievalCatalog,
)
from personagraph.retrieval.lifecycle.sync import RetrievalOutboxConsumer, RetrievalSyncService
from personagraph.session import store as session_store
from tests.documents._authority import (
    ProjectDocumentAuthority,
    bound_project_document_authority,
)


@pytest.fixture(autouse=True)
def authority(tmp_path, monkeypatch, seed_session_ids):
    seed_session_ids("s1", "s2")
    # 本测试套件负责文档权威域行为。其合成挂载 ID 代表已受信任的活动会话；
    # 会话生命周期由来源适配器集成测试单独覆盖。
    monkeypatch.setattr(
        source_adapters.session_store,
        "get_session",
        lambda session_id: {"id": session_id, "status": "active"},
    )
    with bound_project_document_authority(
        tmp_path,
        project_root=tmp_path,
    ) as value:
        value.database.initialize()
        SqliteRetrievalCatalog(value.database.db_path).create_data_version(
            version_id="v1",
            fingerprint="fake-bge",
            role=RetrievalDataVersionRole.ACTIVE,
            state=RetrievalDataVersionState.READY,
        )
        yield value


def _vector(value: float) -> tuple[float, ...]:
    return (value,) + (0.0,) * 1023


@dataclass
class FakeEncoder:
    values: dict[str, BgeM3EncodedText]

    def encode(self, texts):
        return tuple(self.values[text] for text in texts)

    def token_ids(self, text: str):
        return self.values[text].token_ids

    def fingerprint(self) -> str:
        return "fake-bge"


@dataclass(frozen=True)
class NoMatchMethod:
    """让覆盖测试聚焦目录状态，而非排序行为。"""

    method: RetrievalMethod

    def search(self, query, *, query_index, source_filter, limit, retrieval_data_version_id=None):
        del query, query_index, source_filter, limit, retrieval_data_version_id
        return ()


def _elements(content: str) -> list[dict[str, str]]:
    return [{"content": content, "loc": "p.1"}]


def _typed_chunk(producer_chunk_id: str, content: str) -> DocumentChunk:
    locator = DocumentLocator(page=1, ordinal=0, section_path=("Methods",))
    return DocumentChunk(
        chunk_id=producer_chunk_id,
        text=content,
        span=ChunkSpan(start=locator, end=locator),
        section_path=("Methods",),
        element_ids=("el-1",),
        token_count=8,
        kind=ElementKind.PARAGRAPH,
    )


def _consume_document_events(*, authority_conn, sync: RetrievalSyncService, now: str):
    return RetrievalOutboxConsumer(
        outbox=SqliteRetrievalOutbox(),
        sync_service=sync,
    ).consume_due(
        authority_conn,
        worker_id="document-test-worker",
        now=now,
    )


def _ingest(
    authority: ProjectDocumentAuthority,
    path: str,
    title: str,
    content: str,
    *,
    session_id: str,
    **kwargs: object,
) -> dict[str, object]:
    candidate = Path(path)
    try:
        relative_path = candidate.resolve().relative_to(authority.root).as_posix()
    except ValueError:
        relative_path = path.removeprefix("/workspace/").lstrip("/")
    with authority.session(session_id):
        return authority.ingest(
            relative_path,
            title,
            "text/markdown",
            _elements(content),
            session_id=session_id,
            source_payload=content,
            **kwargs,
        )


def _session_snapshot_count(
    authority: ProjectDocumentAuthority,
    session_id: str = "s1",
) -> int:
    del authority
    with session_store.session_database_scope(session_id):
        session_store.init_db()
        with session_store._connect() as conn:
            return int(
                conn.execute(
                    "SELECT COUNT(*) FROM doc_retrieval_snapshots"
                ).fetchone()[0]
            )


def test_document_lifecycle_is_atomic_and_forms_a_real_retrieval_loop(
    authority,
    tmp_path,
    monkeypatch,
):
    content = "论文采用双路检索，并在重排阶段处理关键词与语义证据。"
    query = "论文的双路检索方法"
    source_path = tmp_path / "paper.md"
    source_path.write_text(content, encoding="utf-8")
    document = _ingest(
        authority,
        str(source_path),
        "paper",
        content,
        session_id="s1",
        retrieval_data_version="v1",
    )
    source_filter = SourceFilter.from_mapping(
        SourceType.DOCUMENT,
        {"doc_id": document["doc_id"], "session_id": "s1"},
    )
    adapter = MountedDocumentChunkSourceAdapter()
    source_unit = adapter.list_indexable_units_for_backfill(source_filter)[0]

    with authority.database.connect() as conn:
        rows = conn.execute("SELECT * FROM retrieval_update_outbox ORDER BY event_id").fetchall()
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(retrieval_update_outbox)")}
    assert len(rows) == 1
    assert rows[0]["kind"] == "upsert"
    assert rows[0]["source_unit_id"] == source_unit.ref.source_unit_id
    assert rows[0]["source_revision"] == source_unit.ref.source_revision
    assert rows[0]["indexed_content_hash"] == source_unit.ref.indexed_content_hash
    assert "content" not in columns

    encoder = FakeEncoder(
        {
            source_unit.content: BgeM3EncodedText(_vector(1.0), {1: 1.0, 2: 0.7}, (1, 2)),
            query: BgeM3EncodedText(_vector(1.0), {1: 1.0, 2: 0.7}, (1, 2)),
        }
    )
    catalog = SqliteRetrievalCatalog(authority.database.db_path)
    method_store = SqliteRetrievalMethodStore(catalog=catalog, encoder=encoder)
    sync = RetrievalSyncService(
        catalog=catalog,
        source_readers={SourceType.DOCUMENT: adapter},
        index_writer=SqliteRetrievalIndexWriter(method_store),
    )
    with authority.database.connect() as conn:
        results = _consume_document_events(
            authority_conn=conn,
            sync=sync,
            now="2099-01-01T00:00:00+00:00",
        )
    assert [result.status.value for result in results] == ["applied"]

    service = RetrievalService(
        policy=DefaultRetrievalPolicy(),
        query_guard=QueryGuard(),
        sources={SourceType.DOCUMENT: adapter},
        methods={
            DenseRetrieval.method: DenseRetrieval(method_store),
            LearnedSparseRetrieval.method: LearnedSparseRetrieval(method_store),
            BM25Retrieval.method: BM25Retrieval(method_store),
            LiteralBooleanFallback.method: LiteralBooleanFallback(method_store),
        },
        data_version_provider=catalog,
        token_estimator=lambda text: len(text),
    )
    context = service.retrieve_context(
        RetrievalRequest(
            request_id="document-outbox-e2e",
            model_call_purpose="TEST",
            query_proposal=QueryProposal((query,)),
            boundary=TrustedRetrievalBoundary(
                {SourceType.DOCUMENT: source_filter},
                {SourceType.DOCUMENT: SourceDependency.OPTIONAL},
            ),
        ),
        RetrievalBudget(candidate_limit_per_source=3, context_token_limit=500, max_items=3),
    )
    assert [item.ref for item in context.items] == [source_unit.ref]
    assert [item.content for item in context.items] == [content]
    source_snapshot_id = context.source_outcomes[SourceType.DOCUMENT].source_snapshot_id
    assert source_snapshot_id
    snapshot = docstore.get_retrieval_snapshot(source_snapshot_id, "s1")
    assert snapshot is not None

    # 物理源在文档权威库之外变化时，统一 RAG 必须在搜索旧派生索引前
    # 关闭该 Source；它不能把这一事实降格成 no_match。
    source_path.write_text("论文已改为仅使用关键词检索。", encoding="utf-8")
    changed_context = service.retrieve_context(
        RetrievalRequest(
            request_id="document-physical-source-changed",
            model_call_purpose="TEST",
            query_proposal=QueryProposal((query,)),
            boundary=TrustedRetrievalBoundary(
                {SourceType.DOCUMENT: source_filter},
                {SourceType.DOCUMENT: SourceDependency.OPTIONAL},
            ),
        ),
        RetrievalBudget(candidate_limit_per_source=3, context_token_limit=500, max_items=3),
    )
    changed_outcome = changed_context.source_outcomes[SourceType.DOCUMENT]
    assert changed_context.items == ()
    assert changed_context.status.value == "partial"
    assert changed_outcome.availability.value == "blocked"
    assert changed_outcome.reason_code == "document_source_changed"
    source_path.write_text(content, encoding="utf-8")

    # 会话软删除会立即关闭已挂载的文档来源，即使其物理目录墓碑
    # 尚未由后续的跨数据库级联协调器处理。
    monkeypatch.setattr(
        source_adapters.session_store,
        "get_session",
        lambda session_id: {"id": session_id, "status": "trashed"},
    )
    blocked_context = service.retrieve_context(
        RetrievalRequest(
            request_id="document-session-trashed",
            model_call_purpose="TEST",
            query_proposal=QueryProposal((query,)),
            boundary=TrustedRetrievalBoundary(
                {SourceType.DOCUMENT: source_filter},
                {SourceType.DOCUMENT: SourceDependency.OPTIONAL},
            ),
        ),
        RetrievalBudget(candidate_limit_per_source=3, context_token_limit=500, max_items=3),
    )
    assert blocked_context.items == ()
    assert blocked_context.source_outcomes[SourceType.DOCUMENT].availability.value == "blocked"
    monkeypatch.setattr(
        source_adapters.session_store,
        "get_session",
        lambda session_id: {"id": session_id, "status": "active"},
    )

    with authority.session("s1"):
        assert docstore.detach(document["doc_id"], "s1")
    with authority.database.connect() as conn:
        results = _consume_document_events(
            authority_conn=conn,
            sync=sync,
            now="2099-01-01T00:01:00+00:00",
        )
    assert results == ()
    assert [unit.unit.ref for unit in catalog.active_units("v1")] == [source_unit.ref]

    # 重新挂载会发出 RESTORE，而不是重放原始 UPSERT 事件。
    remounted = _ingest(
        authority,
        str(source_path),
        "paper",
        content,
        session_id="s1",
        retrieval_data_version="v1",
    )
    assert remounted["doc_id"] == document["doc_id"] and remounted["deduped"]
    with authority.database.connect() as conn:
        results = _consume_document_events(
            authority_conn=conn,
            sync=sync,
            now="2099-01-01T00:02:00+00:00",
        )
    assert [result.status.value for result in results] == ["applied"]
    assert [unit.unit.ref for unit in catalog.active_units("v1")] == [source_unit.ref]

    with authority.session("s1"):
        assert docstore.remove(
            document["doc_id"],
            retrieval_data_version="v1",
            document_index_port=authority.document_index_port,
        )
    with authority.database.connect() as conn:
        results = _consume_document_events(
            authority_conn=conn,
            sync=sync,
            now="2099-01-01T00:03:00+00:00",
        )
    assert [result.status.value for result in results] == ["applied"]
    assert catalog.get_unit(source_unit.ref, "v1") is None


def test_document_index_lag_is_partial_until_every_current_chunk_is_searchable(
    authority,
    tmp_path,
):
    """新文档不得把缺失派生行解释成完整的无匹配结果。"""

    content = "新文档当前版本的材料。"
    query = "不会命中的问题"
    source_path = tmp_path / "lagged-paper.md"
    source_path.write_text(content, encoding="utf-8")
    document = _ingest(
        authority,
        str(source_path),
        "lagged-paper",
        content,
        session_id="s1",
        retrieval_data_version="v1",
        processor_fingerprint="test-reader@1",
        document_chunks=(_typed_chunk("ch-lagged-paper-v1", content),),
        chunker_fingerprint="test-chunker@1",
        processing_status="complete",
    )
    source_filter = SourceFilter.from_mapping(
        SourceType.DOCUMENT,
        {"doc_id": document["doc_id"], "session_id": "s1"},
    )
    adapter = MountedDocumentChunkSourceAdapter()
    source_unit = adapter.list_indexable_units_for_backfill(source_filter)[0]
    encoder = FakeEncoder(
        {
            source_unit.content: BgeM3EncodedText(_vector(1.0), {1: 1.0}, (1,)),
            query: BgeM3EncodedText(_vector(0.0), {2: 1.0}, (2,)),
        }
    )
    catalog = SqliteRetrievalCatalog(authority.database.db_path)
    method_store = SqliteRetrievalMethodStore(catalog=catalog, encoder=encoder)
    sync = RetrievalSyncService(
        catalog=catalog,
        source_readers={SourceType.DOCUMENT: adapter},
        index_writer=SqliteRetrievalIndexWriter(method_store),
    )
    service = RetrievalService(
        policy=DefaultRetrievalPolicy(),
        query_guard=QueryGuard(),
        sources={SourceType.DOCUMENT: adapter},
        methods={
            method: NoMatchMethod(method)
            for method in (RetrievalMethod.DENSE, RetrievalMethod.LEARNED_SPARSE)
        },
        source_coverage_probes={
            SourceType.DOCUMENT: DocumentIndexCoverageProbe(
                catalog=catalog,
                binding_reader=adapter,
            )
        },
        data_version_provider=catalog,
        token_estimator=lambda text: len(text),
    )

    def retrieve(request_id: str):
        return service.retrieve_context(
            RetrievalRequest(
                request_id=request_id,
                model_call_purpose="TEST",
                query_proposal=QueryProposal((query,)),
                boundary=TrustedRetrievalBoundary(
                    {SourceType.DOCUMENT: source_filter},
                    {SourceType.DOCUMENT: SourceDependency.OPTIONAL},
                ),
            ),
            RetrievalBudget(candidate_limit_per_source=3, context_token_limit=500, max_items=3),
        )

    before_sync = retrieve("document-index-lag-before-sync")
    before_outcome = before_sync.source_outcomes[SourceType.DOCUMENT]
    assert before_sync.items == ()
    assert before_sync.status.value == "partial"
    assert before_outcome.availability.value == "ready"
    assert before_outcome.retrieval.value == "partial"
    assert before_outcome.reason_code == "document_index_coverage_incomplete"
    assert [
        limitation.reason_code
        for limitation in before_sync.coverage_limitations
        if limitation.source_type is SourceType.DOCUMENT
    ] == ["document_index_coverage_incomplete"]

    with authority.database.connect() as conn:
        results = _consume_document_events(
            authority_conn=conn,
            sync=sync,
            now="2099-01-01T00:00:00+00:00",
        )
    assert [result.status.value for result in results] == ["applied"]

    after_sync = retrieve("document-index-lag-after-sync")
    after_outcome = after_sync.source_outcomes[SourceType.DOCUMENT]
    assert after_sync.items == ()
    assert after_sync.status.value == "complete"
    assert after_outcome.availability.value == "ready"
    assert after_outcome.retrieval.value == "no_match"
    assert not [
        limitation
        for limitation in after_sync.coverage_limitations
        if limitation.source_type is SourceType.DOCUMENT
    ]

    # 来源重新导入会先推进权威版本，随后清除/UPSERT 事件才到达派生目录。因此，
    # 旧的就绪绑定不能让新快照看起来像完整的无匹配结果。
    revised_content = "重导入后的新版本材料。"
    source_path.write_text(revised_content, encoding="utf-8")
    reimported = _ingest(
        authority,
        str(source_path),
        "lagged-paper-revised",
        revised_content,
        session_id="s1",
        retrieval_data_version="v1",
        processor_fingerprint="test-reader@1",
        document_chunks=(_typed_chunk("ch-lagged-paper-v2", revised_content),),
        chunker_fingerprint="test-chunker@1",
        processing_status="complete",
    )
    assert reimported["doc_id"] == document["doc_id"]
    current_revision = docstore.list_current_document_chunks(document["doc_id"], session_id="s1")[0][
        "source_version_id"
    ]
    assert catalog.active_units("v1")[0].unit.ref.source_revision != current_revision

    after_reimport_before_sync = retrieve("document-index-lag-after-reimport")
    revised_outcome = after_reimport_before_sync.source_outcomes[SourceType.DOCUMENT]
    assert after_reimport_before_sync.items == ()
    assert after_reimport_before_sync.status.value == "partial"
    assert revised_outcome.retrieval.value == "partial"
    assert revised_outcome.reason_code == "document_index_coverage_incomplete"


def test_document_index_binding_snapshot_is_content_free_and_does_not_write_authority(
    authority,
    tmp_path,
):
    content = "仅用于核验当前 chunk 指针。"
    source_path = tmp_path / "metadata-only.md"
    source_path.write_text(content, encoding="utf-8")
    document = _ingest(
        authority,
        str(source_path),
        "metadata-only",
        content,
        session_id="s1",
    )
    source_filter = SourceFilter.from_mapping(
        SourceType.DOCUMENT,
        {"doc_id": document["doc_id"], "session_id": "s1"},
    )
    adapter = MountedDocumentChunkSourceAdapter()
    access = adapter.open_retrieval_access(source_filter)
    assert access.availability.value == "ready"

    snapshots_before = _session_snapshot_count(authority)
    bindings = adapter.get_current_index_binding_snapshot(access, maximum_bindings=5_000)
    snapshots_after = _session_snapshot_count(authority)

    current_chunk = docstore.list_current_document_chunks(document["doc_id"], session_id="s1")[0]
    assert snapshots_after == snapshots_before
    assert [
        (
            binding.source_unit_id,
            binding.source_revision,
            binding.indexed_content_hash,
        )
        for binding in bindings.bindings
    ] == [
        (
            f"s1:{current_chunk['id']}",
            current_chunk["source_version_id"],
            hashlib.sha256(content.strip().encode("utf-8")).hexdigest(),
        )
    ]


def test_document_index_binding_snapshot_fails_safe_when_pointer_enumeration_exceeds_cap(
    authority,
    tmp_path,
):
    source_path = tmp_path / "many-chunks.md"
    source_path.write_text("two chunks", encoding="utf-8")
    with authority.session("s1"):
        document = authority.ingest(
            "many-chunks.md",
            "many-chunks",
            "text/markdown",
            [
                {"content": "甲" * 1_001, "loc": "p.1"},
                {"content": "乙" * 1_001, "loc": "p.2"},
            ],
            session_id="s1",
            source_payload="two chunks",
        )
    source_filter = SourceFilter.from_mapping(
        SourceType.DOCUMENT,
        {"doc_id": document["doc_id"], "session_id": "s1"},
    )
    access = MountedDocumentChunkSourceAdapter().open_retrieval_access(source_filter)

    snapshot = MountedDocumentChunkSourceAdapter().get_current_index_binding_snapshot(
        access,
        maximum_bindings=1,
    )

    assert snapshot.source_snapshot_is_current
    assert not snapshot.binding_enumeration_complete
    assert snapshot.bindings == ()


def test_document_access_revalidation_detects_new_session_mount_without_a_second_snapshot_write(
    authority,
    tmp_path,
):
    first_path = tmp_path / "first.md"
    second_path = tmp_path / "second.md"
    first_path.write_text("first", encoding="utf-8")
    second_path.write_text("second", encoding="utf-8")
    first = _ingest(
        authority,
        str(first_path),
        "first",
        "第一份文档",
        session_id="s1",
    )
    source_filter = SourceFilter.from_mapping(SourceType.DOCUMENT, {"session_id": "s1"})
    adapter = MountedDocumentChunkSourceAdapter()
    initial_access = adapter.open_retrieval_access(source_filter)
    assert initial_access.availability.value == "ready"
    assert set(initial_access.source_revision_map) == {first["doc_id"]}
    snapshots_before = _session_snapshot_count(authority)

    second = _ingest(
        authority,
        str(second_path),
        "second",
        "第二份文档",
        session_id="s1",
    )
    final_access = adapter.revalidate_retrieval_access(initial_access)
    snapshots_after = _session_snapshot_count(authority)

    assert final_access.availability.value == "ready"
    assert set(final_access.source_revision_map) == {first["doc_id"], second["doc_id"]}
    assert final_access.source_snapshot_id == initial_access.source_snapshot_id
    assert snapshots_after == snapshots_before


def test_document_access_revalidation_blocks_an_external_file_change_without_writing_snapshot(
    authority,
    tmp_path,
):
    source_path = tmp_path / "changed-after-search.md"
    source_path.write_text("原始文件", encoding="utf-8")
    document = _ingest(
        authority,
        str(source_path),
        "changed-after-search",
        "原始文件",
        session_id="s1",
    )
    source_filter = SourceFilter.from_mapping(
        SourceType.DOCUMENT,
        {"doc_id": document["doc_id"], "session_id": "s1"},
    )
    adapter = MountedDocumentChunkSourceAdapter()
    initial_access = adapter.open_retrieval_access(source_filter)
    snapshots_before = _session_snapshot_count(authority)

    source_path.write_text("外部更新但尚未重导入", encoding="utf-8")
    final_access = adapter.revalidate_retrieval_access(initial_access)
    snapshots_after = _session_snapshot_count(authority)

    assert final_access.availability.value == "blocked"
    assert final_access.reason_code == "document_source_changed"
    assert snapshots_after == snapshots_before


def test_document_access_is_normally_empty_when_the_session_starts_without_mounts():
    source_filter = SourceFilter.from_mapping(SourceType.DOCUMENT, {"session_id": "s1"})

    access = MountedDocumentChunkSourceAdapter().open_retrieval_access(source_filter)

    assert access.availability.value == "empty"


def test_document_access_revalidation_blocks_when_all_session_documents_are_unmounted(
    authority,
    tmp_path,
):
    source_path = tmp_path / "unmounted-after-search.md"
    source_path.write_text("原始文件", encoding="utf-8")
    document = _ingest(
        authority,
        str(source_path),
        "unmounted-after-search",
        "原始文件",
        session_id="s1",
    )
    source_filter = SourceFilter.from_mapping(SourceType.DOCUMENT, {"session_id": "s1"})
    adapter = MountedDocumentChunkSourceAdapter()
    initial_access = adapter.open_retrieval_access(source_filter)
    snapshots_before = _session_snapshot_count(authority)

    with authority.session("s1"):
        assert docstore.detach(document["doc_id"], "s1")
    final_access = adapter.revalidate_retrieval_access(initial_access)
    snapshots_after = _session_snapshot_count(authority)

    assert final_access.availability.value == "blocked"
    assert final_access.reason_code == "document_source_changed_during_retrieval"
    assert snapshots_after == snapshots_before


def test_document_reimport_purges_the_old_project_unit_before_indexing_the_new_unit(
    authority,
):
    first = _ingest(
        authority,
        "/workspace/paper.md",
        "paper",
        "旧版本：仅使用关键词检索。",
        session_id="s1",
        retrieval_data_version="v1",
        processor_fingerprint="reader@1",
        document_chunks=(
            _typed_chunk("ch-paper-v1", "旧版本：仅使用关键词检索。"),
        ),
        chunker_fingerprint="chunker@1",
        processing_status="complete",
    )
    shared = _ingest(
        authority,
        "/workspace/paper.md",
        "paper",
        "旧版本：仅使用关键词检索。",
        session_id="s2",
        retrieval_data_version="v1",
        processor_fingerprint="reader@1",
        document_chunks=(
            _typed_chunk("ch-paper-v1", "旧版本：仅使用关键词检索。"),
        ),
        chunker_fingerprint="chunker@1",
        processing_status="complete",
    )
    assert shared["doc_id"] == first["doc_id"]
    old_chunks = docstore.list_current_document_chunks(first["doc_id"])

    replaced = _ingest(
        authority,
        "/workspace/paper.md",
        "paper revised",
        "新版本：使用稠密、稀疏与 BM25 混合检索。",
        session_id="s1",
        retrieval_data_version="v1",
        processor_fingerprint="reader@1",
        document_chunks=(
            _typed_chunk(
                "ch-paper-v2",
                "新版本：使用稠密、稀疏与 BM25 混合检索。",
            ),
        ),
        chunker_fingerprint="chunker@1",
        processing_status="complete",
    )
    assert replaced["doc_id"] == first["doc_id"] and replaced["reindexed"]
    new_chunks = docstore.list_current_document_chunks(first["doc_id"])

    with authority.database.connect() as conn:
        events = conn.execute(
            "SELECT kind, source_unit_id, source_revision FROM retrieval_update_outbox ORDER BY event_id"
        ).fetchall()
    purges = [row for row in events if row["kind"] == "purge"]
    new_upserts = [row for row in events if row["kind"] == "upsert" and row["source_revision"] == new_chunks[0]["source_version_id"]]
    assert {row["source_unit_id"] for row in purges} == {
        f"document-v3:{first['doc_id']}:{chunk['producer_chunk_id']}"
        for chunk in old_chunks
    }
    assert {row["source_unit_id"] for row in new_upserts} == {
        f"document-v3:{first['doc_id']}:{chunk['producer_chunk_id']}"
        for chunk in new_chunks
    }


def test_legacy_generation_reindexes_to_doc_scoped_typed_outbox_identity(authority):
    content = "论文采用可追溯的证据综合。"
    first = _ingest(
        authority,
        "/workspace/paper.md",
        "paper",
        content,
        session_id="s1",
        retrieval_data_version="v1",
        processor_fingerprint="reader@1",
        chunks=[{"seq": 0, "loc": "p1", "content": content}],
        chunker_fingerprint="chunker@1",
    )
    legacy_chunk = docstore.list_current_document_chunks(first["doc_id"], session_id="s1")[0]

    upgraded = _ingest(
        authority,
        "/workspace/paper.md",
        "paper",
        content,
        session_id="s1",
        retrieval_data_version="v1",
        processor_fingerprint="reader@1",
        document_chunks=(_typed_chunk("ch_evidence", content),),
        chunker_fingerprint="chunker@1",
        processing_status="complete",
        processing_diagnostics=(),
    )

    assert upgraded["doc_id"] == first["doc_id"]
    assert upgraded["reindexed"] is True
    typed_chunk = docstore.list_current_document_chunks(first["doc_id"], session_id="s1")[0]
    assert typed_chunk["producer_chunk_id"] == "ch_evidence"
    legacy_identity = f"s1:{legacy_chunk['id']}"
    typed_identity = f"document-v3:{first['doc_id']}:ch_evidence"
    with authority.database.connect() as conn:
        events = conn.execute(
            "SELECT kind, source_unit_id FROM retrieval_update_outbox "
            "ORDER BY authority_sequence"
        ).fetchall()
    assert [(row["kind"], row["source_unit_id"]) for row in events] == [
        ("upsert", legacy_identity),
        ("purge", legacy_identity),
        ("upsert", typed_identity),
    ]


def test_document_write_rolls_back_when_its_outbox_append_fails(authority, monkeypatch):
    from personagraph.retrieval.lifecycle import outbox

    def fail_enqueue(self, conn, event):
        del self, conn, event
        raise RuntimeError("simulated_outbox_failure")

    monkeypatch.setattr(outbox.SqliteRetrievalOutbox, "enqueue", fail_enqueue)
    with pytest.raises(RuntimeError, match="simulated_outbox_failure"):
        _ingest(
            authority,
            "/workspace/paper.md",
            "paper",
            "失败时不应留下文档。",
            session_id="s1",
            retrieval_data_version="v1",
        )
    assert docstore.list_documents() == []
