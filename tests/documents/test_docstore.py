"""G1 DocSet 核心：入库分块/去重/挂载作用域/检索引用/管理操作。"""
import sqlite3

import pytest

from personagraph.input_processing.documents import (
    ChunkSpan,
    DocumentChunk,
    DocumentLocator,
    ElementKind,
)
from personagraph.workspace.documents import application as ds
from personagraph.retrieval.sqlite_store import (
    RetrievalDataVersionRole,
    RetrievalDataVersionState,
    SqliteRetrievalCatalog,
)
from tests.documents._authority import (
    ProjectDocumentAuthority,
    bound_project_document_authority,
)


@pytest.fixture
def authority(tmp_path, seed_session_ids):
    seed_session_ids("sA", "sB")
    with bound_project_document_authority(tmp_path) as value:
        yield value


def _els(n=5, page=True):
    return [{"content": f"第{i}节：关于华为竞赛评审的说明，" + "内容" * 200,
             "loc": f"p{i}" if page else str(i)} for i in range(1, n + 1)]


def _typed_chunk(ordinal: int) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=f"producer-{ordinal}",
        text=f"Evidence statement {ordinal}.",
        span=ChunkSpan(
            start=DocumentLocator(page=ordinal + 1, ordinal=0),
            end=DocumentLocator(page=ordinal + 1, ordinal=0),
        ),
        section_path=(f"Section {ordinal}",),
        element_ids=(f"element-{ordinal}",),
        token_count=3,
        kind=ElementKind.PARAGRAPH,
        source_pages=(ordinal + 1,),
    )


def _ingest_typed_document(
    authority: ProjectDocumentAuthority,
    chunk_count: int = 5,
) -> dict[str, object]:
    chunks = tuple(_typed_chunk(index) for index in range(chunk_count))
    with authority.session("sA"):
        return authority.ingest(
            "windowed.pdf",
            "windowed",
            "pdf",
            [{"content": item.text, "loc": item.loc} for item in chunks],
            session_id="sA",
            processor_fingerprint="reader@test",
            document_chunks=chunks,
            chunker_fingerprint="chunker@test",
            processing_status="complete",
        )


def _ingest(
    authority: ProjectDocumentAuthority,
    relative_path: str,
    title: str,
    elements: list[dict[str, object]],
    *,
    session_id: str,
) -> dict[str, object]:
    with authority.session(session_id):
        return authority.ingest(
            relative_path,
            title,
            "pdf",
            elements,
            session_id=session_id,
        )


def test_mounted_resource_snapshot_can_read_an_exact_late_chunk_window(authority) -> None:
    chunks = tuple(_typed_chunk(index) for index in range(5))
    stored = authority.ingest(
        "long.pdf",
        "long",
        "pdf",
        [{"content": item.text, "loc": item.loc} for item in chunks],
        session_id="sA",
        processor_fingerprint="reader@test",
        document_chunks=chunks,
        chunker_fingerprint="chunker@test",
        processing_status="complete",
    )

    snapshot = ds.get_mounted_current_document_resource_snapshot(
        stored["doc_id"],
        session_id="sA",
        start_sequence=2,
        maximum_chunks=2,
    )

    assert snapshot is not None
    assert snapshot.total_chunk_count == 5
    assert snapshot.physical_page_count is None
    assert snapshot.page_inventory_status == "unavailable"
    assert [item.sequence for item in snapshot.chunks] == [2, 3]
    assert [item.content for item in snapshot.chunks] == [
        "Evidence statement 2.",
        "Evidence statement 3.",
    ]
    assert snapshot.truncated is True


def test_mounted_resource_snapshot_windows_cover_each_chunk_exactly_once(authority) -> None:
    stored = _ingest_typed_document(authority)

    snapshots = tuple(
        ds.get_mounted_current_document_resource_snapshot(
            str(stored["doc_id"]),
            session_id="sA",
            start_sequence=start_sequence,
            maximum_chunks=2,
        )
        for start_sequence in (0, 2, 4)
    )

    assert all(snapshot is not None for snapshot in snapshots)
    windows = [
        [chunk.sequence for chunk in snapshot.chunks]
        for snapshot in snapshots
        if snapshot is not None
    ]
    assert windows == [[0, 1], [2, 3], [4]]
    combined = [sequence for window in windows for sequence in window]
    assert combined == list(range(5))
    assert len(combined) == len(set(combined))
    assert snapshots[-1] is not None
    assert snapshots[-1].total_chunk_count == 5
    assert snapshots[-1].truncated is True


def test_real_sqlite_long_document_windows_cover_310_chunks_exactly_once(authority) -> None:
    stored = _ingest_typed_document(authority, chunk_count=310)
    cursor = 0
    window_starts: list[int] = []
    observed_sequences: list[int] = []

    while True:
        window_starts.append(cursor)
        snapshot = ds.get_mounted_current_document_resource_snapshot(
            str(stored["doc_id"]),
            session_id="sA",
            start_sequence=cursor,
            maximum_chunks=64,
        )

        assert snapshot is not None
        assert snapshot.total_chunk_count == 310
        assert snapshot.chunks
        observed_sequences.extend(chunk.sequence for chunk in snapshot.chunks)
        cursor += len(snapshot.chunks)
        if cursor == snapshot.total_chunk_count:
            break

    assert window_starts == [0, 64, 128, 192, 256]
    assert observed_sequences == list(range(310))
    assert len(observed_sequences) == len(set(observed_sequences))


def test_mounted_resource_snapshot_start_at_total_returns_empty_window(authority) -> None:
    stored = _ingest_typed_document(authority)

    snapshot = ds.get_mounted_current_document_resource_snapshot(
        str(stored["doc_id"]),
        session_id="sA",
        start_sequence=5,
        maximum_chunks=2,
    )

    assert snapshot is not None
    assert snapshot.total_chunk_count == 5
    assert snapshot.chunks == ()
    assert snapshot.truncated is True


@pytest.mark.parametrize(
    "invalid_start_sequence",
    (-1, True, False),
    ids=("negative", "true-is-not-one", "false-is-not-zero"),
)
def test_mounted_resource_snapshot_rejects_invalid_start_sequence(
    authority,
    invalid_start_sequence: object,
) -> None:
    stored = _ingest_typed_document(authority)

    with pytest.raises(ValueError, match="start_sequence"):
        ds.get_mounted_current_document_resource_snapshot(
            str(stored["doc_id"]),
            session_id="sA",
            start_sequence=invalid_start_sequence,  # type: ignore[arg-type]
            maximum_chunks=2,
        )


def test_mounted_resource_snapshot_default_start_preserves_prefix_behavior(authority) -> None:
    stored = _ingest_typed_document(authority)

    default_window = ds.get_mounted_current_document_resource_snapshot(
        str(stored["doc_id"]),
        session_id="sA",
        maximum_chunks=2,
    )
    explicit_zero_window = ds.get_mounted_current_document_resource_snapshot(
        str(stored["doc_id"]),
        session_id="sA",
        start_sequence=0,
        maximum_chunks=2,
    )
    complete_window = ds.get_mounted_current_document_resource_snapshot(
        str(stored["doc_id"]),
        session_id="sA",
        maximum_chunks=5,
    )

    assert default_window == explicit_zero_window
    assert default_window is not None
    assert [chunk.sequence for chunk in default_window.chunks] == [0, 1]
    assert default_window.truncated is True
    assert complete_window is not None
    assert [chunk.sequence for chunk in complete_window.chunks] == list(range(5))
    assert complete_window.truncated is False


def test_mounted_resource_snapshot_rejects_duplicate_sequence_hidden_before_late_window(authority) -> None:
    stored = _ingest_typed_document(authority)
    with sqlite3.connect(authority.database.db_path) as conn:
    # 独立于模式守卫验证读取侧不变量，模拟打开 V20 之前或被外部损坏的存储。
        conn.execute("DROP INDEX idx_doc_chunks_sequence_generation")
        row = conn.execute(
            "SELECT id FROM doc_chunks WHERE doc_id=? AND seq=2",
            (str(stored["doc_id"]),),
        ).fetchone()
        assert row is not None
        conn.execute("UPDATE doc_chunks SET seq=1 WHERE id=?", (str(row[0]),))

    with pytest.raises(ValueError, match="chunk extent"):
        ds.get_mounted_current_document_resource_snapshot(
            str(stored["doc_id"]),
            session_id="sA",
            start_sequence=3,
            maximum_chunks=2,
        )


def test_ingest_chunks_and_mounts(authority):
    r = _ingest(authority, "a/1.pdf", "评审规则", _els(), session_id="sA")
    assert r["n_chunks"] >= 2 and not r["deduped"]
    assert [d["id"] for d in ds.mounted_docs("sA")] == [r["doc_id"]]
    assert ds.mounted_docs("sB") == []                       # 挂载按会话隔离


def test_same_name_different_content_no_conflict(authority):
    a = _ingest(authority, "a/1.pdf", "1", _els(3), session_id="sA")
    b = _ingest(
        authority,
        "b/1.pdf",
        "1",
        [{"content": "完全不同的内容" * 100, "loc": "p1"}],
        session_id="sB",
    )
    assert a["doc_id"] != b["doc_id"]                        # 同名不同内容互不冲突


def test_same_registered_file_dedup_and_cross_session_mount(authority):
    a = _ingest(authority, "a/1.pdf", "1", _els(3), session_id="sA")
    b = _ingest(authority, "a/1.pdf", "1", _els(3), session_id="sB")
    assert b["deduped"] and b["doc_id"] == a["doc_id"]
    assert ds.mounted_docs("sB")[0]["id"] == a["doc_id"]     # 两会话各自可见


def test_doc_read_is_session_scoped_and_keeps_locators(authority):
    r = _ingest(authority, "a/rules.pdf", "评审规则", _els(), session_id="sA")
    chunks = ds.doc_read(r["doc_id"], n=10, session_id="sA")
    assert chunks and all(item["doc_id"] == r["doc_id"] for item in chunks)
    assert chunks[0]["loc"].startswith("p")                 # 带页码引用
    assert ds.doc_read(r["doc_id"], n=10, session_id="sB") == []


def test_explicit_document_id_and_read_cannot_bypass_session_mount_scope(authority):
    r = _ingest(authority, "a/rules.pdf", "评审规则", _els(), session_id="sA")

    assert ds.doc_read(r["doc_id"], session_id="sB") == []
    assert ds.doc_read(r["doc_id"], session_id="sA")
    assert ds.is_mounted(r["doc_id"], "sA") is True
    assert ds.is_mounted(r["doc_id"], "sB") is False


def test_detach_edit_remove(authority):
    r = _ingest(authority, "a/1.pdf", "1", _els(3), session_id="sA")
    assert ds.edit(r["doc_id"], summary="竞赛评分维度说明")
    assert ds.detach(r["doc_id"], "sA") and ds.mounted_docs("sA") == []
    with authority.session("sA"):
        assert ds.remove(r["doc_id"])
    assert ds.doc_read(r["doc_id"]) == []                    # chunk 一并清除


def test_document_removal_resolves_generation_inside_project_write_fence(authority):
    result = _ingest(
        authority,
        "a/fenced.pdf",
        "fenced",
        _els(2),
        session_id="sA",
    )
    catalog = SqliteRetrievalCatalog(authority.database.db_path)
    catalog.initialize()
    catalog.create_data_version(
        version_id="active-documents-v1",
        fingerprint="document-removal-test@1",
        role=RetrievalDataVersionRole.ACTIVE,
        state=RetrievalDataVersionState.READY,
    )
    provider_calls = 0

    def active_generation(conn: sqlite3.Connection, candidate: str | None) -> str:
        nonlocal provider_calls
        provider_calls += 1
        assert conn.in_transaction
        assert candidate is None
        observer = sqlite3.connect(authority.database.db_path, timeout=0)
        try:
            with pytest.raises(sqlite3.OperationalError, match="locked"):
                observer.execute("BEGIN IMMEDIATE")
        finally:
            observer.close()
        return "active-documents-v1"

    with authority.session("sA"):
        changed = ds.remove(
            result["doc_id"],
            retrieval_data_version_provider=active_generation,
            document_index_port=authority.document_index_port,
        )

    assert changed is True
    assert provider_calls == 1
    with authority.database.connect() as conn:
        versions = {
            str(row["data_version_id"])
            for row in conn.execute(
                "SELECT data_version_id FROM retrieval_update_outbox"
            ).fetchall()
        }
    assert versions == {"active-documents-v1"}
