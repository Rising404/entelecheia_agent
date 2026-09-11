"""分块不仅要记录文件版本，还必须记录由哪个代码版本生成。

``source_version_id`` 能回答“它来自哪个文件版本”，却无法回答“哪个解析器和
分块器版本生成了该文本”。若没有独立指纹，Docling 或读取器升级将与未发生
变化无法区分，唯一安全的应对方式就只能是重新分块所有已摄取文档。
"""

from __future__ import annotations

import os
import sqlite3
from dataclasses import replace

import pytest

from personagraph.input_processing.documents import ProcessorFingerprint
from personagraph.input_processing.documents.preparation import service as document_service
from personagraph.retrieval.profile import (
    DocumentRetrievalProfile,
    durable_chunking_profile,
    production_document_chunking_profile,
)
from personagraph.retrieval.indexing.encoder import DeterministicLexicalEncoder
from personagraph.workspace.storage.context import current
from personagraph.workspace.documents import application as docstore
from personagraph.session import store as session_store
from tests.documents._authority import (
    ProjectDocumentAuthority,
    ingest_registered_document_fixture,
)


@pytest.fixture
def isolated_docstore(tmp_path, partitioned_project_state):
    """Exercise provenance against the real Session -> Project DB router."""
    durable_chunking_profile.cache_clear()
    project_root = tmp_path / "project"
    project_root.mkdir()
    session_id = session_store.create_session(
        "Entelecheia",
        title="chunk provenance",
        working_dir=str(project_root),
    )
    with session_store.session_database_scope(session_id):
        database = current()
        assert database is not None
        yield ProjectDocumentAuthority(database)
    durable_chunking_profile.cache_clear()


def _chunks(
    authority: ProjectDocumentAuthority,
    where: str = "",
    params: tuple = (),
) -> list[dict]:
    conn = sqlite3.connect(authority.database.db_path)
    conn.row_factory = sqlite3.Row
    with conn:
        rows = conn.execute(
            "SELECT seq, loc, source_version_id, processor_fingerprint, chunker_fingerprint "
            f"FROM doc_chunks {where} ORDER BY seq",
            params,
        ).fetchall()
    return [dict(row) for row in rows]


def test_ingested_chunks_record_both_producer_identities(isolated_docstore):
    document = isolated_docstore.root / "plan.md"
    document.write_text("# 方案\n\n第一段。\n\n## 风险\n\n第二段。\n", encoding="utf-8")

    result = ingest_registered_document_fixture(document)

    assert result["ok"] is True
    rows = _chunks(isolated_docstore)
    assert rows, "ingest must produce at least one chunk"
    for row in rows:
        assert row["processor_fingerprint"] == "plain_text@4"
    # 结构优先分块器替换了词元聚合实现；正是指纹变化让该切换可以限定范围。
    # 若检索分词器可用，摄取分块就使用它，使指纹属于配置而非裸默认值。
        assert row["chunker_fingerprint"].startswith("structure_first@")
    # 文件版本是独立事实，仍必须记录。
        assert row["source_version_id"]


def test_a_processor_upgrade_can_be_scoped_to_the_chunks_it_affects(isolated_docstore):
    """这正是核心目标：否则只能重建整个语料库。"""
    document = isolated_docstore.root / "plan.md"
    document.write_text("内容一。\n\n内容二。\n", encoding="utf-8")
    ingest_registered_document_fixture(document)

    stale = _chunks(
        isolated_docstore,
        "WHERE processor_fingerprint IS NOT ?", ("some_future_parser@9",)
    )
    current = _chunks(
        isolated_docstore,
        "WHERE processor_fingerprint IS ?", ("plain_text@4",)
    )
    assert stale and current
    assert len(stale) == len(current)


def test_unchanged_source_with_same_processor_and_chunker_is_deduped(isolated_docstore):
    document = isolated_docstore.root / "plan.md"
    document.write_text("内容一。\n\n内容二。\n", encoding="utf-8")

    first = ingest_registered_document_fixture(document)
    original_versions = docstore.list_document_versions(first["doc_id"])
    observed = document.stat()
    os.utime(
        document,
        ns=(observed.st_atime_ns, observed.st_mtime_ns + 1_000_000_000),
    )
    repeated = ingest_registered_document_fixture(document)

    assert repeated["doc_id"] == first["doc_id"]
    assert repeated["deduped"] is True
    assert repeated["reindexed"] is False
    assert docstore.list_document_versions(first["doc_id"]) == original_versions


def test_returning_to_earlier_bytes_preserves_document_file_version_lineage(
    isolated_docstore,
):
    document = isolated_docstore.root / "plan.md"
    original_content = "第一版内容。\n"
    document.write_text(original_content, encoding="utf-8")
    first = ingest_registered_document_fixture(document)
    original_version = docstore.list_document_versions(first["doc_id"])[0]

    # Register the intervening bytes without producing a Document version for them.
    isolated_docstore.write("plan.md", "第二版内容。\n")
    document.write_text(original_content, encoding="utf-8")
    repeated = ingest_registered_document_fixture(document)

    assert repeated["doc_id"] == first["doc_id"]
    assert repeated["deduped"] is False
    assert repeated["reindexed"] is True
    versions = docstore.list_document_versions(first["doc_id"])
    assert [version["version_number"] for version in versions] == [2, 1]
    assert versions[0]["file_version_id"] != original_version["file_version_id"]
    assert versions[0]["source_sha256"] == original_version["source_sha256"]
    for field in ("id", "file_version_id", "source_sha256", "source_size", "source_mtime_ns"):
        assert versions[1][field] == original_version[field]
    assert {row["source_version_id"] for row in _chunks(isolated_docstore)} == {
        versions[0]["id"]
    }


def test_unchanged_source_reindexes_when_the_typed_reader_changes(isolated_docstore, monkeypatch):
    """V1 生成的行仍可读取，新配方则会绕过过时的去重结果。"""
    document = isolated_docstore.root / "plan.md"
    document.write_text("内容一。\n\n内容二。\n", encoding="utf-8")

    expected_chunker = durable_chunking_profile().fingerprint()
    current = document_service.read_document(document)
    monkeypatch.setattr(
        document_service,
        "read_document",
        lambda path: replace(
            current,
            processor=ProcessorFingerprint(current.processor.reader, "1"),
        ),
    )
    first = ingest_registered_document_fixture(document)
    assert {row["processor_fingerprint"] for row in _chunks(isolated_docstore)} == {"plain_text@1"}

    monkeypatch.setattr(document_service, "read_document", lambda path: current)

    upgraded = ingest_registered_document_fixture(document)

    assert upgraded["doc_id"] == first["doc_id"]
    assert upgraded["deduped"] is False
    assert upgraded["reindexed"] is True
    assert {row["processor_fingerprint"] for row in _chunks(isolated_docstore)} == {"plain_text@4"}
    assert {row["chunker_fingerprint"] for row in _chunks(isolated_docstore)} == {expected_chunker}
    assert [version["version_number"] for version in docstore.list_document_versions(first["doc_id"])] == [2, 1]


def test_processor_only_reindex_preserves_user_title_and_summary(isolated_docstore, monkeypatch):
    document = isolated_docstore.root / "plan.md"
    document.write_text("内容一。\n\n内容二。\n", encoding="utf-8")
    first = ingest_registered_document_fixture(document)
    assert docstore.edit(first["doc_id"], title="用户标题", summary="用户摘要")
    original = document_service.read_document(document)
    monkeypatch.setattr(
        document_service,
        "read_document",
        lambda path: replace(
            original,
            processor=ProcessorFingerprint(original.processor.reader, "5"),
        ),
    )

    upgraded = ingest_registered_document_fixture(document)

    assert upgraded["reindexed"] is True
    stored = docstore.get_document(first["doc_id"])
    assert stored and stored["title"] == "用户标题"
    assert stored["summary"] == "用户摘要"


def test_source_reindex_preserves_user_title_and_summary(isolated_docstore):
    document = isolated_docstore.root / "plan.md"
    document.write_text("第一版内容。\n", encoding="utf-8")
    first = ingest_registered_document_fixture(document)
    assert docstore.edit(first["doc_id"], title="用户标题", summary="用户摘要")
    document.write_text("第二版内容发生变化。\n", encoding="utf-8")

    upgraded = ingest_registered_document_fixture(document)

    assert upgraded["reindexed"] is True
    stored = docstore.get_document(first["doc_id"])
    assert stored and stored["title"] == "用户标题"
    assert stored["summary"] == "用户摘要"


def test_committed_chunks_use_the_index_tokenizer(isolated_docstore):
    """按启发式规则确定大小的分块可能溢出嵌入模型的真实窗口，
    即使分块器认为它能够容纳。"""
    profile = durable_chunking_profile()
    document = isolated_docstore.root / "plan.md"
    document.write_text("# 方案\n\n第一段内容。\n\n## 风险\n\n第二段内容。\n", encoding="utf-8")

    assert ingest_registered_document_fixture(document)["ok"] is True

    for row in _chunks(isolated_docstore):
        assert row["chunker_fingerprint"] == profile.fingerprint()
    # 无论最终使用哪个可用计数器，指纹都会标明它，因此以不同方式切分的文档
    # 始终可区分，而不会静默混合。
    assert profile.tokenizer_id in _chunks(isolated_docstore)[0]["chunker_fingerprint"]


def test_chunking_profile_never_changes_counter_behind_one_fingerprint():
    class _ProbeThenFailEncoder:
        calls = 0

        def fingerprint(self) -> str:
            return "strict-test-tokenizer"

        def tokenizer_fingerprint(self) -> str:
            return "strict-test-tokenizer"

        def token_ids(self, text: str):
            del text
            self.calls += 1
            if self.calls == 1:
                return (1, 2)
            raise RuntimeError("tokenizer became unavailable")

    encoder = _ProbeThenFailEncoder()
    profile = production_document_chunking_profile(
        DocumentRetrievalProfile.lexical(),
        encoder=encoder,
    )

    assert profile.tokenizer_id == "strict-test-tokenizer"
    assert profile.tokens_of("正文") == 2
    with pytest.raises(RuntimeError, match="tokenizer became unavailable"):
        profile.tokens_of("下一段")
    assert encoder.calls == 2


def test_chunking_profile_names_and_fails_closed_with_unavailable_counter():
    class _UnavailableEncoder:
        def fingerprint(self) -> str:
            return "unavailable-test-tokenizer"

        def tokenizer_fingerprint(self) -> str:
            return "unavailable-test-tokenizer"

        def token_ids(self, text: str):
            del text
            raise RuntimeError("no tokenizer")

    profile = production_document_chunking_profile(
        DocumentRetrievalProfile.lexical(),
        encoder=_UnavailableEncoder(),
    )

    assert profile.tokenizer_id == "unavailable-test-tokenizer"
    assert profile.count_tokens is not None
    with pytest.raises(RuntimeError, match="no tokenizer"):
        profile.tokens_of("正文")


def test_durable_ingest_uses_the_effective_retrieval_tokenizer_and_limits():
    profile = durable_chunking_profile()
    tokenizer_id = DeterministicLexicalEncoder().tokenizer_fingerprint()

    assert profile.tokenizer_id == tokenizer_id
    assert profile.target_tokens == 450
    assert profile.max_tokens == 600
    assert profile.min_tokens == 80
    assert profile.split_overlap_tokens == 64
    assert profile.tokens_of("abc") == 1
    assert profile.tokens_of("论文") == 3
    assert tokenizer_id in profile.fingerprint()


def test_document_ingest_never_falls_back_to_the_structureless_chunker(isolated_docstore):
    """忘记传入分块会让后续所有摄取静默降级为词元聚合，而结构优先实现
    正是为替换这种方式而生。"""
    document = isolated_docstore.root / "plan.md"
    document.write_text("# 方案\n\n第一段。\n\n## 风险\n\n第二段。\n", encoding="utf-8")

    assert ingest_registered_document_fixture(document)["ok"] is True

    fingerprints = {
        row["chunker_fingerprint"] for row in _chunks(isolated_docstore)
    }
    assert docstore.CHUNKER_FINGERPRINT not in fingerprints
    assert all(value.startswith("structure_first@") for value in fingerprints)
