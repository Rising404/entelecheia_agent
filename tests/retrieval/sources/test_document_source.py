from __future__ import annotations

from dataclasses import replace
import hashlib

from personagraph.retrieval.contracts import (
    SourceAccess,
    SourceAvailability,
    SourceFilter,
    SourceType,
)
from personagraph.retrieval.sources.document import (
    MountedDocumentChunkSourceAdapter,
)
from personagraph.retrieval.sources.document_policy import (
    DOCUMENT_EVIDENCE_SCOPE_KEY,
    DOCUMENT_USER_EVIDENCE_SCOPE,
)


def _hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def test_document_adapter_uses_current_chunk_and_doc_id_scope(monkeypatch):
    chunk = {
        "id": "chunk-1",
        "doc_id": "doc-1",
        "source_version_id": "version-2",
        "content": "论文方法采用双路检索。",
        "loc": "p.3",
        "title": "paper",
    }
    from personagraph.retrieval.sources import document as source_adapters

    monkeypatch.setattr(
        source_adapters.docstore,
        "get_document",
        lambda doc_id: {"id": doc_id, "current_version_id": "version-2"} if doc_id == "doc-1" else None,
    )
    monkeypatch.setattr(
        source_adapters.docstore,
        "get_current_document_chunk",
        lambda chunk_id, *, session_id=None: chunk if (chunk_id, session_id) == ("chunk-1", "s1") else None,
    )
    monkeypatch.setattr(
        source_adapters.docstore,
        "is_mounted",
        lambda doc_id, session_id: (doc_id, session_id) == ("doc-1", "s1"),
    )
    monkeypatch.setattr(
        source_adapters.session_store,
        "get_session",
        lambda session_id: {"id": session_id, "status": "active"} if session_id == "s1" else None,
    )
    monkeypatch.setattr(
        source_adapters.docstore,
        "preflight_mounted_documents",
        lambda session_id, doc_id=None: {
            "ok": True,
            "status": "verified_current",
            "snapshot_id": "snapshot-1",
            "version_map": {"doc-1": "version-2"},
        }
        if (session_id, doc_id) == ("s1", "doc-1")
        else {"ok": False, "status": "freshness_blocked", "documents": [{"status": "not_mounted"}]},
    )
    adapter = MountedDocumentChunkSourceAdapter()
    source_filter = SourceFilter.from_mapping(SourceType.DOCUMENT, {"doc_id": "doc-1", "session_id": "s1"})
    access = adapter.open_retrieval_access(source_filter)
    assert access.availability is SourceAvailability.READY
    event_ref = adapter._ref_from_chunk(chunk, session_id="s1")
    units = adapter.fetch_units(access, [event_ref])
    assert [unit.ref for unit in units] == [event_ref]
    assert units[0].citation["processing_status"] == "legacy_unknown"
    assert units[0].citation["processing_diagnostic_codes"] == "null"
    assert units[0].citation["needs_vision"] == "unknown"
    assert units[0].citation["coverage_gap"] == "unknown"
    monkeypatch.setattr(
        source_adapters.docstore,
        "list_current_document_chunks",
        lambda doc_id, *, session_id: [chunk] if (doc_id, session_id) == ("doc-1", "s1") else [],
    )
    assert [unit.ref for unit in adapter.list_indexable_units_for_backfill(source_filter)] == [event_ref]
    wrong_doc = SourceFilter.from_mapping(SourceType.DOCUMENT, {"doc_id": "doc-2", "session_id": "s1"})
    assert adapter.fetch_units(adapter.open_retrieval_access(wrong_doc), [event_ref]) == ()

    # 文档挂载受所属会话约束。软删除会话必须立即关闭该作用域，
    # 即使其派生文档单元的异步删除尚未完成。
    monkeypatch.setattr(
        source_adapters.session_store,
        "get_session",
        lambda session_id: {"id": session_id, "status": "trashed"} if session_id == "s1" else None,
    )
    blocked_access = adapter.open_retrieval_access(source_filter)
    assert blocked_access.availability is SourceAvailability.BLOCKED
    assert adapter.fetch_units(blocked_access, [event_ref]) == ()
    assert adapter.list_indexable_units_for_backfill(source_filter) == ()


def test_document_adapter_uses_durable_preflight_by_default(monkeypatch):
    from personagraph.retrieval.sources import document as source_adapters

    calls = {"preflight": 0, "check": 0}
    monkeypatch.setattr(
        source_adapters.session_store,
        "get_session",
        lambda session_id: {"id": session_id, "status": "active"},
    )
    monkeypatch.setattr(
        source_adapters.docstore,
        "preflight_mounted_documents",
        lambda session_id, doc_id=None: calls.__setitem__(
            "preflight", calls["preflight"] + 1
        )
        or {
            "ok": True,
            "status": "verified_current",
            "snapshot_id": "durable-snapshot-1",
            "version_map": {"doc-1": "version-2"},
        },
    )
    monkeypatch.setattr(
        source_adapters.docstore,
        "check_mounted_document_freshness",
        lambda session_id, doc_id=None: calls.__setitem__("check", calls["check"] + 1)
        or {"ok": False},
    )

    access = MountedDocumentChunkSourceAdapter().open_retrieval_access(
        SourceFilter.from_mapping(
            SourceType.DOCUMENT,
            {"doc_id": "doc-1", "session_id": "s1"},
        )
    )

    assert access.availability is SourceAvailability.READY
    assert access.source_snapshot_id == "durable-snapshot-1"
    assert calls == {"preflight": 1, "check": 0}


def test_document_adapter_empty_backfill_scope_enumerates_the_whole_project(
    monkeypatch,
):
    from personagraph.retrieval.sources import document as source_adapters

    chunks = {
        "doc-1": [{
            "id": "storage-1",
            "doc_id": "doc-1",
            "source_version_id": "version-1",
            "producer_chunk_id": "chunk-a",
            "content": "第一份项目文档。",
        }],
        "doc-2": [{
            "id": "storage-2",
            "doc_id": "doc-2",
            "source_version_id": "version-4",
            "producer_chunk_id": "chunk-b",
            "content": "第二份项目文档。",
        }],
    }
    monkeypatch.setattr(
        source_adapters.docstore,
        "list_documents",
        lambda: [{"id": "doc-2"}, {"id": "doc-1"}],
    )
    monkeypatch.setattr(
        source_adapters.docstore,
        "list_current_document_chunks",
        lambda doc_id, *, session_id=None: chunks[doc_id]
        if session_id is None
        else [],
    )

    units = MountedDocumentChunkSourceAdapter().list_indexable_units_for_backfill(
        SourceFilter.from_mapping(SourceType.DOCUMENT)
    )

    assert [unit.ref.source_unit_id for unit in units] == [
        "document-v3:doc-2:chunk-b",
        "document-v3:doc-1:chunk-a",
    ]
    assert [unit.source_filter.as_mapping() for unit in units] == [
        {"doc_id": "doc-2"},
        {"doc_id": "doc-1"},
    ]


def test_document_adapter_read_only_preflight_uses_freshness_and_hides_snapshot_ids(
    monkeypatch,
):
    from personagraph.retrieval.sources import document as source_adapters

    calls = {"preflight": 0, "check": 0}
    version_maps = (
        {
            "private-document-b": "private-version-b",
            "private-document-a": "private-version-a",
        },
        {
            "private-document-a": "private-version-a",
            "private-document-b": "private-version-b",
        },
    )
    monkeypatch.setattr(
        source_adapters.session_store,
        "get_session",
        lambda session_id: {"id": session_id, "status": "active"},
    )
    monkeypatch.setattr(
        source_adapters.docstore,
        "preflight_mounted_documents",
        lambda session_id, doc_id=None: calls.__setitem__(
            "preflight", calls["preflight"] + 1
        )
        or {"ok": False},
    )

    def _check(session_id, doc_id=None):
        assert (session_id, doc_id) == ("private-session", None)
        calls["check"] += 1
        return {
            "ok": True,
            "status": "verified_current",
            "version_map": version_maps[(calls["check"] - 1) % len(version_maps)],
            "documents": [
                {
                    "doc_id": "private-document-a",
                    "version_id": "private-version-a",
                    "processing_status": "complete",
                    "diagnostics": [],
                },
                {
                    "doc_id": "private-document-b",
                    "version_id": "private-version-b",
                    "processing_status": "partial",
                    "diagnostics": [
                        {"code": "page_needs_vision", "detail": "private"},
                        {"code": "parser_partial", "detail": "private"},
                    ],
                },
            ],
        }

    monkeypatch.setattr(
        source_adapters.docstore,
        "check_mounted_document_freshness",
        _check,
    )
    adapter = MountedDocumentChunkSourceAdapter(preflight_mode="read_only")
    source_filter = SourceFilter.from_mapping(
        SourceType.DOCUMENT,
        {"session_id": "private-session"},
    )

    first = adapter.open_retrieval_access(source_filter)
    second = adapter.open_retrieval_access(source_filter)
    revalidated = adapter.revalidate_retrieval_access(first)

    assert first.availability is SourceAvailability.READY
    assert first.source_snapshot_id
    assert first.source_snapshot_id == second.source_snapshot_id
    assert revalidated.source_snapshot_id == first.source_snapshot_id
    assert first.source_snapshot_id.startswith("document-readonly:")
    assert all(
        value not in first.source_snapshot_id
        for value in (
            "private-session",
            "private-document-a",
            "private-document-b",
            "private-version-a",
            "private-version-b",
        )
    )
    assert dict(first.coverage_facts) == {
        "processing_status": "partial",
        "processing_diagnostic_codes": '["page_needs_vision","parser_partial"]',
        "needs_vision": "true",
        "coverage_gap": "true",
    }
    assert calls == {"preflight": 0, "check": 3}


def test_document_adapter_uses_doc_scoped_typed_identity_and_pinned_lookup(monkeypatch):
    normalized_content = "论文方法采用双路检索。"
    raw_content = f"  {normalized_content} \n"
    chunk = {
        "id": "storage-1",
        "doc_id": "doc-1",
        "source_version_id": "version-2",
        "producer_chunk_id": "ch_method",
        "chunk_contract_version": 1,
        "content": raw_content,
        "loc": "p3",
        "span_json": '{"end":{"page":3},"start":{"page":3}}',
        "metadata_json": '{"section_path":["方法"]}',
        "content_sha256": _hash(raw_content),
        "title": "paper",
        "processing_status": "partial",
        "diagnostics_json": (
            '[{"detail":"fallback","code":"fallback_reader_used"},'
            '{"detail":"scanned page","at":"p4","code":"page_needs_vision"}]'
        ),
    }
    from personagraph.retrieval.sources import document as source_adapters

    monkeypatch.setattr(
        source_adapters.session_store,
        "get_session",
        lambda session_id: {"id": session_id, "status": "active"} if session_id == "s1" else None,
    )
    monkeypatch.setattr(
        source_adapters.docstore,
        "preflight_mounted_documents",
        lambda session_id, doc_id=None: {
            "ok": True,
            "status": "verified_current",
            "snapshot_id": "snapshot-1",
            "version_map": {"doc-1": "version-2"},
            "documents": [{
                "doc_id": "doc-1",
                "version_id": "version-2",
                "status": "verified_current",
                "processing_status": "partial",
                "diagnostics": [
                    {"code": "page_needs_vision", "at": "p4", "detail": "private"},
                    {"code": "fallback_reader_used", "at": "p1", "detail": "private"},
                ],
                "needs_vision": True,
            }],
        },
    )
    monkeypatch.setattr(
        source_adapters.docstore,
        "get_current_typed_document_chunk",
        lambda doc_id, producer_chunk_id, *, expected_version_id, session_id: chunk
        if (
            (doc_id, producer_chunk_id, expected_version_id)
            == ("doc-1", "ch_method", "version-2")
            and session_id in {"s1", None}
        )
        else None,
    )
    monkeypatch.setattr(
        source_adapters.docstore,
        "get_current_document_chunk",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("typed identity must not use the legacy storage-row lookup")
        ),
    )
    monkeypatch.setattr(
        source_adapters.docstore,
        "is_mounted",
        lambda doc_id, session_id: (doc_id, session_id) == ("doc-1", "s1"),
    )
    monkeypatch.setattr(
        source_adapters.docstore,
        "list_current_document_chunks",
        lambda doc_id, *, session_id: [chunk]
        if (doc_id, session_id) == ("doc-1", "s1")
        else [],
    )
    adapter = MountedDocumentChunkSourceAdapter()
    source_filter = SourceFilter.from_mapping(
        SourceType.DOCUMENT, {"doc_id": "doc-1", "session_id": "s1"}
    )
    access = adapter.open_retrieval_access(source_filter)
    assert dict(access.coverage_facts) == {
        "processing_status": "partial",
        "processing_diagnostic_codes": '["fallback_reader_used","page_needs_vision"]',
        "needs_vision": "true",
        "coverage_gap": "true",
    }
    ref = adapter._ref_from_chunk(chunk, session_id="s1")

    assert ref.source_unit_id == "document-v3:doc-1:ch_method"
    assert adapter._ref_from_chunk(chunk, session_id="s2") == ref
    units = adapter.fetch_units(access, [ref])
    assert [unit.ref for unit in units] == [ref]
    assert units[0].content == normalized_content
    assert units[0].ref.indexed_content_hash == _hash(units[0].content)
    assert units[0].citation["producer_chunk_id"] == "ch_method"
    assert units[0].citation["span_json"] == chunk["span_json"]
    assert units[0].citation["processing_status"] == "partial"
    assert units[0].citation["processing_diagnostic_codes"] == (
        '["fallback_reader_used","page_needs_vision"]'
    )
    assert units[0].citation["needs_vision"] == "true"
    assert units[0].citation["coverage_gap"] == "true"
    assert "scanned page" not in str(units[0].citation)
    assert adapter.fetch_units(
        access,
        [replace(ref, source_unit_id="document-v3:doc-2:ch_method")],
    ) == ()
    assert adapter.fetch_units(
        access,
        [replace(ref, source_revision="version-1")],
    ) == ()
    assert adapter.fetch_units(
        access,
        [replace(ref, source_unit_id="s2:document-v2:doc-1:ch_method")],
    ) == ()

    indexable = adapter.read_current_for_reconcile(ref)
    assert indexable is not None
    assert indexable.content == normalized_content
    assert indexable.ref.indexed_content_hash == _hash(indexable.content)
    assert indexable.source_filter.as_mapping() == {"doc_id": "doc-1"}

    assert [
        item.as_mapping() for item in adapter.catalog_source_filters(access)
    ] == [{"doc_id": "doc-1"}]

    backfill = adapter.list_indexable_units_for_backfill(source_filter)
    assert len(backfill) == 1
    assert backfill[0].content == normalized_content
    assert backfill[0].ref.indexed_content_hash == _hash(backfill[0].content)


def test_document_access_reports_physical_source_change_before_search(monkeypatch):
    from personagraph.retrieval.sources import document as source_adapters

    monkeypatch.setattr(
        source_adapters.session_store,
        "get_session",
        lambda session_id: {"id": session_id, "status": "active"} if session_id == "s1" else None,
    )
    monkeypatch.setattr(
        source_adapters.docstore,
        "preflight_mounted_documents",
        lambda session_id, doc_id=None: {
            "ok": False,
            "status": "freshness_blocked",
            "documents": [{"doc_id": "doc-1", "status": "source_changed"}],
        },
    )
    adapter = MountedDocumentChunkSourceAdapter()
    source_filter = SourceFilter.from_mapping(SourceType.DOCUMENT, {"doc_id": "doc-1", "session_id": "s1"})

    access = adapter.open_retrieval_access(source_filter)

    assert access.availability is SourceAvailability.BLOCKED
    assert access.reason_code == "document_source_changed"
    assert access.source_snapshot_id is None


def test_document_access_refuses_content_from_a_newer_source_revision(monkeypatch):
    from personagraph.retrieval.sources import document as source_adapters

    chunk = {
        "id": "chunk-1",
        "doc_id": "doc-1",
        "source_version_id": "version-2",
        "content": "新版内容。",
        "loc": "p.1",
        "title": "paper",
    }
    monkeypatch.setattr(
        source_adapters.session_store,
        "get_session",
        lambda session_id: {"id": session_id, "status": "active"} if session_id == "s1" else None,
    )
    monkeypatch.setattr(
        source_adapters.docstore,
        "preflight_mounted_documents",
        lambda session_id, doc_id=None: {
            "ok": True,
            "status": "verified_current",
            "snapshot_id": "snapshot-1",
            "version_map": {"doc-1": "version-1"},
        },
    )
    monkeypatch.setattr(
        source_adapters.docstore,
        "get_current_document_chunk",
        lambda chunk_id, *, session_id=None: chunk if (chunk_id, session_id) == ("chunk-1", "s1") else None,
    )
    adapter = MountedDocumentChunkSourceAdapter()
    source_filter = SourceFilter.from_mapping(SourceType.DOCUMENT, {"doc_id": "doc-1", "session_id": "s1"})
    access = adapter.open_retrieval_access(source_filter)
    candidate_ref = adapter._ref_from_chunk(chunk, session_id="s1")

    assert adapter.fetch_units(access, [candidate_ref]) == ()


def test_document_index_binding_snapshot_cannot_expand_a_document_scoped_access():
    adapter = MountedDocumentChunkSourceAdapter()
    source_filter = SourceFilter.from_mapping(
        SourceType.DOCUMENT,
        {"doc_id": "doc-1", "session_id": "s1"},
    )
    forged_access = SourceAccess(
        SourceType.DOCUMENT,
        source_filter,
        SourceAvailability.READY,
        source_snapshot_id="snapshot-1",
        source_revision_map={"doc-2": "version-1"},
    )

    snapshot = adapter.get_current_index_binding_snapshot(forged_access, maximum_bindings=5_000)

    assert not snapshot.source_snapshot_is_current
    assert snapshot.bindings == ()


# --- 检索结果要说清这段文字是谁写的 ---------------------------------------------
#
# agent 自己的产出和用户给的材料进的是同一个索引，取出来长得一模一样。不标来源，
# 模型几轮之后会把上一轮自己的草稿当成原文引用。


def _document_adapter_for(
    monkeypatch,
    path: str,
    *,
    file_origin: str | None = None,
):
    from personagraph.retrieval.sources import document as source_adapters

    chunk = {
        "id": "chunk-1",
        "doc_id": "doc-1",
        "source_version_id": "version-2",
        "content": "第三区域的吞吐量为 938。",
        "loc": "p.1",
        "title": "report",
        "path": path,
    }
    if file_origin is not None:
        chunk["file_origin"] = file_origin
    monkeypatch.setattr(
        source_adapters.docstore,
        "get_document",
        lambda doc_id: {"id": doc_id, "current_version_id": "version-2"},
    )
    monkeypatch.setattr(
        source_adapters.docstore,
        "get_current_document_chunk",
        lambda chunk_id, *, session_id=None: chunk,
    )
    monkeypatch.setattr(
        source_adapters.docstore, "is_mounted", lambda doc_id, session_id: True
    )
    monkeypatch.setattr(
        source_adapters.session_store,
        "get_session",
        lambda session_id: {"id": session_id, "status": "active"},
    )
    monkeypatch.setattr(
        source_adapters.docstore,
        "preflight_mounted_documents",
        lambda session_id, doc_id=None: {
            "ok": True,
            "status": "verified_current",
            "snapshot_id": "snapshot-1",
            "version_map": {"doc-1": "version-2"},
        },
    )
    adapter = MountedDocumentChunkSourceAdapter()
    source_filter = SourceFilter.from_mapping(
        SourceType.DOCUMENT, {"doc_id": "doc-1", "session_id": "s1"}
    )
    access = adapter.open_retrieval_access(source_filter)
    return adapter.fetch_units(access, [adapter._ref_from_chunk(chunk, session_id="s1")])


def test_a_citation_prefers_the_project_file_origin_ledger(monkeypatch):
    cases = {
        "agent_output": "agent_output",
        "user_upload": "user_upload",
        "workspace": "workspace_existing",
    }
    for expected, file_origin in cases.items():
        units = _document_adapter_for(
            monkeypatch,
            "/path-is-not-origin-authority",
            file_origin=file_origin,
        )
        assert units[0].citation["origin"] == expected


def test_the_origin_reaches_the_prompt_text(monkeypatch):
    """模型从未看到的字段不会改变其任何行为。"""

    from personagraph.retrieval.prompt_context import _citation_text

    units = _document_adapter_for(
        monkeypatch,
        "/path-is-not-origin-authority",
        file_origin="agent_output",
    )
    assert "origin=agent_output" in _citation_text(units[0].citation)


def test_an_unknown_path_is_reported_as_workspace_rather_than_crashing(monkeypatch):
    units = _document_adapter_for(monkeypatch, "")
    assert units[0].citation["origin"] == "workspace"


def test_user_evidence_scope_filters_agent_outputs_before_catalog_and_revalidation(
    monkeypatch,
):
    from personagraph.retrieval.sources import document as source_adapters

    mounted_documents = [
        {
            "id": "doc-upload",
            "path": "/project/input/upload.pdf",
            "file_origin": "user_upload",
        },
        {
            "id": "doc-workspace",
            "path": "/project/reference.md",
            "file_origin": "workspace_existing",
        },
        {
            "id": "doc-agent-ledger",
            "path": "/project/reference-looking.md",
            "file_origin": "agent_output",
        },
        {
            "id": "doc-agent-path",
            "path": "/project/output/draft.md",
        },
    ]
    version_map = {
        document["id"]: f"version-{index}"
        for index, document in enumerate(mounted_documents, start=1)
    }
    report = {
        "ok": True,
        "status": "verified_current",
        "snapshot_id": "snapshot-all-mounted",
        "version_map": version_map,
        "documents": [
            {
                "doc_id": document_id,
                "version_id": version_id,
                "status": "verified_current",
                "processing_status": "complete",
                "diagnostics": [],
            }
            for document_id, version_id in version_map.items()
        ],
    }
    monkeypatch.setattr(
        source_adapters.session_store,
        "get_session",
        lambda session_id: {"id": session_id, "status": "active"},
    )
    monkeypatch.setattr(
        source_adapters.docstore,
        "mounted_docs",
        lambda session_id: mounted_documents if session_id == "s1" else [],
    )
    monkeypatch.setattr(
        source_adapters.docstore,
        "preflight_mounted_documents",
        lambda session_id, doc_id=None: dict(report),
    )
    monkeypatch.setattr(
        source_adapters.docstore,
        "check_mounted_document_freshness",
        lambda session_id, doc_id=None: dict(report),
    )
    monkeypatch.setattr(
        source_adapters,
        "_document_path_area",
        lambda session_id, path: "output" if path.endswith("/draft.md") else "input",
    )
    source_filter = SourceFilter.from_mapping(
        SourceType.DOCUMENT,
        {
            "session_id": "s1",
            DOCUMENT_EVIDENCE_SCOPE_KEY: DOCUMENT_USER_EVIDENCE_SCOPE,
        },
    )
    adapter = MountedDocumentChunkSourceAdapter()

    access = adapter.open_retrieval_access(source_filter)
    revalidated = adapter.revalidate_retrieval_access(access)

    expected_map = {
        "doc-upload": "version-1",
        "doc-workspace": "version-2",
    }
    assert dict(access.source_revision_map) == expected_map
    assert dict(revalidated.source_revision_map) == expected_map
    assert [
        source_filter.as_mapping()
        for source_filter in adapter.catalog_source_filters(access)
    ] == [
        {"doc_id": "doc-upload"},
        {"doc_id": "doc-workspace"},
    ]


def test_explicit_agent_document_is_filtered_only_when_user_evidence_is_requested(
    monkeypatch,
):
    from personagraph.retrieval.sources import document as source_adapters

    report = {
        "ok": True,
        "status": "verified_current",
        "snapshot_id": "snapshot-agent",
        "version_map": {"doc-agent": "version-agent"},
        "documents": [{
            "doc_id": "doc-agent",
            "version_id": "version-agent",
            "status": "verified_current",
        }],
    }
    monkeypatch.setattr(
        source_adapters.session_store,
        "get_session",
        lambda session_id: {"id": session_id, "status": "active"},
    )
    monkeypatch.setattr(
        source_adapters.docstore,
        "preflight_mounted_documents",
        lambda session_id, doc_id=None: dict(report),
    )
    monkeypatch.setattr(
        source_adapters.docstore,
        "mounted_docs",
        lambda session_id: [{
            "id": "doc-agent",
            "path": "/project/output/draft.md",
            "file_origin": "agent_output",
        }],
    )
    monkeypatch.setattr(source_adapters, "_document_path_area", lambda *args: "output")
    adapter = MountedDocumentChunkSourceAdapter()
    ordinary_filter = SourceFilter.from_mapping(
        SourceType.DOCUMENT,
        {"session_id": "s1", "doc_id": "doc-agent"},
    )
    evidence_filter = SourceFilter.from_mapping(
        SourceType.DOCUMENT,
        {
            "session_id": "s1",
            "doc_id": "doc-agent",
            DOCUMENT_EVIDENCE_SCOPE_KEY: DOCUMENT_USER_EVIDENCE_SCOPE,
        },
    )

    ordinary = adapter.open_retrieval_access(ordinary_filter)
    filtered = adapter.open_retrieval_access(evidence_filter)

    assert ordinary.availability is SourceAvailability.READY
    assert dict(ordinary.source_revision_map) == {"doc-agent": "version-agent"}
    assert filtered.availability is SourceAvailability.EMPTY
    assert dict(filtered.source_revision_map) == {}
