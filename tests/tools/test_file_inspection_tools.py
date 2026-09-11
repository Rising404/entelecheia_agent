"""源文本字面扫描、分块定位和会话精确版本边界。"""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from personagraph.tools.documents.file_inspection_adapter import FileInspectionRuntime
from personagraph.tools.documents.file_inspection_catalog import bind_file_inspection_tools
from personagraph.tools.execution import ToolBusinessFailure
from personagraph.tools.schema_validation import ToolSchemaCompiler
from personagraph.workspace.documents.inspection import (
    ChunkLocation, FileDocumentInspection, SourceTextElement,
)
from personagraph.workspace.documents.reading import CurrentFileDocument


def _snapshot(*, file_id="f", elements=None):
    document = CurrentFileDocument(file_id, "fv", "d", "dv", "a" * 64, 3, "complete", 2, None)
    return FileDocumentInspection(document, (
        ChunkLocation("c0", 0, "p1", (1,), ("e0",)),
        ChunkLocation("c1", 1, "p1", (1,), ("e0", "e1")),
        ChunkLocation("c2", 2, "p2", (2,), ("e1", "e2")),
    ), elements if elements is not None else (
        SourceTextElement("e0", "Net working capital", "p1#0", (1,)),
        SourceTextElement("e1", "The net working capital improved.", "p1#1", (1,)),
        SourceTextElement("e2", "Net working capital | 42", "p2#0", (2,)),
    ))


def _runtime(snapshot=None, **overrides):
    snapshot = snapshot or _snapshot()
    kwargs = dict(session_id="s", effect_scope="scope", resolve_file=lambda **kw: SimpleNamespace(
        file_id=kw["file_id"], file_version_id="fv"), revalidate=lambda _: True,
        inspect_document=lambda **_: snapshot, read_document=lambda **_: snapshot.document)
    kwargs.update(overrides)
    return FileInspectionRuntime(**kwargs)


def _target(**kwargs):
    return {"file_id": "f", "document_version_id": "dv", **kwargs}


def _call(runtime, tool_id, payload):
    registration = next(item for item in runtime.registrations if item.tool_id == tool_id)
    result = registration.handler(payload)
    ToolSchemaCompiler().compile(registration.spec.output_schema, role="output").validate(result)
    return result["results"]


def test_original_elements_count_real_repetition_without_overlap_duplicates():
    result = _call(_runtime(), "search_file_text", {"targets": [_target()], "text": "net working capital"})[0]
    assert result["total_match_count"] == 3
    assert result["scan_complete"] is True
    assert result["matches"][0]["chunk_sequences"] == [0, 1]
    assert result["matches"][2]["source_pages"] == [2]
    assert result["matches"][2]["start_character"] == 0


def test_case_and_whitespace_are_explicit_and_match_positions_remain_original():
    snapshot = _snapshot(elements=(SourceTextElement("e0", "A\n  B Straße STRASSE", "p1", (1,)),))
    runtime = _runtime(snapshot)
    assert _call(runtime, "search_file_text", {"targets": [_target()], "text": "a b"})[0]["total_match_count"] == 0
    result = _call(runtime, "search_file_text", {"targets": [_target()], "text": "a b", "whitespace": "collapse"})[0]
    assert result["matches"][0]["text"] == "A\n  B"
    assert result["matches"][0]["end_character"] == 5
    assert _call(runtime, "search_file_text", {"targets": [_target()], "text": "strasse"})[0]["total_match_count"] == 2
    assert _call(runtime, "search_file_text", {"targets": [_target()], "text": "STRASSE", "case_sensitive": True})[0]["total_match_count"] == 1


def test_search_pagination_does_not_change_total_and_old_snapshot_cannot_fake_count():
    runtime = _runtime()
    result = _call(runtime, "search_file_text", {"targets": [_target()], "text": "net working capital", "limit": 1, "match_offset": 1})[0]
    assert result["total_match_count"] == 3 and result["next_match_offset"] == 2
    assert [match["match_index"] for match in result["matches"]] == [1]
    legacy = _runtime(replace(_snapshot(), source_elements=None))
    unavailable = _call(legacy, "search_file_text", {"targets": [_target()], "text": "net"})[0]
    assert unavailable["scan_complete"] is False and unavailable["total_match_count"] is None
    assert unavailable["reason_code"] == "source_text_unavailable"


def test_index_can_batch_locate_and_paginate_without_returning_content():
    results = _call(_runtime(), "inspect_file_chunks", {"targets": [
        _target(chunk_ids=["c2", "missing", "c0"]), _target(start_sequence=1, limit=1),
    ]})
    assert results[0]["total_chunk_count"] == 3 and results[0]["last_sequence"] == 2
    assert [item["sequence"] for item in results[0]["chunks"]] == [2, 0]
    assert results[0]["unavailable_selectors"] == ["missing"]
    assert results[1]["next_sequence"] == 2
    assert "content" not in results[0]["chunks"][0]


@pytest.mark.parametrize("tool_id,payload", [
    ("inspect_file_chunks", {"targets": [_target(chunk_sequences=[0])]}),
    ("search_file_text", {"targets": [_target()], "text": "net"}),
])
def test_revocation_or_version_change_discards_all_query_results(tool_id, payload):
    decisions = iter([True, False])
    revoked = _call(_runtime(revalidate=lambda _: next(decisions)), tool_id, payload)[0]
    assert revoked["reason_code"] == "file_access_changed"
    assert not revoked.get("matches", revoked.get("chunks"))
    stale = _call(_runtime(read_document=lambda **_: None), tool_id, payload)[0]
    assert stale["reason_code"] == "document_version_changed"
    assert not stale.get("matches", stale.get("chunks"))
    missing = _call(_runtime(inspect_document=lambda **_: None), tool_id, payload)[0]
    assert missing["reason_code"] == "document_version_unavailable"
    denied = _call(_runtime(revalidate=lambda _: False,
                            inspect_document=lambda **_: pytest.fail("denied source read")), tool_id, payload)[0]
    assert denied["reason_code"] == "file_access_unavailable"
    wrong_file = _call(_runtime(_snapshot(file_id="another-file")), tool_id, payload)[0]
    assert wrong_file["reason_code"] == "document_version_unavailable"


@pytest.mark.parametrize("target", [
    _target(chunk_ids=["c0"], chunk_sequences=[0]), _target(chunk_sequences=[True]),
    _target(chunk_ids=["c0"], start_sequence=0),
])
def test_invalid_selectors_do_not_reach_document_storage(target):
    with pytest.raises(ToolBusinessFailure):
        _runtime(inspect_document=lambda **_: pytest.fail("invalid input reached storage")).inspect({"targets": [target]})


def test_new_tools_bind_through_the_production_catalog():
    import hashlib
    from personagraph.tools.composition.default_catalog import build_production_default_catalog_seeds
    from personagraph.tools.catalog.binding import BoundToolRegistration
    runtime = _runtime()
    bindings = bind_file_inspection_tools(runtime.registrations,
                                         scope_sha256=hashlib.sha256(b"scope").hexdigest(), authority_sha256="a" * 64)
    seeds = {seed.definition.identity: seed.definition for seed in build_production_default_catalog_seeds()}
    for binding in bindings:
        assert BoundToolRegistration(seeds[binding.identity], binding).tool_id in {"inspect_file_chunks", "search_file_text"}
