"""业务投影保留原生身份与正文；只在已知协议位置移除审计信息。"""

from copy import deepcopy

from personagraph.tools.model_interface import project_file_record, project_tool_result, project_tool_result_metadata


def _file():
    return {"file_id": "f", "file_version_id": "fv", "document_id": "d",
            "document_version_id": "dv", "content_sha256": "a" * 64}


def test_file_projection_keeps_identity_and_version_not_a_second_reference():
    raw = _file() | {"relative_path": "docs/a.pdf"}
    before = deepcopy(raw)
    projected = project_file_record(raw)
    assert projected == {key: value for key, value in raw.items() if key != "content_sha256"}
    assert raw == before


def test_retrieval_groups_file_metadata_and_keeps_chunks_and_visual_identity():
    source = {"file_name": "a.pdf", "relative_path": "docs/a.pdf",
              "source_modified_at": "2026-09-01", "corpus_recorded_at": "2026-09-02"}
    rows = [_file() | {"evidence_type": "document_chunk", "chunk_id": f"c{i}",
                      "chunk_sequence": i, "snippet": '{"rank":"literal"}', "locator": "p1",
                      "source": source, "rank": i, "query_index": 0,
                      "query_matches": [{"fusion_score": 1}], "snippet_truncated": False}
            for i in range(2)]
    raw = {"contract_version": "v", "evidence": rows, "status": "partial",
           "gaps": [{"code": "file_missing", "file_id": "other"}], "truncated": True}
    before = deepcopy(raw)
    projected = project_tool_result("retrieve_files", raw)
    assert "evidence" not in projected and len(projected["files"]) == 1
    file = projected["files"][0]
    assert file["file_id"] == "f" and file["document_version_id"] == "dv"
    assert file["source"]["source_modified_at"] == "2026-09-01"
    assert "corpus_recorded_at" not in file["source"]
    assert len(file["chunks"]) == 2
    for chunk in file["chunks"]:
        assert chunk["snippet"] == '{"rank":"literal"}'
        assert not {"file_id", "rank", "query_index", "query_matches", "content_sha256"} & chunk.keys()
    assert projected["gaps"] == raw["gaps"] and projected["truncated"]
    assert raw == before


def test_picture_evidence_keeps_native_observation_identity():
    raw = {"evidence": [{"evidence_type": "picture_observation",
                        "file_id": "f", "file_version_id": "v", "picture_id": "p",
                        "picture_unit_id": "u", "observation_id": "o", "snippet": "chart",
                        "content_sha256": "a" * 64, "rank": 1}]}
    item = project_tool_result("retrieve_files", raw)["files"][0]["observations"][0]
    assert item == {"picture_id": "p", "picture_unit_id": "u", "observation_id": "o", "snippet": "chart"}


def test_chunk_body_user_dictionary_is_not_recursively_scrubbed():
    literal = {"content_sha256": "user", "file_id": "user-data"}
    raw = {"results": [_file() | {"chunks": [{"chunk_id": "c", "content": literal,
                                            "content_sha256": "a" * 64}]}],
           "unavailable_targets": [{"file_id": "f", "document_version_id": "old",
                                    "reason_code": "stale"}]}
    result = project_tool_result("read_file_chunks", raw)
    assert result["results"][0]["chunks"] == [{"chunk_id": "c", "content": literal}]
    assert result["unavailable_targets"] == raw["unavailable_targets"]


def test_history_preserves_native_source_but_never_rewrites_selected_user_value():
    source = {"tool_call_id": "call", "tool_result_id": "result", "result_sha256": "a" * 64,
              "tool_id": "read_text", "status": "succeeded"}
    raw = {"source": source, "value": {"result_sha256": "user value"},
           "path": "/result", "next_offset": 1, "partial": True}
    result = project_tool_result("read_tool_result", raw)
    assert result["source"] == {"tool_result_id": "result", "tool_id": "read_text", "status": "succeeded"}
    assert result["value"] == raw["value"] and result["next_offset"] == 1


def test_format_and_unknown_tool_body_stay_intact():
    raw = {"source": {"sha256": "a" * 64, "path": "paper.pdf"},
           "elements": [{"content": "literal file_id"}],
           "images": [{"sent_sha256": "b" * 64, "page": 1}],
           "next_cursor": {"element_offset": 1}}
    result = project_tool_result("read_pdf_text", raw)
    assert result == raw | {"source": {"path": "paper.pdf"}, "images": [{"page": 1}]}
    assert project_tool_result("custom_tool", raw) == raw


def test_host_selected_chunk_keeps_native_context_and_body_without_retrieval_diagnostics():
    from personagraph.persistent_turn_content.tool_results import select_tool_result_chunk

    literal = {"rank": "body rank", "content_sha256": "body hash"}
    raw = {"contract_version": "v", "evidence": [_file() | {
        "chunk_id": "c", "content": literal, "locator": "p1", "rank": 1,
        "query_matches": [{"rank": 1, "fusion_score": 0.5}],
        "source": {"relative_path": "a.pdf", "corpus_recorded_at": "yesterday"},
    }]}
    selected = select_tool_result_chunk(raw, "c")
    before = deepcopy(selected)
    projected = project_tool_result("retrieve_files", selected)
    assert projected["chunk"]["content"] == literal
    assert projected["chunk"]["source"] == {"relative_path": "a.pdf"}
    assert projected["chunk"]["document_version_id"] == "dv"
    assert not {"rank", "query_matches", "content_sha256"} & projected["chunk"].keys()
    assert projected["source_context"] == []
    assert selected == before
    assert project_tool_result("custom_tool", selected) == selected


def test_host_selected_chunk_preserves_file_identity_in_ancestor_context():
    from personagraph.persistent_turn_content.tool_results import select_tool_result_chunk

    selected = select_tool_result_chunk({"results": [_file() | {
        "relative_path": "a.txt", "chunks": [{"chunk_id": "c", "content": "body"}],
    }]}, "c")
    projected = project_tool_result("read_file_chunks", selected)
    assert projected["source_context"] == [{
        "file_id": "f", "file_version_id": "fv", "document_id": "d",
        "document_version_id": "dv", "relative_path": "a.txt",
    }]


def test_result_metadata_retains_only_scope_and_continuation_without_rewriting_cursor():
    scope = {"truncated": True, "partial": True, "has_more": True,
             "omitted_count": 4, "omitted_chunk_count": 2, "truncation_reason": "budget",
             "cursor": {"content_sha256": "opaque-cursor"}, "next_cursor": "next", "next_offset": 2}
    raw = scope | {"contract_version": "v", "implementation_version": "impl",
                   "sha256": "a" * 64, "unknown_body": {"private": "diagnostic"}}
    before = deepcopy(raw)
    assert project_tool_result_metadata(raw) == scope
    assert raw == before
    assert project_tool_result_metadata(None) == {}
