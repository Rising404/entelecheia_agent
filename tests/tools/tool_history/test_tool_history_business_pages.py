"""业务分页保留结构、来源和单次预算，不重跑工具或返回 JSON 字符串切片。"""

import json
from types import SimpleNamespace

import pytest

from personagraph.persistent_turn_content.tool_results import (
    MAX_HISTORY_VALUE_BYTES, ToolResultRecord, ToolResultSource,
)
from personagraph.tools.execution import ToolBusinessFailure
from personagraph.tools.tool_history import ToolHistoryRuntime


def _runtime(result=None, error=None):
    source = ToolResultSource(tool_call_id="call", tool_id="read_text", status="failed" if error else "succeeded",
                              tool_result_id="l1result_" + "b" * 64, result_sha256="a" * 64)
    record = ToolResultRecord(source=source, outcome={"result": result, "error": error})
    calls = []
    port = SimpleNamespace(read_result=lambda **kw: calls.append(kw) or record)
    return ToolHistoryRuntime(port=port, effect_scope="run-a"), source.tool_result_id, calls


def test_nested_long_text_pages_reassemble_body_without_escaping_json():
    body = "中文📄" * 300
    runtime, result_id, calls = _runtime({"nested": {"text": body}, "file_id": "file-a"})
    text, offset = "", 0
    while True:
        page = runtime.read_result({"tool_result_id": result_id, "path": "/result/nested/text", "offset": offset, "limit": 31})
        assert page["kind"] == "string" and page["total_items"] == len(body)
        text += page["value"]
        if page["next_offset"] is None:
            break
        offset = page["next_offset"]
    assert text == body
    assert all(call == {"tool_result_id": result_id} for call in calls)


def test_array_pages_keep_full_objects_and_native_ids():
    runtime, result_id, _ = _runtime({"chunks": [{"chunk_id": "c1", "content": "one"}, {"chunk_id": "c2", "content": "two"}]})
    page = runtime.read_result({"tool_result_id": result_id, "path": "/result/chunks", "limit": 1})
    assert page["value"] == [{"chunk_id": "c1", "content": "one"}]
    assert page["next_offset"] == 1 and page["partial"]
    second = runtime.read_result({"tool_result_id": result_id, "path": page["path"], "offset": 1})
    assert second["value"][0]["chunk_id"] == "c2"


def test_large_members_provide_expand_paths_instead_of_incomplete_json():
    body = "x" * (MAX_HISTORY_VALUE_BYTES + 1)
    runtime, result_id, _ = _runtime({"text": body})
    root = runtime.read_result({"tool_result_id": result_id})
    assert root["partial"] and "/result" in root["expand_paths"]
    nested = runtime.read_result({"tool_result_id": result_id, "path": "/result"})
    assert nested["expand_paths"] == ["/result/text"]
    page = runtime.read_result({"tool_result_id": result_id, "path": "/result/text"})
    assert isinstance(page["value"], str) and page["next_offset"]
    assert len(json.dumps(page["value"]).encode()) <= MAX_HISTORY_VALUE_BYTES


def test_error_path_and_escaped_pointer_keys_are_supported():
    runtime, result_id, _ = _runtime(error={"code": "unavailable", "a/b~c": "reason"})
    page = runtime.read_result({"tool_result_id": result_id, "path": "/error/a~1b~0c"})
    assert page["value"] == "reason" and page["source"]["status"] == "failed"


@pytest.mark.parametrize("arguments", [{"path": "bad"}, {"path": "/missing"}, {"path": "/result/~2"},
                                        {"offset": -1}, {"limit": 0}, {"offset": True}])
def test_invalid_business_page_requests_fail_explicitly(arguments):
    runtime, result_id, _ = _runtime({"text": "one"})
    with pytest.raises(ToolBusinessFailure):
        runtime.read_result({"tool_result_id": result_id, **arguments})
