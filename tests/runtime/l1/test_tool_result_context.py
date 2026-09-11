"""工具正文只跨一个逻辑步骤；原始持久记录不被投影修改。"""

import json

import pytest

from personagraph.runtime.l1.tool_context import (
    ToolResultProjectionError,
    project_recent_tool_results,
    project_tool_call_arguments,
)
from personagraph.runtime.l1.model_view.projection import project_result_record


def _attempt(ordinal, *, action="call_tools", results=None):
    return {
        "attempt_id": f"attempt-{ordinal}",
        "ordinal": ordinal,
        "status": "closed",
        "action_kind": action,
        "tool_results_json": json.dumps({"tool_results": results or []})
        if action == "call_tools"
        else None,
    }


def _result(name):
    return {
        "tool_call_id": f"call-{name}",
        "tool_result_id": f"result-{name}",
        "tool_id": "source",
        "status": "succeeded",
        "result_sha256": "a" * 64,
        "result": {"text": name},
        "error": None,
        "metadata": {"partial": False},
    }


def _project(attempts):
    return project_recent_tool_results({"attempts": attempts})


def test_only_immediately_previous_batch_is_automatically_injected():
    attempts = [
        _attempt(1, results=[_result("old")]),
        _attempt(2, results=[_result("new-a"), _result("new-b")]),
    ]
    before = json.dumps(attempts)
    results, report = _project(attempts)
    assert [result["tool_call_id"] for result in results] == [
        "call-new-a",
        "call-new-b",
    ]
    assert report["durable_tool_result_count"] == 3
    assert report["omitted_tool_result_count"] == 1
    assert json.dumps(attempts) == before


@pytest.mark.parametrize(
    "last",
    [
        _attempt(2, action="submit_final_reply"),
        _attempt(2, results=[]),
    ],
)
def test_empty_or_rejected_submission_does_not_resurrect_older_batch(last):
    results, report = _project([_attempt(1, results=[_result("old")]), last])
    assert results == []
    assert report["omitted_tool_result_count"] == 1


def test_latest_failures_and_partial_results_keep_diagnostics():
    failed = _result("failure") | {
        "status": "failed",
        "result": None,
        "error": {"code": "NOT_FOUND"},
    }
    partial = _result("partial") | {
        "metadata": {
            "partial": True,
            "cursor": "next-page",
            "tool_id": "source",
            "contract_version": "1",
            "implementation_version": "local-v1",
            "execution_mode": "sync",
        }
    }
    results, report = _project([_attempt(1, results=[failed, partial])])
    assert results[0]["error"] == failed["error"]
    assert results[1]["metadata"] == {
        "partial": True,
        "cursor": "next-page",
    }
    assert report["omitted_audit_metadata_field_count"] == 4


def test_public_model_projection_retains_content_gaps_without_audit_hashes():
    raw = _result("partial") | {
        "metadata": {"partial": True, "cursor": "next-page", "contract_version": "1"},
        "result_partially_compacted": True,
        "arguments": {"path": "<absolute-path-redacted>"},
        "arguments_projection": {
            "source": "persisted_normalized_arguments",
            "arguments_sha256": "a" * 64,
            "complete": False,
            "redacted_value_count": 1,
            "truncated_value_count": 0,
        },
    }
    before = json.dumps(raw)

    projected = project_result_record(raw)

    assert projected["metadata"] == {"partial": True, "cursor": "next-page"}
    assert projected["result_partially_compacted"] is True
    assert projected["arguments_projection"] == {
        "complete": False,
        "redacted_value_count": 1,
        "truncated_value_count": 0,
    }
    assert "result_sha256" not in projected
    assert projected["tool_result_id"] == raw["tool_result_id"]
    assert json.dumps(raw) == before


def test_repeated_projection_is_stable_without_consumption_state():
    attempts = [_attempt(1, results=[_result("same")])]
    assert _project(attempts) == _project(attempts)


def test_first_decision_has_no_tool_body_or_omitted_results():
    results, report = _project([])
    assert results == []
    assert report["source_attempt_id"] is None
    assert (
        report["durable_tool_result_count"] == report["omitted_tool_result_count"] == 0
    )


@pytest.mark.parametrize(
    "raw", ["not-json", "[]", '{"tool_results":{}}', '{"tool_results":[null]}']
)
def test_corrupt_persisted_results_fail_explicitly_instead_of_disappearing(raw):
    with pytest.raises(ToolResultProjectionError):
        _project([_attempt(1) | {"tool_results_json": raw}])


def test_tool_call_arguments_projection_preserves_useful_values_and_identity():
    arguments = json.dumps(
        {
            "file_id": "file-1",
            "pages": [2, 10],
            "queries": ["capital expenditure", "2025 guidance"],
            "max_characters": 8_000,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    import hashlib

    digest = hashlib.sha256(arguments.encode("utf-8")).hexdigest()
    projection = project_tool_call_arguments(
        {"arguments_json": arguments, "arguments_hash": digest}
    )

    assert projection == {
        "arguments": {
            "file_id": "file-1",
            "max_characters": 8_000,
            "pages": [2, 10],
            "queries": ["capital expenditure", "2025 guidance"],
        },
        "arguments_projection": {
            "source": "persisted_normalized_arguments",
            "arguments_sha256": digest,
            "complete": True,
            "redacted_value_count": 0,
            "truncated_value_count": 0,
        },
    }


def test_tool_call_arguments_projection_redacts_credentials_and_private_paths():
    arguments = json.dumps(
        {
            "api_key": "private-value",
            "path": "/Users/private/document.pdf",
            "relative_path": "attachments/paper.pdf",
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    import hashlib

    digest = hashlib.sha256(arguments.encode("utf-8")).hexdigest()
    projection = project_tool_call_arguments(
        {"arguments_json": arguments, "arguments_hash": digest}
    )

    assert projection["arguments"] == {
        "api_key": "<redacted>",
        "path": "<absolute-path-redacted>",
        "relative_path": "attachments/paper.pdf",
    }
    assert projection["arguments_projection"] == {
        "source": "persisted_normalized_arguments",
        "arguments_sha256": digest,
        "complete": False,
        "redacted_value_count": 2,
        "truncated_value_count": 0,
    }


@pytest.mark.parametrize(
    "call",
    [
        {"arguments_json": "not-json", "arguments_hash": "a" * 64},
        {"arguments_json": "[]", "arguments_hash": "a" * 64},
        {"arguments_json": "{}", "arguments_hash": "a" * 64},
    ],
)
def test_tool_call_arguments_projection_rejects_corrupt_persisted_authority(call):
    with pytest.raises(ToolResultProjectionError):
        project_tool_call_arguments(call)


def test_recent_results_bind_arguments_from_the_same_persisted_tool_call():
    import hashlib

    arguments = '{"max_characters":8000,"pages":[2,10]}'
    execution = {
        "attempts": [_attempt(1, results=[_result("read")])],
        "tool_calls": [
            {
                "attempt_id": "attempt-1",
                "tool_call_id": "call-read",
                "tool_id": "source",
                "status": "succeeded",
                "outcome_hash": "a" * 64,
                "arguments_json": arguments,
                "arguments_hash": hashlib.sha256(arguments.encode("utf-8")).hexdigest(),
            }
        ],
    }

    results, report = project_recent_tool_results(execution)

    assert results[0]["arguments"] == {"max_characters": 8000, "pages": [2, 10]}
    assert results[0]["arguments_projection"]["complete"] is True
    assert report["arguments_projected_tool_result_count"] == 1


def test_recent_results_fail_closed_when_persisted_call_disagrees_with_batch():
    arguments = "{}"
    import hashlib

    execution = {
        "attempts": [_attempt(1, results=[_result("read")])],
        "tool_calls": [
            {
                "attempt_id": "attempt-1",
                "tool_call_id": "call-read",
                "tool_id": "different-tool",
                "status": "succeeded",
                "outcome_hash": "a" * 64,
                "arguments_json": arguments,
                "arguments_hash": hashlib.sha256(arguments.encode("utf-8")).hexdigest(),
            }
        ],
    }
    with pytest.raises(ToolResultProjectionError, match="disagrees"):
        project_recent_tool_results(execution)
