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


def _result(name, *, call_ordinal=1):
    return {
        "tool_call_id": f"call-{name}",
        "call_ordinal": call_ordinal,
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
        _attempt(2, results=[_result("new-a"), _result("new-b", call_ordinal=2)]),
    ]
    before = json.dumps(attempts)
    results, report = _project(attempts)
    assert [result["tool_call_id"] for result in results] == [
        "call-new-a",
        "call-new-b",
    ]
    assert report["durable_tool_result_count"] == 3
    assert report["omitted_tool_result_count"] == 1
    assert [result["call_ref"] for result in results] == ["c2.1", "c2.2"]
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
    partial = _result("partial", call_ordinal=2) | {
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
        "call_ref": "c4.2", "attempt_ordinal": 4, "call_ordinal": 2,
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
    assert not {"tool_call_id", "tool_result_id", "result_sha256"} & projected.keys()
    assert projected["call_ref"] == "c4.2"
    assert json.dumps(raw) == before


def test_repeated_projection_is_stable_without_consumption_state():
    attempts = [_attempt(1, results=[_result("same")])]
    assert _project(attempts) == _project(attempts)


@pytest.mark.parametrize("arguments", [
    {"items": "invalid"},
    {"items": [{"source_refs": [{"tool_result_id": "l1result_" + "f" * 64}]}]},
])
def test_rejected_findings_arguments_remain_readable_without_becoming_evidence(arguments):
    import hashlib

    raw_arguments = json.dumps(arguments, sort_keys=True, separators=(",", ":"))
    result = _result("bad-finding") | {
        "tool_id": "record_execution_findings", "status": "rejected",
        "result": None, "error": {"code": "invalid_tool_input"},
    }
    call = {
        "tool_call_id": result["tool_call_id"], "attempt_id": "attempt-1",
        "call_ordinal": 1, "tool_id": result["tool_id"], "status": "rejected",
        "outcome_hash": result["result_sha256"], "arguments_json": raw_arguments,
        "arguments_hash": hashlib.sha256(raw_arguments.encode()).hexdigest(),
    }
    execution = {"attempts": [_attempt(1, results=[result])], "tool_calls": [call]}
    before = json.dumps(execution)
    projected, _ = project_recent_tool_results(execution)
    model_result = project_result_record(projected[0])
    assert model_result["error"] == {"code": "invalid_tool_input"}
    assert model_result["arguments_unavailable"] is True
    assert model_result["arguments_unavailable_reason"] == "unsuccessful_findings_call"
    assert "arguments" not in model_result and "arguments_projection" not in model_result
    assert json.dumps(execution) == before
    call["arguments_hash"] = "f" * 64
    with pytest.raises(ToolResultProjectionError, match="authority validation"):
        project_recent_tool_results(execution)


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
                "call_ordinal": 1,
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
    assert results[0]["call_ref"] == "c1.1"
    assert report["arguments_projected_tool_result_count"] == 1


def test_recent_results_fail_closed_when_persisted_call_disagrees_with_batch():
    arguments = "{}"
    import hashlib

    execution = {
        "attempts": [_attempt(1, results=[_result("read")])],
        "tool_calls": [
            {
                "attempt_id": "attempt-1",
                "call_ordinal": 1,
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
