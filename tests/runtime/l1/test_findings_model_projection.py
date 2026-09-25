"""每步 note 的原始正文与实际输入上下文分开，旧协议不得冒充当前笔记。"""

import hashlib
import json
import pytest

from personagraph.output_protocol.l1 import L1_ATTEMPT_PROTOCOL_VERSION
from personagraph.output_protocol.l1_persistence import legacy_l1_tool_result_id
from personagraph.persistent_turn_content.findings import (
    derive_execution_finding_entry_id, derive_execution_findings_mutation_id,
    l1_execution_note_writer_id,
)
from personagraph.runtime.l1.tool_context import (
    FindingsProjectionError, project_execution_findings_for_model,
)


def _fixture():
    writer = l1_execution_note_writer_id("attempt-1")
    mutation = derive_execution_findings_mutation_id(writer_tool_call_id=writer)
    entry = {
        "entry_id": derive_execution_finding_entry_id(ledger_id="ledger", mutation_id=mutation, item_ordinal=1),
        "writer_unit_id": "attempt-1", "writer_tool_call_id": writer,
        "mutation_id": mutation, "kind": "decision", "claim": "继续核对目标文件。",
        "scope_keys": [], "source_refs": [],
    }
    outcome = json.dumps({"status": "succeeded", "result": {"text": "source"}})
    digest = hashlib.sha256(outcome.encode()).hexdigest()
    call = {"tool_call_id": "call-source", "attempt_id": "attempt-source", "call_ordinal": 1,
            "status": "succeeded", "outcome_json": outcome, "outcome_hash": digest}
    request = {"schema_version": L1_ATTEMPT_PROTOCOL_VERSION,
               "prior_tool_results": [{"tool_call_id": "call-source", "result_sha256": digest,
                                       "call_ref": "c1.1"}]}
    decision = {"note": entry["claim"], "action": {"kind": "call_tools",
                 "calls": [{"tool_id": "get_today", "arguments": {}}]}}
    attempt = {"attempt_id": "attempt-1", "ordinal": 2}
    for name, value in (("request", request), ("decision", decision)):
        raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        attempt[f"{name}_json"] = raw
        attempt[f"{name}_hash"] = hashlib.sha256(raw.encode()).hexdigest()
    projection = {"ledger_id": "ledger", "ledger_revision": 1, "active_entries": [entry],
                  "omitted_active_count": 0, "remaining_durable_revisions": 20,
                  "remaining_durable_utf8_bytes": 5000}
    source_request = json.dumps({"schema_version": L1_ATTEMPT_PROTOCOL_VERSION,
                                 "prior_tool_results": []})
    source_attempt = {"attempt_id": "attempt-source", "ordinal": 1,
                      "request_json": source_request,
                      "request_hash": hashlib.sha256(source_request.encode()).hexdigest()}
    return projection, {"attempts": [source_attempt, attempt], "tool_calls": [call]}


def test_note_keeps_native_entry_and_input_context_without_claiming_observation():
    projection, execution = _fixture()
    output = project_execution_findings_for_model(projection, execution=execution)
    assert output["notes"] == [{"entry_id": projection["active_entries"][0]["entry_id"],
                               "summary": "继续核对目标文件。", "attempt_ordinal": 2,
                               "context_call_refs": ["c1.1"]}]
    assert "observed" not in json.dumps(output) and "sha256" not in json.dumps(output)


@pytest.mark.parametrize("field,value", [
    ("claim", "伪造正文"), ("kind", "finding"), ("entry_id", "wrong"),
    ("scope_keys", ["invented"]), ("source_refs", [{"tool_result_id": "invented"}]),
])
def test_fixed_note_must_match_frozen_decision(field, value):
    projection, execution = _fixture()
    projection["active_entries"][0][field] = value
    with pytest.raises(FindingsProjectionError):
        project_execution_findings_for_model(projection, execution=execution)


@pytest.mark.parametrize("invalid_hash", [False, True])
def test_old_or_corrupt_frozen_request_fails_closed(invalid_hash):
    projection, execution = _fixture()
    attempt = execution["attempts"][-1]
    if invalid_hash:
        attempt["request_hash"] = "0" * 64
    else:
        raw = json.dumps({"schema_version": "old"})
        attempt.update(request_json=raw, request_hash=hashlib.sha256(raw.encode()).hexdigest())
    with pytest.raises(FindingsProjectionError):
        project_execution_findings_for_model(projection, execution=execution)


def test_explicit_revision_keeps_its_own_source_not_the_fixed_note_role():
    projection, execution = _fixture()
    entry = projection["active_entries"][0]
    call = execution["tool_calls"][0]
    entry.update(writer_tool_call_id="ordinary-tool", claim="后续已核对。",
                 source_refs=[{"tool_result_id": "call-source", "result_sha256": call["outcome_hash"]}])
    output = project_execution_findings_for_model(projection, execution=execution)
    assert output["notes"][0]["source_refs"] == [{"call_ref": "c1.1"}]
    assert output["notes"][0]["context_call_refs"] == []


def test_legacy_frozen_context_projects_same_short_coordinate_without_rewriting():
    projection, execution = _fixture()
    attempt = execution["attempts"][-1]
    call = execution["tool_calls"][0]
    request = {"schema_version": "l1-attempt-model-view-v5", "prior_tool_results": [
        {"tool_result_id": legacy_l1_tool_result_id(
            tool_call_id=call["tool_call_id"], result_sha256=call["outcome_hash"],
        )},
    ]}
    raw = json.dumps(request)
    attempt.update(request_json=raw, request_hash=hashlib.sha256(raw.encode()).hexdigest())
    output = project_execution_findings_for_model(projection, execution=execution)
    assert output["notes"][0]["context_call_refs"] == ["c1.1"]
    assert attempt["request_json"] == raw


def test_frozen_context_cannot_reassign_call_coordinate():
    projection, execution = _fixture()
    attempt = execution["attempts"][-1]
    request = json.loads(attempt["request_json"])
    request["prior_tool_results"][0]["call_ref"] = "c2.1"
    raw = json.dumps(request)
    attempt.update(request_json=raw, request_hash=hashlib.sha256(raw.encode()).hexdigest())
    with pytest.raises(FindingsProjectionError, match="inconsistent call coordinates"):
        project_execution_findings_for_model(projection, execution=execution)
