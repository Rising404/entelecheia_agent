"""每步 note 的原始正文与实际输入上下文分开，旧协议不得冒充当前笔记。"""

import hashlib
import json
import pytest

from personagraph.output_protocol.l1 import L1_ATTEMPT_PROTOCOL_VERSION
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
    request = {"schema_version": L1_ATTEMPT_PROTOCOL_VERSION,
               "prior_tool_results": [{"tool_result_id": "result-1"}]}
    decision = {"note": entry["claim"], "action": {"kind": "call_tools",
                 "calls": [{"tool_id": "get_today", "arguments": {}}]}}
    attempt = {"attempt_id": "attempt-1", "ordinal": 1}
    for name, value in (("request", request), ("decision", decision)):
        raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        attempt[f"{name}_json"] = raw
        attempt[f"{name}_hash"] = hashlib.sha256(raw.encode()).hexdigest()
    projection = {"ledger_id": "ledger", "ledger_revision": 1, "active_entries": [entry],
                  "omitted_active_count": 0, "remaining_durable_revisions": 20,
                  "remaining_durable_utf8_bytes": 5000}
    return projection, {"attempts": [attempt]}


def test_note_keeps_native_entry_and_input_context_without_claiming_observation():
    projection, execution = _fixture()
    output = project_execution_findings_for_model(projection, execution=execution)
    assert output["notes"] == [{"entry_id": projection["active_entries"][0]["entry_id"],
                               "summary": "继续核对目标文件。", "attempt_ordinal": 1,
                               "context_tool_result_ids": ["result-1"]}]
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
    attempt = execution["attempts"][0]
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
    entry.update(writer_tool_call_id="ordinary-tool", claim="后续已核对。",
                 source_refs=[{"tool_result_id": "result-1"}])
    output = project_execution_findings_for_model(projection, execution=execution)
    assert output["notes"][0]["source_refs"] == [{"tool_result_id": "result-1"}]
    assert output["notes"][0]["context_tool_result_ids"] == []
