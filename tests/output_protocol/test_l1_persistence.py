"""Stored v5 reads normalize without rewriting frozen JSON; v6 pins outcomes."""

from copy import deepcopy
import hashlib
import json

import pytest

from personagraph.output_protocol.l1 import L1_ATTEMPT_PROTOCOL_VERSION
from personagraph.output_protocol.l1_persistence import (
    LEGACY_L1_ATTEMPT_PROTOCOL_VERSION,
    decode_l1_decision,
    legacy_l1_tool_result_id,
    normalize_l1_finding_arguments,
)
from personagraph.persistent_turn_content.json_values import freeze_json


def _case():
    outcome = json.dumps({"status": "succeeded", "result": {"answer": 7}})
    call = {
        "tool_call_id": "call-original",
        "status": "succeeded",
        "outcome_json": outcome,
        "outcome_hash": hashlib.sha256(outcome.encode()).hexdigest(),
    }
    decision = {
        "note": "已核对原文。",
        "action": {"kind": "submit_final_reply", "reply": "7"},
        "references": [{
            "tool_call_id": call["tool_call_id"], "result_sha256": call["outcome_hash"],
        }],
    }
    return decision, call


def test_current_decision_keeps_host_pin_and_rejects_model_short_ref():
    decision, call = _case()
    value = decode_l1_decision(
        decision, request_schema_version=L1_ATTEMPT_PROTOCOL_VERSION, tool_calls=[call],
    )
    assert value.references[0].tool_call_id == call["tool_call_id"]
    assert value.references[0].result_sha256 == call["outcome_hash"]
    decision["references"] = [{"call_ref": "c1.1"}]
    with pytest.raises(ValueError):
        decode_l1_decision(
            decision, request_schema_version=L1_ATTEMPT_PROTOCOL_VERSION, tool_calls=[call],
        )


def test_legacy_committed_decision_resolves_without_rewriting_frozen_payload():
    decision, call = _case()
    decision["references"] = [{"tool_result_id": legacy_l1_tool_result_id(
        tool_call_id=call["tool_call_id"], result_sha256=call["outcome_hash"],
    ), "chunk_id": "chunk-7"}]
    raw = json.dumps(decision, ensure_ascii=False)
    expected_raw, expected_call = raw, deepcopy(call)
    value = decode_l1_decision(
        raw, request_schema_version=LEGACY_L1_ATTEMPT_PROTOCOL_VERSION, tool_calls=[call],
    )
    assert value.references[0].model_dump() == {
        "tool_call_id": call["tool_call_id"], "result_sha256": call["outcome_hash"],
        "chunk_id": "chunk-7",
    }
    assert raw == expected_raw and call == expected_call


@pytest.mark.parametrize("protocol", [L1_ATTEMPT_PROTOCOL_VERSION, LEGACY_L1_ATTEMPT_PROTOCOL_VERSION])
@pytest.mark.parametrize("corruption", ["body", "digest_and_body", "missing", "duplicate", "failed"])
def test_persisted_reference_rejects_changed_or_unscoped_result(protocol, corruption):
    decision, call = _case()
    if protocol == LEGACY_L1_ATTEMPT_PROTOCOL_VERSION:
        decision["references"] = [{"tool_result_id": legacy_l1_tool_result_id(
            tool_call_id=call["tool_call_id"], result_sha256=call["outcome_hash"],
        )}]
    calls = [call]
    if corruption in {"body", "digest_and_body"}:
        call["outcome_json"] = json.dumps({"status": "succeeded", "result": {"answer": 8}})
        if corruption == "digest_and_body":
            call["outcome_hash"] = hashlib.sha256(call["outcome_json"].encode()).hexdigest()
    elif corruption == "missing":
        calls = []
    elif corruption == "duplicate":
        calls.append(deepcopy(call))
    else:
        call["status"] = "failed"
    with pytest.raises(ValueError):
        decode_l1_decision(decision, request_schema_version=protocol, tool_calls=calls)


def test_unknown_persisted_protocol_is_explicitly_rejected():
    decision, call = _case()
    with pytest.raises(ValueError, match="unsupported frozen"):
        decode_l1_decision(decision, request_schema_version="unknown", tool_calls=[call])


def test_rejected_candidate_can_retain_failed_call_as_audit_reference():
    decision, call = _case()
    call["status"] = "failed"
    call["outcome_json"] = json.dumps({"status": "failed", "error": {"code": "read_failed"}})
    call["outcome_hash"] = hashlib.sha256(call["outcome_json"].encode()).hexdigest()
    decision["references"][0]["result_sha256"] = call["outcome_hash"]
    decoded = decode_l1_decision(
        decision, request_schema_version=L1_ATTEMPT_PROTOCOL_VERSION, tool_calls=[call],
    )
    assert decoded.references[0].tool_call_id == call["tool_call_id"]
    assert decoded.references[0].result_sha256 == call["outcome_hash"]


@pytest.mark.parametrize("legacy", [False, True])
def test_finding_arguments_normalize_pins_without_mutating_frozen_inputs(legacy):
    _, call = _case()
    source = {"tool_result_id": call["tool_call_id"], "result_sha256": call["outcome_hash"],
              "chunk_id": "chunk-7"}
    if legacy:
        source = {"tool_result_id": legacy_l1_tool_result_id(
            tool_call_id=call["tool_call_id"], result_sha256=call["outcome_hash"],
        ), "chunk_id": "chunk-7"}
    arguments = freeze_json({"expected_ledger_revision": 4, "items": [{
        "kind": "finding", "claim": "Seven units", "source_refs": [source],
        "scope_keys": ["answer"],
    }]})
    raw_before = json.dumps(arguments, ensure_ascii=False)
    normalized = normalize_l1_finding_arguments(arguments, tool_calls=[call])
    assert normalized["items"][0]["source_refs"] == [{
        "tool_result_id": call["tool_call_id"], "result_sha256": call["outcome_hash"],
        "chunk_id": "chunk-7",
    }]
    assert normalized["expected_ledger_revision"] == 4
    assert normalized["items"][0]["scope_keys"] == ["answer"]
    normalized["items"][0]["claim"] = "Changed display only"
    assert json.dumps(arguments, ensure_ascii=False) == raw_before


@pytest.mark.parametrize("corruption", ["wrong_digest", "missing_digest", "body", "foreign_run"])
def test_finding_arguments_reject_unscoped_or_changed_source(corruption):
    _, call = _case()
    source = {"tool_result_id": call["tool_call_id"], "result_sha256": call["outcome_hash"]}
    calls = [call]
    if corruption == "wrong_digest":
        source["result_sha256"] = "0" * 64
    elif corruption == "missing_digest":
        source.pop("result_sha256")
    elif corruption == "body":
        call["outcome_json"] = "{}"
    else:
        calls = []
    with pytest.raises(ValueError):
        normalize_l1_finding_arguments({"items": [{"source_refs": [source]}]}, tool_calls=calls)
