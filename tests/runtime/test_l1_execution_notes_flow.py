"""公开 note 必填、真实落库、Host 存储失败及崩溃恢复边界。"""

import json
import pytest
from personagraph.model_io.gateway import ModelResult
from personagraph.persistent_turn_content.findings import ExecutionFindingsOwnerKind
from personagraph.runtime import entry
from personagraph.runtime.entry.ingress import model as ingress_model
from personagraph.runtime.entry.routing.policy import (
    TurnRoutingPolicy,
    freeze_turn_routing_policy,
)
from personagraph.runtime.l1 import controller, execution_notes
from personagraph.session import store as session_store
from personagraph.tools.contracts import ExecutionOutcome, ExecutionStatus, ToolError
from tests.helpers.prepared_model_provider import as_prepared_test_provider

PLAN = {"objective": "查询并回答", "acceptances": [{"criterion": "根据真实日期回答"}]}


def _result(payload, call_id):
    return ModelResult(
        reply=json.dumps(payload),
        provider="mock",
        model="mock-structured",
        latency_ms=1,
        model_call_id=str(call_id),
    )


@pytest.fixture
def notes_session(monkeypatch, tmp_path, bound_partitioned_session):
    session_id = bound_partitioned_session(working_dir=tmp_path)
    monkeypatch.setattr(
        ingress_model,
        "complete_structured",
        as_prepared_test_provider(
            lambda *args, **kw: _result(
                {"processing_level": "L1", "task_matches": []}, kw["model_call_id"]
            ),
        ),
    )
    return session_id


def _run(session_id):
    return entry.run_entry_turn(
        user_input="查询今天日期并回答。",
        session_id=session_id,
        client_request_id="current-note-flow",
        features={"l1_semantic_verification_mode": "off", "context_guard_limit": 24000},
        routing_policy=freeze_turn_routing_policy(
            TurnRoutingPolicy(l1_enabled=True, l2_enabled=False),
            source="request_override",
        ),
        store=session_store,
    )


def _install(monkeypatch, requests, *, missing_first=False, first_note="查询日期。"):
    def decide(_system, text, **kwargs):
        request = json.loads(text)
        requests.append(request)
        if request["plan"] is None:
            value = {
                "plan": PLAN,
                "note": first_note,
                "action": {
                    "kind": "call_tools",
                    "calls": [{"tool_id": "get_today", "arguments": {}}],
                },
            }
        else:
            source = request["prior_tool_results"][0]
            value = {
                "note": "已取得日期，提交答复。",
                "references": [{"call_ref": source["call_ref"]}],
                "action": {
                    "kind": "submit_final_reply",
                    "reply": source["result"]["date"],
                },
            }
        if missing_first and len(requests) == 1:
            value.pop("note")
        return _result(value, kwargs["model_call_id"])

    monkeypatch.setattr(
        "personagraph.runtime.l1.model.complete_structured",
        as_prepared_test_provider(decide),
    )


def _execution(result):
    return session_store.get_l1_turn_execution(
        session_id=result.session_id, turn_id=result.turn_id
    )


def _ledger(execution):
    return session_store.get_execution_findings_ledger_for_owner(
        owner_kind=ExecutionFindingsOwnerKind.L1_TURN_RUN,
        execution_owner_id=execution["run"]["l1_turn_run_id"],
    )


@pytest.mark.parametrize("first_note", ["查询日期。", "观" * 4_096])
def test_notes_record_once_without_automatic_scope_or_evidence(
    monkeypatch, notes_session, first_note
):
    requests = []
    _install(monkeypatch, requests, first_note=first_note)
    result = _run(notes_session)
    assert result.status == "completed", result
    execution = _execution(result)
    notes = _ledger(execution).active_projection.active_entries
    assert len(notes) == 2
    assert [item.claim for item in notes] == [first_note, "已取得日期，提交答复。"]
    assert all(not item.scope_keys and not item.source_refs for item in notes)
    assert requests[1]["execution_findings"]["notes"][0]["summary"] == first_note
    assert all(
        "acceptance_updates" not in json.loads(item["decision_json"])
        for item in execution["attempts"]
    )
    replayed = _run(notes_session)
    assert replayed == result
    assert len(requests) == 2
    assert len(_ledger(_execution(replayed)).active_projection.active_entries) == 2


def test_missing_note_repairs_same_attempt_before_any_effect(
    monkeypatch, notes_session
):
    requests = []
    _install(monkeypatch, requests, missing_first=True)
    result = _run(notes_session)
    assert result.status == "completed", result
    execution = _execution(result)
    assert len(requests) == 3
    assert len(execution["attempts"]) == 2
    assert len(execution["tool_calls"]) == 1
    assert len(_ledger(execution).active_projection.active_entries) == 2
    first = session_store.get_runtime_model_logical_call(
        session_id=notes_session,
        logical_call_id=execution["attempts"][0]["logical_model_call_id"],
    )
    assert len(first.physical_attempts) == 2
    assert first.physical_attempts[
        0
    ].settlement.next_output_repair_feedback.current_issues[0].paths == ("/note",)


def test_committed_note_recovers_before_tool_without_regenerating(
    monkeypatch, notes_session
):
    requests = []
    _install(monkeypatch, requests)
    original = controller._record_execution_notes

    def crash(store, ledger_id, attempt, *, execution):
        if attempt.get("decision_json"):
            raise KeyboardInterrupt("after decision before note")
        return original(store, ledger_id, attempt, execution=execution)

    monkeypatch.setattr(controller, "_record_execution_notes", crash)
    with pytest.raises(KeyboardInterrupt):
        _run(notes_session)
    assert len(requests) == 1
    monkeypatch.setattr(controller, "_record_execution_notes", original)
    result = _run(notes_session)
    assert result.status == "completed", result
    assert len(requests) == 2
    execution = _execution(result)
    assert len(execution["tool_calls"]) == 1
    assert len(_ledger(execution).active_projection.active_entries) == 2


def test_host_note_storage_failure_does_not_repair_answer_or_run_tool(
    monkeypatch, notes_session
):
    requests = []
    _install(monkeypatch, requests)
    monkeypatch.setattr(
        execution_notes,
        "execute_execution_findings_tool",
        lambda **kw: ExecutionOutcome(
            ExecutionStatus.FAILED,
            error=ToolError(code="storage_unavailable", message="offline failure"),
        ),
    )
    result = _run(notes_session)
    assert result.status == "incomplete"
    assert len(requests) == 1
    assert _execution(result)["tool_calls"] == []
