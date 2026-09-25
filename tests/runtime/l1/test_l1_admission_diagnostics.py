"""L1 Host 规则在拒绝发生处生成安全、可定位的 repair issue。"""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest

from personagraph.output_protocol.l1 import (
    L1AttemptDecision,
    L1AttemptDecisionProposal,
    L1PlanProposal,
    materialize_l1_plan,
)
from personagraph.runtime.l1.controller import L1ControllerFailure, _admit_decision
from personagraph.runtime.l1.execution_notes import bind_explicit_findings_revision
from personagraph.runtime.l1.model import L1AttemptDecisionAdmissionError
from personagraph.runtime.l1.plan_revision import (
    L1PlanRevisionError,
    validate_l1_plan_revision,
)


def _plan():
    return materialize_l1_plan(
        L1PlanProposal(
            objective="验证文件",
            acceptances=[{"criterion": "交付基于文件的答复"}],
        ),
        input_message_id="message_test",
        user_text="请验证文件。",
    )


def _decision(*, calls=None, **fields):
    action = (
        {"kind": "submit_final_reply", "reply": "当前已知结果。"}
        if calls is None
        else {"kind": "call_tools", "calls": calls}
    )
    return L1AttemptDecisionProposal.model_validate(
        {"note": "检查可用信息。", "action": action, **fields}
    )


def _admit(decision, **overrides):
    runtime = SimpleNamespace(
        knows_tool=lambda tool_id: tool_id in {"inspect_file", "findings_write"},
        is_execution_findings_tool=lambda tool_id: tool_id == "findings_write",
    )
    return _admit_decision(
        decision,
        **{
            "current_plan": _plan(),
            "accepted": SimpleNamespace(
                input_message_id="message_test", turn_id="turn_test", user_input="请求"
            ),
            "l1_turn_run_id": "l1run_test",
            "tool_runtime": runtime,
            "execution": {"tool_calls": []},
            "finalization_required": False,
            "max_tool_calls_per_attempt": 8,
            **overrides,
        },
    )


def _execution_with_result(*, tool_id="read_text", status="succeeded"):
    outcome = json.dumps({
        "status": status,
        "result": {"text": "source"} if status == "succeeded" else None,
        "error": None if status == "succeeded" else {"code": "file_unavailable"},
    }, sort_keys=True, separators=(",", ":")) if status != "pending" else None
    return {
        "attempts": [{"attempt_id": "attempt_private_identity", "ordinal": 4}],
        "tool_calls": [{
            "tool_call_id": "l1tool_" + "a" * 64,
            "attempt_id": "attempt_private_identity", "call_ordinal": 2,
            "tool_id": tool_id, "status": status, "outcome_json": outcome,
            "outcome_hash": hashlib.sha256(outcome.encode()).hexdigest() if outcome else None,
        }],
    }


@pytest.mark.parametrize(
    ("decision", "overrides", "code", "path", "expected_text"),
    [
        (
            _decision(), {"current_plan": None}, "host_guard.l1_plan_required",
            "/plan", ("首次", "plan"),
        ),
        (
            _decision(calls=[{"tool_id": "inspect_file", "arguments": {}}]),
            {"finalization_required": True}, "host_guard.l1_finalization_required",
            "/action/kind", ("execution_limits.finalization_required=true", "submit_final_reply"),
        ),
        (
            _decision(calls=[{"tool_id": "inspect_file", "arguments": {}}] * 3),
            {"max_tool_calls_per_attempt": 2}, "host_guard.l1_tool_batch_limit",
            "/action/calls", ("3", "2"),
        ),
        (
            _decision(calls=[
                {"tool_id": "inspect_file", "arguments": {}},
                {"tool_id": "PRIVATE_SENTINEL", "arguments": {}},
            ]),
            {}, "host_guard.l1_unavailable_tool", "/action/calls/1/tool_id", ("tool_catalog",),
        ),
        (
            _decision(references=[{"call_ref": "c99.1"}]),
            {}, "host_guard.l1_unavailable_reference", "/references/0/call_ref",
            ("成功", "call_ref"),
        ),
        (
            _decision(calls=[{"tool_id": "findings_write", "arguments": {}}] * 2),
            {}, "host_guard.l1_findings_batch_limit", "/action/calls", ("findings", "1", "2"),
        ),
        (
            _decision(plan={"objective": "修订", "acceptances": [
                {"criterion": "新增项目"},
                {"acceptance_id": "a_deadbeefdeadbeef", "criterion": "不存在的项目"},
            ]}),
            {}, "host_guard.l1_unknown_acceptance", "/plan/acceptances/1/acceptance_id",
            ("acceptance_id", "省略"),
        ),
    ],
)
def test_admission_rule_emits_specific_safe_issue(decision, overrides, code, path, expected_text):
    with pytest.raises(L1ControllerFailure) as raised:
        _admit(decision, **overrides)
    issue = raised.value.repair_issue
    assert issue.code == code
    assert issue.paths == (path,)
    assert all(text in issue.safe_explanation for text in expected_text)
    assert "PRIVATE_SENTINEL" not in issue.model_dump_json()
    wrapped = L1AttemptDecisionAdmissionError(
        str(raised.value), terminal_error=raised.value, repair_issue=issue,
    )
    assert wrapped.repair_issue is issue


def test_unknown_host_failure_is_not_guessed_or_copied_into_repair():
    failure = L1AttemptDecisionAdmissionError(
        "PRIVATE_SENTINEL Plan path=/private/key", terminal_error=ValueError("private"),
    )
    assert failure.repair_issue.paths == ("",)
    assert "PRIVATE_SENTINEL" not in failure.repair_issue.model_dump_json()
    assert "/private/key" not in failure.repair_issue.model_dump_json()


def test_plan_revision_preserves_protected_item_diagnostic():
    previous = _plan()
    candidate = materialize_l1_plan(
        L1PlanProposal(objective="新目标", acceptances=[{"criterion": "新项目"}]),
        input_message_id="message_test", user_text="请求", previous=previous, revision=2,
    )
    with pytest.raises(L1PlanRevisionError) as raised:
        validate_l1_plan_revision(
            previous, candidate,
            protected_acceptance_ids=[previous.acceptances[0].acceptance_id],
        )
    issue = raised.value.repair_issue
    assert issue.code == "host_guard.l1_protected_acceptance_removed"
    assert issue.paths == ("/plan/acceptances",)
    assert "不能删除" in issue.safe_explanation


def test_valid_plan_and_tool_batch_remain_admissible():
    current = _plan()
    decision = _decision(calls=[{"tool_id": "inspect_file", "arguments": {}}] * 2)
    admitted, materialized = _admit(
        decision, current_plan=current, max_tool_calls_per_attempt=2,
    )
    assert isinstance(admitted, L1AttemptDecision)
    assert admitted.model_dump() == decision.model_dump()
    assert materialized is current


def test_short_reference_admission_pins_exact_call_and_result_digest():
    execution = _execution_with_result()
    admitted, _ = _admit(_decision(references=[{"call_ref": "c4.2"}]), execution=execution)
    original = execution["tool_calls"][0]
    assert isinstance(admitted, L1AttemptDecision)
    assert admitted.references[0].model_dump() == {
        "tool_call_id": original["tool_call_id"],
        "result_sha256": original["outcome_hash"],
        "chunk_id": None,
    }


@pytest.mark.parametrize(("tool_id", "status"), [
    ("read_text", "failed"), ("read_text", "pending"),
    ("list_tool_results", "succeeded"), ("read_tool_result", "succeeded"),
    ("record_execution_findings", "succeeded"),
])
def test_short_reference_cannot_turn_failed_or_bookkeeping_calls_into_evidence(tool_id, status):
    with pytest.raises(L1ControllerFailure) as raised:
        _admit(_decision(references=[{"call_ref": "c4.2"}]),
               execution=_execution_with_result(tool_id=tool_id, status=status))
    issue = raised.value.repair_issue
    assert issue.code == "host_guard.l1_unavailable_reference"
    assert issue.paths == ("/references/0/call_ref",)
    assert "l1tool_" not in issue.model_dump_json()


def test_simultaneous_violations_keep_the_first_host_rule():
    decision = _decision(calls=[{"tool_id": "PRIVATE_SENTINEL", "arguments": {}}] * 3)
    with pytest.raises(L1ControllerFailure) as raised:
        _admit(decision, finalization_required=True, max_tool_calls_per_attempt=2)
    assert raised.value.repair_issue.code == "host_guard.l1_finalization_required"
    assert raised.value.repair_issue.paths == ("/action/kind",)


def test_repeated_plan_is_still_normalized_without_repair():
    current = _plan()
    decision = _decision(plan={
        "objective": current.objective,
        "acceptances": [{
            "acceptance_id": current.acceptances[0].acceptance_id,
            "criterion": current.acceptances[0].criterion,
        }],
    })
    admitted, materialized = _admit(decision, current_plan=current)
    assert admitted.plan is None
    assert materialized is current


def test_plan_source_rebinding_has_exact_existing_item_location():
    previous = _plan()
    original_item = previous.acceptances[0]
    rebound_item = original_item.model_copy(update={
        "source": original_item.source.model_copy(update={"message_id": "other_message"}),
    })
    candidate = previous.model_copy(update={"revision": 2, "acceptances": (rebound_item,)})
    with pytest.raises(L1PlanRevisionError) as raised:
        validate_l1_plan_revision(previous, candidate)
    assert raised.value.repair_issue.code == "host_guard.l1_acceptance_source_rebound"
    assert raised.value.repair_issue.paths == ("/plan/acceptances/0/acceptance_id",)


@pytest.mark.parametrize("expected_revision", [1, True, "PRIVATE_SENTINEL"])
def test_explicit_findings_revision_rejection_is_located_and_safe(expected_revision):
    decision = _decision(calls=[
        {"tool_id": "inspect_file", "arguments": {}},
        {"tool_id": "record_execution_findings", "arguments": {
            "expected_ledger_revision": expected_revision,
        }},
    ])
    with pytest.raises(ValueError) as raised:
        bind_explicit_findings_revision(decision, request={
            "execution_notes_required": True,
            "execution_findings": {"ledger_revision": 2},
        })
    issue = raised.value.repair_issue
    assert issue.code == "host_guard.l1_findings_revision_mismatch"
    assert issue.paths == ("/action/calls/1/arguments/expected_ledger_revision",)
    assert "2" in issue.safe_explanation
    assert "省略" in issue.safe_explanation
    assert "PRIVATE_SENTINEL" not in issue.model_dump_json()


@pytest.mark.parametrize("arguments", [{}, {"expected_ledger_revision": 2}])
def test_matching_findings_revision_is_still_host_advanced_once(arguments):
    admitted = bind_explicit_findings_revision(
        _decision(calls=[{"tool_id": "record_execution_findings", "arguments": arguments}]),
        request={"execution_notes_required": True, "execution_findings": {"ledger_revision": 2}},
    )
    assert admitted.action.calls[0].arguments["expected_ledger_revision"] == 3
