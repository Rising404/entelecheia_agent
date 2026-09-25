from __future__ import annotations

import pytest
from pydantic import ValidationError

from personagraph.output_protocol import (
    AcceptanceUpdate,
    L1PlanProposal,
    SubmitFinalReplyAction,
    materialize_l1_plan,
    require_completed_support_submission,
)
from personagraph.persistent_turn_content import (
    AcceptanceProgressItem,
    EmptySupportJustification,
)
from personagraph.output_protocol.l1 import L1ResultReference
from personagraph.runtime.l1.verification import verify_l1_final_reply
from personagraph.runtime.l1.identity import canonical_json, sha256_json


def _justification() -> EmptySupportJustification:
    return EmptySupportJustification(
        reason_code="candidate_is_primary_artifact",
        explanation="The submitted answer is the primary artifact under review.",
    )


def test_acceptance_update_and_progress_allow_optional_empty_support() -> None:
    update = AcceptanceUpdate.model_validate(
        {
            "acceptance_id": "answer_ready",
            "model_claimed_satisfied": True,
            "supporting_tool_result_ids": [],
        }
    )
    progress_item = AcceptanceProgressItem.model_validate(
        {
            "acceptance_id": "answer_ready",
            "model_claimed_satisfied": True,
            "supporting_tool_result_ids": [],
        }
    )

    assert update.empty_support_justification is None
    assert progress_item.empty_support_justification is None


def test_support_ids_and_empty_support_reason_are_mutually_exclusive() -> None:
    with pytest.raises(ValidationError, match="cannot carry"):
        AcceptanceUpdate(
            acceptance_id="answer_ready",
            model_claimed_satisfied=True,
            supporting_tool_result_ids=("result-1",),
            empty_support_justification=_justification(),
        )


def test_unavailable_evidence_reason_is_not_treated_as_evidence() -> None:
    justification = EmptySupportJustification(
        reason_code="evidence_unavailable",
        explanation="The required external source could not be retrieved.",
    )
    require_completed_support_submission(
        claimed_completed=True,
        supporting_ids=(),
        empty_support_justification=justification,
        subject_label="Acceptance 'external_fact'",
    )
    assert justification.reason_code.value == "evidence_unavailable"


@pytest.mark.parametrize(
    ("tool_id", "issue"),
    [
        (
            "record_execution_findings",
            "execution_findings_result_not_admissible_evidence",
        ),
        ("list_tool_results", "tool_history_receipt_not_admissible_evidence"),
        ("read_tool_result", "tool_history_receipt_not_admissible_evidence"),
    ],
)
def test_l1_internal_receipt_is_not_admissible_evidence(tool_id, issue) -> None:
    plan = materialize_l1_plan(
        L1PlanProposal.model_validate(
            {
                "objective": "Answer the request",
                "acceptances": [
                    {
                        "criterion": "Answer the request",
                    }
                ],
            }
        ),
        input_message_id="message-1",
        user_text="Answer the request",
    )
    action = SubmitFinalReplyAction(reply="The answer is complete.")
    outcome_hash = "a" * 64
    result = verify_l1_final_reply(
        plan=plan,
        reply=action.reply,
        references=(L1ResultReference(tool_call_id="findings-call", result_sha256=outcome_hash),),
        execution={
            "tool_calls": [
                {
                    "tool_call_id": "findings-call",
                    "tool_id": tool_id,
                    "status": "succeeded",
                    "outcome_hash": outcome_hash,
                }
            ]
        },
    )

    assert result.passed is False
    assert [item.code for item in result.issues] == [issue]


@pytest.mark.parametrize("changed", ["digest", "body"])
def test_l1_supporting_call_identity_does_not_remove_result_integrity_check(changed):
    text = "Answer from the read result"
    plan = materialize_l1_plan(
        L1PlanProposal(objective=text, acceptances=({"criterion": text},)),
        input_message_id="message-1",
        user_text=text,
    )
    outcome = {"result": {"text": "original"}, "error": None}
    digest = sha256_json(outcome)
    call = {
        "tool_call_id": "read-call",
        "attempt_id": "attempt-1",
        "call_ordinal": 1,
        "tool_id": "read_text",
        "status": "succeeded",
        "outcome_hash": digest,
        "outcome_json": canonical_json(outcome),
    }
    replacement = {"result": {"text": "changed"}, "error": None}
    call["outcome_json"] = canonical_json(replacement)
    if changed == "digest":
        call["outcome_hash"] = sha256_json(replacement)
    result = verify_l1_final_reply(
        plan=plan,
        reply="The answer is original.",
        references=(L1ResultReference(tool_call_id="read-call", result_sha256=digest),),
        execution={
            "tool_calls": [call],
            "attempts": [{"attempt_id": "attempt-1", "ordinal": 1}],
        },
    )
    assert result.passed is False
    assert [issue.code for issue in result.issues] == [
        "supporting_tool_result_hash_mismatch"
        if changed == "digest"
        else "durable_result_hash_mismatch"
    ]
