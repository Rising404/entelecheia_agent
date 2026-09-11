"""Current public L1 contract and Host-owned plan identities."""

import json
import pytest
from pydantic import ValidationError
from personagraph.output_protocol.l1 import (
    L1AttemptDecisionProposal,
    L1PlanProposal,
    materialize_l1_plan,
)
from personagraph.runtime.l1.model import _l1_system_prompt
from personagraph.runtime.l1.plan_revision import validate_l1_plan_revision
from personagraph.persistent_turn_content.findings import EXECUTION_FINDING_CLAIM_MAX_CHARACTERS


def decision():
    return {
        "note": "  核对材料。  ",
        "action": {"kind": "submit_final_reply", "reply": "答复"},
    }


def test_schema_is_prompt_and_parser_single_source():
    prompt = _l1_system_prompt()
    schema = json.loads(prompt[prompt.index("{") :])
    assert schema == L1AttemptDecisionProposal.model_json_schema()
    assert set(schema["required"]) == {"note", "action"}
    assert schema["properties"]["note"]["maxLength"] == EXECUTION_FINDING_CLAIM_MAX_CHARACTERS == 4_096
    for retired in (
        "scope_keys",
        "source_quote",
        "result_sha256",
        "acceptance_updates",
        "execution_notes",
    ):
        assert retired not in prompt


@pytest.mark.parametrize("note", [None, "", "  ", "x" * 4097])
def test_note_required_and_bounded(note):
    with pytest.raises(ValidationError):
        L1AttemptDecisionProposal.model_validate({**decision(), "note": note})


@pytest.mark.parametrize("character", ["x", "观"])
def test_automatic_finding_accepts_4096_characters(character):
    text = character * 4_096
    assert L1AttemptDecisionProposal.model_validate({**decision(), "note": text}).note == text


def test_note_normalized_and_old_fields_rejected():
    assert L1AttemptDecisionProposal.model_validate(decision()).note == "核对材料。"
    for field in ("execution_notes", "acceptance_updates"):
        with pytest.raises(ValidationError):
            L1AttemptDecisionProposal.model_validate({**decision(), field: []})


def test_host_plan_identity_survives_reordering_and_long_messages():
    proposal = L1PlanProposal(
        objective="完成", acceptances=({"criterion": "甲"}, {"criterion": "乙"})
    )
    plan = materialize_l1_plan(
        proposal, input_message_id="m-1", user_text="长消息" * 3000
    )
    assert plan == materialize_l1_plan(
        proposal, input_message_id="m-1", user_text="长消息" * 3000
    )
    a, b = plan.acceptances
    revised = materialize_l1_plan(
        L1PlanProposal(
            objective="细化",
            acceptances=(
                {"acceptance_id": b.acceptance_id, "criterion": "乙修订"},
                {"acceptance_id": a.acceptance_id, "criterion": "甲"},
                {"criterion": "丙"},
            ),
        ),
        input_message_id="m-1",
        user_text="长消息" * 3000,
        revision=2,
        previous=plan,
    )
    validate_l1_plan_revision(
        plan, revised, protected_acceptance_ids={a.acceptance_id, b.acceptance_id}
    )
    assert revised.acceptances[0].acceptance_id == b.acceptance_id
    assert revised.acceptances[2].acceptance_id not in {
        a.acceptance_id,
        b.acceptance_id,
    }
    assert revised.acceptances[0].source == b.source
    assert "quote" not in plan.model_dump_json()


def test_new_plan_cannot_forge_existing_id():
    proposal = L1PlanProposal(
        objective="完成",
        acceptances=({"criterion": "甲", "acceptance_id": "invented"},),
    )
    with pytest.raises(ValueError, match="existing"):
        materialize_l1_plan(proposal, input_message_id="m-1", user_text="原文")
