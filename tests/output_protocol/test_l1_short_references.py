import pytest
from pydantic import ValidationError

from personagraph.output_protocol.l1 import L1AttemptDecisionProposal


def test_model_cites_short_call_reference_at_top_level():
    decision = L1AttemptDecisionProposal.model_validate({
        "note": "已核对来源。",
        "references": [{"call_ref": "c2.1"}],
        "action": {"kind": "submit_final_reply", "reply": "数量为 7。"},
    })
    assert decision.references[0].call_ref == "c2.1"


@pytest.mark.parametrize("reference", ["c0.1", "c01.1", "c1.0", "c1.17", "c1.1 ", "l1result_" + "a" * 64])
def test_model_rejects_noncanonical_reference(reference):
    with pytest.raises(ValidationError):
        L1AttemptDecisionProposal.model_validate({
            "note": "已核对来源。",
            "references": [{"call_ref": reference}],
            "action": {"kind": "submit_final_reply", "reply": "数量为 7。"},
        })
