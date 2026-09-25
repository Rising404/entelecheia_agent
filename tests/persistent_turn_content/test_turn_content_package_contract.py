"""新包名应直接表达模型提案与本轮可恢复内容的边界。"""

from __future__ import annotations

from personagraph.output_protocol import (
    L1AttemptDecisionProposal,
    SubmitFinalReplyAction,
    SubmitOutputWindowAction,
)
from personagraph.persistent_turn_content import (
    L1Plan,
    OutputWindow,
    format_l1_call_ref,
    parse_l1_call_ref,
)
from personagraph.persistent_turn_content.findings import (
    ExecutionFindingsActiveProjection,
)


def test_public_symbols_have_one_canonical_owner() -> None:
    assert L1AttemptDecisionProposal.__module__ == "personagraph.output_protocol.l1"
    assert SubmitFinalReplyAction.__module__ == "personagraph.output_protocol.l1"
    assert SubmitOutputWindowAction.__module__ == (
        "personagraph.output_protocol.output_window"
    )
    assert L1Plan.__module__ == "personagraph.persistent_turn_content.plan"
    assert OutputWindow.__module__ == (
        "personagraph.persistent_turn_content.output_window"
    )
    assert ExecutionFindingsActiveProjection.__module__ == (
        "personagraph.persistent_turn_content.findings"
    )
    assert format_l1_call_ref.__module__ == "personagraph.persistent_turn_content.evidence"
    assert parse_l1_call_ref.__module__ == "personagraph.persistent_turn_content.evidence"


def test_l1_call_reference_preserves_durable_coordinates() -> None:
    assert format_l1_call_ref(attempt_ordinal=4, call_ordinal=2) == "c4.2"
    assert parse_l1_call_ref("c4.2") == (4, 2)
