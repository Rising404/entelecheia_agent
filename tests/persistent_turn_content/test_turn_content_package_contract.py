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
    l1_tool_result_id,
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
    assert l1_tool_result_id.__module__ == "personagraph.persistent_turn_content.evidence"


def test_l1_tool_result_identity_preserves_persisted_hash_algorithm() -> None:
    assert l1_tool_result_id(tool_call_id="call-1", result_sha256="a" * 64) == (
        "l1result_24fe5ee8f6eea97b3d0196f5f25e5669bf6a5943d60667632cfc6086a421c805"
    )
