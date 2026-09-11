from __future__ import annotations

import pytest
from pydantic import ValidationError

from personagraph.l2.task_graph import (
    InSessionTaskGraphRevisionCommitResult,
    InSessionTaskGraphRevisionProposal,
    InSessionTaskGraphRevisionValidationContext,
    InSessionTaskGraphValidationCode,
    InSessionTaskSourceAnchor,
    validate_insession_task_graph_revision,
)


def _anchor(
    anchor_id: str,
    *,
    source_turn_id: str,
    source_kind: str,
    excerpt: str,
) -> InSessionTaskSourceAnchor:
    return InSessionTaskSourceAnchor.model_validate(
        {
            "anchor_id": anchor_id,
            "source_turn_id": source_turn_id,
            "source_kind": source_kind,
            "start": 0,
            "end": len(excerpt),
            "excerpt": excerpt,
        }
    )


def _proposal(
    *source_anchor_ids: str,
    constraints: tuple[str, ...] = (),
) -> InSessionTaskGraphRevisionProposal:
    return InSessionTaskGraphRevisionProposal.model_validate(
        {
            "root": {
                "root_key": "research",
                "nodes": [
                    {
                        "node_key": "research",
                        "node_kind": "root",
                        "title": "研究任务",
                        "objective": "分析材料并形成结论",
                        "source_anchor_ids": list(source_anchor_ids),
                        "acceptance_criteria": [
                            {
                                "acceptance_id": "analysis_complete",
                                "criterion": "结论覆盖材料分析",
                                "source_anchor_ids": list(source_anchor_ids),
                            }
                        ],
                        "constraints": list(constraints),
                    }
                ],
            }
        }
    )


def _context(
    *source_anchors: InSessionTaskSourceAnchor,
    authorization_anchor_ids: tuple[str, ...],
    required_anchor_ids: tuple[str, ...],
    expected_current_graph_revision: int | None,
) -> InSessionTaskGraphRevisionValidationContext:
    return InSessionTaskGraphRevisionValidationContext(
        session_id="session_alpha",
        source_turn_id="turn_current",
        target_insession_task_id="insession_task_alpha",
        expected_current_graph_revision=expected_current_graph_revision,
        source_anchors=source_anchors,
        authorization_anchor_ids=authorization_anchor_ids,
        required_anchor_ids=required_anchor_ids,
    )


def test_shell_initialization_accepts_host_supplied_previous_task_authority():
    shell = _anchor(
        "shell",
        source_turn_id="turn_created",
        source_kind="previously_authorized_task_state",
        excerpt="分析材料并形成结论",
    )
    proposal = _proposal("shell")
    context = _context(
        shell,
        authorization_anchor_ids=("shell",),
        required_anchor_ids=("shell",),
        expected_current_graph_revision=None,
    )

    result = validate_insession_task_graph_revision(proposal, context=context)

    assert result.status == "accepted"
    assert result.proposal == proposal
    assert result.trusted_context == context
    assert result.error_codes == ()
    assert set(proposal.model_dump(mode="json")) == {"schema_version", "root"}
    assert proposal.schema_version == "insession-task-graph-revision-v2"


def test_graph_revision_wire_contract_rejects_v1_discriminator():
    payload = _proposal("shell").model_dump(mode="json")
    payload["schema_version"] = "insession-task-graph-revision-v1"

    with pytest.raises(ValidationError):
        InSessionTaskGraphRevisionProposal.model_validate(payload)


def test_revision_accepts_current_user_and_previous_task_authority_together():
    current = _anchor(
        "request",
        source_turn_id="turn_current",
        source_kind="current_user_instruction",
        excerpt="继续原任务",
    )
    previous = _anchor(
        "previous",
        source_turn_id="turn_created",
        source_kind="previously_authorized_task_state",
        excerpt="原任务目标",
    )
    proposal = _proposal("request", "previous")
    context = _context(
        current,
        previous,
        authorization_anchor_ids=("request", "previous"),
        required_anchor_ids=("request", "previous"),
        expected_current_graph_revision=None,
    )

    result = validate_insession_task_graph_revision(proposal, context=context)

    assert result.status == "accepted"
    assert result.error_codes == ()


def test_revision_rejects_current_user_authority_from_another_turn():
    stale_current = _anchor(
        "request",
        source_turn_id="turn_old",
        source_kind="current_user_context",
        excerpt="旧输入",
    )
    context = _context(
        stale_current,
        authorization_anchor_ids=("request",),
        required_anchor_ids=("request",),
        expected_current_graph_revision=None,
    )

    result = validate_insession_task_graph_revision(
        _proposal("request"),
        context=context,
    )

    assert result.status == "rejected"
    assert result.proposal is None
    assert result.trusted_context is None
    assert InSessionTaskGraphValidationCode.SOURCE_TURN_MISMATCH in result.error_codes


def test_non_null_base_is_only_structurally_accepted_without_continuity_claims():
    previous = _anchor(
        "previous",
        source_turn_id="turn_created",
        source_kind="previously_authorized_task_state",
        excerpt="原任务目标",
    )
    context = _context(
        previous,
        authorization_anchor_ids=("previous",),
        required_anchor_ids=("previous",),
        expected_current_graph_revision=4,
    )

    result = validate_insession_task_graph_revision(
        _proposal("previous"),
        context=context,
    )

    assert result.status == "accepted"
    assert result.trusted_context is not None
    assert result.trusted_context.expected_current_graph_revision == 4


def test_revision_rejects_external_evidence_as_mutation_authority():
    retrieved = _anchor(
        "retrieved",
        source_turn_id="turn_current",
        source_kind="retrieved_document",
        excerpt="检索事实",
    )
    context = _context(
        retrieved,
        authorization_anchor_ids=("retrieved",),
        required_anchor_ids=("retrieved",),
        expected_current_graph_revision=None,
    )

    result = validate_insession_task_graph_revision(
        _proposal("retrieved"),
        context=context,
    )

    assert result.status == "rejected"
    assert (
        InSessionTaskGraphValidationCode.UNAUTHORIZED_AUTHORIZATION_SOURCE
        in result.error_codes
    )


def test_revision_reuses_source_bound_constraint_guard():
    previous = _anchor(
        "previous",
        source_turn_id="turn_created",
        source_kind="previously_authorized_task_state",
        excerpt="原任务目标",
    )
    context = _context(
        previous,
        authorization_anchor_ids=("previous",),
        required_anchor_ids=("previous",),
        expected_current_graph_revision=None,
    )

    result = validate_insession_task_graph_revision(
        _proposal("previous", constraints=("模型新增约束",)),
        context=context,
    )

    assert result.status == "rejected"
    assert InSessionTaskGraphValidationCode.UNSOURCED_CONSTRAINT in result.error_codes


def test_model_proposal_cannot_select_target_or_base_revision():
    payload = _proposal("previous").model_dump(mode="json")
    payload["target_insession_task_id"] = "insession_task_other"
    payload["expected_current_graph_revision"] = 7

    with pytest.raises(ValueError):
        InSessionTaskGraphRevisionProposal.model_validate(payload)


def test_public_commit_result_represents_one_shell_initialization():
    result = InSessionTaskGraphRevisionCommitResult(
        status="applied",
        insession_task_id="insession_task_alpha",
        previous_graph_revision=None,
        committed_graph_revision=1,
        task_state_version=2,
        turn_task_link_revision=3,
        window_state_version=4,
    )

    assert result.previous_graph_revision is None
    assert result.committed_graph_revision == 1
