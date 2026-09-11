from __future__ import annotations

from dataclasses import dataclass

import pytest

from personagraph.l2.auxiliary_graph import (
    AuxiliaryGraphEdgeProposal,
    AuxiliaryGraphRevisionProposalDisposition,
    AuxiliaryGraphRevisionProposal,
    AuxiliaryGraphRevisionReason,
    AuxiliaryGraphStructureProposal,
    AuxiliaryNodeExecutorKind,
    AuxiliaryNodeKind,
    AuxiliaryNodeProposal,
)
from personagraph.l2.task_graph import InSessionTaskAcceptanceProposal
from personagraph.l2.auxiliary_execution.planning.architect import (
    AuxiliaryGraphArchitectAction,
    AuxiliaryGraphArchitectDecision,
)
from personagraph.l2.auxiliary_execution.planning.architect_adapter import (
    AuxiliaryArchitectAdapterError,
    auxiliary_graph_revision_proposal_from_architect_decision,
    build_terminal_only_auxiliary_graph_bootstrap_proposal,
)
from personagraph.session.persistence.l2.auxiliary_graph.auxiliary_graphs import (
    AuxiliaryGraphRevisionProposalRecord,
)


@dataclass(frozen=True)
class _DecisionRequest:
    architect_request_id: str = "architect_request_adapter_01"
    binding_sha256: str = "a" * 64
    logical_call_id: str = "architect_logical_call_adapter_01"


def _acceptance(
    acceptance_id: str,
    *,
    anchors: tuple[str, ...] = ("auth_primary", "evidence_paper"),
) -> InSessionTaskAcceptanceProposal:
    return InSessionTaskAcceptanceProposal(
        acceptance_id=acceptance_id,
        criterion="Produce a source-bound and inspectable result.",
        source_anchor_ids=anchors,
    )


def _structure(*, with_origins: bool) -> AuxiliaryGraphStructureProposal:
    return AuxiliaryGraphStructureProposal(
        terminal_node_key="terminal",
        nodes=(
            AuxiliaryNodeProposal(
                node_key="analyze",
                node_kind=AuxiliaryNodeKind.ANALYZE,
                executor_kind=AuxiliaryNodeExecutorKind.MODEL_WORK_RUN,
                title="Analyze the mounted evidence",
                objective="Extract the exact facts needed by the terminal planner.",
                acceptance_criteria=(_acceptance("analysis_grounded"),),
                capability_profile_id="bounded_research_v1",
                input_resource_aliases=("artifact_context", "resource_input"),
                source_anchor_ids=("auth_primary", "evidence_paper"),
                output_contract="planning_context_artifact_v1",
                required=False,
                origin_node_alias="prior_analyze" if with_origins else None,
            ),
            AuxiliaryNodeProposal(
                node_key="terminal",
                node_kind=AuxiliaryNodeKind.SYNTHESIZE,
                executor_kind=AuxiliaryNodeExecutorKind.TERMINAL_PLANNER,
                title="Synthesize the task graph",
                objective="Produce the complete source-bound TaskGraph proposal.",
                acceptance_criteria=(_acceptance("task_graph_grounded"),),
                capability_profile_id=None,
                input_resource_aliases=("artifact_context",),
                source_anchor_ids=("auth_primary", "evidence_paper"),
                output_contract="task_graph_revision_proposal_v2",
                required=True,
                origin_node_alias="prior_terminal" if with_origins else None,
            ),
        ),
        edges=(
            AuxiliaryGraphEdgeProposal(
                source_node_key="analyze",
                target_node_key="terminal",
                required=True,
            ),
        ),
    )


def _revision_proposal(
    *,
    disposition: AuxiliaryGraphRevisionProposalDisposition,
    expected_current_revision: int | None,
    structure: AuxiliaryGraphStructureProposal | None,
    reason: AuxiliaryGraphRevisionReason | None,
    question: str | None = None,
    failure_reason: str | None = None,
) -> AuxiliaryGraphRevisionProposal:
    return AuxiliaryGraphRevisionProposal(
        disposition=disposition,
        expected_current_auxiliary_graph_revision=expected_current_revision,
        revision_reason=reason,
        structure=structure,
        explanation="This decision is bound to the exact current planning authority.",
        blocking_gap_ids=("blocking_gap",) if question is not None else (),
        requested_user_question=question,
        failure_reason=failure_reason,
    )


def _decision(
    proposal: AuxiliaryGraphRevisionProposal,
) -> AuxiliaryGraphArchitectDecision:
    return AuxiliaryGraphArchitectDecision.create(
        request=_DecisionRequest(),  # type: ignore[arg-type]
        proposal=proposal,
    )


@pytest.mark.parametrize(
    ("disposition", "current_revision", "reason", "with_origins"),
    (
        (
            AuxiliaryGraphRevisionProposalDisposition.CREATE_REVISION,
            None,
            AuxiliaryGraphRevisionReason.INITIAL,
            False,
        ),
        (
            AuxiliaryGraphRevisionProposalDisposition.REVISE_REVISION,
            7,
            AuxiliaryGraphRevisionReason.EVIDENCE_CHANGED,
            True,
        ),
    ),
)
def test_create_and_revise_structure_conversion_is_field_exact(
    disposition: AuxiliaryGraphRevisionProposalDisposition,
    current_revision: int | None,
    reason: AuxiliaryGraphRevisionReason,
    with_origins: bool,
) -> None:
    structure = _structure(with_origins=with_origins)
    decision = _decision(
        _revision_proposal(
            disposition=disposition,
            expected_current_revision=current_revision,
            structure=structure,
            reason=reason,
        )
    )

    converted = auxiliary_graph_revision_proposal_from_architect_decision(
        decision,
        expected_current_auxiliary_graph_revision=current_revision,
    )

    assert isinstance(converted, AuxiliaryGraphRevisionProposalRecord)
    assert converted.revision_reason is reason
    assert converted.terminal_node_key == structure.terminal_node_key
    assert len(converted.nodes) == len(structure.nodes)
    for source, stored in zip(structure.nodes, converted.nodes, strict=True):
        assert stored.local_node_key == source.node_key
        assert stored.node_kind is source.node_kind
        assert stored.executor_kind is source.executor_kind
        assert stored.title == source.title
        assert stored.objective == source.objective
        assert stored.acceptance_criteria == source.acceptance_criteria
        assert stored.capability_profile_id == source.capability_profile_id
        assert stored.input_resource_aliases == source.input_resource_aliases
        assert stored.source_anchor_ids == source.source_anchor_ids
        assert stored.output_contract == source.output_contract
        assert stored.required is source.required
        assert stored.origin_node_alias == source.origin_node_alias
    assert len(converted.edges) == len(structure.edges)
    for source, stored in zip(structure.edges, converted.edges, strict=True):
        assert stored.dependency_node_key == source.source_node_key
        assert stored.consumer_node_key == source.target_node_key
        assert stored.required is source.required


@pytest.mark.parametrize(
    ("proposal", "expected_action", "host_current_revision"),
    (
        (
            _revision_proposal(
                disposition=(
                    AuxiliaryGraphRevisionProposalDisposition.CONTINUE_CURRENT
                ),
                expected_current_revision=4,
                structure=None,
                reason=None,
            ),
            AuxiliaryGraphArchitectAction.CONTINUE_CURRENT,
            4,
        ),
        (
            _revision_proposal(
                disposition=(
                    AuxiliaryGraphRevisionProposalDisposition.SUPERSEDE_AND_REBASE
                ),
                expected_current_revision=4,
                structure=None,
                reason=None,
            ),
            AuxiliaryGraphArchitectAction.SUPERSEDE_AND_REBASE,
            4,
        ),
        (
            _revision_proposal(
                disposition=AuxiliaryGraphRevisionProposalDisposition.TERMINAL_FAIL,
                expected_current_revision=4,
                structure=None,
                reason=None,
                failure_reason="The frozen authority cannot support a legal graph.",
            ),
            AuxiliaryGraphArchitectAction.TERMINAL_FAIL,
            4,
        ),
        (
            _revision_proposal(
                disposition=AuxiliaryGraphRevisionProposalDisposition.CREATE_REVISION,
                expected_current_revision=None,
                structure=_structure(with_origins=False),
                reason=AuxiliaryGraphRevisionReason.INITIAL,
                question="May the blocking gap be resolved by a user answer?",
            ),
            AuxiliaryGraphArchitectAction.REQUEST_USER_INPUT,
            None,
        ),
    ),
)
def test_non_structural_architect_actions_are_never_committable(
    proposal: AuxiliaryGraphRevisionProposal,
    expected_action: AuxiliaryGraphArchitectAction,
    host_current_revision: int | None,
) -> None:
    decision = _decision(proposal)
    assert decision.action is expected_action

    with pytest.raises(AuxiliaryArchitectAdapterError, match="not committable"):
        auxiliary_graph_revision_proposal_from_architect_decision(
            decision,
            expected_current_auxiliary_graph_revision=host_current_revision,
        )


@pytest.mark.parametrize(
    ("decision", "host_current_revision"),
    (
        (
            _decision(
                _revision_proposal(
                    disposition=(
                        AuxiliaryGraphRevisionProposalDisposition.CREATE_REVISION
                    ),
                    expected_current_revision=None,
                    structure=_structure(with_origins=False),
                    reason=AuxiliaryGraphRevisionReason.INITIAL,
                )
            ),
            1,
        ),
        (
            _decision(
                _revision_proposal(
                    disposition=(
                        AuxiliaryGraphRevisionProposalDisposition.REVISE_REVISION
                    ),
                    expected_current_revision=7,
                    structure=_structure(with_origins=True),
                    reason=AuxiliaryGraphRevisionReason.MANUAL_REPLAN,
                )
            ),
            6,
        ),
    ),
)
def test_conversion_rejects_a_stale_or_wrong_host_current_revision(
    decision: AuxiliaryGraphArchitectDecision,
    host_current_revision: int | None,
) -> None:
    with pytest.raises(AuxiliaryArchitectAdapterError, match="current revision"):
        auxiliary_graph_revision_proposal_from_architect_decision(
            decision,
            expected_current_auxiliary_graph_revision=host_current_revision,
        )


def test_conversion_enforces_the_host_required_revision_reason() -> None:
    decision = _decision(
        _revision_proposal(
            disposition=AuxiliaryGraphRevisionProposalDisposition.REVISE_REVISION,
            expected_current_revision=3,
            structure=_structure(with_origins=True),
            reason=AuxiliaryGraphRevisionReason.VERIFICATION_FAILED,
        )
    )

    converted = auxiliary_graph_revision_proposal_from_architect_decision(
        decision,
        expected_current_auxiliary_graph_revision=3,
        required_revision_reason=(
            AuxiliaryGraphRevisionReason.VERIFICATION_FAILED
        ),
    )
    assert (
        converted.revision_reason
        is AuxiliaryGraphRevisionReason.VERIFICATION_FAILED
    )

    with pytest.raises(AuxiliaryArchitectAdapterError, match="required reason"):
        auxiliary_graph_revision_proposal_from_architect_decision(
            decision,
            expected_current_auxiliary_graph_revision=3,
            required_revision_reason=AuxiliaryGraphRevisionReason.EVIDENCE_CHANGED,
        )


def test_conversion_revalidates_revision_reason_instead_of_trusting_model_copy() -> None:
    valid = _revision_proposal(
        disposition=AuxiliaryGraphRevisionProposalDisposition.REVISE_REVISION,
        expected_current_revision=3,
        structure=_structure(with_origins=True),
        reason=AuxiliaryGraphRevisionReason.AUTHORITY_CHANGED,
    )
    invalid = valid.model_copy(
        update={"revision_reason": AuxiliaryGraphRevisionReason.INITIAL}
    )
    decision = _decision(valid).model_copy(update={"proposal": invalid})

    with pytest.raises(AuxiliaryArchitectAdapterError, match="contract validation"):
        auxiliary_graph_revision_proposal_from_architect_decision(
            decision,
            expected_current_auxiliary_graph_revision=3,
        )


def test_terminal_only_bootstrap_is_host_owned_initial_and_origin_free() -> None:
    acceptance = _acceptance(
        "bootstrap_task_graph",
        anchors=("auth_primary",),
    )

    proposal = build_terminal_only_auxiliary_graph_bootstrap_proposal(
        terminal_node_key="bootstrap_terminal",
        title="Bootstrap the terminal planning authority",
        objective="Establish durable authority before invoking the Architect.",
        source_anchor_ids=("auth_primary",),
        acceptance_criteria=(acceptance,),
        input_resource_aliases=("artifact_context",),
    )

    assert proposal.revision_reason is AuxiliaryGraphRevisionReason.INITIAL
    assert proposal.terminal_node_key == "bootstrap_terminal"
    assert proposal.edges == ()
    assert len(proposal.nodes) == 1
    terminal = proposal.nodes[0]
    assert terminal.local_node_key == "bootstrap_terminal"
    assert terminal.node_kind is AuxiliaryNodeKind.SYNTHESIZE
    assert terminal.executor_kind is AuxiliaryNodeExecutorKind.TERMINAL_PLANNER
    assert terminal.capability_profile_id is None
    assert terminal.output_contract == "task_graph_revision_proposal_v2"
    assert terminal.required is True
    assert terminal.origin_node_alias is None
    assert terminal.source_anchor_ids == ("auth_primary",)
    assert terminal.acceptance_criteria == (acceptance,)
    assert terminal.input_resource_aliases == ("artifact_context",)


def test_terminal_only_bootstrap_rejects_acceptance_authority_not_on_node() -> None:
    with pytest.raises(AuxiliaryArchitectAdapterError, match="bootstrap"):
        build_terminal_only_auxiliary_graph_bootstrap_proposal(
            terminal_node_key="bootstrap_terminal",
            title="Bootstrap the terminal planning authority",
            objective="Establish durable authority before invoking the Architect.",
            source_anchor_ids=("auth_primary",),
            acceptance_criteria=(
                _acceptance("wrong_authority", anchors=("other_authority",)),
            ),
        )
