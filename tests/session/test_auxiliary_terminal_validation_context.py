from __future__ import annotations

import json

import pytest

from personagraph.l2.task_graph import InSessionTaskGraphRevisionProposal
from personagraph.l2.task_graph.validation import (
    bind_used_evidence_to_required_acceptance_coverage,
    validate_insession_task_graph_revision,
)
from personagraph.l2.auxiliary_execution.driver import (
    canonical_auxiliary_graph_driver_state_guard,
)
from personagraph.l2.auxiliary_execution.work_run.controller import (
    AuxiliaryWorkRunStatus,
    AuxiliaryWorkRunRequest,
    derive_auxiliary_work_run_ids,
    run_auxiliary_model_node,
)
from personagraph.l2.planning.invocation_contracts import (
    FrozenPlanningContextArtifactBinding,
)
from personagraph.l2.planning.resource_perception import (
    FrozenPlanningResource,
    PlanningResourceCoverage,
    PlanningResourceEvidenceKind,
    PlanningResourceEvidenceUnit,
    PlanningResourceFormat,
    PlanningResourceGapReason,
    PlanningResourcePerceptionRequest,
    PlanningResourceReadOutcome,
    PlanningResourceReadRequest,
    freeze_planning_resource_perception_invocation,
    run_planning_resource_perception,
)
from personagraph.l2.auxiliary_graph import PlanningObservationStatus
from personagraph.session.l2_store import terminal as terminal_store
from personagraph.session.l2_store import auxiliary_graph as auxiliary_graph_store
from personagraph.session.l2_store import planning as planning_store
from tests.runtime.test_auxiliary_work_run_controller import (
    _PassVerifier,
    _ReplyProvider,
    _profile,
)
from tests.session.test_planning_artifact_seal_persistence import (
    _primitive,
    _seed_graph,
)


def _seal_resource_host(
    *,
    resource_format: PlanningResourceFormat,
    evidence_kind: PlanningResourceEvidenceKind,
    status: PlanningObservationStatus = PlanningObservationStatus.SUCCESS,
):
    session_id, turn_id, task_id = _seed_graph()
    frontier = auxiliary_graph_store.project_auxiliary_graph_execution_frontier(
        session_id=session_id,
        turn_id=turn_id,
        insession_task_id=task_id,
    )
    candidate = frontier.ready_fresh[0]
    prefix = resource_format.value
    binding = FrozenPlanningContextArtifactBinding(
        session_id=session_id,
        task_id=task_id,
        auxiliary_graph_id=frontier.auxiliary_graph_id,
        goal_id=frontier.goal_id,
        producer_auxiliary_node=candidate.subject,
        primitive_call_id=f"terminal-{prefix}-primitive",
        artifact_id=f"terminal-{prefix}-artifact",
        verification_receipt_id=f"terminal-{prefix}-verification",
        authority_snapshot_id=frontier.authority_snapshot_id,
        scope_snapshot_sha256="a" * 64,
        alias_prefix=prefix,
        artifact_alias=f"{prefix}_artifact",
        producer_node_alias="observe",
    )
    request = PlanningResourcePerceptionRequest(
        binding=binding,
        read_request=PlanningResourceReadRequest(
            resource=FrozenPlanningResource(
                session_id=session_id,
                resource_alias=f"{prefix}_01",
                resource_id=f"mounted-{prefix}-01",
                resource_version="version-1",
                content_sha256="b" * 64,
                coverage=PlanningResourceCoverage.COMPLETE,
                resource_format=resource_format,
                media_type=(
                    "image/png"
                    if resource_format is PlanningResourceFormat.PNG
                    else "application/pdf"
                ),
                file_extension=f".{resource_format.value}",
            )
        ),
    )
    invocation = freeze_planning_resource_perception_invocation(
        request,
        invocation_turn_id=turn_id,
        expected_task_state_version=frontier.task_state_version,
        expected_node_state_version=candidate.node_state_version,
        expected_control_state_version=frontier.control_state_version,
        expected_goal_state_version=frontier.goal_state_version,
        expected_revision_state_version=frontier.revision_state_version,
        expected_budget_state_version=frontier.budget_state_version,
        authority_snapshot_sha256=frontier.authority_snapshot_sha256,
        structure_sha256=frontier.structure_sha256,
        budget_snapshot_sha256=frontier.budget_snapshot_sha256,
    )
    planning_store.reserve_planning_primitive_invocation(invocation=invocation)

    class _Port:
        def read_frozen_resource(self, read_request):
            assert read_request == request.read_request
            return PlanningResourceReadOutcome(
                status=status,
                observed_resource_version="version-1",
                observed_content_sha256="b" * 64,
                observed_coverage=(
                    PlanningResourceCoverage.PARTIAL
                    if status is PlanningObservationStatus.PARTIAL
                    else PlanningResourceCoverage.COMPLETE
                ),
                evidence=(
                    PlanningResourceEvidenceUnit(
                        source_unit_id=f"{prefix}-unit-1",
                        statement=f"Verified {prefix.upper()} evidence.",
                        locator=f"{prefix} unit 1",
                        content_sha256="c" * 64,
                        evidence_kind=evidence_kind,
                        disclosure_receipt_id=(
                            "visual-disclosure-1"
                            if evidence_kind
                            is PlanningResourceEvidenceKind.VISUAL_OBSERVATION
                            else None
                        ),
                    ),
                ),
                gap_reasons=(
                    (PlanningResourceGapReason.INCOMPLETE_COVERAGE,)
                    if status is PlanningObservationStatus.PARTIAL
                    else ()
                ),
            )

    result = run_planning_resource_perception(request, read_port=_Port())
    command = planning_store.SealAuxiliaryHostPrimitiveResultCommand(
        apply_id=f"terminal-{prefix}-seal",
        session_id=session_id,
        invocation_turn_id=turn_id,
        task_id=task_id,
        auxiliary_graph_id=frontier.auxiliary_graph_id,
        goal_id=frontier.goal_id,
        auxiliary_graph_revision=frontier.auxiliary_graph_revision,
        auxiliary_node_id=candidate.subject.node_id,
        node_revision=candidate.subject.node_revision,
        primitive_kind="resource_perception",
        primitive_call_id=binding.primitive_call_id,
        expected_artifact_id=binding.artifact_id,
        expected_verification_receipt_id=binding.verification_receipt_id,
        expected_scope_snapshot_sha256=binding.scope_snapshot_sha256,
        expected_base_task_graph_revision=frontier.base_task_graph_revision,
        expected_task_state_version=invocation.expected_task_state_version,
        expected_node_state_version=invocation.expected_node_state_version,
        expected_control_state_version=invocation.expected_control_state_version,
        expected_goal_state_version=invocation.expected_goal_state_version,
        expected_revision_state_version=invocation.expected_revision_state_version,
        expected_budget_state_version=invocation.expected_budget_state_version,
        expected_authority_snapshot_id=frontier.authority_snapshot_id,
        expected_authority_snapshot_sha256=invocation.authority_snapshot_sha256,
        expected_structure_sha256=invocation.structure_sha256,
        expected_budget_snapshot_sha256=invocation.budget_snapshot_sha256,
        logical_request_json=invocation.logical_request_json,
        logical_request_sha256=invocation.logical_request_sha256,
        state_guard_sha256=invocation.state_guard_sha256,
    )
    planning_store.seal_auxiliary_host_primitive_result(command=command, result=result)
    return session_id, turn_id, task_id, result


def test_terminal_validation_context_preserves_gap_role_and_blocking() -> None:
    session_id, turn_id, task_id, result = _seal_resource_host(
        resource_format=PlanningResourceFormat.PDF,
        evidence_kind=PlanningResourceEvidenceKind.DOCUMENT_TEXT,
        status=PlanningObservationStatus.PARTIAL,
    )

    context = terminal_store.build_auxiliary_terminal_task_graph_validation_context(
        session_id=session_id,
        invocation_turn_id=turn_id,
        task_id=task_id,
    )

    gap = result.prompt_inputs.context_artifact.gaps[0]
    anchor = next(item for item in context.source_anchors if item.anchor_id == gap.gap_alias)
    assert anchor.source_kind == "gap"
    assert anchor.gap_blocking is gap.blocking
    assert gap.blocking is True
    assert anchor.model_dump(mode="json")["gap_blocking"] is True
    assert "gap_blocking" not in context.source_anchors[0].model_dump(mode="json")


def _proposal(*source_aliases: str) -> InSessionTaskGraphRevisionProposal:
    return InSessionTaskGraphRevisionProposal.model_validate(
        {
            "root": {
                "root_key": "root",
                "nodes": [
                    {
                        "node_key": "root",
                        "node_kind": "root",
                        "title": "执行计划",
                        "objective": "按已核验材料交付执行计划",
                        "source_anchor_ids": [
                            "task_creation_source",
                            *source_aliases,
                        ],
                        "acceptance_criteria": [
                            {
                                "acceptance_id": "source_understood",
                                "criterion": "计划引用的材料可核验",
                                "source_anchor_ids": [
                                    "task_creation_source",
                                    *source_aliases,
                                ],
                            }
                        ],
                    }
                ],
            }
        }
    )


def test_terminal_validation_context_projects_verified_host_source_cards(
    tmp_path,
) -> None:
    session_id, turn_id, task_id, command, result = _primitive(tmp_path)
    planning_store.seal_auxiliary_host_primitive_result(command=command, result=result)

    context = terminal_store.build_auxiliary_terminal_task_graph_validation_context(
        session_id=session_id,
        invocation_turn_id=turn_id,
        task_id=task_id,
    )

    assert context.source_anchors[0].anchor_id == "task_creation_source"
    assert tuple(anchor.anchor_id for anchor in context.source_anchors[1:]) == tuple(
        card.alias for card in result.prompt_inputs.source_cards
    )


@pytest.mark.parametrize(
    ("resource_format", "evidence_kind", "expected_source_kind"),
    (
        (
            PlanningResourceFormat.PDF,
            PlanningResourceEvidenceKind.DOCUMENT_TEXT,
            "retrieved_document",
        ),
        (
            PlanningResourceFormat.PNG,
            PlanningResourceEvidenceKind.VISUAL_OBSERVATION,
            "attachment",
        ),
    ),
)
def test_pdf_and_png_aliases_pass_terminal_work_run_and_store_preflight(
    resource_format,
    evidence_kind,
    expected_source_kind,
) -> None:
    session_id, turn_id, task_id, primitive_result = _seal_resource_host(
        resource_format=resource_format,
        evidence_kind=evidence_kind,
    )
    context = terminal_store.build_auxiliary_terminal_task_graph_validation_context(
        session_id=session_id,
        invocation_turn_id=turn_id,
        task_id=task_id,
    )
    card_alias = primitive_result.prompt_inputs.source_cards[0].alias
    source = next(item for item in context.source_anchors if item.anchor_id == card_alias)
    assert source.source_kind == expected_source_kind
    auxiliary_graph_store.require_auxiliary_graph_commit_context_valid(
        session_id=session_id,
        invocation_turn_id=turn_id,
        task_id=task_id,
        context=context,
    )

    proposal = _proposal(card_alias)
    uncovered_payload = proposal.model_dump(mode="json")
    uncovered_payload["root"]["nodes"][0]["acceptance_criteria"][0][
        "source_anchor_ids"
    ] = ["task_creation_source"]
    uncovered = InSessionTaskGraphRevisionProposal.model_validate(
        uncovered_payload
    )
    tightened = bind_used_evidence_to_required_acceptance_coverage(
        uncovered,
        context=context,
    )
    assert card_alias in tightened.required_anchor_ids
    rejected = validate_insession_task_graph_revision(
        uncovered,
        context=tightened,
    )
    assert rejected.status == "rejected"
    assert "unmapped_required_anchor" in {
        code.value for code in rejected.error_codes
    }

    reply = json.dumps(
        {
            "acceptance_updates": [
                {
                    "acceptance_id": "source_understood",
                    "model_claimed_satisfied": True,
                }
            ],
            "action": {
                "kind": "submit_task_graph",
                "proposal": proposal.model_dump(mode="json"),
            },
        },
        ensure_ascii=False,
    )
    frontier = auxiliary_graph_store.project_auxiliary_graph_execution_frontier(
        session_id=session_id,
        turn_id=turn_id,
        insession_task_id=task_id,
    )
    terminal = frontier.ready_fresh[0]
    result = run_auxiliary_model_node(
        AuxiliaryWorkRunRequest(
            session_id=session_id,
            turn_id=turn_id,
            subject=terminal.subject,
            executor_kind="terminal_planner",
            initial_driver_state_guard_sha256=(
                canonical_auxiliary_graph_driver_state_guard(frontier)
            ),
            id_plan=derive_auxiliary_work_run_ids(
                session_id=session_id,
                subject=terminal.subject,
            ),
            task_graph_validation_context=context,
        ),
        profile=_profile(),
        capability_catalogs={},
        attempt_provider=_ReplyProvider([reply]),
        verification_provider=_PassVerifier(),
        emit=lambda _event: None,
        monotonic_clock=iter(range(1, 100)).__next__,
    )
    assert result.status is AuxiliaryWorkRunStatus.COMPLETED


def test_forged_excerpt_and_foreign_artifact_alias_are_rejected(tmp_path) -> None:
    session_id, turn_id, task_id, command, result = _primitive(tmp_path)
    planning_store.seal_auxiliary_host_primitive_result(command=command, result=result)
    context = terminal_store.build_auxiliary_terminal_task_graph_validation_context(
        session_id=session_id,
        invocation_turn_id=turn_id,
        task_id=task_id,
    )
    evidence = context.source_anchors[1]
    forged = context.model_copy(
        update={
            "source_anchors": (
                context.source_anchors[0],
                evidence.model_copy(update={"excerpt": "forged excerpt"}),
            )
        }
    )
    with pytest.raises(auxiliary_graph_store.AuxiliaryGraphPersistenceError):
        auxiliary_graph_store.require_auxiliary_graph_commit_context_valid(
            session_id=session_id,
            invocation_turn_id=turn_id,
            task_id=task_id,
            context=forged,
        )

    foreign = context.model_copy(
        update={
            "source_anchors": (
                *context.source_anchors,
                evidence.model_copy(update={"anchor_id": "foreign_artifact_obs"}),
            )
        }
    )
    with pytest.raises(auxiliary_graph_store.AuxiliaryGraphPersistenceError):
        auxiliary_graph_store.require_auxiliary_graph_commit_context_valid(
            session_id=session_id,
            invocation_turn_id=turn_id,
            task_id=task_id,
            context=foreign,
        )


def test_unfinished_host_source_card_is_not_authority(tmp_path) -> None:
    session_id, turn_id, task_id, _command, result = _primitive(tmp_path)
    context = terminal_store.build_auxiliary_terminal_task_graph_validation_context(
        session_id=session_id,
        invocation_turn_id=turn_id,
        task_id=task_id,
    )
    assert tuple(item.anchor_id for item in context.source_anchors) == (
        "task_creation_source",
    )
    unsealed = result.prompt_inputs.source_cards[0]
    forged = context.model_copy(
        update={
            "source_anchors": (
                *context.source_anchors,
                context.source_anchors[0].model_copy(
                    update={
                        "anchor_id": unsealed.alias,
                        "source_kind": "retrieved_document",
                        "start": 0,
                        "end": len(unsealed.excerpt),
                        "excerpt": unsealed.excerpt,
                    }
                ),
            )
        }
    )
    with pytest.raises(auxiliary_graph_store.AuxiliaryGraphPersistenceError):
        auxiliary_graph_store.require_auxiliary_graph_commit_context_valid(
            session_id=session_id,
            invocation_turn_id=turn_id,
            task_id=task_id,
            context=forged,
        )
