from __future__ import annotations

import pytest
from pydantic import ValidationError

from personagraph.l2.auxiliary_graph import (
    AuxiliaryGraphEdgeProposal,
    AuxiliaryGraphEdge,
    AuxiliaryGraphRevisionProposalDisposition,
    AuxiliaryGraphRevisionProposal,
    AuxiliaryGraphRevisionReason,
    AuxiliaryGraphRevision,
    AuxiliaryGraphStructureProposal,
    AuxiliaryNodeDefinition,
    AuxiliaryNodeExecutorKind,
    AuxiliaryNodeKind,
    AuxiliaryNodeProposal,
    AuxiliaryPlanningGoal,
    PlanningAuthorityAnchor,
    PlanningAuthorityClass,
    PlanningAuthorityOriginKind,
    PlanningAuthorityProjection,
    PlanningAuthoritySourceCard,
    PlanningAuthoritySourceKind,
    PlanningAuthoritySnapshot,
    PlanningCapabilityCatalogProjection,
    PlanningCapabilityDescriptor,
    PlanningCapabilityEffect,
    PlanningContextArtifact,
    PlanningContextArtifactProjection,
    PlanningContextConstraint,
    PlanningContextConstraintProjection,
    PlanningContextFact,
    PlanningContextFactProjection,
    PlanningContextGapProjection,
    PlanningEpisodeBudgetDisposition,
    PlanningEpisodeBudgetExtensionReceipt,
    PlanningEpisodeBudgetProfile,
    PlanningEpisodeBudgetUsage,
    PlanningEpisodeBudget,
    PlanningEvidenceRef,
    PlanningGoalConstraintPrompt,
    PlanningGoalPromptContext,
    PlanningObservationStatus,
    TaskGraphSemanticBaseNode,
    TaskGraphSemanticBaseSnapshot,
    TaskGraphSemanticDeliveryCoverage,
    TaskGraphSemanticFailureScope,
    TaskGraphSemanticLineageDisposition,
    TaskGraphSemanticLineageProjection,
    TaskGraphRevisionCandidate,
    TaskGraphSemanticReviewPolicy,
    TaskGraphSemanticVerificationDimension,
    TaskGraphSemanticVerificationBindingError,
    TaskGraphSemanticVerificationDisposition,
    TaskGraphSemanticVerificationItem,
    TaskGraphSemanticVerificationRequest,
    TaskGraphSemanticVerificationResult,
    TaskGraphSemanticVerificationVerdict,
    TaskGraphSemanticVerificationPromptPayload,
    assess_planning_episode_budget,
    required_task_graph_semantic_reviewer_count,
    validate_task_graph_semantic_verification_quorum,
    validate_task_graph_semantic_verification_result,
)
from personagraph.l2.task_graph import (
    InSessionTaskAcceptanceProposal,
    InSessionTaskGraphRevisionProposal,
    InSessionTaskNodeKind,
    InSessionTaskNodeProposal,
    InSessionTaskRootGraphProposal,
)
from personagraph.l2.work_run import AuxiliaryNodeSubject


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64


def _acceptance(
    acceptance_id: str = "accept_context",
    source_anchor_ids: tuple[str, ...] = ("src_01",),
) -> InSessionTaskAcceptanceProposal:
    return InSessionTaskAcceptanceProposal(
        acceptance_id=acceptance_id,
        criterion="The required planning output is present and verifiable.",
        source_anchor_ids=source_anchor_ids,
    )


def _materialized_nodes() -> tuple[AuxiliaryNodeDefinition, ...]:
    observe = AuxiliaryNodeDefinition.create(
        node_id="aux_node_observe",
        node_revision=1,
        ordinal=0,
        node_kind=AuxiliaryNodeKind.OBSERVE,
        executor_kind=AuxiliaryNodeExecutorKind.HOST_PRIMITIVE,
        title="Read the bounded document context",
        objective="Extract the facts needed to design the TaskGraph.",
        acceptance_criteria=(_acceptance(),),
        capability_profile_id="document_context_v1",
        input_resource_aliases=("res_01",),
        source_anchor_ids=("src_01",),
        output_contract="planning_context_artifact_v1",
    )
    terminal = AuxiliaryNodeDefinition.create(
        node_id="aux_node_terminal",
        node_revision=1,
        ordinal=1,
        node_kind=AuxiliaryNodeKind.SYNTHESIZE,
        executor_kind=AuxiliaryNodeExecutorKind.TERMINAL_PLANNER,
        title="Synthesize the formal TaskGraph",
        objective="Create the complete source-bound TaskGraph proposal.",
        acceptance_criteria=(_acceptance("accept_graph"),),
        capability_profile_id=None,
        input_resource_aliases=(),
        source_anchor_ids=("src_01",),
        output_contract="task_graph_revision_proposal_v2",
    )
    return observe, terminal


def _graph_revision() -> AuxiliaryGraphRevision:
    nodes = _materialized_nodes()
    return AuxiliaryGraphRevision.create(
        session_id="session_01",
        task_id="task_01",
        auxiliary_graph_id="aux_graph_01",
        goal_id="goal_01",
        auxiliary_graph_revision=1,
        parent_auxiliary_graph_revision=None,
        base_task_graph_revision=None,
        source_turn_id="turn_01",
        revision_reason=AuxiliaryGraphRevisionReason.INITIAL,
        authority_snapshot_id="authority_snapshot_01",
        authority_snapshot_sha256=SHA_A,
        terminal_node_id="aux_node_terminal",
        nodes=nodes,
        edges=(
            AuxiliaryGraphEdge(
                source_node_id="aux_node_observe",
                target_node_id="aux_node_terminal",
            ),
        ),
    )


def _authority_snapshot() -> PlanningAuthoritySnapshot:
    snapshot_id = "authority_snapshot_01"
    evidence = PlanningAuthorityAnchor(
        anchor_id="anchor_evidence",
        authority_snapshot_id=snapshot_id,
        projection_alias="obs_01",
        authority_class=PlanningAuthorityClass.EVIDENCE,
        origin_kind=PlanningAuthorityOriginKind.TOOL_RESULT,
        origin_id="tool_result_01",
        source_revision=1,
        content_sha256=SHA_A,
        item_ordinal=0,
        projection_sha256=SHA_B,
        freshness_binding_sha256=SHA_C,
    )
    authorization = PlanningAuthorityAnchor(
        anchor_id="anchor_authorization",
        authority_snapshot_id=snapshot_id,
        projection_alias="src_01",
        authority_class=PlanningAuthorityClass.AUTHORIZATION,
        origin_kind=PlanningAuthorityOriginKind.USER_INSTRUCTION_SPAN,
        origin_id="turn_01",
        content_sha256=SHA_B,
        item_ordinal=0,
        span_start=0,
        span_end=42,
        projection_sha256=SHA_C,
        freshness_binding_sha256=SHA_D,
    )
    return PlanningAuthoritySnapshot.create(
        authority_snapshot_id=snapshot_id,
        session_id="session_01",
        task_id="task_01",
        auxiliary_graph_id="aux_graph_01",
        goal_id="goal_01",
        source_turn_id="turn_01",
        anchors=(evidence, authorization),
    )


def _goal() -> AuxiliaryPlanningGoal:
    return AuxiliaryPlanningGoal(
        session_id="session_01",
        task_id="task_01",
        auxiliary_graph_id="aux_graph_01",
        goal_id="goal_01",
        base_task_graph_revision=None,
        target_task_graph_revision=1,
        creation_turn_id="turn_01",
        authorization_manifest_id="authorization_manifest_01",
        budget_ledger_id="budget_01",
    )


def _context_artifact() -> PlanningContextArtifact:
    return PlanningContextArtifact.create(
        artifact_id="context_artifact_01",
        session_id="session_01",
        task_id="task_01",
        auxiliary_graph_id="aux_graph_01",
        goal_id="goal_01",
        producer_auxiliary_node=AuxiliaryNodeSubject(
            task_id="task_01",
            auxiliary_graph_id="aux_graph_01",
            auxiliary_graph_revision=1,
            node_id="aux_node_observe",
            node_revision=1,
        ),
        producer_primitive_call_id="primitive_call_01",
        scope_snapshot_sha256=SHA_A,
        facts=(
            PlanningContextFact(
                fact_id="fact_01",
                statement="The source requests a document-grounded execution plan.",
                evidence_anchor_ids=("anchor_evidence",),
            ),
        ),
        constraints=(
            PlanningContextConstraint(
                constraint_id="constraint_01",
                statement="The result must remain within the requested Task scope.",
                authorization_anchor_ids=("anchor_authorization",),
            ),
        ),
        evidence_refs=(
            PlanningEvidenceRef(
                evidence_anchor_id="anchor_evidence",
                source_alias="obs_01",
                locator="document 1, page 2",
                content_sha256=SHA_A,
                source_revision=1,
                freshness_binding_sha256=SHA_C,
            ),
        ),
        freshness_manifest_sha256=SHA_B,
        verification_receipt_id="verification_receipt_01",
        verification_receipt_sha256=SHA_C,
    )


def _task_graph_proposal() -> InSessionTaskGraphRevisionProposal:
    acceptance = _acceptance("accept_delivery")
    root = InSessionTaskNodeProposal(
        node_key="root",
        node_kind=InSessionTaskNodeKind.ROOT,
        title="Deliver the requested document analysis",
        objective="Produce the authorized, evidence-grounded deliverable.",
        source_anchor_ids=("obs_01", "src_01"),
        acceptance_criteria=(acceptance,),
    )
    return InSessionTaskGraphRevisionProposal(
        root=InSessionTaskRootGraphProposal(root_key="root", nodes=(root,))
    )


def _semantic_request() -> TaskGraphSemanticVerificationRequest:
    private_snapshot = _authority_snapshot()
    authority_projection = PlanningAuthorityProjection.create(
        authority_snapshot_id=private_snapshot.authority_snapshot_id,
        authority_snapshot_sha256=private_snapshot.snapshot_sha256,
        cards=(
            PlanningAuthoritySourceCard(
                alias="obs_01",
                authority_class=PlanningAuthorityClass.EVIDENCE,
                source_kind=PlanningAuthoritySourceKind.DOCUMENT,
                source_label="Mounted document, page 2",
                excerpt="The source requests a document-grounded execution plan.",
                projection_sha256=SHA_B,
            ),
            PlanningAuthoritySourceCard(
                alias="src_01",
                authority_class=PlanningAuthorityClass.AUTHORIZATION,
                source_kind=PlanningAuthoritySourceKind.USER_INSTRUCTION,
                source_label="Current user instruction",
                excerpt="Interpret the supplied document and produce the requested output.",
                projection_sha256=SHA_C,
            ),
        ),
    )
    private_artifact = _context_artifact()
    artifact_projection = PlanningContextArtifactProjection.create(
        artifact_alias="artifact_01",
        artifact_id=private_artifact.artifact_id,
        artifact_sha256=private_artifact.artifact_sha256,
        producer_node_alias="observe_01",
        facts=(
            PlanningContextFactProjection(
                fact_alias="fact_01",
                statement="The document contains the evidence required for planning.",
                evidence_aliases=("obs_01",),
            ),
        ),
        constraints=(
            PlanningContextConstraintProjection(
                constraint_alias="constraint_01",
                statement="Keep the work within the requested document-analysis scope.",
                authorization_aliases=("src_01",),
            ),
        ),
    )
    budget = PlanningEpisodeBudget.create(
        budget_ledger_id="budget_01",
        goal_id="goal_01",
    )
    prompt_payload = TaskGraphSemanticVerificationPromptPayload.create(
        goal=PlanningGoalPromptContext(
            goal_id="goal_01",
            objective="Interpret the supplied document correctly and produce the requested output.",
            desired_output="A source-grounded result that satisfies the explicit user request.",
            constraints=(
                PlanningGoalConstraintPrompt(
                    constraint_id="constraint_01",
                    statement="Do not expand beyond the authorized document-analysis task.",
                    authorization_aliases=("src_01",),
                ),
            ),
            authorization_aliases=("src_01",),
        ),
        authority=authority_projection,
        base_task_graph=None,
        context_artifacts=(artifact_projection,),
        capabilities=PlanningCapabilityCatalogProjection.create(
            capability_catalog_snapshot_id="capability_catalog_01",
            capability_catalog_snapshot_sha256=SHA_D,
            capabilities=(
                PlanningCapabilityDescriptor(
                    capability_alias="document_analysis",
                    label="Document analysis",
                    description="Read mounted documents and produce evidence-grounded analysis.",
                    available=True,
                    effect=PlanningCapabilityEffect.READ_ONLY,
                    supported_operations=("read", "synthesize"),
                    supported_resource_kinds=("pdf",),
                ),
            ),
        ),
        budget=budget,
        task_graph_proposal=_task_graph_proposal(),
    )
    return TaskGraphSemanticVerificationRequest.create(
        verification_request_id="semantic_request_01",
        logical_call_id="logical_call_01",
        verification_profile_id="semantic_verifier_v1",
        reviewer_ordinal=1,
        required_reviewer_count=1,
        goal=_goal(),
        auxiliary_graph_revision=1,
        auxiliary_graph_structure_sha256=_graph_revision().structure_sha256,
        prompt_payload=prompt_payload,
        review_policy=TaskGraphSemanticReviewPolicy.create(
            policy_source_sha256=SHA_A,
            distinct_document_count=1,
            has_visual_input=False,
            requires_protected_effect=False,
            modifies_executed_task_graph=False,
        ),
    )


def _result_items(
    *,
    failed_dimension: TaskGraphSemanticVerificationDimension | None = None,
    gap_aliases: tuple[str, ...] = (),
) -> tuple[TaskGraphSemanticVerificationItem, ...]:
    return tuple(
        TaskGraphSemanticVerificationItem(
            dimension=dimension,
            verdict=(
                TaskGraphSemanticVerificationVerdict.FAIL
                if dimension is failed_dimension
                else TaskGraphSemanticVerificationVerdict.PASS
            ),
            failure_scope=(
                TaskGraphSemanticFailureScope.TERMINAL_PROPOSAL
                if dimension is failed_dimension
                else None
            ),
            finding=f"The {dimension.value} check is resolved.",
            evidence_aliases=(
                ("obs_01",)
                if dimension
                is TaskGraphSemanticVerificationDimension.EVIDENCE_GROUNDING
                else ()
            ),
            gap_aliases=(
                gap_aliases
                if dimension
                is TaskGraphSemanticVerificationDimension.GAP_DISPOSITION
                else ()
            ),
        )
        for dimension in TaskGraphSemanticVerificationDimension
    )


def test_goal_binds_target_to_frozen_base_and_is_frozen() -> None:
    assert _goal().target_task_graph_revision == 1
    revised = _goal().model_copy(
        update={"base_task_graph_revision": 4, "target_task_graph_revision": 5}
    )
    validated = AuxiliaryPlanningGoal.model_validate(revised.model_dump())
    assert validated.target_task_graph_revision == 5

    with pytest.raises(ValidationError, match="one above"):
        AuxiliaryPlanningGoal.model_validate(
            revised.model_dump() | {"target_task_graph_revision": 6}
        )
    with pytest.raises(ValidationError, match="frozen"):
        _goal().status = "failed"  # type: ignore[misc]


def test_materialized_revision_is_hashed_and_rejects_invalid_topology() -> None:
    revision = _graph_revision()
    assert len(revision.structure_sha256) == 64
    assert revision.nodes[-1].executor_kind is AuxiliaryNodeExecutorKind.TERMINAL_PLANNER

    with pytest.raises(ValidationError, match="structure hash"):
        AuxiliaryGraphRevision.model_validate(
            revision.model_dump() | {"structure_sha256": SHA_D}
        )
    with pytest.raises(ValidationError, match="sink|acyclic"):
        AuxiliaryGraphRevision.create(
            **(
                revision.model_dump(exclude={"structure_sha256", "edges"})
                | {
                    "edges": (
                        AuxiliaryGraphEdge(
                            source_node_id="aux_node_observe",
                            target_node_id="aux_node_terminal",
                        ),
                        AuxiliaryGraphEdge(
                            source_node_id="aux_node_terminal",
                            target_node_id="aux_node_observe",
                        ),
                    )
                }
            )
        )


def test_architect_proposal_uses_local_keys_and_disposition_shape() -> None:
    observe, terminal = _materialized_nodes()
    proposal_nodes = tuple(
        AuxiliaryNodeProposal(
            node_key="observe" if item.ordinal == 0 else "terminal",
            node_kind=item.node_kind,
            executor_kind=item.executor_kind,
            title=item.title,
            objective=item.objective,
            acceptance_criteria=item.acceptance_criteria,
            capability_profile_id=item.capability_profile_id,
            input_resource_aliases=item.input_resource_aliases,
            source_anchor_ids=item.source_anchor_ids,
            output_contract=item.output_contract,
        )
        for item in (observe, terminal)
    )
    structure = AuxiliaryGraphStructureProposal(
        terminal_node_key="terminal",
        nodes=proposal_nodes,
        edges=(
            AuxiliaryGraphEdgeProposal(
                source_node_key="observe",
                target_node_key="terminal",
            ),
        ),
    )
    decision = AuxiliaryGraphRevisionProposal(
        disposition=AuxiliaryGraphRevisionProposalDisposition.CREATE_REVISION,
        revision_reason=AuxiliaryGraphRevisionReason.INITIAL,
        structure=structure,
        explanation="The source must be observed before formal synthesis.",
    )
    assert decision.structure is structure

    with pytest.raises(ValidationError, match="complete DAG"):
        AuxiliaryGraphRevisionProposal(
            disposition=AuxiliaryGraphRevisionProposalDisposition.CREATE_REVISION,
            revision_reason=AuxiliaryGraphRevisionReason.INITIAL,
            explanation="A missing structure cannot be committed.",
        )
    with pytest.raises(ValidationError, match="extra_forbidden"):
        AuxiliaryGraphRevisionProposal.model_validate(
            decision.model_dump() | {"durable_node_id": "model_chosen_id"}
        )


def test_only_terminal_planner_may_declare_the_formal_task_graph_output() -> None:
    observe, _terminal = _materialized_nodes()
    forged_definition = observe.model_dump(exclude={"semantic_fingerprint"}) | {
        "output_contract": "task_graph_revision_proposal_v2"
    }
    with pytest.raises(ValidationError, match="only the terminal planner"):
        AuxiliaryNodeDefinition.create(**forged_definition)

    with pytest.raises(ValidationError, match="only the terminal planner"):
        AuxiliaryNodeProposal(
            node_key="forged_planner",
            node_kind=AuxiliaryNodeKind.ANALYZE,
            executor_kind=AuxiliaryNodeExecutorKind.MODEL_WORK_RUN,
            title="Bypass the terminal planner",
            objective="Attempt to emit the formal TaskGraph from an investigation node.",
            acceptance_criteria=(_acceptance(),),
            capability_profile_id="document_context_v1",
            source_anchor_ids=("src_01",),
            output_contract="task_graph_revision_proposal_v2",
        )


def test_host_primitive_requires_planning_context_output_contract() -> None:
    with pytest.raises(ValidationError, match="Host primitive.*planning_context"):
        AuxiliaryNodeProposal(
            node_key="read_document",
            node_kind=AuxiliaryNodeKind.OBSERVE,
            executor_kind=AuxiliaryNodeExecutorKind.HOST_PRIMITIVE,
            title="Read the mounted document",
            objective="Produce bounded evidence for downstream planning.",
            acceptance_criteria=(_acceptance(),),
            capability_profile_id="mounted_document_read",
            input_resource_aliases=("mounted_document_01",),
            source_anchor_ids=("src_01",),
            output_contract="verified_context_artifact_v1",
        )


@pytest.mark.parametrize(
    "legacy_action",
    ("call_tools", "write_output_window", "submit_output_window", "submit_task_graph"),
)
def test_legacy_attempt_action_cannot_masquerade_as_an_output_contract(
    legacy_action: str,
) -> None:
    with pytest.raises(ValidationError, match="legacy Attempt actions"):
        AuxiliaryNodeProposal(
            node_key="forged_action",
            node_kind=AuxiliaryNodeKind.OBSERVE,
            executor_kind=AuxiliaryNodeExecutorKind.MODEL_WORK_RUN,
            title="Attempt a legacy action",
            objective="Try to route an old Attempt action through a V2 node contract.",
            acceptance_criteria=(_acceptance(),),
            capability_profile_id="document_context_v1",
            source_anchor_ids=("src_01",),
            output_contract=legacy_action,
        )


def test_authority_and_context_artifact_are_exactly_bound() -> None:
    snapshot = _authority_snapshot()
    artifact = _context_artifact()
    assert snapshot.anchors[-1].authority_class is PlanningAuthorityClass.AUTHORIZATION
    assert artifact.facts[0].evidence_anchor_ids == ("anchor_evidence",)

    with pytest.raises(ValidationError, match="only user"):
        PlanningAuthorityAnchor(
            anchor_id="forged_authority",
            authority_snapshot_id="authority_snapshot_01",
            projection_alias="src_02",
            authority_class=PlanningAuthorityClass.AUTHORIZATION,
            origin_kind=PlanningAuthorityOriginKind.TOOL_RESULT,
            origin_id="tool_result_02",
            content_sha256=SHA_A,
            item_ordinal=0,
            projection_sha256=SHA_B,
            freshness_binding_sha256=SHA_C,
        )
    with pytest.raises(ValidationError, match="exact source span"):
        PlanningAuthorityAnchor(
            anchor_id="unlocated_user_authority",
            authority_snapshot_id="authority_snapshot_01",
            projection_alias="src_02",
            authority_class=PlanningAuthorityClass.AUTHORIZATION,
            origin_kind=PlanningAuthorityOriginKind.USER_INSTRUCTION_SPAN,
            origin_id="turn_01",
            content_sha256=SHA_A,
            item_ordinal=0,
            projection_sha256=SHA_B,
            freshness_binding_sha256=SHA_C,
        )
    with pytest.raises(ValidationError, match="source revision"):
        PlanningAuthorityAnchor(
            anchor_id="unversioned_prior_task_authority",
            authority_snapshot_id="authority_snapshot_01",
            projection_alias="src_03",
            authority_class=PlanningAuthorityClass.AUTHORIZATION,
            origin_kind=PlanningAuthorityOriginKind.PRIOR_TASK_STATE,
            origin_id="delivery_01",
            content_sha256=SHA_A,
            item_ordinal=0,
            projection_sha256=SHA_B,
            freshness_binding_sha256=SHA_C,
        )
    with pytest.raises(ValidationError, match="undeclared evidence"):
        PlanningContextArtifact.create(
            **(
                artifact.model_dump(exclude={"artifact_sha256", "facts"})
                | {
                    "facts": (
                        PlanningContextFact(
                            fact_id="fact_forged",
                            statement="This fact does not have declared evidence.",
                            evidence_anchor_ids=("anchor_missing",),
                        ),
                    )
                }
            )
        )


def test_planning_budget_defaults_soft_hard_and_append_only_extension() -> None:
    profile = PlanningEpisodeBudgetProfile()
    assert profile.hard_current_graph_nodes == 64
    assert profile.hard_current_graph_depth == 12
    at_structure_max = assess_planning_episode_budget(
        profile=profile,
        usage=PlanningEpisodeBudgetUsage(
            current_graph_nodes=64,
            current_graph_depth=12,
        ),
    )
    assert at_structure_max.disposition is PlanningEpisodeBudgetDisposition.WITHIN_LIMIT

    soft = assess_planning_episode_budget(
        profile=profile,
        usage=PlanningEpisodeBudgetUsage(auxiliary_graph_revisions=8),
    )
    assert soft.disposition is PlanningEpisodeBudgetDisposition.SOFT_LIMIT_REACHED
    hard = assess_planning_episode_budget(
        profile=profile,
        usage=PlanningEpisodeBudgetUsage(logical_model_calls=384),
    )
    assert hard.disposition is PlanningEpisodeBudgetDisposition.HARD_LIMIT_REACHED

    extended_profile = PlanningEpisodeBudgetProfile.model_validate(
        profile.model_dump() | {"hard_logical_model_calls": 400}
    )
    extension = PlanningEpisodeBudgetExtensionReceipt.create(
        extension_receipt_id="budget_extension_01",
        goal_id="goal_01",
        approved_turn_id="turn_02",
        authorization_anchor_id="anchor_authorization",
        reason="The user explicitly approved additional planning capacity.",
        profile_before=profile,
        profile_after=extended_profile,
    )
    budget = PlanningEpisodeBudget.create(
        budget_ledger_id="budget_01",
        goal_id="goal_01",
        usage=PlanningEpisodeBudgetUsage(logical_model_calls=300),
        extensions=(extension,),
    )
    assert budget.effective_profile.hard_logical_model_calls == 400
    assert budget.usage.logical_model_calls == 300

    reduced_profile = PlanningEpisodeBudgetProfile.model_validate(
        profile.model_dump() | {"hard_logical_model_calls": 383}
    )
    with pytest.raises(ValidationError, match="cannot reduce"):
        PlanningEpisodeBudgetExtensionReceipt.create(
            extension_receipt_id="budget_extension_bad",
            goal_id="goal_01",
            approved_turn_id="turn_02",
            authorization_anchor_id="anchor_authorization",
            reason="This invalid extension attempts to reset consumed capacity.",
            profile_before=profile,
            profile_after=reduced_profile,
        )


def test_semantic_verifier_request_binds_all_authority_and_host_derives_pass() -> None:
    request = _semantic_request()
    assert request.task_graph_proposal_sha256
    assert request.binding_sha256
    assert request.blocking_gap_aliases == ()
    prompt_json = request.to_prompt_payload().model_dump_json()
    assert request.to_prompt_payload().goal.objective.startswith("Interpret")
    assert "tool_result_01" not in prompt_json
    assert "document 1, page 2" not in prompt_json

    result = TaskGraphSemanticVerificationResult.create(
        verification_result_id="semantic_result_01",
        verification_request_id=request.verification_request_id,
        request_binding_sha256=request.binding_sha256,
        logical_call_id=request.logical_call_id,
        verification_profile_id=request.verification_profile_id,
        reviewer_ordinal=1,
        required_reviewer_count=1,
        items=_result_items(),
    )
    assert result.all_pass is True
    assert result.host_disposition is TaskGraphSemanticVerificationDisposition.PASS
    assert "overall_pass" not in TaskGraphSemanticVerificationResult.model_fields
    assert (
        validate_task_graph_semantic_verification_result(
            request=request,
            result=result,
        )
        is result
    )

    revised = TaskGraphSemanticVerificationResult.create(
        verification_result_id="semantic_result_02",
        verification_request_id=request.verification_request_id,
        request_binding_sha256=request.binding_sha256,
        logical_call_id=request.logical_call_id,
        verification_profile_id=request.verification_profile_id,
        reviewer_ordinal=1,
        required_reviewer_count=1,
        items=_result_items(
            failed_dimension=TaskGraphSemanticVerificationDimension.EDGE_DEPENDENCY_VALIDITY
        ),
    )
    assert revised.all_pass is False
    assert revised.host_disposition is TaskGraphSemanticVerificationDisposition.REVISE

    private_echo_items = list(_result_items())
    private_echo_items[0] = TaskGraphSemanticVerificationItem(
        dimension=private_echo_items[0].dimension,
        verdict=TaskGraphSemanticVerificationVerdict.PASS,
        failure_scope=None,
        finding="The private goal_01 binding must not be echoed by a reviewer.",
    )
    private_echo = TaskGraphSemanticVerificationResult.create(
        verification_result_id="semantic_result_private_echo",
        verification_request_id=request.verification_request_id,
        request_binding_sha256=request.binding_sha256,
        logical_call_id=request.logical_call_id,
        verification_profile_id=request.verification_profile_id,
        reviewer_ordinal=1,
        required_reviewer_count=1,
        items=tuple(private_echo_items),
    )
    with pytest.raises(TaskGraphSemanticVerificationBindingError, match="private binding"):
        validate_task_graph_semantic_verification_result(
            request=request,
            result=private_echo,
        )


def test_semantic_verifier_quorum_requires_independent_logical_calls() -> None:
    base = _semantic_request()
    quorum_policy = TaskGraphSemanticReviewPolicy.create(
        policy_source_sha256=SHA_B,
        distinct_document_count=2,
        has_visual_input=False,
        requires_protected_effect=False,
        modifies_executed_task_graph=False,
    )

    def request_for(
        *, ordinal: int, request_id: str, logical_call_id: str
    ) -> TaskGraphSemanticVerificationRequest:
        return TaskGraphSemanticVerificationRequest.create(
            verification_request_id=request_id,
            logical_call_id=logical_call_id,
            verification_profile_id=base.verification_profile_id,
            reviewer_ordinal=ordinal,
            required_reviewer_count=2,
            goal=base.goal,
            auxiliary_graph_revision=base.auxiliary_graph_revision,
            auxiliary_graph_structure_sha256=base.auxiliary_graph_structure_sha256,
            prompt_payload=base.prompt_payload,
            review_policy=quorum_policy,
        )

    first_request = request_for(
        ordinal=1,
        request_id="semantic_request_reviewer_01",
        logical_call_id="logical_call_reviewer_01",
    )
    second_request = request_for(
        ordinal=2,
        request_id="semantic_request_reviewer_02",
        logical_call_id="logical_call_reviewer_02",
    )

    def result_for(
        request: TaskGraphSemanticVerificationRequest,
        *,
        result_id: str,
    ) -> TaskGraphSemanticVerificationResult:
        return TaskGraphSemanticVerificationResult.create(
            verification_result_id=result_id,
            verification_request_id=request.verification_request_id,
            request_binding_sha256=request.binding_sha256,
            logical_call_id=request.logical_call_id,
            verification_profile_id=request.verification_profile_id,
            reviewer_ordinal=request.reviewer_ordinal,
            required_reviewer_count=request.required_reviewer_count,
            items=_result_items(),
        )

    first_result = result_for(first_request, result_id="semantic_result_reviewer_01")
    second_result = result_for(second_request, result_id="semantic_result_reviewer_02")
    assert validate_task_graph_semantic_verification_quorum(
        requests=(second_request, first_request),
        results=(second_result, first_result),
    ) == (first_result, second_result)

    reused_call_request = request_for(
        ordinal=2,
        request_id="semantic_request_reviewer_reused",
        logical_call_id=first_request.logical_call_id,
    )
    reused_call_result = result_for(
        reused_call_request,
        result_id="semantic_result_reviewer_reused",
    )
    with pytest.raises(TaskGraphSemanticVerificationBindingError, match="logical call"):
        validate_task_graph_semantic_verification_quorum(
            requests=(first_request, reused_call_request),
            results=(first_result, reused_call_result),
        )

    failed_result = TaskGraphSemanticVerificationResult.create(
        verification_result_id="semantic_result_reviewer_failed",
        verification_request_id=second_request.verification_request_id,
        request_binding_sha256=second_request.binding_sha256,
        logical_call_id=second_request.logical_call_id,
        verification_profile_id=second_request.verification_profile_id,
        reviewer_ordinal=second_request.reviewer_ordinal,
        required_reviewer_count=second_request.required_reviewer_count,
        items=_result_items(
            failed_dimension=TaskGraphSemanticVerificationDimension.GOAL_COVERAGE
        ),
    )
    assert validate_task_graph_semantic_verification_quorum(
        requests=(first_request, second_request),
        results=(first_result, failed_result),
    ) == (first_result, failed_result)

    drifted_proposal = _task_graph_proposal()
    drifted_proposal = InSessionTaskGraphRevisionProposal.model_validate(
        {
            "root": drifted_proposal.root.model_dump()
            | {
                "nodes": [
                    drifted_proposal.root.nodes[0].model_dump()
                    | {"objective": "Produce a drifted output."}
                ]
            }
        }
    )
    drifted_prompt = TaskGraphSemanticVerificationPromptPayload.create(
        goal=base.prompt_payload.goal,
        authority=base.prompt_payload.authority,
        base_task_graph=base.prompt_payload.base_task_graph,
        context_artifacts=base.prompt_payload.context_artifacts,
        capabilities=base.prompt_payload.capabilities,
        budget=base.prompt_payload.budget,
        task_graph_proposal=drifted_proposal,
        lineage=base.prompt_payload.lineage,
    )
    drifted_request = TaskGraphSemanticVerificationRequest.create(
        verification_request_id="semantic_request_reviewer_drifted",
        logical_call_id="logical_call_reviewer_drifted",
        verification_profile_id=base.verification_profile_id,
        reviewer_ordinal=2,
        required_reviewer_count=2,
        goal=base.goal,
        auxiliary_graph_revision=base.auxiliary_graph_revision,
        auxiliary_graph_structure_sha256=base.auxiliary_graph_structure_sha256,
        prompt_payload=drifted_prompt,
        review_policy=quorum_policy,
    )
    drifted_result = result_for(
        drifted_request,
        result_id="semantic_result_reviewer_drifted",
    )
    with pytest.raises(TaskGraphSemanticVerificationBindingError, match="frozen proposal"):
        validate_task_graph_semantic_verification_quorum(
            requests=(first_request, drifted_request),
            results=(first_result, drifted_result),
        )


def test_semantic_verifier_large_graph_cannot_reduce_reviewer_quorum() -> None:
    base = _semantic_request()
    root = base.task_graph_proposal.root.nodes[0]
    children = tuple(
        InSessionTaskNodeProposal(
            node_key=f"node_{ordinal:02d}",
            node_kind=InSessionTaskNodeKind.SUBTASK,
            parent_node_key="root",
            title=f"Execute bounded subtask {ordinal}",
            objective=f"Produce independently verifiable output {ordinal}.",
            source_anchor_ids=("obs_01", "src_01"),
            acceptance_criteria=(
                _acceptance(f"accept_{ordinal:02d}"),
            ),
        )
        for ordinal in range(1, 13)
    )
    proposal = InSessionTaskGraphRevisionProposal(
        root=InSessionTaskRootGraphProposal(
            root_key="root",
            nodes=(root, *children),
        )
    )
    prompt = TaskGraphSemanticVerificationPromptPayload.create(
        goal=base.prompt_payload.goal,
        authority=base.prompt_payload.authority,
        base_task_graph=None,
        context_artifacts=base.prompt_payload.context_artifacts,
        capabilities=base.prompt_payload.capabilities,
        budget=base.prompt_payload.budget,
        task_graph_proposal=proposal,
    )
    assert required_task_graph_semantic_reviewer_count(
        prompt_payload=prompt,
        review_policy=base.review_policy,
    ) == 2
    with pytest.raises(ValidationError, match="Host-derived policy"):
        TaskGraphSemanticVerificationRequest.create(
            verification_request_id="semantic_request_large_graph",
            logical_call_id="logical_call_large_graph",
            verification_profile_id=base.verification_profile_id,
            reviewer_ordinal=1,
            required_reviewer_count=1,
            goal=base.goal,
            auxiliary_graph_revision=base.auxiliary_graph_revision,
            auxiliary_graph_structure_sha256=base.auxiliary_graph_structure_sha256,
            prompt_payload=prompt,
            review_policy=base.review_policy,
        )


def test_base_task_graph_prompt_snapshot_carries_completed_authority() -> None:
    root = TaskGraphSemanticBaseNode(
        node_alias="base_root",
        node_revision=3,
        node_kind=InSessionTaskNodeKind.ROOT,
        title="Existing document analysis",
        objective="Preserve the already completed evidence-grounded result.",
        source_anchor_aliases=("src_01",),
        acceptance_criteria=(_acceptance("accept_existing"),),
        status="completed",
        completed_delivery_summary="The existing node already delivered the requested overview.",
        completed_delivery_coverage=TaskGraphSemanticDeliveryCoverage.FULL,
        delivery_authority_aliases=("obs_01", "src_01"),
    )
    snapshot = TaskGraphSemanticBaseSnapshot.create(
        base_task_graph_revision=4,
        root_node_alias="base_root",
        nodes=(root,),
        source_snapshot_sha256=SHA_A,
    )
    assert snapshot.nodes[0].completed_delivery_summary is not None

    with pytest.raises(ValidationError, match="Delivery summary"):
        TaskGraphSemanticBaseNode.model_validate(
            root.model_dump() | {"completed_delivery_summary": None}
        )

    current_prompt = _semantic_request().prompt_payload
    revised_prompt = TaskGraphSemanticVerificationPromptPayload.create(
        goal=current_prompt.goal,
        authority=current_prompt.authority,
        base_task_graph=snapshot,
        context_artifacts=current_prompt.context_artifacts,
        capabilities=current_prompt.capabilities,
        budget=current_prompt.budget,
        task_graph_proposal=current_prompt.task_graph_proposal,
        lineage=(
            TaskGraphSemanticLineageProjection(
                proposal_node_key="root",
                disposition=TaskGraphSemanticLineageDisposition.REUSE,
                base_node_alias="base_root",
            ),
        ),
    )
    assert revised_prompt.lineage[0].base_node_alias == "base_root"

    revised_lineage_prompt = TaskGraphSemanticVerificationPromptPayload.create(
        goal=current_prompt.goal,
        authority=current_prompt.authority,
        base_task_graph=snapshot,
        context_artifacts=current_prompt.context_artifacts,
        capabilities=current_prompt.capabilities,
        budget=current_prompt.budget,
        task_graph_proposal=current_prompt.task_graph_proposal,
        lineage=(
            TaskGraphSemanticLineageProjection(
                proposal_node_key="root",
                disposition=TaskGraphSemanticLineageDisposition.REVISE,
                base_node_alias="base_root",
            ),
        ),
    )
    revised_goal = AuxiliaryPlanningGoal.model_validate(
        _goal().model_dump()
        | {
            "base_task_graph_revision": 4,
            "target_task_graph_revision": 5,
        }
    )
    with pytest.raises(ValidationError, match="base-graph modification"):
        TaskGraphSemanticVerificationRequest.create(
            verification_request_id="semantic_request_unsealed_base_revision",
            logical_call_id="logical_call_unsealed_base_revision",
            verification_profile_id="semantic_verifier_v1",
            reviewer_ordinal=1,
            required_reviewer_count=1,
            goal=revised_goal,
            auxiliary_graph_revision=2,
            auxiliary_graph_structure_sha256=SHA_D,
            prompt_payload=revised_lineage_prompt,
            review_policy=TaskGraphSemanticReviewPolicy.create(
                policy_source_sha256=SHA_A,
                distinct_document_count=1,
                has_visual_input=False,
                requires_protected_effect=False,
                modifies_executed_task_graph=False,
            ),
        )

    with pytest.raises(ValidationError, match="base-null"):
        TaskGraphSemanticVerificationPromptPayload.create(
            goal=current_prompt.goal,
            authority=current_prompt.authority,
            base_task_graph=None,
            context_artifacts=current_prompt.context_artifacts,
            capabilities=current_prompt.capabilities,
            budget=current_prompt.budget,
            task_graph_proposal=current_prompt.task_graph_proposal,
            lineage=revised_prompt.lineage,
        )

    partial_root = TaskGraphSemanticBaseNode.model_validate(
        root.model_dump()
        | {
            "completed_delivery_coverage": "partial",
            "completed_delivery_gap_aliases": ("gap_base_delivery",),
        }
    )
    partial_snapshot = TaskGraphSemanticBaseSnapshot.create(
        base_task_graph_revision=4,
        root_node_alias="base_root",
        nodes=(partial_root,),
        source_snapshot_sha256=SHA_A,
    )
    with pytest.raises(ValidationError, match="unprojected typed gap"):
        TaskGraphSemanticVerificationPromptPayload.create(
            goal=current_prompt.goal,
            authority=current_prompt.authority,
            base_task_graph=partial_snapshot,
            context_artifacts=current_prompt.context_artifacts,
            capabilities=current_prompt.capabilities,
            budget=current_prompt.budget,
            task_graph_proposal=current_prompt.task_graph_proposal,
            lineage=revised_prompt.lineage,
        )


def test_task_graph_revision_candidate_requires_full_ordered_unique_lineage() -> None:
    proposal = _task_graph_proposal()
    candidate = TaskGraphRevisionCandidate(
        proposal=proposal,
        lineage=(
            TaskGraphSemanticLineageProjection(
                proposal_node_key="root",
                disposition="reuse",
                base_node_alias="base_root",
            ),
        ),
    )
    assert candidate.proposal is proposal
    assert candidate.lineage[0].base_node_alias == "base_root"

    with pytest.raises(ValidationError, match="exactly cover proposal nodes"):
        TaskGraphRevisionCandidate(
            proposal=proposal,
            lineage=(
                TaskGraphSemanticLineageProjection(
                    proposal_node_key="other",
                    disposition="new",
                ),
            ),
        )

    child = InSessionTaskNodeProposal(
        node_key="child",
        node_kind=InSessionTaskNodeKind.SUBTASK,
        parent_node_key="root",
        title="Check the revised plan",
        objective="Verify the new execution step before delivery.",
        source_anchor_ids=("src_01",),
        acceptance_criteria=(_acceptance("accept_child"),),
    )
    two_node_proposal = InSessionTaskGraphRevisionProposal(
        root=InSessionTaskRootGraphProposal(
            root_key="root",
            nodes=(*proposal.root.nodes, child),
        )
    )
    with pytest.raises(ValidationError, match="canonical order"):
        TaskGraphRevisionCandidate(
            proposal=two_node_proposal,
            lineage=(
                TaskGraphSemanticLineageProjection(
                    proposal_node_key="child",
                    disposition="reuse",
                    base_node_alias="base_root",
                ),
                TaskGraphSemanticLineageProjection(
                    proposal_node_key="root",
                    disposition="revise",
                    base_node_alias="base_child",
                ),
            ),
        )

    with pytest.raises(ValidationError, match="cannot map to multiple"):
        TaskGraphRevisionCandidate(
            proposal=two_node_proposal,
            lineage=(
                TaskGraphSemanticLineageProjection(
                    proposal_node_key="root",
                    disposition="reuse",
                    base_node_alias="base_root",
                ),
                TaskGraphSemanticLineageProjection(
                    proposal_node_key="child",
                    disposition="revise",
                    base_node_alias="base_root",
                ),
            ),
        )


def test_semantic_verifier_rejects_incomplete_or_tampered_bindings() -> None:
    request = _semantic_request()
    with pytest.raises(ValidationError, match="request hash"):
        TaskGraphSemanticVerificationRequest.model_validate(
            request.model_dump() | {"binding_sha256": SHA_A}
        )
    with pytest.raises(ValidationError, match="policy hash"):
        TaskGraphSemanticReviewPolicy.model_validate(
            request.review_policy.model_dump() | {"policy_sha256": SHA_D}
        )

    with pytest.raises(ValidationError, match="all dimensions"):
        TaskGraphSemanticVerificationResult.create(
            verification_result_id="semantic_result_incomplete",
            verification_request_id=request.verification_request_id,
            request_binding_sha256=request.binding_sha256,
            logical_call_id=request.logical_call_id,
            verification_profile_id=request.verification_profile_id,
            reviewer_ordinal=1,
            required_reviewer_count=1,
            items=_result_items()[:-1]
            + (
                TaskGraphSemanticVerificationItem(
                    dimension=TaskGraphSemanticVerificationDimension.GOAL_COVERAGE,
                    verdict=TaskGraphSemanticVerificationVerdict.PASS,
                    failure_scope=None,
                    finding="This duplicate cannot replace the missing dimension.",
                ),
            ),
        )

    with pytest.raises(ValidationError, match="extra_forbidden"):
        TaskGraphSemanticVerificationResult.model_validate(
            TaskGraphSemanticVerificationResult.create(
                verification_result_id="semantic_result_extra",
                verification_request_id=request.verification_request_id,
                request_binding_sha256=request.binding_sha256,
                logical_call_id=request.logical_call_id,
                verification_profile_id=request.verification_profile_id,
                reviewer_ordinal=1,
                required_reviewer_count=1,
                items=_result_items(),
            ).model_dump()
            | {"overall_pass": True}
        )


def test_semantic_verifier_requires_exact_evidence_and_gap_coverage() -> None:
    request = _semantic_request()
    omitted_evidence = list(_result_items())
    evidence_index = next(
        index
        for index, item in enumerate(omitted_evidence)
        if item.dimension
        is TaskGraphSemanticVerificationDimension.EVIDENCE_GROUNDING
    )
    omitted_evidence[evidence_index] = omitted_evidence[evidence_index].model_copy(
        update={"evidence_aliases": ()}
    )
    evidence_result = TaskGraphSemanticVerificationResult.create(
        verification_result_id="semantic_result_omitted_evidence",
        verification_request_id=request.verification_request_id,
        request_binding_sha256=request.binding_sha256,
        logical_call_id=request.logical_call_id,
        verification_profile_id=request.verification_profile_id,
        reviewer_ordinal=1,
        required_reviewer_count=1,
        items=tuple(omitted_evidence),
    )
    with pytest.raises(TaskGraphSemanticVerificationBindingError, match="exactly cover"):
        validate_task_graph_semantic_verification_result(
            request=request,
            result=evidence_result,
        )

    prompt = request.prompt_payload
    artifact = prompt.context_artifacts[0]
    artifact_with_gap = PlanningContextArtifactProjection.create(
        artifact_alias=artifact.artifact_alias,
        artifact_id=artifact.artifact_id,
        artifact_sha256=artifact.artifact_sha256,
        producer_node_alias=artifact.producer_node_alias,
        facts=artifact.facts,
        constraints=artifact.constraints,
        conflicts=artifact.conflicts,
        gaps=(
            PlanningContextGapProjection(
                gap_alias="gap_01",
                observation_status=PlanningObservationStatus.PARTIAL,
                description="One required source region remains unreadable.",
                blocking=True,
                affected_obligations=("accept_delivery",),
                evidence_aliases=("obs_01",),
                resolution_hint="Read the missing region before final execution.",
            ),
        ),
    )
    authority_with_gap = PlanningAuthorityProjection.create(
        authority_snapshot_id=prompt.authority.authority_snapshot_id,
        authority_snapshot_sha256=prompt.authority.authority_snapshot_sha256,
        cards=(
            PlanningAuthoritySourceCard(
                alias="gap_01",
                authority_class=PlanningAuthorityClass.GAP,
                source_kind=PlanningAuthoritySourceKind.GAP,
                source_label="Unreadable required source region",
                excerpt="A required source region remains unresolved.",
                projection_sha256=SHA_D,
            ),
            *prompt.authority.cards,
        ),
    )
    prompt_with_gap = TaskGraphSemanticVerificationPromptPayload.create(
        goal=prompt.goal,
        authority=authority_with_gap,
        base_task_graph=prompt.base_task_graph,
        context_artifacts=(artifact_with_gap,),
        capabilities=prompt.capabilities,
        budget=prompt.budget,
        task_graph_proposal=prompt.task_graph_proposal,
        lineage=prompt.lineage,
    )
    request_with_gap = TaskGraphSemanticVerificationRequest.create(
        verification_request_id="semantic_request_with_gap",
        logical_call_id="logical_call_with_gap",
        verification_profile_id=request.verification_profile_id,
        reviewer_ordinal=1,
        required_reviewer_count=1,
        goal=request.goal,
        auxiliary_graph_revision=request.auxiliary_graph_revision,
        auxiliary_graph_structure_sha256=request.auxiliary_graph_structure_sha256,
        prompt_payload=prompt_with_gap,
        review_policy=request.review_policy,
    )
    omitted_gap_result = TaskGraphSemanticVerificationResult.create(
        verification_result_id="semantic_result_omitted_gap",
        verification_request_id=request_with_gap.verification_request_id,
        request_binding_sha256=request_with_gap.binding_sha256,
        logical_call_id=request_with_gap.logical_call_id,
        verification_profile_id=request_with_gap.verification_profile_id,
        reviewer_ordinal=1,
        required_reviewer_count=1,
        items=_result_items(),
    )
    with pytest.raises(TaskGraphSemanticVerificationBindingError, match="exactly cover"):
        validate_task_graph_semantic_verification_result(
            request=request_with_gap,
            result=omitted_gap_result,
        )

    covered_result = omitted_gap_result.model_copy(
        update={
            "items": _result_items(gap_aliases=("gap_01",)),
        }
    )
    covered_result = TaskGraphSemanticVerificationResult.create(
        **covered_result.model_dump(exclude={"result_sha256"})
    )
    with pytest.raises(TaskGraphSemanticVerificationBindingError, match="blocking gap"):
        validate_task_graph_semantic_verification_result(
            request=request_with_gap,
            result=covered_result,
        )

    blocked_items = list(_result_items(gap_aliases=("gap_01",)))
    gap_index = next(
        index
        for index, item in enumerate(blocked_items)
        if item.dimension is TaskGraphSemanticVerificationDimension.GAP_DISPOSITION
    )
    blocked_items[gap_index] = blocked_items[gap_index].model_copy(
        update={
            "verdict": TaskGraphSemanticVerificationVerdict.INSUFFICIENT_EVIDENCE,
            "failure_scope": TaskGraphSemanticFailureScope.MISSING_INFORMATION,
        }
    )
    blocked_result = TaskGraphSemanticVerificationResult.create(
        verification_result_id="semantic_result_blocked_gap",
        verification_request_id=request_with_gap.verification_request_id,
        request_binding_sha256=request_with_gap.binding_sha256,
        logical_call_id=request_with_gap.logical_call_id,
        verification_profile_id=request_with_gap.verification_profile_id,
        reviewer_ordinal=1,
        required_reviewer_count=1,
        items=tuple(blocked_items),
    )
    assert validate_task_graph_semantic_verification_result(
        request=request_with_gap,
        result=blocked_result,
    ).host_disposition is TaskGraphSemanticVerificationDisposition.BLOCKED
    assert validate_task_graph_semantic_verification_quorum(
        requests=(request_with_gap,),
        results=(blocked_result,),
    ) == (blocked_result,)

    with pytest.raises(ValidationError, match="exactly match projected typed gaps"):
        TaskGraphSemanticVerificationPromptPayload.create(
            goal=prompt.goal,
            authority=authority_with_gap,
            base_task_graph=prompt.base_task_graph,
            context_artifacts=prompt.context_artifacts,
            capabilities=prompt.capabilities,
            budget=prompt.budget,
            task_graph_proposal=prompt.task_graph_proposal,
            lineage=prompt.lineage,
        )
