from __future__ import annotations

import hashlib
import json

import pytest
from pydantic import ValidationError

from personagraph.l2.auxiliary_graph import (
    PlanningAuthorityClass,
    PlanningAuthorityProjection,
    PlanningAuthoritySourceCard,
    PlanningAuthoritySourceKind,
    PlanningCapabilityCatalogProjection,
    PlanningCapabilityDescriptor,
    PlanningCapabilityEffect,
    PlanningContextArtifactProjection,
    PlanningGoalPromptContext,
    PlanningObservationStatus,
    TaskGraphSemanticFailureScope,
    TaskGraphSemanticLineageProjection,
    TaskGraphSemanticVerificationDimension,
    TaskGraphSemanticVerificationDisposition,
    TaskGraphSemanticVerificationItem,
    TaskGraphSemanticVerificationPromptPayload,
    TaskGraphSemanticVerificationRequest,
    TaskGraphSemanticVerificationResult,
    TaskGraphSemanticVerificationVerdict,
    required_task_graph_semantic_reviewer_count,
)
from personagraph.l2.task_graph.contracts import (
    InSessionTaskGraphRevisionProposal,
)
from personagraph.l2.auxiliary_execution.driver import (
    canonical_auxiliary_graph_driver_state_guard,
)
from personagraph.l2.auxiliary_execution.work_run.controller import (
    AuxiliaryWorkRunStatus,
    AuxiliaryWorkRunRequest,
    run_auxiliary_model_node,
)
from personagraph.session import store
from personagraph.session.l2_store import task_graph as task_graph_store
from personagraph.session.l2_store import terminal as terminal_store
from personagraph.session.l2_store import auxiliary_graph as auxiliary_graph_store
from personagraph.session.l2_store import planning as planning_store
from personagraph.session.l2_store import semantic_verification as semantic_store
from personagraph.session.persistence.l2.auxiliary_graph import auxiliary_graphs
from personagraph.session.persistence.l2.task_graph import insession_tasks as insession_task_records
from tests.runtime.test_auxiliary_work_run_controller import (
    _PassVerifier,
    _ReplyProvider,
    _acceptance,
    _commit_graph,
    _context,
    _ids,
    _profile,
    _request,
    _run,
    _seed_task,
    _subject,
    _task_graph_submit,
)
from tests.session.test_planning_artifact_seal_persistence import (
    _USER_TEXT as _HOST_USER_TEXT,
    _primitive,
)


def _terminal_proposal() -> InSessionTaskGraphRevisionProposal:
    payload = json.loads(_task_graph_submit())
    return InSessionTaskGraphRevisionProposal.model_validate(
        payload["action"]["proposal"]
    )


def _terminal_submit_for(acceptance_id: str) -> str:
    payload = json.loads(_task_graph_submit())
    payload["acceptance_updates"] = [
        {
            "acceptance_id": acceptance_id,
            "model_claimed_satisfied": True,
        }
    ]
    return json.dumps(payload, ensure_ascii=False)


def _pass_items() -> tuple[TaskGraphSemanticVerificationItem, ...]:
    return tuple(
        TaskGraphSemanticVerificationItem(
            dimension=dimension,
            verdict=TaskGraphSemanticVerificationVerdict.PASS,
            failure_scope=None,
            finding=f"The {dimension.value} check passed against frozen authority.",
        )
        for dimension in TaskGraphSemanticVerificationDimension
    )


def _complete_terminal_only(prefix: str):
    session_id, turn_id, task_id = _seed_task()
    _commit_graph(
        session_id=session_id,
        turn_id=turn_id,
        task_id=task_id,
        terminal_only=True,
    )
    subject = _subject(
        session_id=session_id,
        turn_id=turn_id,
        task_id=task_id,
        executor="terminal_planner",
    )
    request = _request(
        session_id=session_id,
        turn_id=turn_id,
        task_id=task_id,
        subject=subject,
        executor="terminal_planner",
        prefix=prefix,
    )
    completed = _run(
        request,
        attempt_provider=_ReplyProvider([_task_graph_submit()]),
        verifier=_PassVerifier(),
    )
    assert completed.status is AuxiliaryWorkRunStatus.COMPLETED
    details = auxiliary_graphs.get_auxiliary_graph_for_task(
        store._deps(),
        session_id=session_id,
        insession_task_id=task_id,
    )
    assert details is not None
    assert details.authority_snapshot is not None
    assert details.budget is not None
    return session_id, turn_id, task_id, details, ()


def _complete_terminal_with_host_artifact(
    tmp_path,
    prefix: str,
    *,
    observation_status: PlanningObservationStatus = (
        PlanningObservationStatus.SUCCESS
    ),
):
    session_id, turn_id, task_id, seal_command, primitive_result = _primitive(
        tmp_path,
        observation_status=observation_status,
    )
    planning_store.seal_auxiliary_host_primitive_result(
        command=seal_command,
        result=primitive_result,
    )
    frontier = auxiliary_graph_store.project_auxiliary_graph_execution_frontier(
        session_id=session_id,
        turn_id=turn_id,
        insession_task_id=task_id,
    )
    terminal = frontier.ready_fresh[0]
    validation_context = (
        terminal_store.build_auxiliary_terminal_task_graph_validation_context(
            session_id=session_id,
            invocation_turn_id=turn_id,
            task_id=task_id,
        )
    )
    runtime_request = AuxiliaryWorkRunRequest(
        session_id=session_id,
        turn_id=turn_id,
        subject=terminal.subject,
        executor_kind="terminal_planner",
        initial_driver_state_guard_sha256=(
            canonical_auxiliary_graph_driver_state_guard(frontier)
        ),
        id_plan=_ids(prefix),
        task_graph_validation_context=validation_context,
    )
    completed = run_auxiliary_model_node(
        runtime_request,
        profile=_profile(),
        capability_catalogs={},
        attempt_provider=_ReplyProvider([_terminal_submit_for("source_understood")]),
        verification_provider=_PassVerifier(),
        emit=lambda _event: None,
        monotonic_clock=iter(range(1, 100)).__next__,
    )
    assert completed.status is AuxiliaryWorkRunStatus.COMPLETED
    details = auxiliary_graphs.get_auxiliary_graph_for_task(
        store._deps(),
        session_id=session_id,
        insession_task_id=task_id,
    )
    assert details is not None
    assert details.authority_snapshot is not None
    assert details.budget is not None
    return (
        session_id,
        turn_id,
        task_id,
        details,
        (primitive_result.prompt_inputs,),
    )


def _complete_positive_base_terminal(prefix: str):
    """创建图修订版一，随后完成一个正向基础终态。"""

    session_id, turn_id, task_id = _seed_task()
    task = task_graph_store.get_insession_task_details(session_id, task_id)
    window = store.get_turn_execution_window(session_id)
    assert task is not None and window is not None
    committed_base = insession_task_records.commit_insession_task_graph_revision(
        store._deps(),
        session_id=session_id,
        source_turn_id=turn_id,
        target_insession_task_id=task_id,
        expected_current_graph_revision=None,
        expected_task_state_version=task.task_state_version,
        expected_window_revision=int(window["state_version"]),
        apply_id=f"{prefix}-base-commit",
        proposal=_terminal_proposal(),
        trusted_context=_context(
            session_id=session_id,
            turn_id=turn_id,
            task_id=task_id,
        ),
    )
    base_projection = task_graph_store.project_task_graph_semantic_base(
        session_id=session_id,
        task_id=task_id,
        graph_revision=1,
    )
    terminal = auxiliary_graphs.AuxiliaryGraphNodeProposalRecord(
        local_node_key="synthesize",
        node_kind="synthesize",
        executor_kind="terminal_planner",
        title="修订任务图",
        objective="形成下一版完整 TaskGraph 快照",
        source_anchor_ids=("task_creation_source",),
        acceptance_criteria=(_acceptance(),),
        output_contract="task_graph_revision_proposal_v2",
    )
    auxiliary_graphs.commit_auxiliary_graph_revision(
        store._deps(),
        session_id=session_id,
        turn_id=turn_id,
        insession_task_id=task_id,
        expected_task_state_version=committed_base.task_state_version,
        expected_base_task_graph_revision=1,
        expected_control_state_version=None,
        expected_current_auxiliary_graph_revision=None,
        apply_id=f"{prefix}-aux-commit",
        goal_objective="在保留根任务身份的同时形成下一版任务图",
        proposal=auxiliary_graphs.AuxiliaryGraphRevisionProposalRecord(
            revision_reason="initial",
            terminal_node_key="synthesize",
            nodes=(terminal,),
            edges=(),
        ),
        authority_context={"anchors": []},
        budget_profile={"profile_id": "planning-test-v1"},
        auxiliary_graph_id=f"{prefix}-aux-graph",
        goal_id=f"{prefix}-goal",
    )
    details = auxiliary_graphs.get_auxiliary_graph_for_task(
        store._deps(),
        session_id=session_id,
        insession_task_id=task_id,
    )
    frontier = auxiliary_graph_store.project_auxiliary_graph_execution_frontier(
        session_id=session_id,
        turn_id=turn_id,
        insession_task_id=task_id,
    )
    assert details is not None and len(frontier.ready_fresh) == 1
    submit = json.loads(_task_graph_submit())
    submit["action"]["lineage"] = [
        {
            "proposal_node_key": "root",
            "disposition": "reuse",
            "base_node_alias": base_projection.snapshot.root_node_alias,
        }
    ]
    request = AuxiliaryWorkRunRequest(
        session_id=session_id,
        turn_id=turn_id,
        subject=frontier.ready_fresh[0].subject,
        executor_kind="terminal_planner",
        initial_driver_state_guard_sha256=(
            canonical_auxiliary_graph_driver_state_guard(frontier)
        ),
        id_plan=_ids(prefix),
        task_graph_validation_context=(
            terminal_store.build_auxiliary_terminal_task_graph_validation_context(
                session_id=session_id,
                invocation_turn_id=turn_id,
                task_id=task_id,
            )
        ),
        task_graph_semantic_base_snapshot=base_projection.snapshot,
    )
    completed = run_auxiliary_model_node(
        request,
        profile=_profile(),
        capability_catalogs={},
        attempt_provider=_ReplyProvider(
            [json.dumps(submit, ensure_ascii=False)]
        ),
        verification_provider=_PassVerifier(),
        emit=lambda _event: None,
        monotonic_clock=iter(range(1, 100)).__next__,
    )
    assert completed.status is AuxiliaryWorkRunStatus.COMPLETED
    details = auxiliary_graphs.get_auxiliary_graph_for_task(
        store._deps(),
        session_id=session_id,
        insession_task_id=task_id,
    )
    assert details is not None
    return (
        session_id,
        turn_id,
        task_id,
        details,
        base_projection.snapshot,
    )


def _catalog(*, protected: bool) -> PlanningCapabilityCatalogProjection:
    return PlanningCapabilityCatalogProjection.create(
        capability_catalog_snapshot_id=(
            "semantic-capabilities-protected"
            if protected
            else "semantic-capabilities-readonly"
        ),
        capability_catalog_snapshot_sha256=("d" if protected else "c") * 64,
        capabilities=(
            PlanningCapabilityDescriptor(
                capability_alias=("publish" if protected else "read_documents"),
                label=("Publish" if protected else "Read documents"),
                description=(
                    "Publish an externally visible task artifact."
                    if protected
                    else "Read bounded documents without side effects."
                ),
                available=True,
                effect=(
                    PlanningCapabilityEffect.PROTECTED
                    if protected
                    else PlanningCapabilityEffect.READ_ONLY
                ),
                supported_operations=("publish" if protected else "read",),
                supported_resource_kinds=("task_graph",),
            ),
        ),
    )


def _authority_projection(details, prompt_inputs):
    assert details.authority_snapshot is not None
    cards: list[PlanningAuthoritySourceCard] = []
    for anchor in details.authority_snapshot.anchors:
        assert anchor.authority_class is PlanningAuthorityClass.AUTHORIZATION
        cards.append(
            PlanningAuthoritySourceCard(
                alias=anchor.projection_alias,
                authority_class=anchor.authority_class,
                source_kind=PlanningAuthoritySourceKind.USER_INSTRUCTION,
                source_label="Task creation instruction",
                excerpt=(
                    _HOST_USER_TEXT
                    if prompt_inputs
                    else "请分析材料并形成一个可执行计划"
                ),
                projection_sha256=anchor.projection_sha256,
            )
        )
    for prompt in prompt_inputs:
        cards.extend(prompt.source_cards)
    cards.sort(key=lambda item: item.alias)
    return PlanningAuthorityProjection.create(
        authority_snapshot_id=details.authority_snapshot_id,
        authority_snapshot_sha256=details.authority_snapshot_sha256,
        cards=tuple(cards),
    )


def _semantic_request(
    *,
    request_id: str,
    logical_call_id: str,
    reviewer_ordinal: int,
    details,
    catalog: PlanningCapabilityCatalogProjection,
    prompt_inputs: tuple[object, ...],
) -> TaskGraphSemanticVerificationRequest:
    artifacts = tuple(prompt.context_artifact for prompt in prompt_inputs)
    with store._connect() as conn:
        goal_objective = str(
            conn.execute(
                "SELECT objective FROM insession_auxiliary_graph_goals "
                "WHERE session_id=? AND insession_task_id=? AND goal_id=?",
                (details.session_id, details.task_id, details.goal_id),
            ).fetchone()[0]
        )
    payload = TaskGraphSemanticVerificationPromptPayload.create(
        goal=PlanningGoalPromptContext(
            goal_id=details.goal_id,
            objective=goal_objective,
            desired_output="A complete, source-bound TaskGraph proposal.",
            authorization_aliases=("task_creation_source",),
        ),
        authority=_authority_projection(details, prompt_inputs),
        base_task_graph=None,
        context_artifacts=artifacts,
        capabilities=catalog,
        budget=details.budget,
        task_graph_proposal=_terminal_proposal(),
    )
    policy = semantic_store.derive_auxiliary_semantic_review_policy(
        prompt_payload=payload
    )
    required = (
        2
        if catalog.capabilities[0].effect is PlanningCapabilityEffect.PROTECTED
        else 1
    )
    return TaskGraphSemanticVerificationRequest.create(
        verification_request_id=request_id,
        logical_call_id=logical_call_id,
        verification_profile_id="semantic-verifier-v1",
        reviewer_ordinal=reviewer_ordinal,
        required_reviewer_count=required,
        goal=details.goal,
        auxiliary_graph_revision=details.auxiliary_graph_revision,
        auxiliary_graph_structure_sha256=details.structure_sha256,
        prompt_payload=payload,
        review_policy=policy,
    )


def _semantic_policy_payload_with_document_cards(
    *,
    details,
    source_cards: tuple[PlanningAuthoritySourceCard, ...],
) -> TaskGraphSemanticVerificationPromptPayload:
    base_authority = _authority_projection(details, ())
    return TaskGraphSemanticVerificationPromptPayload.create(
        goal=PlanningGoalPromptContext(
            goal_id=details.goal_id,
            objective="Understand the mounted documents.",
            desired_output="A source-bound TaskGraph proposal.",
            authorization_aliases=("task_creation_source",),
        ),
        authority=PlanningAuthorityProjection.create(
            authority_snapshot_id=base_authority.authority_snapshot_id,
            authority_snapshot_sha256=base_authority.authority_snapshot_sha256,
            cards=tuple(
                sorted(
                    (*base_authority.cards, *source_cards),
                    key=lambda item: item.alias,
                )
            ),
        ),
        base_task_graph=None,
        capabilities=_catalog(protected=False),
        budget=details.budget,
        task_graph_proposal=_terminal_proposal(),
    )


def test_semantic_policy_counts_one_document_group_across_multiple_chunks() -> None:
    _session_id, _turn_id, _task_id, details, _prompt_inputs = (
        _complete_terminal_only("semantic-document-group-one")
    )
    cards = tuple(
        PlanningAuthoritySourceCard(
            alias=f"document_01_obs_{ordinal:03d}",
            authority_class=PlanningAuthorityClass.EVIDENCE,
            source_kind=PlanningAuthoritySourceKind.DOCUMENT,
            document_group_alias="document_group_01",
            source_label="Untrusted PDF document evidence",
            excerpt=f"Document chunk {ordinal}.",
            projection_sha256=f"{ordinal:x}" * 64,
        )
        for ordinal in (1, 2)
    )
    payload = _semantic_policy_payload_with_document_cards(
        details=details,
        source_cards=cards,
    )

    policy = semantic_store.derive_auxiliary_semantic_review_policy(
        prompt_payload=payload
    )

    assert policy.distinct_document_count == 1
    assert required_task_graph_semantic_reviewer_count(
        prompt_payload=payload,
        review_policy=policy,
    ) == 1


def test_semantic_policy_keeps_distinct_document_groups_separate() -> None:
    _session_id, _turn_id, _task_id, details, _prompt_inputs = (
        _complete_terminal_only("semantic-document-group-two")
    )
    cards = tuple(
        PlanningAuthoritySourceCard(
            alias=f"document_{ordinal:02d}_obs_001",
            authority_class=PlanningAuthorityClass.EVIDENCE,
            source_kind=PlanningAuthoritySourceKind.DOCUMENT,
            document_group_alias=f"document_group_{ordinal:02d}",
            source_label="Untrusted PDF document evidence",
            excerpt=f"Document {ordinal} evidence.",
            projection_sha256=f"{ordinal:x}" * 64,
        )
        for ordinal in (1, 2)
    )
    payload = _semantic_policy_payload_with_document_cards(
        details=details,
        source_cards=cards,
    )

    policy = semantic_store.derive_auxiliary_semantic_review_policy(
        prompt_payload=payload
    )

    assert policy.distinct_document_count == 2
    assert required_task_graph_semantic_reviewer_count(
        prompt_payload=payload,
        review_policy=policy,
    ) == 2


def test_legacy_document_cards_keep_conservative_per_card_counting() -> None:
    _session_id, _turn_id, _task_id, details, _prompt_inputs = (
        _complete_terminal_only("semantic-document-group-legacy")
    )
    cards = tuple(
        PlanningAuthoritySourceCard(
            alias=f"legacy_document_obs_{ordinal:03d}",
            authority_class=PlanningAuthorityClass.EVIDENCE,
            source_kind=PlanningAuthoritySourceKind.DOCUMENT,
            source_label="Legacy document evidence",
            excerpt=f"Legacy evidence card {ordinal}.",
            projection_sha256=f"{ordinal:x}" * 64,
        )
        for ordinal in (1, 2)
    )
    payload = _semantic_policy_payload_with_document_cards(
        details=details,
        source_cards=cards,
    )

    policy = semantic_store.derive_auxiliary_semantic_review_policy(
        prompt_payload=payload
    )

    assert all(
        "document_group_alias" not in card.model_dump(mode="json")
        for card in cards
    )
    assert policy.distinct_document_count == 2
    assert required_task_graph_semantic_reviewer_count(
        prompt_payload=payload,
        review_policy=policy,
    ) == 2


def test_visual_input_still_requires_two_reviewers_for_one_document() -> None:
    _session_id, _turn_id, _task_id, details, _prompt_inputs = (
        _complete_terminal_only("semantic-document-group-visual")
    )
    source_cards = (
        PlanningAuthoritySourceCard(
            alias="document_01_obs_001",
            authority_class=PlanningAuthorityClass.EVIDENCE,
            source_kind=PlanningAuthoritySourceKind.DOCUMENT,
            document_group_alias="document_group_01",
            source_label="Untrusted PDF document evidence",
            excerpt="Document text evidence.",
            projection_sha256="1" * 64,
        ),
        PlanningAuthoritySourceCard(
            alias="document_01_visual_001",
            authority_class=PlanningAuthorityClass.EVIDENCE,
            source_kind=PlanningAuthoritySourceKind.VISUAL,
            source_label="Untrusted document visual evidence",
            excerpt="A chart extracted from the same document.",
            projection_sha256="2" * 64,
        ),
    )
    payload = _semantic_policy_payload_with_document_cards(
        details=details,
        source_cards=source_cards,
    )

    policy = semantic_store.derive_auxiliary_semantic_review_policy(
        prompt_payload=payload
    )

    assert policy.distinct_document_count == 1
    assert policy.has_visual_input is True
    assert required_task_graph_semantic_reviewer_count(
        prompt_payload=payload,
        review_policy=policy,
    ) == 2


def _freeze_command(session_id, turn_id, task_id, details, catalog):
    return semantic_store.FreezeAuxiliarySemanticCapabilityCatalogCommand(
        session_id=session_id,
        created_turn_id=turn_id,
        task_id=task_id,
        auxiliary_graph_id=details.auxiliary_graph_id,
        goal_id=details.goal_id,
        auxiliary_graph_revision=details.auxiliary_graph_revision,
        expected_control_state_version=details.control_state_version,
        expected_goal_state_version=details.goal_state_version,
        expected_revision_state_version=details.revision_state_version,
        expected_budget_state_version=details.budget_state_version,
        expected_authority_snapshot_id=details.authority_snapshot_id,
        expected_authority_snapshot_sha256=details.authority_snapshot_sha256,
        expected_structure_sha256=details.structure_sha256,
        catalog=catalog,
    )


def _request_command(session_id, turn_id, details, request):
    return semantic_store.CommitAuxiliarySemanticVerificationRequestCommand(
        session_id=session_id,
        created_turn_id=turn_id,
        expected_control_state_version=details.control_state_version,
        expected_goal_state_version=details.goal_state_version,
        expected_revision_state_version=details.revision_state_version,
        expected_budget_state_version=details.budget_state_version,
        request=request,
    )


def _result(request, result_id, *, failed_dimension=None):
    items = list(_pass_items())
    if failed_dimension is not None:
        failed_index = next(
            index
            for index, item in enumerate(items)
            if item.dimension is failed_dimension
        )
        items[failed_index] = items[failed_index].model_copy(
            update={
                "verdict": TaskGraphSemanticVerificationVerdict.FAIL,
                "failure_scope": (
                    TaskGraphSemanticFailureScope.TERMINAL_PROPOSAL
                ),
                "finding": (
                    f"The {failed_dimension.value} check requires revision."
                ),
            }
        )
    return TaskGraphSemanticVerificationResult.create(
        verification_result_id=result_id,
        verification_request_id=request.verification_request_id,
        request_binding_sha256=request.binding_sha256,
        logical_call_id=request.logical_call_id,
        verification_profile_id=request.verification_profile_id,
        reviewer_ordinal=request.reviewer_ordinal,
        required_reviewer_count=request.required_reviewer_count,
        items=tuple(items),
    )


def _result_command(session_id, turn_id, details, result):
    return semantic_store.CommitAuxiliarySemanticVerificationResultCommand(
        session_id=session_id,
        created_turn_id=turn_id,
        expected_control_state_version=details.control_state_version,
        expected_goal_state_version=details.goal_state_version,
        expected_revision_state_version=details.revision_state_version,
        expected_budget_state_version=details.budget_state_version,
        result=result,
    )


def _settlement_command(
    session_id, turn_id, details, requests, results, settlement_id
):
    return semantic_store.SettleAuxiliarySemanticVerificationQuorumCommand(
        settlement_id=settlement_id,
        session_id=session_id,
        created_turn_id=turn_id,
        expected_control_state_version=details.control_state_version,
        expected_goal_state_version=details.goal_state_version,
        expected_revision_state_version=details.revision_state_version,
        expected_budget_state_version=details.budget_state_version,
        requests=requests,
        results=results,
    )


def test_semantic_material_projection_rebuilds_verified_terminal_after_restart() -> None:
    session_id, turn_id, task_id, details, _prompt_inputs = (
        _complete_terminal_only("semantic-material-terminal")
    )

    material = semantic_store.project_auxiliary_semantic_material(
        session_id=session_id,
        turn_id=turn_id,
        task_id=task_id,
    )

    assert material.auxiliary_graph_id == details.auxiliary_graph_id
    assert material.goal_id == details.goal_id
    assert material.auxiliary_graph_revision == details.auxiliary_graph_revision
    assert material.terminal_subject.node_id == details.terminal_auxiliary_node_id
    assert material.task_graph_proposal == _terminal_proposal()
    assert material.lineage == ()
    assert material.context_artifacts == ()
    assert material.observation_source_cards == ()


def test_semantic_material_projection_rebuilds_host_artifact_and_cards(
    tmp_path,
) -> None:
    session_id, turn_id, task_id, details, prompt_inputs = (
        _complete_terminal_with_host_artifact(
            tmp_path,
            "semantic-material-host",
        )
    )

    material = semantic_store.project_auxiliary_semantic_material(
        session_id=session_id,
        turn_id=turn_id,
        task_id=task_id,
    )

    assert material.structure_sha256 == details.structure_sha256
    assert material.context_artifacts == tuple(
        item.context_artifact for item in prompt_inputs
    )
    assert material.observation_source_cards == tuple(
        sorted(
            (
                card
                for item in prompt_inputs
                for card in item.source_cards
            ),
            key=lambda card: card.alias,
        )
    )


def test_positive_base_semantic_material_exposes_authenticated_ordered_lineage() -> None:
    session_id, turn_id, task_id, details, base_snapshot = (
        _complete_positive_base_terminal("semantic-material-positive")
    )

    material = semantic_store.project_auxiliary_semantic_material(
        session_id=session_id,
        turn_id=turn_id,
        task_id=task_id,
    )

    expected_lineage = (
        TaskGraphSemanticLineageProjection(
            proposal_node_key="root",
            disposition="reuse",
            base_node_alias=base_snapshot.root_node_alias,
        ),
    )
    assert details.base_task_graph_revision == 1
    assert material.task_graph_proposal == _terminal_proposal()
    assert material.lineage == expected_lineage
    assert material == type(material).model_validate_json(
        material.model_dump_json()
    )

    tampered = material.model_dump(mode="json")
    tampered["lineage"][0]["disposition"] = "revise"
    with pytest.raises(ValidationError, match="projection hash"):
        type(material).model_validate(tampered)


def test_positive_base_semantic_material_rejects_authenticated_bare_proposal() -> None:
    session_id, turn_id, task_id, _details, _base_snapshot = (
        _complete_positive_base_terminal("semantic-material-bare-positive")
    )

    def canonical_json(value: object) -> str:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    with store._connect() as conn:
        row = conn.execute(
            "SELECT completion.completion_id, completion.completion_json, "
            "output.work_run_id, output.snapshot_json "
            "FROM insession_auxiliary_node_completions_v2 AS completion "
            "JOIN insession_work_run_output_windows AS output "
            "ON output.work_run_id=completion.work_run_id "
            "AND output.output_revision=completion.output_revision "
            "WHERE completion.insession_task_id=?",
            (task_id,),
        ).fetchone()
        assert row is not None
        output_payload = json.loads(str(row["snapshot_json"]))
        output_payload["content"] = _terminal_proposal().model_dump_json()
        output_json = canonical_json(output_payload)
        output_sha256 = hashlib.sha256(output_json.encode("utf-8")).hexdigest()
        completion_payload = json.loads(str(row["completion_json"]))
        completion_payload["output_snapshot_sha256"] = output_sha256
        completion_json = canonical_json(completion_payload)
        completion_sha256 = hashlib.sha256(
            completion_json.encode("utf-8")
        ).hexdigest()
        conn.execute(
            "UPDATE insession_work_run_output_windows "
            "SET snapshot_json=?, snapshot_hash=? WHERE work_run_id=?",
            (output_json, output_sha256, str(row["work_run_id"])),
        )
        conn.execute(
            "UPDATE insession_auxiliary_node_completions_v2 "
            "SET completion_json=?, completion_sha256=? WHERE completion_id=?",
            (completion_json, completion_sha256, str(row["completion_id"])),
        )

    with pytest.raises(
        semantic_store.AuxiliarySemanticVerificationStoredAuthorityCorrupt,
        match="wrong typed TaskGraph envelope",
    ):
        semantic_store.project_auxiliary_semantic_material(
            session_id=session_id,
            turn_id=turn_id,
            task_id=task_id,
        )


def test_positive_base_semantic_request_rejects_lineage_different_from_terminal() -> None:
    session_id, turn_id, task_id, details, base_snapshot = (
        _complete_positive_base_terminal("semantic-request-positive")
    )
    material = semantic_store.project_auxiliary_semantic_material(
        session_id=session_id,
        turn_id=turn_id,
        task_id=task_id,
    )
    catalog_template = _catalog(protected=False)
    catalog = type(catalog_template).create(
        capability_catalog_snapshot_id="semantic-positive-capabilities",
        capability_catalog_snapshot_sha256="e" * 64,
        capabilities=catalog_template.capabilities,
    )
    semantic_store.freeze_auxiliary_semantic_capability_catalog(
        command=_freeze_command(
            session_id,
            turn_id,
            task_id,
            details,
            catalog,
        )
    )
    with store._connect() as conn:
        objective = str(
            conn.execute(
                "SELECT objective FROM insession_auxiliary_graph_goals "
                "WHERE goal_id=?",
                (details.goal_id,),
            ).fetchone()[0]
        )
    payload = TaskGraphSemanticVerificationPromptPayload.create(
        goal=PlanningGoalPromptContext(
            goal_id=details.goal_id,
            objective=objective,
            desired_output="A complete next TaskGraph snapshot.",
            authorization_aliases=("task_creation_source",),
        ),
        authority=_authority_projection(details, ()),
        base_task_graph=base_snapshot,
        capabilities=catalog,
        budget=details.budget,
        task_graph_proposal=material.task_graph_proposal,
        lineage=material.lineage,
    )
    request = TaskGraphSemanticVerificationRequest.create(
        verification_request_id="semantic-positive-request",
        logical_call_id="semantic-positive-call",
        verification_profile_id="semantic-verifier-v1",
        reviewer_ordinal=1,
        required_reviewer_count=1,
        goal=details.goal,
        auxiliary_graph_revision=details.auxiliary_graph_revision,
        auxiliary_graph_structure_sha256=details.structure_sha256,
        prompt_payload=payload,
        review_policy=semantic_store.derive_auxiliary_semantic_review_policy(
            prompt_payload=payload
        ),
    )
    applied = semantic_store.commit_auxiliary_semantic_verification_request(
        command=_request_command(session_id, turn_id, details, request)
    )
    assert applied.record.request.prompt_payload.lineage == material.lineage
    assert semantic_store.commit_auxiliary_semantic_verification_request(
        command=_request_command(session_id, turn_id, details, request)
    ) == applied.model_copy(update={"status": "replayed"})

    different_lineage = (
        TaskGraphSemanticLineageProjection(
            proposal_node_key=material.lineage[0].proposal_node_key,
            disposition="revise",
            base_node_alias=material.lineage[0].base_node_alias,
        ),
    )
    tampered_payload = TaskGraphSemanticVerificationPromptPayload.create(
        **payload.model_dump(
            mode="python",
            exclude={"payload_sha256", "lineage"},
        ),
        lineage=different_lineage,
    )
    tampered_request = TaskGraphSemanticVerificationRequest.create(
        **request.model_dump(
            mode="python",
            exclude={
                "binding_sha256",
                "prompt_payload",
                "review_policy",
                "task_graph_proposal_sha256",
                "blocking_gap_aliases",
                "non_blocking_gap_aliases",
                "verification_request_id",
                "logical_call_id",
            },
        ),
        verification_request_id="semantic-positive-tampered-request",
        logical_call_id="semantic-positive-tampered-call",
        prompt_payload=tampered_payload,
        review_policy=semantic_store.derive_auxiliary_semantic_review_policy(
            prompt_payload=tampered_payload
        ),
    )
    with pytest.raises(
        semantic_store.AuxiliarySemanticVerificationStoredAuthorityCorrupt,
        match="terminal proposal completion binding",
    ):
        semantic_store.commit_auxiliary_semantic_verification_request(
            command=_request_command(
                session_id,
                turn_id,
                details,
                tampered_request,
            )
        )


def test_one_reviewer_semantic_authority_is_atomic_and_exactly_replayable() -> None:
    session_id, turn_id, task_id, details, prompt_inputs = _complete_terminal_only(
        "semantic-one"
    )
    catalog = _catalog(protected=False)
    freeze_command = _freeze_command(
        session_id, turn_id, task_id, details, catalog
    )
    frozen = semantic_store.freeze_auxiliary_semantic_capability_catalog(
        command=freeze_command
    )
    assert frozen.status == "applied"
    assert frozen.catalog.projection == catalog
    assert semantic_store.freeze_auxiliary_semantic_capability_catalog(
        command=freeze_command
    ) == frozen.model_copy(update={"status": "replayed"})

    collision_catalog = PlanningCapabilityCatalogProjection.create(
        capability_catalog_snapshot_id=catalog.capability_catalog_snapshot_id,
        capability_catalog_snapshot_sha256=(
            catalog.capability_catalog_snapshot_sha256
        ),
        capabilities=(
            catalog.capabilities[0].model_copy(
                update={"description": "A different immutable catalog payload."}
            ),
        ),
    )
    with pytest.raises(semantic_store.AuxiliarySemanticVerificationIdentityCollision):
        semantic_store.freeze_auxiliary_semantic_capability_catalog(
            command=freeze_command.model_copy(update={"catalog": collision_catalog})
        )

    request = _semantic_request(
        request_id="semantic-request-one",
        logical_call_id="semantic-call-one",
        reviewer_ordinal=1,
        details=details,
        catalog=catalog,
        prompt_inputs=prompt_inputs,
    )
    request_command = _request_command(session_id, turn_id, details, request)
    requested = semantic_store.commit_auxiliary_semantic_verification_request(
        command=request_command
    )
    assert requested.status == "applied"
    assert requested.record.request == request
    assert semantic_store.commit_auxiliary_semantic_verification_request(
        command=request_command
    ) == requested.model_copy(update={"status": "replayed"})
    collision_request = TaskGraphSemanticVerificationRequest.create(
        **request.model_dump(
            mode="python",
            exclude={
                "binding_sha256",
                "task_graph_proposal_sha256",
                "blocking_gap_aliases",
                "non_blocking_gap_aliases",
                "logical_call_id",
            },
        ),
        logical_call_id="semantic-call-collision",
    )
    with pytest.raises(semantic_store.AuxiliarySemanticVerificationIdentityCollision):
        semantic_store.commit_auxiliary_semantic_verification_request(
            command=request_command.model_copy(update={"request": collision_request})
        )

    result = _result(request, "semantic-result-one")
    result_command = _result_command(session_id, turn_id, details, result)
    with pytest.raises(semantic_store.AuxiliarySemanticVerificationStaleAuthority):
        semantic_store.commit_auxiliary_semantic_verification_result(
            command=result_command.model_copy(
                update={
                    "expected_budget_state_version": details.budget_state_version
                    + 1
                }
            )
        )
    committed_result = semantic_store.commit_auxiliary_semantic_verification_result(
        command=result_command
    )
    assert committed_result.status == "applied"
    assert committed_result.record.result == result
    assert semantic_store.commit_auxiliary_semantic_verification_result(
        command=result_command
    ) == committed_result.model_copy(update={"status": "replayed"})
    collision_items = list(result.items)
    collision_items[0] = collision_items[0].model_copy(
        update={"finding": "A different immutable reviewer finding."}
    )
    collision_result = TaskGraphSemanticVerificationResult.create(
        **result.model_dump(
            mode="python",
            exclude={"items", "result_sha256"},
        ),
        items=tuple(collision_items),
    )
    with pytest.raises(semantic_store.AuxiliarySemanticVerificationIdentityCollision):
        semantic_store.commit_auxiliary_semantic_verification_result(
            command=result_command.model_copy(update={"result": collision_result})
        )

    settlement_command = _settlement_command(
        session_id,
        turn_id,
        details,
        (request,),
        (result,),
        "semantic-settlement-one",
    )
    with pytest.raises(semantic_store.AuxiliarySemanticVerificationStaleAuthority):
        semantic_store.settle_auxiliary_semantic_verification_quorum(
            command=settlement_command.model_copy(
                update={
                    "expected_revision_state_version": (
                        details.revision_state_version + 1
                    )
                }
            )
        )
    settled = semantic_store.settle_auxiliary_semantic_verification_quorum(
        command=settlement_command
    )
    assert settled.status == "applied"
    assert settled.settlement.required_reviewer_count == 1
    assert settled.settlement.host_disposition == "pass"
    assert semantic_store.settle_auxiliary_semantic_verification_quorum(
        command=settlement_command
    ) == settled.model_copy(update={"status": "replayed"})
    with pytest.raises(semantic_store.AuxiliarySemanticVerificationIdentityCollision):
        semantic_store.settle_auxiliary_semantic_verification_quorum(
            command=settlement_command.model_copy(
                update={"created_turn_id": "semantic-collision-turn"}
            )
        )
    with pytest.raises(semantic_store.AuxiliarySemanticVerificationIdentityCollision):
        semantic_store.settle_auxiliary_semantic_verification_quorum(
            command=settlement_command.model_copy(
                update={"settlement_id": "semantic-settlement-other"}
            )
        )
    loaded = semantic_store.get_auxiliary_semantic_quorum_settlement(
        session_id=session_id,
        task_id=task_id,
        auxiliary_graph_id=details.auxiliary_graph_id,
        goal_id=details.goal_id,
        auxiliary_graph_revision=details.auxiliary_graph_revision,
        frozen_prompt_payload_sha256=request.prompt_payload.payload_sha256,
    )
    assert loaded == settled.settlement

    with store._connect() as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM insession_auxiliary_capability_catalog_items"
        ).fetchone()[0] == 1
        assert conn.execute(
            "SELECT COUNT(*) FROM insession_auxiliary_semantic_verification_requests"
        ).fetchone()[0] == 1
        assert conn.execute(
            "SELECT COUNT(*) FROM insession_auxiliary_semantic_verification_results"
        ).fetchone()[0] == 1
        assert conn.execute(
            "SELECT COUNT(*) FROM insession_auxiliary_semantic_quorum_reviewers"
        ).fetchone()[0] == 1
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []


def test_revise_quorum_exactly_settles_and_replays_without_pass_coercion() -> None:
    session_id, turn_id, task_id, details, prompt_inputs = _complete_terminal_only(
        "semantic-revise"
    )
    catalog = _catalog(protected=True)
    semantic_store.freeze_auxiliary_semantic_capability_catalog(
        command=_freeze_command(session_id, turn_id, task_id, details, catalog)
    )
    requests = tuple(
        _semantic_request(
            request_id=f"semantic-request-revise-{ordinal}",
            logical_call_id=f"semantic-call-revise-{ordinal}",
            reviewer_ordinal=ordinal,
            details=details,
            catalog=catalog,
            prompt_inputs=prompt_inputs,
        )
        for ordinal in (1, 2)
    )
    for request in requests:
        semantic_store.commit_auxiliary_semantic_verification_request(
            command=_request_command(session_id, turn_id, details, request)
        )
    results = (
        _result(requests[0], "semantic-result-revise-1"),
        _result(
            requests[1],
            "semantic-result-revise-2",
            failed_dimension=TaskGraphSemanticVerificationDimension.GOAL_COVERAGE,
        ),
    )
    for result in results:
        semantic_store.commit_auxiliary_semantic_verification_result(
            command=_result_command(session_id, turn_id, details, result)
        )
    command = _settlement_command(
        session_id,
        turn_id,
        details,
        requests,
        results,
        "semantic-settlement-revise",
    )

    settled = semantic_store.settle_auxiliary_semantic_verification_quorum(command=command)

    assert settled.status == "applied"
    assert (
        settled.settlement.host_disposition
        is TaskGraphSemanticVerificationDisposition.REVISE
    )
    assert semantic_store.settle_auxiliary_semantic_verification_quorum(
        command=command
    ) == settled.model_copy(update={"status": "replayed"})
    assert semantic_store.get_auxiliary_semantic_quorum_settlement(
        session_id=session_id,
        task_id=task_id,
        auxiliary_graph_id=details.auxiliary_graph_id,
        goal_id=details.goal_id,
        auxiliary_graph_revision=details.auxiliary_graph_revision,
        frozen_prompt_payload_sha256=requests[0].prompt_payload.payload_sha256,
    ) == settled.settlement


def test_two_reviewer_quorum_rejects_stale_incomplete_and_tampered_authority() -> None:
    session_id, turn_id, task_id, details, prompt_inputs = _complete_terminal_only(
        "semantic-two"
    )
    catalog = _catalog(protected=True)
    semantic_store.freeze_auxiliary_semantic_capability_catalog(
        command=_freeze_command(session_id, turn_id, task_id, details, catalog)
    )
    requests = tuple(
        _semantic_request(
            request_id=f"semantic-request-two-{ordinal}",
            logical_call_id=f"semantic-call-two-{ordinal}",
            reviewer_ordinal=ordinal,
            details=details,
            catalog=catalog,
            prompt_inputs=prompt_inputs,
        )
        for ordinal in (1, 2)
    )
    stale = _request_command(session_id, turn_id, details, requests[0]).model_copy(
        update={
            "expected_revision_state_version": details.revision_state_version + 1
        }
    )
    with pytest.raises(semantic_store.AuxiliarySemanticVerificationStaleAuthority):
        semantic_store.commit_auxiliary_semantic_verification_request(command=stale)
    for request in requests:
        semantic_store.commit_auxiliary_semantic_verification_request(
            command=_request_command(session_id, turn_id, details, request)
        )
    results = tuple(
        _result(request, f"semantic-result-two-{request.reviewer_ordinal}")
        for request in requests
    )
    for result in results:
        semantic_store.commit_auxiliary_semantic_verification_result(
            command=_result_command(session_id, turn_id, details, result)
        )
    with pytest.raises(semantic_store.AuxiliarySemanticVerificationPersistenceError):
        semantic_store.settle_auxiliary_semantic_verification_quorum(
            command=_settlement_command(
                session_id,
                turn_id,
                details,
                requests[:1],
                results[:1],
                "semantic-settlement-incomplete",
            )
        )
    settled = semantic_store.settle_auxiliary_semantic_verification_quorum(
        command=_settlement_command(
            session_id,
            turn_id,
            details,
            requests,
            results,
            "semantic-settlement-two",
        )
    )
    assert settled.settlement.required_reviewer_count == 2
    assert tuple(
        item.reviewer_ordinal for item in settled.settlement.requests
    ) == (1, 2)

    with store._connect() as conn:
        conn.execute(
            "UPDATE insession_auxiliary_capability_catalog_items "
            "SET descriptor_json='{}' WHERE capability_catalog_snapshot_id=?",
            (catalog.capability_catalog_snapshot_id,),
        )
    with pytest.raises(semantic_store.AuxiliarySemanticVerificationStoredAuthorityCorrupt):
        semantic_store.get_auxiliary_semantic_quorum_settlement(
            session_id=session_id,
            task_id=task_id,
            auxiliary_graph_id=details.auxiliary_graph_id,
            goal_id=details.goal_id,
            auxiliary_graph_revision=details.auxiliary_graph_revision,
            frozen_prompt_payload_sha256=(
                requests[0].prompt_payload.payload_sha256
            ),
        )
    drifted = TaskGraphSemanticVerificationRequest.create(
        **requests[0].model_dump(
            mode="python",
            exclude={
                "binding_sha256",
                "task_graph_proposal_sha256",
                "blocking_gap_aliases",
                "non_blocking_gap_aliases",
                "verification_request_id",
            },
        ),
        verification_request_id="semantic-request-after-tamper",
    )
    with pytest.raises(semantic_store.AuxiliarySemanticVerificationStoredAuthorityCorrupt):
        semantic_store.commit_auxiliary_semantic_verification_request(
            command=_request_command(session_id, turn_id, details, drifted)
        )


def test_request_binds_exact_sealed_context_artifact_projection(tmp_path) -> None:
    session_id, turn_id, task_id, details, prompt_inputs = (
        _complete_terminal_with_host_artifact(tmp_path, "semantic-context")
    )
    catalog = _catalog(protected=False)
    semantic_store.freeze_auxiliary_semantic_capability_catalog(
        command=_freeze_command(session_id, turn_id, task_id, details, catalog)
    )
    request = _semantic_request(
        request_id="semantic-request-context",
        logical_call_id="semantic-call-context",
        reviewer_ordinal=1,
        details=details,
        catalog=catalog,
        prompt_inputs=prompt_inputs,
    )
    committed = semantic_store.commit_auxiliary_semantic_verification_request(
        command=_request_command(session_id, turn_id, details, request)
    )
    assert committed.record.request.prompt_payload.context_artifacts == (
        prompt_inputs[0].context_artifact,
    )
    with store._connect() as conn:
        row = conn.execute(
            "SELECT artifact_id, artifact_sha256, projection_sha256 "
            "FROM insession_auxiliary_semantic_request_context_artifacts"
        ).fetchone()
        assert tuple(row) == (
            prompt_inputs[0].context_artifact.artifact_id,
            prompt_inputs[0].context_artifact.artifact_sha256,
            prompt_inputs[0].context_artifact.projection_sha256,
        )
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []

    forged_projection = PlanningContextArtifactProjection.create(
        **prompt_inputs[0].context_artifact.model_dump(
            mode="python",
            exclude={"projection_sha256", "facts"},
        ),
        facts=(
            prompt_inputs[0].context_artifact.facts[0].model_copy(
                update={"statement": "A forged but locally well-typed fact."}
            ),
        ),
    )
    forged_payload = TaskGraphSemanticVerificationPromptPayload.create(
        **request.prompt_payload.model_dump(
            mode="python",
            exclude={"payload_sha256", "context_artifacts"},
        ),
        context_artifacts=(forged_projection,),
    )
    forged_policy = semantic_store.derive_auxiliary_semantic_review_policy(
        prompt_payload=forged_payload
    )
    forged_request = TaskGraphSemanticVerificationRequest.create(
        **request.model_dump(
            mode="python",
            exclude={
                "binding_sha256",
                "prompt_payload",
                "review_policy",
                "task_graph_proposal_sha256",
                "blocking_gap_aliases",
                "non_blocking_gap_aliases",
                "verification_request_id",
                "logical_call_id",
            },
        ),
        verification_request_id="semantic-request-context-forged",
        logical_call_id="semantic-call-context-forged",
        prompt_payload=forged_payload,
        review_policy=forged_policy,
    )
    with pytest.raises(semantic_store.AuxiliarySemanticVerificationPersistenceError):
        semantic_store.commit_auxiliary_semantic_verification_request(
            command=_request_command(
                session_id,
                turn_id,
                details,
                forged_request,
            )
        )
    with store._connect() as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM "
            "insession_auxiliary_semantic_verification_requests"
        ).fetchone()[0] == 1
        assert conn.execute(
            "SELECT COUNT(*) FROM "
            "insession_auxiliary_semantic_request_context_artifacts"
        ).fetchone()[0] == 1
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
