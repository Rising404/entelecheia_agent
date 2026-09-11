from __future__ import annotations

from dataclasses import fields
from types import SimpleNamespace

import pytest

from personagraph.l2.task_graph.contracts import InSessionTaskAcceptanceProposal
from personagraph.l2.task_graph.task_matching import InSessionTaskMatchesProposal
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
from personagraph.l2.auxiliary_graph import (
    AuxiliaryNodeReference,
    PlanningAuthorityClass,
    PlanningAuthorityOriginKind,
    PlanningObservationStatus,
)
from personagraph.session import store
from personagraph.session.l2_store import task_graph as task_graph_store
from personagraph.session.l2_store import auxiliary_graph as auxiliary_graph_store
from personagraph.session.l2_store import planning as planning_store
from personagraph.session.persistence.l2.auxiliary_graph import auxiliary_graphs


_USER_TEXT = "请读取文档材料并给出一份可执行计划"


def _window_revision(session_id: str) -> int:
    window = store.get_turn_execution_window(session_id)
    assert window is not None
    return int(window["state_version"])


def _seed_graph() -> tuple[str, str, str]:
    session_id = store.create_session("Entelecheia")
    accepted = store.accept_turn_execution(
        session_id=session_id,
        client_request_id="planning-artifact-seal-shell",
        source="planning_artifact_seal_test",
        user_text=_USER_TEXT,
        lease_owner="planning-artifact-seal-test",
    )
    turn_id = str(accepted["turn"]["turn_id"])
    matched = task_graph_store.apply_insession_task_matches(
        session_id=session_id,
        source_turn_id=turn_id,
        apply_id="planning-artifact-seal-match",
        proposal=InSessionTaskMatchesProposal.model_validate(
            {
                "task_matches": [
                    {
                        "match_type": "new_root",
                        "local_key": "plan",
                        "title": "读取并规划",
                        "objective": "读取文档材料并形成执行计划",
                        "source_excerpt": _USER_TEXT,
                    }
                ]
            }
        ),
        exposed_catalog_ids=(),
        expected_window_revision=_window_revision(session_id),
    )
    task_id = matched.created_insession_task_ids_by_local_key["plan"]
    acceptance = InSessionTaskAcceptanceProposal(
        acceptance_id="source_understood",
        criterion="输出必须受任务创建来源约束",
        source_anchor_ids=("task_creation_source",),
    )
    proposal = auxiliary_graphs.AuxiliaryGraphRevisionProposalRecord(
        revision_reason="initial",
        terminal_node_key="synthesize",
        nodes=(
            auxiliary_graphs.AuxiliaryGraphNodeProposalRecord(
                local_node_key="observe",
                node_kind="observe",
                executor_kind="host_primitive",
                title="读取材料",
                objective="读取并核对材料",
                source_anchor_ids=("task_creation_source",),
                acceptance_criteria=(acceptance,),
                output_contract="planning_context_v1",
                capability_profile_id="mounted_document_read",
            ),
            auxiliary_graphs.AuxiliaryGraphNodeProposalRecord(
                local_node_key="synthesize",
                node_kind="synthesize",
                executor_kind="terminal_planner",
                title="形成任务图",
                objective="根据已核对材料形成任务图提案",
                source_anchor_ids=("task_creation_source",),
                acceptance_criteria=(acceptance,),
                output_contract="task_graph_revision_proposal_v2",
            ),
        ),
        edges=(
            auxiliary_graphs.AuxiliaryGraphEdgeProposalRecord(
                dependency_node_key="observe",
                consumer_node_key="synthesize",
            ),
        ),
    )
    auxiliary_graphs.commit_auxiliary_graph_revision(
        store._deps(),
        session_id=session_id,
        turn_id=turn_id,
        insession_task_id=task_id,
        expected_task_state_version=1,
        expected_base_task_graph_revision=None,
        expected_control_state_version=None,
        expected_current_auxiliary_graph_revision=None,
        apply_id="planning-artifact-seal-graph",
        goal_objective="形成可执行且受来源约束的任务图",
        proposal=proposal,
        authority_context={"anchors": []},
        budget_profile={"profile_id": "planning-artifact-seal-budget-v1"},
        auxiliary_graph_id="planning-artifact-seal-aux",
        goal_id="planning-artifact-seal-goal",
    )
    return session_id, turn_id, task_id


def _resource_read_outcome(
    status: PlanningObservationStatus,
) -> PlanningResourceReadOutcome:
    if status is PlanningObservationStatus.SUCCESS:
        return PlanningResourceReadOutcome(
            status=status,
            observed_resource_version="version-1",
            observed_content_sha256="c" * 64,
            observed_coverage=PlanningResourceCoverage.COMPLETE,
            evidence=(
                PlanningResourceEvidenceUnit(
                    source_unit_id="page-1",
                    statement="The document contains material needed for planning.",
                    locator="page 1",
                    content_sha256="d" * 64,
                    evidence_kind=PlanningResourceEvidenceKind.DOCUMENT_TEXT,
                    disclosure_receipt_id="document-disclosure-1",
                ),
            ),
        )
    if status is PlanningObservationStatus.BLOCKED:
        return PlanningResourceReadOutcome(
            status=status,
            observed_resource_version=None,
            observed_content_sha256=None,
            observed_coverage=None,
            gap_reasons=(PlanningResourceGapReason.ACCESS_BLOCKED,),
        )
    raise ValueError("test fixture supports only success or blocked observations")


def _primitive(
    tmp_path,
    *,
    observation_status: PlanningObservationStatus = (
        PlanningObservationStatus.SUCCESS
    ),
):
    session_id, turn_id, task_id = _seed_graph()
    frontier = auxiliary_graph_store.project_auxiliary_graph_execution_frontier(
        session_id=session_id,
        turn_id=turn_id,
        insession_task_id=task_id,
    )
    candidate = frontier.ready_fresh[0]
    binding = FrozenPlanningContextArtifactBinding(
        session_id=session_id,
        task_id=task_id,
        auxiliary_graph_id=frontier.auxiliary_graph_id,
        goal_id=frontier.goal_id,
        producer_auxiliary_node=candidate.subject,
        primitive_call_id="planning-primitive-call-1",
        artifact_id="planning-context-artifact-1",
        verification_receipt_id="planning-verification-receipt-1",
        authority_snapshot_id=frontier.authority_snapshot_id,
        scope_snapshot_sha256="a" * 64,
        alias_prefix="document",
        artifact_alias="document_artifact",
        producer_node_alias="observe",
    )
    resource = FrozenPlanningResource(
        session_id=session_id,
        resource_alias="document_001",
        resource_id="private-document-1",
        resource_version="version-1",
        content_sha256="c" * 64,
        coverage=PlanningResourceCoverage.COMPLETE,
        resource_format=PlanningResourceFormat.PDF,
        media_type="application/pdf",
        file_extension=".pdf",
    )
    request = PlanningResourcePerceptionRequest(
        binding=binding,
        read_request=PlanningResourceReadRequest(resource=resource),
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
    planning_store.require_planning_primitive_invocation_current(invocation=invocation)

    class _Port:
        def read_frozen_resource(self, read_request):
            assert read_request == request.read_request
            return _resource_read_outcome(observation_status)

    result = run_planning_resource_perception(request, read_port=_Port())
    command = planning_store.SealAuxiliaryHostPrimitiveResultCommand(
        apply_id="planning-artifact-seal-apply",
        session_id=session_id,
        invocation_turn_id=turn_id,
        task_id=task_id,
        auxiliary_graph_id=frontier.auxiliary_graph_id,
        goal_id=frontier.goal_id,
        auxiliary_graph_revision=frontier.auxiliary_graph_revision,
        auxiliary_node_id=candidate.subject.node_id,
        node_revision=candidate.subject.node_revision,
        primitive_kind=invocation.primitive_kind.value,
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
        active_seconds_delta=0.25,
    )
    return session_id, turn_id, task_id, command, result


def test_seal_host_primitive_is_atomic_and_advances_frontier(tmp_path) -> None:
    session_id, turn_id, task_id, command, result = _primitive(tmp_path)

    sealed = planning_store.seal_auxiliary_host_primitive_result(
        command=command,
        result=result,
    )

    assert sealed.status == "applied"
    assert sealed.observation_id == command.primitive_call_id
    assert sealed.completion_id == command.expected_artifact_id
    assert sealed.task_state_version == command.expected_task_state_version + 1
    assert sealed.goal_state_version == command.expected_goal_state_version + 1
    assert sealed.budget_state_version == command.expected_budget_state_version + 1
    with store._connect() as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM insession_auxiliary_observations"
        ).fetchone()[0] == 1
        assert conn.execute(
            "SELECT COUNT(*) FROM insession_auxiliary_observation_items"
        ).fetchone()[0] >= 1
        assert conn.execute(
            "SELECT COUNT(*) FROM "
            "insession_auxiliary_context_verification_receipts"
        ).fetchone()[0] == 1
        assert conn.execute(
            "SELECT COUNT(*) FROM "
            "insession_auxiliary_planning_context_artifacts"
        ).fetchone()[0] == 1
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
    frontier = auxiliary_graph_store.project_auxiliary_graph_execution_frontier(
        session_id=session_id,
        turn_id=turn_id,
        insession_task_id=task_id,
    )
    assert [item.ordinal for item in frontier.ready_fresh] == [1]
    assert frontier.ready_fresh[0].dependency_completion_ids == (
        command.expected_artifact_id,
    )
    assert frontier.completed_node_refs == (
        AuxiliaryNodeReference(
            node_id=command.auxiliary_node_id,
            node_revision=command.node_revision,
        ),
    )
    other_session_id = store.create_session("Entelecheia")
    assert store.purge_session(session_id) is True
    assert store.get_session(other_session_id) is not None
    with store._connect() as conn:
        for table in (
            "insession_auxiliary_planning_primitive_invocations",
            "insession_auxiliary_planning_context_artifacts",
            "insession_auxiliary_context_verification_receipts",
            "insession_auxiliary_observations",
        ):
            assert int(
                conn.execute(
                    f"SELECT COUNT(*) FROM {table} WHERE session_id=?",
                    (session_id,),
                ).fetchone()[0]
            ) == 0


def test_seal_host_primitive_exact_replay_and_collision(tmp_path) -> None:
    _, _, _, command, result = _primitive(
        tmp_path,
        observation_status=PlanningObservationStatus.BLOCKED,
    )
    assert result.artifact.gaps
    applied = planning_store.seal_auxiliary_host_primitive_result(
        command=command,
        result=result,
    )
    replayed = planning_store.seal_auxiliary_host_primitive_result(
        command=command,
        result=result,
    )
    assert replayed == applied.model_copy(update={"status": "replayed"})

    changed = command.model_copy(update={"active_seconds_delta": 0.5})
    with pytest.raises(planning_store.PlanningArtifactSealApplyIdCollision):
        planning_store.seal_auxiliary_host_primitive_result(
            command=changed,
            result=result,
        )


def test_seal_host_primitive_rejects_stale_state_without_partial_write(
    tmp_path,
) -> None:
    _, _, _, command, result = _primitive(tmp_path)
    stale = command.model_copy(
        update={
            "expected_task_state_version": command.expected_task_state_version + 1
        }
    )
    # Pydantic model_copy 不会重新运行验证器；使用新的应用 ID，让事务进入
    # 当前状态验证。
    stale = stale.model_copy(update={"apply_id": "planning-artifact-stale-apply"})
    with pytest.raises(planning_store.PlanningArtifactSealPersistenceError):
        planning_store.seal_auxiliary_host_primitive_result(command=stale, result=result)
    with store._connect() as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM insession_auxiliary_observations"
        ).fetchone()[0] == 0
        assert conn.execute(
            "SELECT COUNT(*) FROM insession_auxiliary_goal_budget_charges "
            "WHERE charge_key='planning-artifact-stale-apply'"
        ).fetchone()[0] == 0


def test_seal_host_primitive_rejects_observation_authorization(tmp_path) -> None:
    _, _, _, command, result = _primitive(tmp_path)
    evidence = result.authority_anchors[0]
    forged_anchor = evidence.model_copy(
        update={
            "authority_class": PlanningAuthorityClass.AUTHORIZATION,
            "origin_kind": PlanningAuthorityOriginKind.USER_INSTRUCTION_SPAN,
            "origin_id": command.invocation_turn_id,
            "span_start": 0,
            "span_end": 1,
        }
    )
    forged = SimpleNamespace(
        **{
            item.name: (
                (forged_anchor, *result.authority_anchors[1:])
                if item.name == "authority_anchors"
                else getattr(result, item.name)
            )
            for item in fields(result)
        }
    )
    with pytest.raises(
        planning_store.PlanningArtifactSealPersistenceError,
        match="cannot create authorization",
    ):
        planning_store.seal_auxiliary_host_primitive_result(command=command, result=forged)
    with store._connect() as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM insession_auxiliary_observations"
        ).fetchone()[0] == 0


def test_seal_host_primitive_replay_detects_artifact_tamper(tmp_path) -> None:
    _, _, _, command, result = _primitive(tmp_path)
    planning_store.seal_auxiliary_host_primitive_result(command=command, result=result)
    with store._connect() as conn:
        conn.execute(
            "UPDATE insession_auxiliary_planning_context_artifacts "
            "SET facts_json='[]' WHERE artifact_id=?",
            (command.expected_artifact_id,),
        )
    # facts_json 是带类型镜像；重放最终也必须验证它。
    with pytest.raises(planning_store.PlanningArtifactSealPersistenceError):
        planning_store.seal_auxiliary_host_primitive_result(command=command, result=result)


def test_seal_resource_perception_records_visual_budget_and_kind() -> None:
    session_id, turn_id, task_id = _seed_graph()
    frontier = auxiliary_graph_store.project_auxiliary_graph_execution_frontier(
        session_id=session_id,
        turn_id=turn_id,
        insession_task_id=task_id,
    )
    candidate = frontier.ready_fresh[0]
    binding = FrozenPlanningContextArtifactBinding(
        session_id=session_id,
        task_id=task_id,
        auxiliary_graph_id=frontier.auxiliary_graph_id,
        goal_id=frontier.goal_id,
        producer_auxiliary_node=candidate.subject,
        primitive_call_id="planning-resource-call-1",
        artifact_id="planning-resource-artifact-1",
        verification_receipt_id="planning-resource-verification-1",
        authority_snapshot_id=frontier.authority_snapshot_id,
        scope_snapshot_sha256="b" * 64,
        alias_prefix="image",
        artifact_alias="image_artifact",
        producer_node_alias="observe",
    )
    resource = FrozenPlanningResource(
        session_id=session_id,
        resource_alias="image_001",
        resource_id="private-image-1",
        resource_version="version-1",
        content_sha256="c" * 64,
        coverage=PlanningResourceCoverage.COMPLETE,
        resource_format=PlanningResourceFormat.PNG,
        media_type="image/png",
        file_extension=".png",
    )
    request = PlanningResourcePerceptionRequest(
        binding=binding,
        read_request=PlanningResourceReadRequest(resource=resource),
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
                status=PlanningObservationStatus.SUCCESS,
                observed_resource_version="version-1",
                observed_content_sha256="c" * 64,
                observed_coverage=PlanningResourceCoverage.COMPLETE,
                evidence=(
                    PlanningResourceEvidenceUnit(
                        source_unit_id="figure-1",
                        statement="The image contains a labelled pipeline diagram.",
                        locator="image region 1",
                        content_sha256="d" * 64,
                        evidence_kind=(
                            PlanningResourceEvidenceKind.VISUAL_OBSERVATION
                        ),
                        disclosure_receipt_id="visual-disclosure-1",
                    ),
                ),
            )

    result = run_planning_resource_perception(request, read_port=_Port())
    command = planning_store.SealAuxiliaryHostPrimitiveResultCommand(
        apply_id="planning-resource-seal-apply",
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
    sealed = planning_store.seal_auxiliary_host_primitive_result(
        command=command,
        result=result,
    )
    assert sealed.status == "applied"
    details = auxiliary_graph_store.get_auxiliary_graph_for_task(
        session_id=session_id,
        insession_task_id=task_id,
    )
    assert details is not None
    budget = details.budget
    assert budget.usage.logical_tool_calls == 1
    assert budget.usage.evidence_units == 1
    assert budget.usage.visual_units == 1
    with store._connect() as conn:
        observation = conn.execute(
            "SELECT observation_kind FROM insession_auxiliary_observations "
            "WHERE observation_id=?",
            (binding.primitive_call_id,),
        ).fetchone()
        assert str(observation["observation_kind"]) == "visual"
