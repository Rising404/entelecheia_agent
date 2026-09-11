from __future__ import annotations

from dataclasses import replace

import pytest

from personagraph.l2.task_graph.contracts import (
    InSessionTaskAcceptanceProposal,
)
from personagraph.l2.task_graph.task_matching import InSessionTaskMatchesProposal
from personagraph.l2.planning.invocation_contracts import (
    FrozenPlanningContextArtifactBinding,
)
from personagraph.l2.planning.resource_perception import (
    FrozenPlanningResource,
    PlanningResourceCoverage,
    PlanningResourceFormat,
    PlanningResourcePerceptionRequest,
    PlanningResourceReadRequest,
    freeze_planning_resource_perception_invocation,
)
from personagraph.session import store
from personagraph.session.l2_store import task_graph as task_graph_store
from personagraph.session.l2_store import auxiliary_graph as auxiliary_graph_store
from personagraph.session.l2_store import planning as planning_store
from personagraph.session.persistence.l2.auxiliary_graph import auxiliary_graphs
from personagraph.session.persistence import schema


USER_TEXT = "请读取文档材料并形成一份受来源约束的任务图"
SHA_D = "d" * 64


def _window_revision(session_id: str) -> int:
    window = store.get_turn_execution_window(session_id)
    assert window is not None
    return int(window["state_version"])


def _seed_current_host_node():
    session_id = store.create_session("Entelecheia")
    accepted = store.accept_turn_execution(
        session_id=session_id,
        client_request_id="primitive-reservation-shell",
        source="primitive_reservation_test",
        user_text=USER_TEXT,
        lease_owner="primitive-reservation-test",
    )
    turn_id = str(accepted["turn"]["turn_id"])
    applied = task_graph_store.apply_insession_task_matches(
        session_id=session_id,
        source_turn_id=turn_id,
        apply_id="primitive-reservation-match",
        proposal=InSessionTaskMatchesProposal.model_validate(
            {
                "task_matches": [
                    {
                        "match_type": "new_root",
                        "local_key": "document",
                        "title": "读取文档",
                        "objective": "读取文档并形成任务图",
                        "source_excerpt": USER_TEXT,
                    }
                ]
            }
        ),
        exposed_catalog_ids=(),
        expected_window_revision=_window_revision(session_id),
    )
    task_id = applied.created_insession_task_ids_by_local_key["document"]
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
                title="读取文档",
                objective="读取并核对文档材料",
                source_anchor_ids=("task_creation_source",),
                acceptance_criteria=(acceptance,),
                output_contract="planning_context_artifact_v1",
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
                capability_profile_id=None,
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
        apply_id="primitive-reservation-graph",
        goal_objective="形成可执行且受来源约束的任务图",
        proposal=proposal,
        authority_context={"anchors": []},
        budget_profile={"profile_id": "planning-test-v1"},
        auxiliary_graph_id="primitive-reservation-aux",
        goal_id="primitive-reservation-goal",
    )
    frontier = auxiliary_graph_store.project_auxiliary_graph_execution_frontier(
        session_id=session_id,
        turn_id=turn_id,
        insession_task_id=task_id,
    )
    assert len(frontier.ready_fresh) == 1
    return session_id, turn_id, task_id, frontier


def _resource_request(
    binding: FrozenPlanningContextArtifactBinding,
) -> PlanningResourcePerceptionRequest:
    return PlanningResourcePerceptionRequest(
        binding=binding,
        read_request=PlanningResourceReadRequest(
            resource=FrozenPlanningResource(
                session_id=binding.session_id,
                resource_alias="document_001",
                resource_id="private-document-1",
                resource_version="version-1",
                content_sha256="c" * 64,
                coverage=PlanningResourceCoverage.COMPLETE,
                resource_format=PlanningResourceFormat.PDF,
                media_type="application/pdf",
                file_extension=".pdf",
            )
        ),
    )


def _invocation(
    *,
    primitive_call_id: str = "primitive-call-01",
    artifact_id: str = "primitive-artifact-01",
    verification_receipt_id: str = "primitive-verification-01",
):
    session_id, turn_id, task_id, frontier = _seed_current_host_node()
    candidate = frontier.ready_fresh[0]
    binding = FrozenPlanningContextArtifactBinding(
        session_id=session_id,
        task_id=task_id,
        auxiliary_graph_id=frontier.auxiliary_graph_id,
        goal_id=frontier.goal_id,
        producer_auxiliary_node=candidate.subject,
        primitive_call_id=primitive_call_id,
        artifact_id=artifact_id,
        verification_receipt_id=verification_receipt_id,
        authority_snapshot_id=frontier.authority_snapshot_id,
        scope_snapshot_sha256=SHA_D,
        alias_prefix="document",
        artifact_alias="document_context",
        producer_node_alias="observe",
    )
    request = _resource_request(binding)
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
    return session_id, invocation


def test_planning_primitive_reserves_before_io_and_exactly_replays():
    session_id, invocation = _invocation()

    applied = planning_store.reserve_planning_primitive_invocation(invocation=invocation)
    replayed = planning_store.reserve_planning_primitive_invocation(invocation=invocation)
    loaded = planning_store.get_planning_primitive_invocation(
        session_id=session_id,
        primitive_call_id=invocation.binding.primitive_call_id,
    )
    assert applied.status == "applied"
    assert replayed.status == "replayed"
    assert applied.record == replayed.record == loaded
    assert applied.record.status == "reserved"
    planning_store.require_planning_primitive_invocation_current(invocation=invocation)
    frontier = auxiliary_graph_store.project_auxiliary_graph_execution_frontier(
        session_id=session_id,
        turn_id=invocation.invocation_turn_id,
        insession_task_id=invocation.binding.task_id,
    )
    assert frontier.ready_fresh == ()
    assert frontier.recoverable == ()
    assert len(frontier.recoverable_primitive) == 1
    pending = frontier.recoverable_primitive[0]
    assert pending.subject == invocation.binding.producer_auxiliary_node
    assert pending.primitive_call_id == invocation.binding.primitive_call_id
    assert pending.primitive_kind == invocation.primitive_kind.value
    with store._connect() as conn:
        assert int(conn.execute("PRAGMA user_version").fetchone()[0]) == schema.SCHEMA_VERSION
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []


def test_planning_primitive_rejects_state_drift_before_external_read():
    _session_id, invocation = _invocation()
    planning_store.reserve_planning_primitive_invocation(invocation=invocation)
    subject = invocation.binding.producer_auxiliary_node
    with store._connect() as conn:
        conn.execute(
            "UPDATE insession_auxiliary_node_states_v2 "
            "SET state_version=state_version+1 WHERE auxiliary_graph_id=? "
            "AND auxiliary_graph_revision=? AND auxiliary_node_id=? "
            "AND node_revision=?",
            (
                subject.auxiliary_graph_id,
                subject.auxiliary_graph_revision,
                subject.node_id,
                subject.node_revision,
            ),
        )

    with pytest.raises(
        planning_store.PlanningPrimitiveInvocationStateGuardRejected,
        match="node changed",
    ):
        planning_store.require_planning_primitive_invocation_current(invocation=invocation)


def test_planning_primitive_rejects_second_call_identity_for_one_node():
    _session_id, invocation = _invocation()
    planning_store.reserve_planning_primitive_invocation(invocation=invocation)
    second_binding = replace(
        invocation.binding,
        primitive_call_id="primitive-call-02",
        artifact_id="primitive-artifact-02",
        verification_receipt_id="primitive-verification-02",
    )
    second = freeze_planning_resource_perception_invocation(
        _resource_request(second_binding),
        invocation_turn_id=invocation.invocation_turn_id,
        expected_task_state_version=invocation.expected_task_state_version,
        expected_node_state_version=invocation.expected_node_state_version,
        expected_control_state_version=invocation.expected_control_state_version,
        expected_goal_state_version=invocation.expected_goal_state_version,
        expected_revision_state_version=invocation.expected_revision_state_version,
        expected_budget_state_version=invocation.expected_budget_state_version,
        authority_snapshot_sha256=invocation.authority_snapshot_sha256,
        structure_sha256=invocation.structure_sha256,
        budget_snapshot_sha256=invocation.budget_snapshot_sha256,
    )

    with pytest.raises(planning_store.PlanningPrimitiveInvocationIdentityCollision):
        planning_store.reserve_planning_primitive_invocation(invocation=second)


def test_planning_primitive_detects_tampered_invocation_at_read_time():
    session_id, invocation = _invocation()
    planning_store.reserve_planning_primitive_invocation(invocation=invocation)
    with store._connect() as conn:
        conn.execute(
            "UPDATE insession_auxiliary_planning_primitive_invocations "
            "SET invocation_json='{}' WHERE primitive_call_id=?",
            (invocation.binding.primitive_call_id,),
        )

    with pytest.raises(
        planning_store.PlanningPrimitiveInvocationPersistenceError,
        match="JSON/hash binding",
    ):
        planning_store.get_planning_primitive_invocation(
            session_id=session_id,
            primitive_call_id=invocation.binding.primitive_call_id,
        )
