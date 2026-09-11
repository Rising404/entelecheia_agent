from __future__ import annotations

import pytest

from personagraph.l2.auxiliary_graph import (
    AuxiliaryGraphRevisionReason,
    AuxiliaryReplanTriggerReceipt,
    AuxiliaryReplanTriggerReason,
    TaskGraphSemanticVerificationDimension,
    TaskGraphSemanticVerificationDisposition,
)
from personagraph.session import store
from personagraph.session.l2_store import planning as planning_store
from personagraph.session.l2_store import semantic_verification as semantic_store
from personagraph.session.persistence.l2.auxiliary_graph import auxiliary_graphs
from tests.session.test_auxiliary_semantic_verification_persistence import (
    _catalog,
    _complete_terminal_only,
    _freeze_command,
    _request_command,
    _result,
    _result_command,
    _semantic_request,
    _settlement_command,
)


def _settled_non_pass(
    prefix: str,
):
    session_id, turn_id, task_id, details, prompt_inputs = (
        _complete_terminal_only(prefix)
    )
    catalog = _catalog(protected=False)
    semantic_store.freeze_auxiliary_semantic_capability_catalog(
        command=_freeze_command(
            session_id,
            turn_id,
            task_id,
            details,
            catalog,
        )
    )
    request = _semantic_request(
        request_id=f"{prefix}-semantic-request",
        logical_call_id=f"{prefix}-semantic-call",
        reviewer_ordinal=1,
        details=details,
        catalog=catalog,
        prompt_inputs=prompt_inputs,
    )
    semantic_store.commit_auxiliary_semantic_verification_request(
        command=_request_command(session_id, turn_id, details, request)
    )
    result = _result(
        request,
        f"{prefix}-semantic-result",
        failed_dimension=(
            TaskGraphSemanticVerificationDimension.GOAL_COVERAGE
        ),
    )
    semantic_store.commit_auxiliary_semantic_verification_result(
        command=_result_command(session_id, turn_id, details, result)
    )
    settled = semantic_store.settle_auxiliary_semantic_verification_quorum(
        command=_settlement_command(
            session_id,
            turn_id,
            details,
            (request,),
            (result,),
            f"{prefix}-semantic-settlement",
        )
    ).settlement
    return session_id, turn_id, task_id, details, settled


def _create_command(prefix: str, session_id, turn_id, task_id, details, settlement):
    return planning_store.CreateAuxiliaryReplanTriggerCommand(
        apply_id=f"{prefix}-trigger-apply",
        trigger_id=f"{prefix}-trigger",
        session_id=session_id,
        created_turn_id=turn_id,
        task_id=task_id,
        auxiliary_graph_id=details.auxiliary_graph_id,
        goal_id=details.goal_id,
        expected_control_state_version=details.control_state_version,
        expected_goal_state_version=details.goal_state_version,
        expected_revision_state_version=details.revision_state_version,
        expected_budget_state_version=details.budget_state_version,
        expected_current_auxiliary_graph_revision=(
            details.auxiliary_graph_revision
        ),
        expected_current_structure_sha256=details.structure_sha256,
        expected_current_authority_snapshot_id=details.authority_snapshot_id,
        expected_current_authority_snapshot_sha256=(
            details.authority_snapshot_sha256
        ),
        semantic_settlement_id=settlement.settlement_id,
        expected_semantic_settlement_sha256=settlement.settlement_sha256,
    )


def test_non_pass_settlement_creates_one_host_derived_exact_replan_trigger() -> None:
    session_id, turn_id, task_id, details, settlement = _settled_non_pass(
        "replan-trigger"
    )
    command = _create_command(
        "replan-trigger",
        session_id,
        turn_id,
        task_id,
        details,
        settlement,
    )

    applied = planning_store.create_auxiliary_replan_trigger(command=command)

    assert applied.status == "applied"
    trigger = applied.trigger
    assert trigger.semantic_disposition is TaskGraphSemanticVerificationDisposition.REVISE
    assert trigger.trigger_reason is AuxiliaryReplanTriggerReason.SEMANTIC_REVISION_REQUIRED
    assert trigger.revision_reason is AuxiliaryGraphRevisionReason.VERIFICATION_FAILED
    assert trigger.source_auxiliary_graph_revision == details.auxiliary_graph_revision
    assert trigger.source_structure_sha256 == details.structure_sha256
    assert trigger.source_authority_snapshot_sha256 == details.authority_snapshot_sha256
    assert (
        trigger.semantic_authority_projection_sha256
        == settlement.requests[0].authority_projection.projection_sha256
    )
    assert trigger.semantic_prompt_payload_sha256 == settlement.frozen_prompt_payload_sha256
    assert trigger.task_graph_proposal_sha256 == settlement.requests[0].task_graph_proposal_sha256
    assert trigger.semantic_settlement_sha256 == settlement.settlement_sha256
    assert trigger.semantic_result_sha256s == tuple(
        item.result_sha256 for item in settlement.results
    )
    assert trigger.budget_ledger_id == details.goal.budget_ledger_id
    assert trigger.budget_snapshot_sha256 == details.budget.snapshot_sha256
    assert len(trigger.evidence_epoch_sha256) == 64
    assert planning_store.create_auxiliary_replan_trigger(
        command=command
    ) == applied.model_copy(update={"status": "replayed"})
    assert planning_store.get_active_auxiliary_replan_trigger(
        session_id=session_id,
        task_id=task_id,
    ) == trigger
    assert planning_store.get_auxiliary_replan_trigger(
        session_id=session_id,
        trigger_id=trigger.trigger_id,
    ) == trigger
    assert planning_store.count_auxiliary_replan_triggers(
        session_id=session_id,
        task_id=task_id,
        goal_id=details.goal_id,
    ) == 1

    with pytest.raises(planning_store.AuxiliaryReplanTriggerIdentityCollision):
        planning_store.create_auxiliary_replan_trigger(
            command=command.model_copy(
                update={
                    "apply_id": "replan-trigger-other-apply",
                    "trigger_id": "replan-trigger-other",
                }
            )
        )


def test_blocked_trigger_contract_preserves_typed_reason() -> None:
    session_id, turn_id, task_id, details, settlement = _settled_non_pass(
        "replan-blocked",
    )

    revise_trigger = planning_store.create_auxiliary_replan_trigger(
        command=_create_command(
            "replan-blocked",
            session_id,
            turn_id,
            task_id,
            details,
            settlement,
        )
    ).trigger
    trigger = AuxiliaryReplanTriggerReceipt.create(
        **revise_trigger.model_dump(
            mode="python",
            exclude={
                "semantic_disposition",
                "trigger_reason",
                "revision_reason",
                "evidence_epoch_sha256",
                "receipt_sha256",
            },
        ),
        semantic_disposition=TaskGraphSemanticVerificationDisposition.BLOCKED,
    )

    assert trigger.semantic_disposition is TaskGraphSemanticVerificationDisposition.BLOCKED
    assert trigger.trigger_reason is AuxiliaryReplanTriggerReason.SEMANTIC_EVIDENCE_BLOCKED


def test_pass_settlement_cannot_create_replan_trigger() -> None:
    other_session, other_turn, other_task, other_details, _ = _complete_terminal_only(
        "replan-pass"
    )
    catalog = _catalog(protected=False)
    semantic_store.freeze_auxiliary_semantic_capability_catalog(
        command=_freeze_command(
            other_session, other_turn, other_task, other_details, catalog
        )
    )
    request = _semantic_request(
        request_id="replan-pass-request",
        logical_call_id="replan-pass-call",
        reviewer_ordinal=1,
        details=other_details,
        catalog=catalog,
        prompt_inputs=(),
    )
    semantic_store.commit_auxiliary_semantic_verification_request(
        command=_request_command(other_session, other_turn, other_details, request)
    )
    result = _result(request, "replan-pass-result")
    semantic_store.commit_auxiliary_semantic_verification_result(
        command=_result_command(other_session, other_turn, other_details, result)
    )
    pass_settlement = semantic_store.settle_auxiliary_semantic_verification_quorum(
        command=_settlement_command(
            other_session,
            other_turn,
            other_details,
            (request,),
            (result,),
            "replan-pass-settlement",
        )
    ).settlement
    with pytest.raises(planning_store.AuxiliaryReplanTriggerPersistenceError):
        planning_store.create_auxiliary_replan_trigger(
            command=_create_command(
                "replan-pass",
                other_session,
                other_turn,
                other_task,
                other_details,
                pass_settlement,
            )
        )


def test_trigger_cas_and_stored_authority_tamper_fail_closed() -> None:
    session_id, turn_id, task_id, details, settlement = _settled_non_pass(
        "replan-guard"
    )
    command = _create_command(
        "replan-guard",
        session_id,
        turn_id,
        task_id,
        details,
        settlement,
    )
    with pytest.raises(planning_store.AuxiliaryReplanTriggerStaleAuthority):
        planning_store.create_auxiliary_replan_trigger(
            command=command.model_copy(
                update={
                    "expected_current_auxiliary_graph_revision": (
                        details.auxiliary_graph_revision + 1
                    )
                }
            )
        )
    assert planning_store.create_auxiliary_replan_trigger(command=command).status == "applied"
    with pytest.raises(planning_store.AuxiliaryReplanTriggerStoredAuthorityCorrupt):
        planning_store.get_auxiliary_replan_trigger(
            session_id="another-session",
            trigger_id=command.trigger_id,
        )

    with store._connect() as conn:
        conn.execute(
            "UPDATE insession_auxiliary_replan_triggers "
            "SET source_structure_sha256=? WHERE trigger_id=?",
            ("f" * 64, command.trigger_id),
        )
    with pytest.raises(planning_store.AuxiliaryReplanTriggerStoredAuthorityCorrupt):
        planning_store.get_auxiliary_replan_trigger(
            session_id=session_id,
            trigger_id=command.trigger_id,
        )


def _commit_replan_revision(
    *,
    session_id: str,
    turn_id: str,
    task_id: str,
    details,
):
    nodes = tuple(
        auxiliary_graphs.AuxiliaryGraphNodeProposalRecord(
            local_node_key=node.local_node_key,
            node_kind=node.node_kind,
            executor_kind=node.executor_kind,
            title=node.title,
            objective=node.objective,
            source_anchor_ids=node.source_anchor_ids,
            acceptance_criteria=node.acceptance_criteria,
            output_contract=node.output_contract,
            capability_profile_id=node.capability_profile_id,
            input_resource_aliases=node.input_resource_aliases,
            required=node.required,
            origin_node_alias=node.local_node_key,
        )
        for node in details.nodes
    )
    keys = {
        node.auxiliary_node_id: node.local_node_key for node in details.nodes
    }
    edges = tuple(
        auxiliary_graphs.AuxiliaryGraphEdgeProposalRecord(
            dependency_node_key=keys[edge.dependency_auxiliary_node_id],
            consumer_node_key=keys[edge.consumer_auxiliary_node_id],
        )
        for edge in details.edges
    )
    terminal_key = keys[details.terminal_auxiliary_node_id]
    with store._connect() as conn:
        task_state_version = int(
            conn.execute(
                "SELECT state_version FROM insession_tasks "
                "WHERE session_id=? AND insession_task_id=?",
                (session_id, task_id),
            ).fetchone()[0]
        )
    return auxiliary_graphs.commit_auxiliary_graph_revision(
        store._deps(),
        session_id=session_id,
        turn_id=turn_id,
        insession_task_id=task_id,
        expected_task_state_version=task_state_version,
        expected_base_task_graph_revision=details.base_task_graph_revision,
        expected_control_state_version=details.control_state_version,
        expected_current_auxiliary_graph_revision=details.auxiliary_graph_revision,
        apply_id="replan-revision-apply",
        goal_objective="形成受来源约束的可执行任务图",
        proposal=auxiliary_graphs.AuxiliaryGraphRevisionProposalRecord(
            revision_reason=AuxiliaryGraphRevisionReason.VERIFICATION_FAILED,
            terminal_node_key=terminal_key,
            nodes=nodes,
            edges=edges,
        ),
        authority_context={"anchors": []},
        budget_profile=details.budget.base_profile.model_dump(mode="json"),
        auxiliary_graph_id=details.auxiliary_graph_id,
        goal_id=details.goal_id,
    )


def test_trigger_consumption_is_durable_and_recoverable_after_revision_commit() -> None:
    session_id, turn_id, task_id, details, settlement = _settled_non_pass(
        "replan-consume"
    )
    trigger = planning_store.create_auxiliary_replan_trigger(
        command=_create_command(
            "replan-consume",
            session_id,
            turn_id,
            task_id,
            details,
            settlement,
        )
    ).trigger
    committed = _commit_replan_revision(
        session_id=session_id,
        turn_id=turn_id,
        task_id=task_id,
        details=details,
    )
    assert _commit_replan_revision(
        session_id=session_id,
        turn_id=turn_id,
        task_id=task_id,
        details=details,
    ).status == "replayed"
    assert planning_store.get_active_auxiliary_replan_trigger(
        session_id=session_id,
        task_id=task_id,
    ) == trigger
    consume = planning_store.ConsumeAuxiliaryReplanTriggerCommand(
        apply_id="replan-consume-apply",
        session_id=session_id,
        consumed_turn_id=turn_id,
        task_id=task_id,
        trigger_id=trigger.trigger_id,
        expected_goal_id=details.goal_id,
        expected_applied_auxiliary_graph_revision=(
            committed.committed_auxiliary_graph_revision
        ),
        expected_applied_structure_sha256=committed.structure_sha256,
    )

    consumed = planning_store.consume_auxiliary_replan_trigger(command=consume)

    assert consumed.status == "applied"
    assert consumed.application.trigger_id == trigger.trigger_id
    assert (
        consumed.application.applied_auxiliary_graph_revision
        == trigger.source_auxiliary_graph_revision + 1
    )
    assert planning_store.consume_auxiliary_replan_trigger(
        command=consume
    ) == consumed.model_copy(update={"status": "replayed"})
    assert planning_store.create_auxiliary_replan_trigger(
        command=_create_command(
            "replan-consume",
            session_id,
            turn_id,
            task_id,
            details,
            settlement,
        )
    ).status == "replayed"
    assert planning_store.get_active_auxiliary_replan_trigger(
        session_id=session_id,
        task_id=task_id,
    ) is None
    assert planning_store.get_auxiliary_replan_trigger_application(
        session_id=session_id,
        trigger_id=trigger.trigger_id,
    ) == consumed.application
    assert planning_store.count_auxiliary_replan_triggers(
        session_id=session_id,
        task_id=task_id,
        goal_id=details.goal_id,
    ) == 1
    with store._connect() as conn:
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
