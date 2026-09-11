from __future__ import annotations

import hashlib
import sqlite3
import json
from collections.abc import Callable
from types import SimpleNamespace

import pytest

from personagraph.session import store
from personagraph.session.l2_store import task_graph as task_graph_store
from personagraph.session.l2_store import terminal as terminal_store
from personagraph.session.l2_store import verification as verification_store
from personagraph.session.l2_store import work_run as work_run_store
from personagraph.session.l2_store import auxiliary_graph as auxiliary_graph_store
from personagraph.session.l2_store import semantic_verification as semantic_store
from personagraph.session.l2_store import task_delivery as task_delivery_store
from personagraph.l2.auxiliary_graph import (
    PlanningGoalPromptContext,
    TaskGraphSemanticBaseSnapshot,
    TaskGraphRevisionCandidate,
    TaskGraphSemanticLineageProjection,
    TaskGraphSemanticVerificationPromptPayload,
    TaskGraphSemanticVerificationRequest,
    required_task_graph_semantic_reviewer_count,
)
from personagraph.l2.task_graph import (
    InSessionTaskGraphRevisionProposal,
    TaskDeliveryValidationDimension,
    TaskGraphRevisionTrigger,
)
from personagraph.model_io.endpoint_identity import (
    configured_structured_model_facts,
)
from personagraph.model_io.gateway import ModelResult
from personagraph.l2.auxiliary_execution.delivery.composition import (
    AuxiliaryTaskDeliveryPorts,
    AuxiliaryTaskDeliveryRequest,
    AuxiliaryTaskDeliveryStatus,
    run_auxiliary_committed_task_to_delivery,
)
from personagraph.l2.auxiliary_execution.driver import (
    canonical_auxiliary_graph_driver_state_guard,
)
from personagraph.l2.auxiliary_execution.work_run.controller import (
    AuxiliaryWorkRunStatus,
    AuxiliaryWorkRunRequest,
    run_auxiliary_model_node,
)
from personagraph.l2.task_execution.task_graph.controller import (
    TaskGraphWorkRunProfile,
    TaskGraphWorkRunRequest,
    run_task_graph_work_runs,
)
from personagraph.session.persistence.l2.auxiliary_graph import auxiliary_graphs
from personagraph.session.persistence.l2.auxiliary_graph import (
    auxiliary_task_graph_commit as task_graph_commit_records,
)
from personagraph.session.persistence.l2.delivery import (
    task_delivery_validation as task_delivery_validation_records,
)
from personagraph.l2.work_run import (
    CurrentTaskNodeDeliveryResolutionKind,
    TaskNodeSubject,
)
from tests.runtime.test_auxiliary_work_run_controller import (
    _PassVerifier,
    _ReplyProvider,
    _acceptance,
    _ids,
    _profile,
)
from tests.session.test_auxiliary_semantic_verification_persistence import (
    _authority_projection,
    _catalog,
    _freeze_command,
    _request_command,
    _result,
    _result_command,
    _settlement_command,
)
from tests.session.test_auxiliary_terminal_seal_persistence import (
    _seal_command,
    _settled_terminal,
)
from tests.helpers.prepared_model_provider import as_prepared_test_provider


def _commit_command(prefix: str):
    seal_command, _settlement = _settled_terminal(prefix)
    sealed = terminal_store.seal_auxiliary_terminal_proposal(command=seal_command)
    with store._connect() as conn:
        task = conn.execute(
            "SELECT current_graph_revision, state_version FROM insession_tasks "
            "WHERE insession_task_id=?",
            (seal_command.task_id,),
        ).fetchone()
    window = store.get_turn_execution_window(seal_command.session_id)
    assert task is not None and window is not None
    return terminal_store.CommitAuxiliaryTaskGraphProposalCommand(
        apply_id=f"{prefix}-task-graph-commit",
        session_id=seal_command.session_id,
        source_turn_id=seal_command.invocation_turn_id,
        task_id=seal_command.task_id,
        terminal_proposal_receipt_id=sealed.terminal_proposal_receipt_id,
        expected_base_task_graph_revision=(
            int(task["current_graph_revision"])
            if task["current_graph_revision"] is not None
            else None
        ),
        expected_task_state_version=int(task["state_version"]),
        expected_window_revision=int(window["state_version"]),
    )


def _revision_proposal() -> InSessionTaskGraphRevisionProposal:
    return InSessionTaskGraphRevisionProposal.model_validate(
        {
            "root": {
                "root_key": "root",
                "nodes": [
                    {
                        "node_key": "root",
                        "node_kind": "root",
                        "title": "执行计划",
                        "objective": "交付一份可核对的完整执行计划",
                        "source_anchor_ids": ["task_creation_source"],
                        "acceptance_criteria": [
                            {
                                "acceptance_id": "plan_ready",
                                "criterion": "执行计划完整且可核对",
                                "source_anchor_ids": ["task_creation_source"],
                            }
                        ],
                    },
                    {
                        "node_key": "release",
                        "node_kind": "subtask",
                        "parent_node_key": "root",
                        "title": "发布",
                        "objective": "完成正式发布与验收",
                        "source_anchor_ids": ["task_creation_source"],
                        "acceptance_criteria": [
                            {
                                "acceptance_id": "release_ready",
                                "criterion": "发布结果可核对",
                                "source_anchor_ids": ["task_creation_source"],
                            }
                        ],
                    },
                ],
            }
        }
    )


def _default_revision_lineage() -> tuple[
    TaskGraphSemanticLineageProjection, ...
]:
    return (
        TaskGraphSemanticLineageProjection(
            proposal_node_key="root",
            disposition="reuse",
            base_node_alias="base_node_000",
        ),
        TaskGraphSemanticLineageProjection(
            proposal_node_key="release",
            disposition="new",
        ),
    )


def _submit_revision(
    proposal: InSessionTaskGraphRevisionProposal,
    lineage: tuple[TaskGraphSemanticLineageProjection, ...],
) -> str:
    return json.dumps(
        {
            "acceptance_updates": [
                {
                    "acceptance_id": "grounded",
                    "model_claimed_satisfied": True,
                }
            ],
            "action": {
                "kind": "submit_task_graph",
                "proposal": proposal.model_dump(mode="json"),
                "lineage": [item.model_dump(mode="json") for item in lineage],
            },
        },
        ensure_ascii=False,
    )


def _settle_task_revision_trigger(
    *,
    prefix: str,
    session_id: str,
    turn_id: str,
    task_id: str,
) -> TaskGraphRevisionTrigger:
    configured_provider, configured_model = configured_structured_model_facts()

    def candidate_provider(
        _system_prompt: str,
        _user_content: str,
        *,
        model_call_id: str,
        purpose: str,
    ) -> ModelResult:
        findings = []
        for dimension in TaskDeliveryValidationDimension:
            is_gap = (
                dimension is TaskDeliveryValidationDimension.GOAL_COMPLETENESS
            )
            findings.append(
                {
                    "dimension": dimension.value,
                    "verdict": "fail" if is_gap else "pass",
                    "fault_domain": "task_graph_design" if is_gap else "none",
                    "finding": (
                        "The final body omits the requested release step."
                        if is_gap
                        else f"{dimension.value} is satisfied."
                    ),
                    "affected_node_ids": [task_id] if is_gap else [],
                    "evidence_anchor_ids": [],
                }
            )
        return ModelResult(
            reply=json.dumps(
                {
                    "findings": findings,
                    "summary": "Revision required.",
                    "execution_repair_objective": None,
                    "task_graph_revision_objective": (
                        "Preserve correct content and add the missing release step."
                    ),
                    "blocking_questions": [],
                },
                ensure_ascii=False,
            ),
            provider=configured_provider,
            model=configured_model,
            latency_ms=1,
            model_call_id=model_call_id,
            purpose=purpose,
        )

    result = run_auxiliary_committed_task_to_delivery(
        AuxiliaryTaskDeliveryRequest(
            session_id=session_id,
            turn_id=turn_id,
            task_id=task_id,
        ),
        ports=AuxiliaryTaskDeliveryPorts(
            model_ledger_store=store,
            emit=lambda _event: None,
            task_candidate_validation_provider=as_prepared_test_provider(
                candidate_provider
            ),
        ),
    )
    assert result.status is AuxiliaryTaskDeliveryStatus.REVISION_REQUIRED
    trigger = task_delivery_store.get_active_task_graph_revision_trigger(
        session_id=session_id,
        task_id=task_id,
    )
    assert trigger is not None
    return trigger


def _settled_positive_base(
    prefix: str,
    *,
    prepare_base: Callable[[object, object, str], None] | None = None,
    revision_proposal: InSessionTaskGraphRevisionProposal | None = None,
    revision_lineage: tuple[
        TaskGraphSemanticLineageProjection, ...
    ] | None = None,
    goal_objective: str = "在保留根身份的同时增加发布步骤",
    revision_reason: str = "manual_replan",
):
    first_command = _commit_command(f"{prefix}-base")
    first = terminal_store.commit_auxiliary_task_graph_proposal(command=first_command)
    base = task_graph_store.get_insession_task_details(
        first_command.session_id,
        first_command.task_id,
    )
    assert base is not None and base.current_graph_revision == 1
    assert len(base.nodes) == 1
    base_node_id = str(base.nodes[0]["insession_task_node_id"])
    if prepare_base is not None:
        prepare_base(first_command, first, base_node_id)
        base = task_graph_store.get_insession_task_details(
            first_command.session_id,
            first_command.task_id,
        )
        assert base is not None and base.current_graph_revision == 1
    base_projection = task_graph_store.project_task_graph_semantic_base(
        session_id=first_command.session_id,
        task_id=first_command.task_id,
        graph_revision=1,
    )
    assert base_projection.snapshot.root_node_alias == "base_node_000"
    current = auxiliary_graphs.get_auxiliary_graph_for_task(
        store._deps(),
        session_id=first_command.session_id,
        insession_task_id=first_command.task_id,
    )
    assert current is not None
    acceptance = _acceptance()
    terminal = auxiliary_graphs.AuxiliaryGraphNodeProposalRecord(
        local_node_key="synthesize",
        node_kind="synthesize",
        executor_kind="terminal_planner",
        title="修订任务图",
        objective="形成下一版完整 TaskGraph 快照",
        source_anchor_ids=("task_creation_source",),
        acceptance_criteria=(acceptance,),
        output_contract="task_graph_revision_proposal_v2",
    )
    auxiliary_graphs.commit_auxiliary_graph_revision(
        store._deps(),
        session_id=first_command.session_id,
        turn_id=first_command.source_turn_id,
        insession_task_id=first_command.task_id,
        expected_task_state_version=base.task_state_version,
        expected_base_task_graph_revision=1,
        expected_control_state_version=current.control_state_version,
        expected_current_auxiliary_graph_revision=(
            current.auxiliary_graph_revision
        ),
        apply_id=f"{prefix}-aux-revision",
        goal_objective=goal_objective,
        proposal=auxiliary_graphs.AuxiliaryGraphRevisionProposalRecord(
            revision_reason=revision_reason,
            terminal_node_key="synthesize",
            nodes=(terminal,),
            edges=(),
        ),
        authority_context={"anchors": []},
        budget_profile={"profile_id": "planning-test-v1"},
        auxiliary_graph_id=current.auxiliary_graph_id,
        goal_id=f"{prefix}-goal-2",
    )
    details = auxiliary_graphs.get_auxiliary_graph_for_task(
        store._deps(),
        session_id=first_command.session_id,
        insession_task_id=first_command.task_id,
    )
    assert details is not None and details.budget is not None
    frontier = auxiliary_graph_store.project_auxiliary_graph_execution_frontier(
        session_id=first_command.session_id,
        turn_id=first_command.source_turn_id,
        insession_task_id=first_command.task_id,
    )
    candidate = frontier.ready_fresh[0]
    proposal = revision_proposal or _revision_proposal()
    lineage = revision_lineage or _default_revision_lineage()
    with store._connect() as conn:
        task_authority = conn.execute(
            "SELECT current_graph_revision, current_status, state_version, "
            "created_turn_id, creation_source_start, creation_source_end, "
            "creation_source_sha256 FROM insession_tasks "
            "WHERE session_id=? AND insession_task_id=?",
            (first_command.session_id, first_command.task_id),
        ).fetchone()
        assert task_authority is not None
        validation_context = auxiliary_graphs._terminal_graph_validation_context(
            conn,
            session_id=first_command.session_id,
            invocation_turn_id=first_command.source_turn_id,
            task_id=first_command.task_id,
            task_authority=task_authority,
        )
    runtime_request = AuxiliaryWorkRunRequest(
        session_id=first_command.session_id,
        turn_id=first_command.source_turn_id,
        subject=candidate.subject,
        executor_kind="terminal_planner",
        initial_driver_state_guard_sha256=(
            canonical_auxiliary_graph_driver_state_guard(frontier)
        ),
        id_plan=_ids(f"{prefix}-terminal"),
        task_graph_validation_context=validation_context,
        task_graph_semantic_base_snapshot=base_projection.snapshot,
    )
    completed = run_auxiliary_model_node(
        runtime_request,
        profile=_profile(),
        capability_catalogs={},
        attempt_provider=_ReplyProvider([_submit_revision(proposal, lineage)]),
        verification_provider=_PassVerifier(),
        emit=lambda _event: None,
        monotonic_clock=iter(range(1, 100)).__next__,
    )
    assert completed.status is AuxiliaryWorkRunStatus.COMPLETED
    details = auxiliary_graphs.get_auxiliary_graph_for_task(
        store._deps(),
        session_id=first_command.session_id,
        insession_task_id=first_command.task_id,
    )
    assert details is not None and details.budget is not None
    with store._connect() as conn:
        objective = str(
            conn.execute(
                "SELECT objective FROM insession_auxiliary_graph_goals "
                "WHERE goal_id=?",
                (details.goal_id,),
            ).fetchone()[0]
        )
    catalog_template = _catalog(protected=False)
    catalog = type(catalog_template).create(
        capability_catalog_snapshot_id=f"{prefix}-capabilities",
        capability_catalog_snapshot_sha256="e" * 64,
        capabilities=catalog_template.capabilities,
    )
    semantic_store.freeze_auxiliary_semantic_capability_catalog(
        command=_freeze_command(
            first_command.session_id,
            first_command.source_turn_id,
            first_command.task_id,
            details,
            catalog,
        )
    )
    payload = TaskGraphSemanticVerificationPromptPayload.create(
        goal=PlanningGoalPromptContext(
            goal_id=details.goal_id,
            objective=objective,
            desired_output="A complete next TaskGraph snapshot.",
            authorization_aliases=("task_creation_source",),
        ),
        authority=_authority_projection(details, ()),
        base_task_graph=base_projection.snapshot,
        capabilities=catalog,
        budget=details.budget,
        task_graph_proposal=proposal,
        lineage=lineage,
    )
    policy = semantic_store.derive_auxiliary_semantic_review_policy(
        prompt_payload=payload
    )
    reviewer_count = required_task_graph_semantic_reviewer_count(
        prompt_payload=payload,
        review_policy=policy,
    )
    requests = tuple(
        TaskGraphSemanticVerificationRequest.create(
            verification_request_id=f"{prefix}-semantic-request-{ordinal}",
            logical_call_id=f"{prefix}-semantic-call-{ordinal}",
            verification_profile_id="semantic-verifier-v1",
            reviewer_ordinal=ordinal,
            required_reviewer_count=reviewer_count,
            goal=details.goal,
            auxiliary_graph_revision=details.auxiliary_graph_revision,
            auxiliary_graph_structure_sha256=details.structure_sha256,
            prompt_payload=payload,
            review_policy=policy,
        )
        for ordinal in range(1, reviewer_count + 1)
    )
    for request in requests:
        semantic_store.commit_auxiliary_semantic_verification_request(
            command=_request_command(
                first_command.session_id,
                first_command.source_turn_id,
                details,
                request,
            )
        )
    semantic_results = tuple(
        _result(
            request,
            f"{prefix}-semantic-result-{request.reviewer_ordinal}",
        )
        for request in requests
    )
    for semantic_result in semantic_results:
        semantic_store.commit_auxiliary_semantic_verification_result(
            command=_result_command(
                first_command.session_id,
                first_command.source_turn_id,
                details,
                semantic_result,
            )
        )
    settlement = semantic_store.settle_auxiliary_semantic_verification_quorum(
        command=_settlement_command(
            first_command.session_id,
            first_command.source_turn_id,
            details,
            requests,
            semantic_results,
            f"{prefix}-semantic-settlement",
        )
    ).settlement
    seal_command = _seal_command(
        prefix=f"{prefix}-positive",
        session_id=first_command.session_id,
        turn_id=first_command.source_turn_id,
        task_id=first_command.task_id,
        details=details,
        semantic_settlement_id=settlement.settlement_id,
        semantic_prompt_payload_sha256=(
            settlement.frozen_prompt_payload_sha256
        ),
    )
    sealed = terminal_store.seal_auxiliary_terminal_proposal(command=seal_command)
    current_task = task_graph_store.get_insession_task_details(
        first_command.session_id,
        first_command.task_id,
    )
    window = store.get_turn_execution_window(first_command.session_id)
    assert current_task is not None and window is not None
    command = terminal_store.CommitAuxiliaryTaskGraphProposalCommand(
        apply_id=f"{prefix}-positive-commit",
        session_id=first_command.session_id,
        source_turn_id=first_command.source_turn_id,
        task_id=first_command.task_id,
        terminal_proposal_receipt_id=sealed.terminal_proposal_receipt_id,
        expected_base_task_graph_revision=1,
        expected_task_state_version=current_task.task_state_version,
        expected_window_revision=int(window["state_version"]),
        base_node_alias_bindings=base_projection.base_node_alias_bindings,
    )
    return command, base_node_id


def test_positive_commit_alias_authentication_accepts_multiple_sources_in_durable_order(
) -> None:
    acceptance = _acceptance()
    snapshot = TaskGraphSemanticBaseSnapshot.create(
        base_task_graph_revision=1,
        root_node_alias="base_node_000",
        nodes=(
            {
                "node_alias": "base_node_000",
                "node_revision": 1,
                "node_kind": "root",
                "parent_node_alias": None,
                "title": "现有执行计划",
                "objective": "交付一份来源可核对的执行计划",
                "source_anchor_aliases": (
                    "document_01_obs_001",
                    "task_creation_source",
                ),
                "acceptance_criteria": (acceptance,),
                "constraints": (),
                "status": "proposed",
            },
        ),
        source_snapshot_sha256="a" * 64,
    )
    command = terminal_store.CommitAuxiliaryTaskGraphProposalCommand(
        apply_id="multi-source-positive-commit",
        session_id="multi-source-session",
        source_turn_id="multi-source-turn",
        task_id="multi-source-task",
        terminal_proposal_receipt_id="multi-source-terminal-receipt",
        expected_base_task_graph_revision=1,
        expected_task_state_version=1,
        expected_window_revision=1,
        base_node_alias_bindings=(
            terminal_store.AuxiliaryBaseNodeAliasBinding(
                node_alias="base_node_000",
                insession_task_node_id="durable-root",
            ),
        ),
    )
    durable_base = SimpleNamespace(
        nodes=(
            {
                "insession_task_node_id": "durable-root",
                "node_revision": 1,
                "node_kind": "root",
                "parent_insession_task_node_id": None,
                "title": "现有执行计划",
                "objective": "交付一份来源可核对的执行计划",
    # 存储保留来源顺序；语义契约使用别名升序。
                "source_anchor_ids": (
                    "task_creation_source",
                    "document_01_obs_001",
                ),
                "acceptance_criteria": (
                    acceptance.model_dump(mode="json"),
                ),
                "constraints": (),
                "status": "proposed",
            },
        )
    )

    bindings = task_graph_commit_records._authenticate_base_alias_bindings(
        command,
        request=SimpleNamespace(base_task_graph=snapshot),
        base=durable_base,
    )

    assert bindings == {"base_node_000": "durable-root"}


def _insert_base_release_node(
    *,
    task_id: str,
    base_node_id: str,
    child_id: str,
) -> None:
    release = _revision_proposal().root.nodes[1]
    now = "2026-08-23T12:00:00+00:00"
    with store._connect() as conn:
        conn.execute(
            "INSERT INTO insession_task_graph_nodes "
            "(insession_task_id, graph_revision, insession_task_node_id, "
            "node_revision, node_kind, ordinal, title, objective, "
            "source_anchor_ids_json, acceptance_criteria_json, "
            "constraints_json, created_at) VALUES (?, 1, ?, 1, 'subtask', "
            "1, ?, ?, ?, ?, '[]', ?)",
            (
                task_id,
                child_id,
                release.title,
                release.objective,
                json.dumps(list(release.source_anchor_ids)),
                json.dumps(
                    [
                        item.model_dump(mode="json")
                        for item in release.acceptance_criteria
                    ],
                    ensure_ascii=False,
                ),
                now,
            ),
        )
        conn.execute(
            "INSERT INTO insession_task_node_states "
            "(insession_task_id, insession_task_node_id, node_revision, "
            "status, state_version, updated_at) "
            "VALUES (?, ?, 1, 'proposed', 1, ?)",
            (task_id, child_id, now),
        )
        conn.execute(
            "INSERT INTO insession_task_graph_edges "
            "(insession_task_id, graph_revision, "
            "child_insession_task_node_id, parent_insession_task_node_id, "
            "ordinal) VALUES (?, 1, ?, ?, 1)",
            (task_id, child_id, base_node_id),
        )


def _completed_child_positive_base(prefix: str):
    child_id = f"{prefix}-release-node"

    def prepare_base(first_command, _first, base_node_id: str) -> None:
        _insert_base_release_node(
            task_id=first_command.task_id,
            base_node_id=base_node_id,
            child_id=child_id,
        )
        window = store.get_turn_execution_window(first_command.session_id)
        assert window is not None
        child_only = run_task_graph_work_runs(
            TaskGraphWorkRunRequest(
                session_id=first_command.session_id,
                turn_id=first_command.source_turn_id,
                task_id=first_command.task_id,
                expected_window_revision=int(window["state_version"]),
                profile=TaskGraphWorkRunProfile(
                    max_work_runs_per_turn=1
                ),
            ),
            monotonic_clock=iter(range(1, 500)).__next__,
        )
        assert child_only.status == "turn_limit_reached"
        assert len(child_only.work_run_ids) == 1
        current = task_graph_store.get_insession_task_details(
            first_command.session_id,
            first_command.task_id,
        )
        assert current is not None
        status_by_id = {
            str(item["insession_task_node_id"]): str(item["status"])
            for item in current.nodes
        }
        assert status_by_id[child_id] == "completed"
        assert status_by_id[base_node_id] == "proposed"

    proposal_json = _revision_proposal().model_dump(mode="json")
    proposal_json["root"]["nodes"][0]["objective"] = (
        "交付一份吸收已验证发布结果的修订执行计划"
    )
    proposal = InSessionTaskGraphRevisionProposal.model_validate(proposal_json)
    lineage = (
        TaskGraphSemanticLineageProjection(
            proposal_node_key="root",
            disposition="revise",
            base_node_alias="base_node_000",
        ),
        TaskGraphSemanticLineageProjection(
            proposal_node_key="release",
            disposition="reuse",
            base_node_alias="base_node_001",
        ),
    )
    command, root_id = _settled_positive_base(
        prefix,
        prepare_base=prepare_base,
        revision_proposal=proposal,
        revision_lineage=lineage,
        goal_objective="修订根节点并可信复用已完成发布节点",
    )
    return command, root_id, child_id


def _triggered_positive_base(
    prefix: str,
    *,
    reuse_failed_root: bool = False,
):
    child_id = None if reuse_failed_root else f"{prefix}-release-node"
    trigger_holder: list[TaskGraphRevisionTrigger] = []

    def prepare_base(first_command, _first, base_node_id: str) -> None:
        if child_id is not None:
            _insert_base_release_node(
                task_id=first_command.task_id,
                base_node_id=base_node_id,
                child_id=child_id,
            )
        trigger_holder.append(
            _settle_task_revision_trigger(
                prefix=prefix,
                session_id=first_command.session_id,
                turn_id=first_command.source_turn_id,
                task_id=first_command.task_id,
            )
        )

    if reuse_failed_root:
        root = _revision_proposal().root.nodes[0]
        proposal = InSessionTaskGraphRevisionProposal.model_validate(
            {
                "root": {
                    "root_key": "root",
                    "nodes": [root.model_dump(mode="json")],
                }
            }
        )
        lineage = (
            TaskGraphSemanticLineageProjection(
                proposal_node_key="root",
                disposition="reuse",
                base_node_alias="base_node_000",
            ),
        )
    else:
        proposal_payload = _revision_proposal().model_dump(mode="json")
        proposal_payload["root"]["nodes"][0]["objective"] = (
            "交付一份修复完整交付缺口的第二版执行计划"
        )
        proposal = InSessionTaskGraphRevisionProposal.model_validate(
            proposal_payload
        )
        lineage = (
            TaskGraphSemanticLineageProjection(
                proposal_node_key="root",
                disposition="revise",
                base_node_alias="base_node_000",
            ),
            TaskGraphSemanticLineageProjection(
                proposal_node_key="release",
                disposition="reuse",
                base_node_alias="base_node_001",
            ),
        )
    command, root_id = _settled_positive_base(
        prefix,
        prepare_base=prepare_base,
        revision_proposal=proposal,
        revision_lineage=lineage,
        goal_objective=(
            "Preserve correct content and add the missing release step."
        ),
        revision_reason="verification_failed",
    )
    assert len(trigger_holder) == 1
    trigger = trigger_holder[0]
    return (
        command.model_copy(
            update={
                "task_graph_revision_trigger_id": trigger.trigger_id,
                "expected_task_graph_revision_trigger_sha256": (
                    trigger.trigger_sha256
                ),
            }
        ),
        trigger,
        root_id,
        child_id,
    )


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def test_task_graph_commit_null_to_one_is_atomic_and_exactly_replayable() -> None:
    command = _commit_command("v70-null-pass")
    current_payload = command.model_dump(mode="json")
    assert current_payload["task_graph_execution_replan_request_id"] is None
    assert (
        current_payload[
            "expected_task_graph_execution_replan_request_sha256"
        ]
        is None
    )
    current_hash = _canonical_sha256(current_payload)
    old_payload = dict(current_payload)
    old_payload.pop("task_graph_execution_replan_request_id")
    old_payload.pop("expected_task_graph_execution_replan_request_sha256")
    old_hash = _canonical_sha256(old_payload)
    assert old_hash != current_hash

    applied = terminal_store.commit_auxiliary_task_graph_proposal(command=command)
    assert applied.status == "applied"
    assert applied.previous_graph_revision is None
    assert applied.committed_graph_revision == 1
    assert applied.transition_sha256 is None
    assert applied.carry_receipt_ids == ()

    replayed = terminal_store.commit_auxiliary_task_graph_proposal(command=command)
    assert replayed == applied.model_copy(update={"status": "replayed"})

    details = task_graph_store.get_insession_task_details(command.session_id, command.task_id)
    assert details is not None
    assert details.current_graph_revision == 1
    assert len(details.nodes) == 1
    with store._connect() as conn:
        goal = conn.execute(
            "SELECT status FROM insession_auxiliary_graph_goals "
            "WHERE insession_task_id=?",
            (command.task_id,),
        ).fetchone()
        revision = conn.execute(
            "SELECT state.status FROM "
            "insession_auxiliary_graph_revision_states_v2 AS state "
            "JOIN insession_auxiliary_graph_revision_snapshots AS snapshot "
            "ON snapshot.auxiliary_graph_id=state.auxiliary_graph_id "
            "AND snapshot.auxiliary_graph_revision=state.auxiliary_graph_revision "
            "WHERE snapshot.insession_task_id=?",
            (command.task_id,),
        ).fetchone()
        assert tuple(goal) == ("committed",)
        assert tuple(revision) == ("committed",)
        assert conn.execute(
            "SELECT command_sha256 FROM "
            "insession_auxiliary_v2_task_graph_commit_receipts "
            "WHERE apply_id=?",
            (command.apply_id,),
        ).fetchone()[0] == current_hash
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []

    with store._connect() as conn:
        conn.execute(
            "UPDATE insession_auxiliary_v2_task_graph_commit_receipts "
            "SET command_sha256=? WHERE apply_id=?",
            (old_hash, command.apply_id),
        )
    with pytest.raises(terminal_store.AuxiliaryTaskGraphCommitIdentityCollision):
        terminal_store.commit_auxiliary_task_graph_proposal(command=command)


def test_task_graph_commit_rejects_base_drift_and_apply_collision() -> None:
    command = _commit_command("v70-guards")
    with pytest.raises(
        terminal_store.AuxiliaryTaskGraphCommitPersistenceError
    ) as drift:
        terminal_store.commit_auxiliary_task_graph_proposal(
            command=command.model_copy(
                update={"expected_base_task_graph_revision": 1}
            )
        )
    assert drift.value.code == "base_task_graph_drift"

    terminal_store.commit_auxiliary_task_graph_proposal(command=command)
    with pytest.raises(terminal_store.AuxiliaryTaskGraphCommitIdentityCollision):
        terminal_store.commit_auxiliary_task_graph_proposal(
            command=command.model_copy(
                update={"expected_window_revision": command.expected_window_revision + 1}
            )
        )


def test_task_graph_commit_replay_rejects_raw_graph_tamper() -> None:
    command = _commit_command("v70-raw-tamper")
    result = terminal_store.commit_auxiliary_task_graph_proposal(command=command)
    with store._connect() as conn:
        conn.execute("PRAGMA foreign_keys=OFF")
        conn.execute(
            "UPDATE insession_task_graph_nodes SET objective='tampered' "
            "WHERE insession_task_id=? AND graph_revision=?",
            (command.task_id, result.committed_graph_revision),
        )
        conn.execute("PRAGMA foreign_keys=ON")

    with pytest.raises(
        terminal_store.AuxiliaryTaskGraphCommitPersistenceError
    ) as corrupt:
        terminal_store.commit_auxiliary_task_graph_proposal(command=command)
    assert corrupt.value.code == "stored_authority_corrupt"


def test_task_graph_commit_rejects_rehashed_finish_context_rewrite() -> None:
    command = _commit_command("v70-context-rewrite")
    with store._connect() as conn:
        row = conn.execute(
            "SELECT finish.finish_gate_receipt_id, finish.receipt_json FROM "
            "insession_auxiliary_v2_finish_gate_receipts AS finish JOIN "
            "insession_auxiliary_v2_terminal_proposal_receipts AS terminal "
            "ON terminal.finish_gate_receipt_id=finish.finish_gate_receipt_id "
            "WHERE terminal.terminal_proposal_receipt_id=?",
            (command.terminal_proposal_receipt_id,),
        ).fetchone()
        assert row is not None
        receipt = json.loads(str(row["receipt_json"]))
        receipt["validation_context"]["source_anchors"][0][
            "source_turn_id"
        ] = "turn_forged"
        context_json = json.dumps(
            receipt["validation_context"],
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        context_sha256 = hashlib.sha256(context_json.encode("utf-8")).hexdigest()
        receipt["validation_context_sha256"] = context_sha256
        receipt_json = json.dumps(
            receipt,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        conn.execute(
            "UPDATE insession_auxiliary_v2_finish_gate_receipts SET "
            "validation_context_sha256=?, receipt_json=?, receipt_sha256=? "
            "WHERE finish_gate_receipt_id=?",
            (
                context_sha256,
                receipt_json,
                hashlib.sha256(receipt_json.encode("utf-8")).hexdigest(),
                str(row["finish_gate_receipt_id"]),
            ),
        )

    with pytest.raises(terminal_store.AuxiliaryTaskGraphCommitPersistenceError) as corrupt:
        terminal_store.commit_auxiliary_task_graph_proposal(command=command)
    assert corrupt.value.code == "production_evaluation_failed"


def test_task_graph_commit_late_failure_rolls_back_every_projection() -> None:
    command = _commit_command("v70-rollback")
    with store._connect() as conn:
        conn.execute(
            "CREATE TRIGGER reject_v70_commit BEFORE INSERT ON "
            "insession_auxiliary_v2_task_graph_commit_receipts "
            "BEGIN SELECT RAISE(ABORT, 'injected late failure'); END"
        )

    with pytest.raises(sqlite3.IntegrityError, match="injected late failure"):
        terminal_store.commit_auxiliary_task_graph_proposal(command=command)

    with store._connect() as conn:
        task = conn.execute(
            "SELECT current_graph_revision, state_version FROM insession_tasks "
            "WHERE insession_task_id=?",
            (command.task_id,),
        ).fetchone()
        goal = conn.execute(
            "SELECT status FROM insession_auxiliary_graph_goals "
            "WHERE insession_task_id=?",
            (command.task_id,),
        ).fetchone()
        assert tuple(task) == (None, command.expected_task_state_version)
        assert tuple(goal) == ("proposal_ready",)
        assert conn.execute(
            "SELECT COUNT(*) FROM insession_task_graph_revisions "
            "WHERE insession_task_id=?",
            (command.task_id,),
        ).fetchone()[0] == 0
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []


def test_task_graph_commit_positive_base_preserves_identity_and_history() -> None:
    command, base_root_id = _settled_positive_base("v70-positive")

    sealed = terminal_store.get_auxiliary_terminal_proposal_receipt(
        session_id=command.session_id,
        terminal_proposal_receipt_id=command.terminal_proposal_receipt_id,
    )
    assert tuple(item.disposition.value for item in sealed.lineage) == (
        "reuse",
        "new",
    )
    assert sealed.output_window.content == json.dumps(
        {
            "schema_version": "task-graph-revision-candidate-v1",
            "proposal": sealed.proposal.model_dump(mode="json"),
            "lineage": [item.model_dump(mode="json") for item in sealed.lineage],
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )

    result = terminal_store.commit_auxiliary_task_graph_proposal(command=command)
    assert result.status == "applied"
    assert result.previous_graph_revision == 1
    assert result.committed_graph_revision == 2
    assert result.transition_sha256 is not None
    assert result.carry_receipt_ids == ()

    with store._connect() as conn:
        roots = conn.execute(
            "SELECT graph_revision, insession_task_node_id, node_revision "
            "FROM insession_task_graph_nodes WHERE insession_task_id=? "
            "AND node_kind='root' ORDER BY graph_revision",
            (command.task_id,),
        ).fetchall()
        target_nodes = conn.execute(
            "SELECT insession_task_node_id, node_revision, node_kind, ordinal "
            "FROM insession_task_graph_nodes WHERE insession_task_id=? "
            "AND graph_revision=2 ORDER BY ordinal",
            (command.task_id,),
        ).fetchall()
        assert [tuple(row) for row in roots] == [
            (1, base_root_id, 1),
            (2, base_root_id, 1),
        ]
        assert len(target_nodes) == 2
        assert tuple(target_nodes[0]) == (base_root_id, 1, "root", 0)
        assert tuple(target_nodes[1])[1:] == (1, "subtask", 1)
        assert conn.execute(
            "SELECT COUNT(*) FROM insession_task_node_states "
            "WHERE insession_task_id=? AND insession_task_node_id=? "
            "AND node_revision=1",
            (command.task_id, base_root_id),
        ).fetchone()[0] == 1
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []

    assert terminal_store.commit_auxiliary_task_graph_proposal(
        command=command
    ) == result.model_copy(update={"status": "replayed"})


def test_positive_terminal_rehashed_lineage_tamper_fails_get_and_commit() -> None:
    command, _base_root_id = _settled_positive_base("v70-lineage-tamper")

    with store._connect() as conn:
        row = conn.execute(
            "SELECT terminal.output_window_json, completion.completion_id, "
            "completion.completion_json, output.work_run_id, "
            "output.output_revision FROM "
            "insession_auxiliary_v2_terminal_proposal_receipts AS terminal "
            "JOIN insession_auxiliary_node_completions_v2 AS completion "
            "ON completion.completion_id=terminal.terminal_completion_id "
            "JOIN insession_work_run_output_windows AS output "
            "ON output.work_run_id=completion.work_run_id "
            "AND output.output_revision=completion.output_revision "
            "WHERE terminal.terminal_proposal_receipt_id=?",
            (command.terminal_proposal_receipt_id,),
        ).fetchone()
        assert row is not None
        output_payload = json.loads(str(row["output_window_json"]))
        candidate = TaskGraphRevisionCandidate.model_validate_json(
            output_payload["content"]
        )
        tampered_lineage = (
            TaskGraphSemanticLineageProjection(
                proposal_node_key=candidate.lineage[0].proposal_node_key,
                disposition="revise",
                base_node_alias=candidate.lineage[0].base_node_alias,
            ),
            *candidate.lineage[1:],
        )
        tampered_candidate = TaskGraphRevisionCandidate(
            proposal=candidate.proposal,
            lineage=tampered_lineage,
        )
        output_payload["content"] = tampered_candidate.model_dump_json()
        output_json = json.dumps(
            output_payload,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        output_sha256 = hashlib.sha256(output_json.encode("utf-8")).hexdigest()
        completion_payload = json.loads(str(row["completion_json"]))
        completion_payload["output_snapshot_sha256"] = output_sha256
        completion_json = json.dumps(
            completion_payload,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        conn.execute(
            "UPDATE insession_work_run_output_windows SET snapshot_json=?, "
            "snapshot_hash=? WHERE work_run_id=? AND output_revision=?",
            (
                output_json,
                output_sha256,
                str(row["work_run_id"]),
                int(row["output_revision"]),
            ),
        )
        conn.execute(
            "UPDATE insession_auxiliary_node_completions_v2 SET "
            "completion_json=?, completion_sha256=? WHERE completion_id=?",
            (
                completion_json,
                hashlib.sha256(completion_json.encode("utf-8")).hexdigest(),
                str(row["completion_id"]),
            ),
        )
        conn.execute(
            "UPDATE insession_auxiliary_v2_terminal_proposal_receipts SET "
            "output_window_json=?, output_window_sha256=? "
            "WHERE terminal_proposal_receipt_id=?",
            (
                output_json,
                output_sha256,
                command.terminal_proposal_receipt_id,
            ),
        )

    with pytest.raises(
        terminal_store.AuxiliaryTerminalSealPersistenceError
    ) as corrupt_receipt:
        terminal_store.get_auxiliary_terminal_proposal_receipt(
            session_id=command.session_id,
            terminal_proposal_receipt_id=command.terminal_proposal_receipt_id,
        )
    assert corrupt_receipt.value.code == "stored_authority_corrupt"

    with pytest.raises(
        terminal_store.AuxiliaryTaskGraphCommitPersistenceError
    ) as rejected_commit:
        terminal_store.commit_auxiliary_task_graph_proposal(command=command)
    assert rejected_commit.value.code == "stored_authority_corrupt"


def test_task_graph_commit_completed_reuse_without_carry_fails_closed() -> None:
    command, base_root_id = _settled_positive_base("v70-carry-unsafe")
    with store._connect() as conn:
        conn.execute(
            "UPDATE insession_task_node_states SET status='completed', "
            "state_version=state_version+1 WHERE insession_task_id=? "
            "AND insession_task_node_id=? AND node_revision=1",
            (command.task_id, base_root_id),
        )

    with pytest.raises(
        terminal_store.AuxiliaryTaskGraphCommitPersistenceError
    ) as unsafe:
        terminal_store.commit_auxiliary_task_graph_proposal(command=command)
    assert unsafe.value.code in {"base_alias_binding_invalid", "carry_unsafe"}
    with store._connect() as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM insession_task_graph_revisions "
            "WHERE insession_task_id=? AND graph_revision=2",
            (command.task_id,),
        ).fetchone()[0] == 0
        assert conn.execute(
            "SELECT COUNT(*) FROM "
            "insession_auxiliary_v2_task_graph_node_carry_receipts "
            "WHERE apply_id=?",
            (command.apply_id,),
        ).fetchone()[0] == 0

def test_completed_reuse_carry_executes_revision_two_and_passes_finish_gate() -> None:
    command, root_id, child_id = _completed_child_positive_base(
        "v70-real-completed-carry"
    )
    committed = terminal_store.commit_auxiliary_task_graph_proposal(command=command)

    assert committed.committed_graph_revision == 2
    assert len(committed.carry_receipt_ids) == 1
    details = task_graph_store.get_insession_task_details(command.session_id, command.task_id)
    assert details is not None and details.current_graph_revision == 2
    node_by_id = {
        str(item["insession_task_node_id"]): item for item in details.nodes
    }
    assert int(node_by_id[root_id]["node_revision"]) == 2
    assert str(node_by_id[root_id]["status"]) == "proposed"
    assert int(node_by_id[child_id]["node_revision"]) == 1
    assert str(node_by_id[child_id]["status"]) == "completed"

    root_subject = TaskNodeSubject(
        task_id=command.task_id,
        graph_revision=2,
        node_id=root_id,
        node_revision=2,
    )
    dependencies = work_run_store.get_current_task_node_dependency_deliveries(
        session_id=command.session_id,
        subject=root_subject,
    )
    assert len(dependencies) == 1
    carried = dependencies[0]
    assert (
        carried.resolution_kind
        is CurrentTaskNodeDeliveryResolutionKind.CARRIED
    )
    assert carried.target_subject == TaskNodeSubject(
        task_id=command.task_id,
        graph_revision=2,
        node_id=child_id,
        node_revision=1,
    )
    assert carried.source_delivery.delivery.subject == carried.target_subject.model_copy(
        update={"graph_revision": 1}
    )
    assert carried.carry_authority is not None
    assert (
        carried.carry_authority.carry_receipt_id
        == committed.carry_receipt_ids[0]
    )

    window = store.get_turn_execution_window(command.session_id)
    assert window is not None
    completed = run_task_graph_work_runs(
        TaskGraphWorkRunRequest(
            session_id=command.session_id,
            turn_id=command.source_turn_id,
            task_id=command.task_id,
            expected_window_revision=int(window["state_version"]),
        ),
        monotonic_clock=iter(range(1, 500)).__next__,
    )

    assert completed.status == "completed"
    assert completed.final_delivery_id is not None
    final = verification_store.get_task_node_delivery(
        session_id=command.session_id,
        delivery_id=completed.final_delivery_id,
    )
    assert final.delivery.subject == root_subject
    assert work_run_store.get_completed_task_final_delivery_id(
        session_id=command.session_id,
        task_id=command.task_id,
    ) == completed.final_delivery_id


def test_completed_reuse_carry_receipt_tamper_fails_closed() -> None:
    command, root_id, _child_id = _completed_child_positive_base(
        "v70-carry-receipt-tamper"
    )
    committed = terminal_store.commit_auxiliary_task_graph_proposal(command=command)
    with store._connect() as conn:
        row = conn.execute(
            "SELECT receipt_json FROM "
            "insession_auxiliary_v2_task_graph_node_carry_receipts "
            "WHERE carry_receipt_id=?",
            (committed.carry_receipt_ids[0],),
        ).fetchone()
        assert row is not None
        receipt = json.loads(str(row["receipt_json"]))
        receipt["dependency_delivery_ids"] = ["forged-dependency"]
        receipt_json = json.dumps(
            receipt,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        conn.execute(
            "UPDATE insession_auxiliary_v2_task_graph_node_carry_receipts "
            "SET dependency_delivery_ids_json='[\"forged-dependency\"]', "
            "receipt_json=?, receipt_sha256=? WHERE carry_receipt_id=?",
            (
                receipt_json,
                hashlib.sha256(receipt_json.encode("utf-8")).hexdigest(),
                committed.carry_receipt_ids[0],
            ),
        )

    with pytest.raises(
        work_run_store.WorkExecutionPersistenceError,
        match="carry source verification is stale",
    ):
        work_run_store.get_current_task_node_dependency_deliveries(
            session_id=command.session_id,
            subject=TaskNodeSubject(
                task_id=command.task_id,
                graph_revision=2,
                node_id=root_id,
                node_revision=2,
            ),
        )


def test_completed_reuse_carry_wrong_target_fails_closed() -> None:
    command, root_id, _child_id = _completed_child_positive_base(
        "v70-carry-wrong-target"
    )
    committed = terminal_store.commit_auxiliary_task_graph_proposal(command=command)
    with store._connect() as conn:
        conn.execute("PRAGMA foreign_keys=OFF")
        conn.execute(
            "UPDATE insession_auxiliary_v2_task_graph_node_carry_receipts "
            "SET base_task_graph_revision=2, target_task_graph_revision=3 "
            "WHERE carry_receipt_id=?",
            (committed.carry_receipt_ids[0],),
        )
        conn.execute("PRAGMA foreign_keys=ON")

    with pytest.raises(
        work_run_store.WorkExecutionPersistenceError,
        match="no unique Delivery or carry receipt",
    ):
        work_run_store.get_current_task_node_dependency_deliveries(
            session_id=command.session_id,
            subject=TaskNodeSubject(
                task_id=command.task_id,
                graph_revision=2,
                node_id=root_id,
                node_revision=2,
            ),
        )


def test_triggered_positive_commit_atomically_consumes_and_exactly_replays() -> None:
    command, trigger, _root_id, child_id = _triggered_positive_base(
        "v72-trigger-atomic"
    )
    assert child_id is not None

    applied = terminal_store.commit_auxiliary_task_graph_proposal(command=command)

    assert applied.committed_graph_revision == trigger.target_graph_revision
    assert len(applied.carry_receipt_ids) == 1
    assert task_delivery_store.get_active_task_graph_revision_trigger(
        session_id=command.session_id,
        task_id=command.task_id,
    ) is None
    with store._connect() as conn:
        application_row = conn.execute(
            "SELECT * FROM "
            "insession_task_graph_revision_trigger_applications "
            "WHERE trigger_id=?",
            (trigger.trigger_id,),
        ).fetchone()
        assert application_row is not None
        application = (
            task_delivery_validation_records._application_from_row(
                application_row
            )
        )
        assert application.trigger_sha256 == trigger.trigger_sha256
        assert application.base_graph_revision == trigger.base_graph_revision
        assert (
            application.committed_graph_revision
            == trigger.target_graph_revision
        )
        assert application.task_graph_commit_apply_id == command.apply_id
        assert conn.execute(
            "SELECT COUNT(*) FROM "
            "insession_task_graph_revision_trigger_applications "
            "WHERE trigger_id=?",
            (trigger.trigger_id,),
        ).fetchone()[0] == 1
    assert terminal_store.commit_auxiliary_task_graph_proposal(
        command=command
    ) == applied.model_copy(update={"status": "replayed"})


def test_triggered_positive_commit_rejects_forged_trigger_cas_without_mutation() -> None:
    command, trigger, _root_id, _child_id = _triggered_positive_base(
        "v72-trigger-forged"
    )

    forged_commands = (
        command.model_copy(
            update={
                "task_graph_revision_trigger_id": None,
                "expected_task_graph_revision_trigger_sha256": None,
            }
        ),
        command.model_copy(
            update={"task_graph_revision_trigger_id": "forged-trigger"}
        ),
        command.model_copy(
            update={
                "expected_task_graph_revision_trigger_sha256": "f" * 64
            }
        ),
    )
    for forged in forged_commands:
        with pytest.raises(
            terminal_store.AuxiliaryTaskGraphCommitPersistenceError
        ) as rejected:
            terminal_store.commit_auxiliary_task_graph_proposal(command=forged)
        assert rejected.value.code == "authority_not_current"

    assert task_delivery_store.get_active_task_graph_revision_trigger(
        session_id=command.session_id,
        task_id=command.task_id,
    ) == trigger
    with store._connect() as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM insession_task_graph_revisions "
            "WHERE insession_task_id=? AND graph_revision=2",
            (command.task_id,),
        ).fetchone()[0] == 0
        assert conn.execute(
            "SELECT COUNT(*) FROM "
            "insession_task_graph_revision_trigger_applications "
            "WHERE trigger_id=?",
            (trigger.trigger_id,),
        ).fetchone()[0] == 0

def test_triggered_positive_commit_reexecutes_failed_root_without_carry() -> None:
    command, trigger, root_id, child_id = _triggered_positive_base(
        "v72-trigger-root-reuse",
        reuse_failed_root=True,
    )
    assert child_id is None

    applied = terminal_store.commit_auxiliary_task_graph_proposal(command=command)
    assert applied.status == "applied"
    assert applied.committed_graph_revision == 2
    assert applied.carry_receipt_ids == ()

    assert task_delivery_store.get_active_task_graph_revision_trigger(
        session_id=command.session_id,
        task_id=command.task_id,
    ) is None
    details = task_graph_store.get_insession_task_details(command.session_id, command.task_id)
    assert details is not None
    root = next(
        item
        for item in details.nodes
        if item["insession_task_node_id"] == root_id
    )
    assert root["node_revision"] == 2
    assert root["status"] == "proposed"
    with store._connect() as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM insession_task_graph_revisions "
            "WHERE insession_task_id=? AND graph_revision=2",
            (command.task_id,),
        ).fetchone()[0] == 1
        assert conn.execute(
            "SELECT COUNT(*) FROM "
            "insession_task_graph_revision_trigger_applications "
            "WHERE trigger_id=?",
            (trigger.trigger_id,),
        ).fetchone()[0] == 1


def test_triggered_positive_commit_rejects_wrong_active_target_atomically() -> None:
    command, trigger, _root_id, _child_id = _triggered_positive_base(
        "v72-trigger-wrong-target"
    )
    with store._connect() as conn:
        conn.execute(
            "UPDATE insession_active_task_graph_revision_triggers "
            "SET base_graph_revision=2, target_graph_revision=3 "
            "WHERE trigger_id=?",
            (trigger.trigger_id,),
        )

    with pytest.raises(
        terminal_store.AuxiliaryTaskGraphCommitPersistenceError
    ) as rejected:
        terminal_store.commit_auxiliary_task_graph_proposal(command=command)
    assert rejected.value.code == "stored_authority_corrupt"

    with store._connect() as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM insession_task_graph_revisions "
            "WHERE insession_task_id=? AND graph_revision=2",
            (command.task_id,),
        ).fetchone()[0] == 0
        assert conn.execute(
            "SELECT COUNT(*) FROM "
            "insession_auxiliary_v2_task_graph_commit_receipts "
            "WHERE apply_id=?",
            (command.apply_id,),
        ).fetchone()[0] == 0
        assert conn.execute(
            "SELECT COUNT(*) FROM "
            "insession_task_graph_revision_trigger_applications "
            "WHERE trigger_id=?",
            (trigger.trigger_id,),
        ).fetchone()[0] == 0


@pytest.mark.parametrize(
    "tamper_kind",
    ("settlement", "root_delivery", "reopened_authority"),
)
def test_triggered_positive_commit_authenticates_full_trigger_source_chain(
    tamper_kind: str,
) -> None:
    command, trigger, _root_id, _child_id = _triggered_positive_base(
        f"v72-trigger-authority-{tamper_kind}"
    )
    if tamper_kind == "settlement":
        with store._connect() as conn:
            conn.execute(
                "UPDATE insession_task_delivery_validation_settlements "
                "SET settlement_json='{}' WHERE settlement_id=?",
                (trigger.settlement_id,),
            )
    else:
        values = trigger.model_dump(
            mode="python",
            exclude={"trigger_sha256"},
        )
        values["gap_diagnosis"] = trigger.gap_diagnosis
        if tamper_kind == "root_delivery":
            values["root_delivery_id"] = "forged-root-delivery"
        else:
            values["reopened_task_state_version"] = (
                trigger.reopened_task_state_version + 100
            )
        tampered = TaskGraphRevisionTrigger.create(**values)
        trigger_json = json.dumps(
            tampered.model_dump(mode="json"),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        with store._connect() as conn:
            if tamper_kind == "root_delivery":
                conn.execute("PRAGMA foreign_keys=OFF")
            conn.execute(
                "UPDATE insession_task_graph_revision_triggers SET "
                "root_delivery_id=?, reopened_task_state_version=?, "
                "trigger_sha256=?, trigger_json=? WHERE trigger_id=?",
                (
                    tampered.root_delivery_id,
                    tampered.reopened_task_state_version,
                    tampered.trigger_sha256,
                    trigger_json,
                    trigger.trigger_id,
                ),
            )
        command = command.model_copy(
            update={
                "expected_task_graph_revision_trigger_sha256": (
                    tampered.trigger_sha256
                )
            }
        )

    with pytest.raises(
        terminal_store.AuxiliaryTaskGraphCommitPersistenceError
    ) as rejected:
        terminal_store.commit_auxiliary_task_graph_proposal(command=command)
    assert rejected.value.code == "stored_authority_corrupt"
    with store._connect() as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM insession_task_graph_revisions "
            "WHERE insession_task_id=? AND graph_revision=2",
            (command.task_id,),
        ).fetchone()[0] == 0
        assert conn.execute(
            "SELECT COUNT(*) FROM "
            "insession_task_graph_revision_trigger_applications "
            "WHERE trigger_id=?",
            (trigger.trigger_id,),
        ).fetchone()[0] == 0
