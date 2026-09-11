from __future__ import annotations

import hashlib
import json

import pytest

from personagraph.l2.auxiliary_graph.dependency_projection import (
    AuxiliaryDependencyInputLimits,
)
from personagraph.l2.task_execution.attempts.decision import AttemptDecisionInputLimits
from personagraph.l2.auxiliary_execution.driver import (
    canonical_auxiliary_graph_driver_state_guard,
)
from personagraph.l2.auxiliary_execution.work_run.controller import (
    AuxiliaryWorkRunStatus,
    AuxiliaryWorkRunRequest,
    AuxiliaryWorkRunProfile,
    run_auxiliary_model_node,
)
from personagraph.l2.task_execution.verification.decision import NodeVerificationInputLimits
from personagraph.l2.task_execution.task_node.dependencies import (
    TaskNodeDependencyInputLimits,
)
from personagraph.session import store
from personagraph.session.l2_store import work_run as work_run_store
from personagraph.session.l2_store import auxiliary_graph as auxiliary_graph_store
from personagraph.session.l2_store import planning as planning_store
from personagraph.session.persistence.l2.auxiliary_graph import auxiliary_graphs
from personagraph.tools.catalog import CatalogSnapshot
from personagraph.l2.work_run import AuxiliaryNodeSubject
from tests.runtime.test_auxiliary_work_run_controller import (
    _PassVerifier,
    _ReplyProvider,
    _acceptance,
    _clock,
    _commit_graph,
    _ids,
    _interrupt_and_accept_followup_turn,
    _seed_task,
    _submit_text,
)
from tests.session.test_planning_artifact_seal_persistence import (
    _primitive,
    _seed_graph,
)


def _window_state_version(session_id: str) -> int:
    window = store.get_turn_execution_window(session_id)
    assert window is not None
    return int(window["state_version"])


def _run_ready_model(
    *,
    session_id: str,
    turn_id: str,
    task_id: str,
    prefix: str,
    execution_findings_enabled: bool = True,
) -> None:
    frontier = auxiliary_graph_store.project_auxiliary_graph_execution_frontier(
        session_id=session_id,
        turn_id=turn_id,
        insession_task_id=task_id,
    )
    candidate = frontier.ready_fresh[0]
    request = AuxiliaryWorkRunRequest(
        session_id=session_id,
        turn_id=turn_id,
        subject=candidate.subject,
        executor_kind="model_work_run",
        initial_driver_state_guard_sha256=(
            canonical_auxiliary_graph_driver_state_guard(frontier)
        ),
        id_plan=_ids(prefix),
    )
    legacy_dependencies = TaskNodeDependencyInputLimits(
        profile_id="aux-v2-test-legacy-dependencies",
        max_items=0,
        max_serialized_utf8_bytes=1_024,
    )
    profile = AuxiliaryWorkRunProfile(
        attempt_input_limits=AttemptDecisionInputLimits(
            profile_id="aux-v2-test-attempt",
            max_prior_tool_result_items=16,
            max_prior_tool_results_serialized_utf8_bytes=64_000,
            dependency_delivery_limits=legacy_dependencies,
            max_serialized_utf8_bytes=256_000,
        ),
        verification_input_limits=NodeVerificationInputLimits(
            profile_id="aux-v2-test-verification",
            max_acceptance_items=64,
            max_supporting_tool_result_items=32,
            dependency_delivery_limits=legacy_dependencies,
            max_serialized_utf8_bytes=256_000,
        ),
        dependency_input_limits=AuxiliaryDependencyInputLimits(
            profile_id="aux-v2-test-dependency-input",
            max_items=64,
            max_serialized_utf8_bytes=1_000_000,
        ),
        execution_findings_enabled=execution_findings_enabled,
    )
    result = run_auxiliary_model_node(
        request,
        profile=profile,
        capability_catalogs={
            str(candidate.capability_profile_id): CatalogSnapshot(
                revision=1,
                entries=(),
            )
        },
        attempt_provider=_ReplyProvider([_submit_text()]),
        verification_provider=_PassVerifier(),
        emit=lambda _event: None,
        monotonic_clock=_clock(),
    )
    assert result.status is AuxiliaryWorkRunStatus.COMPLETED


def _pure_model_carry_proposal(
    *,
    reason: str,
    carry_origins: bool,
) -> auxiliary_graphs.AuxiliaryGraphRevisionProposalRecord:
    acceptance = _acceptance()
    return auxiliary_graphs.AuxiliaryGraphRevisionProposalRecord(
        revision_reason=reason,
        terminal_node_key="synthesize",
        nodes=(
            auxiliary_graphs.AuxiliaryGraphNodeProposalRecord(
                local_node_key="analysis",
                node_kind="analyze",
                executor_kind="model_work_run",
                title="纯模型分析",
                objective="只依据当前任务授权形成分析结论",
                source_anchor_ids=("task_creation_source",),
                acceptance_criteria=(acceptance,),
                output_contract="planning_context_v1",
                capability_profile_id="model_analysis",
                origin_node_alias="analysis" if carry_origins else None,
            ),
            auxiliary_graphs.AuxiliaryGraphNodeProposalRecord(
                local_node_key="synthesize",
                node_kind="synthesize",
                executor_kind="terminal_planner",
                title="形成任务图",
                objective="使用纯模型分析形成 TaskGraph 提案",
                source_anchor_ids=("task_creation_source",),
                acceptance_criteria=(acceptance,),
                output_contract="task_graph_revision_proposal_v2",
                origin_node_alias="synthesize" if carry_origins else None,
            ),
        ),
        edges=(
            auxiliary_graphs.AuxiliaryGraphEdgeProposalRecord(
                dependency_node_key="analysis",
                consumer_node_key="synthesize",
            ),
        ),
    )


def test_resolver_binds_empty_bundle_to_current_consumer() -> None:
    session_id, turn_id, task_id = _seed_task()
    _commit_graph(
        session_id=session_id,
        turn_id=turn_id,
        task_id=task_id,
        terminal_only=True,
    )
    frontier = auxiliary_graph_store.project_auxiliary_graph_execution_frontier(
        session_id=session_id,
        turn_id=turn_id,
        insession_task_id=task_id,
    )
    consumer = frontier.ready_fresh[0]

    bundle = auxiliary_graph_store.resolve_auxiliary_dependencies(
        session_id=session_id,
        turn_id=turn_id,
        consumer_subject=consumer.subject,
    )

    assert bundle.consumer_subject == consumer.subject
    assert bundle.consumer_node_alias == consumer.local_node_key
    assert bundle.structure_sha256 == frontier.structure_sha256
    assert bundle.dependency_completion_ids == ()
    assert bundle.items == ()


def test_resolver_returns_verified_model_outputs_in_producer_order() -> None:
    session_id, turn_id, task_id = _seed_task()
    acceptance = _acceptance()
    proposal = auxiliary_graphs.AuxiliaryGraphRevisionProposalRecord(
        revision_reason="initial",
        terminal_node_key="synthesize",
        nodes=(
            auxiliary_graphs.AuxiliaryGraphNodeProposalRecord(
                local_node_key="research_a",
                node_kind="analyze",
                executor_kind="model_work_run",
                title="调查 A",
                objective="完成第一份受约束调查",
                source_anchor_ids=("task_creation_source",),
                acceptance_criteria=(acceptance,),
                output_contract="planning_context_v1",
                capability_profile_id="readonly_documents_v1",
            ),
            auxiliary_graphs.AuxiliaryGraphNodeProposalRecord(
                local_node_key="research_b",
                node_kind="analyze",
                executor_kind="model_work_run",
                title="调查 B",
                objective="完成第二份受约束调查",
                source_anchor_ids=("task_creation_source",),
                acceptance_criteria=(acceptance,),
                output_contract="planning_context_v1",
                capability_profile_id="readonly_documents_v1",
            ),
            auxiliary_graphs.AuxiliaryGraphNodeProposalRecord(
                local_node_key="synthesize",
                node_kind="synthesize",
                executor_kind="terminal_planner",
                title="形成任务图",
                objective="综合两份调查形成任务图",
                source_anchor_ids=("task_creation_source",),
                acceptance_criteria=(acceptance,),
                output_contract="task_graph_revision_proposal_v2",
            ),
        ),
        edges=(
            auxiliary_graphs.AuxiliaryGraphEdgeProposalRecord(
                dependency_node_key="research_b",
                consumer_node_key="synthesize",
            ),
            auxiliary_graphs.AuxiliaryGraphEdgeProposalRecord(
                dependency_node_key="research_a",
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
        apply_id="dependency-model-graph",
        goal_objective="综合两份调查",
        proposal=proposal,
        authority_context={"anchors": []},
        budget_profile={"profile_id": "dependency-model-budget-v1"},
        auxiliary_graph_id="dependency-model-aux",
        goal_id="dependency-model-goal",
    )

    _run_ready_model(
        session_id=session_id,
        turn_id=turn_id,
        task_id=task_id,
        prefix="dependency-model-a",
    )
    _run_ready_model(
        session_id=session_id,
        turn_id=turn_id,
        task_id=task_id,
        prefix="dependency-model-b",
    )
    frontier = auxiliary_graph_store.project_auxiliary_graph_execution_frontier(
        session_id=session_id,
        turn_id=turn_id,
        insession_task_id=task_id,
    )
    terminal = frontier.ready_fresh[0]

    bundle = auxiliary_graph_store.resolve_auxiliary_dependencies(
        session_id=session_id,
        turn_id=turn_id,
        consumer_subject=terminal.subject,
    )

    assert [item.producer_node_alias for item in bundle.items] == [
        "research_a",
        "research_b",
    ]
    assert [item.producer_ordinal for item in bundle.items] == [0, 1]
    assert [item.completion_id for item in bundle.items] == [
        "dependency-model-a-completion",
        "dependency-model-b-completion",
    ]
    assert all(item.dependency_kind == "model_output" for item in bundle.items)
    assert all(item.content for item in bundle.items)


def test_revision_carries_only_pure_model_completion_and_resolves_its_body() -> None:
    session_id, turn_id, task_id = _seed_task()
    initial_kwargs = {
        "session_id": session_id,
        "turn_id": turn_id,
        "insession_task_id": task_id,
        "expected_task_state_version": 1,
        "expected_base_task_graph_revision": None,
        "expected_control_state_version": None,
        "expected_current_auxiliary_graph_revision": None,
        "apply_id": "pure-model-carry-r1",
        "goal_objective": "形成受来源约束的可执行任务图",
        "proposal": _pure_model_carry_proposal(
            reason="initial",
            carry_origins=False,
        ),
        "authority_context": {"anchors": []},
        "budget_profile": {"profile_id": "pure-model-carry-budget-v1"},
        "auxiliary_graph_id": "pure-model-carry-graph",
        "goal_id": "pure-model-carry-goal",
    }
    auxiliary_graphs.commit_auxiliary_graph_revision(
        store._deps(),
        **initial_kwargs,
    )
    with store._connect() as conn:
        raw_initial_result = str(
            conn.execute(
                "SELECT result_json FROM "
                "insession_auxiliary_graph_revision_apply_receipts_v2 "
                "WHERE apply_id='pure-model-carry-r1'"
            ).fetchone()[0]
        )
    assert "carried_completion_receipt_ids" not in json.loads(
        raw_initial_result
    )

    _run_ready_model(
        session_id=session_id,
        turn_id=turn_id,
        task_id=task_id,
        prefix="pure-model-carry-source",
        execution_findings_enabled=False,
    )
    before_revision = auxiliary_graphs.get_auxiliary_graph_for_task(
        store._deps(),
        session_id=session_id,
        insession_task_id=task_id,
    )
    assert before_revision is not None
    source = next(
        node for node in before_revision.nodes if node.local_node_key == "analysis"
    )
    frontier = auxiliary_graph_store.project_auxiliary_graph_execution_frontier(
        session_id=session_id,
        turn_id=turn_id,
        insession_task_id=task_id,
    )
    revision_kwargs = {
        "session_id": session_id,
        "turn_id": turn_id,
        "insession_task_id": task_id,
        "expected_task_state_version": frontier.task_state_version,
        "expected_base_task_graph_revision": None,
        "expected_control_state_version": frontier.control_state_version,
        "expected_current_auxiliary_graph_revision": 1,
        "apply_id": "pure-model-carry-r2",
        "goal_objective": "形成受来源约束的可执行任务图",
        "proposal": _pure_model_carry_proposal(
            reason="verification_failed",
            carry_origins=True,
        ),
        "authority_context": {"anchors": []},
        "budget_profile": {"profile_id": "pure-model-carry-budget-v1"},
        "auxiliary_graph_id": "pure-model-carry-graph",
        "goal_id": "pure-model-carry-goal",
    }
    revised = auxiliary_graphs.commit_auxiliary_graph_revision(
        store._deps(),
        **revision_kwargs,
    )

    assert len(revised.carried_completion_receipt_ids) == 1
    current = auxiliary_graphs.get_auxiliary_graph_for_task(
        store._deps(),
        session_id=session_id,
        insession_task_id=task_id,
    )
    assert current is not None
    carried = next(
        node for node in current.nodes if node.local_node_key == "analysis"
    )
    assert carried.auxiliary_node_id == source.auxiliary_node_id
    assert carried.node_revision == source.node_revision + 1
    assert carried.status == "completed"

    current_frontier = auxiliary_graph_store.project_auxiliary_graph_execution_frontier(
        session_id=session_id,
        turn_id=turn_id,
        insession_task_id=task_id,
    )
    assert [item.local_node_key for item in current_frontier.ready_fresh] == [
        "synthesize"
    ]
    assert current_frontier.ready_fresh[0].dependency_completion_ids == (
        "pure-model-carry-source-completion",
    )
    bundle = auxiliary_graph_store.resolve_auxiliary_dependencies(
        session_id=session_id,
        turn_id=turn_id,
        consumer_subject=current_frontier.ready_fresh[0].subject,
    )
    assert bundle.dependency_completion_ids == (
        "pure-model-carry-source-completion",
    )
    assert bundle.items[0].content
    assert bundle.items[0].producer_subject.auxiliary_graph_revision == 2
    assert bundle.items[0].producer_subject.node_revision == 2

    replayed_revision = auxiliary_graphs.commit_auxiliary_graph_revision(
        store._deps(),
        **revision_kwargs,
    )
    assert replayed_revision.status == "replayed"
    assert (
        replayed_revision.carried_completion_receipt_ids
        == revised.carried_completion_receipt_ids
    )
    with store._connect() as conn:
        carry_row = conn.execute(
            "SELECT carry_receipt_id, receipt_json FROM "
            "insession_auxiliary_node_completion_carries_v2"
        ).fetchone()
        assert carry_row is not None
        original_receipt_json = str(carry_row["receipt_json"])
        forged_receipt = json.loads(original_receipt_json)
        forged_receipt["source_authority_projection_sha256"] = "0" * 64
        forged_receipt_json = json.dumps(
            forged_receipt,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        conn.execute(
            "UPDATE insession_auxiliary_node_completion_carries_v2 SET "
            "receipt_json=?, receipt_sha256=? WHERE carry_receipt_id=?",
            (
                forged_receipt_json,
                hashlib.sha256(forged_receipt_json.encode("utf-8")).hexdigest(),
                str(carry_row["carry_receipt_id"]),
            ),
        )
    with pytest.raises(
        auxiliary_graphs.AuxiliaryGraphPersistenceError,
        match="carry authority changed",
    ):
        auxiliary_graphs.commit_auxiliary_graph_revision(
            store._deps(),
            **revision_kwargs,
        )
    with store._connect() as conn:
        conn.execute(
            "UPDATE insession_auxiliary_node_completion_carries_v2 SET "
            "receipt_json=?, receipt_sha256=? WHERE carry_receipt_id=?",
            (
                original_receipt_json,
                hashlib.sha256(original_receipt_json.encode("utf-8")).hexdigest(),
                str(carry_row["carry_receipt_id"]),
            ),
        )
    replayed_historical = auxiliary_graphs.commit_auxiliary_graph_revision(
        store._deps(),
        **initial_kwargs,
    )
    assert replayed_historical.status == "replayed"
    assert replayed_historical.carried_completion_receipt_ids == ()

    # 携带回执有意不表示传递式完成。R3 必须重新执行节点，而不是经由 R2 串接 R1 正文。
    third_kwargs = revision_kwargs | {
        "expected_control_state_version": revised.control_state_version,
        "expected_current_auxiliary_graph_revision": 2,
        "apply_id": "pure-model-carry-r3",
    }
    third = auxiliary_graphs.commit_auxiliary_graph_revision(
        store._deps(),
        **third_kwargs,
    )
    assert third.carried_completion_receipt_ids == ()
    after_third = auxiliary_graphs.get_auxiliary_graph_for_task(
        store._deps(),
        session_id=session_id,
        insession_task_id=task_id,
    )
    assert after_third is not None
    assert next(
        node for node in after_third.nodes if node.local_node_key == "analysis"
    ).status == "proposed"
    with store._connect() as conn:
        assert int(
            conn.execute(
                "SELECT COUNT(*) FROM "
                "insession_auxiliary_node_completion_carries_v2"
            ).fetchone()[0]
        ) == 1
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
        conn.execute(
            "UPDATE insession_work_run_output_windows SET snapshot_json='{}' "
            "WHERE work_run_id='pure-model-carry-source-run'"
        )
    with pytest.raises(
        auxiliary_graphs.AuxiliaryGraphPersistenceError,
        match="completion payload is invalid",
    ):
        auxiliary_graphs.commit_auxiliary_graph_revision(
            store._deps(),
            **revision_kwargs,
        )
    assert store.purge_session(session_id) is True
    with store._connect() as conn:
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []


@pytest.mark.parametrize(
    "payload",
    (
        {"revision": 1},
        {"revision": 1, "tools": ["workspace_read"]},
        {"revision": 1, "entries": [{"tool_id": "workspace_read"}]},
    ),
)
def test_nonempty_or_undeclared_tool_catalog_is_not_carryable(
    payload: dict[str, object],
) -> None:
    raw = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    )
    assert not auxiliary_graphs._catalog_snapshot_has_no_tools(
        raw,
        hashlib.sha256(raw.encode("utf-8")).hexdigest(),
    )


@pytest.mark.parametrize(
    "boundary",
    (
        "semantic_change",
        "incoming_dependency",
        "input_resource",
        "tool_history",
        "cross_turn",
        "corrupt_source",
    ),
)
def test_pure_model_carry_boundaries_reexecute_instead_of_carrying(
    boundary: str,
) -> None:
    session_id, turn_id, task_id = _seed_task()
    prefix = f"pure-carry-boundary-{boundary}"
    first = auxiliary_graphs.commit_auxiliary_graph_revision(
        store._deps(),
        session_id=session_id,
        turn_id=turn_id,
        insession_task_id=task_id,
        expected_task_state_version=1,
        expected_base_task_graph_revision=None,
        expected_control_state_version=None,
        expected_current_auxiliary_graph_revision=None,
        apply_id=f"{prefix}-r1",
        goal_objective="验证纯模型 completion carry 边界",
        proposal=_pure_model_carry_proposal(
            reason="initial",
            carry_origins=False,
        ),
        authority_context={"anchors": []},
        budget_profile={"profile_id": "pure-model-boundary-budget-v1"},
        auxiliary_graph_id=f"{prefix}-graph",
        goal_id=f"{prefix}-goal",
    )
    _run_ready_model(
        session_id=session_id,
        turn_id=turn_id,
        task_id=task_id,
        prefix=f"{prefix}-source",
    )
    if boundary == "tool_history":
        catalog_json = json.dumps(
            {"revision": 1, "tools": ["workspace_read"]},
            sort_keys=True,
            separators=(",", ":"),
        )
        with store._connect() as conn:
            conn.execute(
                "UPDATE insession_work_run_attempts SET "
                "catalog_snapshot_json=?, catalog_snapshot_hash=? "
                "WHERE work_run_id=?",
                (
                    catalog_json,
                    hashlib.sha256(catalog_json.encode("utf-8")).hexdigest(),
                    f"{prefix}-source-run",
                ),
            )
    elif boundary == "corrupt_source":
        with store._connect() as conn:
            conn.execute(
                "UPDATE insession_work_run_output_windows SET "
                "snapshot_json='{}' WHERE work_run_id=?",
                (f"{prefix}-source-run",),
            )

    target = _pure_model_carry_proposal(
        reason="verification_failed",
        carry_origins=True,
    )
    if boundary in {"semantic_change", "input_resource"}:
        analysis = target.nodes[0]
        update = (
            {"title": "发生语义变化的分析"}
            if boundary == "semantic_change"
            else {"input_resource_aliases": ("paper",)}
        )
        target = target.model_copy(
            update={
                "nodes": (analysis.model_copy(update=update), target.nodes[1])
            }
        )
    elif boundary == "incoming_dependency":
        upstream = auxiliary_graphs.AuxiliaryGraphNodeProposalRecord(
            local_node_key="observe",
            node_kind="observe",
            executor_kind="host_primitive",
            title="补充观察",
            objective="读取新的上游事实",
            source_anchor_ids=("task_creation_source",),
            acceptance_criteria=(_acceptance(),),
            output_contract="planning_context_v1",
            capability_profile_id="readonly_documents_v1",
        )
        target = target.model_copy(
            update={
                "nodes": (upstream, *target.nodes),
                "edges": (
                    auxiliary_graphs.AuxiliaryGraphEdgeProposalRecord(
                        dependency_node_key="observe",
                        consumer_node_key="analysis",
                    ),
                    *target.edges,
                ),
            }
        )
    target_turn_id = turn_id
    if boundary == "cross_turn":
        target_turn_id = _interrupt_and_accept_followup_turn(
            session_id=session_id,
            turn_id=turn_id,
            task_id=task_id,
            client_request_id=f"{prefix}-followup",
        )
    frontier = auxiliary_graph_store.project_auxiliary_graph_execution_frontier(
        session_id=session_id,
        turn_id=target_turn_id,
        insession_task_id=task_id,
    )
    commit_kwargs = {
        "session_id": session_id,
        "turn_id": target_turn_id,
        "insession_task_id": task_id,
        "expected_task_state_version": frontier.task_state_version,
        "expected_base_task_graph_revision": None,
        "expected_control_state_version": first.control_state_version,
        "expected_current_auxiliary_graph_revision": 1,
        "apply_id": f"{prefix}-r2",
        "goal_objective": "验证纯模型 completion carry 边界",
        "proposal": target,
        "authority_context": {"anchors": []},
        "budget_profile": {"profile_id": "pure-model-boundary-budget-v1"},
        "auxiliary_graph_id": f"{prefix}-graph",
        "goal_id": f"{prefix}-goal",
    }
    if boundary == "corrupt_source":
        with pytest.raises(
            auxiliary_graphs.AuxiliaryGraphPersistenceError,
            match="completion payload is invalid",
        ):
            auxiliary_graphs.commit_auxiliary_graph_revision(
                store._deps(),
                **commit_kwargs,
            )
        rolled_back = auxiliary_graphs.get_auxiliary_graph_for_task(
            store._deps(),
            session_id=session_id,
            insession_task_id=task_id,
        )
        assert rolled_back is not None
        assert rolled_back.auxiliary_graph_revision == 1
        with store._connect() as conn:
            assert int(
                conn.execute(
                    "SELECT COUNT(*) FROM "
                    "insession_auxiliary_graph_revision_snapshots "
                    "WHERE auxiliary_graph_id=?",
                    (f"{prefix}-graph",),
                ).fetchone()[0]
            ) == 1
            assert int(
                conn.execute(
                    "SELECT COUNT(*) FROM "
                    "insession_auxiliary_node_completion_carries_v2"
                ).fetchone()[0]
            ) == 0
            assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
        return
    revised = auxiliary_graphs.commit_auxiliary_graph_revision(
        store._deps(),
        **commit_kwargs,
    )

    assert revised.carried_completion_receipt_ids == ()
    current = auxiliary_graphs.get_auxiliary_graph_for_task(
        store._deps(),
        session_id=session_id,
        insession_task_id=task_id,
    )
    assert current is not None
    assert next(
        node for node in current.nodes if node.local_node_key == "analysis"
    ).status == "proposed"
    with store._connect() as conn:
        assert int(
            conn.execute(
                "SELECT COUNT(*) FROM "
                "insession_auxiliary_node_completion_carries_v2"
            ).fetchone()[0]
        ) == 0
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []


def test_resolver_projects_sealed_host_artifact_from_observation(tmp_path) -> None:
    session_id, turn_id, task_id, command, result = _primitive(tmp_path)
    sealed = planning_store.seal_auxiliary_host_primitive_result(
        command=command,
        result=result,
    )
    frontier = auxiliary_graph_store.project_auxiliary_graph_execution_frontier(
        session_id=session_id,
        turn_id=turn_id,
        insession_task_id=task_id,
    )
    terminal = frontier.ready_fresh[0]

    bundle = auxiliary_graph_store.resolve_auxiliary_dependencies(
        session_id=session_id,
        turn_id=turn_id,
        consumer_subject=terminal.subject,
    )

    assert bundle.dependency_completion_ids == (sealed.artifact_id,)
    item = bundle.items[0]
    assert item.dependency_kind == "host_context"
    assert item.artifact == result.prompt_inputs.context_artifact
    assert item.observation_settlement_sha256 == result.settlement_sha256
    assert item.verification_receipt_sha256 == result.verification_receipt_sha256


def test_work_run_create_accepts_exact_verified_host_completion(tmp_path) -> None:
    session_id, turn_id, task_id, command, result = _primitive(tmp_path)
    planning_store.seal_auxiliary_host_primitive_result(command=command, result=result)
    frontier = auxiliary_graph_store.project_auxiliary_graph_execution_frontier(
        session_id=session_id,
        turn_id=turn_id,
        insession_task_id=task_id,
    )
    terminal = frontier.ready_fresh[0]

    created = work_run_store.create_auxiliary_node_work_run(
        session_id=session_id,
        turn_id=turn_id,
        subject=terminal.subject,
        expected_task_state_version=frontier.task_state_version,
        expected_node_state_version=terminal.node_state_version,
        expected_window_revision=_window_state_version(session_id),
        apply_id="host-dependency-consumer-create",
        work_run_id="host-dependency-consumer-run",
    )

    assert created.work_run_id == "host-dependency-consumer-run"
    assert created.node_state_version == terminal.node_state_version + 1


def test_work_run_create_rejects_unfinished_host_dependency() -> None:
    session_id, turn_id, task_id = _seed_graph()
    details = auxiliary_graphs.get_auxiliary_graph_for_task(
        store._deps(),
        session_id=session_id,
        insession_task_id=task_id,
    )
    assert details is not None
    terminal = next(
        node for node in details.nodes if node.local_node_key == "synthesize"
    )
    subject = AuxiliaryNodeSubject(
        task_id=task_id,
        auxiliary_graph_id=details.auxiliary_graph_id,
        auxiliary_graph_revision=details.auxiliary_graph_revision,
        node_id=terminal.auxiliary_node_id,
        node_revision=terminal.node_revision,
    )
    with store._connect() as conn:
        task_version = int(
            conn.execute(
                "SELECT state_version FROM insession_tasks "
                "WHERE insession_task_id=?",
                (task_id,),
            ).fetchone()[0]
        )

    with pytest.raises(
        auxiliary_graphs.AuxiliaryGraphPersistenceError,
        match="dependency is incomplete",
    ):
        work_run_store.create_auxiliary_node_work_run(
            session_id=session_id,
            turn_id=turn_id,
            subject=subject,
            expected_task_state_version=task_version,
            expected_node_state_version=terminal.state_version,
            expected_window_revision=_window_state_version(session_id),
            apply_id="unfinished-host-consumer-create",
            work_run_id="unfinished-host-consumer-run",
        )


def test_work_run_create_rejects_tampered_host_completion(tmp_path) -> None:
    session_id, turn_id, task_id, command, result = _primitive(tmp_path)
    planning_store.seal_auxiliary_host_primitive_result(command=command, result=result)
    frontier = auxiliary_graph_store.project_auxiliary_graph_execution_frontier(
        session_id=session_id,
        turn_id=turn_id,
        insession_task_id=task_id,
    )
    terminal = frontier.ready_fresh[0]
    with store._connect() as conn:
        conn.execute(
            "UPDATE insession_auxiliary_planning_context_artifacts "
            "SET artifact_json='{}' WHERE artifact_id=?",
            (command.expected_artifact_id,),
        )

    with pytest.raises(
        auxiliary_graphs.AuxiliaryGraphPersistenceError,
        match="Host completion payload is invalid",
    ):
        work_run_store.create_auxiliary_node_work_run(
            session_id=session_id,
            turn_id=turn_id,
            subject=terminal.subject,
            expected_task_state_version=frontier.task_state_version,
            expected_node_state_version=terminal.node_state_version,
            expected_window_revision=_window_state_version(session_id),
            apply_id="tampered-host-consumer-create",
            work_run_id="tampered-host-consumer-run",
        )


def test_resolver_fails_closed_on_dependency_body_tamper() -> None:
    session_id, turn_id, task_id = _seed_task()
    _commit_graph(session_id=session_id, turn_id=turn_id, task_id=task_id)
    _run_ready_model(
        session_id=session_id,
        turn_id=turn_id,
        task_id=task_id,
        prefix="dependency-tamper",
    )
    frontier = auxiliary_graph_store.project_auxiliary_graph_execution_frontier(
        session_id=session_id,
        turn_id=turn_id,
        insession_task_id=task_id,
    )
    terminal = frontier.ready_fresh[0]
    with store._connect() as conn:
        conn.execute(
            "UPDATE insession_work_run_output_windows SET snapshot_json='{}' "
            "WHERE work_run_id='dependency-tamper-run'"
        )

    with pytest.raises(
        auxiliary_graph_store.AuxiliaryDependencyPersistenceError,
        match="invalid output or verification JSON",
    ):
        auxiliary_graph_store.resolve_auxiliary_dependencies(
            session_id=session_id,
            turn_id=turn_id,
            consumer_subject=terminal.subject,
        )


def test_resolver_rejects_host_projection_tamper_even_with_new_outer_hash(
    tmp_path,
) -> None:
    session_id, turn_id, task_id, command, result = _primitive(tmp_path)
    planning_store.seal_auxiliary_host_primitive_result(command=command, result=result)
    frontier = auxiliary_graph_store.project_auxiliary_graph_execution_frontier(
        session_id=session_id,
        turn_id=turn_id,
        insession_task_id=task_id,
    )
    terminal = frontier.ready_fresh[0]
    with store._connect() as conn:
        row = conn.execute(
            "SELECT snapshot_json FROM insession_auxiliary_observations "
            "WHERE observation_id=?",
            (command.primitive_call_id,),
        ).fetchone()
        payload = json.loads(str(row[0]))
        payload["result"]["prompt_inputs"]["context_artifact"]["facts"][0][
            "statement"
        ] = "伪造的上游事实"
        forged = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        conn.execute(
            "UPDATE insession_auxiliary_observations SET snapshot_json=?, "
            "snapshot_sha256=? WHERE observation_id=?",
            (
                forged,
                hashlib.sha256(forged.encode("utf-8")).hexdigest(),
                command.primitive_call_id,
            ),
        )

    with pytest.raises(
        auxiliary_graph_store.AuxiliaryDependencyPersistenceError,
        match="invalid sealed JSON",
    ):
        auxiliary_graph_store.resolve_auxiliary_dependencies(
            session_id=session_id,
            turn_id=turn_id,
            consumer_subject=terminal.subject,
        )


def test_resolver_rejects_stale_consumer_revision() -> None:
    session_id, turn_id, task_id = _seed_task()
    _commit_graph(
        session_id=session_id,
        turn_id=turn_id,
        task_id=task_id,
        terminal_only=True,
    )
    frontier = auxiliary_graph_store.project_auxiliary_graph_execution_frontier(
        session_id=session_id,
        turn_id=turn_id,
        insession_task_id=task_id,
    )
    stale = frontier.ready_fresh[0].subject.model_copy(
        update={"auxiliary_graph_revision": 2}
    )

    with pytest.raises(
        auxiliary_graph_store.AuxiliaryDependencyPersistenceError,
        match="outside the active current graph",
    ):
        auxiliary_graph_store.resolve_auxiliary_dependencies(
            session_id=session_id,
            turn_id=turn_id,
            consumer_subject=stale,
        )
