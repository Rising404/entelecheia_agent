from __future__ import annotations

import json
from types import SimpleNamespace

from personagraph.l2.task_graph import (
    InSessionTaskAcceptanceProposal,
    InSessionTaskGraphRevisionProposal,
    InSessionTaskNodeKind,
)
from personagraph.l2.task_graph.task_matching import InSessionTaskMatchesProposal
from personagraph.l2.auxiliary_execution.planning.architect_adapter import (
    build_terminal_only_auxiliary_graph_bootstrap_proposal,
)
from personagraph.l2.auxiliary_execution.planning.mounted_document_authority import (
    build_mounted_document_authority_projection,
    freeze_mounted_document_planning_authority,
)
from personagraph.l2.auxiliary_execution.planning.profiles import (
    MODEL_ANALYSIS_CAPABILITY,
    MOUNTED_DOCUMENT_READ_CAPABILITY,
    build_auxiliary_graph_current_revision_projection,
    build_auxiliary_architect_request,
    build_auxiliary_execution_capability_catalogs,
    build_auxiliary_planning_capability_catalog,
    canonical_auxiliary_architect_state_guard,
)
from personagraph.l2.auxiliary_execution.work_run.contracts import (
    KNOWLEDGE_COGNITION_CAPABILITY,
    KNOWLEDGE_INDEXING_CAPABILITY,
)
from personagraph.l2.auxiliary_execution.planning.model_provider import (
    AuxiliaryArchitectModelProfile,
    build_mock_auxiliary_architect_proposal,
)
from personagraph.l2.auxiliary_execution.planning.architect import (
    AuxiliaryGraphArchitectReplanTrigger,
    AuxiliaryGraphReplanReviewerFindings,
    serialize_auxiliary_graph_architect_prompt,
    validate_auxiliary_graph_architect_proposal,
)
from personagraph.l2.auxiliary_graph import (
    AuxiliaryGraphRevisionProposal,
    TaskGraphSemanticVerificationDimension,
    TaskGraphSemanticVerificationDisposition,
    TaskGraphSemanticVerificationItem,
)
from personagraph.session import store
from personagraph.session.l2_store import task_graph as task_graph_store
from personagraph.session.l2_store import auxiliary_graph as auxiliary_graph_store


USER_TEXT = "请理解我挂载的材料并形成一份可执行任务图"
SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64


def test_architect_profile_covers_complex_reasoning_envelope() -> None:
    profile = AuxiliaryArchitectModelProfile()

    assert profile.max_output_tokens == 131_072
    assert profile.timeout_s == 600.0


def _window_revision(session_id: str) -> int:
    window = store.get_turn_execution_window(session_id)
    assert window is not None
    return int(window["state_version"])


def _seed_task() -> tuple[str, str, str]:
    session_id = store.create_session("Entelecheia")
    accepted = store.accept_turn_execution(
        session_id=session_id,
        client_request_id="aux-v2-planning-profile",
        source="auxiliary_v2_planning_profile_test",
        user_text=USER_TEXT,
        lease_owner="aux-v2-planning-profile-test",
    )
    turn_id = str(accepted["turn"]["turn_id"])
    applied = task_graph_store.apply_insession_task_matches(
        session_id=session_id,
        source_turn_id=turn_id,
        apply_id="aux-v2-planning-profile-task",
        proposal=InSessionTaskMatchesProposal.model_validate(
            {
                "task_matches": [
                    {
                        "match_type": "new_root",
                        "local_key": "root",
                        "title": "理解材料",
                        "objective": "理解材料并形成可执行任务图",
                        "source_excerpt": USER_TEXT,
                    }
                ]
            }
        ),
        exposed_catalog_ids=(),
        expected_window_revision=_window_revision(session_id),
    )
    return session_id, turn_id, applied.created_insession_task_ids_by_local_key["root"]


def _replan_trigger(
    *, revision: int, structure_sha256: str, blocked: bool = False
) -> AuxiliaryGraphArchitectReplanTrigger:
    rejected = InSessionTaskGraphRevisionProposal.model_validate(
        {
            "root": {
                "root_key": "answer",
                "nodes": [
                    {
                        "node_key": "answer",
                        "node_kind": InSessionTaskNodeKind.ROOT.value,
                        "parent_node_key": None,
                        "title": "回答",
                        "objective": "给出有依据的回答",
                        "source_anchor_ids": ["task_creation_source"],
                        "acceptance_criteria": [
                            {
                                "acceptance_id": "answer_grounded",
                                "criterion": "回答必须有来源依据",
                                "source_anchor_ids": ["task_creation_source"],
                            }
                        ],
                        "constraints": [],
                    }
                ],
            }
        }
    )
    return AuxiliaryGraphArchitectReplanTrigger.create(
        expected_current_auxiliary_graph_revision=revision,
        expected_current_structure_sha256=structure_sha256,
        semantic_host_disposition=(
            TaskGraphSemanticVerificationDisposition.BLOCKED
            if blocked
            else TaskGraphSemanticVerificationDisposition.REVISE
        ),
        semantic_settlement_id="semantic_settlement_profiles_01",
        semantic_settlement_sha256=SHA_A,
        semantic_prompt_payload_sha256=SHA_B,
        rejected_task_graph_proposal=rejected,
        reviewers=(
            AuxiliaryGraphReplanReviewerFindings(
                reviewer_ordinal=1,
                semantic_result_sha256=SHA_C,
                items=tuple(
                    TaskGraphSemanticVerificationItem(
                        dimension=dimension,
                        verdict=(
                            "insufficient_evidence"
                            if blocked
                            and dimension
                            is TaskGraphSemanticVerificationDimension.GOAL_COVERAGE
                            else "fail"
                            if dimension
                            is TaskGraphSemanticVerificationDimension.GOAL_COVERAGE
                            else "pass"
                        ),
                        failure_scope=(
                            "missing_authority"
                            if blocked
                            and dimension
                            is TaskGraphSemanticVerificationDimension.GOAL_COVERAGE
                            else "auxiliary_investigation"
                            if dimension
                            is TaskGraphSemanticVerificationDimension.GOAL_COVERAGE
                            else None
                        ),
                        finding=(
                            "目标覆盖不足"
                            if dimension
                            is TaskGraphSemanticVerificationDimension.GOAL_COVERAGE
                            else "该维度通过"
                        ),
                        affected_node_keys=(
                            ("answer",)
                            if dimension
                            is TaskGraphSemanticVerificationDimension.GOAL_COVERAGE
                            else ()
                        ),
                        gap_aliases=(
                            ("missing_answer_evidence",)
                            if blocked
                            and dimension
                            is TaskGraphSemanticVerificationDimension.GOAL_COVERAGE
                            else ()
                        ),
                    )
                    for dimension in TaskGraphSemanticVerificationDimension
                ),
            ),
        ),
    )


def test_production_planning_projections_round_trip_current_store_authority(
    monkeypatch,
) -> None:
    session_id, turn_id, task_id = _seed_task()
    monkeypatch.setattr(
        "personagraph.tools.documents.mounted_document_source_authority.docstore.mounted_docs",
        lambda selected_session_id: (
            () if selected_session_id == session_id else None
        ),
    )
    mounted = freeze_mounted_document_planning_authority(session_id=session_id)
    acceptance = InSessionTaskAcceptanceProposal(
        acceptance_id="bootstrap_graph_ready",
        criterion="输出必须形成受用户目标约束的完整任务图提案",
        source_anchor_ids=("task_creation_source",),
    )
    proposal = build_terminal_only_auxiliary_graph_bootstrap_proposal(
        terminal_node_key="bootstrap_terminal",
        title="建立规划权限",
        objective="在实际执行前由 Architect 形成完整调查图",
        source_anchor_ids=("task_creation_source",),
        acceptance_criteria=(acceptance,),
    )
    auxiliary_graph_store.commit_auxiliary_graph_revision(
        session_id=session_id,
        turn_id=turn_id,
        insession_task_id=task_id,
        expected_task_state_version=1,
        expected_base_task_graph_revision=None,
        expected_control_state_version=None,
        expected_current_auxiliary_graph_revision=None,
        apply_id="aux-v2-planning-bootstrap",
        goal_objective="理解材料并形成可执行任务图",
        proposal=proposal,
        authority_context=mounted.authority_context,
        budget_profile={"profile_id": "planning_episode_production_v1"},
        auxiliary_graph_id="aux-v2-planning-graph",
        goal_id="aux-v2-planning-goal",
    )

    details = auxiliary_graph_store.get_auxiliary_graph_for_task(
        session_id=session_id,
        insession_task_id=task_id,
    )
    assert details is not None and details.authority_snapshot is not None
    creation = task_graph_store.get_insession_task_creation_source(
        session_id=session_id,
        insession_task_id=task_id,
    )
    authority = build_mounted_document_authority_projection(
        authority_snapshot=details.authority_snapshot,
        task_creation_source=creation,
        mounted_authority=mounted,
    )
    capabilities = build_auxiliary_planning_capability_catalog(mounted)
    request = build_auxiliary_architect_request(
        details=details,
        authority=authority,
        capabilities=capabilities,
        objective="理解材料并形成可执行任务图",
        desired_output="完整、可执行、可验证的 TaskGraph",
    )

    assert request == build_auxiliary_architect_request(
        details=details,
        authority=authority,
        capabilities=capabilities,
        objective="理解材料并形成可执行任务图",
        desired_output="完整、可执行、可验证的 TaskGraph",
    )
    assert request.prompt_payload.current_revision == (
        build_auxiliary_graph_current_revision_projection(details)
    )
    assert request.prompt_payload.current_revision is not None
    assert request.prompt_payload.current_revision.structure.terminal_node_key == (
        "bootstrap_terminal"
    )
    assert request.prompt_payload.goal.authorization_aliases == (
        "task_creation_source",
    )
    by_alias = {
        item.capability_alias: item
        for item in request.prompt_payload.capabilities.capabilities
    }
    assert by_alias[MODEL_ANALYSIS_CAPABILITY].available is True
    assert by_alias[MOUNTED_DOCUMENT_READ_CAPABILITY].available is False
    assert tuple(build_auxiliary_execution_capability_catalogs()) == (
        MODEL_ANALYSIS_CAPABILITY,
    )
    assert len(canonical_auxiliary_architect_state_guard(request)) == 64

    workspace_runtime = SimpleNamespace(
        scope_snapshot_sha256=SHA_A,
        tool_ids=("read_text",),
        visual_analysis_available=False,
        visual_analysis_reason="not_configured",
    )
    file_disabled_capabilities = build_auxiliary_planning_capability_catalog(
        mounted,
        workspace_runtime=workspace_runtime,
    )
    file_disabled_cognition = next(
        item
        for item in file_disabled_capabilities.capabilities
        if item.capability_alias == KNOWLEDGE_COGNITION_CAPABILITY
    )
    assert file_disabled_cognition.available is False
    assert "candidate" not in file_disabled_cognition.description
    assert "file alias" not in file_disabled_cognition.description.lower()

    history_only_capabilities = build_auxiliary_planning_capability_catalog(
        mounted,
        workspace_runtime=workspace_runtime,
        knowledge_cognition_history_enabled=True,
    )
    history_only_cognition = next(
        item
        for item in history_only_capabilities.capabilities
        if item.capability_alias == KNOWLEDGE_COGNITION_CAPABILITY
    )
    assert history_only_cognition.available is True

    assert KNOWLEDGE_INDEXING_CAPABILITY not in {item.capability_alias for item in history_only_capabilities.capabilities}

    mock_proposal = AuxiliaryGraphRevisionProposal.model_validate(
        build_mock_auxiliary_architect_proposal(
            serialize_auxiliary_graph_architect_prompt(request)
        )
    )
    validate_auxiliary_graph_architect_proposal(
        request=request,
        proposal=mock_proposal,
    )
    assert mock_proposal.expected_current_auxiliary_graph_revision == 1
    assert mock_proposal.structure is not None
    assert [node.node_key for node in mock_proposal.structure.nodes] == [
        "analyze_verified_context",
        "synthesize_task_graph",
    ]
    assert mock_proposal.structure.nodes[-1].origin_node_alias == (
        "bootstrap_terminal"
    )

    trigger = _replan_trigger(
        revision=details.auxiliary_graph_revision,
        structure_sha256=details.structure_sha256,
    )
    replan_request = build_auxiliary_architect_request(
        details=details,
        authority=authority,
        capabilities=capabilities,
        objective="理解材料并形成可执行任务图",
        desired_output="完整、可执行、可验证的 TaskGraph",
        replan_trigger=trigger,
    )
    assert replan_request.prompt_payload.replan_trigger == trigger
    assert replan_request.binding_sha256 != request.binding_sha256

    replan_proposal = AuxiliaryGraphRevisionProposal.model_validate(
        build_mock_auxiliary_architect_proposal(
            serialize_auxiliary_graph_architect_prompt(replan_request)
        )
    )
    assert replan_proposal.revision_reason == "verification_failed"
    assert replan_proposal.structure is not None
    assert all(
        node.origin_node_alias is None for node in replan_proposal.structure.nodes
    )
    validate_auxiliary_graph_architect_proposal(
        request=replan_request,
        proposal=replan_proposal,
    )

    same_name_payload = json.loads(
        serialize_auxiliary_graph_architect_prompt(replan_request)
    )
    same_name_payload["current_revision"]["structure"] = (
        replan_proposal.structure.model_dump(mode="json")
    )
    same_name_mock = AuxiliaryGraphRevisionProposal.model_validate(
        build_mock_auxiliary_architect_proposal(
            json.dumps(same_name_payload, ensure_ascii=False)
        )
    )
    assert same_name_mock.structure is not None
    assert tuple(
        node.origin_node_alias for node in same_name_mock.structure.nodes
    ) == tuple(node.node_key for node in same_name_mock.structure.nodes)

    blocked_trigger = _replan_trigger(
        revision=details.auxiliary_graph_revision,
        structure_sha256=details.structure_sha256,
        blocked=True,
    )
    blocked_request = build_auxiliary_architect_request(
        details=details,
        authority=authority,
        capabilities=capabilities,
        objective="理解材料并形成可执行任务图",
        desired_output="完整、可执行、可验证的 TaskGraph",
        replan_trigger=blocked_trigger,
    )
    blocked_mock = AuxiliaryGraphRevisionProposal.model_validate(
        build_mock_auxiliary_architect_proposal(
            serialize_auxiliary_graph_architect_prompt(blocked_request)
        )
    )
    assert blocked_mock.structure is not None
    assert any(
        node.executor_kind == "user_gate" for node in blocked_mock.structure.nodes
    )
    validate_auxiliary_graph_architect_proposal(
        request=blocked_request,
        proposal=blocked_mock,
    )
