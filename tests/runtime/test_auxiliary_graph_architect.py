from __future__ import annotations

from copy import deepcopy
import json

import pytest
from pydantic import ValidationError

from personagraph.l2.auxiliary_graph import (
    AuxiliaryGraphEdgeProposal,
    AuxiliaryGraphRevisionProposal,
    AuxiliaryGraphStructureProposal,
    AuxiliaryNodeExecutorKind,
    AuxiliaryNodeKind,
    AuxiliaryNodeProposal,
    AuxiliaryPlanningGoalStatus,
    AuxiliaryPlanningGoal,
    PlanningAuthorityClass,
    PlanningAuthorityProjection,
    PlanningAuthoritySourceCard,
    PlanningAuthoritySourceKind,
    PlanningCapabilityCatalogProjection,
    PlanningCapabilityDescriptor,
    PlanningCapabilityEffect,
    PlanningContextArtifactProjection,
    PlanningContextFactProjection,
    PlanningContextGapProjection,
    PlanningEpisodeBudgetProfile,
    PlanningEpisodeBudgetUsage,
    PlanningEpisodeBudget,
    PlanningGoalPromptContext,
    PlanningObservationStatus,
    TaskGraphSemanticVerificationDimension,
    TaskGraphSemanticVerificationDisposition,
    TaskGraphSemanticVerificationItem,
    TaskGraphSemanticVerificationVerdict,
    canonical_task_graph_revision_proposal_sha256,
)
from personagraph.l2.task_graph import (
    InSessionTaskAcceptanceProposal,
    InSessionTaskGraphRevisionProposal,
    InSessionTaskNodeKind,
)
from personagraph.model_io.gateway import ModelGatewayError, ModelResult, PreparedModelCall
import personagraph.l2.auxiliary_execution.planning.architect as architect_module
from personagraph.runtime.model_calls import (
    DurableModelCallReplay,
    DurableModelCallTerminalState,
)
from personagraph.model_io.output_repair_contracts import (
    RuntimeModelOutputRepairProtocol,
    RuntimeModelStructuredPrompt,
)
from personagraph.runtime.model_calls import requests as model_requests
from personagraph.l2.auxiliary_execution.planning.architect import (
    _AUXILIARY_GRAPH_ARCHITECT_SYSTEM_PROMPT,
    _architect_guard_repair_feedback,
    AuxiliaryGraphArchitectAction,
    AuxiliaryGraphArchitectBudgetExhausted,
    AuxiliaryGraphArchitectGuardError,
    AuxiliaryGraphArchitectPrompt,
    AuxiliaryGraphArchitectReplanTrigger,
    AuxiliaryGraphProtectedCapabilityGrant,
    AuxiliaryGraphArchitectRequest,
    AuxiliaryGraphCurrentRevisionProjection,
    AuxiliaryGraphReplanReviewerFindings,
    request_auxiliary_graph_architect,
    serialize_auxiliary_graph_architect_prompt,
    validate_auxiliary_graph_architect_proposal,
)
from tests.helpers.prepared_model_provider import as_prepared_test_provider


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64

_ARCHITECT_NULLABLE_HASH_FIELDS = (
    "task_graph_semantic_base",
    "task_graph_revision_trigger",
    "task_graph_revision_route",
)


@pytest.fixture(autouse=True)
def _no_retry_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(model_requests, "_sleep", lambda _seconds: None)


def test_architect_prompt_forbids_preterminal_validation_of_future_delivery() -> None:
    assert "未来 TaskGraph 执行后才会出现的最终交付物" in (
        _AUXILIARY_GRAPH_ARCHITECT_SYSTEM_PROMPT
    )
    assert "不得要求被检查对象本身必须 pass" in (
        _AUXILIARY_GRAPH_ARCHITECT_SYSTEM_PROMPT
    )
    assert "若没有一个已存在的 ancestor artifact 可供验证" in (
        _AUXILIARY_GRAPH_ARCHITECT_SYSTEM_PROMPT
    )


def test_architect_prompt_requires_minimal_sufficient_irreducible_graph() -> None:
    prompt = _AUXILIARY_GRAPH_ARCHITECT_SYSTEM_PROMPT

    assert "最少充分节点" in prompt
    assert "最短必要依赖链" in prompt
    assert "无法安全并入相邻节点" in prompt
    assert "固定阶段模板" in prompt


def test_architect_prompt_keeps_mounted_reads_inside_host_primitive() -> None:
    prompt = _AUXILIARY_GRAPH_ARCHITECT_SYSTEM_PROMPT

    assert "mounted_document_read 只能配合 executor_kind=host_primitive" in prompt
    assert "host_mounted_visual_read 只能配合 executor_kind=host_primitive" in prompt
    assert "视觉授权、披露与调用账本边界" in prompt
    assert "不能从“Host 给了哪些 resource”反向逐项建节点" in prompt
    assert "禁止把 visual cards 或 document cards 一对一展开成观察节点" in prompt
    assert "资源枚举图不是最少充分规划" in prompt
    assert "把读取工作下沉到 TaskGraph 执行阶段" in prompt


def test_architect_prompt_defines_positive_base_terminal_material_contract() -> None:
    prompt = _AUXILIARY_GRAPH_ARCHITECT_SYSTEM_PROMPT

    assert "TaskGraphRevisionCandidate" in prompt
    assert "schema_version、proposal、lineage" in prompt
    assert "target_graph_revision" in prompt
    assert "不得要求" in prompt
    assert "task_graph_semantic_base" in prompt


def test_architect_prompt_separates_planning_observations_from_task_runtime() -> None:
    prompt = _AUXILIARY_GRAPH_ARCHITECT_SYSTEM_PROMPT

    assert "base-null" in prompt
    assert "AuxiliaryGraph 使用 create_revision 还是 revise_revision 只由 current_revision 决定" in prompt
    assert "不表示 AuxiliaryGraph 处于 create_revision 状态" in prompt
    assert "不存在 model-owned lineage" in prompt
    assert "任何 criterion 都不得出现独立 token lineage" in prompt
    assert "即使是否定句、示例或元说明也不允许" in prompt
    assert "planning-only" in prompt
    assert "显式命名或引用 AuxiliaryGraph node alias" in prompt
    assert "不得出现任何 planning-only alias 的独立 token" in prompt
    assert "叶节点重新观察已授权原始来源" in prompt


def test_architect_base_prompt_contains_no_failure_only_repair_context() -> None:
    prompt = _AUXILIARY_GRAPH_ARCHITECT_SYSTEM_PROMPT

    assert "host_output_repair_feedback" not in prompt
    assert "上一份输出" not in prompt
    assert "修复清单" not in prompt


def test_architect_prompt_requires_one_verbatim_gate_per_frozen_question() -> None:
    prompt = _AUXILIARY_GRAPH_ARCHITECT_SYSTEM_PROMPT

    assert "blocking_questions 中的每一条" in prompt
    assert "建立一个独立的 required clarify + user_gate 节点" in prompt
    assert "objective 必须逐字复制对应的完整问题" in prompt
    assert "不得添加前缀、后缀、编号、合并或改写" in prompt
    assert "全由 required=true edge 构成的路径到达 terminal" in prompt


def _authority(
    *, include_secondary_authorization: bool = False
) -> PlanningAuthorityProjection:
    cards = [
        PlanningAuthoritySourceCard(
            alias="gap_block",
            authority_class=PlanningAuthorityClass.GAP,
            source_kind=PlanningAuthoritySourceKind.GAP,
            source_label="Missing required page",
            excerpt="A required page cannot currently be read.",
            projection_sha256=SHA_A,
        ),
        PlanningAuthoritySourceCard(
            alias="obs_doc",
            authority_class=PlanningAuthorityClass.EVIDENCE,
            source_kind=PlanningAuthoritySourceKind.DOCUMENT,
            source_label="Mounted paper",
            excerpt="The mounted paper is available for bounded inspection.",
            projection_sha256=SHA_B,
        ),
        PlanningAuthoritySourceCard(
            alias="src_user",
            authority_class=PlanningAuthorityClass.AUTHORIZATION,
            source_kind=PlanningAuthoritySourceKind.USER_INSTRUCTION,
            source_label="Current instruction",
            excerpt="Understand the paper and produce an evidence-grounded answer.",
            projection_sha256=SHA_C,
        ),
    ]
    if include_secondary_authorization:
        cards.append(
            PlanningAuthoritySourceCard(
                alias="src_other",
                authority_class=PlanningAuthorityClass.AUTHORIZATION,
                source_kind=PlanningAuthoritySourceKind.USER_ANSWER,
                source_label="Separate user authorization",
                excerpt="Authorize a separate protected scope.",
                projection_sha256=SHA_D,
            )
        )
    return PlanningAuthorityProjection.create(
        authority_snapshot_id="authority_snapshot_01",
        authority_snapshot_sha256=SHA_A,
        cards=tuple(sorted(cards, key=lambda item: item.alias)),
    )


def _artifact() -> PlanningContextArtifactProjection:
    return PlanningContextArtifactProjection.create(
        artifact_alias="artifact_doc",
        artifact_id="private_artifact_01",
        artifact_sha256=SHA_B,
        producer_node_alias="bootstrap_observe",
        facts=(
            PlanningContextFactProjection(
                fact_alias="fact_document_present",
                statement="The paper is mounted and its readable portion is indexed.",
                evidence_aliases=("obs_doc",),
            ),
        ),
        gaps=(
            PlanningContextGapProjection(
                gap_alias="gap_block",
                observation_status=PlanningObservationStatus.BLOCKED,
                description="A required page is not readable.",
                blocking=True,
                affected_obligations=("understand_complete_paper",),
                evidence_aliases=("obs_doc",),
                resolution_hint="Ask whether partial coverage is acceptable.",
            ),
        ),
    )


def _capabilities(
    *,
    document_available: bool = True,
    include_fallback: bool = False,
) -> PlanningCapabilityCatalogProjection:
    capabilities = [
        PlanningCapabilityDescriptor(
            capability_alias="document_read",
            label="Document read",
            description="Read bounded mounted-document content.",
            available=document_available,
            effect=PlanningCapabilityEffect.READ_ONLY,
            supported_operations=("read",),
            supported_resource_kinds=("pdf",),
        )
    ]
    if include_fallback:
        capabilities.append(
            PlanningCapabilityDescriptor(
                capability_alias="fallback_read",
                label="Fallback read",
                description="Read a previously materialized ContextArtifact.",
                available=True,
                effect=PlanningCapabilityEffect.READ_ONLY,
                supported_operations=("read",),
                supported_resource_kinds=("artifact",),
            )
        )
    return PlanningCapabilityCatalogProjection.create(
        capability_catalog_snapshot_id="capability_catalog_01",
        capability_catalog_snapshot_sha256=SHA_D,
        capabilities=tuple(capabilities),
    )


def _acceptance(acceptance_id: str) -> InSessionTaskAcceptanceProposal:
    return InSessionTaskAcceptanceProposal(
        acceptance_id=acceptance_id,
        criterion="The node produces a source-grounded, inspectable result.",
        source_anchor_ids=("obs_doc", "src_user"),
    )


@pytest.mark.parametrize(
    "capability_alias",
    ("mounted_document_read", "host_mounted_visual_read"),
)
def test_guard_rejects_host_read_capability_on_model_work_run(
    capability_alias: str,
) -> None:
    capabilities = PlanningCapabilityCatalogProjection.create(
        capability_catalog_snapshot_id="host_read_capability_catalog_01",
        capability_catalog_snapshot_sha256=SHA_D,
        capabilities=(
            PlanningCapabilityDescriptor(
                capability_alias=capability_alias,
                label="Mounted resource read",
                description="Read one exact Host-frozen mounted resource.",
                available=True,
                effect=PlanningCapabilityEffect.READ_ONLY,
                supported_operations=("read_one_mounted_resource",),
                supported_resource_kinds=("pdf",),
            ),
        ),
    )

    with pytest.raises(
        AuxiliaryGraphArchitectGuardError,
        match=rf"must execute {capability_alias} through a host_primitive",
    ):
        validate_auxiliary_graph_architect_proposal(
            request=_request(capabilities=capabilities),
            proposal=request_proposal(
                _proposal_dict(
                    structure=_structure(capability_alias=capability_alias),
                )
            ),
        )


def _structure(
    *,
    capability_alias: str = "document_read",
    with_origins: bool = False,
) -> AuxiliaryGraphStructureProposal:
    return AuxiliaryGraphStructureProposal(
        terminal_node_key="terminal",
        nodes=(
            AuxiliaryNodeProposal(
                node_key="observe",
                node_kind=AuxiliaryNodeKind.OBSERVE,
                executor_kind=AuxiliaryNodeExecutorKind.MODEL_WORK_RUN,
                title="Inspect the mounted paper",
                objective="Collect the evidence needed to plan the formal task graph.",
                acceptance_criteria=(_acceptance("accept_observation"),),
                capability_profile_id=capability_alias,
                input_resource_aliases=("artifact_doc",),
                source_anchor_ids=("obs_doc", "src_user"),
                output_contract="planning_context_artifact_v1",
                origin_node_alias="observe" if with_origins else None,
            ),
            AuxiliaryNodeProposal(
                node_key="terminal",
                node_kind=AuxiliaryNodeKind.SYNTHESIZE,
                executor_kind=AuxiliaryNodeExecutorKind.TERMINAL_PLANNER,
                title="Synthesize the formal graph",
                objective="Create the complete evidence-grounded TaskGraph proposal.",
                acceptance_criteria=(_acceptance("accept_task_graph"),),
                input_resource_aliases=("artifact_doc",),
                source_anchor_ids=("obs_doc", "src_user"),
                output_contract="task_graph_revision_proposal_v2",
                origin_node_alias="terminal" if with_origins else None,
            ),
        ),
        edges=(
            AuxiliaryGraphEdgeProposal(
                source_node_key="observe",
                target_node_key="terminal",
            ),
        ),
    )


def _structure_with_user_gate(
    *,
    with_origins: bool = True,
) -> AuxiliaryGraphStructureProposal:
    base = _structure(with_origins=with_origins)
    observe, terminal = base.nodes
    user_gate = AuxiliaryNodeProposal(
        node_key="clarify_gap",
        node_kind=AuxiliaryNodeKind.CLARIFY,
        executor_kind=AuxiliaryNodeExecutorKind.USER_GATE,
        title="Clarify the blocking evidence gap",
        objective="Obtain the missing information before replanning the TaskGraph.",
        acceptance_criteria=(
            InSessionTaskAcceptanceProposal(
                acceptance_id="gap_clarified",
                criterion="The blocking gap is resolved or explicitly preserved.",
                source_anchor_ids=("src_user",),
            ),
        ),
        input_resource_aliases=(),
        source_anchor_ids=("src_user",),
        output_contract="user_response_v1",
    )
    return AuxiliaryGraphStructureProposal(
        terminal_node_key=terminal.node_key,
        nodes=(observe, user_gate, terminal),
        edges=(
            AuxiliaryGraphEdgeProposal(
                source_node_key=observe.node_key,
                target_node_key=terminal.node_key,
            ),
            AuxiliaryGraphEdgeProposal(
                source_node_key=user_gate.node_key,
                target_node_key=terminal.node_key,
            ),
        ),
    )


@pytest.mark.parametrize(
    "criterion",
    (
        "TaskGraph proposal 的 lineage 必须引用 observe 的已核实事实与 gap。",
        "TaskGraph 提案不得包含 lineage 字段。",
    ),
)
def test_guard_rejects_base_null_terminal_acceptance_with_lineage_token(
    criterion: str,
) -> None:
    structure = _structure().model_dump(mode="json")
    structure["nodes"][1]["acceptance_criteria"][0]["criterion"] = criterion

    with pytest.raises(
        AuxiliaryGraphArchitectGuardError,
        match="base-null.*lineage",
    ):
        validate_auxiliary_graph_architect_proposal(
            request=_request(),
            proposal=request_proposal(
                _proposal_dict(
                    structure=AuxiliaryGraphStructureProposal.model_validate(
                        structure
                    )
                )
            ),
        )


@pytest.mark.parametrize(
    "criterion",
    (
        "TaskGraph proposal 的节点与 Acceptance 必须显式引用 release_readiness_analysis。",
        "TaskGraph proposal 不得引用 release_readiness_analysis。",
    ),
)
def test_guard_rejects_terminal_acceptance_with_planning_node_alias_token(
    criterion: str,
) -> None:
    structure = _structure().model_dump(mode="json")
    structure["nodes"][0]["node_key"] = "release_readiness_analysis"
    structure["edges"][0]["source_node_key"] = "release_readiness_analysis"
    structure["nodes"][1]["acceptance_criteria"][0]["criterion"] = criterion

    with pytest.raises(
        AuxiliaryGraphArchitectGuardError,
        match="planning-only alias",
    ):
        validate_auxiliary_graph_architect_proposal(
            request=_request(),
            proposal=request_proposal(
                _proposal_dict(
                    structure=AuxiliaryGraphStructureProposal.model_validate(
                        structure
                    )
                )
            ),
        )


def test_guard_reports_transition_lineage_and_planning_aliases_in_one_pass() -> None:
    structure = _structure().model_dump(mode="json")
    terminal = structure["nodes"][1]
    terminal["acceptance_criteria"] = [
        {
            "acceptance_id": "accept_task_graph_primary",
            "criterion": (
                "TaskGraph lineage must name planning node alias `observe`."
            ),
            "source_anchor_ids": ["obs_doc", "src_user"],
        },
        {
            "acceptance_id": "accept_task_graph_secondary",
            "criterion": (
                "TaskGraph lineage must name artifact alias `artifact_doc`."
            ),
            "source_anchor_ids": ["obs_doc", "src_user"],
        },
    ]
    proposal = request_proposal(
        _proposal_dict(
            disposition="create_revision",
            structure=AuxiliaryGraphStructureProposal.model_validate(structure),
        )
    )

    with pytest.raises(AuxiliaryGraphArchitectGuardError) as captured:
        validate_auxiliary_graph_architect_proposal(
            request=_request(current=True),
            proposal=proposal,
        )

    error = captured.value
    assert "create_revision" in str(error)
    assert "lineage" in str(error)
    assert "planning-only alias" in str(error)
    assert error.issue_coverage == "complete"
    assert [issue.code for issue in error.repair_issues] == [
        "architect_base_null_lineage_forbidden",
        "architect_base_null_lineage_forbidden",
        "architect_planning_alias_export_forbidden",
        "architect_planning_alias_export_forbidden",
        "architect_revision_transition_invalid",
    ]
    assert [issue.paths for issue in error.repair_issues] == [
        ("/structure/nodes/1/acceptance_criteria/0/criterion",),
        ("/structure/nodes/1/acceptance_criteria/1/criterion",),
        ("/structure/nodes/1/acceptance_criteria/0/criterion",),
        ("/structure/nodes/1/acceptance_criteria/1/criterion",),
        (
            "/disposition",
            "/expected_current_auxiliary_graph_revision",
            "/revision_reason",
        ),
    ]


def test_guard_allows_terminal_acceptance_to_require_upstream_semantic_influence() -> None:
    structure = _structure().model_dump(mode="json")
    structure["nodes"][1]["acceptance_criteria"][0]["criterion"] = (
        "TaskGraph proposal 的语义必须反映已验证的上游事实、约束与 gap，"
        "并安排运行时重新观察所需的原始来源。"
    )

    accepted = validate_auxiliary_graph_architect_proposal(
        request=_request(),
        proposal=request_proposal(
            _proposal_dict(
                structure=AuxiliaryGraphStructureProposal.model_validate(structure)
            )
        ),
    )

    assert accepted.structure is not None


def test_guard_allows_plain_word_node_key_used_as_an_ordinary_verb() -> None:
    structure = _structure().model_dump(mode="json")
    structure["nodes"][1]["acceptance_criteria"][0]["criterion"] = (
        "The TaskGraph must let an authorized leaf observe the document before "
        "downstream synthesis."
    )

    accepted = validate_auxiliary_graph_architect_proposal(
        request=_request(),
        proposal=request_proposal(
            _proposal_dict(
                structure=AuxiliaryGraphStructureProposal.model_validate(structure)
            )
        ),
    )

    assert accepted.structure is not None


def _budget(
    *,
    current: bool = False,
    logical_model_calls: int = 0,
    profile: PlanningEpisodeBudgetProfile | None = None,
) -> PlanningEpisodeBudget:
    return PlanningEpisodeBudget.create(
        budget_ledger_id="budget_01",
        goal_id="goal_01",
        base_profile=profile or PlanningEpisodeBudgetProfile(),
        usage=PlanningEpisodeBudgetUsage(
            current_graph_nodes=2 if current else 0,
            current_graph_depth=2 if current else 0,
            auxiliary_graph_revisions=1 if current else 0,
            distinct_auxiliary_nodes=2 if current else 0,
            logical_model_calls=logical_model_calls,
        ),
    )


def _current() -> AuxiliaryGraphCurrentRevisionProjection:
    return AuxiliaryGraphCurrentRevisionProjection.create(
        auxiliary_graph_revision=3,
        base_task_graph_revision=None,
        structure=_structure(),
        source_structure_sha256=SHA_D,
    )


def _rejected_task_graph() -> InSessionTaskGraphRevisionProposal:
    return InSessionTaskGraphRevisionProposal.model_validate(
        {
            "root": {
                "root_key": "answer",
                "nodes": [
                    {
                        "node_key": "answer",
                        "node_kind": InSessionTaskNodeKind.ROOT.value,
                        "parent_node_key": None,
                        "title": "Answer from the mounted paper",
                        "objective": "Produce the requested grounded answer.",
                        "source_anchor_ids": ["src_user"],
                        "acceptance_criteria": [
                            {
                                "acceptance_id": "answer_grounded",
                                "criterion": "The answer is grounded in the source.",
                                "source_anchor_ids": ["src_user"],
                            }
                        ],
                        "constraints": [],
                    }
                ],
            }
        }
    )


def _semantic_items(
    *,
    failed_dimension: TaskGraphSemanticVerificationDimension = (
        TaskGraphSemanticVerificationDimension.EVIDENCE_GROUNDING
    ),
    verdict: TaskGraphSemanticVerificationVerdict = (
        TaskGraphSemanticVerificationVerdict.FAIL
    ),
) -> tuple[TaskGraphSemanticVerificationItem, ...]:
    return tuple(
        TaskGraphSemanticVerificationItem(
            dimension=dimension,
            verdict=(verdict if dimension is failed_dimension else "pass"),
            failure_scope=(
                "missing_authority"
                if dimension is failed_dimension
                and verdict
                is TaskGraphSemanticVerificationVerdict.INSUFFICIENT_EVIDENCE
                else "auxiliary_investigation"
                if dimension is failed_dimension
                else None
            ),
            finding=(
                "The rejected proposal does not preserve the verified evidence."
                if dimension is failed_dimension
                else "This semantic dimension passed."
            ),
            affected_node_keys=("answer",) if dimension is failed_dimension else (),
            evidence_aliases=("obs_doc",) if dimension is failed_dimension else (),
            gap_aliases=(
                ("gap_block",)
                if dimension is failed_dimension
                and verdict
                is TaskGraphSemanticVerificationVerdict.INSUFFICIENT_EVIDENCE
                else ()
            ),
        )
        for dimension in TaskGraphSemanticVerificationDimension
    )


def _replan_trigger(
    *,
    disposition: TaskGraphSemanticVerificationDisposition = (
        TaskGraphSemanticVerificationDisposition.REVISE
    ),
    verdict: TaskGraphSemanticVerificationVerdict = (
        TaskGraphSemanticVerificationVerdict.FAIL
    ),
) -> AuxiliaryGraphArchitectReplanTrigger:
    current = _current()
    return AuxiliaryGraphArchitectReplanTrigger.create(
        expected_current_auxiliary_graph_revision=(
            current.auxiliary_graph_revision
        ),
        expected_current_structure_sha256=current.source_structure_sha256,
        semantic_host_disposition=disposition,
        semantic_settlement_id="semantic_settlement_01",
        semantic_settlement_sha256=SHA_A,
        semantic_prompt_payload_sha256=SHA_B,
        rejected_task_graph_proposal=_rejected_task_graph(),
        reviewers=(
            AuxiliaryGraphReplanReviewerFindings(
                reviewer_ordinal=1,
                semantic_result_sha256=SHA_C,
                items=_semantic_items(verdict=verdict),
            ),
        ),
    )


def _request(
    *,
    current: bool = False,
    budget: PlanningEpisodeBudget | None = None,
    capabilities: PlanningCapabilityCatalogProjection | None = None,
    authority: PlanningAuthorityProjection | None = None,
    authorization_aliases: tuple[str, ...] = ("src_user",),
    protected_capability_grants: tuple[
        AuxiliaryGraphProtectedCapabilityGrant, ...
    ] = (),
    replan_trigger: AuxiliaryGraphArchitectReplanTrigger | None = None,
) -> AuxiliaryGraphArchitectRequest:
    prompt = AuxiliaryGraphArchitectPrompt.create(
        goal=PlanningGoalPromptContext(
            goal_id="goal_01",
            objective="Understand the mounted paper before planning the requested work.",
            desired_output="A complete and executable TaskGraph.",
            authorization_aliases=authorization_aliases,
        ),
        authority=authority or _authority(),
        context_artifacts=(_artifact(),),
        capabilities=capabilities or _capabilities(),
        protected_capability_grants=protected_capability_grants,
        budget=budget or _budget(current=current),
        current_revision=_current() if current else None,
        replan_trigger=replan_trigger,
    )
    return AuxiliaryGraphArchitectRequest.create(
        architect_request_id="architect_request_01",
        logical_call_id="architect_logical_call_01",
        architect_profile_id="auxiliary_architect_v2",
        goal=AuxiliaryPlanningGoal(
            session_id="session_01",
            task_id="task_01",
            auxiliary_graph_id="auxiliary_graph_01",
            goal_id="goal_01",
            base_task_graph_revision=None,
            target_task_graph_revision=1,
            creation_turn_id="turn_01",
            authorization_manifest_id="authorization_manifest_01",
            budget_ledger_id="budget_01",
        ),
        prompt_payload=prompt,
    )


def _proposal_dict(
    *,
    disposition: str = "create_revision",
    current_revision: int | None = None,
    structure: AuxiliaryGraphStructureProposal | None = None,
    question: str | None = None,
    failure_reason: str | None = None,
) -> dict[str, object]:
    if structure is None and disposition in {"create_revision", "revise_revision"}:
        structure = _structure(with_origins=disposition == "revise_revision")
    return {
        "schema_version": "auxiliary-graph-revision-proposal-v2",
        "disposition": disposition,
        "expected_current_auxiliary_graph_revision": current_revision,
        "revision_reason": (
            "initial"
            if disposition == "create_revision"
            else "evidence_changed"
            if disposition == "revise_revision"
            else None
        ),
        "structure": None if structure is None else structure.model_dump(mode="json"),
        "explanation": "The proposal follows the frozen evidence and authority.",
        "blocking_gap_ids": ["gap_block"] if question is not None else [],
        "requested_user_question": question,
        "failure_reason": failure_reason,
    }


def _model_result(reply: object, model_call_id: str) -> ModelResult:
    return ModelResult(
        reply=reply if isinstance(reply, str) else json.dumps(reply),
        provider="test",
        model="test",
        latency_ms=1,
        model_call_id=model_call_id,
    )


def test_prompt_only_exact_logical_identity_and_host_bound_create_decision() -> None:
    request = _request()
    received: dict[str, object] = {}
    events = []

    def provider(system: str, user: str, **kwargs: object) -> ModelResult:
        received.update(system=system, user=user, kwargs=kwargs)
        return _model_result(_proposal_dict(), str(kwargs["model_call_id"]))

    result = request_auxiliary_graph_architect(
        request,
        invocation_turn_id="turn_architect_01",
        provider=as_prepared_test_provider(provider),
        emit=events.append,
    )

    assert json.loads(str(received["user"])) == request.to_prompt_payload().model_dump(
        mode="json"
    )
    assert str(received["user"]) == serialize_auxiliary_graph_architect_prompt(request)
    assert request.architect_request_id not in str(received["user"])
    assert request.logical_call_id not in str(received["user"])
    assert request.binding_sha256 not in str(received["user"])
    assert "不得调用工具" in str(received["system"])
    assert "禁止输出 call_tools" in str(received["system"])
    assert '"source_node_key":"..."' in str(received["system"])
    assert "绝不存在 task" in str(received["system"])
    assert "manual_replan" in str(received["system"])
    assert "字典升序排列" in str(received["system"])
    assert "新建的 node_key 写入 input_resource_aliases" in str(
        received["system"]
    )
    assert received["kwargs"] == {
        "model_call_id": request.logical_call_id,
        "purpose": "runtime_auxiliary_graph_architect_v2",
    }
    assert result.model_call_id == request.logical_call_id
    assert result.value.action is AuxiliaryGraphArchitectAction.CREATE_REVISION
    assert result.value.architect_request_id == request.architect_request_id
    assert result.value.request_binding_sha256 == request.binding_sha256
    assert result.value.logical_call_id == request.logical_call_id
    assert {event.model_call_id for event in events} == {request.logical_call_id}
    assert [event.stage.value for event in events] == ["L2_PLAN", "L2_PLAN"]


def test_guard_rejection_is_sent_as_structured_bounded_repair_feedback() -> None:
    request = _request()
    rejected = _proposal_dict()
    structure = rejected["structure"]
    assert isinstance(structure, dict)
    nodes = structure["nodes"]
    assert isinstance(nodes, list)
    terminal = next(node for node in nodes if node["node_key"] == "terminal")
    secret_marker = "PRIVATE_REJECTED_OUTPUT_MARKER"
    terminal["acceptance_criteria"][0]["criterion"] = (
        f"Require TaskGraph lineage and never repeat {secret_marker}."
    )
    seen_user_payloads: list[dict[str, object]] = []
    seen_repair_messages: list[list[dict[str, str]]] = []

    def provider(_system: str, user: str, **kwargs: object) -> ModelResult:
        payload = json.loads(user)
        assert isinstance(payload, dict)
        seen_user_payloads.append(payload)
        repair_messages = kwargs.get("repair_messages")
        if repair_messages is not None:
            assert isinstance(repair_messages, list)
            seen_repair_messages.append(repair_messages)
        reply = rejected if len(seen_user_payloads) == 1 else _proposal_dict()
        return _model_result(reply, str(kwargs["model_call_id"]))

    result = request_auxiliary_graph_architect(
        request,
        invocation_turn_id="turn_architect_repair",
        provider=as_prepared_test_provider(provider),
        emit=lambda _event: None,
    )

    assert result.attempts == 2
    assert "host_output_repair_feedback" not in seen_user_payloads[0]
    assert "host_output_repair_feedback" not in seen_user_payloads[1]
    assert len(seen_repair_messages) == 1
    messages = seen_repair_messages[0]
    assert [message["role"] for message in messages] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
    repair = json.loads(messages[3]["content"].split("Host 修复清单：", 1)[1])
    issue = repair["current_issues"][0]
    assert set(issue) == {"paths", "safe_explanation"}
    assert issue["safe_explanation"] == (
        "当前 TaskGraph 没有语义基线；该验收条件不得出现独立 lineage token，"
        "请改写为正向、可验证的 TaskGraph 内容要求。"
    )
    assert set(repair) == {"current_issues"}
    assert messages[2]["content"] == json.dumps(rejected)
    assert secret_marker not in messages[3]["content"]


def test_prepared_architect_repair_keeps_frozen_first_two_messages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request()
    invalid_reply = json.dumps({"unexpected": "PRIVATE_ARCHITECT_VALUE"})
    valid_reply = json.dumps(_proposal_dict())
    prepared_calls: list[dict[str, object]] = []
    frozen_system_prompt = "FROZEN_ARCHITECT_SYSTEM"
    frozen_user_content = '{"frozen_architect_prompt":true}'

    monkeypatch.setattr(
        architect_module,
        "durable_structured_provider_prompt",
        lambda _durable_call, **_current: (
            frozen_system_prompt,
            frozen_user_content,
        ),
    )

    def provider(*_args: object, **_kwargs: object) -> ModelResult:
        raise AssertionError("prepared architect provider used direct dispatch")

    def prepare(
        system_prompt: str,
        user_content: str,
        **kwargs: object,
    ) -> PreparedModelCall:
        prepared_calls.append(
            {
                "system_prompt": system_prompt,
                "user_content": user_content,
                **kwargs,
            }
        )
        reply = invalid_reply if len(prepared_calls) == 1 else valid_reply

        def dispatch(model_call_id: str | None) -> ModelResult:
            assert model_call_id is not None
            return _model_result(reply, model_call_id)

        return PreparedModelCall(_dispatch=dispatch)

    provider.prepare = prepare  # type: ignore[attr-defined]
    result = request_auxiliary_graph_architect(
        request,
        invocation_turn_id="turn_architect_prepared_repair",
        provider=as_prepared_test_provider(provider),
        emit=lambda _event: None,
    )

    assert result.attempts == 2
    assert len(prepared_calls) == 2
    initial, repair = prepared_calls
    assert initial["system_prompt"] == frozen_system_prompt
    assert initial["user_content"] == frozen_user_content
    assert repair["system_prompt"] == frozen_system_prompt
    assert repair["user_content"] == frozen_user_content
    messages = repair["repair_messages"]
    assert isinstance(messages, list)
    assert messages[:2] == [
        {"role": "system", "content": frozen_system_prompt},
        {"role": "user", "content": frozen_user_content},
    ]
    assert messages[2]["content"] == invalid_reply


def test_contract_repair_feedback_exposes_safe_location_not_bad_input() -> None:
    request = _request()
    rejected = _proposal_dict()
    structure = rejected["structure"]
    assert isinstance(structure, dict)
    nodes = structure["nodes"]
    assert isinstance(nodes, list)
    first_node = nodes[0]
    assert isinstance(first_node, dict)
    first_node.pop("executor_kind")
    secret_bad_input = "PRIVATE_BAD_INPUT_VALUE"
    first_node["unexpected_private_field"] = secret_bad_input
    seen_repair_messages: list[list[dict[str, str]]] = []

    def provider(_system: str, user: str, **kwargs: object) -> ModelResult:
        payload = json.loads(user)
        assert isinstance(payload, dict)
        assert "host_output_repair_feedback" not in payload
        repair_messages = kwargs.get("repair_messages")
        if repair_messages is not None:
            assert isinstance(repair_messages, list)
            seen_repair_messages.append(repair_messages)
        reply = rejected if not seen_repair_messages else _proposal_dict()
        return _model_result(reply, str(kwargs["model_call_id"]))

    result = request_auxiliary_graph_architect(
        request,
        invocation_turn_id="turn_architect_contract_repair",
        provider=as_prepared_test_provider(provider),
        emit=lambda _event: None,
    )

    assert result.attempts == 2
    assert len(seen_repair_messages) == 1
    messages = seen_repair_messages[0]
    feedback = json.loads(messages[3]["content"].split("Host 修复清单：", 1)[1])
    issues = feedback["current_issues"]
    assert {issue["safe_explanation"] for issue in issues} == {
        "该位置含有目标合同未声明的额外字段。",
        "目标合同要求此位置必须存在。",
    }
    assert {
        path
        for issue in issues
        for path in issue["paths"]
    } == {"/structure/nodes/0", "/structure/nodes/0/executor_kind"}
    assert secret_bad_input not in messages[3]["content"]
    assert "unexpected_private_field" not in messages[3]["content"]


@pytest.mark.parametrize(
    ("guard_message", "expected_code"),
    (
        (
            "proposal expected-current revision is stale or absent PRIVATE_ALIAS",
            "architect_revision_transition_invalid",
        ),
        (
            "a blocked verification replan must plan a typed user_gate node PRIVATE_ALIAS",
            "architect_user_gate_invalid",
        ),
        (
            "proposal references an unknown ContextArtifact gap PRIVATE_ALIAS",
            "architect_gap_reference_invalid",
        ),
        (
            "node PRIVATE_ALIAS lacks authorization authority",
            "architect_authorization_scope_invalid",
        ),
        (
            "node PRIVATE_ALIAS references unknown source authority",
            "architect_unknown_source_authority",
        ),
        (
            "node PRIVATE_ALIAS references an unknown resource alias",
            "architect_unknown_resource_alias",
        ),
        (
            "node PRIVATE_ALIAS must execute workspace_readonly through a model_work_run",
            "architect_executor_capability_mismatch",
        ),
        (
            "node PRIVATE_ALIAS must execute host_mounted_visual_read through a host_primitive",
            "architect_executor_capability_mismatch",
        ),
        (
            "node PRIVATE_ALIAS requires an exact protected capability grant",
            "architect_protected_capability_grant_missing",
        ),
        (
            "proposal references an unknown current-node origin alias PRIVATE_ALIAS",
            "architect_unknown_origin_alias",
        ),
        (
            "terminal Acceptance exports a planning-only alias PRIVATE_ALIAS",
            "architect_planning_alias_export_forbidden",
        ),
    ),
)
def test_guard_repair_reason_families_do_not_echo_model_aliases(
    guard_message: str,
    expected_code: str,
) -> None:
    code, safe_reason = _architect_guard_repair_feedback(
        AuxiliaryGraphArchitectGuardError(guard_message)
    )

    assert code == expected_code
    assert "PRIVATE_ALIAS" not in safe_reason
    assert len(safe_reason.encode("utf-8")) <= 500


@pytest.mark.parametrize(
    ("guard_message", "expected_code", "expected_reason"),
    (
        (
            "create_revision cannot replace an existing revision",
            "architect_revision_transition_invalid",
            "请按冻结的 current_revision 与 trigger 重新选择 revision transition："
            "仅 current_revision=null 时可使用 create_revision/initial；"
            "current_revision 非 null 时不得使用 create_revision，且必须精确回显当前 "
            "revision。首次替换 bootstrap_terminal 使用 revise_revision/manual_replan；"
            "verification trigger 使用 revise_revision/verification_failed。"
            "重新生成完整提案并按全部合同复查。",
        ),
        (
            "a base-null terminal Acceptance cannot require TaskGraph lineage",
            "architect_base_null_lineage_forbidden",
            "当前 task_graph_semantic_base 为 null。terminal 的每条 Acceptance "
            "criterion 中都不得出现独立 token lineage，包括否定句、示例或元说明；"
            "只写正向、可验证的 TaskGraph 内容要求，并重新生成完整提案后按全部合同复查。",
        ),
        (
            "terminal Acceptance exports a planning-only alias PRIVATE_ALIAS",
            "architect_planning_alias_export_forbidden",
            "terminal 的每条 Acceptance criterion 中都不得出现任何 AuxiliaryGraph "
            "node alias 或 ContextArtifact artifact_alias 的独立 token，包括否定句、"
            "示例或元说明；不得抄写或转述本条拒绝原因，只写正向、可验证的 "
            "TaskGraph 内容要求，并重新生成完整提案后按全部合同复查。",
        ),
    ),
)
def test_lineage_and_planning_alias_repair_feedback_states_lexical_rule(
    guard_message: str,
    expected_code: str,
    expected_reason: str,
) -> None:
    code, safe_reason = _architect_guard_repair_feedback(
        AuxiliaryGraphArchitectGuardError(guard_message)
    )

    assert code == expected_code
    assert safe_reason == expected_reason
    assert len(safe_reason.encode("utf-8")) <= 500


def test_user_gate_repair_feedback_states_the_exact_mechanical_contract() -> None:
    code, safe_reason = _architect_guard_repair_feedback(
        AuxiliaryGraphArchitectGuardError(
            "required user_gate objectives do not exactly cover the "
            "authenticated blocking questions"
        )
    )

    assert code == "architect_user_gate_invalid"
    assert safe_reason == (
        "For every frozen blocking question, create one separate required "
        "clarify user_gate. Copy that complete question verbatim as the gate "
        "objective, with no prefix, suffix, numbering, merging, or paraphrase. "
        "Give each gate a path made only of required edges to the terminal. "
        "Regenerate the entire proposal."
    )
    assert len(safe_reason.encode("utf-8")) <= 500


def test_base_null_request_roundtrips_with_current_explicit_null_fields() -> None:
    request = _request()
    wire = request.model_dump(mode="json")
    prompt = wire["prompt_payload"]
    assert isinstance(prompt, dict)
    for field_name in _ARCHITECT_NULLABLE_HASH_FIELDS:
        assert field_name in prompt
        assert prompt[field_name] is None

    prompt_hash_payload = deepcopy(prompt)
    payload_sha256 = prompt_hash_payload.pop("payload_sha256")
    assert payload_sha256 == architect_module._canonical_sha256(prompt_hash_payload)

    request_hash_payload = deepcopy(wire)
    binding_sha256 = request_hash_payload.pop("binding_sha256")
    assert binding_sha256 == architect_module._canonical_sha256(request_hash_payload)

    assert AuxiliaryGraphArchitectRequest.model_validate(wire) == request


def test_base_null_request_rejects_legacy_omitted_fields_even_when_rehashed() -> None:
    legacy = _request().model_dump(mode="json")
    prompt = legacy["prompt_payload"]
    assert isinstance(prompt, dict)
    for field_name in _ARCHITECT_NULLABLE_HASH_FIELDS:
        prompt.pop(field_name)

    legacy_prompt_hash_payload = deepcopy(prompt)
    legacy_prompt_hash_payload.pop("payload_sha256")
    prompt["payload_sha256"] = architect_module._canonical_sha256(
        legacy_prompt_hash_payload
    )
    legacy_request_hash_payload = deepcopy(legacy)
    legacy_request_hash_payload.pop("binding_sha256")
    legacy["binding_sha256"] = architect_module._canonical_sha256(
        legacy_request_hash_payload
    )

    with pytest.raises(ValidationError) as exc_info:
        AuxiliaryGraphArchitectRequest.model_validate(legacy)

    for field_name in _ARCHITECT_NULLABLE_HASH_FIELDS:
        assert field_name in str(exc_info.value)


@pytest.mark.parametrize(
    ("hash_layer", "expected_message"),
    (
        ("prompt", "Architect prompt hash does not match its payload"),
        ("request", "Architect request hash does not match its authority"),
    ),
)
def test_base_null_request_rejects_legacy_null_omitting_hashes(
    hash_layer: str,
    expected_message: str,
) -> None:
    wire = _request().model_dump(mode="json")
    prompt = wire["prompt_payload"]
    assert isinstance(prompt, dict)

    if hash_layer == "prompt":
        legacy_hash_payload = deepcopy(prompt)
        legacy_hash_payload.pop("payload_sha256")
        for field_name in _ARCHITECT_NULLABLE_HASH_FIELDS:
            legacy_hash_payload.pop(field_name)
        prompt["payload_sha256"] = architect_module._canonical_sha256(
            legacy_hash_payload
        )
    else:
        legacy_hash_payload = deepcopy(wire)
        legacy_hash_payload.pop("binding_sha256")
        legacy_prompt = legacy_hash_payload["prompt_payload"]
        assert isinstance(legacy_prompt, dict)
        for field_name in (
            "task_graph_semantic_base",
            "task_graph_revision_trigger",
            "task_graph_revision_route",
        ):
            legacy_prompt.pop(field_name)
        wire["binding_sha256"] = architect_module._canonical_sha256(
            legacy_hash_payload
        )

    with pytest.raises(ValidationError, match=expected_message):
        AuxiliaryGraphArchitectRequest.model_validate(wire)


def test_architect_parser_canonicalizes_set_like_alias_order() -> None:
    request = _request()
    proposal = _proposal_dict()
    structure = proposal["structure"]
    assert isinstance(structure, dict)
    nodes = structure["nodes"]
    assert isinstance(nodes, list)
    for node in nodes:
        assert isinstance(node, dict)
        node["source_anchor_ids"] = ["src_user", "obs_doc"]
        acceptances = node["acceptance_criteria"]
        assert isinstance(acceptances, list)
        for acceptance in acceptances:
            acceptance["source_anchor_ids"] = ["src_user", "obs_doc"]

    result = request_auxiliary_graph_architect(
        request,
        invocation_turn_id="turn_architect_canonical_order",
        provider=as_prepared_test_provider(
            lambda _system, _user, **kwargs: _model_result(
                proposal,
                str(kwargs["model_call_id"]),
            )
        ),
        emit=lambda _event: None,
    )

    assert result.attempts == 1
    assert result.value.proposal.structure is not None
    assert all(
        node.source_anchor_ids == ("obs_doc", "src_user")
        for node in result.value.proposal.structure.nodes
    )


def test_durable_architect_success_persists_and_replays_the_typed_proposal() -> None:
    request = _request()
    proposal_json = json.dumps(_proposal_dict(), sort_keys=True, separators=(",", ":"))
    settled: list[object] = []

    class LogicalRequest:
        structured_prompt = RuntimeModelStructuredPrompt.create(
            system_prompt=_AUXILIARY_GRAPH_ARCHITECT_SYSTEM_PROMPT,
            user_content=serialize_auxiliary_graph_architect_prompt(request),
        )
        output_repair_protocol = (
            RuntimeModelOutputRepairProtocol.FOUR_MESSAGE_WHOLE_RESPONSE_REGENERATION
        )
        typed_result_contract = "auxiliary-graph-revision-proposal-v2"

    class DurableCall:
        semantic_call_id = request.logical_call_id
        logical_request = LogicalRequest()

        def __init__(self, *, replay: bool) -> None:
            self.replay = replay

        def require_current_state(self) -> None:
            return None

        def reserve(self, *, turn_id: str) -> object:
            return object()

        def replay_succeeded_result(self) -> DurableModelCallReplay | None:
            if not self.replay:
                return None
            return DurableModelCallReplay(
                model_result=ModelResult(
                    reply=proposal_json,
                    provider="test",
                    model="test",
                    latency_ms=0,
                    finish_reason="replayed_typed_result",
                    model_call_id=f"{request.logical_call_id}:physical:1",
                ),
                physical_ordinal=1,
            )

        def recover_output_repair_feedback(self) -> None:
            return None

        def begin_physical_attempt(self, **_values: object) -> object:
            return type(
                "Physical",
                (),
                {
                    "physical_attempt_id": "architect_physical_01",
                    "physical_ordinal": 1,
                    "provider": "test",
                    "model": "test",
                    "model_call_id": f"{request.logical_call_id}:physical:1",
                },
            )()

        def settle_physical_attempt(self, **values: object) -> None:
            settled.append(values.get("typed_result"))

        def typed_result_payload(
            self, *, model_result: object, value: object
        ) -> object:
            assert isinstance(model_result, ModelResult)
            return value

        def success_fingerprint(self, _result: object) -> str:
            return "success_fingerprint"

        def failure_fingerprint(self, **_values: object) -> str:
            return "failure_fingerprint"

        def terminal_state_error(self, message: str) -> DurableModelCallTerminalState:
            return DurableModelCallTerminalState(message)

    fresh = request_auxiliary_graph_architect(
        request,
        invocation_turn_id="turn_architect_01",
        provider=as_prepared_test_provider(
            lambda _system, _user, **kwargs: _model_result(
                proposal_json,
                str(kwargs["model_call_id"]),
            )
        ),
        emit=lambda _event: None,
        durable_call=DurableCall(replay=False),  # type: ignore[arg-type]
    )

    assert len(settled) == 1
    assert isinstance(settled[0], AuxiliaryGraphRevisionProposal)
    assert fresh.value.action is AuxiliaryGraphArchitectAction.CREATE_REVISION

    replayed = request_auxiliary_graph_architect(
        request,
        invocation_turn_id="turn_architect_02",
        provider=as_prepared_test_provider(
            lambda *_args, **_kwargs: pytest.fail("replay reached provider")
        ),
        emit=lambda _event: None,
        durable_call=DurableCall(replay=True),  # type: ignore[arg-type]
    )

    assert replayed.replayed is True
    assert replayed.value == fresh.value


@pytest.mark.parametrize(
    "status",
    tuple(status for status in AuxiliaryPlanningGoalStatus if status.value != "active"),
)
def test_only_active_goal_can_create_an_architect_request(
    status: AuxiliaryPlanningGoalStatus,
) -> None:
    active = _request()
    with pytest.raises(ValidationError, match="only an active planning goal"):
        AuxiliaryGraphArchitectRequest.create(
            architect_request_id=active.architect_request_id,
            logical_call_id=active.logical_call_id,
            architect_profile_id=active.architect_profile_id,
            goal=active.goal.model_copy(update={"status": status}),
            prompt_payload=active.prompt_payload,
        )


@pytest.mark.parametrize(
    ("current", "proposal", "expected_action"),
    (
        (
            True,
            _proposal_dict(disposition="continue_current", current_revision=3),
            AuxiliaryGraphArchitectAction.CONTINUE_CURRENT,
        ),
        (
            True,
            _proposal_dict(disposition="revise_revision", current_revision=3),
            AuxiliaryGraphArchitectAction.REVISE_REVISION,
        ),
        (
            True,
            _proposal_dict(disposition="supersede_and_rebase", current_revision=3),
            AuxiliaryGraphArchitectAction.SUPERSEDE_AND_REBASE,
        ),
        (
            False,
            _proposal_dict(
                disposition="terminal_fail",
                failure_reason="No legal planning graph can satisfy the frozen authority.",
            ),
            AuxiliaryGraphArchitectAction.TERMINAL_FAIL,
        ),
    ),
)
def test_typed_architect_decisions(
    current: bool,
    proposal: dict[str, object],
    expected_action: AuxiliaryGraphArchitectAction,
) -> None:
    def provider(_system: str, _user: str, **kwargs: object) -> ModelResult:
        return _model_result(proposal, str(kwargs["model_call_id"]))

    result = request_auxiliary_graph_architect(
        _request(current=current),
        invocation_turn_id="turn_architect_01",
        provider=as_prepared_test_provider(provider),
        emit=lambda _event: None,
    )

    assert result.value.action is expected_action


def test_top_level_question_is_rejected_and_initial_gap_uses_durable_gate() -> None:
    request = _request()
    question = "May the unreadable page be treated as a disclosed gap?"
    top_level = request_proposal(_proposal_dict(question=question))
    with pytest.raises(AuxiliaryGraphArchitectGuardError, match="top-level"):
        validate_auxiliary_graph_architect_proposal(
            request=request,
            proposal=top_level,
        )

    missing_gate_payload = _proposal_dict()
    missing_gate_payload["blocking_gap_ids"] = ["gap_block"]
    with pytest.raises(AuxiliaryGraphArchitectGuardError, match="user_gate"):
        validate_auxiliary_graph_architect_proposal(
            request=request,
            proposal=request_proposal(missing_gate_payload),
        )

    durable_payload = _proposal_dict(
        structure=_structure_with_user_gate(with_origins=False),
    )
    durable_payload["blocking_gap_ids"] = ["gap_block"]
    durable = validate_auxiliary_graph_architect_proposal(
        request=request,
        proposal=request_proposal(durable_payload),
    )
    assert durable.requested_user_question is None
    assert any(
        node.required
        and node.node_kind is AuxiliaryNodeKind.CLARIFY
        and node.executor_kind is AuxiliaryNodeExecutorKind.USER_GATE
        for node in durable.structure.nodes  # type: ignore[union-attr]
    )


def test_unavailable_capability_is_repaired_under_the_same_logical_call() -> None:
    request = _request(
        capabilities=_capabilities(document_available=False, include_fallback=True)
    )
    invalid = _proposal_dict(structure=_structure(capability_alias="document_read"))
    valid = _proposal_dict(structure=_structure(capability_alias="fallback_read"))
    calls: list[str] = []

    def provider(_system: str, _user: str, **kwargs: object) -> ModelResult:
        calls.append(str(kwargs["model_call_id"]))
        return _model_result(
            invalid if len(calls) == 1 else valid,
            str(kwargs["model_call_id"]),
        )

    result = request_auxiliary_graph_architect(
        request,
        invocation_turn_id="turn_architect_01",
        provider=as_prepared_test_provider(provider),
        emit=lambda _event: None,
    )

    assert result.attempts == 2
    assert calls == [request.logical_call_id, request.logical_call_id]
    assert (
        result.value.proposal.structure.nodes[0].capability_profile_id
        == "fallback_read"
    )


@pytest.mark.parametrize(
    "capability_alias",
    ("workspace_readonly", "mounted_document_cognition"),
)
def test_iterative_readonly_capability_requires_a_model_work_run(
    capability_alias: str,
) -> None:
    catalog = PlanningCapabilityCatalogProjection.create(
        capability_catalog_snapshot_id="capability_catalog_workspace",
        capability_catalog_snapshot_sha256=SHA_D,
        capabilities=(
            PlanningCapabilityDescriptor(
                capability_alias=capability_alias,
                label="Workspace read",
                description="Discover and read files with model-selected tools.",
                available=True,
                effect=PlanningCapabilityEffect.READ_ONLY,
                supported_operations=("iterative_model_tool_calls",),
                supported_resource_kinds=("pdf", "text"),
            ),
        ),
    )
    request = _request(capabilities=catalog)
    accepted = validate_auxiliary_graph_architect_proposal(
        request=request,
        proposal=request_proposal(
            _proposal_dict(
                structure=_structure(capability_alias=capability_alias)
            )
        ),
    )
    assert accepted.structure is not None

    invalid = _structure(
        capability_alias=capability_alias
    ).model_dump(mode="json")
    invalid["nodes"][0]["executor_kind"] = "host_primitive"
    with pytest.raises(AuxiliaryGraphArchitectGuardError, match="model_work_run"):
        validate_auxiliary_graph_architect_proposal(
            request=request,
            proposal=request_proposal(
                _proposal_dict(
                    structure=AuxiliaryGraphStructureProposal.model_validate(
                        invalid
                    )
                )
            ),
        )


def test_available_capability_with_unsatisfied_protected_effect_is_rejected() -> None:
    protected_catalog = PlanningCapabilityCatalogProjection.create(
        capability_catalog_snapshot_id="capability_catalog_01",
        capability_catalog_snapshot_sha256=SHA_D,
        capabilities=(
            PlanningCapabilityDescriptor(
                capability_alias="document_read",
                label="Protected document read",
                description="Read content that still requires effect approval.",
                available=True,
                effect=PlanningCapabilityEffect.PROTECTED,
                supported_operations=("read",),
                supported_resource_kinds=("pdf",),
            ),
        ),
    )
    request = _request(capabilities=protected_catalog)

    with pytest.raises(AuxiliaryGraphArchitectGuardError, match="exact protected"):
        validate_auxiliary_graph_architect_proposal(
            request=request,
            proposal=request_proposal(_proposal_dict()),
        )


def _protected_grant(
    capability_alias: str,
    *,
    resources: tuple[str, ...] = ("artifact_doc",),
    authorization_aliases: tuple[str, ...] = ("src_user",),
) -> AuxiliaryGraphProtectedCapabilityGrant:
    return AuxiliaryGraphProtectedCapabilityGrant(
        capability_alias=capability_alias,
        allowed_input_resource_aliases=resources,
        authorization_aliases=authorization_aliases,
        authorization_receipt_sha256=SHA_C,
    )


def _protected_catalog() -> PlanningCapabilityCatalogProjection:
    return PlanningCapabilityCatalogProjection.create(
        capability_catalog_snapshot_id="capability_catalog_01",
        capability_catalog_snapshot_sha256=SHA_D,
        capabilities=tuple(
            PlanningCapabilityDescriptor(
                capability_alias=alias,
                label=f"Protected capability {alias}",
                description="A protected capability requiring exact Host authority.",
                available=True,
                effect=PlanningCapabilityEffect.PROTECTED,
                supported_operations=("read",),
                supported_resource_kinds=("artifact",),
            )
            for alias in ("document_read", "workspace_export")
        ),
    )


def test_exact_protected_capability_grant_allows_only_its_capability_and_scope() -> None:
    request = _request(
        capabilities=_protected_catalog(),
        protected_capability_grants=(_protected_grant("document_read"),),
    )
    accepted = validate_auxiliary_graph_architect_proposal(
        request=request,
        proposal=request_proposal(_proposal_dict()),
    )
    assert accepted.structure is not None

    with pytest.raises(AuxiliaryGraphArchitectGuardError, match="exact protected"):
        validate_auxiliary_graph_architect_proposal(
            request=request,
            proposal=request_proposal(
                _proposal_dict(
                    structure=_structure(capability_alias="workspace_export")
                )
            ),
        )

    resource_overreach = _structure().model_dump(mode="json")
    resource_overreach["nodes"][0]["input_resource_aliases"] = ["src_user"]
    with pytest.raises(AuxiliaryGraphArchitectGuardError, match="resource grant"):
        validate_auxiliary_graph_architect_proposal(
            request=request,
            proposal=request_proposal(
                _proposal_dict(
                    structure=AuxiliaryGraphStructureProposal.model_validate(
                        resource_overreach
                    )
                )
            ),
        )


def test_protected_grant_cannot_expand_to_another_authorization_alias() -> None:
    request = _request(
        capabilities=_protected_catalog(),
        authority=_authority(include_secondary_authorization=True),
        authorization_aliases=("src_other", "src_user"),
        protected_capability_grants=(_protected_grant("document_read"),),
    )
    structure = _structure().model_dump(mode="json")
    structure["nodes"][0]["source_anchor_ids"] = [
        "obs_doc",
        "src_other",
        "src_user",
    ]
    with pytest.raises(AuxiliaryGraphArchitectGuardError, match="authorization grant"):
        validate_auxiliary_graph_architect_proposal(
            request=request,
            proposal=request_proposal(
                _proposal_dict(
                    structure=AuxiliaryGraphStructureProposal.model_validate(structure)
                )
            ),
        )


def test_any_node_cannot_expand_beyond_goal_authorization_or_resource_scope() -> None:
    request = _request(
        authority=_authority(include_secondary_authorization=True),
    )
    unauthorized_source = _structure().model_dump(mode="json")
    unauthorized_source["nodes"][0]["source_anchor_ids"] = [
        "obs_doc",
        "src_other",
        "src_user",
    ]
    with pytest.raises(AuxiliaryGraphArchitectGuardError, match="goal authorization"):
        validate_auxiliary_graph_architect_proposal(
            request=request,
            proposal=request_proposal(
                _proposal_dict(
                    structure=AuxiliaryGraphStructureProposal.model_validate(
                        unauthorized_source
                    )
                )
            ),
        )

    unauthorized_resource = _structure().model_dump(mode="json")
    unauthorized_resource["nodes"][0]["input_resource_aliases"] = ["src_other"]
    with pytest.raises(AuxiliaryGraphArchitectGuardError, match="goal resource"):
        validate_auxiliary_graph_architect_proposal(
            request=request,
            proposal=request_proposal(
                _proposal_dict(
                    structure=AuxiliaryGraphStructureProposal.model_validate(
                        unauthorized_resource
                    )
                )
            ),
        )


def test_structure_reference_guard_collects_all_independent_issues() -> None:
    protected_catalog = PlanningCapabilityCatalogProjection.create(
        capability_catalog_snapshot_id="capability_catalog_aggregate",
        capability_catalog_snapshot_sha256=SHA_D,
        capabilities=(
            PlanningCapabilityDescriptor(
                capability_alias="protected_read",
                label="Protected document read",
                description="Read a protected document resource.",
                available=True,
                effect=PlanningCapabilityEffect.PROTECTED,
                supported_operations=("read",),
                supported_resource_kinds=("artifact",),
            ),
        ),
    )
    request = _request(
        capabilities=protected_catalog,
        authority=_authority(include_secondary_authorization=True),
        authorization_aliases=("src_other", "src_user"),
        protected_capability_grants=(
            _protected_grant(
                "protected_read",
                resources=("artifact_doc",),
                authorization_aliases=("src_user",),
            ),
        ),
    )
    structure = _structure().model_dump(mode="json")
    observe = structure["nodes"][0]
    terminal = structure["nodes"][1]
    observe["source_anchor_ids"] = ["obs_doc", "src_user", "unknown_source"]
    observe["capability_profile_id"] = "unknown_capability"

    protected = deepcopy(observe)
    protected.update(
        {
            "node_key": "protected_analysis",
            "node_kind": "analyze",
            "title": "Analyze protected evidence",
            "objective": "Analyze the authorized protected evidence.",
            "capability_profile_id": "protected_read",
            "input_resource_aliases": ["src_user"],
            "source_anchor_ids": ["obs_doc", "src_other", "src_user"],
            "acceptance_criteria": [
                {
                    "acceptance_id": "accept_protected_analysis",
                    "criterion": "The analysis is grounded in authorized evidence.",
                    "source_anchor_ids": ["obs_doc", "src_user"],
                }
            ],
        }
    )
    terminal["source_anchor_ids"] = ["obs_doc"]
    terminal["input_resource_aliases"] = ["unknown_resource"]
    terminal["acceptance_criteria"][0]["source_anchor_ids"] = ["obs_doc"]
    structure["nodes"] = [observe, protected, terminal]
    structure["edges"] = [
        {
            "source_node_key": "observe",
            "target_node_key": "protected_analysis",
            "required": True,
        },
        {
            "source_node_key": "protected_analysis",
            "target_node_key": "terminal",
            "required": True,
        },
    ]

    with pytest.raises(AuxiliaryGraphArchitectGuardError) as captured:
        validate_auxiliary_graph_architect_proposal(
            request=request,
            proposal=request_proposal(
                _proposal_dict(
                    structure=AuxiliaryGraphStructureProposal.model_validate(
                        structure
                    )
                )
            ),
        )

    error = captured.value
    assert "source authority" in str(error)
    assert "authorization authority" in str(error)
    assert "resource alias" in str(error)
    assert "unknown capability" in str(error)
    assert "resource grant" in str(error)
    assert "authorization grant" in str(error)
    assert error.issue_coverage == "complete"
    assert [(issue.code, issue.paths) for issue in error.repair_issues] == [
        (
            "architect_authorization_scope_invalid",
            ("/structure/nodes/1/source_anchor_ids",),
        ),
        (
            "architect_authorization_scope_invalid",
            ("/structure/nodes/2/acceptance_criteria/0/source_anchor_ids",),
        ),
        (
            "architect_authorization_scope_invalid",
            ("/structure/nodes/2/source_anchor_ids",),
        ),
        (
            "architect_capability_unavailable",
            ("/structure/nodes/0/capability_profile_id",),
        ),
        (
            "architect_protected_resource_scope_invalid",
            ("/structure/nodes/1/input_resource_aliases",),
        ),
        (
            "architect_unknown_resource_alias",
            ("/structure/nodes/2/input_resource_aliases",),
        ),
        (
            "architect_unknown_source_authority",
            ("/structure/nodes/0/source_anchor_ids",),
        ),
    ]


def test_structure_reference_issue_limit_reports_truncated_not_complete() -> None:
    structure = _structure().model_dump(mode="json")
    for node_index, node in enumerate(structure["nodes"]):
        node["source_anchor_ids"] = ["obs_doc"]
        node["acceptance_criteria"] = [
            {
                "acceptance_id": f"node_{node_index}_acceptance_{index:02d}",
                "criterion": "The result is grounded in frozen evidence.",
                "source_anchor_ids": ["obs_doc"],
            }
            for index in range(40)
        ]

    with pytest.raises(AuxiliaryGraphArchitectGuardError) as captured:
        validate_auxiliary_graph_architect_proposal(
            request=_request(),
            proposal=request_proposal(
                _proposal_dict(
                    structure=AuxiliaryGraphStructureProposal.model_validate(
                        structure
                    )
                )
            ),
        )

    error = captured.value
    assert error.issue_coverage == "truncated"
    assert len(error.repair_issues) == 64
    assert error.omitted_issue_count == 18


def test_protected_grant_resource_scope_cannot_escape_the_planning_goal() -> None:
    with pytest.raises(ValidationError, match="grant resource scope"):
        _request(
            capabilities=_protected_catalog(),
            authority=_authority(include_secondary_authorization=True),
            protected_capability_grants=(
                _protected_grant("document_read", resources=("src_other",)),
            ),
        )


def test_continue_current_rechecks_revoked_capability_and_exact_grant() -> None:
    unavailable = _request(
        current=True,
        capabilities=_capabilities(document_available=False),
    )
    continuation = request_proposal(
        _proposal_dict(disposition="continue_current", current_revision=3)
    )
    with pytest.raises(AuxiliaryGraphArchitectGuardError, match="unavailable"):
        validate_auxiliary_graph_architect_proposal(
            request=unavailable,
            proposal=continuation,
        )

    protected_catalog = _protected_catalog()
    without_grant = _request(current=True, capabilities=protected_catalog)
    with pytest.raises(AuxiliaryGraphArchitectGuardError, match="exact protected"):
        validate_auxiliary_graph_architect_proposal(
            request=without_grant,
            proposal=continuation,
        )

    granted = _request(
        current=True,
        capabilities=protected_catalog,
        protected_capability_grants=(_protected_grant("document_read"),),
    )
    assert (
        validate_auxiliary_graph_architect_proposal(
            request=granted,
            proposal=continuation,
        )
        is continuation
    )


@pytest.mark.parametrize(
    "legacy_reply",
    (
        {
            "acceptance_updates": [],
            "action": {"kind": "call_tools", "calls": []},
        },
        {"kind": "write_output_window", "content": "forbidden"},
        {"disposition": "submit_task_graph", "proposal": {}},
    ),
)
def test_legacy_attempt_actions_are_nonretryable_and_never_expose_tools(
    legacy_reply: dict[str, object],
) -> None:
    calls = 0

    def provider(_system: str, _user: str, **kwargs: object) -> ModelResult:
        nonlocal calls
        calls += 1
        return _model_result(
            legacy_reply,
            str(kwargs["model_call_id"]),
        )

    with pytest.raises(ModelGatewayError) as captured:
        request_auxiliary_graph_architect(
            _request(),
            invocation_turn_id="turn_architect_01",
            provider=as_prepared_test_provider(provider),
            emit=lambda _event: None,
        )

    assert captured.value.code == "MODEL_BAD_RESPONSE"
    assert captured.value.retryable is False
    assert calls == 1


def test_hard_budget_stops_before_provider() -> None:
    request = _request(
        budget=_budget(
            logical_model_calls=PlanningEpisodeBudgetProfile().hard_logical_model_calls
        )
    )
    calls = 0

    def provider(*_args: object, **_kwargs: object) -> ModelResult:
        nonlocal calls
        calls += 1
        raise AssertionError("hard budget reached the provider")

    with pytest.raises(AuxiliaryGraphArchitectBudgetExhausted) as captured:
        request_auxiliary_graph_architect(
            request,
            invocation_turn_id="turn_architect_01",
            provider=as_prepared_test_provider(provider),
            emit=lambda _event: None,
        )

    assert "logical_model_calls" in captured.value.dimensions
    assert calls == 0


def test_current_graph_budget_gauges_are_exact_and_checked_before_provider() -> None:
    bad_budget = PlanningEpisodeBudget.create(
        budget_ledger_id="budget_01",
        goal_id="goal_01",
        usage=PlanningEpisodeBudgetUsage(
            current_graph_nodes=1,
            current_graph_depth=2,
        ),
    )
    request = _request(current=True, budget=bad_budget)

    with pytest.raises(AuxiliaryGraphArchitectGuardError, match="gauges"):
        request_auxiliary_graph_architect(
            request,
            invocation_turn_id="turn_architect_01",
            provider=as_prepared_test_provider(
                lambda *_args, **_kwargs: pytest.fail("provider was called")
            ),
            emit=lambda _event: None,
        )


def test_post_parse_budget_rejects_graph_above_frozen_profile() -> None:
    profile = PlanningEpisodeBudgetProfile(hard_current_graph_nodes=1)
    request = _request(budget=_budget(profile=profile))
    proposal = _proposal_dict()
    calls = 0

    def provider(_system: str, _user: str, **kwargs: object) -> ModelResult:
        nonlocal calls
        calls += 1
        return _model_result(proposal, str(kwargs["model_call_id"]))

    with pytest.raises(ModelGatewayError) as captured:
        request_auxiliary_graph_architect(
            request,
            invocation_turn_id="turn_architect_01",
            provider=as_prepared_test_provider(provider),
            emit=lambda _event: None,
        )

    assert captured.value.code == "MODEL_BAD_RESPONSE"
    assert calls == 6


def test_guard_rejects_stale_revision_unknown_authority_resource_gap_and_origin() -> None:
    request = _request(current=True)
    base = _proposal_dict(disposition="revise_revision", current_revision=3)

    stale = deepcopy(base)
    stale["expected_current_auxiliary_graph_revision"] = 2
    with pytest.raises(AuxiliaryGraphArchitectGuardError, match="stale"):
        validate_auxiliary_graph_architect_proposal(
            request=request,
            proposal=request_proposal(stale),
        )

    unknown_authority = deepcopy(base)
    unknown_authority["structure"]["nodes"][0]["source_anchor_ids"] = [  # type: ignore[index]
        "obs_doc",
        "src_user",
        "unknown_source",
    ]
    with pytest.raises(AuxiliaryGraphArchitectGuardError, match="source authority"):
        validate_auxiliary_graph_architect_proposal(
            request=request,
            proposal=request_proposal(unknown_authority),
        )

    acceptance_without_authority = deepcopy(base)
    acceptance_without_authority["structure"]["nodes"][0][  # type: ignore[index]
        "acceptance_criteria"
    ][0]["source_anchor_ids"] = ["obs_doc"]
    with pytest.raises(AuxiliaryGraphArchitectGuardError, match="Acceptance"):
        validate_auxiliary_graph_architect_proposal(
            request=request,
            proposal=request_proposal(acceptance_without_authority),
        )

    unknown_resource = deepcopy(base)
    unknown_resource["structure"]["nodes"][0]["input_resource_aliases"] = [  # type: ignore[index]
        "unknown_resource"
    ]
    with pytest.raises(AuxiliaryGraphArchitectGuardError, match="resource alias"):
        validate_auxiliary_graph_architect_proposal(
            request=request,
            proposal=request_proposal(unknown_resource),
        )

    unknown_origin = deepcopy(base)
    unknown_origin["structure"]["nodes"][0]["origin_node_alias"] = "old_unknown"  # type: ignore[index]
    with pytest.raises(AuxiliaryGraphArchitectGuardError, match="origin"):
        validate_auxiliary_graph_architect_proposal(
            request=request,
            proposal=request_proposal(unknown_origin),
        )

    question = _proposal_dict(
        disposition="continue_current",
        current_revision=3,
        question="Can planning continue?",
    )
    with pytest.raises(AuxiliaryGraphArchitectGuardError, match="top-level"):
        validate_auxiliary_graph_architect_proposal(
            request=request,
            proposal=request_proposal(question),
        )

    unknown_gap = _proposal_dict(
        disposition="continue_current",
        current_revision=3,
    )
    unknown_gap["blocking_gap_ids"] = ["unknown_gap"]
    with pytest.raises(AuxiliaryGraphArchitectGuardError, match="unknown ContextArtifact gap"):
        validate_auxiliary_graph_architect_proposal(
            request=request,
            proposal=request_proposal(unknown_gap),
        )


def test_replan_trigger_is_self_authenticating_and_bound_to_the_full_current_graph() -> None:
    trigger = _replan_trigger()
    request = _request(current=True, replan_trigger=trigger)

    assert trigger == _replan_trigger()
    assert request.prompt_payload.replan_trigger == trigger
    assert (
        trigger.rejected_task_graph_proposal_sha256
        == canonical_task_graph_revision_proposal_sha256(
            trigger.rejected_task_graph_proposal
        )
    )
    assert json.loads(serialize_auxiliary_graph_architect_prompt(request))[
        "replan_trigger"
    ] == trigger.model_dump(mode="json")

    with pytest.raises(ValidationError, match="trigger hash"):
        AuxiliaryGraphArchitectReplanTrigger.model_validate(
            trigger.model_dump(mode="json")
            | {"semantic_settlement_sha256": SHA_D}
        )
    with pytest.raises(ValidationError, match="current AuxiliaryGraph"):
        AuxiliaryGraphArchitectPrompt.create(
            goal=request.prompt_payload.goal,
            authority=request.prompt_payload.authority,
            context_artifacts=request.prompt_payload.context_artifacts,
            capabilities=request.prompt_payload.capabilities,
            budget=request.prompt_payload.budget,
            current_revision=request.prompt_payload.current_revision,
            replan_trigger=AuxiliaryGraphArchitectReplanTrigger.create(
                expected_current_auxiliary_graph_revision=2,
                expected_current_structure_sha256=(
                    trigger.expected_current_structure_sha256
                ),
                semantic_host_disposition=trigger.semantic_host_disposition,
                semantic_settlement_id=trigger.semantic_settlement_id,
                semantic_settlement_sha256=trigger.semantic_settlement_sha256,
                semantic_prompt_payload_sha256=(
                    trigger.semantic_prompt_payload_sha256
                ),
                rejected_task_graph_proposal=(
                    trigger.rejected_task_graph_proposal
                ),
                reviewers=trigger.reviewers,
            ),
        )


def test_replan_trigger_requires_ordered_non_pass_reviewer_findings() -> None:
    with pytest.raises(ValidationError, match="semantic disposition"):
        _replan_trigger(
            disposition=TaskGraphSemanticVerificationDisposition.BLOCKED,
            verdict=TaskGraphSemanticVerificationVerdict.FAIL,
        )

    with pytest.raises(ValidationError, match="reviewer ordinals"):
        AuxiliaryGraphArchitectReplanTrigger.create(
            expected_current_auxiliary_graph_revision=3,
            expected_current_structure_sha256=SHA_D,
            semantic_host_disposition=(
                TaskGraphSemanticVerificationDisposition.REVISE
            ),
            semantic_settlement_id="semantic_settlement_01",
            semantic_settlement_sha256=SHA_A,
            semantic_prompt_payload_sha256=SHA_B,
            rejected_task_graph_proposal=_rejected_task_graph(),
            reviewers=(
                AuxiliaryGraphReplanReviewerFindings(
                    reviewer_ordinal=2,
                    semantic_result_sha256=SHA_C,
                    items=_semantic_items(),
                ),
            ),
        )

    with pytest.raises(ValidationError, match="non-pass"):
        AuxiliaryGraphArchitectReplanTrigger.create(
            expected_current_auxiliary_graph_revision=3,
            expected_current_structure_sha256=SHA_D,
            semantic_host_disposition=(
                TaskGraphSemanticVerificationDisposition.REVISE
            ),
            semantic_settlement_id="semantic_settlement_01",
            semantic_settlement_sha256=SHA_A,
            semantic_prompt_payload_sha256=SHA_B,
            rejected_task_graph_proposal=_rejected_task_graph(),
            reviewers=(
                AuxiliaryGraphReplanReviewerFindings(
                    reviewer_ordinal=1,
                    semantic_result_sha256=SHA_C,
                    items=tuple(
                        TaskGraphSemanticVerificationItem(
                            dimension=dimension,
                            verdict="pass",
                            failure_scope=None,
                            finding="This semantic dimension passed.",
                        )
                        for dimension in TaskGraphSemanticVerificationDimension
                    ),
                ),
            ),
        )


@pytest.mark.parametrize(
    ("proposal", "message"),
    (
        (
            _proposal_dict(disposition="continue_current", current_revision=3),
            "must revise",
        ),
        (
            _proposal_dict(disposition="revise_revision", current_revision=3),
            "verification_failed",
        ),
        (
            _proposal_dict(
                disposition="revise_revision",
                current_revision=3,
                question="Should the blocking gap be resolved by the user?",
            ),
            "top-level user question",
        ),
    ),
)
def test_replan_guard_is_reason_bound_and_only_accepts_a_complete_revision(
    proposal: dict[str, object],
    message: str,
) -> None:
    request = _request(current=True, replan_trigger=_replan_trigger())
    if proposal["disposition"] == "revise_revision" and proposal[
        "requested_user_question"
    ] is not None:
        proposal["revision_reason"] = "verification_failed"

    with pytest.raises(AuxiliaryGraphArchitectGuardError, match=message):
        validate_auxiliary_graph_architect_proposal(
            request=request,
            proposal=request_proposal(proposal),
        )

    accepted = _proposal_dict(
        disposition="revise_revision",
        current_revision=3,
    )
    accepted["revision_reason"] = "verification_failed"
    validate_auxiliary_graph_architect_proposal(
        request=request,
        proposal=request_proposal(accepted),
    )


def test_blocked_replan_requires_a_user_gate_inside_the_revised_dag() -> None:
    trigger = _replan_trigger(
        disposition=TaskGraphSemanticVerificationDisposition.BLOCKED,
        verdict=TaskGraphSemanticVerificationVerdict.INSUFFICIENT_EVIDENCE,
    )
    request = _request(current=True, replan_trigger=trigger)
    without_gate = _proposal_dict(
        disposition="revise_revision",
        current_revision=3,
    )
    without_gate["revision_reason"] = "verification_failed"
    with pytest.raises(AuxiliaryGraphArchitectGuardError, match="user_gate"):
        validate_auxiliary_graph_architect_proposal(
            request=request,
            proposal=request_proposal(without_gate),
        )

    with_gate = _proposal_dict(
        disposition="revise_revision",
        current_revision=3,
        structure=_structure_with_user_gate(),
    )
    with_gate["revision_reason"] = "verification_failed"
    validate_auxiliary_graph_architect_proposal(
        request=request,
        proposal=request_proposal(with_gate),
    )


def test_request_and_current_projection_hashes_reject_tampering() -> None:
    request = _request(current=True)
    with pytest.raises(ValidationError, match="request hash"):
        AuxiliaryGraphArchitectRequest.model_validate(
            request.model_dump(mode="json")
            | {"architect_profile_id": "different_profile"}
        )
    current = request.prompt_payload.current_revision
    assert current is not None
    with pytest.raises(ValidationError, match="projection hash"):
        AuxiliaryGraphCurrentRevisionProjection.model_validate(
            current.model_dump(mode="json")
            | {"source_structure_sha256": SHA_A}
        )

    protected = _request(
        capabilities=_protected_catalog(),
        protected_capability_grants=(_protected_grant("document_read"),),
    )
    forged_prompt = protected.prompt_payload.model_dump(mode="json")
    forged_prompt["protected_capability_grants"][0][
        "authorization_receipt_sha256"
    ] = SHA_A
    with pytest.raises(ValidationError, match="prompt hash"):
        AuxiliaryGraphArchitectPrompt.model_validate(forged_prompt)


def test_duplicate_json_keys_are_rejected_without_silent_collapse() -> None:
    valid = json.dumps(_proposal_dict())
    duplicated = valid.replace(
        '"disposition": "create_revision"',
        '"disposition": "create_revision", "disposition": "terminal_fail"',
        1,
    )
    calls = 0

    def provider(_system: str, _user: str, **kwargs: object) -> ModelResult:
        nonlocal calls
        calls += 1
        return _model_result(duplicated, str(kwargs["model_call_id"]))

    with pytest.raises(ModelGatewayError):
        request_auxiliary_graph_architect(
            _request(),
            invocation_turn_id="turn_architect_01",
            provider=as_prepared_test_provider(provider),
            emit=lambda _event: None,
        )
    assert calls == 6


def test_deeply_nested_json_is_normalized_to_a_bounded_bad_response() -> None:
    reply = "[" * 2_000 + "0" + "]" * 2_000
    calls = 0

    def provider(_system: str, _user: str, **kwargs: object) -> ModelResult:
        nonlocal calls
        calls += 1
        return _model_result(reply, str(kwargs["model_call_id"]))

    with pytest.raises(ModelGatewayError) as captured:
        request_auxiliary_graph_architect(
            _request(),
            invocation_turn_id="turn_architect_01",
            provider=as_prepared_test_provider(provider),
            emit=lambda _event: None,
        )

    assert captured.value.code == "MODEL_BAD_RESPONSE"
    assert calls == 6


def request_proposal(value: dict[str, object]):
    from personagraph.l2.auxiliary_graph import AuxiliaryGraphRevisionProposal

    return AuxiliaryGraphRevisionProposal.model_validate(value)
