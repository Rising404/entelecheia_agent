"""生产辅助图用于实现 AuxiliaryGraph 目标。

图驱动器故意停在命名的应用端口上。此模块实现终端端口而不压缩其持久化阶段：重建验证的语义材料，结算独立评审团，密封终端提案，并原子性提交精确的目标 TaskGraph 版本。每个身份都源自不可变的图权威状态，因此响应丢失恢复重新进入相同的回执。

对于正数基础，Store 拥有安全提示的 TaskGraph ``N`` 投影及其私有别名到节点绑定。模型拥有的终端候选者仅提供不透明的血统别名；语义门在 Host 可以密封和提交它之前会审查完整的``N -> N+1`` 转换。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from functools import partial

from personagraph.l2.auxiliary_graph import (
    PlanningAuthorityClass,
    PlanningAuthorityProjection,
    PlanningCapabilityCatalogProjection,
    PlanningGoalPromptContext,
    TaskGraphRevisionCandidate,
    TaskGraphSemanticTerminalRoute,
    TaskGraphSemanticVerificationDisposition,
    TaskGraphSemanticVerificationPromptPayload,
    derive_task_graph_semantic_terminal_route,
    required_task_graph_semantic_reviewer_count,
)
from personagraph.l2.auxiliary_graph.contracts import (
    is_task_graph_semantic_user_information_block,
)
from personagraph.l2.task_graph import (
    InSessionTaskGraphLimits,
    InSessionTaskGraphRevisionProposal,
    InSessionTaskGraphRevisionValidationContext,
    bind_used_evidence_to_required_acceptance_coverage,
)
from personagraph.l2.task_graph.production_gate import (
    TaskGraphAcceptanceRef,
    TaskGraphNodeExecutionRequirement,
    TaskGraphPlanningGap,
    TaskGraphProductionCapabilityContext,
    TaskGraphProductionEvaluationContext,
    TaskGraphProductionEvaluation,
    TaskGraphProductionFailureCode,
    TaskGraphProductionFreshness,
    evaluate_task_graph_production,
)
from personagraph.session import store
from personagraph.session.l2_store import auxiliary_graph as auxiliary_graph_store
from personagraph.session.l2_store import planning as planning_store
from personagraph.session.l2_store import semantic_verification as semantic_store
from personagraph.session.l2_store import task_delivery as task_delivery_store
from personagraph.session.l2_store import task_graph as task_graph_store
from personagraph.session.l2_store import terminal as terminal_store
from personagraph.session.l2_store import work_run as work_run_store
from personagraph.l2.work_run import (
    DownstreamVerificationDisposition,
    DownstreamVerificationFeedback,
    NodeVerificationResult,
)
from personagraph.l2.auxiliary_execution.driver import (
    AuxiliaryGraphDriverAction,
    decide_auxiliary_graph_driver_step,
)
from personagraph.l2.auxiliary_execution.planning.mounted_document_authority import (
    MountedDocumentPlanningAuthorityError,
    build_mounted_document_authority_projection,
    freeze_mounted_document_planning_authority,
    recover_task_scoped_managed_document_ids,
)
from ..planning.mounted_visual_resource import (
    MountedVisualPlanningAuthorityError,
)
from personagraph.l2.auxiliary_execution.terminal.id_contracts import (
    AuxiliaryTerminalCandidateSemanticIdPlan,
    AuxiliaryTerminalIdPlan,
    _sha256_value,
    derive_auxiliary_terminal_candidate_semantic_ids,
    derive_auxiliary_terminal_ids,
)
from personagraph.l2.auxiliary_execution.adapters.model_authority import (
    create_auxiliary_semantic_reviewer_model_call_authority,
)
from personagraph.l2.auxiliary_execution.planning.profiles import (
    build_auxiliary_planning_capability_catalog,
)
from personagraph.l2.auxiliary_execution.verification.model_provider import (
    build_auxiliary_semantic_structured_provider,
)
from personagraph.l2.auxiliary_execution.verification.controller import (
    AuxiliarySemanticModelCallAuthorityFactory,
    AuxiliarySemanticReviewerIdPlan,
    AuxiliarySemanticVerificationRequest,
    AuxiliarySemanticVerificationResult,
    AuxiliarySemanticVerificationStatus,
    AuxiliaryTerminalCandidateBinding,
    AuxiliaryTerminalCandidateSemanticReview,
    review_auxiliary_terminal_candidate_semantics,
    run_auxiliary_semantic_verification,
)
from personagraph.runtime.turn_deadline import TurnDeadline
from personagraph.runtime.model_calls.contracts import RuntimeModelLedgerStore
from personagraph.l2.task_execution.verification.decision import NodeVerificationContext
from personagraph.l2.auxiliary_execution.verification.task_graph_semantic import (
    TaskGraphSemanticVerificationStructuredProvider,
)
from personagraph.tools.workspace.session_read_source import (
    build_session_workspace_readonly_runtime,
)
from personagraph.runtime.turn_events import TurnEvent


_SEMANTIC_PROFILE_ID = "auxiliary_v2_semantic_production_v1"
_PRODUCTION_STALE_FAILURES = frozenset(
    {
        TaskGraphProductionFailureCode.BASE_AUTHORITY_STALE,
        TaskGraphProductionFailureCode.CURRENT_AUTHORITY_STALE,
        TaskGraphProductionFailureCode.SOURCE_MANIFEST_STALE,
        TaskGraphProductionFailureCode.CAPABILITY_CATALOG_STALE,
    }
)


class AuxiliaryTerminalCompositionStatus(StrEnum):
    COMMITTED = "committed"
    ALREADY_COMMITTED = "already_committed"
    WAITING_EXTERNAL = "waiting_external"
    TURN_LIMIT_REACHED = "turn_limit_reached"
    MODEL_INTERRUPTED = "model_interrupted"
    REVISION_REQUIRED = "revision_required"
    FAILED_CLOSED = "failed_closed"


@dataclass(frozen=True, slots=True)
class AuxiliaryTerminalCompositionResult:
    status: AuxiliaryTerminalCompositionStatus
    reason_code: str
    id_plan: AuxiliaryTerminalIdPlan
    semantic_result: AuxiliarySemanticVerificationResult | None = None
    terminal_seal: terminal_store.AuxiliaryTerminalSealResult | None = None
    task_graph_commit: terminal_store.AuxiliaryTaskGraphCommitResult | None = None


class AuxiliaryTerminalCompositionError(RuntimeError):
    code = "auxiliary_v2_terminal_composition_rejected"


class AuxiliaryTerminalCandidateSemanticRouteRequired(RuntimeError):
    """候选者需要一个专用的非 WorkRun 路线事务。"""

    def __init__(
        self,
        *,
        route: TaskGraphSemanticTerminalRoute,
        review: AuxiliaryTerminalCandidateSemanticReview,
        settlement: semantic_store.StoredAuxiliarySemanticQuorumSettlement
        | None = None,
    ) -> None:
        if route not in {
            TaskGraphSemanticTerminalRoute.REPLAN_AUXILIARY,
            TaskGraphSemanticTerminalRoute.BLOCKED,
        }:
            raise ValueError("candidate route exception requires replan or blocked")
        self.route = route
        self.review = review
        if (
            route is TaskGraphSemanticTerminalRoute.REPLAN_AUXILIARY
            and settlement is None
        ):
            raise ValueError("candidate replan route requires durable settlement")
        self.settlement = settlement
        self.code = (
            "terminal_candidate_semantic_replan_required"
            if route is TaskGraphSemanticTerminalRoute.REPLAN_AUXILIARY
            else "terminal_candidate_semantic_blocked"
        )
        super().__init__(self.code)


def _semantic_revision_reason_code(
    disposition: TaskGraphSemanticVerificationDisposition,
) -> str:
    if disposition is TaskGraphSemanticVerificationDisposition.BLOCKED:
        return "semantic_review_blocked_by_insufficient_evidence"
    if disposition is TaskGraphSemanticVerificationDisposition.REVISE:
        return "semantic_review_requires_revision"
    raise ValueError("a passing semantic settlement does not require revision")


def _select_semantic_model_call_authority_factory(
    *,
    factory: AuxiliarySemanticModelCallAuthorityFactory | None,
    model_ledger_store: RuntimeModelLedgerStore | None,
) -> AuxiliarySemanticModelCallAuthorityFactory:
    if factory is not None:
        return factory
    if model_ledger_store is None:
        raise AuxiliaryTerminalCompositionError(
            "runtime_model_ledger_unavailable"
        )
    return partial(
        create_auxiliary_semantic_reviewer_model_call_authority,
        ledger_store=model_ledger_store,
    )


def run_auxiliary_terminal_candidate_semantic_gate(
    *,
    session_id: str,
    turn_id: str,
    task_id: str,
    validation_context: InSessionTaskGraphRevisionValidationContext,
    context: NodeVerificationContext,
    node_result: NodeVerificationResult,
    emit: Callable[[TurnEvent], object],
    semantic_provider: TaskGraphSemanticVerificationStructuredProvider | None = None,
    semantic_model_call_authority_factory: (
        AuxiliarySemanticModelCallAuthorityFactory | None
    ) = None,
    model_ledger_store: RuntimeModelLedgerStore | None = None,
    desired_output: str = "完整、可执行、可验证的 TaskGraph",
    deadline: TurnDeadline | None = None,
    allowed_managed_document_ids: tuple[str, ...] | None = None,
) -> tuple[DownstreamVerificationFeedback, ...]:
    """无需冻结完成情况地审查一个节点-PASS 终端候选者。"""

    if not node_result.all_pass:
        raise AuxiliaryTerminalCompositionError(
            "terminal candidate gate requires a node-PASS candidate"
        )
    if (
        context.session_id != session_id
        or context.invocation_turn_id != turn_id
        or context.subject.task_id != task_id
        or node_result.work_run_id != context.work_run_id
        or node_result.submitted_attempt_id != context.submitted_attempt_id
        or node_result.output_revision
        != context.locked_output_window.output_revision
    ):
        raise AuxiliaryTerminalCompositionError(
            "terminal candidate gate crossed its WorkRun binding"
        )
    details = auxiliary_graph_store.get_auxiliary_graph_for_task(
        session_id=session_id,
        insession_task_id=task_id,
    )
    if details is None:
        raise AuxiliaryTerminalCompositionError(
            "terminal candidate gate lost current graph authority"
        )
    support = terminal_store.project_auxiliary_terminal_semantic_support(
        session_id=session_id,
        invocation_turn_id=turn_id,
        task_id=task_id,
    )
    if support.validation_context != validation_context:
        raise AuxiliaryTerminalCompositionError(
            "terminal candidate semantic support crossed validation authority"
        )
    proposal, lineage = _parse_candidate_output(
        context.locked_output_window.content,
        positive_base=(details.base_task_graph_revision is not None),
    )
    base_projection = (
        None
        if details.base_task_graph_revision is None
        else task_graph_store.project_task_graph_semantic_base(
            session_id=session_id,
            task_id=task_id,
            graph_revision=details.base_task_graph_revision,
        )
    )
    base_authority, capabilities = _rebuild_planning_prompt_projections(
        session_id=session_id,
        task_id=task_id,
        details=details,
        allowed_managed_document_ids=allowed_managed_document_ids,
    )
    prompt_payload = _semantic_prompt_payload(
        details=details,
        desired_output=desired_output,
        base_authority=base_authority,
        capabilities=capabilities,
        base_task_graph=(
            None if base_projection is None else base_projection.snapshot
        ),
        context_artifacts=support.context_artifacts,
        observation_source_cards=support.observation_source_cards,
        task_graph_proposal=proposal,
        lineage=lineage,
    )
    task = task_graph_store.get_insession_task_details(session_id, task_id)
    if task is None:
        raise AuxiliaryTerminalCompositionError(
            "terminal candidate production gate lost its Task authority"
        )
    production_evaluation = _evaluate_terminal_candidate_production(
        proposal=proposal,
        validation_context=_prepare_terminal_production_validation_context(
            proposal=proposal,
            validation_context=validation_context,
            details=details,
        ),
        prompt_payload=prompt_payload,
        task_current_graph_revision=task.current_graph_revision,
    )
    if not production_evaluation.passed and (
        TaskGraphProductionFailureCode.BLOCKING_GAP
        not in production_evaluation.failure_codes
    ):
        if _PRODUCTION_STALE_FAILURES.intersection(
            production_evaluation.failure_codes
        ):
            raise AuxiliaryTerminalCompositionError(
                "terminal candidate production authority became stale"
            )
        finding = _production_failure_summary(production_evaluation)
        return (
            DownstreamVerificationFeedback(
                gate_id="task_graph_production",
                disposition=DownstreamVerificationDisposition.RETRY_ATTEMPT,
                finding=finding,
                repair_objective=(
                    "Revise only the terminal TaskGraph proposal to satisfy "
                    "the deterministic production gate. Every required source "
                    "anchor must be referenced by both a relevant node and a "
                    f"relevant acceptance criterion. Failures: {finding}"
                )[:4_000],
                source_result_id=(
                    "taskgraph-production-"
                    + production_evaluation.evaluation_sha256[:32]
                ),
                source_result_sha256=(
                    production_evaluation.evaluation_sha256
                ),
                affected_subject_ids=(context.subject.node_id,),
            ),
        )
    review_policy = semantic_store.derive_auxiliary_semantic_review_policy(
        prompt_payload=prompt_payload
    )
    reviewer_count = required_task_graph_semantic_reviewer_count(
        prompt_payload=prompt_payload,
        review_policy=review_policy,
    )
    candidate_binding = AuxiliaryTerminalCandidateBinding(
        work_run_id=context.work_run_id,
        submitted_attempt_id=context.submitted_attempt_id,
        node_verification_request_id=context.verification_request_id,
        output_revision=context.locked_output_window.output_revision,
        output_snapshot_sha256=_sha256_value(
            context.locked_output_window.model_dump(mode="json")
        ),
    )
    id_plan = derive_auxiliary_terminal_candidate_semantic_ids(
        session_id=session_id,
        task_id=task_id,
        auxiliary_graph_id=details.auxiliary_graph_id,
        goal_id=details.goal_id,
        auxiliary_graph_revision=details.auxiliary_graph_revision,
        structure_sha256=details.structure_sha256,
        candidate_binding=candidate_binding,
    )
    frontier = auxiliary_graph_store.project_auxiliary_graph_execution_frontier(
        session_id=session_id,
        turn_id=turn_id,
        insession_task_id=task_id,
    )
    controller_request = AuxiliarySemanticVerificationRequest(
            frontier=frontier,
            prompt_payload=prompt_payload,
            capability_catalog=capabilities,
            reviewers=_candidate_reviewers(
                id_plan=id_plan,
                reviewer_count=reviewer_count,
            ),
            settlement_id=id_plan.semantic_settlement_id,
            terminal_candidate_binding=candidate_binding,
        )
    selected_provider = (
        semantic_provider or build_auxiliary_semantic_structured_provider()
    )
    selected_authority_factory = _select_semantic_model_call_authority_factory(
        factory=semantic_model_call_authority_factory,
        model_ledger_store=model_ledger_store,
    )
    review = review_auxiliary_terminal_candidate_semantics(
        controller_request,
        provider=selected_provider,
        model_call_authority_factory=selected_authority_factory,
        emit=emit,
        deadline=deadline,
    )
    review_sha256 = _sha256_value(
        {
            "candidate_review_id": id_plan.candidate_review_id,
            "route": review.route.value,
            "request_bindings": [item.binding_sha256 for item in review.requests],
            "result_hashes": [item.result_sha256 for item in review.results],
        }
    )
    if review.route is TaskGraphSemanticTerminalRoute.PASS:
        return (
            DownstreamVerificationFeedback(
                gate_id="task_graph_semantic",
                disposition=DownstreamVerificationDisposition.PASS,
                finding="Terminal TaskGraph candidate passed independent semantic review.",
                source_result_id=id_plan.candidate_review_id,
                source_result_sha256=review_sha256,
            ),
        )
    if review.route is TaskGraphSemanticTerminalRoute.RETRY_TERMINAL_ATTEMPT:
        findings = _candidate_nonpass_findings(review)
        return (
            DownstreamVerificationFeedback(
                gate_id="task_graph_semantic",
                disposition=DownstreamVerificationDisposition.RETRY_ATTEMPT,
                finding=findings,
                repair_objective=(
                    "Revise only the terminal TaskGraph proposal to address: "
                    f"{findings}"
                )[:4_000],
                source_result_id=id_plan.candidate_review_id,
                source_result_sha256=review_sha256,
                affected_subject_ids=(context.subject.node_id,),
            ),
        )
    settlement = None
    durable_route = (
        review.route is TaskGraphSemanticTerminalRoute.REPLAN_AUXILIARY
        or (
            review.route is TaskGraphSemanticTerminalRoute.BLOCKED
            and is_task_graph_semantic_user_information_block(
                requests=review.requests,
                results=review.results,
            )
        )
    )
    if durable_route:
        durable = run_auxiliary_semantic_verification(
            controller_request,
            provider=selected_provider,
            model_call_authority_factory=selected_authority_factory,
            emit=emit,
            deadline=deadline,
        )
        if (
            durable.status is not AuxiliarySemanticVerificationStatus.SETTLED
            or durable.settlement is None
            or durable.settlement.host_disposition
            is not (
                TaskGraphSemanticVerificationDisposition.BLOCKED
                if review.route is TaskGraphSemanticTerminalRoute.BLOCKED
                else TaskGraphSemanticVerificationDisposition.REVISE
            )
            or derive_task_graph_semantic_terminal_route(
                durable.settlement.results
            )
            is not review.route
        ):
            raise AuxiliaryTerminalCompositionError(
                "terminal candidate route could not seal semantic authority"
            )
        settlement = durable.settlement
    raise AuxiliaryTerminalCandidateSemanticRouteRequired(
        route=review.route,
        review=review,
        settlement=settlement,
    )


def _evaluate_terminal_candidate_production(
    *,
    proposal: InSessionTaskGraphRevisionProposal,
    validation_context: InSessionTaskGraphRevisionValidationContext,
    prompt_payload: TaskGraphSemanticVerificationPromptPayload,
    task_current_graph_revision: int | None,
) -> TaskGraphProductionEvaluation:
    """在语义模型调用前，镜像 Store 的纯生产门。

    Store 在封存交易期间重新计算此评估。此早期副本仅用于将可修复的机械故障返回给仍可变的终端 WorkRun，而不是在语义共识结算和节点完成之后才发现它们。
    """

    capabilities = prompt_payload.capabilities.capabilities
    available_ids = tuple(
        sorted(item.capability_alias for item in capabilities if item.available)
    )
    gaps: list[TaskGraphPlanningGap] = []
    for artifact in prompt_payload.context_artifacts:
        for gap in artifact.gaps:
            gaps.append(
                TaskGraphPlanningGap(
                    gap_id=gap.gap_alias,
                    blocking=gap.blocking,
                    affected_required_anchor_ids=(gap.gap_alias,),
                    mapped_node_keys=tuple(
                        node.node_key
                        for node in proposal.root.nodes
                        if gap.gap_alias in node.source_anchor_ids
                    ),
                    mapped_acceptances=tuple(
                        TaskGraphAcceptanceRef(
                            node_key=node.node_key,
                            acceptance_id=acceptance.acceptance_id,
                        )
                        for node in proposal.root.nodes
                        for acceptance in node.acceptance_criteria
                        if gap.gap_alias in acceptance.source_anchor_ids
                    ),
                )
            )
    authority_hash = prompt_payload.authority.projection_sha256
    catalog_hash = (
        prompt_payload.capabilities.capability_catalog_snapshot_sha256
    )
    return evaluate_task_graph_production(
        proposal,
        context=TaskGraphProductionEvaluationContext(
            validation_context=validation_context,
            freshness=TaskGraphProductionFreshness(
                observed_current_graph_revision=task_current_graph_revision,
                expected_authority_snapshot_sha256=(
                    prompt_payload.authority.authority_snapshot_sha256
                ),
                observed_authority_snapshot_sha256=(
                    prompt_payload.authority.authority_snapshot_sha256
                ),
                expected_source_manifest_sha256=authority_hash,
                observed_source_manifest_sha256=authority_hash,
                expected_capability_catalog_sha256=catalog_hash,
                observed_capability_catalog_sha256=catalog_hash,
            ),
            capabilities=TaskGraphProductionCapabilityContext(
                available_capability_ids=available_ids,
                satisfiable_effect_ids=available_ids,
                node_requirements=tuple(
                    TaskGraphNodeExecutionRequirement(node_key=node.node_key)
                    for node in proposal.root.nodes
                ),
            ),
            gaps=tuple(gaps),
        ),
    )


def _prepare_terminal_production_validation_context(
    *,
    proposal: InSessionTaskGraphRevisionProposal,
    validation_context: InSessionTaskGraphRevisionValidationContext,
    details,
) -> InSessionTaskGraphRevisionValidationContext:
    """应用终端封存所使用的精确可变候选约束条件。

    标准的 Store 上下文包含所有经过认证的观察结果，但最初仅需要任务创建的权威状态。一个提议如果选择引用一个观察结果，就会产生额外的审计义务：证据必须出现在一个相关节点和其 Acceptance 标准之一中。Store 独立地在封存交易中重复这个纯转换；在这里进行可以防止语义调用，这些调用对于 Store 后来必须拒绝的候选者来说是必要的。
    """

    profile = details.budget.effective_profile
    bounded = validation_context.model_copy(
        update={
            "limits": InSessionTaskGraphLimits(
                max_root_tasks=1,
                max_nodes_per_task=min(512, profile.hard_current_graph_nodes),
                max_depth=min(64, profile.hard_current_graph_depth),
            )
        }
    )
    return bind_used_evidence_to_required_acceptance_coverage(
        proposal,
        context=bounded,
    )


def _production_failure_summary(
    evaluation: TaskGraphProductionEvaluation,
) -> str:
    locations: list[str] = []
    for finding in evaluation.findings:
        location = (
            finding.anchor_id
            or finding.node_key
            or finding.acceptance_id
            or finding.capability_id
        )
        rendered = finding.code.value
        if location is not None:
            rendered += f"[{location}]"
        locations.append(rendered)
    return (
        "; ".join(locations)
        or ",".join(code.value for code in evaluation.failure_codes)
        or "unknown_production_failure"
    )[:4_000]


def _parse_candidate_output(
    content: str,
    *,
    positive_base: bool,
) -> tuple[InSessionTaskGraphRevisionProposal, tuple]:
    try:
        if positive_base:
            candidate = TaskGraphRevisionCandidate.model_validate_json(content)
            if candidate.model_dump_json() != content:
                raise ValueError("positive-base terminal candidate is not canonical")
            return candidate.proposal, candidate.lineage
        proposal = InSessionTaskGraphRevisionProposal.model_validate_json(content)
        if proposal.model_dump_json() != content:
            raise ValueError("base-null terminal proposal is not canonical")
        return proposal, ()
    except (TypeError, ValueError) as exc:
        raise AuxiliaryTerminalCompositionError(
            "terminal candidate output is not canonical typed TaskGraph material"
        ) from exc


def _semantic_prompt_payload(
    *,
    details: auxiliary_graph_store.StoredAuxiliaryGraphDetails,
    desired_output: str,
    base_authority: PlanningAuthorityProjection,
    capabilities: PlanningCapabilityCatalogProjection,
    base_task_graph,
    context_artifacts,
    observation_source_cards,
    task_graph_proposal: InSessionTaskGraphRevisionProposal,
    lineage,
) -> TaskGraphSemanticVerificationPromptPayload:
    authority = PlanningAuthorityProjection.create(
        authority_snapshot_id=base_authority.authority_snapshot_id,
        authority_snapshot_sha256=base_authority.authority_snapshot_sha256,
        cards=tuple(
            sorted(
                (
                    *(
                        card
                        for card in base_authority.cards
                        if card.authority_class
                        is PlanningAuthorityClass.AUTHORIZATION
                    ),
                    *observation_source_cards,
                ),
                key=lambda item: item.alias,
            )
        ),
    )
    authorization_aliases = tuple(
        item.alias
        for item in authority.cards
        if item.authority_class is PlanningAuthorityClass.AUTHORIZATION
    )
    return TaskGraphSemanticVerificationPromptPayload.create(
        goal=PlanningGoalPromptContext(
            goal_id=details.goal_id,
            objective=details.goal_objective,
            desired_output=desired_output,
            authorization_aliases=authorization_aliases,
        ),
        authority=authority,
        base_task_graph=base_task_graph,
        context_artifacts=context_artifacts,
        capabilities=capabilities,
        budget=details.budget,
        task_graph_proposal=task_graph_proposal,
        lineage=lineage,
    )


def _candidate_reviewers(
    *,
    id_plan: AuxiliaryTerminalCandidateSemanticIdPlan,
    reviewer_count: int,
) -> tuple[AuxiliarySemanticReviewerIdPlan, ...]:
    return tuple(
        AuxiliarySemanticReviewerIdPlan(
            reviewer_ordinal=ordinal,
            verification_profile_id=_SEMANTIC_PROFILE_ID,
            verification_request_id=id_plan.reviewer_request_ids[ordinal - 1],
            logical_call_id=id_plan.reviewer_logical_call_ids[ordinal - 1],
            verification_result_id=id_plan.reviewer_result_ids[ordinal - 1],
        )
        for ordinal in range(1, reviewer_count + 1)
    )


def _candidate_nonpass_findings(
    review: AuxiliaryTerminalCandidateSemanticReview,
) -> str:
    findings = tuple(
        item.finding
        for result in review.results
        for item in result.items
        if item.verdict.value != "pass"
    )
    return ("; ".join(findings) or "Terminal proposal requires revision.")[:4_000]


def run_auxiliary_terminal_composition(
    *,
    session_id: str,
    turn_id: str,
    task_id: str,
    emit: Callable[[TurnEvent], object],
    semantic_provider: TaskGraphSemanticVerificationStructuredProvider | None = None,
    semantic_model_call_authority_factory: (
        AuxiliarySemanticModelCallAuthorityFactory | None
    ) = None,
    model_ledger_store: RuntimeModelLedgerStore | None = None,
    desired_output: str = "完整、可执行、可验证的 TaskGraph",
    deadline: TurnDeadline | None = None,
    allowed_managed_document_ids: tuple[str, ...] | None = None,
) -> AuxiliaryTerminalCompositionResult:
    """审查、封存并提交一个已完成的 终端提议。"""

    if not callable(emit):
        raise TypeError("emit must be callable")
    task = task_graph_store.get_insession_task_details(session_id, task_id)
    details = auxiliary_graph_store.get_auxiliary_graph_for_task(
        session_id=session_id,
        insession_task_id=task_id,
    )
    if task is None or details is None:
        raise AuxiliaryTerminalCompositionError(
            "terminal composition requires a current Task and graph"
        )
    id_plan = derive_auxiliary_terminal_ids(
        session_id=session_id,
        task_id=task_id,
        auxiliary_graph_id=details.auxiliary_graph_id,
        goal_id=details.goal_id,
        auxiliary_graph_revision=details.auxiliary_graph_revision,
        structure_sha256=details.structure_sha256,
    )

    if details.goal_status == "committed" and details.revision_status == "committed":
        if task.current_graph_revision != details.target_task_graph_revision:
            raise AuxiliaryTerminalCompositionError(
                "committed AuxiliaryGraph disagrees with the current TaskGraph"
            )
        return AuxiliaryTerminalCompositionResult(
            status=AuxiliaryTerminalCompositionStatus.ALREADY_COMMITTED,
            reason_code="task_graph_already_committed",
            id_plan=id_plan,
        )

    base_projection = (
        None
        if details.base_task_graph_revision is None
        else task_graph_store.project_task_graph_semantic_base(
            session_id=session_id,
            task_id=task_id,
            graph_revision=details.base_task_graph_revision,
        )
    )

    frontier = auxiliary_graph_store.project_auxiliary_graph_execution_frontier(
        session_id=session_id,
        turn_id=turn_id,
        insession_task_id=task_id,
    )
    decision = decide_auxiliary_graph_driver_step(frontier)
    terminal_receipt_id: str
    semantic_result: AuxiliarySemanticVerificationResult | None = None
    terminal_seal: terminal_store.AuxiliaryTerminalSealResult | None = None

    if decision.action is AuxiliaryGraphDriverAction.COMMIT_READY_PROPOSAL:
        receipt = terminal_store.get_auxiliary_terminal_proposal_receipt(
            session_id=session_id,
            terminal_proposal_receipt_id=id_plan.terminal_proposal_receipt_id,
        )
        terminal_receipt_id = receipt.terminal_proposal_receipt_id
    elif decision.action is AuxiliaryGraphDriverAction.SEAL_REVISION:
        material = semantic_store.project_auxiliary_semantic_material(
            session_id=session_id,
            turn_id=turn_id,
            task_id=task_id,
        )
        semantic_support = terminal_store.project_auxiliary_terminal_semantic_support(
            session_id=session_id,
            invocation_turn_id=turn_id,
            task_id=task_id,
        )
        base_authority, capabilities = _rebuild_planning_prompt_projections(
            session_id=session_id,
            task_id=task_id,
            details=details,
            allowed_managed_document_ids=allowed_managed_document_ids,
        )
        prompt_payload = _semantic_prompt_payload(
            details=details,
            desired_output=desired_output,
            base_authority=base_authority,
            capabilities=capabilities,
            base_task_graph=(
                None if base_projection is None else base_projection.snapshot
            ),
            context_artifacts=semantic_support.context_artifacts,
            observation_source_cards=(
                semantic_support.observation_source_cards
            ),
            task_graph_proposal=material.task_graph_proposal,
            lineage=material.lineage,
        )
        review_policy = semantic_store.derive_auxiliary_semantic_review_policy(
            prompt_payload=prompt_payload
        )
        reviewer_count = required_task_graph_semantic_reviewer_count(
            prompt_payload=prompt_payload,
            review_policy=review_policy,
        )
        candidate_binding = AuxiliaryTerminalCandidateBinding(
            work_run_id=material.terminal_work_run_id,
            submitted_attempt_id=material.terminal_submitted_attempt_id,
            node_verification_request_id=(
                material.terminal_verification_request_id
            ),
            output_revision=material.terminal_output_revision,
            output_snapshot_sha256=(
                material.terminal_output_snapshot_sha256
            ),
        )
        candidate_id_plan = derive_auxiliary_terminal_candidate_semantic_ids(
            session_id=session_id,
            task_id=task_id,
            auxiliary_graph_id=details.auxiliary_graph_id,
            goal_id=details.goal_id,
            auxiliary_graph_revision=details.auxiliary_graph_revision,
            structure_sha256=details.structure_sha256,
            candidate_binding=candidate_binding,
        )
        reviewers = _candidate_reviewers(
            id_plan=candidate_id_plan,
            reviewer_count=reviewer_count,
        )
        semantic_result = run_auxiliary_semantic_verification(
            AuxiliarySemanticVerificationRequest(
                frontier=frontier,
                prompt_payload=prompt_payload,
                capability_catalog=capabilities,
                reviewers=reviewers,
                settlement_id=candidate_id_plan.semantic_settlement_id,
                terminal_candidate_binding=candidate_binding,
            ),
            provider=(
                semantic_provider
                or build_auxiliary_semantic_structured_provider()
            ),
            model_call_authority_factory=(
                _select_semantic_model_call_authority_factory(
                    factory=semantic_model_call_authority_factory,
                    model_ledger_store=model_ledger_store,
                )
            ),
            emit=emit,
            deadline=deadline,
        )
        if semantic_result.status is not AuxiliarySemanticVerificationStatus.SETTLED:
            status = {
                AuxiliarySemanticVerificationStatus.WAITING_EXTERNAL: (
                    AuxiliaryTerminalCompositionStatus.WAITING_EXTERNAL
                ),
                AuxiliarySemanticVerificationStatus.TURN_LIMIT_REACHED: (
                    AuxiliaryTerminalCompositionStatus.TURN_LIMIT_REACHED
                ),
                AuxiliarySemanticVerificationStatus.MODEL_INTERRUPTED: (
                    AuxiliaryTerminalCompositionStatus.MODEL_INTERRUPTED
                ),
            }.get(
                semantic_result.status,
                AuxiliaryTerminalCompositionStatus.FAILED_CLOSED,
            )
            return AuxiliaryTerminalCompositionResult(
                status=status,
                reason_code=semantic_result.reason_code,
                id_plan=id_plan,
                semantic_result=semantic_result,
            )
        assert semantic_result.settlement is not None
        if (
            semantic_result.settlement.host_disposition
            is not TaskGraphSemanticVerificationDisposition.PASS
        ):
            return AuxiliaryTerminalCompositionResult(
                status=AuxiliaryTerminalCompositionStatus.REVISION_REQUIRED,
                reason_code=_semantic_revision_reason_code(
                    semantic_result.settlement.host_disposition
                ),
                id_plan=id_plan,
                semantic_result=semantic_result,
            )
        terminal_seal = terminal_store.seal_auxiliary_terminal_proposal(
            command=terminal_store.SealAuxiliaryTerminalProposalCommand(
                apply_id=id_plan.terminal_seal_apply_id,
                finish_gate_receipt_id=id_plan.finish_gate_receipt_id,
                terminal_proposal_receipt_id=(
                    id_plan.terminal_proposal_receipt_id
                ),
                session_id=session_id,
                invocation_turn_id=turn_id,
                task_id=task_id,
                auxiliary_graph_id=details.auxiliary_graph_id,
                goal_id=details.goal_id,
                auxiliary_graph_revision=details.auxiliary_graph_revision,
                terminal_auxiliary_node_id=material.terminal_subject.node_id,
                terminal_node_revision=material.terminal_subject.node_revision,
                terminal_completion_id=material.terminal_completion_id,
                semantic_settlement_id=(
                    semantic_result.settlement.settlement_id
                ),
                semantic_prompt_payload_sha256=(
                    semantic_result.settlement.frozen_prompt_payload_sha256
                ),
                expected_base_task_graph_revision=(
                    details.base_task_graph_revision
                ),
                expected_task_state_version=frontier.task_state_version,
                expected_control_state_version=frontier.control_state_version,
                expected_goal_state_version=frontier.goal_state_version,
                expected_revision_state_version=frontier.revision_state_version,
                expected_budget_state_version=frontier.budget_state_version,
                expected_structure_sha256=frontier.structure_sha256,
                expected_budget_snapshot_sha256=(
                    frontier.budget_snapshot_sha256
                ),
            )
        )
        terminal_receipt_id = terminal_seal.terminal_proposal_receipt_id
    else:
        raise AuxiliaryTerminalCompositionError(
            "terminal composition was called outside a seal/commit frontier: "
            f"{decision.action.value}"
        )

    current_task = task_graph_store.get_insession_task_details(session_id, task_id)
    window = store.get_turn_execution_window(session_id)
    if (
        current_task is None
        or window is None
        or window.get("turn_id") != turn_id
    ):
        raise AuxiliaryTerminalCompositionError(
            "terminal commit lost its current Task/Turn window authority"
        )
    task_graph_revision_trigger = (
        None
        if details.base_task_graph_revision is None
        else task_delivery_store.get_active_task_graph_revision_trigger(
            session_id=session_id,
            task_id=task_id,
        )
    )
    task_graph_execution_replan_request = (
        None
        if details.base_task_graph_revision is None
        else work_run_store.get_active_task_graph_execution_replan_request(
            session_id=session_id,
            task_id=task_id,
        )
    )
    if (
        task_graph_revision_trigger is not None
        and task_graph_execution_replan_request is not None
    ):
        raise AuxiliaryTerminalCompositionError(
            "Task owns conflicting TaskGraph revision authorities"
        )
    if task_graph_revision_trigger is not None and (
        task_graph_revision_trigger.session_id != session_id
        or task_graph_revision_trigger.task_id != task_id
        or task_graph_revision_trigger.base_graph_revision
        != details.base_task_graph_revision
        or task_graph_revision_trigger.target_graph_revision
        != details.base_task_graph_revision + 1
    ):
        raise AuxiliaryTerminalCompositionError(
            "active TaskGraph revision trigger does not own this exact N -> N+1"
        )
    if task_graph_execution_replan_request is not None and (
        task_graph_execution_replan_request.session_id != session_id
        or task_graph_execution_replan_request.task_id != task_id
        or task_graph_execution_replan_request.base_graph_revision
        != details.base_task_graph_revision
        or task_graph_execution_replan_request.target_graph_revision
        != details.base_task_graph_revision + 1
    ):
        raise AuxiliaryTerminalCompositionError(
            "active TaskGraph execution request does not own this exact N -> N+1"
        )
    committed = terminal_store.commit_auxiliary_task_graph_proposal(
        command=terminal_store.CommitAuxiliaryTaskGraphProposalCommand(
            apply_id=id_plan.task_graph_commit_apply_id,
            session_id=session_id,
            source_turn_id=turn_id,
            task_id=task_id,
            terminal_proposal_receipt_id=terminal_receipt_id,
            expected_base_task_graph_revision=details.base_task_graph_revision,
            expected_task_state_version=current_task.task_state_version,
            expected_window_revision=int(window["state_version"]),
            base_node_alias_bindings=(
                ()
                if base_projection is None
                else base_projection.base_node_alias_bindings
            ),
            task_graph_revision_trigger_id=(
                None
                if task_graph_revision_trigger is None
                else task_graph_revision_trigger.trigger_id
            ),
            expected_task_graph_revision_trigger_sha256=(
                None
                if task_graph_revision_trigger is None
                else task_graph_revision_trigger.trigger_sha256
            ),
            task_graph_execution_replan_request_id=(
                None
                if task_graph_execution_replan_request is None
                else task_graph_execution_replan_request.request_id
            ),
            expected_task_graph_execution_replan_request_sha256=(
                None
                if task_graph_execution_replan_request is None
                else task_graph_execution_replan_request.request_sha256
            ),
        )
    )
    return AuxiliaryTerminalCompositionResult(
        status=AuxiliaryTerminalCompositionStatus.COMMITTED,
        reason_code=(
            "task_graph_revision_one_committed"
            if details.base_task_graph_revision is None
            else "task_graph_positive_revision_committed"
        ),
        id_plan=id_plan,
        semantic_result=semantic_result,
        terminal_seal=terminal_seal,
        task_graph_commit=committed,
    )


def _rebuild_planning_prompt_projections(
    *,
    session_id: str,
    task_id: str,
    details: auxiliary_graph_store.StoredAuxiliaryGraphDetails,
    allowed_managed_document_ids: tuple[str, ...] | None,
) -> tuple[
    PlanningAuthorityProjection,
    PlanningCapabilityCatalogProjection,
]:
    """在进程重启后重建提示安全的规划投影。

    一个成功的初始规划回执包含被 Architect 接纳的确切来源卡片和能力目录。 绑定那些不可变卡片可以保持重放身份，但终端规划必须在语义审查或提交前重新生成相同的当前挂载版本。直接的遗留/测试创建的图没有回执，并保留相同的全新挂载重建路径。
    """

    completion = planning_store.get_auxiliary_initial_planning_completion(
        session_id=session_id,
        insession_task_id=task_id,
    )
    if completion is not None:
        prompt = completion.binding.architect_request.prompt_payload
        anchors = {
            item.projection_alias: item
            for item in details.authority_snapshot.anchors
        }
        cards = {item.alias: item for item in prompt.authority.cards}
        if len(cards) != len(prompt.authority.cards) or set(anchors) != set(cards):
            raise AuxiliaryTerminalCompositionError(
                "initial-planning cards differ from current authority aliases"
            )
        if any(
            anchors[alias].authority_class is not card.authority_class
            or anchors[alias].projection_sha256 != card.projection_sha256
            for alias, card in cards.items()
        ):
            raise AuxiliaryTerminalCompositionError(
                "initial-planning cards differ from current authority bindings"
            )
        authority = PlanningAuthorityProjection.create(
            authority_snapshot_id=details.authority_snapshot_id,
            authority_snapshot_sha256=details.authority_snapshot_sha256,
            cards=tuple(sorted(cards.values(), key=lambda item: item.alias)),
        )
        scoped_document_ids = allowed_managed_document_ids
        if scoped_document_ids is None:
            scoped_document_ids = recover_task_scoped_managed_document_ids(
                session_id=session_id,
                task_id=task_id,
                authority_snapshot=details.authority_snapshot,
            )
        try:
            current_mounted = freeze_mounted_document_planning_authority(
                session_id=session_id,
                task_id=task_id,
                allowed_managed_document_ids=scoped_document_ids,
            )
            current_authority = build_mounted_document_authority_projection(
                authority_snapshot=details.authority_snapshot,
                task_creation_source=(
                    task_graph_store.get_insession_task_creation_source(
                        session_id=session_id,
                        insession_task_id=task_id,
                    )
                ),
                mounted_authority=current_mounted,
            )
        except (
            MountedDocumentPlanningAuthorityError,
            MountedVisualPlanningAuthorityError,
            TypeError,
            ValueError,
        ) as exc:
            raise AuxiliaryTerminalCompositionError(
                "terminal mounted-resource authority is stale or unavailable"
            ) from exc
        if current_authority != authority:
            raise AuxiliaryTerminalCompositionError(
                "terminal mounted-resource authority changed before commit"
            )
        return authority, _bind_semantic_capability_catalog(
            details=details,
            catalog=prompt.capabilities,
        )

    creation_source = task_graph_store.get_insession_task_creation_source(
        session_id=session_id,
        insession_task_id=task_id,
    )
    mounted = freeze_mounted_document_planning_authority(
        session_id=session_id,
        task_id=task_id,
        allowed_managed_document_ids=allowed_managed_document_ids,
    )
    authority = build_mounted_document_authority_projection(
        authority_snapshot=details.authority_snapshot,
        task_creation_source=creation_source,
        mounted_authority=mounted,
    )
    return authority, _bind_semantic_capability_catalog(
        details=details,
        catalog=build_auxiliary_planning_capability_catalog(
            mounted,
            workspace_runtime=build_session_workspace_readonly_runtime(
                session_id
            ),
        ),
    )


def _bind_semantic_capability_catalog(
    *,
    details: auxiliary_graph_store.StoredAuxiliaryGraphDetails,
    catalog: PlanningCapabilityCatalogProjection,
) -> PlanningCapabilityCatalogProjection:
    """赋予一个不可变目录一次修订的语义权威 ID。

    底层能力快照可能在图的修订之间保持字节相同。语义持久性故意将每个被接纳的目录投影绑定到一个确切的图修订，因此其持久身份在验证失败创建修订 ``N+1`` 时不得重用。
    """

    digest = _sha256_value(
        {
            "schema_version": "auxiliary-v2-semantic-capability-binding-v1",
            "session_id": details.session_id,
            "task_id": details.task_id,
            "auxiliary_graph_id": details.auxiliary_graph_id,
            "goal_id": details.goal_id,
            "auxiliary_graph_revision": details.auxiliary_graph_revision,
            "structure_sha256": details.structure_sha256,
            "source_capability_catalog_snapshot_id": (
                catalog.capability_catalog_snapshot_id
            ),
            "source_capability_catalog_snapshot_sha256": (
                catalog.capability_catalog_snapshot_sha256
            ),
            "source_projection_sha256": catalog.projection_sha256,
        }
    )
    return PlanningCapabilityCatalogProjection.create(
        capability_catalog_snapshot_id=(
            "auxv2-semantic-capabilities-v1-" + digest[:32]
        ),
        capability_catalog_snapshot_sha256=(
            catalog.capability_catalog_snapshot_sha256
        ),
        capabilities=catalog.capabilities,
    )


__all__ = [
    "AuxiliaryTerminalCandidateSemanticIdPlan",
    "AuxiliaryTerminalCandidateSemanticRouteRequired",
    "AuxiliaryTerminalCompositionError",
    "AuxiliaryTerminalCompositionResult",
    "AuxiliaryTerminalCompositionStatus",
    "AuxiliaryTerminalIdPlan",
    "derive_auxiliary_terminal_candidate_semantic_ids",
    "derive_auxiliary_terminal_ids",
    "run_auxiliary_terminal_candidate_semantic_gate",
    "run_auxiliary_terminal_composition",
]
