"""模型授权的条目，用于一个基于正数基的``TaskGraph N -> N+1`` 目标。

交付后的验证器或执行中的普通 TaskNode 拥有需要 TaskGraph 修订的决定权。该控制器仅接受相应的不可变修订权，创建一个非执行的 AuxiliaryGraph 目标启动，绑定到精确的 TaskGraph 基，并让持久化的 Architect 模型用完整的可执行 DAG 替换那个外壳。这里不合成固定的可执行图。

稳定的目标、模型调用和应用标识符使响应丢失恢复具有幂等性。只有当持久化的成功模型结算仍然描述了精确的当前持久化图时，才承认完成的恢复。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
import hashlib
import json

from personagraph.l2.auxiliary_graph import (
    AuxiliaryGraphRevisionProposal,
    AuxiliaryGraphRevisionReason,
    PlanningEpisodeBudgetProfile,
)
from personagraph.l2.task_graph import (
    InSessionTaskAcceptanceProposal,
    InSessionTaskStatus,
    TaskDeliveryValidationDisposition,
    TaskGraphRevisionTrigger,
)
from personagraph.session import store
from personagraph.session.l2_store import auxiliary_graph as auxiliary_graph_store
from personagraph.session.l2_store import planning as planning_store
from personagraph.session.l2_store import task_delivery as task_delivery_store
from personagraph.session.l2_store import task_graph as task_graph_store
from personagraph.session.l2_store import work_run as work_run_store
from personagraph.session.l2_store.auxiliary_graph import (
    AuxiliaryGraphRevisionCommitResult,
    StoredAuxiliaryGraphDetails,
)
from personagraph.l2.work_run.contracts import TaskGraphExecutionReplanRequest
from .architect import (
    AuxiliaryGraphArchitectAction,
    AuxiliaryGraphArchitectDecision,
    AuxiliaryGraphArchitectRequest,
    AuxiliaryGraphArchitectStructuredProvider,
    TaskGraphRevisionRouteAuthority,
    TaskGraphRevisionRouteKind,
    TaskGraphRevisionPlanningAuthority,
    request_auxiliary_graph_architect,
)
from .architect_adapter import (
    auxiliary_graph_revision_proposal_from_architect_decision,
    build_terminal_only_auxiliary_graph_bootstrap_proposal,
)
from .mounted_document_authority import (
    build_mounted_document_authority_projection,
    freeze_mounted_document_planning_authority,
)
from ..adapters.model_authority import (
    create_auxiliary_graph_architect_model_call_authority,
)
from .model_provider import (
    build_auxiliary_architect_structured_provider,
)
from .profiles import (
    build_auxiliary_architect_request,
    build_auxiliary_planning_capability_catalog,
    canonical_auxiliary_architect_state_guard,
)
from personagraph.runtime.model_calls.contracts import (
    RuntimeModelLogicalRequest,
    RuntimeModelPhysicalOutcome,
)
from personagraph.runtime.turn_deadline import TurnDeadline
from personagraph.runtime.model_calls.contracts import RuntimeModelLedgerStore
from personagraph.runtime.turn_events import TurnEvent
from personagraph.tools.workspace.session_read_source import (
    SessionWorkspaceReadonlyRuntime,
    build_session_workspace_readonly_runtime,
)
from .task_document_scope import (
    AuxiliaryTaskDocumentScopeError,
)


_BOOTSTRAP_TERMINAL_KEY = "positive_base_bootstrap"
_BOOTSTRAP_OUTPUT_CONTRACT = "task_graph_revision_proposal_v2"
_SHA256_ZERO = "0" * 64


class AuxiliaryPositivePlanningStatus(StrEnum):
    PLANNED = "planned"
    ALREADY_PLANNED = "already_planned"
    PLANNER_DECLINED_TRIGGER = "planner_declined_trigger"


class AuxiliaryPositivePlanningError(RuntimeError):
    """基于正数基的规划不能跨越其冻结的触发权。"""

    code = "auxiliary_v2_positive_planning_rejected"


@dataclass(frozen=True, slots=True)
class AuxiliaryPositivePlanningRequest:
    """在当前调用 Turn 期间从一个不可变的触发器生成的计划。

    ``trigger.created_turn_id`` 是整个任务验证结算的来源，该结算创建了触发器。``turn_id`` 是调用此规划尝试的 Turn，可能晚于那个源 Turn；在任何规划效果之前，其执行归属权会被检查。
    """

    session_id: str
    turn_id: str
    task_id: str
    trigger: TaskGraphRevisionPlanningAuthority
    desired_output: str = "完整、可执行、可验证的下一版 TaskGraph"
    budget_profile: PlanningEpisodeBudgetProfile | Mapping[str, object] | None = None
    deadline: TurnDeadline | None = None
    allowed_managed_document_ids: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        for name in ("session_id", "turn_id", "task_id"):
            _require_identifier(name, getattr(self, name))
        admitted = _admit_revision_authority(self.trigger)
        if (
            admitted.session_id != self.session_id
            or admitted.task_id != self.task_id
        ):
            raise ValueError(
                "positive planning request crossed trigger Session/Task authority"
            )
        if not isinstance(self.desired_output, str) or not self.desired_output.strip():
            raise ValueError("desired_output must not be empty")
        if self.allowed_managed_document_ids is not None:
            if not isinstance(self.allowed_managed_document_ids, tuple) or any(
                not isinstance(value, str)
                or not value
                or len(value) > 200
                for value in self.allowed_managed_document_ids
            ):
                raise ValueError(
                    "allowed_managed_document_ids must be bounded identities"
                )
            if len(self.allowed_managed_document_ids) != len(
                set(self.allowed_managed_document_ids)
            ):
                raise ValueError(
                    "allowed_managed_document_ids must be unique"
                )


@dataclass(frozen=True, slots=True)
class AuxiliaryPositivePlanningResult:
    status: AuxiliaryPositivePlanningStatus
    details: StoredAuxiliaryGraphDetails
    trigger: TaskGraphRevisionPlanningAuthority
    bootstrapped: bool
    bootstrap_commit: AuxiliaryGraphRevisionCommitResult | None = None
    architect_decision: AuxiliaryGraphArchitectDecision | None = None
    revision_commit: AuxiliaryGraphRevisionCommitResult | None = None
    model_call_id: str | None = None
    model_attempts: int = 0
    model_replayed: bool = False
    failure_reason: str | None = None


TurnEventEmitter = Callable[[TurnEvent], object]


def run_positive_base_auxiliary_planning(
    request: AuxiliaryPositivePlanningRequest,
    *,
    emit: TurnEventEmitter,
    provider: AuxiliaryGraphArchitectStructuredProvider | None = None,
    ledger_store: RuntimeModelLedgerStore,
    knowledge_cognition_history_enabled: bool = False,
) -> AuxiliaryPositivePlanningResult:
    """创建并规划一个精确的正基 AuxiliaryGraph 目标。"""

    if not isinstance(request, AuxiliaryPositivePlanningRequest):
        raise TypeError("request must be AuxiliaryPositivePlanningRequest")
    if not callable(emit):
        raise TypeError("emit must be callable")
    trigger = _admit_revision_authority(request.trigger)
    _require_invocation_turn_authority(request)
    task = _require_trigger_current_task(request, trigger)
    route_authority = _project_task_graph_revision_route_authority(
        request,
        trigger,
    )
    base_projection = _project_exact_base(request, trigger)
    creation_source = task_graph_store.get_insession_task_creation_source(
        session_id=request.session_id,
        insession_task_id=request.task_id,
    )
    mounted = freeze_mounted_document_planning_authority(
        session_id=request.session_id,
        task_id=request.task_id,
        allowed_managed_document_ids=request.allowed_managed_document_ids,
    )
    selected_profile = _coerce_budget_profile(request.budget_profile)
    identity = _positive_identity(trigger)
    goal_id = f"auxgoalv2_positive_{identity}"
    logical_call_id = f"auxv2positive_{identity}:model"
    details = _require_details(request.session_id, request.task_id)
    bootstrapped = False
    bootstrap_commit: AuxiliaryGraphRevisionCommitResult | None = None

    if details.goal_id != goal_id:
        if (
            details.goal_status != "committed"
            or details.revision_status != "committed"
        ):
            raise AuxiliaryPositivePlanningError(
                "a nonterminal AuxiliaryGraph goal blocks positive-base planning"
            )
        bootstrap = build_terminal_only_auxiliary_graph_bootstrap_proposal(
            terminal_node_key=_BOOTSTRAP_TERMINAL_KEY,
            title="确立 TaskGraph 修订的授权",
            objective=(
                "占位节点；等待受约束的 Architect 用完整的 TaskGraph "
                f"{trigger.base_graph_revision} -> {trigger.target_graph_revision} "
                "订正方案取代它。"
            ),
            source_anchor_ids=(creation_source.anchor_id,),
            acceptance_criteria=(
                InSessionTaskAcceptanceProposal(
                    acceptance_id="positive_architect_revision_committed",
                    criterion=(
                        "在任何 AuxiliaryGraph 节点被派发之前，必须由受约束的 "
                        "Architect 修订取代这个占位节点。"
                    ),
                    source_anchor_ids=(creation_source.anchor_id,),
                ),
            ),
            revision_reason=AuxiliaryGraphRevisionReason.VERIFICATION_FAILED,
        )
        bootstrap_commit = auxiliary_graph_store.commit_auxiliary_graph_revision(
            session_id=request.session_id,
            turn_id=request.turn_id,
            insession_task_id=request.task_id,
            expected_task_state_version=task.task_state_version,
            expected_base_task_graph_revision=trigger.base_graph_revision,
            expected_control_state_version=details.control_state_version,
            expected_current_auxiliary_graph_revision=(
                details.auxiliary_graph_revision
            ),
            apply_id=f"auxv2positive_{identity}:bootstrap",
            goal_objective=trigger.revision_objective,
            proposal=bootstrap,
            authority_context=mounted.authority_context,
            budget_profile=selected_profile.model_dump(mode="json"),
            auxiliary_graph_id=details.auxiliary_graph_id,
            goal_id=goal_id,
        )
        details = _require_details(request.session_id, request.task_id)
        bootstrapped = True

    _require_positive_goal_binding(
        details,
        trigger=trigger,
        goal_id=goal_id,
        task_status=task.status,
    )
    # 一些 Store 实现可能在附加新的规划目标时推进任务壳版本。重新冻结后绑定的权威状态；
    # 授权状态检查；
    # 活动触发器、基底修订和确定性目标绑定上述内容
    # 仍然防止与之无关的任务变异被接纳。
    task = _require_trigger_current_task(request, trigger)
    recovered = _recover_completed_planning(
        request=request,
        details=details,
        trigger=trigger,
        route_authority=route_authority,
        base_snapshot=base_projection.snapshot,
        logical_call_id=logical_call_id,
        task_status=task.status,
        ledger_store=ledger_store,
    )
    if recovered is not None:
        decision, attempts = recovered
        return AuxiliaryPositivePlanningResult(
            status=AuxiliaryPositivePlanningStatus.ALREADY_PLANNED,
            details=details,
            trigger=trigger,
            bootstrapped=False,
            architect_decision=decision,
            revision_commit=_replayed_revision_commit(details),
            model_call_id=logical_call_id,
            model_attempts=attempts,
            model_replayed=True,
        )

    _require_exact_positive_bootstrap(details)
    authority = build_mounted_document_authority_projection(
        authority_snapshot=details.authority_snapshot,
        task_creation_source=creation_source,
        mounted_authority=mounted,
    )
    workspace_runtime = build_session_workspace_readonly_runtime(
        request.session_id,
    )
    built_request = build_auxiliary_architect_request(
        details=details,
        authority=authority,
        capabilities=build_auxiliary_planning_capability_catalog(
            mounted,
            workspace_runtime=workspace_runtime,
            knowledge_cognition_history_enabled=(
                knowledge_cognition_history_enabled
            ),
        ),
        objective=details.goal_objective,
        desired_output=request.desired_output.strip(),
        task_graph_semantic_base=base_projection.snapshot,
        task_graph_revision_trigger=trigger,
        task_graph_revision_route=route_authority,
    )
    architect_request = AuxiliaryGraphArchitectRequest.create(
        architect_request_id=f"auxv2positive_{identity}:request",
        logical_call_id=logical_call_id,
        architect_profile_id=built_request.architect_profile_id,
        goal=built_request.goal,
        prompt_payload=built_request.prompt_payload,
    )
    def rederive_state_guard() -> str:
        try:
            _require_invocation_turn_authority(request)
            current_task = _require_trigger_current_task(request, trigger)
            current_base = _project_exact_base(request, trigger)
            current_route = _project_task_graph_revision_route_authority(
                request,
                trigger,
            )
            current_details = _require_details(request.session_id, request.task_id)
            current_mounted = freeze_mounted_document_planning_authority(
                session_id=request.session_id,
                task_id=request.task_id,
                allowed_managed_document_ids=(
                    request.allowed_managed_document_ids
                ),
            )
            current_workspace_runtime = build_session_workspace_readonly_runtime(
                request.session_id,
            )
            if (
                current_task != task
                or current_base != base_projection
                or current_route != route_authority
                or current_details != details
                or current_mounted != mounted
            ):
                return _SHA256_ZERO
            current_authority = build_mounted_document_authority_projection(
                authority_snapshot=current_details.authority_snapshot,
                task_creation_source=creation_source,
                mounted_authority=current_mounted,
            )
            current_built = build_auxiliary_architect_request(
                details=current_details,
                authority=current_authority,
                capabilities=build_auxiliary_planning_capability_catalog(
                    current_mounted,
                    workspace_runtime=current_workspace_runtime,
                    knowledge_cognition_history_enabled=(
                        knowledge_cognition_history_enabled
                    ),
                ),
                objective=current_details.goal_objective,
                desired_output=request.desired_output.strip(),
                task_graph_semantic_base=current_base.snapshot,
                task_graph_revision_trigger=trigger,
                task_graph_revision_route=current_route,
            )
            current_request = AuxiliaryGraphArchitectRequest.create(
                architect_request_id=architect_request.architect_request_id,
                logical_call_id=logical_call_id,
                architect_profile_id=current_built.architect_profile_id,
                goal=current_built.goal,
                prompt_payload=current_built.prompt_payload,
            )
            if current_request != architect_request:
                return _SHA256_ZERO
            return canonical_auxiliary_architect_state_guard(current_request)
        except Exception:
            return _SHA256_ZERO

    authority_values: dict[str, object] = {
        "request": architect_request,
        # 日志请求记录了不可变的原始发起的 Turn。  后续
        # Turn 只是一个用于重放/新物理调度的活执行租约。
        "invocation_turn_id": _architect_originating_turn_id(
            request=request,
            architect_request=architect_request,
            ledger_store=ledger_store,
        ),
        "rederive_state_guard_sha256": rederive_state_guard,
        "ledger_store": ledger_store,
    }
    durable_call = create_auxiliary_graph_architect_model_call_authority(
        **authority_values  # type: ignore[arg-type]
    )
    model_result = request_auxiliary_graph_architect(
        architect_request,
        invocation_turn_id=request.turn_id,
        provider=provider or build_auxiliary_architect_structured_provider(),
        emit=emit,
        deadline=request.deadline,
        durable_call=durable_call,
    )
    decision = model_result.value
    if decision.action is not AuxiliaryGraphArchitectAction.REVISE_REVISION:
        return AuxiliaryPositivePlanningResult(
            status=(
                AuxiliaryPositivePlanningStatus.PLANNER_DECLINED_TRIGGER
            ),
            details=details,
            trigger=trigger,
            bootstrapped=bootstrapped,
            bootstrap_commit=bootstrap_commit,
            architect_decision=decision,
            model_call_id=model_result.model_call_id,
            model_attempts=model_result.attempts,
            model_replayed=model_result.replayed,
            failure_reason=(
                "Architect did not replace the positive-base bootstrap"
            ),
        )

    durable_call.require_current_state()
    if freeze_mounted_document_planning_authority(
        session_id=request.session_id,
        task_id=request.task_id,
        allowed_managed_document_ids=request.allowed_managed_document_ids,
    ) != mounted:
        raise AuxiliaryPositivePlanningError(
            "mounted Document authority changed before positive revision commit"
        )
    if _project_exact_base(request, trigger) != base_projection:
        raise AuxiliaryPositivePlanningError(
            "TaskGraph semantic base changed before positive revision commit"
        )
    if (
        _project_task_graph_revision_route_authority(request, trigger)
        != route_authority
    ):
        raise AuxiliaryPositivePlanningError(
            "TaskGraph revision route changed before positive revision commit"
        )
    current_task = _require_trigger_current_task(request, trigger)
    current_details = _require_details(request.session_id, request.task_id)
    if current_task != task or current_details != details:
        raise AuxiliaryPositivePlanningError(
            "positive planning authority changed before revision commit"
        )
    revision_proposal = auxiliary_graph_revision_proposal_from_architect_decision(
        decision,
        expected_current_auxiliary_graph_revision=(
            details.auxiliary_graph_revision
        ),
        required_revision_reason=(
            AuxiliaryGraphRevisionReason.VERIFICATION_FAILED
        ),
    )
    revision_apply_id = f"auxv2positive_{identity}:architect"
    positive_completion = (
        planning_store.AuxiliaryPositivePlanningCompletionBinding.create(
            session_id=request.session_id,
            turn_id=request.turn_id,
            task_id=request.task_id,
            auxiliary_graph_id=details.auxiliary_graph_id,
            goal_id=details.goal_id,
            bootstrap_auxiliary_graph_revision=(
                details.auxiliary_graph_revision
            ),
            bootstrap_structure_sha256=details.structure_sha256,
            revision_apply_id=revision_apply_id,
            revision_authority=trigger,
            architect_request=architect_request,
            architect_decision=decision,
            runtime_logical_request_binding_sha256=(
                durable_call.logical_request.binding_sha256
            ),
        )
    )
    revision_commit = auxiliary_graph_store.commit_auxiliary_graph_revision(
        session_id=request.session_id,
        turn_id=request.turn_id,
        insession_task_id=request.task_id,
        expected_task_state_version=task.task_state_version,
        expected_base_task_graph_revision=trigger.base_graph_revision,
        expected_control_state_version=details.control_state_version,
        expected_current_auxiliary_graph_revision=(
            details.auxiliary_graph_revision
        ),
        apply_id=revision_apply_id,
        goal_objective=details.goal_objective,
        proposal=revision_proposal,
        authority_context=mounted.authority_context,
        budget_profile=details.budget.base_profile.model_dump(mode="json"),
        auxiliary_graph_id=details.auxiliary_graph_id,
        goal_id=details.goal_id,
        positive_planning_completion=positive_completion,
    )
    committed = _require_details(request.session_id, request.task_id)
    if (
        committed.goal_id != goal_id
        or committed.base_task_graph_revision != trigger.base_graph_revision
        or committed.target_task_graph_revision != trigger.target_graph_revision
        or committed.auxiliary_graph_revision
        != details.auxiliary_graph_revision + 1
        or committed.structure_sha256 != revision_commit.structure_sha256
    ):
        raise AuxiliaryPositivePlanningError(
            "Architect revision did not become the exact positive current graph"
        )
    return AuxiliaryPositivePlanningResult(
        status=AuxiliaryPositivePlanningStatus.PLANNED,
        details=committed,
        trigger=trigger,
        bootstrapped=bootstrapped,
        bootstrap_commit=bootstrap_commit,
        architect_decision=decision,
        revision_commit=revision_commit,
        model_call_id=model_result.model_call_id,
        model_attempts=model_result.attempts,
        model_replayed=model_result.replayed,
    )


def _recover_completed_planning(
    *,
    request: AuxiliaryPositivePlanningRequest,
    details: StoredAuxiliaryGraphDetails,
    trigger: TaskGraphRevisionPlanningAuthority,
    route_authority: TaskGraphRevisionRouteAuthority | None,
    base_snapshot: object,
    logical_call_id: str,
    task_status: InSessionTaskStatus,
    ledger_store: RuntimeModelLedgerStore,
) -> tuple[AuxiliaryGraphArchitectDecision, int] | None:
    if _is_exact_positive_bootstrap(details):
        return None
    if (
        details.parent_auxiliary_graph_revision is None
        or details.auxiliary_graph_revision
        != details.parent_auxiliary_graph_revision + 1
        or details.reason != AuxiliaryGraphRevisionReason.VERIFICATION_FAILED.value
        or not _is_recoverable_positive_execution_state(
            details,
            task_status=task_status,
        )
    ):
        raise AuxiliaryPositivePlanningError(
            "positive goal is neither its bootstrap nor a recoverable plan"
        )
    logical = ledger_store.get_runtime_model_logical_call(
        session_id=request.session_id,
        logical_call_id=logical_call_id,
    )
    if logical is None:
        raise AuxiliaryPositivePlanningError(
            "positive current graph has no durable Architect model call"
        )
    stored = logical.request
    try:
        architect_request = AuxiliaryGraphArchitectRequest.model_validate_json(
            stored.request_json
        )
    except Exception as exc:
        raise AuxiliaryPositivePlanningError(
            "positive Architect request authority is corrupt"
        ) from exc
    current = architect_request.prompt_payload.current_revision
    if (
        stored.task_id != request.task_id
        or stored.auxiliary_graph_id != details.auxiliary_graph_id
        or stored.goal_id != details.goal_id
        or stored.call_kind != "auxiliary_graph_architect"
        or stored.request_contract != "auxiliary-graph-architect-request-v1"
        or stored.typed_result_contract != "auxiliary-graph-revision-proposal-v2"
        or stored.state_guard_sha256 != architect_request.binding_sha256
        or architect_request.goal.session_id != details.goal.session_id
        or architect_request.goal.task_id != details.goal.task_id
        or architect_request.goal.auxiliary_graph_id
        != details.goal.auxiliary_graph_id
        or architect_request.goal.goal_id != details.goal.goal_id
        or architect_request.goal.base_task_graph_revision
        != details.goal.base_task_graph_revision
        or architect_request.goal.target_task_graph_revision
        != details.goal.target_task_graph_revision
        or architect_request.goal.status.value != "active"
        or details.goal.state_version < architect_request.goal.state_version + 1
        or architect_request.prompt_payload.goal.objective
        != details.goal_objective
        or architect_request.prompt_payload.task_graph_revision_trigger != trigger
        or architect_request.prompt_payload.task_graph_revision_route
        != route_authority
        or architect_request.prompt_payload.task_graph_semantic_base != base_snapshot
        or current is None
        or current.auxiliary_graph_revision
        != details.parent_auxiliary_graph_revision
        or current.base_task_graph_revision != trigger.base_graph_revision
    ):
        raise AuxiliaryPositivePlanningError(
            "positive Architect request crossed immutable planning authority"
        )
    if not logical.physical_attempts:
        raise AuxiliaryPositivePlanningError(
            "positive Architect call has no physical attempt"
        )
    final = logical.physical_attempts[-1]
    settlement = final.settlement
    if (
        settlement is None
        or settlement.outcome is not RuntimeModelPhysicalOutcome.SUCCEEDED
        or settlement.typed_result is None
        or settlement.typed_result.result_contract
        != "auxiliary-graph-revision-proposal-v2"
    ):
        raise AuxiliaryPositivePlanningError(
            "positive Architect call lacks an exact succeeded settlement"
        )
    _require_authenticated_logical_execution_history(
        request=request,
        logical=logical,
    )
    try:
        proposal = AuxiliaryGraphRevisionProposal.model_validate(
            settlement.typed_result.parsed()
        )
        decision = AuxiliaryGraphArchitectDecision.create(
            request=architect_request,
            proposal=proposal,
        )
        persisted = auxiliary_graph_revision_proposal_from_architect_decision(
            decision,
            expected_current_auxiliary_graph_revision=(
                details.parent_auxiliary_graph_revision
            ),
            required_revision_reason=(
                AuxiliaryGraphRevisionReason.VERIFICATION_FAILED
            ),
        )
    except Exception as exc:
        raise AuxiliaryPositivePlanningError(
            "positive Architect settlement is not a committable revision"
        ) from exc
    if not _details_match_persistence_proposal(details, persisted):
        raise AuxiliaryPositivePlanningError(
            "positive current graph differs from its settled Architect proposal"
        )
    return decision, len(logical.physical_attempts)


def _details_match_persistence_proposal(
    details: StoredAuxiliaryGraphDetails,
    proposal: object,
) -> bool:
    if not isinstance(proposal, auxiliary_graph_store.AuxiliaryGraphRevisionProposalRecord):
        return False
    if (
        details.reason != proposal.revision_reason.value
        or len(details.nodes) != len(proposal.nodes)
        or len(details.edges) != len(proposal.edges)
    ):
        return False
    key_by_id = {
        node.auxiliary_node_id: node.local_node_key for node in details.nodes
    }
    terminal_key = key_by_id.get(details.terminal_auxiliary_node_id)
    if terminal_key != proposal.terminal_node_key:
        return False
    for stored_node, proposed_node in zip(
        details.nodes,
        proposal.nodes,
        strict=True,
    ):
        if (
            stored_node.local_node_key != proposed_node.local_node_key
            or stored_node.node_kind != proposed_node.node_kind.value
            or stored_node.executor_kind != proposed_node.executor_kind.value
            or stored_node.title != proposed_node.title
            or stored_node.objective != proposed_node.objective
            or stored_node.source_anchor_ids != proposed_node.source_anchor_ids
            or stored_node.acceptance_criteria
            != proposed_node.acceptance_criteria
            or stored_node.output_contract != proposed_node.output_contract
            or stored_node.capability_profile_id
            != proposed_node.capability_profile_id
            or stored_node.input_resource_aliases
            != proposed_node.input_resource_aliases
            or stored_node.required != proposed_node.required
            or (stored_node.origin_node_ref is None)
            != (proposed_node.origin_node_alias is None)
        ):
            return False
        if (
            proposed_node.origin_node_alias is not None
            and proposed_node.origin_node_alias != _BOOTSTRAP_TERMINAL_KEY
        ):
            return False
    stored_edges = tuple(
        (
            key_by_id.get(edge.dependency_auxiliary_node_id),
            key_by_id.get(edge.consumer_auxiliary_node_id),
            edge.required,
        )
        for edge in details.edges
    )
    proposed_edges = tuple(
        (
            edge.dependency_node_key,
            edge.consumer_node_key,
            edge.required,
        )
        for edge in proposal.edges
    )
    return stored_edges == proposed_edges


def _require_trigger_current_task(
    request: AuxiliaryPositivePlanningRequest,
    trigger: TaskGraphRevisionPlanningAuthority,
):
    try:
        if isinstance(trigger, TaskGraphRevisionTrigger):
            active_trigger = task_delivery_store.get_active_task_graph_revision_trigger(
                session_id=request.session_id,
                task_id=request.task_id,
            )
        else:
            active_trigger = (
                work_run_store.get_active_task_graph_execution_replan_request(
                    session_id=request.session_id,
                    task_id=request.task_id,
                )
            )
    except (
        task_delivery_store.TaskDeliveryValidationPersistenceError,
        work_run_store.WorkExecutionPersistenceError,
    ) as exc:
        raise AuxiliaryPositivePlanningError(
            "active TaskGraph revision trigger failed authority validation"
        ) from exc
    if active_trigger != trigger:
        raise AuxiliaryPositivePlanningError(
            "revision trigger is not the exact active Store authority"
        )
    task = task_graph_store.get_insession_task_details(
        request.session_id,
        request.task_id,
    )
    if task is None:
        raise AuxiliaryPositivePlanningError("unknown Task in this Session")
    # ``get_active_task_graph_revision_trigger`` 只接受``AWAITING_USER``
    # 当 Store 已验证唯一待处理的 用户门控游标时
    # 由这个精确的触发目标拥有。保持那个狭窄的跨 Turn 租约。
    # 这里；任意等待中的任务仍然会在 Store 加载器上方失败。
    if (
        task.status
        not in {
            InSessionTaskStatus.ACTIVE,
            InSessionTaskStatus.AWAITING_USER,
        }
        or task.current_graph_revision != trigger.base_graph_revision
        or task.task_state_version < _authority_task_state_version(trigger)
    ):
        raise AuxiliaryPositivePlanningError(
            "Task differs from the verifier-reopened trigger authority"
        )
    if task.task_state_version != _authority_task_state_version(trigger):
        details = auxiliary_graph_store.get_auxiliary_graph_for_task(
            session_id=request.session_id,
            insession_task_id=request.task_id,
        )
        expected_goal_id = (
            f"auxgoalv2_positive_{_positive_identity(trigger)}"
        )
        if (
            details is None
            or details.goal_id != expected_goal_id
            or details.base_task_graph_revision != trigger.base_graph_revision
            or details.target_task_graph_revision
            != trigger.target_graph_revision
            or not _is_recoverable_positive_execution_state(
                details,
                task_status=task.status,
            )
        ):
            raise AuxiliaryPositivePlanningError(
                "Task state advanced outside the exact positive planning goal"
            )
    return task




def _require_invocation_turn_authority(
    request: AuxiliaryPositivePlanningRequest,
) -> None:
    """要求当前正在运行的 Turn 及其精确的持久化任务通道。

    触发器保留的验证 Turn 证明了为何需要修订，却不授权后续模型调用。
    授权来自活动的 Runtime Turn，以及为该 Turn 持久化的来源绑定通道清单。
    此预检发生在启动或 Provider 效果之前。Store 对活动触发器和 TaskGraph
    提交所做的 CAS 检查，在各自的变更边界上仍是权威依据。
    """

    try:
        execution = store.inspect_turn_execution(request.session_id)
        turn = execution.get("turn")
        window = execution.get("window")
        if not isinstance(turn, Mapping) or not isinstance(window, Mapping):
            raise AuxiliaryPositivePlanningError(
                "positive planning invocation has no active Runtime Turn"
            )
        if (
            turn.get("turn_id") != request.turn_id
            or turn.get("session_id") != request.session_id
            or turn.get("status") != "running"
            or window.get("turn_id") != request.turn_id
            or window.get("window_state") != "active"
        ):
            raise AuxiliaryPositivePlanningError(
                "positive planning requires its exact running invocation Turn"
            )
        manifest = task_graph_store.get_insession_task_execution_lane_manifest(
            session_id=request.session_id,
            turn_id=request.turn_id,
        )
    except AuxiliaryPositivePlanningError:
        raise
    except Exception as exc:
        raise AuxiliaryPositivePlanningError(
            "positive planning invocation authority is unavailable"
        ) from exc

    lanes = tuple(
        lane
        for lane in manifest.lanes
        if lane.insession_task_id == request.task_id
    )
    if len(lanes) != 1 or lanes[0].execution_requested is not True:
        raise AuxiliaryPositivePlanningError(
            "positive planning invocation Turn did not request this Task lane"
        )


def _architect_originating_turn_id(
    *,
    request: AuxiliaryPositivePlanningRequest,
    architect_request: AuxiliaryGraphArchitectRequest,
    ledger_store: RuntimeModelLedgerStore,
) -> str:
    """保持预留的逻辑请求在 Turn 租约移交过程中不变。"""

    logical = ledger_store.get_runtime_model_logical_call(
        session_id=request.session_id,
        logical_call_id=architect_request.logical_call_id,
    )
    if logical is None:
        return request.turn_id
    stored = getattr(logical, "request", None)
    if not isinstance(stored, RuntimeModelLogicalRequest):
        raise AuxiliaryPositivePlanningError(
            "positive Architect logical request has the wrong durable contract"
        )
    try:
        frozen_request = AuxiliaryGraphArchitectRequest.model_validate_json(
            stored.request_json
        )
    except Exception as exc:
        raise AuxiliaryPositivePlanningError(
            "positive Architect logical request payload is corrupt"
        ) from exc
    if (
        frozen_request != architect_request
        or stored.logical_call_id != architect_request.logical_call_id
        or stored.session_id != request.session_id
        or stored.task_id != request.task_id
        or stored.auxiliary_graph_id != architect_request.goal.auxiliary_graph_id
        or stored.goal_id != architect_request.goal.goal_id
        or stored.execution_subject_id is not None
        or stored.call_kind != "auxiliary_graph_architect"
        or stored.purpose != "runtime_auxiliary_graph_architect_v2"
        or stored.request_contract != "auxiliary-graph-architect-request-v1"
        or stored.typed_result_contract
        != "auxiliary-graph-revision-proposal-v2"
        or stored.state_guard_sha256 != architect_request.binding_sha256
    ):
        raise AuxiliaryPositivePlanningError(
            "positive Architect logical request crossed immutable planning authority"
        )
    _require_authenticated_logical_execution_history(
        request=request,
        logical=logical,
    )
    return stored.invocation_turn_id


def _require_authenticated_logical_execution_history(
    *,
    request: AuxiliaryPositivePlanningRequest,
    logical: object,
) -> None:
    stored = getattr(logical, "request", None)
    attempts = getattr(logical, "physical_attempts", None)
    if not isinstance(stored, RuntimeModelLogicalRequest) or not isinstance(
        attempts,
        tuple,
    ):
        raise AuxiliaryPositivePlanningError(
            "positive Architect ledger history has the wrong contract"
        )
    turn_ids = [stored.invocation_turn_id]
    for physical in attempts:
        physical_request = getattr(physical, "request", None)
        if physical_request is None:
            raise AuxiliaryPositivePlanningError(
                "positive Architect physical history is corrupt"
            )
        turn_ids.append(str(physical_request.started_turn_id))
        settlement = getattr(physical, "settlement", None)
        if settlement is not None:
            turn_ids.append(str(settlement.settled_turn_id))
    for turn_id in dict.fromkeys(turn_ids):
        _require_historical_task_execution_lease(
            session_id=request.session_id,
            task_id=request.task_id,
            turn_id=turn_id,
        )


def _require_historical_task_execution_lease(
    *,
    session_id: str,
    task_id: str,
    turn_id: str,
) -> None:
    """Turn：在无需要求其持续运行的情况下对旧模型调用进行身份验证。"""

    try:
        manifest = task_graph_store.get_insession_task_execution_lane_manifest(
            session_id=session_id,
            turn_id=turn_id,
        )
    except Exception as exc:
        raise AuxiliaryPositivePlanningError(
            "positive Architect history references an unauthenticated Turn"
        ) from exc
    lanes = tuple(
        lane
        for lane in manifest.lanes
        if lane.insession_task_id == task_id
    )
    if len(lanes) != 1 or lanes[0].execution_requested is not True:
        raise AuxiliaryPositivePlanningError(
            "positive Architect history lacks an executable Task lane"
        )


def _project_exact_base(
    request: AuxiliaryPositivePlanningRequest,
    trigger: TaskGraphRevisionPlanningAuthority,
):
    try:
        return task_graph_store.project_task_graph_semantic_base(
            session_id=request.session_id,
            task_id=request.task_id,
            graph_revision=trigger.base_graph_revision,
        )
    except Exception as exc:
        raise AuxiliaryPositivePlanningError(
            "TaskGraph semantic base projection failed closed"
        ) from exc


def _project_task_graph_revision_route_authority(
    request: AuxiliaryPositivePlanningRequest,
    trigger: TaskGraphRevisionPlanningAuthority,
) -> TaskGraphRevisionRouteAuthority | None:
    """在存在时重新加载类型化的 原因 ``trigger``。

    遗留的 REVISE 触发器没有候选结算，因此没有扩展投影。一个候选触发器必须完全匹配经过身份验证的 Store 装载器；其目标或差距说明不会用于推断路径。
    """

    if isinstance(trigger, TaskGraphExecutionReplanRequest):
        return None
    try:
        stored = task_delivery_store.get_task_delivery_candidate_settlement(
            session_id=request.session_id,
            task_id=request.task_id,
            graph_revision=trigger.base_graph_revision,
        )
    except Exception as exc:
        raise AuxiliaryPositivePlanningError(
            "TaskGraph revision candidate authority failed validation"
        ) from exc
    if stored is None:
        return None
    if stored.trigger != trigger:
        raise AuxiliaryPositivePlanningError(
            "TaskGraph revision trigger crossed its candidate settlement"
        )
    result = stored.intent.result
    if result.disposition is TaskDeliveryValidationDisposition.REPLAN_TASK_GRAPH:
        route_kind = TaskGraphRevisionRouteKind.TASK_GRAPH_DESIGN_REPAIR
        requires_user_gate = False
    elif result.disposition is TaskDeliveryValidationDisposition.BLOCKED:
        route_kind = (
            TaskGraphRevisionRouteKind.MISSING_INFORMATION_CLARIFICATION
        )
        requires_user_gate = True
    else:
        raise AuxiliaryPositivePlanningError(
            "candidate trigger has a non-revision disposition"
        )
    try:
        return TaskGraphRevisionRouteAuthority.create(
            trigger_id=trigger.trigger_id,
            trigger_sha256=trigger.trigger_sha256,
            settlement_id=stored.settlement.settlement_id,
            settlement_sha256=stored.settlement.settlement_sha256,
            candidate_authority_sha256=(
                stored.settlement.candidate_authority_sha256
            ),
            verification_request_id=(
                stored.settlement.verification_request_id
            ),
            request_binding_sha256=(
                stored.settlement.request_binding_sha256
            ),
            verification_result_id=(
                stored.settlement.verification_result_id
            ),
            result_sha256=stored.settlement.result_sha256,
            disposition=result.disposition,
            route_kind=route_kind,
            findings=result.findings,
            summary=result.summary,
            blocking_questions=result.blocking_questions,
            requires_user_gate=requires_user_gate,
        )
    except Exception as exc:
        raise AuxiliaryPositivePlanningError(
            "candidate cannot become a safe typed revision route"
        ) from exc


def _require_positive_goal_binding(
    details: StoredAuxiliaryGraphDetails,
    *,
    trigger: TaskGraphRevisionPlanningAuthority,
    goal_id: str,
    task_status: InSessionTaskStatus,
) -> None:
    if (
        details.goal_id != goal_id
        or details.goal_objective != trigger.revision_objective
        or details.base_task_graph_revision != trigger.base_graph_revision
        or details.target_task_graph_revision != trigger.target_graph_revision
        or not _is_recoverable_positive_execution_state(
            details,
            task_status=task_status,
        )
    ):
        raise AuxiliaryPositivePlanningError(
            "positive AuxiliaryGraph goal differs from its revision trigger"
        )


def _is_recoverable_positive_execution_state(
    details: StoredAuxiliaryGraphDetails,
    *,
    task_status: InSessionTaskStatus,
) -> bool:
    """TaskGraph：仅允许由当前触发器认证的执行后代通过。

    Architect 的权威状态冻结图结构，而不是可变执行光标的结构。普通执行保持 ``active``。崩溃也可能使精确的正终端在 ``proposal_ready`` 或 ``gapped_ready`` 状态下被密封，这些状态必须在没有另一个模型调用的情况下恢复。唯一的跨 Turn 停顿是当前触发器拥有的精确 USER_GATE，它在到达此辅助函数之前已被 ``get_active_task_graph_revision_trigger`` 认证。
    """

    return (
        task_status is InSessionTaskStatus.ACTIVE
        and (
            (details.goal_status, details.revision_status)
            in {
                ("active", "active"),
                ("proposal_ready", "proposal_ready"),
                ("gapped_ready", "gapped_ready"),
            }
        )
    ) or (
        task_status is InSessionTaskStatus.AWAITING_USER
        and details.goal_status == "waiting_user"
        and details.revision_status == "waiting_user"
    )


def _is_exact_positive_bootstrap(details: StoredAuxiliaryGraphDetails) -> bool:
    return (
        len(details.nodes) == 1
        and not details.edges
        and details.nodes[0].local_node_key == _BOOTSTRAP_TERMINAL_KEY
        and details.nodes[0].auxiliary_node_id
        == details.terminal_auxiliary_node_id
        and details.nodes[0].executor_kind == "terminal_planner"
        and details.nodes[0].output_contract == _BOOTSTRAP_OUTPUT_CONTRACT
        and details.nodes[0].capability_profile_id is None
        and details.nodes[0].status == "proposed"
        and details.reason
        == AuxiliaryGraphRevisionReason.VERIFICATION_FAILED.value
        and details.goal_status == "active"
        and details.revision_status == "active"
    )


def _require_exact_positive_bootstrap(
    details: StoredAuxiliaryGraphDetails,
) -> None:
    if not _is_exact_positive_bootstrap(details):
        raise AuxiliaryPositivePlanningError(
            "positive goal current revision is not its non-executable bootstrap"
        )


def _replayed_revision_commit(
    details: StoredAuxiliaryGraphDetails,
) -> AuxiliaryGraphRevisionCommitResult:
    return AuxiliaryGraphRevisionCommitResult(
        status="replayed",
        auxiliary_graph_id=details.auxiliary_graph_id,
        goal_id=details.goal_id,
        committed_auxiliary_graph_revision=details.auxiliary_graph_revision,
        control_state_version=details.control_state_version,
        goal_state_version=details.goal_state_version,
        revision_state_version=details.revision_state_version,
        budget_state_version=details.budget_state_version,
        authority_snapshot_id=details.authority_snapshot_id,
        authority_snapshot_sha256=details.authority_snapshot_sha256,
        structure_sha256=details.structure_sha256,
    )


def _require_details(
    session_id: str,
    task_id: str,
) -> StoredAuxiliaryGraphDetails:
    details = auxiliary_graph_store.get_auxiliary_graph_for_task(
        session_id=session_id,
        insession_task_id=task_id,
    )
    if details is None:
        raise AuxiliaryPositivePlanningError(
            "positive planning requires an existing AuxiliaryGraph container"
        )
    return details


def _coerce_budget_profile(
    value: PlanningEpisodeBudgetProfile | Mapping[str, object] | None,
) -> PlanningEpisodeBudgetProfile:
    if value is None:
        return PlanningEpisodeBudgetProfile()
    if isinstance(value, PlanningEpisodeBudgetProfile):
        return PlanningEpisodeBudgetProfile.model_validate_json(
            value.model_dump_json()
        )
    try:
        return PlanningEpisodeBudgetProfile.model_validate(dict(value))
    except (TypeError, ValueError) as exc:
        raise AuxiliaryPositivePlanningError(
            "positive planning budget profile is invalid"
        ) from exc


def _positive_identity(trigger: TaskGraphRevisionPlanningAuthority) -> str:
    authority_id, authority_sha256 = _authority_identity(trigger)
    return hashlib.sha256(
        json.dumps(
            {
                "schema_version": "auxiliary-v2-positive-planning-identity-v1",
                "trigger_id": authority_id,
                "trigger_sha256": authority_sha256,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()[:32]


def _admit_revision_authority(
    value: object,
) -> TaskGraphRevisionPlanningAuthority:
    """通过其类型验证器复制一个支持的不可变权威。"""

    if isinstance(value, TaskGraphRevisionTrigger):
        return TaskGraphRevisionTrigger.model_validate_json(
            value.model_dump_json()
        )
    if isinstance(value, TaskGraphExecutionReplanRequest):
        return TaskGraphExecutionReplanRequest.model_validate_json(
            value.model_dump_json()
        )
    raise TypeError(
        "trigger must be TaskGraphRevisionTrigger or "
        'TaskGraphExecutionReplanRequest'
    )


def _authority_task_state_version(
    authority: TaskGraphRevisionPlanningAuthority,
) -> int:
    if isinstance(authority, TaskGraphRevisionTrigger):
        return authority.reopened_task_state_version
    return authority.task_state_version


def _authority_identity(
    authority: TaskGraphRevisionPlanningAuthority,
) -> tuple[str, str]:
    if isinstance(authority, TaskGraphRevisionTrigger):
        return authority.trigger_id, authority.trigger_sha256
    return authority.request_id, authority.request_sha256


def _require_identifier(name: str, value: str) -> None:
    if not isinstance(value, str) or not value or len(value) > 200:
        raise ValueError(f"{name} must be 1..200 characters")


__all__ = [
    "AuxiliaryPositivePlanningError",
    "AuxiliaryPositivePlanningRequest",
    "AuxiliaryPositivePlanningResult",
    "AuxiliaryPositivePlanningStatus",
    "run_positive_base_auxiliary_planning",
]
