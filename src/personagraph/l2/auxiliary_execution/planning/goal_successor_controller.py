"""目标被取代后，基于已认证回执规划新目标。

取代陈旧的规划阶段后，容器指针会刻意停留在其已终止的旧修订上。此控制器是从该密封回执通往替代目标的唯一生产桥梁：它在下一个全局 AuxiliaryGraph 修订中提交一个不可执行的 Host 外壳，再由新的持久化 Architect 逻辑调用将该外壳替换为模型生成的 DAG。
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
)
from personagraph.session import store
from personagraph.session.l2_store import auxiliary_graph as auxiliary_graph_store
from personagraph.session.l2_store import planning as planning_store
from personagraph.session.l2_store import task_graph as task_graph_store
from personagraph.session.l2_store.auxiliary_graph import (
    AuxiliaryGraphRevisionCommitResult,
    StoredAuxiliaryGraphDetails,
)
from personagraph.l2.auxiliary_execution.planning.architect import (
    AuxiliaryGraphArchitectAction,
    AuxiliaryGraphArchitectDecision,
    AuxiliaryGraphArchitectRequest,
    AuxiliaryGraphArchitectStructuredProvider,
    request_auxiliary_graph_architect,
)
from personagraph.l2.auxiliary_execution.planning.architect_adapter import (
    auxiliary_graph_revision_proposal_from_architect_decision,
    build_terminal_only_auxiliary_graph_bootstrap_proposal,
)
from .mounted_document_authority import (
    build_mounted_document_authority_projection,
    freeze_mounted_document_planning_authority,
)
from personagraph.l2.auxiliary_execution.adapters.model_authority import (
    create_auxiliary_graph_architect_model_call_authority,
)
from personagraph.l2.auxiliary_execution.planning.model_provider import (
    build_auxiliary_architect_structured_provider,
)
from personagraph.l2.auxiliary_execution.planning.profiles import (
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
from personagraph.l2.auxiliary_execution.planning.task_document_scope import (
    AuxiliaryTaskDocumentScopeError,
)


_BOOTSTRAP_TERMINAL_KEY = "goal_successor_bootstrap"
_BOOTSTRAP_OUTPUT_CONTRACT = "task_graph_revision_proposal_v2"
_SHA256_ZERO = "0" * 64


class AuxiliaryGoalSuccessorPlanningStatus(StrEnum):
    PLANNED = "planned"
    ALREADY_PLANNED = "already_planned"
    PLANNER_DECLINED_SUCCESSOR = "planner_declined_successor"


class AuxiliaryGoalSuccessorPlanningError(RuntimeError):
    """取代回执不能安全地授权其继任目标。"""

    code = "auxiliary_v2_goal_successor_planning_rejected"


@dataclass(frozen=True, slots=True)
class AuxiliaryGoalSuccessorIds:
    identity: str
    goal_id: str
    bootstrap_apply_id: str
    architect_request_id: str
    logical_call_id: str
    architect_apply_id: str


@dataclass(frozen=True, slots=True)
class AuxiliaryGoalSuccessorPlanningRequest:
    """在当前 Turn 执行租约下计划一个回执绑定的继任目标。

    ``supersede_receipt.invocation_turn_id`` 是取代的不变证明来源。``turn_id`` 是当前运行的延续 Turn，可能会不同；它从不重写目标变更的源权威状态。
    """

    session_id: str
    turn_id: str
    task_id: str
    supersede_receipt: planning_store.PlanningGoalSupersedeReceipt
    desired_output: str = "完整、可执行、可验证的 TaskGraph"
    budget_profile: PlanningEpisodeBudgetProfile | Mapping[str, object] | None = None
    deadline: TurnDeadline | None = None
    allowed_managed_document_ids: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        for name in ("session_id", "turn_id", "task_id"):
            _require_identifier(name, getattr(self, name))
        if not isinstance(
            self.supersede_receipt,
            planning_store.PlanningGoalSupersedeReceipt,
        ):
            raise TypeError(
                "supersede_receipt must be PlanningGoalSupersedeReceipt"
            )
        if (
            self.supersede_receipt.session_id != self.session_id
            or self.supersede_receipt.task_id != self.task_id
        ):
            raise ValueError(
                "successor request crossed supersede Session/Task authority"
            )
        if not isinstance(self.desired_output, str) or not self.desired_output.strip():
            raise ValueError("desired_output must not be empty")
        if self.allowed_managed_document_ids is not None:
            values = self.allowed_managed_document_ids
            if (
                not isinstance(values, tuple)
                or len(values) != len(set(values))
                or any(
                    not isinstance(value, str)
                    or not value
                    or len(value) > 200
                    for value in values
                )
            ):
                raise ValueError(
                    "allowed_managed_document_ids must be unique bounded identities"
                )


@dataclass(frozen=True, slots=True)
class AuxiliaryGoalSuccessorPlanningResult:
    status: AuxiliaryGoalSuccessorPlanningStatus
    details: StoredAuxiliaryGraphDetails
    supersede_receipt: planning_store.PlanningGoalSupersedeReceipt
    ids: AuxiliaryGoalSuccessorIds
    bootstrapped: bool
    bootstrap_commit: AuxiliaryGraphRevisionCommitResult | None = None
    architect_decision: AuxiliaryGraphArchitectDecision | None = None
    revision_commit: AuxiliaryGraphRevisionCommitResult | None = None
    model_call_id: str | None = None
    model_attempts: int = 0
    model_replayed: bool = False
    failure_reason: str | None = None


TurnEventEmitter = Callable[[TurnEvent], object]


def derive_auxiliary_goal_successor_ids(
    receipt: planning_store.PlanningGoalSupersedeReceipt,
) -> AuxiliaryGoalSuccessorIds:
    """仅从密封回执中推导出所有后继身份。"""

    if not isinstance(receipt, planning_store.PlanningGoalSupersedeReceipt):
        raise TypeError("receipt must be PlanningGoalSupersedeReceipt")
    canonical = json.dumps(
        {
            "schema_version": "auxiliary-v2-goal-successor-identity-v1",
            "supersede_apply_id": receipt.apply_id,
            "supersede_receipt_sha256": receipt.receipt_sha256,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    identity = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:32]
    prefix = f"auxv2successor_{identity}"
    return AuxiliaryGoalSuccessorIds(
        identity=identity,
        goal_id=f"auxgoalv2_successor_{identity}",
        bootstrap_apply_id=f"{prefix}:bootstrap",
        architect_request_id=f"{prefix}:request",
        logical_call_id=f"{prefix}:model",
        architect_apply_id=f"{prefix}:architect",
    )


def run_auxiliary_goal_successor_planning(
    request: AuxiliaryGoalSuccessorPlanningRequest,
    *,
    emit: TurnEventEmitter,
    provider: AuxiliaryGraphArchitectStructuredProvider | None = None,
    ledger_store: RuntimeModelLedgerStore,
    knowledge_cognition_history_enabled: bool = False,
) -> AuxiliaryGoalSuccessorPlanningResult:
    """启动并建模计划一个精确的后继规划阶段。"""

    if not isinstance(request, AuxiliaryGoalSuccessorPlanningRequest):
        raise TypeError(
            "request must be AuxiliaryGoalSuccessorPlanningRequest"
        )
    if not callable(emit):
        raise TypeError("emit must be callable")
    try:
        receipt = planning_store.require_authenticated_planning_goal_supersede_receipt(
            receipt=request.supersede_receipt
        )
    except planning_store.AuxiliaryGoalSupersedePersistenceError as exc:
        raise AuxiliaryGoalSuccessorPlanningError(
            "successor planning requires an authenticated supersede receipt"
        ) from exc
    ids = derive_auxiliary_goal_successor_ids(receipt)
    _require_invocation_turn_authority(request)
    task = _require_receipt_current_task(request, receipt)
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
    details = _require_details(request.session_id, request.task_id)
    bootstrapped = False
    bootstrap_commit: AuxiliaryGraphRevisionCommitResult | None = None

    if details.goal_id != ids.goal_id:
        _require_pending_old_pointer(details, receipt=receipt)
        bootstrap_reason = _successor_revision_reason(receipt)
        bootstrap = build_terminal_only_auxiliary_graph_bootstrap_proposal(
            terminal_node_key=_BOOTSTRAP_TERMINAL_KEY,
            title="确立新一轮规划目标的授权",
            objective=(
                "占位节点，不可执行；等待新一次受约束的 Architect 调用用完整"
                "方案取代它。"
            ),
            source_anchor_ids=(creation_source.anchor_id,),
            acceptance_criteria=(
                InSessionTaskAcceptanceProposal(
                    acceptance_id="successor_architect_revision_committed",
                    criterion=(
                        "在任何 AuxiliaryGraph 节点被派发之前，必须由新一次受约束的 "
                        "Architect 修订取代这个占位节点。"
                    ),
                    source_anchor_ids=(creation_source.anchor_id,),
                ),
            ),
            revision_reason=bootstrap_reason,
        )
        bootstrap_commit = auxiliary_graph_store.commit_auxiliary_graph_revision(
            session_id=request.session_id,
            turn_id=request.turn_id,
            insession_task_id=request.task_id,
            expected_task_state_version=receipt.task_state_version_after,
            expected_base_task_graph_revision=(
                receipt.next_base_task_graph_revision
            ),
            expected_control_state_version=receipt.control_state_version_after,
            expected_current_auxiliary_graph_revision=(
                receipt.superseded_auxiliary_graph_revision
            ),
            apply_id=ids.bootstrap_apply_id,
            goal_objective=receipt.next_goal_objective,
            proposal=bootstrap,
            authority_context=mounted.authority_context,
            budget_profile=selected_profile.model_dump(mode="json"),
            auxiliary_graph_id=receipt.auxiliary_graph_id,
            goal_id=ids.goal_id,
        )
        details = _require_details(request.session_id, request.task_id)
        bootstrapped = True

    _require_successor_goal_binding(details, receipt=receipt, ids=ids)
    task = _require_receipt_current_task(request, receipt)
    recovered = _recover_completed_planning(
        request=request,
        details=details,
        receipt=receipt,
        ids=ids,
        ledger_store=ledger_store,
    )
    if recovered is not None:
        decision, attempts = recovered
        return AuxiliaryGoalSuccessorPlanningResult(
            status=AuxiliaryGoalSuccessorPlanningStatus.ALREADY_PLANNED,
            details=details,
            supersede_receipt=receipt,
            ids=ids,
            bootstrapped=False,
            architect_decision=decision,
            revision_commit=_replayed_revision_commit(details),
            model_call_id=ids.logical_call_id,
            model_attempts=attempts,
            model_replayed=True,
        )

    _require_exact_successor_bootstrap(details, receipt=receipt)
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
        objective=receipt.next_goal_objective,
        desired_output=request.desired_output.strip(),
    )
    architect_request = AuxiliaryGraphArchitectRequest.create(
        architect_request_id=ids.architect_request_id,
        logical_call_id=ids.logical_call_id,
        architect_profile_id=built_request.architect_profile_id,
        goal=built_request.goal,
        prompt_payload=built_request.prompt_payload,
    )
    def rederive_state_guard() -> str:
        try:
            planning_store.require_authenticated_planning_goal_supersede_receipt(
                receipt=receipt
            )
            _require_invocation_turn_authority(request)
            current_task = _require_receipt_current_task(request, receipt)
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
                objective=receipt.next_goal_objective,
                desired_output=request.desired_output.strip(),
            )
            current_request = AuxiliaryGraphArchitectRequest.create(
                architect_request_id=ids.architect_request_id,
                logical_call_id=ids.logical_call_id,
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
        # 保留之前预留的逻辑请求的来源 Turn；
        # 当前的 Turn 只授权重放/新物理分发。
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
        return AuxiliaryGoalSuccessorPlanningResult(
            status=(
                AuxiliaryGoalSuccessorPlanningStatus.PLANNER_DECLINED_SUCCESSOR
            ),
            details=details,
            supersede_receipt=receipt,
            ids=ids,
            bootstrapped=bootstrapped,
            bootstrap_commit=bootstrap_commit,
            architect_decision=decision,
            model_call_id=model_result.model_call_id,
            model_attempts=model_result.attempts,
            model_replayed=model_result.replayed,
            failure_reason=(
                "Architect did not replace the non-executable successor shell"
            ),
        )

    durable_call.require_current_state()
    if freeze_mounted_document_planning_authority(
        session_id=request.session_id,
        task_id=request.task_id,
        allowed_managed_document_ids=request.allowed_managed_document_ids,
    ) != mounted:
        raise AuxiliaryGoalSuccessorPlanningError(
            "mounted Document authority changed before successor revision commit"
        )
    current_task = _require_receipt_current_task(request, receipt)
    current_details = _require_details(request.session_id, request.task_id)
    if current_task != task or current_details != details:
        raise AuxiliaryGoalSuccessorPlanningError(
            "successor planning authority changed before revision commit"
        )
    revision_proposal = auxiliary_graph_revision_proposal_from_architect_decision(
        decision,
        expected_current_auxiliary_graph_revision=(
            details.auxiliary_graph_revision
        ),
    )
    revision_commit = auxiliary_graph_store.commit_auxiliary_graph_revision(
        session_id=request.session_id,
        turn_id=request.turn_id,
        insession_task_id=request.task_id,
        expected_task_state_version=task.task_state_version,
        expected_base_task_graph_revision=receipt.next_base_task_graph_revision,
        expected_control_state_version=details.control_state_version,
        expected_current_auxiliary_graph_revision=(
            details.auxiliary_graph_revision
        ),
        apply_id=ids.architect_apply_id,
        goal_objective=receipt.next_goal_objective,
        proposal=revision_proposal,
        authority_context=mounted.authority_context,
        budget_profile=details.budget.base_profile.model_dump(mode="json"),
        auxiliary_graph_id=receipt.auxiliary_graph_id,
        goal_id=ids.goal_id,
    )
    committed = _require_details(request.session_id, request.task_id)
    if (
        committed.goal_id != ids.goal_id
        or committed.base_task_graph_revision
        != receipt.next_base_task_graph_revision
        or committed.target_task_graph_revision
        != receipt.next_target_task_graph_revision
        or committed.goal_objective != receipt.next_goal_objective
        or committed.auxiliary_graph_revision
        != details.auxiliary_graph_revision + 1
        or committed.structure_sha256 != revision_commit.structure_sha256
    ):
        raise AuxiliaryGoalSuccessorPlanningError(
            "Architect revision did not become the exact successor current graph"
        )
    return AuxiliaryGoalSuccessorPlanningResult(
        status=AuxiliaryGoalSuccessorPlanningStatus.PLANNED,
        details=committed,
        supersede_receipt=receipt,
        ids=ids,
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
    request: AuxiliaryGoalSuccessorPlanningRequest,
    details: StoredAuxiliaryGraphDetails,
    receipt: planning_store.PlanningGoalSupersedeReceipt,
    ids: AuxiliaryGoalSuccessorIds,
    ledger_store: RuntimeModelLedgerStore,
) -> tuple[AuxiliaryGraphArchitectDecision, int] | None:
    if _is_exact_successor_bootstrap(details, receipt=receipt):
        return None
    bootstrap_revision = receipt.superseded_auxiliary_graph_revision + 1
    if (
        details.parent_auxiliary_graph_revision != bootstrap_revision
        or details.auxiliary_graph_revision != bootstrap_revision + 1
        or details.reason == AuxiliaryGraphRevisionReason.INITIAL.value
        or details.goal_status != "active"
        or details.revision_status != "active"
    ):
        raise AuxiliaryGoalSuccessorPlanningError(
            "successor goal is neither its bootstrap nor a recoverable plan"
        )
    logical = ledger_store.get_runtime_model_logical_call(
        session_id=request.session_id,
        logical_call_id=ids.logical_call_id,
    )
    if logical is None:
        raise AuxiliaryGoalSuccessorPlanningError(
            "successor current graph has no fresh durable Architect call"
        )
    stored = logical.request
    try:
        architect_request = AuxiliaryGraphArchitectRequest.model_validate_json(
            stored.request_json
        )
    except Exception as exc:
        raise AuxiliaryGoalSuccessorPlanningError(
            "successor Architect request authority is corrupt"
        ) from exc
    current = architect_request.prompt_payload.current_revision
    if (
        stored.task_id != request.task_id
        or stored.auxiliary_graph_id != details.auxiliary_graph_id
        or stored.goal_id != details.goal_id
        or stored.call_kind != "auxiliary_graph_architect"
        or stored.request_contract != "auxiliary-graph-architect-request-v1"
        or stored.typed_result_contract
        != "auxiliary-graph-revision-proposal-v2"
        or stored.state_guard_sha256 != architect_request.binding_sha256
        or architect_request.architect_request_id != ids.architect_request_id
        or architect_request.logical_call_id != ids.logical_call_id
        or architect_request.goal.session_id != details.goal.session_id
        or architect_request.goal.task_id != details.goal.task_id
        or architect_request.goal.auxiliary_graph_id
        != details.goal.auxiliary_graph_id
        or architect_request.goal.goal_id != ids.goal_id
        or architect_request.goal.base_task_graph_revision
        != receipt.next_base_task_graph_revision
        or architect_request.goal.target_task_graph_revision
        != receipt.next_target_task_graph_revision
        or architect_request.goal.status.value != "active"
        or details.goal.status.value != "active"
        or details.goal.state_version != architect_request.goal.state_version + 1
        or architect_request.prompt_payload.goal.objective
        != receipt.next_goal_objective
        or architect_request.prompt_payload.replan_trigger is not None
        or architect_request.prompt_payload.task_graph_revision_trigger is not None
        or current is None
        or current.auxiliary_graph_revision != bootstrap_revision
        or current.base_task_graph_revision
        != receipt.next_base_task_graph_revision
    ):
        raise AuxiliaryGoalSuccessorPlanningError(
            "successor Architect request crossed immutable receipt authority"
        )
    if not logical.physical_attempts:
        raise AuxiliaryGoalSuccessorPlanningError(
            "successor Architect call has no physical attempt"
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
        raise AuxiliaryGoalSuccessorPlanningError(
            "successor Architect call lacks an exact succeeded settlement"
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
            expected_current_auxiliary_graph_revision=bootstrap_revision,
        )
    except Exception as exc:
        raise AuxiliaryGoalSuccessorPlanningError(
            "successor Architect settlement is not a committable revision"
        ) from exc
    if not _details_match_persistence_proposal(details, persisted):
        raise AuxiliaryGoalSuccessorPlanningError(
            "successor graph differs from its settled Architect proposal"
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
    if key_by_id.get(details.terminal_auxiliary_node_id) != proposal.terminal_node_key:
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
            or stored_node.acceptance_criteria != proposed_node.acceptance_criteria
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




def _require_invocation_turn_authority(
    request: AuxiliaryGoalSuccessorPlanningRequest,
) -> None:
    try:
        execution = store.inspect_turn_execution(request.session_id)
        turn = execution.get("turn")
        window = execution.get("window")
        if not isinstance(turn, Mapping) or not isinstance(window, Mapping):
            raise AuxiliaryGoalSuccessorPlanningError(
                "successor planning has no active Runtime Turn"
            )
        if (
            turn.get("turn_id") != request.turn_id
            or turn.get("session_id") != request.session_id
            or turn.get("status") != "running"
            or window.get("turn_id") != request.turn_id
            or window.get("window_state") != "active"
        ):
            raise AuxiliaryGoalSuccessorPlanningError(
                "successor planning requires its exact running Turn"
            )
        manifest = task_graph_store.get_insession_task_execution_lane_manifest(
            session_id=request.session_id,
            turn_id=request.turn_id,
        )
    except AuxiliaryGoalSuccessorPlanningError:
        raise
    except Exception as exc:
        raise AuxiliaryGoalSuccessorPlanningError(
            "successor invocation authority is unavailable"
        ) from exc
    lanes = tuple(
        lane
        for lane in manifest.lanes
        if lane.insession_task_id == request.task_id
    )
    if len(lanes) != 1 or lanes[0].execution_requested is not True:
        raise AuxiliaryGoalSuccessorPlanningError(
            "successor invocation Turn did not request this Task lane"
        )


def _architect_originating_turn_id(
    *,
    request: AuxiliaryGoalSuccessorPlanningRequest,
    architect_request: AuxiliaryGraphArchitectRequest,
    ledger_store: RuntimeModelLedgerStore,
) -> str:
    """独立于当前租约解决一个不可变的逻辑起源。"""

    logical = ledger_store.get_runtime_model_logical_call(
        session_id=request.session_id,
        logical_call_id=architect_request.logical_call_id,
    )
    if logical is None:
        return request.turn_id
    stored = getattr(logical, "request", None)
    if not isinstance(stored, RuntimeModelLogicalRequest):
        raise AuxiliaryGoalSuccessorPlanningError(
            "successor Architect logical request has the wrong durable contract"
        )
    try:
        frozen_request = AuxiliaryGraphArchitectRequest.model_validate_json(
            stored.request_json
        )
    except Exception as exc:
        raise AuxiliaryGoalSuccessorPlanningError(
            "successor Architect logical request payload is corrupt"
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
        raise AuxiliaryGoalSuccessorPlanningError(
            "successor Architect logical request crossed immutable receipt authority"
        )
    _require_authenticated_logical_execution_history(
        request=request,
        logical=logical,
    )
    return stored.invocation_turn_id


def _require_authenticated_logical_execution_history(
    *,
    request: AuxiliaryGoalSuccessorPlanningRequest,
    logical: object,
) -> None:
    stored = getattr(logical, "request", None)
    attempts = getattr(logical, "physical_attempts", None)
    if not isinstance(stored, RuntimeModelLogicalRequest) or not isinstance(
        attempts,
        tuple,
    ):
        raise AuxiliaryGoalSuccessorPlanningError(
            "successor Architect ledger history has the wrong contract"
        )
    turn_ids = [stored.invocation_turn_id]
    for physical in attempts:
        physical_request = getattr(physical, "request", None)
        if physical_request is None:
            raise AuxiliaryGoalSuccessorPlanningError(
                "successor Architect physical history is corrupt"
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
    try:
        manifest = task_graph_store.get_insession_task_execution_lane_manifest(
            session_id=session_id,
            turn_id=turn_id,
        )
    except Exception as exc:
        raise AuxiliaryGoalSuccessorPlanningError(
            "successor Architect history references an unauthenticated Turn"
        ) from exc
    lanes = tuple(
        lane
        for lane in manifest.lanes
        if lane.insession_task_id == task_id
    )
    if len(lanes) != 1 or lanes[0].execution_requested is not True:
        raise AuxiliaryGoalSuccessorPlanningError(
            "successor Architect history lacks an executable Task lane"
        )


def _require_receipt_current_task(
    request: AuxiliaryGoalSuccessorPlanningRequest,
    receipt: planning_store.PlanningGoalSupersedeReceipt,
):
    task = task_graph_store.get_insession_task_details(
        request.session_id,
        request.task_id,
    )
    if task is None:
        raise AuxiliaryGoalSuccessorPlanningError("unknown Task in this Session")
    if (
        task.status in {
            InSessionTaskStatus.COMPLETED,
            InSessionTaskStatus.CANCELLED,
        }
        or task.current_graph_revision != receipt.next_base_task_graph_revision
        or task.task_state_version != receipt.task_state_version_after
    ):
        raise AuxiliaryGoalSuccessorPlanningError(
            "Task differs from the supersede receipt successor authority: "
            f"status={task.status.value}, base={task.current_graph_revision}, "
            f"state={task.task_state_version}; expected nonterminal status, "
            f"base={receipt.next_base_task_graph_revision}, "
            f"state={receipt.task_state_version_after}"
        )
    return task


def _require_pending_old_pointer(
    details: StoredAuxiliaryGraphDetails,
    *,
    receipt: planning_store.PlanningGoalSupersedeReceipt,
) -> None:
    if (
        details.auxiliary_graph_id != receipt.auxiliary_graph_id
        or details.goal_id != receipt.superseded_goal_id
        or details.auxiliary_graph_revision
        != receipt.superseded_auxiliary_graph_revision
        or details.control_state_version != receipt.control_state_version_after
        or details.goal_state_version != receipt.goal_state_version_after
        or details.revision_state_version
        != receipt.revision_state_version_after
        or details.budget_state_version != receipt.budget_state_version_after
        or details.goal_status != "superseded"
        or details.revision_status != "superseded"
    ):
        raise AuxiliaryGoalSuccessorPlanningError(
            "current pointer is not the exact superseded receipt authority"
        )


def _require_successor_goal_binding(
    details: StoredAuxiliaryGraphDetails,
    *,
    receipt: planning_store.PlanningGoalSupersedeReceipt,
    ids: AuxiliaryGoalSuccessorIds,
) -> None:
    if (
        details.auxiliary_graph_id != receipt.auxiliary_graph_id
        or details.goal_id != ids.goal_id
        or details.goal_objective != receipt.next_goal_objective
        or details.base_task_graph_revision
        != receipt.next_base_task_graph_revision
        or details.target_task_graph_revision
        != receipt.next_target_task_graph_revision
        or details.goal.status.value != "active"
    ):
        raise AuxiliaryGoalSuccessorPlanningError(
            "successor goal differs from its supersede receipt"
        )


def _successor_revision_reason(
    receipt: planning_store.PlanningGoalSupersedeReceipt,
) -> AuxiliaryGraphRevisionReason:
    return (
        AuxiliaryGraphRevisionReason.AUTHORITY_CHANGED
        if receipt.reason is planning_store.PlanningGoalSupersedeReason.BASE_DRIFT
        else AuxiliaryGraphRevisionReason.MANUAL_REPLAN
    )


def _is_exact_successor_bootstrap(
    details: StoredAuxiliaryGraphDetails,
    *,
    receipt: planning_store.PlanningGoalSupersedeReceipt,
) -> bool:
    return (
        details.auxiliary_graph_revision
        == receipt.superseded_auxiliary_graph_revision + 1
        and details.parent_auxiliary_graph_revision
        == receipt.superseded_auxiliary_graph_revision
        and len(details.nodes) == 1
        and not details.edges
        and details.nodes[0].local_node_key == _BOOTSTRAP_TERMINAL_KEY
        and details.nodes[0].auxiliary_node_id
        == details.terminal_auxiliary_node_id
        and details.nodes[0].executor_kind == "terminal_planner"
        and details.nodes[0].output_contract == _BOOTSTRAP_OUTPUT_CONTRACT
        and details.nodes[0].capability_profile_id is None
        and details.nodes[0].status == "proposed"
        and details.reason == _successor_revision_reason(receipt).value
        and details.goal_status == "active"
        and details.revision_status == "active"
    )


def _require_exact_successor_bootstrap(
    details: StoredAuxiliaryGraphDetails,
    *,
    receipt: planning_store.PlanningGoalSupersedeReceipt,
) -> None:
    if not _is_exact_successor_bootstrap(details, receipt=receipt):
        raise AuxiliaryGoalSuccessorPlanningError(
            "successor current revision is not its non-executable bootstrap"
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
        raise AuxiliaryGoalSuccessorPlanningError(
            "successor planning requires an AuxiliaryGraph container"
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
        raise AuxiliaryGoalSuccessorPlanningError(
            "successor planning budget profile is invalid"
        ) from exc


def _require_identifier(name: str, value: str) -> None:
    if not isinstance(value, str) or not value or len(value) > 200:
        raise ValueError(f"{name} must be 1..200 characters")


__all__ = [
    "AuxiliaryGoalSuccessorIds",
    "AuxiliaryGoalSuccessorPlanningError",
    "AuxiliaryGoalSuccessorPlanningRequest",
    "AuxiliaryGoalSuccessorPlanningResult",
    "AuxiliaryGoalSuccessorPlanningStatus",
    "derive_auxiliary_goal_successor_ids",
    "run_auxiliary_goal_successor_planning",
]
