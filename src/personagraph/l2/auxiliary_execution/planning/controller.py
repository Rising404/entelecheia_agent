"""用于初始 AuxiliaryGraph 规划的有界生产组合。

持久化的辅助图容器必须存在，才能使 Architect 接收一个自认证的目标、权威状态快照和累积预算。因此，此控制器执行一个故意非执行的 Host 启动过程，调用一个确切的持久化 Architect 逻辑调用，并最多提交一个完整的替换版本。

它是一个故意的初始规划器，而不是一个自主的修订循环。一旦存在修订版二，调用者将收到``already_planned``，并可以将图形传递给执行驱动器。后续的证据/用户/验证重规划必须通过它们自己的理由约束的继续命令进入。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
import hashlib
import json

from personagraph.l2.auxiliary_graph import PlanningEpisodeBudgetProfile
from personagraph.l2.task_graph import InSessionTaskAcceptanceProposal
from personagraph.session import store
from personagraph.session.l2_store import auxiliary_graph as auxiliary_graph_store
from personagraph.session.l2_store import planning as planning_store
from personagraph.session.l2_store import task_graph as task_graph_store
from personagraph.session.l2_store.auxiliary_graph import (
    AuxiliaryGraphRevisionCommitResult,
    StoredAuxiliaryGraphDetails,
)
from .architect import (
    AuxiliaryGraphArchitectAction,
    AuxiliaryGraphArchitectDecision,
    AuxiliaryGraphArchitectRequest,
    AuxiliaryGraphArchitectStructuredProvider,
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
from personagraph.runtime.model_calls.contracts import RuntimeModelLogicalRequest
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


_BOOTSTRAP_TERMINAL_KEY = "bootstrap_terminal"
_BOOTSTRAP_OUTPUT_CONTRACT = "task_graph_revision_proposal_v2"
_SHA256_ZERO = "0" * 64


class AuxiliaryInitialPlanningStatus(StrEnum):
    PLANNED = "planned"
    ALREADY_PLANNED = "already_planned"
    REQUEST_USER_INPUT = "request_user_input"
    SUPERSEDE_REQUIRED = "supersede_required"
    TERMINAL_FAILED = "terminal_failed"
    PLANNER_DECLINED_BOOTSTRAP = "planner_declined_bootstrap"


class AuxiliaryInitialPlanningError(RuntimeError):
    """初始规划不能安全地跨越其下一个权威边界。"""

    code = "auxiliary_v2_initial_planning_rejected"


@dataclass(frozen=True, slots=True)
class AuxiliaryInitialPlanningResult:
    status: AuxiliaryInitialPlanningStatus
    details: StoredAuxiliaryGraphDetails
    bootstrapped: bool
    bootstrap_commit: AuxiliaryGraphRevisionCommitResult | None = None
    architect_decision: AuxiliaryGraphArchitectDecision | None = None
    revision_commit: AuxiliaryGraphRevisionCommitResult | None = None
    model_call_id: str | None = None
    model_attempts: int = 0
    model_replayed: bool = False
    requested_user_question: str | None = None
    failure_reason: str | None = None


TurnEventEmitter = Callable[[TurnEvent], object]


def run_initial_auxiliary_planning(
    *,
    session_id: str,
    turn_id: str,
    insession_task_id: str,
    emit: TurnEventEmitter,
    provider: AuxiliaryGraphArchitectStructuredProvider | None = None,
    desired_output: str = "完整、可执行、可验证的 TaskGraph",
    budget_profile: PlanningEpisodeBudgetProfile | Mapping[str, object] | None = None,
    deadline: TurnDeadline | None = None,
    ledger_store: RuntimeModelLedgerStore,
    allowed_managed_document_ids: tuple[str, ...] | None = None,
    knowledge_cognition_history_enabled: bool = False,
) -> AuxiliaryInitialPlanningResult:
    """实现启动权威并提交一个受保护的 Architect 图形。

    在提交修订后重新发出相同的调用是安全的：只有确切的不可变规划完成回执才能授权``already_planned``。
    """

    _require_identifier("session_id", session_id)
    _require_identifier("turn_id", turn_id)
    _require_identifier("insession_task_id", insession_task_id)
    if not callable(emit):
        raise TypeError("emit must be callable")
    normalized_output = desired_output.strip()
    if not normalized_output:
        raise ValueError("desired_output must not be empty")

    task = task_graph_store.get_insession_task_details(session_id, insession_task_id)
    if task is None:
        raise AuxiliaryInitialPlanningError("unknown Task in this Session")
    _require_invocation_turn_authority(
        session_id=session_id,
        turn_id=turn_id,
        task_id=insession_task_id,
    )
    details = auxiliary_graph_store.get_auxiliary_graph_for_task(
        session_id=session_id,
        insession_task_id=insession_task_id,
    )
    try:
        completion = planning_store.get_auxiliary_initial_planning_completion(
            session_id=session_id,
            insession_task_id=insession_task_id,
        )
    except planning_store.AuxiliaryInitialPlanningPersistenceError as exc:
        raise AuxiliaryInitialPlanningError(
            "stored initial-planning completion failed authority validation"
        ) from exc
    if details is not None and details.auxiliary_graph_revision >= 2:
        if completion is None:
            raise AuxiliaryInitialPlanningError(
                "revision two has no verifiable initial-planning completion"
            )
        return AuxiliaryInitialPlanningResult(
            status=AuxiliaryInitialPlanningStatus.ALREADY_PLANNED,
            details=details,
            bootstrapped=False,
            architect_decision=completion.architect_decision,
            revision_commit=AuxiliaryGraphRevisionCommitResult(
                status="replayed",
                auxiliary_graph_id=completion.binding.auxiliary_graph_id,
                goal_id=completion.binding.goal_id,
                committed_auxiliary_graph_revision=(
                    completion.committed_auxiliary_graph_revision
                ),
                control_state_version=(
                    completion.committed_control_state_version
                ),
                goal_state_version=completion.committed_goal_state_version,
                revision_state_version=(
                    completion.committed_revision_state_version
                ),
                budget_state_version=completion.committed_budget_state_version,
                authority_snapshot_id=(
                    completion.committed_authority_snapshot_id
                ),
                authority_snapshot_sha256=(
                    completion.committed_authority_snapshot_sha256
                ),
                structure_sha256=completion.committed_structure_sha256,
                initial_planning_completion=completion,
            ),
            model_call_id=completion.binding.architect_request.logical_call_id,
            model_attempts=completion.model_attempt_count,
            model_replayed=True,
        )
    creation_source = task_graph_store.get_insession_task_creation_source(
        session_id=session_id,
        insession_task_id=insession_task_id,
    )
    mounted = freeze_mounted_document_planning_authority(
        session_id=session_id,
        task_id=insession_task_id,
        allowed_managed_document_ids=allowed_managed_document_ids,
    )
    selected_profile = _coerce_budget_profile(budget_profile)
    bootstrapped = False
    bootstrap_commit: AuxiliaryGraphRevisionCommitResult | None = None

    if details is None:
        bootstrap = build_terminal_only_auxiliary_graph_bootstrap_proposal(
            terminal_node_key=_BOOTSTRAP_TERMINAL_KEY,
            title="确立本次规划的授权",
            objective=(
                "占位节点，不可执行；等待受约束的 Architect 用完整的初始 "
                "AuxiliaryGraph 取代它。"
            ),
            source_anchor_ids=(creation_source.anchor_id,),
            acceptance_criteria=(
                InSessionTaskAcceptanceProposal(
                    acceptance_id="architect_revision_committed",
                    criterion=(
                        "在任何 AuxiliaryGraph 节点被派发之前，必须由受约束的 "
                        "Architect 修订取代这个占位节点。"
                    ),
                    source_anchor_ids=(creation_source.anchor_id,),
                ),
            ),
        )
        identity = _stable_digest(
            {
                "schema_version": "auxiliary-v2-initial-bootstrap-identity-v1",
                "session_id": session_id,
                "task_id": insession_task_id,
            }
        )[:32]
        bootstrap_commit = auxiliary_graph_store.commit_auxiliary_graph_revision(
            session_id=session_id,
            turn_id=turn_id,
            insession_task_id=insession_task_id,
            expected_task_state_version=task.task_state_version,
            expected_base_task_graph_revision=task.current_graph_revision,
            expected_control_state_version=None,
            expected_current_auxiliary_graph_revision=None,
            apply_id=_stable_id(
                "auxv2bootstrap",
                {
                    "identity": identity,
                    "turn_id": turn_id,
                    "task_state_version": task.task_state_version,
                    "base_task_graph_revision": task.current_graph_revision,
                    "mounted_scope_sha256": mounted.scope_snapshot_sha256,
                    "proposal": bootstrap.model_dump(mode="json"),
                },
            ),
            goal_objective=task.objective,
            proposal=bootstrap,
            authority_context=mounted.authority_context,
            budget_profile=selected_profile.model_dump(mode="json"),
            auxiliary_graph_id=f"auxgraphv2_{identity}",
            goal_id=f"auxgoalv2_{identity}",
        )
        details = _require_details(session_id, insession_task_id)
        bootstrapped = True

    _require_exact_bootstrap(details)
    authority = build_mounted_document_authority_projection(
        authority_snapshot=details.authority_snapshot,
        task_creation_source=creation_source,
        mounted_authority=mounted,
    )
    workspace_runtime = build_session_workspace_readonly_runtime(
        session_id,
    )
    capabilities = build_auxiliary_planning_capability_catalog(
        mounted,
        workspace_runtime=workspace_runtime,
        knowledge_cognition_history_enabled=(
            knowledge_cognition_history_enabled
        ),
    )
    request = build_auxiliary_architect_request(
        details=details,
        authority=authority,
        capabilities=capabilities,
        objective=task.objective,
        desired_output=normalized_output,
    )

    def rederive_state_guard() -> str:
        try:
            _require_invocation_turn_authority(
                session_id=session_id,
                turn_id=turn_id,
                task_id=insession_task_id,
            )
            current_task = task_graph_store.get_insession_task_details(
                session_id,
                insession_task_id,
            )
            current_details = _require_details(session_id, insession_task_id)
            current_source = task_graph_store.get_insession_task_creation_source(
                session_id=session_id,
                insession_task_id=insession_task_id,
            )
            current_mounted = freeze_mounted_document_planning_authority(
                session_id=session_id,
                task_id=insession_task_id,
                allowed_managed_document_ids=allowed_managed_document_ids,
            )
            current_workspace_runtime = build_session_workspace_readonly_runtime(
                session_id,
            )
            if (
                current_task != task
                or current_source != creation_source
                or current_mounted != mounted
                or current_details != details
            ):
                return _SHA256_ZERO
            current_authority = build_mounted_document_authority_projection(
                authority_snapshot=current_details.authority_snapshot,
                task_creation_source=current_source,
                mounted_authority=current_mounted,
            )
            current_request = build_auxiliary_architect_request(
                details=current_details,
                authority=current_authority,
                capabilities=build_auxiliary_planning_capability_catalog(
                    current_mounted,
                    workspace_runtime=current_workspace_runtime,
                    knowledge_cognition_history_enabled=(
                        knowledge_cognition_history_enabled
                    ),
                ),
                objective=current_task.objective,
                desired_output=normalized_output,
            )
            if current_request != request:
                return _SHA256_ZERO
            return canonical_auxiliary_architect_state_guard(current_request)
        except Exception:
            return _SHA256_ZERO

    authority_values: dict[str, object] = {
        "request": request,
        # 账簿将第一个经过身份验证的 Turn 延迟为不可变
        # 来源。  ``turn_id`` 下面仅为当前执行租约
        # 用于重放或新授权的物理尝试。
        "invocation_turn_id": _architect_originating_turn_id(
            session_id=session_id,
            turn_id=turn_id,
            task_id=insession_task_id,
            architect_request=request,
            ledger_store=ledger_store,
        ),
        "rederive_state_guard_sha256": rederive_state_guard,
        "ledger_store": ledger_store,
    }
    durable_call = create_auxiliary_graph_architect_model_call_authority(
        **authority_values  # type: ignore[arg-type]
    )
    model_result = request_auxiliary_graph_architect(
        request,
        invocation_turn_id=turn_id,
        provider=provider or build_auxiliary_architect_structured_provider(),
        emit=emit,
        deadline=deadline,
        durable_call=durable_call,
    )
    decision = model_result.value

    if decision.action is AuxiliaryGraphArchitectAction.REQUEST_USER_INPUT:
        return _decision_stop_result(
            status=AuxiliaryInitialPlanningStatus.REQUEST_USER_INPUT,
            details=details,
            bootstrapped=bootstrapped,
            bootstrap_commit=bootstrap_commit,
            decision=decision,
            model_call_id=model_result.model_call_id,
            attempts=model_result.attempts,
            replayed=model_result.replayed,
            question=decision.proposal.requested_user_question,
        )
    if decision.action is AuxiliaryGraphArchitectAction.SUPERSEDE_AND_REBASE:
        return _decision_stop_result(
            status=AuxiliaryInitialPlanningStatus.SUPERSEDE_REQUIRED,
            details=details,
            bootstrapped=bootstrapped,
            bootstrap_commit=bootstrap_commit,
            decision=decision,
            model_call_id=model_result.model_call_id,
            attempts=model_result.attempts,
            replayed=model_result.replayed,
            failure=decision.proposal.explanation,
        )
    if decision.action is AuxiliaryGraphArchitectAction.TERMINAL_FAIL:
        return _decision_stop_result(
            status=AuxiliaryInitialPlanningStatus.TERMINAL_FAILED,
            details=details,
            bootstrapped=bootstrapped,
            bootstrap_commit=bootstrap_commit,
            decision=decision,
            model_call_id=model_result.model_call_id,
            attempts=model_result.attempts,
            replayed=model_result.replayed,
            failure=decision.proposal.failure_reason,
        )
    if decision.action is not AuxiliaryGraphArchitectAction.REVISE_REVISION:
        return _decision_stop_result(
            status=(
                AuxiliaryInitialPlanningStatus.PLANNER_DECLINED_BOOTSTRAP
            ),
            details=details,
            bootstrapped=bootstrapped,
            bootstrap_commit=bootstrap_commit,
            decision=decision,
            model_call_id=model_result.model_call_id,
            attempts=model_result.attempts,
            replayed=model_result.replayed,
            failure=(
                "Architect did not replace the non-executable bootstrap revision"
            ),
        )

    durable_call.require_current_state()
    current_mounted = freeze_mounted_document_planning_authority(
        session_id=session_id,
        task_id=insession_task_id,
        allowed_managed_document_ids=allowed_managed_document_ids,
    )
    if current_mounted != mounted:
        raise AuxiliaryInitialPlanningError(
            "mounted Document authority changed before revision commit"
        )
    revision_proposal = (
        auxiliary_graph_revision_proposal_from_architect_decision(
            decision,
            expected_current_auxiliary_graph_revision=(
                details.auxiliary_graph_revision
            ),
        )
    )
    revision_apply_id = _stable_id(
        "auxv2architectapply",
        {
            "session_id": session_id,
            "task_id": insession_task_id,
            "turn_id": turn_id,
            "request_binding_sha256": request.binding_sha256,
            "decision_sha256": decision.decision_sha256,
        },
    )
    completion_binding = planning_store.AuxiliaryInitialPlanningCompletionBinding.create(
        session_id=session_id,
        turn_id=turn_id,
        task_id=insession_task_id,
        auxiliary_graph_id=details.auxiliary_graph_id,
        goal_id=details.goal_id,
        bootstrap_structure_sha256=details.structure_sha256,
        revision_apply_id=revision_apply_id,
        architect_request=request,
        architect_decision=decision,
        runtime_logical_request_binding_sha256=(
            durable_call.logical_request.binding_sha256
        ),
    )
    revision_commit = auxiliary_graph_store.commit_auxiliary_graph_revision(
        session_id=session_id,
        turn_id=turn_id,
        insession_task_id=insession_task_id,
        expected_task_state_version=task.task_state_version,
        expected_base_task_graph_revision=task.current_graph_revision,
        expected_control_state_version=details.control_state_version,
        expected_current_auxiliary_graph_revision=(
            details.auxiliary_graph_revision
        ),
        apply_id=revision_apply_id,
        # 正式目标故意只存储权威身份；
        # 可读的人类目标仍然是不可变的 Task 壳字段。
        goal_objective=task.objective,
        proposal=revision_proposal,
        authority_context=current_mounted.authority_context,
        budget_profile=details.budget.base_profile.model_dump(mode="json"),
        auxiliary_graph_id=details.auxiliary_graph_id,
        goal_id=details.goal_id,
        initial_planning_completion=completion_binding,
    )
    committed = _require_details(session_id, insession_task_id)
    if (
        committed.auxiliary_graph_revision
        != details.auxiliary_graph_revision + 1
        or committed.structure_sha256 != revision_commit.structure_sha256
        or committed.auxiliary_graph_revision
        != revision_commit.committed_auxiliary_graph_revision
    ):
        raise AuxiliaryInitialPlanningError(
            "Architect revision commit did not become the exact current graph"
        )
    return AuxiliaryInitialPlanningResult(
        status=AuxiliaryInitialPlanningStatus.PLANNED,
        details=committed,
        bootstrapped=bootstrapped,
        bootstrap_commit=bootstrap_commit,
        architect_decision=decision,
        revision_commit=revision_commit,
        model_call_id=model_result.model_call_id,
        model_attempts=model_result.attempts,
        model_replayed=model_result.replayed,
    )


def _decision_stop_result(
    *,
    status: AuxiliaryInitialPlanningStatus,
    details: StoredAuxiliaryGraphDetails,
    bootstrapped: bool,
    bootstrap_commit: AuxiliaryGraphRevisionCommitResult | None,
    decision: AuxiliaryGraphArchitectDecision,
    model_call_id: str,
    attempts: int,
    replayed: bool,
    question: str | None = None,
    failure: str | None = None,
) -> AuxiliaryInitialPlanningResult:
    return AuxiliaryInitialPlanningResult(
        status=status,
        details=details,
        bootstrapped=bootstrapped,
        bootstrap_commit=bootstrap_commit,
        architect_decision=decision,
        model_call_id=model_call_id,
        model_attempts=attempts,
        model_replayed=replayed,
        requested_user_question=question,
        failure_reason=failure,
    )


def _require_exact_bootstrap(details: StoredAuxiliaryGraphDetails) -> None:
    if (
        details.auxiliary_graph_revision != 1
        or len(details.nodes) != 1
        or details.edges
        or details.nodes[0].local_node_key != _BOOTSTRAP_TERMINAL_KEY
        or details.nodes[0].auxiliary_node_id
        != details.terminal_auxiliary_node_id
        or details.nodes[0].executor_kind != "terminal_planner"
        or details.nodes[0].output_contract != _BOOTSTRAP_OUTPUT_CONTRACT
        or details.nodes[0].capability_profile_id is not None
        or details.nodes[0].status != "proposed"
    ):
        raise AuxiliaryInitialPlanningError(
            "revision one is not the non-executable initial-planning bootstrap"
        )


def _require_details(
    session_id: str,
    insession_task_id: str,
) -> StoredAuxiliaryGraphDetails:
    details = auxiliary_graph_store.get_auxiliary_graph_for_task(
        session_id=session_id,
        insession_task_id=insession_task_id,
    )
    if details is None:
        raise AuxiliaryInitialPlanningError(
            "AuxiliaryGraph state disappeared during planning"
        )
    return details


def _require_invocation_turn_authority(
    *,
    session_id: str,
    turn_id: str,
    task_id: str,
) -> None:
    """需要一个正在运行的 Turn，其中包含用于此具体 Task 的可执行通道。"""

    try:
        execution = store.inspect_turn_execution(session_id)
        turn = execution.get("turn")
        window = execution.get("window")
        if not isinstance(turn, Mapping) or not isinstance(window, Mapping):
            raise AuxiliaryInitialPlanningError(
                "initial planning invocation has no active Runtime Turn"
            )
        if (
            turn.get("turn_id") != turn_id
            or turn.get("session_id") != session_id
            or turn.get("status") != "running"
            or window.get("turn_id") != turn_id
            or window.get("window_state") != "active"
        ):
            raise AuxiliaryInitialPlanningError(
                "initial planning requires its exact running invocation Turn"
            )
        manifest = task_graph_store.get_insession_task_execution_lane_manifest(
            session_id=session_id,
            turn_id=turn_id,
        )
    except AuxiliaryInitialPlanningError:
        raise
    except Exception as exc:
        raise AuxiliaryInitialPlanningError(
            "initial planning invocation authority is unavailable"
        ) from exc
    lanes = tuple(
        lane
        for lane in manifest.lanes
        if lane.insession_task_id == task_id
    )
    if len(lanes) != 1 or lanes[0].execution_requested is not True:
        raise AuxiliaryInitialPlanningError(
            "initial planning invocation lacks an executable Task lane"
        )


def _architect_originating_turn_id(
    *,
    session_id: str,
    turn_id: str,
    task_id: str,
    architect_request: AuxiliaryGraphArchitectRequest,
    ledger_store: RuntimeModelLedgerStore,
) -> str:
    """保留初始的 Architect 逻辑请求以跨越 Turn 的传递。"""

    logical_call_id = architect_request.logical_call_id
    logical = ledger_store.get_runtime_model_logical_call(
        session_id=session_id,
        logical_call_id=logical_call_id,
    )
    if logical is None:
        return turn_id
    stored = getattr(logical, "request", None)
    if not isinstance(stored, RuntimeModelLogicalRequest):
        raise AuxiliaryInitialPlanningError(
            "initial Architect logical request has the wrong durable contract"
        )
    try:
        frozen_request = AuxiliaryGraphArchitectRequest.model_validate_json(
            stored.request_json
        )
    except Exception as exc:
        raise AuxiliaryInitialPlanningError(
            "initial Architect logical request payload is corrupt"
        ) from exc
    if (
        frozen_request != architect_request
        or stored.logical_call_id != logical_call_id
        or stored.session_id != session_id
        or stored.task_id != task_id
        or stored.auxiliary_graph_id
        != architect_request.goal.auxiliary_graph_id
        or stored.goal_id != architect_request.goal.goal_id
        or stored.execution_subject_id is not None
        or stored.call_kind != "auxiliary_graph_architect"
        or stored.purpose != "runtime_auxiliary_graph_architect_v2"
        or stored.request_contract != "auxiliary-graph-architect-request-v1"
        or stored.typed_result_contract
        != "auxiliary-graph-revision-proposal-v2"
        or stored.state_guard_sha256 != architect_request.binding_sha256
    ):
        raise AuxiliaryInitialPlanningError(
            "initial Architect logical request crossed immutable planning authority"
        )
    _require_authenticated_logical_execution_history(
        session_id=session_id,
        task_id=task_id,
        logical=logical,
    )
    return stored.invocation_turn_id




def _require_authenticated_logical_execution_history(
    *,
    session_id: str,
    task_id: str,
    logical: object,
) -> None:
    stored = getattr(logical, "request", None)
    attempts = getattr(logical, "physical_attempts", None)
    if not isinstance(stored, RuntimeModelLogicalRequest) or not isinstance(
        attempts,
        tuple,
    ):
        raise AuxiliaryInitialPlanningError(
            "initial Architect ledger history has the wrong contract"
        )
    turn_ids = [stored.invocation_turn_id]
    for physical in attempts:
        physical_request = getattr(physical, "request", None)
        if physical_request is None:
            raise AuxiliaryInitialPlanningError(
                "initial Architect physical history is corrupt"
            )
        turn_ids.append(str(physical_request.started_turn_id))
        settlement = getattr(physical, "settlement", None)
        if settlement is not None:
            turn_ids.append(str(settlement.settled_turn_id))
    for historical_turn_id in dict.fromkeys(turn_ids):
        _require_historical_task_execution_lease(
            session_id=session_id,
            task_id=task_id,
            turn_id=historical_turn_id,
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
        raise AuxiliaryInitialPlanningError(
            "initial Architect history references an unauthenticated Turn"
        ) from exc
    lanes = tuple(
        lane
        for lane in manifest.lanes
        if lane.insession_task_id == task_id
    )
    if len(lanes) != 1 or lanes[0].execution_requested is not True:
        raise AuxiliaryInitialPlanningError(
            "initial Architect history lacks an executable Task lane"
        )


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
        raise AuxiliaryInitialPlanningError(
            "initial planning budget profile is invalid"
        ) from exc


def _stable_id(prefix: str, value: object) -> str:
    return f"{prefix}_{_stable_digest(value)[:40]}"


def _stable_digest(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_identifier(name: str, value: str) -> None:
    if not isinstance(value, str) or not value or len(value) > 200:
        raise ValueError(f"{name} must be 1..200 characters")


__all__ = [
    "AuxiliaryInitialPlanningError",
    "AuxiliaryInitialPlanningResult",
    "AuxiliaryInitialPlanningStatus",
    "run_initial_auxiliary_planning",
]
