"""全新或狭义恢复 TaskNode WorkRun 的内部应用流程。

这刻意不是 Entry 适配器。它接受一个由 Host 预选且已就绪的 TaskNode，创建 WorkRun、重绑定一个
尚未决定的活跃 Attempt，或以原子方式将一个精确的等待用户问题消费进 Attempt N+1。随后，它会
驱动持久 Attempt，直至运行产出内部已验证 NodeDelivery 或到达类型化停止点。它绝不追加转录行，
也不发出公共回复。Attempt 结算与 Store 根据 Host 测量结果进行的活跃时间计费耦合；恢复后的
Attempt 会启动新计时器，不会重建已丢失进程中未提交的尾部。
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

from ....context_budget import ContextBudgetExceeded
from ....model_io.gateway import ModelGatewayError
from ...task_graph.contracts import (
    InSessionTaskAcceptanceProposal,
    InSessionTaskDetails,
)
from ....session import store as session_store
from ....session.l2_store import continuation as continuation_store
from ....session.l2_store import task_graph as task_graph_store
from ....session.l2_store import verification as verification_store
from ....session.l2_store import work_run as work_run_store
from ....session.l2_store.work_run import StoredWorkRun
from ....tools.catalog import CatalogSnapshot
from ....tools.contracts import ToolSpec
from ...work_run import (
    AttemptStatus,
    HostMaterializedCallToolsAction,
    TaskNodeVerificationRequestStatus,
    TaskNodeSubject,
    ToolResultStatus,
    WorkExecutionMutationResult,
    WorkRunStatus,
)
from ..attempts.active_time import AttemptActiveTimeMeter
from ..attempts.controller import (
    run_started_attempt,
)
from ..attempts.context_authority import project_authoritative_attempt_user_input
from ..tool_bridge.attempt_contracts import AttemptToolBridgeRequest
from ..tool_bridge.contracts import (
    AttemptToolBridge,
    bridge_supports_protected_recovery,
)
from ..attempts.input_projection import (
    AttemptDecisionContext,
    AttemptDecisionInputTooLarge,
    AttemptDecisionInputUnsupported,
    AttemptVerificationFeedback,
    PriorToolResultProjection,
    PriorToolResultsProjection,
    RequiredPriorToolResultsUnavailable,
    mandatory_prior_tool_result_ids,
    select_bounded_prior_tool_results,
)
from ..attempts.decision import (
    AttemptDecisionStructuredProvider,
    task_node_attempt_state_guard_sha256,
)
from ....runtime.turn_deadline import TurnDeadline
from ..verification.decision import (
    NodeVerificationStructuredProvider,
    request_node_verification,
    task_node_verification_state_guard_sha256,
)
from ..verification.controller import (
    NodeSemanticVerifier,
    NodeVerificationApplicationRequest,
    NodeVerificationApplicationResult,
    NodeVerificationNextAttemptPlan,
    NodeVerificationResumeRequest,
    SqliteNodeVerificationApplicationStore,
    resume_and_run_node_verification,
    run_node_verification,
)
from ..task_node.dependencies import project_task_node_dependencies
from ..task_node.dependency_delivery_contracts import (
    TaskNodeDependencyDeliveries,
    TaskNodeDependencyInputTooLarge,
    TaskNodeDependencyInputUnsupported,
)
from ..task_node.source_context import (
    TaskNodeSourceContextAuthorityError,
    build_task_node_source_context,
)
from ..task_node.model_authority_contracts import (
    TaskNodeModelCallAuthorityFactory,
    TaskNodeModelCallPlan,
)
from ..delivery.candidate_gate import (
    NodeDownstreamVerificationGateFactory,
    TaskDeliveryCandidateUnsupportedRoute,
)
from ..task_node.frontier import build_task_node_tree
from ..task_node.input_limits import AttemptDecisionInputLimits
from ....runtime.turn_events import TurnEvent
from .execution_findings import (
    project_work_run_execution_findings_for_tools,
)
from .stable_ids import WorkRunTurnStableIdPlan, _attempt_ordinal
from .turn_outcome_contracts import (
    WorkRunTurnApplicationResult,
    WorkRunTurnAuthorityUnavailable,
    WorkRunTurnFailureCode,
    WorkRunTurnOutcome,
    _failed,
)
from .turn_request_contracts import (
    WorkRunTurnApplicationRequest,
    WorkRunTurnRequest,
    WorkRunTurnResumeRequest,
    WorkRunTurnUnpreparedVerificationRecoveryRequest,
    WorkRunTurnVerificationRecoveryRequest,
    WorkRunTurnWaitingUserContinuationRequest,
)

def run_new_task_node_work_run(
    request: WorkRunTurnApplicationRequest,
    *,
    catalog_snapshot: CatalogSnapshot,
    allowed_tools: tuple[ToolSpec, ...],
    attempt_provider: AttemptDecisionStructuredProvider,
    verification_provider: NodeVerificationStructuredProvider,
    emit: Callable[[TurnEvent], object],
    id_plan: WorkRunTurnStableIdPlan,
    tool_bridge: AttemptToolBridge | None = None,
    deadline: TurnDeadline | None = None,
    verifier: NodeSemanticVerifier = request_node_verification,
    downstream_gate_factory: NodeDownstreamVerificationGateFactory | None = None,
    model_call_authority_factory: TaskNodeModelCallAuthorityFactory | None = None,
    monotonic_clock: Callable[[], float],
) -> WorkRunTurnApplicationResult:
    """驱动一个新 WorkRun，但不跨越公共交付边界。"""

    return _run_task_node_work_run(
        request,
        catalog_snapshot=catalog_snapshot,
        allowed_tools=allowed_tools,
        attempt_provider=attempt_provider,
        verification_provider=verification_provider,
        emit=emit,
        id_plan=id_plan,
        tool_bridge=tool_bridge,
        deadline=deadline,
        verifier=verifier,
        downstream_gate_factory=downstream_gate_factory,
        model_call_authority_factory=model_call_authority_factory,
        monotonic_clock=monotonic_clock,
    )


def resume_active_task_node_work_run(
    request: WorkRunTurnResumeRequest,
    *,
    catalog_snapshot: CatalogSnapshot,
    allowed_tools: tuple[ToolSpec, ...],
    attempt_provider: AttemptDecisionStructuredProvider,
    verification_provider: NodeVerificationStructuredProvider,
    emit: Callable[[TurnEvent], object],
    id_plan: WorkRunTurnStableIdPlan,
    tool_bridge: AttemptToolBridge | None = None,
    deadline: TurnDeadline | None = None,
    verifier: NodeSemanticVerifier = request_node_verification,
    downstream_gate_factory: NodeDownstreamVerificationGateFactory | None = None,
    model_call_authority_factory: TaskNodeModelCallAuthorityFactory | None = None,
    monotonic_clock: Callable[[], float],
) -> WorkRunTurnApplicationResult:
    """在先前 Turn 结算后恢复一个尚未决定的 Attempt。"""

    return _run_task_node_work_run(
        request,
        catalog_snapshot=catalog_snapshot,
        allowed_tools=allowed_tools,
        attempt_provider=attempt_provider,
        verification_provider=verification_provider,
        emit=emit,
        id_plan=id_plan,
        tool_bridge=tool_bridge,
        deadline=deadline,
        verifier=verifier,
        downstream_gate_factory=downstream_gate_factory,
        model_call_authority_factory=model_call_authority_factory,
        monotonic_clock=monotonic_clock,
    )


def recover_task_node_work_run_verification(
    request: WorkRunTurnVerificationRecoveryRequest,
    *,
    catalog_snapshot: CatalogSnapshot,
    allowed_tools: tuple[ToolSpec, ...],
    attempt_provider: AttemptDecisionStructuredProvider,
    verification_provider: NodeVerificationStructuredProvider,
    emit: Callable[[TurnEvent], object],
    id_plan: WorkRunTurnStableIdPlan,
    tool_bridge: AttemptToolBridge | None = None,
    deadline: TurnDeadline | None = None,
    verifier: NodeSemanticVerifier = request_node_verification,
    downstream_gate_factory: NodeDownstreamVerificationGateFactory | None = None,
    model_call_authority_factory: TaskNodeModelCallAuthorityFactory | None = None,
    monotonic_clock: Callable[[], float],
) -> WorkRunTurnApplicationResult:
    """在新 Turn 上重绑定并驱动一项待处理/中断的验证。"""

    return _run_task_node_work_run(
        request,
        catalog_snapshot=catalog_snapshot,
        allowed_tools=allowed_tools,
        attempt_provider=attempt_provider,
        verification_provider=verification_provider,
        emit=emit,
        id_plan=id_plan,
        tool_bridge=tool_bridge,
        deadline=deadline,
        verifier=verifier,
        downstream_gate_factory=downstream_gate_factory,
        model_call_authority_factory=model_call_authority_factory,
        monotonic_clock=monotonic_clock,
    )


def recover_unprepared_task_node_work_run_verification(
    request: WorkRunTurnUnpreparedVerificationRecoveryRequest,
    *,
    catalog_snapshot: CatalogSnapshot,
    allowed_tools: tuple[ToolSpec, ...],
    attempt_provider: AttemptDecisionStructuredProvider,
    verification_provider: NodeVerificationStructuredProvider,
    emit: Callable[[TurnEvent], object],
    id_plan: WorkRunTurnStableIdPlan,
    tool_bridge: AttemptToolBridge | None = None,
    deadline: TurnDeadline | None = None,
    verifier: NodeSemanticVerifier = request_node_verification,
    downstream_gate_factory: NodeDownstreamVerificationGateFactory | None = None,
    model_call_authority_factory: TaskNodeModelCallAuthorityFactory | None = None,
    monotonic_clock: Callable[[], float],
) -> WorkRunTurnApplicationResult:
    """准备并驱动一次在请求具体化前崩溃的提交。"""

    return _run_task_node_work_run(
        request,
        catalog_snapshot=catalog_snapshot,
        allowed_tools=allowed_tools,
        attempt_provider=attempt_provider,
        verification_provider=verification_provider,
        emit=emit,
        id_plan=id_plan,
        tool_bridge=tool_bridge,
        deadline=deadline,
        verifier=verifier,
        downstream_gate_factory=downstream_gate_factory,
        model_call_authority_factory=model_call_authority_factory,
        monotonic_clock=monotonic_clock,
    )


def continue_waiting_user_task_node_work_run(
    request: WorkRunTurnWaitingUserContinuationRequest,
    *,
    catalog_snapshot: CatalogSnapshot,
    allowed_tools: tuple[ToolSpec, ...],
    attempt_provider: AttemptDecisionStructuredProvider,
    verification_provider: NodeVerificationStructuredProvider,
    emit: Callable[[TurnEvent], object],
    id_plan: WorkRunTurnStableIdPlan,
    tool_bridge: AttemptToolBridge | None = None,
    deadline: TurnDeadline | None = None,
    verifier: NodeSemanticVerifier = request_node_verification,
    downstream_gate_factory: NodeDownstreamVerificationGateFactory | None = None,
    model_call_authority_factory: TaskNodeModelCallAuthorityFactory | None = None,
    monotonic_clock: Callable[[], float],
) -> WorkRunTurnApplicationResult:
    """消费一个精确待处理问题，并驱动其唯一的新 Attempt。"""

    return _run_task_node_work_run(
        request,
        catalog_snapshot=catalog_snapshot,
        allowed_tools=allowed_tools,
        attempt_provider=attempt_provider,
        verification_provider=verification_provider,
        emit=emit,
        id_plan=id_plan,
        tool_bridge=tool_bridge,
        deadline=deadline,
        verifier=verifier,
        downstream_gate_factory=downstream_gate_factory,
        model_call_authority_factory=model_call_authority_factory,
        monotonic_clock=monotonic_clock,
    )


def _run_task_node_work_run(
    request: WorkRunTurnRequest,
    *,
    catalog_snapshot: CatalogSnapshot,
    allowed_tools: tuple[ToolSpec, ...],
    attempt_provider: AttemptDecisionStructuredProvider,
    verification_provider: NodeVerificationStructuredProvider,
    emit: Callable[[TurnEvent], object],
    id_plan: WorkRunTurnStableIdPlan,
    tool_bridge: AttemptToolBridge | None = None,
    deadline: TurnDeadline | None = None,
    verifier: NodeSemanticVerifier = request_node_verification,
    downstream_gate_factory: NodeDownstreamVerificationGateFactory | None = None,
    model_call_authority_factory: TaskNodeModelCallAuthorityFactory | None = None,
    monotonic_clock: Callable[[], float],
) -> WorkRunTurnApplicationResult:
    """在内部驱动一个全新、崩溃恢复或等待用户的 WorkRun。"""

    catalog_descriptor = catalog_snapshot.to_descriptor()

    if isinstance(request, WorkRunTurnApplicationRequest):
        preflight = _preflight_new_run(
            request,
            catalog_snapshot=catalog_snapshot,
            allowed_tools=allowed_tools,
            tool_bridge=tool_bridge,
        )
        if isinstance(preflight, WorkRunTurnApplicationResult):
            return preflight
        details, _node = preflight
        assert request.expected_task_state_version is not None
        assert request.expected_node_state_version is not None

        try:
            created = _create_work_run(request=request, id_plan=id_plan)
        except Exception:
            # 创建事务可能已在响应丢失前提交。只有完全相同的命令才能确认这一事实。
            try:
                created = _create_work_run(request=request, id_plan=id_plan)
            except Exception:
                window = _authoritative_window(
                    request.session_id,
                    expected_turn_id=request.turn_id,
                )
                if str(window.get("current_work_run_id") or "") == id_plan.work_run_id:
                    try:
                        stored_after_create = work_run_store.get_work_run(
                            session_id=request.session_id,
                            work_run_id=id_plan.work_run_id,
                        )
                    except Exception:
                        stored_after_create = None
                    if stored_after_create is not None and (
                        stored_after_create.work_run.subject != request.subject
                        or request.turn_id not in stored_after_create.related_turn_ids
                    ):
                        raise WorkRunTurnAuthorityUnavailable(
                            "created WorkRun identity crossed TaskNode or origin Turn authority"
                        )
                    return _reconcile_after_durable_exception(
                        request=request,
                        id_plan=id_plan,
                        reason="work_run_create_response_lost",
                    )
                return _failed(
                    "work_run_create_failed",
                    window_revision=_window_revision(window),
                )

        try:
            current_mutation = _start_attempt(
                request=request,
                prior=created,
                ordinal=1,
                catalog_descriptor=catalog_descriptor,
                id_plan=id_plan,
            )
            current_active_time_meter = AttemptActiveTimeMeter.starting_now(
                monotonic_clock
            )
            resume_decided_call_tools = False
        except Exception:
            return _reconcile_after_durable_exception(
                request=request,
                id_plan=id_plan,
                reason="attempt_start_interrupted",
            )
    elif isinstance(request, WorkRunTurnUnpreparedVerificationRecoveryRequest):
        preflight = _preflight_unprepared_verification_recovery(
            request,
            catalog_snapshot=catalog_snapshot,
            allowed_tools=allowed_tools,
            id_plan=id_plan,
            tool_bridge=tool_bridge,
        )
        if isinstance(preflight, WorkRunTurnApplicationResult):
            return preflight
        details, attempt_ordinal = preflight
        verification_request = NodeVerificationApplicationRequest(
            session_id=request.session_id,
            turn_id=request.turn_id,
            work_run_id=request.work_run_id,
            submitted_attempt_id=request.submitted_attempt_id,
            verification_request_id=request.verification_request_id,
            expected_work_run_revision=request.expected_work_run_revision,
            expected_progress_revision=request.expected_progress_revision,
            expected_output_revision=request.expected_output_revision,
            expected_window_revision=request.expected_window_revision,
            prepare_apply_id=id_plan.verification_prepare_apply_id(attempt_ordinal),
            commit_apply_id=id_plan.verification_commit_apply_id(
                attempt_ordinal,
                1,
            ),
            interrupt_apply_id=id_plan.verification_interrupt_apply_id(
                attempt_ordinal,
                1,
            ),
            delivery_id=id_plan.delivery_id(attempt_ordinal),
            input_limits=request.verification_input_limits,
            recover_unprepared_submit=True,
            next_attempt=NodeVerificationNextAttemptPlan(
                attempt_id=id_plan.attempt_id(attempt_ordinal + 1),
                apply_id=id_plan.start_attempt_apply_id(attempt_ordinal + 1),
                catalog_snapshot=catalog_descriptor,
            ),
        )
        try:
            verification = _invoke_node_verification(
                verification_request,
                verification_provider=verification_provider,
                emit=emit,
                deadline=deadline,
                verifier=verifier,
                monotonic_clock=monotonic_clock,
                model_call_authority_factory=model_call_authority_factory,
                model_call_id=id_plan.verification_model_call_id(
                    attempt_ordinal
                ),
                downstream_gate_factory=downstream_gate_factory,
            )
        except TaskDeliveryCandidateUnsupportedRoute as route:
            return _reconcile_after_durable_exception(
                request=request,
                id_plan=id_plan,
                reason=_candidate_unsupported_route_reason(route),
                verification_attempt_ordinal=attempt_ordinal,
            )
        except ContextBudgetExceeded:
            raise
        except Exception:
            try:
                verification = _invoke_node_verification(
                    verification_request,
                    verification_provider=verification_provider,
                    emit=emit,
                    deadline=deadline,
                    verifier=verifier,
                    monotonic_clock=monotonic_clock,
                    model_call_authority_factory=(
                        model_call_authority_factory
                    ),
                    model_call_id=id_plan.verification_model_call_id(
                        attempt_ordinal
                    ),
                    downstream_gate_factory=downstream_gate_factory,
                )
            except ContextBudgetExceeded:
                raise
            except Exception:
                return _reconcile_after_durable_exception(
                    request=request,
                    id_plan=id_plan,
                    reason="unprepared_verification_recovery_interrupted",
                    verification_attempt_ordinal=attempt_ordinal,
                )
        projected = _project_node_verification_result(
            request=request,
            work_run_id=request.work_run_id,
            id_plan=id_plan,
            attempt_ordinal=attempt_ordinal,
            verification=verification,
        )
        if isinstance(projected, WorkRunTurnApplicationResult):
            return projected
        created = projected
        current_mutation = projected
        try:
            current_active_time_meter = AttemptActiveTimeMeter.starting_now(
                monotonic_clock
            )
            resume_decided_call_tools = False
        except Exception:
            return _reconcile_after_durable_exception(
                request=request,
                id_plan=id_plan,
                reason="next_attempt_active_time_clock_interrupted",
                verification_attempt_ordinal=attempt_ordinal,
            )
    elif isinstance(request, WorkRunTurnVerificationRecoveryRequest):
        preflight = _preflight_verification_recovery(
            request,
            catalog_snapshot=catalog_snapshot,
            allowed_tools=allowed_tools,
            id_plan=id_plan,
            tool_bridge=tool_bridge,
        )
        if isinstance(preflight, WorkRunTurnApplicationResult):
            return preflight
        details, attempt_ordinal = preflight
        prepared_request_revision = (
            request.expected_verification_request_revision + 1
        )
        recovery = NodeVerificationResumeRequest(
            session_id=request.session_id,
            turn_id=request.turn_id,
            work_run_id=request.work_run_id,
            submitted_attempt_id=request.submitted_attempt_id,
            verification_request_id=request.verification_request_id,
            expected_work_run_revision=request.expected_work_run_revision,
            expected_verification_request_revision=(
                request.expected_verification_request_revision
            ),
            expected_window_revision=request.expected_window_revision,
            resume_apply_id=id_plan.verification_recovery_apply_id(
                attempt_ordinal,
                request.expected_verification_request_revision,
                request.turn_id,
            ),
            commit_apply_id=id_plan.verification_commit_apply_id(
                attempt_ordinal,
                prepared_request_revision,
            ),
            interrupt_apply_id=id_plan.verification_interrupt_apply_id(
                attempt_ordinal,
                prepared_request_revision,
            ),
            delivery_id=id_plan.delivery_id(attempt_ordinal),
            input_limits=request.verification_input_limits,
            next_attempt=NodeVerificationNextAttemptPlan(
                attempt_id=id_plan.attempt_id(attempt_ordinal + 1),
                apply_id=id_plan.start_attempt_apply_id(attempt_ordinal + 1),
                catalog_snapshot=catalog_descriptor,
            ),
        )
        try:
            verification = _invoke_recovered_node_verification(
                recovery,
                verification_provider=verification_provider,
                emit=emit,
                deadline=deadline,
                verifier=verifier,
                monotonic_clock=monotonic_clock,
                model_call_authority_factory=model_call_authority_factory,
                model_call_id=id_plan.verification_model_call_id(
                    attempt_ordinal
                ),
                downstream_gate_factory=downstream_gate_factory,
            )
        except TaskDeliveryCandidateUnsupportedRoute as route:
            return _reconcile_after_durable_exception(
                request=request,
                id_plan=id_plan,
                reason=_candidate_unsupported_route_reason(route),
                verification_attempt_ordinal=attempt_ordinal,
            )
        except ContextBudgetExceeded:
            raise
        except Exception:
            try:
                verification = _invoke_recovered_node_verification(
                    recovery,
                    verification_provider=verification_provider,
                    emit=emit,
                    deadline=deadline,
                    verifier=verifier,
                    monotonic_clock=monotonic_clock,
                    model_call_authority_factory=(
                        model_call_authority_factory
                    ),
                    model_call_id=id_plan.verification_model_call_id(
                        attempt_ordinal
                    ),
                    downstream_gate_factory=downstream_gate_factory,
                )
            except ContextBudgetExceeded:
                raise
            except Exception:
                return _reconcile_after_durable_exception(
                    request=request,
                    id_plan=id_plan,
                    reason="verification_recovery_interrupted",
                    verification_attempt_ordinal=attempt_ordinal,
                )
        projected = _project_node_verification_result(
            request=request,
            work_run_id=request.work_run_id,
            id_plan=id_plan,
            attempt_ordinal=attempt_ordinal,
            verification=verification,
        )
        if isinstance(projected, WorkRunTurnApplicationResult):
            return projected
        created = projected
        current_mutation = projected
        try:
            current_active_time_meter = AttemptActiveTimeMeter.starting_now(
                monotonic_clock
            )
            resume_decided_call_tools = False
        except Exception:
            return _reconcile_after_durable_exception(
                request=request,
                id_plan=id_plan,
                reason="next_attempt_active_time_clock_interrupted",
                verification_attempt_ordinal=attempt_ordinal,
            )
    elif isinstance(request, WorkRunTurnResumeRequest):
        preflight = _preflight_active_attempt_resume(
            request,
            catalog_snapshot=catalog_snapshot,
            allowed_tools=allowed_tools,
            id_plan=id_plan,
            tool_bridge=tool_bridge,
        )
        if isinstance(preflight, WorkRunTurnApplicationResult):
            return preflight
        details, attempt_ordinal, resume_kind = preflight
        try:
            created = _resume_active_attempt(
                request=request,
                resume_kind=resume_kind,
                attempt_ordinal=attempt_ordinal,
                catalog_descriptor=catalog_descriptor,
                id_plan=id_plan,
                allow_protected_recovery=(
                    bridge_supports_protected_recovery(tool_bridge)
                ),
            )
        except Exception:
            try:
                created = _resume_active_attempt(
                    request=request,
                    resume_kind=resume_kind,
                    attempt_ordinal=attempt_ordinal,
                    catalog_descriptor=catalog_descriptor,
                    id_plan=id_plan,
                    allow_protected_recovery=(
                        bridge_supports_protected_recovery(tool_bridge)
                    ),
                )
            except Exception:
                window = _authoritative_window(
                    request.session_id,
                    expected_turn_id=request.turn_id,
                )
                if str(window.get("current_work_run_id") or "") == request.work_run_id:
                    return _reconcile_after_durable_exception(
                        request=request,
                        id_plan=id_plan,
                        reason="active_attempt_resume_response_lost",
                    )
                return _failed(
                    "active_attempt_resume_failed",
                    window_revision=_window_revision(window),
                    work_run_id=request.work_run_id,
                    current_attempt_id=request.attempt_id,
                )
        current_mutation = created
        try:
            # 尽力而为的崩溃尾部策略：保留持久聚合，丢弃已丢失进程的尾部，并从现在重新计时。
            current_active_time_meter = AttemptActiveTimeMeter.starting_now(
                monotonic_clock
            )
            resume_decided_call_tools = resume_kind == "decided_readonly_tools"
        except Exception:
            return _reconcile_after_durable_exception(
                request=request,
                id_plan=id_plan,
                reason="resumed_attempt_active_time_clock_interrupted",
            )
    else:
        preflight = _preflight_waiting_user_continuation(
            request,
            catalog_snapshot=catalog_snapshot,
            allowed_tools=allowed_tools,
            id_plan=id_plan,
            tool_bridge=tool_bridge,
        )
        if isinstance(preflight, WorkRunTurnApplicationResult):
            return preflight
        try:
            created = _continue_waiting_user_attempt(
                request=request,
                catalog_descriptor=catalog_descriptor,
                id_plan=id_plan,
            )
        except Exception:
            try:
                created = _continue_waiting_user_attempt(
                    request=request,
                    catalog_descriptor=catalog_descriptor,
                    id_plan=id_plan,
                )
            except Exception:
                window = _authoritative_window(
                    request.session_id,
                    expected_turn_id=request.turn_id,
                )
                expected_attempt_id = id_plan.attempt_id(
                    request.pending_question.question_attempt_ordinal + 1
                )
                if (
                    str(window.get("current_work_run_id") or "")
                    == request.work_run_id
                    and str(window.get("current_attempt_id") or "")
                    == expected_attempt_id
                ):
                    return _reconcile_after_durable_exception(
                        request=request,
                        id_plan=id_plan,
                        reason="waiting_user_continuation_response_lost",
                    )
                return _failed(
                    "waiting_user_continuation_failed",
                    window_revision=_window_revision(window),
                    work_run_id=request.work_run_id,
                )
        current_mutation = created
        try:
            current_active_time_meter = AttemptActiveTimeMeter.starting_now(
                monotonic_clock
            )
            resume_decided_call_tools = False
        except Exception:
            return _reconcile_after_durable_exception(
                request=request,
                id_plan=id_plan,
                reason="waiting_user_attempt_active_time_clock_interrupted",
            )

    while True:
        if resume_decided_call_tools:
            try:
                mutation = _dispatch_resumed_decided_call_tools_attempt(
                    request=request,
                    catalog_snapshot=catalog_snapshot,
                    allowed_tools=allowed_tools,
                    current_mutation=current_mutation,
                    active_time_meter=current_active_time_meter,
                    tool_bridge=tool_bridge,
                    id_plan=id_plan,
                )
            except Exception:
                return _reconcile_after_durable_exception(
                    request=request,
                    id_plan=id_plan,
                    reason="decided_readonly_tool_recovery_interrupted",
                )
            action = "call_tools"
            attempt_ordinal = _attempt_ordinal(request.attempt_id)
            resume_decided_call_tools = False
        else:
            try:
                context = _build_attempt_context(
                    request=request,
                    work_run_id=created.work_run_id,
                    allowed_tools=allowed_tools,
                )
                runtime_model_call_plan = _attempt_model_call_plan(
                    request=request,
                    context=context,
                    allowed_tools=allowed_tools,
                    id_plan=id_plan,
                    factory=model_call_authority_factory,
                )
            except (
                AttemptDecisionInputTooLarge,
                AttemptDecisionInputUnsupported,
                TaskNodeDependencyInputTooLarge,
                TaskNodeDependencyInputUnsupported,
            ) as exc:
                return _reconcile_after_durable_exception(
                    request=request,
                    id_plan=id_plan,
                    reason=exc.code,
                )
            except RequiredPriorToolResultsUnavailable:
                return _reconcile_after_durable_exception(
                    request=request,
                    id_plan=id_plan,
                    reason="attempt_supporting_tool_result_unavailable",
                )
            except TaskNodeSourceContextAuthorityError as exc:
                return _reconcile_after_durable_exception(
                    request=request,
                    id_plan=id_plan,
                    reason=exc.code,
                )
            except ContextBudgetExceeded:
                raise
            except Exception:
                return _reconcile_after_durable_exception(
                    request=request,
                    id_plan=id_plan,
                    reason="attempt_context_interrupted",
                )
            try:
                decision = run_started_attempt(
                    context,
                    expected_window_revision=current_mutation.window_state_version,
                    apply_id=id_plan.attempt_action_apply_id(context.attempt_ordinal),
                    provider=attempt_provider,
                    emit=emit,
                    active_time_meter=current_active_time_meter,
                    tool_bridge=tool_bridge,
                    tool_call_id_factory=id_plan.tool_call_id,
                    deadline=deadline,
                    runtime_model_call_plan=runtime_model_call_plan,
                )
            except (
                AttemptDecisionInputTooLarge,
                AttemptDecisionInputUnsupported,
                TaskNodeDependencyInputTooLarge,
                TaskNodeDependencyInputUnsupported,
            ) as exc:
                return _reconcile_after_durable_exception(
                    request=request,
                    id_plan=id_plan,
                    reason=exc.code,
                )
            except ContextBudgetExceeded:
                raise
            except Exception as exc:
                if (
                    isinstance(exc, ModelGatewayError)
                    and exc.code == "MODEL_BAD_RESPONSE"
                ):
                    # 类型化输出重试耗尽时没有写入 Attempt 动作。这不是有歧义的持久变更；
                    # 其公共 MODEL_OUTPUT_INVALID 结算由 Entry 负责。传输与截止时间错误保留下方的
                    # 跨 Turn 恢复路径。
                    raise
                try:
                    mutation = _replay_same_turn_decided_call_tools_attempt(
                        context=context,
                        expected_window_revision=(
                            current_mutation.window_state_version
                        ),
                        catalog_snapshot=catalog_snapshot,
                        allowed_tools=allowed_tools,
                        active_time_meter=current_active_time_meter,
                        tool_bridge=tool_bridge,
                        id_plan=id_plan,
                    )
                except Exception:
                    return _reconcile_after_durable_exception(
                        request=request,
                        id_plan=id_plan,
                        reason="attempt_decision_interrupted",
                    )
                action = "call_tools"
                attempt_ordinal = context.attempt_ordinal
            else:
                action = decision.action
                mutation = decision.mutation
                attempt_ordinal = context.attempt_ordinal

        try:
            stopped = _settlement_stop_result(
                request=request,
                action=action,
                mutation=mutation,
            )
        except Exception:
            return _reconcile_after_durable_exception(
                request=request,
                id_plan=id_plan,
                reason="attempt_settlement_projection_interrupted",
            )
        if stopped is not None:
            return stopped

        if action == "submit_output_window":
            verification_request_id = id_plan.verification_request_id(
                attempt_ordinal
            )
            verification_request = NodeVerificationApplicationRequest(
                session_id=request.session_id,
                turn_id=request.turn_id,
                work_run_id=created.work_run_id,
                submitted_attempt_id=context.attempt_id,
                verification_request_id=verification_request_id,
                expected_work_run_revision=mutation.work_run_revision,
                expected_progress_revision=(
                    mutation.acceptance_progress_revision
                ),
                expected_output_revision=mutation.output_window_revision,
                expected_window_revision=mutation.window_state_version,
                prepare_apply_id=id_plan.verification_prepare_apply_id(
                    attempt_ordinal
                ),
                commit_apply_id=id_plan.verification_commit_apply_id(
                    attempt_ordinal,
                    1,
                ),
                interrupt_apply_id=id_plan.verification_interrupt_apply_id(
                    attempt_ordinal,
                    1,
                ),
                delivery_id=id_plan.delivery_id(attempt_ordinal),
                input_limits=request.verification_input_limits,
                next_attempt=NodeVerificationNextAttemptPlan(
                    attempt_id=id_plan.attempt_id(attempt_ordinal + 1),
                    apply_id=id_plan.start_attempt_apply_id(
                        attempt_ordinal + 1
                    ),
                    catalog_snapshot=catalog_descriptor,
                ),
            )
            try:
                verification = _invoke_node_verification(
                    verification_request,
                    verification_provider=verification_provider,
                    emit=emit,
                    deadline=deadline,
                    verifier=verifier,
                    monotonic_clock=monotonic_clock,
                    model_call_authority_factory=(
                        model_call_authority_factory
                    ),
                    model_call_id=id_plan.verification_model_call_id(
                        attempt_ordinal
                    ),
                    downstream_gate_factory=downstream_gate_factory,
                )
            except TaskDeliveryCandidateUnsupportedRoute as route:
                return _reconcile_after_durable_exception(
                    request=request,
                    id_plan=id_plan,
                    reason=_candidate_unsupported_route_reason(route),
                    verification_attempt_ordinal=attempt_ordinal,
                )
            except ContextBudgetExceeded:
                raise
            except Exception:
                # 准备/结算事务可能已在响应丢失前提交。重新进入完全相同的逻辑请求一次；Store 应用
                # 收据会决定是恢复待处理工作，还是在不发起新语义请求的情况下投影已结算结果。
                try:
                    verification = _invoke_node_verification(
                        verification_request,
                        verification_provider=verification_provider,
                        emit=emit,
                        deadline=deadline,
                        verifier=verifier,
                        monotonic_clock=monotonic_clock,
                        model_call_authority_factory=(
                            model_call_authority_factory
                        ),
                        model_call_id=id_plan.verification_model_call_id(
                            attempt_ordinal
                        ),
                        downstream_gate_factory=downstream_gate_factory,
                    )
                except ContextBudgetExceeded:
                    raise
                except Exception:
                    return _reconcile_after_durable_exception(
                        request=request,
                        id_plan=id_plan,
                        reason="verification_application_interrupted",
                        verification_attempt_ordinal=attempt_ordinal,
                    )
            projected = _project_node_verification_result(
                request=request,
                work_run_id=created.work_run_id,
                id_plan=id_plan,
                attempt_ordinal=attempt_ordinal,
                verification=verification,
            )
            if isinstance(projected, WorkRunTurnApplicationResult):
                return projected
            current_mutation = projected
            try:
                current_active_time_meter = AttemptActiveTimeMeter.starting_now(
                    monotonic_clock
                )
            except Exception:
                return _reconcile_after_durable_exception(
                    request=request,
                    id_plan=id_plan,
                    reason="next_attempt_active_time_clock_interrupted",
                    verification_attempt_ordinal=attempt_ordinal,
                )
            continue

        try:
            stored = work_run_store.get_work_run(
                session_id=request.session_id,
                work_run_id=created.work_run_id,
            )
        except Exception:
            return _reconcile_after_durable_exception(
                request=request,
                id_plan=id_plan,
                reason="post_attempt_projection_interrupted",
            )
        if stored.work_run.status is WorkRunStatus.WAITING_EXTERNAL:
            return WorkRunTurnApplicationResult(
                outcome="waiting_external",
                work_run_id=created.work_run_id,
                window_revision=mutation.window_state_version,
            )
        if (
            stored.work_run.status is not WorkRunStatus.ACTIVE
            or stored.work_run.reason is not None
            or stored.current_attempt_id is not None
        ):
            return _reconcile_after_durable_exception(
                request=request,
                id_plan=id_plan,
                reason="unsupported_post_attempt_state",
            )
        try:
            current_mutation = _start_attempt(
                request=request,
                prior=mutation,
                ordinal=attempt_ordinal + 1,
                catalog_descriptor=catalog_descriptor,
                id_plan=id_plan,
            )
            current_active_time_meter = AttemptActiveTimeMeter.starting_now(
                monotonic_clock
            )
        except Exception:
            return _reconcile_after_durable_exception(
                request=request,
                id_plan=id_plan,
                reason="next_attempt_start_interrupted",
            )


def _invoke_node_verification(
    request: NodeVerificationApplicationRequest,
    *,
    verification_provider: NodeVerificationStructuredProvider,
    emit: Callable[[TurnEvent], object],
    deadline: TurnDeadline | None,
    verifier: NodeSemanticVerifier,
    monotonic_clock: Callable[[], float],
    model_call_authority_factory: TaskNodeModelCallAuthorityFactory | None,
    model_call_id: str,
    downstream_gate_factory: NodeDownstreamVerificationGateFactory | None,
):
    application_store = SqliteNodeVerificationApplicationStore()
    bound_verifier = _bind_runtime_node_verifier(
        verifier=verifier,
        application_store=application_store,
        model_call_authority_factory=model_call_authority_factory,
        model_call_id=model_call_id,
    )
    return run_node_verification(
        request,
        store=application_store,
        provider=verification_provider,
        emit=emit,
        monotonic_clock=monotonic_clock,
        deadline=deadline,
        verifier=bound_verifier,
        downstream_gate=(
            downstream_gate_factory(request)
            if downstream_gate_factory is not None
            else None
        ),
    )


def _invoke_recovered_node_verification(
    request: NodeVerificationResumeRequest,
    *,
    verification_provider: NodeVerificationStructuredProvider,
    emit: Callable[[TurnEvent], object],
    deadline: TurnDeadline | None,
    verifier: NodeSemanticVerifier,
    monotonic_clock: Callable[[], float],
    model_call_authority_factory: TaskNodeModelCallAuthorityFactory | None,
    model_call_id: str,
    downstream_gate_factory: NodeDownstreamVerificationGateFactory | None,
):
    application_store = SqliteNodeVerificationApplicationStore()
    bound_verifier = _bind_runtime_node_verifier(
        verifier=verifier,
        application_store=application_store,
        model_call_authority_factory=model_call_authority_factory,
        model_call_id=model_call_id,
    )
    return resume_and_run_node_verification(
        request,
        store=application_store,
        provider=verification_provider,
        emit=emit,
        monotonic_clock=monotonic_clock,
        deadline=deadline,
        verifier=bound_verifier,
        downstream_gate=(
            downstream_gate_factory(request)
            if downstream_gate_factory is not None
            else None
        ),
    )


def _bind_runtime_node_verifier(
    *,
    verifier: NodeSemanticVerifier,
    application_store: SqliteNodeVerificationApplicationStore,
    model_call_authority_factory: TaskNodeModelCallAuthorityFactory | None,
    model_call_id: str,
) -> NodeSemanticVerifier:
    if model_call_authority_factory is None:
        return verifier
    if verifier is not request_node_verification:
        raise ValueError(
            "generic TaskNode authority requires the standard semantic verifier"
        )

    def durable_verifier(
        context,
        *,
        provider,
        emit,
        deadline=None,
    ):
        plan = TaskNodeModelCallPlan(
            logical_call_id=model_call_id,
            request_turn_id=context.request_turn_id,
            authority_factory=model_call_authority_factory,
            rederive_state_guard_sha256=lambda: (
                task_node_verification_state_guard_sha256(
                    application_store.reproject_pending_context(
                        session_id=context.session_id,
                        invocation_turn_id=context.invocation_turn_id,
                        verification_request_id=(
                            context.verification_request_id
                        ),
                        input_limits=context.input_limits,
                    )
                )
            ),
        )
        return request_node_verification(
            context,
            provider=provider,
            emit=emit,
            deadline=deadline,
            runtime_model_call_plan=plan,
        )

    return durable_verifier


def _project_node_verification_result(
    *,
    request: WorkRunTurnRequest,
    work_run_id: str,
    id_plan: WorkRunTurnStableIdPlan,
    attempt_ordinal: int,
    verification: NodeVerificationApplicationResult,
) -> WorkExecutionMutationResult | WorkRunTurnApplicationResult:
    """映射一次语义结算，或暴露由 Store 启动的下一个 Attempt。"""

    if verification.outcome == "work_run_limit_reached":
        return WorkRunTurnApplicationResult(
            outcome="work_run_failed",
            work_run_id=work_run_id,
            window_revision=verification.store_projection.window_revision,
        )
    if verification.outcome == "interrupted":
        return WorkRunTurnApplicationResult(
            outcome="verification_interrupted",
            work_run_id=work_run_id,
            window_revision=verification.store_projection.window_revision,
            interruption_reason=verification.interruption_reason,
        )
    if verification.outcome == "passed":
        delivery_id = verification.store_projection.delivery_id
        if delivery_id is None:
            return _reconcile_after_durable_exception(
                request=request,
                id_plan=id_plan,
                reason="delivery_projection_missing",
                verification_attempt_ordinal=attempt_ordinal,
            )
        try:
            verification_store.get_task_node_delivery(
                session_id=request.session_id,
                delivery_id=delivery_id,
            )
        except Exception:
            return _reconcile_after_durable_exception(
                request=request,
                id_plan=id_plan,
                reason="delivery_projection_interrupted",
                verification_attempt_ordinal=attempt_ordinal,
            )
        return WorkRunTurnApplicationResult(
            outcome="delivery_ready",
            work_run_id=work_run_id,
            delivery_id=delivery_id,
            window_revision=verification.store_projection.window_revision,
        )
    if verification.next_attempt_mutation is not None:
        return verification.next_attempt_mutation
    if (
        verification.store_projection.work_run_status
        is WorkRunStatus.TURN_LIMIT_REACHED
        and verification.store_projection.work_run_reason == "turn_limit_reached"
    ):
        return WorkRunTurnApplicationResult(
            outcome="turn_limit_reached",
            work_run_id=work_run_id,
            window_revision=verification.store_projection.window_revision,
        )
    return _reconcile_after_durable_exception(
        request=request,
        id_plan=id_plan,
        reason="verification_next_attempt_missing",
        verification_attempt_ordinal=attempt_ordinal,
    )


def _create_work_run(
    *,
    request: WorkRunTurnApplicationRequest,
    id_plan: WorkRunTurnStableIdPlan,
) -> WorkExecutionMutationResult:
    assert request.expected_task_state_version is not None
    assert request.expected_node_state_version is not None
    return work_run_store.create_task_node_work_run(
        session_id=request.session_id,
        turn_id=request.turn_id,
        subject=request.subject,
        expected_task_state_version=request.expected_task_state_version,
        expected_node_state_version=request.expected_node_state_version,
        expected_window_revision=request.expected_window_revision,
        apply_id=id_plan.create_apply_id,
        work_run_id=id_plan.work_run_id,
    )


def _resume_active_attempt(
    *,
    request: WorkRunTurnResumeRequest,
    resume_kind: Literal[
        "undecided",
        "decided_readonly_tools",
        "closed_readonly_tools",
    ],
    attempt_ordinal: int,
    catalog_descriptor: dict[str, object],
    id_plan: WorkRunTurnStableIdPlan,
    allow_protected_recovery: bool,
) -> WorkExecutionMutationResult:
    common = {
        "session_id": request.session_id,
        "turn_id": request.turn_id,
        "work_run_id": request.work_run_id,
        "attempt_id": request.attempt_id,
        "expected_work_run_revision": request.expected_work_run_revision,
        "expected_window_revision": request.expected_window_revision,
    }
    if resume_kind == "undecided":
        return work_run_store.resume_active_work_run_attempt(
            **common,
            apply_id=id_plan.resume_active_attempt_apply_id(
                attempt_ordinal,
                request.turn_id,
            ),
        )
    if resume_kind == "decided_readonly_tools":
        return work_run_store.resume_decided_readonly_tool_attempt(
            **common,
            catalog_snapshot=catalog_descriptor,
            apply_id=id_plan.resume_active_attempt_apply_id(
                attempt_ordinal,
                request.turn_id,
            ),
            allow_protected_recovery=allow_protected_recovery,
        )
    return work_run_store.resume_idle_readonly_work_run_and_start_attempt(
        session_id=request.session_id,
        turn_id=request.turn_id,
        work_run_id=request.work_run_id,
        closed_attempt_id=request.attempt_id,
        next_attempt_id=id_plan.attempt_id(attempt_ordinal + 1),
        expected_work_run_revision=request.expected_work_run_revision,
        expected_window_revision=request.expected_window_revision,
        catalog_snapshot=catalog_descriptor,
        apply_id=id_plan.resume_idle_work_run_apply_id(
            attempt_ordinal,
            request.turn_id,
        ),
        allow_protected_recovery=allow_protected_recovery,
    )


def _dispatch_resumed_decided_call_tools_attempt(
    *,
    request: WorkRunTurnResumeRequest,
    catalog_snapshot: CatalogSnapshot,
    allowed_tools: tuple[ToolSpec, ...],
    current_mutation: WorkExecutionMutationResult,
    active_time_meter: AttemptActiveTimeMeter,
    tool_bridge: AttemptToolBridge | None,
    id_plan: WorkRunTurnStableIdPlan,
) -> WorkExecutionMutationResult:
    """在不调用模型的情况下继续一个已决定且桥接器可恢复的批次。"""

    if tool_bridge is None:
        raise RuntimeError("decided tool recovery requires a Tool Bridge")
    stored = work_run_store.get_work_run(
        session_id=request.session_id,
        work_run_id=request.work_run_id,
    )
    attempts = tuple(
        item
        for item in stored.attempts
        if item.attempt.attempt_id == request.attempt_id
    )
    if len(attempts) != 1:
        raise RuntimeError("decided tool recovery Attempt is not unique")
    attempt = attempts[0]
    decision = attempt.decision
    supports_protected_recovery = bridge_supports_protected_recovery(tool_bridge)
    if (
        stored.work_run.status is not WorkRunStatus.ACTIVE
        or stored.work_run.reason is not None
        or stored.current_attempt_id != request.attempt_id
        or attempt.turn_id != request.turn_id
        or attempt.attempt.status is not AttemptStatus.ACTIVE
        or attempt.action != "call_tools"
        or decision is None
        or not isinstance(decision.action, HostMaterializedCallToolsAction)
        or attempt.catalog_snapshot != catalog_snapshot.to_descriptor()
        or id_plan.attempt_id(attempt.attempt.ordinal) != request.attempt_id
        or tuple(call.tool_call_id for call in decision.action.calls)
        != tuple(
            id_plan.tool_call_id(request.attempt_id, ordinal)
            for ordinal in range(1, len(decision.action.calls) + 1)
        )
        or (
            not supports_protected_recovery
            and (
                any(call.modifies_environment for call in decision.action.calls)
                or any(
                    item.attempt_id == request.attempt_id
                    and item.status is ToolResultStatus.COMPLETION_UNCONFIRMED
                    for item in stored.tool_results
                )
            )
        )
    ):
        raise RuntimeError("decided tool recovery authority changed after rebind")
    return tool_bridge.dispatch(
        AttemptToolBridgeRequest(
            session_id=request.session_id,
            turn_id=request.turn_id,
            work_run_id=request.work_run_id,
            attempt_id=request.attempt_id,
            expected_work_run_revision=current_mutation.work_run_revision,
            expected_progress_revision=(
                current_mutation.acceptance_progress_revision
            ),
            expected_output_revision=current_mutation.output_window_revision,
            expected_window_revision=current_mutation.window_state_version,
            apply_id=id_plan.attempt_action_apply_id(attempt.attempt.ordinal),
            decision=decision,
            allowed_tools=allowed_tools,
            catalog_snapshot=attempt.catalog_snapshot,
            recovery_mutation=current_mutation,
        ),
        active_time_meter=active_time_meter,
    )


def _replay_same_turn_decided_call_tools_attempt(
    *,
    context: AttemptDecisionContext,
    expected_window_revision: int,
    catalog_snapshot: CatalogSnapshot,
    allowed_tools: tuple[ToolSpec, ...],
    active_time_meter: AttemptActiveTimeMeter,
    tool_bridge: AttemptToolBridge | None,
    id_plan: WorkRunTurnStableIdPlan,
) -> WorkExecutionMutationResult:
    """在持久决定/结果/关闭写入后，重放同一 Turn 的桥接器。"""

    if tool_bridge is None:
        raise RuntimeError("call_tools replay requires a Tool Bridge")
    stored = work_run_store.get_work_run(
        session_id=context.session_id,
        work_run_id=context.work_run_id,
    )
    attempts = tuple(
        item
        for item in stored.attempts
        if item.attempt.attempt_id == context.attempt_id
    )
    if len(attempts) != 1:
        raise RuntimeError("same-Turn tool replay Attempt is not unique")
    attempt = attempts[0]
    decision = attempt.decision
    results = tuple(
        item for item in stored.tool_results if item.attempt_id == context.attempt_id
    )
    supports_protected_recovery = bridge_supports_protected_recovery(tool_bridge)
    if (
        attempt.turn_id != context.turn_id
        or attempt.action != "call_tools"
        or decision is None
        or not isinstance(decision.action, HostMaterializedCallToolsAction)
        or attempt.catalog_snapshot != catalog_snapshot.to_descriptor()
        or id_plan.attempt_id(attempt.attempt.ordinal) != context.attempt_id
        or tuple(call.tool_call_id for call in decision.action.calls)
        != tuple(
            id_plan.tool_call_id(context.attempt_id, ordinal)
            for ordinal in range(1, len(decision.action.calls) + 1)
        )
        or (
            not supports_protected_recovery
            and (
                any(call.modifies_environment for call in decision.action.calls)
                or any(
                    item.status is ToolResultStatus.COMPLETION_UNCONFIRMED
                    for item in results
                )
            )
        )
    ):
        raise RuntimeError("same-Turn tool replay authority is unavailable")
    return tool_bridge.dispatch(
        AttemptToolBridgeRequest(
            session_id=context.session_id,
            turn_id=context.turn_id,
            work_run_id=context.work_run_id,
            attempt_id=context.attempt_id,
            expected_work_run_revision=context.work_run_revision,
            expected_progress_revision=context.acceptance_progress.revision,
            expected_output_revision=context.output_window.output_revision,
            expected_window_revision=expected_window_revision,
            apply_id=id_plan.attempt_action_apply_id(context.attempt_ordinal),
            decision=decision,
            allowed_tools=allowed_tools,
            catalog_snapshot=attempt.catalog_snapshot,
        ),
        active_time_meter=active_time_meter,
    )


def _continue_waiting_user_attempt(
    *,
    request: WorkRunTurnWaitingUserContinuationRequest,
    catalog_descriptor: dict[str, object],
    id_plan: WorkRunTurnStableIdPlan,
) -> WorkExecutionMutationResult:
    question = request.pending_question
    next_ordinal = question.question_attempt_ordinal + 1
    expected_attempt_id = id_plan.attempt_id(next_ordinal)
    mutation = continuation_store.continue_waiting_user_work_run_and_start_attempt(
        session_id=request.session_id,
        turn_id=request.turn_id,
        work_run_id=question.work_run_id,
        subject=question.subject,
        question_attempt_id=question.question_attempt_id,
        expected_work_run_revision=question.work_run_revision,
        expected_task_state_version=question.task_state_version,
        expected_node_state_version=question.node_state_version,
        expected_progress_revision=question.acceptance_progress_revision,
        expected_window_revision=request.expected_window_revision,
        apply_id=id_plan.continue_waiting_user_apply_id(
            question.question_attempt_ordinal
        ),
        catalog_snapshot=catalog_descriptor,
        attempt_id=expected_attempt_id,
    )
    if (
        mutation.work_run_id != question.work_run_id
        or mutation.current_attempt_id != expected_attempt_id
        or mutation.attempt is None
        or mutation.attempt.attempt_id != expected_attempt_id
        or mutation.attempt.ordinal != next_ordinal
        or mutation.work_run_status is not WorkRunStatus.ACTIVE
        or mutation.work_run_reason is not None
    ):
        raise RuntimeError("Store continued an unexpected waiting-user Attempt")
    return mutation


def _candidate_unsupported_route_reason(
    route: TaskDeliveryCandidateUnsupportedRoute,
) -> str:
    return f"task_delivery_candidate_{route.disposition.value}"


def _reconcile_after_durable_exception(
    *,
    request: WorkRunTurnRequest,
    id_plan: WorkRunTurnStableIdPlan,
    reason: str,
    verification_attempt_ordinal: int | None = None,
) -> WorkRunTurnApplicationResult:
    """投影已提交权威，或返回可恢复的中断游标。

    此路径绝不会将变更后异常报告为终态失败。返回的 Window 修订版本在异常后读取，因此是所属
    Turn 可用于其中断变更的唯一修订版本。
    """

    window = _authoritative_window(
        request.session_id,
        expected_turn_id=request.turn_id,
    )
    try:
        stored = work_run_store.get_work_run(
            session_id=request.session_id,
            work_run_id=id_plan.work_run_id,
        )
    except Exception:
        stored = None

    if stored is not None:
        if (
            stored.work_run.subject != request.subject
            or request.turn_id not in stored.related_turn_ids
        ):
            raise WorkRunTurnAuthorityUnavailable(
                "reconciled WorkRun crossed TaskNode or origin Turn authority"
            )
        run = stored.work_run
        if run.status is WorkRunStatus.COMPLETED and stored.node_delivery_id is not None:
            return WorkRunTurnApplicationResult(
                outcome="delivery_ready",
                work_run_id=id_plan.work_run_id,
                delivery_id=stored.node_delivery_id,
                window_revision=_window_revision(window),
            )
        if run.status is WorkRunStatus.WAITING_USER and stored.pending_user_question:
            return WorkRunTurnApplicationResult(
                outcome="waiting_user",
                work_run_id=id_plan.work_run_id,
                pending_user_question=stored.pending_user_question,
                window_revision=_window_revision(window),
            )
        if run.status is WorkRunStatus.WAITING_EXTERNAL:
            return WorkRunTurnApplicationResult(
                outcome="waiting_external",
                work_run_id=id_plan.work_run_id,
                window_revision=_window_revision(window),
            )
        if (
            run.status is WorkRunStatus.CANCELLED
            and run.reason == "task_graph_revision_requested"
        ):
            authority = (
                work_run_store.get_active_task_graph_execution_replan_request(
                    session_id=request.session_id,
                    task_id=request.subject.task_id,
                )
            )
            if (
                authority is None
                or authority.work_run_id != id_plan.work_run_id
                or authority.requesting_subject != request.subject
            ):
                raise WorkRunTurnAuthorityUnavailable(
                    "execution replan reconciliation lost active authority"
                )
            return WorkRunTurnApplicationResult(
                outcome="task_graph_revision_requested",
                work_run_id=id_plan.work_run_id,
                window_revision=_window_revision(window),
            )

    verification_record = None
    if verification_attempt_ordinal is not None:
        try:
            verification_record = verification_store.get_task_node_verification_record(
                session_id=request.session_id,
                verification_request_id=id_plan.verification_request_id(
                    verification_attempt_ordinal
                ),
            )
        except Exception:
            verification_record = None
    elif stored is not None and stored.current_verification_request_id is not None:
        try:
            verification_record = verification_store.get_task_node_verification_record(
                session_id=request.session_id,
                verification_request_id=stored.current_verification_request_id,
            )
        except Exception:
            verification_record = None

    if verification_record is not None:
        verification_request = verification_record.request
        recovering_same_request = (
            isinstance(request, WorkRunTurnVerificationRecoveryRequest)
            and verification_request.verification_request_id
            == request.verification_request_id
            and verification_request.submitted_attempt_id
            == request.submitted_attempt_id
        )
        if (
            verification_request.work_run_id != id_plan.work_run_id
            or verification_request.subject != request.subject
            or (
                not recovering_same_request
                and verification_request.request_turn_id != request.turn_id
            )
        ):
            raise WorkRunTurnAuthorityUnavailable(
                "verification reconciliation crossed WorkRun or origin Turn authority"
            )
        if verification_record.result is not None and verification_record.result.all_pass:
            ordinal = verification_attempt_ordinal or _attempt_ordinal(
                verification_request.submitted_attempt_id
            )
            return WorkRunTurnApplicationResult(
                outcome="delivery_ready",
                work_run_id=id_plan.work_run_id,
                delivery_id=id_plan.delivery_id(ordinal),
                window_revision=_window_revision(window),
            )
        if verification_request.status.value == "interrupted":
            return WorkRunTurnApplicationResult(
                outcome="verification_interrupted",
                work_run_id=id_plan.work_run_id,
                window_revision=_window_revision(window),
                interruption_reason=verification_request.technical_error_code,
            )

    current_attempt_id = None
    if str(window.get("current_work_run_id") or "") == id_plan.work_run_id:
        raw_attempt_id = window.get("current_attempt_id")
        current_attempt_id = str(raw_attempt_id) if raw_attempt_id is not None else None
    elif stored is not None:
        current_attempt_id = stored.current_attempt_id
    return WorkRunTurnApplicationResult(
        outcome="internal_interrupted",
        work_run_id=id_plan.work_run_id,
        current_attempt_id=current_attempt_id,
        window_revision=_window_revision(window),
        interruption_reason=reason,
    )


def _authoritative_window(
    session_id: str,
    *,
    expected_turn_id: str,
) -> dict[str, object]:
    window = session_store.get_turn_execution_window(session_id)
    if window is None:
        raise WorkRunTurnAuthorityUnavailable(
            "authoritative TurnExecutionWindow is unavailable"
        )
    if (
        str(window.get("turn_id") or "") != expected_turn_id
        or str(window.get("window_state") or "") != "active"
    ):
        raise WorkRunTurnAuthorityUnavailable(
            "originating Turn no longer owns the active execution Window"
        )
    _window_revision(window)
    return window


def _window_revision(window: dict[str, object]) -> int:
    value = window.get("state_version")
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise WorkRunTurnAuthorityUnavailable(
            "authoritative TurnExecutionWindow revision is invalid"
        )
    return value


def _preflight_new_run(
    request: WorkRunTurnApplicationRequest,
    *,
    catalog_snapshot: CatalogSnapshot,
    allowed_tools: tuple[ToolSpec, ...],
    tool_bridge: AttemptToolBridge | None,
) -> (
    tuple[InSessionTaskDetails, dict[str, object]]
    | WorkRunTurnApplicationResult
):
    if (
        request.expected_task_state_version is None
        or request.expected_node_state_version is None
    ):
        return _failed(
            "missing_authority_cas",
            window_revision=request.expected_window_revision,
        )
    try:
        window = session_store.get_turn_execution_window(request.session_id)
    except Exception:
        window = None
    if (
        window is None
        or str(window.get("turn_id") or "") != request.turn_id
        or str(window.get("window_state") or "") != "active"
        or int(window.get("state_version") or 0) != request.expected_window_revision
    ):
        return _failed(
            "turn_window_stale",
            window_revision=request.expected_window_revision,
        )
    if window.get("current_work_run_id") is not None:
        return _failed(
            "existing_nonterminal_work_run",
            window_revision=request.expected_window_revision,
        )
    try:
        details = task_graph_store.get_insession_task_details(
            request.session_id,
            request.subject.task_id,
        )
    except Exception:
        details = None
    if details is None:
        return _failed(
            "task_details_unavailable",
            window_revision=request.expected_window_revision,
        )
    if (
        details.task_state_version != request.expected_task_state_version
        or details.current_graph_revision != request.subject.graph_revision
    ):
        return _failed(
            "task_authority_stale",
            window_revision=request.expected_window_revision,
        )
    matches = tuple(
        node
        for node in details.nodes
        if node.get("insession_task_node_id") == request.subject.node_id
        and node.get("node_revision") == request.subject.node_revision
    )
    if len(matches) != 1 or (
        matches[0].get("state_version") != request.expected_node_state_version
    ):
        return _failed(
            "node_authority_stale",
            window_revision=request.expected_window_revision,
        )
    node = matches[0]
    unfinished_children = tuple(
        child
        for child in details.nodes
        if child.get("parent_insession_task_node_id") == request.subject.node_id
        and child.get("status") != "completed"
    )
    if node.get("status") not in {"proposed", "active", "interrupted"} or unfinished_children:
        return _failed(
            "node_not_ready",
            window_revision=request.expected_window_revision,
        )
    try:
        linked_task_ids = session_store.list_turn_insession_task_ids(
            request.session_id,
            request.turn_id,
        )
    except Exception:
        linked_task_ids = ()
    if request.subject.task_id not in linked_task_ids:
        return _failed(
            "task_not_linked",
            window_revision=request.expected_window_revision,
        )
    try:
        candidates = work_run_store.list_turn_linked_nonterminal_work_runs(
            session_id=request.session_id,
            turn_id=request.turn_id,
        )
    except Exception:
        return _failed(
            "existing_nonterminal_work_run",
            window_revision=request.expected_window_revision,
        )
    if any(candidate.subject == request.subject for candidate in candidates):
        return _failed(
            "existing_nonterminal_work_run",
            window_revision=request.expected_window_revision,
        )
    exposed = tuple(entry.registration.spec for entry in catalog_snapshot.exposed())
    if tuple(item.to_dict() for item in exposed) != tuple(
        item.to_dict() for item in allowed_tools
    ):
        return _failed(
            "catalog_projection_mismatch",
            window_revision=request.expected_window_revision,
        )
    if allowed_tools and tool_bridge is None:
        return _failed(
            "tool_bridge_unavailable",
            window_revision=request.expected_window_revision,
        )
    return details, node


def _preflight_unprepared_verification_recovery(
    request: WorkRunTurnUnpreparedVerificationRecoveryRequest,
    *,
    catalog_snapshot: CatalogSnapshot,
    allowed_tools: tuple[ToolSpec, ...],
    id_plan: WorkRunTurnStableIdPlan,
    tool_bridge: AttemptToolBridge | None,
) -> tuple[InSessionTaskDetails, int] | WorkRunTurnApplicationResult:
    """验证一个从未生成验证请求的已提交 WorkRun。"""

    def failed() -> WorkRunTurnApplicationResult:
        return _failed(
            "unprepared_verification_recovery_failed",
            window_revision=request.expected_window_revision,
            work_run_id=request.work_run_id,
            current_attempt_id=request.submitted_attempt_id,
        )

    try:
        window = session_store.get_turn_execution_window(request.session_id)
        details = task_graph_store.get_insession_task_details(
            request.session_id,
            request.subject.task_id,
        )
        linked_task_ids = session_store.list_turn_insession_task_ids(
            request.session_id,
            request.turn_id,
        )
        candidates = work_run_store.list_turn_linked_nonterminal_work_runs(
            session_id=request.session_id,
            turn_id=request.turn_id,
        )
        stored = work_run_store.get_work_run(
            session_id=request.session_id,
            work_run_id=request.work_run_id,
        )
    except Exception:
        return failed()
    if (
        window is None
        or str(window.get("turn_id") or "") != request.turn_id
        or str(window.get("window_state") or "") != "active"
        or int(window.get("state_version") or 0)
        != request.expected_window_revision
        or window.get("current_work_run_id") is not None
        or window.get("current_attempt_id") is not None
        or window.get("pending_operation_id") is not None
    ):
        return _failed(
            "turn_window_stale",
            window_revision=request.expected_window_revision,
            work_run_id=request.work_run_id,
            current_attempt_id=request.submitted_attempt_id,
        )
    exact_candidates = tuple(
        candidate
        for candidate in candidates
        if candidate.work_run_id == request.work_run_id
        and candidate.subject == request.subject
    )
    if (
        id_plan.work_run_id != request.work_run_id
        or details is None
        or details.current_graph_revision != request.subject.graph_revision
        or request.subject.task_id not in linked_task_ids
        or len(exact_candidates) != 1
        or exact_candidates[0].status is not WorkRunStatus.ACTIVE
        or exact_candidates[0].reason != "verification_pending"
        or exact_candidates[0].work_run_revision
        != request.expected_work_run_revision
        or exact_candidates[0].current_attempt_id is not None
        or exact_candidates[0].current_verification_request_id is not None
        or stored.work_run.subject != request.subject
        or stored.work_run.status is not WorkRunStatus.ACTIVE
        or stored.work_run.reason != "verification_pending"
        or stored.work_run.revision != request.expected_work_run_revision
        or stored.acceptance_progress.revision
        != request.expected_progress_revision
        or stored.output_window.output_revision
        != request.expected_output_revision
        or stored.current_attempt_id is not None
        or stored.current_verification_request_id is not None
    ):
        return failed()
    nodes = tuple(
        node
        for node in details.nodes
        if node.get("insession_task_node_id") == request.subject.node_id
        and node.get("node_revision") == request.subject.node_revision
    )
    if len(nodes) != 1 or nodes[0].get("status") != "active":
        return _failed(
            "node_authority_stale",
            window_revision=request.expected_window_revision,
            work_run_id=request.work_run_id,
            current_attempt_id=request.submitted_attempt_id,
        )
    if any(
        child.get("status") != "completed"
        for child in details.nodes
        if child.get("parent_insession_task_node_id") == request.subject.node_id
    ):
        return _failed(
            "node_not_ready",
            window_revision=request.expected_window_revision,
            work_run_id=request.work_run_id,
            current_attempt_id=request.submitted_attempt_id,
        )
    submitted = tuple(
        item
        for item in stored.attempts
        if item.attempt.attempt_id == request.submitted_attempt_id
    )
    if (
        len(submitted) != 1
        or submitted[0].attempt.status is not AttemptStatus.CLOSED
        or submitted[0].action != "submit_output_window"
        or submitted[0].attempt.submitted_output_revision
        != request.expected_output_revision
        or id_plan.attempt_id(submitted[0].attempt.ordinal)
        != request.submitted_attempt_id
        or id_plan.verification_request_id(submitted[0].attempt.ordinal)
        != request.verification_request_id
    ):
        return failed()
    exposed = tuple(entry.registration.spec for entry in catalog_snapshot.exposed())
    if tuple(item.to_dict() for item in exposed) != tuple(
        item.to_dict() for item in allowed_tools
    ):
        return _failed(
            "catalog_projection_mismatch",
            window_revision=request.expected_window_revision,
            work_run_id=request.work_run_id,
            current_attempt_id=request.submitted_attempt_id,
        )
    if allowed_tools and tool_bridge is None:
        return _failed(
            "tool_bridge_unavailable",
            window_revision=request.expected_window_revision,
            work_run_id=request.work_run_id,
            current_attempt_id=request.submitted_attempt_id,
        )
    return details, submitted[0].attempt.ordinal


def _preflight_verification_recovery(
    request: WorkRunTurnVerificationRecoveryRequest,
    *,
    catalog_snapshot: CatalogSnapshot,
    allowed_tools: tuple[ToolSpec, ...],
    id_plan: WorkRunTurnStableIdPlan,
    tool_bridge: AttemptToolBridge | None,
) -> (
    tuple[InSessionTaskDetails, int]
    | WorkRunTurnApplicationResult
):
    """在重绑定前验证一个精确的待处理/中断请求。"""

    def failed() -> WorkRunTurnApplicationResult:
        return _failed(
            "verification_recovery_failed",
            window_revision=request.expected_window_revision,
            work_run_id=request.work_run_id,
            current_attempt_id=request.submitted_attempt_id,
        )

    try:
        window = session_store.get_turn_execution_window(request.session_id)
    except Exception:
        window = None
    if (
        window is None
        or str(window.get("turn_id") or "") != request.turn_id
        or str(window.get("window_state") or "") != "active"
        or int(window.get("state_version") or 0)
        != request.expected_window_revision
    ):
        return _failed(
            "turn_window_stale",
            window_revision=request.expected_window_revision,
            work_run_id=request.work_run_id,
            current_attempt_id=request.submitted_attempt_id,
        )
    if (
        window.get("current_work_run_id") is not None
        or window.get("current_attempt_id") is not None
        or window.get("pending_operation_id") is not None
    ):
        return _failed(
            "existing_nonterminal_work_run",
            window_revision=request.expected_window_revision,
            work_run_id=request.work_run_id,
            current_attempt_id=request.submitted_attempt_id,
        )
    if id_plan.work_run_id != request.work_run_id:
        return failed()
    try:
        details = task_graph_store.get_insession_task_details(
            request.session_id,
            request.subject.task_id,
        )
        linked_task_ids = session_store.list_turn_insession_task_ids(
            request.session_id,
            request.turn_id,
        )
        candidates = work_run_store.list_turn_linked_nonterminal_work_runs(
            session_id=request.session_id,
            turn_id=request.turn_id,
        )
        stored = work_run_store.get_work_run(
            session_id=request.session_id,
            work_run_id=request.work_run_id,
        )
        record = verification_store.get_task_node_verification_record(
            session_id=request.session_id,
            verification_request_id=request.verification_request_id,
        )
    except Exception:
        return failed()
    if details is None:
        return _failed(
            "task_details_unavailable",
            window_revision=request.expected_window_revision,
            work_run_id=request.work_run_id,
            current_attempt_id=request.submitted_attempt_id,
        )
    if details.current_graph_revision != request.subject.graph_revision:
        return _failed(
            "task_authority_stale",
            window_revision=request.expected_window_revision,
            work_run_id=request.work_run_id,
            current_attempt_id=request.submitted_attempt_id,
        )
    if request.subject.task_id not in linked_task_ids:
        return _failed(
            "task_not_linked",
            window_revision=request.expected_window_revision,
            work_run_id=request.work_run_id,
            current_attempt_id=request.submitted_attempt_id,
        )
    verification_request = record.request
    if (
        verification_request.session_id != request.session_id
        or verification_request.work_run_id != request.work_run_id
        or verification_request.subject != request.subject
        or verification_request.submitted_attempt_id
        != request.submitted_attempt_id
        or verification_request.revision
        != request.expected_verification_request_revision
        or verification_request.status
        not in {
            TaskNodeVerificationRequestStatus.PENDING,
            TaskNodeVerificationRequestStatus.INTERRUPTED,
        }
        or verification_request.request_turn_id == request.turn_id
        or record.result is not None
    ):
        return failed()
    if verification_request.status is TaskNodeVerificationRequestStatus.PENDING:
        expected_run_status = WorkRunStatus.ACTIVE
        expected_run_reason = "verification_pending"
        expected_node_status = "active"
    else:
        expected_run_status = WorkRunStatus.INTERRUPTED
        expected_run_reason = "verification_technical_failure"
        expected_node_status = "interrupted"
    exact_candidates = tuple(
        candidate
        for candidate in candidates
        if candidate.work_run_id == request.work_run_id
        and candidate.subject == request.subject
    )
    if (
        len(exact_candidates) != 1
        or exact_candidates[0].status is not expected_run_status
        or exact_candidates[0].reason != expected_run_reason
        or exact_candidates[0].work_run_revision
        != request.expected_work_run_revision
        or exact_candidates[0].current_attempt_id is not None
        or exact_candidates[0].current_verification_request_id
        != request.verification_request_id
        or stored.work_run.subject != request.subject
        or stored.work_run.status is not expected_run_status
        or stored.work_run.reason != expected_run_reason
        or stored.work_run.revision != request.expected_work_run_revision
        or stored.current_attempt_id is not None
        or stored.current_verification_request_id
        != request.verification_request_id
    ):
        return failed()
    nodes = tuple(
        node
        for node in details.nodes
        if node.get("insession_task_node_id") == request.subject.node_id
        and node.get("node_revision") == request.subject.node_revision
    )
    if len(nodes) != 1 or nodes[0].get("status") != expected_node_status:
        return _failed(
            "node_authority_stale",
            window_revision=request.expected_window_revision,
            work_run_id=request.work_run_id,
            current_attempt_id=request.submitted_attempt_id,
        )
    if any(
        child.get("status") != "completed"
        for child in details.nodes
        if child.get("parent_insession_task_node_id") == request.subject.node_id
    ):
        return _failed(
            "node_not_ready",
            window_revision=request.expected_window_revision,
            work_run_id=request.work_run_id,
            current_attempt_id=request.submitted_attempt_id,
        )
    submitted = tuple(
        item
        for item in stored.attempts
        if item.attempt.attempt_id == request.submitted_attempt_id
    )
    if (
        len(submitted) != 1
        or submitted[0].attempt.status is not AttemptStatus.CLOSED
        or submitted[0].action != "submit_output_window"
        or id_plan.attempt_id(submitted[0].attempt.ordinal)
        != request.submitted_attempt_id
    ):
        return failed()
    exposed = tuple(entry.registration.spec for entry in catalog_snapshot.exposed())
    if tuple(item.to_dict() for item in exposed) != tuple(
        item.to_dict() for item in allowed_tools
    ):
        return _failed(
            "catalog_projection_mismatch",
            window_revision=request.expected_window_revision,
            work_run_id=request.work_run_id,
            current_attempt_id=request.submitted_attempt_id,
        )
    if allowed_tools and tool_bridge is None:
        return _failed(
            "tool_bridge_unavailable",
            window_revision=request.expected_window_revision,
            work_run_id=request.work_run_id,
            current_attempt_id=request.submitted_attempt_id,
        )
    return details, submitted[0].attempt.ordinal


def _preflight_active_attempt_resume(
    request: WorkRunTurnResumeRequest,
    *,
    catalog_snapshot: CatalogSnapshot,
    allowed_tools: tuple[ToolSpec, ...],
    id_plan: WorkRunTurnStableIdPlan,
    tool_bridge: AttemptToolBridge | None,
) -> (
    tuple[
        InSessionTaskDetails,
        int,
        Literal[
            "undecided",
            "decided_readonly_tools",
            "closed_readonly_tools",
        ],
    ]
    | WorkRunTurnApplicationResult
):
    """在任何 Store 变更前验证完整恢复权威。"""

    try:
        window = session_store.get_turn_execution_window(request.session_id)
    except Exception:
        window = None
    if (
        window is None
        or str(window.get("turn_id") or "") != request.turn_id
        or str(window.get("window_state") or "") != "active"
        or int(window.get("state_version") or 0) != request.expected_window_revision
    ):
        return _failed(
            "turn_window_stale",
            window_revision=request.expected_window_revision,
            work_run_id=request.work_run_id,
            current_attempt_id=request.attempt_id,
        )
    if (
        window.get("current_work_run_id") is not None
        or window.get("current_attempt_id") is not None
        or window.get("pending_operation_id") is not None
    ):
        return _failed(
            "existing_nonterminal_work_run",
            window_revision=request.expected_window_revision,
            work_run_id=request.work_run_id,
            current_attempt_id=request.attempt_id,
        )
    if id_plan.work_run_id != request.work_run_id:
        return _failed(
            "active_attempt_resume_failed",
            window_revision=request.expected_window_revision,
            work_run_id=request.work_run_id,
            current_attempt_id=request.attempt_id,
        )
    try:
        details = task_graph_store.get_insession_task_details(
            request.session_id,
            request.subject.task_id,
        )
    except Exception:
        details = None
    if details is None:
        return _failed(
            "task_details_unavailable",
            window_revision=request.expected_window_revision,
            work_run_id=request.work_run_id,
            current_attempt_id=request.attempt_id,
        )
    if details.current_graph_revision != request.subject.graph_revision:
        return _failed(
            "task_authority_stale",
            window_revision=request.expected_window_revision,
            work_run_id=request.work_run_id,
            current_attempt_id=request.attempt_id,
        )
    nodes = tuple(
        node
        for node in details.nodes
        if node.get("insession_task_node_id") == request.subject.node_id
        and node.get("node_revision") == request.subject.node_revision
    )
    if len(nodes) != 1 or nodes[0].get("status") != "active":
        return _failed(
            "node_authority_stale",
            window_revision=request.expected_window_revision,
            work_run_id=request.work_run_id,
            current_attempt_id=request.attempt_id,
        )
    if any(
        child.get("status") != "completed"
        for child in details.nodes
        if child.get("parent_insession_task_node_id") == request.subject.node_id
    ):
        return _failed(
            "node_not_ready",
            window_revision=request.expected_window_revision,
            work_run_id=request.work_run_id,
            current_attempt_id=request.attempt_id,
        )
    try:
        linked_task_ids = session_store.list_turn_insession_task_ids(
            request.session_id,
            request.turn_id,
        )
        candidates = work_run_store.list_turn_linked_nonterminal_work_runs(
            session_id=request.session_id,
            turn_id=request.turn_id,
        )
    except Exception:
        linked_task_ids = ()
        candidates = ()
    if request.subject.task_id not in linked_task_ids:
        return _failed(
            "task_not_linked",
            window_revision=request.expected_window_revision,
            work_run_id=request.work_run_id,
            current_attempt_id=request.attempt_id,
        )
    exact_candidates = tuple(
        candidate
        for candidate in candidates
        if candidate.work_run_id == request.work_run_id
        and candidate.subject == request.subject
    )
    if (
        len(exact_candidates) != 1
        or exact_candidates[0].status is not WorkRunStatus.ACTIVE
        or exact_candidates[0].reason is not None
        or exact_candidates[0].current_attempt_id
        not in {None, request.attempt_id}
        or exact_candidates[0].current_verification_request_id is not None
        or exact_candidates[0].work_run_revision
        != request.expected_work_run_revision
    ):
        return _failed(
            "active_attempt_resume_failed",
            window_revision=request.expected_window_revision,
            work_run_id=request.work_run_id,
            current_attempt_id=request.attempt_id,
        )
    try:
        stored = work_run_store.get_work_run(
            session_id=request.session_id,
            work_run_id=request.work_run_id,
        )
    except Exception:
        stored = None
    if (
        stored is None
        or stored.work_run.subject != request.subject
        or stored.work_run.status is not WorkRunStatus.ACTIVE
        or stored.work_run.reason is not None
        or stored.current_attempt_id not in {None, request.attempt_id}
        or stored.current_verification_request_id is not None
        or stored.work_run.budget.active_seconds_consumed
        >= stored.work_run.budget.soft_active_seconds
    ):
        return _failed(
            "active_attempt_resume_failed",
            window_revision=request.expected_window_revision,
            work_run_id=request.work_run_id,
            current_attempt_id=request.attempt_id,
        )
    current_attempts = tuple(
        item
        for item in stored.attempts
        if item.attempt.attempt_id == request.attempt_id
    )
    if len(current_attempts) != 1:
        return _failed(
            "active_attempt_resume_failed",
            window_revision=request.expected_window_revision,
            work_run_id=request.work_run_id,
            current_attempt_id=request.attempt_id,
        )
    current = current_attempts[0]
    if (
        current.turn_id == request.turn_id
        or current.catalog_snapshot != catalog_snapshot.to_descriptor()
        or id_plan.attempt_id(current.attempt.ordinal) != request.attempt_id
    ):
        return _failed(
            "active_attempt_resume_failed",
            window_revision=request.expected_window_revision,
            work_run_id=request.work_run_id,
            current_attempt_id=request.attempt_id,
        )
    current_calls = tuple(
        item for item in stored.tool_calls if item.attempt_id == request.attempt_id
    )
    current_results = tuple(
        item for item in stored.tool_results if item.attempt_id == request.attempt_id
    )
    supports_protected_recovery = bridge_supports_protected_recovery(tool_bridge)
    if (
        stored.current_attempt_id == request.attempt_id
        and current.attempt.status is AttemptStatus.ACTIVE
        and current.action is None
        and current.decision is None
    ):
        if current_calls or current_results:
            return _failed(
                "active_attempt_resume_failed",
                window_revision=request.expected_window_revision,
                work_run_id=request.work_run_id,
                current_attempt_id=request.attempt_id,
            )
        resume_kind: Literal[
            "undecided",
            "decided_readonly_tools",
            "closed_readonly_tools",
        ] = "undecided"
    elif (
        stored.current_attempt_id == request.attempt_id
        and current.attempt.status is AttemptStatus.ACTIVE
        and current.action == "call_tools"
        and current.decision is not None
        and isinstance(
            current.decision.action,
            HostMaterializedCallToolsAction,
        )
        and len(current_calls) == len(current.decision.action.calls)
        and len(current_results) <= len(current_calls)
        and (
            supports_protected_recovery
            or (
                not any(call.call.modifies_environment for call in current_calls)
                and not any(
                    call.modifies_environment
                    for call in current.decision.action.calls
                )
                and not any(
                    item.status is ToolResultStatus.COMPLETION_UNCONFIRMED
                    for item in current_results
                )
            )
        )
        and tuple(call.call.tool_call_id for call in current_calls)
        == tuple(
            id_plan.tool_call_id(request.attempt_id, ordinal)
            for ordinal in range(1, len(current_calls) + 1)
        )
    ):
        resume_kind = "decided_readonly_tools"
    elif (
        stored.current_attempt_id is None
        and current is stored.attempts[-1]
        and current.attempt.status is AttemptStatus.CLOSED
        and current.action == "call_tools"
        and current.decision is not None
        and isinstance(
            current.decision.action,
            HostMaterializedCallToolsAction,
        )
        and len(current_calls) == len(current.decision.action.calls)
        and len(current_results) == len(current_calls)
        and (
            supports_protected_recovery
            or (
                not any(call.call.modifies_environment for call in current_calls)
                and not any(
                    call.modifies_environment
                    for call in current.decision.action.calls
                )
                and not any(
                    item.status is ToolResultStatus.COMPLETION_UNCONFIRMED
                    for item in current_results
                )
            )
        )
        and tuple(call.call.tool_call_id for call in current_calls)
        == tuple(
            id_plan.tool_call_id(request.attempt_id, ordinal)
            for ordinal in range(1, len(current_calls) + 1)
        )
    ):
        resume_kind = "closed_readonly_tools"
    else:
        return _failed(
            "active_attempt_resume_failed",
            window_revision=request.expected_window_revision,
            work_run_id=request.work_run_id,
            current_attempt_id=request.attempt_id,
        )
    exposed = tuple(entry.registration.spec for entry in catalog_snapshot.exposed())
    if tuple(item.to_dict() for item in exposed) != tuple(
        item.to_dict() for item in allowed_tools
    ):
        return _failed(
            "catalog_projection_mismatch",
            window_revision=request.expected_window_revision,
            work_run_id=request.work_run_id,
            current_attempt_id=request.attempt_id,
        )
    if allowed_tools and tool_bridge is None:
        return _failed(
            "tool_bridge_unavailable",
            window_revision=request.expected_window_revision,
            work_run_id=request.work_run_id,
            current_attempt_id=request.attempt_id,
        )
    return details, current.attempt.ordinal, resume_kind


def _preflight_waiting_user_continuation(
    request: WorkRunTurnWaitingUserContinuationRequest,
    *,
    catalog_snapshot: CatalogSnapshot,
    allowed_tools: tuple[ToolSpec, ...],
    id_plan: WorkRunTurnStableIdPlan,
    tool_bridge: AttemptToolBridge | None,
) -> InSessionTaskDetails | WorkRunTurnApplicationResult:
    """在原子消费前重新读取精确的待处理交互。"""

    question = request.pending_question
    try:
        window = session_store.get_turn_execution_window(request.session_id)
    except Exception:
        window = None
    if (
        window is None
        or str(window.get("turn_id") or "") != request.turn_id
        or str(window.get("window_state") or "") != "active"
        or int(window.get("state_version") or 0) != request.expected_window_revision
    ):
        return _failed(
            "turn_window_stale",
            window_revision=request.expected_window_revision,
            work_run_id=question.work_run_id,
        )
    if (
        window.get("current_work_run_id") is not None
        or window.get("current_attempt_id") is not None
        or window.get("pending_operation_id") is not None
    ):
        return _failed(
            "existing_nonterminal_work_run",
            window_revision=request.expected_window_revision,
            work_run_id=question.work_run_id,
        )
    if id_plan.work_run_id != question.work_run_id:
        return _failed(
            "waiting_user_continuation_failed",
            window_revision=request.expected_window_revision,
            work_run_id=question.work_run_id,
        )

    try:
        pending = tuple(
            item
            for item in continuation_store.list_pending_user_questions(
                session_id=request.session_id
            )
            if item.work_run_id == question.work_run_id
        )
    except Exception:
        pending = ()
    if len(pending) != 1 or pending[0] != question:
        return _failed(
            "waiting_user_continuation_failed",
            window_revision=request.expected_window_revision,
            work_run_id=question.work_run_id,
        )

    try:
        details = task_graph_store.get_insession_task_details(
            request.session_id,
            question.subject.task_id,
        )
    except Exception:
        details = None
    if details is None:
        return _failed(
            "task_details_unavailable",
            window_revision=request.expected_window_revision,
            work_run_id=question.work_run_id,
        )
    if (
        details.task_state_version != question.task_state_version
        or details.current_graph_revision != question.subject.graph_revision
    ):
        return _failed(
            "task_authority_stale",
            window_revision=request.expected_window_revision,
            work_run_id=question.work_run_id,
        )
    nodes = tuple(
        node
        for node in details.nodes
        if node.get("insession_task_node_id") == question.subject.node_id
        and node.get("node_revision") == question.subject.node_revision
    )
    if (
        len(nodes) != 1
        or nodes[0].get("state_version") != question.node_state_version
        or nodes[0].get("status") != "awaiting_user"
    ):
        return _failed(
            "node_authority_stale",
            window_revision=request.expected_window_revision,
            work_run_id=question.work_run_id,
        )
    if any(
        child.get("status") != "completed"
        for child in details.nodes
        if child.get("parent_insession_task_node_id") == question.subject.node_id
    ):
        return _failed(
            "node_not_ready",
            window_revision=request.expected_window_revision,
            work_run_id=question.work_run_id,
        )
    try:
        linked_task_ids = session_store.list_turn_insession_task_ids(
            request.session_id,
            request.turn_id,
        )
    except Exception:
        linked_task_ids = ()
    if question.subject.task_id not in linked_task_ids:
        return _failed(
            "task_not_linked",
            window_revision=request.expected_window_revision,
            work_run_id=question.work_run_id,
        )
    exposed = tuple(entry.registration.spec for entry in catalog_snapshot.exposed())
    if tuple(item.to_dict() for item in exposed) != tuple(
        item.to_dict() for item in allowed_tools
    ):
        return _failed(
            "catalog_projection_mismatch",
            window_revision=request.expected_window_revision,
            work_run_id=question.work_run_id,
        )
    if allowed_tools and tool_bridge is None:
        return _failed(
            "tool_bridge_unavailable",
            window_revision=request.expected_window_revision,
            work_run_id=question.work_run_id,
        )
    return details


def _start_attempt(
    *,
    request: WorkRunTurnRequest,
    prior: WorkExecutionMutationResult,
    ordinal: int,
    catalog_descriptor: dict[str, object],
    id_plan: WorkRunTurnStableIdPlan,
) -> WorkExecutionMutationResult:
    mutation = work_run_store.start_work_run_attempt(
        session_id=request.session_id,
        turn_id=request.turn_id,
        work_run_id=prior.work_run_id,
        expected_work_run_revision=prior.work_run_revision,
        expected_progress_revision=prior.acceptance_progress_revision,
        expected_window_revision=prior.window_state_version,
        apply_id=id_plan.start_attempt_apply_id(ordinal),
        catalog_snapshot=catalog_descriptor,
        attempt_id=id_plan.attempt_id(ordinal),
    )
    if (
        mutation.current_attempt_id != id_plan.attempt_id(ordinal)
        or mutation.attempt is None
        or mutation.attempt.ordinal != ordinal
    ):
        raise RuntimeError("Store started an unexpected Attempt")
    return mutation


def _build_attempt_context(
    *,
    request: WorkRunTurnRequest,
    work_run_id: str,
    allowed_tools: tuple[ToolSpec, ...],
) -> AttemptDecisionContext:
    details = task_graph_store.get_insession_task_details(
        request.session_id,
        request.subject.task_id,
    )
    if details is None:
        raise TaskNodeSourceContextAuthorityError("task_authority_mismatch")
    stored = work_run_store.get_work_run(
        session_id=request.session_id,
        work_run_id=work_run_id,
    )
    if stored.current_attempt_id is None:
        raise RuntimeError("WorkRun has no current Attempt")
    matching_attempts = tuple(
        item
        for item in stored.attempts
        if item.attempt.attempt_id == stored.current_attempt_id
    )
    if len(matching_attempts) != 1:
        raise RuntimeError("current Attempt is not unique")
    current = matching_attempts[0]
    if current.attempt.status is not AttemptStatus.ACTIVE:
        raise RuntimeError("current Attempt is not active")
    nodes = tuple(
        node
        for node in details.nodes
        if node.get("insession_task_node_id") == request.subject.node_id
        and node.get("node_revision") == request.subject.node_revision
    )
    if len(nodes) != 1:
        raise RuntimeError("TaskNode semantics are unavailable")
    node = nodes[0]
    acceptances = tuple(
        InSessionTaskAcceptanceProposal.model_validate(item)
        for item in node.get("acceptance_criteria", ())
    )
    verification_feedback = (
        AttemptVerificationFeedback(
            submitted_output_revision=current.input_verification_result.output_revision,
            acceptance_results=current.input_verification_result.acceptance_results,
            downstream_results=current.input_verification_result.downstream_results,
        )
        if current.input_verification_result is not None
        else None
    )
    dependency_deliveries = _load_attempt_dependency_deliveries(
        session_id=request.session_id,
        subject=request.subject,
        details=details,
    )
    return AttemptDecisionContext(
        session_id=request.session_id,
        turn_id=request.turn_id,
        work_run_id=stored.work_run.work_run_id,
        work_run_revision=stored.work_run.revision,
        attempt_id=current.attempt.attempt_id,
        attempt_ordinal=current.attempt.ordinal,
        user_input=project_authoritative_attempt_user_input(
            stored=stored,
            current_attempt=current,
        ),
        subject=request.subject,
        node_title=str(node["title"]),
        node_objective=str(node["objective"]),
        acceptances=acceptances,
        acceptance_progress=stored.acceptance_progress,
        output_window=stored.output_window,
        dependency_deliveries=dependency_deliveries,
        source_context=build_task_node_source_context(
            session_id=request.session_id,
            details=details,
            subject=request.subject,
        ),
        prior_tool_results=_prior_tool_results(
            stored,
            current.attempt.ordinal,
            limits=request.attempt_input_limits,
        ),
        execution_findings=project_work_run_execution_findings_for_tools(
            session_id=request.session_id,
            work_run_id=stored.work_run.work_run_id,
            acceptance_ids=tuple(item.acceptance_id for item in acceptances),
            allowed_tools=allowed_tools,
        ),
        input_limits=request.attempt_input_limits,
        allowed_tools=allowed_tools,
        allow_user_input=request.allow_user_input,
        verification_feedback=verification_feedback,
        paper_resources=request.paper_resources,
    )


def _attempt_model_call_plan(
    *,
    request: WorkRunTurnRequest,
    context: AttemptDecisionContext,
    allowed_tools: tuple[ToolSpec, ...],
    id_plan: WorkRunTurnStableIdPlan,
    factory: TaskNodeModelCallAuthorityFactory | None,
) -> TaskNodeModelCallPlan | None:
    if factory is None:
        return None
    stored = work_run_store.get_work_run(
        session_id=request.session_id,
        work_run_id=context.work_run_id,
    )
    attempts = tuple(
        item
        for item in stored.attempts
        if item.attempt.attempt_id == context.attempt_id
    )
    if len(attempts) != 1 or attempts[0].turn_id != context.turn_id:
        raise RuntimeError("TaskNode Attempt model origin is unavailable")
    origin_turn_id = attempts[0].input_turn_id
    return TaskNodeModelCallPlan(
        logical_call_id=id_plan.attempt_model_call_id(context.attempt_ordinal),
        request_turn_id=origin_turn_id,
        authority_factory=factory,
        rederive_state_guard_sha256=lambda: (
            task_node_attempt_state_guard_sha256(
                _build_attempt_context(
                    request=request,
                    work_run_id=context.work_run_id,
                    allowed_tools=allowed_tools,
                )
            )
        ),
    )


def _load_attempt_dependency_deliveries(
    *,
    session_id: str,
    subject: TaskNodeSubject,
    details: InSessionTaskDetails,
) -> TaskNodeDependencyDeliveries:
    """解析当前直接子节点正文，但不持久化副本。

    Store 会重新验证当前 Task/图/节点权威，并在将每个不可变来源正文重绑定到当前树之前，
    将其与直接所有权或一张经认证的沿用收据一同解析。
    """

    resolved = work_run_store.get_current_task_node_dependency_deliveries(
        session_id=session_id,
        subject=subject,
    )
    # 旧版单节点图早于 ``build_task_node_tree`` 所用的规范 ``root node_id == task_id`` 不变量。
    # 经 Store 授权的空依赖集无需重建图，并继续兼容这些持久 TaskNode 记录。
    if not resolved:
        return TaskNodeDependencyDeliveries()
    projection = project_task_node_dependencies(
        build_task_node_tree(details),
        parent_subject=subject,
        resolved_deliveries=resolved,
    )
    delivery_ids = tuple(item.delivery_id for item in resolved)
    if projection.delivery_ids != delivery_ids:
        raise RuntimeError(
            "Attempt dependency Delivery order crossed current frontier authority"
        )
    return projection.to_model_deliveries()


def _prior_tool_results(
    stored: StoredWorkRun,
    current_attempt_ordinal: int,
    *,
    limits: AttemptDecisionInputLimits,
) -> PriorToolResultsProjection:
    attempts = {item.attempt.attempt_id: item.attempt for item in stored.attempts}
    calls = {item.call.tool_call_id: item for item in stored.tool_calls}
    projected: list[PriorToolResultProjection] = []
    for result in stored.tool_results:
        attempt = attempts.get(result.attempt_id)
        call = calls.get(result.tool_call_id)
        if (
            attempt is None
            or call is None
            or call.attempt_id != result.attempt_id
            or attempt.status is not AttemptStatus.CLOSED
            or attempt.ordinal >= current_attempt_ordinal
        ):
            continue
        projected.append(
            PriorToolResultProjection(
                tool_id=call.call.tool_id,
                tool_version=call.call.tool_version,
                result=result,
            )
        )
    supporting_ids = {
        result_id
        for item in stored.acceptance_progress.items
        for result_id in item.supporting_tool_result_ids
    }
    required_ids = mandatory_prior_tool_result_ids(
        tuple(projected),
        supporting_result_ids=supporting_ids,
    )
    return select_bounded_prior_tool_results(
        tuple(projected),
        required_result_ids=required_ids,
        limits=limits,
    )


def _settlement_stop_result(
    *,
    request: WorkRunTurnRequest,
    action: str,
    mutation: WorkExecutionMutationResult,
) -> WorkRunTurnApplicationResult | None:
    if action == "request_task_graph_revision":
        if (
            mutation.work_run_status is not WorkRunStatus.CANCELLED
            or mutation.work_run_reason != "task_graph_revision_requested"
        ):
            raise RuntimeError(
                "TaskGraph revision request produced an invalid WorkRun settlement"
            )
        authority = work_run_store.get_active_task_graph_execution_replan_request(
            session_id=request.session_id,
            task_id=request.subject.task_id,
        )
        if (
            authority is None
            or authority.work_run_id != mutation.work_run_id
            or authority.requesting_subject != request.subject
        ):
            raise RuntimeError(
                "TaskGraph revision request has no durable active authority"
            )
        return WorkRunTurnApplicationResult(
            outcome="task_graph_revision_requested",
            work_run_id=mutation.work_run_id,
            window_revision=mutation.window_state_version,
        )
    if mutation.work_run_status is WorkRunStatus.FAILED:
        if mutation.work_run_reason != "work_run_limit_reached":
            raise RuntimeError("Attempt settlement produced an unsupported WorkRun failure")
        return WorkRunTurnApplicationResult(
            outcome="work_run_failed",
            work_run_id=mutation.work_run_id,
            window_revision=mutation.window_state_version,
        )
    if mutation.work_run_status is WorkRunStatus.TURN_LIMIT_REACHED:
        if mutation.work_run_reason != "turn_limit_reached":
            raise RuntimeError("Attempt settlement produced an invalid soft-limit state")
        return WorkRunTurnApplicationResult(
            outcome="turn_limit_reached",
            work_run_id=mutation.work_run_id,
            window_revision=mutation.window_state_version,
        )
    if mutation.work_run_status is WorkRunStatus.WAITING_EXTERNAL:
        return WorkRunTurnApplicationResult(
            outcome="waiting_external",
            work_run_id=mutation.work_run_id,
            window_revision=mutation.window_state_version,
        )
    if action != "request_user_input":
        return None
    stored = work_run_store.get_work_run(
        session_id=request.session_id,
        work_run_id=mutation.work_run_id,
    )
    if (
        stored.work_run.status is not WorkRunStatus.WAITING_USER
        or stored.pending_user_question is None
    ):
        raise RuntimeError("request_user mutation has no durable waiting projection")
    return WorkRunTurnApplicationResult(
        outcome="waiting_user",
        work_run_id=mutation.work_run_id,
        pending_user_question=stored.pending_user_question,
        window_revision=mutation.window_state_version,
    )


__all__ = [
    'WorkRunTurnApplicationRequest',
    'WorkRunTurnApplicationResult',
    "WorkRunTurnAuthorityUnavailable",
    'WorkRunTurnFailureCode',
    'WorkRunTurnOutcome',
    'WorkRunTurnResumeRequest',
    'WorkRunTurnStableIdPlan',
    'WorkRunTurnUnpreparedVerificationRecoveryRequest',
    'WorkRunTurnVerificationRecoveryRequest',
    'WorkRunTurnWaitingUserContinuationRequest',
    "continue_waiting_user_task_node_work_run",
    "recover_unprepared_task_node_work_run_verification",
    "recover_task_node_work_run_verification",
    "resume_active_task_node_work_run",
    "run_new_task_node_work_run",
]
