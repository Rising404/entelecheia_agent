"""基于 AuxiliaryGraph 的有界生产应用程序根节点。

底层 控制器会刻意停在具名断点处。本模块在这些断点之上提供最小且完整的应用组合：初始规划、确定性前沿选择、有界 Host 原语、模型 WorkRun，以及终端审查、密封和 TaskGraph 提交；同时支持空基础创建和正基础上的 ``N -> N+1`` 修订。

每项效果都从新加载的 Store 前沿中选择。根组件执行的变更绝不超过 ``max_effect_steps`` 次，不会臆造 Runtime 身份，并始终注入生产环境的 WorkRun 模型调用权威状态。因此，响应丢失后的重试会重新进入控制器已有的预留、WorkRun、模型账本调用、语义回执和提交 ID，而不会分发第二项逻辑效果。

类型化的 ``user_gate`` 节点使用与模型调查节点相同的持久化 WorkRun/Attempt 控制器。它们受限的执行器契约保留了一个精确的问题，只接受 Guard/Store-授权的答案 Turn，并在终端规划恢复前冻结 Host-材料化的用户响应证据。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from functools import partial
import time

from personagraph.l2.auxiliary_graph import AuxiliaryNodeExecutorKind
from personagraph.l2.task_graph import (
    InSessionTaskGraphRevisionValidationContext,
)
from personagraph.model_io.gateway import ModelGatewayError, ModelResult
from personagraph.session.l2_store import auxiliary_graph as auxiliary_graph_store
from personagraph.session.l2_store import continuation as continuation_store
from personagraph.session.l2_store import planning as planning_store
from personagraph.session.l2_store import semantic_verification as semantic_store
from personagraph.session.l2_store import task_delivery as task_delivery_store
from personagraph.session.l2_store import task_graph as task_graph_store
from personagraph.session.l2_store import terminal as terminal_store
from personagraph.session.l2_store import work_run as work_run_store
from personagraph.l2.work_run import AuxiliaryNodeSubject
from personagraph.l2.auxiliary_execution.planning.architect import (
    AuxiliaryGraphArchitectGuardError,
    AuxiliaryGraphArchitectInputTooLarge,
    AuxiliaryGraphArchitectStructuredProvider,
)
from personagraph.l2.auxiliary_execution.driver import (
    AuxiliaryGraphDriverAction,
    canonical_auxiliary_graph_driver_state_guard,
    decide_auxiliary_graph_driver_step,
)
from personagraph.l2.auxiliary_execution.planning.mounted_document_authority import (
    FrozenMountedDocumentPlanningAuthority,
    MountedDocumentPlanningAuthorityError,
    build_mounted_document_authority_projection,
    build_mounted_resource_perception_request,
    build_mounted_resource_read_port,
    freeze_mounted_document_planning_authority,
)
from personagraph.l2.auxiliary_execution.planning.task_document_scope import (
    AuxiliaryTaskDocumentScopeError,
    AuxiliaryTaskDocumentScope,
    prepare_auxiliary_task_document_scope,
)
from personagraph.l2.auxiliary_execution.planning.host_primitive_controller import (
    AuxiliaryHostPrimitiveControllerError,
    AuxiliaryHostPrimitiveControllerRequest,
    AuxiliaryHostPrimitiveControllerResult,
    AuxiliaryHostPrimitiveControllerStatus,
    AuxiliaryHostPrimitiveRequestFactory,
    run_auxiliary_host_primitive,
)
from personagraph.l2.auxiliary_execution.planning.goal_successor_controller import (
    AuxiliaryGoalSuccessorPlanningError,
    AuxiliaryGoalSuccessorPlanningRequest,
    AuxiliaryGoalSuccessorPlanningResult,
    AuxiliaryGoalSuccessorPlanningStatus,
    run_auxiliary_goal_successor_planning,
)
from personagraph.l2.auxiliary_execution.planning.goal_supersede_controller import (
    AuxiliaryGoalSupersedeControllerError,
    AuxiliaryGoalSupersedeRequest,
    run_auxiliary_goal_supersede,
)
from personagraph.l2.auxiliary_execution.adapters.model_authority import (
    create_auxiliary_semantic_reviewer_model_call_authority,
    create_auxiliary_work_run_model_call_authority,
)
from personagraph.l2.auxiliary_execution.planning.controller import (
    AuxiliaryInitialPlanningError,
    AuxiliaryInitialPlanningResult,
    AuxiliaryInitialPlanningStatus,
    run_initial_auxiliary_planning,
)
from personagraph.l2.auxiliary_execution.planning.profiles import (
    MOUNTED_DOCUMENT_READ_CAPABILITY,
    MOUNTED_VISUAL_READ_CAPABILITY,
    WORKSPACE_READONLY_CAPABILITY,
    build_auxiliary_execution_capability_catalogs,
)
from personagraph.l2.auxiliary_execution.planning.positive_controller import (
    AuxiliaryPositivePlanningError,
    AuxiliaryPositivePlanningRequest,
    AuxiliaryPositivePlanningResult,
    AuxiliaryPositivePlanningStatus,
    run_positive_base_auxiliary_planning,
)
from personagraph.l2.auxiliary_execution.planning.replanning_controller import (
    AuxiliaryReplanningError,
    AuxiliaryReplanningRequest,
    AuxiliaryReplanningResult,
    AuxiliaryReplanningStatus,
    run_auxiliary_replanning,
)
from personagraph.l2.auxiliary_execution.verification.controller import (
    AuxiliarySemanticModelCallAuthorityFactory,
)
from personagraph.l2.auxiliary_execution.terminal.composition import (
    AuxiliaryTerminalCandidateSemanticRouteRequired,
    AuxiliaryTerminalCompositionError,
    AuxiliaryTerminalCompositionResult,
    AuxiliaryTerminalCompositionStatus,
    run_auxiliary_terminal_candidate_semantic_gate,
    run_auxiliary_terminal_composition,
)
from personagraph.tools.workspace.session_read_source import (
    build_session_workspace_readonly_runtime,
)
from personagraph.tools.documents.mounted_document_cognition_tools import (
    MOUNTED_DOCUMENT_COGNITION_CAPABILITY,
)
from personagraph.l2.task_execution.tool_bridge.mounted_document_adapter import (
    build_session_mounted_document_cognition_runtime,
)
from personagraph.l2.task_execution.tool_bridge.workspace_readonly_adapter import (
    build_workspace_readonly_work_run_bridge,
)
from personagraph.l2.auxiliary_execution.work_run.contracts import (
    AuxiliaryWorkRunStatus,
    AuxiliaryWorkRunRequest,
    AuxiliaryWorkRunResult,
    derive_auxiliary_work_run_ids,
)
from personagraph.l2.auxiliary_execution.work_run.profile import (
    AuxiliaryWorkRunProfile,
)
from personagraph.l2.auxiliary_execution.work_run.controller import (
    CapabilityCatalogs,
    CapabilityToolBridges,
    run_auxiliary_model_node,
)
from personagraph.l2.auxiliary_execution.work_run.contracts import (
    AuxiliaryNodeToolRuntimeFactory,
)
from .planning.mounted_visual_resource import (
    MountedVisualPlanningAuthorityError,
)
from personagraph.runtime.model_calls.contracts import (
    DurableModelCallTerminalState,
    RuntimeModelLedgerStore,
)
from personagraph.l2.model_output_budgets import L2_MAX_OUTPUT_TOKENS
from personagraph.runtime.turn_deadline import TurnDeadline, TurnDeadlineExceeded
from personagraph.l2.task_execution.verification.decision import NodeVerificationStructuredProvider
from personagraph.l2.planning.invocation_contracts import (
    PlanningContextPrimitiveKind,
)
from personagraph.l2.planning.resource_perception import PlanningResourceReadPort
from personagraph.runtime.model_calls.authority import (
    RuntimeModelCallWaitingExternal,
)
from personagraph.runtime.tool_calls import (
    RuntimeToolLedgerStore,
)
from personagraph.l2.auxiliary_execution.verification.task_graph_semantic import (
    TaskGraphSemanticVerificationStructuredProvider,
)
from personagraph.runtime.turn_events import TurnEvent
from personagraph.l2.task_execution.work_run.model_providers import (
    WorkRunStructuredModelProfile,
    build_attempt_structured_provider,
    build_verification_structured_provider,
)


class AuxiliaryApplicationStatus(StrEnum):
    """单个有界应用程序运行的外部有意义边界。"""

    COMMITTED = "committed"
    WAITING_USER = "waiting_user"
    WAITING_AUTHORIZATION = "waiting_authorization"
    WAITING_EXTERNAL = "waiting_external"
    TURN_LIMIT_REACHED = "turn_limit_reached"
    STEP_LIMIT_REACHED = "step_limit_reached"
    REVISION_REQUIRED = "revision_required"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class AuxiliaryApplicationRequest:
    session_id: str
    turn_id: str
    task_id: str
    desired_output: str = "完整、可执行、可验证的 TaskGraph"
    max_effect_steps: int = 64
    max_autonomous_replans: int = 3
    deadline: TurnDeadline | None = None

    def __post_init__(self) -> None:
        for name in ("session_id", "turn_id", "task_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value or len(value) > 200:
                raise ValueError(f"{name} must be a 1..200 character identity")
        if (
            not isinstance(self.desired_output, str)
            or not self.desired_output.strip()
        ):
            raise ValueError("desired_output must be non-empty")
        if (
            isinstance(self.max_effect_steps, bool)
            or not isinstance(self.max_effect_steps, int)
            or not 1 <= self.max_effect_steps <= 128
        ):
            raise ValueError("max_effect_steps must be within 1..128")
        if (
            isinstance(self.max_autonomous_replans, bool)
            or not isinstance(self.max_autonomous_replans, int)
            or not 1 <= self.max_autonomous_replans <= 12
        ):
            raise ValueError("max_autonomous_replans must be within 1..12")


@dataclass(frozen=True, slots=True, kw_only=True)
class AuxiliaryApplicationPorts:
    """由生产控制器使用的应用程序拥有的物理端口。"""

    emit: Callable[[TurnEvent], object]
    model_ledger_store: RuntimeModelLedgerStore
    allow_user_input: bool = True
    monotonic_clock: Callable[[], float] = time.monotonic
    planning_provider: AuxiliaryGraphArchitectStructuredProvider | None = None
    task_document_scope: AuxiliaryTaskDocumentScope | None = None
    knowledge_cognition_history_enabled: bool = False
    attempt_provider: Callable[..., ModelResult] | None = None
    verification_provider: NodeVerificationStructuredProvider | None = None
    semantic_provider: TaskGraphSemanticVerificationStructuredProvider | None = None
    work_run_model_profile: WorkRunStructuredModelProfile = field(
        default_factory=lambda: WorkRunStructuredModelProfile(
            attempt_max_output_tokens=L2_MAX_OUTPUT_TOKENS,
            verification_max_output_tokens=L2_MAX_OUTPUT_TOKENS,
            timeout_s=600.0,
        )
    )
    work_run_runtime_profile: AuxiliaryWorkRunProfile = field(
        default_factory=AuxiliaryWorkRunProfile
    )
    capability_catalogs: CapabilityCatalogs | None = None
    capability_tool_bridges: CapabilityToolBridges | None = None
    tool_ledger_store: RuntimeToolLedgerStore | None = None
    node_retrieval_runtime_builder: (
        Callable[
            [FrozenMountedDocumentPlanningAuthority],
            AuxiliaryNodeToolRuntimeFactory | None,
        ]
        | None
    ) = None
    host_request_factory: AuxiliaryHostPrimitiveRequestFactory | None = None
    primitive_kinds_by_capability_profile: Mapping[
        str, PlanningContextPrimitiveKind
    ] | None = None
    resource_read_port: PlanningResourceReadPort | None = None
    semantic_model_call_authority_factory: (
        AuxiliarySemanticModelCallAuthorityFactory | None
    ) = None

    def __post_init__(self) -> None:
        if not callable(self.emit):
            raise TypeError("emit must be callable")
        for name in (
            "reserve_runtime_model_logical_call",
            "append_runtime_model_physical_attempt",
            "settle_runtime_model_physical_attempt",
            "get_runtime_model_logical_call",
            "get_runtime_model_rejected_output",
        ):
            if not callable(getattr(self.model_ledger_store, name, None)):
                raise TypeError(
                    f"model_ledger_store.{name} must be callable"
                )
        if type(self.allow_user_input) is not bool:
            raise TypeError("allow_user_input must be a boolean")
        if not callable(self.monotonic_clock):
            raise TypeError("monotonic_clock must be callable")
        if not isinstance(self.knowledge_cognition_history_enabled, bool):
            raise TypeError("knowledge history switch must be boolean")


@dataclass(frozen=True, slots=True)
class AuxiliaryApplicationResult:
    status: AuxiliaryApplicationStatus
    reason_code: str
    effect_steps: int
    last_driver_action: AuxiliaryGraphDriverAction | None = None
    subject: AuxiliaryNodeSubject | None = None
    requested_user_question: str | None = None
    planning_result: AuxiliaryPlanningResult | None = None
    host_result: AuxiliaryHostPrimitiveControllerResult | None = None
    work_run_result: AuxiliaryWorkRunResult | None = None
    terminal_result: AuxiliaryTerminalCompositionResult | None = None
    replan_result: AuxiliaryReplanningResult | None = None

    def __post_init__(self) -> None:
        if not self.reason_code or len(self.reason_code) > 240:
            raise ValueError("application result requires a bounded reason code")
        if not 0 <= self.effect_steps <= 128:
            raise ValueError("application effect step count is invalid")


class AuxiliaryApplicationError(RuntimeError):
    """应用程序根目录无法保留当前 Auxiliary 权威状态。"""

    code = "auxiliary_v2_application_rejected"


AuxiliaryPlanningResult = (
    AuxiliaryInitialPlanningResult
    | AuxiliaryPositivePlanningResult
    | AuxiliaryGoalSuccessorPlanningResult
)


def run_auxiliary_application_to_boundary(
    request: AuxiliaryApplicationRequest,
    *,
    ports: AuxiliaryApplicationPorts,
) -> AuxiliaryApplicationResult:
    """驱动一个 Auxiliary 任务直到遇到外部边界或效果步限。

    首次成功的初始或正基计划替换被视为一个效果步。一个活跃的 TaskGraph 修订触发总是优先于历史初始规划完成。一个已验证的计划是一个只读的恢复预检，不会。每次后续控制器结果之后都会进行完整的前沿重新加载，然后再选择另一个效果。
    """

    if not isinstance(request, AuxiliaryApplicationRequest):
        raise TypeError("request must be AuxiliaryApplicationRequest")
    if not isinstance(ports, AuxiliaryApplicationPorts):
        raise TypeError("ports must be AuxiliaryApplicationPorts")

    model_ledger_store = ports.model_ledger_store
    work_run_model_call_authority_factory = partial(
        create_auxiliary_work_run_model_call_authority,
        ledger_store=model_ledger_store,
    )
    semantic_model_call_authority_factory = (
        ports.semantic_model_call_authority_factory
        or partial(
            create_auxiliary_semantic_reviewer_model_call_authority,
            ledger_store=model_ledger_store,
        )
    )

    effect_steps = 0
    planning_result: AuxiliaryPlanningResult | None = None
    replan_result: AuxiliaryReplanningResult | None = None
    task_graph_revision_authority = None
    pending_supersede_receipt = None
    target_change_intent = None
    allowed_managed_document_ids: tuple[str, ...] = ()
    try:
        document_scope = ports.task_document_scope
        if document_scope is None:
            document_scope = prepare_auxiliary_task_document_scope(
                session_id=request.session_id,
                turn_id=request.turn_id,
                task_id=request.task_id,
            )
        elif (
            document_scope.session_id != request.session_id
            or document_scope.turn_id != request.turn_id
            or document_scope.task_id != request.task_id
        ):
            raise AuxiliaryTaskDocumentScopeError(
                "attachment_authority_scope_crossed_task"
            )
        target_change_intent = document_scope.target_change_intent
        current_lane = document_scope.current_lane
        allowed_managed_document_ids = (
            document_scope.allowed_managed_document_ids
        )
    except AuxiliaryTaskDocumentScopeError as exc:
        return _result(
            status=AuxiliaryApplicationStatus.FAILED,
            reason_code=exc.code,
            effect_steps=effect_steps,
        )
    except task_graph_store.InSessionTaskPersistenceError:
        return _result(
            status=AuxiliaryApplicationStatus.FAILED,
            reason_code="attachment_authority_turn_not_authorized_for_task",
            effect_steps=effect_steps,
        )
    except MountedDocumentPlanningAuthorityError:
        return _result(
            status=AuxiliaryApplicationStatus.FAILED,
            reason_code="attachment_authority_persisted_scope_invalid",
            effect_steps=effect_steps,
        )
    try:
        task_graph_trigger = task_delivery_store.get_active_task_graph_revision_trigger(
            session_id=request.session_id,
            task_id=request.task_id,
        )
        execution_replan_request = (
            work_run_store.get_active_task_graph_execution_replan_request(
                session_id=request.session_id,
                task_id=request.task_id,
            )
        )
        if (
            task_graph_trigger is not None
            and execution_replan_request is not None
        ):
            raise AuxiliaryApplicationError(
                "Task has conflicting TaskGraph revision authorities"
            )
        task_graph_revision_authority = (
            task_graph_trigger or execution_replan_request
        )
        pending_supersede_receipt = (
            planning_store.get_pending_auxiliary_goal_supersede_receipt(
                session_id=request.session_id,
                task_id=request.task_id,
            )
        )
        if task_graph_revision_authority is not None and (
            pending_supersede_receipt is not None
            or target_change_intent is not None
        ):
            raise AuxiliaryApplicationError(
                "Auxiliary goal supersede conflicts with active TaskGraph "
                "revision authority"
            )
        if target_change_intent is not None:
            source_span = target_change_intent.source_span
            replacement_objective = target_change_intent.replacement_objective
            if replacement_objective is None:
                raise AuxiliaryApplicationError(
                    "target-change lane omitted its replacement objective"
                )
            admitted_target_receipt = (
                planning_store.get_authenticated_user_target_change_receipt(
                    session_id=request.session_id,
                    task_id=request.task_id,
                    invocation_turn_id=request.turn_id,
                    replacement_objective=replacement_objective,
                    source_start=source_span.start,
                    source_end=source_span.end,
                    source_sha256=source_span.text_sha256,
                )
            )
            if admitted_target_receipt is not None:
                if (
                    pending_supersede_receipt is not None
                    and pending_supersede_receipt != admitted_target_receipt
                ):
                    raise AuxiliaryApplicationError(
                        "target-change receipt conflicts with pending supersede authority"
                    )
                pending_supersede_receipt = admitted_target_receipt
            elif pending_supersede_receipt is not None:
                raise AuxiliaryApplicationError(
                    "target change cannot overtake another pending supersede"
                )
            else:
                current_goal = auxiliary_graph_store.get_auxiliary_graph_for_task(
                    session_id=request.session_id,
                    insession_task_id=request.task_id,
                )
                current_task = task_graph_store.get_insession_task_details(
                    request.session_id,
                    request.task_id,
                )
                if current_goal is None or current_task is None:
                    raise AuxiliaryApplicationError(
                        "target change requires an existing planning goal"
                    )
                superseded = run_auxiliary_goal_supersede(
                    AuxiliaryGoalSupersedeRequest(
                        session_id=request.session_id,
                        turn_id=request.turn_id,
                        task_id=request.task_id,
                        reason=(
                            planning_store.PlanningGoalSupersedeReason.USER_TARGET_CHANGED
                        ),
                        auxiliary_graph_id=current_goal.auxiliary_graph_id,
                        goal_id=current_goal.goal_id,
                        expected_task_state_version=(
                            current_task.task_state_version
                        ),
                        expected_control_state_version=(
                            current_goal.control_state_version
                        ),
                        expected_goal_state_version=(
                            current_goal.goal_state_version
                        ),
                        expected_revision_state_version=(
                            current_goal.revision_state_version
                        ),
                        expected_budget_state_version=(
                            current_goal.budget_state_version
                        ),
                        expected_current_auxiliary_graph_revision=(
                            current_goal.auxiliary_graph_revision
                        ),
                        expected_base_task_graph_revision=(
                            current_goal.base_task_graph_revision
                        ),
                        observed_task_graph_revision=(
                            current_task.current_graph_revision
                        ),
                        replacement_objective=replacement_objective,
                        source_start=source_span.start,
                        source_end=source_span.end,
                        source_sha256=source_span.text_sha256,
                    )
                )
                pending_supersede_receipt = superseded.store_result.receipt
                effect_steps += 1
                if effect_steps >= request.max_effect_steps:
                    return _result(
                        status=(
                            AuxiliaryApplicationStatus.STEP_LIMIT_REACHED
                        ),
                        reason_code="application_effect_step_limit_reached",
                        effect_steps=effect_steps,
                    )
        # 完全由模型生成的计划可能比其冻结的 TaskGraph 基础存活得更久。
        # 在要求历史初始规划器重放前，应检测这种持久化漂移；
        # 更重要的是，要在投影必然拒绝陈旧基础的执行前沿之前检测。
        # 目标取代本身是一项持久化效果；下方的后继目标仍须通过
        # 回执认证。
        if (
            pending_supersede_receipt is None
            and target_change_intent is None
            and task_graph_revision_authority is None
        ):
            persisted_goal = auxiliary_graph_store.get_auxiliary_graph_for_task(
                session_id=request.session_id,
                insession_task_id=request.task_id,
            )
            persisted_task = task_graph_store.get_insession_task_details(
                request.session_id,
                request.task_id,
            )
            if (
                persisted_goal is not None
                and persisted_task is not None
                and persisted_goal.auxiliary_graph_revision >= 2
                and persisted_goal.goal_status == "active"
                and persisted_goal.revision_status == "active"
                and persisted_task.current_graph_revision
                != persisted_goal.base_task_graph_revision
            ):
                superseded = run_auxiliary_goal_supersede(
                    AuxiliaryGoalSupersedeRequest(
                        session_id=request.session_id,
                        turn_id=request.turn_id,
                        task_id=request.task_id,
                        reason=planning_store.PlanningGoalSupersedeReason.BASE_DRIFT,
                        auxiliary_graph_id=persisted_goal.auxiliary_graph_id,
                        goal_id=persisted_goal.goal_id,
                        expected_task_state_version=(
                            persisted_task.task_state_version
                        ),
                        expected_control_state_version=(
                            persisted_goal.control_state_version
                        ),
                        expected_goal_state_version=(
                            persisted_goal.goal_state_version
                        ),
                        expected_revision_state_version=(
                            persisted_goal.revision_state_version
                        ),
                        expected_budget_state_version=(
                            persisted_goal.budget_state_version
                        ),
                        expected_current_auxiliary_graph_revision=(
                            persisted_goal.auxiliary_graph_revision
                        ),
                        expected_base_task_graph_revision=(
                            persisted_goal.base_task_graph_revision
                        ),
                        observed_task_graph_revision=(
                            persisted_task.current_graph_revision
                        ),
                    )
                )
                pending_supersede_receipt = superseded.store_result.receipt
                effect_steps += 1
                if effect_steps >= request.max_effect_steps:
                    return _result(
                        status=(
                            AuxiliaryApplicationStatus.STEP_LIMIT_REACHED
                        ),
                        reason_code="application_effect_step_limit_reached",
                        effect_steps=effect_steps,
                    )
        if (
            pending_supersede_receipt is not None
            and task_graph_revision_authority is not None
        ):
            raise AuxiliaryApplicationError(
                "pending Auxiliary goal supersede conflicts with active "
                "TaskGraph revision authority"
            )
        if pending_supersede_receipt is not None:
            planning_result = run_auxiliary_goal_successor_planning(
                AuxiliaryGoalSuccessorPlanningRequest(
                    session_id=request.session_id,
                    turn_id=request.turn_id,
                    task_id=request.task_id,
                    supersede_receipt=pending_supersede_receipt,
                    desired_output=request.desired_output,
                    deadline=request.deadline,
                    allowed_managed_document_ids=(
                        allowed_managed_document_ids
                    ),
                ),
                emit=ports.emit,
                provider=ports.planning_provider,
                ledger_store=model_ledger_store,
                knowledge_cognition_history_enabled=(
                    ports.knowledge_cognition_history_enabled
                ),
            )
        if (
            pending_supersede_receipt is None
            and task_graph_revision_authority is not None
        ):
            planning_result = run_positive_base_auxiliary_planning(
                AuxiliaryPositivePlanningRequest(
                    session_id=request.session_id,
                    turn_id=request.turn_id,
                    task_id=request.task_id,
                    trigger=task_graph_revision_authority,
                    desired_output=request.desired_output,
                    deadline=request.deadline,
                    allowed_managed_document_ids=(
                        allowed_managed_document_ids
                    ),
                ),
                emit=ports.emit,
                provider=ports.planning_provider,
                ledger_store=model_ledger_store,
                knowledge_cognition_history_enabled=(
                    ports.knowledge_cognition_history_enabled
                ),
            )
        elif pending_supersede_receipt is None:
            planning_result = run_initial_auxiliary_planning(
                session_id=request.session_id,
                turn_id=request.turn_id,
                insession_task_id=request.task_id,
                emit=ports.emit,
                provider=ports.planning_provider,
                ledger_store=model_ledger_store,
                desired_output=request.desired_output,
                deadline=request.deadline,
                allowed_managed_document_ids=allowed_managed_document_ids,
                knowledge_cognition_history_enabled=(
                    ports.knowledge_cognition_history_enabled
                ),
            )
    except RuntimeModelCallWaitingExternal:
        return _result(
            status=AuxiliaryApplicationStatus.WAITING_EXTERNAL,
            reason_code=(
                "successor_planner_model_call_waiting_external"
                if pending_supersede_receipt is not None
                else (
                    "positive_planner_model_call_waiting_external"
                    if task_graph_revision_authority is not None
                    else "initial_planner_model_call_waiting_external"
                )
            ),
            effect_steps=1,
        )
    except TurnDeadlineExceeded:
        return _result(
            status=AuxiliaryApplicationStatus.TURN_LIMIT_REACHED,
            reason_code=(
                "successor_planner_turn_deadline_reached"
                if pending_supersede_receipt is not None
                else (
                    "positive_planner_turn_deadline_reached"
                    if task_graph_revision_authority is not None
                    else "initial_planner_turn_deadline_reached"
                )
            ),
            effect_steps=1,
        )
    except (
        AuxiliaryInitialPlanningError,
        AuxiliaryPositivePlanningError,
        AuxiliaryGoalSuccessorPlanningError,
        AuxiliaryGoalSupersedeControllerError,
        planning_store.AuxiliaryGoalSupersedePersistenceError,
        task_delivery_store.TaskDeliveryValidationPersistenceError,
        work_run_store.WorkExecutionPersistenceError,
        AuxiliaryGraphArchitectGuardError,
        AuxiliaryGraphArchitectInputTooLarge,
        DurableModelCallTerminalState,
        AuxiliaryApplicationError,
        MountedVisualPlanningAuthorityError,
    ) as exc:
        return _result(
            status=AuxiliaryApplicationStatus.FAILED,
            reason_code=getattr(exc, "code", type(exc).__name__),
            effect_steps=effect_steps,
        )

    if (
        isinstance(planning_result, AuxiliaryInitialPlanningResult)
        and planning_result.status
        is AuxiliaryInitialPlanningStatus.SUPERSEDE_REQUIRED
    ):
        task = task_graph_store.get_insession_task_details(
            request.session_id,
            request.task_id,
        )
        stale = planning_result.details
        # 只有在 TaskGraph 基础已明确推进时，Architect 重定基决策
        # 才能在不臆造用户意图的情况下执行。目标变更通过
        # 已通过认证的回执进入此处。
        if (
            task is None
            or task.current_graph_revision == stale.base_task_graph_revision
        ):
            return _result(
                status=AuxiliaryApplicationStatus.REVISION_REQUIRED,
                reason_code=planning_result.status.value,
                effect_steps=effect_steps,
                planning_result=planning_result,
            )
        try:
            superseded = run_auxiliary_goal_supersede(
                AuxiliaryGoalSupersedeRequest(
                    session_id=request.session_id,
                    turn_id=request.turn_id,
                    task_id=request.task_id,
                    reason=planning_store.PlanningGoalSupersedeReason.BASE_DRIFT,
                    auxiliary_graph_id=stale.auxiliary_graph_id,
                    goal_id=stale.goal_id,
                    expected_task_state_version=task.task_state_version,
                    expected_control_state_version=stale.control_state_version,
                    expected_goal_state_version=stale.goal_state_version,
                    expected_revision_state_version=(
                        stale.revision_state_version
                    ),
                    expected_budget_state_version=stale.budget_state_version,
                    expected_current_auxiliary_graph_revision=(
                        stale.auxiliary_graph_revision
                    ),
                    expected_base_task_graph_revision=(
                        stale.base_task_graph_revision
                    ),
                    observed_task_graph_revision=task.current_graph_revision,
                )
            )
        except AuxiliaryGoalSupersedeControllerError as exc:
            return _result(
                status=AuxiliaryApplicationStatus.FAILED,
                reason_code=exc.code,
                effect_steps=effect_steps,
                planning_result=planning_result,
            )
        effect_steps += 1
        if effect_steps >= request.max_effect_steps:
            return _result(
                status=AuxiliaryApplicationStatus.STEP_LIMIT_REACHED,
                reason_code="application_effect_step_limit_reached",
                effect_steps=effect_steps,
                planning_result=planning_result,
            )
        try:
            planning_result = run_auxiliary_goal_successor_planning(
                AuxiliaryGoalSuccessorPlanningRequest(
                    session_id=request.session_id,
                    turn_id=request.turn_id,
                    task_id=request.task_id,
                    supersede_receipt=superseded.store_result.receipt,
                    desired_output=request.desired_output,
                    deadline=request.deadline,
                    allowed_managed_document_ids=(
                        allowed_managed_document_ids
                    ),
                ),
                emit=ports.emit,
                provider=ports.planning_provider,
                ledger_store=model_ledger_store,
                knowledge_cognition_history_enabled=(
                    ports.knowledge_cognition_history_enabled
                ),
            )
        except RuntimeModelCallWaitingExternal:
            return _result(
                status=AuxiliaryApplicationStatus.WAITING_EXTERNAL,
                reason_code="successor_planner_model_call_waiting_external",
                effect_steps=effect_steps + 1,
            )
        except TurnDeadlineExceeded:
            return _result(
                status=AuxiliaryApplicationStatus.TURN_LIMIT_REACHED,
                reason_code="successor_planner_turn_deadline_reached",
                effect_steps=effect_steps + 1,
            )
        except (
            AuxiliaryGoalSuccessorPlanningError,
            AuxiliaryGraphArchitectGuardError,
            AuxiliaryGraphArchitectInputTooLarge,
            DurableModelCallTerminalState,
        ) as exc:
            return _result(
                status=AuxiliaryApplicationStatus.FAILED,
                reason_code=getattr(exc, "code", type(exc).__name__),
                effect_steps=effect_steps,
            )

    if planning_result.status in {
        AuxiliaryInitialPlanningStatus.PLANNED,
        AuxiliaryPositivePlanningStatus.PLANNED,
        AuxiliaryGoalSuccessorPlanningStatus.PLANNED,
    }:
        effect_steps += 1
    elif (
        planning_result.status
        is AuxiliaryPositivePlanningStatus.PLANNER_DECLINED_TRIGGER
    ):
        return _result(
            status=AuxiliaryApplicationStatus.REVISION_REQUIRED,
            reason_code=planning_result.status.value,
            effect_steps=effect_steps,
            planning_result=planning_result,
        )
    elif (
        planning_result.status
        is AuxiliaryGoalSuccessorPlanningStatus.PLANNER_DECLINED_SUCCESSOR
    ):
        return _result(
            status=AuxiliaryApplicationStatus.REVISION_REQUIRED,
            reason_code=planning_result.status.value,
            effect_steps=effect_steps,
            planning_result=planning_result,
        )
    elif (
        planning_result.status
        is AuxiliaryInitialPlanningStatus.REQUEST_USER_INPUT
    ):
        return _result(
            status=AuxiliaryApplicationStatus.WAITING_USER,
            reason_code="initial_planner_requested_user_input",
            effect_steps=effect_steps,
            requested_user_question=planning_result.requested_user_question,
            planning_result=planning_result,
        )
    elif planning_result.status in {
        AuxiliaryInitialPlanningStatus.PLANNER_DECLINED_BOOTSTRAP,
    }:
        return _result(
            status=AuxiliaryApplicationStatus.REVISION_REQUIRED,
            reason_code=planning_result.status.value,
            effect_steps=effect_steps,
            planning_result=planning_result,
        )
    elif planning_result.status is AuxiliaryInitialPlanningStatus.TERMINAL_FAILED:
        return _result(
            status=AuxiliaryApplicationStatus.FAILED,
            reason_code="initial_planner_terminal_failed",
            effect_steps=effect_steps,
            planning_result=planning_result,
        )

    details = planning_result.details
    if details.goal_status == "committed" and details.revision_status == "committed":
        task = task_graph_store.get_insession_task_details(
            request.session_id,
            request.task_id,
        )
        if task is None or task.current_graph_revision != details.target_task_graph_revision:
            return _result(
                status=AuxiliaryApplicationStatus.FAILED,
                reason_code="committed_auxiliary_graph_task_graph_mismatch",
                effect_steps=effect_steps,
                planning_result=planning_result,
            )
        return _result(
            status=AuxiliaryApplicationStatus.COMMITTED,
            reason_code="task_graph_already_committed",
            effect_steps=effect_steps,
            planning_result=planning_result,
        )

    model_profile = ports.work_run_model_profile
    attempt_provider = ports.attempt_provider or build_attempt_structured_provider(
        model_profile
    )
    verification_provider = (
        ports.verification_provider
        or build_verification_structured_provider(model_profile)
    )
    try:
        execution_mounted_authority = (
            freeze_mounted_document_planning_authority(
                session_id=request.session_id,
                task_id=request.task_id,
                allowed_managed_document_ids=(
                    allowed_managed_document_ids
                ),
            )
        )
        creation_source = task_graph_store.get_insession_task_creation_source(
            session_id=request.session_id,
            insession_task_id=request.task_id,
        )
        # 在暴露模型工具之前，必须精确再现规划投影。
        # 目录中。版本变更必须以失败关闭并重新规划；它
        # 不能默默地扩大或替换一个活动的 WorkRun's 证据范围。
        build_mounted_document_authority_projection(
            authority_snapshot=details.authority_snapshot,
            task_creation_source=creation_source,
            mounted_authority=execution_mounted_authority,
        )
        mounted_document_runtime = (
            build_session_mounted_document_cognition_runtime(
                execution_mounted_authority
            )
        )
    except (
        MountedDocumentPlanningAuthorityError,
        MountedVisualPlanningAuthorityError,
        TypeError,
        ValueError,
    ):
        return _result(
            status=AuxiliaryApplicationStatus.FAILED,
            reason_code="mounted_document_cognition_runtime_unavailable",
            effect_steps=effect_steps,
            planning_result=planning_result,
            replan_result=replan_result,
        )
    workspace_runtime = (
        build_session_workspace_readonly_runtime(
            request.session_id,
        )
        if ports.capability_catalogs is None
        else None
    )
    if workspace_runtime is not None and ports.tool_ledger_store is None:
        return _result(
            status=AuxiliaryApplicationStatus.FAILED,
            reason_code="runtime_tool_ledger_unavailable",
            effect_steps=effect_steps,
            planning_result=planning_result,
            replan_result=replan_result,
        )
    workspace_tool_bridge = (
        build_workspace_readonly_work_run_bridge(
            workspace_runtime,
            ledger_store=ports.tool_ledger_store,
        )
        if workspace_runtime is not None
        else None
    )
    try:
        node_retrieval_runtime_factory = (
            None
            if ports.node_retrieval_runtime_builder is None
            else ports.node_retrieval_runtime_builder(
                execution_mounted_authority
            )
        )
    except (RuntimeError, TypeError, ValueError) as exc:
        return _result(
            status=AuxiliaryApplicationStatus.FAILED,
            reason_code=getattr(
                exc,
                "code",
                "auxiliary_node_retrieval_runtime_unavailable",
            ),
            effect_steps=effect_steps,
            planning_result=planning_result,
            replan_result=replan_result,
        )
    capability_catalogs = (
        ports.capability_catalogs
        if ports.capability_catalogs is not None
        else build_auxiliary_execution_capability_catalogs(
            workspace_runtime=workspace_runtime,
            mounted_document_runtime=mounted_document_runtime,
        )
    )
    if mounted_document_runtime is not None:
        # 这个精确的任务/版本绑定是安全性和新鲜性权威状态；
        # 调用者提供的同名目录永远不应遮蔽它。
        capability_catalogs = {
            **capability_catalogs,
            MOUNTED_DOCUMENT_COGNITION_CAPABILITY: (
                mounted_document_runtime.catalog_snapshot
            ),
        }
    capability_tool_bridges = (
        ports.capability_tool_bridges
        if ports.capability_tool_bridges is not None
        else {
            **(
                {WORKSPACE_READONLY_CAPABILITY: workspace_tool_bridge}
                if workspace_runtime is not None
                else {}
            ),
        }
    )
    if mounted_document_runtime is not None:
        # 保持桥与目录立即上方的冻结状态。
        capability_tool_bridges = {
            **capability_tool_bridges,
            MOUNTED_DOCUMENT_COGNITION_CAPABILITY: (
                mounted_document_runtime.tool_bridge
            ),
        }

    try:
        active_settlement = _active_replan_settlement(request)
    except AuxiliaryApplicationError as exc:
        return _result(
            status=AuxiliaryApplicationStatus.FAILED,
            reason_code=exc.code,
            effect_steps=effect_steps,
            planning_result=planning_result,
        )
    if active_settlement is not None:
        if effect_steps >= request.max_effect_steps:
            return _result(
                status=AuxiliaryApplicationStatus.STEP_LIMIT_REACHED,
                reason_code="application_effect_step_limit_reached",
                effect_steps=effect_steps,
                planning_result=planning_result,
            )
        try:
            replan_result = run_auxiliary_replanning(
                AuxiliaryReplanningRequest(
                    session_id=request.session_id,
                    turn_id=request.turn_id,
                    task_id=request.task_id,
                    settlement=active_settlement,
                    max_autonomous_replans=request.max_autonomous_replans,
                    deadline=request.deadline,
                ),
                provider=ports.planning_provider,
                emit=ports.emit,
                ledger_store=model_ledger_store,
                knowledge_cognition_history_enabled=(
                    ports.knowledge_cognition_history_enabled
                ),
            )
        except RuntimeModelCallWaitingExternal:
            return _result(
                status=AuxiliaryApplicationStatus.WAITING_EXTERNAL,
                reason_code="replan_architect_model_call_waiting_external",
                effect_steps=effect_steps + 1,
                planning_result=planning_result,
            )
        except TurnDeadlineExceeded:
            return _result(
                status=AuxiliaryApplicationStatus.TURN_LIMIT_REACHED,
                reason_code="replan_architect_turn_deadline_reached",
                effect_steps=effect_steps + 1,
                planning_result=planning_result,
            )
        except DurableModelCallTerminalState as exc:
            return _result(
                status=AuxiliaryApplicationStatus.FAILED,
                reason_code=getattr(exc, "code", type(exc).__name__),
                effect_steps=effect_steps + 1,
                planning_result=planning_result,
            )
        except (
            AuxiliaryReplanningError,
            AuxiliaryGraphArchitectGuardError,
            AuxiliaryGraphArchitectInputTooLarge,
        ) as exc:
            return _result(
                status=AuxiliaryApplicationStatus.FAILED,
                reason_code=getattr(exc, "code", type(exc).__name__),
                effect_steps=effect_steps + 1,
                planning_result=planning_result,
            )
        if replan_result.status is AuxiliaryReplanningStatus.REPLANNED or (
            replan_result.status
            is AuxiliaryReplanningStatus.ALREADY_REPLANNED
            and not replan_result.application_replayed
        ):
            effect_steps += 1

    while effect_steps < request.max_effect_steps:
        frontier = auxiliary_graph_store.project_auxiliary_graph_execution_frontier(
            session_id=request.session_id,
            turn_id=request.turn_id,
            insession_task_id=request.task_id,
        )
        decision = decide_auxiliary_graph_driver_step(frontier)
        action = decision.action
        if (
            action is AuxiliaryGraphDriverAction.WAIT_USER
            and _authenticated_waiting_user_continuation(
                request=request,
                frontier=frontier,
                lane=current_lane,
                subject=decision.subject,
                work_run_id=decision.work_run_id,
            )
        ):
            # 纯图 Driver 无法看到 Entry 中精确绑定来源通道的回执。
            # 组合层只有在联结这项额外权威状态后，才能解除该停止状态；
            # WorkRun 控制器会在同一原子延续操作中再次执行相同的
            # 待处理问题检查。
            action = AuxiliaryGraphDriverAction.RESUME_WORK_RUN

        if action in {
            AuxiliaryGraphDriverAction.RUN_HOST_PRIMITIVE,
            AuxiliaryGraphDriverAction.RESUME_HOST_PRIMITIVE,
        }:
            if decision.subject is None:
                raise AuxiliaryApplicationError(
                    "Host dispatch lost its selected subject"
                )
            try:
                factory = ports.host_request_factory
                resource_read_port = ports.resource_read_port
                if factory is None:
                    factory, mounted_authority = _mounted_resource_factory(
                        request=request,
                        subject=decision.subject,
                        allowed_managed_document_ids=(
                            allowed_managed_document_ids
                        ),
                    )
                    if resource_read_port is None:
                        resource_read_port = build_mounted_resource_read_port(
                            mounted_authority=mounted_authority,
                        )
                primitive_kinds = (
                    ports.primitive_kinds_by_capability_profile
                    if ports.primitive_kinds_by_capability_profile is not None
                    else {
                        MOUNTED_DOCUMENT_READ_CAPABILITY: (
                            PlanningContextPrimitiveKind.RESOURCE_PERCEPTION
                        ),
                        MOUNTED_VISUAL_READ_CAPABILITY: (
                            PlanningContextPrimitiveKind.RESOURCE_PERCEPTION
                        ),
                    }
                )
                host_result = run_auxiliary_host_primitive(
                    AuxiliaryHostPrimitiveControllerRequest(
                        session_id=request.session_id,
                        turn_id=request.turn_id,
                        subject=decision.subject,
                        initial_driver_state_guard_sha256=(
                            canonical_auxiliary_graph_driver_state_guard(frontier)
                        ),
                    ),
                    request_factory=factory,
                    primitive_kinds_by_capability_profile=primitive_kinds,
                    monotonic_clock=ports.monotonic_clock,
                    resource_read_port=resource_read_port,
                )
            except (
                AuxiliaryHostPrimitiveControllerError,
                MountedDocumentPlanningAuthorityError,
                MountedVisualPlanningAuthorityError,
            ) as exc:
                return _result(
                    status=AuxiliaryApplicationStatus.FAILED,
                    reason_code=getattr(exc, "code", type(exc).__name__),
                    effect_steps=effect_steps,
                    last_driver_action=action,
                    subject=decision.subject,
                    planning_result=planning_result,
                    replan_result=replan_result,
                )
            effect_steps += 1
            if (
                host_result.status
                is AuxiliaryHostPrimitiveControllerStatus.WAITING_EXTERNAL
            ):
                return _result(
                    status=AuxiliaryApplicationStatus.WAITING_EXTERNAL,
                    reason_code=host_result.reason_code,
                    effect_steps=effect_steps,
                    last_driver_action=action,
                    subject=decision.subject,
                    planning_result=planning_result,
                    host_result=host_result,
                    replan_result=replan_result,
                )
            continue

        if action in {
            AuxiliaryGraphDriverAction.RUN_MODEL_WORK_RUN,
            AuxiliaryGraphDriverAction.OPEN_USER_GATE,
            AuxiliaryGraphDriverAction.RUN_TERMINAL_PLANNER,
            AuxiliaryGraphDriverAction.RESUME_WORK_RUN,
        }:
            if decision.subject is None:
                raise AuxiliaryApplicationError(
                    "WorkRun dispatch lost its selected subject"
                )
            executor_kind = _selected_model_executor(frontier, action=action)
            validation_context = (
                _task_graph_validation_context(request=request, frontier=frontier)
                if executor_kind
                is AuxiliaryNodeExecutorKind.TERMINAL_PLANNER
                else None
            )
            semantic_base_snapshot = (
                None
                if validation_context is None
                or validation_context.expected_current_graph_revision is None
                else task_graph_store.project_task_graph_semantic_base(
                    session_id=request.session_id,
                    task_id=request.task_id,
                    graph_revision=(
                        validation_context.expected_current_graph_revision
                    ),
                ).snapshot
            )
            try:
                terminal_downstream_gate = None
                if (
                    executor_kind
                    is AuxiliaryNodeExecutorKind.TERMINAL_PLANNER
                ):
                    assert validation_context is not None

                    def terminal_downstream_gate(context, node_result):
                        return run_auxiliary_terminal_candidate_semantic_gate(
                            session_id=request.session_id,
                            turn_id=request.turn_id,
                            task_id=request.task_id,
                            validation_context=validation_context,
                            context=context,
                            node_result=node_result,
                            emit=ports.emit,
                            semantic_provider=ports.semantic_provider,
                            semantic_model_call_authority_factory=(
                                semantic_model_call_authority_factory
                            ),
                            desired_output=request.desired_output,
                            deadline=request.deadline,
                            allowed_managed_document_ids=(
                                allowed_managed_document_ids
                            ),
                        )

                work_result = run_auxiliary_model_node(
                    AuxiliaryWorkRunRequest(
                        session_id=request.session_id,
                        turn_id=request.turn_id,
                        subject=decision.subject,
                        executor_kind=executor_kind,
                        initial_driver_state_guard_sha256=(
                            canonical_auxiliary_graph_driver_state_guard(frontier)
                        ),
                        id_plan=derive_auxiliary_work_run_ids(
                            session_id=request.session_id,
                            subject=decision.subject,
                        ),
                        allow_user_input=ports.allow_user_input,
                        task_graph_validation_context=validation_context,
                        task_graph_semantic_base_snapshot=(
                            semantic_base_snapshot
                        ),
                    ),
                    profile=ports.work_run_runtime_profile,
                    capability_catalogs=capability_catalogs,
                    capability_tool_bridges=capability_tool_bridges,
                    node_retrieval_runtime_factory=(
                        node_retrieval_runtime_factory
                    ),
                    attempt_provider=attempt_provider,
                    verification_provider=verification_provider,
                    emit=ports.emit,
                    monotonic_clock=ports.monotonic_clock,
                    deadline=request.deadline,
                    # 此项刻意不允许由应用请求配置。每次 WorkRun 调用
                    # 都必须经过生产环境中可自认证的持久化模型权威状态。
                    model_call_authority_factory=(
                        work_run_model_call_authority_factory
                    ),
                    terminal_downstream_gate=terminal_downstream_gate,
                )
            except AuxiliaryTerminalCandidateSemanticRouteRequired as exc:
                effect_steps += 1
                if exc.settlement is None:
                    return _result(
                        status=AuxiliaryApplicationStatus.FAILED,
                        reason_code=exc.code,
                        effect_steps=effect_steps,
                        last_driver_action=action,
                        subject=decision.subject,
                        planning_result=planning_result,
                        replan_result=replan_result,
                    )
                if effect_steps >= request.max_effect_steps:
                    return _result(
                        status=AuxiliaryApplicationStatus.STEP_LIMIT_REACHED,
                        reason_code="application_effect_step_limit_reached",
                        effect_steps=effect_steps,
                        last_driver_action=action,
                        subject=decision.subject,
                        planning_result=planning_result,
                        replan_result=replan_result,
                    )
                try:
                    replan_result = run_auxiliary_replanning(
                        AuxiliaryReplanningRequest(
                            session_id=request.session_id,
                            turn_id=request.turn_id,
                            task_id=request.task_id,
                            settlement=exc.settlement,
                            max_autonomous_replans=(
                                request.max_autonomous_replans
                            ),
                            deadline=request.deadline,
                        ),
                        provider=ports.planning_provider,
                        emit=ports.emit,
                        ledger_store=model_ledger_store,
                        knowledge_cognition_history_enabled=(
                            ports.knowledge_cognition_history_enabled
                        ),
                    )
                except RuntimeModelCallWaitingExternal:
                    return _result(
                        status=AuxiliaryApplicationStatus.WAITING_EXTERNAL,
                        reason_code="replan_architect_model_call_waiting_external",
                        effect_steps=effect_steps + 1,
                        last_driver_action=action,
                        subject=decision.subject,
                        planning_result=planning_result,
                    )
                except TurnDeadlineExceeded:
                    return _result(
                        status=AuxiliaryApplicationStatus.TURN_LIMIT_REACHED,
                        reason_code="replan_architect_turn_deadline_reached",
                        effect_steps=effect_steps + 1,
                        last_driver_action=action,
                        subject=decision.subject,
                        planning_result=planning_result,
                    )
                except DurableModelCallTerminalState as replan_exc:
                    return _result(
                        status=AuxiliaryApplicationStatus.FAILED,
                        reason_code=getattr(
                            replan_exc,
                            "code",
                            type(replan_exc).__name__,
                        ),
                        effect_steps=effect_steps + 1,
                        last_driver_action=action,
                        subject=decision.subject,
                        planning_result=planning_result,
                    )
                except (
                    AuxiliaryReplanningError,
                    AuxiliaryGraphArchitectGuardError,
                    AuxiliaryGraphArchitectInputTooLarge,
                ) as replan_exc:
                    return _result(
                        status=AuxiliaryApplicationStatus.FAILED,
                        reason_code=getattr(
                            replan_exc,
                            "code",
                            type(replan_exc).__name__,
                        ),
                        effect_steps=effect_steps + 1,
                        last_driver_action=action,
                        subject=decision.subject,
                        planning_result=planning_result,
                    )
                if (
                    replan_result.status
                    is AuxiliaryReplanningStatus.LIMIT_REACHED
                ):
                    return _result(
                        status=AuxiliaryApplicationStatus.REVISION_REQUIRED,
                        reason_code=replan_result.reason_code,
                        effect_steps=effect_steps,
                        last_driver_action=action,
                        subject=decision.subject,
                        planning_result=planning_result,
                        replan_result=replan_result,
                    )
                if (
                    replan_result.status
                    is AuxiliaryReplanningStatus.REPLANNED
                    or not replan_result.application_replayed
                ):
                    effect_steps += 1
                continue
            except AuxiliaryTerminalCompositionError as exc:
                return _result(
                    status=AuxiliaryApplicationStatus.FAILED,
                    reason_code=exc.code,
                    effect_steps=effect_steps + 1,
                    last_driver_action=action,
                    subject=decision.subject,
                    planning_result=planning_result,
                    replan_result=replan_result,
                )
            except RuntimeModelCallWaitingExternal:
                return _result(
                    status=AuxiliaryApplicationStatus.WAITING_EXTERNAL,
                    reason_code="work_run_model_call_waiting_external",
                    effect_steps=effect_steps + 1,
                    last_driver_action=action,
                    subject=decision.subject,
                    planning_result=planning_result,
                    replan_result=replan_result,
                )
            except TurnDeadlineExceeded:
                return _result(
                    status=AuxiliaryApplicationStatus.TURN_LIMIT_REACHED,
                    reason_code="work_run_turn_deadline_reached",
                    effect_steps=effect_steps + 1,
                    last_driver_action=action,
                    subject=decision.subject,
                    planning_result=planning_result,
                    replan_result=replan_result,
                )
            except ModelGatewayError:
                return _result(
                    status=AuxiliaryApplicationStatus.FAILED,
                    reason_code="terminal_candidate_semantic_model_interrupted",
                    effect_steps=effect_steps + 1,
                    last_driver_action=action,
                    subject=decision.subject,
                    planning_result=planning_result,
                    replan_result=replan_result,
                )
            except DurableModelCallTerminalState as exc:
                return _result(
                    status=AuxiliaryApplicationStatus.FAILED,
                    reason_code=getattr(exc, "code", type(exc).__name__),
                    effect_steps=effect_steps + 1,
                    last_driver_action=action,
                    subject=decision.subject,
                    planning_result=planning_result,
                    replan_result=replan_result,
                )
            effect_steps += 1
            boundary = _work_run_boundary(
                work_result,
                session_id=request.session_id,
                effect_steps=effect_steps,
                action=action,
                planning_result=planning_result,
                replan_result=replan_result,
            )
            if boundary is not None:
                return boundary
            continue

        if action in {
            AuxiliaryGraphDriverAction.SEAL_REVISION,
            AuxiliaryGraphDriverAction.COMMIT_READY_PROPOSAL,
        }:
            try:
                terminal_result = run_auxiliary_terminal_composition(
                    session_id=request.session_id,
                    turn_id=request.turn_id,
                    task_id=request.task_id,
                    emit=ports.emit,
                    semantic_provider=ports.semantic_provider,
                    semantic_model_call_authority_factory=(
                        semantic_model_call_authority_factory
                    ),
                    desired_output=request.desired_output,
                    deadline=request.deadline,
                    allowed_managed_document_ids=(
                        allowed_managed_document_ids
                    ),
                )
            except AuxiliaryTerminalCompositionError as exc:
                return _result(
                    status=AuxiliaryApplicationStatus.FAILED,
                    reason_code=exc.code,
                    effect_steps=effect_steps,
                    last_driver_action=action,
                    planning_result=planning_result,
                    replan_result=replan_result,
                )
            effect_steps += 1
            if (
                terminal_result.status
                is AuxiliaryTerminalCompositionStatus.REVISION_REQUIRED
            ):
                semantic_result = terminal_result.semantic_result
                settlement = (
                    semantic_result.settlement
                    if semantic_result is not None
                    else None
                )
                if settlement is None:
                    return _result(
                        status=AuxiliaryApplicationStatus.FAILED,
                        reason_code="semantic_revision_lost_settlement",
                        effect_steps=effect_steps,
                        last_driver_action=action,
                        planning_result=planning_result,
                        terminal_result=terminal_result,
                        replan_result=replan_result,
                    )
                if effect_steps >= request.max_effect_steps:
                    return _result(
                        status=AuxiliaryApplicationStatus.STEP_LIMIT_REACHED,
                        reason_code="application_effect_step_limit_reached",
                        effect_steps=effect_steps,
                        last_driver_action=action,
                        planning_result=planning_result,
                        terminal_result=terminal_result,
                        replan_result=replan_result,
                    )
                try:
                    replan_result = run_auxiliary_replanning(
                        AuxiliaryReplanningRequest(
                            session_id=request.session_id,
                            turn_id=request.turn_id,
                            task_id=request.task_id,
                            settlement=settlement,
                            max_autonomous_replans=(
                                request.max_autonomous_replans
                            ),
                            deadline=request.deadline,
                        ),
                        provider=ports.planning_provider,
                        emit=ports.emit,
                        ledger_store=model_ledger_store,
                        knowledge_cognition_history_enabled=(
                            ports.knowledge_cognition_history_enabled
                        ),
                    )
                except RuntimeModelCallWaitingExternal:
                    return _result(
                        status=AuxiliaryApplicationStatus.WAITING_EXTERNAL,
                        reason_code="replan_architect_model_call_waiting_external",
                        effect_steps=effect_steps + 1,
                        last_driver_action=action,
                        planning_result=planning_result,
                        terminal_result=terminal_result,
                    )
                except TurnDeadlineExceeded:
                    return _result(
                        status=AuxiliaryApplicationStatus.TURN_LIMIT_REACHED,
                        reason_code="replan_architect_turn_deadline_reached",
                        effect_steps=effect_steps + 1,
                        last_driver_action=action,
                        planning_result=planning_result,
                        terminal_result=terminal_result,
                    )
                except DurableModelCallTerminalState as exc:
                    return _result(
                        status=AuxiliaryApplicationStatus.FAILED,
                        reason_code=getattr(exc, "code", type(exc).__name__),
                        effect_steps=effect_steps + 1,
                        last_driver_action=action,
                        planning_result=planning_result,
                        terminal_result=terminal_result,
                    )
                except (
                    AuxiliaryReplanningError,
                    AuxiliaryGraphArchitectGuardError,
                    AuxiliaryGraphArchitectInputTooLarge,
                ) as exc:
                    return _result(
                        status=AuxiliaryApplicationStatus.FAILED,
                        reason_code=getattr(exc, "code", type(exc).__name__),
                        effect_steps=effect_steps + 1,
                        last_driver_action=action,
                        planning_result=planning_result,
                        terminal_result=terminal_result,
                    )
                if (
                    replan_result.status
                    is AuxiliaryReplanningStatus.LIMIT_REACHED
                ):
                    return _result(
                        status=AuxiliaryApplicationStatus.REVISION_REQUIRED,
                        reason_code=replan_result.reason_code,
                        effect_steps=effect_steps,
                        last_driver_action=action,
                        planning_result=planning_result,
                        terminal_result=terminal_result,
                        replan_result=replan_result,
                    )
                if (
                    replan_result.status
                    is AuxiliaryReplanningStatus.REPLANNED
                    or not replan_result.application_replayed
                ):
                    effect_steps += 1
                continue
            return _terminal_boundary(
                terminal_result,
                effect_steps=effect_steps,
                action=action,
                planning_result=planning_result,
                replan_result=replan_result,
            )

        boundary = _driver_boundary(
            request=request,
            action=action,
            reason_code=decision.reason_code,
            effect_steps=effect_steps,
            subject=decision.subject,
            planning_result=planning_result,
            replan_result=replan_result,
        )
        if boundary is not None:
            return boundary
        raise AuxiliaryApplicationError(
            f"unsupported AuxiliaryGraph driver action: {action.value}"
        )

    return _result(
        status=AuxiliaryApplicationStatus.STEP_LIMIT_REACHED,
        reason_code="application_effect_step_limit_reached",
        effect_steps=effect_steps,
        planning_result=planning_result,
        replan_result=replan_result,
    )


def _active_replan_settlement(request: AuxiliaryApplicationRequest):
    trigger = planning_store.get_active_auxiliary_replan_trigger(
        session_id=request.session_id,
        task_id=request.task_id,
    )
    if trigger is None:
        return None
    settlement = semantic_store.get_auxiliary_semantic_quorum_settlement(
        session_id=request.session_id,
        task_id=request.task_id,
        auxiliary_graph_id=trigger.auxiliary_graph_id,
        goal_id=trigger.goal_id,
        auxiliary_graph_revision=trigger.source_auxiliary_graph_revision,
        frozen_prompt_payload_sha256=trigger.semantic_prompt_payload_sha256,
    )
    if (
        settlement is None
        or settlement.settlement_id != trigger.semantic_settlement_id
        or settlement.settlement_sha256 != trigger.semantic_settlement_sha256
    ):
        raise AuxiliaryApplicationError(
            "active replan trigger lost its semantic settlement authority"
        )
    return settlement


def _authenticated_waiting_user_continuation(
    *,
    request: AuxiliaryApplicationRequest,
    frontier,
    lane,
    subject: AuxiliaryNodeSubject | None,
    work_run_id: str | None,
) -> bool:
    """将一个 Driver 等待光标与一个精确答案意图收据连接起来。"""

    if subject is None or work_run_id is None or len(frontier.recoverable) != 1:
        return False
    candidate = frontier.recoverable[0]
    if candidate.subject != subject or candidate.work_run_id != work_run_id:
        return False
    if (
        lane is None
        or lane.insession_task_id != request.task_id
        or lane.execution_requested is not True
        or not any(
            item.match_type == "existing_root"
            and item.execution_requested is True
            for item in lane.matches
        )
    ):
        return False
    try:
        pending = continuation_store.get_auxiliary_pending_user_question(
            session_id=request.session_id,
            insession_task_id=request.task_id,
        )
    except continuation_store.AuxiliaryContinuationPersistenceError:
        return False
    return bool(
        pending is not None
        and pending.subject == subject
        and pending.work_run_id == work_run_id
        and pending.question_turn_id != request.turn_id
    )


def _mounted_resource_factory(
    *,
    request: AuxiliaryApplicationRequest,
    subject: AuxiliaryNodeSubject,
    allowed_managed_document_ids: tuple[str, ...],
) -> tuple[
    AuxiliaryHostPrimitiveRequestFactory,
    FrozenMountedDocumentPlanningAuthority,
]:
    details = auxiliary_graph_store.get_auxiliary_graph_for_task(
        session_id=request.session_id,
        insession_task_id=request.task_id,
    )
    if details is None:
        raise AuxiliaryApplicationError(
            "mounted Host dispatch lost graph authority"
        )
    node = next(
        (
            item
            for item in details.nodes
            if item.auxiliary_node_id == subject.node_id
            and item.node_revision == subject.node_revision
        ),
        None,
    )
    if node is None:
        raise AuxiliaryApplicationError(
            "mounted Host dispatch lost its current node"
        )
    mounted = freeze_mounted_document_planning_authority(
        session_id=request.session_id,
        task_id=request.task_id,
        allowed_managed_document_ids=allowed_managed_document_ids,
    )
    creation_source = task_graph_store.get_insession_task_creation_source(
        session_id=request.session_id,
        insession_task_id=request.task_id,
    )
    # 这验证当前私有资源标识仍能精确再现任何读取尝试前的规划快照。
    # 这验证当前私有资源标识仍能精确再现任何读取尝试前的规划快照。
    build_mounted_document_authority_projection(
        authority_snapshot=details.authority_snapshot,
        task_creation_source=creation_source,
        mounted_authority=mounted,
    )

    def factory(context):
        return build_mounted_resource_perception_request(
            context=context,
            input_resource_aliases=node.input_resource_aliases,
            mounted_authority=mounted,
        )

    return factory, mounted


def _selected_model_executor(
    frontier,
    *,
    action: AuxiliaryGraphDriverAction,
) -> AuxiliaryNodeExecutorKind:
    if action is AuxiliaryGraphDriverAction.RUN_MODEL_WORK_RUN:
        return AuxiliaryNodeExecutorKind.MODEL_WORK_RUN
    if action is AuxiliaryGraphDriverAction.OPEN_USER_GATE:
        return AuxiliaryNodeExecutorKind.USER_GATE
    if action is AuxiliaryGraphDriverAction.RUN_TERMINAL_PLANNER:
        return AuxiliaryNodeExecutorKind.TERMINAL_PLANNER
    if action is AuxiliaryGraphDriverAction.RESUME_WORK_RUN:
        if len(frontier.recoverable) != 1:
            raise AuxiliaryApplicationError(
                "WorkRun recovery frontier lost its sole cursor"
            )
        executor = frontier.recoverable[0].executor_kind
        if executor not in {
            AuxiliaryNodeExecutorKind.MODEL_WORK_RUN,
            AuxiliaryNodeExecutorKind.USER_GATE,
            AuxiliaryNodeExecutorKind.TERMINAL_PLANNER,
        }:
            raise AuxiliaryApplicationError(
                "recoverable WorkRun has a non-Attempt executor"
            )
        return executor
    raise AuxiliaryApplicationError("driver action is not a WorkRun dispatch")


def _task_graph_validation_context(
    *,
    request: AuxiliaryApplicationRequest,
    frontier,
) -> InSessionTaskGraphRevisionValidationContext:
    context = terminal_store.build_auxiliary_terminal_task_graph_validation_context(
        session_id=request.session_id,
        invocation_turn_id=request.turn_id,
        task_id=request.task_id,
    )
    if context.expected_current_graph_revision != frontier.base_task_graph_revision:
        raise AuxiliaryApplicationError(
            "terminal planner base TaskGraph authority changed"
        )
    return context


def _work_run_boundary(
    result: AuxiliaryWorkRunResult,
    *,
    session_id: str,
    effect_steps: int,
    action: AuxiliaryGraphDriverAction,
    planning_result: AuxiliaryPlanningResult,
    replan_result: AuxiliaryReplanningResult | None,
) -> AuxiliaryApplicationResult | None:
    if result.status is AuxiliaryWorkRunStatus.COMPLETED:
        return None
    status = {
        AuxiliaryWorkRunStatus.WAITING_USER: (
            AuxiliaryApplicationStatus.WAITING_USER
        ),
        AuxiliaryWorkRunStatus.WAITING_AUTHORIZATION: (
            AuxiliaryApplicationStatus.WAITING_AUTHORIZATION
        ),
        AuxiliaryWorkRunStatus.WAITING_EXTERNAL: (
            AuxiliaryApplicationStatus.WAITING_EXTERNAL
        ),
        AuxiliaryWorkRunStatus.TURN_LIMIT_REACHED: (
            AuxiliaryApplicationStatus.TURN_LIMIT_REACHED
        ),
        AuxiliaryWorkRunStatus.MODEL_INTERRUPTED: (
            AuxiliaryApplicationStatus.TURN_LIMIT_REACHED
        ),
        AuxiliaryWorkRunStatus.VERIFICATION_INTERRUPTED: (
            AuxiliaryApplicationStatus.TURN_LIMIT_REACHED
        ),
        AuxiliaryWorkRunStatus.STEP_LIMIT_REACHED: (
            AuxiliaryApplicationStatus.STEP_LIMIT_REACHED
        ),
    }.get(result.status, AuxiliaryApplicationStatus.FAILED)
    requested_user_question = None
    if result.status is AuxiliaryWorkRunStatus.WAITING_USER:
        pending = continuation_store.get_auxiliary_pending_user_question(
            session_id=session_id,
            insession_task_id=result.subject.task_id,
        )
        if pending is not None and pending.work_run_id == result.work_run_id:
            requested_user_question = pending.question
    return _result(
        status=status,
        reason_code=result.reason_code,
        effect_steps=effect_steps,
        last_driver_action=action,
        subject=result.subject,
        requested_user_question=requested_user_question,
        planning_result=planning_result,
        work_run_result=result,
        replan_result=replan_result,
    )


def _terminal_boundary(
    result: AuxiliaryTerminalCompositionResult,
    *,
    effect_steps: int,
    action: AuxiliaryGraphDriverAction,
    planning_result: AuxiliaryPlanningResult,
    replan_result: AuxiliaryReplanningResult | None,
) -> AuxiliaryApplicationResult:
    status = {
        AuxiliaryTerminalCompositionStatus.COMMITTED: (
            AuxiliaryApplicationStatus.COMMITTED
        ),
        AuxiliaryTerminalCompositionStatus.ALREADY_COMMITTED: (
            AuxiliaryApplicationStatus.COMMITTED
        ),
        AuxiliaryTerminalCompositionStatus.WAITING_EXTERNAL: (
            AuxiliaryApplicationStatus.WAITING_EXTERNAL
        ),
        AuxiliaryTerminalCompositionStatus.TURN_LIMIT_REACHED: (
            AuxiliaryApplicationStatus.TURN_LIMIT_REACHED
        ),
        AuxiliaryTerminalCompositionStatus.MODEL_INTERRUPTED: (
            AuxiliaryApplicationStatus.TURN_LIMIT_REACHED
        ),
        AuxiliaryTerminalCompositionStatus.REVISION_REQUIRED: (
            AuxiliaryApplicationStatus.REVISION_REQUIRED
        ),
        AuxiliaryTerminalCompositionStatus.FAILED_CLOSED: (
            AuxiliaryApplicationStatus.FAILED
        ),
    }[result.status]
    return _result(
        status=status,
        reason_code=result.reason_code,
        effect_steps=effect_steps,
        last_driver_action=action,
        planning_result=planning_result,
        terminal_result=result,
        replan_result=replan_result,
    )


def _driver_boundary(
    *,
    request: AuxiliaryApplicationRequest,
    action: AuxiliaryGraphDriverAction,
    reason_code: str,
    effect_steps: int,
    subject: AuxiliaryNodeSubject | None,
    planning_result: AuxiliaryPlanningResult,
    replan_result: AuxiliaryReplanningResult | None,
) -> AuxiliaryApplicationResult | None:
    if action is AuxiliaryGraphDriverAction.WAIT_USER:
        status = AuxiliaryApplicationStatus.WAITING_USER
    elif action is AuxiliaryGraphDriverAction.WAIT_AUTHORIZATION:
        status = AuxiliaryApplicationStatus.WAITING_AUTHORIZATION
    elif action is AuxiliaryGraphDriverAction.WAIT_EXTERNAL:
        status = AuxiliaryApplicationStatus.WAITING_EXTERNAL
    elif action is AuxiliaryGraphDriverAction.STOP_TURN:
        status = AuxiliaryApplicationStatus.TURN_LIMIT_REACHED
    elif action in {
        AuxiliaryGraphDriverAction.REQUEST_REVISION,
        AuxiliaryGraphDriverAction.BUDGET_EXHAUSTED,
    }:
        status = AuxiliaryApplicationStatus.REVISION_REQUIRED
    elif action in {
        AuxiliaryGraphDriverAction.FAILED_CLOSED,
        AuxiliaryGraphDriverAction.TERMINAL_STOP,
    }:
        details = auxiliary_graph_store.get_auxiliary_graph_for_task(
            session_id=request.session_id,
            insession_task_id=request.task_id,
        )
        if (
            action is AuxiliaryGraphDriverAction.TERMINAL_STOP
            and details is not None
            and details.goal_status == "committed"
            and details.revision_status == "committed"
        ):
            status = AuxiliaryApplicationStatus.COMMITTED
        else:
            status = AuxiliaryApplicationStatus.FAILED
    else:
        return None
    return _result(
        status=status,
        reason_code=reason_code,
        effect_steps=effect_steps,
        last_driver_action=action,
        subject=subject,
        planning_result=planning_result,
        replan_result=replan_result,
    )


def _result(
    *,
    status: AuxiliaryApplicationStatus,
    reason_code: str,
    effect_steps: int,
    last_driver_action: AuxiliaryGraphDriverAction | None = None,
    subject: AuxiliaryNodeSubject | None = None,
    requested_user_question: str | None = None,
    planning_result: AuxiliaryPlanningResult | None = None,
    host_result: AuxiliaryHostPrimitiveControllerResult | None = None,
    work_run_result: AuxiliaryWorkRunResult | None = None,
    terminal_result: AuxiliaryTerminalCompositionResult | None = None,
    replan_result: AuxiliaryReplanningResult | None = None,
) -> AuxiliaryApplicationResult:
    return AuxiliaryApplicationResult(
        status=status,
        reason_code=reason_code,
        effect_steps=effect_steps,
        last_driver_action=last_driver_action,
        subject=subject,
        requested_user_question=requested_user_question,
        planning_result=planning_result,
        host_result=host_result,
        work_run_result=work_run_result,
        terminal_result=terminal_result,
        replan_result=replan_result,
    )


__all__ = [
    "AuxiliaryApplicationError",
    "AuxiliaryApplicationPorts",
    "AuxiliaryApplicationRequest",
    "AuxiliaryApplicationResult",
    "AuxiliaryApplicationStatus",
    "run_auxiliary_application_to_boundary",
]
