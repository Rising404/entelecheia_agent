"""AuxiliaryGraph：执行一个精确的当前 TaskGraph 提交到其根 Delivery。

一个终端提交已经选择了精确的任务。因此，此组合直接操作该授权目标，从不假设无关的多通道权威状态。

此模块是下一个较低缝合处的生产适配器。它认证当前 TaskGraph 是已提交
AuxiliaryGraph 目标的确切修订版，重新加载活动 TurnWindow 修订版，并将所有节点
选择、WorkRun 恢复、OutputWindow 提交、验证和 FinishGate 完成委托给
``run_task_graph_work_runs``。它返回的唯一主体是通过 Store 的自验证根
``NodeDelivery`` 投影重新加载的。一个完成的任务在实时 Window 要求之前处理，
因此在响应丢失后（或在后续 Turn 关闭后）的重试不能分发第二个模型调用或创建
第二个 Delivery。

多任务通道结算和完成节点携带解决仍在此组合之外。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
import time

from personagraph.context_budget import ContextBudgetExceeded
from personagraph.session import store
from personagraph.session.l2_store import continuation as continuation_store
from personagraph.session.l2_store import verification as verification_store
from personagraph.session.l2_store import auxiliary_graph as auxiliary_graph_store
from personagraph.session.l2_store import task_delivery as task_delivery_store
from personagraph.session.l2_store import task_graph as task_graph_store
from personagraph.session.l2_store import work_run as work_run_store
from personagraph.tools.catalog import CatalogSnapshot
from personagraph.l2.task_execution.attempts.decision import AttemptDecisionStructuredProvider
from personagraph.runtime.turn_deadline import TurnDeadline
from personagraph.l2.model_output_budgets import L2_MAX_OUTPUT_TOKENS
from personagraph.runtime.model_calls.contracts import (
    RuntimeModelLedgerStore,
)
from personagraph.l2.task_execution.tool_bridge.contracts import AttemptToolBridge
from personagraph.l2.task_execution.verification.decision import NodeVerificationStructuredProvider
from personagraph.l2.task_execution.task_node.model_authority_contracts import (
    TaskNodeModelCallAuthorityFactory,
)
from personagraph.l2.task_execution.task_node.model_binding_contracts import (
    TaskNodeBoundModelCall,
)
from personagraph.l2.task_execution.task_node.model_authority import (
    create_task_node_work_run_model_call_authority,
)
from personagraph.l2.task_execution.task_graph.controller import (
    run_task_graph_work_runs,
)
from personagraph.l2.task_execution.task_graph.contracts import (
    TaskGraphWorkRunRequest,
    TaskGraphWorkRunResult,
)
from personagraph.l2.task_execution.task_graph.profile import (
    TaskGraphWorkRunProfile,
)
from personagraph.l2.task_execution.task_node.tool_runtime_contracts import (
    TaskNodeToolRuntimeFactory,
)
from personagraph.runtime.turn_events import TurnEvent
from personagraph.l2.task_execution.delivery.model_contracts import (
    TaskDeliveryValidationStructuredProvider,
)
from personagraph.l2.task_execution.delivery.candidate_gate import (
    TaskDeliveryCandidateAuthority,
    TaskDeliveryCandidateModelAuthorityFactory,
    build_task_delivery_candidate_gate_factory,
    create_task_delivery_candidate_model_call_authority,
)
from personagraph.l2.task_graph import (
    TaskDeliveryValidationDisposition,
)


class AuxiliaryTaskDeliveryStatus(StrEnum):
    """提交图执行的外部有意义的停止。"""

    DELIVERY_READY = "delivery_ready"
    WAITING_USER = "waiting_user"
    WAITING_EXTERNAL = "waiting_external"
    TURN_LIMIT = "turn_limit"
    REVISION_REQUIRED = "revision_required"
    BLOCKED = "blocked"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class AuxiliaryTaskDeliveryRequest:
    session_id: str
    turn_id: str
    task_id: str
    deadline: TurnDeadline | None = None

    def __post_init__(self) -> None:
        for name in ("session_id", "turn_id", "task_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value or len(value) > 200:
                raise ValueError(f"{name} must be a 1..200 character identity")


def _production_task_graph_profile() -> TaskGraphWorkRunProfile:
    """将现有的驱动器限制命名为此生产组合的 ABI。"""

    profile = TaskGraphWorkRunProfile()
    attempt_dependencies = profile.attempt_input_limits.dependency_delivery_limits
    verification_dependencies = (
        profile.verification_input_limits.dependency_delivery_limits
    )
    return profile.model_copy(
        update={
            "attempt_max_output_tokens": L2_MAX_OUTPUT_TOKENS,
            "verification_max_output_tokens": L2_MAX_OUTPUT_TOKENS,
            "model_timeout_s": 180.0,
            "attempt_input_limits": profile.attempt_input_limits.model_copy(
                update={
                    "profile_id": "auxiliary-v2-task-delivery-attempt-v1",
                    "dependency_delivery_limits": (
                        attempt_dependencies.model_copy(
                            update={
                                "profile_id": (
                                    "auxiliary-v2-task-delivery-"
                                    "attempt-dependencies-v1"
                                )
                            }
                        )
                    ),
                }
            ),
            "verification_input_limits": (
                profile.verification_input_limits.model_copy(
                    update={
                        "profile_id": (
                            "auxiliary-v2-task-delivery-verification-v1"
                        ),
                        "dependency_delivery_limits": (
                            verification_dependencies.model_copy(
                                update={
                                    "profile_id": (
                                        "auxiliary-v2-task-delivery-"
                                        "verification-dependencies-v1"
                                    )
                                }
                            )
                        ),
                    }
                )
            ),
        }
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class AuxiliaryTaskDeliveryPorts:
    """物理端口对应现有的 TaskGraph/WorkRun Runtime。"""

    emit: Callable[[TurnEvent], object]
    model_ledger_store: RuntimeModelLedgerStore
    allow_user_input: bool = True
    monotonic_clock: Callable[[], float] = time.monotonic
    profile: TaskGraphWorkRunProfile = field(
        default_factory=_production_task_graph_profile
    )
    catalog_snapshot: CatalogSnapshot | None = None
    attempt_provider: AttemptDecisionStructuredProvider | None = None
    verification_provider: NodeVerificationStructuredProvider | None = None
    tool_bridge: AttemptToolBridge | None = None
    node_tool_runtime_factory: TaskNodeToolRuntimeFactory | None = None
    task_node_model_call_authority_factory: (
        TaskNodeModelCallAuthorityFactory | None
    ) = None
    task_candidate_validation_provider: (
        TaskDeliveryValidationStructuredProvider | None
    ) = None
    task_candidate_validation_model_call_authority_factory: (
        TaskDeliveryCandidateModelAuthorityFactory | None
    ) = None

    def __post_init__(self) -> None:
        if not callable(self.emit):
            raise TypeError("emit must be callable")
        if type(self.allow_user_input) is not bool:
            raise TypeError("allow_user_input must be a boolean")
        if not callable(self.monotonic_clock):
            raise TypeError("monotonic_clock must be callable")
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
        if self.node_tool_runtime_factory is not None and not callable(
            self.node_tool_runtime_factory
        ):
            raise TypeError("node_tool_runtime_factory must be callable")
        for name in (
            "task_candidate_validation_provider",
            "task_node_model_call_authority_factory",
            "task_candidate_validation_model_call_authority_factory",
        ):
            value = getattr(self, name)
            if value is not None and not callable(value):
                raise TypeError(f"{name} must be callable")


@dataclass(frozen=True, slots=True)
class AuxiliaryTaskDeliveryResult:
    status: AuxiliaryTaskDeliveryStatus
    reason_code: str
    window_state_version: int | None = None
    final_delivery_id: str | None = None
    publication_body: str | None = None
    publication_format: str | None = None
    work_run_ids: tuple[str, ...] = ()
    pending_question_attempt_ids: tuple[str, ...] = ()
    requested_user_questions: tuple[str, ...] = ()
    replayed_delivery: bool = False
    task_graph_result: TaskGraphWorkRunResult | None = None
    task_candidate_settlement: (
        task_delivery_store.StoredTaskDeliveryCandidateSettlement | None
    ) = None

    def __post_init__(self) -> None:
        if not self.reason_code or len(self.reason_code) > 240:
            raise ValueError("task delivery result requires a bounded reason code")
        if self.window_state_version is not None and (
            isinstance(self.window_state_version, bool)
            or self.window_state_version < 1
        ):
            raise ValueError("window_state_version must be positive")
        delivery_fields = (
            self.final_delivery_id,
            self.publication_body,
            self.publication_format,
        )
        if self.status is AuxiliaryTaskDeliveryStatus.DELIVERY_READY:
            if any(value is None for value in delivery_fields):
                raise ValueError("DELIVERY_READY requires the verified body projection")
            if not self.publication_body or not self.publication_body.strip():
                raise ValueError("a publishable Delivery body must be non-empty")
        elif any(value is not None for value in delivery_fields):
            raise ValueError("only DELIVERY_READY may expose publication material")
        if self.replayed_delivery and (
            self.status is not AuxiliaryTaskDeliveryStatus.DELIVERY_READY
        ):
            raise ValueError("only a ready Delivery can be replayed")
        if len(self.work_run_ids) != len(set(self.work_run_ids)):
            raise ValueError("WorkRun IDs must be unique")
        if len(self.pending_question_attempt_ids) != len(
            set(self.pending_question_attempt_ids)
        ):
            raise ValueError("pending question Attempt IDs must be unique")
        if len(self.requested_user_questions) != len(
            set(self.requested_user_questions)
        ) or any(not item.strip() for item in self.requested_user_questions):
            raise ValueError("requested user questions must be unique non-empty text")
        if self.requested_user_questions and (
            self.status is not AuxiliaryTaskDeliveryStatus.WAITING_USER
        ):
            raise ValueError("only WAITING_USER may expose validation questions")


class AuxiliaryTaskDeliveryCompositionError(RuntimeError):
    """适配器无法保存提交的图或 Delivery 的权威状态。"""

    code = "auxiliary_v2_task_delivery_composition_rejected"


def _task_node_model_call_authority_factory(
    ports: AuxiliaryTaskDeliveryPorts,
) -> TaskNodeModelCallAuthorityFactory | None:
    """默认仅配置的生产提供者进行持久化调用。"""

    if ports.task_node_model_call_authority_factory is not None:
        return ports.task_node_model_call_authority_factory
    if ports.attempt_provider is None and ports.verification_provider is None:

        def factory(
            binding: TaskNodeBoundModelCall,
            *,
            rederive_state_guard_sha256: Callable[[], str],
        ):
            return create_task_node_work_run_model_call_authority(
                binding,
                rederive_state_guard_sha256=rederive_state_guard_sha256,
                ledger_store=ports.model_ledger_store,
            )

        return factory
    # 自定义 Provider 可能报告不同的提供者/模型事实。它必须
    # 明确提供匹配的持久化工厂，而不是被静默地标记为配置的结构网关。
    return None


def _task_candidate_model_call_authority_factory(
    ports: AuxiliaryTaskDeliveryPorts,
) -> TaskDeliveryCandidateModelAuthorityFactory:
    """为默认整 Task 审查器绑定与 TaskNode 相同的模型账本。"""

    if ports.task_candidate_validation_model_call_authority_factory is not None:
        return ports.task_candidate_validation_model_call_authority_factory

    def factory(
        authority: TaskDeliveryCandidateAuthority,
        *,
        rederive_state_guard_sha256: Callable[[], str],
    ):
        return create_task_delivery_candidate_model_call_authority(
            authority,
            rederive_state_guard_sha256=rederive_state_guard_sha256,
            ledger_store=ports.model_ledger_store,
        )

    return factory


def run_auxiliary_committed_task_to_delivery(
    request: AuxiliaryTaskDeliveryRequest,
    *,
    ports: AuxiliaryTaskDeliveryPorts,
) -> AuxiliaryTaskDeliveryResult:
    """推进一个已提交的 TaskGraph 到一个验证过的根 Delivery。

    该函数故意不进行初始化、重新规划、发布转录或结算 Turn。 每一步的修改都保留在现有的 TaskGraph/WorkRun 控制器中。
    """

    if not isinstance(request, AuxiliaryTaskDeliveryRequest):
        raise TypeError("request must be AuxiliaryTaskDeliveryRequest")
    if not isinstance(ports, AuxiliaryTaskDeliveryPorts):
        raise TypeError("ports must be AuxiliaryTaskDeliveryPorts")

    committed_revision = _validate_committed_current_revision(request)
    if isinstance(committed_revision, AuxiliaryTaskDeliveryResult):
        return committed_revision

    task = task_graph_store.get_insession_task_details(
        request.session_id,
        request.task_id,
    )
    if task is None:
        return _failed("task_missing")
    if task.current_graph_revision != committed_revision:
        return _failed("committed_task_graph_revision_mismatch")
    try:
        candidate_settlement = task_delivery_store.get_task_delivery_candidate_settlement(
            session_id=request.session_id,
            task_id=request.task_id,
            graph_revision=committed_revision,
        )
    except Exception as exc:
        return _failed(
            f"task_delivery_candidate_settlement_unreadable:{type(exc).__name__}"
        )
    if candidate_settlement is not None:
        return _project_candidate_settlement(
            request,
            stored=candidate_settlement,
            window_state_version=_optional_current_window_revision(request),
            work_run_ids=(),
            task_graph_result=None,
            replayed=True,
        )
    if task.status.value == "active":
        try:
            execution_replan = (
                work_run_store.get_active_task_graph_execution_replan_request(
                    session_id=request.session_id,
                    task_id=request.task_id,
                )
            )
            revision_trigger = task_delivery_store.get_active_task_graph_revision_trigger(
                session_id=request.session_id,
                task_id=request.task_id,
            )
        except Exception as exc:
            return _failed(
                f"task_graph_revision_trigger_unreadable:{type(exc).__name__}"
            )
        if execution_replan is not None and revision_trigger is not None:
            return _failed("multiple_task_graph_revision_authorities")
        if execution_replan is not None:
            if (
                execution_replan.base_graph_revision != committed_revision
                or execution_replan.target_graph_revision
                != committed_revision + 1
            ):
                return _failed("execution_replan_lineage_mismatch")
            return AuxiliaryTaskDeliveryResult(
                status=AuxiliaryTaskDeliveryStatus.REVISION_REQUIRED,
                reason_code="task_node_requested_task_graph_revision",
                window_state_version=_optional_current_window_revision(request),
                replayed_delivery=False,
            )
        if revision_trigger is not None:
            if (
                revision_trigger.base_graph_revision != committed_revision
                or revision_trigger.target_graph_revision
                != committed_revision + 1
            ):
                return _failed("task_graph_revision_trigger_lineage_mismatch")
            return _failed(
                "task_delivery_candidate_settlement_missing_for_revision_trigger",
                window_state_version=_optional_current_window_revision(request),
            )
    if task.status.value == "completed":
        return _failed(
            "task_delivery_candidate_settlement_missing_for_completed_task",
            window_state_version=_optional_current_window_revision(request),
        )
    if task.status.value == "cancelled":
        return _failed("task_cancelled")

    window = store.get_turn_execution_window(request.session_id)
    if window is None:
        return _failed("active_turn_window_unavailable")
    revision = window.get("state_version")
    if (
        window.get("turn_id") != request.turn_id
        or window.get("window_state") != "active"
        or isinstance(revision, bool)
        or not isinstance(revision, int)
        or revision < 1
    ):
        return _failed("active_turn_window_authority_mismatch")

    try:
        graph_result = run_task_graph_work_runs(
            TaskGraphWorkRunRequest(
                session_id=request.session_id,
                turn_id=request.turn_id,
                task_id=request.task_id,
                expected_window_revision=revision,
                profile=ports.profile,
                allow_user_input=ports.allow_user_input,
            ),
            monotonic_clock=ports.monotonic_clock,
            catalog_snapshot=ports.catalog_snapshot,
            attempt_provider=ports.attempt_provider,
            verification_provider=ports.verification_provider,
            emit=ports.emit,
            tool_bridge=ports.tool_bridge,
            node_tool_runtime_factory=ports.node_tool_runtime_factory,
            model_call_authority_factory=(
                _task_node_model_call_authority_factory(ports)
            ),
            downstream_gate_factory=build_task_delivery_candidate_gate_factory(
                provider=ports.task_candidate_validation_provider,
                emit=ports.emit,
                model_call_authority_factory=(
                    _task_candidate_model_call_authority_factory(ports)
                ),
                deadline=request.deadline,
            ),
            deadline=request.deadline,
        )
    except ContextBudgetExceeded:
        raise
    except Exception as exc:
        return _failed(
            f"task_graph_execution_interrupted:{type(exc).__name__}",
            window_state_version=_optional_current_window_revision(request),
        )

    try:
        candidate_settlement = task_delivery_store.get_task_delivery_candidate_settlement(
            session_id=request.session_id,
            task_id=request.task_id,
            graph_revision=committed_revision,
        )
    except Exception as exc:
        return _failed(
            f"task_delivery_candidate_settlement_unreadable:{type(exc).__name__}",
            window_state_version=graph_result.window_state_version,
            work_run_ids=graph_result.work_run_ids,
            task_graph_result=graph_result,
        )
    if candidate_settlement is not None:
        return _project_candidate_settlement(
            request,
            stored=candidate_settlement,
            window_state_version=graph_result.window_state_version,
            work_run_ids=graph_result.work_run_ids,
            task_graph_result=graph_result,
            replayed=False,
        )

    if graph_result.status == "completed":
        return _failed(
            "task_delivery_candidate_settlement_missing_after_graph_completion",
            window_state_version=graph_result.window_state_version,
            work_run_ids=graph_result.work_run_ids,
            task_graph_result=graph_result,
        )
    return _project_stop(request, graph_result)


def _project_candidate_settlement(
    request: AuxiliaryTaskDeliveryRequest,
    *,
    stored: task_delivery_store.StoredTaskDeliveryCandidateSettlement,
    window_state_version: int | None,
    work_run_ids: tuple[str, ...],
    task_graph_result: TaskGraphWorkRunResult | None,
    replayed: bool,
) -> AuxiliaryTaskDeliveryResult:
    settlement = stored.settlement
    if (
        settlement.session_id != request.session_id
        or settlement.task_id != request.task_id
    ):
        return _failed(
            "task_delivery_candidate_settlement_authority_mismatch",
            window_state_version=window_state_version,
            work_run_ids=work_run_ids,
            task_graph_result=task_graph_result,
        )
    if settlement.disposition is TaskDeliveryValidationDisposition.PASS:
        try:
            task = task_graph_store.get_insession_task_details(
                request.session_id,
                request.task_id,
            )
            resolved = verification_store.get_task_node_delivery(
                session_id=request.session_id,
                delivery_id=settlement.root_delivery_id,
            )
            if (
                task is None
                or task.status.value != "completed"
                or task.current_graph_revision != settlement.graph_revision
                or task.task_state_version
                != settlement.completed_task_state_version
                or resolved.delivery.subject != stored.intent.subject
            ):
                raise AuxiliaryTaskDeliveryCompositionError(
                    "candidate PASS is not the exact completed root"
                )
        except Exception as exc:
            return _failed(
                f"candidate_pass_delivery_unavailable:{type(exc).__name__}",
                window_state_version=window_state_version,
                work_run_ids=work_run_ids,
                task_graph_result=task_graph_result,
            )
        output = resolved.output_window
        return AuxiliaryTaskDeliveryResult(
            status=AuxiliaryTaskDeliveryStatus.DELIVERY_READY,
            reason_code=(
                "whole_task_candidate_pass_replayed"
                if replayed
                else "whole_task_candidate_pass"
            ),
            window_state_version=window_state_version,
            final_delivery_id=settlement.root_delivery_id,
            publication_body=output.content,
            publication_format=output.format.value,
            work_run_ids=work_run_ids,
            replayed_delivery=replayed,
            task_graph_result=task_graph_result,
            task_candidate_settlement=stored,
        )
    if (
        settlement.disposition
        is TaskDeliveryValidationDisposition.REPLAN_TASK_GRAPH
    ):
        trigger = stored.trigger
        if (
            trigger is None
            or trigger.base_graph_revision != settlement.graph_revision
            or trigger.target_graph_revision != settlement.graph_revision + 1
            or trigger.root_delivery_id != settlement.root_delivery_id
        ):
            return _failed(
                "task_delivery_candidate_trigger_mismatch",
                window_state_version=window_state_version,
                work_run_ids=work_run_ids,
                task_graph_result=task_graph_result,
            )
        return AuxiliaryTaskDeliveryResult(
            status=AuxiliaryTaskDeliveryStatus.REVISION_REQUIRED,
            reason_code=(
                "whole_task_candidate_replan_task_graph_replayed"
                if replayed
                else "whole_task_candidate_replan_task_graph"
            ),
            window_state_version=window_state_version,
            work_run_ids=work_run_ids,
            task_graph_result=task_graph_result,
            task_candidate_settlement=stored,
        )
    if settlement.disposition is TaskDeliveryValidationDisposition.BLOCKED:
        questions = stored.intent.result.blocking_questions
        trigger = stored.trigger
        if (
            not questions
            or trigger is None
            or trigger.base_graph_revision != settlement.graph_revision
            or trigger.target_graph_revision != settlement.graph_revision + 1
            or trigger.root_delivery_id != settlement.root_delivery_id
        ):
            return _failed(
                "task_delivery_candidate_blocked_trigger_mismatch",
                window_state_version=window_state_version,
                work_run_ids=work_run_ids,
                task_graph_result=task_graph_result,
            )
        return AuxiliaryTaskDeliveryResult(
            status=AuxiliaryTaskDeliveryStatus.REVISION_REQUIRED,
            reason_code=(
                "whole_task_candidate_blocked_replan_replayed"
                if replayed
                else "whole_task_candidate_blocked_replan"
            ),
            window_state_version=window_state_version,
            work_run_ids=work_run_ids,
            task_graph_result=task_graph_result,
            task_candidate_settlement=stored,
        )
    return _failed(
        "task_delivery_candidate_settlement_route_unsupported",
        window_state_version=window_state_version,
        work_run_ids=work_run_ids,
        task_graph_result=task_graph_result,
    )


def _validate_committed_current_revision(
    request: AuxiliaryTaskDeliveryRequest,
) -> int | AuxiliaryTaskDeliveryResult:
    try:
        auxiliary = auxiliary_graph_store.get_auxiliary_graph_for_task(
            session_id=request.session_id,
            insession_task_id=request.task_id,
        )
        task = task_graph_store.get_insession_task_details(
            request.session_id,
            request.task_id,
        )
    except Exception as exc:
        return _failed(f"committed_authority_unreadable:{type(exc).__name__}")
    if auxiliary is None:
        return _failed("committed_auxiliary_v2_authority_missing")
    expected_target_revision = (
        1
        if auxiliary.base_task_graph_revision is None
        else auxiliary.base_task_graph_revision + 1
    )
    if (
        auxiliary.target_task_graph_revision != expected_target_revision
        or auxiliary.goal_status != "committed"
        or auxiliary.revision_status != "committed"
    ):
        return _failed("unsupported_auxiliary_v2_commit_lineage")
    if (
        task is None
        or task.current_graph_revision != auxiliary.target_task_graph_revision
    ):
        return _failed("committed_task_graph_revision_mismatch")
    return auxiliary.target_task_graph_revision


def _project_stop(
    request: AuxiliaryTaskDeliveryRequest,
    result: TaskGraphWorkRunResult,
) -> AuxiliaryTaskDeliveryResult:
    status = {
        "revision_required": AuxiliaryTaskDeliveryStatus.REVISION_REQUIRED,
        "waiting_user": AuxiliaryTaskDeliveryStatus.WAITING_USER,
        "waiting_external": AuxiliaryTaskDeliveryStatus.WAITING_EXTERNAL,
        "turn_limit_reached": AuxiliaryTaskDeliveryStatus.TURN_LIMIT,
        "deadline_reached": AuxiliaryTaskDeliveryStatus.TURN_LIMIT,
    }.get(result.status, AuxiliaryTaskDeliveryStatus.FAILED)
    reason = _task_graph_stop_reason(result)
    requested_user_questions: tuple[str, ...] = ()
    if status is AuxiliaryTaskDeliveryStatus.WAITING_USER:
        requested_user_questions = _resolve_pending_user_questions(
            request,
            result,
        )
        if not requested_user_questions:
            return _failed(
                "task_graph_pending_question_authority_unavailable",
                window_state_version=result.window_state_version,
                work_run_ids=result.work_run_ids,
                task_graph_result=result,
            )
    return AuxiliaryTaskDeliveryResult(
        status=status,
        reason_code=reason,
        window_state_version=result.window_state_version,
        work_run_ids=result.work_run_ids,
        pending_question_attempt_ids=result.pending_question_attempt_ids,
        requested_user_questions=requested_user_questions,
        task_graph_result=result,
    )


def _resolve_pending_user_questions(
    request: AuxiliaryTaskDeliveryRequest,
    result: TaskGraphWorkRunResult,
) -> tuple[str, ...]:
    """从持久化的 Attempt 权威状态重新加载精确的问题文本。"""

    expected_ids = result.pending_question_attempt_ids
    if not expected_ids:
        return ()
    try:
        questions_by_id: dict[str, str] = {}
        for item in continuation_store.list_pending_user_questions(
            session_id=request.session_id
        ):
            question_attempt_id = getattr(item, "question_attempt_id", None)
            if question_attempt_id not in expected_ids:
                continue
            subject = getattr(item, "subject", None)
            question = getattr(item, "question", None)
            if (
                getattr(subject, "task_id", None) != request.task_id
                or result.task_id != request.task_id
                or not isinstance(question, str)
                or not question.strip()
                or question_attempt_id in questions_by_id
            ):
                return ()
            questions_by_id[question_attempt_id] = question
    except Exception:
        return ()
    if set(questions_by_id) != set(expected_ids):
        return ()
    return tuple(questions_by_id[item] for item in expected_ids)


def _task_graph_stop_reason(result: TaskGraphWorkRunResult) -> str:
    if result.failure_code:
        return result.failure_code
    if result.interruption_reason:
        return result.interruption_reason
    if result.last_work_run_outcome:
        return result.last_work_run_outcome
    return f"task_graph_{result.status}"


def _optional_current_window_revision(
    request: AuxiliaryTaskDeliveryRequest,
) -> int | None:
    try:
        window = store.get_turn_execution_window(request.session_id)
    except Exception:
        return None
    if window is None or window.get("turn_id") != request.turn_id:
        return None
    revision = window.get("state_version")
    if isinstance(revision, bool) or not isinstance(revision, int) or revision < 1:
        return None
    return revision


def _failed(
    reason_code: str,
    *,
    window_state_version: int | None = None,
    work_run_ids: tuple[str, ...] = (),
    task_graph_result: TaskGraphWorkRunResult | None = None,
) -> AuxiliaryTaskDeliveryResult:
    return AuxiliaryTaskDeliveryResult(
        status=AuxiliaryTaskDeliveryStatus.FAILED,
        reason_code=reason_code[:240],
        window_state_version=window_state_version,
        work_run_ids=work_run_ids,
        task_graph_result=task_graph_result,
    )


__all__ = [
    "AuxiliaryTaskDeliveryCompositionError",
    "AuxiliaryTaskDeliveryPorts",
    "AuxiliaryTaskDeliveryRequest",
    "AuxiliaryTaskDeliveryResult",
    "AuxiliaryTaskDeliveryStatus",
    "run_auxiliary_committed_task_to_delivery",
]
