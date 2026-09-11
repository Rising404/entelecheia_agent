"""基于已启动的单一 WorkRun Attempt 的有界应用控制器。

此切片执行精确的一个逻辑 Attempt 决策和精确的一个动作分发。它故意不启动一个 Attempt，恢复或移交一个 Turn，执行一个实际的工具，运行节点验证器，发布交付物，或垃圾回收终端执行状态。

调用者提供一个可信的 :class:`AttemptDecisionContext`，当前 TurnExecutionWindow 版本和一个稳定的应用 ID。在花费模型调用之前，控制器重新读取 Store 权威状态并拒绝过时或跨边界上下文。在那个读取之后发生的竞争仍然由 Store 命令的原子 CAS 拒绝。
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from ....session.l2_store import work_run as work_run_store
from ...work_run import (
    AttemptDecision,
    CallToolsAction,
    HostAcceptedAttemptDecision,
    HostMaterializedCallToolsAction,
    RequestUserInputAction,
    SubmitOutputWindowAction,
    WriteOutputWindowAction,
    WorkExecutionMutationResult,
)
from ...work_run.contracts import RequestTaskGraphRevisionAction
from .active_time import (
    AttemptActiveTimeMeasurementError,
    AttemptActiveTimeMeter,
)
from .context_authority import (
    AttemptControllerStateConflict,
    project_authoritative_attempt_user_input,
    require_fresh_attempt_context,
)
from ..tool_bridge.attempt_contracts import (
    AttemptToolBridgePreflightRequest,
    AttemptToolBridgeRequest,
)
from .input_projection import AttemptDecisionContext
from .decision import (
    AttemptDecisionStructuredProvider,
    request_attempt_decision,
)
from ....runtime.turn_deadline import TurnDeadline
from ....model_io.output_validation import ModelOutputValidationError
from ..task_node.model_authority_contracts import TaskNodeModelCallPlan
from ..tool_bridge.contracts import AttemptToolBridge
from ....runtime.turn_events import TurnEvent


AttemptControllerActionKind = Literal[
    "call_tools",
    "write_output_window",
    "submit_output_window",
    "request_user_input",
    "request_task_graph_revision",
]


class _ControllerContract(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        arbitrary_types_allowed=True,
    )


class AttemptControllerConfigurationError(RuntimeError):
    """宿主机 Host 没有提供执行所需的一个端口或稳定的标识。"""


ToolCallIdFactory = Callable[[str, int], str]


class AttemptControllerModelCall(_ControllerContract):
    model_call_id: str = Field(min_length=1)
    provider: str
    model: str
    physical_attempts: int = Field(ge=1)
    latency_ms: int = Field(ge=0)


class AttemptControllerResult(_ControllerContract):
    """小型结果；它从不嵌入模型回复或 OutputWindow 体。"""

    action: AttemptControllerActionKind
    apply_id: str = Field(min_length=1, max_length=200)
    mutation: WorkExecutionMutationResult
    model_call: AttemptControllerModelCall


def run_started_attempt(
    context: AttemptDecisionContext,
    *,
    expected_window_revision: int,
    apply_id: str,
    provider: AttemptDecisionStructuredProvider,
    emit: Callable[[TurnEvent], object],
    active_time_meter: AttemptActiveTimeMeter,
    tool_bridge: AttemptToolBridge | None = None,
    tool_call_id_factory: ToolCallIdFactory | None = None,
    deadline: TurnDeadline | None = None,
    runtime_model_call_plan: TaskNodeModelCallPlan | None = None,
) -> AttemptControllerResult:
    """为当前确切活动的 Attempt 制定并分发一个决策。

    Provider 或类型输出耗尽故意不会导致 WorkRun 写入： 已经启动的 Attempt 保持活动状态，以便后续恢复控制器 可以检查它。此函数本身不实现 Turn 的移交或恢复。
    """

    _require_positive_revision("expected_window_revision", expected_window_revision)
    _require_apply_id(apply_id)
    stored = require_fresh_attempt_context(
        context,
        expected_window_revision=expected_window_revision,
    )

    current_attempt = next(
        item
        for item in stored.attempts
        if item.attempt.attempt_id == context.attempt_id
    )
    planned_tool_call_ids: dict[int, str] = {}
    materialized_tool_decision: HostAcceptedAttemptDecision | None = None

    def validate_proposal(decision: AttemptDecision) -> None:
        nonlocal materialized_tool_decision
        if not isinstance(decision.action, CallToolsAction):
            materialized_tool_decision = None
            return
        if tool_bridge is None or tool_call_id_factory is None:
            raise AttemptControllerConfigurationError(
                "call_tools requires a Tool Bridge and stable tool_call_id factory"
            )
        call_ids = tuple(
            _planned_tool_call_id(
                planned_tool_call_ids,
                attempt_id=context.attempt_id,
                ordinal=ordinal,
                factory=tool_call_id_factory,
            )
            for ordinal in range(1, len(decision.action.calls) + 1)
        )
        accepted = tool_bridge.preflight(
            AttemptToolBridgePreflightRequest(
                session_id=context.session_id,
                turn_id=context.turn_id,
                work_run_id=context.work_run_id,
                attempt_id=context.attempt_id,
                decision=decision,
                tool_call_ids=call_ids,
                allowed_tools=context.allowed_tools,
                catalog_snapshot=current_attempt.catalog_snapshot,
            )
        )
        if not isinstance(accepted, HostAcceptedAttemptDecision) or (
            accepted.acceptance_updates != decision.acceptance_updates
            or not isinstance(accepted.action, HostMaterializedCallToolsAction)
            or tuple(call.tool_call_id for call in accepted.action.calls) != call_ids
            or any(
                materialized.tool_id != proposed.tool_id
                or materialized.arguments != proposed.arguments
                for materialized, proposed in zip(
                    accepted.action.calls,
                    decision.action.calls,
                    strict=True,
                )
            )
        ):
            raise ModelOutputValidationError(
                "Tool Bridge returned a misbound Host-materialized decision"
            )
        materialized_tool_decision = accepted

    requested = request_attempt_decision(
        context,
        provider=provider,
        emit=emit,
        deadline=deadline,
        proposal_validator=validate_proposal,
        runtime_model_call_plan=runtime_model_call_plan,
    )
    decision = requested.value
    action = decision.action

    if isinstance(
        action,
        (
            WriteOutputWindowAction,
            SubmitOutputWindowAction,
        ),
    ):
        mutation = work_run_store.commit_work_run_output_action(
            session_id=context.session_id,
            turn_id=context.turn_id,
            work_run_id=context.work_run_id,
            attempt_id=context.attempt_id,
            decision=decision,
            expected_work_run_revision=context.work_run_revision,
            expected_progress_revision=context.acceptance_progress.revision,
            expected_output_revision=context.output_window.output_revision,
            expected_window_revision=expected_window_revision,
            apply_id=apply_id,
            active_seconds_delta=active_time_meter.freeze(),
        )
    elif isinstance(
        action,
        (RequestUserInputAction, RequestTaskGraphRevisionAction),
    ):
        materialized = HostAcceptedAttemptDecision(
            acceptance_updates=decision.acceptance_updates,
            action=action,
        )
        mutation = work_run_store.commit_work_run_attempt_decision(
            session_id=context.session_id,
            turn_id=context.turn_id,
            work_run_id=context.work_run_id,
            attempt_id=context.attempt_id,
            decision=materialized,
            expected_work_run_revision=context.work_run_revision,
            expected_progress_revision=context.acceptance_progress.revision,
            expected_window_revision=expected_window_revision,
            apply_id=apply_id,
            active_seconds_delta=active_time_meter.freeze(),
        )
    elif isinstance(action, CallToolsAction):
        if tool_bridge is None or materialized_tool_decision is None:
            raise AttemptControllerStateConflict(
                "call_tools decision has no successful Host preflight"
            )
        mutation = tool_bridge.dispatch(
            AttemptToolBridgeRequest(
                session_id=context.session_id,
                turn_id=context.turn_id,
                work_run_id=context.work_run_id,
                attempt_id=context.attempt_id,
                expected_work_run_revision=context.work_run_revision,
                expected_progress_revision=context.acceptance_progress.revision,
                expected_output_revision=context.output_window.output_revision,
                expected_window_revision=expected_window_revision,
                apply_id=apply_id,
                decision=materialized_tool_decision,
                allowed_tools=context.allowed_tools,
                catalog_snapshot=current_attempt.catalog_snapshot,
            ),
            active_time_meter=active_time_meter,
        )
    else:  # pragma: no cover - 这种情况不可能到达。
        raise TypeError(f"unsupported Attempt action: {type(action).__name__}")

    if mutation.work_run_id != context.work_run_id:
        raise AttemptControllerStateConflict(
            "action dispatch returned a result for another WorkRun"
        )

    model_result = requested.model_result
    return AttemptControllerResult(
        action=action.kind,
        apply_id=apply_id,
        mutation=mutation,
        model_call=AttemptControllerModelCall(
            model_call_id=requested.model_call_id,
            provider=model_result.provider,
            model=model_result.model,
            physical_attempts=requested.attempts,
            latency_ms=max(0, model_result.latency_ms),
        ),
    )


def _require_apply_id(value: str) -> None:
    if not isinstance(value, str) or not value.strip() or len(value) > 200:
        raise ValueError("apply_id must be a non-empty identifier of at most 200 characters")


def _require_positive_revision(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _require_tool_call_id(value: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 200:
        raise ValueError(
            "tool_call_id factory must return a non-empty identifier of at most 200 characters"
        )
    return value


def _planned_tool_call_id(
    planned: dict[int, str],
    *,
    attempt_id: str,
    ordinal: int,
    factory: ToolCallIdFactory,
) -> str:
    existing = planned.get(ordinal)
    if existing is not None:
        return existing
    allocated = _require_tool_call_id(factory(attempt_id, ordinal))
    planned[ordinal] = allocated
    return allocated

__all__ = [
    "AttemptActiveTimeMeasurementError",
    "AttemptActiveTimeMeter",
    'AttemptControllerModelCall',
    'AttemptControllerResult',
    "AttemptControllerConfigurationError",
    "AttemptControllerStateConflict",
    "AttemptToolBridge",
    'AttemptToolBridgePreflightRequest',
    'AttemptToolBridgeRequest',
    "ToolCallIdFactory",
    "project_authoritative_attempt_user_input",
    "run_started_attempt",
]
