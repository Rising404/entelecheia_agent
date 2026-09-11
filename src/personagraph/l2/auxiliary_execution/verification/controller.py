"""InSession、TaskGraph：持久化语义审查调度在 终端完成之后。

此控制器冻结并验证审查员的权威状态，驱动现有的结构化语义验证端口，并达成一个不可变的一/两审查员共识。它故意不对终端提案进行密封，并且从不提交 InSessionTaskGraph。
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from dataclasses import dataclass, field
from enum import StrEnum
from queue import Queue
from threading import Event as ThreadEvent
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from personagraph.l2.auxiliary_graph import (
    AuxiliaryNodeExecutorKind,
    AuxiliaryPlanningGoalStatus,
    PlanningCapabilityCatalogProjection,
    TaskGraphSemanticTerminalRoute,
    TaskGraphSemanticTerminalCandidateBinding,
    TaskGraphSemanticVerificationPromptPayload,
    TaskGraphSemanticVerificationRequest,
    TaskGraphSemanticVerificationResult,
    derive_task_graph_semantic_terminal_route,
    required_task_graph_semantic_reviewer_count,
    validate_task_graph_semantic_verification_quorum,
    validate_task_graph_semantic_verification_result,
)
from personagraph.l2.auxiliary_graph.execution_frontier_contracts import (
    AuxiliaryGraphExecutionFrontier,
)
from personagraph.model_io.gateway import ModelGatewayError
from personagraph.session import store as session_store
from personagraph.session.l2_store import verification as verification_store
from personagraph.session.l2_store import auxiliary_graph as auxiliary_graph_store
from personagraph.session.l2_store import planning as planning_store
from personagraph.session.l2_store import semantic_verification as semantic_store
from personagraph.session.l2_store.semantic_verification import (
    StoredAuxiliarySemanticQuorumSettlement,
)
from personagraph.l2.work_run import WorkRunStatus
from .model_binding_contracts import (
    AuxiliarySemanticReviewerModelCallBinding,
    _REQUEST_CONTRACT,
    _canonical_json,
    _sha256_text,
    _sha256_value,
)
from personagraph.runtime.model_calls.contracts import (
    DurableLogicalModelCallAuthority,
    DurableModelCallStateGuardRejected,
    DurableModelCallTerminalState,
)
from personagraph.runtime.turn_deadline import (
    TurnDeadline,
    TurnDeadlineExceeded,
)
from personagraph.runtime.model_calls.authority import (
    RuntimeLogicalModelCallAuthority,
    RuntimeModelCallWaitingExternal,
)
from .task_graph_semantic import (
    TaskGraphSemanticVerificationInputTooLarge,
    TaskGraphSemanticVerificationInputUnsupported,
    TaskGraphSemanticVerificationStructuredProvider,
    request_task_graph_semantic_verification,
    serialize_task_graph_semantic_verification_prompt,
)
from personagraph.runtime.turn_events import TurnEvent


_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,199}$"


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class AuxiliarySemanticReviewerIdPlan(_Contract):
    """一个独立审查员调用的稳定不变的身份标识。"""

    reviewer_ordinal: int = Field(ge=1, le=2)
    verification_profile_id: str = Field(pattern=_ID_PATTERN)
    verification_request_id: str = Field(pattern=_ID_PATTERN)
    logical_call_id: str = Field(pattern=_ID_PATTERN)
    verification_result_id: str = Field(pattern=_ID_PATTERN)


AuxiliaryTerminalCandidateBinding = (
    TaskGraphSemanticTerminalCandidateBinding
)


class AuxiliarySemanticVerificationRequest(_Contract):
    """精确的当前前沿、提示权威状态和调用者拥有的身份标识。"""

    schema_version: Literal[
        "auxiliary-v2-semantic-verification-controller-request-v1"
    ] = "auxiliary-v2-semantic-verification-controller-request-v1"
    frontier: AuxiliaryGraphExecutionFrontier
    prompt_payload: TaskGraphSemanticVerificationPromptPayload
    capability_catalog: PlanningCapabilityCatalogProjection
    reviewers: tuple[AuxiliarySemanticReviewerIdPlan, ...] = Field(
        min_length=1,
        max_length=2,
    )
    settlement_id: str = Field(pattern=_ID_PATTERN)
    terminal_candidate_binding: AuxiliaryTerminalCandidateBinding | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )

    @model_validator(mode="after")
    def _validate_identity_plan(self) -> "AuxiliarySemanticVerificationRequest":
        expected_ordinals = tuple(range(1, len(self.reviewers) + 1))
        if tuple(item.reviewer_ordinal for item in self.reviewers) != expected_ordinals:
            raise ValueError("semantic reviewer identity plan must be ordinal ordered")
        for values, label in (
            (
                tuple(item.verification_request_id for item in self.reviewers),
                "verification request IDs",
            ),
            (
                tuple(item.logical_call_id for item in self.reviewers),
                "logical call IDs",
            ),
            (
                tuple(item.verification_result_id for item in self.reviewers),
                "verification result IDs",
            ),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"semantic reviewer plan reused {label}")
        if len({item.verification_profile_id for item in self.reviewers}) != 1:
            raise ValueError(
                "semantic reviewers must share one frozen verification profile"
            )
        return self


class AuxiliarySemanticModelCallAuthorityFactory(Protocol):
    def __call__(
        self,
        binding: AuxiliarySemanticReviewerModelCallBinding,
        *,
        rederive_state_guard_sha256: Callable[[], str],
    ) -> DurableLogicalModelCallAuthority: ...


class AuxiliarySemanticVerificationStatus(StrEnum):
    SETTLED = "settled"
    WAITING_EXTERNAL = "waiting_external"
    TURN_LIMIT_REACHED = "turn_limit_reached"
    MODEL_INTERRUPTED = "model_interrupted"
    FAILED_CLOSED = "failed_closed"


class AuxiliarySemanticVerificationResult(_Contract):
    schema_version: Literal[
        "auxiliary-v2-semantic-verification-controller-result-v1"
    ] = "auxiliary-v2-semantic-verification-controller-result-v1"
    status: AuxiliarySemanticVerificationStatus
    reason_code: str = Field(pattern=_ID_PATTERN)
    session_id: str = Field(pattern=_ID_PATTERN)
    task_id: str = Field(pattern=_ID_PATTERN)
    auxiliary_graph_id: str = Field(pattern=_ID_PATTERN)
    goal_id: str = Field(pattern=_ID_PATTERN)
    auxiliary_graph_revision: int = Field(ge=1)
    required_reviewer_count: int = Field(ge=1, le=2)
    completed_reviewer_ordinals: tuple[int, ...] = Field(max_length=2)
    pending_reviewer_ordinal: int | None = Field(default=None, ge=1, le=2)
    settlement: StoredAuxiliarySemanticQuorumSettlement | None = None
    replayed: bool = False

    @model_validator(mode="after")
    def _validate_status_projection(self) -> "AuxiliarySemanticVerificationResult":
        settled = self.status is AuxiliarySemanticVerificationStatus.SETTLED
        if settled != (self.settlement is not None):
            raise ValueError("only a settled semantic result carries a settlement")
        if self.replayed and not settled:
            raise ValueError("only an immutable settlement can be replayed")
        if self.completed_reviewer_ordinals != tuple(
            range(1, len(self.completed_reviewer_ordinals) + 1)
        ):
            raise ValueError("completed semantic reviewer ordinals are not canonical")
        if any(
            item > self.required_reviewer_count
            for item in self.completed_reviewer_ordinals
        ):
            raise ValueError("completed semantic reviewer exceeds quorum size")
        if settled and (
            self.completed_reviewer_ordinals
            != tuple(range(1, self.required_reviewer_count + 1))
            or self.pending_reviewer_ordinal is not None
        ):
            raise ValueError("settled semantic result must cover its complete quorum")
        if (
            self.status is AuxiliarySemanticVerificationStatus.WAITING_EXTERNAL
        ) != (self.pending_reviewer_ordinal is not None):
            raise ValueError("waiting-external semantic result requires its reviewer")
        return self


class AuxiliaryTerminalCandidateSemanticReview(_Contract):
    """候选审核，其 Host 派生的路径选择下一个边界。"""

    route: TaskGraphSemanticTerminalRoute
    requests: tuple[TaskGraphSemanticVerificationRequest, ...] = Field(
        min_length=1,
        max_length=2,
    )
    results: tuple[TaskGraphSemanticVerificationResult, ...] = Field(
        min_length=1,
        max_length=2,
    )


class _FrontierAuthorityRejected(RuntimeError):
    pass


class _StoredSemanticAuthorityRejected(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class _SemanticReviewerInvocationPlan:
    reviewer: AuxiliarySemanticReviewerIdPlan
    semantic_request: TaskGraphSemanticVerificationRequest
    durable_call: DurableLogicalModelCallAuthority
    invocation_turn_id: str


@dataclass(frozen=True, slots=True)
class _SemanticReviewerInvocationOutcome:
    reviewer_ordinal: int
    result: TaskGraphSemanticVerificationResult | None
    events: tuple[TurnEvent, ...]
    error: BaseException | None


@dataclass(slots=True)
class _SemanticReviewerFirstEventHandoff:
    event: TurnEvent | None
    released: ThreadEvent = field(default_factory=ThreadEvent)
    error: BaseException | None = None


@dataclass(frozen=True, slots=True)
class _SemanticReviewerFutureDone:
    """当 Future 在完成首次事件移交前结束时，唤醒协调器。"""


_SEMANTIC_REVIEWER_FUTURE_DONE = _SemanticReviewerFutureDone()


@dataclass(slots=True)
class _SemanticReviewerCoordinatorAbort:
    """在扇出提交失败后，唤醒已启动或延迟取消队列的工作者。"""

    raised: ThreadEvent = field(default_factory=ThreadEvent)
    error: BaseException | None = None

    def fail(self, error: BaseException) -> None:
        self.error = error
        self.raised.set()


class _SemanticReviewerEventCapture:
    """发布一个实时有序事件，然后缓冲剩余的生命周期。"""

    def __init__(
        self,
        *,
        first_event_queue: Queue[
            _SemanticReviewerFirstEventHandoff | _SemanticReviewerFutureDone
        ]
        | None,
        direct_first_event_emit: Callable[[TurnEvent], object] | None,
        coordinator_abort: _SemanticReviewerCoordinatorAbort | None,
    ) -> None:
        if (first_event_queue is None) == (direct_first_event_emit is None):
            raise ValueError(
                "semantic event capture requires exactly one first-event port"
            )
        if (first_event_queue is None) != (coordinator_abort is None):
            raise ValueError(
                "only queued semantic event capture uses coordinator abort"
            )
        self._first_event_queue = first_event_queue
        self._direct_first_event_emit = direct_first_event_emit
        self._coordinator_abort = coordinator_abort
        self._first_event_handed_off = False
        self.remaining_events: list[TurnEvent] = []

    def __call__(self, event: TurnEvent) -> None:
        if self._first_event_handed_off:
            self.remaining_events.append(event)
            return
        self._first_event_handed_off = True
        if self._direct_first_event_emit is not None:
            self._direct_first_event_emit(event)
            return
        self._hand_off(event)

    def complete_without_first_event(self) -> None:
        """推进一个序号，其调用在发出生命周期之前失败。"""

        if self._first_event_handed_off:
            return
        self._first_event_handed_off = True
        if self._first_event_queue is not None:
            self._hand_off(None)

    def _hand_off(self, event: TurnEvent | None) -> None:
        assert self._first_event_queue is not None
        handoff = _SemanticReviewerFirstEventHandoff(event=event)
        self._first_event_queue.put(handoff)
        while not handoff.released.wait(timeout=0.05):
            coordinator_abort = self._coordinator_abort
            if coordinator_abort is not None and coordinator_abort.raised.is_set():
                error = coordinator_abort.error
                if error is None:  # pragma: no cover - 私有 DTO 不变量
                    raise RuntimeError("semantic reviewer coordinator aborted")
                raise error
        if handoff.error is not None:
            raise handoff.error


def review_auxiliary_terminal_candidate_semantics(
    request: AuxiliarySemanticVerificationRequest,
    *,
    provider: TaskGraphSemanticVerificationStructuredProvider,
    model_call_authority_factory: AuxiliarySemanticModelCallAuthorityFactory,
    emit: Callable[[TurnEvent], object],
    deadline: TurnDeadline | None = None,
) -> AuxiliaryTerminalCandidateSemanticReview:
    """在节点完成前审查一个确切的开放终端候选。

    终端局部失败仅保留在持久化模型日志中，直到节点验证器提交认证重试反馈。上游重新规划调用者可以保存来自同一重放审查调用的结算；通过的候选者在节点完成后进行结算。
    """

    if not isinstance(request, AuxiliarySemanticVerificationRequest):
        raise TypeError("request must be AuxiliarySemanticVerificationRequest")
    if request.terminal_candidate_binding is None:
        raise ValueError("terminal candidate review requires its frozen binding")
    semantic_requests = _prepare_semantic_requests(request)
    _freeze_catalog(request)
    _preflight_semantic_prompts(semantic_requests)
    plans: list[_SemanticReviewerInvocationPlan] = []
    for reviewer, semantic_request in zip(
        request.reviewers,
        semantic_requests,
        strict=True,
    ):
        binding = _model_call_binding(
            request,
            reviewer=reviewer,
            semantic_request=semantic_request,
        )
        durable_call = _bind_model_call_authority(
            model_call_authority_factory,
            binding=binding,
            rederive=lambda semantic_request=semantic_request, binding=binding: (
                _rederive_terminal_candidate_state_guard(
                    request,
                    semantic_request=semantic_request,
                    binding=binding,
                )
            ),
        )
        plans.append(
            _SemanticReviewerInvocationPlan(
                reviewer=reviewer,
                semantic_request=semantic_request,
                durable_call=durable_call,
                invocation_turn_id=request.frontier.turn_id,
            )
        )
    outcomes = _invoke_semantic_reviewers(
        tuple(plans),
        provider=provider,
        emit=emit,
        deadline=deadline,
    )
    _emit_reviewer_events_in_ordinal_order(outcomes, emit=emit)
    results: list[TaskGraphSemanticVerificationResult] = []
    for outcome in outcomes:
        if outcome.error is not None:
            raise outcome.error
        if outcome.result is None:  # pragma: no cover - 私有 DTO 不变性
            raise RuntimeError("semantic reviewer returned no result or error")
        results.append(outcome.result)
    ordered = validate_task_graph_semantic_verification_quorum(
        requests=semantic_requests,
        results=tuple(results),
    )
    return AuxiliaryTerminalCandidateSemanticReview(
        route=derive_task_graph_semantic_terminal_route(ordered),
        requests=semantic_requests,
        results=ordered,
    )


def _preflight_semantic_prompts(
    semantic_requests: tuple[TaskGraphSemanticVerificationRequest, ...],
) -> None:
    """在打开任何 Provider 调用前拒绝确定性提示错误。"""

    for semantic_request in semantic_requests:
        serialize_task_graph_semantic_verification_prompt(semantic_request)


def _invoke_semantic_reviewers(
    plans: tuple[_SemanticReviewerInvocationPlan, ...],
    *,
    provider: TaskGraphSemanticVerificationStructuredProvider,
    emit: Callable[[TurnEvent], object],
    deadline: TurnDeadline | None,
) -> tuple[_SemanticReviewerInvocationOutcome, ...]:
    """仅并发运行独立的复审模型调用。

    每个工人拥有自己的后启动事件缓冲区，每个持久化 Store 变更都会通过权威状态打开自己的连接。协调器会实时发布每个复审者的第一个生命周期事件，并联结所有工人，然后以相同的冻结顺序返回结果。调用者随后发布剩余事件并按顺序提交语义结果行。
    """

    if not plans:
        return ()
    if tuple(plan.reviewer.reviewer_ordinal for plan in plans) != tuple(
        sorted(plan.reviewer.reviewer_ordinal for plan in plans)
    ):
        raise ValueError("semantic invocation plans must be ordinal ordered")
    if len(plans) == 1:
        return (
            _invoke_one_semantic_reviewer(
                plans[0],
                provider=provider,
                first_event_queue=None,
                direct_first_event_emit=emit,
                coordinator_abort=None,
                deadline=deadline,
            ),
        )
    first_event_queues = {
        plan.reviewer.reviewer_ordinal: Queue() for plan in plans
    }
    coordinator_abort = _SemanticReviewerCoordinatorAbort()
    with ThreadPoolExecutor(
        max_workers=len(plans),
        thread_name_prefix="personagraph-semantic-reviewer",
    ) as executor:
        futures = []
        try:
            for plan in plans:
                first_event_queue = first_event_queues[
                    plan.reviewer.reviewer_ordinal
                ]
                inherited_context = copy_context()
                future = executor.submit(
                    inherited_context.run,
                    _invoke_one_semantic_reviewer,
                    plan,
                    provider=provider,
                    first_event_queue=first_event_queue,
                    direct_first_event_emit=None,
                    coordinator_abort=coordinator_abort,
                    deadline=deadline,
                )
                futures.append(future)
                future.add_done_callback(
                    lambda _done, queue=first_event_queue: queue.put(
                        _SEMANTIC_REVIEWER_FUTURE_DONE
                    )
                )
        except BaseException as exc:
            # 所有提交成功前，不会释放首次事件门。即便如此，
            # 提交也可能在抛出异常前先把工作入队，因而失败时
            # 没有返回该工作线程的 Future。共享中止信号会在 Provider I/O 前
            # 通知已知及稍后出队的工作线程；随后执行器上下文
            # 会联结所有工作线程。
            coordinator_abort.fail(exc)
            for future in futures:
                future.cancel()
            for future in futures:
                try:
                    future.result()
                except BaseException:
                    # 原始提交失败才是协调器失败；
                    # 此处只排空工作线程的失败，以确保完成清理。
                    pass
            raise
        # 协调线程在等待 Provider 完成前，会为每位审阅者
        # 发布恰好一个事件（或消费一个“无事件完成”信号）。
        # 释放审阅者 1 后，它的 Provider 请求可以运行，同时
        # 审阅者 2 的 STARTED 事件会被发布，因此排序不会牺牲
        # 任何有效的 Provider 并发。
        for plan, future in zip(plans, futures, strict=True):
            handoff_or_done = first_event_queues[
                plan.reviewer.reviewer_ordinal
            ].get()
            if handoff_or_done is _SEMANTIC_REVIEWER_FUTURE_DONE:
                # 抛出取消异常或原始工作线程异常。正常返回的
                # 结果不能跳过强制移交。
                future.result()
                raise RuntimeError(
                    "semantic reviewer completed without a first-event handoff"
                )
            handoff = handoff_or_done
            try:
                if handoff.event is not None:
                    emit(handoff.event)
            except BaseException as exc:
                handoff.error = exc
            finally:
                # 事件发射器失败会成为该审阅者的普通结果，
                # 但绝不能让下一个序号永久阻塞在此门后。
                handoff.released.set()
        # Future 按计划顺序而非完成顺序获取。每个工作线程都会捕获
        # BaseException，因此上下文管理器总会在控制权返回 Host 前
        # 联结每个已分发的 Provider 调用。
        return tuple(future.result() for future in futures)


def _invoke_one_semantic_reviewer(
    plan: _SemanticReviewerInvocationPlan,
    *,
    provider: TaskGraphSemanticVerificationStructuredProvider,
    first_event_queue: Queue[
        _SemanticReviewerFirstEventHandoff | _SemanticReviewerFutureDone
    ]
    | None,
    direct_first_event_emit: Callable[[TurnEvent], object] | None,
    coordinator_abort: _SemanticReviewerCoordinatorAbort | None,
    deadline: TurnDeadline | None,
) -> _SemanticReviewerInvocationOutcome:
    event_capture = _SemanticReviewerEventCapture(
        first_event_queue=first_event_queue,
        direct_first_event_emit=direct_first_event_emit,
        coordinator_abort=coordinator_abort,
    )
    try:
        requested = request_task_graph_semantic_verification(
            plan.semantic_request,
            invocation_turn_id=plan.invocation_turn_id,
            verification_result_id=plan.reviewer.verification_result_id,
            provider=provider,
            emit=event_capture,
            deadline=deadline,
            durable_call=plan.durable_call,
        )
        result = validate_task_graph_semantic_verification_result(
            request=plan.semantic_request,
            result=requested.value,
        )
        error = None
    except BaseException as exc:  # 参见调用者的联结保证
        result = None
        error = exc
    finally:
        try:
            event_capture.complete_without_first_event()
        except BaseException as exc:
            result = None
            error = exc
    return _SemanticReviewerInvocationOutcome(
        reviewer_ordinal=plan.reviewer.reviewer_ordinal,
        result=result,
        events=tuple(event_capture.remaining_events),
        error=error,
    )


def _emit_reviewer_events_in_ordinal_order(
    outcomes: tuple[_SemanticReviewerInvocationOutcome, ...],
    *,
    emit: Callable[[TurnEvent], object],
) -> None:
    for outcome in outcomes:
        for event in outcome.events:
            emit(event)


def run_auxiliary_semantic_verification(
    request: AuxiliarySemanticVerificationRequest,
    *,
    provider: TaskGraphSemanticVerificationStructuredProvider,
    model_call_authority_factory: AuxiliarySemanticModelCallAuthorityFactory,
    emit: Callable[[TurnEvent], object],
    deadline: TurnDeadline | None = None,
) -> AuxiliarySemanticVerificationResult:
    """驱动或重放终端提案的确切语义审核共识集。"""

    if not isinstance(request, AuxiliarySemanticVerificationRequest):
        raise TypeError(
            "request must be AuxiliarySemanticVerificationRequest"
        )
    if not callable(model_call_authority_factory):
        raise TypeError("model_call_authority_factory must be callable")
    try:
        request = AuxiliarySemanticVerificationRequest.model_validate(
            request.model_dump(mode="json")
        )
    except (TypeError, ValueError) as exc:
        return _stop(request, "semantic_controller_request_invalid", cause=exc)

    try:
        semantic_requests = _prepare_semantic_requests(request)
    except (
        _FrontierAuthorityRejected,
        auxiliary_graph_store.AuxiliaryGraphPersistenceError,
    ) as exc:
        return _stop(
            request,
            "semantic_frontier_authority_rejected",
            cause=exc,
        )
    except (
        semantic_store.AuxiliarySemanticVerificationPersistenceError,
        _StoredSemanticAuthorityRejected,
    ) as exc:
        return _stop(
            request,
            "semantic_stored_authority_rejected",
            cause=exc,
        )
    except (TypeError, ValueError, ValidationError) as exc:
        return _stop(request, "semantic_prompt_authority_rejected", cause=exc)

    try:
        existing_settlement = semantic_store.get_auxiliary_semantic_quorum_settlement(
            session_id=request.frontier.session_id,
            task_id=request.frontier.task_id,
            auxiliary_graph_id=request.frontier.auxiliary_graph_id,
            goal_id=request.frontier.goal_id,
            auxiliary_graph_revision=request.frontier.auxiliary_graph_revision,
            frozen_prompt_payload_sha256=request.prompt_payload.payload_sha256,
        )
        if existing_settlement is not None:
            _require_exact_settlement(
                request,
                semantic_requests=semantic_requests,
                settlement=existing_settlement,
            )
            return _settled(
                request,
                settlement=existing_settlement,
                replayed=True,
            )

        _freeze_catalog(request)
    except (
        semantic_store.AuxiliarySemanticVerificationPersistenceError,
        _StoredSemanticAuthorityRejected,
    ) as exc:
        return _stop(
            request,
            "semantic_stored_authority_rejected",
            cause=exc,
        )

    try:
        _preflight_semantic_prompts(semantic_requests)
    except (
        TaskGraphSemanticVerificationInputTooLarge,
        TaskGraphSemanticVerificationInputUnsupported,
        TypeError,
        ValueError,
        ValidationError,
    ) as exc:
        return _stop(
            request,
            "semantic_reviewer_contract_rejected",
            cause=exc,
        )

    # 在任何 Provider I/O 之前持久化每个不可变的评审请求。这
    # 即使独立模型不同，仍保持固定的 Host 序列。
    # 调用重叠在墙钟时间上。
    stored_results: dict[int, TaskGraphSemanticVerificationResult] = {}
    plans: list[_SemanticReviewerInvocationPlan] = []
    for reviewer, semantic_request in zip(
        request.reviewers,
        semantic_requests,
        strict=True,
    ):
        try:
            existing_request = (
                semantic_store.get_auxiliary_semantic_verification_request(
                    session_id=request.frontier.session_id,
                    verification_request_id=(
                        semantic_request.verification_request_id
                    ),
                )
            )
            request_origin_turn_id = (
                request.frontier.turn_id
                if existing_request is None
                else existing_request.created_turn_id
            )
            committed_request = semantic_store.commit_auxiliary_semantic_verification_request(
                command=semantic_store.CommitAuxiliarySemanticVerificationRequestCommand(
                    session_id=request.frontier.session_id,
                    created_turn_id=request_origin_turn_id,
                    expected_control_state_version=(
                        request.frontier.control_state_version
                    ),
                    expected_goal_state_version=request.frontier.goal_state_version,
                    expected_revision_state_version=(
                        request.frontier.revision_state_version
                    ),
                    expected_budget_state_version=(
                        request.frontier.budget_state_version
                    ),
                    request=semantic_request,
                )
            )
            if committed_request.record.request != semantic_request:
                raise _StoredSemanticAuthorityRejected(
                    "semantic request Store projection crossed reviewer authority"
                )
            stored_result = semantic_store.get_auxiliary_semantic_verification_result(
                session_id=request.frontier.session_id,
                verification_result_id=reviewer.verification_result_id,
            )
            if stored_result is not None:
                stored_results[reviewer.reviewer_ordinal] = (
                    validate_task_graph_semantic_verification_result(
                        request=semantic_request,
                        result=stored_result.result,
                    )
                )
                continue
            binding = _model_call_binding(
                request,
                reviewer=reviewer,
                semantic_request=semantic_request,
            )
            durable_call = _bind_model_call_authority(
                model_call_authority_factory,
                binding=binding,
                rederive=lambda semantic_request=semantic_request, binding=binding: (
                    _rederive_state_guard(
                        request,
                        semantic_request=semantic_request,
                        binding=binding,
                    )
                ),
            )
            plans.append(
                _SemanticReviewerInvocationPlan(
                    reviewer=reviewer,
                    semantic_request=semantic_request,
                    durable_call=durable_call,
                    invocation_turn_id=request.frontier.turn_id,
                )
            )
        except BaseException as exc:
            return _reviewer_error_result(
                request,
                reviewer_ordinal=reviewer.reviewer_ordinal,
                completed=_stored_result_prefix(
                    request.reviewers,
                    stored_results=stored_results,
                ),
                error=exc,
            )

    outcomes = _invoke_semantic_reviewers(
        tuple(plans),
        provider=provider,
        emit=emit,
        deadline=deadline,
    )
    # 首先，生命周期事件已经通过序号门发布。
    # 在两个调用都加入后，仅通过序号发布剩余的事件，以便
    # Provider 延迟的竞争无法改变持久化事件序列。
    _emit_reviewer_events_in_ordinal_order(outcomes, emit=emit)
    outcomes_by_ordinal = {
        outcome.reviewer_ordinal: outcome for outcome in outcomes
    }

    results: list[TaskGraphSemanticVerificationResult] = []
    completed: list[int] = []
    for reviewer, _semantic_request in zip(
        request.reviewers,
        semantic_requests,
        strict=True,
    ):
        result = stored_results.get(reviewer.reviewer_ordinal)
        if result is None:
            outcome = outcomes_by_ordinal[reviewer.reviewer_ordinal]
            if outcome.error is not None:
                return _reviewer_error_result(
                    request,
                    reviewer_ordinal=reviewer.reviewer_ordinal,
                    completed=tuple(completed),
                    error=outcome.error,
                )
            if outcome.result is None:  # pragma: no cover - 私有 DTO 不变量
                raise RuntimeError("semantic reviewer returned no result or error")
            try:
                committed_result = (
                    semantic_store.commit_auxiliary_semantic_verification_result(
                        command=semantic_store.CommitAuxiliarySemanticVerificationResultCommand(
                            session_id=request.frontier.session_id,
                            created_turn_id=request.frontier.turn_id,
                            expected_control_state_version=(
                                request.frontier.control_state_version
                            ),
                            expected_goal_state_version=(
                                request.frontier.goal_state_version
                            ),
                            expected_revision_state_version=(
                                request.frontier.revision_state_version
                            ),
                            expected_budget_state_version=(
                                request.frontier.budget_state_version
                            ),
                            result=outcome.result,
                        )
                    )
                )
                result = committed_result.record.result
            except BaseException as exc:
                return _reviewer_error_result(
                    request,
                    reviewer_ordinal=reviewer.reviewer_ordinal,
                    completed=tuple(completed),
                    error=exc,
                )
        results.append(result)
        completed.append(reviewer.reviewer_ordinal)

    try:
        settlement = semantic_store.settle_auxiliary_semantic_verification_quorum(
            command=semantic_store.SettleAuxiliarySemanticVerificationQuorumCommand(
                settlement_id=request.settlement_id,
                session_id=request.frontier.session_id,
                created_turn_id=request.frontier.turn_id,
                expected_control_state_version=(
                    request.frontier.control_state_version
                ),
                expected_goal_state_version=request.frontier.goal_state_version,
                expected_revision_state_version=(
                    request.frontier.revision_state_version
                ),
                expected_budget_state_version=request.frontier.budget_state_version,
                requests=semantic_requests,
                results=tuple(results),
            )
        )
        return _settled(
            request,
            settlement=settlement.settlement,
            replayed=settlement.status == "replayed",
        )
    except semantic_store.AuxiliarySemanticVerificationPersistenceError as exc:
        return _stop(
            request,
            "semantic_stored_authority_rejected",
            completed=tuple(completed),
            cause=exc,
        )


def _stored_result_prefix(
    reviewers: tuple[AuxiliarySemanticReviewerIdPlan, ...],
    *,
    stored_results: dict[int, TaskGraphSemanticVerificationResult],
) -> tuple[int, ...]:
    prefix: list[int] = []
    for reviewer in reviewers:
        if reviewer.reviewer_ordinal not in stored_results:
            break
        prefix.append(reviewer.reviewer_ordinal)
    return tuple(prefix)


def _reviewer_error_result(
    request: AuxiliarySemanticVerificationRequest,
    *,
    reviewer_ordinal: int,
    completed: tuple[int, ...],
    error: BaseException,
) -> AuxiliarySemanticVerificationResult:
    """精确地投影一个序数失败，就像之前的串行循环所做的那样。"""

    if isinstance(error, RuntimeModelCallWaitingExternal):
        return _stop(
            request,
            "semantic_reviewer_model_call_waiting_external",
            status=AuxiliarySemanticVerificationStatus.WAITING_EXTERNAL,
            completed=completed,
            pending_reviewer_ordinal=reviewer_ordinal,
            cause=error,
        )
    if isinstance(error, TurnDeadlineExceeded):
        return _stop(
            request,
            "semantic_reviewer_turn_deadline_exceeded",
            status=AuxiliarySemanticVerificationStatus.TURN_LIMIT_REACHED,
            completed=completed,
            cause=error,
        )
    if isinstance(error, DurableModelCallStateGuardRejected):
        return _stop(
            request,
            "semantic_reviewer_model_state_guard_changed",
            completed=completed,
            cause=error,
        )
    if isinstance(error, (ModelGatewayError, DurableModelCallTerminalState)):
        return _stop(
            request,
            "semantic_reviewer_model_call_interrupted",
            status=AuxiliarySemanticVerificationStatus.MODEL_INTERRUPTED,
            completed=completed,
            cause=error,
        )
    if isinstance(
        error,
        (
            semantic_store.AuxiliarySemanticVerificationPersistenceError,
            session_store.RuntimeModelCallPersistenceError,
            _StoredSemanticAuthorityRejected,
        ),
    ):
        return _stop(
            request,
            "semantic_stored_authority_rejected",
            completed=completed,
            cause=error,
        )
    if isinstance(
        error,
        (
            TaskGraphSemanticVerificationInputTooLarge,
            TaskGraphSemanticVerificationInputUnsupported,
            TypeError,
            ValueError,
            ValidationError,
        ),
    ):
        return _stop(
            request,
            "semantic_reviewer_contract_rejected",
            completed=completed,
            cause=error,
        )
    raise error


def _prepare_semantic_requests(
    request: AuxiliarySemanticVerificationRequest,
) -> tuple[TaskGraphSemanticVerificationRequest, ...]:
    frontier = request.frontier
    current = auxiliary_graph_store.project_auxiliary_graph_execution_frontier(
        session_id=frontier.session_id,
        turn_id=frontier.turn_id,
        insession_task_id=frontier.task_id,
    )
    if current != frontier:
        raise _FrontierAuthorityRejected(
            "semantic controller frontier is no longer Store-current"
        )
    terminal_refs = tuple(
        item
        for item in frontier.completed_node_refs
        if item.node_id == frontier.terminal_node_id
    )
    completed_terminal = (
        frontier.goal_status is AuxiliaryPlanningGoalStatus.ACTIVE
        and frontier.revision_status == "active"
        and not frontier.ready_fresh
        and not frontier.recoverable
        and not frontier.recoverable_primitive
        and not frontier.blocking_node_refs
        and len(terminal_refs) == 1
    )
    candidate_terminal = _terminal_candidate_frontier_matches(request)
    if not completed_terminal and not candidate_terminal:
        raise _FrontierAuthorityRejected(
            "semantic controller requires a completed or exact candidate terminal frontier"
        )
    if request.terminal_candidate_binding is None and not completed_terminal:
        raise _FrontierAuthorityRejected(
            "ordinary semantic review requires one completed terminal frontier"
        )
    payload = request.prompt_payload
    if (
        payload.goal.goal_id != frontier.goal_id
        or payload.authority.authority_snapshot_id
        != frontier.authority_snapshot_id
        or payload.authority.authority_snapshot_sha256
        != frontier.authority_snapshot_sha256
        or payload.budget.goal_id != frontier.goal_id
        or payload.budget.state_version != frontier.budget_state_version
        or payload.budget.snapshot_sha256 != frontier.budget_snapshot_sha256
        or payload.capabilities != request.capability_catalog
    ):
        raise _StoredSemanticAuthorityRejected(
            "semantic prompt differs from frontier/catalog authority"
        )
    goal = planning_store.get_current_auxiliary_planning_goal(
        session_id=frontier.session_id,
        insession_task_id=frontier.task_id,
    )
    if (
        goal is None
        or goal.session_id != frontier.session_id
        or goal.task_id != frontier.task_id
        or goal.auxiliary_graph_id != frontier.auxiliary_graph_id
        or goal.goal_id != frontier.goal_id
        or goal.state_version != frontier.goal_state_version
        or goal.status is not AuxiliaryPlanningGoalStatus.ACTIVE
    ):
        raise _FrontierAuthorityRejected(
            "semantic controller goal is no longer Store-current"
        )
    policy = semantic_store.derive_auxiliary_semantic_review_policy(
        prompt_payload=payload
    )
    required_count = required_task_graph_semantic_reviewer_count(
        prompt_payload=payload,
        review_policy=policy,
    )
    if len(request.reviewers) != required_count:
        raise _StoredSemanticAuthorityRejected(
            "semantic reviewer plan differs from Host-derived policy"
        )
    return tuple(
        TaskGraphSemanticVerificationRequest.create(
            verification_request_id=reviewer.verification_request_id,
            logical_call_id=reviewer.logical_call_id,
            verification_profile_id=reviewer.verification_profile_id,
            reviewer_ordinal=reviewer.reviewer_ordinal,
            required_reviewer_count=required_count,
            goal=goal,
            auxiliary_graph_revision=frontier.auxiliary_graph_revision,
            auxiliary_graph_structure_sha256=frontier.structure_sha256,
            prompt_payload=payload,
            review_policy=policy,
            terminal_candidate_binding=request.terminal_candidate_binding,
        )
        for reviewer in request.reviewers
    )


def _terminal_candidate_frontier_matches(
    request: AuxiliarySemanticVerificationRequest,
) -> bool:
    frontier = request.frontier
    binding = request.terminal_candidate_binding
    if binding is None or len(frontier.recoverable) != 1:
        return False
    candidate = frontier.recoverable[0]
    return (
        frontier.goal_status is AuxiliaryPlanningGoalStatus.ACTIVE
        and frontier.revision_status == "active"
        and not frontier.ready_fresh
        and not frontier.recoverable_primitive
        and not frontier.blocking_node_refs
        and all(
            item.node_id != frontier.terminal_node_id
            for item in frontier.completed_node_refs
        )
        and candidate.subject.task_id == frontier.task_id
        and candidate.subject.auxiliary_graph_id == frontier.auxiliary_graph_id
        and candidate.subject.auxiliary_graph_revision
        == frontier.auxiliary_graph_revision
        and candidate.subject.node_id == frontier.terminal_node_id
        and candidate.executor_kind is AuxiliaryNodeExecutorKind.TERMINAL_PLANNER
        and candidate.work_run_id == binding.work_run_id
        and candidate.work_run_status is WorkRunStatus.ACTIVE
        and candidate.work_run_reason == "verification_pending"
        and candidate.current_attempt_id is None
        and candidate.current_verification_request_id
        == binding.node_verification_request_id
    )


def _freeze_catalog(request: AuxiliarySemanticVerificationRequest) -> None:
    frontier = request.frontier
    stored = semantic_store.get_auxiliary_semantic_capability_catalog(
        session_id=frontier.session_id,
        capability_catalog_snapshot_id=(
            request.capability_catalog.capability_catalog_snapshot_id
        ),
        projection_sha256=request.capability_catalog.projection_sha256,
    )
    created_turn_id = (
        frontier.turn_id if stored is None else stored.created_turn_id
    )
    semantic_store.freeze_auxiliary_semantic_capability_catalog(
        command=semantic_store.FreezeAuxiliarySemanticCapabilityCatalogCommand(
            session_id=frontier.session_id,
            created_turn_id=created_turn_id,
            task_id=frontier.task_id,
            auxiliary_graph_id=frontier.auxiliary_graph_id,
            goal_id=frontier.goal_id,
            auxiliary_graph_revision=frontier.auxiliary_graph_revision,
            expected_control_state_version=frontier.control_state_version,
            expected_goal_state_version=frontier.goal_state_version,
            expected_revision_state_version=frontier.revision_state_version,
            expected_budget_state_version=frontier.budget_state_version,
            expected_authority_snapshot_id=frontier.authority_snapshot_id,
            expected_authority_snapshot_sha256=(
                frontier.authority_snapshot_sha256
            ),
            expected_structure_sha256=frontier.structure_sha256,
            catalog=request.capability_catalog,
        )
    )


def _model_call_binding(
    request: AuxiliarySemanticVerificationRequest,
    *,
    reviewer: AuxiliarySemanticReviewerIdPlan,
    semantic_request: TaskGraphSemanticVerificationRequest,
) -> AuxiliarySemanticReviewerModelCallBinding:
    request_json = _canonical_json(
        {
            "schema_version": _REQUEST_CONTRACT,
            "verification_result_id": reviewer.verification_result_id,
            "verification_request": semantic_request.model_dump(mode="json"),
        }
    )
    return AuxiliarySemanticReviewerModelCallBinding(
        logical_call_id=reviewer.logical_call_id,
        verification_request_id=reviewer.verification_request_id,
        verification_result_id=reviewer.verification_result_id,
        reviewer_ordinal=reviewer.reviewer_ordinal,
        required_reviewer_count=len(request.reviewers),
        session_id=request.frontier.session_id,
        task_id=request.frontier.task_id,
        auxiliary_graph_id=request.frontier.auxiliary_graph_id,
        goal_id=request.frontier.goal_id,
        auxiliary_graph_revision=request.frontier.auxiliary_graph_revision,
        invocation_turn_id=request.frontier.turn_id,
        request_json=request_json,
        request_sha256=_sha256_text(request_json),
        state_guard_sha256=_semantic_state_guard(
            request.frontier,
            semantic_request=semantic_request,
            verification_result_id=reviewer.verification_result_id,
            terminal_candidate_binding=request.terminal_candidate_binding,
        ),
    )


def _bind_model_call_authority(
    factory: AuxiliarySemanticModelCallAuthorityFactory,
    *,
    binding: AuxiliarySemanticReviewerModelCallBinding,
    rederive: Callable[[], str],
) -> DurableLogicalModelCallAuthority:
    authority = factory(
        binding,
        rederive_state_guard_sha256=rederive,
    )
    if not isinstance(authority, DurableLogicalModelCallAuthority):
        raise TypeError("semantic model authority factory returned an invalid port")
    if authority.semantic_call_id != binding.logical_call_id:
        raise _StoredSemanticAuthorityRejected(
            "semantic model authority crossed logical-call identity"
        )
    if isinstance(authority, RuntimeLogicalModelCallAuthority):
        logical = authority.logical_request
        stable_mismatch = (
            logical.session_id != binding.session_id
            or logical.task_id != binding.task_id
            or logical.auxiliary_graph_id != binding.auxiliary_graph_id
            or logical.goal_id != binding.goal_id
            or logical.execution_subject_id is not None
            or logical.call_kind != "task_graph_semantic_verification"
            or logical.purpose != binding.purpose
            or logical.request_contract != binding.request_contract
            or logical.request_json != binding.request_json
            or logical.request_sha256 != binding.request_sha256
            or logical.typed_result_contract != binding.typed_result_contract
            or logical.max_physical_attempts != binding.max_physical_attempts
        )
        exact_logical_binding = (
            logical.invocation_turn_id == binding.invocation_turn_id
            and logical.state_guard_sha256 == binding.state_guard_sha256
        )
        admitted_dispatch_continuation = (
            authority.dispatch_binding_sha256
            == _sha256_value(binding.model_dump(mode="json"))
            and authority.dispatch_state_guard_sha256
            == binding.state_guard_sha256
        )
        if stable_mismatch or not (
            exact_logical_binding or admitted_dispatch_continuation
        ):
            raise _StoredSemanticAuthorityRejected(
                "Runtime model authority differs from semantic reviewer binding"
            )
    return authority


def _rederive_state_guard(
    request: AuxiliarySemanticVerificationRequest,
    *,
    semantic_request: TaskGraphSemanticVerificationRequest,
    binding: AuxiliarySemanticReviewerModelCallBinding,
) -> str:
    try:
        current = auxiliary_graph_store.project_auxiliary_graph_execution_frontier(
            session_id=request.frontier.session_id,
            turn_id=request.frontier.turn_id,
            insession_task_id=request.frontier.task_id,
        )
        stored = semantic_store.get_auxiliary_semantic_verification_request(
            session_id=request.frontier.session_id,
            verification_request_id=semantic_request.verification_request_id,
        )
        if current != request.frontier or stored is None or stored.request != semantic_request:
            return "0" * 64
        return _semantic_state_guard(
            current,
            semantic_request=semantic_request,
            verification_result_id=binding.verification_result_id,
            terminal_candidate_binding=request.terminal_candidate_binding,
        )
    except Exception:
        return "0" * 64


def _rederive_terminal_candidate_state_guard(
    request: AuxiliarySemanticVerificationRequest,
    *,
    semantic_request: TaskGraphSemanticVerificationRequest,
    binding: AuxiliarySemanticReviewerModelCallBinding,
) -> str:
    candidate = request.terminal_candidate_binding
    if candidate is None:
        return "0" * 64
    try:
        current = auxiliary_graph_store.project_auxiliary_graph_execution_frontier(
            session_id=request.frontier.session_id,
            turn_id=request.frontier.turn_id,
            insession_task_id=request.frontier.task_id,
        )
        prepared = verification_store.get_prepared_auxiliary_node_verification(
            session_id=request.frontier.session_id,
            invocation_turn_id=request.frontier.turn_id,
            verification_request_id=candidate.node_verification_request_id,
        )
        output_snapshot_sha256 = _sha256_text(
            _canonical_json(prepared.locked_output_window.model_dump(mode="json"))
        )
        if (
            current != request.frontier
            or not _terminal_candidate_frontier_matches(request)
            or prepared.work_run.work_run_id != candidate.work_run_id
            or prepared.submitted_attempt.attempt_id
            != candidate.submitted_attempt_id
            or prepared.record.request.verification_request_id
            != candidate.node_verification_request_id
            or prepared.locked_output_window.output_revision
            != candidate.output_revision
            or output_snapshot_sha256 != candidate.output_snapshot_sha256
            or prepared.work_run.subject.node_id
            != request.frontier.terminal_node_id
        ):
            return "0" * 64
        return _semantic_state_guard(
            current,
            semantic_request=semantic_request,
            verification_result_id=binding.verification_result_id,
            terminal_candidate_binding=candidate,
        )
    except Exception:
        return "0" * 64


def _semantic_state_guard(
    frontier: AuxiliaryGraphExecutionFrontier,
    *,
    semantic_request: TaskGraphSemanticVerificationRequest,
    verification_result_id: str,
    terminal_candidate_binding: AuxiliaryTerminalCandidateBinding | None = None,
) -> str:
    if terminal_candidate_binding is not None:
        return _sha256_value(
            {
                "schema_version": "auxiliary-v2-terminal-candidate-semantic-model-state-guard-v1",
                "session_id": frontier.session_id,
                "turn_id": frontier.turn_id,
                "task_id": frontier.task_id,
                "auxiliary_graph_id": frontier.auxiliary_graph_id,
                "goal_id": frontier.goal_id,
                "auxiliary_graph_revision": frontier.auxiliary_graph_revision,
                "structure_sha256": frontier.structure_sha256,
                "candidate_binding": terminal_candidate_binding.model_dump(
                    mode="json"
                ),
                "verification_request_binding_sha256": (
                    semantic_request.binding_sha256
                ),
                "verification_result_id": verification_result_id,
            }
        )
    return _sha256_value(
        {
            "schema_version": "auxiliary-v2-semantic-model-state-guard-v1",
            "frontier": frontier.model_dump(mode="json"),
            "verification_request_binding_sha256": (
                semantic_request.binding_sha256
            ),
            "verification_result_id": verification_result_id,
        }
    )


def _require_exact_settlement(
    request: AuxiliarySemanticVerificationRequest,
    *,
    semantic_requests: tuple[TaskGraphSemanticVerificationRequest, ...],
    settlement: StoredAuxiliarySemanticQuorumSettlement,
) -> None:
    if (
        settlement.settlement_id != request.settlement_id
        or settlement.requests != semantic_requests
        or tuple(item.verification_result_id for item in settlement.results)
        != tuple(item.verification_result_id for item in request.reviewers)
    ):
        raise _StoredSemanticAuthorityRejected(
            "stored semantic settlement crossed stable controller identities"
        )


def _settled(
    request: AuxiliarySemanticVerificationRequest,
    *,
    settlement: StoredAuxiliarySemanticQuorumSettlement,
    replayed: bool,
) -> AuxiliarySemanticVerificationResult:
    return AuxiliarySemanticVerificationResult(
        status=AuxiliarySemanticVerificationStatus.SETTLED,
        reason_code="semantic_quorum_settled",
        session_id=request.frontier.session_id,
        task_id=request.frontier.task_id,
        auxiliary_graph_id=request.frontier.auxiliary_graph_id,
        goal_id=request.frontier.goal_id,
        auxiliary_graph_revision=request.frontier.auxiliary_graph_revision,
        required_reviewer_count=len(request.reviewers),
        completed_reviewer_ordinals=tuple(
            item.reviewer_ordinal for item in request.reviewers
        ),
        settlement=settlement,
        replayed=replayed,
    )


def _stop(
    request: AuxiliarySemanticVerificationRequest,
    reason_code: str,
    *,
    status: AuxiliarySemanticVerificationStatus = (
        AuxiliarySemanticVerificationStatus.FAILED_CLOSED
    ),
    completed: tuple[int, ...] = (),
    pending_reviewer_ordinal: int | None = None,
    cause: BaseException | None = None,
) -> AuxiliarySemanticVerificationResult:
    del cause
    return AuxiliarySemanticVerificationResult(
        status=status,
        reason_code=reason_code,
        session_id=request.frontier.session_id,
        task_id=request.frontier.task_id,
        auxiliary_graph_id=request.frontier.auxiliary_graph_id,
        goal_id=request.frontier.goal_id,
        auxiliary_graph_revision=request.frontier.auxiliary_graph_revision,
        required_reviewer_count=len(request.reviewers),
        completed_reviewer_ordinals=completed,
        pending_reviewer_ordinal=pending_reviewer_ordinal,
    )


__all__ = [
    "AuxiliarySemanticModelCallAuthorityFactory",
    "AuxiliarySemanticReviewerIdPlan",
    "AuxiliarySemanticReviewerModelCallBinding",
    "AuxiliarySemanticVerificationRequest",
    "AuxiliarySemanticVerificationResult",
    "AuxiliarySemanticVerificationStatus",
    "AuxiliaryTerminalCandidateBinding",
    "AuxiliaryTerminalCandidateSemanticReview",
    "review_auxiliary_terminal_candidate_semantics",
    "run_auxiliary_semantic_verification",
]
