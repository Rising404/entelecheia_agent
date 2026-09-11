"""当前多节点 TaskGraph 的内部确定性驱动器。

驱动器不持有调度器表，也不让模型选择节点。它会在每次持久 WorkRun 停止后重新投影精确的
Task 局部前沿，维护 Session 的标量执行游标，并且只将图序号用作已就绪节点之间的稳定决胜项。
等待分支会在 Store 证明安全的检查点分离，以便独立的同级分支运行；问题和 WorkRun 仍保持
持久且可发现。

只有规范根 Delivery 能作为最终 Task 结果离开此层。中间 NodeDelivery 仍是其父节点的依赖权威。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

from personagraph.session import store as session_store
from personagraph.session.l2_store import continuation as continuation_store
from personagraph.session.l2_store import task_graph as task_graph_store
from personagraph.session.l2_store import verification as verification_store
from personagraph.session.l2_store import work_run as work_run_store
from personagraph.tools.catalog import CatalogSnapshot
from personagraph.l2.work_run import (
    AttemptStatus,
    PendingUserQuestion,
    TaskNodeSubject,
    TaskNodeVerificationRequestStatus,
    WorkRunStatus,
)
from personagraph.l2.task_execution.attempts.decision import AttemptDecisionStructuredProvider
from personagraph.runtime.turn_deadline import TurnDeadline
from personagraph.l2.task_execution.tool_bridge.contracts import AttemptToolBridge
from personagraph.l2.task_execution.verification.decision import NodeVerificationStructuredProvider
from personagraph.l2.task_execution.delivery.candidate_gate import (
    NodeDownstreamVerificationGateFactory,
)
from personagraph.l2.task_execution.task_graph.contracts import (
    TaskGraphWorkRunRequest,
    TaskGraphWorkRunResult,
    TaskGraphWorkRunStatus,
)
from personagraph.l2.task_execution.task_graph.id_plan import (
    derive_task_graph_safe_lane_detach_apply_id as _stable_detach_apply_id,
    derive_task_graph_work_run_stable_ids as _fresh_id_plan,
    recover_task_graph_work_run_stable_ids as _id_plan_from_work_run_id,
)
from personagraph.l2.task_execution.task_graph.profile import (
    TaskGraphWorkRunProfile,
)
from personagraph.l2.task_execution.task_node.model_authority_contracts import (
    TaskNodeModelCallAuthorityFactory,
)
from personagraph.l2.task_execution.task_node.tool_runtime_contracts import (
    TaskNodeToolRuntimeBinding,
    TaskNodeToolRuntimeFactory,
    TaskNodeToolRuntimePlan,
    TaskNodeToolRuntime,
)
from personagraph.l2.task_execution.task_node.tool_runtime_policy import (
    ordinary_task_node_model_authority_factory as _ordinary_task_node_model_authority_factory,
    validate_task_node_tool_runtime as _validate_node_tool_runtime,
)
from personagraph.runtime.turn_events import TurnEvent
from personagraph.l2.task_execution.tool_bridge.execution_findings_catalog import (
    augment_execution_findings_tool_runtime,
)
from personagraph.l2.task_execution.work_run.model_providers import (
    build_attempt_structured_provider,
    build_verification_structured_provider,
)
from personagraph.l2.task_execution.work_run.turn_request_contracts import (
    WorkRunTurnApplicationRequest,
    WorkRunTurnResumeRequest,
    WorkRunTurnUnpreparedVerificationRecoveryRequest,
    WorkRunTurnVerificationRecoveryRequest,
    WorkRunTurnWaitingUserContinuationRequest,
)
from personagraph.l2.task_execution.work_run.turn_controller import (
    continue_waiting_user_task_node_work_run,
    recover_unprepared_task_node_work_run_verification,
    recover_task_node_work_run_verification,
    resume_active_task_node_work_run,
    run_new_task_node_work_run,
)


def run_task_graph_work_runs(
    request: TaskGraphWorkRunRequest,
    *,
    monotonic_clock: Callable[[], float],
    catalog_snapshot: CatalogSnapshot | None = None,
    attempt_provider: AttemptDecisionStructuredProvider | None = None,
    verification_provider: NodeVerificationStructuredProvider | None = None,
    emit: Callable[[TurnEvent], object] | None = None,
    tool_bridge: AttemptToolBridge | None = None,
    node_tool_runtime_factory: TaskNodeToolRuntimeFactory | None = None,
    node_tool_runtime_plan: TaskNodeToolRuntimePlan | None = None,
    downstream_gate_factory: NodeDownstreamVerificationGateFactory | None = None,
    model_call_authority_factory: TaskNodeModelCallAuthorityFactory | None = None,
    deadline: TurnDeadline | None = None,
) -> TaskGraphWorkRunResult:
    """让一个 Task 沿已就绪节点推进，直至完成或安全停止。"""

    profile = request.profile
    snapshot = catalog_snapshot or CatalogSnapshot(revision=1, entries=())
    structured = profile.structured_model_profile()
    attempt = attempt_provider or build_attempt_structured_provider(structured)
    verification = verification_provider or build_verification_structured_provider(
        structured
    )
    event_sink = emit or _discard_event
    revision = request.expected_window_revision
    work_run_ids: list[str] = []
    pending_question_attempt_ids: list[str] = []
    attempted_subjects: set[TaskNodeSubject] = set()
    safe_stop_outcomes: list[str] = []
    result = request.initial_work_run_result
    dispatch_count = 1 if result is not None and result.work_run_id is not None else 0
    default_runtime = TaskNodeToolRuntime(
        catalog_snapshot=snapshot,
        tool_bridge=tool_bridge,
        paper_resources=request.paper_resources,
    )
    try:
        if (
            node_tool_runtime_plan is not None
            and node_tool_runtime_factory is not None
        ):
            raise ValueError("provide a node runtime factory or a frozen plan, not both")
        runtime_plan = node_tool_runtime_plan or preflight_task_node_tool_runtimes(
            request=request,
            default_runtime=default_runtime,
            factory=node_tool_runtime_factory,
        )
        _validate_node_tool_runtime_plan(runtime_plan, request=request)
    except Exception:
        return _result(
            request,
            status="failed_closed",
            revision=request.expected_window_revision,
            work_run_ids=work_run_ids,
            pending_question_attempt_ids=pending_question_attempt_ids,
            failure_code="task_node_tool_runtime_unavailable",
        )

    while True:
        if result is not None and result.outcome == "task_graph_revision_requested":
            revision = result.window_revision
            if result.work_run_id is not None:
                _append_unique(work_run_ids, result.work_run_id)
            return _result(
                request,
                status="revision_required",
                revision=revision,
                work_run_ids=work_run_ids,
                pending_question_attempt_ids=pending_question_attempt_ids,
                last_outcome=result.outcome,
            )
        try:
            active_execution_replan = (
                work_run_store.get_active_task_graph_execution_replan_request(
                    session_id=request.session_id,
                    task_id=request.task_id,
                )
            )
        except Exception:
            return _result(
                request,
                status="failed_closed",
                revision=revision,
                work_run_ids=work_run_ids,
                pending_question_attempt_ids=pending_question_attempt_ids,
                failure_code="execution_replan_authority_unreadable",
            )
        if active_execution_replan is not None:
            return _result(
                request,
                status="revision_required",
                revision=revision,
                work_run_ids=work_run_ids,
                pending_question_attempt_ids=pending_question_attempt_ids,
                last_outcome="task_graph_revision_requested",
            )
        if result is not None:
            revision = result.window_revision
            if result.work_run_id is not None:
                _append_unique(work_run_ids, result.work_run_id)
                stored = work_run_store.get_work_run(
                    session_id=request.session_id,
                    work_run_id=result.work_run_id,
                )
                if (
                    not isinstance(stored.work_run.subject, TaskNodeSubject)
                    or stored.work_run.subject.task_id != request.task_id
                ):
                    return _result(
                        request,
                        status="failed_closed",
                        revision=revision,
                        work_run_ids=work_run_ids,
                        pending_question_attempt_ids=pending_question_attempt_ids,
                        last_outcome=result.outcome,
                        failure_code="subject_authority_mismatch",
                    )
                attempted_subjects.add(stored.work_run.subject)
            if result.outcome == "delivery_ready":
                result = None
                continue
            if result.outcome in {
                "waiting_user",
                "waiting_external",
                "turn_limit_reached",
                "work_run_failed",
            }:
                assert result.work_run_id is not None
                if result.outcome == "waiting_user":
                    pending = tuple(
                        item
                        for item in continuation_store.list_pending_user_questions(
                            session_id=request.session_id
                        )
                        if item.work_run_id == result.work_run_id
                    )
                    if len(pending) != 1:
                        return _result(
                            request,
                            status="failed_closed",
                            revision=revision,
                            work_run_ids=work_run_ids,
                            pending_question_attempt_ids=pending_question_attempt_ids,
                            last_outcome=result.outcome,
                            failure_code="pending_question_authority_ambiguous",
                        )
                    _append_unique(
                        pending_question_attempt_ids,
                        pending[0].question_attempt_id,
                    )
                safe_stop_outcomes.append(result.outcome)
                try:
                    detached = work_run_store.detach_safe_work_run_lane(
                        session_id=request.session_id,
                        turn_id=request.turn_id,
                        work_run_id=result.work_run_id,
                        expected_window_revision=result.window_revision,
                        apply_id=_stable_detach_apply_id(
                            turn_id=request.turn_id,
                            work_run_id=result.work_run_id,
                        ),
                    )
                except Exception:
                    # 精确重放可区分响应丢失与分离未应用；第二次失败则具有歧义。
                    try:
                        detached = work_run_store.detach_safe_work_run_lane(
                            session_id=request.session_id,
                            turn_id=request.turn_id,
                            work_run_id=result.work_run_id,
                            expected_window_revision=result.window_revision,
                            apply_id=_stable_detach_apply_id(
                                turn_id=request.turn_id,
                                work_run_id=result.work_run_id,
                            ),
                        )
                    except Exception:
                        return _result(
                            request,
                            status="internal_interrupted",
                            revision=_current_window_revision(
                                request.session_id,
                                fallback=revision,
                            ),
                            work_run_ids=work_run_ids,
                            pending_question_attempt_ids=pending_question_attempt_ids,
                            last_outcome=result.outcome,
                            interruption_reason="safe_lane_detach_interrupted",
                        )
                revision = detached.window_state_version
                if result.outcome == "waiting_user":
                    # Entry 在 Task 范围路由继续执行，并且刻意不携带 Attempt/问题选择器。
                    # 在第一个持久问题后停止此 Task 泳道，使后续 Turn 始终拥有恰好一个继续执行权威。
                    # 外层根泳道协调器仍可推进另一个 Task。
                    return _result(
                        request,
                        status="waiting_user",
                        revision=revision,
                        work_run_ids=work_run_ids,
                        pending_question_attempt_ids=pending_question_attempt_ids,
                        last_outcome=result.outcome,
                    )
                result = None
                continue
            if result.outcome in {"verification_interrupted", "internal_interrupted"}:
                return _result(
                    request,
                    status=result.outcome,
                    revision=revision,
                    work_run_ids=work_run_ids,
                    pending_question_attempt_ids=pending_question_attempt_ids,
                    last_outcome=result.outcome,
                    interruption_reason=result.interruption_reason,
                )
            return _result(
                request,
                status="failed_closed",
                revision=revision,
                work_run_ids=work_run_ids,
                pending_question_attempt_ids=pending_question_attempt_ids,
                last_outcome=result.outcome,
                failure_code=result.failure_code or "task_node_work_run_failed_closed",
            )

        details = task_graph_store.get_insession_task_details(
            request.session_id,
            request.task_id,
        )
        if details is None or details.current_graph_revision is None:
            return _result(
                request,
                status="failed_closed",
                revision=revision,
                work_run_ids=work_run_ids,
                pending_question_attempt_ids=pending_question_attempt_ids,
                failure_code="task_graph_authority_unavailable",
            )
        if details.status.value == "completed":
            delivery_id = work_run_store.get_completed_task_final_delivery_id(
                session_id=request.session_id,
                task_id=request.task_id,
            )
            return TaskGraphWorkRunResult(
                status="completed",
                task_id=request.task_id,
                final_delivery_id=delivery_id,
                pending_question_attempt_ids=tuple(pending_question_attempt_ids),
                work_run_ids=tuple(work_run_ids),
                window_state_version=revision,
                last_work_run_outcome="delivery_ready",
            )
        if details.status.value == "cancelled":
            return _result(
                request,
                status="blocked",
                revision=revision,
                work_run_ids=work_run_ids,
                pending_question_attempt_ids=pending_question_attempt_ids,
            )
        if deadline is not None and deadline.expired():
            return _result(
                request,
                status="deadline_reached",
                revision=revision,
                work_run_ids=work_run_ids,
                pending_question_attempt_ids=pending_question_attempt_ids,
            )

        frontier = work_run_store.project_task_node_execution_frontier(
            session_id=request.session_id,
            turn_id=request.turn_id,
            task_id=request.task_id,
        )
        try:
            pending_for_task = tuple(
                item
                for item in continuation_store.list_pending_user_questions(
                    session_id=request.session_id
                )
                if isinstance(item, PendingUserQuestion)
                and item.subject.task_id == request.task_id
            )
        except Exception:
            return _result(
                request,
                status="failed_closed",
                revision=revision,
                work_run_ids=work_run_ids,
                pending_question_attempt_ids=pending_question_attempt_ids,
                failure_code="pending_question_authority_unreadable",
            )
        if len(pending_for_task) > 1:
            return _result(
                request,
                status="failed_closed",
                revision=revision,
                work_run_ids=work_run_ids,
                pending_question_attempt_ids=pending_question_attempt_ids,
                failure_code="pending_question_authority_ambiguous",
            )
        if pending_for_task:
            if not request.allow_user_input:
                return _result(
                    request,
                    status="failed_closed",
                    revision=revision,
                    work_run_ids=work_run_ids,
                    pending_question_attempt_ids=(
                        pending_for_task[0].question_attempt_id,
                    ),
                    failure_code="closed_world_user_input_forbidden",
                )
            pending = pending_for_task[0]
            waiting_candidates = tuple(
                candidate
                for candidate in frontier.recoverable
                if candidate.subject == pending.subject
                and candidate.work_run_id == pending.work_run_id
            )
            if (
                len(waiting_candidates) != 1
                or waiting_candidates[0].node_status != "awaiting_user"
                or waiting_candidates[0].node_state_version
                != pending.node_state_version
                or waiting_candidates[0].work_run_status
                is not WorkRunStatus.WAITING_USER
                or waiting_candidates[0].work_run_reason != "needs_input"
                or waiting_candidates[0].work_run_revision
                != pending.work_run_revision
                or waiting_candidates[0].current_attempt_id is not None
                or waiting_candidates[0].current_verification_request_id is not None
            ):
                return _result(
                    request,
                    status="failed_closed",
                    revision=revision,
                    work_run_ids=work_run_ids,
                    pending_question_attempt_ids=pending_question_attempt_ids,
                    failure_code="pending_question_authority_invalid",
                )
            if _turn_authorizes_waiting_user_answer(
                request=request,
                pending=pending,
            ):
                if dispatch_count >= profile.max_work_runs_per_turn:
                    _append_unique(
                        pending_question_attempt_ids,
                        pending.question_attempt_id,
                    )
                    return _result(
                        request,
                        status="turn_limit_reached",
                        revision=revision,
                        work_run_ids=work_run_ids,
                        pending_question_attempt_ids=(
                            pending_question_attempt_ids
                        ),
                        last_outcome="host_work_run_limit_reached",
                    )
                candidate = waiting_candidates[0]
                node_runtime = runtime_plan.runtime_for(candidate.subject)
                allowed_tools = tuple(
                    entry.registration.spec
                    for entry in node_runtime.catalog_snapshot.exposed()
                )
                dispatch_count += 1
                result = continue_waiting_user_task_node_work_run(
                    WorkRunTurnWaitingUserContinuationRequest(
                        session_id=request.session_id,
                        turn_id=request.turn_id,
                        pending_question=pending,
                        expected_window_revision=revision,
                        attempt_input_limits=profile.attempt_input_limits,
                        verification_input_limits=(
                            profile.verification_input_limits
                        ),
                        allow_user_input=request.allow_user_input,
                        paper_resources=node_runtime.paper_resources,
                    ),
                    catalog_snapshot=node_runtime.catalog_snapshot,
                    allowed_tools=allowed_tools,
                    attempt_provider=attempt,
                    verification_provider=verification,
                    emit=event_sink,
                    id_plan=_id_plan_from_work_run_id(candidate.work_run_id),
                    tool_bridge=node_runtime.tool_bridge,
                    deadline=deadline,
                    downstream_gate_factory=downstream_gate_factory,
                    model_call_authority_factory=(
                        _ordinary_task_node_model_authority_factory(
                            node_runtime,
                            factory=model_call_authority_factory,
                        )
                    ),
                    monotonic_clock=monotonic_clock,
                )
                continue
            _append_unique(
                pending_question_attempt_ids,
                pending.question_attempt_id,
            )
            return _result(
                request,
                status="waiting_user",
                revision=revision,
                work_run_ids=work_run_ids,
                pending_question_attempt_ids=pending_question_attempt_ids,
                last_outcome="waiting_user",
            )
        verification_marked = tuple(
            candidate
            for candidate in frontier.recoverable
            if candidate.current_verification_request_id is not None
            or candidate.work_run_reason
            in {"verification_pending", "verification_technical_failure"}
        )
        if len(verification_marked) > 1:
            return _result(
                request,
                status="failed_closed",
                revision=revision,
                work_run_ids=work_run_ids,
                pending_question_attempt_ids=pending_question_attempt_ids,
                failure_code="verification_recovery_authority_ambiguous",
            )
        if verification_marked:
            candidate = verification_marked[0]
            pending_shape = (
                candidate.work_run_status is WorkRunStatus.ACTIVE
                and candidate.work_run_reason == "verification_pending"
                and candidate.node_status == "active"
            )
            interrupted_shape = (
                candidate.work_run_status is WorkRunStatus.INTERRUPTED
                and candidate.work_run_reason == "verification_technical_failure"
                and candidate.node_status == "interrupted"
            )
            verification_request_id = candidate.current_verification_request_id
            if (
                pending_shape
                and verification_request_id is None
                and candidate.current_attempt_id is None
            ):
                try:
                    stored = work_run_store.get_work_run(
                        session_id=request.session_id,
                        work_run_id=candidate.work_run_id,
                    )
                    submitted = tuple(
                        item
                        for item in stored.attempts
                        if item.attempt.status is AttemptStatus.CLOSED
                        and item.action == "submit_output_window"
                    )
                    if len(submitted) != 1:
                        raise ValueError(
                            "unprepared verification has no unique submit Attempt"
                        )
                    submitted_attempt = submitted[0]
                    node_runtime = runtime_plan.runtime_for(candidate.subject)
                    id_plan = _id_plan_from_work_run_id(candidate.work_run_id)
                    if (
                        stored.work_run.subject != candidate.subject
                        or stored.work_run.status is not WorkRunStatus.ACTIVE
                        or stored.work_run.reason != "verification_pending"
                        or stored.work_run.revision != candidate.work_run_revision
                        or stored.current_attempt_id is not None
                        or stored.current_verification_request_id is not None
                        or id_plan.attempt_id(submitted_attempt.attempt.ordinal)
                        != submitted_attempt.attempt.attempt_id
                    ):
                        raise ValueError(
                            "unprepared verification crossed WorkRun authority"
                        )
                    verification_request_id = id_plan.verification_request_id(
                        submitted_attempt.attempt.ordinal
                    )
                except Exception:
                    return _result(
                        request,
                        status="failed_closed",
                        revision=revision,
                        work_run_ids=work_run_ids,
                        pending_question_attempt_ids=pending_question_attempt_ids,
                        failure_code="verification_recovery_authority_invalid",
                    )
                if dispatch_count >= profile.max_work_runs_per_turn:
                    return _result(
                        request,
                        status="turn_limit_reached",
                        revision=revision,
                        work_run_ids=work_run_ids,
                        pending_question_attempt_ids=pending_question_attempt_ids,
                        last_outcome="host_work_run_limit_reached",
                    )
                allowed_tools = tuple(
                    entry.registration.spec
                    for entry in node_runtime.catalog_snapshot.exposed()
                )
                dispatch_count += 1
                result = recover_unprepared_task_node_work_run_verification(
                    WorkRunTurnUnpreparedVerificationRecoveryRequest(
                        session_id=request.session_id,
                        turn_id=request.turn_id,
                        subject=candidate.subject,
                        work_run_id=candidate.work_run_id,
                        submitted_attempt_id=(
                            submitted_attempt.attempt.attempt_id
                        ),
                        verification_request_id=verification_request_id,
                        expected_work_run_revision=candidate.work_run_revision,
                        expected_progress_revision=(
                            stored.acceptance_progress.revision
                        ),
                        expected_output_revision=(
                            stored.output_window.output_revision
                        ),
                        expected_window_revision=revision,
                        attempt_input_limits=profile.attempt_input_limits,
                        verification_input_limits=profile.verification_input_limits,
                        allow_user_input=request.allow_user_input,
                        paper_resources=node_runtime.paper_resources,
                    ),
                    catalog_snapshot=node_runtime.catalog_snapshot,
                    allowed_tools=allowed_tools,
                    attempt_provider=attempt,
                    verification_provider=verification,
                    emit=event_sink,
                    id_plan=id_plan,
                    tool_bridge=node_runtime.tool_bridge,
                    deadline=deadline,
                    downstream_gate_factory=downstream_gate_factory,
                    model_call_authority_factory=(
                        _ordinary_task_node_model_authority_factory(
                            node_runtime,
                            factory=model_call_authority_factory,
                        )
                    ),
                    monotonic_clock=monotonic_clock,
                )
                continue
            if (
                (not pending_shape and not interrupted_shape)
                or verification_request_id is None
                or candidate.current_attempt_id is not None
            ):
                return _result(
                    request,
                    status="failed_closed",
                    revision=revision,
                    work_run_ids=work_run_ids,
                    pending_question_attempt_ids=pending_question_attempt_ids,
                    failure_code="verification_recovery_authority_invalid",
                )
            try:
                record = verification_store.get_task_node_verification_record(
                    session_id=request.session_id,
                    verification_request_id=verification_request_id,
                )
                stored = work_run_store.get_work_run(
                    session_id=request.session_id,
                    work_run_id=candidate.work_run_id,
                )
                expected_request_status = (
                    TaskNodeVerificationRequestStatus.PENDING
                    if pending_shape
                    else TaskNodeVerificationRequestStatus.INTERRUPTED
                )
                verification_request = record.request
                if (
                    record.result is not None
                    or verification_request.verification_request_id
                    != verification_request_id
                    or verification_request.session_id != request.session_id
                    or verification_request.work_run_id != candidate.work_run_id
                    or not isinstance(
                        verification_request.subject,
                        TaskNodeSubject,
                    )
                    or verification_request.subject != candidate.subject
                    or verification_request.subject.task_id != request.task_id
                    or verification_request.status is not expected_request_status
                    or verification_request.request_turn_id == request.turn_id
                    or verification_request.locked_work_run_revision
                    >= candidate.work_run_revision
                    or stored.work_run.subject != candidate.subject
                    or stored.work_run.status is not candidate.work_run_status
                    or stored.work_run.reason != candidate.work_run_reason
                    or stored.work_run.revision != candidate.work_run_revision
                    or stored.current_attempt_id is not None
                    or stored.current_verification_request_id
                    != verification_request_id
                ):
                    raise ValueError(
                        "verification recovery crossed durable authority"
                    )
                node_runtime = runtime_plan.runtime_for(candidate.subject)
                id_plan = _id_plan_from_work_run_id(candidate.work_run_id)
            except Exception:
                return _result(
                    request,
                    status="failed_closed",
                    revision=revision,
                    work_run_ids=work_run_ids,
                    pending_question_attempt_ids=pending_question_attempt_ids,
                    failure_code="verification_recovery_authority_invalid",
                )
            if dispatch_count >= profile.max_work_runs_per_turn:
                return _result(
                    request,
                    status="turn_limit_reached",
                    revision=revision,
                    work_run_ids=work_run_ids,
                    pending_question_attempt_ids=pending_question_attempt_ids,
                    last_outcome="host_work_run_limit_reached",
                )
            allowed_tools = tuple(
                entry.registration.spec
                for entry in node_runtime.catalog_snapshot.exposed()
            )
            dispatch_count += 1
            result = recover_task_node_work_run_verification(
                WorkRunTurnVerificationRecoveryRequest(
                    session_id=request.session_id,
                    turn_id=request.turn_id,
                    subject=candidate.subject,
                    work_run_id=candidate.work_run_id,
                    submitted_attempt_id=(
                        verification_request.submitted_attempt_id
                    ),
                    verification_request_id=verification_request_id,
                    expected_work_run_revision=candidate.work_run_revision,
                    expected_verification_request_revision=(
                        verification_request.revision
                    ),
                    expected_window_revision=revision,
                    attempt_input_limits=profile.attempt_input_limits,
                    verification_input_limits=profile.verification_input_limits,
                    allow_user_input=request.allow_user_input,
                    paper_resources=node_runtime.paper_resources,
                ),
                catalog_snapshot=node_runtime.catalog_snapshot,
                allowed_tools=allowed_tools,
                attempt_provider=attempt,
                verification_provider=verification,
                emit=event_sink,
                id_plan=id_plan,
                tool_bridge=node_runtime.tool_bridge,
                deadline=deadline,
                downstream_gate_factory=downstream_gate_factory,
                model_call_authority_factory=(
                    _ordinary_task_node_model_authority_factory(
                        node_runtime,
                        factory=model_call_authority_factory,
                    )
                ),
                monotonic_clock=monotonic_clock,
            )
            continue

        active_recoveries = tuple(
            candidate
            for candidate in frontier.recoverable
            if candidate.work_run_status is WorkRunStatus.ACTIVE
            and candidate.work_run_reason is None
            and candidate.current_verification_request_id is None
        )
        if len(active_recoveries) > 1:
            return _result(
                request,
                status="failed_closed",
                revision=revision,
                work_run_ids=work_run_ids,
                pending_question_attempt_ids=pending_question_attempt_ids,
                failure_code="active_recovery_ambiguous",
            )
        if active_recoveries:
            if dispatch_count >= profile.max_work_runs_per_turn:
                return _result(
                    request,
                    status="turn_limit_reached",
                    revision=revision,
                    work_run_ids=work_run_ids,
                    pending_question_attempt_ids=pending_question_attempt_ids,
                    last_outcome="host_work_run_limit_reached",
                )
            candidate = active_recoveries[0]
            node_runtime = runtime_plan.runtime_for(candidate.subject)
            allowed_tools = tuple(
                entry.registration.spec
                for entry in node_runtime.catalog_snapshot.exposed()
            )
            id_plan = _id_plan_from_work_run_id(candidate.work_run_id)
            recovery_attempt_id = candidate.current_attempt_id
            if recovery_attempt_id is None:
                try:
                    stored_recovery = work_run_store.get_work_run(
                        session_id=request.session_id,
                        work_run_id=candidate.work_run_id,
                    )
                except Exception:
                    stored_recovery = None
                if stored_recovery is None or not stored_recovery.attempts:
                    return _result(
                        request,
                        status="failed_closed",
                        revision=revision,
                        work_run_ids=work_run_ids,
                        pending_question_attempt_ids=pending_question_attempt_ids,
                        failure_code="active_idle_recovery_attempt_unavailable",
                    )
                recovery_attempt_id = stored_recovery.attempts[-1].attempt.attempt_id
            dispatch_count += 1
            result = resume_active_task_node_work_run(
                WorkRunTurnResumeRequest(
                    session_id=request.session_id,
                    turn_id=request.turn_id,
                    subject=candidate.subject,
                    work_run_id=candidate.work_run_id,
                    attempt_id=recovery_attempt_id,
                    expected_work_run_revision=candidate.work_run_revision,
                    expected_window_revision=revision,
                    attempt_input_limits=profile.attempt_input_limits,
                    verification_input_limits=profile.verification_input_limits,
                    allow_user_input=request.allow_user_input,
                    paper_resources=node_runtime.paper_resources,
                ),
                catalog_snapshot=node_runtime.catalog_snapshot,
                allowed_tools=allowed_tools,
                attempt_provider=attempt,
                verification_provider=verification,
                emit=event_sink,
                id_plan=id_plan,
                tool_bridge=node_runtime.tool_bridge,
                deadline=deadline,
                downstream_gate_factory=downstream_gate_factory,
                model_call_authority_factory=(
                    _ordinary_task_node_model_authority_factory(
                        node_runtime,
                        factory=model_call_authority_factory,
                    )
                ),
                monotonic_clock=monotonic_clock,
            )
            continue

        ready = tuple(
            candidate
            for candidate in frontier.ready_fresh
            if candidate.subject not in attempted_subjects
        )
        if ready:
            if dispatch_count >= profile.max_work_runs_per_turn:
                return _result(
                    request,
                    status="turn_limit_reached",
                    revision=revision,
                    work_run_ids=work_run_ids,
                    pending_question_attempt_ids=pending_question_attempt_ids,
                    last_outcome="host_work_run_limit_reached",
                )
            candidate = ready[0]
            node_runtime = runtime_plan.runtime_for(candidate.subject)
            allowed_tools = tuple(
                entry.registration.spec
                for entry in node_runtime.catalog_snapshot.exposed()
            )
            id_plan = _fresh_id_plan(
                session_id=request.session_id,
                turn_id=request.turn_id,
                subject=candidate.subject,
            )
            dispatch_count += 1
            result = run_new_task_node_work_run(
                WorkRunTurnApplicationRequest(
                    session_id=request.session_id,
                    turn_id=request.turn_id,
                    subject=candidate.subject,
                    expected_task_state_version=frontier.task_state_version,
                    expected_node_state_version=candidate.node_state_version,
                    expected_window_revision=revision,
                    attempt_input_limits=profile.attempt_input_limits,
                    verification_input_limits=profile.verification_input_limits,
                    allow_user_input=request.allow_user_input,
                    paper_resources=node_runtime.paper_resources,
                ),
                catalog_snapshot=node_runtime.catalog_snapshot,
                allowed_tools=allowed_tools,
                attempt_provider=attempt,
                verification_provider=verification,
                emit=event_sink,
                id_plan=id_plan,
                tool_bridge=node_runtime.tool_bridge,
                deadline=deadline,
                downstream_gate_factory=downstream_gate_factory,
                model_call_authority_factory=(
                    _ordinary_task_node_model_authority_factory(
                        node_runtime,
                        factory=model_call_authority_factory,
                    )
                ),
                monotonic_clock=monotonic_clock,
            )
            continue

        pending_for_task = tuple(
            item
            for item in continuation_store.list_pending_user_questions(
                session_id=request.session_id
            )
            if item.subject.task_id == request.task_id
        )
        for item in pending_for_task:
            _append_unique(pending_question_attempt_ids, item.question_attempt_id)
        if pending_question_attempt_ids:
            status: TaskGraphWorkRunStatus = "waiting_user"
        elif any(
            item.work_run_status is WorkRunStatus.WAITING_EXTERNAL
            for item in frontier.recoverable
        ):
            status = "waiting_external"
        elif any(
            item.work_run_status is WorkRunStatus.TURN_LIMIT_REACHED
            for item in frontier.recoverable
        ) or "turn_limit_reached" in safe_stop_outcomes:
            status = "turn_limit_reached"
        elif "work_run_failed" in safe_stop_outcomes:
            status = "work_run_failed"
        elif frontier.recoverable:
            status = "internal_interrupted"
        else:
            status = "blocked"
        return _result(
            request,
            status=status,
            revision=revision,
            work_run_ids=work_run_ids,
            pending_question_attempt_ids=pending_question_attempt_ids,
            interruption_reason=(
                "unsupported_recoverable_task_node_phase"
                if status == "internal_interrupted"
                else None
            ),
        )


def preflight_task_node_tool_runtimes(
    *,
    request: TaskGraphWorkRunRequest,
    default_runtime: TaskNodeToolRuntime,
    factory: TaskNodeToolRuntimeFactory | None,
) -> TaskNodeToolRuntimePlan:
    """在第一次 WorkRun 变更前解析每个当前节点。

    后期 P2/根运行时失败不得遗留已成功执行的 P1 前缀。Store 仍是图权威；此次预检仅将可执行
    Catalog 绑定到该已是当前版本的图中每个主体。
    """

    default_runtime = _with_execution_findings_runtime(
        default_runtime,
        enabled=request.profile.execution_findings_enabled,
    )
    _validate_node_tool_runtime(default_runtime, request=request)
    details = task_graph_store.get_insession_task_details(
        request.session_id,
        request.task_id,
    )
    if details is None or details.current_graph_revision is None:
        raise ValueError("TaskGraph authority is unavailable for node runtime preflight")
    bindings: list[TaskNodeToolRuntimeBinding] = []
    for node in details.nodes:
        node_id = node.get("insession_task_node_id")
        node_revision = node.get("node_revision")
        if (
            not isinstance(node_id, str)
            or not node_id
            or isinstance(node_revision, bool)
            or not isinstance(node_revision, int)
            or node_revision < 1
        ):
            raise ValueError("TaskGraph node identity is malformed")
        subject = TaskNodeSubject(
            task_id=request.task_id,
            graph_revision=details.current_graph_revision,
            node_id=node_id,
            node_revision=node_revision,
        )
        if factory is None:
            continue
        runtime = factory(subject)
        if not isinstance(runtime, TaskNodeToolRuntime):
            raise TypeError("node Tool runtime factory returned an invalid bundle")
        runtime = _with_execution_findings_runtime(
            runtime,
            enabled=request.profile.execution_findings_enabled,
        )
        _validate_node_tool_runtime(
            runtime,
            request=request,
            subject=subject,
        )
        bindings.append(TaskNodeToolRuntimeBinding(subject=subject, runtime=runtime))
    return TaskNodeToolRuntimePlan(
        session_id=request.session_id,
        task_id=request.task_id,
        graph_revision=details.current_graph_revision,
        default_runtime=default_runtime,
        bindings=tuple(bindings),
    )


def _with_execution_findings_runtime(
    runtime: TaskNodeToolRuntime,
    *,
    enabled: bool,
) -> TaskNodeToolRuntime:
    snapshot, bridge = augment_execution_findings_tool_runtime(
        catalog_snapshot=runtime.catalog_snapshot,
        tool_bridge=runtime.tool_bridge,
        enabled=enabled,
        strict_bridge_rebind=False,
    )
    if snapshot is runtime.catalog_snapshot and bridge is runtime.tool_bridge:
        return runtime
    return replace(
        runtime,
        catalog_snapshot=snapshot,
        tool_bridge=bridge,
    )


def _validate_node_tool_runtime_plan(
    plan: TaskNodeToolRuntimePlan,
    *,
    request: TaskGraphWorkRunRequest,
) -> None:
    if (
        not isinstance(plan, TaskNodeToolRuntimePlan)
        or plan.session_id != request.session_id
        or plan.task_id != request.task_id
    ):
        raise ValueError("node Tool runtime plan crossed Session/Task authority")
    details = task_graph_store.get_insession_task_details(
        request.session_id,
        request.task_id,
    )
    if details is None or details.current_graph_revision != plan.graph_revision:
        raise ValueError("node Tool runtime plan is stale for the current TaskGraph")
    _validate_node_tool_runtime(plan.default_runtime, request=request)
    seen: set[TaskNodeSubject] = set()
    for binding in plan.bindings:
        if (
            binding.subject.task_id != request.task_id
            or binding.subject.graph_revision != plan.graph_revision
            or binding.subject in seen
        ):
            raise ValueError("node Tool runtime plan contains an invalid binding")
        seen.add(binding.subject)
        _validate_node_tool_runtime(
            binding.runtime,
            request=request,
            subject=binding.subject,
        )
    if plan.bindings:
        expected = {
            TaskNodeSubject(
                task_id=request.task_id,
                graph_revision=plan.graph_revision,
                node_id=str(node["insession_task_node_id"]),
                node_revision=int(node["node_revision"]),
            )
            for node in details.nodes
        }
        if seen != expected:
            raise ValueError("node Tool runtime plan is incomplete for the current TaskGraph")


def _result(
    request: TaskGraphWorkRunRequest,
    *,
    status: TaskGraphWorkRunStatus,
    revision: int,
    work_run_ids: list[str],
    pending_question_attempt_ids: list[str],
    last_outcome: str | None = None,
    failure_code: str | None = None,
    interruption_reason: str | None = None,
) -> TaskGraphWorkRunResult:
    return TaskGraphWorkRunResult(
        status=status,
        task_id=request.task_id,
        pending_question_attempt_ids=tuple(pending_question_attempt_ids),
        work_run_ids=tuple(work_run_ids),
        window_state_version=revision,
        last_work_run_outcome=last_outcome,
        failure_code=failure_code,
        interruption_reason=interruption_reason,
    )


def _current_window_revision(session_id: str, *, fallback: int) -> int:
    try:
        window = session_store.get_turn_execution_window(session_id)
        if window is not None:
            revision = window.get("state_version")
            if isinstance(revision, int) and not isinstance(revision, bool) and revision >= 1:
                return revision
    except Exception:
        pass
    return fallback


def _turn_authorizes_waiting_user_answer(
    *,
    request: TaskGraphWorkRunRequest,
    pending: PendingUserQuestion,
) -> bool:
    """将一个待处理问题与 Entry 的精确回答意图泳道关联。"""

    if pending.question_turn_id == request.turn_id:
        return False
    try:
        manifest = task_graph_store.get_insession_task_execution_lane_manifest(
            session_id=request.session_id,
            turn_id=request.turn_id,
        )
    except Exception:
        return False
    lanes = tuple(getattr(manifest, "lanes", ()))
    if len(lanes) != 1:
        return False
    lane = lanes[0]
    if (
        getattr(lane, "insession_task_id", None) != request.task_id
        or getattr(lane, "execution_requested", None) is not True
    ):
        return False
    return any(
        getattr(item, "match_type", None) == "existing_root"
        and getattr(item, "execution_requested", None) is True
        for item in tuple(getattr(lane, "matches", ()))
    )


def _append_unique(items: list[str], value: str) -> None:
    if value not in items:
        items.append(value)


def _discard_event(_event: TurnEvent) -> None:
    return None


__all__ = [
    "TaskNodeToolRuntimeFactory",
    'TaskNodeToolRuntimeBinding',
    'TaskNodeToolRuntimePlan',
    'TaskNodeToolRuntime',
    'TaskGraphWorkRunProfile',
    'TaskGraphWorkRunRequest',
    'TaskGraphWorkRunResult',
    'TaskGraphWorkRunStatus',
    "preflight_task_node_tool_runtimes",
    "run_task_graph_work_runs",
]
