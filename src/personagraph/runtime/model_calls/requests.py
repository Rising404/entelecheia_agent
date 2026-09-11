"""Runtime entry 路径的共享有界模型请求包装器。"""

from __future__ import annotations

import time
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass
from time import monotonic
from typing import Generic, TypeVar
from uuid import uuid4

from personagraph.model_io.gateway import (
    ModelGatewayError,
    ModelResult,
    model_http_timeout_ceiling,
)
from personagraph.model_io.tier_bindings import ModelTierBinding, model_tier_binding_scope
from personagraph.model_io.output_validation import (
    ModelOutputValidationError as _ModelOutputValidationError,
)
from personagraph.model_io.prepared_request_contracts import (
    PreparedModelRequest as _PreparedModelRequest,
)
from .contracts import (
    DurableLogicalModelCallAuthority,
    DurableModelCallReplay,
    DurableModelCallStateGuardRejected,
)
from .attempt_preparation import prepare_model_attempt_request
from .policy import (
    MAX_MODEL_ATTEMPTS,
    MODEL_OUTPUT_LIMIT_FINISH_REASONS,
    backoff_delay_s,
)
from .observability import (
    record_model_output_rejection as _record_model_output_rejection,
    record_terminal_model_failure as _record_terminal_model_failure,
    runtime_error_code as _runtime_error_code,
)
from .output_repair import (
    rejected_response_sha256_text as _rejected_response_sha256_text,
    resolve_model_output_rejection,
)
from .quota import acquire_quota_dispatch_handle
from .request_policy import resolve_model_request_policy
from ...model_io.output_repair_contracts import (
    RuntimeModelOutputRepairFeedback,
)
from ..turn_deadline import (
    TurnDeadline as _TurnDeadline,
    TurnDeadlineExceeded as _TurnDeadlineExceeded,
)
from ..turn_events import RuntimeErrorCode, RuntimeStage, TurnEventStatus, new_turn_event


T = TypeVar("T")


@dataclass(frozen=True)
class ModelRequestResult(Generic[T]):
    value: T
    model_result: ModelResult
    model_call_id: str
    attempts: int
    replayed: bool = False


def _sleep(seconds: float) -> None:
    """让测试无需等待即可演练重试阶梯的间接层。"""

    time.sleep(seconds)


def _terminal_error(error: ModelGatewayError) -> ModelGatewayError:
    """关闭本逻辑调用的重试权，不修改 Provider/内层调用持有的原始异常。"""
    if not error.retryable:
        return error
    return ModelGatewayError(
        error.code, error.message, retryable=False,
        details={**error.details, "physical_error_retryable": True},
    )


def request_model_with_retry(
    *,
    turn_id: str,
    session_id: str | None,
    purpose: str,
    stage: RuntimeStage,
    prepare_request: Callable[[], _PreparedModelRequest],
    validate: Callable[[ModelResult], T],
    validate_replay: Callable[[ModelResult], T] | None = None,
    emit: Callable[[object], None],
    prepare_repair_request: Callable[
        [RuntimeModelOutputRepairFeedback, str], _PreparedModelRequest
    ]
    | None = None,
    repair_target_contract: str | None = None,
    max_attempts: int = MAX_MODEL_ATTEMPTS,
    deadline: _TurnDeadline | None = None,
    durable_call: DurableLogicalModelCallAuthority | None = None,
    logical_model_call_id: str | None = None,
) -> ModelRequestResult[T]:
    """使用 host 持有的重试语义调用一个逻辑模型请求。

    调用方提供已准备 provider 请求与有类型响应 parser。传输错误与格式错误的必需输出
    都只在共享有界 policy 内重试，且绝不越过 Turn 墙钟 deadline。wrapper 绝不决定
    路由或写入任务状态。

    ``prepare_request`` 在无 Provider I/O 的情况下序列化并准入精确 wire 请求。
    每次物理重试都会重新运行；
    首次 attempt 时则在持久化逻辑 reservation/replay 前运行。其返回对象的
    ``dispatch`` 只在物理权威状态与 STARTED 事件存在后运行。Repair 模式通过
    ``prepare_repair_request`` 遵循同一规则，并同时接收结构化反馈与精确的
    被拒响应正文。所有初始请求和 Repair 请求均只有这一条准备后分派路径。

    ``validate_replay`` 只解析已经通过持久化成功结算认证的规范结果；未提供时沿用
    ``validate``。任何新 Provider 响应（包括 Repair）始终只经过 ``validate``，不得
    根据响应内容或调用方字段改走 replay parser。

    阅读时分清三层计数：L1 Attempt 是外层决策轮；本 wrapper 负责一个 logical call；
    循环内才是 physical attempt。durable_call 是显式注入的能力，并非所有调用都有
    持久账本。顺序为 prepare/admit → reserve/replay → quota → physical begin
    → dispatch → validate → settle；传输退避和输出 repair 共用本次调用预算。
    正常落到循环末尾的失败会补一条终态 trajectory 摘要；此前直接抛出的 guard、
    deadline 等异常不保证经过该记录点，轨迹也不是完整状态机日志。
    """

    policy = resolve_model_request_policy(
        max_attempts=max_attempts,
        prepare_request=prepare_request,
        prepare_repair_request=prepare_repair_request,
        repair_target_contract=repair_target_contract,
        durable_call=durable_call,
        logical_model_call_id=logical_model_call_id,
    )
    repair_enabled = policy.repair_enabled
    effective_repair_target_contract = policy.repair_target_contract
    effective_max_attempts = policy.max_attempts
    recover_durable_repair_feedback = policy.recover_durable_repair_feedback
    model_call_id = (
        durable_call.semantic_call_id
        if durable_call is not None
        else logical_model_call_id or f"model_{uuid4().hex}"
    )
    last_error: ModelGatewayError | None = None
    last_cause: BaseException | None = None
    durable_reserved = False
    repair_feedback: RuntimeModelOutputRepairFeedback | None = None
    rejected_response_text: str | None = None
    durable_repair_recovered = False
    local_attempt = 0
    request_started = monotonic()
    while local_attempt < effective_max_attempts:
        if deadline is not None and deadline.expired():
            # 在启动前而非之后检查：无法在预算内完成的物理 attempt 只会增加 provider
            # 成本和用户已经无法容忍的额外延迟。
            deadline_error = _TurnDeadlineExceeded()
            emit(new_turn_event(
                turn_id=turn_id,
                session_id=session_id,
                stage=stage,
                status=TurnEventStatus.FAILED,
                model_call_id=model_call_id,
                model_attempt=max(1, local_attempt + 1),
                error_code=RuntimeErrorCode.TURN_DEADLINE_EXCEEDED,
            ))
            raise deadline_error from last_cause
        if durable_call is not None:
            # 此仅 Host 检查针对每个物理请求运行。它必须先于一次性逻辑 reservation
            # 与物理 append，使来源/generation 漂移不会创建新调用权威状态。
            durable_call.require_current_state()
        # 在规范 envelope 序列化并准入期间，同时冻结路由与剩余 HTTP 上限。此 block
        # 必须位于逻辑 reservation、物理权威状态与 STARTED 之前。
        prepare_remaining_s = (
            deadline.remaining_s() if deadline is not None else None
        )
        if prepare_remaining_s is not None and prepare_remaining_s <= 0.0:
            deadline_error = _TurnDeadlineExceeded()
            emit(new_turn_event(
                turn_id=turn_id,
                session_id=session_id,
                stage=stage,
                status=TurnEventStatus.FAILED,
                model_call_id=model_call_id,
                model_attempt=max(1, local_attempt + 1),
                error_code=RuntimeErrorCode.TURN_DEADLINE_EXCEEDED,
            ))
            raise deadline_error from last_cause
        prepared_dispatch = prepare_model_attempt_request(
            purpose=purpose,
            prepare_request=prepare_request,
            prepare_repair_request=prepare_repair_request,
            repair_feedback=repair_feedback,
            rejected_response_text=rejected_response_text,
            remaining_s=prepare_remaining_s,
            durable_call=durable_call,
        )

        # 准备可能耗时足以使路由权威或 Turn 租约过期。准入后、一次性持久化
        # reservation 前重新检查两者。
        if durable_call is not None:
            durable_call.require_current_state()
        post_prepare_remaining_s = (
            deadline.remaining_s() if deadline is not None else None
        )
        if (
            post_prepare_remaining_s is not None
            and post_prepare_remaining_s <= 0.0
        ):
            deadline_error = _TurnDeadlineExceeded()
            emit(new_turn_event(
                turn_id=turn_id,
                session_id=session_id,
                stage=stage,
                status=TurnEventStatus.FAILED,
                model_call_id=model_call_id,
                model_attempt=max(1, local_attempt + 1),
                error_code=RuntimeErrorCode.TURN_DEADLINE_EXCEEDED,
            ))
            raise deadline_error from last_cause
        if durable_call is not None and not durable_reserved:
            # 持久化 reservation 在任何物理 provider attempt 前完成。尤其是第十三次
            # 语义调用会在此抛出，因此调用 provider 零次。
            durable_call.reserve(turn_id=turn_id)
            durable_reserved = True
            replay = durable_call.replay_succeeded_result()
            if replay is not None:
                return _replay_durable_success(
                    replay,
                    durable_call=durable_call,
                    validate_replay=(
                        validate if validate_replay is None else validate_replay
                    ),
                    turn_id=turn_id,
                    session_id=session_id,
                    stage=stage,
                    emit=emit,
                )
        if (
            durable_call is not None
            and repair_enabled
            and not durable_repair_recovered
        ):
            assert recover_durable_repair_feedback is not None
            recovered = recover_durable_repair_feedback()
            if recovered is not None and not isinstance(
                recovered,
                RuntimeModelOutputRepairFeedback,
            ):
                raise durable_call.terminal_state_error(
                    "durable output-repair feedback has the wrong contract"
                )
            repair_feedback = recovered
            if repair_feedback is not None:
                recover_rejected_response = getattr(
                    durable_call,
                    "recover_output_repair_response",
                    None,
                )
                if not callable(recover_rejected_response):
                    raise durable_call.terminal_state_error(
                        "durable output repair cannot recover the rejected response"
                    )
                recovered_response = recover_rejected_response(repair_feedback)
                if not isinstance(recovered_response, str) or (
                    _rejected_response_sha256_text(recovered_response)
                    != repair_feedback.rejected_response_sha256
                ):
                    raise durable_call.terminal_state_error(
                        "durable rejected model response does not match repair feedback"
                    )
                rejected_response_text = recovered_response
            durable_repair_recovered = True
            if repair_feedback is not None:
                # 恢复的逻辑调用只能在其幂等逻辑 reservation 可用后揭示已持久化 repair
                # feedback。在打开下一物理记录前重新准入该精确 repair 变体；原始变体
                # 已在上方 reservation 前准入。
                repair_prepare_remaining_s = (
                    deadline.remaining_s() if deadline is not None else None
                )
                if (
                    repair_prepare_remaining_s is not None
                    and repair_prepare_remaining_s <= 0.0
                ):
                    deadline_error = _TurnDeadlineExceeded()
                    emit(new_turn_event(
                        turn_id=turn_id,
                        session_id=session_id,
                        stage=stage,
                        status=TurnEventStatus.FAILED,
                        model_call_id=model_call_id,
                        model_attempt=max(1, local_attempt + 1),
                        error_code=RuntimeErrorCode.TURN_DEADLINE_EXCEEDED,
                    ))
                    raise deadline_error from last_cause
                prepared_dispatch = prepare_model_attempt_request(
                    purpose=purpose,
                    prepare_request=prepare_request,
                    prepare_repair_request=prepare_repair_request,
                    repair_feedback=repair_feedback,
                    rejected_response_text=rejected_response_text,
                    remaining_s=repair_prepare_remaining_s,
                    durable_call=durable_call,
                    durable_repair_recovery=True,
                )
                durable_call.require_current_state()
                post_repair_prepare_remaining_s = (
                    deadline.remaining_s() if deadline is not None else None
                )
                if (
                    post_repair_prepare_remaining_s is not None
                    and post_repair_prepare_remaining_s <= 0.0
                ):
                    deadline_error = _TurnDeadlineExceeded()
                    emit(new_turn_event(
                        turn_id=turn_id,
                        session_id=session_id,
                        stage=stage,
                        status=TurnEventStatus.FAILED,
                        model_call_id=model_call_id,
                        model_attempt=max(1, local_attempt + 1),
                        error_code=RuntimeErrorCode.TURN_DEADLINE_EXCEEDED,
                    ))
                    raise deadline_error from last_cause
        # 在消费任何持久化物理 attempt 权威状态前，立即重新检查并冻结剩余 Turn 租约。
        # 较早的 ``expired`` 检查可能与逻辑 reservation/replay 恢复竞态。若竞态失败，
        # 则不存在需对账的 Provider dispatch，也不应计收物理 attempt 预算。
        remaining_s = deadline.remaining_s() if deadline is not None else None
        if remaining_s is not None and remaining_s <= 0.0:
            deadline_error = _TurnDeadlineExceeded()
            emit(new_turn_event(
                turn_id=turn_id,
                session_id=session_id,
                stage=stage,
                status=TurnEventStatus.FAILED,
                model_call_id=model_call_id,
                model_attempt=max(1, local_attempt + 1),
                error_code=RuntimeErrorCode.TURN_DEADLINE_EXCEEDED,
            ))
            raise deadline_error from last_cause
        try:
            quota_dispatch = acquire_quota_dispatch_handle(
                prepared_request=prepared_dispatch,
                model_call_id=model_call_id,
                wait_timeout_seconds=remaining_s,
            )
        except ModelGatewayError as exc:
            # 配额票据失效不等于整轮时间耗尽（例如进程/机器暂停后恢复）。
            # 只有共享 Turn 时钟确已过期，才把等待失败归因为整轮超时。
            if (
                exc.code == "MODEL_QUOTA_WAIT_TIMEOUT"
                and deadline is not None
                and deadline.expired()
            ):
                deadline_error = _TurnDeadlineExceeded()
                emit(new_turn_event(
                    turn_id=turn_id,
                    session_id=session_id,
                    stage=stage,
                    status=TurnEventStatus.FAILED,
                    model_call_id=model_call_id,
                    model_attempt=max(1, local_attempt + 1),
                    error_code=RuntimeErrorCode.TURN_DEADLINE_EXCEEDED,
                ))
                raise deadline_error from exc
            raise

        if quota_dispatch is not None:
            # 队列等待不是物理模型 attempt。因此只有准入成功后、追加任何持久化 attempt
            # 记录前，才重新检查可变来源权威状态与冷 Turn 时钟。复查失败会取消调用方
            # 持有的 ticket。
            try:
                if durable_call is not None:
                    durable_call.require_current_state()
                remaining_s = (
                    deadline.remaining_s() if deadline is not None else None
                )
                if remaining_s is not None and remaining_s <= 0.0:
                    raise _TurnDeadlineExceeded()
            except BaseException as exc:
                quota_dispatch.abandon_if_not_handed_off()
                if isinstance(exc, _TurnDeadlineExceeded):
                    emit(new_turn_event(
                        turn_id=turn_id,
                        session_id=session_id,
                        stage=stage,
                        status=TurnEventStatus.FAILED,
                        model_call_id=model_call_id,
                        model_attempt=max(1, local_attempt + 1),
                        error_code=RuntimeErrorCode.TURN_DEADLINE_EXCEEDED,
                    ))
                raise

        try:
            physical = (
                (
                    durable_call.begin_physical_attempt(
                        turn_id=turn_id,
                        max_physical_attempts=effective_max_attempts,
                        output_repair_enabled=True,
                        output_repair_feedback=repair_feedback,
                    )
                    if repair_enabled
                    else durable_call.begin_physical_attempt(
                        turn_id=turn_id,
                        max_physical_attempts=effective_max_attempts,
                    )
                )
                if durable_call is not None
                else None
            )
        except BaseException:
            if quota_dispatch is not None:
                quota_dispatch.abandon_if_not_handed_off()
            raise
        local_attempt += 1
        attempt = (
            physical.physical_ordinal
            if physical is not None
            else local_attempt
        )
        if physical is not None:
            model_call_id = physical.model_call_id
        started = monotonic()
        try:
            emit(new_turn_event(
                turn_id=turn_id,
                session_id=session_id,
                stage=stage,
                status=TurnEventStatus.STARTED,
                model_call_id=model_call_id,
                model_attempt=attempt,
            ))
        except Exception as exc:
            # 持久化物理记录在发出 STARTED 前追加，而 Provider I/O 只在 emitter 返回后
            # 开始。因此可知 event-sink 失败尚未到达 Provider，可授权后续精确重试；
            # 将记录留为 pending 会错误地要求外部对账。
            if durable_call is not None and physical is not None:
                durable_call.settle_physical_attempt(
                    turn_id=turn_id,
                    physical=physical,
                    outcome=(
                        "retryable_failure"
                        if attempt < effective_max_attempts
                        else "terminal_failure"
                    ),
                    result_fingerprint=durable_call.failure_fingerprint(
                        error=exc,
                        provider_result=None,
                    ),
                    provider_result=None,
                    error_code="RUNTIME_EVENT_EMIT_FAILED",
                )
            if quota_dispatch is not None:
                quota_dispatch.abandon_if_not_handed_off()
            raise
        model_result: ModelResult | None = None
        next_repair_feedback: RuntimeModelOutputRepairFeedback | None = None
        next_rejected_response_text: str | None = None
        try:
            frozen_binding = (
                getattr(durable_call, "model_binding", None)
                if durable_call is not None
                else None
            )
            if frozen_binding is not None and not isinstance(
                frozen_binding,
                ModelTierBinding,
            ):
                raise durable_call.terminal_state_error(
                    "durable model binding has the wrong contract"
                )
            binding_scope = (
                model_tier_binding_scope(frozen_binding)
                if frozen_binding is not None
                else nullcontext()
            )
            timeout_scope = (
                model_http_timeout_ceiling(remaining_s)
                if remaining_s is not None
                else nullcontext()
            )
            with binding_scope, timeout_scope:
                if quota_dispatch is not None:
                    model_result = quota_dispatch.dispatch(
                        model_call_id=model_call_id
                    )
                else:
                    model_result = prepared_dispatch.dispatch(
                        model_call_id=model_call_id
                    )
            if model_result.model_call_id is None:
                raise _ModelOutputValidationError(
                    "model response omitted its logical call ID",
                    retryable=False,
                )
            elif model_result.model_call_id != model_call_id:
                raise _ModelOutputValidationError(
                    "model response belongs to another logical call",
                    retryable=False,
                )
            if str(model_result.finish_reason or "").strip().lower() in (
                MODEL_OUTPUT_LIMIT_FINISH_REASONS
            ):
                # 在完全相同的物理输出上限下重放精确请求，无法修复 provider 声明的截断。
                # 这也会在 parser 可能误将非空前缀当作完整结构化结果前拒绝它。
                raise _ModelOutputValidationError(
                    "model response ended at the output-token limit",
                    retryable=False,
                )
            value = validate(model_result)
            success_fingerprint = (
                durable_call.success_fingerprint(model_result)
                if durable_call is not None
                else None
            )
        except _ModelOutputValidationError as exc:
            last_cause = exc
            rejection = resolve_model_output_rejection(
                validation_error=exc,
                model_result=model_result,
                repair_enabled=repair_enabled,
                repair_target_contract=effective_repair_target_contract,
                rejected_physical_ordinal=attempt,
            )
            next_repair_feedback = rejection.next_repair_feedback
            next_rejected_response_text = rejection.rejected_response_text
            last_error = ModelGatewayError(
                "MODEL_BAD_RESPONSE",
                "Model response did not satisfy the required runtime contract.",
                retryable=rejection.retryable,
                details={"purpose": purpose, "reason": "invalid_typed_output"},
            )
            _record_model_output_rejection(
                purpose=purpose,
                turn_id=turn_id,
                session_id=session_id,
                model_call_id=model_call_id,
                logical_model_call_id=(
                    durable_call.semantic_call_id
                    if durable_call is not None
                    else logical_model_call_id or model_call_id
                ),
                physical_ordinal=attempt,
                feedback=next_repair_feedback,
                fallback_code=exc.repair_code,
                model_result=model_result,
                repair_scheduled=(
                    next_repair_feedback is not None
                    and rejection.retryable
                    and attempt < effective_max_attempts
                    and (deadline is None or not deadline.expired())
                ),
            )
        except ModelGatewayError as exc:
            if quota_dispatch is not None:
                quota_dispatch.abandon_if_not_handed_off()
            last_cause = exc
            # 已取得本次响应后，GatewayError 来自 validator 的下游能力（例如
            # reviewer），不是本次 Provider 传输失败。不能重新生成已收到的主响应，
            # 更不能拿外层预算重置下游调用的耗尽预算。
            last_error = _terminal_error(exc) if model_result is not None else exc
        except Exception as exc:
            # Provider adapter 应将传输错误转成 ModelGatewayError。若泄漏其他异常，
            # 通用路径会保留该异常。M2 仍会在传播前将已追加物理请求结算为终态。
            if quota_dispatch is not None:
                quota_dispatch.abandon_if_not_handed_off()
            if durable_call is not None and physical is not None:
                durable_call.settle_physical_attempt(
                    turn_id=turn_id,
                    physical=physical,
                    outcome="terminal_failure",
                    result_fingerprint=durable_call.failure_fingerprint(
                        error=exc,
                        provider_result=model_result,
                    ),
                    provider_result=model_result,
                    error_code=type(exc).__name__,
                )
                raise durable_call.terminal_state_error(
                    "provider request ended in a durable terminal failure"
                ) from exc
            raise
        else:
            if durable_call is not None and physical is not None:
                assert success_fingerprint is not None
                try:
                    # provider 可能与本地来源/generation 变化发生竞态。绝不将现已过期的
                    # 响应结算为成功。
                    durable_call.require_current_state()
                except DurableModelCallStateGuardRejected as exc:
                    durable_call.settle_physical_attempt(
                        turn_id=turn_id,
                        physical=physical,
                        outcome="terminal_failure",
                        result_fingerprint=durable_call.failure_fingerprint(
                            error=exc,
                            provider_result=model_result,
                        ),
                        provider_result=model_result,
                        error_code=type(exc).__name__,
                    )
                    emit(new_turn_event(
                        turn_id=turn_id,
                        session_id=session_id,
                        stage=stage,
                        status=TurnEventStatus.FAILED,
                        duration_ms=max(
                            0,
                            round((monotonic() - started) * 1000),
                        ),
                        model_call_id=model_call_id,
                        model_attempt=attempt,
                        error_code=(
                            RuntimeErrorCode.MODEL_CONFIGURATION_FAILURE
                        ),
                        retryable=False,
                    ))
                    raise durable_call.terminal_state_error(
                        "model-call authority changed during the provider request"
                    ) from exc
                durable_call.settle_physical_attempt(
                    turn_id=turn_id,
                    physical=physical,
                    outcome="succeeded",
                    result_fingerprint=success_fingerprint,
                    typed_result=durable_call.typed_result_payload(
                        model_result=model_result,
                        value=value,
                    ),
                    provider_result=model_result,
                )
            emit(new_turn_event(
                turn_id=turn_id,
                session_id=session_id,
                stage=stage,
                status=TurnEventStatus.COMPLETED,
                duration_ms=max(0, round((monotonic() - started) * 1000)),
                model_call_id=model_call_id,
                model_attempt=attempt,
            ))
            return ModelRequestResult(
                value=value,
                model_result=model_result,
                model_call_id=model_call_id,
                attempts=attempt,
            )

        assert last_error is not None
        if durable_call is not None and physical is not None:
            try:
                # 可重试传输/验证失败可能和成功响应一样与可变来源权威状态发生竞态。
                # 在记录 retryable_failure 前检查，使漂移请求在其现有物理记录上终态化，
                # 而不是变成不可恢复的下一 attempt guard 拒绝。
                durable_call.require_current_state()
            except DurableModelCallStateGuardRejected as exc:
                durable_call.settle_physical_attempt(
                    turn_id=turn_id,
                    physical=physical,
                    outcome="terminal_failure",
                    result_fingerprint=durable_call.failure_fingerprint(
                        error=exc,
                        provider_result=model_result,
                    ),
                    provider_result=model_result,
                    error_code=type(exc).__name__,
                )
                emit(new_turn_event(
                    turn_id=turn_id,
                    session_id=session_id,
                    stage=stage,
                    status=TurnEventStatus.FAILED,
                    duration_ms=max(
                        0,
                        round((monotonic() - started) * 1000),
                    ),
                    model_call_id=model_call_id,
                    model_attempt=attempt,
                    error_code=RuntimeErrorCode.MODEL_CONFIGURATION_FAILURE,
                    retryable=False,
                ))
                raise durable_call.terminal_state_error(
                    "model-call authority changed during the provider request"
                ) from exc
        retryable = (
            bool(last_error.retryable)
            and attempt < effective_max_attempts
        )
        if durable_call is not None and physical is not None:
            settlement_values: dict[str, object] = dict(
                turn_id=turn_id,
                physical=physical,
                outcome=(
                    "retryable_failure" if retryable else "terminal_failure"
                ),
                result_fingerprint=durable_call.failure_fingerprint(
                    error=last_error,
                    provider_result=model_result,
                ),
                provider_result=model_result,
                error_code=last_error.code,
            )
            if retryable and next_repair_feedback is not None:
                settlement_values["next_output_repair_feedback"] = (
                    next_repair_feedback
                )
                settlement_values["rejected_response_text"] = (
                    next_rejected_response_text
                )
            durable_call.settle_physical_attempt(**settlement_values)
        emit(new_turn_event(
            turn_id=turn_id,
            session_id=session_id,
            stage=stage,
            status=TurnEventStatus.FAILED,
            duration_ms=max(0, round((monotonic() - started) * 1000)),
            model_call_id=model_call_id,
            model_attempt=attempt,
            error_code=_runtime_error_code(last_error),
            retryable=retryable,
        ))
        if not retryable:
            break
        if next_repair_feedback is not None:
            repair_feedback = next_repair_feedback
            rejected_response_text = next_rejected_response_text
        if last_error.code != "MODEL_BAD_RESPONSE":
            # 格式错误的有类型输出是我们自己的解析 verdict，并不表示 provider 需要更多
            # 空间；只有传输级失败会等待。等待本身受限，使退避绝不会超出 Turn。
            delay = backoff_delay_s(attempt)
            if deadline is not None:
                delay = min(delay, deadline.remaining_s())
            _sleep(delay)

    assert last_error is not None
    # 退出循环即本逻辑调用终止；不得把最后一次物理错误的 retryable 标记交给
    # 上层，再开启一组相同请求。原错误和细节仍通过 cause 保留。
    last_error = _terminal_error(last_error)
    _record_terminal_model_failure(
        purpose=purpose,
        turn_id=turn_id,
        session_id=session_id,
        model_call_id=model_call_id,
        reason_code=last_error.code,
        attempts=attempt,
        duration_ms=max(0, round((monotonic() - request_started) * 1000)),
    )
    raise last_error from last_cause


def _replay_durable_success(
    replay: DurableModelCallReplay,
    *,
    durable_call: DurableLogicalModelCallAuthority,
    validate_replay: Callable[[ModelResult], T],
    turn_id: str,
    session_id: str | None,
    stage: RuntimeStage,
    emit: Callable[[object], None],
) -> ModelRequestResult[T]:
    """重新验证一个已存储的有类型成功结果，而不打开 provider I/O。"""

    model_result = replay.model_result
    model_call_id = model_result.model_call_id
    assert model_call_id is not None
    try:
        value = validate_replay(model_result)
        durable_call.require_current_state()
    except Exception as exc:
        raise durable_call.terminal_state_error(
            "durable typed model result failed replay validation"
        ) from exc
    emit(new_turn_event(
        turn_id=turn_id,
        session_id=session_id,
        stage=stage,
        status=TurnEventStatus.COMPLETED,
        duration_ms=0,
        model_call_id=model_call_id,
        model_attempt=replay.physical_ordinal,
    ))
    return ModelRequestResult(
        value=value,
        model_result=model_result,
        model_call_id=model_call_id,
        attempts=replay.physical_ordinal,
        replayed=True,
    )


__all__ = ["ModelRequestResult", "request_model_with_retry"]
