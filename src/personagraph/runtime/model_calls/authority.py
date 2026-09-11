"""由 Runtime 账本支持的持久化、provider 中立模型调用权威状态。"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
import hashlib
import json
import re

from pydantic import BaseModel

from personagraph.model_io.tier_bindings import ModelTierBinding
from personagraph.model_io.contracts import ModelResult
from personagraph.model_io.gateway_core import ModelGatewayError
from .contracts import (
    DurableModelCallOutcome,
    DurableModelCallReplay,
    DurableModelCallStateGuardRejected,
    DurableModelCallTerminalState,
    DurablePhysicalModelAttempt,
    RuntimeModelLogicalRequest,
    RuntimeModelLedgerStore,
    RuntimeModelPhysicalAttemptRequest,
    RuntimeModelPhysicalAttemptSettlement,
    RuntimeModelPhysicalOutcome,
    RuntimeModelTypedResult,
    RuntimeModelUsage,
    runtime_model_dispatch_request_sha256,
)
from .recovery import (
    RuntimeModelCallRecoverySnapshot,
    project_runtime_model_call_recovery,
)
from ...model_io.output_repair_contracts import (
    RuntimeModelOutputRepairFeedback,
    RuntimeModelOutputRepairProtocol,
)


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ERROR_CODE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,199}$")


class RuntimeModelCallAuthorityError(DurableModelCallTerminalState):
    """通用 Runtime 模型调用权威状态无法安全推进。"""


class RuntimeModelCallWaitingExternal(RuntimeModelCallAuthorityError):
    """已分派的物理请求没有可对账的持久化结果。"""


TypedReplayPayloadBuilder = Callable[[ModelResult, object], object]
StateGuardSha256 = Callable[[], str]


@dataclass(frozen=True, slots=True)
class RuntimeLogicalModelCallAuthority:
    """供 ``request_model_with_retry`` 消费的一个精确逻辑请求。

    调用方提供请求专用的编码器，因为绑定 Host 的返回 DTO 不一定就是面向
    Provider 的 JSON 信封。其输出必须恰好能在重放时被常规响应验证器再次消费。
    """

    logical_request: RuntimeModelLogicalRequest
    state_guard_sha256: StateGuardSha256 = field(repr=False, compare=False)
    typed_replay_payload_builder: TypedReplayPayloadBuilder = field(
        repr=False,
        compare=False,
    )
    store: RuntimeModelLedgerStore = field(repr=False, compare=False)
    # 逻辑请求可以跨越 Turn 租约交接继续存在。其持久化语义请求与来源 guard
    # 保持不可变，而每次新的物理分派都必须由针对当前租约重新推导的 guard
    # 加以围栏保护。``None`` 保留原有的单 Turn 行为。
    dispatch_state_guard_sha256: str | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    # 由调用方持有的不透明证明，表明 factory 已准入精确的当前分派绑定。
    # 通用账本从不解释该摘要。
    dispatch_binding_sha256: str | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    # 仅由在分派时确实转发同一 key 的 Provider 专用组合提供此值。通用的
    # OpenAI 兼容和 Anthropic 兼容聊天 adapter 将其保留为 ``None``；Runtime
    # 绝不能根据本地 ID 虚构 Provider 幂等支持。
    provider_idempotency_key: str | None = field(default=None, repr=False)
    # 携带凭据的绑定，仅在分派此精确逻辑调用时于内存中使用。持久化请求会保存
    # 其非机密 endpoint 指纹；API key 绝不能进入账本。
    model_binding: ModelTierBinding | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    def __post_init__(self) -> None:
        if not isinstance(self.logical_request, RuntimeModelLogicalRequest):
            raise TypeError("logical_request must be RuntimeModelLogicalRequest")
        if not callable(self.state_guard_sha256):
            raise TypeError("state_guard_sha256 must be callable")
        if not callable(self.typed_replay_payload_builder):
            raise TypeError("typed_replay_payload_builder must be callable")
        for name, value in (
            ("dispatch_state_guard_sha256", self.dispatch_state_guard_sha256),
            ("dispatch_binding_sha256", self.dispatch_binding_sha256),
        ):
            if value is not None and not _SHA256.fullmatch(value):
                raise ValueError(f"{name} must be a canonical SHA-256")
        if self.provider_idempotency_key is not None and (
            not self.provider_idempotency_key.strip()
            or self.provider_idempotency_key != self.provider_idempotency_key.strip()
            or "\x00" in self.provider_idempotency_key
            or len(self.provider_idempotency_key) > 300
        ):
            raise ValueError(
                "provider_idempotency_key must be canonical 1..300 non-NUL text"
            )
        if self.model_binding is not None and not isinstance(
            self.model_binding,
            ModelTierBinding,
        ):
            raise TypeError("model_binding must be ModelTierBinding")

    @property
    def semantic_call_id(self) -> str:
        return self.logical_request.logical_call_id

    @property
    def frozen_max_physical_attempts(self) -> int:
        """由持久化请求冻结的物理 attempt 总授权量。"""

        return self.logical_request.max_physical_attempts

    def require_current_state(self) -> None:
        try:
            current = self.state_guard_sha256()
        except DurableModelCallStateGuardRejected:
            raise
        except Exception as exc:
            raise DurableModelCallStateGuardRejected(
                "model-call state guard could not rederive current authority"
            ) from exc
        expected = (
            self.dispatch_state_guard_sha256
            or self.logical_request.state_guard_sha256
        )
        if (
            not isinstance(current, str)
            or not _SHA256.fullmatch(current)
            or current != expected
        ):
            raise DurableModelCallStateGuardRejected(
                "model-call state authority no longer matches its frozen request"
            )

    def reserve(self, *, turn_id: str) -> object:
        if not turn_id:
            raise ValueError("turn_id must be non-empty")
        return self.store.reserve_runtime_model_logical_call(
            request=self.logical_request
        )

    def replay_succeeded_result(self) -> DurableModelCallReplay | None:
        stored = self._stored()
        attempts = stored.physical_attempts
        if not attempts:
            return None
        last = attempts[-1]
        settlement = last.settlement
        if settlement is None:
            raise RuntimeModelCallWaitingExternal(
                "pending Provider dispatch requires external reconciliation"
            )
        if settlement.outcome is RuntimeModelPhysicalOutcome.RETRYABLE_FAILURE:
            return None
        if settlement.outcome is RuntimeModelPhysicalOutcome.UNCERTAIN:
            raise RuntimeModelCallWaitingExternal(
                "uncertain Provider response requires external reconciliation"
            )
        if settlement.outcome is RuntimeModelPhysicalOutcome.TERMINAL_FAILURE:
            raise RuntimeModelCallAuthorityError(
                "logical model call already has a terminal failure"
            )
        typed = settlement.typed_result
        if typed is None:
            raise RuntimeModelCallAuthorityError(
                "successful model receipt has no replayable typed result"
            )
        usage = settlement.usage
        return DurableModelCallReplay(
            model_result=ModelResult(
                reply=typed.result_json,
                provider=self.logical_request.provider,
                model=self.logical_request.model,
                latency_ms=0,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                cache_read_tokens=usage.cache_read_tokens,
                cache_write_tokens=usage.cache_write_tokens,
                finish_reason=settlement.finish_reason or "durable_typed_replay",
                model_call_id=last.request.model_call_id,
                purpose=self.logical_request.purpose,
            ),
            physical_ordinal=last.request.physical_ordinal,
        )

    def inspect_recovery(self) -> RuntimeModelCallRecoverySnapshot:
        """投影精确的持久化状态，但不授予重试权限。"""

        stored = self._stored()
        return project_runtime_model_call_recovery(
            logical_request=self.logical_request,
            physical_attempts=stored.physical_attempts,
        )

    def settle_pending_external(
        self,
        *,
        turn_id: str,
        outcome: DurableModelCallOutcome,
        result_fingerprint: str,
        provider_request_id: str | None = None,
        typed_result: object | None = None,
        provider_result: object | None = None,
        error_code: str | None = None,
    ) -> RuntimeModelCallRecoverySnapshot:
        """在对账后显式结算一个仍处于 pending 状态的分派。

        ``uncertain`` 回执已经不可变，不能通过此接缝转换成重试授权。Provider
        专用 reconciler 必须将其保持为 ``waiting_uncertain``，直至存在单独的
        持久化解决机制；此方法绝不会盲目重发或覆盖该回执。
        """

        stored = self._stored()
        attempts = stored.physical_attempts
        if not attempts or attempts[-1].settlement is not None:
            raise RuntimeModelCallWaitingExternal(
                "only a pending Provider dispatch accepts external settlement"
            )
        self.settle_physical_attempt(
            turn_id=turn_id,
            physical=attempts[-1].request,
            outcome=outcome,
            result_fingerprint=result_fingerprint,
            provider_request_id=provider_request_id,
            typed_result=typed_result,
            provider_result=provider_result,
            error_code=error_code,
        )
        return self.inspect_recovery()

    def begin_physical_attempt(
        self,
        *,
        turn_id: str,
        max_physical_attempts: int,
        output_repair_enabled: bool = False,
        output_repair_feedback: RuntimeModelOutputRepairFeedback | None = None,
    ) -> RuntimeModelPhysicalAttemptRequest:
        if not 1 <= max_physical_attempts <= self.logical_request.max_physical_attempts:
            raise ValueError(
                "wrapper physical-attempt bound exceeds logical request authority"
            )
        frozen_repair_protocol = self.logical_request.output_repair_protocol
        if output_repair_enabled != (frozen_repair_protocol is not None):
            raise RuntimeModelCallAuthorityError(
                "physical output-repair mode differs from the logical request"
            )
        stored = self._stored()
        attempts = stored.physical_attempts
        expected_repair_feedback: RuntimeModelOutputRepairFeedback | None = None
        if attempts:
            last = attempts[-1]
            if last.settlement is None:
                raise RuntimeModelCallWaitingExternal(
                    "pending Provider dispatch requires external reconciliation"
                )
            if (
                last.settlement.outcome
                is not RuntimeModelPhysicalOutcome.RETRYABLE_FAILURE
            ):
                raise RuntimeModelCallAuthorityError(
                    "logical model call already has a terminal physical outcome"
                )
            if last.settlement.error_code == "MODEL_BAD_RESPONSE":
                expected_repair_feedback = (
                    last.settlement.next_output_repair_feedback
                )
            else:
                expected_repair_feedback = last.request.output_repair_feedback
            last_repair_enabled = bool(last.request.output_repair_enabled)
            if last_repair_enabled != output_repair_enabled:
                raise RuntimeModelCallAuthorityError(
                    "physical retry changed its durable output-repair mode"
                )
        if output_repair_feedback != expected_repair_feedback:
            raise RuntimeModelCallAuthorityError(
                "physical retry does not match the repair feedback authorized "
                "by the preceding immutable attempt"
            )
        ordinal = len(attempts) + 1
        if ordinal > max_physical_attempts:
            raise RuntimeModelCallAuthorityError(
                "logical model call physical-attempt limit is exhausted"
            )
        identity_hash = _fingerprint(
            {
                "contract": "runtime-model-physical-attempt-id-v1",
                "logical_call_id": self.semantic_call_id,
                "logical_request_binding_sha256": (
                    self.logical_request.binding_sha256
                ),
                "physical_ordinal": ordinal,
            }
        )
        if output_repair_enabled and self.provider_idempotency_key is not None:
            raise RuntimeModelCallAuthorityError(
                "durable output repair requires a Provider dispatch seam that "
                "can bind a distinct idempotency key to each request body"
            )
        if output_repair_feedback is not None:
            if not isinstance(
                output_repair_feedback,
                RuntimeModelOutputRepairFeedback,
            ):
                raise TypeError(
                    "output_repair_feedback has an unsupported contract"
                )
            if (
                frozen_repair_protocol
                is not RuntimeModelOutputRepairProtocol.FOUR_MESSAGE_WHOLE_RESPONSE_REGENERATION
            ):
                raise RuntimeModelCallAuthorityError(
                    "output-repair feedback differs from the frozen protocol"
                )
            if output_repair_feedback.rejected_physical_ordinal >= ordinal:
                raise RuntimeModelCallAuthorityError(
                    "output-repair feedback must reference an earlier physical attempt"
                )
            if (
                output_repair_feedback.target_contract
                != self.logical_request.output_repair_target_contract
            ):
                raise RuntimeModelCallAuthorityError(
                    "output-repair target differs from the logical result contract"
                )
        if output_repair_feedback is not None and not output_repair_enabled:
            raise RuntimeModelCallAuthorityError(
                "output-repair feedback requires durable repair mode"
            )
        dispatch_request_sha256 = runtime_model_dispatch_request_sha256(
            logical_request_sha256=self.logical_request.request_sha256,
            output_repair_feedback=output_repair_feedback,
        )
        dispatch_sha256 = _fingerprint(
            {
                "contract": "runtime-model-dispatch-authority-v1",
                "logical_request_binding_sha256": (
                    self.logical_request.binding_sha256
                ),
                "physical_ordinal": ordinal,
                "started_turn_id": turn_id,
                "provider_idempotency_key": self.provider_idempotency_key,
                "dispatch_request_sha256": dispatch_request_sha256,
                "state_guard_sha256": (
                    self.dispatch_state_guard_sha256
                    or self.logical_request.state_guard_sha256
                ),
            }
        )
        physical = RuntimeModelPhysicalAttemptRequest.create(
            physical_attempt_id=f"rmpa_v1_{identity_hash}",
            physical_attempt_key=f"rmcall_v1:{identity_hash}:{ordinal}",
            logical_call_id=self.semantic_call_id,
            logical_request_binding_sha256=self.logical_request.binding_sha256,
            physical_ordinal=ordinal,
            started_turn_id=turn_id,
            provider=self.logical_request.provider,
            model=self.logical_request.model,
            endpoint_fingerprint=self.logical_request.endpoint_fingerprint,
            request_sha256=self.logical_request.request_sha256,
            dispatch_request_sha256=dispatch_request_sha256,
            output_repair_enabled=output_repair_enabled,
            output_repair_feedback=output_repair_feedback,
            provider_idempotency_key=self.provider_idempotency_key,
            dispatch_authority_sha256=dispatch_sha256,
        )
        result = self.store.append_runtime_model_physical_attempt(request=physical)
        stored_physical_id = getattr(result, "physical_attempt_id", None)
        if stored_physical_id not in {None, physical.physical_attempt_id}:
            raise RuntimeModelCallAuthorityError(
                "model Store returned a cross-bound physical attempt"
            )
        return physical

    def settle_physical_attempt(
        self,
        *,
        turn_id: str,
        physical: DurablePhysicalModelAttempt,
        outcome: DurableModelCallOutcome,
        result_fingerprint: str,
        provider_request_id: str | None = None,
        typed_result: object | None = None,
        provider_result: object | None = None,
        error_code: str | None = None,
        next_output_repair_feedback: RuntimeModelOutputRepairFeedback | None = None,
        rejected_response_text: str | None = None,
    ) -> None:
        if not isinstance(physical, RuntimeModelPhysicalAttemptRequest):
            raise RuntimeModelCallAuthorityError(
                "generic ledger settlement requires its exact physical request DTO"
            )
        if not _SHA256.fullmatch(result_fingerprint):
            raise RuntimeModelCallAuthorityError(
                "model outcome fingerprint must be a canonical SHA-256"
            )
        outcome_value = RuntimeModelPhysicalOutcome(outcome)
        model_result = provider_result if isinstance(provider_result, ModelResult) else None
        if model_result is not None and (
            model_result.provider != self.logical_request.provider
            or model_result.model != self.logical_request.model
            or model_result.model_call_id != physical.model_call_id
        ):
            raise RuntimeModelCallAuthorityError(
                "Provider result differs from physical request authority"
            )
        persisted_typed = None
        if outcome_value is RuntimeModelPhysicalOutcome.SUCCEEDED:
            if typed_result is None:
                raise RuntimeModelCallAuthorityError(
                    "successful model settlement requires a replay payload"
                )
            persisted_typed = RuntimeModelTypedResult.create(
                result_contract=self.logical_request.typed_result_contract,
                result_payload=_jsonable(typed_result),
            )
            normalized_error_code = None
        else:
            if typed_result is not None:
                raise RuntimeModelCallAuthorityError(
                    "failed/uncertain model settlement cannot carry typed output"
                )
            normalized_error_code = _normalize_error_code(error_code)
        if next_output_repair_feedback is not None:
            if not isinstance(
                next_output_repair_feedback,
                RuntimeModelOutputRepairFeedback,
            ):
                raise TypeError(
                    "next_output_repair_feedback has an unsupported contract"
                )
            if (
                self.logical_request.output_repair_protocol
                is not RuntimeModelOutputRepairProtocol.FOUR_MESSAGE_WHOLE_RESPONSE_REGENERATION
            ):
                raise RuntimeModelCallAuthorityError(
                    "output-repair feedback differs from the frozen protocol"
                )
            if (
                next_output_repair_feedback.target_contract
                != self.logical_request.output_repair_target_contract
            ):
                raise RuntimeModelCallAuthorityError(
                    "output-repair target differs from the logical result contract"
                )
            if (
                not physical.output_repair_enabled
                or outcome_value
                is not RuntimeModelPhysicalOutcome.RETRYABLE_FAILURE
                or normalized_error_code != "MODEL_BAD_RESPONSE"
                or next_output_repair_feedback.rejected_physical_ordinal
                != physical.physical_ordinal
            ):
                raise RuntimeModelCallAuthorityError(
                    "output-repair feedback requires the matching retryable "
                    "MODEL_BAD_RESPONSE settlement"
                )
            try:
                rejected_provider_bytes = (
                    None
                    if model_result is None
                    else model_result.reply.encode("utf-8")
                )
            except UnicodeEncodeError as exc:
                raise RuntimeModelCallAuthorityError(
                    "rejected Provider response is not valid UTF-8 text"
                ) from exc
            if model_result is None or (
                next_output_repair_feedback.rejected_response_sha256
                != hashlib.sha256(rejected_provider_bytes).hexdigest()
            ):
                raise RuntimeModelCallAuthorityError(
                    "output-repair feedback does not bind the rejected Provider "
                    "response"
                )
        elif (
            physical.output_repair_enabled
            and outcome_value
            is RuntimeModelPhysicalOutcome.RETRYABLE_FAILURE
            and normalized_error_code == "MODEL_BAD_RESPONSE"
        ):
            raise RuntimeModelCallAuthorityError(
                "durable repair mode requires feedback on every retryable "
                "typed-output rejection"
            )
        if (next_output_repair_feedback is not None) != (
            rejected_response_text is not None
        ):
            raise RuntimeModelCallAuthorityError(
                "output-repair settlement requires exactly one rejected response body"
            )
        if (
            next_output_repair_feedback is not None
            and (
                model_result is None
                or rejected_response_text != model_result.reply
                or hashlib.sha256(rejected_provider_bytes).hexdigest()
                != next_output_repair_feedback.rejected_response_sha256
            )
        ):
            raise RuntimeModelCallAuthorityError(
                "rejected response body differs from the rejected Provider response"
            )
        settlement_identity = _fingerprint(
            {
                "contract": "runtime-model-settlement-id-v1",
                "physical_request_binding_sha256": physical.binding_sha256,
                "outcome_fingerprint": result_fingerprint,
            }
        )
        settlement = RuntimeModelPhysicalAttemptSettlement.create(
            settlement_id=f"rmps_v1_{settlement_identity}",
            settle_apply_id=f"rmps_apply_v1_{_fingerprint({'physical': physical.binding_sha256})}",
            physical_attempt_id=physical.physical_attempt_id,
            physical_request_binding_sha256=physical.binding_sha256,
            logical_call_id=self.semantic_call_id,
            physical_ordinal=physical.physical_ordinal,
            settled_turn_id=turn_id,
            outcome=outcome_value,
            provider_request_id=provider_request_id,
            finish_reason=(
                None
                if model_result is None or not model_result.finish_reason
                else str(model_result.finish_reason)
            ),
            error_code=normalized_error_code,
            usage=_usage(model_result),
            typed_result=persisted_typed,
            next_output_repair_feedback=next_output_repair_feedback,
            outcome_fingerprint=result_fingerprint,
        )
        if rejected_response_text is None:
            self.store.settle_runtime_model_physical_attempt(
                settlement=settlement,
            )
        else:
            self.store.settle_runtime_model_physical_attempt(
                settlement=settlement,
                rejected_response_text=rejected_response_text,
            )

    def recover_output_repair_feedback(
        self,
    ) -> RuntimeModelOutputRepairFeedback | None:
        """恢复已获准用于下一次重试的精确 prompt 变体。

        typed-output 拒绝必须在不可变结算中携带其下一份反馈。传输失败会保留
        已绑定到失败物理请求的反馈。
        """

        stored = self._stored()
        attempts = stored.physical_attempts
        if not attempts:
            return None
        last = attempts[-1]
        settlement = last.settlement
        if settlement is None:
            raise RuntimeModelCallWaitingExternal(
                "pending Provider dispatch requires external reconciliation"
            )
        if (
            settlement.outcome
            is not RuntimeModelPhysicalOutcome.RETRYABLE_FAILURE
        ):
            return None
        if settlement.error_code == "MODEL_BAD_RESPONSE":
            feedback = settlement.next_output_repair_feedback
            if feedback is None:
                raise RuntimeModelCallAuthorityError(
                    "retryable typed-output rejection has no durable repair feedback"
                )
            return feedback
        return last.request.output_repair_feedback

    def recover_output_repair_response(
        self,
        feedback: RuntimeModelOutputRepairFeedback,
    ) -> str:
        """加载并验证反馈引用的被拒绝正文。"""

        if not isinstance(feedback, RuntimeModelOutputRepairFeedback):
            raise TypeError("feedback must be RuntimeModelOutputRepairFeedback")
        stored = self._stored()
        rejected_attempt = next(
            (
                item
                for item in stored.physical_attempts
                if item.request.physical_ordinal
                == feedback.rejected_physical_ordinal
            ),
            None,
        )
        if rejected_attempt is None:
            raise RuntimeModelCallAuthorityError(
                "repair feedback references an unknown physical attempt"
            )
        getter = getattr(self.store, "get_runtime_model_rejected_output", None)
        if not callable(getter):
            raise RuntimeModelCallAuthorityError(
                "model Store cannot recover rejected output"
            )
        record = getter(
            session_id=self.logical_request.session_id,
            logical_call_id=self.semantic_call_id,
            rejected_physical_ordinal=feedback.rejected_physical_ordinal,
            rejected_response_sha256=feedback.rejected_response_sha256,
        )
        response_text = getattr(record, "response_text", None)
        if not isinstance(response_text, str):
            raise RuntimeModelCallAuthorityError(
                "durable rejected output is missing"
            )
        if hashlib.sha256(response_text.encode("utf-8")).hexdigest() != (
            feedback.rejected_response_sha256
        ):
            raise RuntimeModelCallAuthorityError(
                "durable rejected output hash is invalid"
            )
        return response_text

    def typed_result_payload(
        self,
        *,
        model_result: object,
        value: object,
    ) -> object:
        if not isinstance(model_result, ModelResult):
            raise RuntimeModelCallAuthorityError(
                "typed replay encoder requires a ModelResult"
            )
        payload = self.typed_replay_payload_builder(model_result, value)
        # 此处即证明结算/重放会保留规范 JSON。
        _canonical_json(_jsonable(payload))
        return payload

    def success_fingerprint(self, result: object) -> str:
        if not isinstance(result, ModelResult):
            raise RuntimeModelCallAuthorityError(
                "successful Provider response must be ModelResult"
            )
        return _fingerprint(
            {
                "contract": "runtime-model-provider-result-v1",
                "outcome": "succeeded",
                "provider": result.provider,
                "model": result.model,
                "model_call_id": result.model_call_id,
                "reply": result.reply,
                "finish_reason": result.finish_reason,
            }
        )

    def failure_fingerprint(
        self,
        *,
        error: BaseException,
        provider_result: object | None,
    ) -> str:
        result = provider_result if isinstance(provider_result, ModelResult) else None
        error_payload: dict[str, object] = {"type": type(error).__name__}
        if isinstance(error, ModelGatewayError):
            error_payload.update(code=error.code, retryable=bool(error.retryable))
        return _fingerprint(
            {
                "contract": "runtime-model-provider-result-v1",
                "outcome": "failure",
                "error": error_payload,
                "provider_result": (
                    None
                    if result is None
                    else {
                        "provider": result.provider,
                        "model": result.model,
                        "model_call_id": result.model_call_id,
                        "reply": result.reply,
                        "finish_reason": result.finish_reason,
                    }
                ),
            }
        )

    @staticmethod
    def terminal_state_error(message: str) -> RuntimeModelCallAuthorityError:
        return RuntimeModelCallAuthorityError(message)

    def _stored(self):
        stored = self.store.get_runtime_model_logical_call(
            session_id=self.logical_request.session_id,
            logical_call_id=self.semantic_call_id,
        )
        if stored is None:
            raise RuntimeModelCallAuthorityError(
                "logical model call has not been durably reserved"
            )
        request = getattr(stored, "request", None)
        if request != self.logical_request:
            raise RuntimeModelCallAuthorityError(
                "stored logical model request crossed immutable authority"
            )
        for attempt in getattr(stored, "physical_attempts", ()):
            physical = getattr(attempt, "request", None)
            if (
                not isinstance(physical, RuntimeModelPhysicalAttemptRequest)
                or physical.provider_idempotency_key
                != self.provider_idempotency_key
            ):
                raise RuntimeModelCallAuthorityError(
                    "stored physical model request crossed Provider idempotency authority"
                )
        return stored


def _usage(result: ModelResult | None) -> RuntimeModelUsage:
    if result is None:
        return RuntimeModelUsage()
    return RuntimeModelUsage(
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        cache_read_tokens=result.cache_read_tokens,
        cache_write_tokens=result.cache_write_tokens,
    )


def _normalize_error_code(value: str | None) -> str:
    candidate = str(value or "model_call_failure")
    return candidate if _ERROR_CODE.fullmatch(candidate) else "model_call_failure"


def _jsonable(value: object) -> object:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    return value


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _fingerprint(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


__all__ = [
    'RuntimeLogicalModelCallAuthority',
    "RuntimeModelCallAuthorityError",
    "RuntimeModelCallWaitingExternal",
    'RuntimeModelLedgerStore',
]
