"""provider 中立的持久化模型账本契约。

这些不可变 DTO 将一个逻辑有类型请求与每次物理 provider attempt 及其结算分离。
它们有意持久化规范有类型输入/输出，而非隐藏推理或 provider 原生响应正文。持久化
与重试编排在其他位置实现。
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Literal, Protocol, Self, runtime_checkable

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from ...model_io.output_repair_contracts import (
    RuntimeModelOutputRepairFeedback as _RuntimeModelOutputRepairFeedback,
    RuntimeModelOutputRepairProtocol as _RuntimeModelOutputRepairProtocol,
    RuntimeModelStructuredPrompt as _RuntimeModelStructuredPrompt,
    _validate_runtime_model_output_repair_feedback,
)
from ...model_io.contracts import ModelResult


_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,199}$"
_SHA256_PATTERN = r"^[0-9a-f]{64}$"
_MAX_LOGICAL_REQUEST_BYTES = 1_500_000
_MAX_TYPED_RESULT_BYTES = 1_500_000


DurableModelCallOutcome = Literal[
    "succeeded",
    "retryable_failure",
    "terminal_failure",
    "uncertain",
]


class DurableModelCallRuntimeError(RuntimeError):
    """provider 中立持久化模型调用权威状态的基础失败。"""


class DurableModelCallStateGuardRejected(DurableModelCallRuntimeError):
    """冻结的 input/provider 权威状态在物理调用前后发生变化。"""


class DurableModelCallTerminalState(DurableModelCallRuntimeError):
    """某个逻辑调用已到达持久化终态。"""


@dataclass(frozen=True, slots=True)
class DurableModelCallReplay:
    """从持久化权威状态重建的规范有类型成功结果。"""

    model_result: ModelResult
    physical_ordinal: int

    def __post_init__(self) -> None:
        if self.physical_ordinal < 1:
            raise ValueError("physical_ordinal must be positive")
        if not self.model_result.model_call_id:
            raise ValueError("durable replay requires its physical model_call_id")


@runtime_checkable
class DurablePhysicalModelAttempt(Protocol):
    physical_attempt_id: str
    physical_ordinal: int
    provider: str
    model: str

    @property
    def model_call_id(self) -> str: ...


@runtime_checkable
class DurableLogicalModelCallAuthority(Protocol):
    """共享有界模型包装器可调用的精确方法。"""

    semantic_call_id: str

    def require_current_state(self) -> None: ...

    def reserve(self, *, turn_id: str) -> object: ...

    def replay_succeeded_result(self) -> DurableModelCallReplay | None: ...

    def begin_physical_attempt(
        self,
        *,
        turn_id: str,
        max_physical_attempts: int,
        output_repair_enabled: bool = False,
        output_repair_feedback: object | None = None,
    ) -> DurablePhysicalModelAttempt: ...

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
        next_output_repair_feedback: object | None = None,
        rejected_response_text: str | None = None,
    ) -> None: ...

    def typed_result_payload(
        self,
        *,
        model_result: object,
        value: object,
    ) -> object: ...

    def success_fingerprint(self, result: object) -> str: ...

    def failure_fingerprint(
        self,
        *,
        error: BaseException,
        provider_result: object | None,
    ) -> str: ...

    def terminal_state_error(self, message: str) -> DurableModelCallTerminalState: ...


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _require_canonical_json(
    value: str,
    *,
    label: str,
    max_utf8_bytes: int,
) -> object:
    if len(value.encode("utf-8")) > max_utf8_bytes:
        raise ValueError(f"{label} exceeds its UTF-8 byte limit")
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError, RecursionError) as exc:
        raise ValueError(f"{label} must be valid canonical JSON") from exc
    if _canonical_json(parsed) != value:
        raise ValueError(f"{label} must use canonical JSON encoding")
    return parsed


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class RuntimeModelPhysicalOutcome(StrEnum):
    SUCCEEDED = "succeeded"
    RETRYABLE_FAILURE = "retryable_failure"
    TERMINAL_FAILURE = "terminal_failure"
    UNCERTAIN = "uncertain"


class RuntimeModelUsage(_Contract):
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    cache_read_tokens: int | None = Field(default=None, ge=0)
    cache_write_tokens: int | None = Field(default=None, ge=0)
    cost_microusd: int | None = Field(default=None, ge=0)


def runtime_model_dispatch_request_sha256(
    *,
    logical_request_sha256: str,
    output_repair_feedback: _RuntimeModelOutputRepairFeedback | None,
) -> str:
    """对正在分派的精确不可变逻辑请求变体执行哈希。"""

    if not isinstance(logical_request_sha256, str) or not re.fullmatch(
        _SHA256_PATTERN,
        logical_request_sha256,
    ):
        raise ValueError("logical_request_sha256 must be a canonical SHA-256")
    return _sha256_text(
        _canonical_json(
            {
                "contract": "runtime-model-dispatch-request-v1",
                "logical_request_sha256": logical_request_sha256,
                "output_repair_feedback": (
                    None
                    if output_repair_feedback is None
                    else output_repair_feedback.model_dump(mode="json")
                ),
            }
        )
    )


class RuntimeModelLogicalRequest(_Contract):
    """一个已预留语义请求及其完整有类型输入。"""

    schema_version: Literal["runtime-model-logical-request-v1"] = (
        "runtime-model-logical-request-v1"
    )
    logical_call_id: str = Field(pattern=_ID_PATTERN)
    session_id: str = Field(pattern=_ID_PATTERN)
    task_id: str | None = Field(default=None, pattern=_ID_PATTERN)
    auxiliary_graph_id: str | None = Field(default=None, pattern=_ID_PATTERN)
    goal_id: str | None = Field(default=None, pattern=_ID_PATTERN)
    execution_subject_id: str | None = Field(default=None, pattern=_ID_PATTERN)
    invocation_turn_id: str = Field(pattern=_ID_PATTERN)
    call_kind: str = Field(pattern=_ID_PATTERN)
    purpose: str = Field(pattern=_ID_PATTERN)
    provider: str = Field(pattern=_ID_PATTERN)
    model: str = Field(min_length=1, max_length=300)
    endpoint_fingerprint: str = Field(pattern=_SHA256_PATTERN)
    request_contract: str = Field(pattern=_ID_PATTERN)
    request_json: str
    request_sha256: str = Field(pattern=_SHA256_PATTERN)
    output_repair_protocol: _RuntimeModelOutputRepairProtocol | None = Field()
    structured_prompt: _RuntimeModelStructuredPrompt | None = Field()
    typed_result_contract: str = Field(pattern=_ID_PATTERN)
    max_physical_attempts: int = Field(ge=1, le=32)
    state_guard_sha256: str = Field(pattern=_SHA256_PATTERN)
    binding_sha256: str = Field(pattern=_SHA256_PATTERN)

    @field_validator("model")
    @classmethod
    def _validate_model(cls, value: str) -> str:
        if value != value.strip() or "\x00" in value:
            raise ValueError("model must be canonical non-NUL text")
        return value

    @model_validator(mode="after")
    def _validate_request(self) -> 'RuntimeModelLogicalRequest':
        if (self.auxiliary_graph_id is None) != (self.goal_id is None):
            raise ValueError(
                "auxiliary_graph_id and goal_id must be supplied together"
            )
        if self.goal_id is not None and self.task_id is None:
            raise ValueError("a planning-goal model call requires its Task owner")
        _require_canonical_json(
            self.request_json,
            label="logical model request",
            max_utf8_bytes=_MAX_LOGICAL_REQUEST_BYTES,
        )
        if self.request_sha256 != _sha256_text(self.request_json):
            raise ValueError("logical model request hash does not match its JSON")
        self.output_repair_target_contract  # 同时验证可选的外部提案合同，不改变序列化 wire。
        if (
            self.output_repair_protocol
            is _RuntimeModelOutputRepairProtocol.FOUR_MESSAGE_WHOLE_RESPONSE_REGENERATION
            and self.structured_prompt is None
        ):
            raise ValueError(
                "four-message output repair requires an exact structured prompt"
            )
        expected = _logical_request_binding_sha256(self)
        if self.binding_sha256 != expected:
            raise ValueError("logical model request binding hash does not match")
        return self

    @property
    def output_repair_target_contract(self) -> str:
        """修复面向模型提案；成功持久化可使用另一种 Host 内部结果合同。

        声明在已有 request_json 中冻结、参与原有 hash，不新增数据库字段。
        没有该声明的历史/其他调用保持 typed_result_contract 的原行为。
        """
        payload = json.loads(self.request_json)
        value = (payload.get("model_output_contract", self.typed_result_contract)
                 if isinstance(payload, dict) else self.typed_result_contract)
        if not isinstance(value, str) or re.fullmatch(_ID_PATTERN, value) is None:
            raise ValueError("model output contract must be a canonical identifier")
        return value

    @classmethod
    def create(cls, *, request_payload: object, **values: object) -> Self:
        request_json = _canonical_json(request_payload)
        values = dict(values)
        values.setdefault("output_repair_protocol", None)
        values.setdefault("structured_prompt", None)
        protocol = values["output_repair_protocol"]
        if protocol is not None and not isinstance(
            protocol,
            _RuntimeModelOutputRepairProtocol,
        ):
            values["output_repair_protocol"] = (
                _RuntimeModelOutputRepairProtocol(protocol)
            )
        prompt = values["structured_prompt"]
        if prompt is not None and not isinstance(
            prompt,
            _RuntimeModelStructuredPrompt,
        ):
            values["structured_prompt"] = (
                _RuntimeModelStructuredPrompt.model_validate(prompt)
            )
        values.update(
            request_json=request_json,
            request_sha256=_sha256_text(request_json),
            binding_sha256="0" * 64,
        )
        provisional = cls.model_construct(**values)
        values["binding_sha256"] = _logical_request_binding_sha256(provisional)
        return cls.model_validate(values)


def _logical_request_binding_sha256(request: RuntimeModelLogicalRequest) -> str:
    payload = request.model_dump(mode="json", exclude={"binding_sha256"})
    return _sha256_text(_canonical_json(payload))


class RuntimeModelPhysicalAttemptRequest(_Contract):
    """一个逻辑请求下的一次只追加物理分派。"""

    schema_version: Literal["runtime-model-physical-attempt-request-v1"] = (
        "runtime-model-physical-attempt-request-v1"
    )
    physical_attempt_id: str = Field(pattern=_ID_PATTERN)
    physical_attempt_key: str = Field(pattern=_ID_PATTERN)
    logical_call_id: str = Field(pattern=_ID_PATTERN)
    logical_request_binding_sha256: str = Field(pattern=_SHA256_PATTERN)
    physical_ordinal: int = Field(ge=1, le=32)
    started_turn_id: str = Field(pattern=_ID_PATTERN)
    provider: str = Field(pattern=_ID_PATTERN)
    model: str = Field(min_length=1, max_length=300)
    endpoint_fingerprint: str = Field(pattern=_SHA256_PATTERN)
    request_sha256: str = Field(pattern=_SHA256_PATTERN)
    dispatch_request_sha256: str = Field(pattern=_SHA256_PATTERN)
    output_repair_enabled: bool
    output_repair_feedback: _RuntimeModelOutputRepairFeedback | None = Field()
    provider_idempotency_key: str | None = Field(default=None, min_length=1, max_length=300)
    dispatch_authority_sha256: str = Field(pattern=_SHA256_PATTERN)
    binding_sha256: str = Field(pattern=_SHA256_PATTERN)

    @model_validator(mode="after")
    def _validate_attempt(self) -> 'RuntimeModelPhysicalAttemptRequest':
        if self.model != self.model.strip() or "\x00" in self.model:
            raise ValueError("physical model must be canonical non-NUL text")
        expected_dispatch = runtime_model_dispatch_request_sha256(
            logical_request_sha256=self.request_sha256,
            output_repair_feedback=self.output_repair_feedback,
        )
        if self.dispatch_request_sha256 != expected_dispatch:
            raise ValueError("physical dispatch request hash does not match")
        if self.output_repair_feedback is not None and not self.output_repair_enabled:
            raise ValueError(
                "output-repair feedback requires output_repair_enabled"
            )
        expected = _sha256_text(
            _canonical_json(self.model_dump(mode="json", exclude={"binding_sha256"}))
        )
        if self.binding_sha256 != expected:
            raise ValueError("physical model request binding hash does not match")
        return self

    @classmethod
    def create(cls, **values: object) -> Self:
        values = dict(values)
        values["output_repair_enabled"] = bool(
            values.get("output_repair_enabled", False)
        )
        values.setdefault("output_repair_feedback", None)
        feedback = values.get("output_repair_feedback")
        if feedback is not None and not isinstance(
            feedback,
            _RuntimeModelOutputRepairFeedback,
        ):
            feedback = _validate_runtime_model_output_repair_feedback(feedback)
            values["output_repair_feedback"] = feedback
        if feedback is not None and not values["output_repair_enabled"]:
            raise ValueError(
                "output_repair_feedback requires output_repair_enabled"
            )
        values["dispatch_request_sha256"] = runtime_model_dispatch_request_sha256(
            logical_request_sha256=str(values["request_sha256"]),
            output_repair_feedback=feedback,
        )
        values["binding_sha256"] = "0" * 64
        provisional = cls.model_construct(**values)
        values["binding_sha256"] = _sha256_text(
            _canonical_json(
                provisional.model_dump(mode="json", exclude={"binding_sha256"})
            )
        )
        return cls.model_validate(values)

    @property
    def model_call_id(self) -> str:
        return self.physical_attempt_key


class RuntimeModelTypedResult(_Contract):
    schema_version: Literal["runtime-model-typed-result-v1"] = (
        "runtime-model-typed-result-v1"
    )
    result_contract: str = Field(pattern=_ID_PATTERN)
    result_json: str
    result_sha256: str = Field(pattern=_SHA256_PATTERN)

    @model_validator(mode="after")
    def _validate_result(self) -> 'RuntimeModelTypedResult':
        _require_canonical_json(
            self.result_json,
            label="typed model result",
            max_utf8_bytes=_MAX_TYPED_RESULT_BYTES,
        )
        if self.result_sha256 != _sha256_text(self.result_json):
            raise ValueError("typed model result hash does not match its JSON")
        return self

    @classmethod
    def create(cls, *, result_contract: str, result_payload: object) -> Self:
        result_json = _canonical_json(result_payload)
        return cls(
            result_contract=result_contract,
            result_json=result_json,
            result_sha256=_sha256_text(result_json),
        )

    def parsed(self) -> Any:
        return json.loads(self.result_json)


class RuntimeModelPhysicalAttemptSettlement(_Contract):
    """不可变结算；``uncertain`` 绝不表示自动重试授权。"""

    schema_version: Literal["runtime-model-physical-attempt-settlement-v1"] = (
        "runtime-model-physical-attempt-settlement-v1"
    )
    settlement_id: str = Field(pattern=_ID_PATTERN)
    settle_apply_id: str = Field(pattern=_ID_PATTERN)
    physical_attempt_id: str = Field(pattern=_ID_PATTERN)
    physical_request_binding_sha256: str = Field(pattern=_SHA256_PATTERN)
    logical_call_id: str = Field(pattern=_ID_PATTERN)
    physical_ordinal: int = Field(ge=1, le=32)
    settled_turn_id: str = Field(pattern=_ID_PATTERN)
    outcome: RuntimeModelPhysicalOutcome
    provider_request_id: str | None = Field(default=None, min_length=1, max_length=300)
    finish_reason: str | None = Field(default=None, min_length=1, max_length=200)
    error_code: str | None = Field(default=None, pattern=_ID_PATTERN)
    usage: RuntimeModelUsage = RuntimeModelUsage()
    typed_result: RuntimeModelTypedResult | None = None
    next_output_repair_feedback: _RuntimeModelOutputRepairFeedback | None = Field()
    outcome_fingerprint: str = Field(pattern=_SHA256_PATTERN)
    receipt_sha256: str = Field(pattern=_SHA256_PATTERN)

    @model_validator(mode="after")
    def _validate_settlement(self) -> 'RuntimeModelPhysicalAttemptSettlement':
        succeeded = self.outcome is RuntimeModelPhysicalOutcome.SUCCEEDED
        if succeeded != (self.typed_result is not None):
            raise ValueError("only a succeeded physical attempt requires a typed result")
        if succeeded and self.error_code is not None:
            raise ValueError("a succeeded physical attempt cannot carry an error code")
        if not succeeded and self.error_code is None:
            raise ValueError("a non-success settlement requires a typed error code")
        if self.next_output_repair_feedback is not None:
            if (
                self.outcome is not RuntimeModelPhysicalOutcome.RETRYABLE_FAILURE
                or self.error_code != "MODEL_BAD_RESPONSE"
                or self.next_output_repair_feedback.rejected_physical_ordinal
                != self.physical_ordinal
            ):
                raise ValueError(
                    "output-repair feedback requires the matching retryable "
                    "MODEL_BAD_RESPONSE settlement"
                )
        expected = _sha256_text(
            _canonical_json(self.model_dump(mode="json", exclude={"receipt_sha256"}))
        )
        if self.receipt_sha256 != expected:
            raise ValueError("physical settlement receipt hash does not match")
        return self

    @classmethod
    def create(cls, **values: object) -> Self:
        values = dict(values)
        values.setdefault("next_output_repair_feedback", None)
        usage = values.get("usage")
        if usage is not None and not isinstance(usage, RuntimeModelUsage):
            values["usage"] = RuntimeModelUsage.model_validate(usage)
        typed_result = values.get("typed_result")
        if typed_result is not None and not isinstance(
            typed_result, RuntimeModelTypedResult
        ):
            values["typed_result"] = RuntimeModelTypedResult.model_validate(
                typed_result
            )
        feedback = values.get("next_output_repair_feedback")
        if feedback is not None and not isinstance(
            feedback,
            _RuntimeModelOutputRepairFeedback,
        ):
            values["next_output_repair_feedback"] = (
                _validate_runtime_model_output_repair_feedback(feedback)
            )
        values["receipt_sha256"] = "0" * 64
        provisional = cls.model_construct(**values)
        values["receipt_sha256"] = _sha256_text(
            _canonical_json(
                provisional.model_dump(mode="json", exclude={"receipt_sha256"})
            )
        )
        return cls.model_validate(values)


class RuntimeModelLedgerStore(Protocol):
    """一个 provider 中立模型调用账本的狭窄持久化 port。"""

    def reserve_runtime_model_logical_call(
        self, *, request: RuntimeModelLogicalRequest
    ) -> object: ...

    def append_runtime_model_physical_attempt(
        self, *, request: RuntimeModelPhysicalAttemptRequest
    ) -> object: ...

    def settle_runtime_model_physical_attempt(
        self,
        *,
        settlement: RuntimeModelPhysicalAttemptSettlement,
        rejected_response_text: str | None = None,
    ) -> object: ...

    def get_runtime_model_logical_call(
        self, *, session_id: str, logical_call_id: str
    ) -> object | None: ...

    def get_runtime_model_rejected_output(
        self,
        *,
        session_id: str,
        logical_call_id: str,
        rejected_physical_ordinal: int,
        rejected_response_sha256: str,
    ) -> object | None: ...


__all__ = [
    "DurableLogicalModelCallAuthority",
    "DurableModelCallOutcome",
    "DurableModelCallReplay",
    "DurableModelCallRuntimeError",
    "DurableModelCallStateGuardRejected",
    "DurableModelCallTerminalState",
    "DurablePhysicalModelAttempt",
    "RuntimeModelLedgerStore",
    "RuntimeModelLogicalRequest",
    "RuntimeModelPhysicalAttemptRequest",
    "RuntimeModelPhysicalAttemptSettlement",
    "RuntimeModelPhysicalOutcome",
    "RuntimeModelTypedResult",
    "RuntimeModelUsage",
    "runtime_model_dispatch_request_sha256",
]
