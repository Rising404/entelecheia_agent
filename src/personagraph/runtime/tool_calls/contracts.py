"""Runtime 工具调用的崩溃安全逻辑/物理账本契约。

一个具体化的 WorkRun ``tool_call_id`` 即逻辑调用。每次实际分派都是独立且不可变的物理尝试。
这些 DTO 不授予重试权威：它们保留精确的效果/重试策略，并显式表示未知响应，使所属归约器可以
重试读取、复用提供商幂等键，或等待协调。
"""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from typing import Any, Literal, Protocol, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

# Tool 实现版本是 Catalog 标识，而不只是 Runtime ID。
# Vision 注册项刻意包含诸如 ``1+http-vision@1`` 的处理器指纹；持久账本必须保留该精确值。
_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._:/+@-]{0,199}$"
_SHA256_PATTERN = r"^[0-9a-f]{64}$"
_MAX_ARGUMENT_BYTES = 1_000_000
_MAX_RESULT_BYTES = 1_500_000
_MISSING = object()


class RuntimeToolCallAuthorityError(RuntimeError):
    """逻辑 ToolCall 若继续推进就会违反其持久权威。"""


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


def _require_canonical_json(value: str, *, label: str, limit: int) -> object:
    if len(value.encode("utf-8")) > limit:
        raise ValueError(f"{label} exceeds its UTF-8 byte limit")
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError, RecursionError) as exc:
        raise ValueError(f"{label} must be valid canonical JSON") from exc
    if _canonical_json(parsed) != value:
        raise ValueError(f"{label} must use canonical JSON encoding")
    return parsed


def _jsonable(value: object) -> object:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    return value


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class RuntimeToolEffectClass(StrEnum):
    READ_ONLY = "read_only"
    PROTECTED_EFFECT = "protected_effect"


class RuntimeToolRetryAuthority(StrEnum):
    READ_ONLY_REPLAY = "read_only_replay"
    PROVIDER_IDEMPOTENCY = "provider_idempotency"
    RECONCILIATION_REQUIRED = "reconciliation_required"


class RuntimeToolPhysicalOutcome(StrEnum):
    SUCCEEDED = "succeeded"
    RETRYABLE_FAILURE = "retryable_failure"
    TERMINAL_FAILURE = "terminal_failure"
    UNCERTAIN = "uncertain"


class RuntimeToolLogicalRequest(_Contract):
    """已决定 WorkRun Attempt 下获准的精确逻辑 ToolCall。"""

    schema_version: Literal["runtime-tool-logical-request-v1"] = (
        "runtime-tool-logical-request-v1"
    )
    logical_tool_call_id: str = Field(pattern=_ID_PATTERN)
    session_id: str = Field(pattern=_ID_PATTERN)
    work_run_id: str = Field(pattern=_ID_PATTERN)
    attempt_id: str = Field(pattern=_ID_PATTERN)
    call_ordinal: int = Field(ge=1, le=4)
    invocation_turn_id: str = Field(pattern=_ID_PATTERN)
    catalog_snapshot_sha256: str = Field(pattern=_SHA256_PATTERN)
    tool_id: str = Field(pattern=_ID_PATTERN)
    contract_version: str = Field(pattern=_ID_PATTERN)
    implementation_version: str = Field(pattern=_ID_PATTERN)
    provider_identity_sha256: str = Field(pattern=_SHA256_PATTERN)
    effect_profile_sha256: str = Field(pattern=_SHA256_PATTERN)
    effect_class: RuntimeToolEffectClass
    retry_authority: RuntimeToolRetryAuthority
    arguments_json: str
    arguments_sha256: str = Field(pattern=_SHA256_PATTERN)
    result_contract: str = Field(pattern=_ID_PATTERN)
    max_physical_attempts: int = Field(ge=1, le=16)
    state_guard_sha256: str = Field(pattern=_SHA256_PATTERN)
    binding_sha256: str = Field(pattern=_SHA256_PATTERN)

    @model_validator(mode="after")
    def _validate_request(self) -> 'RuntimeToolLogicalRequest':
        _require_canonical_json(
            self.arguments_json,
            label="logical tool arguments",
            limit=_MAX_ARGUMENT_BYTES,
        )
        if self.arguments_sha256 != _sha256_text(self.arguments_json):
            raise ValueError("logical tool arguments hash does not match")
        if (
            self.effect_class is RuntimeToolEffectClass.READ_ONLY
        ) != (
            self.retry_authority
            is RuntimeToolRetryAuthority.READ_ONLY_REPLAY
        ):
            raise ValueError(
                "read_only_replay authority belongs exactly to read-only calls"
            )
        expected = _sha256_text(
            _canonical_json(self.model_dump(mode="json", exclude={"binding_sha256"}))
        )
        if self.binding_sha256 != expected:
            raise ValueError("logical tool request binding hash does not match")
        return self

    @classmethod
    def create(cls, *, arguments: object, **values: object) -> Self:
        values = dict(values)
        arguments_json = _canonical_json(arguments)
        values.update(
            arguments_json=arguments_json,
            arguments_sha256=_sha256_text(arguments_json),
            binding_sha256="0" * 64,
        )
        provisional = cls.model_construct(**values)
        values["binding_sha256"] = _sha256_text(
            _canonical_json(
                provisional.model_dump(mode="json", exclude={"binding_sha256"})
            )
        )
        return cls.model_validate(values)


class RuntimeToolPhysicalAttemptRequest(_Contract):
    """逻辑 ToolCall 的一次实际分派。"""

    schema_version: Literal["runtime-tool-physical-attempt-request-v1"] = (
        "runtime-tool-physical-attempt-request-v1"
    )
    physical_attempt_id: str = Field(pattern=_ID_PATTERN)
    physical_attempt_key: str = Field(pattern=_ID_PATTERN)
    logical_tool_call_id: str = Field(pattern=_ID_PATTERN)
    logical_request_binding_sha256: str = Field(pattern=_SHA256_PATTERN)
    physical_ordinal: int = Field(ge=1, le=16)
    started_turn_id: str = Field(pattern=_ID_PATTERN)
    retry_authority: RuntimeToolRetryAuthority
    provider_idempotency_key: str | None = Field(
        default=None,
        min_length=1,
        max_length=300,
    )
    dispatch_authority_sha256: str = Field(pattern=_SHA256_PATTERN)
    binding_sha256: str = Field(pattern=_SHA256_PATTERN)

    @model_validator(mode="after")
    def _validate_attempt(self) -> 'RuntimeToolPhysicalAttemptRequest':
        requires_key = (
            self.retry_authority
            is RuntimeToolRetryAuthority.PROVIDER_IDEMPOTENCY
        )
        if requires_key != (self.provider_idempotency_key is not None):
            raise ValueError(
                "provider idempotency retry authority requires exactly one key"
            )
        expected = _sha256_text(
            _canonical_json(self.model_dump(mode="json", exclude={"binding_sha256"}))
        )
        if self.binding_sha256 != expected:
            raise ValueError("physical tool request binding hash does not match")
        return self

    @classmethod
    def create(cls, **values: object) -> Self:
        values = dict(values)
        values["binding_sha256"] = "0" * 64
        provisional = cls.model_construct(**values)
        values["binding_sha256"] = _sha256_text(
            _canonical_json(
                provisional.model_dump(mode="json", exclude={"binding_sha256"})
            )
        )
        return cls.model_validate(values)


class RuntimeToolTypedResult(_Contract):
    schema_version: Literal["runtime-tool-typed-result-v1"] = (
        "runtime-tool-typed-result-v1"
    )
    result_contract: str = Field(pattern=_ID_PATTERN)
    result_json: str
    result_sha256: str = Field(pattern=_SHA256_PATTERN)

    @model_validator(mode="after")
    def _validate_result(self) -> 'RuntimeToolTypedResult':
        _require_canonical_json(
            self.result_json,
            label="typed tool result",
            limit=_MAX_RESULT_BYTES,
        )
        if self.result_sha256 != _sha256_text(self.result_json):
            raise ValueError("typed tool result hash does not match")
        return self

    @classmethod
    def create(cls, *, result_contract: str, result: object) -> Self:
        result_json = _canonical_json(result)
        return cls(
            result_contract=result_contract,
            result_json=result_json,
            result_sha256=_sha256_text(result_json),
        )

    def parsed(self) -> Any:
        return json.loads(self.result_json)


class RuntimeToolDispatchObservation(_Contract):
    """一次提供商/本地工具分派的 Host 规范观察。

    提供商 SDK 载荷与隐藏传输细节不会进入账本。适配器会在输出验证后发出此有界值。其自身哈希
    也是物理结算的结果指纹。
    """

    schema_version: Literal["runtime-tool-dispatch-observation-v1"] = (
        "runtime-tool-dispatch-observation-v1"
    )
    session_id: str = Field(pattern=_ID_PATTERN)
    work_run_id: str = Field(pattern=_ID_PATTERN)
    attempt_id: str = Field(pattern=_ID_PATTERN)
    logical_tool_call_id: str = Field(pattern=_ID_PATTERN)
    logical_request_binding_sha256: str = Field(pattern=_SHA256_PATTERN)
    physical_attempt_id: str = Field(pattern=_ID_PATTERN)
    physical_request_binding_sha256: str = Field(pattern=_SHA256_PATTERN)
    physical_ordinal: int = Field(ge=1, le=16)
    provider_identity_sha256: str = Field(pattern=_SHA256_PATTERN)
    tool_id: str = Field(pattern=_ID_PATTERN)
    contract_version: str = Field(pattern=_ID_PATTERN)
    implementation_version: str = Field(pattern=_ID_PATTERN)
    outcome: RuntimeToolPhysicalOutcome
    provider_request_id: str | None = Field(default=None, min_length=1, max_length=300)
    error_code: str | None = Field(default=None, pattern=_ID_PATTERN)
    duration_ms: int | None = Field(default=None, ge=0)
    result_json: str | None = None
    result_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    binding_sha256: str = Field(pattern=_SHA256_PATTERN)

    @model_validator(mode="after")
    def _validate_observation(self) -> 'RuntimeToolDispatchObservation':
        succeeded = self.outcome is RuntimeToolPhysicalOutcome.SUCCEEDED
        has_result = self.result_json is not None and self.result_sha256 is not None
        if succeeded != has_result:
            raise ValueError("only a succeeded dispatch observation has a result")
        if (self.result_json is None) != (self.result_sha256 is None):
            raise ValueError("dispatch result JSON and hash must appear together")
        if succeeded != (self.error_code is None):
            raise ValueError("non-success dispatch observations require an error code")
        if self.result_json is not None:
            _require_canonical_json(
                self.result_json,
                label="dispatch observation result",
                limit=_MAX_RESULT_BYTES,
            )
            if self.result_sha256 != _sha256_text(self.result_json):
                raise ValueError("dispatch observation result hash does not match")
        expected = _sha256_text(
            _canonical_json(self.model_dump(mode="json", exclude={"binding_sha256"}))
        )
        if self.binding_sha256 != expected:
            raise ValueError("dispatch observation binding hash does not match")
        return self

    @classmethod
    def create(cls, *, result: object = _MISSING, **values: object) -> Self:
        values = dict(values)
        if result is not _MISSING:
            result_json = _canonical_json(_jsonable(result))
            values.update(
                result_json=result_json,
                result_sha256=_sha256_text(result_json),
            )
        values["binding_sha256"] = "0" * 64
        provisional = cls.model_construct(**values)
        values["binding_sha256"] = _sha256_text(
            _canonical_json(
                provisional.model_dump(mode="json", exclude={"binding_sha256"})
            )
        )
        return cls.model_validate(values)

    def parsed_result(self) -> Any:
        if self.result_json is None:
            raise RuntimeToolCallAuthorityError(
                "failed dispatch observation has no typed result"
            )
        return json.loads(self.result_json)


class RuntimeToolPhysicalAttemptSettlement(_Contract):
    """不可变的物理收据；不确定性绝不授予再次分派。"""

    schema_version: Literal["runtime-tool-physical-attempt-settlement-v1"] = (
        "runtime-tool-physical-attempt-settlement-v1"
    )
    settlement_id: str = Field(pattern=_ID_PATTERN)
    settle_apply_id: str = Field(pattern=_ID_PATTERN)
    physical_attempt_id: str = Field(pattern=_ID_PATTERN)
    physical_request_binding_sha256: str = Field(pattern=_SHA256_PATTERN)
    logical_tool_call_id: str = Field(pattern=_ID_PATTERN)
    physical_ordinal: int = Field(ge=1, le=16)
    settled_turn_id: str = Field(pattern=_ID_PATTERN)
    outcome: RuntimeToolPhysicalOutcome
    provider_request_id: str | None = Field(default=None, min_length=1, max_length=300)
    error_code: str | None = Field(default=None, pattern=_ID_PATTERN)
    duration_ms: int | None = Field(default=None, ge=0)
    typed_result: RuntimeToolTypedResult | None = None
    outcome_fingerprint: str = Field(pattern=_SHA256_PATTERN)
    receipt_sha256: str = Field(pattern=_SHA256_PATTERN)

    @model_validator(mode="after")
    def _validate_settlement(self) -> 'RuntimeToolPhysicalAttemptSettlement':
        succeeded = self.outcome is RuntimeToolPhysicalOutcome.SUCCEEDED
        if succeeded != (self.typed_result is not None):
            raise ValueError("only a succeeded physical tool attempt has a typed result")
        if succeeded != (self.error_code is None):
            raise ValueError("non-success physical tool outcomes require an error code")
        expected = _sha256_text(
            _canonical_json(self.model_dump(mode="json", exclude={"receipt_sha256"}))
        )
        if self.receipt_sha256 != expected:
            raise ValueError("physical tool settlement receipt hash does not match")
        return self

    @classmethod
    def create(cls, **values: object) -> Self:
        values = dict(values)
        typed_result = values.get("typed_result")
        if typed_result is not None and not isinstance(
            typed_result, RuntimeToolTypedResult
        ):
            values["typed_result"] = RuntimeToolTypedResult.model_validate(
                typed_result
            )
        values["receipt_sha256"] = "0" * 64
        provisional = cls.model_construct(**values)
        values["receipt_sha256"] = _sha256_text(
            _canonical_json(
                provisional.model_dump(mode="json", exclude={"receipt_sha256"})
            )
        )
        return cls.model_validate(values)


class RuntimeToolLedgerStore(Protocol):
    """工具调用账本的狭窄持久化 port。"""

    def reserve_runtime_tool_logical_call(
        self, *, request: RuntimeToolLogicalRequest
    ) -> object: ...

    def append_runtime_tool_physical_attempt(
        self, *, request: RuntimeToolPhysicalAttemptRequest
    ) -> object: ...

    def settle_runtime_tool_physical_attempt(
        self, *, settlement: RuntimeToolPhysicalAttemptSettlement
    ) -> object: ...

    def get_runtime_tool_logical_call(
        self, *, session_id: str, logical_tool_call_id: str
    ) -> object | None: ...


__all__ = [
    "RuntimeToolCallAuthorityError",
    'RuntimeToolEffectClass',
    'RuntimeToolDispatchObservation',
    'RuntimeToolLedgerStore',
    'RuntimeToolLogicalRequest',
    'RuntimeToolPhysicalAttemptRequest',
    'RuntimeToolPhysicalAttemptSettlement',
    'RuntimeToolPhysicalOutcome',
    'RuntimeToolRetryAuthority',
    'RuntimeToolTypedResult',
]
