"""不含内容的模型上下文预算契约。

此包特意将来源选择与容量准入分离。调用方可以使用所需的任意检索或投影策略，
但跨越此边界的值只包含大小、哈希、修订和路由事实。原始提示词或文档内容绝不能
持久化到这些契约中。
"""

from __future__ import annotations

from enum import StrEnum
import math
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _FrozenContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class TokenMeterIdentity(Protocol):
    name: str
    kind: str
    requested: str
    model: str | None
    fallback_reason: str | None


class TextTokenCounter(TokenMeterIdentity, Protocol):
    """由现有词元计数器实现的小型结构接缝。"""

    def count_text(self, text: str) -> int: ...


class ContextBlockStability(StrEnum):
    IMMUTABLE = "immutable"
    EPOCH = "epoch"
    VOLATILE = "volatile"


class TokenEstimateKind(StrEnum):
    HEURISTIC = "heuristic"
    TOKENIZER = "tokenizer"
    PROVIDER_USAGE_ANCHORED = "provider_usage_anchored"
    PROVIDER_EXACT = "provider_exact"


class ContextBudgetPressure(StrEnum):
    NORMAL = "normal"
    SOFT_LIMIT_EXCEEDED = "soft_limit_exceeded"
    HARD_LIMIT_EXCEEDED = "hard_limit_exceeded"


class ContextBudgetViolation(StrEnum):
    INPUT_TOKENS = "input_tokens"
    REQUEST_BODY_BYTES = "request_body_bytes"


class ContextBlockAdmissionReason(StrEnum):
    ADMITTED = "admitted"
    INSUFFICIENT_BUDGET = "insufficient_budget"


class TokenCounterFingerprint(_FrozenContract):
    """用于可重放估算的计数器标识。"""

    schema_version: int = Field(default=1, ge=1, le=1)
    name: str = Field(min_length=1, max_length=200)
    kind: str = Field(min_length=1, max_length=100)
    requested: str = Field(min_length=1, max_length=200)
    model: str | None = Field(default=None, max_length=500)
    fallback_reason: str | None = Field(default=None, max_length=500)


class ContextBudgetBlock(_FrozenContract):
    """一个经过测量且不含模型可见内容的投影块。"""

    schema_version: int = Field(default=1, ge=1, le=1)
    block_id: str = Field(
        min_length=1,
        max_length=200,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:/_-]{0,199}$",
    )
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_revision: str | None = Field(default=None, max_length=300)
    stability: ContextBlockStability
    mandatory: bool
    priority: int = Field(ge=0, le=1_000)
    estimated_tokens: int = Field(ge=0)
    serialized_utf8_bytes: int = Field(ge=0)
    token_counter: TokenCounterFingerprint


class ContextBlockAdmission(_FrozenContract):
    """一个确定性且不含内容的第一级装填决策。"""

    schema_version: int = Field(default=1, ge=1, le=1)
    block: ContextBudgetBlock
    accepted: bool
    reason: ContextBlockAdmissionReason
    used_tokens_before: int = Field(ge=0)
    used_tokens_after: int = Field(ge=0)
    remaining_tokens: int = Field(ge=0)
    input_budget_tokens: int = Field(gt=0)

    @model_validator(mode="after")
    def _require_consistent_admission(self) -> 'ContextBlockAdmission':
        expected_after = (
            self.used_tokens_before + self.block.estimated_tokens
            if self.accepted
            else self.used_tokens_before
        )
        if self.used_tokens_after != expected_after:
            raise ValueError("context-block admission usage is inconsistent")
        if self.used_tokens_after > self.input_budget_tokens:
            raise ValueError("context-block admission exceeds its input budget")
        fits = (
            self.used_tokens_before + self.block.estimated_tokens
            <= self.input_budget_tokens
        )
        if self.accepted != fits:
            raise ValueError("context-block admission fit decision is inconsistent")
        if self.remaining_tokens != (
            self.input_budget_tokens - self.used_tokens_after
        ):
            raise ValueError("context-block admission remainder is inconsistent")
        expected_reason = (
            ContextBlockAdmissionReason.ADMITTED
            if self.accepted
            else ContextBlockAdmissionReason.INSUFFICIENT_BUDGET
        )
        if self.reason is not expected_reason:
            raise ValueError("context-block admission reason is inconsistent")
        return self


class ContextBlockBudgetSnapshot(_FrozenContract):
    """经过零个或多个决策后的当前第一级装填状态。"""

    schema_version: int = Field(default=1, ge=1, le=1)
    input_budget_tokens: int = Field(gt=0)
    fixed_overhead_tokens: int = Field(ge=0)
    admitted_blocks: tuple[ContextBudgetBlock, ...]
    used_tokens: int = Field(ge=0)
    remaining_tokens: int = Field(ge=0)

    @model_validator(mode="after")
    def _require_consistent_snapshot(self) -> 'ContextBlockBudgetSnapshot':
        block_ids = tuple(block.block_id for block in self.admitted_blocks)
        if len(set(block_ids)) != len(block_ids):
            raise ValueError("context-block budget contains duplicate block ids")
        counters = {block.token_counter for block in self.admitted_blocks}
        if len(counters) > 1:
            raise ValueError("context-block budget mixes token counters")
        expected_used = self.fixed_overhead_tokens + sum(
            block.estimated_tokens for block in self.admitted_blocks
        )
        if self.used_tokens != expected_used:
            raise ValueError("context-block budget usage is inconsistent")
        if self.used_tokens > self.input_budget_tokens:
            raise ValueError("context-block budget exceeds its input limit")
        if self.remaining_tokens != self.input_budget_tokens - self.used_tokens:
            raise ValueError("context-block budget remainder is inconsistent")
        return self


class TokenCountCacheInfo(_FrozenContract):
    schema_version: int = Field(default=1, ge=1, le=1)
    hits: int = Field(ge=0)
    misses: int = Field(ge=0)
    size: int = Field(ge=0)
    capacity: int = Field(gt=0)

    @model_validator(mode="after")
    def _require_size_within_capacity(self) -> 'TokenCountCacheInfo':
        if self.size > self.capacity:
            raise ValueError("token-count cache size exceeds capacity")
        return self


class ProjectionBudgetEstimate(_FrozenContract):
    """一个当前有效上下文投影的第一级账单。"""

    schema_version: int = Field(default=1, ge=1, le=1)
    projection_epoch: str = Field(
        min_length=1,
        max_length=300,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:/_-]{0,299}$",
    )
    purpose: str = Field(
        min_length=1,
        max_length=200,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:/_-]{0,199}$",
    )
    projection_generation: int = Field(ge=0)
    token_counter: TokenCounterFingerprint
    blocks: tuple[ContextBudgetBlock, ...]
    block_count: int = Field(ge=0)
    fixed_overhead_tokens: int = Field(ge=0)
    fixed_overhead_utf8_bytes: int = Field(ge=0)
    mandatory_tokens: int = Field(ge=0)
    optional_tokens: int = Field(ge=0)
    total_estimated_tokens: int = Field(ge=0)
    total_serialized_utf8_bytes: int = Field(ge=0)

    @model_validator(mode="after")
    def _require_consistent_bill(self) -> 'ProjectionBudgetEstimate':
        if self.block_count != len(self.blocks):
            raise ValueError("projection block_count is inconsistent")
        block_ids = tuple(block.block_id for block in self.blocks)
        if len(set(block_ids)) != len(block_ids):
            raise ValueError("projection contains duplicate block_id values")
        if any(block.token_counter != self.token_counter for block in self.blocks):
            raise ValueError("projection blocks use different token counters")
        mandatory = sum(
            block.estimated_tokens for block in self.blocks if block.mandatory
        )
        optional = sum(
            block.estimated_tokens for block in self.blocks if not block.mandatory
        )
        if self.mandatory_tokens != mandatory or self.optional_tokens != optional:
            raise ValueError("projection mandatory/optional token bill is inconsistent")
        if self.total_estimated_tokens != (
            mandatory + optional + self.fixed_overhead_tokens
        ):
            raise ValueError("projection total token bill is inconsistent")
        if self.total_serialized_utf8_bytes != (
            sum(block.serialized_utf8_bytes for block in self.blocks)
            + self.fixed_overhead_utf8_bytes
        ):
            raise ValueError("projection total byte bill is inconsistent")
        return self


class ContextWindowProfile(_FrozenContract):
    """一个物理提供商路由的冻结容量事实。"""

    schema_version: int = Field(default=1, ge=1, le=1)
    profile_id: str = Field(
        min_length=1,
        max_length=200,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:/_-]{0,199}$",
    )
    provider: str = Field(min_length=1, max_length=200)
    model: str = Field(min_length=1, max_length=500)
    dialect: str = Field(min_length=1, max_length=200)
    context_window_tokens: int = Field(gt=0)
    configured_input_limit_tokens: int | None = Field(default=None, gt=0)
    reserved_output_tokens: int = Field(ge=0)
    shared_reasoning_reserve_tokens: int = Field(ge=0)
    safety_margin_tokens: int = Field(ge=0)
    soft_pressure_ratio: float = Field(gt=0.0, lt=1.0)
    max_serialized_request_utf8_bytes: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _require_positive_input_budget(self) -> 'ContextWindowProfile':
        if self.input_budget_tokens <= 0:
            raise ValueError("context-window reserves leave no input-token budget")
        return self

    @property
    def provider_input_capacity_tokens(self) -> int:
        return (
            self.context_window_tokens
            - self.reserved_output_tokens
            - self.shared_reasoning_reserve_tokens
            - self.safety_margin_tokens
        )

    @property
    def input_budget_tokens(self) -> int:
        return min(
            self.provider_input_capacity_tokens,
            self.configured_input_limit_tokens
            or self.provider_input_capacity_tokens,
        )


class EnvelopeTokenComponent(_FrozenContract):
    """最终词元账单中的一个提供商信封组件。"""

    schema_version: int = Field(default=1, ge=1, le=1)
    component_id: str = Field(
        min_length=1,
        max_length=200,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:/_-]{0,199}$",
    )
    tokens: int = Field(ge=0)
    estimate_kind: TokenEstimateKind


class ProviderEnvelopeMeterResult(_FrozenContract):
    """由可信提供商线上计量器生成的不含内容结果。"""

    schema_version: int = Field(default=1, ge=1, le=1)
    output_token_limit: int = Field(ge=0)
    shared_reasoning_reserve_tokens: int = Field(ge=0)
    request_controls_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    input_tokens: int = Field(gt=0)
    quota_input_token_estimate: int = Field(gt=0)
    components: tuple[EnvelopeTokenComponent, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _require_consistent_meter_result(self) -> 'ProviderEnvelopeMeterResult':
        component_ids = tuple(item.component_id for item in self.components)
        if len(set(component_ids)) != len(component_ids):
            raise ValueError("provider meter returned duplicate component ids")
        if self.input_tokens != sum(item.tokens for item in self.components):
            raise ValueError("provider meter token bill is inconsistent")
        if self.quota_input_token_estimate > self.input_tokens:
            raise ValueError("quota input estimate exceeds the hard input bound")
        return self


class ProviderEnvelopeTokenMeter(TokenMeterIdentity, Protocol):
    """针对一个精确提供商请求正文、可信且绑定路由的适配器接缝。"""

    provider: str
    model: str
    dialect: str

    def measure(self, serialized_envelope: bytes) -> ProviderEnvelopeMeterResult: ...


class CanonicalEnvelopeMeasurement(_FrozenContract):
    """对选定用于派发的精确序列化请求执行的测量。"""

    schema_version: int = Field(default=1, ge=1, le=1)
    provider: str = Field(min_length=1, max_length=200)
    model: str = Field(min_length=1, max_length=500)
    dialect: str = Field(min_length=1, max_length=200)
    request_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    serialized_request_utf8_bytes: int = Field(ge=0)
    output_token_limit: int = Field(ge=0)
    shared_reasoning_reserve_tokens: int = Field(ge=0)
    request_controls_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    input_tokens: int = Field(gt=0)
    quota_input_token_estimate: int = Field(gt=0)
    token_counter: TokenCounterFingerprint
    components: tuple[EnvelopeTokenComponent, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _require_consistent_components(self) -> 'CanonicalEnvelopeMeasurement':
        component_ids = tuple(item.component_id for item in self.components)
        if len(set(component_ids)) != len(component_ids):
            raise ValueError("envelope contains duplicate token component ids")
        if self.input_tokens != sum(item.tokens for item in self.components):
            raise ValueError("envelope input token bill is inconsistent")
        if self.quota_input_token_estimate > self.input_tokens:
            raise ValueError("quota input estimate exceeds the hard input bound")
        if (
            self.token_counter.kind != "heuristic"
            and self.quota_input_token_estimate != self.input_tokens
        ):
            raise ValueError(
                "non-heuristic input measurement requires equal token counts"
            )
        estimate_kinds = {item.estimate_kind for item in self.components}
        if (
            TokenEstimateKind.PROVIDER_EXACT in estimate_kinds
            and self.token_counter.kind != "provider_exact"
        ):
            raise ValueError(
                "provider-exact components require a provider-exact meter"
            )
        if (
            TokenEstimateKind.PROVIDER_USAGE_ANCHORED in estimate_kinds
            and self.token_counter.kind
            not in {"provider_usage_anchored", "provider_exact"}
        ):
            raise ValueError(
                "usage-anchored components require a usage-anchored meter"
            )
        if (
            TokenEstimateKind.PROVIDER_EXACT not in estimate_kinds
            and TokenEstimateKind.PROVIDER_USAGE_ANCHORED not in estimate_kinds
        ):
            if estimate_kinds == {TokenEstimateKind.TOKENIZER}:
                if self.token_counter.kind != "tokenizer":
                    raise ValueError(
                        "tokenizer components require a tokenizer meter"
                    )
            elif estimate_kinds == {TokenEstimateKind.HEURISTIC}:
                if self.token_counter.kind != "heuristic":
                    raise ValueError(
                        "heuristic components require a heuristic meter"
                    )
            elif estimate_kinds == {
                TokenEstimateKind.HEURISTIC,
                TokenEstimateKind.TOKENIZER,
            }:
                # tokenizer 可以精确计量文本部分，而非文本模态（目前为内联图像）会如实保留
                # 较弱的启发式来源。这一区别由组件账单而非汇总计数器标签来保存。
                if self.token_counter.kind != "tokenizer":
                    raise ValueError(
                        "mixed tokenizer/heuristic components require a tokenizer meter"
                    )
            else:
                raise ValueError(
                    "non-provider envelope components must use one estimate kind"
                )
        return self


class ContextBudgetReceipt(_FrozenContract):
    """最终派发前上下文门产生的不含内容结果。"""

    schema_version: int = Field(default=1, ge=1, le=1)
    profile_id: str = Field(
        min_length=1,
        max_length=200,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:/_-]{0,199}$",
    )
    provider: str = Field(min_length=1, max_length=200)
    model: str = Field(min_length=1, max_length=500)
    dialect: str = Field(min_length=1, max_length=200)
    purpose: str = Field(
        min_length=1,
        max_length=200,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:/_-]{0,199}$",
    )
    projection_epoch: str = Field(
        min_length=1,
        max_length=300,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:/_-]{0,299}$",
    )
    projection_generation: int = Field(ge=0)
    request_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    context_window_tokens: int = Field(gt=0)
    configured_input_limit_tokens: int | None = Field(default=None, gt=0)
    provider_input_capacity_tokens: int = Field(gt=0)
    reserved_output_tokens: int = Field(ge=0)
    shared_reasoning_reserve_tokens: int = Field(ge=0)
    safety_margin_tokens: int = Field(ge=0)
    request_controls_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    input_budget_tokens: int = Field(gt=0)
    soft_input_limit_tokens: int = Field(gt=0)
    input_tokens: int = Field(gt=0)
    quota_input_token_estimate: int = Field(gt=0)
    serialized_request_utf8_bytes: int = Field(ge=0)
    max_serialized_request_utf8_bytes: int | None = Field(default=None, gt=0)
    soft_pressure_ratio: float = Field(gt=0.0, lt=1.0)
    pressure_ratio: float = Field(ge=0.0)
    pressure: ContextBudgetPressure
    admitted: bool
    violations: tuple[ContextBudgetViolation, ...]
    token_counter: TokenCounterFingerprint
    components: tuple[EnvelopeTokenComponent, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _require_consistent_decision(self) -> 'ContextBudgetReceipt':
        expected_provider_capacity = (
            self.context_window_tokens
            - self.reserved_output_tokens
            - self.shared_reasoning_reserve_tokens
            - self.safety_margin_tokens
        )
        if self.provider_input_capacity_tokens != expected_provider_capacity:
            raise ValueError("receipt provider input capacity is inconsistent")
        expected_input_budget = min(
            expected_provider_capacity,
            self.configured_input_limit_tokens or expected_provider_capacity,
        )
        if self.input_budget_tokens != expected_input_budget:
            raise ValueError("receipt input-token budget is inconsistent")
        expected_soft_limit = max(
            1,
            math.ceil(self.input_budget_tokens * self.soft_pressure_ratio),
        )
        if self.soft_input_limit_tokens != expected_soft_limit:
            raise ValueError("receipt soft input limit is inconsistent")
        expected_ratio = self.input_tokens / self.input_budget_tokens
        if not math.isclose(
            self.pressure_ratio,
            expected_ratio,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError("receipt pressure ratio is inconsistent")
        if len(set(self.violations)) != len(self.violations):
            raise ValueError("receipt contains duplicate budget violations")
        expected_violations: list[ContextBudgetViolation] = []
        if self.input_tokens > self.input_budget_tokens:
            expected_violations.append(ContextBudgetViolation.INPUT_TOKENS)
        if (
            self.max_serialized_request_utf8_bytes is not None
            and self.serialized_request_utf8_bytes
            > self.max_serialized_request_utf8_bytes
        ):
            expected_violations.append(ContextBudgetViolation.REQUEST_BODY_BYTES)
        if self.violations != tuple(expected_violations):
            raise ValueError("receipt budget violations are inconsistent")
        if self.admitted == bool(self.violations):
            raise ValueError("budget admission disagrees with its violations")
        if self.soft_input_limit_tokens > self.input_budget_tokens:
            raise ValueError("soft input limit exceeds the hard input budget")
        if self.pressure is ContextBudgetPressure.HARD_LIMIT_EXCEEDED:
            if not self.violations:
                raise ValueError("hard pressure requires a budget violation")
        elif self.violations:
            raise ValueError("budget violations require hard pressure")
        expected_pressure = (
            ContextBudgetPressure.HARD_LIMIT_EXCEEDED
            if self.violations
            else (
                ContextBudgetPressure.SOFT_LIMIT_EXCEEDED
                if self.input_tokens >= self.soft_input_limit_tokens
                else ContextBudgetPressure.NORMAL
            )
        )
        if self.pressure is not expected_pressure:
            raise ValueError("receipt pressure state is inconsistent")
        if self.input_tokens != sum(item.tokens for item in self.components):
            raise ValueError("receipt input token bill is inconsistent")
        if self.quota_input_token_estimate > self.input_tokens:
            raise ValueError("receipt quota input estimate exceeds the hard bound")
        if (
            self.token_counter.kind != "heuristic"
            and self.quota_input_token_estimate != self.input_tokens
        ):
            raise ValueError(
                "non-heuristic receipt requires equal input token counts"
            )
        component_ids = tuple(item.component_id for item in self.components)
        if len(set(component_ids)) != len(component_ids):
            raise ValueError("receipt contains duplicate token component ids")
        estimate_kinds = {item.estimate_kind for item in self.components}
        if (
            TokenEstimateKind.PROVIDER_EXACT in estimate_kinds
            and self.token_counter.kind != "provider_exact"
        ):
            raise ValueError(
                "receipt provider-exact components require a provider-exact meter"
            )
        if (
            TokenEstimateKind.PROVIDER_USAGE_ANCHORED in estimate_kinds
            and self.token_counter.kind
            not in {"provider_usage_anchored", "provider_exact"}
        ):
            raise ValueError(
                "receipt usage-anchored components require an anchored meter"
            )
        return self


__all__ = [
    'CanonicalEnvelopeMeasurement',
    'ContextBlockStability',
    'ContextBlockAdmissionReason',
    'ContextBlockAdmission',
    'ContextBlockBudgetSnapshot',
    'ContextBudgetBlock',
    'ContextBudgetPressure',
    'ContextBudgetReceipt',
    'ContextBudgetViolation',
    'ContextWindowProfile',
    'EnvelopeTokenComponent',
    'ProjectionBudgetEstimate',
    'ProviderEnvelopeMeterResult',
    "ProviderEnvelopeTokenMeter",
    "TextTokenCounter",
    'TokenCountCacheInfo',
    'TokenCounterFingerprint',
    'TokenEstimateKind',
    "TokenMeterIdentity",
]
