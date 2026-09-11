"""对当前有效上下文块执行第一级增量测量。"""

from __future__ import annotations

from collections import OrderedDict
from hashlib import sha256

from .contracts import (
    ContextBlockStability,
    ContextBlockAdmissionReason,
    ContextBlockAdmission,
    ContextBlockBudgetSnapshot,
    ContextBudgetBlock,
    ProjectionBudgetEstimate,
    TextTokenCounter,
    TokenCountCacheInfo,
    TokenCounterFingerprint,
    TokenMeterIdentity,
)
from .errors import (
    ContextBudgetEncodingError,
    ContextBudgetMeasurementError,
    MandatoryContextBlockBudgetExceeded,
)


def token_counter_fingerprint(counter: TokenMeterIdentity) -> TokenCounterFingerprint:
    """冻结一次估算或信封计量所用的计数器标识。"""

    try:
        name = str(counter.name or "").strip()
        kind = str(counter.kind or "").strip()
        requested = str(counter.requested or "").strip()
        model_value = counter.model
        fallback_value = counter.fallback_reason
        return TokenCounterFingerprint(
            name=name,
            kind=kind,
            requested=requested,
            model=(str(model_value).strip() if model_value is not None else None),
            fallback_reason=(
                str(fallback_value).strip()
                if fallback_value is not None
                else None
            ),
        )
    except Exception as exc:
        raise ContextBudgetMeasurementError(
            "token counter has no valid content-free fingerprint"
        ) from exc


def _count_text(counter: TextTokenCounter, text: str) -> int:
    try:
        value = counter.count_text(text)
    except Exception as exc:
        raise ContextBudgetMeasurementError(
            "token counter failed while measuring a context block"
        ) from exc
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ContextBudgetMeasurementError(
            "token counter must return a non-negative integer"
        )
    return value


class IncrementalContextBudgetEstimator:
    """在投影重建之间复用不可变文本测量结果。

    缓存键是 UTF-8 内容哈希，且估算器只持有一个冻结计数器标识。因此，发生更改的块会自动
    缓存未命中，而相同内容可跨块 ID 和投影复用。
    """

    def __init__(
        self,
        counter: TextTokenCounter,
        *,
        max_cache_entries: int = 4_096,
    ) -> None:
        if isinstance(max_cache_entries, bool) or not isinstance(
            max_cache_entries, int
        ) or max_cache_entries <= 0:
            raise ValueError("max_cache_entries must be a positive integer")
        self._counter = counter
        self._fingerprint = token_counter_fingerprint(counter)
        self._capacity = max_cache_entries
        self._cache: OrderedDict[str, tuple[int, int]] = OrderedDict()
        self._hits = 0
        self._misses = 0

    @property
    def token_counter(self) -> TokenCounterFingerprint:
        return self._fingerprint

    def measure_text(
        self,
        *,
        block_id: str,
        content: str,
        stability: ContextBlockStability,
        mandatory: bool,
        priority: int,
        source_revision: str | None = None,
    ) -> ContextBudgetBlock:
        """测量一个文本块，但不保留其原始内容。"""

        if not isinstance(content, str):
            raise TypeError("content must be a string")
        if token_counter_fingerprint(self._counter) != self._fingerprint:
            raise ContextBudgetMeasurementError(
                "token counter identity changed after estimator construction"
            )
        try:
            encoded = content.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise ContextBudgetEncodingError(
                "context block cannot be encoded as UTF-8"
            ) from exc
        content_sha256 = sha256(encoded).hexdigest()
        cached = self._cache.get(content_sha256)
        if cached is None:
            measured = (_count_text(self._counter, content), len(encoded))
            self._cache[content_sha256] = measured
            self._cache.move_to_end(content_sha256)
            if len(self._cache) > self._capacity:
                self._cache.popitem(last=False)
            self._misses += 1
        else:
            measured = cached
            self._cache.move_to_end(content_sha256)
            self._hits += 1
        estimated_tokens, serialized_utf8_bytes = measured
        return ContextBudgetBlock(
            block_id=block_id,
            content_sha256=content_sha256,
            source_revision=source_revision,
            stability=stability,
            mandatory=mandatory,
            priority=priority,
            estimated_tokens=estimated_tokens,
            serialized_utf8_bytes=serialized_utf8_bytes,
            token_counter=self._fingerprint,
        )

    def estimate_projection(
        self,
        blocks: tuple[ContextBudgetBlock, ...],
        *,
        projection_epoch: str,
        purpose: str,
        projection_generation: int,
        fixed_overhead_tokens: int = 0,
        fixed_overhead_utf8_bytes: int = 0,
    ) -> ProjectionBudgetEstimate:
        """构建供早期装填决策使用的不含内容账单。"""

        mandatory_tokens = sum(
            block.estimated_tokens for block in blocks if block.mandatory
        )
        optional_tokens = sum(
            block.estimated_tokens for block in blocks if not block.mandatory
        )
        return ProjectionBudgetEstimate(
            projection_epoch=projection_epoch,
            purpose=purpose,
            projection_generation=projection_generation,
            token_counter=self._fingerprint,
            blocks=blocks,
            block_count=len(blocks),
            fixed_overhead_tokens=fixed_overhead_tokens,
            fixed_overhead_utf8_bytes=fixed_overhead_utf8_bytes,
            mandatory_tokens=mandatory_tokens,
            optional_tokens=optional_tokens,
            total_estimated_tokens=(
                mandatory_tokens + optional_tokens + fixed_overhead_tokens
            ),
            total_serialized_utf8_bytes=(
                sum(block.serialized_utf8_bytes for block in blocks)
                + fixed_overhead_utf8_bytes
            ),
        )

    def cache_info(self) -> TokenCountCacheInfo:
        return TokenCountCacheInfo(
            hits=self._hits,
            misses=self._misses,
            size=len(self._cache),
            capacity=self._capacity,
        )

    def clear_cache(self) -> None:
        self._cache.clear()
        self._hits = 0
        self._misses = 0


class IncrementalContextBlockBudget:
    """准入已测量的块，同时绝不超出输入预算。

    调用方持有装填策略，且必须按确定性的优先级顺序提供块。无法装入的可选块会被拒绝，
    且不改变词元用量；无法装入的必选块则采用失败关闭策略。
    """

    def __init__(
        self,
        *,
        input_budget_tokens: int,
        fixed_overhead_tokens: int = 0,
    ) -> None:
        if (
            isinstance(input_budget_tokens, bool)
            or not isinstance(input_budget_tokens, int)
            or input_budget_tokens <= 0
        ):
            raise ValueError("input_budget_tokens must be a positive integer")
        if (
            isinstance(fixed_overhead_tokens, bool)
            or not isinstance(fixed_overhead_tokens, int)
            or fixed_overhead_tokens < 0
            or fixed_overhead_tokens > input_budget_tokens
        ):
            raise ValueError(
                "fixed_overhead_tokens must fit within the input budget"
            )
        self._input_budget_tokens = input_budget_tokens
        self._fixed_overhead_tokens = fixed_overhead_tokens
        self._used_tokens = fixed_overhead_tokens
        self._admitted: list[ContextBudgetBlock] = []
        self._seen_block_ids: set[str] = set()
        self._token_counter: TokenCounterFingerprint | None = None

    def try_admit(
        self,
        block: ContextBudgetBlock,
    ) -> ContextBlockAdmission:
        if not isinstance(block, ContextBudgetBlock):
            raise TypeError("block must be a ContextBudgetBlock")
        if block.block_id in self._seen_block_ids:
            raise ValueError(f"duplicate context block id: {block.block_id}")
        if (
            self._token_counter is not None
            and block.token_counter != self._token_counter
        ):
            raise ValueError("context blocks use different token counters")

        before = self._used_tokens
        fits = before + block.estimated_tokens <= self._input_budget_tokens
        if not fits and block.mandatory:
            raise MandatoryContextBlockBudgetExceeded(
                block_id=block.block_id,
                block_tokens=block.estimated_tokens,
                remaining_tokens=self._input_budget_tokens - before,
                input_budget_tokens=self._input_budget_tokens,
            )

        self._seen_block_ids.add(block.block_id)
        if self._token_counter is None:
            self._token_counter = block.token_counter
        if fits:
            self._admitted.append(block)
            self._used_tokens += block.estimated_tokens
        return ContextBlockAdmission(
            block=block,
            accepted=fits,
            reason=(
                ContextBlockAdmissionReason.ADMITTED
                if fits
                else ContextBlockAdmissionReason.INSUFFICIENT_BUDGET
            ),
            used_tokens_before=before,
            used_tokens_after=self._used_tokens,
            remaining_tokens=self._input_budget_tokens - self._used_tokens,
            input_budget_tokens=self._input_budget_tokens,
        )

    def snapshot(self) -> ContextBlockBudgetSnapshot:
        return ContextBlockBudgetSnapshot(
            input_budget_tokens=self._input_budget_tokens,
            fixed_overhead_tokens=self._fixed_overhead_tokens,
            admitted_blocks=tuple(self._admitted),
            used_tokens=self._used_tokens,
            remaining_tokens=self._input_budget_tokens - self._used_tokens,
        )


__all__ = [
    "IncrementalContextBudgetEstimator",
    "IncrementalContextBlockBudget",
    "token_counter_fingerprint",
]
