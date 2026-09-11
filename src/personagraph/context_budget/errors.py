"""上下文预算测量与准入所引发的类型化失败。"""

from __future__ import annotations

from .contracts import ContextBudgetReceipt


class ContextBudgetInputError(ValueError):
    """无法确定性地测量输入。"""


class ContextBudgetEncodingError(ContextBudgetInputError):
    """提示词材料或提供商信封不是受支持的 UTF-8 JSON。"""


class ContextBudgetRouteMismatch(ContextBudgetInputError):
    """测量结果所依据的物理提供商路由与检查路由不同。"""


class ContextBudgetControlMismatch(ContextBudgetInputError):
    """测得的线上控制参数与冻结容量配置不一致。"""


class ContextBudgetEnvelopeMismatch(ContextBudgetInputError):
    """派发字节与硬门所准入的信封不同。"""


class ContextBudgetMeasurementError(RuntimeError):
    """配置的计数器无法生成安全的输入测量结果。"""

    code = "CONTEXT_BUDGET_MEASUREMENT_FAILED"
    retryable = False


class ContextBudgetExceeded(RuntimeError):
    """块装配结果或最终提供商信封超出了硬限制。"""

    code = "CONTEXT_BUDGET_EXCEEDED"
    retryable = False

    def __init__(
        self,
        receipt: ContextBudgetReceipt | None = None,
        *,
        limit: int | None = None,
        estimated_tokens: int | None = None,
        degraded: tuple[str, ...] = (),
    ) -> None:
        if receipt is not None:
            if receipt.admitted:
                raise ValueError(
                    "an admitted receipt cannot raise ContextBudgetExceeded"
                )
            if limit is not None or estimated_tokens is not None or degraded:
                raise ValueError(
                    "receipt cannot be combined with legacy budget arguments"
                )
            self.receipt = receipt
            self.violations = receipt.violations
            self.limit = receipt.input_budget_tokens
            self.estimated_tokens = receipt.input_tokens
            self.degraded: tuple[str, ...] = ()
            self.stage = "provider_envelope"
            joined = ",".join(item.value for item in receipt.violations)
            message = (
                "Canonical model request exceeds its context budget: "
                f"profile={receipt.profile_id}, violations={joined}"
            )
        else:
            if (
                isinstance(limit, bool)
                or not isinstance(limit, int)
                or limit <= 0
                or isinstance(estimated_tokens, bool)
                or not isinstance(estimated_tokens, int)
                or estimated_tokens < 0
            ):
                raise ValueError(
                    "legacy context budget failure requires positive limit and "
                    "non-negative estimated_tokens"
                )
            self.receipt = None
            self.violations = ()
            self.limit = limit
            self.estimated_tokens = estimated_tokens
            self.degraded = tuple(degraded)
            self.stage = "block_assembly"
            message = (
                "Prompt context exceeds hard limit: "
                f"estimated={estimated_tokens}, limit={limit}"
            )
        super().__init__(message)


class MandatoryContextBlockBudgetExceeded(ContextBudgetExceeded):
    """不可降级块无法装入第一级输入预算。"""

    code = "CONTEXT_BUDGET_EXCEEDED"
    retryable = False

    def __init__(
        self,
        *,
        block_id: str,
        block_tokens: int,
        remaining_tokens: int,
        input_budget_tokens: int,
    ) -> None:
        used_tokens_before = input_budget_tokens - remaining_tokens
        super().__init__(
            limit=input_budget_tokens,
            estimated_tokens=used_tokens_before + block_tokens,
            degraded=(),
        )
        self.block_id = block_id
        self.block_tokens = block_tokens
        self.remaining_tokens = remaining_tokens
        self.input_budget_tokens = input_budget_tokens
        self.stage = "block_admission"
        self.args = (
            "Mandatory context block exceeds the remaining first-level budget: "
            f"block_id={block_id}, tokens={block_tokens}, "
            f"remaining={remaining_tokens}",
        )


__all__ = [
    "ContextBudgetEncodingError",
    "ContextBudgetControlMismatch",
    "ContextBudgetEnvelopeMismatch",
    "ContextBudgetExceeded",
    "ContextBudgetInputError",
    "ContextBudgetMeasurementError",
    "ContextBudgetRouteMismatch",
    "MandatoryContextBlockBudgetExceeded",
]
