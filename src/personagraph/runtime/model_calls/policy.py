"""物理模型调用共享、可冷导入的重试限制。"""

from __future__ import annotations

import random


# 这是一个持久化逻辑调用内 provider 重试的 Host 全局上限。它有意与请求 executor
# 分离，使 binding 与 ledger 无需导入 model/runtime 机制即可携带精确限制。
MAX_MODEL_ATTEMPTS = 6
# 无延迟重试受限流或过载 provider 会使故障放大，因此传输重试采用带完整抖动的
# 指数退避；输出上限类响应则属于同一请求不可能自愈的终态。
BACKOFF_BASE_S = 0.5
BACKOFF_MAX_S = 8.0
MODEL_OUTPUT_LIMIT_FINISH_REASONS = frozenset(
    {"max_tokens", "length", "model_length", "token_limit"}
)


def backoff_delay_s(attempt: int) -> float:
    """物理 attempt ``attempt + 1`` 前带完整抖动的指数延迟。"""

    ceiling = min(BACKOFF_MAX_S, BACKOFF_BASE_S * (2 ** max(0, attempt - 1)))
    return random.uniform(0.0, ceiling)


__all__ = [
    "BACKOFF_BASE_S",
    "BACKOFF_MAX_S",
    "MAX_MODEL_ATTEMPTS",
    "MODEL_OUTPUT_LIMIT_FINISH_REASONS",
    "backoff_delay_s",
]
