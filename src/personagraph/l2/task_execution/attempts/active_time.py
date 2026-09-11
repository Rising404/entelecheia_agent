"""由 Host 管理的单个持久化 Attempt 活动时间测量。

该小型组件刻意不感知 Store 权威状态、模型调用、工具执行或 Attempt 结算。它只把 Host 单调时钟转换为一个缓存的正时间间隔，供幂等结算使用。
"""

from __future__ import annotations

import math
from collections.abc import Callable


class AttemptActiveTimeMeasurementError(RuntimeError):
    """注入的单调时钟无法生成一个正间隔。"""


class AttemptActiveTimeMeter:
    """由 Host 管理的单次活动时间间隔，从 Attempt 启动回执开始测量。

    ``freeze`` 会缓存第一个正时间差。因此，凡是保留该计时器的 Store
    重放或响应丢失处理，都会复用完全相同的负载值，而不会再次读取时钟。
    """

    def __init__(
        self,
        *,
        started_at_monotonic: float,
        monotonic_clock: Callable[[], float],
    ) -> None:
        self._started_at_monotonic = _require_monotonic_value(
            started_at_monotonic,
            label="Attempt start",
        )
        self._monotonic_clock = monotonic_clock
        self._frozen_delta: float | None = None

    @classmethod
    def starting_now(
        cls,
        monotonic_clock: Callable[[], float],
    ) -> "AttemptActiveTimeMeter":
        return cls(
            started_at_monotonic=monotonic_clock(),
            monotonic_clock=monotonic_clock,
        )

    def freeze(self) -> float:
        if self._frozen_delta is not None:
            return self._frozen_delta
        stopped_at = _require_monotonic_value(
            self._monotonic_clock(),
            label="Attempt settlement",
        )
        delta = stopped_at - self._started_at_monotonic
        if not math.isfinite(delta) or delta <= 0:
            raise AttemptActiveTimeMeasurementError(
                "Attempt active-time delta must be finite and positive"
            )
        self._frozen_delta = delta
        return delta


def _require_monotonic_value(value: float, *, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(value)
    ):
        raise AttemptActiveTimeMeasurementError(
            f"{label} monotonic timestamp must be finite"
        )
    return float(value)


__all__ = ["AttemptActiveTimeMeasurementError", "AttemptActiveTimeMeter"]
