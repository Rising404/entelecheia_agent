"""用于节点验证结算的纯 Host 单调时间测量。"""

from __future__ import annotations

import math
from collections.abc import Callable


class NodeVerificationActiveTimeMeasurementError(RuntimeError):
    """注入的 Host 单调时钟未产生正时间间隔。"""


class NodeVerificationActiveTimeMeter:
    """仅在获得持久化 prepare/resume receipt 后启动的一次性 Host 计时器。

    ``freeze`` 会记忆首个正时间差。因此结算命令可使用逐字节等价的计时输入重试，
    而无需再次采样时钟。
    """

    def __init__(
        self,
        *,
        started_at_monotonic: float,
        monotonic_clock: Callable[[], float],
    ) -> None:
        self._started_at_monotonic = _require_monotonic_value(
            started_at_monotonic,
            label="verification start",
        )
        self._monotonic_clock = monotonic_clock
        self._frozen_delta: float | None = None

    @classmethod
    def starting_now(
        cls,
        monotonic_clock: Callable[[], float],
    ) -> "NodeVerificationActiveTimeMeter":
        return cls(
            started_at_monotonic=monotonic_clock(),
            monotonic_clock=monotonic_clock,
        )

    def freeze(self) -> float:
        if self._frozen_delta is not None:
            return self._frozen_delta
        stopped_at = _require_monotonic_value(
            self._monotonic_clock(),
            label="verification settlement",
        )
        delta = stopped_at - self._started_at_monotonic
        if not math.isfinite(delta) or delta <= 0:
            raise NodeVerificationActiveTimeMeasurementError(
                "verification active-time delta must be finite and positive"
            )
        self._frozen_delta = delta
        return delta


def _require_monotonic_value(value: object, *, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(float(value))
    ):
        raise NodeVerificationActiveTimeMeasurementError(
            f"{label} monotonic value must be finite"
        )
    return float(value)


__all__ = [
    "NodeVerificationActiveTimeMeasurementError",
    "NodeVerificationActiveTimeMeter",
]
