"""一个运行时轮次共享的冷态墙钟截止时间契约。"""

from __future__ import annotations

from dataclasses import dataclass
from time import monotonic

from personagraph.model_io.gateway_core import ModelGatewayError


@dataclass(frozen=True)
class TurnDeadline:
    """整个轮次共享给所有模型调用的墙钟时间预算。

    仅限制单次调用无法约束用户的总等待时间：每次六轮、每轮六十秒，再乘以
    一个轮次发出的逻辑调用数，从用户视角看仍没有上限。本预算就是该总时长
    的硬上限。
    """

    expires_at_monotonic: float

    @classmethod
    def starting_now(cls, budget_s: float) -> "TurnDeadline":
        return cls(expires_at_monotonic=monotonic() + max(0.0, budget_s))

    def remaining_s(self) -> float:
        return max(0.0, self.expires_at_monotonic - monotonic())

    def expired(self) -> bool:
        return self.remaining_s() <= 0.0


class TurnDeadlineExceeded(ModelGatewayError):
    """整个 Turn 的墙钟时间耗尽，与任何单次调用无关。"""

    def __init__(self) -> None:
        super().__init__(
            "TURN_DEADLINE_EXCEEDED",
            "The turn exceeded its wall-clock budget.",
            retryable=False,
        )


__all__ = ["TurnDeadline", "TurnDeadlineExceeded"]
