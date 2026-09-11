"""针对临时检索基础设施故障的有界单请求恢复。"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import sqlite3
from typing import TypeVar

from ..execution import checkpoint
from ..ports import RetrievalCancelled


T = TypeVar("T")


class RetrievalCircuitOpen(RuntimeError):
    """某条后端路径在本次请求期间已耗尽重试预算。"""


@dataclass(frozen=True, slots=True)
class RecoveryResult:
    value: object | None
    attempts: int
    error: Exception | None
    circuit_open: bool = False
    attempt_errors: tuple[Exception, ...] = ()

    @property
    def succeeded(self) -> bool:
        return self.error is None and not self.circuit_open


class RetrievalRecoveryController:
    """每个请求最多按固定次数重试临时工作。

    熔断器作用域限定为一次 ``retrieve_context`` 调用。它会阻止第二个查询反复访问已耗尽
    预算的后端，同时不会把临时局部事故转化为持久全局状态。
    """

    def __init__(self, *, max_attempts: int = 3) -> None:
        if max_attempts <= 0:
            raise ValueError("max_attempts must be greater than zero")
        self._max_attempts = max_attempts
        self._open_keys: set[tuple[str, ...]] = set()

    def run(self, key: tuple[str, ...], operation: Callable[[], T]) -> RecoveryResult:
        if key in self._open_keys:
            return RecoveryResult(
                value=None,
                attempts=0,
                error=RetrievalCircuitOpen("retrieval_circuit_open"),
                circuit_open=True,
            )
        # 只在本次请求内保留异常对象；上层会把它们投影为安全错误码后再写 trajectory。
        # 这让“重试后成功”不再抹掉前几次基础设施故障。
        attempt_errors: list[Exception] = []
        for attempt in range(1, self._max_attempts + 1):
            try:
                checkpoint()
                value = operation()
                checkpoint()
                return RecoveryResult(
                    value=value,
                    attempts=attempt,
                    error=None,
                    attempt_errors=tuple(attempt_errors),
                )
            except RetrievalCancelled:
                raise
            except Exception as exc:
                attempt_errors.append(exc)
                if not _is_transient_infrastructure_error(exc) or attempt == self._max_attempts:
                    self._open_keys.add(key)
                    return RecoveryResult(
                        value=None,
                        attempts=attempt,
                        error=exc,
                        attempt_errors=tuple(attempt_errors),
                    )
        raise AssertionError("recovery loop must return")  # pragma: no cover


def _is_transient_infrastructure_error(exc: Exception) -> bool:
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True
    if isinstance(exc, sqlite3.OperationalError):
        message = str(exc).lower()
        return "locked" in message or "busy" in message
    return False
