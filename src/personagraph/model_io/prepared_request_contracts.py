"""Provider-neutral contracts for admitted model requests awaiting dispatch."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .contracts import ModelResult


@runtime_checkable
class PreparedModelRequest(Protocol):
    """Provider I/O 前已准备请求的结构化接缝。

    内置 provider 在实现此接缝前负责规范序列化与最终上下文预算 gate。Python 的结构化
    runtime 检查只能证明 ``dispatch`` 存在，无法证明第三方实现持有
    ``AdmittedContextRequest``；因此外部实现尚不属于封存生产边界。只有消费持久化
    权威状态后，dispatch 才接收物理调用标识，所以内置已准入 wire 字节无需仅为附加
    Host 侧标识而重建。
    """

    def dispatch(self, *, model_call_id: str) -> ModelResult: ...


@runtime_checkable
class QuotaPreparedModelRequest(Protocol):
    """由内置 gateway 实现的可选两阶段配额接缝。"""

    def acquire_api_quota(
        self,
        *,
        model_call_id: str,
        wait_timeout_seconds: float,
    ) -> object | None: ...

    def abandon_api_quota(
        self,
        permit: object | None,
        *,
        disposition: str = "cancel",
    ) -> None: ...

    def dispatch_with_api_quota(
        self,
        *,
        model_call_id: str,
        permit: object | None,
    ) -> ModelResult: ...


__all__ = ["PreparedModelRequest", "QuotaPreparedModelRequest"]
