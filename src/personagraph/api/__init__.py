"""面向前端和桌面 shell 的薄 HTTP API 适配器。

API 层围绕 Runtime、Project 文档和 Session 状态公开稳定 JSON 视图。它不能成为第二个业务
逻辑所有者。
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .service import ApiError

__all__ = ["ApiError"]


def __getattr__(name: str):
    """Keep importing API contracts independent of the service composition root."""

    if name == "ApiError":
        from .service import ApiError

        return ApiError
    raise AttributeError(name)
