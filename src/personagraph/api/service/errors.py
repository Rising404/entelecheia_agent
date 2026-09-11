from __future__ import annotations

from typing import Any

class ApiError(Exception):
    """可渲染为 JSON 的结构化 API 错误。"""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        status: int = 400,
        details: dict[str, Any] | None = None,
        outcome: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status
        self.details = details or {}
        self.outcome = outcome

    def to_payload(self) -> dict[str, Any]:
        payload = {
            "error": {
                "code": self.code,
                "message": self.message,
                "details": self.details,
            }
        }
        if self.outcome is not None:
            payload["outcome"] = self.outcome
        return payload
