"""Workspace ingestion 的可重试与终止控制流错误。"""

from __future__ import annotations


class DocumentIngestTerminalFailure(RuntimeError):
    def __init__(self, reason_code: str) -> None:
        self.reason_code = reason_code
        super().__init__(reason_code)


class DocumentIngestRetryableFailure(RuntimeError):
    def __init__(self, reason_code: str) -> None:
        self.reason_code = reason_code
        super().__init__(reason_code)


__all__ = ["DocumentIngestRetryableFailure", "DocumentIngestTerminalFailure"]
