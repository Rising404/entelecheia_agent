"""文件准备与状态查询共用的应用结果，不依赖模型侧资源句柄。"""

from dataclasses import dataclass, field
from enum import StrEnum


class FilePreparationStatus(StrEnum):
    READY = "ready"
    PENDING = "pending"
    BLOCKED = "blocked"
    STALE = "stale"


@dataclass(frozen=True, slots=True)
class FilePreparationResult:
    status: FilePreparationStatus
    request_id: str | None = field(default=None, repr=False)
    operation_id: str | None = field(default=None, repr=False)
    file_id: str | None = None
    file_version_id: str | None = None
    reason_code: str | None = None
    replayed: bool = False
    retrieval_data_version: str | None = field(default=None, repr=False)
    document_id: str | None = field(default=None, repr=False)
    document_version_id: str | None = field(default=None, repr=False)
