"""用户附加到单个聊天 Turn 的持久 wire 契约。"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from personagraph.input_processing.files import FileKind


# 附件契约保留领域名称；粗粒度文件分类由 input_processing 共享层拥有。
AttachmentKind = FileKind


class AttachmentOrigin(StrEnum):
    """文件的生产者；当前只会记录 ``USER_UPLOAD``。"""

    USER_UPLOAD = "user_upload"
    AGENT_CREATED = "agent_created"
    AGENT_MODIFIED = "agent_modified"


AttachmentUploadOutcome = Literal["stored", "rejected"]
