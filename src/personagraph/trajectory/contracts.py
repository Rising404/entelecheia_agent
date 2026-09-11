"""独立于存储方式定义一条记录步骤。

这些类型描述运行时从不保留的两项内容：步骤收到什么、产生什么。事件流已经
记录步骤发生过——阶段、状态、时间与父级——因此这里不重复。记录步骤通过
``model_call_id`` / ``turn_id`` 指回事件。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import StrEnum


class StepKind(StrEnum):
    """哪个关键入口产生了该步骤，或 Recorder 自身是否失败。"""

    MODEL_CALL = "model_call"
    TOOL_CALL = "tool_call"
    RETRIEVAL = "retrieval"
    RECORDING_FAILURE = "recording_failure"


class PartRole(StrEnum):
    """步骤中一个组成部分的角色。

    即使载荷看起来相似，也要区分角色，因为读取轨迹是在询问角色相关问题：
    它看到了什么、决定了什么、丢弃了什么。尤其是 ``REJECTED_OUTPUT``，事件流
    无法回答它；事件流只记录某物被拒绝，却不记录被拒绝的具体内容。
    """

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    THINKING = "thinking"
    TOOL_ARGUMENTS = "tool_arguments"
    TOOL_RESULT = "tool_result"
    QUERY = "query"
    EVIDENCE = "evidence"
    RETRIEVAL_METHOD = "retrieval_method"
    FUSION = "fusion"
    RERANKER = "reranker"
    RETRIEVAL_DIAGNOSTIC = "retrieval_diagnostic"
    REJECTED_OUTPUT = "rejected_output"


class StepOutcome(StrEnum):
    OK = "ok"
    REJECTED = "rejected"
    FAILED = "failed"


# 数据块在此大小以内会完整保存；超过后存储文本会截断，但 `byte_count` 保留真实
# 大小，`truncated` 明确说明。静默缩短自身证据的轨迹，比完全不保留更糟。
MAX_BLOB_BYTES = 1 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class Blob:
    """一段按完整内容而非来源位置寻址的文本。

    内容寻址让存储成本可控：某阶段 384 次调用都重复发送的系统提示只占一行，
    被引用 384 次。超长文本只保存前缀，但哈希和字节数仍描述完整原文。
    """

    sha256: str
    byte_count: int
    text: str
    truncated: bool = False

    def __post_init__(self) -> None:
        if len(self.sha256) != 64 or not all(c in "0123456789abcdef" for c in self.sha256):
            raise ValueError("sha256 must be 64 lowercase hex characters")
        if self.byte_count < 0:
            raise ValueError("byte_count must not be negative")
        if not isinstance(self.text, str):
            raise ValueError("blob text must be a string")


def text_blob(text: str) -> Blob:
    """按完整内容寻址一段文本，并在上限内保存。"""

    encoded = text.encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    if len(encoded) <= MAX_BLOB_BYTES:
        return Blob(sha256=digest, byte_count=len(encoded), text=text)
    # 在字符边界而非字节边界截断，使保留前缀仍可读取，不会以残缺序列结尾。
    kept = encoded[:MAX_BLOB_BYTES].decode("utf-8", errors="ignore")
    return Blob(sha256=digest, byte_count=len(encoded), text=kept, truncated=True)


@dataclass(frozen=True, slots=True)
class Part:
    role: PartRole
    blob: Blob


@dataclass(frozen=True, slots=True)
class Step:
    """一条记录步骤：它看到了什么、产生了什么，以及成本是多少。"""

    step_id: str
    kind: StepKind
    occurred_at: str
    parts: tuple[Part, ...] = ()
    session_id: str | None = None
    turn_id: str | None = None
    model_call_id: str | None = None
    purpose: str | None = None
    duration_ms: int | None = None
    outcome: StepOutcome = StepOutcome.OK
    reason_code: str | None = None
    metrics: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.step_id.strip():
            raise ValueError("step_id must not be empty")
        if not self.occurred_at.strip():
            raise ValueError("occurred_at must not be empty")
        if self.outcome is not StepOutcome.OK and not self.reason_code:
            raise ValueError("a step that did not succeed must carry a reason_code")
