"""持久证据引用的稳定身份与“无普通证据”说明。

L1 的结果 ID 从调用 ID 和结果摘要派生，无需独立数据库行。该身份仅供精确引用，
不代表证据已通过验证。“无普通证据”说明同样不使完成声明自动成立。
"""

from __future__ import annotations

from enum import StrEnum
import hashlib
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


EMPTY_SUPPORT_JUSTIFICATION_CONTRACT_VERSION = (
    "empty-support-justification-v1"
)


def l1_tool_result_id(*, tool_call_id: str, result_sha256: str) -> str:
    """从 Host 权威调用与结果摘要派生身份；保持既有持久 ID 的逐字节算法。"""

    if not tool_call_id or not result_sha256 or len(result_sha256) != 64:
        raise ValueError("L1 ToolResult identity requires a call ID and sha256")
    payload = json.dumps(
        {"tool_call_id": tool_call_id, "result_sha256": result_sha256},
        ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"),
    )
    return "l1result_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


class EmptySupportReason(StrEnum):
    CANDIDATE_IS_PRIMARY_ARTIFACT = "candidate_is_primary_artifact"
    PROVIDED_CONTEXT_SUFFICIENT = "provided_context_sufficient"
    DEPENDENCY_DELIVERY_SUFFICIENT = "dependency_delivery_sufficient"
    EXTERNAL_EVIDENCE_NOT_REQUIRED = "external_evidence_not_required"
    SPECIALIZED_EVIDENCE_SIDECAR = "specialized_evidence_sidecar"
    EVIDENCE_UNAVAILABLE = "evidence_unavailable"
    EVIDENCE_ACCESS_BLOCKED = "evidence_access_blocked"


class EmptySupportJustification(BaseModel):
    """说明为何已完成声明没有常规支持 ID，而不充当替代证据。"""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    schema_version: Literal["empty-support-justification-v1"] = (
        EMPTY_SUPPORT_JUSTIFICATION_CONTRACT_VERSION
    )
    reason_code: EmptySupportReason
    explanation: str = Field(min_length=1, max_length=600)

    @field_validator("explanation")
    @classmethod
    def _require_canonical_explanation(cls, value: str) -> str:
        if value != value.strip() or "\x00" in value:
            raise ValueError(
                "empty-support explanation must be trimmed canonical text"
            )
        return value


__all__ = [
    "EMPTY_SUPPORT_JUSTIFICATION_CONTRACT_VERSION",
    "EmptySupportJustification",
    "EmptySupportReason",
    "l1_tool_result_id",
]
