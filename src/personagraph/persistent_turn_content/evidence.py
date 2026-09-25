"""持久证据引用的稳定身份与“无普通证据”说明。

L1 的短引用来自 Run 内持久调用坐标；完整身份与结果摘要保留在 Host。
引用本身不代表证据已通过验证。“无普通证据”说明同样不使完成声明自动成立。
"""

from __future__ import annotations

from enum import StrEnum
from collections.abc import Mapping
from copy import deepcopy
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


EMPTY_SUPPORT_JUSTIFICATION_CONTRACT_VERSION = (
    "empty-support-justification-v1"
)

L1_CALL_REF_PATTERN = r"^c[1-9][0-9]{0,18}\.(?:[1-9]|1[0-6])$"


def parse_l1_call_ref(value: str) -> tuple[int, int]:
    """解析 Run 内的持久调用坐标；拒绝别名、前导零及超出存储范围的编号。"""
    if not isinstance(value, str) or re.fullmatch(L1_CALL_REF_PATTERN, value) is None:
        raise ValueError("invalid L1 call reference")
    attempt, call = (int(part) for part in value[1:].split("."))
    if attempt > 2**63 - 1:
        raise ValueError("L1 attempt ordinal exceeds storage range")
    return attempt, call


def format_l1_call_ref(*, attempt_ordinal: int, call_ordinal: int) -> str:
    if type(attempt_ordinal) is not int or type(call_ordinal) is not int:
        raise ValueError("L1 call coordinates must be integers")
    value = f"c{attempt_ordinal}.{call_ordinal}"
    parse_l1_call_ref(value)
    return value


def l1_calls_by_ref(execution: Mapping[str, object]) -> dict[str, Mapping[str, object]]:
    """从已按 Run 授权的执行快照读取固定坐标，不按显示位置或内容生成身份。"""
    calls = execution.get("tool_calls")
    if not isinstance(calls, list):
        raise ValueError("L1 execution has no tool-call ledger")
    if not calls:
        return {}
    attempts = execution.get("attempts")
    if not isinstance(attempts, list):
        raise ValueError("L1 execution has no attempt ledger")
    ordinals = {}
    seen_ordinals = set()
    for attempt in attempts:
        if not isinstance(attempt, Mapping):
            raise ValueError("invalid L1 attempt record")
        identity, ordinal = attempt.get("attempt_id"), attempt.get("ordinal")
        if not isinstance(identity, str) or not identity or identity in ordinals or ordinal in seen_ordinals:
            raise ValueError("invalid or duplicated L1 attempt identity")
        format_l1_call_ref(attempt_ordinal=ordinal, call_ordinal=1)
        ordinals[identity] = ordinal
        seen_ordinals.add(ordinal)
    indexed = {}
    identities = set()
    for call in calls:
        if not isinstance(call, Mapping):
            raise ValueError("invalid L1 call record")
        identity = call.get("tool_call_id")
        if not isinstance(identity, str) or not identity or identity in identities:
            raise ValueError("invalid or duplicated L1 call identity")
        ref = format_l1_call_ref(
            attempt_ordinal=ordinals.get(call.get("attempt_id")),
            call_ordinal=call.get("call_ordinal"),
        )
        if ref in indexed:
            raise ValueError("duplicated L1 call coordinates")
        indexed[ref] = call
        identities.add(identity)
    return indexed


def l1_call_refs_by_id(execution: Mapping[str, object]) -> dict[str, str]:
    return {str(call["tool_call_id"]): ref for ref, call in l1_calls_by_ref(execution).items()}


def project_findings_arguments(arguments: dict, *, call_refs: Mapping[str, str]) -> dict:
    """展示已规范化的 findings 参数；只改协议来源位置，不改原始参数或任意正文。"""
    projected = deepcopy(arguments)
    for item in projected.get("items", []):
        if not isinstance(item, dict):
            continue
        if "source_refs" in item:
            item["source_refs"] = [
                {"call_ref": call_refs[source["tool_result_id"]], **(
                    {"chunk_id": source["chunk_id"]} if source.get("chunk_id") else {}
                )}
                for source in item["source_refs"]
            ]
    return projected




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
    "L1_CALL_REF_PATTERN",
    "format_l1_call_ref",
    "parse_l1_call_ref",
    "l1_calls_by_ref",
    "l1_call_refs_by_id",
    "project_findings_arguments",
]
