"""AuxiliaryGraph 的规范不可变绑定，供语义评审专家使用。

该绑定冻结了精确的提供者中立的语义评审请求，但在创建持久化模型调用授权之前不会读取 前沿、访问模型账本、调用提供者或结算语义评审共识。
"""

from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from personagraph.runtime.model_calls.policy import MAX_MODEL_ATTEMPTS


_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,199}$"
_SHA256_PATTERN = r"^[0-9a-f]{64}$"
_PURPOSE = "runtime_task_graph_semantic_verification"
_REQUEST_CONTRACT = "auxiliary-v2-semantic-reviewer-model-call-v1"
AUXILIARY_SEMANTIC_RESULT_CONTRACT = (
    "task-graph-semantic-verification-model-envelope-v1"
)


class _BindingContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class AuxiliarySemanticReviewerModelCallBinding(_BindingContract):
    """完整的 Host 绑定，用于一个持久化的语义评审者模型调用。"""

    schema_version: Literal[
        "auxiliary-v2-semantic-reviewer-model-call-binding-v1"
    ] = "auxiliary-v2-semantic-reviewer-model-call-binding-v1"
    logical_call_id: str = Field(pattern=_ID_PATTERN)
    verification_request_id: str = Field(pattern=_ID_PATTERN)
    verification_result_id: str = Field(pattern=_ID_PATTERN)
    reviewer_ordinal: int = Field(ge=1, le=2)
    required_reviewer_count: int = Field(ge=1, le=2)
    session_id: str = Field(pattern=_ID_PATTERN)
    task_id: str = Field(pattern=_ID_PATTERN)
    auxiliary_graph_id: str = Field(pattern=_ID_PATTERN)
    goal_id: str = Field(pattern=_ID_PATTERN)
    auxiliary_graph_revision: int = Field(ge=1)
    invocation_turn_id: str = Field(pattern=_ID_PATTERN)
    purpose: Literal["runtime_task_graph_semantic_verification"] = _PURPOSE
    request_contract: Literal[
        "auxiliary-v2-semantic-reviewer-model-call-v1"
    ] = _REQUEST_CONTRACT
    request_json: str
    request_sha256: str = Field(pattern=_SHA256_PATTERN)
    typed_result_contract: Literal[
        "task-graph-semantic-verification-model-envelope-v1"
    ] = AUXILIARY_SEMANTIC_RESULT_CONTRACT
    max_physical_attempts: Literal[MAX_MODEL_ATTEMPTS] = MAX_MODEL_ATTEMPTS
    state_guard_sha256: str = Field(pattern=_SHA256_PATTERN)

    @model_validator(mode="after")
    def _validate_request_json(self) -> "AuxiliarySemanticReviewerModelCallBinding":
        try:
            parsed = json.loads(self.request_json)
        except (TypeError, ValueError, RecursionError) as exc:
            raise ValueError("semantic model binding request is not JSON") from exc
        if (
            _canonical_json(parsed) != self.request_json
            or _sha256_text(self.request_json) != self.request_sha256
        ):
            raise ValueError("semantic model binding request is not canonical")
        return self


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_value(value: object) -> str:
    return _sha256_text(_canonical_json(value))


__all__ = [
    "AUXILIARY_SEMANTIC_RESULT_CONTRACT",
    "AuxiliarySemanticReviewerModelCallBinding",
]
