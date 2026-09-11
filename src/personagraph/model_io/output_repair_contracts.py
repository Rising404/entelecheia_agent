"""Provider-neutral contracts for durable structured-output repair.

These immutable DTOs freeze the exact structured prompt and bounded Host-written
repair feedback used across provider attempts.  Serialized schema literals and
field names are persistence identities; moving their Python owner must not alter
their wire representation.
"""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from typing import Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    field_validator,
    model_validator,
)


_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,199}$"
_SHA256_PATTERN = r"^[0-9a-f]{64}$"
_MAX_OUTPUT_REPAIR_REASON_UTF8_BYTES = 500
_MAX_OUTPUT_REPAIR_ISSUE_PATHS = 8
_MAX_OUTPUT_REPAIR_ISSUE_PATH_UTF8_BYTES = 2_000
_MAX_OUTPUT_REPAIR_ISSUE_UTF8_BYTES = 16_000
_MAX_OUTPUT_REPAIR_ISSUES = 64
_MAX_OUTPUT_REPAIR_FEEDBACK_UTF8_BYTES = 64_000
RUNTIME_MODEL_REJECTED_OUTPUT_MAX_UTF8_BYTES = 1_500_000
RUNTIME_MODEL_STRUCTURED_PROMPT_COMPONENT_MAX_UTF8_BYTES = 1_500_000


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


class _OutputRepairContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class RuntimeModelOutputRepairIssueCategory(StrEnum):
    JSON_SYNTAX = "json_syntax"
    SCHEMA = "schema"
    HOST_GUARD = "host_guard"


class RuntimeModelOutputRepairIssueCoverage(StrEnum):
    COMPLETE = "complete"
    FIRST_ONLY = "first_only"
    PARTIAL = "partial"
    TRUNCATED = "truncated"


class RuntimeModelOutputRepairProtocol(StrEnum):
    """在一个逻辑调用生命周期内冻结的消息契约。"""

    FOUR_MESSAGE_WHOLE_RESPONSE_REGENERATION = (
        "four-message-whole-response-regeneration-v1"
    )


class RuntimeModelStructuredPrompt(_OutputRepairContract):
    """为持久化结构化重试冻结的前两条精确消息。"""

    schema_version: Literal["runtime-model-structured-prompt-v1"] = (
        "runtime-model-structured-prompt-v1"
    )
    system_prompt: str = Field(min_length=1)
    user_content: str = Field(min_length=1)
    system_prompt_sha256: str = Field(pattern=_SHA256_PATTERN)
    user_content_sha256: str = Field(pattern=_SHA256_PATTERN)

    @model_validator(mode="after")
    def _validate_prompt(self) -> 'RuntimeModelStructuredPrompt':
        for label, value, expected_sha256 in (
            ("system prompt", self.system_prompt, self.system_prompt_sha256),
            ("user content", self.user_content, self.user_content_sha256),
        ):
            if "\x00" in value:
                raise ValueError(f"structured {label} must not contain NUL")
            if (
                len(value.encode("utf-8"))
                > RUNTIME_MODEL_STRUCTURED_PROMPT_COMPONENT_MAX_UTF8_BYTES
            ):
                raise ValueError(f"structured {label} exceeds its UTF-8 limit")
            if _sha256_text(value) != expected_sha256:
                raise ValueError(f"structured {label} hash does not match")
        return self

    @classmethod
    def create(
        cls,
        *,
        system_prompt: str,
        user_content: str,
    ) -> Self:
        return cls(
            system_prompt=system_prompt,
            user_content=user_content,
            system_prompt_sha256=_sha256_text(system_prompt),
            user_content_sha256=_sha256_text(user_content),
        )


class RuntimeModelOutputRepairIssue(_OutputRepairContract):
    """被拒结构化响应中一个由 Host 撰写的有界问题。

    ``paths`` 包含指向被拒响应的规范 JSON Pointer。未知模型创建键必须由 producer
    投影到已知父路径，而不能复制到此契约中。
    """

    schema_version: Literal["runtime-model-output-repair-issue-v2"] = (
        "runtime-model-output-repair-issue-v2"
    )
    category: RuntimeModelOutputRepairIssueCategory
    code: str = Field(pattern=_ID_PATTERN)
    paths: tuple[str, ...] = Field(
        min_length=1,
        max_length=_MAX_OUTPUT_REPAIR_ISSUE_PATHS,
    )
    json_line: int | None = Field(default=None, ge=1, le=100_000_000)
    json_column: int | None = Field(default=None, ge=1, le=100_000_000)
    safe_explanation: str = Field(min_length=1, max_length=500)

    @field_validator("paths")
    @classmethod
    def _validate_paths(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if value != tuple(sorted(set(value))):
            raise ValueError("output-repair paths must use unique stable order")
        for path in value:
            if not _is_canonical_json_pointer(path):
                raise ValueError(
                    "output-repair paths must use canonical JSON Pointer syntax"
                )
            if (
                "\x00" in path
                or "\n" in path
                or "\r" in path
                or len(path.encode("utf-8"))
                > _MAX_OUTPUT_REPAIR_ISSUE_PATH_UTF8_BYTES
            ):
                raise ValueError(
                    "output-repair JSON Pointer exceeds its canonical byte limit"
                )
        return value

    @field_validator("safe_explanation")
    @classmethod
    def _validate_safe_explanation(cls, value: str) -> str:
        if (
            value != value.strip()
            or "\x00" in value
            or "\n" in value
            or "\r" in value
            or len(value.encode("utf-8"))
            > _MAX_OUTPUT_REPAIR_REASON_UTF8_BYTES
        ):
            raise ValueError(
                "output-repair safe_explanation must be canonical single-line "
                "text within 500 UTF-8 bytes"
            )
        return value

    @model_validator(mode="after")
    def _validate_issue(self) -> "RuntimeModelOutputRepairIssue":
        if (self.json_line is None) != (self.json_column is None):
            raise ValueError(
                "output-repair JSON line and column must be supplied together"
            )
        if (
            self.category is not RuntimeModelOutputRepairIssueCategory.JSON_SYNTAX
            and (self.json_line is not None or self.json_column is not None)
        ):
            raise ValueError(
                "only a json_syntax issue may carry JSON line and column"
            )
        if len(
            _canonical_json(self.model_dump(mode="json")).encode("utf-8")
        ) > _MAX_OUTPUT_REPAIR_ISSUE_UTF8_BYTES:
            raise ValueError("output-repair issue exceeds its UTF-8 byte limit")
        return self


class RuntimeModelOutputRepairFeedback(_OutputRepairContract):
    """一次拒绝后用于完整响应重新生成的 envelope。

    被拒响应正文有意缺席。持久化存储将该正文保留在自身内容寻址记录中，并通过
    ``rejected_response_sha256`` 绑定。
    """

    schema_version: Literal["runtime-model-output-repair-feedback-v2"] = (
        "runtime-model-output-repair-feedback-v2"
    )
    message_contract: Literal[
        "four-message-whole-response-regeneration-v1"
    ] = "four-message-whole-response-regeneration-v1"
    target_contract: str = Field(pattern=_ID_PATTERN)
    rejected_physical_ordinal: int = Field(ge=1, le=32)
    rejected_response_sha256: str = Field(pattern=_SHA256_PATTERN)
    repair_mode: Literal["regenerate_complete_response"] = (
        "regenerate_complete_response"
    )
    issue_coverage: RuntimeModelOutputRepairIssueCoverage
    omitted_issue_count: int = Field(ge=0, le=1_000_000_000)
    current_issues: tuple[RuntimeModelOutputRepairIssue, ...] = Field(
        min_length=1,
        max_length=_MAX_OUTPUT_REPAIR_ISSUES,
    )

    @model_validator(mode="after")
    def _validate_feedback(self) -> "RuntimeModelOutputRepairFeedback":
        if (
            self.issue_coverage
            is RuntimeModelOutputRepairIssueCoverage.TRUNCATED
        ) != (self.omitted_issue_count > 0):
            raise ValueError(
                "truncated output-repair feedback requires a positive "
                "omitted_issue_count; other coverage values require zero"
            )
        ordered = tuple(
            sorted(self.current_issues, key=runtime_model_output_repair_issue_sort_key)
        )
        if self.current_issues != ordered or len(
            {runtime_model_output_repair_issue_sort_key(issue) for issue in ordered}
        ) != len(ordered):
            raise ValueError(
                "output-repair current_issues must use unique stable order"
            )
        if len(
            _canonical_json(self.model_dump(mode="json")).encode("utf-8")
        ) > _MAX_OUTPUT_REPAIR_FEEDBACK_UTF8_BYTES:
            raise ValueError("output-repair feedback exceeds its UTF-8 byte limit")
        return self


_OUTPUT_REPAIR_FEEDBACK_ADAPTER: TypeAdapter[
    RuntimeModelOutputRepairFeedback
] = TypeAdapter(RuntimeModelOutputRepairFeedback)


def runtime_model_output_repair_issue_sort_key(
    issue: RuntimeModelOutputRepairIssue,
) -> tuple[object, ...]:
    """Return the canonical issue order shared by producers and validation."""

    category_order = {
        RuntimeModelOutputRepairIssueCategory.JSON_SYNTAX: 0,
        RuntimeModelOutputRepairIssueCategory.SCHEMA: 1,
        RuntimeModelOutputRepairIssueCategory.HOST_GUARD: 2,
    }
    return (
        category_order[issue.category],
        issue.code,
        issue.paths,
        -1 if issue.json_line is None else issue.json_line,
        -1 if issue.json_column is None else issue.json_column,
        issue.safe_explanation,
    )


def _validate_runtime_model_output_repair_feedback(
    value: object,
) -> RuntimeModelOutputRepairFeedback:
    return _OUTPUT_REPAIR_FEEDBACK_ADAPTER.validate_python(value)


def _is_canonical_json_pointer(value: str) -> bool:
    if value == "":
        return True
    if not value.startswith("/"):
        return False
    index = 0
    while index < len(value):
        if value[index] != "~":
            index += 1
            continue
        if index + 1 >= len(value) or value[index + 1] not in {"0", "1"}:
            return False
        index += 2
    return True


__all__ = [
    "RUNTIME_MODEL_REJECTED_OUTPUT_MAX_UTF8_BYTES",
    "RUNTIME_MODEL_STRUCTURED_PROMPT_COMPONENT_MAX_UTF8_BYTES",
    "RuntimeModelOutputRepairFeedback",
    "RuntimeModelOutputRepairIssue",
    "RuntimeModelOutputRepairIssueCategory",
    "RuntimeModelOutputRepairIssueCoverage",
    "RuntimeModelOutputRepairProtocol",
    'RuntimeModelStructuredPrompt',
    "runtime_model_output_repair_issue_sort_key",
]
