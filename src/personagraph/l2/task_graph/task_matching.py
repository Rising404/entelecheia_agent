"""L2 TaskGraph 的类型化入口任务匹配提案及其确定性 Host 守卫。

入口模型提出当前用户输入与 Session 持有的根任务之间的关系。提案并非权威信息：
Host 会将每个提案绑定到权威用户文本中精确且唯一的片段，并且只检查其可在本地证明的事实。
自然语言意图、任务分组以及摘要在语义上是否良好，均特意不属于此守卫的职责范围。
"""

from __future__ import annotations

from enum import StrEnum
from hashlib import sha256
from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .contracts import InSessionTaskCatalog


_LOCAL_KEY_PATTERN = r"^[a-z][a-z0-9_-]{0,63}$"


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class NewRootTaskMatchProposal(_Contract):
    """提出一个新根 Task；其持久 Task ID 之后由 Host 分配。"""

    match_type: Literal["new_root"] = "new_root"
    local_key: str = Field(pattern=_LOCAL_KEY_PATTERN)
    title: str = Field(min_length=1, max_length=240)
    objective: str = Field(min_length=1, max_length=2_000)
    source_excerpt: str = Field(min_length=1, max_length=4_000)


class ExistingRootTaskMatchProposal(_Contract):
    """提出与此 Session 中现有根 Task 的根级关联。"""

    match_type: Literal["existing_root"] = "existing_root"
    insession_task_id: str = Field(min_length=1, max_length=128)
    source_excerpt: str = Field(min_length=1, max_length=4_000)
    execute_current: bool = False


class ExistingRootBranchTaskMatchProposal(_Contract):
    """在现有根任务下提出分支意图，但不创建节点。"""

    match_type: Literal["existing_root_branch"] = "existing_root_branch"
    insession_task_id: str = Field(min_length=1, max_length=128)
    branch_key: str = Field(pattern=_LOCAL_KEY_PATTERN)
    branch_summary: str = Field(min_length=1, max_length=2_000)
    source_excerpt: str = Field(min_length=1, max_length=4_000)
    execute_current: bool = False


class ExistingRootTargetChangeTaskMatchProposal(_Contract):
    """提出一个由用户明确给出的根目标替代项。"""

    match_type: Literal["existing_root_target_change"] = (
        "existing_root_target_change"
    )
    insession_task_id: str = Field(min_length=1, max_length=128)
    replacement_objective: str = Field(min_length=1, max_length=2_000)
    source_excerpt: str = Field(min_length=1, max_length=4_000)
    execute_current: Literal[True]


InSessionTaskMatchProposal: TypeAlias = Annotated[
    NewRootTaskMatchProposal
    | ExistingRootTaskMatchProposal
    | ExistingRootBranchTaskMatchProposal
    | ExistingRootTargetChangeTaskMatchProposal,
    Field(discriminator="match_type"),
]


class InSessionTaskMatchesProposal(_Contract):
    """一批入口模型提案；空批次是有效的 L0 结果。"""

    schema_version: Literal["insession-task-matches-v1"] = "insession-task-matches-v1"
    task_matches: tuple[InSessionTaskMatchProposal, ...] = Field(
        default=(), max_length=24
    )


class InSessionTaskMatchingLimits(_Contract):
    """由 Host 持有、独立于模型所写提案的限制。"""

    max_new_root_tasks: int = Field(default=3, ge=0, le=24)


class InSessionTaskMatchGuardCode(StrEnum):
    """任务匹配提案被拒绝时稳定且确定性的原因。"""

    DUPLICATE_CATALOG_TASK_ID = "duplicate_catalog_task_id"
    DUPLICATE_LOCAL_KEY = "duplicate_local_key"
    DUPLICATE_TASK_MATCH = "duplicate_task_match"
    NEW_ROOT_LIMIT_EXCEEDED = "new_root_limit_exceeded"
    BLANK_TITLE = "blank_title"
    BLANK_OBJECTIVE = "blank_objective"
    BLANK_BRANCH_SUMMARY = "blank_branch_summary"
    BLANK_INSESSION_TASK_ID = "blank_insession_task_id"
    BLANK_SOURCE_EXCERPT = "blank_source_excerpt"
    UNKNOWN_INSESSION_TASK_ID = "unknown_insession_task_id"
    SOURCE_EXCERPT_NOT_FOUND = "source_excerpt_not_found"
    SOURCE_EXCERPT_AMBIGUOUS = "source_excerpt_ambiguous"


class InSessionTaskMatchedSourceSpan(_Contract):
    """Host 针对精确摘录推导出的 Python/Unicode 偏移量与摘要。"""

    start: int = Field(ge=0)
    end: int = Field(ge=1)
    text_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def _validate_range(self) -> 'InSessionTaskMatchedSourceSpan':
        if self.end <= self.start:
            raise ValueError("matched source span end must exceed start")
        return self


class AcceptedInSessionTaskMatch(_Contract):
    """与 Host 推导出的权威来源片段配对的模型提案。"""

    proposal: InSessionTaskMatchProposal
    source_span: InSessionTaskMatchedSourceSpan


class InSessionTaskMatchGuardResult(_Contract):
    """确定性任务匹配守卫返回的全有或全无结果。"""

    status: Literal["accepted", "rejected"]
    accepted_task_matches: tuple[AcceptedInSessionTaskMatch, ...] = ()
    error_codes: tuple[InSessionTaskMatchGuardCode, ...] = ()

    @field_validator("error_codes")
    @classmethod
    def _require_unique_error_codes(
        cls, values: tuple[InSessionTaskMatchGuardCode, ...]
    ) -> tuple[InSessionTaskMatchGuardCode, ...]:
        if len(values) != len(set(values)):
            raise ValueError("task-match guard error codes must be unique")
        return values

    @model_validator(mode="after")
    def _validate_status_shape(self) -> 'InSessionTaskMatchGuardResult':
        if self.status == "accepted" and self.error_codes:
            raise ValueError("accepted task matches cannot carry error codes")
        if self.status == "rejected" and (
            self.accepted_task_matches or not self.error_codes
        ):
            raise ValueError(
                "rejected task matches require errors and cannot carry accepted matches"
            )
        return self


class InSessionTaskMatchApplyResult(_Contract):
    """一次原子入口任务匹配应用或精确重放的持久结果。"""

    status: Literal["applied", "replayed"]
    created_insession_task_ids_by_local_key: dict[str, str] = Field(
        default_factory=dict,
        max_length=24,
    )
    related_insession_task_ids: tuple[str, ...] = Field(default=(), max_length=48)
    branch_intent_ids: tuple[str, ...] = Field(default=(), max_length=24)
    turn_task_link_revision: int = Field(ge=0)
    window_state_version: int | None = Field(default=None, ge=0)

    @field_validator("related_insession_task_ids", "branch_intent_ids")
    @classmethod
    def _require_unique_ids(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if len(values) != len(set(values)):
            raise ValueError("task-match apply result ids must be unique")
        return values


def guard_insession_task_matches(
    proposal: InSessionTaskMatchesProposal,
    *,
    authoritative_user_text: str,
    trusted_root_catalog: InSessionTaskCatalog,
    limits: InSessionTaskMatchingLimits | None = None,
) -> InSessionTaskMatchGuardResult:
    """只验证精确来源、标识符、唯一性和限制。

    特别地，此函数不判断提议的标题、目标、分支摘要或任务分组是否与其来源摘录语义相同。
    这种脚本级语义检查并不可靠。
    """

    effective_limits = limits or InSessionTaskMatchingLimits()
    errors: set[InSessionTaskMatchGuardCode] = set()
    accepted_matches: list[AcceptedInSessionTaskMatch] = []

    catalog_ids = [item.insession_task_id for item in trusted_root_catalog.items]
    if len(catalog_ids) != len(set(catalog_ids)):
        errors.add(InSessionTaskMatchGuardCode.DUPLICATE_CATALOG_TASK_ID)
    known_task_ids = set(catalog_ids)

    new_root_count = sum(
        match.match_type == "new_root" for match in proposal.task_matches
    )
    if new_root_count > effective_limits.max_new_root_tasks:
        errors.add(InSessionTaskMatchGuardCode.NEW_ROOT_LIMIT_EXCEEDED)

    local_keys: set[str] = set()
    match_identities: set[tuple[object, ...]] = set()
    for match in proposal.task_matches:
        identity = _match_identity(match)
        if identity in match_identities:
            errors.add(InSessionTaskMatchGuardCode.DUPLICATE_TASK_MATCH)
        match_identities.add(identity)

        local_key = _local_key(match)
        if local_key is not None:
            if local_key in local_keys:
                errors.add(InSessionTaskMatchGuardCode.DUPLICATE_LOCAL_KEY)
            local_keys.add(local_key)

        _validate_nonblank_fields(match, errors)
        if match.match_type in {
            "existing_root",
            "existing_root_branch",
            "existing_root_target_change",
        }:
            if (
                match.insession_task_id.strip()
                and match.insession_task_id not in known_task_ids
            ):
                errors.add(InSessionTaskMatchGuardCode.UNKNOWN_INSESSION_TASK_ID)

        source_span = _locate_exact_source_span(
            authoritative_user_text,
            match.source_excerpt,
            errors,
        )
        if source_span is not None:
            accepted_matches.append(
                AcceptedInSessionTaskMatch(
                    proposal=match,
                    source_span=source_span,
                )
            )

    ordered_errors = tuple(sorted(errors, key=str))
    if ordered_errors:
        return InSessionTaskMatchGuardResult(
            status="rejected",
            error_codes=ordered_errors,
        )
    return InSessionTaskMatchGuardResult(
        status="accepted",
        accepted_task_matches=tuple(accepted_matches),
    )


def _validate_nonblank_fields(
    match: InSessionTaskMatchProposal,
    errors: set[InSessionTaskMatchGuardCode],
) -> None:
    if not match.source_excerpt.strip():
        errors.add(InSessionTaskMatchGuardCode.BLANK_SOURCE_EXCERPT)
    if match.match_type == "new_root":
        if not match.title.strip():
            errors.add(InSessionTaskMatchGuardCode.BLANK_TITLE)
        if not match.objective.strip():
            errors.add(InSessionTaskMatchGuardCode.BLANK_OBJECTIVE)
    elif match.match_type == "existing_root_branch":
        if not match.insession_task_id.strip():
            errors.add(InSessionTaskMatchGuardCode.BLANK_INSESSION_TASK_ID)
        if not match.branch_summary.strip():
            errors.add(InSessionTaskMatchGuardCode.BLANK_BRANCH_SUMMARY)
    elif match.match_type == "existing_root_target_change":
        if not match.insession_task_id.strip():
            errors.add(InSessionTaskMatchGuardCode.BLANK_INSESSION_TASK_ID)
        if not match.replacement_objective.strip():
            errors.add(InSessionTaskMatchGuardCode.BLANK_OBJECTIVE)
    elif not match.insession_task_id.strip():
        errors.add(InSessionTaskMatchGuardCode.BLANK_INSESSION_TASK_ID)


def _locate_exact_source_span(
    authoritative_user_text: str,
    source_excerpt: str,
    errors: set[InSessionTaskMatchGuardCode],
) -> InSessionTaskMatchedSourceSpan | None:
    if not source_excerpt.strip():
        return None
    start = authoritative_user_text.find(source_excerpt)
    if start < 0:
        errors.add(InSessionTaskMatchGuardCode.SOURCE_EXCERPT_NOT_FOUND)
        return None
    if authoritative_user_text.find(source_excerpt, start + 1) >= 0:
        errors.add(InSessionTaskMatchGuardCode.SOURCE_EXCERPT_AMBIGUOUS)
        return None
    end = start + len(source_excerpt)
    return InSessionTaskMatchedSourceSpan(
        start=start,
        end=end,
        text_sha256=sha256(source_excerpt.encode("utf-8")).hexdigest(),
    )


def _local_key(match: InSessionTaskMatchProposal) -> str | None:
    if match.match_type == "new_root":
        return match.local_key
    if match.match_type == "existing_root_branch":
        return match.branch_key
    return None


def _match_identity(match: InSessionTaskMatchProposal) -> tuple[object, ...]:
    if match.match_type == "new_root":
        return (
            match.match_type,
            match.title,
            match.objective,
            match.source_excerpt,
        )
    if match.match_type == "existing_root_branch":
        return (
            match.match_type,
            match.insession_task_id,
            match.branch_summary,
            match.source_excerpt,
        )
    if match.match_type == "existing_root_target_change":
        return (
            match.match_type,
            match.insession_task_id,
            match.replacement_objective,
            match.source_excerpt,
        )
    return (match.match_type, match.insession_task_id, match.source_excerpt)
