"""持久单 Session 对话摘要的纯合同与生成规则。

摘要是派生上下文辅助信息，不能替代权威对话、Task 或 WorkRun 事实。这些类型刻意不携带
SQLite、模型 provider 或 Runtime 依赖；本模块也只负责可信摘要输入、预算与更新提案，
不负责后台 job 或持久状态转换。
"""

from __future__ import annotations

from collections.abc import Callable
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..context_budget.token_counter import (
    block_cap,
    cap_text,
    estimate_tokens,
    task_budget,
)


SESSION_SUMMARY_JOB_KIND = "session_summary"
SESSION_SUMMARY_CANDIDATE_PAIR_LIMIT = 24
SESSION_SUMMARY_SYSTEM_PROMPT = """你负责维护会话的长期摘要。请把已有摘要与较早的完整对话合并为简洁、准确的中文事实记录。
只保留后续对话有用的用户目标、已知事实、决定、约束、未完成事项和明确承诺；不要编造、不要把模型猜测写成事实、不要复述寒暄。
这不是对用户的回复，不要称呼用户，也不要解释摘要过程。"""


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SessionSummaryStatus(StrEnum):
    """持久运行摘要的真实可用性。"""

    OK = "ok"
    STALE = "stale"
    UNAVAILABLE = "unavailable"


class SessionSummaryStateError(RuntimeError):
    """持久摘要状态无法安全满足请求的操作。"""


class SessionSummaryStateConflict(SessionSummaryStateError):
    """工作器尝试写入过期摘要状态。"""

    def __init__(
        self,
        *,
        expected_state_version: int,
        actual_state_version: int,
        expected_summarized_through_turn_id: str | None,
        actual_summarized_through_turn_id: str | None,
    ) -> None:
        self.expected_state_version = expected_state_version
        self.actual_state_version = actual_state_version
        self.expected_summarized_through_turn_id = expected_summarized_through_turn_id
        self.actual_summarized_through_turn_id = actual_summarized_through_turn_id
        super().__init__(
            "session summary state changed before the post-commit worker could apply its result"
        )


class SessionSummaryProgressError(SessionSummaryStateError):
    """提议的摘要边界不是安全的向前推进。"""


class SessionSummaryState(_Contract):
    """一个 Session 所有的持久摘要投影。

    ``summarized_through_turn_id`` 是稳定对话对身份，绝不是可变对话位置。
    ``state_version`` 是工作器返回结果时必须携带的乐观写入边界。
    """

    session_id: str = Field(min_length=1, max_length=200)
    running_summary: str = ""
    summarized_through_turn_id: str | None = None
    status: SessionSummaryStatus = SessionSummaryStatus.OK
    state_version: int = Field(ge=0)
    updated_at: str = Field(min_length=1)
    last_error_code: str | None = None

    @field_validator("summarized_through_turn_id", "last_error_code")
    @classmethod
    def _require_nonblank_optional_identifier(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not value.strip():
            raise ValueError("optional summary identifiers must not be blank")
        return value

    @model_validator(mode="after")
    def _validate_status_error_pair(self) -> 'SessionSummaryState':
        if self.status is SessionSummaryStatus.OK and self.last_error_code is not None:
            raise ValueError("an ok summary state cannot retain an error code")
        if self.status is not SessionSummaryStatus.OK and self.last_error_code is None:
            raise ValueError("a non-ok summary state requires an error code")
        if self.status is SessionSummaryStatus.OK:
            has_summary = bool(self.running_summary.strip())
            has_boundary = self.summarized_through_turn_id is not None
            if has_summary != has_boundary:
                raise ValueError("an ok summary state must bind its content to a turn boundary")
        return self


class SessionSummaryUpdate(_Contract):
    """一个成功生成的替换运行摘要。

    成功的空操作任务使用无边界的空摘要。任何非空摘要都必须标明它覆盖的精确最后 Turn
    对，使存储层可以拒绝倒退以及对受保护热历史的写入。
    """

    running_summary: str = Field(max_length=200_000)
    summarized_through_turn_id: str | None = None

    @field_validator("summarized_through_turn_id")
    @classmethod
    def _require_nonblank_boundary(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("summarized_through_turn_id must not be blank")
        return value

    @model_validator(mode="after")
    def _bind_summary_to_boundary(self) -> 'SessionSummaryUpdate':
        has_summary = bool(self.running_summary.strip())
        has_boundary = self.summarized_through_turn_id is not None
        if has_summary != has_boundary:
            raise ValueError(
                "a summary update must provide both running_summary and summarized_through_turn_id"
            )
        return self


class SessionSummaryTurnPair(_Contract):
    """可安全用作摘要输入的一个完整按时间排序对话对。

    较旧对话对可能早于 Runtime Turn 身份迁移。工作器重建初始摘要时可以使用其文本，
    但只有非 null 的 ``turn_id`` 才能成为新的持久摘要边界。
    """

    turn_id: str | None = None
    user_turn_idx: int = Field(ge=0)
    assistant_turn_idx: int = Field(ge=0)
    user_content: str
    assistant_content: str
    created_at: str = Field(min_length=1)

    @field_validator("turn_id")
    @classmethod
    def _require_nonblank_turn_id_when_present(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("turn_id must not be blank when present")
        return value

    @model_validator(mode="after")
    def _validate_pair_order(self) -> 'SessionSummaryTurnPair':
        if self.assistant_turn_idx <= self.user_turn_idx:
            raise ValueError("assistant turn index must follow the user turn index")
        return self


class SessionSummaryGenerationError(RuntimeError):
    """摘要输入或输出无法形成可信的持久更新。"""

    def __init__(self, code: str, message: str, *, retryable: bool) -> None:
        self.code = code
        self.retryable = retryable
        super().__init__(message)


SessionSummaryTextGenerator = Callable[
    [SessionSummaryState | None, tuple[SessionSummaryTurnPair, ...]],
    str,
]


def build_session_summary_update(
    *,
    state: SessionSummaryState | None,
    pairs: tuple[SessionSummaryTurnPair, ...],
    generate: SessionSummaryTextGenerator,
) -> SessionSummaryUpdate:
    """用最早的连续完整对话对构建下一份摘要更新。"""

    if not pairs:
        if state is not None and state.status is not SessionSummaryStatus.OK:
            # 不能仅因当前作业没有合格转录对，就悄然重新认证过时/不可用的摘要；
            # 此时不存在可用于修复的可信新鲜来源。
            raise SessionSummaryGenerationError(
                "SUMMARY_NO_FRESH_SOURCE_FOR_STALE_STATE",
                "No eligible transcript pair can repair the degraded session summary.",
                retryable=False,
            )
        return SessionSummaryUpdate(
            running_summary=state.running_summary if state is not None else "",
            summarized_through_turn_id=(
                state.summarized_through_turn_id if state is not None else None
            ),
        )
    bounded_pairs = _select_summary_pairs_within_input_budget(state, pairs)
    target_turn_id = bounded_pairs[-1].turn_id
    if target_turn_id is None:
        raise SessionSummaryGenerationError(
            "SUMMARY_BOUNDARY_UNAVAILABLE",
            "The summary source does not have a stable Runtime turn identity.",
            retryable=False,
        )
    summary = generate(state, bounded_pairs).strip()
    if not summary:
        raise SessionSummaryGenerationError(
            "SUMMARY_EMPTY_OUTPUT",
            "The summary model returned no usable content.",
            retryable=True,
        )
    return SessionSummaryUpdate(
        running_summary=cap_text(summary, block_cap("summary")),
        summarized_through_turn_id=target_turn_id,
    )


def _select_summary_pairs_within_input_budget(
    state: SessionSummaryState | None,
    pairs: tuple[SessionSummaryTurnPair, ...],
) -> tuple[SessionSummaryTurnPair, ...]:
    """返回能够放入本次模型请求的最早连续对话对。

    摘要边界表示模型确实看过截至该轮次的全部材料。因此，我们绝不截断过大的
    对话对，也不会越过无法放入的对话对。剩余部分留给后续持久作业；若单个
    对话对完全无法放入，则以封闭方式失败，而不会被错误认证为已摘要。
    """

    prompt_budget = max(
        1,
        task_budget() - estimate_tokens(SESSION_SUMMARY_SYSTEM_PROMPT),
    )
    selected: list[SessionSummaryTurnPair] = []
    for pair in pairs:
        candidate = tuple((*selected, pair))
        if estimate_tokens(render_session_summary_input(state, candidate)) > prompt_budget:
            break
        selected.append(pair)
    if not selected:
        raise SessionSummaryGenerationError(
            "SUMMARY_INPUT_OVER_BUDGET",
            "No complete transcript pair fits the session summary input budget.",
            retryable=False,
        )
    return tuple(selected)


def render_session_summary_input(
    state: SessionSummaryState | None,
    pairs: tuple[SessionSummaryTurnPair, ...],
) -> str:
    """渲染 provider-neutral 的可信摘要输入正文。"""

    sections: list[str] = []
    if (
        state is not None
        and state.status is SessionSummaryStatus.OK
        and state.running_summary.strip()
    ):
        sections.append(f"已有摘要（已验证）：\n{state.running_summary.strip()}")
    for pair in pairs:
        sections.append(
            "较早完整回合：\n"
            f"用户：{pair.user_content}\n"
            f"助手：{pair.assistant_content}"
        )
    return "\n\n".join(sections)


def mock_session_summary(
    state: SessionSummaryState | None,
    pairs: tuple[SessionSummaryTurnPair, ...],
) -> str:
    """使无 provider 路径保持确定性，且不重放原始对话。"""

    existing = (
        state.running_summary.strip()
        if state is not None and state.status is SessionSummaryStatus.OK
        else ""
    )
    newest = pairs[-1]
    turn_fact = f"用户此前提出：{newest.user_content.strip()}"
    return "\n".join(item for item in (existing, turn_fact) if item)


__all__ = [
    "SESSION_SUMMARY_CANDIDATE_PAIR_LIMIT",
    "SESSION_SUMMARY_JOB_KIND",
    "SESSION_SUMMARY_SYSTEM_PROMPT",
    "SessionSummaryGenerationError",
    "SessionSummaryProgressError",
    "SessionSummaryStateConflict",
    "SessionSummaryStateError",
    'SessionSummaryState',
    "SessionSummaryStatus",
    "SessionSummaryTextGenerator",
    'SessionSummaryTurnPair',
    'SessionSummaryUpdate',
    "build_session_summary_update",
    "mock_session_summary",
    "render_session_summary_input",
]
