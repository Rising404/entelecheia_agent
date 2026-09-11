"""一个 TaskNode WorkRun 轮次经验证的宿主请求契约。

这些不可变值描述新建、恢复、复原或等待用户的 WorkRun 权威。本模块是语义
契约持有者，而非冷导入模块。
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ...work_run import TaskNodeSubject
from ...work_run.pending_user_question_contracts import PendingUserQuestion
from ..paper_prompt_context import PaperAttemptContext
from ..task_node.input_limits import (
    AttemptDecisionInputLimits,
    NodeVerificationInputLimits,
)


class _ApplicationContract(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        arbitrary_types_allowed=True,
    )


class WorkRunTurnApplicationRequest(_ApplicationContract):
    """一次新建、同轮次 TaskNode 执行的宿主权威。"""

    session_id: str = Field(min_length=1, max_length=200)
    turn_id: str = Field(min_length=1, max_length=200)
    subject: TaskNodeSubject
    expected_task_state_version: int | None = Field(default=None, ge=1)
    expected_node_state_version: int | None = Field(default=None, ge=1)
    expected_window_revision: int = Field(ge=1)
    attempt_input_limits: AttemptDecisionInputLimits
    verification_input_limits: NodeVerificationInputLimits
    allow_user_input: bool = True
    paper_resources: PaperAttemptContext | None = None

    @model_validator(mode="after")
    def _bind_paper_resources(self) -> 'WorkRunTurnApplicationRequest':
        _validate_task_paper_resources(
            paper_resources=self.paper_resources,
            session_id=self.session_id,
            task_id=self.subject.task_id,
        )
        return self


class WorkRunTurnResumeRequest(_ApplicationContract):
    """新轮次中一个尚未决定的活动 Attempt 所对应的宿主权威。"""

    session_id: str = Field(min_length=1, max_length=200)
    turn_id: str = Field(min_length=1, max_length=200)
    subject: TaskNodeSubject
    work_run_id: str = Field(min_length=1, max_length=200)
    attempt_id: str = Field(min_length=1, max_length=200)
    expected_work_run_revision: int = Field(ge=1)
    expected_window_revision: int = Field(ge=1)
    attempt_input_limits: AttemptDecisionInputLimits
    verification_input_limits: NodeVerificationInputLimits
    allow_user_input: bool = True
    paper_resources: PaperAttemptContext | None = None

    @model_validator(mode="after")
    def _bind_paper_resources(self) -> 'WorkRunTurnResumeRequest':
        _validate_task_paper_resources(
            paper_resources=self.paper_resources,
            session_id=self.session_id,
            task_id=self.subject.task_id,
        )
        return self


class WorkRunTurnVerificationRecoveryRequest(_ApplicationContract):
    """一次待处理或已中断验证的跨轮次权威。

    恢复操作会重新绑定同一持久逻辑验证请求。已关闭的提交 Attempt 与已锁定的
    语义快照保持不变；只有调用轮次及请求/WorkRun 代次会推进。
    """

    session_id: str = Field(min_length=1, max_length=200)
    turn_id: str = Field(min_length=1, max_length=200)
    subject: TaskNodeSubject
    work_run_id: str = Field(min_length=1, max_length=200)
    submitted_attempt_id: str = Field(min_length=1, max_length=200)
    verification_request_id: str = Field(min_length=1, max_length=200)
    expected_work_run_revision: int = Field(ge=1)
    expected_verification_request_revision: int = Field(ge=1)
    expected_window_revision: int = Field(ge=1)
    attempt_input_limits: AttemptDecisionInputLimits
    verification_input_limits: NodeVerificationInputLimits
    allow_user_input: bool = True
    paper_resources: PaperAttemptContext | None = None

    @model_validator(mode="after")
    def _bind_paper_resources(
        self,
    ) -> 'WorkRunTurnVerificationRecoveryRequest':
        _validate_task_paper_resources(
            paper_resources=self.paper_resources,
            session_id=self.session_id,
            task_id=self.subject.task_id,
        )
        return self


class WorkRunTurnUnpreparedVerificationRecoveryRequest(_ApplicationContract):
    """面向准备操作从未提交之已提交输出的跨轮次权威。"""

    session_id: str = Field(min_length=1, max_length=200)
    turn_id: str = Field(min_length=1, max_length=200)
    subject: TaskNodeSubject
    work_run_id: str = Field(min_length=1, max_length=200)
    submitted_attempt_id: str = Field(min_length=1, max_length=200)
    verification_request_id: str = Field(min_length=1, max_length=200)
    expected_work_run_revision: int = Field(ge=1)
    expected_progress_revision: int = Field(ge=1)
    expected_output_revision: int = Field(ge=1)
    expected_window_revision: int = Field(ge=1)
    attempt_input_limits: AttemptDecisionInputLimits
    verification_input_limits: NodeVerificationInputLimits
    allow_user_input: bool = True
    paper_resources: PaperAttemptContext | None = None

    @model_validator(mode="after")
    def _bind_paper_resources(
        self,
    ) -> 'WorkRunTurnUnpreparedVerificationRecoveryRequest':
        _validate_task_paper_resources(
            paper_resources=self.paper_resources,
            session_id=self.session_id,
            task_id=self.subject.task_id,
        )
        return self


class WorkRunTurnWaitingUserContinuationRequest(_ApplicationContract):
    """精确的待处理问题权威，以及新接纳的回答轮次。"""

    session_id: str = Field(min_length=1, max_length=200)
    turn_id: str = Field(min_length=1, max_length=200)
    pending_question: PendingUserQuestion
    expected_window_revision: int = Field(ge=1)
    attempt_input_limits: AttemptDecisionInputLimits
    verification_input_limits: NodeVerificationInputLimits
    allow_user_input: bool = True
    paper_resources: PaperAttemptContext | None = None

    @model_validator(mode="after")
    def _bind_pending_question(
        self,
    ) -> 'WorkRunTurnWaitingUserContinuationRequest':
        if self.pending_question.session_id != self.session_id:
            raise ValueError("pending question belongs to another Session")
        if self.pending_question.question_turn_id == self.turn_id:
            raise ValueError("answer Turn must follow the question Turn")
        _validate_task_paper_resources(
            paper_resources=self.paper_resources,
            session_id=self.session_id,
            task_id=self.pending_question.subject.task_id,
        )
        return self

    @property
    def subject(self) -> TaskNodeSubject:
        return self.pending_question.subject

    @property
    def work_run_id(self) -> str:
        return self.pending_question.work_run_id


WorkRunTurnRequest = (
    WorkRunTurnApplicationRequest
    | WorkRunTurnResumeRequest
    | WorkRunTurnVerificationRecoveryRequest
    | WorkRunTurnUnpreparedVerificationRecoveryRequest
    | WorkRunTurnWaitingUserContinuationRequest
)


def _validate_task_paper_resources(
    *,
    paper_resources: PaperAttemptContext | None,
    session_id: str,
    task_id: str,
) -> None:
    if paper_resources is None:
        return
    if paper_resources.session_id != session_id:
        raise ValueError("paper resources belong to another Session")
    if paper_resources.task_id != task_id:
        raise ValueError("paper resources belong to another Task")


__all__ = [
    'WorkRunTurnApplicationRequest',
    'WorkRunTurnRequest',
    'WorkRunTurnResumeRequest',
    'WorkRunTurnUnpreparedVerificationRecoveryRequest',
    'WorkRunTurnVerificationRecoveryRequest',
    'WorkRunTurnWaitingUserContinuationRequest',
]
