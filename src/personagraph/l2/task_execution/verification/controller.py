"""一个持久化 TaskNode 验证请求的应用 controller。

Store 负责请求创建、恢复、精确上下文组装与每次状态转换。本模块只负责三阶段应用流程::

    Store prepare -> SQLite 外部语义 verifier -> Store commit

Provider、有类型输出或 verifier 模块失败会走独立 Store interrupt 路径。因进程丢失
搁浅的 pending 请求与技术中断请求，都是同一 WorkRun、验证请求、已关闭 submit
Attempt 和锁定 OutputWindow 的跨 Turn 可恢复权威状态；此 controller 对两类失败均
绝不创建替代请求。非 pass 语义结果会在独立 Store 调用中启动下一 Attempt，而通过
结果有意不作为用户可见发布。所属 Turn orchestrator 负责在使用显式恢复入口前结算
已放弃 Turn；Turn handoff/finalization 有意位于此切片之外。

狭窄的注入 Store port 使应用流程可测试，而 SQLite facade 负责其事务。具体 adapter
只能导入 ``personagraph.session.store``——Runtime 代码绝不能越过边界访问
``session.persistence`` 实现模块。
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ....context_budget import ContextBudgetExceeded
from ....model_io.gateway import ModelGatewayError
from ....session.l2_store import task_delivery as task_delivery_store
from ....session.l2_store import task_graph as task_graph_store
from ....session.l2_store import verification as verification_store
from ....session.l2_store import work_run as work_run_store
from ....session.l2_store.verification import VerificationSettlementReceiptNotFound
from ...work_run import (
    DownstreamVerificationDisposition,
    DownstreamVerificationFeedback,
    PreparedTaskNodeVerification,
    ResolvedCurrentTaskNodeDelivery,
    TaskNodeSubject,
    TaskNodeVerificationMutationResult,
    TaskNodeVerificationRecord,
    TaskNodeVerificationRequestStatus,
    WorkExecutionMutationResult,
    WorkRunBudgetDisposition,
    WorkRunBudgetTransition,
    WorkRunStatus,
)
from ....model_io.output_validation import ModelOutputValidationError
from ....runtime.model_calls.requests import ModelRequestResult
from ....runtime.turn_deadline import TurnDeadline
from .decision import (
    NodeVerificationContext,
    NodeVerificationInputLimits,
    NodeVerificationInputTooLarge,
    NodeVerificationInputUnsupported,
    NodeVerificationResult,
    NodeVerificationStructuredProvider,
    SupportingToolResults,
    request_node_verification,
)
from .active_time import (
    NodeVerificationActiveTimeMeasurementError,
    NodeVerificationActiveTimeMeter,
)
from ..task_node.dependencies import project_task_node_dependencies
from ..task_node.dependency_delivery_contracts import (
    TaskNodeDependencyDeliveries,
    TaskNodeDependencyInputTooLarge,
    TaskNodeDependencyInputUnsupported,
)
from ..task_node.source_context import (
    TaskNodeSourceContext,
    build_task_node_source_context,
)
from ..task_node.frontier import build_task_node_tree
from ....runtime.turn_events import TurnEvent


NodeVerificationOutcome = Literal[
    "passed",
    "not_passed",
    "interrupted",
    "work_run_limit_reached",
]
NodeVerificationInterruptionReason = Literal[
    "verification_input_too_large",
    "verification_input_unsupported",
    "verification_unavailable",
    "verification_output_invalid",
    "context_budget_exceeded",
    "runtime_module_error",
    "work_run_limit_reached",
]


class _ControllerContract(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        arbitrary_types_allowed=True,
    )


class NodeVerificationControllerStateConflict(RuntimeError):
    """某个 Store/verifier 投影越过了精确权威绑定。"""


class NodeVerificationNextAttemptPlan(_ControllerContract):
    """仅在非 pass 语义结果后使用的稳定 Host 计划。"""

    attempt_id: str = Field(min_length=1, max_length=200)
    apply_id: str = Field(min_length=1, max_length=200)
    catalog_snapshot: dict[str, object]


class NodeVerificationApplicationRequest(_ControllerContract):
    """用于一次已提交 OutputWindow 验证的稳定 Host 命令。

    不接受应用调用方提供的节点标题、目标、Acceptance 文本、OutputWindow 正文或证据。
    Store 必须在 ``prepare`` 期间从当前持久化权威状态组装该语义上下文。
    """

    session_id: str = Field(min_length=1, max_length=200)
    turn_id: str = Field(min_length=1, max_length=200)
    work_run_id: str = Field(min_length=1, max_length=200)
    submitted_attempt_id: str = Field(min_length=1, max_length=200)
    verification_request_id: str = Field(min_length=1, max_length=200)
    expected_work_run_revision: int = Field(ge=1)
    expected_progress_revision: int = Field(ge=1)
    expected_output_revision: int = Field(ge=1)
    expected_window_revision: int = Field(ge=1)
    prepare_apply_id: str = Field(min_length=1, max_length=200)
    commit_apply_id: str = Field(min_length=1, max_length=200)
    interrupt_apply_id: str = Field(min_length=1, max_length=200)
    delivery_id: str = Field(min_length=1, max_length=200)
    input_limits: NodeVerificationInputLimits
    next_attempt: NodeVerificationNextAttemptPlan
    recover_unprepared_submit: bool = False


class NodeVerificationResumeRequest(_ControllerContract):
    """用于一个 pending/interrupted 请求的显式跨 Turn 恢复命令。"""

    session_id: str = Field(min_length=1, max_length=200)
    turn_id: str = Field(min_length=1, max_length=200)
    work_run_id: str = Field(min_length=1, max_length=200)
    submitted_attempt_id: str = Field(min_length=1, max_length=200)
    verification_request_id: str = Field(min_length=1, max_length=200)
    expected_work_run_revision: int = Field(ge=1)
    expected_verification_request_revision: int = Field(ge=1)
    expected_window_revision: int = Field(ge=1)
    resume_apply_id: str = Field(min_length=1, max_length=200)
    commit_apply_id: str = Field(min_length=1, max_length=200)
    interrupt_apply_id: str = Field(min_length=1, max_length=200)
    delivery_id: str = Field(min_length=1, max_length=200)
    input_limits: NodeVerificationInputLimits
    next_attempt: NodeVerificationNextAttemptPlan


class NodeVerificationPrepareStoreCommand(_ControllerContract):
    """狭窄 initial-prepare 命令；不能携带未来 Catalog。"""

    session_id: str = Field(min_length=1, max_length=200)
    turn_id: str = Field(min_length=1, max_length=200)
    work_run_id: str = Field(min_length=1, max_length=200)
    submitted_attempt_id: str = Field(min_length=1, max_length=200)
    verification_request_id: str = Field(min_length=1, max_length=200)
    expected_work_run_revision: int = Field(ge=1)
    expected_progress_revision: int = Field(ge=1)
    expected_output_revision: int = Field(ge=1)
    expected_window_revision: int = Field(ge=1)
    apply_id: str = Field(min_length=1, max_length=200)
    commit_settlement_apply_id: str = Field(min_length=1, max_length=200)
    interrupt_settlement_apply_id: str = Field(min_length=1, max_length=200)
    delivery_id: str = Field(min_length=1, max_length=200)
    input_limits: NodeVerificationInputLimits
    recover_unprepared_submit: bool = False


class NodeVerificationResumeStoreCommand(_ControllerContract):
    """用于 pending/interrupted 请求的狭窄显式恢复命令。"""

    session_id: str = Field(min_length=1, max_length=200)
    turn_id: str = Field(min_length=1, max_length=200)
    work_run_id: str = Field(min_length=1, max_length=200)
    submitted_attempt_id: str = Field(min_length=1, max_length=200)
    verification_request_id: str = Field(min_length=1, max_length=200)
    expected_work_run_revision: int = Field(ge=1)
    expected_verification_request_revision: int = Field(ge=1)
    expected_window_revision: int = Field(ge=1)
    apply_id: str = Field(min_length=1, max_length=200)
    commit_settlement_apply_id: str = Field(min_length=1, max_length=200)
    interrupt_settlement_apply_id: str = Field(min_length=1, max_length=200)
    delivery_id: str = Field(min_length=1, max_length=200)
    input_limits: NodeVerificationInputLimits


class NodeVerificationStoreProjection(_ControllerContract):
    """小型变更 receipt 与请求权威记录的临时连接。

    ``resolved_result`` 从唯一持久化验证 request/result 记录解引用，不会复制到
    idempotency/apply receipt 中。
    """

    status: Literal["applied", "replayed"]
    outcome: NodeVerificationOutcome
    verification_request_id: str = Field(min_length=1, max_length=200)
    verified_verification_request_revision: int = Field(ge=1)
    verification_request_revision: int = Field(ge=1)
    verification_request_status: TaskNodeVerificationRequestStatus
    work_run_id: str = Field(min_length=1, max_length=200)
    locked_work_run_revision: int = Field(ge=1)
    verified_acceptance_progress_revision: int = Field(ge=1)
    verified_output_revision: int = Field(ge=1)
    work_run_revision: int = Field(ge=1)
    work_run_status: WorkRunStatus
    work_run_reason: str | None = Field(default=None, min_length=1, max_length=160)
    submitted_attempt_id: str = Field(min_length=1, max_length=200)
    acceptance_progress_revision: int = Field(ge=1)
    output_revision: int = Field(ge=1)
    window_revision: int = Field(ge=1)
    delivery_id: str | None = Field(default=None, min_length=1, max_length=200)
    resolved_result: NodeVerificationResult | None = None
    budget_transition: WorkRunBudgetTransition | None = None

    @model_validator(mode="after")
    def _require_result_only_for_semantic_outcome(
        self,
    ) -> 'NodeVerificationStoreProjection':
        if self.outcome == "work_run_limit_reached":
            if (
                self.resolved_result is not None
                or self.delivery_id is not None
                or self.verification_request_status
                is not TaskNodeVerificationRequestStatus.INTERRUPTED
                or self.work_run_status is not WorkRunStatus.FAILED
                or self.work_run_reason != "work_run_limit_reached"
                or self.budget_transition is None
                or self.budget_transition.disposition
                is not WorkRunBudgetDisposition.HARD_LIMIT_REACHED
            ):
                raise ValueError(
                    "a hard-limit verification cannot retain a semantic result"
                )
            return self
        if self.outcome == "interrupted":
            if (
                self.resolved_result is not None
                or self.delivery_id is not None
                or self.verification_request_status
                is not TaskNodeVerificationRequestStatus.INTERRUPTED
                or self.work_run_status is not WorkRunStatus.INTERRUPTED
            ):
                raise ValueError("an interrupted verification cannot have a result")
            return self
        if (
            self.resolved_result is None
            or self.verification_request_status
            is not TaskNodeVerificationRequestStatus.COMPLETED
        ):
            raise ValueError("a settled semantic verification requires its typed result")
        if self.resolved_result.all_pass is not (self.outcome == "passed"):
            raise ValueError("projection outcome disagrees with the Host-derived result")
        if self.outcome == "passed":
            if (
                self.delivery_id is None
                or self.work_run_status is not WorkRunStatus.COMPLETED
            ):
                raise ValueError("passing verification must create a terminal delivery")
        elif self.delivery_id is not None or self.work_run_status not in {
            WorkRunStatus.ACTIVE,
            WorkRunStatus.TURN_LIMIT_REACHED,
        }:
            raise ValueError(
                "non-pass verification must unlock or soft-stop the WorkRun"
            )
        elif self.work_run_status is WorkRunStatus.TURN_LIMIT_REACHED and (
            self.work_run_reason != "turn_limit_reached"
            or self.budget_transition is None
            or self.budget_transition.disposition
            is not WorkRunBudgetDisposition.SOFT_LIMIT_REACHED
        ):
            raise ValueError("soft-limit non-pass projection is inconsistent")
        if (
            self.resolved_result.verification_request_id
            != self.verification_request_id
            or self.resolved_result.verification_request_revision
            != self.verified_verification_request_revision
            or self.resolved_result.work_run_id != self.work_run_id
            or self.resolved_result.locked_work_run_revision
            != self.locked_work_run_revision
            or self.resolved_result.submitted_attempt_id
            != self.submitted_attempt_id
            or self.resolved_result.acceptance_progress_revision
            != self.verified_acceptance_progress_revision
            or self.resolved_result.output_revision
            != self.verified_output_revision
        ):
            raise ValueError("resolved result is not exactly bound to its Store projection")
        return self


class PreparedNodeVerification(_ControllerContract):
    """为外部调用以事务方式准备的权威输入。"""

    status: Literal["ready"] = "ready"
    verification_request_id: str = Field(min_length=1, max_length=200)
    verification_request_revision: int = Field(ge=1)
    window_revision: int = Field(ge=1)
    rebound_for_recovery: bool = False
    context: NodeVerificationContext


class SettledNodeVerification(_ControllerContract):
    """精确 prepare 重放；不允许第二次 provider 调用或 commit。"""

    status: Literal["settled"] = "settled"
    projection: NodeVerificationStoreProjection

    @model_validator(mode="after")
    def _exclude_recoverable_interrupt(self) -> 'SettledNodeVerification':
        if self.projection.outcome == "interrupted":
            raise ValueError(
                "an interrupted request is recoverable and cannot be a settled replay"
            )
        return self


NodeVerificationPreparation = (
    PreparedNodeVerification | SettledNodeVerification
)


class NodeVerificationCommit(_ControllerContract):
    """从一个已准备请求与模型结果推导的精确 commit 命令。"""

    session_id: str = Field(min_length=1, max_length=200)
    turn_id: str = Field(min_length=1, max_length=200)
    work_run_id: str = Field(min_length=1, max_length=200)
    verification_request_id: str = Field(min_length=1, max_length=200)
    expected_work_run_revision: int = Field(ge=1)
    expected_verification_request_revision: int = Field(ge=1)
    expected_window_revision: int = Field(ge=1)
    verification_result: NodeVerificationResult
    active_seconds_delta: float = Field(gt=0)
    apply_id: str = Field(min_length=1, max_length=200)
    delivery_id: str = Field(min_length=1, max_length=200)
    task_delivery_candidate_settlement: (
        task_delivery_store.TaskDeliveryCandidateSettlementIntent | None
    ) = None


class NodeVerificationInterrupt(_ControllerContract):
    """用于一个已准备请求的精确可恢复中断命令。"""

    session_id: str = Field(min_length=1, max_length=200)
    turn_id: str = Field(min_length=1, max_length=200)
    work_run_id: str = Field(min_length=1, max_length=200)
    verification_request_id: str = Field(min_length=1, max_length=200)
    expected_work_run_revision: int = Field(ge=1)
    expected_verification_request_revision: int = Field(ge=1)
    expected_window_revision: int = Field(ge=1)
    reason: NodeVerificationInterruptionReason
    active_seconds_delta: float = Field(gt=0)
    apply_id: str = Field(min_length=1, max_length=200)


class NodeVerificationStartNextAttempt(_ControllerContract):
    """非 pass 后命令；feedback 仍由 Store 持有且不是输入。"""

    session_id: str = Field(min_length=1, max_length=200)
    turn_id: str = Field(min_length=1, max_length=200)
    work_run_id: str = Field(min_length=1, max_length=200)
    expected_work_run_revision: int = Field(ge=1)
    expected_progress_revision: int = Field(ge=1)
    expected_output_revision: int = Field(ge=1)
    expected_window_revision: int = Field(ge=1)
    plan: NodeVerificationNextAttemptPlan


class NodeVerificationApplicationStore(Protocol):
    """应用 controller 所需的短事务权威状态。"""

    def prepare_node_verification(
        self,
        command: NodeVerificationPrepareStoreCommand,
    ) -> NodeVerificationPreparation: ...

    def resume_node_verification(
        self,
        command: NodeVerificationResumeStoreCommand,
    ) -> NodeVerificationPreparation: ...

    def commit_node_verification(
        self,
        command: NodeVerificationCommit,
    ) -> NodeVerificationStoreProjection: ...

    def interrupt_node_verification(
        self,
        command: NodeVerificationInterrupt,
    ) -> NodeVerificationStoreProjection: ...

    def start_next_attempt(
        self,
        command: NodeVerificationStartNextAttempt,
    ) -> WorkExecutionMutationResult: ...


class _SqliteNodeVerificationFacade(Protocol):
    """具体 adapter 使用的 node-verification 持久化接口。"""

    def prepare_task_node_verification(
        self, **kwargs: object
    ) -> TaskNodeVerificationMutationResult: ...

    def get_task_node_verification_record(
        self, **kwargs: object
    ) -> TaskNodeVerificationRecord: ...

    def get_prepared_task_node_verification(
        self, **kwargs: object
    ) -> PreparedTaskNodeVerification: ...

    def replay_task_node_verification_settlement(
        self, **kwargs: object
    ) -> TaskNodeVerificationMutationResult: ...

    def commit_task_node_verification_result(
        self, **kwargs: object
    ) -> TaskNodeVerificationMutationResult: ...

    def interrupt_task_node_verification(
        self, **kwargs: object
    ) -> TaskNodeVerificationMutationResult: ...

    def resume_task_node_verification(
        self, **kwargs: object
    ) -> TaskNodeVerificationMutationResult: ...


class _SqliteNodeVerificationWorkRunFacade(Protocol):
    def get_current_task_node_dependency_deliveries(
        self, **kwargs: object
    ) -> tuple[ResolvedCurrentTaskNodeDelivery, ...]: ...

    def start_work_run_attempt(
        self, **kwargs: object
    ) -> WorkExecutionMutationResult: ...


class _SqliteNodeVerificationTaskFacade(Protocol):
    def get_insession_task_details(self, *args: object): ...


class SqliteNodeVerificationApplicationStore:
    """基于 ``personagraph.session.store`` 的具体短事务 adapter。

    Prepare/resume apply receipt 会先重放，因此即使活动 WorkRun 已推进，也能恢复精确
    prepare 后 revision。已结算请求通过仅含 receipt 的 Store 重放 port 投影；临时
    provider 输出与 active-time 测量绝不重建。
    """

    def __init__(
        self,
        *,
        verification_store: _SqliteNodeVerificationFacade = verification_store,
        work_run_store: _SqliteNodeVerificationWorkRunFacade = work_run_store,
        task_store: _SqliteNodeVerificationTaskFacade = task_graph_store,
    ) -> None:
        self._verification_store = verification_store
        self._work_run_store = work_run_store
        self._task_store = task_store

    def prepare_node_verification(
        self,
        command: NodeVerificationPrepareStoreCommand,
    ) -> NodeVerificationPreparation:
        mutation = self._verification_store.prepare_task_node_verification(
            session_id=command.session_id,
            turn_id=command.turn_id,
            work_run_id=command.work_run_id,
            expected_work_run_revision=command.expected_work_run_revision,
            expected_progress_revision=command.expected_progress_revision,
            expected_output_revision=command.expected_output_revision,
            expected_window_revision=command.expected_window_revision,
            apply_id=command.apply_id,
            verification_request_id=command.verification_request_id,
            recover_unprepared_submit=command.recover_unprepared_submit,
        )
        return self._resolve_preparation(
            command=command,
            mutation=mutation,
            rebound=False,
        )

    def resume_node_verification(
        self,
        command: NodeVerificationResumeStoreCommand,
    ) -> NodeVerificationPreparation:
        mutation = self._verification_store.resume_task_node_verification(
            session_id=command.session_id,
            turn_id=command.turn_id,
            work_run_id=command.work_run_id,
            verification_request_id=command.verification_request_id,
            expected_work_run_revision=command.expected_work_run_revision,
            expected_verification_request_revision=(
                command.expected_verification_request_revision
            ),
            expected_window_revision=command.expected_window_revision,
            apply_id=command.apply_id,
        )
        return self._resolve_preparation(
            command=command,
            mutation=mutation,
            rebound=True,
        )

    def commit_node_verification(
        self,
        command: NodeVerificationCommit,
    ) -> NodeVerificationStoreProjection:
        mutation = self._verification_store.commit_task_node_verification_result(
            session_id=command.session_id,
            turn_id=command.turn_id,
            work_run_id=command.work_run_id,
            verification_request_id=command.verification_request_id,
            result=command.verification_result,
            expected_work_run_revision=command.expected_work_run_revision,
            expected_verification_request_revision=(
                command.expected_verification_request_revision
            ),
            expected_window_revision=command.expected_window_revision,
            active_seconds_delta=command.active_seconds_delta,
            apply_id=command.apply_id,
            delivery_id=(
                command.delivery_id if command.verification_result.all_pass else None
            ),
            task_delivery_candidate_settlement=(
                command.task_delivery_candidate_settlement
            ),
        )
        record = self._verification_store.get_task_node_verification_record(
            session_id=command.session_id,
            verification_request_id=command.verification_request_id,
        )
        return _project_store_resolution(mutation=mutation, record=record)

    def interrupt_node_verification(
        self,
        command: NodeVerificationInterrupt,
    ) -> NodeVerificationStoreProjection:
        mutation = self._verification_store.interrupt_task_node_verification(
            session_id=command.session_id,
            turn_id=command.turn_id,
            work_run_id=command.work_run_id,
            verification_request_id=command.verification_request_id,
            technical_error_code=command.reason,
            expected_work_run_revision=command.expected_work_run_revision,
            expected_verification_request_revision=(
                command.expected_verification_request_revision
            ),
            expected_window_revision=command.expected_window_revision,
            active_seconds_delta=command.active_seconds_delta,
            apply_id=command.apply_id,
        )
        record = self._verification_store.get_task_node_verification_record(
            session_id=command.session_id,
            verification_request_id=command.verification_request_id,
        )
        return _project_store_resolution(mutation=mutation, record=record)

    def start_next_attempt(
        self,
        command: NodeVerificationStartNextAttempt,
    ) -> WorkExecutionMutationResult:
        return self._work_run_store.start_work_run_attempt(
            session_id=command.session_id,
            turn_id=command.turn_id,
            work_run_id=command.work_run_id,
            expected_work_run_revision=command.expected_work_run_revision,
            expected_progress_revision=command.expected_progress_revision,
            expected_window_revision=command.expected_window_revision,
            apply_id=command.plan.apply_id,
            catalog_snapshot=command.plan.catalog_snapshot,
            # Store 推导精确的非 pass 验证请求 checkpoint；调用方无法替换为无关
            # checkpoint 标识。
            input_checkpoint_id=None,
            attempt_id=command.plan.attempt_id,
        )

    def reproject_pending_context(
        self,
        *,
        session_id: str,
        invocation_turn_id: str,
        verification_request_id: str,
        input_limits: NodeVerificationInputLimits,
    ) -> NodeVerificationContext:
        """为模型状态 guard 重建精确活动 verifier 输入。

        此操作只读，并有意与 prepare/resume 共享相同依赖/来源投影函数。因此缺失、
        已结算或过期请求会拒绝 Provider 分派，而不会将持久化模型 guard 削弱为少量
        记录 revision。
        """

        prepared = self._verification_store.get_prepared_task_node_verification(
            session_id=session_id,
            invocation_turn_id=invocation_turn_id,
            verification_request_id=verification_request_id,
        )
        return _build_semantic_verification_context(
            prepared,
            input_limits=input_limits,
            dependency_deliveries=self._resolve_dependency_deliveries(prepared),
            source_context=self._resolve_source_context(prepared),
        )

    def _resolve_preparation(
        self,
        *,
        command: (
            NodeVerificationPrepareStoreCommand
            | NodeVerificationResumeStoreCommand
        ),
        mutation: TaskNodeVerificationMutationResult,
        rebound: bool,
    ) -> NodeVerificationPreparation:
        if (
            mutation.verification_request_id != command.verification_request_id
            or mutation.work_run_id != command.work_run_id
            or mutation.verification_request_status
            is not TaskNodeVerificationRequestStatus.PENDING
        ):
            raise NodeVerificationControllerStateConflict(
                "verification prepare receipt crossed request authority"
            )
        record = self._verification_store.get_task_node_verification_record(
            session_id=command.session_id,
            verification_request_id=command.verification_request_id,
        )
        request = record.request
        if (
            request.session_id != command.session_id
            or request.work_run_id != command.work_run_id
            or request.submitted_attempt_id != command.submitted_attempt_id
        ):
            raise NodeVerificationControllerStateConflict(
                "verification request row crossed application authority"
            )
        if request.status is TaskNodeVerificationRequestStatus.COMPLETED:
            if record.result is None:
                raise NodeVerificationControllerStateConflict(
                    "completed verification request has no result"
                )
            replay = self._verification_store.replay_task_node_verification_settlement(
                session_id=command.session_id,
                work_run_id=command.work_run_id,
                verification_request_id=command.verification_request_id,
                apply_id=command.commit_settlement_apply_id,
                operation="commit_verification_result",
            )
            if replay.status != "replayed":
                raise NodeVerificationControllerStateConflict(
                    "settled verification did not replay its original commit"
                )
            return SettledNodeVerification(
                projection=_project_store_resolution(
                    mutation=replay,
                    record=record,
                )
            )
        if request.status is TaskNodeVerificationRequestStatus.INTERRUPTED:
            if request.technical_error_code == "work_run_limit_reached":
                replay = self._replay_hard_limit_settlement(
                    command=command,
                )
                return SettledNodeVerification(
                    projection=_project_store_resolution(
                        mutation=replay,
                        record=record,
                    )
                )
            raise NodeVerificationControllerStateConflict(
                "interrupted verification requires the explicit resume entry"
            )
        if request.revision != mutation.verification_request_revision:
            raise NodeVerificationControllerStateConflict(
                "prepared request revision does not match its mutation receipt"
            )
        prepared = self._verification_store.get_prepared_task_node_verification(
            session_id=command.session_id,
            invocation_turn_id=command.turn_id,
            verification_request_id=command.verification_request_id,
        )
        if (
            prepared.record != record
            or prepared.window_state_version != mutation.window_state_version
            or prepared.work_run.revision != mutation.work_run_revision
        ):
            raise NodeVerificationControllerStateConflict(
                "prepared verification projection is stale"
            )
        return PreparedNodeVerification(
            verification_request_id=request.verification_request_id,
            verification_request_revision=request.revision,
            window_revision=prepared.window_state_version,
            rebound_for_recovery=rebound,
            context=_build_semantic_verification_context(
                prepared,
                input_limits=command.input_limits,
                dependency_deliveries=self._resolve_dependency_deliveries(
                    prepared
                ),
                source_context=self._resolve_source_context(prepared),
            ),
        )

    def _resolve_source_context(
        self,
        prepared: PreparedTaskNodeVerification,
    ) -> TaskNodeSourceContext | None:
        request = prepared.record.request
        if not isinstance(request.subject, TaskNodeSubject):
            return None
        details = self._task_store.get_insession_task_details(
            request.session_id,
            request.subject.task_id,
        )
        if details is None:
            raise NodeVerificationControllerStateConflict(
                "verification source TaskGraph is unavailable"
            )
        return build_task_node_source_context(
            session_id=request.session_id,
            details=details,
            subject=request.subject,
        )

    def _resolve_dependency_deliveries(
        self,
        prepared: PreparedTaskNodeVerification,
    ) -> TaskNodeDependencyDeliveries:
        request = prepared.record.request
        # 空值是精确持久化请求快照。避免为旧版单节点 TaskGraph 重建规范树；其根节点
        # ID 早于当前 ``node_id == task_id`` 不变量。
        if not request.dependency_delivery_ids:
            return TaskNodeDependencyDeliveries()
        details = self._task_store.get_insession_task_details(
            request.session_id,
            request.subject.task_id,
        )
        if details is None:
            raise NodeVerificationControllerStateConflict(
                "verification dependency TaskGraph is unavailable"
            )
        resolved = tuple(
            self._work_run_store.get_current_task_node_dependency_deliveries(
                session_id=request.session_id,
                subject=request.subject,
            )
        )
        try:
            projection = project_task_node_dependencies(
                build_task_node_tree(details),
                parent_subject=request.subject,
                resolved_deliveries=resolved,
            )
        except (TypeError, ValueError) as exc:
            raise NodeVerificationControllerStateConflict(
                "verification dependency Delivery projection is stale"
            ) from exc
        if projection.delivery_ids != request.dependency_delivery_ids:
            raise NodeVerificationControllerStateConflict(
                "verification dependency Delivery order crossed request authority"
            )
        return projection.to_model_deliveries()

    def _replay_hard_limit_settlement(
        self,
        *,
        command: (
            NodeVerificationPrepareStoreCommand
            | NodeVerificationResumeStoreCommand
        ),
    ) -> TaskNodeVerificationMutationResult:
        try:
            return self._verification_store.replay_task_node_verification_settlement(
                session_id=command.session_id,
                work_run_id=command.work_run_id,
                verification_request_id=command.verification_request_id,
                apply_id=command.commit_settlement_apply_id,
                operation="commit_verification_result",
            )
        except VerificationSettlementReceiptNotFound:
            try:
                return self._verification_store.replay_task_node_verification_settlement(
                    session_id=command.session_id,
                    work_run_id=command.work_run_id,
                    verification_request_id=command.verification_request_id,
                    apply_id=command.interrupt_settlement_apply_id,
                    operation="interrupt_verification",
                )
            except VerificationSettlementReceiptNotFound as exc:
                raise NodeVerificationControllerStateConflict(
                    "hard-limit verification has no settlement receipt"
                ) from exc


class NodeSemanticVerifier(Protocol):
    """一次逻辑语义验证调用，可能包含物理重试。"""

    def __call__(
        self,
        context: NodeVerificationContext,
        *,
        provider: NodeVerificationStructuredProvider,
        emit: Callable[[TurnEvent], object],
        deadline: TurnDeadline | None = None,
    ) -> ModelRequestResult[NodeVerificationResult]: ...


class NodeDownstreamVerificationGate(Protocol):
    """在 Store 冻结 Delivery 前审查 node-PASS 候选项。

    gate 可添加独立 PASS 或 RETRY_ATTEMPT 结果，但不能改写节点 Acceptance verdict；
    更广泛的 wait/replan 路由仍由专用状态转换负责。
    """

    def __call__(
        self,
        prepared: PreparedNodeVerification,
        node_result: NodeVerificationResult,
    ) -> tuple[DownstreamVerificationFeedback, ...]: ...


class NodeDownstreamVerificationRouteRequired(RuntimeError):
    """PASS 候选项需要超出普通 feedback 的原子路由。"""

    def __init__(
        self,
        *,
        settlement_intent: task_delivery_store.TaskDeliveryCandidateSettlementIntent,
    ) -> None:
        if not isinstance(
            settlement_intent,
            task_delivery_store.TaskDeliveryCandidateSettlementIntent,
        ):
            raise TypeError("downstream route requires a typed settlement intent")
        self.settlement_intent = settlement_intent
        super().__init__("node PASS requires a dedicated downstream settlement")


class NodeVerificationModelCall(_ControllerContract):
    model_call_id: str = Field(min_length=1)
    provider: str
    model: str
    physical_attempts: int = Field(ge=1)
    latency_ms: int = Field(ge=0)


class NodeVerificationApplicationResult(_ControllerContract):
    """不产生用户可见发布副作用的应用结果。"""

    outcome: NodeVerificationOutcome
    store_projection: NodeVerificationStoreProjection
    model_call: NodeVerificationModelCall | None = None
    interruption_reason: NodeVerificationInterruptionReason | None = None
    replayed_without_model: bool = False
    next_attempt_mutation: WorkExecutionMutationResult | None = None

    @model_validator(mode="after")
    def _require_consistent_outcome(self) -> 'NodeVerificationApplicationResult':
        if self.store_projection.outcome != self.outcome:
            raise ValueError("application outcome disagrees with its Store projection")
        if self.outcome == "work_run_limit_reached":
            if (
                self.interruption_reason != "work_run_limit_reached"
                or self.next_attempt_mutation is not None
            ):
                raise ValueError(
                    "hard-limit verification requires its durable budget reason"
                )
            if self.replayed_without_model and (
                self.model_call is not None
                or self.store_projection.status != "replayed"
            ):
                raise ValueError(
                    "hard-limit replay must skip the model and replay Store authority"
                )
            return self
        if self.outcome == "interrupted":
            if (
                self.interruption_reason is None
                or self.model_call is not None
                or self.next_attempt_mutation is not None
            ):
                raise ValueError(
                    "interruption requires a reason and cannot claim a completed model call"
                )
            if self.replayed_without_model:
                raise ValueError("recoverable interruption is not a settled replay")
            return self
        if self.interruption_reason is not None:
            raise ValueError("semantic outcome cannot carry an interruption reason")
        if self.replayed_without_model:
            if (
                self.model_call is not None
                or self.store_projection.status != "replayed"
            ):
                raise ValueError(
                    "settled replay must skip the model and return a replay projection"
                )
        elif self.model_call is None:
            raise ValueError("a newly committed semantic outcome requires model-call facts")
        if self.outcome == "not_passed":
            requires_next_attempt = (
                self.store_projection.work_run_status is WorkRunStatus.ACTIVE
            )
            if requires_next_attempt is (self.next_attempt_mutation is None):
                raise ValueError(
                    "only an active non-pass must start or replay the next Attempt"
                )
        elif self.next_attempt_mutation is not None:
            raise ValueError("only active non-pass may carry a next Attempt")
        return self


def run_node_verification(
    request: NodeVerificationApplicationRequest,
    *,
    store: NodeVerificationApplicationStore,
    provider: NodeVerificationStructuredProvider,
    emit: Callable[[TurnEvent], object],
    monotonic_clock: Callable[[], float],
    deadline: TurnDeadline | None = None,
    verifier: NodeSemanticVerifier = request_node_verification,
    downstream_gate: NodeDownstreamVerificationGate | None = None,
) -> NodeVerificationApplicationResult:
    """准备、在 Store 事务外验证，随后提交或中断。

    Store prepare/commit/interrupt 方法分别是独立调用，必须各自持有短事务。此处不
    转换任何 Store 异常：CAS、collision、replay 与 late-result 语义保持权威并失败
    关闭。只有调用外部语义 verifier 时产生的失败才走可恢复中断路径。
    """

    stable_next_attempt = _snapshot_next_attempt_plan(request.next_attempt)
    prepared_or_settled = store.prepare_node_verification(
        NodeVerificationPrepareStoreCommand(
            session_id=request.session_id,
            turn_id=request.turn_id,
            work_run_id=request.work_run_id,
            submitted_attempt_id=request.submitted_attempt_id,
            verification_request_id=request.verification_request_id,
            expected_work_run_revision=request.expected_work_run_revision,
            expected_progress_revision=request.expected_progress_revision,
            expected_output_revision=request.expected_output_revision,
            expected_window_revision=request.expected_window_revision,
            apply_id=request.prepare_apply_id,
            commit_settlement_apply_id=request.commit_apply_id,
            interrupt_settlement_apply_id=request.interrupt_apply_id,
            delivery_id=request.delivery_id,
            input_limits=request.input_limits,
            recover_unprepared_submit=request.recover_unprepared_submit,
        )
    )
    return _run_prepared_node_verification(
        request=request,
        prepared_or_settled=prepared_or_settled,
        stable_next_attempt=stable_next_attempt,
        store=store,
        provider=provider,
        emit=emit,
        monotonic_clock=monotonic_clock,
        deadline=deadline,
        verifier=verifier,
        expect_rebound=False,
        downstream_gate=downstream_gate,
    )


def resume_and_run_node_verification(
    request: NodeVerificationResumeRequest,
    *,
    store: NodeVerificationApplicationStore,
    provider: NodeVerificationStructuredProvider,
    emit: Callable[[TurnEvent], object],
    monotonic_clock: Callable[[], float],
    deadline: TurnDeadline | None = None,
    verifier: NodeSemanticVerifier = request_node_verification,
    downstream_gate: NodeDownstreamVerificationGate | None = None,
) -> NodeVerificationApplicationResult:
    """重新绑定一个 pending/interrupted 请求，再运行共享流程。"""

    stable_next_attempt = _snapshot_next_attempt_plan(request.next_attempt)
    prepared_or_settled = store.resume_node_verification(
        NodeVerificationResumeStoreCommand(
            session_id=request.session_id,
            turn_id=request.turn_id,
            work_run_id=request.work_run_id,
            submitted_attempt_id=request.submitted_attempt_id,
            verification_request_id=request.verification_request_id,
            expected_work_run_revision=request.expected_work_run_revision,
            expected_verification_request_revision=(
                request.expected_verification_request_revision
            ),
            expected_window_revision=request.expected_window_revision,
            apply_id=request.resume_apply_id,
            commit_settlement_apply_id=request.commit_apply_id,
            interrupt_settlement_apply_id=request.interrupt_apply_id,
            delivery_id=request.delivery_id,
            input_limits=request.input_limits,
        )
    )
    return _run_prepared_node_verification(
        request=request,
        prepared_or_settled=prepared_or_settled,
        stable_next_attempt=stable_next_attempt,
        store=store,
        provider=provider,
        emit=emit,
        monotonic_clock=monotonic_clock,
        deadline=deadline,
        verifier=verifier,
        expect_rebound=True,
        downstream_gate=downstream_gate,
    )


def _run_prepared_node_verification(
    *,
    request: NodeVerificationApplicationRequest | NodeVerificationResumeRequest,
    prepared_or_settled: NodeVerificationPreparation,
    stable_next_attempt: NodeVerificationNextAttemptPlan,
    store: NodeVerificationApplicationStore,
    provider: NodeVerificationStructuredProvider,
    emit: Callable[[TurnEvent], object],
    monotonic_clock: Callable[[], float],
    deadline: TurnDeadline | None,
    verifier: NodeSemanticVerifier,
    expect_rebound: bool,
    downstream_gate: NodeDownstreamVerificationGate | None,
) -> NodeVerificationApplicationResult:
    if isinstance(prepared_or_settled, SettledNodeVerification):
        _require_exact_store_projection(request, prepared_or_settled.projection)
        next_attempt_mutation = _start_next_attempt_if_needed(
            request=request,
            projection=prepared_or_settled.projection,
            stable_plan=stable_next_attempt,
            store=store,
        )
        return NodeVerificationApplicationResult(
            outcome=prepared_or_settled.projection.outcome,
            store_projection=prepared_or_settled.projection,
            interruption_reason=(
                "work_run_limit_reached"
                if prepared_or_settled.projection.outcome
                == "work_run_limit_reached"
                else None
            ),
            replayed_without_model=True,
            next_attempt_mutation=next_attempt_mutation,
        )

    prepared = prepared_or_settled
    _require_exact_preparation(request, prepared, expect_rebound=expect_rebound)
    active_time_meter = NodeVerificationActiveTimeMeter.starting_now(
        monotonic_clock
    )

    def interrupt_after_failure(
        failure: Exception,
    ) -> tuple[str, NodeVerificationStoreProjection]:
        """在失败逸出前持久关闭此已准备请求。"""

        reason = _interruption_reason(failure)
        active_seconds_delta = active_time_meter.freeze()
        projection = store.interrupt_node_verification(
            NodeVerificationInterrupt(
                session_id=request.session_id,
                turn_id=request.turn_id,
                work_run_id=request.work_run_id,
                verification_request_id=request.verification_request_id,
                expected_work_run_revision=prepared.context.work_run.revision,
                expected_verification_request_revision=(
                    prepared.verification_request_revision
                ),
                expected_window_revision=prepared.window_revision,
                reason=reason,
                active_seconds_delta=active_seconds_delta,
                apply_id=request.interrupt_apply_id,
            )
        )
        _require_exact_store_projection(
            request,
            projection,
            prepared=prepared,
            expected_outcomes={"interrupted", "work_run_limit_reached"},
        )
        return reason, projection

    try:
        requested = verifier(
            prepared.context,
            provider=provider,
            emit=emit,
            deadline=deadline,
        )
    except Exception as exc:
        reason, projection = interrupt_after_failure(exc)
        if isinstance(exc, ContextBudgetExceeded):
            # 请求级持久化状态现在已安全。保留预算失败，使所属 Entry 能以精确公开错误码
            # 结算 Turn，而不是将其视为验证结果。
            raise
        outcome = projection.outcome
        return NodeVerificationApplicationResult(
            outcome=outcome,
            store_projection=projection,
            interruption_reason=(
                reason
                if outcome == "interrupted"
                else "work_run_limit_reached"
            ),
        )

    # 过期语义响应在 request-generation 竞态中落败；它不是 provider 失败。将精确绑定
    # 验证保留在 interrupt 回退外，使已放弃调用无法中断重新绑定的请求。
    _require_exact_verification_result(prepared, requested.value)
    verification_result = requested.value
    candidate_settlement_intent = None
    if downstream_gate is not None and verification_result.all_pass:
        try:
            downstream_results = tuple(
                downstream_gate(prepared, verification_result)
            )
        except NodeDownstreamVerificationRouteRequired as route:
            candidate_settlement_intent = route.settlement_intent
            downstream_results = ()
        except ContextBudgetExceeded as exc:
            # whole-task/candidate gate 在语义 provider 调用后运行，但仍属于此已准备的
            # 持久化请求。在所属 Entry 结算 Turn 级预算失败前中断它。
            interrupt_after_failure(exc)
            raise
        if any(
            item.disposition
            not in {
                DownstreamVerificationDisposition.PASS,
                DownstreamVerificationDisposition.RETRY_ATTEMPT,
            }
            for item in downstream_results
        ):
            raise NodeVerificationControllerStateConflict(
                "the ordinary node settlement cannot apply a wait or replan route"
            )
        verification_result = NodeVerificationResult.model_validate(
            {
                **verification_result.model_dump(mode="json"),
                "downstream_results": [
                    item.model_dump(mode="json") for item in downstream_results
                ],
                "all_pass": all(
                    item.disposition
                    is DownstreamVerificationDisposition.PASS
                    for item in downstream_results
                ),
            }
        )
        _require_exact_verification_result(prepared, verification_result)
        if (
            verification_result.acceptance_results
            != requested.value.acceptance_results
        ):
            raise NodeVerificationControllerStateConflict(
                "a downstream gate rewrote node semantic verdicts"
            )
    model_call = NodeVerificationModelCall(
        model_call_id=requested.model_call_id,
        provider=requested.model_result.provider,
        model=requested.model_result.model,
        physical_attempts=requested.attempts,
        latency_ms=requested.model_result.latency_ms,
    )
    active_seconds_delta = active_time_meter.freeze()
    projection = store.commit_node_verification(
        NodeVerificationCommit(
            session_id=request.session_id,
            turn_id=request.turn_id,
            work_run_id=request.work_run_id,
            verification_request_id=request.verification_request_id,
            expected_work_run_revision=prepared.context.work_run.revision,
            expected_verification_request_revision=(
                prepared.verification_request_revision
            ),
            expected_window_revision=prepared.window_revision,
            verification_result=verification_result,
            active_seconds_delta=active_seconds_delta,
            apply_id=request.commit_apply_id,
            delivery_id=request.delivery_id,
            task_delivery_candidate_settlement=candidate_settlement_intent,
        )
    )
    expected_outcome: Literal["passed", "not_passed"] = (
        "passed" if verification_result.all_pass else "not_passed"
    )
    _require_exact_store_projection(
        request,
        projection,
        prepared=prepared,
        expected_outcomes={expected_outcome, "work_run_limit_reached"},
    )
    next_attempt_mutation = _start_next_attempt_if_needed(
        request=request,
        projection=projection,
        stable_plan=stable_next_attempt,
        store=store,
    )
    return NodeVerificationApplicationResult(
        outcome=projection.outcome,
        store_projection=projection,
        model_call=model_call,
        interruption_reason=(
            "work_run_limit_reached"
            if projection.outcome == "work_run_limit_reached"
            else None
        ),
        next_attempt_mutation=next_attempt_mutation,
    )


def _require_exact_preparation(
    request: NodeVerificationApplicationRequest | NodeVerificationResumeRequest,
    prepared: PreparedNodeVerification,
    *,
    expect_rebound: bool,
) -> None:
    context = prepared.context
    if prepared.verification_request_id != request.verification_request_id:
        raise NodeVerificationControllerStateConflict(
            "Store prepared another verification request"
        )
    if (
        context.session_id != request.session_id
        or context.invocation_turn_id != request.turn_id
        or context.verification_request_id != request.verification_request_id
        or context.verification_request_revision
        != prepared.verification_request_revision
        or context.work_run_id != request.work_run_id
        or context.submitted_attempt_id != request.submitted_attempt_id
        or context.input_limits != request.input_limits
    ):
        raise NodeVerificationControllerStateConflict(
            "Store verification context crossed request authority"
        )
    if prepared.rebound_for_recovery is not expect_rebound:
        raise NodeVerificationControllerStateConflict(
            "Store verification preparation used the wrong lifecycle path"
        )
    if not expect_rebound and context.request_turn_id != request.turn_id:
        raise NodeVerificationControllerStateConflict(
            "initial verification request has the wrong origin Turn"
        )
    if isinstance(request, NodeVerificationApplicationRequest) and (
        context.locked_work_run_revision != request.expected_work_run_revision
        or context.acceptance_progress_revision
        != request.expected_progress_revision
        or context.locked_output_window.output_revision
        != request.expected_output_revision
    ):
        raise NodeVerificationControllerStateConflict(
            "initial prepared context does not match the caller CAS snapshot"
        )


def _require_exact_verification_result(
    prepared: PreparedNodeVerification,
    result: NodeVerificationResult,
) -> None:
    context = prepared.context
    expected_acceptance_ids = tuple(
        item.acceptance_id for item in context.acceptances
    )
    result_acceptance_ids = tuple(
        item.acceptance_id for item in result.acceptance_results
    )
    if (
        result.verification_request_id != context.verification_request_id
        or result.verification_request_revision
        != context.verification_request_revision
        or result.work_run_id != context.work_run_id
        or result.locked_work_run_revision
        != context.locked_work_run_revision
        or result.submitted_attempt_id != context.submitted_attempt_id
        or result.acceptance_progress_revision
        != context.acceptance_progress_revision
        or result.subject != context.subject
        or result.output_revision != context.locked_output_window.output_revision
        or result_acceptance_ids != expected_acceptance_ids
    ):
        raise NodeVerificationControllerStateConflict(
            "semantic verifier returned a stale or misbound result"
        )


def _require_exact_store_projection(
    request: NodeVerificationApplicationRequest | NodeVerificationResumeRequest,
    projection: NodeVerificationStoreProjection,
    *,
    prepared: PreparedNodeVerification | None = None,
    expected_outcomes: set[NodeVerificationOutcome] | None = None,
) -> None:
    if (
        projection.verification_request_id != request.verification_request_id
        or projection.work_run_id != request.work_run_id
        or projection.submitted_attempt_id != request.submitted_attempt_id
    ):
        raise NodeVerificationControllerStateConflict(
            "Store projection crossed verification request authority"
        )
    if expected_outcomes is not None and projection.outcome not in expected_outcomes:
        raise NodeVerificationControllerStateConflict(
            "Store projection outcome disagrees with the requested transition"
        )
    if isinstance(request, NodeVerificationApplicationRequest) and (
        projection.locked_work_run_revision != request.expected_work_run_revision
        or projection.verified_acceptance_progress_revision
        != request.expected_progress_revision
        or projection.verified_output_revision != request.expected_output_revision
    ):
        raise NodeVerificationControllerStateConflict(
            "Store projection does not match the initial caller CAS snapshot"
        )
    if prepared is not None and (
        projection.verified_verification_request_revision
        != prepared.verification_request_revision
        or projection.locked_work_run_revision
        != prepared.context.locked_work_run_revision
        or projection.verified_acceptance_progress_revision
        != prepared.context.acceptance_progress_revision
        or projection.verified_output_revision
        != prepared.context.locked_output_window.output_revision
    ):
        raise NodeVerificationControllerStateConflict(
            "Store projection does not bind the prepared semantic snapshot"
        )


def _start_next_attempt_if_needed(
    *,
    request: NodeVerificationApplicationRequest | NodeVerificationResumeRequest,
    projection: NodeVerificationStoreProjection,
    stable_plan: NodeVerificationNextAttemptPlan,
    store: NodeVerificationApplicationStore,
) -> WorkExecutionMutationResult | None:
    if (
        projection.outcome != "not_passed"
        or projection.work_run_status is not WorkRunStatus.ACTIVE
    ):
        return None
    mutation = store.start_next_attempt(
        NodeVerificationStartNextAttempt(
            session_id=request.session_id,
            turn_id=request.turn_id,
            work_run_id=request.work_run_id,
            expected_work_run_revision=projection.work_run_revision,
            expected_progress_revision=projection.acceptance_progress_revision,
            expected_output_revision=projection.output_revision,
            expected_window_revision=projection.window_revision,
            plan=stable_plan,
        )
    )
    if (
        mutation.work_run_id != request.work_run_id
        or mutation.current_attempt_id != stable_plan.attempt_id
        or mutation.attempt is None
        or mutation.attempt.attempt_id != stable_plan.attempt_id
    ):
        raise NodeVerificationControllerStateConflict(
            "Store started a different post-verification Attempt"
        )
    return mutation


def _build_semantic_verification_context(
    prepared: PreparedTaskNodeVerification,
    *,
    input_limits: NodeVerificationInputLimits,
    dependency_deliveries: TaskNodeDependencyDeliveries,
    source_context: TaskNodeSourceContext | None,
) -> NodeVerificationContext:
    request = prepared.record.request
    return NodeVerificationContext(
        session_id=request.session_id,
        request_turn_id=request.request_turn_id,
        invocation_turn_id=prepared.invocation_turn_id,
        verification_request_id=request.verification_request_id,
        verification_request_revision=request.revision,
        locked_work_run_revision=request.locked_work_run_revision,
        work_run=prepared.work_run,
        submitted_attempt=prepared.submitted_attempt,
        acceptance_progress=prepared.acceptance_progress,
        node_title=prepared.node_title,
        node_objective=prepared.node_objective,
        acceptances=prepared.acceptances,
        locked_output_window=prepared.locked_output_window,
        dependency_deliveries=dependency_deliveries,
        source_context=source_context,
        supporting_tool_results=SupportingToolResults(
            items=prepared.supporting_tool_results
        ),
        input_limits=input_limits,
    )


def _project_store_resolution(
    *,
    mutation: TaskNodeVerificationMutationResult,
    record: TaskNodeVerificationRecord,
) -> NodeVerificationStoreProjection:
    request = record.request
    if (
        mutation.verification_request_id != request.verification_request_id
        or mutation.verification_request_revision != request.revision
        or mutation.verification_request_status is not request.status
        or mutation.work_run_id != request.work_run_id
    ):
        raise NodeVerificationControllerStateConflict(
            "verification mutation and request row are inconsistent"
        )
    if request.status is TaskNodeVerificationRequestStatus.PENDING:
        raise NodeVerificationControllerStateConflict(
            "pending verification is not a settled Store projection"
        )
    if request.status is TaskNodeVerificationRequestStatus.INTERRUPTED:
        if record.result is not None or mutation.all_pass is not None:
            raise NodeVerificationControllerStateConflict(
                "interrupted verification carries a semantic result"
            )
        if (
            mutation.work_run_status is WorkRunStatus.FAILED
            and mutation.work_run_reason == "work_run_limit_reached"
            and request.technical_error_code == "work_run_limit_reached"
            and mutation.budget_transition is not None
            and mutation.budget_transition.disposition
            is WorkRunBudgetDisposition.HARD_LIMIT_REACHED
        ):
            outcome: NodeVerificationOutcome = "work_run_limit_reached"
        else:
            outcome = "interrupted"
        verified_request_revision = request.revision - 1
    else:
        if (
            record.result is None
            or mutation.all_pass is not record.result.all_pass
        ):
            raise NodeVerificationControllerStateConflict(
                "completed verification result does not match its mutation"
            )
        outcome = "passed" if record.result.all_pass else "not_passed"
        verified_request_revision = record.result.verification_request_revision
    return NodeVerificationStoreProjection(
        status=mutation.status,
        outcome=outcome,
        verification_request_id=request.verification_request_id,
        verified_verification_request_revision=verified_request_revision,
        verification_request_revision=request.revision,
        verification_request_status=request.status,
        work_run_id=request.work_run_id,
        locked_work_run_revision=request.locked_work_run_revision,
        verified_acceptance_progress_revision=(
            request.acceptance_progress_revision
        ),
        verified_output_revision=request.output_revision,
        work_run_revision=mutation.work_run_revision,
        work_run_status=mutation.work_run_status,
        work_run_reason=mutation.work_run_reason,
        submitted_attempt_id=request.submitted_attempt_id,
        acceptance_progress_revision=request.acceptance_progress_revision,
        output_revision=request.output_revision,
        window_revision=mutation.window_state_version,
        delivery_id=mutation.delivery_id,
        resolved_result=record.result,
        budget_transition=mutation.budget_transition,
    )


def _snapshot_next_attempt_plan(
    plan: NodeVerificationNextAttemptPlan,
) -> NodeVerificationNextAttemptPlan:
    try:
        catalog = json.loads(
            json.dumps(
                plan.catalog_snapshot,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("next Attempt catalog snapshot must be finite JSON") from exc
    return plan.model_copy(update={"catalog_snapshot": catalog})


def _interruption_reason(
    error: Exception,
) -> NodeVerificationInterruptionReason:
    if isinstance(
        error,
        (NodeVerificationInputTooLarge, TaskNodeDependencyInputTooLarge),
    ):
        return "verification_input_too_large"
    if isinstance(
        error,
        (NodeVerificationInputUnsupported, TaskNodeDependencyInputUnsupported),
    ):
        return "verification_input_unsupported"
    if isinstance(error, ModelOutputValidationError):
        return "verification_output_invalid"
    if isinstance(error, ContextBudgetExceeded):
        return "context_budget_exceeded"
    if isinstance(error, ModelGatewayError):
        if error.code == "MODEL_BAD_RESPONSE":
            return "verification_output_invalid"
        return "verification_unavailable"
    return "runtime_module_error"


__all__ = [
    "NodeDownstreamVerificationGate",
    "NodeDownstreamVerificationRouteRequired",
    "NodeSemanticVerifier",
    "NodeVerificationActiveTimeMeasurementError",
    "NodeVerificationActiveTimeMeter",
    'NodeVerificationApplicationRequest',
    'NodeVerificationApplicationResult',
    "NodeVerificationApplicationStore",
    'NodeVerificationCommit',
    "NodeVerificationControllerStateConflict",
    'NodeVerificationInterruptionReason',
    'NodeVerificationInputLimits',
    'NodeVerificationInterrupt',
    'NodeVerificationModelCall',
    'NodeVerificationNextAttemptPlan',
    'NodeVerificationOutcome',
    'NodeVerificationPrepareStoreCommand',
    'NodeVerificationPreparation',
    'NodeVerificationResumeStoreCommand',
    'NodeVerificationStoreProjection',
    'PreparedNodeVerification',
    'SettledNodeVerification',
    'SqliteNodeVerificationApplicationStore',
    'NodeVerificationResumeRequest',
    'NodeVerificationStartNextAttempt',
    "resume_and_run_node_verification",
    "run_node_verification",
]
