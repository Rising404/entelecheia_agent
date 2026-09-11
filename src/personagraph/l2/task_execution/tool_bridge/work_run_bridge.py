"""一个已启动 WorkRun Attempt 所用的 Tool Platform 桥接器。

此桥接器刻意比 WorkRun Controller 更窄。它只接受模型为当前 Attempt 提出的 ``call_tools`` 提案，
并负责从冻结 Catalog 到持久关闭 Attempt 的 Host 序列。它不启动 Attempt、不变更 OutputWindow、
不调用 Verifier、不创建批准记录，也不向 API 暴露结果。

``HostMaterializedToolCall.tool_version`` 在此只有一个精确含义：即已解析注册项的不可变
``implementation_version``。工具的 ``contract_version`` 仍由 Attempt Catalog 快照和已解析的
``ToolRegistration`` 绑定；两个版本绝不会拼接成臆造的标识符。
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, Callable, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from ....session import store as session_store
from ....session.l2_store import work_run as work_run_store
from ....session.l2_store.work_run import (
    StoredAttempt,
    StoredWorkRun,
)
from ....tools.catalog import CatalogSnapshot
from ....tools.contracts import (
    ExecutionOutcome,
    ExecutionStatus,
    ToolError,
    ToolSpec,
    thaw_json,
)
from ....tools.execution import (
    ResolvedInvocation,
    ToolExecutor,
    is_known_retryable_technical_failure,
)
from ....tools.policy import (
    AuthorityFacts,
    BudgetFacts,
    ToolPolicyCore,
)
from ....tools.schema_validation import ToolSchemaCompiler
from ...work_run import (
    AttemptDecision,
    AttemptStatus,
    CallToolsAction,
    HostAcceptedAttemptDecision,
    HostMaterializedCallToolsAction,
    ToolResultStatus,
    ToolResult,
    WorkExecutionMutationResult,
    WorkRunStatus,
    merge_acceptance_progress,
)
from ..attempts.active_time import AttemptActiveTimeMeter
from .attempt_contracts import (
    AttemptToolBridgePreflightRequest,
    AttemptToolBridgeRequest,
)
from ....model_io.output_validation import ModelOutputValidationError
from ....tools.policy import ProtectedToolExecutionAuthority
from .protected_dispatch import (
    ProtectedToolDispatchRequest,
    RuntimeProtectedToolDispatcher,
)
from .contracts import DurableToolResultObserver
from .persistence_contracts import (
    ToolBridgeCallPersistence,
    ToolBridgePersistencePlan,
)
from .preflight_contracts import (
    ToolBridgeMaterialized,
    ToolBridgePreflightResult,
    ToolBridgeRejected,
    ToolBridgeRejectionCode,
)
from .preflight import (
    _NON_MODIFYING_ACTIONS,
    _PreparedCall,
    _is_execution_findings_runtime_state,
    _prepare_calls,
    _rejected,
    preflight_and_materialize_call_tools_decision,
)
from ....tools.findings.dispatcher import execute_execution_findings_tool
from ..work_run.execution_findings import require_work_run_execution_findings


class _BridgeContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class ToolBridgeAttemptClosed(_BridgeContract):
    status: Literal["attempt_closed"] = "attempt_closed"
    decision: HostAcceptedAttemptDecision
    tool_results: tuple[ToolResult, ...] = Field(min_length=1)
    executed_tool_call_ids: tuple[str, ...] = ()
    replayed_tool_call_ids: tuple[str, ...] = ()
    mutation: WorkExecutionMutationResult


ToolBridgeResult = ToolBridgeAttemptClosed | ToolBridgeRejected
ToolBridgePersistencePlanFactory = Callable[
    [AttemptToolBridgeRequest],
    ToolBridgePersistencePlan,
]
ProtectedToolAuthorityKey = tuple[str, str]
ProtectedToolAuthorityMap = Mapping[
    ProtectedToolAuthorityKey,
    ProtectedToolExecutionAuthority,
]
_EMPTY_PROTECTED_TOOL_AUTHORITIES: ProtectedToolAuthorityMap = MappingProxyType({})


class ToolBridgeDispatchRejected(RuntimeError):
    """已预检的批次无法安全跨越执行边界。"""

    def __init__(self, rejection: ToolBridgeRejected) -> None:
        self.rejection = rejection
        super().__init__(
            f"Tool Bridge dispatch rejected: {rejection.code.value}: {rejection.message}"
        )


class _WorkExecutionStorePort(Protocol):
    def get_work_run(self, *, session_id: str, work_run_id: str) -> StoredWorkRun: ...

    def commit_work_run_attempt_decision(self, **kwargs: Any) -> WorkExecutionMutationResult: ...

    def append_work_run_tool_result(self, **kwargs: Any) -> WorkExecutionMutationResult: ...

    def close_work_run_attempt(self, **kwargs: Any) -> WorkExecutionMutationResult: ...


class _ExecutionFindingsStorePort(Protocol):
    def create_execution_findings_ledger(self, **kwargs: Any) -> object: ...

    def get_execution_findings_ledger_for_owner(self, **kwargs: Any) -> object: ...

    def apply_execution_findings_mutation(self, **kwargs: Any) -> object: ...


class SqliteWorkRunToolBridge:
    """:class:`AttemptToolBridge` 的具体两阶段适配器。

    真正的不可变 CatalogSnapshot 和显式持久化计划工厂是构造函数依赖。因此，Controller 绝不会
    臆造可执行注册项、结果 ID、收据 ID 或截止时间。
    """

    def __init__(
        self,
        *,
        catalog_snapshot: CatalogSnapshot,
        persistence_plan_factory: ToolBridgePersistencePlanFactory,
        authority: AuthorityFacts | None = None,
        budget: BudgetFacts | None = None,
        schemas: ToolSchemaCompiler | None = None,
        policy: ToolPolicyCore | None = None,
        executor: ToolExecutor | None = None,
        protected_dispatcher: RuntimeProtectedToolDispatcher | None = None,
        protected_authority_by_key: ProtectedToolAuthorityMap = (
            _EMPTY_PROTECTED_TOOL_AUTHORITIES
        ),
        durable_result_observer: DurableToolResultObserver | None = None,
        store: _WorkExecutionStorePort = work_run_store,
        execution_findings_store: _ExecutionFindingsStorePort = session_store,
    ) -> None:
        self._catalog_snapshot = catalog_snapshot
        self._persistence_plan_factory = persistence_plan_factory
        self._authority = authority
        self._budget = budget
        self._schemas = schemas
        self._policy = policy
        self._executor = executor
        self._protected_dispatcher = protected_dispatcher
        self._protected_authority_by_key = _freeze_protected_authorities(
            protected_authority_by_key,
            catalog_snapshot=catalog_snapshot,
        )
        if self._protected_authority_by_key and protected_dispatcher is None:
            raise ValueError(
                "protected tool authority requires a durable Operation dispatcher"
            )
        self._durable_result_observer = durable_result_observer
        self._store = store
        self._execution_findings_store = execution_findings_store

    @property
    def supports_protected_recovery(self) -> bool:
        """仅当无重发 Operation 账本存在时声明支持恢复。"""

        return self._protected_dispatcher is not None

    @property
    def authority(self) -> AuthorityFacts | None:
        """返回绑定到此桥接器的不可变策略事实。

        组装根只有在同时添加对应授权时，才能向已冻结的节点 Catalog 添加注册项。暴露此只读值
        可防止这些根深入桥接器内部，或意外地将更宽的 Catalog 重绑定到旧权威。
        """

        return self._authority

    def with_catalog_snapshot(
        self,
        catalog_snapshot: CatalogSnapshot,
    ) -> "SqliteWorkRunToolBridge":
        """在保留执行权威的同时重绑定不可变注册项。"""

        return SqliteWorkRunToolBridge(
            catalog_snapshot=catalog_snapshot,
            persistence_plan_factory=self._persistence_plan_factory,
            authority=self._authority,
            budget=self._budget,
            schemas=self._schemas,
            policy=self._policy,
            executor=self._executor,
            protected_dispatcher=self._protected_dispatcher,
            protected_authority_by_key=self._authorities_for_catalog(
                catalog_snapshot
            ),
            durable_result_observer=self._durable_result_observer,
            store=self._store,
            execution_findings_store=self._execution_findings_store,
        )

    def with_catalog_snapshot_and_authority(
        self,
        catalog_snapshot: CatalogSnapshot,
        *,
        authority: AuthorityFacts,
    ) -> "SqliteWorkRunToolBridge":
        """重绑定 Catalog 及其精确、显式组装的权威。

        此方法刻意与 :meth:`with_catalog_snapshot` 分离：旧辅助方法会为仅 Catalog 增强保留权威，
        而添加 Tool 能力必须让权威变更在调用点可见。
        """

        if not isinstance(authority, AuthorityFacts):
            raise TypeError("authority must be AuthorityFacts")
        return SqliteWorkRunToolBridge(
            catalog_snapshot=catalog_snapshot,
            persistence_plan_factory=self._persistence_plan_factory,
            authority=authority,
            budget=self._budget,
            schemas=self._schemas,
            policy=self._policy,
            executor=self._executor,
            protected_dispatcher=self._protected_dispatcher,
            protected_authority_by_key=self._authorities_for_catalog(
                catalog_snapshot
            ),
            durable_result_observer=self._durable_result_observer,
            store=self._store,
            execution_findings_store=self._execution_findings_store,
        )

    def with_catalog_snapshot_authority_and_protected_authorities(
        self,
        catalog_snapshot: CatalogSnapshot,
        *,
        authority: AuthorityFacts,
        additional_protected_authority_by_key: ProtectedToolAuthorityMap,
        protected_dispatcher: RuntimeProtectedToolDispatcher,
    ) -> "SqliteWorkRunToolBridge":
        """使用精确的额外受保护权威重绑定已扩大的 Catalog。

        能力组装只能将受保护注册项与其精确收据/后端/重新验证证明一同添加。现有精确权威会得到
        保留；任何修改型注册项都必须继续拥有精确映射权威。
        """

        if not isinstance(authority, AuthorityFacts):
            raise TypeError("authority must be AuthorityFacts")
        if not isinstance(
            protected_dispatcher, RuntimeProtectedToolDispatcher
        ):
            raise TypeError(
                "protected_dispatcher must be RuntimeProtectedToolDispatcher"
            )
        combined = dict(self._authorities_for_catalog(catalog_snapshot))
        for key, item in additional_protected_authority_by_key.items():
            previous = combined.get(key)
            if previous is not None and previous != item:
                raise ValueError("protected tool authority conflicts during rebind")
            combined[key] = item
        return SqliteWorkRunToolBridge(
            catalog_snapshot=catalog_snapshot,
            persistence_plan_factory=self._persistence_plan_factory,
            authority=authority,
            budget=self._budget,
            schemas=self._schemas,
            policy=self._policy,
            executor=self._executor,
            protected_dispatcher=(
                self._protected_dispatcher or protected_dispatcher
            ),
            protected_authority_by_key=combined,
            durable_result_observer=self._durable_result_observer,
            store=self._store,
            execution_findings_store=self._execution_findings_store,
        )

    def preflight(
        self,
        request: AttemptToolBridgePreflightRequest,
    ) -> HostAcceptedAttemptDecision:
        result = preflight_and_materialize_call_tools_decision(
            request.decision,
            catalog_snapshot=self._catalog_snapshot,
            stored_catalog_snapshot=request.catalog_snapshot,
            allowed_tools=request.allowed_tools,
            tool_call_ids=request.tool_call_ids,
            authority=self._authority,
            budget=self._budget,
            schemas=self._schemas,
            policy=self._policy,
            allow_protected_effects=self._protected_dispatcher is not None,
        )
        if isinstance(result, ToolBridgeRejected):
            raise ModelOutputValidationError(
                f"Tool Bridge preflight rejected: {result.code.value}"
            )
        authority_rejection = _guard_materialized_protected_authorities(
            result.decision,
            catalog_snapshot=self._catalog_snapshot,
            protected_authority_by_key=self._protected_authority_by_key,
        )
        if authority_rejection is not None:
            raise ModelOutputValidationError(
                "Tool Bridge preflight rejected: "
                f"{authority_rejection.code.value}"
            )
        return result.decision

    def dispatch(
        self,
        request: AttemptToolBridgeRequest,
        *,
        active_time_meter: AttemptActiveTimeMeter,
    ) -> WorkExecutionMutationResult:
        persistence = self._persistence_plan_factory(request)
        if persistence.decision_apply_id != request.apply_id:
            raise ValueError(
                "Tool Bridge persistence plan decision_apply_id must equal the Controller apply_id"
            )
        result = execute_materialized_call_tools_attempt(
            session_id=request.session_id,
            turn_id=request.turn_id,
            work_run_id=request.work_run_id,
            attempt_id=request.attempt_id,
            decision=request.decision,
            catalog_snapshot=self._catalog_snapshot,
            allowed_tools=request.allowed_tools,
            persistence=persistence,
            expected_work_run_revision=request.expected_work_run_revision,
            expected_progress_revision=request.expected_progress_revision,
            expected_window_revision=request.expected_window_revision,
            recovery_mutation=request.recovery_mutation,
            active_time_meter=active_time_meter,
            authority=self._authority,
            budget=self._budget,
            schemas=self._schemas,
            policy=self._policy,
            executor=self._executor,
            protected_dispatcher=self._protected_dispatcher,
            protected_authority_by_key=self._protected_authority_by_key,
            durable_result_observer=self._durable_result_observer,
            store=self._store,
            execution_findings_store=self._execution_findings_store,
        )
        if isinstance(result, ToolBridgeRejected):
            raise ToolBridgeDispatchRejected(result)
        return result.mutation

    def _authorities_for_catalog(
        self,
        catalog_snapshot: CatalogSnapshot,
    ) -> ProtectedToolAuthorityMap:
        authorities = self._protected_authority_by_key
        catalog_keys = {
            (entry.key.tool_id, entry.key.contract_version)
            for entry in catalog_snapshot.entries
        }
        return {
            key: authority
            for key, authority in authorities.items()
            if key in catalog_keys
        }


def execute_materialized_call_tools_attempt(
    *,
    session_id: str,
    turn_id: str,
    work_run_id: str,
    attempt_id: str,
    decision: HostAcceptedAttemptDecision,
    catalog_snapshot: CatalogSnapshot,
    allowed_tools: tuple[ToolSpec, ...],
    persistence: ToolBridgePersistencePlan,
    expected_work_run_revision: int,
    expected_progress_revision: int,
    expected_window_revision: int,
    active_time_meter: AttemptActiveTimeMeter,
    recovery_mutation: WorkExecutionMutationResult | None = None,
    authority: AuthorityFacts | None = None,
    budget: BudgetFacts | None = None,
    schemas: ToolSchemaCompiler | None = None,
    policy: ToolPolicyCore | None = None,
    executor: ToolExecutor | None = None,
    protected_dispatcher: RuntimeProtectedToolDispatcher | None = None,
    protected_authority_by_key: ProtectedToolAuthorityMap = (
        _EMPTY_PROTECTED_TOOL_AUTHORITIES
    ),
    durable_result_observer: DurableToolResultObserver | None = None,
    store: _WorkExecutionStorePort = work_run_store,
    execution_findings_store: _ExecutionFindingsStorePort = session_store,
) -> ToolBridgeResult:
    """持久化、执行并持久关闭一个已预检的 ``call_tools`` Attempt。

    输入已经是模型输出验证期间由 :func:`preflight_and_materialize_call_tools_decision`
    生成、经 Host 具体化的已接受批次。桥接器会在 Store 提交前，依据同一冻结 Catalog 确定性地
    复核该具体化结果。决定持久化后，每个 ``ExecutionOutcome`` 都会成为不可变
    ``ToolResult``，并由现有 Store 负责全部修订版本/状态转换。

    精确重放会跳过已经存在不可变 ToolResult 的调用。受保护调用只有在其语义 ToolCall 持久化后，
    才会抵达注入的分派器。该分派器持有逻辑/物理 Operation 账本；普通 READ/SEARCH 调用保留
    透明重试路径。
    """

    protected_authority_by_key = _freeze_protected_authorities(
        protected_authority_by_key,
        catalog_snapshot=catalog_snapshot,
    )
    if not isinstance(decision.action, HostMaterializedCallToolsAction):
        return _rejected(
            ToolBridgeRejectionCode.NOT_CALL_TOOLS,
            "The execution bridge accepts only a Host-materialized call_tools decision.",
        )
    if len(decision.action.calls) != len(persistence.calls):
        return _rejected(
            ToolBridgeRejectionCode.ID_PLAN_MISMATCH,
            "The Host persistence plan must cover every proposed call exactly once.",
        )
    if tuple(call.tool_call_id for call in decision.action.calls) != tuple(
        item.tool_call_id for item in persistence.calls
    ):
        return _rejected(
            ToolBridgeRejectionCode.ID_PLAN_MISMATCH,
            "The persistence plan must preserve materialized call identity and ordinal.",
        )

    record = store.get_work_run(session_id=session_id, work_run_id=work_run_id)
    attempt = _find_attempt(record, attempt_id)
    if attempt is None or attempt.turn_id != turn_id:
        return _rejected(
            ToolBridgeRejectionCode.ATTEMPT_NOT_CURRENT,
            "The requested Attempt does not belong to this WorkRun and Turn.",
        )
    if attempt.catalog_snapshot != catalog_snapshot.to_descriptor():
        return _rejected(
            ToolBridgeRejectionCode.CATALOG_SNAPSHOT_MISMATCH,
            "The supplied catalog snapshot is not the snapshot frozen for this Attempt.",
        )

    proposal_decision = AttemptDecision(
        acceptance_updates=decision.acceptance_updates,
        action=CallToolsAction(
            calls=tuple(
                {
                    "tool_id": call.tool_id,
                    "arguments": call.arguments,
                }
                for call in decision.action.calls
            )
        ),
    )
    preflight = preflight_and_materialize_call_tools_decision(
        proposal_decision,
        catalog_snapshot=catalog_snapshot,
        stored_catalog_snapshot=attempt.catalog_snapshot,
        allowed_tools=allowed_tools,
        tool_call_ids=tuple(call.tool_call_id for call in decision.action.calls),
        authority=authority,
        budget=budget,
        schemas=schemas,
        policy=policy,
        allow_protected_effects=protected_dispatcher is not None,
    )
    if isinstance(preflight, ToolBridgeRejected):
        return preflight
    if preflight.decision != decision:
        return _rejected(
            ToolBridgeRejectionCode.MATERIALIZED_DECISION_MISMATCH,
            "The supplied materialized batch does not match Host catalog/effect authority.",
        )
    authority_rejection = _guard_materialized_protected_authorities(
        decision,
        catalog_snapshot=catalog_snapshot,
        protected_authority_by_key=protected_authority_by_key,
    )
    if authority_rejection is not None:
        return authority_rejection
    prepared_or_rejection = _prepare_calls(
        proposal_decision.action,
        tool_call_ids=tuple(call.tool_call_id for call in decision.action.calls),
        snapshot=catalog_snapshot,
        schemas=schemas or ToolSchemaCompiler(),
        policy=policy or ToolPolicyCore(),
        authority=authority or AuthorityFacts(),
        budget=budget or BudgetFacts(),
        allow_protected_effects=protected_dispatcher is not None,
    )
    assert not isinstance(prepared_or_rejection, ToolBridgeRejected)
    prepared = tuple(
        _PreparedCall(
            ordinal=item.ordinal,
            registration=item.registration,
            arguments=item.arguments,
            modifies_environment=item.modifies_environment,
            tool_call_id=item.tool_call_id,
        )
        for item in prepared_or_rejection
    )

    if attempt.decision is None:
        if recovery_mutation is not None:
            return _rejected(
                ToolBridgeRejectionCode.ATTEMPT_NOT_CURRENT,
                "A recovery cursor cannot authorize an undecided Attempt.",
            )
        state_rejection = _guard_undecided_attempt(
            record,
            attempt=attempt,
            attempt_id=attempt_id,
            expected_work_run_revision=expected_work_run_revision,
            expected_progress_revision=expected_progress_revision,
            decision=proposal_decision,
        )
        if state_rejection is not None:
            return state_rejection
    elif attempt.decision != decision:
        return _rejected(
            ToolBridgeRejectionCode.ATTEMPT_ALREADY_DECIDED,
            "The Attempt already contains a different Host-accepted decision.",
        )

    if recovery_mutation is None:
        mutation = store.commit_work_run_attempt_decision(
            session_id=session_id,
            turn_id=turn_id,
            work_run_id=work_run_id,
            attempt_id=attempt_id,
            decision=decision,
            expected_work_run_revision=expected_work_run_revision,
            expected_progress_revision=expected_progress_revision,
            expected_window_revision=expected_window_revision,
            apply_id=persistence.decision_apply_id,
        )
    else:
        if (
            record.current_attempt_id != attempt_id
            or record.work_run.status is not WorkRunStatus.ACTIVE
            or record.work_run.reason is not None
            or attempt.attempt.status is not AttemptStatus.ACTIVE
            or attempt.turn_id != turn_id
            or record.work_run.revision != expected_work_run_revision
            or record.acceptance_progress.revision != expected_progress_revision
            or record.output_window.output_revision != recovery_mutation.output_window_revision
            or recovery_mutation.work_run_id != work_run_id
            or recovery_mutation.current_attempt_id != attempt_id
            or recovery_mutation.work_run_revision != expected_work_run_revision
            or recovery_mutation.acceptance_progress_revision
            != expected_progress_revision
            or recovery_mutation.window_state_version != expected_window_revision
            or recovery_mutation.work_run_status is not WorkRunStatus.ACTIVE
            or recovery_mutation.work_run_reason is not None
            or recovery_mutation.attempt is None
            or recovery_mutation.attempt.attempt_id != attempt_id
            or recovery_mutation.attempt.status is not AttemptStatus.ACTIVE
        ):
            return _rejected(
                ToolBridgeRejectionCode.ATTEMPT_NOT_CURRENT,
                "The decided-Attempt recovery cursor is stale or cross-bound.",
            )
        # 旧决定收据绑定到先前 Turn。重绑定变更是新的权威游标；以此 Turn 重放旧收据会正确地
        # 发生冲突，而不是推进状态。
        mutation = recovery_mutation
    persisted = store.get_work_run(session_id=session_id, work_run_id=work_run_id)
    existing_by_call = {
        result.tool_call_id: result
        for result in persisted.tool_results
        if result.attempt_id == attempt_id
    }
    mismatch = _guard_existing_results(
        prepared,
        existing_by_call,
        persistence,
        attempt_id=attempt_id,
    )
    if mismatch is not None:
        return mismatch
    tool_executor = executor or ToolExecutor(schemas=schemas)
    results: list[ToolResult] = []
    executed_call_ids: list[str] = []
    replayed_call_ids: list[str] = []
    for item in prepared:
        call_persistence = persistence.calls[item.ordinal - 1]
        existing = existing_by_call.get(item.tool_call_id)
        if existing is not None:
            if (
                recovery_mutation is not None
                and existing.status is ToolResultStatus.COMPLETION_UNCONFIRMED
                and (not item.modifies_environment or protected_dispatcher is None)
            ):
                return _rejected(
                    ToolBridgeRejectionCode.STORED_RESULT_MISMATCH,
                    "Only protected recovery can consume completion-uncertain results.",
                    call_ordinal=item.ordinal,
                    tool_id=item.registration.tool_id,
                )
            result = existing
            replayed_call_ids.append(item.tool_call_id)
        else:
            # 复用同一个已解析对象，使每次物理尝试都明确绑定到同一注册项/版本、规范化参数和
            # 截止时间。中间结果保持进程局部，绝不会成为模型可见的 ToolResult。
            invocation = ResolvedInvocation(
                registration=item.registration,
                arguments=item.arguments,
                deadline_monotonic=call_persistence.deadline_monotonic,
            )
            if _is_execution_findings_runtime_state(item.registration):
                findings = require_work_run_execution_findings(
                    session_id=session_id,
                    work_run_id=work_run_id,
                    acceptance_ids=tuple(
                        progress.acceptance_id
                        for progress in persisted.acceptance_progress.items
                    ),
                    store=execution_findings_store,
                )
                outcome = execute_execution_findings_tool(
                    store=execution_findings_store,
                    tool_id=item.registration.tool_id,
                    normalized_arguments=item.arguments,
                    ledger_id=findings.ledger_id,
                    writer_unit_id=attempt_id,
                    writer_tool_call_id=item.tool_call_id,
                )
            elif item.modifies_environment:
                if protected_dispatcher is None:
                    return _rejected(
                        ToolBridgeRejectionCode.MODIFYING_CALL_UNSUPPORTED,
                        "Protected ToolCall has no durable Operation dispatcher.",
                        call_ordinal=item.ordinal,
                        tool_id=item.registration.tool_id,
                    )
                protected_authority = _protected_authority_for_call(
                    item,
                    protected_authority_by_key=protected_authority_by_key,
                )
                state_guard = _protected_call_state_guard(
                    session_id=session_id,
                    work_run_id=work_run_id,
                    attempt_id=attempt_id,
                    decision=decision,
                    catalog_snapshot=catalog_snapshot,
                    call=item,
                    protected_authority=protected_authority,
                )

                def rederive_state_guard_sha256(
                    item: _PreparedCall = item,
                    protected_authority: ProtectedToolExecutionAuthority
                    | None = protected_authority,
                ) -> str:
                    return _rederive_protected_call_state_guard(
                        session_id=session_id,
                        work_run_id=work_run_id,
                        attempt_id=attempt_id,
                        decision=decision,
                        catalog_snapshot=catalog_snapshot,
                        call=item,
                        protected_authority=protected_authority,
                        store=store,
                    )

                outcome = protected_dispatcher.dispatch(
                    ProtectedToolDispatchRequest(
                        session_id=session_id,
                        turn_id=turn_id,
                        invocation_turn_id=attempt.input_turn_id,
                        work_run_id=work_run_id,
                        attempt_id=attempt_id,
                        call_ordinal=item.ordinal,
                        tool_call_id=item.tool_call_id,
                        catalog_snapshot=catalog_snapshot,
                        registration=item.registration,
                        arguments=item.arguments,
                        deadline_monotonic=call_persistence.deadline_monotonic,
                        state_guard_sha256=state_guard,
                        rederive_state_guard_sha256=(
                            rederive_state_guard_sha256
                        ),
                        provider_identity_sha256=(
                            None
                            if protected_authority is None
                            else protected_authority.execution_backend_identity_sha256
                        ),
                    ),
                    executor=tool_executor,
                )
            else:
                outcome = _execute_with_transparent_physical_retry(
                    tool_executor,
                    invocation,
                )
            result = tool_result_from_execution_outcome(
                outcome,
                tool_result_id=call_persistence.tool_result_id,
                tool_call_id=item.tool_call_id,
                attempt_id=attempt_id,
                ordinal=item.ordinal,
            )
            executed_call_ids.append(item.tool_call_id)

        if existing is None or recovery_mutation is None:
            mutation = store.append_work_run_tool_result(
                session_id=session_id,
                turn_id=turn_id,
                work_run_id=work_run_id,
                result=result,
                expected_work_run_revision=mutation.work_run_revision,
                expected_progress_revision=mutation.acceptance_progress_revision,
                expected_window_revision=mutation.window_state_version,
                apply_id=call_persistence.result_apply_id,
            )
        if durable_result_observer is not None:
            durable_result_observer(
                session_id=session_id,
                turn_id=turn_id,
                work_run_id=work_run_id,
                attempt_id=attempt_id,
                result=result,
            )
        results.append(result)

    mutation = store.close_work_run_attempt(
        session_id=session_id,
        turn_id=turn_id,
        work_run_id=work_run_id,
        attempt_id=attempt_id,
        expected_work_run_revision=mutation.work_run_revision,
        expected_progress_revision=mutation.acceptance_progress_revision,
        expected_window_revision=mutation.window_state_version,
        apply_id=persistence.close_apply_id,
        active_seconds_delta=active_time_meter.freeze(),
    )
    return ToolBridgeAttemptClosed(
        decision=decision,
        tool_results=tuple(results),
        executed_tool_call_ids=tuple(executed_call_ids),
        replayed_tool_call_ids=tuple(replayed_call_ids),
        mutation=mutation,
    )


def _execute_with_transparent_physical_retry(
    executor: ToolExecutor,
    invocation: ResolvedInvocation,
) -> ExecutionOutcome:
    """运行一次只读逻辑调用：初始调用一次，外加最多 3 次重试。

    只有终态、已知技术失败符合条件。策略/Catalog/输入拒绝发生在此边界之前；取消、业务失败和
    完成不确定性会立即返回。崩溃安全的尝试序号需要持久收据 schema，此处刻意不声称具备该能力。
    """

    if any(
        descriptor.action not in _NON_MODIFYING_ACTIONS
        for descriptor in invocation.registration.effect_profile.effects
    ):
        raise AssertionError(
            "the transparent retry bridge accepts READ/SEARCH tools only"
        )

    max_retries = invocation.registration.execution_profile.max_transparent_retries
    for physical_attempt in range(1, max_retries + 2):
        outcome = executor.execute(invocation)
        if outcome.status is ExecutionStatus.COMPLETION_UNCONFIRMED:
            # READ/SEARCH 没有需要协调的持久环境效果。因此，响应丢失属于已知不可用结果，
            # 而不是等待外部观察的有副作用 Operation。
            assert outcome.error is not None
            outcome = ExecutionOutcome(
                ExecutionStatus.FAILED,
                error=ToolError(
                    "read_only_response_unavailable",
                    outcome.error.message,
                    {"reported_code": outcome.error.code},
                ),
                metadata=outcome.metadata,
            )
        if (
            not is_known_retryable_technical_failure(outcome)
            or physical_attempt > max_retries
        ):
            return outcome
    raise AssertionError("transparent physical retry loop did not terminate")


def tool_result_from_execution_outcome(
    outcome: ExecutionOutcome,
    *,
    tool_result_id: str,
    tool_call_id: str,
    attempt_id: str,
    ordinal: int,
) -> ToolResult:
    """映射全部六种 Tool Platform 结果，同时保留其差异。"""

    status = ToolResultStatus(outcome.status.value)
    if outcome.status is ExecutionStatus.SUCCEEDED:
        return ToolResult(
            status=status,
            tool_result_id=tool_result_id,
            tool_call_id=tool_call_id,
            attempt_id=attempt_id,
            ordinal=ordinal,
            output=thaw_json(outcome.result),
        )
    assert outcome.error is not None  # 由 ExecutionOutcome 保证
    return ToolResult(
        status=status,
        tool_result_id=tool_result_id,
        tool_call_id=tool_call_id,
        attempt_id=attempt_id,
        ordinal=ordinal,
        output=None,
        error_code=outcome.error.code,
        error_message=outcome.error.message,
    )


def _freeze_protected_authorities(
    authorities: ProtectedToolAuthorityMap,
    *,
    catalog_snapshot: CatalogSnapshot,
) -> ProtectedToolAuthorityMap:
    """在组装时复制并验证精确注册权威。"""

    if not isinstance(authorities, Mapping):
        raise TypeError("protected_authority_by_key must be a mapping")
    catalog_keys = {
        (entry.key.tool_id, entry.key.contract_version)
        for entry in catalog_snapshot.entries
    }
    frozen: dict[ProtectedToolAuthorityKey, ProtectedToolExecutionAuthority] = {}
    for key, authority in authorities.items():
        if (
            not isinstance(key, tuple)
            or len(key) != 2
            or any(not isinstance(value, str) or not value for value in key)
        ):
            raise ValueError("protected tool authority key is invalid")
        if key not in catalog_keys:
            raise ValueError("protected tool authority references an unknown registration")
        if not isinstance(authority, ProtectedToolExecutionAuthority):
            raise TypeError(
                "protected tool authority must be ProtectedToolExecutionAuthority"
            )
        frozen[key] = authority
    return MappingProxyType(frozen)


def _guard_materialized_protected_authorities(
    decision: HostAcceptedAttemptDecision,
    *,
    catalog_snapshot: CatalogSnapshot,
    protected_authority_by_key: ProtectedToolAuthorityMap,
) -> ToolBridgeRejected | None:
    """要求每个受保护调用都具备精确注册权威。"""
    for ordinal, call in enumerate(decision.action.calls, start=1):
        if not call.modifies_environment:
            continue
        try:
            registration = catalog_snapshot.resolve(call.tool_id)
        except ValueError:
            return _rejected(
                ToolBridgeRejectionCode.CATALOG_SNAPSHOT_MISMATCH,
                "Protected ToolCall no longer resolves in the frozen catalog.",
                call_ordinal=ordinal,
                tool_id=call.tool_id,
            )
        key = (registration.tool_id, registration.contract_version)
        if key not in protected_authority_by_key:
            return _rejected(
                ToolBridgeRejectionCode.MODIFYING_CALL_UNSUPPORTED,
                "Protected ToolCall has no exact registration authority.",
                call_ordinal=ordinal,
                tool_id=call.tool_id,
            )
    return None


def _protected_authority_for_call(
    call: _PreparedCall,
    *,
    protected_authority_by_key: ProtectedToolAuthorityMap,
) -> ProtectedToolExecutionAuthority:
    key = (call.registration.tool_id, call.registration.contract_version)
    authority = protected_authority_by_key.get(key)
    if authority is None:
        # 精确守卫在决定持久化前运行。抵达此分支意味着进程内不变量遭到破坏，而不是一种可以
        # 安全搁置已决定 Attempt 的新模型可见拒绝。
        raise RuntimeError("protected tool exact authority disappeared")
    return authority


def _protected_call_state_guard(
    *,
    session_id: str,
    work_run_id: str,
    attempt_id: str,
    decision: HostAcceptedAttemptDecision,
    catalog_snapshot: CatalogSnapshot,
    call: _PreparedCall,
    protected_authority: ProtectedToolExecutionAuthority,
) -> str:
    """将一次物理受保护分派绑定到其已决定 Attempt 游标。"""

    payload: dict[str, object] = {
        "schema_version": "work-run-protected-tool-state-guard-v2",
        "session_id": session_id,
        "work_run_id": work_run_id,
        "attempt_id": attempt_id,
        "catalog": catalog_snapshot.to_descriptor(),
        "decision": decision.model_dump(mode="json"),
        "tool_call_id": call.tool_call_id,
        "tool_id": call.registration.tool_id,
        "tool_version": call.registration.implementation_version,
        "arguments": call.arguments,
        "protected_authority_binding_sha256": (
            protected_authority.binding_sha256
        ),
    }
    return _canonical_sha256(payload)


def _rederive_protected_call_state_guard(
    *,
    session_id: str,
    work_run_id: str,
    attempt_id: str,
    decision: HostAcceptedAttemptDecision,
    catalog_snapshot: CatalogSnapshot,
    call: _PreparedCall,
    protected_authority: ProtectedToolExecutionAuthority,
    store: _WorkExecutionStorePort,
) -> str:
    try:
        if protected_authority.revalidate() is not True:
            return "0" * 64
    except Exception:
        return "0" * 64
    try:
        record = store.get_work_run(
            session_id=session_id,
            work_run_id=work_run_id,
        )
        attempts = tuple(
            item
            for item in record.attempts
            if item.attempt.attempt_id == attempt_id
        )
        results = tuple(
            item
            for item in record.tool_results
            if item.attempt_id == attempt_id
            and item.tool_call_id == call.tool_call_id
        )
        if (
            len(attempts) != 1
            or results
            or record.current_attempt_id != attempt_id
            or record.work_run.status is not WorkRunStatus.ACTIVE
            or record.work_run.reason is not None
            or attempts[0].attempt.status is not AttemptStatus.ACTIVE
            or attempts[0].decision != decision
            or attempts[0].catalog_snapshot != catalog_snapshot.to_descriptor()
        ):
            return "0" * 64
    except Exception:
        return "0" * 64
    return _protected_call_state_guard(
        session_id=session_id,
        work_run_id=work_run_id,
        attempt_id=attempt_id,
        decision=decision,
        catalog_snapshot=catalog_snapshot,
        call=call,
        protected_authority=protected_authority,
    )


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _guard_undecided_attempt(
    record: StoredWorkRun,
    *,
    attempt: StoredAttempt,
    attempt_id: str,
    expected_work_run_revision: int,
    expected_progress_revision: int,
    decision: AttemptDecision,
) -> ToolBridgeRejected | None:
    if (
        record.current_attempt_id != attempt_id
        or record.work_run.status is not WorkRunStatus.ACTIVE
        or record.work_run.reason is not None
        or attempt.attempt.status is not AttemptStatus.ACTIVE
        or attempt.action is not None
    ):
        return _rejected(
            ToolBridgeRejectionCode.ATTEMPT_NOT_CURRENT,
            "Only the current undecided active Attempt may accept a new proposal.",
        )
    if (
        record.work_run.revision != expected_work_run_revision
        or record.acceptance_progress.revision != expected_progress_revision
    ):
        return _rejected(
            ToolBridgeRejectionCode.ATTEMPT_NOT_CURRENT,
            "The supplied WorkRun or Acceptance progress revision is stale.",
            details={
                "actual_work_run_revision": record.work_run.revision,
                "actual_progress_revision": record.acceptance_progress.revision,
            },
        )
    closed_attempt_ids = {
        item.attempt.attempt_id
        for item in record.attempts
        if item.attempt.status is AttemptStatus.CLOSED
    }
    historical_result_ids = {
        result.tool_result_id
        for result in record.tool_results
        if result.attempt_id in closed_attempt_ids
        and result.status is ToolResultStatus.SUCCEEDED
    }
    merged = merge_acceptance_progress(
        record.acceptance_progress,
        decision.acceptance_updates,
        known_historical_tool_result_ids=historical_result_ids,
        expected_progress_revision=expected_progress_revision,
        current_work_run_revision=record.work_run.revision,
        expected_work_run_revision=expected_work_run_revision,
    )
    if merged.status != "applied":
        return _rejected(
            ToolBridgeRejectionCode.INVALID_ACCEPTANCE_UPDATES,
            "Acceptance updates failed the deterministic progress guard.",
            details={"codes": [issue.code.value for issue in merged.issues]},
        )
    return None


def _guard_existing_results(
    prepared: tuple[_PreparedCall, ...],
    existing_by_call: dict[str, ToolResult],
    persistence: ToolBridgePersistencePlan,
    *,
    attempt_id: str,
) -> ToolBridgeRejected | None:
    expected_call_ids = {item.tool_call_id for item in prepared}
    if set(existing_by_call) - expected_call_ids:
        return _rejected(
            ToolBridgeRejectionCode.STORED_RESULT_MISMATCH,
            "The Attempt contains a ToolResult outside the materialized call batch.",
        )
    for item in prepared:
        existing = existing_by_call.get(item.tool_call_id)
        if existing is None:
            continue
        call_persistence = persistence.calls[item.ordinal - 1]
        if (
            existing.tool_result_id != call_persistence.tool_result_id
            or existing.tool_call_id != call_persistence.tool_call_id
            or existing.attempt_id != attempt_id
            or existing.ordinal != item.ordinal
        ):
            return _rejected(
                ToolBridgeRejectionCode.STORED_RESULT_MISMATCH,
                "A stored ToolResult does not match the supplied Host identity plan.",
                call_ordinal=item.ordinal,
                tool_id=item.registration.tool_id,
            )
    return None


def _find_attempt(
    record: StoredWorkRun,
    attempt_id: str,
) -> StoredAttempt | None:
    return next(
        (
            item
            for item in record.attempts
            if item.attempt.attempt_id == attempt_id
        ),
        None,
    )


__all__ = [
    "SqliteWorkRunToolBridge",
    "DurableToolResultObserver",
    "ToolBridgeAttemptClosed",
    'ToolBridgeCallPersistence',
    "ToolBridgeDispatchRejected",
    'ToolBridgeMaterialized',
    'ToolBridgePersistencePlan',
    "ToolBridgePersistencePlanFactory",
    'ToolBridgePreflightResult',
    'ToolBridgeRejected',
    "ToolBridgeRejectionCode",
    "ToolBridgeResult",
    "execute_materialized_call_tools_attempt",
    "preflight_and_materialize_call_tools_decision",
    "tool_result_from_execution_outcome",
]
