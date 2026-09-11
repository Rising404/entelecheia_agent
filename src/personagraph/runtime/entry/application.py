"""权威 L0/L1/L2 Runtime 入口路径。
【turn入口的重点模块】
entry 持有一个狭窄生命周期：恰好一次接受可信用户 Turn，运行确定性 ingress 与有界
classification/generation，随后以原子方式提交一份正式交付。
"""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
import os
from threading import Lock
from typing import Any, Callable, Literal, Mapping
from uuid import uuid4

from ...context_budget import ContextBudgetExceeded
from ...configuration.features import DEFAULT_TURN_WALL_CLOCK_BUDGET_S
from ...persistent_turn_content.delivery import (
    L1TerminalNotification,
    build_l1_terminal_notification,
)
from personagraph.model_io.gateway import ModelGatewayError
from ...session.turn_execution_contracts import TurnExecutionBusyError
from ...trajectory.scope import turn_linkage_scope
from ..concurrency import is_session_run_active, session_run_guard
from ..concurrency import SessionRunBusyError
from .context.contracts import EntryContext
from .ingress.model_contracts import EntryClassification
from .lifecycle.contracts import EntryAcceptedEmitter
from .lifecycle.ports import (
    EntryReplayStorePort,
    EntryTurnAcceptanceStorePort,
    EntryTurnFinalizationStorePort,
    EntryTurnMutationStorePort,
)
from .ports import EntryApplicationStorePort
from ..turn.contracts import (
    AcceptedEntryTurn,
    EntryExecutionSnapshot,
    EntryTurnResult,
    ProcessingRouteAuthorityError,
)
from .lifecycle.active_window import (
    load_authoritative_active_turn_window as _authoritative_active_turn_window,
)
from .ingress.classification import run_entry_classification_stage
from .context.application import build_entry_context
from .ingress.policy import select_entry_ingress_short_circuit_outcome
from .ingress.model import classify_turn
from .response.model import generate_response
from .lifecycle.persisted_turn import (
    accepted_entry_turn_from_l1_recovery_inspection as _accepted_l1_recovery,
    accepted_entry_turn_from_persisted_receipt,
)
from ..turn.persisted_projection import (
    optional_text as _optional_text,
    require_entry_turn_status as _entry_turn_status,
    require_entry_window_state,
    require_persisted_mapping as _required_mapping,
    require_processing_level as _processing_level,
)
from .lifecycle.replay import reconcile_replayed_entry_turn
from .lifecycle.settlement import recover_completed_entry_turn_from_commit
from .routing.task_admission import admit_entry_task_matches
from .routing.selection import (
    EntryL1ProcessingRoute,
    EntryL2TaskProcessingRoute,
    EntryProcessingRoute,
    EntryResponseProcessingRoute,
    select_entry_processing_route,
)
from .ingress.contracts import (
    IngressDecision,
    IngressDisposition,
    IngressHandler,
    evaluate_ingress,
)
from ..turn_deadline import TurnDeadline, TurnDeadlineExceeded
from ..model_calls.authority import RuntimeModelCallWaitingExternal
from ..model_calls.observability import runtime_error_code as _public_model_error_code
from ..turn_events import (
    EntryEventEmitter,
    RuntimeErrorCode,
    RuntimeStage,
    TurnEventStatus,
    TurnEvent,
    new_turn_event,
    project_turn_event,
)
from .lifecycle.window_audit import audit_turn_window_before_accept
from .routing.policy import (
    DEFAULT_TURN_ROUTING_POLICY_SNAPSHOT,
    ProcessingLevel,
    TurnRoutingPolicySnapshot,
    canonical_policy_json,
    canonical_snapshot_json,
    policy_sha256,
    snapshot_sha256,
)

_RUNTIME_ENTRY_LEASE_OWNER: str | None = None
_RUNTIME_ENTRY_LEASE_OWNER_PID: int | None = None
_RUNTIME_ENTRY_LEASE_OWNER_LOCK = Lock()
# Entry 只有 API 组合根；该常量仅满足仍在现行 schema 中的历史审计列。
_PERSISTED_TURN_SOURCE = "api"
_TURN_POST_COMMIT_JOB_KINDS: ContextVar[tuple[str, ...]] = ContextVar(
    "personagraph_turn_post_commit_job_kinds",
    default=("session_summary",),
)


class _AuxiliaryTurnLeaseLost(RuntimeError):
    """辅助执行返回时，当前 Turn 已不再由本 worker 持有。"""


@dataclass(frozen=True, slots=True)
class _EntryProcessingScope:
    """各 processing lane 共享的 Turn 生命周期权威。"""

    accepted: AcceptedEntryTurn
    context: EntryContext
    related_insession_task_ids: tuple[str, ...]
    deadline: TurnDeadline
    features: dict[str, Any]
    emit: EntryEventEmitter
    advance: Callable[[RuntimeStage, int | None], int]
    store: EntryApplicationStorePort
    lease_owner: str
    supervisor_completed_event_sequence: int | None


def _runtime_entry_lease_owner() -> str:
    """返回 Runtime 进程专用的持久化 Window 租约标识。

    模块级 UUID 会由预 fork 子进程继承。只为当前 PID 缓存所有者，使不同 Runtime
    进程无法伪装成同一持久化 Window 所有者。
    """

    global _RUNTIME_ENTRY_LEASE_OWNER, _RUNTIME_ENTRY_LEASE_OWNER_PID
    pid = os.getpid()
    with _RUNTIME_ENTRY_LEASE_OWNER_LOCK:
        if (
            _RUNTIME_ENTRY_LEASE_OWNER is None
            or _RUNTIME_ENTRY_LEASE_OWNER_PID != pid
        ):
            _RUNTIME_ENTRY_LEASE_OWNER = f"runtime-entry-{pid}-{uuid4().hex}"
            _RUNTIME_ENTRY_LEASE_OWNER_PID = pid
        return _RUNTIME_ENTRY_LEASE_OWNER


def run_entry_turn(
    *,
    user_input: str,
    features: dict[str, Any],
    session_id: str,
    client_request_id: str | None = None,
    on_stream_event: Callable[[dict[str, Any]], None] | None = None,
    on_turn_accepted: EntryAcceptedEmitter | None = None,
    attachment_ids: tuple[str, ...] = (),
    routing_policy: TurnRoutingPolicySnapshot | None = None,
    persist_routing_policy_as_session_default: bool = False,
    store: EntryApplicationStorePort,
) -> EntryTurnResult:
    """Runtime 主入口：先识别 request replay，再冻结新 Turn 的执行快照。

    新请求把 features、检索 generation、历史截止点和提交后 job 种类一起冻结；
    重复 client_request_id 复用已接受事实，不能用当前配置改变原 Turn。
    下一跳 _run_entry_turn_with_execution_snapshot 负责 Session guard、接受和执行；
    本函数不选择 L1 的下一步动作。
    """

    request_id = client_request_id or f"runtime_{uuid4().hex}"
    # 重复请求必须先恢复已接受快照。在查找前引导调用方当前 generation，可能让重试
    # 改变仍在运行的原始 Turn 下方的 ACTIVE 检索状态。
    existing = (
        store.get_turn_execution_for_client_request(
            session_id=session_id,
            client_request_id=request_id,
        )
        if client_request_id is not None
        else None
    )
    execution_snapshot = None
    if existing is None:
        session_binding = _prepare_session_retrieval_for_turn(
            features,
            session_id=session_id,
            store=store,
        )
        if (
            session_binding is not None
            and features.get("history_retrieval_write_enabled", False) is True
        ):
            from ..post_commit.session_retrieval_recovery import (
                reconcile_session_retrieval_before_turn,
            )

            reconcile_session_retrieval_before_turn(
                session_id=session_id,
                store=store,
                binding=session_binding,
            )
        file_binding = _prepare_file_retrieval_for_turn(features)
        execution_snapshot = EntryExecutionSnapshot.create(
            features=features,
            post_commit_job_kinds=_post_commit_job_kinds(features),
            file_retrieval_data_version=(
                file_binding.data_version_id
                if file_binding is not None
                else None
            ),
            session_retrieval_data_version=(
                session_binding.data_version_id
                if session_binding is not None
                else None
            ),
            session_retrieval_assistant_turn_cutoff=(
                session_binding.assistant_turn_cutoff
                if session_binding is not None
                else None
            ),
        )
    return _run_entry_turn_with_execution_snapshot(
        user_input=user_input,
        features=features,
        execution_snapshot=execution_snapshot,
        session_id=session_id,
        client_request_id=request_id,
        on_stream_event=on_stream_event,
        on_turn_accepted=on_turn_accepted,
        attachment_ids=attachment_ids,
        routing_policy=routing_policy,
        persist_routing_policy_as_session_default=(
            persist_routing_policy_as_session_default
        ),
        store=store,
    )


def _prepare_file_retrieval_for_turn(features: Mapping[str, Any]) -> object | None:
    """Keep the optional document stack outside Entry/L1 cold startup."""

    if features.get("file_retrieval_read_enabled", False) is not True:
        return None
    from ...retrieval.operations.turn_binding import prepare_file_retrieval_for_turn

    return prepare_file_retrieval_for_turn(features=features)


def _prepare_session_retrieval_for_turn(
    features: Mapping[str, Any],
    *,
    session_id: str,
    store: object,
) -> object | None:
    """冻结 L1 Session 配方与截止点；实际派生回填留在工具组合阶段。"""

    if not (
        features.get("history_retrieval_read_enabled", False) is True
        or features.get("history_retrieval_write_enabled", False) is True
    ):
        return None
    from ...retrieval.sources.session.composition import (
        prepare_session_retrieval_binding,
    )

    return prepare_session_retrieval_binding(
        session_id=session_id,
        store=store,
    )


def _post_commit_job_kinds(features: Mapping[str, Any]) -> tuple[str, ...]:
    jobs = ["session_summary"]
    if features.get("history_retrieval_write_enabled", False) is True:
        jobs.append("session_retrieval_index")
    return tuple(sorted(jobs))


def _run_entry_turn_with_execution_snapshot(
    *,
    user_input: str,
    features: dict[str, Any],
    execution_snapshot: EntryExecutionSnapshot | None,
    session_id: str,
    client_request_id: str | None = None,
    on_stream_event: Callable[[dict[str, Any]], None] | None = None,
    on_turn_accepted: EntryAcceptedEmitter | None = None,
    attachment_ids: tuple[str, ...] = (),
    routing_policy: TurnRoutingPolicySnapshot | None = None,
    persist_routing_policy_as_session_default: bool = False,
    store: EntryApplicationStorePort,
) -> EntryTurnResult:
    """接受并执行一个 Turn，同时保持 request-id 幂等性。

    公开 API 始终提供 ``client_request_id``。可选默认值只支持聚焦 Runtime 测试与旧版
    程序调用方；它们的调用边界已是进程内边界，而非 HTTP 重放契约。

    先拿进程内 guard 再接受，避免 acceptance→执行之间被另一线程误判为可恢复。
    guard 忙时只允许已存在的同 request-id 重读准入回执；不同请求直接拒绝，不能
    趁窗口尚未持久化的间隙插入孤立 Turn。持有 guard 的正常分支才可尝试 L1 恢复。
    """

    request_id = client_request_id or f"runtime_{uuid4().hex}"
    accepted_routing_policy = (
        routing_policy or DEFAULT_TURN_ROUTING_POLICY_SNAPSHOT
    )
    try:
        # 在接受持久化 Turn *之前*认领本地执行租约。这样，即使在 acceptance 到执行的
        # 极短间隙内，竞争 HTTP 重试也能安全区分活动本地 run 与过期 Window。
        with session_run_guard(session_id):
            accepted = accept_entry_turn(
                user_input=user_input,
                session_id=session_id,
                client_request_id=request_id,
                attachment_ids=attachment_ids,
                execution_snapshot=execution_snapshot,
                routing_policy=accepted_routing_policy,
                persist_routing_policy_as_session_default=(
                    persist_routing_policy_as_session_default
                ),
                has_existing_local_execution=False,
                store=store,
            )
            _emit_turn_accepted(on_turn_accepted, accepted)
            return execute_accepted_entry_turn(
                accepted=accepted,
                on_stream_event=on_stream_event,
                store=store,
                session_lease_held=True,
            )
    except SessionRunBusyError:
        # 允许重复 POST 观察原始已接受 Turn。本地执行租约活动时，不要为不同请求调用
        # acceptance 写入路径：在运行租约持久化 Window 前的短暂间隔内，这可能创建
        # 孤立 Window。
        if store.get_turn_execution_for_client_request(
            session_id=session_id,
            client_request_id=request_id,
        ) is None:
            raise
        accepted = accept_entry_turn(
            user_input=user_input,
            session_id=session_id,
            client_request_id=request_id,
            attachment_ids=attachment_ids,
            execution_snapshot=execution_snapshot,
            routing_policy=accepted_routing_policy,
            persist_routing_policy_as_session_default=(
                persist_routing_policy_as_session_default
            ),
            has_existing_local_execution=True,
            store=store,
        )
        _emit_turn_accepted(on_turn_accepted, accepted)
        return execute_accepted_entry_turn(
            accepted=accepted,
            on_stream_event=on_stream_event,
            store=store,
        )


def accept_entry_turn(
    *,
    user_input: str,
    session_id: str,
    client_request_id: str,
    attachment_ids: tuple[str, ...],
    execution_snapshot: EntryExecutionSnapshot | None = None,
    routing_policy: TurnRoutingPolicySnapshot = (
        DEFAULT_TURN_ROUTING_POLICY_SNAPSHOT
    ),
    persist_routing_policy_as_session_default: bool = False,
    store: EntryTurnAcceptanceStorePort,
    has_existing_local_execution: bool | None = None,
) -> AcceptedEntryTurn:
    """在模型调用前，将用户输入、附件绑定和执行/路由快照交给 Store 持久准入。

    返回的 AcceptedEntryTurn 是后续上下文和恢复的可信输入（accepted facts）。
    request-id 重放检查先于窗口占用检查；窗口忙时先审计，再尝试同一准入命令。
    此处不生成答案，也不把请求中的附件 ID 直接视作已获准的文件内容。
    """

    # Storage 先检查现有 request id，再检查 Window 占用。此顺序至关重要：活动 run
    # 期间的重复 POST 必须返回其原始 Turn，而不是尝试恢复或重新开始。
    audit = None
    lease_owner = _runtime_entry_lease_owner()
    policy_json = canonical_policy_json(routing_policy.policy)
    policy_hash = policy_sha256(routing_policy.policy)
    policy_write = {
        "routing_policy_source": routing_policy.source,
        "routing_policy_snapshot_json": canonical_snapshot_json(routing_policy),
        "routing_policy_snapshot_hash": snapshot_sha256(routing_policy),
        "session_routing_policy_json": (
            policy_json if persist_routing_policy_as_session_default else None
        ),
        "session_routing_policy_hash": (
            policy_hash if persist_routing_policy_as_session_default else None
        ),
        "execution_snapshot_json": (
            execution_snapshot.to_json()
            if execution_snapshot is not None
            else None
        ),
        "execution_snapshot_sha256": (
            execution_snapshot.sha256
            if execution_snapshot is not None
            else None
        ),
    }
    try:
        persisted = store.accept_turn_execution(
            session_id=session_id,
            client_request_id=client_request_id,
            source=_PERSISTED_TURN_SOURCE,
            user_text=user_input,
            attachment_ids=attachment_ids,
            lease_owner=lease_owner,
            initial_stage=RuntimeStage.INGRESS.value,
            **policy_write,
        )
    except TurnExecutionBusyError:
        audit = audit_turn_window_before_accept(
            session_id=session_id,
            store=store,
            has_local_execution=(
                is_session_run_active(session_id)
                if has_existing_local_execution is None
                else has_existing_local_execution
            ),
        )
        persisted = store.accept_turn_execution(
            session_id=session_id,
            client_request_id=client_request_id,
            source=_PERSISTED_TURN_SOURCE,
            user_text=user_input,
            attachment_ids=attachment_ids,
            lease_owner=lease_owner,
            initial_stage=RuntimeStage.INGRESS.value,
            **policy_write,
        )
    return accepted_entry_turn_from_persisted_receipt(
        persisted=persisted,
        session_id=session_id,
        client_request_id=client_request_id,
        recovery_projection_supplier=(
            (lambda: audit.recovery_projection) if audit is not None else None
        ),
    )


def execute_accepted_entry_turn(
    *,
    accepted: AcceptedEntryTurn,
    on_stream_event: Callable[[dict[str, Any]], None] | None,
    store: EntryApplicationStorePort,
    session_lease_held: bool = False,
) -> EntryTurnResult:
    """执行新 Turn，或安全重新进入其精确未完成 L1 lane。

    另一地方 executor 活动时观察到的重复请求保持只读。只有已持有进程内 Session
    guard 的调用方才可请求 Store 隔离并恢复由同一 request-id 持有的 L1 聚合。

    trajectory linkage 在这里绑定到已接受的 session_id / turn_id。它只传递记录归属，
    不授予执行租约；线程池还须显式 copy_context，后台 job 则需重新绑定原 Turn。
    """

    with turn_linkage_scope(
        session_id=accepted.session_id,
        turn_id=accepted.turn_id,
    ):
        return _execute_accepted_entry_turn_scoped(
            accepted=accepted,
            on_stream_event=on_stream_event,
            store=store,
            session_lease_held=session_lease_held,
        )


def _execute_accepted_entry_turn_scoped(
    *,
    accepted: AcceptedEntryTurn,
    on_stream_event: Callable[[dict[str, Any]], None] | None,
    store: EntryApplicationStorePort,
    session_lease_held: bool,
) -> EntryTurnResult:
    """在原 Turn trajectory scope 内，按已接受快照选择重放、恢复或新执行。

    features 和 post-commit job kinds 均从 execution_snapshot 读取；后者用 ContextVar
    临时传给 finalizer，finally 恢复外层值。replayed 默认走持久结果投影；只有已持有
    本地 Session guard 时才尝试认领原 L1 run，不能把每次重复 POST 当作重新执行。
    """

    snapshot = accepted.execution_snapshot
    effective_features = snapshot.features
    post_commit_jobs = snapshot.post_commit_job_kinds
    jobs_token = _TURN_POST_COMMIT_JOB_KINDS.set(post_commit_jobs)
    try:
        if accepted.replayed:
            if session_lease_held:
                resumed = _try_resume_replayed_l1_turn(
                    accepted=accepted,
                    features=effective_features,
                    on_stream_event=on_stream_event,
                    store=store,
                )
                if resumed is not None:
                    return resumed
            return _replayed_turn_result(
                accepted=accepted,
                store=store,
            )

        if session_lease_held:
            return _execute_new_turn(
                accepted=accepted,
                features=effective_features,
                on_stream_event=on_stream_event,
                store=store,
            )
        with session_run_guard(accepted.session_id):
            return _execute_new_turn(
                accepted=accepted,
                features=effective_features,
                on_stream_event=on_stream_event,
                store=store,
            )
    finally:
        _TURN_POST_COMMIT_JOB_KINDS.reset(jobs_token)


def _try_resume_replayed_l1_turn(
    *,
    accepted: AcceptedEntryTurn,
    features: dict[str, Any],
    on_stream_event: Callable[[dict[str, Any]], None] | None,
    store: EntryApplicationStorePort,
) -> EntryTurnResult | None:
    """只恢复此请求重放背后的精确活动 L1 聚合。"""

    with turn_linkage_scope(
        session_id=accepted.session_id,
        turn_id=accepted.turn_id,
    ):
        return _try_resume_replayed_l1_turn_scoped(
            accepted=accepted,
            features=features,
            on_stream_event=on_stream_event,
            store=store,
        )


def _try_resume_replayed_l1_turn_scoped(
    *,
    accepted: AcceptedEntryTurn,
    features: dict[str, Any],
    on_stream_event: Callable[[dict[str, Any]], None] | None,
    store: EntryApplicationStorePort,
) -> EntryTurnResult | None:
    """Resume L1 with the original accepted Turn bound to trajectory."""

    from ..l1.recovery import claim_replayed_l1_turn, can_resume_pending_file_preparations

    try:
        recovery = claim_replayed_l1_turn(
            accepted=accepted,
            features=features,
            lease_owner=_runtime_entry_lease_owner(),
            on_stream_event=on_stream_event,
            store=store,
        )
    except Exception:
        return None
    if recovery is None:
        return None
    if not recovery.heartbeat_started:
        try:
            return _incomplete_internal_turn(
                accepted=accepted,
                revision=recovery.window_revision,
                emit=recovery.emit,
                store=store,
                expected_lease_owner=recovery.lease_owner,
            )
        finally:
            recovery.close()

    try:
        if not recovery.lease_is_current():
            return _replayed_turn_result(
                accepted=accepted,
                store=store,
            )
        if recovery.terminal_notification is not None:
            # 失败事实已持久化：只补交付，不重建模型/工具上下文或赠送新执行预算。
            jobs_token = _TURN_POST_COMMIT_JOB_KINDS.set(
                accepted.execution_snapshot.post_commit_job_kinds
            )
            try:
                return _finalize_l1_terminal_notification(
                    accepted=accepted, revision=recovery.window_revision,
                    notification=recovery.terminal_notification, emit=recovery.emit,
                    store=store, expected_lease_owner=recovery.lease_owner,
                )
            finally:
                _TURN_POST_COMMIT_JOB_KINDS.reset(jobs_token)
        if recovery.has_unconfirmed_tool_call and not can_resume_pending_file_preparations(
            accepted=accepted, l1_turn_run_id=recovery.l1_turn_run_id, store=store,
        ):
            try:
                store.fail_l1_turn_run(
                    session_id=accepted.session_id,
                    turn_id=accepted.turn_id,
                    l1_turn_run_id=recovery.l1_turn_run_id,
                    failure_code=RuntimeErrorCode.TOOL_COMPLETION_UNCONFIRMED.value,
                )
            except Exception:
                pass
            return _incomplete_turn(
                accepted=accepted,
                revision=recovery.window_revision,
                end_reason="tool_completion_unconfirmed",
                error_code=RuntimeErrorCode.TOOL_COMPLETION_UNCONFIRMED,
                stage=RuntimeStage.TOOL,
                processing_level="L1",
                emit=recovery.emit,
                store=store,
                expected_lease_owner=recovery.lease_owner,
            )
        from ..l1.execution_config import (
            L1ExecutionConfigError,
            load_l1_execution_config,
        )

        try:
            frozen_execution = load_l1_execution_config(
                recovery.execution_config_json,
                recovery.execution_config_hash,
            )
            if (
                dict(frozen_execution.snapshot.features)
                != accepted.execution_snapshot.features
            ):
                raise L1ExecutionConfigError(
                    "L1 execution features changed after Turn acceptance"
                )
        except L1ExecutionConfigError:
            try:
                store.fail_l1_turn_run(
                    session_id=accepted.session_id,
                    turn_id=accepted.turn_id,
                    l1_turn_run_id=recovery.l1_turn_run_id,
                    failure_code=RuntimeErrorCode.MODEL_CONFIGURATION_FAILURE.value,
                )
            except Exception:
                pass
            return _incomplete_turn(
                accepted=accepted,
                revision=recovery.window_revision,
                end_reason="model_configuration_failure",
                error_code=RuntimeErrorCode.MODEL_CONFIGURATION_FAILURE,
                stage=RuntimeStage.L1_BOOTSTRAP,
                processing_level="L1",
                emit=recovery.emit,
                store=store,
                expected_lease_owner=recovery.lease_owner,
            )
        recovery_features = dict(frozen_execution.snapshot.features)
        context = build_entry_context(
            accepted=accepted,
            features=recovery_features,
            store=store,
            deadline=recovery.deadline,
        )
        post_commit_jobs = accepted.execution_snapshot.post_commit_job_kinds
        jobs_token = _TURN_POST_COMMIT_JOB_KINDS.set(post_commit_jobs)
        try:
            return _run_l1_controller_and_finalize(
                accepted=accepted,
                context=context,
                l1_turn_run_id=recovery.l1_turn_run_id,
                revision=recovery.window_revision,
                deadline=recovery.deadline,
                features=recovery_features,
                emit=recovery.emit,
                store=store,
                lease_owner=recovery.lease_owner,
                lease_is_current=recovery.lease_is_current,
            )
        finally:
            _TURN_POST_COMMIT_JOB_KINDS.reset(jobs_token)
    except TurnDeadlineExceeded:
        if not recovery.lease_is_current():
            return _replayed_turn_result(
                accepted=accepted,
                store=store,
            )
        return _incomplete_turn(
            accepted=accepted,
            revision=recovery.window_revision,
            end_reason="host_stopped",
            error_code=RuntimeErrorCode.TURN_DEADLINE_EXCEEDED,
            stage=RuntimeStage.L1_BOOTSTRAP,
            processing_level="L1",
            emit=recovery.emit,
            store=store,
            expected_lease_owner=recovery.lease_owner,
        )
    except ContextBudgetExceeded:
        if not recovery.lease_is_current():
            return _replayed_turn_result(
                accepted=accepted,
                store=store,
            )
        return _incomplete_context_budget_turn(
            accepted=accepted,
            revision=recovery.window_revision,
            processing_level="L1",
            emit=recovery.emit,
            store=store,
            expected_lease_owner=recovery.lease_owner,
        )
    except Exception:
        if not recovery.lease_is_current():
            return _replayed_turn_result(
                accepted=accepted,
                store=store,
            )
        return _incomplete_internal_turn(
            accepted=accepted,
            revision=recovery.window_revision,
            emit=recovery.emit,
            store=store,
            expected_lease_owner=recovery.lease_owner,
        )
    finally:
        recovery.close()


def resume_active_l1_entry_turn(
    *,
    session_id: str,
    features: dict[str, Any],
    store: EntryApplicationStorePort,
) -> EntryTurnResult | Literal["busy", "not_waiting"]:
    """恢复一个已持久化 L1 checkpoint，而不接受另一个 Turn。"""

    return _resume_active_l1_entry_turn_with_bound_retrieval(
        session_id=session_id,
        features=features,
        store=store,
    )


def _resume_active_l1_entry_turn_with_bound_retrieval(
    *,
    session_id: str,
    features: dict[str, Any],
    store: EntryApplicationStorePort,
) -> EntryTurnResult | Literal["busy", "not_waiting"]:
    """私有实现；冻结 run feature 在最终化时绑定。"""

    try:
        with session_run_guard(session_id):
            try:
                inspected = store.inspect_turn_execution(session_id)
                window = inspected.get("window")
                if (
                    not isinstance(window, dict)
                    or str(window.get("window_state") or "") != "active"
                    or not window.get("current_l1_turn_run_id")
                    or window.get("current_work_run_id") is not None
                    or window.get("current_attempt_id") is not None
                ):
                    return "not_waiting"
                execution = store.get_l1_turn_execution(
                    session_id=session_id,
                    turn_id=str(window.get("turn_id") or ""),
                )
                if not isinstance(execution, dict):
                    return "not_waiting"
                run = execution.get("run")
                state = execution.get("state")
                delivery_pending = (
                    isinstance(run, dict)
                    and isinstance(state, dict)
                    and run.get("status") == "failed"
                    and state.get("stage") == "failed"
                    and build_l1_terminal_notification(state.get("failure_code")) is not None
                )
                if (
                    not isinstance(run, dict)
                    or not isinstance(state, dict)
                    or (not delivery_pending and (
                        str(run.get("status") or "") != "active"
                        or str(state.get("stage") or "") not in {
                            "bootstrap", "model", "tool", "observation", "finalizing",
                        }
                    ))
                    or str(run.get("l1_turn_run_id") or "")
                    != str(window.get("current_l1_turn_run_id") or "")
                ):
                    return "not_waiting"
                accepted = _accepted_l1_recovery(inspected)
            except Exception:
                return "busy"
            resumed = _try_resume_replayed_l1_turn(
                accepted=accepted,
                features=features,
                on_stream_event=None,
                store=store,
            )
            return resumed if resumed is not None else "busy"
    except SessionRunBusyError:
        return "busy"


def _run_l1_controller_and_finalize(
    *,
    accepted: AcceptedEntryTurn,
    context: EntryContext,
    l1_turn_run_id: str,
    revision: int,
    deadline: TurnDeadline,
    features: dict[str, Any],
    emit: EntryEventEmitter,
    store: EntryApplicationStorePort,
    lease_owner: str,
    lease_is_current: Callable[[], bool] | None = None,
    l1_tool_runtime: Any | None = None,
    l1_corpus_manifest: Any | None = None,
) -> EntryTurnResult:
    """在同一 finalizer 下运行一个新建或重新认领的 L1 controller。

    controller 产出的是已验证的 L1 答复与最新 Window revision；Entry 随后检查租约并
    通过 _finalize_formal_reply 发布正式消息。执行失败统一映射为可公开的中断结果，
    新执行和恢复执行共用这一收尾边界。
    """

    from ..l1.controller import L1ControllerFailure
    from ..l1.delivery import terminal_l1_notification
    from ..l1.entry_lane import run_l1_entry_lane

    try:
        l1_result = run_l1_entry_lane(
            accepted=accepted,
            context=context,
            l1_turn_run_id=l1_turn_run_id,
            initial_turn_window_revision=revision,
            deadline=deadline,
            features=features,
            emit=emit,
            store=store,
            lease_owner=lease_owner,
            frozen_tool_runtime=l1_tool_runtime,
            frozen_corpus_manifest=l1_corpus_manifest,
        )
    except L1ControllerFailure as exc:
        authoritative_window = _authoritative_active_turn_window(
            accepted=accepted,
            store=store,
            expected_lease_owner=lease_owner,
        )
        if authoritative_window is None:
            return _replayed_turn_result(
                accepted=accepted,
                store=store,
            )
        notification = terminal_l1_notification(exc)
        if notification is not None:
            return _finalize_l1_terminal_notification(
                accepted=accepted, revision=int(authoritative_window["state_version"]),
                notification=notification, emit=emit, store=store,
                expected_lease_owner=lease_owner,
            )
        tool_completion_unknown = (
            exc.error_code is RuntimeErrorCode.TOOL_COMPLETION_UNCONFIRMED
        )
        return _incomplete_turn(
            accepted=accepted,
            revision=int(authoritative_window["state_version"]),
            end_reason=(
                "tool_completion_unconfirmed" if tool_completion_unknown
                else "l1_controller_rejected"
            ),
            error_code=exc.error_code,
            stage=RuntimeStage.TOOL if tool_completion_unknown else RuntimeStage.L1_BOOTSTRAP,
            processing_level="L1",
            emit=emit,
            store=store,
            expected_lease_owner=lease_owner,
        )
    except ModelGatewayError as exc:
        authoritative_window = _authoritative_active_turn_window(
            accepted=accepted,
            store=store,
            expected_lease_owner=lease_owner,
        )
        if authoritative_window is None:
            return _replayed_turn_result(
                accepted=accepted,
                store=store,
            )
        notification = terminal_l1_notification(exc)
        if notification is not None:
            return _finalize_l1_terminal_notification(
                accepted=accepted, revision=int(authoritative_window["state_version"]),
                notification=notification, emit=emit, store=store,
                expected_lease_owner=lease_owner,
            )
        return _incomplete_model_turn(
            accepted=accepted,
            revision=int(authoritative_window["state_version"]),
            processing_level="L1",
            error=exc,
            emit=emit,
            store=store,
            expected_lease_owner=lease_owner,
        )
    except ContextBudgetExceeded as exc:
        authoritative_window = _authoritative_active_turn_window(
            accepted=accepted,
            store=store,
            expected_lease_owner=lease_owner,
        )
        if authoritative_window is None:
            return _replayed_turn_result(
                accepted=accepted,
                store=store,
            )
        notification = terminal_l1_notification(exc)
        if notification is not None:
            return _finalize_l1_terminal_notification(
                accepted=accepted, revision=int(authoritative_window["state_version"]),
                notification=notification, emit=emit, store=store,
                expected_lease_owner=lease_owner,
            )
        return _incomplete_context_budget_turn(
            accepted=accepted,
            revision=int(authoritative_window["state_version"]),
            processing_level="L1",
            emit=emit,
            store=store,
            expected_lease_owner=lease_owner,
        )
    except RuntimeModelCallWaitingExternal:
        authoritative_window = _authoritative_active_turn_window(
            accepted=accepted,
            store=store,
            expected_lease_owner=lease_owner,
        )
        if authoritative_window is None:
            return _replayed_turn_result(
                accepted=accepted,
                store=store,
            )
        return _incomplete_turn(
            accepted=accepted,
            revision=int(authoritative_window["state_version"]),
            end_reason="model_completion_unconfirmed",
            error_code=RuntimeErrorCode.MODEL_COMPLETION_UNCONFIRMED,
            stage=RuntimeStage.L1_BOOTSTRAP,
            processing_level="L1",
            emit=emit,
            store=store,
            expected_lease_owner=lease_owner,
        )
    except Exception:
        authoritative_window = _authoritative_active_turn_window(
            accepted=accepted,
            store=store,
            expected_lease_owner=lease_owner,
        )
        if authoritative_window is None:
            return _replayed_turn_result(
                accepted=accepted,
                store=store,
            )
        return _incomplete_internal_turn(
            accepted=accepted,
            revision=int(authoritative_window["state_version"]),
            emit=emit,
            store=store,
            expected_lease_owner=lease_owner,
        )
    if lease_is_current is not None and not lease_is_current():
        return _replayed_turn_result(
            accepted=accepted,
            store=store,
        )
    return _finalize_formal_reply(
        accepted=accepted,
        revision=l1_result.turn_window_revision,
        processing_level="L1",
        reply=l1_result.reply,
        related_insession_task_ids=(),
        emit=emit,
        store=store,
        expected_lease_owner=lease_owner,
    )


def _execute_new_turn(
    *,
    accepted: AcceptedEntryTurn,
    features: dict[str, Any],
    on_stream_event: Callable[[dict[str, Any]], None] | None,
    store: EntryApplicationStorePort,
) -> EntryTurnResult:
    """推进一个新接受的 Turn：context → ingress/classifier → admission → lane。

    本函数持有阶段事件、Window revision 和同一墙钟 deadline。空输入/预估超限
    先短路；其余输入并行做确定性 ingress 和模型分类，再由 Host 校验并分派。
    emit 先持久化事件再向流客户端投影；advance 用 revision 推进窗口，避免旧执行者
    覆盖新状态。异常最终收敛为 incomplete，不能凭内存中的文本伪造正式答复。
    """

    revision = accepted.window_revision
    lease_owner = _runtime_entry_lease_owner()
    processing_level: ProcessingLevel | None = None

    def emit(event: TurnEvent) -> int | None:
        # 公开事件是持久化 observation。stream 客户端绝不能收到缺少用于追赶的已存储
        # sequence 的事件。
        try:
            sequence = store.append_runtime_turn_event(
                event,
                active_window_lease_owner=lease_owner,
            )
        except Exception:
            return None
        if on_stream_event is not None:
            try:
                on_stream_event({
                    "event": "runtime_event",
                    "runtime_event": project_turn_event(
                        event, sequence=sequence
                    ).model_dump(mode="json"),
                })
            except OSError:
                pass
        return sequence

    def advance(
        stage: RuntimeStage,
        event_sequence: int | None = None,
    ) -> int:
        nonlocal revision
        window = store.advance_turn_execution_window(
            session_id=accepted.session_id,
            turn_id=accepted.turn_id,
            expected_window_revision=revision,
            stage=stage.value,
            lease_owner=lease_owner,
            last_event_sequence=event_sequence,
        )
        revision = int(window["state_version"])
        return revision

    try:
        deadline = TurnDeadline.starting_now(
            float(features.get(
                "turn_wall_clock_budget_s",
                DEFAULT_TURN_WALL_CLOCK_BUDGET_S,
            ))
        )
        ingress_started = emit(new_turn_event(
            turn_id=accepted.turn_id,
            session_id=accepted.session_id,
            stage=RuntimeStage.INGRESS,
            status=TurnEventStatus.STARTED,
        ))
        advance(RuntimeStage.INGRESS, event_sequence=ingress_started)
        context = build_entry_context(
            accepted=accepted,
            features=features,
            store=store,
            deadline=deadline,
        )
        if (
            not accepted.user_input.strip()
            or context.estimated_input_tokens > int(features.get("context_guard_limit", 24000))
        ):
            ingress = _evaluate_ingress_context(context, features)
            return _complete_nonroutable_turn(
                accepted=accepted,
                ingress=ingress,
                revision=revision,
                emit=emit,
                store=store,
            )

        advance(RuntimeStage.CLASSIFY)

        def complete_nonmodel_ingress(ingress: IngressDecision) -> EntryTurnResult:
            return _complete_nonroutable_turn(
                accepted=accepted,
                ingress=ingress,
                revision=revision,
                emit=emit,
                store=store,
            )

        def complete_classifier_error(error: ModelGatewayError) -> EntryTurnResult:
            return _incomplete_model_turn(
                accepted=accepted,
                revision=revision,
                processing_level=None,
                error=error,
                emit=emit,
                store=store,
            )

        # 分类只给提案：先等 ingress 的消费许可，再做 Host admission 和实际 lane 路由。
        classification_stage = run_entry_classification_stage(
            context=context,
            features=features,
            emit=emit,
            deadline=deadline,
            ingress_evaluator=_evaluate_ingress_context,
            classifier=classify_turn,
            on_nonmodel_ingress=complete_nonmodel_ingress,
            on_classifier_error=complete_classifier_error,
        )
        ingress = classification_stage.ingress
        if classification_stage.terminal_result is not None:
            return classification_stage.terminal_result
        classification = classification_stage.classification
        assert classification is not None

        ingress_completed = emit(new_turn_event(
            turn_id=accepted.turn_id,
            session_id=accepted.session_id,
            stage=RuntimeStage.INGRESS,
            status=TurnEventStatus.COMPLETED,
        ))
        advance(RuntimeStage.SUPERVISOR, event_sequence=ingress_completed)
        emit(new_turn_event(
            turn_id=accepted.turn_id,
            session_id=accepted.session_id,
            stage=RuntimeStage.SUPERVISOR,
            status=TurnEventStatus.STARTED,
        ))
        admission = admit_entry_task_matches(
            accepted=accepted,
            classification=classification,
            context=context,
            ingress=ingress,
            expected_window_revision=revision,
        )
        processing_level = admission.processing_level
        applied = admission.applied
        related_insession_task_ids = admission.related_insession_task_ids
        revision = admission.window_revision
        processing_route = select_entry_processing_route(
            accepted=accepted,
            classification=classification,
            applied=applied,
            processing_level=processing_level,
        )
        supervisor_completed = emit(new_turn_event(
            turn_id=accepted.turn_id,
            session_id=accepted.session_id,
            stage=RuntimeStage.SUPERVISOR,
            status=TurnEventStatus.COMPLETED,
        ))
        return _execute_entry_processing_route(
            route=processing_route,
            revision=revision,
            scope=_EntryProcessingScope(
                accepted=accepted,
                context=context,
                related_insession_task_ids=related_insession_task_ids,
                deadline=deadline,
                features=features,
                emit=emit,
                advance=advance,
                store=store,
                lease_owner=lease_owner,
                supervisor_completed_event_sequence=supervisor_completed,
            ),
        )
    except ProcessingRouteAuthorityError:
        return _incomplete_turn(
            accepted=accepted,
            revision=revision,
            end_reason="persistence_error",
            error_code=RuntimeErrorCode.PERSIST_FAILED,
            stage=RuntimeStage.SUPERVISOR,
            processing_level="L2",
            emit=emit,
            store=store,
            expected_lease_owner=lease_owner,
        )
    except TurnDeadlineExceeded:
        return _incomplete_turn(
            accepted=accepted,
            revision=revision,
            end_reason="host_stopped",
            error_code=RuntimeErrorCode.TURN_DEADLINE_EXCEEDED,
            stage=RuntimeStage.INGRESS,
            processing_level=None,
            emit=emit,
            store=store,
        )
    except ContextBudgetExceeded:
        authoritative_window = _authoritative_active_turn_window(
            accepted=accepted,
            store=store,
        )
        if authoritative_window is not None:
            revision = int(authoritative_window["state_version"])
        return _incomplete_context_budget_turn(
            accepted=accepted,
            revision=revision,
            processing_level=processing_level,
            emit=emit,
            store=store,
        )
    except Exception:
        authoritative_window = _authoritative_active_turn_window(
            accepted=accepted,
            store=store,
        )
        if authoritative_window is not None:
            revision = int(authoritative_window["state_version"])
        return _incomplete_internal_turn(
            accepted=accepted,
            revision=revision,
            emit=emit,
            store=store,
        )


def _execute_entry_processing_route(
    *,
    route: EntryProcessingRoute,
    revision: int,
    scope: _EntryProcessingScope,
) -> EntryTurnResult:
    """将 Host 已选定的 route DTO 分派到唯一执行分支，不再询问 Router 模型。

    L1 进入 _execute_l1_processing_lane，直接回复 route 进入 response lane；
    revision、deadline、emit 与 Store 从同一个 Entry scope 传递，保证收尾回到
    同一正式提交边界。未知 route 是内部合同错误，不能猜测一个默认分支。
    """

    if isinstance(route, EntryL1ProcessingRoute):
        return _execute_l1_processing_lane(
            revision=revision,
            scope=scope,
        )
    if isinstance(route, EntryL2TaskProcessingRoute):
        return _execute_l2_task_processing_lane(
            task_id=route.task_id,
            revision=revision,
            scope=scope,
        )
    if isinstance(route, EntryResponseProcessingRoute):
        return _execute_response_processing_lane(
            processing_level=route.processing_level,
            revision=revision,
            scope=scope,
        )
    raise RuntimeError(f"unsupported Entry processing route: {route.kind}")


def _execute_l1_processing_lane(
    *,
    revision: int,
    scope: _EntryProcessingScope,
) -> EntryTurnResult:
    """Entry 到 L1 bootstrap 的适配：发布阶段事件，委托 entry_lane.prepare_l1_entry_lane。

    工具/模型/语料快照及 TurnRun 创建由 L1 owner 持有；Entry 把准备失败映射为
    公开中断，成功则把同一准备材料和最新 Window revision 交给 controller。
    controller 的已验证答复仍回到 _run_l1_controller_and_finalize 正式提交。
    """

    from ..l1.attachment_contracts import L1TurnAttachmentToolRuntimeError
    from ..l1.entry_lane import prepare_l1_entry_lane
    from ..l1.tool_runtime import (
        L1ToolCatalogUnavailableError,
        L1WorkspaceUnavailableError,
    )

    accepted = scope.accepted
    scope.emit(new_turn_event(
        turn_id=accepted.turn_id,
        session_id=accepted.session_id,
        stage=RuntimeStage.L1_BOOTSTRAP,
        status=TurnEventStatus.STARTED,
    ))
    try:
        prepared = prepare_l1_entry_lane(
            accepted=accepted,
            context=scope.context,
            features=scope.features,
            initial_turn_window_revision=revision,
            routing_policy_snapshot_hash=snapshot_sha256(
                accepted.routing_policy
            ),
            deadline=scope.deadline,
            store=scope.store,
            lease_owner=scope.lease_owner,
        )
    except L1TurnAttachmentToolRuntimeError:
        return _incomplete_turn(
            accepted=accepted,
            revision=revision,
            end_reason="l1_attachment_authority_unavailable",
            error_code=RuntimeErrorCode.L1_RUNTIME_NOT_READY,
            stage=RuntimeStage.L1_BOOTSTRAP,
            processing_level="L1",
            emit=scope.emit,
            store=scope.store,
            expected_lease_owner=scope.lease_owner,
        )
    except L1WorkspaceUnavailableError:
        return _incomplete_turn(
            accepted=accepted,
            revision=revision,
            end_reason="l1_workspace_unavailable",
            error_code=RuntimeErrorCode.L1_RUNTIME_NOT_READY,
            stage=RuntimeStage.L1_BOOTSTRAP,
            processing_level="L1",
            emit=scope.emit,
            store=scope.store,
            expected_lease_owner=scope.lease_owner,
        )
    except L1ToolCatalogUnavailableError:
        return _incomplete_turn(
            accepted=accepted,
            revision=revision,
            end_reason="l1_tool_catalog_unavailable",
            error_code=RuntimeErrorCode.L1_RUNTIME_NOT_READY,
            stage=RuntimeStage.L1_BOOTSTRAP,
            processing_level="L1",
            emit=scope.emit,
            store=scope.store,
            expected_lease_owner=scope.lease_owner,
        )
    except Exception:
        return _incomplete_internal_turn(
            accepted=accepted,
            revision=revision,
            emit=scope.emit,
            store=scope.store,
            expected_lease_owner=scope.lease_owner,
        )
    return _run_l1_controller_and_finalize(
        accepted=accepted,
        context=scope.context,
        l1_turn_run_id=prepared.l1_turn_run_id,
        revision=prepared.turn_window_revision,
        deadline=scope.deadline,
        features=scope.features,
        emit=scope.emit,
        store=scope.store,
        lease_owner=scope.lease_owner,
        l1_tool_runtime=prepared.tool_runtime,
        l1_corpus_manifest=prepared.corpus_manifest,
    )


def _execute_l2_task_processing_lane(
    *,
    task_id: str,
    revision: int,
    scope: _EntryProcessingScope,
) -> EntryTurnResult:
    """运行已经 durable Guard 认证的 L2 Task lane。"""

    accepted = scope.accepted
    revision = scope.advance(
        RuntimeStage.L2_PLAN,
        scope.supervisor_completed_event_sequence,
    )
    try:
        return _execute_auxiliary_production_chain(
            accepted=accepted,
            task_id=task_id,
            revision=revision,
            related_insession_task_ids=scope.related_insession_task_ids,
            deadline=scope.deadline,
            emit=scope.emit,
            store=scope.store,
            features=scope.features,
        )
    except ModelGatewayError as exc:
        authoritative_window = _authoritative_active_turn_window(
            accepted=accepted,
            store=scope.store,
            expected_lease_owner=scope.lease_owner,
        )
        if authoritative_window is None:
            return _replayed_turn_result(
                accepted=accepted,
                store=scope.store,
            )
        return _incomplete_model_turn(
            accepted=accepted,
            revision=int(authoritative_window["state_version"]),
            processing_level="L2",
            error=exc,
            emit=scope.emit,
            store=scope.store,
            expected_lease_owner=scope.lease_owner,
        )


def _execute_response_processing_lane(
    *,
    processing_level: Literal["L0", "L2"],
    revision: int,
    scope: _EntryProcessingScope,
) -> EntryTurnResult:
    """生成不需要 L1 controller 或 L2 Task executor 的单轮回复。"""

    stage = (
        RuntimeStage.L0_GENERATE
        if processing_level == "L0"
        else RuntimeStage.L2_UNDERSTAND
    )
    revision = scope.advance(
        stage,
        scope.supervisor_completed_event_sequence,
    )
    try:
        response = generate_response(
            scope.context,
            processing_level,
            scope.emit,
            scope.deadline,
        )
    except ModelGatewayError as exc:
        return _incomplete_model_turn(
            accepted=scope.accepted,
            revision=revision,
            processing_level=processing_level,
            error=exc,
            emit=scope.emit,
            store=scope.store,
        )
    return _finalize_formal_reply(
        accepted=scope.accepted,
        revision=revision,
        processing_level=processing_level,
        reply=response.model_result.reply,
        related_insession_task_ids=scope.related_insession_task_ids,
        emit=scope.emit,
        store=scope.store,
    )


def _evaluate_ingress_context(
    context: EntryContext,
    features: dict[str, Any],
) -> IngressDecision:
    return evaluate_ingress(
        context.envelope,
        context.snapshot,
        context.ceiling,
        estimated_input_tokens=context.estimated_input_tokens,
        context_hard_limit=int(features.get("context_guard_limit", 24000)),
    )


def _complete_nonroutable_turn(
    *,
    accepted: AcceptedEntryTurn,
    ingress: IngressDecision,
    revision: int,
    emit: EntryEventEmitter,
    store: EntryTurnFinalizationStorePort,
) -> EntryTurnResult:
    """为确定性 ingress 短路生成 Host 固定答复，并经同一正式提交边界保存。

    此分支不调用回复模型。INGRESS failed 事件表达输入被拒绝，但带说明的正式
    L0 回复仍可完成；因此不能用某一条阶段 failed 推断整个 Turn 一定 incomplete。
    """

    outcome = select_entry_ingress_short_circuit_outcome(ingress.disposition)
    emit(new_turn_event(
        turn_id=accepted.turn_id,
        session_id=accepted.session_id,
        stage=RuntimeStage.INGRESS,
        status=TurnEventStatus.FAILED,
        error_code=outcome.error_code,
    ))
    return _finalize_formal_reply(
        accepted=accepted,
        revision=revision,
        processing_level="L0",
        reply=outcome.reply,
        related_insession_task_ids=(),
        error_code=outcome.error_code.value,
        emit=emit,
        store=store,
    )


def _finalize_l1_terminal_notification(
    *, accepted: AcceptedEntryTurn, revision: int, notification: L1TerminalNotification,
    emit: EntryEventEmitter, store: EntryTurnFinalizationStorePort,
    expected_lease_owner: str | None,
) -> EntryTurnResult:
    """L1 选择通知内容，Entry 只复用唯一提交事务；被拒候选不会经过此接口。"""
    emit(new_turn_event(
        turn_id=accepted.turn_id, session_id=accepted.session_id,
        stage=RuntimeStage.L1_BOOTSTRAP, status=TurnEventStatus.FAILED,
        error_code=RuntimeErrorCode(notification.error_code),
    ))
    return _finalize_formal_reply(
        accepted=accepted, revision=revision, processing_level="L1",
        reply=notification.reply, related_insession_task_ids=(), emit=emit, store=store,
        error_code=notification.error_code, expected_lease_owner=expected_lease_owner,
        l1_terminal_failure_code=notification.failure_code,
    )


def _finalize_formal_reply(
    *,
    accepted: AcceptedEntryTurn,
    revision: int,
    processing_level: ProcessingLevel,
    reply: str,
    related_insession_task_ids: tuple[str, ...],
    emit: EntryEventEmitter,
    store: EntryTurnFinalizationStorePort,
    error_code: str | None = None,
    expected_lease_owner: str | None = None,
    l1_terminal_failure_code: str | None = None,
) -> EntryTurnResult:
    """正式提交边界（formal commit）：持久答案与 post-commit jobs 一起落入 Store。

    先以 Window revision / lease 推进到 persist，再提交唯一正式 assistant 消息。
    若写入抛错，先核对是否其实已提交成功，防止响应丢失导致重复交付。
    返回 completed 只证明正式答复已提交；post_commit_pending 仍保留窗口，
    直到摘要/索引等派生任务结算，下一 Turn 才能看到一致的上下文。
    """

    end_reason = "l1_terminal_notification" if l1_terminal_failure_code is not None else None
    persist_started = emit(new_turn_event(
        turn_id=accepted.turn_id,
        session_id=accepted.session_id,
        stage=RuntimeStage.PERSIST,
        status=TurnEventStatus.STARTED,
    ))
    try:
        window = store.advance_turn_execution_window(
            session_id=accepted.session_id,
            turn_id=accepted.turn_id,
            expected_window_revision=revision,
            stage=RuntimeStage.PERSIST.value,
            lease_owner=(
                expected_lease_owner or _runtime_entry_lease_owner()
            ),
            last_event_sequence=persist_started,
        )
        revision = int(window["state_version"])
        finalized = store.finalize_turn_execution(
            session_id=accepted.session_id,
            turn_id=accepted.turn_id,
            expected_window_revision=revision,
            processing_level=processing_level,
            assistant_content=reply,
            # 正式交付与 job 同时持久化；API 在 Runtime 返回后调度 worker。
            # 派生任务未结算前，Window 阻止后续 Turn 观察到半更新上下文。
            post_commit_job_kinds=_TURN_POST_COMMIT_JOB_KINDS.get(),
            expected_lease_owner=expected_lease_owner,
            **({"l1_terminal_failure_code": l1_terminal_failure_code}
               if l1_terminal_failure_code is not None else {}),
        )
        window = _required_mapping(finalized, "window")
        revision = int(window["state_version"])
    except Exception:
        recovered = recover_completed_entry_turn_from_commit(
            accepted=accepted,
            processing_level=processing_level,
            related_insession_task_ids=related_insession_task_ids,
            store=store,
            error_code=error_code,
            end_reason=end_reason,
        )
        if recovered is not None:
            return recovered
        return _incomplete_turn(
            accepted=accepted,
            revision=revision,
            end_reason="persistence_error",
            error_code=RuntimeErrorCode.PERSIST_FAILED,
            stage=RuntimeStage.PERSIST,
            emit=emit,
            store=store,
            expected_lease_owner=expected_lease_owner,
        )
    emit(new_turn_event(
        turn_id=accepted.turn_id,
        session_id=accepted.session_id,
        stage=RuntimeStage.PERSIST,
        status=TurnEventStatus.COMPLETED,
    ))
    emit(new_turn_event(
        turn_id=accepted.turn_id,
        session_id=accepted.session_id,
        stage=RuntimeStage.RESPONSE,
        status=TurnEventStatus.COMPLETED,
    ))
    return EntryTurnResult(
        session_id=accepted.session_id,
        turn_id=accepted.turn_id,
        status="completed",
        processing_level=processing_level,
        reply=reply,
        error_code=error_code,
        end_reason=end_reason,
        related_insession_task_ids=related_insession_task_ids,
        window_state="post_commit_pending",
        window_revision=revision,
    )


def _execute_auxiliary_production_chain(
    *,
    accepted: AcceptedEntryTurn,
    task_id: str,
    revision: int,
    related_insession_task_ids: tuple[str, ...],
    deadline: TurnDeadline,
    emit: EntryEventEmitter,
    store: EntryTurnFinalizationStorePort,
    expected_lease_owner: str | None = None,
    features: Mapping[str, Any] | None = None,
) -> EntryTurnResult:
    """运行所选 Auxiliary Task，并且只发布其冻结 Delivery 引用。"""

    with turn_linkage_scope(
        session_id=accepted.session_id,
        turn_id=accepted.turn_id,
    ):
        return _execute_auxiliary_production_chain_scoped(
            accepted=accepted,
            task_id=task_id,
            revision=revision,
            related_insession_task_ids=related_insession_task_ids,
            deadline=deadline,
            emit=emit,
            store=store,
            expected_lease_owner=expected_lease_owner,
            features=features,
        )


def _execute_auxiliary_production_chain_scoped(
    *,
    accepted: AcceptedEntryTurn,
    task_id: str,
    revision: int,
    related_insession_task_ids: tuple[str, ...],
    deadline: TurnDeadline,
    emit: EntryEventEmitter,
    store: EntryTurnFinalizationStorePort,
    expected_lease_owner: str | None,
    features: Mapping[str, Any] | None,
) -> EntryTurnResult:
    """Run the selected auxiliary chain inside the accepted Turn scope."""

    from personagraph.l2.entry_adapter.auxiliary_outcome import (
        AuxiliaryEntryOutcomeDecisionKind,
        interpret_auxiliary_production_chain_outcome,
    )
    from personagraph.l2.entry_adapter.application import (
        L2TaskTargetUnavailableError,
        run_l2_task_lane,
    )

    try:
        result = run_l2_task_lane(
            session_id=accepted.session_id,
            turn_id=accepted.turn_id,
            task_id=task_id,
            emit=emit,
            deadline=deadline,
            features=features,
            file_retrieval_data_version=(
                accepted.execution_snapshot.file_retrieval_data_version
            ),
        )
    except L2TaskTargetUnavailableError:
        return _incomplete_turn(
            accepted=accepted,
            revision=revision,
            end_reason="module_error",
            error_code=RuntimeErrorCode.TRANSITION_DENIED,
            stage=RuntimeStage.TRANSITION_GUARD,
            processing_level="L2",
            emit=emit,
            store=store,
            expected_lease_owner=expected_lease_owner,
        )
    authoritative_window = _authoritative_active_turn_window(
        accepted=accepted,
        store=store,
        expected_lease_owner=expected_lease_owner,
    )
    if expected_lease_owner is not None and authoritative_window is None:
        raise _AuxiliaryTurnLeaseLost(
            "auxiliary execution lost its Turn lease"
        )
    if authoritative_window is not None:
        revision = int(authoritative_window["state_version"])

    outcome = interpret_auxiliary_production_chain_outcome(
        result,
        authoritative_window_available=authoritative_window is not None,
        allow_user_input=(
            (features or {}).get("user_interaction_mode", "interactive")
            == "interactive"
        ),
    )
    if (
        outcome.kind
        is AuxiliaryEntryOutcomeDecisionKind.FINALIZE_VERIFIED_DELIVERY
    ):
        assert outcome.delivery_id is not None
        # 有意丢弃组合中的 ``publication_body``。Entry finalizer 会重新加载冻结
        # NodeDelivery，并请求 Store 将该精确已验证引用原子提交到对话记录。
        return _finalize_verified_work_run_reply(
            accepted=accepted,
            delivery_id=outcome.delivery_id,
            revision=revision,
            related_insession_task_ids=related_insession_task_ids,
            emit=emit,
            store=store,
            expected_lease_owner=expected_lease_owner,
        )

    if (
        outcome.kind
        is AuxiliaryEntryOutcomeDecisionKind.FINALIZE_FORMAL_QUESTION
    ):
        assert outcome.reply is not None
        # 问题是 UserGate 持有的唯一公开材料。它将此 Turn 作为正式交互完成，而 Task
        # 及其精确 WorkRun 会持久地等待下一回答 Turn。
        return _finalize_formal_reply(
            accepted=accepted,
            revision=revision,
            processing_level="L2",
            reply=outcome.reply,
            related_insession_task_ids=related_insession_task_ids,
            emit=emit,
            store=store,
            expected_lease_owner=expected_lease_owner,
        )

    assert outcome.kind is AuxiliaryEntryOutcomeDecisionKind.INCOMPLETE
    assert outcome.end_reason is not None
    assert outcome.error_code is not None
    assert outcome.stage is not None
    return _incomplete_turn(
        accepted=accepted,
        revision=revision,
        end_reason=outcome.end_reason,
        error_code=outcome.error_code,
        stage=outcome.stage,
        processing_level="L2",
        emit=emit,
        store=store,
        expected_lease_owner=expected_lease_owner,
    )


def _entry_turn_from_authoritative_no_public_stop(
    *,
    accepted: AcceptedEntryTurn,
    settled: object,
    related_insession_task_ids: tuple[str, ...],
    work_run_ids: tuple[str, ...],
) -> EntryTurnResult | None:
    """验证并投影一个 Store 持有的非公开停止结果。"""

    from personagraph.l2.entry_adapter.no_public_stop import (
        project_authoritative_no_public_stop_settlement,
    )

    projection = project_authoritative_no_public_stop_settlement(
        settled,
        expected_turn_id=accepted.turn_id,
    )
    if projection is None:
        return None
    return EntryTurnResult(
        session_id=accepted.session_id,
        turn_id=accepted.turn_id,
        status="incomplete",
        processing_level="L2",
        end_reason=projection.end_reason,
        error_code=projection.error_code.value,
        related_insession_task_ids=related_insession_task_ids,
        work_run_ids=work_run_ids,
        window_state="interrupted",
        window_revision=projection.window_revision,
    )


def _try_replay_authoritative_turn_settlement(
    *,
    accepted: AcceptedEntryTurn,
    revision: int,
    related_insession_task_ids: tuple[str, ...],
    store: EntryTurnFinalizationStorePort,
) -> EntryTurnResult | None:
    """重放经完整证明的公开或非公开 lane 集合，绝不重放前缀。"""

    try:
        store.finalize_authoritative_referenced_turn_execution(
            session_id=accepted.session_id,
            turn_id=accepted.turn_id,
            expected_window_revision=revision,
            post_commit_job_kinds=_TURN_POST_COMMIT_JOB_KINDS.get(),
        )
    except Exception:
        # 不完整 lane 前缀会回滚 Store 事务。成功提交后丢失的响应从对话记录恢复。
        completed = recover_completed_entry_turn_from_commit(
            accepted=accepted,
            processing_level="L2",
            related_insession_task_ids=related_insession_task_ids,
            store=store,
        )
        if completed is not None:
            return completed
    else:
        completed = recover_completed_entry_turn_from_commit(
            accepted=accepted,
            processing_level="L2",
            related_insession_task_ids=related_insession_task_ids,
            store=store,
        )
        if completed is not None:
            return completed

    try:
        stopped = store.mark_authoritative_no_public_turn_stop(
            session_id=accepted.session_id,
            turn_id=accepted.turn_id,
            expected_window_revision=revision,
        )
    except Exception:
        # 对部分 lane 前缀，两个 reducer 都会在写入前失败。一次最终 commit 读取可覆盖
        # 响应/读取暂时丢失的公开最终化，而不将该歧义转成停止。
        return recover_completed_entry_turn_from_commit(
            accepted=accepted,
            processing_level="L2",
            related_insession_task_ids=related_insession_task_ids,
            store=store,
        )
    try:
        work_run_ids = store.list_turn_linked_work_run_ids(
            session_id=accepted.session_id,
            turn_id=accepted.turn_id,
        )
    except Exception:
        try:
            work_run_ids = store.list_turn_linked_work_run_ids(
                session_id=accepted.session_id,
                turn_id=accepted.turn_id,
            )
        except Exception:
            return None
    return _entry_turn_from_authoritative_no_public_stop(
        accepted=accepted,
        settled=stopped,
        related_insession_task_ids=related_insession_task_ids,
        work_run_ids=work_run_ids,
    )


def _finalize_verified_work_run_reply(
    *,
    accepted: AcceptedEntryTurn,
    delivery_id: str,
    revision: int,
    related_insession_task_ids: tuple[str, ...],
    emit: EntryEventEmitter,
    store: EntryTurnFinalizationStorePort,
    expected_lease_owner: str | None = None,
) -> EntryTurnResult:
    """通过引用发布一个冻结 NodeDelivery，并返回其投影。"""

    from ...session.l2_store import verification as verification_store

    try:
        try:
            resolved = verification_store.get_task_node_delivery(
                session_id=accepted.session_id,
                delivery_id=delivery_id,
            )
        except Exception:
            # 读取不存在变更歧义。在将 Delivery 视为不可用前，对精确权威投影重试一次。
            resolved = verification_store.get_task_node_delivery(
                session_id=accepted.session_id,
                delivery_id=delivery_id,
            )
    except Exception:
        from ...l2.entry_adapter.settlement import (
            project_pending_verified_publication_result,
        )

        pending = project_pending_verified_publication_result(
            accepted=accepted,
            delivery_id=delivery_id,
            related_insession_task_ids=related_insession_task_ids,
            store=store,
        )
        if pending is not None:
            return pending
        return _incomplete_turn(
            accepted=accepted,
            revision=revision,
            end_reason="persistence_error",
            error_code=RuntimeErrorCode.PERSIST_FAILED,
            stage=RuntimeStage.PERSIST,
            processing_level="L2",
            emit=emit,
            store=store,
            expected_lease_owner=expected_lease_owner,
        )
    output_window = getattr(resolved, "output_window", None)
    reply = getattr(output_window, "content", None)
    if not isinstance(reply, str) or not reply:
        return _incomplete_turn(
            accepted=accepted,
            revision=revision,
            end_reason="persistence_error",
            error_code=RuntimeErrorCode.PERSIST_FAILED,
            stage=RuntimeStage.PERSIST,
            processing_level="L2",
            emit=emit,
            store=store,
            expected_lease_owner=expected_lease_owner,
        )

    delivery = getattr(resolved, "delivery", None)
    delivery_work_run_id = getattr(delivery, "work_run_id", None)

    def _load_public_work_run_ids() -> tuple[str, ...]:
        work_run_ids = tuple(
            store.list_turn_linked_work_run_ids(
                session_id=accepted.session_id,
                turn_id=accepted.turn_id,
            )
        )
        if (
            not isinstance(delivery_work_run_id, str)
            or not delivery_work_run_id
            or not work_run_ids
            or delivery_work_run_id not in work_run_ids
            or len(work_run_ids) != len(set(work_run_ids))
        ):
            raise ValueError(
                "verified Delivery is missing its authoritative Turn--WorkRun link"
            )
        return work_run_ids

    try:
        try:
            work_run_ids = _load_public_work_run_ids()
        except Exception:
            # 这是只读投影。重试一次，但当持久化权威状态仍不可读时，绝不使用伪造或
            # 不完整 WorkRun 引用发布形式上已完成的 Turn。
            work_run_ids = _load_public_work_run_ids()
    except Exception:
        return _incomplete_turn(
            accepted=accepted,
            revision=revision,
            end_reason="persistence_error",
            error_code=RuntimeErrorCode.PERSIST_FAILED,
            stage=RuntimeStage.PERSIST,
            processing_level="L2",
            emit=emit,
            store=store,
            expected_lease_owner=expected_lease_owner,
        )

    persist_started = emit(new_turn_event(
        turn_id=accepted.turn_id,
        session_id=accepted.session_id,
        stage=RuntimeStage.PERSIST,
        status=TurnEventStatus.STARTED,
    ))
    del persist_started  # WorkRun PASS 已将 Window 移至 PERSIST。
    try:
        try:
            finalized = store.finalize_verified_turn_execution(
                session_id=accepted.session_id,
                turn_id=accepted.turn_id,
                delivery_id=delivery_id,
                expected_window_revision=revision,
                post_commit_job_kinds=_TURN_POST_COMMIT_JOB_KINDS.get(),
                expected_lease_owner=expected_lease_owner,
            )
        except Exception:
            # commit 后丢失的响应只能通过完全相同的不可变 Delivery 命令恢复；Store
            # 重放会拒绝 collision。
            finalized = store.finalize_verified_turn_execution(
                session_id=accepted.session_id,
                turn_id=accepted.turn_id,
                delivery_id=delivery_id,
                expected_window_revision=revision,
                post_commit_job_kinds=_TURN_POST_COMMIT_JOB_KINDS.get(),
                expected_lease_owner=expected_lease_owner,
            )
        window = _required_mapping(finalized, "window")
        revision = int(window["state_version"])
    except Exception:
        recovered = recover_completed_entry_turn_from_commit(
            accepted=accepted,
            processing_level="L2",
            related_insession_task_ids=related_insession_task_ids,
            store=store,
            work_run_ids=work_run_ids,
        )
        if recovered is not None:
            return recovered
        return _incomplete_turn(
            accepted=accepted,
            revision=revision,
            end_reason="persistence_error",
            error_code=RuntimeErrorCode.PERSIST_FAILED,
            stage=RuntimeStage.PERSIST,
            processing_level="L2",
            emit=emit,
            store=store,
            expected_lease_owner=expected_lease_owner,
        )
    emit(new_turn_event(
        turn_id=accepted.turn_id,
        session_id=accepted.session_id,
        stage=RuntimeStage.PERSIST,
        status=TurnEventStatus.COMPLETED,
    ))
    emit(new_turn_event(
        turn_id=accepted.turn_id,
        session_id=accepted.session_id,
        stage=RuntimeStage.RESPONSE,
        status=TurnEventStatus.COMPLETED,
    ))
    return EntryTurnResult(
        session_id=accepted.session_id,
        turn_id=accepted.turn_id,
        status="completed",
        processing_level="L2",
        reply=reply,
        related_insession_task_ids=related_insession_task_ids,
        work_run_ids=work_run_ids,
        window_state="post_commit_pending",
        window_revision=revision,
    )


def _incomplete_model_turn(
    *,
    accepted: AcceptedEntryTurn,
    revision: int,
    processing_level: ProcessingLevel | None,
    error: ModelGatewayError,
    emit: EntryEventEmitter,
    store: EntryTurnMutationStorePort,
    expected_lease_owner: str | None = None,
) -> EntryTurnResult:
    return _incomplete_turn(
        accepted=accepted,
        revision=revision,
        end_reason="provider_unavailable",
        error_code=_public_model_error_code(error),
        stage=RuntimeStage.RESPONSE,
        processing_level=processing_level,
        emit=emit,
        store=store,
        expected_lease_owner=expected_lease_owner,
    )


def _incomplete_context_budget_turn(
    *,
    accepted: AcceptedEntryTurn,
    revision: int,
    processing_level: ProcessingLevel | None,
    emit: EntryEventEmitter,
    store: EntryTurnMutationStorePort,
    expected_lease_owner: str | None = None,
) -> EntryTurnResult:
    """结算精确 Provider envelope 无法容纳的已准入 Turn。"""

    return _incomplete_turn(
        accepted=accepted,
        revision=revision,
        end_reason="context_budget_exceeded",
        error_code=RuntimeErrorCode.CONTEXT_BUDGET_EXCEEDED,
        stage=RuntimeStage.RESPONSE,
        processing_level=processing_level,
        emit=emit,
        store=store,
        expected_lease_owner=expected_lease_owner,
    )


def _incomplete_internal_turn(
    *,
    accepted: AcceptedEntryTurn,
    revision: int,
    emit: EntryEventEmitter,
    store: EntryTurnMutationStorePort,
    expected_lease_owner: str | None = None,
) -> EntryTurnResult:
    return _incomplete_turn(
        accepted=accepted,
        revision=revision,
        end_reason="module_error",
        error_code=RuntimeErrorCode.INTERNAL_FAILURE,
        stage=RuntimeStage.RESPONSE,
        emit=emit,
        store=store,
        expected_lease_owner=expected_lease_owner,
    )


def _incomplete_turn(
    *,
    accepted: AcceptedEntryTurn,
    revision: int,
    end_reason: str,
    error_code: RuntimeErrorCode,
    stage: RuntimeStage,
    emit: EntryEventEmitter,
    store: EntryTurnMutationStorePort,
    processing_level: ProcessingLevel | None = None,
    expected_lease_owner: str | None = None,
) -> EntryTurnResult:
    """将已接受但未正式交付的 Turn 标为中断，并返回公开的失败分类。

    不写 assistant 正文。若持久中断本身失败，返回仍占用的 active Window，
    让后续恢复/审计处理真实状态；不能仅凭本次异常就宣称窗口已经释放。
    """

    emit(new_turn_event(
        turn_id=accepted.turn_id,
        session_id=accepted.session_id,
        stage=stage,
        status=TurnEventStatus.FAILED,
        error_code=error_code,
    ))
    window_state: Literal["active", "interrupted"] = "active"
    window_revision = revision
    try:
        window = store.mark_turn_execution_interrupted(
            session_id=accepted.session_id,
            turn_id=accepted.turn_id,
            expected_window_revision=revision,
            stage=stage.value,
            interruption_reason=error_code.value,
            expected_lease_owner=expected_lease_owner,
        )
        window_state = "interrupted"
        window_revision = int(window["state_version"])
    except Exception:
        # store 失败无法在进程内修复。已接受输入与非空 Window 仍足以让下一输入审计
        # 报告进程丢失，而不是伪造正式 assistant 消息。
        pass
    return EntryTurnResult(
        session_id=accepted.session_id,
        turn_id=accepted.turn_id,
        status="incomplete",
        processing_level=processing_level,
        end_reason=end_reason,
        error_code=error_code.value,
        window_state=window_state,
        window_revision=window_revision,
    )


def _replayed_turn_result(
    *,
    accepted: AcceptedEntryTurn,
    store: EntryReplayStorePort,
) -> EntryTurnResult:
    def reconcile_authoritative_settlement(
        *,
        revision: int,
        related_insession_task_ids: tuple[str, ...],
    ) -> EntryTurnResult | None:
        return _try_replay_authoritative_turn_settlement(
            accepted=accepted,
            revision=revision,
            related_insession_task_ids=related_insession_task_ids,
            store=store,
        )

    return reconcile_replayed_entry_turn(
        accepted=accepted,
        store=store,
        reconcile_authoritative_settlement=reconcile_authoritative_settlement,
    )


def _emit_turn_accepted(
    callback: EntryAcceptedEmitter | None,
    accepted: AcceptedEntryTurn,
) -> None:
    """仅在持久化 acceptance 后尽力发送传输通知。"""

    if callback is None:
        return
    try:
        callback(accepted)
    except Exception:
        # 这只是传输通知。损坏 serializer、已关闭 SSE peer 或 adapter 故障绝不能在
        # 权威执行路径有机会结算前使已接受 Turn 搁浅。
        pass
