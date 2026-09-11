"""单 Turn 的有界 L1 控制器（durable ReAct loop）。

阅读顺序：run_l1_turn → _run_l1_turn_impl → _model_payload / _admit_decision
→ _execute_tool_batch 或最终答复的两层 verification。Store 中的 TurnRun、Attempt
和 ToolCall 是恢复权威，内存变量只是当前投影；trajectory 是诊断记录。
controller 返回已验证答复，正式 assistant 消息与窗口收尾由 Entry 持有。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Literal

from ...context_budget import ContextBudgetExceeded
from personagraph.model_io.tier_bindings import ModelTierBinding
from personagraph.model_io.gateway import ModelGatewayError
from personagraph.model_io.output_repair_contracts import RuntimeModelOutputRepairIssue
from .corpus_contracts import (
    FrozenL1CorpusManifest,
    L1_CORPUS_MANIFEST_CONTRACT_VERSION,
    L1CorpusManifestError,
    load_l1_corpus_manifest,
)
from .semantic_contracts import (
    L1_SEMANTIC_VERIFICATION_FEATURE,
    L1SemanticVerificationReceipt,
    L1SemanticVerificationTrigger,
    derive_l1_semantic_verification_trigger,
    parse_l1_semantic_verification_mode,
)
from ...tools.contracts import ExecutionStatus, thaw_json
from ...trajectory import record_tool_call
from ...output_protocol import (
    CallToolsAction,
    L1AttemptDecisionProposal,
    SubmitFinalReplyAction,
    materialize_l1_plan,
)
from ...output_protocol.l1 import L1PlanProposal, L1ResultReference, L1_ATTEMPT_PROTOCOL_VERSION
from ...persistent_turn_content import (
    L1Plan,
    l1_tool_result_id,
)
from ...persistent_turn_content.findings import (
    ExecutionFindingsLedgerCreateResult,
    ExecutionFindingsOwnerKind,
    ExecutionFindingsSnapshot,
)
from ...tools.findings.contracts import EXECUTION_FINDINGS_TOOL_IDS
from ...tools.tool_history.definitions import TOOL_HISTORY_TOOL_IDS
from ...tools.findings.dispatcher import execute_execution_findings_tool
from ..turn.contracts import AcceptedEntryTurn
from ..model_calls.authority import RuntimeModelCallWaitingExternal
from ..turn_deadline import TurnDeadline
from ..turn_events import (
    EntryEventEmitter,
    RuntimeErrorCode,
    RuntimeStage,
    TurnEventStatus,
    new_turn_event,
)
from .identity import (
    canonical_json,
    sha256_json,
)
from .corpus_manifest import freeze_l1_corpus_manifest
from .context import L1TurnContext
from .execution_config import (
    L1ExecutionConfigError,
    freeze_l1_execution_config,
    load_l1_execution_config,
)
from .model import L1AttemptDecisionAdmissionError, request_l1_attempt_decision
from .plan_revision import (
    L1PlanRevisionError,
    L1PlanUnchangedError,
    validate_l1_plan_revision,
)
from .ports import L1StorePort
from .execution_notes import (
    L1FindingsRevisionError,
    bind_explicit_findings_revision,
    record_committed_execution_notes,
)
from .protected_tool_dispatch import (
    L1ProtectedToolDispatchRequest,
    L1ProtectedToolDispatcher,
)
from .semantic_verification import (
    L1SemanticVerificationInputError,
    build_l1_semantic_verification_receipt,
    request_l1_semantic_verification,
)
from .tool_runtime import (
    L1CatalogSnapshotRestoreError,
    L1ToolRuntime,
    L1ToolCatalogUnavailableError,
    L1WorkspaceUnavailableError,
    build_l1_tool_runtime,
)
from .tool_context import (
    ToolResultProjectionError,
    project_execution_findings_for_model,
    project_recent_tool_results,
)
from .attachment_contracts import (
    L1TurnAttachmentToolRuntimeError,
)
from .verification import (
    L1VerificationResult,
    L1VerificationStateError,
    protected_l1_acceptance_ids,
    verify_l1_final_reply,
)


_L1_MODEL_VIEW_SOFT_UTF8_BYTES = 900_000
_L1_MODEL_VIEW_HARD_UTF8_BYTES = 1_350_000


@dataclass(frozen=True, slots=True)
class L1ControllerResult:
    reply: str
    turn_window_revision: int


@dataclass(frozen=True, slots=True)
class _AdmittedL1AttemptDecision:
    decision: L1AttemptDecisionProposal
    materialized_plan: L1Plan
    mechanical_verification: L1VerificationResult | None
    semantic_verification: L1SemanticVerificationReceipt | None
    verification_rejection: dict[str, Any] | None = None


def run_l1_turn(
    *,
    accepted: AcceptedEntryTurn,
    context: L1TurnContext,
    l1_turn_run_id: str,
    initial_turn_window_revision: int,
    deadline: TurnDeadline,
    emit: EntryEventEmitter,
    store: L1StorePort,
    lease_owner: str | None,
    max_attempts: int = 24,
    max_tool_calls_per_attempt: int = 8,
    execution_features: dict[str, Any] | None = None,
    frozen_tool_runtime: L1ToolRuntime | None = None,
    frozen_corpus_manifest: FrozenL1CorpusManifest | None = None,
) -> L1ControllerResult:
    """L1 对外执行边界：驱动控制循环，并将外层异常映射到持久 run 的失败状态。

    max_attempts 限制 L1 决策轮数，不是某次模型请求的 HTTP 重试次数；
    max_tool_calls_per_attempt 限制一轮工具批次。已存在 run 的预算、deadline 与
    模型/工具快照由内部恢复路径读取，调用方不能靠重入获得一份新预算。
    """

    try:
        return _run_l1_turn_impl(
            accepted=accepted,
            context=context,
            l1_turn_run_id=l1_turn_run_id,
            initial_turn_window_revision=initial_turn_window_revision,
            deadline=deadline,
            emit=emit,
            store=store,
            lease_owner=lease_owner,
            max_attempts=max_attempts,
            max_tool_calls_per_attempt=max_tool_calls_per_attempt,
            execution_features=execution_features,
            frozen_tool_runtime=frozen_tool_runtime,
            frozen_corpus_manifest=frozen_corpus_manifest,
        )
    except L1ControllerFailure as exc:
        _best_effort_fail_run(
            accepted=accepted,
            l1_turn_run_id=l1_turn_run_id,
            failure_code=exc.error_code.value,
            store=store,
        )
        raise
    except ModelGatewayError as exc:
        _best_effort_fail_run(
            accepted=accepted,
            l1_turn_run_id=l1_turn_run_id,
            failure_code=exc.code,
            store=store,
        )
        raise
    except ContextBudgetExceeded:
        _best_effort_fail_run(
            accepted=accepted,
            l1_turn_run_id=l1_turn_run_id,
            failure_code=RuntimeErrorCode.CONTEXT_BUDGET_EXCEEDED.value,
            store=store,
        )
        raise
    except RuntimeModelCallWaitingExternal:
        _best_effort_fail_run(
            accepted=accepted,
            l1_turn_run_id=l1_turn_run_id,
            failure_code=RuntimeErrorCode.MODEL_COMPLETION_UNCONFIRMED.value,
            store=store,
        )
        raise
    except Exception:
        _best_effort_fail_run(
            accepted=accepted,
            l1_turn_run_id=l1_turn_run_id,
            failure_code=RuntimeErrorCode.INTERNAL_FAILURE.value,
            store=store,
        )
        raise


def _run_l1_turn_impl(
    *,
    accepted: AcceptedEntryTurn,
    context: L1TurnContext,
    l1_turn_run_id: str,
    initial_turn_window_revision: int,
    deadline: TurnDeadline,
    emit: EntryEventEmitter,
    store: L1StorePort,
    lease_owner: str | None,
    max_attempts: int,
    max_tool_calls_per_attempt: int,
    execution_features: dict[str, Any] | None,
    frozen_tool_runtime: L1ToolRuntime | None,
    frozen_corpus_manifest: FrozenL1CorpusManifest | None,
) -> L1ControllerResult:
    """以持久 checkpoint 为起点，循环执行“读状态 → 决策 → 校验 → 提交/工具”。

    先验证或初始化工具、语料、模型和预算快照；每轮重新读 Store。已决定的工具批次
    接着执行，尚未返回决策的 Attempt 复用 request_json，均不凭新上下文另起一次决策。
    新模型提案经 admission；最终候选再过机械/语义验证。候选被驳回会带反馈进入
    下一 L1 Attempt；合法 call_tools 先提交决定再执行，合法 final reply 返回 Entry。
    """

    # Bootstrap / restore：先对齐不可变快照，恢复不能按当前环境悄悄换资源或预算。
    existing_execution = store.get_l1_turn_execution(
        session_id=accepted.session_id,
        turn_id=accepted.turn_id,
    )
    existing_state = _optional_mapping(existing_execution, "state")

    if frozen_tool_runtime is not None:
        tool_runtime = frozen_tool_runtime
    else:
        try:
            tool_runtime = build_l1_tool_runtime(
                accepted.session_id,
                turn_id=accepted.turn_id,
                execution_findings_enabled=True,
                execution_features=execution_features,
                file_retrieval_data_version=(
                    accepted.execution_snapshot.file_retrieval_data_version
                    if accepted.execution_snapshot is not None
                    else None
                ),
                session_retrieval_data_version=(
                    accepted.execution_snapshot.session_retrieval_data_version
                    if accepted.execution_snapshot is not None
                    else None
                ),
                session_retrieval_assistant_turn_cutoff=(
                    accepted.execution_snapshot.session_retrieval_assistant_turn_cutoff
                    if accepted.execution_snapshot is not None
                    else None
                ),
                expected_catalog_snapshot_json=(
                    str(existing_state["catalog_snapshot_json"])
                    if existing_state is not None
                    else None
                ),
                expected_catalog_snapshot_sha256=(
                    str(existing_state["catalog_snapshot_hash"])
                    if existing_state is not None
                    else None
                ),
            )
        except L1TurnAttachmentToolRuntimeError as exc:
            raise L1ControllerFailure(
                RuntimeErrorCode.L1_RUNTIME_NOT_READY,
                "L1 Turn attachment authority is unavailable or changed",
            ) from exc
        except L1WorkspaceUnavailableError as exc:
            raise L1ControllerFailure(
                RuntimeErrorCode.L1_RUNTIME_NOT_READY,
                "L1 fixed workspace is unavailable",
            ) from exc
        except (
            L1CatalogSnapshotRestoreError,
            L1ToolCatalogUnavailableError,
        ) as exc:
            raise L1ControllerFailure(
                RuntimeErrorCode.L1_RUNTIME_NOT_READY,
                "L1 Tool Catalog is unavailable or changed",
            ) from exc
    try:
        corpus_manifest = frozen_corpus_manifest or freeze_l1_corpus_manifest(
            session_id=accepted.session_id,
            turn_id=accepted.turn_id,
            l1_turn_run_id=l1_turn_run_id,
            tool_runtime=tool_runtime,
            attachments=context.attachments,
        )
    except Exception as exc:
        raise L1ControllerFailure(
            RuntimeErrorCode.L1_RUNTIME_NOT_READY,
            "L1 corpus authority could not be frozen",
        ) from exc
    if (
        corpus_manifest.manifest.session_id != accepted.session_id
        or corpus_manifest.manifest.turn_id != accepted.turn_id
        or corpus_manifest.manifest.l1_turn_run_id != l1_turn_run_id
        or corpus_manifest.manifest.catalog_snapshot_sha256
        != tool_runtime.catalog_snapshot_sha256
    ):
        raise L1ControllerFailure(
            RuntimeErrorCode.L1_RUNTIME_NOT_READY,
            "L1 corpus authority crossed its TurnRun or Tool Catalog",
        )
    if existing_state is not None:
        try:
            if (
                existing_state.get("corpus_manifest_contract_version")
                != L1_CORPUS_MANIFEST_CONTRACT_VERSION
            ):
                raise L1CorpusManifestError(
                    "L1 corpus manifest contract is unavailable"
                )
            persisted_corpus = load_l1_corpus_manifest(
                existing_state.get("corpus_manifest_json"),
                existing_state.get("corpus_manifest_hash"),
                expected_session_id=accepted.session_id,
                expected_turn_id=accepted.turn_id,
                expected_l1_turn_run_id=l1_turn_run_id,
            )
            if (
                persisted_corpus.manifest_sha256 != corpus_manifest.manifest_sha256
                or persisted_corpus.manifest_json != corpus_manifest.manifest_json
            ):
                raise L1CorpusManifestError("L1 corpus sources changed after bootstrap")
        except L1CorpusManifestError as exc:
            raise L1ControllerFailure(
                RuntimeErrorCode.L1_RUNTIME_NOT_READY,
                "L1 corpus authority is unavailable or changed",
            ) from exc
    if existing_state is None:
        frozen_execution = freeze_l1_execution_config(execution_features or {})
        deadline_at = (
            datetime.now(timezone.utc) + timedelta(seconds=deadline.remaining_s())
        ).isoformat()
        effective_max_attempts = max_attempts
        effective_max_tool_calls_per_attempt = max_tool_calls_per_attempt
        run_deadline = deadline
    else:
        try:
            frozen_execution = load_l1_execution_config(
                existing_state.get("execution_config_json"),
                existing_state.get("execution_config_hash"),
            )
        except L1ExecutionConfigError as exc:
            raise L1ControllerFailure(
                RuntimeErrorCode.MODEL_CONFIGURATION_FAILURE,
                "L1 execution configuration is unavailable or changed",
            ) from exc
        deadline_at = _required_text(existing_state, "deadline_at")
        effective_max_attempts = int(existing_state["max_attempts"])
        effective_max_tool_calls_per_attempt = int(
            existing_state["max_tool_calls_per_attempt"]
        )
        run_deadline = _deadline_from_utc(deadline_at)
    initialized = store.initialize_l1_turn_run(
        session_id=accepted.session_id,
        turn_id=accepted.turn_id,
        l1_turn_run_id=l1_turn_run_id,
        deadline_at=deadline_at,
        max_attempts=effective_max_attempts,
        max_tool_calls_per_attempt=effective_max_tool_calls_per_attempt,
        catalog_snapshot_json=tool_runtime.catalog_snapshot_json,
        catalog_snapshot_hash=tool_runtime.catalog_snapshot_sha256,
        execution_config_json=frozen_execution.snapshot_json,
        execution_config_hash=frozen_execution.snapshot_sha256,
        corpus_manifest_contract_version=L1_CORPUS_MANIFEST_CONTRACT_VERSION,
        corpus_manifest_json=corpus_manifest.manifest_json,
        corpus_manifest_hash=corpus_manifest.manifest_sha256,
        expected_window_revision=initial_turn_window_revision,
        expected_lease_owner=lease_owner,
    )
    revision = _turn_window_revision(initialized)
    findings_created = store.create_execution_findings_ledger(
        session_id=accepted.session_id,
        owner_kind=ExecutionFindingsOwnerKind.L1_TURN_RUN,
        execution_owner_id=l1_turn_run_id,
    )
    if not isinstance(
        findings_created,
        ExecutionFindingsLedgerCreateResult,
    ):
        raise L1ControllerFailure(
            RuntimeErrorCode.INTERNAL_FAILURE,
            "L1 findings ledger creation returned another contract",
        )
    findings_ledger_id = findings_created.ledger.ledger_id
    try:
        while True:
            if run_deadline.expired():
                raise L1ControllerFailure(
                    RuntimeErrorCode.TURN_DEADLINE_EXCEEDED,
                    "L1 Turn deadline expired",
                )
            # Checkpoint 是权威：每轮重读已提交状态，覆盖新执行与进程恢复两种入口。
            execution = store.get_l1_turn_execution(
                session_id=accepted.session_id,
                turn_id=accepted.turn_id,
            )
            if not isinstance(execution, dict):
                raise L1ControllerFailure(
                    RuntimeErrorCode.INTERNAL_FAILURE,
                    "L1 Store omitted the executable run projection",
                )
            state = _mapping(execution, "state")
            current_plan = _load_plan(state.get("plan_json"))
            current_attempt = _current_attempt(execution)
            if current_attempt is not None:
                _record_execution_notes(store, findings_ledger_id, current_attempt)
                attempt_status = str(current_attempt.get("status") or "")
                action_kind = str(current_attempt.get("action_kind") or "")
                if (
                    attempt_status == "closed"
                    and action_kind == "submit_final_reply"
                    and not current_attempt.get("verification_feedback_json")
                ):
                    return _submitted_final_reply_result(
                        accepted=accepted,
                        attempt=current_attempt,
                        revision=revision,
                    )
                # 决定已提交但批次尚未关闭：续做原批次，已结算工具由 reservation 复用。
                if attempt_status == "active" and action_kind == "call_tools":
                    decision = _load_decision(current_attempt.get("decision_json"))
                    action = decision.action
                    if not isinstance(action, CallToolsAction):
                        raise L1ControllerFailure(
                            RuntimeErrorCode.INTERNAL_FAILURE,
                            "persisted decided L1 Attempt has no ToolCall action",
                        )
                    tool_results = _execute_tool_batch(
                        accepted=accepted,
                        l1_turn_run_id=l1_turn_run_id,
                        attempt_id=str(current_attempt["attempt_id"]),
                        action=action,
                        tool_runtime=tool_runtime,
                        deadline=run_deadline,
                        emit=emit,
                        store=store,
                        max_tool_calls_per_attempt=(
                            effective_max_tool_calls_per_attempt
                        ),
                        expected_window_revision=revision,
                        expected_lease_owner=lease_owner,
                        findings_ledger_id=findings_ledger_id,
                    )
                    closed = store.close_l1_attempt(
                        session_id=accepted.session_id,
                        turn_id=accepted.turn_id,
                        l1_turn_run_id=l1_turn_run_id,
                        attempt_id=str(current_attempt["attempt_id"]),
                        tool_results_json=canonical_json(tool_results),
                        tool_results_hash=sha256_json(tool_results),
                        expected_window_revision=revision,
                        expected_lease_owner=lease_owner,
                    )
                    revision = _turn_window_revision(closed)
                    continue
            used = int(state.get("attempts_started") or 0)
            maximum = int(state.get("max_attempts") or effective_max_attempts)
            remaining = maximum - used
            if remaining <= 0:
                verification_feedback = _parse_optional_json(
                    state.get("verification_feedback_json")
                )
                exhausted_after_rejection = (
                    isinstance(verification_feedback, dict)
                    and verification_feedback.get("kind") == "verification_rejected"
                )
                raise L1ControllerFailure(
                    (
                        RuntimeErrorCode.VERIFICATION_FAILED
                        if exhausted_after_rejection
                        else RuntimeErrorCode.MODEL_OUTPUT_INVALID
                    ),
                    (
                        "L1 verification did not pass before the Attempt limit"
                        if exhausted_after_rejection
                        else "L1 Attempt limit ended before final-reply submission"
                    ),
                )
            finalization_required = remaining == 1
            pending_model_attempt = (
                current_attempt is not None
                and str(current_attempt.get("status") or "") == "active"
                and not current_attempt.get("action_kind")
            )
            # 模型请求已落 checkpoint 时复用精确 request_json，不能重新采样 deadline/上下文。
            if pending_model_attempt:
                payload = _required_json_mapping(
                    current_attempt.get("request_json"),
                    label="persisted L1 model request",
                )
                finalization_required = _persisted_finalization_required(payload)
            else:
                payload = _model_payload(
                    accepted=accepted,
                    context=context,
                    execution=execution,
                    state=state,
                    current_plan=current_plan,
                    tool_runtime=tool_runtime,
                    remaining=remaining,
                    deadline=run_deadline,
                    finalization_required=finalization_required,
                    findings_snapshot=_require_l1_findings_snapshot(
                        store=store,
                        l1_turn_run_id=l1_turn_run_id,
                    ),
                )
                request_json = canonical_json(payload)
                started = store.start_l1_attempt(
                    session_id=accepted.session_id,
                    turn_id=accepted.turn_id,
                    l1_turn_run_id=l1_turn_run_id,
                    request_json=request_json,
                    request_hash=sha256_json(payload),
                    expected_window_revision=revision,
                    expected_lease_owner=lease_owner,
                )
                revision = _turn_window_revision(started)
                attempt = _mapping(started, "attempt")
                # Store 在同一事务分配步骤身份并冻结最终请求。必须发送返回的精确版本，
                # 不能在 HTTP 前临时加 ID，也不能继续发送分配身份前的候选 payload。
                payload = _required_json_mapping(
                    attempt.get("request_json"),
                    label="prepared L1 model request",
                )
            if pending_model_attempt:
                assert current_attempt is not None
                attempt = current_attempt
            attempt_id = str(attempt["attempt_id"])
            state_guard_hash = store.get_l1_attempt_state_guard(
                session_id=accepted.session_id,
                turn_id=accepted.turn_id,
                l1_turn_run_id=l1_turn_run_id,
                attempt_id=attempt_id,
            )
            admitted: _AdmittedL1AttemptDecision | None = None

            def admit(decision: L1AttemptDecisionProposal) -> None:
                nonlocal admitted
                try:
                    (
                        admitted_decision,
                        materialized_plan,
                    ) = _admit_decision(
                        decision,
                        current_plan=current_plan,
                        accepted=accepted,
                        l1_turn_run_id=l1_turn_run_id,
                        tool_runtime=tool_runtime,
                        execution=execution,
                        finalization_required=finalization_required,
                        max_tool_calls_per_attempt=(
                            effective_max_tool_calls_per_attempt
                        ),
                    )
                    try:
                        admitted_decision = bind_explicit_findings_revision(
                            admitted_decision,
                            request=payload,
                        )
                    except ValueError as exc:
                        raise L1ControllerFailure(
                            RuntimeErrorCode.MODEL_OUTPUT_INVALID,
                            str(exc),
                            repair_issue=(
                                exc.repair_issue
                                if isinstance(exc, L1FindingsRevisionError)
                                else None
                            ),
                        ) from exc
                    mechanical_verification = None
                    semantic_verification = None
                    verification_rejection = None
                    if isinstance(
                        admitted_decision.action,
                        SubmitFinalReplyAction,
                    ):
                        try:
                            mechanical_verification = _verify_final_reply(
                                plan=materialized_plan,
                                reply=admitted_decision.action.reply,
                                references=admitted_decision.references,
                                execution=execution,
                            )
                        except L1MechanicalVerificationRejected as exc:
                            verification_rejection = _verification_feedback(
                                source="mechanical",
                                feedback=str(exc),
                                details=exc.verification.safe_details(),
                            )
                        if mechanical_verification is not None:
                            (
                                semantic_verification,
                                verification_rejection,
                            ) = _semantic_receipt_for_final_reply(
                                decision=admitted_decision,
                                plan=materialized_plan,
                                reply=admitted_decision.action.reply,
                                mechanical_verification=mechanical_verification,
                                execution=execution,
                                execution_features=dict(
                                    frozen_execution.snapshot.features
                                ),
                                model_view=payload,
                                accepted=accepted,
                                l1_turn_run_id=l1_turn_run_id,
                                attempt_id=attempt_id,
                                state_guard_hash=state_guard_hash,
                                emit=emit,
                                deadline=run_deadline,
                                store=store,
                                model_binding=frozen_execution.model_binding,
                            )
                    admitted = _AdmittedL1AttemptDecision(
                        decision=admitted_decision,
                        materialized_plan=materialized_plan,
                        mechanical_verification=mechanical_verification,
                        semantic_verification=semantic_verification,
                        verification_rejection=verification_rejection,
                    )
                except L1ControllerFailure as exc:
                    if exc.error_code is not RuntimeErrorCode.MODEL_OUTPUT_INVALID:
                        raise
                    raise L1AttemptDecisionAdmissionError(
                        str(exc),
                        terminal_error=exc,
                        repair_issue=exc.repair_issue,
                    ) from exc

            request_l1_attempt_decision(
                turn_id=accepted.turn_id,
                session_id=accepted.session_id,
                logical_model_call_id=str(attempt["logical_model_call_id"]),
                payload=payload,
                emit=emit,
                deadline=run_deadline,
                l1_turn_run_id=l1_turn_run_id,
                attempt_id=attempt_id,
                state_guard_hash=state_guard_hash,
                store=store,
                model_binding=frozen_execution.model_binding,
                admit=admit,
            )
            if admitted is None:
                raise L1ControllerFailure(
                    RuntimeErrorCode.INTERNAL_FAILURE,
                    "L1 model decision bypassed Host admission",
                )
            decision = admitted.decision
            materialized_plan = admitted.materialized_plan
            action = decision.action
            # 候选验证驳回会消耗这一 L1 Attempt；反馈交给下一轮，而非重试同一次 HTTP。
            if admitted.verification_rejection is not None:
                if not isinstance(action, SubmitFinalReplyAction):
                    raise L1ControllerFailure(
                        RuntimeErrorCode.INTERNAL_FAILURE,
                        "L1 verification rejection is not bound to a final candidate",
                    )
                verification_feedback = {
                    **admitted.verification_rejection,
                    "attempt_id": attempt_id,
                    "candidate_decision_hash": sha256_json(decision),
                    "candidate_final_reply": action.reply,
                }
                plan_json = (
                    canonical_json(materialized_plan)
                    if decision.plan is not None
                    else None
                )
                rejected = store.reject_l1_final_reply_candidate(
                    session_id=accepted.session_id,
                    turn_id=accepted.turn_id,
                    l1_turn_run_id=l1_turn_run_id,
                    attempt_id=attempt_id,
                    state_guard_hash=state_guard_hash,
                    decision_json=canonical_json(decision),
                    decision_hash=sha256_json(decision),
                    plan_json=plan_json,
                    plan_hash=(
                        sha256_json(materialized_plan)
                        if plan_json is not None
                        else None
                    ),
                    verification_feedback_json=canonical_json(verification_feedback),
                    verification_feedback_hash=sha256_json(verification_feedback),
                    expected_window_revision=revision,
                    expected_lease_owner=lease_owner,
                )
                revision = _turn_window_revision(rejected)
                _record_execution_notes(
                    store, findings_ledger_id, _mapping(rejected, "attempt")
                )
                continue
            final_reply_action = (
                action if isinstance(action, SubmitFinalReplyAction) else None
            )
            formal_verification: L1VerificationResult | None = None
            formal_semantic_verification = admitted.semantic_verification
            if final_reply_action is not None:
                emit(
                    new_turn_event(
                        turn_id=accepted.turn_id,
                        session_id=accepted.session_id,
                        stage=RuntimeStage.VERIFICATION,
                        status=TurnEventStatus.STARTED,
                    )
                )
                assert materialized_plan is not None
                try:
                    formal_verification = _verify_final_reply(
                        plan=materialized_plan,
                        reply=final_reply_action.reply,
                        references=decision.references,
                        execution=execution,
                    )
                    if formal_semantic_verification is None:
                        raise L1ControllerFailure(
                            RuntimeErrorCode.INTERNAL_FAILURE,
                            "L1 semantic verification receipt is missing",
                        )
                    _require_current_semantic_receipt(
                        formal_semantic_verification,
                        decision=decision,
                        plan=materialized_plan,
                        mechanical_verification=formal_verification,
                        execution_features=dict(frozen_execution.snapshot.features),
                        state_guard_hash=state_guard_hash,
                    )
                except L1ControllerFailure as exc:
                    emit(
                        new_turn_event(
                            turn_id=accepted.turn_id,
                            session_id=accepted.session_id,
                            stage=RuntimeStage.VERIFICATION,
                            status=TurnEventStatus.FAILED,
                            error_code=exc.error_code,
                        )
                    )
                    raise
                emit(
                    new_turn_event(
                        turn_id=accepted.turn_id,
                        session_id=accepted.session_id,
                        stage=RuntimeStage.VERIFICATION,
                        status=TurnEventStatus.COMPLETED,
                    )
                )
            plan_json = (
                canonical_json(materialized_plan)
                if decision.plan is not None and materialized_plan is not None
                else None
            )
            # Commit-before-effect：先用 state guard 提交决定，随后才能执行工具或返回最终答复。
            accepted_decision = store.commit_l1_attempt_decision(
                session_id=accepted.session_id,
                turn_id=accepted.turn_id,
                l1_turn_run_id=l1_turn_run_id,
                attempt_id=attempt_id,
                state_guard_hash=state_guard_hash,
                decision_json=canonical_json(decision),
                decision_hash=sha256_json(decision),
                action_kind=action.kind,
                tool_call_count=(
                    len(action.calls) if isinstance(action, CallToolsAction) else 0
                ),
                plan_json=plan_json,
                plan_hash=(
                    sha256_json(materialized_plan)
                    if plan_json is not None and materialized_plan is not None
                    else None
                ),
                final_reply=(
                    final_reply_action.reply if final_reply_action is not None else None
                ),
                verification_contract_version=(
                    formal_verification.contract_version
                    if formal_verification is not None
                    else None
                ),
                verification_report_json=(
                    canonical_json(formal_verification.to_dict())
                    if formal_verification is not None
                    else None
                ),
                verification_report_hash=(
                    sha256_json(formal_verification.to_dict())
                    if formal_verification is not None
                    else None
                ),
                semantic_verification_contract_version=(
                    formal_semantic_verification.contract_version
                    if formal_semantic_verification is not None
                    else None
                ),
                semantic_verification_report_json=(
                    canonical_json(formal_semantic_verification)
                    if formal_semantic_verification is not None
                    else None
                ),
                semantic_verification_report_hash=(
                    sha256_json(formal_semantic_verification)
                    if formal_semantic_verification is not None
                    else None
                ),
                expected_window_revision=revision,
                expected_lease_owner=lease_owner,
            )
            revision = _turn_window_revision(accepted_decision)
            _record_execution_notes(
                store, findings_ledger_id, _mapping(accepted_decision, "attempt")
            )
            if final_reply_action is not None:
                return L1ControllerResult(
                    reply=final_reply_action.reply,
                    turn_window_revision=revision,
                )

            assert isinstance(action, CallToolsAction)
            tool_results = _execute_tool_batch(
                accepted=accepted,
                l1_turn_run_id=l1_turn_run_id,
                attempt_id=attempt_id,
                action=action,
                tool_runtime=tool_runtime,
                deadline=run_deadline,
                emit=emit,
                store=store,
                max_tool_calls_per_attempt=effective_max_tool_calls_per_attempt,
                expected_window_revision=revision,
                expected_lease_owner=lease_owner,
                findings_ledger_id=findings_ledger_id,
            )
            closed = store.close_l1_attempt(
                session_id=accepted.session_id,
                turn_id=accepted.turn_id,
                l1_turn_run_id=l1_turn_run_id,
                attempt_id=attempt_id,
                tool_results_json=canonical_json(tool_results),
                tool_results_hash=sha256_json(tool_results),
                expected_window_revision=revision,
                expected_lease_owner=lease_owner,
            )
            revision = _turn_window_revision(closed)
    except L1ControllerFailure as exc:
        _best_effort_fail_run(
            accepted=accepted,
            l1_turn_run_id=l1_turn_run_id,
            failure_code=exc.error_code.value,
            store=store,
        )
        raise


class L1ControllerFailure(RuntimeError):
    def __init__(
        self,
        error_code: RuntimeErrorCode,
        message: str,
        *,
        repair_issue: RuntimeModelOutputRepairIssue | None = None,
    ) -> None:
        self.error_code = error_code
        self.repair_issue = repair_issue
        super().__init__(message)


class L1MechanicalVerificationRejected(L1ControllerFailure):
    """一个结构上无效的候选者及其无负载的问题事实。"""

    def __init__(self, verification: L1VerificationResult) -> None:
        if verification.passed or not verification.issues:
            raise ValueError(
                "mechanical verification rejection requires at least one issue"
            )
        self.verification = verification
        super().__init__(
            RuntimeErrorCode.VERIFICATION_FAILED,
            verification.safe_feedback(),
        )


def _best_effort_fail_run(
    *,
    accepted: AcceptedEntryTurn,
    l1_turn_run_id: str,
    failure_code: str,
    store: L1StorePort,
) -> None:
    try:
        store.fail_l1_turn_run(
            session_id=accepted.session_id,
            turn_id=accepted.turn_id,
            l1_turn_run_id=l1_turn_run_id,
            failure_code=failure_code,
        )
    except Exception:
        pass


def _model_payload(
    *,
    accepted: AcceptedEntryTurn,
    context: L1TurnContext,
    execution: dict[str, object],
    state: dict[str, Any],
    current_plan: L1Plan | None,
    tool_runtime: L1ToolRuntime,
    remaining: int,
    deadline: TurnDeadline,
    finalization_required: bool,
    findings_snapshot: ExecutionFindingsSnapshot | None,
) -> dict[str, Any]:
    """组装下一步的完整 Host 快照；此处不写持久状态。

    Entry 输入/历史/摘要与附件目录，加上当前 Plan、工具结果、
    findings、验证反馈和执行限制，经 model_view 的白名单投影后才发送给模型；
    原始身份直接复用；Host 诊断不发送。system prompt 在 l1/model.py。
    工具正文只投影紧邻前一步的批次，原始 ToolCall outcome 仍在 Store；corpus manifest
    留在 Host 校验侧。这里的 UTF-8 容量检查之后，Provider 仍会按完整 wire 请求准入。
    """

    verification_feedback = _parse_optional_json(
        state.get("verification_feedback_json")
    )
    try:
        prior_tool_results, tool_result_projection = project_recent_tool_results(
            execution
        )
    except ToolResultProjectionError as exc:
        raise L1ControllerFailure(RuntimeErrorCode.INTERNAL_FAILURE, str(exc)) from exc
    payload = {
        "schema_version": L1_ATTEMPT_PROTOCOL_VERSION,
        "execution_notes_required": True,
        "current_user_text": accepted.user_input,
        "input_message_id": accepted.input_message_id,
        "history_pairs": list(context.history_pairs),
        "session_summary": context.session_summary,
        "attachments": _json_projection(
            context.attachments,
            redact_attachment_authority=(bool(tool_runtime.attachment_file_catalog)),
        ),
        "file_catalog": [dict(item) for item in tool_runtime.attachment_file_catalog],
        "plan": (
            current_plan.model_dump(mode="json") if current_plan is not None else None
        ),
        "prior_tool_results": prior_tool_results,
        "tool_result_projection": tool_result_projection,
        "verification_feedback": (
            verification_feedback
            if isinstance(verification_feedback, dict)
            and verification_feedback.get("kind") == "verification_rejected"
            else None
        ),
        "execution_findings": (
            project_execution_findings_for_model(
                findings_snapshot.active_projection,
                execution=execution,
            )
            if findings_snapshot is not None
            else None
        ),
        "tool_catalog": list(tool_runtime.model_catalog()),
        "execution_limits": {
            "attempts_remaining": remaining,
            "milliseconds_until_deadline": max(
                0,
                int(deadline.remaining_s() * 1000),
            ),
            "max_tool_calls_this_attempt": int(
                state.get("max_tool_calls_per_attempt") or 0
            ),
            "finalization_required": finalization_required,
        },
    }
    # 请求冻结原始上下文，但输入预算只计算真正发送的模型视图。
    # 转换属于 L1 model_view；控制器不展开具体工具身份字段。
    from .model_view.projection import project_attempt_view

    try:
        estimated_bytes = len(
            canonical_json(project_attempt_view(payload)).encode("utf-8")
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise L1ControllerFailure(
            RuntimeErrorCode.INTERNAL_FAILURE,
            "L1 model view could not be projected",
        ) from exc
    if estimated_bytes > _L1_MODEL_VIEW_HARD_UTF8_BYTES:
        raise L1ControllerFailure(
            RuntimeErrorCode.CONTEXT_BUDGET_EXCEEDED,
            "L1 model view exceeds its hard UTF-8 budget",
        )
    limits = payload["execution_limits"]
    assert isinstance(limits, dict)
    limits.update(
        estimated_input_utf8_bytes=estimated_bytes,
        soft_input_utf8_bytes=_L1_MODEL_VIEW_SOFT_UTF8_BYTES,
        hard_input_utf8_bytes=_L1_MODEL_VIEW_HARD_UTF8_BYTES,
        input_utf8_bytes_remaining=max(
            0,
            _L1_MODEL_VIEW_HARD_UTF8_BYTES - estimated_bytes,
        ),
        input_pressure=(
            "soft_limit_exceeded"
            if estimated_bytes > _L1_MODEL_VIEW_SOFT_UTF8_BYTES
            else "normal"
        ),
    )
    return payload


def _require_l1_findings_snapshot(
    *,
    store: L1StorePort,
    l1_turn_run_id: str,
) -> ExecutionFindingsSnapshot:
    value = store.get_execution_findings_ledger_for_owner(
        owner_kind=ExecutionFindingsOwnerKind.L1_TURN_RUN,
        execution_owner_id=l1_turn_run_id,
    )
    if not isinstance(value, ExecutionFindingsSnapshot):
        raise L1ControllerFailure(
            RuntimeErrorCode.INTERNAL_FAILURE,
            "L1 findings ledger projection is unavailable",
        )
    if (
        value.ledger.execution_owner_id != l1_turn_run_id
        or value.ledger.owner_kind is not ExecutionFindingsOwnerKind.L1_TURN_RUN
    ):
        raise L1ControllerFailure(
            RuntimeErrorCode.INTERNAL_FAILURE,
            "L1 findings ledger crossed execution ownership",
        )
    return value


def _record_execution_notes(
    store: L1StorePort,
    ledger_id: str,
    attempt: dict[str, object],
) -> None:
    try:
        record_committed_execution_notes(
            store=store, ledger_id=ledger_id, attempt=attempt
        )
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        raise L1ControllerFailure(
            RuntimeErrorCode.INTERNAL_FAILURE,
            str(exc),
        ) from exc


def _known_tool_result_ids(execution: dict[str, object]) -> frozenset[str]:
    """返回可供下一个 Attempt 使用的成功普通 ToolResult ID。"""

    raw_calls = execution.get("tool_calls")
    if not isinstance(raw_calls, list):
        return frozenset()
    result_ids: set[str] = set()
    for raw_call in raw_calls:
        if not isinstance(raw_call, dict):
            continue
        if raw_call.get("status") != "succeeded":
            continue
        if raw_call.get("tool_id") in (
            *EXECUTION_FINDINGS_TOOL_IDS,
            *TOOL_HISTORY_TOOL_IDS,
        ):
            continue
        tool_call_id = raw_call.get("tool_call_id")
        result_sha256 = raw_call.get("outcome_hash")
        if not isinstance(tool_call_id, str) or not isinstance(
            result_sha256,
            str,
        ):
            continue
        result_ids.add(
            l1_tool_result_id(
                tool_call_id=tool_call_id,
                result_sha256=result_sha256,
            )
        )
    return frozenset(result_ids)


def _admit_decision(
    decision: L1AttemptDecisionProposal,
    *,
    current_plan: L1Plan | None,
    accepted: AcceptedEntryTurn,
    l1_turn_run_id: str,
    tool_runtime: L1ToolRuntime,
    execution: dict[str, object],
    finalization_required: bool,
    max_tool_calls_per_attempt: int,
) -> tuple[
    L1AttemptDecisionProposal,
    L1Plan,
]:
    """Host admission：把不可信 Plan / Acceptance / Action 提案校验为可提交的决定。

    首轮必须有 Plan；后续修订保护工具调用准入后冻结的 Acceptance 身份。证据只能引用
    已有成功工具结果，工具动作还受目录、批次和最终收尾预算约束。
    返回规范化决定与物化 Plan，尚未写 Store 或执行工具；
    不推断逐项完成状态，最终候选还需单独 verification。
    """

    if current_plan is None:
        if decision.plan is None:
            raise L1ControllerFailure(
                RuntimeErrorCode.MODEL_OUTPUT_INVALID,
                "first L1 Attempt omitted its Plan",
                repair_issue=RuntimeModelOutputRepairIssue(
                    category="host_guard",
                    code="host_guard.l1_plan_required",
                    paths=("/plan",),
                    safe_explanation=(
                        "首次 Attempt 必须提供非 null 的 plan，包含 objective 和 acceptances；"
                        "新增项目省略 acceptance_id，由 Host 分配。"
                    ),
                ),
            )
        materialized_plan = _materialize_proposed_plan(
            decision.plan,
            input_message_id=(accepted.input_message_id or f"input_{accepted.turn_id}"),
            user_text=accepted.user_input,
            revision=1,
        )
    else:
        if decision.plan is None:
            materialized_plan = current_plan
        else:
            materialized_plan = _materialize_proposed_plan(
                decision.plan,
                input_message_id=(
                    accepted.input_message_id or f"input_{accepted.turn_id}"
                ),
                user_text=accepted.user_input,
                revision=current_plan.revision + 1,
                previous=current_plan,
            )
            try:
                protected_ids = protected_l1_acceptance_ids(
                    plan=current_plan,
                    execution=execution,
                )
                validate_l1_plan_revision(
                    current_plan,
                    materialized_plan,
                    protected_acceptance_ids=protected_ids,
                )
            except L1VerificationStateError as exc:
                raise L1ControllerFailure(
                    RuntimeErrorCode.INTERNAL_FAILURE,
                    "persisted L1 ToolCall evidence is invalid",
                ) from exc
            except L1PlanUnchangedError:
                # 线协议允许用 null 保留当前计划。若完整计划与当前计划
                # 完全重复，则规范化为同一含义；动作仍完全由本次模型响应决定。
                decision = decision.model_copy(update={"plan": None})
                materialized_plan = current_plan
            except L1PlanRevisionError as exc:
                raise L1ControllerFailure(
                    RuntimeErrorCode.MODEL_OUTPUT_INVALID,
                    str(exc),
                    repair_issue=exc.repair_issue,
                ) from exc
    known_tool_result_ids = _known_tool_result_ids(execution)
    for index, ref in enumerate(decision.references):
        if ref.tool_result_id not in known_tool_result_ids:
            raise L1ControllerFailure(
                RuntimeErrorCode.MODEL_OUTPUT_INVALID,
                "L1 references contain an unavailable ToolResult",
                repair_issue=RuntimeModelOutputRepairIssue(
                    category="host_guard",
                    code="host_guard.l1_unavailable_reference",
                    paths=(f"/references/{index}/tool_result_id",),
                    safe_explanation=(
                        "此 tool_result_id 不属于当前已成功执行的普通工具结果；"
                        "请使用当前 prior_tool_results 中可用的 tool_result_id，"
                        "或移除无法绑定的引用，不要编造 ID。"
                    ),
                ),
            )

    action = decision.action
    if isinstance(action, CallToolsAction):
        if finalization_required:
            raise L1ControllerFailure(
                RuntimeErrorCode.MODEL_OUTPUT_INVALID,
                "L1 decision called tools after the finalization boundary",
                repair_issue=RuntimeModelOutputRepairIssue(
                    category="host_guard",
                    code="host_guard.l1_finalization_required",
                    paths=("/action/kind",),
                    safe_explanation=(
                        "当前 execution_limits.finalization_required=true，不能再调用工具；"
                        "action.kind 必须为 submit_final_reply，并在 action.reply 中依据"
                        "已有结果交付答复；信息不足时如实说明限制。"
                    ),
                ),
            )
        if len(action.calls) > max_tool_calls_per_attempt:
            raise L1ControllerFailure(
                RuntimeErrorCode.MODEL_OUTPUT_INVALID,
                "L1 tool batch exceeded the per-Attempt limit",
                repair_issue=RuntimeModelOutputRepairIssue(
                    category="host_guard",
                    code="host_guard.l1_tool_batch_limit",
                    paths=("/action/calls",),
                    safe_explanation=(
                        f"当前 action.calls 有 {len(action.calls)} 次工具调用；"
                        f"本 Attempt 上限为 {max_tool_calls_per_attempt} 次。"
                        "请缩小当前批次，其余必要调用留到后续允许的 Attempt。"
                    ),
                ),
            )
        findings_call_count = 0
        for index, call in enumerate(action.calls):
            try:
                if not tool_runtime.knows_tool(call.tool_id):
                    raise ValueError(f"unknown L1 tool: {call.tool_id!r}")
            except Exception as exc:
                raise L1ControllerFailure(
                    RuntimeErrorCode.MODEL_OUTPUT_INVALID,
                    "L1 ToolCall referenced an unavailable tool",
                    repair_issue=RuntimeModelOutputRepairIssue(
                        category="host_guard",
                        code="host_guard.l1_unavailable_tool",
                        paths=(f"/action/calls/{index}/tool_id",),
                        safe_explanation=(
                            "此 tool_id 不在当前可用工具目录中；请只使用当前 tool_catalog "
                            "暴露的工具 ID，或移除此调用。"
                        ),
                    ),
                ) from exc
            if tool_runtime.is_execution_findings_tool(call.tool_id):
                findings_call_count += 1
        if findings_call_count > 1:
            raise L1ControllerFailure(
                RuntimeErrorCode.MODEL_OUTPUT_INVALID,
                "L1 tool batch may contain at most one findings mutation",
                repair_issue=RuntimeModelOutputRepairIssue(
                    category="host_guard",
                    code="host_guard.l1_findings_batch_limit",
                    paths=("/action/calls",),
                    safe_explanation=(
                        f"当前批次包含 {findings_call_count} 次 findings 写入/修改工具调用；"
                        "一批最多允许 1 次。请保留一次，其余留到后续允许的 Attempt。"
                    ),
                ),
            )
    return decision, materialized_plan


def _materialize_proposed_plan(
    proposal: L1PlanProposal,
    *,
    input_message_id: str,
    user_text: str,
    revision: int,
    previous: L1Plan | None = None,
) -> L1Plan:
    try:
        return materialize_l1_plan(
            proposal,
            input_message_id=input_message_id,
            user_text=user_text,
            revision=revision,
            previous=previous,
        )
    except ValueError as exc:
        # 提案已经通过 schema；仅从已知字段定位确定的未知 ID，不回灌异常原文。
        known_ids = (
            {item.acceptance_id for item in previous.acceptances} if previous else set()
        )
        unknown_index = next(
            (
                index
                for index, item in enumerate(proposal.acceptances)
                if item.acceptance_id is not None and item.acceptance_id not in known_ids
            ),
            None,
        )
        issue = (
            RuntimeModelOutputRepairIssue(
                category="host_guard",
                code="host_guard.l1_unknown_acceptance",
                paths=(f"/plan/acceptances/{unknown_index}/acceptance_id",),
                safe_explanation=(
                    "此 acceptance_id 不属于当前计划；修改既有项目只能使用当前 plan 的 ID，"
                    "新增项目必须省略 acceptance_id，由 Host 分配。"
                ),
            )
            if unknown_index is not None
            else RuntimeModelOutputRepairIssue(
                category="host_guard",
                code="host_guard.l1_plan_materialization_rejected",
                paths=("/plan",),
                safe_explanation=(
                    "Host 无法将此 plan 物化为有效计划；请检查 objective、acceptances "
                    "与当前计划的身份约束；保留已有计划时使用 plan=null。"
                ),
            )
        )
        raise L1ControllerFailure(
            RuntimeErrorCode.MODEL_OUTPUT_INVALID,
            f"L1 Plan could not be materialized: {exc}",
            repair_issue=issue,
        ) from exc


def _verify_final_reply(
    *,
    plan: L1Plan,
    reply: str,
    references: tuple[L1ResultReference, ...],
    execution: dict[str, object],
) -> L1VerificationResult:
    """调用纯机械验证，并区分“候选不合格”与“持久证据损坏”。

    前者抛 L1MechanicalVerificationRejected，供控制循环形成下一轮反馈；
    后者终止为内部状态错误。机械通过只证明字段/来源关系合法，不证明答案语义正确。
    """

    try:
        verification = verify_l1_final_reply(
            plan=plan,
            reply=reply,
            references=references,
            execution=execution,
        )
    except L1VerificationStateError as exc:
        raise L1ControllerFailure(
            RuntimeErrorCode.INTERNAL_FAILURE,
            "persisted L1 verification evidence is invalid",
        ) from exc
    if not verification.passed:
        raise L1MechanicalVerificationRejected(verification)
    return verification


def _semantic_receipt_for_final_reply(
    *,
    decision: L1AttemptDecisionProposal,
    plan: L1Plan,
    reply: str,
    mechanical_verification: L1VerificationResult,
    execution: dict[str, object],
    execution_features: dict[str, Any],
    model_view: dict[str, Any],
    accepted: AcceptedEntryTurn,
    l1_turn_run_id: str,
    attempt_id: str,
    state_guard_hash: str,
    emit: EntryEventEmitter,
    deadline: TurnDeadline,
    store: L1StorePort,
    model_binding: ModelTierBinding,
) -> tuple[L1SemanticVerificationReceipt | None, dict[str, Any] | None]:
    """机械门通过后，按冻结策略触发语义 reviewer 并构造绑定候选的 receipt。

    required 时调用独立逻辑模型请求；reviewer 的 revise 是有效审查结论，转成
    下一 L1 Attempt 的验证反馈，不是 HTTP 失败或 JSON repair。
    未触发也显式留下 not_required receipt；候选、Plan、机械报告和 state guard
    都按 hash 绑定，防止把另一个状态下的通过结论用于本次提交。
    """

    trigger = _semantic_trigger(
        execution_features=execution_features,
        plan=plan,
        mechanical_verification=mechanical_verification,
    )
    decision_hash = sha256_json(decision)
    plan_hash = sha256_json(plan)
    mechanical_hash = sha256_json(mechanical_verification.to_dict())
    invocation = None
    if trigger.required:
        try:
            invocation = request_l1_semantic_verification(
                turn_id=accepted.turn_id,
                session_id=accepted.session_id,
                l1_turn_run_id=l1_turn_run_id,
                attempt_id=attempt_id,
                state_guard_hash=state_guard_hash,
                decision_hash=decision_hash,
                plan=plan,
                plan_hash=plan_hash,
                reply=reply,
                references=decision.references,
                mechanical_verification=mechanical_verification,
                mechanical_verification_hash=mechanical_hash,
                model_view=model_view,
                candidate_note=decision.note,
                execution=execution,
                emit=emit,
                deadline=deadline,
                store=store,
                model_binding=model_binding,
            )
        except L1SemanticVerificationInputError as exc:
            return None, _verification_feedback(
                source="semantic",
                feedback=exc.safe_feedback,
                details={"code": exc.code},
            )
        if invocation.result.verdict != "pass":
            return None, _verification_feedback(
                source="semantic",
                feedback=invocation.result.safe_feedback(),
                details={
                    "reviewer_logical_call_id": invocation.logical_model_call_id,
                    "reviewer_result_hash": invocation.result_hash,
                    "reviewer_result": invocation.result.model_dump(mode="json"),
                },
            )
    try:
        return (
            build_l1_semantic_verification_receipt(
                trigger=trigger,
                decision_hash=decision_hash,
                plan_hash=plan_hash,
                mechanical_verification_hash=mechanical_hash,
                state_guard_hash=state_guard_hash,
                checked_acceptances=mechanical_verification.checked_acceptances,
                checked_tool_results=mechanical_verification.checked_tool_results,
                invocation=invocation,
            ),
            None,
        )
    except ValueError as exc:
        raise L1ControllerFailure(
            RuntimeErrorCode.INTERNAL_FAILURE,
            "L1 semantic verification receipt could not be materialized",
        ) from exc


def _verification_feedback(
    *,
    source: Literal["mechanical", "semantic"],
    feedback: str,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Attempt、L1：返回公共安全的 VerificationFeedback 用于下一个 L1Attempt。"""

    return {
        "schema_version": "l1-attempt-verification-feedback",
        "kind": "verification_rejected",
        "source": source,
        "feedback": feedback,
        "details": details or {},
    }


def _require_current_semantic_receipt(
    receipt: L1SemanticVerificationReceipt,
    *,
    decision: L1AttemptDecisionProposal,
    plan: L1Plan,
    mechanical_verification: L1VerificationResult,
    execution_features: dict[str, Any],
    state_guard_hash: str,
) -> None:
    """提交前重算候选绑定，确认语义 receipt 仍对应当前决定和验证状态。

    这里只比对策略、哈希与检查计数，不再次调用 reviewer；不匹配是 Host 状态错误，
    不能靠沿用旧的 pass 或 not_required 结论继续发布。
    """

    trigger = _semantic_trigger(
        execution_features=execution_features,
        plan=plan,
        mechanical_verification=mechanical_verification,
    )
    expected = {
        "trigger": trigger,
        "decision_hash": sha256_json(decision),
        "plan_hash": sha256_json(plan),
        "mechanical_verification_hash": sha256_json(mechanical_verification.to_dict()),
        "state_guard_hash": state_guard_hash,
        "checked_acceptances": mechanical_verification.checked_acceptances,
        "checked_tool_results": mechanical_verification.checked_tool_results,
    }
    if any(getattr(receipt, key) != value for key, value in expected.items()):
        raise L1ControllerFailure(
            RuntimeErrorCode.INTERNAL_FAILURE,
            "L1 semantic verification receipt crossed candidate authority",
        )


def _semantic_trigger(
    *,
    execution_features: dict[str, Any],
    plan: L1Plan,
    mechanical_verification: L1VerificationResult,
) -> L1SemanticVerificationTrigger:
    try:
        mode = parse_l1_semantic_verification_mode(
            execution_features.get(L1_SEMANTIC_VERIFICATION_FEATURE)
        )
        return derive_l1_semantic_verification_trigger(
            mode=mode,
            acceptance_count=len(plan.acceptances),
            tool_result_count=mechanical_verification.checked_tool_results,
        )
    except ValueError as exc:
        raise L1ControllerFailure(
            RuntimeErrorCode.MODEL_CONFIGURATION_FAILURE,
            "L1 semantic verification policy is invalid",
        ) from exc


def _execute_tool_batch(
    *,
    accepted: AcceptedEntryTurn,
    l1_turn_run_id: str,
    attempt_id: str,
    action: CallToolsAction,
    tool_runtime: L1ToolRuntime,
    deadline: TurnDeadline,
    emit: EntryEventEmitter,
    store: L1StorePort,
    max_tool_calls_per_attempt: int,
    expected_window_revision: int,
    expected_lease_owner: str | None,
    findings_ledger_id: str,
) -> dict[str, Any]:
    """按声明顺序执行一轮工具：prepare → durable reserve → dispatch → settle。

    每个序号先绑定工具版本、规范化参数和 policy。已结算调用直接复用 outcome；
    pending 调用才进入 findings、protected effect 或普通 ToolExecutor 分支。
    同批可以包含多个受保护调用；每条独立授权、派发、结算，不并行或组成全批事务。
    返回已结算结果投影；执行前拒绝也补记 trajectory，持久 ToolCall 仍是恢复权威。
    """

    emit(
        new_turn_event(
            turn_id=accepted.turn_id,
            session_id=accepted.session_id,
            stage=RuntimeStage.TOOL,
            status=TurnEventStatus.STARTED,
        )
    )
    tool_results: list[dict[str, Any]] = []
    for ordinal, call in enumerate(action.calls, start=1):
        prepared = tool_runtime.prepare(
            tool_id=call.tool_id,
            arguments=call.arguments,
            remaining_tool_calls=max_tool_calls_per_attempt - ordinal + 1,
        )
        arguments_json = canonical_json(prepared.normalized_arguments)
        policy_json = canonical_json(prepared.policy)
        reservation_kwargs: dict[str, object] = {
            "session_id": accepted.session_id,
            "turn_id": accepted.turn_id,
            "l1_turn_run_id": l1_turn_run_id,
            "attempt_id": attempt_id,
            "call_ordinal": ordinal,
            "tool_id": prepared.tool_id,
            "contract_version": prepared.contract_version,
            "implementation_version": prepared.implementation_version,
            "arguments_json": arguments_json,
            "arguments_hash": sha256_json(prepared.normalized_arguments),
            "policy_json": policy_json,
        }
        if prepared.requires_protected_dispatch:
            protected_authority = prepared.protected_authority
            assert protected_authority is not None
            receipt_ids = tuple(sorted(protected_authority.approval_receipt_ids))
            receipt_ids_json = canonical_json(list(receipt_ids))
            reservation_kwargs.update(
                execution_class="protected_effect",
                effect_profile_sha256=sha256_json(
                    [item.to_dict() for item in prepared.effect_profile.effects]
                ),
                provider_identity_sha256=(
                    protected_authority.execution_backend_identity_sha256
                ),
                approval_receipt_ids_json=receipt_ids_json,
                approval_receipts_sha256=sha256_json(list(receipt_ids)),
                expected_window_revision=expected_window_revision,
                expected_lease_owner=expected_lease_owner,
            )
        elif (
            tool_runtime.is_execution_findings_tool(prepared.tool_id)
            and prepared.rejected_outcome is None
        ):
            reservation_kwargs.update(
                execution_class="runtime_state",
                effect_profile_sha256=sha256_json(
                    [item.to_dict() for item in prepared.effect_profile.effects]
                ),
                expected_window_revision=expected_window_revision,
                expected_lease_owner=expected_lease_owner,
            )
        reservation = store.reserve_l1_tool_call(
            **reservation_kwargs,
        )
        stored_call = _mapping(reservation, "tool_call")
        tool_call_id = str(stored_call["tool_call_id"])
        # Replay：终态调用只读回持久 outcome；pending 才进入对应执行边界。
        if str(stored_call["status"]) == "pending":
            externally_dispatched = (
                prepared.requires_protected_dispatch
                or tool_runtime.is_execution_findings_tool(prepared.tool_id)
            )
            dispatch_denial = (
                tool_runtime.authorize_external_dispatch(prepared)
                if prepared.rejected_outcome is None and externally_dispatched
                else None
            )
            if dispatch_denial is not None:
                outcome_payload = dispatch_denial.to_dict()
                settled = store.settle_l1_tool_call(
                    tool_call_id=tool_call_id,
                    outcome_status=dispatch_denial.status.value,
                    outcome_json=canonical_json(outcome_payload),
                    outcome_hash=sha256_json(outcome_payload),
                )
                stored_call = _mapping(settled, "tool_call")
            # Findings 写工作笔记；protected effect 有专用账本；普通工具才走通用 executor。
            elif (
                tool_runtime.is_execution_findings_tool(prepared.tool_id)
                and prepared.rejected_outcome is None
            ):
                outcome = execute_execution_findings_tool(
                    store=store,
                    tool_id=prepared.tool_id,
                    normalized_arguments=prepared.normalized_arguments,
                    ledger_id=findings_ledger_id,
                    writer_unit_id=attempt_id,
                    writer_tool_call_id=tool_call_id,
                )
                outcome_payload = outcome.to_dict()
                settled = store.settle_l1_tool_call(
                    tool_call_id=tool_call_id,
                    outcome_status=outcome.status.value,
                    outcome_json=canonical_json(outcome_payload),
                    outcome_hash=sha256_json(outcome_payload),
                )
                stored_call = _mapping(settled, "tool_call")
            elif prepared.requires_protected_dispatch:
                protected_authority = prepared.protected_authority
                assert protected_authority is not None
                registration = prepared.registration
                assert registration is not None
                dispatched = L1ProtectedToolDispatcher(store=store).dispatch(
                    L1ProtectedToolDispatchRequest(
                        session_id=accepted.session_id,
                        turn_id=accepted.turn_id,
                        l1_turn_run_id=l1_turn_run_id,
                        attempt_id=attempt_id,
                        tool_call_id=tool_call_id,
                        expected_window_revision=expected_window_revision,
                        expected_lease_owner=expected_lease_owner,
                        protected_operation_binding_sha256=str(
                            stored_call["protected_operation_binding_sha256"]
                        ),
                        registration=registration,
                        arguments=prepared.normalized_arguments,
                        deadline_monotonic=deadline.expires_at_monotonic,
                        revalidate_authority=protected_authority.revalidate,
                        continuation_check=(
                            lambda: store.renew_l1_turn_run_resume_lease(
                                session_id=accepted.session_id, turn_id=accepted.turn_id,
                                l1_turn_run_id=l1_turn_run_id, lease_owner=expected_lease_owner,
                            )
                        ) if prepared.tool_id == "prepare_files" and expected_lease_owner is not None else None,
                    )
                )
                if not dispatched.durably_settled or dispatched.tool_call is None:
                    raise L1ControllerFailure(
                        RuntimeErrorCode.TOOL_COMPLETION_UNCONFIRMED,
                        "L1 protected ToolCall could not be durably settled",
                    )
                stored_call = dict(dispatched.tool_call)
            else:
                executed = tool_runtime.execute_prepared(
                    prepared,
                    deadline_monotonic=deadline.expires_at_monotonic,
                    logical_tool_call_id=tool_call_id,
                )
                outcome_payload = executed.outcome.to_dict()
                settled = store.settle_l1_tool_call(
                    tool_call_id=tool_call_id,
                    outcome_status=executed.outcome.status.value,
                    outcome_json=canonical_json(outcome_payload),
                    outcome_hash=sha256_json(outcome_payload),
                )
                stored_call = _mapping(settled, "tool_call")
        outcome = _parse_optional_json(stored_call.get("outcome_json"))
        if not isinstance(outcome, dict):
            raise L1ControllerFailure(
                RuntimeErrorCode.TOOL_COMPLETION_UNCONFIRMED,
                "L1 ToolCall has no durable outcome",
            )
        if str(stored_call["status"]) == ExecutionStatus.COMPLETION_UNCONFIRMED.value:
            # 记录 unknown 不等于 handler 已退出；不要与可能仍在进行的副作用交叠。
            raise L1ControllerFailure(
                RuntimeErrorCode.TOOL_COMPLETION_UNCONFIRMED,
                "L1 ToolCall completion is unknown; remaining batch calls were not started",
            )
        if prepared.rejected_outcome is not None:
            # 未经过 ToolExecutor 的拒绝也需完整留痕；先结算再记录，恢复时用持久 ID 去重。
            error = outcome.get("error")
            record_tool_call(
                tool_id=prepared.tool_id,
                arguments=prepared.normalized_arguments,
                status=str(stored_call["status"]),
                result=outcome.get("result"),
                error=error,
                error_code=error.get("code") if error else None,
                duration_ms=0,
                session_id=accepted.session_id,
                turn_id=accepted.turn_id,
                step_id=tool_call_id,
            )
        result_sha256 = str(stored_call.get("outcome_hash") or "")
        tool_results.append(
            {
                "tool_result_id": l1_tool_result_id(
                    tool_call_id=tool_call_id,
                    result_sha256=result_sha256,
                ),
                "tool_call_id": tool_call_id,
                "tool_id": str(stored_call["tool_id"]),
                "status": str(stored_call["status"]),
                "result_sha256": result_sha256,
                "result": outcome.get("result"),
                "error": outcome.get("error"),
                "metadata": outcome.get("metadata"),
            }
        )
    emit(
        new_turn_event(
            turn_id=accepted.turn_id,
            session_id=accepted.session_id,
            stage=RuntimeStage.TOOL,
            status=TurnEventStatus.COMPLETED,
        )
    )
    emit(
        new_turn_event(
            turn_id=accepted.turn_id,
            session_id=accepted.session_id,
            stage=RuntimeStage.OBSERVATION,
            status=TurnEventStatus.COMPLETED,
        )
    )
    return {
        "schema_version": "l1-attempt-tool-results-v1",
        "attempt_id": attempt_id,
        "tool_results": tool_results,
    }


def _current_attempt(execution: dict[str, object]) -> dict[str, Any] | None:
    attempts = execution.get("attempts")
    if not isinstance(attempts, list):
        raise L1ControllerFailure(
            RuntimeErrorCode.INTERNAL_FAILURE,
            "persisted L1 Attempts have the wrong shape",
        )
    if not attempts:
        return None
    attempt = attempts[-1]
    if not isinstance(attempt, dict):
        raise L1ControllerFailure(
            RuntimeErrorCode.INTERNAL_FAILURE,
            "persisted L1 current Attempt has the wrong shape",
        )
    return attempt


def _submitted_final_reply_result(
    *,
    accepted: AcceptedEntryTurn,
    attempt: dict[str, Any],
    revision: int,
) -> L1ControllerResult:
    if str(attempt.get("turn_id") or "") != accepted.turn_id:
        raise L1ControllerFailure(
            RuntimeErrorCode.INTERNAL_FAILURE,
            "persisted L1 final reply crossed Turn ownership",
        )
    decision = _load_decision(attempt.get("decision_json"))
    action = decision.action
    if not isinstance(action, SubmitFinalReplyAction):
        raise L1ControllerFailure(
            RuntimeErrorCode.INTERNAL_FAILURE,
            "persisted L1 submitted Attempt has incomplete authority",
        )
    return L1ControllerResult(
        reply=action.reply,
        turn_window_revision=revision,
    )


def _load_decision(value: object) -> L1AttemptDecisionProposal:
    if not isinstance(value, str):
        raise L1ControllerFailure(
            RuntimeErrorCode.INTERNAL_FAILURE,
            "persisted L1 decision has the wrong representation",
        )
    try:
        return L1AttemptDecisionProposal.model_validate_json(value)
    except Exception as exc:
        raise L1ControllerFailure(
            RuntimeErrorCode.INTERNAL_FAILURE,
            "persisted L1 decision failed validation",
        ) from exc


def _load_plan(value: object) -> L1Plan | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise L1ControllerFailure(
            RuntimeErrorCode.INTERNAL_FAILURE,
            "persisted L1 Plan has the wrong representation",
        )
    try:
        return L1Plan.model_validate_json(value)
    except Exception as exc:
        raise L1ControllerFailure(
            RuntimeErrorCode.INTERNAL_FAILURE,
            "persisted L1 Plan failed validation",
        ) from exc


def _deadline_from_utc(value: str) -> TurnDeadline:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise L1ControllerFailure(
            RuntimeErrorCode.INTERNAL_FAILURE,
            "persisted L1 deadline is invalid",
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise L1ControllerFailure(
            RuntimeErrorCode.INTERNAL_FAILURE,
            "persisted L1 deadline must be timezone-aware",
        )
    remaining = (
        parsed.astimezone(timezone.utc) - datetime.now(timezone.utc)
    ).total_seconds()
    return TurnDeadline.starting_now(remaining)


def _persisted_finalization_required(payload: dict[str, Any]) -> bool:
    limits = payload.get("execution_limits")
    if not isinstance(limits, dict) or not isinstance(
        limits.get("finalization_required"),
        bool,
    ):
        raise L1ControllerFailure(
            RuntimeErrorCode.INTERNAL_FAILURE,
            "persisted L1 model request omitted its finalization boundary",
        )
    return bool(limits["finalization_required"])


def _required_json_mapping(value: object, *, label: str) -> dict[str, Any]:
    if not isinstance(value, str):
        raise L1ControllerFailure(
            RuntimeErrorCode.INTERNAL_FAILURE,
            f"{label} has the wrong representation",
        )
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError) as exc:
        raise L1ControllerFailure(
            RuntimeErrorCode.INTERNAL_FAILURE,
            f"{label} is invalid JSON",
        ) from exc
    if not isinstance(parsed, dict):
        raise L1ControllerFailure(
            RuntimeErrorCode.INTERNAL_FAILURE,
            f"{label} must be a JSON object",
        )
    return parsed


def _required_text(value: dict[str, Any], key: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item:
        raise L1ControllerFailure(
            RuntimeErrorCode.INTERNAL_FAILURE,
            f"persisted L1 state omitted {key}",
        )
    return item


def _optional_mapping(
    value: dict[str, object] | None,
    key: str,
) -> dict[str, Any] | None:
    if value is None or value.get(key) is None:
        return None
    item = value.get(key)
    if not isinstance(item, dict):
        raise L1ControllerFailure(
            RuntimeErrorCode.INTERNAL_FAILURE,
            f"L1 Store projection has invalid {key}",
        )
    return item


def _turn_window_revision(value: dict[str, object]) -> int:
    return int(_mapping(value, "window")["state_version"])


def _mapping(value: object, key: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not isinstance(value.get(key), dict):
        raise L1ControllerFailure(
            RuntimeErrorCode.INTERNAL_FAILURE,
            f"L1 Store projection omitted {key}",
        )
    return value[key]  # type: ignore[return-value]


def _parse_optional_json(value: object) -> object | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise L1ControllerFailure(
            RuntimeErrorCode.INTERNAL_FAILURE,
            "L1 persisted JSON has the wrong representation",
        )
    try:
        return json.loads(value)
    except (TypeError, ValueError) as exc:
        raise L1ControllerFailure(
            RuntimeErrorCode.INTERNAL_FAILURE,
            "L1 persisted JSON is invalid",
        ) from exc


def _json_projection(
    value: object,
    *,
    redact_attachment_authority: bool = False,
) -> object:
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        excluded_fields = {"stored_rel_path"}
        if redact_attachment_authority:
            excluded_fields.update({"attachment_id", "content_hash"})
        return model_dump(
            mode="json",
            exclude={"items": {"__all__": excluded_fields}},
        )
    return thaw_json(value)


__all__ = ["L1ControllerFailure", "L1ControllerResult", "run_l1_turn"]
