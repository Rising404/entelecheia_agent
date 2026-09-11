"""针对被拒绝的 AuxiliaryGraph 提案，执行与原因绑定的自主重规划。

此控制器负责一次有界的 ``N -> N+1`` AuxiliaryGraph 修订。唯一允许的原因是经过认证、结论并非 PASS 的 TaskGraph 语义共识。Store 会在分发 Architect 前根据该共识推导不可变触发器；只有在精确的 ``verification_failed`` 修订存在后，第二份不可变回执才会消费该触发器。

重新进入的处理刻意前置：已完成的应用会直接返回，不再执行模型工作；若活动触发器对应的 ``N+1`` 修订已存在，则只消费触发器而不重新分发；若活动触发器仍停留在 ``N``，则重建同一 Architect 请求，让持久化模型账本可以重放其类型化结果。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
import hashlib
import json

from personagraph.l2.auxiliary_graph import (
    AuxiliaryGraphRevisionReason,
    AuxiliaryReplanTriggerApplicationReceipt,
    AuxiliaryReplanTriggerReceipt,
    PlanningAuthorityProjection,
    PlanningCapabilityCatalogProjection,
    TaskGraphSemanticVerificationDisposition,
)
from personagraph.l2.auxiliary_graph.contracts import (
    is_task_graph_semantic_user_information_block,
)
from personagraph.session.l2_store import auxiliary_graph as auxiliary_graph_store
from personagraph.session.l2_store.auxiliary_graph import (
    AuxiliaryGraphRevisionCommitResult,
    StoredAuxiliaryGraphDetails,
)
from personagraph.session.l2_store import planning as planning_store
from personagraph.session.l2_store import semantic_verification as semantic_store
from personagraph.session.l2_store import task_graph as task_graph_store
from personagraph.session.l2_store.semantic_verification import (
    StoredAuxiliarySemanticQuorumSettlement,
)
from .architect import (
    AuxiliaryGraphArchitectAction,
    AuxiliaryGraphArchitectDecision,
    AuxiliaryGraphArchitectReplanTrigger,
    AuxiliaryGraphArchitectRequest,
    AuxiliaryGraphArchitectStructuredProvider,
    AuxiliaryGraphReplanReviewerFindings,
    request_auxiliary_graph_architect,
)
from .architect_adapter import (
    auxiliary_graph_revision_proposal_from_architect_decision,
)
from ..adapters.model_authority import (
    create_auxiliary_graph_architect_model_call_authority,
)
from .model_provider import (
    build_auxiliary_architect_structured_provider,
)
from .profiles import (
    build_auxiliary_architect_request,
    build_auxiliary_planning_capability_catalog,
    canonical_auxiliary_architect_state_guard,
)
from .mounted_document_authority import (
    build_mounted_document_authority_projection,
    freeze_mounted_document_planning_authority,
    recover_task_scoped_managed_document_ids,
)
from personagraph.tools.workspace.session_read_source import (
    build_session_workspace_readonly_runtime,
)
from personagraph.runtime.turn_deadline import TurnDeadline
from personagraph.runtime.model_calls.contracts import RuntimeModelLedgerStore
from personagraph.runtime.turn_events import TurnEvent


_SHA256_ZERO = "0" * 64


class AuxiliaryReplanningStatus(StrEnum):
    REPLANNED = "replanned"
    ALREADY_REPLANNED = "already_replanned"
    LIMIT_REACHED = "limit_reached"


@dataclass(frozen=True, slots=True)
class AuxiliaryReplanningRequest:
    session_id: str
    turn_id: str
    task_id: str
    settlement: StoredAuxiliarySemanticQuorumSettlement
    max_autonomous_replans: int = 3
    deadline: TurnDeadline | None = None

    def __post_init__(self) -> None:
        for name in ("session_id", "turn_id", "task_id"):
            _require_identifier(name, getattr(self, name))
        if not isinstance(
            self.settlement,
            StoredAuxiliarySemanticQuorumSettlement,
        ):
            raise TypeError(
                "settlement must be StoredAuxiliarySemanticQuorumSettlement"
            )
        if (
            isinstance(self.max_autonomous_replans, bool)
            or not isinstance(self.max_autonomous_replans, int)
            or not 1 <= self.max_autonomous_replans <= 12
        ):
            raise ValueError("max_autonomous_replans must be within 1..12")


@dataclass(frozen=True, slots=True)
class AuxiliaryReplanningIds:
    trigger_id: str
    trigger_create_apply_id: str

    def __post_init__(self) -> None:
        _require_identifier("trigger_id", self.trigger_id)
        _require_identifier("trigger_create_apply_id", self.trigger_create_apply_id)


@dataclass(frozen=True, slots=True)
class AuxiliaryReplanningResult:
    status: AuxiliaryReplanningStatus
    reason_code: str
    stable_ids: AuxiliaryReplanningIds
    trigger: AuxiliaryReplanTriggerReceipt | None = None
    architect_decision: AuxiliaryGraphArchitectDecision | None = None
    revision_commit: AuxiliaryGraphRevisionCommitResult | None = None
    application: AuxiliaryReplanTriggerApplicationReceipt | None = None
    trigger_replayed: bool = False
    application_replayed: bool = False
    model_call_id: str | None = None
    model_attempts: int = 0
    model_replayed: bool = False

    def __post_init__(self) -> None:
        if not self.reason_code or len(self.reason_code) > 240:
            raise ValueError("replanning result requires a bounded reason code")
        if not 0 <= self.model_attempts <= 3:
            raise ValueError("replanning model attempt count is invalid")
        if self.status is AuxiliaryReplanningStatus.REPLANNED and (
            self.trigger is None
            or self.architect_decision is None
            or self.revision_commit is None
            or self.application is None
            or self.model_call_id is None
        ):
            raise ValueError("REPLANNED requires the complete revision authority")
        if self.status is AuxiliaryReplanningStatus.ALREADY_REPLANNED and (
            self.trigger is None or self.application is None
        ):
            raise ValueError(
                "ALREADY_REPLANNED requires trigger and application authority"
            )
        if self.status is AuxiliaryReplanningStatus.LIMIT_REACHED and any(
            value is not None
            for value in (
                self.trigger,
                self.architect_decision,
                self.revision_commit,
                self.application,
                self.model_call_id,
            )
        ):
            raise ValueError("LIMIT_REACHED cannot expose unapplied authority")


class AuxiliaryReplanningError(RuntimeError):
    """一个语义结算无法授权一个精确的图修订版本。"""

    code = "auxiliary_v2_replanning_rejected"


TurnEventEmitter = Callable[[TurnEvent], object]


def derive_auxiliary_replanning_ids(
    settlement: StoredAuxiliarySemanticQuorumSettlement,
) -> AuxiliaryReplanningIds:
    """从已结算的原因中推导出过程独立的触发器标识。"""

    if not isinstance(settlement, StoredAuxiliarySemanticQuorumSettlement):
        raise TypeError(
            "settlement must be StoredAuxiliarySemanticQuorumSettlement"
        )
    identity = _stable_digest(
        {
            "schema_version": "auxiliary-v2-replanning-trigger-identity-v1",
            "session_id": settlement.session_id,
            "task_id": settlement.task_id,
            "auxiliary_graph_id": settlement.auxiliary_graph_id,
            "goal_id": settlement.goal_id,
            "auxiliary_graph_revision": settlement.auxiliary_graph_revision,
            "frozen_prompt_payload_sha256": (
                settlement.frozen_prompt_payload_sha256
            ),
            "semantic_settlement_id": settlement.settlement_id,
            "semantic_settlement_sha256": settlement.settlement_sha256,
        }
    )
    return AuxiliaryReplanningIds(
        trigger_id=f"auxreplan_{identity[:40]}",
        trigger_create_apply_id=f"auxreplancreate_{identity[:40]}",
    )


def run_auxiliary_replanning(
    request: AuxiliaryReplanningRequest,
    *,
    provider: AuxiliaryGraphArchitectStructuredProvider | None = None,
    emit: TurnEventEmitter,
    ledger_store: RuntimeModelLedgerStore,
    knowledge_cognition_history_enabled: bool | None = None,
) -> AuxiliaryReplanningResult:
    """应用或恢复恰好一个语义失败的 AuxiliaryGraph 修订版本。"""

    if not isinstance(request, AuxiliaryReplanningRequest):
        raise TypeError("request must be AuxiliaryReplanningRequest")
    if not callable(emit):
        raise TypeError("emit must be callable")
    capability_switches = (
        knowledge_cognition_history_enabled,
    )
    rebuild_current_capabilities = any(
        value is not None for value in capability_switches
    )
    if rebuild_current_capabilities and any(
        not isinstance(value, bool) for value in capability_switches
    ):
        raise TypeError(
            "replanning capability switches must be supplied together as booleans"
        )

    settlement = _require_exact_stored_settlement(request)
    stable_ids = derive_auxiliary_replanning_ids(settlement)
    existing_application = planning_store.get_auxiliary_replan_trigger_application(
        session_id=request.session_id,
        trigger_id=stable_ids.trigger_id,
    )
    if existing_application is not None:
        trigger = _require_trigger_for_settlement(
            planning_store.get_auxiliary_replan_trigger(
                session_id=request.session_id,
                trigger_id=stable_ids.trigger_id,
            ),
            settlement=settlement,
            stable_ids=stable_ids,
        )
        if existing_application.trigger_id != trigger.trigger_id:
            raise AuxiliaryReplanningError(
                "stored replan application crossed its settlement trigger"
            )
        return AuxiliaryReplanningResult(
            status=AuxiliaryReplanningStatus.ALREADY_REPLANNED,
            reason_code="semantic_replan_already_applied",
            stable_ids=stable_ids,
            trigger=trigger,
            application=existing_application,
            trigger_replayed=True,
            application_replayed=True,
        )

    active = planning_store.get_active_auxiliary_replan_trigger(
        session_id=request.session_id,
        task_id=request.task_id,
    )
    trigger_replayed = active is not None
    if active is None:
        details = _require_current_details(request.session_id, request.task_id)
        _require_settlement_current(details=details, settlement=settlement)
        count = planning_store.count_auxiliary_replan_triggers(
            session_id=request.session_id,
            task_id=request.task_id,
            goal_id=settlement.goal_id,
        )
        if count >= request.max_autonomous_replans:
            return AuxiliaryReplanningResult(
                status=AuxiliaryReplanningStatus.LIMIT_REACHED,
                reason_code="autonomous_replan_limit_reached",
                stable_ids=stable_ids,
            )
        trigger_result = planning_store.create_auxiliary_replan_trigger(
            command=planning_store.CreateAuxiliaryReplanTriggerCommand(
                apply_id=stable_ids.trigger_create_apply_id,
                trigger_id=stable_ids.trigger_id,
                session_id=request.session_id,
                created_turn_id=request.turn_id,
                task_id=request.task_id,
                auxiliary_graph_id=details.auxiliary_graph_id,
                goal_id=details.goal_id,
                expected_control_state_version=details.control_state_version,
                expected_goal_state_version=details.goal_state_version,
                expected_revision_state_version=details.revision_state_version,
                expected_budget_state_version=details.budget_state_version,
                expected_current_auxiliary_graph_revision=(
                    details.auxiliary_graph_revision
                ),
                expected_current_structure_sha256=details.structure_sha256,
                expected_current_authority_snapshot_id=(
                    details.authority_snapshot_id
                ),
                expected_current_authority_snapshot_sha256=(
                    details.authority_snapshot_sha256
                ),
                semantic_settlement_id=settlement.settlement_id,
                expected_semantic_settlement_sha256=(
                    settlement.settlement_sha256
                ),
            )
        )
        trigger = trigger_result.trigger
        trigger_replayed = trigger_result.status == "replayed"
    else:
        trigger = active
    trigger = _require_trigger_for_settlement(
        trigger,
        settlement=settlement,
        stable_ids=stable_ids,
    )

    details = _require_current_details(request.session_id, request.task_id)
    if details.auxiliary_graph_revision == trigger.source_auxiliary_graph_revision + 1:
        application_result = _consume_trigger(
            request=request,
            trigger=trigger,
            details=details,
        )
        return AuxiliaryReplanningResult(
            status=AuxiliaryReplanningStatus.ALREADY_REPLANNED,
            reason_code="semantic_replan_revision_recovered",
            stable_ids=stable_ids,
            trigger=trigger,
            application=application_result.application,
            trigger_replayed=True,
            application_replayed=application_result.status == "replayed",
        )
    if details.auxiliary_graph_revision != trigger.source_auxiliary_graph_revision:
        raise AuxiliaryReplanningError(
            "active replan trigger does not bind current revision N or N+1"
        )
    _require_settlement_current(details=details, settlement=settlement)

    task = task_graph_store.get_insession_task_details(
        request.session_id,
        request.task_id,
    )
    if task is None or task.insession_task_id != request.task_id:
        raise AuxiliaryReplanningError("replanning Task authority is missing")
    architect_trigger = _build_architect_trigger(
        details=details,
        settlement=settlement,
    )
    semantic_prompt = settlement.requests[0].prompt_payload
    planning_capabilities = (
        _build_current_planning_capability_catalog(
            details=details,
            knowledge_cognition_history_enabled=(
                knowledge_cognition_history_enabled
            ),
        )
        if rebuild_current_capabilities
        else semantic_prompt.capabilities
    )
    architect_request = _build_architect_request(
        details=details,
        settlement=settlement,
        replan_trigger=architect_trigger,
        capabilities=planning_capabilities,
    )

    def rederive_state_guard() -> str:
        try:
            current_task = task_graph_store.get_insession_task_details(
                request.session_id,
                request.task_id,
            )
            current_details = _require_current_details(
                request.session_id,
                request.task_id,
            )
            current_settlement = _require_exact_stored_settlement(request)
            current_trigger = planning_store.get_active_auxiliary_replan_trigger(
                session_id=request.session_id,
                task_id=request.task_id,
            )
            if (
                current_task != task
                or current_details != details
                or current_settlement != settlement
                or current_trigger != trigger
            ):
                return _SHA256_ZERO
            current_architect_trigger = _build_architect_trigger(
                details=current_details,
                settlement=current_settlement,
            )
            current_capabilities = (
                _build_current_planning_capability_catalog(
                    details=current_details,
                    knowledge_cognition_history_enabled=(
                        knowledge_cognition_history_enabled
                    ),
                )
                if rebuild_current_capabilities
                else current_settlement.requests[0].prompt_payload.capabilities
            )
            if current_capabilities != planning_capabilities:
                return _SHA256_ZERO
            current_request = _build_architect_request(
                details=current_details,
                settlement=current_settlement,
                replan_trigger=current_architect_trigger,
                capabilities=current_capabilities,
            )
            if current_request != architect_request:
                return _SHA256_ZERO
            return canonical_auxiliary_architect_state_guard(current_request)
        except Exception:
            return _SHA256_ZERO

    authority_values: dict[str, object] = {
        "request": architect_request,
        "invocation_turn_id": request.turn_id,
        "rederive_state_guard_sha256": rederive_state_guard,
        "ledger_store": ledger_store,
    }
    durable_call = create_auxiliary_graph_architect_model_call_authority(
        **authority_values  # type: ignore[arg-type]
    )
    model_result = request_auxiliary_graph_architect(
        architect_request,
        invocation_turn_id=request.turn_id,
        provider=provider or build_auxiliary_architect_structured_provider(),
        emit=emit,
        deadline=request.deadline,
        durable_call=durable_call,
    )
    decision = model_result.value
    if (
        decision.action is not AuxiliaryGraphArchitectAction.REVISE_REVISION
        or decision.proposal.revision_reason
        is not AuxiliaryGraphRevisionReason.VERIFICATION_FAILED
    ):
        raise AuxiliaryReplanningError(
            "semantic replan Architect did not return verification_failed revision"
        )
    revision_proposal = (
        auxiliary_graph_revision_proposal_from_architect_decision(
            decision,
            expected_current_auxiliary_graph_revision=(
                trigger.source_auxiliary_graph_revision
            ),
        )
    )
    if (
        revision_proposal.revision_reason
        is not AuxiliaryGraphRevisionReason.VERIFICATION_FAILED
    ):
        raise AuxiliaryReplanningError(
            "Architect adapter changed the reason-bound replan cause"
        )

    durable_call.require_current_state()
    revision_apply_id = _stable_id(
        "auxreplanrevision",
        {
            "trigger_receipt_sha256": trigger.receipt_sha256,
            "architect_request_binding_sha256": architect_request.binding_sha256,
            "architect_decision_sha256": decision.decision_sha256,
        },
    )
    revision_commit = auxiliary_graph_store.commit_auxiliary_graph_revision(
        session_id=request.session_id,
        turn_id=request.turn_id,
        insession_task_id=request.task_id,
        expected_task_state_version=task.task_state_version,
        expected_base_task_graph_revision=details.base_task_graph_revision,
        expected_control_state_version=details.control_state_version,
        expected_current_auxiliary_graph_revision=details.auxiliary_graph_revision,
        apply_id=revision_apply_id,
        goal_objective=task.objective,
        proposal=revision_proposal,
        authority_context=_authority_context_for_revision(details),
        budget_profile=details.budget.base_profile.model_dump(mode="json"),
        auxiliary_graph_id=details.auxiliary_graph_id,
        goal_id=details.goal_id,
        terminal_candidate_semantic_settlement_id=(
            settlement.settlement_id
            if settlement.requests[0].terminal_candidate_binding is not None
            else None
        ),
    )
    committed = _require_current_details(request.session_id, request.task_id)
    if (
        committed.auxiliary_graph_revision
        != trigger.source_auxiliary_graph_revision + 1
        or committed.parent_auxiliary_graph_revision
        != trigger.source_auxiliary_graph_revision
        or committed.structure_sha256 != revision_commit.structure_sha256
        or committed.goal_id != trigger.goal_id
        or committed.reason
        != AuxiliaryGraphRevisionReason.VERIFICATION_FAILED.value
    ):
        raise AuxiliaryReplanningError(
            "semantic replan commit did not become exact current revision N+1"
        )
    application_result = _consume_trigger(
        request=request,
        trigger=trigger,
        details=committed,
    )
    return AuxiliaryReplanningResult(
        status=AuxiliaryReplanningStatus.REPLANNED,
        reason_code="semantic_replan_revision_committed",
        stable_ids=stable_ids,
        trigger=trigger,
        architect_decision=decision,
        revision_commit=revision_commit,
        application=application_result.application,
        trigger_replayed=trigger_replayed,
        application_replayed=application_result.status == "replayed",
        model_call_id=model_result.model_call_id,
        model_attempts=model_result.attempts,
        model_replayed=model_result.replayed,
    )


def _require_exact_stored_settlement(
    request: AuxiliaryReplanningRequest,
) -> StoredAuxiliarySemanticQuorumSettlement:
    settlement = request.settlement
    if (
        settlement.session_id != request.session_id
        or settlement.task_id != request.task_id
        or settlement.created_turn_id != request.turn_id
        or settlement.host_disposition
        not in {
            TaskGraphSemanticVerificationDisposition.REVISE,
            TaskGraphSemanticVerificationDisposition.BLOCKED,
        }
    ):
        raise AuxiliaryReplanningError(
            "semantic settlement is not an exact non-PASS request authority"
        )
    stored = semantic_store.get_auxiliary_semantic_quorum_settlement(
        session_id=settlement.session_id,
        task_id=settlement.task_id,
        auxiliary_graph_id=settlement.auxiliary_graph_id,
        goal_id=settlement.goal_id,
        auxiliary_graph_revision=settlement.auxiliary_graph_revision,
        frozen_prompt_payload_sha256=(
            settlement.frozen_prompt_payload_sha256
        ),
    )
    if stored is None or stored != settlement:
        raise AuxiliaryReplanningError(
            "semantic settlement differs from authenticated Store authority"
        )
    _require_blocked_settlement_is_user_information(stored)
    return stored


def _require_blocked_settlement_is_user_information(
    settlement: StoredAuxiliarySemanticQuorumSettlement,
) -> None:
    """保持正式的权威/访问限制在文本外的 UserGate 路径之外。"""

    if (
        settlement.host_disposition
        is not TaskGraphSemanticVerificationDisposition.BLOCKED
    ):
        return
    if not is_task_graph_semantic_user_information_block(
        requests=settlement.requests,
        results=settlement.results,
    ):
        raise AuxiliaryReplanningError(
            "blocked semantic settlement is not a typed ordinary-information gap"
        )


def _require_settlement_current(
    *,
    details: StoredAuxiliaryGraphDetails,
    settlement: StoredAuxiliarySemanticQuorumSettlement,
) -> None:
    first = settlement.requests[0]
    if (
        details.session_id != settlement.session_id
        or details.task_id != settlement.task_id
        or details.auxiliary_graph_id != settlement.auxiliary_graph_id
        or details.goal_id != settlement.goal_id
        or details.auxiliary_graph_revision
        != settlement.auxiliary_graph_revision
        or details.structure_sha256
        != first.auxiliary_graph_structure_sha256
        or details.authority_snapshot_id
        != first.authority_projection.authority_snapshot_id
        or details.authority_snapshot_sha256
        != first.authority_projection.authority_snapshot_sha256
        or details.budget != first.budget
        or details.goal != first.goal
    ):
        raise AuxiliaryReplanningError(
            "semantic settlement no longer binds current graph authority"
        )


def _require_trigger_for_settlement(
    trigger: AuxiliaryReplanTriggerReceipt | None,
    *,
    settlement: StoredAuxiliarySemanticQuorumSettlement,
    stable_ids: AuxiliaryReplanningIds,
) -> AuxiliaryReplanTriggerReceipt:
    if (
        trigger is None
        or trigger.trigger_id != stable_ids.trigger_id
        or trigger.create_apply_id != stable_ids.trigger_create_apply_id
        or trigger.session_id != settlement.session_id
        or trigger.task_id != settlement.task_id
        or trigger.auxiliary_graph_id != settlement.auxiliary_graph_id
        or trigger.goal_id != settlement.goal_id
        or trigger.source_auxiliary_graph_revision
        != settlement.auxiliary_graph_revision
        or trigger.semantic_settlement_id != settlement.settlement_id
        or trigger.semantic_settlement_sha256 != settlement.settlement_sha256
        or trigger.semantic_prompt_payload_sha256
        != settlement.frozen_prompt_payload_sha256
        or trigger.semantic_disposition != settlement.host_disposition
        or trigger.semantic_result_sha256s
        != tuple(item.result_sha256 for item in settlement.results)
    ):
        raise AuxiliaryReplanningError(
            "stored replan trigger differs from the exact semantic settlement"
        )
    return trigger


def _build_architect_trigger(
    *,
    details: StoredAuxiliaryGraphDetails,
    settlement: StoredAuxiliarySemanticQuorumSettlement,
) -> AuxiliaryGraphArchitectReplanTrigger:
    _require_settlement_current(details=details, settlement=settlement)
    first = settlement.requests[0]
    reviewers = tuple(
        AuxiliaryGraphReplanReviewerFindings(
            reviewer_ordinal=result.reviewer_ordinal,
            semantic_result_sha256=result.result_sha256,
            items=result.items,
        )
        for result in settlement.results
    )
    if tuple(item.reviewer_ordinal for item in reviewers) != tuple(
        range(1, len(reviewers) + 1)
    ):
        raise AuxiliaryReplanningError(
            "semantic settlement reviewers are not canonically ordered"
        )
    return AuxiliaryGraphArchitectReplanTrigger.create(
        expected_current_auxiliary_graph_revision=(
            details.auxiliary_graph_revision
        ),
        expected_current_structure_sha256=details.structure_sha256,
        semantic_host_disposition=settlement.host_disposition,
        semantic_settlement_id=settlement.settlement_id,
        semantic_settlement_sha256=settlement.settlement_sha256,
        semantic_prompt_payload_sha256=(
            settlement.frozen_prompt_payload_sha256
        ),
        rejected_task_graph_proposal=first.prompt_payload.task_graph_proposal,
        reviewers=reviewers,
    )


def _build_architect_request(
    *,
    details: StoredAuxiliaryGraphDetails,
    settlement: StoredAuxiliarySemanticQuorumSettlement,
    replan_trigger: AuxiliaryGraphArchitectReplanTrigger,
    capabilities: PlanningCapabilityCatalogProjection | None = None,
) -> AuxiliaryGraphArchitectRequest:
    prompt = settlement.requests[0].prompt_payload
    return build_auxiliary_architect_request(
        details=details,
        authority=_build_replan_architect_authority(
            details=details,
            semantic_authority=prompt.authority,
        ),
        capabilities=capabilities or prompt.capabilities,
        objective=prompt.goal.objective,
        desired_output=prompt.goal.desired_output,
        context_artifacts=prompt.context_artifacts,
        replan_trigger=replan_trigger,
    )


def _build_current_planning_capability_catalog(
    *,
    details: StoredAuxiliaryGraphDetails,
    knowledge_cognition_history_enabled: bool | None,
) -> PlanningCapabilityCatalogProjection:
    """冻结当前任务范围内的 Host 权威状态的能效表面。"""

    if any(
        not isinstance(value, bool)
        for value in (
                knowledge_cognition_history_enabled,
            )
    ):
        raise TypeError("current replanning capability switches must be booleans")
    allowed_managed_document_ids = recover_task_scoped_managed_document_ids(
        session_id=details.session_id,
        task_id=details.task_id,
        authority_snapshot=details.authority_snapshot,
    )
    mounted = freeze_mounted_document_planning_authority(
        session_id=details.session_id,
        task_id=details.task_id,
        allowed_managed_document_ids=allowed_managed_document_ids,
    )
    creation_source = task_graph_store.get_insession_task_creation_source(
        session_id=details.session_id,
        insession_task_id=details.task_id,
    )
    build_mounted_document_authority_projection(
        authority_snapshot=details.authority_snapshot,
        task_creation_source=creation_source,
        mounted_authority=mounted,
    )
    workspace_runtime = build_session_workspace_readonly_runtime(
        details.session_id,
    )
    return build_auxiliary_planning_capability_catalog(
        mounted,
        workspace_runtime=workspace_runtime,
        knowledge_cognition_history_enabled=knowledge_cognition_history_enabled,
    )


def _build_replan_architect_authority(
    *,
    details: StoredAuxiliaryGraphDetails,
    semantic_authority: PlanningAuthorityProjection,
) -> PlanningAuthorityProjection:
    """将持久化规划资源与精确语义观察结合。

    语义验证故意接收权威状态和执行观察；它不接收初始 Architect 使用的挂载资源卡片。重新规划必须看到两个命名空间：当前图仍然引用挂载资源，而审阅发现则引用观察结果。不可变的初始规划回执是恢复启动后提示安全资源卡片的唯一来源。

    每个恢复的卡片都被重新绑定到当前修订版独立认证的权威状态快照。修订提交分配一个新的快照标识符，因此在比较修订版2 之后，修订版会比较别名、权威类和投影哈希，而不是错误地要求基础 ID。
    """

    snapshot = details.authority_snapshot
    if snapshot is None:
        raise AuxiliaryReplanningError(
            "current replan graph lost its authority snapshot"
        )
    if (
        snapshot.authority_snapshot_id != details.authority_snapshot_id
        or snapshot.snapshot_sha256 != details.authority_snapshot_sha256
        or snapshot.session_id != details.session_id
        or snapshot.task_id != details.task_id
        or snapshot.auxiliary_graph_id != details.auxiliary_graph_id
        or snapshot.goal_id != details.goal_id
    ):
        raise AuxiliaryReplanningError(
            "current replan authority snapshot is not bound to the graph"
        )
    if (
        semantic_authority.authority_snapshot_id
        != details.authority_snapshot_id
        or semantic_authority.authority_snapshot_sha256
        != details.authority_snapshot_sha256
    ):
        raise AuxiliaryReplanningError(
            "semantic review authority is not bound to the current revision"
        )

    completion = planning_store.get_auxiliary_initial_planning_completion(
        session_id=details.session_id,
        insession_task_id=details.task_id,
    )
    if completion is None:
        # 直接创建的遗留/测试图没有初始规划回执。
        # 保留其现有行为；初始 Architect 合同仍然失败。
        # 如果语义投影无法认证当前别名，则延迟关闭.
        return semantic_authority

    binding = completion.binding
    if (
        binding.session_id != details.session_id
        or binding.task_id != details.task_id
        or binding.auxiliary_graph_id != details.auxiliary_graph_id
        or binding.goal_id != details.goal_id
    ):
        raise AuxiliaryReplanningError(
            "initial-planning completion crossed graph ownership"
        )
    initial_authority = binding.architect_request.prompt_payload.authority
    if (
        details.auxiliary_graph_revision
        == completion.committed_auxiliary_graph_revision
    ):
        if (
            completion.committed_authority_snapshot_id
            != details.authority_snapshot_id
            or completion.committed_authority_snapshot_sha256
            != details.authority_snapshot_sha256
        ):
            raise AuxiliaryReplanningError(
                "bootstrap planning completion differs from current authority"
            )
    elif (
        details.auxiliary_graph_revision
        < completion.committed_auxiliary_graph_revision
    ):
        raise AuxiliaryReplanningError(
            "initial-planning completion is newer than the current revision"
        )

    anchors = {item.projection_alias: item for item in snapshot.anchors}
    initial_cards = {item.alias: item for item in initial_authority.cards}
    if (
        len(anchors) != len(snapshot.anchors)
        or len(initial_cards) != len(initial_authority.cards)
        or set(anchors) != set(initial_cards)
    ):
        raise AuxiliaryReplanningError(
            "initial-planning cards differ from current authority aliases"
        )
    if any(
        anchors[alias].authority_class is not card.authority_class
        or anchors[alias].projection_sha256 != card.projection_sha256
        for alias, card in initial_cards.items()
    ):
        raise AuxiliaryReplanningError(
            "initial-planning cards differ from current authority bindings"
        )

    merged = dict(initial_cards)
    for card in semantic_authority.cards:
        existing = merged.get(card.alias)
        if existing is not None and existing != card:
            raise AuxiliaryReplanningError(
                "semantic review authority collides with planning authority"
            )
        merged[card.alias] = card
    return PlanningAuthorityProjection.create(
        authority_snapshot_id=details.authority_snapshot_id,
        authority_snapshot_sha256=details.authority_snapshot_sha256,
        cards=tuple(sorted(merged.values(), key=lambda item: item.alias)),
    )


def _authority_context_for_revision(
    details: StoredAuxiliaryGraphDetails,
) -> dict[str, object]:
    snapshot = details.authority_snapshot
    anchors = [
        anchor.model_dump(mode="json", exclude={"authority_snapshot_id"})
        for anchor in snapshot.anchors
        if anchor.projection_alias != "task_creation_source"
    ]
    return {"anchors": anchors}


def _consume_trigger(
    *,
    request: AuxiliaryReplanningRequest,
    trigger: AuxiliaryReplanTriggerReceipt,
    details: StoredAuxiliaryGraphDetails,
) -> planning_store.AuxiliaryReplanTriggerApplicationMutationResult:
    apply_id = _stable_id(
        "auxreplanconsume",
        {
            "trigger_receipt_sha256": trigger.receipt_sha256,
            "applied_auxiliary_graph_revision": (
                details.auxiliary_graph_revision
            ),
            "applied_structure_sha256": details.structure_sha256,
        },
    )
    return planning_store.consume_auxiliary_replan_trigger(
        command=planning_store.ConsumeAuxiliaryReplanTriggerCommand(
            apply_id=apply_id,
            session_id=request.session_id,
            consumed_turn_id=request.turn_id,
            task_id=request.task_id,
            trigger_id=trigger.trigger_id,
            expected_goal_id=trigger.goal_id,
            expected_applied_auxiliary_graph_revision=(
                trigger.source_auxiliary_graph_revision + 1
            ),
            expected_applied_structure_sha256=details.structure_sha256,
        )
    )


def _require_current_details(
    session_id: str,
    task_id: str,
) -> StoredAuxiliaryGraphDetails:
    details = auxiliary_graph_store.get_auxiliary_graph_for_task(
        session_id=session_id,
        insession_task_id=task_id,
    )
    if details is None:
        raise AuxiliaryReplanningError(
            "semantic replan requires current AuxiliaryGraph authority"
        )
    return details


def _stable_id(prefix: str, value: object) -> str:
    return f"{prefix}_{_stable_digest(value)[:40]}"


def _stable_digest(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_identifier(name: str, value: str) -> None:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or len(value) > 200
    ):
        raise ValueError(f"{name} must be a canonical 1..200 character identity")


__all__ = [
    "AuxiliaryReplanningError",
    "AuxiliaryReplanningRequest",
    "AuxiliaryReplanningResult",
    "AuxiliaryReplanningIds",
    "AuxiliaryReplanningStatus",
    "derive_auxiliary_replanning_ids",
    "run_auxiliary_replanning",
]
