"""生产基于 AuxiliaryGraph 的持久化模型授权工厂。

仅允许包含完整且自认证的 Host 权威状态的绑定进入这里。Provider 凭据和原始端点 URL 从不进入逻辑请求：配置的结构网关解析一个非秘密的 身份，该身份绑定其摘要、协议、提供者、模型和控制配置文件。
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from typing import Any

from personagraph.model_io.tier_bindings import ModelTierBinding, ModelTier, resolve_tier
from personagraph.l2.auxiliary_graph import (
    AuxiliaryNodeExecutorKind,
    AuxiliaryGraphRevisionProposal,
    PlanningEpisodeBudgetDisposition,
    TaskGraphSemanticBaseSnapshot,
    TaskGraphSemanticVerificationRequest,
    TaskGraphSemanticVerificationResult,
    TaskGraphRevisionCandidate,
)
from personagraph.l2.task_graph.contracts import InSessionTaskGraphRevisionProposal
from personagraph.l2.work_run import (
    AttemptDecision,
    NodeVerificationResult,
    OutputWindowFormat,
    RequestUserInputAction,
    SubmitOutputWindowAction,
)
from ..planning.architect import (
    AUXILIARY_GRAPH_ARCHITECT_RESULT_CONTRACT,
    AuxiliaryGraphArchitectRequest,
    _AUXILIARY_GRAPH_ARCHITECT_SYSTEM_PROMPT,
    serialize_auxiliary_graph_architect_prompt,
)
from .model_binding_contracts import AuxiliaryBoundModelCall
from ..verification.model_binding_contracts import (
    AuxiliarySemanticReviewerModelCallBinding,
)
from personagraph.runtime.model_calls.contracts import (
    RuntimeModelLedgerStore,
    RuntimeModelLogicalRequest,
)
from personagraph.model_io.output_repair_contracts import (
    RuntimeModelOutputRepairProtocol,
    RuntimeModelStructuredPrompt,
)
from personagraph.runtime.model_calls.policy import MAX_MODEL_ATTEMPTS
from personagraph.runtime.model_calls.authority import (
    RuntimeLogicalModelCallAuthority,
)
from personagraph.l2.auxiliary_execution.verification.task_graph_semantic import (
    _TASK_GRAPH_SEMANTIC_VERIFICATION_SYSTEM_PROMPT,
    serialize_task_graph_semantic_verification_prompt,
)
from personagraph.model_io.endpoint_identity import (
    configured_structured_model_endpoint_identity,
)


_ARCHITECT_CALL_KIND = "auxiliary_graph_architect"
_ARCHITECT_PURPOSE = "runtime_auxiliary_graph_architect_v2"
_ARCHITECT_REQUEST_CONTRACT = "auxiliary-graph-architect-request-v1"
_SEMANTIC_CALL_KIND = "task_graph_semantic_verification"

_OUTPUT_REPAIR_PROTOCOL = (
    RuntimeModelOutputRepairProtocol.FOUR_MESSAGE_WHOLE_RESPONSE_REGENERATION
)


class AuxiliaryModelAuthorityFactoryError(ValueError):
    """一个类型化的 绑定不能安全地授权一个持久化模型调用。"""

    code = "auxiliary_v2_model_authority_factory_rejected"


def create_auxiliary_graph_architect_model_call_authority(
    request: AuxiliaryGraphArchitectRequest,
    *,
    invocation_turn_id: str,
    rederive_state_guard_sha256: Callable[[], str],
    ledger_store: RuntimeModelLedgerStore,
    model_binding: ModelTierBinding | None = None,
) -> RuntimeLogicalModelCallAuthority:
    """将一个密封的 Architect 请求绑定到提供者中立的模型账本上。

    ``request.binding_sha256`` 是一个冻结的守卫，因为它覆盖了精确的目标、权威状态、预算和当前修订提示。调用者提供的回调必须在每次物理尝试边界上从当前 Host 状态重构相同的绑定。
    """

    admitted = _admit_architect_request(request)
    selected_model_binding = _admit_model_binding(
        model_binding or resolve_tier(ModelTier.ARCHITECT),
        allowed_tiers=(ModelTier.ARCHITECT,),
    )
    provider, model, endpoint_fingerprint = _configured_model_endpoint(
        selected_model_binding
    )
    max_attempts = _architect_max_physical_attempts(admitted)
    try:
        proposed_logical = RuntimeModelLogicalRequest.create(
            logical_call_id=admitted.logical_call_id,
            session_id=admitted.goal.session_id,
            task_id=admitted.goal.task_id,
            auxiliary_graph_id=admitted.goal.auxiliary_graph_id,
            goal_id=admitted.goal.goal_id,
            execution_subject_id=None,
            invocation_turn_id=invocation_turn_id,
            call_kind=_ARCHITECT_CALL_KIND,
            purpose=_ARCHITECT_PURPOSE,
            provider=provider,
            model=model,
            endpoint_fingerprint=endpoint_fingerprint,
            request_contract=_ARCHITECT_REQUEST_CONTRACT,
            request_payload=admitted.model_dump(mode="json"),
            output_repair_protocol=_OUTPUT_REPAIR_PROTOCOL,
            structured_prompt=RuntimeModelStructuredPrompt.create(
                system_prompt=_AUXILIARY_GRAPH_ARCHITECT_SYSTEM_PROMPT,
                user_content=serialize_auxiliary_graph_architect_prompt(admitted),
            ),
            typed_result_contract=AUXILIARY_GRAPH_ARCHITECT_RESULT_CONTRACT,
            max_physical_attempts=max_attempts,
            state_guard_sha256=admitted.binding_sha256,
        )
    except Exception as exc:
        raise AuxiliaryModelAuthorityFactoryError(
            "Architect logical model request failed contract validation"
        ) from exc
    if json.loads(proposed_logical.request_json) != admitted.model_dump(mode="json"):
        raise AuxiliaryModelAuthorityFactoryError(
            "Architect logical request changed its sealed request payload"
        )
    logical = _resolve_architect_logical_request(
        proposed=proposed_logical,
        ledger_store=ledger_store,
    )
    return _authority(
        logical=logical,
        rederive_state_guard_sha256=rederive_state_guard_sha256,
        typed_replay_payload_builder=_architect_typed_replay_payload,
        ledger_store=ledger_store,
        model_binding=selected_model_binding,
    )


def create_auxiliary_semantic_reviewer_model_call_authority(
    binding: AuxiliarySemanticReviewerModelCallBinding,
    *,
    rederive_state_guard_sha256: Callable[[], str],
    ledger_store: RuntimeModelLedgerStore,
    model_binding: ModelTierBinding | None = None,
) -> RuntimeLogicalModelCallAuthority:
    """实现语义控制器的确切持久化权威工厂."""

    admitted = _admit_semantic_binding(binding)
    selected_model_binding = _admit_model_binding(
        model_binding or resolve_tier(ModelTier.FINAL_GATE),
        allowed_tiers=(ModelTier.FINAL_GATE,),
    )
    semantic_request = _semantic_request_payload(admitted)
    provider, model, endpoint_fingerprint = _configured_model_endpoint(
        selected_model_binding
    )
    try:
        proposed_logical = RuntimeModelLogicalRequest.create(
            logical_call_id=admitted.logical_call_id,
            session_id=admitted.session_id,
            task_id=admitted.task_id,
            auxiliary_graph_id=admitted.auxiliary_graph_id,
            goal_id=admitted.goal_id,
            execution_subject_id=None,
            invocation_turn_id=admitted.invocation_turn_id,
            call_kind=_SEMANTIC_CALL_KIND,
            purpose=admitted.purpose,
            provider=provider,
            model=model,
            endpoint_fingerprint=endpoint_fingerprint,
            request_contract=admitted.request_contract,
            request_payload=json.loads(admitted.request_json),
            output_repair_protocol=_OUTPUT_REPAIR_PROTOCOL,
            structured_prompt=RuntimeModelStructuredPrompt.create(
                system_prompt=_TASK_GRAPH_SEMANTIC_VERIFICATION_SYSTEM_PROMPT,
                user_content=serialize_task_graph_semantic_verification_prompt(
                    semantic_request
                ),
            ),
            typed_result_contract=admitted.typed_result_contract,
            max_physical_attempts=admitted.max_physical_attempts,
            state_guard_sha256=admitted.state_guard_sha256,
        )
    except Exception as exc:
        raise AuxiliaryModelAuthorityFactoryError(
            "semantic reviewer logical request failed contract validation"
        ) from exc
    if (
        proposed_logical.request_json != admitted.request_json
        or proposed_logical.request_sha256 != admitted.request_sha256
    ):
        raise AuxiliaryModelAuthorityFactoryError(
            "semantic reviewer logical request changed its frozen request payload"
        )

    def typed_replay_payload(_model_result: object, value: object) -> object:
        return _semantic_typed_replay_payload(
            admitted,
            semantic_request=semantic_request,
            value=value,
        )

    logical = _resolve_semantic_logical_request(
        proposed=proposed_logical,
        binding=admitted,
        ledger_store=ledger_store,
    )
    return _authority(
        logical=logical,
        rederive_state_guard_sha256=rederive_state_guard_sha256,
        typed_replay_payload_builder=typed_replay_payload,
        ledger_store=ledger_store,
        dispatch_state_guard_sha256=admitted.state_guard_sha256,
        dispatch_binding_sha256=_semantic_binding_sha256(admitted),
        model_binding=selected_model_binding,
    )


def create_auxiliary_work_run_model_call_authority(
    binding: AuxiliaryBoundModelCall,
    *,
    rederive_state_guard_sha256: Callable[[], str],
    ledger_store: RuntimeModelLedgerStore,
    model_binding: ModelTierBinding | None = None,
) -> RuntimeLogicalModelCallAuthority:
    """将一个确切的 Attempt/验证提示绑定到 Runtime 记账簿."""

    admitted = _admit_work_run_binding(binding)
    expected_tier = _work_run_model_tier(admitted)
    selected_model_binding = _admit_model_binding(
        model_binding or resolve_tier(expected_tier),
        allowed_tiers=(expected_tier,),
    )
    provider, model, endpoint_fingerprint = _configured_model_endpoint(
        selected_model_binding
    )
    try:
        proposed_logical = RuntimeModelLogicalRequest.create(
            logical_call_id=admitted.logical_call_id,
            session_id=admitted.session_id,
            task_id=admitted.task_id,
            auxiliary_graph_id=admitted.auxiliary_graph_id,
            goal_id=admitted.goal_id,
            execution_subject_id=admitted.execution_subject_id,
            invocation_turn_id=admitted.invocation_turn_id,
            call_kind=admitted.call_kind,
            purpose=admitted.purpose,
            provider=provider,
            model=model,
            endpoint_fingerprint=endpoint_fingerprint,
            request_contract=admitted.request_contract,
            request_payload=json.loads(admitted.request_json),
            output_repair_protocol=_OUTPUT_REPAIR_PROTOCOL,
            structured_prompt=RuntimeModelStructuredPrompt.create(
                system_prompt=admitted.system_prompt,
                user_content=admitted.user_content,
            ),
            typed_result_contract=admitted.typed_result_contract,
            max_physical_attempts=admitted.max_physical_attempts,
            state_guard_sha256=admitted.state_guard_sha256,
        )
    except Exception as exc:
        raise AuxiliaryModelAuthorityFactoryError(
            "WorkRun logical model request failed contract validation"
        ) from exc
    if (
        proposed_logical.request_json != admitted.request_json
        or proposed_logical.request_sha256 != admitted.request_sha256
    ):
        raise AuxiliaryModelAuthorityFactoryError(
            "WorkRun logical request changed its frozen request payload"
        )

    def typed_replay_payload(_model_result: object, value: object) -> object:
        return _work_run_typed_replay_payload(admitted, value=value)

    logical = _resolve_work_run_logical_request(
        proposed=proposed_logical,
        binding=admitted,
        ledger_store=ledger_store,
    )
    return _authority(
        logical=logical,
        rederive_state_guard_sha256=rederive_state_guard_sha256,
        typed_replay_payload_builder=typed_replay_payload,
        ledger_store=ledger_store,
        dispatch_state_guard_sha256=admitted.state_guard_sha256,
        dispatch_binding_sha256=admitted.binding_sha256,
        model_binding=selected_model_binding,
    )


def _authority(
    *,
    logical: RuntimeModelLogicalRequest,
    rederive_state_guard_sha256: Callable[[], str],
    typed_replay_payload_builder: Callable[[object, object], object],
    ledger_store: RuntimeModelLedgerStore,
    dispatch_state_guard_sha256: str | None = None,
    dispatch_binding_sha256: str | None = None,
    model_binding: ModelTierBinding | None = None,
) -> RuntimeLogicalModelCallAuthority:
    if not callable(rederive_state_guard_sha256):
        raise AuxiliaryModelAuthorityFactoryError(
            "state-guard rederivation must be callable"
        )
    values: dict[str, object] = {
        "logical_request": logical,
        "state_guard_sha256": rederive_state_guard_sha256,
        "typed_replay_payload_builder": typed_replay_payload_builder,
    }
    if dispatch_state_guard_sha256 is not None:
        values["dispatch_state_guard_sha256"] = dispatch_state_guard_sha256
    if dispatch_binding_sha256 is not None:
        values["dispatch_binding_sha256"] = dispatch_binding_sha256
    if model_binding is not None:
        values["model_binding"] = model_binding
    values["store"] = ledger_store
    try:
        return RuntimeLogicalModelCallAuthority(**values)  # type: ignore[arg-type]
    except Exception as exc:
        raise AuxiliaryModelAuthorityFactoryError(
            "Runtime logical model authority could not be constructed"
        ) from exc


def _resolve_architect_logical_request(
    *,
    proposed: RuntimeModelLogicalRequest,
    ledger_store: RuntimeModelLedgerStore,
) -> RuntimeModelLogicalRequest:
    """仅重用当前确切的 Architect 请求."""

    stored = _get_stored_runtime_logical_call(
        ledger_store=ledger_store,
        session_id=proposed.session_id,
        logical_call_id=proposed.logical_call_id,
    )
    if stored is None:
        return proposed
    existing = getattr(stored, "request", None)
    if not isinstance(existing, RuntimeModelLogicalRequest):
        raise AuxiliaryModelAuthorityFactoryError(
            "stored Architect logical request has the wrong contract"
        )
    if existing == proposed:
        return existing
    raise AuxiliaryModelAuthorityFactoryError(
        "Architect logical request changed after reservation"
    )


def _resolve_work_run_logical_request(
    *,
    proposed: RuntimeModelLogicalRequest,
    binding: AuxiliaryBoundModelCall,
    ledger_store: RuntimeModelLedgerStore,
) -> RuntimeModelLogicalRequest:
    """在一次确认的 Turn 手递过程中重用一个不可变的语义请求。

    一个 WorkRun 继续体仅改变分发租约事实：调用 Turn、可变的 WorkRun/请求修订围栏，以及重新推导出的新鲜状态守卫。这些事实授权下一个物理尝试，但必须不制造另一个在相同稳定逻辑 ID 下的语义请求。
    """

    stored = _get_stored_runtime_logical_call(
        ledger_store=ledger_store,
        session_id=binding.session_id,
        logical_call_id=binding.logical_call_id,
    )
    if stored is None:
        return proposed
    existing = getattr(stored, "request", None)
    if not isinstance(existing, RuntimeModelLogicalRequest):
        raise AuxiliaryModelAuthorityFactoryError(
            "stored WorkRun logical request has the wrong contract"
        )
    if existing == proposed:
        return existing
    if not _is_exact_work_run_dispatch_continuation(
        existing=existing,
        proposed=proposed,
        binding=binding,
    ):
        raise AuxiliaryModelAuthorityFactoryError(
            "WorkRun logical request changed beyond its dispatch lease"
        )
    return existing


def _get_stored_runtime_logical_call(
    *,
    ledger_store: RuntimeModelLedgerStore,
    session_id: str,
    logical_call_id: str,
) -> object | None:
    return ledger_store.get_runtime_model_logical_call(
        session_id=session_id,
        logical_call_id=logical_call_id,
    )


def _resolve_semantic_logical_request(
    *,
    proposed: RuntimeModelLogicalRequest,
    binding: AuxiliarySemanticReviewerModelCallBinding,
    ledger_store: RuntimeModelLedgerStore,
) -> RuntimeModelLogicalRequest:
    """在分发 Turn 变化时保持一个审阅请求不可变。"""

    stored = _get_stored_runtime_logical_call(
        ledger_store=ledger_store,
        session_id=binding.session_id,
        logical_call_id=binding.logical_call_id,
    )
    if stored is None:
        return proposed
    existing = getattr(stored, "request", None)
    if not isinstance(existing, RuntimeModelLogicalRequest):
        raise AuxiliaryModelAuthorityFactoryError(
            "stored semantic reviewer logical request has the wrong contract"
        )
    if existing == proposed:
        return existing
    stable_fields = (
        "logical_call_id",
        "session_id",
        "task_id",
        "auxiliary_graph_id",
        "goal_id",
        "execution_subject_id",
        "call_kind",
        "purpose",
        "provider",
        "model",
        "endpoint_fingerprint",
        "request_contract",
        "request_json",
        "request_sha256",
        "typed_result_contract",
        "max_physical_attempts",
    )
    if any(
        getattr(existing, name) != getattr(proposed, name)
        for name in stable_fields
    ) or (
        existing.invocation_turn_id == proposed.invocation_turn_id
        or existing.output_repair_protocol is not _OUTPUT_REPAIR_PROTOCOL
        or existing.structured_prompt is None
        or existing.structured_prompt != proposed.structured_prompt
    ):
        raise AuxiliaryModelAuthorityFactoryError(
            "semantic reviewer logical request changed beyond its dispatch lease"
        )
    return existing


def _semantic_binding_sha256(
    binding: AuxiliarySemanticReviewerModelCallBinding,
) -> str:
    return _sha256(binding.model_dump(mode="json"))


def _is_exact_work_run_dispatch_continuation(
    *,
    existing: RuntimeModelLogicalRequest,
    proposed: RuntimeModelLogicalRequest,
    binding: AuxiliaryBoundModelCall,
) -> bool:
    same_turn = existing.invocation_turn_id == proposed.invocation_turn_id
    if (
        existing.output_repair_protocol is not _OUTPUT_REPAIR_PROTOCOL
        or existing.structured_prompt is None
    ):
        return False
    stable_fields = (
        "logical_call_id",
        "session_id",
        "task_id",
        "auxiliary_graph_id",
        "goal_id",
        "execution_subject_id",
        "call_kind",
        "purpose",
        "provider",
        "model",
        "endpoint_fingerprint",
        "request_contract",
        "typed_result_contract",
        "max_physical_attempts",
    )
    if any(
        getattr(existing, name) != getattr(proposed, name)
        for name in stable_fields
    ):
        return False
    try:
        old_payload = json.loads(existing.request_json)
        current_payload = json.loads(proposed.request_json)
        if not isinstance(old_payload, dict) or not isinstance(current_payload, dict):
            return False
        old_authority = old_payload.get("authority")
        current_authority = current_payload.get("authority")
        if not isinstance(old_authority, dict) or not isinstance(
            current_authority, dict
        ):
            return False
        old_work_run_revision = int(old_authority["work_run_revision"])
        current_work_run_revision = int(current_authority["work_run_revision"])
        if current_work_run_revision < old_work_run_revision:
            return False
        if binding.call_kind == "node_verification":
            old_request_revision = int(
                old_authority["verification_request_revision"]
            )
            current_request_revision = int(
                current_authority["verification_request_revision"]
            )
            if current_request_revision < old_request_revision:
                return False
        old_projection = _work_run_semantic_request_projection(
            old_payload,
            call_kind=binding.call_kind,
        )
        current_projection = _work_run_semantic_request_projection(
            current_payload,
            call_kind=binding.call_kind,
        )
        return old_projection == current_projection and not same_turn
    except (KeyError, TypeError, ValueError):
        return False


def _work_run_semantic_request_projection(
    payload: dict[str, Any],
    *,
    call_kind: str,
) -> dict[str, Any]:
    """仅从一个精确请求中移除认证的分发租约字段。"""

    projected = json.loads(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    )
    authority = projected["authority"]
    for name in (
        "invocation_turn_id",
        "work_run_revision",
        "verification_request_revision",
        "state_guard_sha256",
    ):
        authority.pop(name, None)
    if call_kind == "attempt_decision":
        bindings = projected["user_content"]["bindings"]
        bindings.pop("turn_id", None)
        bindings.pop("work_run_revision", None)
    return projected


def _admit_architect_request(
    request: AuxiliaryGraphArchitectRequest,
) -> AuxiliaryGraphArchitectRequest:
    if not isinstance(request, AuxiliaryGraphArchitectRequest):
        raise AuxiliaryModelAuthorityFactoryError(
            "Architect request must use AuxiliaryGraphArchitectRequest"
        )
    try:
        return AuxiliaryGraphArchitectRequest.model_validate_json(
            request.model_dump_json()
        )
    except Exception as exc:
        raise AuxiliaryModelAuthorityFactoryError(
            "Architect request failed fresh binding validation"
        ) from exc


def _admit_semantic_binding(
    binding: AuxiliarySemanticReviewerModelCallBinding,
) -> AuxiliarySemanticReviewerModelCallBinding:
    if not isinstance(binding, AuxiliarySemanticReviewerModelCallBinding):
        raise AuxiliaryModelAuthorityFactoryError(
            "semantic reviewer binding has the wrong contract"
        )
    try:
        return AuxiliarySemanticReviewerModelCallBinding.model_validate_json(
            binding.model_dump_json()
        )
    except Exception as exc:
        raise AuxiliaryModelAuthorityFactoryError(
            "semantic reviewer binding failed fresh validation"
        ) from exc


def _admit_work_run_binding(
    binding: AuxiliaryBoundModelCall,
) -> AuxiliaryBoundModelCall:
    if not isinstance(binding, AuxiliaryBoundModelCall):
        raise AuxiliaryModelAuthorityFactoryError(
            "WorkRun binding has the wrong contract"
        )
    try:
        return AuxiliaryBoundModelCall.model_validate_json(
            binding.model_dump_json()
        )
    except Exception as exc:
        raise AuxiliaryModelAuthorityFactoryError(
            "WorkRun binding failed fresh validation"
        ) from exc


def _architect_max_physical_attempts(
    request: AuxiliaryGraphArchitectRequest,
) -> int:
    budget = request.prompt_payload.budget
    assessment = budget.assessment
    if (
        assessment.disposition
        is PlanningEpisodeBudgetDisposition.HARD_LIMIT_REACHED
    ):
        raise AuxiliaryModelAuthorityFactoryError(
            "Architect budget is already at a hard limit"
        )
    remaining = (
        budget.effective_profile.hard_physical_provider_tries
        - budget.usage.physical_provider_tries
    )
    if remaining < 1:
        raise AuxiliaryModelAuthorityFactoryError(
            "Architect has no physical Provider attempt authority"
        )
    return min(MAX_MODEL_ATTEMPTS, remaining)


def _configured_model_endpoint(
    model_binding: ModelTierBinding,
) -> tuple[str, str, str]:
    try:
        identity = configured_structured_model_endpoint_identity(model_binding)
    except Exception as exc:
        raise AuxiliaryModelAuthorityFactoryError(
            "configured structured model endpoint identity is unavailable"
        ) from exc
    return identity.provider, identity.model, identity.endpoint_fingerprint


def _work_run_model_tier(
    binding: AuxiliaryBoundModelCall,
) -> ModelTier:
    if binding.call_kind == "node_verification":
        return ModelTier.NODE_VERIFICATION
    if (
        binding.call_kind == "attempt_decision"
        and binding.executor_kind is AuxiliaryNodeExecutorKind.TERMINAL_PLANNER
    ):
        # 终端规划器是真正发出 TaskGraph 创建/修订提案的模型调用。
        # 因此，即使它运行在通用 WorkRun 机制内，面向产品的“建图”层
        # 仍负责该调用。
        return ModelTier.ARCHITECT
    return ModelTier.ATTEMPT


def _admit_model_binding(
    binding: ModelTierBinding,
    *,
    allowed_tiers: tuple[ModelTier, ...],
) -> ModelTierBinding:
    if not isinstance(binding, ModelTierBinding):
        raise AuxiliaryModelAuthorityFactoryError(
            "model tier binding has the wrong contract"
        )
    if binding.tier not in allowed_tiers:
        raise AuxiliaryModelAuthorityFactoryError(
            "model tier binding crossed the call-site role"
        )
    return binding


def _architect_typed_replay_payload(
    _model_result: object,
    value: object,
) -> object:
    if not isinstance(value, AuxiliaryGraphRevisionProposal):
        raise AuxiliaryModelAuthorityFactoryError(
            "Architect typed replay requires its exact revision proposal"
        )
    try:
        admitted = AuxiliaryGraphRevisionProposal.model_validate_json(
            value.model_dump_json()
        )
    except Exception as exc:
        raise AuxiliaryModelAuthorityFactoryError(
            "Architect typed replay proposal failed fresh validation"
        ) from exc
    return admitted.model_dump(mode="json")


def _semantic_request_payload(
    binding: AuxiliarySemanticReviewerModelCallBinding,
) -> TaskGraphSemanticVerificationRequest:
    try:
        envelope = json.loads(binding.request_json)
    except (TypeError, ValueError, RecursionError) as exc:
        raise AuxiliaryModelAuthorityFactoryError(
            "semantic reviewer binding request JSON is invalid"
        ) from exc
    if not isinstance(envelope, dict) or set(envelope) != {
        "schema_version",
        "verification_result_id",
        "verification_request",
    }:
        raise AuxiliaryModelAuthorityFactoryError(
            "semantic reviewer binding request envelope is invalid"
        )
    if (
        envelope["schema_version"] != binding.request_contract
        or envelope["verification_result_id"] != binding.verification_result_id
        or not isinstance(envelope["verification_request"], dict)
    ):
        raise AuxiliaryModelAuthorityFactoryError(
            "semantic reviewer binding request envelope crossed identity"
        )
    raw_request = envelope["verification_request"]
    try:
        request = TaskGraphSemanticVerificationRequest.model_validate(
            raw_request
        )
    except Exception as exc:
        raise AuxiliaryModelAuthorityFactoryError(
            "semantic reviewer request failed fresh typed validation"
        ) from exc
    if request.model_dump(mode="json") != raw_request:
        raise AuxiliaryModelAuthorityFactoryError(
            "semantic reviewer request changed during typed validation"
        )
    expected = {
        "verification_request_id": binding.verification_request_id,
        "logical_call_id": binding.logical_call_id,
        "reviewer_ordinal": binding.reviewer_ordinal,
        "required_reviewer_count": binding.required_reviewer_count,
        "auxiliary_graph_revision": binding.auxiliary_graph_revision,
    }
    if any(getattr(request, key) != value for key, value in expected.items()):
        raise AuxiliaryModelAuthorityFactoryError(
            "semantic reviewer request differs from its factory binding"
        )
    expected_goal = {
        "session_id": binding.session_id,
        "task_id": binding.task_id,
        "auxiliary_graph_id": binding.auxiliary_graph_id,
        "goal_id": binding.goal_id,
    }
    if any(
        getattr(request.goal, key) != value
        for key, value in expected_goal.items()
    ):
        raise AuxiliaryModelAuthorityFactoryError(
            "semantic reviewer goal differs from its factory binding"
        )
    return request


def _semantic_typed_replay_payload(
    binding: AuxiliarySemanticReviewerModelCallBinding,
    *,
    semantic_request: TaskGraphSemanticVerificationRequest,
    value: object,
) -> object:
    if not isinstance(value, TaskGraphSemanticVerificationResult):
        raise AuxiliaryModelAuthorityFactoryError(
            "semantic typed replay requires its exact verification result"
        )
    try:
        admitted = TaskGraphSemanticVerificationResult.model_validate_json(
            value.model_dump_json()
        )
    except Exception as exc:
        raise AuxiliaryModelAuthorityFactoryError(
            "semantic verification result failed fresh validation"
        ) from exc
    expected = (
        (admitted.verification_result_id, binding.verification_result_id),
        (admitted.verification_request_id, binding.verification_request_id),
        (admitted.logical_call_id, binding.logical_call_id),
        (admitted.reviewer_ordinal, binding.reviewer_ordinal),
        (admitted.required_reviewer_count, binding.required_reviewer_count),
        (
            admitted.verification_profile_id,
            semantic_request.verification_profile_id,
        ),
        (
            admitted.request_binding_sha256,
            semantic_request.binding_sha256,
        ),
    )
    if any(actual != frozen for actual, frozen in expected):
        raise AuxiliaryModelAuthorityFactoryError(
            "semantic verification result crossed its frozen reviewer binding"
        )
    return {
        "items": [
            {
                **item.model_dump(mode="json"),
                "failure_scope": (
                    None
                    if item.failure_scope is None
                    else item.failure_scope.value
                ),
            }
            for item in admitted.items
        ]
    }


def _work_run_typed_replay_payload(
    binding: AuxiliaryBoundModelCall,
    *,
    value: object,
) -> object:
    if binding.call_kind == "attempt_decision":
        return _attempt_typed_replay_payload(binding, value=value)
    if binding.call_kind == "node_verification":
        return _verification_typed_replay_payload(binding, value=value)
    raise AuxiliaryModelAuthorityFactoryError(
        "WorkRun typed replay has an unsupported call kind"
    )


def _attempt_typed_replay_payload(
    binding: AuxiliaryBoundModelCall,
    *,
    value: object,
) -> object:
    if not isinstance(value, AttemptDecision):
        raise AuxiliaryModelAuthorityFactoryError(
            "Attempt typed replay requires its exact AttemptDecision"
        )
    try:
        admitted = AttemptDecision.model_validate_json(value.model_dump_json())
    except Exception as exc:
        raise AuxiliaryModelAuthorityFactoryError(
            "Attempt typed replay failed fresh validation"
        ) from exc
    if binding.executor_kind is not AuxiliaryNodeExecutorKind.TERMINAL_PLANNER:
        return admitted.model_dump(mode="json")
    if isinstance(admitted.action, RequestUserInputAction):
        return admitted.model_dump(mode="json")
    if not isinstance(admitted.action, SubmitOutputWindowAction):
        raise AuxiliaryModelAuthorityFactoryError(
            "terminal Attempt replay has a forbidden action"
        )
    if admitted.action.format is not OutputWindowFormat.PLAIN_TEXT:
        raise AuxiliaryModelAuthorityFactoryError(
            "terminal Attempt replay is not a TaskGraph proposal"
        )
    try:
        prompt_payload = json.loads(binding.user_content)
        if not isinstance(prompt_payload, dict):
            raise ValueError("terminal prompt payload is not an object")
        base_snapshot = _terminal_prompt_base_snapshot(prompt_payload)
        if base_snapshot is not None:
            candidate = TaskGraphRevisionCandidate.model_validate_json(
                admitted.action.content
            )
            if admitted.action.content != candidate.model_dump_json():
                raise ValueError("positive terminal material is not canonical")
            known_base_aliases = {
                item.node_alias for item in base_snapshot.nodes
            }
            selected_base_aliases = {
                item.base_node_alias
                for item in candidate.lineage
                if item.base_node_alias is not None
            }
            if not selected_base_aliases.issubset(known_base_aliases):
                raise ValueError(
                    "positive terminal lineage references an unknown base alias"
                )
            proposal = candidate.proposal
            lineage = candidate.lineage
        else:
            proposal = InSessionTaskGraphRevisionProposal.model_validate_json(
                admitted.action.content
            )
            if admitted.action.content != proposal.model_dump_json():
                raise ValueError("base-null terminal material is not canonical")
            lineage = ()
    except Exception as exc:
        raise AuxiliaryModelAuthorityFactoryError(
            "terminal Attempt replay proposal failed fresh validation"
        ) from exc
    action = {
        "kind": "submit_task_graph",
        "proposal": proposal.model_dump(mode="json"),
    }
    if base_snapshot is not None:
        action["lineage"] = [item.model_dump(mode="json") for item in lineage]
    return {
        "acceptance_updates": [
            item.model_dump(mode="json")
            for item in admitted.acceptance_updates
        ],
        "action": action,
    }


def _terminal_prompt_base_snapshot(
    prompt_payload: dict[str, object],
) -> TaskGraphSemanticBaseSnapshot | None:
    """验证固定在终端提示中的正基标记。"""

    proposal_contract = prompt_payload.get("task_graph_proposal_contract")
    expected_revision = (
        proposal_contract.get("expected_current_graph_revision")
        if isinstance(proposal_contract, dict)
        else None
    )
    if "task_graph_revision_base" not in prompt_payload:
        if expected_revision is not None:
            raise ValueError(
                "positive terminal prompt contract has no frozen base snapshot"
            )
        return None
    raw_snapshot = prompt_payload["task_graph_revision_base"]
    snapshot = TaskGraphSemanticBaseSnapshot.model_validate(raw_snapshot)
    if snapshot.model_dump(mode="json") != raw_snapshot:
        raise ValueError("positive terminal base snapshot is not exact")
    if (
        not isinstance(proposal_contract, dict)
        or not isinstance(expected_revision, int)
        or isinstance(expected_revision, bool)
        or expected_revision != snapshot.base_task_graph_revision
    ):
        raise ValueError(
            "positive terminal base snapshot crossed its proposal contract"
        )
    return snapshot


def _verification_typed_replay_payload(
    binding: AuxiliaryBoundModelCall,
    *,
    value: object,
) -> object:
    if not isinstance(value, NodeVerificationResult):
        raise AuxiliaryModelAuthorityFactoryError(
            "verification typed replay requires its exact Host result"
        )
    try:
        admitted = NodeVerificationResult.model_validate_json(
            value.model_dump_json()
        )
    except Exception as exc:
        raise AuxiliaryModelAuthorityFactoryError(
            "verification typed replay failed fresh validation"
        ) from exc
    expected = (
        (admitted.verification_request_id, binding.verification_request_id),
        (
            admitted.verification_request_revision,
            binding.verification_request_revision,
        ),
        (admitted.work_run_id, binding.work_run_id),
        (admitted.locked_work_run_revision, binding.work_run_revision),
        (admitted.submitted_attempt_id, binding.attempt_id),
        (admitted.subject, binding.subject),
    )
    if any(actual != frozen for actual, frozen in expected):
        raise AuxiliaryModelAuthorityFactoryError(
            "verification result crossed its frozen WorkRun binding"
        )
    return {
        "acceptance_results": [
            item.model_dump(mode="json")
            for item in admitted.acceptance_results
        ]
    }


def _sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "AuxiliaryModelAuthorityFactoryError",
    "create_auxiliary_graph_architect_model_call_authority",
    "create_auxiliary_semantic_reviewer_model_call_authority",
    "create_auxiliary_work_run_model_call_authority",
]
