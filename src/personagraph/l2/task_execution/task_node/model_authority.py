"""TaskNode WorkRun 的持久、提供商中立模型权威。

此处的契约将任意有效图或节点修订版本中的 TaskGraph 节点绑定到通用 Runtime 模型账本。
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from personagraph.model_io.tier_bindings import ModelTierBinding, ModelTier, resolve_tier
from personagraph.l2.work_run import (
    AttemptDecision,
    NodeVerificationResult,
)
from ....runtime.model_calls.contracts import DurableLogicalModelCallAuthority
from ....runtime.model_calls.contracts import (
    RuntimeModelLedgerStore,
    RuntimeModelLogicalRequest,
)
from ....model_io.output_repair_contracts import (
    RuntimeModelOutputRepairProtocol,
    RuntimeModelStructuredPrompt,
)
from ....runtime.model_calls.authority import (
    RuntimeLogicalModelCallAuthority,
)
from personagraph.l2.task_execution.task_node.model_binding_contracts import (
    TaskNodeBoundModelCall,
    _canonical_json,
    task_node_model_state_guard_sha256,
)
from personagraph.l2.task_execution.task_node.model_authority_contracts import (
    TaskNodeModelCallAuthorityFactory,
    TaskNodeModelCallPlan,
)


_OUTPUT_REPAIR_PROTOCOL = (
    RuntimeModelOutputRepairProtocol.FOUR_MESSAGE_WHOLE_RESPONSE_REGENERATION
)


class TaskNodeModelAuthorityFactoryError(ValueError):
    """TaskNode 绑定无法安全授权持久模型调用。"""

    code = "task_node_model_authority_factory_rejected"


def bind_task_node_model_call_authority(
    *,
    plan: TaskNodeModelCallPlan,
    binding: TaskNodeBoundModelCall,
) -> DurableLogicalModelCallAuthority:
    """依据精确绑定调用并验证注入的工厂。"""

    authority = plan.authority_factory(
        binding,
        rederive_state_guard_sha256=plan.rederive_state_guard_sha256,
    )
    semantic_call_id = getattr(authority, "semantic_call_id", None)
    for name in (
        "require_current_state",
        "reserve",
        "replay_succeeded_result",
        "begin_physical_attempt",
        "settle_physical_attempt",
        "typed_result_payload",
        "success_fingerprint",
        "failure_fingerprint",
        "terminal_state_error",
    ):
        if not callable(getattr(authority, name, None)):
            raise TypeError("TaskNode model authority factory returned an invalid port")
    if semantic_call_id != binding.logical_call_id:
        raise TypeError("TaskNode model authority crossed logical-call identity")
    if isinstance(authority, RuntimeLogicalModelCallAuthority):
        logical = authority.logical_request
        stable_mismatch = (
            logical.session_id != binding.session_id
            or logical.task_id != binding.task_id
            or logical.auxiliary_graph_id is not None
            or logical.goal_id is not None
            or logical.execution_subject_id != binding.execution_subject_id
            or logical.call_kind != binding.call_kind
            or logical.purpose != binding.purpose
            or logical.request_contract != binding.request_contract
            or logical.typed_result_contract != binding.typed_result_contract
            or logical.max_physical_attempts != binding.max_physical_attempts
        )
        exact_logical_binding = (
            logical.invocation_turn_id == binding.invocation_turn_id
            and logical.request_json == binding.request_json
            and logical.request_sha256 == binding.request_sha256
            and logical.state_guard_sha256 == binding.state_guard_sha256
        )
        dispatch_continuation = (
            authority.dispatch_binding_sha256 == binding.binding_sha256
            and authority.dispatch_state_guard_sha256
            == binding.state_guard_sha256
        )
        if stable_mismatch or not (
            exact_logical_binding or dispatch_continuation
        ):
            raise TypeError("Runtime authority differs from TaskNode binding")
    return authority


def create_task_node_work_run_model_call_authority(
    binding: TaskNodeBoundModelCall,
    *,
    rederive_state_guard_sha256: Callable[[], str],
    ledger_store: RuntimeModelLedgerStore,
    model_binding: ModelTierBinding | None = None,
) -> RuntimeLogicalModelCallAuthority:
    """将一个精确的普通 TaskNode 提示词绑定到通用 Runtime 账本。"""

    admitted = _admit_binding(binding)
    expected_tier = (
        ModelTier.NODE_VERIFICATION
        if admitted.call_kind == "node_verification"
        else ModelTier.ATTEMPT
    )
    selected_model_binding = model_binding or resolve_tier(expected_tier)
    if (
        not isinstance(selected_model_binding, ModelTierBinding)
        or selected_model_binding.tier is not expected_tier
    ):
        raise TaskNodeModelAuthorityFactoryError(
            "TaskNode model tier binding crossed the call-site role"
        )
    provider, model, endpoint_fingerprint = _configured_model_endpoint(
        selected_model_binding
    )
    try:
        proposed = RuntimeModelLogicalRequest.create(
            logical_call_id=admitted.logical_call_id,
            session_id=admitted.session_id,
            task_id=admitted.task_id,
            auxiliary_graph_id=None,
            goal_id=None,
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
        raise TaskNodeModelAuthorityFactoryError(
            "TaskNode logical model request failed contract validation"
        ) from exc
    if (
        proposed.request_json != admitted.request_json
        or proposed.request_sha256 != admitted.request_sha256
    ):
        raise TaskNodeModelAuthorityFactoryError(
            "TaskNode logical request changed its frozen request payload"
        )
    logical = _resolve_logical_request(
        proposed=proposed,
        binding=admitted,
        ledger_store=ledger_store,
    )

    def typed_replay_payload(_model_result: object, value: object) -> object:
        return _typed_replay_payload(admitted, value=value)

    try:
        return RuntimeLogicalModelCallAuthority(
            logical_request=logical,
            state_guard_sha256=rederive_state_guard_sha256,
            typed_replay_payload_builder=typed_replay_payload,
            dispatch_state_guard_sha256=admitted.state_guard_sha256,
            dispatch_binding_sha256=admitted.binding_sha256,
            model_binding=selected_model_binding,
            store=ledger_store,
        )
    except Exception as exc:
        raise TaskNodeModelAuthorityFactoryError(
            "Runtime TaskNode model authority could not be constructed"
        ) from exc


def task_node_durable_provider_prompt(
    *,
    durable: DurableLogicalModelCallAuthority | None,
    invocation_turn_id: str,
    system_prompt: str,
    user_content: str,
) -> tuple[str, str]:
    """当逻辑调用跨越 Turn 时，使用不可变的原始提示词。"""

    if not isinstance(durable, RuntimeLogicalModelCallAuthority):
        return system_prompt, user_content
    frozen_prompt = durable.logical_request.structured_prompt
    if frozen_prompt is None:
        raise TypeError(
            "Runtime TaskNode authority lost its exact Provider prompt"
        )
    return frozen_prompt.system_prompt, frozen_prompt.user_content


def _resolve_logical_request(
    *,
    proposed: RuntimeModelLogicalRequest,
    binding: TaskNodeBoundModelCall,
    ledger_store: RuntimeModelLedgerStore,
) -> RuntimeModelLogicalRequest:
    stored = _get_stored_runtime_logical_call(
        ledger_store=ledger_store,
        session_id=binding.session_id,
        logical_call_id=binding.logical_call_id,
    )
    if stored is None:
        return proposed
    existing = getattr(stored, "request", None)
    if not isinstance(existing, RuntimeModelLogicalRequest):
        raise TaskNodeModelAuthorityFactoryError(
            "stored TaskNode logical request has the wrong contract"
        )
    if existing == proposed:
        return existing
    if not _is_exact_dispatch_continuation(
        existing=existing,
        proposed=proposed,
        call_kind=binding.call_kind,
    ):
        raise TaskNodeModelAuthorityFactoryError(
            "TaskNode logical request changed beyond its dispatch lease"
        )
    return existing


def _is_exact_dispatch_continuation(
    *,
    existing: RuntimeModelLogicalRequest,
    proposed: RuntimeModelLogicalRequest,
    call_kind: str,
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
        old_authority = old_payload["authority"]
        current_authority = current_payload["authority"]
        if not isinstance(old_authority, dict) or not isinstance(
            current_authority, dict
        ):
            return False
        if int(current_authority["dispatch_work_run_revision"]) < int(
            old_authority["dispatch_work_run_revision"]
        ):
            return False
        if call_kind == "node_verification" and int(
            current_authority["verification_request_revision"]
        ) < int(old_authority["verification_request_revision"]):
            return False
        old_projection = _semantic_projection(
            old_payload,
            call_kind=call_kind,
        )
        current_projection = _semantic_projection(
            current_payload,
            call_kind=call_kind,
        )
        return old_projection == current_projection and not same_turn
    except (KeyError, TypeError, ValueError):
        return False


def _semantic_projection(
    payload: dict[str, Any],
    *,
    call_kind: str,
) -> dict[str, Any]:
    projected = json.loads(_canonical_json(payload))
    authority = projected["authority"]
    for name in (
        "invocation_turn_id",
        "dispatch_work_run_revision",
        "verification_request_revision",
        "state_guard_sha256",
    ):
        authority.pop(name, None)
    if call_kind == "attempt_decision":
        bindings = projected["user_content"]["bindings"]
        bindings.pop("turn_id", None)
        bindings.pop("work_run_revision", None)
    return projected


def _typed_replay_payload(
    binding: TaskNodeBoundModelCall,
    *,
    value: object,
) -> object:
    if binding.call_kind == "attempt_decision":
        if not isinstance(value, AttemptDecision):
            raise TaskNodeModelAuthorityFactoryError(
                "TaskNode Attempt replay requires AttemptDecision"
            )
        try:
            admitted = AttemptDecision.model_validate_json(
                value.model_dump_json()
            )
        except Exception as exc:
            raise TaskNodeModelAuthorityFactoryError(
                "TaskNode Attempt replay failed fresh validation"
            ) from exc
        return admitted.model_dump(mode="json")
    if not isinstance(value, NodeVerificationResult):
        raise TaskNodeModelAuthorityFactoryError(
            "TaskNode verification replay requires NodeVerificationResult"
        )
    try:
        admitted = NodeVerificationResult.model_validate_json(
            value.model_dump_json()
        )
    except Exception as exc:
        raise TaskNodeModelAuthorityFactoryError(
            "TaskNode verification replay failed fresh validation"
        ) from exc
    expected = (
        (admitted.verification_request_id, binding.verification_request_id),
        (
            admitted.verification_request_revision,
            binding.verification_request_revision,
        ),
        (admitted.work_run_id, binding.work_run_id),
        (admitted.locked_work_run_revision, binding.locked_work_run_revision),
        (admitted.submitted_attempt_id, binding.attempt_id),
        (admitted.subject, binding.subject),
    )
    if any(actual != frozen for actual, frozen in expected):
        raise TaskNodeModelAuthorityFactoryError(
            "TaskNode verification result crossed its frozen binding"
        )
    payload: dict[str, object] = {
        "acceptance_results": [
            item.model_dump(mode="json")
            for item in admitted.acceptance_results
        ]
    }
    return payload


def _admit_binding(binding: TaskNodeBoundModelCall) -> TaskNodeBoundModelCall:
    if not isinstance(binding, TaskNodeBoundModelCall):
        raise TaskNodeModelAuthorityFactoryError(
            "TaskNode model binding has the wrong contract"
        )
    try:
        return TaskNodeBoundModelCall.model_validate_json(
            binding.model_dump_json()
        )
    except Exception as exc:
        raise TaskNodeModelAuthorityFactoryError(
            "TaskNode model binding failed fresh validation"
        ) from exc


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


def _configured_model_endpoint(
    model_binding: ModelTierBinding,
) -> tuple[str, str, str]:
    # 端点配置应置于通用绑定模块导入范围之外。
    from ....model_io.endpoint_identity import (
        configured_structured_model_endpoint_identity,
    )

    try:
        identity = configured_structured_model_endpoint_identity(model_binding)
    except Exception as exc:
        raise TaskNodeModelAuthorityFactoryError(
            "configured structured model endpoint identity is unavailable"
        ) from exc
    return identity.provider, identity.model, identity.endpoint_fingerprint


__all__ = [
    'TaskNodeModelCallPlan',
    'TaskNodeBoundModelCall',
    "TaskNodeModelAuthorityFactoryError",
    "TaskNodeModelCallAuthorityFactory",
    "bind_task_node_model_call_authority",
    "create_task_node_work_run_model_call_authority",
    "task_node_durable_provider_prompt",
    "task_node_model_state_guard_sha256",
]
