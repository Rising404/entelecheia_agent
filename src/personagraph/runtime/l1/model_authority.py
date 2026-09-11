"""持久化的模型调用组合用于一个准备好的 L1 Attempt。"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from personagraph.model_io.tier_bindings import ModelTierBinding
from .semantic_contracts import (
    L1SemanticVerificationResult,
    L1_SEMANTIC_RESULT_CONTRACT,
)
from ..model_calls.authority import RuntimeLogicalModelCallAuthority
from ..model_calls.contracts import RuntimeModelLogicalRequest
from ...model_io.output_repair_contracts import (
    RuntimeModelOutputRepairProtocol,
    RuntimeModelStructuredPrompt,
)
from personagraph.model_io.endpoint_identity import (
    configured_structured_model_endpoint_identity,
)
from ...output_protocol import L1AttemptDecisionProposal
from .identity import canonical_json
from .model_view.projection import project_attempt_view
from ...output_protocol.l1 import L1_ATTEMPT_PROTOCOL_VERSION
from .ports import L1StorePort


class L1ModelAuthorityError(ValueError):
    """一个准备好的 L1 Attempt 不能安全地绑定一个持久化的模型请求。"""


L1_SEMANTIC_FROZEN_CANDIDATE_CLAUSE = (
    "候选答案是冻结的审查对象；不得改写候选答案，只能输出完整的审查 JSON。"
)
L1_ATTEMPT_RESULT_CONTRACT = "l1-attempt-decision-proposal"


def l1_model_output_repair_policy(
    *,
    max_physical_attempts: int,
) -> dict[str, object]:
    """返回一个新 L1 逻辑调用的持久化修复声明。"""

    return {
        "kind": "bounded_whole_response_regeneration",
        "max_physical_attempts": max_physical_attempts,
        "field_merge": False,
        "feedback_contract": "runtime-model-output-repair-feedback-v2",
        "message_contract": "four-message-whole-response-regeneration-v1",
    }


def create_l1_attempt_model_call_authority(
    *,
    session_id: str,
    turn_id: str,
    l1_turn_run_id: str,
    attempt_id: str,
    logical_model_call_id: str,
    state_guard_hash: str,
    system_prompt: str,
    model_payload: dict[str, object],
    max_physical_attempts: int,
    store: L1StorePort,
    model_binding: ModelTierBinding,
    rederive_state_guard_hash: Callable[[], str],
) -> RuntimeLogicalModelCallAuthority:
    """将精确准备的 L1 Attempt 绑定到共享 Runtime 模型账本。"""

    if model_payload.get("schema_version") == L1_ATTEMPT_PROTOCOL_VERSION:
        ordinal = model_payload.get("attempt_ordinal")
        if (
            model_payload.get("attempt_id") != attempt_id
            or isinstance(ordinal, bool)
            or not isinstance(ordinal, int)
            or ordinal < 1
        ):
            raise L1ModelAuthorityError(
                "current L1 model view has no exact Attempt identity"
            )

    model_interface = {"provider_view": project_attempt_view(model_payload)}
    return _create_l1_model_call_authority(
        session_id=session_id,
        turn_id=turn_id,
        logical_model_call_id=logical_model_call_id,
        state_guard_hash=state_guard_hash,
        call_kind="l1_attempt",
        purpose="runtime_l1_attempt",
        request_contract="l1-attempt-model-protocol",
        request_payload={
            "schema_version": "l1-attempt-model-protocol",
            "l1_turn_run_id": l1_turn_run_id,
            "attempt_id": attempt_id,
            "system_prompt": system_prompt,
            "model_view": model_payload,
            **model_interface,
            "output_contract": "L1AttemptDecisionProposal",
            "repair_policy": l1_model_output_repair_policy(
                max_physical_attempts=max_physical_attempts,
            ),
        },
        typed_result_contract=L1_ATTEMPT_RESULT_CONTRACT,
        result_type=L1AttemptDecisionProposal,
        max_physical_attempts=max_physical_attempts,
        store=store,
        model_binding=model_binding,
        rederive_state_guard_hash=rederive_state_guard_hash,
    )


def create_l1_semantic_model_call_authority(
    *,
    session_id: str,
    turn_id: str,
    logical_model_call_id: str,
    state_guard_hash: str,
    request_payload: dict[str, object],
    max_physical_attempts: int,
    store: L1StorePort,
    model_binding: ModelTierBinding,
    rederive_state_guard_hash: Callable[[], str],
) -> RuntimeLogicalModelCallAuthority:
    """将一个候选特定的聚合评审绑定到共享账本。"""

    return _create_l1_model_call_authority(
        session_id=session_id,
        turn_id=turn_id,
        logical_model_call_id=logical_model_call_id,
        state_guard_hash=state_guard_hash,
        call_kind="l1_semantic_verifier",
        purpose="runtime_l1_semantic_verifier",
        request_contract="l1-semantic-verifier-model-protocol",
        request_payload=request_payload,
        typed_result_contract=L1_SEMANTIC_RESULT_CONTRACT,
        result_type=L1SemanticVerificationResult,
        max_physical_attempts=max_physical_attempts,
        store=store,
        model_binding=model_binding,
        rederive_state_guard_hash=rederive_state_guard_hash,
    )


def _create_l1_model_call_authority(
    *,
    session_id: str,
    turn_id: str,
    logical_model_call_id: str,
    state_guard_hash: str,
    call_kind: str,
    purpose: str,
    request_contract: str,
    request_payload: dict[str, object],
    typed_result_contract: str,
    result_type: type[Any],
    max_physical_attempts: int,
    store: L1StorePort,
    model_binding: ModelTierBinding,
    rederive_state_guard_hash: Callable[[], str],
) -> RuntimeLogicalModelCallAuthority:
    """创建一个不可变的 L1-拥有的逻辑模型请求。"""

    stored = store.get_runtime_model_logical_call(
        session_id=session_id,
        logical_call_id=logical_model_call_id,
    )
    existing = getattr(stored, "request", None)
    try:
        identity = configured_structured_model_endpoint_identity(model_binding)
        if existing is not None and not isinstance(
            existing,
            RuntimeModelLogicalRequest,
        ):
            raise L1ModelAuthorityError(
                "stored L1 logical request has the wrong contract"
            )
        if existing is not None and call_kind == "l1_semantic_verifier":
            # 冻结 DTO 也可能来自缓存或 model_construct；恢复前重验全部嵌套
            # hash，不能把“已经是该类型”当作持久请求完整性证明。
            existing = RuntimeModelLogicalRequest.model_validate_json(
                existing.model_dump_json()
            )
        repair_policy = request_payload.get("repair_policy")
        if not isinstance(repair_policy, dict):
            raise L1ModelAuthorityError("L1 logical request has no repair policy")
        if (
            repair_policy.get("feedback_contract")
            != "runtime-model-output-repair-feedback-v2"
            or repair_policy.get("message_contract")
            != "four-message-whole-response-regeneration-v1"
        ):
            raise L1ModelAuthorityError(
                "L1 logical request declares an unsupported repair policy"
            )
        system_prompt = request_payload.get("system_prompt")
        # 完整 request_payload 参与持久 hash；provider_view 才是实际发送的精简内容。
        # 业务投影与 Host 审计字段分离，原生引用身份不变。
        model_view = request_payload.get(
            "provider_view",
            request_payload.get(
                "model_view",
                request_payload.get("review_view"),
            ),
        )
        if not isinstance(system_prompt, str) or not isinstance(
            model_view,
            dict,
        ):
            raise L1ModelAuthorityError(
                "L1 logical request cannot freeze its exact model prompt"
            )
        structured_prompt = RuntimeModelStructuredPrompt.create(
            system_prompt=system_prompt,
            user_content=canonical_json(model_view),
        )
        proposed = RuntimeModelLogicalRequest.create(
            logical_call_id=logical_model_call_id,
            session_id=session_id,
            task_id=None,
            auxiliary_graph_id=None,
            goal_id=None,
            # 共享账本的 execution_subject_id 在当前数据库合同中是故意
            # 由 Task 拥有的。L1 没有 Task。
            # 这是它的 Turn FK，并且下面的冻结 Attempt/运行 ID 提供了权威状态。
            execution_subject_id=None,
            invocation_turn_id=turn_id,
            call_kind=call_kind,
            purpose=purpose,
            provider=identity.provider,
            model=identity.model,
            endpoint_fingerprint=identity.endpoint_fingerprint,
            request_contract=request_contract,
            request_payload=request_payload,
            output_repair_protocol=(
                RuntimeModelOutputRepairProtocol.FOUR_MESSAGE_WHOLE_RESPONSE_REGENERATION
            ),
            structured_prompt=structured_prompt,
            typed_result_contract=typed_result_contract,
            max_physical_attempts=max_physical_attempts,
            state_guard_sha256=state_guard_hash,
        )
    except Exception as exc:
        raise L1ModelAuthorityError(
            "L1 logical model request failed contract validation"
        ) from exc

    if existing is None:
        logical_request = proposed
    elif isinstance(existing, RuntimeModelLogicalRequest) and existing == proposed:
        logical_request = existing
    else:
        raise L1ModelAuthorityError(
            "L1 logical model request changed after durable reservation"
        )

    def typed_replay_payload(_model_result: object, value: object) -> object:
        if not isinstance(value, result_type):
            raise L1ModelAuthorityError(
                "L1 typed replay received another result contract"
            )
        try:
            admitted = result_type.model_validate_json(value.model_dump_json())
        except Exception as exc:
            raise L1ModelAuthorityError(
                "L1 typed replay failed fresh validation"
            ) from exc
        return admitted.model_dump(mode="json")

    return RuntimeLogicalModelCallAuthority(
        logical_request=logical_request,
        state_guard_sha256=rederive_state_guard_hash,
        typed_replay_payload_builder=typed_replay_payload,
        model_binding=model_binding,
        store=store,
    )


__all__ = [
    "L1_ATTEMPT_RESULT_CONTRACT",
    "L1_SEMANTIC_FROZEN_CANDIDATE_CLAUSE",
    "L1_SEMANTIC_RESULT_CONTRACT",
    "L1ModelAuthorityError",
    "create_l1_semantic_model_call_authority",
    "create_l1_attempt_model_call_authority",
    "l1_model_output_repair_policy",
]
