"""单次逻辑模型请求的冻结重试与 Repair 策略。"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import re

from personagraph.model_io.output_repair_contracts import (
    RuntimeModelOutputRepairFeedback,
    RuntimeModelOutputRepairProtocol,
)
from personagraph.model_io.prepared_request_contracts import PreparedModelRequest

from .contracts import DurableLogicalModelCallAuthority
from .policy import MAX_MODEL_ATTEMPTS


@dataclass(frozen=True, slots=True)
class ResolvedModelRequestPolicy:
    """进入物理 attempt 循环前冻结的 Host 请求策略。"""

    repair_enabled: bool
    repair_target_contract: str | None
    max_attempts: int
    recover_durable_repair_feedback: Callable[[], object] | None


def resolve_model_request_policy(
    *,
    max_attempts: int,
    prepare_request: Callable[[], PreparedModelRequest],
    prepare_repair_request: Callable[
        [RuntimeModelOutputRepairFeedback, str], PreparedModelRequest
    ]
    | None,
    repair_target_contract: str | None,
    durable_call: DurableLogicalModelCallAuthority | None,
    logical_model_call_id: str | None,
) -> ResolvedModelRequestPolicy:
    """验证调用合同，并解析持久 authority 冻结的有效策略。"""

    if not 1 <= max_attempts <= MAX_MODEL_ATTEMPTS:
        raise ValueError(f"max_attempts must be within 1..{MAX_MODEL_ATTEMPTS}")
    if not callable(prepare_request):
        raise TypeError("prepare_request must be callable")

    frozen_repair_enabled = _durable_output_repair_enabled(durable_call)
    repair_enabled = frozen_repair_enabled or prepare_repair_request is not None
    if repair_enabled and prepare_repair_request is None:
        raise ValueError("output repair requires prepare_repair_request")
    if (
        durable_call is not None
        and prepare_repair_request is not None
        and not frozen_repair_enabled
    ):
        raise durable_call.terminal_state_error(
            "durable output repair requires an explicitly frozen protocol"
        )

    effective_repair_target_contract = repair_target_contract
    if frozen_repair_enabled:
        logical_request = getattr(durable_call, "logical_request", None)
        candidate_contract = getattr(
            logical_request,
            "output_repair_target_contract",
            getattr(logical_request, "typed_result_contract", None),
        )
        if not isinstance(candidate_contract, str):
            assert durable_call is not None
            raise durable_call.terminal_state_error(
                "durable output repair has no frozen typed result contract"
            )
        if effective_repair_target_contract is None:
            effective_repair_target_contract = candidate_contract
        elif effective_repair_target_contract != candidate_contract:
            assert durable_call is not None
            raise durable_call.terminal_state_error(
                "repair target contract differs from the durable logical request"
            )
    if repair_enabled:
        if (
            not isinstance(effective_repair_target_contract, str)
            or not re.fullmatch(
                r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,199}",
                effective_repair_target_contract,
            )
        ):
            raise ValueError(
                "output repair requires a canonical repair_target_contract"
            )
    elif effective_repair_target_contract is not None:
        raise ValueError("repair_target_contract requires an output-repair callback")

    if logical_model_call_id is not None and (
        not isinstance(logical_model_call_id, str)
        or not logical_model_call_id.strip()
        or len(logical_model_call_id) > 200
    ):
        raise ValueError("logical_model_call_id must be 1..200 non-whitespace chars")
    if (
        durable_call is not None
        and logical_model_call_id is not None
        and logical_model_call_id != durable_call.semantic_call_id
    ):
        raise ValueError(
            "logical_model_call_id must match the durable semantic call authority"
        )

    effective_max_attempts = max_attempts
    if durable_call is not None:
        frozen_attempt_limit = getattr(
            durable_call,
            "frozen_max_physical_attempts",
            None,
        )
        if frozen_attempt_limit is not None:
            if (
                isinstance(frozen_attempt_limit, bool)
                or not isinstance(frozen_attempt_limit, int)
                or frozen_attempt_limit < 1
            ):
                raise durable_call.terminal_state_error(
                    "durable physical-attempt authority has an invalid bound"
                )
            effective_max_attempts = min(max_attempts, frozen_attempt_limit)

    recover_durable_repair_feedback: Callable[[], object] | None = None
    if durable_call is not None and repair_enabled:
        candidate = getattr(
            durable_call,
            "recover_output_repair_feedback",
            None,
        )
        if not callable(candidate):
            raise durable_call.terminal_state_error(
                "durable authority does not support persisted output repair"
            )
        recover_durable_repair_feedback = candidate

    return ResolvedModelRequestPolicy(
        repair_enabled=repair_enabled,
        repair_target_contract=effective_repair_target_contract,
        max_attempts=effective_max_attempts,
        recover_durable_repair_feedback=recover_durable_repair_feedback,
    )


def _durable_output_repair_enabled(
    durable_call: DurableLogicalModelCallAuthority | None,
) -> bool:
    """验证持久化请求只使用唯一现行 Repair 协议。"""

    if durable_call is None:
        return False
    logical_request = getattr(durable_call, "logical_request", None)
    frozen_protocol = getattr(logical_request, "output_repair_protocol", None)
    if frozen_protocol is None:
        return False
    if (
        frozen_protocol
        is RuntimeModelOutputRepairProtocol.FOUR_MESSAGE_WHOLE_RESPONSE_REGENERATION
    ):
        return True
    raise durable_call.terminal_state_error(
        "durable logical request declares an unsupported output-repair protocol"
    )


__all__ = ["ResolvedModelRequestPolicy", "resolve_model_request_policy"]
