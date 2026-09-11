"""Prepare one admitted model request without consuming attempt authority."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import nullcontext

from personagraph.model_io.gateway_core import (
    ModelGatewayError,
    model_http_timeout_ceiling,
)
from personagraph.model_io.output_repair_contracts import (
    RuntimeModelOutputRepairFeedback,
)
from personagraph.model_io.prepared_request_contracts import PreparedModelRequest
from personagraph.model_io.tier_bindings import (
    ModelTierBinding,
    model_tier_binding_scope,
)

from .contracts import DurableLogicalModelCallAuthority


def prepare_model_attempt_request(
    *,
    purpose: str,
    prepare_request: Callable[[], PreparedModelRequest],
    prepare_repair_request: Callable[
        [RuntimeModelOutputRepairFeedback, str], PreparedModelRequest
    ]
    | None,
    repair_feedback: RuntimeModelOutputRepairFeedback | None,
    rejected_response_text: str | None,
    remaining_s: float | None,
    durable_call: DurableLogicalModelCallAuthority | None,
    durable_repair_recovery: bool = False,
) -> PreparedModelRequest:
    """Prepare the exact request variant under frozen binding and timeout scopes.

    This gate deliberately performs no deadline reads, state-guard checks,
    reservations, event emission, quota acquisition, or Provider dispatch.  The
    request executor owns that ordering around this function.
    """

    frozen_binding = _frozen_model_binding(durable_call)
    binding_scope = (
        model_tier_binding_scope(frozen_binding)
        if frozen_binding is not None
        else nullcontext()
    )
    timeout_scope = (
        model_http_timeout_ceiling(remaining_s)
        if remaining_s is not None
        else nullcontext()
    )
    with binding_scope, timeout_scope:
        if repair_feedback is None:
            candidate = prepare_request()
        else:
            if prepare_repair_request is None or rejected_response_text is None:
                message = (
                    "Repair recovery requires its prepared callback and "
                    "rejected response."
                    if durable_repair_recovery
                    else "Repair preparation requires its rejected response."
                )
                raise ModelGatewayError(
                    "MODEL_CONFIGURATION_FAILURE",
                    message,
                    retryable=False,
                    details={
                        "purpose": purpose,
                        "reason": "missing_repair_material",
                    },
                )
            candidate = prepare_repair_request(
                repair_feedback,
                rejected_response_text,
            )

    if not isinstance(candidate, PreparedModelRequest):
        callback_name = (
            "prepare_repair_request"
            if durable_repair_recovery
            else "prepare_request"
        )
        raise TypeError(
            f"{callback_name} must return an object with "
            "dispatch(*, model_call_id)"
        )
    return candidate


def _frozen_model_binding(
    durable_call: DurableLogicalModelCallAuthority | None,
) -> ModelTierBinding | None:
    frozen_binding = (
        getattr(durable_call, "model_binding", None)
        if durable_call is not None
        else None
    )
    if frozen_binding is not None and not isinstance(
        frozen_binding,
        ModelTierBinding,
    ):
        assert durable_call is not None
        raise durable_call.terminal_state_error(
            "durable model binding has the wrong contract"
        )
    return frozen_binding


__all__ = ["prepare_model_attempt_request"]
