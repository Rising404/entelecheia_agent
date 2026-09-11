"""Runtime-owned model-call package with lazy public exports.

Production modules should import the canonical owning submodule directly.  The
lazy package API remains convenient for external consumers without making a
contract-only import initialize provider gateways, request execution, or Session.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


__all__ = [
    "BACKOFF_BASE_S",
    "BACKOFF_MAX_S",
    "DurableLogicalModelCallAuthority",
    "DurableModelCallOutcome",
    "DurableModelCallReplay",
    "DurableModelCallRuntimeError",
    "DurableModelCallStateGuardRejected",
    "DurableModelCallTerminalState",
    "DurablePhysicalModelAttempt",
    "MAX_MODEL_ATTEMPTS",
    "MODEL_OUTPUT_LIMIT_FINISH_REASONS",
    "ModelRequestResult",
    "ResolvedModelRequestPolicy",
    "RuntimeLogicalModelCallAuthority",
    "RuntimeModelCallAuthorityError",
    "RuntimeModelCallRecoveryDisposition",
    "RuntimeModelCallRecoverySnapshot",
    "RuntimeModelCallWaitingExternal",
    "RuntimeModelLedgerStore",
    "RuntimeModelLogicalRequest",
    "RuntimeModelPhysicalAttemptRequest",
    "RuntimeModelPhysicalAttemptSettlement",
    "RuntimeModelPhysicalOutcome",
    "RuntimeModelTypedResult",
    "RuntimeModelUsage",
    "backoff_delay_s",
    "generate_session_summary",
    "request_model_with_retry",
    "resolve_model_request_policy",
    "runtime_model_dispatch_request_sha256",
]


_EXPORT_MODULE_BY_NAME = {
    **{
        name: ".contracts"
        for name in (
            "DurableLogicalModelCallAuthority",
            "DurableModelCallOutcome",
            "DurableModelCallReplay",
            "DurableModelCallRuntimeError",
            "DurableModelCallStateGuardRejected",
            "DurableModelCallTerminalState",
            "DurablePhysicalModelAttempt",
            "RuntimeModelLedgerStore",
            "RuntimeModelLogicalRequest",
            "RuntimeModelPhysicalAttemptRequest",
            "RuntimeModelPhysicalAttemptSettlement",
            "RuntimeModelPhysicalOutcome",
            "RuntimeModelTypedResult",
            "RuntimeModelUsage",
            "runtime_model_dispatch_request_sha256",
        )
    },
    **{
        name: ".authority"
        for name in (
            "RuntimeLogicalModelCallAuthority",
            "RuntimeModelCallAuthorityError",
            "RuntimeModelCallWaitingExternal",
        )
    },
    **{
        name: ".recovery"
        for name in (
            "RuntimeModelCallRecoveryDisposition",
            "RuntimeModelCallRecoverySnapshot",
        )
    },
    **{
        name: ".policy"
        for name in (
            "BACKOFF_BASE_S",
            "BACKOFF_MAX_S",
            "MAX_MODEL_ATTEMPTS",
            "MODEL_OUTPUT_LIMIT_FINISH_REASONS",
            "backoff_delay_s",
        )
    },
    "ModelRequestResult": ".requests",
    "request_model_with_retry": ".requests",
    "ResolvedModelRequestPolicy": ".request_policy",
    "resolve_model_request_policy": ".request_policy",
    "generate_session_summary": ".session_summary",
}


def __getattr__(name: str) -> Any:
    module_name = _EXPORT_MODULE_BY_NAME.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
