"""Runtime ownership of a prepared model request's quota dispatch handoff."""

from __future__ import annotations

from dataclasses import dataclass

from personagraph.model_io.api_quota_controller import (
    DEFAULT_API_QUOTA_WAIT_TIMEOUT_SECONDS,
)
from personagraph.model_io.contracts import ModelResult
from personagraph.model_io.prepared_request_contracts import (
    PreparedModelRequest,
    QuotaPreparedModelRequest,
)


@dataclass(slots=True)
class QuotaDispatchHandle:
    """Hold an optional quota permit until Runtime hands dispatch to the gateway.

    ``permit`` may be ``None`` for an unlimited built-in request that still
    implements the optional two-phase quota protocol.  Once
    :meth:`dispatch` is invoked, all cancellation and reconciliation belongs
    to the provider gateway, even when that call raises before Provider I/O.
    """

    _request: QuotaPreparedModelRequest
    _permit: object | None
    _handoff_invoked: bool = False
    _abandoned: bool = False

    def abandon_if_not_handed_off(self) -> None:
        """Cancel a pre-dispatch reservation exactly while Runtime owns it."""

        if self._handoff_invoked or self._abandoned:
            return
        self._request.abandon_api_quota(
            self._permit,
            disposition="cancel",
        )
        self._abandoned = True

    def dispatch(self, *, model_call_id: str) -> ModelResult:
        """Hand quota ownership to the gateway and dispatch the request."""

        # Set before invoking the adapter: it may mark the permit dispatched
        # and then raise. Runtime must never cancel after that handoff begins.
        self._handoff_invoked = True
        return self._request.dispatch_with_api_quota(
            model_call_id=model_call_id,
            permit=self._permit,
        )


def acquire_quota_dispatch_handle(
    *,
    prepared_request: PreparedModelRequest,
    model_call_id: str,
    wait_timeout_seconds: float | None,
) -> QuotaDispatchHandle | None:
    """Acquire the optional quota protocol without consuming attempt authority.

    Deadline reads, durable state guards, physical-attempt creation, event
    emission, and Provider dispatch deliberately remain with the request loop.
    """

    if not isinstance(prepared_request, QuotaPreparedModelRequest):
        return None
    permit = prepared_request.acquire_api_quota(
        model_call_id=model_call_id,
        wait_timeout_seconds=(
            wait_timeout_seconds
            if wait_timeout_seconds is not None
            else DEFAULT_API_QUOTA_WAIT_TIMEOUT_SECONDS
        ),
    )
    return QuotaDispatchHandle(
        _request=prepared_request,
        _permit=permit,
    )


__all__ = ["QuotaDispatchHandle", "acquire_quota_dispatch_handle"]
