from __future__ import annotations

from personagraph.model_io.gateway_core import ModelResult
from personagraph.model_io.prepared_request_contracts import (
    PreparedModelRequest,
    QuotaPreparedModelRequest,
)


class _PreparedRequest:
    def dispatch(self, *, model_call_id: str) -> ModelResult:
        return ModelResult(
            reply="ok",
            provider="test",
            model="test",
            latency_ms=1,
            model_call_id=model_call_id,
        )


class _QuotaPreparedRequest(_PreparedRequest):
    def acquire_api_quota(
        self,
        *,
        model_call_id: str,
        wait_timeout_seconds: float,
    ) -> object:
        del model_call_id, wait_timeout_seconds
        return object()

    def abandon_api_quota(
        self,
        permit: object | None,
        *,
        disposition: str = "cancel",
    ) -> None:
        del permit, disposition

    def dispatch_with_api_quota(
        self,
        *,
        model_call_id: str,
        permit: object | None,
    ) -> ModelResult:
        del permit
        return self.dispatch(model_call_id=model_call_id)


def test_prepared_request_contracts_are_structural_and_provider_neutral() -> None:
    prepared = _PreparedRequest()
    quota_prepared = _QuotaPreparedRequest()

    assert isinstance(prepared, PreparedModelRequest)
    assert not isinstance(prepared, QuotaPreparedModelRequest)
    assert isinstance(quota_prepared, PreparedModelRequest)
    assert isinstance(quota_prepared, QuotaPreparedModelRequest)
    assert quota_prepared.dispatch(model_call_id="physical-1").model_call_id == (
        "physical-1"
    )
