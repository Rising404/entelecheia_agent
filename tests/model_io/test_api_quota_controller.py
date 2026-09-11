from __future__ import annotations

from pathlib import Path
import sqlite3
from types import SimpleNamespace

import pytest

from personagraph.model_io import endpoint_profiles as model_profiles
from personagraph.model_io import api_quota_controller
from personagraph.model_io.api_quota_controller import (
    API_QUOTA_DATABASE_PATH_ENVIRONMENT_VARIABLE,
    MODEL_API_QUOTA_ENVIRONMENT_VARIABLE,
    ApiQuotaAdmissionError,
    actual_model_tokens,
    model_profile_quota_environment_value,
    model_profile_quota_from_environment,
    prepare_api_quota_request,
    resolve_api_quota_database_path,
)
from personagraph.model_io.api_quota_queue import (
    ModelApiQuotaQueue,
    QuotaQueueError,
)
from personagraph.model_io.gateway import ModelGatewayError, ModelResult, PreparedModelCall


def _quota(**overrides: object) -> model_profiles.ModelProfileQuota:
    values: dict[str, object] = {
        "requests_per_minute": 10,
        "tokens_per_minute": 100_000,
        "tokens_per_week": 1_000_000_000,
        "max_in_flight": 2,
        "quota_group": None,
    }
    values.update(overrides)
    return model_profiles.ModelProfileQuota(**values)  # type: ignore[arg-type]


def test_all_unset_limits_bypass_the_queue() -> None:
    prepared = prepare_api_quota_request(
        base_url="https://models.example.test/v1",
        credential="never-persist-this-key",
        quota=model_profiles.ModelProfileQuota(),
        estimated_input_tokens=20,
        output_token_limit=30,
        provider_timeout_seconds=5,
    )
    assert prepared is None


def test_quota_environment_projection_round_trips_without_credentials() -> None:
    quota = _quota(quota_group="shared-provider-account")

    encoded = model_profile_quota_environment_value(quota)
    restored = model_profile_quota_from_environment(
        {MODEL_API_QUOTA_ENVIRONMENT_VARIABLE: encoded}
    )

    assert restored == quota
    assert "credential" not in encoded.casefold()
    assert "api_key" not in encoded.casefold()


def test_explicit_quota_database_path_must_be_absolute(tmp_path: Path) -> None:
    expected = (tmp_path / "shared/model_api_quota.sqlite3").resolve()

    assert resolve_api_quota_database_path(
        {API_QUOTA_DATABASE_PATH_ENVIRONMENT_VARIABLE: str(expected)}
    ) == expected
    with pytest.raises(ValueError, match="absolute"):
        resolve_api_quota_database_path(
            {API_QUOTA_DATABASE_PATH_ENVIRONMENT_VARIABLE: "relative.sqlite3"}
        )


def test_prepared_quota_is_content_free_and_two_phase_usage_is_settled(
    tmp_path: Path,
    monkeypatch,
) -> None:
    database = tmp_path / "model_api_quota.sqlite3"
    monkeypatch.setattr(api_quota_controller, "API_QUOTA_DATABASE_PATH", database)
    api_quota_controller.clear_api_quota_queue_cache()
    key = "sk-plain-text-must-not-enter-quota-db"
    prepared = prepare_api_quota_request(
        base_url="https://models.example.test/v1",
        credential=key,
        quota=_quota(),
        estimated_input_tokens=20,
        output_token_limit=30,
        provider_timeout_seconds=5,
    )
    assert prepared is not None
    assert key not in repr(prepared)
    assert prepared.token_reservation == 50

    permit = prepared.acquire(
        logical_call_id="logical-call-1",
        wait_timeout_seconds=5,
    )
    before_dispatch = permit.queue.usage_snapshot(permit.scope_hash)
    assert before_dispatch.requests_last_minute == 0
    assert before_dispatch.admitted_not_dispatched == 1

    permit.mark_dispatched()
    during_dispatch = permit.queue.usage_snapshot(permit.scope_hash)
    assert during_dispatch.requests_last_minute == 1
    permit.settle(outcome="succeeded", actual_tokens=7)
    settled = permit.queue.usage_snapshot(permit.scope_hash)
    assert settled.tokens_last_minute == 7
    assert settled.in_flight == 0

    with sqlite3.connect(database) as conn:
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
    persisted = b"".join(
        path.read_bytes()
        for path in tmp_path.glob("model_api_quota.sqlite3*")
        if path.is_file()
    )
    assert tables
    assert key.encode("utf-8") not in persisted


def test_actual_usage_requires_both_provider_counts_and_includes_cache() -> None:
    known = ModelResult(
        reply="ok",
        provider="test",
        model="test",
        latency_ms=1,
        input_tokens=11,
        output_tokens=4,
        cache_read_tokens=3,
        cache_write_tokens=2,
    )
    incomplete = ModelResult(
        reply="ok",
        provider="test",
        model="test",
        latency_ms=1,
        input_tokens=11,
        output_tokens=None,
    )

    assert actual_model_tokens(known) == 20
    assert actual_model_tokens(incomplete) is None


def test_direct_prepared_gateway_dispatch_cannot_bypass_configured_quota(
    tmp_path: Path,
    monkeypatch,
) -> None:
    database = tmp_path / "direct.sqlite3"
    monkeypatch.setattr(api_quota_controller, "API_QUOTA_DATABASE_PATH", database)
    api_quota_controller.clear_api_quota_queue_cache()
    quota_request = prepare_api_quota_request(
        base_url="https://models.example.test/v1",
        credential="direct-key",
        quota=_quota(),
        estimated_input_tokens=10,
        output_token_limit=20,
        provider_timeout_seconds=5,
    )
    assert quota_request is not None
    provider_calls: list[str | None] = []

    def dispatch(model_call_id: str | None) -> ModelResult:
        provider_calls.append(model_call_id)
        return ModelResult(
            reply="ok",
            provider="test",
            model="test",
            latency_ms=1,
            input_tokens=3,
            output_tokens=2,
            model_call_id=model_call_id,
        )

    prepared = PreparedModelCall(_dispatch=dispatch, api_quota=quota_request)
    result = prepared.dispatch(model_call_id="direct-call-1")

    assert result.reply == "ok"
    assert provider_calls == ["direct-call-1"]
    snapshot = ModelApiQuotaQueue(database).usage_snapshot(
        quota_request.scope_hash
    )
    assert snapshot.requests_last_minute == 1
    assert snapshot.tokens_last_minute == 5


def test_provider_429_cools_the_whole_scope_before_retry(
    tmp_path: Path,
    monkeypatch,
) -> None:
    database = tmp_path / "rate-limit.sqlite3"
    monkeypatch.setattr(api_quota_controller, "API_QUOTA_DATABASE_PATH", database)
    api_quota_controller.clear_api_quota_queue_cache()
    quota_request = prepare_api_quota_request(
        base_url="https://models.example.test/v1",
        credential="rate-key",
        quota=_quota(),
        estimated_input_tokens=10,
        output_token_limit=20,
        provider_timeout_seconds=5,
    )
    assert quota_request is not None

    def rate_limited(_model_call_id: str | None) -> ModelResult:
        raise ModelGatewayError(
            "MODEL_CALL_FAILED",
            "rate limited",
            retryable=True,
            details={"status_code": 429, "retry_after_seconds": 15.0},
        )

    prepared = PreparedModelCall(_dispatch=rate_limited, api_quota=quota_request)
    with pytest.raises(ModelGatewayError) as captured:
        prepared.dispatch(model_call_id="rate-limited-call")

    assert captured.value.retryable is True
    snapshot = ModelApiQuotaQueue(database).usage_snapshot(
        quota_request.scope_hash
    )
    assert snapshot.requests_last_minute == 1
    assert snapshot.tokens_last_minute == 30  # 用量未知时保留预留额度
    assert snapshot.cooldown_until > 0


def test_queue_failure_after_enqueue_cancels_the_unsent_ticket(monkeypatch) -> None:
    cancelled: list[str] = []

    class BrokenQueue:
        def configure_scope(self, *_args: object) -> None:
            return None

        def enqueue(self, **_values: object) -> object:
            return SimpleNamespace(ticket_id="ticket-before-wait-error")

        def wait_for_ticket(self, *_args: object, **_kwargs: object) -> object:
            raise QuotaQueueError("injected wait failure")

        def cancel(self, ticket_id: str) -> bool:
            cancelled.append(ticket_id)
            return True

    monkeypatch.setattr(
        api_quota_controller,
        "_default_queue",
        lambda _path: BrokenQueue(),
    )
    prepared = prepare_api_quota_request(
        base_url="https://models.example.test/v1",
        credential="queue-error-key",
        quota=_quota(),
        estimated_input_tokens=10,
        output_token_limit=20,
        provider_timeout_seconds=5,
    )
    assert prepared is not None

    with pytest.raises(ApiQuotaAdmissionError):
        prepared.acquire(
            logical_call_id="queue-error-call",
            wait_timeout_seconds=5,
        )

    assert cancelled == ["ticket-before-wait-error"]


def test_quota_permit_cannot_dispatch_a_different_prepared_request(
    tmp_path: Path,
    monkeypatch,
) -> None:
    database = tmp_path / "permit-binding.sqlite3"
    monkeypatch.setattr(api_quota_controller, "API_QUOTA_DATABASE_PATH", database)
    api_quota_controller.clear_api_quota_queue_cache()
    first = prepare_api_quota_request(
        base_url="https://first.example.test/v1",
        credential="first-key",
        quota=_quota(),
        estimated_input_tokens=10,
        output_token_limit=20,
        provider_timeout_seconds=5,
    )
    second = prepare_api_quota_request(
        base_url="https://second.example.test/v1",
        credential="second-key",
        quota=_quota(),
        estimated_input_tokens=10,
        output_token_limit=20,
        provider_timeout_seconds=5,
    )
    assert first is not None and second is not None
    permit = first.acquire(
        logical_call_id="permit-owner",
        wait_timeout_seconds=5,
    )
    provider_calls: list[str | None] = []
    prepared = PreparedModelCall(
        _dispatch=lambda call_id: (
            provider_calls.append(call_id)
            or ModelResult(
                reply="wrong permit",
                provider="test",
                model="test",
                latency_ms=1,
                input_tokens=1,
                output_tokens=1,
                model_call_id=call_id,
            )
        ),
        api_quota=second,
    )

    with pytest.raises(TypeError, match="permit"):
        prepared.dispatch_with_api_quota(
            model_call_id="cross-bound-call",
            permit=permit,
        )

    assert provider_calls == []
    permit.abandon_before_dispatch(disposition="cancel")


def test_policy_change_after_admission_is_retryable_without_provider_dispatch(
    tmp_path: Path,
    monkeypatch,
) -> None:
    database = tmp_path / "dispatch-deferred.sqlite3"
    monkeypatch.setattr(api_quota_controller, "API_QUOTA_DATABASE_PATH", database)
    api_quota_controller.clear_api_quota_queue_cache()
    quota_request = prepare_api_quota_request(
        base_url="https://models.example.test/v1",
        credential="cooldown-key",
        quota=_quota(),
        estimated_input_tokens=10,
        output_token_limit=20,
        provider_timeout_seconds=5,
    )
    assert quota_request is not None
    permit = quota_request.acquire(
        logical_call_id="late-cooldown-call",
        wait_timeout_seconds=5,
    )
    permit.queue.apply_rate_limit_cooldown(
        quota_request.scope_hash,
        retry_after_seconds=15,
    )
    provider_calls: list[str | None] = []
    prepared = PreparedModelCall(
        _dispatch=lambda call_id: provider_calls.append(call_id),  # type: ignore[arg-type]
        api_quota=quota_request,
    )

    with pytest.raises(ModelGatewayError) as captured:
        prepared.dispatch_with_api_quota(
            model_call_id="late-cooldown-call",
            permit=permit,
        )

    assert captured.value.code == "MODEL_QUOTA_DISPATCH_DEFERRED"
    assert captured.value.retryable is True
    assert provider_calls == []
    ticket = permit.queue.get_ticket(permit.lease.ticket.ticket_id)
    assert ticket is not None
    assert ticket.status == "cancelled"
