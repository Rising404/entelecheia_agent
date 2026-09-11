"""已配置结构化模型端点身份的边界覆盖。"""

from __future__ import annotations

import pytest

from personagraph.model_io import endpoint_identity
from personagraph.model_io.tier_bindings import (
    EndpointOrigin,
    ModelTierBinding,
    ModelTier,
    ReasoningEffort,
)


def test_endpoint_identity_owner_keeps_canonical_secret_safe_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PERSONAGRAPH_MODEL_PROVIDER", "openai-compatible")
    monkeypatch.setenv("PERSONAGRAPH_MODEL", "model-a")
    monkeypatch.setenv("PERSONAGRAPH_API_KEY", "sk-private-a")
    monkeypatch.setenv("PERSONAGRAPH_BASE_URL", "HTTPS://API.EXAMPLE:443/v1/")
    assembled = endpoint_identity.configured_structured_model_endpoint_identity()

    monkeypatch.setenv(
        "PERSONAGRAPH_BASE_URL",
        "https://api.example/v1/chat/completions",
    )
    explicit = endpoint_identity.configured_structured_model_endpoint_identity()

    assert assembled.endpoint_fingerprint == explicit.endpoint_fingerprint
    assert explicit.protocol == "openai-chat-completions-v1"
    assert "sk-private-a" not in repr(explicit)


def _openai_binding(
    *,
    thinking_enabled: bool,
    effort: ReasoningEffort | None = None,
) -> ModelTierBinding:
    return ModelTierBinding(
        tier=ModelTier.ATTEMPT,
        provider="openai-compatible",
        base_url="https://api.openai.com/v1",
        model="reasoning-model",
        api_key="sk-test",
        thinking_enabled=thinking_enabled,
        origin=EndpointOrigin.PROFILE,
        reasoning_effort=effort,
        request_dialect="openai-native",
    )


def test_openai_identity_tracks_wire_effort_not_the_ignored_legacy_checkbox() -> None:
    disabled = endpoint_identity.configured_structured_model_endpoint_identity(
        _openai_binding(thinking_enabled=False)
    )
    enabled = endpoint_identity.configured_structured_model_endpoint_identity(
        _openai_binding(thinking_enabled=True)
    )
    high = endpoint_identity.configured_structured_model_endpoint_identity(
        _openai_binding(
            thinking_enabled=False,
            effort=ReasoningEffort.HIGH,
        )
    )

    assert disabled.endpoint_fingerprint == enabled.endpoint_fingerprint
    assert disabled.control_profile_id == enabled.control_profile_id
    assert high.endpoint_fingerprint != disabled.endpoint_fingerprint


def test_blank_bound_openai_model_uses_the_gateway_default() -> None:
    binding = ModelTierBinding(
        tier=ModelTier.ATTEMPT,
        provider="openai-compatible",
        base_url="https://api.openai.com/v1",
        model="",
        api_key="sk-test",
        thinking_enabled=False,
        origin=EndpointOrigin.PROFILE,
        request_dialect="openai-native",
    )

    assert endpoint_identity.configured_structured_model_facts(binding) == (
        "openai-compatible",
        "gpt-4o-mini",
    )


def test_blank_unbound_openai_model_uses_the_gateway_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PERSONAGRAPH_MODEL_PROVIDER", "openai-compatible")
    monkeypatch.delenv("PERSONAGRAPH_MODEL", raising=False)

    assert endpoint_identity.configured_structured_model_facts() == (
        "openai-compatible",
        "gpt-4o-mini",
    )


def test_raw_anthropic_identity_requires_an_explicit_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PERSONAGRAPH_MODEL_PROVIDER", "anthropic")
    monkeypatch.setenv("PERSONAGRAPH_API_KEY", "sk-test")
    monkeypatch.delenv("PERSONAGRAPH_BASE_URL", raising=False)
    monkeypatch.delenv("PERSONAGRAPH_MODEL", raising=False)

    facts = endpoint_identity.configured_structured_model_facts()

    assert facts == ("anthropic-compatible", "")
    identity = endpoint_identity.configured_structured_model_endpoint_identity()
    assert identity.protocol == "anthropic-messages-2023-06-01"
