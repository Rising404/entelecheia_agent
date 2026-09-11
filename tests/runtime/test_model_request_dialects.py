from __future__ import annotations

import pytest

from personagraph.model_io.dialects import (
    RequestDialect,
    build_request_controls,
    effective_reasoning_effort,
    reasoning_capability,
    resolve_request_dialect,
)


@pytest.mark.parametrize(
    ("provider", "base_url", "explicit", "expected"),
    [
        (
            "anthropic-compatible",
            "https://api.deepseek.com/anthropic",
            "auto",
            RequestDialect.DEEPSEEK_ANTHROPIC,
        ),
        (
            "openai-compatible",
            "https://api.deepseek.com/v1",
            "auto",
            RequestDialect.DEEPSEEK_OPENAI,
        ),
        (
            "openai-compatible",
            "https://api.openai.com/v1",
            "auto",
            RequestDialect.OPENAI_NATIVE,
        ),
        (
            "anthropic-compatible",
            "https://api.anthropic.com",
            "auto",
            RequestDialect.ANTHROPIC_NATIVE,
        ),
        (
            "openai-compatible",
            "https://proxy.example/v1",
            "auto",
            RequestDialect.GENERIC_OPENAI,
        ),
    ],
)
def test_auto_dialect_is_resolved_from_protocol_and_endpoint(
    provider, base_url, explicit, expected
):
    assert resolve_request_dialect(provider, base_url, explicit) is expected


def test_explicit_dialect_cannot_cross_protocol_families():
    with pytest.raises(ValueError, match="incompatible"):
        resolve_request_dialect(
            "anthropic-compatible",
            "https://proxy.example",
            "openai-native",
        )


def test_explicit_vendor_dialect_cannot_target_another_first_party_host():
    with pytest.raises(ValueError, match="conflicts"):
        resolve_request_dialect(
            "anthropic-compatible",
            "https://api.deepseek.com/anthropic",
            "anthropic",
        )


def test_openai_native_uses_native_token_and_reasoning_fields():
    controls = build_request_controls(
        provider="openai-compatible",
        dialect="openai-native",
        thinking_enabled=False,
        reasoning_effort="high",
        max_tokens=4096,
        temperature=0.0,
        json_mode=True,
    )

    assert controls.payload_fields == {
        "max_completion_tokens": 4096,
        "reasoning_effort": "high",
        "response_format": {"type": "json_object"},
    }
    assert controls.fingerprint["token_limit_field"] == "max_completion_tokens"
    assert controls.fingerprint["temperature"] == "omitted"


def test_openai_native_provider_default_effort_omits_temperature():
    controls = build_request_controls(
        provider="openai-compatible",
        dialect="openai-native",
        thinking_enabled=False,
        reasoning_effort=None,
        max_tokens=4096,
        temperature=0.0,
        json_mode=False,
    )

    assert controls.payload_fields == {"max_completion_tokens": 4096}
    assert controls.fingerprint["reasoning_control"] == "provider_default"
    assert controls.fingerprint["temperature"] == "omitted"


def test_deepseek_openai_keeps_max_tokens_and_maps_effort():
    controls = build_request_controls(
        provider="openai-compatible",
        dialect="deepseek-openai",
        thinking_enabled=True,
        reasoning_effort="xhigh",
        max_tokens=4096,
        temperature=0.0,
        json_mode=True,
    )

    assert controls.payload_fields["max_tokens"] == 4096
    assert "max_completion_tokens" not in controls.payload_fields
    assert controls.payload_fields["thinking"] == {"type": "enabled"}
    assert controls.payload_fields["reasoning_effort"] == "high"
    assert "temperature" not in controls.payload_fields
    assert effective_reasoning_effort("deepseek-openai", "xhigh") == "high"


def test_deepseek_anthropic_supports_toggle_and_graded_effort():
    controls = build_request_controls(
        provider="anthropic-compatible",
        dialect="deepseek-anthropic",
        thinking_enabled=True,
        reasoning_effort="max",
        max_tokens=4096,
        temperature=0.0,
        json_mode=True,
    )

    assert controls.payload_fields["max_tokens"] == 4096
    assert controls.payload_fields["thinking"] == {"type": "enabled"}
    assert controls.payload_fields["output_config"] == {"effort": "max"}
    assert controls.payload_fields["response_format"] == {"type": "json_object"}
    assert "temperature" not in controls.payload_fields


def test_deepseek_none_effort_does_not_override_an_enabled_thinking_switch():
    controls = build_request_controls(
        provider="anthropic-compatible",
        dialect="deepseek-anthropic",
        thinking_enabled=True,
        reasoning_effort="none",
        max_tokens=4096,
        temperature=0.0,
        json_mode=False,
    )

    assert controls.payload_fields["thinking"] == {"type": "enabled"}
    assert "output_config" not in controls.payload_fields
    assert controls.fingerprint["reasoning_effort"] is None


def test_anthropic_native_uses_adaptive_thinking_without_openai_json_field():
    controls = build_request_controls(
        provider="anthropic-compatible",
        dialect="anthropic-native",
        thinking_enabled=True,
        reasoning_effort="high",
        max_tokens=4096,
        temperature=0.0,
        json_mode=True,
    )

    assert controls.payload_fields == {
        "max_tokens": 4096,
        "thinking": {"type": "adaptive"},
        "output_config": {"effort": "high"},
    }
    assert controls.fingerprint["json_transport"] == "prompt_only"


def test_anthropic_native_effort_remains_active_with_thinking_disabled():
    controls = build_request_controls(
        provider="anthropic-compatible",
        dialect="anthropic-native",
        thinking_enabled=False,
        reasoning_effort="medium",
        max_tokens=4096,
        temperature=0.0,
        json_mode=False,
    )

    assert controls.payload_fields == {
        "max_tokens": 4096,
        "thinking": {"type": "disabled"},
        "output_config": {"effort": "medium"},
    }
    assert controls.fingerprint["temperature"] == "omitted"
    assert controls.fingerprint["reasoning_effort"] == "medium"
    assert reasoning_capability("anthropic-native")[0] == (
        "toggle_independent_effort"
    )


def test_anthropic_native_omits_an_incompatible_legacy_effort():
    controls = build_request_controls(
        provider="anthropic-compatible",
        dialect="anthropic-native",
        thinking_enabled=True,
        reasoning_effort="minimal",
        max_tokens=4096,
        temperature=0.0,
        json_mode=False,
    )

    assert controls.payload_fields["thinking"] == {"type": "adaptive"}
    assert "output_config" not in controls.payload_fields
    assert controls.fingerprint["reasoning_effort"] is None


def test_generic_compatible_endpoint_gets_no_vendor_reasoning_fields():
    controls = build_request_controls(
        provider="anthropic-compatible",
        dialect="generic-anthropic",
        thinking_enabled=True,
        reasoning_effort="max",
        max_tokens=1024,
        temperature=0.2,
        json_mode=True,
    )

    assert controls.payload_fields == {"max_tokens": 1024, "temperature": 0.2}
    assert controls.fingerprint["reasoning_control"] == "not_supported"
