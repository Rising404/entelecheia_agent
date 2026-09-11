import pytest

from personagraph.model_io import capabilities


def test_auto_does_not_assume_compatible_endpoint_supports_native(tmp_path, monkeypatch):
    monkeypatch.setattr(capabilities, "CAPABILITY_RECORD_PATH", tmp_path / "capabilities.json")
    selected, reason = capabilities.resolve_control_transport(
        "auto", provider="deepseek", base_url="https://api.deepseek.com/anthropic", model="deepseek-v4-pro"
    )
    assert selected == "prompt_json"
    assert reason == "native_probe_not_verified"


def test_auto_requires_matching_persisted_probe_evidence(tmp_path, monkeypatch):
    monkeypatch.setattr(capabilities, "CAPABILITY_RECORD_PATH", tmp_path / "capabilities.json")
    capabilities.record_native_probe(
        provider="anthropic-compatible",
        base_url="https://example.test/anthropic",
        model="model-a",
        passed=True,
        evidence={"observed_kind": "tool_call_batch", "provider_call_id": True},
    )
    assert capabilities.resolve_control_transport(
        "auto", provider="anthropic-compatible", base_url="https://example.test/anthropic", model="model-a"
    ) == ("native", "verified_native_probe")
    assert capabilities.resolve_control_transport(
        "auto", provider="anthropic-compatible", base_url="https://example.test/anthropic", model="model-b"
    )[0] == "prompt_json"


def test_probe_evidence_is_scoped_to_the_concrete_request_dialect(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        capabilities,
        "CAPABILITY_RECORD_PATH",
        tmp_path / "capabilities.json",
    )
    capabilities.record_native_probe(
        provider="openai-compatible",
        base_url="https://proxy.example/v1",
        model="model-a",
        request_dialect="deepseek",
        passed=True,
        evidence={"observed_kind": "tool_call_batch"},
    )

    assert capabilities.resolve_control_transport(
        "auto",
        provider="openai-compatible",
        base_url="https://proxy.example/v1",
        model="model-a",
        request_dialect="deepseek",
    ) == ("native", "verified_native_probe")
    assert capabilities.resolve_control_transport(
        "auto",
        provider="openai-compatible",
        base_url="https://proxy.example/v1",
        model="model-a",
        request_dialect="generic",
    )[0] == "prompt_json"


def test_failed_probe_does_not_enable_native(tmp_path, monkeypatch):
    monkeypatch.setattr(capabilities, "CAPABILITY_RECORD_PATH", tmp_path / "capabilities.json")
    capabilities.record_native_probe(
        provider="anthropic",
        base_url="https://api.anthropic.com",
        model="model-a",
        passed=False,
        evidence={"error_code": "MODEL_CALL_FAILED"},
    )
    assert capabilities.resolve_control_transport(
        "auto", provider="anthropic", base_url="https://api.anthropic.com", model="model-a"
    )[0] == "prompt_json"


def test_explicit_native_is_available_for_controlled_probe_runs():
    assert capabilities.resolve_control_transport(
        "native", provider="anthropic-compatible", base_url="https://example.test", model="model-a"
    ) == ("native", "explicit_native")


@pytest.mark.parametrize(
    ("provider", "base_url", "request_dialect"),
    (
        ("not-a-provider", "https://example.test/v1", "auto"),
        ("openai-compatible", "https://api.openai.com/v1", "responses-v9"),
        ("anthropic-compatible", "http://[", "generic"),
    ),
)
def test_summary_fails_safe_for_invalid_manual_configuration(
    tmp_path,
    monkeypatch,
    provider,
    base_url,
    request_dialect,
):
    monkeypatch.setattr(
        capabilities,
        "CAPABILITY_RECORD_PATH",
        tmp_path / "capabilities.json",
    )

    summary = capabilities.model_control_capability_summary(
        "native",
        provider=provider,
        base_url=base_url,
        model="model-a",
        request_dialect=request_dialect,
    )

    assert summary["effective"] == "prompt_json"
    assert summary["reason"] == "invalid_configuration"
    assert summary["native_probe_verified"] is False
    assert summary["native_probe_verified_at"] is None
