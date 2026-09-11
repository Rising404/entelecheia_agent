from __future__ import annotations

import pytest

from personagraph.model_io.tier_bindings import (
    EndpointOrigin,
    ModelTierBinding,
    ModelTier,
)
from personagraph.runtime.l1 import execution_config


def _binding(*, thinking_enabled: bool, api_key: str) -> ModelTierBinding:
    return ModelTierBinding(
        tier=ModelTier.L1,
        provider="openai-compatible",
        base_url="https://models.example.test/v1",
        model="model-for-l1",
        api_key=api_key,
        thinking_enabled=thinking_enabled,
        origin=EndpointOrigin.PROFILE,
        profile_id="profile-l1",
        profile_name="L1 test profile",
    )


def test_l1_execution_config_is_canonical_and_never_persists_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binding = _binding(thinking_enabled=False, api_key="secret-never-persisted")
    monkeypatch.setattr(execution_config, "resolve_tier", lambda _tier: binding)

    frozen = execution_config.freeze_l1_execution_config(
        {
            "execution_findings_enabled": True,
            "l1_max_attempts": 12,
            "nested": {"enabled": False},
        }
    )
    loaded = execution_config.load_l1_execution_config(
        frozen.snapshot_json,
        frozen.snapshot_sha256,
    )

    assert loaded.snapshot == frozen.snapshot
    assert loaded.model_binding == binding
    assert "secret-never-persisted" not in frozen.snapshot_json
    assert loaded.snapshot.features["nested"] == {"enabled": False}


def test_l1_execution_config_fails_closed_when_endpoint_identity_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    accepted_binding = _binding(
        thinking_enabled=False,
        api_key="credential-before-restart",
    )
    monkeypatch.setattr(
        execution_config,
        "resolve_tier",
        lambda _tier: accepted_binding,
    )
    frozen = execution_config.freeze_l1_execution_config(
        {"execution_findings_enabled": True}
    )

    changed_binding = _binding(
        thinking_enabled=True,
        api_key="credential-after-restart",
    )
    monkeypatch.setattr(
        execution_config,
        "resolve_tier",
        lambda _tier: changed_binding,
    )

    with pytest.raises(
        execution_config.L1ExecutionConfigError,
        match="identity changed",
    ):
        execution_config.load_l1_execution_config(
            frozen.snapshot_json,
            frozen.snapshot_sha256,
        )


def test_l1_execution_config_rejects_tampered_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binding = _binding(thinking_enabled=False, api_key="credential")
    monkeypatch.setattr(execution_config, "resolve_tier", lambda _tier: binding)
    frozen = execution_config.freeze_l1_execution_config(
        {"execution_findings_enabled": True}
    )

    with pytest.raises(
        execution_config.L1ExecutionConfigError,
        match="hash changed",
    ):
        execution_config.load_l1_execution_config(
            frozen.snapshot_json.replace("true", "false"),
            frozen.snapshot_sha256,
        )
