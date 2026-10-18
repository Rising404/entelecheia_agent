"""用户控制 L0/L1 路由的 P1 契约。"""

from __future__ import annotations

import inspect

import pytest

from personagraph.configuration.features import load_features, resolve_features
from personagraph.runtime.l1.controller import run_l1_turn
from personagraph.runtime.entry.routing.policy import (
    TurnRoutingPolicyError,
    TurnRoutingPolicy,
    available_processing_levels,
    canonical_snapshot_json,
    default_turn_routing_policy,
    freeze_turn_routing_policy,
    parse_turn_routing_policy_snapshot,
    resolve_turn_routing_policy,
    snapshot_sha256,
)
from personagraph.session import store as session_store


def _create_l1_session(tmp_path) -> str:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    return session_store.create_session(
        "Entelecheia",
        working_dir=str(workspace),
    )


def test_l1_execution_limits_use_attempt_vocabulary() -> None:
    defaults = load_features(None)
    assert defaults["l1_max_attempts"] == 24
    assert defaults["l1_max_tool_calls_per_attempt"] == 8
    assert inspect.signature(run_l1_turn).parameters["max_attempts"].default == 24

    configured = resolve_features(
        {
            "l1_max_attempts": 16,
            "l1_max_tool_calls_per_attempt": 4,
        }
    )
    assert configured["l1_max_attempts"] == 16
    assert configured["l1_max_tool_calls_per_attempt"] == 4


def test_l1_semantic_verification_mode_defaults_always_and_is_closed(
    tmp_path,
) -> None:
    assert load_features(None)["l1_semantic_verification_mode"] == "always"
    config_path = tmp_path / "semantic-features.yaml"
    config_path.write_text(
        "features:\n  l1_semantic_verification_mode: sometimes\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="must be off, conditional, or always",
    ):
        load_features(str(config_path))


def test_policy_keeps_l0_always_available_and_validates_the_l1_switch() -> None:
    assert TurnRoutingPolicy(
        l1_enabled=False,
    ).allowed_processing_levels == ("L0",)
    assert default_turn_routing_policy().allowed_processing_levels == ("L0", "L1")

    with pytest.raises(ValueError, match="must be booleans"):
        TurnRoutingPolicy(l1_enabled=1)  # type: ignore[arg-type]


def test_host_capabilities_expose_all_implemented_processing_levels() -> None:
    assert available_processing_levels() == ("L0", "L1")


def test_stored_session_policy_is_the_only_session_routing_authority() -> None:
    direct = resolve_turn_routing_policy(
        request_policy=None,
        request_policy_provided=False,
        stored_session_policy_json=(
            '{"l1_enabled":false,"schema_version":2}'
        ),
    )
    turn = resolve_turn_routing_policy(
        request_policy=None,
        request_policy_provided=False,
        stored_session_policy_json=(
            '{"l1_enabled":true,"schema_version":2}'
        ),
    )

    assert direct.snapshot.allowed_processing_levels == ("L0",)
    assert direct.snapshot.source == "session_default"
    assert turn.snapshot.allowed_processing_levels == ("L0", "L1")
    assert turn.snapshot.source == "session_default"


def test_policy_resolution_is_request_then_session_then_default() -> None:
    explicit = resolve_turn_routing_policy(
        request_policy={"l1_enabled": False},
        request_policy_provided=True,
        stored_session_policy_json=None,
    )
    assert explicit.snapshot.source == "request_override"
    assert explicit.snapshot.allowed_processing_levels == ("L0",)
    assert explicit.persist_as_session_default is True

    stored = resolve_turn_routing_policy(
        request_policy=None,
        request_policy_provided=False,
        stored_session_policy_json=(
            '{"l1_enabled":true,"schema_version":2}'
        ),
    )
    assert stored.snapshot.source == "session_default"
    assert stored.snapshot.allowed_processing_levels == ("L0", "L1")

    default = resolve_turn_routing_policy(
        request_policy=None,
        request_policy_provided=False,
        stored_session_policy_json=None,
    )
    assert default.snapshot.source == "default"
    assert default.snapshot.allowed_processing_levels == ("L0", "L1")


def test_l1_request_policy_authorizes_l1_without_a_second_gate() -> None:
    resolved = resolve_turn_routing_policy(
        request_policy={"l1_enabled": True},
        request_policy_provided=True,
        stored_session_policy_json=None,
    )

    assert resolved.snapshot.source == "request_override"
    assert resolved.snapshot.allowed_processing_levels == ("L0", "L1")


def test_snapshot_round_trip_checks_its_exact_hash() -> None:
    snapshot = freeze_turn_routing_policy(
        TurnRoutingPolicy(l1_enabled=True),
        source="request_override",
    )

    assert (
        parse_turn_routing_policy_snapshot(
            canonical_snapshot_json(snapshot),
            expected_sha256=snapshot_sha256(snapshot),
        )
        == snapshot
    )
    with pytest.raises(TurnRoutingPolicyError):
        parse_turn_routing_policy_snapshot(
            canonical_snapshot_json(snapshot),
            expected_sha256="0" * 64,
        )
