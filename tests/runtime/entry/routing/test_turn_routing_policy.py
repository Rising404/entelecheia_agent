"""用户控制 L0/L1/L2 路由的 P1 契约。"""

from __future__ import annotations

import inspect
import json

import pytest

from personagraph.configuration.features import load_features, resolve_features
from personagraph.model_io.gateway import ModelResult
from personagraph.runtime import entry
from personagraph.runtime.entry.ingress import model as ingress_model
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
from tests.helpers.prepared_model_provider import (
    as_prepared_test_provider,
    repair_feedback_from_provider_kwargs,
)


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


def test_policy_keeps_l0_always_available_and_forbids_l1_l2_double_enable() -> None:
    assert TurnRoutingPolicy(
        l1_enabled=False,
        l2_enabled=False,
    ).allowed_processing_levels == ("L0",)
    assert default_turn_routing_policy().allowed_processing_levels == ("L0", "L1")

    with pytest.raises(ValueError, match="mutually exclusive"):
        TurnRoutingPolicy(l1_enabled=True, l2_enabled=True)
    with pytest.raises(ValueError, match="must be booleans"):
        TurnRoutingPolicy(l1_enabled=1)  # type: ignore[arg-type]


def test_host_capabilities_expose_all_implemented_processing_levels() -> None:
    assert available_processing_levels() == ("L0", "L1", "L2")


def test_stored_session_policy_is_the_only_session_routing_authority() -> None:
    direct = resolve_turn_routing_policy(
        request_policy=None,
        request_policy_provided=False,
        stored_session_policy_json=(
            '{"l1_enabled":false,"l2_enabled":false,"schema_version":1}'
        ),
    )
    task = resolve_turn_routing_policy(
        request_policy=None,
        request_policy_provided=False,
        stored_session_policy_json=(
            '{"l1_enabled":false,"l2_enabled":true,"schema_version":1}'
        ),
    )
    turn = resolve_turn_routing_policy(
        request_policy=None,
        request_policy_provided=False,
        stored_session_policy_json=(
            '{"l1_enabled":true,"l2_enabled":false,"schema_version":1}'
        ),
    )

    assert direct.snapshot.allowed_processing_levels == ("L0",)
    assert direct.snapshot.source == "session_default"
    assert task.snapshot.allowed_processing_levels == ("L0", "L2")
    assert task.snapshot.source == "session_default"
    assert turn.snapshot.allowed_processing_levels == ("L0", "L1")
    assert turn.snapshot.source == "session_default"


def test_policy_resolution_is_request_then_session_then_default() -> None:
    explicit = resolve_turn_routing_policy(
        request_policy={"l1_enabled": False, "l2_enabled": False},
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
            '{"l1_enabled":false,"l2_enabled":true,"schema_version":1}'
        ),
    )
    assert stored.snapshot.source == "session_default"
    assert stored.snapshot.allowed_processing_levels == ("L0", "L2")

    default = resolve_turn_routing_policy(
        request_policy=None,
        request_policy_provided=False,
        stored_session_policy_json=None,
    )
    assert default.snapshot.source == "default"
    assert default.snapshot.allowed_processing_levels == ("L0", "L1")


def test_l1_request_policy_authorizes_l1_without_a_second_gate() -> None:
    resolved = resolve_turn_routing_policy(
        request_policy={"l1_enabled": True, "l2_enabled": False},
        request_policy_provided=True,
        stored_session_policy_json=None,
    )

    assert resolved.snapshot.source == "request_override"
    assert resolved.snapshot.allowed_processing_levels == ("L0", "L1")


def test_snapshot_round_trip_checks_its_exact_hash() -> None:
    snapshot = freeze_turn_routing_policy(
        TurnRoutingPolicy(l1_enabled=True, l2_enabled=False),
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


def test_retired_legacy_routing_snapshot_fails_closed() -> None:
    with pytest.raises(TurnRoutingPolicyError):
        parse_turn_routing_policy_snapshot(
            {
                "schema_version": 1,
                "source": "legacy_compat",
                "policy": {
                    "schema_version": 1,
                    "l1_enabled": False,
                    "l2_enabled": True,
                },
                "allowed_processing_levels": ["L0", "L2"],
            }
        )


def test_l1_entry_repairs_one_disabled_level_then_executes_the_turn(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    session_id = _create_l1_session(tmp_path)
    snapshot = freeze_turn_routing_policy(
        TurnRoutingPolicy(l1_enabled=True, l2_enabled=False),
        source="request_override",
    )
    prompts: list[str] = []
    user_contents: list[str] = []
    provider_feedbacks: list[dict[str, object] | None] = []

    def classify(*args: object, **kwargs: object) -> ModelResult:
        prompts.append(str(args[0]))
        user_contents.append(str(args[1]))
        provider_feedbacks.append(repair_feedback_from_provider_kwargs(kwargs))
        level = "L2" if len(prompts) == 1 else "L1"
        return ModelResult(
            reply=json.dumps({"processing_level": level, "task_matches": []}),
            provider="test",
            model="test",
            latency_ms=1,
            model_call_id=str(kwargs["model_call_id"]),
        )

    monkeypatch.setattr(
        ingress_model,
        "complete_structured",
        as_prepared_test_provider(classify),
    )
    result = entry.run_entry_turn(
        user_input="在当前目录内完成一次有界检查",
        features={"context_guard_limit": 24_000},
        session_id=session_id,
        client_request_id="l1-p1-shell",
        routing_policy=snapshot,
        persist_routing_policy_as_session_default=True,
        store=session_store,
    )

    assert len(prompts) == 2
    assert prompts[1] == prompts[0]
    assert user_contents[1] == user_contents[0]
    assert json.loads(user_contents[1])["allowed_processing_levels"] == ["L0", "L1"]
    assert provider_feedbacks[0] is None
    repair_feedback = provider_feedbacks[1]
    assert repair_feedback is not None
    assert repair_feedback == {"current_issues": [{
        "paths": ["/processing_level"],
        "safe_explanation": "processing_level 必须从本轮 Host 允许的等级中选择。",
    }]}
    assert result.status == "completed"
    assert result.processing_level == "L1"
    assert result.error_code is None
    assert result.window_state == "post_commit_pending"
    run = session_store.get_l1_turn_run(
        session_id=session_id,
        turn_id=result.turn_id,
    )
    assert run is not None
    assert run["status"] == "completed"
    execution = session_store.get_l1_turn_execution(
        session_id=session_id,
        turn_id=result.turn_id,
    )
    assert execution is not None
    assert execution["state"]["stage"] == "completed"
    assert execution["state"]["attempts_started"] == 1
    assert session_store.list_insession_task_catalog(session_id) == ()
    stored = session_store.get_session_turn_routing_policy(session_id)
    assert stored is not None
    assert json.loads(str(stored["policy_json"])) == snapshot.policy.to_dict()
