from __future__ import annotations

from pathlib import Path

import pytest

from personagraph.configuration import features as config
from personagraph.runtime.l1.tool_runtime import build_l1_tool_runtime
from personagraph.session import store as session_store
from personagraph.tools.composition.default_catalog import (
    build_production_default_catalog_seeds,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOCBENCH_RUNTIME_FEATURES = (
    PROJECT_ROOT / "evals/docbench/configs/runtime_closed_world.yaml"
)
WEB_TOOL_IDS = frozenset({"web_search", "web_fetch"})
EXPECTED_L1_TOOL_IDS = tuple(
    seed.definition.identity.tool_id
    for seed in build_production_default_catalog_seeds()
)


def _model_tool_ids(runtime) -> tuple[str, ...]:
    return tuple(item["tool_id"] for item in runtime.model_catalog())


def _assert_bound_enabled_catalog(runtime) -> None:
    assert tuple(definition.identity.tool_id for definition in runtime.definitions) == EXPECTED_L1_TOOL_IDS
    assert _model_tool_ids(runtime) == tuple(
        tool_id for tool_id in EXPECTED_L1_TOOL_IDS
        if tool_id in runtime.registrations_by_tool_id and tool_id not in runtime.disabled_tool_ids
    )


def test_l1_external_web_capability_feature_is_explicit_and_boolean() -> None:
    assert config.resolve_features({})["l1_external_web_tools_enabled"] is True
    assert (
        config.resolve_features({"l1_external_web_tools_enabled": False})[
            "l1_external_web_tools_enabled"
        ]
        is False
    )

    with pytest.raises(
        ValueError,
        match="l1_external_web_tools_enabled must be a boolean",
    ):
        config.resolve_features({"l1_external_web_tools_enabled": "false"})


def test_l1_external_web_capability_gate_hides_disabled_tools_and_denies_dispatch(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    session_id = session_store.create_session(
        "Entelecheia",
        working_dir=str(workspace),
    )

    with session_store.session_database_scope(session_id):
        ordinary = build_l1_tool_runtime(session_id)
        closed = build_l1_tool_runtime(
            session_id,
            execution_features={"l1_external_web_tools_enabled": False},
        )

    _assert_bound_enabled_catalog(ordinary)
    _assert_bound_enabled_catalog(closed)
    assert not WEB_TOOL_IDS.intersection(_model_tool_ids(closed))
    arguments_by_tool = {
        "web_search": {"query": "stable L1 tool surface"},
        "web_fetch": {"url": "https://example.com"},
    }
    for tool_id, arguments in arguments_by_tool.items():
        prepared = closed.prepare(
            tool_id=tool_id,
            arguments=arguments,
            remaining_tool_calls=1,
        )
        assert prepared.rejected_outcome is not None
        assert prepared.rejected_outcome.error is not None
        assert prepared.rejected_outcome.error.code == "tool_disabled_by_host"
        assert prepared.policy == {
            "disposition": "deny",
            "reason_codes": ["tool_disabled_by_host"],
        }


def test_docbench_runtime_features_hard_disable_l1_external_web(
    tmp_path: Path,
    bound_partitioned_session,
) -> None:
    features = config.load_features(str(DOCBENCH_RUNTIME_FEATURES))

    assert features["user_interaction_mode"] == "closed_world"
    assert features["l1_external_web_tools_enabled"] is False

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    session_id = bound_partitioned_session(working_dir=workspace)
    runtime = build_l1_tool_runtime(
        session_id,
        execution_features=features,
    )

    _assert_bound_enabled_catalog(runtime)
    assert not WEB_TOOL_IDS.intersection(_model_tool_ids(runtime))
    for tool_id in WEB_TOOL_IDS:
        assert tool_id in runtime.disabled_tool_ids
