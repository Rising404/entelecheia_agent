from typing import NotRequired, get_args, get_origin, get_type_hints

import pytest

from personagraph.configuration import features as config
from personagraph.configuration.feature_contracts import FeatureFlags

@pytest.mark.parametrize("key", ("misspelled_feature", "removed_feature"))
def test_unknown_feature_keys_fail_closed(key: str) -> None:
    with pytest.raises(ValueError, match=rf"unsupported feature key: {key}"):
        config.resolve_features({key: True})


def test_resolved_defaults_keep_active_defaults(monkeypatch) -> None:
    monkeypatch.delenv("PERSONAGRAPH_RUNTIME_CONFIG", raising=False)

    features = config.load_features(None)

    assert features["user_interaction_mode"] == "interactive"
    assert features["file_retrieval_write_enabled"] is True
    assert features["file_retrieval_read_enabled"] is True
    assert features["history_retrieval_write_enabled"] is True
    assert features["history_retrieval_read_enabled"] is True
    assert features["l1_retrieval_tools_enabled"] is True
    assert features["l2_aux_retrieval_tools_enabled"] is False
    assert features["l2_task_retrieval_tools_enabled"] is False


def test_user_interaction_mode_accepts_only_canonical_supported_modes():
    closed_world = config.resolve_features(
        {"user_interaction_mode": "  CLOSED_WORLD  "}
    )

    assert closed_world["user_interaction_mode"] == "closed_world"

    with pytest.raises(
        ValueError,
        match="user_interaction_mode must be 'interactive' or 'closed_world'",
    ):
        config.resolve_features({"user_interaction_mode": "unattended"})

    with pytest.raises(
        ValueError,
        match="user_interaction_mode must be 'interactive' or 'closed_world'",
    ):
        config.resolve_features({"user_interaction_mode": False})


def test_dual_corpus_read_and_lane_gates_require_their_write_read_parents():
    with pytest.raises(
        ValueError,
        match="file_retrieval_read_enabled requires file_retrieval_write_enabled",
    ):
        config.resolve_features({
            "file_retrieval_write_enabled": False,
            "file_retrieval_read_enabled": True,
        })

    with pytest.raises(
        ValueError,
        match=(
            "history_retrieval_read_enabled requires "
            "history_retrieval_write_enabled"
        ),
    ):
        config.resolve_features({
            "history_retrieval_write_enabled": False,
            "history_retrieval_read_enabled": True,
        })

    with pytest.raises(
        ValueError,
        match="l1_retrieval_tools_enabled requires at least one retrieval read gate",
    ):
        config.resolve_features({
            "file_retrieval_write_enabled": False,
            "file_retrieval_read_enabled": False,
            "history_retrieval_write_enabled": False,
            "history_retrieval_read_enabled": False,
            "l1_retrieval_tools_enabled": True,
        })

    session_retrieval = config.resolve_features({
        "history_retrieval_write_enabled": True,
        "history_retrieval_read_enabled": True,
        "l1_retrieval_tools_enabled": True,
    })
    assert session_retrieval["history_retrieval_read_enabled"] is True
    assert session_retrieval["l2_aux_retrieval_tools_enabled"] is False
    assert session_retrieval["l2_task_retrieval_tools_enabled"] is False

def test_execution_findings_gate_is_typed_default_on_and_can_roll_back():
    annotation = get_type_hints(FeatureFlags, include_extras=True)[
        "execution_findings_enabled"
    ]
    assert get_origin(annotation) is NotRequired
    assert get_args(annotation) == (bool,)
    assert config.resolve_features({})["execution_findings_enabled"] is True
    assert config.resolve_features(
        {"execution_findings_enabled": False}
    )["execution_findings_enabled"] is False

    with pytest.raises(
        ValueError,
        match="execution_findings_enabled must be a boolean",
    ):
        config.resolve_features({"execution_findings_enabled": "false"})


def test_runtime_config_environment_selects_one_consistent_default(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CONFIG_DIR", tmp_path)
    path = tmp_path / "closed-world.yaml"
    path.write_text(
        "features:\n  user_interaction_mode: closed_world\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("PERSONAGRAPH_RUNTIME_CONFIG", "closed-world.yaml")

    assert config.load_features(None)["user_interaction_mode"] == "closed_world"


def test_blank_runtime_config_environment_preserves_product_default(monkeypatch):
    monkeypatch.setenv("PERSONAGRAPH_RUNTIME_CONFIG", "   ")

    assert config.load_features(None)["user_interaction_mode"] == "interactive"
