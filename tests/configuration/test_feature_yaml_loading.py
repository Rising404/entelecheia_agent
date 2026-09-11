from __future__ import annotations

import pytest
import yaml

from personagraph.configuration.features import load_features


def test_feature_yaml_requires_a_top_level_mapping(tmp_path) -> None:
    path = tmp_path / "features.yaml"
    path.write_text("- not\n- a\n- mapping\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Expected mapping"):
        load_features(str(path))


def test_feature_yaml_preserves_parser_errors(tmp_path) -> None:
    path = tmp_path / "features.yaml"
    path.write_text("features: [unterminated\n", encoding="utf-8")

    with pytest.raises(yaml.YAMLError):
        load_features(str(path))
