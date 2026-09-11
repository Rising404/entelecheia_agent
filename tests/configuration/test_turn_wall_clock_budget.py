from __future__ import annotations

import math

import pytest

from personagraph.configuration.features import (
    DEFAULT_TURN_WALL_CLOCK_BUDGET_S,
    load_features,
    resolve_features,
)


def test_turn_wall_clock_budget_defaults_to_twenty_five_minutes() -> None:
    assert DEFAULT_TURN_WALL_CLOCK_BUDGET_S == 1500.0
    assert load_features(None)["turn_wall_clock_budget_s"] == 1500.0


@pytest.mark.parametrize("value", [True, 0, -1, math.inf, math.nan, "1248"])
def test_turn_wall_clock_budget_rejects_non_positive_or_non_finite_values(
    value: object,
) -> None:
    with pytest.raises(ValueError, match="turn_wall_clock_budget_s"):
        resolve_features({"turn_wall_clock_budget_s": value})


def test_turn_wall_clock_budget_accepts_a_positive_finite_override() -> None:
    assert resolve_features({"turn_wall_clock_budget_s": 90})[
        "turn_wall_clock_budget_s"
    ] == 90.0
