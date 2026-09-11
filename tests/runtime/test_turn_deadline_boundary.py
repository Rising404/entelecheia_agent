"""冷态共享轮次截止时间契约的边界覆盖。"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from personagraph.runtime import turn_deadline as deadlines
from personagraph.runtime.model_calls import requests as model_requests


def test_deadline_preserves_start_remaining_expiry_and_frozen_behavior(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ticks = iter((100.0, 103.5, 105.0, 100.0))
    monkeypatch.setattr(deadlines, "monotonic", lambda: next(ticks))

    deadline = deadlines.TurnDeadline.starting_now(5.0)
    assert deadline.expires_at_monotonic == 105.0
    assert deadline.remaining_s() == 1.5
    assert deadline.expired()
    assert deadlines.TurnDeadline.starting_now(-1.0).expires_at_monotonic == 100.0
    with pytest.raises(FrozenInstanceError):
        deadline.expires_at_monotonic = 200.0  # type: ignore[misc]


def test_deadline_failure_has_one_owner_and_no_model_request_proxy() -> None:
    error = deadlines.TurnDeadlineExceeded()

    assert error.code == "TURN_DEADLINE_EXCEEDED"
    assert error.message == "The turn exceeded its wall-clock budget."
    assert error.retryable is False
    assert deadlines.TurnDeadlineExceeded.__module__ == (
        "personagraph.runtime.turn_deadline"
    )
    assert not hasattr(model_requests, "TurnDeadline")
    assert not hasattr(model_requests, "TurnDeadlineExceeded")
    assert not hasattr(model_requests, "MODEL_REQUEST_TIMEOUT_S")
