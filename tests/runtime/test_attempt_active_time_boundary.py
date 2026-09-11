"""主机拥有的尝试活跃时间计量的边界覆盖。"""

from __future__ import annotations

import pytest

from personagraph.l2.task_execution.attempts import active_time


def test_active_time_owner_memoizes_and_rejects_invalid_clock_values() -> None:
    readings = iter((10.0, 15.5, 999.0))
    meter = active_time.AttemptActiveTimeMeter.starting_now(lambda: next(readings))

    assert meter.freeze() == 5.5
    assert meter.freeze() == 5.5
    assert next(readings) == 999.0

    invalid = active_time.AttemptActiveTimeMeter(
        started_at_monotonic=10.0,
        monotonic_clock=lambda: float("nan"),
    )
    with pytest.raises(
        active_time.AttemptActiveTimeMeasurementError,
        match="Attempt settlement monotonic timestamp must be finite",
    ):
        invalid.freeze()
