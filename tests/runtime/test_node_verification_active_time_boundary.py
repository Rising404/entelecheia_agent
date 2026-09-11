"""主机拥有的节点验证活跃时间计量的边界覆盖。"""

from __future__ import annotations

import pytest

from personagraph.l2.task_execution.verification import active_time
def test_active_time_owner_memoizes_first_positive_interval() -> None:
    readings = iter((10.0, 15.5, 999.0))
    meter = active_time.NodeVerificationActiveTimeMeter.starting_now(
        lambda: next(readings)
    )

    assert meter.freeze() == 5.5
    assert meter.freeze() == 5.5
    assert next(readings) == 999.0


@pytest.mark.parametrize("value", (True, float("nan"), float("inf"), float("-inf")))
def test_active_time_owner_rejects_invalid_start_values(value: object) -> None:
    with pytest.raises(
        active_time.NodeVerificationActiveTimeMeasurementError,
        match="verification start monotonic value must be finite",
    ):
        active_time.NodeVerificationActiveTimeMeter(
            started_at_monotonic=value,  # type: ignore[arg-type]
            monotonic_clock=lambda: 11.0,
        )


@pytest.mark.parametrize("value", (True, float("nan"), float("inf"), float("-inf")))
def test_active_time_owner_rejects_invalid_settlement_values(value: object) -> None:
    meter = active_time.NodeVerificationActiveTimeMeter(
        started_at_monotonic=10.0,
        monotonic_clock=lambda: value,  # type: ignore[return-value]
    )

    with pytest.raises(
        active_time.NodeVerificationActiveTimeMeasurementError,
        match="verification settlement monotonic value must be finite",
    ):
        meter.freeze()


@pytest.mark.parametrize("stopped_at", (10.0, 9.0))
def test_active_time_owner_rejects_zero_or_negative_interval(
    stopped_at: float,
) -> None:
    meter = active_time.NodeVerificationActiveTimeMeter(
        started_at_monotonic=10.0,
        monotonic_clock=lambda: stopped_at,
    )

    with pytest.raises(
        active_time.NodeVerificationActiveTimeMeasurementError,
        match="verification active-time delta must be finite and positive",
    ):
        meter.freeze()
