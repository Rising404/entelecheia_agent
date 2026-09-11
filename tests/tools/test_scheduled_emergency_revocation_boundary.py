from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from sqlite3 import Row

from pytest import MonkeyPatch

from personagraph.tools.catalog.persistence import lifecycle
from personagraph.tools.catalog.persistence import (
    EmergencyRevocation,
    EmergencyRevocationSelector,
    ToolCatalogRepository,
)


_BEFORE_EFFECTIVE_AT = datetime(2026, 9, 3, 12, 0, tzinfo=timezone.utc)
_EFFECTIVE_AT = _BEFORE_EFFECTIVE_AT + timedelta(seconds=1)
_AFTER_EFFECTIVE_AT = _EFFECTIVE_AT + timedelta(seconds=1)


class _MutableClock:
    def __init__(self, instant: datetime) -> None:
        self.instant = instant

    def __call__(self) -> datetime:
        return self.instant


def _repository(path: Path, clock: _MutableClock) -> ToolCatalogRepository:
    repository = ToolCatalogRepository(path, clock=clock)
    repository.initialize()
    return repository


def _advance_clock_during_revocation_materialization(
    clock: _MutableClock,
    monkeypatch: MonkeyPatch,
) -> None:
    original = lifecycle._revocation_from_row

    def materialize_and_cross_boundary(row: Row) -> EmergencyRevocation:
        revocation = original(row)
        clock.instant = _AFTER_EFFECTIVE_AT
        return revocation

    monkeypatch.setattr(
        lifecycle,
        "_revocation_from_row",
        materialize_and_cross_boundary,
    )


def test_implicit_active_read_uses_clock_after_consistent_materialization(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    clock = _MutableClock(_BEFORE_EFFECTIVE_AT)
    repository = _repository(tmp_path / "catalog.sqlite", clock)
    issued = repository.issue_emergency_revocation(
        "scheduled-revoke",
        EmergencyRevocationSelector(tool_id="scheduled_tool"),
        issued_by="security-operator",
        reason="scheduled deny",
        effective_at=_EFFECTIVE_AT,
    )
    _advance_clock_during_revocation_materialization(clock, monkeypatch)

    assert repository.list_active_emergency_revocations() == (issued,)


def test_explicit_active_read_keeps_historical_instant_across_materialization(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    clock = _MutableClock(_BEFORE_EFFECTIVE_AT)
    repository = _repository(tmp_path / "catalog.sqlite", clock)
    repository.issue_emergency_revocation(
        "scheduled-revoke",
        EmergencyRevocationSelector(tool_id="scheduled_tool"),
        issued_by="security-operator",
        reason="scheduled deny",
        effective_at=_EFFECTIVE_AT,
    )
    _advance_clock_during_revocation_materialization(clock, monkeypatch)

    assert repository.list_active_emergency_revocations(at=_BEFORE_EFFECTIVE_AT) == ()
