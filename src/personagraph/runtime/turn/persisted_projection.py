"""Decode lane-neutral facts from persisted Runtime Turn records.

This module owns only cold, deterministic projections shared by Entry, L1, and
L2.  It does not read storage, select a processing lane, or import an Entry
orchestrator.
"""

from __future__ import annotations

from .contracts import (
    EntryExecutionSnapshot,
    EntryTurnStatus,
    EntryWindowState,
    ProcessingLevel,
)


def require_persisted_mapping(
    value: dict[str, object],
    key: str,
) -> dict[str, object]:
    """Return a required mapping from a persisted receipt."""

    item = value.get(key)
    if not isinstance(item, dict):
        raise RuntimeError(f"missing persisted {key}")
    return item


def require_entry_turn_status(value: object) -> EntryTurnStatus:
    """Decode the closed public Runtime Turn-status vocabulary."""

    text = str(value or "")
    if text not in {"running", "completed", "incomplete"}:
        raise RuntimeError(f"unexpected runtime Turn status: {text}")
    return text  # type: ignore[return-value]


def optional_text(value: object) -> str | None:
    """Project an optional persisted scalar as text."""

    return str(value) if value is not None else None


def require_processing_level(value: object) -> ProcessingLevel | None:
    """Decode the closed public processing-level vocabulary."""

    text = optional_text(value)
    if text is None:
        return None
    if text not in {"L0", "L1", "L2"}:
        raise RuntimeError(f"unexpected processing level: {text}")
    return text  # type: ignore[return-value]


def require_entry_window_state(value: object) -> EntryWindowState:
    """Decode the closed public execution Window-state vocabulary."""

    text = str(value or "")
    if text not in {"empty", "active", "post_commit_pending", "interrupted"}:
        raise RuntimeError(f"unexpected execution Window state: {text}")
    return text  # type: ignore[return-value]


def execution_snapshot_from_persisted_turn(
    turn: dict[str, object],
) -> EntryExecutionSnapshot:
    """Authenticate and decode a Turn's immutable execution snapshot."""

    payload = turn.get("execution_snapshot_json")
    expected_hash = turn.get("execution_snapshot_sha256")
    if payload is None and expected_hash is None:
        raise RuntimeError("accepted Turn is missing its execution snapshot")
    if not isinstance(payload, str) or not isinstance(expected_hash, str):
        raise RuntimeError("accepted Turn execution snapshot is incomplete")
    try:
        return EntryExecutionSnapshot.from_json(
            payload,
            expected_sha256=expected_hash,
        )
    except ValueError as exc:
        raise RuntimeError(
            "accepted Turn execution snapshot authentication failed"
        ) from exc


__all__ = [
    "execution_snapshot_from_persisted_turn",
    "optional_text",
    "require_entry_turn_status",
    "require_entry_window_state",
    "require_persisted_mapping",
    "require_processing_level",
]
