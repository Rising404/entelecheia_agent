"""Persist an already-admitted Entry task-match proposal through L2 authority.

Entry owns classification, processing-level invariants, and Turn lifecycle effects.
This adapter owns only the idempotent L2 store command and its response-loss retry.
"""

from __future__ import annotations

from typing import Protocol

from ..task_graph.task_matching import InSessionTaskMatchApplyResult


class L2TaskAdmissionStorePort(Protocol):
    """The sole persistence operation required by L2 task admission."""

    def apply_insession_task_matches(
        self,
        *,
        session_id: str,
        source_turn_id: str,
        apply_id: str,
        proposal: object,
        exposed_catalog_ids: tuple[str, ...],
        expected_window_revision: int,
    ) -> InSessionTaskMatchApplyResult: ...


def apply_persisted_task_matches(
    *,
    session_id: str,
    source_turn_id: str,
    proposal: object,
    exposed_catalog_ids: tuple[str, ...],
    expected_window_revision: int,
    store: L2TaskAdmissionStorePort | None = None,
) -> InSessionTaskMatchApplyResult:
    """Apply one validated proposal, replaying the exact command after response loss."""

    if store is None:
        from ...session.l2_store import task_graph as store

    command = dict(
        session_id=session_id,
        source_turn_id=source_turn_id,
        apply_id=f"entry-task-match-{source_turn_id}",
        proposal=proposal,
        exposed_catalog_ids=exposed_catalog_ids,
        expected_window_revision=expected_window_revision,
    )
    try:
        return store.apply_insession_task_matches(**command)
    except Exception:
        # The stable apply id and immutable payload let the store distinguish a
        # pre-commit failure from a committed mutation whose response was lost.
        return store.apply_insession_task_matches(**command)


__all__ = [
    "L2TaskAdmissionStorePort",
    "apply_persisted_task_matches",
]
