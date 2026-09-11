"""Recover durable Current Session retrieval effects before accepting a Turn."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from ...retrieval.sources.session.post_commit import (
    index_committed_session_pair,
    reconcile_session_retrieval_effects,
)
from ...session.session_summary_jobs import SessionSummaryJobGenerator
from .contracts import TurnPostCommitJobStore
from .runner import (
    process_due_turn_post_commit_jobs,
    release_turn_window_if_post_commit_settled,
)

if TYPE_CHECKING:
    from ...retrieval.sources.session.contracts import SessionTurnRetrievalBinding


def reconcile_session_retrieval_before_turn(
    *,
    session_id: str,
    store: TurnPostCommitJobStore,
    binding: SessionTurnRetrievalBinding,
    summary_generator: SessionSummaryJobGenerator | None = None,
) -> None:
    """Repair the latest committed pair before a new Turn can be accepted."""

    def index_pair(pair: Mapping[str, object]) -> str:
        return index_committed_session_pair(binding, pair)

    # Advance ordinary pending work first. The effect-proof pass below also handles
    # a process that died after indexing but before releasing its worker lease.
    process_due_turn_post_commit_jobs(
        session_id=session_id,
        store=store,
        summary_generator=summary_generator,
        session_retrieval_indexer=index_pair,
    )
    reconcile_session_retrieval_effects(
        session_id=session_id,
        store=store,
        binding=binding,
    )
    release_turn_window_if_post_commit_settled(
        session_id=session_id,
        store=store,
    )
