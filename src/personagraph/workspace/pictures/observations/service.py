"""Application service for one atomic picture observation append."""

from __future__ import annotations

from dataclasses import dataclass
import sqlite3

from .contracts import (
    DEFAULT_PICTURE_OBSERVATION_WINDOW_POLICY,
    PictureObservationCommitResult,
    PictureObservationDraft,
    PictureObservationWindowPolicy,
)
from .projection import project_picture_observation_window
from .repository import PictureObservationRepository


@dataclass(frozen=True, slots=True)
class PictureObservationService:
    """Append an observation and return its current derived FIFO window.

    Transaction lifetime belongs to the caller. This service deliberately performs no
    commit, rollback, schema initialization, model call, network access or retrieval write.
    """

    repository: PictureObservationRepository
    policy: PictureObservationWindowPolicy = DEFAULT_PICTURE_OBSERVATION_WINDOW_POLICY

    def __post_init__(self) -> None:
        if not isinstance(self.repository, PictureObservationRepository):
            raise TypeError("repository must be PictureObservationRepository")
        if not isinstance(self.policy, PictureObservationWindowPolicy):
            raise TypeError("policy must be PictureObservationWindowPolicy")

    def commit_in_transaction(
        self,
        conn: sqlite3.Connection,
        draft: PictureObservationDraft,
        *,
        created_at: str,
    ) -> PictureObservationCommitResult:
        append = self.repository.append_in_transaction(
            conn,
            draft,
            created_at=created_at,
        )
        observations = self.repository.list_recent_fifo(
            conn,
            picture_id=draft.picture_id,
            limit=self.policy.max_active_entries,
        )
        active_window = project_picture_observation_window(
            observations,
            policy=self.policy,
        )
        return PictureObservationCommitResult(
            observation=append.observation,
            active_window=active_window,
            window_policy=self.policy,
            inserted=append.inserted,
        )


__all__ = ["PictureObservationService"]
