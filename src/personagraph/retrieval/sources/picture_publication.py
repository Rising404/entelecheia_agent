"""Retrieval outbox adapter for newly committed picture observations."""

from __future__ import annotations

from collections.abc import Callable
import sqlite3

from ...workspace.pictures.observations.contracts import (
    DEFAULT_PICTURE_OBSERVATION_WINDOW_POLICY,
    PictureObservationCommitResult,
    PictureObservationRecord,
    PictureObservationWindowPolicy,
)
from ...workspace.pictures.observations.publication import (
    PictureObservationPublicationResult,
)
from ...workspace.pictures.observations.repository import (
    PictureObservationRepository,
)
from ..lifecycle.outbox import SqliteRetrievalOutbox
from .events import (
    build_picture_observation_trash_event,
    build_picture_observation_upsert_event,
)


RetrievalDataVersionResolver = Callable[[sqlite3.Connection], str | None]


class PictureObservationPublicationUnavailable(RuntimeError):
    """An indexable commit cannot be atomically paired with its outbox event."""


class PictureObservationOutboxPublisher:
    """Implement the neutral publication port using the shared-project outbox.

    The Host picture-publication application invokes this adapter after committing one
    logical OCR/VLM observation, using the same SQLite connection, transaction and FIFO
    policy.  It remains provider- and tool-agnostic: only the composition layer selects
    this concrete retrieval adapter for production publication.
    """

    def __init__(
        self,
        *,
        retrieval_data_version_resolver: RetrievalDataVersionResolver,
        repository: PictureObservationRepository | None = None,
        outbox: SqliteRetrievalOutbox | None = None,
        window_policy: PictureObservationWindowPolicy = (
            DEFAULT_PICTURE_OBSERVATION_WINDOW_POLICY
        ),
    ) -> None:
        if not callable(retrieval_data_version_resolver):
            raise TypeError("retrieval_data_version_resolver must be callable")
        self._resolve_data_version = retrieval_data_version_resolver
        self._repository = repository or PictureObservationRepository()
        self._outbox = outbox or SqliteRetrievalOutbox()
        if not isinstance(window_policy, PictureObservationWindowPolicy):
            raise TypeError("window_policy must be PictureObservationWindowPolicy")
        self._policy = window_policy

    def publish_in_transaction(
        self,
        conn: sqlite3.Connection,
        commit: PictureObservationCommitResult,
        *,
        occurred_at: str,
    ) -> PictureObservationPublicationResult:
        if not isinstance(conn, sqlite3.Connection) or not conn.in_transaction:
            raise ValueError("picture publication requires a caller-owned transaction")
        if not isinstance(commit, PictureObservationCommitResult):
            raise TypeError("commit must be PictureObservationCommitResult")
        if commit.window_policy != self._policy:
            raise ValueError("commit FIFO policy does not match publisher policy")
        if not isinstance(occurred_at, str) or not occurred_at.strip():
            raise ValueError("occurred_at must be non-empty")
        if not commit.inserted:
            return PictureObservationPublicationResult(
                reason_code="picture_observation_replay",
            )

        authoritative_window = self._repository.list_recent_fifo(
            conn,
            picture_id=commit.observation.picture_id,
            limit=self._policy.max_active_entries,
        )
        if authoritative_window != commit.active_window.observations:
            raise ValueError("commit active window does not match picture authority")
        if (
            not authoritative_window
            or authoritative_window[-1].observation_id
            != commit.observation.observation_id
        ):
            raise ValueError("inserted observation is not the active FIFO tail")
        if not self._repository.is_in_current_active_window(
            conn,
            observation_id=commit.observation.observation_id,
            max_active_entries=self._policy.max_active_entries,
        ):
            raise PictureObservationPublicationUnavailable(
                "inserted observation is not current and active"
            )

        evicted: PictureObservationRecord | None = None
        if len(authoritative_window) == self._policy.max_active_entries:
            evicted = self._repository.get_immediately_before(
                conn,
                picture_id=commit.observation.picture_id,
                sequence=authoritative_window[0].sequence,
            )

        candidates: list[tuple[str, PictureObservationRecord]] = []
        if evicted is not None and evicted.draft.text.strip():
            candidates.append(("trash", evicted))
        if commit.observation.draft.text.strip():
            candidates.append(("upsert", commit.observation))
        if not candidates:
            return PictureObservationPublicationResult(
                reason_code="picture_observation_has_no_indexable_transition",
            )

        retrieval_data_version = self._resolve_data_version(conn)
        if not isinstance(retrieval_data_version, str) or not retrieval_data_version.strip():
            raise PictureObservationPublicationUnavailable(
                "retrieval data version is unavailable"
            )

        events = tuple(
            event
            for transition, observation in candidates
            if (
                event := (
                    build_picture_observation_trash_event(
                        observation=observation,
                        evicted_by_observation_id=commit.observation.observation_id,
                        retrieval_data_version=retrieval_data_version,
                        occurred_at=occurred_at,
                    )
                    if transition == "trash"
                    else build_picture_observation_upsert_event(
                        observation=observation,
                        retrieval_data_version=retrieval_data_version,
                        occurred_at=occurred_at,
                    )
                )
            )
            is not None
        )
        enqueued_ids = tuple(
            event.event_id
            for event in events
            if self._outbox.enqueue(conn, event)
        )
        return PictureObservationPublicationResult(
            publication_ids=tuple(event.event_id for event in events),
            enqueued_publication_ids=enqueued_ids,
        )


__all__ = [
    "PictureObservationOutboxPublisher",
    "PictureObservationPublicationUnavailable",
]
