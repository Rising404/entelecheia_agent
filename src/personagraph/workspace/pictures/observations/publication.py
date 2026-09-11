"""Neutral post-commit publication boundary for picture observations.

The picture domain owns the fact that a new observation was committed.  It does not
know which derived index consumes that fact, how generations are selected, or which
outbox implementation is used.  A composition layer may call this port inside the
same caller-owned transaction after :class:`PictureObservationService` returns.
"""

from __future__ import annotations

from dataclasses import dataclass
import sqlite3
from typing import Protocol

from .contracts import (
    PictureObservationCommitResult,
)


@dataclass(frozen=True, slots=True)
class PictureObservationPublicationResult:
    """Content-free acknowledgement from a derived-data publisher."""

    publication_ids: tuple[str, ...] = ()
    enqueued_publication_ids: tuple[str, ...] = ()
    reason_code: str | None = None

    def __post_init__(self) -> None:
        publication_ids = tuple(self.publication_ids)
        enqueued_ids = tuple(self.enqueued_publication_ids)
        if any(not isinstance(item, str) or not item.strip() for item in publication_ids):
            raise ValueError("publication_ids must contain non-empty strings")
        if len(set(publication_ids)) != len(publication_ids):
            raise ValueError("publication_ids must not contain duplicates")
        if any(item not in publication_ids for item in enqueued_ids):
            raise ValueError("enqueued publications must belong to publication_ids")
        if len(set(enqueued_ids)) != len(enqueued_ids):
            raise ValueError("enqueued_publication_ids must not contain duplicates")
        if self.reason_code is not None and (
            not isinstance(self.reason_code, str) or not self.reason_code.strip()
        ):
            raise ValueError("reason_code must be non-empty when provided")
        object.__setattr__(self, "publication_ids", publication_ids)
        object.__setattr__(self, "enqueued_publication_ids", enqueued_ids)


class PictureObservationPublicationPort(Protocol):
    """Publish one commit without taking ownership of its transaction."""

    def publish_in_transaction(
        self,
        conn: sqlite3.Connection,
        commit: PictureObservationCommitResult,
        *,
        occurred_at: str,
    ) -> PictureObservationPublicationResult: ...


__all__ = [
    "PictureObservationPublicationPort",
    "PictureObservationPublicationResult",
]
