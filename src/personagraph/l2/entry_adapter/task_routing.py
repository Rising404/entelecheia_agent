"""Authenticate the durable Task lane selected for L2 Entry execution."""

from __future__ import annotations

import sqlite3
from typing import Mapping, Protocol, cast

from personagraph.runtime.turn.contracts import (
    AcceptedEntryTurn,
    ProcessingRouteAuthorityError,
)
from personagraph.session.insession_task_contracts import (
    InSessionTaskPersistenceError,
)


class L2TaskMatchPort(Protocol):
    """Task-match fields inspected after the proposal has been validated."""

    match_type: str
    execute_current: bool
    local_key: str
    insession_task_id: str


class L2TaskClassificationPort(Protocol):
    """Validated classification projection required by L2 routing."""

    task_matches: tuple[object, ...]


class L2TaskMatchApplyResultPort(Protocol):
    """Durable task-match receipt fields required by L2 routing."""

    created_insession_task_ids_by_local_key: Mapping[str, str]
    related_insession_task_ids: tuple[str, ...]


class L2TaskRoutingStorePort(Protocol):
    """Read-side durable facts required to admit the L2 execution lane."""

    def get_insession_task_details(
        self,
        session_id: str,
        insession_task_id: str,
    ) -> object | None: ...

    def get_insession_task_execution_lane_manifest(
        self,
        **kwargs: object,
    ) -> object: ...

    def list_pending_user_questions(
        self,
        **kwargs: object,
    ) -> tuple[object, ...]: ...


class _StoreBackedL2TaskRouting:
    """Resolve persistence owners only after the public route selects L2."""

    def get_insession_task_details(
        self,
        session_id: str,
        insession_task_id: str,
    ) -> object | None:
        from personagraph.session.l2_store import task_graph

        return task_graph.get_insession_task_details(
            session_id,
            insession_task_id,
        )

    def get_insession_task_execution_lane_manifest(
        self,
        **kwargs: object,
    ) -> object:
        from personagraph.session.l2_store import task_graph

        return task_graph.get_insession_task_execution_lane_manifest(**kwargs)

    def list_pending_user_questions(
        self,
        **kwargs: object,
    ) -> tuple[object, ...]:
        from personagraph.session import store

        return store.list_pending_user_questions(**kwargs)


_STORE_BACKED_L2_TASK_ROUTING = _StoreBackedL2TaskRouting()
_KNOWN_TASK_STATUSES = frozenset({
    "proposed",
    "active",
    "awaiting_user",
    "waiting_external",
    "interrupted",
    "blocked",
    "cancelled",
    "completed",
})


class L2TaskRoutingAuthorityError(ProcessingRouteAuthorityError):
    """L2 execution request cannot be authenticated from durable authority."""


_AUTHORITY_READ_ERRORS = (
    InSessionTaskPersistenceError,
    OSError,
    sqlite3.Error,
    TypeError,
    ValueError,
)


def select_l2_task_processing_target(
    *,
    accepted: AcceptedEntryTurn,
    classification: L2TaskClassificationPort,
    applied: L2TaskMatchApplyResultPort | None,
    store: L2TaskRoutingStorePort | None = None,
) -> str | None:
    """Return the one durable Task eligible for L2 execution, if any.

    Before the task-match batch is applied, ``execute_current`` is only a model
    proposal. The persisted lane manifest is the replay authority, so missing or
    inconsistent read-side facts fail closed before any model effect begins.
    """

    if applied is None or len(classification.task_matches) != 1:
        return None
    if store is None:
        store = _STORE_BACKED_L2_TASK_ROUTING
    match = cast(L2TaskMatchPort, classification.task_matches[0])
    match_type = match.match_type
    if match_type not in {
        "new_root",
        "existing_root",
        "existing_root_target_change",
    }:
        return None
    if match_type != "new_root" and match.execute_current is not True:
        return None

    try:
        manifest = store.get_insession_task_execution_lane_manifest(
            session_id=accepted.session_id,
            turn_id=accepted.turn_id,
        )
    except _AUTHORITY_READ_ERRORS as exc:
        raise L2TaskRoutingAuthorityError(
            "L2 execution-lane manifest is unavailable"
        ) from exc

    lanes = getattr(manifest, "lanes", None)
    if not isinstance(lanes, tuple) or len(lanes) != 1:
        raise L2TaskRoutingAuthorityError(
            "L2 execution request requires exactly one durable lane"
        )
    lane = lanes[0]
    task_id = getattr(lane, "insession_task_id", None)
    lane_matches = getattr(lane, "matches", None)
    if not isinstance(task_id, str) or not task_id.strip():
        raise L2TaskRoutingAuthorityError(
            "L2 execution lane has no durable Task identity"
        )
    if getattr(lane, "execution_requested", None) is not True:
        raise L2TaskRoutingAuthorityError(
            "L2 execution request disagrees with its durable lane"
        )
    if not isinstance(lane_matches, tuple) or len(lane_matches) != 1:
        raise L2TaskRoutingAuthorityError(
            "L2 execution lane has an invalid match receipt"
        )
    if getattr(lane_matches[0], "match_type", None) != match_type:
        raise L2TaskRoutingAuthorityError(
            "L2 execution lane does not match the accepted proposal"
        )
    if match_type == "new_root":
        if (
            applied.created_insession_task_ids_by_local_key.get(match.local_key)
            != task_id
        ):
            raise L2TaskRoutingAuthorityError(
                "L2 new-Task mapping disagrees with its durable lane"
            )
    elif match.insession_task_id != task_id:
        raise L2TaskRoutingAuthorityError(
            "L2 Task identity disagrees with its durable lane"
        )
    if applied.related_insession_task_ids != (task_id,):
        raise L2TaskRoutingAuthorityError(
            "L2 applied receipt disagrees with its durable lane"
        )

    try:
        task = store.get_insession_task_details(accepted.session_id, task_id)
    except _AUTHORITY_READ_ERRORS as exc:
        raise L2TaskRoutingAuthorityError(
            "L2 durable Task authority is unavailable"
        ) from exc
    if task is None:
        raise L2TaskRoutingAuthorityError(
            "L2 durable Task authority is missing"
        )
    raw_status = getattr(task, "status", None)
    status = getattr(raw_status, "value", raw_status)
    if not isinstance(status, str) or status not in _KNOWN_TASK_STATUSES:
        raise L2TaskRoutingAuthorityError(
            "L2 durable Task has an invalid status"
        )
    if status in {"blocked", "cancelled", "completed"}:
        return None
    if status == "awaiting_user":
        try:
            questions = store.list_pending_user_questions(
                session_id=accepted.session_id
            )
        except _AUTHORITY_READ_ERRORS as exc:
            raise L2TaskRoutingAuthorityError(
                "L2 pending-question authority is unavailable"
            ) from exc
        if not isinstance(questions, tuple):
            raise L2TaskRoutingAuthorityError(
                "L2 pending-question authority has an invalid shape"
            )
        pending = tuple(
            question
            for question in questions
            if getattr(question, "insession_task_id", None) == task_id
        )
        if len(pending) != 1:
            raise L2TaskRoutingAuthorityError(
                "L2 awaiting-user Task lacks one durable pending question"
            )
    return task_id


__all__ = [
    "L2TaskRoutingAuthorityError",
    "L2TaskRoutingStorePort",
    "select_l2_task_processing_target",
]
