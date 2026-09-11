"""Entry owns the Turn-local trajectory linkage lifecycle."""

from __future__ import annotations

from personagraph.runtime import entry
from personagraph.runtime.entry import application as entry_application
from personagraph.runtime.turn.contracts import (
    AcceptedEntryTurn,
    EntryExecutionSnapshot,
)
from personagraph.runtime.turn_deadline import TurnDeadline
from personagraph.trajectory.scope import current_turn_linkage


def _accepted(*, replayed: bool = False) -> AcceptedEntryTurn:
    return AcceptedEntryTurn(
        session_id="session-trajectory",
        turn_id="turn-trajectory",
        client_request_id="request-trajectory",
        user_input="probe",
        attachment_ids=(),
        window_revision=1,
        replayed=replayed,
        execution_snapshot=EntryExecutionSnapshot.create(
            features={},
            post_commit_job_kinds=(),
        ),
    )


def test_execute_accepted_turn_binds_and_releases_trajectory_linkage(
    monkeypatch,
) -> None:
    accepted = _accepted()
    marker = object()
    observed = []

    def execute(**_kwargs):
        observed.append(current_turn_linkage())
        return marker

    monkeypatch.setattr(entry_application, "_execute_new_turn", execute)

    result = entry.execute_accepted_entry_turn(
        accepted=accepted,
        on_stream_event=None,
        store=object(),  # type: ignore[arg-type]
        session_lease_held=True,
    )

    assert result is marker
    assert observed[0] is not None
    assert (observed[0].session_id, observed[0].turn_id) == (
        accepted.session_id,
        accepted.turn_id,
    )
    assert current_turn_linkage() is None


def test_standalone_l1_resume_binds_and_releases_trajectory_linkage(
    monkeypatch,
) -> None:
    from personagraph.runtime.l1 import recovery

    accepted = _accepted(replayed=True)
    observed = []

    def claim(**_kwargs):
        observed.append(current_turn_linkage())
        return None

    monkeypatch.setattr(recovery, "claim_replayed_l1_turn", claim)

    result = entry_application._try_resume_replayed_l1_turn(
        accepted=accepted,
        features={},
        on_stream_event=None,
        store=object(),  # type: ignore[arg-type]
    )

    assert result is None
    assert observed[0] is not None
    assert (observed[0].session_id, observed[0].turn_id) == (
        accepted.session_id,
        accepted.turn_id,
    )
    assert current_turn_linkage() is None


def test_standalone_auxiliary_resume_chain_binds_trajectory_linkage(
    monkeypatch,
) -> None:
    accepted = _accepted(replayed=True)
    observed = []
    marker = object()

    def execute(**_kwargs):
        observed.append(current_turn_linkage())
        return marker

    monkeypatch.setattr(
        entry_application,
        "_execute_auxiliary_production_chain_scoped",
        execute,
    )

    result = entry_application._execute_auxiliary_production_chain(
        accepted=accepted,
        task_id="task-1",
        revision=1,
        related_insession_task_ids=("task-1",),
        deadline=TurnDeadline.starting_now(60.0),
        emit=lambda _event: None,
        store=object(),  # type: ignore[arg-type]
        features={},
    )

    assert result is marker
    assert observed[0] is not None
    assert (observed[0].session_id, observed[0].turn_id) == (
        accepted.session_id,
        accepted.turn_id,
    )
    assert current_turn_linkage() is None
