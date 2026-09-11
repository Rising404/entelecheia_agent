"""Durable L2 Task/lane routing authority tests."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from personagraph.l2.entry_adapter.task_routing import (
    L2TaskRoutingAuthorityError,
    select_l2_task_processing_target,
)
from personagraph.l2.task_graph.task_matching import (
    ExistingRootTaskMatchProposal,
    InSessionTaskMatchApplyResult,
    NewRootTaskMatchProposal,
)
from personagraph.runtime.entry.ingress.model_contracts import EntryClassification
from personagraph.runtime.entry.routing.selection import (
    select_entry_processing_route,
)
from personagraph.runtime.turn.contracts import (
    AcceptedEntryTurn,
    EntryExecutionSnapshot,
)


def _new_root(key: str) -> NewRootTaskMatchProposal:
    return NewRootTaskMatchProposal(
        local_key=key,
        title=f"Task {key}",
        objective="Refactor the entry routing policy",
        source_excerpt="Refactor the entry routing policy",
    )


def _existing_root(
    task_id: str,
    *,
    execute_current: bool = False,
) -> ExistingRootTaskMatchProposal:
    return ExistingRootTaskMatchProposal(
        insession_task_id=task_id,
        source_excerpt="Continue the current task",
        execute_current=execute_current,
    )


class _RoutingStore:
    def __init__(
        self,
        *,
        status: str = "active",
        fail_read: str | None = None,
        pending_questions: tuple[object, ...] = (),
    ) -> None:
        self.status = status
        self.fail_read = fail_read
        self.pending_questions = pending_questions
        self.manifest_reads = 0
        self.task_reads = 0
        self.question_reads = 0
        self.manifest = SimpleNamespace(lanes=())

    def get_insession_task_details(self, _session_id: str, _task_id: str):
        self.task_reads += 1
        if self.fail_read == "task":
            raise OSError("Task authority unavailable")
        return SimpleNamespace(status=SimpleNamespace(value=self.status))

    def get_insession_task_execution_lane_manifest(self, **_kwargs: object):
        self.manifest_reads += 1
        if self.fail_read == "manifest":
            raise OSError("manifest authority unavailable")
        if self.fail_read == "programming":
            raise AssertionError("injected programming error")
        return self.manifest

    def list_pending_user_questions(self, **_kwargs: object):
        self.question_reads += 1
        if self.fail_read == "questions":
            raise OSError("pending-question authority unavailable")
        return self.pending_questions


def _durable_lane(
    *,
    task_id: str = "task-1",
    match_type: str = "existing_root",
) -> object:
    return SimpleNamespace(
        lanes=(
            SimpleNamespace(
                insession_task_id=task_id,
                execution_requested=True,
                matches=(SimpleNamespace(match_type=match_type),),
            ),
        )
    )


def _accepted_turn() -> AcceptedEntryTurn:
    return AcceptedEntryTurn(
        session_id="session-1",
        turn_id="turn-1",
        client_request_id="request-1",
        user_input="Continue the current task",
        attachment_ids=(),
        window_revision=1,
        replayed=False,
        execution_snapshot=EntryExecutionSnapshot.create(
            features={},
            post_commit_job_kinds=(),
        ),
    )


def _existing_root_case() -> tuple[
    EntryClassification,
    InSessionTaskMatchApplyResult,
]:
    return (
        EntryClassification(
            processing_level="L2",
            task_matches=(_existing_root("task-1", execute_current=True),),
        ),
        InSessionTaskMatchApplyResult(
            status="applied",
            related_insession_task_ids=("task-1",),
            turn_task_link_revision=1,
        ),
    )


def test_durable_lane_receipt_selects_the_l2_target() -> None:
    classification, applied = _existing_root_case()
    store = _RoutingStore()
    store.manifest = _durable_lane()

    target = select_l2_task_processing_target(
        accepted=_accepted_turn(),
        classification=classification,
        applied=applied,
        store=store,
    )
    public_route = select_entry_processing_route(
        accepted=_accepted_turn(),
        classification=classification,
        applied=applied,
        processing_level="L2",
        store=store,
    )

    assert target == "task-1"
    assert public_route.kind == "l2_task"
    assert public_route.task_id == "task-1"


def test_l2_target_requires_exactly_one_durable_lane() -> None:
    classification, applied = _existing_root_case()

    with pytest.raises(
        L2TaskRoutingAuthorityError,
        match="exactly one durable lane",
    ):
        select_l2_task_processing_target(
            accepted=_accepted_turn(),
            classification=classification,
            applied=applied,
            store=_RoutingStore(),
        )


@pytest.mark.parametrize("failed_read", ("manifest", "task", "questions"))
def test_authority_read_failure_never_degrades_to_a_plain_reply(
    failed_read: str,
) -> None:
    classification, applied = _existing_root_case()
    store = _RoutingStore(
        status="awaiting_user" if failed_read == "questions" else "active",
        fail_read=failed_read,
    )
    store.manifest = _durable_lane()

    with pytest.raises(L2TaskRoutingAuthorityError):
        select_l2_task_processing_target(
            accepted=_accepted_turn(),
            classification=classification,
            applied=applied,
            store=store,
        )


def test_programming_errors_are_not_misclassified_as_store_outages() -> None:
    classification, applied = _existing_root_case()
    store = _RoutingStore(fail_read="programming")

    with pytest.raises(AssertionError, match="injected programming error"):
        select_l2_task_processing_target(
            accepted=_accepted_turn(),
            classification=classification,
            applied=applied,
            store=store,
        )


def test_nonexecution_and_terminal_task_do_not_select_an_l2_target() -> None:
    no_execution_store = _RoutingStore()
    no_execution = select_l2_task_processing_target(
        accepted=_accepted_turn(),
        classification=EntryClassification(
            processing_level="L2",
            task_matches=(_existing_root("task-1", execute_current=False),),
        ),
        applied=None,
        store=no_execution_store,
    )
    assert no_execution is None
    assert no_execution_store.manifest_reads == 0

    classification, applied = _existing_root_case()
    terminal_store = _RoutingStore(status="completed")
    terminal_store.manifest = _durable_lane()
    terminal = select_l2_task_processing_target(
        accepted=_accepted_turn(),
        classification=classification,
        applied=applied,
        store=terminal_store,
    )
    assert terminal is None


def test_awaiting_user_task_requires_its_durable_question() -> None:
    classification, applied = _existing_root_case()
    store = _RoutingStore(
        status="awaiting_user",
        pending_questions=(
            SimpleNamespace(
                insession_task_id="task-1",
                question="Which source should I use?",
            ),
        ),
    )
    store.manifest = _durable_lane()

    assert select_l2_task_processing_target(
        accepted=_accepted_turn(),
        classification=classification,
        applied=applied,
        store=store,
    ) == "task-1"


def test_new_root_mapping_must_match_the_durable_lane() -> None:
    classification = EntryClassification(
        processing_level="L2",
        task_matches=(_new_root("new-task"),),
    )
    applied = InSessionTaskMatchApplyResult(
        status="applied",
        created_insession_task_ids_by_local_key={"new-task": "task-1"},
        related_insession_task_ids=("task-1",),
        turn_task_link_revision=1,
    )
    store = _RoutingStore()
    store.manifest = _durable_lane(match_type="new_root")

    assert select_l2_task_processing_target(
        accepted=_accepted_turn(),
        classification=classification,
        applied=applied,
        store=store,
    ) == "task-1"

    mismatched = applied.model_copy(
        update={
            "created_insession_task_ids_by_local_key": {
                "new-task": "task-other"
            }
        }
    )
    with pytest.raises(
        L2TaskRoutingAuthorityError,
        match="new-Task mapping",
    ):
        select_l2_task_processing_target(
            accepted=_accepted_turn(),
            classification=classification,
            applied=mismatched,
            store=store,
        )
