"""L2 task-match persistence adapter tests."""

from __future__ import annotations

from personagraph.l2.entry_adapter.task_admission import (
    apply_persisted_task_matches,
)
from personagraph.l2.task_graph.task_matching import InSessionTaskMatchApplyResult


class _AdmissionStore:
    def __init__(
        self,
        *,
        apply_result: InSessionTaskMatchApplyResult,
        lose_first_response: bool = False,
    ) -> None:
        self.apply_result = apply_result
        self.lose_first_response = lose_first_response
        self.apply_commands: list[dict[str, object]] = []

    def apply_insession_task_matches(
        self,
        **kwargs: object,
    ) -> InSessionTaskMatchApplyResult:
        self.apply_commands.append(kwargs)
        if self.lose_first_response and len(self.apply_commands) == 1:
            raise OSError("task-match response lost")
        return self.apply_result


def test_adapter_retries_the_exact_stable_command_after_response_loss() -> None:
    result = InSessionTaskMatchApplyResult(
        status="replayed",
        related_insession_task_ids=("task-1",),
        turn_task_link_revision=3,
    )
    store = _AdmissionStore(apply_result=result, lose_first_response=True)
    proposal = object()

    outcome = apply_persisted_task_matches(
        session_id="session-1",
        source_turn_id="turn-1",
        proposal=proposal,
        exposed_catalog_ids=("task-visible",),
        expected_window_revision=7,
        store=store,
    )

    assert outcome is result
    assert len(store.apply_commands) == 2
    assert store.apply_commands[0] == store.apply_commands[1]
    assert store.apply_commands[0] == {
        "session_id": "session-1",
        "source_turn_id": "turn-1",
        "apply_id": "entry-task-match-turn-1",
        "proposal": proposal,
        "exposed_catalog_ids": ("task-visible",),
        "expected_window_revision": 7,
    }
