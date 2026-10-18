from __future__ import annotations


from personagraph.runtime import entry
from personagraph.runtime.entry import application as entry_application
from personagraph.runtime.turn.contracts import (
    AcceptedEntryTurn,
    EntryExecutionSnapshot,
    EntryTurnResult,
)


def test_execution_snapshot_round_trips_file_and_session_retrieval_authority() -> None:
    snapshot = EntryExecutionSnapshot.create(
        features={
            "file_retrieval_read_enabled": True,
            "history_retrieval_read_enabled": True,
        },
        file_retrieval_data_version="file-generation-frozen",
        session_retrieval_data_version="session-generation-frozen",
        session_retrieval_assistant_turn_cutoff=7,
        post_commit_job_kinds=(),
    )

    restored = EntryExecutionSnapshot.from_json(
        snapshot.to_json(),
        expected_sha256=snapshot.sha256,
    )

    assert restored == snapshot
    assert restored.session_retrieval_data_version == "session-generation-frozen"
    assert restored.session_retrieval_assistant_turn_cutoff == 7
    assert "history_retrieval_data_version" not in snapshot.to_json()


def test_execute_accepted_turn_uses_persisted_snapshot_as_its_only_configuration(
    monkeypatch,
) -> None:
    snapshot = EntryExecutionSnapshot.create(
        features={
            "file_retrieval_read_enabled": True,
            "turn_wall_clock_budget_s": 321.0,
        },
        file_retrieval_data_version="file-generation-frozen",
        post_commit_job_kinds=("memory_consolidation", "session_summary"),
    )
    accepted = AcceptedEntryTurn(
        session_id="session-1",
        turn_id="turn-1",
        client_request_id="request-1",
        user_input="continue",
        attachment_ids=(),
        window_revision=1,
        replayed=False,
        execution_snapshot=snapshot,
    )
    observed: dict[str, object] = {}

    def execute(**kwargs):
        observed["features"] = kwargs["features"]
        observed["post_commit_jobs"] = (
            entry_application._TURN_POST_COMMIT_JOB_KINDS.get()
        )
        return EntryTurnResult(
            session_id="session-1",
            turn_id="turn-1",
            status="incomplete",
            processing_level="L1",
            window_state="active",
            window_revision=1,
        )

    monkeypatch.setattr(entry_application, "_execute_new_turn", execute)
    result = entry.execute_accepted_entry_turn(
        accepted=accepted,
        on_stream_event=None,
        store=object(),  # type: ignore[arg-type]
        session_lease_held=True,
    )

    assert result.turn_id == "turn-1"
    assert observed == {
        "features": snapshot.features,
        "post_commit_jobs": ("memory_consolidation", "session_summary"),
    }
    assert not hasattr(entry, "_HISTORY_RETRIEVAL_DATA_VERSION")
    assert entry_application._TURN_POST_COMMIT_JOB_KINDS.get() == (
        "session_summary",
    )
