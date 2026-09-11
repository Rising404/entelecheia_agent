"""Session queries have one contract across physical database layouts."""

from unittest.mock import patch

import pytest

from personagraph.session import store
from personagraph.session.persistence.history import turns
from tests.helpers.session_records import append_test_turn


@pytest.fixture(params=["partitioned", "single"])
def query_state(request, partitioned_project_state, monkeypatch):
    if request.param == "single":
        monkeypatch.setattr(
            store, "DB_PATH", partitioned_project_state / "single.sqlite"
        )


def test_session_query_limits_and_normalization(query_state):
    session_id = store.create_session("Entelecheia", title="Needle")
    with store.session_database_scope(session_id):
        append_test_turn(session_id, "user", "a needle in the text")

    with (
        patch.object(store, "_catalog", side_effect=AssertionError("catalog read")),
        patch.object(
            turns, "_get_turns_in_transaction",
            side_effect=AssertionError("transcript read"),
        ),
    ):
        for limit in (0, -1):
            assert store.list_sessions(query="needle", limit=limit) == []
            assert store.list_sessions(limit=limit) == []
            assert store.search_sessions("needle", limit=limit) == []
        assert store.search_sessions("   ") == []

    assert [row["id"] for row in store.list_sessions(query=" NEEDLE ")] == [
        session_id
    ]
    assert [row["id"] for row in store.list_sessions(query="   ")] == [session_id]
    assert store.list_sessions(folder_id="") == []


def test_session_search_filters_projects_matches_and_reads_transcript_once(query_state):
    folder_id = store.create_folder("Project")
    session_id = store.create_session(
        "Entelecheia", title="Needle", folder_id=folder_id
    )
    with store.session_database_scope(session_id):
        append_test_turn(session_id, "user", "a needle in the text")
        append_test_turn(session_id, "assistant", "another NEEDLE")
    other = store.create_session("other", title="Needle")
    store.archive_session(other)

    with patch.object(
        turns, "_get_turns_in_transaction", wraps=turns._get_turns_in_transaction
    ) as read_turns:
        results = store.search_sessions(
            " NEEDLE ", status="all", folder_id=folder_id, persona_id="Entelecheia"
        )
    assert read_turns.call_count == 1
    assert [row["id"] for row in results] == [session_id]
    assert {key: results[0][key] for key in (
        "hit_count", "match_turn_idx", "match_role", "match_snippet"
    )} == {
        "hit_count": 3,
        "match_turn_idx": 0,
        "match_role": "user",
        "match_snippet": "a needle in the text",
    }
    assert [row["id"] for row in store.list_sessions()] == [session_id]
    assert {row["id"] for row in store.list_sessions(include_archived=True)} == {
        session_id, other
    }
    archived_hits = store.search_sessions("needle", status="archived")
    assert [row["id"] for row in archived_hits] == [other]
    assert archived_hits[0]["hit_count"] == 1
    assert archived_hits[0]["match_role"] == "title"
    assert archived_hits[0]["match_turn_idx"] is None
    assert archived_hits[0]["match_snippet"] == "Needle"
    with pytest.raises(ValueError, match="invalid session status"):
        store.list_sessions(status="unknown")
