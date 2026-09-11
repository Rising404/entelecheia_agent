"""公开会话历史投影的行为覆盖。"""

from __future__ import annotations

from personagraph.session import store
from tests.helpers.session_records import complete_test_turn_execution


def test_history_facade_keeps_metadata_transcript_and_index_behavior(
    tmp_path,
    monkeypatch,
) -> None:
    """组合端口端到端保留公开历史投影。"""

    monkeypatch.setattr(store, "DB_PATH", tmp_path / "sessions.sqlite")
    session_id = store.create_session("morgan", title="history port")
    first_result = complete_test_turn_execution(
        session_id,
        1,
        user_content="first user question",
        assistant_content="first assistant answer",
    )
    second_result = complete_test_turn_execution(
        session_id,
        2,
        user_content="second user question",
        assistant_content="second assistant answer",
    )
    first = first_result["pair"]
    second = second_result["pair"]
    first_run_id = str(first["run_id"])
    second_run_id = str(second["run_id"])

    assert store.get_session(session_id)["id"] == session_id
    assert [row["id"] for row in store.list_sessions(query="first assistant")] == [
        session_id
    ]
    assert [row["id"] for row in store.search_sessions("second assistant")] == [
        session_id
    ]
    assert store.get_turn(session_id, int(first["assistant_turn_idx"]))["content"] == (
        "first assistant answer"
    )
    assert [turn["content"] for turn in store.get_turns(session_id)] == [
        "first user question",
        "first assistant answer",
        "second user question",
        "second assistant answer",
    ]
    assert store.get_committed_turn_pair(session_id, first_run_id)[
        "user_turn_idx"
    ] == first["user_turn_idx"]
    assert [pair["run_id"] for pair in store.list_committed_turn_pairs(session_id)] == [
        first_run_id,
        second_run_id,
    ]

    state = store.get_committed_turn_pair_index_state(session_id)
    assert state.pair_count == 2
    assert state.latest_user_turn_idx == second["user_turn_idx"]
    snapshot = store.get_committed_turn_pair_index_binding_snapshot(
        session_id,
        maximum_bindings=1,
    )
    assert [binding.run_id for binding in snapshot.bindings] == [first_run_id]
    assert snapshot.binding_enumeration_complete is False
    assert [index for index, _ in store.get_user_turn_markers(session_id)] == [0, 2]
    assert store.get_history_messages(session_id) == [
        {"role": "user", "content": "first user question"},
        {"role": "assistant", "content": "first assistant answer"},
        {"role": "user", "content": "second user question"},
        {"role": "assistant", "content": "second assistant answer"},
    ]
