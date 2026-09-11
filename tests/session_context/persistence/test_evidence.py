import hashlib
import json
import sqlite3

import pytest

from personagraph.session.context import evidence
from personagraph.session import store as ss
from personagraph.session.context.models import EvidenceKind
from tests.helpers.session_records import (
    append_test_turn,
    complete_test_turn_execution,
)


@pytest.fixture(autouse=True)
def _isolated_db(tmp_path, monkeypatch):
    monkeypatch.setattr(ss, "DB_PATH", tmp_path / "sessions.sqlite")


def _session():
    return ss.create_session("Entelecheia", "evidence test")


def test_formal_turn_pairs_are_complete_and_session_scoped():
    first_sid = _session()
    second_sid = _session()

    first_result = complete_test_turn_execution(
        first_sid,
        1,
        user_content="hello",
        assistant_content="hi",
    )
    second_result = complete_test_turn_execution(
        second_sid,
        1,
        user_content="other",
        assistant_content="reply",
    )
    first = first_result["pair"]
    second = second_result["pair"]

    assert first["user_turn_idx"] == 0
    assert first["assistant_turn_idx"] == 1
    assert second["user_turn_idx"] == 0
    assert second["assistant_turn_idx"] == 1
    assert [turn["content"] for turn in ss.get_turns(first_sid)] == ["hello", "hi"]
    assert [turn["content"] for turn in ss.get_turns(second_sid)] == [
        "other",
        "reply",
    ]
    assert ss.get_committed_turn_pair(second_sid, str(first["run_id"])) is None


def test_turns_are_adapted_with_stable_ids_without_second_fulltext_table():
    sid = _session()
    user_idx = append_test_turn(sid, "user", "请逐步讲解")
    assistant_idx = append_test_turn(sid, "assistant", "好的")

    records = evidence.list_turn_evidence(sid)
    assert [record.id for record in records] == [
        evidence.turn_evidence_id(sid, user_idx),
        evidence.turn_evidence_id(sid, assistant_idx),
    ]
    assert [record.kind for record in records] == [
        EvidenceKind.USER_TURN,
        EvidenceKind.ASSISTANT_TURN,
    ]
    exact = evidence.get_evidence(records[0].id)
    assert exact == records[0]

    with sqlite3.connect(ss.DB_PATH) as conn:
        assert conn.execute("SELECT COUNT(*) FROM session_evidence_events").fetchone()[0] == 0


def test_non_turn_event_stores_only_bounded_excerpt_hash_and_reference():
    sid = _session()
    payload = "x" * 5000
    record = evidence.append_event(
        sid,
        EvidenceKind.TOOL_RESULT,
        "tool-run:abc",
        payload,
        metadata={"tool_id": "doc_read", "artifact_ref": "artifact:1"},
        created_at="2026-07-12T18:00:00+08:00",
    )
    assert len(record.content_excerpt) == evidence.MAX_EVIDENCE_EXCERPT_CHARS
    assert record.content_excerpt.endswith("…")
    assert record.content_hash == hashlib.sha256(payload.encode()).hexdigest()
    assert record.created_at == "2026-07-12T10:00:00+00:00"
    assert record.metadata["artifact_ref"] == "artifact:1"

    with sqlite3.connect(ss.DB_PATH) as conn:
        stored = conn.execute(
            "SELECT content_excerpt, content_hash FROM session_evidence_events WHERE id=?",
            (record.id,),
        ).fetchone()
    assert stored[0] != payload
    assert stored[1] == record.content_hash


def test_event_append_is_idempotent_and_explicit_id_collision_fails_closed():
    sid = _session()
    kwargs = dict(
        session_id=sid,
        kind=EvidenceKind.APPROVAL,
        source_ref="approval:42",
        content="approved",
        event_id="event:fixed",
        created_at="2026-07-12T10:00:00+00:00",
    )
    first = evidence.append_event(**kwargs)
    second = evidence.append_event(**kwargs)
    assert second == first
    assert len(evidence.list_event_evidence(sid)) == 1

    with pytest.raises(ValueError, match="id collision"):
        evidence.append_event(**{**kwargs, "content": "rejected"})


def test_invalid_event_inputs_fail_before_storage():
    sid = _session()
    with pytest.raises(ValueError, match="turn evidence"):
        evidence.append_event(sid, EvidenceKind.USER_TURN, "turn:1", "hello")
    with pytest.raises(ValueError, match="unknown session"):
        evidence.append_event("missing", EvidenceKind.CORRECTION, "c:1", "fix")
    with pytest.raises(ValueError, match="source_ref"):
        evidence.append_event(sid, EvidenceKind.CORRECTION, " ", "fix")
    with pytest.raises(TypeError):
        evidence.append_event(
            sid, EvidenceKind.CORRECTION, "c:2", "fix", metadata={"bad": object()}
        )


def test_evidence_reads_are_session_isolated_and_stably_ordered():
    first_sid = _session()
    second_sid = _session()
    first_turn = append_test_turn(first_sid, "user", "first")
    append_test_turn(second_sid, "user", "second")
    event = evidence.append_event(
        first_sid,
        EvidenceKind.ARTIFACT,
        "artifact:1",
        "report.pdf",
        created_at="2026-07-12T09:00:00+00:00",
    )

    first_records = evidence.list_evidence(first_sid)
    assert {record.session_id for record in first_records} == {first_sid}
    assert {record.id for record in first_records} == {
        event.id,
        evidence.turn_evidence_id(first_sid, first_turn),
    }
    assert [record.id for record in first_records] == sorted(
        [record.id for record in first_records],
        key=lambda item_id: next(
            (record.created_at, record.id) for record in first_records if record.id == item_id
        ),
    )


def test_export_includes_non_turn_events_and_purge_removes_them():
    sid = _session()
    append_test_turn(sid, "user", "hello")
    event = evidence.append_event(
        sid,
        EvidenceKind.CORRECTION,
        "correction:1",
        "use current directory",
        metadata={"field": "path"},
    )

    exported = json.loads(ss.export_session(sid, "json"))
    assert exported["evidence_events"][0]["id"] == event.id
    assert "Evidence Events" in ss.export_session(sid, "md")

    assert ss.purge_session(sid) is True
    assert evidence.list_event_evidence(sid) == []
    assert evidence.get_evidence(event.id) is None


def test_verify_search_is_session_scoped_ranked_and_bounded():
    first_sid = _session()
    second_sid = _session()
    append_test_turn(first_sid, "user", "请慢一点解释记忆架构")
    append_test_turn(first_sid, "assistant", "好的")
    append_test_turn(second_sid, "user", "另一个会话也提到记忆架构")
    evidence.append_event(
        first_sid,
        EvidenceKind.CORRECTION,
        "correction:pace",
        "不是快速解释，要慢一点",
        created_at="2026-07-12T11:00:00+00:00",
    )

    results = evidence.search_evidence(first_sid, "为什么认为我要慢一点", limit=2)
    assert len(results) == 2
    assert {record.session_id for record in results} == {first_sid}
    assert all("慢一点" in record.content_excerpt for record in results)


def test_verify_search_supports_kind_time_and_subject_filters():
    sid = _session()
    evidence.append_event(
        sid,
        EvidenceKind.TOOL_RESULT,
        "tool:reader:1",
        "first report",
        metadata={"subject": "memory-report"},
        created_at="2026-07-12T09:00:00+00:00",
    )
    expected = evidence.append_event(
        sid,
        EvidenceKind.ARTIFACT,
        "artifact:report:2",
        "second report",
        metadata={"subject": "memory-report"},
        created_at="2026-07-12T11:00:00+00:00",
    )
    results = evidence.search_evidence(
        sid,
        "report",
        kinds={EvidenceKind.ARTIFACT},
        created_after="2026-07-12T10:00:00+00:00",
        subject="memory-report",
    )
    assert results == [expected]
