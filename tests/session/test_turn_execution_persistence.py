from __future__ import annotations

import sqlite3

import pytest

from personagraph.runtime.turn.contracts import EntryExecutionSnapshot
from personagraph.runtime.turn_events import RuntimeStage, TurnEventStatus, new_turn_event
from personagraph.session import store
from personagraph.session.turn_execution_contracts import (
    TurnExecutionPersistenceError,
)


def _accept(
    session_id: str,
    *,
    request_id: str = "request-1",
    text: str = "请处理这条消息",
    attachment_ids: tuple[str, ...] = (),
) -> dict[str, object]:
    return store.accept_turn_execution(
        session_id=session_id,
        client_request_id=request_id,
        source="runtime_test",
        user_text=text,
        attachment_ids=attachment_ids,
        lease_owner="test-host",
    )


def _window(result: dict[str, object]) -> dict[str, object]:
    return result["window"]  # type: ignore[return-value]


def _turn(result: dict[str, object]) -> dict[str, object]:
    return result["turn"]  # type: ignore[return-value]


def test_accept_turn_execution_persists_input_attachments_and_window_atomically():
    session_id = store.create_session("Entelecheia")
    store.create_attachment(
        attachment_id="attachment-1",
        session_id=session_id,
        origin="user_upload",
        original_name="brief.txt",
        stored_rel_path="uploads/brief.txt",
        media_type="text/plain",
        declared_media_type="text/plain",
        size_bytes=5,
        content_hash="hash-1",
        kind="text",
    )

    accepted = _accept(session_id, attachment_ids=("attachment-1",))

    assert accepted["replayed"] is False
    assert _turn(accepted)["status"] == "running"
    assert accepted["input_message"] == {
        "message_id": _turn(accepted)["input_message_id"],
        "session_id": session_id,
        "turn_id": _turn(accepted)["turn_id"],
        "turn_idx": 0,
        "content": "请处理这条消息",
        "created_at": accepted["input_message"]["created_at"],  # type: ignore[index]
    }
    assert accepted["attachments"] == [
        {
            "attachment_id": "attachment-1",
            "binding_ordinal": 0,
            "created_at": accepted["attachments"][0]["created_at"],  # type: ignore[index]
        }
    ]
    assert _window(accepted)["window_state"] == "active"
    assert _window(accepted)["state_version"] == 1
    assert store.get_turns(session_id) == [
        {
            "turn_idx": 0,
            "role": "user",
            "content": "请处理这条消息",
            "created_at": store.get_turns(session_id)[0]["created_at"],
        }
    ]
    assert store.list_committed_turn_pairs(session_id) == []
    assert store.get_attachment("attachment-1")["turn_id"] == _turn(accepted)["turn_id"]


def test_list_turn_attachments_preserves_the_accepted_binding_order():
    session_id = store.create_session("Entelecheia")
    for attachment_id in ("attachment-a", "attachment-b"):
        store.create_attachment(
            attachment_id=attachment_id,
            session_id=session_id,
            origin="user_upload",
            original_name=f"{attachment_id}.txt",
            stored_rel_path=f"uploads/{attachment_id}.txt",
            media_type="text/plain",
            declared_media_type="text/plain",
            size_bytes=5,
            content_hash=f"hash-{attachment_id}",
            kind="text",
            project_id=f"project-{attachment_id}",
            file_id=f"file-{attachment_id}",
            file_version_id=f"version-{attachment_id}",
        )

    accepted = _accept(
        session_id,
        attachment_ids=("attachment-b", "attachment-a"),
    )
    turn_id = str(_turn(accepted)["turn_id"])

    assert [item["attachment_id"] for item in accepted["attachments"]] == [
        "attachment-b",
        "attachment-a",
    ]
    listed = store.list_turn_attachments(session_id, turn_id)
    assert [item["attachment_id"] for item in listed] == [
        "attachment-b",
        "attachment-a",
    ]
    input_message_id = str(_turn(accepted)["input_message_id"])
    assert [
        {
            "binding_ordinal": item["binding_ordinal"],
            "input_message_id": item["input_message_id"],
            "input_file_ordinal": item["input_file_ordinal"],
            "input_project_id": item["input_project_id"],
            "input_file_id": item["input_file_id"],
            "input_file_version_id": item["input_file_version_id"],
        }
        for item in listed
    ] == [
        {
            "binding_ordinal": 0,
            "input_message_id": input_message_id,
            "input_file_ordinal": 0,
            "input_project_id": "project-attachment-b",
            "input_file_id": "file-attachment-b",
            "input_file_version_id": "version-attachment-b",
        },
        {
            "binding_ordinal": 1,
            "input_message_id": input_message_id,
            "input_file_ordinal": 1,
            "input_project_id": "project-attachment-a",
            "input_file_id": "file-attachment-a",
            "input_file_version_id": "version-attachment-a",
        },
    ]


def test_list_turn_attachments_keeps_accepted_attachment_without_file_ref():
    session_id = store.create_session("Entelecheia")
    store.create_attachment(
        attachment_id="attachment-without-file-ref",
        session_id=session_id,
        origin="user_upload",
        original_name="notes.txt",
        stored_rel_path="uploads/notes.txt",
        media_type="text/plain",
        declared_media_type="text/plain",
        size_bytes=5,
        content_hash="hash-without-file-ref",
        kind="text",
    )

    accepted = _accept(
        session_id,
        attachment_ids=("attachment-without-file-ref",),
    )
    turn_id = str(_turn(accepted)["turn_id"])
    attachment = store.get_attachment("attachment-without-file-ref")
    assert attachment is not None

    assert store.list_turn_attachments(session_id, turn_id) == [
        {
            **attachment,
            "binding_ordinal": 0,
            "input_message_id": str(_turn(accepted)["input_message_id"]),
            "input_file_ordinal": None,
            "input_project_id": None,
            "input_file_id": None,
            "input_file_version_id": None,
        }
    ]


def test_list_turn_attachments_falls_back_for_legacy_bindings_without_ordinals():
    session_id = store.create_session("Entelecheia")
    for attachment_id in ("legacy-a", "legacy-b"):
        store.create_attachment(
            attachment_id=attachment_id,
            session_id=session_id,
            origin="user_upload",
            original_name=f"{attachment_id}.txt",
            stored_rel_path=f"uploads/{attachment_id}.txt",
            media_type="text/plain",
            declared_media_type="text/plain",
            size_bytes=5,
            content_hash=f"hash-{attachment_id}",
            kind="text",
            project_id=f"legacy-project-{attachment_id}",
            file_id=f"legacy-file-{attachment_id}",
            file_version_id=f"legacy-version-{attachment_id}",
        )

    store.bind_attachments_to_turn(
        session_id=session_id,
        turn_id="legacy-turn",
        attachment_ids=("legacy-b", "legacy-a"),
    )

    listed = store.list_turn_attachments(session_id, "legacy-turn")
    assert [item["attachment_id"] for item in listed] == ["legacy-a", "legacy-b"]
    assert [
        (
            item["project_id"],
            item["file_id"],
            item["file_version_id"],
            item["binding_ordinal"],
            item["input_message_id"],
            item["input_file_ordinal"],
            item["input_project_id"],
            item["input_file_id"],
            item["input_file_version_id"],
        )
        for item in listed
    ] == [
        (
            "legacy-project-legacy-a",
            "legacy-file-legacy-a",
            "legacy-version-legacy-a",
            None,
            None,
            None,
            None,
            None,
            None,
        ),
        (
            "legacy-project-legacy-b",
            "legacy-file-legacy-b",
            "legacy-version-legacy-b",
            None,
            None,
            None,
            None,
            None,
            None,
        ),
    ]


def test_accept_turn_execution_replays_same_client_request_without_second_turn():
    session_id = store.create_session("Entelecheia")
    accepted = _accept(session_id)

    replayed = store.accept_turn_execution(
        session_id=session_id,
        client_request_id="request-1",
        source="runtime_test",
        user_text="请处理这条消息",
        turn_id="different-generated-id",
        input_message_id="different-input-id",
        lease_owner="other-host",
    )

    assert replayed["replayed"] is True
    assert _turn(replayed)["turn_id"] == _turn(accepted)["turn_id"]
    assert len(store.get_turns(session_id)) == 1

    with pytest.raises(store.TurnExecutionRequestIdCollision):
        _accept(session_id, request_id="request-1", text="不同输入")


def test_turn_execution_snapshot_is_authenticated_immutable_and_replay_owned():
    session_id = store.create_session("Entelecheia")
    original = EntryExecutionSnapshot.create(
        features={"file_retrieval_read_enabled": True, "context_guard_limit": 24000},
        post_commit_job_kinds=("memory_consolidation", "session_summary"),
    )
    accepted = store.accept_turn_execution(
        session_id=session_id,
        client_request_id="snapshot-request",
        source="runtime_test",
        user_text="freeze this execution",
        execution_snapshot_json=original.to_json(),
        execution_snapshot_sha256=original.sha256,
    )
    turn_id = str(_turn(accepted)["turn_id"])
    assert _turn(accepted)["execution_snapshot_json"] == original.to_json()
    assert _turn(accepted)["execution_snapshot_sha256"] == original.sha256

    changed = EntryExecutionSnapshot.create(
        features={"file_retrieval_read_enabled": False},
        post_commit_job_kinds=("session_summary",),
    )
    replayed = store.accept_turn_execution(
        session_id=session_id,
        client_request_id="snapshot-request",
        source="runtime_test",
        user_text="freeze this execution",
        execution_snapshot_json=changed.to_json(),
        execution_snapshot_sha256=changed.sha256,
    )
    assert replayed["replayed"] is True
    assert _turn(replayed)["execution_snapshot_json"] == original.to_json()
    assert _turn(replayed)["execution_snapshot_sha256"] == original.sha256

    with store._connect() as conn, pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "UPDATE runtime_turns SET execution_snapshot_json=? WHERE turn_id=?",
            (changed.to_json(), turn_id),
        )


def test_turn_execution_snapshot_rejects_partial_or_unauthenticated_payloads():
    session_id = store.create_session("Entelecheia")
    snapshot = EntryExecutionSnapshot.create(
        features={},
        post_commit_job_kinds=("session_summary",),
    )
    with pytest.raises(ValueError, match="supplied atomically"):
        store.accept_turn_execution(
            session_id=session_id,
            client_request_id="partial-snapshot",
            source="runtime_test",
            user_text="partial",
            execution_snapshot_json=snapshot.to_json(),
        )
    with pytest.raises(ValueError, match="hash is invalid"):
        store.accept_turn_execution(
            session_id=session_id,
            client_request_id="bad-snapshot-hash",
            source="runtime_test",
            user_text="bad hash",
            execution_snapshot_json=snapshot.to_json(),
            execution_snapshot_sha256="f" * 64,
        )


def test_turn_finalization_rejects_a_stale_lease_owner() -> None:
    session_id = store.create_session("Entelecheia")
    accepted = _accept(session_id)
    turn_id = str(_turn(accepted)["turn_id"])

    with pytest.raises(TurnExecutionPersistenceError):
        store.finalize_turn_execution(
            session_id=session_id,
            turn_id=turn_id,
            expected_window_revision=1,
            processing_level="L0",
            assistant_content="旧 worker 不得提交。",
            post_commit_job_kinds=(),
            expected_lease_owner="stale-host",
        )

    inspection = store.inspect_turn_execution(session_id)
    assert inspection["turn"]["status"] == "running"
    assert inspection["window"]["window_state"] == "active"
    assert inspection["window"]["lease_owner"] == "test-host"

    completed = store.finalize_turn_execution(
        session_id=session_id,
        turn_id=turn_id,
        expected_window_revision=1,
        processing_level="L0",
        assistant_content="当前 worker 可以提交。",
        post_commit_job_kinds=(),
        expected_lease_owner="test-host",
    )
    assert completed["turn"]["status"] == "completed"


def test_client_request_lookup_is_read_only_and_survives_window_release():
    session_id = store.create_session("Entelecheia")
    accepted = _accept(session_id)
    turn_id = str(_turn(accepted)["turn_id"])

    active = store.get_turn_execution_for_client_request(
        session_id=session_id,
        client_request_id="request-1",
    )
    assert active is not None
    assert _turn(active)["turn_id"] == turn_id
    assert _window(active)["window_state"] == "active"

    finalized = store.finalize_turn_execution(
        session_id=session_id,
        turn_id=turn_id,
        expected_window_revision=1,
        processing_level="L0",
        assistant_content="已完成。",
        post_commit_job_kinds=(),
    )
    store.release_turn_execution_window(
        session_id=session_id,
        turn_id=turn_id,
        expected_window_revision=int(finalized["window"]["state_version"]),  # type: ignore[index]
    )

    completed = store.get_turn_execution_for_client_request(
        session_id=session_id,
        client_request_id="request-1",
    )
    assert completed is not None
    assert _turn(completed)["status"] == "completed"
    assert _window(completed)["window_state"] == "empty"
    assert store.get_turn_execution_for_client_request(
        session_id=session_id,
        client_request_id="missing-request",
    ) is None


def test_accepted_turn_input_remains_queryable_after_its_window_is_released():
    session_id = store.create_session("Entelecheia")
    accepted = _accept(session_id)
    turn_id = str(_turn(accepted)["turn_id"])
    finalized = store.finalize_turn_execution(
        session_id=session_id,
        turn_id=turn_id,
        expected_window_revision=1,
        processing_level="L0",
        assistant_content="已完成。",
        post_commit_job_kinds=(),
    )
    store.release_turn_execution_window(
        session_id=session_id,
        turn_id=turn_id,
        expected_window_revision=int(finalized["window"]["state_version"]),  # type: ignore[index]
    )

    assert store.get_turn_execution_input(
        session_id=session_id,
        turn_id=turn_id,
    )["content"] == "请处理这条消息"


def test_active_turn_window_rejects_other_request_without_writing_partial_input():
    session_id = store.create_session("Entelecheia")
    accepted = _accept(session_id)

    with pytest.raises(store.TurnExecutionBusyError) as caught:
        _accept(session_id, request_id="request-2", text="第二条输入")

    assert caught.value.details == {
        "session_id": session_id,
        "turn_id": _turn(accepted)["turn_id"],
        "window_state": "active",
        "window_revision": 1,
    }
    assert [turn["content"] for turn in store.get_turns(session_id)] == ["请处理这条消息"]


def test_attachment_binding_failure_rolls_back_turn_input_and_window_together():
    session_id = store.create_session("Entelecheia")

    with pytest.raises(store.AttachmentBindingError, match="unknown attachment"):
        _accept(session_id, attachment_ids=("missing",))

    assert store.get_turns(session_id) == []
    assert store.get_turn_execution_window(session_id) is None
    with store._connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM runtime_turns").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM runtime_turn_inputs").fetchone()[0] == 0


def test_window_advance_requires_the_current_compare_and_swap_revision():
    session_id = store.create_session("Entelecheia")
    accepted = _accept(session_id)
    turn_id = str(_turn(accepted)["turn_id"])

    advanced = store.advance_turn_execution_window(
        session_id=session_id,
        turn_id=turn_id,
        expected_window_revision=1,
        stage="SUPERVISOR",
        lease_owner="test-host",
        last_event_sequence=7,
    )

    assert advanced["state_version"] == 2
    assert advanced["stage"] == "SUPERVISOR"
    assert advanced["last_event_sequence"] == 7
    with pytest.raises(store.TurnExecutionWindowRevisionConflict):
        store.advance_turn_execution_window(
            session_id=session_id,
            turn_id=turn_id,
            expected_window_revision=1,
            stage="RESPONSE",
        )


def test_event_append_and_active_window_heartbeat_are_one_atomic_lease_update():
    session_id = store.create_session("Entelecheia")
    accepted = _accept(session_id)
    turn_id = str(_turn(accepted)["turn_id"])
    observed_heartbeat_at = "2000-01-01T00:00:00+00:00"
    with store._connect() as conn:
        conn.execute(
            "UPDATE turn_execution_windows SET heartbeat_at=? WHERE session_id=?",
            (observed_heartbeat_at, session_id),
        )

    sequence = store.append_runtime_turn_event(
        new_turn_event(
            turn_id=turn_id,
            session_id=session_id,
            stage=RuntimeStage.CLASSIFY,
            status=TurnEventStatus.STARTED,
        ),
        active_window_lease_owner="test-host",
    )

    assert sequence >= 1
    window = store.get_turn_execution_window(session_id)
    assert window is not None
    assert window["state_version"] == 1
    assert window["heartbeat_at"] != observed_heartbeat_at
    with pytest.raises(store.TurnExecutionLeaseConflict):
        store.mark_turn_execution_interrupted(
            session_id=session_id,
            turn_id=turn_id,
            expected_window_revision=1,
            stage="RESPONSE",
            interruption_reason="process_lost",
            expected_heartbeat_at=observed_heartbeat_at,
            require_heartbeat_match=True,
        )


def test_next_input_audit_can_settle_an_interrupted_turn_without_assistant_draft():
    session_id = store.create_session("Entelecheia")
    accepted = _accept(session_id)
    turn_id = str(_turn(accepted)["turn_id"])
    interrupted = store.mark_turn_execution_interrupted(
        session_id=session_id,
        turn_id=turn_id,
        expected_window_revision=1,
        stage="RESPONSE",
        interruption_reason="provider_unavailable",
    )
    assert interrupted["window_state"] == "interrupted"
    assert interrupted["state_version"] == 2

    settled = store.settle_interrupted_turn_execution(
        session_id=session_id,
        turn_id=turn_id,
        expected_window_revision=2,
        end_reason="provider_unavailable",
        error_code="MODEL_TRANSPORT_FAILURE",
    )

    assert settled["replayed"] is False
    assert settled["turn"]["status"] == "incomplete"  # type: ignore[index]
    assert settled["turn"]["end_reason"] == "provider_unavailable"  # type: ignore[index]
    assert settled["window"]["window_state"] == "empty"  # type: ignore[index]
    assert [turn["role"] for turn in store.get_turns(session_id)] == ["user"]
    assert store.list_committed_turn_pairs(session_id) == []

    replayed = store.settle_interrupted_turn_execution(
        session_id=session_id,
        turn_id=turn_id,
        expected_window_revision=2,
        end_reason="provider_unavailable",
        error_code="MODEL_TRANSPORT_FAILURE",
    )
    assert replayed["replayed"] is True


def test_finalization_writes_only_assistant_then_keeps_window_for_post_commit_jobs():
    session_id = store.create_session("Entelecheia")
    accepted = _accept(session_id)
    turn_id = str(_turn(accepted)["turn_id"])

    finalized = store.finalize_turn_execution(
        session_id=session_id,
        turn_id=turn_id,
        expected_window_revision=1,
        processing_level="L0",
        assistant_content="这是已经正式提交的回答。",
        post_commit_job_kinds=("session_summary", "session_rag"),
    )

    assert finalized["replayed"] is False
    assert finalized["turn"]["status"] == "completed"  # type: ignore[index]
    assert finalized["delivery"] == {
        "user_turn_idx": 0,
        "assistant_turn_idx": 1,
        "created": True,
    }
    assert [turn["role"] for turn in store.get_turns(session_id)] == ["user", "assistant"]
    assert [job["job_kind"] for job in finalized["post_commit_jobs"]] == [  # type: ignore[index]
        "session_rag",
        "session_summary",
    ]
    assert finalized["window"]["window_state"] == "post_commit_pending"  # type: ignore[index]
    assert finalized["window"]["state_version"] == 2  # type: ignore[index]

    replayed = store.finalize_turn_execution(
        session_id=session_id,
        turn_id=turn_id,
        expected_window_revision=1,
        processing_level="L0",
        assistant_content="这是已经正式提交的回答。",
        post_commit_job_kinds=("session_summary", "session_rag"),
    )
    assert replayed["replayed"] is True
    assert len(store.get_turns(session_id)) == 2


def test_post_commit_jobs_gate_window_release_and_support_explicit_stale_waiver():
    session_id = store.create_session("Entelecheia")
    accepted = _accept(session_id)
    turn_id = str(_turn(accepted)["turn_id"])
    finalized = store.finalize_turn_execution(
        session_id=session_id,
        turn_id=turn_id,
        expected_window_revision=1,
        processing_level="L2",
        assistant_content="任务结果已保存。",
        post_commit_job_kinds=("session_summary",),
    )
    job = finalized["post_commit_jobs"][0]  # type: ignore[index]
    claimed = store.claim_due_turn_post_commit_jobs(
        session_id=session_id,
        worker_id="summary-worker",
        lease_seconds=60,
    )
    assert [item["job_id"] for item in claimed] == [job["job_id"]]
    failed = store.mark_turn_post_commit_job_failed(
        job_id=str(job["job_id"]),
        worker_id="summary-worker",
        reason_code="SUMMARY_PROVIDER_UNAVAILABLE",
        retry_after_seconds=None,
    )
    assert failed["status"] == "terminal_failed"

    with pytest.raises(store.TurnPostCommitJobsPending):
        store.release_turn_execution_window(
            session_id=session_id,
            turn_id=turn_id,
            expected_window_revision=2,
        )

    inspection = store.inspect_turn_execution(session_id)
    controlled = store.apply_turn_post_commit_job_control(
        session_id=session_id,
        turn_id=turn_id,
        expected_window_revision=2,
        request_id="waive-summary-job",
        action="waive",
        job_ids=(str(job["job_id"]),),
        expected_failed_job_digest=str(inspection["failed_job_digest"]),
        actor="user",
    )
    assert controlled["replayed"] is False
    assert controlled["post_commit_jobs"][0]["status"] == "waived"  # type: ignore[index]
    assert controlled["window"]["state_version"] == 3  # type: ignore[index]

    released = store.release_turn_execution_window(
        session_id=session_id,
        turn_id=turn_id,
        expected_window_revision=3,
    )
    assert released["window_state"] == "empty"
    assert released["turn_id"] is None


def _settled_post_commit_control(action):
    session_id = store.create_session("Entelecheia")
    accepted = _accept(session_id)
    turn_id = str(_turn(accepted)["turn_id"])
    finalized = store.finalize_turn_execution(
        session_id=session_id, turn_id=turn_id, expected_window_revision=1,
        processing_level="L0", assistant_content="已保存。",
        post_commit_job_kinds=("session_retrieval_index", "session_summary"),
    )
    for job in store.claim_due_turn_post_commit_jobs(
        session_id=session_id, worker_id="test-worker", lease_seconds=60,
    ):
        store.mark_turn_post_commit_job_failed(
            job_id=job["job_id"], worker_id="test-worker",
            reason_code="TEST_FAILURE", retry_after_seconds=None,
        )
    inspection = store.inspect_turn_execution(session_id)
    command = {
        "session_id": session_id, "turn_id": turn_id, "request_id": "explicit-control",
        "action": action, "expected_window_revision": _window(finalized)["state_version"],
        "job_ids": tuple(job["job_id"] for job in inspection["post_commit_jobs"]),
        "expected_failed_job_digest": inspection["failed_job_digest"], "actor": "test-user",
    }
    controlled = store.apply_turn_post_commit_job_control(**command)
    if action == "retry":
        for job in store.claim_due_turn_post_commit_jobs(
            session_id=session_id, worker_id="retry-worker", lease_seconds=60,
        ):
            store.mark_turn_post_commit_job_applied(job_id=job["job_id"], worker_id="retry-worker")
    store.release_turn_execution_window(
        session_id=session_id, turn_id=turn_id,
        expected_window_revision=_window(controlled)["state_version"],
    )
    return command, controlled


@pytest.mark.parametrize("action", ["retry", "waive"])
@pytest.mark.parametrize("start_next_turn", [False, True])
def test_post_commit_control_exact_replay_survives_settlement_without_touching_current_window(
    action, start_next_turn,
):
    command, original = _settled_post_commit_control(action)
    session_id = command["session_id"]
    if start_next_turn:
        _accept(session_id, request_id="next-turn", text="后续任务")
    before = store.inspect_turn_execution(session_id)
    old_jobs = store.list_turn_post_commit_jobs(command["turn_id"])
    transcript = store.get_turns(session_id)

    # 集合顺序无关；相同 ID 和命令身份才是 exact replay。
    replayed = store.apply_turn_post_commit_job_control(
        **{**command, "job_ids": tuple(reversed(command["job_ids"]))},
    )

    assert replayed["replayed"] is True
    assert replayed["control"] == original["control"]
    assert replayed["post_commit_jobs"] == old_jobs
    assert replayed["window"] == before["window"]
    assert store.inspect_turn_execution(session_id) == before
    assert store.get_turns(session_id) == transcript
    with store._connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM turn_post_commit_job_controls").fetchone()[0] == 1


@pytest.mark.parametrize("change", [
    {"action": "retry"}, {"turn_id": "another-turn"}, {"actor": "another-user"},
    {"expected_window_revision": 999}, {"expected_failed_job_digest": "0" * 64},
    {"job_ids": ("another-job",)},
])
def test_post_commit_control_changed_replay_is_rejected_after_next_turn_starts(change):
    command, _original = _settled_post_commit_control("waive")
    session_id = command["session_id"]
    _accept(session_id, request_id="next-turn", text="后续任务")
    before = store.inspect_turn_execution(session_id)
    old_jobs = store.list_turn_post_commit_jobs(command["turn_id"])

    with pytest.raises(store.TurnExecutionRequestIdCollision):
        store.apply_turn_post_commit_job_control(**{**command, **change})

    assert store.inspect_turn_execution(session_id) == before
    assert store.list_turn_post_commit_jobs(command["turn_id"]) == old_jobs


def test_proven_post_commit_effect_can_settle_an_abandoned_worker_lease():
    session_id = store.create_session("Entelecheia")
    accepted = _accept(session_id)
    turn_id = str(_turn(accepted)["turn_id"])
    finalized = store.finalize_turn_execution(
        session_id=session_id,
        turn_id=turn_id,
        expected_window_revision=1,
        processing_level="L0",
        assistant_content="回答已经提交。",
        post_commit_job_kinds=("session_retrieval_index",),
    )
    job = finalized["post_commit_jobs"][0]  # type: ignore[index]
    claimed = store.claim_due_turn_post_commit_jobs(
        session_id=session_id,
        worker_id="worker-that-crashed",
        lease_seconds=420,
    )
    assert claimed[0]["status"] == "processing"

    reconciled = store.reconcile_turn_post_commit_job_applied(
        session_id=session_id,
        turn_id=turn_id,
        job_id=str(job["job_id"]),
        expected_job_kind="session_retrieval_index",
    )

    assert reconciled["status"] == "applied"
    assert reconciled["lease_owner"] is None
    assert reconciled["reason_code"] is None
    released = store.release_turn_execution_window(
        session_id=session_id,
        turn_id=turn_id,
        expected_window_revision=2,
    )
    assert released["window_state"] == "empty"


def test_applied_post_commit_job_clears_retry_schedule_metadata():
    session_id = store.create_session("Entelecheia")
    accepted = _accept(session_id)
    turn_id = str(_turn(accepted)["turn_id"])
    finalized = store.finalize_turn_execution(
        session_id=session_id,
        turn_id=turn_id,
        expected_window_revision=1,
        processing_level="L0",
        assistant_content="回答已经提交。",
        post_commit_job_kinds=("session_retrieval_index",),
    )
    job = finalized["post_commit_jobs"][0]  # type: ignore[index]
    assert job["next_retry_at"] is not None

    claimed = store.claim_due_turn_post_commit_jobs(
        session_id=session_id,
        worker_id="first-worker",
        lease_seconds=60,
    )
    applied = store.mark_turn_post_commit_job_applied(
        job_id=str(claimed[0]["job_id"]),
        worker_id="first-worker",
    )

    assert applied["status"] == "applied"
    assert applied["next_retry_at"] is None
    assert applied["lease_owner"] is None
    assert applied["lease_until"] is None
    assert applied["reason_code"] is None
    assert applied["completed_at"] is not None


def test_retryable_post_commit_job_clears_retry_schedule_when_later_applied(
    monkeypatch,
):
    clock = ["2026-09-03T06:00:00+00:00"]
    monkeypatch.setattr(store, "_now", lambda: clock[0])
    session_id = store.create_session("Entelecheia")
    accepted = _accept(session_id)
    turn_id = str(_turn(accepted)["turn_id"])
    store.finalize_turn_execution(
        session_id=session_id,
        turn_id=turn_id,
        expected_window_revision=1,
        processing_level="L0",
        assistant_content="回答已经提交。",
        post_commit_job_kinds=("session_retrieval_index",),
    )
    claimed = store.claim_due_turn_post_commit_jobs(
        session_id=session_id,
        worker_id="failing-worker",
        lease_seconds=60,
    )
    failed = store.mark_turn_post_commit_job_failed(
        job_id=str(claimed[0]["job_id"]),
        worker_id="failing-worker",
        reason_code="INDEX_ENCODER_TEMPORARILY_UNAVAILABLE",
        retry_after_seconds=1,
    )
    assert failed["status"] == "retryable_failed"
    assert failed["next_retry_at"] is not None

    clock[0] = "2026-09-03T06:00:02+00:00"
    retried = store.claim_due_turn_post_commit_jobs(
        session_id=session_id,
        worker_id="recovery-worker",
        lease_seconds=60,
    )
    applied = store.mark_turn_post_commit_job_applied(
        job_id=str(retried[0]["job_id"]),
        worker_id="recovery-worker",
    )

    assert applied["status"] == "applied"
    assert applied["attempts"] == 2
    assert applied["next_retry_at"] is None
    assert applied["reason_code"] is None
    assert applied["completed_at"] is not None
