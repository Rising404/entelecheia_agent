from __future__ import annotations

import pytest

from personagraph.session import store
from personagraph.session.session_summary import (
    SessionSummaryProgressError,
    SessionSummaryStateConflict,
    SessionSummaryStatus,
    SessionSummaryUpdate,
)


def _complete_turn(
    session_id: str,
    ordinal: int,
    *,
    post_commit_job_kinds: tuple[str, ...] = (),
) -> dict[str, object]:
    accepted = store.accept_turn_execution(
        session_id=session_id,
        client_request_id=f"summary-request-{ordinal}",
        source="summary-storage-test",
        user_text=f"用户消息 {ordinal}",
        lease_owner="summary-test-host",
    )
    turn = accepted["turn"]  # type: ignore[assignment]
    window = accepted["window"]  # type: ignore[assignment]
    finalized = store.finalize_turn_execution(
        session_id=session_id,
        turn_id=str(turn["turn_id"]),  # type: ignore[index]
        expected_window_revision=int(window["state_version"]),  # type: ignore[index]
        processing_level="L0",
        assistant_content=f"助手回复 {ordinal}",
        post_commit_job_kinds=post_commit_job_kinds,
    )
    if not post_commit_job_kinds:
        finalized_window = finalized["window"]  # type: ignore[assignment]
        store.release_turn_execution_window(
            session_id=session_id,
            turn_id=str(turn["turn_id"]),  # type: ignore[index]
            expected_window_revision=int(finalized_window["state_version"]),  # type: ignore[index]
        )
    return {"accepted": accepted, "finalized": finalized}


def _turn_id(result: dict[str, object]) -> str:
    accepted = result["accepted"]  # type: ignore[assignment]
    return str(accepted["turn"]["turn_id"])  # type: ignore[index]


def _summary_job(result: dict[str, object]) -> dict[str, object]:
    finalized = result["finalized"]  # type: ignore[assignment]
    return finalized["post_commit_jobs"][0]  # type: ignore[index]


def _seed_summary_job(session_id: str) -> tuple[list[dict[str, object]], dict[str, object]]:
    completed = [_complete_turn(session_id, ordinal) for ordinal in range(1, 6)]
    summary_turn = _complete_turn(
        session_id,
        6,
        post_commit_job_kinds=("session_summary",),
    )
    return completed, summary_turn


def test_summary_candidates_exclude_hot_pairs_and_progress_atomically():
    session_id = store.create_session("Entelecheia")
    completed, summary_turn = _seed_summary_job(session_id)
    job = _summary_job(summary_turn)
    claimed = store.claim_due_turn_post_commit_jobs(
        session_id=session_id,
        worker_id="summary-worker",
        lease_seconds=60,
    )
    assert [item["job_id"] for item in claimed] == [job["job_id"]]

    candidates = store.list_committed_turn_pairs_for_summary(
        session_id,
        after_turn_id=None,
        retain_recent_pairs=5,
        limit=8,
    )
    assert [pair.turn_id for pair in candidates] == [_turn_id(completed[0])]
    assert _turn_id(summary_turn) not in {pair.turn_id for pair in candidates}

    state = store.commit_session_summary_post_commit_job(
        session_id=session_id,
        job_id=str(job["job_id"]),
        worker_id="summary-worker",
        expected_state_version=0,
        expected_summarized_through_turn_id=None,
        update=SessionSummaryUpdate(
            running_summary="已压缩第一轮的会话事实。",
            summarized_through_turn_id=_turn_id(completed[0]),
        ),
    )

    assert state.status is SessionSummaryStatus.OK
    assert state.state_version == 1
    assert state.summarized_through_turn_id == _turn_id(completed[0])
    assert state.last_error_code is None
    assert store.list_turn_post_commit_jobs(_turn_id(summary_turn))[0]["status"] == "applied"

    replayed = store.commit_session_summary_post_commit_job(
        session_id=session_id,
        job_id=str(job["job_id"]),
        worker_id="different-worker-after-restart",
        expected_state_version=0,
        expected_summarized_through_turn_id=None,
        update=SessionSummaryUpdate(
            running_summary="已压缩第一轮的会话事实。",
            summarized_through_turn_id=_turn_id(completed[0]),
        ),
    )
    assert replayed == state


def test_summary_candidate_reader_is_bounded_and_keeps_current_hot_history():
    session_id = store.create_session("Entelecheia")
    completed = [_complete_turn(session_id, ordinal) for ordinal in range(1, 7)]
    summary_turn = _complete_turn(
        session_id,
        7,
        post_commit_job_kinds=("session_summary",),
    )

    first_page = store.list_committed_turn_pairs_for_summary(
        session_id,
        after_turn_id=None,
        limit=1,
    )
    assert [pair.turn_id for pair in first_page] == [_turn_id(completed[0])]

    remaining_page = store.list_committed_turn_pairs_for_summary(
        session_id,
        after_turn_id=_turn_id(completed[0]),
        limit=8,
    )
    assert [pair.turn_id for pair in remaining_page] == [_turn_id(completed[1])]
    assert _turn_id(summary_turn) not in {pair.turn_id for pair in remaining_page}

    with pytest.raises(ValueError, match="protected recent"):
        store.list_committed_turn_pairs_for_summary(
            session_id,
            after_turn_id=None,
            retain_recent_pairs=4,
            limit=1,
        )


def test_summary_commit_rejects_stale_cas_and_hot_boundary_without_settling_job():
    session_id = store.create_session("Entelecheia")
    completed, summary_turn = _seed_summary_job(session_id)
    job = _summary_job(summary_turn)
    store.claim_due_turn_post_commit_jobs(
        session_id=session_id,
        worker_id="summary-worker",
        lease_seconds=60,
    )

    with pytest.raises(SessionSummaryStateConflict):
        store.commit_session_summary_post_commit_job(
            session_id=session_id,
            job_id=str(job["job_id"]),
            worker_id="summary-worker",
            expected_state_version=1,
            expected_summarized_through_turn_id=None,
            update=SessionSummaryUpdate(
                running_summary="不应写入。",
                summarized_through_turn_id=_turn_id(completed[0]),
            ),
        )
    assert store.get_session_summary_state(session_id) is None
    assert store.list_turn_post_commit_jobs(_turn_id(summary_turn))[0]["status"] == "processing"

    with pytest.raises(SessionSummaryProgressError, match="protected recent"):
        store.commit_session_summary_post_commit_job(
            session_id=session_id,
            job_id=str(job["job_id"]),
            worker_id="summary-worker",
            expected_state_version=0,
            expected_summarized_through_turn_id=None,
            update=SessionSummaryUpdate(
                running_summary="不应覆盖当前热窗口。",
                summarized_through_turn_id=_turn_id(completed[1]),
            ),
        )
    assert store.get_session_summary_state(session_id) is None
    assert store.list_turn_post_commit_jobs(_turn_id(summary_turn))[0]["status"] == "processing"


def test_summary_failure_persists_honest_status_with_job_failure_in_one_transaction():
    session_id = store.create_session("Entelecheia")
    _, summary_turn = _seed_summary_job(session_id)
    job = _summary_job(summary_turn)
    store.claim_due_turn_post_commit_jobs(
        session_id=session_id,
        worker_id="summary-worker",
        lease_seconds=60,
    )

    state = store.fail_session_summary_post_commit_job(
        session_id=session_id,
        job_id=str(job["job_id"]),
        worker_id="summary-worker",
        expected_state_version=0,
        expected_summarized_through_turn_id=None,
        status=SessionSummaryStatus.UNAVAILABLE,
        reason_code="SUMMARY_PROVIDER_UNAVAILABLE",
        retry_after_seconds=None,
    )

    assert state.status is SessionSummaryStatus.UNAVAILABLE
    assert state.state_version == 1
    assert state.last_error_code == "SUMMARY_PROVIDER_UNAVAILABLE"
    assert store.list_turn_post_commit_jobs(_turn_id(summary_turn))[0]["status"] == "terminal_failed"
    finalized = summary_turn["finalized"]  # type: ignore[assignment]
    with pytest.raises(store.TurnPostCommitJobsPending):
        store.release_turn_execution_window(
            session_id=session_id,
            turn_id=_turn_id(summary_turn),
            expected_window_revision=int(finalized["window"]["state_version"]),  # type: ignore[index]
        )
