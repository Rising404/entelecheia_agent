"""Durable Turn post-commit Runtime kernel tests."""

from __future__ import annotations

from contextlib import contextmanager
from threading import Event

import pytest

from personagraph.context_budget import ContextBudgetExceeded
from personagraph.runtime.post_commit import scheduler as post_commit_scheduler
from personagraph.runtime.post_commit.runner import (
    process_due_turn_post_commit_jobs,
    release_turn_window_if_post_commit_settled,
)
from personagraph.session import session_summary, store
from personagraph.session.session_summary import (
    SessionSummaryGenerationError,
    SessionSummaryStatus,
)
from personagraph.trajectory import TrajectoryStore
from personagraph.trajectory import recorder as trajectory_recorder
from personagraph.trajectory.scope import current_turn_linkage
from tests.helpers.session_records import (
    complete_test_turn_execution as _complete_turn,
    completed_test_turn_id as _turn_id,
)


def test_scheduled_worker_uses_the_injected_session_database_scope(monkeypatch):
    observed: list[str] = []

    class ScopedStore:
        @contextmanager
        def session_database_scope(self, session_id: str):
            observed.append(f"enter:{session_id}")
            try:
                yield
            finally:
                observed.append(f"exit:{session_id}")

    monkeypatch.setattr(
        post_commit_scheduler,
        "_process_due_turn_post_commit_jobs",
        lambda *, session_id, store: observed.append(f"process:{session_id}"),
    )

    post_commit_scheduler._run_scheduled_post_commit_jobs(
        session_id="session-scoped-worker",
        store=ScopedStore(),  # type: ignore[arg-type]
    )

    assert observed == [
        "enter:session-scoped-worker",
        "process:session-scoped-worker",
        "exit:session-scoped-worker",
    ]


def test_runner_claims_each_job_only_when_ready_to_execute(monkeypatch):
    session_id = store.create_session("Entelecheia")
    _complete_turn(
        session_id, 1,
        post_commit_job_kinds=("session_summary", "session_retrieval_index"),
    )
    original_claim = store.claim_due_turn_post_commit_jobs
    claim_limits = []

    def claim(**kwargs):
        claim_limits.append(kwargs["limit"])
        return original_claim(**kwargs)

    def index(_pair):
        jobs = store.inspect_turn_execution(session_id)["post_commit_jobs"]
        assert sum(job["status"] == "processing" for job in jobs) == 1
        return "test-generation"

    monkeypatch.setattr(store, "claim_due_turn_post_commit_jobs", claim)
    result = process_due_turn_post_commit_jobs(
        session_id=session_id, store=store, session_retrieval_indexer=index,
    )
    assert result.claimed_job_count == 2
    assert result.applied_job_count == 2
    assert result.released_window is True
    assert set(claim_limits) == {1}


def test_scheduler_does_not_claim_next_job_after_shutdown(monkeypatch):
    from personagraph.runtime.post_commit import runner

    session_id = store.create_session("Entelecheia")
    _complete_turn(
        session_id, 1,
        post_commit_job_kinds=("session_retrieval_index", "session_summary"),
    )
    started, release = Event(), Event()
    original_index = runner.process_session_retrieval_index_job
    original_summary = runner.process_session_summary_job
    executed = []

    def before_execute(kind):
        executed.append(kind)
        if len(executed) == 1:
            started.set()
            assert release.wait(2)

    def process_index(**kwargs):
        before_execute("index")
        return original_index(**{**kwargs, "index_pair": lambda pair: "test-generation"})

    def process_summary(**kwargs):
        before_execute("summary")
        return original_summary(**kwargs)

    monkeypatch.setattr(runner, "process_session_retrieval_index_job", process_index)
    monkeypatch.setattr(runner, "process_session_summary_job", process_summary)
    post_commit_scheduler.schedule_turn_post_commit_jobs(session_id=session_id, store=store)
    try:
        assert started.wait(2)
        assert post_commit_scheduler.stop_turn_post_commit_workers(
            store=store, timeout_seconds=0,
        ) is False
    finally:
        release.set()
    assert post_commit_scheduler.stop_turn_post_commit_workers(
        store=store, timeout_seconds=2,
    ) is True
    jobs = store.inspect_turn_execution(session_id)["post_commit_jobs"]
    assert len(executed) == 1
    assert sorted(job["status"] for job in jobs) == ["applied", "pending"]


@pytest.mark.parametrize(
    ("jobs", "revision", "expected_release"),
    [
        ([{"status": "applied"}], 7, True),
        ([{"status": "waived"}], 7, True),
        ([{"status": "pending"}], 7, False),
        ([], 7, False),
        ([{"status": "applied"}], "7", False),
    ],
)
def test_window_release_requires_a_fully_settled_well_formed_inspection(
    jobs,
    revision,
    expected_release,
) -> None:
    releases: list[dict[str, object]] = []

    class Store:
        def inspect_turn_execution(self, session_id: str) -> dict[str, object]:
            assert session_id == "session-release-contract"
            return {
                "window": {
                    "window_state": "post_commit_pending",
                    "turn_id": "turn-release-contract",
                    "state_version": revision,
                },
                "post_commit_jobs": jobs,
            }

        def release_turn_execution_window(self, **kwargs: object) -> dict[str, object]:
            releases.append(dict(kwargs))
            return {}

    released = release_turn_window_if_post_commit_settled(
        session_id="session-release-contract",
        store=Store(),  # type: ignore[arg-type]
    )

    assert released is expected_release
    assert releases == (
        [
            {
                "session_id": "session-release-contract",
                "turn_id": "turn-release-contract",
                "expected_window_revision": 7,
            }
        ]
        if expected_release
        else []
    )


def test_summary_job_keeps_five_hot_pairs_and_releases_the_window():
    session_id = store.create_session("Entelecheia")
    completed = [_complete_turn(session_id, ordinal) for ordinal in range(1, 6)]
    summary_turn = _complete_turn(
        session_id,
        6,
        post_commit_job_kinds=("session_summary",),
    )
    generated_from: list[tuple[str | None, ...]] = []

    def generate(_state, pairs, _turn_id, _job_id):
        generated_from.append(tuple(pair.turn_id for pair in pairs))
        return "第一轮已纳入长期摘要。"

    result = process_due_turn_post_commit_jobs(
        session_id=session_id,
        store=store,
        worker_id="summary-worker",
        summary_generator=generate,
    )

    assert result.claimed_job_count == 1
    assert result.applied_job_count == 1
    assert result.failed_job_count == 0
    assert result.released_window is True
    assert generated_from == [(_turn_id(completed[0]),)]
    state = store.get_session_summary_state(session_id)
    assert state is not None
    assert state.status is SessionSummaryStatus.OK
    assert state.summarized_through_turn_id == _turn_id(completed[0])
    assert store.get_turn_execution_window(session_id)["window_state"] == "empty"  # type: ignore[index]
    assert store.list_turn_post_commit_jobs(_turn_id(summary_turn))[0]["status"] == "applied"


def test_summary_job_trajectory_is_bound_to_its_claimed_turn_without_leaking(
    tmp_path,
):
    session_id = store.create_session("Entelecheia")
    for ordinal in range(1, 6):
        _complete_turn(session_id, ordinal)
    summary_turn = _complete_turn(
        session_id,
        6,
        post_commit_job_kinds=("session_summary",),
    )
    summary_turn_id = _turn_id(summary_turn)
    trajectory = TrajectoryStore(tmp_path / "trajectory.sqlite")
    observed = []

    def generate(_state, _pairs, _turn_id, _job_id):
        observed.append(current_turn_linkage())
        trajectory_recorder.record_rejected_output(
            stage="summary-probe",
            rejected="safe projection",
            reason_code="summary_probe",
            step_id="summary-probe-step",
            store=trajectory,
        )
        return "已生成摘要。"

    result = process_due_turn_post_commit_jobs(
        session_id=session_id,
        store=store,
        worker_id="summary-worker",
        summary_generator=generate,
    )

    assert result.applied_job_count == 1
    assert observed[0] is not None
    assert (observed[0].session_id, observed[0].turn_id) == (
        session_id,
        summary_turn_id,
    )
    summary_step = trajectory.get("summary-probe-step")
    assert summary_step is not None
    assert (summary_step.session_id, summary_step.turn_id) == (
        session_id,
        summary_turn_id,
    )

    trajectory_recorder.record_rejected_output(
        stage="unrelated-background-probe",
        rejected="safe projection",
        reason_code="background_probe",
        step_id="background-probe-step",
        store=trajectory,
    )
    background_step = trajectory.get("background-probe-step")
    assert background_step is not None
    assert background_step.session_id is None
    assert background_step.turn_id is None


def test_summary_job_without_eligible_pairs_settles_an_empty_summary():
    session_id = store.create_session("Entelecheia")
    summary_turn = _complete_turn(
        session_id,
        1,
        post_commit_job_kinds=("session_summary",),
    )

    def should_not_generate(*_args):
        raise AssertionError("no eligible pair should invoke a summary model")

    result = process_due_turn_post_commit_jobs(
        session_id=session_id,
        store=store,
        worker_id="summary-worker",
        summary_generator=should_not_generate,
    )

    assert result.applied_job_count == 1
    assert result.released_window is True
    state = store.get_session_summary_state(session_id)
    assert state is not None
    assert state.status is SessionSummaryStatus.OK
    assert state.running_summary == ""
    assert state.summarized_through_turn_id is None
    assert store.list_turn_post_commit_jobs(_turn_id(summary_turn))[0]["status"] == "applied"


def test_summary_failure_keeps_window_and_records_only_a_safe_failure_category():
    session_id = store.create_session("Entelecheia")
    for ordinal in range(1, 6):
        _complete_turn(session_id, ordinal)
    summary_turn = _complete_turn(
        session_id,
        6,
        post_commit_job_kinds=("session_summary",),
    )

    def provider_down(*_args):
        raise SessionSummaryGenerationError(
            "SUMMARY_MODEL_TIMEOUT",
            "private provider detail",
            retryable=False,
        )

    result = process_due_turn_post_commit_jobs(
        session_id=session_id,
        store=store,
        worker_id="summary-worker",
        summary_generator=provider_down,
    )

    assert result.applied_job_count == 0
    assert result.failed_job_count == 1
    assert result.released_window is False
    state = store.get_session_summary_state(session_id)
    assert state is not None
    assert state.status is SessionSummaryStatus.UNAVAILABLE
    assert state.last_error_code == "SUMMARY_MODEL_TIMEOUT"
    assert "private provider detail" not in state.last_error_code
    inspection = store.inspect_turn_execution(session_id)
    assert inspection["window"]["window_state"] == "post_commit_pending"  # type: ignore[index]
    assert store.list_turn_post_commit_jobs(_turn_id(summary_turn))[0]["status"] == "terminal_failed"


def test_summary_context_budget_failure_uses_stable_terminal_category():
    session_id = store.create_session("Entelecheia")
    for ordinal in range(1, 6):
        _complete_turn(session_id, ordinal)
    summary_turn = _complete_turn(
        session_id,
        6,
        post_commit_job_kinds=("session_summary",),
    )

    def over_budget(*_args):
        raise ContextBudgetExceeded(limit=1_000, estimated_tokens=1_001)

    result = process_due_turn_post_commit_jobs(
        session_id=session_id,
        store=store,
        worker_id="summary-worker",
        summary_generator=over_budget,
    )

    assert result.applied_job_count == 0
    assert result.failed_job_count == 1
    assert result.released_window is False
    state = store.get_session_summary_state(session_id)
    assert state is not None
    assert state.last_error_code == "SUMMARY_CONTEXT_BUDGET_EXCEEDED"
    assert store.list_turn_post_commit_jobs(_turn_id(summary_turn))[0]["status"] == (
        "terminal_failed"
    )


def test_degraded_summary_without_fresh_sources_is_never_recognized_as_ok():
    session_id = store.create_session("Entelecheia")
    summary_turn = _complete_turn(
        session_id,
        1,
        post_commit_job_kinds=("session_summary",),
    )
    job = summary_turn["finalized"]["post_commit_jobs"][0]  # type: ignore[index]
    store.claim_due_turn_post_commit_jobs(
        session_id=session_id,
        worker_id="first-worker",
        lease_seconds=60,
    )
    store.fail_session_summary_post_commit_job(
        session_id=session_id,
        job_id=str(job["job_id"]),
        worker_id="first-worker",
        expected_state_version=0,
        expected_summarized_through_turn_id=None,
        status=SessionSummaryStatus.UNAVAILABLE,
        reason_code="SUMMARY_MODEL_TIMEOUT",
        retry_after_seconds=None,
    )
    inspection = store.inspect_turn_execution(session_id)
    old_window = inspection["window"]
    assert isinstance(old_window, dict)
    failed_digest = inspection["failed_job_digest"]
    assert isinstance(failed_digest, str)
    waived = store.apply_turn_post_commit_job_control(
        session_id=session_id,
        turn_id=_turn_id(summary_turn),
        expected_window_revision=int(old_window["state_version"]),
        request_id="waive-summary-failure",
        action="waive",
        job_ids=(str(job["job_id"]),),
        expected_failed_job_digest=failed_digest,
        actor="test",
    )
    store.release_turn_execution_window(
        session_id=session_id,
        turn_id=_turn_id(summary_turn),
        expected_window_revision=int(waived["window"]["state_version"]),  # type: ignore[index]
    )
    next_summary_turn = _complete_turn(
        session_id,
        2,
        post_commit_job_kinds=("session_summary",),
    )

    result = process_due_turn_post_commit_jobs(
        session_id=session_id,
        store=store,
        worker_id="second-worker",
    )

    assert result.failed_job_count == 1
    state = store.get_session_summary_state(session_id)
    assert state is not None
    assert state.status is SessionSummaryStatus.UNAVAILABLE
    assert state.last_error_code == "SUMMARY_NO_FRESH_SOURCE_FOR_STALE_STATE"
    assert store.list_turn_post_commit_jobs(_turn_id(next_summary_turn))[0]["status"] == "terminal_failed"


def test_summary_input_budget_failure_is_persisted_without_calling_the_model(monkeypatch):
    session_id = store.create_session("Entelecheia")
    for ordinal in range(1, 6):
        _complete_turn(session_id, ordinal)
    summary_turn = _complete_turn(
        session_id,
        6,
        post_commit_job_kinds=("session_summary",),
    )
    monkeypatch.setattr(session_summary, "task_budget", lambda: 1)

    def should_not_generate(*_args):
        raise AssertionError("an over-budget input must fail before the model call")

    result = process_due_turn_post_commit_jobs(
        session_id=session_id,
        store=store,
        worker_id="summary-worker",
        summary_generator=should_not_generate,
    )

    assert result.applied_job_count == 0
    assert result.failed_job_count == 1
    assert result.released_window is False
    state = store.get_session_summary_state(session_id)
    assert state is not None
    assert state.status is SessionSummaryStatus.UNAVAILABLE
    assert state.last_error_code == "SUMMARY_INPUT_OVER_BUDGET"
    assert store.list_turn_post_commit_jobs(_turn_id(summary_turn))[0]["status"] == "terminal_failed"


def test_retired_memory_job_has_no_runtime_handler():
    session_id = store.create_session("Entelecheia")
    completed = _complete_turn(
        session_id,
        1,
        post_commit_job_kinds=("memory_consolidation",),
    )

    result = process_due_turn_post_commit_jobs(
        session_id=session_id,
        store=store,
        worker_id="retired-job-worker",
    )

    job = store.list_turn_post_commit_jobs(_turn_id(completed))[0]
    assert result.failed_job_count == 1
    assert result.released_window is False
    assert job["status"] == "terminal_failed"
    assert job["reason_code"] == "POST_COMMIT_HANDLER_UNAVAILABLE"
