"""提交后收尾报告不伪造成功，也不依赖真实 Provider。"""

from types import SimpleNamespace

import pytest

from evals.docbench.reproduce_or_run_script.post_commit import settle_post_commit
from personagraph.runtime.post_commit import scheduler


def test_missing_turn_is_unavailable_without_dispatch(monkeypatch):
    monkeypatch.setattr(
        scheduler, "wait_for_turn_post_commit_jobs",
        lambda **_: pytest.fail("no Turn identity may be guessed"),
        raising=False,
    )
    result = settle_post_commit(session_id="session", turn_id=None, store=object())
    assert result["post_commit"]["status"] == "unavailable"
    assert result["post_commit_complete"] is False


@pytest.mark.parametrize("released,timed_out", [(False, False), (True, True)])
def test_settled_label_alone_cannot_imply_completion(monkeypatch, released, timed_out):
    snapshot = {
        "status": "settled", "turn_id": "turn", "jobs": [],
        "window_released": released, "timed_out": timed_out,
    }
    monkeypatch.setattr(
        scheduler, "wait_for_turn_post_commit_jobs",
        lambda **_: SimpleNamespace(to_dict=lambda: snapshot), raising=False,
    )
    result = settle_post_commit(session_id="session", turn_id="turn", store=object())
    assert result["post_commit"] == snapshot
    assert result["post_commit_complete"] is False


def test_unavailable_wait_does_not_leak_exception_body(monkeypatch):
    def fail(**_):
        raise RuntimeError("provider-secret-not-for-report")

    monkeypatch.setattr(scheduler, "wait_for_turn_post_commit_jobs", fail, raising=False)
    result = settle_post_commit(session_id="session", turn_id="turn", store=object())
    assert result["post_commit"]["status"] == "unavailable"
    assert result["post_commit"]["exception_type"] == "RuntimeError"
    assert "provider-secret-not-for-report" not in str(result)
    assert result["post_commit_complete"] is False


@pytest.mark.parametrize("jobs", [[], [{"job_kind": "session_retrieval_index", "status": "waived"}]])
def test_released_window_with_skipped_or_missing_jobs_is_not_index_success(monkeypatch, jobs):
    snapshot = {
        "status": "settled", "turn_id": "turn", "jobs": jobs,
        "window_released": True, "timed_out": False,
    }
    monkeypatch.setattr(
        scheduler, "wait_for_turn_post_commit_jobs",
        lambda **_: SimpleNamespace(to_dict=lambda: snapshot), raising=False,
    )
    result = settle_post_commit(session_id="session", turn_id="turn", store=object())
    assert result["post_commit"] == snapshot
    assert result["post_commit_complete"] is False
