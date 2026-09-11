from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from personagraph.model_io.gateway import ModelGatewayError
from personagraph.runtime.model_calls import session_summary as model_call
from personagraph.session.session_summary import (
    SessionSummaryGenerationError,
    SessionSummaryTurnPair,
)


def _pair() -> SessionSummaryTurnPair:
    return SessionSummaryTurnPair(
        turn_id="turn-summary-source",
        user_turn_idx=0,
        assistant_turn_idx=1,
        user_content="用户希望保留投影式上下文。",
        assistant_content="已确认使用每轮重组。",
        created_at="2026-08-27T00:00:00+00:00",
    )


def test_post_commit_scheduler_cold_import_does_not_load_handlers_or_model_adapter() -> None:
    project_root = Path(__file__).resolve().parents[3]
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import personagraph.runtime.post_commit.scheduler; "
            "assert 'personagraph.runtime.post_commit.contracts' not in sys.modules; "
            "assert 'personagraph.runtime.post_commit.runner' not in sys.modules; "
            "assert 'personagraph.retrieval' not in sys.modules; "
            "assert 'personagraph.session.session_summary_jobs' not in sys.modules; "
            "assert 'personagraph.trajectory' not in sys.modules; "
            "assert 'personagraph.runtime.model_calls.session_summary' not in sys.modules; "
            "assert 'personagraph.session.store' not in sys.modules; "
            "assert 'personagraph.runtime.model_calls.requests' not in sys.modules; "
            "assert 'personagraph.model_io.gateway' not in sys.modules; "
            "assert 'personagraph.configuration.app_settings' not in sys.modules",
        ],
        cwd=project_root,
        env={**os.environ, "PYTHONPATH": str(project_root / "src")},
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_default_mock_summary_does_not_enter_prepared_provider_path(monkeypatch):
    class MustNotReachProvider:
        def __call__(self, *_args, **_kwargs):
            raise AssertionError("mock summary must not dispatch a provider")

        def prepare(self, *_args, **_kwargs):
            raise AssertionError("mock summary must not prepare a provider request")

    monkeypatch.setattr(
        model_call,
        "get_setting",
        lambda name, default=None: "mock" if name == "provider" else default,
    )
    monkeypatch.setattr(
        model_call,
        "anthropic_compatible_chat",
        MustNotReachProvider(),
    )
    events = []

    summary = model_call.generate_session_summary(
        None,
        (_pair(),),
        "turn-summary-job",
        "job-summary",
        events.append,
    )

    assert "投影式上下文" in summary
    assert events


def test_default_summary_trace_emitter_is_a_noop(monkeypatch):
    observed: dict[str, object] = {}

    def capture_request(**kwargs):
        observed["emit"] = kwargs["emit"]
        return SimpleNamespace(value="bounded summary")

    monkeypatch.setattr(model_call, "request_model_with_retry", capture_request)

    summary = model_call.generate_session_summary(
        None,
        (_pair(),),
        "turn-summary-job",
        "job-summary",
    )

    assert summary == "bounded summary"
    emit = observed["emit"]
    assert emit is model_call._discard_summary_trace
    assert callable(emit)
    assert emit(object()) is None


@pytest.mark.parametrize(
    ("gateway_code", "retryable", "summary_code"),
    (
        ("MODEL_CALL_TIMEOUT", True, "SUMMARY_MODEL_TIMEOUT"),
        ("MODEL_BAD_RESPONSE", True, "SUMMARY_MODEL_OUTPUT_INVALID"),
        ("MODEL_TRANSPORT_FAILURE", True, "SUMMARY_MODEL_TRANSPORT_FAILURE"),
        ("MODEL_CONFIGURATION_FAILURE", False, "SUMMARY_MODEL_CONFIGURATION_FAILURE"),
    ),
)
def test_model_failures_cross_into_session_as_stable_summary_errors(
    monkeypatch,
    gateway_code: str,
    retryable: bool,
    summary_code: str,
) -> None:
    monkeypatch.setattr(
        model_call,
        "request_model_with_retry",
        lambda **_kwargs: (_ for _ in ()).throw(
            ModelGatewayError(gateway_code, "private provider detail", retryable=retryable)
        ),
    )

    with pytest.raises(SessionSummaryGenerationError) as caught:
        model_call.generate_session_summary(
            None,
            (_pair(),),
            "turn-summary-job",
            "job-summary",
        )

    assert caught.value.code == summary_code
    assert "private provider detail" not in str(caught.value)
