"""Offline call-path diagnostic for the DocBench MPS post-commit crash."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from threading import Event, current_thread
from types import SimpleNamespace
import sys
import traceback

from evals.docbench.reproduce_or_run_script.retrieval_observability import (
    retrieval_runtime_snapshot,
)
from personagraph.retrieval.compute import devices
from personagraph.retrieval.operations.document_maintenance import (
    build_document_retrieval_composition,
)
from personagraph.retrieval.profile import DocumentRetrievalProfile
from personagraph.runtime.post_commit.scheduler import (
    schedule_turn_post_commit_jobs,
    stop_turn_post_commit_workers,
)
from personagraph.workspace.storage import DocumentDatabase
from personagraph.workspace.storage.context import bind


class _DeviceProbeBoundary(RuntimeError):
    """Stop both paths immediately before a device operation would run."""


@dataclass(frozen=True, slots=True)
class _ProbeTrace:
    thread_name: str
    device: str
    frames: tuple[tuple[str, str, int], ...]

    @property
    def function_names(self) -> frozenset[str]:
        return frozenset(function for _filename, function, _lineno in self.frames)


class _PostCommitPathStore:
    """Small post-commit port double; it exposes no model ToolCall surface."""

    def __init__(self) -> None:
        self.finished = Event()
        self.failed: dict[str, object] | None = None
        self.claimed = False

    @contextmanager
    def session_database_scope(self, session_id: str):
        assert session_id == "session-device-path"
        yield

    def claim_due_turn_post_commit_jobs(self, **kwargs: object):
        assert kwargs["session_id"] == "session-device-path"
        if self.claimed:
            return []
        self.claimed = True
        return [
            {
                "job_id": "job-session-index",
                "job_kind": "session_retrieval_index",
                "session_id": "session-device-path",
                "turn_id": "turn-device-path",
            }
        ]

    def get_committed_turn_pair_for_post_commit(self, **kwargs: object):
        assert kwargs == {
            "session_id": "session-device-path",
            "turn_id": "turn-device-path",
        }
        return {"turn_id": "turn-device-path"}

    def mark_turn_post_commit_job_failed(self, **kwargs: object):
        self.failed = dict(kwargs)
        self.finished.set()
        return {"status": "retryable_failed"}

    def inspect_turn_execution(self, session_id: str) -> dict[str, object]:
        assert session_id == "session-device-path"
        return {
            "window": {"window_state": "post_commit_pending"},
            "post_commit_jobs": [{"status": "retryable_failed" if self.failed else "pending"}],
        }


def test_docbench_final_snapshot_reuses_composition_while_post_commit_reaches_probe(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Keep final observation off the probe while tracing the post-commit path.

    The post-commit path starts from its durable ``session_retrieval_index`` job. It
    intentionally contains no ``retrieve_files`` model tool call: indexing a newly
    committed conversation pair is sufficient to reach the device selector.
    """

    monkeypatch.setenv("PERSONAGRAPH_RETRIEVAL_PROFILE", "bge_m3")
    monkeypatch.setenv("PERSONAGRAPH_RETRIEVAL_FAILURE_POLICY", "strict")
    monkeypatch.setenv("PERSONAGRAPH_RETRIEVAL_RERANKER", "bge_v2_m3")
    monkeypatch.setenv("PERSONAGRAPH_RETRIEVAL_DEVICE", "auto")
    monkeypatch.setenv("PERSONAGRAPH_RETRIEVAL_USE_FP16", "false")
    monkeypatch.setenv("PERSONAGRAPH_RETRIEVAL_LOCAL_FILES_ONLY", "true")
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace())
    monkeypatch.setattr(devices, "_available", lambda _torch, device: device == "mps")

    traces: list[_ProbeTrace] = []

    def stop_before_device_probe(_torch: object, device: str) -> None:
        frames = tuple(
            (frame.filename, frame.name, frame.lineno)
            for frame in traceback.extract_stack()[:-1]
        )
        traces.append(
            _ProbeTrace(
                thread_name=current_thread().name,
                device=device,
                frames=frames,
            )
        )
        raise _DeviceProbeBoundary("offline diagnostic stopped before device access")

    monkeypatch.setattr(devices, "_probe", stop_before_device_probe)

    project_root = tmp_path / "project"
    project_root.mkdir()
    database = DocumentDatabase(
        "project-device-path",
        project_root,
        tmp_path / "var/projects/project-device-path/documents.sqlite",
    )
    with bind(database):
        established = build_document_retrieval_composition(
            profile=replace(DocumentRetrievalProfile.lexical(), device="cpu")
        )
        post_commit_store = _PostCommitPathStore()
        schedule_turn_post_commit_jobs(
            session_id="session-device-path",
            store=post_commit_store,  # type: ignore[arg-type]
        )
        snapshot = retrieval_runtime_snapshot(
            preflight={"status": "ready", "degradation_reasons": []},
            composition=established,
        )

    assert post_commit_store.finished.wait(timeout=5.0)
    assert stop_turn_post_commit_workers(store=post_commit_store, timeout_seconds=5.0)
    assert post_commit_store.failed is not None
    assert snapshot["status"] == "degraded"
    assert len(traces) == 1

    background = traces[0]
    assert background.thread_name.startswith("turn-post-commit-")
    assert background.device == "mps"

    assert {
        "_run_scheduled_post_commit_jobs",
        "_process_due_turn_post_commit_jobs",
        "process_due_turn_post_commit_jobs",
        "process_session_retrieval_index_job",
        "_build_default_composition",
        "build_session_retrieval_composition",
        "build_document_retrieval_runtime",
        "preflight_document_retrieval_profile",
        "select_device",
    } <= background.function_names
    assert "retrieve_file_query_batch" not in background.function_names
    assert "retrieve_files" not in background.function_names
