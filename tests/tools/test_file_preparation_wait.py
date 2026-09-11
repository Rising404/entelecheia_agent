"""批量准备共用等待预算，不改变只读检查、图片或冻结来源语义。"""

from types import SimpleNamespace
import time
import pytest

from personagraph.tools.execution_context import ToolExecutionContext, tool_execution_scope

from personagraph.tools.files import file_adapter
from personagraph.workspace.files.access import FileAccessError
from personagraph.workspace.ingestion.contracts import FilePreparationResult, FilePreparationStatus


def test_prepare_uses_turn_deadline_and_preserves_call_request_identity(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(file_adapter, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    calls = []

    def prepare(source, *, pending_wait_seconds=0, request_id=None, checkpoint=None):
        calls.append((pending_wait_seconds, request_id))
        if not pending_wait_seconds:
            return FilePreparationResult(status=FilePreparationStatus.PENDING, operation_id="job")
        assert checkpoint is not None
        checkpoint()
        clock[0] += 191
        return FilePreparationResult(status=FilePreparationStatus.READY, operation_id="job")

    for _ in range(2):
        control = ToolExecutionContext(deadline_monotonic=1300, clock=lambda: clock[0],
                                       logical_tool_call_id="same-durable-call")
        with tool_execution_scope(control):
            result = _runtime(prepare).prepare_files({"files": [{"path": "report.pdf"}]})
        assert result["ready_indices"] == [0]
    assert calls[1][0] == 1200
    assert all(identity and identity == calls[0][1] for _, identity in calls)


def test_prepare_registration_has_no_shorter_deadline_than_the_turn():
    from personagraph.tools.files.file_tools import build_file_state_registration, PREPARE_FILES_TOOL_ID

    profile = build_file_state_registration(
        tool_id=PREPARE_FILES_TOOL_ID, handler=lambda _: {}, effect_scope="test",
    ).execution_profile
    assert profile.default_timeout_s is None
    assert profile.hard_timeout_s is None


def _source(path):
    return SimpleNamespace(
        relative_path=path, file_name=path, file_id=None, file_version_id=None,
    )


def _runtime(prepare, check=None):
    return file_adapter.FileStateRuntime(
        access=SimpleNamespace(resolve_path=_source), effect_scope="test-scope",
        check_state=check or prepare, prepare=prepare,
    )


def test_batch_submits_before_waiting_and_shares_the_remaining_turn_budget(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(file_adapter, "time", SimpleNamespace(monotonic=lambda: clock[0]), raising=False)
    submitted, waits = [], []

    def prepare(source, *, pending_wait_seconds=None, checkpoint=None):
        if pending_wait_seconds is None:
            submitted.append(source)
            clock[0] += 10.0
            return FilePreparationResult(status=FilePreparationStatus.PENDING, operation_id=source.file_name)
        assert len(submitted) == 3
        assert source is submitted[len(waits)]
        waits.append(pending_wait_seconds)
        clock[0] += min(25.0 if source.file_name == "a.md" else 90.0, pending_wait_seconds)
        return FilePreparationResult(
            status=FilePreparationStatus.READY,
            operation_id=source.file_name, replayed=True,
        )

    with tool_execution_scope(ToolExecutionContext(deadline_monotonic=1300, clock=lambda: clock[0])):
        result = _runtime(prepare).prepare_files({"files": [{"path": f"{name}.md"} for name in "abc"]})

    assert waits == [1170.0, 1145.0, 1055.0]
    assert result["ready_indices"] == [0, 1, 2]
    assert result["not_ready_indices"] == []
    assert all(item["reused"] is False for item in result["results"])


def test_check_and_non_job_results_never_wait():
    states = [
        FilePreparationResult(status=FilePreparationStatus.PENDING, reason_code="image_visual_ready"),
        FilePreparationResult(status=FilePreparationStatus.READY),
        FilePreparationResult(status=FilePreparationStatus.BLOCKED),
    ]
    calls = []

    def prepare(source):
        calls.append(source)
        return states[int(source.file_name)]

    runtime = _runtime(prepare, check=lambda _: FilePreparationResult(
        status=FilePreparationStatus.PENDING, operation_id="already-queued",
    ))
    payload = {"files": [{"path": str(index)} for index in range(3)]}
    result = runtime.prepare_files(payload)
    assert [item["status"] for item in result["results"]] == ["partial", "ready", "unavailable"]
    runtime.check_files_state(payload)
    assert len(calls) == 3


def test_source_change_during_wait_is_reported_per_item():
    def prepare(source, *, pending_wait_seconds=None, checkpoint=None):
        if pending_wait_seconds is None:
            return FilePreparationResult(status=FilePreparationStatus.PENDING, operation_id=source.file_name)
        if source.file_name == "changed.md":
            raise FileAccessError("file_content_changed")
        return FilePreparationResult(status=FilePreparationStatus.READY, operation_id=source.file_name)

    with tool_execution_scope(ToolExecutionContext(deadline_monotonic=time.monotonic() + 30)):
        result = _runtime(prepare).prepare_files({"files": [{"path": "changed.md"}, {"path": "ready.md"}]})
    assert result["ready_indices"] == [1]
    assert result["results"][0]["status"] == "changed"
    assert result["results"][0]["reason_code"] == "file_content_changed"


def test_wait_needs_a_host_deadline_and_does_not_publish_pending_as_ready():
    from personagraph.tools.execution import ToolBusinessFailure

    with pytest.raises(ToolBusinessFailure) as error:
        _runtime(lambda _: FilePreparationResult(
            status=FilePreparationStatus.PENDING, operation_id="job",
        )).prepare_files({"files": [{"path": "report.pdf"}]})
    assert error.value.error.code == "file_preparation_deadline_required"


@pytest.mark.parametrize("stop_reason", ["deadline", "cancel", "lost_authority"])
def test_wait_observes_deadline_cancellation_and_lost_turn_authority(monkeypatch, stop_reason):
    from personagraph.tools.execution_context import ToolInvocationCancelled

    clock = [0.0]
    current = [True]
    control = ToolExecutionContext(
        deadline_monotonic=10, clock=lambda: clock[0],
        continuation_check=lambda: current[0],
    )
    monkeypatch.setattr(file_adapter, "time", SimpleNamespace(monotonic=lambda: clock[0]))

    def prepare(source, *, pending_wait_seconds=0, checkpoint=None):
        if pending_wait_seconds:
            if stop_reason == "deadline":
                clock[0] = 10
            elif stop_reason == "cancel":
                control.cancel("execution_cancelled")
            else:
                clock[0] = 2
                current[0] = False
            checkpoint()
            pytest.fail("A stopped Turn must not receive ready")
        return FilePreparationResult(status=FilePreparationStatus.PENDING, operation_id="job")

    with tool_execution_scope(control), pytest.raises(ToolInvocationCancelled):
        _runtime(prepare).prepare_files({"files": [{"path": "report.pdf"}]})
    assert control.snapshot()["cancellation_acknowledged"] is True
