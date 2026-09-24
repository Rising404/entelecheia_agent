"""真实后台 owner 的 lease 等待与无下一次视觉工具调用的发布确认。"""

import threading
import time

import pytest
from PIL import Image

from personagraph.retrieval.lifecycle.sync import RetrievalSyncService
from personagraph.runtime.model_calls.vision import SqliteMountedVisualCallLedger
from personagraph.tools.execution_context import (
    ToolExecutionContext, ToolInvocationCancelled, tool_execution_scope,
)
from personagraph.tools.visual.publication_recovery import build_visual_publication_recovery
from personagraph.tools.workspace.session_read_source import build_session_workspace_readonly_runtime
from personagraph.workspace.files import FileSource, WorkspaceFileAuthority
from personagraph.workspace.ingestion.composition import build_document_maintenance_lifecycle
from personagraph.workspace.storage.context import require_current
from tests.tools.test_path_visual_question_publication import _VisualProvider


@pytest.mark.parametrize("caller_times_out", [False, True])
def test_background_owner_finishes_and_acknowledges_without_another_visual_call(
    tmp_path, bound_partitioned_session, monkeypatch, caller_times_out,
):
    root = tmp_path / "workspace"
    root.mkdir()
    target = root / "source.png"
    Image.new("RGB", (48, 32), "white").save(target)
    session_id = bound_partitioned_session(working_dir=root)
    database = require_current()
    WorkspaceFileAuthority(database).register_path(
        target.name, source=FileSource.USER_UPLOAD, media_type="image/png",
    )
    provider = _VisualProvider()
    monkeypatch.setenv("PERSONAGRAPH_RETRIEVAL_PROFILE", "lexical")
    monkeypatch.setattr(
        "personagraph.tools.workspace.session_read_source.default_vision_adapter",
        lambda: provider,
    )
    claimed = threading.Event()
    release = threading.Event()
    original_apply = RetrievalSyncService.apply
    consumers = []

    def delayed_apply(service, event):
        consumers.append(threading.current_thread().name)
        claimed.set()
        assert release.wait(timeout=5)
        return original_apply(service, event)

    monkeypatch.setattr(RetrievalSyncService, "apply", delayed_apply)
    lifecycle = build_document_maintenance_lifecycle(
        after_pass=build_visual_publication_recovery(database),
    )
    runtime = build_session_workspace_readonly_runtime(session_id)
    registration = next(
        item.registration for item in runtime.catalog_snapshot.entries
        if item.registration.tool_id == "analyze_image"
    )
    ledger = SqliteMountedVisualCallLedger()
    lifecycle.start()
    releaser = None
    if not caller_times_out:
        def release_after_claim():
            if claimed.wait(timeout=5):
                time.sleep(0.1)
            release.set()
        releaser = threading.Thread(target=release_after_claim)
        releaser.start()
    try:
        call_deadline = time.monotonic() + 5
        with tool_execution_scope(ToolExecutionContext(
            deadline_monotonic=call_deadline,
            # 超时定位在后台领取之后，避免把准备速度误当成恢复行为。
            clock=lambda: (
                call_deadline if caller_times_out and claimed.is_set() else time.monotonic()
            ),
            logical_tool_call_id="one-physical-visual-call",
        )):
            payload = {
                "path": target.name, "purpose": "question", "question": "What is shown?",
                "detail": "low", "region": "page",
            }
            if caller_times_out:
                with pytest.raises(ToolInvocationCancelled) as error:
                    registration.handler(payload)
                assert error.value.reason == "execution_timeout"
                assert claimed.is_set()
                assert ledger.list_ready_publications(session_id=session_id)
                release.set()
            else:
                result = registration.handler(payload)
                assert result["observations"][0]["status"] == "completed"
        deadline = time.monotonic() + 5
        while ledger.list_ready_publications(session_id=session_id) and time.monotonic() < deadline:
            time.sleep(0.01)
        assert ledger.list_ready_publications(session_id=session_id) == ()
        assert len(provider.requests) == 1
        assert consumers == ["personagraph-document-maintenance"]
        with database.open_connection() as conn:
            assert tuple(conn.execute("SELECT status, attempts FROM retrieval_update_outbox").fetchone()) == ("applied", 1)
    finally:
        release.set()
        if releaser is not None:
            releaser.join(timeout=5)
        assert lifecycle.stop(timeout_seconds=5)
