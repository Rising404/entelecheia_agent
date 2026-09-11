"""路径视觉问答无需 prepare_files，成功语义必须沿共享发布器落库。"""

from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest
from jsonschema import Draft202012Validator
from PIL import Image

from personagraph.input_processing.vision.contracts import (
    VisionCapabilitySnapshot,
    VisionFailureDiagnostics,
    VisionObservation,
    VisionPurpose,
    VisionResult,
    VisionStatus,
)
from personagraph.tools.execution_context import (
    ToolExecutionContext,
    ToolInvocationCancelled,
    tool_execution_scope,
)
from personagraph.tools.workspace.session_read_source import (
    build_session_workspace_readonly_runtime,
)
from personagraph.workspace.files import FileSource, WorkspaceFileAuthority
from personagraph.workspace.pictures import (
    PictureSourceLocator,
    PictureUnitLocator,
    ensure_picture_in_transaction,
    ensure_picture_unit_in_transaction,
)
from personagraph.workspace.storage.context import require_current


class _VisualProvider:
    transmits_externally = True

    def __init__(self):
        self.requests = []

    def capabilities(self):
        return VisionCapabilitySnapshot(
            available=True,
            provider="test",
            model="question",
            endpoint_identity="test:question",
            processor_fingerprint="test-question",
            supported_purposes=tuple(VisionPurpose),
        )

    def analyze(self, request):
        self.requests.append(request)
        assert request.prepared_payload is not None
        return VisionResult(
            status=VisionStatus.COMPLETED,
            provider="test",
            model="question",
            endpoint_identity="test:question",
            processor_fingerprint="test-question",
            input_sha256=request.prepared_payload.sent_sha256,
            observations=(
                VisionObservation(
                    "provider-question",
                    request.purpose.value,
                    f"观察 {len(self.requests)}：{request.question or request.purpose.value}",
                    0.1,
                ),
            ),
        )


def _seed_two_pdf_page_units(database, registered) -> None:
    """模拟同一 PDF 页已有多个视觉候选；路径工具仍只应派发整页一次。"""

    source = PictureSourceLocator.document_surface("pdf_page", 1)
    locator = PictureUnitLocator.from_payload(
        "render",
        {"bbox": None, "detail": "seed", "page": 1, "region": "page"},
    )
    with database.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        picture = ensure_picture_in_transaction(
            conn,
            file_id=registered.file.file_id,
            file_version_id=registered.version.file_version_id,
            source_locator=source,
            source_content_sha256=registered.version.content_sha256,
            source_media_type="application/pdf",
        ).picture
        for ordinal in range(2):
            ensure_picture_unit_in_transaction(
                conn,
                picture_id=picture.picture_id,
                locator=locator,
                producer_fingerprint=f"seed-renderer-{ordinal}",
                parent_picture_unit_id=None,
                pixel_sha256=hashlib.sha256(f"seed-{ordinal}".encode()).hexdigest(),
                media_type="image/png",
                width=24,
                height=16,
            )


@pytest.mark.parametrize("kind", ["png", "pdf"])
def test_path_question_publishes_two_fresh_calls_and_replays_without_prepare_files(
    kind,
    tmp_path,
    bound_partitioned_session,
    monkeypatch,
):
    root = tmp_path / "workspace"
    root.mkdir()
    target = root / f"source.{kind}"
    if kind == "png":
        Image.new("RGB", (48, 32), "white").save(target)
    else:
        from reportlab.pdfgen.canvas import Canvas

        canvas = Canvas(str(target), pagesize=(240, 160))
        canvas.drawString(20, 80, "No full document preparation")
        canvas.showPage()
        canvas.save()
    session_id = bound_partitioned_session(working_dir=root)
    database = require_current()
    WorkspaceFileAuthority(database).register_path(
        target.name,
        source=FileSource.USER_UPLOAD,
        media_type="image/png" if kind == "png" else "application/pdf",
    )
    provider = _VisualProvider()
    # 保留真实生命周期/SQLite/BM25，只显式选无大模型的本地测试配方。
    monkeypatch.setenv("PERSONAGRAPH_RETRIEVAL_PROFILE", "lexical")
    monkeypatch.setattr(
        "personagraph.tools.workspace.session_read_source.default_vision_adapter",
        lambda: provider,
    )
    runtime = build_session_workspace_readonly_runtime(session_id)
    tool_id = "analyze_image" if kind == "png" else "analyze_pdf_page"
    registration = next(
        item.registration
        for item in runtime.catalog_snapshot.entries
        if item.registration.tool_id == tool_id
    )
    payload = {
        "path": target.name,
        "purpose": "question",
        "question": "图中表示了什么？",
        "detail": "standard",
        "region": "page",
    }
    if kind == "pdf":
        payload["pages"] = [1]
    observations = []
    for call_id in ("question-one", "question-two", "question-one"):
        with tool_execution_scope(
            ToolExecutionContext(deadline_monotonic=None, logical_tool_call_id=call_id)
        ):
            result = registration.handler(payload)
        observation = result["observations"][0]
        assert observation["status"] == "completed"
        assert observation["picture_id"]
        assert observation["picture_unit_id"]
        observations.append(observation)
    assert len(provider.requests) == 2
    assert observations[2] == observations[0]
    assert len({item["picture_unit_id"] for item in observations}) == 1
    with database.connect() as conn:
        rows = conn.execute(
            "SELECT question, text FROM picture_observations ORDER BY sequence"
        ).fetchall()
        assert conn.execute("SELECT COUNT(*) FROM doc_chunks").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM picture_units").fetchone()[0] == 1
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM retrieval_data_versions WHERE role='active' AND state='ready'"
            ).fetchone()[0]
            == 1
        )
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM retrieval_update_outbox WHERE status != 'applied'"
            ).fetchone()[0]
            == 0
        )
    assert len(rows) == 2
    assert [(row[0], row[1]) for row in rows] == [
        (payload["question"], item["observation"]) for item in observations[:2]
    ]


@pytest.mark.parametrize("kind", ["png", "pdf"])
def test_path_general_uses_shared_project_publication_and_retrieval(
    kind,
    tmp_path,
    bound_partitioned_session,
    monkeypatch,
):
    root = tmp_path / "workspace"
    root.mkdir()
    target = root / f"general.{kind}"
    if kind == "png":
        Image.new("RGB", (48, 32), "white").save(target)
    else:
        from reportlab.pdfgen.canvas import Canvas

        canvas = Canvas(str(target), pagesize=(240, 160))
        canvas.drawString(20, 80, "One page, independent of prepared visual units")
        canvas.showPage()
        canvas.save()

    session_id = bound_partitioned_session(working_dir=root)
    database = require_current()
    registered = WorkspaceFileAuthority(database).register_path(
        target.name,
        source=FileSource.USER_UPLOAD,
        media_type="image/png" if kind == "png" else "application/pdf",
    )
    if kind == "pdf":
        _seed_two_pdf_page_units(database, registered)

    provider = _VisualProvider()
    monkeypatch.setenv("PERSONAGRAPH_RETRIEVAL_PROFILE", "lexical")
    monkeypatch.setattr(
        "personagraph.tools.workspace.session_read_source.default_vision_adapter",
        lambda: provider,
    )
    runtime = build_session_workspace_readonly_runtime(session_id)
    tool_id = "analyze_image" if kind == "png" else "analyze_pdf_page"
    registration = next(
        item.registration
        for item in runtime.catalog_snapshot.entries
        if item.registration.tool_id == tool_id
    )
    payload = {
        "path": target.name,
        "purpose": "general",
        "detail": "standard",
        "region": "page",
    }
    if kind == "pdf":
        payload["pages"] = [1]

    with tool_execution_scope(
        ToolExecutionContext(
            deadline_monotonic=None,
            logical_tool_call_id=f"general-{kind}",
        )
    ):
        result = registration.handler(payload)

    observation = result["observations"][0]
    assert observation["status"] == "completed"
    assert observation["picture_id"]
    assert observation["picture_unit_id"]
    assert len(provider.requests) == 1
    request = provider.requests[0]
    assert request.purpose is VisionPurpose.GENERAL
    assert request.prepared_payload is not None
    assert request.pixel_size == request.prepared_payload.pixel_size
    assert request.prepared_payload.data.startswith(b"\x89PNG\r\n\x1a\n")
    if kind == "pdf":
        source_sha256 = hashlib.sha256(target.read_bytes()).hexdigest()
        assert request.image_path == str(target.resolve())
        assert request.image_sha256 == source_sha256
        assert request.prepared_payload.source_sha256 == source_sha256
        assert request.prepared_payload.sent_sha256 != source_sha256

    with database.connect() as conn:
        rows = conn.execute(
            "SELECT purpose, question, text FROM picture_observations"
        ).fetchall()
        assert conn.execute("SELECT COUNT(*) FROM doc_chunks").fetchone()[0] == 0
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM retrieval_data_versions "
                "WHERE role='active' AND state='ready'"
            ).fetchone()[0]
            == 1
        )
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM retrieval_units "
                "WHERE source_type='picture' "
                "AND retrieval_status='active' AND index_state='ready'"
            ).fetchone()[0]
            == 1
        )
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM retrieval_update_outbox WHERE status != 'applied'"
            ).fetchone()[0]
            == 0
        )
    assert [(row[0], row[1], row[2]) for row in rows] == [
        ("general", None, observation["observation"])
    ]


def test_question_without_project_publication_binding_is_rejected_before_provider(
    tmp_path,
):
    from personagraph.tools.documents.format_observation_tools import (
        build_external_visual_analysis_tool_registrations,
    )
    from personagraph.tools.workspace.workspace_tools import FrozenWorkspaceToolBoundary
    from personagraph.tools.execution import ToolBusinessFailure

    target = tmp_path / "image.png"
    Image.new("RGB", (48, 32), "white").save(target)
    provider = _VisualProvider()
    registration = build_external_visual_analysis_tool_registrations(
        FrozenWorkspaceToolBoundary(session_id="unbound-publication", root=tmp_path),
        vision_adapter=provider,
        
    )[0]
    with tool_execution_scope(
        ToolExecutionContext(
            deadline_monotonic=None, logical_tool_call_id="unbound-question"
        )
    ):
        with pytest.raises(ToolBusinessFailure) as exc:
            registration.handler(
                {
                    "path": target.name,
                    "purpose": "question",
                    "question": "图中有什么？",
                    "detail": "standard",
                    "region": "page",
                }
            )
    assert exc.value.error.code == "visual_project_publication_unavailable"
    assert provider.requests == []


@pytest.mark.parametrize("index_fails_once", [False, True, "cancel", "authority", "network", "identity"])
def test_pdf_batch_publishes_each_page_and_replays_without_duplicate_pictures(
    tmp_path, bound_partitioned_session, monkeypatch, index_fails_once,
):
    from reportlab.pdfgen.canvas import Canvas

    root = tmp_path / "workspace"
    root.mkdir()
    target = root / "pages.pdf"
    canvas = Canvas(str(target), pagesize=(240, 160))
    for page in range(1, 4):
        canvas.drawString(20, 80, f"Independent page {page}")
        canvas.showPage()
    canvas.save()
    session_id = bound_partitioned_session(working_dir=root)
    database = require_current()
    WorkspaceFileAuthority(database).register_path(
        target.name, source=FileSource.USER_UPLOAD, media_type="application/pdf",
    )
    provider = _VisualProvider()
    if index_fails_once in {"network", "identity"}:
        original_analyze = provider.analyze

        def analyze(request):
            result = original_analyze(request)
            if len(provider.requests) != 2:
                return result
            if index_fails_once == "identity":
                return replace(result, input_sha256="0" * 64)
            return replace(
                result, status=VisionStatus.FAILED, observations=(),
                unresolved_gap_refs=(request.source_unit_id,),
                failure_code="vision_request_timeout",
                failure_diagnostics=VisionFailureDiagnostics(
                    phase="request", exception_type="TimeoutError", cause_type=None,
                    http_status=None, retry_after_s=None, elapsed_ms=120_000,
                    timeout_s=120, completion_uncertain=True,
                ),
            )

        monkeypatch.setattr(provider, "analyze", analyze)
    elif index_fails_once:
        from personagraph.retrieval.operations.picture_index import (
            PictureObservationIndex, PictureObservationIndexError,
        )
        original_sync = PictureObservationIndex.synchronize
        sync_calls = []

        def synchronize(index, observation_ids, **kwargs):
            sync_calls.append(observation_ids)
            if len(sync_calls) == 2:
                if index_fails_once == "cancel":
                    raise ToolInvocationCancelled("execution_cancelled")
                if index_fails_once == "authority":
                    raise PictureObservationIndexError("picture_retrieval_database_mismatch")
                raise PictureObservationIndexError("picture_retrieval_outbox_incomplete")
            return original_sync(index, observation_ids, **kwargs)

        monkeypatch.setattr(PictureObservationIndex, "synchronize", synchronize)
    monkeypatch.setenv("PERSONAGRAPH_RETRIEVAL_PROFILE", "lexical")
    monkeypatch.setattr(
        "personagraph.tools.workspace.session_read_source.default_vision_adapter",
        lambda: provider,
    )
    runtime = build_session_workspace_readonly_runtime(session_id)
    registration = next(
        item.registration for item in runtime.catalog_snapshot.entries
        if item.registration.tool_id == "analyze_pdf_page"
    )
    payload = {
        "path": target.name, "pages": [3, 1, 2], "purpose": "question",
        "question": "这一页显示什么？", "detail": "low", "region": "page",
    }
    if index_fails_once in {"cancel", "authority", "identity"}:
        from personagraph.tools.execution import ToolBusinessFailure

        expected = ToolInvocationCancelled if index_fails_once == "cancel" else ToolBusinessFailure
        with tool_execution_scope(ToolExecutionContext(
            deadline_monotonic=None, logical_tool_call_id="interrupted-multipage-call",
        )):
            with pytest.raises(expected):
                registration.handler(payload)
        assert [request.page for request in provider.requests] == [3, 1]
        return
    results = []
    for _ in range(2):
        with tool_execution_scope(ToolExecutionContext(
            deadline_monotonic=None, logical_tool_call_id="same-multipage-call",
        )):
            results.append(registration.handler(payload))
            Draft202012Validator(registration.spec.output_schema).validate(results[-1])
    assert [request.page for request in provider.requests] == [3, 1, 2]
    if index_fails_once == "network":
        assert results[0]["observations"][0] == results[1]["observations"][0]
        assert results[0]["observations"][2] == results[1]["observations"][2]
        assert results[0]["analysis"]["resolved_units"] == 2
        failure = results[0]["observations"][1]
        assert failure["failure_code"] == "vision_request_timeout"
        assert failure["failure_diagnostics"]["completion_uncertain"] is True
        assert results[1]["observations"][1]["failure_code"] == "vision_request_timeout"
        return
    if index_fails_once:
        assert results[0]["status"] == "partial"
        assert results[0]["analysis"]["resolved_units"] == 2
        assert [item["status"] for item in results[0]["observations"]] == [
            "completed", "failed", "completed",
        ]
        assert results[0]["observations"][1]["failure_code"] == "picture_retrieval_outbox_incomplete"
        assert results[0]["observations"][0] == results[1]["observations"][0]
        assert results[0]["observations"][2] == results[1]["observations"][2]
        assert results[1]["analysis"]["resolved_units"] == 3
    else:
        assert results[0]["observations"] == results[1]["observations"]
        assert results[0]["analysis"]["resolved_units"] == 3
    assert len({item["picture_id"] for item in results[1]["observations"]}) == 3
    assert len({item["picture_unit_id"] for item in results[1]["observations"]}) == 3
    with database.connect() as conn:
        for table in ("pictures", "picture_units", "picture_observations"):
            assert conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 3
        assert conn.execute("SELECT COUNT(*) FROM doc_chunks").fetchone()[0] == 0
        assert conn.execute(
            "SELECT COUNT(*) FROM retrieval_update_outbox WHERE status != 'applied'"
        ).fetchone()[0] == 0
        assert conn.execute(
            "SELECT COUNT(*) FROM retrieval_units WHERE source_type='picture' "
            "AND retrieval_status='active' AND index_state='ready'"
        ).fetchone()[0] == 3
