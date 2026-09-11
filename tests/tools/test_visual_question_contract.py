"""自然语言视觉问答在全部工具入口使用同一合同；不调用真实 provider。"""

from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from jsonschema import Draft202012Validator
from PIL import Image

from personagraph.input_processing.documents.contracts import DocumentLocator, DocumentNonTextKind
from personagraph.input_processing.files import fingerprint_file
from personagraph.input_processing.vision.contracts import (
    PixelSize, VisionCapabilitySnapshot, VisionObservation, VisionPurpose,
    VisionResult, VisionStatus,
)
from personagraph.runtime.model_calls.vision import SqliteMountedVisualCallLedger
from personagraph.tools.documents.format_observation_tools import (
    build_external_visual_analysis_tool_registrations,
)
from personagraph.tools.execution import ToolBusinessFailure
from personagraph.tools.execution_context import ToolExecutionContext, tool_execution_scope
from personagraph.tools.visual.file_visual_adapter import build_file_visual_runtime
from personagraph.tools.visual.mounted_visual_tools import (
    FrozenMountedVisualToolScope, MountedVisualToolBinding, build_mounted_visual_tool_source,
)
from personagraph.tools.visual.visual_tool_boundary import FrozenVisualToolBoundary, VisualUnitRef
from personagraph.tools.visual.visual_tools import build_visual_tool_registrations
from personagraph.tools.workspace.workspace_tools import FrozenWorkspaceToolBoundary
from personagraph.workspace.files import FileSource
from personagraph.workspace.files.access import AuthorizedFileSource


QUESTION = "左侧标注与右侧曲线之间是什么关系？"
TOOL_IDS = ("analyze_image", "analyze_pdf_page", "read_file_visuals", "read_visual_unit", "read_mounted_visuals")


class _LocalVision:
    transmits_externally = False

    def __init__(self):
        self.calls = []

    def capabilities(self):
        return VisionCapabilitySnapshot(
            available=True, provider="local", model="question-test", endpoint_identity="local",
            processor_fingerprint="question-test@1", supported_purposes=tuple(VisionPurpose),
        )

    def analyze(self, request):
        self.calls.append(request)
        return VisionResult(
            status=VisionStatus.COMPLETED, provider="local", model="question-test",
            endpoint_identity="local", processor_fingerprint="question-test@1",
            input_sha256=request.image_sha256,
            observations=(VisionObservation("observation-test", request.purpose.value, "局部曲线对应该标注。", 0.1),),
        )


class _ContractPublisher:
    """只隔离测试工具参数映射；真实事务/恢复由专门发布测试覆盖。"""
    def recover_ready(self):
        return 0

    def observe(self, *, service, boundary, request, binding):
        projection = service.observe(boundary, (request,)).results[0]
        return projection, SimpleNamespace(
            picture_id="picture-contract", picture_unit_id="unit-contract",
            observation_commits=(SimpleNamespace(observation=SimpleNamespace(observation_id="published-contract")),),
        )


def _plain(value):
    if hasattr(value, "items"):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


@pytest.fixture(params=TOOL_IDS)
def question_tool(request, tmp_path: Path):
    tool_id = request.param
    picture = tmp_path / "image.png"
    Image.new("RGB", (48, 32), "white").save(picture)
    digest = hashlib.sha256(picture.read_bytes()).hexdigest()
    adapter = _LocalVision()
    options = {"purpose": "question", "question": QUESTION}
    if tool_id in {"analyze_image", "analyze_pdf_page"}:
        source = picture
        if tool_id == "analyze_pdf_page":
            from reportlab.pdfgen.canvas import Canvas
            source = tmp_path / "document.pdf"
            canvas = Canvas(str(source), pagesize=(240, 160))
            canvas.drawString(24, 80, "Question fixture")
            canvas.showPage()
            canvas.save()
        tools = build_external_visual_analysis_tool_registrations(
            FrozenWorkspaceToolBoundary(session_id="question-session", root=tmp_path),
            vision_adapter=adapter,
            resolve_visual_file=lambda relative: AuthorizedFileSource(
                project_id="question-project", file_id="file-question", file_version_id="version-question",
                canonical_path=str(source), relative_path=relative, file_name=source.name,
                origin=FileSource.USER_UPLOAD,
                media_type="application/pdf" if tool_id == "analyze_pdf_page" else "image/png",
                fingerprint=fingerprint_file(source),
            ),
            visual_publisher=_ContractPublisher(),
        )
        registration = next(tool for tool in tools if tool.tool_id == tool_id)
        payload = {"path": source.name, "detail": "standard", "region": "page", **options}
        if tool_id == "analyze_pdf_page":
            payload["pages"] = [1]
        return registration, payload, payload, adapter
    unit = VisualUnitRef(
        unit_id="visual-1", kind=DocumentNonTextKind.FIGURE, image_path=str(picture),
        source_sha256=digest, image_sha256=digest, locator=DocumentLocator(page=1),
        mime_type="image/png", pixel_size=PixelSize(48, 32), byte_count=picture.stat().st_size,
    )
    if tool_id == "read_visual_unit":
        (registration,) = build_visual_tool_registrations(
            FrozenVisualToolBoundary("question-session", (unit,)), adapter=adapter,
        )
        item = {"unit_id": unit.unit_id, **options}
        return registration, {"units": [item]}, item, adapter
    if tool_id == "read_mounted_visuals":
        source = build_mounted_visual_tool_source(
            FrozenMountedVisualToolScope(
                "question-session", "a" * 64,
                (MountedVisualToolBinding("mounted_visual_001", "mounted_document_01", unit),),
            ),
            adapter=adapter, freshness=SimpleNamespace(is_current=lambda _: True),
        )
        item = {"visual_alias": "mounted_visual_001", **options}
        return source.registration, {"visuals": [item]}, item, adapter
    source = AuthorizedFileSource(
        project_id="question-project", file_id="file-1", file_version_id="version-1",
        canonical_path=str(picture), relative_path=picture.name, file_name=picture.name,
        origin=FileSource.WORKSPACE_EXISTING, media_type="image/png", fingerprint=fingerprint_file(picture),
    )
    runtime = build_file_visual_runtime(
        session_id="question-session", resolve_file=lambda **_: source,
        revalidate_source=lambda _: True, adapter=adapter,
        
        call_ledger=SqliteMountedVisualCallLedger(tmp_path / "vision-calls.sqlite"),
    )
    registration = next(tool for tool in runtime.registrations if tool.tool_id == tool_id)
    item = {"file_id": "file-1", "file_version_id": "version-1", "visual_unit_id": "whole_file",
            "detail": "standard", "region": "detected", **options}
    return registration, {"requests": [item]}, item, adapter


def test_question_schema_accepts_question_without_exposing_host_identity(question_tool):
    registration, payload, _, _ = question_tool
    validator = Draft202012Validator(_plain(registration.spec.input_schema))
    assert not list(validator.iter_errors(payload))
    assert list(validator.iter_errors({**payload, "logical_tool_call_id": "model-invented"}))


@pytest.mark.parametrize("invalid", [None, "", "   ", 7, "x" * 4001])
def test_invalid_question_is_rejected_by_schema_and_handler_before_provider(question_tool, invalid):
    registration, payload, item, adapter = question_tool
    item["question"] = invalid
    assert list(Draft202012Validator(_plain(registration.spec.input_schema)).iter_errors(payload))
    with pytest.raises(ToolBusinessFailure):
        registration.handler(payload)
    assert adapter.calls == []


def test_nonquestion_mode_rejects_question_instead_of_silently_ignoring_it(question_tool):
    registration, payload, item, adapter = question_tool
    item["purpose"] = "general"
    assert list(Draft202012Validator(_plain(registration.spec.input_schema)).iter_errors(payload))
    with pytest.raises(ToolBusinessFailure):
        registration.handler(payload)
    assert adapter.calls == []


def test_question_needs_host_call_identity_before_provider(question_tool):
    registration, payload, _, adapter = question_tool
    with pytest.raises(ToolBusinessFailure) as exc:
        registration.handler(payload)
    assert exc.value.error.code == "visual_question_call_identity_required"
    assert adapter.calls == []


def test_question_missing_parameter_is_rejected_before_provider(question_tool):
    registration, payload, item, adapter = question_tool
    del item["question"]
    assert list(Draft202012Validator(_plain(registration.spec.input_schema)).iter_errors(payload))
    with pytest.raises(ToolBusinessFailure) as exc:
        registration.handler(payload)
    assert exc.value.error.code == "invalid_request"
    assert adapter.calls == []


@pytest.mark.parametrize("with_host_identity", [False, True])
def test_existing_mode_accepts_null_question_and_preserves_available_host_identity(question_tool, with_host_identity):
    registration, payload, item, adapter = question_tool
    item.update(purpose="general", question=None)
    identity = "host-existing-purpose" if with_host_identity else None
    with tool_execution_scope(ToolExecutionContext(deadline_monotonic=None, logical_tool_call_id=identity)):
        result = registration.handler(payload)
    assert not list(Draft202012Validator(_plain(registration.spec.input_schema)).iter_errors(payload))
    assert len(adapter.calls) == 1
    assert adapter.calls[0].question is None
    assert adapter.calls[0].logical_tool_call_id == identity
    outputs = result.get("results", result.get("observations", ()))
    assert "question" not in outputs[0]


@pytest.mark.parametrize("kind", [
    DocumentNonTextKind.TABLE, DocumentNonTextKind.FORMULA, DocumentNonTextKind.FIGURE,
    DocumentNonTextKind.VECTOR_GRAPHICS, DocumentNonTextKind.PAGE_VISUAL,
])
def test_question_is_available_for_each_pixel_backed_visual_kind(kind):
    unit = VisualUnitRef(
        unit_id="visual-1", kind=kind, image_path="/not-read-by-this-test.png",
        source_sha256="a" * 64, image_sha256="a" * 64, locator=DocumentLocator(page=1),
        mime_type="image/png", pixel_size=PixelSize(48, 32), byte_count=100,
    )
    assert VisionPurpose.QUESTION in unit.allowed_purposes


def test_question_and_host_identity_reach_the_shared_provider_and_result(question_tool):
    registration, payload, item, adapter = question_tool
    item["question"] = f"  {QUESTION}  "
    control = ToolExecutionContext(deadline_monotonic=None, logical_tool_call_id="host-tool-question-1")
    with tool_execution_scope(control):
        result = registration.handler(payload)
    assert len(adapter.calls) == 1
    assert adapter.calls[0].question == QUESTION
    assert adapter.calls[0].logical_tool_call_id == "host-tool-question-1"
    outputs = result.get("results", result.get("observations", ()))
    assert outputs[0]["question"] == QUESTION
    assert not list(Draft202012Validator(_plain(registration.spec.output_schema)).iter_errors(result))


@pytest.mark.parametrize("tool_id", ["read_visual_unit", "read_mounted_visuals", "read_file_visuals"])
def test_later_question_without_host_identity_never_dispatches_earlier_general(tool_id, tmp_path):
    picture = tmp_path / "image.png"
    Image.new("RGB", (48, 32), "white").save(picture)
    digest = hashlib.sha256(picture.read_bytes()).hexdigest()
    adapter = _LocalVision()
    unit = VisualUnitRef(
        unit_id="visual-1", kind=DocumentNonTextKind.FIGURE, image_path=str(picture),
        source_sha256=digest, image_sha256=digest, locator=DocumentLocator(page=1),
        mime_type="image/png", pixel_size=PixelSize(48, 32), byte_count=picture.stat().st_size,
    )
    second = replace(unit, unit_id="visual-2")
    if tool_id == "read_visual_unit":
        (registration,) = build_visual_tool_registrations(
            FrozenVisualToolBoundary("question-session", (unit, second)), adapter=adapter,
        )
        payload = {"units": [
            {"unit_id": unit.unit_id, "purpose": "general"},
            {"unit_id": second.unit_id, "purpose": "question", "question": QUESTION},
        ]}
    elif tool_id == "read_mounted_visuals":
        source = build_mounted_visual_tool_source(
            FrozenMountedVisualToolScope("question-session", "a" * 64, (
                MountedVisualToolBinding("mounted_visual_001", "mounted_document_01", unit),
                MountedVisualToolBinding("mounted_visual_002", "mounted_document_01", second),
            )), adapter=adapter, freshness=SimpleNamespace(is_current=lambda _: True),
        )
        registration = source.registration
        payload = {"visuals": [
            {"visual_alias": "mounted_visual_001", "purpose": "general"},
            {"visual_alias": "mounted_visual_002", "purpose": "question", "question": QUESTION},
        ]}
    else:
        first_source = AuthorizedFileSource(
            project_id="question-project", file_id="file-1", file_version_id="version-1",
            canonical_path=str(picture), relative_path=picture.name, file_name=picture.name,
            origin=FileSource.WORKSPACE_EXISTING, media_type="image/png", fingerprint=fingerprint_file(picture),
        )
        sources = {"file-1": first_source, "file-2": replace(first_source, file_id="file-2", file_version_id="version-2")}
        runtime = build_file_visual_runtime(
            session_id="question-session", resolve_file=lambda **kw: sources[kw["file_id"]],
            revalidate_source=lambda _: True, adapter=adapter,
            
            call_ledger=SqliteMountedVisualCallLedger(tmp_path / "vision-calls.sqlite"),
        )
        registration = next(tool for tool in runtime.registrations if tool.tool_id == tool_id)
        payload = {"requests": [
            {"file_id": "file-1", "file_version_id": "version-1", "visual_unit_id": "whole_file",
             "purpose": "general", "detail": "standard", "region": "detected"},
            {"file_id": "file-2", "file_version_id": "version-2", "visual_unit_id": "whole_file",
             "purpose": "question", "question": QUESTION, "detail": "standard", "region": "detected"},
        ]}
    with pytest.raises(ToolBusinessFailure) as exc:
        registration.handler(payload)
    assert exc.value.error.code == "visual_question_call_identity_required"
    assert adapter.calls == []
