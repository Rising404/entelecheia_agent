from __future__ import annotations

import base64
import hashlib
import json
import sqlite3
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from personagraph.l2.auxiliary_graph import PlanningContextArtifact
from personagraph.input_processing.documents import (
    ChunkSpan,
    DiagnosticCode,
    DocumentChunk,
    DocumentLocator,
    DocumentNonTextKind,
    DocumentNonTextUnit,
    DocumentPageInventoryStatus,
    DocumentPageManifest,
    DocumentPageRecord,
    DocumentPageState,
    ElementKind,
    ProcessingDiagnostic,
)
from personagraph.input_processing.files import fingerprint_file
from personagraph.workspace.files.attachments import SessionStorageArea
from personagraph.input_processing.vision.providers import (
    UnavailableVisionModelAdapter,
)
from personagraph.input_processing.vision.contracts import (
    PixelSize,
    VisionCapabilitySnapshot,
    VisionObservation,
    VisionPurpose,
    VisionRequest,
    VisionResult,
    VisionStatus,
)
from personagraph.input_processing.vision.imaging import (
    VisionPayload,
    prepare_payload,
)
from personagraph.input_processing.vision.providers.http import (
    PROMPT_CONTRACT_VERSION,
    HttpVisionModelAdapter,
    VisionProviderConfig,
)
from personagraph.l2.task_graph.contracts import (
    InSessionTaskAcceptanceProposal,
)
from personagraph.l2.task_graph.task_matching import (
    InSessionTaskMatchesProposal,
)
from personagraph.workspace.documents import application as docstore
from personagraph.l2.auxiliary_execution.planning import (
    host_primitive_controller as controller,
)
from personagraph.l2.auxiliary_execution.planning import (
    mounted_visual_resource as visual_resource_module,
)
from personagraph.tools.visual import (
    mounted_visual_source_authority as visual_authority_module,
)
from personagraph.tools.documents import (
    mounted_document_source_authority as source_authority_module,
)
from personagraph.l2.auxiliary_execution.driver import (
    canonical_auxiliary_graph_driver_state_guard,
)
from personagraph.l2.auxiliary_execution.planning.mounted_document_authority import (
    build_mounted_document_authority_projection,
    build_mounted_resource_perception_request,
    build_mounted_resource_read_port,
    freeze_mounted_document_planning_authority,
)
from personagraph.l2.auxiliary_execution.planning.controller import (
    run_initial_auxiliary_planning,
)
from personagraph.l2.auxiliary_execution.planning.model_provider import (
    build_auxiliary_architect_structured_provider,
)
from personagraph.l2.auxiliary_execution.planning.profiles import (
    MOUNTED_VISUAL_READ_CAPABILITY,
    build_auxiliary_planning_capability_catalog,
)
from personagraph.runtime.model_calls.vision import (
    DurableMountedVisionAdapter,
    SqliteMountedVisualCallLedger,
)
from personagraph.tools.visual.mounted_visual_source_authority import (
    MountedVisualResourceFormat,
)
from personagraph.l2.planning.invocation_contracts import (
    PlanningContextPrimitiveKind,
)
from personagraph.l2.planning.resource_perception import (
    PlanningResourceReadRequest,
)
from personagraph.session import store
from personagraph.session.l2_store import task_graph as task_graph_store
from personagraph.session.l2_store import auxiliary_graph as auxiliary_graph_store
from personagraph.session.persistence.l2.auxiliary_graph import auxiliary_graphs


USER_TEXT = "请读取挂载图片中的图表并形成任务图"


@pytest.fixture(autouse=True)
def _project_document_scope(tmp_path):
    from tests.documents._authority import bound_project_document_authority

    with bound_project_document_authority(tmp_path, project_root=tmp_path):
        yield


class _RemoteVisionAdapter:
    transmits_externally = True

    def __init__(self) -> None:
        self.seen = []

    def capabilities(self) -> VisionCapabilitySnapshot:
        return VisionCapabilitySnapshot(
            available=True,
            provider="synthetic",
            model="synthetic-vl",
            endpoint_identity="synthetic:https://vision.invalid",
            processor_fingerprint="synthetic-vision@1",
            supported_purposes=tuple(VisionPurpose),
        )

    def analyze(self, request) -> VisionResult:
        self.seen.append(request)
        return VisionResult(
            status=VisionStatus.COMPLETED,
            provider="synthetic",
            model="synthetic-vl",
            endpoint_identity="synthetic:https://vision.invalid",
            processor_fingerprint="synthetic-vision@1",
            input_sha256=(
                request.prepared_payload.sent_sha256
                if request.prepared_payload is not None
                else request.image_sha256
            ),
            observations=(
                VisionObservation(
                    observation_id="visual_observation_01",
                    kind=request.purpose.value,
                    text="A blue bar reaches 42 while a gray bar reaches 18.",
                    uncertainty=0.1,
                ),
            ),
        )


class _RenderingVisionAdapter(_RemoteVisionAdapter):
    transmits_externally = False

    def __init__(self) -> None:
        super().__init__()
        self.payloads: list[VisionPayload] = []

    def analyze(self, request) -> VisionResult:
        payload = prepare_payload(request)
        assert isinstance(payload, VisionPayload)
        self.payloads.append(payload)
        return super().analyze(request)


class _ExternalRenderingVisionAdapter(_RenderingVisionAdapter):
    transmits_externally = True


class _PayloadFingerprintVisionAdapter(_RemoteVisionAdapter):
    """匹配真实 HTTP 适配器针对载荷生成的结果指纹。"""

    def analyze(self, request) -> VisionResult:
        result = super().analyze(request)
        return VisionResult(
            status=result.status,
            provider=result.provider,
            model=result.model,
            endpoint_identity=result.endpoint_identity,
            processor_fingerprint=(
                f"{result.processor_fingerprint}+image/png:320x180:original"
            ),
            input_sha256=result.input_sha256,
            output_sha256=result.output_sha256,
            observations=result.observations,
            unresolved_gap_refs=result.unresolved_gap_refs,
            warnings=result.warnings,
            failure_code=result.failure_code,
        )


class _SimulatedProcessLoss(BaseException):
    pass


class _CrashAfterExternalSendAdapter(_RemoteVisionAdapter):
    def analyze(self, request) -> VisionResult:
        self.seen.append(request)
        raise _SimulatedProcessLoss()


class _UncertainExternalSendAdapter(_RemoteVisionAdapter):
    def analyze(self, request) -> VisionResult:
        self.seen.append(request)
        raise RuntimeError("response lost after send")


def _window_revision(session_id: str) -> int:
    window = store.get_turn_execution_window(session_id)
    assert window is not None
    return int(window["state_version"])


def _create_task() -> tuple[str, str, str]:
    session_id = store.create_session("Entelecheia")
    accepted = store.accept_turn_execution(
        session_id=session_id,
        client_request_id="aux-v2-mounted-visual",
        source="auxiliary_v2_visual_resource_test",
        user_text=USER_TEXT,
        lease_owner="auxiliary-v2-visual-resource-test",
    )
    turn_id = str(accepted["turn"]["turn_id"])
    matched = task_graph_store.apply_insession_task_matches(
        session_id=session_id,
        source_turn_id=turn_id,
        apply_id="aux-v2-mounted-visual-task",
        proposal=InSessionTaskMatchesProposal.model_validate(
            {
                "task_matches": [
                    {
                        "match_type": "new_root",
                        "local_key": "visual",
                        "title": "理解挂载图片",
                        "objective": "读取图片视觉语义并形成任务图",
                        "source_excerpt": USER_TEXT,
                    }
                ]
            }
        ),
        exposed_catalog_ids=(),
        expected_window_revision=_window_revision(session_id),
    )
    return (
        session_id,
        turn_id,
        matched.created_insession_task_ids_by_local_key["visual"],
    )


def _ingest_visual_source(
    path: Path,
    *,
    session_id: str,
    visual_kind: DocumentNonTextKind = DocumentNonTextKind.FIGURE,
) -> dict[str, object]:
    if path.suffix.lower() == ".pdf":
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas

        sheet = canvas.Canvas(str(path), pagesize=A4)
        sheet.drawString(72, 760, "Chart values require visual interpretation")
        sheet.rect(72, 520, 180, 180, fill=0)
        sheet.showPage()
        sheet.save()
    else:
        image_format = (
            "JPEG"
            if path.suffix.lower() in {".jpg", ".jpeg"}
            else "PNG"
        )
        Image.new("RGB", (320, 180), "white").save(
            path,
            format=image_format,
        )
    fingerprint = fingerprint_file(path)
    text = "The attached chart contains two bars whose values require vision."
    text_id = "visual_text_element_01"
    visual_id = "visual_element_01"
    bbox = (
        (72.0, 142.0, 252.0, 322.0)
        if path.suffix.lower() == ".pdf"
        else (0.0, 0.0, 1.0, 1.0)
    )
    unit = DocumentNonTextUnit(
        unit_id="visual_unit_01",
        kind=visual_kind,
        source_pages=(1,),
        text_element_ids=(text_id,),
        element_id=visual_id,
        locator=DocumentLocator(page=1, ordinal=1, bbox=bbox),
        requires_visual_read=True,
    )
    diagnostic = ProcessingDiagnostic(
        DiagnosticCode.PAGE_NEEDS_VISION,
        DocumentLocator(page=1),
        detail="visual_semantics_unresolved",
    )
    manifest = DocumentPageManifest(
        physical_page_count=1,
        inventory_status=DocumentPageInventoryStatus.COMPLETE,
        detector_fingerprint="synthetic-image-reader@1",
        detector_capabilities=("visual_unit_inventory",),
        pages=(
            DocumentPageRecord(
                page_number=1,
                state=DocumentPageState.MIXED,
                text_element_ids=(text_id,),
                nontext_units=(unit,),
                diagnostics=(diagnostic,),
            ),
        ),
    )
    chunk = DocumentChunk(
        chunk_id="visual_chunk_01",
        text=text,
        span=ChunkSpan(
            DocumentLocator(page=1, ordinal=0),
            DocumentLocator(page=1, ordinal=0),
        ),
        section_path=("Image",),
        element_ids=(text_id,),
        token_count=12,
        kind=ElementKind.PARAGRAPH,
        source_pages=(1,),
    )
    from personagraph.workspace.files import (
        FileSource,
        WorkspaceFileAuthority,
    )
    from personagraph.workspace.storage.context import current

    database = current()
    if database is None:
        raise RuntimeError("visual document fixture requires project authority")
    relative_path = path.resolve().relative_to(database.project_root)
    registration = WorkspaceFileAuthority(database).ensure_current_path(
        relative_path,
        source=FileSource.WORKSPACE_EXISTING,
        media_type="image/png",
    )
    stored = docstore.ingest(
        str(path),
        "synthetic visual",
        "png",
        [{"content": text, "loc": chunk.loc}],
        session_id=session_id,
        file_id=registration.file.file_id,
        file_version_id=registration.version.file_version_id,
        source_fingerprint=fingerprint,
        processor_fingerprint="synthetic-image-reader@1",
        document_chunks=(chunk,),
        chunker_fingerprint="synthetic-image-chunker@1",
        processing_status="partial",
        processing_diagnostics=(diagnostic.to_dict(),),
        page_manifest=manifest,
    )
    return {
        **stored,
        "source_sha256": fingerprint.sha256,
        "manifest_sha256": manifest.manifest_sha256,
        "private_path": str(path),
    }


def test_default_visual_projection_binds_more_than_sixty_four_units(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_path = tmp_path / "many-visual-units.png"
    Image.new("RGB", (320, 180), "white").save(image_path)
    fingerprint = fingerprint_file(image_path)
    units = tuple(
        DocumentNonTextUnit(
            unit_id=f"visual_unit_{ordinal:03d}",
            kind=DocumentNonTextKind.FIGURE,
            source_pages=(1,),
            locator=DocumentLocator(page=1, ordinal=ordinal),
            requires_visual_read=True,
        )
        for ordinal in range(65)
    )
    diagnostic = ProcessingDiagnostic(
        DiagnosticCode.PAGE_NEEDS_VISION,
        DocumentLocator(page=1),
    )
    manifest = DocumentPageManifest(
        physical_page_count=1,
        inventory_status=DocumentPageInventoryStatus.COMPLETE,
        detector_fingerprint="synthetic-many-visuals@1",
        detector_capabilities=("visual_unit_inventory",),
        pages=(
            DocumentPageRecord(
                page_number=1,
                state=DocumentPageState.VISUAL_ONLY,
                text_element_ids=(),
                nontext_units=units,
                diagnostics=(diagnostic,),
            ),
        ),
    )
    monkeypatch.setattr(
        visual_authority_module.docstore,
        "get_current_document_page_authority",
        lambda *_args, **_kwargs: SimpleNamespace(
            document_id="document-many-visuals",
            document_version_id="version-many-visuals",
            source_sha256=fingerprint.sha256,
            page_manifest=manifest,
        ),
    )
    source = visual_resource_module.MountedVisualDocumentSource(
        session_id="session-many-visuals",
        document_ordinal=1,
        document_alias="mounted_document_01",
        document_id="document-many-visuals",
        document_version="version-many-visuals",
        source_sha256=fingerprint.sha256,
        resource_format=MountedVisualResourceFormat.PNG,
        private_path=str(image_path),
    )

    bindings = visual_resource_module.freeze_mounted_visual_planning_bindings(
        session_id="session-many-visuals",
        sources=(source,),
    )

    assert len(bindings) == 65
    assert bindings[0].ordinal == 1
    assert bindings[-1].ordinal == 65


def test_selected_pdf_pages_without_units_get_stable_unique_page_visuals(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas

    pdf_path = tmp_path / "text-only.pdf"
    sheet = canvas.Canvas(str(pdf_path), pagesize=A4)
    for page_number in range(1, 4):
        sheet.drawString(
            72,
            760,
            f"Text page {page_number} the caller explicitly chose to inspect",
        )
        sheet.showPage()
    sheet.save()
    fingerprint = fingerprint_file(pdf_path)
    native_figure = DocumentNonTextUnit(
        unit_id="native_figure_03",
        kind=DocumentNonTextKind.FIGURE,
        source_pages=(3,),
        locator=DocumentLocator(page=3, ordinal=1),
        requires_visual_read=True,
    )
    manifest = DocumentPageManifest(
        physical_page_count=3,
        inventory_status=DocumentPageInventoryStatus.COMPLETE,
        detector_fingerprint="synthetic-text-pdf@1",
        detector_capabilities=("text",),
        pages=tuple(
            DocumentPageRecord(
                page_number=page_number,
                state=(
                    DocumentPageState.MIXED
                    if page_number == 3
                    else DocumentPageState.TEXT
                ),
                text_element_ids=(f"text_{page_number:02d}",),
                nontext_units=(native_figure,) if page_number == 3 else (),
            )
            for page_number in range(1, 4)
        ),
    )
    monkeypatch.setattr(
        visual_authority_module.docstore,
        "get_current_document_page_authority",
        lambda *_args, **_kwargs: SimpleNamespace(
            document_id="document-text-only",
            document_version_id="version-text-only",
            source_sha256=fingerprint.sha256,
            page_manifest=manifest,
        ),
    )
    monkeypatch.setattr(
        visual_authority_module.docstore,
        "is_mounted",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(
        visual_authority_module.docstore,
        "mounted_docs",
        lambda *_args, **_kwargs: (
            {
                "id": "document-text-only",
                "path": str(pdf_path),
            },
        ),
    )
    source = visual_resource_module.MountedVisualDocumentSource(
        session_id="session-text-only",
        document_ordinal=1,
        document_alias="mounted_document_01",
        document_id="document-text-only",
        document_version="version-text-only",
        source_sha256=fingerprint.sha256,
        resource_format=MountedVisualResourceFormat.PDF,
        private_path=str(pdf_path),
    )

    first = visual_resource_module.freeze_mounted_visual_planning_bindings(
        session_id="session-text-only",
        sources=(source,),
        source_pages=(1, 2),
    )
    repeated = visual_resource_module.freeze_mounted_visual_planning_bindings(
        session_id="session-text-only",
        sources=(source,),
        source_pages=(1, 2),
    )
    shifted = visual_resource_module.freeze_mounted_visual_planning_bindings(
        session_id="session-text-only",
        sources=(source,),
        source_pages=(2, 3),
    )
    union = visual_resource_module.freeze_mounted_visual_planning_bindings(
        session_id="session-text-only",
        sources=(source,),
        source_pages=(1, 2, 3),
    )
    retained = visual_resource_module.freeze_mounted_visual_planning_bindings(
        session_id="session-text-only",
        sources=(source,),
        source_pages=(1,),
        include_all_manifest_units=True,
    )

    assert len(first) == 2
    assert first == repeated
    assert [item.visual_unit.kind for item in union] == [
        DocumentNonTextKind.PAGE_VISUAL,
        DocumentNonTextKind.PAGE_VISUAL,
        DocumentNonTextKind.FIGURE,
    ]
    refs_by_page = {
        item.visual_unit.locator.page: item.visual_unit.unit_id
        for item in union
    }
    assert len(refs_by_page) == 3
    assert first[1].visual_unit.unit_id == shifted[0].visual_unit.unit_id
    assert len({item.visual_unit.unit_id for item in union}) == 3
    assert {item.visual_unit.unit_id for item in retained} == {
        first[0].visual_unit.unit_id,
        native_figure.unit_id,
    }
    assert visual_resource_module.mounted_visual_binding_is_current(first[0]) is True

    multi_page = DocumentNonTextUnit(
        unit_id="multi_page_figure_01",
        kind=DocumentNonTextKind.FIGURE,
        source_pages=(1, 2),
        locator=DocumentLocator(page=1, ordinal=1),
        requires_visual_read=True,
    )
    multi_page_manifest = DocumentPageManifest(
        physical_page_count=3,
        inventory_status=DocumentPageInventoryStatus.COMPLETE,
        detector_fingerprint="synthetic-multi-page-pdf@1",
        detector_capabilities=("visual_unit_inventory",),
        pages=tuple(
            DocumentPageRecord(
                page_number=page_number,
                state=(
                    DocumentPageState.MIXED
                    if page_number in multi_page.source_pages
                    else DocumentPageState.TEXT
                ),
                text_element_ids=(f"text_{page_number:02d}",),
                nontext_units=(
                    (multi_page,)
                    if page_number in multi_page.source_pages
                    else ()
                ),
            )
            for page_number in range(1, 4)
        ),
    )
    monkeypatch.setattr(
        visual_authority_module.docstore,
        "get_current_document_page_authority",
        lambda *_args, **_kwargs: SimpleNamespace(
            document_id="document-text-only",
            document_version_id="version-text-only",
            source_sha256=fingerprint.sha256,
            page_manifest=multi_page_manifest,
        ),
    )

    off_locator_page = (
        visual_resource_module.freeze_mounted_visual_planning_bindings(
            session_id="session-text-only",
            sources=(source,),
            source_pages=(2,),
        )
    )

    assert any(
        item.visual_unit.kind is DocumentNonTextKind.PAGE_VISUAL
        and item.visual_unit.locator.page == 2
        for item in off_locator_page
    )


def test_pdf_table_requiring_visual_read_uses_existing_render_and_vision_path(
    tmp_path: Path,
) -> None:
    session_id, _turn_id, _task_id = _create_task()
    _ingest_visual_source(
        tmp_path / "scanned-table.pdf",
        session_id=session_id,
        visual_kind=DocumentNonTextKind.TABLE,
    )
    mounted = freeze_mounted_document_planning_authority(session_id=session_id)

    assert len(mounted.visual_bindings) == 1
    visual = mounted.visual_bindings[0]
    assert visual.visual_unit.kind is DocumentNonTextKind.TABLE
    assert visual.purpose is VisionPurpose.GENERAL
    adapter = _RenderingVisionAdapter()
    observed = build_mounted_resource_read_port(
        mounted_authority=mounted,
        vision_adapter=adapter,

    ).read_frozen_resource(
        PlanningResourceReadRequest(resource=visual.resource)
    )

    assert observed.status.value == "success"
    assert len(adapter.seen) == 1
    assert adapter.seen[0].source_unit_id == visual.visual_unit.unit_id
    assert len(adapter.payloads) == 1
    assert adapter.payloads[0].mime_type == "image/png"


def _seed_visual_host_graph(
    tmp_path: Path,
    *,
    suffix: str = ".png",
    managed_input: bool = False,
    monkeypatch: pytest.MonkeyPatch | None = None,
):
    session_id, turn_id, task_id = _create_task()
    source_path = tmp_path / f"mounted-chart{suffix}"
    if managed_input:
        assert monkeypatch is not None
        managed_root = tmp_path / "附件"
        source_path = managed_root / "attachment_visual_01" / f"mounted-chart{suffix}"
        source_path.parent.mkdir(parents=True, exist_ok=True)
        def classify(_session_id, path):
            return (
                    SessionStorageArea.INPUT
                    if Path(path).is_relative_to(managed_root)
                    else None
                )
        monkeypatch.setattr(source_authority_module, "classify_session_path", classify)
    private = _ingest_visual_source(
        source_path,
        session_id=session_id,
    )
    mounted = freeze_mounted_document_planning_authority(session_id=session_id)
    assert len(mounted.visual_bindings) == 1
    visual = mounted.visual_bindings[0]
    sources = tuple(
        sorted(("task_creation_source", visual.resource.resource_alias))
    )
    acceptance = InSessionTaskAcceptanceProposal(
        acceptance_id="visual_grounded",
        criterion="输出必须保留视觉证据或明确视觉缺口",
        source_anchor_ids=sources,
    )
    proposal = auxiliary_graphs.AuxiliaryGraphRevisionProposalRecord(
        revision_reason="initial",
        terminal_node_key="synthesize",
        nodes=(
            auxiliary_graphs.AuxiliaryGraphNodeProposalRecord(
                local_node_key="observe_visual",
                node_kind="observe",
                executor_kind="host_primitive",
                title="读取挂载视觉单元",
                objective="通过受控视觉边界读取图表",
                source_anchor_ids=sources,
                acceptance_criteria=(acceptance,),
                output_contract="planning_context_artifact_v1",
                capability_profile_id=MOUNTED_VISUAL_READ_CAPABILITY,
                input_resource_aliases=(visual.resource.resource_alias,),
            ),
            auxiliary_graphs.AuxiliaryGraphNodeProposalRecord(
                local_node_key="synthesize",
                node_kind="synthesize",
                executor_kind="terminal_planner",
                title="形成任务图",
                objective="综合视觉证据形成任务图",
                source_anchor_ids=sources,
                acceptance_criteria=(acceptance,),
                output_contract="task_graph_revision_proposal_v2",
                capability_profile_id=None,
            ),
        ),
        edges=(
            auxiliary_graphs.AuxiliaryGraphEdgeProposalRecord(
                dependency_node_key="observe_visual",
                consumer_node_key="synthesize",
            ),
        ),
    )
    auxiliary_graph_store.commit_auxiliary_graph_revision(
        session_id=session_id,
        turn_id=turn_id,
        insession_task_id=task_id,
        expected_task_state_version=1,
        expected_base_task_graph_revision=None,
        expected_control_state_version=None,
        expected_current_auxiliary_graph_revision=None,
        apply_id="aux-v2-mounted-visual-graph",
        goal_objective="理解挂载图片并形成任务图",
        proposal=proposal,
        authority_context=mounted.authority_context,
        budget_profile={"profile_id": "planning-test-v1"},
        auxiliary_graph_id="aux-v2-mounted-visual-graph",
        goal_id="aux-v2-mounted-visual-goal",
    )
    frontier = auxiliary_graph_store.project_auxiliary_graph_execution_frontier(
        session_id=session_id,
        turn_id=turn_id,
        insession_task_id=task_id,
    )
    assert len(frontier.ready_fresh) == 1
    return session_id, turn_id, task_id, private, mounted, frontier


def _request(session_id: str, turn_id: str, frontier):
    candidate = (
        frontier.ready_fresh[0]
        if frontier.ready_fresh
        else frontier.recoverable_primitive[0]
    )
    return controller.AuxiliaryHostPrimitiveControllerRequest(
        session_id=session_id,
        turn_id=turn_id,
        subject=candidate.subject,
        initial_driver_state_guard_sha256=(
            canonical_auxiliary_graph_driver_state_guard(frontier)
        ),
    )


def _factory(mounted):
    visual = mounted.visual_bindings[0]

    def factory(context):
        return build_mounted_resource_perception_request(
            context=context,
            input_resource_aliases=(visual.resource.resource_alias,),
            mounted_authority=mounted,
        )

    return factory


def _run_host(request, *, mounted, port):
    return controller.run_auxiliary_host_primitive(
        request,
        request_factory=_factory(mounted),
        primitive_kinds_by_capability_profile={
            MOUNTED_VISUAL_READ_CAPABILITY: (
                PlanningContextPrimitiveKind.RESOURCE_PERCEPTION
            )
        },
        resource_read_port=port,
        monotonic_clock=lambda: 10.0,
    )


def _stored_artifact(artifact_id: str) -> PlanningContextArtifact:
    with store._connect() as conn:
        row = conn.execute(
            "SELECT artifact_json FROM "
            "insession_auxiliary_planning_context_artifacts WHERE artifact_id=?",
            (artifact_id,),
        ).fetchone()
    assert row is not None
    return PlanningContextArtifact.model_validate_json(row["artifact_json"])


def test_synthetic_png_seals_visual_evidence_without_persisting_private_path(
    tmp_path: Path,
) -> None:
    session_id, turn_id, _task_id, private, mounted, frontier = (
        _seed_visual_host_graph(tmp_path)
    )
    adapter = _RemoteVisionAdapter()
    adapter.capabilities()
    completed = _run_host(
        _request(session_id, turn_id, frontier),
        mounted=mounted,
        port=build_mounted_resource_read_port(
            mounted_authority=mounted,
            vision_adapter=adapter,

        ),
    )

    assert completed.status.value == "completed"
    assert completed.observation_status is not None
    assert completed.observation_status.value == "success"
    assert len(adapter.seen) == 1
    artifact = _stored_artifact(completed.artifact_id)
    assert artifact.facts[0].statement.startswith("A blue bar reaches 42")
    assert artifact.evidence_refs[0].locator == "resource:mounted_visual_001#visual=1"
    with store._connect() as conn:
        serialized = json.dumps(
            [
                dict(row)
                for row in conn.execute(
                    "SELECT logical_request_json FROM "
                    "insession_auxiliary_planning_primitive_invocations"
                ).fetchall()
            ]
        )
        citation = conn.execute(
            "SELECT citation_json FROM "
            "insession_auxiliary_observation_items"
        ).fetchone()
    assert str(private["private_path"]) not in serialized
    assert str(private["doc_id"]) not in serialized
    assert mounted.visual_bindings[0].visual_unit.unit_id not in serialized
    assert citation is not None
    assert (
        json.loads(citation["citation_json"])["authority_anchor"][
            "disclosure_receipt_id"
        ].startswith("auto_visual_egress_")
    )


def test_synthetic_jpg_uses_the_same_frozen_visual_unit_read_path(
    tmp_path: Path,
) -> None:
    session_id, turn_id, _task_id, _private, mounted, frontier = (
        _seed_visual_host_graph(tmp_path, suffix=".jpg")
    )
    visual = mounted.visual_bindings[0]
    assert visual.resource.resource_format.value == "jpg"
    assert visual.resource.media_type == "image/jpeg"
    assert visual.visual_unit.mime_type == "image/jpeg"
    assert visual.visual_unit.image_path.endswith(".jpg")
    adapter = _RenderingVisionAdapter()

    completed = _run_host(
        _request(session_id, turn_id, frontier),
        mounted=mounted,
        port=build_mounted_resource_read_port(
            mounted_authority=mounted,
            vision_adapter=adapter,

        ),
    )

    assert completed.status.value == "completed"
    assert completed.observation_status is not None
    assert completed.observation_status.value == "success"
    assert len(adapter.seen) == 1
    assert adapter.seen[0].mime_type == "image/jpeg"
    assert len(adapter.payloads) == 1
    assert adapter.payloads[0].mime_type == "image/png"




def test_managed_attachment_visual_is_implicitly_authorized_and_sent_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_id, turn_id, _task_id, _private, mounted, frontier = (
        _seed_visual_host_graph(
            tmp_path,
            managed_input=True,
            monkeypatch=monkeypatch,
        )
    )
    adapter = _RemoteVisionAdapter()
    call_ledger = SqliteMountedVisualCallLedger(
        tmp_path / "managed-attachment-visual-calls.sqlite"
    )

    completed = _run_host(
        _request(session_id, turn_id, frontier),
        mounted=mounted,
        port=build_mounted_resource_read_port(
            mounted_authority=mounted,
            vision_adapter=adapter,

            visual_call_ledger=call_ledger,
        ),
    )

    assert completed.status.value == "completed"
    assert completed.observation_status is not None
    assert completed.observation_status.value == "success"
    assert len(adapter.seen) == 1
    with store._connect() as connection:
        citation = connection.execute(
            "SELECT citation_json FROM insession_auxiliary_observation_items"
        ).fetchone()
    assert citation is not None
    receipt_id = json.loads(citation["citation_json"])["authority_anchor"][
        "disclosure_receipt_id"
    ]
    assert receipt_id is not None
    assert receipt_id.startswith("auto_visual_egress_")


def test_managed_attachment_pdf_uses_the_durable_external_visual_bridge(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_id, turn_id, _task_id, private, mounted, frontier = (
        _seed_visual_host_graph(
            tmp_path,
            suffix=".pdf",
            managed_input=True,
            monkeypatch=monkeypatch,
        )
    )
    adapter = _ExternalRenderingVisionAdapter()
    ledger_path = tmp_path / "managed-pdf-visual-calls.sqlite"

    completed = _run_host(
        _request(session_id, turn_id, frontier),
        mounted=mounted,
        port=build_mounted_resource_read_port(
            mounted_authority=mounted,
            vision_adapter=adapter,

            visual_call_ledger=SqliteMountedVisualCallLedger(ledger_path),
        ),
    )

    assert completed.status.value == "completed"
    assert completed.observation_status is not None
    assert completed.observation_status.value == "success"
    assert len(adapter.seen) == 1
    assert len(adapter.payloads) == 1
    assert adapter.payloads[0].mime_type == "image/png"
    request = adapter.seen[0]
    prepared = request.prepared_payload
    assert prepared is not None
    assert request.source_sha256 == private["source_sha256"]
    assert prepared.source_sha256 == request.image_sha256
    assert prepared.sent_sha256 != request.source_sha256
    with sqlite3.connect(ledger_path) as connection:
        persisted = json.loads(
            connection.execute(
                "SELECT request_json FROM mounted_visual_provider_calls"
            ).fetchone()[0]
        )
    assert persisted["source_sha256"] == private["source_sha256"]
    assert persisted["prepared_payload"]["sent_sha256"] == prepared.sent_sha256


def test_external_visual_crash_is_never_automatically_resent(
    tmp_path: Path,
) -> None:
    session_id, turn_id, task_id, private, mounted, frontier = (
        _seed_visual_host_graph(tmp_path)
    )
    initial_adapter = _CrashAfterExternalSendAdapter()
    initial_adapter.capabilities()
    ledger_path = tmp_path / "crashed-visual-calls.sqlite"

    with pytest.raises(_SimulatedProcessLoss):
        _run_host(
            _request(session_id, turn_id, frontier),
            mounted=mounted,
            port=build_mounted_resource_read_port(
                mounted_authority=mounted,
                vision_adapter=initial_adapter,

                visual_call_ledger=SqliteMountedVisualCallLedger(ledger_path),
            ),
        )
    assert len(initial_adapter.seen) == 1
    with sqlite3.connect(ledger_path) as connection:
        assert connection.execute(
            "SELECT status FROM mounted_visual_provider_calls"
        ).fetchone()[0] == "pending"

    pending = auxiliary_graph_store.project_auxiliary_graph_execution_frontier(
        session_id=session_id,
        turn_id=turn_id,
        insession_task_id=task_id,
    )
    recovery_adapter = _RemoteVisionAdapter()
    waiting = _run_host(
        _request(session_id, turn_id, pending),
        mounted=mounted,
        port=build_mounted_resource_read_port(
            mounted_authority=mounted,
            vision_adapter=recovery_adapter,

            visual_call_ledger=SqliteMountedVisualCallLedger(ledger_path),
        ),
    )

    assert waiting.status.value == "waiting_external"
    assert waiting.reason_code == "visual_completion_unconfirmed"
    assert recovery_adapter.seen == []


def test_uncertain_external_visual_response_is_never_automatically_resent(
    tmp_path: Path,
) -> None:
    session_id, turn_id, task_id, private, mounted, frontier = (
        _seed_visual_host_graph(tmp_path)
    )
    initial_adapter = _UncertainExternalSendAdapter()
    initial_adapter.capabilities()
    ledger_path = tmp_path / "uncertain-visual-calls.sqlite"

    first = _run_host(
        _request(session_id, turn_id, frontier),
        mounted=mounted,
        port=build_mounted_resource_read_port(
            mounted_authority=mounted,
            vision_adapter=initial_adapter,

            visual_call_ledger=SqliteMountedVisualCallLedger(ledger_path),
        ),
    )
    assert first.status.value == "waiting_external"
    assert first.reason_code == "visual_completion_unconfirmed"
    assert len(initial_adapter.seen) == 1
    with sqlite3.connect(ledger_path) as connection:
        assert connection.execute(
            "SELECT status FROM mounted_visual_provider_calls"
        ).fetchone()[0] == "uncertain"

    pending = auxiliary_graph_store.project_auxiliary_graph_execution_frontier(
        session_id=session_id,
        turn_id=turn_id,
        insession_task_id=task_id,
    )
    recovery_adapter = _RemoteVisionAdapter()
    second = _run_host(
        _request(session_id, turn_id, pending),
        mounted=mounted,
        port=build_mounted_resource_read_port(
            mounted_authority=mounted,
            vision_adapter=recovery_adapter,

            visual_call_ledger=SqliteMountedVisualCallLedger(ledger_path),
        ),
    )
    assert second.status.value == "waiting_external"
    assert second.reason_code == "visual_completion_unconfirmed"
    assert recovery_adapter.seen == []


def test_completed_external_visual_result_is_exactly_replayed(
    tmp_path: Path,
) -> None:
    session_id, _turn_id, _task_id, private, mounted, _frontier = (
        _seed_visual_host_graph(tmp_path)
    )
    adapter = _RemoteVisionAdapter()
    adapter.capabilities()
    ledger_path = tmp_path / "replayed-visual-calls.sqlite"
    resource = mounted.visual_bindings[0].resource
    read_request = PlanningResourceReadRequest(resource=resource)

    first_port = build_mounted_resource_read_port(
        mounted_authority=mounted,
        vision_adapter=adapter,

        visual_call_ledger=SqliteMountedVisualCallLedger(ledger_path),
    )
    first = first_port.read_frozen_resource(read_request)
    second_port = build_mounted_resource_read_port(
        mounted_authority=mounted,
        vision_adapter=adapter,

        visual_call_ledger=SqliteMountedVisualCallLedger(ledger_path),
    )
    second = second_port.read_frozen_resource(read_request)

    assert first == second
    assert first.status.value == "success"
    assert len(adapter.seen) == 1


def test_durable_visual_reservation_binds_the_verified_sent_payload(
    tmp_path: Path,
) -> None:
    session_id, _turn_id, _task_id, private, mounted, _frontier = (
        _seed_visual_host_graph(tmp_path)
    )
    adapter = _RemoteVisionAdapter()
    adapter.capabilities()
    ledger_path = tmp_path / "payload-bound-visual-calls.sqlite"
    port = build_mounted_resource_read_port(
        mounted_authority=mounted,
        vision_adapter=adapter,

        visual_call_ledger=SqliteMountedVisualCallLedger(ledger_path),
    )

    observed = port.read_frozen_resource(
        PlanningResourceReadRequest(resource=mounted.visual_bindings[0].resource)
    )

    assert observed.status.value == "success"
    request = adapter.seen[0]
    assert request.prepared_payload is not None
    with sqlite3.connect(ledger_path) as connection:
        request_json = connection.execute(
            "SELECT request_json FROM mounted_visual_provider_calls"
        ).fetchone()[0]
    persisted = json.loads(request_json)["prepared_payload"]
    assert persisted["source_sha256"] == request.prepared_payload.source_sha256
    assert persisted["sent_sha256"] == request.prepared_payload.sent_sha256
    assert persisted["byte_count"] == len(request.prepared_payload.data)


def test_changed_prepared_payload_allocates_a_distinct_visual_call(
    tmp_path: Path,
) -> None:
    picture = tmp_path / "reserved.png"
    Image.new("RGB", (32, 24), "white").save(picture)
    raw = picture.read_bytes()
    image_sha256 = hashlib.sha256(raw).hexdigest()
    request = VisionRequest(
        source_unit_id="reserved-visual",
        source_sha256=image_sha256,
        image_sha256=image_sha256,
        locator=DocumentLocator(page=1),
        mime_type="image/png",
        pixel_size=PixelSize(32, 24),
        byte_count=len(raw),
        purpose=VisionPurpose.GENERAL,
        prompt_contract_version=PROMPT_CONTRACT_VERSION,
        image_path=str(picture),
        disclosure_receipt_id="receipt-reserved",
    )
    prepared = prepare_payload(request)
    assert isinstance(prepared, VisionPayload)
    request = replace(request, prepared_payload=prepared)
    ledger_path = tmp_path / "reserved-race.sqlite"
    sent: list[bytes] = []

    def transport(_config, body):
        data_url = body["messages"][0]["content"][1]["image_url"]["url"]
        sent.append(base64.b64decode(data_url.split(",", 1)[1]))
        return {"choices": [{"message": {"content": "verified bytes"}}]}

    class ReplaceAfterReservation(HttpVisionModelAdapter):
        def analyze(self, bound_request):
            with sqlite3.connect(ledger_path) as connection:
                assert connection.execute(
                    "SELECT status FROM mounted_visual_provider_calls"
                ).fetchone()[0] == "pending"
            Image.new("RGB", (32, 24), "black").save(picture)
            return super().analyze(bound_request)

    config = VisionProviderConfig(
        provider="test",
        base_url="https://vision.invalid",
        api_key="secret",
        model="test-vl",
    )
    delegate = ReplaceAfterReservation(
        config,
        transport=transport,
    )
    durable = DurableMountedVisionAdapter(
        delegate,
        session_id="reserved-session",
        ledger=SqliteMountedVisualCallLedger(ledger_path),
    )

    result = durable.analyze(request)

    assert result.status is VisionStatus.COMPLETED
    assert sent == [prepared.data]
    with sqlite3.connect(ledger_path) as connection:
        persisted = json.loads(
            connection.execute(
                "SELECT request_json FROM mounted_visual_provider_calls"
            ).fetchone()[0]
        )
    assert persisted["prepared_payload"]["sent_sha256"] == prepared.sent_sha256

    alternate_data = prepared.data + b"processor-drift"
    alternate = VisionPayload(
        data=alternate_data,
        mime_type=prepared.mime_type,
        pixel_size=prepared.pixel_size,
        sent_sha256=hashlib.sha256(alternate_data).hexdigest(),
        source_sha256=prepared.source_sha256,
        resampled=prepared.resampled,
    )
    changed_request = replace(request, prepared_payload=alternate)
    recovery = DurableMountedVisionAdapter(
        HttpVisionModelAdapter(
            config,
            transport=transport,
        ),
        session_id="reserved-session",
        ledger=SqliteMountedVisualCallLedger(ledger_path),
    )

    changed_result = recovery.analyze(changed_request)

    assert changed_result.status is VisionStatus.COMPLETED
    assert sent == [prepared.data, alternate.data]
    with sqlite3.connect(ledger_path) as connection:
        rows = connection.execute(
            "SELECT call_key, request_binding_sha256 "
            "FROM mounted_visual_provider_calls ORDER BY created_at"
        ).fetchall()
    assert len(rows) == 2
    assert rows[0][0] != rows[1][0]
    assert rows[0][1] != rows[1][1]


def test_payload_specific_processor_fingerprint_is_not_a_provider_change(
    tmp_path: Path,
) -> None:
    session_id, _turn_id, _task_id, private, mounted, _frontier = (
        _seed_visual_host_graph(tmp_path)
    )
    adapter = _PayloadFingerprintVisionAdapter()
    adapter.capabilities()
    port = build_mounted_resource_read_port(
        mounted_authority=mounted,
        vision_adapter=adapter,

        visual_call_ledger=SqliteMountedVisualCallLedger(
            tmp_path / "payload-fingerprint-visual-calls.sqlite"
        ),
    )

    observed = port.read_frozen_resource(
        PlanningResourceReadRequest(
            resource=mounted.visual_bindings[0].resource,
        )
    )

    assert observed.status.value == "success"
    assert len(adapter.seen) == 1


def test_pdf_visual_unit_renders_its_bounded_region_through_existing_adapter(
    tmp_path: Path,
) -> None:
    session_id, turn_id, _task_id, _private, mounted, frontier = (
        _seed_visual_host_graph(tmp_path, suffix=".pdf")
    )
    visual = mounted.visual_bindings[0]
    assert visual.resource.resource_format.value == "pdf"
    assert visual.visual_unit.image_path.endswith(".pdf")
    assert visual.visual_unit.pixel_size.width == 180
    assert visual.visual_unit.pixel_size.height == 180
    adapter = _RenderingVisionAdapter()

    completed = _run_host(
        _request(session_id, turn_id, frontier),
        mounted=mounted,
        port=build_mounted_resource_read_port(
            mounted_authority=mounted,
            vision_adapter=adapter,

        ),
    )

    assert completed.observation_status is not None
    assert completed.observation_status.value == "success"
    assert len(adapter.payloads) == 1
    assert adapter.payloads[0].pixel_size.pixel_count > 100_000


def test_unavailable_provider_seals_a_failed_gap_instead_of_success(
    tmp_path: Path,
) -> None:
    session_id, turn_id, _task_id, _private, mounted, frontier = (
        _seed_visual_host_graph(tmp_path)
    )
    completed = _run_host(
        _request(session_id, turn_id, frontier),
        mounted=mounted,
        port=build_mounted_resource_read_port(
            mounted_authority=mounted,
            vision_adapter=UnavailableVisionModelAdapter(),

        ),
    )

    assert completed.status.value == "completed"
    assert completed.observation_status is not None
    assert completed.observation_status.value == "failed"
    artifact = _stored_artifact(completed.artifact_id)
    assert artifact.facts == ()
    assert all(gap.observation_status.value == "failed" for gap in artifact.gaps)
    assert any(
        "No configured vision provider" in gap.description
        for gap in artifact.gaps
    )


def test_architect_sees_visual_alias_and_capability_but_never_private_binding(
    tmp_path: Path,
) -> None:
    session_id, turn_id, task_id = _create_task()
    private = _ingest_visual_source(
        tmp_path / "planner-private-chart.png",
        session_id=session_id,
    )
    physical = build_auxiliary_architect_structured_provider()
    prompts: list[str] = []

    class RecordingProvider:
        def prepare(self, system_prompt, user_content, **kwargs):
            prompts.append(user_content)
            return physical.prepare(system_prompt, user_content, **kwargs)

    planned = run_initial_auxiliary_planning(
        session_id=session_id,
        turn_id=turn_id,
        insession_task_id=task_id,
        provider=RecordingProvider(),
        ledger_store=store,
        emit=lambda _event: None,
    )

    assert planned.details.auxiliary_graph_revision == 2
    assert len(prompts) == 1
    prompt = prompts[0]
    assert "mounted_visual_001" in prompt
    assert MOUNTED_VISUAL_READ_CAPABILITY in prompt
    assert str(private["private_path"]) not in prompt
    assert str(private["doc_id"]) not in prompt
    mounted = freeze_mounted_document_planning_authority(session_id=session_id)
    assert mounted.visual_bindings[0].visual_unit.unit_id not in prompt
    assert any(
        node.capability_profile_id == MOUNTED_VISUAL_READ_CAPABILITY
        for node in planned.details.nodes
    )
    assert planned.details.authority_snapshot is not None
    creation = task_graph_store.get_insession_task_creation_source(
        session_id=session_id,
        insession_task_id=task_id,
    )
    projection = build_mounted_document_authority_projection(
        authority_snapshot=planned.details.authority_snapshot,
        task_creation_source=creation,
        mounted_authority=mounted,
    )
    assert any(card.source_kind.value == "visual" for card in projection.cards)


def test_ooxml_visual_without_pixel_part_index_is_not_advertised(
    tmp_path: Path,
) -> None:
    session_id, _turn_id, _task_id = _create_task()
    private = _ingest_visual_source(
        tmp_path / "embedded-picture.docx",
        session_id=session_id,
    )

    mounted = freeze_mounted_document_planning_authority(session_id=session_id)
    catalog = build_auxiliary_planning_capability_catalog(mounted)
    visual = next(
        item
        for item in catalog.capabilities
        if item.capability_alias == MOUNTED_VISUAL_READ_CAPABILITY
    )

    assert mounted.visual_bindings == ()
    assert visual.available is False
    assert "ooxml_embedded_pixels_not_indexed" in visual.limitations
    assert str(private["private_path"]) not in catalog.model_dump_json()
