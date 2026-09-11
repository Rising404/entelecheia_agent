"""Focused read_file_visuals -> durable Picture publication vertical tests."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import io
import json
from pathlib import Path

import pytest
from PIL import Image

from personagraph.workspace.files.access import AuthorizedFileSource
from personagraph.input_processing.files import fingerprint_file
from personagraph.workspace.files import FileSource
from personagraph.input_processing.documents.contracts import (
    DocumentLocator,
    DocumentNonTextKind,
)
from personagraph.input_processing.vision.contracts import (
    PixelSize,
    VisionCapabilitySnapshot,
    VisionDetail,
    VisionObservation,
    VisionPurpose,
    VisionRegion,
    VisionRequest,
    VisionResult,
    VisionStatus,
)
from personagraph.input_processing.vision.providers.http import (
    PROMPT_CONTRACT_VERSION,
)
from personagraph.input_processing.vision.imaging import (
    PreparedVisualArtifactBundle,
    prepare_payload_with_receipt,
)
from personagraph.runtime.model_calls.vision import (
    MountedVisualCallLedgerError,
    MountedVisualPictureLocator,
    MountedVisualProjectPublicationEnvelope,
    MountedVisualPublicationState,
    SqliteMountedVisualCallLedger,
)
from personagraph.retrieval.sources.picture_publication import (
    PictureObservationOutboxPublisher,
)
from personagraph.tools.visual.file_visual_adapter import (
    FileVisualRuntime,
    _authority_facts,
)
from personagraph.tools.visual.project_observation_publication import (
    build_picture_publication_command,
    build_picture_publication_envelope,
)
from personagraph.tools.visual.file_visual_tools import (
    build_read_file_visuals_registration,
)
from personagraph.tools.policy import PolicyRequest, ToolPolicyCore
from personagraph.tools.execution_context import ToolExecutionContext, tool_execution_scope
from personagraph.tools.visual.visual_tool_boundary import (
    VisualUnitRef,
)
from personagraph.workspace.pictures.project_publication import (
    ProjectPicturePublicationService,
)
from personagraph.workspace.storage.database import DocumentDatabase


NOW = "2026-09-04T12:00:00+00:00"
VISUAL_REF = "visual_ref_" + "a" * 40


class _Provider:
    transmits_externally = True

    def __init__(
        self,
        *,
        observations: tuple[VisionObservation, ...] | None = None,
        status: VisionStatus = VisionStatus.COMPLETED,
    ) -> None:
        self.calls = 0
        self._observations = observations or (
            VisionObservation(
                observation_id="provider-observation-0",
                kind="chart",
                text="The bars increase from left to right.",
                uncertainty=0.1,
            ),
        )
        self._status = status

    def capabilities(self) -> VisionCapabilitySnapshot:
        return VisionCapabilitySnapshot(
            available=True,
            provider="focused-provider",
            model="focused-vlm",
            endpoint_identity="focused-provider:endpoint",
            processor_fingerprint="focused-provider-v1",
            supported_purposes=tuple(VisionPurpose),
        )

    def analyze(self, request) -> VisionResult:
        self.calls += 1
        assert request.prepared_payload is not None
        if self._status is VisionStatus.FAILED:
            return VisionResult(
                status=VisionStatus.FAILED,
                provider="focused-provider",
                model="focused-vlm",
                endpoint_identity="focused-provider:endpoint",
                processor_fingerprint="focused-provider-v1",
                input_sha256=request.prepared_payload.sent_sha256,
                unresolved_gap_refs=(request.source_unit_id,),
                failure_code="vision_provider_rejected",
            )
        return VisionResult(
            status=self._status,
            provider="focused-provider",
            model="focused-vlm",
            endpoint_identity="focused-provider:endpoint",
            processor_fingerprint="focused-provider-v1",
            input_sha256=request.prepared_payload.sent_sha256,
            output_sha256="f" * 64,
            observations=self._observations,
            unresolved_gap_refs=(
                ("unresolved-caption",)
                if self._status is VisionStatus.PARTIAL
                else ()
            ),
            warnings=("provider-warning",),
        )


class _EscapingMetadataProvider:
    """Legal sub-observations that maximize JSON escaping pressure."""

    transmits_externally = True
    _VALUE = '"' * 256

    def __init__(self) -> None:
        self.calls = 0

    def capabilities(self) -> VisionCapabilitySnapshot:
        return VisionCapabilitySnapshot(
            available=True,
            provider="provider-" + self._VALUE,
            model="model-" + self._VALUE,
            endpoint_identity="endpoint-" + self._VALUE,
            processor_fingerprint="processor-" + self._VALUE,
            supported_purposes=tuple(VisionPurpose),
        )

    def analyze(self, request) -> VisionResult:
        self.calls += 1
        capabilities = self.capabilities()
        observations = tuple(
            VisionObservation(
                observation_id=f"observation-{index}-" + self._VALUE,
                kind=f"kind-{index}-" + self._VALUE,
                text=(
                    "primary\x00" + "x" * 20_000
                    if index == 0
                    else f"secondary-{index}"
                ),
                uncertainty=0.1,
            )
            for index in range(24)
        )
        return VisionResult(
            status=VisionStatus.PARTIAL,
            provider=capabilities.provider,
            model=capabilities.model,
            endpoint_identity=capabilities.endpoint_identity,
            processor_fingerprint=capabilities.processor_fingerprint,
            input_sha256=request.prepared_payload.sent_sha256,
            output_sha256="e" * 64,
            observations=observations,
            unresolved_gap_refs=tuple(
                f"gap-{index}-" + self._VALUE for index in range(24)
            ),
            warnings=tuple(
                f"warning-{index}-" + self._VALUE for index in range(24)
            ),
        )


class _FailOncePublicationService(ProjectPicturePublicationService):
    def __init__(self, delegate: ProjectPicturePublicationService) -> None:
        self._delegate = delegate
        self.failures_remaining = 1

    @property
    def project_id(self):
        return self._delegate.project_id

    def publish(self, command):
        if self.failures_remaining:
            self.failures_remaining -= 1
            raise RuntimeError("injected Project commit failure")
        return self._delegate.publish(command)


class _FailOnceAckLedger(SqliteMountedVisualCallLedger):
    def __init__(self, path: Path) -> None:
        super().__init__(path)
        self.failures_remaining = 1

    def mark_publication_published(self, receipt):
        if self.failures_remaining:
            self.failures_remaining -= 1
            raise MountedVisualCallLedgerError("injected terminal ack failure")
        return super().mark_publication_published(receipt)


@dataclass
class _Harness:
    runtime: FileVisualRuntime
    provider: _Provider
    ledger: SqliteMountedVisualCallLedger
    database: DocumentDatabase
    image_path: Path
    source: AuthorizedFileSource

    @property
    def payload(self) -> dict[str, object]:
        return {"requests": [{
            "file_id": self.source.file_id,
            "file_version_id": self.source.file_version_id,
            "visual_unit_id": "whole_file",
            "purpose": "chart", "detail": "standard", "region": "detected",
        }]}


def _file_source(path: Path, *, media_type: str) -> AuthorizedFileSource:
    return AuthorizedFileSource(
        project_id="project-1", file_id="file-1", file_version_id="version-1",
        canonical_path=str(path), relative_path=path.name, file_name=path.name,
        origin=FileSource.USER_UPLOAD, media_type=media_type,
        fingerprint=fingerprint_file(path),
    )


def _harness(
    tmp_path: Path, *, provider=None, fail_first_publication: bool = False,
    fail_first_ack: bool = False,
) -> _Harness:
    image_path = tmp_path / "chart.png"
    Image.new("RGB", (48, 32), color="white").save(image_path)
    source = _file_source(image_path, media_type="image/png")
    database = _database(tmp_path, content_sha256=source.fingerprint.sha256)
    publication_service = ProjectPicturePublicationService(
        database=database,
        publication_port=PictureObservationOutboxPublisher(
            retrieval_data_version_resolver=lambda _conn: "retrieval-v1",
        ),
    )
    if fail_first_publication:
        publication_service = _FailOncePublicationService(publication_service)
    provider = provider or _Provider()
    ledger = (
        _FailOnceAckLedger(tmp_path / "mounted-visual.sqlite")
        if fail_first_ack
        else SqliteMountedVisualCallLedger(tmp_path / "mounted-visual.sqlite")
    )
    provider.capabilities()
    runtime = FileVisualRuntime(
        session_id="session-1", resolve_file=lambda **kwargs: source,
        revalidate_source=lambda observed: observed == source,
        adapter=provider,  call_ledger=ledger,
        picture_publication_service=publication_service,
    )
    return _Harness(runtime, provider, ledger, database, image_path, source)


def _database(tmp_path: Path, *, content_sha256: str) -> DocumentDatabase:
    root = tmp_path / "project"
    root.mkdir()
    database = DocumentDatabase(
        "project-1",
        root,
        tmp_path / "project.sqlite",
    )
    database.initialize()
    with database.connect() as conn:
        conn.execute(
            "INSERT INTO files "
            "(id, project_id, relative_path, origin, media_type, "
            "current_version_id, created_at, updated_at) "
            "VALUES ('file-1', 'project-1', 'chart.png', 'user_upload', "
            "'image/png', NULL, ?, ?)",
            (NOW, NOW),
        )
        conn.execute(
            "INSERT INTO file_versions "
            "(id, file_id, version_number, producer, content_sha256, size_bytes, "
            "source_mtime_ns, created_at) "
            "VALUES ('version-1', 'file-1', 1, 'user_upload', ?, 128, 1, ?)",
            (content_sha256, NOW),
        )
        conn.execute(
            "UPDATE files SET current_version_id='version-1' WHERE id='file-1'"
        )
        conn.execute(
            "INSERT INTO retrieval_data_versions "
            "(id, fingerprint, role, state, created_at, activated_at) "
            "VALUES ('retrieval-v1', 'fingerprint-v1', 'active', 'ready', ?, ?)",
            (NOW, NOW),
        )
    return database


def _counts(database: DocumentDatabase) -> tuple[int, int, int, int]:
    with database.connect() as conn:
        return tuple(
            int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
            for table in (
                "pictures",
                "picture_units",
                "picture_observations",
                "retrieval_update_outbox",
            )
        )


def _change_current_media_type(
    database: DocumentDatabase,
    media_type: str,
) -> None:
    with database.connect() as conn:
        conn.execute(
            "UPDATE files SET media_type=? WHERE id='file-1'",
            (media_type,),
        )


def test_read_replay_calls_provider_once_and_publishes_once(tmp_path: Path) -> None:
    harness = _harness(tmp_path)

    first = harness.runtime.read_file_visuals(harness.payload)
    second = harness.runtime.read_file_visuals(harness.payload)

    assert first["results"][0]["status"] == "completed"
    assert second == first
    assert harness.provider.calls == 1
    assert _counts(harness.database) == (1, 1, 1, 1)
    assert harness.ledger.list_ready_publications(session_id="session-1") == ()


@pytest.mark.parametrize("fail_first_publication", [False, True])
def test_file_question_is_published_per_call_and_recovers_without_duplicate_dispatch(
    tmp_path: Path, fail_first_publication: bool,
) -> None:
    harness = _harness(tmp_path, fail_first_publication=fail_first_publication)
    payload = harness.payload
    payload["requests"][0].update(purpose="question", question="这些标记表示什么？")
    if fail_first_publication:
        with tool_execution_scope(ToolExecutionContext(deadline_monotonic=None, logical_tool_call_id="file-question-1")):
            failed = harness.runtime.read_file_visuals(payload)
        assert failed["results"][0]["status"] == "blocked"
        assert _counts(harness.database) == (0, 0, 0, 0)
    results = []
    for call_id in ("file-question-1", "file-question-2", "file-question-1"):
        with tool_execution_scope(ToolExecutionContext(deadline_monotonic=None, logical_tool_call_id=call_id)):
            result = harness.runtime.read_file_visuals(payload)["results"][0]
        assert result["status"] == "completed"
        assert result["question"] == "这些标记表示什么？"
        results.append(result)
    assert harness.provider.calls == 2
    assert results[0] == results[2]
    assert results[0]["observation_id"] != results[1]["observation_id"]
    assert _counts(harness.database) == (1, 1, 2, 2)
    with harness.database.connect() as conn:
        rows = conn.execute("SELECT question, text FROM picture_observations ORDER BY sequence").fetchall()
    assert [(row[0], row[1]) for row in rows] == [("这些标记表示什么？", result["observation"]) for result in results[:2]]


def test_visual_publisher_rejects_a_different_project_before_dispatch(tmp_path):
    harness = _harness(tmp_path)

    class WrongProject:
        project_id = "different-project"

        def publish(self, _command):
            raise AssertionError("wrong Project must be rejected before publication")

    harness.runtime._visual_publisher._picture_publication_service = WrongProject()
    payload = harness.payload
    payload["requests"][0].update(purpose="question", question="图中有什么？")
    with tool_execution_scope(ToolExecutionContext(deadline_monotonic=None, logical_tool_call_id="wrong-project-question")):
        result = harness.runtime.read_file_visuals(payload)
    assert result["results"][0]["status"] == "blocked"
    assert harness.provider.calls == 0


def test_index_failure_keeps_receipt_ready_and_recovers_from_observation_identity(tmp_path):
    harness = _harness(tmp_path)

    class Index:
        def __init__(self):
            self.observation_batches = []

        def prepare_target(self):
            return "retrieval-v1"

        def synchronize(self, observation_ids, **_):
            assert observation_ids
            self.observation_batches.append(tuple(observation_ids))
            if len(self.observation_batches) == 1:
                raise RuntimeError("injected indexing failure")

    index = Index()
    harness.runtime._visual_publisher._picture_index = index
    payload = harness.payload
    payload["requests"][0].update(purpose="question", question="图中有什么？")
    with tool_execution_scope(ToolExecutionContext(deadline_monotonic=None, logical_tool_call_id="index-recovery-question")):
        failed = harness.runtime.read_file_visuals(payload)
    assert failed["results"][0]["status"] == "blocked"
    assert len(harness.ledger.list_ready_publications(session_id="session-1")) == 1
    assert _counts(harness.database) == (1, 1, 1, 1)
    with tool_execution_scope(ToolExecutionContext(deadline_monotonic=None, logical_tool_call_id="index-recovery-question")):
        recovered = harness.runtime.read_file_visuals(payload)
    assert recovered["results"][0]["status"] == "completed"
    assert harness.provider.calls == 1
    assert harness.ledger.list_ready_publications(session_id="session-1") == ()
    assert len(index.observation_batches) >= 2
    assert len(set(index.observation_batches)) == 1


def test_published_replay_does_not_resynchronize_a_historical_observation(tmp_path):
    harness = _harness(tmp_path)

    class Index:
        def __init__(self):
            self.synchronized = []

        def prepare_target(self):
            return "retrieval-v1"

        def synchronize(self, observation_ids, **_):
            self.synchronized.append(tuple(observation_ids))

    index = Index()
    harness.runtime._visual_publisher._picture_index = index
    payload = harness.payload
    payload["requests"][0].update(purpose="question", question="图中有什么？")
    with tool_execution_scope(ToolExecutionContext(deadline_monotonic=None, logical_tool_call_id="historical-question")):
        first = harness.runtime.read_file_visuals(payload)
        replay = harness.runtime.read_file_visuals(payload)
    assert replay == first
    assert harness.provider.calls == 1
    assert len(index.synchronized) == 1




def test_project_commit_failure_reuses_result_and_recovers(tmp_path: Path) -> None:
    harness = _harness(tmp_path, fail_first_publication=True)

    first = harness.runtime.read_file_visuals(harness.payload)
    assert first["results"][0]["status"] == "blocked"
    assert _counts(harness.database) == (0, 0, 0, 0)
    (ready,) = harness.ledger.list_ready_publications(session_id="session-1")
    assert ready.publication_state is MountedVisualPublicationState.READY

    second = harness.runtime.read_file_visuals(harness.payload)
    assert second["results"][0]["status"] == "completed"
    assert harness.provider.calls == 1
    assert _counts(harness.database) == (1, 1, 1, 1)


def test_ready_scanner_needs_no_source_or_provider_replay(tmp_path: Path) -> None:
    harness = _harness(tmp_path, fail_first_publication=True)
    harness.runtime.read_file_visuals(harness.payload)
    harness.image_path.unlink()

    recovered = harness.runtime._visual_publisher.recover_ready()

    assert recovered == 1
    assert harness.provider.calls == 1
    assert _counts(harness.database) == (1, 1, 1, 1)


def test_stale_project_binding_is_terminally_skipped_and_fail_closed(
    tmp_path: Path,
) -> None:
    harness = _harness(tmp_path)
    _change_current_media_type(harness.database, "image/jpeg")

    first = harness.runtime.read_file_visuals(harness.payload)
    second = harness.runtime.read_file_visuals(harness.payload)

    assert first["results"][0]["status"] == "blocked"
    assert second["results"][0]["status"] == "blocked"
    assert harness.provider.calls == 1
    assert _counts(harness.database) == (0, 0, 0, 0)
    assert harness.ledger.list_ready_publications(session_id="session-1") == ()
    with harness.ledger._connect("session-1") as conn:
        state = conn.execute(
            "SELECT state FROM mounted_visual_project_publications"
        ).fetchone()[0]
    assert state == "skipped"


def test_ready_scanner_terminally_skips_binding_that_became_stale(
    tmp_path: Path,
) -> None:
    harness = _harness(tmp_path, fail_first_publication=True)
    first = harness.runtime.read_file_visuals(harness.payload)
    assert first["results"][0]["status"] == "blocked"
    assert len(harness.ledger.list_ready_publications(session_id="session-1")) == 1
    _change_current_media_type(harness.database, "image/jpeg")

    assert harness.runtime._visual_publisher.recover_ready() == 0

    assert harness.provider.calls == 1
    assert _counts(harness.database) == (0, 0, 0, 0)
    assert harness.ledger.list_ready_publications(session_id="session-1") == ()


def test_ack_failure_replays_project_idempotently_without_provider_resend(
    tmp_path: Path,
) -> None:
    harness = _harness(tmp_path, fail_first_ack=True)

    first = harness.runtime.read_file_visuals(harness.payload)
    assert first["results"][0]["status"] == "blocked"
    assert _counts(harness.database) == (1, 1, 1, 1)
    assert harness.provider.calls == 1
    assert len(harness.ledger.list_ready_publications(session_id="session-1")) == 1

    second = harness.runtime.read_file_visuals(harness.payload)
    assert second["results"][0]["status"] == "completed"
    assert harness.provider.calls == 1
    assert _counts(harness.database) == (1, 1, 1, 1)
    assert harness.ledger.list_ready_publications(session_id="session-1") == ()


def test_provider_subobservations_consume_one_fifo_entry_in_order(
    tmp_path: Path,
) -> None:
    provider = _Provider(
        observations=(
            VisionObservation("provider-observation-0", "chart", "First", 0.1),
            VisionObservation("provider-observation-1", "caption", "Second", 0.2),
        )
    )
    harness = _harness(tmp_path, provider=provider)

    output = harness.runtime.read_file_visuals(harness.payload)

    assert output["results"][0]["observation"] == "First"
    assert provider.calls == 1
    assert _counts(harness.database) == (1, 1, 1, 1)
    with harness.database.connect() as conn:
        rows = conn.execute(
            "SELECT request_ordinal, text, structured_payload_json "
            "FROM picture_observations ORDER BY sequence"
        ).fetchall()
    assert [(row[0], row[1]) for row in rows] == [(0, "First")]
    assert "provider-observation-0" in rows[0][2]
    assert "provider-observation-1" in rows[0][2]
    assert "Second" not in rows[0][2]


def test_escaped_oversized_provider_metadata_has_a_total_bounded_projection(
    tmp_path: Path,
) -> None:
    provider = _EscapingMetadataProvider()
    harness = _harness(tmp_path, provider=provider)

    first = harness.runtime.read_file_visuals(harness.payload)
    replay = harness.runtime.read_file_visuals(harness.payload)

    assert first["results"][0]["status"] == "partial"
    assert replay == first
    assert provider.calls == 1
    assert _counts(harness.database) == (1, 1, 1, 1)
    with harness.database.connect() as conn:
        row = conn.execute(
            "SELECT kind, text, processor_fingerprint, structured_payload_json "
            "FROM picture_observations"
        ).fetchone()
    assert len(row[1]) == 16_000
    assert "\x00" not in row[1]
    assert len(row[3].encode("utf-8")) < 65_536
    payload = json.loads(row[3])["payload"]
    assert payload["provider_observation_count"] == 24
    assert payload["provider_observations_truncated"] is True
    assert payload["warning_count"] == 24
    assert payload["warnings_truncated"] is True
    assert payload["unresolved_gap_ref_count"] == 24
    assert payload["unresolved_gap_refs_truncated"] is True


def test_failed_provider_result_is_terminally_skipped(tmp_path: Path) -> None:
    harness = _harness(
        tmp_path,
        provider=_Provider(status=VisionStatus.FAILED),
    )

    output = harness.runtime.read_file_visuals(harness.payload)

    assert output["results"][0]["status"] == "failed"
    assert output["results"][0]["reason_code"] == "vision_provider_rejected"
    assert harness.provider.calls == 1
    assert _counts(harness.database) == (0, 0, 0, 0)
    assert harness.ledger.list_ready_publications(session_id="session-1") == ()


def test_public_result_returns_shared_persisted_ids_without_private_paths(tmp_path: Path) -> None:
    harness = _harness(tmp_path)
    result = harness.runtime.read_file_visuals(harness.payload)["results"][0]
    assert result["file_id"] == "file-1"
    assert result["file_version_id"] == "version-1"
    with harness.database.connect() as conn:
        stored = conn.execute(
            "SELECT picture_id, picture_unit_id, observation_id FROM picture_observations"
        ).fetchone()
    assert tuple(result[key] for key in ("picture_id", "picture_unit_id", "observation_id")) == tuple(stored)
    assert result["observation_id"] != "provider-observation-0"
    assert str(harness.image_path) not in repr(result)
    assert "project_id" not in result
    assert "file_content_sha256" not in result


def test_pdf_final_region_is_a_truthful_parentless_render(tmp_path: Path) -> None:
    canvas_module = pytest.importorskip("reportlab.pdfgen.canvas")
    pdf_path = tmp_path / "paper.pdf"
    output = io.BytesIO()
    sheet = canvas_module.Canvas(output, pagesize=(300, 220))
    sheet.drawString(24, 180, "Chart surface")
    sheet.rect(40, 40, 180, 100)
    sheet.showPage()
    sheet.save()
    pdf_path.write_bytes(output.getvalue())
    source_sha256 = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
    unit = VisualUnitRef(
        unit_id="pdf-visual-unit-1",
        kind=DocumentNonTextKind.FIGURE,
        image_path=str(pdf_path),
        source_sha256=source_sha256,
        image_sha256=source_sha256,
        locator=DocumentLocator(page=1, bbox=(40.0, 40.0, 220.0, 140.0)),
        mime_type="image/png",
        pixel_size=PixelSize(180, 100),
        byte_count=pdf_path.stat().st_size,
    )
    request = VisionRequest(
        source_unit_id=unit.unit_id,
        source_sha256=unit.source_sha256,
        image_sha256=unit.image_sha256,
        locator=unit.locator,
        mime_type=unit.mime_type,
        pixel_size=unit.pixel_size,
        byte_count=unit.byte_count,
        purpose=VisionPurpose.CHART,
        prompt_contract_version=PROMPT_CONTRACT_VERSION,
        image_path=unit.image_path,
        detail=VisionDetail.HIGH,
        region=VisionRegion.EXPANDED,
    )
    bundle = prepare_payload_with_receipt(request)
    assert isinstance(bundle, PreparedVisualArtifactBundle)
    envelope = build_picture_publication_envelope(
        binding=_file_source(pdf_path, media_type="application/pdf"),
        unit=unit,
        purpose=request.purpose,
        detail=request.detail,
        region=request.region,
        prepared_artifact=bundle.receipt,
    )
    provider = _Provider()
    receipt = SqliteMountedVisualCallLedger(
        tmp_path / "pdf-mounted-visual.sqlite"
    ).dispatch_with_receipt(
        session_id="session-1",
        adapter=provider,
        request=bundle.prepared_request,
        publication_envelope=envelope,
    )

    target = envelope.target
    command = build_picture_publication_command(receipt)
    assert target.picture_source_kind == "document_surface"
    assert target.picture_source_locator.as_payload()["payload"] == {
        "ordinal": 1,
        "surface_kind": "pdf_page",
    }
    assert target.picture_unit_kind == "render"
    assert target.picture_unit_locator.as_payload()["payload"] == {
        "bbox": [40.0, 40.0, 220.0, 140.0],
        "detail": "high",
        "page": 1,
        "region": "expanded",
    }
    assert command.unit.parent_picture_unit_id is None
    assert command.unit.pixel_sha256 == bundle.receipt.pixel_sha256

    forged_target = replace(
        target,
        picture_unit_locator=MountedVisualPictureLocator.from_payload(
            kind="render",
            payload={
                "bbox": [40.0, 40.0, 220.0, 140.0],
                "detail": "high",
                "page": 2,
                "region": "expanded",
            },
        ),
    )
    forged_receipt = replace(
        receipt,
        publication_envelope=(
            MountedVisualProjectPublicationEnvelope.from_target(forged_target)
        ),
    )
    with pytest.raises(
        MountedVisualCallLedgerError,
        match="does not bind its document surface",
    ):
        build_picture_publication_command(forged_receipt)


def test_external_effect_state_write_has_exact_host_approval() -> None:
    scope = "file-visuals:" + "a" * 64
    registration = build_read_file_visuals_registration(
        handler=lambda _payload: {"contract_version": "file-visual-read-v2", "results": []},
        effect_scope=scope,
        sends_externally=True,
    )

    decision = ToolPolicyCore().evaluate(
        PolicyRequest.from_registration(
            registration,
            {},
            authority=_authority_facts(
                effect_scope=scope,
                sends_externally=True,
            ),
        )
    )

    assert decision.allowed is True
