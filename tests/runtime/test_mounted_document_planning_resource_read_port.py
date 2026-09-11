from __future__ import annotations

import hashlib

import pytest

from personagraph.l2.auxiliary_graph.contracts import PlanningObservationStatus
from personagraph.input_processing.documents import (
    ChunkSpan,
    DocumentChunk,
    DocumentLocator,
    ElementKind,
)
from personagraph.workspace.documents import application as docstore
from personagraph.l2.auxiliary_execution.planning.mounted_document_resource_read_port import (
    MountedDocumentPlanningResourceReadPort,
)
from personagraph.l2.planning.resource_perception import (
    FrozenPlanningResource,
    PlanningResourceCoverage,
    PlanningResourceFormat,
    PlanningResourceGapReason,
    PlanningResourceReadPortError,
    PlanningResourceReadRequest,
)
from personagraph.tools.documents.frozen_mounted_document_reader import (
    FrozenMountedDocument,
)
from tests.documents._authority import (
    ProjectDocumentAuthority,
    bound_project_document_authority,
)


@pytest.fixture
def authority(tmp_path, seed_session_ids):
    seed_session_ids("session_01", "another_session")
    with bound_project_document_authority(tmp_path) as value:
        yield value


def _chunk(ordinal: int, text: str) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=f"producer_chunk_{ordinal}",
        text=text,
        span=ChunkSpan(
            start=DocumentLocator(page=ordinal + 1, ordinal=0),
            end=DocumentLocator(page=ordinal + 1, ordinal=0),
        ),
        section_path=(f"Section {ordinal}",),
        element_ids=(f"element_{ordinal}",),
        token_count=max(1, len(text.split())),
        kind=ElementKind.PARAGRAPH,
        source_pages=(ordinal + 1,),
    )


def _ingest(
    authority: ProjectDocumentAuthority,
    *,
    session_id: str = "session_01",
    path: str = "/workspace/private/paper.pdf",
    processing_status: str = "complete",
    diagnostics: tuple[dict[str, object], ...] = (),
    chunks: tuple[DocumentChunk, ...] | None = None,
) -> tuple[str, str, str]:
    selected = chunks or (_chunk(0, "The paper reports an accuracy of 94%."),)
    relative_path = path.removeprefix("/workspace/").lstrip("/")
    with authority.session(session_id):
        stored = authority.ingest(
            relative_path,
            "paper",
            "pdf",
            [{"content": item.text, "loc": item.loc} for item in selected],
            session_id=session_id,
            processor_fingerprint="reader@test",
            document_chunks=selected,
            chunker_fingerprint="chunker@test",
            processing_status=processing_status,
            processing_diagnostics=diagnostics,
        )
    version = docstore.list_document_versions(stored["doc_id"])[0]
    return stored["doc_id"], str(version["id"]), str(version["source_sha256"])


def _request(
    doc_id: str,
    version_id: str,
    content_sha256: str,
    *,
    session_id: str = "session_01",
    coverage: PlanningResourceCoverage = PlanningResourceCoverage.COMPLETE,
    maximum: int = 64,
) -> PlanningResourceReadRequest:
    return PlanningResourceReadRequest(
        resource=FrozenPlanningResource(
            session_id=session_id,
            resource_alias="paper_01",
            resource_id=doc_id,
            resource_version=version_id,
            content_sha256=content_sha256,
            coverage=coverage,
            resource_format=PlanningResourceFormat.PDF,
            media_type="application/pdf",
            file_extension=".pdf",
        ),
        max_evidence_units=maximum,
    )


def _frozen_document(resource: FrozenPlanningResource) -> FrozenMountedDocument:
    return FrozenMountedDocument(
        session_id=resource.session_id,
        resource_alias=resource.resource_alias,
        document_id=resource.resource_id,
        document_version_id=resource.resource_version,
        source_sha256=resource.content_sha256,
        processing_status=resource.coverage.value,
        resource_format=resource.resource_format.value,
        media_type=resource.media_type,
        file_extension=resource.file_extension,
    )


def test_exact_mounted_document_snapshot_becomes_bounded_text_evidence(authority):
    doc_id, version_id, source_sha256 = _ingest(authority, )
    port = MountedDocumentPlanningResourceReadPort()

    outcome = port.read_frozen_resource(
        _request(doc_id, version_id, source_sha256)
    )

    assert outcome.status is PlanningObservationStatus.SUCCESS
    assert outcome.observed_resource_version == version_id
    assert outcome.observed_content_sha256 == source_sha256
    assert outcome.observed_coverage is PlanningResourceCoverage.COMPLETE
    assert len(outcome.evidence) == 1
    evidence = outcome.evidence[0]
    assert evidence.statement == "The paper reports an accuracy of 94%."
    assert evidence.content_sha256 == hashlib.sha256(
        evidence.statement.encode("utf-8")
    ).hexdigest()
    assert evidence.locator == "resource:paper_01#page=1&chunk=0"
    assert "/workspace/private" not in evidence.locator
    assert outcome.gap_reasons == ()


def test_partial_document_coverage_stays_an_explicit_gap(authority):
    doc_id, version_id, source_sha256 = _ingest(
        authority,
        processing_status="partial",
        diagnostics=(
            {"code": "page_needs_vision", "loc": {"page": 2}},
        ),
    )
    port = MountedDocumentPlanningResourceReadPort()

    outcome = port.read_frozen_resource(
        _request(
            doc_id,
            version_id,
            source_sha256,
            coverage=PlanningResourceCoverage.PARTIAL,
        )
    )

    assert outcome.status is PlanningObservationStatus.PARTIAL
    assert outcome.evidence
    assert outcome.gap_reasons == (
        PlanningResourceGapReason.INCOMPLETE_COVERAGE,
    )


def test_request_bound_limits_do_not_silently_drop_document_chunks(authority):
    chunks = tuple(_chunk(index, f"Evidence statement {index}.") for index in range(3))
    doc_id, version_id, source_sha256 = _ingest(authority, chunks=chunks)
    port = MountedDocumentPlanningResourceReadPort()

    outcome = port.read_frozen_resource(
        _request(doc_id, version_id, source_sha256, maximum=2)
    )

    assert outcome.status is PlanningObservationStatus.PARTIAL
    assert [item.statement for item in outcome.evidence] == [
        "Evidence statement 0.",
        "Evidence statement 1.",
    ]
    assert outcome.gap_reasons == (
        PlanningResourceGapReason.PROJECTION_BOUNDED,
    )


@pytest.mark.parametrize(
    ("version_id", "content_sha256"),
    (("stale_version", "a" * 64), ("version_ignored", "b" * 64)),
)
def test_current_document_identity_drift_returns_stale_not_old_content(
    authority,
    version_id: str,
    content_sha256: str,
):
    doc_id, current_version_id, current_sha256 = _ingest(authority)
    if version_id == "version_ignored":
        version_id = current_version_id
    port = MountedDocumentPlanningResourceReadPort()

    outcome = port.read_frozen_resource(
        _request(doc_id, version_id, content_sha256)
    )

    assert outcome.status is PlanningObservationStatus.STALE
    assert outcome.evidence == ()
    assert outcome.observed_resource_version == current_version_id
    assert outcome.observed_content_sha256 == current_sha256
    assert outcome.gap_reasons == (PlanningResourceGapReason.RESOURCE_STALE,)


def test_physical_source_drift_returns_stale_without_projecting_old_chunks(
    authority,
):
    doc_id, version_id, source_sha256 = _ingest(authority)
    (authority.root / "private" / "paper.pdf").write_text(
        "replacement bytes",
        encoding="utf-8",
    )

    outcome = MountedDocumentPlanningResourceReadPort().read_frozen_resource(
        _request(doc_id, version_id, source_sha256)
    )

    assert outcome.status is PlanningObservationStatus.STALE
    assert outcome.evidence == ()
    assert outcome.gap_reasons == (PlanningResourceGapReason.RESOURCE_STALE,)


def test_unmounted_or_unknown_document_is_access_blocked(authority):
    doc_id, version_id, source_sha256 = _ingest(authority, session_id="another_session")
    port = MountedDocumentPlanningResourceReadPort()

    with pytest.raises(PlanningResourceReadPortError) as captured:
        port.read_frozen_resource(
            _request(doc_id, version_id, source_sha256, session_id="session_01")
        )

    assert captured.value.status is PlanningObservationStatus.BLOCKED
    assert captured.value.reason is PlanningResourceGapReason.ACCESS_BLOCKED


def test_exact_frozen_allowlist_rejects_another_mounted_resource(authority):
    allowed_id, allowed_version, allowed_sha256 = _ingest(authority)
    denied_id, denied_version, denied_sha256 = _ingest(
        authority,
        session_id="session_01",
        path="/workspace/private/other-paper.pdf",
        chunks=(_chunk(1, "A different mounted document."),),
    )
    allowed = _request(allowed_id, allowed_version, allowed_sha256).resource
    port = MountedDocumentPlanningResourceReadPort((_frozen_document(allowed),))

    with pytest.raises(PlanningResourceReadPortError) as captured:
        port.read_frozen_resource(
            _request(denied_id, denied_version, denied_sha256)
        )

    assert captured.value.status is PlanningObservationStatus.BLOCKED
    assert captured.value.reason is PlanningResourceGapReason.ACCESS_BLOCKED


def test_corrupt_current_chunk_never_crosses_as_evidence(authority):
    doc_id, version_id, source_sha256 = _ingest(authority, )
    with authority.database.connect() as conn:
        conn.execute(
            "UPDATE doc_chunks SET content_sha256=? WHERE doc_id=?",
            ("f" * 64, doc_id),
        )
    port = MountedDocumentPlanningResourceReadPort()

    with pytest.raises(PlanningResourceReadPortError) as captured:
        port.read_frozen_resource(_request(doc_id, version_id, source_sha256))

    assert captured.value.status is PlanningObservationStatus.FAILED
    assert captured.value.reason is PlanningResourceGapReason.READ_FAILED
