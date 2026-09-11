"""真实工具 adapter 使用原生身份往返；仅底层数据/视觉端口使用本地替身。"""

from __future__ import annotations

import hashlib
from types import SimpleNamespace

from PIL import Image

from personagraph.input_processing.files import SourceFingerprint, fingerprint_file
from personagraph.input_processing.vision.contracts import (
    VisionCapabilitySnapshot,
    VisionObservation,
    VisionPurpose,
    VisionResult,
    VisionStatus,
)
from personagraph.retrieval.contracts import RetrievalQueryMatch, SourceType
from personagraph.retrieval.lifecycle.generation import (
    DOCUMENT_HYBRID_INDEX_RECIPE_CONTRACT,
    file_corpus_generation_spec,
)
from personagraph.retrieval.profile import durable_chunking_profile
from personagraph.retrieval.sources.identity import (
    mounted_document_chunk_ref_and_content,
    picture_observation_source_unit_id,
)
from personagraph.retrieval.tooling.contracts import (
    RetrievalStatus,
    RetrievalToolEvidence,
    RetrievalToolResult,
)
from personagraph.runtime.model_calls.vision import SqliteMountedVisualCallLedger
from personagraph.tools.documents.file_chunk_reader import (
    build_file_chunk_reader_runtime,
)
from personagraph.tools.documents.file_inspection_adapter import FileInspectionRuntime
from personagraph.tools.files.file_adapter import FileStateRuntime
from personagraph.tools.model_interface import (
    project_tool_result,
)
from personagraph.tools.retrieval.file_retrieval_adapter import (
    build_file_retrieval_runtime,
)
from personagraph.tools.schema_validation import ToolSchemaCompiler
from personagraph.tools.visual.file_visual_adapter import build_file_visual_runtime
from personagraph.workspace.documents.inspection import (
    ChunkLocation,
    FileDocumentInspection,
    SourceTextElement,
)
from personagraph.workspace.documents.reading import (
    CurrentFileChunk,
    CurrentFileDocument,
)
from personagraph.workspace.files import FileSource
from personagraph.workspace.files.access import AuthorizedFileSource
from personagraph.workspace.ingestion.contracts import (
    FilePreparationResult,
    FilePreparationStatus,
)


def _document_fixture():
    content = "First statement. Second statement."
    source = SimpleNamespace(
        file_id="file-a",
        file_version_id="fv-a",
        project_id="project-a",
        file_name="brief.md",
        relative_path="docs/brief.md",
        origin="workspace_existing",
        fingerprint=SourceFingerprint(
            sha256="a" * 64, size_bytes=40, mtime_ns=1788566400000000000
        ),
    )
    document = CurrentFileDocument(
        "file-a",
        "fv-a",
        "doc-a",
        "dv-a",
        "a" * 64,
        1,
        "complete",
        1,
        "2026-09-05T00:01:00+00:00",
    )
    chunk = CurrentFileChunk(
        document,
        "chunk-a",
        "producer-a",
        0,
        content,
        hashlib.sha256(content.encode()).hexdigest(),
        "page 1",
        (1,),
    )
    return source, document, chunk


def _checked_project(tool_id, payload, registration):
    compiler = ToolSchemaCompiler()
    compiler.compile(registration.spec.input_schema, role="input").validate(payload)
    actual = registration.handler(payload)
    compiler.compile(registration.spec.output_schema, role="output").validate(actual)
    return actual, project_tool_result(tool_id, actual)


def test_actual_state_chunk_inspect_search_shapes_round_trip_exact_versions():
    source, document, chunk = _document_fixture()
    state = FilePreparationResult(
        FilePreparationStatus.READY,
        file_id=source.file_id,
        file_version_id=source.file_version_id,
        document_id=document.document_id,
        document_version_id=document.document_version_id,
    )
    files = FileStateRuntime(
        access=SimpleNamespace(resolve_path=lambda _: source),
        effect_scope="scope",
        check_state=lambda _: state,
        prepare=lambda _: state,
    )
    for registration in files.registrations:
        actual, projected = _checked_project(
            registration.tool_id,
            {"files": [{"path": "docs/brief.md"}]},
            registration,
        )
        assert actual["results"][0]["document_version_id"] == "dv-a"
        assert projected["results"][0]["document_version_id"] == "dv-a"
        assert projected["ready_indices"] == [0]
        assert "files" not in projected
    reader = build_file_chunk_reader_runtime(
        session_id="s",
        scope_id="scope",
        resolve_file=lambda **_: source,
        revalidate=lambda _: True,
        read_document=lambda **_: document,
        read_chunk=lambda **_: chunk,
    )
    _, body = _checked_project(
        "read_file_chunks",
        {"targets": [{"file_id": "file-a", "document_version_id": "dv-a", "chunk_sequences": [0]}]},
        reader.registration(),
    )
    assert body["results"][0]["chunks"][0]["content"] == chunk.content
    assert "content_sha256" not in body["results"][0]["chunks"][0]
    snapshot = FileDocumentInspection(
        document,
        (ChunkLocation("chunk-a", 0, "p1", (1,), ("e0",)),),
        (SourceTextElement("e0", chunk.content, "p1", (1,)),),
    )
    inspection = FileInspectionRuntime(
        session_id="s",
        effect_scope="scope",
        resolve_file=lambda **_: source,
        revalidate=lambda _: True,
        inspect_document=lambda **_: snapshot,
        read_document=lambda **_: document,
    )
    for registration in inspection.registrations:
        payload = {"targets": [{"file_id": "file-a", "document_version_id": "dv-a"}]}
        if registration.tool_id == "search_file_text":
            payload["text"] = "statement"
        _, projected = _checked_project(
            registration.tool_id, payload, registration
        )
        assert projected["results"][0]["document_version_id"] == "dv-a"
        assert projected["results"][0]["status"] == "ready"
    assert projected["results"][0]["total_match_count"] == 2


def test_actual_retrieval_projection_preserves_citation_metadata_and_feeds_chunk_read():
    source, document, chunk = _document_fixture()
    generation = file_corpus_generation_spec(
        encoder_fingerprint="test-encoder",
        document_chunker_fingerprint=durable_chunking_profile().fingerprint(),
        document_chunk_contract_version=1,
        index_recipe=DOCUMENT_HYBRID_INDEX_RECIPE_CONTRACT,
    )
    ref, content = mounted_document_chunk_ref_and_content(
        session_id="s",
        chunk_id=chunk.chunk_id,
        source_version_id=document.document_version_id,
        content=chunk.content,
        doc_id=document.document_id,
        producer_chunk_id=chunk.producer_chunk_id,
    )

    def factory(_bridge):
        def retrieve(request):
            evidence = RetrievalToolEvidence(
                source_type=SourceType.DOCUMENT,
                source_unit_id=ref.source_unit_id,
                source_revision=ref.source_revision,
                indexed_content_hash=ref.indexed_content_hash,
                content=content,
                estimated_tokens=8,
                rank=1,
                query_index=0,
                query_matches=(RetrievalQueryMatch(0, 1, 1),),
                authority_id="file-a",
                citation={
                    "file_id": "file-a",
                    "file_version_id": "fv-a",
                    "doc_id": "doc-a",
                    "chunk_id": "chunk-a",
                    "producer_chunk_id": "producer-a",
                    "location": "page 1",
                },
            )
            picture = RetrievalToolEvidence(
                source_type=SourceType.PICTURE,
                source_unit_id=picture_observation_source_unit_id("obs-a"),
                source_revision="obs-v1",
                indexed_content_hash=hashlib.sha256(b"Three bars").hexdigest(),
                content="Three bars",
                estimated_tokens=3,
                rank=2,
                query_index=0,
                query_matches=(RetrievalQueryMatch(0, 2, 2),),
                authority_id="file-a",
                citation={
                    "file_id": "file-a",
                    "file_version_id": "fv-a",
                    "picture_id": "picture-a",
                    "picture_unit_id": "unit-a",
                    "observation_id": "obs-a",
                    "surface_kind": "pdf_page",
                    "surface_ordinal": "1",
                },
            )
            return RetrievalToolResult(
                status=RetrievalStatus.COMPLETE,
                scope_snapshot_id=request.scope_snapshot_id,
                retrieval_data_version=request.retrieval_data_version,
                evidence=(evidence, picture),
            )

        return SimpleNamespace(retrieve=retrieve)

    runtime = build_file_retrieval_runtime(
        session_id="s",
        scope_id="scope",
        generation_spec=generation,
        resolve_file=lambda **_: source,
        revalidate=lambda _: True,
        port_factory=factory,
        read_document=lambda **_: document,
        read_chunk=lambda **_: chunk,
        picture_file_authority=SimpleNamespace(
            authorize=lambda **_: SimpleNamespace(allowed=True)
        ),
    )
    actual, projected = _checked_project(
        "retrieve_files",
        {"queries": ["statement"], "file_ids": ["file-a"]},
        runtime.registration(),
    )
    assert projected["status"] == actual["status"]
    assert len(projected["files"]) == 2
    file = projected["files"][0]
    item = file["chunks"][0]
    assert item["snippet"] == chunk.content
    assert file["source"]["relative_path"] == source.relative_path
    assert file["document_version_id"] == document.document_version_id
    assert "query_matches" not in item and "content_sha256" not in item
    reader = build_file_chunk_reader_runtime(
        session_id="s", scope_id="scope", resolve_file=lambda **_: source,
        revalidate=lambda _: True, read_document=lambda **_: document,
        read_chunk=lambda **_: chunk, read_chunk_document=lambda **_: document,
    )
    _, body = _checked_project(
        "read_file_chunks", {"targets": [{"chunk_ids": [item["chunk_id"]]}]},
        reader.registration(),
    )
    assert body["results"][0]["chunks"][0]["content"] == item["snippet"]
    picture = projected["files"][1]["observations"][0]
    assert picture["snippet"] == "Three bars" and picture["locator"] == "page 1"
    assert picture["picture_id"] == "picture-a" and picture["observation_id"] == "obs-a"
    assert "content_sha256" not in picture


class _VisionFixture:
    transmits_externally = False

    def capabilities(self):
        return VisionCapabilitySnapshot(
            available=True,
            provider="fixture",
            model="fixture",
            endpoint_identity="local",
            processor_fingerprint="fixture@1",
            supported_purposes=tuple(VisionPurpose),
        )

    def analyze(self, request):
        return VisionResult(
            status=VisionStatus.COMPLETED,
            provider="fixture",
            model="fixture",
            endpoint_identity="local",
            processor_fingerprint="fixture@1",
            input_sha256=request.image_sha256,
            observations=(VisionObservation("vendor-id", "chart", "Three bars", 0.1),),
        )


def test_actual_visual_adapter_returns_results_and_keeps_unit_selector(tmp_path):
    path = tmp_path / "image.png"
    Image.new("RGB", (24, 16)).save(path)
    source = AuthorizedFileSource(
        project_id="p",
        file_id="f",
        file_version_id="fv",
        canonical_path=str(path),
        relative_path=path.name,
        file_name=path.name,
        origin=FileSource.WORKSPACE_EXISTING,
        media_type="image/png",
        fingerprint=fingerprint_file(path),
    )
    runtime = build_file_visual_runtime(
        session_id="s",
        resolve_file=lambda **_: source,
        revalidate_source=lambda _: True,
        adapter=_VisionFixture(),
        call_ledger=SqliteMountedVisualCallLedger(tmp_path / "calls.sqlite"),
    )
    raw_list = runtime.list_file_visuals({"file_id": "f", "file_version_id": "fv"})
    listed = project_tool_result("list_file_visuals", raw_list)
    assert listed["file_id"] == "f" and listed["file_version_id"] == "fv"
    visual = listed["visuals"][0]["visual_unit_id"]
    payload = {
        "requests": [
            {
                "file_id": "f",
                "file_version_id": "fv",
                "visual_unit_id": visual,
                "purpose": "chart",
                "detail": "standard",
                "region": "detected",
            }
        ]
    }
    actual = runtime.read_file_visuals(payload)
    projected = project_tool_result("read_file_visuals", actual)
    assert "observations" not in actual and "requests" not in actual
    assert projected["results"][0]["observation"] == "Three bars"
    assert projected["results"][0]["visual_unit_id"] == visual == "whole_file"
    assert projected["results"][0]["file_id"] == "f" and projected["results"][0]["file_version_id"] == "fv"
