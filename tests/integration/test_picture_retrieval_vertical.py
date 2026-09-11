"""Offline acceptance: native File visual reads publish into shared retrieval."""

from __future__ import annotations

import hashlib
from PIL import Image

from personagraph.input_processing.documents import ChunkSpan, DocumentChunk, DocumentLocator, ElementKind
from personagraph.input_processing.vision.contracts import VisionCapabilitySnapshot, VisionObservation, VisionPurpose, VisionResult, VisionStatus
from personagraph.runtime.model_calls.vision import SqliteMountedVisualCallLedger
from personagraph.retrieval.contracts import SourceAvailability, SourceFilter, SourceType
from personagraph.retrieval.lifecycle.outbox import SqliteRetrievalOutbox
from personagraph.retrieval.lifecycle.sync import RetrievalOutboxConsumer
from personagraph.retrieval.operations.document_maintenance import build_document_retrieval_composition
from personagraph.retrieval.profile import DocumentRetrievalProfile
from personagraph.retrieval.sources.picture import PictureObservationSourceAdapter
from personagraph.retrieval.sources.picture_publication import PictureObservationOutboxPublisher
from personagraph.retrieval.sqlite_store import RetrievalDataVersionRole, RetrievalDataVersionState
from personagraph.tools.documents.file_chunk_reader import build_file_chunk_reader_runtime
from personagraph.tools.schema_validation import ToolSchemaCompiler
from personagraph.tools.retrieval.file_retrieval_adapter import build_file_retrieval_runtime
from personagraph.tools.retrieval.picture_file_authority import PictureRetrievalFileAuthority
from personagraph.tools.visual.file_visual_adapter import build_file_visual_runtime
from personagraph.workspace.documents.mounting import mount_document
from personagraph.workspace.files.access import FileAccess
from personagraph.workspace.pictures.project_publication import ProjectPicturePublicationService
from tests.documents._authority import bound_project_document_authority


NOW="2026-09-04T12:00:00+00:00"


class _OfflineProvider:
    transmits_externally=True

    def __init__(self):
        self.calls=0

    def capabilities(self):
        return VisionCapabilitySnapshot(available=True,provider="offline-provider",model="offline-vlm",
            endpoint_identity="offline-provider:endpoint",processor_fingerprint="offline-provider-v1",
            supported_purposes=tuple(VisionPurpose))

    def analyze(self, request):
        self.calls+=1
        assert request.prepared_payload is not None
        return VisionResult(status=VisionStatus.COMPLETED,provider="offline-provider",model="offline-vlm",
            endpoint_identity="offline-provider:endpoint",processor_fingerprint="offline-provider-v1",
            input_sha256=request.prepared_payload.sent_sha256,output_sha256="f"*64,
            observations=(VisionObservation(observation_id="offline-observation-1",kind="chart",
                text="Northbridge chart revenue rises sharply in the final quarter.",uncertainty=0.05),))


def test_visual_observation_joins_document_evidence_in_shared_file_retrieval(tmp_path, seed_session_ids):
    seed_session_ids("session-1","session-2","session-denied")
    with bound_project_document_authority(tmp_path) as authority:
        image_path=authority.root/"chart.png"
        Image.new("RGB",(48,32),color="white").save(image_path)
        image_bytes=image_path.read_bytes()
        hashlib.sha256(image_bytes).hexdigest()
        picture_authority=PictureRetrievalFileAuthority(session_id="session-1",authorized_file_bindings=lambda:())
        picture_source=PictureObservationSourceAdapter(access_authority=picture_authority)
        composition=build_document_retrieval_composition(retrieval_db_path=authority.database.db_path,
            profile=DocumentRetrievalProfile.lexical(),picture_source_adapter=picture_source)
        generation=composition.generation_spec
        composition.foundation.catalog.create_data_version(version_id=generation.version_id,
            fingerprint=generation.fingerprint,role=RetrievalDataVersionRole.ACTIVE,state=RetrievalDataVersionState.READY)
        text="Northbridge report: the final-quarter chart documents revenue growth."
        locator=DocumentLocator(page=1,ordinal=0,section_path=("Results",))
        document=authority.ingest("chart.png","Northbridge chart","image/png",[{"content":text,"loc":"p.1"}],
            session_id="session-1",source_payload=image_bytes,retrieval_data_version=generation.version_id,
            processor_fingerprint="integration-reader-v1",
            document_chunks=(DocumentChunk(chunk_id="chart-caption-chunk",text=text,
                span=ChunkSpan(start=locator,end=locator),section_path=("Results",),element_ids=("caption-1",),
                token_count=9,kind=ElementKind.PARAGRAPH),),
            chunker_fingerprint=composition.chunking_profile.fingerprint(),processing_status="complete",processing_diagnostics=())
        access=FileAccess(database=authority.database,validate_path=lambda path:path==str(image_path))
        source=access.resolve_path("chart.png")
        provider=_OfflineProvider()
        publication=ProjectPicturePublicationService(database=authority.database,
            publication_port=PictureObservationOutboxPublisher(retrieval_data_version_resolver=lambda conn:generation.version_id))
        visual=build_file_visual_runtime(session_id="session-1",resolve_file=access.resolve_file,
            revalidate_source=access.revalidate,adapter=provider,
            call_ledger=SqliteMountedVisualCallLedger(tmp_path/"visual-calls.sqlite"),picture_publication_service=publication)
        request={"requests":[{"file_id":source.file_id,"file_version_id":source.file_version_id,
                              "visual_unit_id":"whole_file","purpose":"chart","detail":"standard","region":"detected"}]}
        first=visual.read_file_visuals(request)
        replay=visual.read_file_visuals(request)
        assert first["results"][0]["status"]=="completed"
        assert replay==first and provider.calls==1
        with authority.database.connect() as conn:
            counts=tuple(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                         for table in ("pictures","picture_units","picture_observations"))
            assert counts==(1,1,1)
            assert {row[0] for row in conn.execute("SELECT source_type FROM retrieval_update_outbox WHERE status='pending'")}=={"document","picture"}
        consumer=RetrievalOutboxConsumer(outbox=SqliteRetrievalOutbox(),sync_service=composition.foundation.sync_service)
        with authority.database.connect() as conn:
            applied=consumer.consume_due(conn,worker_id="picture-retrieval-integration",now="2099-01-01T00:00:00+00:00",limit=20)
        assert len(applied)==2 and {item.status.value for item in applied}=={"applied"}
        retrieval=build_file_retrieval_runtime(session_id="session-1",scope_id="files-session-1",
            generation_spec=generation,file_foundation=composition.foundation,resolve_file=access.resolve_file,
            revalidate=access.revalidate,picture_file_authority=picture_authority)
        result=retrieval.retrieve({"queries":["Northbridge chart revenue"]})
        ToolSchemaCompiler().validate_output(retrieval.registration().spec.output_schema,result)
        assert {item["evidence_type"] for item in result["evidence"]}=={"document_chunk","picture_observation"}
        assert {item["snippet"] for item in result["evidence"]}=={text,"Northbridge chart revenue rises sharply in the final quarter."}
        chunk=next(item for item in result["evidence"] if item["evidence_type"]=="document_chunk")
        reader=build_file_chunk_reader_runtime(session_id="session-1",scope_id="files-session-1",
            resolve_file=access.resolve_file,revalidate=access.revalidate)
        exact=reader.read_chunks({"targets":[{"file_id":chunk["file_id"],"document_version_id":chunk["document_version_id"],"chunk_ids":[chunk["chunk_id"]]}]})
        ToolSchemaCompiler().validate_output(reader.registration().spec.output_schema,exact)
        assert exact["results"][0]["chunks"][0]["content"]==text
        assert mount_document(str(document["doc_id"]),"session-2")
        for session_id,expected in (("session-2",SourceAvailability.READY),("session-denied",SourceAvailability.BLOCKED)):
            other=PictureObservationSourceAdapter(access_authority=PictureRetrievalFileAuthority(
                session_id=session_id,authorized_file_bindings=lambda:()))
            result=other.open_retrieval_access(SourceFilter.from_mapping(SourceType.PICTURE,
                {"session_id":session_id,"file_id":source.file_id,"file_version_id":source.file_version_id}))
            assert result.availability is expected
