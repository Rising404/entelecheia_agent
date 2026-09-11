"""File retrieval keeps exact shared identity, authority, and multi-query behavior."""

from dataclasses import replace
import hashlib
import json
from types import SimpleNamespace
import threading

import pytest

from personagraph.input_processing.files import SourceFingerprint
from personagraph.retrieval.contracts import RetrievalQueryMatch, SourceType
from personagraph.retrieval.lifecycle.generation import file_corpus_generation_spec, DOCUMENT_HYBRID_INDEX_RECIPE_CONTRACT
from personagraph.retrieval.profile import durable_chunking_profile
from personagraph.retrieval.sources.identity import mounted_document_chunk_ref_and_content
from personagraph.retrieval.tooling.contracts import (
    FileRetrievalScope, RetrievalStatus, RetrievalToolEvidence, RetrievalToolResult,
)
from personagraph.tools.documents.file_chunk_reader import build_file_chunk_reader_runtime
from personagraph.tools.contracts import ExecutionStatus
from personagraph.tools.execution import ResolvedInvocation, ToolBusinessFailure, ToolExecutor
from personagraph.tools.retrieval.file_retrieval_adapter import build_file_retrieval_runtime, _picture_public_locator, _response
from personagraph.tools.retrieval.file_retrieval_tools import build_retrieve_files_registration
from personagraph.tools.schema_validation import ToolSchemaCompiler
from personagraph.workspace.documents.reading import CurrentFileChunk, CurrentFileDocument


def _fixture(content="exact prepared text"):
    digest = hashlib.sha256(content.encode()).hexdigest()
    source = SimpleNamespace(file_id="file-a", file_version_id="fv-a", project_id="project-a",
                             file_name="brief.md", relative_path="reports/brief.md", origin="workspace_existing",
                             fingerprint=SourceFingerprint(sha256="a"*64, size_bytes=40, mtime_ns=1788566400000000000))
    document = CurrentFileDocument("file-a", "fv-a", "doc-a", "dv-a", "a"*64, 1, "complete", None, "2026-09-05T00:01:00+00:00")
    chunk = CurrentFileChunk(document, "chunk-a", "producer-a", 0, content, digest, "page 1", (1,))
    generation = file_corpus_generation_spec(
        encoder_fingerprint="test-encoder", document_chunker_fingerprint=durable_chunking_profile().fingerprint(),
        document_chunk_contract_version=1,
        index_recipe=DOCUMENT_HYBRID_INDEX_RECIPE_CONTRACT,
    )
    return source, document, chunk, generation


def _runtime(*, session_id="session-a", content="exact prepared text", mode="ready", extra=None):
    source, document, chunk, generation = _fixture(content)
    requests, records = [], []
    def resolve(**kwargs):
        return source if kwargs["file_id"] == source.file_id else None
    def factory(bridge):
        def retrieve(request):
            requests.append(request)
            if request.file_bindings:
                readiness = bridge.ensure_ready(request, request.file_bindings[0])
                if readiness.status.value != "ready":
                    return RetrievalToolResult(status=RetrievalStatus.PARTIAL,
                                               scope_snapshot_id=request.scope_snapshot_id,
                                               retrieval_data_version=request.retrieval_data_version)
            ref, normalized = mounted_document_chunk_ref_and_content(
                session_id=session_id, chunk_id=chunk.chunk_id,
                source_version_id=document.document_version_id, content=chunk.content,
                doc_id=document.document_id, producer_chunk_id=chunk.producer_chunk_id,
            )
            evidence = RetrievalToolEvidence(
                source_type=SourceType.DOCUMENT, source_unit_id=ref.source_unit_id,
                source_revision=ref.source_revision, indexed_content_hash=ref.indexed_content_hash,
                content=normalized, estimated_tokens=5, rank=1, query_index=1 if len(request.queries)>1 else 0,
                query_matches=(RetrievalQueryMatch(1 if len(request.queries)>1 else 0, 1, 1),),
                authority_id="file-a" if request.file_bindings else None,
                citation={"file_id":"file-a", "file_version_id":"fv-a", "doc_id":"doc-a",
                          "chunk_id":"chunk-a", "producer_chunk_id":"producer-a", "location":chunk.locator},
            )
            if mode == "forged":
                evidence = replace(evidence, content="forged", indexed_content_hash=hashlib.sha256(b"forged").hexdigest())
            return RetrievalToolResult(status=RetrievalStatus.COMPLETE, scope_snapshot_id=request.scope_snapshot_id,
                                       retrieval_data_version=request.retrieval_data_version, evidence=(evidence,))
        return SimpleNamespace(retrieve_file_readonly_unrecorded=lambda request: (
            retrieve(request), SimpleNamespace(record=lambda **kwargs: records.append(kwargs))))
    kwargs = dict(session_id=session_id, scope_id="scope-a", generation_spec=generation,
                  resolve_file=resolve, revalidate=lambda _:mode!="revoked", port_factory=factory,
                  read_document=lambda **_:None if mode=="pending" else document,
                  read_chunk=lambda **_:chunk)
    kwargs.update(extra or {})
    return build_file_retrieval_runtime(**kwargs), requests, records, source, document, chunk


def test_selected_retrieval_preserves_multi_queries_limits_source_metadata_and_audit():
    runtime, requests, records, *_ = _runtime()
    result = runtime.retrieve({"queries":["brief", "摘要"], "file_ids":["file-a"], "result_limit":24, "context_token_limit":18000})
    assert requests[0].file_scope is FileRetrievalScope.SELECTED_FILES
    assert requests[0].queries == ("brief", "摘要")
    assert (requests[0].limit, requests[0].context_token_limit)==(24,18000)
    item = result["evidence"][0]
    assert (item["file_id"],item["file_version_id"],item["document_id"],item["document_version_id"],item["chunk_id"]) == ("file-a","fv-a","doc-a","dv-a","chunk-a")
    assert item["query_index"] == 1
    assert item["source"] == {"file_name":"brief.md", "relative_path":"reports/brief.md", "origin":"workspace",
                              "source_modified_at":"2026-09-05T00:00:00+00:00", "corpus_recorded_at":"2026-09-05T00:01:00+00:00"}
    assert records[0]["final_projection"] == result


def test_retrieval_audit_waits_for_tool_output_schema_settlement():
    runtime, _, records, *_ = _runtime()
    registration = runtime.registration()
    invalid_output_spec = replace(
        registration.spec,
        output_schema={"type": "object", "required": ["impossible_field"]},
    )
    registration = replace(registration, spec=invalid_output_spec)

    outcome = ToolExecutor().execute(ResolvedInvocation(registration, {"queries": ["brief"]}))

    assert outcome.status is ExecutionStatus.FAILED
    assert len(records) == 1
    assert records[0]["final_projection"]["evidence"] == []
    assert records[0]["diagnostics"][0]["code"] == "tool_failed"


def test_expiry_during_authority_projection_cannot_record_success_evidence():
    from personagraph.tools.execution_context import current_tool_execution

    now = [100.0]
    runtime, _, records, _, _, chunk = _runtime()

    def read_chunk(**kwargs):
        invocation = current_tool_execution()
        assert invocation is not None and invocation.deadline_monotonic is not None
        now[0] = invocation.deadline_monotonic + 1.0
        return chunk

    runtime = replace(runtime, read_chunk=read_chunk)
    parent_cancel = threading.Event()
    outcome = ToolExecutor(clock=lambda: now[0]).execute(ResolvedInvocation(
        runtime.registration(), {"queries": ["brief"]}, cancellation_event=parent_cancel,
    ))

    assert outcome.status is ExecutionStatus.TIMED_OUT
    assert not parent_cancel.is_set()
    assert len(records) == 1
    assert records[0]["final_projection"]["evidence"] == []
    assert records[0]["diagnostics"][0]["code"] == "tool_timed_out"


def test_tool_timeout_reaches_the_retrieval_control_port_without_backend_retry():
    from personagraph.retrieval.execution import checkpoint, current_execution
    from personagraph.tools.execution_context import current_tool_execution

    now = [100.0]
    acknowledged = threading.Event()
    calls = []

    def factory(_bridge):
        def retrieve(_request):
            calls.append(1)
            assert current_execution() is not None
            invocation = current_tool_execution()
            assert invocation is not None
            assert invocation.deadline_monotonic is not None
            now[0] = invocation.deadline_monotonic + 1.0
            assert invocation.cancellation_event.wait(timeout=1.0)
            try:
                checkpoint()
            finally:
                acknowledged.set()
        return SimpleNamespace(retrieve=retrieve)

    runtime, *_ = _runtime(extra={"port_factory": factory})
    outcome = ToolExecutor(clock=lambda: now[0]).execute(ResolvedInvocation(
        runtime.registration(), {"queries": ["brief"]},
    ))

    assert outcome.status is ExecutionStatus.TIMED_OUT
    assert outcome.error.code == "execution_timeout"
    assert acknowledged.wait(timeout=1.0)
    assert calls == [1]


def test_backend_cancellation_audit_waits_for_final_handler_cleanup(monkeypatch):
    from personagraph.retrieval.contracts import CorpusKey
    from personagraph.retrieval.execution import checkpoint
    from personagraph.retrieval.tooling.service import facade
    from personagraph.tools.execution_context import current_tool_execution

    now = [100.0]
    recorded = threading.Event()
    records = []

    def backend(*args, **kwargs):
        invocation = current_tool_execution()
        assert invocation is not None and invocation.deadline_monotonic is not None
        now[0] = invocation.deadline_monotonic + 1.0
        assert invocation.cancellation_event.wait(timeout=1.0)
        checkpoint()

    def factory(_bridge):
        foundation = SimpleNamespace(
            corpus_key=CorpusKey.FILE,
            service=SimpleNamespace(
                retrieve_context=backend, retrieve_context_readonly=backend,
                retrieve_file_query_batch=backend,
            ),
            data_version_provider=SimpleNamespace(
                active_retrieval_data_version_id=lambda: runtime.generation_spec.version_id,
            ),
        )
        return facade.RetrievalServiceToolPort(file_foundation=foundation, file_readonly=True)

    def audit(**kwargs):
        records.append((kwargs, kwargs["audit"].execution.snapshot()))
        recorded.set()

    monkeypatch.setattr(facade, "_record_query_audit", audit)
    runtime, *_ = _runtime(extra={"port_factory": factory})
    outcome = ToolExecutor(clock=lambda: now[0]).execute(ResolvedInvocation(
        runtime.registration(), {"queries": ["brief"]},
    ))
    assert outcome.status is ExecutionStatus.TIMED_OUT
    assert recorded.wait(timeout=1.0)
    assert len(records) == 1
    assert records[0][0]["final_projection"]["evidence"] == []
    assert records[0][1]["cancellation_acknowledged"] is True
    assert records[0][1]["handler_finished"] is True


def test_global_retrieval_chunk_can_be_read_directly_and_identity_is_session_independent():
    runtime, requests, _, source, document, chunk = _runtime()
    result = runtime.retrieve({"queries":["brief"]})
    assert requests[0].file_scope is FileRetrievalScope.SESSION_CORPUS
    item=result["evidence"][0]
    reader=build_file_chunk_reader_runtime(session_id="session-a",scope_id="scope-a",
        resolve_file=lambda **_:source,revalidate=lambda _:True,
        read_document=lambda **_:document,read_chunk=lambda **_:chunk)
    read=reader.read_chunks({"targets":[{"file_id":item["file_id"],"document_version_id":item["document_version_id"],"chunk_ids":[item["chunk_id"]]}]})
    assert read["results"][0]["chunks"][0]["content"] == chunk.content
    other, *_ = _runtime(session_id="session-b")
    assert other.retrieve({"queries":["brief"]})["evidence"][0] == item


@pytest.mark.parametrize("payload", [
    {"queries":["x"],"file_ids":[]}, {"query":"x"},
    {"queries":["x"],"candidate_ids":["candidate_a"]},
    {"queries":["x"," x "]}, {"queries":["x"]*5},
    {"queries":["x"],"limit":32}, {"queries":["x"],"per_query_limit":32},
    {"queries":["x"],"result_limit":97},
    {"queries":["x"],"context_token_limit":96001},
])
def test_invalid_protocol_never_calls_backend(payload):
    runtime, requests, *_ = _runtime()
    with pytest.raises(ToolBusinessFailure):
        runtime.retrieve(payload)
    assert requests == []


def test_unknown_or_revoked_files_do_not_expand_to_global_corpus():
    for mode, file_id in (("ready","unknown"),("revoked","file-a")):
        runtime, requests, *_=_runtime(mode=mode)
        result=runtime.retrieve({"queries":["x"],"file_ids":[file_id]})
        assert result["status"]=="blocked" and result["evidence"]==[]
        assert requests==[]


def test_pending_file_does_not_schedule_preparation_and_forged_evidence_is_dropped():
    for mode in ("pending","forged"):
        runtime,*_=_runtime(mode=mode)
        result=runtime.retrieve({"queries":["x"],"file_ids":["file-a"]})
        assert result["evidence"]==[] and result["status"]=="partial"


def test_canonical_index_normalization_does_not_change_exact_returned_text():
    text="\n  | Metric | Value |\n  | --- | --- |\n  "
    runtime,*_=_runtime(content=text)
    assert runtime.retrieve({"queries":["metric"]})["evidence"][0]["snippet"]==text


def test_picture_public_locator_preserves_one_based_surfaces():
    assert _picture_public_locator({"surface_kind":"pdf_page","surface_ordinal":"1"})=="page 1"
    assert _picture_public_locator({"surface_kind":"pptx_slide","surface_ordinal":"3"})=="slide 3"
    assert _picture_public_locator({"surface_kind":"pdf_page","surface_ordinal":"0"}) is None


def test_four_queries_default_to_96_total_and_96000_text_tokens():
    runtime, requests, *_ = _runtime()
    result = runtime.retrieve({"queries": ["one", "two", "three", "four"]})
    assert (requests[0].limit, requests[0].context_token_limit) == (96, 96000)
    registration = build_retrieve_files_registration(handler=lambda _: {}, effect_scope="scope")
    assert registration.execution_profile.max_output_bytes == 1024 * 1024
    ToolSchemaCompiler().validate_output(registration.spec.output_schema, result)


def test_old_limit_is_rejected_with_an_explicit_migration_hint():
    runtime, requests, *_ = _runtime()
    with pytest.raises(ToolBusinessFailure, match="result_limit"):
        runtime.retrieve({"queries": ["one"], "limit": 12})
    assert requests == []


def test_snippet_cut_is_explicit_without_changing_authoritative_hash():
    content = "界" * 20000
    runtime, *_ = _runtime(content=content)
    result = runtime.retrieve({"queries": ["text"]})
    item = result["evidence"][0]
    assert item["snippet_truncated"] and item["content_character_count"] == len(content)
    assert item["content_sha256"] == hashlib.sha256(content.encode()).hexdigest()
    assert len(item["snippet"]) == 16000
    assert result["truncated"] and result["status"] == "partial"


def test_utf8_response_budget_returns_explicit_partial_instead_of_tool_failure():
    runtime, *_ = _runtime(content="界" * 20000)
    item = runtime.retrieve({"queries": ["text"]})["evidence"][0]
    evidence = [{**item, "rank": index + 1} for index in range(128)]
    result = _response(("text",), FileRetrievalScope.SESSION_CORPUS, evidence, [], 1, 1)
    encoded = json.dumps(result, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode()
    assert len(encoded) <= 1024 * 1024
    assert 0 < len(result["evidence"]) < 128
    assert result["status"] == "partial" and result["truncated"]
    gap = next(gap for gap in result["gaps"] if gap["code"] == "response_byte_limit_exceeded")
    assert gap["known_count"] == 128 - len(result["evidence"])
    assert result["coverage"]["returned_evidence_count"] == len(result["evidence"])
