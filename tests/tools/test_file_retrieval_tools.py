"""Only the native File retrieval and exact Document read protocols are exposed."""

from dataclasses import replace
import hashlib

import pytest

from personagraph.tools.effects import EffectScopeKind
from personagraph.tools.execution import ToolExecutor
from personagraph.tools.retrieval.file_retrieval_catalog import (
    FileRetrievalBindingFacts, FileRetrievalCatalogError,
    build_file_retrieval_tool_definition_manifest, build_file_retrieval_tool_bindings,
)
from personagraph.tools.documents.file_chunk_catalog import (
    FileChunkBindingFacts, build_file_chunk_tool_definition_manifest, build_file_chunk_tool_bindings,
)
from personagraph.tools.retrieval.file_retrieval_tools import build_retrieve_files_registration
from personagraph.tools.documents.file_chunk_tools import build_read_file_chunks_registration
from personagraph.tools.schema_validation import ToolSchemaCompiler, SchemaValidationError


def test_definitions_are_stable_and_only_own_retrieval_or_exact_reading():
    retrieval=build_file_retrieval_tool_definition_manifest()
    reading=build_file_chunk_tool_definition_manifest()
    assert [item.definition.spec.tool_id for item in retrieval]==["retrieve_files"]
    assert [item.definition.spec.tool_id for item in reading]==["read_file_chunks"]
    assert retrieval==build_file_retrieval_tool_definition_manifest()
    assert reading==build_file_chunk_tool_definition_manifest()
    assert "candidate" not in str(retrieval[0].definition.spec.to_dict())
    assert "candidate" not in str(reading[0].definition.spec.to_dict())


def test_file_retrieval_budget_is_ninety_seconds_and_respects_earlier_deadlines():
    registration = build_retrieve_files_registration(
        handler=lambda _: {}, effect_scope="scope",
    )
    profile = registration.execution_profile
    assert profile.default_timeout_s == 90
    assert profile.hard_timeout_s == 90
    manifest = build_file_retrieval_tool_definition_manifest()
    assert manifest[0].definition.execution_profile == profile

    executor = ToolExecutor(clock=lambda: 100.0)
    assert executor._effective_timeout(registration, None) == 90
    assert executor._effective_timeout(registration, 500.0) == 90
    assert executor._effective_timeout(registration, 160.0) == 60


def test_contextual_bindings_match_exact_definitions_and_validate_scope():
    scope="file-corpus:test"
    digest=hashlib.sha256(scope.encode()).hexdigest()
    registration=build_retrieve_files_registration(handler=lambda _: {},effect_scope=scope,filesystem_scope_kind=EffectScopeKind.SESSION)
    facts=FileRetrievalBindingFacts(digest,"a"*64,"b"*64)
    assert len(build_file_retrieval_tool_bindings((registration,),facts=facts))==1
    with pytest.raises(FileRetrievalCatalogError):
        build_file_retrieval_tool_bindings((registration,),facts=replace(facts,scope_sha256="f"*64))
    with pytest.raises(FileRetrievalCatalogError):
        build_file_retrieval_tool_bindings((replace(registration,implementation_version="wrong"),),facts=facts)
    read=build_read_file_chunks_registration(handler=lambda _: {},effect_scope=scope,filesystem_scope_kind=EffectScopeKind.SESSION)
    assert len(build_file_chunk_tool_bindings((read,),facts=FileChunkBindingFacts(digest,"c"*64)))==1


def test_retrieval_schema_preserves_query_variants_and_adjustable_budgets():
    schema=build_retrieve_files_registration(handler=lambda _: {},effect_scope="scope").spec.input_schema
    compiler=ToolSchemaCompiler()
    compiler.validate_input(schema,{"queries":["主查询","source query","term query","sub question"],"file_ids":["file-a"],"result_limit":96,"context_token_limit":96000})
    compiler.validate_input(schema,{"queries":["default corpus"]})
    assert schema["properties"]["result_limit"]["default"] == 96
    assert schema["properties"]["context_token_limit"]["default"] == 96000
    for value in ({"queries":["x"],"file_ids":[]},{"query":"old"},{"queries":["x"],"candidate_ids":["old"]},{"queries":["x"],"limit":32},{"queries":["x"],"per_query_limit":32},{"queries":["x"],"result_limit":97}):
        with pytest.raises(SchemaValidationError):
            compiler.validate_input(schema, value)


def test_exact_reader_schema_requires_file_version_and_exclusive_selectors():
    schema=build_read_file_chunks_registration(handler=lambda _: {},effect_scope="scope").spec.input_schema
    compiler=ToolSchemaCompiler()
    base={"file_id":"file-a","document_version_id":"dv-a"}
    for selector in ({"chunk_ids":["chunk-a"]},{"chunk_sequences":[0,1]}):
        compiler.validate_input(schema,{"targets":[{**base,**selector}]})
    for selector in ({},{"chunk_ids":[]},{"chunk_ids":["c"],"chunk_sequences":[0]},{"chunk_sequences":[True]}):
        with pytest.raises(SchemaValidationError):
            compiler.validate_input(schema, {"targets": [{**base, **selector}]})
