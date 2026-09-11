"""挂载文档模型认知工具的专项契约测试。"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace

import pytest

from personagraph.session import store
from personagraph.input_processing.documents import (
    ChunkSpan,
    DocumentChunk,
    DocumentLocator,
    ElementKind,
)
from personagraph.input_processing.files import fingerprint_file
from personagraph.workspace.files import (
    FileSource,
    WorkspaceFileAuthority,
)
from personagraph.workspace.documents import application as docstore
from personagraph.workspace.storage.context import current
from personagraph.l2.auxiliary_execution.planning.mounted_document_authority import (
    MountedDocumentPlanningAuthorityError,
    freeze_mounted_document_planning_authority,
)
from personagraph.l2.task_execution.attempts.controller import (
    AttemptToolBridgePreflightRequest,
)
from personagraph.tools.documents.mounted_document_cognition_tools import (
    MOUNTED_DOCUMENT_COGNITION_CAPABILITY,
    FrozenMountedDocumentToolScope,
)
from personagraph.tools.documents.mounted_document_catalog import (
    build_mounted_document_binding_facts,
    build_mounted_document_tool_definition_manifest,
)
from personagraph.l2.task_execution.tool_bridge.mounted_document_adapter import (
    SessionMountedDocumentCognitionRuntime,
    build_session_mounted_document_cognition_runtime,
)
from personagraph.l2.task_execution.task_node.document_tool_runtime import (
    compose_task_node_document_tool_runtime,
)
from personagraph.tools.catalog.binding import BoundToolRegistration
from personagraph.tools.contracts import ExecutionStatus
from personagraph.tools.effects import (
    EffectAction,
    EffectResource,
    EffectScopeKind,
)
from personagraph.tools.execution import ResolvedInvocation, ToolExecutor
from personagraph.l2.work_run import (
    AttemptDecision,
    CallToolsAction,
    ToolCallProposal,
)


SESSION_ID = "session-mounted-cognition"
DOCUMENT_ID = "private-document-id-must-not-leak"
DOCUMENT_VERSION = "document-version-frozen-001"
PRIVATE_PATH = "/private/customer/secret-long-document.txt"
SOURCE_SHA256 = hashlib.sha256(b"frozen long document").hexdigest()
DOCUMENT_ALIAS = "mounted_document_01"
TOTAL_CHUNKS = 310


def _chunk(sequence: int) -> docstore.CurrentDocumentResourceChunk:
    if sequence == 200:
        content = "本文主要贡献包括一种可靠的长文档分块读取方法。"
    elif sequence == TOTAL_CHUNKS - 1:
        content = "terminal needle appears only in the final frozen block"
    else:
        content = f"ordinary evidence block {sequence}"
    return docstore.CurrentDocumentResourceChunk(
        storage_chunk_id=f"storage-{sequence:04d}",
        producer_chunk_id=f"producer-{sequence:04d}",
        sequence=sequence,
        locator=f"{PRIVATE_PATH}#private-chunk={sequence}",
        content=content,
        content_sha256=hashlib.sha256(content.encode("utf-8")).hexdigest(),
        source_pages=(sequence // 2 + 1,),
    )


@dataclass
class _MutableMountedDocument:
    chunks: tuple[docstore.CurrentDocumentResourceChunk, ...] = field(
        default_factory=lambda: tuple(_chunk(index) for index in range(TOTAL_CHUNKS))
    )
    mounted: bool = True
    version: str = DOCUMENT_VERSION
    source_sha256: str = SOURCE_SHA256
    processing_status: str = "complete"
    diagnostic_codes: tuple[str, ...] = ()
    calls: list[tuple[str, str, int, int]] = field(default_factory=list)
    freshness_calls: list[tuple[str, str]] = field(default_factory=list)

    def snapshot(
        self,
        document_id: str,
        *,
        session_id: str,
        start_sequence: int = 0,
        maximum_chunks: int,
    ) -> docstore.CurrentDocumentResourceSnapshot | None:
        self.calls.append(
            (document_id, session_id, start_sequence, maximum_chunks)
        )
        if not self.mounted:
            return None
        selected = self.chunks[start_sequence : start_sequence + maximum_chunks]
        return docstore.CurrentDocumentResourceSnapshot(
            session_id=session_id,
            document_id=document_id,
            document_version_id=self.version,
            source_sha256=self.source_sha256,
            file_extension=".txt",
            processing_status=self.processing_status,
            processing_diagnostic_codes=self.diagnostic_codes,
            total_chunk_count=len(self.chunks),
            physical_page_count=None,
            page_inventory_status="unavailable",
            chunks=selected,
            truncated=(
                start_sequence > 0
                or start_sequence + maximum_chunks < len(self.chunks)
            ),
        )

    def freshness(self, session_id: str, document_id: str | None = None):
        assert document_id is not None
        self.freshness_calls.append((session_id, document_id))
        if not self.mounted:
            return {
                "ok": False,
                "status": "freshness_blocked",
                "documents": [
                    {"doc_id": document_id, "status": "not_mounted"}
                ],
            }
        return {
            "ok": True,
            "status": "verified_current",
            "documents": [
                {
                    "doc_id": document_id,
                    "version_id": self.version,
                    "status": "verified_current",
                }
            ],
        }


@pytest.fixture
def mounted_runtime(monkeypatch):
    source = _MutableMountedDocument()
    monkeypatch.setattr(
        docstore,
        "mounted_docs",
        lambda session_id: (
            ({"id": DOCUMENT_ID, "path": PRIVATE_PATH},)
            if session_id == SESSION_ID and source.mounted
            else ()
        ),
    )
    monkeypatch.setattr(
        docstore,
        "get_mounted_current_document_resource_snapshot",
        source.snapshot,
    )
    monkeypatch.setattr(
        docstore,
        "get_current_document_page_authority",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        docstore,
        "check_mounted_document_freshness",
        source.freshness,
    )
    authority = freeze_mounted_document_planning_authority(
        session_id=SESSION_ID
    )
    runtime = build_session_mounted_document_cognition_runtime(authority)
    source.calls.clear()
    source.freshness_calls.clear()
    return source, runtime


def _execute(runtime, tool_id: str, arguments: dict[str, object]):
    registration = runtime.catalog_snapshot.resolve(tool_id)
    return ToolExecutor().execute(
        ResolvedInvocation(registration=registration, arguments=arguments)
    )


def _ingest_real_text_source(
    source,
    *,
    content: str,
    session_id: str,
) -> str:
    database = current()
    if database is None:
        raise RuntimeError("test document ingest requires a Project database")
    relative_path = source.resolve().relative_to(database.project_root)
    registration = WorkspaceFileAuthority(database).ensure_current_path(
        relative_path,
        source=FileSource.WORKSPACE_EXISTING,
        media_type="text/plain",
    )
    chunk = DocumentChunk(
        chunk_id="external-source-producer-chunk",
        text=content,
        span=ChunkSpan(
            start=DocumentLocator(page=1, ordinal=0),
            end=DocumentLocator(page=1, ordinal=0),
        ),
        section_path=("Document",),
        element_ids=("external-source-element",),
        token_count=max(1, len(content.split())),
        kind=ElementKind.PARAGRAPH,
        source_pages=(1,),
    )
    stored = docstore.ingest(
        str(source),
        source.stem,
        "text/plain",
        [{"content": content, "loc": chunk.loc}],
        session_id=session_id,
        file_id=registration.file.file_id,
        file_version_id=registration.version.file_version_id,
        source_fingerprint=fingerprint_file(source),
        processor_fingerprint="mounted-cognition-test-reader@1",
        document_chunks=(chunk,),
        chunker_fingerprint="mounted-cognition-test-chunker@1",
        processing_status="complete",
        processing_diagnostics=(),
    )
    return str(stored["doc_id"])


def test_runtime_exports_closed_session_read_and_search_catalog(
    mounted_runtime,
) -> None:
    _source, runtime = mounted_runtime

    assert isinstance(runtime, SessionMountedDocumentCognitionRuntime)
    assert MOUNTED_DOCUMENT_COGNITION_CAPABILITY == (
        "mounted_document_cognition"
    )
    assert runtime.tool_ids == (
        "inspect_mounted_document",
        "search_mounted_document",
        "read_mounted_document_chunks",
    )
    assert {
        entry.registration.tool_id
        for entry in runtime.catalog_snapshot.exposed()
    } == set(runtime.tool_ids)
    assert len(runtime.scope_snapshot_sha256) == 64

    for entry in runtime.catalog_snapshot.exposed():
        (effect,) = entry.registration.effect_profile.effects
        assert effect.resource is EffectResource.FILESYSTEM
        assert effect.action in {EffectAction.READ, EffectAction.SEARCH}
        assert effect.scope_kind is EffectScopeKind.SESSION
        assert effect.default_scope == SESSION_ID

    public_descriptor = json.dumps(
        runtime.catalog_snapshot.to_descriptor(),
        ensure_ascii=False,
        sort_keys=True,
    )
    assert DOCUMENT_ID not in public_descriptor
    assert PRIVATE_PATH not in public_descriptor


def test_runtime_exposes_exact_bindings_for_its_single_frozen_source(
    mounted_runtime,
) -> None:
    _source, _initial_runtime = mounted_runtime
    authority = freeze_mounted_document_planning_authority(
        session_id=SESSION_ID
    )
    scope = FrozenMountedDocumentToolScope(
        session_id=authority.session_id,
        scope_snapshot_sha256=authority.scope_snapshot_sha256,
        documents=tuple(
            binding.frozen_document for binding in authority.bindings
        ),
    )
    expected_facts = build_mounted_document_binding_facts(scope)

    runtime = build_session_mounted_document_cognition_runtime(authority)

    assert runtime is not None
    assert tuple(
        binding.identity.tool_id for binding in runtime.contextual_bindings
    ) == runtime.tool_ids
    definitions = {
        item.definition.identity.tool_id: item.definition
        for item in build_mounted_document_tool_definition_manifest()
    }
    for binding in runtime.contextual_bindings:
        registration = runtime.catalog_snapshot.resolve(
            binding.identity.tool_id
        )
        rebound = BoundToolRegistration(
            definitions[binding.identity.tool_id],
            binding,
        )
        assertion = binding.binding_assertion

        assert rebound.descriptor() == registration.descriptor()
        assert rebound.handler is registration.handler
        assert assertion["session_scope_sha256"] == (
            expected_facts.session_scope_sha256
        )
        assert assertion["scope_snapshot_sha256"] == (
            expected_facts.scope_snapshot_sha256
        )
        assert assertion["document_alias_projection_sha256"] == (
            expected_facts.document_alias_projection_sha256
        )
        assert assertion["document_generation_snapshot_sha256"] == (
            expected_facts.document_generation_snapshot_sha256
        )
        assert assertion["document_freshness_snapshot_sha256"] == (
            expected_facts.document_freshness_snapshot_sha256
        )


def test_task_node_runtime_preserves_mounted_document_binding_candidates(
    mounted_runtime,
) -> None:
    _source, mounted = mounted_runtime

    runtime = compose_task_node_document_tool_runtime(
        session_id=SESSION_ID,
        workspace_runtime=None,
        workspace_tool_bridge=None,
        mounted_runtime=mounted,
        ledger_store=store,
    )

    assert {
        binding.identity.tool_id for binding in runtime.contextual_bindings
    } == set(mounted.tool_ids)
    assert runtime.contextual_bindings == mounted.contextual_bindings
    definitions = {
        item.definition.identity.tool_id: item.definition
        for item in build_mounted_document_tool_definition_manifest()
    }
    for binding in runtime.contextual_bindings:
        source_registration = mounted.catalog_snapshot.resolve(
            binding.identity.tool_id
        )
        rebound = BoundToolRegistration(
            definitions[binding.identity.tool_id],
            binding,
        )
        assert rebound.handler is source_registration.handler
        assert rebound.descriptor() == source_registration.descriptor()


def test_runtime_bridge_preflight_accepts_its_session_scoped_read(
    mounted_runtime,
) -> None:
    _source, runtime = mounted_runtime
    allowed_tools = tuple(
        entry.registration.spec for entry in runtime.catalog_snapshot.exposed()
    )

    accepted = runtime.tool_bridge.preflight(
        AttemptToolBridgePreflightRequest(
            session_id=SESSION_ID,
            turn_id="turn-mounted-cognition",
            work_run_id="workrun-mounted-cognition",
            attempt_id="attempt-mounted-cognition",
            decision=AttemptDecision(
                action=CallToolsAction(
                    calls=(
                        ToolCallProposal(
                            tool_id="inspect_mounted_document",
                            arguments={"document_alias": DOCUMENT_ALIAS},
                        ),
                    )
                )
            ),
            tool_call_ids=("call-mounted-cognition",),
            allowed_tools=allowed_tools,
            catalog_snapshot=runtime.catalog_snapshot.to_descriptor(),
        )
    )

    assert accepted.action.calls[0].modifies_environment is False


def test_inspect_returns_frozen_generation_count_and_processing_coverage(
    mounted_runtime,
) -> None:
    source, runtime = mounted_runtime

    outcome = _execute(
        runtime,
        "inspect_mounted_document",
        {"document_alias": DOCUMENT_ALIAS},
    )

    assert outcome.status is ExecutionStatus.SUCCEEDED
    assert dict(outcome.result or {}) == {
        "document_alias": DOCUMENT_ALIAS,
        "resource_version": DOCUMENT_VERSION,
        "content_sha256": SOURCE_SHA256,
        "resource_format": "txt",
        "processing_coverage": "complete",
        "processing_diagnostic_codes": (),
        "total_chunk_count": TOTAL_CHUNKS,
    }
    assert source.calls == [(DOCUMENT_ID, SESSION_ID, 0, 1)]


def test_read_returns_exact_late_window_and_stable_continuation_fields(
    mounted_runtime,
) -> None:
    source, runtime = mounted_runtime

    outcome = _execute(
        runtime,
        "read_mounted_document_chunks",
        {
            "document_alias": DOCUMENT_ALIAS,
            "start_sequence": 255,
            "limit": 64,
        },
    )

    assert outcome.status is ExecutionStatus.SUCCEEDED
    result = dict(outcome.result or {})
    assert result["returned_start_sequence"] == 255
    assert result["returned_end_sequence_exclusive"] == 310
    assert result["next_start_sequence"] is None
    assert result["complete"] is True
    assert result["total_chunk_count"] == TOTAL_CHUNKS
    assert [item["sequence"] for item in result["chunks"]] == list(
        range(255, 310)
    )
    assert result["chunks"][-1] == {
        "sequence": 309,
        "locator": (
            "resource:mounted_document_01#page=155&chunk=309"
        ),
        "source_pages": (155,),
        "content": "terminal needle appears only in the final frozen block",
        "content_sha256": hashlib.sha256(
            b"terminal needle appears only in the final frozen block"
        ).hexdigest(),
    }
    assert source.calls == [(DOCUMENT_ID, SESSION_ID, 255, 64)]


def test_read_cursor_walks_the_complete_document_without_gaps_or_duplicates(
    mounted_runtime,
) -> None:
    source, runtime = mounted_runtime
    cursor = 0
    observed_sequences: list[int] = []

    while True:
        outcome = _execute(
            runtime,
            "read_mounted_document_chunks",
            {
                "document_alias": DOCUMENT_ALIAS,
                "start_sequence": cursor,
                "limit": 64,
            },
        )

        assert outcome.status is ExecutionStatus.SUCCEEDED
        result = dict(outcome.result or {})
        observed_sequences.extend(
            int(item["sequence"]) for item in result["chunks"]
        )
        if result["complete"]:
            assert result["next_start_sequence"] is None
            break
        cursor = int(result["next_start_sequence"])

    assert observed_sequences == list(range(TOTAL_CHUNKS))
    assert [call[2] for call in source.calls] == [0, 64, 128, 192, 256]


def test_read_limits_the_serialized_projection_and_returns_a_safe_cursor(
    mounted_runtime,
) -> None:
    source, runtime = mounted_runtime
    chunks = list(source.chunks)
    escaped_content = "\\" * 40_000
    escaped_sha256 = hashlib.sha256(escaped_content.encode("utf-8")).hexdigest()
    for sequence in (0, 1):
        chunks[sequence] = replace(
            chunks[sequence],
            content=escaped_content,
            content_sha256=escaped_sha256,
        )
    source.chunks = tuple(chunks)

    outcome = _execute(
        runtime,
        "read_mounted_document_chunks",
        {
            "document_alias": DOCUMENT_ALIAS,
            "start_sequence": 0,
            "limit": 2,
        },
    )

    assert outcome.status is ExecutionStatus.SUCCEEDED
    result = dict(outcome.result or {})
    assert [item["sequence"] for item in result["chunks"]] == [0]
    assert result["next_start_sequence"] == 1
    assert result["complete"] is False


def test_search_scans_every_window_and_finds_match_after_chunk_256(
    mounted_runtime,
) -> None:
    source, runtime = mounted_runtime

    outcome = _execute(
        runtime,
        "search_mounted_document",
        {
            "document_alias": DOCUMENT_ALIAS,
            "query": "terminal needle",
            "limit": 20,
        },
    )

    assert outcome.status is ExecutionStatus.SUCCEEDED
    result = dict(outcome.result or {})
    assert result["document_alias"] == DOCUMENT_ALIAS
    assert result["query"] == "terminal needle"
    assert result["scanned_chunk_count"] == TOTAL_CHUNKS
    assert result["total_chunk_count"] == TOTAL_CHUNKS
    assert result["search_complete"] is True
    assert result["total_match_count"] == 1
    assert result["matches_truncated"] is False
    assert [item["sequence"] for item in result["matches"]] == [309]
    assert result["matches"][0]["locator"] == (
        "resource:mounted_document_01#page=155&chunk=309"
    )
    assert [call[2] for call in source.calls] == [0, 256]
    assert all(call[3] <= 256 for call in source.calls)


@pytest.mark.parametrize(
    "tool_id,arguments,expected_snapshot_starts",
    (
        (
            "inspect_mounted_document",
            {"document_alias": DOCUMENT_ALIAS},
            [0],
        ),
        (
            "search_mounted_document",
            {
                "document_alias": DOCUMENT_ALIAS,
                "query": "terminal needle",
                "limit": 5,
            },
            [0, 256],
        ),
        (
            "read_mounted_document_chunks",
            {
                "document_alias": DOCUMENT_ALIAS,
                "start_sequence": 255,
                "limit": 5,
            },
            [255],
        ),
    ),
)
def test_each_public_tool_call_hashes_physical_source_once(
    mounted_runtime,
    tool_id: str,
    arguments: dict[str, object],
    expected_snapshot_starts: list[int],
) -> None:
    source, runtime = mounted_runtime

    outcome = _execute(runtime, tool_id, arguments)

    assert outcome.status is ExecutionStatus.SUCCEEDED
    assert source.freshness_calls == [(SESSION_ID, DOCUMENT_ID)]
    assert [call[2] for call in source.calls] == expected_snapshot_starts


def test_search_matches_chinese_question_against_a_paraphrased_passage(
    mounted_runtime,
) -> None:
    _source, runtime = mounted_runtime

    outcome = _execute(
        runtime,
        "search_mounted_document",
        {
            "document_alias": DOCUMENT_ALIAS,
            "query": "这篇论文的主要贡献是什么",
            "limit": 5,
        },
    )

    assert outcome.status is ExecutionStatus.SUCCEEDED
    result = dict(outcome.result or {})
    assert [item["sequence"] for item in result["matches"]] == [200]
    assert result["search_complete"] is True


def test_search_reports_all_matches_but_returns_only_the_best_bounded_set(
    mounted_runtime,
) -> None:
    _source, runtime = mounted_runtime

    outcome = _execute(
        runtime,
        "search_mounted_document",
        {
            "document_alias": DOCUMENT_ALIAS,
            "query": "ordinary evidence",
            "limit": 3,
        },
    )

    assert outcome.status is ExecutionStatus.SUCCEEDED
    result = dict(outcome.result or {})
    assert result["total_match_count"] == TOTAL_CHUNKS - 2
    assert result["matches_truncated"] is True
    assert [item["sequence"] for item in result["matches"]] == [0, 1, 2]


@pytest.mark.parametrize(
    "tool_id,arguments",
    (
        (
            "inspect_mounted_document",
            {"document_alias": DOCUMENT_ALIAS},
        ),
        (
            "search_mounted_document",
            {
                "document_alias": DOCUMENT_ALIAS,
                "query": "needle",
                "limit": 5,
            },
        ),
        (
            "read_mounted_document_chunks",
            {
                "document_alias": DOCUMENT_ALIAS,
                "start_sequence": 0,
                "limit": 5,
            },
        ),
    ),
)
def test_every_call_fails_closed_when_current_generation_drifts(
    mounted_runtime,
    tool_id: str,
    arguments: dict[str, object],
) -> None:
    source, runtime = mounted_runtime
    source.version = "document-version-replaced-002"

    outcome = _execute(runtime, tool_id, arguments)

    assert outcome.status is ExecutionStatus.FAILED
    assert outcome.error is not None
    assert outcome.error.code == "mounted_document_authority_drift"
    serialized = json.dumps(outcome.error.to_dict(), sort_keys=True)
    assert DOCUMENT_ID not in serialized
    assert PRIVATE_PATH not in serialized


def test_external_source_replacement_blocks_all_tools_and_future_freeze(
    tmp_path,
    bound_partitioned_session,
) -> None:
    session_id = bound_partitioned_session(working_dir=tmp_path)
    source = tmp_path / "external-report.txt"
    original = "old external evidence\n" * 100
    source.write_text(original, encoding="utf-8")
    document_id = _ingest_real_text_source(
        source,
        content=original,
        session_id=session_id,
    )
    authority = freeze_mounted_document_planning_authority(
        session_id=session_id
    )
    runtime = build_session_mounted_document_cognition_runtime(authority)
    assert runtime is not None

    source.write_text("replacement external evidence\n" * 100, encoding="utf-8")

    calls = (
        (
            "inspect_mounted_document",
            {"document_alias": DOCUMENT_ALIAS},
        ),
        (
            "search_mounted_document",
            {
                "document_alias": DOCUMENT_ALIAS,
                "query": "old external evidence",
                "limit": 5,
            },
        ),
        (
            "read_mounted_document_chunks",
            {
                "document_alias": DOCUMENT_ALIAS,
                "start_sequence": 0,
                "limit": 5,
            },
        ),
    )
    for tool_id, arguments in calls:
        outcome = _execute(runtime, tool_id, arguments)
        assert outcome.status is ExecutionStatus.FAILED
        assert outcome.error is not None
        assert outcome.error.code == "mounted_document_authority_drift"
        serialized = json.dumps(outcome.error.to_dict(), sort_keys=True)
        assert document_id not in serialized
        assert str(source.resolve()) not in serialized

    with pytest.raises(
        MountedDocumentPlanningAuthorityError,
        match="current source bytes changed",
    ):
        freeze_mounted_document_planning_authority(session_id=session_id)


def test_every_call_checks_the_session_mount_again(mounted_runtime) -> None:
    source, runtime = mounted_runtime
    source.mounted = False

    outcome = _execute(
        runtime,
        "read_mounted_document_chunks",
        {
            "document_alias": DOCUMENT_ALIAS,
            "start_sequence": 0,
            "limit": 1,
        },
    )

    assert outcome.status is ExecutionStatus.FAILED
    assert outcome.error is not None
    assert outcome.error.code == "mounted_document_unavailable"


def test_chunk_count_is_part_of_the_frozen_generation_authority(
    mounted_runtime,
) -> None:
    source, runtime = mounted_runtime
    source.chunks = source.chunks[:-1]

    outcome = _execute(
        runtime,
        "inspect_mounted_document",
        {"document_alias": DOCUMENT_ALIAS},
    )

    assert outcome.status is ExecutionStatus.FAILED
    assert outcome.error is not None
    assert outcome.error.code == "mounted_document_authority_drift"


@pytest.mark.parametrize("private_name", ("path", "doc_id"))
def test_model_input_cannot_supply_paths_or_private_document_ids(
    mounted_runtime,
    private_name: str,
) -> None:
    _source, runtime = mounted_runtime

    outcome = _execute(
        runtime,
        "inspect_mounted_document",
        {
            "document_alias": DOCUMENT_ALIAS,
            private_name: PRIVATE_PATH if private_name == "path" else DOCUMENT_ID,
        },
    )

    assert outcome.status is ExecutionStatus.REJECTED
    assert outcome.error is not None
    assert outcome.error.code == "invalid_tool_input"


def test_handler_rejects_alias_outside_the_frozen_scope(
    mounted_runtime,
) -> None:
    _source, runtime = mounted_runtime

    outcome = _execute(
        runtime,
        "inspect_mounted_document",
        {"document_alias": "mounted_document_99"},
    )

    assert outcome.status is ExecutionStatus.FAILED
    assert outcome.error is not None
    assert outcome.error.code == "mounted_document_alias_unavailable"
