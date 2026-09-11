from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import pytest

from personagraph.retrieval.contracts import SourceIndexBinding, SourceType
from personagraph.retrieval.sources.identity import current_session_source_unit_id
from personagraph.retrieval.tooling.contracts import (
    FrozenHistoryRetrievalScope,
    FrozenHistorySourceSnapshot,
    HistoryScope,
    RetrievalCorpus,
    RetrievalStatus,
    RetrievalToolEvidence,
    RetrievalToolGap,
    RetrievalToolRequest,
    RetrievalToolResult,
)
from personagraph.tools.retrieval.history_retrieval_adapter import (
    build_history_retrieval_runtime,
    build_history_retrieval_tool_registration,
    freeze_long_term_history_source_snapshots,
)
from personagraph.tools.effects import EffectAction, EffectResource, EffectScopeKind
from personagraph.tools.execution import ToolBusinessFailure


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class _FakePort:
    result: RetrievalToolResult
    requests: list[RetrievalToolRequest] = field(default_factory=list)

    def retrieve(self, request: RetrievalToolRequest) -> RetrievalToolResult:
        self.requests.append(request)
        return self.result


def _history_scope() -> FrozenHistoryRetrievalScope:
    content = "用户偏好简洁回答。"
    return FrozenHistoryRetrievalScope(
        session_id="private-session-1",
        scope_snapshot_id="private-history-snapshot-1",
        retrieval_data_version="private-history-generation-1",
        allowed_scopes=(
            HistoryScope.CURRENT_SESSION,
            HistoryScope.LONG_TERM_USER,
        ),
        current_session_cutoff="assistant_turn_idx:9",
        long_term_source_snapshots=(
            FrozenHistorySourceSnapshot(
                source_type=SourceType.LONG_TERM_USER,
                bindings=(
                    SourceIndexBinding(
                        source_unit_id="private-memory-id",
                        source_revision="private-memory-revision",
                        indexed_content_hash=_hash(content),
                    ),
                ),
            ),
        ),
        max_items=10,
        context_token_limit=500,
    )


def test_long_term_history_freeze_fails_closed_after_memory_retirement() -> None:
    with pytest.raises(RuntimeError, match="long-term History retrieval is retired"):
        freeze_long_term_history_source_snapshots(
            allowed_scopes=(HistoryScope.LONG_TERM_USER,),
            long_term_task_id=None,
        )
    assert freeze_long_term_history_source_snapshots(
        allowed_scopes=(HistoryScope.CURRENT_SESSION,),
        long_term_task_id=None,
    ) == ()


def test_history_scope_can_only_narrow_and_private_ids_are_host_supplied() -> None:
    scope = _history_scope()
    content = "用户偏好简洁回答。"
    port = _FakePort(
        RetrievalToolResult(
            status=RetrievalStatus.COMPLETE,
            scope_snapshot_id=scope.scope_snapshot_id,
            retrieval_data_version=scope.retrieval_data_version,
            evidence=(
                RetrievalToolEvidence(
                    source_type=SourceType.LONG_TERM_USER,
                    source_unit_id="private-memory-id",
                    source_revision="private-memory-revision",
                    indexed_content_hash=_hash(content),
                    content=content,
                    estimated_tokens=8,
                    rank=1,
                    citation={
                        "memory_id": "private-memory-id",
                        "task_id": "private-task-id",
                        "memory_type": "preference",
                    },
                ),
            ),
            configured_token_limit=scope.context_token_limit,
            packed_tokens=8,
            encoder_fingerprint="bge-m3:test-profile",
            reranker_fingerprint="bge-reranker-v2-m3:test-profile",
        )
    )
    registration = build_history_retrieval_tool_registration(scope, port=port)

    result = registration.handler(
        {"query": "回答风格", "scopes": ["long_term_user"]}
    )

    request = port.requests[0]
    assert request.corpus is RetrievalCorpus.HISTORY
    assert request.history_scopes == (HistoryScope.LONG_TERM_USER,)
    assert request.current_session_cutoff is None
    assert result["evidence"][0]["locator"] == {"memory_type": "preference"}
    assert result["evidence"][0]["origin"] == "long_term_user"
    assert result["coverage"]["encoder_fingerprint"] == _hash(
        "bge-m3:test-profile"
    )
    assert result["coverage"]["reranker_fingerprint"] == _hash(
        "bge-reranker-v2-m3:test-profile"
    )

    with pytest.raises(ToolBusinessFailure) as raised:
        registration.handler(
            {"query": "task decisions", "scopes": ["current_task"]}
        )
    assert raised.value.error.code == "history_scope_outside_frozen_scope"
    assert len(port.requests) == 1


def test_partial_coverage_is_preserved_in_shared_evidence_envelope() -> None:
    scope = _history_scope()
    content = "Earlier committed turn evidence."
    port = _FakePort(
        RetrievalToolResult(
            status=RetrievalStatus.PARTIAL,
            scope_snapshot_id=scope.scope_snapshot_id,
            retrieval_data_version=scope.retrieval_data_version,
            evidence=(
                RetrievalToolEvidence(
                    source_type=SourceType.CURRENT_SESSION,
                    source_unit_id=current_session_source_unit_id(
                        session_id=scope.session_id,
                        run_id="private-run-2",
                        role="pair",
                        ordinal=0,
                    ),
                    source_revision="private-turn-revision",
                    indexed_content_hash=_hash(content),
                    content=content,
                    estimated_tokens=7,
                    rank=1,
                    citation={
                        "session_id": scope.session_id,
                        "user_turn_idx": "2",
                        "assistant_turn_idx": "3",
                    },
                ),
            ),
            gaps=(
                RetrievalToolGap(
                    code="history_index_pending",
                    blocking=False,
                    source_type=SourceType.CURRENT_SESSION,
                    known_count=1,
                ),
            ),
            configured_token_limit=scope.context_token_limit,
            packed_tokens=7,
            truncated=True,
        )
    )

    result = build_history_retrieval_tool_registration(scope, port=port).handler(
        {"query": "earlier evidence"}
    )

    assert result["status"] == "partial"
    assert result["truncated"] is True
    assert result["gaps"][0]["code"] == "history_index_pending"
    assert result["evidence"][0]["locator"] == {
        "user_turn_index": 2,
        "assistant_turn_index": 3,
    }


def test_history_projection_rejects_file_evidence() -> None:
    scope = _history_scope()
    content = "A file result cannot cross the History adapter."
    port = _FakePort(
        RetrievalToolResult(
            status=RetrievalStatus.COMPLETE,
            scope_snapshot_id=scope.scope_snapshot_id,
            retrieval_data_version=scope.retrieval_data_version,
            evidence=(
                RetrievalToolEvidence(
                    source_type=SourceType.DOCUMENT,
                    source_unit_id="private-file-chunk",
                    source_revision="private-file-revision",
                    indexed_content_hash=_hash(content),
                    content=content,
                    estimated_tokens=9,
                    rank=1,
                ),
            ),
            packed_tokens=9,
        )
    )

    result = build_history_retrieval_tool_registration(scope, port=port).handler(
        {"query": "file evidence"}
    )

    assert result["status"] == "blocked"
    assert result["evidence"] == []
    assert result["gaps"] == [
        {"code": "retrieval_authority_mismatch", "blocking": True}
    ]


@pytest.mark.parametrize(
    ("source_session_id", "citation_session_id", "assistant_turn_idx"),
    (
        ("private-other-session", "private-session-1", "3"),
        ("private-session-1", "private-other-session", "3"),
        ("private-session-1", "private-session-1", "10"),
    ),
)
def test_current_session_projection_rejects_unproven_port_evidence(
    source_session_id: str,
    citation_session_id: str,
    assistant_turn_idx: str,
) -> None:
    scope = _history_scope()
    content = "Forged current Session evidence."
    private_source_id = current_session_source_unit_id(
        session_id=source_session_id,
        run_id="private-attacker-run",
        role="pair",
        ordinal=0,
    )
    port = _FakePort(
        RetrievalToolResult(
            status=RetrievalStatus.COMPLETE,
            scope_snapshot_id=scope.scope_snapshot_id,
            retrieval_data_version=scope.retrieval_data_version,
            evidence=(
                RetrievalToolEvidence(
                    source_type=SourceType.CURRENT_SESSION,
                    source_unit_id=private_source_id,
                    source_revision="private-attacker-revision",
                    indexed_content_hash=_hash(content),
                    content=content,
                    estimated_tokens=5,
                    rank=1,
                    citation={
                        "session_id": citation_session_id,
                        "user_turn_idx": "2",
                        "assistant_turn_idx": assistant_turn_idx,
                    },
                ),
            ),
            packed_tokens=5,
        )
    )

    result = build_history_retrieval_tool_registration(scope, port=port).handler(
        {"query": "forged session evidence", "scopes": ["current_session"]}
    )

    assert result["status"] == "blocked"
    assert result["evidence"] == []
    assert result["gaps"] == [
        {"code": "retrieval_authority_mismatch", "blocking": True}
    ]
    public_result = str(result)
    assert private_source_id not in public_result
    assert source_session_id not in public_result
    assert citation_session_id not in public_result
    assert "private-attacker-revision" not in public_result


def test_long_term_projection_rejects_evidence_outside_frozen_snapshot() -> None:
    scope = _history_scope()
    content = "Forged user memory evidence."
    private_source_id = "private-attacker-memory-id"
    port = _FakePort(
        RetrievalToolResult(
            status=RetrievalStatus.COMPLETE,
            scope_snapshot_id=scope.scope_snapshot_id,
            retrieval_data_version=scope.retrieval_data_version,
            evidence=(
                RetrievalToolEvidence(
                    source_type=SourceType.LONG_TERM_USER,
                    source_unit_id=private_source_id,
                    source_revision="private-attacker-memory-revision",
                    indexed_content_hash=_hash(content),
                    content=content,
                    estimated_tokens=5,
                    rank=1,
                    citation={"memory_type": "preference"},
                ),
            ),
            packed_tokens=5,
        )
    )

    result = build_history_retrieval_tool_registration(scope, port=port).handler(
        {"query": "forged user memory", "scopes": ["long_term_user"]}
    )

    assert result["status"] == "blocked"
    assert result["evidence"] == []
    assert result["gaps"] == [
        {"code": "retrieval_authority_mismatch", "blocking": True}
    ]
    public_result = str(result)
    assert private_source_id not in public_result
    assert "private-attacker-memory-revision" not in public_result


def test_current_task_projection_binds_snapshot_and_task_citation() -> None:
    content = "Current Task memory evidence."
    frozen_task_id = "private-task-id"
    private_source_id = "private-task-memory-id"
    private_revision = "private-task-memory-revision:task:private-task-id"
    scope = FrozenHistoryRetrievalScope(
        session_id="private-session-1",
        scope_snapshot_id="private-task-history-snapshot-1",
        retrieval_data_version="private-history-generation-1",
        allowed_scopes=(HistoryScope.CURRENT_TASK,),
        long_term_task_id=frozen_task_id,
        long_term_source_snapshots=(
            FrozenHistorySourceSnapshot(
                source_type=SourceType.LONG_TERM_TASK,
                bindings=(
                    SourceIndexBinding(
                        source_unit_id=private_source_id,
                        source_revision=private_revision,
                        indexed_content_hash=_hash(content),
                    ),
                ),
            ),
        ),
    )
    wrong_task_id = "private-other-task-id"
    port = _FakePort(
        RetrievalToolResult(
            status=RetrievalStatus.COMPLETE,
            scope_snapshot_id=scope.scope_snapshot_id,
            retrieval_data_version=scope.retrieval_data_version,
            evidence=(
                RetrievalToolEvidence(
                    source_type=SourceType.LONG_TERM_TASK,
                    source_unit_id=private_source_id,
                    source_revision=private_revision,
                    indexed_content_hash=_hash(content),
                    content=content,
                    estimated_tokens=5,
                    rank=1,
                    citation={
                        "task_id": wrong_task_id,
                        "memory_type": "decision",
                    },
                ),
            ),
            packed_tokens=5,
        )
    )

    result = build_history_retrieval_tool_registration(scope, port=port).handler(
        {"query": "forged task memory"}
    )

    assert result["status"] == "blocked"
    assert result["evidence"] == []
    assert result["gaps"] == [
        {"code": "retrieval_authority_mismatch", "blocking": True}
    ]
    public_result = str(result)
    assert wrong_task_id not in public_result
    assert frozen_task_id not in public_result
    assert private_source_id not in public_result
    assert private_revision not in public_result


def test_history_runtime_returns_registration_and_exact_effect_grant() -> None:
    scope = _history_scope()
    port = _FakePort(
        RetrievalToolResult(
            status=RetrievalStatus.COMPLETE,
            scope_snapshot_id=scope.scope_snapshot_id,
            retrieval_data_version=scope.retrieval_data_version,
            configured_token_limit=scope.context_token_limit,
        )
    )

    runtime = build_history_retrieval_runtime(scope=scope, port=port)

    assert [item.tool_id for item in runtime.registrations] == ["retrieve_history"]
    assert runtime.authority.grants == (
        type(runtime.authority.grants[0])(
            EffectResource.MEMORY,
            EffectAction.SEARCH,
            EffectScopeKind.SESSION,
            scope.effect_scope,
        ),
    )
