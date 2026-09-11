from __future__ import annotations

import hashlib

import pytest

from personagraph.retrieval.contracts import SourceIndexBinding, SourceType
from personagraph.retrieval.tooling.contracts import (
    FileRetrievalOrigin,
    FileRetrievalScope,
    FrozenFileRetrievalBinding,
    FrozenFileVersionBinding,
    FrozenHistorySourceSnapshot,
    HistoryScope,
    RetrievalCorpus,
    RetrievalStatus,
    RetrievalToolEvidence,
    RetrievalToolRequest,
    RetrievalToolResult,
    format_current_session_cutoff,
    parse_current_session_cutoff,
)


def _file_binding() -> FrozenFileRetrievalBinding:
    return FrozenFileRetrievalBinding(
        source_id="file_01",
        authority_id="private-file-authority",
        origin=FileRetrievalOrigin.WORKSPACE,
    )


def _history_snapshot(
    source_type: SourceType,
    *,
    source_unit_id: str = "private-memory-id",
) -> FrozenHistorySourceSnapshot:
    return FrozenHistorySourceSnapshot(
        source_type=source_type,
        bindings=(
            SourceIndexBinding(
                source_unit_id=source_unit_id,
                source_revision="private-memory-revision",
                indexed_content_hash="a" * 64,
            ),
        ),
    )


def test_file_request_uses_file_source_identity_without_history_authority() -> None:
    request = RetrievalToolRequest(
        request_id="request-file-1",
        corpus=RetrievalCorpus.FILES,
        query="target paragraph",
        limit=4,
        context_token_limit=500,
        session_id="private-session",
        scope_snapshot_id="private-file-scope",
        retrieval_data_version="private-file-generation",
        file_scope=FileRetrievalScope.SELECTED_FILES,
        file_bindings=(_file_binding(),),
    )

    assert request.file_bindings[0].source_id == "file_01"
    with pytest.raises(ValueError, match="history authority"):
        RetrievalToolRequest(
            request_id="request-file-mixed",
            corpus=RetrievalCorpus.FILES,
            query="target paragraph",
            limit=4,
            context_token_limit=500,
            session_id="private-session",
            scope_snapshot_id="private-file-scope",
            retrieval_data_version="private-file-generation",
            file_scope=FileRetrievalScope.SELECTED_FILES,
            file_bindings=(_file_binding(),),
            history_scopes=(HistoryScope.CURRENT_SESSION,),
        )


def test_file_request_without_bindings_selects_the_session_document_corpus() -> None:
    request = RetrievalToolRequest(
        request_id="request-session-documents",
        corpus=RetrievalCorpus.FILES,
        query="target paragraph",
        limit=4,
        context_token_limit=500,
        session_id="private-session",
        scope_snapshot_id="private-file-scope",
        retrieval_data_version="private-file-generation",
        file_scope=FileRetrievalScope.SESSION_CORPUS,
    )

    assert request.file_scope is FileRetrievalScope.SESSION_CORPUS
    assert request.file_bindings == ()


def test_file_budget_expands_without_changing_history_item_limit() -> None:
    file_request = RetrievalToolRequest(
        request_id="request-file-expanded-budget",
        corpus=RetrievalCorpus.FILES,
        query="target paragraph",
        limit=96,
        context_token_limit=20_000,
        session_id="private-session",
        scope_snapshot_id="private-file-scope",
        retrieval_data_version="private-file-generation",
        file_scope=FileRetrievalScope.SESSION_CORPUS,
    )

    assert file_request.limit == 96
    assert file_request.context_token_limit == 20_000
    with pytest.raises(ValueError, match="max_items must be from 1 to 20"):
        RetrievalToolRequest(
            request_id="request-history-unchanged-budget",
            corpus=RetrievalCorpus.HISTORY,
            query="earlier decision",
            limit=21,
            context_token_limit=20_001,
            session_id="private-session",
            scope_snapshot_id="private-history-scope",
            retrieval_data_version="private-history-generation",
            history_scopes=(HistoryScope.CURRENT_SESSION,),
            current_session_cutoff="assistant_turn_idx:4",
        )


def test_session_file_inventory_is_exact_private_file_versions() -> None:
    binding = FrozenFileVersionBinding(
        project_id="private-project",
        file_id="private-file",
        file_version_id="private-file-version",
    )
    request = RetrievalToolRequest(
        request_id="request-session-files",
        corpus=RetrievalCorpus.FILES,
        query="target paragraph",
        limit=4,
        context_token_limit=500,
        session_id="private-session",
        scope_snapshot_id="private-file-scope",
        retrieval_data_version="private-file-generation",
        file_scope=FileRetrievalScope.SESSION_CORPUS,
        session_file_bindings=(binding,),
    )

    assert request.session_file_bindings == (binding,)
    with pytest.raises(ValueError, match="must be supplied together"):
        FrozenFileRetrievalBinding(
            source_id="file_01",
            authority_id="private-file-authority",
            origin=FileRetrievalOrigin.USER_UPLOAD,
            file_id="private-file",
        )


def test_file_request_requires_an_explicit_scope() -> None:
    with pytest.raises(ValueError, match="explicit File retrieval scope"):
        RetrievalToolRequest(
            request_id="request-file-missing-scope",
            corpus=RetrievalCorpus.FILES,
            query="target paragraph",
            limit=4,
            context_token_limit=500,
            session_id="private-session",
            scope_snapshot_id="private-file-scope",
            retrieval_data_version="private-file-generation",
        )


@pytest.mark.parametrize(
    ("file_scope", "bindings", "message"),
    (
        (
            FileRetrievalScope.SESSION_CORPUS,
            (_file_binding(),),
            "cannot carry file bindings",
        ),
        (
            FileRetrievalScope.SELECTED_FILES,
            (),
            "requires file bindings",
        ),
    ),
)
def test_file_request_scope_and_bindings_must_agree(
    file_scope: FileRetrievalScope,
    bindings: tuple[FrozenFileRetrievalBinding, ...],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        RetrievalToolRequest(
            request_id="request-file-invalid-scope-bindings",
            corpus=RetrievalCorpus.FILES,
            query="target paragraph",
            limit=4,
            context_token_limit=500,
            session_id="private-session",
            scope_snapshot_id="private-file-scope",
            retrieval_data_version="private-file-generation",
            file_scope=file_scope,
            file_bindings=bindings,
        )


@pytest.mark.parametrize(
    "bindings",
    (
        (
            _file_binding(),
            FrozenFileRetrievalBinding(
                source_id="file_01",
                authority_id="private-file-authority-2",
                origin=FileRetrievalOrigin.WORKSPACE,
            ),
        ),
        (
            _file_binding(),
            FrozenFileRetrievalBinding(
                source_id="file_02",
                authority_id="private-file-authority",
                origin=FileRetrievalOrigin.WORKSPACE,
            ),
        ),
    ),
)
def test_file_request_requires_unique_source_and_authority_bindings(
    bindings: tuple[FrozenFileRetrievalBinding, ...],
) -> None:
    with pytest.raises(ValueError, match="unique"):
        RetrievalToolRequest(
            request_id="request-file-duplicate",
            corpus=RetrievalCorpus.FILES,
            query="target paragraph",
            limit=4,
            context_token_limit=500,
            session_id="private-session",
            scope_snapshot_id="private-file-scope",
            retrieval_data_version="private-file-generation",
            file_scope=FileRetrievalScope.SELECTED_FILES,
            file_bindings=bindings,
        )


def test_history_request_cannot_carry_file_authority() -> None:
    with pytest.raises(ValueError, match="file authority"):
        RetrievalToolRequest(
            request_id="request-history-mixed",
            corpus=RetrievalCorpus.HISTORY,
            query="earlier decision",
            limit=4,
            context_token_limit=500,
            session_id="private-session",
            scope_snapshot_id="private-history-scope",
            retrieval_data_version="private-history-generation",
            file_bindings=(_file_binding(),),
            history_scopes=(HistoryScope.CURRENT_SESSION,),
            current_session_cutoff="assistant_turn_idx:4",
        )

    with pytest.raises(ValueError, match="File retrieval scope"):
        RetrievalToolRequest(
            request_id="request-history-file-scope",
            corpus=RetrievalCorpus.HISTORY,
            query="earlier decision",
            limit=4,
            context_token_limit=500,
            session_id="private-session",
            scope_snapshot_id="private-history-scope",
            retrieval_data_version="private-history-generation",
            file_scope=FileRetrievalScope.SESSION_CORPUS,
            history_scopes=(HistoryScope.CURRENT_SESSION,),
            current_session_cutoff="assistant_turn_idx:4",
        )


def test_history_request_reuses_scope_pairing_and_canonical_order() -> None:
    snapshot = _history_snapshot(SourceType.LONG_TERM_USER)
    request = RetrievalToolRequest(
        request_id="request-history-canonical",
        corpus=RetrievalCorpus.HISTORY,
        query="earlier decision",
        limit=4,
        context_token_limit=500,
        session_id="private-session",
        scope_snapshot_id="private-history-scope",
        retrieval_data_version="private-history-generation",
        history_scopes=(
            HistoryScope.LONG_TERM_USER,
            HistoryScope.CURRENT_SESSION,
        ),
        current_session_cutoff="assistant_turn_idx:4",
        history_source_snapshots=(snapshot,),
    )

    assert request.history_scopes == (
        HistoryScope.CURRENT_SESSION,
        HistoryScope.LONG_TERM_USER,
    )
    with pytest.raises(ValueError, match="unique"):
        RetrievalToolRequest(
            request_id="request-history-duplicate-scope",
            corpus=RetrievalCorpus.HISTORY,
            query="earlier decision",
            limit=4,
            context_token_limit=500,
            session_id="private-session",
            scope_snapshot_id="private-history-scope",
            retrieval_data_version="private-history-generation",
            history_scopes=(
                HistoryScope.CURRENT_SESSION,
                HistoryScope.CURRENT_SESSION,
            ),
            current_session_cutoff="assistant_turn_idx:4",
        )


def test_history_request_requires_paired_scope_authority() -> None:
    with pytest.raises(ValueError, match="committed-turn cutoff"):
        RetrievalToolRequest(
            request_id="request-history-missing-cutoff",
            corpus=RetrievalCorpus.HISTORY,
            query="earlier decision",
            limit=4,
            context_token_limit=500,
            session_id="private-session",
            scope_snapshot_id="private-history-scope",
            retrieval_data_version="private-history-generation",
            history_scopes=(HistoryScope.CURRENT_SESSION,),
        )
    with pytest.raises(ValueError, match="current_session scope"):
        RetrievalToolRequest(
            request_id="request-history-orphan-cutoff",
            corpus=RetrievalCorpus.HISTORY,
            query="earlier decision",
            limit=4,
            context_token_limit=500,
            session_id="private-session",
            scope_snapshot_id="private-history-scope",
            retrieval_data_version="private-history-generation",
            history_scopes=(HistoryScope.LONG_TERM_USER,),
            current_session_cutoff="assistant_turn_idx:4",
            history_source_snapshots=(
                _history_snapshot(SourceType.LONG_TERM_USER),
            ),
        )
    with pytest.raises(ValueError, match="trusted Task ID"):
        RetrievalToolRequest(
            request_id="request-history-missing-task",
            corpus=RetrievalCorpus.HISTORY,
            query="earlier decision",
            limit=4,
            context_token_limit=500,
            session_id="private-session",
            scope_snapshot_id="private-history-scope",
            retrieval_data_version="private-history-generation",
            history_scopes=(HistoryScope.CURRENT_TASK,),
            history_source_snapshots=(
                _history_snapshot(SourceType.LONG_TERM_TASK),
            ),
        )


def test_history_request_requires_unique_exact_snapshot_coverage() -> None:
    snapshot = _history_snapshot(SourceType.LONG_TERM_USER)
    with pytest.raises(ValueError, match="unique per Source"):
        RetrievalToolRequest(
            request_id="request-history-duplicate-snapshot",
            corpus=RetrievalCorpus.HISTORY,
            query="earlier decision",
            limit=4,
            context_token_limit=500,
            session_id="private-session",
            scope_snapshot_id="private-history-scope",
            retrieval_data_version="private-history-generation",
            history_scopes=(HistoryScope.LONG_TERM_USER,),
            history_source_snapshots=(snapshot, snapshot),
        )
    with pytest.raises(ValueError, match="every selected long-term Source"):
        RetrievalToolRequest(
            request_id="request-history-missing-snapshot",
            corpus=RetrievalCorpus.HISTORY,
            query="earlier decision",
            limit=4,
            context_token_limit=500,
            session_id="private-session",
            scope_snapshot_id="private-history-scope",
            retrieval_data_version="private-history-generation",
            history_scopes=(HistoryScope.LONG_TERM_USER,),
        )


def test_verified_result_requires_content_hash_and_internal_status() -> None:
    content = "verified evidence"
    evidence = RetrievalToolEvidence(
        source_type=SourceType.DOCUMENT,
        source_unit_id="private-unit",
        source_revision="private-revision",
        indexed_content_hash=hashlib.sha256(content.encode("utf-8")).hexdigest(),
        content=content,
        estimated_tokens=3,
        rank=1,
    )

    result = RetrievalToolResult(
        status=RetrievalStatus.COMPLETE,
        scope_snapshot_id="private-scope",
        retrieval_data_version="private-generation",
        evidence=(evidence,),
        packed_tokens=3,
    )

    assert result.status is RetrievalStatus.COMPLETE
    with pytest.raises(ValueError, match="authoritative content"):
        RetrievalToolEvidence(
            source_type=SourceType.DOCUMENT,
            source_unit_id="private-unit",
            source_revision="private-revision",
            indexed_content_hash="0" * 64,
            content=content,
            estimated_tokens=3,
            rank=1,
        )


def test_current_session_cutoff_round_trips_and_rejects_untrusted_shapes() -> None:
    cutoff = format_current_session_cutoff(12)

    assert cutoff == "assistant_turn_idx:12"
    assert parse_current_session_cutoff(cutoff) == 12
    with pytest.raises(ValueError, match="assistant_turn_idx"):
        parse_current_session_cutoff("turn:12")
