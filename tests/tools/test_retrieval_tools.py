from __future__ import annotations

import hashlib

import pytest

from personagraph.tools.effects import (
    DataEgress,
    EffectAction,
    EffectResource,
    EffectScopeKind,
)
from personagraph.tools.execution import ResolvedInvocation, ToolExecutor
from personagraph.tools.retrieval.retrieval_tools import (
    HistoryRetrievalScope,
    RETRIEVAL_EVIDENCE_CONTRACT_VERSION,
    RETRIEVE_HISTORY_CONTRACT_VERSION,
    RETRIEVE_HISTORY_TOOL_ID,
    RetrievalCorpus,
    RetrievalEvidenceCoverage,
    RetrievalEvidenceEnvelope,
    RetrievalEvidenceLocator,
    RetrievalEvidenceOutcome,
    RetrievalEvidenceStatus,
    build_retrieve_history_registration,
    opaque_fingerprint,
)


def _empty_result(corpus: RetrievalCorpus, query: str) -> dict[str, object]:
    return RetrievalEvidenceEnvelope(
        corpus=corpus,
        status=RetrievalEvidenceStatus.COMPLETE,
        outcome=RetrievalEvidenceOutcome.NO_MATCH,
        query=query,
        evidence=(),
        gaps=(),
        coverage=RetrievalEvidenceCoverage(
            scope_fingerprint=hashlib.sha256(b"scope").hexdigest(),
            retrieval_generation_fingerprint=hashlib.sha256(
                b"generation"
            ).hexdigest(),
            returned_items=0,
            configured_token_limit=100,
            packed_tokens=0,
        ),
    ).to_dict()


def test_history_tool_spec_exposes_only_narrowing_fields_and_shared_envelope() -> None:
    history = build_retrieve_history_registration(
        handler=lambda payload: _empty_result(
            RetrievalCorpus.HISTORY,
            payload["query"],
        ),
        effect_scope="history-corpus:scope",
    )

    assert (history.tool_id, history.contract_version) == (
        RETRIEVE_HISTORY_TOOL_ID,
        RETRIEVE_HISTORY_CONTRACT_VERSION,
    )
    assert set(history.spec.input_schema["properties"]) == {
        "query",
        "scopes",
        "limit",
    }
    assert history.spec.input_schema["properties"]["scopes"]["items"][
        "enum"
    ] == tuple(scope.value for scope in HistoryRetrievalScope)
    assert history.spec.output_schema["properties"]["corpus"] == {
        "type": "string",
        "const": RetrievalCorpus.HISTORY.value,
    }
    assert history.spec.output_schema["properties"]["contract_version"][
        "const"
    ] == RETRIEVAL_EVIDENCE_CONTRACT_VERSION
    schema_text = str(history.spec.input_schema).casefold()
    assert "session_id" not in schema_text
    assert "task_id" not in schema_text
    assert "path" not in schema_text
    assert "generation" not in schema_text


def test_history_tool_declares_session_memory_search_only() -> None:
    registration = build_retrieve_history_registration(
        handler=lambda payload: _empty_result(
            RetrievalCorpus.HISTORY,
            payload["query"],
        ),
        effect_scope="history-corpus:scope",
    )

    assert len(registration.effect_profile.effects) == 1
    effect = registration.effect_profile.effects[0]
    assert (
        effect.resource,
        effect.action,
        effect.scope_kind,
        effect.data_egress,
    ) == (
        EffectResource.MEMORY,
        EffectAction.SEARCH,
        EffectScopeKind.SESSION,
        DataEgress.CONTENT,
    )


def test_shared_envelope_validates_through_tool_executor() -> None:
    registration = build_retrieve_history_registration(
        handler=lambda payload: _empty_result(
            RetrievalCorpus.HISTORY,
            payload["query"],
        ),
        effect_scope="history-corpus:scope",
    )

    outcome = ToolExecutor().execute(
        ResolvedInvocation(registration, {"query": "What did we decide?"})
    )

    assert outcome.error is None
    assert outcome.result is not None
    assert outcome.result["status"] == "complete"
    assert outcome.result["outcome"] == "no_match"
    assert outcome.result["corpus"] == "history"


@pytest.mark.parametrize(
    "location",
    (
        "/Users/private/file.pdf",
        "../private/file.pdf",
        "~/private/file.pdf",
        "file:///private/file.pdf",
        r"C:\\private\\file.pdf",
    ),
)
def test_public_evidence_locator_rejects_private_paths(location: str) -> None:
    with pytest.raises(ValueError, match="filesystem path"):
        RetrievalEvidenceLocator(location=location)


def test_opaque_fingerprint_never_forwards_private_identity() -> None:
    private = "/Users/private/retrieval.sqlite:generation-7"
    fingerprint = opaque_fingerprint(private)

    assert fingerprint == hashlib.sha256(private.encode("utf-8")).hexdigest()
    assert private not in fingerprint
