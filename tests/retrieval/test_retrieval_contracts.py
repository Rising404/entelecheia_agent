from __future__ import annotations

import pytest

from personagraph.retrieval.contracts import (
    ContextStatus,
    QueryProposal,
    RetrievalBudget,
    RetrievalRequest,
    RetrievedContext,
    RetrievalUnit,
    RetrievalStatus,
    SourceAvailability,
    SourceAccess,
    SourceDependency,
    SourceFilter,
    SourceIndexBinding,
    SourceIndexBindingSnapshot,
    SourceOutcome,
    SourceRetrievalStatus,
    SourceType,
    SourceUnitRef,
    TrustedRetrievalBoundary,
)
from personagraph.retrieval.query_guard import QueryGuard, QueryGuardError, QueryGuardErrorCode


def _ref(source_type: SourceType = SourceType.CURRENT_SESSION) -> SourceUnitRef:
    return SourceUnitRef(source_type, "unit-1", "r1", "hash-1")


def test_picture_is_a_first_class_retrieval_source_type():
    assert SourceType.PICTURE.value == "picture"


def test_source_unit_ref_is_the_minimal_pointer_and_freshness_contract():
    ref = _ref()
    assert ref.source_type is SourceType.CURRENT_SESSION
    assert ref.source_unit_id == "unit-1"
    assert ref.source_revision == "r1"
    assert ref.indexed_content_hash == "hash-1"
    assert not hasattr(ref, "content")


def test_source_index_binding_requires_the_exact_content_hash_and_rejects_conflicts():
    binding = SourceIndexBinding("unit-1", "r1", "hash-1")

    assert binding.indexed_content_hash == "hash-1"
    with pytest.raises(ValueError, match="indexed_content_hash"):
        SourceIndexBinding("unit-1", "r1", "")
    with pytest.raises(ValueError, match="duplicate source identities"):
        SourceIndexBindingSnapshot(
            True,
            (
                binding,
                SourceIndexBinding("unit-1", "r1", "different-hash"),
            ),
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"source_type": "current_session"},
        {"source_unit_id": 1},
        {"source_revision": None},
        {"indexed_content_hash": object()},
    ],
)
def test_source_unit_ref_rejects_untyped_identity_fields(kwargs: dict[str, object]):
    values: dict[str, object] = {
        "source_type": SourceType.CURRENT_SESSION,
        "source_unit_id": "unit-1",
        "source_revision": "r1",
        "indexed_content_hash": "hash-1",
    }
    values.update(kwargs)

    with pytest.raises(ValueError):
        SourceUnitRef(**values)  # type: ignore[arg-type]


def test_unit_content_binding_cannot_mix_source_types():
    with pytest.raises(ValueError, match="source_filter"):
        RetrievalUnit(
            ref=_ref(SourceType.CURRENT_SESSION),
            retrieval_data_version="v1",
            retrieval_status=RetrievalStatus.ACTIVE,
            source_filter=SourceFilter.from_mapping(SourceType.DOCUMENT, {"document_id": "d1"}),
        )


def test_trusted_boundary_rejects_a_filter_for_a_different_source_type():
    session_filter = SourceFilter.from_mapping(SourceType.CURRENT_SESSION, {"session_id": "s1"})
    with pytest.raises(ValueError, match="mapping key"):
        TrustedRetrievalBoundary({SourceType.DOCUMENT: session_filter})


def test_trusted_boundary_snapshots_verified_scope_against_later_mutation():
    session_filter = SourceFilter.from_mapping(SourceType.CURRENT_SESSION, {"session_id": "s1"})
    source_filters = {SourceType.CURRENT_SESSION: session_filter}
    source_dependencies = {SourceType.CURRENT_SESSION: SourceDependency.REQUIRED}
    boundary = TrustedRetrievalBoundary(source_filters, source_dependencies)

    source_filters[SourceType.DOCUMENT] = SourceFilter.from_mapping(
        SourceType.DOCUMENT,
        {"session_id": "s1", "doc_id": "untrusted"},
    )
    source_dependencies[SourceType.CURRENT_SESSION] = SourceDependency.OPTIONAL

    assert boundary.allowed_sources == (SourceType.CURRENT_SESSION,)
    assert boundary.dependency_for(SourceType.CURRENT_SESSION) is SourceDependency.REQUIRED
    with pytest.raises(TypeError):
        boundary.source_filters[SourceType.DOCUMENT] = source_filters[SourceType.DOCUMENT]  # type: ignore[index]
    with pytest.raises(TypeError):
        boundary.source_dependencies[SourceType.CURRENT_SESSION] = SourceDependency.OPTIONAL  # type: ignore[index]


def test_trusted_boundary_owns_a_snapshot_of_each_source_filter():
    session_filter = SourceFilter.from_mapping(SourceType.CURRENT_SESSION, {"session_id": "s1"})
    boundary = TrustedRetrievalBoundary({SourceType.CURRENT_SESSION: session_filter})

    # ``SourceFilter`` 对常规调用方不可变。这里有意使用底层逃生口，以证明边界
    # 复制了嵌套叶节点，而不只是复制外层映射。
    object.__setattr__(session_filter, "scope", (("session_id", "changed-after-build"),))

    assert boundary.source_filters[SourceType.CURRENT_SESSION] == SourceFilter.from_mapping(
        SourceType.CURRENT_SESSION,
        {"session_id": "s1"},
    )


@pytest.mark.parametrize(
    ("source_filters", "source_dependencies", "message"),
    [
        (
            {"current_session": SourceFilter.from_mapping(SourceType.CURRENT_SESSION, {"session_id": "s1"})},
            {},
            "filter keys",
        ),
        (
            {SourceType.CURRENT_SESSION: object()},
            {},
            "filter values",
        ),
        (
            {SourceType.CURRENT_SESSION: SourceFilter.from_mapping(SourceType.CURRENT_SESSION, {"session_id": "s1"})},
            {"current_session": SourceDependency.REQUIRED},
            "dependency keys",
        ),
        (
            {SourceType.CURRENT_SESSION: SourceFilter.from_mapping(SourceType.CURRENT_SESSION, {"session_id": "s1"})},
            {SourceType.CURRENT_SESSION: "required"},
            "dependency values",
        ),
    ],
)
def test_trusted_boundary_rejects_untyped_keys_or_values(
    source_filters: dict[object, SourceFilter],
    source_dependencies: dict[object, object],
    message: str,
):
    with pytest.raises(ValueError, match=message):
        TrustedRetrievalBoundary(source_filters, source_dependencies)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "scope",
    [
        (("session.id", "s1"),),
        (("session_id", "s1"), ("session_id", "s2")),
        (("session_id", "s1\nother"),),
    ],
)
def test_source_filter_rejects_noncanonical_or_unsafe_direct_scope(scope):
    with pytest.raises(ValueError):
        SourceFilter(SourceType.CURRENT_SESSION, scope)


def test_source_filter_direct_construction_canonicalizes_scope_order():
    source_filter = SourceFilter(
        SourceType.DOCUMENT,
        (("session_id", "s1"), ("doc_id", "d1")),
    )

    assert source_filter.scope == (("doc_id", "d1"), ("session_id", "s1"))


def test_source_access_freezes_observed_revision_facts_without_widening_scope():
    source_filter = SourceFilter.from_mapping(
        SourceType.DOCUMENT,
        {"doc_id": "doc-1", "session_id": "s1"},
    )
    revision_map = {"doc-1": "version-1"}

    access = SourceAccess(
        SourceType.DOCUMENT,
        source_filter,
        SourceAvailability.READY,
        source_snapshot_id="snapshot-1",
        source_revision_map=revision_map,
    )

    revision_map["doc-2"] = "version-2"
    assert access.source_revision_map == {"doc-1": "version-1"}
    with pytest.raises(TypeError):
        access.source_revision_map["doc-2"] = "version-2"  # type: ignore[index]


def test_source_access_rejects_mismatched_source_scope():
    session_filter = SourceFilter.from_mapping(SourceType.CURRENT_SESSION, {"session_id": "s1"})
    with pytest.raises(ValueError, match="match source_type"):
        SourceAccess(SourceType.DOCUMENT, session_filter, SourceAvailability.READY)


def test_source_access_rejects_an_untyped_revision_manifest():
    source_filter = SourceFilter.from_mapping(SourceType.DOCUMENT, {"doc_id": "doc-1", "session_id": "s1"})
    with pytest.raises(ValueError, match="source_revision_map"):
        SourceAccess(
            SourceType.DOCUMENT,
            source_filter,
            SourceAvailability.READY,
            source_revision_map=[("doc-1", "version-1")],  # type: ignore[arg-type]
        )


def test_source_access_freezes_only_bounded_public_coverage_facts():
    source_filter = SourceFilter.from_mapping(
        SourceType.DOCUMENT,
        {"doc_id": "doc-1", "session_id": "s1"},
    )
    facts = {
        "processing_status": "partial",
        "processing_diagnostic_codes": '["page_needs_vision"]',
        "needs_vision": "true",
        "coverage_gap": "true",
    }
    access = SourceAccess(
        SourceType.DOCUMENT,
        source_filter,
        SourceAvailability.READY,
        coverage_facts=facts,
    )

    facts["processing_status"] = "complete"
    assert dict(access.coverage_facts) == {
        "coverage_gap": "true",
        "needs_vision": "true",
        "processing_diagnostic_codes": '["page_needs_vision"]',
        "processing_status": "partial",
    }
    with pytest.raises(TypeError):
        access.coverage_facts["coverage_gap"] = "false"  # type: ignore[index]
    with pytest.raises(ValueError, match="unsupported source coverage fact"):
        SourceAccess(
            SourceType.DOCUMENT,
            source_filter,
            SourceAvailability.READY,
            coverage_facts={"diagnostic_detail": "must not reach prompt"},
        )


def test_source_access_accepts_parser_partial_as_a_bounded_coverage_fact():
    source_filter = SourceFilter.from_mapping(
        SourceType.DOCUMENT,
        {"doc_id": "doc-1", "session_id": "s1"},
    )

    access = SourceAccess(
        SourceType.DOCUMENT,
        source_filter,
        SourceAvailability.READY,
        coverage_facts={
            "processing_status": "partial",
            "processing_diagnostic_codes": '["parser_partial"]',
            "needs_vision": "false",
            "coverage_gap": "true",
        },
    )

    assert access.coverage_facts["processing_diagnostic_codes"] == (
        '["parser_partial"]'
    )


def test_budget_rejects_zero_limits():
    with pytest.raises(ValueError, match="greater than zero"):
        RetrievalBudget(candidate_limit_per_source=0, context_token_limit=1, max_items=1)


def test_retrieved_context_rejects_a_missing_source_outcome():
    with pytest.raises(ValueError, match="every SourceType"):
        RetrievedContext(
            status=ContextStatus.COMPLETE,
            items=(),
            source_outcomes={
                SourceType.CURRENT_SESSION: SourceOutcome(
                    source_type=SourceType.CURRENT_SESSION,
                    availability=SourceAvailability.READY,
                    retrieval=SourceRetrievalStatus.NO_MATCH,
                    dependency=SourceDependency.OPTIONAL,
                )
            },
            method_outcomes=(),
            configured_token_limit=100,
            packed_tokens=0,
        )


def test_retrieved_context_rejects_an_untyped_long_term_memory_write_guard():
    outcomes = {
        source_type: SourceOutcome(
            source_type=source_type,
            availability=SourceAvailability.DISABLED,
            retrieval=SourceRetrievalStatus.NOT_RUN,
            dependency=SourceDependency.OPTIONAL,
        )
        for source_type in SourceType
    }
    with pytest.raises(ValueError, match="long_term_memory_write_guard"):
        RetrievedContext(
            status=ContextStatus.COMPLETE,
            items=(),
            source_outcomes=outcomes,
            method_outcomes=(),
            configured_token_limit=100,
            packed_tokens=0,
            long_term_memory_write_guard=object(),  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "query",
    [
        "SELECT * FROM memories",
        "owner_id='other-user'",  # 查询不得夹带作用域过滤器。
        "session_id = 'other-session'",
        "doc_id = 'untrusted-document'",
        "persona_id = 'other-persona'",
        "path = '/outside/authorized/root'",
        "retrieval_data_version = 'staging'",
        "retrieval_method = 'bm25'",
    ],
)
def test_query_guard_rejects_low_level_retrieval_syntax(query: str):
    guard = QueryGuard()
    with pytest.raises(QueryGuardError):
        guard.validate(QueryProposal((query,)))


def test_query_guard_preserves_open_language_query_without_rewriting_it():
    guard = QueryGuard()
    query = "回忆上次关于北桥项目的可交付风险"
    assert guard.validate(QueryProposal((query,))) == (query,)


def test_query_guard_rejects_duplicates_after_whitespace_normalization():
    guard = QueryGuard()
    with pytest.raises(QueryGuardError, match="duplicate") as error:
        guard.validate(QueryProposal(("北桥项目风险", "  北桥项目风险  ")))
    assert error.value.code is QueryGuardErrorCode.DUPLICATE_QUERY


def test_query_guard_exposes_a_safe_code_for_low_level_selector_rejection():
    with pytest.raises(QueryGuardError) as error:
        QueryGuard().validate(QueryProposal(("document_id = 'untrusted'",)))
    assert error.value.code is QueryGuardErrorCode.FORBIDDEN_LOW_LEVEL_SYNTAX


def test_query_proposal_rejects_untyped_source_hints_before_policy_compilation():
    with pytest.raises(ValueError, match="source_hints"):
        QueryProposal(("北桥项目风险",), frozenset({"document"}))  # type: ignore[arg-type]


def test_retrieval_request_rejects_non_pointer_excluded_direct_references():
    boundary = TrustedRetrievalBoundary(
        {
            SourceType.CURRENT_SESSION: SourceFilter.from_mapping(
                SourceType.CURRENT_SESSION,
                {"session_id": "s1"},
            )
        }
    )
    with pytest.raises(ValueError, match="excluded_direct_refs"):
        RetrievalRequest(
            request_id="request-1",
            model_call_purpose="test",
            query_proposal=QueryProposal(("北桥项目",)),
            boundary=boundary,
            excluded_direct_refs=frozenset({"not-a-source-ref"}),  # type: ignore[arg-type]
        )


def test_retrieval_request_requires_excluded_direct_references_to_use_a_frozenset():
    boundary = TrustedRetrievalBoundary(
        {
            SourceType.CURRENT_SESSION: SourceFilter.from_mapping(
                SourceType.CURRENT_SESSION,
                {"session_id": "s1"},
            )
        }
    )
    with pytest.raises(ValueError, match="excluded_direct_refs"):
        RetrievalRequest(
            request_id="request-1",
            model_call_purpose="test",
            query_proposal=QueryProposal(("北桥项目",)),
            boundary=boundary,
            excluded_direct_refs={_ref()},  # type: ignore[arg-type]
        )
