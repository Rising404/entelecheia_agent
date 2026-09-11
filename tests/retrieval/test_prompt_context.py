from __future__ import annotations

from personagraph.retrieval.contracts import (
    ContextStatus,
    LongTermMemoryWriteGuard,
    LongTermMemoryWriteGuardStatus,
    RetrievedContext,
    RetrievedItem,
    SourceAvailability,
    SourceDependency,
    SourceOutcome,
    SourceRetrievalStatus,
    SourceUnitRef,
    SourceType,
)
from personagraph.retrieval.prompt_context import RetrievedContextPromptSerializer


def _ref(unit_id: str) -> SourceUnitRef:
    return SourceUnitRef(SourceType.CURRENT_SESSION, unit_id, "r1", "h1")


def test_prompt_serializer_preserves_pack_order_without_retrieval_or_diagnostics():
    first, second = _ref("s1:run-1"), _ref("s1:run-2")
    context = RetrievedContext(
        status=ContextStatus.PARTIAL,
        items=(
            RetrievedItem(first, "第一条已验证正文", {"run_id": "run-1"}, 4, 2),
            RetrievedItem(second, "第二条已验证正文", {"run_id": "run-2"}, 5, 1),
        ),
        source_outcomes={
            **{
                source_type: SourceOutcome(
                    source_type=source_type,
                    availability=SourceAvailability.DISABLED,
                    retrieval=SourceRetrievalStatus.NOT_RUN,
                    dependency=SourceDependency.OPTIONAL,
                    reason_code="source_excluded_by_trusted_boundary",
                )
                for source_type in SourceType
            },
            SourceType.LONG_TERM_USER: SourceOutcome(
                source_type=SourceType.LONG_TERM_USER,
                availability=SourceAvailability.UNAVAILABLE,
                retrieval=SourceRetrievalStatus.NOT_RUN,
                dependency=SourceDependency.OPTIONAL,
                reason_code="source_availability_failed:TimeoutError",
            )
        },
        method_outcomes=(),
        configured_token_limit=128,
        packed_tokens=9,
        diagnostic_codes=("internal_sqlite_detail_must_not_reach_prompt",),
        long_term_memory_write_guard=LongTermMemoryWriteGuard(
            status=LongTermMemoryWriteGuardStatus.BLOCKED,
            evaluated_sources=(SourceType.LONG_TERM_USER, SourceType.LONG_TERM_TASK),
            blocking_sources=(SourceType.LONG_TERM_USER,),
            reason_codes=("long_term_user:availability_unavailable",),
        ),
    )

    serialized = RetrievedContextPromptSerializer().serialize(context)

    assert serialized.item_refs == (first, second)
    assert serialized.text.index("第一条已验证正文") < serialized.text.index("第二条已验证正文")
    assert 'position="1"' in serialized.text
    assert 'position="2"' in serialized.text
    assert "internal_sqlite_detail_must_not_reach_prompt" not in serialized.text
    assert 'source="long_term_user"' in serialized.text
    assert 'retrieval="not_run"' in serialized.text
    assert '<long_term_memory_write_guard status="blocked"' in serialized.text
    assert "long_term_user:availability_unavailable" in serialized.text


def test_prompt_serializer_exposes_bounded_document_coverage_facts_without_diagnostic_detail():
    document_ref = SourceUnitRef(SourceType.DOCUMENT, "s1:doc:chunk-1", "v1", "h1")
    citation = {
        "doc_id": "doc-1",
        "processing_status": "partial",
        "processing_diagnostic_codes": '["page_needs_vision"]',
        "needs_vision": "true",
        "coverage_gap": "true",
    }
    context = RetrievedContext(
        status=ContextStatus.COMPLETE,
        items=(RetrievedItem(document_ref, "已验证的文本页", citation, 5, 1),),
        source_outcomes={
            source_type: SourceOutcome(
                source_type=source_type,
                availability=(
                    SourceAvailability.READY
                    if source_type is SourceType.DOCUMENT
                    else SourceAvailability.DISABLED
                ),
                retrieval=(
                    SourceRetrievalStatus.MATCHED
                    if source_type is SourceType.DOCUMENT
                    else SourceRetrievalStatus.NOT_RUN
                ),
                dependency=SourceDependency.OPTIONAL,
            )
            for source_type in SourceType
        },
        method_outcomes=(),
        configured_token_limit=128,
        packed_tokens=5,
    )

    serialized = RetrievedContextPromptSerializer().serialize(context)

    assert "processing_status=partial" in serialized.text
    assert "processing_diagnostic_codes=[&quot;" in serialized.text
    assert "page_needs_vision" in serialized.text
    assert "needs_vision=true" in serialized.text
    assert "coverage_gap=true" in serialized.text
    assert "scanned page" not in serialized.text
