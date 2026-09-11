"""审查仅使用真实原生引用和有界结果正文，不用逐项完成自评选证据。"""

from copy import deepcopy

import pytest

from personagraph.output_protocol.l1 import (
    L1PlanProposal,
    L1ResultReference,
    materialize_l1_plan,
)
from personagraph.persistent_turn_content.evidence import l1_tool_result_id
from personagraph.runtime.l1.identity import canonical_json, sha256_json
from personagraph.runtime.l1.semantic_evidence import (
    L1EvidenceProjectionError,
    project_referenced_results,
    project_review_evidence,
)
from personagraph.runtime.l1.semantic_verification import (
    L1SemanticVerificationInputError,
    _semantic_review_view,
)
from personagraph.tools.model_interface import project_tool_result


def _call(
    call_id="call-1",
    *,
    result=None,
    tool_id="read_file_chunks",
    status="succeeded",
    metadata=None,
):
    outcome = {
        "result": result
        or {
            "chunks": [
                {"chunk_id": "chunk-1", "content": "first fact"},
                {"chunk_id": "chunk-2", "content": "second fact"},
            ],
            "file_id": "file-1",
            "partial": True,
        },
        "error": None,
        "metadata": {"truncated": False} if metadata is None else metadata,
    }
    digest = sha256_json(outcome)
    call = {
        "tool_call_id": call_id,
        "tool_id": tool_id,
        "status": status,
        "outcome_hash": digest,
        "outcome_json": canonical_json(outcome),
    }
    return call, l1_tool_result_id(tool_call_id=call_id, result_sha256=digest)


def _view(*, references=(), calls=()):
    text = "根据材料回答问题"
    plan = materialize_l1_plan(
        L1PlanProposal(objective=text, acceptances=({"criterion": text},)),
        input_message_id="input-1",
        user_text=text,
    )
    return _semantic_review_view(
        plan=plan,
        reply="目前只能回答部分内容。",
        references=references,
        model_view={
            "current_user_text": text,
            "execution_limits": {
                "attempts_remaining": 1,
                "finalization_required": True,
            },
        },
        execution={"tool_calls": list(calls), "state": {"max_attempts": 24}},
        candidate_note="缺少后半段材料，说明当前限制。",
    )


def test_review_reconstructs_each_selected_result_once_and_keeps_real_ids():
    call, result_id = _call()
    view = _view(
        references=(L1ResultReference(tool_result_id=result_id),), calls=(call,)
    )
    evidence = view["durable_evidence"]
    assert len(evidence["results"]) == 1
    assert evidence["results"][0]["tool_result_id"] == result_id
    assert evidence["results"][0]["metadata"] == {"truncated": False}
    assert canonical_json(view).count("first fact") == 1
    assert not {"acceptance_progress", "candidate_binding"} & view.keys()
    assert "result_ref" not in canonical_json(view)


def test_reviewer_uses_the_same_tool_projection_without_changing_host_evidence():
    body = {
        "evidence": [
            {
                "file_id": "file-1",
                "file_version_id": "version-1",
                "document_id": "doc-1",
                "document_version_id": "doc-version-1",
                "evidence_type": "document_chunk",
                "chunk_id": "chunk-1",
                "content": "Business text mentions rank and content_sha256 verbatim.",
                "content_sha256": "private-integrity-hash",
                "rank": 1,
                "query_matches": [{"query_index": 0, "score": 0.9}],
            }
        ]
    }
    call, result_id = _call(result=body, tool_id="retrieve_files")
    references = (L1ResultReference(tool_result_id=result_id),)
    before = deepcopy(call)
    view = _view(references=references, calls=(call,))
    result = view["durable_evidence"]["results"][0]
    assert result["result"] == project_tool_result("retrieve_files", body)
    assert result["result"]["files"][0]["chunks"][0] == {
        "chunk_id": "chunk-1",
        "content": body["evidence"][0]["content"],
    }
    assert result["metadata"] == {"truncated": False}
    assert call == before
    assert (
        project_referenced_results(
            references=references, execution={"tool_calls": [call]}
        )[0]["result"]
        == body
    )


def test_reviewer_preserves_result_scope_metadata_without_execution_diagnostics():
    scope = {"truncated": True, "partial": True, "next_cursor": "cursor-2"}
    metadata = {**scope, "contract_version": "internal", "duration_ms": 400}
    call, _ = _call(metadata=metadata)
    result = _view(calls=(call,))["durable_evidence"]["results"][0]
    assert result["metadata"] == scope
    assert "duration_ms" not in canonical_json(result)
    assert (
        project_review_evidence(references=(), execution={"tool_calls": [call]})[
            "results"
        ][0]["metadata"]
        == metadata
    )


def test_selected_chunk_cannot_claim_sibling_body_and_preserves_parent_metadata():
    call, result_id = _call()
    before = deepcopy(call)
    result = project_referenced_results(
        references=(L1ResultReference(tool_result_id=result_id, chunk_id="chunk-2"),),
        execution={"tool_calls": [call]},
    )[0]
    assert result["result_scope"] == "selected_chunk_only"
    assert result["result"]["chunk"]["content"] == "second fact"
    assert result["result"]["source_context"] == [
        {"file_id": "file-1", "partial": True}
    ]
    assert "first fact" not in canonical_json(result)
    assert call == before
    review_result = _view(
        references=(L1ResultReference(tool_result_id=result_id, chunk_id="chunk-2"),),
        calls=(call,),
    )["durable_evidence"]["results"][0]
    assert review_result["result_scope"] == "selected_chunk_only"
    assert review_result["chunk_id"] == "chunk-2"
    assert review_result["result"] == project_tool_result(
        "read_file_chunks", result["result"]
    )
    assert "first fact" not in canonical_json(review_result)


@pytest.mark.parametrize(
    "fault", ["missing", "failed", "internal_receipt", "hash", "chunk"]
)
def test_invalid_or_non_evidence_sources_are_not_silently_omitted(fault):
    call, result_id = _call()
    reference = L1ResultReference(tool_result_id=result_id)
    calls = [call]
    if fault == "missing":
        calls = []
    elif fault == "failed":
        call["status"] = "failed"
    elif fault == "internal_receipt":
        call["tool_id"] = "read_tool_result"
    elif fault == "hash":
        call["outcome_json"] = canonical_json({"result": "tampered"})
    else:
        reference = L1ResultReference(
            tool_result_id=result_id, chunk_id="not-in-result"
        )
    with pytest.raises(L1EvidenceProjectionError):
        project_referenced_results(
            references=(reference,), execution={"tool_calls": calls}
        )
    with pytest.raises(L1SemanticVerificationInputError):
        _view(references=(reference,), calls=calls)


def test_no_references_still_exposes_recent_successful_results_not_failed_facts():
    good, result_id = _call()
    failed, _ = _call(
        "failed-1", result={"text": "must not become evidence"}, status="failed"
    )
    view = _view(calls=(good, failed))
    assert [item["tool_result_id"] for item in view["durable_evidence"]["results"]] == [
        result_id
    ]
    assert "must not become evidence" not in canonical_json(view)
    assert view["execution_context"]["tool_calls"][-1]["status"] == "failed"
    assert view["execution_context"]["stop"]["must_finalize"] is True
    assert (
        view["execution_context"]["candidate_note"] == "缺少后半段材料，说明当前限制。"
    )


def test_budget_preserves_explicit_reference_and_reports_unselected_recent_bodies():
    cited, cited_id = _call()
    recent, _ = _call("recent", result={"text": "large" * 1000})
    refs = (L1ResultReference(tool_result_id=cited_id),)
    explicit_size = len(
        canonical_json(
            project_referenced_results(
                references=refs,
                execution={"tool_calls": [cited]},
            )
        ).encode("utf-8")
    )
    evidence = project_review_evidence(
        references=refs,
        execution={"tool_calls": [cited, recent]},
        max_utf8_bytes=explicit_size,
    )
    assert [item["tool_result_id"] for item in evidence["results"]] == [cited_id]
    assert evidence["omitted_recent_result_count"] == 1
    with pytest.raises(L1EvidenceProjectionError, match="exceeds_budget"):
        project_review_evidence(
            references=refs,
            execution={"tool_calls": [cited]},
            max_utf8_bytes=explicit_size - 1,
        )


def test_duplicate_or_ambiguous_chunk_identity_is_rejected():
    call, result_id = _call(
        result={
            "chunks": [
                {"chunk_id": "same", "content": "one"},
                {"chunk_id": "same", "content": "two"},
            ]
        }
    )
    with pytest.raises(L1EvidenceProjectionError, match="ambiguous"):
        project_referenced_results(
            references=(L1ResultReference(tool_result_id=result_id, chunk_id="same"),),
            execution={"tool_calls": [call]},
        )
