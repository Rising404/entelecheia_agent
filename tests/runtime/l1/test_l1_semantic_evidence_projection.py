"""审查仅使用真实原生引用和有界结果正文，不用逐项完成自评选证据。"""

from copy import deepcopy

import pytest

from personagraph.output_protocol.l1 import (
    L1PlanProposal,
    L1ResultReference,
    materialize_l1_plan,
)
from personagraph.output_protocol.l1_persistence import legacy_l1_tool_result_id
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
    attempt_ordinal=1,
    call_ordinal=1,
):
    outcome = {
        "status": status,
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
        "attempt_id": f"attempt-{attempt_ordinal}",
        "call_ordinal": call_ordinal,
        "tool_id": tool_id,
        "status": status,
        "outcome_hash": digest,
        "outcome_json": canonical_json(outcome),
    }
    return call, L1ResultReference(tool_call_id=call_id, result_sha256=digest)


def _execution(calls):
    ordinals = sorted({int(call["attempt_id"].removeprefix("attempt-")) for call in calls})
    return {
        "tool_calls": list(calls),
        "attempts": [{"attempt_id": f"attempt-{i}", "ordinal": i} for i in ordinals],
        "state": {"max_attempts": 24},
    }


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
        execution=_execution(calls),
        candidate_note="缺少后半段材料，说明当前限制。",
    )


def test_review_reconstructs_selected_result_once_and_shows_stable_short_reference():
    call, result_id = _call()
    view = _view(
        references=(result_id,), calls=(call,)
    )
    evidence = view["durable_evidence"]
    assert len(evidence["results"]) == 1
    assert evidence["results"][0]["call_ref"] == "c1.1"
    assert "tool_call_id" not in evidence["results"][0]
    assert "result_sha256" not in evidence["results"][0]
    assert evidence["results"][0]["metadata"] == {"truncated": False}
    assert canonical_json(view).count("first fact") == 1
    assert not {"acceptance_progress", "candidate_binding"} & view.keys()
    assert "result_ref" not in canonical_json(view)
    assert "tool_result_id" not in canonical_json(view)
    assert result_id.result_sha256 not in canonical_json(view)


def test_reference_pins_original_result_even_if_body_and_ledger_digest_both_change():
    call, reference = _call()
    replacement = {"result": {"text": "replacement fact"}, "error": None}
    call["outcome_json"] = canonical_json(replacement)
    call["outcome_hash"] = sha256_json(replacement)
    with pytest.raises(L1EvidenceProjectionError, match="referenced_result_hash_mismatch"):
        project_referenced_results(references=(reference,), execution=_execution([call]))


def test_review_short_references_keep_persisted_coordinates_when_display_order_changes():
    older, reference = _call("older", attempt_ordinal=7, call_ordinal=3)
    newer, _ = _call("newer", attempt_ordinal=11, call_ordinal=2)
    view = _view(references=(reference,), calls=(newer, older))
    assert [item["call_ref"] for item in view["durable_evidence"]["results"]] == [
        "c7.3", "c11.2"
    ]
    assert [item["call_ref"] for item in view["execution_context"]["tool_calls"]] == [
        "c11.2", "c7.3"
    ]
    assert "tool_call_id" not in canonical_json(view)
    assert "result_sha256" not in canonical_json(view)


@pytest.mark.parametrize("legacy", [False, True])
def test_reviewer_findings_arguments_use_short_refs_without_mutating_durable_arguments(legacy):
    source, reference = _call("l1tool_" + "a" * 64, attempt_ordinal=3)
    finding, _ = _call("finding-call", attempt_ordinal=4, tool_id="record_execution_findings")
    arguments = {
        "items": [{
            "claim": "First fact was observed.",
            "kind": "fact",
            "source_refs": [{
                "tool_result_id": source["tool_call_id"],
                "result_sha256": source["outcome_hash"],
                "chunk_id": "chunk-1",
            }],
        }],
    }
    if legacy:
        arguments["items"][0]["source_refs"] = [{
            "tool_result_id": legacy_l1_tool_result_id(
                tool_call_id=source["tool_call_id"], result_sha256=source["outcome_hash"],
            ),
            "chunk_id": "chunk-1",
        }]
    finding.update(arguments_json=canonical_json(arguments), arguments_hash=sha256_json(arguments))
    before = deepcopy(finding)
    view = _view(references=(reference,), calls=(source, finding))
    projected = view["execution_context"]["tool_calls"][1]
    assert projected["arguments"]["items"][0]["source_refs"] == [{
        "call_ref": "c3.1", "chunk_id": "chunk-1",
    }]
    assert source["tool_call_id"] not in canonical_json(view)
    assert source["outcome_hash"] not in canonical_json(view)
    assert projected["arguments_projection"] == {
        "complete": True, "redacted_value_count": 0, "truncated_value_count": 0,
    }
    assert "arguments_sha256" not in canonical_json(view)
    assert finding == before


@pytest.mark.parametrize("arguments", [
    {"items": "bad"},
    {"items": [{"source_refs": [{"tool_result_id": "l1result_" + "f" * 64}]}]},
])
def test_failed_findings_keep_failure_visible_without_decoding_invalid_arguments(arguments):
    finding, _ = _call("finding-rejected", tool_id="record_execution_findings", status="rejected")
    outcome = {"status": "rejected", "result": None, "error": {"code": "invalid_tool_input"}}
    finding.update(
        arguments_json=canonical_json(arguments), arguments_hash=sha256_json(arguments),
        outcome_json=canonical_json(outcome), outcome_hash=sha256_json(outcome),
    )
    before = deepcopy(finding)
    view = _view(calls=(finding,))
    item = view["execution_context"]["tool_calls"][0]
    assert item["status"] == "rejected"
    assert item["error_code"] == "invalid_tool_input"
    assert item["arguments_unavailable"] is True
    assert item["arguments_unavailable_reason"] == "unsuccessful_findings_call"
    assert "arguments" not in item and "arguments_projection" not in item
    assert view["durable_evidence"]["results"] == []
    assert finding == before
    finding["arguments_json"] = canonical_json({"items": "tampered"})
    with pytest.raises(L1SemanticVerificationInputError):
        _view(calls=(finding,))


def test_failed_ordinary_tool_keeps_verified_business_arguments():
    call, _ = _call("read-failed", tool_id="read_text", status="failed")
    arguments = {"path": "missing.txt"}
    call.update(arguments_json=canonical_json(arguments), arguments_hash=sha256_json(arguments))
    item = _view(calls=(call,))["execution_context"]["tool_calls"][0]
    assert item["arguments"] == arguments
    assert "arguments_unavailable" not in item


@pytest.mark.parametrize("fault", ["missing_attempt", "missing_call_ordinal", "duplicate_coordinate"])
def test_invalid_persisted_coordinates_are_not_replaced_with_display_indices(fault):
    call, reference = _call()
    execution = _execution([call])
    if fault == "missing_attempt":
        execution["attempts"] = []
    elif fault == "missing_call_ordinal":
        del call["call_ordinal"]
    else:
        execution["tool_calls"].append({**call, "tool_call_id": "different-call"})
    with pytest.raises(L1EvidenceProjectionError, match="coordinates_invalid"):
        project_referenced_results(references=(reference,), execution=execution)


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
    references = (result_id,)
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
            references=references, execution=_execution([call])
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
        project_review_evidence(references=(), execution=_execution([call]))[
            "results"
        ][0]["metadata"]
        == metadata
    )


def test_selected_chunk_cannot_claim_sibling_body_and_preserves_parent_metadata():
    call, result_id = _call()
    before = deepcopy(call)
    result = project_referenced_results(
        references=(result_id.model_copy(update={"chunk_id": "chunk-2"}),),
        execution=_execution([call]),
    )[0]
    assert result["result_scope"] == "selected_chunk_only"
    assert result["result"]["chunk"]["content"] == "second fact"
    assert result["result"]["source_context"] == [
        {"file_id": "file-1", "partial": True}
    ]
    assert "first fact" not in canonical_json(result)
    assert call == before
    review_result = _view(
        references=(result_id.model_copy(update={"chunk_id": "chunk-2"}),),
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
    reference = result_id
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
        reference = result_id.model_copy(update={"chunk_id": "not-in-result"})
    with pytest.raises(L1EvidenceProjectionError):
        project_referenced_results(
            references=(reference,), execution=_execution(calls)
        )
    with pytest.raises(L1SemanticVerificationInputError):
        _view(references=(reference,), calls=calls)


def test_no_references_still_exposes_recent_successful_results_not_failed_facts():
    good, result_id = _call()
    failed, _ = _call(
        "failed-1", result={"text": "must not become evidence"}, status="failed", call_ordinal=2
    )
    view = _view(calls=(good, failed))
    assert [item["call_ref"] for item in view["durable_evidence"]["results"]] == [
        "c1.1"
    ]
    assert "must not become evidence" not in canonical_json(view)
    assert view["execution_context"]["tool_calls"][-1]["status"] == "failed"
    assert view["execution_context"]["stop"]["must_finalize"] is True
    assert (
        view["execution_context"]["candidate_note"] == "缺少后半段材料，说明当前限制。"
    )


def test_budget_preserves_explicit_reference_and_reports_unselected_recent_bodies():
    cited, cited_id = _call()
    recent, _ = _call("recent", result={"text": "large" * 1000}, call_ordinal=2)
    refs = (cited_id,)
    explicit_size = len(
        canonical_json(
            project_referenced_results(
                references=refs,
                execution=_execution([cited]),
            )
        ).encode("utf-8")
    )
    evidence = project_review_evidence(
        references=refs,
        execution=_execution([cited, recent]),
        max_utf8_bytes=explicit_size,
    )
    assert [item["tool_call_id"] for item in evidence["results"]] == [cited_id.tool_call_id]
    assert evidence["omitted_recent_result_count"] == 1
    with pytest.raises(L1EvidenceProjectionError, match="exceeds_budget"):
        project_review_evidence(
            references=refs,
            execution=_execution([cited]),
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
            references=(result_id.model_copy(update={"chunk_id": "same"}),),
            execution=_execution([call]),
        )
