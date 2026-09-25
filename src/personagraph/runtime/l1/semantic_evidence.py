"""从当前执行的真实结果投影有界审查材料，不推断任务完成或模型理解。"""

from collections.abc import Mapping
from copy import deepcopy
import hashlib
import json

from ...output_protocol.l1 import L1ResultReference
from ...output_protocol.l1_persistence import normalize_l1_finding_arguments
from ...persistent_turn_content.evidence import l1_calls_by_ref, project_findings_arguments
from ...persistent_turn_content.tool_results import (
    ToolHistoryError,
    select_tool_result_chunk,
)
from ...persistent_turn_content.findings import EXECUTION_FINDINGS_TOOL_IDS
from ...tools.tool_history.definitions import TOOL_HISTORY_TOOL_IDS
from .identity import canonical_json


REVIEW_EVIDENCE_MAX_UTF8_BYTES = 1_000_000
REVIEW_RECENT_RESULT_LIMIT = 8
_NON_EVIDENCE_TOOL_IDS = frozenset(
    (*EXECUTION_FINDINGS_TOOL_IDS, *TOOL_HISTORY_TOOL_IDS)
)


class L1EvidenceProjectionError(ValueError):
    """来源不存在、损坏或所选片段不属于该结果；不能当作无证据继续。"""


def project_referenced_results(
    *,
    references: tuple[L1ResultReference, ...],
    execution: Mapping[str, object],
) -> list[dict[str, object]]:
    """验证原生结果身份及可选 chunk 归属，再投影真实正文。"""
    calls = _calls_by_call_id(execution)
    return [_reference_record(reference, calls) for reference in references]


def project_review_evidence(
    *,
    references: tuple[L1ResultReference, ...],
    execution: Mapping[str, object],
    max_utf8_bytes: int = REVIEW_EVIDENCE_MAX_UTF8_BYTES,
) -> dict[str, object]:
    """显式引用优先，随后补充最近成功结果；省略有显式标记，不冒充完整语料。"""
    calls = _calls_by_call_id(execution)
    records = [_reference_record(reference, calls) for reference in references]
    used = len(canonical_json(records).encode("utf-8"))
    if used > max_utf8_bytes:
        raise L1EvidenceProjectionError("referenced_evidence_exceeds_budget")
    selected_ids = {reference.tool_call_id for reference in references}
    recent = [
        (result_id, call)
        for result_id, call in calls.items()
        if call.get("status") == "succeeded"
        and call.get("tool_id") not in _NON_EVIDENCE_TOOL_IDS
        and result_id not in selected_ids
    ]
    added = 0
    for result_id, call in reversed(recent):
        if added >= REVIEW_RECENT_RESULT_LIMIT:
            break
        record = _result_record(result_id, call)
        size = len(canonical_json(record).encode("utf-8")) + 1
        if used + size > max_utf8_bytes:
            continue
        records.append(record)
        used += size
        added += 1
    return {
        "results": records,
        "selection": "explicit_references_then_recent_successes",
        "omitted_recent_result_count": len(recent) - added,
        "evidence_scope": "Only the supplied result bodies and selected chunks are available for review.",
    }


def _calls_by_call_id(
    execution: Mapping[str, object],
) -> dict[str, Mapping[str, object]]:
    try:
        raw_calls = l1_calls_by_ref(execution)
    except ValueError as exc:
        raise L1EvidenceProjectionError("tool_call_coordinates_invalid") from exc
    calls: dict[str, Mapping[str, object]] = {}
    for call_ref, call in raw_calls.items():
        call_id = call.get("tool_call_id")
        if call.get("status") != "succeeded":
            continue
        digest = call.get("outcome_hash")
        if not isinstance(digest, str) or len(digest) != 64:
            raise L1EvidenceProjectionError("tool_result_identity_invalid")
        calls[call_id] = {**call, "call_ref": call_ref}
    return calls


def _reference_record(
    reference: L1ResultReference,
    calls: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    call = calls.get(reference.tool_call_id)
    if call is None:
        raise L1EvidenceProjectionError("referenced_result_unavailable")
    if call.get("outcome_hash") != reference.result_sha256:
        raise L1EvidenceProjectionError("referenced_result_hash_mismatch")
    record = _result_record(reference.tool_call_id, call)
    if reference.chunk_id is not None:
        try:
            record["result"] = select_tool_result_chunk(
                record["result"], reference.chunk_id
            )
        except ToolHistoryError as exc:
            raise L1EvidenceProjectionError(str(exc)) from exc
        record["chunk_id"] = reference.chunk_id
        record["result_scope"] = "selected_chunk_only"
    return record


def _result_record(call_id: str, call: Mapping[str, object]) -> dict[str, object]:
    tool_id = call.get("tool_id")
    if not isinstance(tool_id, str) or not tool_id or tool_id in _NON_EVIDENCE_TOOL_IDS:
        raise L1EvidenceProjectionError("internal_receipt_is_not_task_evidence")
    raw = call.get("outcome_json")
    if isinstance(raw, str):
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        try:
            outcome = json.loads(raw)
        except ValueError as exc:
            raise L1EvidenceProjectionError("durable_result_invalid") from exc
    elif isinstance(raw, Mapping):
        outcome = raw
        digest = hashlib.sha256(canonical_json(raw).encode("utf-8")).hexdigest()
    else:
        raise L1EvidenceProjectionError("durable_result_missing")
    if not isinstance(outcome, Mapping) or digest != call.get("outcome_hash"):
        raise L1EvidenceProjectionError("durable_result_hash_mismatch")
    return {
        "tool_call_id": call_id,
        "result_sha256": call.get("outcome_hash"),
        "call_ref": call["call_ref"],
        "tool_id": tool_id,
        "status": "succeeded",
        "result": deepcopy(outcome.get("result")),
        "error": deepcopy(outcome.get("error")),
        "metadata": deepcopy(outcome.get("metadata")),
    }


def project_review_execution_history(
    execution: Mapping[str, object],
) -> list[dict[str, object]]:
    """执行经过只证明调用发生；失败也可解释限制，但不提供失败正文作为事实。"""
    from .tool_context.projection import project_tool_call_arguments

    try:
        calls = l1_calls_by_ref(execution)
    except ValueError as exc:
        raise L1EvidenceProjectionError("tool_call_coordinates_invalid") from exc
    call_refs = {str(call["tool_call_id"]): ref for ref, call in calls.items()}
    attempts = execution.get("attempts")
    ordinals = {
        item.get("attempt_id"): item.get("ordinal")
        for item in attempts or []
        if isinstance(item, Mapping)
    }
    history = []
    for call_ref, call in calls.items():
        raw = call.get("outcome_json")
        outcome = json.loads(raw) if isinstance(raw, str) else raw or {}
        if not isinstance(outcome, Mapping):
            raise L1EvidenceProjectionError("durable_result_invalid")
        error, result = outcome.get("error"), outcome.get("result")
        item = {
            "call_ref": call_ref,
            "tool_id": call.get("tool_id"),
            "attempt_ordinal": ordinals.get(call.get("attempt_id")),
            "call_ordinal": call.get("call_ordinal"),
            "status": call.get("status"),
            "error_code": error.get("code") if isinstance(error, Mapping) else None,
            "result_status": result.get("status")
            if isinstance(result, Mapping)
            else None,
        }
        if call.get("arguments_json") is not None or isinstance(
            call.get("arguments"), Mapping
        ):
            item.update(project_tool_call_arguments(call))
            failed_finding = (
                call.get("tool_id") in EXECUTION_FINDINGS_TOOL_IDS
                and call.get("status") != "succeeded"
            )
            if failed_finding:
                # Schema-rejected calls intentionally retain their original invalid
                # arguments. Verify the stored bytes above, but do not interpret
                # those diagnostic arguments as valid source-reference contracts.
                item.pop("arguments")
                item.pop("arguments_projection")
                item["arguments_unavailable"] = True
                item["arguments_unavailable_reason"] = "unsuccessful_findings_call"
            elif call.get("tool_id") in EXECUTION_FINDINGS_TOOL_IDS:
                item["arguments"] = project_findings_arguments(
                    normalize_l1_finding_arguments(
                        item["arguments"], tool_calls=list(calls.values()),
                    ),
                    call_refs=call_refs,
                )
            if not failed_finding:
                item["arguments_projection"] = {
                    key: value for key, value in item["arguments_projection"].items()
                    if key in {"complete", "redacted_value_count", "truncated_value_count"}
                }
        else:
            item["arguments_unavailable"] = True
        history.append(item)
    return history


__all__ = [
    "L1EvidenceProjectionError",
    "project_referenced_results",
    "project_review_evidence",
    "project_review_execution_history",
]
