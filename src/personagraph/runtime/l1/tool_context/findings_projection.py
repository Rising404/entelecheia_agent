"""核对持久每步笔记，投影原生记录身份与实际输入批次，不生成证据结论。"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from typing import Any

from pydantic import ValidationError

from ....output_protocol.l1 import L1AttemptDecisionProposal, L1_ATTEMPT_PROTOCOL_VERSION
from ....persistent_turn_content.findings import (
    derive_execution_finding_entry_id,
    derive_execution_findings_mutation_id,
    l1_execution_note_writer_id,
)


class FindingsProjectionError(ValueError):
    """持久台账无法安全投影为模型输入。"""


def project_execution_findings_for_model(
    projection: object, *, execution: Mapping[str, object],
) -> dict[str, Any]:
    raw = projection.model_dump(mode="json") if hasattr(projection, "model_dump") else projection
    if not isinstance(raw, Mapping):
        raise FindingsProjectionError("execution findings projection is not an object")
    entries = raw.get("active_entries")
    if not isinstance(entries, (list, tuple)):
        raise FindingsProjectionError("execution findings active entries have the wrong shape")
    ledger_id = _string(raw, "ledger_id")
    attempts = _attempt_context(execution)
    notes = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise FindingsProjectionError("execution finding entry is not an object")
        attempt_id = _string(entry, "writer_unit_id")
        attempt = attempts.get(attempt_id)
        if attempt is None:
            raise FindingsProjectionError("execution note has no source step")
        item = {
            "entry_id": _string(entry, "entry_id"),
            "summary": _string(entry, "claim"),
            "attempt_ordinal": attempt["ordinal"],
            "context_tool_result_ids": [],
        }
        if _string(entry, "writer_tool_call_id").startswith("l1notes:"):
            _require_fixed_note(entry, ledger_id=ledger_id, attempt=attempt)
            item["context_tool_result_ids"] = attempt["context_tool_result_ids"]
        else:
            # Explicit notes retain their own submitted references, not inferred support.
            item["source_refs"] = list(entry.get("source_refs", ()))
        notes.append(item)
    return {
        "ledger_id": ledger_id,
        "ledger_revision": _integer(raw, "ledger_revision"),
        "notes": notes,
        "active_entry_count": len(entries),
        "display_note_count": len(notes),
        "omitted_active_count": _integer(raw, "omitted_active_count"),
        "remaining_durable_revisions": _integer(raw, "remaining_durable_revisions"),
        "remaining_durable_utf8_bytes": _integer(raw, "remaining_durable_utf8_bytes"),
    }


def _attempt_context(execution: Mapping[str, object]) -> dict[str, dict]:
    attempts = execution.get("attempts")
    if not isinstance(attempts, list):
        raise FindingsProjectionError("persisted L1 Attempts have the wrong shape")
    result = {}
    for attempt in attempts:
        if not isinstance(attempt, Mapping):
            raise FindingsProjectionError("persisted L1 Attempt is not an object")
        attempt_id = _string(attempt, "attempt_id")
        if attempt_id in result:
            raise FindingsProjectionError("persisted L1 Attempt identity is duplicated")
        request = _verified_json(attempt, "request_json", "request_hash")
        if request.get("schema_version") != L1_ATTEMPT_PROTOCOL_VERSION:
            raise FindingsProjectionError("unsupported frozen L1 note protocol; start a new turn")
        proposal = None
        if attempt.get("decision_json") is not None:
            decision = _verified_json(attempt, "decision_json", "decision_hash")
            try:
                proposal = L1AttemptDecisionProposal.model_validate(decision)
            except ValidationError as exc:
                raise FindingsProjectionError("persisted L1 decision violates its contract") from exc
        prior = request.get("prior_tool_results", [])
        if not isinstance(prior, list) or any(not isinstance(item, Mapping) for item in prior):
            raise FindingsProjectionError("frozen input batch has the wrong shape")
        result[attempt_id] = {
            "attempt_id": attempt_id,
            "ordinal": _integer(attempt, "ordinal", minimum=1),
            "proposal": proposal,
            "context_tool_result_ids": list(dict.fromkeys(
                _string(item, "tool_result_id") for item in prior
            )),
        }
    return result


def _require_fixed_note(entry: Mapping, *, ledger_id: str, attempt: Mapping) -> None:
    proposal = attempt["proposal"]
    if not isinstance(proposal, L1AttemptDecisionProposal):
        raise FindingsProjectionError("fixed note has no frozen decision")
    writer = l1_execution_note_writer_id(attempt["attempt_id"])
    mutation_id = derive_execution_findings_mutation_id(writer_tool_call_id=writer)
    expected = {
        "entry_id": derive_execution_finding_entry_id(
            ledger_id=ledger_id, mutation_id=mutation_id, item_ordinal=1,
        ),
        "writer_tool_call_id": writer, "mutation_id": mutation_id,
        "kind": "decision", "claim": proposal.note,
    }
    if any(entry.get(key) != value for key, value in expected.items()) or (
        entry.get("source_refs") not in ([], ()) or entry.get("scope_keys") not in ([], ())
    ):
        raise FindingsProjectionError("fixed note disagrees with its frozen L1 decision")


def _verified_json(values: Mapping, json_key: str, hash_key: str) -> dict:
    raw, digest = values.get(json_key), values.get(hash_key)
    if not isinstance(raw, str) or hashlib.sha256(raw.encode()).hexdigest() != digest:
        raise FindingsProjectionError("persisted L1 record hash is invalid")
    try:
        result = json.loads(raw)
    except ValueError as exc:
        raise FindingsProjectionError("persisted L1 record is not JSON") from exc
    if not isinstance(result, dict):
        raise FindingsProjectionError("persisted L1 record is not an object")
    return result


def _string(values: Mapping, key: str) -> str:
    value = values.get(key)
    if not isinstance(value, str) or not value:
        raise FindingsProjectionError(f"execution findings {key} is invalid")
    return value


def _integer(values: Mapping, key: str, minimum: int = 0) -> int:
    value = values.get(key)
    if type(value) is not int or value < minimum:
        raise FindingsProjectionError(f"execution findings {key} is invalid")
    return value


__all__ = ["FindingsProjectionError", "project_execution_findings_for_model"]
