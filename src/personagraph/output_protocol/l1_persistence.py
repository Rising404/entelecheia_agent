"""Decode immutable L1 decisions at the persistence boundary.

Committed v5 rows retain their original JSON and hashes. Their legacy derived
result IDs are resolved to the one current call/digest contract when read. This
does not authorize resending an unfinished v5 model request under a new prompt.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json

from pydantic import BaseModel, ConfigDict, Field

from .l1 import L1AttemptDecision, L1ResultReference, L1_ATTEMPT_PROTOCOL_VERSION
from ..persistent_turn_content.json_values import freeze_json


LEGACY_L1_ATTEMPT_PROTOCOL_VERSION = "l1-attempt-model-view-v5"
SUPPORTED_L1_DECISION_PROTOCOLS = frozenset(
    {LEGACY_L1_ATTEMPT_PROTOCOL_VERSION, L1_ATTEMPT_PROTOCOL_VERSION}
)


class L1StoredResultIntegrityError(ValueError):
    """A matching persisted call has corrupt outcome bytes or status."""


class _LegacyResultReference(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    tool_result_id: str = Field(pattern=r"^l1result_[0-9a-f]{64}$")
    chunk_id: str | None = Field(default=None, min_length=1, max_length=200)


class _LegacyDecision(L1AttemptDecision):
    references: tuple[_LegacyResultReference, ...] = Field(default=(), max_length=24)


def legacy_l1_tool_result_id(*, tool_call_id: str, result_sha256: str) -> str:
    """Exact retired wire algorithm, used only to decode existing records."""
    if (
        not isinstance(tool_call_id, str)
        or not tool_call_id
        or not isinstance(result_sha256, str)
        or len(result_sha256) != 64
        or any(character not in "0123456789abcdef" for character in result_sha256)
    ):
        raise ValueError("legacy L1 result identity requires a call ID and sha256")
    payload = json.dumps(
        {"tool_call_id": tool_call_id, "result_sha256": result_sha256},
        ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"),
    )
    return "l1result_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def decode_l1_decision(
    raw: object,
    *,
    request_schema_version: str,
    tool_calls: Sequence[Mapping[str, object]],
) -> L1AttemptDecision:
    """Normalize a supported stored decision and check every pinned result."""
    value = json.loads(raw) if isinstance(raw, str) else raw
    calls = _index_calls(tool_calls)
    if request_schema_version == L1_ATTEMPT_PROTOCOL_VERSION:
        decision = L1AttemptDecision.model_validate(value)
    elif request_schema_version == LEGACY_L1_ATTEMPT_PROTOCOL_VERSION:
        legacy = _LegacyDecision.model_validate(value)
        references = []
        for reference in legacy.references:
            call = resolve_legacy_l1_result(reference.tool_result_id, tool_calls=tool_calls)
            references.append(L1ResultReference(
                tool_call_id=call["tool_call_id"],
                result_sha256=call["outcome_hash"],
                chunk_id=reference.chunk_id,
            ))
        decision = L1AttemptDecision.model_validate({
            **legacy.model_dump(mode="json", exclude={"references"}),
            "references": [reference.model_dump(mode="json") for reference in references],
        })
    else:
        raise ValueError("unsupported frozen L1 decision protocol")
    for reference in decision.references:
        call = calls.get(reference.tool_call_id)
        if call is None or call.get("status") == "pending":
            raise ValueError("persisted L1 reference has no finalized call in this run")
        # A rejected candidate remains valid audit data, even when it cited a
        # failed call. Evidence eligibility belongs to admission and verification.
        _validate_call_outcome(call, expected_digest=reference.result_sha256)
    return decision


def resolve_legacy_l1_result(
    tool_result_id: str, *, tool_calls: Sequence[Mapping[str, object]],
) -> Mapping[str, object]:
    """Read an old result identity without creating a second current identity."""
    matching = []
    for call in tool_calls:
        digest = call.get("outcome_hash")
        if digest is None:
            continue
        if legacy_l1_tool_result_id(
            tool_call_id=call.get("tool_call_id"), result_sha256=digest,
        ) == tool_result_id:
            matching.append(call)
    if len(matching) != 1:
        raise ValueError("legacy L1 result reference has no unique call in this run")
    call = matching[0]
    _validate_call_outcome(call, expected_digest=call.get("outcome_hash"))
    return call


def normalize_l1_finding_arguments(
    arguments: Mapping[str, object],
    *,
    tool_calls: Sequence[Mapping[str, object]],
) -> dict:
    """Read old/current finding sources as pinned calls, preserving stored bytes.

    This validates referenced outcome identity, not whether the recorded source
    was eligible evidence. Failed finding calls must remain readable in history.
    """
    normalized = json.loads(json.dumps(freeze_json(arguments), allow_nan=False))
    calls = _index_calls(tool_calls)
    items = normalized.get("items", [])
    if not isinstance(items, list):
        raise ValueError("persisted L1 finding items are not an array")
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("persisted L1 finding item is not an object")
        sources = item.get("source_refs", [])
        if not isinstance(sources, list):
            raise ValueError("persisted L1 finding sources are not an array")
        for source in sources:
            if not isinstance(source, dict):
                raise ValueError("persisted L1 finding source is not an object")
            identity, digest = source.get("tool_result_id"), source.get("result_sha256")
            if not isinstance(identity, str):
                raise ValueError("persisted L1 finding source has no call identity")
            if identity.startswith("l1result_"):
                call = resolve_legacy_l1_result(identity, tool_calls=tool_calls)
                if digest is not None and digest != call["outcome_hash"]:
                    raise L1StoredResultIntegrityError("legacy L1 finding result digest changed")
                identity, digest = call["tool_call_id"], call["outcome_hash"]
            else:
                call = calls.get(identity)
                if call is None or call.get("status") == "pending" or not isinstance(digest, str):
                    raise ValueError("persisted L1 finding source has no pinned finalized call")
                _validate_call_outcome(call, expected_digest=digest)
            source.update(tool_result_id=identity, result_sha256=digest)
    return normalized


def _index_calls(
    tool_calls: Sequence[Mapping[str, object]],
) -> dict[str, Mapping[str, object]]:
    calls = {}
    for call in tool_calls:
        identity = call.get("tool_call_id")
        if not isinstance(identity, str) or not identity or identity in calls:
            raise ValueError("persisted L1 call identity is missing or duplicated")
        calls[identity] = call
    return calls


def _validate_call_outcome(call: Mapping[str, object], *, expected_digest: object) -> None:
    raw = call.get("outcome_json")
    if (
        not isinstance(raw, str)
        or call.get("outcome_hash") != expected_digest
        or hashlib.sha256(raw.encode("utf-8")).hexdigest() != expected_digest
    ):
        raise L1StoredResultIntegrityError("persisted L1 reference result digest changed")
    try:
        outcome = json.loads(raw)
    except ValueError as exc:
        raise L1StoredResultIntegrityError("persisted L1 result is not JSON") from exc
    if not isinstance(outcome, dict) or outcome.get("status") != call.get("status"):
        raise L1StoredResultIntegrityError("persisted L1 reference outcome disagrees with call status")


__all__ = [
    "LEGACY_L1_ATTEMPT_PROTOCOL_VERSION",
    "L1StoredResultIntegrityError",
    "SUPPORTED_L1_DECISION_PROTOCOLS",
    "decode_l1_decision",
    "legacy_l1_tool_result_id",
    "normalize_l1_finding_arguments",
    "resolve_legacy_l1_result",
]
