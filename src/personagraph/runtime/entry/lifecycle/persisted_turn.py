"""Decode Entry-owned acceptance and L1-recovery receipts.

Lane-neutral scalar and execution-snapshot decoding lives in
``runtime.turn.persisted_projection``.  This module retains the Entry-specific
receipt shapes that bind acceptance and L1 recovery to an ``AcceptedEntryTurn``.
"""

from __future__ import annotations

from typing import Callable

from ...turn import persisted_projection as turn_projection
from ...turn.contracts import (
    AcceptedEntryTurn,
    EntryRecoveryProjection,
    TurnRoutingPolicySnapshot,
)
from ..routing.policy import (
    parse_turn_routing_policy_snapshot,
)


def accepted_entry_turn_from_persisted_receipt(
    *,
    persisted: dict[str, object],
    session_id: str,
    client_request_id: str,
    recovery_projection_supplier: (
        Callable[[], EntryRecoveryProjection | None] | None
    ) = None,
) -> AcceptedEntryTurn:
    """解码 Entry 持久化 acceptance 写入的权威结果。"""

    turn = turn_projection.require_persisted_mapping(persisted, "turn")
    input_message = turn_projection.require_persisted_mapping(
        persisted,
        "input_message",
    )
    window = persisted.get("window")
    if not isinstance(window, dict):
        raise RuntimeError("accepted Turn is missing its execution window")
    replayed = bool(persisted.get("replayed"))
    routing_policy = _routing_policy_from_persisted(persisted.get("routing_policy"))
    return AcceptedEntryTurn(
        session_id=session_id,
        turn_id=str(turn["turn_id"]),
        client_request_id=client_request_id,
        user_input=str(input_message["content"]),
        attachment_ids=tuple(
            str(item["attachment_id"])
            for item in persisted.get("attachments", [])
            if isinstance(item, dict) and item.get("attachment_id")
        ),
        window_revision=int(window.get("state_version") or 0),
        replayed=replayed,
        turn_status=turn_projection.require_entry_turn_status(turn.get("status")),
        processing_level=turn_projection.require_processing_level(
            turn.get("processing_level")
        ),
        end_reason=turn_projection.optional_text(turn.get("end_reason")),
        error_code=turn_projection.optional_text(turn.get("error_code")),
        window_state=turn_projection.require_entry_window_state(
            window.get("window_state")
        ),
        input_message_id=str(input_message["message_id"]),
        recovery_projection=(
            recovery_projection_supplier()
            if recovery_projection_supplier is not None and not replayed
            else None
        ),
        routing_policy=routing_policy,
        execution_snapshot=(
            turn_projection.execution_snapshot_from_persisted_turn(turn)
        ),
    )


def _routing_policy_from_persisted(
    value: object,
) -> TurnRoutingPolicySnapshot:
    if value is None:
        raise RuntimeError("accepted Turn is missing its routing-policy snapshot")
    if not isinstance(value, dict):
        raise RuntimeError("accepted Turn has an invalid routing-policy record")
    payload = value.get("snapshot_json")
    expected_hash = value.get("snapshot_hash")
    if not isinstance(payload, str) or not isinstance(expected_hash, str):
        raise RuntimeError("accepted Turn routing-policy record is incomplete")
    try:
        return parse_turn_routing_policy_snapshot(
            payload,
            expected_sha256=expected_hash,
        )
    except ValueError as exc:
        raise RuntimeError(
            "accepted Turn routing-policy snapshot hash changed"
        ) from exc


def accepted_entry_turn_from_l1_recovery_inspection(
    inspected: dict[str, object],
) -> AcceptedEntryTurn:
    """为 Host 启动恢复重建精确且不可变的 AcceptedTurn。"""

    turn = turn_projection.require_persisted_mapping(inspected, "turn")
    client_request_id = str(turn.get("client_request_id") or "")
    session_id = str(turn.get("session_id") or "")
    if not client_request_id or not session_id:
        raise RuntimeError("L1 recovery inspection lost request identity")
    persisted = dict(inspected)
    persisted["replayed"] = True
    return accepted_entry_turn_from_persisted_receipt(
        persisted=persisted,
        session_id=session_id,
        client_request_id=client_request_id,
    )


__all__ = [
    "accepted_entry_turn_from_l1_recovery_inspection",
    "accepted_entry_turn_from_persisted_receipt",
]
