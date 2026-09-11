"""一个已接受 Entry Turn 的有界上下文契约。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ...turn.contracts import (
    DEFAULT_TURN_ROUTING_POLICY_SNAPSHOT,
    EntryRecoveryProjection,
    TurnRoutingPolicySnapshot,
)
from ..ingress.contracts import (
    AuthoritativeRuntimeSnapshot,
    CapabilityCeiling,
    TrustedTurnEnvelope,
)
from .attachments import AttachmentProjection
from .task_catalog import EntryTaskCatalog


@dataclass(frozen=True)
class EntryContext:
    """恰好一个 entry turn 的可信输入与有界先前上下文。"""

    envelope: TrustedTurnEnvelope
    snapshot: AuthoritativeRuntimeSnapshot
    ceiling: CapabilityCeiling
    estimated_input_tokens: int
    history_pairs: tuple[dict[str, str], ...]
    session_summary: str | None
    session_summary_status: Literal["empty", "ok", "stale", "unavailable"] = "empty"
    attachments: AttachmentProjection = AttachmentProjection()
    recovery_projection: EntryRecoveryProjection | None = None
    task_catalog: EntryTaskCatalog = EntryTaskCatalog()
    routing_policy: TurnRoutingPolicySnapshot = DEFAULT_TURN_ROUTING_POLICY_SNAPSHOT


__all__ = ["EntryContext"]
