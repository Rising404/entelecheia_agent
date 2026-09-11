"""Bind default visual egress to its exact source and provider, without consent state."""

from __future__ import annotations

import hashlib
import json

from ...input_processing.vision.contracts import VisionPurpose


def auto_visual_egress_receipt(
    *,
    session_id: str,
    source_sha256: str,
    endpoint_identity: str,
    model: str,
    purpose: VisionPurpose,
) -> str:
    """Deterministic execution proof, not a user approval or a file access grant.

    Existing durable visual-call contracts retain their receipt field. Its value
    now identifies the default-egress policy and immutable execution inputs;
    callers still validate source access, freshness and provider binding.
    """

    values = (session_id, source_sha256, endpoint_identity, model, purpose.value)
    if any(not isinstance(value, str) or not value.strip() for value in values):
        raise ValueError("visual egress requires Session, source, provider and purpose")
    digest = hashlib.sha256(json.dumps(values, ensure_ascii=False).encode("utf-8")).hexdigest()
    return f"auto_visual_egress_{digest}"
