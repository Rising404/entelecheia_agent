"""SessionContext 的只读检查、说明和导出视图。"""

from __future__ import annotations

from typing import Any

from . import catalog as context_catalog
from . import evidence
from . import reset as context_reset
from . import store as context_store
from .models import Operation, SessionDomain, SourceKind, StateStatus


EXPORT_SCHEMA_VERSION = 3


def view_session_context(
    session_id: str,
    *,
    include_inactive: bool = False,
) -> dict[str, Any]:
    statuses = None if include_inactive else (StateStatus.ACTIVE, StateStatus.CONFLICTED)
    items = context_store.list_state_items(session_id, statuses=statuses)
    grouped = {"user_state": [], "task_state": [], "interaction_state": []}
    domain_names = {
        SessionDomain.USER: "user_state",
        SessionDomain.TASK: "task_state",
        SessionDomain.INTERACTION: "interaction_state",
    }
    for item in items:
        payload = item.to_dict()
        policy = context_catalog.policy_for(item.domain, item.state_type)
        payload["correction_operations"] = [
            operation.value
            for operation in Operation
            if policy is not None
            and SourceKind.EXPLICIT in policy.allowed_sources
            and SourceKind.EXPLICIT in policy.auto_apply_sources
            and operation in policy.allowed_operations
        ]
        grouped[domain_names[item.domain]].append(payload)
    return {
        "session_id": session_id,
        **grouped,
        "latest_reset": context_reset.latest_reset(session_id),
        "counts": {
            "user_state": len(grouped["user_state"]),
            "task_state": len(grouped["task_state"]),
            "interaction_state": len(grouped["interaction_state"]),
            "total": len(items),
        },
        "include_inactive": include_inactive,
    }


def explain_state(
    session_id: str,
    domain: SessionDomain,
    state_type: str,
    key: str,
    *,
    include_evidence_excerpt: bool = False,
) -> dict[str, Any] | None:
    item = context_store.get_state_item(session_id, domain, state_type, key)
    if item is None:
        return None
    evidence_rows = []
    missing = []
    for evidence_id in item.derived_from:
        record = evidence.get_evidence(evidence_id)
        if record is None or record.session_id != session_id:
            missing.append(evidence_id)
            evidence_rows.append({"id": evidence_id, "status": "missing"})
            continue
        evidence_rows.append(
            _evidence_view(record, include_excerpt=include_evidence_excerpt)
        )
    return {
        "session_id": session_id,
        "state": item.to_dict(),
        "explanation": {
            "basis": "exact_derived_from",
            "support_status": "complete" if not missing else "incomplete",
            "evidence_count": len(evidence_rows),
            "missing_evidence_ids": missing,
            "evidence": evidence_rows,
        },
    }


def export_session_context(
    session_id: str,
    *,
    include_evidence_excerpt: bool = False,
) -> dict[str, Any]:
    evidence_rows = [
        _evidence_view(record, include_excerpt=include_evidence_excerpt)
        for record in evidence.list_evidence(session_id)
    ]
    state_items = [item.to_dict() for item in context_store.list_state_items(session_id)]
    candidates = [candidate.to_dict() for candidate in context_store.list_candidates(session_id)]
    transitions = [audit.to_dict() for audit in context_store.list_transition_audits(session_id)]
    resets = context_reset.list_resets(session_id)
    return {
        "schema_version": EXPORT_SCHEMA_VERSION,
        "session_id": session_id,
        "state_items": state_items,
        "candidates": candidates,
        "transitions": transitions,
        "resets": resets,
        "evidence": evidence_rows,
        "content_policy": {
            "includes_transcript": False,
            "includes_evidence_excerpt": include_evidence_excerpt,
            "evidence_metadata_included": True,
        },
        "counts": {
            "state_items": len(state_items),
            "candidates": len(candidates),
            "transitions": len(transitions),
            "resets": len(resets),
            "evidence": len(evidence_rows),
        },
    }


def _evidence_view(record, *, include_excerpt: bool) -> dict[str, Any]:
    payload = {
        "id": record.id,
        "session_id": record.session_id,
        "kind": record.kind.value,
        "source_ref": record.source_ref,
        "content_hash": record.content_hash,
        "created_at": record.created_at,
        "metadata": dict(record.metadata),
        "status": "available",
    }
    if include_excerpt:
        payload["content_excerpt"] = record.content_excerpt
    return payload
