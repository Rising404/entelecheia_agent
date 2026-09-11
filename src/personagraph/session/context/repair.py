"""SessionContext 的确定性影子重建与纠正门面。

本模块在 LangGraph 外组合公开证据、抽取器、reducer 和上下文存储契约。试运行绝不写入；
受控应用会把最终视图和审计替换委托给一个 SQLite 事务。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace as dataclass_replace
from typing import Callable, Sequence

from .. import store
from . import catalog as context_catalog
from . import evidence, extractor, reducer
from . import reset as context_reset
from . import store as context_store
from .models import (
    EvidenceKind,
    EvidenceRecord,
    ObservationCandidate,
    Operation,
    SessionDomain,
    SessionExtractionResult,
    SessionStateItem,
    SourceKind,
    TransitionAudit,
)


CORRECTION_VERSION = "manual-correction-v1"
CandidateExtractor = Callable[[str, Sequence[EvidenceRecord]], SessionExtractionResult]
StateSlot = tuple[SessionDomain, str, str]


class RepairError(RuntimeError):
    """无法证明重建完整时，在替换前抛出。"""


@dataclass(frozen=True)
class RepairResult:
    """一次修复尝试的可序列化影子差异与应用状态。"""

    session_id: str
    dry_run: bool
    applied: bool
    extractor_version: str
    candidate_source: str
    candidate_count: int
    transition_count: int
    diff: dict
    changes: dict
    preview_token: str
    context_revision: int
    applied_revision: int | None = None
    already_applied: bool = False
    slot: dict | None = None
    reset_id: str | None = None
    cutoff_turn_idx: int | None = None

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "dry_run": self.dry_run,
            "applied": self.applied,
            "extractor_version": self.extractor_version,
            "candidate_source": self.candidate_source,
            "candidate_count": self.candidate_count,
            "transition_count": self.transition_count,
            "diff": self.diff,
            "changes": self.changes,
            "preview_token": self.preview_token,
            "context_revision": self.context_revision,
            "applied_revision": self.applied_revision,
            "already_applied": self.already_applied,
            "slot": self.slot,
            "reset_id": self.reset_id,
            "cutoff_turn_idx": self.cutoff_turn_idx,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "RepairResult":
        return cls(
            session_id=str(data["session_id"]),
            dry_run=bool(data["dry_run"]),
            applied=bool(data["applied"]),
            extractor_version=str(data["extractor_version"]),
            candidate_source=str(data["candidate_source"]),
            candidate_count=int(data["candidate_count"]),
            transition_count=int(data["transition_count"]),
            diff=dict(data.get("diff") or {}),
            changes=dict(data.get("changes") or {}),
            preview_token=str(data["preview_token"]),
            context_revision=int(data["context_revision"]),
            applied_revision=(
                int(data["applied_revision"])
                if data.get("applied_revision") is not None else None
            ),
            already_applied=bool(data.get("already_applied", False)),
            slot=dict(data["slot"]) if data.get("slot") else None,
            reset_id=str(data["reset_id"]) if data.get("reset_id") else None,
            cutoff_turn_idx=(
                int(data["cutoff_turn_idx"])
                if data.get("cutoff_turn_idx") is not None else None
            ),
        )


def record_correction(
    session_id: str,
    domain: SessionDomain,
    state_type: str,
    key: str,
    value,
    *,
    operation: Operation = Operation.SET,
    actor: str = "user",
) -> EvidenceRecord:
    """追加结构化用户纠正，不修改既有证据或视图。"""
    validate_correction(domain, state_type, key, value, operation=operation)
    actor = actor.strip()
    if not actor or len(actor) > 200:
        raise ValueError("correction actor is invalid")
    payload = json.dumps(
        {
            "domain": domain.value,
            "state_type": state_type,
            "key": key,
            "value": value,
            "operation": operation.value,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    if len(payload) > evidence.MAX_EVIDENCE_EXCERPT_CHARS:
        raise ValueError("correction payload exceeds recoverable evidence excerpt limit")
    source_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]
    return evidence.append_event(
        session_id,
        EvidenceKind.CORRECTION,
        f"manual-correction:{source_hash}",
        payload,
        metadata={"actor": actor, "schema_version": CORRECTION_VERSION},
    )


def validate_correction(
    domain: SessionDomain,
    state_type: str,
    key: str,
    value,
    *,
    operation: Operation,
) -> None:
    """使用与 reducer 相同的目录校验手动纠正。"""
    state_type = state_type.strip()
    key = key.strip()
    if not state_type or not key or len(state_type) > 120 or len(key) > 200:
        raise ValueError("correction state_type/key is invalid")
    policy = context_catalog.policy_for(domain, state_type)
    if policy is None:
        raise ValueError("correction state type is not in the SessionContext catalog")
    if SourceKind.EXPLICIT not in policy.allowed_sources:
        raise ValueError("correction source is not allowed for this state type")
    if SourceKind.EXPLICIT not in policy.auto_apply_sources:
        raise ValueError("correction source requires unsupported verification")
    if operation not in policy.allowed_operations:
        raise ValueError("correction operation is not allowed for this state type")
    if operation in {Operation.SET, Operation.APPEND} and value is None:
        raise ValueError("correction value is required for set/append")
    try:
        json.dumps(value, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise ValueError("correction value must be JSON serializable") from exc


def repair_session(
    session_id: str,
    *,
    dry_run: bool = True,
    force_reextract: bool = False,
    candidate_extractor: CandidateExtractor | None = None,
    slot: StateSlot | None = None,
    expected_preview_token: str | None = None,
    allow_reextract: bool = True,
    apply_actor: str = "system",
) -> RepairResult:
    """构建影子视图、返回差异，并可选择原子替换。"""
    if store.get_session(session_id) is None:
        raise RepairError(f"unknown session: {session_id}")
    if not dry_run and not expected_preview_token:
        raise RepairError("preview_token_required")
    if not dry_run and expected_preview_token:
        recorded = context_store.get_repair_apply(expected_preview_token)
        if recorded is not None:
            result = RepairResult.from_dict(recorded)
            requested_slot = (
                {"domain": slot[0].value, "state_type": slot[1], "key": slot[2]}
                if slot else None
            )
            if result.session_id != session_id or result.slot != requested_slot:
                raise RepairError("preview_token_scope_collision")
            return dataclass_replace(result, already_applied=True)
    if force_reextract and not allow_reextract:
        raise RepairError("reextract_not_allowed")
    start_revision = context_store.context_revision(session_id)
    all_evidence, reset_boundary = context_reset.evidence_after_latest_reset(
        session_id,
        evidence.list_evidence(session_id),
    )
    allowed_evidence_ids = {record.id for record in all_evidence}
    corrections = _correction_candidates(session_id, all_evidence)
    stored = [
        candidate
        for candidate in context_store.list_candidates(
            session_id,
            extractor_version=extractor.EXTRACTOR_VERSION,
        )
        if candidate.derived_from
        and all(ref in allowed_evidence_ids for ref in candidate.derived_from)
    ]
    if (stored or corrections) and not force_reextract:
        candidates = [*stored, *corrections]
        candidate_source = "stored" if stored else "corrections"
    else:
        if not allow_reextract:
            raise RepairError("reextract_required")
        extracted = _extract_from_evidence(
            session_id,
            all_evidence,
            candidate_extractor=candidate_extractor,
        )
        candidates = [*extracted, *corrections]
        candidate_source = "reextracted"
    candidates = _deduplicate_candidates(candidates)

    old_items = context_store.list_state_items(session_id)
    old_audits = context_store.list_transition_audits(session_id)
    replay_candidates = candidates
    base_items: list[SessionStateItem] = []
    base_audits: list[TransitionAudit] = []
    if slot is not None:
        replay_candidates = [candidate for candidate in candidates if _slot(candidate) == slot]
        base_items = [item for item in old_items if _item_slot(item) != slot]
        known_target_ids = {
            candidate.candidate_id for candidate in candidates if _slot(candidate) == slot
        }
        base_audits = [audit for audit in old_audits if audit.candidate_id not in known_target_ids]

    shadow_items, shadow_audits = _replay(
        session_id,
        replay_candidates,
        all_evidence,
        base_items=base_items,
        base_audits=base_audits,
    )
    diff = _view_diff(old_items, shadow_items)
    changes = _view_changes(old_items, shadow_items)
    end_revision = context_store.context_revision(session_id)
    if end_revision != start_revision:
        raise RepairError(
            f"session_context_changed_during_preview:{start_revision}:{end_revision}"
        )
    preview_token = _build_preview_token(
        session_id=session_id,
        context_revision=start_revision,
        slot=slot,
        force_reextract=force_reextract,
        candidate_source=candidate_source,
        evidence_records=all_evidence,
        candidates=candidates,
        old_items=old_items,
        old_audits=old_audits,
        shadow_items=shadow_items,
        shadow_audits=shadow_audits,
        reset_boundary=reset_boundary,
    )
    result = RepairResult(
        session_id=session_id,
        dry_run=dry_run,
        applied=not dry_run,
        extractor_version=extractor.EXTRACTOR_VERSION,
        candidate_source=candidate_source,
        candidate_count=len(replay_candidates),
        transition_count=len(shadow_audits) - len(base_audits),
        diff=diff,
        changes=changes,
        preview_token=preview_token,
        context_revision=start_revision,
        slot=(
            {"domain": slot[0].value, "state_type": slot[1], "key": slot[2]}
            if slot else None
        ),
        reset_id=str(reset_boundary["reset_id"]) if reset_boundary else None,
        cutoff_turn_idx=int(reset_boundary["cutoff_turn_idx"]) if reset_boundary else None,
    )
    if not dry_run:
        if expected_preview_token != preview_token:
            raise RepairError("preview_stale")
        try:
            applied_revision = context_store.replace_session_context(
                session_id,
                candidates=candidates,
                state_items=shadow_items,
                audits=shadow_audits,
                expected_revision=start_revision,
                repair_apply={
                    "preview_token": preview_token,
                    "actor": apply_actor,
                    "result": result.to_dict(),
                },
            )
        except context_store.RepairApplyAlreadyRecorded as exc:
            replay = RepairResult.from_dict(exc.result)
            if replay.session_id != session_id or replay.slot != result.slot:
                raise RepairError("preview_token_scope_collision") from exc
            return dataclass_replace(replay, already_applied=True)
        except context_store.ContextRevisionConflict as exc:
            raise RepairError(
                f"preview_stale_revision:{exc.expected}:{exc.actual}"
            ) from exc
        result = dataclass_replace(result, applied_revision=applied_revision)
    return result


def has_recorded_apply(session_id: str, preview_token: str | None) -> bool:
    if not preview_token:
        return False
    recorded = context_store.get_repair_apply(preview_token)
    return recorded is not None and str(recorded.get("session_id")) == session_id


def _extract_from_evidence(
    session_id: str,
    records: Sequence[EvidenceRecord],
    *,
    candidate_extractor: CandidateExtractor | None,
) -> list[ObservationCandidate]:
    if candidate_extractor is not None:
        result = candidate_extractor(session_id, records)
        if result.error:
            raise RepairError(f"extractor failed: {result.error}")
        return list(result.candidates)

    turns = [record for record in records if record.kind in {
        EvidenceKind.USER_TURN, EvidenceKind.ASSISTANT_TURN,
    }]
    output: list[ObservationCandidate] = []
    for index, user_record in enumerate(turns):
        if user_record.kind != EvidenceKind.USER_TURN:
            continue
        next_user_time = next(
            (item.created_at for item in turns[index + 1:] if item.kind == EvidenceKind.USER_TURN),
            None,
        )
        assistant_record = next(
            (item for item in turns[index + 1:] if item.kind == EvidenceKind.ASSISTANT_TURN
             and (next_user_time is None or item.created_at < next_user_time)),
            None,
        )
        batch = [
            item for item in records
            if item.created_at >= user_record.created_at
            and (next_user_time is None or item.created_at < next_user_time)
        ]
        result = extractor.extract_candidates(
            session_id,
            user_record.content_excerpt,
            assistant_record.content_excerpt if assistant_record else "",
            batch,
        )
        if result.error:
            raise RepairError(f"extractor failed: {result.error}")
        output.extend(result.candidates)
    return output


def _correction_candidates(
    session_id: str,
    records: Sequence[EvidenceRecord],
) -> list[ObservationCandidate]:
    output = []
    for record in records:
        if record.kind != EvidenceKind.CORRECTION:
            continue
        try:
            payload = json.loads(record.content_excerpt)
            domain = SessionDomain(str(payload["domain"]))
            operation = Operation(str(payload["operation"]))
            state_type = str(payload["state_type"])
            key = str(payload["key"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise RepairError(f"invalid correction evidence: {record.id}") from exc
        raw = "\x1f".join((record.id, domain, state_type, key, operation, CORRECTION_VERSION))
        output.append(ObservationCandidate(
            candidate_id="candidate:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20],
            session_id=session_id,
            domain=domain,
            state_type=state_type,
            key=key,
            proposed_value=payload.get("value"),
            operation=operation,
            source_kind=SourceKind.EXPLICIT,
            derived_from=(record.id,),
            extractor_version=CORRECTION_VERSION,
            valid_from=record.created_at,
        ))
    return output


def _replay(
    session_id: str,
    candidates: Sequence[ObservationCandidate],
    records: Sequence[EvidenceRecord],
    *,
    base_items: Sequence[SessionStateItem],
    base_audits: Sequence[TransitionAudit],
) -> tuple[list[SessionStateItem], list[TransitionAudit]]:
    by_evidence = {record.id: record for record in records}
    by_slot = {_item_slot(item): item for item in base_items}
    audits = list(base_audits)
    for candidate in sorted(candidates, key=lambda item: (item.valid_from or "", item.candidate_id)):
        if candidate.session_id != session_id:
            raise RepairError(f"candidate scope mismatch: {candidate.candidate_id}")
        refs = [by_evidence.get(ref) for ref in candidate.derived_from]
        if not refs or any(record is None or record.session_id != session_id for record in refs):
            raise RepairError(f"candidate evidence missing: {candidate.candidate_id}")
        slot = _slot(candidate)
        timestamp = candidate.valid_from or max(record.created_at for record in refs if record)
        result = reducer.reduce_candidate(candidate, by_slot.get(slot), now=timestamp)
        if result.state_item is not None:
            by_slot[slot] = result.state_item
        audits.append(result.audit)
    items = sorted(by_slot.values(), key=lambda item: (*_item_slot(item), item.id))
    audits.sort(key=lambda audit: (audit.created_at, audit.transition_id))
    return items, audits


def _view_diff(old: Sequence[SessionStateItem], new: Sequence[SessionStateItem]) -> dict:
    old_map = {item.id: item.to_dict() for item in old}
    new_map = {item.id: item.to_dict() for item in new}
    added = sorted(item_id for item_id in new_map if item_id not in old_map)
    removed = sorted(item_id for item_id in old_map if item_id not in new_map)
    changed = sorted(
        item_id for item_id in old_map.keys() & new_map.keys()
        if old_map[item_id] != new_map[item_id]
    )
    return {
        "added": added,
        "removed": removed,
        "changed": changed,
        "unchanged_count": sum(
            1 for item_id in old_map.keys() & new_map.keys()
            if old_map[item_id] == new_map[item_id]
        ),
    }


def _view_changes(old: Sequence[SessionStateItem], new: Sequence[SessionStateItem]) -> dict:
    old_map = {item.id: item.to_dict() for item in old}
    new_map = {item.id: item.to_dict() for item in new}
    return {
        "added": [new_map[item_id] for item_id in sorted(new_map.keys() - old_map.keys())],
        "removed": [old_map[item_id] for item_id in sorted(old_map.keys() - new_map.keys())],
        "changed": [
            {"id": item_id, "before": old_map[item_id], "after": new_map[item_id]}
            for item_id in sorted(old_map.keys() & new_map.keys())
            if old_map[item_id] != new_map[item_id]
        ],
    }


def _build_preview_token(
    *,
    session_id: str,
    context_revision: int,
    slot: StateSlot | None,
    force_reextract: bool,
    candidate_source: str,
    evidence_records: Sequence[EvidenceRecord],
    candidates: Sequence[ObservationCandidate],
    old_items: Sequence[SessionStateItem],
    old_audits: Sequence[TransitionAudit],
    shadow_items: Sequence[SessionStateItem],
    shadow_audits: Sequence[TransitionAudit],
    reset_boundary: dict | None,
) -> str:
    payload = {
        "session_id": session_id,
        "context_revision": context_revision,
        "slot": [slot[0].value, slot[1], slot[2]] if slot else None,
        "force_reextract": force_reextract,
        "candidate_source": candidate_source,
        "evidence": [
            {
                "id": record.id,
                "kind": record.kind.value,
                "content_hash": record.content_hash,
                "created_at": record.created_at,
            }
            for record in evidence_records
        ],
        "candidates": [candidate.to_dict() for candidate in candidates],
        "old_items": [item.to_dict() for item in old_items],
        "old_audits": [audit.to_dict() for audit in old_audits],
        "shadow_items": [item.to_dict() for item in shadow_items],
        "shadow_audits": [audit.to_dict() for audit in shadow_audits],
        "reset_id": reset_boundary.get("reset_id") if reset_boundary else None,
        "extractor_version": extractor.EXTRACTOR_VERSION,
    }
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return "repair-preview:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _deduplicate_candidates(
    candidates: Sequence[ObservationCandidate],
) -> list[ObservationCandidate]:
    by_id: dict[str, ObservationCandidate] = {}
    for candidate in candidates:
        existing = by_id.get(candidate.candidate_id)
        if existing is not None and existing != candidate:
            raise RepairError(f"candidate id collision: {candidate.candidate_id}")
        by_id[candidate.candidate_id] = candidate
    return sorted(by_id.values(), key=lambda item: (item.valid_from or "", item.candidate_id))


def _slot(candidate: ObservationCandidate) -> StateSlot:
    return candidate.domain, candidate.state_type, candidate.key


def _item_slot(item: SessionStateItem) -> StateSlot:
    return item.domain, item.state_type, item.key
