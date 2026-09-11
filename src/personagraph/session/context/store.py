"""SessionContext 物化视图与审计的 SQLite 持久化。

本模块原子存储已经归约的 TransitionResult 值。它不分类文本、不应用转换策略、不组装
Prompt，也不访问 Graph。
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping

from .. import store as session_store
from .models import (
    ObservationCandidate,
    Operation,
    ReasonCode,
    SessionDomain,
    SessionStateItem,
    SourceKind,
    StateStatus,
    TransitionAudit,
    TransitionDecision,
    TransitionResult,
)


class ContextRevisionConflict(RuntimeError):
    """受控替换不再匹配其预览 revision 时抛出。"""

    def __init__(self, expected: int, actual: int) -> None:
        super().__init__(f"context revision changed: expected {expected}, actual {actual}")
        self.expected = expected
        self.actual = actual


class RepairApplyAlreadyRecorded(RuntimeError):
    """携带精确应用重试的原子存储结果。"""

    def __init__(self, result: dict[str, Any]) -> None:
        super().__init__("repair apply already recorded")
        self.result = result


def _connect() -> sqlite3.Connection:
    return session_store.connect_session_context_authority()


def _json(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _context_revision(conn: sqlite3.Connection, session_id: str) -> int:
    row = conn.execute(
        "SELECT revision FROM session_context_revisions WHERE session_id=?",
        (session_id,),
    ).fetchone()
    return int(row["revision"]) if row is not None else 0


def context_revision(session_id: str) -> int:
    """返回覆盖所有修复相关 Session 写入的单调 revision。"""
    with _connect() as conn:
        return _context_revision(conn, session_id)


def _repair_apply_result(row: sqlite3.Row) -> dict[str, Any]:
    value = json.loads(str(row["result_json"]))
    if not isinstance(value, dict):
        raise ValueError("repair apply result is invalid")
    return value


def get_repair_apply(preview_token: str) -> dict[str, Any] | None:
    """返回精确 token 重试对应的原子记录修复结果。"""
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM session_context_repair_applies WHERE preview_token=?",
            (preview_token,),
        ).fetchone()
    return _repair_apply_result(row) if row is not None else None


def _item_from_row(row: sqlite3.Row) -> SessionStateItem:
    return SessionStateItem(
        id=str(row["id"]),
        session_id=str(row["session_id"]),
        domain=SessionDomain(str(row["domain"])),
        state_type=str(row["state_type"]),
        key=str(row["state_key"]),
        value_json=json.loads(str(row["value_json"])),
        status=StateStatus(str(row["status"])),
        source_kind=SourceKind(str(row["source_kind"])),
        derived_from=tuple(json.loads(str(row["derived_from_json"]))),
        extractor_version=str(row["extractor_version"]),
        reducer_version=str(row["reducer_version"]),
        valid_from=str(row["valid_from"]),
        expires_at=str(row["expires_at"]) if row["expires_at"] is not None else None,
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
    )


def _audit_from_row(row: sqlite3.Row) -> TransitionAudit:
    return TransitionAudit(
        transition_id=str(row["transition_id"]),
        session_id=str(row["session_id"]),
        candidate_id=str(row["candidate_id"]),
        old_state_ref=str(row["old_state_ref"]) if row["old_state_ref"] is not None else None,
        new_state_ref=str(row["new_state_ref"]) if row["new_state_ref"] is not None else None,
        decision=TransitionDecision(str(row["decision"])),
        reason_code=ReasonCode(str(row["reason_code"])),
        derived_from=tuple(json.loads(str(row["derived_from_json"]))),
        reducer_version=str(row["reducer_version"]),
        created_at=str(row["created_at"]),
    )


def _candidate_from_row(row: sqlite3.Row) -> ObservationCandidate:
    return ObservationCandidate(
        candidate_id=str(row["candidate_id"]),
        session_id=str(row["session_id"]),
        domain=SessionDomain(str(row["domain"])),
        state_type=str(row["state_type"]),
        key=str(row["state_key"]),
        proposed_value=json.loads(str(row["proposed_value_json"])),
        operation=Operation(str(row["operation"])),
        source_kind=SourceKind(str(row["source_kind"])),
        derived_from=tuple(json.loads(str(row["derived_from_json"]))),
        extractor_version=str(row["extractor_version"]),
        confidence_hint=float(row["confidence_hint"]),
        valid_from=str(row["valid_from"]) if row["valid_from"] is not None else None,
        expires_at=str(row["expires_at"]) if row["expires_at"] is not None else None,
    )


def _insert_candidate(conn: sqlite3.Connection, candidate: ObservationCandidate) -> None:
    existing = conn.execute(
        "SELECT * FROM session_observation_candidates WHERE candidate_id=?",
        (candidate.candidate_id,),
    ).fetchone()
    if existing is not None:
        if _candidate_from_row(existing) != candidate:
            raise ValueError(f"candidate id collision: {candidate.candidate_id}")
        return
    conn.execute(
        "INSERT INTO session_observation_candidates"
        " (candidate_id, session_id, domain, state_type, state_key, proposed_value_json,"
        " operation, source_kind, derived_from_json, extractor_version, confidence_hint,"
        " valid_from, expires_at, created_at)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            candidate.candidate_id,
            candidate.session_id,
            candidate.domain,
            candidate.state_type,
            candidate.key,
            _json(candidate.proposed_value),
            candidate.operation,
            candidate.source_kind,
            _json(candidate.derived_from),
            candidate.extractor_version,
            candidate.confidence_hint,
            candidate.valid_from,
            candidate.expires_at,
            datetime.now(timezone.utc).isoformat(),
        ),
    )


def save_candidates(candidates: Iterable[ObservationCandidate]) -> None:
    """以原子且幂等方式持久化已验证抽取器候选。"""
    values = tuple(candidates)
    if not values:
        return
    session_ids = {candidate.session_id for candidate in values}
    if len(session_ids) != 1 or session_store.get_session(next(iter(session_ids))) is None:
        raise ValueError("candidates must belong to one existing session")
    with _connect() as conn:
        for candidate in values:
            _insert_candidate(conn, candidate)


def list_candidates(
    session_id: str,
    *,
    extractor_version: str | None = None,
) -> list[ObservationCandidate]:
    """按确定性证据时间和 ID 顺序返回已存储候选。"""
    query = "SELECT * FROM session_observation_candidates WHERE session_id=?"
    params: list[object] = [session_id]
    if extractor_version is not None:
        query += " AND extractor_version=?"
        params.append(extractor_version)
    query += " ORDER BY COALESCE(valid_from, ''), candidate_id"
    with _connect() as conn:
        rows = conn.execute(query, params).fetchall()
    return [_candidate_from_row(row) for row in rows]


def _validate_result(result: TransitionResult) -> None:
    item, audit = result.state_item, result.audit
    if session_store.get_session(audit.session_id) is None:
        raise ValueError(f"unknown session: {audit.session_id}")
    if item is not None and item.session_id != audit.session_id:
        raise ValueError("state item and audit belong to different sessions")
    if item is not None and audit.new_state_ref != item.id:
        raise ValueError("audit new_state_ref does not match state item")
    if item is not None and item.status == StateStatus.ACTIVE and not item.derived_from:
        raise ValueError("active state item must reference evidence")


def _upsert_item(conn: sqlite3.Connection, item: SessionStateItem) -> None:
    conn.execute(
        "INSERT INTO session_state_items"
        " (id, session_id, domain, state_type, state_key, value_json, status, source_kind,"
        " derived_from_json, extractor_version, reducer_version, valid_from, expires_at,"
        " created_at, updated_at)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
        " ON CONFLICT(id) DO UPDATE SET"
        " value_json=excluded.value_json, status=excluded.status,"
        " source_kind=excluded.source_kind, derived_from_json=excluded.derived_from_json,"
        " extractor_version=excluded.extractor_version, reducer_version=excluded.reducer_version,"
        " valid_from=excluded.valid_from, expires_at=excluded.expires_at,"
        " updated_at=excluded.updated_at",
        (
            item.id,
            item.session_id,
            item.domain,
            item.state_type,
            item.key,
            _json(item.value_json),
            item.status,
            item.source_kind,
            _json(item.derived_from),
            item.extractor_version,
            item.reducer_version,
            item.valid_from,
            item.expires_at,
            item.created_at,
            item.updated_at,
        ),
    )


def _insert_audit(conn: sqlite3.Connection, audit: TransitionAudit) -> None:
    existing = conn.execute(
        "SELECT * FROM session_state_transitions WHERE transition_id=?",
        (audit.transition_id,),
    ).fetchone()
    if existing is not None:
        if _audit_from_row(existing) != audit:
            raise ValueError(f"transition id collision: {audit.transition_id}")
        return
    conn.execute(
        "INSERT INTO session_state_transitions"
        " (transition_id, session_id, candidate_id, old_state_ref, new_state_ref,"
        " decision, reason_code, derived_from_json, reducer_version, created_at)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            audit.transition_id,
            audit.session_id,
            audit.candidate_id,
            audit.old_state_ref,
            audit.new_state_ref,
            audit.decision,
            audit.reason_code,
            _json(audit.derived_from),
            audit.reducer_version,
            audit.created_at,
        ),
    )


def apply_transition(result: TransitionResult) -> None:
    """原子持久化一个 reducer 结果；精确重试具有幂等性。"""
    apply_transitions((result,))


def apply_transitions(results: Iterable[TransitionResult]) -> None:
    """在单一事务中持久化一个确定性转换批次。"""
    values = tuple(results)
    for result in values:
        _validate_result(result)
    with _connect() as conn:
        for result in values:
            if result.state_item is not None:
                _upsert_item(conn, result.state_item)
            _insert_audit(conn, result.audit)


def replace_session_context(
    session_id: str,
    *,
    candidates: Iterable[ObservationCandidate],
    state_items: Iterable[SessionStateItem],
    audits: Iterable[TransitionAudit],
    expected_revision: int | None = None,
    repair_apply: Mapping[str, Any] | None = None,
) -> int:
    """原子替换一个 Session 的视图与审计，并保留带版本候选。"""
    candidate_values = tuple(candidates)
    item_values = tuple(state_items)
    audit_values = tuple(audits)
    if session_store.get_session(session_id) is None:
        raise ValueError(f"unknown session: {session_id}")
    if any(candidate.session_id != session_id for candidate in candidate_values):
        raise ValueError("candidate scope mismatch")
    if any(item.session_id != session_id for item in item_values):
        raise ValueError("state item scope mismatch")
    if any(audit.session_id != session_id for audit in audit_values):
        raise ValueError("audit scope mismatch")
    apply_token = str(repair_apply.get("preview_token", "")).strip() if repair_apply else ""
    apply_actor = str(repair_apply.get("actor", "")).strip() if repair_apply else ""
    apply_result = repair_apply.get("result") if repair_apply else None
    if repair_apply and (
        not apply_token or not apply_actor or not isinstance(apply_result, Mapping)
    ):
        raise ValueError("repair apply ledger metadata is invalid")
    with _connect() as conn:
        if expected_revision is not None or repair_apply is not None:
            conn.execute("BEGIN IMMEDIATE")
        if repair_apply is not None:
            existing_apply = conn.execute(
                "SELECT * FROM session_context_repair_applies WHERE preview_token=?",
                (apply_token,),
            ).fetchone()
            if existing_apply is not None:
                if str(existing_apply["session_id"]) != session_id:
                    raise ValueError("repair preview token scope collision")
                raise RepairApplyAlreadyRecorded(_repair_apply_result(existing_apply))
        if expected_revision is not None:
            actual_revision = _context_revision(conn, session_id)
            if actual_revision != expected_revision:
                raise ContextRevisionConflict(expected_revision, actual_revision)
        for candidate in candidate_values:
            _insert_candidate(conn, candidate)
        conn.execute("DELETE FROM session_state_transitions WHERE session_id=?", (session_id,))
        conn.execute("DELETE FROM session_state_items WHERE session_id=?", (session_id,))
        for item in item_values:
            _upsert_item(conn, item)
        for audit in audit_values:
            _insert_audit(conn, audit)
        applied_revision = _context_revision(conn, session_id)
        if repair_apply is not None:
            stored_result = dict(apply_result)
            stored_result["applied_revision"] = applied_revision
            stored_result["already_applied"] = False
            conn.execute(
                "INSERT INTO session_context_repair_applies"
                " (preview_token, session_id, preview_revision, applied_revision, actor,"
                " applied_at, result_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    apply_token,
                    session_id,
                    int(expected_revision or 0),
                    applied_revision,
                    apply_actor,
                    datetime.now(timezone.utc).isoformat(),
                    _json(stored_result),
                ),
            )
        return applied_revision


def get_state_item(
    session_id: str,
    domain: SessionDomain,
    state_type: str,
    key: str,
) -> SessionStateItem | None:
    """按 Session 作用域物化视图键返回一个条目。"""
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM session_state_items"
            " WHERE session_id=? AND domain=? AND state_type=? AND state_key=?",
            (session_id, domain, state_type, key),
        ).fetchone()
    return _item_from_row(row) if row else None


def list_state_items(
    session_id: str,
    *,
    domain: SessionDomain | None = None,
    statuses: Iterable[StateStatus] | None = None,
) -> list[SessionStateItem]:
    """按稳定领域、类型和键顺序列出条目，并支持可选过滤器。"""
    clauses = ["session_id=?"]
    params: list[object] = [session_id]
    if domain is not None:
        clauses.append("domain=?")
        params.append(domain)
    status_values = tuple(statuses or ())
    if status_values:
        clauses.append("status IN (" + ",".join("?" for _ in status_values) + ")")
        params.extend(status_values)
    query = (
        "SELECT * FROM session_state_items WHERE " + " AND ".join(clauses)
        + " ORDER BY domain, state_type, state_key, id"
    )
    with _connect() as conn:
        rows = conn.execute(query, params).fetchall()
    return [_item_from_row(row) for row in rows]


def list_transition_audits(session_id: str) -> list[TransitionAudit]:
    """按稳定时间顺序返回一个 Session 的 reducer 决策。"""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM session_state_transitions"
            " WHERE session_id=? ORDER BY created_at, transition_id",
            (session_id,),
        ).fetchall()
    return [_audit_from_row(row) for row in rows]
