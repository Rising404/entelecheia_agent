"""SessionContext 的证据适配器和非 Turn 事件存储。

Turn 的权威数据仍位于 ``session.store/session_turns``，读取时会适配为 EvidenceRecord。
本模块只存储非 Turn 摘录、哈希和所有者引用；不抽取状态，也不更新物化视图。
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import re
from datetime import datetime, timezone
from typing import Any, Collection, Mapping

from .. import store as session_store
from .models import EvidenceKind, EvidenceRecord


MAX_EVIDENCE_EXCERPT_CHARS = 1200
_TURN_KINDS = {EvidenceKind.USER_TURN, EvidenceKind.ASSISTANT_TURN}


def _connect() -> sqlite3.Connection:
    return session_store.connect_session_context_authority()


def _utc_iso(value: str | None = None) -> str:
    parsed = datetime.fromisoformat(value) if value else datetime.now(timezone.utc)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("created_at must include a UTC offset")
    return parsed.astimezone(timezone.utc).isoformat()


def _hash_text(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _excerpt(content: str, limit: int = MAX_EVIDENCE_EXCERPT_CHARS) -> str:
    if limit < 1:
        raise ValueError("excerpt limit must be positive")
    if len(content) <= limit:
        return content
    return content[: max(1, limit - 1)] + "…"


def _event_id(session_id: str, kind: EvidenceKind, source_ref: str, content_hash: str) -> str:
    raw = "\x1f".join((session_id, kind, source_ref, content_hash))
    return "event:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]


def turn_evidence_id(session_id: str, turn_idx: int) -> str:
    """返回一个权威对话 Turn 的稳定证据 ID。"""
    return f"turn:{session_id}:{int(turn_idx)}"


def _turn_record(session_id: str, turn: Mapping[str, Any]) -> EvidenceRecord:
    role = str(turn["role"])
    if role not in {"user", "assistant"}:
        raise ValueError(f"unsupported turn role: {role}")
    content = str(turn["content"])
    turn_idx = int(turn["turn_idx"])
    return EvidenceRecord(
        id=turn_evidence_id(session_id, turn_idx),
        session_id=session_id,
        kind=EvidenceKind.USER_TURN if role == "user" else EvidenceKind.ASSISTANT_TURN,
        source_ref=f"session_turns:{session_id}:{turn_idx}",
        content_excerpt=_excerpt(content),
        content_hash=_hash_text(content),
        created_at=_utc_iso(str(turn["created_at"])),
        metadata={"turn_idx": turn_idx, "role": role},
    )


def list_turn_evidence(session_id: str) -> list[EvidenceRecord]:
    """将对话行适配为证据，不把 Turn 文本复制到新表。"""
    return [_turn_record(session_id, turn) for turn in session_store.get_turns(session_id)]


def _event_record(row: Mapping[str, Any]) -> EvidenceRecord:
    return EvidenceRecord(
        id=str(row["id"]),
        session_id=str(row["session_id"]),
        kind=EvidenceKind(str(row["kind"])),
        source_ref=str(row["source_ref"]),
        content_excerpt=str(row["content_excerpt"]),
        content_hash=str(row["content_hash"]),
        created_at=str(row["created_at"]),
        metadata=json.loads(str(row["metadata_json"])),
    )


def append_event(
    session_id: str,
    kind: EvidenceKind,
    source_ref: str,
    content: str,
    *,
    metadata: Mapping[str, Any] | None = None,
    event_id: str | None = None,
    created_at: str | None = None,
) -> EvidenceRecord:
    """幂等追加一个非 Turn 证据事件，并返回其存储形态。"""
    if kind in _TURN_KINDS:
        raise ValueError(
            "turn evidence must be written through formal Runtime turn finalization"
        )
    if session_store.get_session(session_id) is None:
        raise ValueError(f"unknown session: {session_id}")
    if not source_ref.strip():
        raise ValueError("source_ref must not be empty")
    if not isinstance(content, str):
        raise TypeError("content must be a string")
    metadata_json = json.dumps(metadata or {}, ensure_ascii=False, sort_keys=True)
    content_hash = _hash_text(content)
    stable_id = event_id or _event_id(session_id, kind, source_ref, content_hash)
    timestamp = _utc_iso(created_at)

    with _connect() as conn:
        existing = conn.execute(
            "SELECT * FROM session_evidence_events WHERE id=?", (stable_id,)
        ).fetchone()
        if existing is not None:
            record = _event_record(existing)
            expected = (session_id, kind, source_ref, content_hash)
            actual = (record.session_id, record.kind, record.source_ref, record.content_hash)
            if actual != expected:
                raise ValueError(f"evidence event id collision: {stable_id}")
            return record
        conn.execute(
            "INSERT INTO session_evidence_events"
            " (id, session_id, kind, source_ref, content_excerpt, content_hash, metadata_json, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                stable_id,
                session_id,
                kind,
                source_ref,
                _excerpt(content),
                content_hash,
                metadata_json,
                timestamp,
            ),
        )
        row = conn.execute(
            "SELECT * FROM session_evidence_events WHERE id=?", (stable_id,)
        ).fetchone()
    return _event_record(row)


def list_event_evidence(session_id: str) -> list[EvidenceRecord]:
    """按稳定时间顺序返回一个 Session 的非 Turn 证据。"""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM session_evidence_events"
            " WHERE session_id=? ORDER BY created_at, id",
            (session_id,),
        ).fetchall()
    return [_event_record(row) for row in rows]


def get_evidence(evidence_id: str) -> EvidenceRecord | None:
    """解析稳定 Turn 引用或已存储的非 Turn 事件 ID。"""
    if evidence_id.startswith("turn:"):
        parts = evidence_id.split(":", 2)
        if len(parts) != 3 or not parts[2].isdigit():
            return None
        session_id, turn_idx = parts[1], int(parts[2])
        turn = session_store.get_turn(session_id, turn_idx)
        return _turn_record(session_id, turn) if turn else None
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM session_evidence_events WHERE id=?", (evidence_id,)
        ).fetchone()
    return _event_record(row) if row else None


def list_evidence(session_id: str) -> list[EvidenceRecord]:
    """按稳定的时间戳和 ID 顺序合并 Turn 与事件证据。"""
    records = [*list_turn_evidence(session_id), *list_event_evidence(session_id)]
    return sorted(records, key=lambda item: (item.created_at, item.id))


def evidence_inventory(session_id: str) -> tuple[set[str], dict[str, int]]:
    """返回 ID 和类型计数，不加载对话或事件内容。"""
    ids: set[str] = set()
    counts: dict[str, int] = {}
    with _connect() as conn:
        turn_rows = conn.execute(
            "SELECT turn_idx, role FROM session_turns WHERE session_id=? ORDER BY turn_idx",
            (session_id,),
        ).fetchall()
        event_rows = conn.execute(
            "SELECT id, kind FROM session_evidence_events WHERE session_id=? ORDER BY id",
            (session_id,),
        ).fetchall()
    for row in turn_rows:
        kind = EvidenceKind.USER_TURN if row["role"] == "user" else EvidenceKind.ASSISTANT_TURN
        ids.add(turn_evidence_id(session_id, int(row["turn_idx"])))
        counts[kind.value] = counts.get(kind.value, 0) + 1
    for row in event_rows:
        kind = EvidenceKind(str(row["kind"]))
        ids.add(str(row["id"]))
        counts[kind.value] = counts.get(kind.value, 0) + 1
    return ids, dict(sorted(counts.items()))


def search_evidence(
    session_id: str,
    query: str,
    *,
    limit: int = 8,
    kinds: Collection[EvidenceKind] | None = None,
    created_after: str | None = None,
    created_before: str | None = None,
    subject: str | None = None,
) -> list[EvidenceRecord]:
    """在一个 Session 内执行仅验证的词法、时间、类型和主题扩展。"""
    if limit <= 0:
        return []
    after = _utc_iso(created_after) if created_after else None
    before = _utc_iso(created_before) if created_before else None
    allowed_kinds = set(kinds or ())
    subject_term = (subject or "").strip().casefold()
    terms: set[str] = set()
    for token in re.findall(r"[\w\u4e00-\u9fff-]{2,}", query or ""):
        folded = token.casefold()
        terms.add(folded)
        if any("\u4e00" <= char <= "\u9fff" for char in folded) and len(folded) > 4:
            terms.update(folded[index:index + 2] for index in range(len(folded) - 1))
    records = [
        record for record in list_evidence(session_id)
        if (not allowed_kinds or record.kind in allowed_kinds)
        and (after is None or record.created_at >= after)
        and (before is None or record.created_at <= before)
        and (
            not subject_term
            or subject_term in record.source_ref.casefold()
            or subject_term in json.dumps(
                record.metadata, ensure_ascii=False, sort_keys=True, default=str
            ).casefold()
        )
    ]
    if not terms:
        return list(reversed(records[-limit:]))

    scored = []
    for record in records:
        text = record.content_excerpt.casefold()
        score = sum(1 for term in terms if term in text)
        if score:
            scored.append((score, record.created_at, record.id, record))
    scored.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
    return [item[3] for item in scored[:limit]]
