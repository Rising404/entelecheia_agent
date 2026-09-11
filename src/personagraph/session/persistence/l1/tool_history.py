"""从现有 L1 ToolCall 行读取当前执行的历史；不建表、不迁移、不重执行。

绑定时冻结数据库及 Session/Run 范围；每次访问用 mode=ro 连接并复核 owner。
结果 ID 是调用 ID 与持久 outcome_hash 的派生值，不新增结果表或索引副本。
"""

from __future__ import annotations

from contextlib import closing
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import sqlite3

from personagraph.persistent_turn_content.evidence import l1_tool_result_id
from personagraph.persistent_turn_content.tool_results import (
    DEFAULT_HISTORY_LIST_LIMIT,
    MAX_HISTORY_LIST_LIMIT,
    MAX_ARGUMENT_SUMMARY_CHARACTERS,
    ToolHistoryError,
    ToolResultSource,
    ToolResultHistoryItem,
    ToolResultHistoryPage,
    ToolResultRecord,
    validate_canonical_result,
    validate_history_window,
)

_SCOPE = """
    FROM l1_turn_tool_calls c
    JOIN l1_turn_steps s ON s.step_id=c.step_id AND s.l1_turn_run_id=c.l1_turn_run_id
    WHERE c.l1_turn_run_id=? AND c.session_id=? AND c.status!='pending'
      AND c.outcome_json IS NOT NULL AND c.outcome_hash IS NOT NULL
"""


@dataclass(frozen=True, slots=True)
class L1ToolHistoryReader:
    database_path: Path
    session_id: str
    l1_turn_run_id: str

    def __post_init__(self) -> None:
        if not self.session_id or not self.l1_turn_run_id:
            raise ValueError("tool history requires an exact Session and L1 run")
        object.__setattr__(self, "database_path", Path(self.database_path).resolve())

    @property
    def authority_sha256(self) -> str:
        bound = json.dumps(
            [str(self.database_path), self.session_id, self.l1_turn_run_id],
            ensure_ascii=False,
        )
        return hashlib.sha256(bound.encode()).hexdigest()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.database_path.as_uri() + "?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        return conn

    def _require_scope(self, conn: sqlite3.Connection) -> None:
        if (
            conn.execute(
                "SELECT 1 FROM l1_turn_runs WHERE l1_turn_run_id=? AND session_id=?",
                (self.l1_turn_run_id, self.session_id),
            ).fetchone()
            is None
        ):
            raise ToolHistoryError("tool_history_scope_unavailable")

    def list_results(
        self,
        *,
        offset: int = 0,
        limit: int = DEFAULT_HISTORY_LIST_LIMIT,
    ) -> ToolResultHistoryPage:
        validate_history_window(
            offset=offset, limit=limit, maximum=MAX_HISTORY_LIST_LIMIT
        )
        try:
            with closing(self._connect()) as conn:
                # 同一只读事务内的数量与分页共享快照，避免新结算结果使页码漂移。
                conn.execute("BEGIN")
                self._require_scope(conn)
                params = (self.l1_turn_run_id, self.session_id)
                total = conn.execute("SELECT COUNT(*) " + _SCOPE, params).fetchone()[0]
                rows = conn.execute(
                    "SELECT c.tool_call_id,c.tool_id,c.status,c.outcome_hash,c.arguments_json,c.arguments_hash,"
                    "s.ordinal attempt_ordinal,c.call_ordinal,length(c.outcome_json) total_characters "
                    + _SCOPE
                    + " ORDER BY s.ordinal,c.call_ordinal,c.tool_call_id LIMIT ? OFFSET ?",
                    (*params, limit, offset),
                ).fetchall()
                items = tuple(_history_item(row) for row in rows)
                end = offset + len(items)
                return ToolResultHistoryPage(
                    results=items,
                    offset=offset,
                    total_results=total,
                    next_offset=end if end < total else None,
                    partial=offset > 0 or end < total,
                )
        except sqlite3.Error:
            raise ToolHistoryError("tool_history_unavailable") from None

    def read_result(
        self,
        *,
        tool_result_id: str,
    ) -> ToolResultRecord:
        if not isinstance(tool_result_id, str) or not re.fullmatch(
            r"l1result_[0-9a-f]{64}", tool_result_id
        ):
            raise ToolHistoryError("invalid_history_request")
        try:
            with closing(self._connect()) as conn:
                conn.execute("BEGIN")
                self._require_scope(conn)
                # 原表无独立结果 ID 列；只按本执行的轻量身份匹配，再读取命中项的正文。
                identities = conn.execute(
                    "SELECT c.tool_call_id,c.outcome_hash " + _SCOPE,
                    (self.l1_turn_run_id, self.session_id),
                ).fetchall()
                call_id = next(
                    (
                        row["tool_call_id"]
                        for row in identities
                        if _result_id(row) == tool_result_id
                    ),
                    None,
                )
                if call_id is None:
                    raise ToolHistoryError("tool_result_unavailable")
                row = conn.execute(
                    "SELECT c.tool_call_id,c.tool_id,c.status,c.outcome_hash,c.outcome_json "
                    + _SCOPE
                    + " AND c.tool_call_id=?",
                    (self.l1_turn_run_id, self.session_id, call_id),
                ).fetchone()
                source = _source(row)
                if source.tool_result_id != tool_result_id:
                    raise ToolHistoryError("tool_result_integrity_invalid")
                outcome = validate_canonical_result(row["outcome_json"], source.result_sha256)
                if outcome.get("status") != source.status:
                    raise ToolHistoryError("tool_result_integrity_invalid")
                return ToolResultRecord(source=source, outcome=outcome)
        except sqlite3.Error:
            raise ToolHistoryError("tool_history_unavailable") from None


def _result_id(row: sqlite3.Row) -> str:
    try:
        return l1_tool_result_id(
            tool_call_id=row["tool_call_id"], result_sha256=row["outcome_hash"]
        )
    except (ValueError, TypeError):
        raise ToolHistoryError("tool_result_integrity_invalid") from None


def _source(row: sqlite3.Row | None) -> ToolResultSource:
    if row is None:
        raise ToolHistoryError("tool_result_unavailable")
    try:
        return ToolResultSource(
            tool_call_id=row["tool_call_id"],
            tool_id=row["tool_id"],
            status=row["status"],
            result_sha256=row["outcome_hash"],
            tool_result_id=_result_id(row),
        )
    except (ValueError, TypeError):
        raise ToolHistoryError("tool_result_integrity_invalid") from None


def _history_item(row: sqlite3.Row) -> ToolResultHistoryItem:
    arguments = row["arguments_json"]
    validate_canonical_result(arguments, row["arguments_hash"])
    return ToolResultHistoryItem(
        source=_source(row),
        attempt_ordinal=row["attempt_ordinal"],
        call_ordinal=row["call_ordinal"],
        arguments_summary=arguments[:MAX_ARGUMENT_SUMMARY_CHARACTERS],
        arguments_sha256=row["arguments_hash"],
        arguments_truncated=len(arguments) > MAX_ARGUMENT_SUMMARY_CHARACTERS,
        total_characters=row["total_characters"],
    )
