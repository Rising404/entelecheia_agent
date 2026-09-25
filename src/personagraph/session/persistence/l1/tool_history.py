"""从现有 L1 ToolCall 行读取当前执行的历史；不建表、不迁移、不重执行。

绑定时冻结数据库及 Session/Run 范围；每次访问用 mode=ro 连接并复核 owner。
短引用由持久步骤序号与批内调用序号组成；保留调用身份与 outcome_hash 的完整性校验。
"""

from __future__ import annotations

from contextlib import closing
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sqlite3

from personagraph.persistent_turn_content.evidence import (
    format_l1_call_ref,
    parse_l1_call_ref,
    project_findings_arguments,
)
from personagraph.output_protocol.l1_persistence import normalize_l1_finding_arguments
from personagraph.persistent_turn_content.findings import EXECUTION_FINDINGS_TOOL_IDS
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
_UNSUCCESSFUL_FINDINGS_ARGUMENTS = "Arguments omitted for unsuccessful findings call; read its error."


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
                # 仅 findings 参数含持久来源引用；先核验原参数，再投影为当前 Run 短引用。
                # 查询仍在同一只读快照中，不改写历史参数或其摘要。
                reference_calls = None
                if any(row["tool_id"] in EXECUTION_FINDINGS_TOOL_IDS and row["status"] == "succeeded" for row in rows):
                    reference_calls = [dict(row) for row in conn.execute(
                        "SELECT c.tool_call_id,c.status,c.outcome_hash,c.outcome_json,"
                        "s.ordinal attempt_ordinal,c.call_ordinal " + _SCOPE,
                        params,
                    ).fetchall()]
                items = tuple(_history_item(row, reference_calls=reference_calls) for row in rows)
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
        call_ref: str,
    ) -> ToolResultRecord:
        try:
            attempt_ordinal, call_ordinal = parse_l1_call_ref(call_ref)
        except (TypeError, ValueError):
            raise ToolHistoryError("invalid_history_request") from None
        try:
            with closing(self._connect()) as conn:
                conn.execute("BEGIN")
                self._require_scope(conn)
                # 已有 (Run, step, call ordinal) 唯一约束就是短引用的 authority。
                row = conn.execute(
                    "SELECT c.tool_call_id,c.tool_id,c.status,c.outcome_hash,c.outcome_json,"
                    "s.ordinal attempt_ordinal,c.call_ordinal "
                    + _SCOPE
                    + " AND s.ordinal=? AND c.call_ordinal=?",
                    (self.l1_turn_run_id, self.session_id, attempt_ordinal, call_ordinal),
                ).fetchone()
                source = _source(row)
                outcome = validate_canonical_result(row["outcome_json"], source.result_sha256)
                if outcome.get("status") != source.status:
                    raise ToolHistoryError("tool_result_integrity_invalid")
                return ToolResultRecord(source=source, outcome=outcome)
        except sqlite3.Error:
            raise ToolHistoryError("tool_history_unavailable") from None


def _source(row: sqlite3.Row | None) -> ToolResultSource:
    if row is None:
        raise ToolHistoryError("tool_result_unavailable")
    try:
        return ToolResultSource(
            tool_call_id=row["tool_call_id"],
            tool_id=row["tool_id"],
            status=row["status"],
            result_sha256=row["outcome_hash"],
            call_ref=format_l1_call_ref(
                attempt_ordinal=row["attempt_ordinal"],
                call_ordinal=row["call_ordinal"],
            ),
        )
    except (ValueError, TypeError):
        raise ToolHistoryError("tool_result_integrity_invalid") from None


def _history_item(row: sqlite3.Row, *, reference_calls: list[dict] | None) -> ToolResultHistoryItem:
    arguments = row["arguments_json"]
    parsed = validate_canonical_result(arguments, row["arguments_hash"])
    unavailable = row["tool_id"] in EXECUTION_FINDINGS_TOOL_IDS and row["status"] != "succeeded"
    if unavailable:
        arguments = _UNSUCCESSFUL_FINDINGS_ARGUMENTS
    elif row["tool_id"] in EXECUTION_FINDINGS_TOOL_IDS:
        try:
            calls = reference_calls or []
            normalized = normalize_l1_finding_arguments(parsed, tool_calls=calls)
            call_refs = {
                call["tool_call_id"]: format_l1_call_ref(
                    attempt_ordinal=call["attempt_ordinal"], call_ordinal=call["call_ordinal"],
                ) for call in calls
            }
            projected = project_findings_arguments(normalized, call_refs=call_refs)
            arguments = json.dumps(projected, ensure_ascii=False, allow_nan=False,
                                   sort_keys=True, separators=(",", ":"))
        except (ValueError, TypeError, KeyError):
            raise ToolHistoryError("tool_result_integrity_invalid") from None
    return ToolResultHistoryItem(
        source=_source(row),
        attempt_ordinal=row["attempt_ordinal"],
        call_ordinal=row["call_ordinal"],
        arguments_summary=arguments[:MAX_ARGUMENT_SUMMARY_CHARACTERS],
        arguments_sha256=row["arguments_hash"],
        arguments_truncated=len(arguments) > MAX_ARGUMENT_SUMMARY_CHARACTERS,
        arguments_unavailable=unavailable,
        arguments_unavailable_reason="unsuccessful_findings_call" if unavailable else None,
        total_characters=row["total_characters"],
    )
