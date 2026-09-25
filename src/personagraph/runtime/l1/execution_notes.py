"""Host 幂等保存每步公开笔记；输入批次来自冻结请求，不冒充事实支持关系。"""

from __future__ import annotations
import json
from collections.abc import Mapping
from ...model_io.output_repair_contracts import RuntimeModelOutputRepairIssue
from ...output_protocol.actions import CallToolsAction, ToolCallProposal
from ...output_protocol.l1 import L1AttemptDecision
from ...output_protocol.l1_persistence import decode_l1_decision
from ...persistent_turn_content.findings import (
    RecordExecutionFinding,
    l1_execution_note_writer_id,
)
from ...tools.contracts import ExecutionStatus
from ...persistent_turn_content.findings import (
    EXECUTION_FINDINGS_TOOL_IDS,
    RECORD_EXECUTION_FINDINGS_TOOL_ID,
)
from ...tools.findings.dispatcher import execute_execution_findings_tool
from .ports import L1StorePort


class L1FindingsRevisionError(ValueError):
    """模型显式 CAS 与冻结台账不一致；定位只由调用序号与 Host revision 产生。"""

    def __init__(self, *, call_index: int, ledger_revision: object) -> None:
        current = f"（当前为 {ledger_revision}）" if type(ledger_revision) is int else ""
        self.repair_issue = RuntimeModelOutputRepairIssue(
            category="host_guard",
            code="host_guard.l1_findings_revision_mismatch",
            paths=(f"/action/calls/{call_index}/arguments/expected_ledger_revision",),
            safe_explanation=(
                "expected_ledger_revision 必须为整数，且等于当前冻结输入中的 "
                f"execution_findings.ledger_revision{current}；也可省略此字段，"
                "由 Host 绑定，不要自行加一。"
            ),
        )
        super().__init__("L1 findings mutation revision differs from model input")


def bind_explicit_findings_revision(
    decision: L1AttemptDecision,
    *,
    request: Mapping[str, object],
) -> L1AttemptDecision:
    """Host 将显式台账调用绑定到冻结快照及固定笔记写入后的版本。

    模型可省略 CAS 版本；显式提供时仍拒绝错误或过期值。
    变换结果随决定持久化，恢复时不会再加一次。
    """
    if request.get("execution_notes_required") is not True:
        return decision
    if not isinstance(decision.action, CallToolsAction):
        return decision
    projection = request.get("execution_findings")
    if not isinstance(projection, Mapping):
        raise ValueError("L1 notes have no frozen ledger revision")
    revision = projection["ledger_revision"]
    calls = []
    for index, call in enumerate(decision.action.calls):
        if call.tool_id not in EXECUTION_FINDINGS_TOOL_IDS:
            calls.append(call)
            continue
        expected = call.arguments.get("expected_ledger_revision", revision)
        if type(expected) is not int or expected != revision:
            raise L1FindingsRevisionError(call_index=index, ledger_revision=revision)
        calls.append(
            ToolCallProposal(
                tool_id=call.tool_id,
                arguments={**call.arguments, "expected_ledger_revision": expected + 1},
            )
        )
    return decision.model_copy(update={"action": CallToolsAction(calls=tuple(calls))})


def record_committed_execution_notes(
    *,
    store: L1StorePort,
    ledger_id: str,
    attempt: Mapping[str, object],
    execution: Mapping[str, object],
) -> None:
    """存储失败显式终止；不把 Host 配额故障交给模型重写答复。"""
    raw = attempt.get("decision_json")
    if not isinstance(raw, str):
        return
    request = json.loads(str(attempt["request_json"]))
    decision = decode_l1_decision(
        raw,
        request_schema_version=request.get("schema_version"),
        tool_calls=execution.get("tool_calls", []),
    )
    projection = request.get("execution_findings")
    if not isinstance(projection, dict):
        raise RuntimeError("L1 notes have no frozen ledger revision")
    note = RecordExecutionFinding(kind="decision", claim=decision.note)
    result = execute_execution_findings_tool(
        store=store,
        tool_id=RECORD_EXECUTION_FINDINGS_TOOL_ID,
        normalized_arguments={
            "expected_ledger_revision": projection["ledger_revision"],
            "items": [note.model_dump(mode="json", exclude={"operation"})],
        },
        ledger_id=ledger_id,
        writer_unit_id=str(attempt["attempt_id"]),
        writer_tool_call_id=l1_execution_note_writer_id(str(attempt["attempt_id"])),
    )
    if result.status is not ExecutionStatus.SUCCEEDED:
        code = result.error.code if result.error is not None else "unknown"
        raise RuntimeError(f"Host could not record L1 execution note: {code}")
