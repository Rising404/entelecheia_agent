"""共享执行发现账本的 WorkRun 绑定。

本模块不负责模型或工具执行。它为活跃 WorkRun 创建/重放唯一的 Host 所有账本，校验其所属
WorkRun 的 Acceptance 身份，并提供稳定的 Tool Bridge 持久化标识。
"""

from __future__ import annotations

import hashlib
import json
from typing import Protocol

from ....session import store as session_store
from ....tools.contracts import ToolSpec
from ...work_run import ToolResult
from ..tool_bridge.attempt_contracts import AttemptToolBridgeRequest
from ....persistent_turn_content.findings import (
    ExecutionFindingsActiveProjection,
    ExecutionFindingsLedgerStatus,
    ExecutionFindingsOwnerKind,
    ExecutionFindingsSnapshot,
    derive_execution_findings_ledger_id,
    sha256_json,
)
from ....tools.findings.contracts import EXECUTION_FINDINGS_TOOL_IDS
from ..tool_bridge.persistence_contracts import (
    ToolBridgeCallPersistence,
    ToolBridgePersistencePlan,
)


EXECUTION_FINDINGS_SYSTEM_PROMPT_CLAUSE = """execution_findings 是 Host 绑定当前 WorkRun 的执行期发现台账，不是长期记忆，也不是最终交付正文。
- 每次模型调用都收到同一 WorkRun 的有界 active projection；ledger_id、ledger_revision、配额余量和 projection hash 都是 Host 权威字段，不得自行改写。
- 是否记录或修订 finding、decision、gap 由模型根据当前执行需要决定；若决定更新，只能使用 allowed_tools 中实际暴露的台账写工具，且一次 call_tools 批次最多包含一个。
- finding、decision、gap 的 source_refs 都可为空；空引用表示尚未绑定原始证据的模型候选笔记，不能单独充当已验证证据。若填写引用，使用 prior_tool_results 中成功结果的原生 tool_result_id；可选 chunk_id 必须确实属于该结果，由 Host 核对，不填写调用 ID、alias 或 hash，也不得伪造来源。
- active_entries 是按 Host sequence 排列的有界 FIFO 工作集，不是 Host 已验证事实或完整 coverage 证明。新记录进入队尾，supersede 刷新到队尾，retract 移出工作集；omitted_active_count 表示旧 active 条目已退出 prompt 投影且不会自动回填，durable 审计历史仍保留。
- record_execution_findings/revise_execution_finding 返回的 mutation ToolResult 只确认内部台账状态变化，不能写入 Acceptance 的 supporting_tool_result_ids，也不能作为节点语义验证证据。
- OutputWindow 只保存当前节点的完整交付草稿。阶段发现、选择理由和未解缺口可写入 execution_findings；台账不是机械 coverage 账本，也不能凭空证明“已读完”。
- 台账工具返回新的 ledger_revision。后续写入必须使用最新 revision；CAS 冲突或引用拒绝后，根据工具错误与下一次投影修正，不得猜测成功。"""


class WorkRunExecutionFindingsStore(Protocol):
    def create_execution_findings_ledger(self, **kwargs: object) -> object: ...

    def get_execution_findings_ledger_for_owner(
        self,
        **kwargs: object,
    ) -> ExecutionFindingsSnapshot | None: ...


def require_work_run_execution_findings(
    *,
    session_id: str,
    work_run_id: str,
    acceptance_ids: tuple[str, ...],
    store: WorkRunExecutionFindingsStore = session_store,
) -> ExecutionFindingsActiveProjection:
    """创建/重放并返回当前 WorkRun 的有界投影。"""

    if not acceptance_ids or len(acceptance_ids) != len(set(acceptance_ids)):
        raise ValueError("WorkRun findings require unique Acceptance IDs")
    created = store.create_execution_findings_ledger(
        session_id=session_id,
        owner_kind=ExecutionFindingsOwnerKind.WORK_RUN,
        execution_owner_id=work_run_id,
    )
    snapshot = store.get_execution_findings_ledger_for_owner(
        owner_kind=ExecutionFindingsOwnerKind.WORK_RUN,
        execution_owner_id=work_run_id,
    )
    if snapshot is None:
        raise RuntimeError("WorkRun findings ledger disappeared after creation")
    expected_ledger_id = derive_execution_findings_ledger_id(
        owner_kind=ExecutionFindingsOwnerKind.WORK_RUN,
        execution_owner_id=work_run_id,
    )
    if (
        snapshot.ledger.ledger_id != expected_ledger_id
        or snapshot.ledger.session_id != session_id
        or snapshot.ledger.execution_owner_id != work_run_id
        or snapshot.ledger.owner_kind is not ExecutionFindingsOwnerKind.WORK_RUN
        or snapshot.ledger.status is not ExecutionFindingsLedgerStatus.OPEN
        or snapshot.active_projection.ledger_id != expected_ledger_id
        or snapshot.active_projection.ledger_revision != snapshot.ledger.revision
        or getattr(created, "ledger", None) is None
        or created.ledger.ledger_id != expected_ledger_id
    ):
        raise RuntimeError("WorkRun findings authority is inconsistent")
    return snapshot.active_projection


def project_work_run_execution_findings_for_tools(
    *,
    session_id: str,
    work_run_id: str,
    acceptance_ids: tuple[str, ...],
    allowed_tools: tuple[ToolSpec, ...],
    store: WorkRunExecutionFindingsStore = session_store,
) -> ExecutionFindingsActiveProjection | None:
    """恰好在完整工具对暴露时注入投影。"""

    tool_ids = {item.tool_id for item in allowed_tools}
    exposed = EXECUTION_FINDINGS_TOOL_IDS.intersection(tool_ids)
    if not exposed:
        return None
    if exposed != EXECUTION_FINDINGS_TOOL_IDS:
        raise ValueError("execution-findings tools are only partially exposed")
    return require_work_run_execution_findings(
        session_id=session_id,
        work_run_id=work_run_id,
        acceptance_ids=acceptance_ids,
        store=store,
    )


def work_run_tool_result_sha256(result: ToolResult) -> str:
    """对 WorkRun Store 持久化的精确规范 ToolResult JSON 进行哈希。"""

    return sha256_json(result)


def execution_findings_tool_persistence_plan(
    request: AttemptToolBridgeRequest,
) -> ToolBridgePersistencePlan:
    """为仅含发现或增强型 WorkRun 桥接器返回稳定收据。"""

    common = {
        "schema_version": "execution-findings-work-run-tool-persistence-v1",
        "session_id": request.session_id,
        "work_run_id": request.work_run_id,
        "attempt_id": request.attempt_id,
        "decision_apply_id": request.apply_id,
    }
    return ToolBridgePersistencePlan(
        decision_apply_id=request.apply_id,
        close_apply_id=_stable_id("execution-findings-tool-close", common),
        calls=tuple(
            ToolBridgeCallPersistence(
                tool_call_id=call.tool_call_id,
                tool_result_id=_stable_id(
                    "execution-findings-tool-result",
                    {**common, "tool_call_id": call.tool_call_id},
                ),
                result_apply_id=_stable_id(
                    "execution-findings-tool-result-apply",
                    {**common, "tool_call_id": call.tool_call_id},
                ),
            )
            for call in request.decision.action.calls
        ),
    )


def _stable_id(prefix: str, value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"{prefix}-{hashlib.sha256(encoded).hexdigest()}"


__all__ = [
    "EXECUTION_FINDINGS_SYSTEM_PROMPT_CLAUSE",
    'WorkRunExecutionFindingsStore',
    "execution_findings_tool_persistence_plan",
    "project_work_run_execution_findings_for_tools",
    "require_work_run_execution_findings",
    "work_run_tool_result_sha256",
]
