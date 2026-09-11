"""L1 lane 的启动快照与 controller 调用边界。

本模块拥有 L1 工具目录、执行配置、语料身份和 TurnRun 的冻结。它返回经过 L1
验证的 controller 结果，但不提交正式 assistant 消息，也不改变 Entry 的最终状态。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from .context import L1TurnContext
from .controller import L1ControllerResult, run_l1_turn
from .corpus_contracts import (
    FrozenL1CorpusManifest,
    L1_CORPUS_MANIFEST_CONTRACT_VERSION,
    derive_l1_turn_run_id,
)
from .corpus_manifest import freeze_l1_corpus_manifest
from .execution_config import freeze_l1_execution_config
from .ports import L1EntryBootstrapStorePort, L1StorePort
from .tool_runtime import (
    L1ToolRuntime,
    build_l1_tool_runtime,
)
from ..turn.contracts import AcceptedEntryTurn
from ..turn_deadline import TurnDeadline
from ..turn_events import (
    EntryEventEmitter,
)


@dataclass(frozen=True, slots=True)
class PreparedL1EntryLane:
    """一次新 L1 TurnRun 已持久化且不可变的启动材料。"""

    l1_turn_run_id: str
    turn_window_revision: int
    tool_runtime: L1ToolRuntime
    corpus_manifest: FrozenL1CorpusManifest


def prepare_l1_entry_lane(
    *,
    accepted: AcceptedEntryTurn,
    context: L1TurnContext,
    features: dict[str, Any],
    initial_turn_window_revision: int,
    routing_policy_snapshot_hash: str,
    deadline: TurnDeadline,
    store: L1EntryBootstrapStorePort,
    lease_owner: str,
) -> PreparedL1EntryLane:
    """L1 bootstrap 的资源/持久化 owner：冻结执行材料，再创建唯一 TurnRun。

    Tool Catalog、L1 binding、corpus manifest 与绝对 deadline 一起保存；创建时
    核对原 routing snapshot、Window revision 和 lease。返回同一套冻结对象与新
    revision，供首次 controller 执行复用；此处不产生模型答案或提交 assistant 消息。
    """

    tool_runtime = build_l1_tool_runtime(
        accepted.session_id,
        turn_id=accepted.turn_id,
        execution_findings_enabled=True,
        execution_features=features,
        file_retrieval_data_version=(
            accepted.execution_snapshot.file_retrieval_data_version
        ),
        session_retrieval_data_version=(
            accepted.execution_snapshot.session_retrieval_data_version
        ),
        session_retrieval_assistant_turn_cutoff=(
            accepted.execution_snapshot.session_retrieval_assistant_turn_cutoff
        ),
    )
    execution_config = freeze_l1_execution_config(features)
    l1_turn_run_id = derive_l1_turn_run_id(
        session_id=accepted.session_id,
        turn_id=accepted.turn_id,
    )
    corpus_manifest = freeze_l1_corpus_manifest(
        session_id=accepted.session_id,
        turn_id=accepted.turn_id,
        l1_turn_run_id=l1_turn_run_id,
        tool_runtime=tool_runtime,
        attachments=context.attachments,
    )
    deadline_at = (
        datetime.now(timezone.utc) + timedelta(seconds=deadline.remaining_s())
    ).isoformat()
    created = store.create_l1_turn_run(
        session_id=accepted.session_id,
        turn_id=accepted.turn_id,
        routing_policy_snapshot_hash=routing_policy_snapshot_hash,
        expected_window_revision=initial_turn_window_revision,
        expected_lease_owner=lease_owner,
        l1_turn_run_id=l1_turn_run_id,
        deadline_at=deadline_at,
        max_attempts=int(features.get("l1_max_attempts", 24)),
        max_tool_calls_per_attempt=int(
            features.get("l1_max_tool_calls_per_attempt", 8)
        ),
        catalog_snapshot_json=tool_runtime.catalog_snapshot_json,
        catalog_snapshot_hash=tool_runtime.catalog_snapshot_sha256,
        execution_config_json=execution_config.snapshot_json,
        execution_config_hash=execution_config.snapshot_sha256,
        corpus_manifest_contract_version=L1_CORPUS_MANIFEST_CONTRACT_VERSION,
        corpus_manifest_json=corpus_manifest.manifest_json,
        corpus_manifest_hash=corpus_manifest.manifest_sha256,
    )
    window = _required_mapping(created, "window")
    run = _required_mapping(created, "run")
    return PreparedL1EntryLane(
        l1_turn_run_id=str(run["l1_turn_run_id"]),
        turn_window_revision=int(window["state_version"]),
        tool_runtime=tool_runtime,
        corpus_manifest=corpus_manifest,
    )


def run_l1_entry_lane(
    *,
    accepted: AcceptedEntryTurn,
    context: L1TurnContext,
    l1_turn_run_id: str,
    initial_turn_window_revision: int,
    deadline: TurnDeadline,
    features: dict[str, Any],
    emit: EntryEventEmitter,
    store: L1StorePort,
    lease_owner: str,
    frozen_tool_runtime: L1ToolRuntime | None = None,
    frozen_corpus_manifest: FrozenL1CorpusManifest | None = None,
) -> L1ControllerResult:
    """将新建或恢复的 L1 材料交给 controller，并返回已验证候选与 Window revision。

    本层是 Entry→L1 的窄调用边界：接受用户输入、异常公开投影与正式答复提交仍在
    Entry；已有 run 的配置/预算校验仍在 controller，不能靠调用参数刷新恢复预算。
    """

    return run_l1_turn(
        accepted=accepted,
        context=context,
        l1_turn_run_id=l1_turn_run_id,
        initial_turn_window_revision=initial_turn_window_revision,
        deadline=deadline,
        emit=emit,
        store=store,
        lease_owner=lease_owner,
        max_attempts=int(features.get("l1_max_attempts", 24)),
        max_tool_calls_per_attempt=int(
            features.get("l1_max_tool_calls_per_attempt", 8)
        ),
        execution_features=features,
        frozen_tool_runtime=frozen_tool_runtime,
        frozen_corpus_manifest=frozen_corpus_manifest,
    )


def _required_mapping(
    value: dict[str, object],
    key: str,
) -> dict[str, object]:
    item = value.get(key)
    if not isinstance(item, dict):
        raise RuntimeError(f"L1 TurnRun creation omitted {key}")
    return item


__all__ = [
    "PreparedL1EntryLane",
    "prepare_l1_entry_lane",
    "run_l1_entry_lane",
]
