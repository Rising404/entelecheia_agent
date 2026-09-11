"""在权威 Entry 之下组合已准入的 L2 executor。

``entry`` 保留持久化 Turn 生命周期：acceptance/replay、租约、Window 转换、事件
持久化、异常恢复与正式回复交付。本模块只接收 Entry 已准入的 executor，并构建其
运行所需的本地 Runtime 资源；绝不接收 Entry Store port，也绝不最终化 Turn。

此边界有意覆盖现行 Auxiliary executor。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

from personagraph.runtime.model_calls.contracts import RuntimeModelLedgerStore
from personagraph.runtime.tool_calls import RuntimeToolLedgerStore
from personagraph.runtime.turn_deadline import TurnDeadline
from personagraph.runtime.turn_events import EntryEventEmitter

if TYPE_CHECKING:
    from personagraph.l2.auxiliary_execution.production_chain import (
        AuxiliaryProductionChainResult,
    )


@dataclass(frozen=True, slots=True)
class AuxiliaryTaskExecutorPorts:
    """仅为内部 Auxiliary 生产链准备的 port。"""

    auxiliary: object
    delivery: object


class RuntimeCallLedgerStore(
    RuntimeModelLedgerStore,
    RuntimeToolLedgerStore,
    Protocol,
):
    """Entry 注入给 L2 的统一模型/工具调用账本 port。"""


def build_auxiliary_task_executor_ports(
    *,
    session_id: str,
    task_id: str,
    turn_id: str | None = None,
    emit: EntryEventEmitter,
    ledger_store: RuntimeCallLedgerStore,
    build_workspace_runtime: Callable[[str], object | None],
    build_workspace_work_run_bridge: Callable[[object], object],
    build_node_tool_runtime_factory: Callable[..., object],
    build_capability_catalogs: Callable[..., object],
    workspace_readonly_capability: str,
    build_auxiliary_ports: Callable[..., object],
    build_delivery_ports: Callable[..., object],
    features: Mapping[str, Any] | None = None,
    file_retrieval_data_version: str | None = None,
) -> AuxiliaryTaskExecutorPorts:
    """构建 Auxiliary executor port，而不读取或更改 Turn Window。"""

    feature_snapshot = {} if features is None else features
    interaction_mode = feature_snapshot.get(
        "user_interaction_mode", "interactive"
    )
    if interaction_mode not in {"interactive", "closed_world"}:
        raise ValueError(
            "user_interaction_mode must be 'interactive' or 'closed_world'"
        )
    allow_user_input = interaction_mode == "interactive"
    # L2 file tools are paused until its exact Task scope uses FileVersion contracts.
    workspace_runtime = build_workspace_runtime(session_id)
    workspace_tool_bridge = (
        build_workspace_work_run_bridge(
            workspace_runtime,
            ledger_store=ledger_store,
        )
        if workspace_runtime is not None
        else None
    )
    capability_tool_bridges = {
        **(
            {workspace_readonly_capability: workspace_tool_bridge}
            if workspace_runtime is not None
            else {}
        ),
    }
    task_document_scope = None
    return AuxiliaryTaskExecutorPorts(
        auxiliary=build_auxiliary_ports(
            emit=emit,
            allow_user_input=allow_user_input,
            model_ledger_store=ledger_store,
            tool_ledger_store=ledger_store,
            capability_catalogs=build_capability_catalogs(
                workspace_runtime=workspace_runtime,
            ),
            capability_tool_bridges=capability_tool_bridges,
            task_document_scope=task_document_scope,
            knowledge_cognition_history_enabled=False,
            node_retrieval_runtime_builder=None,
        ),
        delivery=build_delivery_ports(
            emit=emit,
            allow_user_input=allow_user_input,
            model_ledger_store=ledger_store,
            catalog_snapshot=(
                None
                if workspace_runtime is None
                else workspace_runtime.catalog_snapshot
            ),
            tool_bridge=workspace_tool_bridge,
            node_tool_runtime_factory=build_node_tool_runtime_factory(
                session_id=session_id,
                task_id=task_id,
                workspace_runtime=workspace_runtime,
                workspace_tool_bridge=workspace_tool_bridge,
                ledger_store=ledger_store,
                features=feature_snapshot,
                file_retrieval_data_version=file_retrieval_data_version,
                task_document_scope=task_document_scope,
            ),
        ),
    )


def run_auxiliary_task_executor(
    *,
    session_id: str,
    turn_id: str,
    task_id: str,
    desired_output: str,
    emit: EntryEventEmitter,
    deadline: TurnDeadline,
    features: Mapping[str, Any] | None = None,
    file_retrieval_data_version: str | None = None,
) -> AuxiliaryProductionChainResult:
    """运行已准入 Auxiliary 链并返回其私有边界结果。"""

    from personagraph.l2.auxiliary_execution.application import (
        AuxiliaryApplicationPorts,
    )
    from personagraph.l2.auxiliary_execution.planning.profiles import (
        WORKSPACE_READONLY_CAPABILITY,
        build_auxiliary_execution_capability_catalogs,
    )
    from personagraph.l2.auxiliary_execution.production_chain import (
        AuxiliaryProductionChainPorts,
        AuxiliaryProductionChainRequest,
        run_auxiliary_to_verified_delivery,
    )
    from personagraph.l2.auxiliary_execution.delivery.composition import (
        AuxiliaryTaskDeliveryPorts,
    )
    from personagraph.tools.workspace.session_read_source import (
        build_session_workspace_readonly_runtime,
    )
    from personagraph.l2.task_execution.tool_bridge.workspace_readonly_adapter import (
        build_workspace_readonly_work_run_bridge,
    )
    from personagraph.l2.task_execution.task_node.document_tool_runtime import (
        build_task_scoped_document_node_runtime_factory,
    )
    from personagraph.session import store as session_store
    from personagraph.session.runtime_call_ledger_store_facade import (
        build_runtime_call_ledger_store_facade,
    )

    ledger_store = build_runtime_call_ledger_store_facade(
        deps_factory=session_store.current_store_deps,
    )

    executor_ports = build_auxiliary_task_executor_ports(
        session_id=session_id,
        task_id=task_id,
        turn_id=turn_id,
        emit=emit,
        ledger_store=ledger_store,
        build_workspace_runtime=build_session_workspace_readonly_runtime,
        build_workspace_work_run_bridge=(
            build_workspace_readonly_work_run_bridge
        ),
        build_node_tool_runtime_factory=(
            build_task_scoped_document_node_runtime_factory
        ),
        build_capability_catalogs=build_auxiliary_execution_capability_catalogs,
        workspace_readonly_capability=WORKSPACE_READONLY_CAPABILITY,
        build_auxiliary_ports=AuxiliaryApplicationPorts,
        build_delivery_ports=AuxiliaryTaskDeliveryPorts,
        features=features,
        file_retrieval_data_version=file_retrieval_data_version,
    )
    return run_auxiliary_to_verified_delivery(
        AuxiliaryProductionChainRequest(
            session_id=session_id,
            turn_id=turn_id,
            task_id=task_id,
            desired_output=desired_output,
            deadline=deadline,
        ),
        ports=AuxiliaryProductionChainPorts(
            auxiliary=executor_ports.auxiliary,
            delivery=executor_ports.delivery,
        ),
    )


__all__ = [
    "AuxiliaryTaskExecutorPorts",
    "build_auxiliary_task_executor_ports",
    "run_auxiliary_task_executor",
]
