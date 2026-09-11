"""入口 L2 执行器组合边界的回归测试。"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest

from personagraph.l2.auxiliary_execution.planning import (
    profiles as auxiliary_planning_profiles,
)
from personagraph.l2.auxiliary_execution import (
    production_chain as auxiliary_production_chain,
)
from personagraph.l2.task_execution.task_node import (
    document_tool_runtime as task_node_document_tool_runtime,
)
from personagraph.l2.task_execution.tool_bridge import (
    workspace_readonly_adapter,
)
from personagraph.tools.workspace import session_read_source
from personagraph.l2.entry_adapter import executor as l2_entry_executor
from personagraph.l2.entry_adapter.executor import (
    build_auxiliary_task_executor_ports,
    run_auxiliary_task_executor,
)
from personagraph.runtime.turn_deadline import TurnDeadline
from personagraph.session.runtime_call_ledger_store_facade import (
    RuntimeCallLedgerStoreFacade,
)


def test_entry_executor_api_has_no_history_generation_parameter() -> None:
    assert "history_retrieval_data_version" not in inspect.signature(
        build_auxiliary_task_executor_ports
    ).parameters
    assert "history_retrieval_data_version" not in inspect.signature(
        run_auxiliary_task_executor
    ).parameters


def test_auxiliary_executor_wrapper_prepares_capabilities_but_not_entry_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    result_marker = object()
    workspace_runtime = SimpleNamespace(
        session_id="session-1",
        catalog_snapshot="workspace-catalog",
    )
    monkeypatch.setattr(
        session_read_source,
        "build_session_workspace_readonly_runtime",
        lambda _session_id, **_kwargs: workspace_runtime,
    )

    def build_workspace_bridge(
        source: object,
        *,
        ledger_store: object,
    ) -> str:
        assert source is workspace_runtime
        captured["ledger_store"] = ledger_store
        return "workspace-bridge"

    monkeypatch.setattr(
        workspace_readonly_adapter,
        "build_workspace_readonly_work_run_bridge",
        build_workspace_bridge,
    )
    monkeypatch.setattr(
        auxiliary_planning_profiles,
        "build_auxiliary_execution_capability_catalogs",
        lambda **kwargs: ("capabilities", kwargs),
    )

    def run_chain(request: object, *, ports: object) -> object:
        captured["request"] = request
        captured["ports"] = ports
        return result_marker

    monkeypatch.setattr(
        auxiliary_production_chain,
        "run_auxiliary_to_verified_delivery",
        run_chain,
    )
    deadline = TurnDeadline.starting_now(60.0)
    result = run_auxiliary_task_executor(
        session_id="session-1",
        turn_id="turn-1",
        task_id="task-1",
        desired_output="produce verified output",
        emit=lambda _event: None,
        deadline=deadline,
    )

    request = captured["request"]
    ports = captured["ports"]
    assert result is result_marker
    assert request.session_id == "session-1"
    assert request.turn_id == "turn-1"
    assert request.task_id == "task-1"
    assert request.desired_output == "produce verified output"
    assert request.deadline is deadline
    assert ports.auxiliary.capability_catalogs == (
        "capabilities",
        {
            "workspace_runtime": workspace_runtime,
        },
    )
    assert ports.auxiliary.capability_tool_bridges == {
        auxiliary_planning_profiles.WORKSPACE_READONLY_CAPABILITY: (
            "workspace-bridge"
        ),
    }
    assert isinstance(captured["ledger_store"], RuntimeCallLedgerStoreFacade)
    assert ports.auxiliary.model_ledger_store is captured["ledger_store"]
    assert ports.auxiliary.tool_ledger_store is captured["ledger_store"]
    assert ports.delivery.model_ledger_store is captured["ledger_store"]
    assert ports.delivery.catalog_snapshot == "workspace-catalog"
    assert ports.delivery.tool_bridge == "workspace-bridge"
    assert ports.delivery.node_tool_runtime_factory is not None
    assert ports.auxiliary.allow_user_input is True
    assert not hasattr(ports.auxiliary, "file_retrieval_enabled")
    assert ports.delivery.allow_user_input is True
    assert (
        ports.delivery.node_tool_runtime_factory.__module__
        == task_node_document_tool_runtime.__name__
    )


def test_auxiliary_executor_maps_closed_world_mode_to_both_port_surfaces() -> None:
    ledger_store = object()
    ports = build_auxiliary_task_executor_ports(
        session_id="session-1",
        task_id="task-1",
        emit=lambda _event: None,
        ledger_store=ledger_store,
        build_workspace_runtime=lambda _session_id, **_kwargs: None,
        build_workspace_work_run_bridge=lambda _source: "unexpected-bridge",
        build_node_tool_runtime_factory=lambda **_kwargs: "node-runtime",
        build_capability_catalogs=lambda **_kwargs: "capabilities",
        workspace_readonly_capability="workspace-readonly",
        build_auxiliary_ports=lambda **kwargs: SimpleNamespace(**kwargs),
        build_delivery_ports=lambda **kwargs: SimpleNamespace(**kwargs),
        features={"user_interaction_mode": "closed_world"},
    )

    assert ports.auxiliary.allow_user_input is False
    assert ports.auxiliary.model_ledger_store is ledger_store
    assert ports.auxiliary.tool_ledger_store is ledger_store
    assert ports.delivery.allow_user_input is False
    assert ports.delivery.model_ledger_store is ledger_store


def test_auxiliary_executor_rejects_unknown_interaction_mode_before_io() -> None:
    def unexpected(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("invalid interaction mode must fail before composition I/O")

    with pytest.raises(
        ValueError,
        match="user_interaction_mode must be 'interactive' or 'closed_world'",
    ):
        build_auxiliary_task_executor_ports(
            session_id="session-1",
            task_id="task-1",
            emit=lambda _event: None,
            ledger_store=object(),
            build_workspace_runtime=unexpected,
            build_workspace_work_run_bridge=unexpected,
            build_node_tool_runtime_factory=unexpected,
            build_capability_catalogs=unexpected,
            workspace_readonly_capability="workspace-readonly",
            build_auxiliary_ports=unexpected,
            build_delivery_ports=unexpected,
            features={"user_interaction_mode": "unattended"},
        )


def test_executor_composition_has_no_entry_lifecycle_or_finalization_calls() -> None:
    source_path = Path(l2_entry_executor.__file__)
    module = ast.parse(source_path.read_text(encoding="utf-8"))
    called_attributes = {
        node.func.attr
        for node in ast.walk(module)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
    }
    relative_imports = {
        node.module
        for node in ast.walk(module)
        if isinstance(node, ast.ImportFrom) and node.level == 1
    }

    assert "entry" not in relative_imports
    assert "personagraph.runtime.entry" not in source_path.read_text(encoding="utf-8")
    assert not {
        "accept_turn_execution",
        "advance_turn_execution_window",
        "finalize_turn_execution",
        "finalize_verified_turn_execution",
        "finalize_authoritative_referenced_turn_execution",
        "mark_turn_execution_interrupted",
    } & called_attributes
