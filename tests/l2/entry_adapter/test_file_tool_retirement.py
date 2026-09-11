from types import SimpleNamespace

from personagraph.l2.entry_adapter.executor import build_auxiliary_task_executor_ports


def test_l2_entry_never_enables_retired_candidate_file_tools():
    ports = build_auxiliary_task_executor_ports(
        session_id="session", task_id="task", turn_id="turn", emit=lambda *_: None,
        ledger_store=SimpleNamespace(), build_workspace_runtime=lambda _: None,
        build_workspace_work_run_bridge=lambda *_: None,
        build_node_tool_runtime_factory=lambda **_: None,
        build_capability_catalogs=lambda **_: {}, workspace_readonly_capability="workspace_readonly",
        build_auxiliary_ports=lambda **values: SimpleNamespace(**values),
        build_delivery_ports=lambda **values: SimpleNamespace(**values),
        features={"file_retrieval_read_enabled": True, "l2_aux_retrieval_tools_enabled": True},
    )
    assert not hasattr(ports.auxiliary, "file_retrieval_enabled")
    assert not hasattr(ports.auxiliary, "knowledge_cognition_file_enabled")
    assert not hasattr(ports.auxiliary, "file_candidate_scope_snapshot_id")
    assert ports.auxiliary.node_retrieval_runtime_builder is None
