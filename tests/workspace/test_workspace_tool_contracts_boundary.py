"""工作区工具线上契约的边界覆盖。"""

from __future__ import annotations

from pathlib import Path

from jsonschema import Draft202012Validator

from personagraph.tools.workspace import workspace_tool_contracts
from personagraph.tools.workspace.workspace_tools import (
    FrozenWorkspaceToolBoundary,
    build_workspace_discovery_tool_registrations,
)


def _plain(value):
    if hasattr(value, "items"):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def test_workspace_discovery_builder_emits_valid_contract_owner_schemas(
    tmp_path: Path,
) -> None:
    """当前 discovery builder 使用唯一的线上 schema owner。"""

    boundary = FrozenWorkspaceToolBoundary(session_id="session-1", root=tmp_path)
    registrations = {
        registration.tool_id: registration
        for registration in build_workspace_discovery_tool_registrations(boundary)
    }
    expected = {
        "workspace_overview": (
            workspace_tool_contracts._overview_input_schema(
                max_entries_ceiling=workspace_tool_contracts.DISCOVERY_MAX_ENTRIES
            ),
            workspace_tool_contracts._discovery_overview_output_schema(),
            workspace_tool_contracts.WORKSPACE_DISCOVERY_TOOL_CONTRACT_VERSION,
        ),
        "list_workspace_directory": (
            workspace_tool_contracts._list_workspace_directory_input_schema(),
            workspace_tool_contracts._list_workspace_directory_output_schema(),
            workspace_tool_contracts.WORKSPACE_DISCOVERY_TOOL_CONTRACT_VERSION,
        ),
        "find_files": (
            workspace_tool_contracts._find_files_input_schema(),
            workspace_tool_contracts._find_files_output_schema(),
            workspace_tool_contracts.WORKSPACE_DISCOVERY_TOOL_CONTRACT_VERSION,
        ),
        "search_text_files": (
            workspace_tool_contracts._search_text_files_input_schema(),
            workspace_tool_contracts._search_text_files_output_schema(),
            workspace_tool_contracts.WORKSPACE_DISCOVERY_TOOL_CONTRACT_VERSION,
        ),
        "inspect_file": (
            workspace_tool_contracts._inspect_file_input_schema(),
            workspace_tool_contracts._inspect_file_output_schema(),
            workspace_tool_contracts.WORKSPACE_DISCOVERY_TOOL_CONTRACT_VERSION,
        ),
    }

    assert set(registrations) == set(expected)
    for tool_id, (input_schema, output_schema, contract_version) in expected.items():
        registration = registrations[tool_id]
        actual_input = _plain(registration.spec.input_schema)
        actual_output = _plain(registration.spec.output_schema)
        assert registration.spec.contract_version == contract_version
        assert actual_input == input_schema
        assert actual_output == output_schema
        Draft202012Validator.check_schema(actual_input)
        Draft202012Validator.check_schema(actual_output)
