"""当前面向模型的工作区发现契约。"""

from __future__ import annotations

from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from personagraph.tools.execution import ToolBusinessFailure
from personagraph.tools.execution import ResolvedInvocation, ToolExecutor
from personagraph.tools.workspace.workspace_tools import (
    FrozenWorkspaceToolBoundary,
    build_workspace_discovery_tool_registrations,
)


def _plain(value):
    if hasattr(value, "items"):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    root = tmp_path / "workspace"
    (root / "contracts").mkdir(parents=True)
    (root / "contracts" / "acme.md").write_text(
        "renewal clause\nsecond line\n", encoding="utf-8"
    )
    (root / "contracts" / "beta.md").write_text(
        "renewal clause\n", encoding="utf-8"
    )
    (root / "notes").mkdir()
    (root / "notes" / "renewal-name-only.txt").write_text(
        "nothing relevant\n", encoding="utf-8"
    )
    (root / "image.png").write_bytes(b"\x89PNG\r\n\x1a\n\x00binary")
    (root / ".env").write_text("API_KEY=do-not-return\n", encoding="utf-8")
    return root


@pytest.fixture
def registrations(tree: Path):
    boundary = FrozenWorkspaceToolBoundary(session_id="session-1", root=tree)
    return build_workspace_discovery_tool_registrations(boundary)


def _by_id(registrations):
    return {registration.tool_id: registration for registration in registrations}


def test_builder_exposes_only_the_five_separated_ids(tree: Path):
    boundary = FrozenWorkspaceToolBoundary(session_id="session-1", root=tree)

    current = build_workspace_discovery_tool_registrations(boundary)

    assert tuple(registration.tool_id for registration in current) == (
        "workspace_overview",
        "list_workspace_directory",
        "find_files",
        "search_text_files",
        "inspect_file",
    )
    assert all(
        registration.execution_profile.max_output_bytes < 128_000
        for registration in current
    )
    schemas = {
        registration.tool_id: _plain(registration.spec.input_schema)
        for registration in current
    }
    assert schemas["workspace_overview"]["properties"]["max_entries"]["maximum"] <= 60
    assert schemas["list_workspace_directory"]["properties"]["limit"]["maximum"] <= 100
    assert schemas["find_files"]["properties"]["limit"]["maximum"] <= 50
    assert schemas["search_text_files"]["properties"]["limit"]["maximum"] <= 50


def test_all_new_contract_schemas_validate_and_describe_real_results(registrations):
    tools = _by_id(registrations)
    payloads = {
        "workspace_overview": {"depth": 1},
        "list_workspace_directory": {"limit": 2},
        "find_files": {"name": "*.md"},
        "search_text_files": {"content": "renewal clause"},
        "inspect_file": {"path": "contracts/acme.md"},
    }

    for tool_id, registration in tools.items():
        input_schema = _plain(registration.spec.input_schema)
        output_schema = _plain(registration.spec.output_schema)
        Draft202012Validator.check_schema(input_schema)
        Draft202012Validator.check_schema(output_schema)
        Draft202012Validator(input_schema).validate(payloads[tool_id])
        result = registration.handler(payloads[tool_id])
        Draft202012Validator(output_schema).validate(result)
        assert {"scope", "total", "truncated", "skipped"} <= result.keys()
        assert result["scope"]["session_id"] == "session-1"
        if tool_id != "inspect_file":
            assert result["scan_truncated"] is False
            assert result["total_relation"] == "exact"


def test_boundary_rejects_a_root_that_is_not_a_directory(tmp_path: Path):
    target = tmp_path / "a-file.txt"
    target.write_text("x", encoding="utf-8")

    with pytest.raises(ValueError):
        FrozenWorkspaceToolBoundary(session_id="session-1", root=target)


def test_overview_maps_a_missing_search_backend_to_a_business_failure(
    registrations, monkeypatch: pytest.MonkeyPatch
):
    from personagraph.workspace.discovery import overview as overview_module
    from personagraph.workspace.discovery import RipgrepUnavailable

    def _absent(*_args, **_kwargs):
        raise RipgrepUnavailable("not installed")

    monkeypatch.setattr(overview_module, "list_files", _absent)

    with pytest.raises(ToolBusinessFailure) as raised:
        _by_id(registrations)["workspace_overview"].handler({})
    assert raised.value.error.code == "search_backend_unavailable"


def test_overview_keeps_a_search_timeout_retryable(
    registrations, monkeypatch: pytest.MonkeyPatch
):
    from personagraph.workspace.discovery import overview as overview_module
    from personagraph.workspace.discovery import RipgrepTimeout

    def _slow(*_args, **_kwargs):
        raise RipgrepTimeout("too slow")

    monkeypatch.setattr(overview_module, "list_files", _slow)

    with pytest.raises(RuntimeError) as raised:
        _by_id(registrations)["workspace_overview"].handler({})
    assert not isinstance(raised.value, ToolBusinessFailure)


def test_name_and_content_search_are_distinct_capabilities(registrations):
    tools = _by_id(registrations)

    names = tools["find_files"].handler({"name": "*renewal*"})
    contents = tools["search_text_files"].handler({"content": "renewal"})

    assert [item["path"] for item in names["items"]] == [
        "notes/renewal-name-only.txt"
    ]
    assert {item["path"] for item in contents["items"]} == {
        "contracts/acme.md",
        "contracts/beta.md",
    }
    assert all("line" not in item for item in names["items"])
    assert all(item["line"] >= 1 for item in contents["items"])

    with pytest.raises(ToolBusinessFailure):
        tools["find_files"].handler({"content": "renewal"})
    with pytest.raises(ToolBusinessFailure):
        tools["search_text_files"].handler({"name": "*.md"})


def test_scoped_search_paths_stay_workspace_relative_and_feed_inspect(registrations):
    tools = _by_id(registrations)

    result = tools["find_files"].handler(
        {"name": "acme.md", "subpath": "contracts"}
    )

    assert result["scope"] == {"session_id": "session-1", "path": "contracts"}
    assert [item["path"] for item in result["items"]] == ["contracts/acme.md"]
    inspected = tools["inspect_file"].handler({"path": result["items"][0]["path"]})
    assert inspected["items"][0]["text"].startswith("renewal clause")


def test_directory_listing_is_direct_stable_and_cursor_paged(registrations):
    tool = _by_id(registrations)["list_workspace_directory"]

    first = tool.handler({"limit": 2})
    second = tool.handler({"cursor": first["next_cursor"], "limit": 2})

    assert [
        (item["path"], item["kind"])
        for item in (*first["items"], *second["items"])
    ] == [
        ("contracts", "directory"),
        ("notes", "directory"),
        ("image.png", "file"),
    ]
    assert first["scope"] == {"session_id": "session-1", "path": "."}
    assert first["total"] == 3
    assert first["returned"] == 2
    assert first["truncated"] is True
    assert first["scan_truncated"] is False
    assert first["total_relation"] == "exact"
    assert isinstance(first["next_cursor"], str)
    assert first["next_cursor"].startswith("workspacedircursor_")
    assert second["next_cursor"] is None
    assert second["truncated"] is False
    assert all("/" not in item["name"] for item in first["items"])

    nested = tool.handler({"subpath": "contracts", "limit": 100})
    assert [item["path"] for item in nested["items"]] == [
        "contracts/acme.md",
        "contracts/beta.md",
    ]
    assert all(item["kind"] == "file" for item in nested["items"])


def test_directory_cursor_is_bound_to_scope_and_listing_snapshot(
    tree: Path,
    registrations,
) -> None:
    tool = _by_id(registrations)["list_workspace_directory"]
    first = tool.handler({"limit": 1})
    cursor = first["next_cursor"]
    assert cursor is not None

    with pytest.raises(ToolBusinessFailure) as wrong_scope:
        tool.handler({"subpath": "contracts", "cursor": cursor, "limit": 1})
    assert wrong_scope.value.error.code == "invalid_request"

    (tree / "added.txt").write_text("new", encoding="utf-8")
    with pytest.raises(ToolBusinessFailure) as stale:
        tool.handler({"cursor": cursor, "limit": 1})
    assert stale.value.error.code == "invalid_request"


def test_discovery_rows_expose_paths_without_candidate_ids(
    tree: Path,
) -> None:
    tools = _by_id(
        build_workspace_discovery_tool_registrations(
            FrozenWorkspaceToolBoundary(session_id="session-1", root=tree),
        )
    )

    by_name = tools["find_files"].handler({"name": "*.md"})
    by_text = tools["search_text_files"].handler({"content": "renewal"})
    by_directory = tools["list_workspace_directory"].handler(
        {"subpath": "contracts"}
    )
    inspected = tools["inspect_file"].handler({"path": "contracts/acme.md"})

    expected_paths = {"contracts/acme.md", "contracts/beta.md"}
    for result in (by_name, by_text, by_directory):
        assert {item["path"] for item in result["items"]} == expected_paths
        assert all("candidate_id" not in item for item in result["items"])
    assert inspected["items"][0]["path"] == "contracts/acme.md"
    assert "candidate_id" not in inspected["items"][0]

    for result, registration in (
        (by_directory, tools["list_workspace_directory"]),
        (by_name, tools["find_files"]),
        (by_text, tools["search_text_files"]),
        (inspected, tools["inspect_file"]),
    ):
        Draft202012Validator(_plain(registration.spec.output_schema)).validate(
            result
        )
        serialized = repr(result)
        assert "authority" not in serialized
        assert "fingerprint" not in serialized




@pytest.mark.parametrize(
    ("tool_id", "payload"),
    [
        ("workspace_overview", {"subpath": ".."}),
        ("list_workspace_directory", {"subpath": "../outside"}),
        ("find_files", {"name": "*", "subpath": "../outside"}),
        ("search_text_files", {"content": "x", "subpath": "/etc"}),
        ("inspect_file", {"path": "../outside.txt"}),
        ("inspect_file", {"path": "/etc/passwd"}),
    ],
)
def test_relative_path_inputs_cannot_escape_the_frozen_boundary(
    registrations, tool_id: str, payload: dict[str, object]
):
    with pytest.raises(ToolBusinessFailure) as raised:
        _by_id(registrations)[tool_id].handler(payload)
    assert raised.value.error.code == "invalid_request"


def test_symlink_escape_is_rejected_for_inspect_and_directory_scope(
    tree: Path, registrations, tmp_path: Path
):
    outside_file = tmp_path / "outside.txt"
    outside_file.write_text("outside", encoding="utf-8")
    outside_dir = tmp_path / "outside-dir"
    outside_dir.mkdir()
    (outside_dir / "outside.md").write_text("outside", encoding="utf-8")
    (tree / "file-link").symlink_to(outside_file)
    (tree / "dir-link").symlink_to(outside_dir, target_is_directory=True)
    tools = _by_id(registrations)

    with pytest.raises(ToolBusinessFailure):
        tools["inspect_file"].handler({"path": "file-link"})
    with pytest.raises(ToolBusinessFailure):
        tools["find_files"].handler({"name": "*", "subpath": "dir-link"})
    with pytest.raises(ToolBusinessFailure):
        tools["list_workspace_directory"].handler({"subpath": "dir-link"})


def test_inspect_returns_bounded_text_and_binary_metadata(registrations):
    tools = _by_id(registrations)

    text = tools["inspect_file"].handler(
        {"path": "contracts/acme.md", "max_bytes": 7}
    )
    binary = tools["inspect_file"].handler({"path": "image.png"})

    assert text["items"][0]["text"] == "renewal"
    assert text["truncated"] is True
    assert binary["items"][0]["content_kind"] == "binary"
    assert "text" not in binary["items"][0]
    assert binary["truncated"] is False


def test_inspect_authorized_file_is_not_blocked_by_its_name(registrations):
    result = _by_id(registrations)["inspect_file"].handler({"path": ".env"})

    assert result["total"] == 1
    assert result["returned"] == 1
    assert result["items"][0]["text"] == "API_KEY=do-not-return\n"
    assert result["skipped"] == []


def test_agent_private_tree_is_excluded_even_when_hidden_files_are_enabled(
    tmp_path: Path,
) -> None:
    root = tmp_path / "workspace"
    (root / "notes").mkdir(parents=True)
    (root / "notes" / "public.txt").write_text("ordinary note", encoding="utf-8")
    private = root / ".personagraph" / "output" / "session-1" / "draft.txt"
    private.parent.mkdir(parents=True)
    private.write_text("private control-plane secret", encoding="utf-8")
    tools = _by_id(
        build_workspace_discovery_tool_registrations(
            FrozenWorkspaceToolBoundary(
                session_id="session-1",
                root=root,
                hidden=True,
            )
        )
    )

    overview = tools["workspace_overview"].handler({"depth": 2})
    by_name = tools["find_files"].handler({"name": "*.txt", "limit": 50})
    by_text = tools["search_text_files"].handler(
        {"content": "private control-plane secret", "limit": 50}
    )

    assert overview["total_files"] == 1
    assert [item["path"] for item in by_name["items"]] == ["notes/public.txt"]
    assert by_name["total"] == 1
    assert by_text["items"] == []
    assert by_text["total"] == 0
    assert by_text["skipped"] == [{"reason": "denied_by_policy", "count": 1}]

    for tool_id, payload in (
        ("list_workspace_directory", {"subpath": ".personagraph"}),
        ("find_files", {"name": "*", "subpath": ".personagraph"}),
        ("inspect_file", {"path": ".personagraph/output/session-1/draft.txt"}),
    ):
        with pytest.raises(ToolBusinessFailure) as raised:
            tools[tool_id].handler(payload)
        assert raised.value.error.code == "invalid_request"


def test_outputs_never_disclose_the_absolute_workspace_root(
    tree: Path, registrations
):
    tools = _by_id(registrations)
    outputs = (
        tools["workspace_overview"].handler({}),
        tools["list_workspace_directory"].handler({}),
        tools["find_files"].handler({"name": "*.md"}),
        tools["search_text_files"].handler({"content": "renewal"}),
        tools["inspect_file"].handler({"path": "contracts/acme.md"}),
    )

    assert all(str(tree) not in repr(output) for output in outputs)


def test_text_search_reads_authorized_files_without_keyword_name_denials(
    tmp_path: Path,
):
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "secret_notes.txt").write_text(
        "ultra-private-needle\n",
        encoding="utf-8",
    )
    tools = _by_id(
        build_workspace_discovery_tool_registrations(
            FrozenWorkspaceToolBoundary(session_id="session-1", root=root)
        )
    )

    result = tools["search_text_files"].handler(
        {"content": "ultra-private-needle"}
    )

    assert result["items"][0]["path"] == "secret_notes.txt"
    assert result["returned"] == 1
    assert result["total"] == 1
    assert result["scanned_files"] == 1
    assert result["truncated"] is False
    assert result["skipped"] == []


def test_local_config_is_hidden_even_when_git_ignore_is_disabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from personagraph.configuration import paths

    root = tmp_path / "workspace"
    private = root / "configs" / "local"
    private.mkdir(parents=True)
    (private / "app_config.json").write_text(
        '{"marker": "must-not-be-seen"}\n',
        encoding="utf-8",
    )
    (root / "configs" / "preset.json").write_text(
        '{"marker": "public"}\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(paths, "LOCAL_CONFIG_DIR", private.resolve())
    tools = _by_id(
        build_workspace_discovery_tool_registrations(
            FrozenWorkspaceToolBoundary(
                session_id="session-1",
                root=root,
                respect_ignore=False,
            )
        )
    )

    overview = tools["workspace_overview"].handler({"depth": 2})
    by_name = tools["find_files"].handler({"name": "*.json"})
    by_text = tools["search_text_files"].handler(
        {"content": "must-not-be-seen"}
    )
    inspected = tools["inspect_file"].handler(
        {"path": "configs/local/app_config.json"}
    )

    assert overview["total_files"] == 1
    assert [item["path"] for item in by_name["items"]] == [
        "configs/preset.json"
    ]
    assert by_text["items"] == []
    assert by_text["scanned_files"] == 1
    assert by_text["skipped"] == [
        {"reason": "denied_by_policy", "count": 1}
    ]
    assert inspected["items"] == []
    assert inspected["skipped"] == [
        {"reason": "denied_by_policy", "count": 1}
    ]


def test_discovery_scan_cap_is_reported_as_incomplete_lower_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    import personagraph.tools.workspace.workspace_tools as module

    root = tmp_path / "workspace"
    root.mkdir()
    for name in ("a.txt", "b.txt", "c.txt"):
        (root / name).write_text("needle", encoding="utf-8")
    monkeypatch.setattr(module, "DISCOVERY_MAX_SCAN_PATHS", 2)
    tools = _by_id(
        build_workspace_discovery_tool_registrations(
            FrozenWorkspaceToolBoundary(session_id="session-1", root=root)
        )
    )

    names = tools["find_files"].handler({"name": "*.txt", "limit": 50})
    directory = tools["list_workspace_directory"].handler({"limit": 100})
    contents = tools["search_text_files"].handler(
        {"content": "needle", "limit": 50}
    )

    for result in (directory, names, contents):
        assert result["returned"] == 2
        assert result["scan_truncated"] is True
        assert result["total_relation"] == "lower_bound"
        assert result["truncated"] is True


def test_inspect_worst_case_json_escaping_stays_model_visible(
    tmp_path: Path,
):
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "controls.txt").write_bytes(b"\x01" * 12_001)
    registration = _by_id(
        build_workspace_discovery_tool_registrations(
            FrozenWorkspaceToolBoundary(session_id="session-1", root=root)
        )
    )["inspect_file"]

    outcome = ToolExecutor().execute(
        ResolvedInvocation(
            registration=registration,
            arguments={"path": "controls.txt", "max_bytes": 12_000},
        )
    )

    assert outcome.status.value == "succeeded"
    assert outcome.result is not None
    assert outcome.result["truncated"] is True


def test_offset_beyond_last_match_does_not_invent_a_larger_total(
    tmp_path: Path,
):
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "only.md").write_text("one", encoding="utf-8")
    tool = _by_id(
        build_workspace_discovery_tool_registrations(
            FrozenWorkspaceToolBoundary(session_id="session-1", root=root)
        )
    )["find_files"]

    result = tool.handler({"name": "*.md", "offset": 100, "limit": 10})

    assert result["items"] == []
    assert result["total"] == 1
    assert result["total_relation"] == "exact"
    assert result["truncated"] is False


def test_mixed_case_sensitive_directory_cannot_bypass_inspect_policy(
    tmp_path: Path,
):
    root = tmp_path / "workspace"
    protected = root / "kEyChAiNs"
    protected.mkdir(parents=True)
    (protected / "hidden.md").write_text("do not expose", encoding="utf-8")
    tool = _by_id(
        build_workspace_discovery_tool_registrations(
            FrozenWorkspaceToolBoundary(session_id="session-1", root=root)
        )
    )["inspect_file"]

    result = tool.handler({"path": "kEyChAiNs/hidden.md"})

    assert result["items"] == []
    assert result["skipped"] == [
        {"reason": "denied_by_policy", "count": 1}
    ]
