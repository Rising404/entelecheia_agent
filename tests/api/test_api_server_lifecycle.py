from __future__ import annotations

import pytest

from personagraph.api import server
from personagraph.tools.catalog import CatalogStatus
from personagraph.tools.catalog.persistence import (
    DefaultProfileAvailability,
    DefaultProfileSelection,
    ToolCatalogSeed,
)
from personagraph.tools.catalog.persistence import repository as catalog_persistence
from personagraph.tools.composition.default_catalog import (
    build_production_default_catalog_seeds,
)
from personagraph.tools.documents import (
    build_mounted_document_tool_definition_manifest,
)
from personagraph.tools.visual import (
    build_mounted_visual_tool_definition_manifest,
)


class _StopServer(RuntimeError):
    pass


_EXPECTED_PRODUCTION_TOOL_IDS = (
    "get_today",
    "date_after",
    "web_search",
    "web_fetch",
    "workspace_overview",
    "list_workspace_directory",
    "find_files",
    "search_text_files",
    "inspect_file",
    "read_text",
    "read_pdf_text",
    "read_word",
    "read_slides",
    "inspect_image",
    "analyze_image",
    "analyze_pdf_page",
    "retrieve_files",
    "check_files_state",
    "prepare_files",
    "read_file_chunks",
    "inspect_file_chunks",
    "search_file_text",
    "write_workspace_file",
    "create_output_file",
    "retrieve_history",
    "list_file_visuals",
    "read_file_visuals",
    "record_execution_findings",
    "revise_execution_finding",
    "list_tool_results",
    "read_tool_result",
)


def test_host_bootstrap_publishes_the_process_default_catalog(
    tmp_path,
    monkeypatch,
) -> None:
    database_path = tmp_path / "tool_catalog.sqlite"
    monkeypatch.setattr(
        catalog_persistence,
        "DEFAULT_TOOL_CATALOG_DATABASE_PATH",
        database_path,
    )

    server._bootstrap_tool_catalog()
    server._bootstrap_tool_catalog()

    repository = catalog_persistence.ToolCatalogRepository(database_path)
    profile = repository.current_default_profile()
    assert tuple(item.identity.tool_id for item in profile.items) == (
        _EXPECTED_PRODUCTION_TOOL_IDS
    )
    assert repository.current_snapshot().revision == 1


@pytest.mark.parametrize("predecessor_count", (4, 8, 14, 16, 21))
def test_host_bootstrap_hard_cuts_noncanonical_prefix_profiles(
    tmp_path,
    monkeypatch,
    predecessor_count: int,
) -> None:
    database_path = tmp_path / "tool_catalog.sqlite"
    repository = catalog_persistence.ToolCatalogRepository(database_path)
    seeds = build_production_default_catalog_seeds()
    repository.bootstrap(seeds[:predecessor_count])
    monkeypatch.setattr(
        catalog_persistence,
        "DEFAULT_TOOL_CATALOG_DATABASE_PATH",
        database_path,
    )

    server._bootstrap_tool_catalog()

    snapshot = repository.current_snapshot()
    profile = repository.current_default_profile()
    assert snapshot.revision == len(seeds) + 2 - predecessor_count
    assert profile.revision == 2
    assert profile.catalog_revision == snapshot.revision
    assert tuple(item.identity.tool_id for item in profile.items) == tuple(
        seed.definition.identity.tool_id
        for seed in seeds
    )
    assert {entry.status for entry in snapshot.entries} == {CatalogStatus.ACTIVE}

    server._bootstrap_tool_catalog()

    assert repository.current_snapshot() == snapshot
    assert repository.current_default_profile() == profile


def test_host_bootstrap_migrates_the_legacy_mounted_profile(
    tmp_path,
    monkeypatch,
) -> None:
    database_path = tmp_path / "tool_catalog.sqlite"
    repository = catalog_persistence.ToolCatalogRepository(database_path)
    current = build_production_default_catalog_seeds()
    mounted = (
        *build_mounted_document_tool_definition_manifest(),
        *build_mounted_visual_tool_definition_manifest(),
    )
    legacy = (
        *current[:-1],
        *(
            ToolCatalogSeed(
                item.definition,
                DefaultProfileAvailability.IF_AVAILABLE,
            )
            for item in mounted
        ),
    )
    repository.bootstrap(legacy)
    monkeypatch.setattr(
        catalog_persistence,
        "DEFAULT_TOOL_CATALOG_DATABASE_PATH",
        database_path,
    )

    server._bootstrap_tool_catalog()

    snapshot = repository.current_snapshot()
    profile = repository.current_default_profile()
    assert snapshot.revision == 3
    assert profile.revision == 2
    assert tuple(item.identity.tool_id for item in profile.items) == (
        _EXPECTED_PRODUCTION_TOOL_IDS
    )
    assert {entry.status for entry in snapshot.entries} == {CatalogStatus.ACTIVE}


def test_host_bootstrap_replaces_a_custom_profile_with_the_canonical_profile(
    tmp_path,
    monkeypatch,
) -> None:
    database_path = tmp_path / "tool_catalog.sqlite"
    repository = catalog_persistence.ToolCatalogRepository(database_path)
    seeds = build_production_default_catalog_seeds()
    repository.bootstrap(seeds[:4])
    custom_profile = repository.publish_default_profile(
        tuple(
            DefaultProfileSelection(seed.definition.identity, seed.availability)
            for seed in reversed(seeds[:3])
        ),
        expected_catalog_revision=1,
        expected_profile_revision=1,
        actor="test-operator",
        reason="customize default tool order and membership",
    )
    monkeypatch.setattr(
        catalog_persistence,
        "DEFAULT_TOOL_CATALOG_DATABASE_PATH",
        database_path,
    )

    server._bootstrap_tool_catalog()

    snapshot = repository.current_snapshot()
    profile = repository.current_default_profile()
    assert snapshot.revision == len(seeds) - 2
    assert profile != custom_profile
    assert profile.revision == 3
    assert profile.catalog_revision == snapshot.revision
    assert tuple(item.identity for item in profile.items) == tuple(
        seed.definition.identity for seed in seeds
    )
    assert {entry.status for entry in snapshot.entries} == {CatalogStatus.ACTIVE}


def test_run_owns_document_maintenance_lifecycle_until_http_shutdown(monkeypatch):
    events: list[object] = []

    class FakeHttpServer:
        def __init__(self, address, handler, *, bind_and_activate) -> None:
            events.append(
                ("http_created", address, handler, bind_and_activate)
            )

        def server_bind(self) -> None:
            events.append("http_bound")

        def server_activate(self) -> None:
            events.append("http_activated")

        def serve_forever(self) -> None:
            events.append("http_serving")
            raise _StopServer

        def server_close(self) -> None:
            events.append("http_closed")

    class FakeLifecycle:
        def start(self) -> bool:
            events.append("maintenance_started")
            return True

        def stop(self, *, timeout_seconds: float) -> bool:
            events.append(("maintenance_stopped", timeout_seconds))
            return True

    class FakePostCommitLifecycle:
        def start(self) -> bool:
            events.append("post_commit_started")
            return True

        def stop(self, *, timeout_seconds: float) -> bool:
            events.append(("post_commit_stopped", timeout_seconds))
            return True

    monkeypatch.setattr(server, "ThreadingHTTPServer", FakeHttpServer)
    monkeypatch.setattr(
        server.api_security,
        "initialize_api_token",
        lambda token: events.append(("token_initialized", token)),
    )
    monkeypatch.setattr(
        server,
        "_bootstrap_tool_catalog",
        lambda: events.append("tool_catalog_bootstrapped"),
        raising=False,
    )
    monkeypatch.setattr(
        server,
        "_build_document_maintenance_lifecycle",
        lambda: events.append("maintenance_built") or FakeLifecycle(),
        raising=False,
    )
    monkeypatch.setattr(
        server,
        "_recover_active_l1_turns",
        lambda: events.append("l1_turns_recovered") or 0,
        raising=False,
    )
    monkeypatch.setattr(
        server, "_build_turn_post_commit_lifecycle",
        lambda: events.append("post_commit_built") or FakePostCommitLifecycle(),
        raising=False,
    )

    with pytest.raises(_StopServer):
        server.run(port=4321, api_token="a" * 40)

    assert events == [
        (
            "http_created",
            (server.DEFAULT_HOST, 4321),
            server.ApiHandler,
            False,
        ),
        "http_bound",
        ("token_initialized", "a" * 40),
        "tool_catalog_bootstrapped",
        "maintenance_built",
        "maintenance_started",
        "post_commit_built",
        "post_commit_started",
        "l1_turns_recovered",
        "http_activated",
        "http_serving",
        ("post_commit_stopped", 5.0),
        ("maintenance_stopped", 5.0),
        "http_closed",
    ]


def test_run_fails_before_activating_the_listener_when_catalog_bootstrap_fails(
    monkeypatch,
) -> None:
    events: list[object] = []

    class FakeHttpServer:
        def __init__(self, *_args, **kwargs) -> None:
            events.append(("http_created", kwargs["bind_and_activate"]))

        def server_bind(self) -> None:
            events.append("http_bound")

        def server_activate(self) -> None:
            events.append("http_activated")

        def server_close(self) -> None:
            events.append("http_closed")

    monkeypatch.setattr(
        server.api_security,
        "initialize_api_token",
        lambda token: events.append(("token_initialized", token)),
    )

    def fail_catalog_bootstrap() -> None:
        events.append("tool_catalog_bootstrap_failed")
        raise RuntimeError("catalog unavailable")

    monkeypatch.setattr(server, "_bootstrap_tool_catalog", fail_catalog_bootstrap)
    monkeypatch.setattr(server, "ThreadingHTTPServer", FakeHttpServer)

    with pytest.raises(RuntimeError, match="catalog unavailable"):
        server.run(port=4321, api_token="a" * 40)

    assert events == [
        ("http_created", False),
        "http_bound",
        ("token_initialized", "a" * 40),
        "tool_catalog_bootstrap_failed",
        "http_closed",
    ]


def test_port_conflict_fails_before_rotating_the_shared_api_token(
    monkeypatch,
) -> None:
    events: list[str] = []
    token_initializations: list[str | None] = []

    class PortAlreadyOwnedServer:
        def __init__(self, *_args, **_kwargs) -> None:
            events.append("http_created")

        def server_bind(self) -> None:
            events.append("http_bind_failed")
            raise OSError("address already in use")

        def server_close(self) -> None:
            events.append("http_closed")

    monkeypatch.setattr(server, "ThreadingHTTPServer", PortAlreadyOwnedServer)
    monkeypatch.setattr(
        server.api_security,
        "initialize_api_token",
        lambda token: token_initializations.append(token),
    )

    with pytest.raises(OSError, match="address already in use"):
        server.run(port=4321, api_token="a" * 40)

    assert token_initializations == []
    assert events == ["http_created", "http_bind_failed", "http_closed"]
