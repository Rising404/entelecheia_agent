from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

import pytest

from personagraph.tools.catalog.binding import FrozenToolBinding
from personagraph.tools.catalog.persistence import ToolCatalogRepository
from personagraph.tools.composition.default_catalog import (
    bootstrap_production_default_catalog,
    build_production_default_catalog_seeds,
    build_production_default_factory_registry,
)
from personagraph.tools.contracts import thaw_json
from personagraph.tools.catalog.default_profile import (
    materialize_default_catalog,
)
from personagraph.tools.execution import ResolvedInvocation, ToolExecutor
from personagraph.tools.catalog.snapshots.attempt import (
    encode_frozen_attempt_tool_catalog,
)
from personagraph.tools.catalog.materialization import (
    RuntimeCatalogMaterializer,
)
from personagraph.tools.catalog.trusted_factories import (
    TrustedContextualDefaultToolFactory,
    TrustedStaticDefaultToolFactory,
)
from personagraph.tools.web import web_tools
from personagraph.tools.web import web_search_providers
from personagraph.tools.web.web_search_providers import SearchResult
from personagraph.tools.web.web_tools import (
    WEB_FETCH_TOOL_ID,
    WEB_SEARCH_TOOL_ID,
    build_web_tool_registrations,
)


def _alpha_search(_request):
    return [
        SearchResult(
            title="Alpha",
            url="https://example.test/alpha",
            provider="alpha_callable",
        )
    ]


def _beta_search(_request):
    return [
        SearchResult(
            title="Beta",
            url="https://example.test/beta",
            provider="beta_callable",
        )
    ]


def _web_search(materialized):
    return next(
        registration
        for registration in materialized.registrations
        if registration.tool_id == WEB_SEARCH_TOOL_ID
    )


def _bootstrap_factory_owned_defaults(
    repository: ToolCatalogRepository,
):
    """Keep factory-specific tests independent of contextual workspace candidates."""

    registry = build_production_default_factory_registry()
    factory_identities = {factory.identity for factory in registry.factories()}
    repository.bootstrap(
        tuple(
            seed
            for seed in build_production_default_catalog_seeds()
            if seed.definition.identity in factory_identities
        )
    )
    return registry


def test_web_search_definition_is_environment_independent_and_contextual(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(web_tools, "provider_order", lambda: ["alpha", "beta"])
    monkeypatch.setattr(
        web_tools,
        "PROVIDERS",
        {"alpha": _alpha_search, "beta": _beta_search},
    )
    first = build_production_default_catalog_seeds()

    monkeypatch.setattr(web_tools, "provider_order", lambda: ["beta", "alpha"])
    monkeypatch.setattr(
        web_tools,
        "PROVIDERS",
        {"alpha": _beta_search, "beta": _alpha_search},
    )
    second = build_production_default_catalog_seeds()

    first_search = next(
        seed for seed in first if seed.definition.identity.tool_id == WEB_SEARCH_TOOL_ID
    )
    second_search = next(
        seed for seed in second if seed.definition.identity.tool_id == WEB_SEARCH_TOOL_ID
    )
    assert first_search.definition == second_search.definition
    legacy_search_registration = next(
        registration
        for registration in build_web_tool_registrations()
        if registration.tool_id == WEB_SEARCH_TOOL_ID
    )
    legacy_search_factory = TrustedStaticDefaultToolFactory.from_registration(
        implementation_ref="builtin/web_search",
        declared_behavior_revision="bounded-web-search-handler-1",
        registration=legacy_search_registration,
    )
    assert first_search.definition == legacy_search_factory.definition

    factories = build_production_default_factory_registry().factories()
    search_factory = next(
        factory for factory in factories if factory.identity.tool_id == WEB_SEARCH_TOOL_ID
    )
    fetch_factory = next(
        factory for factory in factories if factory.identity.tool_id == WEB_FETCH_TOOL_ID
    )
    assert isinstance(search_factory, TrustedContextualDefaultToolFactory)
    assert isinstance(fetch_factory, TrustedStaticDefaultToolFactory)


def test_new_materialization_freezes_provider_order_and_callable_mapping(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(web_tools, "provider_order", lambda: ["alpha", "beta"])
    monkeypatch.setattr(
        web_tools,
        "PROVIDERS",
        {"alpha": _alpha_search, "beta": _beta_search},
    )
    repository = ToolCatalogRepository(tmp_path / "catalog.sqlite")
    registry = _bootstrap_factory_owned_defaults(repository)
    first = _web_search(materialize_default_catalog(repository, registry))
    frozen = FrozenToolBinding.from_binding(first.binding)

    monkeypatch.setattr(web_tools, "provider_order", lambda: ["beta", "alpha"])
    monkeypatch.setattr(
        web_tools,
        "PROVIDERS",
        {"alpha": _beta_search, "beta": _alpha_search},
    )
    second = _web_search(materialize_default_catalog(repository, registry))

    assert first.definition == second.definition
    assert first.binding_digest != second.binding_digest
    with pytest.raises(ValueError, match="does not match frozen binding"):
        frozen.require_exact_live_binding(second.binding)

    first_outcome = ToolExecutor().execute(
        ResolvedInvocation(first, {"query": "frozen provider"})
    )
    second_outcome = ToolExecutor().execute(
        ResolvedInvocation(second, {"query": "new provider"})
    )
    assert first_outcome.status.value == "succeeded"
    assert second_outcome.status.value == "succeeded"
    assert thaw_json(first_outcome.result)["provider"] == "alpha"
    assert thaw_json(first_outcome.result)["results"][0]["provider"] == (
        "alpha_callable"
    )
    assert thaw_json(second_outcome.result)["provider"] == "beta"
    assert thaw_json(second_outcome.result)["results"][0]["provider"] == (
        "alpha_callable"
    )


def test_provider_pipeline_freezes_credential_authority_without_secret_material(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(web_tools, "provider_order", lambda: ["tavily"])
    monkeypatch.setattr(
        web_tools,
        "PROVIDERS",
        {"tavily": web_search_providers.tavily_search},
    )
    monkeypatch.setenv("TAVILY_API_KEY", "first-secret-value")
    repository = ToolCatalogRepository(tmp_path / "catalog.sqlite")
    registry = _bootstrap_factory_owned_defaults(repository)
    first = _web_search(materialize_default_catalog(repository, registry))

    monkeypatch.setenv("TAVILY_API_KEY", "second-secret-value")
    second = _web_search(materialize_default_catalog(repository, registry))

    import httpx

    request_headers: list[dict[str, str]] = []

    class _Response:
        def raise_for_status(self) -> None:
            return None

        def json(self):
            return {
                "results": [
                    {
                        "title": "Frozen credential",
                        "url": "https://example.test/frozen-credential",
                        "content": "bounded",
                    }
                ]
            }

    def post(_url, *, json, headers, timeout):
        del json, timeout
        request_headers.append(dict(headers))
        return _Response()

    monkeypatch.setattr(httpx, "post", post)
    outcome = ToolExecutor().execute(
        ResolvedInvocation(first, {"query": "credential binding"})
    )

    descriptor = json.dumps(first.binding.descriptor(), sort_keys=True)
    assert outcome.status.value == "succeeded"
    assert request_headers == [{"Authorization": "Bearer first-secret-value"}]
    assert first.binding_digest != second.binding_digest
    with pytest.raises(ValueError, match="does not match frozen binding"):
        FrozenToolBinding.from_binding(first.binding).require_exact_live_binding(
            second.binding
        )
    assert "environment" in descriptor
    assert "TAVILY_API_KEY" in descriptor
    for secret in ("first-secret-value", "second-secret-value"):
        assert secret not in descriptor
        assert hashlib.sha256(secret.encode("utf-8")).hexdigest() not in descriptor


def test_known_contextual_factory_reports_empty_pipeline_as_optional_unavailable(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(web_tools, "provider_order", lambda: ["missing"])
    monkeypatch.setattr(web_tools, "PROVIDERS", {})
    repository = ToolCatalogRepository(tmp_path / "catalog.sqlite")
    registry = _bootstrap_factory_owned_defaults(repository)

    materialized = materialize_default_catalog(
        repository,
        registry,
    )

    assert WEB_SEARCH_TOOL_ID not in {
        registration.tool_id for registration in materialized.registrations
    }
    unavailable = next(
        item for item in materialized.unavailable if item.identity.tool_id == WEB_SEARCH_TOOL_ID
    )
    assert unavailable.implementation_ref == "builtin/web_search"
    assert unavailable.reason == "web_search_provider_pipeline_unavailable"


def test_production_catalog_materializes_and_exactly_rebinds_one_attempt(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(web_tools, "provider_order", lambda: ["alpha"])
    monkeypatch.setattr(web_tools, "PROVIDERS", {"alpha": _alpha_search})
    repository = ToolCatalogRepository(tmp_path / "catalog.sqlite")
    bootstrap_production_default_catalog(repository)
    materializer = RuntimeCatalogMaterializer(
        repository,
        build_production_default_factory_registry(),
    )

    original = materializer.materialize_new(
        attempt_id="production-attempt-1",
        created_at=datetime(2026, 9, 3, 17, 0, tzinfo=timezone.utc),
    )
    encoded = encode_frozen_attempt_tool_catalog(
        original.frozen_attempt_catalog
    )
    rebound = materializer.rebind_frozen(
        canonical_json=encoded.canonical_json,
        expected_sha256=encoded.sha256,
        expected_attempt_id="production-attempt-1",
    )

    assert [item.identity.tool_id for item in original.exposed_definitions] == [
        "get_today",
        "date_after",
        "web_search",
        "web_fetch",
    ]
    assert [item.identity.tool_id for item in original.unavailable] == [
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
    ]
    assert {item.reason for item in original.unavailable} == {
        "contextual_binding_unavailable"
    }
    assert rebound.frozen_attempt_catalog == original.frozen_attempt_catalog
    assert rebound.exposure_order == original.exposure_order
