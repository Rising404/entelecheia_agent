"""Web 工具注册必须保持无状态，并独立于 ToolRegistry。"""

from __future__ import annotations

import json
import subprocess
import sys

from personagraph.tools.contracts import thaw_json
from personagraph.tools.effects import EffectAction, EffectResource, EffectScopeKind
from personagraph.tools.execution import ResolvedInvocation, ToolExecutor
from personagraph.tools.policy import PolicyDisposition, PolicyRequest, ToolPolicyCore
from personagraph.tools.web.web_search_providers import SearchResult
from personagraph.tools.web.web_tools import (
    WEB_FETCH_TOOL_ID,
    WEB_SEARCH_TOOL_ID,
    build_web_tool_registrations,
)


def _registrations_by_id():
    return {
        item.tool_id: item for item in build_web_tool_registrations()
    }


def test_web_registrations_are_bounded_read_search_tools() -> None:
    registrations = _registrations_by_id()

    assert set(registrations) == {WEB_SEARCH_TOOL_ID, WEB_FETCH_TOOL_ID}
    search_effect = registrations[WEB_SEARCH_TOOL_ID].effect_profile.effects
    fetch_effect = registrations[WEB_FETCH_TOOL_ID].effect_profile.effects
    assert [
        (item.resource, item.action, item.scope_kind) for item in search_effect
    ] == [
        (EffectResource.NETWORK, EffectAction.SEARCH, EffectScopeKind.REMOTE_DOMAIN)
    ]
    assert [
        (item.resource, item.action, item.scope_kind) for item in fetch_effect
    ] == [
        (EffectResource.NETWORK, EffectAction.READ, EffectScopeKind.REMOTE_DOMAIN)
    ]
    assert registrations[WEB_SEARCH_TOOL_ID].execution_profile.max_transparent_retries == 1
    assert registrations[WEB_FETCH_TOOL_ID].execution_profile.max_transparent_retries == 0


def test_web_search_uses_provider_failover_without_research_session_side_effects(
    monkeypatch,
) -> None:
    from personagraph.tools.web import web_tools

    def unavailable(_request):
        raise RuntimeError("offline")

    def healthy(_request):
        return [
            SearchResult(
                title="Example",
                url="https://example.test/source",
                snippet="bounded source snippet",
                provider="healthy",
            )
        ]

    monkeypatch.setattr(web_tools, "provider_order", lambda: ["first", "second"])
    monkeypatch.setattr(
        web_tools,
        "PROVIDERS",
        {"first": unavailable, "second": healthy},
    )
    registration = _registrations_by_id()[WEB_SEARCH_TOOL_ID]

    outcome = ToolExecutor().execute(
        ResolvedInvocation(registration=registration, arguments={"query": "current facts"})
    )

    assert outcome.status.value == "succeeded"
    assert thaw_json(outcome.result) == {
        "query": "current facts",
        "provider": "second",
        "searched_at": outcome.result["searched_at"],
        "results": [
            {
                "title": "Example",
                "url": "https://example.test/source",
                "snippet": "bounded source snippet",
                "published_at": None,
                "provider": "healthy",
                "score": None,
            }
        ],
        "citations": [
            {
                "title": "Example",
                "url": "https://example.test/source",
                "snippet": "bounded source snippet",
                "published_at": None,
                "retrieved_at": outcome.result["searched_at"],
                "provider": "healthy",
            }
        ],
        "provider_failures": [
            {"provider": "first", "reason": "provider_error:RuntimeError"}
        ],
    }


def test_web_fetch_rejects_nonpublic_url_before_http_client_construction(
    monkeypatch,
) -> None:
    from personagraph.tools.web import web_tools

    monkeypatch.setattr(
        web_tools,
        "validate_stable_public_http_url",
        lambda _url: (False, "private_address_blocked", None),
    )
    registration = _registrations_by_id()[WEB_FETCH_TOOL_ID]

    outcome = ToolExecutor().execute(
        ResolvedInvocation(registration=registration, arguments={"url": "http://127.0.0.1/"})
    )

    assert outcome.status.value == "failed"
    assert outcome.error is not None
    assert outcome.error.code == "web_fetch_url_blocked"


def test_web_query_egress_is_allowed_without_separate_approval() -> None:
    registration = _registrations_by_id()[WEB_SEARCH_TOOL_ID]

    decision = ToolPolicyCore().evaluate(
        PolicyRequest.from_registration(
            registration,
            {"query": "contact alice@example.com"},
        )
    )

    assert decision.disposition is PolicyDisposition.ALLOW
    assert not decision.approval_required
    assert decision.effects[0].sensitive_egress is True


def test_importing_web_tools_does_not_load_registry_or_research_workflow() -> None:
    code = """
import json
import sys
import personagraph.tools.web.web_tools
blocked = {
    'personagraph.tools.registry',
    'personagraph.tools.defaults',
    'personagraph.tools.web_research',
    'personagraph.tools.web_tools',
    'personagraph.runtime',
    'personagraph.graph',
}
print(json.dumps(sorted(blocked & set(sys.modules))))
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(completed.stdout) == []
