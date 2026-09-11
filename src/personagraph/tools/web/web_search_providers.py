"""Web 搜索工具的搜索提供商适配器。"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from typing import Any, Callable, cast

DEFAULT_PROVIDER_ORDER = ("tavily", "brave", "firecrawl", "exa", "ddg")
SEARCH_TIMEOUT_S = 15


@dataclass(frozen=True)
class SearchRequest:
    query: str
    k: int = 5
    topic: str | None = None
    time_range: str | None = None
    include_domains: tuple[str, ...] = ()
    exclude_domains: tuple[str, ...] = ()
    country: str | None = None
    search_depth: str | None = None


@dataclass(frozen=True)
class SearchResult:
    title: str
    url: str
    snippet: str = ""
    score: float | None = None
    published_at: str | None = None
    provider: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
            "score": self.score,
            "published_at": self.published_at,
            "provider": self.provider,
        }

    def citation(self, searched_at: str) -> dict[str, Any]:
        return {
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
            "published_at": self.published_at,
            "retrieved_at": searched_at,
            "provider": self.provider,
            "score": self.score,
        }


ProviderFn = Callable[[SearchRequest], list[SearchResult]]
CredentialProviderFn = Callable[
    [SearchRequest, str | None],
    list[SearchResult],
]


class SearchProviderError(RuntimeError):
    pass


def env(name: str) -> str:
    return os.getenv(name, "").strip()


def bounded_k(k: int) -> int:
    return max(1, min(int(k or 5), 20))


def as_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return tuple(x.strip() for x in value.split(",") if x.strip())
    if isinstance(value, list):
        return tuple(str(x).strip() for x in value if str(x).strip())
    return ()


def provider_order() -> list[str]:
    configured_order = env("PERSONAGRAPH_SEARCH_PROVIDER_ORDER")
    if configured_order:
        order = [x.strip() for x in configured_order.split(",") if x.strip()]
    else:
        first = env("PERSONAGRAPH_SEARCH_PROVIDER") or DEFAULT_PROVIDER_ORDER[0]
        order = [first, *DEFAULT_PROVIDER_ORDER]
    return list(dict.fromkeys(order))


def tavily_search(
    req: SearchRequest,
    credential: str | None = None,
) -> list[SearchResult]:
    import httpx

    if credential is None:
        raise SearchProviderError("tavily_credential_not_bound")
    key = credential
    if not key:
        raise SearchProviderError("no_tavily_key")
    body: dict[str, Any] = {
        "query": req.query,
        "max_results": req.k,
        "search_depth": req.search_depth or "basic",
        "include_answer": False,
        "include_raw_content": False,
    }
    if req.topic:
        body["topic"] = req.topic
    if req.time_range:
        body["time_range"] = req.time_range
    if req.include_domains:
        body["include_domains"] = list(req.include_domains)
    if req.exclude_domains:
        body["exclude_domains"] = list(req.exclude_domains)
    if req.country:
        body["country"] = req.country
    resp = httpx.post(
        "https://api.tavily.com/search",
        json=body,
        headers={"Authorization": f"Bearer {key}"},
        timeout=SEARCH_TIMEOUT_S,
    )
    resp.raise_for_status()
    out = []
    for r in resp.json().get("results", [])[:req.k]:
        out.append(SearchResult(
            title=r.get("title", ""),
            url=r.get("url", ""),
            snippet=(r.get("content") or "")[:700],
            score=r.get("score"),
            provider="tavily",
            raw=r,
        ))
    return out


def brave_search(
    req: SearchRequest,
    credential: str | None = None,
) -> list[SearchResult]:
    import httpx

    if credential is None:
        raise SearchProviderError("brave_credential_not_bound")
    key = credential
    if not key:
        raise SearchProviderError("no_brave_key")
    params: dict[str, Any] = {"q": req.query, "count": req.k, "extra_snippets": "true"}
    if req.country:
        params["country"] = req.country.upper()
    if req.time_range:
        freshness = {"day": "pd", "week": "pw", "month": "pm", "year": "py",
                     "d": "pd", "w": "pw", "m": "pm", "y": "py"}.get(req.time_range, req.time_range)
        params["freshness"] = freshness
    resp = httpx.get(
        "https://api.search.brave.com/res/v1/web/search",
        params=params,
        headers={"X-Subscription-Token": key, "Accept": "application/json"},
        timeout=SEARCH_TIMEOUT_S,
    )
    resp.raise_for_status()
    results = resp.json().get("web", {}).get("results", [])
    out = []
    for r in results[:req.k]:
        snippets = [r.get("description", ""), *r.get("extra_snippets", [])]
        out.append(SearchResult(
            title=r.get("title", ""),
            url=r.get("url", ""),
            snippet=" [...] ".join(s for s in snippets if s)[:900],
            published_at=r.get("age"),
            provider="brave",
            raw=r,
        ))
    return out


def firecrawl_search(
    req: SearchRequest,
    credential: str | None = None,
) -> list[SearchResult]:
    import httpx

    if credential is None:
        raise SearchProviderError("firecrawl_credential_not_bound")
    key = credential
    if not key:
        raise SearchProviderError("no_firecrawl_key")
    body: dict[str, Any] = {"query": req.query, "limit": req.k, "sources": ["web"]}
    if req.country:
        body["country"] = req.country.upper()
    if req.include_domains:
        body["includeDomains"] = list(req.include_domains)
    if req.exclude_domains:
        body["excludeDomains"] = list(req.exclude_domains)
    if req.time_range:
        body["tbs"] = {"day": "qdr:d", "week": "qdr:w", "month": "qdr:m", "year": "qdr:y",
                       "d": "qdr:d", "w": "qdr:w", "m": "qdr:m", "y": "qdr:y"}.get(req.time_range, req.time_range)
    resp = httpx.post(
        "https://api.firecrawl.dev/v2/search",
        json=body,
        headers={"Authorization": f"Bearer {key}"},
        timeout=SEARCH_TIMEOUT_S,
    )
    resp.raise_for_status()
    results = resp.json().get("data", {}).get("web", [])
    return [
        SearchResult(
            title=r.get("title", ""),
            url=r.get("url", ""),
            snippet=(r.get("description") or r.get("markdown") or "")[:900],
            provider="firecrawl",
            raw=r,
        )
        for r in results[:req.k]
    ]


def exa_search(
    req: SearchRequest,
    credential: str | None = None,
) -> list[SearchResult]:
    import httpx

    if credential is None:
        raise SearchProviderError("exa_credential_not_bound")
    key = credential
    if not key:
        raise SearchProviderError("no_exa_key")
    body: dict[str, Any] = {
        "query": req.query,
        "numResults": req.k,
        "type": req.search_depth or "auto",
        "contents": {"text": {"maxCharacters": 700}, "highlights": {"numSentences": 2}},
    }
    if req.include_domains:
        body["includeDomains"] = list(req.include_domains)
    if req.exclude_domains:
        body["excludeDomains"] = list(req.exclude_domains)
    if req.topic:
        body["category"] = req.topic
    resp = httpx.post(
        "https://api.exa.ai/search",
        json=body,
        headers={"x-api-key": key, "Content-Type": "application/json"},
        timeout=SEARCH_TIMEOUT_S,
    )
    resp.raise_for_status()
    out = []
    for r in resp.json().get("results", [])[:req.k]:
        text = r.get("text") or " ".join(r.get("highlights", []) or [])
        out.append(SearchResult(
            title=r.get("title", ""),
            url=r.get("url", ""),
            snippet=text[:900],
            score=r.get("score"),
            published_at=r.get("publishedDate"),
            provider="exa",
            raw=r,
        ))
    return out


def ddg_search(req: SearchRequest) -> list[SearchResult]:
    from ddgs import DDGS

    with DDGS() as d:
        return [
            SearchResult(
                title=r.get("title", ""),
                url=r.get("href", ""),
                snippet=(r.get("body") or "")[:700],
                provider="ddg",
                raw=r,
            )
            for r in d.text(req.query, max_results=req.k)
        ]


PROVIDERS: dict[str, ProviderFn] = {
    "tavily": tavily_search,
    "brave": brave_search,
    "firecrawl": firecrawl_search,
    "exa": exa_search,
    "ddg": ddg_search,
}

_CREDENTIAL_SLOTS_BY_PROVIDER: dict[ProviderFn, str] = {
    tavily_search: "TAVILY_API_KEY",
    brave_search: "BRAVE_SEARCH_API_KEY",
    firecrawl_search: "FIRECRAWL_API_KEY",
    exa_search: "EXA_API_KEY",
}


def credential_authority_slot(provider: ProviderFn) -> str | None:
    """Return the non-sensitive environment authority slot for a known adapter."""

    return _CREDENTIAL_SLOTS_BY_PROVIDER.get(provider)


def bind_provider_credential(
    provider: ProviderFn,
    credential: str,
) -> ProviderFn:
    """Close a credential into a known adapter without serializing it."""

    if provider not in _CREDENTIAL_SLOTS_BY_PROVIDER:
        return provider
    credential_provider = cast(CredentialProviderFn, provider)

    def bound(request: SearchRequest) -> list[SearchResult]:
        return credential_provider(request, credential)

    return bound
