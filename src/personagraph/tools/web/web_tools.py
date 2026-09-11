"""不依赖全局注册表的有界 Web 搜索与获取工具。

这些工具不依赖 ResearchSession、证据缓存或 ``ToolRegistry`` API。
上下文绑定会把凭据只保留在进程内闭包中；可序列化描述符不包含凭据或其哈希。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import secrets
from threading import Lock
from typing import Any
from urllib.parse import urljoin

from ..contracts import ToolSourceDescriptor, ToolSourceKind, ToolSpec
from ..effects import (
    DataEgress,
    EffectAction,
    EffectDescriptor,
    EffectResource,
    EffectScopeKind,
    Idempotency,
    Reversibility,
    ToolEffectProfile,
)
from ..execution import ToolBusinessFailure
from ..registration import ToolExecutionProfile, ToolRegistration
from .web_security import (
    pin_host,
    resolve_pin_target,
    validate_stable_public_http_url,
)
from .web_search_providers import (
    PROVIDERS,
    SEARCH_TIMEOUT_S,
    ProviderFn,
    SearchRequest,
    as_tuple,
    bind_provider_credential,
    bounded_k,
    credential_authority_slot,
    env as provider_env,
    provider_order,
)


WEB_SEARCH_TOOL_ID = "web_search"
WEB_FETCH_TOOL_ID = "web_fetch"
WEB_SEARCH_CONTRACT_VERSION = "web-search-v2"
WEB_FETCH_CONTRACT_VERSION = "web-fetch-v2"
WEB_IMPLEMENTATION_VERSION = "2"
WEB_SEARCH_PROVIDER_PIPELINE_SCHEMA = "web-search-provider-pipeline-v1"

WEB_TOOL_SOURCE = ToolSourceDescriptor(
    kind=ToolSourceKind.PROVIDER,
    source_id="personagraph.web.v2",
    fingerprint="web-v2-2026-08-26",
    display_name="Bounded public web access",
)
_FETCH_MAX_BYTES = 1_000_000
_FETCH_MAX_CHARACTERS = 48_000
_FETCH_MAX_REDIRECTS = 5
_USER_AGENT = "Entelecheia-Agent/2.0 (+local assistant)"


@dataclass(frozen=True)
class _FrozenSearchProvider:
    name: str
    callable_ref: str | None
    handler: ProviderFn | None
    credential_authority_source: str | None = None
    credential_authority_slot: str | None = None
    credential_authority_generation: str | None = None
    credential_configured: bool | None = None

    def credential_authority_descriptor(self) -> dict[str, object] | None:
        if self.credential_authority_slot is None:
            return None
        return {
            "source": self.credential_authority_source,
            "slot": self.credential_authority_slot,
            "generation": self.credential_authority_generation,
            "configured": self.credential_configured,
        }


class _ProcessCredentialAuthority:
    """Track credential rotation without deriving or persisting secret material.

    A random process-local incarnation is stable while the raw credential in this
    private memory cell is unchanged. Replacing a value in the same authority slot
    issues a new incarnation, so exact frozen rebind fails closed. A process restart
    also issues a new incarnation intentionally: without a safe account identifier,
    continuity across process memory cannot be proven.
    """

    def __init__(self) -> None:
        self._lock = Lock()
        self._current: dict[tuple[str, str], tuple[str, str]] = {}

    def freeze(self, *, source: str, slot: str, credential: str) -> str:
        key = (source, slot)
        with self._lock:
            if not credential:
                self._current.pop(key, None)
                return "unconfigured"
            current = self._current.get(key)
            if current is not None and current[0] == credential:
                return current[1]
            generation = secrets.token_hex(16)
            self._current[key] = (credential, generation)
            return generation


_PROCESS_CREDENTIAL_AUTHORITY = _ProcessCredentialAuthority()


@dataclass(frozen=True)
class FrozenWebSearchProviderPipeline:
    """One secret-free provider order and its exact live callable mapping."""

    providers: tuple[_FrozenSearchProvider, ...]

    @property
    def has_callable(self) -> bool:
        return any(provider.handler is not None for provider in self.providers)

    def descriptor(self) -> dict[str, object]:
        return {
            "schema_version": WEB_SEARCH_PROVIDER_PIPELINE_SCHEMA,
            "providers": [
                {
                    "name": provider.name,
                    "callable_ref": provider.callable_ref,
                    "credential_authority": (
                        provider.credential_authority_descriptor()
                    ),
                }
                for provider in self.providers
            ],
        }

    @property
    def fingerprint(self) -> str:
        encoded = json.dumps(
            self.descriptor(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def search(self, payload: dict[str, Any]) -> dict[str, Any]:
        request = _search_request(payload)
        searched_at = _now_iso()
        provider_failures: list[dict[str, str]] = []
        for provider in self.providers:
            if provider.handler is None:
                provider_failures.append(
                    {"provider": provider.name, "reason": "unknown_provider"}
                )
                continue
            try:
                raw_results = provider.handler(request)
            except Exception as exc:
                provider_failures.append(
                    {
                        "provider": provider.name,
                        "reason": f"provider_error:{type(exc).__name__}",
                    }
                )
                continue
            results = [
                _search_result_projection(item, provider.name)
                for item in raw_results
                if str(getattr(item, "url", "")).strip()
            ][: request.k]
            citations = [
                {
                    "title": item["title"],
                    "url": item["url"],
                    "snippet": item["snippet"],
                    "published_at": item["published_at"],
                    "retrieved_at": searched_at,
                    "provider": item["provider"],
                }
                for item in results
            ]
            return {
                "query": request.query,
                "provider": provider.name,
                "searched_at": searched_at,
                "results": results,
                "citations": citations,
                "provider_failures": provider_failures,
            }
        raise ToolBusinessFailure(
            "web_search_unavailable",
            "No configured web search provider completed the request.",
            {"provider_count": len(provider_failures)},
        )


def freeze_web_search_provider_pipeline(
    *,
    ordered_provider_names: Sequence[str] | None = None,
    providers: Mapping[str, ProviderFn] | None = None,
) -> FrozenWebSearchProviderPipeline:
    """Freeze provider callables and credentials into one process-local pipeline."""

    raw_order = provider_order() if ordered_provider_names is None else ordered_provider_names
    if isinstance(raw_order, (str, bytes)):
        raise TypeError("ordered_provider_names must be a sequence of provider names")
    order = tuple(
        dict.fromkeys(
            name
            for item in raw_order
            if (name := str(item).strip())
        )
    )
    selected = dict(PROVIDERS if providers is None else providers)
    frozen: list[_FrozenSearchProvider] = []
    for name in order:
        handler = selected.get(name)
        if handler is None:
            frozen.append(_FrozenSearchProvider(name, None, None))
            continue
        if not callable(handler):
            raise TypeError(f"web search provider {name!r} must be callable")
        slot = credential_authority_slot(handler)
        if slot is None:
            frozen.append(
                _FrozenSearchProvider(
                    name=name,
                    callable_ref=_stable_provider_callable_ref(handler),
                    handler=handler,
                )
            )
            continue
        source = "environment"
        credential = provider_env(slot)
        generation = _PROCESS_CREDENTIAL_AUTHORITY.freeze(
            source=source,
            slot=slot,
            credential=credential,
        )
        frozen.append(
            _FrozenSearchProvider(
                name=name,
                callable_ref=_stable_provider_callable_ref(handler),
                handler=bind_provider_credential(handler, credential),
                credential_authority_source=source,
                credential_authority_slot=slot,
                credential_authority_generation=generation,
                credential_configured=bool(credential),
            )
        )
    return FrozenWebSearchProviderPipeline(tuple(frozen))


def _stable_provider_callable_ref(handler: ProviderFn) -> str:
    module = str(getattr(handler, "__module__", "")).strip()
    qualname = str(getattr(handler, "__qualname__", "")).strip()
    if not module or not qualname:
        raise TypeError("web search provider must expose a stable module and qualname")
    return f"{module}:{qualname}"


def build_web_tool_registrations() -> tuple[ToolRegistration, ...]:
    """返回两个基础且独立于全局注册表的 Web 能力。"""

    return (
        ToolRegistration(
            spec=ToolSpec(
                tool_id=WEB_SEARCH_TOOL_ID,
                contract_version=WEB_SEARCH_CONTRACT_VERSION,
                name="Search the public web",
                description=(
                    "Search current public web sources and return bounded titles, "
                    "URLs, snippets, and citations. Use fetch for source text "
                    "when a result must be checked in detail."
                ),
                input_schema=_search_input_schema(),
                output_schema=_search_output_schema(),
                catalog_tags=("web", "search", "read"),
            ),
            implementation_version=WEB_IMPLEMENTATION_VERSION,
            source=WEB_TOOL_SOURCE,
            handler=_search,
            effect_profile=ToolEffectProfile(
                (
                    EffectDescriptor(
                        resource=EffectResource.NETWORK,
                        action=EffectAction.SEARCH,
                        scope_kind=EffectScopeKind.REMOTE_DOMAIN,
                        data_egress=DataEgress.CONTENT,
                        idempotency=Idempotency.IDEMPOTENT,
                        reversibility=Reversibility.REVERSIBLE,
                        egress_arguments=("query",),
                    ),
                )
            ),
            execution_profile=ToolExecutionProfile(
                default_timeout_s=float(SEARCH_TIMEOUT_S),
                hard_timeout_s=float(SEARCH_TIMEOUT_S) + 5.0,
                max_output_bytes=48_000,
                max_transparent_retries=1,
                concurrency_class="web_search",
            ),
        ),
        ToolRegistration(
            spec=ToolSpec(
                tool_id=WEB_FETCH_TOOL_ID,
                contract_version=WEB_FETCH_CONTRACT_VERSION,
                name="Fetch a public web page",
                description=(
                    "Fetch and extract bounded text from one public HTTP(S) URL. "
                    "Private, local, credential-bearing, and DNS-rebinding URLs "
                    "are rejected before any request."
                ),
                input_schema=_fetch_input_schema(),
                output_schema=_fetch_output_schema(),
                catalog_tags=("web", "fetch", "read"),
            ),
            implementation_version=WEB_IMPLEMENTATION_VERSION,
            source=WEB_TOOL_SOURCE,
            handler=_fetch,
            effect_profile=ToolEffectProfile(
                (
                    EffectDescriptor(
                        resource=EffectResource.NETWORK,
                        action=EffectAction.READ,
                        scope_kind=EffectScopeKind.REMOTE_DOMAIN,
                        data_egress=DataEgress.CONTENT,
                        idempotency=Idempotency.IDEMPOTENT,
                        reversibility=Reversibility.REVERSIBLE,
                        resource_argument="url",
                        scope_argument="url",
                    ),
                )
            ),
            execution_profile=ToolExecutionProfile(
                default_timeout_s=float(SEARCH_TIMEOUT_S),
                hard_timeout_s=float(SEARCH_TIMEOUT_S) + 5.0,
                max_output_bytes=80_000,
                # 重复 GET 仍可能向响应已经变化的来源重放请求。Runtime 会记录
                # 一次有界观测，而不是静默发起第二次获取。
                max_transparent_retries=0,
                concurrency_class="web_fetch",
            ),
        ),
    )


def _search(payload: dict[str, Any]) -> dict[str, Any]:
    return freeze_web_search_provider_pipeline().search(payload)


def _fetch(payload: dict[str, Any]) -> dict[str, Any]:
    raw_url = str(payload.get("url") or "").strip()
    ok, reason, current_url = validate_stable_public_http_url(raw_url)
    if not ok or current_url is None:
        raise ToolBusinessFailure(
            "web_fetch_url_blocked",
            "The requested URL is not a stable public HTTP(S) address.",
            {"reason": reason},
        )

    try:
        import httpx
        from bs4 import BeautifulSoup
    except ImportError as exc:
        raise ToolBusinessFailure(
            "web_fetch_unavailable",
            "This host does not provide the bounded web fetch dependencies.",
        ) from exc

    redirects = 0
    try:
        with httpx.Client(
            timeout=float(SEARCH_TIMEOUT_S),
            follow_redirects=False,
            headers={"User-Agent": _USER_AGENT},
            trust_env=False,
        ) as client:
            while True:
                ok, reason, safe_url, host, pin_ip = resolve_pin_target(current_url)
                if not ok or safe_url is None or host is None or pin_ip is None:
                    raise ToolBusinessFailure(
                        "web_fetch_url_blocked",
                        "The redirected URL is not a stable public HTTP(S) address.",
                        {"reason": reason},
                    )
                with pin_host(host, pin_ip), client.stream("GET", safe_url) as response:
                    if response.is_redirect:
                        redirects += 1
                        if redirects > _FETCH_MAX_REDIRECTS:
                            raise ToolBusinessFailure(
                                "web_fetch_redirect_limit",
                                "The web page exceeded the redirect limit.",
                            )
                        location = response.headers.get("location", "")
                        if not location:
                            raise ToolBusinessFailure(
                                "web_fetch_redirect_invalid",
                                "The web page returned an invalid redirect.",
                            )
                        current_url = urljoin(safe_url, location)
                        continue
                    response.raise_for_status()
                    content_type = response.headers.get("content-type", "").lower()
                    if content_type and not any(
                        allowed in content_type
                        for allowed in ("text/", "html", "xml", "json")
                    ):
                        raise ToolBusinessFailure(
                            "web_fetch_content_type_blocked",
                            "The web page did not return text content.",
                            {"content_type": content_type[:120]},
                        )
                    body, truncated_bytes = _read_response_bytes(response)
                    final_url = str(response.url)
                    encoding = response.encoding or "utf-8"
                    break
    except ToolBusinessFailure:
        raise
    except Exception as exc:
        raise ToolBusinessFailure(
            "web_fetch_failed",
            "The public web page could not be fetched.",
            {"exception_type": type(exc).__name__},
        ) from exc

    text = body.decode(encoding, errors="replace")
    soup = BeautifulSoup(text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()
    plain = "\n".join(
        line.strip() for line in soup.get_text("\n").splitlines() if line.strip()
    )
    content, truncated_characters = _bounded_text(plain, _FETCH_MAX_CHARACTERS)
    fetched_at = _now_iso()
    title = (soup.title.string or "").strip() if soup.title else ""
    citation = {
        "title": title,
        "url": final_url,
        "retrieved_at": fetched_at,
        "provider": "public_http",
    }
    return {
        "url": final_url,
        "title": title,
        "content": content,
        "content_type": content_type[:160],
        "truncated": truncated_bytes or truncated_characters,
        "fetched_at": fetched_at,
        "citations": [citation],
    }


def _search_request(payload: Mapping[str, Any]) -> SearchRequest:
    query = str(payload.get("query") or "").strip()
    if not query:
        raise ToolBusinessFailure("invalid_request", "query must not be empty.")
    try:
        k = bounded_k(int(payload.get("k", 5)))
    except (TypeError, ValueError) as exc:
        raise ToolBusinessFailure("invalid_request", "k must be an integer.") from exc
    return SearchRequest(
        query=query,
        k=min(k, 10),
        topic=_optional_text(payload.get("topic")),
        time_range=_optional_text(payload.get("time_range")),
        include_domains=as_tuple(payload.get("include_domains")),
        exclude_domains=as_tuple(payload.get("exclude_domains")),
        country=_optional_text(payload.get("country")),
        search_depth=_optional_text(payload.get("search_depth")),
    )


def _search_result_projection(item: object, default_provider: str) -> dict[str, Any]:
    return {
        "title": str(getattr(item, "title", "")).strip(),
        "url": str(getattr(item, "url", "")).strip(),
        "snippet": _bounded_text(str(getattr(item, "snippet", "")), 1_000)[0],
        "published_at": _optional_text(getattr(item, "published_at", None)),
        "provider": _optional_text(getattr(item, "provider", None))
        or default_provider,
        "score": _number_or_none(getattr(item, "score", None)),
    }


def _read_response_bytes(response: Any) -> tuple[bytes, bool]:
    chunks: list[bytes] = []
    total = 0
    for chunk in response.iter_bytes():
        remaining = _FETCH_MAX_BYTES - total
        if remaining <= 0:
            return b"".join(chunks), True
        if len(chunk) > remaining:
            chunks.append(chunk[:remaining])
            return b"".join(chunks), True
        chunks.append(chunk)
        total += len(chunk)
    return b"".join(chunks), False


def _bounded_text(value: str, maximum_characters: int) -> tuple[str, bool]:
    if len(value) <= maximum_characters:
        return value, False
    return value[:maximum_characters], True


def _number_or_none(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_text(value: object) -> str | None:
    text = str(value).strip() if value is not None else ""
    return text or None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _search_input_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["query"],
        "properties": {
            "query": {"type": "string", "minLength": 1, "maxLength": 800},
            "k": {"type": "integer", "minimum": 1, "maximum": 10, "default": 5},
            "topic": {"type": "string", "maxLength": 80},
            "time_range": {"type": "string", "maxLength": 80},
            "include_domains": {
                "type": "array",
                "maxItems": 20,
                "items": {"type": "string", "minLength": 1, "maxLength": 253},
            },
            "exclude_domains": {
                "type": "array",
                "maxItems": 20,
                "items": {"type": "string", "minLength": 1, "maxLength": 253},
            },
            "country": {"type": "string", "maxLength": 8},
            "search_depth": {"type": "string", "maxLength": 24},
        },
    }


def _search_output_schema() -> dict[str, Any]:
    citation = {
        "type": "object",
        "additionalProperties": False,
        "required": ["title", "url", "snippet", "published_at", "retrieved_at", "provider"],
        "properties": {
            "title": {"type": "string"},
            "url": {"type": "string", "minLength": 1},
            "snippet": {"type": "string"},
            "published_at": {"type": ["string", "null"]},
            "retrieved_at": {"type": "string", "minLength": 1},
            "provider": {"type": "string", "minLength": 1},
        },
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["query", "provider", "searched_at", "results", "citations", "provider_failures"],
        "properties": {
            "query": {"type": "string", "minLength": 1},
            "provider": {"type": "string", "minLength": 1},
            "searched_at": {"type": "string", "minLength": 1},
            "results": {
                "type": "array",
                "maxItems": 10,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["title", "url", "snippet", "published_at", "provider", "score"],
                    "properties": {
                        "title": {"type": "string"},
                        "url": {"type": "string", "minLength": 1},
                        "snippet": {"type": "string"},
                        "published_at": {"type": ["string", "null"]},
                        "provider": {"type": "string", "minLength": 1},
                        "score": {"type": ["number", "null"]},
                    },
                },
            },
            "citations": {"type": "array", "maxItems": 10, "items": citation},
            "provider_failures": {
                "type": "array",
                "maxItems": 5,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["provider", "reason"],
                    "properties": {
                        "provider": {"type": "string", "minLength": 1},
                        "reason": {"type": "string", "minLength": 1, "maxLength": 120},
                    },
                },
            },
        },
    }


def _fetch_input_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["url"],
        "properties": {"url": {"type": "string", "minLength": 1, "maxLength": 4_096}},
    }


def _fetch_output_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["url", "title", "content", "content_type", "truncated", "fetched_at", "citations"],
        "properties": {
            "url": {"type": "string", "minLength": 1},
            "title": {"type": "string"},
            "content": {"type": "string"},
            "content_type": {"type": "string"},
            "truncated": {"type": "boolean"},
            "fetched_at": {"type": "string", "minLength": 1},
            "citations": {
                "type": "array",
                "minItems": 1,
                "maxItems": 1,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["title", "url", "retrieved_at", "provider"],
                    "properties": {
                        "title": {"type": "string"},
                        "url": {"type": "string", "minLength": 1},
                        "retrieved_at": {"type": "string", "minLength": 1},
                        "provider": {"type": "string", "minLength": 1},
                    },
                },
            },
        },
    }


__all__ = [
    "FrozenWebSearchProviderPipeline",
    "WEB_FETCH_CONTRACT_VERSION",
    "WEB_FETCH_TOOL_ID",
    "WEB_IMPLEMENTATION_VERSION",
    "WEB_SEARCH_CONTRACT_VERSION",
    "WEB_SEARCH_PROVIDER_PIPELINE_SCHEMA",
    "WEB_SEARCH_TOOL_ID",
    "WEB_TOOL_SOURCE",
    "build_web_tool_registrations",
    "freeze_web_search_provider_pipeline",
]
