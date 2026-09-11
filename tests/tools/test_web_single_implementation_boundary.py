"""Lock bounded Web tools to one responsibility-named implementation."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import re


_ROOT = Path(__file__).resolve().parents[2]
_SOURCE_ROOT = _ROOT / "src" / "personagraph"
_CANONICAL_MODULE = "personagraph.tools.web.web_tools"
_RETIRED_MODULE = "personagraph.tools.web.web_v2"
_RETIRED_DEFAULT_BUILDER = "build_default_v2_tool_registrations"
_RETIRED_SOURCE = _SOURCE_ROOT / "tools" / "web" / "web_v2.py"
_CANONICAL_SOURCE = _SOURCE_ROOT / "tools" / "web" / "web_tools.py"
_WEB_TEST = _ROOT / "tests" / "tools" / "test_web_tools.py"
_RETIRED_WORDING = (
    "V2 Web",
    "Web V2",
    "test_v2_web_",
    "_requires_v2_",
    "_without_v1_",
)


def test_retired_web_and_default_catalog_implementations_cannot_return() -> None:
    retired_tokens = (
        re.compile(r"\bweb_v2\b"),
        re.compile(rf"\b{_RETIRED_DEFAULT_BUILDER}\b"),
    )
    paths = (
        *sorted(_SOURCE_ROOT.rglob("*.py")),
        *sorted((_ROOT / "tests").rglob("test_*.py")),
    )
    violations: dict[str, list[str]] = {}
    for path in paths:
        if path == Path(__file__).resolve():
            continue
        content = path.read_text(encoding="utf-8")
        matches = [
            token.pattern
            for token in retired_tokens
            if token.search(content)
        ]
        if matches:
            violations[str(path.relative_to(_ROOT))] = matches

    assert violations == {}
    assert not _RETIRED_SOURCE.exists()
    assert _CANONICAL_SOURCE.is_file()
    assert importlib.util.find_spec(_RETIRED_MODULE) is None
    assert importlib.util.find_spec(_CANONICAL_MODULE) is not None


def test_web_module_and_tests_use_responsibility_language() -> None:
    web_sources = tuple(sorted((_SOURCE_ROOT / "tools" / "web").glob("*.py")))
    for path in (*web_sources, _WEB_TEST):
        content = path.read_text(encoding="utf-8")
        assert all(wording not in content for wording in _RETIRED_WORDING)


def test_web_wire_and_provider_identities_do_not_drift() -> None:
    from personagraph.tools.web import web_tools

    registrations = web_tools.build_web_tool_registrations()

    assert tuple(
        (
            registration.tool_id,
            registration.contract_version,
            registration.implementation_version,
            registration.source.source_id,
            registration.source.fingerprint,
        )
        for registration in registrations
    ) == (
        (
            "web_search",
            "web-search-v2",
            "2",
            "personagraph.web.v2",
            "web-v2-2026-08-26",
        ),
        (
            "web_fetch",
            "web-fetch-v2",
            "2",
            "personagraph.web.v2",
            "web-v2-2026-08-26",
        ),
    )
    assert web_tools._USER_AGENT == "Entelecheia-Agent/2.0 (+local assistant)"
