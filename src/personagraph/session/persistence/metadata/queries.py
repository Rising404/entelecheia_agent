"""Shared Session filtering and search projection, independent of database layout.

Storage adapters supply rows ordered by last activity and load a transcript only
when needed. Search reads each matching Session's transcript once, not once for
selection and again for hit details.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from itertools import islice
from typing import Any


def _filtered_sessions(
    rows: Iterable[dict[str, Any]],
    *,
    status: str,
    include_archived: bool = False,
    include_trashed: bool = False,
    folder_id: str | None,
    persona_id: str | None,
) -> Iterator[dict[str, Any]]:
    if status == "all":
        statuses = {"active", "archived", "trashed"}
    elif status in {"active", "archived", "trashed"}:
        statuses = {status}
        if status == "active" and include_archived:
            statuses.add("archived")
        if status == "active" and include_trashed:
            statuses.add("trashed")
    else:
        raise ValueError(f"invalid session status: {status}")
    return (
        row for row in rows
        if row.get("status") in statuses
        and (folder_id is None or row.get("folder_id") == folder_id)
        and (persona_id is None or row.get("persona_id") == persona_id)
    )


def _take_rows(
    rows: Iterable[dict[str, Any]], limit: int | None
) -> list[dict[str, Any]]:
    return list(rows if limit is None else islice(rows, max(0, int(limit))))


def list_sessions(
    ordered_rows: Iterable[dict[str, Any]],
    load_turns: Callable[[str], Iterable[dict[str, Any]]],
    *,
    include_archived: bool = False,
    include_trashed: bool = False,
    status: str = "active",
    folder_id: str | None = None,
    persona_id: str | None = None,
    query: str | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    rows = _filtered_sessions(
        ordered_rows,
        status=status,
        include_archived=include_archived,
        include_trashed=include_trashed,
        folder_id=folder_id,
        persona_id=persona_id,
    )
    needle = str(query or "").strip().lower()
    if needle:
        rows = (
            row for row in rows
            if needle in str(row.get("title") or "").lower()
            or any(
                needle in str(turn.get("content") or "").lower()
                for turn in load_turns(str(row["id"]))
            )
        )
    return _take_rows(rows, limit)


def _snippet(text: str, query: str, radius: int = 24) -> str:
    position = text.lower().find(query.lower())
    if position < 0:
        return text[: radius * 2]
    start = max(0, position - radius)
    end = min(len(text), position + len(query) + radius)
    return (
        ("…" if start > 0 else "")
        + text[start:end]
        + ("…" if end < len(text) else "")
    )


def _search_results(
    rows: Iterable[dict[str, Any]],
    load_turns: Callable[[str], Iterable[dict[str, Any]]],
    query: str,
) -> Iterator[dict[str, Any]]:
    needle = query.lower()
    for row in rows:
        title = str(row.get("title") or "")
        hit_count = int(needle in title.lower())
        first = None
        for turn in load_turns(str(row["id"])):
            if needle in str(turn.get("content") or "").lower():
                hit_count += 1
                if first is None:
                    first = turn
        if hit_count:
            yield {
                **row,
                "hit_count": hit_count,
                "match_turn_idx": first["turn_idx"] if first else None,
                "match_role": first["role"] if first else "title",
                "match_snippet": (
                    _snippet(str(first["content"]), query) if first else title
                ),
            }


def search_sessions(
    ordered_rows: Iterable[dict[str, Any]],
    load_turns: Callable[[str], Iterable[dict[str, Any]]],
    *,
    query: str,
    status: str = "active",
    folder_id: str | None = None,
    persona_id: str | None = None,
    limit: int = 20,
) -> list[dict[str, Any]]:
    normalized_query = str(query or "").strip()
    if not normalized_query or int(limit) <= 0:
        return []
    rows = _filtered_sessions(
        ordered_rows, status=status, folder_id=folder_id, persona_id=persona_id
    )
    return _take_rows(_search_results(rows, load_turns, normalized_query), limit)
