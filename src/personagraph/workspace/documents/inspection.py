"""当前文档的分块位置和原始解析文本查询，不执行解析或相关性检索。"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
import hashlib
import json
import re

from personagraph.workspace.storage.context import current

from .reading import CurrentFileDocument, _current_file_document
from .storage import chunks as repository


@dataclass(frozen=True, slots=True)
class ChunkLocation:
    chunk_id: str
    sequence: int
    locator: str
    source_pages: tuple[int, ...]
    element_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SourceTextElement:
    element_id: str
    content: str
    locator: str
    source_pages: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class FileDocumentInspection:
    document: CurrentFileDocument
    chunks: tuple[ChunkLocation, ...]
    source_elements: tuple[SourceTextElement, ...] | None


def inspect_current_file_document(
    *, file_id: str, file_version_id: str, document_version_id: str, session_id: str,
    include_source_text: bool = True,
) -> FileDocumentInspection | None:
    """一次只读事务读取精确版本。正文来源是原始元素，不能从重叠块猜测还原。"""
    database = current()
    if database is None:
        return None
    with database.connect_readonly() as connection:
        connection.execute("BEGIN")
        document = _current_file_document(connection, file_id, file_version_id, session_id)
        if document is None or document.document_version_id != document_version_id:
            return None
        rows, source = repository.get_file_inspection_rows(
            connection, document_id=document.document_id,
            document_version_id=document_version_id,
            include_source_text=include_source_text,
        )
    if len(rows) != document.total_chunk_count:
        raise ValueError("document chunk inventory is incomplete")
    locations = []
    for sequence, row in enumerate(rows):
        metadata = json.loads(row["metadata_json"] or "{}")
        if row["seq"] != sequence or not isinstance(metadata, dict):
            raise ValueError("document chunk sequence is not contiguous")
        element_ids = metadata.get("element_ids", [])
        if not isinstance(element_ids, list) or any(
            not isinstance(value, str) or not value for value in element_ids
        ) or len(element_ids) != len(set(element_ids)):
            raise ValueError("invalid chunk source element identities")
        locations.append(ChunkLocation(
            chunk_id=str(row["id"]), sequence=sequence, locator=str(row["loc"] or ""),
            source_pages=_pages(json.loads(row["source_pages_json"] or "[]")),
            element_ids=tuple(element_ids),
        ))
    return FileDocumentInspection(document, tuple(locations), _source_elements(source))


def _pages(value) -> tuple[int, ...]:
    if not isinstance(value, list) or any(
        isinstance(page, bool) or not isinstance(page, int) or page < 1 for page in value
    ) or value != sorted(set(value)):
        raise ValueError("invalid source page identities")
    return tuple(value)


def _source_elements(row) -> tuple[SourceTextElement, ...] | None:
    if row is None or (row["source_elements_json"], row["source_elements_sha256"]) == (None, None):
        return None
    raw, expected = row["source_elements_json"], row["source_elements_sha256"]
    if not isinstance(raw, str) or hashlib.sha256(raw.encode()).hexdigest() != expected:
        raise ValueError("source text snapshot hash mismatch")
    payload = json.loads(raw)
    if not isinstance(payload, list):
        raise ValueError("source text snapshot is not an element list")
    elements, identities = [], set()
    for item in payload:
        if not isinstance(item, dict) or set(item) != {"element_id", "content", "locator", "source_pages"}:
            raise ValueError("invalid source text element")
        if any(not isinstance(item[key], str) for key in ("element_id", "content", "locator")):
            raise ValueError("invalid source text fields")
        if not item["element_id"] or item["element_id"] in identities:
            raise ValueError("source text element identities are not unique")
        identities.add(item["element_id"])
        elements.append(SourceTextElement(
            item["element_id"], item["content"], item["locator"], _pages(item["source_pages"]),
        ))
    return tuple(elements)


def find_literal_text(
    snapshot: FileDocumentInspection, *, text: str, case_sensitive: bool,
    whitespace: str, match_offset: int, limit: int,
) -> dict:
    """完整扫描原始元素一次；返回窗口限制只影响展示，不影响总匹配数。

    元素按解析顺序以换行连接。重复章节标题和检索块 overlap 不参与输入，因此不用
    按内容去重（文档不同位置真正重复的文字必须分别计数）。匹配之间允许重叠。
    """
    if not text or whitespace not in {"exact", "collapse"}:
        raise ValueError("invalid literal matching options")
    elements = snapshot.source_elements
    if elements is None:
        return {"scan_complete": False, "reason_code": "source_text_unavailable",
                "total_match_count": None, "matches": [], "next_match_offset": None,
                "matches_truncated": False, "scanned_element_count": 0}
    starts, position = [], 0
    for element in elements:
        starts.append(position)
        position += len(element.content) + 1
    original = "\n".join(element.content for element in elements)
    haystack, offsets = _normalized_with_offsets(original, case_sensitive, whitespace)
    needle, _ = _normalized_with_offsets(text, case_sensitive, whitespace)
    if not needle:
        raise ValueError("matching text becomes empty")
    matches, total, cursor = [], 0, 0
    while (found := haystack.find(needle, cursor)) >= 0:
        start, end = offsets[found][0], offsets[found + len(needle) - 1][1]
        # casefold 可以把一个字符扩展成多个字符；部分扩展不是原文的一次字面匹配。
        complete_boundaries = (
            (found == 0 or offsets[found - 1] != offsets[found])
            and (found + len(needle) == len(offsets) or offsets[found + len(needle)] != offsets[found + len(needle) - 1])
        )
        if complete_boundaries:
            if match_offset <= total < match_offset + limit:
                first = max(0, bisect_right(starts, start) - 1)
                last = max(first, bisect_right(starts, max(start, end - 1)) - 1)
                touched = elements[first:last + 1]
                element_ids = {element.element_id for element in touched}
                matches.append({
                    "match_index": total, "text": original[start:end],
                    "start_element_index": first, "end_element_index": last,
                    "start_character": min(start - starts[first], len(elements[first].content)),
                    "end_character": min(end - starts[last], len(elements[last].content)),
                    "element_ids": [element.element_id for element in touched],
                    "source_pages": sorted({page for element in touched for page in element.source_pages}),
                    "locators": [element.locator for element in touched],
                    "chunk_sequences": [chunk.sequence for chunk in snapshot.chunks
                                        if element_ids.intersection(chunk.element_ids)],
                })
            total += 1
        cursor = found + 1
    next_offset = match_offset + len(matches)
    return {"scan_complete": True, "reason_code": None, "total_match_count": total,
            "matches": matches, "next_match_offset": next_offset if next_offset < total else None,
            "matches_truncated": match_offset > 0 or next_offset < total,
            "scanned_element_count": len(elements)}


def _normalized_with_offsets(text: str, case_sensitive: bool, whitespace: str):
    characters, offsets = [], []
    spans = re.finditer(r"\s+|[^\s]", text) if whitespace == "collapse" else re.finditer(r"[\s\S]", text)
    for span in spans:
        value = " " if whitespace == "collapse" and span.group().isspace() else span.group()
        value = value if case_sensitive else value.casefold()
        characters.append(value)
        offsets.extend([(span.start(), span.end())] * len(value))
    return "".join(characters), offsets
