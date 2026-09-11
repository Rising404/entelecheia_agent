"""一个 Attempt 提示词可见且不含内容的论文范围契约。

这些值描述持久论文资源快照，但不导入之后使用它的工具执行、检索或工作区边界实现。
它们特意只包含别名、有界元数据和世代指纹；私有文档标识仍仅供 Host 使用。
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any

from ..task_graph.paper_resource_contracts import (
    PAPER_RESOURCE_SNAPSHOT_ID_PREFIX,
    PaperOutlineEntry,
    PaperResourceSnapshot,
)


MAX_BOUND_DOCUMENTS = 2
_SHA256 = re.compile(r"^[a-f0-9]{64}$")
_KNOWN_PROCESSING_CODES = frozenset(
    {"fallback_reader_used", "page_empty", "page_needs_vision", "parser_partial"}
)


def _bounded_printable(
    value: str,
    field_name: str,
    *,
    maximum: int,
    allow_empty: bool = True,
) -> None:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    if (not allow_empty and not value.strip()) or len(value) > maximum:
        raise ValueError(f"{field_name} is empty or exceeds its bounded length")
    if any(ord(character) < 32 for character in value):
        raise ValueError(f"{field_name} must not contain control characters")


@dataclass(frozen=True, slots=True)
class PaperOutlineEntryBoundary:
    """一个持久且提示词安全之大纲条目的无损 Runtime 投影。"""

    outline_key: str
    title: str
    start_handle: str
    page_start: int | None = None
    page_end: int | None = None

    def __post_init__(self) -> None:
        if not re.fullmatch(r"P[12]:S[1-9][0-9]*", self.outline_key):
            raise ValueError("paper outline key is invalid")
        if not re.fullmatch(r"P[12]:C[1-9][0-9]*", self.start_handle):
            raise ValueError("paper outline start handle is invalid")
        if self.outline_key[:2] != self.start_handle[:2]:
            raise ValueError("paper outline key and handle must share an alias")
        _bounded_printable(self.title, "paper outline title", maximum=500, allow_empty=False)
        if (self.page_start is None) != (self.page_end is None):
            raise ValueError("paper outline page range must provide both endpoints")
        if (
            self.page_start is not None
            and (
                isinstance(self.page_start, bool)
                or not isinstance(self.page_start, int)
                or isinstance(self.page_end, bool)
                or not isinstance(self.page_end, int)
                or self.page_start <= 0
                or self.page_end < self.page_start
            )
        ):
            raise ValueError("paper outline page range is invalid")

    @classmethod
    def from_resource_entry(
        cls,
        value: PaperOutlineEntry,
    ) -> 'PaperOutlineEntryBoundary':
        return cls(
            outline_key=value.outline_key,
            title=value.title,
            start_handle=value.start_handle,
            page_start=value.page_start,
            page_end=value.page_end,
        )

    def to_resource_entry(self) -> PaperOutlineEntry:
        return PaperOutlineEntry(
            outline_key=self.outline_key,
            title=self.title,
            start_handle=self.start_handle,
            page_start=self.page_start,
            page_end=self.page_end,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "outline_key": self.outline_key,
            "title": self.title,
            "start_handle": self.start_handle,
            "page_start": self.page_start,
            "page_end": self.page_end,
        }


@dataclass(frozen=True, slots=True)
class PaperAttemptGeneration:
    data_version_id: str
    fingerprint: str
    encoder_fingerprint: str

    def __post_init__(self) -> None:
        _bounded_printable(self.data_version_id, "generation ID", maximum=256, allow_empty=False)
        _bounded_printable(self.fingerprint, "generation fingerprint", maximum=256, allow_empty=False)
        _bounded_printable(
            self.encoder_fingerprint,
            "encoder fingerprint",
            maximum=1_024,
            allow_empty=False,
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "data_version_id": self.data_version_id,
            "fingerprint": self.fingerprint,
            "encoder_fingerprint": self.encoder_fingerprint,
        }


@dataclass(frozen=True, slots=True)
class PaperAttemptPaper:
    alias: str
    title: str
    page_start: int | None
    page_end: int | None
    processing_status: str
    diagnostic_codes: tuple[str, ...] = ()
    outline: tuple[PaperOutlineEntryBoundary, ...] = ()
    outline_truncated: bool = False

    def __post_init__(self) -> None:
        if not re.fullmatch(r"P[12]", self.alias):
            raise ValueError("paper alias must be P1 or P2")
        _bounded_printable(self.title, "paper title", maximum=1_024)
        if (self.page_start is None) != (self.page_end is None):
            raise ValueError("paper prompt page range must provide both endpoints")
        if self.page_start is not None and (
            self.page_start <= 0
            or self.page_end is None
            or self.page_end < self.page_start
        ):
            raise ValueError("paper prompt page range is invalid")
        if self.processing_status not in {"complete", "partial"}:
            raise ValueError("paper prompt processing status is invalid")
        diagnostic_codes = tuple(sorted(set(self.diagnostic_codes)))
        if (
            len(diagnostic_codes) > 32
            or any(code not in _KNOWN_PROCESSING_CODES for code in diagnostic_codes)
            or len(self.outline) > 64
        ):
            raise ValueError("paper prompt context exceeds its bounded shape")
        if not isinstance(self.outline_truncated, bool):
            raise ValueError("paper prompt outline_truncated must be a bool")
        object.__setattr__(self, "diagnostic_codes", diagnostic_codes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "alias": self.alias,
            "title": self.title,
            "page_range": (
                {"start": self.page_start, "end": self.page_end}
                if self.page_start is not None
                else None
            ),
            "processing_status": self.processing_status,
            "diagnostic_codes": list(self.diagnostic_codes),
            "outline": [item.to_dict() for item in self.outline],
            "outline_truncated": self.outline_truncated,
        }


@dataclass(frozen=True, slots=True)
class PaperAttemptContext:
    """供一个 Attempt 提示词使用、不含内容且有界的论文能力摘要。"""

    session_id: str
    task_id: str
    snapshot_id: str
    manifest_sha256: str
    generation: PaperAttemptGeneration
    papers: tuple[PaperAttemptPaper, ...]
    schema_version: str = field(default="paper-attempt-context-v1", init=False)

    def __post_init__(self) -> None:
        _bounded_printable(self.session_id, "Attempt Session ID", maximum=256, allow_empty=False)
        _bounded_printable(self.task_id, "Attempt Task ID", maximum=256, allow_empty=False)
        if self.snapshot_id != f"{PAPER_RESOURCE_SNAPSHOT_ID_PREFIX}{self.manifest_sha256}":
            raise ValueError("Attempt resource snapshot ID does not match its hash")
        if not _SHA256.fullmatch(self.manifest_sha256):
            raise ValueError("Attempt paper snapshot hash must be SHA-256")
        if not 1 <= len(self.papers) <= MAX_BOUND_DOCUMENTS:
            raise ValueError("Attempt paper context requires one or two papers")
        aliases = tuple(paper.alias for paper in self.papers)
        if aliases not in {("P1",), ("P2",), ("P1", "P2")}:
            raise ValueError("Attempt paper aliases must be an ordered P1/P2 subset")

    @classmethod
    def from_snapshot(
        cls,
        snapshot: PaperResourceSnapshot,
    ) -> 'PaperAttemptContext':
        if not isinstance(snapshot, PaperResourceSnapshot):
            raise TypeError("snapshot must be a PaperResourceSnapshot")
        return cls(
            session_id=snapshot.session_id,
            task_id=snapshot.task_id,
            snapshot_id=snapshot.snapshot_id,
            manifest_sha256=snapshot.manifest_sha256,
            generation=PaperAttemptGeneration(
                data_version_id=snapshot.retrieval_data_version_id,
                fingerprint=snapshot.retrieval_generation_fingerprint,
                encoder_fingerprint=snapshot.encoder_fingerprint,
            ),
            papers=tuple(
                PaperAttemptPaper(
                    alias=document.paper_key,
                    title=document.title,
                    page_start=document.admitted_text_page_start,
                    page_end=document.admitted_text_page_end,
                    processing_status=document.processing_status,
                    diagnostic_codes=document.processing_diagnostic_codes,
                    outline=tuple(
                        PaperOutlineEntryBoundary.from_resource_entry(item)
                        for item in document.outline
                    ),
                    outline_truncated=document.outline_truncated,
                )
                for document in snapshot.documents
            ),
        )

    def for_papers(self, *paper_keys: str) -> 'PaperAttemptContext':
        """投影节点持有的提示词视图，但不改变快照事实。"""

        requested = tuple(paper_keys)
        if requested not in {("P1",), ("P2",), ("P1", "P2")}:
            raise ValueError("paper prompt scope must be P1, P2, or P1/P2")
        by_alias = {paper.alias: paper for paper in self.papers}
        if any(alias not in by_alias for alias in requested):
            raise ValueError("paper prompt scope exceeds the frozen Task snapshot")
        return PaperAttemptContext(
            session_id=self.session_id,
            task_id=self.task_id,
            snapshot_id=self.snapshot_id,
            manifest_sha256=self.manifest_sha256,
            generation=self.generation,
            papers=tuple(by_alias[alias] for alias in requested),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "snapshot_id": self.snapshot_id,
            "manifest_sha256": self.manifest_sha256,
            "generation": self.generation.to_dict(),
            "papers": [paper.to_dict() for paper in self.papers],
        }


__all__ = [
    "MAX_BOUND_DOCUMENTS",
    'PaperAttemptContext',
    'PaperAttemptGeneration',
    'PaperAttemptPaper',
    'PaperOutlineEntryBoundary',
]
