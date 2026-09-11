"""结构优先切块器的稳定配置、位置与输出合同。"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
import hashlib

from ..contracts import DocumentLocator, ElementKind


CHUNKER_NAME = "structure_first"
CHUNKER_VERSION = "3"
CHUNK_LOCATION_MAX_CHARS = 256
CHUNK_LOCATION_HASH_LABEL = "loc-sha256"
DEFAULT_TARGET_TOKENS = 450
DEFAULT_MAX_TOKENS = 600
DEFAULT_MIN_TOKENS = 80
DEFAULT_SPLIT_OVERLAP_TOKENS = 64
HEURISTIC_TOKENIZER_ID = "heuristic"


@dataclass(frozen=True)
class ChunkSpan:
    """用共享 locator 词汇表示 chunk 的起止位置。"""

    start: DocumentLocator
    end: DocumentLocator

    @property
    def pages(self) -> tuple[int, ...]:
        first, last = self.start.page, self.end.page
        if first is None or last is None:
            return ()
        return tuple(range(min(first, last), max(first, last) + 1))

    def full_description(self) -> str:
        """渲染未截断的兼容位置投影。"""

        head, tail = self.start.describe(), self.end.describe()
        return head if head == tail else f"{head} → {tail}"

    def describe(self) -> str:
        return bounded_chunk_location(self.full_description())


def bounded_chunk_location(
    value: str,
    *,
    maximum: int = CHUNK_LOCATION_MAX_CHARS,
) -> str:
    """返回不超过 ``maximum``、确定且明显省略的位置。"""

    if not isinstance(value, str):
        raise TypeError("chunk location must be a string")
    if maximum < 96:
        raise ValueError("chunk location maximum cannot fit its digest marker")
    if len(value) <= maximum:
        return value
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    marker = f" … [{CHUNK_LOCATION_HASH_LABEL}:{digest}] … "
    retained = maximum - len(marker)
    left = (retained + 1) // 2
    right = retained - left
    return f"{value[:left]}{marker}{value[-right:]}"


@dataclass(frozen=True)
class DocumentChunk:
    """一个可检索单元及其精确来源绑定。"""

    chunk_id: str
    text: str
    span: ChunkSpan
    section_path: tuple[str, ...]
    element_ids: tuple[str, ...]
    token_count: int
    kind: ElementKind = ElementKind.PARAGRAPH
    was_split: bool = False
    source_pages: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.source_pages and (
            any(
                isinstance(page, bool) or not isinstance(page, int) or page < 1
                for page in self.source_pages
            )
            or self.source_pages != tuple(sorted(set(self.source_pages)))
        ):
            raise ValueError("chunk source_pages must be a sorted exact set")

    @property
    def loc(self) -> str:
        return self.span.describe()


@dataclass(frozen=True)
class ChunkingProfile:
    """约束一个语料的 chunk，并冻结所用 tokenizer 的身份。"""

    target_tokens: int = DEFAULT_TARGET_TOKENS
    max_tokens: int = DEFAULT_MAX_TOKENS
    min_tokens: int = DEFAULT_MIN_TOKENS
    include_section_heading: bool = True
    split_overlap_tokens: int = DEFAULT_SPLIT_OVERLAP_TOKENS
    count_tokens: Callable[[str], int] | None = field(default=None, compare=False)
    token_offsets: Callable[[str], Sequence[tuple[int, int]]] | None = field(
        default=None,
        compare=False,
    )
    tokenizer_id: str = HEURISTIC_TOKENIZER_ID

    def __post_init__(self) -> None:
        if not 0 < self.target_tokens <= self.max_tokens:
            raise ValueError("target_tokens must be positive and within max_tokens")
        if self.min_tokens < 0:
            raise ValueError("min_tokens must not be negative")
        if not self.tokenizer_id.strip():
            raise ValueError("tokenizer_id must not be empty")
        if not 0 <= self.split_overlap_tokens < self.target_tokens:
            raise ValueError("split_overlap_tokens must fit inside target_tokens")
        if self.count_tokens is not None and self.tokenizer_id == HEURISTIC_TOKENIZER_ID:
            raise ValueError("a supplied token counter must be named by tokenizer_id")
        if self.token_offsets is not None and self.count_tokens is None:
            raise ValueError("token_offsets requires a supplied token counter")

    def tokens_of(self, text: str) -> int:
        if self.count_tokens is not None:
            return self.count_tokens(text)
        from ....context_budget.token_counter import estimate_tokens

        return estimate_tokens(text)

    def fingerprint(self) -> str:
        return (
            f"{CHUNKER_NAME}@{CHUNKER_VERSION}"
            f"+{self.tokenizer_id}"
            f"+target{self.target_tokens}"
            f"+max{self.max_tokens}"
            f"+min{self.min_tokens}"
            f"+heading{int(self.include_section_heading)}"
            f"+ov{self.split_overlap_tokens}"
        )


__all__ = [
    "CHUNKER_NAME",
    "CHUNKER_VERSION",
    "CHUNK_LOCATION_HASH_LABEL",
    "CHUNK_LOCATION_MAX_CHARS",
    "ChunkSpan",
    "ChunkingProfile",
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_MIN_TOKENS",
    "DEFAULT_SPLIT_OVERLAP_TOKENS",
    "DEFAULT_TARGET_TOKENS",
    "DocumentChunk",
    "HEURISTIC_TOKENIZER_ID",
    "bounded_chunk_location",
]
