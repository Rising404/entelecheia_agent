"""对格式中立元素执行结构优先切块。

在采用当前方案前，曾否定两种形状：

*纯 token 聚合*——旧 chunker 的做法——会在预算耗尽处切断，因此 chunk 经常在一个主题中途
结束，又从另一个主题开始。retrieval 命中后便无法说明其主题。

*每个元素一个 chunk* 则是相反的失败：元素本身是段落和表格行，生成的 chunk 太小且太多，
无法有效 retrieval。

这里的规则是：**结构决定边界，token 只在必要时强制拆分**。不同 section 绝不合并，表格
绝不并入正文；只有结构单元确实过大时，token 预算才会介入——且拆分会被明确报告，而不是
静默发生。

本模块处理 ``ProcessingResult``，而不是某个解析器自己的文档模型，因此原生与 Docling
引擎通过相同代码和相同身份规则生成 chunk。
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from dataclasses import replace

from ..contracts import DocumentElement, DocumentLocator, ElementKind, ProcessingResult
from .contracts import ChunkSpan, ChunkingProfile, DocumentChunk

_SENTENCE_BOUNDARY = re.compile(r"(?<=[。．.!?！？;；\n])\s*")

def chunker_fingerprint(profile: ChunkingProfile | None = None) -> str:
    """切块配置指纹；默认为启发式配置。"""

    return (profile or ChunkingProfile()).fingerprint()


def chunk_document(
    result: ProcessingResult,
    *,
    source_key: str,
    profile: ChunkingProfile | None = None,
) -> tuple[DocumentChunk, ...]:
    """将一个已解析文档转换为可 retrieval 的 chunk。"""

    active = profile or ChunkingProfile()
    chunks: list[DocumentChunk] = []
    for group in _structural_groups(result.elements):
        chunks.extend(_chunk_group(group, source_key=source_key, profile=active))
    return tuple(_disambiguate_repeated_chunk_ids(chunks))


def _structural_groups(
    elements: Sequence[DocumentElement],
) -> list[list[DocumentElement]]:
    """在结构表明单元结束的位置拆分元素流。

    新 section 或表格会开始新组；其他内容则持续累积，因此同一 heading 下的段落仍可彼此
    合并，但不会与其他内容合并。
    """

    groups: list[list[DocumentElement]] = []
    current: list[DocumentElement] = []
    current_section: tuple[str, ...] | None = None

    for element in elements:
        if element.kind is ElementKind.IMAGE or not (element.text or "").strip():
            # 图像不携带可 retrieval 文本。它们通过元素层保持可寻址，而不会变成空 chunk。
            continue
        if element.kind is ElementKind.TABLE:
            if current:
                groups.append(current)
                current = []
            groups.append([element])
            current_section = None
            continue
        if current_section is not None and element.locator.section_path != current_section:
            groups.append(current)
            current = []
        current_section = element.locator.section_path
        current.append(element)

    if current:
        groups.append(current)
    return groups


def _chunk_group(
    group: list[DocumentElement],
    *,
    source_key: str,
    profile: ChunkingProfile,
) -> list[DocumentChunk]:
    chunks: list[DocumentChunk] = []
    buffer: list[DocumentElement] = []
    buffer_tokens = 0

    def flush() -> None:
        nonlocal buffer, buffer_tokens
        if buffer:
            chunks.append(_build_chunk(buffer, source_key=source_key, profile=profile))
            buffer, buffer_tokens = [], 0

    overhead = _heading_overhead(
        group[0].locator.section_path if group else (), profile
    )
    body_target = max(1, profile.target_tokens - overhead)
    body_max = max(1, profile.max_tokens - overhead)

    for element in group:
        text = element.text or ""
        tokens = profile.tokens_of(text)

        if tokens > body_max:
            # 单个元素完全无法容纳。先 flush 可避免超大元素的分片吸收无关相邻内容。
            flush()
            chunks.extend(_split_element(element, source_key=source_key, profile=profile))
            continue

        if buffer and buffer_tokens + tokens > body_target:
            flush()
        buffer.append(element)
        buffer_tokens += tokens

    flush()
    merged = _merge_undersized(chunks, profile=profile, source_key=source_key)
    return _apply_group_overlap(merged, profile=profile, source_key=source_key)


def _build_chunk(
    elements: list[DocumentElement],
    *,
    source_key: str,
    profile: ChunkingProfile,
    was_split: bool = False,
    text_override: str | None = None,
) -> DocumentChunk:
    section = elements[0].locator.section_path
    body = text_override if text_override is not None else "\n".join(
        (element.text or "") for element in elements
    )
    text = _contextualize(body, section, profile)
    start, end = elements[0].locator, elements[-1].locator
    return DocumentChunk(
        chunk_id=_chunk_id(source_key, section, body),
        text=text,
        span=ChunkSpan(start=start, end=end),
        section_path=section,
        element_ids=tuple(element.element_id for element in elements),
        token_count=profile.tokens_of(text),
        kind=elements[0].kind if len(elements) == 1 else ElementKind.PARAGRAPH,
        was_split=was_split,
        source_pages=tuple(sorted({
            page
            for element in elements
            for page in element.source_pages
        })),
    )


def _contextualize(
    body: str, section: tuple[str, ...], profile: ChunkingProfile
) -> str:
    """前置 heading path，使 retrieval 到的 chunk 能够自描述。"""

    if not profile.include_section_heading or not section:
        return body
    heading = " › ".join(section)
    return body if body.startswith(heading) else f"{heading}\n{body}"


def _heading_overhead(section: tuple[str, ...], profile: ChunkingProfile) -> int:
    """heading 前缀会为本 section 每个 chunk 增加的 token 数。

    硬限制作用于 chunk 的最终内容，因此必须在拆分前预留前缀。若只为正文分配预算并事后
    添加 heading，所谓“有界”chunk 最终就会超出边界。
    """

    if not profile.include_section_heading or not section:
        return 0
    return profile.tokens_of(f"{' › '.join(section)}\n")


def _split_element(
    element: DocumentElement,
    *,
    source_key: str,
    profile: ChunkingProfile,
) -> list[DocumentChunk]:
    """尽可能在句子边界拆分一个超大元素。

    拆分是最后手段，且会记录在 chunk 上而非隐藏：消费者把引用与源对照时，需要知道当前
    查看的是元素的一部分。
    """

    overhead = _heading_overhead(element.locator.section_path, profile)
    body_target = max(1, profile.target_tokens - overhead)
    body_max = max(1, profile.max_tokens - overhead)

    pieces: list[str] = []
    current = ""
    for sentence in _SENTENCE_BOUNDARY.split(element.text or ""):
        if not sentence:
            continue
        candidate = f"{current}{sentence}" if current else sentence
        if current and profile.tokens_of(candidate) > body_target:
            pieces.append(current)
            current = sentence
        else:
            current = candidate
    if current:
        pieces.append(current)

    # 超过硬限制的单个句子仍须有界。真实 tokenizer profile 直接依据原文 offset 切分；
    # 启发式测试/降级 profile 才使用保守的字符比例回退。
    bounded: list[str] = []
    for piece in pieces:
        if profile.tokens_of(piece) > body_max:
            bounded.extend(
                _split_text_to_token_budget(
                    piece,
                    budget_tokens=min(body_target, body_max),
                    profile=profile,
                )
            )
        elif piece:
            bounded.append(piece)

    return [
        _build_chunk(
            [element], source_key=source_key, profile=profile,
            was_split=True, text_override=piece,
        )
        for piece in bounded
    ]


def _split_text_to_token_budget(
    text: str,
    *,
    budget_tokens: int,
    profile: ChunkingProfile,
) -> list[str]:
    if not text or budget_tokens <= 0:
        return []
    offsets = _validated_token_offsets(text, profile)
    if offsets is not None:
        if not offsets:
            return [text]
        pieces: list[str] = []
        start_token = 0
        start_char = 0
        while start_token < len(offsets):
            end_token = min(len(offsets), start_token + budget_tokens)
            end_char = (
                len(text)
                if end_token == len(offsets)
                else offsets[end_token][0]
            )
            piece = text[start_char:end_char]
            if piece:
                pieces.append(piece)
            start_char = end_char
            start_token = end_token
        return pieces

    pieces = []
    remainder = text
    while profile.tokens_of(remainder) > budget_tokens:
        ratio = budget_tokens / max(1, profile.tokens_of(remainder))
        cut = max(1, int(len(remainder) * ratio))
        candidate = remainder[:cut]
        while len(candidate) > 1 and profile.tokens_of(candidate) > budget_tokens:
            candidate = candidate[: max(1, len(candidate) - max(1, len(candidate) // 10))]
        pieces.append(candidate)
        remainder = remainder[len(candidate):]
    if remainder:
        pieces.append(remainder)
    return pieces


def _apply_group_overlap(
    chunks: list[DocumentChunk],
    *,
    profile: ChunkingProfile,
    source_key: str,
) -> list[DocumentChunk]:
    """为同一结构组内相邻检索块携带有界的前文尾部。"""

    if profile.split_overlap_tokens <= 0 or len(chunks) < 2:
        return chunks
    result = [chunks[0]]
    for previous, current in zip(chunks, chunks[1:]):
        available = profile.max_tokens - current.token_count
        overlap_budget = min(profile.split_overlap_tokens, max(0, available))
        tail = _tail_within_tokens(
            _strip_heading(previous),
            overlap_budget,
            profile,
        )
        if not tail:
            result.append(current)
            continue
        current_body = _strip_heading(current)
        body = f"{tail}\n{current_body}"
        text = _contextualize(body, current.section_path, profile)
        while tail and profile.tokens_of(text) > profile.max_tokens:
            overlap_budget -= 1
            tail = _tail_within_tokens(
                _strip_heading(previous),
                overlap_budget,
                profile,
            )
            body = f"{tail}\n{current_body}" if tail else current_body
            text = _contextualize(body, current.section_path, profile)
        if not tail:
            result.append(current)
            continue
        result.append(
            replace(
                current,
                chunk_id=_chunk_id(source_key, current.section_path, body),
                text=text,
                span=ChunkSpan(start=previous.span.start, end=current.span.end),
                element_ids=tuple(dict.fromkeys(previous.element_ids + current.element_ids)),
                token_count=profile.tokens_of(text),
                source_pages=tuple(
                    sorted(set(previous.source_pages) | set(current.source_pages))
                ),
            )
        )
    return result


def _tail_within_tokens(text: str, budget_tokens: int, profile: ChunkingProfile) -> str:
    """截取尾部片段，并优先选择句子边界。"""

    if budget_tokens <= 0 or not text:
        return ""
    offsets = _validated_token_offsets(text, profile)
    if offsets is not None:
        if len(offsets) <= budget_tokens:
            return text
        return text[offsets[-budget_tokens][0]:]
    ratio = budget_tokens / max(1, profile.tokens_of(text))
    cut = max(1, int(len(text) * (1 - min(1.0, ratio))))
    tail = text[cut:]
    boundary = _SENTENCE_BOUNDARY.search(tail)
    if boundary is not None and boundary.end() < len(tail) // 2:
        tail = tail[boundary.end():]
    while tail and profile.tokens_of(tail) > budget_tokens:
        tail = tail[len(tail) // 10 or 1:]
    return tail


def _validated_token_offsets(
    text: str,
    profile: ChunkingProfile,
) -> tuple[tuple[int, int], ...] | None:
    if profile.token_offsets is None:
        return None
    offsets = tuple((int(start), int(end)) for start, end in profile.token_offsets(text))
    previous_start = -1
    for start, end in offsets:
        if start < previous_start or start < 0 or end <= start or end > len(text):
            raise ValueError("token_offsets returned invalid source spans")
        previous_start = start
    if len(offsets) != profile.tokens_of(text):
        raise ValueError("token_offsets and token counter disagree")
    return offsets


def _merge_undersized(
    chunks: list[DocumentChunk],
    *,
    profile: ChunkingProfile,
    source_key: str,
) -> list[DocumentChunk]:
    """当合并后仍低于 target 时，把过小 chunk 并入相邻 chunk。

    仅在组内执行，因此绝不会重新合并被结构边界分开的内容。
    """

    if len(chunks) < 2 or profile.min_tokens <= 0:
        return chunks
    merged: list[DocumentChunk] = []
    for chunk in chunks:
        if (
            merged
            and not chunk.was_split
            and not merged[-1].was_split
            and chunk.kind is not ElementKind.TABLE
            and merged[-1].kind is not ElementKind.TABLE
            and min(chunk.token_count, merged[-1].token_count) < profile.min_tokens
            and merged[-1].token_count + chunk.token_count <= profile.target_tokens
        ):
            merged[-1] = _combine(merged[-1], chunk, profile=profile, source_key=source_key)
            continue
        merged.append(chunk)
    return merged


def _combine(
    left: DocumentChunk,
    right: DocumentChunk,
    *,
    profile: ChunkingProfile,
    source_key: str,
) -> DocumentChunk:
    body = f"{left.text}\n{_strip_heading(right)}"
    return DocumentChunk(
        chunk_id=_chunk_id(source_key, left.section_path, body),
        text=body,
        span=ChunkSpan(start=left.span.start, end=right.span.end),
        section_path=left.section_path,
        element_ids=left.element_ids + right.element_ids,
        token_count=profile.tokens_of(body),
        kind=ElementKind.PARAGRAPH,
        source_pages=tuple(sorted(set(left.source_pages) | set(right.source_pages))),
    )


def _strip_heading(chunk: DocumentChunk) -> str:
    heading = " › ".join(chunk.section_path)
    if heading and chunk.text.startswith(f"{heading}\n"):
        return chunk.text[len(heading) + 1:]
    return chunk.text


def _chunk_id(source_key: str, section: tuple[str, ...], body: str) -> str:
    """根据源、结构和内容派生稳定 id。

    它不是序号：编辑一个段落会使其后每个 chunk 重新编号，从而丢弃未变化文本的 embedding。
    对常见的唯一内容场景，位置被刻意排除，因此页面或字符范围位移不会重命名其他方面未变化
    的 chunk。
    """

    digest = hashlib.sha256()
    digest.update(source_key.encode("utf-8"))
    digest.update(b"\x1f")
    digest.update("\x1f".join(section).encode("utf-8"))
    digest.update(b"\x1f")
    digest.update(body.encode("utf-8"))
    return f"ch_{digest.hexdigest()[:24]}"


def _disambiguate_repeated_chunk_ids(
    chunks: Sequence[DocumentChunk],
) -> list[DocumentChunk]:
    """保留首个基础 ID，只为后续同内容 chunk 添加后缀。

    上一 generation 中唯一的 chunk，不能仅因追加了相同表格或拆分片段就被重命名。因此首个
    出现项保留内容派生基础 ID；只有后续出现项会添加类型化端点和其在组内的确定性序次。
    """

    occurrences: dict[str, int] = {}
    result: list[DocumentChunk] = []
    for chunk in chunks:
        occurrence = occurrences.get(chunk.chunk_id, 0)
        occurrences[chunk.chunk_id] = occurrence + 1
        if occurrence == 0:
            result.append(chunk)
            continue
        digest = hashlib.sha256()
        digest.update(chunk.chunk_id.encode("utf-8"))
        digest.update(b"\x1f")
        digest.update(repr((
            _locator_identity(chunk.span.start),
            _locator_identity(chunk.span.end),
            occurrence,
        )).encode("utf-8"))
        result.append(replace(chunk, chunk_id=f"ch_{digest.hexdigest()[:24]}"))
    return result


def _locator_identity(locator: DocumentLocator) -> tuple[object, ...]:
    return (
        locator.page,
        locator.ordinal,
        locator.section_path,
        locator.bbox,
        locator.char_range,
    )
