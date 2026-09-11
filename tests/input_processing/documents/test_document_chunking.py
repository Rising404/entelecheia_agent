"""结构决定分块边界；词元只负责强制拆分。

被替换的两种形态都会以检索命中可见的方式失败：纯词元聚合会在主题中途截断，
每元素一个分块则会让分块过小，难以检索。
"""

from __future__ import annotations

import pytest

from personagraph.input_processing.documents.chunking import (
    ChunkingProfile,
    chunk_document,
    chunker_fingerprint,
)
from personagraph.input_processing.documents.contracts import (
    DocumentElement,
    DocumentLocator,
    ElementKind,
    ProcessingResult,
    ProcessorFingerprint,
)


FINGERPRINT = ProcessorFingerprint("test", "1")


def _element(text, *, section=(), ordinal=0, page=1, kind=ElementKind.PARAGRAPH):
    return DocumentElement(
        element_id=f"e{ordinal}",
        kind=kind,
        text=text,
        locator=DocumentLocator(page=page, ordinal=ordinal, section_path=section),
    )


def _result(*elements):
    return ProcessingResult(elements=tuple(elements), processor=FINGERPRINT)


def _chunk(*elements, **profile_kwargs):
    profile = ChunkingProfile(**profile_kwargs) if profile_kwargs else ChunkingProfile()
    return chunk_document(_result(*elements), source_key="doc:1", profile=profile)


def test_paragraphs_in_one_section_merge_into_a_retrievable_chunk():
    chunks = _chunk(
        _element("第一段内容在这里。", section=("方案",), ordinal=0),
        _element("第二段继续论述。", section=("方案",), ordinal=1),
    )
    assert len(chunks) == 1
    assert len(chunks[0].element_ids) == 2


def test_a_new_section_always_starts_a_new_chunk():
    """跨标题合并会生成同时讨论两个主题的分块。"""
    chunks = _chunk(
        _element("方案正文。", section=("方案",), ordinal=0),
        _element("风险正文。", section=("方案", "风险"), ordinal=1),
    )
    assert len(chunks) == 2
    assert chunks[0].section_path == ("方案",)
    assert chunks[1].section_path == ("方案", "风险")


def test_a_table_is_never_merged_into_prose():
    chunks = _chunk(
        _element("说明文字。", section=("数据",), ordinal=0),
        _element("列A | 列B\n1 | 2", section=("数据",), ordinal=1, kind=ElementKind.TABLE),
        _element("后续说明。", section=("数据",), ordinal=2),
    )
    kinds = [chunk.kind for chunk in chunks]
    assert ElementKind.TABLE in kinds
    table_chunk = next(c for c in chunks if c.kind is ElementKind.TABLE)
    assert len(table_chunk.element_ids) == 1


def test_the_chunk_carries_a_typed_span_not_a_concatenated_string():
    chunks = _chunk(
        _element("第一页内容。", section=("章",), ordinal=0, page=1),
        _element("第二页内容。", section=("章",), ordinal=1, page=2),
    )
    span = chunks[0].span
    assert span.start.page == 1 and span.end.page == 2
    assert span.pages == (1, 2)
    # 消费者读取字段；该字符串仅用于展示。
    assert "→" in span.describe()


def test_real_shaped_long_section_location_is_bounded_without_silent_collision():
    """Docling 可能把长表格文本提升为重复的标题路径。

    带类型的范围仍保持无损，而旧版展示投影必须适配论文引用边界，并让任何
    省略都能够自证。
    """
    heading = (
        "Then Executor Agent preserve private information well and mention all "
        "the public information well in the meeting summary"
    )
    first = DocumentElement(
        element_id="real-shaped-start",
        kind=ElementKind.PARAGRAPH,
        text="First extracted table cell.",
        locator=DocumentLocator(
            page=12,
            ordinal=237,
            section_path=(heading,),
            char_range=(0, 120),
        ),
    )
    last = DocumentElement(
        element_id="real-shaped-end",
        kind=ElementKind.PARAGRAPH,
        text="Last extracted table cell.",
        locator=DocumentLocator(
            page=12,
            ordinal=242,
            section_path=(heading,),
            char_range=(0, 49),
        ),
    )

    location = _chunk(first, last)[0].loc
    changed_heading = f"{heading[:-8]}abstract"
    changed_first = DocumentElement(
        element_id="real-shaped-start-changed",
        kind=ElementKind.PARAGRAPH,
        text="First extracted table cell.",
        locator=DocumentLocator(
            page=12,
            ordinal=237,
            section_path=(changed_heading,),
            char_range=(0, 120),
        ),
    )
    changed_last = DocumentElement(
        element_id="real-shaped-end-changed",
        kind=ElementKind.PARAGRAPH,
        text="Last extracted table cell.",
        locator=DocumentLocator(
            page=12,
            ordinal=242,
            section_path=(changed_heading,),
            char_range=(0, 49),
        ),
    )
    changed_location = _chunk(changed_first, changed_last)[0].loc

    assert len(location) <= 256
    assert location.startswith("p12#237")
    assert location.endswith("c0-49")
    assert "[loc-sha256:" in location
    assert location == _chunk(first, last)[0].loc
    assert "[loc-sha256:" in changed_location
    assert changed_location != location


def test_the_heading_path_is_prepended_so_a_chunk_describes_itself():
    chunks = _chunk(_element("正文。", section=("方案", "风险"), ordinal=0))
    assert chunks[0].text.startswith("方案 › 风险")

    plain = _chunk(
        _element("正文。", section=("方案",), ordinal=0), include_section_heading=False
    )
    assert not plain[0].text.startswith("方案")


def test_no_chunk_ever_exceeds_the_hard_limit():
    """`22/004` 要求每个输出分块（包括重叠部分）都满足此条件。"""
    long_text = "。".join(f"这是第{i}个句子" for i in range(400))
    chunks = _chunk(
        _element(long_text, section=("长",), ordinal=0),
        target_tokens=100, max_tokens=200,
    )
    assert len(chunks) > 1
    assert all(chunk.token_count <= 200 for chunk in chunks)


def test_an_unsplittable_run_is_still_bounded():
    """超过上限的单个句子不得变成一个巨型分块。"""
    chunks = _chunk(
        _element("字" * 5000, section=("长",), ordinal=0),
        target_tokens=100, max_tokens=200,
    )
    assert chunks and all(chunk.token_count <= 200 for chunk in chunks)


def test_a_split_is_recorded_rather_than_hidden():
    chunks = _chunk(
        _element("。".join(f"句子{i}" for i in range(300)), section=("长",), ordinal=0),
        target_tokens=50, max_tokens=100, split_overlap_tokens=16,
    )
    assert any(chunk.was_split for chunk in chunks)


def test_undersized_neighbours_merge_but_never_across_structure():
    same_section = _chunk(
        _element("短一。", section=("章",), ordinal=0),
        _element("短二。", section=("章",), ordinal=1),
        target_tokens=500, min_tokens=100,
    )
    assert len(same_section) == 1

    across = _chunk(
        _element("短一。", section=("章",), ordinal=0),
        _element("短二。", section=("章", "节"), ordinal=1),
        target_tokens=500, min_tokens=100,
    )
    assert len(across) == 2


def test_images_do_not_become_empty_chunks():
    """它们仍可作为元素寻址；空分块只会污染索引，并可能像真实内容一样被检索。"""
    chunks = _chunk(
        _element("正文。", section=("章",), ordinal=0),
        DocumentElement(
            "img", ElementKind.IMAGE, None,
            DocumentLocator(page=1, ordinal=1, section_path=("章",)), needs_vision=True,
        ),
    )
    assert len(chunks) == 1
    assert all(chunk.text.strip() for chunk in chunks)


def test_chunk_ids_are_content_derived_so_an_edit_does_not_renumber_the_rest():
    first = _chunk(
        _element("段一。", section=("章",), ordinal=0),
        _element("段二。", section=("章", "节"), ordinal=1),
    )
    edited = _chunk(
        _element("段一改了。", section=("章",), ordinal=0),
        _element("段二。", section=("章", "节"), ordinal=1),
    )
    assert first[0].chunk_id != edited[0].chunk_id
    # 未改动的章节保留其 ID，因此可以复用嵌入。
    assert first[1].chunk_id == edited[1].chunk_id


def test_reprocessing_an_unchanged_document_is_stable():
    elements = [
        _element("段一。", section=("章",), ordinal=0),
        _element("段二。", section=("章",), ordinal=1),
    ]
    assert [c.chunk_id for c in _chunk(*elements)] == [c.chunk_id for c in _chunk(*elements)]


def test_repeated_body_in_one_section_has_distinct_producer_chunk_ids():
    chunks = _chunk(
        _element("重复表格", section=("数据",), ordinal=0, page=1, kind=ElementKind.TABLE),
        _element("重复表格", section=("数据",), ordinal=1, page=9, kind=ElementKind.TABLE),
    )

    assert len(chunks) == 2
    assert chunks[0].text == chunks[1].text
    assert chunks[0].chunk_id != chunks[1].chunk_id


def test_appending_a_duplicate_keeps_the_existing_chunks_base_id():
    original = _chunk(
        _element("重复表格", section=("数据",), ordinal=0, page=1, kind=ElementKind.TABLE),
    )
    with_appended_duplicate = _chunk(
        _element("重复表格", section=("数据",), ordinal=0, page=1, kind=ElementKind.TABLE),
        _element("重复表格", section=("数据",), ordinal=1, page=9, kind=ElementKind.TABLE),
    )

    assert with_appended_duplicate[0].chunk_id == original[0].chunk_id
    assert with_appended_duplicate[1].chunk_id != original[0].chunk_id


def test_the_chunker_reports_a_fingerprint_for_invalidation():
    assert chunker_fingerprint().startswith("structure_first@3+")


@pytest.mark.parametrize("kwargs", [
    {"target_tokens": 0},
    {"target_tokens": 200, "max_tokens": 100},
    {"min_tokens": -1},
])
def test_an_incoherent_profile_is_refused(kwargs):
    with pytest.raises(ValueError):
        ChunkingProfile(**kwargs)


class TestTokenizerIdentity:
    """计数器是分块定义的一部分，而不只是实现细节。

    使用真实分词器与启发式方法计数会以不同方式切分同一文档，因此两种结果
    不可互换，也不得共享身份。
    """

    def test_a_named_counter_changes_the_fingerprint(self):
        heuristic = ChunkingProfile()
        real = ChunkingProfile(count_tokens=lambda text: len(text), tokenizer_id="bge_m3:test")

        assert "+heuristic+" in heuristic.fingerprint()
        assert "+bge_m3:test+" in real.fingerprint()
        assert heuristic.fingerprint() != real.fingerprint()

    def test_an_unnamed_counter_is_refused(self):
        """否则两种不兼容的分块会共享同一指纹，导致重新分块永远无法正确限定范围。"""
        with pytest.raises(ValueError, match="tokenizer_id"):
            ChunkingProfile(count_tokens=lambda text: len(text))

    def test_the_counter_actually_drives_the_boundaries(self):
        elements = [
            _element("。".join(f"这是第{i}句话" for i in range(40)), section=("章",), ordinal=0),
        ]
        # 报告双倍计数的计数器会让所有内容的成本翻倍，因此相同文本必须切成更多片段。
        cheap = chunk_document(
            _result(*elements), source_key="k",
            profile=ChunkingProfile(
                target_tokens=60,
                max_tokens=120,
                split_overlap_tokens=16,
            ),
        )
        expensive = chunk_document(
            _result(*elements), source_key="k",
            profile=ChunkingProfile(
                target_tokens=60, max_tokens=120,
                split_overlap_tokens=16,
                count_tokens=lambda text: len(text) * 2, tokenizer_id="double:1",
            ),
        )
        assert len(expensive) > len(cheap)
        assert [c.chunk_id for c in cheap] != [c.chunk_id for c in expensive]

    def test_the_default_fingerprint_still_names_the_fallback(self):
        assert chunker_fingerprint().startswith("structure_first@3+heuristic+")

    def test_overlap_is_part_of_the_identity_too(self):
        """采用不同重叠量切分的两个语料库不可互换。"""
        none = ChunkingProfile(split_overlap_tokens=0)
        some = ChunkingProfile(split_overlap_tokens=32)
        assert none.fingerprint() != some.fingerprint()
        assert none.fingerprint().endswith("+ov0")

    def test_every_output_affecting_profile_field_changes_the_fingerprint(self):
        baseline = ChunkingProfile()
        variants = (
            ChunkingProfile(target_tokens=baseline.target_tokens - 1),
            ChunkingProfile(max_tokens=baseline.max_tokens + 1),
            ChunkingProfile(min_tokens=baseline.min_tokens + 1),
            ChunkingProfile(include_section_heading=not baseline.include_section_heading),
        )

        assert all(variant.fingerprint() != baseline.fingerprint() for variant in variants)

    def test_overlap_that_would_break_the_hard_limit_is_dropped(self):
        """重叠用于辅助检索，而不是绕过边界限制。"""
        long_text = "。".join(f"这是第{i}个句子" for i in range(300))
        chunks = chunk_document(
            _result(_element(long_text, section=("章",), ordinal=0)),
            source_key="k",
            profile=ChunkingProfile(target_tokens=100, max_tokens=140, split_overlap_tokens=32),
        )
        assert len(chunks) > 1
        assert all(chunk.token_count <= 140 for chunk in chunks)

    def test_tokenizer_offset_splits_preserve_every_source_character(self):
        text = "alpha  beta\ngamma   delta epsilon"

        def offsets(value: str):
            import re

            return tuple(match.span() for match in re.finditer(r"\S+", value))

        chunks = chunk_document(
            _result(_element(text, ordinal=0)),
            source_key="k",
            profile=ChunkingProfile(
                target_tokens=2,
                max_tokens=3,
                min_tokens=0,
                include_section_heading=False,
                split_overlap_tokens=0,
                count_tokens=lambda value: len(offsets(value)),
                token_offsets=offsets,
                tokenizer_id="word-offsets@1",
            ),
        )

        assert "".join(chunk.text for chunk in chunks) == text
        assert all(chunk.token_count <= 3 for chunk in chunks)

    def test_overlap_is_applied_to_ordinary_adjacent_chunks_in_one_structure_group(self):
        chunks = chunk_document(
            _result(
                _element("abcdefgh", section=("章",), ordinal=0),
                _element("ijklmnop", section=("章",), ordinal=1),
            ),
            source_key="k",
            profile=ChunkingProfile(
                target_tokens=10,
                max_tokens=14,
                min_tokens=0,
                include_section_heading=False,
                split_overlap_tokens=3,
                count_tokens=len,
                tokenizer_id="character@1",
            ),
        )

        assert [chunk.text for chunk in chunks] == ["abcdefgh", "fgh\nijklmnop"]
        assert all(chunk.token_count <= 14 for chunk in chunks)

    @pytest.mark.parametrize("overlap", [-1, 512])
    def test_an_impossible_overlap_is_refused(self, overlap):
        with pytest.raises(ValueError, match="split_overlap_tokens"):
            ChunkingProfile(target_tokens=256, split_overlap_tokens=overlap)
