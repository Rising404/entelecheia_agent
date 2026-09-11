from __future__ import annotations

import pytest

from personagraph.input_processing.documents.contracts import (
    DiagnosticCode,
    DocumentElement,
    DocumentLocator,
    ElementKind,
    ProcessingAdmissionStatus,
    ProcessingDiagnostic,
    ProcessingResult,
    ProcessorFingerprint,
    make_element_id,
)


FINGERPRINT = ProcessorFingerprint("test", "1")


def _text_element() -> DocumentElement:
    return DocumentElement(
        element_id="text-1",
        kind=ElementKind.PARAGRAPH,
        text="可检索正文",
        locator=DocumentLocator(page=1, ordinal=0),
    )


def test_a_locator_renders_only_the_fields_its_format_actually_has():
    """不同格式表达位置的能力不同；共享词汇可防止这种差异泄漏到所有消费者。"""
    assert DocumentLocator(page=3, ordinal=2).describe() == "p3#2"
    assert DocumentLocator(section_path=("方案", "风险")).describe() == "方案 › 风险"
    assert DocumentLocator(char_range=(0, 40)).describe() == "c0-40"
    assert DocumentLocator(ordinal=7).describe() == "#7"
    assert DocumentLocator().describe() == "?"


@pytest.mark.parametrize("kwargs", [
    {"page": 0},
    {"ordinal": -1},
    {"bbox": (10.0, 0.0, 5.0, 20.0)},
    {"char_range": (10, 5)},
])
def test_incoherent_locators_are_refused_at_construction(kwargs):
    with pytest.raises(ValueError):
        DocumentLocator(**kwargs)


def test_a_text_element_cannot_be_empty():
    """空元素进入提示后与缺失元素无法区分，扫描页正会由此变成臆造内容。"""
    with pytest.raises(ValueError):
        DocumentElement(
            element_id="e", kind=ElementKind.PARAGRAPH, text="   ",
            locator=DocumentLocator(ordinal=0),
        )


def test_an_image_element_may_carry_no_text():
    element = DocumentElement(
        element_id="e", kind=ElementKind.IMAGE, text=None,
        locator=DocumentLocator(page=1), needs_vision=True,
    )
    assert element.needs_vision is True


def test_element_ids_change_with_content_and_position_but_not_with_reruns():
    locator = DocumentLocator(page=1, ordinal=0)
    same = make_element_id("src", locator, "hello")
    assert same == make_element_id("src", locator, "hello")
    assert same != make_element_id("src", locator, "hello!")
    assert same != make_element_id("src", DocumentLocator(page=2, ordinal=0), "hello")
    assert same != make_element_id("other", locator, "hello")


def test_a_result_with_any_diagnostic_is_not_complete():
    clean = ProcessingResult(elements=(), processor=FINGERPRINT)
    degraded = ProcessingResult(
        elements=(), processor=FINGERPRINT,
        diagnostics=(ProcessingDiagnostic(DiagnosticCode.PAGE_NEEDS_VISION),),
    )
    assert clean.is_complete is True
    assert degraded.is_complete is False


def test_clean_text_is_complete_and_has_a_persistable_admission_status():
    result = ProcessingResult(elements=(_text_element(),), processor=FINGERPRINT)

    assert result.admission_status is ProcessingAdmissionStatus.COMPLETE
    assert result.admission_status.value == "complete"
    assert result.is_complete is True


def test_mixed_nonfatal_diagnostics_admit_text_as_partial():
    result = ProcessingResult(
        elements=(_text_element(),),
        processor=FINGERPRINT,
        diagnostics=tuple(
            ProcessingDiagnostic(code)
            for code in (
                DiagnosticCode.PAGE_NEEDS_VISION,
                DiagnosticCode.PAGE_EMPTY,
                DiagnosticCode.FALLBACK_READER_USED,
            )
        ),
    )

    assert result.admission_status is ProcessingAdmissionStatus.PARTIAL
    assert result.admission_status.value == "partial"
    assert result.is_complete is False


def test_vision_diagnostic_is_authoritative_even_without_an_image_element():
    result = ProcessingResult(
        elements=(_text_element(),),
        processor=FINGERPRINT,
        diagnostics=(
            ProcessingDiagnostic(
                DiagnosticCode.PAGE_NEEDS_VISION,
                DocumentLocator(page=2),
            ),
        ),
    )

    assert result.needs_vision is True


def test_vision_element_without_a_coverage_diagnostic_is_rejected():
    result = ProcessingResult(
        elements=(
            _text_element(),
            DocumentElement(
                "image-1",
                ElementKind.IMAGE,
                None,
                DocumentLocator(page=2),
                needs_vision=True,
            ),
        ),
        processor=FINGERPRINT,
    )

    assert result.needs_vision is True
    assert result.admission_status is ProcessingAdmissionStatus.REJECTED


@pytest.mark.parametrize(
    "fatal_code",
    [
        DiagnosticCode.PASSWORD_REQUIRED,
        DiagnosticCode.PERMISSION_DENIED,
        DiagnosticCode.CORRUPT_SOURCE,
        DiagnosticCode.EMPTY_SOURCE,
        DiagnosticCode.LIMIT_REACHED,
    ],
)
def test_any_fatal_diagnostic_rejects_text_even_when_mixed_with_nonfatal_diagnostics(fatal_code):
    result = ProcessingResult(
        elements=(_text_element(),),
        processor=FINGERPRINT,
        diagnostics=(
            ProcessingDiagnostic(DiagnosticCode.PAGE_NEEDS_VISION),
            ProcessingDiagnostic(fatal_code),
            ProcessingDiagnostic(DiagnosticCode.FALLBACK_READER_USED),
        ),
    )

    assert result.admission_status is ProcessingAdmissionStatus.REJECTED


def test_no_text_is_rejected_even_when_the_old_completeness_flag_is_true():
    result = ProcessingResult(elements=(), processor=FINGERPRINT)

    assert result.admission_status is ProcessingAdmissionStatus.REJECTED
    assert result.is_complete is True


def test_the_legacy_projection_uses_the_shared_renderer_not_reader_vocabulary():
    """即使兼容路径也不得重新引入逐格式字符串。"""
    result = ProcessingResult(
        elements=(
            DocumentElement("a", ElementKind.PARAGRAPH, "正文", DocumentLocator(page=2, ordinal=1)),
            DocumentElement("b", ElementKind.IMAGE, None, DocumentLocator(page=3), needs_vision=True),
        ),
        processor=FINGERPRINT,
    )
    legacy = result.legacy_elements()

    assert legacy == [{"content": "正文", "loc": "p2#1"}]
    assert "段" not in str(legacy) and "slide" not in str(legacy)
