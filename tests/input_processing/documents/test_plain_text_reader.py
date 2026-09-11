from __future__ import annotations

from personagraph.input_processing.documents.contracts import (
    DiagnosticCode,
    ProcessingAdmissionStatus,
)
from personagraph.input_processing.documents.readers import plain_text as plain_text_reader


def test_a_long_plain_text_block_is_split_losslessly_with_exact_stable_spans(tmp_path):
    text = "A" * plain_text_reader.MAX_ELEMENT_CHARS + "TAIL-SENTINEL"
    path = tmp_path / "long.txt"
    path.write_text(text, encoding="utf-8")

    first = plain_text_reader.read_plain_text(path)
    second = plain_text_reader.read_plain_text(path)

    assert first.admission_status is ProcessingAdmissionStatus.COMPLETE
    assert first.diagnostics == ()
    assert "".join(element.text or "" for element in first.elements) == text
    assert all(
        len(element.text or "") <= plain_text_reader.MAX_ELEMENT_CHARS
        for element in first.elements
    )
    assert [element.locator.char_range for element in first.elements] == [
        (0, plain_text_reader.MAX_ELEMENT_CHARS),
        (plain_text_reader.MAX_ELEMENT_CHARS, len(text)),
    ]
    assert all(
        element.text == text[element.locator.char_range[0]:element.locator.char_range[1]]
        for element in first.elements
    )
    first_ids = [element.element_id for element in first.elements]
    assert len(first_ids) == len(set(first_ids))
    assert first_ids == [element.element_id for element in second.elements]


def test_plain_text_segment_budget_fails_typed_instead_of_claiming_completeness(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(plain_text_reader, "MAX_ELEMENT_CHARS", 4)
    monkeypatch.setattr(plain_text_reader, "MAX_ELEMENTS", 1)
    path = tmp_path / "limited.txt"
    path.write_text("headTAIL", encoding="utf-8")

    result = plain_text_reader.read_plain_text(path)

    assert [element.text for element in result.elements] == ["head"]
    assert [element.locator.char_range for element in result.elements] == [(0, 4)]
    assert [diagnostic.code for diagnostic in result.diagnostics] == [
        DiagnosticCode.LIMIT_REACHED
    ]
    assert result.admission_status is ProcessingAdmissionStatus.REJECTED


def test_invalid_utf8_is_rejected_instead_of_guessed_as_gb18030(tmp_path):
    path = tmp_path / "unknown.txt"
    # 该字节对是合法 GB18030，但不是 UTF-8。旧读取器会猜测编码，并错误声称
    # 来源覆盖完整。
    path.write_bytes(b"\x81\x40")

    result = plain_text_reader.read_plain_text(path)

    assert result.elements == ()
    assert [diagnostic.code for diagnostic in result.diagnostics] == [
        DiagnosticCode.CORRUPT_SOURCE
    ]
    assert result.admission_status is ProcessingAdmissionStatus.REJECTED


def test_plain_text_byte_cap_is_typed_and_checked_before_decoding(tmp_path, monkeypatch):
    monkeypatch.setattr(plain_text_reader, "MAX_TEXT_BYTES", 4)
    path = tmp_path / "too-large.txt"
    path.write_bytes(b"abcde")

    result = plain_text_reader.read_plain_text(path)

    assert result.elements == ()
    assert [diagnostic.code for diagnostic in result.diagnostics] == [
        DiagnosticCode.LIMIT_REACHED
    ]
    assert result.admission_status is ProcessingAdmissionStatus.REJECTED


def test_plain_text_preserves_semantic_indentation_and_exact_source_spans(tmp_path):
    text = "def f():\n    x = 1\n\n    return 73500\n"
    path = tmp_path / "indented.py"
    path.write_text(text, encoding="utf-8")

    result = plain_text_reader.read_plain_text(path)

    assert [element.text for element in result.elements] == [
        "def f():\n    x = 1",
        "    return 73500\n",
    ]
    assert all(
        element.text
        == text[element.locator.char_range[0] : element.locator.char_range[1]]
        for element in result.elements
    )
    assert result.admission_status is ProcessingAdmissionStatus.COMPLETE
