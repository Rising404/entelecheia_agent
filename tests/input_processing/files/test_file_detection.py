from __future__ import annotations

import io
import zipfile

import pytest

from personagraph.input_processing.files import (
    FileKind,
    MAX_STORED_NAME_LENGTH,
    detect_type,
    sanitize_original_name,
)


def _detect(head: bytes, name: str):
    return detect_type(head, sanitize_original_name(name))


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("report.pdf", "report.pdf"),
        ("../../etc/passwd", "passwd"),
        ("..\\..\\Windows\\system32\\cmd.exe", "cmd.exe"),
        ("/absolute/path/notes.md", "notes.md"),
        ("..", "file"),
        ("", "file"),
        ("   ...   ", "file"),
        ("图 表 (最终).png", "图 表 (最终).png"),
    ],
)
def test_sanitized_names_keep_meaning_without_keeping_structure(raw, expected):
    assert sanitize_original_name(raw) == expected


def test_control_characters_are_stripped_from_names():
    assert "\x00" not in sanitize_original_name("evil\x00.png")
    assert "\n" not in sanitize_original_name("two\nlines.txt")


def test_long_names_are_truncated_but_keep_their_extension():
    name = sanitize_original_name("a" * 300 + ".png")
    assert len(name) <= MAX_STORED_NAME_LENGTH
    assert name.endswith(".png")


@pytest.mark.parametrize(
    ("head", "name", "media_type", "kind"),
    [
        (b"\x89PNG\r\n\x1a\n" + b"\x00" * 32, "a.png", "image/png", FileKind.IMAGE),
        (b"\xff\xd8\xff\xe0" + b"\x00" * 32, "a.jpg", "image/jpeg", FileKind.IMAGE),
        (b"RIFF\x00\x00\x00\x00WEBP" + b"\x00" * 8, "a.webp", "image/webp", FileKind.IMAGE),
        (b"GIF89a" + b"\x00" * 32, "a.gif", "image/gif", FileKind.IMAGE),
        (b"%PDF-1.7\n" + b"x" * 32, "a.pdf", "application/pdf", FileKind.DOCUMENT),
        (b"ID3\x03" + b"\x00" * 32, "a.mp3", "audio/mpeg", FileKind.AUDIO),
        (b"OggS" + b"\x00" * 32, "a.ogg", "audio/ogg", FileKind.AUDIO),
        (b"\x00\x00\x00\x20ftypmp42" + b"\x00" * 8, "a.mp4", "video/mp4", FileKind.VIDEO),
        ("会议记录\n第一行".encode(), "a.md", "text/markdown", FileKind.TEXT),
        (b"a,b,c\n1,2,3\n", "a.csv", "text/csv", FileKind.TEXT),
        (b"def f():\n    return 1\n", "a.py", "text/x-python", FileKind.TEXT),
    ],
)
def test_detection_reads_the_bytes(head, name, media_type, kind):
    detected = _detect(head, name)
    assert detected.media_type == media_type
    assert detected.kind is kind


@pytest.mark.parametrize(
    ("main_part", "media_type"),
    [
        (b"word/document.xml", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
        (b"ppt/presentation.xml", "application/vnd.openxmlformats-officedocument.presentationml.presentation"),
        (b"xl/workbook.xml", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
    ],
)
def test_office_formats_are_told_apart_by_their_zip_members(main_part, media_type):
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("padding.bin", b"x" * 32_000)
        archive.writestr("[Content_Types].xml", b"<Types/>")
        archive.writestr("_rels/.rels", b"<Relationships/>")
        archive.writestr(main_part.decode(), b"<root/>")
    assert main_part not in payload.getvalue()[:8192]

    # 前置自解压器/填充区域属于合法 ZIP 结构，也证明检测并未与零字节处的
    # PK 签名耦合。
    padded_payload = b"p" * 9000 + payload.getvalue()
    detected = _detect(padded_payload, "a.docx")
    assert detected.media_type == media_type
    assert detected.kind is FileKind.DOCUMENT


@pytest.mark.parametrize(
    ("name", "media_type", "extension"),
    [
        ("legacy.doc", "application/msword", ".doc"),
        ("legacy.ppt", "application/vnd.ms-powerpoint", ".ppt"),
    ],
)
def test_legacy_office_container_is_identified_but_not_claimed_as_readable(
    name,
    media_type,
    extension,
):
    detected = _detect(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"\x00" * 32, name)

    assert detected.media_type == media_type
    assert detected.kind is FileKind.DOCUMENT
    assert detected.extension == extension


def test_a_plain_archive_stays_unknown_because_nothing_unpacks_it():
    """声称类型可读会承诺 V1 实际并不具备的能力。"""
    detected = _detect(b"PK\x03\x04" + b"\x00" * 20 + b"random", "bundle.zip")
    assert detected.media_type == "application/zip"
    assert detected.kind is FileKind.UNKNOWN


def test_a_lying_extension_never_wins_over_the_content():
    detected = _detect(b"%PDF-1.4\n" + b"x" * 32, "definitely_an_image.png")
    assert detected.media_type == "application/pdf"
    assert detected.kind is FileKind.DOCUMENT


@pytest.mark.parametrize("name", ("notes.pdf", "notes.doc", "notes.png"))
def test_plain_text_with_a_disproven_suffix_is_stored_as_txt(name):
    detected = _detect(b"plain UTF-8 evidence 73500\n", name)

    assert detected.media_type == "text/plain"
    assert detected.kind is FileKind.TEXT
    assert detected.extension == ".txt"


@pytest.mark.parametrize(
    "head",
    [
        b"\x7fELF\x02\x01\x01\x00" + bytes(range(1, 30)),  # 可按 UTF-8 解码，但不是文本
        b"\x01\x02\x03\x04\x05\x06\x07\x08",
        b"text then a nul\x00and more",
    ],
)
def test_binary_payloads_are_not_admitted_as_text(head):
    """对模型而言，提示中的乱码与真实内容无法区分。"""
    assert _detect(head, "mystery.txt").kind is FileKind.UNKNOWN


def test_an_empty_payload_is_treated_as_empty_text_not_as_binary():
    assert _detect(b"", "notes.txt").kind is FileKind.TEXT
