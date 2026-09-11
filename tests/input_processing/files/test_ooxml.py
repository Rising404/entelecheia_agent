from __future__ import annotations

import io
import struct
import zipfile
from dataclasses import replace

import pytest

from personagraph.input_processing.files import (
    DEFAULT_OOXML_LIMITS,
    OoxmlFailureKind,
    OoxmlKind,
    OoxmlValidationError,
    probe_ooxml_kind,
    validate_ooxml,
)


def _package(
    entries: dict[str, bytes],
    *,
    compression: int = zipfile.ZIP_DEFLATED,
) -> bytes:
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w", compression=compression) as archive:
        for name, content in entries.items():
            archive.writestr(name, content)
    return payload.getvalue()


def _docx_entries() -> dict[str, bytes]:
    return {
        "[Content_Types].xml": b"<Types/>",
        "_rels/.rels": b"<Relationships/>",
        "word/document.xml": b"<w:document/>",
    }


def _mark_first_member_encrypted(payload: bytes) -> bytes:
    changed = bytearray(payload)
    central = changed.index(b"PK\x01\x02")
    flags = struct.unpack_from("<H", changed, central + 8)[0]
    struct.pack_into("<H", changed, central + 8, flags | 1)
    return bytes(changed)


def _forge_small_eocd_count(payload: bytes) -> bytes:
    changed = bytearray(payload)
    eocd = changed.rindex(b"PK\x05\x06")
    struct.pack_into("<H", changed, eocd + 8, 1)
    struct.pack_into("<H", changed, eocd + 10, 1)
    return bytes(changed)


def test_probe_reads_the_central_directory_not_an_early_head_hint():
    entries = {"padding.bin": b"x" * 32_000, **_docx_entries()}
    payload = _package(entries, compression=zipfile.ZIP_STORED)

    assert b"word/document.xml" not in payload[:8192]
    assert probe_ooxml_kind(payload) is OoxmlKind.DOCX


def test_validation_never_opens_or_expands_a_member(monkeypatch):
    payload = _package(_docx_entries())

    def forbidden_member_open(*_args, **_kwargs):
        raise AssertionError("preflight expanded an archive member")

    monkeypatch.setattr(zipfile.ZipFile, "open", forbidden_member_open)

    assert validate_ooxml(payload, expected_kind=OoxmlKind.DOCX) is OoxmlKind.DOCX


@pytest.mark.parametrize(
    ("payload", "limits", "expected"),
    [
        (
            _package({**_docx_entries(), "extra.xml": b"x"}),
            replace(DEFAULT_OOXML_LIMITS, max_entries=3),
            OoxmlFailureKind.LIMIT,
        ),
        (
            _package({**_docx_entries(), "large.bin": b"12345"}, compression=zipfile.ZIP_STORED),
            replace(DEFAULT_OOXML_LIMITS, max_entry_uncompressed_bytes=4),
            OoxmlFailureKind.LIMIT,
        ),
        (
            _package({**_docx_entries(), "a.bin": b"123", "b.bin": b"456"}, compression=zipfile.ZIP_STORED),
            replace(DEFAULT_OOXML_LIMITS, max_total_uncompressed_bytes=20),
            OoxmlFailureKind.LIMIT,
        ),
        (
            _package({**_docx_entries(), "ratio.bin": b"A" * 10_000}),
            replace(DEFAULT_OOXML_LIMITS, max_compression_ratio=2.0),
            OoxmlFailureKind.LIMIT,
        ),
        (
            _package({**_docx_entries(), "../escape.xml": b"x"}),
            DEFAULT_OOXML_LIMITS,
            OoxmlFailureKind.CORRUPT,
        ),
        (
            _mark_first_member_encrypted(_package(_docx_entries())),
            DEFAULT_OOXML_LIMITS,
            OoxmlFailureKind.ENCRYPTED,
        ),
        (
            _package({"[Content_Types].xml": b"<Types/>", "_rels/.rels": b"<Relationships/>"}),
            DEFAULT_OOXML_LIMITS,
            OoxmlFailureKind.CORRUPT,
        ),
    ],
)
def test_ooxml_preflight_fails_closed_before_package_expansion(
    payload,
    limits,
    expected,
):
    with pytest.raises(OoxmlValidationError) as exc:
        validate_ooxml(payload, expected_kind=OoxmlKind.DOCX, limits=limits)

    assert exc.value.kind is expected


def test_archive_and_central_directory_byte_caps_are_checked_before_zipfile_parse():
    payload = _package(_docx_entries(), compression=zipfile.ZIP_STORED)

    with pytest.raises(OoxmlValidationError) as archive_exc:
        validate_ooxml(
            payload,
            expected_kind=OoxmlKind.DOCX,
            limits=replace(DEFAULT_OOXML_LIMITS, max_archive_bytes=len(payload) - 1),
        )
    with pytest.raises(OoxmlValidationError) as directory_exc:
        validate_ooxml(
            payload,
            expected_kind=OoxmlKind.DOCX,
            limits=replace(DEFAULT_OOXML_LIMITS, max_central_directory_bytes=1),
        )

    assert archive_exc.value.kind is OoxmlFailureKind.LIMIT
    assert directory_exc.value.kind is OoxmlFailureKind.LIMIT


def test_forged_eocd_count_cannot_bypass_the_entry_cap():
    payload = _forge_small_eocd_count(_package({**_docx_entries(), "extra.xml": b"x"}))

    with pytest.raises(OoxmlValidationError) as exc:
        validate_ooxml(
            payload,
            expected_kind=OoxmlKind.DOCX,
            limits=replace(DEFAULT_OOXML_LIMITS, max_entries=3),
        )

    assert exc.value.kind is OoxmlFailureKind.LIMIT


@pytest.mark.parametrize("compression", (zipfile.ZIP_BZIP2, zipfile.ZIP_LZMA))
def test_ooxml_rejects_memory_intensive_compression_methods(compression):
    payload = _package(_docx_entries(), compression=compression)

    with pytest.raises(OoxmlValidationError) as exc:
        validate_ooxml(payload, expected_kind=OoxmlKind.DOCX)

    assert exc.value.kind is OoxmlFailureKind.LIMIT
    assert "compression method" in exc.value.detail
