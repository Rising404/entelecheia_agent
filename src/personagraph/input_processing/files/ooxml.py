"""对不可信 OOXML ZIP 容器执行有界、非展开检查。

标准 Office 解析器会先打开 ZIP 成员，再公开文档结构。本模块是它们之前的资源与身份 gate：
只检查有界中央目录、拒绝不安全包，并且绝不对归档成员调用 ``read()``、``open()`` 或
``testzip()``。
"""

from __future__ import annotations

import hashlib
import io
import stat
import struct
import zipfile
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path


MAX_OOXML_ARCHIVE_BYTES = 64 * 1024 * 1024
MAX_OOXML_CENTRAL_DIRECTORY_BYTES = 16 * 1024 * 1024
MAX_OOXML_ENTRIES = 10_000
MAX_OOXML_ENTRY_UNCOMPRESSED_BYTES = 64 * 1024 * 1024
MAX_OOXML_TOTAL_UNCOMPRESSED_BYTES = 256 * 1024 * 1024
MAX_OOXML_COMPRESSION_RATIO = 200.0


class OoxmlKind(StrEnum):
    DOCX = "docx"
    PPTX = "pptx"
    XLSX = "xlsx"


class OoxmlFailureKind(StrEnum):
    CORRUPT = "corrupt"
    ENCRYPTED = "encrypted"
    LIMIT = "limit"


@dataclass(frozen=True, slots=True)
class OoxmlLimits:
    max_archive_bytes: int
    max_central_directory_bytes: int
    max_entries: int
    max_entry_uncompressed_bytes: int
    max_total_uncompressed_bytes: int
    max_compression_ratio: float

    def __post_init__(self) -> None:
        for field_name in (
            "max_archive_bytes",
            "max_central_directory_bytes",
            "max_entries",
            "max_entry_uncompressed_bytes",
            "max_total_uncompressed_bytes",
        ):
            if getattr(self, field_name) < 1:
                raise ValueError(f"{field_name} must be positive")
        if self.max_compression_ratio < 1:
            raise ValueError("max_compression_ratio must be at least 1")


DEFAULT_OOXML_LIMITS = OoxmlLimits(
    max_archive_bytes=MAX_OOXML_ARCHIVE_BYTES,
    max_central_directory_bytes=MAX_OOXML_CENTRAL_DIRECTORY_BYTES,
    max_entries=MAX_OOXML_ENTRIES,
    max_entry_uncompressed_bytes=MAX_OOXML_ENTRY_UNCOMPRESSED_BYTES,
    max_total_uncompressed_bytes=MAX_OOXML_TOTAL_UNCOMPRESSED_BYTES,
    max_compression_ratio=MAX_OOXML_COMPRESSION_RATIO,
)


class OoxmlValidationError(ValueError):
    """OOXML 容器被拒绝的安全且不含内容原因。"""

    def __init__(self, kind: OoxmlFailureKind, detail: str) -> None:
        super().__init__(detail)
        self.kind = kind
        self.detail = detail


@dataclass(frozen=True, slots=True)
class ValidatedOoxmlSource:
    payload: bytes
    source_sha256: str
    kind: OoxmlKind


_MAIN_PARTS = {
    OoxmlKind.DOCX: "word/document.xml",
    OoxmlKind.PPTX: "ppt/presentation.xml",
    OoxmlKind.XLSX: "xl/workbook.xml",
}
_COMMON_REQUIRED_PARTS = frozenset({"[Content_Types].xml", "_rels/.rels"})
_ALLOWED_COMPRESSION_METHODS = frozenset({
    zipfile.ZIP_STORED,
    zipfile.ZIP_DEFLATED,
})


def read_validated_ooxml(
    path: Path,
    *,
    expected_kind: OoxmlKind,
    limits: OoxmlLimits | None = None,
) -> ValidatedOoxmlSource:
    """最多读取一个有界归档，并把验证绑定到这些字节。"""

    active = limits or DEFAULT_OOXML_LIMITS
    with path.open("rb") as source:
        payload = source.read(active.max_archive_bytes + 1)
    if len(payload) > active.max_archive_bytes:
        raise OoxmlValidationError(
            OoxmlFailureKind.LIMIT,
            "OOXML archive byte limit reached",
        )
    kind = validate_ooxml(payload, expected_kind=expected_kind, limits=active)
    return ValidatedOoxmlSource(
        payload=payload,
        source_sha256=hashlib.sha256(payload).hexdigest(),
        kind=kind,
    )


def probe_ooxml_kind(
    payload: bytes,
    *,
    limits: OoxmlLimits | None = None,
) -> OoxmlKind | None:
    """返回已证明的 OOXML 类型；通用/无效 ZIP 字节返回 ``None``。

    检测被刻意设计为保守失败：判断媒体类型的调用方无须区分不安全 ZIP 与非 Office ZIP。
    严格 Office reader 会使用 :func:`validate_ooxml` 并保留类型化失败。
    """

    try:
        return _inspect_ooxml(payload, expected_kind=None, limits=limits)
    except OoxmlValidationError:
        return None


def validate_ooxml(
    payload: bytes,
    *,
    expected_kind: OoxmlKind,
    limits: OoxmlLimits | None = None,
) -> OoxmlKind:
    """在不展开成员的情况下严格验证一个预期 OOXML 包。"""

    result = _inspect_ooxml(payload, expected_kind=expected_kind, limits=limits)
    if result is None:  # 严格检查绝不会返回 None
        raise OoxmlValidationError(OoxmlFailureKind.CORRUPT, "OOXML main part missing")
    return result


def _inspect_ooxml(
    payload: bytes,
    *,
    expected_kind: OoxmlKind | None,
    limits: OoxmlLimits | None,
) -> OoxmlKind | None:
    if not isinstance(payload, bytes):
        raise TypeError("OOXML payload must be bytes")
    active = limits or DEFAULT_OOXML_LIMITS
    if len(payload) > active.max_archive_bytes:
        raise OoxmlValidationError(
            OoxmlFailureKind.LIMIT,
            "OOXML archive byte limit reached",
        )

    stream = io.BytesIO(payload)
    try:
        end_record = zipfile._EndRecData(stream)  # type: ignore[attr-defined]
    except (OSError, zipfile.BadZipFile, struct.error) as exc:
        raise OoxmlValidationError(
            OoxmlFailureKind.CORRUPT,
            f"invalid ZIP directory:{type(exc).__name__}",
        ) from exc
    if end_record is None:
        raise OoxmlValidationError(OoxmlFailureKind.CORRUPT, "ZIP directory missing")

    if (
        int(end_record[zipfile._ECD_DISK_NUMBER]) != 0  # type: ignore[attr-defined]
        or int(end_record[zipfile._ECD_DISK_START]) != 0  # type: ignore[attr-defined]
        or int(end_record[zipfile._ECD_ENTRIES_THIS_DISK])  # type: ignore[attr-defined]
        != int(end_record[zipfile._ECD_ENTRIES_TOTAL])  # type: ignore[attr-defined]
    ):
        raise OoxmlValidationError(
            OoxmlFailureKind.CORRUPT,
            "multi-disk OOXML archives are unsupported",
        )

    entry_count = int(end_record[zipfile._ECD_ENTRIES_TOTAL])  # type: ignore[attr-defined]
    directory_bytes = int(end_record[zipfile._ECD_SIZE])  # type: ignore[attr-defined]
    if entry_count > active.max_entries:
        raise OoxmlValidationError(
            OoxmlFailureKind.LIMIT,
            "OOXML central directory entry limit reached",
        )
    if directory_bytes > active.max_central_directory_bytes:
        raise OoxmlValidationError(
            OoxmlFailureKind.LIMIT,
            "OOXML central directory byte limit reached",
        )

    # 不允许伪造的 EOCD 条目数诱使 ZipFile 分配无界数量的 ZipInfo 对象。先对有界原始目录
    # 记录计数；成员内容保持不动。
    actual_entry_count = _count_central_directory_entries(
        payload,
        end_record=end_record,
        directory_bytes=directory_bytes,
        max_entries=active.max_entries,
    )
    if actual_entry_count != entry_count:
        raise OoxmlValidationError(
            OoxmlFailureKind.CORRUPT,
            "OOXML central directory entry count mismatch",
        )

    stream.seek(0)
    try:
        with zipfile.ZipFile(stream, "r") as archive:
            members = archive.infolist()
    except (OSError, ValueError, zipfile.BadZipFile, zipfile.LargeZipFile) as exc:
        raise OoxmlValidationError(
            OoxmlFailureKind.CORRUPT,
            f"invalid ZIP directory:{type(exc).__name__}",
        ) from exc
    if len(members) != actual_entry_count or len(members) > active.max_entries:
        raise OoxmlValidationError(
            OoxmlFailureKind.CORRUPT,
            "OOXML central directory entry count mismatch",
        )

    names: set[str] = set()
    total_uncompressed = 0
    for member in members:
        name = _validated_member_name(member)
        if name in names:
            raise OoxmlValidationError(
                OoxmlFailureKind.CORRUPT,
                "duplicate OOXML member name",
            )
        names.add(name)
        if member.flag_bits & 1:
            raise OoxmlValidationError(
                OoxmlFailureKind.ENCRYPTED,
                "encrypted OOXML member",
            )
        if member.compress_type not in _ALLOWED_COMPRESSION_METHODS:
            raise OoxmlValidationError(
                OoxmlFailureKind.LIMIT,
                "OOXML compression method is not permitted",
            )
        if member.file_size > active.max_entry_uncompressed_bytes:
            raise OoxmlValidationError(
                OoxmlFailureKind.LIMIT,
                "OOXML member uncompressed byte limit reached",
            )
        total_uncompressed += member.file_size
        if total_uncompressed > active.max_total_uncompressed_bytes:
            raise OoxmlValidationError(
                OoxmlFailureKind.LIMIT,
                "OOXML total uncompressed byte limit reached",
            )
        if member.file_size:
            if member.compress_size <= 0:
                raise OoxmlValidationError(
                    OoxmlFailureKind.LIMIT,
                    "OOXML member has an unbounded compression ratio",
                )
            ratio = member.file_size / member.compress_size
            if ratio > active.max_compression_ratio:
                raise OoxmlValidationError(
                    OoxmlFailureKind.LIMIT,
                    "OOXML member compression ratio limit reached",
                )

    claimed = tuple(kind for kind, main_part in _MAIN_PARTS.items() if main_part in names)
    if expected_kind is None:
        if not claimed:
            return None
        if len(claimed) != 1:
            raise OoxmlValidationError(
                OoxmlFailureKind.CORRUPT,
                "ambiguous OOXML main parts",
            )
        kind = claimed[0]
    else:
        kind = expected_kind
        if claimed != (kind,):
            raise OoxmlValidationError(
                OoxmlFailureKind.CORRUPT,
                "OOXML main part does not match expected format",
            )

    required = _COMMON_REQUIRED_PARTS | {_MAIN_PARTS[kind]}
    if not required <= names:
        raise OoxmlValidationError(
            OoxmlFailureKind.CORRUPT,
            "required OOXML package parts missing",
        )
    return kind


def _validated_member_name(member: zipfile.ZipInfo) -> str:
    name = member.orig_filename
    if (
        not name
        or "\x00" in name
        or "\\" in name
        or name.startswith("/")
        or (len(name) >= 2 and name[1] == ":" and name[0].isalpha())
    ):
        raise OoxmlValidationError(OoxmlFailureKind.CORRUPT, "unsafe OOXML member name")
    path_text = name[:-1] if name.endswith("/") else name
    if not path_text or any(part in {"", ".", ".."} for part in path_text.split("/")):
        raise OoxmlValidationError(OoxmlFailureKind.CORRUPT, "unsafe OOXML member name")

    unix_mode = (member.external_attr >> 16) & 0xFFFF
    file_type = stat.S_IFMT(unix_mode)
    if file_type not in {0, stat.S_IFREG, stat.S_IFDIR}:
        raise OoxmlValidationError(OoxmlFailureKind.CORRUPT, "unsafe OOXML member type")
    if member.is_dir() and member.file_size:
        raise OoxmlValidationError(OoxmlFailureKind.CORRUPT, "invalid OOXML directory member")
    return name


def _count_central_directory_entries(
    payload: bytes,
    *,
    end_record: list[object],
    directory_bytes: int,
    max_entries: int,
) -> int:
    location = int(end_record[zipfile._ECD_LOCATION])  # type: ignore[attr-defined]
    start = location - directory_bytes
    if start < 0 or location > len(payload):
        raise OoxmlValidationError(
            OoxmlFailureKind.CORRUPT,
            "invalid OOXML central directory bounds",
        )
    directory = memoryview(payload)[start:location]
    cursor = 0
    count = 0
    fixed_size = zipfile.sizeCentralDir  # type: ignore[attr-defined]
    while cursor < directory_bytes:
        if directory_bytes - cursor < fixed_size:
            raise OoxmlValidationError(
                OoxmlFailureKind.CORRUPT,
                "truncated OOXML central directory",
            )
        try:
            record = struct.unpack_from(zipfile.structCentralDir, directory, cursor)  # type: ignore[attr-defined]
        except struct.error as exc:
            raise OoxmlValidationError(
                OoxmlFailureKind.CORRUPT,
                "invalid OOXML central directory record",
            ) from exc
        if record[zipfile._CD_SIGNATURE] != zipfile.stringCentralDir:  # type: ignore[attr-defined]
            raise OoxmlValidationError(
                OoxmlFailureKind.CORRUPT,
                "invalid OOXML central directory signature",
            )
        variable_size = (
            int(record[zipfile._CD_FILENAME_LENGTH])  # type: ignore[attr-defined]
            + int(record[zipfile._CD_EXTRA_FIELD_LENGTH])  # type: ignore[attr-defined]
            + int(record[zipfile._CD_COMMENT_LENGTH])  # type: ignore[attr-defined]
        )
        cursor += fixed_size + variable_size
        if cursor > directory_bytes:
            raise OoxmlValidationError(
                OoxmlFailureKind.CORRUPT,
                "truncated OOXML central directory member",
            )
        count += 1
        if count > max_entries:
            raise OoxmlValidationError(
                OoxmlFailureKind.LIMIT,
                "OOXML central directory entry limit reached",
            )
    return count


__all__ = [
    "DEFAULT_OOXML_LIMITS",
    "OoxmlFailureKind",
    "OoxmlKind",
    "OoxmlLimits",
    "OoxmlValidationError",
    "ValidatedOoxmlSource",
    "probe_ooxml_kind",
    "read_validated_ooxml",
    "validate_ooxml",
]
