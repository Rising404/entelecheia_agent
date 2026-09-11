"""与摄取格式无关的图片身份和值对象。

``picture_id`` 标识一个文件版本中的精确图片来源；``picture_unit_id`` 标识该
图片的完整像素或由 Host 产生的裁剪/切片。定位器使用规范 JSON，因而存储层可以在
不理解 PDF、DOCX、PPTX 或工具实现的前提下提供稳定幂等语义。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
import hashlib
import json
import math
from typing import TypeAlias


JsonValue: TypeAlias = (
    None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]
)

_MAX_LOCATOR_BYTES = 16_384
_MAX_IDENTIFIER_LENGTH = 512


class PictureSourceKind(StrEnum):
    """图片相对其权威文件版本的来源类别。"""

    WHOLE_FILE = "whole_file"
    EMBEDDED_ASSET = "embedded_asset"
    DOCUMENT_SURFACE = "document_surface"


class PictureDocumentSurfaceKind(StrEnum):
    """可稳定定位、但尚未规定栅格化方式的文档表面。"""

    PDF_PAGE = "pdf_page"
    PPTX_SLIDE = "pptx_slide"


class PictureUnitKind(StrEnum):
    """可交给 OCR/VLM 的确定性像素单元类别。"""

    FULL = "full"
    RENDER = "render"
    CROP = "crop"
    TILE = "tile"


@dataclass(frozen=True, slots=True)
class PictureSourceLocator:
    """文件版本内一张图片的规范、格式中立定位器。"""

    kind: PictureSourceKind
    canonical_json: str

    def __post_init__(self) -> None:
        if not isinstance(self.kind, PictureSourceKind):
            raise ValueError("picture source locator requires a known source kind")
        payload = _parse_locator(self.canonical_json, expected_kind=self.kind.value)
        if self.kind is PictureSourceKind.WHOLE_FILE and payload:
            raise ValueError("whole-file picture locator payload must be empty")
        if self.kind is PictureSourceKind.DOCUMENT_SURFACE:
            _validate_document_surface_payload(payload)

    @classmethod
    def from_payload(
        cls,
        kind: PictureSourceKind | str,
        payload: Mapping[str, object],
    ) -> PictureSourceLocator:
        source_kind = PictureSourceKind(kind)
        return cls(
            kind=source_kind,
            canonical_json=_canonical_locator(source_kind.value, payload),
        )

    @classmethod
    def whole_file(cls) -> PictureSourceLocator:
        return cls.from_payload(PictureSourceKind.WHOLE_FILE, {})

    @classmethod
    def document_surface(
        cls,
        surface_kind: PictureDocumentSurfaceKind | str,
        ordinal: int,
    ) -> PictureSourceLocator:
        kind = PictureDocumentSurfaceKind(surface_kind)
        return cls.from_payload(
            PictureSourceKind.DOCUMENT_SURFACE,
            {"surface_kind": kind.value, "ordinal": ordinal},
        )

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json.encode("utf-8")).hexdigest()

    @property
    def payload(self) -> dict[str, JsonValue]:
        parsed = _load_json(self.canonical_json)
        return dict(parsed["payload"])


@dataclass(frozen=True, slots=True)
class PictureUnitLocator:
    """一张图片内部完整像素、裁剪或切片的规范定位器。"""

    kind: PictureUnitKind
    canonical_json: str

    def __post_init__(self) -> None:
        if not isinstance(self.kind, PictureUnitKind):
            raise ValueError("picture unit locator requires a known unit kind")
        payload = _parse_locator(self.canonical_json, expected_kind=self.kind.value)
        if self.kind is PictureUnitKind.FULL and payload:
            raise ValueError("full-picture unit locator payload must be empty")

    @classmethod
    def from_payload(
        cls,
        kind: PictureUnitKind | str,
        payload: Mapping[str, object],
    ) -> PictureUnitLocator:
        unit_kind = PictureUnitKind(kind)
        return cls(
            kind=unit_kind,
            canonical_json=_canonical_locator(unit_kind.value, payload),
        )

    @classmethod
    def full(cls) -> PictureUnitLocator:
        return cls.from_payload(PictureUnitKind.FULL, {})

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json.encode("utf-8")).hexdigest()

    @property
    def payload(self) -> dict[str, JsonValue]:
        parsed = _load_json(self.canonical_json)
        return dict(parsed["payload"])


@dataclass(frozen=True, slots=True)
class PictureRecord:
    """一个精确文件版本中已登记图片的不可变身份。"""

    picture_id: str
    file_id: str
    file_version_id: str
    source_locator: PictureSourceLocator
    source_content_sha256: str
    source_media_type: str
    created_at: str

    def __post_init__(self) -> None:
        _validate_identifier(self.picture_id, field="picture_id")
        _validate_identifier(self.file_id, field="file_id")
        _validate_identifier(self.file_version_id, field="file_version_id")
        if not isinstance(self.source_locator, PictureSourceLocator):
            raise ValueError("picture record requires a source locator")
        _validate_sha256(
            self.source_content_sha256,
            field="source_content_sha256",
        )
        _validate_media_type(self.source_media_type, field="source_media_type")
        _validate_timestamp(self.created_at, field="created_at")


@dataclass(frozen=True, slots=True)
class PictureUnitRecord:
    """一张图片中可独立处理的不可变像素单元。"""

    picture_unit_id: str
    picture_id: str
    locator: PictureUnitLocator
    producer_fingerprint: str
    parent_picture_unit_id: str | None
    pixel_sha256: str
    media_type: str
    width: int
    height: int
    created_at: str

    def __post_init__(self) -> None:
        _validate_identifier(self.picture_unit_id, field="picture_unit_id")
        _validate_identifier(self.picture_id, field="picture_id")
        if not isinstance(self.locator, PictureUnitLocator):
            raise ValueError("picture unit record requires a unit locator")
        _validate_identifier(
            self.producer_fingerprint,
            field="producer_fingerprint",
        )
        if self.parent_picture_unit_id is not None:
            _validate_identifier(
                self.parent_picture_unit_id,
                field="parent_picture_unit_id",
            )
        if self.locator.kind in {PictureUnitKind.FULL, PictureUnitKind.RENDER}:
            if self.parent_picture_unit_id is not None:
                raise ValueError("full/render picture unit cannot have a parent")
        elif self.parent_picture_unit_id is None:
            raise ValueError("crop/tile picture unit requires a parent")
        _validate_sha256(self.pixel_sha256, field="pixel_sha256")
        _validate_image_media_type(self.media_type)
        if not isinstance(self.width, int) or isinstance(self.width, bool) or self.width < 1:
            raise ValueError("picture unit width must be a positive integer")
        if not isinstance(self.height, int) or isinstance(self.height, bool) or self.height < 1:
            raise ValueError("picture unit height must be a positive integer")
        _validate_timestamp(self.created_at, field="created_at")


@dataclass(frozen=True, slots=True)
class RegisteredPicture:
    """图片幂等登记的结果。"""

    picture: PictureRecord
    created: bool


@dataclass(frozen=True, slots=True)
class RegisteredPictureUnit:
    """图片单元幂等登记的结果。"""

    unit: PictureUnitRecord
    created: bool


def _canonical_locator(kind: str, payload: Mapping[str, object]) -> str:
    if not isinstance(payload, Mapping):
        raise ValueError("picture locator payload must be an object")
    normalized = _normalize_json_value(payload)
    assert isinstance(normalized, dict)
    encoded = json.dumps(
        {"kind": kind, "payload": normalized},
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    if len(encoded.encode("utf-8")) > _MAX_LOCATOR_BYTES:
        raise ValueError("picture locator exceeds the bounded storage contract")
    return encoded


def _parse_locator(canonical_json: str, *, expected_kind: str) -> dict[str, JsonValue]:
    if not isinstance(canonical_json, str) or not canonical_json:
        raise ValueError("picture locator must be non-empty canonical JSON")
    if len(canonical_json.encode("utf-8")) > _MAX_LOCATOR_BYTES:
        raise ValueError("picture locator exceeds the bounded storage contract")
    parsed = _load_json(canonical_json)
    if set(parsed) != {"kind", "payload"}:
        raise ValueError("picture locator must contain only kind and payload")
    if parsed["kind"] != expected_kind or not isinstance(parsed["payload"], dict):
        raise ValueError("picture locator kind/payload does not match its contract")
    normalized = _canonical_locator(expected_kind, parsed["payload"])
    if normalized != canonical_json:
        raise ValueError("picture locator JSON is not canonical")
    return dict(parsed["payload"])


def _load_json(encoded: str) -> dict[str, JsonValue]:
    try:
        value = json.loads(
            encoded,
            parse_constant=_raise_invalid_constant,
        )
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("picture locator is not valid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("picture locator must be a JSON object")
    return value


def _raise_invalid_constant(constant: str) -> None:
    raise ValueError(f"non-finite JSON number is not allowed: {constant}")


def _normalize_json_value(value: object) -> JsonValue:
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("picture locator numbers must be finite")
        return value
    if isinstance(value, Mapping):
        normalized: dict[str, JsonValue] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise ValueError("picture locator object keys must be non-empty strings")
            normalized[key] = _normalize_json_value(item)
        return normalized
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_normalize_json_value(item) for item in value]
    raise ValueError(f"unsupported picture locator JSON value: {type(value).__name__}")


def _validate_document_surface_payload(payload: Mapping[str, JsonValue]) -> None:
    if set(payload) != {"surface_kind", "ordinal"}:
        raise ValueError(
            "document-surface locator requires only surface_kind and ordinal"
        )
    try:
        PictureDocumentSurfaceKind(payload["surface_kind"])
    except (TypeError, ValueError) as exc:
        raise ValueError("document-surface locator has an unknown surface_kind") from exc
    ordinal = payload["ordinal"]
    if isinstance(ordinal, bool) or not isinstance(ordinal, int) or ordinal < 1:
        raise ValueError("document-surface locator ordinal must be a positive integer")


def _validate_identifier(value: str, *, field: str) -> None:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or len(value) > _MAX_IDENTIFIER_LENGTH
        or any(ord(character) < 32 for character in value)
    ):
        raise ValueError(f"{field} must be a bounded non-empty identifier")


def _validate_sha256(value: str, *, field: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")


def _validate_image_media_type(value: str) -> None:
    _validate_media_type(value, field="media_type")
    if not value.startswith("image/"):
        raise ValueError("picture unit media_type must be an image MIME type")


def _validate_media_type(value: str, *, field: str) -> None:
    if (
        not isinstance(value, str)
        or value != value.strip().lower()
        or ";" in value
        or value.count("/") != 1
        or any(not part for part in value.split("/"))
    ):
        raise ValueError(f"{field} must be a canonical type/subtype MIME type")


def _validate_timestamp(value: str, *, field: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a timezone-aware ISO timestamp")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{field} must be a timezone-aware ISO timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field} must be a timezone-aware ISO timestamp")
