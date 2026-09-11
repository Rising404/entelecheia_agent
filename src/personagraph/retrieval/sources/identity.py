"""Source 写入与只读适配器共享的纯稳定标识符。"""

from __future__ import annotations

import base64
import binascii
import hashlib
import re
from dataclasses import dataclass
from ..contracts import SourceType, SourceUnitRef


_PROJECT_DOCUMENT_IDENTITY_MARKER = "document-v3"
_LEGACY_TYPED_DOCUMENT_IDENTITY_MARKER = "document-v2"
_CURRENT_SESSION_IDENTITY_MARKER = "cs2"
_PICTURE_OBSERVATION_IDENTITY_MARKER = "picture-observation-v1"
_CURRENT_SESSION_ROLES = frozenset({"pair", "user", "assistant"})
_CANONICAL_BASE64_COMPONENT = re.compile(r"[A-Za-z0-9_-]+\Z")
_CANONICAL_ORDINAL = re.compile(r"(?:0|[1-9][0-9]{0,17})\Z")


@dataclass(frozen=True, slots=True)
class CurrentSessionSourceUnitIdentity:
    session_id: str
    run_id: str
    role: str
    ordinal: int


@dataclass(frozen=True, slots=True)
class MountedDocumentChunkIdentity:
    session_id: str | None = None
    storage_chunk_id: str | None = None
    doc_id: str | None = None
    producer_chunk_id: str | None = None

    @property
    def is_typed(self) -> bool:
        return self.doc_id is not None and self.producer_chunk_id is not None

    @property
    def is_project_scoped(self) -> bool:
        return self.is_typed and self.session_id is None


@dataclass(frozen=True, slots=True)
class PictureObservationIdentity:
    """Typed identity for one immutable picture-observation Source Unit."""

    observation_id: str


def current_session_pair_ref_and_content(
    *,
    session_id: str,
    run_id: str,
    created_at: str,
    user_content: str,
    assistant_content: str,
) -> tuple[SourceUnitRef, str]:
    """为一个已提交 Turn 对构建唯一规范检索身份。"""

    content = f"用户：{user_content.strip()}\n\n助手：{assistant_content.strip()}"
    return (
        SourceUnitRef(
            source_type=SourceType.CURRENT_SESSION,
            source_unit_id=current_session_source_unit_id(
                session_id=session_id,
                run_id=run_id,
                role="pair",
                ordinal=0,
            ),
            source_revision=created_at,
            indexed_content_hash=hashlib.sha256(content.encode("utf-8")).hexdigest(),
        ),
        content,
    )


def current_session_source_unit_id(
    *,
    session_id: str,
    run_id: str,
    role: str,
    ordinal: int,
) -> str:
    """格式化当前 Session 派生索引使用的规范 Source 身份。"""

    if not isinstance(role, str) or role not in _CURRENT_SESSION_ROLES:
        raise ValueError("current Session source role is invalid")
    if (
        isinstance(ordinal, bool)
        or not isinstance(ordinal, int)
        or not 0 <= ordinal < 10**18
    ):
        raise ValueError(
            "current Session source ordinal is outside its supported range"
        )
    return ":".join(
        (
            _CURRENT_SESSION_IDENTITY_MARKER,
            _encode_current_session_component(session_id, "session_id"),
            _encode_current_session_component(run_id, "run_id"),
            role,
            str(ordinal),
        )
    )


def parse_current_session_source_unit_id(
    value: str,
) -> CurrentSessionSourceUnitIdentity | None:
    """严格解析规范身份；不接受填充、额外段或非规范十进制。"""

    if not isinstance(value, str):
        return None
    parts = value.split(":")
    if (
        len(parts) != 5
        or parts[0] != _CURRENT_SESSION_IDENTITY_MARKER
        or parts[3] not in _CURRENT_SESSION_ROLES
        or _CANONICAL_ORDINAL.fullmatch(parts[4]) is None
    ):
        return None
    session_id = _decode_current_session_component(parts[1])
    run_id = _decode_current_session_component(parts[2])
    if session_id is None or run_id is None:
        return None
    return CurrentSessionSourceUnitIdentity(
        session_id=session_id,
        run_id=run_id,
        role=parts[3],
        ordinal=int(parts[4]),
    )


def picture_observation_ref_and_content(
    *,
    observation_id: str,
    payload_sha256: str,
    text: str,
    question: str | None = None,
) -> tuple[SourceUnitRef, str]:
    """Build the canonical pointer and normalized text for one observation.

    Empty observations remain valid picture-domain ledger entries, but they are not
    retrieval Source Units.  Rejecting them here keeps every caller from inventing a
    second normalization rule.
    """

    normalized_answer = text.strip()
    if not normalized_answer:
        raise ValueError("picture observation text must not be empty")
    if question is None:
        normalized_content = normalized_answer
    else:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("picture observation question must not be empty")
        normalized_question = question.strip()
        normalized_content = (
            f"问题：{normalized_question}\n\n回答：{normalized_answer}"
        )
    if (
        not isinstance(payload_sha256, str)
        or len(payload_sha256) != 64
        or any(character not in "0123456789abcdef" for character in payload_sha256)
    ):
        raise ValueError("payload_sha256 must be a canonical lowercase sha256")
    return (
        SourceUnitRef(
            source_type=SourceType.PICTURE,
            source_unit_id=picture_observation_source_unit_id(observation_id),
            source_revision=payload_sha256,
            indexed_content_hash=hashlib.sha256(
                normalized_content.encode("utf-8")
            ).hexdigest(),
        ),
        normalized_content,
    )


def picture_observation_source_unit_id(observation_id: str) -> str:
    """Format a canonical, delimiter-safe ``picture-observation-v1`` identity."""

    return ":".join(
        (
            _PICTURE_OBSERVATION_IDENTITY_MARKER,
            _encode_current_session_component(observation_id, "observation_id"),
        )
    )


def parse_picture_observation_source_unit_id(
    value: str,
) -> PictureObservationIdentity | None:
    """Strictly parse a canonical picture-observation Source Unit identity."""

    if not isinstance(value, str):
        return None
    marker, separator, encoded_observation_id = value.partition(":")
    if (
        not separator
        or marker != _PICTURE_OBSERVATION_IDENTITY_MARKER
        or ":" in encoded_observation_id
    ):
        return None
    observation_id = _decode_current_session_component(encoded_observation_id)
    if observation_id is None:
        return None
    return PictureObservationIdentity(observation_id=observation_id)


def _encode_current_session_component(value: str, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    encoded = base64.urlsafe_b64encode(value.encode("utf-8")).decode("ascii")
    return encoded.rstrip("=")


def _decode_current_session_component(value: str) -> str | None:
    if _CANONICAL_BASE64_COMPONENT.fullmatch(value) is None:
        return None
    padded = value + "=" * (-len(value) % 4)
    try:
        decoded = base64.b64decode(
            padded.encode("ascii"),
            altchars=b"-_",
            validate=True,
        ).decode("utf-8")
    except (binascii.Error, UnicodeDecodeError, ValueError):
        return None
    if not decoded:
        return None
    if _encode_current_session_component(decoded, "component") != value:
        return None
    return decoded


def mounted_document_chunk_ref_and_content(
    *,
    session_id: str,
    chunk_id: str,
    source_version_id: str,
    content: str,
    doc_id: str | None = None,
    producer_chunk_id: str | None = None,
) -> tuple[SourceUnitRef, str]:
    """为一个项目文档块构建规范身份。

    带类型块是项目所有的派生数据，因此其稳定身份与当前挂载文档的 Session 无关。Session
    可见性仍由 Source 适配器在搜索前以及获取权威内容时分别执行授权检查。

    无类型旧版行只在显式注入的迁移或测试存储中保留其历史 Session 作用域身份。
    """

    normalized_content = content.strip()
    source_unit_id = mounted_document_chunk_source_unit_id(
        session_id=session_id,
        storage_chunk_id=chunk_id,
        doc_id=doc_id,
        producer_chunk_id=producer_chunk_id,
    )
    return (
        SourceUnitRef(
            source_type=SourceType.DOCUMENT,
            source_unit_id=source_unit_id,
            source_revision=source_version_id,
            indexed_content_hash=hashlib.sha256(normalized_content.encode("utf-8")).hexdigest(),
        ),
        normalized_content,
    )


def mounted_document_chunk_source_unit_id(
    *,
    session_id: str,
    storage_chunk_id: str,
    doc_id: str | None = None,
    producer_chunk_id: str | None = None,
) -> str:
    """格式化项目所有的带类型身份，同时保留旧版行。"""

    if not storage_chunk_id.strip():
        raise ValueError("document chunk identity values must not be empty")
    if (doc_id is None) != (producer_chunk_id is None):
        raise ValueError("typed document identity requires doc_id and producer_chunk_id")
    if doc_id is None:
        if not session_id.strip():
            raise ValueError("legacy document identity requires session_id")
        return f"{session_id}:{storage_chunk_id}"
    if not doc_id.strip() or not producer_chunk_id or not producer_chunk_id.strip():
        raise ValueError("typed document identity values must not be empty")
    return f"{_PROJECT_DOCUMENT_IDENTITY_MARKER}:{doc_id}:{producer_chunk_id}"


def parse_mounted_document_chunk_source_unit_id(
    value: str,
) -> MountedDocumentChunkIdentity | None:
    """解析项目身份及两种已退役的 Session 绑定形态。"""

    project_marker = f"{_PROJECT_DOCUMENT_IDENTITY_MARKER}:"
    if value.startswith(project_marker):
        remainder = value[len(project_marker):]
        doc_id, separator, producer_chunk_id = remainder.partition(":")
        if not separator or not doc_id or not producer_chunk_id:
            return None
        return MountedDocumentChunkIdentity(
            doc_id=doc_id,
            producer_chunk_id=producer_chunk_id,
        )

    session_id, separator, remainder = value.partition(":")
    if not separator or not session_id or not remainder:
        return None
    marker = f"{_LEGACY_TYPED_DOCUMENT_IDENTITY_MARKER}:"
    if not remainder.startswith(marker):
        return MountedDocumentChunkIdentity(
            session_id=session_id,
            storage_chunk_id=remainder,
        )
    typed_remainder = remainder[len(marker):]
    doc_id, typed_separator, producer_chunk_id = typed_remainder.partition(":")
    if not typed_separator or not doc_id or not producer_chunk_id:
        return None
    return MountedDocumentChunkIdentity(
        session_id=session_id,
        doc_id=doc_id,
        producer_chunk_id=producer_chunk_id,
    )
