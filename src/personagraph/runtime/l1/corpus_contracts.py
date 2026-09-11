"""由一个 L1 Turn 持有的来源语料库纯权威契约。"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import hmac
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


L1_CORPUS_MANIFEST_CONTRACT_VERSION = "l1-corpus-manifest-v3"
_SHA256_PATTERN = r"^[0-9a-f]{64}$"
_IDENTIFIER_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,199}$"


class L1CorpusManifestError(ValueError):
    """持久化语料库权威信息缺失、损坏或已不再是当前版本。"""


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class L1CorpusWorkspace(_Contract):
    """此 L1 TurnRun 的工作区权限边界，不声明目录内容已入库。"""

    boundary_fingerprint: str = Field(pattern=_SHA256_PATTERN)
    scope_snapshot_sha256: str = Field(pattern=_SHA256_PATTERN)


class L1CorpusAttachment(_Contract):
    """一个已接受的 Turn 附件及其精确 Project FileVersion。"""

    alias: str = Field(pattern=r"^attachment_[0-9]{3}$")
    attachment_id: str = Field(pattern=_IDENTIFIER_PATTERN)
    ordinal: int = Field(ge=0, le=255)
    input_message_id: str = Field(min_length=1, max_length=256)
    project_id: str = Field(pattern=_IDENTIFIER_PATTERN)
    file_id: str = Field(pattern=_IDENTIFIER_PATTERN)
    file_version_id: str = Field(pattern=_IDENTIFIER_PATTERN)
    name: str = Field(min_length=1, max_length=512)
    media_type: str = Field(min_length=1, max_length=160)
    suffix: str = Field(max_length=32)
    size_bytes: int = Field(ge=0)
    kind: Literal["image", "text", "document", "audio", "video", "unknown"]
    content_sha256: str = Field(pattern=_SHA256_PATTERN)
    access: Literal["on_demand"]
    origin: Literal["user_upload"]
    projection_identity_sha256: str = Field(pattern=_SHA256_PATTERN)

    @model_validator(mode="after")
    def _consistent_projection(self) -> 'L1CorpusAttachment':
        if self.suffix and not self.name.casefold().endswith(self.suffix):
            raise ValueError("attachment suffix does not match its name")
        return self


class L1CorpusManifest(_Contract):
    """在任何 L1 模型/工具 I/O 前捕获的不可变语料库权威信息。"""

    schema_version: Literal["l1-corpus-manifest-v3"] = (
        L1_CORPUS_MANIFEST_CONTRACT_VERSION
    )
    session_id: str = Field(pattern=_IDENTIFIER_PATTERN)
    turn_id: str = Field(pattern=_IDENTIFIER_PATTERN)
    l1_turn_run_id: str = Field(pattern=_IDENTIFIER_PATTERN)
    catalog_snapshot_sha256: str = Field(pattern=_SHA256_PATTERN)
    workspace: L1CorpusWorkspace | None = None
    attachment_count: int = Field(ge=0, le=256)
    attachments: tuple[L1CorpusAttachment, ...] = Field(
        default=(),
        max_length=256,
    )

    @model_validator(mode="after")
    def _closed_attachment_inventory(self) -> 'L1CorpusManifest':
        aliases = tuple(item.alias for item in self.attachments)
        identifiers = tuple(item.attachment_id for item in self.attachments)
        ordinals = tuple(item.ordinal for item in self.attachments)
        file_versions = tuple(
            (item.project_id, item.file_id, item.file_version_id)
            for item in self.attachments
        )
        if self.attachment_count != len(self.attachments):
            raise ValueError("attachment count changed")
        if ordinals != tuple(range(len(self.attachments))):
            raise ValueError("attachment ordinals must be complete and ordered")
        if len(aliases) != len(set(aliases)):
            raise ValueError("attachment aliases must be unique")
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("attachment IDs must be unique")
        if len(file_versions) != len(set(file_versions)):
            raise ValueError("attachment FileVersion bindings must be unique")
        return self


@dataclass(frozen=True, slots=True)
class FrozenL1CorpusManifest:
    manifest: L1CorpusManifest
    manifest_json: str
    manifest_sha256: str


def derive_l1_turn_run_id(*, session_id: str, turn_id: str) -> str:
    """在原子引导前推导可稳定重放的运行标识。"""

    identity = l1_corpus_canonical_json(
        {
            "contract": "l1-turn-run-id-v1",
            "session_id": session_id,
            "turn_id": turn_id,
        }
    )
    return f"l1run_{hashlib.sha256(identity.encode('utf-8')).hexdigest()}"


def freeze_l1_corpus_contract(
    manifest: L1CorpusManifest,
) -> FrozenL1CorpusManifest:
    payload = l1_corpus_canonical_json(manifest)
    return FrozenL1CorpusManifest(
        manifest=manifest,
        manifest_json=payload,
        manifest_sha256=hashlib.sha256(payload.encode("utf-8")).hexdigest(),
    )


def load_l1_corpus_manifest(
    payload: object,
    payload_hash: object,
    *,
    expected_session_id: str,
    expected_turn_id: str,
    expected_l1_turn_run_id: str,
) -> FrozenL1CorpusManifest:
    """认证一个持久化清单及其聚合所有权。"""

    if not isinstance(payload, str) or not payload:
        raise L1CorpusManifestError("L1 corpus manifest is unavailable")
    if not isinstance(payload_hash, str) or len(payload_hash) != 64:
        raise L1CorpusManifestError("L1 corpus manifest hash is unavailable")
    actual_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    if not hmac.compare_digest(actual_hash, payload_hash):
        raise L1CorpusManifestError("L1 corpus manifest hash changed")
    try:
        parsed = json.loads(payload)
        if l1_corpus_canonical_json(parsed) != payload:
            raise ValueError("manifest is not canonical")
        manifest = L1CorpusManifest.model_validate(parsed)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise L1CorpusManifestError("L1 corpus manifest is invalid") from exc
    expected = (
        expected_session_id,
        expected_turn_id,
        expected_l1_turn_run_id,
    )
    actual = (
        manifest.session_id,
        manifest.turn_id,
        manifest.l1_turn_run_id,
    )
    if actual != expected:
        raise L1CorpusManifestError("L1 corpus manifest ownership changed")
    return FrozenL1CorpusManifest(
        manifest=manifest,
        manifest_json=payload,
        manifest_sha256=payload_hash,
    )


def l1_corpus_canonical_json(value: object) -> str:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


__all__ = [
    'FrozenL1CorpusManifest',
    "L1_CORPUS_MANIFEST_CONTRACT_VERSION",
    'L1CorpusAttachment',
    "L1CorpusManifestError",
    'L1CorpusManifest',
    'L1CorpusWorkspace',
    "derive_l1_turn_run_id",
    "freeze_l1_corpus_contract",
    "l1_corpus_canonical_json",
    "load_l1_corpus_manifest",
]
