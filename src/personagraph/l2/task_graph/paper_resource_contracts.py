"""L2 TaskGraph 绑定论文资源快照的纯不可变契约。

这些值绑定一组小型 P1/P2 文档、检索生成事实以及有界的大纲元数据。
它们特意不包含 SQLite、Session、Runtime、检索或文档 I/O 行为。
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


PAPER_RESOURCE_SNAPSHOT_CONTRACT_VERSION = "paper-resource-snapshot-v1"
PAPER_RESOURCE_SNAPSHOT_ID_PREFIX = "prs_v1_"
MAX_PAPER_OUTLINE_ENTRIES = 64
_DIAGNOSTIC_CODE = re.compile(r"^[a-z][a-z0-9_.:-]{0,159}$")


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class PaperOutlineEntry(_Contract):
    """冻结论文版本中一个有界且可安全用于提示词的章节锚点。"""

    outline_key: str = Field(pattern=r"^P[12]:S[1-9][0-9]*$", max_length=40)
    title: str = Field(min_length=1, max_length=500)
    start_handle: str = Field(pattern=r"^P[12]:C[1-9][0-9]*$", max_length=40)
    page_start: int | None = Field(default=None, ge=1)
    page_end: int | None = Field(default=None, ge=1)

    @field_validator("title")
    @classmethod
    def _strip_title(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized or any(ord(character) < 32 for character in normalized):
            raise ValueError("outline title must be bounded printable text")
        return normalized

    @model_validator(mode="after")
    def _validate_anchor(self) -> 'PaperOutlineEntry':
        if self.outline_key.split(":", 1)[0] != self.start_handle.split(":", 1)[0]:
            raise ValueError("outline key and start handle must belong to one paper")
        if (self.page_start is None) != (self.page_end is None):
            raise ValueError("outline page range must provide both endpoints or neither")
        if (
            self.page_start is not None
            and self.page_end is not None
            and self.page_end < self.page_start
        ):
            raise ValueError("outline page range must be ordered")
        return self


class PaperDocumentBinding(_Contract):
    """Task 快照中一个 P1/P2 别名的私有精确权威信息。"""

    paper_key: str = Field(pattern=r"^P[12]$", max_length=2)
    document_id: str = Field(min_length=1, max_length=200)
    source_version_id: str = Field(min_length=1, max_length=200)
    title: str = Field(min_length=1, max_length=500)
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    processing_status: Literal["complete", "partial"]
    processing_diagnostic_codes: tuple[str, ...] = Field(default=(), max_length=32)
    admitted_chunk_count: int = Field(ge=1, le=5_000)
    chunk_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    admitted_text_page_start: int | None = Field(default=None, ge=1)
    admitted_text_page_end: int | None = Field(default=None, ge=1)
    outline: tuple[PaperOutlineEntry, ...] = Field(
        default=(),
        max_length=MAX_PAPER_OUTLINE_ENTRIES,
    )
    outline_truncated: bool = False

    @field_validator("document_id", "source_version_id", "title")
    @classmethod
    def _strip_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized or any(ord(character) < 32 for character in normalized):
            raise ValueError("paper document identity and title must be printable")
        return normalized

    @field_validator("processing_diagnostic_codes", mode="before")
    @classmethod
    def _canonical_diagnostic_codes(cls, value: Any) -> tuple[str, ...]:
        if not isinstance(value, (tuple, list)):
            raise ValueError("processing diagnostic codes must be a sequence")
        normalized: list[str] = []
        for item in value:
            if not isinstance(item, str):
                raise ValueError("processing diagnostic codes must be strings")
            code = item.strip()
            if not _DIAGNOSTIC_CODE.fullmatch(code):
                raise ValueError("processing diagnostics must be bounded stable codes")
            normalized.append(code)
        return tuple(sorted(set(normalized)))

    @model_validator(mode="after")
    def _validate_document_binding(self) -> 'PaperDocumentBinding':
        if self.processing_status == "complete" and self.processing_diagnostic_codes:
            raise ValueError("complete processing cannot carry diagnostic codes")
        if self.processing_status == "partial" and not self.processing_diagnostic_codes:
            raise ValueError("partial processing requires diagnostic codes")
        if (self.admitted_text_page_start is None) != (
            self.admitted_text_page_end is None
        ):
            raise ValueError("admitted text page range requires both endpoints or neither")
        if (
            self.admitted_text_page_start is not None
            and self.admitted_text_page_end is not None
            and self.admitted_text_page_end < self.admitted_text_page_start
        ):
            raise ValueError("admitted text page range must be ordered")

        expected_outline_keys = tuple(
            f"{self.paper_key}:S{ordinal}"
            for ordinal in range(1, len(self.outline) + 1)
        )
        if tuple(item.outline_key for item in self.outline) != expected_outline_keys:
            raise ValueError("outline keys must be contiguous and belong to the paper")
        previous_chunk_ordinal = 0
        for item in self.outline:
            if not item.start_handle.startswith(f"{self.paper_key}:C"):
                raise ValueError("outline handles must belong to the paper")
            chunk_ordinal = int(item.start_handle.rsplit("C", 1)[1])
            if chunk_ordinal <= previous_chunk_ordinal:
                raise ValueError("outline handles must be strictly ordered")
            if chunk_ordinal > self.admitted_chunk_count:
                raise ValueError("outline handle exceeds admitted chunk count")
            previous_chunk_ordinal = chunk_ordinal
            if (
                item.page_start is not None
                and self.admitted_text_page_start is not None
                and (
                    item.page_start < self.admitted_text_page_start
                    or item.page_end is None
                    or self.admitted_text_page_end is None
                    or item.page_end > self.admitted_text_page_end
                )
            ):
                raise ValueError("outline page range exceeds admitted text page range")
        return self


class PaperResourceSnapshot(_Contract):
    """在单个 Task 边界一次性冻结、可自验证的论文权威信息。"""

    contract_version: Literal["paper-resource-snapshot-v1"] = (
        PAPER_RESOURCE_SNAPSHOT_CONTRACT_VERSION
    )
    snapshot_id: str = Field(pattern=r"^prs_v1_[0-9a-f]{64}$")
    session_id: str = Field(min_length=1, max_length=200)
    task_id: str = Field(min_length=1, max_length=200)
    bound_graph_revision: int | None = Field(default=None, ge=1)
    bound_task_state_version: int = Field(ge=1)
    retrieval_data_version_id: str = Field(min_length=1, max_length=200)
    retrieval_generation_fingerprint: str = Field(min_length=1, max_length=1024)
    encoder_fingerprint: str = Field(min_length=1, max_length=1024)
    documents: tuple[PaperDocumentBinding, ...] = Field(min_length=1, max_length=2)
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator(
        "session_id",
        "task_id",
        "retrieval_data_version_id",
        "retrieval_generation_fingerprint",
        "encoder_fingerprint",
    )
    @classmethod
    def _strip_binding_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized or any(ord(character) < 32 for character in normalized):
            raise ValueError("snapshot binding strings must be printable and non-empty")
        return normalized

    @model_validator(mode="after")
    def _validate_manifest_identity(self) -> 'PaperResourceSnapshot':
        expected_keys = tuple(f"P{ordinal}" for ordinal in range(1, len(self.documents) + 1))
        if tuple(item.paper_key for item in self.documents) != expected_keys:
            raise ValueError("paper document keys must be ordered exactly as P1, P2")
        document_ids = tuple(item.document_id for item in self.documents)
        if len(document_ids) != len(set(document_ids)):
            raise ValueError("paper document IDs must be unique")
        expected_hash = _sha256(self.canonical_manifest_json)
        if self.manifest_sha256 != expected_hash:
            raise ValueError("manifest_sha256 does not match the canonical manifest")
        if self.snapshot_id != f"{PAPER_RESOURCE_SNAPSHOT_ID_PREFIX}{expected_hash}":
            raise ValueError("snapshot_id does not match manifest_sha256")
        return self

    @classmethod
    def create(
        cls,
        *,
        session_id: str,
        task_id: str,
        bound_graph_revision: int | None,
        bound_task_state_version: int,
        retrieval_data_version_id: str,
        retrieval_generation_fingerprint: str,
        encoder_fingerprint: str,
        documents: tuple[PaperDocumentBinding, ...],
    ) -> 'PaperResourceSnapshot':
        manifest = {
            "contract_version": PAPER_RESOURCE_SNAPSHOT_CONTRACT_VERSION,
            "session_id": session_id,
            "task_id": task_id,
            "bound_graph_revision": bound_graph_revision,
            "bound_task_state_version": bound_task_state_version,
            "retrieval_data_version_id": retrieval_data_version_id,
            "retrieval_generation_fingerprint": retrieval_generation_fingerprint,
            "encoder_fingerprint": encoder_fingerprint,
            "documents": [item.model_dump(mode="json") for item in documents],
        }
        manifest_sha256 = _sha256(_canonical_json(manifest))
        return cls(
            **manifest,
            snapshot_id=f"{PAPER_RESOURCE_SNAPSHOT_ID_PREFIX}{manifest_sha256}",
            manifest_sha256=manifest_sha256,
        )

    @property
    def canonical_manifest_json(self) -> str:
        return _canonical_json(
            self.model_dump(
                mode="json",
                exclude={"snapshot_id", "manifest_sha256"},
            )
        )

    @property
    def canonical_payload_json(self) -> str:
        return _canonical_json(self.model_dump(mode="json"))


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


__all__ = [
    "MAX_PAPER_OUTLINE_ENTRIES",
    "PAPER_RESOURCE_SNAPSHOT_CONTRACT_VERSION",
    "PAPER_RESOURCE_SNAPSHOT_ID_PREFIX",
    'PaperDocumentBinding',
    'PaperOutlineEntry',
    'PaperResourceSnapshot',
]
