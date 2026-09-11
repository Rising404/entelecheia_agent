"""AuxiliaryGraph 领域所有的 prompt 安全依赖投影。

AuxiliaryGraph 边携带持久完成身份，而非调用方文本。持久化层将该身份解析为已验证模型
OutputWindow 或 Host PlanningContextArtifact；本模块持有不可变、可自认证投影及其不可截断
模型输入边界。它不查询存储，也不选择哪些依赖属于节点。
"""

from __future__ import annotations

import hashlib
import json
from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from personagraph.l2.work_run import AuxiliaryNodeSubject, OutputWindowFormat
from .contracts import PlanningContextArtifactProjection


_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,199}$"
_SHA256_PATTERN = r"^[0-9a-f]{64}$"
_MAX_DEPENDENCY_CONTENT_CHARACTERS = 1_000_000


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class AuxiliaryDependencyInputLimits(_Contract):
    profile_id: str = Field(pattern=_ID_PATTERN)
    max_items: int = Field(ge=0, le=64)
    max_serialized_utf8_bytes: int = Field(ge=1, le=4_000_000)


class AuxiliaryDependencyInputTooLarge(RuntimeError):
    code = "auxiliary_v2_dependency_input_too_large"

    def __init__(
        self,
        *,
        item_count: int,
        serialized_utf8_bytes: int,
        limits: AuxiliaryDependencyInputLimits,
    ) -> None:
        self.item_count = item_count
        self.serialized_utf8_bytes = serialized_utf8_bytes
        self.limits = limits
        super().__init__(
            "AuxiliaryGraph dependency input exceeds its Host profile: "
            f"items={item_count}/{limits.max_items}, "
            "serialized_utf8_bytes="
            f"{serialized_utf8_bytes}/{limits.max_serialized_utf8_bytes}, "
            f"profile_id={limits.profile_id}"
        )


class AuxiliaryDependencyInputUnsupported(RuntimeError):
    code = "auxiliary_v2_dependency_input_unsupported"


class AuxiliaryModelOutputDependency(_Contract):
    schema_version: Literal["auxiliary-v2-model-output-dependency-v1"] = (
        "auxiliary-v2-model-output-dependency-v1"
    )
    dependency_kind: Literal["model_output"] = "model_output"
    completion_id: str = Field(pattern=_ID_PATTERN)
    producer_subject: AuxiliaryNodeSubject
    producer_node_alias: str = Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")
    producer_ordinal: int = Field(ge=0, le=63)
    output_contract: str = Field(pattern=_ID_PATTERN)
    output_revision: int = Field(ge=1)
    output_format: OutputWindowFormat
    content: str = Field(min_length=1, max_length=_MAX_DEPENDENCY_CONTENT_CHARACTERS)
    output_snapshot_sha256: str = Field(pattern=_SHA256_PATTERN)
    verification_result_sha256: str = Field(pattern=_SHA256_PATTERN)
    completion_sha256: str = Field(pattern=_SHA256_PATTERN)
    dependency_item_sha256: str = Field(pattern=_SHA256_PATTERN)

    @model_validator(mode="after")
    def _validate_item(self) -> "AuxiliaryModelOutputDependency":
        if not self.content.strip() or "\x00" in self.content:
            raise ValueError("model dependency content must be non-blank non-NUL text")
        if self.dependency_item_sha256 != _sha256_value(
            self.model_dump(mode="json", exclude={"dependency_item_sha256"})
        ):
            raise ValueError("model dependency item hash is invalid")
        return self

    @classmethod
    def create(cls, **values: object) -> Self:
        values = dict(values)
        subject = values.get("producer_subject")
        if not isinstance(subject, AuxiliaryNodeSubject):
            values["producer_subject"] = AuxiliaryNodeSubject.model_validate(
                subject
            )
        values["output_format"] = OutputWindowFormat(values["output_format"])
        values["dependency_item_sha256"] = "0" * 64
        provisional = cls.model_construct(**values)
        values["dependency_item_sha256"] = _sha256_value(
            provisional.model_dump(
                mode="json",
                exclude={"dependency_item_sha256"},
            )
        )
        return cls.model_validate(values)


class AuxiliaryHostContextDependency(_Contract):
    schema_version: Literal["auxiliary-v2-host-context-dependency-v1"] = (
        "auxiliary-v2-host-context-dependency-v1"
    )
    dependency_kind: Literal["host_context"] = "host_context"
    completion_id: str = Field(pattern=_ID_PATTERN)
    producer_subject: AuxiliaryNodeSubject
    producer_node_alias: str = Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")
    producer_ordinal: int = Field(ge=0, le=63)
    output_contract: str = Field(pattern=_ID_PATTERN)
    artifact: PlanningContextArtifactProjection
    observation_settlement_sha256: str = Field(pattern=_SHA256_PATTERN)
    verification_receipt_sha256: str = Field(pattern=_SHA256_PATTERN)
    dependency_item_sha256: str = Field(pattern=_SHA256_PATTERN)

    @model_validator(mode="after")
    def _validate_item(self) -> "AuxiliaryHostContextDependency":
        if (
            self.artifact.artifact_id != self.completion_id
            or self.artifact.producer_node_alias != self.producer_node_alias
        ):
            raise ValueError("Host dependency artifact differs from its producer")
        if self.dependency_item_sha256 != _sha256_value(
            self.model_dump(mode="json", exclude={"dependency_item_sha256"})
        ):
            raise ValueError("Host dependency item hash is invalid")
        return self

    @classmethod
    def create(cls, **values: object) -> Self:
        values = dict(values)
        subject = values.get("producer_subject")
        if not isinstance(subject, AuxiliaryNodeSubject):
            values["producer_subject"] = AuxiliaryNodeSubject.model_validate(
                subject
            )
        artifact = values.get("artifact")
        if not isinstance(artifact, PlanningContextArtifactProjection):
            values["artifact"] = PlanningContextArtifactProjection.model_validate(
                artifact
            )
        values["dependency_item_sha256"] = "0" * 64
        provisional = cls.model_construct(**values)
        values["dependency_item_sha256"] = _sha256_value(
            provisional.model_dump(
                mode="json",
                exclude={"dependency_item_sha256"},
            )
        )
        return cls.model_validate(values)


AuxiliaryDependencyItem = Annotated[
    AuxiliaryModelOutputDependency | AuxiliaryHostContextDependency,
    Field(discriminator="dependency_kind"),
]


class AuxiliaryDependencyBundle(_Contract):
    schema_version: Literal["auxiliary-v2-dependency-bundle-v1"] = (
        "auxiliary-v2-dependency-bundle-v1"
    )
    session_id: str = Field(pattern=_ID_PATTERN)
    task_id: str = Field(pattern=_ID_PATTERN)
    auxiliary_graph_id: str = Field(pattern=_ID_PATTERN)
    auxiliary_graph_revision: int = Field(ge=1)
    consumer_subject: AuxiliaryNodeSubject
    consumer_node_alias: str = Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")
    structure_sha256: str = Field(pattern=_SHA256_PATTERN)
    dependency_completion_ids: tuple[str, ...] = Field(default=(), max_length=64)
    items: tuple[AuxiliaryDependencyItem, ...] = Field(
        default=(),
        max_length=64,
    )
    projection_sha256: str = Field(pattern=_SHA256_PATTERN)

    @model_validator(mode="after")
    def _validate_bundle(self) -> "AuxiliaryDependencyBundle":
        if (
            self.consumer_subject.task_id != self.task_id
            or self.consumer_subject.auxiliary_graph_id != self.auxiliary_graph_id
            or self.consumer_subject.auxiliary_graph_revision
            != self.auxiliary_graph_revision
        ):
            raise ValueError("dependency consumer differs from its graph")
        completion_ids = tuple(item.completion_id for item in self.items)
        if completion_ids != self.dependency_completion_ids:
            raise ValueError("dependency completion order differs from its items")
        producer_subjects = tuple(item.producer_subject for item in self.items)
        if (
            len(completion_ids) != len(set(completion_ids))
            or len(producer_subjects) != len(set(producer_subjects))
        ):
            raise ValueError("dependency completions and producers must be unique")
        if any(
            subject.task_id != self.task_id
            or subject.auxiliary_graph_id != self.auxiliary_graph_id
            or subject.auxiliary_graph_revision != self.auxiliary_graph_revision
            or subject == self.consumer_subject
            for subject in producer_subjects
        ):
            raise ValueError("dependency producer is outside the consumer graph")
        order = tuple(
            (item.producer_ordinal, item.producer_subject.node_id)
            for item in self.items
        )
        if order != tuple(sorted(order)):
            raise ValueError("Auxiliary dependencies must use stable producer order")
        if self.projection_sha256 != _sha256_value(
            self.model_dump(mode="json", exclude={"projection_sha256"})
        ):
            raise ValueError("dependency bundle projection hash is invalid")
        return self

    @classmethod
    def create(cls, **values: object) -> Self:
        values = dict(values)
        values["items"] = tuple(values.get("items", ()))
        values["dependency_completion_ids"] = tuple(
            item.completion_id for item in values["items"]  # type: ignore[union-attr]
        )
        values["projection_sha256"] = "0" * 64
        provisional = cls.model_construct(**values)
        values["projection_sha256"] = _sha256_value(
            provisional.model_dump(mode="json", exclude={"projection_sha256"})
        )
        return cls.model_validate(values)


def build_auxiliary_dependency_model_payload(
    bundle: AuxiliaryDependencyBundle,
) -> dict[str, Any]:
    """将精确已验证依赖正文投影为显式不可信数据。"""

    if not isinstance(bundle, AuxiliaryDependencyBundle):
        raise TypeError("bundle must be AuxiliaryDependencyBundle")
    return {
        "schema_version": "auxiliary-v2-dependency-model-payload-v1",
        "untrusted_dependency_data": True,
        "consumer_node_alias": bundle.consumer_node_alias,
        "structure_sha256": bundle.structure_sha256,
        "projection_sha256": bundle.projection_sha256,
        "items": [item.model_dump(mode="json") for item in bundle.items],
    }


def serialize_auxiliary_dependency_model_payload(
    bundle: AuxiliaryDependencyBundle,
    *,
    limits: AuxiliaryDependencyInputLimits,
) -> str:
    """序列化完整依赖 bundle；若无法容纳则不截断并直接拒绝。"""

    try:
        serialized = _canonical_json(
            build_auxiliary_dependency_model_payload(bundle)
        )
        serialized_bytes = len(serialized.encode("utf-8"))
    except (TypeError, ValueError, OverflowError, UnicodeError, RecursionError) as exc:
        raise AuxiliaryDependencyInputUnsupported(
            "AuxiliaryGraph dependency input is not canonical UTF-8 JSON"
        ) from exc
    if (
        len(bundle.items) > limits.max_items
        or serialized_bytes > limits.max_serialized_utf8_bytes
    ):
        raise AuxiliaryDependencyInputTooLarge(
            item_count=len(bundle.items),
            serialized_utf8_bytes=serialized_bytes,
            limits=limits,
        )
    return serialized


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256_value(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


__all__ = [
    "AuxiliaryDependencyBundle",
    "AuxiliaryDependencyInputLimits",
    "AuxiliaryDependencyInputTooLarge",
    "AuxiliaryDependencyInputUnsupported",
    "AuxiliaryDependencyItem",
    "AuxiliaryHostContextDependency",
    "AuxiliaryModelOutputDependency",
    "build_auxiliary_dependency_model_payload",
    "serialize_auxiliary_dependency_model_payload",
]
