"""一个派生检索 generation 的规范身份与兼容门禁。

只有当 DataVersion 标明所有可能改变派生 Unit 的输入时，它才是有效权威。尤其是，仅有
编码器指纹并不能绑定文档块边界，也不能绑定这些 Unit 构建时采用的来源身份或索引契约。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import hashlib
import json

from ...workspace.pictures.observations import (
    DEFAULT_PICTURE_OBSERVATION_WINDOW_POLICY,
    PICTURE_OBSERVATION_PAYLOAD_CONTRACT,
    PictureObservationWindowPolicy,
)
from ..contracts import RetrievalMethod, SourceType
from .corpus import FILE_CORPUS, SESSION_CORPUS
from ..sqlite_store import (
    SCHEMA_VERSION as RETRIEVAL_CATALOG_SCHEMA_VERSION,
    RetrievalDataVersion,
    RetrievalDataVersionRole,
    RetrievalDataVersionState,
    SqliteRetrievalCatalog,
    UnitIndexState,
)


GENERATION_SPEC_CONTRACT = "retrieval-generation-spec-v2"
DOCUMENT_SOURCE_IDENTITY_CONTRACT = "project-document-source-identity-v3"
FILE_CORPUS_SOURCE_BINDINGS_CONTRACT = "file-corpus-source-bindings-v1"
PICTURE_OBSERVATION_SOURCE_IDENTITY_CONTRACT = (
    "picture-observation-source-identity-v1"
)
PICTURE_OBSERVATION_PROJECTION_FINGERPRINT_CONTRACT = (
    "picture-observation-retrieval-projection-v1"
)
DOCUMENT_LEXICAL_INDEX_RECIPE_CONTRACT = "rag-r1:bm25@1"
DOCUMENT_HYBRID_INDEX_RECIPE_CONTRACT = (
    "rag-r1:dense1024+learned_sparse+bm25@1"
)
SESSION_SOURCE_IDENTITY_CONTRACT = "current-session-retrieval-unit-v2"
SESSION_SOURCE_UNIT_CONTRACT_VERSION = 2


_DOCUMENT_INDEX_RECIPE_METHODS = {
    DOCUMENT_LEXICAL_INDEX_RECIPE_CONTRACT: (RetrievalMethod.BM25,),
    DOCUMENT_HYBRID_INDEX_RECIPE_CONTRACT: (
        RetrievalMethod.DENSE,
        RetrievalMethod.LEARNED_SPARSE,
        RetrievalMethod.BM25,
    ),
}


def document_index_methods(index_recipe: str) -> tuple[RetrievalMethod, ...]:
    """把封闭的 Document recipe 解析成必须逐 Unit 发布的方法。"""

    try:
        return _DOCUMENT_INDEX_RECIPE_METHODS[index_recipe]
    except (KeyError, TypeError) as exc:
        raise ValueError("unsupported document retrieval index recipe") from exc


def document_index_recipe(
    methods: tuple[RetrievalMethod, ...],
) -> str:
    """返回一个能力名实相符、顺序规范的 Document recipe。"""

    normalized = tuple(methods)
    if any(not isinstance(method, RetrievalMethod) for method in normalized):
        raise ValueError("document retrieval methods must be RetrievalMethod values")
    for recipe, expected in _DOCUMENT_INDEX_RECIPE_METHODS.items():
        if normalized == expected:
            return recipe
    raise ValueError("unsupported document retrieval method combination")


class RetrievalGenerationMismatch(RuntimeError):
    """当前目录无法安全提供某项运行时 generation 规格。"""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class RetrievalGenerationRestoreStatus(StrEnum):
    RESTORABLE = "restorable"
    REBUILD_REQUIRED = "rebuild_required"
    UNAVAILABLE = "unavailable"
    RESTORED = "restored"


@dataclass(frozen=True, slots=True)
class RetrievalGenerationRestorePlan:
    status: RetrievalGenerationRestoreStatus
    active_generation_id: str | None
    target_generation_id: str | None
    target_fingerprint: str | None
    reason_code: str | None
    ready_unit_count: int
    pending_unit_count: int
    failed_unit_count: int

    @property
    def can_restore(self) -> bool:
        return self.status is RetrievalGenerationRestoreStatus.RESTORABLE


@dataclass(frozen=True, slots=True)
class ExactGenerationDataVersionProvider:
    """隐藏已漂移活动派生 generation 的查询侧视图。"""

    catalog: SqliteRetrievalCatalog
    spec: 'RetrievalGenerationSpec'

    def active_data_version(self) -> RetrievalDataVersion | None:
        try:
            return require_exact_active_generation(self.catalog, self.spec)
        except RetrievalGenerationMismatch:
            return None

    def active_retrieval_data_version_id(self) -> str | None:
        """在不削弱身份约束的情况下实现查询侧 provider 端口。

        保留上方更丰富的投影有利于诊断，而 RetrievalService 会刻意在一次逻辑读取开始时
        只捕获不可变 ID。
        """

        active = self.active_data_version()
        return active.id if active is not None else None


@dataclass(frozen=True, slots=True)
class RetrievalGenerationSourceBinding:
    """One Source family's complete, content-free generation identity."""

    source_type: str
    source_identity_contract: str
    projection_contract: str
    projection_fingerprint: str

    def __post_init__(self) -> None:
        try:
            parsed_source_type = SourceType(self.source_type)
        except (TypeError, ValueError) as exc:
            raise ValueError("source_type must name a supported SourceType") from exc
        object.__setattr__(self, "source_type", parsed_source_type.value)
        for name in (
            "source_identity_contract",
            "projection_contract",
            "projection_fingerprint",
        ):
            value = getattr(self, name)
            if (
                not isinstance(value, str)
                or not value.strip()
                or len(value) > 1024
                or any(ord(character) < 32 for character in value)
            ):
                raise ValueError(f"{name} must be a bounded printable string")

    @property
    def canonical_value(self) -> dict[str, str]:
        return {
            "projection_contract": self.projection_contract,
            "projection_fingerprint": self.projection_fingerprint,
            "source_identity_contract": self.source_identity_contract,
            "source_type": self.source_type,
        }


@dataclass(frozen=True, slots=True)
class RetrievalGenerationSpec:
    """定义一个派生索引 generation 的全部不含内容输入。

    ``source_bindings`` 是 v2 规范权威：每个 Source 都必须分别冻结身份合同、投影合同和
    投影指纹。旧的 ``chunker`` / ``document`` 字段仍供 Document ingest 和单 Source
    调用方兼容读取，但不会再被误解为覆盖整个 File corpus。
    """

    encoder_fingerprint: str
    chunker_fingerprint: str
    document_chunk_contract_version: int
    retrieval_catalog_schema_version: int
    source_identity_contract: str
    index_recipe: str
    source_types: tuple[str, ...]
    source_bindings: tuple[RetrievalGenerationSourceBinding, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "encoder_fingerprint",
            "chunker_fingerprint",
            "source_identity_contract",
            "index_recipe",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
            if len(value) > 1024 or any(ord(character) < 32 for character in value):
                raise ValueError(f"{name} must be a bounded printable string")
        for name in (
            "document_chunk_contract_version",
            "retrieval_catalog_schema_version",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if (
            not isinstance(self.source_types, tuple)
            or not self.source_types
            or any(
                not isinstance(source_type, str) or not source_type.strip()
                for source_type in self.source_types
            )
            or self.source_types != tuple(sorted(set(self.source_types)))
        ):
            raise ValueError("source_types must be a non-empty canonical tuple")
        if not isinstance(self.source_bindings, tuple):
            raise TypeError("source_bindings must be an immutable tuple")
        bindings = self.source_bindings
        if any(
            not isinstance(binding, RetrievalGenerationSourceBinding)
            for binding in bindings
        ):
            raise TypeError(
                "source_bindings must contain RetrievalGenerationSourceBinding values"
            )
        if bindings:
            binding_types = tuple(binding.source_type for binding in bindings)
            if binding_types != tuple(sorted(set(binding_types))):
                raise ValueError("source_bindings must be canonical and unique")
            if binding_types != self.source_types:
                raise ValueError("source_bindings must exactly match source_types")
        elif len(self.source_types) != 1:
            raise ValueError("multi-source generations require explicit source_bindings")
        resolved = self.resolved_source_bindings
        document_binding = next(
            (
                binding
                for binding in resolved
                if binding.source_type == SourceType.DOCUMENT.value
            ),
            None,
        )
        if document_binding is not None and (
            document_binding.projection_contract
            != document_projection_contract(self.document_chunk_contract_version)
            or document_binding.projection_fingerprint != self.chunker_fingerprint
        ):
            raise ValueError(
                "Document compatibility fields must match its source binding"
            )
        if len(resolved) == 1:
            if resolved[0].source_identity_contract != self.source_identity_contract:
                raise ValueError(
                    "source_identity_contract must match the single source binding"
                )
        elif self.source_identity_contract != FILE_CORPUS_SOURCE_BINDINGS_CONTRACT:
            raise ValueError(
                "multi-source File generations require the aggregate binding contract"
            )

    @property
    def resolved_source_bindings(
        self,
    ) -> tuple[RetrievalGenerationSourceBinding, ...]:
        """Return explicit bindings, synthesizing only legacy single-source specs."""

        if self.source_bindings:
            return self.source_bindings
        source_type = self.source_types[0]
        return (
            RetrievalGenerationSourceBinding(
                source_type=source_type,
                source_identity_contract=self.source_identity_contract,
                projection_contract=(
                    document_projection_contract(
                        self.document_chunk_contract_version
                    )
                    if source_type == SourceType.DOCUMENT.value
                    else f"{source_type}-source-unit-v{self.document_chunk_contract_version}"
                ),
                projection_fingerprint=self.chunker_fingerprint,
            ),
        )

    def source_binding(
        self,
        source_type: SourceType,
    ) -> RetrievalGenerationSourceBinding:
        if not isinstance(source_type, SourceType):
            raise TypeError("source_type must be a SourceType")
        for binding in self.resolved_source_bindings:
            if binding.source_type == source_type.value:
                return binding
        raise ValueError("source_type is not part of this generation")

    @property
    def canonical_json(self) -> str:
        return json.dumps(
            {
                "contract": GENERATION_SPEC_CONTRACT,
                "encoder_fingerprint": self.encoder_fingerprint,
                "index_recipe": self.index_recipe,
                "retrieval_catalog_schema_version": self.retrieval_catalog_schema_version,
                "sources": [
                    binding.canonical_value
                    for binding in self.resolved_source_bindings
                ],
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    @property
    def fingerprint(self) -> str:
        digest = hashlib.sha256(self.canonical_json.encode("utf-8")).hexdigest()
        return f"{GENERATION_SPEC_CONTRACT}:{digest}"

    @property
    def projection_fingerprint(self) -> str:
        return self.chunker_fingerprint

    @property
    def source_unit_contract_version(self) -> int:
        return self.document_chunk_contract_version

    @property
    def version_id(self) -> str:
        digest = hashlib.sha256(self.canonical_json.encode("utf-8")).hexdigest()
        return f"rdv_{digest[:32]}"


def document_projection_contract(contract_version: int) -> str:
    if (
        isinstance(contract_version, bool)
        or not isinstance(contract_version, int)
        or contract_version <= 0
    ):
        raise ValueError("document projection contract version must be positive")
    return f"project-document-chunk-v{contract_version}"


def picture_observation_projection_fingerprint(
    *,
    observation_contract: str = PICTURE_OBSERVATION_PAYLOAD_CONTRACT,
    window_policy: PictureObservationWindowPolicy = (
        DEFAULT_PICTURE_OBSERVATION_WINDOW_POLICY
    ),
) -> str:
    """Bind Picture projection semantics to its observation payload and FIFO."""

    if not isinstance(observation_contract, str) or not observation_contract.strip():
        raise ValueError("observation_contract must be non-empty")
    if not isinstance(window_policy, PictureObservationWindowPolicy):
        raise TypeError("window_policy must be PictureObservationWindowPolicy")
    encoded = json.dumps(
        {
            "contract": PICTURE_OBSERVATION_PROJECTION_FINGERPRINT_CONTRACT,
            "fifo_policy": {
                "contract": "picture-observation-active-fifo-v1",
                "max_active_entries": window_policy.max_active_entries,
            },
            "observation_contract": observation_contract,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    return f"{PICTURE_OBSERVATION_PROJECTION_FINGERPRINT_CONTRACT}:{digest}"


def file_corpus_generation_spec(
    *,
    encoder_fingerprint: str,
    document_chunker_fingerprint: str,
    document_chunk_contract_version: int,
    index_recipe: str,
    picture_observation_contract: str = PICTURE_OBSERVATION_PAYLOAD_CONTRACT,
    picture_window_policy: PictureObservationWindowPolicy = (
        DEFAULT_PICTURE_OBSERVATION_WINDOW_POLICY
    ),
) -> RetrievalGenerationSpec:
    """Build the exact Document + Picture File corpus generation identity."""

    return RetrievalGenerationSpec(
        encoder_fingerprint=encoder_fingerprint,
        # Compatibility owner: Document ingest still reads these two fields.
        chunker_fingerprint=document_chunker_fingerprint,
        document_chunk_contract_version=document_chunk_contract_version,
        retrieval_catalog_schema_version=RETRIEVAL_CATALOG_SCHEMA_VERSION,
        source_identity_contract=FILE_CORPUS_SOURCE_BINDINGS_CONTRACT,
        index_recipe=document_index_recipe(document_index_methods(index_recipe)),
        source_types=FILE_CORPUS.generation_source_types,
        source_bindings=(
            RetrievalGenerationSourceBinding(
                source_type=SourceType.DOCUMENT.value,
                source_identity_contract=DOCUMENT_SOURCE_IDENTITY_CONTRACT,
                projection_contract=document_projection_contract(
                    document_chunk_contract_version
                ),
                projection_fingerprint=document_chunker_fingerprint,
            ),
            RetrievalGenerationSourceBinding(
                source_type=SourceType.PICTURE.value,
                source_identity_contract=(
                    PICTURE_OBSERVATION_SOURCE_IDENTITY_CONTRACT
                ),
                projection_contract=picture_observation_contract,
                projection_fingerprint=picture_observation_projection_fingerprint(
                    observation_contract=picture_observation_contract,
                    window_policy=picture_window_policy,
                ),
            ),
        ),
    )


def document_paper_generation_spec(
    *,
    encoder_fingerprint: str,
    chunker_fingerprint: str,
    document_chunk_contract_version: int,
    index_recipe: str,
) -> RetrievalGenerationSpec:
    """构建论文垂直链路使用的精确纯 Document generation。"""

    return RetrievalGenerationSpec(
        encoder_fingerprint=encoder_fingerprint,
        chunker_fingerprint=chunker_fingerprint,
        document_chunk_contract_version=document_chunk_contract_version,
        retrieval_catalog_schema_version=RETRIEVAL_CATALOG_SCHEMA_VERSION,
        source_identity_contract=DOCUMENT_SOURCE_IDENTITY_CONTRACT,
        index_recipe=document_index_recipe(document_index_methods(index_recipe)),
        source_types=("document",),
        source_bindings=(
            RetrievalGenerationSourceBinding(
                source_type=SourceType.DOCUMENT.value,
                source_identity_contract=DOCUMENT_SOURCE_IDENTITY_CONTRACT,
                projection_contract=document_projection_contract(
                    document_chunk_contract_version
                ),
                projection_fingerprint=chunker_fingerprint,
            ),
        ),
    )


def session_generation_spec(
    *,
    encoder_fingerprint: str,
    projection_fingerprint: str,
    index_recipe: str,
    source_unit_contract_version: int = SESSION_SOURCE_UNIT_CONTRACT_VERSION,
) -> RetrievalGenerationSpec:
    """为 L1 当前 Session 的派生检索单元构建独立 generation。"""

    # Session 与 Document 必须按同一个 recipe 写出同构检索单元。这里复用封闭的
    # Document recipe 校验器，仅复用方法组合，不复用 Document 来源身份。
    canonical_recipe = document_index_recipe(document_index_methods(index_recipe))
    return RetrievalGenerationSpec(
        encoder_fingerprint=encoder_fingerprint,
        chunker_fingerprint=projection_fingerprint,
        document_chunk_contract_version=source_unit_contract_version,
        retrieval_catalog_schema_version=RETRIEVAL_CATALOG_SCHEMA_VERSION,
        source_identity_contract=SESSION_SOURCE_IDENTITY_CONTRACT,
        index_recipe=canonical_recipe,
        source_types=SESSION_CORPUS.generation_source_types,
        source_bindings=(
            RetrievalGenerationSourceBinding(
                source_type=SourceType.CURRENT_SESSION.value,
                source_identity_contract=SESSION_SOURCE_IDENTITY_CONTRACT,
                projection_contract=(
                    f"current-session-source-unit-v{source_unit_contract_version}"
                ),
                projection_fingerprint=projection_fingerprint,
            ),
        ),
    )


def require_exact_active_generation(
    catalog: SqliteRetrievalCatalog,
    spec: RetrievalGenerationSpec,
) -> RetrievalDataVersion:
    """仅当完整身份匹配时返回活动且就绪的 generation。"""

    if not isinstance(spec, RetrievalGenerationSpec):
        raise TypeError("spec must be RetrievalGenerationSpec")
    active = catalog.active_data_version()
    if active is None:
        raise RetrievalGenerationMismatch("active_generation_missing")
    if (
        active.role is not RetrievalDataVersionRole.ACTIVE
        or active.state is not RetrievalDataVersionState.READY
    ):
        raise RetrievalGenerationMismatch("active_generation_not_ready")
    if active.id != spec.version_id or active.fingerprint != spec.fingerprint:
        raise RetrievalGenerationMismatch("active_generation_fingerprint_mismatch")
    return active


def plan_previous_generation_restore(
    catalog: SqliteRetrievalCatalog,
    *,
    target_generation_id: str | None = None,
    expected_fingerprint: str | None = None,
) -> RetrievalGenerationRestorePlan:
    """诊断一个精确 PREVIOUS generation 是否可以重新发布。

    提供 ``target_generation_id`` 时会显式选择目标；未提供目标时，选择最近激活的
    READY/PREVIOUS generation。此处不会重建来源或修改指针。
    """

    versions = catalog.list_data_versions()
    active = next(
        (item for item in versions if item.role is RetrievalDataVersionRole.ACTIVE),
        None,
    )
    previous = [
        item for item in versions if item.role is RetrievalDataVersionRole.PREVIOUS
    ]
    if target_generation_id is None:
        matching = (
            [item for item in previous if item.fingerprint == expected_fingerprint]
            if expected_fingerprint is not None
            else []
        )
        ready = [
            item for item in previous
            if item.state is RetrievalDataVersionState.READY
        ]
        target = (matching or ready or previous or [None])[0]
    else:
        target = next((item for item in versions if item.id == target_generation_id), None)
    if target is None:
        return RetrievalGenerationRestorePlan(
            status=RetrievalGenerationRestoreStatus.UNAVAILABLE,
            active_generation_id=active.id if active else None,
            target_generation_id=target_generation_id,
            target_fingerprint=None,
            reason_code="previous_generation_missing",
            ready_unit_count=0,
            pending_unit_count=0,
            failed_unit_count=0,
        )
    counts = catalog.data_version_index_state_counts(target.id)
    common = dict(
        active_generation_id=active.id if active else None,
        target_generation_id=target.id,
        target_fingerprint=target.fingerprint,
        ready_unit_count=counts[UnitIndexState.READY],
        pending_unit_count=counts[UnitIndexState.PENDING],
        failed_unit_count=counts[UnitIndexState.FAILED],
    )
    if target.role is not RetrievalDataVersionRole.PREVIOUS:
        return RetrievalGenerationRestorePlan(
            status=RetrievalGenerationRestoreStatus.UNAVAILABLE,
            reason_code="target_is_not_previous",
            **common,
        )
    if target.state is not RetrievalDataVersionState.READY:
        return RetrievalGenerationRestorePlan(
            status=RetrievalGenerationRestoreStatus.REBUILD_REQUIRED,
            reason_code="previous_generation_not_ready",
            **common,
        )
    if expected_fingerprint is not None and target.fingerprint != expected_fingerprint:
        return RetrievalGenerationRestorePlan(
            status=RetrievalGenerationRestoreStatus.REBUILD_REQUIRED,
            reason_code="previous_generation_fingerprint_mismatch",
            **common,
        )
    if not sum(counts.values()):
        return RetrievalGenerationRestorePlan(
            status=RetrievalGenerationRestoreStatus.REBUILD_REQUIRED,
            reason_code="previous_generation_empty",
            **common,
        )
    if counts[UnitIndexState.PENDING] or counts[UnitIndexState.FAILED]:
        return RetrievalGenerationRestorePlan(
            status=RetrievalGenerationRestoreStatus.REBUILD_REQUIRED,
            reason_code="previous_generation_index_incomplete",
            **common,
        )
    return RetrievalGenerationRestorePlan(
        status=RetrievalGenerationRestoreStatus.RESTORABLE,
        reason_code=None,
        **common,
    )
