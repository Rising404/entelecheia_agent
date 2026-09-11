"""Project-document Retrieval 的唯一生产配置与能力门。

Profile 统一拥有会改变派生 generation 身份或运行能力的选择：首阶段方法、本地模型及固定
revision、device/precision、reranker、失败策略和文档 chunking recipe。Document ingestion 与
Retrieval composition 从同一个 profile 取得这些参数，避免解析、索引和查询各自维护一份默认值。

导入本模块不会下载或加载模型；生产构建只接受已预检的本地资产，缺失能力按显式 failure
policy 失败或降级。切换 lexical/BGE 表示会要求重建派生索引，但不会静默修改 Source 文档权威
内容。Current Session 可以复用方法 recipe，却仍拥有独立 corpus、generation 与授权范围。

它保留在包根，因为这是产品配置到多个 Retrieval 子域的组合边界，而不是某个 encoder 或
orchestration 实现的私有设置。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from enum import Enum
from functools import lru_cache
import importlib.util
import os
from pathlib import Path
from typing import Final

from personagraph.input_processing.documents.chunking import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_MIN_TOKENS,
    DEFAULT_SPLIT_OVERLAP_TOKENS,
    DEFAULT_TARGET_TOKENS,
    ChunkingProfile,
)

from .contracts import RetrievalMethod
from .compute.devices import (
    DeviceSelection,
    normalize_device_preference,
    select_device,
)
from .lifecycle.generation import (
    DOCUMENT_HYBRID_INDEX_RECIPE_CONTRACT,
    DOCUMENT_LEXICAL_INDEX_RECIPE_CONTRACT,
)
from .indexing.encoder import BgeM3Encoder, DeterministicLexicalEncoder
from .indexing.model_assets import (
    BGE_M3_HYBRID_MANIFEST,
    BGE_M3_MODEL_ID,
    BGE_M3_MODEL_REVISION,
    BGE_V2_M3_RERANKER_MANIFEST,
    BGE_V2_M3_RERANKER_MODEL_ID,
    BGE_V2_M3_RERANKER_REVISION,
    LocalModelAssetRef,
    ModelAssetCapability,
    preflight_local_model_asset,
)
from .ports import BgeM3EncoderPort, RerankerPort
from .orchestration.reranking import BgeM3Reranker


RETRIEVAL_PROFILE_ENV: Final = "PERSONAGRAPH_RETRIEVAL_PROFILE"
RETRIEVAL_METHODS_ENV: Final = "PERSONAGRAPH_RETRIEVAL_METHODS"
RETRIEVAL_FAILURE_POLICY_ENV: Final = "PERSONAGRAPH_RETRIEVAL_FAILURE_POLICY"
RETRIEVAL_DEVICE_ENV: Final = "PERSONAGRAPH_RETRIEVAL_DEVICE"
RETRIEVAL_USE_FP16_ENV: Final = "PERSONAGRAPH_RETRIEVAL_USE_FP16"
RETRIEVAL_LOCAL_ONLY_ENV: Final = "PERSONAGRAPH_RETRIEVAL_LOCAL_FILES_ONLY"
RETRIEVAL_BGE_MODEL_ENV: Final = "PERSONAGRAPH_RETRIEVAL_BGE_MODEL"
RETRIEVAL_BGE_REVISION_ENV: Final = "PERSONAGRAPH_RETRIEVAL_BGE_REVISION"
RETRIEVAL_BGE_IDENTITY_ENV: Final = "PERSONAGRAPH_RETRIEVAL_BGE_IDENTITY"
RETRIEVAL_RERANKER_ENV: Final = "PERSONAGRAPH_RETRIEVAL_RERANKER"
RETRIEVAL_RERANKER_MODEL_ENV: Final = "PERSONAGRAPH_RETRIEVAL_RERANKER_MODEL"
RETRIEVAL_RERANKER_REVISION_ENV: Final = (
    "PERSONAGRAPH_RETRIEVAL_RERANKER_REVISION"
)
RETRIEVAL_RERANKER_IDENTITY_ENV: Final = (
    "PERSONAGRAPH_RETRIEVAL_RERANKER_IDENTITY"
)


class RetrievalProfileUnavailable(RuntimeError):
    """The configured production profile failed its local capability gate."""


class RetrievalProfileMode(str, Enum):
    LEXICAL = "lexical"
    BGE_M3 = "bge_m3"


class RetrievalFailurePolicy(str, Enum):
    STRICT = "strict"
    LEXICAL_FALLBACK = "lexical_fallback"


class RetrievalRerankerMode(str, Enum):
    OFF = "off"
    BGE_V2_M3 = "bge_v2_m3"


def _default_encoder_asset() -> LocalModelAssetRef:
    return LocalModelAssetRef.hub(
        BGE_M3_MODEL_ID,
        revision=BGE_M3_MODEL_REVISION,
    )


def _default_reranker_asset() -> LocalModelAssetRef:
    return LocalModelAssetRef.hub(
        BGE_V2_M3_RERANKER_MODEL_ID,
        revision=BGE_V2_M3_RERANKER_REVISION,
    )


@dataclass(frozen=True, slots=True)
class DocumentRetrievalProfile:
    """Typed identity and failure policy for production Document retrieval."""

    mode: RetrievalProfileMode = RetrievalProfileMode.BGE_M3
    failure_policy: RetrievalFailurePolicy = RetrievalFailurePolicy.STRICT
    reranker_mode: RetrievalRerankerMode = RetrievalRerankerMode.BGE_V2_M3
    encoder_asset: LocalModelAssetRef = field(default_factory=_default_encoder_asset)
    reranker_asset: LocalModelAssetRef = field(default_factory=_default_reranker_asset)
    device: str = "auto"
    use_fp16: bool = False
    local_files_only: bool = True
    retrieval_methods_override: tuple[RetrievalMethod, ...] | None = None

    def __post_init__(self) -> None:
        device = normalize_device_preference(self.device)
        object.__setattr__(self, "device", device)
        if self.local_files_only is not True:
            raise ValueError("production retrieval must be local-only")
        if self.use_fp16 and device in {"auto", "cpu"}:
            raise ValueError(
                f"fp16 retrieval is not supported on the {device} profile"
            )
        if (
            self.mode is RetrievalProfileMode.LEXICAL
            and self.reranker_mode is not RetrievalRerankerMode.OFF
        ):
            raise ValueError("the lexical profile cannot enable a model reranker")
        _validate_retrieval_methods(
            mode=self.mode,
            methods=self.retrieval_methods_override,
        )

    @classmethod
    def production(cls) -> 'DocumentRetrievalProfile':
        """Strict local BGE-M3 hybrid retrieval with BGE reranking."""

        return cls()

    @classmethod
    def lexical(cls) -> 'DocumentRetrievalProfile':
        """Dependency-free rollback/test profile; never selected implicitly."""

        return cls(
            mode=RetrievalProfileMode.LEXICAL,
            failure_policy=RetrievalFailurePolicy.STRICT,
            reranker_mode=RetrievalRerankerMode.OFF,
        )

    @classmethod
    def from_environment(
        cls,
        environ: Mapping[str, str] | None = None,
    ) -> 'DocumentRetrievalProfile':
        values = os.environ if environ is None else environ
        local_only = _parse_bool(
            values.get(RETRIEVAL_LOCAL_ONLY_ENV, "true"),
            name=RETRIEVAL_LOCAL_ONLY_ENV,
        )
        if not local_only:
            raise ValueError(f"{RETRIEVAL_LOCAL_ONLY_ENV} must remain true")
        mode = _parse_enum(
            RetrievalProfileMode,
            values.get(RETRIEVAL_PROFILE_ENV, RetrievalProfileMode.BGE_M3.value),
            name=RETRIEVAL_PROFILE_ENV,
        )
        failure_policy = _parse_enum(
            RetrievalFailurePolicy,
            values.get(
                RETRIEVAL_FAILURE_POLICY_ENV,
                RetrievalFailurePolicy.STRICT.value,
            ),
            name=RETRIEVAL_FAILURE_POLICY_ENV,
        )
        methods_override = _retrieval_methods_from_environment(values, mode=mode)
        if mode is RetrievalProfileMode.LEXICAL:
            reranker_value = values.get(
                RETRIEVAL_RERANKER_ENV,
                RetrievalRerankerMode.OFF.value,
            )
        else:
            reranker_value = values.get(
                RETRIEVAL_RERANKER_ENV,
                RetrievalRerankerMode.BGE_V2_M3.value,
            )
        reranker_mode = _parse_enum(
            RetrievalRerankerMode,
            reranker_value,
            name=RETRIEVAL_RERANKER_ENV,
        )
        if mode is RetrievalProfileMode.LEXICAL:
            return cls(
                mode=mode,
                failure_policy=failure_policy,
                reranker_mode=reranker_mode,
                device="cpu",
                use_fp16=False,
                local_files_only=local_only,
                retrieval_methods_override=methods_override,
            )
        return cls(
            mode=mode,
            failure_policy=failure_policy,
            reranker_mode=reranker_mode,
            encoder_asset=_asset_from_environment(
                values,
                model_env=RETRIEVAL_BGE_MODEL_ENV,
                revision_env=RETRIEVAL_BGE_REVISION_ENV,
                identity_env=RETRIEVAL_BGE_IDENTITY_ENV,
                default_model=BGE_M3_MODEL_ID,
                default_revision=BGE_M3_MODEL_REVISION,
            ),
            reranker_asset=_asset_from_environment(
                values,
                model_env=RETRIEVAL_RERANKER_MODEL_ENV,
                revision_env=RETRIEVAL_RERANKER_REVISION_ENV,
                identity_env=RETRIEVAL_RERANKER_IDENTITY_ENV,
                default_model=BGE_V2_M3_RERANKER_MODEL_ID,
                default_revision=BGE_V2_M3_RERANKER_REVISION,
            ),
            device=str(values.get(RETRIEVAL_DEVICE_ENV, "auto")),
            use_fp16=_parse_bool(
                values.get(RETRIEVAL_USE_FP16_ENV, "false"),
                name=RETRIEVAL_USE_FP16_ENV,
            ),
            local_files_only=local_only,
            retrieval_methods_override=methods_override,
        )

    @property
    def retrieval_methods(self) -> tuple[RetrievalMethod, ...]:
        if self.retrieval_methods_override is not None:
            return self.retrieval_methods_override
        if self.mode is RetrievalProfileMode.LEXICAL:
            return (RetrievalMethod.BM25,)
        return (
            RetrievalMethod.DENSE,
            RetrievalMethod.LEARNED_SPARSE,
            RetrievalMethod.BM25,
        )

    @property
    def index_recipe(self) -> str:
        if self.mode is RetrievalProfileMode.LEXICAL:
            return DOCUMENT_LEXICAL_INDEX_RECIPE_CONTRACT
        return DOCUMENT_HYBRID_INDEX_RECIPE_CONTRACT

    def fingerprint(self) -> str:
        encoder = (
            "deterministic_lexical@2"
            if self.mode is RetrievalProfileMode.LEXICAL
            else self.encoder_asset.identity
        )
        reranker = (
            "off"
            if self.reranker_mode is RetrievalRerankerMode.OFF
            else self.reranker_asset.identity
        )
        return (
            f"document_retrieval_profile@1:mode={self.mode.value};"
            f"methods={','.join(method.value for method in self.retrieval_methods)};"
            f"index_recipe={self.index_recipe};encoder={encoder};"
            f"reranker={reranker};device={self.device};"
            f"fp16={str(self.use_fp16).lower()};"
            f"failure_policy={self.failure_policy.value};local_only=true"
        )


@dataclass(frozen=True, slots=True)
class DocumentRetrievalCapability:
    profile_fingerprint: str
    ready: bool
    reason_codes: tuple[str, ...]
    encoder_asset: ModelAssetCapability | None
    reranker_asset: ModelAssetCapability | None
    device_ready: bool
    sqlite_vec_ready: bool
    device_selection: DeviceSelection | None = None

    def require_ready(self) -> None:
        if self.ready:
            return
        reasons = ",".join(self.reason_codes) or "retrieval_profile_unavailable"
        raise RetrievalProfileUnavailable(reasons)

    def diagnostic_snapshot(self) -> dict[str, object]:
        return {
            "profile_fingerprint": self.profile_fingerprint,
            "ready": self.ready,
            "reason_codes": self.reason_codes,
            "device_ready": self.device_ready,
            "device_selection": (
                self.device_selection.diagnostic_snapshot()
                if self.device_selection is not None
                else None
            ),
            "sqlite_vec_ready": self.sqlite_vec_ready,
            "encoder_asset": (
                self.encoder_asset.diagnostic_snapshot()
                if self.encoder_asset is not None
                else None
            ),
            "reranker_asset": (
                self.reranker_asset.diagnostic_snapshot()
                if self.reranker_asset is not None
                else None
            ),
            "local_files_only": True,
        }


@dataclass(frozen=True, slots=True)
class DocumentRetrievalRuntime:
    requested_profile: DocumentRetrievalProfile
    effective_profile: DocumentRetrievalProfile
    capability: DocumentRetrievalCapability
    encoder: BgeM3EncoderPort
    reranker: RerankerPort | None
    chunking_profile: ChunkingProfile
    degraded_reason: str | None = None


def preflight_document_retrieval_profile(
    profile: DocumentRetrievalProfile,
) -> DocumentRetrievalCapability:
    """Check dependencies, device and complete pinned assets without loading weights."""

    device_selection = select_device(profile.device)
    if profile.mode is RetrievalProfileMode.LEXICAL:
        return DocumentRetrievalCapability(
            profile_fingerprint=profile.fingerprint(),
            ready=True,
            reason_codes=(),
            encoder_asset=None,
            reranker_asset=None,
            device_ready=True,
            sqlite_vec_ready=True,
            device_selection=device_selection,
        )

    encoder_asset = preflight_local_model_asset(
        profile.encoder_asset,
        BGE_M3_HYBRID_MANIFEST,
    )
    reranker_asset = (
        preflight_local_model_asset(
            profile.reranker_asset,
            BGE_V2_M3_RERANKER_MANIFEST,
        )
        if profile.reranker_mode is RetrievalRerankerMode.BGE_V2_M3
        else None
    )
    device_ready = device_selection.ready
    sqlite_vec_ready = importlib.util.find_spec("sqlite_vec") is not None
    reasons: list[str] = []
    if not encoder_asset.ready:
        reasons.append(f"encoder:{encoder_asset.reason_code}")
    if reranker_asset is not None and not reranker_asset.ready:
        reasons.append(f"reranker:{reranker_asset.reason_code}")
    if not device_ready:
        reasons.append("retrieval_device_unavailable")
    if not sqlite_vec_ready:
        reasons.append("sqlite_vec_unavailable")
    return DocumentRetrievalCapability(
        profile_fingerprint=profile.fingerprint(),
        ready=not reasons,
        reason_codes=tuple(reasons),
        encoder_asset=encoder_asset,
        reranker_asset=reranker_asset,
        device_ready=device_ready,
        sqlite_vec_ready=sqlite_vec_ready,
        device_selection=device_selection,
    )


def build_document_retrieval_runtime(
    profile: DocumentRetrievalProfile | None = None,
) -> DocumentRetrievalRuntime:
    """Preflight then build the only supported production retrieval composition."""

    requested = profile or DocumentRetrievalProfile.from_environment()
    capability = preflight_document_retrieval_profile(requested)
    if not capability.ready:
        if requested.failure_policy is RetrievalFailurePolicy.STRICT:
            capability.require_ready()
        fallback = DocumentRetrievalProfile.lexical()
        return DocumentRetrievalRuntime(
            requested_profile=requested,
            effective_profile=fallback,
            capability=capability,
            encoder=DeterministicLexicalEncoder(),
            reranker=None,
            chunking_profile=production_document_chunking_profile(
                fallback,
                encoder=DeterministicLexicalEncoder(),
            ),
            degraded_reason=",".join(capability.reason_codes),
        )
    if requested.mode is RetrievalProfileMode.LEXICAL:
        effective = requested
        encoder: BgeM3EncoderPort = DeterministicLexicalEncoder()
        reranker: RerankerPort | None = None
    else:
        selection = capability.device_selection
        if selection is None or selection.device is None:
            raise RetrievalProfileUnavailable("retrieval_device_unavailable")
        effective = (
            requested
            if requested.device == selection.device
            else replace(requested, device=selection.device)
        )
        assert capability.encoder_asset is not None
        encoder = BgeM3Encoder(
            asset=effective.encoder_asset,
            resolved_model_path=capability.encoder_asset.require_ready(),
            device=effective.device,
            use_fp16=effective.use_fp16,
        )
        if effective.reranker_mode is RetrievalRerankerMode.BGE_V2_M3:
            assert capability.reranker_asset is not None
            reranker = BgeM3Reranker(
                asset=effective.reranker_asset,
                resolved_model_path=capability.reranker_asset.require_ready(),
                device=effective.device,
                use_fp16=effective.use_fp16,
                allow_cpu_fallback=(
                    requested.device == "auto"
                    and effective.device != "cpu"
                    and not requested.use_fp16
                ),
            )
        else:
            reranker = None
    return DocumentRetrievalRuntime(
        requested_profile=requested,
        effective_profile=effective,
        capability=capability,
        encoder=encoder,
        reranker=reranker,
        chunking_profile=production_document_chunking_profile(
            effective,
            encoder=encoder,
        ),
    )


def production_document_chunking_profile(
    profile: DocumentRetrievalProfile | None = None,
    *,
    encoder: BgeM3EncoderPort | None = None,
) -> ChunkingProfile:
    """Build the structure-aware recipe from the effective retrieval tokenizer."""

    selected = profile or DocumentRetrievalProfile.from_environment()
    retrieval_encoder = encoder
    if retrieval_encoder is None:
        if selected.mode is RetrievalProfileMode.LEXICAL:
            retrieval_encoder = DeterministicLexicalEncoder()
        else:
            capability = preflight_local_model_asset(
                selected.encoder_asset,
                BGE_M3_HYBRID_MANIFEST,
            )
            # Chunk boundaries only depend on the pinned tokenizer.  Keep this
            # narrow path independent of accelerator, reranker and sqlite-vec
            # readiness; vector generation still uses the resolved device.
            retrieval_encoder = BgeM3Encoder(
                asset=selected.encoder_asset,
                resolved_model_path=capability.require_ready(),
                device="cpu",
                use_fp16=False,
            )
    token_offsets = getattr(retrieval_encoder, "token_offsets", None)
    return ChunkingProfile(
        target_tokens=DEFAULT_TARGET_TOKENS,
        max_tokens=DEFAULT_MAX_TOKENS,
        min_tokens=DEFAULT_MIN_TOKENS,
        split_overlap_tokens=DEFAULT_SPLIT_OVERLAP_TOKENS,
        count_tokens=lambda text: len(tuple(retrieval_encoder.token_ids(text))),
        token_offsets=(token_offsets if callable(token_offsets) else None),
        tokenizer_id=retrieval_encoder.tokenizer_fingerprint(),
    )


@lru_cache(maxsize=1)
def durable_chunking_profile() -> ChunkingProfile:
    """返回后台摄取与附件准入共享的默认 production recipe。"""

    return production_document_chunking_profile()


def _asset_from_environment(
    values: Mapping[str, str],
    *,
    model_env: str,
    revision_env: str,
    identity_env: str,
    default_model: str,
    default_revision: str,
) -> LocalModelAssetRef:
    model = str(values.get(model_env, default_model)).strip()
    if _looks_like_path(model):
        identity = str(values.get(identity_env, "")).strip()
        if not identity:
            raise ValueError(
                f"{identity_env} is required when {model_env} is a local path"
            )
        return LocalModelAssetRef.path(model, canonical_identity=identity)
    revision = str(values.get(revision_env, default_revision)).strip()
    if not revision:
        raise ValueError(f"{revision_env} must pin a Hub model revision")
    return LocalModelAssetRef.hub(model, revision=revision)


def _looks_like_path(value: str) -> bool:
    return value.startswith(("/", "./", "../", "~")) or Path(value).expanduser().is_dir()


def _parse_bool(value: object, *, name: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean")


def _parse_enum(enum_type, value: object, *, name: str):
    normalized = str(value).strip().lower()
    try:
        return enum_type(normalized)
    except ValueError as exc:
        choices = ", ".join(item.value for item in enum_type)
        raise ValueError(f"{name} must be one of: {choices}") from exc


def _retrieval_methods_from_environment(
    values: Mapping[str, str],
    *,
    mode: RetrievalProfileMode,
) -> tuple[RetrievalMethod, ...] | None:
    if RETRIEVAL_METHODS_ENV not in values:
        return None
    raw = str(values[RETRIEVAL_METHODS_ENV]).strip()
    if not raw:
        raise ValueError(f"{RETRIEVAL_METHODS_ENV} must be non-empty")
    try:
        methods = tuple(
            RetrievalMethod(item.strip().lower()) for item in raw.split(",")
        )
    except ValueError as exc:
        raise ValueError(
            f"{RETRIEVAL_METHODS_ENV} contains an unsupported method"
        ) from exc
    _validate_retrieval_methods(mode=mode, methods=methods)
    return methods


def _validate_retrieval_methods(
    *,
    mode: RetrievalProfileMode,
    methods: tuple[RetrievalMethod, ...] | None,
) -> None:
    if methods is None:
        return
    allowed = (
        ((RetrievalMethod.BM25,),)
        if mode is RetrievalProfileMode.LEXICAL
        else (
            (RetrievalMethod.DENSE, RetrievalMethod.LEARNED_SPARSE),
            (
                RetrievalMethod.DENSE,
                RetrievalMethod.LEARNED_SPARSE,
                RetrievalMethod.BM25,
            ),
        )
    )
    if methods not in allowed:
        raise ValueError(
            f"{RETRIEVAL_METHODS_ENV} must select an approved ordered method set"
        )


__all__ = [
    "BGE_M3_MODEL_ID",
    "BGE_M3_MODEL_REVISION",
    "BGE_V2_M3_RERANKER_MODEL_ID",
    "BGE_V2_M3_RERANKER_REVISION",
    'DocumentRetrievalCapability',
    'DocumentRetrievalProfile',
    'DocumentRetrievalRuntime',
    "RetrievalFailurePolicy",
    "RETRIEVAL_METHODS_ENV",
    "RetrievalProfileMode",
    "RetrievalProfileUnavailable",
    "RetrievalRerankerMode",
    "build_document_retrieval_runtime",
    "durable_chunking_profile",
    "preflight_document_retrieval_profile",
    "production_document_chunking_profile",
]
