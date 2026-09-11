"""Local-only model asset identities and capability checks.

Retrieval inference is deliberately not a model downloader.  A Hub-backed
asset is accepted only with an immutable revision and is resolved from the
local Hugging Face cache.  A managed directory is accepted as-is.  Both paths
are validated before a production composition is built, so a request can
never be the event that starts network I/O or discovers an incomplete model.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys
from typing import Final


BGE_M3_MODEL_ID: Final = "BAAI/bge-m3"
BGE_M3_MODEL_REVISION: Final = "5617a9f61b028005a4858fdac845db406aefb181"
BGE_V2_M3_RERANKER_MODEL_ID: Final = "BAAI/bge-reranker-v2-m3"
BGE_V2_M3_RERANKER_REVISION: Final = (
    "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
)
_FULL_COMMIT_SHA: Final = re.compile(r"[0-9a-fA-F]{40}")


class LocalModelAssetError(RuntimeError):
    """A pinned model is not available as a complete local asset."""


class ModelAssetSource(str, Enum):
    HUB_CACHE = "hub_cache"
    LOCAL_PATH = "local_path"


@dataclass(frozen=True, slots=True)
class LocalModelAssetRef:
    """An immutable local model reference.

    Hub identities require a revision and are always resolved with
    ``local_files_only=True``.  Local paths deliberately do not carry a Hub
    revision: the deployment owns the directory and must change the path when
    its contents change.
    """

    source: ModelAssetSource
    identifier: str
    revision: str | None = None
    canonical_identity: str | None = None
    local_files_only: bool = True

    def __post_init__(self) -> None:
        identifier = str(self.identifier).strip()
        if not identifier:
            raise ValueError("model asset identifier must be non-empty")
        object.__setattr__(self, "identifier", identifier)
        if self.local_files_only is not True:
            raise ValueError("retrieval model assets must be local-only")
        revision = str(self.revision or "").strip() or None
        canonical_identity = str(self.canonical_identity or "").strip() or None
        object.__setattr__(self, "canonical_identity", canonical_identity)
        if self.source is ModelAssetSource.HUB_CACHE and revision is None:
            raise ValueError("Hub model assets require an immutable revision")
        if (
            self.source is ModelAssetSource.HUB_CACHE
            and revision is not None
            and _FULL_COMMIT_SHA.fullmatch(revision) is None
        ):
            raise ValueError(
                "Hub model asset revision must be a full immutable commit SHA"
            )
        object.__setattr__(self, "revision", revision.lower() if revision else None)
        if self.source is ModelAssetSource.LOCAL_PATH and revision is not None:
            raise ValueError("local model paths must not declare a Hub revision")
        if self.source is ModelAssetSource.LOCAL_PATH and canonical_identity is None:
            raise ValueError(
                "local model paths require a canonical identity independent of the load path"
            )

    @classmethod
    def hub(cls, repo_id: str, *, revision: str) -> 'LocalModelAssetRef':
        return cls(
            source=ModelAssetSource.HUB_CACHE,
            identifier=repo_id,
            revision=revision,
        )

    @classmethod
    def path(
        cls,
        path: Path | str,
        *,
        canonical_identity: str,
    ) -> 'LocalModelAssetRef':
        return cls(
            source=ModelAssetSource.LOCAL_PATH,
            identifier=str(Path(path).expanduser()),
            canonical_identity=canonical_identity,
        )

    @property
    def identity(self) -> str:
        if self.source is ModelAssetSource.HUB_CACHE:
            return f"hf:{self.identifier}@{self.revision}"
        assert self.canonical_identity is not None
        return f"managed:{self.canonical_identity}"


@dataclass(frozen=True, slots=True)
class ModelAssetManifest:
    """Small structural manifest sufficient to fail closed before loading."""

    name: str
    required_files: tuple[str, ...]
    required_any: tuple[tuple[str, ...], ...] = ()
    required_modules: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("model asset manifest name must be non-empty")
        if not self.required_files:
            raise ValueError("model asset manifest must require files")
        for group in self.required_any:
            if not group:
                raise ValueError("model asset alternative groups must not be empty")


@dataclass(frozen=True, slots=True)
class ModelAssetCapability:
    """Safe preflight result; it never contains model or Source content."""

    asset_identity: str
    manifest_name: str
    ready: bool
    reason_code: str
    resolved_path: Path | None = None
    missing_files: tuple[str, ...] = ()
    missing_modules: tuple[str, ...] = ()
    content_sha256: str | None = None

    @property
    def generation_identity(self) -> str:
        if self.content_sha256 is None:
            return self.asset_identity
        return f"{self.asset_identity};content_sha256={self.content_sha256}"

    def require_ready(self) -> Path:
        if not self.ready or self.resolved_path is None:
            details = ",".join((*self.missing_modules, *self.missing_files))
            suffix = f":{details}" if details else ""
            raise LocalModelAssetError(f"{self.reason_code}{suffix}")
        return self.resolved_path

    def require_unchanged(self) -> Path:
        path = self.require_ready()
        if (
            self.content_sha256 is not None
            and _local_asset_content_sha256(path) != self.content_sha256
        ):
            raise LocalModelAssetError("local_model_asset_changed_after_preflight")
        return path

    def diagnostic_snapshot(self) -> dict[str, object]:
        return {
            "asset_identity": self.asset_identity,
            "manifest": self.manifest_name,
            "ready": self.ready,
            "reason_code": self.reason_code,
            "missing_files": self.missing_files,
            "missing_modules": self.missing_modules,
            "content_sha256": self.content_sha256,
            "local_files_only": True,
        }


MODEL_WEIGHT_FILES: Final[tuple[str, ...]] = (
    "model.safetensors",
    "pytorch_model.bin",
    "model.safetensors.index.json",
    "pytorch_model.bin.index.json",
)
TOKENIZER_FILES: Final[tuple[str, ...]] = (
    "tokenizer.json",
    "sentencepiece.bpe.model",
)

BGE_M3_BASE_MANIFEST: Final = ModelAssetManifest(
    name="bge_m3_base",
    required_files=("config.json", "tokenizer_config.json"),
    required_any=(MODEL_WEIGHT_FILES, TOKENIZER_FILES),
    required_modules=("FlagEmbedding", "transformers", "huggingface_hub"),
)
BGE_M3_HYBRID_MANIFEST: Final = ModelAssetManifest(
    name="bge_m3_hybrid",
    required_files=(
        "config.json",
        "tokenizer_config.json",
        "sparse_linear.pt",
        "colbert_linear.pt",
    ),
    required_any=(MODEL_WEIGHT_FILES, TOKENIZER_FILES),
    required_modules=("FlagEmbedding", "transformers", "huggingface_hub"),
)
BGE_V2_M3_RERANKER_MANIFEST: Final = ModelAssetManifest(
    name="bge_reranker_v2_m3",
    required_files=("config.json", "tokenizer_config.json"),
    required_any=(MODEL_WEIGHT_FILES, TOKENIZER_FILES),
    required_modules=("FlagEmbedding", "transformers", "huggingface_hub"),
)


def preflight_local_model_asset(
    reference: LocalModelAssetRef,
    manifest: ModelAssetManifest,
) -> ModelAssetCapability:
    """Resolve and validate one model without making a network request."""

    missing_modules = tuple(
        module
        for module in manifest.required_modules
        if not _module_available(module)
    )
    try:
        resolved = _resolve_local_path(reference)
    except LocalModelAssetError as exc:
        return ModelAssetCapability(
            asset_identity=reference.identity,
            manifest_name=manifest.name,
            ready=False,
            reason_code=str(exc),
            missing_modules=missing_modules,
        )

    missing_files = [
        name for name in manifest.required_files if not (resolved / name).is_file()
    ]
    for alternatives in manifest.required_any:
        if not any((resolved / name).is_file() for name in alternatives):
            missing_files.append("one_of(" + "|".join(alternatives) + ")")
    missing_files.extend(_weight_index_problems(resolved))
    content_sha256: str | None = None
    if reference.source is ModelAssetSource.LOCAL_PATH and not missing_files:
        try:
            content_sha256 = _local_asset_content_sha256(resolved)
        except (OSError, RuntimeError, ValueError):
            missing_files.append("local_asset_content_digest_unavailable")
    if missing_modules:
        reason = "model_runtime_dependency_missing"
    elif missing_files:
        reason = "local_model_asset_incomplete"
    else:
        reason = "ready"
    return ModelAssetCapability(
        asset_identity=reference.identity,
        manifest_name=manifest.name,
        ready=not missing_modules and not missing_files,
        reason_code=reason,
        resolved_path=resolved,
        missing_files=tuple(missing_files),
        missing_modules=missing_modules,
        content_sha256=content_sha256,
    )


def _weight_index_problems(model_root: Path) -> tuple[str, ...]:
    """验证 Transformers 分片索引不会把缺失权重推迟到首次请求时发现。"""

    problems: list[str] = []
    for index_name in (
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
    ):
        index_path = model_root / index_name
        if not index_path.is_file():
            continue
        try:
            if index_path.stat().st_size > 16 * 1024 * 1024:
                raise ValueError("weight index is unreasonably large")
            payload = json.loads(index_path.read_text(encoding="utf-8"))
            weight_map = payload.get("weight_map")
            if not isinstance(weight_map, dict) or not weight_map:
                raise ValueError("weight index has no weight map")
            shard_names = set()
            for shard in weight_map.values():
                if not isinstance(shard, str) or not _safe_asset_relative_path(shard):
                    raise ValueError("weight index contains an unsafe shard path")
                shard_names.add(shard)
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError, TypeError):
            problems.append(f"invalid_weight_index({index_name})")
            continue
        if any(not (model_root / shard).is_file() for shard in shard_names):
            problems.append(f"missing_weight_shard({index_name})")
    return tuple(problems)


def _safe_asset_relative_path(value: str) -> bool:
    path = Path(value)
    return bool(
        value.strip()
        and not path.is_absolute()
        and all(part not in {"", ".", ".."} for part in path.parts)
    )


def _local_asset_content_sha256(model_root: Path) -> str:
    """为受管本地目录生成内容身份，并在计算期间检测普通文件漂移。"""

    root = model_root.resolve(strict=True)
    before = _local_asset_snapshot(root)
    if not before:
        raise ValueError("local model asset contains no files")
    digest = _hash_local_asset_snapshot(str(root), before)
    if before != _local_asset_snapshot(root):
        raise RuntimeError("local model asset changed during fingerprinting")
    return digest


def _local_asset_snapshot(
    model_root: Path,
) -> tuple[tuple[str, int, int, int, int], ...]:
    snapshot: list[tuple[str, int, int, int, int]] = []
    for path in sorted(model_root.rglob("*"), key=lambda item: item.as_posix()):
        if not path.is_file():
            continue
        stat = path.stat()
        snapshot.append((
            path.relative_to(model_root).as_posix(),
            int(stat.st_size),
            int(stat.st_mtime_ns),
            int(stat.st_ctime_ns),
            int(stat.st_ino),
        ))
    return tuple(snapshot)


@lru_cache(maxsize=16)
def _hash_local_asset_snapshot(
    model_root: str,
    snapshot: tuple[tuple[str, int, int, int, int], ...],
) -> str:
    digest = hashlib.sha256()
    root = Path(model_root)
    for relative_path, size, _mtime_ns, _ctime_ns, _inode in snapshot:
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(size).encode("ascii"))
        digest.update(b"\0")
        with (root / relative_path).open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def _module_available(name: str) -> bool:
    if name in sys.modules:
        return True
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _resolve_local_path(reference: LocalModelAssetRef) -> Path:
    if reference.source is ModelAssetSource.LOCAL_PATH:
        path = Path(reference.identifier).expanduser().resolve(strict=False)
        if not path.is_dir():
            raise LocalModelAssetError("local_model_path_unavailable")
        return path
    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:
        raise LocalModelAssetError("huggingface_hub_unavailable") from exc
    try:
        path = Path(
            snapshot_download(
                repo_id=reference.identifier,
                revision=reference.revision,
                local_files_only=True,
            )
        ).resolve(strict=False)
    except Exception as exc:
        raise LocalModelAssetError("pinned_model_not_in_local_cache") from exc
    if not path.is_dir():
        raise LocalModelAssetError("pinned_model_not_in_local_cache")
    return path


__all__ = [
    "BGE_M3_MODEL_ID",
    "BGE_M3_MODEL_REVISION",
    "BGE_M3_BASE_MANIFEST",
    "BGE_M3_HYBRID_MANIFEST",
    "BGE_V2_M3_RERANKER_MODEL_ID",
    "BGE_V2_M3_RERANKER_REVISION",
    "BGE_V2_M3_RERANKER_MANIFEST",
    "LocalModelAssetError",
    'LocalModelAssetRef',
    'ModelAssetCapability',
    'ModelAssetManifest',
    "ModelAssetSource",
    "preflight_local_model_asset",
]
