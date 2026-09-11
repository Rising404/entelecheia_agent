"""支持显式能力降级的 BGE-M3 编码。"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
from importlib import metadata as importlib_metadata
from pathlib import Path
import re
import threading
from typing import Any
import unicodedata

from ..compute.resources import compute_resource_lock, normalize_device
from ..compute.inference import (
    classify_device_error,
    guard_device_errors,
    synchronize_device,
)
from ..execution import checkpoint, current_execution, measure, wait_for_lock
from ..ports import RetrievalCancelled

from .model_assets import (
    BGE_M3_BASE_MANIFEST,
    BGE_M3_HYBRID_MANIFEST,
    BGE_M3_MODEL_ID,
    BGE_M3_MODEL_REVISION,
    LocalModelAssetRef,
    preflight_local_model_asset,
)

class RetrievalMethodUnavailable(RuntimeError):
    """某项方法无法运行；这不同于有效的无匹配结果。"""

    def __init__(self, safe_error_code: str, *, stage: str | None = None) -> None:
        normalized_code = _safe_diagnostic_token(
            safe_error_code,
            fallback="retrieval_method_unavailable",
        )
        super().__init__(normalized_code)
        self.safe_error_code = normalized_code
        self.stage = (
            _safe_diagnostic_token(stage, fallback="retrieval_method")
            if stage is not None
            else None
        )


class DeterministicLexicalEncoder:
    """离线默认路径使用的无依赖、带版本 BM25 tokenizer。

    它刻意不合成稠密向量或学习型稀疏向量。方法存储已把不可用的稠密流程视为纯 BM25
    generation；此编码器使这种降级在重启后保持稳定，也不受之后是否恰好安装可选 ML 包
    的影响。
    """

    learned_sparse_available = False
    _FINGERPRINT = (
        "deterministic_lexical:unicode_nfkc_casefold+cjk_bigram+symbol_unit+sha256_u63@2"
    )
    def fingerprint(self) -> str:
        return self._FINGERPRINT

    def encode(self, texts: Sequence[str]):
        del texts
        raise RetrievalMethodUnavailable("dense_encoding_not_configured")

    def token_ids(self, text: str) -> tuple[int, ...]:
        return tuple(_stable_lexical_token_id(token) for token in _lexical_tokens(text))

    def tokenizer_fingerprint(self) -> str:
        return self._FINGERPRINT


@dataclass(frozen=True, slots=True)
class BgeM3EncodedText:
    dense_vector: tuple[float, ...]
    learned_sparse_weights: Mapping[int, float]
    token_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.dense_vector) != 1024:
            raise ValueError("BGE-M3 dense vectors must contain exactly 1024 values")
        if any(weight <= 0 for weight in self.learned_sparse_weights.values()):
            raise ValueError("learned sparse weights must be positive")


class BgeM3Encoder:
    """延迟加载的真实 BGE-M3 编码器，支持有界查询结果复用。

    ``FlagEmbedding`` 刻意设为可选依赖。未安装时调用 ``encode`` 会抛出
    ``RetrievalMethodUnavailable``；调用方随后可采用已有说明的 BM25 或字面回退，而不是
    用其他模型凭空生成稀疏权重。
    """

    def __init__(
        self,
        *,
        asset: LocalModelAssetRef | None = None,
        resolved_model_path: Path | str | None = None,
        local_files_only: bool = True,
        device: str = "cpu",
        use_fp16: bool | None = None,
        query_max_length: int = 512,
        passage_max_length: int = 8_192,
        max_query_cache_entries: int = 128,
    ) -> None:
        if max_query_cache_entries <= 0:
            raise ValueError("max_query_cache_entries must be greater than zero")
        if local_files_only is not True:
            raise ValueError("BGE-M3 retrieval must use local-only model assets")
        if not isinstance(query_max_length, int) or query_max_length <= 0:
            raise ValueError("query_max_length must be a positive integer")
        if not isinstance(passage_max_length, int) or passage_max_length <= 0:
            raise ValueError("passage_max_length must be a positive integer")
        self._asset = asset or LocalModelAssetRef.hub(
            BGE_M3_MODEL_ID,
            revision=BGE_M3_MODEL_REVISION,
        )
        self._resolved_model_path = (
            Path(resolved_model_path).expanduser().resolve(strict=False)
            if resolved_model_path is not None
            else None
        )
        if self._resolved_model_path is not None and not self._resolved_model_path.is_dir():
            raise ValueError("resolved BGE-M3 model path must be a directory")
        self._device = normalize_device(device)
        self._use_fp16 = use_fp16
        self._query_max_length = query_max_length
        self._passage_max_length = passage_max_length
        self._runtime_implementation_versions = (
            ("flagembedding", _distribution_version("FlagEmbedding")),
            ("transformers", _distribution_version("transformers")),
        )
        # FlagEmbedding 1.4 会在任一制品缺失时构造随机的稀疏或 ColBERT 投影层。
        # 请在惰性加载模型前完成判断，避免索引随机的学习型稀疏权重。
        self._base_capability = preflight_local_model_asset(
            self._asset,
            BGE_M3_BASE_MANIFEST,
        )
        self._hybrid_capability = preflight_local_model_asset(
            self._asset,
            BGE_M3_HYBRID_MANIFEST,
        )
        projection_root = self._resolved_model_path or self._hybrid_capability.resolved_path
        self._learned_sparse_available = bool(
            projection_root is not None
            and (projection_root / "sparse_linear.pt").is_file()
            and (projection_root / "colbert_linear.pt").is_file()
        )
        self._model: Any | None = None
        self._tokenizer: Any | None = None
        self._load_lock = threading.Lock()
        self._query_cache: OrderedDict[str, BgeM3EncodedText] = OrderedDict()
        self._cache_lock = threading.Lock()
        self._max_query_cache_entries = max_query_cache_entries
        # 设备属于编码/索引身份。运行中出错只能要求重建，不能在同一代内改用 CPU。
        self._cpu_rebuild_reason: str | None = None

    def fingerprint(self) -> str:
        precision = "auto" if self._use_fp16 is None else str(self._use_fp16).lower()
        sparse_assets = "available" if self._learned_sparse_available else "unavailable"
        return (
            f"bge_m3:model={self._base_capability.generation_identity};"
            "dense=1024;sparse=flagembedding;"
            f"sparse_projection_assets={sparse_assets};device={self._device};"
            f"fp16={precision};query_max_length={self._query_max_length};"
            f"passage_max_length={self._passage_max_length};"
            + ";".join(
                f"{name}={version}"
                for name, version in self._runtime_implementation_versions
            )
            + ";local_only=true"
        )

    def diagnostic_snapshot(self) -> dict[str, object]:
        return {
            "fingerprint": self.fingerprint(),
            "base_asset": self._base_capability.diagnostic_snapshot(),
            "hybrid_asset": self._hybrid_capability.diagnostic_snapshot(),
            "device": self._device,
            "loaded": self._model is not None,
            "cpu_rebuild_required": self._cpu_rebuild_reason is not None,
            "cpu_rebuild_reason": self._cpu_rebuild_reason,
            "local_files_only": True,
        }

    @property
    def learned_sparse_available(self) -> bool:
        """是否可以输出已验证的 BGE-M3 学习型稀疏权重。

        false 表示显式能力降级，而非无匹配。稠密 embedding 和采用 BGE 分词的 BM25
        仍可继续使用。
        """

        return self._learned_sparse_available

    def encode(self, texts: Sequence[str]) -> tuple[BgeM3EncodedText, ...]:
        checkpoint()
        with self._cache_lock:
            self._require_encoding_device()
        if not texts:
            return ()
        return_sparse = self._learned_sparse_available
        try:
            with wait_for_lock(compute_resource_lock(self._device), "encoder_queue"):
                with self._cache_lock:
                    self._require_encoding_device()
                try:
                    model = self._load_model()
                    checkpoint()
                    with measure("encoder_encode"), guard_device_errors(model, self._device):
                        payload = model.encode(
                            list(texts),
                            return_dense=True,
                            return_sparse=return_sparse,
                            return_colbert_vecs=False,
                        )
                        synchronize_device(self._device)
                    checkpoint()
                except RetrievalCancelled:
                    raise
                except Exception as exc:
                    reason = classify_device_error(exc, self._device)
                    if reason is not None:
                        # 在释放设备锁之前冻结故障状态，排队中的调用不能再次进入坏实例。
                        with self._cache_lock:
                            self._cpu_rebuild_reason = reason
                            self._query_cache.clear()
                            self._require_encoding_device(cause=exc)
                    raise
            dense_values = payload["dense_vecs"]
            sparse_values = payload["lexical_weights"] if return_sparse else ({},) * len(texts)
        except (RetrievalCancelled, RetrievalMethodUnavailable):
            raise
        except Exception as exc:
            raise RetrievalMethodUnavailable(
                f"bge_m3_encode_failed:{type(exc).__name__}",
                stage="encode",
            ) from exc
        if len(dense_values) != len(texts) or len(sparse_values) != len(texts):
            raise RetrievalMethodUnavailable(
                "bge_m3_encode_returned_unaligned_batch",
                stage="encode",
            )
        return tuple(
            BgeM3EncodedText(
                dense_vector=tuple(float(value) for value in dense),
                learned_sparse_weights=normalise_sparse_weights(sparse),
                token_ids=tuple(self.token_ids(text)),
            )
            for text, dense, sparse in zip(texts, dense_values, sparse_values, strict=True)
        )

    def encode_query(self, query: str) -> BgeM3EncodedText:
        checkpoint()
        execution = current_execution()
        with self._cache_lock:
            self._require_encoding_device()
            cached = self._query_cache.get(query)
            if cached is not None:
                self._query_cache.move_to_end(query)
                if execution is not None:
                    execution.increment("encoder_query_cache_hits")
                return cached
        if execution is not None:
            execution.increment("encoder_query_cache_misses")
        encoded = self.encode((query,))[0]
        checkpoint()
        with self._cache_lock:
            self._require_encoding_device()
            self._query_cache[query] = encoded
            self._query_cache.move_to_end(query)
            while len(self._query_cache) > self._max_query_cache_entries:
                self._query_cache.popitem(last=False)
        return encoded

    def _require_encoding_device(self, *, cause: Exception | None = None) -> None:
        """调用方持有缓存锁；故障状态与缓存读取/发布必须具有同一先后顺序。"""

        if self._cpu_rebuild_reason is not None:
            raise RetrievalMethodUnavailable(
                f"bge_m3_cpu_rebuild_required:{self._cpu_rebuild_reason}",
                stage="encode",
            ) from cause

    def token_ids(self, text: str) -> tuple[int, ...]:
        checkpoint()
        tokenizer = self._load_tokenizer()
        try:
            with measure("encoder_tokenize"):
                token_ids = tokenizer.encode(text, add_special_tokens=False)
            checkpoint()
        except RetrievalCancelled:
            raise
        except Exception as exc:
            raise RetrievalMethodUnavailable(
                f"bge_m3_tokenize_failed:{type(exc).__name__}",
                stage="tokenize",
            ) from exc
        return tuple(int(token_id) for token_id in token_ids)

    def token_offsets(self, text: str) -> tuple[tuple[int, int], ...]:
        """Return BGE tokenizer spans over the original Python string.

        Chunk authority must retain original source text, so callers cut at
        tokenizer offsets instead of decoding token IDs back into normalized
        model text.  BGE-M3 ships a fast tokenizer with offset mappings; a
        model asset that cannot provide them is not suitable for authoritative
        chunk construction and therefore fails closed.
        """

        tokenizer = self._load_tokenizer()
        try:
            encoded = tokenizer(
                text,
                add_special_tokens=False,
                return_attention_mask=False,
                return_token_type_ids=False,
                return_offsets_mapping=True,
            )
            raw_offsets = encoded["offset_mapping"]
        except Exception as exc:
            raise RetrievalMethodUnavailable(
                f"bge_m3_token_offsets_failed:{type(exc).__name__}",
                stage="tokenize",
            ) from exc
        offsets: list[tuple[int, int]] = []
        for raw_start, raw_end in raw_offsets:
            start, end = int(raw_start), int(raw_end)
            if start < 0 or end <= start or end > len(text):
                raise RetrievalMethodUnavailable(
                    "bge_m3_token_offsets_invalid",
                    stage="tokenize",
                )
            offsets.append((start, end))
        return tuple(offsets)

    def tokenizer_fingerprint(self) -> str:
        versions = dict(self._runtime_implementation_versions)
        return (
            "bge_m3_tokenizer:"
            f"model={self._base_capability.generation_identity};"
            f"transformers={versions['transformers']};"
            "special_tokens=false;offset_mapping=true"
        )

    def _load_model(self):
        checkpoint()
        if self._model is not None:
            return self._model
        with wait_for_lock(self._load_lock, "encoder_load_queue"):
            if self._model is not None:
                return self._model
            try:
                kwargs: dict[str, Any] = {
                    "devices": self._device,
                    "query_max_length": self._query_max_length,
                    "passage_max_length": self._passage_max_length,
                    "trust_remote_code": False,
                }
                if self._use_fp16 is not None:
                    kwargs["use_fp16"] = self._use_fp16
                with measure("encoder_model_load"):
                    from FlagEmbedding import BGEM3FlagModel

                    self._model = BGEM3FlagModel(str(self._local_model_path()), **kwargs)
                self._tokenizer = getattr(self._model, "tokenizer", None)
            except RetrievalCancelled:
                raise
            except Exception as exc:
                raise RetrievalMethodUnavailable(
                    f"bge_m3_model_unavailable:{type(exc).__name__}",
                    stage="model_load",
                ) from exc
        checkpoint()
        return self._model

    def _load_tokenizer(self):
        checkpoint()
        if self._tokenizer is not None:
            return self._tokenizer
        try:
            with measure("encoder_tokenizer_load"):
                from transformers import AutoTokenizer

                self._tokenizer = AutoTokenizer.from_pretrained(
                    str(self._local_model_path()),
                    local_files_only=True,
                    trust_remote_code=False,
                )
        except RetrievalCancelled:
            raise
        except Exception as exc:
            raise RetrievalMethodUnavailable(
                f"bge_m3_tokenizer_unavailable:{type(exc).__name__}",
                stage="tokenizer_load",
            ) from exc
        if self._tokenizer is None:
            raise RetrievalMethodUnavailable(
                "bge_m3_tokenizer_unavailable",
                stage="tokenizer_load",
            )
        checkpoint()
        return self._tokenizer

    def _local_model_path(self) -> Path:
        if self._resolved_model_path is not None:
            try:
                self._base_capability.require_unchanged()
            except Exception as exc:
                raise RetrievalMethodUnavailable(
                    "bge_m3_model_unavailable:local_model_asset_changed_after_preflight",
                    stage="model_load",
                ) from exc
            return self._resolved_model_path
        try:
            return self._base_capability.require_unchanged()
        except Exception as exc:
            raise RetrievalMethodUnavailable(
                f"bge_m3_model_unavailable:{self._base_capability.reason_code}",
                stage="model_load",
            ) from exc


def normalise_sparse_weights(weights: Mapping[Any, Any]) -> dict[int, float]:
    """将编码器的稀疏投影转换为稳定的正 token 权重。"""

    normalised: dict[int, float] = {}
    for token_id, weight in weights.items():
        try:
            parsed_token_id = int(token_id)
            parsed_weight = float(weight)
        except (TypeError, ValueError):
            continue
        if parsed_token_id < 0 or parsed_weight <= 0:
            continue
        normalised[parsed_token_id] = max(parsed_weight, normalised.get(parsed_token_id, 0.0))
    return normalised


def _distribution_version(distribution: str) -> str:
    try:
        return importlib_metadata.version(distribution)
    except importlib_metadata.PackageNotFoundError:
        return "missing"


_SAFE_DIAGNOSTIC_TOKEN_RE = re.compile(r"[A-Za-z0-9_.:-]{1,160}")


def _safe_diagnostic_token(value: object, *, fallback: str) -> str:
    """只保留有限分类码；绝不把路径、正文或 traceback 带入持久审计。"""

    normalized = str(value or "").strip()
    if _SAFE_DIAGNOSTIC_TOKEN_RE.fullmatch(normalized) is None:
        return fallback
    return normalized


def _lexical_tokens(text: str) -> tuple[str, ...]:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    tokens: list[str] = []
    word: list[str] = []
    previous_cjk: str | None = None

    def flush_word() -> None:
        if word:
            tokens.append("w:" + "".join(word))
            word.clear()

    for character in normalized:
        if _is_cjk(character):
            flush_word()
            tokens.append("c:" + character)
            if previous_cjk is not None:
                tokens.append("g:" + previous_cjk + character)
            previous_cjk = character
        elif character.isalnum() or character == "_":
            previous_cjk = None
            word.append(character)
        else:
            flush_word()
            previous_cjk = None
            if not character.isspace():
                tokens.append("s:" + character)
    flush_word()
    return tuple(tokens)


def _stable_lexical_token_id(token: str) -> int:
    digest = hashlib.sha256(token.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def _is_cjk(character: str) -> bool:
    codepoint = ord(character)
    return (
        0x3400 <= codepoint <= 0x4DBF
        or 0x4E00 <= codepoint <= 0x9FFF
        or 0xF900 <= codepoint <= 0xFAFF
    )
