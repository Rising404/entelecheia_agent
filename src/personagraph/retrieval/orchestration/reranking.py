"""对已验证 Source 内容执行可选模型重排。"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from importlib import metadata as importlib_metadata
import math
from pathlib import Path
import threading
from typing import Any

from ..compute.resources import compute_resource_lock, normalize_device
from ..compute.inference import (
    classify_device_error,
    guard_device_errors,
    release_device_cache,
    synchronize_device,
)
from ..contracts import (
    RerankerOutcome,
    RerankerRunStatus,
    RetrievalCandidate,
    SourceType,
)
from ..execution import (
    RerankingDeadlineReached, checkpoint,
    current_execution, measure, wait_for_lock,
)
from ..ports import RerankerPort, RetrievalCancelled
from ..query_guard import DEFAULT_MAX_RETRIEVAL_QUERY_TOKENS
from ..indexing.model_assets import (
    BGE_V2_M3_RERANKER_MANIFEST,
    BGE_V2_M3_RERANKER_MODEL_ID,
    BGE_V2_M3_RERANKER_REVISION,
    LocalModelAssetRef,
    preflight_local_model_asset,
)
from .selection import ResolvedCandidate
from .rerank_execution import observe_reranker_batches


class RerankerUnavailable(RuntimeError):
    """可选第二阶段评分器无法安全生成对齐分数。"""


class BgeM3Reranker:
    """``BAAI/bge-reranker-v2-m3`` 的延迟本地适配器。

    这是交叉编码器评分器，而非 ``BAAI/bge-m3`` embedding 适配器。即使局部 Source 只有
    一个候选也会打分，因为该分数可能参与调用边界外的同查询全局排序。加载失败会被缓存，
    避免离线进程为每个 Source 重复昂贵的下载或导入尝试。
    """

    def __init__(
        self,
        *,
        asset: LocalModelAssetRef | None = None,
        resolved_model_path: Path | str | None = None,
        local_files_only: bool = True,
        device: str = "cpu",
        use_fp16: bool = False,
        allow_cpu_fallback: bool = False,
        batch_size: int = 8,
        query_max_length: int = DEFAULT_MAX_RETRIEVAL_QUERY_TOKENS,
        max_length: int = 1024,
    ) -> None:
        if local_files_only is not True:
            raise ValueError("BGE reranking must use local-only model assets")
        if not isinstance(allow_cpu_fallback, bool):
            raise ValueError("allow_cpu_fallback must be a boolean")
        if allow_cpu_fallback and use_fp16:
            raise ValueError("CPU fallback requires the FP32 precision profile")
        for name, value in (
            ("batch_size", batch_size),
            ("query_max_length", query_max_length),
            ("max_length", max_length),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"reranker {name} must be a positive integer")
        if query_max_length >= max_length:
            raise ValueError("reranker query_max_length must be smaller than max_length")
        self._asset = asset or LocalModelAssetRef.hub(
            BGE_V2_M3_RERANKER_MODEL_ID,
            revision=BGE_V2_M3_RERANKER_REVISION,
        )
        self._asset_capability = preflight_local_model_asset(
            self._asset,
            BGE_V2_M3_RERANKER_MANIFEST,
        )
        self._resolved_model_path = (
            Path(resolved_model_path).expanduser().resolve(strict=False)
            if resolved_model_path is not None
            else None
        )
        if self._resolved_model_path is not None and not self._resolved_model_path.is_dir():
            raise ValueError("resolved reranker model path must be a directory")
        self._device = normalize_device(device)
        self._requested_device = self._device
        self._use_fp16 = bool(use_fp16)
        self._allow_cpu_fallback = allow_cpu_fallback
        self._cpu_fallback_reason: str | None = None
        self._gpu_cleanup_reason: str | None = None
        self._batch_size = batch_size
        self._query_max_length = query_max_length
        self._max_length = max_length
        self._runtime_implementation_versions = (
            ("flagembedding", _distribution_version("FlagEmbedding")),
            ("transformers", _distribution_version("transformers")),
        )
        self._model: Any | None = None
        self._load_failure_reason: str | None = None
        self._load_lock = threading.Lock()
        # 锁住一个实例的模型与设备切换；设备锁仍用于不同实例共享同一计算资源。
        self._lifecycle_lock = threading.Lock()

    def fingerprint(self) -> str:
        """冻结评分配方身份；运行期回退仅更新诊断，不能让工具目录身份漂移。"""

        return (
            "bge_reranker_v2_m3:"
            f"model={self._asset_capability.generation_identity};"
            f"device={self._requested_device};"
            f"batch_size={self._batch_size};"
            f"query_max_length={self._query_max_length};max_length={self._max_length};"
            f"fp16={str(self._use_fp16).lower()};"
            + ";".join(
                f"{name}={version}"
                for name, version in self._runtime_implementation_versions
            )
            + ";local_only=true"
            + (";cpu_fallback=true" if self._allow_cpu_fallback else "")
        )

    def diagnostic_snapshot(self) -> dict[str, object]:
        """返回能力状态，不包含模型输入或 Source 内容。"""

        return {
            "fingerprint": self.fingerprint(),
            "asset": self._asset_capability.diagnostic_snapshot(),
            "device": self._device,
            "requested_device": self._requested_device,
            "cpu_fallback_reason": self._cpu_fallback_reason,
            "gpu_cleanup_reason": self._gpu_cleanup_reason,
            "batch_size": self._batch_size,
            "query_max_length": self._query_max_length,
            "max_length": self._max_length,
            "use_fp16": self._use_fp16,
            "loaded": self._model is not None,
            "load_failure_reason": self._load_failure_reason,
            "local_files_only": True,
        }

    def score(self, pairs: Sequence[tuple[str, str]]) -> tuple[float, ...]:
        normalized = tuple(pairs)
        if not normalized:
            return ()
        for pair in normalized:
            if (
                not isinstance(pair, tuple)
                or len(pair) != 2
                or not isinstance(pair[0], str)
                or not pair[0].strip()
                or not isinstance(pair[1], str)
                or not pair[1].strip()
            ):
                raise ValueError("reranker pairs must contain non-empty query/passage strings")
        execution = current_execution()
        checkpoint()
        if execution is not None:
            execution.increment("reranker_requested_pairs", len(normalized))
        with wait_for_lock(
            self._lifecycle_lock, "reranker_lifecycle_queue", reserve_for_projection=True,
        ):
            raw_scores = self._score_with_device_fallback(normalized)
        if isinstance(raw_scores, (int, float)):
            values = (float(raw_scores),)
        else:
            try:
                values = tuple(float(value) for value in raw_scores)
            except (TypeError, ValueError) as exc:
                raise RerankerUnavailable("bge_reranker_returned_invalid_scores") from exc
        if len(values) != len(normalized) or any(not math.isfinite(value) for value in values):
            raise RerankerUnavailable("bge_reranker_returned_unaligned_scores")
        checkpoint()
        if execution is not None:
            execution.increment("reranker_scored_pairs", len(values))
        return values

    def _score_with_device_fallback(self, pairs: tuple[tuple[str, str], ...]):
        """仅已识别的 GPU 故障可重跑完整输入，不拼接部分评分。"""

        try:
            return self._score_on_current_device(pairs)
        except (RetrievalCancelled, RerankingDeadlineReached):
            raise
        except Exception as exc:
            reason = (
                classify_device_error(exc, self._device)
                if self._allow_cpu_fallback and self._device != "cpu"
                else None
            )
            if reason is None:
                raise

        # 已离开 except，避免其 traceback 持有 GPU 模型；单设备调用也已释放设备锁。
        failed_device = self._device
        self._model = None
        self._load_failure_reason = None
        self._gpu_cleanup_reason = release_device_cache(failed_device)
        execution = current_execution()
        if execution is not None and self._gpu_cleanup_reason is not None:
            execution.set_metric("reranker_gpu_cleanup_reason", self._gpu_cleanup_reason)
        checkpoint()
        # 未知清理错误或取消不能提前改变设备身份；已识别的清理失败由共享层记录后放行。
        self._device = "cpu"
        self._cpu_fallback_reason = reason
        if execution is not None:
            execution.increment("reranker_cpu_fallback_count")
            execution.set_metric("reranker_cpu_fallback_reason", reason)
        return self._score_on_current_device(pairs)

    def _score_on_current_device(self, pairs: tuple[tuple[str, str], ...]):
        execution = current_execution()
        if execution is not None:
            execution.set_metric("reranker_device", self._device)
            execution.set_metric("reranker_requested_device", self._requested_device)
            if self._cpu_fallback_reason is not None:
                execution.set_metric("reranker_cpu_fallback_reason", self._cpu_fallback_reason)
            if self._gpu_cleanup_reason is not None:
                execution.set_metric("reranker_gpu_cleanup_reason", self._gpu_cleanup_reason)
        try:
            with wait_for_lock(
                compute_resource_lock(self._device), "reranker_queue",
                reserve_for_projection=True,
            ):
                model = self._load_model()
                checkpoint()
                with guard_device_errors(model, self._device):
                    with observe_reranker_batches(model, device=self._device):
                        scores = model.compute_score(
                            list(pairs),
                            batch_size=self._batch_size,
                            max_length=self._max_length,
                            normalize=False,
                        )
                    # 没有审计上下文时也等待异步设备错误，使整批失败在此边界被识别。
                    synchronize_device(self._device)
                    return scores
        except (RetrievalCancelled, RerankingDeadlineReached):
            raise
        except RerankerUnavailable:
            raise
        except Exception as exc:
            raise RerankerUnavailable(
                f"bge_reranker_inference_failed:{type(exc).__name__}"
            ) from exc

    def _load_model(self):
        checkpoint()
        if self._model is not None:
            execution = current_execution()
            if execution is not None:
                execution.increment("reranker_model_cache_hits")
            return self._model
        if self._load_failure_reason is not None:
            raise RerankerUnavailable(self._load_failure_reason)
        with wait_for_lock(self._load_lock, "reranker_load_queue", reserve_for_projection=True):
            if self._model is not None:
                return self._model
            if self._load_failure_reason is not None:
                raise RerankerUnavailable(self._load_failure_reason)
            try:
                with measure("reranker_model_load"):
                    from FlagEmbedding import FlagReranker

                    self._model = FlagReranker(
                        str(self._local_model_path()),
                        use_fp16=self._use_fp16,
                        devices=self._device,
                        batch_size=self._batch_size,
                        query_max_length=self._query_max_length,
                        max_length=self._max_length,
                        normalize=False,
                        trust_remote_code=False,
                    )
            except (RetrievalCancelled, RerankingDeadlineReached):
                raise
            except Exception as exc:
                self._load_failure_reason = (
                    f"bge_reranker_model_unavailable:{type(exc).__name__}"
                )
                raise RerankerUnavailable(self._load_failure_reason) from exc
        checkpoint()
        return self._model

    def _local_model_path(self) -> Path:
        if self._resolved_model_path is not None:
            try:
                self._asset_capability.require_unchanged()
            except Exception as exc:
                raise RerankerUnavailable(
                    "bge_reranker_model_unavailable:"
                    "local_model_asset_changed_after_preflight"
                ) from exc
            return self._resolved_model_path
        try:
            return self._asset_capability.require_unchanged()
        except Exception as exc:
            raise RerankerUnavailable(
                f"bge_reranker_model_unavailable:{self._asset_capability.reason_code}"
            ) from exc


def _distribution_version(distribution: str) -> str:
    try:
        return importlib_metadata.version(distribution)
    except importlib_metadata.PackageNotFoundError:
        return "missing"


@dataclass(frozen=True, slots=True)
class VerifiedRerankResult:
    resolved_by_source: Mapping[SourceType, tuple[ResolvedCandidate, ...]]
    outcomes: tuple[RerankerOutcome, ...]
    reranker_fingerprint: str


def rerank_verified_candidates(
    *,
    queries: Sequence[str],
    resolved_by_source: Mapping[SourceType, Sequence[ResolvedCandidate]],
    reranker: RerankerPort | None,
    candidate_limit_per_source: int,
) -> VerifiedRerankResult:
    """重排已验证内容，同时保持 Source 与多查询公平性。

    每个候选只会针对最初召回它的受守卫查询进行评分。候选在该查询通道内重排，然后再次
    轮询合并。这样可避免比较为不同语义查询或不同 Source 生成的交叉编码器 logits。
    """

    if candidate_limit_per_source <= 1:
        raise ValueError("reranker candidate limit must be greater than one")
    stable = {
        source_type: tuple(candidates)
        for source_type, candidates in resolved_by_source.items()
    }
    if reranker is None:
        return VerifiedRerankResult(stable, (), "")
    try:
        fingerprint = reranker.fingerprint()
    except Exception as exc:
        fingerprint = f"reranker_fingerprint_unavailable:{type(exc).__name__}"
    if not isinstance(fingerprint, str) or not fingerprint.strip():
        fingerprint = "reranker_fingerprint_unavailable"

    reranked: dict[SourceType, tuple[ResolvedCandidate, ...]] = {}
    outcomes: list[RerankerOutcome] = []
    circuit_reason: str | None = None
    for source_type, candidates in stable.items():
        if not candidates:
            reranked[source_type] = candidates
            outcomes.append(
                RerankerOutcome(
                    source_type=source_type,
                    status=RerankerRunStatus.NOT_RUN,
                    candidate_count=len(candidates),
                    reason_code="reranker_no_candidates",
                )
            )
            continue
        if circuit_reason is not None:
            reranked[source_type] = candidates
            outcomes.append(
                RerankerOutcome(
                    source_type=source_type,
                    status=RerankerRunStatus.DEGRADED,
                    candidate_count=len(candidates),
                    reason_code="reranker_circuit_open",
                )
            )
            continue

        head = candidates[:candidate_limit_per_source]
        tail = candidates[candidate_limit_per_source:]
        try:
            checkpoint()
            pairs = tuple(
                (_query_for_candidate(candidate, queries), unit.content)
                for candidate, unit in head
            )
            scores = tuple(float(score) for score in reranker.score(pairs))
            checkpoint()
            if len(scores) != len(head) or any(not math.isfinite(score) for score in scores):
                raise RerankerUnavailable("reranker_returned_unaligned_scores")
            ordered_head = _rerank_query_lanes(head, scores)
        except (RetrievalCancelled, RerankingDeadlineReached):
            raise
        except RerankerUnavailable as exc:
            circuit_reason = str(exc) or "reranker_unavailable"
            reranked[source_type] = candidates
            outcomes.append(
                RerankerOutcome(
                    source_type=source_type,
                    status=RerankerRunStatus.DEGRADED,
                    candidate_count=len(candidates),
                    reason_code=circuit_reason,
                )
            )
            continue
        except Exception as exc:
            circuit_reason = f"reranker_failed:{type(exc).__name__}"
            reranked[source_type] = candidates
            outcomes.append(
                RerankerOutcome(
                    source_type=source_type,
                    status=RerankerRunStatus.DEGRADED,
                    candidate_count=len(candidates),
                    reason_code=circuit_reason,
                )
            )
            continue

        combined = tuple((*ordered_head, *tail))
        reranked[source_type] = combined
        outcomes.append(
            RerankerOutcome(
                source_type=source_type,
                status=RerankerRunStatus.USED,
                candidate_count=len(candidates),
                scored_candidate_count=len(head),
            )
        )
    return VerifiedRerankResult(reranked, tuple(outcomes), fingerprint)


def _query_for_candidate(
    candidate: RetrievalCandidate,
    queries: Sequence[str],
) -> str:
    if candidate.query_index < 0 or candidate.query_index >= len(queries):
        raise RerankerUnavailable("reranker_candidate_query_index_out_of_bounds")
    query = queries[candidate.query_index]
    if not isinstance(query, str) or not query.strip():
        raise RerankerUnavailable("reranker_candidate_query_unavailable")
    return query


def _rerank_query_lanes(
    candidates: Sequence[ResolvedCandidate],
    scores: Sequence[float],
) -> tuple[ResolvedCandidate, ...]:
    lanes: dict[int, list[tuple[int, ResolvedCandidate, float]]] = defaultdict(list)
    for original_position, (resolved, score) in enumerate(zip(candidates, scores, strict=True)):
        candidate, _ = resolved
        lanes[candidate.query_index].append((original_position, resolved, score))
    for lane in lanes.values():
        lane.sort(
            key=lambda item: (
                -item[2],
                item[0],
                item[1][0].ref.source_unit_id,
                item[1][0].ref.source_revision,
            )
        )

    ordered: list[tuple[ResolvedCandidate, float]] = []
    lane_positions = {query_index: 0 for query_index in lanes}
    while len(ordered) < len(candidates):
        for query_index in sorted(lanes):
            position = lane_positions[query_index]
            lane = lanes[query_index]
            if position >= len(lane):
                continue
            _, resolved, score = lane[position]
            lane_positions[query_index] += 1
            ordered.append((resolved, score))

    return tuple(
        (
            replace(
                candidate,
                rerank_score=score,
                rerank_rank=rerank_rank,
            ),
            unit,
        )
        for rerank_rank, ((candidate, unit), score) in enumerate(ordered, start=1)
    )


__all__ = [
    "BgeM3Reranker",
    "RerankerUnavailable",
    "VerifiedRerankResult",
    "rerank_verified_candidates",
]
