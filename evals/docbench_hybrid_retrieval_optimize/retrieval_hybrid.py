"""Frozen-corpus evaluation through the production File hybrid retrieval owners.

Only Source and artifact boundaries are adapted here. Encoding, method indexes,
search, RRF, cross-encoder ordering, packing and relevance metrics retain their
production owners. No agent, query rewriting or answer generation is involved.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import shutil
import sqlite3
import tempfile
import time

from personagraph.retrieval.contracts import (
    FILE_RETRIEVAL_CANDIDATES_PER_QUERY, FILE_RETRIEVAL_RRF_LIMIT_PER_QUERY,
    MAX_FILE_RETRIEVAL_ITEMS, ContextStatus, MethodRunStatus, QueryProposal,
    RerankerRunStatus, RetrievalBudget, RetrievalMethod, RetrievalRequest,
    RetrievalStatus, RetrievalUnit, SourceAccess, SourceAvailability,
    SourceDependency, SourceFilter, SourceType, SourceUnit, TrustedRetrievalBoundary,
)
from personagraph.retrieval.compute.inference import release_device_cache
from personagraph.retrieval.indexing.adapters import (
    BM25Retrieval, DenseRetrieval, LearnedSparseRetrieval, SqliteRetrievalIndexWriter,
)
from personagraph.retrieval.indexing.encoder import BgeM3EncodedText
from personagraph.retrieval.indexing.methods import SqliteRetrievalMethodStore
from personagraph.retrieval.indexing.token_estimation import BgeM3RetrievalTokenEstimator
from personagraph.retrieval.policy import DefaultRetrievalPolicy
from personagraph.retrieval.profile import (
    DocumentRetrievalProfile, RetrievalFailurePolicy, RetrievalProfileMode,
    RetrievalRerankerMode, build_document_retrieval_runtime,
)
from personagraph.retrieval.query_guard import QueryGuard
from personagraph.retrieval.service import RetrievalService
from personagraph.retrieval.sqlite_store import SqliteRetrievalCatalog, UnitIndexState
from personagraph.retrieval.tooling.contracts import DEFAULT_FILE_RETRIEVAL_TOKEN_LIMIT

from .paths import PROJECT_ROOT, artifact_path
from .encoding_cache import open_encoding_cache
from .retrieval_eval import (
    QUESTION_KINDS, _dataset_manifest, _reference, _snapshot,
    _validate_visual_assets, score_rankings,
)


class HybridRetrievalError(ValueError):
    """The requested run cannot attest a complete production hybrid evaluation."""


METHODS = (RetrievalMethod.DENSE, RetrievalMethod.LEARNED_SPARSE, RetrievalMethod.BM25)
STAGES = ("dense", "learned_sparse", "bm25", "rrf", "reranker", "packed")
INDEX_SCHEMA = "docbench-hybrid-index-v1"
REPORT_SCHEMA = "docbench-hybrid-evaluation-v1"


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def _file_hash(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _digest(value: object) -> str:
    return sha256(_json(value).encode()).hexdigest()


def _emit(progress: Callable | None, stage: str, **values) -> None:
    if progress is not None:
        progress({"stage": stage, **values})


@contextmanager
def _offline():
    names = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE")
    previous = {name: os.environ.get(name) for name in names}
    os.environ.update(dict.fromkeys(names, "1"))
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


@dataclass
class _Runtime:
    encoder: object
    reranker: object
    passage_tokens: Callable[[str], int]
    pair_tokens: Callable[[str, str], int]
    identity: dict
    cleanup_device: Callable[[], dict] | None = None


def _mps_cleanup() -> dict:
    """Observe and release idle buffers after all GPU outputs became CPU values."""
    import torch

    def memory():
        return {
            "active_bytes": torch.mps.current_allocated_memory(),
            "driver_bytes": torch.mps.driver_allocated_memory(),
        }

    before = memory()
    reason = release_device_cache("mps")
    if reason is not None:
        raise HybridRetrievalError(f"MPS cache cleanup failed: {reason}")
    return {"before": before, "after": memory()}


class _ResourceUsage:
    """Boundary samples only: these are not intra-forward memory peaks."""

    def __init__(self, cleanup: Callable[[], dict] | None) -> None:
        self.cleanup = cleanup
        self.counts = Counter()
        self.peaks = {
            "before": {"active_bytes": 0, "driver_bytes": 0},
            "after": {"active_bytes": 0, "driver_bytes": 0},
        }
        self.last = None

    def sample_and_release(self, phase: str) -> dict | None:
        if self.cleanup is None:
            return None
        sample = self.cleanup()
        for moment in ("before", "after"):
            if not isinstance(sample, dict) or not isinstance(sample.get(moment), dict):
                raise HybridRetrievalError("device cleanup returned an invalid memory sample")
            for metric in ("active_bytes", "driver_bytes"):
                value = sample[moment].get(metric)
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise HybridRetrievalError("device cleanup returned an invalid memory sample")
                self.peaks[moment][metric] = max(self.peaks[moment][metric], value)
        self.counts[phase] += 1
        self.last = sample
        return sample

    def snapshot(self) -> dict:
        return {
            "enabled": self.cleanup is not None,
            "cleanup_count": sum(self.counts.values()),
            "cleanup_count_by_phase": dict(self.counts),
            "observed_peak_bytes": {moment: dict(values) for moment, values in self.peaks.items()},
            "last_sample": self.last,
            "sampling": "completed index batch/query boundaries; not intra-forward peak memory",
        }


def _build_runtime(device: str, batch_size: int) -> _Runtime:
    # Environment may select already-prepared local assets, never another route.
    profile = replace(
        DocumentRetrievalProfile.from_environment(), device=device, use_fp16=False,
        mode=RetrievalProfileMode.BGE_M3, failure_policy=RetrievalFailurePolicy.STRICT,
        reranker_mode=RetrievalRerankerMode.BGE_V2_M3, retrieval_methods_override=None,
    )
    runtime = build_document_retrieval_runtime(profile)
    if runtime.degraded_reason or runtime.reranker is None:
        raise HybridRetrievalError("complete local hybrid runtime is unavailable")
    if getattr(runtime.encoder, "learned_sparse_available", False) is not True:
        raise HybridRetrievalError("learned sparse model projection is unavailable")
    from transformers import AutoTokenizer

    encoder_asset = runtime.capability.encoder_asset
    reranker_asset = runtime.capability.reranker_asset
    if encoder_asset is None or reranker_asset is None:
        raise HybridRetrievalError("model asset preflight did not supply both models")
    encoder_tokenizer = AutoTokenizer.from_pretrained(
        str(encoder_asset.require_ready()), local_files_only=True, trust_remote_code=False,
    )
    reranker_tokenizer = AutoTokenizer.from_pretrained(
        str(reranker_asset.require_ready()), local_files_only=True, trust_remote_code=False,
    )
    is_mps = runtime.effective_profile.device in {"mps", "mps:0"}
    return _Runtime(
        encoder=runtime.encoder, reranker=runtime.reranker,
        passage_tokens=lambda text: len(encoder_tokenizer.encode(
            text, add_special_tokens=True, truncation=False,
        )),
        pair_tokens=lambda query, text: len(reranker_tokenizer.encode(
            query, text_pair=text, add_special_tokens=True, truncation=False,
        )),
        identity={
            "profile": runtime.effective_profile.fingerprint(),
            "encoder": runtime.encoder.fingerprint(),
            "reranker": runtime.reranker.fingerprint(),
            "encoder_asset": encoder_asset.generation_identity,
            "reranker_asset": reranker_asset.generation_identity,
            "device_requested": device, "device_effective": runtime.effective_profile.device,
            "precision": "fp32", "local_files_only": True,
            "index_input_batch_size": batch_size,
            "encoder_passage_max_length": 8192, "encoder_query_max_length": 512,
            "reranker_pair_max_length": 1024, "reranker_query_max_length": 256,
            "device_memory_policy": (
                "release_unused_mps_cache_after_each_encoded_index_batch_and_query"
                if is_mps else "no_device_cache_cleanup"
            ),
        },
        cleanup_device=_mps_cleanup if is_mps else None,
    )


class _BatchEncoder:
    """Feed one bounded batch of production encodings to the production writer."""

    def __init__(self, encoder, encoding_cache=None) -> None:
        self.encoder = encoder
        self.encoding_cache = encoding_cache
        self.cache = {}
        self.reused_units = 0
        self.newly_encoded_units = 0

    @property
    def learned_sparse_available(self):
        return self.encoder.learned_sparse_available

    def fingerprint(self):
        return self.encoder.fingerprint()

    def token_ids(self, text):
        return self.encoder.token_ids(text)

    def prepare(self, texts: Sequence[str]) -> None:
        encoded = [None] * len(texts)
        missing = []
        for position, text in enumerate(texts):
            cached = (
                self.encoding_cache.lookup(sha256(text.encode()).hexdigest())
                if self.encoding_cache is not None else None
            )
            if cached is None:
                missing.append(position)
            else:
                dense, sparse = cached
                # Only model representations are reused. Current canonical
                # tokenization remains the owner of the production BM25 input.
                encoded[position] = BgeM3EncodedText(
                    tuple(dense), dict(sparse), tuple(self.encoder.token_ids(text)),
                )
        fresh = tuple(self.encoder.encode(tuple(texts[position] for position in missing))) if missing else ()
        if len(fresh) != len(missing):
            raise HybridRetrievalError("encoder returned an unaligned index batch")
        for position, value in zip(missing, fresh, strict=True):
            encoded[position] = value
        for value in encoded:
            if (
                len(value.dense_vector) != 1024
                or not all(math.isfinite(number) for number in value.dense_vector)
                or not value.learned_sparse_weights
                or not all(math.isfinite(number) and number > 0 for number in value.learned_sparse_weights.values())
                or not value.token_ids
            ):
                raise HybridRetrievalError("encoder returned an incomplete/nonfinite representation")
        self.cache = dict(zip(texts, encoded, strict=True))
        self.reused_units += len(texts) - len(missing)
        self.newly_encoded_units += len(missing)

    def encode(self, texts):
        if any(text not in self.cache for text in texts):
            raise HybridRetrievalError("index writer requested content outside the prepared batch")
        return tuple(self.cache[text] for text in texts)


def _unit_scope(unit: dict) -> SourceFilter:
    return SourceFilter.from_mapping(_reference(unit).source_type, {"doc_id": unit["doc_id"], "kind": unit["kind"]})


class _FrozenSource:
    """Exact snapshot content and per-document authorization, without live Session state."""

    def __init__(self, source_type: SourceType, units: dict, snapshot_hash: str) -> None:
        self.source_type = source_type
        self.units = {key: value for key, value in units.items() if _reference(value).source_type is source_type}
        self.snapshot_hash = snapshot_hash

    def open_retrieval_access(self, source_filter: SourceFilter) -> SourceAccess:
        if source_filter.source_type is not self.source_type or set(source_filter.as_mapping()) != {"doc_id"}:
            raise HybridRetrievalError("invalid frozen Source scope")
        available = any(source_filter.selects(_unit_scope(unit)) for unit in self.units.values())
        return SourceAccess(
            self.source_type, source_filter,
            SourceAvailability.READY if available else SourceAvailability.EMPTY,
            source_snapshot_id=self.snapshot_hash,
        )

    def revalidate_retrieval_access(self, access):
        if access.source_snapshot_id != self.snapshot_hash:
            raise HybridRetrievalError("frozen Source identity changed")
        return self.open_retrieval_access(access.source_filter)

    def fetch_units(self, access, refs):
        self.revalidate_retrieval_access(access)
        result = []
        for ref in refs:
            unit = self.units.get(ref.source_unit_id)
            if unit is None or _reference(unit) != ref or not access.source_filter.selects(_unit_scope(unit)):
                raise HybridRetrievalError("candidate escaped its frozen Source scope")
            if sha256(unit["content"].encode()).hexdigest() != ref.indexed_content_hash:
                raise HybridRetrievalError("frozen Source content hash changed")
            result.append(SourceUnit(ref, unit["content"], {"doc_id": unit["doc_id"], "kind": unit["kind"]}))
        return tuple(result)


class _ObservedReranker:
    def __init__(self, runtime: _Runtime) -> None:
        self.runtime = runtime
        self.pairs = 0
        self.over_limit = 0
        self.max_pair_tokens = 0

    def fingerprint(self):
        return self.runtime.reranker.fingerprint()

    def score(self, pairs):
        lengths = [self.runtime.pair_tokens(query, text) for query, text in pairs]
        self.pairs += len(lengths)
        self.over_limit += sum(length > 1024 for length in lengths)
        self.max_pair_tokens = max([self.max_pair_tokens, *lengths])
        return self.runtime.reranker.score(pairs)

    def snapshot(self):
        return {
            "scored_pairs": self.pairs,
            "untruncated_pairs_over_1024": self.over_limit,
            "maximum_untruncated_pair_tokens": self.max_pair_tokens,
            "pair_count_method": "reranker tokenizer, query + passage + special tokens, before truncation",
            "production_pair_max_length": 1024,
        }


def _code_identity() -> dict:
    roots = [PROJECT_ROOT / "src/personagraph/retrieval"]
    files = [path for root in roots for path in root.rglob("*.py")]
    files += [Path(__file__), Path(__file__).with_name("retrieval_eval.py"), Path(__file__).with_name("paths.py"), Path(__file__).with_name("encoding_cache.py"), Path(__file__).with_name("index_archive.py")]
    hashes = {path.relative_to(PROJECT_ROOT).as_posix(): _file_hash(path) for path in sorted(files)}
    return {"sha256": _digest(hashes), "files": hashes}


def _token_preflight(units: dict, queries: dict, runtime: _Runtime, progress) -> tuple[dict, dict]:
    unit_counts = []
    unit_lengths = {}
    query_counts = []
    guard = QueryGuard(count_tokens=lambda text: len(tuple(runtime.encoder.token_ids(text))))
    for index, unit in enumerate(units.values(), 1):
        count = runtime.passage_tokens(unit["content"])
        if count > 8192:
            raise HybridRetrievalError(f"corpus unit exceeds embedding limit: {unit['unit_id']}")
        unit_counts.append(count)
        unit_lengths[unit["unit_id"]] = count
        if index % 256 == 0 or index == len(units):
            _emit(progress, "token_preflight", completed=index, total=len(units))
    for query in queries.values():
        proposal = QueryProposal((query["query"],))
        if guard.validate(proposal) != (query["query"],):
            raise HybridRetrievalError("production query guard would rewrite the original question")
        query_counts.append(runtime.passage_tokens(query["query"]))
    return {
        "unit_count": len(unit_counts), "unit_total_tokens": sum(unit_counts),
        "unit_max_tokens": max(unit_counts, default=0),
        "units_over_8192": 0, "units_over_1024": sum(value > 1024 for value in unit_counts),
        "query_count": len(query_counts), "query_max_tokens": max(query_counts, default=0),
        "includes_special_tokens": True, "truncation": False,
    }, unit_lengths


def _require_capabilities(catalog) -> None:
    for name in ("fts5", "sqlite_vec"):
        capability = catalog.capability(name)
        if capability is None or capability[0] is not True:
            raise HybridRetrievalError(f"required production SQLite capability unavailable: {name}")


def _verify_catalog(catalog, store, units: dict, generation: str, *, require_active: bool) -> dict:
    stored = catalog.list_stored_units(generation)
    if len(stored) != len(units):
        raise HybridRetrievalError("index unit coverage differs from frozen corpus")
    seen = set()
    for entry in stored:
        unit = units.get(entry.unit.ref.source_unit_id)
        if (
            unit is None or entry.unit.ref != _reference(unit)
            or entry.unit.source_filter != _unit_scope(unit)
            or entry.unit.retrieval_status is not RetrievalStatus.ACTIVE
            or entry.index_state is not UnitIndexState.READY
            or entry.unit.ref in seen
        ):
            raise HybridRetrievalError("index contains a stale, duplicate or out-of-scope Source binding")
        seen.add(entry.unit.ref)
    health = store.generation_method_index_health(generation)
    if len(health) != len(units) or any(
        len(entries) != len(METHODS)
        or {entry.method for entry in entries} != set(METHODS)
        or any(entry.expected_state != "ready" or not entry.representation_present or entry.reason_code for entry in entries)
        for entries in health.values()
    ):
        raise HybridRetrievalError("production index lacks complete three-method coverage")
    if any(store.orphaned_representation_unit_ids().values()):
        raise HybridRetrievalError("index contains orphaned representations")
    if require_active and catalog.active_retrieval_data_version_id() != generation:
        raise HybridRetrievalError("index active generation does not match its frozen manifest")
    return {method.value: len(units) for method in METHODS}


def _index_recipe(units, dataset_hash, runtime, code) -> dict:
    corpus = [{key: unit[key] for key in ("unit_id", "doc_id", "kind", "source_revision", "content_sha256")} for unit in sorted(units.values(), key=lambda row: row["unit_id"])]
    return {
        "dataset_sha256": dataset_hash, "corpus_sha256": _digest(corpus),
        "unit_count": len(units), "unit_kinds": dict(Counter(unit["kind"] for unit in units.values())),
        "encoder_fingerprint": runtime.encoder.fingerprint(),
        "code_sha256": code["sha256"],
        "index_input_batch_size": runtime.identity["index_input_batch_size"],
        "index_batch_order": "ascending untruncated passage token length, then unit_id",
        "methods": [method.value for method in METHODS], "embedding_dimensions": 1024,
        "scope_fields": ["doc_id", "kind"], "indexed_fields": ["units.content"],
        "label_fields_indexed": False,
        "encoding_reuse_rule": (
            "exact encoder fingerprint and target content SHA256 lookup; cache attests exact content "
            "or verified strip model-input equivalence; current canonical token IDs"
        ),
    }


def _checkpoint(catalog) -> None:
    with catalog.connect() as connection:
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")


def _prepare_index(index_dir, units, runtime, recipe, batch_size, unit_lengths, resources, progress, encoding_cache=None) -> tuple[dict, bool]:
    manifest_path = index_dir / "manifest.json"
    if index_dir.exists():
        if not index_dir.is_dir() or not manifest_path.is_file():
            raise HybridRetrievalError("existing index has no complete manifest; use a new index directory")
        manifest = json.loads(manifest_path.read_text())
        database = index_dir / "retrieval.sqlite"
        if manifest.get("schema_version") != INDEX_SCHEMA or manifest.get("recipe") != recipe:
            raise HybridRetrievalError("existing index recipe differs from corpus, model, parameters or code")
        if not database.is_file() or manifest.get("database_sha256") != _file_hash(database):
            raise HybridRetrievalError("existing index database hash mismatch")
        for suffix in ("-wal", "-journal"):
            sidecar = Path(str(database) + suffix)
            if sidecar.exists() and sidecar.stat().st_size:
                raise HybridRetrievalError("published index has a pending SQLite transaction")
        _emit(progress, "index_reused", completed=len(units), total=len(units))
        return manifest, True
    index_dir.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    cache_context = (
        open_encoding_cache(encoding_cache, expected_encoder_fingerprint=runtime.encoder.fingerprint())
        if encoding_cache is not None else nullcontext(None)
    )
    with cache_context as cached, tempfile.TemporaryDirectory(prefix=f".{index_dir.name}-", dir=index_dir.parent) as temporary:
        stage = Path(temporary)
        catalog = SqliteRetrievalCatalog(stage / "retrieval.sqlite")
        catalog.initialize()
        batched = _BatchEncoder(runtime.encoder, cached)
        store = SqliteRetrievalMethodStore(catalog=catalog, encoder=batched)
        _require_capabilities(catalog)
        generation = "docbench-" + _digest(recipe)
        catalog.create_data_version(version_id=generation, fingerprint=_digest(recipe))
        writer = SqliteRetrievalIndexWriter(store, required_methods=METHODS)
        ordered = sorted(units.values(), key=lambda row: (unit_lengths[row["unit_id"]], row["unit_id"]))
        for offset in range(0, len(ordered), batch_size):
            batch = ordered[offset:offset + batch_size]
            encoded_before_batch = batched.newly_encoded_units
            batched.prepare(tuple(unit["content"] for unit in batch))
            for unit in batch:
                ref = _reference(unit)
                stored = catalog.upsert_pending_unit(RetrievalUnit(
                    ref, generation, source_filter=_unit_scope(unit),
                ))
                writer.index(stored, SourceUnit(ref, unit["content"]))
                catalog.mark_unit_index_ready(stored.unit_id)
            batched.cache.clear()
            memory = (
                resources.sample_and_release("index_batch")
                if batched.newly_encoded_units > encoded_before_batch else None
            )
            _emit(progress, "index_build", completed=offset + len(batch), total=len(ordered), elapsed_seconds=time.monotonic() - started,
                  device_memory=memory, device_memory_summary=resources.snapshot(),
                  reused_units=batched.reused_units, newly_encoded_units=batched.newly_encoded_units)
        coverage = _verify_catalog(catalog, store, units, generation, require_active=False)
        catalog.mark_data_version_ready(generation)
        catalog.activate_data_version(generation)
        _checkpoint(catalog)
        manifest = {
            "schema_version": INDEX_SCHEMA, "recipe": recipe, "generation": generation,
            "database_sha256": _file_hash(stage / "retrieval.sqlite"), "coverage": coverage,
            "build_seconds": time.monotonic() - started,
            "encoding": {
                "reused_units": batched.reused_units,
                "newly_encoded_units": batched.newly_encoded_units,
                "total_units": len(units),
                "cache_provenance": (
                    {"manifest_sha256": cached.manifest_sha256, "manifest": cached.safe_manifest}
                    if cached is not None else None
                ),
            },
        }
        (stage / "manifest.json").write_text(_json(manifest) + "\n")
        if index_dir.exists():
            raise HybridRetrievalError("index directory appeared during build; refusing overwrite")
        stage.rename(index_dir)
    return manifest, False


def _request(case_id: str, query: dict, sources: dict) -> RetrievalRequest:
    kinds = QUESTION_KINDS[query["question_type"]]
    source_type = SourceType.DOCUMENT if kinds == QUESTION_KINDS["text-only"] else SourceType.PICTURE
    scope = SourceFilter.from_mapping(source_type, {"doc_id": query["doc_id"]})
    access = sources[source_type].open_retrieval_access(scope)
    return RetrievalRequest(
        request_id=case_id, model_call_purpose="docbench_frozen_retrieval",
        query_proposal=QueryProposal((query["query"],)),
        boundary=TrustedRetrievalBoundary(
            {source_type: scope}, {source_type: SourceDependency.REQUIRED}, {source_type: access},
        ),
    )


def _strict_context(context, case_id: str) -> None:
    if (
        context.status is ContextStatus.BLOCKED or context.verification_drops
        # The production result explicitly lists every Source omitted by the
        # trusted boundary. Those deliberate exclusions are not coverage loss
        # inside this experiment's document/modality candidate scope.
        or any(value.reason_code != "source_excluded_by_trusted_boundary" for value in context.coverage_limitations)
        or {outcome.method for outcome in context.method_outcomes} != set(METHODS)
        or any(outcome.status is not MethodRunStatus.USED or outcome.degraded_from or outcome.attempt_failures for outcome in context.method_outcomes)
        or any(outcome.status is RerankerRunStatus.DEGRADED for outcome in context.reranker_outcomes)
        or (context.fusion_candidates and not any(outcome.status is RerankerRunStatus.USED for outcome in context.reranker_outcomes))
    ):
        raise HybridRetrievalError(f"incomplete or degraded production hybrid route for case {case_id}")


def _rankings(context) -> dict:
    stages = {
        method.value: [candidate.ref.source_unit_id for candidate in sorted(
            (candidate for candidate in context.lane_candidates if candidate.method is method), key=lambda item: item.rank,
        )]
        for method in METHODS
    }
    stages["rrf"] = [candidate.ref.source_unit_id for candidate in context.fusion_candidates]
    stages["reranker"] = [candidate.ref.source_unit_id for candidate in context.reranker_candidates]
    stages["packed"] = [item.ref.source_unit_id for item in context.items]
    return stages


def _run_queries(stage, index_dir, manifest, units, queries, dataset_hash, runtime, resources, progress):
    # Production store refreshes capability timestamps. Keep the published index
    # immutable and run its normal initialization against an exact working copy.
    working = stage / "working-index.sqlite"
    shutil.copyfile(index_dir / "retrieval.sqlite", working)
    catalog = SqliteRetrievalCatalog(working)
    store = SqliteRetrievalMethodStore(catalog=catalog, encoder=runtime.encoder)
    _require_capabilities(catalog)
    coverage = _verify_catalog(catalog, store, units, manifest["generation"], require_active=True)
    sources = {source_type: _FrozenSource(source_type, units, dataset_hash) for source_type in (SourceType.DOCUMENT, SourceType.PICTURE)}
    estimator = BgeM3RetrievalTokenEstimator(runtime.encoder)
    reranker = _ObservedReranker(runtime)
    service = RetrievalService(
        policy=DefaultRetrievalPolicy(default_methods=METHODS),
        query_guard=QueryGuard(count_tokens=lambda text: len(tuple(runtime.encoder.token_ids(text)))),
        sources=sources, readonly_sources=sources,
        source_access_revalidators=sources, readonly_source_access_revalidators=sources,
        methods={RetrievalMethod.DENSE: DenseRetrieval(store), RetrievalMethod.LEARNED_SPARSE: LearnedSparseRetrieval(store), RetrievalMethod.BM25: BM25Retrieval(store)},
        data_version_provider=catalog, token_estimator=estimator,
        encoder_fingerprint=runtime.encoder.fingerprint(), reranker=reranker,
    )
    budget = RetrievalBudget(
        candidate_limit_per_source=FILE_RETRIEVAL_CANDIDATES_PER_QUERY,
        context_token_limit=DEFAULT_FILE_RETRIEVAL_TOKEN_LIMIT, max_items=MAX_FILE_RETRIEVAL_ITEMS,
    )
    rankings = {name: {} for name in STAGES}
    started = time.monotonic()
    with (stage / "rankings.jsonl").open("w") as ranking_stream, (stage / "diagnostics.jsonl").open("w") as diagnostic_stream:
        for index, (case_id, query) in enumerate(sorted(queries.items()), 1):
            before_pairs = reranker.snapshot()
            case_started = time.monotonic()
            context, = service.retrieve_file_query_batch(
                (_request(case_id, query, sources),), budget, result_limit=MAX_FILE_RETRIEVAL_ITEMS, readonly=True,
            )
            memory = resources.sample_and_release("query")
            _strict_context(context, case_id)
            stages = _rankings(context)
            for name, ranked in stages.items():
                rankings[name][case_id] = ranked
            candidates = sum(unit["doc_id"] == query["doc_id"] and unit["kind"] in QUESTION_KINDS[query["question_type"]] for unit in units.values())
            ranking_stream.write(_json({
                "case_id": case_id, "doc_id": query["doc_id"], "question_type": query["question_type"],
                "candidate_count": candidates, "stages": stages,
            }) + "\n")
            diagnostic_stream.write(_json({
                "case_id": case_id, "status": context.status.value,
                "method_outcomes": [asdict(value) for value in context.method_outcomes],
                "reranker_outcomes": [asdict(value) for value in context.reranker_outcomes],
                "diagnostic_codes": context.diagnostic_codes,
                "coverage_limitations": [asdict(value) for value in context.coverage_limitations],
                "packed_tokens": context.packed_tokens,
                "pack_omissions": [asdict(value) for value in context.pack_omissions],
                "reranker_pairs": reranker.pairs - before_pairs["scored_pairs"],
                "reranker_pairs_over_1024": reranker.over_limit - before_pairs["untruncated_pairs_over_1024"],
                "elapsed_seconds": time.monotonic() - case_started,
                "device_memory": memory,
            }) + "\n")
            ranking_stream.flush()
            diagnostic_stream.flush()
            _emit(progress, "query", completed=index, total=len(queries), case_id=case_id, elapsed_seconds=time.monotonic() - started,
                  device_memory=memory, device_memory_summary=resources.snapshot())
    if estimator.diagnostic_snapshot().fallback_count:
        raise HybridRetrievalError("production packing token estimator degraded")
    _checkpoint(catalog)
    for suffix in ("", "-wal", "-shm"):
        Path(str(working) + suffix).unlink(missing_ok=True)
    return rankings, {"query_seconds": time.monotonic() - started, "index_coverage": coverage, "reranker": reranker.snapshot()}


def _markdown_report(report: dict) -> str:
    lines = [
        "# DocBench production hybrid retrieval", "",
        "One original question per query; document-scoped candidates; no LLM or answer generation.",
        "All corpus units are indexed. Text questions search chunks; multimodal questions search all visual descriptions.",
        "Metrics score reviewed primary labels. Flat qrels measure evidence-unit coverage, not answer completeness.", "",
        "| Stage | Cases | MRR | Hit@5 | Recall@5 | nDCG@5 |", "|---|---:|---:|---:|---:|---:|",
    ]
    for name in STAGES:
        overall = report["stages"][name]["overall"]
        if overall:
            lines.append(f"| {name} | {overall['case_count']} | {overall['mean_reciprocal_rank']:.4f} | {overall['hit_rate_at_k'][5]:.4f} | {overall['mean_recall_at_k'][5]:.4f} | {overall['mean_ndcg_at_k'][5]:.4f} |")
    lines += ["", "Per-type results and all cutoffs (1, 3, 5, 10) are in report.json; per-query ranks are in rankings.jsonl.",
              "Production BM25 uses BGE token-ID shadow terms; it differs from the Unicode61 baseline.",
              "Candidate limits: 128 per method/query, RRF top 64, final at most 96 items and 96,000 estimated tokens.",
              "Reranker input is limited to 1,024 tokens per pair; overlong pairs are counted in diagnostics.",
              "The published index remains immutable; production initialization runs against a temporary exact copy.", ""]
    return "\n".join(lines)


def run_hybrid_retrieval_evaluation(
    dataset_path: Path, index_dir: Path, output_dir: Path, *, device: str = "cpu",
    batch_size: int = 8, allow_live: bool = False, progress: Callable | None = None,
    encoding_cache: Path | None = None,
) -> dict:
    """Build/reuse a complete local production index and score all six stages."""
    if allow_live is not True:
        raise HybridRetrievalError("real local model evaluation requires allow_live=True")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise HybridRetrievalError("batch_size must be a positive integer")
    started = time.monotonic()
    try:
        dataset, index, output = map(artifact_path, (dataset_path, index_dir, output_dir))
        if output.exists():
            raise HybridRetrievalError("evaluation output already exists; refusing overwrite")
        if index == output or index.is_relative_to(output) or output.is_relative_to(index):
            raise HybridRetrievalError("index and evaluation output directories must be disjoint")
        if dataset.is_relative_to(index) or dataset.is_relative_to(output):
            raise HybridRetrievalError("dataset must be separate from index and evaluation output")
        dataset_manifest, manifest_hash = _dataset_manifest(dataset)
        dataset_hash = dataset_manifest["dataset_sha256"]
        with sqlite3.connect(f"{dataset.as_uri()}?mode=ro&immutable=1", uri=True) as connection, _offline():
            connection.execute("PRAGMA query_only=ON")
            connection.execute("BEGIN")
            _validate_visual_assets(connection, dataset.parent)
            units, queries, _ = _snapshot(connection)
            if not units or not queries:
                raise HybridRetrievalError("hybrid evaluation requires nonempty corpus and queries")
            code = _code_identity()
            _emit(progress, "runtime_preflight", total=len(units), queries=len(queries))
            runtime = _build_runtime(device, batch_size)
            resources = _ResourceUsage(runtime.cleanup_device)
            tokens, unit_lengths = _token_preflight(units, queries, runtime, progress)
            recipe = _index_recipe(units, dataset_hash, runtime, code)
            index_manifest, reused = _prepare_index(index, units, runtime, recipe, batch_size, unit_lengths, resources, progress, encoding_cache)
            output.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
                stage = Path(temporary)
                rankings, execution = _run_queries(stage, index, index_manifest, units, queries, dataset_hash, runtime, resources, progress)
                execution["device_memory"] = resources.snapshot()
                metrics = {name: score_rankings(connection, ranked, route_id="production_file_hybrid:" + name) for name, ranked in rankings.items()}
                if _file_hash(dataset) != dataset_hash or _file_hash(index / "retrieval.sqlite") != index_manifest["database_sha256"]:
                    raise HybridRetrievalError("frozen dataset or published index changed during evaluation")
                if _code_identity() != code:
                    raise HybridRetrievalError("retrieval implementation changed during evaluation")
                report = {
                    "schema_version": REPORT_SCHEMA, "status": "complete", "route_id": "production_file_hybrid",
                    "dataset_sha256": dataset_hash, "dataset_manifest_sha256": manifest_hash,
                    "corpus_sha256": recipe["corpus_sha256"], "index_database_sha256": index_manifest["database_sha256"],
                    "index_manifest_sha256": _file_hash(index / "manifest.json"), "index_reused": reused,
                    "index_encoding": index_manifest["encoding"],
                    "runtime": runtime.identity, "code": code, "token_preflight": tokens,
                    "parameters": {
                        "candidate_limit_per_method_query": FILE_RETRIEVAL_CANDIDATES_PER_QUERY,
                        "rrf_k": 60, "rrf_limit_per_query": FILE_RETRIEVAL_RRF_LIMIT_PER_QUERY,
                        "result_limit": MAX_FILE_RETRIEVAL_ITEMS, "context_token_limit": DEFAULT_FILE_RETRIEVAL_TOKEN_LIMIT,
                        "learned_sparse_query_top_weights": 256, "bm25_query_prefix_tokens": 256,
                        "passage_sparse_weights": "all positive production weights", "bm25_passage_tokens": "all production token IDs",
                    },
                    "experiment": {"query": "one unchanged original question", "scope": "question_document_and_allowed_kinds", "llm_calls": 0, "network_calls": 0},
                    "coverage": metrics["packed"]["coverage"], "stages": metrics,
                    "execution": execution, "elapsed_seconds": time.monotonic() - started,
                }
                (stage / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
                (stage / "REPORT.md").write_text(_markdown_report(report))
                hashes = {path.name: _file_hash(path) for path in sorted(stage.iterdir()) if path.is_file()}
                (stage / "manifest.json").write_text(_json({"schema_version": REPORT_SCHEMA, "files_sha256": hashes}) + "\n")
                if output.exists():
                    raise HybridRetrievalError("evaluation output appeared during run; refusing overwrite")
                stage.rename(output)
        _emit(progress, "complete", completed=len(queries), total=len(queries))
        return report
    except HybridRetrievalError:
        raise
    except (OSError, ValueError, RuntimeError, sqlite3.Error) as error:
        raise HybridRetrievalError(f"Cannot complete production hybrid evaluation: {error}") from error


__all__ = ["HybridRetrievalError", "run_hybrid_retrieval_evaluation"]
