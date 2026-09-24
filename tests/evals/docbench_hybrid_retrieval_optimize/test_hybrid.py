"""Production SQLite/File hybrid integration with synthetic content and mock models."""

from collections import Counter
from contextlib import contextmanager
from hashlib import sha256
import json
import sqlite3
import sys
from types import SimpleNamespace

import pytest

from evals.docbench_hybrid_retrieval_optimize import retrieval_hybrid as hybrid
from personagraph.retrieval.indexing.encoder import BgeM3EncodedText
from personagraph.retrieval.indexing.methods import SqliteRetrievalMethodStore


class Encoder:
    learned_sparse_available = True

    def __init__(self):
        self.calls = []
        self.empty_sparse = False

    def fingerprint(self):
        return "synthetic-encoder-fp32"

    def token_ids(self, text):
        return tuple(int.from_bytes(sha256(word.encode()).digest()[:2], "big") for word in text.split())

    def encode(self, texts):
        self.calls.append(tuple(texts))
        encoded = []
        for text in texts:
            counts = Counter(self.token_ids(text))
            dense = [0.0] * 1024
            for token, count in counts.items():
                dense[token % 1024] += count
            magnitude = sum(value * value for value in dense) ** 0.5
            encoded.append(BgeM3EncodedText(
                tuple(value / magnitude for value in dense),
                {} if self.empty_sparse else {key: float(value) for key, value in counts.items()},
                self.token_ids(text),
            ))
        return tuple(encoded)

    def encode_query(self, query):
        return self.encode((query,))[0]


class Reranker:
    def __init__(self):
        self.pairs = []
        self.fail = False

    def fingerprint(self):
        return "synthetic-cross-encoder"

    def score(self, pairs):
        self.pairs.extend(pairs)
        if self.fail:
            raise RuntimeError("synthetic reranker failure")
        return tuple(10.0 if "target" in text else 1.0 for _, text in pairs)


@pytest.fixture
def corpus(tmp_path):
    root = tmp_path / "dataset"
    root.mkdir()
    database = root / "dataset.sqlite"
    rows = [
        ("a", "doc-a", "chunk", "alpha alpha alpha"),
        ("b", "doc-a", "chunk", "alpha target"),
        ("copy", "doc-a", "chunk", "alpha target"),
        ("picture", "doc-a", "figure", "alpha alpha alpha"),
        ("foreign", "doc-b", "chunk", "alpha alpha alpha alpha"),
        ("table", "doc-a", "table", "beta distractor"),
        ("vector", "doc-a", "vector_graphics", "beta target longpair"),
        ("foreign-picture", "doc-b", "figure", "beta beta beta"),
    ]
    with sqlite3.connect(database) as connection:
        connection.executescript('''
            CREATE TABLE documents(doc_id TEXT PRIMARY KEY);
            CREATE TABLE units(unit_id TEXT PRIMARY KEY, doc_id TEXT, kind TEXT, content TEXT,
                source_revision TEXT, content_sha256 TEXT, asset_path TEXT, metadata_json TEXT);
            CREATE TABLE queries(case_id TEXT PRIMARY KEY, doc_id TEXT, question_type TEXT,
                query TEXT, annotation_status TEXT, answer TEXT, evidence TEXT);
            CREATE TABLE qrels(case_id TEXT, unit_id TEXT, role TEXT);
            INSERT INTO documents VALUES ('doc-a'), ('doc-b');
            INSERT INTO queries VALUES ('text', 'doc-a', 'text-only', 'alpha', 'reviewed', 'secretanswer', 'secretevidence');
            INSERT INTO queries VALUES ('table', 'doc-a', 'multimodal-t', 'beta', 'reviewed', 'secretanswer', 'secretevidence');
            INSERT INTO queries VALUES ('figure', 'doc-a', 'multimodal-f', 'beta', 'reviewed', 'secretanswer', 'secretevidence');
            INSERT INTO qrels VALUES ('text', 'b', 'primary'), ('table', 'vector', 'primary'), ('figure', 'vector', 'primary');
        ''')
        for identifier, doc_id, kind, text in rows:
            asset, metadata = None, {}
            if kind != "chunk":
                asset = f"assets/{identifier}.png"
                path = root / asset
                path.parent.mkdir(exist_ok=True)
                path.write_bytes(b"synthetic image " + identifier.encode())
                metadata = {"asset_sha256": sha256(path.read_bytes()).hexdigest()}
            connection.execute("INSERT INTO units VALUES (?, ?, ?, ?, ?, ?, ?, ?)", (
                identifier, doc_id, kind, text, "frozen", sha256(text.encode()).hexdigest(), asset, json.dumps(metadata),
            ))
    _pin(database)
    return database


def _pin(database):
    (database.parent / "manifest.json").write_text(json.dumps({
        "schema_version": "docbench-retrieval-dataset-v1",
        "dataset_sha256": sha256(database.read_bytes()).hexdigest(),
    }))


@pytest.fixture
def runtime(monkeypatch):
    encoder, reranker = Encoder(), Reranker()
    bundle = hybrid._Runtime(
        encoder, reranker,
        passage_tokens=lambda text: len(text.split()) + 2,
        pair_tokens=lambda query, text: 1100 if "longpair" in text else len(query.split()) + len(text.split()) + 4,
        identity={"index_input_batch_size": 2, "encoder": encoder.fingerprint(), "reranker": reranker.fingerprint(), "device_effective": "synthetic-cpu", "precision": "fp32"},
    )
    monkeypatch.setattr(hybrid, "_build_runtime", lambda device, batch_size: bundle)
    return bundle


def _run(corpus, tmp_path, **kwargs):
    return hybrid.run_hybrid_retrieval_evaluation(
        corpus, tmp_path / "index", tmp_path / "result", batch_size=2, allow_live=True, **kwargs,
    )


def test_full_production_search_rrf_rerank_and_packing_keep_exact_scope(corpus, tmp_path, runtime):
    original = corpus.read_bytes()
    events = []
    report = _run(corpus, tmp_path, progress=events.append)
    assert report["status"] == "complete"
    assert report["coverage"]["scored_queries"] == 3
    assert report["execution"]["index_coverage"] == {"dense": 8, "learned_sparse": 8, "bm25": 8}
    assert report["execution"]["reranker"]["untruncated_pairs_over_1024"] == 2
    assert report["token_preflight"]["units_over_8192"] == 0
    assert report["execution"]["device_memory"]["enabled"] is False
    assert report["execution"]["device_memory"]["cleanup_count"] == 0
    assert set(report["stages"]) == set(hybrid.STAGES)
    assert report["stages"]["packed"]["overall"]["hit_rate_at_k"][1] == 1.0
    assert report["stages"]["packed"]["by_question_type"]["multimodal-t"]["case_count"] == 1
    records = {value["case_id"]: value for value in map(json.loads, (tmp_path / "result/rankings.jsonl").read_text().splitlines())}
    assert records["text"]["candidate_count"] == 3
    assert records["figure"]["candidate_count"] == 3
    assert records["text"]["stages"]["dense"][0] == "a"
    assert records["text"]["stages"]["reranker"][0] == "b"
    for record in records.values():
        allowed = {"a", "b", "copy"} if record["case_id"] == "text" else {"picture", "table", "vector"}
        for ranked in record["stages"].values():
            assert set(ranked) <= allowed
    assert {query for query, _ in runtime.reranker.pairs} == {"alpha", "beta"}
    assert all("secret" not in text for batch in runtime.encoder.calls for text in batch)
    assert corpus.read_bytes() == original
    assert not (tmp_path / "result/working-index.sqlite").exists()
    assert events[-1]["stage"] == "complete"
    assert [event["completed"] for event in events if event["stage"] == "index_build"] == [2, 4, 6, 8]
    manifest = json.loads((tmp_path / "result/manifest.json").read_text())
    for filename, digest in manifest["files_sha256"].items():
        assert sha256((tmp_path / "result" / filename).read_bytes()).hexdigest() == digest
    assert str(tmp_path) not in (tmp_path / "result/report.json").read_text()


def test_reuses_verified_index_without_reencoding_corpus_or_mutating_published_db(corpus, tmp_path, runtime):
    first = _run(corpus, tmp_path)
    index = tmp_path / "index/retrieval.sqlite"
    before = index.read_bytes()
    runtime.encoder.calls.clear()
    second = hybrid.run_hybrid_retrieval_evaluation(corpus, tmp_path / "index", tmp_path / "second", batch_size=2, allow_live=True)
    assert first["index_reused"] is False and second["index_reused"] is True
    assert first["stages"] == second["stages"]
    assert index.read_bytes() == before
    assert all(text in {"alpha", "beta"} for batch in runtime.encoder.calls for text in batch)


@pytest.mark.parametrize("defect", ("hash", "recipe", "coverage", "scope"))
def test_reuse_rejects_corrupt_or_incomplete_index(corpus, tmp_path, runtime, defect):
    _run(corpus, tmp_path)
    manifest_path = tmp_path / "index/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    database = tmp_path / "index/retrieval.sqlite"
    if defect == "hash":
        manifest["database_sha256"] = "0" * 64
    elif defect == "recipe":
        manifest["recipe"]["encoder_fingerprint"] = "other-encoder"
    else:
        # Use the production connection so SQLite-vec virtual tables are loaded.
        catalog = hybrid.SqliteRetrievalCatalog(database)
        with catalog.connect() as connection:
            if defect == "coverage":
                connection.execute("DELETE FROM learned_sparse_postings WHERE unit_id=(SELECT MIN(unit_id) FROM retrieval_units)")
            else:
                connection.execute("UPDATE retrieval_units SET scope_json='{}' WHERE unit_id=(SELECT MIN(unit_id) FROM retrieval_units)")
        manifest["database_sha256"] = hybrid._file_hash(database)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(hybrid.HybridRetrievalError, match="hash mismatch|recipe differs|coverage|scope"):
        hybrid.run_hybrid_retrieval_evaluation(corpus, tmp_path / "index", tmp_path / "second", batch_size=2, allow_live=True)
    assert not (tmp_path / "second").exists()
    assert not list(tmp_path.glob(".second-*"))


def test_index_failure_does_not_publish_partial_index_or_results(corpus, tmp_path, runtime):
    runtime.encoder.empty_sparse = True
    with pytest.raises(hybrid.HybridRetrievalError, match="incomplete"):
        _run(corpus, tmp_path)
    assert not (tmp_path / "index").exists()
    assert not (tmp_path / "result").exists()
    assert not list(tmp_path.glob(".index-*"))


def test_method_failure_cannot_be_reported_as_full_hybrid(corpus, tmp_path, runtime, monkeypatch):
    def unavailable(*args, **kwargs):
        raise RuntimeError("synthetic dense unavailable")
    monkeypatch.setattr(SqliteRetrievalMethodStore, "search_dense", unavailable)
    with pytest.raises(hybrid.HybridRetrievalError, match="degraded"):
        _run(corpus, tmp_path)
    assert (tmp_path / "index/manifest.json").is_file()
    assert not (tmp_path / "result").exists()


def test_reranker_failure_cannot_publish_rrf_fallback_as_hybrid(corpus, tmp_path, runtime):
    runtime.reranker.fail = True
    with pytest.raises(hybrid.HybridRetrievalError, match="degraded"):
        _run(corpus, tmp_path)
    assert not (tmp_path / "result").exists()
    assert not list(tmp_path.glob(".result-*"))


def test_embedding_overflow_fails_before_index_build(corpus, tmp_path, runtime):
    runtime.passage_tokens = lambda text: 8193
    with pytest.raises(hybrid.HybridRetrievalError, match="exceeds embedding"):
        _run(corpus, tmp_path)
    assert not runtime.encoder.calls
    assert not (tmp_path / "index").exists()


def test_live_opt_in_is_required_before_loading_models_or_creating_artifacts(tmp_path, monkeypatch):
    before = set(tmp_path.iterdir())
    def forbidden(*args):
        pytest.fail("must not load models without explicit opt-in")
    monkeypatch.setattr(hybrid, "_build_runtime", forbidden)
    with pytest.raises(hybrid.HybridRetrievalError, match="allow_live"):
        hybrid.run_hybrid_retrieval_evaluation(tmp_path / "missing", tmp_path / "index", tmp_path / "result")
    assert set(tmp_path.iterdir()) == before


def test_output_is_not_overwritten_and_dataset_hash_is_verified(corpus, tmp_path, runtime):
    (tmp_path / "result").mkdir()
    with pytest.raises(hybrid.HybridRetrievalError, match="already exists"):
        _run(corpus, tmp_path)
    assert not runtime.encoder.calls
    (tmp_path / "result").rmdir()
    with sqlite3.connect(corpus) as connection:
        connection.execute("UPDATE queries SET query='changed'")
    with pytest.raises(hybrid.HybridRetrievalError, match="hash"):
        _run(corpus, tmp_path)
    assert not runtime.encoder.calls


def test_cleanup_occurs_after_cleared_index_batches_and_scored_queries(corpus, tmp_path, runtime, monkeypatch):
    batches = []
    original = hybrid._BatchEncoder

    class RecordingBatchEncoder(original):
        def __init__(self, encoder, encoding_cache=None):
            super().__init__(encoder, encoding_cache)
            batches.append(self)

    monkeypatch.setattr(hybrid, "_BatchEncoder", RecordingBatchEncoder)
    calls = []

    def cleanup():
        assert batches and not batches[0].cache
        if len(calls) < 4:
            assert not runtime.reranker.pairs
        else:
            assert runtime.reranker.pairs
        number = len(calls) + 1
        sample = {
            "before": {"active_bytes": 100, "driver_bytes": 1000 + number},
            "after": {"active_bytes": 100, "driver_bytes": 200 + number},
        }
        calls.append(sample)
        return sample

    runtime.cleanup_device = cleanup
    runtime.identity["device_memory_policy"] = "synthetic_boundary_cleanup"
    events = []
    report = _run(corpus, tmp_path, progress=events.append)
    summary = report["execution"]["device_memory"]
    assert len(calls) == 7
    assert summary["cleanup_count_by_phase"] == {"index_batch": 4, "query": 3}
    assert summary["observed_peak_bytes"] == {
        "before": {"active_bytes": 100, "driver_bytes": 1007},
        "after": {"active_bytes": 100, "driver_bytes": 207},
    }
    assert summary["last_sample"] == calls[-1]
    assert "not intra-forward" in summary["sampling"]
    observed = [event for event in events if event["stage"] in {"index_build", "query"}]
    assert [event["device_memory"] for event in observed] == calls
    assert [event["device_memory_summary"]["cleanup_count"] for event in observed] == list(range(1, 8))
    rows = list(map(json.loads, (tmp_path / "result/diagnostics.jsonl").read_text().splitlines()))
    assert [row["device_memory"] for row in rows] == calls[-3:]


@pytest.mark.parametrize("fail_at", (1, 5))
def test_cleanup_failure_prevents_partial_index_or_query_result_publication(corpus, tmp_path, runtime, fail_at):
    attempts = []

    def cleanup():
        attempts.append(1)
        if len(attempts) == fail_at:
            raise hybrid.HybridRetrievalError("MPS cache cleanup failed: device_unavailable")
        return {"before": {"active_bytes": 10, "driver_bytes": 20}, "after": {"active_bytes": 10, "driver_bytes": 15}}

    runtime.cleanup_device = cleanup
    with pytest.raises(hybrid.HybridRetrievalError, match="cache cleanup failed"):
        _run(corpus, tmp_path)
    assert len(attempts) == fail_at
    assert (tmp_path / "index").exists() is (fail_at == 5)
    assert not (tmp_path / "result").exists()
    assert not list(tmp_path.glob(".result-*"))


@pytest.mark.parametrize("failure", (None, "device_unavailable"))
def test_mps_cleanup_uses_canonical_release_and_exposes_known_failure(monkeypatch, failure):
    order = []

    def active():
        order.append("active")
        return 100

    def driver():
        order.append("driver")
        return 1000 if "release" not in order else 200

    def release(device):
        assert device == "mps"
        order.append("release")
        return failure

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(mps=SimpleNamespace(
        current_allocated_memory=active, driver_allocated_memory=driver,
    )))
    monkeypatch.setattr(hybrid, "release_device_cache", release)
    if failure:
        with pytest.raises(hybrid.HybridRetrievalError, match=failure):
            hybrid._mps_cleanup()
        assert order == ["active", "driver", "release"]
    else:
        assert hybrid._mps_cleanup() == {
            "before": {"active_bytes": 100, "driver_bytes": 1000},
            "after": {"active_bytes": 100, "driver_bytes": 200},
        }
        assert order == ["active", "driver", "release", "active", "driver"]


def test_cpu_cleanup_policy_never_calls_mps_or_imports_torch(monkeypatch):
    def forbidden(*args):
        pytest.fail("CPU cleanup must not enter a torch/device path")

    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setattr(hybrid, "release_device_cache", forbidden)
    usage = hybrid._ResourceUsage(None)
    assert usage.sample_and_release("index_batch") is None
    assert usage.sample_and_release("query") is None
    assert usage.snapshot()["cleanup_count"] == 0


def _mock_encoding_cache(monkeypatch, runtime, corpus, *, cached_texts=None, invalid=None):
    with sqlite3.connect(corpus) as connection:
        texts = [row[0] for row in connection.execute("SELECT content FROM units")]
    if cached_texts is not None:
        texts = [text for text in texts if text in cached_texts]
    values = runtime.encoder.encode(texts)
    representations = {
        sha256(text.encode()).hexdigest(): (value.dense_vector, dict(value.learned_sparse_weights))
        for text, value in zip(texts, values, strict=True)
    }
    if invalid is not None:
        representations[next(iter(representations))] = invalid
    runtime.encoder.calls.clear()
    opened = []
    cache = SimpleNamespace(
        lookup=representations.get, manifest_sha256="f" * 64,
        safe_manifest={"schema_version": "synthetic-cache", "encoder_fingerprint": runtime.encoder.fingerprint()},
    )

    @contextmanager
    def open_cache(path, *, expected_encoder_fingerprint):
        assert expected_encoder_fingerprint == runtime.encoder.fingerprint()
        opened.append(path)
        yield cache

    monkeypatch.setattr(hybrid, "open_encoding_cache", open_cache)
    return opened, cache


def test_exact_encoding_cache_reuses_all_units_without_corpus_forward(corpus, tmp_path, runtime, monkeypatch):
    opened, cache = _mock_encoding_cache(monkeypatch, runtime, corpus)
    report = _run(corpus, tmp_path, encoding_cache=tmp_path / "cache")
    assert opened == [tmp_path / "cache"]
    assert all(text in {"alpha", "beta"} for batch in runtime.encoder.calls for text in batch)
    expected = {
        "reused_units": 8, "newly_encoded_units": 0, "total_units": 8,
        "cache_provenance": {"manifest_sha256": cache.manifest_sha256, "manifest": cache.safe_manifest},
    }
    assert report["index_encoding"] == expected
    manifest = json.loads((tmp_path / "index/manifest.json").read_text())
    assert manifest["encoding"] == expected
    assert "encoding_reuse_rule" in manifest["recipe"]
    assert "cache" not in {key for key in manifest["recipe"] if key != "encoding_reuse_rule"}

    def forbidden(*args, **kwargs):
        pytest.fail("a complete index must not depend on the old encoding cache")

    monkeypatch.setattr(hybrid, "open_encoding_cache", forbidden)
    second = hybrid.run_hybrid_retrieval_evaluation(
        corpus, tmp_path / "index", tmp_path / "second", batch_size=2, allow_live=True,
    )
    assert second["index_reused"] is True
    assert second["index_encoding"] == expected
    assert second["stages"] == report["stages"]


def test_encoding_cache_only_encodes_misses_and_preserves_all_stage_metrics(corpus, tmp_path, runtime, monkeypatch):
    baseline = _run(corpus, tmp_path / "baseline")
    _mock_encoding_cache(monkeypatch, runtime, corpus, cached_texts={"alpha target"})
    report = _run(corpus, tmp_path, encoding_cache=tmp_path / "cache")
    assert report["index_encoding"]["reused_units"] == 2
    assert report["index_encoding"]["newly_encoded_units"] == 6
    assert all(text != "alpha target" for batch in runtime.encoder.calls for text in batch)
    assert report["stages"] == baseline["stages"]
    assert report["execution"]["index_coverage"] == {"dense": 8, "learned_sparse": 8, "bm25": 8}


@pytest.mark.parametrize("cached_texts, expected_index_cleanups", ((None, 0), ({"alpha target"}, 3)))
def test_encoding_cache_hit_only_batches_skip_device_cleanup(
    corpus, tmp_path, runtime, monkeypatch, cached_texts, expected_index_cleanups,
):
    _mock_encoding_cache(monkeypatch, runtime, corpus, cached_texts=cached_texts)
    cleanup_phases = []

    def cleanup():
        cleanup_phases.append("query" if runtime.reranker.pairs else "index_batch")
        return {"before": {"active_bytes": 10, "driver_bytes": 20}, "after": {"active_bytes": 10, "driver_bytes": 15}}

    runtime.cleanup_device = cleanup
    events = []
    report = _run(corpus, tmp_path, encoding_cache=tmp_path / "cache", progress=events.append)
    assert cleanup_phases == ["index_batch"] * expected_index_cleanups + ["query"] * 3
    index_events = [event for event in events if event["stage"] == "index_build"]
    encoded_previous = 0
    for event in index_events:
        fresh = event["newly_encoded_units"] > encoded_previous
        assert (event["device_memory"] is not None) is fresh
        encoded_previous = event["newly_encoded_units"]
    summary = report["execution"]["device_memory"]
    assert summary["cleanup_count"] == expected_index_cleanups + 3
    assert summary["cleanup_count_by_phase"].get("index_batch", 0) == expected_index_cleanups
    assert summary["cleanup_count_by_phase"]["query"] == 3


def test_encoding_cache_errors_propagate_before_build(corpus, tmp_path, runtime, monkeypatch):
    reason = "synthetic cache validation failure"

    @contextmanager
    def invalid_cache(*args, **kwargs):
        raise ValueError(reason)
        yield  # pragma: no cover

    monkeypatch.setattr(hybrid, "open_encoding_cache", invalid_cache)
    with pytest.raises(hybrid.HybridRetrievalError, match=reason):
        _run(corpus, tmp_path, encoding_cache=tmp_path / "cache")
    assert not runtime.encoder.calls
    assert not (tmp_path / "index").exists()
    assert not (tmp_path / "result").exists()


@pytest.mark.parametrize("invalid", (
    ((0.0,) * 1023, {1: 1.0}),
    ((float("nan"),) + (0.0,) * 1023, {1: 1.0}),
    ((0.0,) * 1024, {1: -1.0}),
    ((0.0,) * 1024, {}),
))
def test_encoding_cache_invalid_representation_cannot_enter_index(corpus, tmp_path, runtime, monkeypatch, invalid):
    _mock_encoding_cache(monkeypatch, runtime, corpus, invalid=invalid)
    with pytest.raises(hybrid.HybridRetrievalError, match="1024|positive|representation"):
        _run(corpus, tmp_path, encoding_cache=tmp_path / "cache")
    assert not (tmp_path / "index").exists()
    assert not (tmp_path / "result").exists()
