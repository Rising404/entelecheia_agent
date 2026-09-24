"""Provenance-bound numeric reuse using synthetic SQLite only; no model calls."""

from hashlib import sha256
import json
import math
import sqlite3
import struct

import pytest

from evals.docbench_hybrid_retrieval_optimize import encoding_cache as cache


FP = "bge_m3:model=synthetic;dense=1024;fp16=false;local_only=true"
CONTENT = sha256(b"synthetic corpus unit").hexdigest()
OTHER = sha256(b"unmatched source content").hexdigest()


def _write_json(path, document):
    path.write_text(json.dumps(document))
    return sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def dataset(tmp_path):
    directory = tmp_path / "dataset"
    directory.mkdir()
    path = directory / "dataset.sqlite"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE units(content_sha256 TEXT,content TEXT)")
        connection.execute("INSERT INTO units VALUES (?,?)", (CONTENT, "synthetic corpus unit"))
    _write_json(directory / "manifest.json", {"schema_version": "docbench-retrieval-dataset-v1", "dataset_sha256": sha256(path.read_bytes()).hexdigest()})
    return path


def _source(tmp_path, bundle, name, *, partial=False, value=1.0):
    directory = tmp_path / name
    directory.mkdir()
    database = directory / "retrieval.sqlite"
    recipe = {"encoder_fingerprint": FP, "code_sha256": "c" * 64}
    recipe_hash = sha256(cache._json(recipe).encode()).hexdigest()
    generation = {"id": "docbench-" + recipe_hash if partial else "generation", "fingerprint": recipe_hash if partial else "generation-fingerprint", "role": "staging" if partial else "active", "state": "building" if partial else "ready"}
    with sqlite3.connect(database) as connection:
        connection.executescript("""
            CREATE TABLE retrieval_data_versions(id TEXT,fingerprint TEXT,role TEXT,state TEXT);
            CREATE TABLE retrieval_units(unit_id INTEGER,data_version_id TEXT,indexed_content_hash TEXT,retrieval_status TEXT,index_state TEXT);
            CREATE TABLE retrieval_unit_method_indexes(unit_id INTEGER,method TEXT,state TEXT,reason_code TEXT);
            CREATE TABLE dense_vectors(unit_id INTEGER,embedding BLOB);
            CREATE TABLE learned_sparse_postings(unit_id INTEGER,token_id INTEGER,weight REAL);
            CREATE TABLE bm25_unit_terms(unit_id INTEGER,shadow_terms TEXT);
            CREATE VIRTUAL TABLE bm25_fts USING fts5(shadow_terms,content='');
        """)
        connection.execute("INSERT INTO retrieval_data_versions VALUES (?,?,?,?)", tuple(generation.values()))
        for unit_id, content in ((1, CONTENT), (2, OTHER)):
            connection.execute("INSERT INTO retrieval_units VALUES (?,?,?,'active','ready')", (unit_id, generation["id"], content))
            connection.executemany("INSERT INTO retrieval_unit_method_indexes VALUES (?,?,'ready',NULL)", [(unit_id, method) for method in ("dense", "learned_sparse", "bm25")])
            connection.execute("INSERT INTO dense_vectors VALUES (?,?)", (unit_id, struct.pack("<1024f", value, *([0.] * 1023))))
            connection.execute("INSERT INTO learned_sparse_postings VALUES (?,7,?)", (unit_id, value))
            connection.execute("INSERT INTO bm25_unit_terms VALUES (?,'t7')", (unit_id,))
            connection.execute("INSERT INTO bm25_fts(rowid,shadow_terms) VALUES (?,'t7')", (unit_id,))
    descriptor = {"source_db_path": str(database), "source_db_sha256": sha256(database.read_bytes()).hexdigest(), "active_generation": generation}
    if partial:
        proof = directory / "proof.json"
        proof_hash = _write_json(proof, {"schema_version": "docbench-partial-index-snapshot-v1", "generation": generation, "database_sha256": descriptor["source_db_sha256"], "recipe": recipe, "encoder_fingerprint": FP, "code_sha256": recipe["code_sha256"]})
        descriptor.update(kind="current_run_partial", source_id=name, proof_path=str(proof), proof_sha256=proof_hash)
    else:
        report, run = directory / "generation_report.json", directory / "run_manifest.json"
        snapshot = {"encoder_fingerprint": FP, "active_generation": {**generation, "expected_fingerprint": generation["fingerprint"], "matches_runtime": True}}
        report_hash = _write_json(report, {"run_id": name, "retrieval": {"preflight": {"encoder_fingerprint": FP}, "case_snapshots": [{"case_id": "case", "snapshot": snapshot}]}, "cases": [{"question": "DO_NOT_COPY_QUESTION", "reference_answer": "DO_NOT_COPY_ANSWER", "submitted_prompt": "DO_NOT_COPY_PROMPT"}]})
        run_hash = _write_json(run, {"run_id": name})
        bundle["runs"][name] = {"generation_report_path": str(report), "generation_report_sha256": report_hash, "run_manifest_path": str(run), "run_manifest_sha256": run_hash, "encoder_fingerprint": FP}
        descriptor.update(source_run_id=name, case_id="case", encoder_fingerprint=FP)
    bundle["sources"].append(descriptor)
    return descriptor


def _bundle(dataset):
    return {"dataset_sha256": sha256(dataset.read_bytes()).hexdigest(), "runs": {}, "sources": []}


def _build(dataset, bundle, tmp_path):
    return cache.build_encoding_cache(dataset, bundle, tmp_path / "cache", expected_encoder_fingerprint=FP)


def test_exact_content_reuse_does_not_copy_source_payloads_or_paths(dataset, tmp_path):
    bundle = _bundle(dataset)
    descriptor = _source(tmp_path, bundle, "history")
    source = cache._path(descriptor["source_db_path"])
    before = source.read_bytes()
    manifest = _build(dataset, bundle, tmp_path)
    assert manifest["counts"]["cached_unique_contents"] == 1
    with cache.open_encoding_cache(tmp_path / "cache", expected_encoder_fingerprint=FP) as opened:
        dense, sparse = opened.lookup(CONTENT)
        assert len(dense) == 1024 and dense[0] == 1.0 and sparse == {7: 1.0}
        assert opened.lookup(OTHER) is None
        assert opened.safe_manifest == manifest
        assert opened.manifest_sha256 == sha256((tmp_path / "cache/manifest.json").read_bytes()).hexdigest()
        serialized = json.dumps(opened.safe_manifest)
        assert str(tmp_path) not in serialized and "DO_NOT_COPY" not in serialized
    assert source.read_bytes() == before
    assert "cases" not in cache._report_metadata(cache._path(bundle["runs"]["history"]["generation_report_path"]))


@pytest.mark.parametrize("partial_first", [False, True])
def test_partial_is_explicitly_preferred_over_conflicting_historical_vectors(dataset, tmp_path, partial_first):
    bundle = _bundle(dataset)
    _source(tmp_path, bundle, "history", value=1.0)
    _source(tmp_path, bundle, "partial", partial=True, value=2.0)
    if partial_first:
        bundle["sources"].reverse()
    manifest = _build(dataset, bundle, tmp_path)
    with cache.open_encoding_cache(tmp_path / "cache", FP) as opened:
        assert opened.lookup(CONTENT)[0][0] == 2.0
    counter = "lower_priority_differences_ignored" if partial_first else "higher_priority_replacements"
    assert manifest["counts"][counter] == 1
    assert "current_run_partial_over_historical" in manifest["conflict_policy"]


def test_equal_priority_conflicts_become_misses_and_cannot_reappear(dataset, tmp_path):
    bundle = _bundle(dataset)
    for name, value in (("one", 1.0), ("two", 2.0), ("three", 1.0)):
        _source(tmp_path, bundle, name, value=value)
    manifest = _build(dataset, bundle, tmp_path)
    assert manifest["counts"]["conflicted_unique_contents"] == 1
    with cache.open_encoding_cache(tmp_path / "cache", FP) as opened:
        assert opened.lookup(CONTENT) is None


@pytest.mark.parametrize("table", ["dense_vectors", "learned_sparse_postings", "bm25_unit_terms", "retrieval_unit_method_indexes"])
def test_incomplete_published_methods_are_not_reused(dataset, tmp_path, table):
    bundle = _bundle(dataset)
    descriptor = _source(tmp_path, bundle, "history")
    source = cache._path(descriptor["source_db_path"])
    with sqlite3.connect(source) as connection:
        connection.execute(f"DELETE FROM {table} WHERE unit_id=1")
    descriptor["source_db_sha256"] = sha256(source.read_bytes()).hexdigest()
    manifest = _build(dataset, bundle, tmp_path)
    assert manifest["counts"]["incomplete_units"] == 1
    with cache.open_encoding_cache(tmp_path / "cache", FP) as opened:
        assert opened.lookup(CONTENT) is None


@pytest.mark.parametrize("defect", ["model", "generation", "report_hash", "db_hash", "wal", "journal", "vector"])
def test_unproven_or_invalid_source_fails_without_publishing(dataset, tmp_path, defect):
    bundle = _bundle(dataset)
    descriptor = _source(tmp_path, bundle, "history")
    source = cache._path(descriptor["source_db_path"])
    if defect == "model":
        descriptor["encoder_fingerprint"] = "wrong"
    elif defect == "generation":
        descriptor["active_generation"]["id"] = "wrong"
    elif defect == "report_hash":
        bundle["runs"]["history"]["generation_report_sha256"] = "a" * 64
    elif defect == "db_hash":
        descriptor["source_db_sha256"] = "a" * 64
    elif defect in {"wal", "journal"}:
        source.with_name(source.name + "-" + defect).write_bytes(b"pending")
    else:
        with sqlite3.connect(source) as connection:
            connection.execute("UPDATE dense_vectors SET embedding=? WHERE unit_id=1", (struct.pack("<1024f", math.inf, *([0.] * 1023)),))
        descriptor["source_db_sha256"] = sha256(source.read_bytes()).hexdigest()
    with pytest.raises(cache.EncodingCacheError):
        _build(dataset, bundle, tmp_path)
    assert not (tmp_path / "cache").exists()
    assert not list(tmp_path.glob(".cache-*"))


@pytest.mark.parametrize("defect", ["model", "database_hash", "schema", "count", "shape", "nonfinite", "sparse"])
def test_cache_reader_checks_model_schema_count_and_numerical_payloads(dataset, tmp_path, defect):
    bundle = _bundle(dataset)
    _source(tmp_path, bundle, "history")
    _build(dataset, bundle, tmp_path)
    root = tmp_path / "cache"
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    expected = FP
    if defect == "model":
        expected = "wrong"
    elif defect == "database_hash":
        manifest["database"]["sha256"] = "a" * 64
    elif defect == "schema":
        manifest["schema_version"] = "unknown"
    elif defect == "count":
        manifest["counts"]["cached_unique_contents"] = 3
    else:
        with sqlite3.connect(root / "encodings.sqlite") as connection:
            if defect == "sparse":
                connection.execute("UPDATE encodings SET sparse_json=?", ('{"7": -1}',))
            else:
                dense = b"too short" if defect == "shape" else struct.pack("<1024f", math.nan, *([0.] * 1023))
                connection.execute("UPDATE encodings SET dense=?", (dense,))
        manifest["database"] = cache._file_record(root / "encodings.sqlite")
    _write_json(manifest_path, manifest)
    with pytest.raises(cache.EncodingCacheError):
        with cache.open_encoding_cache(root, expected):
            pytest.fail("invalid cache must not be exposed")


def test_existing_cache_output_is_preserved(dataset, tmp_path):
    bundle = _bundle(dataset)
    _source(tmp_path, bundle, "history")
    _build(dataset, bundle, tmp_path)
    before = (tmp_path / "cache/encodings.sqlite").read_bytes()
    with pytest.raises(cache.EncodingCacheError, match="overwrite"):
        _build(dataset, bundle, tmp_path)
    assert (tmp_path / "cache/encodings.sqlite").read_bytes() == before


class _Tokenizer:
    def __init__(self, defect=None):
        self.defect = defect

    def __call__(self, texts, *, truncation, max_length=None):
        ids = [[0, *[sum(map(ord, token)) for token in text.split()], 2] for text in texts]
        if self.defect == "ids":
            ids[1][1] += 1
        if self.defect == "special":
            ids[1][0] += 1
        if self.defect == "truncated":
            ids = [[7] * (8192 if truncation else 8193) for _ in texts]
        masks = [[1] * len(row) for row in ids]
        result = {"input_ids": ids, "attention_mask": masks}
        if self.defect == "mask":
            masks[1][0] = 0
        if self.defect == "extra_field":
            result["token_type_ids"] = [[0] * len(ids[0]), [1] * len(ids[1])]
        return result

    def pad(self, rows, *, padding):
        result = {key: [row[key][:] for row in rows] for key in rows[0]}
        if self.defect == "pad":
            result["attention_mask"][1][0] = 0
        return result


def _strip_dataset(dataset, text=" synthetic corpus unit\n"):
    content_hash = sha256(text.encode()).hexdigest()
    with sqlite3.connect(dataset) as connection:
        connection.execute("UPDATE units SET content_sha256=?,content=?", (content_hash, text))
    _write_json(dataset.parent / "manifest.json", {"schema_version": "docbench-retrieval-dataset-v1", "dataset_sha256": sha256(dataset.read_bytes()).hexdigest()})
    return content_hash


def _strip_build(dataset, bundle, tmp_path, monkeypatch, defect=None):
    monkeypatch.setattr(cache, "_strip_tokenizer", lambda expected: (_Tokenizer(defect), {"model_revision": "synthetic"}))
    return cache.build_encoding_cache(dataset, bundle, tmp_path / "cache", expected_encoder_fingerprint=FP, allow_strip_token_equivalence=True)


def test_strip_equivalence_requires_opt_in_and_preserves_raw_cache_keys(dataset, tmp_path, monkeypatch):
    content_hash = _strip_dataset(dataset)
    bundle = _bundle(dataset)
    _source(tmp_path, bundle, "history")
    manifest = cache.build_encoding_cache(dataset, bundle, tmp_path / "exact", expected_encoder_fingerprint=FP)
    assert manifest["counts"]["cached_unique_contents"] == 0
    manifest = _strip_build(dataset, bundle, tmp_path, monkeypatch)
    with cache.open_encoding_cache(tmp_path / "cache", FP) as opened:
        assert opened.lookup(content_hash)[0][0] == 1.0
        assert opened.lookup(CONTENT) is None
    assert manifest["counts"]["cached_strip_equivalent_contents"] == 1
    assert manifest["strip_equivalence"]["verified_target_contents"] == 1
    assert str(tmp_path) not in json.dumps(manifest)
    assert "synthetic corpus unit" not in json.dumps(manifest)


@pytest.mark.parametrize("defect", ["ids", "special", "mask", "extra_field", "pad", "truncated"])
def test_strip_reuse_rejects_any_different_or_truncated_model_input(dataset, tmp_path, monkeypatch, defect):
    content_hash = _strip_dataset(dataset)
    bundle = _bundle(dataset)
    _source(tmp_path, bundle, "history")
    manifest = _strip_build(dataset, bundle, tmp_path, monkeypatch, defect)
    assert manifest["strip_equivalence"]["rejected_target_contents"] == 1
    with cache.open_encoding_cache(tmp_path / "cache", FP) as opened:
        assert opened.lookup(content_hash) is None


def test_interior_whitespace_is_not_an_allowed_content_transform(dataset, tmp_path, monkeypatch):
    content_hash = _strip_dataset(dataset, "synthetic  corpus unit")
    bundle = _bundle(dataset)
    _source(tmp_path, bundle, "history")
    manifest = _strip_build(dataset, bundle, tmp_path, monkeypatch)
    assert manifest["strip_equivalence"]["verified_target_contents"] == 0
    with cache.open_encoding_cache(tmp_path / "cache", FP) as opened:
        assert opened.lookup(content_hash) is None


@pytest.mark.parametrize("partial_first", [False, True])
def test_exact_historical_match_outranks_strip_partial_match(dataset, tmp_path, monkeypatch, partial_first):
    content_hash = _strip_dataset(dataset)
    bundle = _bundle(dataset)
    historical = _source(tmp_path, bundle, "history", value=1.0)
    source = cache._path(historical["source_db_path"])
    with sqlite3.connect(source) as connection:
        connection.execute("UPDATE retrieval_units SET indexed_content_hash=? WHERE unit_id=1", (content_hash,))
    historical["source_db_sha256"] = sha256(source.read_bytes()).hexdigest()
    _source(tmp_path, bundle, "partial", partial=True, value=2.0)
    if partial_first:
        bundle["sources"].reverse()
    manifest = _strip_build(dataset, bundle, tmp_path, monkeypatch)
    assert manifest["counts"]["cached_exact_contents"] == 1
    assert manifest["counts"]["cached_strip_equivalent_contents"] == 0
    with cache.open_encoding_cache(tmp_path / "cache", FP) as opened:
        assert opened.lookup(content_hash)[0][0] == 1.0


def test_strip_reuse_preserves_partial_priority_and_same_priority_conflict_rules(dataset, tmp_path, monkeypatch):
    content_hash = _strip_dataset(dataset)
    bundle = _bundle(dataset)
    _source(tmp_path, bundle, "history", value=1.0)
    _source(tmp_path, bundle, "conflicting", value=3.0)
    _source(tmp_path, bundle, "partial", partial=True, value=2.0)
    manifest = _strip_build(dataset, bundle, tmp_path, monkeypatch)
    assert manifest["counts"]["equal_priority_conflicts"] == 1
    with cache.open_encoding_cache(tmp_path / "cache", FP) as opened:
        assert opened.lookup(content_hash)[0][0] == 2.0


def test_strip_equivalence_does_not_relax_source_provenance(dataset, tmp_path, monkeypatch):
    _strip_dataset(dataset)
    bundle = _bundle(dataset)
    source = _source(tmp_path, bundle, "history")
    source["encoder_fingerprint"] = "wrong"
    with pytest.raises(cache.EncodingCacheError, match="ready encoder generation"):
        _strip_build(dataset, bundle, tmp_path, monkeypatch)
    assert not (tmp_path / "cache").exists()
