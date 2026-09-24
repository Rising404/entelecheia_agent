"""Reuse proven, exact-content BGE encodings without models or source mutations.

Only retrieval identities, published method state and numerical representations
are read from source databases. Target scopes and BM25 tokens must be rebuilt by
the caller. Historical question/answer/prompt fields are never decoded.
"""

from __future__ import annotations

import codecs
from contextlib import closing, contextmanager
from hashlib import sha256
import json
import math
from pathlib import Path
import re
import sqlite3
import struct
import tempfile
from typing import Iterator

from .index_archive import _document, _hash, _metadata, _path, _publish, _reader


_SCHEMA = "docbench-encoding-cache-v1"
_DB = "encodings.sqlite"
_METHODS = {"dense", "learned_sparse", "bm25"}
_CONFLICT_POLICY = "current_run_partial_over_historical; equal_priority_conflicts_are_misses"


class EncodingCacheError(ValueError):
    """A cache cannot attest its model, source provenance or representations."""


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def _file_record(path: Path) -> dict:
    digest, count = sha256(), 0
    with _reader(path) as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
            count += len(block)
    return {"sha256": digest.hexdigest(), "bytes": count}


def _identifier(value: object) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9:._-]{0,240}", value) is None:
        raise EncodingCacheError("source identity must be a portable logical identifier")
    return value


def _fingerprint(value: object) -> str:
    if (
        not isinstance(value, str) or not value or len(value) > 4096
        or any(ord(char) < 32 for char in value)
        or re.search(r"(?:^|[=:])(?:/|~[/\\]|file:|[A-Za-z]:[/\\])", value)
    ):
        raise EncodingCacheError("encoder fingerprint is missing or contains a local path")
    return value


def _no_pending(path: Path) -> None:
    for suffix in ("-wal", "-journal", "-shm"):
        sidecar = Path(str(path) + suffix)
        if sidecar.is_symlink():
            raise EncodingCacheError("source/cache sidecars must not be symlinks")
        if sidecar.exists() and (not sidecar.is_file() or (suffix != "-shm" and sidecar.stat().st_size)):
            raise EncodingCacheError("source/cache has a pending WAL/journal transaction")


@contextmanager
def _readonly(path: Path, *, vectors: bool = False) -> Iterator[sqlite3.Connection]:
    path = _path(path)
    _no_pending(path)
    if not path.is_file():
        raise EncodingCacheError("source/cache database is missing")
    connection = sqlite3.connect(f"{path.as_uri()}?mode=ro&immutable=1", uri=True)
    try:
        if vectors:
            import sqlite_vec
            connection.enable_load_extension(True)
            try:
                sqlite_vec.load(connection)
            finally:
                connection.enable_load_extension(False)
        connection.execute("PRAGMA query_only=ON")
        connection.row_factory = sqlite3.Row
        connection.execute("BEGIN")
        yield connection
    finally:
        connection.close()
    _no_pending(path)


def _report_metadata(path: Path) -> dict:
    """Decode only the report's metadata prefix, stopping before case payloads."""
    permitted = {
        "schema_version", "benchmark_id", "lane", "run_id", "status", "gate_passed",
        "generation_gate", "baseline_eligible", "config_sha256", "selection_sha256", "retrieval",
    }
    decoder = json.JSONDecoder()
    with _reader(path) as stream:
        utf8 = codecs.getincrementaldecoder("utf-8")()
        buffer, position, ended = "", 0, False

        def more():
            nonlocal buffer, ended
            block = stream.read(64 * 1024)
            ended = not block
            buffer += utf8.decode(block, final=ended)
            if len(buffer) > 64 * 1024 * 1024:
                raise EncodingCacheError("retrieval report metadata exceeds its bounded prefix")

        def spaces():
            nonlocal position
            while True:
                while position < len(buffer) and buffer[position].isspace():
                    position += 1
                if position < len(buffer) or ended:
                    return
                more()

        def expect(character):
            nonlocal position
            spaces()
            if position >= len(buffer) or buffer[position] != character:
                raise EncodingCacheError("invalid retrieval report metadata prefix")
            position += 1

        def value():
            nonlocal position
            spaces()
            while True:
                try:
                    result, finish = decoder.raw_decode(buffer, position)
                    position = finish
                    return result
                except json.JSONDecodeError as error:
                    if ended:
                        raise EncodingCacheError("invalid retrieval report metadata") from error
                    more()

        expect("{")
        values = {}
        while True:
            key = value()
            if key not in permitted or key in values:
                raise EncodingCacheError("report retrieval metadata must precede case payloads")
            expect(":")
            values[key] = value()
            if key == "retrieval":
                return values
            expect(",")


def _source_proof(descriptor: dict, bundle: dict, expected: str, reports: dict) -> tuple[dict, int]:
    kind = descriptor.get("kind", "historical_docbench")
    expected_generation = descriptor["active_generation"]
    generation = {key: expected_generation[key] for key in ("id", "fingerprint", "role", "state")}
    for value in generation.values():
        _identifier(value)
    if kind == "current_run_partial":
        source_id = _identifier(descriptor["source_id"])
        proof_path = _path(Path(descriptor["proof_path"]))
        proof_record = _file_record(proof_path)
        if proof_record["sha256"] != descriptor["proof_sha256"]:
            raise EncodingCacheError("partial source proof hash mismatch")
        proof = _document(_metadata(proof_path))
        recipe = proof["recipe"]
        recipe_hash = sha256(_json(recipe).encode()).hexdigest()
        if (
            proof.get("schema_version") != "docbench-partial-index-snapshot-v1"
            or proof["database_sha256"] != descriptor["source_db_sha256"]
            or proof["generation"] != generation
            or proof["encoder_fingerprint"] != expected or recipe["encoder_fingerprint"] != expected
            or recipe["code_sha256"] != proof["code_sha256"] or not _hash(proof["code_sha256"])
            or generation["fingerprint"] != recipe_hash or generation["id"] != "docbench-" + recipe_hash
            or generation["role"] != "staging" or generation["state"] != "building"
        ):
            raise EncodingCacheError("partial source is not bound to its frozen encoding recipe")
        evidence = {"proof": proof_record, "recipe_sha256": recipe_hash, "code_sha256": proof["code_sha256"]}
        priority = 2
    elif kind == "historical_docbench":
        run_id, case_id = _identifier(descriptor["source_run_id"]), _identifier(descriptor["case_id"])
        source_id = _identifier(run_id + ":" + case_id)
        run = bundle["runs"][run_id]
        if run_id not in reports:
            report_path = _path(Path(run["generation_report_path"]))
            report_record = _file_record(report_path)
            run_record = _file_record(_path(Path(run["run_manifest_path"])))
            if report_record["sha256"] != run["generation_report_sha256"] or run_record["sha256"] != run["run_manifest_sha256"]:
                raise EncodingCacheError("historical proof hash mismatch")
            report = _report_metadata(report_path)
            if report["run_id"] != run_id or report["retrieval"]["preflight"]["encoder_fingerprint"] != expected:
                raise EncodingCacheError("historical report encoder identity differs")
            snapshots = {}
            for record in report["retrieval"]["case_snapshots"]:
                if record["case_id"] in snapshots:
                    raise EncodingCacheError("duplicate historical case identity")
                snapshots[record["case_id"]] = record["snapshot"]
            reports[run_id] = (snapshots, {"generation_report": report_record, "run_manifest": run_record})
        snapshots, evidence = reports[run_id]
        snapshot = snapshots[case_id]
        actual = snapshot["active_generation"]
        if (
            run["encoder_fingerprint"] != expected or descriptor["encoder_fingerprint"] != expected
            or snapshot["encoder_fingerprint"] != expected
            or {key: actual[key] for key in generation} != generation
            or actual.get("matches_runtime") is not True or actual.get("expected_fingerprint") != generation["fingerprint"]
            or generation["role"] != "active" or generation["state"] != "ready"
        ):
            raise EncodingCacheError("historical source is not bound to a ready encoder generation")
        priority = 1
    else:
        raise EncodingCacheError("unsupported encoding source kind")
    return {"source_id": source_id, "kind": kind, "generation": generation, "proof": evidence}, priority


def _representation(dense: bytes, sparse: dict) -> tuple[tuple[float, ...], dict[int, float]]:
    if not isinstance(dense, bytes) or len(dense) != 4096:
        raise EncodingCacheError("dense encoding must contain exactly 1024 float32 values")
    values = struct.unpack("<1024f", dense)
    if not all(math.isfinite(value) for value in values) or not isinstance(sparse, dict) or not sparse:
        raise EncodingCacheError("encoding contains nonfinite dense values or empty sparse weights")
    weights = {}
    for token, weight in sparse.items():
        if (
            not isinstance(token, str) or re.fullmatch(r"0|[1-9][0-9]*", token) is None
            or type(weight) not in (int, float) or not math.isfinite(weight) or weight <= 0
        ):
            raise EncodingCacheError("sparse weights must have integer token IDs and positive finite values")
        weights[int(token)] = float(weight)
    return values, weights


def _insert_encoding(target, content_hash, dense, sparse_json, source_id, priority, counts):
    previous = target.execute("SELECT dense,sparse_json,priority FROM encodings WHERE content_sha256=?", (content_hash,)).fetchone()
    conflict = target.execute("SELECT priority FROM conflicts WHERE content_sha256=?", (content_hash,)).fetchone()
    old_priority = previous[2] if previous else conflict[0] if conflict else 0
    if priority < old_priority:
        if previous and (previous[0] != dense or previous[1] != sparse_json):
            counts["lower_priority_differences_ignored"] += 1
        return
    if priority == old_priority:
        if conflict:
            return
        if previous[0] == dense and previous[1] == sparse_json:
            counts["identical_duplicates"] += 1
            return
        target.execute("DELETE FROM encodings WHERE content_sha256=?", (content_hash,))
        target.execute("INSERT INTO conflicts VALUES (?,?)", (content_hash, priority))
        counts["equal_priority_conflicts"] += 1
        return
    if old_priority:
        counts["higher_priority_replacements"] += 1
    target.execute("DELETE FROM conflicts WHERE content_sha256=?", (content_hash,))
    target.execute("INSERT OR REPLACE INTO encodings VALUES (?,?,?,?,?)", (content_hash, dense, sparse_json, source_id, priority))


def _strip_tokenizer(expected: str):
    """Load only the fixed production tokenizer, never the embedding model."""
    from personagraph.retrieval.indexing.encoder import BgeM3Encoder
    from personagraph.retrieval.indexing.model_assets import BGE_M3_MODEL_REVISION

    encoder = BgeM3Encoder(device="mps", use_fp16=False)
    if encoder.fingerprint() != expected:
        raise EncodingCacheError("strip equivalence requires the identical fixed BGE encoder")
    tokenizer = encoder._load_tokenizer()
    return tokenizer, {
        "model_revision": BGE_M3_MODEL_REVISION,
        "tokenizer_backend_sha256": sha256(tokenizer.backend_tokenizer.to_str().encode()).hexdigest(),
        "truncation": True, "max_length": 8192, "special_tokens": "tokenizer default",
        "fields": "all tokenizer fields and same-batch padded fields must match",
        "truncated_inputs": "rejected",
    }


def _strip_input_proof(tokenizer, content: str) -> str | None:
    """Attest the exact FlagEmbedding passage input, including masks/specials."""
    pair = [content, content.strip()]
    inputs = dict(tokenizer(pair, truncation=True, max_length=8192))
    full = dict(tokenizer(pair, truncation=False))
    if not {"input_ids", "attention_mask"}.issubset(inputs) or inputs != full:
        return None
    if any(len(values) != 2 or values[0] != values[1] for values in inputs.values()):
        return None
    if any(len(ids) > 8192 for ids in inputs["input_ids"]):
        return None
    rows = [{key: values[index] for key, values in inputs.items()} for index in range(2)]
    padded = dict(tokenizer.pad(rows, padding=True))
    if any(len(values) != 2 or values[0] != values[1] for values in padded.values()):
        return None
    return sha256(_json({"inputs": rows[0], "padded": {key: value[0] for key, value in padded.items()}}).encode()).hexdigest()


def build_encoding_cache(dataset_path: Path, source_descriptors: Path | dict, output_dir: Path, *, expected_encoder_fingerprint: str, allow_strip_token_equivalence: bool = False) -> dict:
    """Export proven encodings; optional strip reuse requires identical model input."""
    try:
        expected = _fingerprint(expected_encoder_fingerprint)
        dataset, output = _path(dataset_path), _path(output_dir)
        if output.exists():
            raise EncodingCacheError("cache output exists; refusing overwrite")
        bundle = source_descriptors if isinstance(source_descriptors, dict) else _document(_metadata(_path(source_descriptors)))
        dataset_record = _file_record(dataset)
        dataset_manifest = _document(_metadata(dataset.parent / "manifest.json"))
        if dataset_manifest.get("schema_version") != "docbench-retrieval-dataset-v1" or dataset_record["sha256"] != dataset_manifest["dataset_sha256"] or dataset_record["sha256"] != bundle["dataset_sha256"]:
            raise EncodingCacheError("frozen dataset hash mismatch")
        strip_candidates, verified, tokenizer, tokenizer_proof = {}, {}, None, None
        with _readonly(dataset) as source:
            hashes = [row[0] for row in source.execute("SELECT content_sha256 FROM units")]
            if allow_strip_token_equivalence:
                tokenizer, tokenizer_proof = _strip_tokenizer(expected)
                for content_hash, content in source.execute("SELECT content_sha256,content FROM units"):
                    if sha256(content.encode()).hexdigest() != content_hash:
                        raise EncodingCacheError("dataset text differs from its content hash")
                    stripped_hash = sha256(content.strip().encode()).hexdigest()
                    if stripped_hash != content_hash:
                        strip_candidates.setdefault(stripped_hash, {})[content_hash] = content
        targets = set(hashes)
        if not targets or not all(_hash(value) for value in targets):
            raise EncodingCacheError("dataset content identities are missing or invalid")
        output.parent.mkdir(parents=True, exist_ok=True)
        source_records, reports, seen_sources = [], {}, set()
        counts = dict.fromkeys(("matched_units", "eligible_units", "incomplete_units", "identical_duplicates", "equal_priority_conflicts", "higher_priority_replacements", "lower_priority_differences_ignored"), 0)
        with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
            stage = Path(temporary)
            with closing(sqlite3.connect(stage / _DB)) as target, target:
                target.executescript("""
                    PRAGMA user_version=1;
                    CREATE TABLE encodings(content_sha256 TEXT PRIMARY KEY, dense BLOB NOT NULL,
                        sparse_json TEXT NOT NULL, source_id TEXT NOT NULL, priority INTEGER NOT NULL);
                    CREATE TABLE conflicts(content_sha256 TEXT PRIMARY KEY, priority INTEGER NOT NULL);
                """)
                for descriptor in bundle["sources"]:
                    provenance, priority = _source_proof(descriptor, bundle, expected, reports)
                    if provenance["source_id"] in seen_sources:
                        raise EncodingCacheError("duplicate source identity")
                    seen_sources.add(provenance["source_id"])
                    database = _path(Path(descriptor["source_db_path"]))
                    source_record = _file_record(database)
                    if source_record["sha256"] != descriptor["source_db_sha256"]:
                        raise EncodingCacheError("source database hash mismatch")
                    source_eligible = 0
                    with _readonly(database, vectors=True) as source:
                        generation = provenance["generation"]
                        row = source.execute("SELECT id,fingerprint,role,state FROM retrieval_data_versions WHERE id=?", (generation["id"],)).fetchone()
                        if row is None or dict(row) != generation:
                            raise EncodingCacheError("source database generation differs from its proof")
                        active = source.execute("SELECT id FROM retrieval_data_versions WHERE role='active'").fetchall()
                        if priority == 1 and [row[0] for row in active] != [generation["id"]]:
                            raise EncodingCacheError("historical source active generation is ambiguous")
                        candidates = source.execute("SELECT unit_id,indexed_content_hash FROM retrieval_units WHERE data_version_id=? AND retrieval_status='active' AND index_state='ready' ORDER BY unit_id", (generation["id"],))
                        for unit_id, content_hash in candidates:
                            if content_hash not in targets and content_hash not in strip_candidates:
                                continue
                            counts["matched_units"] += 1
                            states = source.execute("SELECT method,state,reason_code FROM retrieval_unit_method_indexes WHERE unit_id=?", (unit_id,)).fetchall()
                            dense = source.execute("SELECT embedding FROM dense_vectors WHERE unit_id=?", (unit_id,)).fetchone()
                            sparse = source.execute("SELECT token_id,weight FROM learned_sparse_postings WHERE unit_id=? ORDER BY token_id", (unit_id,)).fetchall()
                            terms = source.execute("SELECT shadow_terms FROM bm25_unit_terms WHERE unit_id=?", (unit_id,)).fetchone()
                            fts = source.execute("SELECT rowid FROM bm25_fts WHERE rowid=?", (unit_id,)).fetchone()
                            if (
                                len(states) != 3 or {row[0] for row in states} != _METHODS
                                or any(row[1] != "ready" or row[2] for row in states)
                                or dense is None or not sparse or terms is None or not terms[0].strip() or fts is None
                            ):
                                counts["incomplete_units"] += 1
                                continue
                            if any(re.fullmatch(r"t[0-9]+", term) is None for term in terms[0].split()):
                                raise EncodingCacheError("source BM25 representation contains invalid token terms")
                            weights = {str(token): weight for token, weight in sparse}
                            if len(weights) != len(sparse):
                                raise EncodingCacheError("source sparse representation contains duplicate tokens")
                            _representation(dense[0], weights)
                            if content_hash in targets:
                                exact_priority = priority + 2 if allow_strip_token_equivalence else priority
                                _insert_encoding(target, content_hash, dense[0], _json(weights), provenance["source_id"], exact_priority, counts)
                            for target_hash, content in strip_candidates.get(content_hash, {}).items():
                                if target_hash not in verified:
                                    verified[target_hash] = _strip_input_proof(tokenizer, content)
                                if verified[target_hash] is not None:
                                    _insert_encoding(target, target_hash, dense[0], _json(weights), provenance["source_id"], priority, counts)
                            counts["eligible_units"] += 1
                            source_eligible += 1
                    if _file_record(database) != source_record:
                        raise EncodingCacheError("source database changed during cache construction")
                    source_records.append({**provenance, "database": source_record, "eligible_units": source_eligible})
                counts["cached_unique_contents"] = target.execute("SELECT count(*) FROM encodings").fetchone()[0]
                counts["conflicted_unique_contents"] = target.execute("SELECT count(*) FROM conflicts").fetchone()[0]
                if allow_strip_token_equivalence:
                    counts["cached_strip_equivalent_contents"] = target.execute("SELECT count(*) FROM encodings WHERE priority<=2").fetchone()[0]
                    counts["cached_exact_contents"] = counts["cached_unique_contents"] - counts["cached_strip_equivalent_contents"]
            if _file_record(dataset) != dataset_record:
                raise EncodingCacheError("dataset changed during cache construction")
            manifest = {
                "schema_version": _SCHEMA, "encoder_fingerprint": expected,
                "dataset": dataset_record, "corpus_units": len(hashes), "unique_contents": len(targets),
                "matching": "exact content_sha256 only", "conflict_policy": _CONFLICT_POLICY,
                "dense": "1024 little-endian float32", "sparse": "all positive finite token weights",
                "counts": counts, "sources": source_records, "database": _file_record(stage / _DB),
            }
            if allow_strip_token_equivalence:
                pairs = sorted((target_hash, stripped_hash, verified[target_hash]) for stripped_hash, candidates in strip_candidates.items() for target_hash in candidates if verified.get(target_hash) is not None)
                manifest.update(
                    matching="exact content_sha256; explicit str.strip only when full model inputs are identical",
                    conflict_policy="exact_content_over_strip_equivalent; " + _CONFLICT_POLICY,
                    strip_equivalence={
                        "tokenizer": tokenizer_proof, "verified_target_contents": len(pairs),
                        "rejected_target_contents": sum(value is None for value in verified.values()),
                        "proof_pairs_sha256": sha256(_json(pairs).encode()).hexdigest(),
                        "cache_keys": "target raw content_sha256",
                    },
                )
            (stage / "manifest.json").write_text(_json(manifest) + "\n")
            _publish(stage, output)
        return manifest
    except (OSError, ValueError, KeyError, TypeError, sqlite3.Error) as error:
        if isinstance(error, EncodingCacheError):
            raise
        raise EncodingCacheError(f"Cannot build encoding cache: {error}") from error


class _EncodingCache:
    def __init__(self, connection, manifest, manifest_hash):
        self._connection = connection
        self._manifest = manifest
        self.manifest_sha256 = manifest_hash

    @property
    def safe_manifest(self):
        return json.loads(_json(self._manifest))

    def lookup(self, content_sha256: str):
        if not _hash(content_sha256):
            raise EncodingCacheError("lookup requires an exact SHA-256 content identity")
        row = self._connection.execute("SELECT dense,sparse_json FROM encodings WHERE content_sha256=?", (content_sha256,)).fetchone()
        return None if row is None else _representation(row[0], _document(row[1].encode()))


@contextmanager
def open_encoding_cache(path: Path, expected_encoder_fingerprint: str) -> Iterator[_EncodingCache]:
    """Validate a frozen cache before allowing content-addressed, read-only hits."""
    try:
        root = _path(path)
        manifest_bytes = _metadata(root / "manifest.json")
        manifest = _document(manifest_bytes)
        if manifest.get("schema_version") != _SCHEMA or manifest.get("encoder_fingerprint") != _fingerprint(expected_encoder_fingerprint):
            raise EncodingCacheError("encoding cache schema or model fingerprint mismatch")
        database = root / _DB
        if _file_record(database) != manifest["database"]:
            raise EncodingCacheError("encoding cache database hash/size mismatch")
        with _readonly(database) as connection:
            if connection.execute("PRAGMA user_version").fetchone()[0] != 1:
                raise EncodingCacheError("unsupported encoding cache database schema")
            count = 0
            for row in connection.execute("SELECT content_sha256,dense,sparse_json FROM encodings"):
                if not _hash(row[0]):
                    raise EncodingCacheError("invalid cached content identity")
                _representation(row[1], _document(row[2].encode()))
                count += 1
            if count != manifest["counts"]["cached_unique_contents"]:
                raise EncodingCacheError("encoding cache count differs from its manifest")
            yield _EncodingCache(connection, manifest, sha256(manifest_bytes).hexdigest())
        if _file_record(database) != manifest["database"] or _metadata(root / "manifest.json") != manifest_bytes:
            raise EncodingCacheError("encoding cache changed while being consumed")
    except (OSError, ValueError, KeyError, TypeError, sqlite3.Error) as error:
        if isinstance(error, EncodingCacheError):
            raise
        raise EncodingCacheError(f"Cannot consume encoding cache: {error}") from error


__all__ = ["EncodingCacheError", "build_encoding_cache", "open_encoding_cache"]
