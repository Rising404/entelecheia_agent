"""Offline, document-scoped retrieval over a frozen DocBench snapshot.

Only unit content enters the temporary FTS index. This is a SQLite BM25 baseline,
not the production BGE hybrid route. Unreviewed labels never contribute scores.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import re
import sqlite3
import tempfile
from typing import Mapping, Sequence

from personagraph.retrieval.contracts import SourceType, SourceUnitRef
from personagraph.retrieval.relevance import RelevanceCase, evaluate_ranked_retrieval

from .paths import PROJECT_ROOT as PROJECT_ROOT, artifact_path
from .retrieval_dataset import TEXT_UNIT_KINDS, VISUAL_UNIT_KINDS


class RetrievalEvaluationError(ValueError):
    """A retrieval snapshot or evaluation request violates its contract."""


ROUTE_ID = "sqlite_fts5_bm25"
CUTOFFS = (1, 3, 5, 10)
QUESTION_KINDS = {
    "text-only": frozenset(TEXT_UNIT_KINDS),
    "multimodal-t": frozenset(VISUAL_UNIT_KINDS),
    "multimodal-f": frozenset(VISUAL_UNIT_KINDS),
}


def _rows(connection: sqlite3.Connection, sql: str) -> list[dict]:
    cursor = connection.execute(sql)
    columns = [item[0] for item in cursor.description]
    return [dict(zip(columns, row, strict=True)) for row in cursor.fetchall()]


def _snapshot(connection: sqlite3.Connection) -> tuple[dict, dict, list[dict]]:
    documents = {row[0] for row in connection.execute("SELECT doc_id FROM documents")}
    units = {
        row["unit_id"]: row
        for row in _rows(
            connection,
            "SELECT unit_id, doc_id, kind, content, source_revision, content_sha256 FROM units",
        )
    }
    queries = {
        row["case_id"]: row
        for row in _rows(
            connection,
            "SELECT case_id, doc_id, question_type, query, annotation_status FROM queries",
        )
    }
    qrels = _rows(connection, "SELECT case_id, unit_id, role FROM qrels")
    for unit in units.values():
        if unit["doc_id"] not in documents or (
            unit["kind"] not in TEXT_UNIT_KINDS and unit["kind"] not in VISUAL_UNIT_KINDS
        ):
            raise ValueError(f"invalid unit scope or kind: {unit['unit_id']}")
        if sha256(unit["content"].encode("utf-8")).hexdigest() != unit["content_sha256"]:
            raise ValueError(f"unit content hash mismatch: {unit['unit_id']}")
    for query in queries.values():
        if query["doc_id"] not in documents or query["question_type"] not in QUESTION_KINDS:
            raise ValueError(f"invalid query scope or type: {query['case_id']}")
        if query["annotation_status"] not in {"pending", "reviewed"}:
            raise ValueError(f"invalid annotation status: {query['case_id']}")
        if not query["query"].strip():
            raise ValueError(f"empty query: {query['case_id']}")
    for qrel in qrels:
        query = queries.get(qrel["case_id"])
        unit = units.get(qrel["unit_id"])
        if query is None or unit is None:
            raise ValueError("qrel refers to unknown query or unit")
        if qrel["role"] not in {"primary", "support"}:
            raise ValueError(f"invalid qrel role: {qrel['role']}")
        _validate_scope(query, unit, require_retrievable=qrel["role"] == "primary")
    return units, queries, qrels


def _validate_scope(query: dict, unit: dict, *, require_retrievable: bool = True) -> None:
    if unit["doc_id"] != query["doc_id"]:
        raise ValueError(f"unit belongs to wrong document for {query['case_id']}")
    if require_retrievable and unit["kind"] not in QUESTION_KINDS[query["question_type"]]:
        raise ValueError(f"unit kind outside query scope for {query['case_id']}")


def _reference(unit: dict) -> SourceUnitRef:
    return SourceUnitRef(
        source_type=SourceType.DOCUMENT if unit["kind"] in TEXT_UNIT_KINDS else SourceType.PICTURE,
        # The snapshot's unit ID is the scored retrieval unit, including image crops.
        source_unit_id=unit["unit_id"],
        source_revision=unit["source_revision"],
        indexed_content_hash=unit["content_sha256"],
    )


def score_rankings(
    connection: sqlite3.Connection,
    rankings: Mapping[str, Sequence[str]],
    *,
    route_id: str = ROUTE_ID,
) -> dict:
    """Score externally supplied rankings against reviewed primary qrels only.

    Every query must have an explicit ranking (possibly empty). Unknown units,
    out-of-scope results and invalid gold labels are rejected, including pending
    cases. The connection is read only from this function's perspective.
    """
    units, queries, qrels = _snapshot(connection)
    if set(rankings) != set(queries):
        raise ValueError("rankings must contain exactly the snapshot query IDs")
    refs = {unit_id: _reference(unit) for unit_id, unit in units.items()}
    for case_id, ranked in rankings.items():
        if isinstance(ranked, (str, bytes)):
            raise ValueError("a ranking must be a sequence of unit IDs")
        for unit_id in ranked:
            if unit_id not in units:
                raise ValueError(f"ranking refers to unknown unit: {unit_id}")
            _validate_scope(queries[case_id], units[unit_id])

    primary: dict[str, set[str]] = {}
    for qrel in qrels:
        if qrel["role"] == "primary":
            primary.setdefault(qrel["case_id"], set()).add(qrel["unit_id"])
    cases = [
        RelevanceCase(
            case_id=case_id,
            query=query["query"],
            relevant_refs=frozenset(refs[unit_id] for unit_id in primary[case_id]),
        )
        for case_id, query in sorted(queries.items())
        if query["annotation_status"] == "reviewed" and primary.get(case_id)
    ]

    def aggregate(selected: list[RelevanceCase]) -> dict | None:
        if not selected:
            return None
        return asdict(
            evaluate_ranked_retrieval(
                route_id=route_id,
                cases=selected,
                ranked_refs_by_case_id={
                    case.case_id: tuple(refs[unit_id] for unit_id in rankings[case.case_id])
                    for case in selected
                },
                cutoffs=CUTOFFS,
            )
        )

    scored_ids = {case.case_id for case in cases}
    total_by_type = Counter(query["question_type"] for query in queries.values())
    scored_by_type = Counter(queries[case_id]["question_type"] for case_id in scored_ids)
    return {
        "route_id": route_id,
        "scope": (
            "question_document; text=" + "+".join(sorted(TEXT_UNIT_KINDS))
            + "; multimodal=" + "+".join(sorted(VISUAL_UNIT_KINDS))
        ),
        "gold_policy": "reviewed primary qrels only; support is not scored",
        "metric_interpretation": (
            "Binary relevance over a flat primary-gold set: Hit means at least one gold unit; "
            "Recall measures gold-unit coverage. Equivalent alternatives and jointly necessary "
            "evidence are not grouped, so these metrics do not measure answer completeness."
        ),
        "cutoffs": list(CUTOFFS),
        "coverage": {
            "total_queries": len(queries),
            "scored_queries": len(cases),
            "scored_fraction": len(cases) / len(queries) if queries else 0.0,
            "pending_case_ids": sorted(
                case_id for case_id, query in queries.items()
                if query["annotation_status"] == "pending"
            ),
            "reviewed_without_primary_case_ids": sorted(
                case_id for case_id, query in queries.items()
                if query["annotation_status"] == "reviewed" and not primary.get(case_id)
            ),
            "by_question_type": {
                kind: {"total_queries": total_by_type[kind], "scored_queries": scored_by_type[kind]}
                for kind in QUESTION_KINDS
            },
        },
        "overall": aggregate(cases),
        "by_question_type": {
            kind: aggregate([case for case in cases if queries[case.case_id]["question_type"] == kind])
            for kind in QUESTION_KINDS
        },
    }


def _match_expression(query: str) -> str:
    # Treat punctuation/FTS operators as data. OR retains partial lexical matches.
    tokens = dict.fromkeys(re.findall(r"[^\W_]+", query.casefold(), flags=re.UNICODE))
    return " OR ".join(f'"{token}"' for token in tokens)


def _rank_units(units: dict, queries: dict) -> tuple[dict, list[dict]]:
    rankings: dict[str, list[str]] = {}
    records: list[dict] = []
    with sqlite3.connect(":memory:") as index:
        index.execute(
            "CREATE VIRTUAL TABLE candidates USING fts5("
            "unit_id UNINDEXED, doc_id UNINDEXED, kind UNINDEXED, content, "
            "tokenize='unicode61 remove_diacritics 2')"
        )
        index.executemany(
            "INSERT INTO candidates(unit_id, doc_id, kind, content) VALUES (?, ?, ?, ?)",
            [(unit_id, unit["doc_id"], unit["kind"], unit["content"]) for unit_id, unit in sorted(units.items())],
        )
        for case_id, query in sorted(queries.items()):
            expression = _match_expression(query["query"])
            kinds = sorted(QUESTION_KINDS[query["question_type"]])
            placeholders = ",".join("?" for _ in kinds)
            candidates = index.execute(
                f"SELECT count(*) FROM candidates WHERE doc_id=? AND kind IN ({placeholders})",
                [query["doc_id"], *kinds],
            ).fetchone()[0]
            hits = index.execute(
                "SELECT unit_id, bm25(candidates) AS distance FROM candidates "
                f"WHERE candidates MATCH ? AND doc_id=? AND kind IN ({placeholders}) "
                "ORDER BY distance ASC, unit_id ASC",
                [expression, query["doc_id"], *kinds],
            ).fetchall() if expression else []
            rankings[case_id] = [row[0] for row in hits]
            records.append({
                "case_id": case_id,
                "doc_id": query["doc_id"],
                "question_type": query["question_type"],
                "annotation_status": query["annotation_status"],
                "candidate_count": candidates,
                "hits": [
                    {"rank": rank, "unit_id": unit_id, "score": -distance}
                    for rank, (unit_id, distance) in enumerate(hits, start=1)
                ],
            })
    return rankings, records


def _external_path(path: Path) -> Path:
    return artifact_path(path)


def _dataset_manifest(dataset: Path) -> tuple[dict, str]:
    manifest_bytes = (dataset.parent / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    if not isinstance(manifest, dict) or manifest.get("schema_version") != "docbench-retrieval-dataset-v1":
        raise ValueError("unsupported retrieval dataset manifest schema")
    if manifest.get("dataset_sha256") != sha256(dataset.read_bytes()).hexdigest():
        raise ValueError("dataset hash does not match manifest")
    wal = Path(str(dataset) + "-wal")
    if wal.exists() and wal.stat().st_size:
        raise ValueError("frozen retrieval dataset must not have an uncheckpointed WAL")
    return manifest, sha256(manifest_bytes).hexdigest()


def _validate_visual_assets(connection: sqlite3.Connection, dataset_root: Path) -> None:
    visual_kinds = sorted(VISUAL_UNIT_KINDS)
    placeholders = ",".join("?" for _ in visual_kinds)
    for unit_id, raw_path, metadata in connection.execute(
        f"SELECT unit_id, asset_path, metadata_json FROM units WHERE kind IN ({placeholders})",
        visual_kinds,
    ):
        if not isinstance(raw_path, str) or not raw_path:
            raise ValueError(f"visual unit has no asset: {unit_id}")
        relative = Path(raw_path)
        asset = (dataset_root / relative).resolve()
        if relative.is_absolute() or not asset.is_relative_to(dataset_root):
            raise ValueError(f"visual asset escapes dataset: {unit_id}")
        if not asset.is_file():
            raise ValueError(f"visual asset is missing: {unit_id}")
        metadata = json.loads(metadata)
        if not isinstance(metadata, dict):
            raise ValueError(f"invalid visual asset metadata: {unit_id}")
        expected = metadata.get("asset_sha256")
        if expected != sha256(asset.read_bytes()).hexdigest():
            raise ValueError(f"visual asset hash mismatch: {unit_id}")


def _markdown_report(report: dict) -> str:
    coverage = report["coverage"]
    lines = [
        "# DocBench retrieval baseline",
        "",
        f"Route: `{report['route_id']}`. SQLite FTS5 BM25; not production BGE hybrid retrieval.",
        "",
        "Search is scoped to the question document. Text queries search chunks; "
        "multimodal queries search all supported visual descriptions together ("
        + ", ".join(sorted(VISUAL_UNIT_KINDS)) + ").",
        "Only unit content is indexed. Answer, evidence and annotation fields never enter retrieval.",
        "No minimum score threshold or artificial fallback hit is applied.",
        "BM25 uses global IDF and length statistics from all snapshot units; "
        "document and modality restrictions filter candidates without recomputing those statistics.",
        report["metric_interpretation"],
        "",
        f"Scored {coverage['scored_queries']} / {coverage['total_queries']} queries. "
        f"Pending: {len(coverage['pending_case_ids'])}; reviewed without primary labels: "
        f"{len(coverage['reviewed_without_primary_case_ids'])}.",
        "Unscored queries are excluded from metric denominators, not treated as successes or failures.",
        "",
        "| Group | Scored | MRR | Recall@1 | Recall@3 | Recall@5 | Recall@10 | nDCG@10 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, metrics in [("overall", report["overall"]), *report["by_question_type"].items()]:
        if metrics is None:
            lines.append(f"| {name} | 0 | — | — | — | — | — | — |")
            continue
        values = [metrics["mean_reciprocal_rank"]]
        values.extend(metrics["mean_recall_at_k"][cutoff] for cutoff in CUTOFFS)
        values.append(metrics["mean_ndcg_at_k"][10])
        lines.append(f"| {name} | {metrics['case_count']} | " + " | ".join(f"{value:.4f}" for value in values) + " |")
    lines.extend(["", "Full per-case Hit, Recall and nDCG at 1, 3, 5 and 10 are in report.json.", ""])
    return "\n".join(lines)


def run_retrieval_evaluation(dataset_path: Path, output_dir: Path) -> dict:
    """Read a frozen SQLite snapshot and create a new evaluation directory."""
    try:
        dataset = _external_path(dataset_path)
        output = _external_path(output_dir)
        if not dataset.is_file():
            raise RetrievalEvaluationError(f"Dataset does not exist: {dataset}")
        if output.exists():
            raise RetrievalEvaluationError(f"Output already exists: {output}")
        manifest, manifest_sha256 = _dataset_manifest(dataset)
        with sqlite3.connect(f"{dataset.as_uri()}?mode=ro&immutable=1", uri=True) as connection:
            connection.execute("PRAGMA query_only=ON")
            connection.execute("BEGIN")
            _validate_visual_assets(connection, dataset.parent)
            units, queries, _qrels = _snapshot(connection)
            rankings, records = _rank_units(units, queries)
            report = score_rankings(connection, rankings)
        if manifest["dataset_sha256"] != sha256(dataset.read_bytes()).hexdigest():
            raise RetrievalEvaluationError("dataset changed during retrieval evaluation")
    except RetrievalEvaluationError:
        raise
    except (OSError, ValueError, sqlite3.Error) as error:
        raise RetrievalEvaluationError(f"Cannot evaluate retrieval snapshot: {error}") from error
    report["dataset_sha256"] = manifest["dataset_sha256"]
    report["manifest_sha256"] = manifest_sha256
    report["retrieval"] = {
        "engine": "SQLite FTS5",
        "sqlite_version": sqlite3.sqlite_version,
        "tokenizer": "unicode61 remove_diacritics 2",
        "query_operator": "OR over unique literal alphanumeric tokens",
        "indexed_fields": ["units.content"],
        "ranking": "ascending bm25 distance; ties by unit_id",
        "statistics_scope": "global IDF and length statistics over all snapshot units before scope filtering",
        "rank_limit": None,
        "unit_count": len(units),
        "empty_result_case_ids": sorted(case_id for case_id, ranked in rankings.items() if not ranked),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
        stage = Path(temporary)
        (stage / "rankings.jsonl").write_text(
            "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records), encoding="utf-8"
        )
        (stage / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        (stage / "REPORT.md").write_text(_markdown_report(report), encoding="utf-8")
        if output.exists():
            raise RetrievalEvaluationError(f"Output already exists: {output}")
        stage.rename(output)
    return report
