"""Private retrieval evaluation uses frozen units and reviewed, scoped labels."""

from hashlib import sha256
import json
from pathlib import Path
import sqlite3

import pytest

from evals.docbench_hybrid_retrieval_optimize.retrieval_eval import (
    PROJECT_ROOT,
    RetrievalEvaluationError,
    run_retrieval_evaluation,
    score_rankings,
)


def _pin_dataset(path: Path) -> None:
    (path.parent / "manifest.json").write_text(json.dumps({
        "schema_version": "docbench-retrieval-dataset-v1",
        "dataset_sha256": sha256(path.read_bytes()).hexdigest(),
    }))


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    path = tmp_path / "retrieval.sqlite"
    with sqlite3.connect(path) as db:
        db.executescript("""
            CREATE TABLE documents(doc_id TEXT PRIMARY KEY);
            CREATE TABLE units(
                unit_id TEXT PRIMARY KEY, doc_id TEXT, kind TEXT, content TEXT,
                source_revision TEXT, content_sha256 TEXT, asset_path TEXT, metadata_json TEXT
            );
            CREATE TABLE queries(
                case_id TEXT PRIMARY KEY, doc_id TEXT, question_type TEXT,
                query TEXT, annotation_status TEXT, answer TEXT, evidence TEXT,
                annotation_note TEXT
            );
            CREATE TABLE qrels(case_id TEXT, unit_id TEXT, role TEXT);
            INSERT INTO documents VALUES ('doc-a'), ('doc-b');
        """)
        for unit_id, doc_id, kind, content in [
            ("a", "doc-a", "chunk", "banana banana banana"),
            ("b", "doc-a", "chunk", "banana orange grape"),
            ("c", "doc-a", "chunk", "banana banana banana"),
            ("f", "doc-a", "figure", "banana silver silver silver"),
            ("t", "doc-a", "table", "silver orange grape"),
            ("foreign", "doc-b", "chunk", "banana banana banana banana"),
            ("foreign-figure", "doc-b", "figure", "silver silver silver silver"),
        ]:
            asset_path = None
            metadata = {}
            if kind != "chunk":
                asset_path = f"assets/{unit_id}.jpg"
                asset = tmp_path / asset_path
                asset.parent.mkdir(exist_ok=True)
                asset.write_bytes(b"synthetic-image")
                metadata["asset_sha256"] = sha256(asset.read_bytes()).hexdigest()
            db.execute(
                "INSERT INTO units VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (unit_id, doc_id, kind, content, "frozen-revision", sha256(content.encode()).hexdigest(),
                 asset_path, json.dumps(metadata)),
            )
        db.executemany(
            "INSERT INTO queries VALUES (?, 'doc-a', ?, ?, ?, 'annotationsecret', 'annotationsecret', 'annotationsecret')",
            [
                ("text", "text-only", "banana", "reviewed"),
                ("figure", "multimodal-f", "silver", "reviewed"),
                ("pending", "multimodal-t", "silver", "pending"),
                ("empty", "text-only", "annotationsecret", "reviewed"),
                ("no-primary", "text-only", "banana", "reviewed"),
            ],
        )
        db.executemany(
            "INSERT INTO qrels VALUES (?, ?, ?)",
            [("text", "a", "primary"), ("text", "b", "support"),
             ("figure", "f", "primary"), ("pending", "t", "primary"),
             ("empty", "b", "primary"), ("no-primary", "a", "support")],
        )
    _pin_dataset(path)
    return path


def _rankings(output: Path) -> dict:
    return {
        row["case_id"]: row
        for row in map(json.loads, (output / "rankings.jsonl").read_text().splitlines())
    }


def test_baseline_has_real_stable_bm25_and_preserves_private_snapshot(dataset, tmp_path):
    before = dataset.read_bytes()
    first = tmp_path / "first"
    report = run_retrieval_evaluation(dataset, first)
    second = tmp_path / "second"
    assert run_retrieval_evaluation(dataset, second) == report
    assert (first / "rankings.jsonl").read_bytes() == (second / "rankings.jsonl").read_bytes()
    assert dataset.read_bytes() == before
    assert not Path(str(dataset) + "-journal").exists()
    assert report["dataset_sha256"] == sha256(before).hexdigest()
    assert report["route_id"] == "sqlite_fts5_bm25"
    rows = _rankings(first)
    assert [hit["unit_id"] for hit in rows["text"]["hits"]] == ["a", "c", "b"]
    assert rows["text"]["hits"][0]["score"] == rows["text"]["hits"][1]["score"]
    assert rows["text"]["hits"][1]["score"] > rows["text"]["hits"][2]["score"] > 0
    assert rows["text"]["candidate_count"] == 3
    assert [hit["unit_id"] for hit in rows["figure"]["hits"]] == ["f", "t"]
    assert rows["figure"]["candidate_count"] == 2
    assert rows["pending"]["hits"]
    assert rows["empty"]["hits"] == []  # The answer/evidence/note are never indexed.
    assert report["retrieval"]["empty_result_case_ids"] == ["empty"]
    assert report["coverage"]["scored_queries"] == 3
    assert report["coverage"]["total_queries"] == 5
    assert report["coverage"]["pending_case_ids"] == ["pending"]
    assert report["coverage"]["reviewed_without_primary_case_ids"] == ["no-primary"]
    assert report["overall"]["hit_rate_at_k"][1] == pytest.approx(2 / 3)
    assert report["by_question_type"]["multimodal-t"] is None
    assert report["by_question_type"]["text-only"]["case_count"] == 2
    assert "not production BGE hybrid" in (first / "REPORT.md").read_text()


def test_unreviewed_cases_are_not_zero_scored(dataset, tmp_path):
    with sqlite3.connect(dataset) as db:
        db.execute("UPDATE queries SET annotation_status='pending'")
    _pin_dataset(dataset)
    report = run_retrieval_evaluation(dataset, tmp_path / "pending")
    assert report["overall"] is None
    assert report["coverage"]["scored_queries"] == 0
    assert len(report["coverage"]["pending_case_ids"]) == 5
    assert len(_rankings(tmp_path / "pending")) == 5


@pytest.mark.parametrize("unit_id, message", [
    ("missing", "unknown query or unit"),
    ("foreign", "wrong document"),
    ("f", "outside query scope"),
])
def test_invalid_gold_fails_before_output_creation(dataset, tmp_path, unit_id, message):
    with sqlite3.connect(dataset) as db:
        db.execute("UPDATE qrels SET unit_id=? WHERE case_id='text' AND role='primary'", (unit_id,))
    _pin_dataset(dataset)
    output = tmp_path / "invalid"
    with pytest.raises(ValueError, match=message):
        run_retrieval_evaluation(dataset, output)
    assert not output.exists()


def test_external_rankings_reuse_metrics_and_validate_scope(dataset):
    ranking = {case: [] for case in ("text", "figure", "pending", "empty", "no-primary")}
    ranking["text"] = ["b", "a", "a"]  # Support is not primary; duplicate does not add recall.
    with sqlite3.connect(f"{dataset.as_uri()}?mode=ro", uri=True) as db:
        report = score_rankings(db, ranking, route_id="external_test_route")
        text = next(case for case in report["overall"]["cases"] if case["case_id"] == "text")
        assert text["first_relevant_rank"] == 2
        assert text["recall_at_k"][3] == 1
        assert text["recall_at_k"][1] == 0
        assert text["ranked_count"] == 2
        assert report["route_id"] == "external_test_route"
        ranking["pending"] = ["foreign-figure"]
        with pytest.raises(ValueError, match="wrong document"):
            score_rankings(db, ranking)
        ranking["pending"] = ["unknown"]
        with pytest.raises(ValueError, match="unknown unit"):
            score_rankings(db, ranking)
        ranking.pop("pending")
        with pytest.raises(ValueError, match="exactly"):
            score_rankings(db, ranking)


def test_fts_operators_and_punctuation_do_not_change_query_scope(dataset, tmp_path):
    with sqlite3.connect(dataset) as db:
        db.execute("UPDATE queries SET query=? WHERE case_id='text'", ('banana\" OR doc_id:doc-b NOT *',))
        db.execute("UPDATE queries SET query='!?:()' WHERE case_id='empty'")
    _pin_dataset(dataset)
    run_retrieval_evaluation(dataset, tmp_path / "punctuation")
    rows = _rankings(tmp_path / "punctuation")
    assert {hit["unit_id"] for hit in rows["text"]["hits"]} == {"a", "b", "c"}
    assert rows["empty"]["hits"] == []


def test_rejects_overwrite_and_repository_output(dataset, tmp_path):
    output = tmp_path / "already-there"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("untouched")
    with pytest.raises(RetrievalEvaluationError, match="already exists"):
        run_retrieval_evaluation(dataset, output)
    assert list(output.iterdir()) == [marker]
    with pytest.raises(ValueError, match="outside the repository"):
        run_retrieval_evaluation(dataset, PROJECT_ROOT / "var" / "retrieval-output")
    symlink = tmp_path / "repo-link"
    symlink.symlink_to(PROJECT_ROOT, target_is_directory=True)
    with pytest.raises(ValueError, match="outside the repository"):
        run_retrieval_evaluation(dataset, symlink / "private-output")


def test_rejects_content_changes_after_snapshot_creation(dataset, tmp_path):
    with sqlite3.connect(dataset) as db:
        db.execute("UPDATE units SET content='tampered' WHERE unit_id='a'")
    _pin_dataset(dataset)
    with pytest.raises(ValueError, match="content hash mismatch"):
        run_retrieval_evaluation(dataset, tmp_path / "tampered")


def test_multimodal_support_can_be_text_but_rankings_remain_visual(dataset, tmp_path):
    with sqlite3.connect(dataset) as db:
        db.execute("INSERT INTO qrels VALUES ('figure', 'a', 'support')")
    _pin_dataset(dataset)
    report = run_retrieval_evaluation(dataset, tmp_path / "support")
    rows = _rankings(tmp_path / "support")
    assert {hit["unit_id"] for hit in rows["figure"]["hits"]} == {"f", "t"}
    assert report["by_question_type"]["multimodal-f"]["mean_recall_at_k"][1] == 1
    rankings = {case_id: [hit["unit_id"] for hit in row["hits"]] for case_id, row in rows.items()}
    rankings["figure"].append("a")
    with sqlite3.connect(dataset) as db:
        with pytest.raises(ValueError, match="outside query scope"):
            score_rankings(db, rankings)
        db.execute("UPDATE qrels SET unit_id='foreign' WHERE case_id='figure' AND role='support'")
    _pin_dataset(dataset)
    with pytest.raises(ValueError, match="wrong document"):
        run_retrieval_evaluation(dataset, tmp_path / "foreign-support")


@pytest.mark.parametrize("mutation", ["query", "gold", "answer"])
def test_unpinned_dataset_changes_are_rejected(dataset, tmp_path, mutation):
    with sqlite3.connect(dataset) as db:
        if mutation == "gold":
            db.execute("UPDATE qrels SET unit_id='b' WHERE case_id='text' AND role='primary'")
        else:
            db.execute(f"UPDATE queries SET {mutation}='tampered' WHERE case_id='text'")
    with pytest.raises(ValueError, match="dataset hash does not match manifest"):
        run_retrieval_evaluation(dataset, tmp_path / "changed")
    assert not (tmp_path / "changed").exists()


@pytest.mark.parametrize("mutation, message", [
    ("delete", "asset is missing"),
    ("change", "asset hash mismatch"),
    ("escape", "asset escapes dataset"),
    ("missing-path", "has no asset"),
])
def test_visual_assets_must_exist_and_match_frozen_identity(dataset, tmp_path, mutation, message):
    asset = tmp_path / "assets/f.jpg"
    if mutation == "delete":
        asset.unlink()
    elif mutation == "change":
        asset.write_bytes(b"different-image")
    else:
        with sqlite3.connect(dataset) as db:
            db.execute("UPDATE units SET asset_path=? WHERE unit_id='f'", (
                "../../outside.jpg" if mutation == "escape" else None,
            ))
        _pin_dataset(dataset)
    with pytest.raises(ValueError, match=message):
        run_retrieval_evaluation(dataset, tmp_path / "invalid-assets")
    assert not (tmp_path / "invalid-assets").exists()


def test_requires_a_manifest_and_rejects_pending_wal(dataset, tmp_path):
    manifest = tmp_path / "manifest.json"
    payload = manifest.read_bytes()
    manifest.unlink()
    with pytest.raises(RetrievalEvaluationError, match="manifest.json"):
        run_retrieval_evaluation(dataset, tmp_path / "no-manifest")
    manifest.write_bytes(payload)
    Path(str(dataset) + "-wal").write_bytes(b"uncheckpointed-changes")
    with pytest.raises(ValueError, match="uncheckpointed WAL"):
        run_retrieval_evaluation(dataset, tmp_path / "wal")


@pytest.mark.parametrize("filename", ["rankings.jsonl", "report.json", "REPORT.md"])
def test_failed_report_write_does_not_publish_partial_output(dataset, tmp_path, monkeypatch, filename):
    output = tmp_path / "atomic-report"
    original_dataset = dataset.read_bytes()
    write_text = Path.write_text

    def fail_selected_write(path, *args, **kwargs):
        if path.name == filename:
            raise OSError("synthetic report write failure")
        return write_text(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "write_text", fail_selected_write)
        with pytest.raises(OSError, match="synthetic report write failure"):
            run_retrieval_evaluation(dataset, output)
    assert not output.exists()
    assert not list(tmp_path.glob(".atomic-report-*"))
    assert dataset.read_bytes() == original_dataset
    run_retrieval_evaluation(dataset, output)
    assert {path.name for path in output.iterdir()} == {"rankings.jsonl", "report.json", "REPORT.md"}


@pytest.mark.parametrize("defect", ["missing_dataset", "bad_json", "non_object", "bad_database"])
def test_run_input_failures_use_evaluation_error(dataset, tmp_path, defect):
    if defect == "missing_dataset":
        dataset.unlink()
    elif defect == "bad_database":
        dataset.write_bytes(b"not a SQLite database")
        _pin_dataset(dataset)
    else:
        (tmp_path / "manifest.json").write_text("{broken" if defect == "bad_json" else "[]")
    with pytest.raises(RetrievalEvaluationError):
        run_retrieval_evaluation(dataset, tmp_path / "invalid-input")
    assert not (tmp_path / "invalid-input").exists()


def test_report_publication_failure_cleans_stage(dataset, tmp_path, monkeypatch):
    def fail_rename(*args, **kwargs):
        raise OSError("synthetic publication failure")

    monkeypatch.setattr(Path, "rename", fail_rename)
    with pytest.raises(OSError, match="synthetic publication failure"):
        run_retrieval_evaluation(dataset, tmp_path / "unpublished")
    assert not (tmp_path / "unpublished").exists()
    assert not list(tmp_path.glob(".unpublished-*"))


def test_report_does_not_replace_output_created_during_run(dataset, tmp_path, monkeypatch):
    output = tmp_path / "concurrent-output"
    write_text = Path.write_text

    def create_competing_output(path, *args, **kwargs):
        result = write_text(path, *args, **kwargs)
        if path.name == "REPORT.md":
            output.mkdir()
            write_text(output / "keep.txt", "another writer")
        return result

    monkeypatch.setattr(Path, "write_text", create_competing_output)
    with pytest.raises(RetrievalEvaluationError, match="already exists"):
        run_retrieval_evaluation(dataset, output)
    assert [path.name for path in output.iterdir()] == ["keep.txt"]
    assert not list(tmp_path.glob(".concurrent-output-*"))


def test_report_explains_flat_gold_and_global_bm25_statistics(dataset, tmp_path):
    output = tmp_path / "methodology"
    report = run_retrieval_evaluation(dataset, output)
    assert "do not measure answer completeness" in report["metric_interpretation"]
    assert "all snapshot units" in report["retrieval"]["statistics_scope"]
    markdown = (output / "REPORT.md").read_text()
    assert "global IDF" in markdown
    assert "Equivalent alternatives and jointly necessary" in markdown


@pytest.fixture
def vector_dataset(dataset, tmp_path):
    asset = tmp_path / "assets/vector.jpg"
    asset.write_bytes(b"synthetic-vector-graphics-crop")
    with sqlite3.connect(dataset) as connection:
        for unit_id, doc_id in [("vector", "doc-a"), ("foreign-vector", "doc-b")]:
            content = "silver vectoronly"
            connection.execute("INSERT INTO units VALUES (?, ?, ?, ?, ?, ?, ?, ?)", (
                unit_id, doc_id, "vector_graphics", content, "vector-revision",
                sha256(content.encode()).hexdigest(), "assets/vector.jpg",
                json.dumps({"asset_sha256": sha256(asset.read_bytes()).hexdigest()}),
            ))
    _pin_dataset(dataset)
    return dataset


def test_vector_graphics_gold_is_scored_and_shared_as_visual_distractor(vector_dataset, tmp_path):
    with sqlite3.connect(vector_dataset) as connection:
        connection.execute("UPDATE queries SET query='vectoronly' WHERE case_id IN ('figure', 'empty')")
        connection.execute("UPDATE qrels SET unit_id='vector' WHERE case_id='figure' AND role='primary'")
        connection.execute("UPDATE queries SET annotation_status='reviewed' WHERE case_id='pending'")
    _pin_dataset(vector_dataset)
    output = tmp_path / "vector-evaluation"
    report = run_retrieval_evaluation(vector_dataset, output)
    rankings = _rankings(output)
    assert [hit["unit_id"] for hit in rankings["figure"]["hits"]] == ["vector"]
    assert rankings["figure"]["candidate_count"] == 3
    assert {hit["unit_id"] for hit in rankings["pending"]["hits"]} == {"f", "t", "vector"}
    assert rankings["empty"]["hits"] == []
    assert report["by_question_type"]["multimodal-f"]["hit_rate_at_k"][1] == 1
    assert report["coverage"]["scored_queries"] == 4
    assert "vector_graphics" in report["scope"]
    assert "vector_graphics" in (output / "REPORT.md").read_text()


def test_vector_graphics_assets_require_the_same_integrity_as_other_visuals(vector_dataset, tmp_path):
    (tmp_path / "assets/vector.jpg").unlink()
    with pytest.raises(RetrievalEvaluationError, match="visual asset is missing"):
        run_retrieval_evaluation(vector_dataset, tmp_path / "missing-vector")


def test_vector_graphics_cannot_be_primary_for_text_queries(vector_dataset, tmp_path):
    with sqlite3.connect(vector_dataset) as connection:
        connection.execute("UPDATE qrels SET unit_id='vector' WHERE case_id='text' AND role='primary'")
    _pin_dataset(vector_dataset)
    with pytest.raises(RetrievalEvaluationError, match="outside query scope"):
        run_retrieval_evaluation(vector_dataset, tmp_path / "invalid-vector-gold")
