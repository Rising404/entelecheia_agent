"""Source-grounded curation and retrieval-only export contracts."""

import hashlib
import json
import sqlite3

import pytest

from evals.docbench_hybrid_retrieval_optimize import retrieval_dataset as dataset


@pytest.fixture
def corpus(tmp_path):
    path = tmp_path / "dataset.sqlite"
    with sqlite3.connect(path) as connection:
        connection.executescript(dataset.SCHEMA)
        connection.execute("INSERT INTO documents VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", (
            "1", "synthetic.pdf", "pdf-sha", "source.sqlite", "db-sha", "run", "doc", "revision",
            "parser", "chunker", 1, "partial", "{}",
        ))
        content = "Board attendance table."
        connection.execute("INSERT INTO units VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", (
            "docbench:1:visual:table", "1", "table", content, "table", "revision", "[1]",
            '{"page":1}', "assets/table.jpg", hashlib.sha256(content.encode()).hexdigest(),
            "spatial_source_context", "{}",
        ))
        connection.execute("INSERT INTO queries VALUES (?,?,?,?,?,?,?,?)", (
            "docbench:1:0", "1", "text-only", "PRIVATE_QUERY", "PRIVATE_BAD_ANSWER",
            "PRIVATE_EVIDENCE", "pending", "",
        ))
    return path


def batch():
    return {
        "schema_version": "docbench-retrieval-curation-v1",
        "units": [{
            "unit_id": "docbench:1:chunk:table-transcript",
            "derived_from_unit_id": "docbench:1:visual:table", "kind": "chunk",
            "source_pdf_sha256": "pdf-sha", "source_pages": [1],
            "content": "Board attendance\nName | Held | Attended\nPerson | 4 | 4",
            "reviewer": "synthetic-reviewer", "note": "Complete table transcribed from the source image.",
        }],
        "corrections": [{
            "case_id": "docbench:1:0", "field": "answer", "old_value": "PRIVATE_BAD_ANSWER",
            "new_value": "4", "source_unit_id": "docbench:1:chunk:table-transcript",
            "reviewer": "synthetic-reviewer", "note": "Column concatenation corrected.",
        }],
    }


def curate(corpus, payload):
    path = corpus.parent / "curation.json"
    path.write_text(json.dumps(payload))
    with sqlite3.connect(corpus) as connection:
        return dataset._curate_corpus(connection, path)


def test_curation_adds_source_bound_text_and_audits_original_answer(corpus):
    report = curate(corpus, batch())
    assert report["unit_count"] == report["correction_count"] == 1
    with sqlite3.connect(corpus) as connection:
        assert connection.execute("SELECT answer FROM queries").fetchone()[0] == "4"
        assert connection.execute("SELECT original_value,revised_value FROM curation_events WHERE target_type='query'").fetchone() == ("PRIVATE_BAD_ANSWER", "4")
        text, method = connection.execute("SELECT content,description_method FROM units WHERE kind='chunk'").fetchone()
        assert "Held | Attended" in text and method == "reviewed_source_transcription"
        assert connection.execute("SELECT annotation_status FROM queries").fetchone()[0] == "pending"


@pytest.mark.parametrize("defect", ["pdf_hash", "page", "cross_doc_id", "answer_drift", "query_edit", "missing_review"])
def test_curation_rejects_invalid_sources_and_rolls_back_batch(corpus, defect):
    payload = batch()
    if defect == "pdf_hash":
        payload["units"][0]["source_pdf_sha256"] = "different-pdf"
    elif defect == "page":
        payload["units"][0]["source_pages"] = [2]
    elif defect == "cross_doc_id":
        payload["units"][0]["unit_id"] = "docbench:2:chunk:table-transcript"
    elif defect == "answer_drift":
        payload["corrections"][0]["old_value"] = "not-current-answer"
    elif defect == "query_edit":
        payload["corrections"][0]["field"] = "query"
    else:
        payload["units"][0]["reviewer"] = ""
    with pytest.raises(dataset.RetrievalDatasetError):
        curate(corpus, payload)
    with sqlite3.connect(corpus) as connection:
        assert connection.execute("SELECT count(*) FROM units").fetchone()[0] == 1
        assert connection.execute("SELECT answer FROM queries").fetchone()[0] == "PRIVATE_BAD_ANSWER"


def test_semantic_exports_exclude_qa_from_corpus_and_keep_scope(corpus):
    curate(corpus, batch())
    with sqlite3.connect(corpus) as connection:
        connection.execute("UPDATE queries SET annotation_status='reviewed'")
        connection.execute("INSERT INTO qrels VALUES (?,?,?,?,?)", (
            "docbench:1:0", "docbench:1:chunk:table-transcript", "primary", "reviewed", "synthetic-reviewer",
        ))
    out = corpus.parent / "exports"
    dataset.write_retrieval_exports(corpus, out)
    corpus_text = (out / "corpus.jsonl").read_text()
    for forbidden in ("PRIVATE_QUERY", "PRIVATE_BAD_ANSWER", "PRIVATE_EVIDENCE", "Column concatenation"):
        assert forbidden not in corpus_text
    query = json.loads((out / "queries.jsonl").read_text())
    assert query["query"] == "PRIVATE_QUERY" and query["allowed_kinds"] == ["chunk"]
    qrel = json.loads((out / "qrels.jsonl").read_text())
    assert qrel["corpus_id"] == "docbench:1:chunk:table-transcript" and qrel["score"] == 1
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["dataset_sha256"] == hashlib.sha256(corpus.read_bytes()).hexdigest()
    assert manifest["files"]["corpus.jsonl"]["rows"] == 2
