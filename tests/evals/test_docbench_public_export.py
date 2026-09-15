from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest

from evals.docbench.reproduce_or_run_script.export_results import (
    clean, digest, encode, export_archive, export_trajectory,
)
from scripts import check_repository_privacy as privacy


def _trajectory() -> dict:
    bodies = {
        "user": {"current_user_text": "Synthetic benchmark question", "file_catalog": [{
            "file_id": "att_correct", "relative_path": "attachments/att_correct/sample.pdf",
        }]},
        "tool_arguments": {"file_id": "att_wrong"},
        "tool_result": {"status": "unavailable", "reason_code": "file_source_unavailable",
                        "content": "Synthetic document text", "observation": "Generated observation"},
    }
    blobs = {}
    parts = []
    for role, body in bodies.items():
        text = json.dumps(body)
        sha = digest(text.encode())
        blobs[sha] = {"text": text, "byte_count": len(text.encode()), "truncated": False}
        parts.append({"role": role, "blob_sha256": sha, "seq": len(parts)})
    return {"format": "synthetic", "steps": [{
        "kind": "tool_call", "purpose": "prepare_files", "outcome": "ok",
        "duration_ms": 12, "metrics": {"input_tokens": 123}, "parts": parts,
    }], "blobs": blobs, "integrity": {"step_count": 1, "blob_count": 3}}


def test_export_keeps_failed_calls_ids_metrics_and_rehashes_body_references() -> None:
    raw = encode(_trajectory())
    projected = export_trajectory(raw, "runs/example/trajectory.json")
    step = projected["steps"][0]
    bodies = {part["role"]: json.loads(projected["blobs"][part["blob_sha256"]]["text"])
              for part in step["parts"]}
    assert step["outcome"] == "ok"
    assert step["metrics"]["input_tokens"] == 123
    assert bodies["tool_arguments"] == {"file_id": "att_wrong"}
    assert bodies["user"]["file_catalog"][0]["file_id"] == "att_correct"
    assert bodies["tool_result"]["reason_code"] == "file_source_unavailable"
    assert bodies["tool_result"]["observation"] == "Generated observation"
    assert bodies["tool_result"]["content"]["publication_omitted"] == "document_text"
    assert bodies["user"]["current_user_text"]["publication_omitted"] == "benchmark_input"
    assert projected["publication"]["source_sha256"] == digest(raw)
    for sha, blob in projected["blobs"].items():
        assert sha == digest(blob["text"].encode())
        assert blob["byte_count"] == len(blob["text"].encode())


def test_private_paths_and_credentials_are_cleaned_without_masking_relative_ids() -> None:
    data = {"reply": "Cannot read /Users/example/private/result.pdf", "api_key": "synthetic-secret",
            "arguments": {"path": "attachments/att_wrong/source.pdf"}}
    result = clean(data)
    assert result["reply"] == "Cannot read <local-path>"
    assert result["api_key"]["publication_omitted"] == "credential"
    assert result["arguments"] == data["arguments"]


def test_export_archive_preserves_scores_answers_and_never_overwrites(tmp_path: Path) -> None:
    archive = tmp_path / "private"
    run = archive / "runs/example"
    case = run / "cases/docbench:1:0"
    case.mkdir(parents=True)
    trajectory_raw = encode(_trajectory())
    (case / "trajectory.json").write_bytes(trajectory_raw)
    result = {"case_id": "docbench:1:0", "reply": "Unable to read the file", "question": "Private question",
              "reference_answer": "Private reference", "telemetry": {"input_tokens_total": 123},
              "trajectory": {"path": "cases/docbench:1:0/trajectory.json", "sha256": digest(trajectory_raw)}}
    (case / "result.json").write_bytes(encode(result))
    scoring = run / "scoring/cases"
    scoring.mkdir(parents=True)
    score = {"case_id": "docbench:1:0", "question": "Private question", "reference_answer": "Private reference",
             "system_answer": result["reply"], "score": 0, "status": "completed"}
    (scoring / "original.json").write_bytes(encode(score))
    (run / "scoring/summary.json").write_bytes(encode({"case_count": 1, "correct_count": 0, "cases": [score]}))
    (run / "run_manifest.json").write_bytes(encode({"run_id": "example", "frozen_cases": [score]}))
    source_before = {p: p.read_bytes() for p in archive.rglob('*') if p.is_file()}
    output = tmp_path / "public"
    exported = export_archive(archive, output)
    assert exported["executions"] == 1
    public_run = output / "runs/example"
    actual = json.loads((public_run / "cases/docbench-1-0/result.json").read_text())
    assert actual["reply"] == result["reply"]
    assert actual["telemetry"] == result["telemetry"]
    assert "question" not in actual and "reference_answer" not in actual
    assert digest((public_run / actual["trajectory"]["path"]).read_bytes()) == actual["trajectory"]["sha256"]
    assert json.loads((public_run / "scoring/cases/docbench-1-0.json").read_text())["score"] == 0
    assert all(p.read_bytes() == content for p, content in source_before.items())
    with pytest.raises(ValueError, match="already exists"):
        export_archive(archive, output)
    with pytest.raises(ValueError, match="outside"):
        export_archive(archive, archive / "nested-public")


def test_privacy_requires_reviewed_manifest_and_exact_complete_file_set(monkeypatch: pytest.MonkeyPatch) -> None:
    root = "evals/docbench/previous_results/synthetic"
    path = root + "/runs/example/cases/docbench-1-0/result.json"
    raw = encode({"case_id": "docbench:1:0", "reply": "A generated answer", "status": "completed"})
    manifest_path = root + "/publication_manifest.json"
    manifest = encode({"kind": "reviewed_public_evidence", "files": {
        "runs/example/cases/docbench-1-0/result.json": {"sha256": digest(raw), "bytes": len(raw)},
    }})
    monkeypatch.setattr(privacy, "REVIEWED_EVIDENCE_MANIFESTS", {manifest_path: (digest(manifest),)})
    files = {path: raw, manifest_path: manifest}
    assert privacy.inspect_snapshot(files, files.__getitem__) == []
    assert privacy.inspect_snapshot([path], files.__getitem__)
    assert privacy.inspect_snapshot([manifest_path], files.__getitem__)
    mutated = {**files, path: raw + b" "}
    assert any("hash/size mismatch" in v.reason for v in privacy.inspect_snapshot(mutated, mutated.__getitem__))
    mutated = {**files, manifest_path: manifest + b" "}
    assert privacy.inspect_snapshot(mutated, mutated.__getitem__)
    injected = {**files, root + "/raw.json": raw}
    assert privacy.inspect_snapshot(injected, injected.__getitem__)


@pytest.mark.parametrize("payload,expected_reason", [
    ({"case_id": "docbench:1:0", "question": "Private input", "reply": "Answer"}, "raw per-case"),
    ({"password": "opaque-value-should-not-be-public", "reply": "x" * 1000}, "structured credential"),
])
def test_reviewed_evidence_still_receives_secret_and_structured_checks(
    monkeypatch: pytest.MonkeyPatch, payload: dict, expected_reason: str,
) -> None:
    root = "evals/docbench/previous_results/synthetic"
    path = root + "/result.json"
    raw = encode(payload)
    manifest_path = root + "/publication_manifest.json"
    manifest = encode({"kind": "reviewed_public_evidence", "files": {
        "result.json": {"sha256": digest(raw), "bytes": len(raw)},
    }})
    monkeypatch.setattr(privacy, "REVIEWED_EVIDENCE_MANIFESTS", {manifest_path: (digest(manifest),)})
    monkeypatch.setattr(privacy, "MAX_TEXT_SCAN_BYTES", 64)
    files = {path: raw, manifest_path: manifest}
    assert any(expected_reason in v.reason for v in privacy.inspect_snapshot(files, files.__getitem__))


def test_reviewed_evidence_is_read_from_git_index_and_tree_not_worktree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    def git(*args: str) -> None:
        subprocess.run(["git", "-c", "user.name=Export test", "-c", "user.email=test@example.invalid", *args],
                       cwd=tmp_path, check=True, capture_output=True)

    root = "evals/docbench/previous_results/synthetic"
    path = root + "/result.json"
    manifest_path = root + "/publication_manifest.json"
    raw = encode({"case_id": "docbench:1:0", "reply": "Original answer"})
    manifest = encode({"kind": "reviewed_public_evidence", "files": {
        "result.json": {"sha256": digest(raw), "bytes": len(raw)},
    }})
    monkeypatch.setattr(privacy, "REVIEWED_EVIDENCE_MANIFESTS", {manifest_path: (digest(manifest),)})
    (tmp_path / root).mkdir(parents=True)
    (tmp_path / path).write_bytes(raw)
    (tmp_path / manifest_path).write_bytes(manifest)
    git("init", "-q")
    git("add", ".")
    paths, read = privacy.staged_snapshot(tmp_path)
    assert privacy.inspect_snapshot(paths, read) == []
    git("commit", "-qm", "Synthetic publication fixture")
    (tmp_path / path).write_bytes(raw + b" ")
    for paths, read in (privacy.staged_snapshot(tmp_path), privacy.tree_snapshot(tmp_path, "HEAD")):
        assert privacy.inspect_snapshot(paths, read) == []
    paths, read = privacy.worktree_snapshot(tmp_path)
    assert privacy.inspect_snapshot(paths, read)


def test_reviewed_publication_hashes_allow_only_explicitly_approved_snapshots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = "evals/docbench/previous_results/synthetic"
    path = root + "/result.json"
    manifest_path = root + "/publication_manifest.json"
    snapshots = []
    for answer in ("First approved answer", "Second approved answer", "Unreviewed answer"):
        raw = encode({"case_id": "docbench:1:0", "reply": answer})
        manifest = encode({"kind": "reviewed_public_evidence", "files": {
            "result.json": {"sha256": digest(raw), "bytes": len(raw)},
        }})
        snapshots.append({path: raw, manifest_path: manifest})
    monkeypatch.setattr(privacy, "REVIEWED_EVIDENCE_MANIFESTS", {
        manifest_path: tuple(digest(s[manifest_path]) for s in snapshots[:2]),
    })
    for snapshot in snapshots[:2]:
        assert not privacy.inspect_snapshot(snapshot, snapshot.__getitem__, committed_history=True)
    assert privacy.inspect_snapshot(snapshots[2], snapshots[2].__getitem__, committed_history=True)


def test_renamed_publication_keeps_old_paths_exclusive_to_exact_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old_root = "evals/docbench/previous_results/showcase_125"
    new_root = "evals/docbench/previous_results/first_gate_on_125"
    raw = encode({"case_id": "docbench:1:0", "reply": "Preserved generated answer"})
    manifest = encode({"kind": "reviewed_public_evidence", "files": {
        "result.json": {"sha256": digest(raw), "bytes": len(raw)},
    }})
    monkeypatch.setattr(privacy, "REVIEWED_EVIDENCE_MANIFESTS", {
        new_root + "/publication_manifest.json": (digest(manifest),),
    })
    monkeypatch.setattr(privacy, "REVIEWED_HISTORICAL_EVIDENCE_MANIFESTS", {
        old_root + "/publication_manifest.json": (digest(manifest),),
    })
    current = {new_root + "/publication_manifest.json": manifest, new_root + "/result.json": raw}
    historical = {old_root + "/publication_manifest.json": manifest, old_root + "/result.json": raw}
    assert not privacy.inspect_snapshot(current, current.__getitem__)
    assert privacy.inspect_snapshot(historical, historical.__getitem__)
    assert not privacy.inspect_snapshot(historical, historical.__getitem__, committed_history=True)
    for files, history in ((current, False), (historical, True)):
        for path in files:
            replaced = {**files, path: files[path] + b" "}
            assert privacy.inspect_snapshot(replaced, replaced.__getitem__, committed_history=history)
            incomplete = {key: value for key, value in files.items() if key != path}
            assert privacy.inspect_snapshot(incomplete, incomplete.__getitem__, committed_history=history)
    elsewhere = {path.replace("showcase_125", "unreviewed_copy"): data for path, data in historical.items()}
    assert privacy.inspect_snapshot(elsewhere, elsewhere.__getitem__, committed_history=True)
    injected = {**historical, old_root + "/analysis/README.md": b"Unreviewed historical prose"}
    assert privacy.inspect_snapshot(injected, injected.__getitem__, committed_history=True)


def test_publication_directory_migration_separates_worktree_index_and_old_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    def git(*args: str) -> None:
        subprocess.run(
            ["git", "-c", "user.name=Export test", "-c", "user.email=test@example.invalid", *args],
            cwd=tmp_path, check=True, capture_output=True,
        )

    old_root = "evals/docbench/previous_results/showcase_125"
    new_root = "evals/docbench/previous_results/first_gate_on_125"
    raw = encode({"case_id": "docbench:1:0", "reply": "Preserved generated answer"})
    manifest = encode({"kind": "reviewed_public_evidence", "files": {
        "result.json": {"sha256": digest(raw), "bytes": len(raw)},
    }})
    monkeypatch.setattr(privacy, "REVIEWED_EVIDENCE_MANIFESTS", {
        new_root + "/publication_manifest.json": (digest(manifest),),
    })
    monkeypatch.setattr(privacy, "REVIEWED_HISTORICAL_EVIDENCE_MANIFESTS", {
        old_root + "/publication_manifest.json": (digest(manifest),),
    })
    files = {"result.json": raw, "publication_manifest.json": manifest}
    (tmp_path / old_root).mkdir(parents=True)
    for relative, content in files.items():
        (tmp_path / old_root / relative).write_bytes(content)
    git("init", "-q")
    git("add", ".")
    git("commit", "-qm", "Historical publication fixture")
    for paths, read in (privacy.worktree_snapshot(tmp_path), privacy.staged_snapshot(tmp_path)):
        assert privacy.inspect_snapshot(paths, read)
    paths, read = privacy.tree_snapshot(tmp_path, "HEAD")
    assert not privacy.inspect_snapshot(paths, read, committed_history=True)

    (tmp_path / new_root).mkdir(parents=True)
    for relative, content in files.items():
        (tmp_path / old_root / relative).unlink()
        (tmp_path / new_root / relative).write_bytes(content)
    git("add", "--all")
    for paths, read in (privacy.worktree_snapshot(tmp_path), privacy.staged_snapshot(tmp_path)):
        assert not privacy.inspect_snapshot(paths, read)
    (tmp_path / new_root / "result.json").write_bytes(raw + b"unreviewed")
    paths, read = privacy.worktree_snapshot(tmp_path)
    assert privacy.inspect_snapshot(paths, read)
    paths, read = privacy.staged_snapshot(tmp_path)
    assert not privacy.inspect_snapshot(paths, read)
    paths, read = privacy.tree_snapshot(tmp_path, "HEAD")
    assert not privacy.inspect_snapshot(paths, read, committed_history=True)


def test_first_gate_on_publication_matches_its_complete_reviewed_manifest() -> None:
    repository = Path(__file__).resolve().parents[2]
    root = repository / "evals/docbench/previous_results/first_gate_on_125"
    paths = [path.relative_to(repository).as_posix() for path in root.rglob("*") if path.is_file()]
    assert paths, "the current reviewed publication must be present"
    assert not privacy.inspect_snapshot(paths, lambda path: (repository / path).read_bytes())


def test_archived_reviews_reconcile_with_public_answers_and_judge_without_merging_scores() -> None:
    root = Path(__file__).resolve().parents[2] / "evals/docbench/previous_results/gate_off_highland235b_123"
    reviews = {}
    for who, field, expected in (
        ("codex", "document_grounded_review_score", 103),
        ("claude", "claude_review_score", 106),
    ):
        path = root / "reviews" / who / "case_reviews.json"
        review = json.loads(path.read_bytes())
        cases = {c["case_id"]: c for c in review["cases"]}
        assert len(cases) == len(review["cases"]) == 123
        assert sum(c[field] for c in cases.values()) == expected
        assert sum(c["original_judge_score"] for c in cases.values()) == 97
        assert review["independent_human_gold"] is False
        assert review["official_comparable"] is False
        assert review["publication"]["original_scores_and_review_opinions_preserved"] is True
        for cid, case in cases.items():
            assert "question" not in case and "reference_answer" not in case
            evidence = case["public_evidence"]
            paths = {k: (path.parent / evidence[k]).resolve()
                     for k in ("result_path", "trajectory_path", "scoring_path")}
            assert all(p.is_relative_to(root) and p.is_file() for p in paths.values())
            raw = paths["result_path"].read_bytes()
            result = json.loads(raw)
            score = json.loads(paths["scoring_path"].read_bytes())
            assert cid == result["case_id"] == score["case_id"]
            assert score["score"] == case["original_judge_score"]
            assert digest(raw) == evidence["result_sha256"]
            assert digest(paths["trajectory_path"].read_bytes()) == evidence["trajectory_sha256"]
            assert digest(result["reply"].encode()) == case.get("answer_sha256", case.get("reply_sha256"))
            if "result_sha256" in case:
                assert case["result_sha256"] == result["publication"]["source_sha256"]
        reviews[who] = cases
    codex, claude = reviews["codex"], reviews["claude"]
    assert codex.keys() == claude.keys()
    assert sum(c["reference_aligned_review_score"] for c in codex.values()) == 101
    assert {cid for cid in codex if codex[cid]["document_grounded_review_score"] != claude[cid]["claude_review_score"]} == {
        "docbench:66:3", "docbench:80:5", "docbench:144:3",
    }
    assert {cid for cid, c in claude.items() if not c["agrees_with_codex"]} == {
        "docbench:66:3", "docbench:80:5", "docbench:144:3",
    }
