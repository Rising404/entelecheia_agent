from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

from scripts.check_repository_privacy import (
    CheckError,
    FORBIDDEN_SUFFIXES,
    TEXT_SUFFIXES,
    eval_summary_violations,
    inspect_snapshot,
    path_violations,
    structured_privacy_violations,
    text_secret_violations,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CHECKER = REPOSITORY_ROOT / "scripts" / "check_repository_privacy.py"
PUBLIC_EVALUATION = "evals/docbench/previous_results/showcase_125.summary.json"
PUBLIC_SCREENSHOT = "assets/entelecheia-desktop.png"
HISTORICAL_EVALUATION_PATH = "evals/docbench/results/showcase_125.summary.json"
REVIEWED_ANALYSIS_FILES = (
    "evals/docbench/previous_results/showcase_125/analysis/README.md",
    "evals/docbench/previous_results/showcase_125/analysis/EVALUATION_REVIEW.md",
    "evals/docbench/previous_results/showcase_125/analysis/FAILURE_ATTRIBUTION.md",
)
PRIVATE_ANALYSIS_FILES = (
    "evals/docbench/results/analysis/raw.md",
    "evals/docbench/results/analysis/claude_review_scores.json",
    "evals/docbench/results/analysis/nested/README.md",
    "evals/docbench/results/analysis/raw.summary.json",
    "evals/another_benchmark/results/analysis/README.md",
    "evals/docbench/previous_results/showcase_125/analysis/raw.md",
    "evals/docbench/previous_results/showcase_125/analysis/claude_review_scores.json",
    "evals/docbench/previous_results/private_runs/showcase_125/README.md",
    "evals/docbench/previous_results/private_runs/showcase_125/cases/example/result.json",
)
PRODUCT_BINARY_UPLOAD_SUFFIXES = frozenset(
    {
        "avif",
        "bmp",
        "doc",
        "docx",
        "flac",
        "gif",
        "jpg",
        "mkv",
        "mp3",
        "mp4",
        "ogg",
        "ole",
        "pdf",
        "png",
        "ppt",
        "pptx",
        "wav",
        "webp",
        "xlsx",
        "zip",
    }
)
PRODUCT_TEXT_UPLOAD_SUFFIXES = frozenset(
    {
        ".c",
        ".cpp",
        ".css",
        ".csv",
        ".go",
        ".h",
        ".html",
        ".ini",
        ".java",
        ".js",
        ".json",
        ".log",
        ".markdown",
        ".md",
        ".py",
        ".rb",
        ".rs",
        ".sh",
        ".sql",
        ".toml",
        ".ts",
        ".tsv",
        ".txt",
        ".xml",
        ".yaml",
        ".yml",
    }
)


def _public_evaluation() -> dict[str, object]:
    return json.loads((REPOSITORY_ROOT / PUBLIC_EVALUATION).read_bytes())


def _public_evaluation_errors(document: dict[str, object]) -> list[str]:
    return [
        violation.reason
        for violation in eval_summary_violations(
            PUBLIC_EVALUATION, json.dumps(document).encode()
        )
    ]


def test_published_evaluation_is_closed_and_keeps_judge_and_review_scopes_separate() -> None:
    document = _public_evaluation()
    assert not _public_evaluation_errors(document)
    assert document["schema_version"] == "entelecheia-docbench-public-evaluation"
    assert document["official_comparable"] is False
    assert document["cc_review_status"] == "archived_provisional"
    assert document["codex_review_status"] == "archived_provisional"
    assert document["cc_review"] == {
        "artifact_sha256": "de04415ef7871877e16ab6d0ce5cbf74deba9a13c497723a10b69af21bbe5820",
        "reviewer_kind": "model",
        "blind_review": False,
        "source_check_basis": "reviewer_self_report",
    }
    assert document["totals"] == {
        "case_count": 125,
        "execution_count": 133,
        "original_judge_correct": 91,
        "retry_merged_judge_correct": 96,
        "codex_archived_first_correct": 99,
        "codex_archived_selected_correct": 104,
        "cc_archived_first_correct": 100,
        "cc_archived_selected_correct": 105,
        "cc_debatable_count": 8,
        "cc_source_checked_reported_count": 13,
    }
    assert [run["case_count"] for run in document["runs"]] == [125, 3, 5]
    assert any(
        row["worker_exit_code"] != 0 and row["judge_score"] == 1
        for row in document["executions"]
    ), "process exit must not silently replace the answer judge's score"


def test_public_evaluation_rejects_unknown_fields_at_every_record_level() -> None:
    for location in ((), ("cc_review",), ("totals",), ("runs", 0), ("cases", 0), ("executions", 0)):
        document = _public_evaluation()
        record = document
        for key in location:
            record = record[key]
        record["unreviewed_extra"] = "safe-looking-identifier"
        assert _public_evaluation_errors(document), location
    document = _public_evaluation()
    del document["cases"][0]["qa_sha256"]
    assert _public_evaluation_errors(document)


def test_public_evaluation_rejects_wrong_types_nonfinite_numbers_and_unreviewed_states() -> None:
    mutations = (
        (("official_comparable",), True),
        (("cc_review_status",), "completed"),
        (("cc_review_status",), "pending"),
        (("cc_review",), []),
        (("cc_review", "artifact_sha256"), "not-a-sha256"),
        (("cc_review", "reviewer_kind"), "human"),
        (("cc_review", "blind_review"), True),
        (("cc_review", "blind_review"), 0),
        (("cc_review", "source_check_basis"), "independently_verified"),
        (("codex_review_status",), "gold"),
        (("runs",), {}),
        (("runs", 0, "ordinal"), True),
        (("runs", 0, "source_sha256"), "not-a-sha256"),
        (("runs", 0, "input_tokens"), True),
        (("runs", 0, "worktree_dirty"), 1),
        (("cases", 0), []),
        (("cases", 0, "case_id"), "private-session-id"),
        (("cases", 0, "original_judge_score"), True),
        (("cases", 0, "original_judge_score"), 1.0),
        (("cases", 0, "codex_archived_selected_score"), 2),
        (("cases", 0, "cc_archived_first_score"), True),
        (("cases", 0, "cc_archived_selected_score"), 1.0),
        (("cases", 0, "cc_archived_selected_score"), 2),
        (("cases", 0, "cc_debatable"), 0),
        (("cases", 0, "cc_source_checked_reported"), "true"),
        (("cases", 0, "domain"), ["academia"]),
        (("cases", 0, "source_question_type"), {}),
        (("cases", 0, "review_status"), "human_verified"),
        (("executions", 0, "error_code"), "arbitrary_failure_text"),
        (("executions", 0, "worker_exit_code"), True),
        (("executions", 0, "elapsed_s"), float("nan")),
        (("executions", 0, "elapsed_s"), float("inf")),
        (("executions", 0, "elapsed_s"), 10**400),
        (("totals", "case_count"), "125"),
        (("totals", "cc_archived_first_correct"), True),
        (("totals", "cc_debatable_count"), 8.0),
    )
    for location, value in mutations:
        document = _public_evaluation()
        record = document
        for key in location[:-1]:
            record = record[key]
        record[location[-1]] = value
        assert _public_evaluation_errors(document), location


def test_public_evaluation_rejects_duplicate_and_broken_cross_record_identities() -> None:
    for table in ("runs", "cases", "executions"):
        document = _public_evaluation()
        document[table].append(document[table][0])
        assert _public_evaluation_errors(document), table
    mutations = (
        (("executions", 0, "case_id"), "docbench:99999999:99999"),
        (("cases", 0, "selected_run_ordinal"), 3),
        (("cases", 0, "question_type"), "unanswerable"),
        (("cases", 0, "original_judge_score"), 0),
        (("cases", 0, "retry_merged_judge_score"), 0),
        (("runs", 1, "phase"), "execution_retry"),
        (("runs", 1, "source_sha256"), "0" * 64),
        (("runs", 0, "correct_count"), 125),
        (("runs", 0, "elapsed_s"), 1),
        (("totals", "codex_archived_selected_correct"), 125),
        (("totals", "cc_archived_first_correct"), 101),
        (("totals", "cc_archived_selected_correct"), 106),
        (("totals", "cc_debatable_count"), 9),
        (("totals", "cc_source_checked_reported_count"), 14),
        (("totals", "execution_count"), 125),
    )
    for location, value in mutations:
        document = _public_evaluation()
        record = document
        for key in location[:-1]:
            record = record[key]
        record[location[-1]] = value
        assert _public_evaluation_errors(document), location
    document = _public_evaluation()
    document["executions"].pop(0)
    assert _public_evaluation_errors(document)


def _pre_cc_evaluation() -> dict[str, object]:
    document = _public_evaluation()
    document["cc_review_status"] = "pending"
    del document["cc_review"]
    for case in document["cases"]:
        for key in (
            "cc_archived_first_score", "cc_archived_selected_score",
            "cc_debatable", "cc_source_checked_reported",
        ):
            del case[key]
    for key in (
        "cc_archived_first_correct", "cc_archived_selected_correct",
        "cc_debatable_count", "cc_source_checked_reported_count",
    ):
        del document["totals"][key]
    return document


def _pre_cc_evaluation_bytes() -> bytes:
    return (json.dumps(_pre_cc_evaluation(), ensure_ascii=False, indent=2) + "\n").encode()


def test_cc_review_projection_preserves_the_entire_preexisting_summary() -> None:
    document = _pre_cc_evaluation()
    # Pin the canonical, payload-free pre-CC projection, not private source content.
    digest = hashlib.sha256(
        json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert digest == "9832f5a675e8001f054e82ba017602840de38311128cc617d86d0e1917a0ec59"
    assert hashlib.sha256(_pre_cc_evaluation_bytes()).hexdigest() == (
        "2e25675eb999d397e2e7db6795a3b9036e25b1e0b2c0d3e7cf91df01ffecba18"
    )


def test_only_exact_reviewed_summary_bytes_and_path_are_accepted_as_history() -> None:
    content = _pre_cc_evaluation_bytes()
    assert eval_summary_violations(PUBLIC_EVALUATION, content)
    assert eval_summary_violations(PUBLIC_EVALUATION, content, committed_history=True)
    assert not eval_summary_violations(HISTORICAL_EVALUATION_PATH, content, committed_history=True)
    assert eval_summary_violations(
        "evals/docbench/results/other.summary.json", content, committed_history=True,
    )
    assert eval_summary_violations(HISTORICAL_EVALUATION_PATH, content + b" ", committed_history=True)
    mutated = _pre_cc_evaluation()
    mutated["cases"][0]["original_judge_score"] = 0
    assert eval_summary_violations(
        HISTORICAL_EVALUATION_PATH, json.dumps(mutated).encode(), committed_history=True,
    )


def test_reviewed_history_still_runs_the_ordinary_sensitive_value_scan(monkeypatch) -> None:
    monkeypatch.setattr(
        "scripts.check_repository_privacy.SECRET_VALUE_PATTERNS",
        (re.compile("deepseek-chat"),),
    )
    reasons = eval_summary_violations(
        HISTORICAL_EVALUATION_PATH, _pre_cc_evaluation_bytes(), committed_history=True,
    )
    assert any("secret-like value" in violation.reason for violation in reasons)


def test_cc_review_matches_the_public_case_table_and_keeps_disagreements() -> None:
    document = _public_evaluation()
    review = (REPOSITORY_ROOT / REVIEWED_ANALYSIS_FILES[1]).read_text()
    rows = re.findall(
        r"^\| (docbench:\d+:\d+) \| ([01]) \| ([01]) \| (true|false) \| (true|false) \|$",
        review, re.MULTILINE,
    )
    by_id = {
        identity: (int(first), int(selected), debatable == "true", checked == "true")
        for identity, first, selected, debatable, checked in rows
    }
    assert len(rows) == len(by_id) == 125
    assert by_id == {
        case["case_id"]: (
            case["cc_archived_first_score"], case["cc_archived_selected_score"],
            case["cc_debatable"], case["cc_source_checked_reported"],
        )
        for case in document["cases"]
    }
    assert {
        case["case_id"] for case in document["cases"]
        if case["cc_archived_selected_score"] != case["codex_archived_selected_score"]
    } == {"docbench:80:5", "docbench:178:1", "docbench:220:1"}
    retried = [case for case in document["cases"] if case["selected_run_ordinal"] != 1]
    assert len(retried) == 8
    assert sum(case["cc_archived_first_score"] for case in retried) == 0
    assert sum(case["cc_archived_selected_score"] for case in retried) == 5


def test_cc_review_rejects_missing_metadata_scores_and_flags() -> None:
    for location in (
        ("cc_review",), ("cc_review", "artifact_sha256"),
        ("cc_review", "reviewer_kind"), ("cc_review", "blind_review"),
        ("cc_review", "source_check_basis"),
        ("cases", 0, "cc_archived_first_score"),
        ("cases", 0, "cc_archived_selected_score"),
        ("cases", 0, "cc_debatable"), ("cases", 0, "cc_source_checked_reported"),
        ("totals", "cc_archived_first_correct"),
    ):
        document = _public_evaluation()
        record = document
        for key in location[:-1]:
            record = record[key]
        del record[location[-1]]
        assert _public_evaluation_errors(document), location


def test_cc_review_rejects_changed_labels_without_retries_even_with_matching_totals() -> None:
    document = _public_evaluation()
    case = next(case for case in document["cases"] if case["selected_run_ordinal"] == 1)
    case["cc_archived_selected_score"] = 1 - case["cc_archived_first_score"]
    document["totals"]["cc_archived_selected_correct"] = sum(
        case["cc_archived_selected_score"] for case in document["cases"]
    )
    assert any("unchanged execution" in reason for reason in _public_evaluation_errors(document))


def test_cc_review_rejects_case_score_and_flag_tampering() -> None:
    for field in (
        "cc_archived_first_score", "cc_archived_selected_score",
        "cc_debatable", "cc_source_checked_reported",
    ):
        document = _public_evaluation()
        case = document["cases"][0]
        case[field] = not case[field] if type(case[field]) is bool else 1 - case[field]
        assert _public_evaluation_errors(document), field


def test_public_evaluation_still_applies_payload_and_secret_value_checks() -> None:
    document = _public_evaluation()
    document["cases"][0]["answer"] = "synthetic-answer"
    reasons = _public_evaluation_errors(document)
    assert any("private evaluation field" in reason for reason in reasons)
    for value in ("/Users/" + "fixture/private-model", "sk-" + "a" * 48):
        document = _public_evaluation()
        document["main_model"] = value
        reasons = _public_evaluation_errors(document)
        assert any("filesystem value" in reason or "secret-like value" in reason for reason in reasons)


def _git(repository: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _check(repository: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(CHECKER), *arguments],
        cwd=repository,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def test_private_roots_local_config_and_data_types_are_rejected() -> None:
    assert path_violations("var/sessions/secret/session.sqlite")
    assert path_violations("data/examples/example.json")
    assert path_violations("doc/development.md")
    assert path_violations("docs/internal/architecture.md")
    assert path_violations(".personagraph/manifest.json")
    assert path_violations("附件/20260901/private.txt")
    assert path_violations("models/bge-m3/model.safetensors")
    assert path_violations("notebooks/private.ipynb")
    assert path_violations("tests/fixtures/dialogue.jsonl")
    assert path_violations("tests/fixtures/private-source.pdf")
    assert path_violations("artifacts/evaluation-results.csv")
    assert path_violations("configs/local/model_profiles.json")
    assert not path_violations("configs/local/README.md")
    assert not path_violations("evals/docbench/results/README.md")


def test_database_journals_and_credential_filename_variants_are_rejected() -> None:
    private_paths = (
        "runtime/session.sqlite-journal",
        "runtime/session.sqlite3-journal",
        "runtime/documents.db-journal",
        "runtime/documents.db3-journal",
        "config/credentials-prod.json",
        "config/secrets.local.json",
        "config/service-account-demo.json",
        "config/service_account_school.json",
        "config/client-secret-oauth.json",
        "config/client_secret_desktop.json",
        "runtime/api_secret.tmp",
        "runtime/api_secret.backup",
        "runtime/api_secret2",
        "runtime/api_secret~",
        "captures/request.har",
        "diagnostics/runtime.dump",
        "backups/session.bak",
        "logs/runtime.log",
        "diagnostics/.coverage.school",
        "diagnostics/coverage.xml",
        ".mcp.local.json",
    )

    for private_path in private_paths:
        assert path_violations(private_path), private_path

    ignored_lines = set((REPOSITORY_ROOT / ".gitignore").read_text().splitlines())
    assert "api_secret*" in ignored_lines
    assert ".mcp.local.json" in ignored_lines
    assert not path_violations("src/personagraph/api/security.py")


def test_raw_benchmark_subtrees_allow_only_explicit_public_metadata() -> None:
    private_paths = (
        "evals/docbench/results/nested/run.summary.json",
        "evals/docbench/results/raw.json",
        "evals/docbench/previous_results/raw.json",
        "evals/docbench/previous_results/nested/run.summary.json",
        "evals/docbench/sources/upstream/questions.json",
        "evals/docbench/sources/document.json",
        "evals/docbench/questions/questions.json",
        "evals/docbench/questions/nested/README.md",
    )
    public_paths = (
        "evals/docbench/results/README.md",
        "evals/docbench/results/public.summary.json",
        "evals/docbench/results/public.baseline.json",
        "evals/docbench/previous_results/README.md",
        "evals/docbench/previous_results/public.summary.json",
        "evals/docbench/sources/README.md",
        "evals/docbench/sources/upstream.manifest.json",
        "evals/docbench/questions/README.md",
    )

    for private_path in private_paths:
        assert path_violations(private_path), private_path
    for public_path in public_paths:
        assert not path_violations(public_path), public_path


def test_user_document_and_media_file_types_are_rejected_even_when_forced() -> None:
    additional_private_suffixes = {
        "odt",
        "ods",
        "odp",
        "rtf",
        "epub",
        "pages",
        "numbers",
        "keynote",
        "eml",
        "msg",
        "jpeg",
        "tif",
        "tiff",
        "heic",
        "heif",
        "m4a",
        "aac",
        "opus",
        "mov",
        "avi",
        "webm",
        "m4v",
        "wmv",
    }

    ignored_lines = set((REPOSITORY_ROOT / ".gitignore").read_text().splitlines())
    for suffix in PRODUCT_BINARY_UPLOAD_SUFFIXES | additional_private_suffixes:
        path = f"notes/private-attachment.{suffix}"
        assert f"*.{suffix}" in ignored_lines, suffix
        assert path_violations(path), path


def test_reviewed_desktop_screenshot_requires_exact_public_bytes() -> None:
    content = (REPOSITORY_ROOT / PUBLIC_SCREENSHOT).read_bytes()
    assert path_violations(PUBLIC_SCREENSHOT), "a filename alone does not approve an image"
    assert not inspect_snapshot([PUBLIC_SCREENSHOT], lambda _: content)
    for replacement in (
        content[:-1] + bytes([content[-1] ^ 1]),
        content[:-1],
        content + b"unreviewed metadata",
    ):
        violations = inspect_snapshot([PUBLIC_SCREENSHOT], lambda _: replacement)
        assert any("screenshot hash/size mismatch" in item.reason for item in violations)


def test_reviewed_desktop_screenshot_does_not_approve_other_image_paths() -> None:
    content = (REPOSITORY_ROOT / PUBLIC_SCREENSHOT).read_bytes()
    for path in (
        "assets/other.png",
        "assets/nested/entelecheia-desktop.png",
        "assets/Entelecheia-desktop.png",
        "notes/entelecheia-desktop.png",
    ):
        assert inspect_snapshot([path], lambda _: content), path


def test_reviewed_desktop_screenshot_fails_closed_when_unreadable() -> None:
    for error_type in (OSError, CheckError):
        def unavailable(_: str) -> bytes:
            raise error_type("unavailable screenshot")

        violations = inspect_snapshot([PUBLIC_SCREENSHOT], unavailable)
        assert any("screenshot could not be read" in item.reason for item in violations)


def test_reviewed_desktop_screenshot_uses_the_selected_git_snapshot(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    _git(repository, "init", "--quiet")
    (repository / ".gitignore").write_bytes((REPOSITORY_ROOT / ".gitignore").read_bytes())
    screenshot = repository / PUBLIC_SCREENSHOT
    screenshot.parent.mkdir()
    screenshot.write_bytes((REPOSITORY_ROOT / PUBLIC_SCREENSHOT).read_bytes())
    ignored = _git(repository, "check-ignore", "--no-index", "--verbose", PUBLIC_SCREENSHOT).stdout
    assert f"!/{PUBLIC_SCREENSHOT}\t{PUBLIC_SCREENSHOT}" in ignored
    assert _git(repository, "check-ignore", "--no-index", "assets/other.png").stdout
    _git(repository, "add", ".gitignore", PUBLIC_SCREENSHOT)
    _git(
        repository, "-c", "user.name=Privacy Test", "-c", "user.email=privacy@example.invalid",
        "commit", "--quiet", "-m", "reviewed screenshot",
    )
    screenshot.write_bytes(b"unreviewed replacement")
    assert _check(repository, "--worktree").returncode == 1
    assert _check(repository, "--staged").returncode == 0
    assert _check(repository, "--tree", "HEAD").returncode == 0
    _git(repository, "add", PUBLIC_SCREENSHOT)
    assert _check(repository, "--staged").returncode == 1
    assert _check(repository, "--tree", "HEAD").returncode == 0


def test_reviewed_analysis_allows_only_three_exact_markdown_paths() -> None:
    for path in REVIEWED_ANALYSIS_FILES:
        assert not path_violations(path), path
        assert not inspect_snapshot([path], lambda _: b"Reviewed aggregate methodology.")
    for path in PRIVATE_ANALYSIS_FILES:
        assert path_violations(path), path


def test_reviewed_analysis_does_not_bypass_secret_or_private_path_scanning() -> None:
    private_contents = (
        "Credential: sk-" + "a" * 48,
        "Local source: /Users/" + "publication_owner/private-result.json",
    )
    for path in REVIEWED_ANALYSIS_FILES:
        for content in private_contents:
            violations = inspect_snapshot([path], lambda _: content.encode())
            assert violations, path
            assert any("content pattern" in violation.reason for violation in violations)


def test_reviewed_analysis_gitignore_and_forced_staging_are_consistent(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    _git(repository, "init", "--quiet")
    shutil.copy2(REPOSITORY_ROOT / ".gitignore", repository / ".gitignore")
    paths = (*REVIEWED_ANALYSIS_FILES, *PRIVATE_ANALYSIS_FILES)
    for relative_path in paths:
        candidate = repository / relative_path
        candidate.parent.mkdir(parents=True, exist_ok=True)
        candidate.write_text("Synthetic test content.", encoding="utf-8")
    ignored = _git(repository, "check-ignore", "--no-index", *paths).stdout.splitlines()
    assert set(ignored) == set(PRIVATE_ANALYSIS_FILES)

    _git(repository, "add", *REVIEWED_ANALYSIS_FILES)
    assert _check(repository, "--staged").returncode == 0
    _git(repository, "add", "--force", *PRIVATE_ANALYSIS_FILES)
    result = _check(repository, "--staged")
    assert result.returncode == 1
    for path in PRIVATE_ANALYSIS_FILES:
        assert path in result.stderr


def test_eval_summary_rejects_payload_fields_paths_and_secrets() -> None:
    secret_value = "sk-proj-" + "A1b2C3d4E5f6G7h8I9j0K1l2"
    content = json.dumps(
        {
            "selection": {"question_type": "text", "question": "private"},
            "runtime": {"sessionId": "private-session"},
            "debug": {"working_directory": "/Users/example/private-project"},
            "models": {"credential": secret_value},
        }
    ).encode()

    reasons = {
        violation.reason
        for violation in eval_summary_violations(
            "evals/docbench/results/private.summary.json", content
        )
    }

    assert "private evaluation field: selection.question" in reasons
    assert "private evaluation field: runtime.sessionId" in reasons
    assert "local filesystem value at: debug.working_directory" in reasons
    assert "secret-like value at: models.credential" in reasons


def test_eval_summary_allows_aggregate_metadata() -> None:
    content = json.dumps(
        {
            "schema_version": "entelecheia-docbench-live-l1-summary-v1",
            "selection": {
                "question_type_counts": {"text": 3},
                "distinct_document_count": 3,
            },
            "retrieval": {
                "evidence_annotation_present": True,
                "mixed_path_score": 1.0,
            },
        }
    ).encode()

    assert not eval_summary_violations(
        "evals/docbench/results/public.summary.json", content
    )


def test_eval_summary_rejects_answers_unknown_sections_and_private_list_values() -> None:
    private_home = "/Users/" + "actual-person/private-eval"
    content = json.dumps(
        {
            "schema_version": "entelecheia-docbench-live-l1-summary-v1",
            "scope": {
                "answer": "private answer",
                "run_id": "private question/answer",
            },
            "aggregate": ["private question/answer"],
            "limitations": [private_home],
        }
    ).encode()

    reasons = {
        violation.reason
        for violation in eval_summary_violations(
            "evals/docbench/results/private.summary.json", content
        )
    }

    assert "private evaluation field: scope.answer" in reasons
    assert "private evaluation field: limitations" in reasons
    assert "unexpected evaluation-summary top-level field: aggregate" in reasons
    assert "local filesystem value at: limitations.0" in reasons
    assert "free-form evaluation-summary string at: scope.run_id" in reasons


def test_eval_summary_rejects_duplicate_json_keys() -> None:
    content = b'{"debug":{"note":"private","note":"apparently-safe"}}'

    violations = eval_summary_violations(
        "evals/docbench/results/duplicate.summary.json", content
    )

    assert [violation.reason for violation in violations] == [
        "evaluation summary contains a duplicate JSON key"
    ]


def test_text_secret_scan_is_high_confidence_and_env_example_is_placeholder_only() -> None:
    provider_token = "hf_" + "A1b2C3d4E5f6G7h8I9j0K1l2"
    assert text_secret_violations("src/settings.py", provider_token.encode())
    assert text_secret_violations(
        ".env.example", b"PERSONAGRAPH_API_KEY=real-school-credential"
    )
    assert not text_secret_violations(
        ".env.example",
        b"PERSONAGRAPH_API_KEY=\nTAVILY_API_KEY=<replace-me>\n",
    )
    assert not text_secret_violations(
        "package-lock.json", b'{"integrity":"sha512-long-lock-hash"}'
    )


def test_product_text_upload_types_are_all_privacy_covered() -> None:
    privacy_covered_suffixes = TEXT_SUFFIXES | frozenset(FORBIDDEN_SUFFIXES)
    assert PRODUCT_TEXT_UPLOAD_SUFFIXES <= privacy_covered_suffixes
    provider_token = "hf_" + "A1b2C3d4E5f6G7h8I9j0K1l2"

    for suffix in (".markdown", ".c", ".h", ".cpp"):
        files = {f"notes/upload{suffix}": provider_token.encode()}
        assert inspect_snapshot(files, files.__getitem__), suffix


def test_text_secret_scan_rejects_real_home_paths_but_allows_test_markers() -> None:
    real_home = "/Users/" + "actual-person/private-project"
    windows_home = "C:\\Users\\" + "actual-person\\private-project"

    assert text_secret_violations("config.yaml", real_home.encode())
    assert text_secret_violations("config.yaml", windows_home.encode())
    assert not text_secret_violations(
        "tests/fixture.py", b"/Users/example/project /home/runner/work"
    )


def test_structured_config_rejects_unprefixed_real_credentials_only() -> None:
    school_key = "campus-school-key-" + "A7b9C2d4E6f8G1h3"
    private_configs = {
        "notes/private.json": json.dumps({"provider": {"api_key": school_key}}).encode(),
        "notes/private.yaml": f"provider:\n  api_key: {school_key}\n".encode(),
        "notes/private.toml": f'api_key = "{school_key}"\n'.encode(),
        "notes/private.ini": f"[provider]\napi_key={school_key}\n".encode(),
        "notes/private.cfg": f"api_key: {school_key}\n".encode(),
        "notes/private.conf": f"api_key = {school_key}\n".encode(),
    }
    for path, content in private_configs.items():
        assert structured_privacy_violations(path, content), path

    safe_schema = json.dumps(
        {"properties": {"api_key": {"type": "string"}, "api_key_env": {"type": "string"}}}
    ).encode()
    safe_config = json.dumps(
        {
            "api_key": "<replace-me>",
            "api_key_env": "SCHOOL_API_KEY",
            "prompt_sha256": school_key,
        }
    ).encode()
    source_constant = f'API_KEY = "{school_key}"\n'.encode()

    assert not structured_privacy_violations("schemas/provider.schema.json", safe_schema)
    assert not structured_privacy_violations("config/example.json", safe_config)
    assert not structured_privacy_violations("src/provider.py", source_constant)
    assert structured_privacy_violations(
        "notes/duplicate.json",
        f'{{"api_key":"{school_key}","api_key":"<replace-me>"}}'.encode(),
    )


def test_structured_payload_detector_rejects_renamed_raw_evaluation_records() -> None:
    raw_json = json.dumps(
        {
            "cases": [
                {
                    "case_id": "docbench:1:0",
                    "question": "private question",
                    "reference_answer": "private reference",
                    "reply": "private model reply",
                }
            ]
        }
    ).encode()
    raw_yaml = b"""cases:
  - case_id: docbench:1:0
    question: private question
    reference_answer: private reference
    reply: private model reply
"""

    assert structured_privacy_violations("notes/archive.json", raw_json)
    assert structured_privacy_violations("notes/archive.yaml", raw_yaml)
    assert not structured_privacy_violations("src/example.py", raw_json)


def test_inspect_snapshot_applies_content_rules_outside_eval_directories() -> None:
    public_summary = json.dumps(
        {
            "schema_version": "entelecheia-docbench-live-l1-summary-v1",
            "scope": {"case_count": 3, "answer_score": 0.5},
            "selection": {"question_type_counts": {"text": 3}},
        }
    ).encode()
    files = {
        "notes/renamed-results.json": json.dumps(
            {
                "case_id": "docbench:1:0",
                "question": "private question",
                "reference_answer": "private reference",
                "reply": "private reply",
            }
        ).encode(),
        "notes/provider.yaml": b"api_key: campus-school-key-A7b9C2d4E6f8G1h3\n",
        "src/example.py": b"payload = {'question': 'fixture', 'reply': 'fixture'}\n",
        "evals/docbench/results/public.summary.json": public_summary,
    }

    violations = inspect_snapshot(files, files.__getitem__)
    violated_paths = {violation.path for violation in violations}

    assert "notes/renamed-results.json" in violated_paths
    assert "notes/provider.yaml" in violated_paths
    assert "src/example.py" not in violated_paths
    assert "evals/docbench/results/public.summary.json" not in violated_paths


def test_worktree_staged_and_tree_views_are_distinct(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    _git(repository, "init", "--quiet")
    private_file = repository / "data" / "old.txt"
    private_file.parent.mkdir()
    private_file.write_text("old private data", encoding="utf-8")
    _git(repository, "add", "data/old.txt")
    _git(
        repository,
        "-c",
        "user.name=Privacy Test",
        "-c",
        "user.email=privacy@example.invalid",
        "commit",
        "--quiet",
        "-m",
        "private ancestor",
    )

    private_file.unlink()
    (repository / "README.md").write_text("safe future tree", encoding="utf-8")

    assert _check(repository, "--worktree").returncode == 0
    tree_result = _check(repository, "--tree", "HEAD")
    assert tree_result.returncode == 1
    assert "data/old.txt" in tree_result.stderr

    _git(repository, "add", "--all")
    assert _check(repository, "--staged").returncode == 0


def test_cc_review_migration_keeps_tree_and_pre_push_history_checks_usable(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    _git(repository, "init", "--quiet")
    summary = repository / HISTORICAL_EVALUATION_PATH
    summary.parent.mkdir(parents=True)
    summary.write_bytes(_pre_cc_evaluation_bytes())
    _git(repository, "add", HISTORICAL_EVALUATION_PATH)
    assert _check(repository, "--worktree").returncode == 1
    assert _check(repository, "--staged").returncode == 1
    _git(
        repository, "-c", "user.name=Privacy Test", "-c",
        "user.email=privacy@example.invalid", "commit", "--quiet", "-m", "reviewed export",
    )
    assert _check(repository, "--tree", "HEAD").returncode == 0

    summary.write_bytes((REPOSITORY_ROOT / PUBLIC_EVALUATION).read_bytes())
    assert _check(repository, "--worktree").returncode == 0
    assert _check(repository, "--staged").returncode == 1
    _git(repository, "add", HISTORICAL_EVALUATION_PATH)
    assert _check(repository, "--staged").returncode == 0
    _git(
        repository, "-c", "user.name=Privacy Test", "-c",
        "user.email=privacy@example.invalid", "commit", "--quiet", "-m", "archive CC review",
    )
    assert _check(repository, "--tree", "HEAD").returncode == 0

    (repository / "scripts").mkdir()
    (repository / ".githooks").mkdir()
    shutil.copy2(CHECKER, repository / "scripts" / CHECKER.name)
    hook = repository / ".githooks" / "pre-push"
    shutil.copy2(REPOSITORY_ROOT / ".githooks" / "pre-push", hook)
    revision = _git(repository, "rev-parse", "HEAD").stdout.strip()
    update = f"refs/heads/main {revision} refs/heads/main {'0' * 40}\n"
    result = subprocess.run(
        ["/bin/sh", str(hook), "origin", "unused"], cwd=repository, input=update,
        check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    assert result.returncode == 0, result.stderr


def test_staged_view_rejects_force_added_private_artifacts(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    _git(repository, "init", "--quiet")
    shutil.copy2(REPOSITORY_ROOT / ".gitignore", repository / ".gitignore")
    private_files = {
        ".mcp.local.json": "{}",
        "api_secret2": "private",
        "doc/development.md": "synthetic local development note",
        "docs/internal/architecture.md": "synthetic private architecture note",
        "notes/private.jpg": "private image",
        "notes/private.odt": "private document",
        "notes/request.har": "private request capture",
        "evals/docbench/questions/raw.json": "{}",
        "evals/docbench/sources/raw.json": "{}",
    }
    for relative_path, content in private_files.items():
        candidate = repository / relative_path
        candidate.parent.mkdir(parents=True, exist_ok=True)
        candidate.write_text(content, encoding="utf-8")
    ignored = _git(
        repository, "check-ignore", "doc/development.md", "docs/internal/architecture.md"
    ).stdout.splitlines()
    assert set(ignored) == {"doc/development.md", "docs/internal/architecture.md"}
    _git(repository, "add", "--force", ".")

    result = _check(repository, "--staged")

    assert result.returncode == 1
    for private_path in private_files:
        assert private_path in result.stderr


def test_pre_push_checks_intermediate_commits_and_new_remote_history(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    _git(repository, "init", "--quiet")
    (repository / "README.md").write_text("safe base", encoding="utf-8")
    _git(repository, "add", "README.md")
    _git(
        repository,
        "-c",
        "user.name=Privacy Test",
        "-c",
        "user.email=privacy@example.invalid",
        "commit",
        "--quiet",
        "-m",
        "safe base",
    )
    base_revision = _git(repository, "rev-parse", "HEAD").stdout.strip()

    private_file = repository / "data" / "private.txt"
    private_file.parent.mkdir()
    private_file.write_text("private intermediate data", encoding="utf-8")
    _git(repository, "add", "data/private.txt")
    _git(
        repository,
        "-c",
        "user.name=Privacy Test",
        "-c",
        "user.email=privacy@example.invalid",
        "commit",
        "--quiet",
        "-m",
        "private intermediate",
    )
    private_file.unlink()
    _git(repository, "add", "--all")
    _git(
        repository,
        "-c",
        "user.name=Privacy Test",
        "-c",
        "user.email=privacy@example.invalid",
        "commit",
        "--quiet",
        "-m",
        "clean tip",
    )
    local_revision = _git(repository, "rev-parse", "HEAD").stdout.strip()

    (repository / "scripts").mkdir()
    (repository / ".githooks").mkdir()
    shutil.copy2(CHECKER, repository / "scripts" / CHECKER.name)
    hook = repository / ".githooks" / "pre-push"
    shutil.copy2(REPOSITORY_ROOT / ".githooks" / "pre-push", hook)

    for remote_revision in (
        base_revision,
        "0000000000000000000000000000000000000000",
    ):
        update = (
            f"refs/heads/main {local_revision} refs/heads/main {remote_revision}\n"
        )
        result = subprocess.run(
            ["/bin/sh", str(hook), "origin", "unused"],
            cwd=repository,
            input=update,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert result.returncode == 1
        assert "data/private.txt" in result.stderr


def test_pre_push_rejects_a_ref_that_does_not_resolve_to_a_commit(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    _git(repository, "init", "--quiet")
    blob = subprocess.run(
        ["git", "-C", str(repository), "hash-object", "-w", "--stdin"],
        input="private blob",
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout.strip()

    (repository / "scripts").mkdir()
    (repository / ".githooks").mkdir()
    shutil.copy2(CHECKER, repository / "scripts" / CHECKER.name)
    hook = repository / ".githooks" / "pre-push"
    shutil.copy2(REPOSITORY_ROOT / ".githooks" / "pre-push", hook)

    update = (
        f"refs/tags/private-blob {blob} refs/tags/private-blob "
        "0000000000000000000000000000000000000000\n"
    )
    result = subprocess.run(
        ["/bin/sh", str(hook), "origin", "unused"],
        cwd=repository,
        input=update,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert result.returncode == 1
    assert "does not resolve to a commit" in result.stderr
