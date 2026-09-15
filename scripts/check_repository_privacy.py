#!/usr/bin/env python3
"""Fail closed when a Git snapshot contains private runtime or evaluation data.

The checker intentionally uses only the Python standard library so it can run
before project dependencies are installed.  It inspects one of three views:

* ``--worktree``: existing tracked files plus untracked, non-ignored files;
* ``--staged``: the index that the next commit would contain;
* ``--tree REV``: the committed tree at ``REV``.

It never opens ignored runtime state.  It scans bounded ordinary text blobs
for high-confidence secret patterns and applies stricter structural checks to
the explicitly publishable evaluation-summary JSON files.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Callable, Iterable, Sequence


FORBIDDEN_ROOTS = frozenset(
    {
        ".claude",
        ".codex",
        ".cache",
        ".entelecheia",
        ".personagraph",
        "artifacts",
        "checkpoints",
        "data",
        "doc",
        "docs",
        "models",
        "output",
        "runs",
        "tmp",
        "third_party",
        "var",
        "workspace",
        "附件",
    }
)
LOCAL_CONFIG_ROOT = ("configs", "local")
LOCAL_CONFIG_EXCEPTIONS = frozenset({("configs", "local", "README.md")})
REVIEWED_EVALUATION_MARKDOWN_PATHS = frozenset(
    {
        # Earlier standalone analysis publications retain their original policy.
        # The current experiment packages instead require complete manifests.
        ("evals", "docbench", "previous_results", "analysis", "README.md"),
        ("evals", "docbench", "previous_results", "analysis", "EVALUATION_REVIEW.md"),
        ("evals", "docbench", "previous_results", "analysis", "FAILURE_ATTRIBUTION.md"),
    }
)

# User-approved, field-cleaned execution evidence. A directory name or a newly
# authored manifest never grants approval: the reviewed manifest bytes are pinned.
# Earlier approved publications remain valid immutable snapshots in Git history.
REVIEWED_EVIDENCE_MANIFESTS = {
    "evals/docbench/previous_results/first_gate_on_125/publication_manifest.json": (
        "be81604d889d3a396d00cdd5d6def02d964d23acf5554eddc283bc9569c74a16",
        "c8e1e8a854eacda1c5a9d65f60ed108df34a4c006ac61f34281057f68372f29c",
    ),
    "evals/docbench/previous_results/gate_off_highland235b_123/publication_manifest.json":
        (
            "cce5bce79d9c5187d874911cb9fc2aee2c16d5c68ccfb85bfd4e69294f006db7",
            "3ff2d0f0675f63c6da11acaadf573ae74912217a13cc19de35fe4b0b68aa5294",
            "dc38d9575d7cc63d6379cd345e565da2e77d21a5ec197a0ea20555be28d78614",
            "f17e25b0ff0b292e3a7d7776b150eef28ec3de168c1c1df0161e541108da60fa",
            "9212c3df4e01557f082414fbe1080aa5f37fd870a9ce8bbbc764d271b07e3551",
        ),
}

# Retired package locations are accepted only when reading committed history,
# with the same exact manifest and payload checks as current publications.
REVIEWED_HISTORICAL_EVIDENCE_MANIFESTS = {
    "evals/docbench/previous_results/showcase_125/publication_manifest.json": (
        "79220bee19eae42aaed0be4f28deef4e4ec419bbd3c0ab645215c99ce7040722",
        "be81604d889d3a396d00cdd5d6def02d964d23acf5554eddc283bc9569c74a16",
    ),
}

# A screenshot is approved only after visual review, at this exact path and byte
# identity. Other images and replacements remain private, including in Git history.
REVIEWED_SCREENSHOT_FILES = {
    "assets/entelecheia-desktop.png": (
        "810e8002c7263ed40165a6fe01ba13ce5b9254f53a1fa66da237209ff54ee2ab",
        1555668,
        2141,
        1274,
    ),
}

FORBIDDEN_SUFFIXES = (
    ".arrow",
    ".aac",
    ".avif",
    ".avi",
    ".bin",
    ".bak",
    ".bmp",
    ".bz2",
    ".csv",
    ".cer",
    ".crt",
    ".db",
    ".db3",
    ".db-journal",
    ".db-shm",
    ".db-wal",
    ".db3-journal",
    ".db3-shm",
    ".db3-wal",
    ".der",
    ".doc",
    ".docx",
    ".dump",
    ".eml",
    ".epub",
    ".feather",
    ".flac",
    ".gif",
    ".ggml",
    ".gguf",
    ".gz",
    ".h5",
    ".hdf5",
    ".har",
    ".heic",
    ".heif",
    ".ipynb",
    ".jks",
    ".jsonl",
    ".joblib",
    ".kdbx",
    ".keynote",
    ".key",
    ".keystore",
    ".log",
    ".m4a",
    ".m4v",
    ".mkv",
    ".mov",
    ".mp3",
    ".mp4",
    ".msg",
    ".ndjson",
    ".npy",
    ".npz",
    ".numbers",
    ".odt",
    ".ods",
    ".odp",
    ".ogg",
    ".ole",
    ".onnx",
    ".opus",
    ".p12",
    ".pages",
    ".parquet",
    ".pdf",
    ".pem",
    ".pfx",
    ".pickle",
    ".pkl",
    ".png",
    ".ppt",
    ".pptx",
    ".ckpt",
    ".pt",
    ".pth",
    ".rar",
    ".rtf",
    ".sqlite",
    ".sqlite3",
    ".sqlite-journal",
    ".sqlite-shm",
    ".sqlite-wal",
    ".sqlite3-journal",
    ".sqlite3-shm",
    ".sqlite3-wal",
    ".safetensors",
    ".tar",
    ".tar.gz",
    ".tgz",
    ".tif",
    ".tiff",
    ".tsv",
    ".xls",
    ".xlsx",
    ".webm",
    ".webp",
    ".wav",
    ".wmv",
    ".xz",
    ".zip",
    ".zst",
    ".7z",
    ".jpg",
    ".jpeg",
)
FORBIDDEN_FILENAMES = frozenset(
    {
        ".netrc",
        ".npmrc",
        ".pypirc",
        ".mcp.local.json",
        "api_secret",
        "client_secret.json",
        "credentials.json",
        "id_dsa",
        "id_ecdsa",
        "id_ed25519",
        "id_rsa",
        "secrets.json",
        "service-account.json",
        "service_account.json",
    }
)
FORBIDDEN_CREDENTIAL_JSON_FILENAME = re.compile(
    r"^(?:credentials|secrets|service[-_]account|client[-_]secret).*\.json$"
)
FORBIDDEN_API_SECRET_FILENAME = re.compile(r"^api_secret.*$")
FORBIDDEN_DIAGNOSTIC_FILENAME = re.compile(
    r"^(?:\.coverage(?:\..*)?|coverage\.xml)$"
)

# These are payload-bearing keys, not safe aggregate descriptors.  For
# example, ``question_type`` and ``distinct_document_count`` remain allowed.
FORBIDDEN_SUMMARY_KEYS = frozenset(
    {
        "absolute_path",
        "access_key",
        "access_token",
        "analysis",
        "api_key",
        "api_token",
        "assistant_output",
        "assistant_reply",
        "answer",
        "authorization",
        "client_secret",
        "developer_prompt",
        "document",
        "document_id",
        "document_ids",
        "document_path",
        "documents",
        "description",
        "detail",
        "details",
        "evidence",
        "evidence_items",
        "evidence_snippet",
        "evidence_text",
        "error_message",
        "expected_answer",
        "failure_analysis",
        "file_path",
        "full_prompt",
        "gold_answer",
        "ground_truth_answer",
        "key",
        "local_path",
        "limitation",
        "limitations",
        "message",
        "messages",
        "model_output",
        "model_prompt",
        "model_reply",
        "generated_answer",
        "original_question",
        "note",
        "notes",
        "observed_failure_analysis",
        "password",
        "path",
        "paths",
        "private_key",
        "project_path",
        "project_root",
        "prompt",
        "prompts",
        "question",
        "question_text",
        "questions",
        "reference_answer",
        "reference_answers",
        "reference_text",
        "refresh_token",
        "relative_path",
        "reply",
        "replies",
        "response",
        "retrieved_evidence",
        "root_path",
        "secret",
        "secret_key",
        "session",
        "session_id",
        "session_ids",
        "sessions",
        "snippet",
        "snippets",
        "source_document",
        "source_evidence",
        "source_path",
        "system_answer",
        "system_prompt",
        "text_snippet",
        "token",
        "traceback",
        "stack_trace",
        "user_prompt",
        "workspace_path",
        "prediction",
        "predicted_answer",
        "judge_response_raw",
        "judge_response_text",
    }
)

PUBLIC_EVALUATION_SCHEMA = "entelecheia-docbench-public-evaluation"
# Only committed-tree scans may accept this exact, previously reviewed export.
# Current worktree/index data must satisfy the canonical CC archival contract.
REVIEWED_HISTORICAL_EVALUATION_SHA256 = {
    "evals/docbench/results/showcase_125.summary.json":
        "2e25675eb999d397e2e7db6795a3b9036e25b1e0b2c0d3e7cf91df01ffecba18",
}
PUBLIC_EVALUATION_TOP_LEVEL_KEYS = frozenset(
    {
        "schema_version", "benchmark_id", "lane", "official_comparable",
        "cc_review_status", "cc_review", "codex_review_status", "main_model", "vision_model",
        "judge_model", "review_artifact_sha256", "runs", "cases", "executions",
        "totals",
    }
)

SUMMARY_TOP_LEVEL_KEYS_BY_SCHEMA = {
    PUBLIC_EVALUATION_SCHEMA: PUBLIC_EVALUATION_TOP_LEVEL_KEYS,
    "entelecheia-docbench-live-l1-summary-v1": frozenset(
        {
            "cases",
            "generation",
            "models",
            "network",
            "quota",
            "recorded_at",
            "retrieval",
            "retries",
            "runtime",
            "schema_version",
            "scope",
            "scoring",
            "selection",
        }
    ),
    "entelecheia-docbench-local-retrieval-summary-v1": frozenset(
        {
            "cases",
            "execution",
            "generation",
            "ingest",
            "models",
            "recorded_at",
            "schema_version",
            "scope",
        }
    ),
}

STRUCTURED_CONFIG_SUFFIXES = frozenset(
    {".cfg", ".conf", ".ini", ".json", ".toml", ".yaml", ".yml"}
)
RAW_EVALUATION_SUFFIXES = frozenset({".json", ".yaml", ".yml"})
STRUCTURED_CREDENTIAL_KEYS = frozenset(
    {
        "access_key",
        "access_token",
        "api_key",
        "auth_token",
        "client_secret",
        "password",
        "private_key",
        "secret",
        "secret_key",
    }
)
RAW_EVALUATION_QUESTION_KEYS = frozenset(
    {"original_question", "question", "question_text", "user_question"}
)
RAW_EVALUATION_REFERENCE_KEYS = frozenset(
    {
        "expected_answer",
        "gold_answer",
        "ground_truth_answer",
        "reference_answer",
        "reference_answers",
    }
)
RAW_EVALUATION_OUTPUT_KEYS = frozenset(
    {
        "assistant_output",
        "assistant_reply",
        "generated_answer",
        "model_output",
        "model_reply",
        "predicted_answer",
        "prediction",
        "reply",
        "response",
        "system_answer",
    }
)
RAW_EVALUATION_CONTEXT_KEYS = frozenset(
    {
        "benchmark_id",
        "case_id",
        "doc_id",
        "judge_response_raw",
        "question_index",
        "question_type",
        "run_id",
        "score",
        "scoring_protocol",
        "submitted_prompt",
    }
)
STRUCTURED_ASSIGNMENT = re.compile(
    r"^\s*(?:-\s*)?(?P<key>[\"']?[A-Za-z_][A-Za-z0-9_.-]*[\"']?)"
    r"\s*[:=]\s*(?P<value>.*?)(?:\s+[#;].*)?$"
)
SUMMARY_SAFE_IDENTIFIER_VALUE = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._:/+\-]{0,255}$"
)

LOCAL_PATH_VALUE = re.compile(
    r"^(?:~[/\\]|[A-Za-z]:[/\\]|/(?:Users|Volumes|home|mnt|opt|private|srv|tmp|var|workspace)(?:/|$))"
)
SECRET_VALUE_PATTERNS = (
    re.compile(r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
    re.compile(
        r"\b(?:sk-(?:ant-|proj-|svcacct-)[A-Za-z0-9_-]{20,}|sk-[A-Za-z0-9]{40,})\b"
    ),
    re.compile(r"(?i)(?:api[_-]?key|access[_-]?token|password|secret)=[^&\s]{8,}"),
)
MAX_SUMMARY_BYTES = 2 * 1024 * 1024
MAX_TEXT_SCAN_BYTES = 2 * 1024 * 1024
MAX_REPORTED_VIOLATIONS = 100

TEXT_SUFFIXES = frozenset(
    {
        ".bash",
        ".c",
        ".cfg",
        ".cjs",
        ".conf",
        ".cpp",
        ".css",
        ".env",
        ".example",
        ".fish",
        ".go",
        ".gradle",
        ".html",
        ".h",
        ".ini",
        ".java",
        ".js",
        ".json",
        ".jsx",
        ".kt",
        ".lock",
        ".markdown",
        ".md",
        ".mjs",
        ".properties",
        ".ps1",
        ".py",
        ".rb",
        ".rs",
        ".sh",
        ".sql",
        ".swift",
        ".svg",
        ".toml",
        ".ts",
        ".tsx",
        ".txt",
        ".vue",
        ".xml",
        ".yaml",
        ".yml",
        ".zsh",
    }
)
TEXT_FILENAMES = frozenset(
    {
        ".dockerignore",
        ".editorconfig",
        ".env.example",
        ".gitattributes",
        ".gitignore",
        "dockerfile",
        "license",
        "makefile",
        "package-lock.json",
        "pnpm-lock.yaml",
    }
)

# Categories are deliberately narrow.  Do not add generic entropy matching:
# model revisions, content hashes, and lock files legitimately contain long
# random-looking strings.
TEXT_SECRET_PATTERNS = (
    (
        "private-key block",
        re.compile(r"-----BEGIN (?:[A-Z0-9 ]+ )?PRIVATE KEY-----"),
    ),
    ("AWS access key", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("GitHub token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b")),
    ("GitLab token", re.compile(r"\bglpat-[A-Za-z0-9_-]{20,}\b")),
    ("Google API key", re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b")),
    ("Hugging Face token", re.compile(r"\bhf_[A-Za-z0-9]{20,}\b")),
    (
        "OpenAI/Anthropic-style API token",
        re.compile(
            r"\b(?:sk-(?:ant-|proj-|svcacct-)[A-Za-z0-9_-]{20,}|sk-[A-Za-z0-9]{40,})\b"
        ),
    ),
    ("npm token", re.compile(r"\bnpm_[A-Za-z0-9]{20,}\b")),
    ("PyPI token", re.compile(r"\bpypi-[A-Za-z0-9_-]{40,}\b")),
    (
        "Slack token",
        re.compile(r"\bx(?:ox[baprs]-|app-)[A-Za-z0-9-]{20,}\b"),
    ),
    ("Stripe live secret", re.compile(r"\b[rs]k_live_[A-Za-z0-9]{16,}\b")),
    (
        "JSON Web Token",
        re.compile(
            r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b"
        ),
    ),
)
ENV_CREDENTIAL_NAME = re.compile(
    r"(?:^|_)(?:API_KEY|ACCESS_KEY|ACCESS_TOKEN|AUTH_TOKEN|CLIENT_SECRET|PRIVATE_KEY|PASSWORD|SECRET)$",
    re.IGNORECASE,
)
ENV_PLACEHOLDER_MARKERS = (
    "changeme",
    "dummy",
    "example",
    "not-set",
    "placeholder",
    "redacted",
    "replace",
    "test-only",
    "your-",
    "your_",
)
HOME_PATH_PATTERNS = (
    re.compile(r"/(?:Users|home)/([^/\\\s\"']+)(?:[/\\])"),
    re.compile(r"\b[A-Za-z]:[/\\]+Users[/\\]+([^/\\\s\"']+)(?:[/\\])"),
)
TEST_HOME_ACCOUNT_MARKERS = frozenset(
    {"ci", "dev", "example", "private", "runner", "test", "user"}
)


@dataclass(frozen=True, order=True)
class Violation:
    path: str
    reason: str


class CheckError(RuntimeError):
    """A repository snapshot could not be inspected safely."""


class DuplicateJsonKey(ValueError):
    """A JSON object repeats a key and therefore has ambiguous content."""


def _run_git(
    repo: Path,
    arguments: Sequence[str],
    *,
    input_bytes: bytes | None = None,
) -> bytes:
    process = subprocess.run(
        ["git", "-C", os.fspath(repo), *arguments],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        input=input_bytes,
        check=False,
    )
    if process.returncode:
        detail = process.stderr.decode("utf-8", "replace").strip()
        raise CheckError(detail or f"git {' '.join(arguments)} failed")
    return process.stdout


def repository_root(start: Path | None = None) -> Path:
    location = (start or Path.cwd()).resolve()
    output = _run_git(location, ("rev-parse", "--show-toplevel"))
    return Path(output.decode("utf-8", "surrogateescape").strip()).resolve()


def _decode_paths(raw: bytes) -> list[str]:
    return sorted(
        {
            item.decode("utf-8", "surrogateescape")
            for item in raw.split(b"\0")
            if item
        }
    )


def _batch_read_blobs(repo: Path, object_ids: Iterable[str]) -> dict[str, bytes]:
    requested_ids = sorted(set(object_ids))
    if not requested_ids:
        return {}
    request = b"".join(object_id.encode("ascii") + b"\n" for object_id in requested_ids)
    output = _run_git(repo, ("cat-file", "--batch"), input_bytes=request)
    stream = io.BytesIO(output)
    blobs: dict[str, bytes] = {}
    for requested_id in requested_ids:
        header = stream.readline().rstrip(b"\n").split()
        if len(header) != 3 or header[1] != b"blob":
            raise CheckError("git returned an invalid blob header")
        try:
            size = int(header[2])
        except ValueError as error:
            raise CheckError("git returned an invalid blob size") from error
        content = stream.read(size)
        delimiter = stream.read(1)
        if len(content) != size or delimiter != b"\n":
            raise CheckError("git returned a truncated blob batch")
        blobs[requested_id] = content
    if stream.read(1):
        raise CheckError("git returned unexpected trailing blob data")
    return blobs


def _reviewed_manifest_hashes(*, committed_history: bool) -> dict[str, tuple[str, ...]]:
    if committed_history:
        return REVIEWED_EVIDENCE_MANIFESTS | REVIEWED_HISTORICAL_EVIDENCE_MANIFESTS
    return REVIEWED_EVIDENCE_MANIFESTS


def _snapshot_content_cache(
    repo: Path, object_ids_by_path: dict[str, str], *, committed_history: bool = False,
) -> dict[str, bytes]:
    manifests = _reviewed_manifest_hashes(committed_history=committed_history)
    selected = {
        path: object_id
        for path, object_id in object_ids_by_path.items()
        if path in REVIEWED_SCREENSHOT_FILES or (
            (not path_violations(path) or path in manifests)
            and (_is_eval_summary(path) or _is_text_candidate(path))
        )
    }
    blobs = _batch_read_blobs(repo, selected.values())
    content = {path: blobs[object_id] for path, object_id in selected.items()}
    # Resolve reviewed manifests from this same Git snapshot before deciding
    # which payloads to preload. Never read manifests from the working directory
    # while checking a staged or historical tree.
    evidence, _ = _reviewed_evidence_entries(
        set(object_ids_by_path), content.__getitem__, committed_history=committed_history,
    )
    additional = {
        path: object_id for path, object_id in object_ids_by_path.items()
        if path in evidence and path not in content and _is_text_candidate(path)
    }
    extra_blobs = _batch_read_blobs(repo, additional.values())
    content.update({path: extra_blobs[object_id] for path, object_id in additional.items()})
    return content


def worktree_snapshot(repo: Path) -> tuple[list[str], Callable[[str], bytes]]:
    raw = _run_git(
        repo,
        ("ls-files", "--cached", "--others", "--exclude-standard", "-z"),
    )
    paths = [
        path
        for path in _decode_paths(raw)
        if os.path.lexists(os.fspath(repo / path))
    ]

    def read(path: str) -> bytes:
        candidate = repo / path
        if candidate.is_symlink():
            return os.fsencode(os.readlink(candidate))
        return candidate.read_bytes()

    return paths, read


def staged_snapshot(repo: Path) -> tuple[list[str], Callable[[str], bytes]]:
    raw = _run_git(repo, ("ls-files", "--cached", "--stage", "-z"))
    object_ids_by_path: dict[str, str] = {}
    paths: set[str] = set()
    for record in raw.split(b"\0"):
        if not record:
            continue
        try:
            metadata, encoded_path = record.split(b"\t", 1)
            _mode, object_id, stage = metadata.split(b" ", 2)
        except ValueError as error:
            raise CheckError("git returned an invalid index record") from error
        path = encoded_path.decode("utf-8", "surrogateescape")
        paths.add(path)
        if stage == b"0":
            object_ids_by_path[path] = object_id.decode("ascii")
    content = _snapshot_content_cache(repo, object_ids_by_path)

    def read(path: str) -> bytes:
        try:
            return content[path]
        except KeyError as error:
            raise CheckError("staged blob is unavailable") from error

    return sorted(paths), read


def tree_snapshot(
    repo: Path, revision: str
) -> tuple[list[str], Callable[[str], bytes]]:
    raw = _run_git(repo, ("ls-tree", "-r", "-z", "--full-tree", revision))
    paths: list[str] = []
    object_ids_by_path: dict[str, str] = {}
    for record in raw.split(b"\0"):
        if not record:
            continue
        try:
            metadata, encoded_path = record.split(b"\t", 1)
            _mode, object_type, _object_id = metadata.split(b" ", 2)
        except ValueError as error:
            raise CheckError("git returned an invalid tree record") from error
        if object_type == b"blob":
            path = encoded_path.decode("utf-8", "surrogateescape")
            paths.append(path)
            object_ids_by_path[path] = _object_id.decode("ascii")
    paths.sort()
    content = _snapshot_content_cache(repo, object_ids_by_path, committed_history=True)

    def read(path: str) -> bytes:
        try:
            return content[path]
        except KeyError as error:
            raise CheckError("tree blob is unavailable") from error

    return paths, read


def _parts(path: str) -> tuple[str, ...]:
    return tuple(part for part in PurePosixPath(path).parts if part not in {"", "."})


def path_violations(
    path: str, *, reviewed_evidence: bool = False, reviewed_screenshot: bool = False,
) -> list[Violation]:
    parts = _parts(path)
    if not parts:
        return []

    violations: list[Violation] = []
    lowered_parts = tuple(part.casefold() for part in parts)
    if lowered_parts[0] in FORBIDDEN_ROOTS:
        violations.append(Violation(path, f"private repository root: {parts[0]}/"))

    if lowered_parts[:2] == LOCAL_CONFIG_ROOT and parts not in LOCAL_CONFIG_EXCEPTIONS:
        violations.append(Violation(path, "private local configuration (only configs/local/README.md is allowed)"))

    filename = lowered_parts[-1]
    if filename.startswith(".env") and filename != ".env.example":
        violations.append(Violation(path, "environment/secrets file"))
    elif filename.endswith(".env") and filename != ".env.example":
        violations.append(Violation(path, "environment/secrets file"))

    if (
        filename in FORBIDDEN_FILENAMES
        or FORBIDDEN_CREDENTIAL_JSON_FILENAME.fullmatch(filename)
        or FORBIDDEN_API_SECRET_FILENAME.fullmatch(filename)
    ):
        violations.append(Violation(path, "credential or private-key filename"))
    if FORBIDDEN_DIAGNOSTIC_FILENAME.fullmatch(filename):
        violations.append(Violation(path, "local diagnostic artifact filename"))
    if filename.endswith(FORBIDDEN_SUFFIXES) and not (
        reviewed_screenshot and path in REVIEWED_SCREENSHOT_FILES
    ):
        violations.append(Violation(path, "private data/database/key material file type"))

    if lowered_parts[0] == "evals":
        evals_index = 0
        for directory, reason, allowed in (
            (
                "results",
                "raw evaluation result (only direct README, *.summary.json, *.baseline.json, and explicitly reviewed Markdown paths are allowed)",
                lambda name: name == "readme.md"
                or name.endswith(".summary.json")
                or name.endswith(".baseline.json"),
            ),
            (
                "previous_results",
                "raw historical evaluation result (only direct README, *.summary.json, *.baseline.json, and explicitly reviewed Markdown paths are allowed)",
                lambda name: name == "readme.md"
                or name.endswith(".summary.json")
                or name.endswith(".baseline.json"),
            ),
            (
                "sources",
                "raw benchmark source (only a direct README or *.manifest.json file is allowed)",
                lambda name: name == "readme.md" or name.endswith(".manifest.json"),
            ),
            (
                "questions",
                "raw benchmark question data (only a direct README is allowed)",
                lambda name: name == "readme.md",
            ),
        ):
            if directory == "previous_results" and reviewed_evidence:
                # Only inspect_snapshot can establish exact manifest membership.
                continue
            if directory in {"results", "previous_results"} and parts in REVIEWED_EVALUATION_MARKDOWN_PATHS:
                # Path approval does not skip the ordinary text/secret scan.
                continue
            try:
                directory_index = lowered_parts.index(directory, evals_index + 1)
            except ValueError:
                continue
            relative_parts = lowered_parts[directory_index + 1 :]
            if len(relative_parts) != 1 or not allowed(filename):
                violations.append(Violation(path, reason))
    return violations


def _normalise_key(key: object) -> str:
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", str(key))
    return re.sub(r"[^a-z0-9]+", "_", text.casefold()).strip("_")


def _walk_json(value: object, location: tuple[str, ...] = ()) -> Iterable[tuple[tuple[str, ...], object]]:
    if isinstance(value, dict):
        for key, nested in value.items():
            key_text = str(key)
            nested_location = (*location, key_text)
            yield nested_location, nested
            yield from _walk_json(nested, nested_location)
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            nested_location = (*location, str(index))
            yield nested_location, nested
            yield from _walk_json(nested, nested_location)


def _walk_mappings(value: object) -> Iterable[dict[object, object]]:
    if isinstance(value, dict):
        yield value
        for nested in value.values():
            yield from _walk_mappings(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _walk_mappings(nested)


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    document: dict[str, object] = {}
    for key, value in pairs:
        if key in document:
            raise DuplicateJsonKey
        document[key] = value
    return document


def _is_eval_summary(path: str) -> bool:
    parts = tuple(part.casefold() for part in _parts(path))
    if not parts or parts[0] != "evals" or not {"results", "previous_results"}.intersection(parts):
        return False
    return parts[-1].endswith(".summary.json") or parts[-1].endswith(".baseline.json")


def _is_text_candidate(path: str) -> bool:
    pure_path = PurePosixPath(path)
    filename = pure_path.name.casefold()
    if filename in TEXT_FILENAMES or pure_path.suffix.casefold() in TEXT_SUFFIXES:
        return True
    parts = tuple(part.casefold() for part in pure_path.parts)
    return not pure_path.suffix and bool(parts) and parts[0] in {".githooks", "scripts"}


def _is_placeholder(value: str) -> bool:
    cleaned = value.strip().rstrip(",;").strip().strip("'\"").strip()
    lowered = cleaned.casefold()
    if not cleaned or lowered in {"none", "null"}:
        return True
    if lowered.startswith(("$", "<")):
        return True
    return any(marker in lowered for marker in ENV_PLACEHOLDER_MARKERS)


def _is_structured_placeholder(value: str) -> bool:
    if _is_placeholder(value):
        return True
    cleaned = value.strip().rstrip(",;").strip().strip("'\"").strip().casefold()
    return bool(
        re.fullmatch(
            r"(?:fake|fixture|invalid|mock|sample|test)(?:[-_].*)?", cleaned
        )
    )


def _structured_assignments(content: bytes) -> list[tuple[str, str]]:
    text = content.decode("utf-8", "replace")
    assignments: list[tuple[str, str]] = []
    for line in text.splitlines():
        candidate = line.strip()
        if not candidate or candidate.startswith(("#", ";", "[")):
            continue
        match = STRUCTURED_ASSIGNMENT.match(line)
        if match is None:
            continue
        raw_key = match.group("key").strip("'\"")
        key = _normalise_key(raw_key.rsplit(".", 1)[-1])
        value = match.group("value").strip()
        assignments.append((key, value))
    return assignments


def _mapping_payload_keys(
    mapping: dict[object, object], candidates: frozenset[str]
) -> set[str]:
    return {
        key
        for raw_key, value in mapping.items()
        if (key := _normalise_key(raw_key)) in candidates
        and not isinstance(value, dict)
    }


def _looks_like_raw_evaluation_mapping(mapping: dict[object, object]) -> bool:
    question_keys = _mapping_payload_keys(mapping, RAW_EVALUATION_QUESTION_KEYS)
    reference_keys = _mapping_payload_keys(mapping, RAW_EVALUATION_REFERENCE_KEYS)
    output_keys = _mapping_payload_keys(mapping, RAW_EVALUATION_OUTPUT_KEYS)
    context_keys = _mapping_payload_keys(mapping, RAW_EVALUATION_CONTEXT_KEYS)
    answer_present = bool(_mapping_payload_keys(mapping, frozenset({"answer"})))

    if question_keys and (reference_keys or output_keys):
        return True
    if reference_keys and output_keys:
        return True
    if question_keys and answer_present and context_keys:
        return True
    return bool(question_keys and len(context_keys) >= 2)


def structured_privacy_violations(
    path: str, content: bytes, *, max_bytes: int = MAX_TEXT_SCAN_BYTES,
) -> list[Violation]:
    """Detect credentials and raw per-case eval payloads after path renaming."""

    suffix = PurePosixPath(path).suffix.casefold()
    if suffix not in STRUCTURED_CONFIG_SUFFIXES or len(content) > max_bytes:
        return []

    document: object | None = None
    if suffix == ".json":
        try:
            document = json.loads(content, object_pairs_hook=_unique_json_object)
        except DuplicateJsonKey:
            return [Violation(path, "structured JSON contains a duplicate key")]
        except (UnicodeDecodeError, json.JSONDecodeError):
            document = None

    violations: list[Violation] = []
    if document is not None:
        for location, value in _walk_json(document):
            key = _normalise_key(location[-1])
            if (
                key in STRUCTURED_CREDENTIAL_KEYS
                and isinstance(value, str)
                and not _is_structured_placeholder(value)
            ):
                violations.append(
                    Violation(
                        path,
                        f"non-placeholder structured credential at: {'.'.join(location)}",
                    )
                )
        if suffix in RAW_EVALUATION_SUFFIXES and any(
            _looks_like_raw_evaluation_mapping(mapping)
            for mapping in _walk_mappings(document)
        ):
            violations.append(
                Violation(path, "raw per-case evaluation payload fields")
            )
        return violations

    assignments = _structured_assignments(content)
    for key, value in assignments:
        if (
            key in STRUCTURED_CREDENTIAL_KEYS
            and value
            and not value.startswith(("{", "["))
            and not _is_structured_placeholder(value)
        ):
            violations.append(
                Violation(path, f"non-placeholder structured credential at: {key}")
            )

    if suffix in RAW_EVALUATION_SUFFIXES:
        scalar_keys = {
            key
            for key, value in assignments
            if value and not value.startswith(("{", "["))
        }
        question_keys = scalar_keys & RAW_EVALUATION_QUESTION_KEYS
        reference_keys = scalar_keys & RAW_EVALUATION_REFERENCE_KEYS
        output_keys = scalar_keys & RAW_EVALUATION_OUTPUT_KEYS
        context_keys = scalar_keys & RAW_EVALUATION_CONTEXT_KEYS
        answer_present = "answer" in scalar_keys
        if (
            (question_keys and (reference_keys or output_keys))
            or (reference_keys and output_keys)
            or (question_keys and answer_present and context_keys)
            or (question_keys and len(context_keys) >= 2)
        ):
            violations.append(
                Violation(path, "raw per-case evaluation payload fields")
            )
    return violations


def text_secret_violations(
    path: str, content: bytes, *, max_bytes: int = MAX_TEXT_SCAN_BYTES,
) -> list[Violation]:
    if len(content) > max_bytes:
        return [
            Violation(
                path,
                f"text file exceeds the {max_bytes}-byte privacy scan limit",
            )
        ]
    text = content.decode("utf-8", "replace")
    violations = [
        Violation(path, f"secret content pattern: {category}")
        for category, pattern in TEXT_SECRET_PATTERNS
        if pattern.search(text)
    ]
    if any(
        match.group(1).casefold() not in TEST_HOME_ACCOUNT_MARKERS
        for pattern in HOME_PATH_PATTERNS
        for match in pattern.finditer(text)
    ):
        violations.append(Violation(path, "local content pattern: user home path"))

    if PurePosixPath(path).name.casefold() == ".env.example":
        for line in text.splitlines():
            candidate = line.strip()
            if not candidate or candidate.startswith("#"):
                continue
            if candidate.startswith("export "):
                candidate = candidate[7:].lstrip()
            if "=" not in candidate:
                continue
            name, value = candidate.split("=", 1)
            if ENV_CREDENTIAL_NAME.search(name.strip()) and not _is_placeholder(value):
                violations.append(
                    Violation(
                        path,
                        "secret content pattern: non-placeholder .env.example credential",
                    )
                )
                break
    return violations


def _public_evaluation_violations(path: str, document: dict[str, object]) -> list[Violation]:
    """Validate the closed, payload-free public projection and its score accounting.

    These records identify benchmark cases, never users or runtime sessions.
    This validates archival consistency, not the correctness of a judge's verdict.
    """
    violations: list[Violation] = []

    def reject(location: str, reason: str) -> None:
        violations.append(Violation(path, f"public evaluation {location}: {reason}"))

    def record(value: object, fields: dict[str, Callable[[object], bool]], location: str) -> None:
        if not isinstance(value, dict):
            reject(location, "must be an object")
            return
        if set(value) != set(fields):
            reject(location, "missing or unexpected fields")
        for key, check in fields.items():
            if key in value and not check(value[key]):
                reject(f"{location}.{key}", "invalid type or value")

    def integer(value: object) -> bool:
        return type(value) is int and 0 <= value <= 10**15

    def score(value: object) -> bool:
        return type(value) is int and value in (0, 1)

    def elapsed(value: object) -> bool:
        return type(value) in (int, float) and 0 <= value <= 10**12 and math.isfinite(value)

    def sha256(value: object) -> bool:
        return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None

    def identifier(value: object) -> bool:
        return isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", value) is not None

    def case_id(value: object) -> bool:
        return isinstance(value, str) and re.fullmatch(r"docbench:[0-9]{1,8}:[0-9]{1,5}", value) is not None

    def run_ordinal(value: object) -> bool:
        return type(value) is int and value in (1, 2, 3)

    def records(value: object) -> bool:
        return isinstance(value, list) and 1 <= len(value) <= 10000

    record(document, {
        "schema_version": lambda v: v == PUBLIC_EVALUATION_SCHEMA,
        "benchmark_id": lambda v: v == "docbench",
        "lane": lambda v: v == "L1",
        "official_comparable": lambda v: v is False,
        "cc_review_status": lambda v: v == "archived_provisional",
        "cc_review": lambda v: isinstance(v, dict),
        "codex_review_status": lambda v: v == "archived_provisional",
        "main_model": identifier, "vision_model": identifier, "judge_model": identifier,
        "review_artifact_sha256": sha256,
        "runs": records, "cases": records, "executions": records,
        "totals": lambda v: isinstance(v, dict),
    }, "root")
    if violations:
        return violations

    record(document["cc_review"], {
        "artifact_sha256": sha256,
        "reviewer_kind": lambda v: v == "model",
        "blind_review": lambda v: v is False,
        "source_check_basis": lambda v: v == "reviewer_self_report",
    }, "cc_review")

    run_fields = {
        "ordinal": run_ordinal,
        "phase": lambda v: v in ("first_pass", "transport_retry", "execution_retry"),
        "run_id": identifier,
        "created_at": lambda v: isinstance(v, str) and re.fullmatch(
            r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?\+00:00", v
        ) is not None,
        "code_revision": lambda v: isinstance(v, str) and re.fullmatch(r"[0-9a-f]{40}", v) is not None,
        "worktree_dirty": lambda v: type(v) is bool,
        "elapsed_s": elapsed,
    }
    for key in (
        "config_sha256", "selection_sha256", "frozen_cases_sha256", "source_sha256",
        "environment_sha256", "manifest_file_sha256", "score_file_sha256",
        "generation_file_sha256", "judge_prompt_sha256",
    ):
        run_fields[key] = sha256
    for key in (
        "case_count", "correct_count", "provider_responses", "rejected_outputs",
        "terminal_model_request_failures", "trajectory_read_failures",
        "trajectory_recording_failures", "tool_calls", "input_tokens", "output_tokens",
    ):
        run_fields[key] = integer

    source_types = {
        "text-only": "text", "multimodal-t": "multimodal", "multimodal-f": "multimodal",
        "meta-data": "metadata", "unanswerable": "unanswerable", "una-web": "unanswerable",
    }
    case_fields = {
        "case_id": case_id,
        "domain": lambda v: v in ("academia", "finance", "government", "laws", "news"),
        "question_type": lambda v: v in ("text", "multimodal", "metadata", "unanswerable"),
        "source_question_type": lambda v: isinstance(v, str) and v in source_types,
        "pdf_sha256": sha256, "qa_sha256": sha256,
        "selected_run_ordinal": run_ordinal,
        "original_judge_score": score, "retry_merged_judge_score": score,
        "codex_archived_first_score": score, "codex_archived_selected_score": score,
        "cc_archived_first_score": score, "cc_archived_selected_score": score,
        "cc_debatable": lambda v: type(v) is bool,
        "cc_source_checked_reported": lambda v: type(v) is bool,
        "review_status": lambda v: v in (
            "original_review_retained", "unresolved_retained_original",
            "new_answer_checked", "confirmed_correction",
        ),
    }
    execution_fields = {
        "run_ordinal": run_ordinal, "case_id": case_id, "judge_score": score,
        "execution_ok": lambda v: type(v) is bool,
        "worker_exit_code": lambda v: type(v) is int and -255 <= v <= 255,
        "elapsed_s": elapsed,
        "error_code": lambda v: v in (
            None, "VERIFICATION_FAILED", "TOOL_COMPLETION_UNCONFIRMED", "MODEL_TRANSPORT_FAILURE",
        ),
    }
    score_totals = {
        "original_judge_correct": "original_judge_score",
        "retry_merged_judge_correct": "retry_merged_judge_score",
        "codex_archived_first_correct": "codex_archived_first_score",
        "codex_archived_selected_correct": "codex_archived_selected_score",
        "cc_archived_first_correct": "cc_archived_first_score",
        "cc_archived_selected_correct": "cc_archived_selected_score",
    }
    flag_totals = {
        "cc_debatable_count": "cc_debatable",
        "cc_source_checked_reported_count": "cc_source_checked_reported",
    }
    record(document["totals"], {
        key: integer for key in ("case_count", "execution_count", *score_totals, *flag_totals)
    }, "totals")
    for table, fields in (("runs", run_fields), ("cases", case_fields), ("executions", execution_fields)):
        for index, row in enumerate(document[table]):
            record(row, fields, f"{table}.{index}")
    if violations:
        return violations

    runs = {row["ordinal"]: row for row in document["runs"]}
    cases = {row["case_id"]: row for row in document["cases"]}
    executions = {(row["run_ordinal"], row["case_id"]): row for row in document["executions"]}
    if len(runs) != len(document["runs"]) or set(runs) != {1, 2, 3}:
        reject("runs", "must contain the three distinct first-pass and retry runs")
    if len(cases) != len(document["cases"]):
        reject("cases", "duplicate case identity")
    if len(executions) != len(document["executions"]):
        reject("executions", "duplicate run/case identity")
    if any(run not in runs or case not in cases for run, case in executions):
        reject("executions", "unknown run or case identity")
    if violations:
        return violations

    for ordinal, run in runs.items():
        if run["phase"] != ("first_pass", "transport_retry", "execution_retry")[ordinal - 1]:
            reject(f"runs.{ordinal}", "phase does not match run ordinal")
        selected = [row for (number, _), row in executions.items() if number == ordinal]
        if run["case_count"] != len(selected) or run["correct_count"] != sum(row["judge_score"] for row in selected):
            reject(f"runs.{ordinal}", "case count or judge-score sum mismatch")
        if not math.isclose(run["elapsed_s"], sum(row["elapsed_s"] for row in selected), abs_tol=0.01):
            reject(f"runs.{ordinal}", "elapsed time must be summed case time, not wall time")
        if run["source_sha256"] != runs[1]["source_sha256"]:
            reject(f"runs.{ordinal}", "retry source fingerprint differs from first pass")

    for identity, case in cases.items():
        first = executions.get((1, identity))
        selected = executions.get((case["selected_run_ordinal"], identity))
        available = [ordinal for ordinal in runs if (ordinal, identity) in executions]
        if first is None or selected is None or case["selected_run_ordinal"] != max(available, default=0):
            reject(f"cases.{identity}", "missing first/selected execution or selected run is not the latest")
            continue
        if case["original_judge_score"] != first["judge_score"] or case["retry_merged_judge_score"] != selected["judge_score"]:
            reject(f"cases.{identity}", "judge scores differ from the referenced executions")
        if case["question_type"] != source_types[case["source_question_type"]]:
            reject(f"cases.{identity}", "source question type does not match its reporting group")
        if case["selected_run_ordinal"] == 1 and case["cc_archived_first_score"] != case["cc_archived_selected_score"]:
            reject(f"cases.{identity}", "CC scores differ for the unchanged execution")

    totals = document["totals"]
    if totals["case_count"] != len(cases) or totals["execution_count"] != len(executions):
        reject("totals", "case or execution count mismatch")
    for total_key, case_key in score_totals.items():
        if totals[total_key] != sum(case[case_key] for case in cases.values()):
            reject(f"totals.{total_key}", "score sum mismatch")
    for total_key, case_key in flag_totals.items():
        if totals[total_key] != sum(case[case_key] for case in cases.values()):
            reject(f"totals.{total_key}", "review declaration count mismatch")
    return violations


def eval_summary_violations(
    path: str, content: bytes, *, committed_history: bool = False,
) -> list[Violation]:
    if len(content) > MAX_SUMMARY_BYTES:
        return [Violation(path, f"evaluation summary exceeds {MAX_SUMMARY_BYTES} bytes")]
    try:
        document = json.loads(content, object_pairs_hook=_unique_json_object)
    except DuplicateJsonKey:
        return [Violation(path, "evaluation summary contains a duplicate JSON key")]
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        if isinstance(error, json.JSONDecodeError):
            reason = f"invalid evaluation-summary JSON (line {error.lineno}, column {error.colno})"
        else:
            reason = "evaluation summary is not valid UTF-8 JSON"
        return [Violation(path, reason)]
    if not isinstance(document, dict):
        return [Violation(path, "evaluation summary root must be a JSON object")]

    violations: list[Violation] = []
    schema_version = document.get("schema_version")
    reviewed_history = (
        committed_history
        and REVIEWED_HISTORICAL_EVALUATION_SHA256.get(path) == hashlib.sha256(content).hexdigest()
    )
    if schema_version == PUBLIC_EVALUATION_SCHEMA and not reviewed_history:
        violations.extend(_public_evaluation_violations(path, document))
    # Exact historical approval replaces only current structural validation;
    # the ordinary field, private-path, free-form and secret checks still run.
    allowed_top_level_keys = (
        SUMMARY_TOP_LEVEL_KEYS_BY_SCHEMA.get(schema_version)
        if isinstance(schema_version, str)
        else None
    )
    if allowed_top_level_keys is None:
        violations.append(
            Violation(path, "unsupported or missing evaluation-summary schema_version")
        )
    else:
        for unexpected_key in sorted(set(document) - allowed_top_level_keys):
            violations.append(
                Violation(
                    path,
                    f"unexpected evaluation-summary top-level field: {unexpected_key}",
                )
            )
    for location, value in _walk_json(document):
        key = _normalise_key(location[-1])
        json_location = ".".join(location)
        if key in FORBIDDEN_SUMMARY_KEYS:
            violations.append(
                Violation(path, f"private evaluation field: {json_location}")
            )
        if isinstance(value, str):
            if LOCAL_PATH_VALUE.search(value):
                violations.append(
                    Violation(path, f"local filesystem value at: {json_location}")
                )
            elif any(pattern.search(value) for pattern in SECRET_VALUE_PATTERNS):
                violations.append(
                    Violation(path, f"secret-like value at: {json_location}")
                )
            elif not SUMMARY_SAFE_IDENTIFIER_VALUE.fullmatch(value):
                violations.append(
                    Violation(
                        path,
                        f"free-form evaluation-summary string at: {json_location}",
                    )
                )
    return violations


def _reviewed_evidence_entries(
    paths: set[str], read: Callable[[str], bytes], *, committed_history: bool = False,
) -> tuple[dict[str, tuple[str, int]], list[Violation]]:
    entries: dict[str, tuple[str, int]] = {}
    violations = []
    for path, approved_hashes in _reviewed_manifest_hashes(committed_history=committed_history).items():
        if path not in paths:
            continue
        try:
            raw = read(path)
            expected = hashlib.sha256(raw).hexdigest()
            if expected not in approved_hashes:
                raise ValueError("manifest is not the reviewed snapshot")
            manifest = json.loads(raw, object_pairs_hook=_unique_json_object)
            if manifest["kind"] != "reviewed_public_evidence":
                raise ValueError("unexpected publication kind")
            root = str(PurePosixPath(path).parent)
            entries[path] = (expected, len(raw))
            for relative, record in manifest["files"].items():
                relative_path = PurePosixPath(relative)
                if relative_path.is_absolute() or ".." in relative_path.parts:
                    raise ValueError("unsafe publication path")
                target = f"{root}/{relative}"
                if target not in paths:
                    violations.append(Violation(target, "reviewed evidence file is missing"))
                entries[target] = (record["sha256"], record["bytes"])
        except (OSError, CheckError, ValueError, KeyError, TypeError) as error:
            violations.append(Violation(path, f"invalid reviewed evidence manifest: {error}"))
    return entries, violations


def _reviewed_screenshot_violations(path: str, content: bytes) -> list[Violation]:
    expected_hash, expected_size, width, height = REVIEWED_SCREENSHOT_FILES[path]
    if (hashlib.sha256(content).hexdigest(), len(content)) != (expected_hash, expected_size):
        return [Violation(path, "reviewed screenshot hash/size mismatch")]
    if (
        content[:16] != b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
        or int.from_bytes(content[16:20], "big") != width
        or int.from_bytes(content[20:24], "big") != height
    ):
        return [Violation(path, "reviewed screenshot PNG header/dimensions mismatch")]
    return []


def inspect_snapshot(
    paths: Iterable[str], read: Callable[[str], bytes], *, committed_history: bool = False,
) -> list[Violation]:
    paths = list(paths)
    evidence, manifest_violations = _reviewed_evidence_entries(
        set(paths), read, committed_history=committed_history,
    )
    violations: set[Violation] = set(manifest_violations)
    for path in paths:
        expected = evidence.get(path)
        reviewed_screenshot = False
        if path in REVIEWED_SCREENSHOT_FILES:
            try:
                screenshot_errors = _reviewed_screenshot_violations(path, read(path))
            except (OSError, CheckError):
                violations.add(Violation(path, "reviewed screenshot could not be read"))
            else:
                violations.update(screenshot_errors)
                reviewed_screenshot = not screenshot_errors
        current_path_violations = path_violations(
            path, reviewed_evidence=expected is not None, reviewed_screenshot=reviewed_screenshot,
        )
        violations.update(current_path_violations)
        is_summary = _is_eval_summary(path)
        is_text = _is_text_candidate(path)
        if not current_path_violations and (is_summary or is_text):
            try:
                content = read(path)
            except (OSError, CheckError):
                violations.add(Violation(path, "privacy-scanned file could not be read"))
            else:
                scan_limit = MAX_TEXT_SCAN_BYTES
                if expected is not None:
                    actual = (hashlib.sha256(content).hexdigest(), len(content))
                    if actual != expected:
                        violations.add(Violation(path, "reviewed evidence hash/size mismatch"))
                        continue
                    # Large approved trajectories still receive a complete scan;
                    # this is not a generic size-limit exemption for raw artifacts.
                    scan_limit = max(scan_limit, expected[1])
                if is_text:
                    violations.update(text_secret_violations(path, content, max_bytes=scan_limit))
                    violations.update(structured_privacy_violations(path, content, max_bytes=scan_limit))
                if is_summary:
                    violations.update(eval_summary_violations(
                        path, content, committed_history=committed_history,
                    ))
    return sorted(violations)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--worktree", action="store_true", help="check the current non-ignored worktree (default)")
    modes.add_argument("--staged", action="store_true", help="check the Git index")
    modes.add_argument("--tree", metavar="REV", help="check a committed Git tree")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        repo = repository_root()
        if arguments.staged:
            label = "staged"
            paths, read = staged_snapshot(repo)
        elif arguments.tree is not None:
            label = f"tree {arguments.tree}"
            paths, read = tree_snapshot(repo, arguments.tree)
        else:
            label = "worktree"
            paths, read = worktree_snapshot(repo)
        violations = inspect_snapshot(paths, read, committed_history=arguments.tree is not None)
    except CheckError as error:
        print(f"repository privacy check could not run: {error}", file=sys.stderr)
        return 2

    if violations:
        print(f"repository privacy check failed ({label}):", file=sys.stderr)
        for violation in violations[:MAX_REPORTED_VIOLATIONS]:
            displayed_path = json.dumps(violation.path, ensure_ascii=True)
            displayed_reason = json.dumps(violation.reason, ensure_ascii=True)
            print(f"  - {displayed_path}: {displayed_reason}", file=sys.stderr)
        hidden_count = len(violations) - MAX_REPORTED_VIOLATIONS
        if hidden_count > 0:
            print(f"  ... and {hidden_count} more violation(s)", file=sys.stderr)
        return 1
    print(f"repository privacy check passed ({label}; {len(paths)} files checked)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
