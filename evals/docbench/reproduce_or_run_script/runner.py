"""由配置驱动、彼此隔离的 DocBench L1 执行与评分入口。

benchmark 数据、运行产物和隔离工作区存放在仓库外的 DocBench 根目录。提交到仓库的
selection manifest 只包含标识符和哈希；授权 PDF/QA 字节与生成答案绝不会复制进仓库。
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence
import uuid

import yaml

from evals.docbench.reproduce_or_run_script import (
    config as docbench_config,
    provenance,
)
from evals.docbench.reproduce_or_run_script.post_commit import (
    pending_post_commit,
    settle_post_commit,
)
from personagraph.configuration.paths import LOCAL_CONFIG_DIR


PROJECT_ROOT = Path(__file__).resolve().parents[3]
INSTALL_CONFIG = LOCAL_CONFIG_DIR / "app_config.json"
RUN_SCHEMA_VERSION = "personagraph-docbench-l1-run-v2"
RETRY_SCHEMA_VERSION = "personagraph-docbench-l1-retry-v1"
DEFAULT_PROJECTS_DIR_ENV = "PERSONAGRAPH_DEFAULT_PROJECTS_DIR"

_MAIN_ENV = {
    "provider": "PERSONAGRAPH_MODEL_PROVIDER",
    "request_dialect": "PERSONAGRAPH_REQUEST_DIALECT",
    "api_key": "PERSONAGRAPH_API_KEY",
    "base_url": "PERSONAGRAPH_BASE_URL",
    "model": "PERSONAGRAPH_MODEL",
    "max_tokens": "PERSONAGRAPH_MAX_TOKENS",
}
_VISION_ENV = {
    "provider": "PERSONAGRAPH_VISION_PROVIDER",
    "api_key": "PERSONAGRAPH_VISION_API_KEY",
    "base_url": "PERSONAGRAPH_VISION_BASE_URL",
    "model": "PERSONAGRAPH_VISION_MODEL",
}
_RETRIEVAL_ENV = {
    "profile": "PERSONAGRAPH_RETRIEVAL_PROFILE",
    "failure_policy": "PERSONAGRAPH_RETRIEVAL_FAILURE_POLICY",
    "device": "PERSONAGRAPH_RETRIEVAL_DEVICE",
    "use_fp16": "PERSONAGRAPH_RETRIEVAL_USE_FP16",
    "local_files_only": "PERSONAGRAPH_RETRIEVAL_LOCAL_FILES_ONLY",
}


class DocBenchRunnerError(RuntimeError):
    """正式运行若继续执行就会违反冻结契约。"""


def _require_live_authorization(allow_live: bool, *, action: str) -> None:
    if not allow_live:
        raise DocBenchRunnerError(
            f"DocBench {action} requires allow_live=True before provider calls"
        )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def _atomic_write_json(path: Path, payload: Any) -> None:
    _atomic_write_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
    )


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DocBenchRunnerError(f"cannot read JSON artifact {path}: {exc}") from exc


def _require_schema_version(
    payload: Mapping[str, Any],
    *,
    expected: str,
    artifact: str,
) -> None:
    actual = payload.get("schema_version")
    if actual != expected:
        raise DocBenchRunnerError(
            f"unsupported {artifact} schema_version: {actual!r}; expected {expected!r}"
        )


def _load_installation_config() -> dict[str, Any]:
    try:
        payload = json.loads(INSTALL_CONFIG.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DocBenchRunnerError(
            f"cannot resolve installation provider config {INSTALL_CONFIG}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise DocBenchRunnerError("installation provider config must be an object")
    return payload


def _active_installation_profile(kind: str) -> dict[str, Any] | None:
    """读取当前 GUI 激活的端点配置，而不改写安装级配置文件。"""

    from personagraph.model_io.endpoint_profiles import resolve_active_profile

    profile_kind = "model" if kind == "main" else "vision"
    profile = resolve_active_profile(profile_kind)
    if profile is None:
        return None
    resolved: dict[str, Any] = {
        "provider": profile.provider,
        "request_dialect": profile.request_dialect,
        "base_url": profile.base_url,
        "model": profile.model,
        "api_key": profile.api_key,
    }
    if profile.kind == "model":
        resolved["quota"] = profile.quota.to_dict()
    return resolved


def _resolve_provider(
    specification: Mapping[str, Any],
    *,
    kind: str,
    require_secret: bool,
) -> dict[str, Any]:
    if kind not in {"main", "vision"}:
        raise ValueError(f"unsupported provider kind: {kind}")
    prefix = "vision_" if kind == "vision" else ""
    source = str(specification.get("source") or "").strip()
    quota: object | None = None
    if source == "installation":
        active_profile = _active_installation_profile(kind)
        if active_profile is not None:
            resolved = {
                field: str(active_profile.get(field) or "").strip()
                for field in ("provider", "base_url", "model")
            }
            resolved["request_dialect"] = str(
                active_profile.get("request_dialect") or "auto"
            ).strip()
            secret = str(active_profile.get("api_key") or "").strip()
            quota = active_profile.get("quota")
            credential_source = "installation_profile"
        else:
            installation = _load_installation_config()
            resolved = {
                field: str(installation.get(f"{prefix}{field}") or "").strip()
                for field in ("provider", "base_url", "model")
            }
            resolved["request_dialect"] = str(
                installation.get("request_dialect") or "auto"
            ).strip()
            secret = str(installation.get(f"{prefix}api_key") or "").strip()
            credential_source = "installation"
    else:
        resolved = {
            field: str(specification.get(field) or "").strip()
            for field in ("provider", "base_url", "model")
        }
        resolved["request_dialect"] = str(
            specification.get("request_dialect") or "auto"
        ).strip()
        api_key_env = str(specification.get("api_key_env") or "").strip()
        secret = str(os.environ.get(api_key_env) or "").strip() if api_key_env else ""
        credential_source = f"env:{api_key_env}" if api_key_env else "none"
    missing = [field for field in ("provider", "base_url", "model") if not resolved[field]]
    if missing:
        raise DocBenchRunnerError(
            f"{kind} provider is missing required fields: {', '.join(missing)}"
        )
    if require_secret and not secret:
        raise DocBenchRunnerError(f"{kind} provider credential is unavailable")
    resolved["api_key"] = secret
    resolved["credential_source"] = credential_source
    if kind == "main":
        from personagraph.model_io.endpoint_profiles import (
            normalize_model_profile_quota,
        )

        resolved["quota"] = normalize_model_profile_quota(quota).to_dict()
    return resolved


def _redacted_provider(provider: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in provider.items()
        if key != "api_key"
    }


def _docbench_root(environment: Mapping[str, str]) -> Path:
    try:
        return Path(
            docbench_config.docbench_root(environment=environment)
        ).expanduser().resolve(strict=False)
    except (OSError, RuntimeError, ValueError) as exc:
        raise DocBenchRunnerError(f"DocBench root is invalid: {exc}") from exc


def _paths_overlap(first: Path, second: Path) -> bool:
    return (
        first == second
        or first.is_relative_to(second)
        or second.is_relative_to(first)
    )


def _run_workspace_root(
    run_root: Path | str,
    *,
    environment: Mapping[str, str],
) -> Path:
    """Return the single product Project allocation root owned by one run."""

    benchmark_root = _docbench_root(environment)
    resolved_run_root = Path(run_root).expanduser().resolve(strict=False)
    expected_runs_root = (benchmark_root / "runs").resolve(strict=False)
    if resolved_run_root.parent != expected_runs_root:
        raise DocBenchRunnerError(
            "DocBench run directory must be an immediate child of its runs directory"
        )
    run_id = _safe_run_id(resolved_run_root.name)
    workspace_root = (
        benchmark_root / "workspaces" / run_id
    ).resolve(strict=False)
    if _paths_overlap(workspace_root, resolved_run_root):
        raise DocBenchRunnerError(
            "DocBench workspace and run paths must not overlap"
        )

    from personagraph.configuration.paths import deny_reason

    reason = deny_reason(workspace_root)
    if reason is not None:
        raise DocBenchRunnerError(
            "DocBench run workspace is rejected by the workspace policy: "
            f"{reason}"
        )
    return workspace_root


def _run_quota_database_path(
    run_root: Path | str,
    *,
    environment: Mapping[str, str],
) -> Path:
    """返回一个 run 内由全部生成 worker 与 scorer 共用的队列数据库。"""

    resolved_run_root = Path(run_root).expanduser().resolve(strict=False)
    _run_workspace_root(resolved_run_root, environment=environment)
    return (resolved_run_root / "model_api_quota.sqlite3").resolve(strict=False)


def _bind_run_workspace(
    environment: Mapping[str, str],
    *,
    run_root: Path,
) -> dict[str, str]:
    """Freeze the run workspace before environment provenance is computed."""

    child = dict(environment)
    child[DEFAULT_PROJECTS_DIR_ENV] = str(
        _run_workspace_root(run_root, environment=child)
    )
    from personagraph.model_io.api_quota_controller import (
        API_QUOTA_DATABASE_PATH_ENVIRONMENT_VARIABLE,
    )

    child[API_QUOTA_DATABASE_PATH_ENVIRONMENT_VARIABLE] = str(
        _run_quota_database_path(run_root, environment=child)
    )
    return child


def _worker_run_root(*, state_dir: Path, environment: Mapping[str, str]) -> Path:
    benchmark_root = _docbench_root(environment)
    resolved_state = state_dir.expanduser().resolve(strict=False)
    cases_ancestors = [
        ancestor for ancestor in resolved_state.parents if ancestor.name == "cases"
    ]
    if len(cases_ancestors) != 1:
        raise DocBenchRunnerError(
            "DocBench worker state must have exactly one cases ancestor"
        )
    run_root = cases_ancestors[0].parent
    if run_root.parent != (benchmark_root / "runs").resolve(strict=False):
        raise DocBenchRunnerError(
            "DocBench worker state run is outside the benchmark runs directory"
        )
    _safe_run_id(run_root.name)
    return run_root


def _require_worker_project_root(
    *,
    state_dir: Path,
    environment: Mapping[str, str],
) -> Path:
    """Require the exact per-run Project root frozen by the parent runner."""

    raw_value = str(environment.get(DEFAULT_PROJECTS_DIR_ENV) or "").strip()
    if not raw_value:
        raise DocBenchRunnerError(
            "DocBench worker requires its Project root from the parent runner"
        )
    supplied = Path(raw_value).expanduser()
    if not supplied.is_absolute():
        raise DocBenchRunnerError(
            "DocBench worker Project root must be an absolute path"
        )
    project_root = supplied.resolve(strict=False)
    run_root = _worker_run_root(state_dir=state_dir, environment=environment)
    expected = _run_workspace_root(run_root, environment=environment)
    if project_root != expected:
        raise DocBenchRunnerError(
            "DocBench worker Project root is not the exact run workspace"
        )
    return project_root


def _session_working_dir_locator(
    session: Mapping[str, Any],
    *,
    run_workspace: Path,
    environment: Mapping[str, str],
) -> str:
    raw_working_dir = str(session.get("working_dir") or "").strip()
    if not raw_working_dir:
        raise DocBenchRunnerError("created Session has no working directory")
    supplied = Path(raw_working_dir).expanduser()
    if not supplied.is_absolute():
        raise DocBenchRunnerError(
            "created Session working directory is not absolute"
        )
    working_dir = supplied.resolve(strict=False)
    if working_dir == run_workspace or not working_dir.is_relative_to(run_workspace):
        raise DocBenchRunnerError(
            "created Session working directory is outside the run workspace"
        )
    benchmark_root = _docbench_root(environment)
    return working_dir.relative_to(benchmark_root).as_posix()


def _require_worker_namespace(
    *,
    state_dir: Path,
    environment: Mapping[str, str],
) -> Path:
    """Prove that import-time product paths match the parent-owned case scope."""

    from personagraph.configuration import paths
    from personagraph.model_io.api_quota_controller import (
        resolve_api_quota_database_path,
    )

    expected_state = state_dir.expanduser().resolve(strict=False)
    expected_local_config = (expected_state / "local_config").resolve(strict=False)
    project_root = _require_worker_project_root(
        state_dir=expected_state,
        environment=environment,
    )
    frozen_paths = {
        "PERSONAGRAPH_STATE_DIR": Path(paths.STATE_DIR).resolve(strict=False),
        "PERSONAGRAPH_LOCAL_CONFIG_DIR": Path(paths.LOCAL_CONFIG_DIR).resolve(
            strict=False
        ),
        "PERSONAGRAPH_DEFAULT_PROJECTS_DIR": Path(
            paths.DEFAULT_SESSION_PROJECTS_DIR
        ).resolve(strict=False),
    }
    expected_paths = {
        "PERSONAGRAPH_STATE_DIR": expected_state,
        "PERSONAGRAPH_LOCAL_CONFIG_DIR": expected_local_config,
        "PERSONAGRAPH_DEFAULT_PROJECTS_DIR": project_root,
    }
    stale = [
        name
        for name, expected in expected_paths.items()
        if frozen_paths[name] != expected
    ]
    if stale:
        raise DocBenchRunnerError(
            "DocBench worker namespace was not frozen before Python started: "
            + ", ".join(stale)
        )
    expected_queue = _run_quota_database_path(
        _worker_run_root(state_dir=expected_state, environment=environment),
        environment=environment,
    )
    if resolve_api_quota_database_path(environment) != expected_queue:
        raise DocBenchRunnerError(
            "DocBench worker quota queue is not owned by its run"
        )
    if (
        project_root == expected_state
        or project_root.is_relative_to(expected_state)
        or expected_state.is_relative_to(project_root)
    ):
        raise DocBenchRunnerError(
            "DocBench worker Project root and private state must not overlap"
        )
    return project_root


def _provider_environment(config: Mapping[str, Any]) -> tuple[dict[str, str], dict[str, Any]]:
    from personagraph.configuration.paths import load_dotenv
    from personagraph.model_io.api_quota_controller import (
        MODEL_API_QUOTA_ENVIRONMENT_VARIABLE,
        model_profile_quota_environment_value,
    )
    from personagraph.model_io.endpoint_profiles import normalize_model_profile_quota

    load_dotenv()
    main = _resolve_provider(config["providers"]["main"], kind="main", require_secret=True)
    vision = _resolve_provider(
        config["providers"]["vision"],
        kind="vision",
        require_secret=True,
    )
    child = dict(os.environ)
    child.pop(MODEL_API_QUOTA_ENVIRONMENT_VARIABLE, None)
    for field, environment_name in _MAIN_ENV.items():
        value = main.get(field)
        if value:
            child[environment_name] = value
    for field, environment_name in _VISION_ENV.items():
        value = vision.get(field)
        if value:
            child[environment_name] = value
    child[MODEL_API_QUOTA_ENVIRONMENT_VARIABLE] = (
        model_profile_quota_environment_value(
            normalize_model_profile_quota(main.get("quota"))
        )
    )
    for name in tuple(child):
        if name.startswith("PERSONAGRAPH_RETRIEVAL_"):
            child.pop(name, None)
    retrieval = config["retrieval"]
    for field, environment_name in _RETRIEVAL_ENV.items():
        value = retrieval[field]
        child[environment_name] = (
            str(value).lower() if isinstance(value, bool) else str(value)
        )
    child["PERSONAGRAPH_RETRIEVAL_BGE_MODEL"] = str(
        retrieval["encoder"]["model_id"]
    )
    child["PERSONAGRAPH_RETRIEVAL_BGE_REVISION"] = str(
        retrieval["encoder"]["revision"]
    )
    child["PERSONAGRAPH_RETRIEVAL_RERANKER"] = str(
        retrieval["reranker"]["mode"]
    )
    child["PERSONAGRAPH_RETRIEVAL_RERANKER_MODEL"] = str(
        retrieval["reranker"]["model_id"]
    )
    child["PERSONAGRAPH_RETRIEVAL_RERANKER_REVISION"] = str(
        retrieval["reranker"]["revision"]
    )
    child["PERSONAGRAPH_RETRIEVAL_METHODS"] = ",".join(
        str(method) for method in retrieval["required_methods"]
    )
    # L1 必须使用冻结的主端点，否则从开发者 shell 继承的 profile ID 会悄悄把本次运行
    # 路由到其他位置。
    for name in tuple(child):
        if name.startswith("PERSONAGRAPH_TIER_"):
            child.pop(name, None)
    # A run owns its Project allocation root.  Drop both the GUI default and the
    # retired eval-root override here; run/retry binds the exact root afterwards.
    child.pop(DEFAULT_PROJECTS_DIR_ENV, None)
    child.pop("PERSONAGRAPH_EVAL_WORKSPACE_DIR", None)
    return child, {
        "main": _redacted_provider(main),
        "vision": _redacted_provider(vision),
    }


def _current_source_provenance() -> dict[str, Any]:
    code_revision, worktree_dirty = provenance.read_git_provenance(PROJECT_ROOT)
    return {
        "code_revision": code_revision,
        "worktree_dirty": worktree_dirty,
        "source_sha256": provenance.compute_source_tree_sha256(PROJECT_ROOT),
    }


def _assert_frozen_provenance_matches(
    *,
    action: str,
    frozen: Mapping[str, Any],
    current: Mapping[str, Any],
    fields: Sequence[str],
) -> None:
    for field in fields:
        frozen_value = frozen.get(field)
        current_value = current.get(field)
        if frozen_value is not None and frozen_value != current_value:
            raise DocBenchRunnerError(
                f"{action} {field} differs from the frozen run"
            )


def _completed_source_provenance(
    frozen: Mapping[str, Any],
) -> dict[str, Any]:
    finished = _current_source_provenance()
    frozen_revision = frozen.get("code_revision")
    frozen_source_sha256 = frozen.get("source_sha256")
    finished_revision = finished["code_revision"]
    finished_source_sha256 = finished["source_sha256"]
    source_fingerprint_complete = all(
        isinstance(value, str) and bool(value)
        for value in (
            frozen_revision,
            frozen_source_sha256,
            finished_revision,
            finished_source_sha256,
        )
    )
    source_changed = bool(
        (
            frozen_revision is not None
            and finished_revision is not None
            and frozen_revision != finished_revision
        )
        or (
            frozen_source_sha256 is not None
            and finished_source_sha256 is not None
            and frozen_source_sha256 != finished_source_sha256
        )
    )
    return dict(frozen) | {
        "finished_code_revision": finished_revision,
        "finished_worktree_dirty": finished["worktree_dirty"],
        "finished_source_sha256": finished_source_sha256,
        "source_fingerprint_complete": source_fingerprint_complete,
        "source_changed_during_run": source_changed,
    }


def _case_value(case: Any, name: str, default: Any = None) -> Any:
    if isinstance(case, Mapping):
        return case.get(name, default)
    return getattr(case, name, default)


def _frozen_case(case: Any) -> dict[str, Any]:
    if is_dataclass(case):
        source = asdict(case)
    elif isinstance(case, Mapping):
        source = dict(case)
    else:
        source = {
            name: getattr(case, name)
            for name in dir(case)
            if not name.startswith("_") and not callable(getattr(case, name))
        }
    aliases = {
        "pdf_path": ("pdf_path", "source_path"),
        "qa_path": ("qa_path", "questions_path"),
        "question_type": ("question_type", "type"),
    }
    for target, names in aliases.items():
        if source.get(target) is None:
            for name in names:
                value = _case_value(case, name)
                if value is not None:
                    source[target] = value
                    break
    for name in ("pdf_path", "qa_path"):
        if source.get(name) is not None:
            source[name] = str(source[name])
    required = (
        "case_id",
        "doc_id",
        "question_index",
        "domain",
        "question_type",
        "pdf_path",
        "question",
        "answer",
        "evidence",
    )
    missing = [name for name in required if source.get(name) is None]
    if missing:
        raise DocBenchRunnerError(
            f"resolved selection case is missing: {', '.join(missing)}"
        )
    return {name: source[name] for name in required} | {
        "pdf_sha256": str(source.get("pdf_sha256") or ""),
        "qa_sha256": str(source.get("qa_sha256") or ""),
        "qa_path": str(source.get("qa_path") or ""),
    }


def _utc_run_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    return f"{timestamp}-{secrets.token_hex(4)}"


def _safe_run_id(value: str) -> str:
    normalized = value.strip()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", normalized):
        raise DocBenchRunnerError("run id must be a safe 1-128 character slug")
    return normalized


def _load_contracts(config_path: Path) -> tuple[Any, Any, list[dict[str, Any]]]:
    from evals.docbench.reproduce_or_run_script.config import load_docbench_config
    from evals.docbench.reproduce_or_run_script.selection import (
        load_selection_manifest,
    )

    loaded_config = load_docbench_config(config_path, project_root=PROJECT_ROOT)
    dataset = loaded_config.resolved["dataset"]
    loaded_selection = load_selection_manifest(
        dataset["selection"],
        data_root=dataset["data_root"],
    )
    cases = [_frozen_case(case) for case in loaded_selection.cases]
    if not cases:
        raise DocBenchRunnerError("selection contains no cases")
    return loaded_config, loaded_selection, cases


def _empty_case_telemetry() -> dict[str, Any]:
    return {
        "provider_response_count": 0,
        "rejected_output_count": 0,
        "terminal_model_request_failure_count": 0,
        "tool_call_count": 0,
        "input_tokens_total": 0,
        "output_tokens_total": 0,
        "cache_read_tokens_total": 0,
        "max_model_input_tokens": 0,
        "provider_response_purposes": {},
        "tool_call_purposes": {},
        "trajectory_database_count": 0,
        "trajectory_read_failure_count": 0,
        "trajectory_recording_failure_count": 0,
        "telemetry_complete": False,
    }


def _summarize_trajectory(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    """Derive the compact benchmark counters from one canonical snapshot."""

    metrics = _empty_case_telemetry()
    model_purposes: Counter[str] = Counter()
    tool_purposes: Counter[str] = Counter()
    steps = snapshot.get("steps")
    blobs = snapshot.get("blobs")
    integrity = snapshot.get("integrity")
    if (
        snapshot.get("format") != "personagraph.trajectory"
        or not isinstance(steps, list)
        or not isinstance(blobs, Mapping)
        or not isinstance(integrity, Mapping)
    ):
        return metrics
    part_count = 0
    snapshot_shape_valid = True
    for step in steps:
        if not isinstance(step, Mapping):
            snapshot_shape_valid = False
            continue
        parts = step.get("parts")
        if isinstance(parts, list):
            part_count += len(parts)
        else:
            snapshot_shape_valid = False
        kind = step.get("kind")
        purpose = str(step.get("purpose") or "unknown")
        outcome = step.get("outcome")
        if kind == "model_call":
            # MODEL_CALL also carries rejected-output diagnostics and one
            # terminal logical-request summary. Neither is another completed
            # Provider response, so usage remains limited to OK observations.
            if outcome == "rejected":
                metrics["rejected_output_count"] += 1
                continue
            if outcome == "failed":
                metrics["terminal_model_request_failure_count"] += 1
                continue
            if outcome != "ok":
                continue
            metrics["provider_response_count"] += 1
            model_purposes[purpose] += 1
            usage = step.get("metrics")
            if not isinstance(usage, Mapping):
                usage = {}
            input_tokens = int(usage.get("input_tokens") or 0)
            metrics["input_tokens_total"] += input_tokens
            metrics["output_tokens_total"] += int(
                usage.get("output_tokens") or 0
            )
            metrics["cache_read_tokens_total"] += int(
                usage.get("cache_read_tokens") or 0
            )
            metrics["max_model_input_tokens"] = max(
                metrics["max_model_input_tokens"], input_tokens
            )
        elif kind == "tool_call":
            metrics["tool_call_count"] += 1
            tool_purposes[purpose] += 1
        elif kind == "recording_failure":
            metrics["trajectory_recording_failure_count"] += 1
    metrics["provider_response_purposes"] = dict(sorted(model_purposes.items()))
    metrics["tool_call_purposes"] = dict(sorted(tool_purposes.items()))
    integrity_matches = (
        snapshot_shape_valid
        and all(isinstance(blob, Mapping) for blob in blobs.values())
        and integrity.get("step_count") == len(steps)
        and integrity.get("part_count") == part_count
        and integrity.get("blob_count") == len(blobs)
        and integrity.get("truncated_blob_count")
        == sum(
            isinstance(blob, Mapping) and blob.get("truncated") is True
            for blob in blobs.values()
        )
        and integrity.get("recording_failure_count")
        == metrics["trajectory_recording_failure_count"]
    )
    metrics["telemetry_complete"] = bool(
        integrity_matches
        and metrics["provider_response_count"] > 0
        and metrics["trajectory_recording_failure_count"] == 0
    )
    return metrics


def _materialize_case_trajectory(
    *,
    state_dir: Path,
    run_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Export one attempt's complete trajectory and derive its telemetry."""

    metrics = _empty_case_telemetry()
    databases = sorted(
        path
        for path in (state_dir / "sessions").glob("*/session.sqlite")
        if path.is_file()
    )
    metrics["trajectory_database_count"] = len(databases)
    if len(databases) != 1:
        metrics["trajectory_read_failure_count"] = 1
        return (
            {
                "status": "missing" if not databases else "ambiguous",
                "database_count": len(databases),
                "error_code": (
                    "trajectory_database_missing"
                    if not databases
                    else "trajectory_database_ambiguous"
                ),
            },
            metrics,
        )

    artifact_path = state_dir / "artifacts" / "trajectory.json"
    try:
        from personagraph.trajectory import export_trajectory

        artifact = export_trajectory(databases[0], artifact_path)
    except Exception as exc:  # noqa: BLE001 - instrumentation must preserve run errors
        metrics["trajectory_read_failure_count"] = 1
        return (
            {
                "status": "failed",
                "database_count": 1,
                "error_code": str(
                    getattr(exc, "code", None) or "trajectory_export_failed"
                ),
                "error_type": type(exc).__name__,
            },
            metrics,
        )

    metrics = _summarize_trajectory(artifact.snapshot)
    metrics["trajectory_database_count"] = 1
    integrity = artifact.snapshot.get("integrity")
    if not isinstance(integrity, Mapping):
        integrity = {}
        metrics["telemetry_complete"] = False
    reference = {
        "status": "complete" if metrics["telemetry_complete"] else "incomplete",
        "path": artifact.path.resolve().relative_to(run_root.resolve()).as_posix(),
        "sha256": artifact.sha256,
        "byte_count": artifact.byte_count,
        "database_count": 1,
        "step_count": int(integrity.get("step_count") or 0),
        "part_count": int(integrity.get("part_count") or 0),
        "blob_count": int(integrity.get("blob_count") or 0),
        "truncated_blob_count": int(
            integrity.get("truncated_blob_count") or 0
        ),
    }
    return reference, metrics


def _safe_exception(exc: BaseException) -> dict[str, Any]:
    message = str(exc).replace("\n", " ").strip()
    return {
        "exception_type": type(exc).__name__,
        "exception_code": getattr(exc, "code", None),
        "exception_message": (message or type(exc).__name__)[:2_000],
    }


def _retrieval_preflight_snapshot(
    environment: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    from evals.docbench.reproduce_or_run_script.retrieval_observability import (
        retrieval_preflight_snapshot,
    )

    return retrieval_preflight_snapshot(environment)


def _retrieval_runtime_snapshot(
    preflight: Mapping[str, Any],
    *,
    composition: Any | None = None,
) -> dict[str, Any]:
    from evals.docbench.reproduce_or_run_script.retrieval_observability import (
        retrieval_runtime_snapshot,
    )

    return retrieval_runtime_snapshot(
        preflight=preflight,
        composition=composition,
    )


def _run_worker(
    *,
    case_input: Path,
    state_dir: Path,
    output: Path,
    runtime_features: Path,
    allow_live: bool = False,
) -> int:
    _require_live_authorization(allow_live, action="worker")
    case = _read_json(case_input)
    if not isinstance(case, dict):
        raise DocBenchRunnerError("case input must be an object")
    run_workspace = _require_worker_namespace(
        state_dir=state_dir,
        environment=os.environ,
    )
    state_dir.mkdir(parents=True, exist_ok=True)
    submitted_prompt = f"{case['preamble']}{case['question']}"
    started = time.monotonic()
    retrieval_preflight = _retrieval_preflight_snapshot(os.environ)
    retrieval_snapshot = retrieval_preflight
    session_working_dir_locator: str | None = None
    item: dict[str, Any] = {
        "case_id": case.get("case_id"),
        "doc_id": case.get("doc_id"),
        "question_index": case.get("question_index"),
        "domain": case.get("domain"),
        "question_type": case.get("question_type"),
        "level": "L1",
        "reply": "",
        "question": case.get("question"),
        "reference_answer": case.get("answer"),
        "evidence": case.get("evidence"),
        "submitted_prompt": submitted_prompt,
        **pending_post_commit(None),
    }
    try:
        from personagraph.api.service import attachments, sessions
        from personagraph.configuration.features import load_features
        from personagraph.configuration.paths import load_dotenv
        from personagraph.workspace.ingestion.composition import (
            build_document_maintenance_lifecycle,
        )
        from personagraph.workspace.storage.context import current as current_project_database
        from personagraph.tools.visual.publication_recovery import build_visual_publication_recovery
        from personagraph.retrieval.operations.document_maintenance import (
            build_document_retrieval_composition,
        )
        from personagraph.configuration.app_settings import (
            active_provider,
            redacted_view,
        )
        from personagraph.session import store as session_store

        load_dotenv()
        features = load_features(str(runtime_features))
        if active_provider() == "mock":
            raise RuntimeError("formal DocBench L1 requires a real provider")
        created = sessions.create_session({
            "title": f"docbench-formal-l1-{case['case_id']}",
        })
        session_id = str(created["session"]["id"])
        # HTTP 会话摘要刻意保持最少内容，不暴露目录定位符元数据。工作线程是本地
        # 可信代码，因此直接从权威会话存储解析项目身份，而不扩大公开 API 响应。
        session_record = session_store.get_session(session_id)
        if session_record is None or not session_record.get("project_id"):
            raise DocBenchRunnerError(
                f"created Session has no Project binding: {case['case_id']}"
            )
        project_id = str(session_record["project_id"])
        session_working_dir_locator = _session_working_dir_locator(
            session_record,
            run_workspace=run_workspace,
            environment=os.environ,
        )
        with session_store.session_database_scope(session_id):
            source = Path(case["pdf_path"])
            if _sha256_file(source) != case["pdf_sha256"]:
                raise DocBenchRunnerError(
                    f"PDF drifted before execution: {case['case_id']}"
                )
            uploaded = attachments.upload_attachment(
                session_id,
                payload=source.read_bytes(),
                filename=source.name,
                declared_media_type="application/pdf",
            )["attachment"]
            retrieval_composition = build_document_retrieval_composition()
            retrieval_snapshot = _retrieval_runtime_snapshot(
                retrieval_preflight,
                composition=retrieval_composition,
            )
            project_database = current_project_database()
            if project_database is None:
                raise DocBenchRunnerError("worker lost its isolated Project binding")
            maintenance = build_document_maintenance_lifecycle(
                profile=retrieval_composition.requested_profile,
                encoder=retrieval_composition.encoder,
                reranker=retrieval_composition.reranker,
                after_pass=build_visual_publication_recovery(project_database),
            )
            maintenance.start()
            try:
                response = sessions.chat_turn(
                    {
                        "session_id": session_id,
                        "message": submitted_prompt,
                        "attachment_ids": [uploaded["attachment_id"]],
                        "client_request_id": f"docbench-formal-l1-{uuid.uuid4().hex}",
                        "config": str(runtime_features),
                        "runtime_policy": {
                            "schema_version": 1,
                            "l1_enabled": True,
                            "l2_enabled": False,
                        },
                    }
                )
                result = dict(response["result"])
                reply = str(result.get("reply") or "")
                item.update({
                    "session_id": session_id,
                    "project_id": project_id,
                    "session_working_dir_locator": session_working_dir_locator,
                    "turn_id": result.get("turn_id"),
                    "answer_elapsed_s": round(time.monotonic() - started, 3),
                    "status": result.get("status"),
                    "processing_level": result.get("processing_level"),
                    "reply": reply,
                    "end_reason": result.get("end_reason"),
                    "error_code": result.get("error_code"),
                    "user_interaction_mode": features.get("user_interaction_mode"),
                    "pending_user_question_count": len(
                        session_store.list_pending_user_questions(session_id=session_id)
                    ),
                    "related_insession_task_ids": result.get("related_insession_task_ids"),
                    "work_run_ids": result.get("work_run_ids"),
                    "model": redacted_view(),
                    **pending_post_commit(result.get("turn_id")),
                })
                # The process may be killed during settlement. Preserve the committed
                # answer first, explicitly distinguishing it from a completed chain.
                _classify_case_completion(item)
                item["elapsed_s"] = item["answer_elapsed_s"]
                _atomic_write_json(output, item)
                item.update(settle_post_commit(
                    session_id=session_id,
                    turn_id=result.get("turn_id"),
                    store=session_store,
                ))
                _classify_case_completion(item)
                item["elapsed_s"] = round(time.monotonic() - started, 3)
                _atomic_write_json(output, item)
            finally:
                maintenance.stop(timeout_seconds=60.0)
            retrieval_snapshot = _retrieval_runtime_snapshot(
                retrieval_preflight,
                composition=retrieval_composition,
            )
    except Exception as exc:
        item.update(_safe_exception(exc))
    item.update({
        "elapsed_s": round(time.monotonic() - started, 3),
        "retrieval": retrieval_snapshot,
        "session_working_dir_locator": session_working_dir_locator,
    })
    _classify_case_completion(item)
    _atomic_write_json(output, item)
    return 0 if item["chain_complete"] else 1


def _classify_case_completion(item: dict[str, Any]) -> None:
    item["lane_ok"] = item.get("processing_level") == "L1"
    # Entry completed means that a reply was committed, including terminal failure
    # notifications. Execution success must exclude those independently of wording.
    item["execution_ok"] = (
        item.get("status") == "completed"
        and bool(str(item.get("reply") or "").strip())
        and not item.get("error_code")
        and item.get("end_reason") != "l1_terminal_notification"
    )
    item["interaction_ok"] = item.get("pending_user_question_count", 0) == 0
    item["chain_complete"] = (
        item["execution_ok"] and item.get("post_commit_complete") is True
        and not item.get("exception_type")
    )


def _execute_case(
    *,
    case: Mapping[str, Any],
    run_root: Path,
    runtime_features: Path,
    preamble: str,
    environment: Mapping[str, str],
    timeout_s: int,
    resume: bool,
    allow_live: bool = False,
    state_dir: Path | None = None,
    output_path: Path | None = None,
) -> dict[str, Any]:
    _require_live_authorization(allow_live, action="case execution")
    case_root = run_root / "cases" / str(case["case_id"])
    output = output_path or (case_root / "result.json")
    state_dir = state_dir or (case_root / "state")
    resolved_run_root = run_root.resolve(strict=False)
    resolved_case_root = case_root.resolve(strict=False)
    resolved_state_dir = state_dir.resolve(strict=False)
    resolved_output = output.resolve(strict=False)
    if (
        resolved_state_dir == resolved_case_root
        or not resolved_state_dir.is_relative_to(resolved_case_root)
    ):
        raise DocBenchRunnerError(
            "case state directory must stay inside its own case directory"
        )
    if (
        resolved_output == resolved_case_root
        or not resolved_output.is_relative_to(resolved_case_root)
    ):
        raise DocBenchRunnerError(
            "case output must stay inside its own case directory"
        )
    expected_workspace = _run_workspace_root(
        resolved_run_root,
        environment=environment,
    )
    frozen_workspace = str(environment.get(DEFAULT_PROJECTS_DIR_ENV) or "").strip()
    if (
        not frozen_workspace
        or not Path(frozen_workspace).expanduser().is_absolute()
        or Path(frozen_workspace).expanduser().resolve(strict=False)
        != expected_workspace
    ):
        raise DocBenchRunnerError(
            "case execution requires the exact run workspace before launch"
        )
    if resume and output.is_file():
        existing = _read_json(output)
        if isinstance(existing, dict) and existing.get("case_id") == case["case_id"]:
            trajectory, telemetry = _materialize_case_trajectory(
                state_dir=resolved_state_dir,
                run_root=resolved_run_root,
            )
            resumed = existing | {
                "trajectory": trajectory,
                "telemetry": telemetry,
                "resumed": True,
            }
            _atomic_write_json(output, resumed)
            return resumed
    case_input = dict(case) | {"preamble": preamble}
    _atomic_write_json(case_root / "case_input.json", case_input)
    command = [
        sys.executable,
        "-m",
        "evals.docbench.reproduce_or_run_script.runner",
        "worker",
        "--allow-live",
        "--case-input",
        str((case_root / "case_input.json").resolve()),
        "--state-dir",
        str(state_dir.resolve()),
        "--output",
        str(resolved_output),
        "--runtime-features",
        str(runtime_features.resolve()),
    ]
    worker_environment = dict(environment)
    # 在 Python 导入 worker 模块前设置 namespace。worker 会防御性地再次赋值，但如果
    # 将来某个依赖更早导入 personagraph，导入期路径常量此时就必须已指向用例状态。
    worker_environment["PERSONAGRAPH_STATE_DIR"] = str(resolved_state_dir)
    # Provider 端点与非密钥 quota 已由父进程环境冻结；隔离本地配置目录可防止评测
    # worker 再读取 GUI 的 tier/profile 指针或写回用户设置。worker 只消费父进程投影。
    worker_environment["PERSONAGRAPH_LOCAL_CONFIG_DIR"] = str(
        (resolved_state_dir / "local_config").resolve()
    )
    started = time.monotonic()
    try:
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            env=worker_environment,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        if output.is_file():
            item = _read_json(output)
        else:
            item = {
                "case_id": case["case_id"],
                "doc_id": case["doc_id"],
                "question_index": case["question_index"],
                "domain": case["domain"],
                "question_type": case["question_type"],
                "reply": "",
                "execution_ok": False,
                "exception_type": "WorkerProcessError",
                "exception_code": "worker_result_missing",
                "exception_message": completed.stderr[-2_000:],
            }
        item["worker_exit_code"] = completed.returncode
        if completed.returncode != 0:
            # A checkpoint may precede a crash during worker shutdown. Preserve its
            # observed answer/settlement, but never label an abnormal exit complete.
            item["chain_complete"] = False
        if completed.stderr.strip():
            item["worker_stderr_tail"] = completed.stderr[-2_000:]
    except subprocess.TimeoutExpired:
        checkpoint = _read_json(output) if output.is_file() else None
        item = dict(checkpoint) if (
            isinstance(checkpoint, dict) and checkpoint.get("case_id") == case["case_id"]
        ) else {
            "case_id": case["case_id"],
            "doc_id": case["doc_id"],
            "question_index": case["question_index"],
            "domain": case["domain"],
            "question_type": case["question_type"],
            "reply": "",
            "execution_ok": False,
            **pending_post_commit(None),
        }
        item.update({
            "elapsed_s": round(time.monotonic() - started, 3),
            "exception_type": "TimeoutExpired",
            "exception_code": "worker_timeout",
            "exception_message": f"case exceeded {timeout_s}s",
            "chain_complete": False,
        })
        if item.get("post_commit_complete") is not True:
            item["post_commit_complete"] = False
            item["post_commit_elapsed_s"] = None
            item["post_commit"] = {
                **(item.get("post_commit") or {}),
                "timed_out": True,
                "reason_code": "worker_timeout_before_settlement_observed",
            }
    trajectory, telemetry = _materialize_case_trajectory(
        state_dir=resolved_state_dir,
        run_root=resolved_run_root,
    )
    item["trajectory"] = trajectory
    item["telemetry"] = telemetry
    _atomic_write_json(output, item)
    return item


def _report(
    run_root: Path,
    manifest: Mapping[str, Any],
    *,
    run_provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    ordered_cases: list[dict[str, Any]] = []
    for case in manifest["frozen_cases"]:
        result_path = run_root / "cases" / case["case_id"] / "result.json"
        if result_path.is_file():
            result = _read_json(result_path)
            if isinstance(result, dict):
                ordered_cases.append(result)
                continue
        ordered_cases.append(
            {
                "case_id": case["case_id"],
                "doc_id": case["doc_id"],
                "domain": case["domain"],
                "question_type": case["question_type"],
                "execution_ok": False,
                "exception_code": "case_not_attempted",
            }
        )
    attempted = sum(item.get("exception_code") != "case_not_attempted" for item in ordered_cases)
    complete = attempted == len(ordered_cases)
    execution_ok = sum(
        item.get("execution_ok") is True for item in ordered_cases
    )
    post_commit_complete = sum(
        item.get("post_commit_complete") is True for item in ordered_cases
    )
    chain_complete = sum(
        item.get("chain_complete") is True for item in ordered_cases
    )
    lane_ok = sum(item.get("lane_ok") is True for item in ordered_cases)
    interaction_ok = sum(
        item.get("interaction_ok") is True for item in ordered_cases
    )
    telemetry_complete = sum(
        (item.get("telemetry") or {}).get("telemetry_complete") is True
        for item in ordered_cases
    )
    effective_provenance: Mapping[str, Any] | None = run_provenance
    existing_report_path = run_root / "generation_report.json"
    if effective_provenance is None and existing_report_path.is_file():
        existing_report = _read_json(existing_report_path)
        existing_provenance = (
            existing_report.get("provenance")
            if isinstance(existing_report, Mapping)
            else None
        )
        if isinstance(existing_provenance, Mapping):
            effective_provenance = existing_provenance
    if effective_provenance is None:
        manifest_provenance = manifest.get("provenance")
        if isinstance(manifest_provenance, Mapping):
            effective_provenance = manifest_provenance
    source_changed = bool(
        effective_provenance
        and effective_provenance.get("source_changed_during_run") is True
    )
    gate_checks = {
        "attempts_complete": complete,
        "execution_ok": execution_ok == len(ordered_cases),
        "post_commit_complete": post_commit_complete == len(ordered_cases),
        "chain_complete": chain_complete == len(ordered_cases),
        "lane_ok": lane_ok == len(ordered_cases),
        "interaction_ok": interaction_ok == len(ordered_cases),
        "telemetry_complete": telemetry_complete == len(ordered_cases),
        "source_stable": not source_changed,
    }
    gate_passed = all(gate_checks.values())
    baseline_eligible = bool(
        gate_passed
        and effective_provenance
        and effective_provenance.get("code_revision")
        and effective_provenance.get("source_sha256")
        and effective_provenance.get("environment_sha256")
        and effective_provenance.get("initial_run_provenance_complete") is True
        and effective_provenance.get("source_fingerprint_complete") is True
        and effective_provenance.get("worktree_dirty") is False
        and effective_provenance.get("finished_worktree_dirty") is False
    )
    retrieval_manifest = manifest.get("retrieval")
    retrieval_cases = [
        {
            "case_id": item.get("case_id"),
            "snapshot": item["retrieval"],
        }
        for item in ordered_cases
        if isinstance(item.get("retrieval"), Mapping)
    ]
    report = {
        "schema_version": RUN_SCHEMA_VERSION,
        "benchmark_id": "docbench",
        "lane": "L1",
        "run_id": manifest["run_id"],
        "status": (
            "invalidated"
            if source_changed
            else ("complete" if complete else "partial")
        ),
        "gate_passed": gate_passed,
        "generation_gate": gate_checks,
        "baseline_eligible": baseline_eligible,
        "config_sha256": manifest["config_sha256"],
        "selection_sha256": manifest["selection_sha256"],
        "retrieval": {
            "configured": (
                retrieval_manifest.get("configured")
                if isinstance(retrieval_manifest, Mapping)
                else None
            ),
            "preflight": (
                retrieval_manifest.get("preflight")
                if isinstance(retrieval_manifest, Mapping)
                else None
            ),
            "case_snapshots": retrieval_cases,
        },
        "case_count": len(ordered_cases),
        "attempted_case_count": attempted,
        "execution_ok_case_count": execution_ok,
        "post_commit_complete_case_count": post_commit_complete,
        "chain_complete_case_count": chain_complete,
        "post_commit_status_counts": dict(Counter(
            str((item.get("post_commit") or {}).get("status") or "unobserved")
            for item in ordered_cases
        )),
        "lane_ok_case_count": lane_ok,
        "interaction_ok_case_count": interaction_ok,
        "telemetry_complete_case_count": telemetry_complete,
        "totals": {
            "answer_elapsed_s": round(sum(
                float(item.get("answer_elapsed_s") or 0) for item in ordered_cases
            ), 3),
            "post_commit_elapsed_s": round(sum(
                float(item.get("post_commit_elapsed_s") or 0) for item in ordered_cases
            ), 3),
            "post_commit_elapsed_unknown_case_count": sum(
                item.get("post_commit_elapsed_s") is None for item in ordered_cases
            ),
            "elapsed_s": round(
                sum(float(item.get("elapsed_s") or 0) for item in ordered_cases), 3
            ),
            "provider_responses": sum(
                int(
                    (item.get("telemetry") or {}).get("provider_response_count")
                    or 0
                )
                for item in ordered_cases
            ),
            "rejected_outputs": sum(
                int((item.get("telemetry") or {}).get("rejected_output_count") or 0)
                for item in ordered_cases
            ),
            "terminal_model_request_failures": sum(
                int(
                    (item.get("telemetry") or {}).get(
                        "terminal_model_request_failure_count"
                    )
                    or 0
                )
                for item in ordered_cases
            ),
            "trajectory_read_failures": sum(
                int(
                    (item.get("telemetry") or {}).get(
                        "trajectory_read_failure_count"
                    )
                    or 0
                )
                for item in ordered_cases
            ),
            "trajectory_recording_failures": sum(
                int(
                    (item.get("telemetry") or {}).get(
                        "trajectory_recording_failure_count"
                    )
                    or 0
                )
                for item in ordered_cases
            ),
            "tool_calls": sum(
                int((item.get("telemetry") or {}).get("tool_call_count") or 0)
                for item in ordered_cases
            ),
            "input_tokens": sum(
                int((item.get("telemetry") or {}).get("input_tokens_total") or 0)
                for item in ordered_cases
            ),
            "output_tokens": sum(
                int((item.get("telemetry") or {}).get("output_tokens_total") or 0)
                for item in ordered_cases
            ),
        },
        "cases": ordered_cases,
    }
    if effective_provenance is not None:
        report["provenance"] = dict(effective_provenance)
    _atomic_write_json(run_root / "generation_report.json", report)
    return report


def run_from_config(
    config_path: Path | str,
    *,
    run_id: str | None = None,
    resume: bool = False,
    allow_live: bool = False,
    require_clean: bool = False,
) -> dict[str, Any]:
    _require_live_authorization(allow_live, action="run")
    current_source = _current_source_provenance()
    if require_clean and (
        current_source["code_revision"] is None
        or current_source["worktree_dirty"] is not False
    ):
        raise DocBenchRunnerError(
            "an official DocBench run requires a clean Git worktree and known revision"
        )
    loaded_config, loaded_selection, cases = _load_contracts(Path(config_path))
    config = loaded_config.resolved
    output_root = Path(config["run"]["output_root"])
    resolved_run_id = _safe_run_id(run_id) if run_id else _utc_run_id()
    run_root = output_root / resolved_run_id
    manifest_path = run_root / "run_manifest.json"

    if run_root.exists() and not resume:
        raise DocBenchRunnerError(f"run directory already exists: {run_root}")
    if run_root.exists() and resume and not manifest_path.is_file():
        raise DocBenchRunnerError(
            f"cannot resume run directory without manifest: {run_root}"
        )
    environment, providers = _provider_environment(config)
    environment = _bind_run_workspace(environment, run_root=run_root)
    current_provenance = current_source | {
        "environment_sha256": provenance.compute_environment_sha256(environment),
        "initial_run_provenance_complete": True,
    }
    run_root.mkdir(parents=True, exist_ok=True)
    if manifest_path.is_file():
        manifest = _read_json(manifest_path)
        if not isinstance(manifest, dict):
            raise DocBenchRunnerError("existing run manifest is invalid")
        _require_schema_version(
            manifest,
            expected=RUN_SCHEMA_VERSION,
            artifact="run manifest",
        )
        if manifest.get("config_sha256") != loaded_config.sha256:
            raise DocBenchRunnerError("resume config differs from the frozen run config")
        if manifest.get("selection_sha256") != loaded_selection.sha256:
            raise DocBenchRunnerError("resume selection differs from the frozen selection")
        existing_report_path = run_root / "generation_report.json"
        if existing_report_path.is_file():
            existing_report = _read_json(existing_report_path)
            if not isinstance(existing_report, Mapping):
                raise DocBenchRunnerError("existing generation report is invalid")
            _require_schema_version(
                existing_report,
                expected=RUN_SCHEMA_VERSION,
                artifact="generation report",
            )
            if existing_report.get("config_sha256") != loaded_config.sha256:
                raise DocBenchRunnerError(
                    "resume generation report config differs from the frozen run"
                )
            if existing_report.get("selection_sha256") != loaded_selection.sha256:
                raise DocBenchRunnerError(
                    "resume generation report selection differs from the frozen run"
                )
        frozen_provenance = manifest.get("provenance")
        if isinstance(frozen_provenance, Mapping):
            _assert_frozen_provenance_matches(
                action="resume",
                frozen=frozen_provenance,
                current=current_provenance,
                fields=("source_sha256", "code_revision", "environment_sha256"),
            )
    else:
        manifest = {
            "schema_version": RUN_SCHEMA_VERSION,
            "benchmark_id": "docbench",
            "lane": "L1",
            "run_id": resolved_run_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "config_source": str(loaded_config.source_path),
            "config_sha256": loaded_config.sha256,
            "selection_sha256": loaded_selection.sha256,
            "providers": providers,
            "provenance": current_provenance,
            "retrieval": {
                "configured": loaded_config.canonical_snapshot["retrieval"],
                "preflight": _retrieval_preflight_snapshot(environment),
                "runtime_observation": (
                    "generation_report.json#/retrieval/case_snapshots"
                ),
            },
            "frozen_cases_sha256": _canonical_sha256(cases),
            "frozen_cases": cases,
        }
        _atomic_write_json(manifest_path, manifest)
        _atomic_write_text(
            run_root / "config.snapshot.yaml",
            yaml.safe_dump(
                loaded_config.canonical_snapshot,
                allow_unicode=True,
                sort_keys=False,
            ),
        )
        _atomic_write_json(run_root / "selection.snapshot.json", loaded_selection.raw)

    preamble = str(config["prompt"]["preamble"])
    runtime_features = Path(config["run"]["runtime_features"])
    timeout_s = int(config["run"]["per_case_timeout_s"])
    max_workers = int(config["run"].get("max_workers", 1))
    print(
        json.dumps(
            {"run_id": resolved_run_id, "run_root": str(run_root), "case_count": len(cases)},
            ensure_ascii=False,
        ),
        flush=True,
    )

    def execute(case: Mapping[str, Any]) -> dict[str, Any]:
        print(json.dumps({"case_started": case["case_id"]}), flush=True)
        item = _execute_case(
            case=case,
            run_root=run_root,
            runtime_features=runtime_features,
            preamble=preamble,
            environment=environment,
            timeout_s=timeout_s,
            resume=resume,
            allow_live=allow_live,
        )
        print(
            json.dumps(
                {
                    "case_finished": case["case_id"],
                    "execution_ok": item.get("execution_ok"),
                    "post_commit_complete": item.get("post_commit_complete"),
                    "chain_complete": item.get("chain_complete"),
                    "status": item.get("status"),
                    "error_code": item.get("error_code") or item.get("exception_code"),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        return item

    if max_workers == 1:
        for case in cases:
            execute(case)
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(execute, case): case for case in cases}
            for future in as_completed(futures):
                future.result()
    frozen_provenance = manifest.get("provenance")
    if not isinstance(frozen_provenance, Mapping):
        frozen_provenance = current_provenance | {
            "initial_run_provenance_complete": False,
        }
    completed_provenance = _completed_source_provenance(frozen_provenance)
    report = _report(
        run_root,
        manifest,
        run_provenance=completed_provenance,
    )
    return {
        "status": report["status"],
        "run_id": resolved_run_id,
        "run_dir": str(run_root),
        "report_path": str(run_root / "generation_report.json"),
        "case_count": report["case_count"],
        "execution_ok_case_count": report["execution_ok_case_count"],
        "post_commit_complete_case_count": report["post_commit_complete_case_count"],
        "chain_complete_case_count": report["chain_complete_case_count"],
        "gate_passed": report["gate_passed"],
        "baseline_eligible": report["baseline_eligible"],
    }


def _validated_retry_run(
    *,
    run_root: Path,
    loaded_config: Any,
    loaded_selection: Any,
    cases: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, dict[str, Any]]]:
    manifest = _read_json(run_root / "run_manifest.json")
    if not isinstance(manifest, dict):
        raise DocBenchRunnerError("run manifest must be an object")
    _require_schema_version(
        manifest,
        expected=RUN_SCHEMA_VERSION,
        artifact="run manifest",
    )
    if manifest.get("config_sha256") != loaded_config.sha256:
        raise DocBenchRunnerError("retry config differs from the frozen run config")
    if manifest.get("selection_sha256") != loaded_selection.sha256:
        raise DocBenchRunnerError("retry selection differs from the frozen run selection")
    if manifest.get("frozen_cases_sha256") != _canonical_sha256(cases):
        raise DocBenchRunnerError("resolved benchmark cases drifted before retry")

    generation = _read_json(run_root / "generation_report.json")
    if not isinstance(generation, dict) or generation.get("status") != "complete":
        raise DocBenchRunnerError(
            "retry requires a complete initial generation report"
        )
    _require_schema_version(
        generation,
        expected=RUN_SCHEMA_VERSION,
        artifact="generation report",
    )
    if generation.get("config_sha256") != loaded_config.sha256:
        raise DocBenchRunnerError("generation report config differs from the run")
    if generation.get("selection_sha256") != loaded_selection.sha256:
        raise DocBenchRunnerError("generation report selection differs from the run")

    current: dict[str, dict[str, Any]] = {}
    for case in cases:
        case_id = str(case["case_id"])
        result = _read_json(run_root / "cases" / case_id / "result.json")
        if not isinstance(result, dict) or result.get("case_id") != case_id:
            raise DocBenchRunnerError(f"invalid current result for retry: {case_id}")
        current[case_id] = result
    return manifest, generation, current


def _initial_retry_history(
    *,
    run_root: Path,
    manifest: Mapping[str, Any],
    generation: Mapping[str, Any],
) -> dict[str, Any]:
    snapshot_path = run_root / "generation_report.initial.json"
    if not snapshot_path.is_file():
        _atomic_write_bytes(
            snapshot_path,
            (run_root / "generation_report.json").read_bytes(),
        )
    initial = _read_json(snapshot_path)
    if not isinstance(initial, dict) or initial.get("run_id") != manifest.get("run_id"):
        raise DocBenchRunnerError("initial generation report snapshot is invalid")
    initial_cases = initial.get("cases")
    if not isinstance(initial_cases, list):
        raise DocBenchRunnerError("initial generation report has no case list")
    initial_ok = sum(
        isinstance(item, dict) and item.get("execution_ok") is True
        for item in initial_cases
    )

    report_path = run_root / "retry_report.json"
    if report_path.is_file():
        history = _read_json(report_path)
        if not isinstance(history, dict):
            raise DocBenchRunnerError("retry report must be an object")
        expected = {
            "schema_version": RETRY_SCHEMA_VERSION,
            "run_id": manifest.get("run_id"),
            "config_sha256": manifest.get("config_sha256"),
            "selection_sha256": manifest.get("selection_sha256"),
        }
        for field, value in expected.items():
            if history.get(field) != value:
                raise DocBenchRunnerError(f"retry report {field} differs from the run")
        if not isinstance(history.get("invocations"), list):
            raise DocBenchRunnerError("retry report invocations must be a list")
        return history
    return {
        "schema_version": RETRY_SCHEMA_VERSION,
        "benchmark_id": "docbench",
        "lane": "L1",
        "run_id": manifest["run_id"],
        "config_sha256": manifest["config_sha256"],
        "selection_sha256": manifest["selection_sha256"],
        "initial_generation_report_path": "generation_report.initial.json",
        "initial_generation_report_sha256": _sha256_file(snapshot_path),
        "initial_case_count": len(initial_cases),
        "initial_execution_ok_case_count": initial_ok,
        "initial_failed_case_count": len(initial_cases) - initial_ok,
        "invocations": [],
    }


def _archive_retry_result(
    *,
    case_root: Path,
    expected_case_id: str,
) -> tuple[Path, int, str]:
    result_path = case_root / "result.json"
    result = _read_json(result_path)
    if not isinstance(result, dict) or result.get("case_id") != expected_case_id:
        raise DocBenchRunnerError(
            f"cannot archive invalid retry result: {expected_case_id}"
        )
    attempts_root = case_root / "attempts"
    indices: list[int] = []
    if attempts_root.is_dir():
        for candidate in attempts_root.iterdir():
            match = re.fullmatch(r"attempt-(\d+)\.json", candidate.name)
            if match:
                indices.append(int(match.group(1)))
    attempt_number = max(indices, default=0) + 1
    archive_path = attempts_root / f"attempt-{attempt_number:03d}.json"
    payload = result_path.read_bytes()
    _atomic_write_bytes(archive_path, payload)
    if archive_path.read_bytes() != payload:
        raise DocBenchRunnerError(f"retry archive verification failed: {archive_path}")
    return archive_path, attempt_number, hashlib.sha256(payload).hexdigest()


def _normalize_retry_error_codes(
    error_codes: Sequence[str] | None,
) -> tuple[str, ...] | None:
    if error_codes is None:
        return None
    if isinstance(error_codes, str):
        values: Sequence[object] = (error_codes,)
    else:
        values = error_codes
    if not values:
        raise DocBenchRunnerError("retry error code filter must not be empty")
    normalized: list[str] = []
    for value in values:
        if not isinstance(value, str) or not value.strip():
            raise DocBenchRunnerError(
                "every retry error code filter value must be a non-empty string"
            )
        code = value.strip()
        if code not in normalized:
            normalized.append(code)
    return tuple(normalized)


def retry_failed_from_config(
    config_path: Path | str,
    *,
    run_dir: Path | str,
    max_workers: int = 1,
    error_codes: Sequence[str] | None = None,
    allow_live: bool = False,
) -> dict[str, Any]:
    """只重试失败用例，并保留每个被取代的结果。"""

    _require_live_authorization(allow_live, action="retry")
    if max_workers < 1:
        raise DocBenchRunnerError("retry max_workers must be at least 1")
    normalized_error_codes = _normalize_retry_error_codes(error_codes)
    error_code_filter = (
        set(normalized_error_codes)
        if normalized_error_codes is not None
        else None
    )
    loaded_config, loaded_selection, cases = _load_contracts(Path(config_path))
    run_root = Path(run_dir).expanduser().resolve()
    _run_workspace_root(run_root, environment=os.environ)
    manifest, generation, current = _validated_retry_run(
        run_root=run_root,
        loaded_config=loaded_config,
        loaded_selection=loaded_selection,
        cases=cases,
    )
    current_source = _current_source_provenance()
    frozen_provenance = manifest.get("provenance")
    if isinstance(frozen_provenance, Mapping):
        _assert_frozen_provenance_matches(
            action="retry",
            frozen=frozen_provenance,
            current=current_source,
            fields=("source_sha256", "code_revision"),
        )
    all_failed_cases = [
        case
        for case in cases
        if current[str(case["case_id"])].get("execution_ok") is not True
    ]
    failed_cases = [
        case
        for case in all_failed_cases
        if error_code_filter is None
        or (
            current[str(case["case_id"])].get("error_code")
            or current[str(case["case_id"])].get("exception_code")
        )
        in error_code_filter
    ]
    environment: dict[str, str] | None = None
    retry_provenance: dict[str, Any] | None = None
    if failed_cases:
        environment, current_providers = _provider_environment(loaded_config.resolved)
        environment = _bind_run_workspace(environment, run_root=run_root)
        current_environment_sha256 = provenance.compute_environment_sha256(environment)
        if isinstance(frozen_provenance, Mapping):
            _assert_frozen_provenance_matches(
                action="retry",
                frozen=frozen_provenance,
                current={"environment_sha256": current_environment_sha256},
                fields=("environment_sha256",),
            )
        frozen_providers = manifest.get("providers")
        if (
            isinstance(frozen_providers, Mapping)
            and dict(frozen_providers) != current_providers
        ):
            raise DocBenchRunnerError(
                "retry provider identities differ from the frozen run"
            )
        retry_provenance = (
            dict(frozen_provenance)
            if isinstance(frozen_provenance, Mapping)
            else current_source
            | {
                "environment_sha256": current_environment_sha256,
                "initial_run_provenance_complete": False,
            }
        )
    history = _initial_retry_history(
        run_root=run_root,
        manifest=manifest,
        generation=generation,
    )
    before_ok = len(cases) - len(all_failed_cases)
    invocation = {
        "retry_id": f"retry-{len(history['invocations']) + 1:03d}",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "max_workers": max_workers,
        "error_code_filter": (
            list(normalized_error_codes)
            if normalized_error_codes is not None
            else None
        ),
        "selected_case_count": len(failed_cases),
        "selected_case_ids": [str(case["case_id"]) for case in failed_cases],
        "before_execution_ok_case_count": before_ok,
        "before_failed_case_count": len(all_failed_cases),
        "attempts": [],
    }
    print(
        json.dumps(
            {
                "retry_id": invocation["retry_id"],
                "run_root": str(run_root),
                "failed_case_count": len(failed_cases),
                "max_workers": max_workers,
                "error_code_filter": invocation["error_code_filter"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    attempt_records: dict[str, dict[str, Any]] = {}
    if failed_cases:
        assert environment is not None
        assert retry_provenance is not None
        preamble = str(loaded_config.resolved["prompt"]["preamble"])
        runtime_features = Path(loaded_config.resolved["run"]["runtime_features"])
        timeout_s = int(loaded_config.resolved["run"]["per_case_timeout_s"])

        def execute_retry(case: Mapping[str, Any]) -> dict[str, Any]:
            case_id = str(case["case_id"])
            case_root = run_root / "cases" / case_id
            prior = current[case_id]
            archive_path, archived_number, archived_sha256 = _archive_retry_result(
                case_root=case_root,
                expected_case_id=case_id,
            )
            retry_number = archived_number + 1
            state_dir = (
                case_root
                / "attempts"
                / f"attempt-{retry_number:03d}-state"
            )
            pending_output = case_root / f".retry-{uuid.uuid4().hex}.json"
            print(json.dumps({"retry_case_started": case_id}), flush=True)
            try:
                item = _execute_case(
                    case=case,
                    run_root=run_root,
                    runtime_features=runtime_features,
                    preamble=preamble,
                    environment=environment,
                    timeout_s=timeout_s,
                    resume=False,
                    allow_live=allow_live,
                    state_dir=state_dir,
                    output_path=pending_output,
                )
                if not isinstance(item, dict) or item.get("case_id") != case_id:
                    raise DocBenchRunnerError(
                        f"retry worker returned an invalid result: {case_id}"
                    )
                _atomic_write_json(case_root / "result.json", item)
            finally:
                pending_output.unlink(missing_ok=True)
            print(
                json.dumps(
                    {
                        "retry_case_finished": case_id,
                        "execution_ok": item.get("execution_ok"),
                        "error_code": item.get("error_code")
                        or item.get("exception_code"),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            return {
                "case_id": case_id,
                "archived_attempt_number": archived_number,
                "retry_attempt_number": retry_number,
                "archived_result_path": str(archive_path.relative_to(run_root)),
                "archived_result_sha256": archived_sha256,
                "prior_execution_ok": prior.get("execution_ok") is True,
                "prior_error_code": prior.get("error_code")
                or prior.get("exception_code"),
                "state_dir": str(state_dir.relative_to(run_root)),
                "result_path": str(
                    (case_root / "result.json").relative_to(run_root)
                ),
                "execution_ok": item.get("execution_ok") is True,
                "error_code": item.get("error_code")
                or item.get("exception_code"),
            }

        if max_workers == 1:
            for case in failed_cases:
                record = execute_retry(case)
                attempt_records[record["case_id"]] = record
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = {
                    pool.submit(execute_retry, case): str(case["case_id"])
                    for case in failed_cases
                }
                for future in as_completed(futures):
                    record = future.result()
                    attempt_records[record["case_id"]] = record

    completed_retry_provenance = (
        _completed_source_provenance(retry_provenance)
        if retry_provenance is not None
        else None
    )
    report = (
        _report(
            run_root,
            manifest,
            run_provenance=completed_retry_provenance,
        )
        if failed_cases
        else generation
    )
    ordered_attempts = [
        attempt_records[str(case["case_id"])]
        for case in failed_cases
    ]
    recovered = sum(item["execution_ok"] for item in ordered_attempts)
    remaining_failed = report["case_count"] - report["execution_ok_case_count"]
    invocation.update(
        {
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "status": report["status"],
            "attempts": ordered_attempts,
            "recovered_case_count": recovered,
            "after_execution_ok_case_count": report["execution_ok_case_count"],
            "remaining_failed_case_count": remaining_failed,
            "generation_report_sha256": _sha256_file(
                run_root / "generation_report.json"
            ),
            "provenance": completed_retry_provenance,
        }
    )
    history["invocations"].append(invocation)
    history["latest_generation_report_sha256"] = invocation[
        "generation_report_sha256"
    ]
    _atomic_write_json(run_root / "retry_report.json", history)
    return {
        "status": report["status"],
        "run_id": manifest["run_id"],
        "run_dir": str(run_root),
        "retry_report_path": str(run_root / "retry_report.json"),
        "selected_case_count": len(failed_cases),
        "error_code_filter": invocation["error_code_filter"],
        "recovered_case_count": recovered,
        "remaining_failed_case_count": remaining_failed,
        "execution_ok_case_count": report["execution_ok_case_count"],
        "gate_passed": report.get("gate_passed"),
        "baseline_eligible": report.get("baseline_eligible"),
    }


def _judge_provider(config: Mapping[str, Any]) -> dict[str, Any]:
    judge = config["scoring"]["judge"]
    if str(judge.get("source") or "") == "main":
        return _resolve_provider(config["providers"]["main"], kind="main", require_secret=True)
    # judge 兼容 OpenAI，因此共享主 provider 字段词汇，但不会暴露给被评测的 L1 运行时。
    return _resolve_provider(judge, kind="main", require_secret=True)


def _frozen_judge_provider(
    manifest: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    from personagraph.model_io.endpoint_profiles import normalize_model_profile_quota

    judge = config["scoring"]["judge"]
    if str(judge.get("source") or "") != "main":
        return _judge_provider(config)

    providers = manifest.get("providers")
    frozen = providers.get("main") if isinstance(providers, Mapping) else None
    if not isinstance(frozen, Mapping):
        raise DocBenchRunnerError(
            "run manifest is missing frozen main provider identity"
        )

    identity_fields = ("provider", "base_url", "model", "request_dialect")
    frozen_identity = {
        field: str(frozen.get(field) or "").strip() for field in identity_fields
    }
    missing = [field for field, value in frozen_identity.items() if not value]
    if missing:
        raise DocBenchRunnerError(
            "frozen main provider identity is missing required fields: "
            + ", ".join(missing)
        )

    current = _judge_provider(config)
    current_identity = {
        field: str(current.get(field) or "").strip() for field in identity_fields
    }
    drifted = [
        field
        for field in identity_fields
        if current_identity[field] != frozen_identity[field]
    ]
    if drifted:
        raise DocBenchRunnerError(
            "current judge provider identity differs from the frozen run provider "
            f"(fields: {', '.join(drifted)})"
        )

    if not isinstance(frozen.get("quota"), Mapping):
        raise DocBenchRunnerError(
            "frozen main provider identity is missing its quota policy"
        )
    frozen_quota = normalize_model_profile_quota(frozen["quota"]).to_dict()
    current_quota = normalize_model_profile_quota(current.get("quota")).to_dict()
    if current_quota != frozen_quota:
        raise DocBenchRunnerError(
            "current judge quota policy differs from the frozen run provider"
        )
    return {
        **frozen_identity,
        "api_key": current["api_key"],
        "quota": frozen_quota,
    }


def score_from_config(
    config_path: Path | str,
    *,
    run_dir: Path | str,
    resume: bool = True,
    allow_live: bool = False,
) -> dict[str, Any]:
    _require_live_authorization(allow_live, action="score")
    from evals.docbench.reproduce_or_run_script.scorer import JudgeConfig, score_run

    loaded_config, loaded_selection, cases = _load_contracts(Path(config_path))
    run_root = Path(run_dir).expanduser().resolve()
    _run_workspace_root(run_root, environment=os.environ)
    manifest = _read_json(run_root / "run_manifest.json")
    if not isinstance(manifest, dict):
        raise DocBenchRunnerError("run manifest must be an object")
    _require_schema_version(
        manifest,
        expected=RUN_SCHEMA_VERSION,
        artifact="run manifest",
    )
    if manifest.get("config_sha256") != loaded_config.sha256:
        raise DocBenchRunnerError("score config differs from the run's frozen config")
    if manifest.get("selection_sha256") != loaded_selection.sha256:
        raise DocBenchRunnerError("score selection differs from the run's frozen selection")
    if manifest.get("frozen_cases_sha256") != _canonical_sha256(cases):
        raise DocBenchRunnerError("resolved benchmark cases drifted after generation")
    generation = _read_json(run_root / "generation_report.json")
    if not isinstance(generation, dict) or generation.get("status") != "complete":
        raise DocBenchRunnerError("generation report is missing or incomplete")
    _require_schema_version(
        generation,
        expected=RUN_SCHEMA_VERSION,
        artifact="generation report",
    )

    from personagraph.configuration.paths import load_dotenv
    from personagraph.model_io.endpoint_profiles import normalize_model_profile_quota

    load_dotenv()
    provider = _frozen_judge_provider(manifest, loaded_config.resolved)
    judge = JudgeConfig(
        provider=provider["provider"],
        base_url=provider["base_url"],
        model=provider["model"],
        api_key=provider["api_key"],
        request_dialect=provider.get("request_dialect", "deepseek"),
        quota=normalize_model_profile_quota(provider.get("quota")),
        quota_database_path=_run_quota_database_path(
            run_root,
            environment=os.environ,
        ),
    )
    scoring = loaded_config.resolved["scoring"]
    summary = score_run(
        run_results=generation["cases"],
        frozen_cases=cases,
        output_dir=run_root / "scoring",
        prompt_path=Path(scoring["prompt"]),
        expected_prompt_sha256=scoring["prompt_sha256"],
        judge=judge,
        resume=resume,
        allow_live=True,
    )
    return {
        "status": summary["status"],
        "run_id": manifest["run_id"],
        "run_dir": str(run_root),
        "score_report_path": str(run_root / "scoring" / "summary.json"),
        "score": summary.get("score"),
        "correct_count": summary.get("correct_count"),
        "case_count": summary.get("case_count"),
        "scoring_protocol": "docbench_prompt_compatible",
        "official_comparable": False,
    }


def _worker_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    worker = commands.add_parser("worker")
    worker.add_argument("--allow-live", action="store_true")
    worker.add_argument("--case-input", type=Path, required=True)
    worker.add_argument("--state-dir", type=Path, required=True)
    worker.add_argument("--output", type=Path, required=True)
    worker.add_argument("--runtime-features", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _worker_parser()
    args = parser.parse_args(argv)
    if not args.allow_live:
        parser.error("--allow-live is required before provider calls")
    return _run_worker(
        case_input=args.case_input,
        state_dir=args.state_dir,
        output=args.output,
        runtime_features=args.runtime_features,
        allow_live=args.allow_live,
    )


if __name__ == "__main__":
    raise SystemExit(main())
