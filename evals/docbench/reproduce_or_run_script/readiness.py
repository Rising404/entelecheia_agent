"""DocBench L1 本地材料与运行依赖的严格只读就绪检查。

本模块只读取配置、冻结选题、Source 字节、安装级 Provider 配置与本地检索资产。
它不创建 run、Project 或 Session，不调用 Provider，也不下载模型或数据。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
from pathlib import Path
from typing import Any

from evals.docbench.reproduce_or_run_script.config import (
    PROJECT_ROOT,
    load_docbench_config,
)
from evals.docbench.reproduce_or_run_script.retrieval_observability import (
    retrieval_preflight_snapshot,
)
from evals.docbench.reproduce_or_run_script.selection import (
    load_selection_manifest,
)
from personagraph.configuration.features import load_features


READINESS_SCHEMA_VERSION = "personagraph-docbench-readiness-v1"

_REQUIRED_RUNTIME_FEATURES: dict[str, object] = {
    "l1_semantic_verification_mode": "always",
    "user_interaction_mode": "closed_world",
    "l1_external_web_tools_enabled": False,
    "file_retrieval_write_enabled": True,
    "file_retrieval_read_enabled": True,
    "l1_retrieval_tools_enabled": True,
    "l2_aux_retrieval_tools_enabled": False,
    "l2_task_retrieval_tools_enabled": False,
}


class DocBenchReadinessError(RuntimeError):
    """一项本地就绪条件失败，并携带可公开的稳定原因码。"""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


def _sha256_file(path: Path, *, label: str) -> str:
    if not path.is_file():
        raise DocBenchReadinessError(
            f"{label}_missing",
            f"{label} is missing or is not a regular file",
        )
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise DocBenchReadinessError(
            f"{label}_unreadable",
            f"{label} cannot be read: {type(exc).__name__}",
        ) from exc
    return digest.hexdigest()


def _failure(exc: BaseException) -> dict[str, object]:
    code = getattr(exc, "code", None)
    if not isinstance(code, str) or not code:
        code = type(exc).__name__
    message = str(exc).replace("\n", " ").strip()
    return {
        "status": "failed",
        "error_code": code,
        "error_type": type(exc).__name__,
        "message": (message or type(exc).__name__)[:1_000],
    }


def _public_provider(
    provider: Mapping[str, object],
    *,
    credential_present: bool,
) -> dict[str, object]:
    """只投影可公开的端点身份与限流配置，不返回凭据或其派生身份。"""

    public: dict[str, object] = {
        "provider": str(provider.get("provider") or ""),
        "base_url": str(provider.get("base_url") or ""),
        "model": str(provider.get("model") or ""),
        "request_dialect": str(provider.get("request_dialect") or "auto"),
        "credential_present": credential_present,
    }
    quota = provider.get("quota")
    if isinstance(quota, Mapping):
        numeric_fields = (
            "requests_per_minute",
            "tokens_per_minute",
            "tokens_per_week",
            "max_in_flight",
        )
        public_quota = {
            name: quota.get(name)
            for name in (*numeric_fields, "quota_group")
        }
        public["quota"] = public_quota
        public["quota_enabled"] = any(
            public_quota[name] is not None for name in numeric_fields
        )
    return public


def _resolve_provider_context(
    config: Mapping[str, Any],
) -> tuple[dict[str, str], dict[str, object]]:
    """按正式 runner 的只读解析语义检查端点，但不执行 Provider 请求。"""

    # 延迟导入避免让纯配置模块依赖运行编排；这里只复用正式运行的唯一端点解析语义。
    from evals.docbench.reproduce_or_run_script import runner

    environment, redacted = runner._provider_environment(config)
    judge = runner._judge_provider(config)
    main = redacted.get("main")
    vision = redacted.get("vision")
    if not isinstance(main, Mapping) or not isinstance(vision, Mapping):
        raise DocBenchReadinessError(
            "provider_projection_invalid",
            "formal runner returned an invalid provider projection",
        )
    judge_source = (
        "main"
        if str(config["scoring"]["judge"].get("source") or "") == "main"
        else "dedicated"
    )
    return environment, {
        "main": _public_provider(main, credential_present=True),
        "vision": _public_provider(vision, credential_present=True),
        "judge": _public_provider(
            judge,
            credential_present=bool(str(judge.get("api_key") or "").strip()),
        )
        | {"source": judge_source},
    }


def _dataset_check(config: Any) -> dict[str, object]:
    dataset = config.resolved["dataset"]
    selection = load_selection_manifest(
        dataset["selection"],
        data_root=dataset["data_root"],
    )
    document_ids = {int(case.doc_id) for case in selection.cases}
    qa_files = {str(case.qa_path.resolve()) for case in selection.cases}
    pdf_files = {str(case.pdf_path.resolve()) for case in selection.cases}
    return {
        "status": "ready",
        "selection_sha256": selection.sha256,
        "case_count": len(selection.cases),
        "document_count": len(document_ids),
        "pdf_file_count": len(pdf_files),
        "qa_file_count": len(qa_files),
        "source_hashes_verified": True,
    }


def _runtime_features_check(config: Any) -> dict[str, object]:
    path = Path(config.resolved["run"]["runtime_features"])
    digest = _sha256_file(path, label="runtime_features")
    features = load_features(str(path))
    drifted = {
        name: {"expected": expected, "actual": features.get(name)}
        for name, expected in _REQUIRED_RUNTIME_FEATURES.items()
        if features.get(name) != expected
    }
    if drifted:
        names = ", ".join(sorted(drifted))
        raise DocBenchReadinessError(
            "runtime_features_not_closed_world_l1",
            f"runtime feature policy differs from DocBench L1: {names}",
        )
    return {
        "status": "ready",
        "sha256": digest,
        "effective_policy": dict(_REQUIRED_RUNTIME_FEATURES),
    }


def _scoring_prompt_check(config: Any) -> dict[str, object]:
    path = Path(config.resolved["scoring"]["prompt"])
    actual = _sha256_file(path, label="scoring_prompt")
    expected = str(config.raw["scoring"]["prompt_sha256"])
    if actual != expected:
        raise DocBenchReadinessError(
            "scoring_prompt_hash_mismatch",
            "scoring prompt SHA-256 differs from the configured digest",
        )
    return {
        "status": "ready",
        "sha256": actual,
        "matches_configured_sha256": True,
    }


def _retrieval_check(environment: Mapping[str, str]) -> dict[str, object]:
    snapshot = retrieval_preflight_snapshot(environment)
    preflight = snapshot.get("preflight")
    preflight_ready = (
        isinstance(preflight, Mapping) and preflight.get("ready") is True
    )
    if snapshot.get("status") != "ready" or not preflight_ready:
        reasons = snapshot.get("degradation_reasons")
        rendered_reasons = (
            ", ".join(str(value) for value in reasons)
            if isinstance(reasons, list) and reasons
            else "retrieval_not_ready"
        )
        raise DocBenchReadinessError(
            "retrieval_not_ready",
            f"local retrieval preflight failed: {rendered_reasons}",
        )
    requested = snapshot.get("requested_profile")
    sqlite_vec = snapshot.get("sqlite_vec")
    return {
        "status": "ready",
        "profile_fingerprint": (
            requested.get("fingerprint")
            if isinstance(requested, Mapping)
            else None
        ),
        "required_methods": (
            list(requested.get("required_methods") or ())
            if isinstance(requested, Mapping)
            else []
        ),
        "encoder_fingerprint": snapshot.get("encoder_fingerprint"),
        "reranker_fingerprint": snapshot.get("reranker_fingerprint"),
        "sqlite_vec_ready": (
            isinstance(sqlite_vec, Mapping) and sqlite_vec.get("ready") is True
        ),
        "local_files_only": (
            requested.get("local_files_only")
            if isinstance(requested, Mapping)
            else None
        ),
    }


def check_config(path: Path) -> dict[str, object]:
    """检查一份配置的全部本地依赖，并只返回可公开的元数据。"""

    config_id = path.stem
    try:
        config = load_docbench_config(path, project_root=PROJECT_ROOT)
    except Exception as exc:  # noqa: BLE001 - aggregate every config failure
        return {
            "config_id": config_id,
            "status": "failed",
            "checks": {"config": _failure(exc)},
        }

    checks: dict[str, dict[str, object]] = {
        "config": {
            "status": "ready",
            "sha256": config.sha256,
            "schema_version": config.raw["schema_version"],
            "lane": config.raw["lane"],
        }
    }
    for name, check in (
        ("dataset", lambda: _dataset_check(config)),
        ("runtime_features", lambda: _runtime_features_check(config)),
        ("scoring_prompt", lambda: _scoring_prompt_check(config)),
    ):
        try:
            checks[name] = check()
        except Exception as exc:  # noqa: BLE001 - aggregate independent checks
            checks[name] = _failure(exc)

    provider_environment: dict[str, str] | None = None
    try:
        provider_environment, providers = _resolve_provider_context(config.resolved)
        checks["providers"] = {"status": "ready", **providers}
    except Exception as exc:  # noqa: BLE001 - report missing local configuration
        checks["providers"] = _failure(exc)

    if provider_environment is None:
        checks["retrieval"] = _failure(
            DocBenchReadinessError(
                "provider_environment_unavailable",
                "retrieval preflight requires the resolved runner environment",
            )
        )
    else:
        try:
            checks["retrieval"] = _retrieval_check(provider_environment)
        except Exception as exc:  # noqa: BLE001 - aggregate independent checks
            checks["retrieval"] = _failure(exc)

    ready = all(check.get("status") == "ready" for check in checks.values())
    return {
        "config_id": config_id,
        "status": "ready" if ready else "failed",
        "checks": checks,
    }


def build_readiness_report(
    config_paths: Sequence[Path],
) -> dict[str, object]:
    """汇总全部现行配置；函数自身不执行任何文件系统写入。"""

    paths = tuple(sorted((Path(path) for path in config_paths), key=str))
    configs = [check_config(path) for path in paths]
    ready_count = sum(config.get("status") == "ready" for config in configs)
    catalog_ready = bool(configs) and ready_count == len(configs)
    report: dict[str, object] = {
        "schema_version": READINESS_SCHEMA_VERSION,
        "benchmark_id": "docbench",
        "lane": "L1",
        "status": "ready" if catalog_ready else "failed",
        "network_calls_performed": 0,
        "run_directories_created": 0,
        "config_count": len(configs),
        "ready_config_count": ready_count,
        "configs": configs,
    }
    if not configs:
        report["catalog_error"] = {
            "error_code": "config_catalog_empty",
            "message": "no checked-in DocBench L1 configs were found",
        }
    return report


__all__ = [
    "DocBenchReadinessError",
    "READINESS_SCHEMA_VERSION",
    "build_readiness_report",
    "check_config",
]
