"""正式 DocBench L1 评测的严格配置加载器。

本模块刻意只负责配置所有权：不读取凭据、不检查所选数据集、不创建运行目录，也不调用
模型。Runner 会同时获得一份规范、可移植的快照，以及一份已解析文件系统路径的视图：
普通相对路径以仓库根为基准，``bench://`` 路径以仓库外的 DocBench 目录为基准。
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import yaml
from jsonschema import Draft202012Validator, FormatChecker

DOCBENCH_PACKAGE_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = DOCBENCH_PACKAGE_ROOT.parents[1]
SCHEMA_PATH = DOCBENCH_PACKAGE_ROOT / "config.schema.json"
SCHEMA_VERSION = "docbench-l1-eval-v1"
BENCH_EVAL_DIR_ENV = "PERSONAGRAPH_BENCH_EVAL_DIR"
BENCH_PATH_PREFIX = "bench://"

_PATH_FIELDS: tuple[tuple[str, str], ...] = (
    ("dataset", "data_root"),
    ("dataset", "selection"),
    ("run", "runtime_features"),
    ("run", "output_root"),
    ("scoring", "prompt"),
)
_FORBIDDEN_SECRET_FIELDS = frozenset(
    {"api_key", "vision_api_key", "judge_api_key", "secret_key", "token"}
)


class DocBenchConfigError(ValueError):
    """正式 DocBench 配置不可读或违反其契约。"""


@dataclass(frozen=True, slots=True)
class LoadedDocBenchConfig:
    """经过校验、同时包含可移植投影和路径解析投影的配置。"""

    source_path: Path
    project_root: Path
    raw: dict[str, Any]
    resolved: dict[str, Any]
    canonical_snapshot: dict[str, Any]
    sha256: str

    @property
    def config_sha256(self) -> str:
        """为需要命名绑定对象的产物代码提供显式别名。"""

        return self.sha256


def docbench_root(environment: Mapping[str, str] | None = None) -> Path:
    """返回仓库外唯一的 DocBench 资产、运行结果及工作区根目录。"""

    source = os.environ if environment is None else environment
    configured = source.get(BENCH_EVAL_DIR_ENV)
    if configured is None:
        bench_eval_dir = Path.home() / "Desktop/bench_eval"
    else:
        configured = str(configured).strip()
        if not configured:
            raise DocBenchConfigError(f"{BENCH_EVAL_DIR_ENV} cannot be empty")
        bench_eval_dir = Path(configured).expanduser()
        if not bench_eval_dir.is_absolute():
            raise DocBenchConfigError(f"{BENCH_EVAL_DIR_ENV} must be an absolute path")
    resolved = (bench_eval_dir / "docbench").resolve(strict=False)
    if resolved == PROJECT_ROOT or resolved.is_relative_to(PROJECT_ROOT):
        raise DocBenchConfigError(
            f"{BENCH_EVAL_DIR_ENV} must keep DocBench outside the source checkout"
        )
    return resolved


def resolve_config_path(
    value: str | Path,
    project_root: Path = PROJECT_ROOT,
    *,
    environment: Mapping[str, str] | None = None,
) -> Path:
    """解析显式、仓库相对或 ``bench://`` 路径，但不要求目标已经存在。"""

    root = Path(project_root).expanduser().resolve()
    raw_value = str(value)
    if raw_value.startswith(BENCH_PATH_PREFIX):
        relative_value = raw_value.removeprefix(BENCH_PATH_PREFIX)
        relative_path = Path(relative_value)
        if (
            not relative_value
            or relative_path.is_absolute()
            or any(part in {".", ".."} for part in relative_path.parts)
            or "\\" in relative_value
        ):
            raise DocBenchConfigError("invalid bench-owned DocBench path")
        benchmark_root = docbench_root(environment)
        resolved = (benchmark_root / relative_path).resolve(strict=False)
        if resolved != benchmark_root and not resolved.is_relative_to(
            benchmark_root
        ):
            raise DocBenchConfigError("DocBench path escapes benchmark root")
        return resolved
    if "://" in raw_value:
        raise DocBenchConfigError("unsupported path namespace")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = root / path
    return path.resolve(strict=False)


def canonical_config_snapshot(payload: Mapping[str, Any]) -> dict[str, Any]:
    """返回与输入解耦、兼容 JSON 且映射顺序稳定的快照。

    路径刻意保持配置中的原始写法，使 fingerprint 能在不同仓库 checkout 之间移植；
    文件系统访问仍以 ``resolved`` 视图为权威。
    """

    try:
        serialized = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        snapshot = json.loads(serialized)
    except (TypeError, ValueError) as exc:
        raise DocBenchConfigError(
            "DocBench config must contain only finite JSON-compatible values"
        ) from exc
    if not isinstance(snapshot, dict):
        raise DocBenchConfigError("DocBench config must be a YAML object")
    return snapshot


def canonical_config_sha256(payload: Mapping[str, Any]) -> str:
    """生成不受 YAML 格式和键顺序影响的配置 fingerprint。"""

    snapshot = canonical_config_snapshot(payload)
    canonical = json.dumps(
        snapshot,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def load_docbench_config(
    path: str | Path,
    *,
    project_root: Path = PROJECT_ROOT,
    environment: Mapping[str, str] | None = None,
) -> LoadedDocBenchConfig:
    """加载并严格校验一份 ``docbench-l1-eval-v1`` YAML 文件。"""

    root = Path(project_root).expanduser().resolve()
    source_path = resolve_config_path(path, root, environment=environment)
    try:
        payload = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise DocBenchConfigError(
            f"cannot read DocBench config {source_path}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise DocBenchConfigError("DocBench config must be a YAML object")

    _reject_inline_secret_fields(payload)
    schema = _load_schema()
    errors = sorted(
        Draft202012Validator(
            schema,
            format_checker=FormatChecker(),
        ).iter_errors(payload),
        key=lambda error: tuple(str(part) for part in error.absolute_path),
    )
    if errors:
        rendered: list[str] = []
        for error in errors[:8]:
            location = ".".join(str(part) for part in error.absolute_path)
            rendered.append(f"{location or '<root>'}: {error.message}")
        suffix = "" if len(errors) <= 8 else f" (+{len(errors) - 8} more)"
        raise DocBenchConfigError(
            "DocBench config schema validation failed: "
            + "; ".join(rendered)
            + suffix
        )

    snapshot = canonical_config_snapshot(payload)
    resolved = _resolved_projection(
        snapshot,
        project_root=root,
        environment=environment,
    )
    return LoadedDocBenchConfig(
        source_path=source_path,
        project_root=root,
        raw=deepcopy(snapshot),
        resolved=resolved,
        canonical_snapshot=deepcopy(snapshot),
        sha256=canonical_config_sha256(snapshot),
    )


def _load_schema() -> dict[str, Any]:
    try:
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DocBenchConfigError(
            f"cannot read DocBench config schema {SCHEMA_PATH}: {exc}"
        ) from exc
    if not isinstance(schema, dict):
        raise DocBenchConfigError("DocBench config schema must be a JSON object")
    try:
        Draft202012Validator.check_schema(schema)
    except Exception as exc:
        raise DocBenchConfigError("DocBench config schema is invalid") from exc
    return schema


def _resolved_projection(
    snapshot: Mapping[str, Any],
    *,
    project_root: Path,
    environment: Mapping[str, str] | None,
) -> dict[str, Any]:
    resolved = deepcopy(dict(snapshot))
    for owner, field in _PATH_FIELDS:
        value = resolved[owner][field]
        resolved[owner][field] = resolve_config_path(
            value,
            project_root,
            environment=environment,
        )
    return resolved


def _reject_inline_secret_fields(payload: object, path: tuple[str, ...] = ()) -> None:
    if isinstance(payload, Mapping):
        for raw_key, value in payload.items():
            key = str(raw_key)
            location = (*path, key)
            if key.casefold() in _FORBIDDEN_SECRET_FIELDS:
                raise DocBenchConfigError(
                    "inline secret field is forbidden: " + ".".join(location)
                )
            _reject_inline_secret_fields(value, location)
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            _reject_inline_secret_fields(value, (*path, str(index)))


__all__ = [
    "BENCH_EVAL_DIR_ENV",
    "BENCH_PATH_PREFIX",
    "DocBenchConfigError",
    "LoadedDocBenchConfig",
    "PROJECT_ROOT",
    "SCHEMA_PATH",
    "SCHEMA_VERSION",
    "canonical_config_sha256",
    "canonical_config_snapshot",
    "docbench_root",
    "load_docbench_config",
    "resolve_config_path",
]
