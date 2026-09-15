"""为 DocBench 正式运行生成不含凭据的源码与环境指纹。"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path


_SOURCE_FINGERPRINT_PATHS = (
    "pyproject.toml",
    "requirements-macos-arm64.lock",
    "runtime-versions.conf",
    "scripts/bootstrap-local-runtime.sh",
    "evals/docbench",
    "src/personagraph",
    "tests/evals",
    "configs",
    # 阅读归档不是被测实现；整理新旧结果不能触发 source_changed_during_run。
    ":(exclude)evals/docbench/results/**",
    ":(exclude)evals/docbench/previous_results/**",
)
_SENSITIVE_ENV_MARKERS = (
    "API_KEY",
    "API_TOKEN",
    "ACCESS_TOKEN",
    "AUTH_TOKEN",
    "AUTHORIZATION",
    "BEARER",
    "CREDENTIAL",
    "PASSWORD",
    "SECRET",
)


def _environment_fingerprint_value(key: str, value: str) -> str:
    upper = key.upper()
    if any(marker in upper for marker in _SENSITIVE_ENV_MARKERS):
        return "<redacted>"
    return value


def read_git_provenance(project_root: Path) -> tuple[str | None, bool | None]:
    """返回当前提交和工作树是否存在改动。"""

    try:
        revision = subprocess.run(  # noqa: S603,S607 - 固定本地 git 命令
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(  # noqa: S603,S607 - 固定本地 git 命令
                ["git", "status", "--porcelain"],
                cwd=project_root,
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            ).stdout.strip()
        )
    except (OSError, subprocess.SubprocessError):
        return None, None
    return revision or None, dirty


def compute_source_tree_sha256(project_root: Path) -> str | None:
    """计算相关已跟踪/未跟踪源码哈希，不把新旧结果归档当作源码。"""

    try:
        completed = subprocess.run(  # noqa: S603,S607 - 固定本地 git 命令
            [
                "git",
                "ls-files",
                "--cached",
                "--others",
                "--exclude-standard",
                "-z",
                "--",
                *_SOURCE_FINGERPRINT_PATHS,
            ],
            cwd=project_root,
            check=True,
            capture_output=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return None

    digest = hashlib.sha256()
    relative_paths = sorted(
        os.fsdecode(value) for value in completed.stdout.split(b"\0") if value
    )
    for relative in relative_paths:
        path = project_root / relative
        digest.update(relative.encode("utf-8", errors="surrogateescape"))
        digest.update(b"\0")
        if path.is_symlink():
            digest.update(b"symlink\0")
            digest.update(os.readlink(path).encode("utf-8", errors="surrogateescape"))
        elif path.is_file():
            digest.update(b"file\0")
            try:
                with path.open("rb") as handle:
                    while chunk := handle.read(1024 * 1024):
                        digest.update(chunk)
            except OSError:
                return None
        else:
            digest.update(b"missing\0")
        digest.update(b"\0")
    return digest.hexdigest()


def compute_environment_sha256(env: Mapping[str, str]) -> str:
    """计算不包含密钥的 Python 与评测环境指纹。"""

    distributions = sorted(
        (
            str(distribution.metadata.get("Name") or "unknown").casefold(),
            str(distribution.version),
        )
        for distribution in importlib.metadata.distributions()
    )
    relevant_environment = sorted(
        (key, _environment_fingerprint_value(key, value))
        for key, value in env.items()
        if key.startswith("PERSONAGRAPH_")
        or key in {"LANG", "LC_ALL", "LC_CTYPE", "PYTHONHASHSEED", "TZ"}
    )
    payload = {
        "executable": sys.executable,
        "platform": platform.platform(),
        "python": sys.version,
        "distributions": distributions,
        "subprocess_environment": relevant_environment,
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=True, sort_keys=True).encode("utf-8")
    ).hexdigest()


__all__ = [
    "compute_environment_sha256",
    "compute_source_tree_sha256",
    "read_git_provenance",
]
