"""启动器复用后端前核对源码身份；只暴露摘要，不返回本机路径。"""

from __future__ import annotations

import hashlib
from pathlib import Path


def build_runtime_identity(root: Path) -> dict[str, str]:
    """标识 checkout 与实际源码内容；未提交修改也必须触发重新启动。"""

    root = root.resolve(strict=True)
    package = root / "src" / "personagraph"
    if not package.is_dir():
        raise ValueError("runtime identity requires a source checkout")
    files = list(package.rglob("*.py"))
    files.extend(root / name for name in ("pyproject.toml", "uv.lock") if (root / name).is_file())
    digest = hashlib.sha256()
    # 与 Electron 使用相同的 UTF-8 排序及长度固定的文件摘要，避免路径/正文拼接歧义。
    for file in sorted(files, key=lambda p: p.relative_to(root).as_posix().encode("utf-8")):
        digest.update(file.relative_to(root).as_posix().encode("utf-8") + b"\0")
        digest.update(hashlib.sha256(file.read_bytes()).digest())
    return {
        "checkout_sha256": hashlib.sha256(root.as_posix().encode("utf-8")).hexdigest(),
        "source_sha256": digest.hexdigest(),
    }


# Router 在 API 启动时导入本模块。不能在健康探测时重新计算，否则旧进程会冒充新源码。
_PROCESS_IDENTITY = build_runtime_identity(Path(__file__).resolve().parents[3])


def runtime_identity() -> dict[str, str]:
    return dict(_PROCESS_IDENTITY)
