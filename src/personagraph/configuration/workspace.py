"""用户工作目录配置的路径校验；不创建目录、不授予工具权限。"""

from pathlib import Path

from . import paths


def validate_workspace_directory(value: object, *, must_exist: bool) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("工作目录必须是非空路径")
    supplied = Path(value.strip()).expanduser()
    if not supplied.is_absolute():
        raise ValueError("工作目录必须是本机绝对路径")
    try:
        resolved = supplied.resolve(strict=must_exist)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError("工作目录不存在或无法解析") from exc
    source_root = Path(paths.PROJECT_ROOT).resolve()
    if resolved == source_root or resolved.is_relative_to(source_root):
        raise ValueError("工作目录不能位于应用源码目录中")
    if paths.deny_reason(resolved) is not None:
        raise ValueError("工作目录被本机路径安全策略拒绝")
    if (must_exist or resolved.exists()) and not resolved.is_dir():
        raise ValueError("工作目录必须是文件夹")
    return resolved
