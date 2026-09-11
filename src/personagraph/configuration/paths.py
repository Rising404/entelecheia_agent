"""应用路径、私有启动环境和模型可访问路径边界的权威。"""

import os
import re
import sys
from collections.abc import Mapping
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = PROJECT_ROOT / "configs"

_APPLICATION_DIRECTORY_NAME = "Entelecheia"


def _absolute_environment_path(
    environment: Mapping[str, str],
    name: str,
) -> Path | None:
    """Return an absolute OS directory override, ignoring invalid relative values."""

    raw_value = str(environment.get(name) or "").strip()
    if not raw_value:
        return None
    candidate = Path(raw_value).expanduser()
    return candidate if candidate.is_absolute() else None


def resolve_absolute_environment_path(
    name: str,
    default: str | Path,
    *,
    environment: Mapping[str, str] | None = None,
) -> Path:
    """Resolve a private-path override, rejecting cwd-dependent relative values.

    Product state, credentials, and local configuration must never move into a Git
    checkout merely because a shell supplied ``var`` or another relative path.
    ``~`` remains supported because it expands to an explicit per-user location.
    """

    selected_environment = os.environ if environment is None else environment
    raw_value = str(selected_environment.get(name) or "").strip()
    if not raw_value:
        resolved = Path(default).expanduser().resolve()
    else:
        candidate = Path(raw_value).expanduser()
        if not candidate.is_absolute():
            raise ValueError(f"{name} must be an absolute path")
        resolved = candidate.resolve()
    if resolved == PROJECT_ROOT or resolved.is_relative_to(PROJECT_ROOT):
        raise ValueError(f"{name} must be outside the source checkout")
    return resolved


def _default_user_directories(
    *,
    platform_name: str | None = None,
    environment: Mapping[str, str] | None = None,
    home_directory: Path | None = None,
) -> tuple[Path, Path]:
    """Resolve secure per-user state and configuration roots with stdlib only.

    Runtime state and credentials must not default to the source checkout: a clone may
    itself be a Git repository or be copied when publishing a demo.  The two returned
    trees are deliberately disjoint so the model-facing path guard can deny the local
    configuration tree without also denying ordinary application state.
    """

    selected_platform = sys.platform if platform_name is None else platform_name
    selected_environment = os.environ if environment is None else environment
    selected_home = Path.home() if home_directory is None else Path(home_directory)

    if selected_platform == "darwin":
        application_root = (
            selected_home
            / "Library"
            / "Application Support"
            / _APPLICATION_DIRECTORY_NAME
        )
        return application_root / "state", application_root / "config"

    if selected_platform == "win32":
        data_root = _absolute_environment_path(selected_environment, "LOCALAPPDATA")
        if data_root is None:
            data_root = selected_home / "AppData" / "Local"
        config_root = _absolute_environment_path(selected_environment, "APPDATA")
        if config_root is None:
            config_root = selected_home / "AppData" / "Roaming"
        return (
            data_root / _APPLICATION_DIRECTORY_NAME / "state",
            config_root / _APPLICATION_DIRECTORY_NAME / "config",
        )

    data_root = _absolute_environment_path(selected_environment, "XDG_DATA_HOME")
    if data_root is None:
        data_root = selected_home / ".local" / "share"
    config_root = _absolute_environment_path(selected_environment, "XDG_CONFIG_HOME")
    if config_root is None:
        config_root = selected_home / ".config"
    return (
        data_root / _APPLICATION_DIRECTORY_NAME / "state",
        config_root / _APPLICATION_DIRECTORY_NAME / "config",
    )


_DEFAULT_STATE_ROOT, _DEFAULT_LOCAL_CONFIG_ROOT = _default_user_directories()
STATE_DIR = resolve_absolute_environment_path(
    "PERSONAGRAPH_STATE_DIR",
    _DEFAULT_STATE_ROOT,
)
LOCAL_CONFIG_DIR = resolve_absolute_environment_path(
    "PERSONAGRAPH_LOCAL_CONFIG_DIR",
    _DEFAULT_LOCAL_CONFIG_ROOT,
)
API_TOKEN_PATH = resolve_absolute_environment_path(
    "PERSONAGRAPH_API_TOKEN_FILE",
    STATE_DIR / "api_secret",
)
PROJECT_CATALOG_DB_PATH = STATE_DIR / "project_catalog.sqlite"
PROJECTS_DIR = STATE_DIR / "projects"
SESSIONS_DIR = STATE_DIR / "sessions"
# 与 ``PROJECTS_DIR`` 不同，这里存放的是用户/Agent 都可见的真实 Project
# 文件树，而不是按 project_id 分区的私有 SQLite 状态。默认放在用户文档目录，
# 避免源码 checkout 和应用私有 state 成为模型工作区。
DEFAULT_SESSION_PROJECTS_DIR = resolve_absolute_environment_path(
    "PERSONAGRAPH_DEFAULT_PROJECTS_DIR",
    Path.home() / "Documents" / _APPLICATION_DIRECTORY_NAME,
)

# 模型可访问路径的共同拒绝规则。它与上述路径配置放在同一 owner，确保本机私有
# 配置目录和 API token 的真实位置变化时，工具、摄取与 API 使用同一条边界。
DENY_DIR_PARTS: tuple[str, ...] = (".git", ".ssh", "Keychains")
DENY_ABS_PREFIXES: tuple[str, ...] = (
    "/etc",
    "/System",
    "/private/etc",
    "/usr",
    "/bin",
    "/sbin",
    str(Path.home() / "Library"),
)


def deny_reason(path: Path) -> str | None:
    """返回模型不可访问本地路径的首个硬拒绝原因；允许时返回 ``None``。"""

    if _same_private_path(path, API_TOKEN_PATH):
        return "denied_sensitive_path"
    # 用户授权范围内的文件不按名称猜测敏感性；仅隔离明确的 Host 私有路径。
    folded_parts = tuple(part.casefold() for part in path.parts)
    if any(part.casefold() in folded_parts for part in DENY_DIR_PARTS):
        return "denied_sensitive_dir"
    # 配置与持久状态均属 Host 私有面；普通用户 SQLite 文件不因后缀被拒读。
    if any(_has_casefolded_component_prefix(path, root) for root in (LOCAL_CONFIG_DIR, STATE_DIR)):
        return "denied_sensitive_dir"
    if any(
        _has_casefolded_component_prefix(path, Path(prefix))
        for prefix in DENY_ABS_PREFIXES
    ):
        return "denied_system_path"
    return None


def _has_casefolded_component_prefix(candidate: Path, prefix: Path) -> bool:
    """比较路径前缀，同时规避大小写或原始字符串前缀歧义。"""

    candidate_parts = tuple(part.casefold() for part in candidate.parts)
    prefix_parts = tuple(part.casefold() for part in prefix.parts)
    return (
        len(candidate_parts) >= len(prefix_parts)
        and candidate_parts[: len(prefix_parts)] == prefix_parts
    )


def _same_private_path(candidate: Path, protected: Path) -> bool:
    """按词法、规范路径及现存 inode 识别同一个私有文件。"""

    candidate = Path(candidate).expanduser()
    protected = Path(protected).expanduser()
    lexical_candidate = candidate if candidate.is_absolute() else candidate.absolute()
    lexical_protected = protected if protected.is_absolute() else protected.absolute()
    if _same_casefolded_components(lexical_candidate, lexical_protected):
        return True
    try:
        if _same_casefolded_components(
            candidate.resolve(strict=False),
            protected.resolve(strict=False),
        ):
            return True
    except (OSError, RuntimeError):
        pass
    try:
        return candidate.samefile(protected)
    except (FileNotFoundError, OSError):
        return False


def _same_casefolded_components(left: Path, right: Path) -> bool:
    return tuple(part.casefold() for part in left.parts) == tuple(
        part.casefold() for part in right.parts
    )


def load_dotenv(path: Path | None = None) -> None:
    """从私有配置目录补充进程环境，不覆盖调用者已设置的值。"""

    env_path = path or LOCAL_CONFIG_DIR / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_STATE_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def resolve_exact_child_path(
    parent: str | Path,
    *components: str,
    description: str,
) -> Path:
    """Resolve one expected child shape without following a planted symlink.

    SQLite follows symlinks when opening a path.  State locators therefore must
    compare the resolved target with the exact lexical child beneath an already
    resolved parent before any directory creation or connection occurs.
    """

    configured_parent = Path(parent).expanduser()
    root = configured_parent.resolve()
    if configured_parent.absolute() != root:
        raise ValueError(f"{description} parent must not traverse a symbolic link")
    expected = root.joinpath(*components)
    try:
        resolved = expected.resolve(strict=False)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError(f"{description} is invalid") from exc
    if resolved != expected:
        raise ValueError(f"{description} must not traverse a symbolic link")
    return resolved


def projects_directory() -> Path:
    """Return the configured Project-state directory without symlink escape."""

    configured = Path(PROJECTS_DIR).expanduser()
    return resolve_exact_child_path(
        configured.parent,
        configured.name,
        description="projects directory",
    )


def resolved_project_catalog_path(configured: str | Path) -> Path:
    """Resolve a catalog file while rejecting a preplanted file symlink."""

    candidate = Path(configured).expanduser()
    if not candidate.name:
        raise ValueError("project catalog path is invalid")
    return resolve_exact_child_path(
        candidate.parent,
        candidate.name,
        description="project catalog path",
    )


def project_documents_db_path(project_id: str) -> Path:
    """返回一个已验证项目 ID 所属的文档数据库。

    项目 ID 会成为用户状态根目录 ``projects`` 下的目录名。将验证逻辑放在路径构造旁，
    可防止目录值（或未来的 API 参数）越过该状态边界。
    """

    value = str(project_id or "").strip()
    if not _STATE_IDENTIFIER.fullmatch(value) or value in {".", ".."}:
        raise ValueError("invalid project_id")
    return resolve_exact_child_path(
        projects_directory(),
        value,
        "documents.sqlite",
        description="project database path",
    )
