import os
import subprocess
import sys
from pathlib import Path

import pytest

from personagraph.configuration import paths
from personagraph.configuration.paths import deny_reason


@pytest.mark.parametrize("name", ["tokenizer.py", "secret_notes.txt", ".env", "cert.pem"])
def test_authorized_project_names_do_not_imply_a_private_host_path(tmp_path, name):
    assert deny_reason(tmp_path / name) is None


@pytest.mark.parametrize("private_root", ["STATE_DIR", "LOCAL_CONFIG_DIR"])
def test_actual_host_private_directories_remain_isolated(tmp_path, monkeypatch, private_root):
    root = tmp_path / "host-private"
    monkeypatch.setattr(paths, private_root, root)
    assert deny_reason(root / "ordinary.txt") == "denied_sensitive_dir"


def _imported_runtime_paths(environment: dict[str, str]) -> list[str]:
    environment["PYTHONPATH"] = str(paths.PROJECT_ROOT / "src")
    return subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from personagraph.configuration import paths; "
                "print(paths.STATE_DIR); "
                "print(paths.PROJECT_CATALOG_DB_PATH); "
                "print(paths.PROJECTS_DIR); "
                "print(paths.SESSIONS_DIR); "
                "print(paths.DEFAULT_SESSION_PROJECTS_DIR); "
                "print(paths.LOCAL_CONFIG_DIR)"
            ),
        ],
        capture_output=True,
        text=True,
        env=environment,
        cwd=paths.PROJECT_ROOT,
        check=True,
    ).stdout.splitlines()


def test_runtime_paths_default_to_per_user_directories():
    """Production imports must never default to mutable paths in the checkout.

    这一条不能直接断言 ``paths.STATE_DIR``：测试进程刻意把它指向临时目录
    （见 ``tests/conftest.py`` 顶部的兜底），断言当前值只会锁死那层隔离。
    改为在一个没有该环境变量的子进程里重新解析，既验证了默认规则，也不会
    污染当前进程已经建立的隔离。
    """

    env = {
        key: value
        for key, value in os.environ.items()
        if key not in {
            "PERSONAGRAPH_STATE_DIR",
            "PERSONAGRAPH_LOCAL_CONFIG_DIR",
            "PERSONAGRAPH_DEFAULT_PROJECTS_DIR",
        }
    }
    default_state, default_config = paths._default_user_directories(environment=env)
    resolved = _imported_runtime_paths(env)

    assert resolved == [
        str(default_state.resolve()),
        str((default_state / "project_catalog.sqlite").resolve()),
        str((default_state / "projects").resolve()),
        str((default_state / "sessions").resolve()),
        str((Path.home() / "Documents" / "Entelecheia").resolve()),
        str(default_config.resolve()),
    ]
    assert not default_state.is_relative_to(paths.PROJECT_ROOT)
    assert not default_config.is_relative_to(paths.PROJECT_ROOT)


def test_explicit_runtime_directory_environment_overrides_take_priority(tmp_path):
    state_dir = tmp_path / "explicit-state"
    config_dir = tmp_path / "explicit-config"
    default_projects = tmp_path / "visible-projects"
    env = dict(os.environ)
    env["PERSONAGRAPH_STATE_DIR"] = str(state_dir)
    env["PERSONAGRAPH_LOCAL_CONFIG_DIR"] = str(config_dir)
    env["PERSONAGRAPH_DEFAULT_PROJECTS_DIR"] = str(default_projects)

    assert _imported_runtime_paths(env) == [
        str(state_dir.resolve()),
        str((state_dir / "project_catalog.sqlite").resolve()),
        str((state_dir / "projects").resolve()),
        str((state_dir / "sessions").resolve()),
        str(default_projects.resolve()),
        str(config_dir.resolve()),
    ]


@pytest.mark.parametrize(
    "name",
    (
        "PERSONAGRAPH_STATE_DIR",
        "PERSONAGRAPH_LOCAL_CONFIG_DIR",
        "PERSONAGRAPH_DEFAULT_PROJECTS_DIR",
    ),
)
def test_relative_runtime_directory_overrides_fail_closed(name):
    env = dict(os.environ)
    env.pop("PERSONAGRAPH_STATE_DIR", None)
    env.pop("PERSONAGRAPH_LOCAL_CONFIG_DIR", None)
    env[name] = "relative-private-directory"
    env["PYTHONPATH"] = str(paths.PROJECT_ROOT / "src")

    result = subprocess.run(
        [sys.executable, "-c", "from personagraph.configuration import paths"],
        capture_output=True,
        text=True,
        env=env,
        cwd=paths.PROJECT_ROOT,
        check=False,
    )

    assert result.returncode != 0
    assert f"{name} must be an absolute path" in result.stderr


def test_api_token_file_environment_override_matches_the_electron_sidecar(tmp_path):
    token_file = tmp_path / "sidecar" / "api_secret"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(paths.PROJECT_ROOT / "src")
    env["PERSONAGRAPH_API_TOKEN_FILE"] = str(token_file)

    resolved = subprocess.run(
        [
            sys.executable,
            "-c",
            "from personagraph.api.security import TOKEN_PATH; print(TOKEN_PATH)",
        ],
        capture_output=True,
        text=True,
        env=env,
        cwd=paths.PROJECT_ROOT,
        check=True,
    ).stdout.strip()

    assert resolved == str(token_file.resolve())
    assert not token_file.parent.exists()


def test_explicit_api_token_file_is_denied_even_with_an_innocent_project_name(
    tmp_path,
):
    project_root = tmp_path / "external-project"
    token_file = project_root / "notes.txt"
    public_file = project_root / "public.txt"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(paths.PROJECT_ROOT / "src")
    env["PERSONAGRAPH_API_TOKEN_FILE"] = str(token_file)

    resolved = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from personagraph.api.security import TOKEN_PATH; "
                "from personagraph.configuration.paths import deny_reason; "
                "print(TOKEN_PATH); "
                "print(deny_reason(TOKEN_PATH)); "
                f"print(deny_reason(__import__('pathlib').Path({str(public_file)!r})))"
            ),
        ],
        capture_output=True,
        text=True,
        env=env,
        cwd=paths.PROJECT_ROOT,
        check=True,
    ).stdout.splitlines()

    assert resolved == [
        str(token_file.resolve()),
        "denied_sensitive_path",
        "None",
    ]
    assert not project_root.exists()


def test_relative_api_token_file_override_fails_closed():
    env = dict(os.environ)
    env["PYTHONPATH"] = str(paths.PROJECT_ROOT / "src")
    env["PERSONAGRAPH_API_TOKEN_FILE"] = "relative-api-secret"

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from personagraph.api.security import TOKEN_PATH",
        ],
        capture_output=True,
        text=True,
        env=env,
        cwd=paths.PROJECT_ROOT,
        check=False,
    )

    assert result.returncode != 0
    assert "PERSONAGRAPH_API_TOKEN_FILE must be an absolute path" in result.stderr


@pytest.mark.parametrize(
    ("name", "value"),
    (
        ("PERSONAGRAPH_STATE_DIR", "private-state"),
        ("PERSONAGRAPH_LOCAL_CONFIG_DIR", "private-config"),
        ("PERSONAGRAPH_API_TOKEN_FILE", "private-api-token"),
        ("PERSONAGRAPH_DEFAULT_PROJECTS_DIR", "visible-projects"),
    ),
)
def test_private_path_overrides_cannot_target_the_source_checkout(name, value):
    env = dict(os.environ)
    env["PYTHONPATH"] = str(paths.PROJECT_ROOT / "src")
    env[name] = str(paths.PROJECT_ROOT / value)
    module = (
        "personagraph.api.security"
        if name == "PERSONAGRAPH_API_TOKEN_FILE"
        else "personagraph.configuration.paths"
    )

    result = subprocess.run(
        [sys.executable, "-c", f"import {module}"],
        capture_output=True,
        text=True,
        env=env,
        cwd=paths.PROJECT_ROOT,
        check=False,
    )

    assert result.returncode != 0
    assert f"{name} must be outside the source checkout" in result.stderr


def test_default_user_directories_cover_macos_windows_and_xdg(tmp_path):
    fake_home = tmp_path / "home"

    mac_state, mac_config = paths._default_user_directories(
        platform_name="darwin",
        environment={},
        home_directory=fake_home,
    )
    assert mac_state == (
        fake_home / "Library/Application Support/Entelecheia/state"
    )
    assert mac_config == (
        fake_home / "Library/Application Support/Entelecheia/config"
    )

    local_app_data = tmp_path / "local"
    roaming_app_data = tmp_path / "roaming"
    windows_state, windows_config = paths._default_user_directories(
        platform_name="win32",
        environment={
            "LOCALAPPDATA": str(local_app_data),
            "APPDATA": str(roaming_app_data),
        },
        home_directory=fake_home,
    )
    assert windows_state == local_app_data / "Entelecheia/state"
    assert windows_config == roaming_app_data / "Entelecheia/config"

    xdg_data_home = tmp_path / "xdg-data"
    xdg_config_home = tmp_path / "xdg-config"
    xdg_state, xdg_config = paths._default_user_directories(
        platform_name="linux",
        environment={
            "XDG_DATA_HOME": str(xdg_data_home),
            "XDG_CONFIG_HOME": str(xdg_config_home),
        },
        home_directory=fake_home,
    )
    assert xdg_state == xdg_data_home / "Entelecheia/state"
    assert xdg_config == xdg_config_home / "Entelecheia/config"


def test_default_user_directories_fall_back_without_valid_os_environment(tmp_path):
    fake_home = tmp_path / "home"
    linux_state, linux_config = paths._default_user_directories(
        platform_name="linux",
        environment={"XDG_DATA_HOME": "relative", "XDG_CONFIG_HOME": ""},
        home_directory=fake_home,
    )
    assert linux_state == fake_home / ".local/share/Entelecheia/state"
    assert linux_config == fake_home / ".config/Entelecheia/config"

    windows_state, windows_config = paths._default_user_directories(
        platform_name="win32",
        environment={},
        home_directory=fake_home,
    )
    assert windows_state == fake_home / "AppData/Local/Entelecheia/state"
    assert windows_config == fake_home / "AppData/Roaming/Entelecheia/config"


def test_importing_paths_does_not_create_configured_directories(tmp_path):
    fake_state_root = tmp_path / "import-state"
    fake_config_root = tmp_path / "import-config"
    fake_project_root = tmp_path / "import-projects"
    env = dict(os.environ)
    env["PERSONAGRAPH_STATE_DIR"] = str(fake_state_root)
    env["PERSONAGRAPH_LOCAL_CONFIG_DIR"] = str(fake_config_root)
    env["PERSONAGRAPH_DEFAULT_PROJECTS_DIR"] = str(fake_project_root)

    _imported_runtime_paths(env)

    assert not fake_state_root.exists()
    assert not fake_config_root.exists()
    assert not fake_project_root.exists()


def test_runtime_state_defaults_are_outside_source_checkout():
    for path in (
        paths.PROJECT_CATALOG_DB_PATH,
        paths.PROJECTS_DIR,
        paths.SESSIONS_DIR,
    ):
        assert path.is_relative_to(paths.STATE_DIR)
        assert not path.is_relative_to(paths.PROJECT_ROOT)


def test_default_session_projects_are_model_visible_and_separate_from_private_state():
    assert not paths.DEFAULT_SESSION_PROJECTS_DIR.is_relative_to(paths.STATE_DIR)
    assert not paths.DEFAULT_SESSION_PROJECTS_DIR.is_relative_to(paths.PROJECT_ROOT)
    assert deny_reason(paths.DEFAULT_SESSION_PROJECTS_DIR) is None


def test_local_config_is_separate_from_runtime_state():
    assert not paths.LOCAL_CONFIG_DIR.is_relative_to(paths.STATE_DIR)
    assert not paths.STATE_DIR.is_relative_to(paths.LOCAL_CONFIG_DIR)


def test_local_config_tree_is_a_model_inaccessible_security_boundary():
    assert deny_reason(paths.LOCAL_CONFIG_DIR) == "denied_sensitive_dir"
    assert (
        deny_reason(paths.LOCAL_CONFIG_DIR / "app_config.json")
        == "denied_sensitive_dir"
    )
    assert deny_reason(
        paths.LOCAL_CONFIG_DIR.parent / "local-public" / "settings.json"
    ) is None


def test_active_runtime_state_paths_name_the_new_storage_roles():
    assert paths.PROJECT_CATALOG_DB_PATH == paths.STATE_DIR / "project_catalog.sqlite"
    assert paths.PROJECTS_DIR == paths.STATE_DIR / "projects"
    assert paths.SESSIONS_DIR == paths.STATE_DIR / "sessions"
    assert not hasattr(paths, "MEMORY_DIR")
    assert not hasattr(paths, "SESSION_HISTORY_DIR")
    assert not hasattr(paths, "SESSION_FILES_DIR")
    assert not hasattr(paths, "RETRIEVAL_INDEXES_DIR")
    assert not hasattr(paths, "TOKEN_USAGE_DIR")
    assert not hasattr(paths, "SANDBOX_DIR")
    assert not hasattr(paths, "EXECUTION_REPORTS_DIR")
    assert not hasattr(paths, "EXECUTION_CHAINS_DIR")
    assert not hasattr(paths, "WORKSPACE_CONFIG")


def test_project_document_database_path_accepts_only_one_safe_id_component():
    assert paths.project_documents_db_path("project-123") == (
        paths.PROJECTS_DIR / "project-123" / "documents.sqlite"
    )
    for invalid in ("", ".", "..", "../escape", "a/b", "project:unsafe"):
        with pytest.raises(ValueError):
            paths.project_documents_db_path(invalid)


def test_project_document_database_rejects_projects_directory_symlink_escape(
    tmp_path,
    monkeypatch,
):
    state_dir = tmp_path / "state"
    outside = tmp_path / "source-checkout"
    state_dir.mkdir(exist_ok=True)
    outside.mkdir()
    projects_link = state_dir / "projects"
    projects_link.symlink_to(outside, target_is_directory=True)
    monkeypatch.setattr(paths, "PROJECTS_DIR", projects_link)

    with pytest.raises(ValueError, match="projects directory"):
        paths.project_documents_db_path("project-123")


def test_project_document_database_rejects_project_id_symlink_escape(
    tmp_path,
    monkeypatch,
):
    projects_dir = tmp_path / "state" / "projects"
    outside = tmp_path / "source-checkout"
    projects_dir.mkdir(parents=True, exist_ok=True)
    outside.mkdir()
    (projects_dir / "project-123").symlink_to(outside, target_is_directory=True)
    monkeypatch.setattr(paths, "PROJECTS_DIR", projects_dir)

    with pytest.raises(ValueError, match="project database path"):
        paths.project_documents_db_path("project-123")


def test_project_document_database_rejects_database_file_symlink_escape(
    tmp_path,
    monkeypatch,
):
    project_dir = tmp_path / "state" / "projects" / "project-123"
    outside = tmp_path / "source-checkout"
    project_dir.mkdir(parents=True)
    outside.mkdir()
    (project_dir / "documents.sqlite").symlink_to(outside / "escaped.sqlite")
    monkeypatch.setattr(paths, "PROJECTS_DIR", project_dir.parent)

    with pytest.raises(ValueError, match="project database path"):
        paths.project_documents_db_path("project-123")
    assert not (outside / "escaped.sqlite").exists()


def test_config_dir_remains_versioned_asset_boundary():
    assert paths.CONFIG_DIR == paths.PROJECT_ROOT / "configs"


def test_retired_repository_data_dir_is_not_exposed():
    assert not hasattr(paths, "DATA_DIR")
