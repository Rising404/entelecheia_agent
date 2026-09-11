"""测试进程绝不能写进开发者真实的状态目录。

这条围栏保护的是一个安静的失败：某个 store 没有被 conftest 的 fixture 逐个重定向时，
它会把会话、租约、项目登记写进真实的 ``var/``，而测试全绿。真实后果已经发生过——
68 个执行窗口被测试租约永久占住，界面上那些会话再也发不出消息；12 个 pytest 临时
目录混进了工作区的项目列表。
"""

from pathlib import Path

from personagraph.configuration import paths


REPO_VAR = Path(__file__).resolve().parents[1] / "var"
REPO_LOCAL_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "local"


def test_state_dir_is_not_the_developer_var_directory() -> None:
    assert paths.STATE_DIR != REPO_VAR.resolve()


def test_local_config_dir_is_not_the_developer_config_directory() -> None:
    assert paths.LOCAL_CONFIG_DIR != REPO_LOCAL_CONFIG.resolve()


def test_every_state_path_stays_under_the_isolated_state_dir() -> None:
    """派生路径必须整体跟着 STATE_DIR 走，不能有漏网的绝对路径。"""

    derived = {
        name: value
        for name, value in vars(paths).items()
        if isinstance(value, Path) and name.endswith(("_DIR", "_CONFIG"))
    }
    # 这几个不从 STATE_DIR 派生，本来就该指向仓库自身。
    non_state_roots = {
        "PROJECT_ROOT",
        "CONFIG_DIR",
        "LOCAL_CONFIG_DIR",
        "DEFAULT_SESSION_PROJECTS_DIR",
    }
    escaped = sorted(
        name
        for name, value in derived.items()
        if name not in non_state_roots
        and paths.STATE_DIR not in value.parents
        and value != paths.STATE_DIR
    )
    assert not escaped, f"这些路径没有跟随 STATE_DIR：{escaped}"
