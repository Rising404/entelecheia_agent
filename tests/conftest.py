"""测试全局配置：隔离本地配置与持久状态。"""

import os
import tempfile

# 必须发生在任何 personagraph 导入之前：``configuration.paths`` 在导入时就把
# STATE_DIR 和 LOCAL_CONFIG_DIR 定死，之后再设环境变量已经来不及。
#
# 这是会话级兜底，不是逐测试隔离——下面的 autouse fixture 才负责后者。两层都要：
# fixture 逐个重定向已知的 store，而 STATE_DIR 一次盖住所有从它派生的路径
# （项目目录、项目文档、session.sqlite、沙箱目录、
# disclosures.sqlite 等等）。只靠逐个重定向就是打地鼠：新增一个 store 而忘了在
# fixture 里登记，它就会安静地写进开发者真实的 var/，而测试照样是绿的。
#
# 用 setdefault：显式设了 PERSONAGRAPH_STATE_DIR 的调用方（例如需要真实状态的
# live 跑批）仍然说了算。
os.environ.setdefault(
    "PERSONAGRAPH_STATE_DIR",
    os.path.join(tempfile.gettempdir(), "entelecheia-test-state"),
)
os.environ.setdefault(
    "PERSONAGRAPH_LOCAL_CONFIG_DIR",
    os.path.join(tempfile.gettempdir(), "entelecheia-test-config"),
)

import pytest

from personagraph.workspace.ingestion.execution import (
    reset_synchronous_ingest_worker,
)


@pytest.fixture(autouse=True)
def _deterministic_test_runtime(monkeypatch, tmp_path):
    """使测试不依赖开发者保存的设置和数据。

    默认统一应用隔离，而不是让每个测试自行处理。测试若忘记重定向某个存储，
    不会明显失败，而会悄悄写入开发者的真实数据库；此前正因如此，
    ``sessions.sqlite`` 中累积了 172 个游离文件夹。需要指定路径的测试
    仍可在此后覆盖这些设置。
    """
    from personagraph.configuration import app_settings as runtime_config
    from personagraph.model_io import endpoint_profiles as model_profiles
    from personagraph.configuration import paths
    from personagraph.session import store as session_store
    from personagraph.tools.catalog.persistence import repository as tool_catalog_repository
    from personagraph.api.service import session_titles
    from personagraph.runtime.post_commit.scheduler import (
        resume_turn_post_commit_scheduling,
        stop_turn_post_commit_workers,
    )

    # 标题 job 不得跨越测试的 tmp_path/monkeypatch 生命周期；专项测试直接运行 worker。
    monkeypatch.setattr(session_titles, "schedule_session_title", lambda _session_id: None)
    monkeypatch.delenv("PERSONAGRAPH_DEFAULT_PROJECTS_DIR", raising=False)

    monkeypatch.setenv("PERSONAGRAPH_MODEL_PROVIDER", "mock")
    # Production is strict local BGE-M3.  Ordinary unit tests explicitly use
    # the dependency-free rollback profile so they never load multi-GB weights.
    monkeypatch.setenv("PERSONAGRAPH_RETRIEVAL_PROFILE", "lexical")
    monkeypatch.setenv("PERSONAGRAPH_RETRIEVAL_RERANKER", "off")
    monkeypatch.setenv("PERSONAGRAPH_RETRIEVAL_FAILURE_POLICY", "strict")
    monkeypatch.setenv("PERSONAGRAPH_RETRIEVAL_LOCAL_FILES_ONLY", "true")

    isolated_config = tmp_path / "config"
    isolated_config.mkdir(exist_ok=True)
    monkeypatch.setattr(
        runtime_config,
        "CONFIG_PATH",
        isolated_config / "app_config.json",
    )
    monkeypatch.setattr(
        model_profiles,
        "CONFIG_PATH",
        isolated_config / "model_profiles.json",
    )

    isolated = tmp_path / "state"
    isolated.mkdir(exist_ok=True)
    # Product API 会为未显式选目录的 GUI 会话创建用户可见的默认 Project。
    # 测试必须连这棵文件树一起隔离，不能写入开发者的 ~/Documents。
    monkeypatch.setattr(
        paths,
        "DEFAULT_SESSION_PROJECTS_DIR",
        tmp_path / "default-projects",
    )
    monkeypatch.setattr(session_store, "DB_PATH", isolated / "sessions.sqlite")
    monkeypatch.setattr(
        tool_catalog_repository,
        "DEFAULT_TOOL_CATALOG_DATABASE_PATH",
        isolated / "tool_catalog.sqlite",
    )
    # 连接缓存以路径为键，因此不能把上一项测试的数据库句柄带入当前测试。
    session_store._INITIALIZED_PATHS.clear()
    reset_synchronous_ingest_worker()
    resume_turn_post_commit_scheduling(store=session_store)

    yield
    # 持续后台任务不能跨越本项测试的数据库与 monkeypatch 生命周期。
    assert stop_turn_post_commit_workers(store=session_store, timeout_seconds=5)
    resume_turn_post_commit_scheduling(store=session_store)
    session_store._INITIALIZED_PATHS.clear()
    reset_synchronous_ingest_worker()


@pytest.fixture
def seed_session_ids(tmp_path, monkeypatch):
    """为使用对话记录存储的领域测试创建显式会话 ID。"""
    from personagraph.session import store as session_store

    monkeypatch.setattr(session_store, "DB_PATH", tmp_path / "graph-sessions.sqlite")

    def seed(*session_ids: str) -> None:
        session_store.init_db()
        with session_store._connect() as conn:
            for session_id in session_ids:
                conn.execute(
                    "INSERT OR IGNORE INTO sessions"
                    " (id, persona_id, title, status, created_at, last_active_at)"
                    " VALUES (?, 'Entelecheia', 'test session', 'active', ?, ?)",
                    (session_id, "2026-07-12T00:00:00+00:00", "2026-07-12T00:00:00+00:00"),
                )

    return seed


@pytest.fixture
def partitioned_project_state(tmp_path, monkeypatch):
    """启用真实的 Project/Session 分库路由，并把全部状态隔离到本用例。

    全局测试夹具默认保留旧单库接缝，供尚未迁移的测试使用。需要验证当前生产
    布局的测试应显式请求本夹具；之后通过 ``session_store.create_session(...,
    working_dir=str(project_root))`` 建立项目绑定。进入
    ``session_database_scope(session_id)`` 时会同时绑定该项目唯一的
    ``documents.sqlite``。
    """
    from personagraph.configuration import paths
    from personagraph.session import catalog as catalog_module
    from personagraph.session import project_catalog
    from personagraph.session import store as session_store

    state_dir = tmp_path / "partitioned-state"
    state_dir.mkdir()
    shared_catalog = state_dir / "project_catalog.sqlite"
    retired_shared_session_path = state_dir / "retired-shared-session.sqlite"

    # ``DB_PATH == _DEFAULT_DB_PATH`` 是生产分库模式的显式哨兵。测试使用一个
    # 临时且不会被打开的路径，避免任何失败分支意外触碰 /dev/null。
    monkeypatch.setattr(
        session_store,
        "_DEFAULT_DB_PATH",
        retired_shared_session_path,
    )
    monkeypatch.setattr(session_store, "DB_PATH", retired_shared_session_path)
    monkeypatch.setattr(paths, "STATE_DIR", state_dir)
    monkeypatch.setattr(paths, "PROJECT_CATALOG_DB_PATH", shared_catalog)
    monkeypatch.setattr(paths, "PROJECTS_DIR", state_dir / "projects")
    monkeypatch.setattr(paths, "SESSIONS_DIR", state_dir / "sessions")
    # ``catalog_module`` 保留 paths 模块对象；此赋值用于明确其查找目标，也防止
    # 未来改为局部导入常量时测试悄悄退回共享状态。
    monkeypatch.setattr(
        catalog_module.paths,
        "PROJECT_CATALOG_DB_PATH",
        shared_catalog,
    )
    monkeypatch.setattr(project_catalog, "DB_PATH", shared_catalog)
    session_store._INITIALIZED_PATHS.clear()
    reset_synchronous_ingest_worker()

    yield state_dir

    session_store._INITIALIZED_PATHS.clear()
    reset_synchronous_ingest_worker()


@pytest.fixture
def bound_partitioned_session(partitioned_project_state):
    """Create one current-layout Session and retain its database scope."""

    del partitioned_project_state
    from contextlib import ExitStack

    from personagraph.session import store as session_store

    with ExitStack() as scopes:
        def create(*, working_dir, title="L1 runtime test"):
            session_id = session_store.create_session(
                "Entelecheia",
                title=title,
                working_dir=str(working_dir),
            )
            scopes.enter_context(
                session_store.session_database_scope(session_id)
            )
            return session_id

        yield create
