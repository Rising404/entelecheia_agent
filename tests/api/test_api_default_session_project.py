from __future__ import annotations

import sqlite3
from datetime import date
from pathlib import Path

import pytest

from personagraph.api import service
from personagraph.api.service import sessions as session_service
from personagraph.configuration import paths
from personagraph.runtime.l1.tool_runtime import build_l1_tool_runtime
from personagraph.session import project_catalog
from personagraph.session import store as session_store
from personagraph.session.catalog import SessionCatalog
from tests.helpers.session_records import append_test_turn


pytestmark = pytest.mark.usefixtures("partitioned_project_state")


@pytest.fixture
def default_project_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "visible-projects"
    monkeypatch.setattr(paths, "DEFAULT_SESSION_PROJECTS_DIR", root)
    monkeypatch.setattr(
        session_service,
        "_current_local_date",
        lambda: date(2026, 9, 2),
    )
    return root


def test_session_without_selected_directory_gets_a_named_default_project(
    default_project_root: Path,
) -> None:
    created = service.create_session({"title": "季度/财务 复盘"})["session"]

    expected = default_project_root / "2026-09-02_季度_财务 复盘"
    assert created["working_dir"] == str(expected.resolve())
    assert expected.is_dir()
    assert session_store.get_session(created["id"])["project_id"]


def test_product_session_workspace_can_boot_the_complete_l1_tool_surface(
    default_project_root: Path,
) -> None:
    created = service.create_session({"title": "可执行工作区"})["session"]

    with session_store.session_database_scope(created["id"]):
        runtime = build_l1_tool_runtime(created["id"])

    assert Path(created["working_dir"]).is_relative_to(default_project_root)
    assert "workspace_overview" in runtime.registrations_by_tool_id
    assert "create_output_file" in runtime.registrations_by_tool_id


def test_default_project_names_are_unique_and_titleless_sessions_are_supported(
    default_project_root: Path,
) -> None:
    first = service.create_session({"title": "同名会话"})["session"]
    second = service.create_session({"title": "同名会话"})["session"]
    titleless = service.create_session({})["session"]

    assert Path(first["working_dir"]).name == "2026-09-02_同名会话"
    assert Path(second["working_dir"]).name == "2026-09-02_同名会话-2"
    assert Path(titleless["working_dir"]).name == "2026-09-02_新会话"
    assert len(list(default_project_root.iterdir())) == 3


def test_project_title_is_not_rejected_for_sensitive_keywords(
    default_project_root: Path,
) -> None:
    created = service.create_session({"title": "检查 API token"})["session"]

    assert Path(created["working_dir"]).name == "2026-09-02_检查 API token"
    assert Path(created["working_dir"]).is_relative_to(default_project_root)


def test_default_project_root_inside_source_checkout_is_rejected_without_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = tmp_path / "checkout"
    source_root.mkdir()
    rejected = source_root / "runtime-projects"
    monkeypatch.setattr(paths, "PROJECT_ROOT", source_root)
    monkeypatch.setattr(paths, "DEFAULT_SESSION_PROJECTS_DIR", rejected)

    with pytest.raises(service.ApiError) as error:
        service.create_session({"title": "不应创建"})

    assert error.value.code == "DEFAULT_PROJECT_ROOT_INVALID"
    assert error.value.details == {"reason": "inside_source_checkout"}
    assert not rejected.exists()


def test_preownership_session_failure_removes_only_the_empty_default_directory(
    default_project_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts: list[str] = []

    def fail_claim(session_id: str) -> Path:
        attempts.append(session_id)
        raise RuntimeError("simulated failure before project registration")

    monkeypatch.setattr(session_store, "_new_id", lambda: "unowned1")
    monkeypatch.setattr(
        session_store,
        "_claim_partitioned_session_directory",
        fail_claim,
    )

    with pytest.raises(session_store.SessionCreationNotOwnedError):
        service.create_session({"title": "失败回收"})

    # 非 claim-conflict 异常不得通过重抽 ID 悄悄重试。
    assert attempts == ["unowned1"]
    assert list(default_project_root.iterdir()) == []


def test_catalog_failure_rolls_back_unpublished_session_authority(
    default_project_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_id = "deadbeef"
    monkeypatch.setattr(session_store, "_new_id", lambda: session_id)
    original_create = SessionCatalog.create_session

    def fail_catalog_publish(self, **_kwargs):
        original_create(self, **_kwargs)
        raise RuntimeError("simulated catalog publication failure")

    monkeypatch.setattr(SessionCatalog, "create_session", fail_catalog_publish)

    with pytest.raises(RuntimeError, match="catalog publication"):
        service.create_session({"title": "目录发布失败"})

    expected_project = default_project_root / "2026-09-02_目录发布失败"
    catalog = SessionCatalog()
    assert catalog.get_session(session_id) is None
    assert not catalog.session_db_path(session_id).parent.exists()
    assert expected_project.is_dir()
    assert project_catalog.get_by_path(str(expected_project)) is not None


def test_workspace_grant_failure_rolls_back_unpublished_session_authority(
    default_project_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_id = "feedface"
    monkeypatch.setattr(session_store, "_new_id", lambda: session_id)
    original_factory = session_store._workspace_file_authority

    class _WriteThenFailAuthority:
        def grant_workspace_root(self, **kwargs):
            original_factory().grant_workspace_root(**kwargs)
            raise RuntimeError("simulated workspace authority failure")

    monkeypatch.setattr(
        session_store,
        "_workspace_file_authority",
        lambda: _WriteThenFailAuthority(),
    )

    with pytest.raises(RuntimeError, match="workspace authority"):
        service.create_session({"title": "授权写入失败"})

    expected_project = default_project_root / "2026-09-02_授权写入失败"
    catalog = SessionCatalog()
    assert catalog.get_session(session_id) is None
    assert not catalog.session_db_path(session_id).parent.exists()
    assert expected_project.is_dir()
    assert project_catalog.get_by_path(str(expected_project)) is not None


def test_incomplete_session_rollback_keeps_default_project_for_recovery(
    default_project_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_id = "badc0ffe"
    generated_ids = iter((session_id, "recovery2"))
    monkeypatch.setattr(session_store, "_new_id", lambda: next(generated_ids))

    original_authority_factory = session_store._workspace_file_authority

    class _FailingAuthority:
        def grant_workspace_root(self, **_kwargs):
            raise RuntimeError("simulated workspace authority failure")

    monkeypatch.setattr(
        session_store,
        "_workspace_file_authority",
        lambda: _FailingAuthority(),
    )

    original_rmtree = session_store.shutil.rmtree

    def fail_session_removal(path, *args, **kwargs):
        if Path(path).name == session_id:
            raise OSError("simulated session payload cleanup failure")
        return original_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(session_store.shutil, "rmtree", fail_session_removal)

    with pytest.raises(session_store.SessionCreationRollbackError):
        service.create_session({"title": "保留恢复目录"})

    expected_project = default_project_root / "2026-09-02_保留恢复目录"
    catalog = SessionCatalog()
    assert expected_project.is_dir()
    assert catalog.session_db_path(session_id).is_file()
    # 读取 authority 发生在最终 publish 前，因此补偿失败也没有活动 catalog locator；
    # pending tombstone 让重启后的创建入口仍能发现并继续清理，但不会公开 Session。
    tombstone = catalog.get_session_purge_tombstone(session_id)
    assert tombstone is not None and tombstone["state"] == "pending"
    assert catalog.get_session(session_id) is None

    monkeypatch.setattr(session_store.shutil, "rmtree", original_rmtree)
    monkeypatch.setattr(
        session_store,
        "_workspace_file_authority",
        original_authority_factory,
    )
    session_store._INITIALIZED_PATHS.clear()

    recovered = service.create_session({"title": "触发残留恢复"})["session"]

    assert recovered["id"] == "recovery2"
    assert not catalog.session_db_path(session_id).parent.exists()
    completed = catalog.get_session_purge_tombstone(session_id)
    assert completed is not None and completed["state"] == "completed"


def test_selected_directory_reuses_project_without_allocating_default(
    tmp_path: Path,
    default_project_root: Path,
) -> None:
    explicit = tmp_path / "chosen-project"
    explicit.mkdir()
    marker = explicit / "keep.txt"
    marker.write_text("user-owned", encoding="utf-8")

    first = service.create_session({"title": "第一轮研究", "working_dir": str(explicit)})["session"]
    second = service.create_session({"title": "另一次研究", "working_dir": str(explicit / ".")})["session"]
    assert first["id"] != second["id"]
    assert first["working_dir"] == second["working_dir"] == str(explicit.resolve())
    assert session_store.get_session(first["id"])["project_id"] == session_store.get_session(second["id"])["project_id"]
    groups = service.list_projects()["projects"]
    assert len(groups) == 1
    assert len(groups[0]["sessions"]) == 2

    assert marker.read_text(encoding="utf-8") == "user-owned"
    assert not default_project_root.exists()


@pytest.mark.parametrize("selected", ["relative/path", "missing", 42, "   "])
def test_invalid_selected_directory_does_not_fall_back(tmp_path, default_project_root, selected):
    value = str(tmp_path / "missing") if selected == "missing" else selected
    with pytest.raises(service.ApiError, match="工作目录"):
        service.create_session({"working_dir": value})
    assert not default_project_root.exists()


def test_saved_default_root_applies_only_to_future_sessions(default_project_root, tmp_path):
    original = service.create_session({"title": "原会话"})["session"]
    chosen = tmp_path / "configured-default"
    response = service.update_config({"default_projects_dir": str(chosen)})
    assert response["config"]["default_projects_dir"] == str(chosen)
    created = service.create_session({"title": "新会话"})["session"]
    assert Path(created["working_dir"]).parent == chosen
    assert session_store.get_session(original["id"])["working_dir"] == original["working_dir"]


def test_default_root_setting_rejects_invalid_paths_without_saving(tmp_path):
    from personagraph.configuration import app_settings

    for value in ("relative", str(paths.PROJECT_ROOT), str(tmp_path / "ordinary-file")):
        (tmp_path / "ordinary-file").write_text("keep")
        with pytest.raises(service.ApiError):
            service.update_config({"default_projects_dir": value})
    assert app_settings.get_setting("default_projects_dir") is None


def test_environment_default_root_still_overrides_saved_setting(default_project_root, tmp_path, monkeypatch):
    override = tmp_path / "env-default"
    monkeypatch.setenv("PERSONAGRAPH_DEFAULT_PROJECTS_DIR", str(override))
    response = service.update_config({"default_projects_dir": str(tmp_path / "saved-default")})
    assert response["config"]["default_projects_dir_managed"] is True
    assert response["config"]["default_projects_dir"] == str(override)
    session = service.create_session({"title": "环境配置"})["session"]
    assert Path(session["working_dir"]).parent == override


def test_selected_directory_grant_failure_keeps_user_files(default_project_root, tmp_path, monkeypatch):
    chosen = tmp_path / "user-files"
    chosen.mkdir()
    marker = chosen / "keep.txt"
    marker.write_text("keep")

    def fail_grant(**_kwargs):
        raise RuntimeError("grant failed")

    monkeypatch.setattr(session_store, "_grant_workspace_read_authority", fail_grant)
    with pytest.raises(RuntimeError, match="grant failed"):
        service.create_session({"working_dir": str(chosen)})
    assert marker.read_text() == "keep"
    assert not default_project_root.exists()
    assert not session_store.list_sessions()


def test_generated_session_id_collision_never_rolls_back_the_existing_session(
    tmp_path: Path,
    default_project_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_id = "samebeef"
    existing_project = tmp_path / "existing-project"
    existing_project.mkdir()
    monkeypatch.setattr(session_store, "_new_id", lambda: session_id)
    created = session_store.create_session(
        "Entelecheia",
        title="existing",
        working_dir=str(existing_project),
    )
    with session_store.session_database_scope(created):
        append_test_turn(created, "user", "must survive collision")
    catalog = SessionCatalog()
    existing_database = catalog.session_db_path(session_id)

    generated_ids = iter((session_id, "new0beef"))
    monkeypatch.setattr(session_store, "_new_id", lambda: next(generated_ids))

    replacement = service.create_session({"title": "冲突不得误删"})["session"]

    assert replacement["id"] == "new0beef"
    assert catalog.get_session(session_id) is not None
    assert existing_database.is_file()
    with session_store.session_database_scope(session_id):
        assert session_store.get_turns(session_id)[0]["content"] == (
            "must survive collision"
        )
    assert (default_project_root / "2026-09-02_冲突不得误删").is_dir()


def test_existing_unpublished_session_partition_is_never_claimed_or_removed(
    default_project_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_id = "stale001"
    generated_ids: list[str] = []

    def same_claimed_id() -> str:
        generated_ids.append(session_id)
        return session_id

    monkeypatch.setattr(session_store, "_new_id", same_claimed_id)
    catalog = SessionCatalog()
    session_directory = catalog.session_db_path(session_id).parent
    session_directory.mkdir(parents=True)
    marker = session_directory / "owner-marker"
    marker.write_text("belongs to another creation", encoding="utf-8")

    with pytest.raises(session_store.SessionStoreError):
        service.create_session({"title": "已有分区不得误删"})

    assert marker.read_text(encoding="utf-8") == "belongs to another creation"
    assert catalog.get_session(session_id) is None
    assert len(generated_ids) == session_store._MAX_SESSION_ID_CLAIM_ATTEMPTS
    assert list(default_project_root.iterdir()) == []


def test_product_session_is_published_only_after_workspace_authority_exists(
    default_project_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_id = "publish1"
    monkeypatch.setattr(session_store, "_new_id", lambda: session_id)
    original_create = SessionCatalog.create_session
    observed: list[tuple[int, int]] = []

    def inspect_before_publish(self, **kwargs):
        assert self.get_session(session_id) is None
        with sqlite3.connect(self.session_db_path(session_id)) as connection:
            payload_count = int(
                connection.execute(
                    "SELECT COUNT(*) FROM sessions WHERE id=?",
                    (session_id,),
                ).fetchone()[0]
            )
            grant_count = int(
                connection.execute(
                    "SELECT COUNT(*) FROM session_workspace_read_grants "
                    "WHERE session_id=? AND revoked_at IS NULL",
                    (session_id,),
                ).fetchone()[0]
            )
        observed.append((payload_count, grant_count))
        return original_create(self, **kwargs)

    monkeypatch.setattr(SessionCatalog, "create_session", inspect_before_publish)

    created = service.create_session({"title": "最终发布顺序"})["session"]

    assert created["id"] == session_id
    assert observed == [(1, 1)]
    assert (
        default_project_root / "2026-09-02_最终发布顺序"
    ).is_dir()
