"""会话生命周期持久化组合的依赖边界。"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from personagraph.session import lifecycle_store_facade, store
from personagraph.session.persistence.metadata import folders, lifecycle, sessions


def test_lifecycle_records_do_not_reach_back_through_the_public_store() -> None:
    """提取出的生命周期门面保持在 API、运行时和公开会话层之下。"""

    tree = ast.parse(Path(lifecycle.__file__).read_text(encoding="utf-8"))
    imported_modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported_modules.append(node.module or "")
            imported_modules.extend(alias.name for alias in node.names)

    forbidden = ("store", "api", "runtime", "graph", "memory", "context")
    assert not any(
        forbidden_name in module
        for module in imported_modules
        for forbidden_name in forbidden
    )


def test_lifecycle_rejects_retired_arguments_before_resolving_storage(monkeypatch):
    def unexpected_storage_access():
        pytest.fail("retired arguments must be rejected before resolving storage")

    monkeypatch.setattr(store, "_uses_explicit_database_override", unexpected_storage_access)
    for name, args in (
        ("trash_session", ("session",)),
        ("restore_session", ("session",)),
        ("purge_session", ("session",)),
        ("set_folder_status", ("folder", "trashed")),
    ):
        owners = (
            store,
            lifecycle_store_facade.SessionLifecycleStoreFacade,
            lifecycle,
            folders if name == "set_folder_status" else sessions,
        )
        for owner in owners:
            assert "retrieval_data_version" not in inspect.signature(
                getattr(owner, name)
            ).parameters
        with pytest.raises(TypeError, match="retrieval_data_version"):
            getattr(store, name)(*args, retrieval_data_version="retired")
