import ast
from pathlib import Path

from personagraph.session import store
from personagraph.session.persistence.metadata import folders


def test_folder_records_depend_only_on_store_support():
    tree = ast.parse(Path(folders.__file__).read_text(encoding="utf-8"))
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
    assert "store" not in imported
    assert not any(
        forbidden in module
        for module in imported
        for forbidden in ("paths", "graph", "memory", "context_store", "session_records")
    )


def test_folder_status_rejects_unknown_value():
    folder_id = store.create_folder("root")
    assert store.set_folder_status(folder_id, "unknown") == (False, "invalid_status")
    assert store.get_folder(folder_id)["status"] == "active"


def test_the_suite_never_writes_into_the_real_state_directory():
    """被遗忘的重定向不得静默访问开发者自己的数据。

    conftest 中自动启用的隔离确保了这一点；若无此隔离，直接调用存储的测试
    可能写入真实项目目录或会话拥有的 ``session.sqlite``。
    """
    from personagraph.configuration.paths import STATE_DIR
    from personagraph.session import store as session_store

    assert not str(session_store.DB_PATH).startswith(str(STATE_DIR))
