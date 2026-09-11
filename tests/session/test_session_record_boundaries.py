import ast
from pathlib import Path

import pytest

from personagraph.session import store
from personagraph.session.persistence.metadata import sessions


def test_session_records_depend_only_on_store_support():
    tree = ast.parse(Path(sessions.__file__).read_text(encoding="utf-8"))
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
        for forbidden in ("paths", "graph", "memory", "context_store", "folder")
    )


def test_invalid_session_status_fails_closed():
    with pytest.raises(ValueError, match="invalid session status"):
        store.list_sessions(status="unknown")
