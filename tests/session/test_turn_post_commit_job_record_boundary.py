"""轮次提交后任务持久化组合的依赖边界。"""

from __future__ import annotations

import ast
from pathlib import Path

from personagraph.session import store
from personagraph.session.persistence.turns import post_commit_jobs


def _imported_modules(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
            imported.extend(alias.name for alias in node.names)
    return imported


def test_post_commit_records_do_not_reach_back_through_public_or_runtime_layers() -> None:
    """提取出的协调器保持在公开会话 Store 端口之下。"""

    imported_modules = _imported_modules(Path(post_commit_jobs.__file__))
    forbidden = ("store", "api", "runtime", "graph", "memory", "context")

    assert "personagraph.session.store" not in post_commit_jobs.__dict__
    assert not any(
        forbidden_name in module
        for module in imported_modules
        for forbidden_name in forbidden
    )


def test_store_post_commit_claim_remains_a_thin_compatibility_facade(monkeypatch) -> None:
    """持久化层拥有组合职责，运行时则保留相同的 Store 方法。"""

    captured: dict[str, object] = {}
    expected = [{"job_id": "job_1", "status": "processing"}]

    def _claim(deps, **kwargs):
        captured["deps"] = deps
        captured["kwargs"] = kwargs
        return expected

    monkeypatch.setattr(post_commit_jobs, "claim_due_turn_post_commit_jobs", _claim)

    assert store.claim_due_turn_post_commit_jobs(
        session_id="session_1",
        worker_id="worker_1",
        lease_seconds=60,
        limit=3,
    ) == expected
    assert captured["kwargs"] == {
        "session_id": "session_1",
        "worker_id": "worker_1",
        "lease_seconds": 60,
        "limit": 3,
    }
