"""Runtime 根目录只保留跨 lane 的执行内核。"""

from __future__ import annotations

from pathlib import Path

import personagraph.runtime as runtime_package
import personagraph.runtime.entry as entry_package


def test_runtime_root_contains_only_cross_lane_kernel_modules() -> None:
    runtime_root = Path(runtime_package.__file__).resolve().parent
    modules = {
        path.name
        for path in runtime_root.glob("*.py")
        if path.is_file()
    }

    assert modules == {
        "__init__.py",
        "concurrency.py",
        "turn_deadline.py",
        "turn_events.py",
    }


def test_post_commit_kernel_has_one_canonical_package_owner() -> None:
    runtime_root = Path(runtime_package.__file__).resolve().parent
    post_commit_root = runtime_root / "post_commit"

    assert not (runtime_root / "turn_post_commit_jobs.py").exists()
    assert {
        path.name
        for path in post_commit_root.glob("*.py")
        if path.is_file()
    } == {
        "__init__.py",
        "contracts.py",
        "lifecycle.py",
        "runner.py",
        "scheduler.py",
        "session_retrieval_recovery.py",
        "settlement.py",
    }


def test_entry_root_contains_only_the_public_facade_and_shared_orchestration() -> None:
    entry_root = Path(entry_package.__file__).resolve().parent

    assert {
        path.name for path in entry_root.glob("*.py") if path.is_file()
    } == {
        "__init__.py",
        "application.py",
        "ports.py",
    }


def test_entry_lazy_facade_exposes_only_public_use_cases() -> None:
    assert entry_package.__all__ == [
        "accept_entry_turn",
        "execute_accepted_entry_turn",
        "resume_active_l1_entry_turn",
        "run_entry_turn",
    ]
    assert not hasattr(entry_package, "_OWNED_SUBMODULES")
    assert not hasattr(entry_package, "Any")
    assert not hasattr(entry_package, "ModuleType")
    assert not hasattr(entry_package, "TYPE_CHECKING")
    assert not hasattr(entry_package, "import_module")
    try:
        entry_package.__getattr__("task_catalog")
    except AttributeError:
        pass
    else:  # pragma: no cover - 失败分支只为给断言提供清楚的错误
        raise AssertionError("Entry facade must not resolve private submodules")
