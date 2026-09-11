"""该包解耦状态的机械化守卫。

目录认知必须可供任意层使用，因此不能反向依赖调用它的层。一旦该包开始导入
运行时、存储或 API，这些测试便会失败。
"""

from __future__ import annotations

import ast
from importlib.util import find_spec
from pathlib import Path

import pytest


WORKSPACE_ROOT = Path(__file__).resolve().parents[2] / "src" / "personagraph" / "workspace"

# 此处有意不包含 `security`：路径拒绝是各层共享的硬边界，在这里重新实现会
# 形成第二个事实来源。
FORBIDDEN_PREFIXES = (
    "personagraph.graph", "personagraph.runtime", "personagraph.api",
    "personagraph.cli", "personagraph.session", "personagraph.memory",
    "personagraph.retrieval", "personagraph.model_io", "personagraph.tools",
)

RELATIVE_FORBIDDEN = (
    "..graph", "..runtime", "..api", "..cli", "..session",
    "..memory", "..retrieval", "..model_io", "..tools",
)


def _modules() -> list[Path]:
    return sorted(WORKSPACE_ROOT.rglob("*.py"))


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            found.add(f"{'.' * node.level}{node.module or ''}")
    return found


def _module_exists(module_name: str) -> bool:
    try:
        return find_spec(module_name) is not None
    except ModuleNotFoundError:
        return False


def test_the_package_has_modules_to_guard():
    assert _modules(), "workspace package is missing; the guard would pass vacuously"


@pytest.mark.parametrize("module", _modules(), ids=lambda p: p.name)
def test_workspace_never_reaches_into_calling_layers(module):
    for name in _imports(module):
        if module == WORKSPACE_ROOT / "ingestion" / "composition.py":
            # 唯一默认装配边界可以注入 Session/检索；核心及其他上层依赖仍禁止。
            if name.startswith(("personagraph.session", "personagraph.retrieval")):
                continue
        assert not name.startswith(FORBIDDEN_PREFIXES), f"{module.name} imports {name}"
        assert not name.startswith(RELATIVE_FORBIDDEN), f"{module.name} imports {name}"


@pytest.mark.parametrize(
    "contract_module",
    (
        WORKSPACE_ROOT / "binding" / "contracts.py",
        WORKSPACE_ROOT / "discovery" / "contracts.py",
        WORKSPACE_ROOT / "files" / "contracts.py",
        WORKSPACE_ROOT / "ingestion" / "contracts.py",
    ),
    ids=("binding", "discovery", "files", "ingestion"),
)
def test_public_contracts_depend_on_the_standard_library_only(
    contract_module: Path,
) -> None:
    """跨包传播的 Workspace 结果类型必须保持可移植。"""

    for name in _imports(contract_module):
        assert not name.startswith("personagraph"), f"contracts imports {name}"
        assert not name.startswith("."), f"contracts imports {name}"


def test_workspace_root_is_navigation_only() -> None:
    import personagraph.workspace as workspace

    assert not hasattr(workspace, "find")
    assert not hasattr(workspace, "build_overview")
    assert not hasattr(workspace, "ensure_or_open_layout")


def test_workspace_file_sql_primitives_do_not_own_connections_or_filesystem() -> None:
    repository = WORKSPACE_ROOT / "files" / "storage" / "repository.py"

    assert repository.is_file()
    imports = _imports(repository)
    assert "sqlite3" in imports
    assert "..storage.database" not in imports
    assert "...storage.database" not in imports
    assert "os" not in imports
    assert "pathlib" not in imports
    source = repository.read_text(encoding="utf-8")
    assert "BEGIN " not in source
    assert ".commit(" not in source
    assert ".rollback(" not in source
    assert "FileRegistrationConflict" not in source


def test_workspace_authority_packages_exist_at_their_canonical_paths() -> None:
    expected = (
        WORKSPACE_ROOT / "storage" / "database.py",
        WORKSPACE_ROOT / "storage" / "context.py",
        WORKSPACE_ROOT / "storage" / "schema.py",
        WORKSPACE_ROOT / "files" / "contracts.py",
        WORKSPACE_ROOT / "files" / "observation.py",
        WORKSPACE_ROOT / "files" / "admission.py",
        WORKSPACE_ROOT / "files" / "uploads.py",
        WORKSPACE_ROOT / "files" / "storage" / "repository.py",
    )

    assert all(path.is_file() for path in expected)


def test_workspace_file_facade_exposes_authority_not_a_repository_alias() -> None:
    from personagraph.workspace import files

    assert hasattr(files, "WorkspaceFileAuthority")
    assert not hasattr(files, "ProjectFileRepository")


@pytest.mark.parametrize(
    "module_name",
    (
        "personagraph.workspace.contracts",
        "personagraph.workspace.finding",
        "personagraph.workspace.layout",
        "personagraph.workspace.overview",
        "personagraph.workspace.reserved",
        "personagraph.workspace.ripgrep",
        "personagraph.workspace.session_file_authority",
        "personagraph.input_processing.documents.storage.context",
        "personagraph.input_processing.documents.storage.database",
        "personagraph.input_processing.documents.storage.files",
        "personagraph.input_processing.documents.storage.schema",
        "personagraph.input_processing.documents.storage.uploads",
        "personagraph.input_processing.documents.storage.repository",
        "personagraph.input_processing.documents.storage",
        "personagraph.workspace.documents.ingest.contracts",
        "personagraph.workspace.documents.ingest.repository",
        "personagraph.workspace.documents.ingest.worker",
        "personagraph.workspace.documents.ingest.supervisor",
    ),
)
def test_retired_workspace_modules_have_no_compatibility_shell(
    module_name: str,
) -> None:
    assert not _module_exists(module_name)


def test_file_preparation_core_has_no_model_candidate_parameters() -> None:
    for path in (WORKSPACE_ROOT / "ingestion").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.arg):
                assert node.arg not in {"candidate_id", "candidate_ids", "alias", "safe_alias"}, (
                    path, node.arg,
                )
