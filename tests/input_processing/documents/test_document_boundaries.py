"""该切片所建立解耦关系的机械化守卫。

只有读取器无法触及其他组件时，契约才能保持格式中立。一旦读取器开始写入
存储、调用模型或了解附件，这些测试便会失败。
"""

from __future__ import annotations

import ast
from importlib.util import resolve_name
from pathlib import Path

import pytest


DOCUMENTS_ROOT = (
    Path(__file__).resolve().parents[3]
    / "src" / "personagraph" / "input_processing" / "documents"
)
SOURCE_ROOT = DOCUMENTS_ROOT.parents[2]
DOCUMENTS_PACKAGE = "personagraph.input_processing.documents"

FORBIDDEN_PRODUCT_PREFIXES = (
    "personagraph.api",
    "personagraph.retrieval",
    "personagraph.runtime",
    "personagraph.session",
    "personagraph.tools",
    "personagraph.workspace",
)


def _modules() -> list[Path]:
    return sorted(path for path in DOCUMENTS_ROOT.rglob("*.py"))


def _reader_modules() -> list[Path]:
    readers_root = DOCUMENTS_ROOT / "readers"
    return sorted(path for path in readers_root.rglob("*.py"))


def _module_name(path: Path) -> str:
    relative = path.relative_to(SOURCE_ROOT).with_suffix("")
    parts = relative.parts
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _imports(path: Path, *, module_name: str | None = None) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    resolved_module = module_name or _module_name(path)
    package = (
        resolved_module
        if path.name == "__init__.py"
        else resolved_module.rpartition(".")[0]
    )
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                if node.module:
                    found.add(node.module)
                continue
            if node.module:
                found.add(resolve_name(f"{'.' * node.level}{node.module}", package))
            else:
                found.update(
                    resolve_name(f"{'.' * node.level}{alias.name}", package)
                    for alias in node.names
                )
    return found


def _matches_prefix(module: str, prefix: str) -> bool:
    return module == prefix or module.startswith(f"{prefix}.")


@pytest.mark.parametrize(
    "path",
    _modules(),
    ids=lambda path: str(path.relative_to(DOCUMENTS_ROOT)),
)
def test_input_processing_documents_do_not_own_product_state(path: Path):
    for name in _imports(path):
        assert not any(
            _matches_prefix(name, prefix)
            for prefix in FORBIDDEN_PRODUCT_PREFIXES
        ), f"{path.relative_to(DOCUMENTS_ROOT)} imports product owner {name}"


def test_relative_imports_are_resolved_before_boundary_checks(tmp_path: Path):
    module = tmp_path / "fake_reader.py"
    module.write_text("from ....session import store\n", encoding="utf-8")

    imports = _imports(
        module,
        module_name=(
            "personagraph.input_processing.documents.readers.fake_reader"
        ),
    )

    assert imports == {"personagraph.session"}
    assert any(
        _matches_prefix(name, prefix)
        for prefix in FORBIDDEN_PRODUCT_PREFIXES
        for name in imports
    )


@pytest.mark.parametrize(
    "path",
    _reader_modules(),
    ids=lambda path: str(path.relative_to(DOCUMENTS_ROOT)),
)
def test_readers_do_not_reach_across_to_attachment_intake(path: Path):
    """文档与附件是同级模块；二者都只依赖契约。

    将二者耦合会重新制造本次拆分旨在避免的混淆——把文件附到消息上，
    并不等于将其准入语料库。
    """
    for name in _imports(path):
        assert "attachments" not in name, f"{path.name} imports {name}"


def test_contracts_depend_on_nothing_but_the_standard_library():
    """所有格式共享的词汇本身不能专属于某种格式或基础设施。"""
    for name in _imports(DOCUMENTS_ROOT / "contracts.py"):
        assert not name.startswith((".", "personagraph")), name


def test_readers_perform_no_io_beyond_reading_their_own_file():
    banned = ("sqlite3", "httpx", "requests", "socket")
    for path in _modules():
        if "readers" not in str(path):
            continue
        for name in _imports(path):
            assert not name.startswith(banned), f"{path.name} imports {name}"
            if name.startswith("subprocess"):
                assert path.name == "legacy_office.py", (
                    f"only the bounded legacy Office converter may import {name}"
                )


def test_the_public_surface_is_the_package_not_its_internal_layout():
    from personagraph.input_processing import documents

    for symbol in documents.__all__:
        assert hasattr(documents, symbol), symbol
    assert {
        "DocumentLocator",
        "DocumentElement",
        "PreparedDocumentIngest",
        "prepare_document_path",
        "read_document",
    } <= set(documents.__all__)
    assert "ingest_document_path" not in documents.__all__


def test_ingest_worker_and_maintenance_have_one_document_owner():
    personagraph_root = DOCUMENTS_ROOT.parents[1]

    assert not (DOCUMENTS_ROOT / "ingest").exists()
    ingestion = personagraph_root / "workspace" / "ingestion"
    assert (ingestion / "worker.py").is_file()
    assert (ingestion / "lifecycle.py").is_file()
    assert (ingestion / "storage" / "repository.py").is_file()
    assert not (personagraph_root / "workspace" / "documents" / "ingest" / "worker.py").exists()
    assert not (personagraph_root / "retrieval" / "operations" / "document_ingest.py").exists()
    assert (personagraph_root / "retrieval" / "operations" / "ingestion_index.py").is_file()
    assert (personagraph_root / "retrieval" / "operations" / "document_maintenance.py").is_file()


def test_turn_attachment_mounting_has_one_document_owner():
    personagraph_root = DOCUMENTS_ROOT.parents[1]

    assert not (DOCUMENTS_ROOT / "mounting").exists()
    assert (
        personagraph_root
        / "workspace"
        / "documents"
        / "admission"
        / "turn_inputs.py"
    ).is_file()


def test_dead_repository_query_interfaces_are_retired():
    from personagraph.workspace.documents import application as documents

    assert not hasattr(documents, "summarize_doc")
    assert not hasattr(documents, "doc_search")
