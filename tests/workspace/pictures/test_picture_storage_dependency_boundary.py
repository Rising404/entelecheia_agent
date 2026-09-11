from __future__ import annotations

import ast
from pathlib import Path
import subprocess
import sys


_PICTURE_ROOT = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "personagraph"
    / "workspace"
    / "pictures"
)

_FORBIDDEN_DEPENDENCIES = {
    "input_processing",
    "retrieval",
    "runtime",
    "session",
    "tools",
}


def test_picture_storage_has_no_upward_domain_imports_or_transaction_control() -> None:
    paths = (
        _PICTURE_ROOT / "__init__.py",
        _PICTURE_ROOT / "admission.py",
        _PICTURE_ROOT / "contracts.py",
        _PICTURE_ROOT / "storage" / "schema.py",
        _PICTURE_ROOT / "storage" / "repository.py",
    )
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported_modules: set[str] = set()
        controlled_transactions: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_modules.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_modules.add(node.module)
            elif (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in {"commit", "rollback"}
            ):
                controlled_transactions.append(node.func.attr)
        assert not {
            component
            for module in imported_modules
            for component in module.split(".")
            if component in _FORBIDDEN_DEPENDENCIES
        }, path
        assert controlled_transactions == [], path


def test_repository_does_not_own_admission_generation_or_public_facade() -> None:
    repository_path = _PICTURE_ROOT / "storage" / "repository.py"
    tree = ast.parse(repository_path.read_text(encoding="utf-8"))
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    functions = {
        node.name for node in tree.body if isinstance(node, ast.FunctionDef)
    }
    assert {"datetime", "uuid"}.isdisjoint(imports)
    assert "ensure_picture_in_transaction" not in functions
    assert "ensure_picture_unit_in_transaction" not in functions

    import personagraph.workspace.pictures as pictures

    assert not hasattr(pictures, "initialize_picture_schema")
    assert not hasattr(pictures, "insert_picture_if_absent_in_transaction")
    assert not hasattr(pictures, "PictureRepositoryError")
    assert not hasattr(pictures, "PicturePersistenceConflict")
    assert not hasattr(pictures, "ExactPictureSourcePort")
    assert not hasattr(pictures, "PictureRaster")
    assert not hasattr(pictures, "WholeImageRasterizer")
    assert callable(pictures.get_picture)
    assert callable(pictures.picture_unit_binding_is_current)


def test_picture_observation_publication_contract_does_not_import_retrieval() -> None:
    path = _PICTURE_ROOT / "observations" / "publication.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported_modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    } | {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }

    assert not any(
        "retrieval" in module.split(".") for module in imported_modules
    )


def test_workspace_database_cold_import_does_not_load_picture_application() -> None:
    facade = ast.parse((_PICTURE_ROOT / "__init__.py").read_text(encoding="utf-8"))
    assert not any(
        isinstance(node, ast.ImportFrom)
        and node.module == "project_publication"
        for node in ast.walk(facade)
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import personagraph.workspace.storage.database; "
            "assert 'personagraph.workspace.pictures.project_publication' "
            "not in sys.modules",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
