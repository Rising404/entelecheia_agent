from __future__ import annotations

import ast
from pathlib import Path


RETRIEVAL_ROOT = Path(__file__).resolve().parents[2] / "src" / "personagraph" / "retrieval"


def _imports(relative_path: Path | str) -> set[str]:
    tree = ast.parse((RETRIEVAL_ROOT / relative_path).read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(f"{'.' * node.level}{node.module or ''}")
    return imported


def test_foundation_composition_root_does_not_depend_on_runtime_or_authority_details():
    forbidden_prefixes = (
        "personagraph.graph",
        "personagraph.runtime",
        "personagraph.memory",
        "personagraph.context",
        "personagraph.session",
        "..graph",
        "..runtime",
        "..memory",
        "..context",
        "..session",
    )
    imports = _imports("foundation.py")
    assert not any(
        imported.startswith(forbidden_prefixes)
        for imported in imports
    )


def test_only_document_source_adapter_may_directly_depend_on_authority_domains():
    forbidden_prefixes = (
        "personagraph.graph",
        "personagraph.runtime",
        "personagraph.memory",
        "personagraph.context",
        "personagraph.session",
        "..graph",
        "..runtime",
        "..memory",
        "..context",
        "..session",
    )
    for path in RETRIEVAL_ROOT.rglob("*.py"):
        relative_path = path.relative_to(RETRIEVAL_ROOT)
        if relative_path == Path("sources/document.py"):
            continue
        imports = _imports(relative_path)
        assert not any(
            imported.startswith(forbidden_prefixes)
            for imported in imports
        ), relative_path


def test_retrieval_package_does_not_store_source_content_schema():
    source = (RETRIEVAL_ROOT / "contracts.py").read_text(encoding="utf-8")
    assert "class RetrievalUnit" in source
    unit_section = source.split("class RetrievalUnit", maxsplit=1)[1].split("class SourceUnit", maxsplit=1)[0]
    assert "content:" not in unit_section


def test_file_retrieval_contract_does_not_depend_on_tool_or_runtime_layers():
    imports = _imports("tooling/contracts.py")
    forbidden_prefixes = (
        "personagraph.runtime",
        "personagraph.tools",
        "..runtime",
        "..tools",
    )
    assert not any(
        imported.startswith(forbidden_prefixes)
        for imported in imports
    )


def test_retrieval_modules_have_one_canonical_owner():
    legacy_paths = (
        "candidates/scope.py",
        "candidates/preparation.py",
        "candidates/refs.py",
        "candidates/authority.py",
        "file_candidates.py",
        "file_candidate_preparation.py",
        "file_candidate_refs.py",
        "tool_contracts.py",
        "tool_service.py",
        "selection.py",
        "reranking.py",
        "recovery.py",
        "diagnostics.py",
        "generation_admin.py",
        "real_smoke.py",
        "corpus.py",
        "generation.py",
        "generation_rollout.py",
        "backfill.py",
        "outbox.py",
        "sync.py",
        "maintenance.py",
        "source_identity.py",
        "source_events.py",
        "source_adapters.py",
        "source_index_coverage.py",
        "document_index_coverage.py",
        "current_session_index_coverage.py",
        "session.py",
        "session_post_commit.py",
        "methods.py",
        "model_assets.py",
        "token_estimation.py",
    )
    canonical_paths = (
        "tooling/contracts.py",
        "orchestration/selection.py",
        "orchestration/reranking.py",
        "orchestration/recovery.py",
        "operations/diagnostics.py",
        "operations/generation_admin.py",
        "operations/real_smoke.py",
        "lifecycle/corpus.py",
        "lifecycle/generation.py",
        "lifecycle/rollout.py",
        "lifecycle/backfill.py",
        "lifecycle/outbox.py",
        "lifecycle/sync.py",
        "lifecycle/maintenance.py",
        "sources/identity.py",
        "sources/events.py",
        "sources/document.py",
        "sources/picture.py",
        "sources/picture_publication.py",
        "sources/coverage/base.py",
        "sources/coverage/document.py",
        "sources/coverage/picture.py",
        "sources/coverage/current_session.py",
        "sources/session/contracts.py",
        "sources/session/projection.py",
        "sources/session/composition.py",
        "sources/session/lifecycle.py",
        "sources/session/post_commit.py",
        "indexing/methods.py",
        "indexing/model_assets.py",
        "indexing/token_estimation.py",
        "tooling/contracts.py",
        "tooling/service/audit.py",
        "tooling/service/facade.py",
        "tooling/service/file.py",
        "tooling/service/history.py",
        "tooling/service/ports.py",
        "tooling/service/projection.py",
    )

    assert not any((RETRIEVAL_ROOT / path).exists() for path in legacy_paths)
    assert all((RETRIEVAL_ROOT / path).is_file() for path in canonical_paths)


def test_production_retrieval_modules_do_not_depend_on_operations():
    operations_root = RETRIEVAL_ROOT / "operations"
    for path in RETRIEVAL_ROOT.rglob("*.py"):
        if path == operations_root or operations_root in path.parents:
            continue
        relative_path = path.relative_to(RETRIEVAL_ROOT)
        imports = _imports(relative_path)
        assert not any(
            imported.startswith("personagraph.retrieval.operations")
            or imported.lstrip(".").startswith("operations")
            for imported in imports
        ), relative_path


def test_document_workers_receive_project_storage_without_reading_context():
    """底层 worker/generation 只消费组合层注入的 Project 能力。"""

    forbidden = "personagraph.workspace.storage.context"
    for relative_path in (
        Path("operations/ingestion_index.py"),
        Path("operations/document_generation.py"),
    ):
        assert forbidden not in _imports(relative_path), relative_path


def test_workspace_ingestion_worker_and_owner_do_not_import_retrieval():
    ingestion_root = RETRIEVAL_ROOT.parent / "workspace" / "ingestion"
    for name in ("worker.py", "worker_errors.py", "indexing_ports.py", "execution.py"):
        source = (ingestion_root / name).read_text(encoding="utf-8")
        tree = ast.parse(source)
        imports = {
            node.module or ""
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
        }
        imports.update(
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        )
        assert not any("retrieval" in name.split(".") for name in imports), name

    generation = _imports(Path("operations/document_generation.py"))
    assert not any("ingestion" in name.split(".") for name in generation)


def test_lifecycle_does_not_depend_on_online_or_tooling_layers():
    forbidden_modules = {
        "foundation",
        "operations",
        "orchestration",
        "profile",
        "service",
        "session",
        "session_post_commit",
        "tooling",
    }
    for path in (RETRIEVAL_ROOT / "lifecycle").glob("*.py"):
        imports = _imports(path.relative_to(RETRIEVAL_ROOT))
        assert not any(
            imported.startswith("personagraph.retrieval.")
            and imported.removeprefix("personagraph.retrieval.").split(".", 1)[0]
            in forbidden_modules
            or imported.lstrip(".").split(".", 1)[0] in forbidden_modules
            for imported in imports
        ), path.name


def test_indexing_does_not_depend_on_lifecycle_or_online_layers():
    forbidden_modules = {
        "foundation",
        "lifecycle",
        "operations",
        "orchestration",
        "profile",
        "service",
        "sources",
        "tooling",
    }
    for path in (RETRIEVAL_ROOT / "indexing").glob("*.py"):
        imports = _imports(path.relative_to(RETRIEVAL_ROOT))
        assert not any(
            imported.startswith("personagraph.retrieval.")
            and imported.removeprefix("personagraph.retrieval.").split(".", 1)[0]
            in forbidden_modules
            or imported.lstrip(".").split(".", 1)[0] in forbidden_modules
            for imported in imports
        ), path.name


def test_tooling_service_does_not_depend_on_runtime_or_authority_domains():
    forbidden_prefixes = (
        "personagraph.graph",
        "personagraph.runtime",
        "personagraph.memory",
        "personagraph.session",
        "....graph",
        "....runtime",
        "....memory",
        "....session",
    )
    for path in (RETRIEVAL_ROOT / "tooling" / "service").glob("*.py"):
        imports = _imports(path.relative_to(RETRIEVAL_ROOT))
        assert not any(
            imported.startswith(forbidden_prefixes)
            for imported in imports
        ), path.name
