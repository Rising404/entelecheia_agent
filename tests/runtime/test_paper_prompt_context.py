from __future__ import annotations

import ast
import json
from pathlib import Path

from personagraph.l2.task_execution import paper_prompt_context
from personagraph.l2.task_execution.paper_prompt_context import PaperAttemptContext
from personagraph.l2.task_graph.paper_resource_contracts import (
    PaperDocumentBinding,
    PaperOutlineEntry,
    PaperResourceSnapshot,
)


def _snapshot() -> PaperResourceSnapshot:
    return PaperResourceSnapshot.create(
        session_id="session-private-alpha",
        task_id="task-private-alpha",
        bound_graph_revision=3,
        bound_task_state_version=7,
        retrieval_data_version_id="rdv-alpha",
        retrieval_generation_fingerprint="retrieval-generation-alpha",
        encoder_fingerprint="deterministic-lexical@1",
        documents=(
            PaperDocumentBinding(
                paper_key="P1",
                document_id="document-private-alpha",
                source_version_id="document-version-private-alpha",
                title="Bounded paper title",
                source_sha256="a" * 64,
                processing_status="complete",
                admitted_chunk_count=2,
                chunk_manifest_sha256="b" * 64,
                admitted_text_page_start=1,
                admitted_text_page_end=4,
                outline=(
                    PaperOutlineEntry(
                        outline_key="P1:S1",
                        title="Method",
                        start_handle="P1:C1",
                        page_start=1,
                        page_end=4,
                    ),
                ),
            ),
        ),
    )


def _relative_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.level > 0
        and node.module is not None
    }


def _imported_names(path: Path, module: str) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module is not None
        and (node.module == module or node.module.endswith(f".{module}"))
        for alias in node.names
    }


def test_prompt_context_projects_a_bounded_payload_without_tool_authority() -> None:
    snapshot = _snapshot()

    context = PaperAttemptContext.from_snapshot(snapshot)
    payload = context.to_dict()
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True)

    assert payload["snapshot_id"] == snapshot.snapshot_id
    assert payload["manifest_sha256"] == snapshot.manifest_sha256
    assert payload["papers"] == [
        {
            "alias": "P1",
            "title": "Bounded paper title",
            "page_range": {"start": 1, "end": 4},
            "processing_status": "complete",
            "diagnostic_codes": [],
            "outline": [
                {
                    "outline_key": "P1:S1",
                    "title": "Method",
                    "start_handle": "P1:C1",
                    "page_start": 1,
                    "page_end": 4,
                }
            ],
            "outline_truncated": False,
        }
    ]
    for private_value in (
        snapshot.session_id,
        snapshot.task_id,
        snapshot.documents[0].document_id,
        snapshot.documents[0].source_version_id,
        snapshot.documents[0].source_sha256,
    ):
        assert private_value not in serialized


def test_prompt_context_projects_parser_partial_without_private_diagnostic_detail() -> None:
    paper = paper_prompt_context.PaperAttemptPaper(
        alias="P1",
        title="Partially parsed paper",
        page_start=1,
        page_end=4,
        processing_status="partial",
        diagnostic_codes=("parser_partial",),
    )

    assert paper.to_dict()["diagnostic_codes"] == ["parser_partial"]


def test_prompt_context_and_its_runtime_consumers_depend_on_the_narrow_owner() -> None:
    task_execution_directory = Path(paper_prompt_context.__file__).parent
    package_directory = task_execution_directory.parents[1]

    assert _relative_imports(Path(paper_prompt_context.__file__)) == {
        "task_graph.paper_resource_contracts"
    }
    for source in (
            package_directory / "l2/task_execution/attempts/input_projection.py",
            package_directory / "l2/task_execution/work_run/turn_request_contracts.py",
        package_directory / "l2/task_execution/task_graph/contracts.py",
        package_directory / "l2/task_execution/task_node/tool_runtime_contracts.py",
        package_directory / "l2/task_execution/task_node/tool_runtime_policy.py",
    ):
        assert 'PaperAttemptContext' in _imported_names(
            source,
            "paper_prompt_context",
        )
        assert 'PaperAttemptContext' not in _imported_names(source, "paper_tools")
