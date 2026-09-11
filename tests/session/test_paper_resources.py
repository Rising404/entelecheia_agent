from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
from pydantic import ValidationError

from personagraph.l2.task_graph import paper_resource_contracts
from personagraph.l2.task_graph.paper_resource_contracts import (
    PaperDocumentBinding,
    PaperOutlineEntry,
    PaperResourceSnapshot,
)


def _document(
    paper_key: str = "P1",
    *,
    document_id: str = "doc-alpha",
    source_version_id: str = "docv-alpha",
    title: str = "Alpha Paper",
    source_sha256: str = "a" * 64,
    chunk_manifest_sha256: str = "b" * 64,
) -> PaperDocumentBinding:
    return PaperDocumentBinding(
        paper_key=paper_key,
        document_id=document_id,
        source_version_id=source_version_id,
        title=title,
        source_sha256=source_sha256,
        processing_status="complete",
        processing_diagnostic_codes=(),
        admitted_chunk_count=8,
        chunk_manifest_sha256=chunk_manifest_sha256,
        admitted_text_page_start=1,
        admitted_text_page_end=8,
        outline=(
            PaperOutlineEntry(
                outline_key=f"{paper_key}:S1",
                title="Introduction",
                start_handle=f"{paper_key}:C1",
                page_start=1,
                page_end=2,
            ),
            PaperOutlineEntry(
                outline_key=f"{paper_key}:S2",
                title="Method",
                start_handle=f"{paper_key}:C3",
                page_start=3,
                page_end=6,
            ),
        ),
        outline_truncated=False,
    )


def _snapshot(
    *,
    session_id: str,
    task_id: str = "task-paper",
    graph_revision: int | None = None,
    task_state_version: int = 1,
    retrieval_data_version_id: str = "rdv_alpha",
    documents: tuple[PaperDocumentBinding, ...] | None = None,
) -> PaperResourceSnapshot:
    return PaperResourceSnapshot.create(
        session_id=session_id,
        task_id=task_id,
        bound_graph_revision=graph_revision,
        bound_task_state_version=task_state_version,
        retrieval_data_version_id=retrieval_data_version_id,
        retrieval_generation_fingerprint="retrieval-generation-spec-v1:" + "c" * 64,
        encoder_fingerprint="deterministic-lexical@1",
        documents=documents or (_document(),),
    )


def test_snapshot_contract_has_canonical_self_authenticating_identity() -> None:
    snapshot = _snapshot(session_id="session-alpha")

    assert snapshot.snapshot_id == f"prs_v1_{snapshot.manifest_sha256}"
    assert json.loads(snapshot.canonical_manifest_json)["session_id"] == "session-alpha"
    assert json.loads(snapshot.canonical_payload_json) == snapshot.model_dump(mode="json")
    assert "doc-alpha" in snapshot.canonical_manifest_json
    assert "Alpha Paper" in snapshot.canonical_manifest_json

    raw = snapshot.model_dump(mode="json")
    raw["manifest_sha256"] = "0" * 64
    with pytest.raises(ValidationError, match="manifest_sha256"):
        PaperResourceSnapshot.model_validate(raw)


def test_snapshot_contract_requires_one_or_two_ordered_exact_documents() -> None:
    with pytest.raises(ValidationError):
        PaperResourceSnapshot.create(
            session_id="session-alpha",
            task_id="task-paper",
            bound_graph_revision=None,
            bound_task_state_version=1,
            retrieval_data_version_id="rdv_alpha",
            retrieval_generation_fingerprint="generation-alpha",
            encoder_fingerprint="encoder-alpha",
            documents=(),
        )

    p2 = _document(
        "P2",
        document_id="doc-beta",
        source_version_id="docv-beta",
        title="Beta Paper",
        source_sha256="d" * 64,
        chunk_manifest_sha256="e" * 64,
    )
    snapshot = _snapshot(session_id="session-alpha", documents=(_document(), p2))
    assert tuple(item.paper_key for item in snapshot.documents) == ("P1", "P2")

    with pytest.raises(ValidationError, match="P1, P2"):
        _snapshot(session_id="session-alpha", documents=(p2, _document()))

    with pytest.raises(ValidationError, match="complete processing"):
        PaperDocumentBinding.model_validate(
            {
                **_document().model_dump(mode="json"),
                "processing_diagnostic_codes": ["page_needs_vision"],
            }
        )

    with pytest.raises(ValidationError, match="stable codes"):
        PaperDocumentBinding.model_validate(
            {
                **_document().model_dump(mode="json"),
                "processing_status": "partial",
                "processing_diagnostic_codes": ["parser failed: /private/paper.pdf"],
            }
        )


def test_paper_resource_contract_owner_is_cold() -> None:
    tree = ast.parse(
        Path(paper_resource_contracts.__file__).read_text(encoding="utf-8")
    )
    imports = {
        node.module or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }
    assert not {module for module in imports if module.startswith("personagraph.")}

    repository_root = Path(__file__).resolve().parents[2]
    environment = dict(os.environ)
    existing_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = os.pathsep.join(
        value
        for value in (str(repository_root / "src"), existing_pythonpath)
        if value
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import json
import sys
import personagraph.l2.task_graph.paper_resource_contracts
blocked = {
    'personagraph.runtime',
    'personagraph.session',
    'personagraph.session.store',
    'personagraph.session.persistence',
}
print(json.dumps(sorted(blocked & set(sys.modules))))
""",
        ],
        cwd=repository_root,
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(completed.stdout) == []


def test_retired_paper_resource_persistence_lane_stays_absent() -> None:
    session_root = Path(__file__).resolve().parents[2] / "src/personagraph/session"
    assert not (session_root / "paper_resource_store_facade.py").exists()
    assert not (session_root / "persistence/paper_resource_port.py").exists()
    assert not (session_root / "persistence/paper_resources.py").exists()
