from __future__ import annotations

import json

import pytest

from personagraph.retrieval.operations.generation_admin import main
from personagraph.retrieval.sqlite_store import (
    RetrievalDataVersionRole,
    RetrievalDataVersionState,
    SqliteRetrievalCatalog,
)


def test_generation_admin_diagnoses_empty_previous_as_rebuild_required(
    tmp_path,
    capsys,
) -> None:
    db_path = tmp_path / "retrieval.sqlite"
    catalog = SqliteRetrievalCatalog(db_path)
    catalog.initialize()
    catalog.create_data_version(
        version_id="v1",
        fingerprint="fingerprint-v1",
        role=RetrievalDataVersionRole.ACTIVE,
        state=RetrievalDataVersionState.READY,
    )
    catalog.create_data_version(
        version_id="v2",
        fingerprint="fingerprint-v2",
        role=RetrievalDataVersionRole.STAGING,
        state=RetrievalDataVersionState.READY,
    )
    catalog.activate_data_version("v2")

    assert main([
        "diagnose-previous",
        "--corpus", "file",
        "--db-path", str(db_path),
        "--expected-fingerprint", "fingerprint-v1",
    ]) == 2
    diagnosed = json.loads(capsys.readouterr().out)
    assert diagnosed["status"] == "rebuild_required"
    assert diagnosed["reason_code"] == "previous_generation_empty"
    assert catalog.active_data_version().id == "v2"


def test_generation_admin_restore_requires_identity_and_project_scope(tmp_path) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main([
            "restore-previous",
            "--corpus", "file",
            "--db-path", str(tmp_path / "retrieval.sqlite"),
        ])

    assert exc_info.value.code == 2

    with pytest.raises(SystemExit) as project_scope:
        main([
            "restore-previous",
            "--corpus", "file",
            "--db-path", str(tmp_path / "retrieval.sqlite"),
            "--target-generation-id", "v1",
            "--expected-fingerprint", "fingerprint-v1",
        ])

    assert project_scope.value.code == 2
