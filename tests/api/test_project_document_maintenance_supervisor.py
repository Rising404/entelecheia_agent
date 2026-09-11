from __future__ import annotations

import threading
from pathlib import Path

from personagraph.api import server
from personagraph.workspace.ingestion.composition import (
    build_project_document_maintenance_supervisor,
)
from personagraph.workspace.storage.context import (
    current as current_project_documents,
)
from personagraph.configuration import paths
from personagraph.session import project_catalog
from personagraph.session.project_catalog import Project


def test_api_document_maintenance_is_empty_project_catalog_safe(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(
        project_catalog,
        "DB_PATH",
        tmp_path / "project_catalog.sqlite",
    )

    lifecycle = server._build_document_maintenance_lifecycle()
    try:
        assert lifecycle.start() is True
        assert lifecycle.is_running is True
    finally:
        assert lifecycle.stop(timeout_seconds=1.0) is True


def test_supervisor_builds_existing_project_in_its_own_documents_database(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(
        project_catalog,
        "DB_PATH",
        tmp_path / "project_catalog.sqlite",
    )
    monkeypatch.setattr(paths, "PROJECTS_DIR", tmp_path / "state" / "projects")
    project_root = tmp_path / "existing-project"
    project_root.mkdir()
    project = project_catalog.remember(str(project_root))
    documents_db_path = Path(project.documents_db_path)
    supervisor = build_project_document_maintenance_supervisor()

    try:
        assert supervisor.start() is True
        assert supervisor.managed_project_ids == (project.project_id,)
        assert documents_db_path.is_file()
    finally:
        assert supervisor.stop(timeout_seconds=1.0) is True


def test_supervisor_discovers_later_projects_and_stops_every_child(tmp_path):
    projects = [
        _project(tmp_path, "project-one"),
    ]
    built_second = threading.Event()
    children: dict[str, _FakeLifecycle] = {}
    bound_databases = {}

    def build_lifecycle(project, database):
        assert current_project_documents() is database
        assert database.project_id == project.project_id
        assert database.project_root == Path(project.canonical_root)
        bound_databases[project.project_id] = database
        child = _FakeLifecycle()
        children[project.project_id] = child
        if project.project_id == "project-two":
            built_second.set()
        return child

    supervisor = build_project_document_maintenance_supervisor(
        project_loader=lambda: tuple(projects),
        lifecycle_builder=build_lifecycle,
        poll_interval_seconds=0.01,
    )

    assert supervisor.start() is True
    assert children["project-one"].starts == 1
    projects.append(_project(tmp_path, "project-two"))
    supervisor.wake()
    assert built_second.wait(1.0)
    assert supervisor.managed_project_ids == ("project-one", "project-two")
    assert current_project_documents() is None

    assert supervisor.stop(timeout_seconds=1.0) is True
    assert supervisor.is_running is False
    assert supervisor.managed_project_ids == ()
    assert set(bound_databases) == {"project-one", "project-two"}
    assert all(child.stops == 1 for child in children.values())


class _FakeLifecycle:
    def __init__(self) -> None:
        self.starts = 0
        self.stops = 0

    def start(self) -> bool:
        self.starts += 1
        return True

    def stop(self, *, timeout_seconds: float) -> bool:
        assert timeout_seconds >= 0
        self.stops += 1
        return True


def _project(tmp_path, project_id: str) -> Project:
    root = tmp_path / project_id
    root.mkdir()
    return Project(
        path=str(root),
        name=project_id,
        project_id=project_id,
        canonical_root=str(root.resolve()),
        documents_db_path=str(tmp_path / "state" / project_id / "documents.sqlite"),
    )
