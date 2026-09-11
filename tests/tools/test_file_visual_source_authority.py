from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from personagraph.input_processing.files import fingerprint_file
from personagraph.workspace.files import FileSource
from personagraph.workspace.files.access import AuthorizedFileSource
from personagraph.tools.visual import file_visual_source_authority as authority
from personagraph.tools.visual.file_visual_source_authority import (
    FileVisualAuthorityError, freeze_file_visual_source, visual_page_number,
)


def _source(tmp_path: Path) -> AuthorizedFileSource:
    path = tmp_path / "shared.png"
    Image.new("RGB", (24, 16)).save(path)
    return AuthorizedFileSource(
        project_id="project-1", file_id="file-1", file_version_id="version-1",
        canonical_path=str(path), relative_path=path.name, file_name=path.name,
        origin=FileSource.WORKSPACE_EXISTING, media_type="image/png",
        fingerprint=fingerprint_file(path),
    )


def test_image_has_shared_stable_identity_without_document_or_session_namespace(tmp_path):
    source = _source(tmp_path)
    first = freeze_file_visual_source(session_id="session-a", source=source, revalidate_source=lambda _: True)
    second = freeze_file_visual_source(session_id="session-b", source=source, revalidate_source=lambda _: True)
    assert first.catalog == second.catalog
    assert first.snapshot_id == second.snapshot_id
    assert first.document_id is None
    public, boundary = first.resolve("whole_file")
    assert public.source_pages == (1,)
    assert boundary.session_id == "session-a"
    assert boundary.units[0].unit_id == "version-1:whole_file"
    assert str(source.canonical_path) not in repr(public.to_model_dict())
    with pytest.raises(FileVisualAuthorityError, match="visual_unit_outside_file"):
        first.resolve("not_a_unit")


def test_revocation_rejects_before_source_io_and_after_freeze(tmp_path, monkeypatch):
    source = _source(tmp_path)
    allowed = True
    frozen = freeze_file_visual_source(
        session_id="session-a", source=source, revalidate_source=lambda _: allowed,
    )
    allowed = False
    monkeypatch.setattr(authority.Image, "open", lambda *_: pytest.fail("revoked source was opened"))
    with pytest.raises(FileVisualAuthorityError, match="file_authority_stale"):
        freeze_file_visual_source(session_id="session-b", source=source, revalidate_source=lambda _: False)
    with pytest.raises(FileVisualAuthorityError, match="file_authority_stale"):
        frozen.resolve("whole_file")


def test_image_revalidation_refuses_mutated_bytes(tmp_path):
    source = _source(tmp_path)
    def validate(item):
        return fingerprint_file(Path(item.canonical_path)).sha256 == item.fingerprint.sha256
    frozen = freeze_file_visual_source(session_id="session", source=source, revalidate_source=validate)
    Image.new("RGB", (24, 16), color="red").save(source.canonical_path)
    with pytest.raises(FileVisualAuthorityError, match="file_authority_stale"):
        frozen.resolve("whole_file")


def test_unprepared_pdf_does_not_parse_or_mount(tmp_path, monkeypatch):
    source = replace(_source(tmp_path), media_type="application/pdf")
    seen = []
    def lookup(**kwargs):
        seen.append(kwargs)
        return None
    monkeypatch.setattr(authority.docstore, "get_current_file_document", lookup)
    monkeypatch.setattr(authority, "freeze_mounted_visual_bindings", lambda **_: pytest.fail("missing manifest materialized"))
    assert freeze_file_visual_source(
        session_id="session", source=source, revalidate_source=lambda _: True, source_pages=(1,),
    ) is None
    assert seen == [{"file_id": "file-1", "file_version_id": "version-1", "session_id": "session"}]


def test_pdf_manifest_changes_invalidate_resolved_authority(tmp_path, monkeypatch):
    source = replace(_source(tmp_path), media_type="application/pdf")
    metadata = SimpleNamespace(document_id="document", document_version_id="document-version", source_sha256=source.fingerprint.sha256)
    page_authority = SimpleNamespace(
        source_sha256=source.fingerprint.sha256,
        page_manifest=SimpleNamespace(manifest_sha256="a" * 64),
    )
    monkeypatch.setattr(authority.docstore, "get_current_file_document", lambda **_: metadata)
    monkeypatch.setattr(authority.docstore, "get_current_document_page_authority", lambda *a, **kw: page_authority)
    monkeypatch.setattr(authority, "freeze_mounted_visual_bindings", lambda **_: ())
    frozen = freeze_file_visual_source(session_id="session", source=source, revalidate_source=lambda _: True)
    page_authority.page_manifest.manifest_sha256 = "b" * 64
    with pytest.raises(FileVisualAuthorityError, match="visual_manifest_stale"):
        frozen.validate_current()


@pytest.mark.parametrize("identifier,expected", [("pdf_page:12", 12), ("pdf_page:0", None), ("whole_file", None), ("pdf_page:1:2", None)])
def test_explicit_page_locator_is_bounded_and_not_a_new_file_identity(identifier, expected):
    assert visual_page_number(identifier) == expected
