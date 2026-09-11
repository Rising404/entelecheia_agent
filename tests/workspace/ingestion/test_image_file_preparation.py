from pathlib import Path

from PIL import Image

from personagraph.workspace.files.access import FileAccess
from personagraph.workspace.ingestion.contracts import FilePreparationStatus
from personagraph.workspace.ingestion.images import check_image_file, prepare_image_file
from personagraph.workspace.storage.database import DocumentDatabase


def _setup(tmp_path: Path):
    root = tmp_path / "project"
    root.mkdir()
    Image.new("RGB", (24, 16)).save(root / "chart.png")
    database = DocumentDatabase("project", root, tmp_path / "documents.sqlite")
    access = FileAccess(database=database, validate_path=lambda _: True)
    return database, access


def test_image_check_is_read_only_then_explicit_prepare_registers_once(tmp_path):
    database, access = _setup(tmp_path)
    source = access.resolve_path("chart.png")
    checked = check_image_file(source=source, revalidate_source=access.revalidate)
    assert checked.reason_code == "image_not_prepared"
    assert not database.db_path.exists()

    prepared = prepare_image_file(source=source, database=database, revalidate_source=access.revalidate)
    assert prepared.status is FilePreparationStatus.PENDING
    assert prepared.reason_code == "image_visual_ready"
    assert prepared.file_id and prepared.file_version_id
    assert prepared.document_id is None
    assert prepared.operation_id is None
    replay = prepare_image_file(
        source=access.resolve_path("chart.png"), database=database,
        revalidate_source=access.revalidate,
    )
    assert replay == prepared
    with database.connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM files").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM file_versions").fetchone()[0] == 1
        for table in ("documents", "document_ingest_jobs", "pictures", "picture_observations"):
            assert conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0


def test_revoked_image_source_does_not_register(tmp_path):
    database, access = _setup(tmp_path)
    source = access.resolve_path("chart.png")
    result = prepare_image_file(source=source, database=database, revalidate_source=lambda _: False)
    assert result.status is FilePreparationStatus.BLOCKED
    assert result.reason_code == "file_authority_denied"
    assert not database.db_path.exists()


def test_registered_image_with_changed_bytes_is_stale_without_new_version(tmp_path):
    database, access = _setup(tmp_path)
    prepared = prepare_image_file(
        source=access.resolve_path("chart.png"), database=database,
        revalidate_source=access.revalidate,
    )
    Image.new("RGB", (24, 16), color="red").save(database.project_root / "chart.png")
    changed_source = access.resolve_path("chart.png")
    assert changed_source.file_id == prepared.file_id
    assert changed_source.file_version_id is None

    checked = check_image_file(source=changed_source, revalidate_source=access.revalidate)

    assert checked.status is FilePreparationStatus.STALE
    assert checked.reason_code == "file_content_changed"
    assert checked.file_id == prepared.file_id
    assert checked.file_version_id is None
    with database.connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM file_versions").fetchone()[0] == 1


def test_image_type_claim_is_checked_against_bytes_before_registration(tmp_path):
    database, access = _setup(tmp_path)
    (database.project_root / "chart.png").write_text("This is not an image")
    source = access.resolve_path("chart.png")
    result = prepare_image_file(source=source, database=database, revalidate_source=access.revalidate)
    assert result.status is FilePreparationStatus.BLOCKED
    assert result.reason_code == "image_source_type_mismatch"
    assert not database.db_path.exists()


def test_first_image_tool_request_does_not_require_existing_document_database(tmp_path):
    from personagraph.workspace.ingestion.application import FileIngestionService
    from personagraph.workspace.storage.context import bind

    database, access = _setup(tmp_path)
    assert not database.db_path.exists()

    def no_document_worker():
        raise AssertionError("image registration must not start document ingestion")

    service = FileIngestionService(
        session_id="image-session", access=access, generation_identity=object(),
        chunking_profile=object(), validate_source=lambda *_: True,
        resolve_owner=no_document_worker,
    )
    with bind(database):
        result = service.prepare(access.resolve_path("chart.png"), request_id="image-call-input-0")

    assert result.reason_code == "image_visual_ready"
    assert result.operation_id is None
    with database.connect() as conn:
        assert conn.execute("SELECT count(*) FROM document_ingest_jobs").fetchone()[0] == 0
