"""Exact-byte archive integrity using only synthetic files and tiny test parts."""

from hashlib import sha256
import json
from pathlib import Path

import pytest

from evals.docbench_hybrid_retrieval_optimize import index_archive as archive


@pytest.fixture
def index(tmp_path, monkeypatch):
    assert archive.PART_BYTES == 64 * 1024 * 1024
    monkeypatch.setattr(archive, "PART_BYTES", 1024)
    root = tmp_path / "index"
    root.mkdir()
    content = bytes(range(251)) * 10
    (root / "retrieval.sqlite").write_bytes(content)
    (root / "manifest.json").write_text(json.dumps({
        "schema_version": "docbench-hybrid-index-v1",
        "database_sha256": sha256(content).hexdigest(),
        "recipe": {"encoder": "synthetic", "methods": ["dense", "learned_sparse", "bm25"]},
    }, indent=4) + "\n\n")
    return root


def _files(root):
    return {path.name: path.read_bytes() for path in root.iterdir()}


def _packed(index, tmp_path):
    destination = tmp_path / "archive"
    archive.pack_index(index, destination)
    return destination


def _edit_manifest(root, update):
    path = root / "archive.json"
    data = json.loads(path.read_bytes())
    update(data)
    path.write_text(json.dumps(data))


def test_round_trip_keeps_original_database_and_manifest_bytes(index, tmp_path):
    before = _files(index)
    packed = _packed(index, tmp_path)
    records = json.loads((packed / "archive.json").read_bytes())
    assert [part["bytes"] for part in records["parts"]] == [1024, 1024, 462]
    assert records["index_manifest"] == {
        "bytes": len(before["manifest.json"]),
        "sha256": sha256(before["manifest.json"]).hexdigest(),
    }
    assert not (packed / "retrieval.sqlite").exists()
    assert b"".join((packed / part["name"]).read_bytes() for part in records["parts"]) == before["retrieval.sqlite"]
    restored = tmp_path / "restored"
    assert archive.restore_index(packed, restored) == records
    assert _files(restored) == before
    assert _files(index) == before
    assert not list(tmp_path.glob(".archive-*"))
    assert not list(tmp_path.glob(".restored-*"))


@pytest.mark.parametrize("defect", ["changed", "short", "long", "missing", "extra", "extra_directory"])
def test_restore_rejects_bad_or_unexpected_parts_atomically(index, tmp_path, defect):
    packed = _packed(index, tmp_path)
    part = packed / "retrieval.sqlite.part-00000"
    if defect == "changed":
        part.write_bytes(b"X" + part.read_bytes()[1:])
    elif defect == "short":
        part.write_bytes(part.read_bytes()[:-1])
    elif defect == "long":
        part.write_bytes(part.read_bytes() + b"X")
    elif defect == "missing":
        part.unlink()
    elif defect == "extra":
        (packed / "retrieval.sqlite.part-00099").write_bytes(b"extra")
    else:
        (packed / "unexpected").mkdir()
    with pytest.raises(archive.IndexArchiveError):
        archive.restore_index(packed, tmp_path / "restored")
    assert not (tmp_path / "restored").exists()
    assert not list(tmp_path.glob(".restored-*"))


@pytest.mark.parametrize("name", ["../escaped", "/absolute", "retrieval.sqlite.part-00001", "a/../../escaped", "retrieval.sqlite.part-00000/child"])
def test_restore_rejects_unsafe_or_out_of_order_part_paths(index, tmp_path, name):
    packed = _packed(index, tmp_path)
    _edit_manifest(packed, lambda data: data["parts"][0].update(name=name))
    with pytest.raises(archive.IndexArchiveError, match="part path"):
        archive.restore_index(packed, tmp_path / "restored")
    assert not (tmp_path / "restored").exists()


@pytest.mark.parametrize("defect", ["original_manifest", "manifest_hash", "database_binding", "overall_hash", "overall_size", "part_size", "part_hash", "duplicate_key", "nonfinite"])
def test_restore_requires_manifest_binding_and_all_hash_size_layers(index, tmp_path, defect):
    packed = _packed(index, tmp_path)
    if defect == "original_manifest":
        path = packed / "manifest.json"
        path.write_bytes(path.read_bytes() + b" ")
    elif defect == "duplicate_key":
        path = packed / "archive.json"
        path.write_bytes(path.read_bytes().replace(b'{', b'{"format":"other",', 1))
    elif defect == "nonfinite":
        path = packed / "archive.json"
        path.write_text('{"bad": NaN}')
    elif defect == "overall_hash":
        # All part hashes remain authentic and both metadata files agree on an
        # incorrect total hash: only whole-stream verification can reject this.
        path = packed / "manifest.json"
        manifest = json.loads(path.read_bytes())
        manifest["database_sha256"] = "a" * 64
        content = json.dumps(manifest).encode()
        path.write_bytes(content)
        def update(data):
            data["database"]["sha256"] = "a" * 64
            data["index_manifest"] = {"bytes": len(content), "sha256": sha256(content).hexdigest()}
        _edit_manifest(packed, update)
    else:
        def update(data):
            if defect == "manifest_hash":
                data["index_manifest"]["sha256"] = "a" * 64
            elif defect == "database_binding":
                data["database"]["sha256"] = "a" * 64
            elif defect == "overall_size":
                data["database"]["bytes"] += 1
            elif defect == "part_size":
                data["parts"][0]["bytes"] -= 1
            else:
                data["parts"][0]["sha256"] = "a" * 64
        _edit_manifest(packed, update)
    with pytest.raises(archive.IndexArchiveError):
        archive.restore_index(packed, tmp_path / "restored")
    assert not (tmp_path / "restored").exists()
    assert not list(tmp_path.glob(".restored-*"))


@pytest.mark.parametrize("sidecar", ["-wal", "-journal"])
def test_pack_rejects_pending_transactions_without_changing_source(index, tmp_path, sidecar):
    (index / ("retrieval.sqlite" + sidecar)).write_bytes(b"pending transaction")
    before = _files(index)
    with pytest.raises(archive.IndexArchiveError, match="pending"):
        archive.pack_index(index, tmp_path / "archive")
    assert _files(index) == before
    assert not (tmp_path / "archive").exists()


def test_empty_wal_and_shared_memory_are_not_archived(index, tmp_path):
    (index / "retrieval.sqlite-wal").write_bytes(b"")
    (index / "retrieval.sqlite-journal").write_bytes(b"")
    (index / "retrieval.sqlite-shm").write_bytes(b"ephemeral shared memory")
    before = _files(index)
    packed = _packed(index, tmp_path)
    restored = tmp_path / "restored"
    archive.restore_index(packed, restored)
    assert set(_files(restored)) == {"manifest.json", "retrieval.sqlite"}
    assert _files(index) == before


@pytest.mark.parametrize("target", ["root", "database", "manifest", "sidecar", "output", "output_parent", "archive_manifest", "part"])
def test_symlinks_are_rejected_at_input_and_output_boundaries(index, tmp_path, target):
    output = tmp_path / "output"
    source = index
    operation = archive.pack_index
    if target in {"archive_manifest", "part"}:
        source = _packed(index, tmp_path)
        operation = archive.restore_index
        path = source / ("archive.json" if target == "archive_manifest" else "retrieval.sqlite.part-00000")
        real = tmp_path / "real-file"
        path.rename(real)
        path.symlink_to(real)
    elif target == "root":
        source = tmp_path / "link"
        source.symlink_to(index, target_is_directory=True)
    elif target in {"database", "manifest"}:
        path = index / ("retrieval.sqlite" if target == "database" else "manifest.json")
        real = tmp_path / "real-file"
        path.rename(real)
        path.symlink_to(real)
    elif target == "sidecar":
        (index / "retrieval.sqlite-wal").symlink_to(tmp_path / "missing")
    elif target == "output":
        output.symlink_to(tmp_path / "missing")
    else:
        (tmp_path / "link").symlink_to(tmp_path, target_is_directory=True)
        output = tmp_path / "link/output"
    with pytest.raises(archive.IndexArchiveError, match="symlink"):
        operation(source, output)
    assert not list(tmp_path.glob(".output-*"))


def test_existing_or_overlapping_outputs_are_never_replaced(index, tmp_path):
    packed = _packed(index, tmp_path)
    for operation, source in ((archive.pack_index, index), (archive.restore_index, packed)):
        existing = tmp_path / "existing"
        existing.mkdir(exist_ok=True)
        (existing / "keep").write_bytes(b"unchanged")
        with pytest.raises(archive.IndexArchiveError, match="overwrite"):
            operation(source, existing)
        assert (existing / "keep").read_bytes() == b"unchanged"
        with pytest.raises(archive.IndexArchiveError, match="disjoint"):
            operation(source, source / "nested")
        assert not (source / "nested").exists()


def test_source_manifest_change_leaves_no_published_archive(index, tmp_path, monkeypatch):
    original = archive._copy_part
    def change_manifest(*args):
        result = original(*args)
        path = index / "manifest.json"
        path.write_bytes(path.read_bytes() + b" ")
        return result
    monkeypatch.setattr(archive, "_copy_part", change_manifest)
    with pytest.raises(archive.IndexArchiveError, match="manifest changed"):
        archive.pack_index(index, tmp_path / "archive")
    assert not (tmp_path / "archive").exists()
    assert not list(tmp_path.glob(".archive-*"))


def test_pack_refuses_stale_database_hash(index, tmp_path):
    database = index / "retrieval.sqlite"
    database.write_bytes(database.read_bytes() + b"changed")
    with pytest.raises(archive.IndexArchiveError, match="database hash"):
        archive.pack_index(index, tmp_path / "archive")
    assert not (tmp_path / "archive").exists()
    assert not list(tmp_path.glob(".archive-*"))


def test_cli_round_trip_and_failure_exit(index, tmp_path, capsys):
    packed, restored = tmp_path / "archive", tmp_path / "restored"
    assert archive.main(["pack", "--index", str(index), "--output", str(packed)]) == 0
    assert json.loads(capsys.readouterr().out)["part_count"] == 3
    assert archive.main(["restore", "--archive", str(packed), "--output", str(restored)]) == 0
    assert _files(restored) == _files(index)
    with pytest.raises(SystemExit) as error:
        archive.main(["restore", "--archive", str(packed), "--output", str(restored)])
    assert error.value.code == 2


def test_publication_refuses_output_created_during_pack(index, tmp_path, monkeypatch):
    output = tmp_path / "archive"
    original = archive._publish
    def race(stage: Path, target: Path):
        output.mkdir()
        (output / "keep").write_bytes(b"other owner")
        original(stage, target)
    monkeypatch.setattr(archive, "_publish", race)
    with pytest.raises(archive.IndexArchiveError, match="appeared"):
        archive.pack_index(index, output)
    assert _files(output) == {"keep": b"other owner"}
    assert not list(tmp_path.glob(".archive-*"))


def test_atomic_publish_cannot_replace_empty_directory_created_after_precheck(index, tmp_path, monkeypatch):
    output = tmp_path / "archive"
    original = archive._rename_exclusive
    inode = []
    def race(stage, target):
        target.mkdir()
        inode.append(target.stat().st_ino)
        original(stage, target)
    monkeypatch.setattr(archive, "_rename_exclusive", race)
    with pytest.raises(archive.IndexArchiveError):
        archive.pack_index(index, output)
    assert output.stat().st_ino == inode[0]
    assert not list(output.iterdir())
    assert not list(tmp_path.glob(".archive-*"))


def test_io_failure_during_restore_does_not_publish_partial_database(index, tmp_path, monkeypatch):
    packed = _packed(index, tmp_path)
    before = _files(packed)
    def fail(source, target, size, whole_hash):
        target.write(source.read(10))
        raise OSError("synthetic interrupted write")
    monkeypatch.setattr(archive, "_copy_part", fail)
    with pytest.raises(archive.IndexArchiveError, match="interrupted write"):
        archive.restore_index(packed, tmp_path / "restored")
    assert _files(packed) == before
    assert not (tmp_path / "restored").exists()
    assert not list(tmp_path.glob(".restored-*"))
