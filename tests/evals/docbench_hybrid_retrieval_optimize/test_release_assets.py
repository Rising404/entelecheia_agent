"""Synthetic, offline release installation and transport-boundary checks."""

from hashlib import sha256
from io import BytesIO
import json
import subprocess
import sys
import tarfile
from urllib.request import Request

import pytest

from evals.docbench_hybrid_retrieval_optimize import release_assets as release


def _record(content):
    return {"bytes": len(content), "sha256": sha256(content).hexdigest()}


def _fixture(tmp_path, *, members=None):
    files = {"README.md": b"synthetic example\n", "assets/picture.bin": bytes(range(251)), "empty": b""}
    if members is None:
        members = [(f"sample/{name}", content, tarfile.REGTYPE) for name, content in files.items()]
    archive = tmp_path / "sample.tar.gz"
    with tarfile.open(archive, "w:gz") as stream:
        for name, content, kind in members:
            info = tarfile.TarInfo(name)
            info.type = kind
            info.size = len(content) if kind == tarfile.REGTYPE else 0
            if kind in (tarfile.SYMTYPE, tarfile.LNKTYPE):
                info.linkname = "../../escaped"
            stream.addfile(info, BytesIO(content) if kind == tarfile.REGTYPE else None)
    asset = {
        "filename": "sample.tar.gz", "root": "sample",
        "url": "https://github.com/owner/repo/releases/download/test/sample.tar.gz",
        **_record(archive.read_bytes()), "files": {name: _record(content) for name, content in files.items()},
    }
    catalogue = tmp_path / "catalogue.json"
    catalogue.write_text(json.dumps({
        "schema_version": release.SCHEMA, "repository": "owner/repo", "tag": "test", "assets": {"sample": asset},
    }))
    return catalogue, archive, files


def _fetch(tmp_path, catalogue, archive):
    return release.fetch_asset("sample", tmp_path / "output", catalogue=catalogue, archive=archive)


def _edit(catalogue, callback):
    data = json.loads(catalogue.read_bytes())
    callback(data)
    catalogue.write_text(json.dumps(data))


def _clean_failure(tmp_path, catalogue, archive, *, match=None):
    with pytest.raises(release.ReleaseAssetError, match=match):
        _fetch(tmp_path, catalogue, archive)
    assert not (tmp_path / "output").exists()
    assert not list(tmp_path.glob(".output-*"))
    assert not (tmp_path.parent / "escaped").exists()


def test_offline_install_preserves_all_bytes_and_cleans_stage(tmp_path, monkeypatch):
    catalogue, archive, files = _fixture(tmp_path)
    monkeypatch.setattr(release, "build_opener", lambda *_: pytest.fail("offline installation attempted network"))
    _fetch(tmp_path, catalogue, archive)
    output = tmp_path / "output"
    assert {path.relative_to(output).as_posix(): path.read_bytes() for path in output.rglob("*") if path.is_file()} == files
    assert not list(tmp_path.glob(".output-*"))


@pytest.mark.parametrize("defect", ["tamper", "short", "long"])
def test_compressed_archive_size_and_hash_are_verified_before_extraction(tmp_path, defect):
    catalogue, archive, _ = _fixture(tmp_path)
    content = archive.read_bytes()
    archive.write_bytes({"tamper": b"X" + content[1:], "short": content[:-1], "long": content + b"X"}[defect])
    _clean_failure(tmp_path, catalogue, archive)


@pytest.mark.parametrize("defect", ["tamper", "size", "missing", "extra", "duplicate", "symlink", "hardlink", "fifo", "extra_dir"])
def test_authenticated_archive_still_requires_exact_regular_members(tmp_path, defect):
    _, _, files = _fixture(tmp_path)
    members = [(f"sample/{name}", content, tarfile.REGTYPE) for name, content in files.items()]
    if defect == "tamper":
        members[0] = (members[0][0], b"X" * len(members[0][1]), tarfile.REGTYPE)
    elif defect == "size":
        members[0] = (members[0][0], b"short", tarfile.REGTYPE)
    elif defect == "missing":
        members.pop()
    elif defect == "duplicate":
        members.append(members[0])
    elif defect in {"symlink", "hardlink", "fifo"}:
        kinds = {"symlink": tarfile.SYMTYPE, "hardlink": tarfile.LNKTYPE, "fifo": tarfile.FIFOTYPE}
        # Keep the registered path and size valid so only the member type is wrong.
        members[-1] = ("sample/empty", b"", kinds[defect])
    else:
        kinds = {"extra": tarfile.REGTYPE, "extra_dir": tarfile.DIRTYPE}
        members.append(("sample/unexpected", b"", kinds[defect]))
    catalogue, archive, _ = _fixture(tmp_path, members=members)
    match = "non-regular member" if defect in {"symlink", "hardlink", "fifo"} else None
    _clean_failure(tmp_path, catalogue, archive, match=match)


@pytest.mark.parametrize("path", ["../escaped", "/absolute", "sample/../escaped", "sample//escaped", "sample/./escaped", "sample\\escaped", "other/README.md"])
def test_unsafe_archive_member_paths_are_rejected(tmp_path, path):
    catalogue, archive, _ = _fixture(tmp_path, members=[(path, b"payload", tarfile.REGTYPE)])
    _clean_failure(tmp_path, catalogue, archive)


def test_optional_parent_directories_are_allowed(tmp_path):
    _, _, files = _fixture(tmp_path)
    members = [("sample/", b"", tarfile.DIRTYPE), ("sample/assets", b"", tarfile.DIRTYPE)]
    members += [(f"sample/{name}", content, tarfile.REGTYPE) for name, content in files.items()]
    catalogue, archive, _ = _fixture(tmp_path, members=members)
    _fetch(tmp_path, catalogue, archive)


@pytest.mark.parametrize("defect", ["url", "root", "file_path", "parent_collision", "bool_bytes", "hash", "duplicate_json", "nonfinite", "schema", "oversize", "repository", "asset_id"])
def test_catalogue_is_validated_before_io(tmp_path, defect, monkeypatch):
    catalogue, archive, _ = _fixture(tmp_path)
    if defect == "duplicate_json":
        catalogue.write_text('{"assets": {}, "assets": {}}')
    elif defect == "nonfinite":
        catalogue.write_text('{"bad": NaN}')
    else:
        def mutate(data):
            asset = data["assets"]["sample"]
            if defect == "url":
                asset["url"] = "https://github.com/other/repo/releases/download/test/sample.tar.gz"
            elif defect == "root":
                asset["root"] = "../sample"
            elif defect == "file_path":
                asset["files"]["../escaped"] = _record(b"bad")
            elif defect == "parent_collision":
                asset["files"]["assets"] = _record(b"bad")
            elif defect == "bool_bytes":
                asset["bytes"] = True
            elif defect == "hash":
                asset["sha256"] = "invalid"
            elif defect == "oversize":
                asset["bytes"] = release._MAX_ARCHIVE_BYTES + 1
            elif defect == "repository":
                data["repository"] = "../repo"
            elif defect == "asset_id":
                data["assets"]["%2fescape"] = data["assets"].pop("sample")
            else:
                data["schema_version"] = "unknown"
        _edit(catalogue, mutate)
    monkeypatch.setattr(release, "build_opener", lambda *_: pytest.fail("invalid catalogue attempted network"))
    _clean_failure(tmp_path, catalogue, archive)


@pytest.mark.parametrize("target", ["output", "parent", "archive", "catalogue"])
def test_filesystem_symlinks_are_refused(tmp_path, target):
    catalogue, archive, _ = _fixture(tmp_path)
    if target == "output":
        (tmp_path / "output").symlink_to(tmp_path / "absent", target_is_directory=True)
    elif target == "parent":
        directory = tmp_path / "real-parent"
        directory.mkdir()
        (tmp_path / "linked-parent").symlink_to(directory, target_is_directory=True)
        with pytest.raises(release.ReleaseAssetError, match="symlink"):
            release.fetch_asset("sample", tmp_path / "linked-parent" / "output", catalogue=catalogue, archive=archive)
        return
    else:
        source = archive if target == "archive" else catalogue
        original = source.with_suffix(".original")
        source.rename(original)
        source.symlink_to(original)
    with pytest.raises(release.ReleaseAssetError, match="symlink"):
        _fetch(tmp_path, catalogue, archive)
    assert not list(tmp_path.glob(".output-*"))


def test_existing_output_is_not_modified_or_downloaded(tmp_path, monkeypatch):
    catalogue, _, _ = _fixture(tmp_path)
    output = tmp_path / "output"
    output.mkdir()
    (output / "keep").write_bytes(b"original")
    monkeypatch.setattr(release, "build_opener", lambda *_: pytest.fail("existing output attempted network"))
    with pytest.raises(release.ReleaseAssetError, match="refusing overwrite"):
        release.fetch_asset("sample", output, catalogue=catalogue)
    assert (output / "keep").read_bytes() == b"original"


def test_atomic_publish_refuses_concurrently_created_empty_directory(tmp_path, monkeypatch):
    catalogue, archive, _ = _fixture(tmp_path)
    rename = release._rename_exclusive
    def race(source, target):
        target.mkdir()
        rename(source, target)
    monkeypatch.setattr(release, "_rename_exclusive", race)
    with pytest.raises(release.ReleaseAssetError):
        _fetch(tmp_path, catalogue, archive)
    assert list((tmp_path / "output").iterdir()) == []
    assert not list(tmp_path.glob(".output-*"))


@pytest.mark.parametrize("url", [
    "http://release-assets.githubusercontent.com/file", "https://example.com/file",
    "https://release-assets.githubusercontent.com.evil.example/file",
    "https://user@release-assets.githubusercontent.com/file", "https://release-assets.githubusercontent.com:8443/file",
    "https://github.com/other/repo/releases/download/test/sample.tar.gz",
])
def test_redirect_handler_rejects_unapproved_destination_before_following(url):
    original = "https://github.com/owner/repo/releases/download/test/sample.tar.gz"
    with pytest.raises(release.ReleaseAssetError):
        release._AssetRedirectHandler(original).redirect_request(Request(original), None, 302, "Found", {}, url)


@pytest.mark.parametrize("defect", [None, "final_host", "content_length", "extra_body", "status"])
def test_mocked_network_checks_final_url_length_status_and_stream(tmp_path, monkeypatch, defect):
    catalogue, archive, files = _fixture(tmp_path)
    content = archive.read_bytes()
    class Response(BytesIO):
        status = 206 if defect == "status" else 200
        headers = {} if defect == "extra_body" else {"Content-Length": str(len(content) + (1 if defect == "content_length" else 0))}
        def geturl(self):
            host = "example.com" if defect == "final_host" else "release-assets.githubusercontent.com"
            return f"https://{host}/signed?token=synthetic"
    class Opener:
        def open(self, request, timeout):
            assert request.full_url == "https://github.com/owner/repo/releases/download/test/sample.tar.gz"
            return Response(content + (b"X" if defect == "extra_body" else b""))
    monkeypatch.setattr(release, "build_opener", lambda *_: Opener())
    if defect is None:
        release.fetch_asset("sample", tmp_path / "output", catalogue=catalogue)
        assert (tmp_path / "output" / "README.md").read_bytes() == files["README.md"]
    else:
        _clean_failure(tmp_path, catalogue, None)


def test_list_is_offline_and_cli_supports_explicit_local_archive(tmp_path, monkeypatch, capsys):
    catalogue, archive, _ = _fixture(tmp_path)
    monkeypatch.setattr(release, "build_opener", lambda *_: pytest.fail("offline command attempted network"))
    assert release.main(["--catalogue", str(catalogue), "list"]) == 0
    assert "sample" in capsys.readouterr().out
    assert release.main(["--catalogue", str(catalogue), "fetch", "--asset", "sample", "--output", str(tmp_path / "output"), "--archive", str(archive)]) == 0
    assert release.main(["--catalogue", str(catalogue), "fetch", "--asset", "missing", "--output", str(tmp_path / "other")]) == 1


def test_cold_import_does_not_load_application_or_model_dependencies():
    script = "import sys; import evals.docbench_hybrid_retrieval_optimize.release_assets; assert not any(n.split('.')[0] in {'personagraph', 'torch', 'numpy'} for n in sys.modules)"
    subprocess.run([sys.executable, "-S", "-c", script], check=True)
