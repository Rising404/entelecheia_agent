"""Fetch explicitly selected, hash-locked Release data using only the stdlib."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from hashlib import sha256
from http.client import HTTPException
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys
import tarfile
import tempfile
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from .index_archive import _rename_exclusive
from .paths import artifact_path


CATALOGUE_PATH = Path(__file__).with_name("release_assets.json")
SCHEMA = "docbench-retrieval-release-assets-v1"
_BUFFER_BYTES = 1024 * 1024
_MAX_CATALOGUE_BYTES = 16 * 1024 * 1024
_MAX_ARCHIVE_BYTES = 2 * 1024**3
_MAX_EXPANDED_BYTES = 8 * 1024**3
_ASSET_HOSTS = {"release-assets.githubusercontent.com", "objects.githubusercontent.com"}


class ReleaseAssetError(ValueError):
    """An asset failed its publication, transport or extraction contract."""


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ReleaseAssetError("catalogue contains duplicate JSON keys")
        result[key] = value
    return result


def _reject_constant(value):
    raise ReleaseAssetError("catalogue contains a nonfinite JSON number")


def _name(value, *, nested=False):
    if (
        not isinstance(value, str) or not value or "\\" in value or ":" in value
        or any(ord(character) < 32 for character in value)
        or any(part in {"", ".", ".."} for part in value.split("/"))
        or (not nested and "/" in value)
    ):
        raise ReleaseAssetError("invalid or unsafe asset path")
    return value


def _record(value):
    if (
        not isinstance(value, dict) or set(value) != {"bytes", "sha256"}
        or type(value["bytes"]) is not int or not 0 <= value["bytes"] <= _MAX_EXPANDED_BYTES
        or not isinstance(value["sha256"], str)
        or re.fullmatch(r"[0-9a-f]{64}", value["sha256"]) is None
    ):
        raise ReleaseAssetError("invalid hash/size record")


def _safe_path(path):
    path = Path(os.path.abspath(Path(path).expanduser()))
    if any(item.is_symlink() for item in (path, *path.parents)):
        raise ReleaseAssetError("asset paths must not contain symlinks")
    return path


@contextmanager
def _reader(path):
    path = _safe_path(path)
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0))
    with os.fdopen(descriptor, "rb") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise ReleaseAssetError("asset inputs must be regular files")
        yield stream


def load_catalogue(path=CATALOGUE_PATH):
    """Validate the local catalogue; listing never performs a network request."""
    with _reader(path) as stream:
        content = stream.read(_MAX_CATALOGUE_BYTES + 1)
    if len(content) > _MAX_CATALOGUE_BYTES:
        raise ReleaseAssetError("catalogue exceeds the metadata size limit")
    data = json.loads(content, object_pairs_hook=_object, parse_constant=_reject_constant)
    if (
        not isinstance(data, dict) or set(data) != {"schema_version", "repository", "tag", "assets"}
        or data["schema_version"] != SCHEMA
        or not isinstance(data["repository"], str)
        or re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", data["repository"]) is None
        or not isinstance(data["tag"], str)
        or re.fullmatch(r"[A-Za-z0-9_.-]+", data["tag"]) is None
        or not isinstance(data["assets"], dict) or not data["assets"]
    ):
        raise ReleaseAssetError("unsupported release catalogue")
    for component in data["repository"].split("/"):
        _name(component)
    _name(data["tag"])
    for asset_id, asset in data["assets"].items():
        _name(asset_id)
        if re.fullmatch(r"[A-Za-z0-9_.-]+", asset_id) is None:
            raise ReleaseAssetError("invalid release asset identifier")
        if not isinstance(asset, dict) or set(asset) != {"filename", "url", "sha256", "bytes", "root", "files"}:
            raise ReleaseAssetError("unsupported release asset")
        _name(asset["filename"])
        _name(asset["root"])
        _record({key: asset[key] for key in ("bytes", "sha256")})
        expected_url = f"https://github.com/{data['repository']}/releases/download/{data['tag']}/{asset['filename']}"
        if (
            asset["root"] != asset_id or asset["filename"] != f"{asset_id}.tar.gz"
            or asset["url"] != expected_url or not 0 < asset["bytes"] <= _MAX_ARCHIVE_BYTES
            or not isinstance(asset["files"], dict) or not asset["files"]
        ):
            raise ReleaseAssetError("invalid release asset location or size")
        for name, record in asset["files"].items():
            _name(name, nested=True)
            _record(record)
            if any(str(parent) in asset["files"] for parent in PurePosixPath(name).parents):
                raise ReleaseAssetError("catalogue file conflicts with a parent directory")
        if sum(item["bytes"] for item in asset["files"].values()) > _MAX_EXPANDED_BYTES:
            raise ReleaseAssetError("expanded asset exceeds the size limit")
    return data


def _check_download_url(url, original):
    parsed = urlsplit(url)
    if (
        parsed.scheme != "https" or parsed.username is not None or parsed.password is not None
        or parsed.port not in (None, 443) or parsed.fragment
        or (url != original and parsed.hostname not in _ASSET_HOSTS)
    ):
        raise ReleaseAssetError("download redirect is outside the permitted GitHub asset hosts")


class _AssetRedirectHandler(HTTPRedirectHandler):
    def __init__(self, original):
        self.original = original

    def redirect_request(self, request, fp, code, msg, headers, newurl):
        _check_download_url(newurl, self.original)
        return super().redirect_request(request, fp, code, msg, headers, newurl)


def _copy_verified(source, target, record):
    digest, count = sha256(), 0
    while True:
        block = source.read(min(_BUFFER_BYTES, record["bytes"] - count + 1))
        if not block:
            break
        count += len(block)
        if count > record["bytes"]:
            raise ReleaseAssetError("asset content exceeds its declared size")
        target.write(block)
        digest.update(block)
    if count != record["bytes"] or digest.hexdigest() != record["sha256"]:
        raise ReleaseAssetError("asset content does not match its declared size/hash")


def _download(asset, target):
    opener = build_opener(_AssetRedirectHandler(asset["url"]))
    request = Request(asset["url"], headers={"User-Agent": "Entelecheia-retrieval-assets", "Accept-Encoding": "identity"})
    with opener.open(request, timeout=60) as response:
        _check_download_url(response.geturl(), asset["url"])
        if response.status != 200:
            raise ReleaseAssetError("asset download did not return HTTP 200")
        length = response.headers.get("Content-Length")
        if length is not None and (not length.isdigit() or int(length) != asset["bytes"]):
            raise ReleaseAssetError("download Content-Length differs from the catalogue")
        _copy_verified(response, target, asset)


def _extract(archive, stage, asset):
    expected = {f"{asset['root']}/{name}": record for name, record in asset["files"].items()}
    directories = {str(parent) for name in expected for parent in PurePosixPath(name).parents if str(parent) != "."}
    seen, extracted = set(), set()
    with tarfile.open(archive, mode="r|gz", ignore_zeros=True) as source:
        for member in source:
            name = member.name[:-1] if member.isdir() and member.name.endswith("/") else member.name
            _name(name, nested=True)
            if name in seen:
                raise ReleaseAssetError("archive contains duplicate members")
            seen.add(name)
            if member.isdir() and name in directories and member.size == 0:
                continue
            if member.type not in (tarfile.REGTYPE, tarfile.AREGTYPE) or member.sparse is not None or name not in expected:
                raise ReleaseAssetError("archive contains an unexpected or non-regular member")
            record = expected[name]
            if member.size != record["bytes"]:
                raise ReleaseAssetError("archive member size differs from the catalogue")
            destination = stage.joinpath(*PurePosixPath(name).parts[1:])
            destination.parent.mkdir(parents=True, exist_ok=True)
            with source.extractfile(member) as content, destination.open("xb") as target:
                _copy_verified(content, target, record)
            extracted.add(name)
    if extracted != set(expected):
        raise ReleaseAssetError("archive is missing catalogue files")


def fetch_asset(asset_id, output, *, catalogue=CATALOGUE_PATH, archive=None):
    """Install one exact snapshot into a new directory, or leave no partial output."""
    try:
        data = load_catalogue(catalogue)
        if asset_id not in data["assets"]:
            raise ReleaseAssetError(f"unknown release asset: {asset_id}")
        asset = data["assets"][asset_id]
        output = artifact_path(_safe_path(output))
        if output.exists():
            raise ReleaseAssetError("output already exists; refusing overwrite")
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
            work = Path(temporary)
            compressed = work / "archive.tar.gz"
            with compressed.open("xb") as target:
                if archive is None:
                    _download(asset, target)
                else:
                    with _reader(archive) as source:
                        _copy_verified(source, target, asset)
            stage = work / "snapshot"
            stage.mkdir()
            _extract(compressed, stage, asset)
            _safe_path(output)
            if output.exists():
                raise ReleaseAssetError("output appeared during operation; refusing overwrite")
            _rename_exclusive(stage, output)
        return asset
    except (OSError, ValueError, EOFError, tarfile.TarError, HTTPException) as error:
        if isinstance(error, ReleaseAssetError):
            raise
        raise ReleaseAssetError(f"Cannot install release asset: {error}") from error


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalogue", type=Path, default=CATALOGUE_PATH)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="List available assets without downloading")
    fetch = commands.add_parser("fetch", help="Download and verify one explicitly selected asset")
    fetch.add_argument("--asset", required=True)
    fetch.add_argument("--output", required=True, type=Path, help="New snapshot directory; existing paths are refused")
    fetch.add_argument("--archive", type=Path, help="Verify and install a local archive instead of downloading")
    args = parser.parse_args(argv)
    try:
        if args.command == "list":
            for asset_id, asset in load_catalogue(args.catalogue)["assets"].items():
                print(f"{asset_id}\t{asset['bytes']} bytes\t{len(asset['files'])} files")
        else:
            fetch_asset(args.asset, args.output, catalogue=args.catalogue, archive=args.archive)
            print(f"Verified snapshot installed: {args.output}")
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
