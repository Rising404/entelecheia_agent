"""Archive frozen retrieval indexes as raw 64 MiB parts; restore exact DB bytes.

This module never opens SQLite, changes the source, compresses data or loads model
code. The original index manifest is preserved byte for byte for index reuse.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import ctypes
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import stat
import sys
import tempfile
from typing import BinaryIO, Iterator, Sequence

from .paths import artifact_path


PART_BYTES = 64 * 1024 * 1024
_BUFFER_BYTES = 1024 * 1024
_MAX_MANIFEST_BYTES = 4 * 1024 * 1024
_ARCHIVE_SCHEMA = "docbench-hybrid-index-archive-v1"
_DATABASE_NAME = "retrieval.sqlite"
_MANIFEST_NAME = "manifest.json"
_ARCHIVE_NAME = "archive.json"


class IndexArchiveError(ValueError):
    """The files cannot attest an unchanged, complete index archive."""


def _path(path: Path) -> Path:
    requested = Path(os.path.abspath(Path(path).expanduser()))
    if any(candidate.is_symlink() for candidate in (requested, *requested.parents)):
        raise IndexArchiveError("archive paths must not contain symlinks")
    return artifact_path(requested)


def _locations(source: Path, output: Path) -> tuple[Path, Path]:
    source, output = _path(source), _path(output)
    if not source.is_dir():
        raise IndexArchiveError("source must be an existing directory")
    if output.exists():
        raise IndexArchiveError("output already exists; refusing overwrite")
    if output.is_relative_to(source) or source.is_relative_to(output):
        raise IndexArchiveError("source and output directories must be disjoint")
    return source, output


@contextmanager
def _reader(path: Path) -> Iterator[BinaryIO]:
    _path(path)
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise IndexArchiveError("archive inputs must be regular files")
        yield stream
        after = os.fstat(stream.fileno())
        current = path.lstat()
        def identity(value):
            return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns
        if identity(before) != identity(after) or identity(before) != identity(current):
            raise IndexArchiveError("source file changed while being read")


def _metadata(path: Path) -> bytes:
    with _reader(path) as stream:
        content = stream.read(_MAX_MANIFEST_BYTES + 1)
    if len(content) > _MAX_MANIFEST_BYTES:
        raise IndexArchiveError("manifest exceeds the metadata size limit")
    return content


def _object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise IndexArchiveError("manifest contains duplicate JSON keys")
        result[key] = value
    return result


def _reject_constant(value: str):
    raise IndexArchiveError("manifest contains a nonfinite JSON number")


def _document(content: bytes) -> dict:
    value = json.loads(content, object_pairs_hook=_object, parse_constant=_reject_constant)
    if not isinstance(value, dict):
        raise IndexArchiveError("manifest must be a JSON object")
    return value


def _hash(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _index_manifest(content: bytes) -> dict:
    value = _document(content)
    if value.get("schema_version") != "docbench-hybrid-index-v1" or not _hash(value.get("database_sha256")):
        raise IndexArchiveError("invalid frozen index manifest")
    return value


def _pending_transactions(root: Path) -> None:
    for suffix in ("-wal", "-journal", "-shm"):
        path = root / (_DATABASE_NAME + suffix)
        if path.is_symlink():
            raise IndexArchiveError("SQLite sidecars must not be symlinks")
        if path.exists():
            if not path.is_file():
                raise IndexArchiveError("SQLite sidecars must be regular files")
            if suffix != "-shm" and path.stat().st_size:
                raise IndexArchiveError("source index has a pending WAL/journal transaction")


def _record(content: bytes) -> dict:
    return {"bytes": len(content), "sha256": sha256(content).hexdigest()}


def _copy_part(source: BinaryIO, target: BinaryIO, size: int, whole_hash) -> dict:
    digest = sha256()
    remaining = size
    while remaining:
        block = source.read(min(_BUFFER_BYTES, remaining))
        if not block:
            raise IndexArchiveError("input is shorter than its declared size")
        target.write(block)
        digest.update(block)
        whole_hash.update(block)
        remaining -= len(block)
    return {"bytes": size, "sha256": digest.hexdigest()}


def _rename_exclusive(source: Path, target: Path) -> None:
    """Atomic publication must not replace even a concurrently created empty dir."""
    if os.name == "nt":
        os.rename(source, target)  # Windows rename already refuses an existing target.
        return
    library = ctypes.CDLL(None, use_errno=True)
    if sys.platform == "darwin" and hasattr(library, "renamex_np"):
        rename = library.renamex_np
        rename.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        args = (os.fsencode(source), os.fsencode(target), 0x00000004)  # RENAME_EXCL
    elif sys.platform.startswith("linux") and hasattr(library, "renameat2"):
        rename = library.renameat2
        rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        args = (-100, os.fsencode(source), -100, os.fsencode(target), 1)  # AT_FDCWD, RENAME_NOREPLACE
    else:
        raise IndexArchiveError("platform does not support atomic no-overwrite directory publication")
    rename.restype = ctypes.c_int
    if rename(*args):
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code), os.fspath(target))


def _publish(stage: Path, output: Path) -> None:
    _path(output)
    if output.exists():
        raise IndexArchiveError("output appeared during operation; refusing overwrite")
    _rename_exclusive(stage, output)


def pack_index(index_dir: Path, output_dir: Path) -> dict:
    """Write a new archive without modifying or deleting the source index."""
    try:
        source, output = _locations(index_dir, output_dir)
        manifest_bytes = _metadata(source / _MANIFEST_NAME)
        manifest = _index_manifest(manifest_bytes)
        _pending_transactions(source)
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
            stage = Path(temporary)
            parts, whole_hash = [], sha256()
            with _reader(source / _DATABASE_NAME) as database:
                size = os.fstat(database.fileno()).st_size
                if size <= 0:
                    raise IndexArchiveError("source database must be nonempty")
                for offset in range(0, size, PART_BYTES):
                    name = f"{_DATABASE_NAME}.part-{len(parts):05d}"
                    with (stage / name).open("xb") as part:
                        record = _copy_part(database, part, min(PART_BYTES, size - offset), whole_hash)
                    parts.append({"name": name, **record})
                if database.read(1):
                    raise IndexArchiveError("source database grew while being read")
            if whole_hash.hexdigest() != manifest["database_sha256"]:
                raise IndexArchiveError("source database hash does not match its manifest")
            if _metadata(source / _MANIFEST_NAME) != manifest_bytes:
                raise IndexArchiveError("source index manifest changed during archiving")
            _pending_transactions(source)
            archive = {
                "schema_version": _ARCHIVE_SCHEMA, "format": "raw-byte-parts",
                "part_bytes": PART_BYTES,
                "database": {"bytes": size, "sha256": whole_hash.hexdigest()},
                "index_manifest": _record(manifest_bytes), "parts": parts,
            }
            (stage / _MANIFEST_NAME).write_bytes(manifest_bytes)
            (stage / _ARCHIVE_NAME).write_text(json.dumps(archive, sort_keys=True, indent=2) + "\n")
            _publish(stage, output)
        return archive
    except (OSError, ValueError) as error:
        if isinstance(error, IndexArchiveError):
            raise
        raise IndexArchiveError(f"Cannot archive index: {error}") from error


def _validate_archive(root: Path, archive: dict, manifest_bytes: bytes) -> None:
    if (
        set(archive) != {"schema_version", "format", "part_bytes", "database", "index_manifest", "parts"}
        or archive["schema_version"] != _ARCHIVE_SCHEMA
        or archive["format"] != "raw-byte-parts"
        or type(archive["part_bytes"]) is not int or archive["part_bytes"] != PART_BYTES
        or not isinstance(archive["parts"], list) or not archive["parts"]
    ):
        raise IndexArchiveError("unsupported index archive layout")
    for key in ("database", "index_manifest"):
        record = archive[key]
        if (
            not isinstance(record, dict) or set(record) != {"bytes", "sha256"}
            or type(record["bytes"]) is not int or record["bytes"] <= 0 or not _hash(record["sha256"])
        ):
            raise IndexArchiveError("invalid archive hash/size record")
    manifest = _index_manifest(manifest_bytes)
    if archive["index_manifest"] != _record(manifest_bytes) or archive["database"]["sha256"] != manifest["database_sha256"]:
        raise IndexArchiveError("archive is not bound to the exact index manifest")
    total = 0
    names = {_ARCHIVE_NAME, _MANIFEST_NAME}
    for index, record in enumerate(archive["parts"]):
        expected_name = f"{_DATABASE_NAME}.part-{index:05d}"
        expected_size = min(PART_BYTES, archive["database"]["bytes"] - total)
        if (
            not isinstance(record, dict) or set(record) != {"name", "bytes", "sha256"}
            or record["name"] != expected_name or type(record["bytes"]) is not int
            or not 0 < record["bytes"] == expected_size or not _hash(record["sha256"])
        ):
            raise IndexArchiveError("invalid part path, order, hash or size")
        total += record["bytes"]
        names.add(expected_name)
    if total != archive["database"]["bytes"]:
        raise IndexArchiveError("archive total size does not equal its parts")
    if {path.name for path in root.iterdir()} != names:
        raise IndexArchiveError("archive has missing or extra files")


def restore_index(archive_dir: Path, output_dir: Path) -> dict:
    """Restore exact bytes to a new directory accepted by the index reuse path."""
    try:
        source, output = _locations(archive_dir, output_dir)
        archive_bytes = _metadata(source / _ARCHIVE_NAME)
        manifest_bytes = _metadata(source / _MANIFEST_NAME)
        archive = _document(archive_bytes)
        _validate_archive(source, archive, manifest_bytes)
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
            stage = Path(temporary)
            whole_hash = sha256()
            with (stage / _DATABASE_NAME).open("xb") as database:
                for record in archive["parts"]:
                    with _reader(source / record["name"]) as part:
                        actual = _copy_part(part, database, record["bytes"], whole_hash)
                        if part.read(1):
                            raise IndexArchiveError("part exceeds its declared size")
                    if actual != {key: record[key] for key in ("bytes", "sha256")}:
                        raise IndexArchiveError("part hash/size mismatch")
            actual = {"bytes": (stage / _DATABASE_NAME).stat().st_size, "sha256": whole_hash.hexdigest()}
            if actual != archive["database"]:
                raise IndexArchiveError("restored database hash/size mismatch")
            if _metadata(source / _ARCHIVE_NAME) != archive_bytes or _metadata(source / _MANIFEST_NAME) != manifest_bytes:
                raise IndexArchiveError("source manifests changed during restoration")
            _validate_archive(source, archive, manifest_bytes)
            (stage / _MANIFEST_NAME).write_bytes(manifest_bytes)
            _publish(stage, output)
        return archive
    except (OSError, ValueError) as error:
        if isinstance(error, IndexArchiveError):
            raise
        raise IndexArchiveError(f"Cannot restore index: {error}") from error


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    pack = commands.add_parser("pack", help="split a frozen index into raw 64 MiB parts")
    pack.add_argument("--index", required=True, type=Path)
    pack.add_argument("--output", required=True, type=Path)
    restore = commands.add_parser("restore", help="restore and verify the exact original index")
    restore.add_argument("--archive", required=True, type=Path)
    restore.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        archive = pack_index(args.index, args.output) if args.command == "pack" else restore_index(args.archive, args.output)
    except IndexArchiveError as error:
        parser.exit(2, f"Index archive error: {error}\n")
    print(json.dumps({"database": archive["database"], "part_count": len(archive["parts"])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
