"""下载冻结的 DocBench 选集或由目录派生的子集。

``gdown`` 刻意仅作为安装依赖，不加入运行时包。上游字节默认保存在统一的
``bench_eval/docbench/source``，位于源码 checkout 之外。
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
import json
import os
from pathlib import Path, PureWindowsPath
import time
from typing import Any, Iterable, Sequence

import httpx

from evals.docbench.reproduce_or_run_script.config import docbench_root
from evals.docbench.reproduce_or_run_script.selection import (
    DEFAULT_BALANCED_DOMAIN_COUNTS,
    DEFAULT_BALANCED_DOMAIN_TYPE_COUNTS,
    DEFAULT_BALANCED_NORMALIZED_TYPE_COUNTS,
    DEFAULT_BALANCED_SEED,
    DEFAULT_MAX_QUESTIONS_PER_DOCUMENT,
    DEFAULT_SEED,
    SelectionValidationError,
    select_balanced_questions,
    select_document_ids,
)


DEFAULT_FOLDER_URL = (
    "https://drive.google.com/drive/folders/"
    "1yxhF1lFF2gKeTNc8Wh0EyBdMT3M4pDYr"
)
DOCBENCH_DOCUMENT_COUNT = 229
DOCBENCH_DOCUMENT_IDS = frozenset(range(DOCBENCH_DOCUMENT_COUNT))
CATALOG_MAPPING_SCHEMA_VERSION = "docbench-drive-catalog-map-v1"
BALANCED_SELECTION_DOCUMENT_COUNT = sum(DEFAULT_BALANCED_DOMAIN_COUNTS.values())


class DocBenchDownloadError(RuntimeError):
    """无法枚举或下载请求的上游文件。"""


def _file_fields(item: Any) -> tuple[str, str]:
    if hasattr(item, "id") and hasattr(item, "path"):
        payload = {"id": getattr(item, "id"), "path": getattr(item, "path")}
    elif is_dataclass(item):
        payload = asdict(item)
    elif isinstance(item, dict):
        payload = item
    else:
        payload = vars(item)
    file_id = str(payload.get("id") or "").strip()
    path = str(payload.get("path") or "").strip()
    if not file_id or not path:
        raise DocBenchDownloadError("Drive listing entry lacks id or path")
    return file_id, path


def _normalized_listing_file(item: Any) -> dict[str, str] | None:
    file_id, raw_path = _file_fields(item)
    if "\\" in raw_path:
        raise DocBenchDownloadError(f"unsafe Drive listing path: {raw_path}")
    path = Path(raw_path)
    windows_path = PureWindowsPath(raw_path)
    parts = path.parts
    if (
        path.is_absolute()
        or windows_path.is_absolute()
        or bool(windows_path.drive)
        or any(part in {".", ".."} for part in parts)
    ):
        raise DocBenchDownloadError(f"unsafe Drive listing path: {raw_path}")
    # gdown 返回相对于公开根目录的路径（通常为 ``<doc_id>/<filename>``）；
    # 同时容许显式的前导 ``data``。
    if parts and parts[0] == "data":
        parts = parts[1:]
    if len(parts) != 2 or not parts[0].isdigit():
        return None
    filename = parts[1]
    if (
        not filename
        or Path(filename).name != filename
        or bool(PureWindowsPath(filename).drive)
    ):
        raise DocBenchDownloadError(f"unsafe Drive listing path: {raw_path}")
    return {
        "id": file_id,
        "doc_id": str(int(parts[0])),
        "filename": filename,
    }


def _validated_document_file_pairs(
    files: Iterable[dict[str, str]],
    *,
    expected_doc_ids: set[int] | frozenset[int],
) -> list[dict[str, str]]:
    normalized = sorted(
        (dict(item) for item in files),
        key=lambda item: (int(item["doc_id"]), item["filename"]),
    )
    file_ids: set[str] = set()
    by_document: dict[int, list[dict[str, str]]] = {
        doc_id: [] for doc_id in expected_doc_ids
    }
    for item in normalized:
        file_id = item["id"]
        if file_id in file_ids:
            raise DocBenchDownloadError("Drive mapping has a duplicate file id")
        file_ids.add(file_id)
        doc_id = int(item["doc_id"])
        if doc_id not in expected_doc_ids:
            raise DocBenchDownloadError("Drive mapping contains an unexpected document")
        by_document[doc_id].append(item)

    bad: list[int] = []
    for doc_id in sorted(expected_doc_ids):
        document_files = by_document[doc_id]
        qa_files = [
            item
            for item in document_files
            if item["filename"] == f"{doc_id}_qa.jsonl"
        ]
        pdf_files = [
            item
            for item in document_files
            if Path(item["filename"]).suffix.casefold() == ".pdf"
        ]
        if len(document_files) != 2 or len(qa_files) != 1 or len(pdf_files) != 1:
            bad.append(doc_id)
    if bad:
        raise DocBenchDownloadError(
            "expected one PDF and one QA file for selected documents: "
            + ", ".join(map(str, bad))
        )
    return normalized


def _drive_files_for_documents(
    files: Iterable[Any],
    *,
    expected_doc_ids: set[int] | frozenset[int],
) -> list[dict[str, str]]:
    selected: list[dict[str, str]] = []
    for item in files:
        normalized = _normalized_listing_file(item)
        if normalized is not None and int(normalized["doc_id"]) in expected_doc_ids:
            selected.append(normalized)
    return _validated_document_file_pairs(
        selected,
        expected_doc_ids=expected_doc_ids,
    )


def selected_drive_files(
    files: Iterable[Any],
    *,
    selected_doc_ids: set[int],
) -> list[dict[str, str]]:
    return _drive_files_for_documents(
        files,
        expected_doc_ids=selected_doc_ids,
    )


def catalog_drive_files(files: Iterable[Any]) -> list[dict[str, str]]:
    """冻结完整的 229 份文档 Drive 清单，但不下载文件。"""

    return _drive_files_for_documents(
        files,
        expected_doc_ids=DOCBENCH_DOCUMENT_IDS,
    )


def _load_mapping(
    path: Path,
    *,
    folder_url: str,
    seed: str,
    selected_doc_ids: set[int],
) -> list[dict[str, str]] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DocBenchDownloadError(f"invalid cached Drive mapping: {path}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != (
        "docbench-drive-selection-map-v1"
    ):
        raise DocBenchDownloadError(f"unsupported cached Drive mapping: {path}")
    if payload.get("folder_url") != folder_url or payload.get("seed") != seed:
        raise DocBenchDownloadError("cached Drive mapping provenance differs")
    files = payload.get("files")
    if not isinstance(files, list):
        raise DocBenchDownloadError("cached Drive mapping files must be a list")
    if (
        payload.get("document_count") != len(selected_doc_ids)
        or payload.get("file_count") != len(selected_doc_ids) * 2
        or len(files) != len(selected_doc_ids) * 2
    ):
        raise DocBenchDownloadError("cached Drive mapping is incomplete")
    return _normalized_cached_files(
        files,
        expected_doc_ids=selected_doc_ids,
    )


def _normalized_cached_files(
    files: Iterable[Any],
    *,
    expected_doc_ids: set[int] | frozenset[int],
) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    for item in files:
        if not isinstance(item, dict) or set(item) != {"id", "doc_id", "filename"}:
            raise DocBenchDownloadError("cached Drive mapping has an invalid file entry")
        file_id = str(item["id"] or "").strip()
        doc_id_text = str(item["doc_id"] or "").strip()
        filename = str(item["filename"] or "").strip()
        if (
            not file_id
            or not doc_id_text.isdigit()
            or doc_id_text != str(int(doc_id_text))
            or int(doc_id_text) not in expected_doc_ids
            or not filename
            or Path(filename).name != filename
            or "\\" in filename
            or bool(PureWindowsPath(filename).drive)
        ):
            raise DocBenchDownloadError("cached Drive mapping has unsafe file metadata")
        normalized.append({"id": file_id, "doc_id": doc_id_text, "filename": filename})
    return _validated_document_file_pairs(
        normalized,
        expected_doc_ids=expected_doc_ids,
    )


def _load_catalog_mapping(
    path: Path,
    *,
    folder_url: str,
) -> list[dict[str, str]] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DocBenchDownloadError(f"invalid cached Drive mapping: {path}") from exc
    expected_keys = {
        "schema_version",
        "folder_url",
        "document_count",
        "file_count",
        "qa_file_count",
        "files",
    }
    if (
        not isinstance(payload, dict)
        or set(payload) != expected_keys
        or payload.get("schema_version") != CATALOG_MAPPING_SCHEMA_VERSION
    ):
        raise DocBenchDownloadError(f"unsupported cached Drive mapping: {path}")
    if payload.get("folder_url") != folder_url:
        raise DocBenchDownloadError("cached Drive mapping provenance differs")
    files = payload.get("files")
    if not isinstance(files, list):
        raise DocBenchDownloadError("cached Drive mapping files must be a list")
    if (
        payload.get("document_count") != DOCBENCH_DOCUMENT_COUNT
        or payload.get("file_count") != DOCBENCH_DOCUMENT_COUNT * 2
        or payload.get("qa_file_count") != DOCBENCH_DOCUMENT_COUNT
        or len(files) != DOCBENCH_DOCUMENT_COUNT * 2
    ):
        raise DocBenchDownloadError("cached Drive mapping is incomplete")
    normalized = _normalized_cached_files(
        files,
        expected_doc_ids=DOCBENCH_DOCUMENT_IDS,
    )
    if len(_qa_catalog_files(normalized)) != DOCBENCH_DOCUMENT_COUNT:
        raise DocBenchDownloadError("cached Drive mapping is incomplete")
    return normalized


def _catalog_mapping(
    *,
    folder_url: str,
    files: list[dict[str, str]],
) -> dict[str, Any]:
    return {
        "schema_version": CATALOG_MAPPING_SCHEMA_VERSION,
        "folder_url": folder_url,
        "document_count": DOCBENCH_DOCUMENT_COUNT,
        "file_count": len(files),
        "qa_file_count": DOCBENCH_DOCUMENT_COUNT,
        "files": files,
    }


def _write_mapping(path: Path, mapping: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(mapping, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_download(path: Path) -> None:
    with path.open("rb") as stream:
        prefix = stream.read(8)
    if path.suffix.casefold() == ".pdf":
        if not prefix.startswith(b"%PDF-"):
            raise DocBenchDownloadError(f"downloaded file is not a PDF: {path}")
        return
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        if not lines:
            raise ValueError("empty JSONL")
        for line in lines:
            if not isinstance(json.loads(line), dict):
                raise ValueError("JSONL record is not an object")
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise DocBenchDownloadError(f"downloaded file is not valid JSONL: {path}") from exc


def _download_one(client: httpx.Client, *, file_id: str, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.is_file() and output.stat().st_size > 0:
        _validate_download(output)
        return
    last_error: BaseException | None = None
    for attempt, delay in enumerate((0, 2, 5, 10), start=1):
        if delay:
            time.sleep(delay)
        temporary = output.with_name(
            f".{output.stem}.download{output.suffix}"
        )
        try:
            with client.stream(
                "GET",
                "https://drive.google.com/uc",
                params={"export": "download", "id": file_id},
            ) as response:
                response.raise_for_status()
                with temporary.open("wb") as stream:
                    for chunk in response.iter_bytes():
                        stream.write(chunk)
            _validate_download(temporary)
            os.replace(temporary, output)
            return
        except Exception as exc:
            last_error = exc
            temporary.unlink(missing_ok=True)
        print(
            json.dumps(
                {"download_retry": str(output), "attempt": attempt},
                ensure_ascii=False,
            ),
            flush=True,
        )
    raise DocBenchDownloadError(
        f"failed to download {output}: {last_error or 'empty result'}"
    )


def _list_drive_folder(
    *,
    folder_url: str,
    listing_output_root: Path,
) -> Iterable[Any]:
    try:
        import gdown
    except ImportError as exc:
        raise DocBenchDownloadError(
            "run ./scripts/bootstrap-local-runtime.sh on the supported platform; "
            "gdown is included in the locked eval dependency set"
        ) from exc
    listed = gdown.download_folder(
        url=folder_url,
        output=str(listing_output_root) + "/",
        skip_download=True,
        quiet=True,
    )
    if listed is None:
        raise DocBenchDownloadError("Drive folder listing returned no files")
    return listed


def _qa_catalog_files(files: Iterable[dict[str, str]]) -> list[dict[str, str]]:
    qa_files = [
        item
        for item in files
        if item["filename"] == f"{int(item['doc_id'])}_qa.jsonl"
    ]
    qa_files.sort(key=lambda item: int(item["doc_id"]))
    if (
        len(qa_files) != DOCBENCH_DOCUMENT_COUNT
        or {int(item["doc_id"]) for item in qa_files} != DOCBENCH_DOCUMENT_IDS
    ):
        raise DocBenchDownloadError("Drive QA catalog mapping is incomplete")
    return qa_files


def _require_complete_qa_catalog(data_root: Path) -> None:
    missing: list[int] = []
    for doc_id in sorted(DOCBENCH_DOCUMENT_IDS):
        qa_path = data_root / str(doc_id) / f"{doc_id}_qa.jsonl"
        if not qa_path.is_file():
            missing.append(doc_id)
            continue
        _validate_download(qa_path)
    if missing:
        raise DocBenchDownloadError(
            "balanced PDF selection requires the complete local QA catalog; "
            "missing document ids: " + ", ".join(map(str, missing))
        )


def _pdf_files_for_documents(
    files: Iterable[dict[str, str]],
    *,
    selected_doc_ids: set[int],
) -> list[dict[str, str]]:
    pdf_files = [
        item
        for item in files
        if int(item["doc_id"]) in selected_doc_ids
        and Path(item["filename"]).suffix.casefold() == ".pdf"
    ]
    pdf_files.sort(key=lambda item: int(item["doc_id"]))
    if (
        len(pdf_files) != len(selected_doc_ids)
        or {int(item["doc_id"]) for item in pdf_files} != selected_doc_ids
    ):
        raise DocBenchDownloadError(
            "cached Drive mapping lacks a unique PDF for the balanced selection"
        )
    return pdf_files


def _pending_downloads(
    files: Iterable[dict[str, str]],
    *,
    data_root: Path,
) -> list[dict[str, str]]:
    pending: list[dict[str, str]] = []
    for item in files:
        output = data_root / item["doc_id"] / item["filename"]
        if output.is_file() and output.stat().st_size > 0:
            _validate_download(output)
            continue
        pending.append(item)
    return pending


def _download_files(
    *,
    files: Sequence[dict[str, str]],
    data_root: Path,
) -> None:
    timeout = httpx.Timeout(300.0, connect=30.0)
    with httpx.Client(follow_redirects=True, timeout=timeout) as client:
        for index, item in enumerate(files, start=1):
            output = data_root / item["doc_id"] / item["filename"]
            print(
                json.dumps(
                    {"file": index, "of": len(files), "output": str(output)},
                    ensure_ascii=False,
                ),
                flush=True,
            )
            _download_one(client, file_id=item["id"], output=output)


def download_selected(
    *,
    data_root: Path,
    mapping_path: Path,
    folder_url: str = DEFAULT_FOLDER_URL,
    seed: str = DEFAULT_SEED,
) -> dict[str, Any]:
    selection = select_document_ids(seed=seed)
    selected_doc_ids = {doc_id for values in selection.values() for doc_id in values}
    files = _load_mapping(
        mapping_path,
        folder_url=folder_url,
        seed=seed,
        selected_doc_ids=selected_doc_ids,
    )
    if files is None:
        listed = _list_drive_folder(
            folder_url=folder_url,
            listing_output_root=data_root.parent,
        )
        files = selected_drive_files(listed, selected_doc_ids=selected_doc_ids)
        mapping = {
            "schema_version": "docbench-drive-selection-map-v1",
            "folder_url": folder_url,
            "seed": seed,
            "document_count": len(selected_doc_ids),
            "file_count": len(files),
            "files": files,
        }
        _write_mapping(mapping_path, mapping)
    else:
        mapping = {
            "schema_version": "docbench-drive-selection-map-v1",
            "folder_url": folder_url,
            "seed": seed,
            "document_count": len(selected_doc_ids),
            "file_count": len(files),
            "files": files,
        }
    _download_files(files=files, data_root=data_root)
    return mapping


def download_qa_catalog(
    *,
    data_root: Path,
    mapping_path: Path,
    folder_url: str = DEFAULT_FOLDER_URL,
) -> dict[str, Any]:
    """下载全部 QA JSONL，同时缓存完整的 Drive 文件映射。

    在传输任何内容前先写入完整的 458 项映射。因此恢复执行时会复用相同的 Drive ID
    和精确目录 URL，而 ``_download_one`` 会校验并跳过每个已存在的 QA 文件。PDF 条目
    继续冻结在映射中，供后续选择性下载使用，但不会进入本次传输循环。
    """

    files = _load_catalog_mapping(mapping_path, folder_url=folder_url)
    if files is None:
        listed = _list_drive_folder(
            folder_url=folder_url,
            listing_output_root=data_root.parent,
        )
        files = catalog_drive_files(listed)
        mapping = _catalog_mapping(folder_url=folder_url, files=files)
        _write_mapping(mapping_path, mapping)
    else:
        mapping = _catalog_mapping(folder_url=folder_url, files=files)
    _download_files(files=_qa_catalog_files(files), data_root=data_root)
    return mapping


def download_balanced_pdfs(
    *,
    data_root: Path,
    mapping_path: Path,
    folder_url: str = DEFAULT_FOLDER_URL,
    seed: str = DEFAULT_BALANCED_SEED,
) -> dict[str, Any]:
    """只下载从完整冻结 QA 目录中选出的 PDF。

    此模式刻意拒绝枚举 Drive。完整映射和所有规范 QA 文件必须已经存在，确保在开始
    传输 PDF 前，能够可复现地求解精确的默认配额矩阵。
    """

    files = _load_catalog_mapping(mapping_path, folder_url=folder_url)
    if files is None:
        raise DocBenchDownloadError(
            "balanced PDF selection requires an existing complete catalog mapping: "
            f"{mapping_path}"
        )
    _require_complete_qa_catalog(data_root)
    try:
        selection = select_balanced_questions(
            data_root,
            seed=seed,
            domain_counts=DEFAULT_BALANCED_DOMAIN_COUNTS,
            normalized_type_counts=DEFAULT_BALANCED_NORMALIZED_TYPE_COUNTS,
            domain_type_counts=DEFAULT_BALANCED_DOMAIN_TYPE_COUNTS,
            max_questions_per_document=DEFAULT_MAX_QUESTIONS_PER_DOCUMENT,
        )
    except SelectionValidationError as exc:
        raise DocBenchDownloadError(
            f"balanced question selection failed: {exc}"
        ) from exc
    selected_doc_ids = {item.doc_id for item in selection}
    if (
        len(selection) != BALANCED_SELECTION_DOCUMENT_COUNT
        or len(selected_doc_ids) != BALANCED_SELECTION_DOCUMENT_COUNT
    ):
        raise DocBenchDownloadError(
            "balanced question selection did not produce exactly "
            f"{BALANCED_SELECTION_DOCUMENT_COUNT} unique documents"
        )
    pdf_files = _pdf_files_for_documents(
        files,
        selected_doc_ids=selected_doc_ids,
    )
    candidates = _pending_downloads(pdf_files, data_root=data_root)
    result = {
        "selection_seed": seed,
        "selected_document_count": len(selected_doc_ids),
        "download_candidate_count": len(candidates),
    }
    print(
        json.dumps(
            {"mode": "balanced_pdfs_only", "phase": "download_plan", **result},
            ensure_ascii=False,
        ),
        flush=True,
    )
    _download_files(files=candidates, data_root=data_root)
    return result


def _add_cli_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--mapping",
        type=Path,
        default=None,
    )
    parser.add_argument("--folder-url", default=DEFAULT_FOLDER_URL)
    parser.add_argument("--seed", default=None)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--qa-catalog-only",
        action="store_true",
        help=(
            "download all 229 QA JSONL files, cache the complete Drive map, "
            "and never download PDFs"
        ),
    )
    mode.add_argument(
        "--balanced-pdfs-only",
        action="store_true",
        help=(
            "select 125 unique documents from the complete local QA catalog "
            "and download only their PDFs without relisting Drive"
        ),
    )


def _run_cli(args: argparse.Namespace) -> int:
    data_root = args.data_root
    mapping_path = args.mapping
    if data_root is None or mapping_path is None:
        source_dir = docbench_root() / "source"
        if data_root is None:
            data_root = source_dir / "data"
        if mapping_path is None:
            mapping_path = source_dir / (
                "drive_catalog_map.json"
                if args.qa_catalog_only or args.balanced_pdfs_only
                else "selected_drive_map.json"
            )
    if args.qa_catalog_only:
        mapping = download_qa_catalog(
            data_root=data_root,
            mapping_path=mapping_path,
            folder_url=args.folder_url,
        )
        summary = {
            "status": "complete",
            "mode": "qa_catalog_only",
            "document_count": mapping["document_count"],
            "mapped_file_count": mapping["file_count"],
            "qa_file_count": mapping["qa_file_count"],
            "mapping": str(mapping_path),
        }
    elif args.balanced_pdfs_only:
        result = download_balanced_pdfs(
            data_root=data_root,
            mapping_path=mapping_path,
            folder_url=args.folder_url,
            seed=args.seed or DEFAULT_BALANCED_SEED,
        )
        summary = {
            "status": "complete",
            "mode": "balanced_pdfs_only",
            **result,
            "mapping": str(mapping_path),
        }
    else:
        mapping = download_selected(
            data_root=data_root,
            mapping_path=mapping_path,
            folder_url=args.folder_url,
            seed=args.seed or DEFAULT_SEED,
        )
        summary = {
            "status": "complete",
            "document_count": mapping["document_count"],
            "file_count": mapping["file_count"],
            "mapping": str(mapping_path),
        }
    print(
        json.dumps(summary, ensure_ascii=False)
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    _add_cli_arguments(parser)
    return _run_cli(parser.parse_args(argv))
