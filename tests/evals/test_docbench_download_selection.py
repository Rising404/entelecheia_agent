from collections import namedtuple
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from evals.docbench.reproduce_or_run_script import download_selection as downloader
from evals.docbench.reproduce_or_run_script.download_selection import (
    DEFAULT_FOLDER_URL,
    DOCBENCH_DOCUMENT_COUNT,
    DocBenchDownloadError,
    _download_one,
    catalog_drive_files,
    download_balanced_pdfs,
    download_qa_catalog,
    main,
    selected_drive_files,
)


def _write_complete_catalog_mapping(path: Path) -> None:
    files = catalog_drive_files(_complete_drive_listing())
    path.write_text(
        json.dumps(
            {
                "schema_version": downloader.CATALOG_MAPPING_SCHEMA_VERSION,
                "folder_url": DEFAULT_FOLDER_URL,
                "document_count": DOCBENCH_DOCUMENT_COUNT,
                "file_count": len(files),
                "qa_file_count": DOCBENCH_DOCUMENT_COUNT,
                "files": files,
            }
        ),
        encoding="utf-8",
    )


def _write_complete_qa_catalog(data_root: Path) -> None:
    for doc_id in range(DOCBENCH_DOCUMENT_COUNT):
        qa_path = data_root / str(doc_id) / f"{doc_id}_qa.jsonl"
        qa_path.parent.mkdir(parents=True, exist_ok=True)
        qa_path.write_text('{"question":"q"}\n', encoding="utf-8")


def _complete_drive_listing() -> list[SimpleNamespace]:
    files: list[SimpleNamespace] = []
    for doc_id in range(DOCBENCH_DOCUMENT_COUNT):
        files.extend(
            [
                SimpleNamespace(
                    id=f"pdf-{doc_id}",
                    path=f"data/{doc_id}/source-{doc_id}.pdf",
                ),
                SimpleNamespace(
                    id=f"qa-{doc_id}",
                    path=f"data/{doc_id}/{doc_id}_qa.jsonl",
                ),
            ]
        )
    return files


def test_selected_drive_files_filters_and_orders_two_files_per_document() -> None:
    files = [
        SimpleNamespace(id="pdf-2", path="data/2/z.pdf"),
        SimpleNamespace(id="qa-1", path="1/1_qa.jsonl"),
        SimpleNamespace(id="ignored", path="3/3_qa.jsonl"),
        SimpleNamespace(id="pdf-1", path="1/a.pdf"),
        SimpleNamespace(id="qa-2", path="data/2/2_qa.jsonl"),
    ]
    assert selected_drive_files(files, selected_doc_ids={1, 2}) == [
        {"id": "qa-1", "doc_id": "1", "filename": "1_qa.jsonl"},
        {"id": "pdf-1", "doc_id": "1", "filename": "a.pdf"},
        {"id": "qa-2", "doc_id": "2", "filename": "2_qa.jsonl"},
        {"id": "pdf-2", "doc_id": "2", "filename": "z.pdf"},
    ]


def test_selected_drive_files_rejects_incomplete_document() -> None:
    with pytest.raises(DocBenchDownloadError, match="expected one PDF"):
        selected_drive_files(
            [SimpleNamespace(id="qa", path="1/1_qa.jsonl")],
            selected_doc_ids={1},
        )


def test_selected_drive_files_accepts_gdown_namedtuple_shape() -> None:
    DriveFile = namedtuple("DriveFile", ("id", "path", "local_path"))
    files = [
        DriveFile("qa", "1/1_qa.jsonl", "/tmp/1/1_qa.jsonl"),
        DriveFile("pdf", "1/source.pdf", "/tmp/1/source.pdf"),
    ]
    assert len(selected_drive_files(files, selected_doc_ids={1})) == 2


def test_catalog_drive_files_freezes_all_document_file_pairs() -> None:
    files = catalog_drive_files(reversed(_complete_drive_listing()))

    assert len(files) == DOCBENCH_DOCUMENT_COUNT * 2
    assert files[0] == {
        "id": "qa-0",
        "doc_id": "0",
        "filename": "0_qa.jsonl",
    }
    assert files[-1] == {
        "id": f"pdf-{DOCBENCH_DOCUMENT_COUNT - 1}",
        "doc_id": str(DOCBENCH_DOCUMENT_COUNT - 1),
        "filename": f"source-{DOCBENCH_DOCUMENT_COUNT - 1}.pdf",
    }


@pytest.mark.parametrize(
    "unsafe_path",
    [
        "data/0/../escaped.pdf",
        "data/0/C:escaped.pdf",
        "data\\0\\escaped.pdf",
    ],
)
def test_catalog_drive_files_rejects_unsafe_listing_paths(unsafe_path: str) -> None:
    files = _complete_drive_listing()
    files[0] = SimpleNamespace(
        id="unsafe",
        path=unsafe_path,
    )

    with pytest.raises(DocBenchDownloadError, match="unsafe Drive listing path"):
        catalog_drive_files(files)


def test_catalog_drive_files_requires_one_named_qa_and_one_pdf_per_document() -> None:
    files = _complete_drive_listing()
    files = [item for item in files if item.id != "qa-117"]

    with pytest.raises(DocBenchDownloadError, match="117"):
        catalog_drive_files(files)


def test_download_one_streams_and_validates_jsonl(tmp_path: Path) -> None:
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            content=b'{"question":"q"}\n',
            request=request,
        )
    )
    output = tmp_path / "1_qa.jsonl"
    with httpx.Client(transport=transport) as client:
        _download_one(client, file_id="file-id", output=output)
    assert output.read_text(encoding="utf-8") == '{"question":"q"}\n'


def test_download_one_rejects_invalid_jsonl_without_publishing_it(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "evals.docbench.reproduce_or_run_script.download_selection.time.sleep",
        lambda _delay: None,
    )
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            content=b"not-json\n",
            request=request,
        )
    )
    output = tmp_path / "1_qa.jsonl"

    with httpx.Client(transport=transport) as client:
        with pytest.raises(DocBenchDownloadError, match="failed to download"):
            _download_one(client, file_id="file-id", output=output)

    assert not output.exists()
    assert not (tmp_path / ".1_qa.download.jsonl").exists()


def test_download_one_preserves_pdf_type_during_temporary_validation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "evals.docbench.reproduce_or_run_script.download_selection.time.sleep",
        lambda _delay: None,
    )
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            content=b"%PDF-1.7\n%%EOF\n",
            request=request,
        )
    )
    output = tmp_path / "source.pdf"
    with httpx.Client(transport=transport) as client:
        _download_one(client, file_id="file-id", output=output)
    assert output.read_bytes().startswith(b"%PDF-")


def test_download_qa_catalog_caches_complete_mapping_without_downloading_pdfs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    listed = _complete_drive_listing()
    monkeypatch.setattr(
        downloader,
        "_list_drive_folder",
        lambda **_kwargs: listed,
    )
    downloaded: list[tuple[str, Path]] = []

    def fake_download(_client, *, file_id: str, output: Path) -> None:
        downloaded.append((file_id, output))
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text('{"question":"q"}\n', encoding="utf-8")

    monkeypatch.setattr(downloader, "_download_one", fake_download)
    mapping_path = tmp_path / "cache" / "drive_catalog.json"
    data_root = tmp_path / "data"

    mapping = download_qa_catalog(
        data_root=data_root,
        mapping_path=mapping_path,
        folder_url=DEFAULT_FOLDER_URL,
    )

    assert mapping["document_count"] == DOCBENCH_DOCUMENT_COUNT
    assert mapping["file_count"] == DOCBENCH_DOCUMENT_COUNT * 2
    assert mapping["qa_file_count"] == DOCBENCH_DOCUMENT_COUNT
    assert len(downloaded) == DOCBENCH_DOCUMENT_COUNT
    assert all(file_id.startswith("qa-") for file_id, _output in downloaded)
    assert all(output.name.endswith("_qa.jsonl") for _file_id, output in downloaded)
    assert not list(data_root.rglob("*.pdf"))
    cached = json.loads(mapping_path.read_text(encoding="utf-8"))
    assert cached["folder_url"] == DEFAULT_FOLDER_URL
    assert len(cached["files"]) == DOCBENCH_DOCUMENT_COUNT * 2


def test_download_selected_still_resumes_from_its_v1_cached_mapping(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    selected = downloader.select_document_ids(seed=downloader.DEFAULT_SEED)
    selected_doc_ids = sorted(
        doc_id for domain_ids in selected.values() for doc_id in domain_ids
    )
    files: list[dict[str, str]] = []
    for doc_id in selected_doc_ids:
        files.extend(
            [
                {
                    "id": f"qa-{doc_id}",
                    "doc_id": str(doc_id),
                    "filename": f"{doc_id}_qa.jsonl",
                },
                {
                    "id": f"pdf-{doc_id}",
                    "doc_id": str(doc_id),
                    "filename": f"source-{doc_id}.pdf",
                },
            ]
        )
    mapping_path = tmp_path / "selected_drive_map.json"
    mapping_path.write_text(
        json.dumps(
            {
                "schema_version": "docbench-drive-selection-map-v1",
                "folder_url": DEFAULT_FOLDER_URL,
                "seed": downloader.DEFAULT_SEED,
                "document_count": len(selected_doc_ids),
                "file_count": len(files),
                "files": files,
            }
        ),
        encoding="utf-8",
    )
    transferred: list[dict[str, str]] = []
    monkeypatch.setattr(
        downloader,
        "_list_drive_folder",
        lambda **_kwargs: pytest.fail("v1 cache must prevent relisting"),
    )
    monkeypatch.setattr(
        downloader,
        "_download_files",
        lambda *, files, data_root: transferred.extend(files),
    )

    mapping = downloader.download_selected(
        data_root=tmp_path / "data",
        mapping_path=mapping_path,
    )

    assert mapping["document_count"] == len(selected_doc_ids)
    assert mapping["file_count"] == len(files)
    assert transferred == files


def test_download_qa_catalog_resumes_from_frozen_mapping_and_valid_files(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    listed = _complete_drive_listing()
    list_calls = 0

    def fake_list(**_kwargs):
        nonlocal list_calls
        list_calls += 1
        return listed

    monkeypatch.setattr(downloader, "_list_drive_folder", fake_list)
    writes: list[Path] = []

    def resumable_download(_client, *, file_id: str, output: Path) -> None:
        if output.is_file():
            downloader._validate_download(output)
            return
        writes.append(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text('{"question":"q"}\n', encoding="utf-8")

    monkeypatch.setattr(downloader, "_download_one", resumable_download)
    mapping_path = tmp_path / "drive_catalog.json"
    data_root = tmp_path / "data"

    download_qa_catalog(data_root=data_root, mapping_path=mapping_path)
    assert len(writes) == DOCBENCH_DOCUMENT_COUNT
    missing = data_root / "117" / "117_qa.jsonl"
    missing.unlink()
    writes.clear()

    download_qa_catalog(data_root=data_root, mapping_path=mapping_path)

    assert list_calls == 1
    assert writes == [missing]


def test_download_balanced_pdfs_uses_frozen_defaults_and_skips_valid_pdf(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    mapping_path = tmp_path / "drive_catalog.json"
    data_root = tmp_path / "data"
    _write_complete_catalog_mapping(mapping_path)
    _write_complete_qa_catalog(data_root)
    existing_pdf = data_root / "1" / "source-1.pdf"
    existing_pdf.write_bytes(b"%PDF-1.7\n%%EOF\n")
    captured: dict[str, object] = {}

    def fake_select(root: Path, **kwargs):
        captured["data_root"] = root
        captured.update(kwargs)
        return tuple(SimpleNamespace(doc_id=doc_id) for doc_id in range(1, 126))

    transferred: list[dict[str, str]] = []
    monkeypatch.setattr(downloader, "select_balanced_questions", fake_select)
    monkeypatch.setattr(
        downloader,
        "_list_drive_folder",
        lambda **_kwargs: pytest.fail("balanced mode must never relist Drive"),
    )
    monkeypatch.setattr(
        downloader,
        "_download_files",
        lambda *, files, data_root: transferred.extend(files),
    )

    result = download_balanced_pdfs(
        data_root=data_root,
        mapping_path=mapping_path,
    )

    assert captured == {
        "data_root": data_root,
        "seed": downloader.DEFAULT_BALANCED_SEED,
        "domain_counts": downloader.DEFAULT_BALANCED_DOMAIN_COUNTS,
        "normalized_type_counts": (
            downloader.DEFAULT_BALANCED_NORMALIZED_TYPE_COUNTS
        ),
        "domain_type_counts": downloader.DEFAULT_BALANCED_DOMAIN_TYPE_COUNTS,
        "max_questions_per_document": (
            downloader.DEFAULT_MAX_QUESTIONS_PER_DOCUMENT
        ),
    }
    assert result == {
        "selection_seed": downloader.DEFAULT_BALANCED_SEED,
        "selected_document_count": 125,
        "download_candidate_count": 124,
    }
    assert len(transferred) == 124
    assert all(item["filename"].endswith(".pdf") for item in transferred)
    assert {int(item["doc_id"]) for item in transferred} == set(range(2, 126))


def test_download_balanced_pdfs_requires_existing_complete_mapping(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        downloader,
        "_list_drive_folder",
        lambda **_kwargs: pytest.fail("balanced mode must never relist Drive"),
    )

    with pytest.raises(DocBenchDownloadError, match="existing complete catalog"):
        download_balanced_pdfs(
            data_root=tmp_path / "data",
            mapping_path=tmp_path / "missing-map.json",
        )


def test_download_balanced_pdfs_requires_every_local_qa_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    mapping_path = tmp_path / "drive_catalog.json"
    data_root = tmp_path / "data"
    _write_complete_catalog_mapping(mapping_path)
    _write_complete_qa_catalog(data_root)
    (data_root / "117" / "117_qa.jsonl").unlink()
    monkeypatch.setattr(
        downloader,
        "select_balanced_questions",
        lambda *_args, **_kwargs: pytest.fail("missing QA must fail before selection"),
    )

    with pytest.raises(DocBenchDownloadError, match="missing document ids: 117"):
        download_balanced_pdfs(
            data_root=data_root,
            mapping_path=mapping_path,
        )


def test_download_balanced_pdfs_rejects_infeasible_or_nonunique_selection(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    mapping_path = tmp_path / "drive_catalog.json"
    data_root = tmp_path / "data"
    _write_complete_catalog_mapping(mapping_path)
    _write_complete_qa_catalog(data_root)
    monkeypatch.setattr(
        downloader,
        "select_balanced_questions",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            downloader.SelectionValidationError("infeasible fixture")
        ),
    )

    with pytest.raises(DocBenchDownloadError, match="infeasible fixture"):
        download_balanced_pdfs(
            data_root=data_root,
            mapping_path=mapping_path,
        )

    monkeypatch.setattr(
        downloader,
        "select_balanced_questions",
        lambda *_args, **_kwargs: tuple(
            SimpleNamespace(doc_id=1) for _index in range(125)
        ),
    )
    with pytest.raises(DocBenchDownloadError, match="125 unique documents"):
        download_balanced_pdfs(
            data_root=data_root,
            mapping_path=mapping_path,
        )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda payload: payload.update({"folder_url": "https://example.invalid"}),
            "provenance differs",
        ),
        (
            lambda payload: payload["files"].pop(),
            "incomplete",
        ),
        (
            lambda payload: payload["files"][0].update(
                {"filename": "../escaped.jsonl"}
            ),
            "unsafe file metadata",
        ),
        (
            lambda payload: payload["files"][0].update({"doc_id": "00"}),
            "unsafe file metadata",
        ),
        (
            lambda payload: payload["files"][0].update(
                {"id": payload["files"][1]["id"]}
            ),
            "duplicate file id",
        ),
    ],
)
def test_download_qa_catalog_rejects_drifted_cached_mapping(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mutate,
    message: str,
) -> None:
    files = catalog_drive_files(_complete_drive_listing())
    payload = {
        "schema_version": "docbench-drive-catalog-map-v1",
        "folder_url": DEFAULT_FOLDER_URL,
        "document_count": DOCBENCH_DOCUMENT_COUNT,
        "file_count": len(files),
        "qa_file_count": DOCBENCH_DOCUMENT_COUNT,
        "files": files,
    }
    mutate(payload)
    mapping_path = tmp_path / "drive_catalog.json"
    mapping_path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(
        downloader,
        "_list_drive_folder",
        lambda **_kwargs: pytest.fail("cached mapping must prevent relisting"),
    )

    with pytest.raises(DocBenchDownloadError, match=message):
        download_qa_catalog(
            data_root=tmp_path / "data",
            mapping_path=mapping_path,
        )


def test_main_catalog_modes_dispatch_without_changing_default_mode(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[tuple[str, Path]] = []

    def fake_catalog(**kwargs):
        calls.append(("catalog", kwargs["mapping_path"]))
        return {
            "document_count": DOCBENCH_DOCUMENT_COUNT,
            "file_count": DOCBENCH_DOCUMENT_COUNT * 2,
            "qa_file_count": DOCBENCH_DOCUMENT_COUNT,
        }

    def fake_selected(**kwargs):
        calls.append(("selected", kwargs["mapping_path"]))
        return {"document_count": 100, "file_count": 200}

    def fake_balanced(**kwargs):
        calls.append(("balanced", kwargs["mapping_path"]))
        assert kwargs["seed"] == downloader.DEFAULT_BALANCED_SEED
        return {
            "selection_seed": kwargs["seed"],
            "selected_document_count": 125,
            "download_candidate_count": 120,
        }

    monkeypatch.setattr(downloader, "download_qa_catalog", fake_catalog)
    monkeypatch.setattr(downloader, "download_selected", fake_selected)
    monkeypatch.setattr(downloader, "download_balanced_pdfs", fake_balanced)
    explicit_mapping = tmp_path / "catalog.json"

    assert main(["--qa-catalog-only", "--mapping", str(explicit_mapping)]) == 0
    assert calls == [("catalog", explicit_mapping)]
    calls.clear()

    assert main(["--mapping", str(tmp_path / "selected.json")]) == 0
    assert calls == [("selected", tmp_path / "selected.json")]
    calls.clear()

    assert main(["--qa-catalog-only"]) == 0
    assert calls == [("catalog", downloader.DEFAULT_QA_CATALOG_MAPPING_PATH)]
    calls.clear()

    assert main(["--balanced-pdfs-only"]) == 0
    assert calls == [("balanced", downloader.DEFAULT_QA_CATALOG_MAPPING_PATH)]
    summary = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert summary["selection_seed"] == downloader.DEFAULT_BALANCED_SEED
    assert summary["selected_document_count"] == 125
    assert summary["download_candidate_count"] == 120
    calls.clear()

    assert main([]) == 0
    assert calls == [("selected", downloader.DEFAULT_SELECTED_MAPPING_PATH)]


def test_main_catalog_modes_are_mutually_exclusive() -> None:
    with pytest.raises(SystemExit):
        main(["--qa-catalog-only", "--balanced-pdfs-only"])
