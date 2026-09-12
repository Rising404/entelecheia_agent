from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import pytest

from evals.docbench.reproduce_or_run_script.config import (
    load_docbench_config,
)
from evals.docbench.reproduce_or_run_script.selection import (
    SelectionValidationError,
    load_selection_manifest,
    select_document_ids,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SELECTION_PATH = (
    PROJECT_ROOT / "evals/docbench/selections/bge_m3_live_1_v1.json"
)
CONFIG_PATH = PROJECT_ROOT / "evals/docbench/configs/l1_bge_m3_live_1.yaml"


def test_bge_m3_live_one_case_selection_is_frozen_and_content_free(
    tmp_path: Path,
) -> None:
    raw_bytes = SELECTION_PATH.read_bytes()
    raw = json.loads(raw_bytes)

    assert sha256(raw_bytes).hexdigest() == (
        "d91c6581d1f04cfd4dd91295931f0df057af5b23c7c8272c76c29ea58afc99a7"
    )
    assert select_document_ids(
        seed=raw["seed"],
        domain_counts=raw["domain_counts"],
    )["government"] == (104,)
    assert raw["case_count"] == 1
    assert raw["cases"] == [
        {
            "case_id": "docbench:104:1",
            "doc_id": 104,
            "domain": "government",
            "pdf_filename": "FBS_H_23FEB2022_PUBLIC-1.pdf",
            "pdf_sha256": (
                "7348d16f0048e7b12260a26dcb7a8cc88cc1fa9e1f62dc796ba2a2c71661e39a"
            ),
            "qa_sha256": (
                "503be0155bca692ec2ab05678ae9793560efffa26793be725627d65d553c75d6"
            ),
            "question_index": 1,
            "question_type": "text-only",
        }
    ]
    assert all(
        key not in raw["cases"][0]
        for key in ("question", "answer", "evidence")
    )

    # Exercise the frozen selection shape with synthetic, isolated source bytes.
    # The checked-in upstream fingerprints above remain immutable.
    case = raw["cases"][0]
    data_root = tmp_path / "data"
    document_root = data_root / str(case["doc_id"])
    document_root.mkdir(parents=True)
    pdf_path = document_root / "synthetic.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\nsynthetic document\n%%EOF\n")
    qa_path = document_root / "104_qa.jsonl"
    records = [
        {
            "question": f"Synthetic question {index}?",
            "answer": f"Synthetic answer {index}",
            "evidence": f"Synthetic evidence {index}",
            "type": "text-only",
        }
        for index in range(2)
    ]
    qa_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    case.update(
        pdf_filename=pdf_path.name,
        pdf_sha256=sha256(pdf_path.read_bytes()).hexdigest(),
        qa_sha256=sha256(qa_path.read_bytes()).hexdigest(),
    )
    synthetic_selection_path = tmp_path / "synthetic-selection.json"
    synthetic_selection_path.write_text(json.dumps(raw), encoding="utf-8")

    loaded = load_selection_manifest(
        synthetic_selection_path,
        data_root=data_root,
        expected_count=1,
    )
    assert loaded.cases[0].case_id == "docbench:104:1"
    assert loaded.cases[0].question == records[1]["question"]
    assert loaded.cases[0].pdf_sha256 == sha256(pdf_path.read_bytes()).hexdigest()
    assert loaded.cases[0].qa_sha256 == sha256(qa_path.read_bytes()).hexdigest()

    # Real source validation must still reject modified bytes, even when all
    # inputs are synthetic. No hash function or loader is mocked.
    for source_path, error in (
        (pdf_path, "PDF sha256 drift"),
        (qa_path, "QA sha256 drift"),
    ):
        original = source_path.read_bytes()
        source_path.write_bytes(original + b"\n")
        with pytest.raises(SelectionValidationError, match=error):
            load_selection_manifest(
                synthetic_selection_path,
                data_root=data_root,
                expected_count=1,
            )
        source_path.write_bytes(original)


def test_bge_m3_live_one_case_config_freezes_strict_local_retrieval() -> None:
    loaded = load_docbench_config(CONFIG_PATH, project_root=PROJECT_ROOT)

    assert loaded.config_sha256 == (
        "28ced34f2ddadbe4fb9b4b10c87bfbc5a05c9b3f4ddf987f46e37e400f5150cc"
    )
    assert loaded.raw["providers"] == {
        "main": {"source": "installation"},
        "vision": {"source": "installation"},
    }
    assert loaded.raw["retrieval"] == {
        "profile": "bge_m3",
        "failure_policy": "strict",
        "required_methods": ["dense", "learned_sparse", "bm25"],
        "encoder": {
            "model_id": "BAAI/bge-m3",
            "revision": "5617a9f61b028005a4858fdac845db406aefb181",
        },
        "reranker": {
            "mode": "bge_v2_m3",
            "model_id": "BAAI/bge-reranker-v2-m3",
            "revision": "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e",
        },
        "local_files_only": True,
        "device": "cpu",
        "use_fp16": False,
    }
    assert loaded.raw["dataset"]["selection"] == (
        "evals/docbench/selections/bge_m3_live_1_v1.json"
    )
    assert loaded.raw["run"] == {
        "runtime_features": "evals/docbench/configs/runtime_closed_world.yaml",
        "output_root": "bench://runs",
        "per_case_timeout_s": 1800,
        "max_workers": 1,
    }
