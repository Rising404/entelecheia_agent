"""派生选集只排除冻结基础清单中的题目，不重选题或放宽原始校验。"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

import pytest

from evals.docbench.reproduce_or_run_script.config import (
    BENCH_EVAL_DIR_ENV,
    load_docbench_config,
)
from evals.docbench.reproduce_or_run_script.selection import (
    DEFAULT_SEED,
    DOMAIN_ORDER,
    SelectionValidationError,
    load_selection_manifest,
    select_document_ids,
    write_selection_manifest,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SELECTIONS = PROJECT_ROOT / "evals/docbench/selections"
CONFIGS = PROJECT_ROOT / "evals/docbench/configs"
DERIVED_SCHEMA = "docbench-derived-selection-v1"


def _write_json(path: Path, raw: dict) -> None:
    path.write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")


@pytest.fixture
def derived_fixture(tmp_path):
    data_root = tmp_path / "data"
    counts = {domain: 2 for domain in DOMAIN_ORDER}
    selected = select_document_ids(seed=DEFAULT_SEED, domain_counts=counts)
    for domain in DOMAIN_ORDER:
        for doc_id in selected[domain]:
            root = data_root / str(doc_id)
            root.mkdir(parents=True)
            (root / f"document-{doc_id}.pdf").write_bytes(b"%PDF-1.4\nfixture\n%%EOF")
            (root / f"{doc_id}_qa.jsonl").write_text(
                json.dumps(
                    {
                        "question": f"Question for {doc_id}?",
                        "answer": "fixture answer",
                        "type": "text-only",
                        "evidence": "fixture evidence",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
    base_path = tmp_path / "base.json"
    base = write_selection_manifest(
        base_path, data_root=data_root, domain_counts=counts
    )
    excluded = [case["case_id"] for case in base["cases"][::2]]
    remaining = [case for case in base["cases"] if case["case_id"] not in excluded]
    raw = {
        "schema_version": DERIVED_SCHEMA,
        "selection_algorithm": "ordered-base-case-exclusion-v1",
        "base_manifest": {
            "path": base_path.name,
            "sha256": sha256(base_path.read_bytes()).hexdigest(),
        },
        "excluded_case_ids": excluded,
        "case_count": len(remaining),
        "domain_counts": dict(Counter(case["domain"] for case in remaining)),
        "cases": remaining,
    }
    path = tmp_path / "derived.json"
    _write_json(path, raw)
    return path, data_root, base_path, base, raw


def test_derived_selection_reuses_verified_base_cases_in_their_original_order(
    derived_fixture,
):
    path, data_root, base_path, _base, raw = derived_fixture
    base_loaded = load_selection_manifest(base_path, data_root=data_root)

    loaded = load_selection_manifest(path, data_root=data_root, expected_count=5)

    assert loaded.raw == raw
    assert loaded.sha256 == sha256(path.read_bytes()).hexdigest()
    assert loaded.cases == tuple(
        case
        for case in base_loaded.cases
        if case.case_id not in raw["excluded_case_ids"]
    )
    assert all(case.answer == "fixture answer" for case in loaded.cases)


@pytest.mark.parametrize(
    "mutation",
    [
        "hash",
        "unknown_exclusion",
        "duplicate_exclusion",
        "case_identity",
        "case_count",
        "domain_counts",
        "reorder",
        "extra_field",
        "absolute_path",
        "parent_path",
        "identity_type",
    ],
)
def test_derived_selection_rejects_manifest_drift(derived_fixture, mutation):
    path, data_root, base_path, _base, original = derived_fixture
    raw = deepcopy(original)
    if mutation == "hash":
        raw["base_manifest"]["sha256"] = "0" * 64
    elif mutation == "unknown_exclusion":
        raw["excluded_case_ids"][0] = "docbench:999:0"
    elif mutation == "duplicate_exclusion":
        raw["excluded_case_ids"].append(raw["excluded_case_ids"][0])
    elif mutation == "case_identity":
        raw["cases"][0]["pdf_sha256"] = "0" * 64
    elif mutation == "identity_type":
        raw["cases"][0]["question_index"] = float(raw["cases"][0]["question_index"])
    elif mutation == "case_count":
        raw["case_count"] += 1
    elif mutation == "domain_counts":
        raw["domain_counts"]["academia"] += 1
    elif mutation == "reorder":
        raw["cases"].reverse()
    elif mutation == "extra_field":
        raw["cases"][0]["answer"] = "must not enter a public selection"
    elif mutation == "absolute_path":
        raw["base_manifest"]["path"] = str(base_path)
    elif mutation == "parent_path":
        raw["base_manifest"]["path"] = "../base.json"
    _write_json(path, raw)

    with pytest.raises(SelectionValidationError):
        load_selection_manifest(path, data_root=data_root)


def test_derived_selection_rejects_recursive_base(derived_fixture):
    path, data_root, base_path, _base, raw = derived_fixture
    _write_json(base_path, raw)
    raw["base_manifest"]["sha256"] = sha256(base_path.read_bytes()).hexdigest()
    _write_json(path, raw)

    with pytest.raises(SelectionValidationError, match="base|recursive"):
        load_selection_manifest(path, data_root=data_root)


def test_derived_selection_still_checks_excluded_document_assets(derived_fixture):
    path, data_root, _base_path, base, _raw = derived_fixture
    excluded = base["cases"][0]
    pdf = data_root / str(excluded["doc_id"]) / excluded["pdf_filename"]
    pdf.write_bytes(b"changed even though excluded from this run")

    with pytest.raises(SelectionValidationError, match="PDF sha256 drift"):
        load_selection_manifest(path, data_root=data_root)


def test_derived_selection_does_not_weaken_base_deterministic_selection(
    derived_fixture,
):
    path, data_root, base_path, base, raw = derived_fixture
    base["cases"].reverse()
    _write_json(base_path, base)
    raw["base_manifest"]["sha256"] = sha256(base_path.read_bytes()).hexdigest()
    _write_json(path, raw)

    with pytest.raises(SelectionValidationError, match="deterministic"):
        load_selection_manifest(path, data_root=data_root)


def test_derived_selection_checks_expected_count(derived_fixture):
    path, data_root, _base_path, _base, _raw = derived_fixture
    with pytest.raises(SelectionValidationError, match="expected"):
        load_selection_manifest(path, data_root=data_root, expected_count=6)


def test_remaining_fifteen_is_exactly_the_ordered_twenty_minus_five():
    base_bytes = (SELECTIONS / "stage_20_v1.json").read_bytes()
    base = json.loads(base_bytes)
    five = json.loads((SELECTIONS / "context_regression_5.json").read_text())
    raw = json.loads((SELECTIONS / "stage_remaining_15.json").read_text())
    excluded = [case["case_id"] for case in five["cases"]]

    assert raw["base_manifest"] == {
        "path": "stage_20_v1.json",
        "sha256": sha256(base_bytes).hexdigest(),
    }
    assert raw["excluded_case_ids"] == excluded
    assert raw["cases"] == [
        case for case in base["cases"] if case["case_id"] not in excluded
    ]
    assert raw["case_count"] == len(raw["cases"]) == 15
    assert raw["domain_counts"] == {domain: 3 for domain in DOMAIN_ORDER}
    assert Counter(case["question_type"] for case in raw["cases"]) == {
        "text-only": 6,
        "meta-data": 3,
        "unanswerable": 6,
    }
    assert all(
        not {"question", "answer", "evidence"} & case.keys() for case in raw["cases"]
    )


def test_gpu_remaining_config_changes_only_the_selection_path(tmp_path: Path):
    environment = {BENCH_EVAL_DIR_ENV: str(tmp_path / "bench_eval")}
    five = load_docbench_config(
        CONFIGS / "l1_gpu_context_regression_5.yaml", environment=environment,
    )
    remaining = load_docbench_config(
        CONFIGS / "l1_gpu_stage_remaining_15.yaml", environment=environment,
    )
    expected = deepcopy(five.raw)
    expected["dataset"]["selection"] = (
        "evals/docbench/selections/stage_remaining_15.json"
    )

    assert remaining.raw == expected
    assert remaining.raw["retrieval"]["device"] == "auto"
    assert remaining.raw["retrieval"]["use_fp16"] is False
    assert remaining.raw["run"]["max_workers"] == 1


def test_visual_publication_regression_uses_only_five_original_case_identities(tmp_path: Path):
    base_bytes = (SELECTIONS / "stage_20_v1.json").read_bytes()
    base = json.loads(base_bytes)
    raw = json.loads((SELECTIONS / "vision_publication_regression_5.json").read_text())
    selected = {"docbench:5:1", "docbench:123:0", "docbench:156:4", "docbench:219:3", "docbench:220:2"}
    assert raw["base_manifest"] == {
        "path": "stage_20_v1.json", "sha256": sha256(base_bytes).hexdigest(),
    }
    assert raw["cases"] == [case for case in base["cases"] if case["case_id"] in selected]
    assert raw["excluded_case_ids"] == [
        case["case_id"] for case in base["cases"] if case["case_id"] not in selected
    ]
    assert raw["case_count"] == 5
    environment = {BENCH_EVAL_DIR_ENV: str(tmp_path / "bench_eval")}
    original = load_docbench_config(
        CONFIGS / "l1_gpu_vision_diagnostics_20.yaml", environment=environment,
    )
    regression = load_docbench_config(
        CONFIGS / "l1_gpu_vision_publication_regression_5.yaml", environment=environment,
    )
    expected = deepcopy(original.raw)
    expected["dataset"]["selection"] = "evals/docbench/selections/vision_publication_regression_5.json"
    assert regression.raw == expected
