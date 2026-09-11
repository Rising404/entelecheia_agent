from __future__ import annotations

from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

from evals.docbench.reproduce_or_run_script.config import load_docbench_config


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SELECTION_PATH = PROJECT_ROOT / "evals/docbench/selections/stage_20_v1.json"
CONFIG_PATH = PROJECT_ROOT / "evals/docbench/configs/l1_stage_20.yaml"
PUBLIC_SUMMARY_PATH = (
    PROJECT_ROOT / "evals/docbench/results/live_bge_m3_l1_20.summary.json"
)


def test_stage_twenty_selection_is_frozen_balanced_by_domain_and_content_free() -> None:
    raw_bytes = SELECTION_PATH.read_bytes()
    raw = json.loads(raw_bytes)

    assert sha256(raw_bytes).hexdigest() == (
        "c0836b7f0136443ce4ff98576aa0737925746e643fba671476325285dac6938b"
    )
    assert raw["case_count"] == 20
    assert raw["domain_counts"] == {
        "academia": 4,
        "finance": 4,
        "government": 4,
        "laws": 4,
        "news": 4,
    }
    assert len(raw["cases"]) == len({case["doc_id"] for case in raw["cases"]}) == 20
    assert Counter(case["question_type"] for case in raw["cases"]) == {
        "text-only": 9,
        "unanswerable": 6,
        "meta-data": 4,
        "multimodal-f": 1,
    }
    assert all(
        key not in case
        for case in raw["cases"]
        for key in ("question", "answer", "evidence")
    )


def test_stage_twenty_config_freezes_strict_local_bge_m3_and_serial_cases() -> None:
    loaded = load_docbench_config(CONFIG_PATH, project_root=PROJECT_ROOT)

    assert loaded.config_sha256 == (
        "9ba314ae087cf78127acc16bc431e6ec658d3a2165cdff98ab22ce7cbdbf4b33"
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
        "evals/docbench/selections/stage_20_v1.json"
    )
    assert loaded.raw["run"] == {
        "runtime_features": "evals/docbench/configs/runtime_closed_world.yaml",
        "output_root": "bench://runs",
        "per_case_timeout_s": 1800,
        "max_workers": 1,
    }


def test_stage_twenty_public_summary_is_aggregate_only_and_internally_consistent() -> None:
    summary = json.loads(PUBLIC_SUMMARY_PATH.read_text())
    cases = summary["cases"]

    assert summary["scope"]["case_count"] == len(cases) == 20
    assert summary["generation"]["execution_ok_case_count"] == sum(
        case["execution_ok"] for case in cases
    ) == 18
    assert summary["scoring"]["correct_count"] == sum(
        case["score"] for case in cases
    ) == 11
    assert summary["retrieval"]["active_generation_ready_case_count"] == sum(
        case["retrieval_status"] == "ready" for case in cases
    ) == 7
    assert summary["retrieval"]["active_bge_query_used_case_count"] == sum(
        case["active_bge_query_used"] for case in cases
    ) == 5
    assert summary["retrieval"]["active_generation_unavailable_case_count"] == 13
    assert summary["scoring"]["delivered_correct_count"] == sum(
        case["score"] for case in cases if case["execution_ok"]
    ) == 10
    assert all(
        key not in case
        for case in cases
        for key in (
            "question",
            "reference_answer",
            "reference_text",
            "system_answer",
            "judge_response_raw",
        )
    )
