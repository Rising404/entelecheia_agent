from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

import pytest

import evals.docbench.reproduce_or_run_script.selection as selection_module
from evals.docbench.reproduce_or_run_script.selection import (
    BALANCED_SCHEMA_VERSION,
    BALANCED_SELECTION_ALGORITHM,
    DEFAULT_BALANCED_DOMAIN_COUNTS,
    DEFAULT_BALANCED_DOMAIN_TYPE_COUNTS,
    DEFAULT_BALANCED_NORMALIZED_TYPE_COUNTS,
    DEFAULT_BALANCED_SEED,
    DEFAULT_MAX_QUESTIONS_PER_DOCUMENT,
    DEFAULT_SEED,
    DOMAIN_ORDER,
    NORMALIZED_TYPE_ORDER,
    TYPE_NORMALIZATION_VERSION,
    SelectionValidationError,
    generate_balanced_selection_manifest,
    generate_selection_manifest,
    load_selection_manifest,
    main,
    normalize_question_type,
    select_balanced_questions,
    select_document_ids,
    write_balanced_selection_manifest,
    write_selection_manifest,
)


EXPECTED_FORMAL_DOCUMENTS = {
    "academia": (1, 4, 5, 6, 7, 9, 10, 15, 18, 19, 20, 21, 25, 28, 35, 37, 40, 41, 42, 46),
    "finance": (49, 50, 51, 53, 54, 55, 58, 63, 65, 67, 69, 70, 72, 74, 78, 79, 80, 82, 83, 87),
    "government": (91, 94, 96, 97, 98, 99, 104, 105, 108, 109, 110, 111, 112, 113, 118, 123, 124, 126, 127, 129),
    "laws": (134, 136, 137, 138, 143, 144, 145, 148, 149, 151, 156, 157, 162, 163, 165, 170, 172, 176, 177, 178),
    "news": (182, 190, 191, 192, 195, 199, 202, 203, 208, 209, 210, 212, 213, 214, 215, 216, 218, 219, 220, 227),
}


def _one_per_domain() -> dict[str, int]:
    return {domain: 1 for domain in DOMAIN_ORDER}


def _write_selected_fixture(data_root: Path) -> None:
    selected = select_document_ids(
        seed=DEFAULT_SEED,
        domain_counts=_one_per_domain(),
    )
    for domain in DOMAIN_ORDER:
        doc_id = selected[domain][0]
        document_root = data_root / str(doc_id)
        document_root.mkdir(parents=True)
        (document_root / f"document-{doc_id}.pdf").write_bytes(
            f"%PDF-1.4\nfixture {doc_id}\n%%EOF\n".encode()
        )
        records = (
            {
                "question": f"Question A for {doc_id}?",
                "answer": f"Answer A for {doc_id}",
                "type": "text-only",
                "evidence": f"Evidence A for {doc_id}",
            },
            {
                "question": f"Question B for {doc_id}?",
                "answer": f"Answer B for {doc_id}",
                "type": "meta-data",
                "evidence": "",
            },
        )
        (document_root / f"{doc_id}_qa.jsonl").write_text(
            "".join(json.dumps(record) + "\n" for record in records),
            encoding="utf-8",
        )


def _write_manifest(path: Path, raw: dict[str, object]) -> None:
    path.write_text(json.dumps(raw, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _balanced_domain_counts(per_domain: int = 2) -> dict[str, int]:
    return {domain: per_domain for domain in DOMAIN_ORDER}


def _balanced_type_counts() -> dict[str, int]:
    return {
        "text": 4,
        "multimodal": 2,
        "metadata": 2,
        "unanswerable": 2,
    }


def _write_balanced_fixture(data_root: Path, *, include_pdfs: bool = True) -> None:
    doc_ids = {
        "academia": 1,
        "finance": 49,
        "government": 89,
        "laws": 133,
        "news": 179,
    }
    for domain in DOMAIN_ORDER:
        doc_id = doc_ids[domain]
        document_root = data_root / str(doc_id)
        document_root.mkdir(parents=True)
        if include_pdfs:
            (document_root / f"document-{doc_id}.pdf").write_bytes(
                f"%PDF-1.4\nfixture {doc_id}\n%%EOF\n".encode()
            )
        records = (
            {
                "question": f"Text question for {doc_id}?",
                "answer": f"Text answer for {doc_id}",
                "type": "text-only",
                "evidence": f"Text evidence for {doc_id}",
            },
            {
                "question": f"Visual question for {doc_id}?",
                "answer": f"Visual answer for {doc_id}",
                "type": "multimodal-f",
                "evidence": f"Visual evidence for {doc_id}",
            },
            {
                "question": f"Metadata question for {doc_id}?",
                "answer": f"Metadata answer for {doc_id}",
                "type": "meta-data",
                "evidence": f"Metadata evidence for {doc_id}",
            },
            {
                "question": f"Unanswerable question for {doc_id}?",
                "answer": f"Unanswerable answer for {doc_id}",
                "type": "una-web",
                "evidence": f"Unanswerable evidence for {doc_id}",
            },
            {
                "question": f"Second text question for {doc_id}?",
                "answer": f"Second text answer for {doc_id}",
                "type": "text-only",
                "evidence": f"Second text evidence for {doc_id}",
            },
        )
        (document_root / f"{doc_id}_qa.jsonl").write_text(
            "".join(json.dumps(record) + "\n" for record in records),
            encoding="utf-8",
        )


def test_formal_document_selection_is_frozen_and_excludes_old_doc_zero() -> None:
    selected = select_document_ids(seed=DEFAULT_SEED)

    assert selected == EXPECTED_FORMAL_DOCUMENTS
    assert sum(len(ids) for ids in selected.values()) == 100
    assert len({doc_id for ids in selected.values() for doc_id in ids}) == 100
    assert 0 not in selected["academia"]


def test_small_manifest_contains_no_qa_body_and_loads_private_runtime_cases(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data"
    _write_selected_fixture(data_root)
    manifest_path = tmp_path / "small.json"

    raw = write_selection_manifest(
        manifest_path,
        data_root=data_root,
        domain_counts=_one_per_domain(),
    )
    loaded = load_selection_manifest(
        manifest_path,
        data_root=data_root,
        expected_count=5,
    )

    serialized = manifest_path.read_text(encoding="utf-8")
    assert raw["case_count"] == 5
    assert len(loaded.cases) == 5
    assert loaded.sha256 and len(loaded.sha256) == 64
    assert all(case.pdf_path.is_file() and case.qa_path.is_file() for case in loaded.cases)
    assert all(case.question.startswith("Question ") for case in loaded.cases)
    assert all(case.answer.startswith("Answer ") for case in loaded.cases)
    assert all("question" not in case_manifest for case_manifest in raw["cases"])
    assert all("answer" not in case_manifest for case_manifest in raw["cases"])
    assert all("evidence" not in case_manifest for case_manifest in raw["cases"])
    assert "Question A" not in serialized
    assert "Answer A" not in serialized
    assert "Evidence A" not in serialized


def test_generation_is_byte_stable_for_unchanged_data(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    _write_selected_fixture(data_root)

    first = generate_selection_manifest(
        data_root,
        domain_counts=_one_per_domain(),
    )
    second = generate_selection_manifest(
        data_root,
        domain_counts=_one_per_domain(),
    )

    assert first == second


@pytest.mark.parametrize("raw_type", ("unanswerable", "una-web"))
def test_real_upstream_unanswerable_types_are_preserved_verbatim(
    tmp_path: Path,
    raw_type: str,
) -> None:
    data_root = tmp_path / "data"
    _write_selected_fixture(data_root)
    selected = select_document_ids(
        seed=DEFAULT_SEED,
        domain_counts=_one_per_domain(),
    )
    doc_id = selected["academia"][0]
    qa_path = data_root / str(doc_id) / f"{doc_id}_qa.jsonl"
    records = [json.loads(line) for line in qa_path.read_text().splitlines()]
    for record in records:
        record["type"] = raw_type
    qa_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    manifest_path = tmp_path / "small.json"

    raw = write_selection_manifest(
        manifest_path,
        data_root=data_root,
        domain_counts=_one_per_domain(),
    )
    loaded = load_selection_manifest(manifest_path, data_root=data_root)

    raw_case = next(case for case in raw["cases"] if case["doc_id"] == doc_id)
    loaded_case = next(case for case in loaded.cases if case.doc_id == doc_id)
    assert raw_case["question_type"] == raw_type
    assert loaded_case.question_type == raw_type


@pytest.mark.parametrize("drift", ("pdf_hash", "qa_hash", "question_type"))
def test_loader_rejects_manifest_or_local_data_drift(
    tmp_path: Path,
    drift: str,
) -> None:
    data_root = tmp_path / "data"
    _write_selected_fixture(data_root)
    manifest_path = tmp_path / "small.json"
    raw = write_selection_manifest(
        manifest_path,
        data_root=data_root,
        domain_counts=_one_per_domain(),
    )
    first_case = raw["cases"][0]
    if drift == "pdf_hash":
        first_case["pdf_sha256"] = "0" * 64
    elif drift == "qa_hash":
        first_case["qa_sha256"] = "0" * 64
    else:
        first_case["question_type"] = "una"
    _write_manifest(manifest_path, raw)

    with pytest.raises(SelectionValidationError, match="drift"):
        load_selection_manifest(manifest_path, data_root=data_root)


def test_loader_rejects_non_deterministic_document_replacement(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    _write_selected_fixture(data_root)
    manifest_path = tmp_path / "small.json"
    raw = write_selection_manifest(
        manifest_path,
        data_root=data_root,
        domain_counts=_one_per_domain(),
    )
    raw["cases"][0]["doc_id"] += 1
    _write_manifest(manifest_path, raw)

    with pytest.raises(SelectionValidationError):
        load_selection_manifest(manifest_path, data_root=data_root)


def test_loader_enforces_expected_count(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    _write_selected_fixture(data_root)
    manifest_path = tmp_path / "small.json"
    write_selection_manifest(
        manifest_path,
        data_root=data_root,
        domain_counts=_one_per_domain(),
    )

    with pytest.raises(SelectionValidationError, match="expected 100 cases"):
        load_selection_manifest(
            manifest_path,
            data_root=data_root,
            expected_count=100,
        )


def test_balanced_target_defaults_are_explicit_and_internally_consistent() -> None:
    assert DEFAULT_BALANCED_SEED == "docbench-formal-balanced-125-v1"
    assert DEFAULT_BALANCED_DOMAIN_COUNTS == {
        domain: 25 for domain in DOMAIN_ORDER
    }
    assert DEFAULT_BALANCED_NORMALIZED_TYPE_COUNTS == {
        "text": 50,
        "multimodal": 25,
        "metadata": 25,
        "unanswerable": 25,
    }
    assert DEFAULT_BALANCED_DOMAIN_TYPE_COUNTS == {
        "academia": {
            "text": 5,
            "multimodal": 12,
            "metadata": 3,
            "unanswerable": 5,
        },
        "finance": {
            "text": 7,
            "multimodal": 12,
            "metadata": 3,
            "unanswerable": 3,
        },
        "government": {
            "text": 13,
            "multimodal": 0,
            "metadata": 7,
            "unanswerable": 5,
        },
        "laws": {
            "text": 12,
            "multimodal": 0,
            "metadata": 7,
            "unanswerable": 6,
        },
        "news": {
            "text": 13,
            "multimodal": 1,
            "metadata": 5,
            "unanswerable": 6,
        },
    }
    assert DEFAULT_MAX_QUESTIONS_PER_DOCUMENT == 1
    assert sum(DEFAULT_BALANCED_DOMAIN_COUNTS.values()) == 125
    assert sum(DEFAULT_BALANCED_NORMALIZED_TYPE_COUNTS.values()) == 125
    assert {
        domain: sum(row.values())
        for domain, row in DEFAULT_BALANCED_DOMAIN_TYPE_COUNTS.items()
    } == DEFAULT_BALANCED_DOMAIN_COUNTS
    assert {
        question_type: sum(
            DEFAULT_BALANCED_DOMAIN_TYPE_COUNTS[domain][question_type]
            for domain in DOMAIN_ORDER
        )
        for question_type in NORMALIZED_TYPE_ORDER
    } == DEFAULT_BALANCED_NORMALIZED_TYPE_COUNTS


@pytest.mark.parametrize(
    ("raw_type", "normalized"),
    (
        ("text-only", "text"),
        ("multimodal-f", "multimodal"),
        ("multimodal-t", "multimodal"),
        ("multimodal", "multimodal"),
        ("meta-data", "metadata"),
        ("una", "unanswerable"),
        ("una-web", "unanswerable"),
        ("unanswerable", "unanswerable"),
    ),
)
def test_balanced_type_normalization_is_frozen(
    raw_type: str,
    normalized: str,
) -> None:
    assert normalize_question_type(raw_type) == normalized


def test_balanced_selection_is_question_level_deterministic_and_round_trips(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data"
    _write_balanced_fixture(data_root)
    manifest_path = tmp_path / "balanced.json"
    kwargs = {
        "seed": "balanced-fixture-v1",
        "domain_counts": _balanced_domain_counts(),
        "normalized_type_counts": _balanced_type_counts(),
        "max_questions_per_document": 2,
    }

    first = select_balanced_questions(data_root, **kwargs)
    second = select_balanced_questions(data_root, **kwargs)
    raw = write_balanced_selection_manifest(
        manifest_path,
        data_root=data_root,
        **kwargs,
    )
    loaded = load_selection_manifest(
        manifest_path,
        data_root=data_root,
        expected_count=10,
    )

    assert first == second
    assert len(first) == 10
    assert len({(case.doc_id, case.question_index) for case in first}) == 10
    assert Counter(case.domain for case in first) == _balanced_domain_counts()
    assert Counter(case.normalized_question_type for case in first) == (
        _balanced_type_counts()
    )
    assert max(Counter(case.doc_id for case in first).values()) == 2
    assert raw["schema_version"] == BALANCED_SCHEMA_VERSION
    assert raw["selection_algorithm"] == BALANCED_SELECTION_ALGORITHM
    assert raw["type_normalization_version"] == TYPE_NORMALIZATION_VERSION
    assert len(raw["candidate_inventory_sha256"]) == 64
    assert raw["max_questions_per_document"] == 2
    assert raw["domain_counts"] == _balanced_domain_counts()
    assert raw["normalized_type_counts"] == _balanced_type_counts()
    assert set(raw["domain_type_counts"]) == set(DOMAIN_ORDER)
    assert all(
        set(row) == set(NORMALIZED_TYPE_ORDER)
        for row in raw["domain_type_counts"].values()
    )
    assert len(loaded.cases) == 10
    assert [case.case_id for case in loaded.cases] == [
        case.case_id for case in first
    ]
    serialized = manifest_path.read_text(encoding="utf-8")
    assert "Text question" not in serialized
    assert "Text answer" not in serialized
    assert "Text evidence" not in serialized


def test_balanced_selection_strictly_executes_explicit_domain_type_counts(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data"
    _write_balanced_fixture(data_root)
    domain_type_counts = {
        "academia": {
            "text": 0,
            "multimodal": 1,
            "metadata": 0,
            "unanswerable": 1,
        },
        "finance": {
            "text": 0,
            "multimodal": 1,
            "metadata": 0,
            "unanswerable": 1,
        },
        "government": {
            "text": 1,
            "multimodal": 0,
            "metadata": 1,
            "unanswerable": 0,
        },
        "laws": {
            "text": 1,
            "multimodal": 0,
            "metadata": 1,
            "unanswerable": 0,
        },
        "news": {
            "text": 2,
            "multimodal": 0,
            "metadata": 0,
            "unanswerable": 0,
        },
    }
    kwargs = {
        "seed": "balanced-explicit-matrix-v1",
        "domain_counts": _balanced_domain_counts(),
        "normalized_type_counts": _balanced_type_counts(),
        "domain_type_counts": domain_type_counts,
        "max_questions_per_document": 2,
    }

    selected = select_balanced_questions(data_root, **kwargs)
    manifest = generate_balanced_selection_manifest(data_root, **kwargs)

    actual = {
        domain: {
            question_type: sum(
                case.domain == domain
                and case.normalized_question_type == question_type
                for case in selected
            )
            for question_type in NORMALIZED_TYPE_ORDER
        }
        for domain in DOMAIN_ORDER
    }
    assert actual == domain_type_counts
    assert manifest["domain_type_counts"] == domain_type_counts


def test_balanced_selection_rejects_explicit_matrix_marginal_mismatch(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data"
    _write_balanced_fixture(data_root)
    domain_type_counts = {
        domain: {
            question_type: 0
            for question_type in NORMALIZED_TYPE_ORDER
        }
        for domain in DOMAIN_ORDER
    }

    with pytest.raises(SelectionValidationError, match="does not match"):
        select_balanced_questions(
            data_root,
            seed="balanced-bad-matrix-v1",
            domain_counts=_balanced_domain_counts(),
            normalized_type_counts=_balanced_type_counts(),
            domain_type_counts=domain_type_counts,
            max_questions_per_document=2,
        )


def test_balanced_cell_flow_reroutes_around_document_capacity(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data"
    records_by_document = {
        1: (
            {
                "question": "Flexible text question?",
                "answer": "Text",
                "type": "text-only",
                "evidence": "Text evidence",
            },
            {
                "question": "Only metadata question?",
                "answer": "Metadata",
                "type": "meta-data",
                "evidence": "Metadata evidence",
            },
        ),
        2: (
            {
                "question": "Only other text question?",
                "answer": "Other text",
                "type": "text-only",
                "evidence": "Other text evidence",
            },
        ),
    }
    for doc_id, records in records_by_document.items():
        document_root = data_root / str(doc_id)
        document_root.mkdir(parents=True)
        (document_root / f"{doc_id}_qa.jsonl").write_text(
            "".join(json.dumps(record) + "\n" for record in records),
            encoding="utf-8",
        )
    domain_counts = {domain: 0 for domain in DOMAIN_ORDER}
    domain_counts["academia"] = 2
    type_counts = {question_type: 0 for question_type in NORMALIZED_TYPE_ORDER}
    type_counts["text"] = 1
    type_counts["metadata"] = 1
    domain_type_counts = {
        domain: {
            question_type: 0
            for question_type in NORMALIZED_TYPE_ORDER
        }
        for domain in DOMAIN_ORDER
    }
    domain_type_counts["academia"]["text"] = 1
    domain_type_counts["academia"]["metadata"] = 1

    selected = select_balanced_questions(
        data_root,
        seed="residual-1",
        domain_counts=domain_counts,
        normalized_type_counts=type_counts,
        domain_type_counts=domain_type_counts,
        max_questions_per_document=1,
    )

    assert {(case.doc_id, case.normalized_question_type) for case in selected} == {
        (1, "metadata"),
        (2, "text"),
    }


def test_balanced_candidate_selection_does_not_require_pdf_until_freeze(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data"
    _write_balanced_fixture(data_root, include_pdfs=False)
    kwargs = {
        "seed": "qa-only-candidates-v1",
        "domain_counts": _balanced_domain_counts(per_domain=1),
        "normalized_type_counts": {
            "text": 5,
            "multimodal": 0,
            "metadata": 0,
            "unanswerable": 0,
        },
        "max_questions_per_document": 1,
    }

    selected = select_balanced_questions(data_root, **kwargs)

    assert len(selected) == 5
    with pytest.raises(SelectionValidationError, match="exactly one PDF"):
        generate_balanced_selection_manifest(data_root, **kwargs)


def test_balanced_selection_rejects_infeasible_document_caps(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    _write_balanced_fixture(data_root)

    with pytest.raises(SelectionValidationError, match="infeasible"):
        select_balanced_questions(
            data_root,
            seed="infeasible-v1",
            domain_counts=_balanced_domain_counts(),
            normalized_type_counts=_balanced_type_counts(),
            max_questions_per_document=1,
        )


def test_balanced_loader_rejects_candidate_inventory_drift_before_selection(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data"
    _write_balanced_fixture(data_root)
    manifest_path = tmp_path / "balanced.json"
    write_balanced_selection_manifest(
        manifest_path,
        data_root=data_root,
        seed="balanced-inventory-v1",
        domain_counts=_balanced_domain_counts(),
        normalized_type_counts=_balanced_type_counts(),
        max_questions_per_document=2,
    )
    new_document_root = data_root / "2"
    new_document_root.mkdir()
    (new_document_root / "2_qa.jsonl").write_text(
        json.dumps(
            {
                "question": "New eligible question?",
                "answer": "New answer",
                "type": "text-only",
                "evidence": "New evidence",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(SelectionValidationError, match="candidate inventory drift"):
        load_selection_manifest(manifest_path, data_root=data_root)


@pytest.mark.parametrize(
    "tamper",
    (
        "algorithm",
        "normalization_version",
        "candidate_inventory",
        "normalized_case_type",
        "domain_type_counts",
        "document_cap",
        "case_order",
    ),
)
def test_balanced_loader_strictly_recomputes_manifest(
    tmp_path: Path,
    tamper: str,
) -> None:
    data_root = tmp_path / "data"
    _write_balanced_fixture(data_root)
    manifest_path = tmp_path / "balanced.json"
    raw = write_balanced_selection_manifest(
        manifest_path,
        data_root=data_root,
        seed="balanced-recompute-v1",
        domain_counts=_balanced_domain_counts(),
        normalized_type_counts=_balanced_type_counts(),
        max_questions_per_document=2,
    )
    if tamper == "algorithm":
        raw["selection_algorithm"] = "future-algorithm"
    elif tamper == "normalization_version":
        raw["type_normalization_version"] = "future-normalization"
    elif tamper == "candidate_inventory":
        raw["candidate_inventory_sha256"] = "0" * 64
    elif tamper == "normalized_case_type":
        current = raw["cases"][0]["normalized_question_type"]
        raw["cases"][0]["normalized_question_type"] = next(
            question_type
            for question_type in NORMALIZED_TYPE_ORDER
            if question_type != current
        )
    elif tamper == "domain_type_counts":
        matrix = raw["domain_type_counts"]
        cycle = next(
            (first_domain, second_domain, first_type, second_type)
            for first_domain in DOMAIN_ORDER
            for second_domain in DOMAIN_ORDER
            if first_domain != second_domain
            for first_type in NORMALIZED_TYPE_ORDER
            for second_type in NORMALIZED_TYPE_ORDER
            if first_type != second_type
            and matrix[first_domain][first_type] > 0
            and matrix[second_domain][second_type] > 0
        )
        first_domain, second_domain, first_type, second_type = cycle
        matrix[first_domain][first_type] -= 1
        matrix[first_domain][second_type] += 1
        matrix[second_domain][first_type] += 1
        matrix[second_domain][second_type] -= 1
    elif tamper == "document_cap":
        raw["max_questions_per_document"] = 1
    else:
        raw["cases"][0], raw["cases"][1] = raw["cases"][1], raw["cases"][0]
    _write_manifest(manifest_path, raw)

    with pytest.raises(SelectionValidationError):
        load_selection_manifest(manifest_path, data_root=data_root)


@pytest.mark.parametrize(
    ("seed_args", "expected_seed"),
    (
        ((), DEFAULT_BALANCED_SEED),
        (("--seed", "balanced-cli-override"), "balanced-cli-override"),
    ),
)
def test_balanced_cli_uses_frozen_defaults_and_strictly_reloads_125_cases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    seed_args: tuple[str, ...],
    expected_seed: str,
) -> None:
    calls: dict[str, object] = {}

    def fake_write(output_path: Path, **kwargs: object) -> dict[str, object]:
        calls["write_output"] = output_path
        calls["write_kwargs"] = kwargs
        return {}

    def fake_load(path: Path, **kwargs: object) -> object:
        calls["load_path"] = path
        calls["load_kwargs"] = kwargs
        return object()

    monkeypatch.setattr(
        selection_module,
        "write_balanced_selection_manifest",
        fake_write,
    )
    monkeypatch.setattr(selection_module, "load_selection_manifest", fake_load)
    data_root = tmp_path / "data"
    output_path = tmp_path / "balanced.json"

    result = main(
        (
            "--balanced",
            "--data-root",
            str(data_root),
            "--output",
            str(output_path),
            *seed_args,
        )
    )

    assert result == 0
    assert calls["write_output"] == output_path
    assert calls["write_kwargs"] == {
        "data_root": data_root,
        "seed": expected_seed,
        "domain_counts": DEFAULT_BALANCED_DOMAIN_COUNTS,
        "normalized_type_counts": DEFAULT_BALANCED_NORMALIZED_TYPE_COUNTS,
        "domain_type_counts": DEFAULT_BALANCED_DOMAIN_TYPE_COUNTS,
        "max_questions_per_document": DEFAULT_MAX_QUESTIONS_PER_DOCUMENT,
    }
    assert calls["load_path"] == output_path
    assert calls["load_kwargs"] == {
        "data_root": data_root,
        "expected_count": 125,
    }


@pytest.mark.parametrize(
    ("seed_args", "expected_seed"),
    (
        ((), DEFAULT_SEED),
        (("--seed", "v1-cli-override"), "v1-cli-override"),
    ),
)
def test_cli_without_balanced_flag_preserves_v1_defaults_and_accepts_seed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    seed_args: tuple[str, ...],
    expected_seed: str,
) -> None:
    calls: dict[str, object] = {}

    def fake_write(output_path: Path, **kwargs: object) -> dict[str, object]:
        calls["write_output"] = output_path
        calls["write_kwargs"] = kwargs
        return {}

    def fake_load(path: Path, **kwargs: object) -> object:
        calls["load_path"] = path
        calls["load_kwargs"] = kwargs
        return object()

    monkeypatch.setattr(selection_module, "write_selection_manifest", fake_write)
    monkeypatch.setattr(selection_module, "load_selection_manifest", fake_load)
    data_root = tmp_path / "data"
    output_path = tmp_path / "v1.json"

    result = main(
        (
            "--data-root",
            str(data_root),
            "--output",
            str(output_path),
            *seed_args,
        )
    )

    assert result == 0
    assert calls["write_output"] == output_path
    assert calls["write_kwargs"] == {
        "data_root": data_root,
        "seed": expected_seed,
        "domain_counts": {domain: 20 for domain in DOMAIN_ORDER},
    }
    assert calls["load_path"] == output_path
    assert calls["load_kwargs"] == {
        "data_root": data_root,
        "expected_count": 100,
    }
