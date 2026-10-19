"""Bind original DocBench questions to four complete, hash-verified PDFs."""

import argparse
import hashlib
import json
from pathlib import Path

from pypdf import PdfReader

from evals.docbench.reproduce_or_run_script.selection import load_selection_manifest

from .cases import Case, SourcePDF, materialize


HERE = Path(__file__).resolve().parent
SELECTION = HERE / "docbench_selection.json"
LOCAL_DATASET = HERE / "local" / "docbench"
SCHEMA = "docbench-multi-document-selection-v1"


def load_cases(data_root: Path, *, selection: Path = SELECTION) -> tuple[Case, ...]:
    """Reuse the canonical DocBench source validation; never rewrite QA or PDFs."""
    data_root = data_root.resolve()
    if data_root.is_relative_to(HERE.parents[1]):
        raise ValueError("DocBench source data must stay outside the checkout")
    spec = json.loads(selection.read_text(encoding="utf-8"))
    if set(spec) != {"schema_version", "source_selection", "source_selection_sha256", "cases"} or spec["schema_version"] != SCHEMA:
        raise ValueError("unsupported multi-document selection")
    base = (selection.parent / spec["source_selection"]).resolve()
    if hashlib.sha256(base.read_bytes()).hexdigest() != spec["source_selection_sha256"]:
        raise ValueError("source selection hash drift")
    verified = load_selection_manifest(base, data_root=data_root)
    questions = {case.case_id: case for case in verified.cases}
    documents = {case.doc_id: case for case in verified.cases}
    page_counts = {}
    cases = []
    seen = set()
    for row in spec["cases"]:
        if set(row) != {"case_id", "attachment_doc_ids", "evidence_pages"}:
            raise ValueError("invalid multi-document case fields")
        case_id = row["case_id"]
        if case_id not in questions or case_id in seen:
            raise ValueError("unknown or duplicate DocBench case")
        seen.add(case_id)
        question = questions[case_id]
        ids = row["attachment_doc_ids"]
        if (not isinstance(ids, list) or len(ids) != 4
                or any(type(doc_id) is not int or doc_id not in documents for doc_id in ids)
                or len(set(ids)) != 4 or question.doc_id not in ids):
            raise ValueError("each case needs the target and three distinct verified PDFs")
        if question.question_type not in {"text-only", "multimodal-t", "multimodal-f", "multimodal"}:
            raise ValueError("this suite requires an answerable document question")
        sources = []
        for index, doc_id in enumerate(ids, 1):
            document = documents[doc_id]
            if doc_id not in page_counts:
                page_counts[doc_id] = len(PdfReader(document.pdf_path).pages)
            sources.append(SourcePDF(
                f"document_{index:02d}.pdf", document.pdf_path,
                document.pdf_sha256, page_counts[doc_id],
            ))
        if len({source.sha256 for source in sources}) != 4:
            raise ValueError("attachment PDFs must have distinct content hashes")
        pages = row["evidence_pages"]
        if (not isinstance(pages, list) or not pages
                or any(type(page) is not int or not 1 <= page <= page_counts[question.doc_id] for page in pages)
                or pages != sorted(set(pages))):
            raise ValueError("evidence pages must be distinct valid physical pages")
        target = sources[ids.index(question.doc_id)].name
        cases.append(Case(
            id=case_id.replace(":", "-"), scenario=f"DocBench {question.domain} / {question.question_type}",
            question=question.question, documents=(), reference=question.answer, expected_facts=(),
            needs_vision=question.question_type.startswith("multimodal"), suite="docbench",
            # Upstream evidence may span these independently reviewed physical pages.
            evidence=tuple((target, page, question.evidence) for page in pages),
            review_rubric=(
                f"唯一目标为 {target}；核对物理页 {', '.join(map(str, pages))}，不得串用干扰文档。",
                "参考答案与 evidence 保持 DocBench 原文；逐项核实最终答复和引用，不将 completed 当作正确。",
            ),
            source_pdfs=tuple(sources),
        ))
    if not cases:
        raise ValueError("empty multi-document selection")
    return tuple(cases)


def export(data_root: Path, *, output: Path = LOCAL_DATASET, selection: Path = SELECTION) -> dict:
    """Write a private, browsable question/attachment bundle with separate gold."""
    output = output.resolve()
    if output.is_relative_to(HERE.parents[1]) and not output.is_relative_to(HERE / "local"):
        raise ValueError("checkout exports must stay in ignored multi_doc/local/")
    cases = load_cases(data_root, selection=selection)
    output.mkdir(parents=True, exist_ok=False)
    questions, gold = [], []
    for case in cases:
        files = materialize(case, output / "attachments" / case.id)
        questions.append({"case_id": case.id, "question": case.question,
                          "attachments": [str(path.relative_to(output)) for path in files]})
        gold.append({"case_id": case.id, "reference": case.reference,
                     "target_pdf": case.evidence[0][0], "upstream_evidence": case.evidence[0][2],
                     "physical_pages": [page for _, page, _ in case.evidence],
                     "review_rubric": list(case.review_rubric)})
    for name, payload in (("questions.json", questions), ("gold.json", gold)):
        (output / name).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "suite": "docbench", "case_count": len(cases), "official_comparable": False,
        "selection_sha256": hashlib.sha256(selection.read_bytes()).hexdigest(),
        "model_input": "Only four PDFs and the original question; no gold, source QA or target hint.",
        "files": [{"path": str(path.relative_to(output)), "bytes": path.stat().st_size,
                   "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                  for path in sorted(output.rglob("*")) if path.is_file()],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=LOCAL_DATASET)
    args = parser.parse_args()
    result = export(args.data_root, output=args.output)
    print(json.dumps({"output": str(args.output.resolve()), "case_count": result["case_count"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
