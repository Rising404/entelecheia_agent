"""Export four shared PDFs and separate question/gold files, without model calls."""

import argparse
import hashlib
import json
from pathlib import Path

from .cases import materialize
from .fixture_documents import chinese_font_path
from .pdf_cases import PDF_CASES


def _save(path: Path, data) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def export(directory: Path) -> dict:
    checkout = Path(__file__).resolve().parents[3]
    if not directory.is_absolute() or directory.resolve().is_relative_to(checkout):
        raise ValueError("output must be an absolute directory outside the checkout")
    directory.mkdir(parents=True, exist_ok=False)
    files = materialize(PDF_CASES[0], directory / "attachments")
    filenames = [path.name for path in files]
    questions = [{"case_id": case.id, "question": case.question, "attachments": filenames}
                 for case in PDF_CASES]
    gold = [{
        "case_id": case.id,
        "target_pdf": case.evidence[0][0],
        "distractor_pdfs": [name for name in filenames if name != case.evidence[0][0]],
        "reference": case.reference,
        "evidence": [{"file": name, "physical_page": page, "quote": fact}
                     for name, page, fact in case.evidence],
        "review_rubric": list(case.review_rubric),
    } for case in PDF_CASES]
    _save(directory / "questions.json", questions)
    _save(directory / "gold.json", gold)
    manifest = {
        "suite": "pdf", "case_count": len(PDF_CASES), "pdf_count": len(files),
        "contract": "Each question receives all four PDFs; exactly one PDF contains all answer evidence.",
        "model_input": "Only attachments/ PDFs and one question; never mount the export root or gold.json.",
        "font_sha256": hashlib.sha256(chinese_font_path().read_bytes()).hexdigest(),
        "files": [{"path": str(path.relative_to(directory)), "bytes": path.stat().st_size,
                   "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                  for path in [*files, directory / "questions.json", directory / "gold.json"]],
    }
    _save(directory / "manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = export(args.output)
    print(json.dumps({"output": str(args.output), "pdf_count": manifest["pdf_count"],
                      "case_count": manifest["case_count"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
