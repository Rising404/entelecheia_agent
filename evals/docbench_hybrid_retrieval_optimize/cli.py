"""Frozen DocBench retrieval datasets and offline retrieval evaluation entry point."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from . import retrieval_dataset, retrieval_eval, retrieval_hybrid


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m evals.docbench_hybrid_retrieval_optimize",
        description="Build, annotate and evaluate frozen DocBench retrieval datasets.",
        epilog=(
            "Run from the repository root. Retrieval artifacts may live outside the repository "
            "or in this package's dataset, indexes and results directories. "
            "evaluate-retrieval defaults to the SQLite BM25 baseline; "
            "hybrid uses local production models and requires --allow-live."
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True)
    build_retrieval = commands.add_parser("build-retrieval", help="freeze a corpus and pending labels from completed runs")
    build_retrieval.add_argument("--selection", required=True, type=Path)
    build_retrieval.add_argument("--data-root", required=True, type=Path)
    build_retrieval.add_argument("--source-run", required=True, type=Path, action="append")
    build_retrieval.add_argument("--output", required=True, type=Path)
    build_retrieval.add_argument("--visual-doc-id", type=int, action="append",
                                 help="explicit visual corpus document scope; default is every selected document")
    build_retrieval.add_argument("--curation", type=Path,
                                 help="reviewed source transcriptions and audited reference corrections")
    annotate = commands.add_parser("annotate-retrieval", help="apply explicitly reviewed retrieval evidence labels")
    annotate.add_argument("--dataset", required=True, type=Path)
    annotate.add_argument("--annotations", required=True, type=Path)
    evaluate = commands.add_parser("evaluate-retrieval", help="evaluate BM25 or production hybrid on a frozen dataset")
    evaluate.add_argument("--dataset", required=True, type=Path)
    evaluate.add_argument("--output", required=True, type=Path)
    evaluate.add_argument("--backend", choices=("bm25", "hybrid"), default="bm25")
    evaluate.add_argument("--index", type=Path, help="hybrid index directory; validated and reused if present")
    evaluate.add_argument("--encoding-cache", type=Path,
                          help="verified exact-content, same-encoder representations for a new hybrid index")
    evaluate.add_argument("--device", choices=("cpu", "mps", "cuda:0"))
    evaluate.add_argument("--batch-size", type=int)
    evaluate.add_argument("--allow-live", action="store_true", help="explicitly enable local encoder/reranker inference")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "build-retrieval":
            report = retrieval_dataset.build_retrieval_dataset(
                selection_path=args.selection, data_root=args.data_root,
                run_dirs=args.source_run, output_dir=args.output,
                visual_doc_ids=args.visual_doc_id, curation_path=args.curation,
            )
            retrieval_dataset.write_retrieval_exports(args.output / "dataset.sqlite", args.output / "exports")
            print(json.dumps({key: value for key, value in report.items() if key != "sources"}, ensure_ascii=False, indent=2))
            return 0
        if args.command == "annotate-retrieval":
            report = retrieval_dataset.apply_annotations(dataset_path=args.dataset, annotation_path=args.annotations)
            try:
                retrieval_dataset.write_review_packets(args.dataset, args.dataset.parent / "review")
                retrieval_dataset.write_retrieval_exports(args.dataset, args.dataset.parent / "exports")
            except OSError as error:
                raise retrieval_dataset.RetrievalDatasetError(
                    "Annotations committed; derived views/exports refresh failed. Do not reapply the batch. "
                    "Regenerate with retrieval_dataset.write_review_packets and write_retrieval_exports."
                ) from error
            print(json.dumps(report.get("annotation_counts"), indent=2))
            return 0
        if args.command == "evaluate-retrieval":
            if args.backend == "hybrid":
                if args.index is None or not args.allow_live:
                    raise retrieval_hybrid.HybridRetrievalError("hybrid requires --index and --allow-live")
                report = retrieval_hybrid.run_hybrid_retrieval_evaluation(
                    args.dataset, args.index, args.output,
                    device=args.device or "cpu", batch_size=args.batch_size if args.batch_size is not None else 8,
                    allow_live=args.allow_live,
                    encoding_cache=args.encoding_cache,
                    progress=lambda event: print(json.dumps(event, ensure_ascii=False), flush=True),
                )
            else:
                if (args.index is not None or args.encoding_cache is not None or args.device is not None
                        or args.batch_size is not None or args.allow_live):
                    raise retrieval_eval.RetrievalEvaluationError("model/index options require --backend hybrid")
                report = retrieval_eval.run_retrieval_evaluation(args.dataset, args.output)
            print(json.dumps(report["coverage"], ensure_ascii=False, indent=2))
            return 0
    except (
        retrieval_dataset.RetrievalDatasetError,
        retrieval_eval.RetrievalEvaluationError,
        retrieval_hybrid.HybridRetrievalError,
        OSError,
    ) as exc:
        print(f"DocBench retrieval error: {exc}")
        return 2
    raise AssertionError(f"Unhandled retrieval command: {args.command}")


__all__ = ["main"]
