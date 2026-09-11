"""DocBench 数据准备、配置校验、运行、恢复与评分的统一入口。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from evals.docbench.reproduce_or_run_script import (
    download_selection,
    readiness,
    runner,
    selection,
)
from evals.docbench.reproduce_or_run_script.config import (
    PROJECT_ROOT,
    DocBenchConfigError,
    LoadedDocBenchConfig,
    load_docbench_config,
)
from evals.docbench.reproduce_or_run_script.scorer import DocBenchScoringError
from evals.docbench.reproduce_or_run_script.download_selection import (
    DocBenchDownloadError,
)
from evals.docbench.reproduce_or_run_script.selection import SelectionValidationError


DOCBENCH_ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = DOCBENCH_ROOT / "configs"


def load_checked_in_configs() -> tuple[LoadedDocBenchConfig, ...]:
    """严格加载按文件名排序的全部现行 L1 配置。"""

    paths = tuple(sorted(CONFIGS_DIR.glob("l1_*.yaml")))
    if not paths:
        raise DocBenchConfigError(
            f"no checked-in DocBench L1 configs found in {CONFIGS_DIR}"
        )
    return tuple(
        load_docbench_config(path, project_root=PROJECT_ROOT) for path in paths
    )


def _parser(
    *,
    prog: str = "python -m evals.docbench.reproduce_or_run_script",
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Validate and reproduce checked-in DocBench L1 evaluations.",
        epilog=(
            "Run from the repository root. Set PERSONAGRAPH_BENCH_EVAL_DIR "
            "to an absolute directory outside the checkout before starting Python; "
            "source data and private runs live in its docbench subdirectory."
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_data = commands.add_parser(
        "prepare-data",
        help="download and validate DocBench source files",
    )
    download_selection._add_cli_arguments(prepare_data)
    build_selection = commands.add_parser(
        "build-selection",
        help="build and strictly reload a deterministic selection",
    )
    selection._add_cli_arguments(build_selection, include_legacy_options=False)
    commands.add_parser("validate", help="validate all checked-in L1 configs")
    commands.add_parser(
        "readiness",
        help="verify local L1 data, providers and retrieval assets without network or writes",
    )
    commands.add_parser("list", help="list all valid checked-in L1 configs")

    run = commands.add_parser("run", help="run a configured L1 evaluation")
    run.add_argument("--config", required=True, type=Path)
    run.add_argument("--run-id")
    run.add_argument("--resume", action="store_true")
    run.add_argument("--require-clean", action="store_true")
    run.add_argument("--allow-live", action="store_true")

    retry = commands.add_parser("retry-failed", help="retry failed cases in one run")
    retry.add_argument("--config", required=True, type=Path)
    retry.add_argument("--run", required=True, type=Path)
    retry.add_argument("--max-workers", type=int, default=1)
    retry.add_argument("--error-code", dest="error_codes", action="append")
    retry.add_argument("--allow-live", action="store_true")

    score = commands.add_parser("score", help="score a completed L1 evaluation")
    score.add_argument("--config", required=True, type=Path)
    score.add_argument("--run", required=True, type=Path)
    score.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    score.add_argument("--allow-live", action="store_true")
    return parser


def _run_action(args: argparse.Namespace) -> int:
    if args.command == "run":
        outcome = runner.run_from_config(
            args.config,
            run_id=args.run_id,
            resume=args.resume,
            require_clean=args.require_clean,
            allow_live=args.allow_live,
        )
        succeeded = (
            outcome.get("status") == "complete"
            and outcome.get("gate_passed") is True
        )
    elif args.command == "retry-failed":
        outcome = runner.retry_failed_from_config(
            args.config,
            run_dir=args.run,
            max_workers=args.max_workers,
            error_codes=args.error_codes,
            allow_live=args.allow_live,
        )
        succeeded = (
            outcome.get("status") == "complete"
            and outcome.get("remaining_failed_case_count") == 0
            and outcome.get("gate_passed") is True
        )
    else:
        outcome = runner.score_from_config(
            args.config,
            run_dir=args.run,
            resume=args.resume,
            allow_live=args.allow_live,
        )
        succeeded = outcome.get("status") == "complete"
    print(json.dumps(outcome, ensure_ascii=False, indent=2))
    return 0 if succeeded else 1


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "prepare-data":
            if not args.qa_catalog_only and not args.balanced_pdfs_only:
                raise DocBenchDownloadError(
                    "prepare-data requires --qa-catalog-only or "
                    "--balanced-pdfs-only; config-driven exact preparation "
                    "is not implemented"
                )
            return download_selection._run_cli(args)
        if args.command == "build-selection":
            if not args.balanced:
                raise SelectionValidationError(
                    "build-selection requires --balanced; legacy formal-100 "
                    "selection generation is retired from the unified entry"
                )
            return selection._run_cli(args)
        if args.command == "validate":
            configs = load_checked_in_configs()
            print(f"valid: {len(configs)} DocBench L1 configs")
            return 0
        if args.command == "readiness":
            report = readiness.build_readiness_report(
                tuple(sorted(CONFIGS_DIR.glob("l1_*.yaml")))
            )
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return 0 if report.get("status") == "ready" else 2
        if args.command == "list":
            for config in load_checked_in_configs():
                print(f"{config.source_path.stem}\t{config.sha256}")
            return 0
        return _run_action(args)
    except (
        DocBenchConfigError,
        DocBenchDownloadError,
        DocBenchScoringError,
        SelectionValidationError,
        runner.DocBenchRunnerError,
    ) as exc:
        print(f"DocBench error: {exc}")
        return 2


__all__ = ["CONFIGS_DIR", "load_checked_in_configs", "main"]
