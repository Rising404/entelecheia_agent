"""用于检索 generation 诊断和恢复的最小显式 CLI。"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from ..lifecycle.generation import plan_previous_generation_restore
from ..sqlite_store import SqliteRetrievalCatalog


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="entelecheia-retrieval")
    parser.add_argument(
        "command",
        choices=(
            "rollout-document",
            "diagnose-previous",
            "restore-previous",
        ),
    )
    parser.add_argument("--corpus", choices=("file",), default="file")
    parser.add_argument("--target-generation-id")
    parser.add_argument("--expected-fingerprint")
    parser.add_argument("--project-id")
    parser.add_argument("--project-root", type=Path)
    parser.add_argument(
        "--db-path",
        type=Path,
        required=True,
        help="explicit project/session database path for this operation",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    db_path = args.db_path
    if args.command == "rollout-document":
        if not args.project_id or args.project_root is None:
            parser.error(
                "rollout-document requires --project-id and --project-root"
            )
        from ...workspace.storage.context import bind
        from ...workspace.storage.database import DocumentDatabase
        from .document_maintenance import (
            rollout_document_retrieval_generation,
        )

        database = DocumentDatabase(
            project_id=args.project_id,
            project_root=args.project_root,
            db_path=db_path,
        )
        with bind(database):
            result = rollout_document_retrieval_generation(
                retrieval_db_path=db_path,
            )
        payload = {
            "status": result.status.value,
            "generation_id": result.generation_id,
            "generation_fingerprint": result.generation_fingerprint,
            "previous_generation_id": result.previous_generation_id,
            "source_unit_count": result.source_unit_count,
            "method_coverage": [
                {
                    **asdict(item),
                    "method": item.method.value,
                    "ready": item.ready,
                }
                for item in result.method_coverage
            ],
            "profile": "environment",
            "local_files_only": True,
            "db_path": str(Path(db_path)),
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    catalog = SqliteRetrievalCatalog(db_path)
    if args.command == "restore-previous" and (
        not args.target_generation_id or not args.expected_fingerprint
    ):
        parser.error(
            "restore-previous requires --target-generation-id and "
            "--expected-fingerprint"
        )
    if args.command == "restore-previous":
        if not args.project_id or args.project_root is None:
            parser.error(
                "restore-previous requires --project-id and --project-root"
            )
        from ...workspace.storage.context import bind
        from ...workspace.storage.database import DocumentDatabase
        from .document_maintenance import (
            restore_previous_document_retrieval_generation,
        )

        database = DocumentDatabase(
            project_id=args.project_id,
            project_root=args.project_root,
            db_path=db_path,
        )
        with bind(database):
            result = restore_previous_document_retrieval_generation(
                previous_generation_id=args.target_generation_id,
                expected_fingerprint=args.expected_fingerprint,
                retrieval_db_path=db_path,
            )
    else:
        result = plan_previous_generation_restore(
            catalog,
            target_generation_id=args.target_generation_id,
            expected_fingerprint=args.expected_fingerprint,
        )
    payload = asdict(result)
    payload["status"] = result.status.value
    payload["corpus"] = "file"
    payload["db_path"] = str(Path(db_path))
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if result.can_restore or result.status.value == "restored" else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
