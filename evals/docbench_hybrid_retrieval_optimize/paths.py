"""Retrieval artifacts have one explicit boundary, separate from source code."""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIRECTORIES = ("dataset", "indexes", "results")


def artifact_path(path: Path) -> Path:
    """Allow external paths or the dedicated artifact directories, never source files.

    Check both the requested location and its resolved target so symlinks cannot
    turn an artifact directory into an output route to another repository location.
    """
    requested = Path(os.path.abspath(Path(path).expanduser()))
    resolved = requested.resolve()
    roots = tuple(
        PROJECT_ROOT / "evals" / "docbench_hybrid_retrieval_optimize" / name
        for name in ARTIFACT_DIRECTORIES
    )
    for location in (requested, resolved):
        if location.is_relative_to(PROJECT_ROOT) and not any(
            location.is_relative_to(root) for root in roots
        ):
            raise ValueError(
                "Retrieval artifacts must stay outside the repository "
                "or inside docbench_hybrid_retrieval_optimize/dataset, indexes, results"
            )
    for root in roots:
        if requested.is_relative_to(root) and not resolved.is_relative_to(root):
            raise ValueError("Retrieval artifact symlink escapes its allowed directory")
    return resolved
