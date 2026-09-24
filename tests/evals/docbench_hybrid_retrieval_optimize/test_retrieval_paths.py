from pathlib import Path

import pytest

from evals.docbench_hybrid_retrieval_optimize import paths


@pytest.fixture
def repository(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "repository"
    root.mkdir()
    monkeypatch.setattr(paths, "PROJECT_ROOT", root)
    return root


def test_artifacts_are_allowed_only_in_dedicated_repository_directories(repository: Path) -> None:
    package = repository / "evals" / "docbench_hybrid_retrieval_optimize"
    for name in ("dataset", "indexes", "results"):
        target = package / name / "example" / "manifest.json"
        assert paths.artifact_path(target) == target
    for target in (repository, package, package / "cli.py", repository / "var" / "output"):
        with pytest.raises(ValueError, match="outside the repository"):
            paths.artifact_path(target)


def test_external_artifacts_remain_supported(repository: Path, tmp_path: Path) -> None:
    target = tmp_path / "external" / "dataset.sqlite"
    assert paths.artifact_path(target) == target


def test_artifact_symlink_cannot_escape_to_source_or_external_directory(
    repository: Path, tmp_path: Path,
) -> None:
    dataset = repository / "evals" / "docbench_hybrid_retrieval_optimize" / "dataset"
    dataset.mkdir(parents=True)
    for name, destination in (("source", repository), ("external", tmp_path)):
        alias = dataset / name
        alias.symlink_to(destination, target_is_directory=True)
        with pytest.raises(ValueError):
            paths.artifact_path(alias / "output")


def test_external_symlink_cannot_target_source_code(repository: Path, tmp_path: Path) -> None:
    alias = tmp_path / "source-link"
    alias.symlink_to(repository, target_is_directory=True)
    with pytest.raises(ValueError, match="outside the repository"):
        paths.artifact_path(alias / "module.py")
