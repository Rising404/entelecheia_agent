"""The canonical installation recipe participates in experiment identity."""
from pathlib import Path
import subprocess

from evals.docbench.reproduce_or_run_script import provenance


def test_source_fingerprint_tracks_the_current_installation_contract(tmp_path: Path):
    subprocess.run(
        ['git', 'init', '--template=', str(tmp_path)],
        check=True, capture_output=True,
    )
    (tmp_path / 'pyproject.toml').write_text('[project]\nname="synthetic"\n')
    baseline = provenance.compute_source_tree_sha256(tmp_path)
    assert baseline is not None
    for relative in (
        'requirements-macos-arm64.lock',
        'runtime-versions.conf',
        'scripts/bootstrap-local-runtime.sh',
    ):
        artifact = tmp_path / relative
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text('synthetic installation input\n')
        after_add = provenance.compute_source_tree_sha256(tmp_path)
        assert after_add != baseline, relative
        artifact.write_text('changed installation input\n')
        after_change = provenance.compute_source_tree_sha256(tmp_path)
        assert after_change != after_add, relative
        baseline = after_change
