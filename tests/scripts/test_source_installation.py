"""Focused source-install contracts; no package installation or model transfers."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import re
import tomllib

from packaging.requirements import Requirement
import pytest


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def model_setup():
    spec = importlib.util.spec_from_file_location(
        "entelecheia_model_setup", ROOT / "scripts" / "prepare-local-models.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_model_check_never_requests_remote_files(model_setup, monkeypatch):
    huggingface_hub = pytest.importorskip("huggingface_hub")

    requests = []

    def missing_snapshot(**kwargs):
        requests.append(kwargs)
        raise FileNotFoundError("synthetic empty cache")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", missing_snapshot)
    result = model_setup.prepare("check")

    assert result["status"] == "not_ready"
    assert len(result["assets"]) == 2
    assert len(requests) == 2
    assert all(request["local_files_only"] is True for request in requests)
    assert all(re.fullmatch(r"[a-f0-9]{40}", request["revision"]) for request in requests)


def test_explicit_download_uses_pinned_public_assets_and_safe_errors(model_setup, monkeypatch):
    huggingface_hub = pytest.importorskip("huggingface_hub")

    requests = []

    def fail_snapshot(**kwargs):
        requests.append(kwargs)
        raise RuntimeError("private exception payload must not be published")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fail_snapshot)
    result = model_setup.prepare("download")

    transfers = [request for request in requests if not request["local_files_only"]]
    assert len(transfers) == 2
    assert all(request["token"] is False for request in transfers)
    assert all(re.fullmatch(r"[a-f0-9]{40}", request["revision"]) for request in transfers)
    assert result["status"] == "not_ready"
    assert len(result["download_failures"]) == 2
    assert "private exception payload" not in str(result)


def test_model_check_without_optional_hub_package_is_structured(model_setup, monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "huggingface_hub", None)

    result = model_setup.prepare("check")

    assert result["status"] == "not_ready"
    assert len(result["assets"]) == 2
    assert all(
        asset["reason_code"] == "huggingface_hub_unavailable"
        for asset in result["assets"]
    )


def test_invalid_model_preparation_command_does_not_download(model_setup):
    with pytest.raises(ValueError, match="check or download"):
        model_setup.prepare("automatic")


def test_source_install_has_one_locked_dependency_path():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    bootstrap = (ROOT / "scripts" / "bootstrap-local-runtime.sh").read_text()
    lock = (ROOT / "requirements-macos-arm64.lock").read_text()

    assert project["build-system"]["build-backend"] == "setuptools.build_meta"
    assert "eval" in project["project"]["optional-dependencies"]
    assert "--require-hashes" in bootstrap
    assert "--no-deps --no-build-isolation" in bootstrap
    assert "--no-index" in bootstrap
    assert "--no-binary=antlr4-python3-runtime --no-build-isolation" in bootstrap
    assert "selected = ($0 ~ /^setuptools==/)" in bootstrap
    assert '"$OS:$ARCH" != "Darwin:arm64"' in bootstrap
    assert "gdown==" in lock
    for raw_requirement in project["build-system"]["requires"]:
        requirement = Requirement(raw_requirement)
        pin = re.search(rf"^{re.escape(requirement.name)}==([^ ;\\\n]+)", lock, re.MULTILINE)
        assert pin is not None, f"build dependency missing from lock: {requirement.name}"
        assert pin.group(1) in requirement.specifier
    assert "--hash=sha256:" in lock
    assert "/Users/" not in lock and "/private/tmp/" not in lock
