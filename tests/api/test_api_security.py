from __future__ import annotations

import os
import stat
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from personagraph.api import security


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="platform has no symlink support")
def test_token_write_never_follows_preplanted_symlinks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    token_path = tmp_path / "api_secret"
    victim = tmp_path / "victim.txt"
    victim.write_text("do-not-overwrite", encoding="utf-8")
    token_path.symlink_to(victim)

    legacy_temporary = token_path.with_suffix(".tmp")
    legacy_temporary.symlink_to(victim)
    planted_random_temporary = tmp_path / ".api_secret.collision.tmp"
    planted_random_temporary.symlink_to(victim)
    names = iter(("collision", "fresh"))
    monkeypatch.setattr(security, "TOKEN_PATH", token_path)
    monkeypatch.setattr(security.secrets, "token_hex", lambda _size: next(names))

    security._write_token_file("new-private-token")

    assert token_path.is_file()
    assert not token_path.is_symlink()
    assert token_path.read_text(encoding="utf-8") == "new-private-token"
    assert victim.read_text(encoding="utf-8") == "do-not-overwrite"
    assert legacy_temporary.is_symlink()
    assert planted_random_temporary.is_symlink()
    assert not (tmp_path / ".api_secret.fresh.tmp").exists()
    if os.name == "posix":
        assert stat.S_IMODE(token_path.stat().st_mode) == 0o600


def test_concurrent_token_writes_are_complete_and_leave_no_temporary_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    token_path = tmp_path / "api_secret"
    tokens = tuple(f"token-{index}-".ljust(64, chr(65 + index)) for index in range(12))
    monkeypatch.setattr(security, "TOKEN_PATH", token_path)

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(security._write_token_file, tokens))

    assert token_path.read_text(encoding="utf-8") in tokens
    assert list(tmp_path.glob(".api_secret.*.tmp")) == []
    if os.name == "posix":
        assert stat.S_IMODE(token_path.stat().st_mode) == 0o600


def test_token_write_cleans_random_temporary_after_replace_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    token_path = tmp_path / "api_secret"
    temporary = tmp_path / ".api_secret.cleanup.tmp"
    monkeypatch.setattr(security, "TOKEN_PATH", token_path)
    monkeypatch.setattr(security.secrets, "token_hex", lambda _size: "cleanup")

    def fail_replace(_source, _destination):
        raise OSError("replace failed")

    monkeypatch.setattr(security.os, "replace", fail_replace)

    with pytest.raises(OSError, match="replace failed"):
        security._write_token_file("new-private-token")

    assert not temporary.exists()
    assert not token_path.with_suffix(".tmp").exists()
    assert not token_path.exists()
