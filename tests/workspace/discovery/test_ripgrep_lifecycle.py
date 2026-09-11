from __future__ import annotations

from io import StringIO
from pathlib import Path

from personagraph.workspace.discovery import ripgrep


class _CompletedProcess:
    def __init__(self) -> None:
        self.stdout = StringIO("one\n")
        self.stderr = StringIO("diagnostic\n")
        self.returncode = 0

    def wait(self, timeout: float | None = None) -> int:
        del timeout
        return self.returncode

    def kill(self) -> None:
        raise AssertionError("a completed process must not be killed")


def test_streaming_run_closes_both_subprocess_pipes(monkeypatch) -> None:
    process = _CompletedProcess()
    monkeypatch.setattr(ripgrep.subprocess, "Popen", lambda *_args, **_kwargs: process)

    assert list(
        ripgrep._run(
            ["rg", "--files"],
            timeout_s=1.0,
            report=ripgrep.ScanReport(),
        )
    ) == ["one\n"]

    assert process.stdout.closed is True
    assert process.stderr.closed is True


def test_binary_path_accepts_a_windows_virtualenv_scripts_binary(
    tmp_path: Path,
    monkeypatch,
) -> None:
    executable = tmp_path / "Scripts" / "rg.exe"
    executable.parent.mkdir()
    executable.write_bytes(b"stub")

    monkeypatch.setattr(ripgrep.sys, "prefix", str(tmp_path))
    monkeypatch.setattr(ripgrep.shutil, "which", lambda _name: None)
    monkeypatch.setattr(
        ripgrep.os,
        "access",
        lambda candidate, _mode: Path(candidate) == executable,
    )

    assert ripgrep.binary_path() == str(executable)
