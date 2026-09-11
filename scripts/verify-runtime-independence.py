from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_PARTS = (
    "/.cache/codex-runtimes/",
    "/Codex.app/Contents/Resources/",
    "/ChatGPT.app/Contents/Resources/",
)


def _reject_forbidden(label: str, value: str) -> None:
    normalized = value.replace("\\", "/")
    if any(part in normalized for part in FORBIDDEN_PARTS):
        raise RuntimeError(f"{label} still resolves through a Codex runtime: {value}")


def _resolved(path: Path) -> str:
    if not path.exists():
        raise RuntimeError(f"required runtime path is missing: {path}")
    return str(path.resolve())


def _runtime_versions() -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in (ROOT / "runtime-versions.conf").read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        name, separator, value = line.partition("=")
        if not separator or not name or not value:
            raise RuntimeError("runtime-versions.conf contains an invalid entry")
        values[name] = value
    return values


def main() -> int:
    versions = _runtime_versions()
    runtime_root = ROOT / ".runtime"
    if os.name == "nt":
        venv_python = ROOT / ".venv" / "Scripts" / "python.exe"
        runtime_node = ROOT / ".runtime" / "node" / "node.exe"
    else:
        venv_python = ROOT / ".venv" / "bin" / "python"
        runtime_node = ROOT / ".runtime" / "node" / "bin" / "node"
    runtime_python = ROOT / ".runtime" / "python"

    for path in (runtime_root, runtime_python, runtime_root / "node"):
        if path.is_symlink():
            raise RuntimeError(f"project runtime root must not be a symlink: {path}")

    clean_environment = {"PATH": os.pathsep.join(_system_path_entries())}
    if os.name == "nt" and os.getenv("SystemRoot"):
        clean_environment["SystemRoot"] = str(os.environ["SystemRoot"])

    python_probe = json.loads(
        subprocess.check_output(
            [
                str(venv_python),
                "-c",
                (
                    "import json,platform,sys,sysconfig;"
                    "print(json.dumps({'executable':sys.executable,"
                    "'version':platform.python_version(),"
                    "'base_prefix':sys.base_prefix,"
                    "'stdlib':sysconfig.get_paths()['stdlib'],"
                    "'purelib':sysconfig.get_paths()['purelib']}))"
                ),
            ],
            text=True,
            env=clean_environment,
        )
    )
    node_probe = json.loads(
        subprocess.check_output(
            [
                str(runtime_node),
                "-p",
                "JSON.stringify({execPath:process.execPath,version:process.version})",
            ],
            text=True,
            env=clean_environment,
        )
    )

    for label, value in (*python_probe.items(), *node_probe.items()):
        _reject_forbidden(label, str(value))
    if Path(python_probe["base_prefix"]).resolve() != runtime_python.resolve():
        raise RuntimeError("venv base_prefix is not the project-owned Python runtime")
    if python_probe["version"] != versions["PERSONAGRAPH_PYTHON_VERSION"]:
        raise RuntimeError("Python version does not match runtime-versions.conf")
    if not Path(python_probe["purelib"]).resolve().is_relative_to(
        (ROOT / ".venv").resolve()
    ):
        raise RuntimeError("Python packages are not isolated in the project venv")
    if _resolved(runtime_node) != str(Path(node_probe["execPath"]).resolve()):
        raise RuntimeError("Node execPath is not the project-owned Node runtime")
    if node_probe["version"] != f"v{versions['PERSONAGRAPH_NODE_VERSION']}":
        raise RuntimeError("Node version does not match runtime-versions.conf")

    startup_files = (
        ROOT / "frontend" / "start-electron.command",
        ROOT / "frontend" / "start-web.command",
        ROOT / "frontend" / "electron" / "macos-launcher.js",
        ROOT / "frontend" / "electron" / "api-sidecar.js",
        ROOT / "frontend" / "start-electron.ps1",
    )
    for path in startup_files:
        contents = path.read_text(encoding="utf-8")
        if "codex-runtimes" in contents or "CODEX_MCP_NODE_PATH" in contents:
            raise RuntimeError(f"startup path still contains a Codex fallback: {path}")

    print(
        json.dumps(
            {
                "status": "independent",
                "python": python_probe,
                "node": node_probe,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def _system_path_entries() -> tuple[str, ...]:
    if os.name == "nt":
        system_root = Path(os.environ.get("SystemRoot", r"C:\Windows"))
        return (str(system_root / "System32"), str(system_root))
    return ("/usr/bin", "/bin", "/usr/sbin", "/sbin")


if __name__ == "__main__":
    sys.exit(main())
