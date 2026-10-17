"""后端身份在进程启动时冻结，不得把磁盘新代码冒充为已加载代码。"""

import json
import shutil
import subprocess
import threading
from pathlib import Path

from personagraph.api import runtime_identity


def _checkout(root: Path) -> Path:
    package = root / "src" / "personagraph"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("# 合成源码\n", encoding="utf-8")
    (root / "pyproject.toml").write_text('[project]\nname = "fixture"\n', encoding="utf-8")
    return root


def test_identity_changes_with_checkout_source_and_dependency_lock(tmp_path):
    root = _checkout(tmp_path / "one")
    initial = runtime_identity.build_runtime_identity(root)
    other = tmp_path / "two"
    shutil.copytree(root, other)
    copied = runtime_identity.build_runtime_identity(other)
    assert initial["checkout_sha256"] != copied["checkout_sha256"]
    assert initial["source_sha256"] == copied["source_sha256"]

    (root / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    locked = runtime_identity.build_runtime_identity(root)
    assert initial["source_sha256"] != locked["source_sha256"]
    (root / "src/personagraph/__init__.py").write_text("# 修改源码\n", encoding="utf-8")
    assert runtime_identity.build_runtime_identity(root)["source_sha256"] != locked["source_sha256"]


def test_identity_ignores_bytecode_and_private_state(tmp_path):
    root = _checkout(tmp_path)
    initial = runtime_identity.build_runtime_identity(root)
    (root / ".env").write_text("not-a-real-secret", encoding="utf-8")
    cache = root / "src/personagraph/__pycache__"
    cache.mkdir()
    (cache / "ignored.pyc").write_bytes(b"bytecode")
    assert runtime_identity.build_runtime_identity(root) == initial
    assert str(root) not in json.dumps(initial)


def test_runtime_endpoint_returns_frozen_identity_without_rehashing(monkeypatch):
    from personagraph.api.router import dispatch_response

    expected = runtime_identity.runtime_identity()
    monkeypatch.setattr(runtime_identity, "build_runtime_identity", lambda _root: {"unexpected": True})
    assert dispatch_response("GET", "/api/runtime-identity", {}).payload == expected


def test_electron_and_python_compute_the_same_identity(tmp_path):
    root = _checkout(tmp_path / "工作区")
    (root / "src/personagraph/资料.py").write_text("# UTF-8\n", encoding="utf-8")
    repo = Path(__file__).resolve().parents[2]
    node = repo / ".runtime/node/bin/node"
    executable = str(node) if node.is_file() else shutil.which("node")
    if executable is None:
        import pytest
        pytest.skip("Node is not installed; frontend CI also covers this contract")
    result = subprocess.run(
        [executable, "-e", "console.log(JSON.stringify(require(process.argv[1]).buildRuntimeIdentity(process.argv[2])))",
         str(repo / "frontend/electron/runtime-identity.js"), str(root)],
        check=True, capture_output=True, text=True,
    )
    assert json.loads(result.stdout) == runtime_identity.build_runtime_identity(root)


def test_electron_reuses_matching_backend_over_authenticated_loopback(tmp_path, monkeypatch):
    """真实本地 HTTP 握手；不启动任务、模型或用户数据后台服务。"""
    from personagraph.api import security, server

    repo = Path(__file__).resolve().parents[2]
    node = repo / ".runtime/node/bin/node"
    executable = str(node) if node.is_file() else shutil.which("node")
    if executable is None:
        import pytest
        pytest.skip("Node is not installed")
    token = "synthetic-loopback-token".ljust(48, "x")
    token_path = tmp_path / "api_secret"
    token_path.write_text(token, encoding="utf-8")
    monkeypatch.setattr(security, "_API_TOKEN", token)
    httpd = server.ThreadingHTTPServer(("127.0.0.1", 0), server.ApiHandler)
    worker = threading.Thread(target=httpd.serve_forever, daemon=True)
    worker.start()
    try:
        result = subprocess.run(
            [executable, "-e", """
process.env.PERSONAGRAPH_API_PORT = process.argv[1];
const { startApiSidecar } = require(process.argv[2] + '/frontend/electron/api-sidecar.js');
startApiSidecar({root: process.argv[2], storage: {
  stateDir: process.argv[3] + '/state', configDir: process.argv[3] + '/config',
  apiTokenPath: process.argv[3] + '/api_secret'
}}).then(({ ok, mode, error }) => console.log(JSON.stringify({ok, mode, error})));
""", str(httpd.server_port), str(repo), str(tmp_path)],
            check=True, capture_output=True, text=True, timeout=15,
        )
        assert json.loads(result.stdout) == {"ok": True, "mode": "existing", "error": None}
    finally:
        httpd.shutdown()
        httpd.server_close()
        worker.join(timeout=5)
