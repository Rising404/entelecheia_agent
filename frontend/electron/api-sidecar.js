const { spawn, spawnSync } = require("node:child_process");
const crypto = require("node:crypto");
const fs = require("node:fs");
const http = require("node:http");
const path = require("node:path");
const { buildRuntimeIdentity } = require("./runtime-identity");

const API_HOST = "127.0.0.1";
const API_PORT = Number(process.env.PERSONAGRAPH_API_PORT || 8765);
const API_BASE = `http://${API_HOST}:${API_PORT}`;
// 先证明令牌所有权，再以受认证的源码身份决定能否复用现有进程。
const API_AUTH_PROBE_PATH = "/api/runtime-identity";
const API_IDENTITY_SCHEME = "hmac-sha256-v1";
const API_IDENTITY_CONTEXT = "entelecheia-loopback-api-ownership-v1\0";
const API_IDENTITY_PROOF_PATTERN = /^[A-Za-z0-9_-]{43}$/;

let apiProcess = null;
let startInFlight = null;

function repoRoot() {
  return path.resolve(__dirname, "..", "..");
}

function inspectPythonEnvironment(candidate) {
  const env = { ...process.env, PYTHONNOUSERSITE: "1" };
  delete env.PYTHONHOME;
  delete env.PYTHONPATH;
  delete env.VIRTUAL_ENV;
  const probe = spawnSync(
    candidate,
    [
      "-I",
      "-c",
      "import json,sys,sysconfig; print(json.dumps({'base_prefix':sys.base_prefix,'purelib':sysconfig.get_paths()['purelib']}))"
    ],
    {
      encoding: "utf8",
      env,
      timeout: 3000,
      windowsHide: true
    }
  );
  if (probe.error || probe.status !== 0) return null;
  try {
    return JSON.parse(String(probe.stdout || "").trim());
  } catch {
    return null;
  }
}

function canonicalPath(candidate) {
  try {
    return fs.realpathSync.native(candidate);
  } catch {
    return null;
  }
}

function pathIsInside(candidate, parent) {
  const relative = path.relative(parent, candidate);
  return (
    relative !== "" &&
    relative !== ".." &&
    !relative.startsWith(`..${path.sep}`) &&
    !path.isAbsolute(relative)
  );
}

function resolvePython(root = repoRoot(), inspect = inspectPythonEnvironment) {
  const candidates = [
    path.join(root, ".venv", "bin", "python"),
    path.join(root, ".venv", "Scripts", "python.exe")
  ];
  const runtimeRoot = path.join(root, ".runtime");
  const runtimePython = path.join(runtimeRoot, "python");
  const venvRoot = path.join(root, ".venv");
  try {
    if (
      fs.lstatSync(runtimeRoot).isSymbolicLink() ||
      fs.lstatSync(runtimePython).isSymbolicLink()
    ) {
      return null;
    }
  } catch {
    return null;
  }
  const expectedBasePrefix = canonicalPath(runtimePython);
  const expectedPurelibRoot = canonicalPath(venvRoot);
  if (!expectedBasePrefix || !expectedPurelibRoot) return null;

  for (const candidate of candidates) {
    try {
      fs.accessSync(candidate, fs.constants.X_OK);
    } catch {
      // Continue through project-owned candidates only.
      continue;
    }
    const environment = inspect(candidate);
    const basePrefix = canonicalPath(environment?.base_prefix || "");
    const purelib = canonicalPath(environment?.purelib || "");
    if (
      basePrefix === expectedBasePrefix &&
      purelib &&
      pathIsInside(purelib, expectedPurelibRoot)
    ) return candidate;
  }
  return null;
}

function healthUrl() {
  return `${API_BASE}/api/health`;
}

function readApiToken(apiTokenPath) {
  try {
    const token = fs.readFileSync(apiTokenPath, "utf8").trim();
    return token.length >= 32 ? token : null;
  } catch {
    return null;
  }
}

function checkHealth(timeoutMs = 800) {
  return new Promise((resolve) => {
    const req = http.get(healthUrl(), { timeout: timeoutMs }, (res) => {
      let body = "";
      res.setEncoding("utf8");
      res.on("data", (chunk) => {
        if (body.length < 4096) body += chunk;
      });
      res.on("end", () => {
        try {
          const payload = JSON.parse(body);
          resolve(
            res.statusCode === 200 &&
            payload?.ok === true &&
            payload?.service === "personagraph-api"
          );
        } catch {
          resolve(false);
        }
      });
      res.on("error", () => resolve(false));
    });
    req.on("timeout", () => {
      req.destroy();
      resolve(false);
    });
    req.on("error", () => resolve(false));
  });
}

function checkApiRuntimeIdentity(apiToken, expectedRuntime, timeoutMs = 800) {
  if (typeof apiToken !== "string" || apiToken.length < 32) {
    return Promise.resolve(false);
  }

  return new Promise((resolve) => {
    let settled = false;
    const finish = (value) => {
      if (settled) return;
      settled = true;
      resolve(value);
    };
    let req;
    try {
      req = http.get(
        `${API_BASE}${API_AUTH_PROBE_PATH}`,
        {
          timeout: timeoutMs,
          headers: { Authorization: `Bearer ${apiToken}` }
        },
        (res) => {
          const contentType = String(res.headers?.["content-type"] || "").toLowerCase();
          let body = "";
          res.setEncoding("utf8");
          res.on("data", (chunk) => { body += chunk.slice(0, Math.max(0, 4096 - body.length)); });
          res.on("end", () => {
            try {
              const identity = JSON.parse(body);
              finish(
                res.statusCode === 200 &&
                contentType.startsWith("application/json") &&
                identity?.checkout_sha256 === expectedRuntime.checkout_sha256 &&
                identity?.source_sha256 === expectedRuntime.source_sha256
              );
            } catch { finish(false); }
          });
          res.on("error", () => finish(false));
        }
      );
    } catch {
      finish(false);
      return;
    }
    req.on("timeout", () => {
      req.destroy();
      finish(false);
    });
    req.on("error", () => finish(false));
  });
}

function checkApiOwnership(apiToken, timeoutMs = 800) {
  if (typeof apiToken !== "string" || apiToken.length < 32) {
    return Promise.resolve(false);
  }

  let challenge;
  let expectedProof;
  try {
    challenge = crypto.randomBytes(32).toString("base64url");
    expectedProof = crypto
      .createHmac("sha256", apiToken)
      .update(API_IDENTITY_CONTEXT, "utf8")
      .update(challenge, "ascii")
      .digest();
  } catch {
    return Promise.resolve(false);
  }

  return new Promise((resolve) => {
    let settled = false;
    const finish = (value) => {
      if (settled) return;
      settled = true;
      resolve(value);
    };
    let req;
    try {
      req = http.get(
        `${healthUrl()}?challenge=${encodeURIComponent(challenge)}`,
        { timeout: timeoutMs },
        (res) => {
          let body = "";
          res.setEncoding("utf8");
          res.on("data", (chunk) => {
            const remaining = 4096 - body.length;
            if (remaining > 0) body += chunk.slice(0, remaining);
          });
          res.on("end", () => {
            try {
              const payload = JSON.parse(body);
              const proof = payload?.identity?.proof;
              if (
                res.statusCode !== 200 ||
                payload?.ok !== true ||
                payload?.service !== "personagraph-api" ||
                payload?.identity?.scheme !== API_IDENTITY_SCHEME ||
                typeof proof !== "string" ||
                !API_IDENTITY_PROOF_PATTERN.test(proof)
              ) {
                finish(false);
                return;
              }
              const receivedProof = Buffer.from(proof, "base64url");
              finish(
                receivedProof.length === expectedProof.length &&
                crypto.timingSafeEqual(receivedProof, expectedProof)
              );
            } catch {
              finish(false);
            }
          });
          res.on("error", () => finish(false));
        }
      );
    } catch {
      finish(false);
      return;
    }
    req.on("timeout", () => {
      req.destroy();
      finish(false);
    });
    req.on("error", () => finish(false));
  });
}

async function readAuthorizedApiToken(apiTokenPath, expectedRuntime, timeoutMs = 800) {
  const apiToken = readApiToken(apiTokenPath);
  if (!apiToken) return null;
  if (!await checkApiOwnership(apiToken, timeoutMs)) return null;
  return await checkApiRuntimeIdentity(apiToken, expectedRuntime, timeoutMs) ? apiToken : null;
}

async function waitForAuthorizedApi(apiTokenPath, expectedRuntime, timeoutMs = 5000) {
  const started = Date.now();
  while (Date.now() - started < timeoutMs) {
    if (await checkHealth(500)) {
      const apiToken = await readAuthorizedApiToken(apiTokenPath, expectedRuntime, 500);
      if (apiToken) return apiToken;
    }
    await new Promise((resolve) => setTimeout(resolve, 200));
  }
  return null;
}

function validateStorageBoundary(storage, root) {
  if (!storage?.stateDir || !storage?.configDir || !storage?.apiTokenPath) {
    throw new Error("startApiSidecar requires resolved state, config, and API token paths");
  }
  const sourceRoot = path.resolve(root);
  for (const [name, candidate] of [
    ["stateDir", storage.stateDir],
    ["configDir", storage.configDir],
    ["apiTokenPath", storage.apiTokenPath]
  ]) {
    if (!path.isAbsolute(candidate)) {
      throw new Error(`${name} must be an absolute path`);
    }
    const relative = path.relative(sourceRoot, path.resolve(candidate));
    if (
      relative === "" ||
      (relative !== ".." && !relative.startsWith(`..${path.sep}`) && !path.isAbsolute(relative))
    ) {
      throw new Error(`${name} must be outside the source checkout`);
    }
  }
}

async function _startApiSidecar(options = {}) {
  const root = options.root || repoRoot();
  const storage = options.storage;
  validateStorageBoundary(storage, root);
  let expectedRuntime;
  try {
    expectedRuntime = buildRuntimeIdentity(root);
  } catch {
    return { apiBase: API_BASE, apiToken: null, mode: "spawn-failed", ok: false,
      error: "local_runtime_identity_unavailable" };
  }
  if (await checkHealth()) {
    const apiToken = await readAuthorizedApiToken(storage.apiTokenPath, expectedRuntime);
    const ok = Boolean(apiToken);
    return {
      apiBase: API_BASE,
      apiToken,
      mode: ok ? "existing" : "existing-incompatible",
      ok,
      error: ok ? null : "existing_api_token_or_runtime_identity_rejected"
    };
  }

  const python = resolvePython(root);
  if (!python) {
    return {
      apiBase: API_BASE,
      mode: "spawn-failed",
      ok: false,
      python: null,
      error: "independent_python_runtime_unavailable"
    };
  }
  if (apiProcess) {
    const apiToken = await waitForAuthorizedApi(
      storage.apiTokenPath,
      expectedRuntime,
      options.waitTimeoutMs || 5000
    );
    const ok = Boolean(apiToken);
    const healthy = ok || await checkHealth(500);
    return {
      apiBase: API_BASE,
      apiToken,
      mode: ok ? "spawned" : (healthy ? "spawned-incompatible" : "starting"),
      ok,
      python,
      error: ok
        ? null
        : (healthy ? "sidecar_token_or_runtime_identity_rejected" : "sidecar_not_healthy_yet")
    };
  }

  const env = {
    ...process.env,
    PERSONAGRAPH_STATE_DIR: storage.stateDir,
    PERSONAGRAPH_LOCAL_CONFIG_DIR: storage.configDir,
    PERSONAGRAPH_API_TOKEN_FILE: storage.apiTokenPath,
    PYTHONNOUSERSITE: "1",
    PYTHONPATH: path.join(root, "src")
  };
  delete env.PYTHONHOME;
  delete env.VIRTUAL_ENV;
  let spawnError = null;
  try {
    apiProcess = spawn(
      python,
      ["-m", "personagraph.api.server", "--host", API_HOST, "--port", String(API_PORT)],
      {
        cwd: root,
        env,
        stdio: ["ignore", "pipe", "pipe"],
        windowsHide: true
      }
    );
  } catch (error) {
    return {
      apiBase: API_BASE,
      mode: "spawn-failed",
      ok: false,
      python,
      error: error.message
    };
  }

  apiProcess.stdout?.on("data", (chunk) => {
    console.log(`[personagraph-api] ${String(chunk).trim()}`);
  });
  apiProcess.stderr?.on("data", (chunk) => {
    console.error(`[personagraph-api] ${String(chunk).trim()}`);
  });
  apiProcess.on("error", (error) => {
    spawnError = error;
    console.error(`[personagraph-api] failed to start: ${error.message}`);
    apiProcess = null;
  });
  apiProcess.on("exit", (code, signal) => {
    if (apiProcess) {
      console.log(`[personagraph-api] exited code=${code} signal=${signal}`);
    }
    apiProcess = null;
  });

  const apiToken = await waitForAuthorizedApi(storage.apiTokenPath, expectedRuntime);
  const ok = Boolean(apiToken);
  const healthy = ok || await checkHealth(500);
  return {
    apiBase: API_BASE,
    apiToken,
    mode: ok ? "spawned" : (healthy ? "spawned-incompatible" : "spawn-failed"),
    ok,
    python,
    error: spawnError?.message || (
      healthy ? "sidecar_token_or_runtime_identity_rejected" : "sidecar_not_healthy"
    )
  };
}

async function startApiSidecar(options = {}) {
  if (startInFlight) return startInFlight;
  startInFlight = _startApiSidecar(options);
  try {
    return await startInFlight;
  } finally {
    startInFlight = null;
  }
}

function stopApiSidecar() {
  if (!apiProcess) {
    return;
  }
  apiProcess.kill();
  apiProcess = null;
}

module.exports = {
  API_BASE,
  API_HOST,
  API_PORT,
  checkHealth,
  resolvePython,
  startApiSidecar,
  stopApiSidecar
};
