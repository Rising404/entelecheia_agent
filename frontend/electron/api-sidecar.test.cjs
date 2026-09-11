const assert = require("node:assert/strict");
const crypto = require("node:crypto");
const { EventEmitter } = require("node:events");
const fs = require("node:fs");
const http = require("node:http");
const os = require("node:os");
const path = require("node:path");
const test = require("node:test");

const API_IDENTITY_CONTEXT = "entelecheia-loopback-api-ownership-v1\0";

test("Python resolution only accepts a project-owned runtime", (t) => {
  const temporaryRoot = fs.mkdtempSync(path.join(os.tmpdir(), "entelecheia-python-resolution-"));
  const projectRoot = path.join(temporaryRoot, "project");
  const projectPython = path.join(projectRoot, ".venv", "bin", "python");
  const basePrefix = path.join(projectRoot, ".runtime", "python");
  const purelib = path.join(projectRoot, ".venv", "lib", "python3.12", "site-packages");
  fs.mkdirSync(path.dirname(projectPython), { recursive: true });
  fs.mkdirSync(basePrefix, { recursive: true });
  fs.mkdirSync(purelib, { recursive: true });
  fs.writeFileSync(projectPython, "project runtime probe", { mode: 0o755 });

  const sidecarPath = require.resolve("./api-sidecar");
  delete require.cache[sidecarPath];
  const { resolvePython } = require("./api-sidecar");
  t.after(() => {
    delete require.cache[sidecarPath];
    fs.rmSync(temporaryRoot, { recursive: true, force: true });
  });

  assert.equal(resolvePython(path.join(temporaryRoot, "missing-project")), null);
  assert.equal(
    resolvePython(projectRoot, () => ({ base_prefix: basePrefix, purelib })),
    projectPython
  );
  assert.equal(
    resolvePython(projectRoot, () => ({
      base_prefix: path.join(temporaryRoot, "external-python"),
      purelib
    })),
    null
  );
});

function identityProof(apiToken, challenge) {
  return crypto
    .createHmac("sha256", apiToken)
    .update(API_IDENTITY_CONTEXT, "utf8")
    .update(challenge, "ascii")
    .digest("base64url");
}

function mockHttpGet(
  expectedToken,
  protectedAuthorizations,
  { proveOwnership = true } = {}
) {
  return (url, options, onResponse) => {
    const request = new EventEmitter();
    request.destroy = () => {};
    queueMicrotask(() => {
      const target = new URL(String(url));
      let statusCode = 404;
      let payload = { error: { code: "NOT_FOUND" } };
      if (target.pathname === "/api/health") {
        statusCode = 200;
        payload = { ok: true, service: "personagraph-api" };
        const challenge = target.searchParams.get("challenge");
        if (challenge && proveOwnership) {
          payload.identity = {
            scheme: "hmac-sha256-v1",
            proof: identityProof(expectedToken, challenge)
          };
        } else if (challenge) {
          payload.identity = {
            scheme: "hmac-sha256-v1",
            proof: "A".repeat(43)
          };
        }
      } else if (target.pathname === "/api/config") {
        const authorization = options?.headers?.Authorization || null;
        protectedAuthorizations.push(authorization);
        if (authorization === `Bearer ${expectedToken}`) {
          statusCode = 200;
          payload = { config: {}, model_tiers: {}, model_configured: false };
        } else {
          statusCode = 401;
          payload = { error: { code: "API_AUTH_REQUIRED" } };
        }
      }

      const response = new EventEmitter();
      response.statusCode = statusCode;
      response.headers = { "content-type": "application/json; charset=utf-8" };
      response.setEncoding = () => {};
      response.resume = () => queueMicrotask(() => response.emit("end"));
      onResponse(response);
      const body = JSON.stringify(payload);
      response.emit("data", body);
      response.emit("end");
    });
    return request;
  };
}

test("reuses an existing API only after its Bearer token is accepted", async (t) => {
  const expectedToken = "accepted-token-".padEnd(48, "x");
  const protectedAuthorizations = [];
  const originalHttpGet = http.get;
  http.get = mockHttpGet(expectedToken, protectedAuthorizations);
  const temporaryRoot = fs.mkdtempSync(path.join(os.tmpdir(), "entelecheia-sidecar-auth-"));
  const tokenPath = path.join(temporaryRoot, "api_secret");

  const sidecarPath = require.resolve("./api-sidecar");
  delete require.cache[sidecarPath];
  const { startApiSidecar } = require("./api-sidecar");
  const storage = {
    stateDir: path.join(temporaryRoot, "state"),
    configDir: path.join(temporaryRoot, "config"),
    apiTokenPath: tokenPath
  };

  t.after(async () => {
    delete require.cache[sidecarPath];
    http.get = originalHttpGet;
    fs.rmSync(temporaryRoot, { recursive: true, force: true });
  });

  const missing = await startApiSidecar({ storage });
  assert.equal(missing.ok, false);
  assert.equal(missing.mode, "existing-incompatible");
  assert.equal(missing.apiToken, null);
  assert.equal(missing.error, "existing_api_token_unavailable_or_rejected");
  assert.equal(protectedAuthorizations.length, 0);

  fs.writeFileSync(tokenPath, "wrong-token".padEnd(48, "y"), { mode: 0o600 });
  const mismatched = await startApiSidecar({ storage });
  assert.equal(mismatched.ok, false);
  assert.equal(mismatched.mode, "existing-incompatible");
  assert.equal(mismatched.apiToken, null);
  assert.equal(mismatched.error, "existing_api_token_unavailable_or_rejected");
  assert.deepEqual(protectedAuthorizations, []);

  fs.writeFileSync(tokenPath, expectedToken, { mode: 0o600 });
  const accepted = await startApiSidecar({ storage });
  assert.equal(accepted.ok, true);
  assert.equal(accepted.mode, "existing");
  assert.equal(accepted.apiToken, expectedToken);
  assert.equal(protectedAuthorizations.at(-1), `Bearer ${expectedToken}`);
});

test("never discloses the Bearer token to a service that only forges health", async (t) => {
  const localToken = "private-local-token-".padEnd(48, "z");
  const protectedAuthorizations = [];
  const originalHttpGet = http.get;
  http.get = mockHttpGet(localToken, protectedAuthorizations, {
    proveOwnership: false
  });
  const temporaryRoot = fs.mkdtempSync(path.join(os.tmpdir(), "entelecheia-sidecar-forgery-"));
  const tokenPath = path.join(temporaryRoot, "api_secret");
  fs.writeFileSync(tokenPath, localToken, { mode: 0o600 });

  const sidecarPath = require.resolve("./api-sidecar");
  delete require.cache[sidecarPath];
  const { startApiSidecar } = require("./api-sidecar");
  t.after(() => {
    delete require.cache[sidecarPath];
    http.get = originalHttpGet;
    fs.rmSync(temporaryRoot, { recursive: true, force: true });
  });

  const result = await startApiSidecar({
    storage: {
      stateDir: path.join(temporaryRoot, "state"),
      configDir: path.join(temporaryRoot, "config"),
      apiTokenPath: tokenPath
    }
  });

  assert.equal(result.ok, false);
  assert.equal(result.mode, "existing-incompatible");
  assert.equal(result.apiToken, null);
  assert.deepEqual(protectedAuthorizations, []);
});

test("rejects direct sidecar storage inside the source checkout", async () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "entelecheia-source-root-"));
  const sidecarPath = require.resolve("./api-sidecar");
  delete require.cache[sidecarPath];
  const { startApiSidecar } = require("./api-sidecar");

  try {
    await assert.rejects(
      () => startApiSidecar({
        root,
        storage: {
          stateDir: path.join(root, "var"),
          configDir: path.join(root, "private-config"),
          apiTokenPath: path.join(root, "var", "api_secret")
        }
      }),
      /stateDir must be outside the source checkout/
    );
  } finally {
    delete require.cache[sidecarPath];
    fs.rmSync(root, { recursive: true, force: true });
  }
});
