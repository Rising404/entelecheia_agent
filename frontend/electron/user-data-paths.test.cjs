const assert = require("node:assert/strict");
const path = require("node:path");
const test = require("node:test");

const { resolveSidecarStorage } = require("./user-data-paths");

test("places macOS sidecar state under the Entelecheia Electron userData root", () => {
  const userDataPath = path.join(path.sep, "users", "example", "Entelecheia");

  assert.deepEqual(resolveSidecarStorage({
    userDataPath,
    env: {},
    platform: "darwin",
    homeDirectory: path.join(path.sep, "users", "example")
  }), {
    stateDir: path.join(userDataPath, "state"),
    configDir: path.join(userDataPath, "config"),
    apiTokenPath: path.join(userDataPath, "state", "api_secret")
  });
});

test("matches the Python Windows state and config roots", () => {
  const local = path.join(path.sep, "windows", "local");
  const roaming = path.join(path.sep, "windows", "roaming");

  assert.deepEqual(resolveSidecarStorage({
    userDataPath: path.join(roaming, "Entelecheia"),
    env: { LOCALAPPDATA: local, APPDATA: roaming },
    platform: "win32",
    homeDirectory: path.join(path.sep, "windows", "home")
  }), {
    stateDir: path.join(local, "Entelecheia", "state"),
    configDir: path.join(roaming, "Entelecheia", "config"),
    apiTokenPath: path.join(local, "Entelecheia", "state", "api_secret")
  });
});

test("matches the Python XDG state and config roots", () => {
  const data = path.join(path.sep, "xdg", "data");
  const config = path.join(path.sep, "xdg", "config");

  assert.deepEqual(resolveSidecarStorage({
    userDataPath: path.join(config, "Entelecheia"),
    env: { XDG_DATA_HOME: data, XDG_CONFIG_HOME: config },
    platform: "linux",
    homeDirectory: path.join(path.sep, "users", "example")
  }), {
    stateDir: path.join(data, "Entelecheia", "state"),
    configDir: path.join(config, "Entelecheia", "config"),
    apiTokenPath: path.join(data, "Entelecheia", "state", "api_secret")
  });
});

test("honors explicit sidecar storage overrides", () => {
  const storage = resolveSidecarStorage({
    userDataPath: path.join(path.sep, "ignored", "Entelecheia"),
    platform: "darwin",
    homeDirectory: path.join(path.sep, "users", "example"),
    env: {
      PERSONAGRAPH_STATE_DIR: path.join(path.sep, "custom", "state"),
      PERSONAGRAPH_LOCAL_CONFIG_DIR: path.join(path.sep, "custom", "config"),
      PERSONAGRAPH_API_TOKEN_FILE: path.join(path.sep, "custom", "secret", "token")
    }
  });

  assert.deepEqual(storage, {
    stateDir: path.join(path.sep, "custom", "state"),
    configDir: path.join(path.sep, "custom", "config"),
    apiTokenPath: path.join(path.sep, "custom", "secret", "token")
  });
});

test("derives the token path from an explicit state directory", () => {
  const stateDir = path.join(path.sep, "custom", "state");
  const userDataPath = path.join(path.sep, "users", "example", "Entelecheia");

  assert.deepEqual(
    resolveSidecarStorage({
      userDataPath,
      platform: "darwin",
      homeDirectory: path.join(path.sep, "users", "example"),
      env: { PERSONAGRAPH_STATE_DIR: stateDir }
    }),
    {
      stateDir,
      configDir: path.join(userDataPath, "config"),
      apiTokenPath: path.join(stateDir, "api_secret")
    }
  );
});

test("rejects relative sidecar overrides before spawning Python", () => {
  assert.throws(
    () => resolveSidecarStorage({
      userDataPath: path.join(path.sep, "users", "example", "Entelecheia"),
      env: { PERSONAGRAPH_STATE_DIR: "var" },
      platform: "darwin",
      homeDirectory: path.join(path.sep, "users", "example")
    }),
    /PERSONAGRAPH_STATE_DIR must be an absolute path/
  );
});

test("rejects private sidecar storage inside the source checkout", () => {
  const sourceRoot = path.join(path.sep, "workspace", "entelecheia");

  assert.throws(
    () => resolveSidecarStorage({
      userDataPath: path.join(path.sep, "users", "example", "Entelecheia"),
      env: {
        PERSONAGRAPH_STATE_DIR: path.join(sourceRoot, "var")
      },
      platform: "darwin",
      homeDirectory: path.join(path.sep, "users", "example"),
      sourceRoot
    }),
    /PERSONAGRAPH_STATE_DIR must be outside the source checkout/
  );
});

test("fails closed when neither userData nor complete overrides are available", () => {
  assert.throws(
    () => resolveSidecarStorage({
      env: {},
      platform: "linux",
      homeDirectory: "",
      userDataPath: ""
    }),
    /per-user storage root is required/
  );
});
