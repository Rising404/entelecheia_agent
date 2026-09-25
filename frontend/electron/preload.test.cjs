const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

function windowArguments(apiInfo) {
  let options;
  const electron = {
    app: { setName() {}, requestSingleInstanceLock: () => false, quit() {}, on() {} },
    BrowserWindow: class {
      constructor(value) { options = value; }
      webContents = {
        session: { webRequest: { onBeforeSendHeaders() {} } },
        setWindowOpenHandler() {}
      };
      once() {}
      loadFile() {}
    }
  };
  const context = vm.createContext({
    __dirname,
    process: { env: {}, platform: "darwin" },
    require(id) {
      if (id === "electron") return electron;
      if (id.startsWith("node:")) return require(id);
      if (id === "./runtime-recovery") {
        return { createRuntimeRecoveryBroadcaster: () => () => {} };
      }
      return {};
    }
  });
  vm.runInContext(fs.readFileSync(path.join(__dirname, "main.js"), "utf8"), context);
  context.createMainWindow(apiInfo);
  return options.webPreferences.additionalArguments;
}

function exposedDesktop(argv) {
  let desktop;
  vm.runInNewContext(fs.readFileSync(path.join(__dirname, "preload.js"), "utf8"), {
    process: { argv, platform: "darwin", versions: { electron: "test" } },
    require(id) {
      assert.equal(id, "electron");
      return {
        contextBridge: {
          exposeInMainWorld(name, value) {
            assert.equal(name, "personagraphDesktop");
            desktop = value;
          }
        },
        ipcRenderer: {}
      };
    }
  });
  return desktop;
}

for (const port of [8765, 8766]) {
  test(`renderer uses the main-process sidecar address on port ${port} without its token`, () => {
    const apiBase = `http://127.0.0.1:${port}`;
    const token = "synthetic-main-process-only-token";
    const args = windowArguments({ apiBase, apiToken: token, mode: "spawned", ok: true });
    const desktop = exposedDesktop(args);

    assert.equal(desktop.apiBase, apiBase);
    assert.equal(desktop.sidecar.apiBase, apiBase);
    assert.equal(desktop.sidecar.ok, true);
    assert.equal(Object.hasOwn(desktop, "apiToken"), false);
    assert.equal(Object.hasOwn(desktop.sidecar, "apiToken"), false);
    assert.equal(JSON.stringify(args).includes(token), false);
  });
}

test("preload keeps the default address when no sidecar argument is available", () => {
  const desktop = exposedDesktop([]);
  assert.equal(desktop.apiBase, "http://127.0.0.1:8765");
  assert.equal(desktop.sidecar, null);
});
