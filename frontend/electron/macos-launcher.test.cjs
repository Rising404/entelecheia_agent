const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");

const {
  BUNDLE_ID,
  installMacosLauncher,
  renderLauncher
} = require("./macos-launcher");

function fakeFrontendRoot(base) {
  const root = path.join(base, "checkout with spaces", "frontend");
  const files = [
    "package.json",
    "node_modules/vite/bin/vite.js",
    "node_modules/electron/dist/Electron.app/Contents/MacOS/Electron"
  ];
  for (const relative of files) {
    const target = path.join(root, relative);
    fs.mkdirSync(path.dirname(target), { recursive: true });
    fs.writeFileSync(target, relative === "package.json" ? "{}" : "stub");
  }
  fs.chmodSync(
    path.join(root, "node_modules/electron/dist/Electron.app/Contents/MacOS/Electron"),
    0o755
  );
  return root;
}

test("installer creates a terminal-free app wrapper for paths with spaces", () => {
  const temp = fs.mkdtempSync(path.join(os.tmpdir(), "entelecheia-launcher-"));
  const frontendRoot = fakeFrontendRoot(temp);
  const outputPath = path.join(temp, "Applications", "Entelecheia.app");

  const result = installMacosLauncher({ frontendRoot, outputPath });
  const plist = fs.readFileSync(path.join(outputPath, "Contents/Info.plist"), "utf8");
  const launcherPath = path.join(outputPath, "Contents/MacOS/EntelecheiaLauncher");
  const launcher = fs.readFileSync(launcherPath, "utf8");

  assert.equal(result.outputPath, outputPath);
  assert.match(plist, new RegExp(BUNDLE_ID.replaceAll(".", "\\.")));
  assert.match(launcher, /FRONTEND_ROOT='.*checkout with spaces.*'/);
  assert.doesNotMatch(launcher, /Terminal\.app|open -a Terminal/);
  assert.equal(fs.statSync(launcherPath).mode & 0o111, 0o111);
});

test("installer refuses to overwrite an unrelated app bundle", () => {
  const temp = fs.mkdtempSync(path.join(os.tmpdir(), "entelecheia-launcher-"));
  const frontendRoot = fakeFrontendRoot(temp);
  const outputPath = path.join(temp, "Applications", "Entelecheia.app");
  fs.mkdirSync(path.join(outputPath, "Contents"), { recursive: true });
  fs.writeFileSync(path.join(outputPath, "Contents/Info.plist"), "unrelated");

  assert.throws(
    () => installMacosLauncher({ frontendRoot, outputPath }),
    /refusing to replace an unknown application/
  );
});

test("launcher sends failures to a native alert and a private log", () => {
  const launcher = renderLauncher("/tmp/frontend");
  assert.match(launcher, /Library\/Logs\/Entelecheia/);
  assert.match(launcher, /osascript -e 'display alert/);
  assert.match(launcher, /umask 077/);
  assert.match(launcher, /\.runtime\/node\/bin/);
  assert.doesNotMatch(launcher, /codex-runtimes|CODEX_MCP_NODE_PATH/);
});
