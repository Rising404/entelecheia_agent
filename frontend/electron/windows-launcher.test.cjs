const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");

const frontendRoot = path.resolve(__dirname, "..");

function read(relativePath) {
  return fs.readFileSync(path.join(frontendRoot, relativePath), "utf8");
}

test("Windows double-click launcher hides the console and invokes only the fixed bootstrap", () => {
  const launcher = read("start-electron-hidden.vbs");

  assert.match(launcher, /WScript\.ScriptFullName/);
  assert.match(launcher, /start-electron\.ps1/);
  assert.match(launcher, /-NoProfile -NonInteractive -WindowStyle Hidden/);
  assert.match(launcher, /-File .*launcherScript/);
  assert.match(launcher, /-HiddenLauncher/);
  assert.match(launcher, /shell\.Run\(command, 0, True\)/);
  assert.doesNotMatch(launcher, /cmd\.exe|WScript\.Arguments|ExecuteGlobal/i);
});

test("Windows installer creates a desktop hotkey shortcut to the hidden launcher", () => {
  const installer = read("install-windows-shortcut.vbs");

  assert.match(installer, /SpecialFolders\("Desktop"\)/);
  assert.match(installer, /Entelecheia\.lnk/);
  assert.match(installer, /%SystemRoot%\\System32\\wscript\.exe/);
  assert.match(installer, /start-electron-hidden\.vbs/);
  assert.match(installer, /shortcut\.Hotkey = "CTRL\+SHIFT\+SPACE"/);
  assert.match(installer, /shortcut\.WorkingDirectory = scriptDir/);
  assert.doesNotMatch(installer, /cmd\.exe|WScript\.Arguments|ExecuteGlobal/i);
});

test("PowerShell bootstrap logs failures and starts the GUI executable directly", () => {
  const bootstrap = read("start-electron.ps1");

  assert.match(bootstrap, /\[switch\]\$HiddenLauncher/);
  assert.match(bootstrap, /LOCALAPPDATA/);
  assert.match(bootstrap, /launcher\.log/);
  assert.match(bootstrap, /\.runtime\\node\\node\.exe/);
  assert.doesNotMatch(bootstrap, /Get-Command|PERSONAGRAPH_NODE_BIN|ProgramFiles/);
  assert.match(bootstrap, /Test-RendererBuildRequired/);
  assert.match(bootstrap, /node_modules\\electron\\dist\\electron\.exe/);
  assert.match(bootstrap, /Start-Process/);
  assert.doesNotMatch(bootstrap, /electron\.cmd|cmd\.exe/i);
});
