const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");

const BUNDLE_ID = "com.personagraph.entelecheia.launcher";
const APP_NAME = "Entelecheia";

function defaultOutputPath() {
  return path.join(os.homedir(), "Applications", `${APP_NAME}.app`);
}

function shellQuote(value) {
  return `'${String(value).replaceAll("'", `'"'"'`)}'`;
}

function renderInfoPlist() {
  return `<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleDevelopmentRegion</key><string>zh_CN</string>
  <key>CFBundleDisplayName</key><string>${APP_NAME}</string>
  <key>CFBundleExecutable</key><string>EntelecheiaLauncher</string>
  <key>CFBundleIdentifier</key><string>${BUNDLE_ID}</string>
  <key>CFBundleName</key><string>${APP_NAME}</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleShortVersionString</key><string>0.1.0</string>
  <key>CFBundleVersion</key><string>1</string>
  <key>LSMinimumSystemVersion</key><string>12.0</string>
  <key>NSHighResolutionCapable</key><true/>
</dict>
</plist>
`;
}

function renderLauncher(frontendRoot) {
  const quotedRoot = shellQuote(path.resolve(frontendRoot));
  return `#!/bin/zsh
set -u
setopt NULL_GLOB
umask 077

FRONTEND_ROOT=${quotedRoot}
LOG_DIR="$HOME/Library/Logs/Entelecheia"
LOG_FILE="$LOG_DIR/launcher.log"
/bin/mkdir -p "$LOG_DIR"
exec >>"$LOG_FILE" 2>&1

fail_launch() {
  echo "[$(/bin/date -u +%Y-%m-%dT%H:%M:%SZ)] $1"
  /usr/bin/osascript -e 'display alert "Entelecheia 无法启动" message "请查看日志：~/Library/Logs/Entelecheia/launcher.log" as critical' >/dev/null 2>&1 || true
  exit 1
}

[ -d "$FRONTEND_ROOT" ] || fail_launch "前端目录不存在：$FRONTEND_ROOT"
REPO_ROOT="$(cd "$FRONTEND_ROOT/.." && pwd)"
NODE_BIN="$REPO_ROOT/.runtime/node/bin"
[ -x "$NODE_BIN/node" ] || fail_launch "项目 Node 不存在；请运行 scripts/bootstrap-local-runtime.sh"
NODE="$NODE_BIN/node"
VITE="$FRONTEND_ROOT/node_modules/vite/bin/vite.js"
ELECTRON="$FRONTEND_ROOT/node_modules/electron/dist/Electron.app/Contents/MacOS/Electron"
[ -f "$VITE" ] || fail_launch "前端依赖未安装：缺少 Vite"
[ -x "$ELECTRON" ] || fail_launch "前端依赖未安装：缺少 Electron"

echo "[$(/bin/date -u +%Y-%m-%dT%H:%M:%SZ)] build and launch $FRONTEND_ROOT"
cd "$FRONTEND_ROOT" || fail_launch "无法进入前端目录"
CI=true "$NODE" "$VITE" build || fail_launch "前端构建失败"
exec "$ELECTRON" "$FRONTEND_ROOT"
`;
}

function requireLauncherPrerequisites(frontendRoot) {
  const required = [
    path.join(frontendRoot, "package.json"),
    path.join(frontendRoot, "node_modules", "vite", "bin", "vite.js"),
    path.join(
      frontendRoot,
      "node_modules",
      "electron",
      "dist",
      "Electron.app",
      "Contents",
      "MacOS",
      "Electron"
    )
  ];
  const missing = required.filter((candidate) => !fs.existsSync(candidate));
  if (missing.length) {
    throw new Error(`launcher prerequisites are missing: ${missing.join(", ")}`);
  }
}

function assertSafeReplacement(outputPath) {
  if (!fs.existsSync(outputPath)) return;
  const plistPath = path.join(outputPath, "Contents", "Info.plist");
  let plist = "";
  try {
    plist = fs.readFileSync(plistPath, "utf8");
  } catch {
    throw new Error(`refusing to replace an unknown application: ${outputPath}`);
  }
  if (!plist.includes(`<string>${BUNDLE_ID}</string>`)) {
    throw new Error(`refusing to replace an unknown application: ${outputPath}`);
  }
}

function installMacosLauncher({
  frontendRoot = path.resolve(__dirname, ".."),
  outputPath = defaultOutputPath()
} = {}) {
  const resolvedRoot = path.resolve(frontendRoot);
  const resolvedOutput = path.resolve(outputPath);
  if (!resolvedOutput.endsWith(".app")) {
    throw new Error("macOS launcher output must end with .app");
  }
  requireLauncherPrerequisites(resolvedRoot);
  assertSafeReplacement(resolvedOutput);

  const parent = path.dirname(resolvedOutput);
  const stage = path.join(parent, `.${path.basename(resolvedOutput)}.tmp-${process.pid}`);
  fs.mkdirSync(parent, { recursive: true, mode: 0o755 });
  fs.rmSync(stage, { recursive: true, force: true });
  fs.mkdirSync(path.join(stage, "Contents", "MacOS"), { recursive: true, mode: 0o755 });
  fs.mkdirSync(path.join(stage, "Contents", "Resources"), { recursive: true, mode: 0o755 });
  fs.writeFileSync(
    path.join(stage, "Contents", "Info.plist"),
    renderInfoPlist(),
    { mode: 0o644 }
  );
  fs.writeFileSync(
    path.join(stage, "Contents", "MacOS", "EntelecheiaLauncher"),
    renderLauncher(resolvedRoot),
    { mode: 0o755 }
  );
  fs.writeFileSync(
    path.join(stage, "Contents", "Resources", "source-root.txt"),
    `${resolvedRoot}\n`,
    { mode: 0o644 }
  );

  if (fs.existsSync(resolvedOutput)) {
    fs.rmSync(resolvedOutput, { recursive: true, force: true });
  }
  fs.renameSync(stage, resolvedOutput);
  return { outputPath: resolvedOutput, frontendRoot: resolvedRoot };
}

if (require.main === module) {
  try {
    const outputArgument = process.argv[2];
    const result = installMacosLauncher({
      outputPath: outputArgument || defaultOutputPath()
    });
    process.stdout.write(`${result.outputPath}\n`);
  } catch (error) {
    process.stderr.write(`${error.message}\n`);
    process.exitCode = 1;
  }
}

module.exports = {
  APP_NAME,
  BUNDLE_ID,
  defaultOutputPath,
  installMacosLauncher,
  renderInfoPlist,
  renderLauncher,
  shellQuote
};
