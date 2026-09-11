const {
  app,
  BrowserWindow,
  Menu,
  dialog,
  globalShortcut,
  ipcMain,
  powerMonitor,
  shell
} = require("electron");
const { spawn } = require("node:child_process");
const path = require("node:path");
const { startApiSidecar, stopApiSidecar } = require("./api-sidecar");
const { installMacosLauncher } = require("./macos-launcher");
const { resolveSidecarStorage } = require("./user-data-paths");
const {
  DEFAULT_WAKE_ACCELERATOR,
  registerWakeShortcut,
  wakeMainWindow
} = require("./window-wakeup");
const {
  createRuntimeRecoveryBroadcaster,
  registerPowerResumeRecovery
} = require("./runtime-recovery");

app.setName("Entelecheia");

const isDev = process.env.NODE_ENV === "development";
const rendererUrl = process.env.PERSONAGRAPH_RENDERER_URL || "";
let runtimeApiInfo = {};
let unregisterRuntimeRecovery = () => {};
let unregisterWakeShortcut = () => {};
const requestRuntimeReconciliation = createRuntimeRecoveryBroadcaster({
  listWindows: () => BrowserWindow.getAllWindows(),
  onDeliveryError: (error) => {
    console.error(`[runtime-recovery] failed to notify renderer: ${error.message}`);
  }
});

function createMainWindow(apiInfo = {}) {
  const { apiToken: _apiToken, ...publicApiInfo } = apiInfo;
  const win = new BrowserWindow({
    width: 1280,
    height: 820,
    minWidth: 960,
    minHeight: 640,
    title: "Entelecheia（隐得莱希）",
    backgroundColor: "#f7f8f5",
    show: false,
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      additionalArguments: [
        `--personagraph-sidecar=${encodeURIComponent(JSON.stringify(publicApiInfo))}`
      ],
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true
    }
  });

  if (apiInfo.apiBase) {
    win.webContents.session.webRequest.onBeforeSendHeaders(
      { urls: [`${apiInfo.apiBase}/api/*`] },
      (details, callback) => {
        const currentToken = runtimeApiInfo.apiToken;
        if (currentToken) {
          details.requestHeaders.Authorization = `Bearer ${currentToken}`;
        }
        callback({ requestHeaders: details.requestHeaders });
      }
    );
  }

  win.once("ready-to-show", () => {
    win.show();
  });

  win.webContents.setWindowOpenHandler(({ url }) => {
    if (url.startsWith("http://127.0.0.1") || url.startsWith("http://localhost")) {
      return { action: "allow" };
    }
    shell.openExternal(url);
    return { action: "deny" };
  });

  if (rendererUrl) {
    win.loadURL(rendererUrl);
  } else {
    win.loadFile(path.join(__dirname, "..", "dist", "index.html"));
  }

  if (isDev) {
    win.webContents.openDevTools({ mode: "detach" });
  }
  return win;
}

function wakeApplication() {
  return wakeMainWindow({
    listWindows: () => BrowserWindow.getAllWindows(),
    createWindow: () => createMainWindow(runtimeApiInfo)
  });
}

async function installMacosLauncherFromMenu() {
  try {
    const result = installMacosLauncher();
    await dialog.showMessageBox({
      type: "info",
      title: "一键启动器已就绪",
      message: "Entelecheia 已安装到你的 Applications 文件夹。",
      detail: `${result.outputPath}\n以后可以直接双击或拖到 Dock，不会打开 Terminal。`
    });
    shell.showItemInFolder(result.outputPath);
  } catch (error) {
    dialog.showErrorBox(
      "无法安装一键启动器",
      `${error.message}\n请确认前端依赖已经安装。`
    );
  }
}

function installWindowsLauncherFromMenu() {
  const installerPath = path.resolve(__dirname, "..", "install-windows-shortcut.vbs");
  const windowsRoot = process.env.SystemRoot || process.env.WINDIR || "C:\\Windows";
  const wscriptPath = path.join(windowsRoot, "System32", "wscript.exe");
  try {
    const installer = spawn(wscriptPath, [installerPath], {
      detached: true,
      stdio: "ignore",
      windowsHide: true
    });
    installer.once("error", (error) => {
      dialog.showErrorBox("无法安装桌面快捷方式", error.message);
    });
    installer.unref();
  } catch (error) {
    dialog.showErrorBox("无法安装桌面快捷方式", error.message);
  }
}

function buildMenu() {
  const template = [
    {
      label: "Entelecheia",
      submenu: [
        { role: "about" },
        { type: "separator" },
        {
          label: "显示主窗口",
          accelerator: DEFAULT_WAKE_ACCELERATOR,
          click: wakeApplication
        },
        ...(process.platform === "darwin" ? [
          {
            label: "安装/更新一键启动器…",
            click: installMacosLauncherFromMenu
          }
        ] : []),
        ...(process.platform === "win32" ? [
          {
            label: "安装/更新桌面快捷方式…",
            click: installWindowsLauncherFromMenu
          }
        ] : []),
        { type: "separator" },
        { role: "hide" },
        { role: "hideOthers" },
        { role: "unhide" },
        { type: "separator" },
        { role: "quit" }
      ]
    },
    {
      label: "Edit",
      submenu: [
        { role: "undo" },
        { role: "redo" },
        { type: "separator" },
        { role: "cut" },
        { role: "copy" },
        { role: "paste" },
        { role: "selectAll" }
      ]
    },
    {
      label: "View",
      submenu: [
        { role: "reload" },
        { role: "forceReload" },
        { role: "toggleDevTools" },
        { type: "separator" },
        { role: "resetZoom" },
        { role: "zoomIn" },
        { role: "zoomOut" },
        { type: "separator" },
        { role: "togglefullscreen" }
      ]
    },
    {
      label: "Window",
      submenu: [
        { role: "minimize" },
        { role: "zoom" },
        { role: "front" }
      ]
    }
  ];
  Menu.setApplicationMenu(Menu.buildFromTemplate(template));
}

const hasSingleInstanceLock = app.requestSingleInstanceLock();
if (!hasSingleInstanceLock) {
  app.quit();
} else {
  app.on("second-instance", () => {
    if (app.isReady()) wakeApplication();
    else app.once("ready", wakeApplication);
  });

  app.whenReady().then(async () => {
    const sidecarOptions = {
      storage: resolveSidecarStorage({
        userDataPath: app.getPath("userData"),
        env: process.env,
        sourceRoot: path.resolve(__dirname, "..", "..")
      })
    };
    buildMenu();
    ipcMain.handle("personagraph:choose-document", async () => {
      const result = await dialog.showOpenDialog({
        title: "选择要收录的文档",
        properties: ["openFile"],
        filters: [
          { name: "Documents", extensions: ["md", "txt", "pdf", "docx"] },
          { name: "All Files", extensions: ["*"] }
        ]
      });
      return { path: result.canceled ? "" : result.filePaths[0] || "" };
    });
    ipcMain.handle("personagraph:choose-directory", async () => {
      const result = await dialog.showOpenDialog({
        title: "选择工作目录",
        properties: ["openDirectory", "createDirectory"]
      });
      return { path: result.canceled ? "" : result.filePaths[0] || "" };
    });
    ipcMain.handle("personagraph:ensure-api-sidecar", async () => {
      runtimeApiInfo = await startApiSidecar(sidecarOptions);
      if (runtimeApiInfo.ok) {
        requestRuntimeReconciliation("manual-api-wake");
      }
      const { apiToken, ...publicApiInfo } = runtimeApiInfo;
      return publicApiInfo;
    });

    runtimeApiInfo = await startApiSidecar(sidecarOptions);
    if (!runtimeApiInfo.ok) {
      console.error("[personagraph-api] failed to become healthy; renderer will use mock fallback");
    }
    createMainWindow(runtimeApiInfo);
    const wakeRegistration = registerWakeShortcut({
      globalShortcut,
      wake: wakeApplication
    });
    unregisterWakeShortcut = wakeRegistration.dispose;
    if (!wakeRegistration.registered) {
      console.warn(`[window-wakeup] shortcut unavailable: ${wakeRegistration.accelerator}`);
    }

    // 唤醒恢复被刻意设计为只读协调信号。渲染进程会追赶日志和待处理评审；
    // 它绝不会重放最初启动或恢复运行的 POST 请求。
    unregisterRuntimeRecovery = registerPowerResumeRecovery({
      powerMonitor,
      broadcast: requestRuntimeReconciliation
    });

    app.on("activate", () => {
      wakeApplication();
    });
  });
}

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") {
    app.quit();
  }
});

app.on("before-quit", () => {
  unregisterWakeShortcut();
  unregisterRuntimeRecovery();
  stopApiSidecar();
});
