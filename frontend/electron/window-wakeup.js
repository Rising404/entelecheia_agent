const DEFAULT_WAKE_ACCELERATOR = "CommandOrControl+Shift+Space";

function firstUsableWindow(windows) {
  return (windows || []).find((win) => (
    win && typeof win.isDestroyed === "function" && !win.isDestroyed()
  )) || null;
}

function wakeMainWindow({ listWindows, createWindow }) {
  if (typeof listWindows !== "function" || typeof createWindow !== "function") {
    throw new TypeError("wakeMainWindow requires listWindows and createWindow");
  }

  const existing = firstUsableWindow(listWindows());
  if (!existing) {
    return createWindow();
  }
  if (typeof existing.isMinimized === "function" && existing.isMinimized()) {
    existing.restore();
  }
  if (typeof existing.show === "function") existing.show();
  if (typeof existing.focus === "function") existing.focus();
  return existing;
}

function registerWakeShortcut({
  globalShortcut,
  wake,
  accelerator = DEFAULT_WAKE_ACCELERATOR
}) {
  if (typeof globalShortcut?.register !== "function" || typeof wake !== "function") {
    throw new TypeError("registerWakeShortcut requires globalShortcut and wake");
  }
  const registered = globalShortcut.register(accelerator, wake);
  return {
    accelerator,
    registered,
    dispose() {
      if (registered && typeof globalShortcut.unregister === "function") {
        globalShortcut.unregister(accelerator);
      }
    }
  };
}

module.exports = {
  DEFAULT_WAKE_ACCELERATOR,
  firstUsableWindow,
  registerWakeShortcut,
  wakeMainWindow
};
