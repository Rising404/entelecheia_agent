const RUNTIME_RECONCILE_CHANNEL = "personagraph:runtime-reconcile";

function createRuntimeRecoveryBroadcaster({
  listWindows,
  now = () => new Date(),
  onDeliveryError = () => {}
}) {
  if (typeof listWindows !== "function") {
    throw new TypeError("listWindows must be a function");
  }
  return function broadcastRuntimeRecovery(reason) {
    const payload = {
      schemaVersion: 1,
      reason: String(reason || "runtime-reconcile"),
      occurredAt: now().toISOString()
    };
    let delivered = 0;
    for (const win of listWindows() || []) {
      if (!win || typeof win.isDestroyed !== "function" || win.isDestroyed()) continue;
      if (typeof win.webContents?.send !== "function") continue;
      try {
        win.webContents.send(RUNTIME_RECONCILE_CHANNEL, payload);
        delivered += 1;
      } catch (error) {
        onDeliveryError(error);
      }
    }
    return delivered;
  };
}

function registerPowerResumeRecovery({ powerMonitor, broadcast }) {
  if (typeof powerMonitor?.on !== "function" || typeof powerMonitor?.removeListener !== "function") {
    throw new TypeError("powerMonitor must support on/removeListener");
  }
  if (typeof broadcast !== "function") {
    throw new TypeError("broadcast must be a function");
  }
  const listener = () => broadcast("system-resume");
  let active = true;
  powerMonitor.on("resume", listener);
  return function unregisterPowerResumeRecovery() {
    if (!active) return;
    active = false;
    powerMonitor.removeListener("resume", listener);
  };
}

module.exports = {
  RUNTIME_RECONCILE_CHANNEL,
  createRuntimeRecoveryBroadcaster,
  registerPowerResumeRecovery
};
