const assert = require("node:assert/strict");
const test = require("node:test");

const {
  RUNTIME_RECONCILE_CHANNEL,
  createRuntimeRecoveryBroadcaster,
  registerPowerResumeRecovery
} = require("./runtime-recovery");

test("broadcasts only a sanitized reconciliation signal to live windows", () => {
  const delivered = [];
  const errors = [];
  const liveWindow = {
    isDestroyed: () => false,
    webContents: { send: (...args) => delivered.push(args) }
  };
  const throwingWindow = {
    isDestroyed: () => false,
    webContents: { send: () => { throw new Error("window closed during send"); } }
  };
  const secondLiveWindow = {
    isDestroyed: () => false,
    webContents: { send: (...args) => delivered.push(args) }
  };
  const broadcast = createRuntimeRecoveryBroadcaster({
    listWindows: () => [
      liveWindow,
      { isDestroyed: () => true, webContents: { send: assert.fail } },
      throwingWindow,
      secondLiveWindow
    ],
    now: () => new Date("2026-07-15T08:30:00Z"),
    onDeliveryError: (error) => errors.push(error)
  });

  assert.equal(broadcast("system-resume"), 2);
  assert.deepEqual(delivered, [
    [RUNTIME_RECONCILE_CHANNEL, {
      schemaVersion: 1,
      reason: "system-resume",
      occurredAt: "2026-07-15T08:30:00.000Z"
    }],
    [RUNTIME_RECONCILE_CHANNEL, {
      schemaVersion: 1,
      reason: "system-resume",
      occurredAt: "2026-07-15T08:30:00.000Z"
    }]
  ]);
  assert.equal(errors.length, 1);
  assert.equal(errors[0].message, "window closed during send");
  assert.deepEqual(Object.keys(delivered[0][1]).sort(), ["occurredAt", "reason", "schemaVersion"]);
});

test("registers one resume listener and removes it idempotently", () => {
  const listeners = new Map();
  const removals = [];
  const powerMonitor = {
    on: (event, listener) => listeners.set(event, listener),
    removeListener: (event, listener) => removals.push([event, listener])
  };
  const reasons = [];
  const unregister = registerPowerResumeRecovery({
    powerMonitor,
    broadcast: (reason) => reasons.push(reason)
  });

  listeners.get("resume")();
  unregister();
  unregister();

  assert.deepEqual(reasons, ["system-resume"]);
  assert.deepEqual(removals, [["resume", listeners.get("resume")]]);
});
