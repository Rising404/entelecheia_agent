const test = require("node:test");
const assert = require("node:assert/strict");

const {
  DEFAULT_WAKE_ACCELERATOR,
  registerWakeShortcut,
  wakeMainWindow
} = require("./window-wakeup");

test("wake restores, shows, and focuses the existing window", () => {
  const calls = [];
  const win = {
    isDestroyed: () => false,
    isMinimized: () => true,
    restore: () => calls.push("restore"),
    show: () => calls.push("show"),
    focus: () => calls.push("focus")
  };

  const selected = wakeMainWindow({
    listWindows: () => [win],
    createWindow: () => assert.fail("must not create another window")
  });

  assert.equal(selected, win);
  assert.deepEqual(calls, ["restore", "show", "focus"]);
});

test("wake creates a window when every old window is gone", () => {
  const created = { id: "new" };
  const selected = wakeMainWindow({
    listWindows: () => [{ isDestroyed: () => true }],
    createWindow: () => created
  });
  assert.equal(selected, created);
});

test("global wake shortcut registers and disposes only its own accelerator", () => {
  const calls = [];
  const shortcut = {
    register: (accelerator, callback) => {
      calls.push(["register", accelerator]);
      callback();
      return true;
    },
    unregister: (accelerator) => calls.push(["unregister", accelerator])
  };
  let wakes = 0;

  const registration = registerWakeShortcut({
    globalShortcut: shortcut,
    wake: () => { wakes += 1; }
  });
  registration.dispose();

  assert.equal(registration.accelerator, DEFAULT_WAKE_ACCELERATOR);
  assert.equal(registration.registered, true);
  assert.equal(wakes, 1);
  assert.deepEqual(calls, [
    ["register", DEFAULT_WAKE_ACCELERATOR],
    ["unregister", DEFAULT_WAKE_ACCELERATOR]
  ]);
});
