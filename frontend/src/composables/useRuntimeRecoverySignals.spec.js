import { describe, expect, it, vi } from "vitest";

import { useRuntimeRecoverySignals } from "./useRuntimeRecoverySignals";

async function settle() {
  await Promise.resolve();
  await Promise.resolve();
}

describe("useRuntimeRecoverySignals", () => {
  it("reconciles on Electron resume and removes the listener on stop", async () => {
    const target = new EventTarget();
    target.document = new EventTarget();
    target.document.visibilityState = "visible";
    let desktopListener = null;
    const removeDesktopListener = vi.fn();
    const desktop = {
      onRuntimeReconciliation: vi.fn((listener) => {
        desktopListener = listener;
        return removeDesktopListener;
      })
    };
    const reconcile = vi.fn().mockResolvedValue(undefined);
    const signals = useRuntimeRecoverySignals({ reconcile, target, desktop });

    signals.start();
    desktopListener({ reason: "system-resume" });
    await settle();

    expect(reconcile).toHaveBeenCalledOnce();
    signals.stop();
    expect(removeDesktopListener).toHaveBeenCalledOnce();
  });

  it("uses only visible/focus recovery signals and reports read failures", async () => {
    const target = new EventTarget();
    target.document = new EventTarget();
    target.document.visibilityState = "hidden";
    const error = new Error("offline");
    const onError = vi.fn();
    const reconcile = vi.fn().mockRejectedValue(error);
    const signals = useRuntimeRecoverySignals({ reconcile, onError, target, desktop: null });

    signals.start();
    target.document.dispatchEvent(new Event("visibilitychange"));
    await settle();
    expect(reconcile).not.toHaveBeenCalled();

    target.dispatchEvent(new Event("focus"));
    await settle();
    expect(reconcile).toHaveBeenCalledOnce();
    expect(onError).toHaveBeenCalledWith(error);
    signals.stop();
  });
});
