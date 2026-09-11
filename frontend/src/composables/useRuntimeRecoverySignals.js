/**
 * 将浏览器/Electron 唤醒信号转成串行的只读运行时协调。重复信号会合并；此处不发送操作 POST。
 */
export function useRuntimeRecoverySignals({
  reconcile,
  onError = () => {},
  target = globalThis,
  desktop = globalThis.personagraphDesktop
}) {
  let started = false;
  let running = false;
  let pending = false;
  let removeDesktopListener = null;

  async function trigger() {
    if (!started) return;
    if (running) {
      pending = true;
      return;
    }
    running = true;
    try {
      do {
        pending = false;
        try {
          await reconcile();
        } catch (error) {
          onError(error);
        }
      } while (started && pending);
    } finally {
      running = false;
    }
  }

  function handleVisibilityChange() {
    if (target.document?.visibilityState !== "hidden") void trigger();
  }

  function start() {
    if (started) return;
    started = true;
    target.addEventListener?.("focus", trigger);
    target.addEventListener?.("online", trigger);
    target.document?.addEventListener?.("visibilitychange", handleVisibilityChange);
    const cleanup = desktop?.onRuntimeReconciliation?.(trigger);
    removeDesktopListener = typeof cleanup === "function" ? cleanup : null;
  }

  function stop() {
    if (!started) return;
    started = false;
    pending = false;
    target.removeEventListener?.("focus", trigger);
    target.removeEventListener?.("online", trigger);
    target.document?.removeEventListener?.("visibilitychange", handleVisibilityChange);
    removeDesktopListener?.();
    removeDesktopListener = null;
  }

  return { start, stop, trigger };
}
