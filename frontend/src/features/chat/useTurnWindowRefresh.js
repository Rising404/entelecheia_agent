import { watch } from "vue";

export const TURN_WINDOW_REFRESH_INTERVAL_MS = 2_000;

/**
 * 在持久提交后 Window 收尾期间刷新当前选中的 Session。
 *
 * 正式回复可以在隐藏的派生状态工作（当前为会话摘要）完成前展示。这个轻量生命周期
 * 所有者让编辑器与存储拥有的 Window 保持一致，不会根据 SSE 或模型事件推断完成状态。
 * 它绝不重试终态失败：这种操作需要独立、明确的用户控制，而不是带来意外副作用的定时器。
 */
export function useTurnWindowRefresh({
  api,
  selectedSessionId,
  sessionDetail,
  shouldRefresh,
  intervalMs = TURN_WINDOW_REFRESH_INTERVAL_MS
}) {
  let timer = null;
  let inFlight = false;
  let generation = 0;

  function clearTimer() {
    if (timer !== null) {
      globalThis.clearTimeout(timer);
      timer = null;
    }
  }

  function eligible() {
    return Boolean(
      shouldRefresh.value
      && selectedSessionId.value
      && typeof api?.getSession === "function"
    );
  }

  function schedule() {
    if (!eligible() || timer !== null || inFlight) return;
    const refreshGeneration = generation;
    timer = globalThis.setTimeout(async () => {
      timer = null;
      if (!eligible() || refreshGeneration !== generation) return;
      inFlight = true;
      const sessionId = selectedSessionId.value;
      try {
        const payload = await api.getSession(sessionId);
        if (
          refreshGeneration === generation
          && selectedSessionId.value === sessionId
        ) {
          sessionDetail.value = payload;
        }
      } catch {
        // 用户已经在等待后台任务时，短暂刷新错误不能反复弹出 modal/toast。现有 Window
        // 继续作为权威状态，下一次计划读取可以自行恢复。
      } finally {
        inFlight = false;
        if (refreshGeneration === generation) schedule();
      }
    }, intervalMs);
  }

  const stop = watch(
    [() => selectedSessionId.value, () => shouldRefresh.value],
    (_values, _previousValues, onCleanup) => {
      generation += 1;
      clearTimer();
      schedule();
      // 所属 Vue 作用域销毁时，watcher 清理无需依赖组件专用生命周期钩子即可运行，
      // 同时也让该 composable 能在未挂载组件的情况下直接测试。
      onCleanup(() => {
        generation += 1;
        clearTimer();
      });
    },
    { immediate: true }
  );

  return { refreshIfPending: schedule, stop };
}
