import { computed, unref } from "vue";

/**
 * 协调未保存的功能局部状态，而不把它提升为全局状态。
 *
 * 每个功能保留自己的表单；应用外壳提供脏状态信号和重置操作，
 * 让导航和页面卸载前的行为保持一致。
 */
export function useUnsavedGuard({ dirtySources, busySignals = [], clearMessage, discard, notice }) {
  const dirtyLabels = computed(() => Object.entries(dirtySources)
    .filter(([, signal]) => Boolean(unref(signal)))
    .map(([label]) => label));
  const hasUnsavedChanges = computed(() => dirtyLabels.value.length > 0);
  const unsavedSummary = computed(() => dirtyLabels.value[0] || "");
  const isInteractionBusy = computed(() => busySignals.some((signal) => Boolean(unref(signal))));

  function confirmDiscardChanges() {
    if (!hasUnsavedChanges.value) return true;
    const ok = globalThis.confirm("当前有未保存改动，切换后会丢弃这些草稿。继续？");
    if (ok) {
      discard();
      clearMessage();
    }
    return ok;
  }

  function discardDraftsFromBanner() {
    discard();
    clearMessage();
    notice.value = "已丢弃未保存改动";
  }

  function handleBeforeUnload(event) {
    if (!hasUnsavedChanges.value && !isInteractionBusy.value) return;
    event.preventDefault();
    event.returnValue = "";
  }

  return {
    hasUnsavedChanges,
    unsavedSummary,
    isInteractionBusy,
    confirmDiscardChanges,
    discardDraftsFromBanner,
    handleBeforeUnload
  };
}
