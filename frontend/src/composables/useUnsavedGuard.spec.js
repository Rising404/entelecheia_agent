import { describe, expect, it, vi } from "vitest";
import { ref } from "vue";
import { useUnsavedGuard } from "./useUnsavedGuard";

describe("useUnsavedGuard", () => {
  it("summarizes dirty sources and discards only after confirmation", () => {
    const sessionTitleDirty = ref(true);
    const taskDirty = ref(true);
    const notice = ref("");
    const clearMessage = vi.fn();
    const discard = vi.fn();
    const confirm = vi.spyOn(globalThis, "confirm").mockReturnValue(false);
    const guard = useUnsavedGuard({
      dirtySources: { "会话名称": sessionTitleDirty, "任务详情": taskDirty },
      clearMessage,
      discard,
      notice
    });

    expect(guard.hasUnsavedChanges.value).toBe(true);
    expect(guard.unsavedSummary.value).toBe("会话名称");
    expect(guard.confirmDiscardChanges()).toBe(false);
    expect(discard).not.toHaveBeenCalled();

    confirm.mockReturnValue(true);
    expect(guard.confirmDiscardChanges()).toBe(true);
    expect(discard).toHaveBeenCalledTimes(1);
    expect(clearMessage).toHaveBeenCalledTimes(1);

    guard.discardDraftsFromBanner();
    expect(notice.value).toBe("已丢弃未保存改动");
    confirm.mockRestore();
  });

  it("protects before unload for dirty or busy state", () => {
    const busy = ref(false);
    const dirty = ref(false);
    const guard = useUnsavedGuard({
      dirtySources: { "文档信息": dirty },
      busySignals: [busy],
      clearMessage: vi.fn(),
      discard: vi.fn(),
      notice: ref("")
    });
    const event = { preventDefault: vi.fn(), returnValue: undefined };

    guard.handleBeforeUnload(event);
    expect(event.preventDefault).not.toHaveBeenCalled();

    busy.value = true;
    guard.handleBeforeUnload(event);
    expect(event.preventDefault).toHaveBeenCalledTimes(1);
    expect(event.returnValue).toBe("");
  });
});
