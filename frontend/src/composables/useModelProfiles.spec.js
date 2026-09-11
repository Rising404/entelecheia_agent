import { afterEach, describe, expect, it, vi } from "vitest";
import { flushPromises } from "@vue/test-utils";

import { REVEAL_TIMEOUT_MS, useModelProfiles } from "./useModelProfiles";

afterEach(() => { vi.useRealTimers(); vi.restoreAllMocks(); });

function harness(overrides = {}) {
  const api = {
    listModelProfiles: vi.fn().mockResolvedValue({
      profiles: {
        model: { active_id: "mp_1", profiles: [{ id: "mp_1", has_api_key: true }] },
        vision: { active_id: null, profiles: [] }
      }
    }),
    revealModelProfileSecret: vi.fn().mockResolvedValue({ id: "mp_1", api_key: "sk-live" }),
    activateModelProfile: vi.fn().mockResolvedValue({}),
    deleteModelProfile: vi.fn().mockResolvedValue({}),
    ...overrides
  };
  const showError = vi.fn();
  const refreshConfig = vi.fn();
  return { api, showError, refreshConfig, profiles: useModelProfiles({ api, showError, refreshConfig }) };
}

describe("useModelProfiles", () => {
  it("keeps a revealed key out of the listing and drops it after a while", async () => {
    vi.useFakeTimers();
    const { api, profiles } = harness();

    await profiles.load();
    // 列表里从来没有 key：设置页每次打开都会拉一次，让它搭这趟车等于送进截图。
    expect(JSON.stringify(profiles.profiles)).not.toContain("sk-live");

    await profiles.reveal("mp_1");
    expect(profiles.revealed.mp_1).toBe("sk-live");
    expect(api.revealModelProfileSecret).toHaveBeenCalledWith("mp_1");

    vi.advanceTimersByTime(REVEAL_TIMEOUT_MS + 100);
    expect(profiles.revealed.mp_1).toBeUndefined();
  });

  it("hides everything on request, for leaving the page", async () => {
    const { profiles } = harness();
    await profiles.reveal("mp_1");
    profiles.hideAll();
    expect(Object.keys(profiles.revealed)).toEqual([]);
  });

  it("refreshes the effective configuration after switching", async () => {
    const { profiles, refreshConfig } = harness();
    await profiles.activate("mp_1");
    await flushPromises();
    // 切换会改写全局设置；界面上那份视图必须跟着更新，否则显示的是上一份。
    expect(refreshConfig).toHaveBeenCalled();
  });

  it("asks before deleting and forgets a revealed key with it", async () => {
    const confirmed = vi.spyOn(globalThis, "confirm").mockReturnValue(false);
    const { profiles, api } = harness();
    await profiles.reveal("mp_1");

    profiles.remove("mp_1");
    expect(confirmed).toHaveBeenCalled();
    expect(api.deleteModelProfile).not.toHaveBeenCalled();

    confirmed.mockReturnValue(true);
    await profiles.remove("mp_1");
    expect(api.deleteModelProfile).toHaveBeenCalledWith("mp_1");
    expect(profiles.revealed.mp_1).toBeUndefined();
  });

  it("reports a failure instead of leaving a half-applied view", async () => {
    const { profiles, showError } = harness({
      activateModelProfile: vi.fn().mockRejectedValue(new Error("boom"))
    });
    await profiles.activate("mp_1");
    expect(showError).toHaveBeenCalled();
    expect(profiles.busy.value).toBe(false);
  });
});
